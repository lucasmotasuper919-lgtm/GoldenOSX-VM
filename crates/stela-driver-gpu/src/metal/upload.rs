//! Upload memory of one command buffer: argument tables, `set*Bytes` data
//! and small transfer sources, placed in host-visible chunks the GPU reads
//! by device address.
//!
//! Encoding needs each upload's address immediately, so space is
//! suballocated from chunks as commands are encoded and the bytes are kept
//! per chunk until commit, when the device copies them into the chunks'
//! mapped memory. Chunks return to the device's pool once the command
//! buffer completes.
use super::{
    error::{Error, Result},
    vulkan::object::{Buffer, Memory},
};
use alloc::vec::Vec;

/// Chunks are at least this large; bigger uploads get a chunk of their own.
pub const CHUNK_BYTES: u64 = 64 << 10;

/// A host-visible buffer with a device address, from the device's pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Chunk {
    pub buffer: Buffer,
    pub address: u64,
    pub size: u64,
}

/// Unused free chunks the pool keeps whatever their age: the stock of
/// default chunks every command buffer takes from.
const KEPT_FREE_BYTES: u64 = 4 << 20;
/// How many command buffers may retire while a free chunk goes unused
/// before the pool gives it up.
const IDLE_RETIREMENTS: u64 = 64;

/// What a device's chunk pool holds.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ChunkReport {
    pub chunks: usize,
    pub bytes: u64,
    pub free: usize,
    pub free_bytes: u64,
}

/// A device's upload chunks, each with its memory. A chunk no command
/// buffer uses is free: the next upload of about its size takes it, and one
/// that stays unused while [`IDLE_RETIREMENTS`] command buffers retire is
/// given up, so a pass's large scratch memory (imageblock data of a whole
/// attachment) is held only while passes of that size keep coming.
#[derive(Debug, Default)]
pub struct ChunkPool {
    chunks: Vec<(Chunk, Memory)>,
    /// Free chunks with the retirement count each was returned at and the
    /// newest submitted serial then.
    free: Vec<(Chunk, u64, u64)>,
    retirements: u64,
}

/// A chunk the pool gave up, with the newest serial submitted when it was
/// returned: only command buffers through that serial may still reach it.
pub type GivenUp = (Chunk, Memory, u64);

impl ChunkPool {
    /// Adds a new chunk, in use.
    pub fn insert(&mut self, chunk: Chunk, memory: Memory) {
        self.chunks.push((chunk, memory));
    }

    /// The memory of the chunk whose buffer is `buffer`.
    pub fn memory(&self, buffer: Buffer) -> Option<Memory> {
        self.chunks
            .iter()
            .find(|(chunk, _)| chunk.buffer == buffer)
            .map(|(_, memory)| *memory)
    }

    /// The smallest free chunk of `bytes` to twice that: a small upload
    /// never takes a large chunk away from the uploads it was made for.
    pub fn take(&mut self, bytes: u64) -> Option<Chunk> {
        let most = bytes.saturating_mul(2);
        let index = self
            .free
            .iter()
            .enumerate()
            .filter(|(_, (chunk, ..))| (bytes..=most).contains(&chunk.size))
            .min_by_key(|(_, (chunk, ..))| chunk.size)
            .map(|(index, _)| index)?;
        Some(self.free.swap_remove(index).0)
    }

    /// Returns chunks no command buffer uses any longer, `serial` being the
    /// newest submitted.
    pub fn give(&mut self, chunks: impl IntoIterator<Item = Chunk>, serial: u64) {
        let now = self.retirements;
        self.free
            .extend(chunks.into_iter().map(|chunk| (chunk, now, serial)));
    }

    /// Counts `retired` command buffers and gives up the free chunks that
    /// went unused through [`IDLE_RETIREMENTS`], the largest first, down to
    /// [`KEPT_FREE_BYTES`] of them. The caller destroys what it answers.
    pub fn retire(&mut self, retired: u64) -> Vec<GivenUp> {
        self.retirements += retired;
        let now = self.retirements;
        let mut idle: Vec<(u64, Buffer)> = self
            .free
            .iter()
            .filter(|&&(_, returned, _)| now - returned >= IDLE_RETIREMENTS)
            .map(|&(chunk, ..)| (chunk.size, chunk.buffer))
            .collect();
        let mut idle_bytes: u64 = idle.iter().map(|&(size, _)| size).sum();
        if idle_bytes <= KEPT_FREE_BYTES {
            return Vec::new();
        }
        idle.sort_unstable_by_key(|&(size, _)| core::cmp::Reverse(size));
        let mut given = Vec::new();
        for (size, buffer) in idle {
            if idle_bytes <= KEPT_FREE_BYTES {
                break;
            }
            idle_bytes -= size;
            given.push(buffer);
        }
        self.remove(&given)
    }

    /// Gives up every free chunk, as a device out of memory does.
    pub fn drain_free(&mut self) -> Vec<GivenUp> {
        let all: Vec<Buffer> = self.free.iter().map(|(chunk, ..)| chunk.buffer).collect();
        self.remove(&all)
    }

    fn remove(&mut self, buffers: &[Buffer]) -> Vec<GivenUp> {
        if buffers.is_empty() {
            return Vec::new();
        }
        let mut serials = Vec::with_capacity(buffers.len());
        self.free.retain(|&(chunk, _, serial)| {
            let kept = !buffers.contains(&chunk.buffer);
            if !kept {
                serials.push((chunk.buffer, serial));
            }
            kept
        });
        let (given, kept): (Vec<_>, Vec<_>) = core::mem::take(&mut self.chunks)
            .into_iter()
            .partition(|(chunk, _)| buffers.contains(&chunk.buffer));
        self.chunks = kept;
        given
            .into_iter()
            .map(|(chunk, memory)| {
                let serial = serials
                    .iter()
                    .find(|&&(buffer, _)| buffer == chunk.buffer)
                    .map_or(0, |&(_, serial)| serial);
                (chunk, memory, serial)
            })
            .collect()
    }

    pub fn report(&self) -> ChunkReport {
        ChunkReport {
            chunks: self.chunks.len(),
            bytes: self.chunks.iter().map(|(chunk, _)| chunk.size).sum(),
            free: self.free.len(),
            free_bytes: self.free.iter().map(|(chunk, ..)| chunk.size).sum(),
        }
    }
}

/// The device's chunk pool.
pub trait ChunkSource {
    /// A free chunk of at least `bytes`.
    fn acquire(&mut self, bytes: u64) -> Result<Chunk>;
}

/// Where an upload landed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Upload {
    pub buffer: Buffer,
    pub offset: u64,
    pub address: u64,
}

#[derive(Debug, Default)]
pub struct UploadArena {
    /// Chunks in use, each with the bytes written so far.
    chunks: Vec<(Chunk, Vec<u8>)>,
    /// Chunks of scratch memory, which no upload shares.
    scratch: Vec<Chunk>,
}

impl UploadArena {
    pub fn write(
        &mut self,
        source: &mut dyn ChunkSource,
        bytes: &[u8],
        alignment: u64,
    ) -> Result<Upload> {
        let (upload, data) = self.reserve(source, bytes.len(), alignment)?;
        data.extend_from_slice(bytes);
        Ok(upload)
    }

    /// Device memory of its own chunk that commit writes nothing into: its
    /// contents are undefined.
    pub(super) fn scratch(&mut self, source: &mut dyn ChunkSource, length: u64) -> Result<Upload> {
        let chunk = source.acquire(length.max(CHUNK_BYTES))?;
        if chunk.size < length {
            return Err(Error::OutOfMemory);
        }
        self.scratch.push(chunk);
        Ok(Upload {
            buffer: chunk.buffer,
            offset: 0,
            address: chunk.address,
        })
    }

    /// Appends storage for a payload encoded directly into the arena.
    pub(super) fn zeroed(
        &mut self,
        source: &mut dyn ChunkSource,
        length: usize,
        alignment: u64,
    ) -> Result<(Upload, &mut [u8])> {
        let (upload, data) = self.reserve(source, length, alignment)?;
        let start = data.len();
        data.resize(start + length, 0);
        Ok((upload, &mut data[start..]))
    }

    fn reserve(
        &mut self,
        source: &mut dyn ChunkSource,
        length: usize,
        alignment: u64,
    ) -> Result<(Upload, &mut Vec<u8>)> {
        debug_assert!(alignment.is_power_of_two());
        let length = length as u64;
        let fits = |(chunk, data): &(Chunk, Vec<u8>)| {
            let offset = (data.len() as u64).next_multiple_of(alignment);
            (offset + length <= chunk.size).then_some(offset)
        };
        let offset = match self.chunks.last().and_then(fits) {
            Some(offset) => offset,
            None => {
                let chunk = source.acquire(length.max(CHUNK_BYTES))?;
                if chunk.size < length || !chunk.address.is_multiple_of(alignment) {
                    return Err(Error::OutOfMemory);
                }
                // Room for a whole default chunk (or this upload) up front,
                // so filling it never reallocates and recopies the bytes.
                let capacity = chunk.size.min(length.max(CHUNK_BYTES)) as usize;
                self.chunks.push((chunk, Vec::with_capacity(capacity)));
                0
            }
        };
        let (chunk, data) = self.chunks.last_mut().expect("a chunk was just ensured");
        data.resize(offset as usize, 0);
        Ok((
            Upload {
                buffer: chunk.buffer,
                offset,
                address: chunk.address + offset,
            },
            data,
        ))
    }

    /// The chunks and the bytes to copy into each before submission; a
    /// scratch chunk has none.
    pub fn finish(self) -> Vec<(Chunk, Vec<u8>)> {
        let mut chunks = self.chunks;
        chunks.extend(self.scratch.into_iter().map(|chunk| (chunk, Vec::new())));
        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Pool {
        next: u64,
    }

    impl ChunkSource for Pool {
        fn acquire(&mut self, bytes: u64) -> Result<Chunk> {
            self.next += 1;
            Ok(Chunk {
                buffer: Buffer(self.next),
                address: self.next << 32,
                size: bytes,
            })
        }
    }

    #[test]
    fn uploads_suballocate_aligned_and_spill_to_new_chunks() {
        let mut pool = Pool::default();
        let mut arena = UploadArena::default();
        let first = arena.write(&mut pool, &[1; 10], 16).unwrap();
        let (second, payload) = arena.zeroed(&mut pool, 4, 16).unwrap();
        assert_eq!(payload, &[0; 4]);
        payload.fill(2);
        assert_eq!((first.offset, second.offset), (0, 16));
        assert_eq!(second.address, (1 << 32) + 16);
        let big = arena
            .write(&mut pool, &alloc::vec![3; CHUNK_BYTES as usize + 1], 16)
            .unwrap();
        assert_eq!((big.buffer, big.offset), (Buffer(2), 0));
        let after = arena.write(&mut pool, &[4], 4).unwrap();
        assert_eq!(after.buffer, Buffer(3), "a full chunk is not reused");
        let chunks = arena.finish();
        assert_eq!(chunks.len(), 3);
        assert_eq!(
            &chunks[0].1[..],
            &[[1; 10].as_slice(), &[0; 6], &[2; 4]].concat()[..]
        );
    }

    fn chunk(id: u64, size: u64) -> Chunk {
        Chunk {
            buffer: Buffer(id),
            address: id << 32,
            size,
        }
    }

    /// A small upload takes the smallest chunk that holds it, and none of
    /// more than twice its size: a pass's scratch chunk stays for the next
    /// pass of its size, where the first free chunk in the list was taken
    /// whatever its size.
    #[test]
    fn an_upload_takes_the_smallest_free_chunk_of_about_its_size() {
        let mut pool = ChunkPool::default();
        let sizes = [64 << 20, CHUNK_BYTES, 3 * CHUNK_BYTES, 2 * CHUNK_BYTES];
        for (id, size) in (1..).zip(sizes) {
            pool.insert(chunk(id, size), Memory(id));
        }
        pool.give((1..).zip(sizes).map(|(id, size)| chunk(id, size)), 0);
        assert_eq!(pool.take(CHUNK_BYTES).unwrap().buffer, Buffer(2));
        assert_eq!(pool.take(CHUNK_BYTES).unwrap().buffer, Buffer(4));
        assert_eq!(
            pool.take(CHUNK_BYTES),
            None,
            "three and a thousand times the size are left"
        );
        assert_eq!(pool.take(40 << 20).unwrap().buffer, Buffer(1));
        assert_eq!(pool.memory(Buffer(3)), Some(Memory(3)));
        let report = pool.report();
        assert_eq!((report.chunks, report.free), (4, 1));
        assert_eq!(report.free_bytes, 3 * CHUNK_BYTES);
    }

    /// A large chunk nothing takes while command buffers retire is given
    /// up; one taken again in time, and the stock of default chunks, stay.
    #[test]
    fn idle_chunks_are_given_up_and_used_ones_kept() {
        let mut pool = ChunkPool::default();
        let (scratch, burst) = (chunk(1, 64 << 20), chunk(2, 32 << 20));
        pool.insert(scratch, Memory(1));
        pool.insert(burst, Memory(2));
        for id in 3..=10 {
            pool.insert(chunk(id, CHUNK_BYTES), Memory(id));
        }
        pool.give([scratch, burst], 1);
        pool.give((3..=10).map(|id| chunk(id, CHUNK_BYTES)), 1);
        for _ in 0..IDLE_RETIREMENTS - 1 {
            // Each frame's pass takes the scratch chunk and returns it.
            let taken = pool.take(64 << 20).unwrap();
            assert_eq!(pool.retire(1), Vec::new());
            pool.give([taken], 2);
        }
        assert_eq!(pool.retire(1), [(burst, Memory(2), 1)]);
        assert_eq!(pool.memory(Buffer(2)), None);
        // The default chunks are as old, and under the kept stock.
        // Each answers the serial it was returned at.
        assert_eq!(pool.retire(IDLE_RETIREMENTS), [(scratch, Memory(1), 2)]);
        assert_eq!(pool.retire(IDLE_RETIREMENTS), Vec::new());
        let report = pool.report();
        assert_eq!((report.chunks, report.free), (8, 8));
        assert_eq!(report.bytes, 8 * CHUNK_BYTES);
        // Out of memory, the device gives up the stock too.
        assert_eq!(pool.drain_free().len(), 8);
        assert_eq!(pool.report(), ChunkReport::default());
    }

    /// Scratch memory has a chunk of its own that no later upload shares,
    /// even when it is the arena's first.
    #[test]
    fn scratch_memory_is_never_shared() {
        let mut pool = Pool::default();
        let mut arena = UploadArena::default();
        let scratch = arena.scratch(&mut pool, 64).unwrap();
        let table = arena.write(&mut pool, &[5; 8], 16).unwrap();
        assert_ne!(scratch.buffer, table.buffer);
        let more = arena.scratch(&mut pool, 16).unwrap();
        let next = arena.write(&mut pool, &[6; 8], 16).unwrap();
        assert_ne!(more.buffer, next.buffer);
        assert_eq!(next.buffer, table.buffer);
        let chunks = arena.finish();
        assert_eq!(chunks.len(), 3);
        assert!(
            chunks
                .iter()
                .filter(|(chunk, _)| chunk.buffer == scratch.buffer || chunk.buffer == more.buffer)
                .all(|(_, data)| data.is_empty())
        );
    }
}
