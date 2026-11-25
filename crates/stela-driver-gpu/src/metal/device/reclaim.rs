//! A device whose backend runs out of memory.
//!
//! The device keeps memory only to save the driver's kernel calls: destroyed
//! private textures for the next equal descriptor, free upload chunks for
//! the next upload, and released objects a bounded retirement left for the
//! next. A driver whose device memory is full spills to system memory, and
//! fails only when that is short too, so every allocation that fails for
//! want of memory frees these and is made once more before the call fails.
//! Each failure is counted, and the first and each power of two after it
//! leave a note for the platform's log: what failed, what was freed and
//! what the device still holds.
//!
//! The same notes carry the driver's own count of its heaps, and one is
//! left at the device's first call and at each step its holdings rise
//! ([`Device::note_memory_step`]), so a log shows how full the device memory
//! was before anything failed.
//!
//! A test build may hold the device to a count of bytes
//! ([`VRAM_CAP`](crate::metal::vulkan::physical::VRAM_CAP)), at which its
//! allocations run out of memory as a full driver's would: a GPU with much
//! memory then takes the paths one with little takes.
//!
//! An allocation whose host kernel call the platform withdrew, no task
//! having taken it in time, fails the same way and is not one of these: the
//! service makes its call again where it may, and the device makes no
//! second allocation in its place.
use super::Device;
use crate::metal::{
    compiler::CompileLock,
    error::{Error, Result},
    shader::Compiler,
    vulkan::{
        backend::{Backend, BufferAllocation},
        object::{BufferInfo, Image, ImageInfo, Memory, MemoryRequest},
    },
};
use alloc::{collections::VecDeque, format, string::String, sync::Arc};

/// Notes the device keeps until a call takes them.
const NOTES: usize = 8;

/// What a device out of memory gave up to try again, in bytes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Reclaimed {
    /// Destroyed private textures kept for reuse.
    pub recycled: u64,
    /// Free upload chunks.
    pub chunks: u64,
    /// Released objects a bounded retirement had left undestroyed.
    pub released: u64,
}

impl Reclaimed {
    fn any(&self) -> bool {
        self.recycled != 0 || self.chunks != 0 || self.released != 0
    }
}

/// The least a device's holdings rise between two notes of them.
const STEP_BYTES: u64 = 64 << 20;

/// The device's out-of-memory count, the holdings its next note of them
/// waits for, and the notes it left.
#[derive(Debug, Default)]
pub(super) struct Shortage {
    count: u64,
    next_step: u64,
    notes: VecDeque<String>,
}

impl<B: Backend, C: Compiler> Device<B, C> {
    /// [`Backend::create_image`], made once more after the device frees
    /// what it keeps for reuse when memory runs out.
    pub(super) fn make_image(
        &mut self,
        info: &ImageInfo,
        request: MemoryRequest,
    ) -> Result<(Image, Memory, u64)> {
        self.reclaiming(
            |backend| backend.create_image(info, request),
            || {
                let [width, height, depth] = info.extent;
                format!(
                    "a {width}x{height}x{depth} {:?} image of {} levels, {} layers and {:?} samples",
                    info.format, info.mip_levels, info.array_layers, info.samples
                )
            },
        )
    }

    /// [`Backend::create_buffer`], as [`Self::make_image`].
    pub(super) fn make_buffer(
        &mut self,
        info: &BufferInfo,
        request: MemoryRequest,
    ) -> Result<BufferAllocation> {
        self.reclaiming(
            |backend| backend.create_buffer(info, request),
            || {
                let place = if request.mapped { "mapped" } else { "device" };
                format!("a {place} buffer of {} bytes", info.size)
            },
        )
    }

    /// [`Backend::allocate_memory`], as [`Self::make_image`].
    pub(super) fn make_memory(
        &mut self,
        bytes: u64,
        request: MemoryRequest,
        memory_types: u32,
    ) -> Result<Memory> {
        self.reclaiming(
            |backend| backend.allocate_memory(bytes, request, memory_types),
            || format!("heap memory of {bytes} bytes"),
        )
    }

    fn reclaiming<T>(
        &mut self,
        mut make: impl FnMut(&mut B) -> Result<T>,
        what: impl FnOnce() -> String,
    ) -> Result<T> {
        let first = if self.over_cap() {
            Err(Error::OutOfMemory)
        } else {
            make(&mut self.backend)
        };
        if !matches!(first, Err(Error::OutOfMemory)) || self.host_call_withdrawn() {
            return first;
        }
        let reclaimed = self.reclaim()?;
        let made = if reclaimed.any() && !self.over_cap() {
            make(&mut self.backend)
        } else {
            first
        };
        self.note_shortage(what, reclaimed, made.is_ok());
        made
    }

    /// Whether a test build's cap leaves the device no memory: it holds as
    /// much as the cap in resources, recycled textures and upload chunks.
    fn over_cap(&self) -> bool {
        self.memory_cap.is_some_and(|cap| {
            self.allocated + self.recycled().bytes + self.upload_chunks().bytes >= cap
        })
    }

    /// Holds the device to `bytes`, as a test build's cap does.
    #[cfg(test)]
    pub(super) fn cap_memory(&mut self, bytes: u64) {
        self.memory_cap = Some(bytes);
    }

    /// Gives the device the service's lock, through which the platform
    /// says that it withdrew a host call of the call served now.
    pub fn watch_host_calls(&mut self, lock: &Arc<CompileLock>) {
        if self.host_calls.is_none() {
            self.host_calls = Some(lock.clone());
        }
    }

    /// Whether the platform withdrew a host call of the call served now,
    /// no task having taken it in time. The driver fails such an allocation
    /// as out of memory, which it is not: nothing is freed for it, and no
    /// second allocation is made, whose host call would hold the caller's
    /// processor as long again. The service makes the call again when it
    /// may.
    fn host_call_withdrawn(&self) -> bool {
        self.host_calls
            .as_ref()
            .is_some_and(|lock| lock.withdrawn())
    }

    /// Frees what the device keeps only for reuse. What submitted work may
    /// still read is freed when that work completes, as ever.
    fn reclaim(&mut self) -> Result<Reclaimed> {
        let recycled = self.recycled().bytes;
        self.trim_recycled(0, 0)?;
        let chunks = self.drop_free_chunks();
        let released = self.destroy_released();
        Ok(Reclaimed {
            recycled,
            chunks,
            released,
        })
    }

    /// Destroys the released objects nothing in flight may read any longer,
    /// which a bounded retirement left; answers their bytes.
    fn destroy_released(&mut self) -> u64 {
        if self.lost || !self.destruction_left {
            return 0;
        }
        let mut bytes = 0;
        let mut kept = VecDeque::with_capacity(self.deferred.len());
        for (release, object, allocated) in core::mem::take(&mut self.deferred) {
            if self.frees_object(&release, object) {
                self.destroy_now(object);
                self.allocated -= allocated;
                bytes += allocated;
            } else {
                kept.push_back((release, object, allocated));
            }
        }
        self.deferred = kept;
        self.destruction_left = false;
        bytes
    }

    fn note_shortage(&mut self, what: impl FnOnce() -> String, reclaimed: Reclaimed, made: bool) {
        self.shortage.count += 1;
        let count = self.shortage.count;
        if !count.is_power_of_two() || self.shortage.notes.len() >= NOTES {
            return;
        }
        let mib = |bytes: u64| bytes >> 20;
        let outcome = match (reclaimed.any(), made) {
            (false, _) => String::from("it kept nothing to free"),
            (true, made) => format!(
                "it freed recycled={}M chunks={}M released={}M and tried again, which {}",
                mib(reclaimed.recycled),
                mib(reclaimed.chunks),
                mib(reclaimed.released),
                if made { "succeeded" } else { "failed" }
            ),
        };
        let held = self.memory();
        let driver = self.driver_heaps();
        self.shortage.notes.push_back(format!(
            "OUT OF MEMORY {count}: {} failed; {outcome}; the device holds allocated={}M \
             deferred={}M behind {} command buffers recycled={}M chunks={}M; {driver}",
            what(),
            mib(held.allocated),
            mib(held.deferred_bytes),
            held.in_flight,
            mib(held.recycled.bytes),
            mib(held.chunks.bytes),
        ));
    }

    /// The driver's heaps as it counts them now, for a note. Not asked
    /// once the platform withdrew a host call of the call served now: the
    /// driver's count is a host call too, which would hold the caller's
    /// processor as long again.
    fn driver_heaps(&mut self) -> String {
        if self.host_call_withdrawn() {
            return String::from("the driver's heaps were not asked, as a host call waited");
        }
        let heaps = self.backend.memory_heaps();
        if heaps.is_empty() {
            return String::from("the driver does not say what its heaps hold");
        }
        let mut text = String::from("driver heaps use/budget/size:");
        for heap in heaps {
            let kind = if heap.device_local { "local" } else { "host" };
            text.push_str(&format!(
                " {}/{}/{}M/{kind}",
                heap.usage >> 20,
                heap.budget >> 20,
                heap.size >> 20
            ));
        }
        text
    }

    /// Notes what the device holds when the call served now is its first,
    /// and each time its holdings pass the next sixteenth of the memory the
    /// guest was told to work in (at least [`STEP_BYTES`]) above the last
    /// note's: a log then shows the device's memory as it filled, with the
    /// driver's own count of its heaps at each step.
    pub fn note_memory_step(&mut self) {
        let held = self.allocated + self.recycled().bytes;
        if held < self.shortage.next_step || self.shortage.notes.len() >= NOTES {
            return;
        }
        let budget = self.capabilities.recommended_max_working_set_size;
        let step = (budget / 16).max(STEP_BYTES);
        self.shortage.next_step = (held / step + 1) * step;
        let mib = |bytes: u64| bytes >> 20;
        let memory = self.memory();
        let driver = self.driver_heaps();
        // A capped test build says so in each note, so its log is never
        // read as a small GPU's.
        let cap = self.memory_cap.map_or_else(String::new, |cap| {
            format!("; a test build caps the device at {}M", mib(cap))
        });
        self.shortage.notes.push_back(format!(
            "VRAM holds {}M of the {}M the guest works in: allocated={}M recycled={}M \
             chunks={}M blocks={}/{}M out_of_memory={}; {driver}{cap}",
            mib(held + memory.chunks.bytes),
            mib(budget),
            mib(memory.allocated),
            mib(memory.recycled.bytes),
            mib(memory.chunks.bytes),
            memory.blocks.blocks,
            mib(memory.blocks.block_bytes),
            memory.out_of_memory,
        ));
    }

    /// How often an allocation of the backend's ran out of memory.
    pub fn out_of_memory(&self) -> u64 {
        self.shortage.count
    }

    /// The oldest note of the device's memory that no call has reported.
    pub fn take_memory_note(&mut self) -> Option<String> {
        self.shortage.notes.pop_front()
    }
}
