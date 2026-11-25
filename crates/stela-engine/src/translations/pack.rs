//! The pack container `stela.efi` keeps its translations in, and the store a
//! boot reads from it and plans the next pack with.
//!
//! A pack is a header, an index sorted by page key with the SHA-256 of the
//! header and index, and the page records (each checked by its own trailer
//! when it is first used). Its length follows from the index.
//!
//! A translator's file is a journal (`stela_formats::journal`) of packs, its
//! segments, and of used-key lists ([`UsedKeys`]), one per boot. A boot
//! reads every segment ([`Store::over`]): a key's record is what all of them
//! hold for it, merged. While the guest runs, the entries the boot added go
//! to the file as a new segment ([`Delta`]); every few boots the segments
//! and used-key lists are merged into one segment ([`Merge`]), the records
//! no boot used for [`MAX_UNUSED_BOOTS`] boots in a row left out. Each is
//! built a step at a time, so no step holds a processor for long.
//!
//! The store hands the engine the pack's records and keeps the stored form
//! of each block the engine compiled anew, which its compiler made as it
//! emitted the block ([`Offer::entry`]). A store shared by machines that
//! run at once (one per PE) also keeps those blocks' native code, within a
//! bound, for the other machines to install instead of compiling them
//! again.
mod codec;

use super::{Digest, Entry, FormatError, Record, StoredRecord};
use crate::machine::engine::SharedTranslation;
use crate::machine::{Looked, Offer, Translations};
use crate::platform::sync::Mutex;
use alloc::{
    borrow::Cow,
    collections::{BTreeMap, VecDeque},
    sync::{Arc, Weak},
    vec::Vec,
};
pub use codec::{DecodeError, DecodeScratch};
use core::mem::size_of;
use core::sync::atomic::{AtomicU32, Ordering};
use stela_crypto::Sha256;
use stela_crypto::digest::Digest as _;
use stela_formats::journal;

pub const MAGIC: [u8; 8] = *b"STLTPACK";
pub const SCHEMA: u32 = 1;
const MIXED_SCHEMA: u32 = 2;
/// The schemas of packs whose index counts each record's uses
/// ([`USES_SHIFT`]): [`SCHEMA`] and [`MIXED_SCHEMA`] this much higher. A
/// pack of the first two loads, its records counted as unused.
const COUNTED: u32 = 2;
const COUNTED_SCHEMA: u32 = SCHEMA + COUNTED;
const COUNTED_MIXED_SCHEMA: u32 = MIXED_SCHEMA + COUNTED;
/// Magic, schema, record count and translator id.
pub const HEADER_BYTES: usize = MAGIC.len() + 4 + 4 + DIGEST_BYTES;
const DIGEST_BYTES: usize = 32;
/// Page key, offset and length. The length field's top byte holds the
/// record's age ([`MAX_UNUSED_BOOTS`]), the byte below it, in a counted
/// pack, the most uses a boot made of it ([`Merge::by_use`], saturating),
/// the rest its length.
const INDEX_ENTRY_BYTES: usize = DIGEST_BYTES + 8 + 8;
/// What a record whose bytes stay in the file costs a boot's memory: its
/// index entry, its place in the layout order ([`LayoutOrder`]) and its
/// key filter bits ([`KeyFilters`]).
pub const RESIDENT_RECORD_BYTES: u64 =
    (INDEX_ENTRY_BYTES + size_of::<u32>() + FILTER_BITS_PER_KEY / 8) as u64;
const AGE_SHIFT: u32 = 56;
const USES_SHIFT: u32 = 48;
pub const LENGTH_MASK: u64 = (1 << USES_SHIFT) - 1;
/// A record no boot looked up for more boots in a row than this leaves the
/// next pack: the pack keeps what boots use, and what they stopped using
/// costs no read time for long. Every boot reads the whole pack.
pub const MAX_UNUSED_BOOTS: u8 = 3;
const CHECKSUM_BYTES: usize = DIGEST_BYTES;
/// The most records a pack holds.
pub const MAX_RECORDS: usize = 1 << 22;
/// The default bound of the decoded-record cache, least recently used first
/// out ([`Store::set_bounds`]). One record takes at most an eighth of it.
const DECODED_BYTES: usize = 8 << 20;
/// The default bound of the entries added during the run ([`Store::keep`]).
const ADDED_BYTES: usize = 64 << 20;
/// The direct-mapped tables that name why a shared lookup missed: offers
/// being compiled, offers evicted, and the last offer of each PC and page.
const MISS_SLOTS: usize = 4096;
/// The journal domain of a translator's pack file, its key the translator
/// id, and the kinds of its blobs: segments and used-key lists.
pub const JOURNAL_DOMAIN: [u8; 8] = *b"TRANSLAT";
pub const SEGMENT_BLOB: u32 = 1;
pub const USED_BLOB: u32 = 2;
/// The most segments a store reads: a key's lookup visits each.
pub const MAX_SEGMENTS: usize = 8;
/// The most used-key lists a file holds: one per boot since the last merge.
pub const MAX_USED_LISTS: usize = 8;
/// The most page keys a used-key list names: 8-byte key prefixes, so a list
/// takes at most 1 MiB. A boot that used more names none and counts as one
/// that used every record ([`UsedKeys::every`]).
pub const MAX_USED_KEYS: usize = 1 << 17;
const USED_MAGIC: [u8; 8] = *b"STLTUSED";
/// Magic, translator id, a flags word and the key count.
const USED_HEADER_BYTES: usize = 8 + DIGEST_BYTES + 4 + 4;
/// The list names no keys: its boot used more than [`MAX_USED_KEYS`].
const USED_EVERY: u32 = 1;
/// The index bytes a merge step hashes at a time.
const INDEX_HASH_BYTES: usize = 64 << 10;
/// The most bytes one read of the pack file asks for: records that lie
/// one after another are read together up to it.
pub const READ_BYTES: u64 = 1 << 20;
/// The most bytes a merge keeps asked for or read and not used yet.
const READS_HELD_BYTES: u64 = 8 << 20;

/// A boot's uses of a page key: how many, and the rank of the first among
/// every key's first use this boot ([`Store`] parts share one count), which
/// lays merged records out in the order a boot first needs them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Use {
    pub count: u32,
    pub first: u32,
}

/// The next first-use rank: every part of a store counts on it.
static FIRST_USES: AtomicU32 = AtomicU32::new(0);

impl Use {
    /// The first use of a key: ranked after every key used before it.
    pub fn first() -> Self {
        Self {
            count: 1,
            first: FIRST_USES.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Another part's uses of the same key.
    fn join(&mut self, other: Self) {
        self.count = self.count.saturating_add(other.count);
        self.first = self.first.min(other.first);
    }
}

/// Reads of the pack file that a merge waits for. The boot carries them
/// out on its device a bounded step at a time ([`Self::take_asked`],
/// [`Self::finish`]); a merge never waits in a step.
#[derive(Debug, Default)]
pub struct FileReads {
    next: u64,
    asked: VecDeque<(u64, u64, u64)>,
    /// Reads started and not finished.
    started: usize,
    /// Reads finished, by id: the bytes, or `None` for a failed read.
    done: BTreeMap<u64, Option<Vec<u8>>>,
    /// Reads finished and their bytes, in all.
    pub reads: u64,
    pub bytes: u64,
}

impl FileReads {
    /// Whether read `id` ended.
    pub fn ended(&self, id: u64) -> bool {
        self.done.contains_key(&id)
    }

    /// Read `id`'s outcome once it ended ([`Self::take`]).
    pub fn take_ended(&mut self, id: u64) -> Option<Option<Vec<u8>>> {
        self.take(id)
    }

    /// Ask for `bytes` bytes at file offset `at`: the read's id.
    pub fn ask(&mut self, at: u64, bytes: u64) -> u64 {
        self.next += 1;
        self.asked.push_back((self.next, at, bytes));
        self.next
    }

    /// The next read to start: its id, file offset and bytes.
    pub fn take_asked(&mut self) -> Option<(u64, u64, u64)> {
        let asked = self.asked.pop_front()?;
        self.started += 1;
        Some(asked)
    }

    /// Read `id` ended: its bytes, or `None` when it failed.
    pub fn finish(&mut self, id: u64, bytes: Option<Vec<u8>>) {
        self.started = self.started.saturating_sub(1);
        self.reads += 1;
        self.bytes += bytes.as_ref().map_or(0, |bytes| bytes.len() as u64);
        self.done.insert(id, bytes);
    }

    /// Read `id`'s outcome once it ended: `Some(None)` for a failed read.
    fn take(&mut self, id: u64) -> Option<Option<Vec<u8>>> {
        self.done.remove(&id)
    }

    /// Whether a read is asked for or under way.
    pub fn waiting(&self) -> bool {
        !self.asked.is_empty() || self.started != 0
    }

    /// Forget every read: those under way end unread.
    fn clear(&mut self) {
        self.asked.clear();
        self.done.clear();
    }
}

/// Whether `bytes`, a record of `format` read from the file, decodes for
/// `translator` and `key` with every check: its trailer's hash and each
/// entry.
fn verified(
    bytes: &[u8],
    format: PackFormat,
    translator: &Digest,
    key: &Digest,
    scratch: &mut DecodeScratch,
) -> bool {
    let record = PackedRecord {
        bytes,
        format,
        checked: false,
    };
    record.decode(translator, key, scratch).is_ok()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackError {
    Truncated,
    Magic,
    Schema,
    Translator,
    Count,
    Checksum,
    /// An index entry is out of order or names bytes outside the pack.
    Index,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
enum PackFormat {
    Raw = SCHEMA,
    Mixed = MIXED_SCHEMA,
}

impl PackFormat {
    fn read(header: &[u8]) -> Result<Self, PackError> {
        match u32::from_le_bytes(
            header[MAGIC.len()..][..4]
                .try_into()
                .expect("a schema word"),
        ) {
            SCHEMA | COUNTED_SCHEMA => Ok(Self::Raw),
            MIXED_SCHEMA | COUNTED_MIXED_SCHEMA => Ok(Self::Mixed),
            _ => Err(PackError::Schema),
        }
    }

    /// The schema a pack of this format is written with: counted.
    fn schema(self) -> u32 {
        self as u32 + COUNTED
    }
}

/// Whether the pack whose header starts `header` counts its records' uses.
fn counted(header: &[u8]) -> bool {
    u32::from_le_bytes(
        header[MAGIC.len()..][..4]
            .try_into()
            .expect("a schema word"),
    ) > MIXED_SCHEMA
}

/// The bytes of the header, index and index checksum of the pack whose
/// header is `header`: what to read before its records.
pub fn prefix_bytes(header: &[u8], translator: &Digest) -> Result<usize, PackError> {
    if header.len() < HEADER_BYTES {
        return Err(PackError::Truncated);
    }
    if header[..MAGIC.len()] != MAGIC {
        return Err(PackError::Magic);
    }
    let word = |at: usize| u32::from_le_bytes(header[at..at + 4].try_into().expect("a word"));
    PackFormat::read(header)?;
    let count = word(MAGIC.len() + 4) as usize;
    if count > MAX_RECORDS {
        return Err(PackError::Count);
    }
    if header[MAGIC.len() + 8..HEADER_BYTES] != translator.0 {
        return Err(PackError::Translator);
    }
    Ok(HEADER_BYTES + count * INDEX_ENTRY_BYTES + CHECKSUM_BYTES)
}

/// A pack's index over its bytes.
#[derive(Debug, Clone, Copy)]
pub struct Pack<'a> {
    bytes: &'a [u8],
    count: usize,
    length: usize,
    format: PackFormat,
    /// Its records are in `bytes`; otherwise only its header and index
    /// are, and a record is read from the file when a lookup asks for it
    /// ([`Store::take_wanted`]).
    resident: bool,
}

impl<'a> Pack<'a> {
    /// The length of the pack whose header, index and checksum start
    /// `prefix`, checked as [`Self::parse`] checks them.
    pub fn length(prefix: &[u8], translator: &Digest) -> Result<u64, PackError> {
        Self::index(prefix, translator).map(|(_, length)| length as u64)
    }

    /// The pack at the start of `bytes`: its header, index checksum, key
    /// order and every record's range are checked; bytes past its length
    /// are ignored.
    pub fn parse(bytes: &'a [u8], translator: &Digest) -> Result<Self, PackError> {
        let (count, length) = Self::index(bytes, translator)?;
        if bytes.len() < length {
            return Err(PackError::Truncated);
        }
        Ok(Self {
            bytes: &bytes[..length],
            count,
            length,
            format: PackFormat::read(bytes)?,
            resident: true,
        })
    }

    /// The pack whose header, index and checksum start `prefix`, checked
    /// as [`Self::parse`] checks them, its records left in the file: each
    /// is read when first wanted, and checked by its trailer as it is
    /// decoded.
    pub fn index_only(prefix: &'a [u8], translator: &Digest) -> Result<Self, PackError> {
        let (count, length) = Self::index(prefix, translator)?;
        let records_start = prefix_bytes(prefix, translator)?;
        Ok(Self {
            bytes: &prefix[..records_start],
            count,
            length,
            format: PackFormat::read(prefix)?,
            resident: false,
        })
    }

    /// Whether its records are in memory.
    pub fn resident(&self) -> bool {
        self.resident
    }

    /// The bytes held in memory: the whole pack, or its header and index.
    pub fn held_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// Where the record of `key` lies: its offset from the pack's start and
    /// its length.
    pub fn locate(&self, key: &Digest) -> Option<(u64, u64)> {
        let (_, offset, bytes) = entry(self.bytes, self.position(key)?);
        Some((offset, bytes))
    }

    /// The record count and the pack's length.
    fn index(prefix: &[u8], translator: &Digest) -> Result<(usize, usize), PackError> {
        let records_start = prefix_bytes(prefix, translator)?;
        if prefix.len() < records_start {
            return Err(PackError::Truncated);
        }
        let index_end = records_start - CHECKSUM_BYTES;
        if Sha256::digest(&prefix[..index_end]).as_slice() != &prefix[index_end..records_start] {
            return Err(PackError::Checksum);
        }
        let count = (index_end - HEADER_BYTES) / INDEX_ENTRY_BYTES;
        let mut length = records_start;
        let mut previous: Option<Digest> = None;
        for index in 0..count {
            let (key, offset, bytes) = entry(prefix, index);
            let end = offset.checked_add(bytes).ok_or(PackError::Index)?;
            if previous.is_some_and(|previous| previous >= key)
                || offset < records_start as u64
                || bytes == 0
            {
                return Err(PackError::Index);
            }
            length = length.max(usize::try_from(end).map_err(|_| PackError::Index)?);
            previous = Some(key);
        }
        Ok((count, length))
    }

    pub fn records(&self) -> usize {
        self.count
    }

    /// The pack's length in bytes.
    pub fn bytes(&self) -> usize {
        self.length
    }

    /// The record bytes of page key `key`, unchecked.
    pub fn get(&self, key: &Digest) -> Option<&'a [u8]> {
        if !self.resident {
            return None;
        }
        let (_, offset, bytes) = entry(self.bytes, self.position(key)?);
        Some(&self.bytes[offset as usize..(offset + bytes) as usize])
    }

    /// The boots in a row since one last used the record of `key`
    /// ([`MAX_UNUSED_BOOTS`]).
    pub fn age(&self, key: &Digest) -> Option<u8> {
        Some(entry_age(self.bytes, self.position(key)?))
    }

    /// The index entry of `key`.
    fn position(&self, key: &Digest) -> Option<usize> {
        let count = self.count;
        if count == 0 {
            return None;
        }
        let key_at = |index: usize| {
            let at = HEADER_BYTES + index * INDEX_ENTRY_BYTES;
            &self.bytes[at..at + DIGEST_BYTES]
        };
        // Page keys are SHA-256 digests, spread evenly: a key's index is
        // near its share of the key space, within a few standard deviations
        // (at most half the square root of the count). The search starts
        // in that window, and in the rest only when the key lies outside.
        let prefix = u64::from_be_bytes(key.0[..8].try_into().expect("a prefix"));
        let guess = ((u128::from(prefix) * count as u128) >> 64) as usize;
        let reach = 2 * count.isqrt() + 8;
        let (mut low, mut high) = (guess.saturating_sub(reach), (guess + reach + 1).min(count));
        if key.0[..] < *key_at(low) {
            (low, high) = (0, low);
        } else if high < count && key.0[..] >= *key_at(high) {
            (low, high) = (high, count);
        }
        while low < high {
            let middle = (low + high) / 2;
            match key_at(middle).cmp(&key.0[..]) {
                core::cmp::Ordering::Less => low = middle + 1,
                core::cmp::Ordering::Greater => high = middle,
                core::cmp::Ordering::Equal => return Some(middle),
            }
        }
        None
    }

    fn record(&self, key: &Digest) -> Option<PackedRecord<'a>> {
        if !self.resident {
            return None;
        }
        Some(self.record_at(self.position(key)?))
    }

    /// The record of index entry `index`, of a resident pack: a merge
    /// reads only those.
    fn record_at(&self, index: usize) -> PackedRecord<'a> {
        assert!(self.resident, "a record of a pack read whole");
        let (_, offset, bytes) = entry(self.bytes, index);
        PackedRecord {
            bytes: &self.bytes[offset as usize..(offset + bytes) as usize],
            format: self.format,
            checked: false,
        }
    }

    /// Every record's page key and the most uses a boot made of it (none
    /// in a pack of a schema before the count), in key order.
    pub fn uses(&self) -> impl Iterator<Item = (Digest, u8)> + '_ {
        (0..self.count).map(|index| (entry(self.bytes, index).0, entry_uses(self.bytes, index)))
    }

    /// Every record's page key and bytes, in key order, of a resident
    /// pack.
    pub fn iter(&self) -> impl Iterator<Item = (Digest, &'a [u8])> + '_ {
        assert!(self.resident, "the records of a pack read whole");
        (0..self.count).map(|index| {
            let (key, offset, bytes) = entry(self.bytes, index);
            (key, &self.bytes[offset as usize..(offset + bytes) as usize])
        })
    }
}

/// A filter of the page keys of each segment: a key a filter does not hold
/// is not in its segment, so a lookup searches only the segments that may
/// hold it, mostly the one that does, and seldom any for a key none holds.
/// Each key sets [`FILTER_BITS`] bits of one 64-bit word,
/// from about 16 bits a key: about one key in 40 a segment does not hold
/// passes its filter.
#[derive(Debug, Default)]
pub struct KeyFilters {
    filters: Vec<alloc::boxed::Box<[u64]>>,
}

const FILTER_BITS: usize = 4;
const FILTER_BITS_PER_KEY: usize = 16;

impl KeyFilters {
    /// The filters of `segments`, shared by every part of a store.
    pub fn new(segments: &[Pack<'_>]) -> Arc<Self> {
        let filters = segments
            .iter()
            .map(|pack| {
                let words = (pack.records() * FILTER_BITS_PER_KEY / 64)
                    .max(1)
                    .next_power_of_two();
                let mut filter = alloc::vec![0u64; words].into_boxed_slice();
                for index in 0..pack.records() {
                    let (key, _, _) = entry(pack.bytes, index);
                    let (word, bits) = Self::place(&key, words);
                    filter[word] |= bits;
                }
                filter
            })
            .collect();
        Arc::new(Self { filters })
    }

    /// The word of a filter of `words` words a key sets, and its bits: from
    /// key bytes no part split or search start depends on.
    fn place(key: &Digest, words: usize) -> (usize, u64) {
        let word = u64::from_le_bytes(key.0[8..16].try_into().expect("a word index"));
        let hash = u64::from_le_bytes(key.0[16..24].try_into().expect("bits"));
        let bits = (0..FILTER_BITS).fold(0u64, |bits, at| bits | 1 << ((hash >> (6 * at)) & 63));
        (word as usize & (words - 1), bits)
    }

    /// Whether segment `index` may hold `key`.
    fn may_hold(&self, index: usize, key: &Digest) -> bool {
        self.filters.get(index).is_none_or(|filter| {
            let (word, bits) = Self::place(key, filter.len());
            filter[word] & bits == bits
        })
    }
}

/// The records of each segment left in the file in the order they lie
/// there, which a merge made the order a boot first used them
/// ([`Merge`]): a lookup's record is followed by those the boot that laid
/// it out needed next (a launch's), and a disk whose reads seek is read
/// ahead in this order, a run of records at a time. Shared by every part of
/// a store; four bytes a record.
#[derive(Debug, Default)]
pub struct LayoutOrder {
    orders: Vec<Vec<u32>>,
}

impl LayoutOrder {
    pub fn new(segments: &[Pack<'_>]) -> Arc<Self> {
        let orders = segments
            .iter()
            .map(|pack| {
                if pack.resident {
                    return Vec::new();
                }
                let mut order: Vec<u32> = (0..pack.count as u32).collect();
                order.sort_unstable_by_key(|&index| entry(pack.bytes, index as usize).1);
                order
            })
            .collect();
        Arc::new(Self { orders })
    }

    /// The index entry `n`th in layout order of segment `segment`.
    pub fn nth(&self, segment: usize, n: usize) -> Option<u32> {
        self.orders.get(segment)?.get(n).copied()
    }

    /// The records laid out after `key`'s in each segment of `segments`
    /// (those [`Self::new`] ordered) that holds it, up to `count` in all:
    /// segment and index entry.
    pub fn after(&self, segments: &[Pack<'_>], key: &Digest, count: usize) -> Vec<(usize, u32)> {
        let mut after = Vec::new();
        for (segment, (pack, order)) in segments.iter().zip(&self.orders).enumerate() {
            if order.is_empty() {
                continue;
            }
            let Some(position) = pack.position(key) else {
                continue;
            };
            let offset = entry(pack.bytes, position).1;
            let from =
                order.partition_point(|&index| entry(pack.bytes, index as usize).1 <= offset);
            for &index in order[from..].iter().take(count - after.len()) {
                after.push((segment, index));
            }
            if after.len() == count {
                break;
            }
        }
        after
    }
}

impl Pack<'_> {
    /// The page key of index entry `index`.
    pub fn key_at(&self, index: usize) -> Digest {
        entry(self.bytes, index).0
    }
}

/// A key's record in each segment that holds it, oldest first: what a
/// lookup decodes and merges ([`DecodedRecord::decode`]).
#[derive(Debug, Clone, Copy)]
pub struct Stored<'a> {
    records: [Option<PackedRecord<'a>>; MAX_SEGMENTS],
}

impl<'a> Stored<'a> {
    fn iter(&self) -> impl Iterator<Item = PackedRecord<'a>> + '_ {
        self.records.iter().flatten().copied()
    }

    pub fn is_compressed(&self) -> bool {
        self.iter().any(PackedRecord::is_compressed)
    }

    fn same_storage(&self, other: &Self) -> bool {
        self.records
            .iter()
            .zip(&other.records)
            .all(|pair| match pair {
                (Some(one), Some(other)) => one.same_storage(*other),
                (None, None) => true,
                _ => false,
            })
    }

    /// Each segment's record indexed, earlier segments' entries first, as
    /// [`Record::merge`] merges them.
    fn stored_record(
        &self,
        translator: &Digest,
        key: &Digest,
        scratch: &mut DecodeScratch,
    ) -> Result<StoredRecord, DecodeError> {
        if self.iter().next().is_none() {
            return Err(DecodeError::Invalid);
        }
        let mut record = StoredRecord::new(*translator, *key);
        for stored in self.iter() {
            stored.add_to(&mut record, scratch)?;
        }
        Ok(record)
    }
}

/// The stored bytes and the verified index's record format.
#[derive(Debug, Clone, Copy)]
pub struct PackedRecord<'a> {
    bytes: &'a [u8],
    format: PackFormat,
    /// These bytes decoded whole once already, with every check: they are
    /// immutable, so a decode again checks only the format's bounds.
    checked: bool,
}

impl PackedRecord<'_> {
    pub fn is_compressed(self) -> bool {
        self.format == PackFormat::Mixed && self.bytes.starts_with(codec::MAGIC)
    }

    fn same_storage(self, other: Self) -> bool {
        self.format == other.format && core::ptr::eq(self.bytes, other.bytes)
    }

    /// Its entries, indexed into `record` ([`StoredRecord::add`]).
    fn add_to(
        self,
        record: &mut StoredRecord,
        scratch: &mut DecodeScratch,
    ) -> Result<(), DecodeError> {
        let raw = if self.is_compressed() {
            scratch.inflate(self.bytes)?
        } else {
            self.bytes
        };
        record.add(raw, !self.checked).map_err(|error| match error {
            FormatError::Capacity => DecodeError::Capacity,
            _ => DecodeError::Invalid,
        })
    }

    fn decode(
        self,
        translator: &Digest,
        key: &Digest,
        scratch: &mut DecodeScratch,
    ) -> Result<Record, DecodeError> {
        let raw = if self.is_compressed() {
            scratch.inflate(self.bytes)?
        } else {
            self.bytes
        };
        Record::decode_checked(raw, translator, key, !self.checked).map_err(|error| match error {
            FormatError::Capacity => DecodeError::Capacity,
            _ => DecodeError::Invalid,
        })
    }
}

/// Index entry `index` of the pack starting `bytes`: page key, offset and
/// length.
fn entry(bytes: &[u8], index: usize) -> (Digest, u64, u64) {
    let at = HEADER_BYTES + index * INDEX_ENTRY_BYTES;
    (
        Digest(bytes[at..at + DIGEST_BYTES].try_into().expect("a key")),
        index_field(bytes, at + DIGEST_BYTES),
        index_field(bytes, at + DIGEST_BYTES + 8) & LENGTH_MASK,
    )
}

/// Index entry `index`'s age: the top byte of its length field.
fn entry_age(bytes: &[u8], index: usize) -> u8 {
    let at = HEADER_BYTES + index * INDEX_ENTRY_BYTES + DIGEST_BYTES + 8;
    (index_field(bytes, at) >> AGE_SHIFT) as u8
}

/// Index entry `index`'s uses: the byte below its age in a counted pack,
/// none in an older one.
fn entry_uses(bytes: &[u8], index: usize) -> u8 {
    let at = HEADER_BYTES + index * INDEX_ENTRY_BYTES + DIGEST_BYTES + 8;
    if counted(bytes) {
        (index_field(bytes, at) >> USES_SHIFT) as u8
    } else {
        0
    }
}

fn index_field(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("a field"))
}

/// A segment a merge reads: one read from the file, or one laid out in
/// memory (a new segment, or an earlier merge's), whose records the merge
/// moves out instead of copying when nothing else holds it.
///
/// A segment whose records stay in the file ([`Segment::Filed`]) is merged
/// from its index: the records a merge keeps are read from the file
/// ([`FileReads`]) and written from there, never held whole in memory.
#[derive(Debug, Clone)]
pub enum Segment<'a> {
    Packed(Pack<'a>),
    Laid(Arc<Layout<'a>>),
    /// Its header, index and checksum, and where it starts in the file.
    Filed {
        index: Cow<'a, [u8]>,
        at: u64,
    },
}

impl<'a> From<Pack<'a>> for Segment<'a> {
    fn from(pack: Pack<'a>) -> Self {
        Self::Packed(pack)
    }
}

impl<'a> Segment<'a> {
    /// A segment at file offset `at`: held whole when its records are in
    /// memory, else by its index ([`Pack::index_only`]).
    pub fn at(pack: Pack<'a>, at: u64) -> Self {
        if pack.resident {
            Self::Packed(pack)
        } else {
            Self::Filed {
                index: Cow::Borrowed(pack.bytes),
                at,
            }
        }
    }

    /// Its index: the pack's bytes, or the layout's prefix.
    fn index(&self) -> &[u8] {
        match self {
            Self::Packed(pack) => pack.bytes,
            Self::Laid(layout) => &layout.prefix,
            Self::Filed { index, .. } => index,
        }
    }

    pub fn records(&self) -> usize {
        match self {
            Self::Packed(pack) => pack.count,
            Self::Laid(layout) => layout.records.len(),
            Self::Filed { index, .. } => (index.len() - prefix_length(0)) / INDEX_ENTRY_BYTES,
        }
    }

    /// Its length in the file.
    pub fn bytes(&self) -> u64 {
        match self {
            Self::Packed(pack) => pack.length as u64,
            Self::Laid(layout) => layout.length,
            Self::Filed { index, .. } => (0..self.records())
                .map(|at| {
                    let (_, offset, bytes) = entry(index, at);
                    offset + bytes
                })
                .max()
                .unwrap_or(index.len() as u64),
        }
    }

    /// Its records' page keys, in order.
    pub fn keys(&self) -> impl Iterator<Item = Digest> + '_ {
        (0..self.records()).map(|index| self.key(index))
    }

    /// Whether its records are in memory, for a merge to read: a pack
    /// read on demand leaves them in the file ([`Pack::index_only`]).
    pub fn in_memory(&self) -> bool {
        !matches!(self, Self::Filed { .. })
    }

    /// Where the record of index entry `index` of a segment left in the
    /// file lies there: its file offset and length.
    fn place(&self, index: usize) -> Option<(u64, u64)> {
        let Self::Filed { index: prefix, at } = self else {
            return None;
        };
        let (_, offset, bytes) = entry(prefix, index);
        Some((at + offset, bytes))
    }

    fn format(&self) -> PackFormat {
        PackFormat::read(self.index()).unwrap_or(PackFormat::Raw)
    }

    fn key(&self, index: usize) -> Digest {
        entry(self.index(), index).0
    }

    fn age(&self, index: usize) -> u8 {
        entry_age(self.index(), index)
    }

    fn uses(&self, index: usize) -> u8 {
        entry_uses(self.index(), index)
    }

    /// The record of index entry `index`, to decode.
    fn record(&self, index: usize) -> PackedRecord<'_> {
        match self {
            Self::Packed(pack) => pack.record_at(index),
            Self::Laid(layout) => PackedRecord {
                bytes: layout.records[layout.place(index)].1.bytes(),
                format: self.format(),
                checked: false,
            },
            Self::Filed { .. } => unreachable!("a merge reads a filed record from the file"),
        }
    }

    /// The record of index entry `index` for a merge's layout, which reads
    /// it no more: moved out of a layout nothing else holds, or where it
    /// lies in the file.
    fn take(&mut self, index: usize) -> Piece<'a> {
        if let Some((at, bytes)) = self.place(index) {
            return Piece::File { at, bytes };
        }
        match self {
            Self::Packed(pack) => Piece::Kept(pack.record_at(index).bytes),
            Self::Laid(layout) => {
                let place = layout.place(index);
                match Arc::get_mut(layout) {
                    Some(layout) => {
                        core::mem::replace(&mut layout.records[place].1, Piece::Kept(&[]))
                    }
                    None => layout.records[place].1.clone(),
                }
            }
            Self::Filed { .. } => unreachable!("placed above"),
        }
    }
}

/// A record of a pack being laid out: one of the read pack's, new, or one
/// that stays in the file until it is written ([`Streamed`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Piece<'a> {
    Kept(&'a [u8]),
    New(Vec<u8>),
    File { at: u64, bytes: u64 },
}

impl Piece<'_> {
    /// Its bytes in memory: none for a record in the file.
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Kept(bytes) => bytes,
            Self::New(bytes) => bytes,
            Self::File { .. } => &[],
        }
    }

    fn len(&self) -> u64 {
        match self {
            Self::File { bytes, .. } => *bytes,
            _ => self.bytes().len() as u64,
        }
    }
}

/// A pack to write: its header, index and checksum, then its records. It is
/// a journal blob's source ([`journal::Source`]): written, its records are
/// read from where they are, never copied whole; those still in the file
/// come through a [`Streamed`] window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout<'a> {
    prefix: Vec<u8>,
    /// Each record's offset and bytes, by offset.
    records: Vec<(u64, Piece<'a>)>,
    length: u64,
    /// The heap bytes it holds ([`Layout::owned`]).
    owned: u64,
}

/// Zeros for the room between records.
static ZEROS: [u8; 4096] = [0; 4096];

impl<'a> Layout<'a> {
    /// The pack of `records` (page keys and their record bytes, each key
    /// once) for `translator`, its records in key order after the index.
    pub fn new(translator: &Digest, records: Vec<(Digest, Piece<'a>)>) -> Self {
        Self::with_format(translator, records, PackFormat::Raw)
    }

    fn with_format(
        translator: &Digest,
        records: Vec<(Digest, Piece<'a>)>,
        format: PackFormat,
    ) -> Self {
        Self::compact(translator, records, format)
    }

    /// Every record laid out in key order after the index, each of age 0.
    fn compact(
        translator: &Digest,
        mut records: Vec<(Digest, Piece<'a>)>,
        format: PackFormat,
    ) -> Self {
        records.sort_by_key(|record| record.0);
        let mut offset = prefix_length(records.len()) as u64;
        let records: Vec<(Digest, u64, Piece<'a>)> = records
            .into_iter()
            .map(|(key, piece)| {
                let at = offset;
                offset += piece.len();
                (key, at, piece)
            })
            .collect();
        let count = records.len();
        let mut prefix = Vec::with_capacity(prefix_length(count));
        prefix.extend(MAGIC);
        prefix.extend(format.schema().to_le_bytes());
        prefix.extend((count as u32).to_le_bytes());
        prefix.extend(translator.0);
        for (key, offset, piece) in &records {
            prefix.extend(key.0);
            prefix.extend(offset.to_le_bytes());
            prefix.extend(piece.len().to_le_bytes());
        }
        let checksum = Sha256::digest(&prefix);
        prefix.extend(checksum);
        let records: Vec<(u64, Piece<'a>)> = records
            .into_iter()
            .map(|(_, offset, piece)| (offset, piece))
            .collect();
        Self::of(prefix, records, offset)
    }

    /// The layout of `prefix` and `records`, `length` bytes long.
    fn of(prefix: Vec<u8>, records: Vec<(u64, Piece<'a>)>, length: u64) -> Self {
        let pieces: usize = records
            .iter()
            .map(|(_, piece)| match piece {
                Piece::New(bytes) => bytes.capacity(),
                Piece::Kept(_) | Piece::File { .. } => 0,
            })
            .sum();
        let owned = prefix.capacity() + records.capacity() * size_of::<(u64, Piece<'a>)>() + pieces;
        Self {
            prefix,
            records,
            length,
            owned: owned as u64,
        }
    }

    pub fn records(&self) -> usize {
        self.records.len()
    }

    /// The place among its records (by offset) of index entry `index`: a
    /// merge lays records out in another order than their keys'.
    fn place(&self, index: usize) -> usize {
        let (_, offset, _) = entry(&self.prefix, index);
        self.records.partition_point(|(start, _)| *start < offset)
    }

    /// The heap bytes it holds, as laid out: its index and the records
    /// encoded for it. Those kept from a pack read are the pack's.
    pub fn owned(&self) -> u64 {
        self.owned
    }

    /// The pack's length in bytes.
    pub fn bytes(&self) -> u64 {
        self.length
    }

    /// Its header, index and checksum: what memory keeps of it once it is
    /// written ([`Segment::Filed`]).
    pub fn into_prefix(self) -> Vec<u8> {
        self.prefix
    }

    pub fn prefix(&self) -> &[u8] {
        &self.prefix
    }

    /// Whether a record of it is still in the file ([`Streamed`]).
    pub fn streamed(&self) -> bool {
        self.records
            .iter()
            .any(|(_, piece)| matches!(piece, Piece::File { .. }))
    }

    /// The pack's bytes, in order, its records still in the file as zeros.
    pub fn chunks(&self) -> impl Iterator<Item = &[u8]> + '_ {
        let mut chunks: Vec<&[u8]> = Vec::with_capacity(1 + 2 * self.records.len());
        chunks.push(&self.prefix);
        let mut at = self.prefix.len() as u64;
        for (offset, piece) in &self.records {
            for gap in zeros(offset - at) {
                chunks.push(gap);
            }
            if matches!(piece, Piece::File { .. }) {
                for gap in zeros(piece.len()) {
                    chunks.push(gap);
                }
            } else {
                chunks.push(piece.bytes());
            }
            at = offset + piece.len();
        }
        chunks.into_iter()
    }

    pub fn to_vec(&self) -> Vec<u8> {
        self.chunks().flatten().copied().collect()
    }

    /// Fill `buffer` with the pack's bytes from `offset`, zeros past its end
    /// and for its records still in the file.
    pub fn read_at(&self, offset: u64, buffer: &mut [u8]) {
        buffer.fill(0);
        let end = offset + buffer.len() as u64;
        let copy = |buffer: &mut [u8], start: u64, bytes: &[u8]| {
            let from = offset.max(start);
            let to = end.min(start + bytes.len() as u64);
            if from < to {
                buffer[(from - offset) as usize..(to - offset) as usize]
                    .copy_from_slice(&bytes[(from - start) as usize..(to - start) as usize]);
            }
        };
        copy(buffer, 0, &self.prefix);
        let first = self
            .records
            .partition_point(|(start, piece)| start + piece.len() <= offset);
        for (start, piece) in &self.records[first..] {
            if *start >= end {
                break;
            }
            copy(buffer, *start, piece.bytes());
        }
    }

    /// The file reads that fill its range `offset..offset + bytes`: of its
    /// records there still in the file, in runs of those that lie one after
    /// another in both, up to [`READ_BYTES`]: each run's file offset, bytes,
    /// and where it starts in the range.
    pub fn file_runs(&self, offset: u64, bytes: u64) -> Vec<(u64, u64, u64)> {
        let end = offset + bytes;
        let first = self
            .records
            .partition_point(|(start, piece)| start + piece.len() <= offset);
        let mut runs: Vec<(u64, u64, u64)> = Vec::new();
        for (start, piece) in &self.records[first..] {
            if *start >= end {
                break;
            }
            let Piece::File { at, bytes } = *piece else {
                continue;
            };
            // The part of the record within the range.
            let from = offset.max(*start);
            let to = end.min(start + bytes);
            let file = at + (from - start);
            let length = to - from;
            let into = from - offset;
            match runs.last_mut() {
                Some(run)
                    if run.0 + run.1 == file
                        && run.2 + run.1 == into
                        && run.1 + length <= READ_BYTES =>
                {
                    run.1 += length;
                }
                _ => runs.push((file, length, into)),
            }
        }
        runs
    }
}

/// A layout whose records partly stay in the file, as a journal blob's
/// source: before each piece is written, the writer reads that piece's
/// records from the file and fills the window with the piece's bytes.
#[derive(Debug)]
pub struct Streamed<'a> {
    pub layout: Arc<Layout<'a>>,
    window: Mutex<(u64, Vec<u8>)>,
}

impl<'a> Streamed<'a> {
    pub fn new(layout: Arc<Layout<'a>>) -> Self {
        Self {
            layout,
            window: Mutex::new((0, Vec::new())),
        }
    }

    /// The layout's bytes `offset..offset + bytes`, with `reads` (each
    /// [`Layout::file_runs`] run's bytes, in order) laid over its records
    /// in the file: what the next piece's write reads.
    pub fn fill(&self, offset: u64, bytes: u64, reads: &[(u64, Vec<u8>)]) {
        let mut piece = alloc::vec![0; bytes as usize];
        self.layout.read_at(offset, &mut piece);
        for (into, read) in reads {
            let into = *into as usize;
            piece[into..into + read.len()].copy_from_slice(read);
        }
        *self.lock() = (offset, piece);
    }

    fn lock(&self) -> crate::platform::sync::MutexGuard<'_, (u64, Vec<u8>)> {
        self.window
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl journal::Source for Streamed<'_> {
    fn length(&self) -> u64 {
        self.layout.length
    }
    fn read_at(&self, offset: u64, into: &mut [u8]) {
        let window = self.lock();
        let (start, bytes) = (&window.0, &window.1);
        let end = offset + into.len() as u64;
        if offset >= *start && end <= start + bytes.len() as u64 {
            let at = (offset - start) as usize;
            into.copy_from_slice(&bytes[at..at + into.len()]);
        } else {
            drop(window);
            // Only a piece with no record in the file reads past the window.
            debug_assert!(self.layout.file_runs(offset, into.len() as u64).is_empty());
            self.layout.read_at(offset, into);
        }
    }
}

/// `bytes` zeros, in slices of [`ZEROS`].
fn zeros(bytes: u64) -> impl Iterator<Item = &'static [u8]> {
    let whole = (bytes / ZEROS.len() as u64) as usize;
    let rest = (bytes % ZEROS.len() as u64) as usize;
    core::iter::repeat_n(ZEROS.as_slice(), whole).chain((rest != 0).then(|| &ZEROS[..rest]))
}

/// The bytes of the header, index and checksum of a pack of `records`.
fn prefix_length(records: usize) -> usize {
    HEADER_BYTES + records * INDEX_ENTRY_BYTES + CHECKSUM_BYTES
}

/// The pack bytes a layout of `records` records holding `bytes` record
/// bytes takes.
fn layout_bytes(records: usize, bytes: u64) -> u64 {
    prefix_length(records) as u64 + bytes
}

impl journal::Source for Layout<'_> {
    fn length(&self) -> u64 {
        self.length
    }
    fn read_at(&self, offset: u64, into: &mut [u8]) {
        Layout::read_at(self, offset, into);
    }
}

/// The page keys a boot looked up and found, as 8-byte prefixes: a merge
/// counts the boots in a row each record went unused by them ([`Merge`]).
/// A prefix two keys share counts both as used, which only keeps a record
/// a little longer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsedKeys {
    every: bool,
    prefixes: Vec<u64>,
}

fn prefix(key: &Digest) -> u64 {
    u64::from_le_bytes(key.0[..8].try_into().expect("8 bytes"))
}

impl UsedKeys {
    /// The list of `keys` for `translator`, at most [`MAX_USED_KEYS`] of
    /// them; past that, a list that counts every record as used.
    pub fn encode<'k>(translator: &Digest, keys: impl Iterator<Item = &'k Digest>) -> Vec<u8> {
        let mut prefixes: Vec<u64> = keys.map(prefix).collect();
        prefixes.sort_unstable();
        prefixes.dedup();
        let every = prefixes.len() > MAX_USED_KEYS;
        if every {
            prefixes.clear();
        }
        let mut bytes = Vec::with_capacity(USED_HEADER_BYTES + prefixes.len() * 8);
        bytes.extend(USED_MAGIC);
        bytes.extend(translator.0);
        bytes.extend((if every { USED_EVERY } else { 0 }).to_le_bytes());
        bytes.extend((prefixes.len() as u32).to_le_bytes());
        for prefix in prefixes {
            bytes.extend(prefix.to_le_bytes());
        }
        bytes
    }

    /// The list of a boot that used more keys than a list names.
    pub fn encode_every(translator: &Digest) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(USED_HEADER_BYTES);
        bytes.extend(USED_MAGIC);
        bytes.extend(translator.0);
        bytes.extend(USED_EVERY.to_le_bytes());
        bytes.extend(0u32.to_le_bytes());
        bytes
    }

    /// The list `bytes` holds for `translator`: `None` for another's, a
    /// damaged one or one of unsorted keys.
    pub fn parse(bytes: &[u8], translator: &Digest) -> Option<Self> {
        let header = bytes.get(..USED_HEADER_BYTES)?;
        let word = |at: usize| u32::from_le_bytes(header[at..at + 4].try_into().expect("a word"));
        if header[..8] != USED_MAGIC || header[8..8 + DIGEST_BYTES] != translator.0 {
            return None;
        }
        let flags = word(8 + DIGEST_BYTES);
        let count = word(12 + DIGEST_BYTES) as usize;
        if flags & !USED_EVERY != 0
            || count > MAX_USED_KEYS
            || bytes.len() != USED_HEADER_BYTES + count * 8
        {
            return None;
        }
        let prefixes: Vec<u64> = bytes[USED_HEADER_BYTES..]
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| u64::from_le_bytes(*chunk))
            .collect();
        prefixes
            .windows(2)
            .all(|pair| pair[0] < pair[1])
            .then_some(Self {
                every: flags & USED_EVERY != 0,
                prefixes,
            })
    }

    /// Whether its boot used more keys than a list names: every record
    /// counts as used.
    pub fn every(&self) -> bool {
        self.every
    }

    /// Whether its boot used `key`.
    pub fn contains(&self, key: &Digest) -> bool {
        self.every || self.prefixes.binary_search(&prefix(key)).is_ok()
    }
}

/// The entries a boot added since its last segment, laid out as a new
/// segment a step at a time: each step encodes or indexes a bounded number
/// of records.
pub struct Delta {
    records: alloc::collections::btree_map::IntoIter<Digest, Record>,
    /// A merge of no segments, given the encoded records in key order.
    merge: Merge<'static>,
}

impl Delta {
    pub fn new(translator: Digest, records: BTreeMap<Digest, Record>) -> Self {
        let mut merge = Merge::new(translator, Vec::new(), Vec::new(), u64::MAX);
        merge.records.reserve_exact(records.len());
        Self {
            records: records.into_iter(),
            merge,
        }
    }

    /// Carry out up to `work` more units (a record encoded or indexed):
    /// whether the segment is laid out.
    pub fn step(&mut self, work: usize) -> bool {
        for done in 0..work {
            let Some((key, record)) = self.records.next() else {
                return self.merge.step(work - done);
            };
            self.merge.push(key, Piece::New(record.encode()));
        }
        false
    }

    /// The segment, every step carried out: `None` when it holds no record.
    pub fn finish(mut self) -> Option<Layout<'static>> {
        while !self.step(usize::MAX) {}
        // A merge of no segment reads nothing.
        let layout = self.merge.finish()?.layout;
        (layout.records() != 0).then_some(layout)
    }
}

/// Every segment of a file and its used-key lists merged into one segment
/// within a bound, a step at a time: page keys are taken in order from every
/// segment's index at once, a record one segment holds kept as its bytes and
/// one several hold decoded, merged and encoded again; then the records
/// that fit are chosen, checked and indexed. A record's age is the boots in
/// a row since one used it: those of its segments, plus the boots of the
/// used-key lists since the last that names it.
///
/// A segment left in the file ([`Segment::Filed`]) is merged from its
/// index: a record several segments hold is read to merge it, a record kept
/// as it is is read once to check it whole (it decodes with every check, as
/// a lookup's first decode does) and once more as its piece is written
/// ([`Streamed`]). Reads go through [`Self::reads`], a bounded number of
/// bytes ahead; a read that fails ends the merge ([`Self::failed`]), and the
/// file keeps what it held.
///
/// The records are laid out in the order the running boot first used them
/// ([`Use::first`]), those it did not use after them in the order earlier
/// layouts had them, so the next boot reads ahead what it needs next.
pub struct Merge<'a> {
    translator: Digest,
    segments: Vec<Segment<'a>>,
    used: Vec<UsedKeys>,
    /// Keys the running boot used so far: their records are of age 0, kept
    /// first whatever their age, and its own list counts the boot at the
    /// next merge.
    protect: Option<UsedKeys>,
    bound: u64,
    /// Each segment's next index entry.
    cursors: Vec<usize>,
    format: PackFormat,
    scratch: DecodeScratch,
    /// The records taken, in key order, and their ranks ([`DROPPED`] for
    /// those left out).
    records: Vec<(Digest, Piece<'a>, u8)>,
    /// The bytes and records of each rank: a record's age, one more for
    /// those the running boot did not use when it names those it did
    /// ([`Merge::protect`]), so they are kept first.
    by_age: [(u64, usize); RANKS],
    /// The running boot's uses of each key ([`Merge::by_use`]), and the
    /// records of the rank that does not fit whole that are kept: those
    /// used most, until the bound is full.
    uses: BTreeMap<Digest, Use>,
    chosen: Vec<bool>,
    /// Per record taken, the most uses its segments counted, and where an
    /// earlier layout placed it (its segment and offset there).
    counted: Vec<u8>,
    origin: Vec<u64>,
    /// Records several segments hold, one or more of them in the file:
    /// merged once their reads end.
    pending: Vec<Pending>,
    /// Records kept as one segment in the file holds them, to check whole:
    /// each record and its segment's format.
    unchecked: Vec<(usize, PackFormat)>,
    reads: FileReads,
    phase: Phase<'a>,
    evicted: u64,
    expired: u64,
    invalid: u64,
}

/// A record several segments hold, some in the file: its place among the
/// records taken, its age, and each holder (segment and index entry) with
/// the read of its bytes when it is in the file.
struct Pending {
    record: usize,
    age: u8,
    holders: Vec<(usize, usize, Option<u64>)>,
}

/// A run of records in the file a merge reads at once to check them: its
/// file offset and bytes, the records (and formats) it holds, and its read.
struct Run {
    at: u64,
    bytes: u64,
    records: Vec<(usize, PackFormat)>,
    read: Option<u64>,
}

/// The age of a record a merge leaves out to stay within its bound.
const DROPPED: u8 = u8::MAX;
/// The ranks a merge keeps records by ([`Merge::rank`]).
const RANKS: usize = MAX_UNUSED_BOOTS as usize + 2;

/// Where a [`Merge`] is.
enum Phase<'a> {
    /// Taking page keys from the segments.
    Taking,
    /// Merging the records of [`Merge::pending`] whose reads ended: `at`
    /// is the next, `asked` the first whose reads are not asked yet.
    Resolving {
        at: usize,
        asked: usize,
    },
    /// Choosing, of the age that does not fit whole (`oldest`), the records
    /// that do, in key order; `count` and `bytes` are those kept so far.
    Choosing {
        oldest: Option<usize>,
        at: usize,
        count: usize,
        bytes: u64,
    },
    /// Checking the records kept from the file, a run at a time: `at` the
    /// next run, `asked` the first not asked yet.
    Checking {
        runs: Vec<Run>,
        at: usize,
        asked: usize,
    },
    /// Writing the index of the `count` records kept, with the checksum of
    /// its first `hashed` bytes, and moving them to the layout at their
    /// places (`places`, by record: the offset and the place in `placed`).
    Indexing {
        at: usize,
        prefix: Vec<u8>,
        hasher: Sha256,
        hashed: usize,
        places: Vec<(u64, u32)>,
        placed: Vec<(u64, Piece<'a>)>,
        length: u64,
    },
    Done(Layout<'a>),
    /// A read failed: nothing is laid out.
    Failed,
}

/// The segment a [`Merge`] made, and what it left out.
#[derive(Debug)]
pub struct Merged<'a> {
    pub layout: Layout<'a>,
    /// Records left out to stay within the bound, those unused for more
    /// than [`MAX_UNUSED_BOOTS`] boots in a row, and copies that did not
    /// decode.
    pub evicted: u64,
    pub expired: u64,
    pub invalid: u64,
}

impl<'a> Merge<'a> {
    /// The merge of `segments`, oldest first, and of `used`, the used-key
    /// lists of the boots since the last merge, oldest first, into a
    /// segment of at most `bound` bytes and [`MAX_RECORDS`] records: the
    /// records used most recently are kept first.
    pub fn new(
        translator: Digest,
        segments: Vec<Segment<'a>>,
        used: Vec<UsedKeys>,
        bound: u64,
    ) -> Self {
        let format = if segments
            .iter()
            .any(|segment| segment.format() == PackFormat::Mixed)
        {
            PackFormat::Mixed
        } else {
            PackFormat::Raw
        };
        Self {
            translator,
            cursors: alloc::vec![0; segments.len()],
            segments,
            used,
            protect: None,
            bound,
            format,
            scratch: DecodeScratch::default(),
            records: Vec::new(),
            by_age: [(0, 0); RANKS],
            uses: BTreeMap::new(),
            chosen: Vec::new(),
            counted: Vec::new(),
            origin: Vec::new(),
            pending: Vec::new(),
            unchecked: Vec::new(),
            reads: FileReads::default(),
            phase: Phase::Taking,
            evicted: 0,
            expired: 0,
            invalid: 0,
        }
    }

    /// The bound its segment stays within.
    pub fn bound(&self) -> u64 {
        self.bound
    }

    /// The reads of the file it waits for, for the boot to carry out.
    pub fn reads(&mut self) -> &mut FileReads {
        &mut self.reads
    }

    /// Whether a read failed: it lays nothing out ([`Self::abandon`]).
    pub fn failed(&self) -> bool {
        matches!(self.phase, Phase::Failed)
    }

    /// Give up the merge: the segments it took back. A layout held only
    /// by the merge may have given it records; the caller merges it no
    /// more.
    pub fn abandon(self) -> Vec<Segment<'a>> {
        self.segments
    }

    /// Keep the records of `keys`, those the running boot used so far,
    /// whatever their age.
    pub fn protect(mut self, keys: UsedKeys) -> Self {
        self.protect = Some(keys);
        self
    }

    /// Of the rank that does not fit whole, keep the records the running
    /// boot used most (`uses`, by page key), not the first in key order:
    /// a merge that ran while a boot had used part of its pack left out
    /// pages by their hash, those a session runs among them. The uses'
    /// first ranks lay the records out in the order the boot needed them.
    pub fn by_use(mut self, uses: BTreeMap<Digest, Use>) -> Self {
        self.uses = uses;
        self
    }

    /// The running boot's uses of `key`.
    fn count(&self, key: &Digest) -> u32 {
        self.uses.get(key).map_or(0, |uses| uses.count)
    }

    /// Which records of rank `oldest` fit after `count` records of `bytes`
    /// kept whole: the most used first, ties in key order.
    fn choose(&self, oldest: Option<usize>, mut count: usize, mut bytes: u64) -> Vec<bool> {
        let Some(oldest) = oldest else {
            return Vec::new();
        };
        let mut order: Vec<(u32, u8, usize)> = self
            .records
            .iter()
            .enumerate()
            .filter(|(_, (_, _, rank))| usize::from(*rank) == oldest)
            .map(|(at, (key, _, _))| (self.count(key), self.counted[at], at))
            .collect();
        order.sort_by_key(|&(uses, counted, at)| {
            (core::cmp::Reverse(uses), core::cmp::Reverse(counted), at)
        });
        let mut chosen = alloc::vec![false; self.records.len()];
        for (_, _, at) in order {
            let more = self.records[at].1.len();
            if count < MAX_RECORDS && layout_bytes(count + 1, bytes + more) <= self.bound {
                chosen[at] = true;
                count += 1;
                bytes += more;
            }
        }
        chosen
    }

    /// Add a record of age 0 after those taken, its key past theirs.
    fn push(&mut self, key: Digest, piece: Piece<'a>) {
        debug_assert!(self.records.last().is_none_or(|last| last.0 < key));
        self.by_age[0].0 += piece.len();
        self.by_age[0].1 += 1;
        self.origin.push(self.records.len() as u64);
        self.records.push((key, piece, 0));
        self.counted.push(0);
    }

    /// The rank a record of `age` is kept by: its age, one more when the
    /// running boot names the keys it used and not this one.
    fn rank(&self, key: &Digest, age: u8) -> u8 {
        age + u8::from(
            self.protect
                .as_ref()
                .is_some_and(|keys| !keys.contains(key)),
        )
    }

    /// The age of a record whose segments gave it `age`.
    fn age(&self, key: &Digest, age: u8) -> u8 {
        let boots = self.used.len();
        let age = match self.used.iter().rposition(|used| used.contains(key)) {
            Some(last) => (boots - 1 - last) as u8,
            None => age.saturating_add(boots as u8),
        };
        match &self.protect {
            Some(protect) if protect.contains(key) => 0,
            _ => age,
        }
    }

    /// Count a record taken at rank `rank`.
    fn count_rank(&mut self, rank: u8, bytes: u64) {
        let slot = &mut self.by_age[usize::from(rank)];
        slot.0 += bytes;
        slot.1 += 1;
    }

    /// Take the next page key from the segments: `false` once none is left.
    fn take(&mut self) -> bool {
        let next = self
            .segments
            .iter()
            .zip(&self.cursors)
            .filter(|(segment, at)| **at < segment.records())
            .map(|(segment, &at)| segment.key(at))
            .min();
        let Some(key) = next else {
            return false;
        };
        let mut holders = Vec::new();
        for (index, segment) in self.segments.iter().enumerate() {
            let at = self.cursors[index];
            if at < segment.records() && segment.key(at) == key {
                holders.push((index, at));
                self.cursors[index] += 1;
            }
        }
        let age = holders
            .iter()
            .map(|&(index, at)| self.segments[index].age(at))
            .min()
            .unwrap_or(0);
        let age = self.age(&key, age);
        if age > MAX_UNUSED_BOOTS {
            self.expired += 1;
            return true;
        }
        let rank = self.rank(&key, age);
        let counted = holders
            .iter()
            .map(|&(index, at)| self.segments[index].uses(at))
            .max()
            .unwrap_or(0);
        // Where the newest holder placed it: the order of records no boot
        // used this time.
        let &(newest, newest_at) = holders.last().expect("a holder");
        let (_, offset, _) = entry(self.segments[newest].index(), newest_at);
        self.origin.push((newest as u64) << 48 | offset);
        self.counted.push(counted);
        let record = self.records.len();
        let piece = if let [(index, at)] = holders[..] {
            let piece = self.segments[index].take(at);
            if matches!(piece, Piece::File { .. }) {
                self.unchecked.push((record, self.segments[index].format()));
            }
            piece
        } else if holders
            .iter()
            .all(|&(index, _)| self.segments[index].in_memory())
        {
            match self.merged(&key, &holders, &[]) {
                Some(piece) => piece,
                None => {
                    self.origin.pop();
                    self.counted.pop();
                    return true;
                }
            }
        } else {
            // Merged once the holders in the file are read.
            self.pending.push(Pending {
                record,
                age: rank,
                holders: holders
                    .iter()
                    .map(|&(index, at)| (index, at, None))
                    .collect(),
            });
            self.records
                .push((key, Piece::File { at: 0, bytes: 0 }, DROPPED));
            return true;
        };
        self.count_rank(rank, piece.len());
        self.records.push((key, piece, rank));
        true
    }

    /// The record of `key` that `holders` hold, decoded, merged and encoded
    /// again, each holder's bytes from memory or `read` (by holder, those
    /// in the file): the bytes of a holder whose record is already the
    /// merged one, else the new encoding; `None` when none decoded.
    fn merged(
        &mut self,
        key: &Digest,
        holders: &[(usize, usize)],
        read: &[Option<&[u8]>],
    ) -> Option<Piece<'a>> {
        let mut merged: Option<Record> = None;
        fn bytes<'s>(
            segments: &'s [Segment<'_>],
            holders: &[(usize, usize)],
            read: &[Option<&'s [u8]>],
            holder: usize,
        ) -> PackedRecord<'s> {
            let (index, at) = holders[holder];
            match read.get(holder).copied().flatten() {
                Some(bytes) => PackedRecord {
                    bytes,
                    format: segments[index].format(),
                    checked: false,
                },
                None => segments[index].record(at),
            }
        }
        let mut scratch = core::mem::take(&mut self.scratch);
        for holder in 0..holders.len() {
            let stored = bytes(&self.segments, holders, read, holder);
            match stored.decode(&self.translator, key, &mut scratch) {
                Ok(record) => match &mut merged {
                    Some(merged) => merged.merge(record),
                    None => merged = Some(record),
                },
                Err(_) => self.invalid += 1,
            }
        }
        self.scratch = scratch;
        // A holder whose record is already the merged one (as a boot's
        // record of a page it found holds what it read) gives its bytes:
        // the merge copies only records it changes.
        let encoded = merged?.encode();
        let same = (0..holders.len())
            .find(|&holder| *bytes(&self.segments, holders, read, holder).bytes == encoded[..]);
        Some(match same {
            Some(holder) => {
                let (index, at) = holders[holder];
                self.segments[index].take(at)
            }
            None => Piece::New(encoded),
        })
    }

    /// Ask for the reads of the pending records from `from` while what is
    /// asked and not used stays within [`READS_HELD_BYTES`]: the first not
    /// asked after.
    fn ask_pending(&mut self, from: usize, mut asked: usize) -> usize {
        let held = |merge: &Self, upto: usize| -> u64 {
            merge.pending[from..upto]
                .iter()
                .flat_map(|pending| &pending.holders)
                .filter_map(|&(index, at, _)| merge.segments[index].place(at))
                .map(|(_, bytes)| bytes)
                .sum()
        };
        while asked < self.pending.len() && (asked == from || held(self, asked) < READS_HELD_BYTES)
        {
            for holder in 0..self.pending[asked].holders.len() {
                let (index, at, _) = self.pending[asked].holders[holder];
                if let Some((file, bytes)) = self.segments[index].place(at) {
                    let id = self.reads.ask(file, bytes);
                    self.pending[asked].holders[holder].2 = Some(id);
                }
            }
            asked += 1;
        }
        asked
    }

    /// Merge pending record `at` once its reads ended: `None` while one is
    /// under way, `Some(false)` when one failed.
    fn resolve(&mut self, at: usize) -> Option<bool> {
        let ids: Vec<Option<u64>> = self.pending[at]
            .holders
            .iter()
            .map(|&(_, _, id)| id)
            .collect();
        if ids
            .iter()
            .flatten()
            .any(|id| !self.reads.done.contains_key(id))
        {
            return None;
        }
        let mut read: Vec<Option<Vec<u8>>> = Vec::with_capacity(ids.len());
        for id in &ids {
            match id.map(|id| self.reads.take(id)) {
                Some(Some(Some(bytes))) => read.push(Some(bytes)),
                Some(_) => return Some(false),
                None => read.push(None),
            }
        }
        let read: Vec<Option<&[u8]>> = read.iter().map(Option::as_deref).collect();
        let holders: Vec<(usize, usize)> = self.pending[at]
            .holders
            .iter()
            .map(|&(index, at, _)| (index, at))
            .collect();
        let (record, rank) = (self.pending[at].record, self.pending[at].age);
        let key = self.records[record].0;
        if let Some(piece) = self.merged(&key, &holders, &read) {
            self.count_rank(rank, piece.len());
            self.records[record].1 = piece;
            self.records[record].2 = rank;
        }
        Some(true)
    }

    /// The runs of the records kept from the file unchecked, in file order.
    fn runs(&self) -> Vec<Run> {
        let mut kept: Vec<(u64, u64, usize, PackFormat)> = self
            .unchecked
            .iter()
            .filter(|&&(record, _)| self.records[record].2 != DROPPED)
            .filter_map(|&(record, format)| match self.records[record].1 {
                Piece::File { at, bytes } => Some((at, bytes, record, format)),
                _ => None,
            })
            .collect();
        kept.sort_unstable_by_key(|&(at, ..)| at);
        let mut runs: Vec<Run> = Vec::new();
        for (at, bytes, record, format) in kept {
            match runs.last_mut() {
                Some(run) if run.at + run.bytes == at && run.bytes + bytes <= READ_BYTES => {
                    run.bytes += bytes;
                    run.records.push((record, format));
                }
                _ => runs.push(Run {
                    at,
                    bytes,
                    records: alloc::vec![(record, format)],
                    read: None,
                }),
            }
        }
        runs
    }

    /// Check run `run` once its read ended: `None` while it is under way,
    /// `Some(false)` when it failed. A record that does not decode whole is
    /// left out.
    fn check(&mut self, runs: &[Run], run: usize) -> Option<bool> {
        let id = runs[run].read.expect("asked");
        let bytes = match self.reads.take(id)? {
            Some(bytes) if bytes.len() as u64 == runs[run].bytes => bytes,
            _ => return Some(false),
        };
        let mut scratch = core::mem::take(&mut self.scratch);
        for &(record, format) in &runs[run].records {
            let (key, piece, rank) = &mut self.records[record];
            let Piece::File { at, bytes: length } = *piece else {
                continue;
            };
            let from = (at - runs[run].at) as usize;
            let record_bytes = &bytes[from..from + length as usize];
            if !verified(record_bytes, format, &self.translator, key, &mut scratch) {
                *rank = DROPPED;
                self.invalid += 1;
            }
        }
        self.scratch = scratch;
        Some(true)
    }

    /// The ages kept whole, youngest first, until one does not fit: that
    /// age, and the records and bytes of those before it.
    fn whole_ages(&self) -> (Option<usize>, usize, u64) {
        let (mut count, mut bytes) = (0usize, 0u64);
        for (age, &(more, records)) in self.by_age.iter().enumerate() {
            if count + records > MAX_RECORDS
                || layout_bytes(count + records, bytes + more) > self.bound
            {
                return (Some(age), count, bytes);
            }
            count += records;
            bytes += more;
        }
        (None, count, bytes)
    }

    /// Where each kept record goes: records the running boot used in the
    /// order it first used them, then the others in their earlier layouts'
    /// order. By record: its offset and its place among the kept.
    fn places(&self, count: usize) -> (Vec<(u64, u32)>, u64) {
        let mut order: Vec<((bool, u64), usize)> = self
            .records
            .iter()
            .enumerate()
            .filter(|(_, (_, _, rank))| *rank != DROPPED)
            .map(|(at, (key, _, _))| {
                let rank = match self.uses.get(key) {
                    Some(uses) => (false, u64::from(uses.first)),
                    None => (true, self.origin[at]),
                };
                (rank, at)
            })
            .collect();
        order.sort_unstable();
        let mut places = alloc::vec![(0u64, 0u32); self.records.len()];
        let mut offset = prefix_length(count) as u64;
        for (place, &(_, at)) in order.iter().enumerate() {
            places[at] = (offset, place as u32);
            offset += self.records[at].1.len();
        }
        (places, offset)
    }

    /// Carry out up to `work` more units (a page key taken, a record merged,
    /// chosen, checked or indexed): whether the segment is laid out, or the
    /// merge failed. While its reads are under way it does no more.
    pub fn step(&mut self, work: usize) -> bool {
        for _ in 0..work {
            match &mut self.phase {
                Phase::Taking => {
                    if !self.take() {
                        self.phase = Phase::Resolving { at: 0, asked: 0 };
                    }
                }
                &mut Phase::Resolving { at, asked } => {
                    if at == self.pending.len() {
                        self.pending = Vec::new();
                        let (oldest, count, bytes) = self.whole_ages();
                        self.chosen = self.choose(oldest, count, bytes);
                        self.phase = Phase::Choosing {
                            oldest,
                            at: 0,
                            count,
                            bytes,
                        };
                        continue;
                    }
                    let asked = self.ask_pending(at, asked);
                    match self.resolve(at) {
                        None => {
                            self.phase = Phase::Resolving { at, asked };
                            return false;
                        }
                        Some(false) => {
                            self.fail();
                            return true;
                        }
                        Some(true) => self.phase = Phase::Resolving { at: at + 1, asked },
                    }
                }
                Phase::Choosing {
                    oldest,
                    at,
                    count,
                    bytes,
                } => {
                    let Some((_, piece, age)) = self.records.get_mut(*at) else {
                        self.phase = Phase::Checking {
                            runs: self.runs(),
                            at: 0,
                            asked: 0,
                        };
                        self.unchecked = Vec::new();
                        continue;
                    };
                    let index = *at;
                    *at += 1;
                    if *age == DROPPED {
                        continue;
                    }
                    let keep = match *oldest {
                        None => true,
                        Some(oldest) if usize::from(*age) < oldest => true,
                        Some(oldest) if usize::from(*age) == oldest => {
                            let more = piece.len();
                            let fits = self.chosen[index];
                            if fits {
                                *bytes += more;
                                *count += 1;
                            }
                            fits
                        }
                        Some(_) => false,
                    };
                    if !keep {
                        *age = DROPPED;
                        self.evicted += 1;
                    }
                }
                Phase::Checking { runs, at, asked } => {
                    if *at == runs.len() {
                        let count = self
                            .records
                            .iter()
                            .filter(|(_, _, rank)| *rank != DROPPED)
                            .count();
                        let mut prefix = Vec::with_capacity(prefix_length(count));
                        prefix.extend(MAGIC);
                        prefix.extend(self.format.schema().to_le_bytes());
                        prefix.extend((count as u32).to_le_bytes());
                        prefix.extend(self.translator.0);
                        let (places, length) = self.places(count);
                        let mut placed = Vec::with_capacity(count);
                        placed.resize_with(count, || (0, Piece::Kept(&[])));
                        self.phase = Phase::Indexing {
                            at: 0,
                            prefix,
                            hasher: Sha256::new(),
                            hashed: 0,
                            places,
                            placed,
                            length,
                        };
                        continue;
                    }
                    // Ask ahead within the bound of bytes held.
                    let from = *at;
                    while *asked < runs.len()
                        && (*asked == from
                            || runs[from..*asked].iter().map(|run| run.bytes).sum::<u64>()
                                < READS_HELD_BYTES)
                    {
                        let run = &mut runs[*asked];
                        run.read = Some(self.reads.ask(run.at, run.bytes));
                        *asked += 1;
                    }
                    let mut runs = core::mem::take(runs);
                    let checked = self.check(&runs, from);
                    let Phase::Checking { runs: held, at, .. } = &mut self.phase else {
                        unreachable!("checking");
                    };
                    match checked {
                        None => {
                            *held = runs;
                            return false;
                        }
                        Some(false) => {
                            self.fail();
                            return true;
                        }
                        Some(true) => {
                            // A checked run's records are needed no more.
                            runs[from].records = Vec::new();
                            *held = runs;
                            *at += 1;
                        }
                    }
                }
                Phase::Indexing {
                    at,
                    prefix,
                    hasher,
                    hashed,
                    places,
                    placed,
                    length,
                } => {
                    let Some((key, piece, age)) = self.records.get_mut(*at) else {
                        hasher.update(&prefix[*hashed..]);
                        let checksum = core::mem::replace(hasher, Sha256::new()).finalize();
                        let mut prefix = core::mem::take(prefix);
                        prefix.extend(checksum);
                        self.phase =
                            Phase::Done(Layout::of(prefix, core::mem::take(placed), *length));
                        self.records = Vec::new();
                        return true;
                    };
                    let index = *at;
                    *at += 1;
                    if *age == DROPPED {
                        *piece = Piece::Kept(&[]);
                        continue;
                    }
                    let bytes = piece.len();
                    // The most uses a boot made of it: this boot's, or an
                    // earlier one's.
                    let boot = self.uses.get(key).map_or(0, |uses| uses.count);
                    let uses = u64::from(self.counted[index].max(boot.min(255) as u8));
                    // The rank less the step for records the running boot
                    // did not use ([`Merge::rank`]).
                    let unused = self
                        .protect
                        .as_ref()
                        .is_some_and(|keys| !keys.contains(key));
                    let age = *age - u8::from(unused);
                    let (offset, place) = places[index];
                    prefix.extend(key.0);
                    prefix.extend(offset.to_le_bytes());
                    prefix.extend(
                        (bytes | uses << USES_SHIFT | u64::from(age) << AGE_SHIFT).to_le_bytes(),
                    );
                    // Hash the index in pieces, so the checksum costs no
                    // step more than its share.
                    if prefix.len() - *hashed >= INDEX_HASH_BYTES {
                        hasher.update(&prefix[*hashed..]);
                        *hashed = prefix.len();
                    }
                    placed[place as usize] = (offset, core::mem::replace(piece, Piece::Kept(&[])));
                }
                Phase::Done(_) | Phase::Failed => return true,
            }
        }
        matches!(self.phase, Phase::Done(_) | Phase::Failed)
    }

    /// A read failed: the merge lays nothing out, and its reads end unread.
    fn fail(&mut self) {
        self.phase = Phase::Failed;
        self.reads.clear();
        self.records = Vec::new();
        self.pending = Vec::new();
    }

    /// The merged segment, every step carried out: `None` when a read
    /// failed or ends only once the boot carries out its reads.
    pub fn finish(mut self) -> Option<Merged<'a>> {
        while !self.step(usize::MAX) {
            if self.reads.waiting() {
                return None;
            }
        }
        let Phase::Done(layout) = self.phase else {
            return None;
        };
        Some(Merged {
            layout,
            evicted: self.evicted,
            expired: self.expired,
            invalid: self.invalid,
        })
    }
}

/// What a boot's store did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Counters {
    /// Records looked up and valid, absent, or invalid.
    pub loaded: u64,
    pub absent: u64,
    pub invalid: u64,
    /// Lookups of a key looked up before (its page let go and read again,
    /// or another PE's), those the decoded cache served, and records
    /// decoded (inflated and checked).
    pub repeated: u64,
    pub cached: u64,
    pub decoded: u64,
    /// Blocks offered whose stored form was added, those whose compiler
    /// made none, and those left out because the added bound was full
    /// ([`Store::keep`]).
    pub added: u64,
    pub unstorable: u64,
    pub unkept: u64,
    /// Of those left out, the distinct entries (each counted at its first
    /// refusal, [`SeenFilter`]); and offers of an entry a new segment took
    /// this boot, not added again.
    pub unkept_entries: u64,
    pub retaken: u64,
    /// Offers not kept because the heap was short ([`Store::shed`]).
    pub shed: u64,
    /// Shared lookups that found a block, and why the others missed: no
    /// offer of it, an offer of its PC and page in another context, its
    /// offer evicted, or another machine compiling it at that moment.
    pub shared_hits: u64,
    pub shared_absent: u64,
    pub shared_other_context: u64,
    pub shared_evicted: u64,
    pub shared_in_flight: u64,
    /// Lookups of a record still in the file (the block compiled
    /// meanwhile), records read from it and their bytes, and records read
    /// ahead of a lookup.
    pub awaited: u64,
    pub read: u64,
    pub read_bytes: u64,
    pub read_ahead: u64,
}

impl Counters {
    /// Add `other`'s counts, as of another part of one store.
    pub fn add(&mut self, other: &Self) {
        let Self {
            loaded,
            absent,
            invalid,
            repeated,
            cached,
            decoded,
            added,
            unstorable,
            unkept,
            unkept_entries,
            retaken,
            shed,
            shared_hits,
            shared_absent,
            shared_other_context,
            shared_evicted,
            shared_in_flight,
            awaited,
            read,
            read_bytes,
            read_ahead,
        } = other;
        self.loaded += loaded;
        self.absent += absent;
        self.invalid += invalid;
        self.repeated += repeated;
        self.cached += cached;
        self.decoded += decoded;
        self.added += added;
        self.unstorable += unstorable;
        self.unkept += unkept;
        self.unkept_entries += unkept_entries;
        self.retaken += retaken;
        self.shed += shed;
        self.shared_hits += shared_hits;
        self.shared_absent += shared_absent;
        self.shared_other_context += shared_other_context;
        self.shared_evicted += shared_evicted;
        self.shared_in_flight += shared_in_flight;
        self.awaited += awaited;
        self.read += read;
        self.read_bytes += read_bytes;
        self.read_ahead += read_ahead;
    }
}

/// A set of entries by page key, offset and context that answers whether
/// it holds one with false positives only: two bits of a bit array each,
/// allocated at the first insert. Its 2^21 bits (256 KiB) give each of a
/// 16-part store's share of 2 million entries 16 bits: under 2% wrong.
#[derive(Debug, Default)]
struct SeenFilter {
    bits: Vec<u64>,
}

impl SeenFilter {
    /// The words of a filter.
    const WORDS: usize = 1 << 15;

    fn bits(key: &Digest, offset: u16, context: &Digest) -> [usize; 2] {
        let word = |bytes: &[u8]| u64::from_le_bytes(bytes.try_into().expect("a word"));
        let mut mixed = word(&key.0[8..16])
            ^ u64::from(offset).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ word(&context.0[..8]).rotate_left(17);
        mixed = (mixed ^ mixed >> 31).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed ^= mixed >> 29;
        let total = Self::WORDS * 64;
        [
            (mixed as u32) as usize % total,
            ((mixed >> 32) as u32) as usize % total,
        ]
    }

    /// Insert the entry: whether it may have been in already.
    fn insert(&mut self, key: &Digest, offset: u16, context: &Digest) -> bool {
        if self.bits.is_empty() {
            self.bits = alloc::vec![0; Self::WORDS];
        }
        let mut held = true;
        for bit in Self::bits(key, offset, context) {
            let (word, mask) = (bit / 64, 1u64 << (bit % 64));
            held &= self.bits[word] & mask != 0;
            self.bits[word] |= mask;
        }
        held
    }

    /// Hold what `other` held too.
    fn absorb(&mut self, other: Self) {
        if self.bits.is_empty() {
            self.bits = other.bits;
        } else {
            for (word, other) in self.bits.iter_mut().zip(other.bits) {
                *word |= other;
            }
        }
    }

    fn contains(&self, key: &Digest, offset: u16, context: &Digest) -> bool {
        !self.bits.is_empty()
            && Self::bits(key, offset, context)
                .into_iter()
                .all(|bit| self.bits[bit / 64] & (1 << (bit % 64)) != 0)
    }
}

/// A shared block's PC and the leading words of its page key and context:
/// a compact index whose rare collision only replaces an entry, as a hit
/// still matches the whole key and context ([`SharedTranslation::matches`]).
type SharedKey = (u64, u64, u64);

fn shared_key(key: &Digest, pc: u64, context: &Digest) -> SharedKey {
    let word = |digest: &Digest| u64::from_le_bytes(digest.0[..8].try_into().expect("a word"));
    (pc, word(key), word(context))
}

/// A slot of the [`MissTables`] for `words`.
fn miss_slot(words: &[u64]) -> usize {
    let mixed = words.iter().fold(0u64, |hash, word| {
        (hash ^ word).wrapping_mul(0x9e37_79b9_7f4a_7c15)
    });
    (mixed >> 32) as usize % MISS_SLOTS
}

/// Why shared lookups miss, in direct-mapped tables a collision only
/// misnames: offers being compiled since their lookup missed, offers
/// evicted, and the context of the last offer of each PC and page.
#[derive(Debug, Default)]
struct MissTables {
    in_flight: Vec<SharedKey>,
    evicted: Vec<SharedKey>,
    offered: Vec<SharedKey>,
}

impl MissTables {
    fn ready(&mut self) -> bool {
        if self.in_flight.is_empty() {
            let mut tables = [Vec::new(), Vec::new(), Vec::new()];
            for table in &mut tables {
                if table.try_reserve_exact(MISS_SLOTS).is_err() {
                    return false;
                }
                table.resize(MISS_SLOTS, (u64::MAX, 0, 0));
            }
            [self.in_flight, self.evicted, self.offered] = tables;
        }
        true
    }

    /// Why the lookup of `key` missed, marking it in flight.
    fn miss(&mut self, key: SharedKey, counters: &mut Counters) {
        if !self.ready() {
            counters.shared_absent += 1;
            return;
        }
        let slot = miss_slot(&[key.0, key.1, key.2]);
        let place = miss_slot(&[key.0, key.1]);
        let reason = if self.in_flight[slot] == key {
            &mut counters.shared_in_flight
        } else if self.evicted[slot] == key {
            &mut counters.shared_evicted
        } else if self.offered[place].0 == key.0
            && self.offered[place].1 == key.1
            && self.offered[place].2 != key.2
        {
            &mut counters.shared_other_context
        } else {
            &mut counters.shared_absent
        };
        *reason += 1;
        self.in_flight[slot] = key;
    }

    fn offered(&mut self, key: SharedKey) {
        if !self.ready() {
            return;
        }
        let slot = miss_slot(&[key.0, key.1, key.2]);
        if self.in_flight[slot] == key {
            self.in_flight[slot] = (u64::MAX, 0, 0);
        }
        if self.evicted[slot] == key {
            self.evicted[slot] = (u64::MAX, 0, 0);
        }
        self.offered[miss_slot(&[key.0, key.1])] = key;
    }

    fn evicted(&mut self, key: SharedKey) {
        if self.ready() {
            self.evicted[miss_slot(&[key.0, key.1, key.2])] = key;
        }
    }
}

#[derive(Debug)]
struct SharedEntry {
    translation: Arc<SharedTranslation>,
    /// Immutable owner charge, computed once before admission.
    bytes: usize,
}

/// One local producer's shared offer, allocated and charged before the
/// producer acquires the store lock.
pub struct StagedOffer {
    key: SharedKey,
    entry: Option<SharedEntry>,
}

impl StagedOffer {
    /// `bound` is the producer's snapshot of the shared cache limit.
    /// Admission rechecks the store's limit.
    pub fn new(offer: Offer, bound: usize) -> Option<Self> {
        let (key, bytes) = Self::charge(&offer, bound)?;
        Some(Self::entry(key, bytes, offer.shared()))
    }

    fn charge(offer: &Offer, bound: usize) -> Option<(SharedKey, usize)> {
        if bound == 0 {
            return None;
        }
        let bytes = offer
            .retained_bytes()
            .saturating_add(size_of::<(SharedKey, SharedEntry)>());
        (bytes <= bound).then(|| {
            (
                shared_key(&offer.key(), offer.pc(), &offer.context()),
                bytes,
            )
        })
    }

    fn entry(key: SharedKey, bytes: usize, translation: Arc<SharedTranslation>) -> Self {
        Self {
            key,
            entry: Some(SharedEntry { translation, bytes }),
        }
    }
}

/// At most one displaced owner per producer can outlive the store lock.
/// The cap covers charged cache bytes, with the same exclusions as the
/// shared cache budget. Larger or additional owners are released inside.
const RETIRED_OWNER_BYTES: usize = 64 << 10;

fn retire(entry: SharedEntry, deferred: &mut Option<Arc<SharedTranslation>>) {
    if deferred.is_none() && entry.bytes <= RETIRED_OWNER_BYTES {
        *deferred = Some(entry.translation);
    }
}

/// A validated record paired with the immutable pack bytes it came from.
pub struct DecodedRecord<'a> {
    stored: Stored<'a>,
    record: Arc<StoredRecord>,
    retained_bytes: usize,
}

impl<'a> DecodedRecord<'a> {
    pub fn decode(
        stored: Stored<'a>,
        translator: &Digest,
        key: &Digest,
        scratch: &mut DecodeScratch,
    ) -> Result<Self, DecodeError> {
        let record = Arc::new(stored.stored_record(translator, key, scratch)?);
        let retained_bytes = record.retained_bytes();
        Ok(Self {
            stored,
            record,
            retained_bytes,
        })
    }

    pub fn into_record(self) -> Arc<StoredRecord> {
        self.record
    }
}

/// Decoded pack records, least recently used first out, within a bound of
/// their retained bytes.
#[derive(Debug, Default)]
struct DecodedCache {
    /// Each record, its charge and its last use.
    records: BTreeMap<Digest, (Arc<StoredRecord>, usize, u64)>,
    /// Keys by last use.
    uses: BTreeMap<u64, Digest>,
    bytes: usize,
    bound: usize,
    clock: u64,
    /// Every record decoded, while a machine still holds it with a page it
    /// keeps: the cache holds a small part of what the machines hold
    /// between them, and a record one of them still has need not be
    /// decoded again for another.
    held: BTreeMap<Digest, Weak<StoredRecord>>,
    /// The size of `held` at which the records nobody holds leave it.
    held_sweep: usize,
}

/// The fewest records [`DecodedCache::held`] sweeps at.
const HELD_SWEEP: usize = 1024;

impl DecodedCache {
    fn new(bound: usize) -> Self {
        Self {
            bound,
            ..Self::default()
        }
    }

    fn get(&mut self, key: &Digest) -> Option<Arc<StoredRecord>> {
        let Some((record, _, used)) = self.records.get_mut(key) else {
            return self.held.get(key)?.upgrade();
        };
        self.clock += 1;
        self.uses.remove(used);
        *used = self.clock;
        self.uses.insert(self.clock, *key);
        Some(Arc::clone(record))
    }

    /// Note that `record` was handed out, for [`Self::get`] to find while
    /// anyone holds it.
    fn hold(&mut self, record: &Arc<StoredRecord>) {
        if self.held.len() >= self.held_sweep.max(HELD_SWEEP) {
            self.held.retain(|_, held| held.strong_count() != 0);
            self.held_sweep = 2 * self.held.len();
        }
        self.held.insert(record.key, Arc::downgrade(record));
    }

    /// Keep `record`, letting the least recently used go until it fits:
    /// those go back to the caller, to drop outside the store's lock.
    fn insert(&mut self, record: &Arc<StoredRecord>, bytes: usize) -> Vec<Arc<StoredRecord>> {
        let mut evicted = Vec::new();
        self.hold(record);
        if bytes > self.bound / 8 || self.records.contains_key(&record.key) {
            return evicted;
        }
        while self.bytes + bytes > self.bound {
            let Some((_, key)) = self.uses.pop_first() else {
                break;
            };
            if let Some((old, charge, _)) = self.records.remove(&key) {
                self.bytes -= charge;
                evicted.push(old);
            }
        }
        self.clock += 1;
        self.records
            .insert(record.key, (Arc::clone(record), bytes, self.clock));
        self.uses.insert(self.clock, record.key);
        self.bytes += bytes;
        evicted
    }

    /// Let records go until the cache fits `bound`.
    fn shrink(&mut self, bound: usize) -> Vec<Arc<StoredRecord>> {
        self.bound = bound;
        let mut evicted = Vec::new();
        while self.bytes > bound {
            let Some((_, key)) = self.uses.pop_first() else {
                break;
            };
            if let Some((old, charge, _)) = self.records.remove(&key) {
                self.bytes -= charge;
                evicted.push(old);
            }
        }
        evicted
    }
}

/// One boot's store over the pack it read.
#[derive(Debug)]
pub struct Store<'a> {
    translator: Digest,
    /// The file's segments, oldest first: at most [`MAX_SEGMENTS`].
    segments: Vec<Pack<'a>>,
    filters: Arc<KeyFilters>,
    /// The keys looked up, and whether the segments held a valid record for
    /// each. The records themselves are the engine's to keep: decoded, they
    /// take several times the pack's bytes.
    loaded: BTreeMap<Digest, bool>,
    /// This boot's uses of each page key: its records handed out (decoded
    /// or from the cache) and its blocks offered. A full merge that must
    /// leave records out keeps those used most ([`Merge::by_use`]).
    uses: BTreeMap<Digest, Use>,
    decoded: DecodedCache,
    decode_scratch: DecodeScratch,
    /// Entries added during the run, by page key, and their retained bytes.
    added: BTreeMap<Digest, Record>,
    added_bytes: usize,
    added_bound: usize,
    /// The entries new segments took this boot, which offers again do not
    /// add, and those the bound left out, counted once each.
    taken: SeenFilter,
    refused: SeenFilter,
    /// The heap is short: offers are counted, not kept ([`Self::shed`]).
    shedding: bool,
    /// The blocks offered, by page key, PC and context, for the other
    /// machines sharing the store, oldest first. The bound charges kept
    /// source/template buffers, Arc payloads, index entries and FIFO capacity;
    /// allocator node slack/bookkeeping and shared Preparation are excluded.
    shared: BTreeMap<SharedKey, SharedEntry>,
    shared_order: VecDeque<SharedKey>,
    shared_bytes: usize,
    shared_bound: usize,
    misses: MissTables,
    /// Keys a lookup found in a segment whose records stay in the file,
    /// oldest first, for the reader to take ([`Self::take_wanted`]), and
    /// every key asked for and not read yet.
    wanted: VecDeque<Digest>,
    asked: alloc::collections::BTreeSet<Digest>,
    /// The records of those segments to read ahead of a lookup, most used
    /// first ([`Self::take_ahead`]): segment and index entry, and how far
    /// the reader went.
    ahead: Vec<(u8, u32)>,
    ahead_at: usize,
    /// A merge this boot replaced the segments left in the file: the space
    /// they took may hold other bytes now, so none is read from them.
    retired: bool,
    pub counters: Counters,
}

/// The pieces of one page key's record in a store's segments, oldest
/// first: a resident segment's in memory, the others' to read from the
/// file at the segment's offset and length ([`Store::read_stored`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Wanted {
    pub key: Digest,
    pub pieces: [Option<(u64, u64)>; MAX_SEGMENTS],
}

impl Wanted {
    /// The bytes to read from the file.
    pub fn bytes(&self) -> u64 {
        self.pieces.iter().flatten().map(|&(_, bytes)| bytes).sum()
    }
}

/// The most keys a store holds asked for and not read: lookups past it
/// compile without asking.
pub const MAX_WANTED: usize = 4096;

/// The most read-ahead entries one [`Store::take_ahead`] looks at, so a
/// call stays bounded while it skips the records other parts own.
pub const AHEAD_SCAN: usize = 256;

/// What a store keeps beyond its pack, in bytes ([`Store::set_bounds`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bounds {
    /// Entries added during the run.
    pub added: usize,
    /// Decoded pack records.
    pub decoded: usize,
}

impl Default for Bounds {
    fn default() -> Self {
        Self {
            added: ADDED_BYTES,
            decoded: DECODED_BYTES,
        }
    }
}

impl Bounds {
    /// Each of `parts` stores' share of these bounds.
    pub fn share(self, parts: usize) -> Self {
        let parts = parts.max(1);
        Self {
            added: self.added / parts,
            decoded: self.decoded / parts,
        }
    }
}

impl<'a> Store<'a> {
    /// A store of `translator` over the pack `pack` holds, if any; `None`
    /// or a damaged pack (whose error comes back too) starts it empty.
    pub fn new(translator: Digest, pack: Option<&'a [u8]>) -> (Self, Option<PackError>) {
        let parsed = pack
            .map(|bytes| Pack::parse(bytes, &translator))
            .transpose();
        match parsed {
            Ok(pack) => (Self::over(translator, pack.into_iter().collect()), None),
            Err(error) => (Self::over(translator, Vec::new()), Some(error)),
        }
    }

    /// A store of `translator` over `segments`, oldest first and already
    /// parsed for it, at most [`MAX_SEGMENTS`] (more are not read): the
    /// store, or one part of a store split by page key.
    pub fn over(translator: Digest, mut segments: Vec<Pack<'a>>) -> Self {
        segments.truncate(MAX_SEGMENTS);
        let filters = KeyFilters::new(&segments);
        Self::filtered(translator, segments, filters)
    }

    /// [`Self::over`] with the segments' filters already built, to share
    /// between the parts of a store.
    pub fn filtered(
        translator: Digest,
        mut segments: Vec<Pack<'a>>,
        filters: Arc<KeyFilters>,
    ) -> Self {
        segments.truncate(MAX_SEGMENTS);
        let bounds = Bounds::default();
        Self {
            translator,
            segments,
            filters,
            loaded: BTreeMap::new(),
            uses: BTreeMap::new(),
            decoded: DecodedCache::new(bounds.decoded),
            decode_scratch: DecodeScratch::default(),
            added: BTreeMap::new(),
            added_bytes: 0,
            added_bound: bounds.added,
            shared: BTreeMap::new(),
            shared_order: VecDeque::new(),
            shared_bytes: 0,
            shared_bound: 0,
            misses: MissTables::default(),
            taken: SeenFilter::default(),
            refused: SeenFilter::default(),
            shedding: false,
            wanted: VecDeque::new(),
            asked: alloc::collections::BTreeSet::new(),
            ahead: Vec::new(),
            ahead_at: 0,
            retired: false,
            counters: Counters::default(),
        }
        .with_ahead()
    }

    /// The order of [`Self::take_ahead`]: the records of the segments left
    /// in the file, those a boot used most first.
    fn with_ahead(mut self) -> Self {
        let mut ahead: Vec<(u8, u8, u32)> = Vec::new();
        for (segment, pack) in self.segments.iter().enumerate() {
            if pack.resident {
                continue;
            }
            ahead.extend(
                (0..pack.count)
                    .map(|index| (entry_uses(pack.bytes, index), segment as u8, index as u32)),
            );
        }
        ahead.sort_unstable_by(|one, other| other.0.cmp(&one.0).then(one.1.cmp(&other.1)));
        self.ahead = ahead
            .into_iter()
            .map(|(_, segment, index)| (segment, index))
            .collect();
        self
    }

    /// Whether a segment whose records stay in the file holds `key`.
    fn on_disk(&self, key: &Digest) -> bool {
        self.segments.iter().enumerate().any(|(index, pack)| {
            !pack.resident && self.filters.may_hold(index, key) && pack.position(key).is_some()
        })
    }

    /// A lookup of `key` found nothing in memory: when a segment left in
    /// the file holds it, ask the reader for it (once, within
    /// [`MAX_WANTED`]) and say so; the caller compiles meanwhile.
    pub fn want(&mut self, key: &Digest) -> bool {
        // A record read before and let go by the decoded cache is read again.
        if self.retired || self.loaded.get(key) == Some(&false) || !self.on_disk(key) {
            return false;
        }
        self.counters.awaited += 1;
        if self.asked.len() < MAX_WANTED && self.asked.insert(*key) {
            self.wanted.push_back(*key);
        }
        true
    }

    /// Where `key`'s record lies in each segment that holds it.
    fn pieces(&self, key: &Digest) -> [Option<(u64, u64)>; MAX_SEGMENTS] {
        let mut pieces = [None; MAX_SEGMENTS];
        for (index, (piece, pack)) in pieces.iter_mut().zip(&self.segments).enumerate() {
            if self.filters.may_hold(index, key) {
                *piece = pack.locate(key);
            }
        }
        pieces
    }

    /// Forget the oldest keys lookups asked for past the newest `keep`: on a
    /// disk whose reads seek, their reads would end long after their blocks
    /// compiled. A later lookup may ask for them again. The keys forgotten.
    pub fn trim_wanted(&mut self, keep: usize) -> usize {
        let mut dropped = 0;
        while self.wanted.len() > keep {
            let Some(key) = self.wanted.pop_front() else {
                break;
            };
            self.asked.remove(&key);
            dropped += 1;
        }
        dropped
    }

    /// The oldest key a lookup asked for, with where its record lies.
    pub fn take_wanted(&mut self) -> Option<Wanted> {
        let key = self.wanted.pop_front()?;
        Some(Wanted {
            key,
            pieces: self.pieces(&key),
        })
    }

    /// Keep in the read-ahead order only the records `owns` (a part of a
    /// store split by page key owns a share of the keys).
    pub fn keep_ahead(&mut self, owns: impl Fn(&Digest) -> bool) {
        let segments = &self.segments;
        self.ahead.retain(|&(segment, index)| {
            owns(&entry(segments[usize::from(segment)].bytes, index as usize).0)
        });
        self.ahead.shrink_to_fit();
    }

    /// The next record to read ahead of any lookup, the most used first,
    /// skipping those looked up or asked for already. A call looks at no
    /// more than [`AHEAD_SCAN`] entries.
    pub fn take_ahead(&mut self) -> Option<Wanted> {
        if self.retired {
            return None;
        }
        for _ in 0..AHEAD_SCAN {
            let &(segment, index) = self.ahead.get(self.ahead_at)?;
            self.ahead_at += 1;
            let key = entry(self.segments[usize::from(segment)].bytes, index as usize).0;
            if let Some(wanted) = self.claim(&key) {
                return Some(wanted);
            }
        }
        None
    }

    /// `key`'s record to read ahead of a lookup, unless it was looked up
    /// or asked for already, or no segment left in the file holds it.
    pub fn claim(&mut self, key: &Digest) -> Option<Wanted> {
        if self.retired
            || self.loaded.contains_key(key)
            || !self.on_disk(key)
            || !self.asked.insert(*key)
        {
            return None;
        }
        self.counters.read_ahead += 1;
        Some(Wanted {
            key: *key,
            pieces: self.pieces(key),
        })
    }

    /// A merge replaced the segments left in the file: read none of them
    /// again, and let lookups compile what they held. Records read already
    /// stay.
    pub fn retire(&mut self) {
        self.retired = true;
        self.wanted.clear();
        self.ahead = Vec::new();
    }

    /// Whether the decoded cache has room for records read ahead of a
    /// lookup: what it holds is under half its bound, which the heap's
    /// governor sets.
    pub fn ahead_room(&self) -> bool {
        self.decoded.bytes * 2 < self.decoded.bound
    }

    /// The bytes kept in memory for the records left in the file: the
    /// read-ahead order.
    pub fn ahead_bytes(&self) -> usize {
        self.ahead.capacity() * size_of::<(u8, u32)>()
    }

    /// `wanted`'s record over `read`, the bytes read of each piece a
    /// segment left in the file holds (`None` for the others): what
    /// [`DecodedRecord::decode`] decodes.
    pub fn read_stored<'b>(
        &self,
        wanted: &Wanted,
        read: &[Option<&'b [u8]>; MAX_SEGMENTS],
    ) -> Option<Stored<'b>>
    where
        'a: 'b,
    {
        let mut stored = Stored {
            records: [None; MAX_SEGMENTS],
        };
        for (index, slot) in stored.records.iter_mut().enumerate() {
            let Some(pack) = self.segments.get(index) else {
                break;
            };
            let Some((_, bytes)) = wanted.pieces[index] else {
                continue;
            };
            *slot = Some(if pack.resident {
                pack.record(&wanted.key)?
            } else {
                let read = read[index].filter(|read| read.len() as u64 == bytes)?;
                PackedRecord {
                    bytes: read,
                    format: pack.format,
                    checked: false,
                }
            });
        }
        stored.records.iter().any(Option::is_some).then_some(stored)
    }

    /// The reader decoded `key`'s record from the file, or found it
    /// invalid (`None` when the read failed: it may be asked for again).
    /// The records the decoded cache let go come back, to drop outside the
    /// store's lock.
    pub fn finish_read(
        &mut self,
        key: &Digest,
        bytes: u64,
        decoded: Option<&Result<DecodedRecord<'_>, DecodeError>>,
    ) -> Vec<Arc<StoredRecord>> {
        self.asked.remove(key);
        // Read from space a merge may have given to other bytes.
        if self.retired {
            return Vec::new();
        }
        match decoded {
            Some(Ok(decoded))
                if decoded.record.key == *key && decoded.record.translator == self.translator =>
            {
                self.counters.read += 1;
                self.counters.read_bytes += bytes;
                self.looked_up(key, true, true);
                self.decoded.insert(&decoded.record, decoded.retained_bytes)
            }
            Some(Err(DecodeError::Invalid)) | Some(Ok(_)) => {
                self.looked_up(key, true, false);
                Vec::new()
            }
            _ => Vec::new(),
        }
    }

    /// Bound what the store keeps beyond its pack. Records the decoded
    /// cache lets go come back, to drop outside a lock.
    pub fn set_bounds(&mut self, bounds: Bounds) -> Vec<Arc<StoredRecord>> {
        self.added_bound = bounds.added;
        self.decoded.shrink(bounds.decoded)
    }

    /// Take `other`'s run (another part of one store over the same pack):
    /// what it looked up, its added entries and its counts. Its shared
    /// blocks are dropped.
    pub fn absorb(&mut self, other: Self) {
        for (key, valid) in other.loaded {
            self.loaded.entry(key).or_insert(valid);
        }
        for (key, uses) in other.uses {
            match self.uses.entry(key) {
                alloc::collections::btree_map::Entry::Vacant(slot) => {
                    slot.insert(uses);
                }
                alloc::collections::btree_map::Entry::Occupied(mut slot) => {
                    slot.get_mut().join(uses)
                }
            }
        }
        for (key, record) in other.added {
            match self.added.entry(key) {
                alloc::collections::btree_map::Entry::Vacant(slot) => {
                    slot.insert(record);
                }
                alloc::collections::btree_map::Entry::Occupied(mut slot) => {
                    slot.get_mut().merge(record);
                }
            }
        }
        self.added_bytes += other.added_bytes;
        self.taken.absorb(other.taken);
        self.refused.absorb(other.refused);
        self.counters.add(&other.counters);
    }

    /// Keep offered blocks and their buffers, up to `bound` bytes, for
    /// [`Translations::prepared`]: for machines that share the store. Full,
    /// it forgets the oldest until a new one fits.
    pub fn share(&mut self, bound: usize) {
        self.shared_bound = bound;
        while self.shared_usage() > bound {
            if self.evict_shared().is_none() {
                self.shared_order = VecDeque::new();
                break;
            }
        }
    }

    fn shared_usage(&self) -> usize {
        self.shared_bytes
            .saturating_add(self.shared_order.capacity() * size_of::<SharedKey>())
    }

    fn evict_shared(&mut self) -> Option<SharedEntry> {
        while let Some(oldest) = self.shared_order.pop_front() {
            if let Some(evicted) = self.shared.remove(&oldest) {
                self.shared_bytes -= evicted.bytes;
                self.misses.evicted(oldest);
                return Some(evicted);
            }
        }
        None
    }

    /// Admit a preallocated shared block. The producer can drop a displaced
    /// owner after releasing the lock. The producer still owns a declined
    /// block, so declining it does not drop its buffers while locked.
    pub fn accept_shared(&mut self, staged: &mut StagedOffer) -> Option<Arc<SharedTranslation>> {
        use alloc::collections::btree_map::Entry;

        let key = staged.key;
        self.misses.offered(key);
        let bytes = staged.entry.as_ref()?.bytes;
        if bytes > self.shared_bound {
            return None;
        }
        let mut deferred = None;
        loop {
            let usage = self.shared_usage();
            match self.shared.entry(key) {
                Entry::Occupied(mut slot)
                    if usage.saturating_sub(slot.get().bytes).saturating_add(bytes)
                        <= self.shared_bound =>
                {
                    self.shared_bytes = self.shared_bytes - slot.get().bytes + bytes;
                    let old = slot.insert(staged.entry.take().expect("staged entry"));
                    retire(old, &mut deferred);
                    return deferred;
                }
                Entry::Vacant(slot)
                    if usage.saturating_add(bytes) <= self.shared_bound
                        && self.shared_order.try_reserve(1).is_ok() =>
                {
                    // A FIFO allocation can itself exceed the budget.
                    let fifo = self.shared_order.capacity() * size_of::<SharedKey>();
                    if self.shared_bytes.saturating_add(fifo).saturating_add(bytes)
                        <= self.shared_bound
                    {
                        slot.insert(staged.entry.take().expect("staged entry"));
                        self.shared_order.push_back(key);
                        self.shared_bytes += bytes;
                        return deferred;
                    }
                }
                _ => {}
            }
            let Some(evicted) = self.evict_shared() else {
                self.shared_order = VecDeque::new();
                return deferred;
            };
            retire(evicted, &mut deferred);
        }
    }

    /// Whether the shared blocks hold one at `pc` of the page `key` in
    /// `context`, without counting a lookup.
    pub fn holds_shared(&self, key: &Digest, pc: u64, context: &Digest) -> bool {
        self.shared.contains_key(&shared_key(key, pc, context))
    }

    /// Keep no offer while `shed`, the heap being short, or keep them again:
    /// the writer sheds the store's work before anything else's.
    pub fn shed(&mut self, shed: bool) {
        self.shedding = shed;
    }

    /// Whether offers are shed.
    pub fn shedding(&self) -> bool {
        self.shedding
    }

    /// Count an offer not kept while shedding, whose stored form was never
    /// made.
    pub fn count_shed(&mut self, offer: &Offer) {
        self.misses
            .offered(shared_key(&offer.key(), offer.pc(), &offer.context()));
        self.counters.shed += 1;
    }

    /// Keep `entry`, the stored form of `offer` ([`Offer::entry`], made
    /// before the store's lock is taken), for the next pack while the added
    /// bound has room.
    pub fn keep(&mut self, offer: &Offer, entry: Option<Entry>) {
        let key = offer.key();
        self.used(&key);
        self.misses
            .offered(shared_key(&key, offer.pc(), &offer.context()));
        if self.shedding {
            self.counters.shed += 1;
            return;
        }
        let Some(entry) = entry else {
            self.counters.unstorable += 1;
            return;
        };
        self.add(key, entry);
    }

    /// Keep `entry`, this translator's stored form of a block of the
    /// previous translator's pack, of the page `key` names
    /// (the engine's `refill.rs`), as an offer's, within half
    /// the bound of the entries added: the blocks this boot compiles keep
    /// the rest. Unused yet, a merge keeps it only while boots use it.
    pub fn refilled(&mut self, key: Digest, entry: Entry) {
        if self.shedding || self.added_bytes + entry.retained_bytes() > self.added_bound / 2 {
            return;
        }
        self.add(key, entry);
    }

    /// Add `entry` to the entries of `key` added this boot, within their
    /// bound.
    fn add(&mut self, key: Digest, entry: Entry) {
        if self
            .added
            .get(&key)
            .is_some_and(|record| record.find(entry.offset, &entry.context).is_some())
        {
            return;
        }
        if self.taken.contains(&key, entry.offset, &entry.context) {
            self.counters.retaken += 1;
            return;
        }
        let bytes = entry.retained_bytes();
        if self.added_bytes + bytes > self.added_bound {
            self.counters.unkept += 1;
            if !self.refused.insert(&key, entry.offset, &entry.context) {
                self.counters.unkept_entries += 1;
            }
            return;
        }
        self.added_bytes += bytes;
        self.counters.added += 1;
        self.added
            .entry(key)
            .or_insert_with(|| Record::new(self.translator, key))
            .insert(entry);
    }

    /// The translator whose records the store keeps.
    pub fn translator(&self) -> &Digest {
        &self.translator
    }

    /// A record already decoded and checked against this pack.
    pub fn cached_record(&mut self, key: &Digest) -> Option<Arc<StoredRecord>> {
        let record = self.decoded.get(key)?;
        self.counters.cached += 1;
        self.counters.repeated += 1;
        self.used(key);
        Some(record)
    }

    /// Count a use of `key` this boot.
    fn used(&mut self, key: &Digest) {
        self.uses
            .entry(*key)
            .and_modify(|uses| uses.count = uses.count.saturating_add(1))
            .or_insert_with(Use::first);
    }

    /// This boot's uses of each page key so far ([`Merge::by_use`]).
    pub fn uses(&self) -> &BTreeMap<Digest, Use> {
        &self.uses
    }

    /// The pack's slot for `key`, unless an earlier lookup found it invalid.
    /// Its immutable bytes may be decoded after releasing the store lock.
    pub fn stored(&self, key: &Digest) -> Option<Stored<'a>> {
        let loaded = self.loaded.get(key).copied();
        if loaded == Some(false) {
            return None;
        }
        let mut stored = Stored {
            records: [None; MAX_SEGMENTS],
        };
        let mut found = false;
        for (index, (slot, pack)) in stored.records.iter_mut().zip(&self.segments).enumerate() {
            if !self.filters.may_hold(index, key) {
                continue;
            }
            // A piece still in the file: the record is read whole
            // ([`Self::want`]), never decoded without it.
            if !pack.resident {
                if !self.retired && pack.position(key).is_some() {
                    return None;
                }
                continue;
            }
            *slot = pack.record(key).map(|mut record| {
                // A first lookup that decoded the slot checked its bytes whole.
                record.checked = loaded == Some(true);
                record
            });
            found |= slot.is_some();
        }
        found.then_some(stored)
    }

    pub fn take_decode_scratch(&mut self, stored: Stored<'a>) -> DecodeScratch {
        if stored.is_compressed() {
            core::mem::take(&mut self.decode_scratch)
        } else {
            DecodeScratch::default()
        }
    }

    /// A transient capacity failure can be retried. Only record a result
    /// for the exact stored slot we gave the decoder. The records the cache
    /// let go come back, for the caller to drop with the scratch outside
    /// the store lock.
    pub fn finish_decode(
        &mut self,
        key: &Digest,
        stored: Stored<'a>,
        decoded: &Result<DecodedRecord<'a>, DecodeError>,
        scratch: &mut DecodeScratch,
    ) -> Vec<Arc<StoredRecord>> {
        self.decode_scratch.retain(scratch);
        if !self
            .stored(key)
            .is_some_and(|current| current.same_storage(&stored))
        {
            return Vec::new();
        }
        match decoded {
            Ok(decoded)
                if decoded.stored.same_storage(&stored)
                    && decoded.record.key == *key
                    && decoded.record.translator == self.translator =>
            {
                self.counters.decoded += 1;
                self.looked_up(key, true, true);
                self.used(key);
                self.decoded.insert(&decoded.record, decoded.retained_bytes)
            }
            Err(DecodeError::Invalid) => {
                self.looked_up(key, true, false);
                Vec::new()
            }
            _ => Vec::new(),
        }
    }

    /// Count `key`'s lookup: on the first, whether the pack held bytes for
    /// it, and whether they decoded.
    pub fn looked_up(&mut self, key: &Digest, found: bool, valid: bool) {
        let alloc::collections::btree_map::Entry::Vacant(slot) = self.loaded.entry(*key) else {
            self.counters.repeated += 1;
            return;
        };
        match (found, valid) {
            (true, true) => self.counters.loaded += 1,
            (true, false) => self.counters.invalid += 1,
            (false, _) => self.counters.absent += 1,
        }
        slot.insert(valid);
    }

    /// The segments read, oldest first.
    pub fn segments(&self) -> &[Pack<'a>] {
        &self.segments
    }

    /// Take the entries added since the last take, for a new segment
    /// ([`Delta`]): the added bound is free again.
    pub fn take_added(&mut self) -> BTreeMap<Digest, Record> {
        self.added_bytes = 0;
        for (key, record) in &self.added {
            for entry in record.entries() {
                self.taken.insert(key, entry.offset, &entry.context);
            }
        }
        core::mem::take(&mut self.added)
    }

    /// The bytes of the entries added since the last take.
    pub fn added_bytes(&self) -> usize {
        self.added_bytes
    }

    /// Whether the entries added since the last take reached half the
    /// bound past which new ones are not kept: time to take them.
    pub fn added_pressed(&self) -> bool {
        self.added_bytes != 0 && self.added_bytes.saturating_mul(2) >= self.added_bound
    }

    /// The page keys looked up whose records were found valid: with the
    /// keys of the entries it added ([`Self::take_added`]), this boot's used
    /// keys ([`UsedKeys`]).
    pub fn used_keys(&self) -> impl Iterator<Item = &Digest> + '_ {
        self.loaded
            .iter()
            .filter(|(_, valid)| **valid)
            .map(|(key, _)| key)
    }
}

impl Translations for Store<'_> {
    fn record(&mut self, key: &Digest) -> Option<Arc<StoredRecord>> {
        match self.lookup(key) {
            Looked::Found(record) => Some(record),
            Looked::Absent | Looked::Awaited => None,
        }
    }

    fn lookup(&mut self, key: &Digest) -> Looked {
        if let Some(record) = self.cached_record(key) {
            return Looked::Found(record);
        }
        let Some(stored) = self.stored(key) else {
            if self.want(key) {
                return Looked::Awaited;
            }
            self.looked_up(key, false, false);
            return Looked::Absent;
        };
        let mut scratch = self.take_decode_scratch(stored);
        let decoded = DecodedRecord::decode(stored, &self.translator, key, &mut scratch);
        drop(self.finish_decode(key, stored, &decoded, &mut scratch));
        decoded.ok().map_or(Looked::Absent, |decoded| {
            Looked::Found(decoded.into_record())
        })
    }

    fn prepared(
        &mut self,
        key: &Digest,
        pc: u64,
        context: &Digest,
    ) -> Option<Arc<SharedTranslation>> {
        let shared = shared_key(key, pc, context);
        match self.shared.get(&shared) {
            Some(entry) => {
                self.counters.shared_hits += 1;
                Some(Arc::clone(&entry.translation))
            }
            None => {
                self.misses.miss(shared, &mut self.counters);
                None
            }
        }
    }

    fn offer(&mut self, offer: Offer) {
        let entry = offer.entry();
        self.keep(&offer, entry);
        if let Some(mut staged) = StagedOffer::new(offer, self.shared_bound) {
            drop(self.accept_shared(&mut staged));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::machine::test_preparation;
    use crate::translations::{PAGE_BYTES, translator_id};

    /// An offer of `add x0, x0, #1; ret` at `pc`, in a page whose bytes
    /// hash to `key`.
    fn offer(pc: u64, key: Digest) -> Offer {
        let preparation = Arc::new(test_preparation());
        let page = pc & !(PAGE_BYTES as u64 - 1);
        let mut input: Vec<u8> = [0x9100_0400u32, 0xd65f_03c0]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        preparation.admit(&mut input);
        Offer::compile(key, page, pc, preparation.context(page), input, preparation).unwrap()
    }

    #[test]
    fn a_shared_block_is_held_without_a_lookup_counted() {
        let key = Digest([7; 32]);
        let mut store = Store::new(translator_id(), None).0;
        store.share(1 << 20);
        let offer = offer(0x1000, key);
        let context = offer.context();
        assert!(!store.holds_shared(&key, 0x1000, &context));
        store.offer(offer);
        assert!(store.holds_shared(&key, 0x1000, &context));
        assert!(!store.holds_shared(&key, 0x1004, &context));
        assert_eq!(store.counters.shared_hits, 0);
        assert!(store.prepared(&key, 0x1000, &context).is_some());
        assert_eq!(store.counters.shared_hits, 1);
    }

    #[test]
    fn decoded_pack_records_share_an_owner_and_the_least_recently_used_goes_first() {
        let translator = translator_id();
        let keys: Vec<Digest> = (0u8..9)
            .map(|index| {
                // Keys alike but for their last byte.
                let mut key = Digest([1; 32]);
                key.0[31] = index;
                key
            })
            .collect();
        let bytes = Layout::new(
            &translator,
            keys.iter()
                .map(|&key| (key, Piece::New(Record::new(translator, key).encode())))
                .collect(),
        )
        .to_vec();
        let mut store = Store::new(translator, Some(&bytes)).0;
        let one = Record::new(translator, keys[0]).retained_bytes();
        drop(store.set_bounds(Bounds {
            decoded: 8 * one,
            ..Bounds::default()
        }));
        assert!(!store.stored(&keys[0]).unwrap().checked());
        let held: Vec<Arc<StoredRecord>> = keys[..8]
            .iter()
            .map(|key| store.record(key).unwrap())
            .collect();
        for (key, held) in keys.iter().zip(&held) {
            assert!(Arc::ptr_eq(held, &store.record(key).unwrap()));
            assert_eq!(held.key, *key);
        }
        // The first is used again; the ninth then evicts the second, which
        // its holder still shares.
        assert!(Arc::ptr_eq(&held[0], &store.record(&keys[0]).unwrap()));
        assert_eq!(store.record(&keys[8]).unwrap().key, keys[8]);
        assert!(Arc::ptr_eq(&held[0], &store.record(&keys[0]).unwrap()));
        assert!(Arc::ptr_eq(&held[1], &store.record(&keys[1]).unwrap()));
        assert_eq!(store.counters.loaded, 9);
        assert_eq!(store.counters.decoded, 9);
        assert_eq!(store.counters.repeated, 11);
        // Once nobody holds it, it is decoded again, from bytes already
        // checked whole.
        assert!(store.stored(&keys[1]).unwrap().checked());
        drop(held);
        assert_eq!(store.record(&keys[1]).unwrap().key, keys[1]);
        assert_eq!(store.counters.decoded, 10);
    }

    #[test]
    fn records_nobody_holds_leave_the_held_index() {
        let translator = translator_id();
        let mut cache = DecodedCache::new(0);
        let key = |index: usize| {
            let mut key = Digest([2; 32]);
            key.0[..8].copy_from_slice(&(index as u64).to_le_bytes());
            key
        };
        let kept = Arc::new(StoredRecord::new(translator, key(0)));
        assert!(cache.insert(&kept, 1).is_empty());
        for index in 1..=2 * HELD_SWEEP {
            assert!(
                cache
                    .insert(&Arc::new(StoredRecord::new(translator, key(index))), 1)
                    .is_empty()
            );
        }
        // Nothing fits the cache itself; only the record still held is found.
        assert!(cache.records.is_empty());
        assert!(cache.held.len() <= HELD_SWEEP + 1);
        assert!(Arc::ptr_eq(&kept, &cache.get(&key(0)).unwrap()));
        assert!(cache.get(&key(1)).is_none());
    }

    #[test]
    fn decoded_cache_declines_oversized_and_foreign_records() {
        let translator = translator_id();
        let key = Digest::of(b"page");
        let mut large = Record::new(translator, key);
        let mut entry = offer(0x1000, key).entry().unwrap();
        entry.code.resize(DECODED_BYTES / 8, 0);
        large.insert(entry);
        let bytes = Layout::new(&translator, vec![(key, Piece::New(large.encode()))]).to_vec();
        let mut store = Store::new(translator, Some(&bytes)).0;
        assert!(store.record(&key).is_some());
        assert!(store.cached_record(&key).is_none());

        let bytes = Layout::new(
            &translator,
            vec![(key, Piece::New(Record::new(translator, key).encode()))],
        )
        .to_vec();
        let foreign = bytes.clone();
        let mut store = Store::new(translator, Some(&bytes)).0;
        let foreign = Stored::one(
            Pack::parse(&foreign, &translator)
                .unwrap()
                .record(&key)
                .unwrap(),
        );
        let mut scratch = DecodeScratch::default();
        let decoded = DecodedRecord::decode(foreign, &translator, &key, &mut scratch);
        assert!(decoded.is_ok());
        assert!(
            store
                .finish_decode(&key, foreign, &decoded, &mut scratch)
                .is_empty()
        );
        assert!(store.cached_record(&key).is_none());
    }

    #[test]
    fn mixed_records_bind_their_original_index_slot_and_survive_rewrite() {
        let translator = translator_id();
        let first = Digest::of(b"compressed");
        let second = Digest::of(b"raw");
        let mut compressed_record = Record::new(translator, first);
        let mut entry = offer(0x1000, first).entry().unwrap();
        entry.code.resize(4096, 0);
        compressed_record.insert(entry);
        let raw = compressed_record.encode();
        let compressed = codec::test_envelope(&raw);
        let records = || {
            vec![
                (first, Piece::New(compressed.clone())),
                (second, Piece::New(Record::new(translator, second).encode())),
            ]
        };
        let bytes = Layout::with_format(&translator, records(), PackFormat::Mixed).to_vec();
        let foreign = bytes.clone();
        let mut store = Store::new(translator, Some(&bytes)).0;
        let mut scratch = DecodeScratch::default();
        let stored = store.stored(&first).unwrap();
        store.finish_decode(&first, stored, &Err(DecodeError::Capacity), &mut scratch);
        assert_eq!(store.counters.invalid, 0);
        assert!(store.stored(&first).is_some());
        let foreign = Stored::one(
            Pack::parse(&foreign, &translator)
                .unwrap()
                .record(&first)
                .unwrap(),
        );
        let decoded = DecodedRecord::decode(foreign, &translator, &first, &mut scratch);
        assert!(decoded.is_ok());
        assert!(
            store
                .finish_decode(&first, stored, &decoded, &mut scratch)
                .is_empty()
        );
        assert_eq!(store.counters.loaded, 0);
        assert!(store.cached_record(&first).is_none());

        let held = store.record(&first).unwrap();
        assert_eq!(held.record().unwrap(), compressed_record);
        assert!(Arc::ptr_eq(&held, &store.record(&first).unwrap()));
        assert_eq!(store.record(&second).unwrap().key, second);
        let output = written(&mut store, u64::MAX);
        let pack = Pack::parse(&output, &translator).unwrap();
        assert_eq!(pack.format, PackFormat::Mixed);
        assert_eq!(pack.get(&first), Some(compressed.as_slice()));

        let old = Layout::new(&translator, records()).to_vec();
        let mut store = Store::new(translator, Some(&old)).0;
        assert!(store.record(&first).is_none());
        assert_eq!(store.counters.invalid, 1);
        assert_eq!(store.record(&second).unwrap().key, second);
    }

    #[test]
    fn shared_cache_budget_includes_source_and_recovery_owners() {
        let translator = Digest::of(b"translator");
        let key = Digest::of(b"page");
        let first = offer(0x1000, key);
        let context = first.context();
        let code = first.block().native.x86_64_code.len();
        let (mut store, _) = Store::new(translator, None);
        // The native bytes fit, but their source/template/collection owners do not.
        store.share(code);
        store.offer(first);
        assert!(store.prepared(&key, 0x1000, &context).is_none());
        assert!(store.shared_order.is_empty());
    }

    #[test]
    fn shared_cache_budget_charges_emitted_recovery_buffers() {
        let key = Digest::of(b"page");
        let preparation = Arc::new(test_preparation());
        let context = preparation.context(0x1000);
        let mut input: Vec<u8> = core::iter::repeat_n(0xf900_0020u32, 16)
            .chain(core::iter::once(0xd65f_03c0))
            .flat_map(u32::to_le_bytes)
            .collect();
        preparation.admit(&mut input);
        let offered = Offer::compile(key, 0x1000, 0x1000, context, input, preparation).unwrap();
        let code = offered.block().native.x86_64_code.len();
        assert!(!offered.block().fault_recovery.is_empty());
        assert!(
            offered.block().fault_recovery.len()
                * size_of::<stela_compiler::block::FaultRecovery>()
                > code
        );
        let (mut store, _) = Store::new(Digest::of(b"translator"), None);
        store.share(2 * code);
        store.offer(offered);
        assert!(store.prepared(&key, 0x1000, &context).is_none());
        assert!(store.shared_order.is_empty());
    }

    #[test]
    fn an_offer_keeps_no_reserved_source_capacity_against_the_shared_bound() {
        let key = Digest::of(b"page");
        let first = offer(0x1000, key);
        let context = first.context();
        let preparation = Arc::clone(first.preparation());
        let mut input = Vec::with_capacity(2 << 20);
        input.extend_from_slice(first.input());
        let replacement = Offer::compile(key, 0x1000, 0x1000, context, input, preparation).unwrap();
        let (mut store, _) = Store::new(Digest::of(b"translator"), None);
        store.share(1 << 20);
        store.offer(replacement);
        assert!(store.prepared(&key, 0x1000, &context).is_some());
        assert_eq!(store.shared_order.len(), 1);
    }

    #[test]
    fn reducing_shared_cache_budget_releases_kept_entries_and_fifo_capacity() {
        let key = Digest::of(b"page");
        let first = offer(0x1000, key);
        let context = first.context();
        let (mut store, _) = Store::new(Digest::of(b"translator"), None);
        store.share(usize::MAX);
        store.offer(first);
        assert!(store.prepared(&key, 0x1000, &context).is_some());
        store.share(0);
        assert!(store.prepared(&key, 0x1000, &context).is_none());
        assert_eq!(store.shared_order.capacity(), 0);
    }

    #[test]
    fn staged_admission_defers_eviction_and_keeps_a_declined_owner() {
        let translator = Digest::of(b"translator");
        let key = Digest::of(b"page");
        let first = offer(0x1000, key);
        let second = offer(0x1008, key);
        let context = first.context();
        let charge = first.retained_bytes() + size_of::<(SharedKey, SharedEntry)>();
        let mut queue = VecDeque::<SharedKey>::new();
        queue.try_reserve(1).unwrap();
        let bound = charge + queue.capacity() * size_of::<SharedKey>();

        let (mut store, _) = Store::new(translator, None);
        store.share(bound);
        let mut staged = StagedOffer::new(first, bound).unwrap();
        assert!(store.accept_shared(&mut staged).is_none());
        assert!(staged.entry.is_none());
        let old = Arc::downgrade(&store.prepared(&key, 0x1000, &context).unwrap());

        let mut staged = StagedOffer::new(second, bound).unwrap();
        let retired = store.accept_shared(&mut staged).unwrap();
        assert!(store.prepared(&key, 0x1000, &context).is_none());
        assert!(Arc::ptr_eq(&retired, &old.upgrade().unwrap()));
        assert!(store.prepared(&key, 0x1008, &context).is_some());
        assert!(store.shared_usage() <= bound);
        assert_eq!(store.counters.added, 0);
        drop(retired);
        assert!(old.upgrade().is_none());

        let mut staged = StagedOffer::new(offer(0x1000, key), bound).unwrap();
        store.share(0);
        assert!(store.accept_shared(&mut staged).is_none());
        assert!(staged.entry.is_some());
        assert!(store.prepared(&key, 0x1000, &context).is_none());
    }

    #[test]
    fn a_shared_store_hands_offered_blocks_to_other_machines_and_forgets_the_oldest_when_full() {
        let translator = Digest::of(b"translator");
        let key = Digest::of(b"page");
        let (first, second) = (offer(0x1000, key), offer(0x1008, key));
        let code = first.block().native.x86_64_code.len();
        let bytes = first.clone().retained_bytes() + size_of::<(SharedKey, SharedEntry)>();
        let mut queue = VecDeque::<SharedKey>::new();
        queue.try_reserve(1).unwrap();
        let fifo = queue.capacity() * size_of::<SharedKey>();
        let (mut store, _) = Store::new(translator, None);
        store.offer(first.clone());
        // A store no machine shares keeps no native code.
        assert!(store.prepared(&key, 0x1000, &first.context()).is_none());
        store.share(bytes + bytes / 2 + fifo);
        store.offer(first.clone());
        store.offer(first.clone());
        store.offer(second.clone());
        assert!(
            store
                .prepared(&key, 0x1000, &Digest::of(b"other"))
                .is_none()
        );
        assert!(store.prepared(&key, 0x1008, &second.context()).is_some());
        assert!(store.prepared(&key, 0x1000, &first.context()).is_none());
        store.share(4 * bytes + fifo);
        store.offer(first.clone());
        assert_eq!(
            store
                .prepared(&key, 0x1000, &first.context())
                .map(|block| block.block().native.x86_64_code.clone()),
            Some(first.block().native.x86_64_code.clone())
        );
        // Offered again, a block keeps its one place and its bytes.
        store.offer(first.clone());
        assert_eq!(store.shared_bytes, 2 * bytes);
        // Full, the oldest goes first: the second, offered before the first
        // came back.
        let third = offer(0x1010, key);
        let fourth = offer(0x1018, key);
        store.offer(third.clone());
        store.offer(fourth.clone());
        let preparation = Arc::new(test_preparation());
        let pc = 0x1020;
        let page = pc & !(PAGE_BYTES as u64 - 1);
        let fifth = Offer::compile(
            key,
            page,
            pc,
            preparation.context(page),
            0xd65f03c0_u32.to_le_bytes().to_vec(),
            preparation,
        )
        .unwrap();
        assert!(
            fifth.block().native.x86_64_code.len() < code,
            "a genuine shorter emitted block"
        );
        store.offer(fifth.clone());
        assert!(store.prepared(&key, 0x1008, &second.context()).is_none());
        assert!(store.prepared(&key, 0x1000, &first.context()).is_some());
        assert!(store.prepared(&key, 0x1020, &fifth.context()).is_some());
        assert!(store.shared_usage() <= 4 * bytes + fifo);
        // Each distinct block is added once, however often offered.
        assert_eq!(store.counters.added, 5);
    }

    #[test]
    fn a_pack_indexes_its_records_by_key_and_ignores_bytes_past_its_end() {
        let translator = Digest::of(b"translator");
        let names: [&[u8]; 3] = [b"b", b"a", b"c"];
        let records: Vec<(Digest, Piece)> = names
            .iter()
            .map(|name| (Digest::of(name), Piece::New(name.repeat(10))))
            .collect();
        let layout = Layout::new(&translator, records);
        let mut bytes = layout.to_vec();
        assert_eq!(bytes.len() as u64, layout.bytes());
        let length = Pack::length(&bytes, &translator).unwrap();
        assert_eq!(length, layout.bytes());
        // A file allocated larger than its pack.
        bytes.extend([0xa5; 100]);
        let pack = Pack::parse(&bytes, &translator).unwrap();
        assert_eq!((pack.records(), pack.bytes() as u64), (3, length));
        assert_eq!(pack.get(&Digest::of(b"c")), Some(&b"cccccccccc"[..]));
        assert_eq!(pack.get(&Digest::of(b"d")), None);
        let mut keys: Vec<Digest> = names.iter().map(|name| Digest::of(name)).collect();
        keys.sort();
        assert_eq!(pack.iter().map(|(key, _)| key).collect::<Vec<_>>(), keys);
        let prefix = prefix_bytes(&bytes, &translator).unwrap();
        assert_eq!(
            prefix,
            HEADER_BYTES + 3 * INDEX_ENTRY_BYTES + CHECKSUM_BYTES
        );
        assert_eq!(
            Pack::parse(&bytes, &Digest::default()).unwrap_err(),
            PackError::Translator
        );
        assert_eq!(
            Pack::parse(&bytes[..prefix - 1], &translator).unwrap_err(),
            PackError::Truncated
        );
        assert_eq!(
            Pack::parse(&bytes[..length as usize - 1], &translator).unwrap_err(),
            PackError::Truncated
        );
        let mut damaged = bytes.clone();
        damaged[HEADER_BYTES + DIGEST_BYTES] ^= 1;
        assert_eq!(
            Pack::parse(&damaged, &translator).unwrap_err(),
            PackError::Checksum
        );
        let mut magic = bytes;
        magic[0] ^= 1;
        assert_eq!(
            Pack::parse(&magic, &translator).unwrap_err(),
            PackError::Magic
        );
    }

    impl<'a> Stored<'a> {
        /// One segment's record.
        fn one(record: PackedRecord<'a>) -> Self {
            let mut records = [None; MAX_SEGMENTS];
            records[0] = Some(record);
            Self { records }
        }

        /// Whether every segment's bytes were checked whole.
        fn checked(&self) -> bool {
            self.iter().all(|record| record.checked)
        }
    }

    /// A boot's segments with a segment of what it added, merged with
    /// the keys it used (found or added), within `bound`: what its writes
    /// leave once merged.
    fn written(store: &mut Store, bound: u64) -> Vec<u8> {
        let translator = *store.translator();
        let added = store.take_added();
        let keys: Vec<Digest> = added.keys().copied().collect();
        let delta = Delta::new(translator, added)
            .finish()
            .map(|layout| layout.to_vec());
        let used = UsedKeys::encode(&translator, store.used_keys().chain(&keys));
        let mut segments = store.segments().to_vec();
        segments.extend(
            delta
                .as_deref()
                .map(|bytes| Pack::parse(bytes, &translator).unwrap()),
        );
        let used = vec![UsedKeys::parse(&used, &translator).unwrap()];
        let segments = segments.into_iter().map(Segment::from).collect();
        Merge::new(translator, segments, used, bound)
            .finish()
            .unwrap()
            .layout
            .to_vec()
    }

    #[test]
    fn a_boot_keeps_its_blocks_in_the_next_pack_and_the_one_after_reads_them() {
        let translator = translator_id();
        let (first, second) = (Digest::of(b"first page"), Digest::of(b"second page"));
        let mut store = Store::new(translator, None).0;
        store.offer(offer(0x1_0000_4008, first));
        store.offer(offer(0x1_0000_8010, second));
        assert_eq!(store.record(&first), None);
        assert_eq!((store.counters.added, store.counters.unstorable), (2, 0));
        let bytes = written(&mut store, u64::MAX);

        let mut next = Store::new(translator, Some(&bytes)).0;
        let record = next.record(&first).expect("the first page's record");
        let context = offer(0x1_0000_4008, first).context();
        assert!(
            record
                .find(8, &context)
                .unwrap()
                .rebase(0x1_0000_4000, offer(0x1_0000_4008, first).input())
                .is_some()
        );
        assert_eq!(next.counters.loaded, 1);
        // The record this boot used is kept first under a tight bound.
        let one = layout_bytes(1, next.segments()[0].get(&first).unwrap().len() as u64);
        let kept = written(&mut next, one);
        let pack = Pack::parse(&kept, &translator).unwrap();
        assert!(pack.get(&first).is_some() && pack.get(&second).is_none());

        // A damaged pack starts the store empty.
        let (mut empty, error) = Store::new(translator, Some(&bytes[..10]));
        assert_eq!(error, Some(PackError::Truncated));
        assert_eq!(empty.record(&first), None);
        assert_eq!(empty.counters.absent, 1);
    }

    /// A merge that cannot keep every record of a rank keeps those the
    /// boot used most, not the first in key order.
    #[test]
    fn a_merge_short_of_room_keeps_the_records_used_most() {
        let translator = translator_id();
        let pages: Vec<Digest> = (0u8..6).map(|page| Digest::of(&[page])).collect();
        let mut store = Store::new(translator, None).0;
        for (at, &key) in pages.iter().enumerate() {
            store.offer(offer(0x1_0000_4008 + 0x4000 * at as u64, key));
        }
        let bytes = Delta::new(translator, store.take_added())
            .finish()
            .unwrap()
            .to_vec();
        let mut keys = pages.clone();
        keys.sort();
        let mut next = Store::new(translator, Some(&bytes)).0;
        // The two last in key order are used most, the first once.
        for (key, uses) in [(keys[5], 3), (keys[4], 2), (keys[0], 1)] {
            for _ in 0..uses {
                assert!(next.record(&key).is_some());
            }
        }
        assert_eq!(next.uses().get(&keys[5]).map(|uses| uses.count), Some(3));
        let record = next.segments()[0].get(&keys[0]).unwrap().len() as u64;
        let bound = layout_bytes(2, 2 * record);
        let merged = |by_use: bool| {
            let segments = next.segments().iter().cloned().map(Segment::from).collect();
            let merge = Merge::new(translator, segments, Vec::new(), bound);
            let merge = if by_use {
                merge.by_use(next.uses().clone())
            } else {
                merge
            };
            let bytes = merge.finish().unwrap().layout.to_vec();
            let pack = Pack::parse(&bytes, &translator).unwrap();
            keys.iter()
                .map(|key| pack.get(key).is_some())
                .collect::<Vec<_>>()
        };
        assert_eq!(merged(false), [true, true, false, false, false, false]);
        assert_eq!(merged(true), [false, false, false, false, true, true]);
    }

    /// A pack's index keeps the most uses a boot made of each record, so a
    /// boot short of room keeps what earlier sessions ran most before it
    /// used them itself; a pack of the schema before loads, its records
    /// unused.
    #[test]
    fn a_pack_counts_uses_across_boots_and_an_older_pack_loads() {
        let translator = translator_id();
        let pages: Vec<Digest> = (0u8..6).map(|page| Digest::of(&[page])).collect();
        let mut store = Store::new(translator, None).0;
        for (at, &key) in pages.iter().enumerate() {
            store.offer(offer(0x1_0000_4008 + 0x4000 * at as u64, key));
        }
        let first = Delta::new(translator, store.take_added())
            .finish()
            .unwrap()
            .to_vec();
        let mut keys = pages.clone();
        keys.sort();
        // A boot uses the two last in key order, and merges in full.
        let mut boot = Store::new(translator, Some(&first)).0;
        for (key, uses) in [(keys[5], 3), (keys[4], 2)] {
            for _ in 0..uses {
                assert!(boot.record(&key).is_some());
            }
        }
        let segments = boot.segments().iter().cloned().map(Segment::from).collect();
        let counted = Merge::new(translator, segments, Vec::new(), u64::MAX)
            .by_use(boot.uses().clone())
            .finish()
            .unwrap()
            .layout
            .to_vec();
        let record = boot.segments()[0].get(&keys[0]).unwrap().len() as u64;
        let bound = layout_bytes(2, 2 * record);
        let kept = |bytes: &[u8]| {
            let pack = Pack::parse(bytes, &translator).unwrap();
            keys.iter()
                .map(|key| pack.get(key).is_some())
                .collect::<Vec<_>>()
        };
        let merged = |store: &Store| {
            let segments = store
                .segments()
                .iter()
                .cloned()
                .map(Segment::from)
                .collect();
            Merge::new(translator, segments, Vec::new(), bound)
                .by_use(store.uses().clone())
                .finish()
                .unwrap()
                .layout
                .to_vec()
        };
        // The next boot has used nothing yet: it keeps the two used most.
        let (next, error) = Store::new(translator, Some(&counted));
        assert_eq!(error, None);
        assert_eq!(
            kept(&merged(&next)),
            [false, false, false, false, true, true]
        );
        // The same pack in the schema before: it loads, and with no uses
        // counted the first in key order stay, as before.
        let mut old = counted.clone();
        old[MAGIC.len()..][..4].copy_from_slice(&SCHEMA.to_le_bytes());
        let end = prefix_length(keys.len()) - CHECKSUM_BYTES;
        let checksum = Sha256::digest(&old[..end]);
        old[end..end + CHECKSUM_BYTES].copy_from_slice(&checksum);
        let (older, error) = Store::new(translator, Some(&old));
        assert_eq!(error, None);
        assert_eq!(
            kept(&merged(&older)),
            [true, true, false, false, false, false]
        );
    }

    /// A key several segments hold reads as their records merged, and a
    /// merge writes it so; a key one segment holds is copied as it is.
    #[test]
    fn segments_of_one_page_read_and_merge_as_one_record() {
        let translator = translator_id();
        let (key, other) = (Digest::of(b"page"), Digest::of(b"other page"));
        let mut store = Store::new(translator, None).0;
        store.offer(offer(0x1_0000_4008, key));
        store.offer(offer(0x1_0000_8008, other));
        let base = Delta::new(translator, store.take_added())
            .finish()
            .unwrap()
            .to_vec();
        store.offer(offer(0x1_0000_4010, key));
        let delta = Delta::new(translator, store.take_added())
            .finish()
            .unwrap()
            .to_vec();
        let segments = vec![
            Pack::parse(&base, &translator).unwrap(),
            Pack::parse(&delta, &translator).unwrap(),
        ];
        let mut next = Store::over(translator, segments.clone());
        let offsets = |record: &Record| {
            let mut offsets: Vec<u16> = record.entries.iter().map(|entry| entry.offset).collect();
            offsets.sort();
            offsets
        };
        assert_eq!(
            offsets(&next.record(&key).unwrap().record().unwrap()),
            [8, 0x10]
        );
        assert_eq!(next.counters.loaded, 1);
        let segments = segments.into_iter().map(Segment::from).collect();
        let mut merge = Merge::new(translator, segments, Vec::new(), u64::MAX);
        while !merge.step(1) {}
        let merged = merge.finish().unwrap();
        assert_eq!((merged.evicted, merged.expired, merged.invalid), (0, 0, 0));
        let bytes = merged.layout.to_vec();
        let pack = Pack::parse(&bytes, &translator).unwrap();
        let record = Record::decode(pack.get(&key).unwrap(), &translator, &key).unwrap();
        assert_eq!(offsets(&record), [8, 0x10]);
        assert_eq!(
            pack.get(&other).unwrap(),
            Pack::parse(&base, &translator)
                .unwrap()
                .get(&other)
                .unwrap()
        );
        // Random access reads what the chunks hold.
        let mut buffer = vec![0xa5; bytes.len() - 200];
        merged.layout.read_at(100, &mut buffer);
        assert_eq!(buffer[..], bytes[100..bytes.len() - 100]);
        merged
            .layout
            .read_at(bytes.len() as u64 - 10, &mut buffer[..20]);
        assert_eq!(buffer[..10], bytes[bytes.len() - 10..]);
        assert!(buffer[10..20].iter().all(|&byte| byte == 0));
    }

    /// A merge or a delta stepped a unit at a time lays out what one done
    /// at once does, in a number of steps bounded by its keys; segments laid
    /// out in memory merge as their written bytes do, and a merge's layout
    /// merges again.
    #[test]
    fn merges_and_deltas_step_a_bounded_unit_at_a_time() {
        let translator = translator_id();
        let keys: Vec<Digest> = (0u32..60).map(|n| Digest::of(&n.to_le_bytes())).collect();
        let mut store = Store::new(translator, None).0;
        let mut laid = Vec::new();
        let mut packed = Vec::new();
        for round in 0..3u64 {
            for key in keys.iter().skip(round as usize * 15).take(30) {
                store.offer(offer(0x1_0000_4008 + 8 * round, *key));
            }
            let mut delta = Delta::new(translator, store.take_added());
            let mut steps = 0;
            while !delta.step(1) {
                steps += 1;
            }
            assert!(steps <= 3 * 30 + 4, "{steps}");
            let layout = delta.finish().unwrap();
            let bytes: &'static [u8] = Vec::leak(layout.to_vec());
            packed.push(Segment::from(Pack::parse(bytes, &translator).unwrap()));
            laid.push(Segment::Laid(Arc::new(layout)));
        }
        let from_laid = Merge::new(translator, laid.clone(), Vec::new(), u64::MAX)
            .finish()
            .unwrap();
        let segments = packed;
        let all = Merge::new(translator, segments.clone(), Vec::new(), u64::MAX)
            .finish()
            .unwrap();
        assert_eq!(from_laid.layout.to_vec(), all.layout.to_vec());
        // A held layout is copied from, one nothing else holds moved from:
        // the held one still reads whole.
        assert_eq!(laid[0].records(), 30);
        let again = Merge::new(
            translator,
            vec![Segment::Laid(Arc::new(from_laid.layout))],
            Vec::new(),
            u64::MAX,
        )
        .finish()
        .unwrap();
        assert_eq!(again.layout.to_vec(), all.layout.to_vec());
        assert_eq!(all.layout.records(), keys.len());
        let bound = all.layout.bytes() / 2;
        let whole = Merge::new(translator, segments.clone(), Vec::new(), bound)
            .finish()
            .unwrap();
        let mut merge = Merge::new(translator, segments, Vec::new(), bound);
        let mut steps = 0;
        while !merge.step(1) {
            steps += 1;
        }
        assert!(steps <= 3 * keys.len() + 4, "{steps}");
        let stepped = merge.finish().unwrap();
        assert_eq!(stepped.layout.to_vec(), whole.layout.to_vec());
        assert_eq!(stepped.evicted, whole.evicted);
        assert!(whole.evicted != 0 && whole.layout.bytes() <= bound);
    }

    /// A record no boot looks up ages by one boot for each used-key list a
    /// merge folds in, and leaves after [`MAX_UNUSED_BOOTS`]; one a list
    /// names is young again.
    #[test]
    fn records_unused_for_too_many_boots_leave_the_pack() {
        let translator = translator_id();
        let (used, unused) = (Digest::of(b"used page"), Digest::of(b"unused page"));
        let mut store = Store::new(translator, None).0;
        store.offer(offer(0x1_0000_4008, used));
        store.offer(offer(0x1_0000_8008, unused));
        let mut bytes = written(&mut store, u64::MAX);
        for boot in 1..=u64::from(MAX_UNUSED_BOOTS) + 1 {
            let mut next = Store::new(translator, Some(&bytes)).0;
            assert!(next.record(&used).is_some());
            bytes = written(&mut next, u64::MAX);
            let pack = Pack::parse(&bytes, &translator).unwrap();
            assert_eq!(pack.age(&used), Some(0));
            if boot <= u64::from(MAX_UNUSED_BOOTS) {
                assert_eq!(pack.records(), 2);
                assert_eq!(pack.age(&unused), Some(boot as u8));
                // The age leaves the record's length whole.
                assert!(Record::decode(pack.get(&unused).unwrap(), &translator, &unused).is_ok());
            } else {
                assert_eq!(pack.records(), 1);
                assert_eq!(pack.age(&unused), None);
            }
        }
        // Several boots' lists folded at once count each boot.
        let mut store = Store::new(translator, None).0;
        store.offer(offer(0x1_0000_4008, used));
        store.offer(offer(0x1_0000_8008, unused));
        let bytes = written(&mut store, u64::MAX);
        let list = |keys: &[Digest]| {
            UsedKeys::parse(&UsedKeys::encode(&translator, keys.iter()), &translator).unwrap()
        };
        let segments = vec![Segment::from(Pack::parse(&bytes, &translator).unwrap())];
        let lists = vec![list(&[unused]), list(&[used]), list(&[])];
        let merged = Merge::new(translator, segments.clone(), lists.clone(), u64::MAX)
            .finish()
            .unwrap();
        let merged = merged.layout.to_vec();
        let pack = Pack::parse(&merged, &translator).unwrap();
        assert_eq!((pack.age(&used), pack.age(&unused)), (Some(1), Some(2)));
        // Past its age, a record the running boot used stays.
        let lists = alloc::vec![list(&[]); usize::from(MAX_UNUSED_BOOTS) + 1];
        let expired = Merge::new(translator, segments.clone(), lists.clone(), u64::MAX)
            .finish()
            .unwrap();
        assert_eq!((expired.layout.records(), expired.expired), (0, 2));
        let kept = Merge::new(translator, segments, lists, u64::MAX)
            .protect(list(&[used]))
            .finish()
            .unwrap();
        let kept = kept.layout.to_vec();
        let pack = Pack::parse(&kept, &translator).unwrap();
        assert_eq!(pack.records(), 1);
        assert_eq!(pack.age(&used), Some(0));
    }

    /// The time a store's lookup of a page key takes over one segment and
    /// over [`MAX_SEGMENTS`] holding the same records, for keys held and
    /// keys absent, as a boot looks up both.
    #[test]
    #[ignore = "manual: opt-in"]
    fn segment_lookup_costs() {
        segment_lookups();
    }

    #[test]
    fn lookups_find_every_held_key_whether_keys_are_spread_or_not() {
        let translator = translator_id();
        let layout = |keys: &[Digest]| {
            let records = keys
                .iter()
                .map(|&key| (key, Piece::New(Record::new(translator, key).encode())))
                .collect();
            Layout::new(&translator, records).to_vec()
        };
        // Digests, and keys bunched at the bottom of the key space, which
        // start the search far from where they lie.
        let spread: Vec<Digest> = (0u32..3000).map(|n| Digest::of(&n.to_le_bytes())).collect();
        let bunched: Vec<Digest> = (0u32..3000)
            .map(|n| {
                let mut key = [0; 32];
                key[4..8].copy_from_slice(&n.to_be_bytes());
                key[8..].copy_from_slice(&Digest::of(&n.to_le_bytes()).0[..24]);
                Digest(key)
            })
            .collect();
        let mut sorted = spread.clone();
        sorted.sort_unstable();
        let mut high = bunched.clone();
        for key in &mut high {
            key.0[0] = 0xff;
        }
        let bytes = [
            layout(&sorted[..2000]),
            layout(&sorted[2000..]),
            layout(&bunched),
            layout(&high),
        ];
        let segments: Vec<Pack> = bytes
            .iter()
            .map(|bytes| Pack::parse(bytes, &translator).unwrap())
            .collect();
        let store = Store::over(translator, segments.clone());
        for (index, keys) in [&sorted[..2000], &sorted[2000..], &bunched, &high]
            .into_iter()
            .enumerate()
        {
            for key in keys {
                assert!(segments[index].get(key).is_some());
                let stored = store.stored(key).expect("a held key");
                assert!(stored.records[index].is_some());
            }
        }
        let absent = (5000u32..9000).map(|n| Digest::of(&n.to_le_bytes()));
        let filtered: Vec<usize> = (0..4).collect();
        let passed: usize = absent
            .map(|key| {
                assert!(store.stored(&key).is_none());
                filtered
                    .iter()
                    .filter(|&&index| store.filters.may_hold(index, &key))
                    .count()
            })
            .sum();
        // About one key in 40 passes each filter.
        assert!(passed < 4 * 4000 / 20, "{passed}");
    }

    /// Lookups of held and absent keys over one segment and over eight
    /// holding the same keys, and over one large segment and seven small.
    fn segment_lookups() {
        const RECORDS: u32 = 1 << 18;
        const PROBES: u32 = 1 << 18;
        let translator = translator_id();
        let keys: Vec<Digest> = (0..RECORDS).map(|n| Digest::of(&n.to_le_bytes())).collect();
        let layout = |keys: &[Digest]| {
            let records = keys
                .iter()
                .map(|&key| (key, Piece::New(Record::new(translator, key).encode())))
                .collect();
            Layout::new(&translator, records).to_vec()
        };
        let one = [layout(&keys)];
        let eight: Vec<Vec<u8>> = keys.chunks(keys.len() / MAX_SEGMENTS).map(layout).collect();
        fn store(translator: Digest, bytes: &[Vec<u8>]) -> Store<'_> {
            let segments = bytes
                .iter()
                .map(|bytes| Pack::parse(bytes, &translator).unwrap())
                .collect();
            Store::over(translator, segments)
        }
        // Sizes a run of pair merges leaves: halves, quarters, and so on.
        let mut tiers = Vec::new();
        let mut rest = &keys[..];
        for _ in 1..MAX_SEGMENTS {
            let (large, small) = rest.split_at(rest.len() / 2);
            tiers.push(layout(large));
            rest = small;
        }
        tiers.push(layout(rest));
        let (one, eight, tiers) = (
            store(translator, &one),
            store(translator, &eight),
            store(translator, &tiers),
        );
        // Held keys in a scattered order, and absent ones.
        let held: Vec<Digest> = (0..PROBES)
            .map(|n| keys[(n.wrapping_mul(2_654_435_761) % RECORDS) as usize])
            .collect();
        let absent: Vec<Digest> = (0..PROBES)
            .map(|n| Digest::of(&(n + RECORDS).to_le_bytes()))
            .collect();
        for (name, probes) in [("held", &held), ("absent", &absent)] {
            for (segments, store) in [("1", &one), ("8 equal", &eight), ("8 tiered", &tiers)] {
                let started = std::time::Instant::now();
                let found = probes
                    .iter()
                    .filter(|key| store.stored(key).is_some())
                    .count();
                let nanos = started.elapsed().as_nanos() / u128::from(PROBES);
                std::println!(
                    "{name} keys over {segments} segments: {nanos} ns per lookup ({found} found)"
                );
            }
        }
    }

    #[test]
    fn used_key_lists_are_bounded_and_checked() {
        let translator = translator_id();
        let keys: Vec<Digest> = (0u32..3)
            .map(|index| Digest::of(&index.to_le_bytes()))
            .collect();
        let bytes = UsedKeys::encode(&translator, keys.iter().chain(&keys));
        assert_eq!(bytes.len(), USED_HEADER_BYTES + 3 * 8);
        let list = UsedKeys::parse(&bytes, &translator).unwrap();
        assert!(keys.iter().all(|key| list.contains(key)));
        assert!(!list.contains(&Digest::of(b"another")));
        assert!(UsedKeys::parse(&bytes, &Digest::of(b"other translator")).is_none());
        assert!(UsedKeys::parse(&bytes[..bytes.len() - 1], &translator).is_none());
        // Past the bound a list names none and counts every key as used.
        let many: Vec<Digest> = (0..=MAX_USED_KEYS as u32)
            .map(|index| {
                let mut key = [0; 32];
                key[..4].copy_from_slice(&index.to_le_bytes());
                Digest(key)
            })
            .collect();
        let bytes = UsedKeys::encode(&translator, many.iter());
        assert_eq!(bytes.len(), USED_HEADER_BYTES);
        assert_eq!(bytes, UsedKeys::encode_every(&translator));
        let every = UsedKeys::parse(&bytes, &translator).unwrap();
        assert!(every.every() && every.contains(&Digest::of(b"anything")));
    }

    #[test]
    fn entries_past_the_added_bound_are_counted_not_kept() {
        let translator = translator_id();
        let key = Digest::of(b"page");
        let offers: Vec<Offer> = (0..64)
            .map(|index| offer(0x1_0000_4000 + 8 * index, key))
            .collect();
        let one = offers[0].entry().unwrap().retained_bytes();
        let mut small = Store::new(translator, None).0;
        drop(small.set_bounds(Bounds {
            added: 4 * one,
            ..Bounds::default()
        }));
        let mut large = Store::new(translator, None).0;
        for offer in &offers {
            small.keep(offer, offer.entry());
            large.keep(offer, offer.entry());
        }
        assert_eq!((small.counters.added, small.counters.unkept), (4, 60));
        assert_eq!((large.counters.added, large.counters.unkept), (64, 0));
        // An offer the compiler gave no stored form is counted apart.
        large.keep(&offers[0], None);
        assert_eq!(large.counters.unstorable, 1);
        let bytes = written(&mut small, u64::MAX);
        let mut next = Store::new(translator, Some(&bytes)).0;
        assert_eq!(next.record(&key).unwrap().len(), 4);
        // A take frees the bound for the entries that come next.
        assert_eq!(small.added_bytes(), 0);
        small.keep(&offers[63], offers[63].entry());
        assert_eq!(small.counters.added, 5);
    }

    #[test]
    fn an_entry_a_segment_took_is_not_added_again_and_refusals_count_each_entry_once() {
        let translator = translator_id();
        let key = Digest::of(b"page");
        let offers: Vec<Offer> = (0..8)
            .map(|index| offer(0x1_0000_4000 + 8 * index, key))
            .collect();
        let one = offers[0].entry().unwrap().retained_bytes();
        let mut store = Store::new(translator, None).0;
        drop(store.set_bounds(Bounds {
            added: 2 * one,
            ..Bounds::default()
        }));
        // Offered twice each while the bound holds two: six refused twice.
        for _ in 0..2 {
            for offer in &offers {
                store.keep(offer, offer.entry());
            }
        }
        assert_eq!(store.counters.added, 2);
        assert_eq!(store.counters.unkept, 12);
        assert_eq!(store.counters.unkept_entries, 6);
        // A segment takes the two; compiled again, they are not added again,
        // and the bound takes two of the others.
        assert_eq!(store.take_added().len(), 1);
        for offer in &offers {
            store.keep(offer, offer.entry());
        }
        assert_eq!(store.counters.retaken, 2);
        assert_eq!(store.counters.added, 4);
        assert_eq!(store.counters.unkept_entries, 6);
        let kept = store.take_added();
        let entries: Vec<u16> = kept[&key].entries().iter().map(|e| e.offset).collect();
        assert_eq!(entries.len(), 2);
        assert!(!entries.contains(&offers[0].entry().unwrap().offset));
    }

    #[test]
    fn shared_misses_are_counted_by_reason() {
        let translator = translator_id();
        let key = Digest::of(b"page");
        let first = offer(0x1_0000_4008, key);
        let context = first.context();
        let other = Digest::of(b"other context");
        let mut store = Store::new(translator, None).0;
        store.share(usize::MAX);
        // Never offered, then compiling elsewhere.
        assert!(store.prepared(&key, first.pc(), &context).is_none());
        assert!(store.prepared(&key, first.pc(), &context).is_none());
        assert_eq!(
            (
                store.counters.shared_absent,
                store.counters.shared_in_flight
            ),
            (1, 1)
        );
        store.offer(first.clone());
        assert!(store.prepared(&key, first.pc(), &context).is_some());
        // Offered in another context.
        assert!(store.prepared(&key, first.pc(), &other).is_none());
        assert_eq!(store.counters.shared_other_context, 1);
        // Evicted.
        store.share(0);
        assert!(store.prepared(&key, first.pc(), &context).is_none());
        assert_eq!(
            (store.counters.shared_hits, store.counters.shared_evicted),
            (1, 1)
        );
    }

    #[test]
    fn parts_of_a_split_store_write_the_pack_one_store_writes() {
        let translator = translator_id();
        let keys: Vec<Digest> = (0u8..4).map(|index| Digest::of(&[index])).collect();
        let offers: Vec<Offer> = keys
            .iter()
            .enumerate()
            .map(|(index, key)| offer(0x1_0000_4008 + 0x1000 * index as u64, *key))
            .collect();
        let mut whole = Store::new(translator, None).0;
        let mut parts: Vec<Store> = (0..2)
            .map(|_| Store::over(translator, Vec::new()))
            .collect();
        for offer in &offers {
            whole.offer(offer.clone());
            parts[usize::from(offer.key().0[0] & 1)].offer(offer.clone());
        }
        let mut merged = parts.remove(0);
        for part in parts {
            merged.absorb(part);
        }
        assert_eq!(merged.counters.added, whole.counters.added);
        assert_eq!(
            written(&mut merged, u64::MAX),
            written(&mut whole, u64::MAX)
        );
    }

    /// The bytes of `wanted`'s pieces in the segments `bytes` hold.
    fn read_pieces<'b>(wanted: &Wanted, bytes: &'b [Vec<u8>]) -> [Option<&'b [u8]>; MAX_SEGMENTS] {
        let mut read = [None; MAX_SEGMENTS];
        for (index, piece) in wanted.pieces.iter().enumerate() {
            if let (Some((offset, length)), Some(bytes)) = (*piece, bytes.get(index)) {
                read[index] = Some(&bytes[offset as usize..(offset + length) as usize]);
            }
        }
        read
    }

    #[test]
    fn a_record_left_in_the_file_is_asked_for_read_and_then_served() {
        let translator = translator_id();
        let keys: Vec<Digest> = (0..64u32).map(|n| Digest::of(&n.to_le_bytes())).collect();
        let layout = |keys: &[Digest]| {
            let records = keys
                .iter()
                .map(|&key| (key, Piece::New(Record::new(translator, key).encode())))
                .collect();
            Layout::new(&translator, records).to_vec()
        };
        // An older segment of every key, left in the file, and a newer one
        // of half of them, read whole.
        let bytes = [layout(&keys), layout(&keys[..32])];
        let prefix = prefix_bytes(&bytes[0], &translator).unwrap();
        let disk = Pack::index_only(&bytes[0][..prefix], &translator).unwrap();
        assert!(!disk.resident() && disk.held_bytes() == prefix);
        assert_eq!(disk.bytes(), bytes[0].len());
        let whole = Pack::parse(&bytes[1], &translator).unwrap();
        let mut store = Store::over(translator, alloc::vec![disk, whole]);

        // A lookup compiles meanwhile, and asks once.
        for key in [keys[40], keys[40], keys[3]] {
            assert!(store.record(&key).is_none());
        }
        assert_eq!(store.counters.awaited, 3);
        assert_eq!(store.counters.absent, 0);
        let absent = Digest([9; 32]);
        assert!(store.record(&absent).is_none());
        assert_eq!(store.counters.absent, 1);

        // The reader reads both pieces of a key the newer segment holds too.
        for _ in 0..2 {
            let wanted = store.take_wanted().expect("a key asked for");
            let held = keys[..32].contains(&wanted.key);
            assert_eq!(wanted.pieces[1].is_some(), held);
            let read = read_pieces(&wanted, &bytes[..1]);
            let stored = store.read_stored(&wanted, &read).expect("its pieces");
            let mut scratch = DecodeScratch::default();
            let decoded = DecodedRecord::decode(stored, &translator, &wanted.key, &mut scratch);
            drop(store.finish_read(&wanted.key, wanted.bytes(), Some(&decoded)));
        }
        assert!(store.take_wanted().is_none());
        assert_eq!(store.counters.read, 2);
        assert!(store.record(&keys[40]).is_some());
        assert!(store.record(&keys[3]).is_some());

        // Bytes that fail their checks: invalid, and never asked for again.
        assert!(store.record(&keys[50]).is_none());
        let wanted = store.take_wanted().unwrap();
        let mut damaged = read_pieces(&wanted, &bytes[..1])[0].unwrap().to_vec();
        let last = damaged.len() - 1;
        damaged[last] ^= 1;
        let mut read = [None; MAX_SEGMENTS];
        read[0] = Some(&damaged[..]);
        let stored = store.read_stored(&wanted, &read).unwrap();
        let mut scratch = DecodeScratch::default();
        let decoded = DecodedRecord::decode(stored, &translator, &wanted.key, &mut scratch);
        assert!(decoded.is_err());
        drop(store.finish_read(&wanted.key, wanted.bytes(), Some(&decoded)));
        assert_eq!(store.counters.invalid, 1);
        assert!(store.record(&keys[50]).is_none());
        assert!(store.take_wanted().is_none());
        // A read of the wrong length is no record.
        assert!(store.record(&keys[60]).is_none());
        let wanted = store.take_wanted().unwrap();
        let mut read = [None; MAX_SEGMENTS];
        read[0] = Some(&damaged[..1]);
        assert!(store.read_stored(&wanted, &read).is_none());
    }

    #[test]
    fn read_ahead_takes_the_records_used_most_first_and_skips_those_asked_for() {
        let translator = translator_id();
        let keys: Vec<Digest> = (0..16u32).map(|n| Digest::of(&n.to_le_bytes())).collect();
        let records = keys
            .iter()
            .map(|&key| (key, Piece::New(Record::new(translator, key).encode())))
            .collect();
        let mut merge_uses = BTreeMap::new();
        for (n, key) in keys.iter().enumerate() {
            merge_uses.insert(
                *key,
                Use {
                    count: n as u32,
                    first: n as u32,
                },
            );
        }
        // A merge writes each record's uses into the index.
        let layout = Layout::new(&translator, records).to_vec();
        let pack = Pack::parse(&layout, &translator).unwrap();
        let mut merge = Merge::new(
            translator,
            alloc::vec![Segment::Packed(pack)],
            alloc::vec::Vec::new(),
            u64::MAX,
        )
        .by_use(merge_uses);
        while !merge.step(usize::MAX) {}
        let merged = merge.finish().unwrap().layout.to_vec();
        let prefix = prefix_bytes(&merged, &translator).unwrap();
        let disk = Pack::index_only(&merged[..prefix], &translator).unwrap();
        let mut store = Store::over(translator, alloc::vec![disk]);
        assert!(store.ahead_bytes() >= 16 * size_of::<(u8, u32)>());
        assert!(store.record(&keys[15]).is_none());
        let order: Vec<Digest> = core::iter::from_fn(|| store.take_ahead())
            .map(|wanted| wanted.key)
            .collect();
        let expected: Vec<Digest> = keys[..15].iter().rev().copied().collect();
        assert_eq!(order, expected);
        assert_eq!(store.counters.read_ahead, 15);
    }

    /// A merge lays records out in the order the boot first used them,
    /// those it did not use after them; the reader finds the records laid
    /// out after one a lookup asks for, a launch's, and a store whose
    /// segments a merge replaced reads none of them again.
    #[test]
    fn merged_records_lie_in_the_order_the_boot_first_used_them() {
        let translator = translator_id();
        let keys: Vec<Digest> = (0..12u32).map(|n| Digest::of(&n.to_le_bytes())).collect();
        let records = keys
            .iter()
            .map(|&key| (key, Piece::New(Record::new(translator, key).encode())))
            .collect();
        let layout = Layout::new(&translator, records).to_vec();
        let pack = Pack::parse(&layout, &translator).unwrap();
        // The boot uses eight keys in an order of its own, and not the rest.
        let order = [7usize, 2, 9, 0, 11, 4, 5, 1];
        let mut store = Store::over(translator, alloc::vec![pack]);
        for &n in &order {
            store.offer(offer(0x1_0000_4008, keys[n]));
        }
        store.offer(offer(0x1_0000_4010, keys[7]));
        assert_eq!(store.uses()[&keys[7]].count, 2);
        let mut merge = Merge::new(
            translator,
            alloc::vec![Segment::Packed(pack)],
            Vec::new(),
            u64::MAX,
        )
        .by_use(store.uses().clone());
        while !merge.step(usize::MAX) {}
        let merged = merge.finish().unwrap().layout.to_vec();
        let merged_pack = Pack::parse(&merged, &translator).unwrap();
        let mut by_offset: Vec<(u64, Digest)> = keys
            .iter()
            .map(|key| (merged_pack.locate(key).unwrap().0, *key))
            .collect();
        by_offset.sort();
        let laid: Vec<Digest> = by_offset.iter().map(|&(_, key)| key).collect();
        let used: Vec<Digest> = order.iter().map(|&n| keys[n]).collect();
        assert_eq!(laid[..8], used[..]);
        // The rest in the order the pack they came from had them: by key.
        let mut rest: Vec<Digest> = [3usize, 6, 8, 10].iter().map(|&n| keys[n]).collect();
        rest.sort();
        assert_eq!(laid[8..], rest[..]);
        // Every record reads back the same.
        for key in &keys {
            assert_eq!(merged_pack.get(key), pack.get(key));
        }
        // Left in the file, the layout order follows a record with those
        // the boot used after it.
        let prefix = prefix_bytes(&merged, &translator).unwrap();
        let disk = Pack::index_only(&merged[..prefix], &translator).unwrap();
        let segments = alloc::vec![disk];
        let order_of = LayoutOrder::new(&segments);
        let after: Vec<Digest> = order_of
            .after(&segments, &keys[9], 3)
            .into_iter()
            .map(|(segment, index)| segments[segment].key_at(index as usize))
            .collect();
        assert_eq!(after, used[3..6]);
        let walked: Vec<Digest> = (0..12)
            .map(|n| segments[0].key_at(order_of.nth(0, n).unwrap() as usize))
            .collect();
        assert_eq!(walked, laid);
        assert_eq!(order_of.nth(0, 12), None);
        // A store over it claims a record once; retired, none.
        let mut store = Store::over(translator, segments);
        assert!(store.claim(&keys[3]).is_some());
        assert!(store.claim(&keys[3]).is_none());
        // Asked for on a seeking disk, the oldest past the bound are
        // forgotten, to be asked again.
        assert!(store.want(&keys[6]) && store.want(&keys[8]));
        assert_eq!(store.trim_wanted(1), 1);
        assert_eq!(store.take_wanted().map(|wanted| wanted.key), Some(keys[8]));
        assert!(store.want(&keys[6]));
        assert_eq!(store.take_wanted().map(|wanted| wanted.key), Some(keys[6]));
        store.retire();
        assert!(store.claim(&keys[6]).is_none());
        assert!(store.take_ahead().is_none());
        assert!(!store.want(&keys[8]));
        assert!(store.stored(&keys[8]).is_none());
    }
}
