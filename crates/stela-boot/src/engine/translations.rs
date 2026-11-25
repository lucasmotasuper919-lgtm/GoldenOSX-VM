//! `stela.efi`'s persistent translation store: the pack file `stela`
//! makes room for in `stela/store` on the boot USB's FAT32 data partition.
//!
//! The file is a journal (`stela_formats::journal`) of pack segments and of
//! used-key lists, one per boot (`translations::pack`): an update writes
//! only into space the newest commit does not use and then commits, so a
//! write cut by a power loss, a panic or a forced stop leaves the previous
//! content whole.
//!
//! Before boot services end, [`read`] finds the volume with `store.cfg`,
//! gives this translator the preallocated spare file when it has no pack
//! yet, reads the newest commit's segments and lists, as many segments as
//! this PC's RAM budget holds (the newest first), and notes the volume's GPT
//! partition. The engine reads blocks from the segments ([`install`]), a
//! key's record merged from all that hold it, and keeps the stored form of
//! each block it compiles anew. While the guest runs, [`step`] writes those
//! entries to the file as new segments and, every few boots, merges the
//! segments and lists into one, a bounded step at a time on processors whose
//! guest waits; when the guest restarts or powers off, [`write`] writes what
//! is left and this boot's used-key list. The pack is a cache: every failure
//! leaves the boot running without a store, or without writing it.
//!
//! When every PE runs on its own host core (the default), their machines
//! share the store ([`Shared`]), split by page key into parts behind locks
//! of their own, and it keeps the blocks any of them compiled for the others
//! to install ([`share`]).
use super::bundle::Bytes;
use super::store_file::{Direct, Held, Queued, Staged};
use crate::{firmware, handoff::Region, output, paging::PAGE};
use alloc::{boxed::Box, collections::BTreeMap, format, string::String, vec, vec::Vec};
use core::{
    cell::UnsafeCell,
    sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering},
};
use stela_boot_core::host_memory::PACK_FLOOR;
use stela_crypto::{Sha256, digest::Digest as _};
use stela_drivers::block::BlockDevice;
use stela_engine::machine::Looked;
use stela_engine::machine::Offer;
use stela_engine::machine::Translations;
use stela_engine::machine::engine::StoreEvent;
use stela_engine::machine::engine::ahead;
use stela_engine::translations;
use stela_engine::translations::Digest;
use stela_engine::translations::pack;
use stela_engine::translations::pack::{DecodedRecord, Pack, StagedOffer, Store, UsedKeys};
use stela_engine::translations::writer::{self, Entries, Event, Writer};
use stela_engine::translations::{Record, StoredRecord};
use stela_formats::journal::{self, Commit, DATA_START, SLOT_BYTES};
use stela_formats::payload::pieces::Extent;
use stela_runtime::pc::fat32::Fat32;
use stela_runtime::pc::gpt::Guid;
use stela_runtime::translation_store;
use uefi::{
    CString16, boot,
    proto::{
        device_path::{DevicePath, DevicePathNodeEnum, media::PartitionSignature},
        media::{
            file::{Directory, File as _, FileAttribute, FileInfo, FileMode, RegularFile},
            fs::SimpleFileSystem,
        },
    },
};

/// The bytes of the pack budget a pack read on demand leaves to the window
/// and the guest (`host_memory::sizes_with_room`), set before the parts
/// are sized.
pub static PACK_ROOM: AtomicU64 = AtomicU64::new(0);

/// The handoff file holding the store's placement and what was read of its
/// pack file.
pub const FILE: &str = "translations";
/// The store's directory on its volume, and `stela`'s bound there.
pub(super) const DIRECTORY: &str = "stela\\store";
const CONFIG: &str = "store.cfg";
const BYTES_KEY: &str = "bytes";
const CONFIG_LIMIT: usize = 256;
/// A pack file's name: the first bytes of its translator id in hex, an 8.3
/// name the FAT32 reader finds.
const NAME_BYTES: usize = 4;
const PACK_EXTENSION: &str = "pak";
/// A blank, preallocated pack file `stela` keeps beside the packs: a
/// new translator's pack takes it by rename. The boot never grows a file,
/// which the firmware does by writing every byte.
const SPARE: &str = "spare.bin";
/// The pack is read in steps of this many bytes, each shown as progress.
const PROGRESS_STEP: usize = 16 << 20;
/// The journal's domain, and the kinds of its blobs: pack segments and
/// used-key lists.
const DOMAIN: [u8; 8] = pack::JOURNAL_DOMAIN;
const SEGMENT: u32 = pack::SEGMENT_BLOB;
const USED: u32 = pack::USED_BLOB;
/// The placement record at the start of the handoff file: magic, the bound
/// the engine writes within, partition GUID, flags, the commit's entry
/// count and the file's size. The commit's slot ([`SLOT_BYTES`], zeros for
/// none), the bytes held of each entry and those bytes, each from an 8-byte
/// boundary, follow.
const PLACEMENT_MAGIC: [u8; 8] = *b"STLTSTO2";
const PLACEMENT_BYTES: usize = 64;
/// The pack file holds the bound's bytes and its volume's partition is
/// known: the engine may write it.
const WRITABLE: u32 = 1;
/// A segment was left out or cut to the budget: a merge would lose what
/// was not read, so none runs.
const PARTIAL: u32 = 2;
/// A segment's held length with this bit holds its header and index only:
/// its records stay in the file, read when a lookup asks for them.
const INDEX_ONLY: u64 = 1 << 63;
/// The words of the buffer a renamed file's `FileInfo` is built in: its
/// fixed fields and an 8.3 name.
const FILE_INFO_WORDS: usize = 64;

/// The pack file's name in the store's directory.
fn pack_name(translator: &Digest) -> String {
    format!("{}.{PACK_EXTENSION}", hex(&translator.0[..NAME_BYTES]))
}

/// The pack file's path, with `separator` between components.
fn pack_path(translator: &Digest, separator: &str) -> String {
    format!(
        "{}{separator}{}",
        DIRECTORY.replace('\\', separator),
        pack_name(translator)
    )
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(super) fn open(root: &mut Directory, path: &str, mode: FileMode) -> Option<RegularFile> {
    let path = CString16::try_from(path).ok()?;
    root.open(&path, mode, FileAttribute::empty())
        .ok()
        .and_then(|handle| crate::file_kind::regular(handle).ok())
}

/// Read `buffer.len()` bytes from the current file position.
pub(super) fn read_exact(file: &mut RegularFile, buffer: &mut [u8]) -> Option<()> {
    let mut done = 0;
    while done < buffer.len() {
        let count = file.read(&mut buffer[done..]).ok()?;
        if count == 0 {
            return None;
        }
        done += count;
    }
    Some(())
}

/// `buffer.len()` bytes from `offset`.
fn read_at(file: &mut RegularFile, offset: u64, buffer: &mut [u8]) -> Option<()> {
    file.set_position(offset).ok()?;
    read_exact(file, buffer)
}

/// `store.cfg`'s bound.
pub(super) fn bound(root: &mut Directory) -> Option<u64> {
    let mut file = open(root, &format!("{DIRECTORY}\\{CONFIG}"), FileMode::Read)?;
    let mut text = vec![0; CONFIG_LIMIT];
    let mut done = 0;
    while done < text.len() {
        let count = file.read(&mut text[done..]).ok()?;
        if count == 0 {
            break;
        }
        done += count;
    }
    core::str::from_utf8(&text[..done])
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix(BYTES_KEY)?.trim().parse().ok())
}

/// The GPT partition GUID of the volume on `handle`.
pub(crate) fn partition(handle: uefi::Handle) -> Option<[u8; 16]> {
    // SAFETY: the path is only read, while no driver changes this handle.
    let path = unsafe {
        boot::open_protocol::<DevicePath>(
            boot::OpenProtocolParams {
                handle,
                agent: boot::image_handle(),
                controller: None,
            },
            boot::OpenProtocolAttributes::GetProtocol,
        )
    }
    .ok()?;
    path.node_iter().find_map(|node| match node.as_enum() {
        Ok(DevicePathNodeEnum::MediaHardDrive(drive)) => match drive.partition_signature() {
            PartitionSignature::Guid(guid) => Some(guid.to_bytes()),
            _ => None,
        },
        _ => None,
    })
}

/// What the loader read of the store's pack segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackRead {
    /// No pack yet, or one the engine would discard.
    None,
    /// All of the segments' `0` bytes.
    Whole(u64),
    /// `read` of their `held` bytes: the newest segments, and of the next
    /// the records that end within this PC's budget.
    Part { read: u64, held: u64 },
    /// Its segments' headers and indexes, `index` bytes of its `held`: the
    /// records are read when a lookup asks for them, and `room` of the
    /// budget is left to the window and the guest.
    OnDisk { index: u64, held: u64, room: u64 },
    /// None of it: this PC's budget holds less than the newest segment's
    /// index, or none at all. The pack stays on the volume, unwritten, for
    /// a PC with more RAM.
    Skipped,
}

/// The store the loader found: the handoff file's pages, whether the engine
/// can write its pack file, and what of the pack it read.
pub struct Opened {
    pub region: Region,
    pub writable: bool,
    pub pack: PackRead,
    /// The previous translator's pack, for a first boot's refill
    /// ([`previous::FILE`]).
    pub previous: Option<Region>,
}

/// The store on the first volume with `store.cfg`, as the handoff file's
/// pages: run before boot services end, with no volume opened exclusively.
/// `budget` is the most of the pack this PC's RAM holds
/// (`host_memory::pack_budget`). A pack read whole is read within it and
/// the engine merges it within it, so that it does not grow past what the
/// next boot on this PC can read. A pack read on demand holds its indexes
/// within it, and the engine merges it within the file
/// ([`writer::demand_limit`]), its records bounded by the indexes the
/// budget holds ([`writer::resident_records`]). `None` when no volume has a
/// store.
pub fn read(budget: u64) -> Option<Opened> {
    let translator = translations::translator_id();
    for handle in boot::find_handles::<SimpleFileSystem>().ok()? {
        let Ok(mut volume) = boot::open_protocol_exclusive::<SimpleFileSystem>(handle) else {
            continue;
        };
        let Ok(mut root) = volume.open_volume() else {
            continue;
        };
        // The file's size, which `stela` allocated.
        let Some(bound) = bound(&mut root) else {
            continue;
        };
        // What this PC's RAM reads of the pack.
        let ram = bound.min(budget);
        let path = pack_path(&translator, "\\");
        if open(&mut root, &path, FileMode::Read).is_none() {
            claim_spare(&mut root, &translator);
        }
        // A read-only volume's pack is read and never written; a missing
        // one leaves the boot's translations in memory.
        let mut opened = open(&mut root, &path, FileMode::ReadWrite)
            .map(|file| (file, true))
            .or_else(|| open(&mut root, &path, FileMode::Read).map(|file| (file, false)));
        let partition = partition(handle);
        // A pack that commits nothing is its translator's first boot: the
        // previous translator's pack, within a share of the budget, is read
        // to refill from, and the arena gives up as much.
        let first = opened.as_mut().is_none_or(|(file, _)| {
            let size = file
                .get_boxed_info::<FileInfo>()
                .map_or(0, |info| info.file_size());
            !previous::committed(file, &translator, size)
        });
        let previous = match first && ram >= PACK_FLOOR {
            true => previous::read(&mut root, &translator, ram / previous::PREVIOUS_SHARE),
            false => None,
        };
        let previous_bytes = previous.as_ref().map_or(0, |(_, bytes)| *bytes);
        let held_ram = ram.saturating_sub(previous_bytes);
        let reading = output::now();
        let (loaded, size, allocated, demand) = match opened {
            Some((mut file, writable)) => {
                let size = file
                    .get_boxed_info::<FileInfo>()
                    .map_or(0, |info| info.file_size());
                let allocated = writable && size >= bound;
                // The engine reads records from the partition it writes:
                // a pack it may write is read on demand, its merges too.
                let demand = allocated && partition.is_some() && ram >= PACK_FLOOR;
                let loaded = read_journal(&mut file, &translator, size, held_ram, demand)?;
                (loaded, size, allocated, demand)
            }
            None => (blank()?, 0, false, false),
        };
        // Read on demand, the pack's records stay in the file: it grows to
        // the file's limit, its records to the indexes the RAM holds.
        let (limit, records) = match demand {
            true => (
                writer::demand_limit(bound),
                writer::resident_records(held_ram),
            ),
            false => (ram, u64::MAX),
        };
        let (read, held) = match loaded.pack {
            PackRead::None | PackRead::Skipped => (0, 0),
            PackRead::Whole(bytes) => (bytes, bytes),
            PackRead::Part { read, held } => (read, held),
            PackRead::OnDisk { index, held, room } => {
                PACK_ROOM.store(room, Ordering::Release);
                (index, held)
            }
        };
        output::arguments(format_args!(
            "TRANSLATIONS {path} bound={bound} budget={budget} limit={limit} records={records} \
             file={size} segments={} of {} lists={} pack={read} held={held} skipped={} \
             allocated={allocated} partition={} read_ms={}\n",
            loaded.segments.0,
            loaded.segments.1,
            loaded.lists,
            loaded.pack == PackRead::Skipped,
            partition.is_some(),
            output::millis_since(reading)
        ));
        // Below the floor, or with its pack left for a larger PC, the boot
        // keeps its translations in memory.
        let writable = allocated
            && partition.is_some()
            && ram >= PACK_FLOOR
            && loaded.pack != PackRead::Skipped;
        let flags = if writable { WRITABLE } else { 0 } | if loaded.partial { PARTIAL } else { 0 };
        // SAFETY: the exclusive loader allocation starts with the placement
        // header, which `read_journal` and `blank` left zero.
        let placement = unsafe {
            core::slice::from_raw_parts_mut(loaded.region.physical as *mut u8, PLACEMENT_BYTES)
        };
        placement[..8].copy_from_slice(&PLACEMENT_MAGIC);
        placement[8..16].copy_from_slice(&limit.to_le_bytes());
        placement[16..32].copy_from_slice(&partition.unwrap_or_default());
        placement[32..36].copy_from_slice(&flags.to_le_bytes());
        placement[40..48].copy_from_slice(&size.to_le_bytes());
        placement[48..56].copy_from_slice(&records.to_le_bytes());
        return Some(Opened {
            region: loaded.region,
            writable,
            pack: loaded.pack,
            previous: previous.map(|(region, _)| region),
        });
    }
    None
}

/// Give `translator` the spare pack file by renaming it, when the store
/// has one: the file keeps its preallocated clusters.
fn claim_spare(root: &mut Directory, translator: &Digest) {
    let Some(mut file) = open(root, &format!("{DIRECTORY}\\{SPARE}"), FileMode::ReadWrite) else {
        return;
    };
    let Ok(info) = file.get_boxed_info::<FileInfo>() else {
        return;
    };
    let Ok(name) = CString16::try_from(pack_name(translator).as_str()) else {
        return;
    };
    let mut storage = vec![0u64; FILE_INFO_WORDS];
    // SAFETY: the words' bytes, 8-byte aligned as FileInfo requires.
    let bytes = unsafe {
        core::slice::from_raw_parts_mut(storage.as_mut_ptr().cast::<u8>(), storage.len() * 8)
    };
    let Ok(renamed) = FileInfo::new(
        bytes,
        info.file_size(),
        info.physical_size(),
        *info.create_time(),
        *info.last_access_time(),
        *info.modification_time(),
        info.attribute(),
        &name,
    ) else {
        return;
    };
    let claimed = file.set_info(renamed).is_ok() && file.flush().is_ok();
    output::arguments(format_args!(
        "TRANSLATIONS spare {} as {name}\n",
        if claimed { "taken" } else { "refused" }
    ));
}

/// What [`read_journal`] read: the handoff region (its placement left
/// zero), the segments read of those committed, the used-key lists read,
/// and whether a segment was left out or cut.
struct Loaded {
    region: Region,
    pack: PackRead,
    segments: (usize, usize),
    lists: usize,
    partial: bool,
}

/// A handoff region of a placement and a blank commit slot.
fn blank() -> Option<Loaded> {
    let bytes = PLACEMENT_BYTES + SLOT_BYTES as usize;
    let physical = firmware::pages(bytes.div_ceil(PAGE)).ok()?;
    // SAFETY: firmware returned this exclusive allocation of `bytes` or more.
    unsafe { (physical as *mut u8).write_bytes(0, bytes) };
    Some(Loaded {
        region: Region {
            physical: physical as u64,
            bytes: bytes as u64,
        },
        pack: PackRead::None,
        segments: (0, 0),
        lists: 0,
        partial: false,
    })
}

/// Read the newest commit of the pack file of `size` bytes directly into
/// the handoff region: every used-key list, checked, and the segments, the
/// newest first, while they fit `limit`; of the first that does not, its
/// records that end within what is left, its index trimmed to name only
/// them (`translation_store::within`). Segments are not checked whole here:
/// a pack checks its index, and each record its own bytes when first used.
/// With `demand`, every segment's header and index alone are held: its
/// records are read from the file as lookups and merges need them, and
/// the budget they leave goes to the window and the guest. `None` only when
/// no memory is left for the region.
fn read_journal(
    file: &mut RegularFile,
    translator: &Digest,
    size: u64,
    limit: u64,
    demand: bool,
) -> Option<Loaded> {
    let mut slots = vec![0; DATA_START as usize];
    let commit = if size >= DATA_START && read_exact(file, &mut slots).is_some() {
        journal::candidates(&slots, DOMAIN, size).into_iter().next()
    } else {
        None
    };
    drop(slots);
    let entries: &[journal::Entry] = match &commit {
        Some(commit) if commit.key == translator.0 => &commit.entries,
        _ => &[],
    };
    // What to hold of each entry: every list, and segments within the room
    // the lists leave, the newest first.
    let mut held: Vec<u64> = vec![0; entries.len()];
    let lists: u64 = entries
        .iter()
        .filter(|entry| entry.kind == USED)
        .map(|entry| entry.length)
        .sum();
    // Read on demand, every segment's records stay in the file: only its
    // header and index are held.
    let mut room = if demand {
        0
    } else {
        limit.saturating_sub(lists)
    };
    let mut partial = false;
    let mut trimmed: Option<(usize, Vec<u8>)> = None;
    let mut indexes = 0;
    let mut prefixes: Vec<(usize, Vec<u8>)> = Vec::new();
    for (index, entry) in entries.iter().enumerate().rev() {
        match entry.kind {
            USED => held[index] = entry.length,
            SEGMENT if demand => {
                if let Some(prefix) = segment_index(file, entry, translator)
                    .filter(|prefix| indexes + prefix.len() as u64 <= limit)
                {
                    indexes += prefix.len() as u64;
                    held[index] = prefix.len() as u64 | INDEX_ONLY;
                    prefixes.push((index, prefix));
                }
            }
            SEGMENT if partial => {}
            SEGMENT if entry.length <= room => {
                held[index] = entry.length;
                room -= entry.length;
            }
            SEGMENT => {
                partial = true;
                if let Some((length, prefix)) = segment_within(file, entry, translator, room) {
                    held[index] = length;
                    trimmed = Some((index, prefix));
                }
            }
            _ => {}
        }
    }
    // Read on demand, segments that held no index stop merges as a
    // partial read does.
    partial |= demand
        && entries
            .iter()
            .zip(&held)
            .any(|(entry, &length)| entry.kind == SEGMENT && length == 0);
    let blobs: usize = held
        .iter()
        .map(|&length| ((length & !INDEX_ONLY) as usize).next_multiple_of(8))
        .sum();
    let head = PLACEMENT_BYTES + SLOT_BYTES as usize + 8 * entries.len();
    let bytes = head + blobs;
    let pages = bytes.div_ceil(PAGE);
    let physical = firmware::pages(pages).ok()?;
    // SAFETY: firmware returned this exclusive allocation of `bytes` or
    // more. Initialize it before taking a mutable slice.
    let region = unsafe {
        let start = physical as *mut u8;
        start.write_bytes(0, bytes);
        core::slice::from_raw_parts_mut(start, bytes)
    };
    if let Some(commit) = &commit {
        region[PLACEMENT_BYTES..PLACEMENT_BYTES + SLOT_BYTES as usize]
            .copy_from_slice(&commit.encode());
    }
    region[36..40].copy_from_slice(&(entries.len() as u32).to_le_bytes());
    let total = blobs >> 20;
    let mut at = head;
    let mut done = 0usize;
    for (index, entry) in entries.iter().enumerate() {
        let length = (held[index] & !INDEX_ONLY) as usize;
        if length == 0 {
            continue;
        }
        let blob = &mut region[at..at + length];
        if let Some((_, prefix)) = prefixes.iter().find(|(held, _)| *held == index) {
            blob.copy_from_slice(prefix);
            at += length.next_multiple_of(8);
            continue;
        }
        let mut from = 0;
        if let Some((_, prefix)) = trimmed.as_ref().filter(|(trim, _)| *trim == index) {
            blob[..prefix.len()].copy_from_slice(prefix);
            from = prefix.len();
        }
        let mut whole = true;
        for part in blob[from..].chunks_mut(PROGRESS_STEP) {
            if read_at(file, entry.offset + from as u64, part).is_none() {
                whole = false;
                break;
            }
            from += part.len();
            done += part.len();
            crate::status::progress(&format!("{} of {total} MB", done >> 20));
        }
        whole &= match entry.kind {
            USED => Sha256::digest(&*blob)[..] == entry.digest,
            _ => true,
        };
        if !whole {
            partial |= entry.kind == SEGMENT;
            held[index] = 0;
        } else if trimmed.as_ref().is_some_and(|(trim, _)| *trim == index) {
            translation_store::trim(blob);
        }
        at += length.next_multiple_of(8);
    }
    for (index, &length) in held.iter().enumerate() {
        let at = head - 8 * (entries.len() - index);
        region[at..at + 8].copy_from_slice(&length.to_le_bytes());
    }
    let segment = |index: usize| entries[index].kind == SEGMENT;
    let (mut read, mut whole, mut segments, mut committed) = (0, 0, 0, 0);
    for (index, &length) in held.iter().enumerate().filter(|(index, _)| segment(*index)) {
        committed += 1;
        whole += entries[index].length;
        if length != 0 {
            segments += 1;
            read += length & !INDEX_ONLY;
        }
    }
    let pack = match () {
        () if committed == 0 => PackRead::None,
        () if segments == 0 => PackRead::Skipped,
        () if demand => PackRead::OnDisk {
            index: read,
            held: whole,
            room: limit.saturating_sub(read),
        },
        () if partial => PackRead::Part { read, held: whole },
        () => PackRead::Whole(read),
    };
    Some(Loaded {
        region: Region {
            physical: physical as u64,
            bytes: bytes as u64,
        },
        pack,
        segments: (segments, committed),
        lists: held
            .iter()
            .enumerate()
            .filter(|&(index, &length)| !segment(index) && length != 0)
            .count(),
        partial,
    })
}

/// The header, index and index checksum of the segment `entry` names,
/// checked as `Pack::parse` checks them: what a pack read on demand holds
/// of it. `None` when they do not read or fail the checks.
fn segment_index(
    file: &mut RegularFile,
    entry: &journal::Entry,
    translator: &Digest,
) -> Option<Vec<u8>> {
    let mut header = [0; pack::HEADER_BYTES];
    read_at(file, entry.offset, &mut header)?;
    let prefix_length = pack::prefix_bytes(&header, translator).ok()?;
    if prefix_length as u64 > entry.length {
        return None;
    }
    let mut prefix = vec![0; prefix_length];
    prefix[..header.len()].copy_from_slice(&header);
    read_exact(file, &mut prefix[header.len()..])?;
    let pack = Pack::index_only(&prefix, translator).ok()?;
    (pack.bytes() as u64 <= entry.length).then_some(())?;
    Some(prefix)
}

/// Of the segment `entry` names, the bytes that hold its records ending
/// within `room`, and its header and index: `None` when none does or its
/// index fails the checks `Pack::parse` makes.
fn segment_within(
    file: &mut RegularFile,
    entry: &journal::Entry,
    translator: &Digest,
    room: u64,
) -> Option<(u64, Vec<u8>)> {
    let mut header = [0; pack::HEADER_BYTES];
    read_at(file, entry.offset, &mut header)?;
    let prefix_length = pack::prefix_bytes(&header, translator).ok()?;
    if prefix_length as u64 > room || prefix_length as u64 > entry.length {
        return None;
    }
    let mut prefix = vec![0; prefix_length];
    prefix[..header.len()].copy_from_slice(&header);
    read_exact(file, &mut prefix[header.len()..])?;
    let within = translation_store::within(&prefix, translator, room).ok()?;
    (within.records != 0 && within.length <= entry.length).then_some((within.length, prefix))
}

/// The boot's translator and the store's partition. Its parts ([`PARTS`])
/// hold the store itself.
struct Boot {
    translator: Digest,
    partition: Option<Box<dyn BlockDevice>>,
}

/// The parts a boot's store is split into by page key, each behind a lock
/// of its own, so that PEs looking up, sharing and offering blocks of
/// different pages do not wait for each other.
const PARTS: usize = 16;

/// A spin lock that counts how often it was found held and how long its
/// waiters spun.
#[repr(align(64))]
struct Lock<T> {
    held: AtomicBool,
    value: UnsafeCell<T>,
    acquired: AtomicU64,
    contended: AtomicU64,
    spun: AtomicU64,
}
// SAFETY: `with` holds `held` for every access to `value`.
unsafe impl<T> Sync for Lock<T> {}

impl<T> Lock<T> {
    const fn new(value: T) -> Self {
        Self {
            held: AtomicBool::new(false),
            value: UnsafeCell::new(value),
            acquired: AtomicU64::new(0),
            contended: AtomicU64::new(0),
            spun: AtomicU64::new(0),
        }
    }

    /// Run `f` on the value, alone. No interrupt handler reaches it.
    fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        if self
            .held
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            let started = output::now();
            while self
                .held
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                while self.held.load(Ordering::Relaxed) {
                    core::hint::spin_loop();
                }
            }
            self.contended.fetch_add(1, Ordering::Relaxed);
            self.spun
                .fetch_add(output::now().wrapping_sub(started), Ordering::Relaxed);
        }
        self.acquired.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `held` excludes every other access until it is released.
        let result = f(unsafe { &mut *self.value.get() });
        self.held.store(false, Ordering::Release);
        result
    }
}

static BOOT: Lock<Option<Boot>> = Lock::new(None);
static STORES: [Lock<Option<Store<'static>>>; PARTS] = [const { Lock::new(None) }; PARTS];

/// The part of the store that holds the page whose bytes hash to `key`.
fn part(key: &Digest) -> &'static Lock<Option<Store<'static>>> {
    &STORES[usize::from(key.0[2]) % PARTS]
}

/// Keep the store's capabilities off the locks' cache lines so a PE can
/// skip operations that have no backing without contending with its users.
#[repr(align(64))]
struct Available {
    capabilities: AtomicU8,
    /// A preflight limit for staging. Admission checks the live store bound.
    shared_bound: AtomicUsize,
    /// The fetch pages the machines keep in all ([`Translations::kept_pages`]).
    kept_pages: AtomicUsize,
    /// The stored-form bytes the board keeps ([`Translations::kept_entry_bytes`]).
    kept_entries: AtomicUsize,
    /// The writer sheds new entries, the heap being short
    /// ([`writer::Entries::shed`]): offers skip making their stored form.
    shed: AtomicBool,
}

static AVAILABLE: Available = Available {
    capabilities: AtomicU8::new(0),
    shared_bound: AtomicUsize::new(0),
    kept_pages: AtomicUsize::new(0),
    kept_entries: AtomicUsize::new(0),
    shed: AtomicBool::new(false),
};
const READ_PACK: u8 = 1 << 0;
const WRITE_PACK: u8 = 1 << 1;
const SHARE_BLOCKS: u8 = 1 << 2;

fn available(capabilities: u8) -> bool {
    AVAILABLE.capabilities.load(Ordering::Acquire) & capabilities != 0
}

/// What the machines' use of the store cost, beyond the parts' counters:
/// fetch pages read and hashed for their records and the engine counter
/// ticks that took, shared blocks found but not installed, and the TSC
/// ticks record lookups took through the store and decoding records.
struct Costs {
    page_reads: AtomicU64,
    page_ticks: AtomicU64,
    shared_rejected: AtomicU64,
    /// Fetch pages whose record was being read at their lookup, their
    /// lookups again at later fetches, and those that found it.
    awaited_pages: AtomicU64,
    awaited_lookups: AtomicU64,
    awaited_found: AtomicU64,
    lookups: AtomicU64,
    lookup_ticks: AtomicU64,
    decode_ticks: AtomicU64,
}

static COSTS: Costs = Costs {
    page_reads: AtomicU64::new(0),
    page_ticks: AtomicU64::new(0),
    shared_rejected: AtomicU64::new(0),
    awaited_pages: AtomicU64::new(0),
    awaited_lookups: AtomicU64::new(0),
    awaited_found: AtomicU64::new(0),
    lookups: AtomicU64::new(0),
    lookup_ticks: AtomicU64::new(0),
    decode_ticks: AtomicU64::new(0),
};

/// What the store keeps beyond its pack, for an engine heap of `heap` bytes:
/// entries added during the boot, decoded records, and the fetch pages the
/// machines keep (a page and its bookkeeping
/// take about 4.5 KiB).
fn bounds(heap: usize) -> (pack::Bounds, usize) {
    const MIB: usize = 1 << 20;
    let defaults = pack::Bounds::default();
    let bounds = pack::Bounds {
        added: (heap / 32).clamp(defaults.added, 512 * MIB),
        decoded: (heap / 64).clamp(defaults.decoded, 128 * MIB),
    };
    (bounds, (heap / 64 / (4608)).max(4096))
}

/// Make `parts` the store's, over the segments read, sharing up to
/// `shared` bytes of blocks across PEs.
fn install_parts(
    translator: Digest,
    segments: Vec<Pack<'static>>,
    shared: usize,
    decoded: Option<usize>,
) {
    let (mut bounds, kept_pages) = bounds(crate::ALLOCATOR.usage().total);
    if let Some(decoded) = decoded {
        bounds.decoded = decoded.max(pack::Bounds::default().decoded);
    }
    AVAILABLE.kept_pages.store(kept_pages, Ordering::Release);
    let filters = pack::KeyFilters::new(&segments);
    for lock in &STORES {
        let evicted = lock.with(|store| {
            let store = store.get_or_insert_with(|| {
                let mut store = Store::filtered(translator, segments.clone(), filters.clone());
                // Each part reads ahead the records of its own keys.
                store.keep_ahead(|key| core::ptr::eq(part(key), lock));
                store
            });
            store.share(shared / PARTS);
            store.set_bounds(bounds.share(PARTS))
        });
        drop(evicted);
    }
}

/// The engine's view of the boot's store, from any PE's machine.
pub struct Shared;

impl Shared {
    fn publish_shared(offer: Offer) {
        let bound = AVAILABLE.shared_bound.load(Ordering::Acquire) / PARTS;
        let lock = part(&offer.key());
        let Some(mut staged) = StagedOffer::new(offer, bound) else {
            return;
        };
        let retired = lock.with(|store| store.as_mut()?.accept_shared(&mut staged));
        drop(retired);
    }

    /// Keep the offer's stored form, made before the lock is taken, and
    /// share its block.
    fn publish_writable(offer: Offer) {
        let bound = AVAILABLE.shared_bound.load(Ordering::Acquire) / PARTS;
        let lock = part(&offer.key());
        if AVAILABLE.shed.load(Ordering::Relaxed) {
            lock.with(|store| {
                if let Some(store) = store {
                    store.count_shed(&offer);
                }
            });
        } else {
            let entry = offer.entry();
            lock.with(|store| {
                if let Some(store) = store {
                    store.keep(&offer, entry);
                }
            });
        }
        let Some(mut staged) = StagedOffer::new(offer, bound) else {
            return;
        };
        let retired = lock.with(|store| store.as_mut()?.accept_shared(&mut staged));
        drop(retired);
    }
}

impl Translations for Shared {
    fn record(&mut self, key: &Digest) -> Option<alloc::sync::Arc<StoredRecord>> {
        match self.lookup(key) {
            Looked::Found(record) => Some(record),
            Looked::Absent | Looked::Awaited => None,
        }
    }

    /// The pack bytes outlive the lock, so other PEs can reach the store
    /// while a record is decoded. A writer also tracks absent keys.
    fn lookup(&mut self, key: &Digest) -> Looked {
        if !available(READ_PACK | WRITE_PACK) {
            return Looked::Absent;
        }
        let started = output::now();
        COSTS.lookups.fetch_add(1, Ordering::Relaxed);
        let lock = part(key);
        let mut awaited = false;
        let lookup = lock.with(|store| {
            let store = store.as_mut()?;
            if let Some(record) = store.cached_record(key) {
                return Some(Ok(record));
            }
            match store.stored(key) {
                Some(stored) => Some(Err((
                    stored,
                    *store.translator(),
                    store.take_decode_scratch(stored),
                ))),
                None => {
                    // A record left in the file is asked for: the block
                    // compiles meanwhile, and a later fetch asks again.
                    awaited = store.want(key);
                    if !awaited {
                        store.looked_up(key, false, false);
                    }
                    None
                }
            }
        });
        let (stored, translator, mut scratch) = match lookup {
            Some(Ok(record)) => {
                COSTS
                    .lookup_ticks
                    .fetch_add(output::now().wrapping_sub(started), Ordering::Relaxed);
                return Looked::Found(record);
            }
            Some(Err(source)) => source,
            None => {
                COSTS
                    .lookup_ticks
                    .fetch_add(output::now().wrapping_sub(started), Ordering::Relaxed);
                return match awaited {
                    true => Looked::Awaited,
                    false => Looked::Absent,
                };
            }
        };
        let decoding = output::now();
        let decoded = DecodedRecord::decode(stored, &translator, key, &mut scratch);
        COSTS
            .decode_ticks
            .fetch_add(output::now().wrapping_sub(decoding), Ordering::Relaxed);
        let evicted = lock.with(|store| {
            let store = store.as_mut()?;
            Some(store.finish_decode(key, stored, &decoded, &mut scratch))
        });
        drop(evicted);
        COSTS
            .lookup_ticks
            .fetch_add(output::now().wrapping_sub(started), Ordering::Relaxed);
        decoded.ok().map_or(Looked::Absent, |decoded| {
            Looked::Found(decoded.into_record())
        })
    }

    fn prepared(
        &mut self,
        key: &Digest,
        pc: u64,
        context: &Digest,
    ) -> Option<alloc::sync::Arc<stela_engine::machine::engine::SharedTranslation>> {
        if !available(SHARE_BLOCKS) {
            return None;
        }
        part(key).with(|store| store.as_mut()?.prepared(key, pc, context))
    }

    fn offer(&mut self, offer: Offer) {
        let capabilities = AVAILABLE.capabilities.load(Ordering::Acquire);
        // Its successors, for a waiting PE to prepare.
        if capabilities & SHARE_BLOCKS != 0 {
            AHEAD.with(|queue| queue.get_or_insert_with(ahead::Queue::new).offered(&offer));
        }
        // A block prepared ahead is only shared: the PE that installs it
        // first offers it again for the pack.
        if offer.ahead() {
            if capabilities & SHARE_BLOCKS != 0 {
                Self::publish_shared(offer);
            }
        } else if capabilities & WRITE_PACK != 0 {
            Self::publish_writable(offer);
        } else if capabilities & SHARE_BLOCKS != 0 {
            Self::publish_shared(offer);
        }
    }

    fn refilled(&mut self, key: &Digest, entry: translations::Entry) {
        if available(WRITE_PACK) {
            part(key).with(|store| {
                if let Some(store) = store.as_mut() {
                    store.refilled(*key, entry);
                }
            });
        }
    }

    fn note(&mut self, event: StoreEvent) {
        match event {
            StoreEvent::PageRead { ticks } => {
                COSTS.page_reads.fetch_add(1, Ordering::Relaxed);
                COSTS.page_ticks.fetch_add(ticks, Ordering::Relaxed);
            }
            StoreEvent::SharedRejected => {
                COSTS.shared_rejected.fetch_add(1, Ordering::Relaxed);
            }
            StoreEvent::AwaitedPage => {
                COSTS.awaited_pages.fetch_add(1, Ordering::Relaxed);
            }
            StoreEvent::LookedAgain { found } => {
                COSTS.awaited_lookups.fetch_add(1, Ordering::Relaxed);
                if found {
                    COSTS.awaited_found.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    fn kept_pages(&mut self) -> Option<usize> {
        match AVAILABLE.kept_pages.load(Ordering::Acquire) {
            0 => None,
            pages => Some(pages),
        }
    }

    fn kept_entry_bytes(&mut self) -> Option<usize> {
        match AVAILABLE.kept_entries.load(Ordering::Acquire) {
            0 => None,
            bytes => Some(bytes),
        }
    }

    /// A 128th of the heap in page keys of 56 bytes: 2^18 keys (14.7 MB,
    /// the engine's most) from a 1.9 GB heap, enough for twice the distinct
    /// code pages a desktop boot reads.
    fn known_pages(&mut self) -> Option<usize> {
        Some(crate::ALLOCATOR.usage().total / 128 / 56)
    }
}

/// The blocks waiting to be prepared ahead of any fetch
/// ([`stela_engine::machine::engine::ahead`]).
static AHEAD: Lock<Option<ahead::Queue>> = Lock::new(None);

/// How long one [`prepare_ahead`] call may take, in microseconds.
const AHEAD_MICROS: u64 = 500;

/// Prepare blocks ahead of any fetch, newest job first, for at most
/// [`AHEAD_MICROS`] or until `stop` says this PE's guest has work: a PE
/// whose guest waits calls it. `read` reads a guest-physical frame. A job
/// whose block the store holds (in a pack record or among its shared
/// blocks) or whose frame no longer holds its page's bytes is skipped; the
/// others' blocks join the shared blocks.
pub fn prepare_ahead(read: &mut dyn FnMut(u64, &mut [u8]) -> bool, stop: &dyn Fn() -> bool) {
    if !available(SHARE_BLOCKS) {
        return;
    }
    let started = super::host::fresh_counter();
    let deadline = started + super::host::COUNTER_HZ * AHEAD_MICROS / 1_000_000;
    let mut page = [0u8; translations::PAGE_BYTES];
    // The frame read last and the key its bytes hashed to.
    let mut verified: Option<(u64, Digest)> = None;
    while super::host::fresh_counter() < deadline && !stop() {
        let Some(job) = AHEAD.with(|queue| queue.as_mut()?.pop()) else {
            break;
        };
        let outcome = ahead_job(&job, read, &mut page, &mut verified);
        AHEAD.with(|queue| {
            if let Some(queue) = queue {
                queue.note(outcome);
            }
        });
    }
}

fn ahead_job(
    job: &ahead::Job,
    read: &mut dyn FnMut(u64, &mut [u8]) -> bool,
    page: &mut [u8; translations::PAGE_BYTES],
    verified: &mut Option<(u64, Digest)>,
) -> ahead::Outcome {
    // A job for another page than its naming block's takes the key of the
    // bytes its frame holds now.
    let key = match job.key() {
        Some(key) => *key,
        None => match *verified {
            Some((frame, key)) if frame == job.frame() => key,
            _ => {
                *verified = None;
                if !read(job.frame(), page) {
                    return ahead::Outcome::Stale;
                }
                let key = Digest::of(page);
                *verified = Some((job.frame(), key));
                key
            }
        },
    };
    let held = part(&key).with(|store| {
        store
            .as_ref()
            .is_some_and(|store| store.holds_shared(&key, job.pc(), job.context()))
    });
    if held
        || Shared
            .record(&key)
            .is_some_and(|record| record.contains(job.offset(), job.context()))
    {
        return ahead::Outcome::Present;
    }
    if *verified != Some((job.frame(), key)) {
        *verified = None;
        if !read(job.frame(), page) || !ahead::page_matches(job, page) {
            return ahead::Outcome::Stale;
        }
        *verified = Some((job.frame(), key));
    }
    match ahead::prepare(job, key, page) {
        Ok(offer) => {
            Shared.offer(offer);
            ahead::Outcome::Prepared
        }
        Err(outcome) => outcome,
    }
}

mod previous;
pub use previous::{FILE as PREVIOUS_FILE, install as install_previous, step as refill};

/// The store the engine uses, if the loader read one or the PEs share one.
pub fn store() -> Option<&'static mut Shared> {
    available(READ_PACK | WRITE_PACK | SHARE_BLOCKS).then(|| Box::leak(Box::new(Shared)))
}

/// Keep up to `bound` bytes of compiled code for the PEs, the planner's
/// share of the code budget beside the window (`host_memory::Sizes::kept`):
/// a third for the pack's decoded records, which held a sixty-fourth of
/// the heap before the budget held them, a quarter for
/// blocks shared as they are compiled (buffers and cache metadata in the
/// loader's store or a store for this boot), the rest for stored forms of
/// blocks the ring took away and the guest runs again, which the board
/// relocates instead of compiling (`Translations::kept_entry_bytes`). The
/// shared blocks alone kept compile order, newer than the ring's code, so
/// a block the ring took away had left them long before (run
/// 20261116-111302: 447 K hits, 28.6 M misses). Allocator overhead and
/// shared immutable preparation storage are outside this budget.
pub fn share(bound: usize) {
    let decoded = bound / 3;
    let kept = bound - bound / 4 - decoded;
    AVAILABLE.kept_entries.store(kept, Ordering::Release);
    let bound = bound / 4;
    BOOT.with(|boot| {
        let translator = boot
            .get_or_insert_with(|| Boot {
                translator: translations::translator_id(),
                partition: None,
            })
            .translator;
        install_parts(translator, Vec::new(), bound, Some(decoded));
        AVAILABLE.shared_bound.store(bound, Ordering::Release);
        if bound != 0 {
            AVAILABLE
                .capabilities
                .fetch_or(SHARE_BLOCKS, Ordering::Release);
        } else {
            AVAILABLE
                .capabilities
                .fetch_and(!SHARE_BLOCKS, Ordering::Release);
        }
    });
}

/// Make the store of the handoff file `region` the engine's: the GPT
/// partition to open for writing it, when it can be written.
pub fn install(region: &'static mut [u8]) -> Option<Guid> {
    let head = PLACEMENT_BYTES + SLOT_BYTES as usize;
    if region.len() < head || region[..8] != PLACEMENT_MAGIC {
        return None;
    }
    let word = |at: usize| u32::from_le_bytes(region[at..at + 4].try_into().expect("a word"));
    let count = word(36) as usize;
    // The blobs end where their lengths, after the head, say.
    let lengths = head.checked_add(count.checked_mul(8)?)?;
    let blobs = region
        .get(head..lengths)?
        .as_chunks::<8>()
        .0
        .iter()
        .try_fold(lengths, |at, length| {
            at.checked_add(
                usize::try_from(u64::from_le_bytes(*length) & !INDEX_ONLY)
                    .ok()?
                    .next_multiple_of(8),
            )
        })?
        .min(region.len());
    let contents: &'static [u8] = &region[..blobs];
    let field = |at: usize| u64::from_le_bytes(contents[at..at + 8].try_into().expect("a field"));
    let word = |at: usize| u32::from_le_bytes(contents[at..at + 4].try_into().expect("a word"));
    let (limit, flags, size, records) = (field(8), word(32), field(40), field(48));
    let translator = translations::translator_id();
    let slot = &contents[PLACEMENT_BYTES..head];
    let committed = (0..2).find_map(|index| Commit::decode(slot, index, DOMAIN, size));
    let mut segments = Vec::new();
    let mut offsets = Vec::new();
    let mut used = Vec::new();
    let mut discarded = 0;
    if let Some(commit) = committed
        .as_ref()
        .filter(|commit| commit.key == translator.0 && commit.entries.len() == count)
    {
        let mut at = head + 8 * count;
        for (index, entry) in commit.entries.iter().enumerate() {
            let word = field(head + 8 * index);
            let length = usize::try_from(word & !INDEX_ONLY).ok()?;
            let bytes = contents.get(at..at.checked_add(length)?)?;
            at += length.next_multiple_of(8);
            let parsed = match word & INDEX_ONLY {
                0 => Pack::parse(bytes, &translator),
                _ => Pack::index_only(bytes, &translator),
            };
            match entry.kind {
                SEGMENT if length != 0 => match parsed {
                    Ok(pack) => {
                        segments.push(pack);
                        offsets.push(entry.offset);
                    }
                    Err(error) => {
                        output::arguments(format_args!(
                            "TRANSLATIONS segment discarded: {error:?}\n"
                        ));
                        discarded += 1;
                    }
                },
                USED if length != 0 => match UsedKeys::parse(bytes, &translator) {
                    Some(list) => used.push(list),
                    None => discarded += 1,
                },
                _ => {}
            }
        }
    }
    let partial = flags & PARTIAL != 0;
    output::arguments(format_args!(
        "TRANSLATIONS pack segments={} records={} bytes={} lists={} discarded={discarded} \
         partial={partial}\n",
        segments.len(),
        segments.iter().map(Pack::records).sum::<usize>(),
        segments.iter().map(Pack::bytes).sum::<usize>(),
        used.len(),
    ));
    let readable = if segments.is_empty() { 0 } else { READ_PACK };
    let file = writer::File {
        translator,
        limit,
        records,
        file_bytes: size,
        committed,
        segments: segments
            .iter()
            .zip(&offsets)
            .map(|(&pack, &offset)| pack::Segment::at(pack, offset))
            .collect(),
        used,
        partial,
    };
    let held = PackWriter {
        writer: Writer::new(file, super::host::COUNTER_HZ),
        extents: None,
        queued: Queued::default(),
        reads: WriterReads::default(),
        steps: 0,
        longest: 0,
    };
    WRITER.holding(true, |slot| *slot = Some(held));
    BOOT.with(|boot| {
        for lock in &STORES {
            let old = lock.with(Option::take);
            drop(old);
        }
        let offsets: Vec<Option<u64>> = segments
            .iter()
            .zip(&offsets)
            .map(|(pack, &offset)| (!pack.resident()).then_some(offset))
            .collect();
        if offsets.iter().any(Option::is_some) {
            let order = pack::LayoutOrder::new(&segments);
            let segments = segments.clone();
            READER.holding(true, |slot| {
                *slot = Some(Reader {
                    segments,
                    offsets,
                    order,
                    extents: None,
                    flights: Vec::new(),
                    launch: alloc::collections::VecDeque::new(),
                    cursor: (0, 0),
                    latency: 0,
                    longest: 0,
                    failed: 0,
                    next: 0,
                    launched: 0,
                    dropped: 0,
                })
            });
        }
        install_parts(translator, segments, 0, None);
        *boot = Some(Boot {
            translator,
            partition: None,
        });
        AVAILABLE.shared_bound.store(0, Ordering::Release);
        AVAILABLE.capabilities.store(readable, Ordering::Release);
    });
    let partition = Guid(contents[16..32].try_into().expect("a GUID"));
    (flags & WRITABLE != 0).then_some(partition)
}

/// Whether the store's partition is staged ([`super::storage::try_stage`]):
/// its commands then queue, and processors step its I/O in bounded steps.
static PARTITION_STAGED: AtomicBool = AtomicBool::new(false);

/// Keep `partition`, the store's volume, for the writes, staged when a
/// slot is free, and find the pack file's device extents while its drivers
/// may still be called directly: the writes while the guest runs only
/// queue.
pub fn set_partition(partition: Box<dyn BlockDevice>) {
    let partition = match super::storage::try_stage(partition) {
        Ok(staged) => {
            PARTITION_STAGED.store(true, Ordering::Release);
            staged
        }
        Err(unstaged) => unstaged,
    };
    let translator = BOOT.with(|boot| {
        let boot = boot.as_mut()?;
        boot.partition = Some(partition);
        AVAILABLE
            .capabilities
            .fetch_or(WRITE_PACK, Ordering::Release);
        Some(boot.translator)
    });
    let Some(translator) = translator else {
        return;
    };
    let extents = with_store_file(&pack_name(&translator), |_, extents| extents.to_vec());
    WRITER.with(|held| match extents {
        Ok(extents) => held.extents = Some(extents),
        Err(error) => {
            output::arguments(format_args!(
                "TRANSLATIONS written only at a stop: {error}\n"
            ));
        }
    });
}

/// Run `f` on the store's partition with the device extents of `name`, a
/// file in the store's directory. Only while the engine may write the
/// store.
pub fn with_store_file<R>(
    name: &str,
    f: impl FnOnce(&mut dyn BlockDevice, &[Extent]) -> R,
) -> Result<R, String> {
    BOOT.with(|boot| {
        let boot = boot.as_mut().ok_or("no translation store")?;
        let partition = boot
            .partition
            .take()
            .ok_or("the store's partition is not open")?;
        let mut volume = Fat32::open(Bytes::new(partition)).map_err(|error| format!("{error}"))?;
        let file = volume.file(&format!("{}/{name}", DIRECTORY.replace('\\', "/")));
        let mut device = volume.into_inner().into_inner();
        let result = file
            .map(|file| f(&mut *device, &file.extents))
            .map_err(|error| format!("{error}"));
        boot.partition = Some(device);
        result
    })
}

/// Lend the store's partition to `f` while the guest runs, for background
/// writes: staged, so `f` only queues commands (`BlockDevice::start`) and
/// takes results (`finish`), never waiting for the disk. The pack file and
/// the pipeline store share it. `None` when it is absent, lent, or not
/// staged.
pub fn lend_staged_partition<R>(f: impl FnOnce(&mut dyn BlockDevice) -> R) -> Option<R> {
    if !PARTITION_STAGED.load(Ordering::Acquire) {
        return None;
    }
    BOOT.with(|boot| {
        let partition = boot.as_mut()?.partition.as_deref_mut()?;
        Some(f(partition))
    })
}

/// Milliseconds of `ticks` TSC ticks.
fn millis(ticks: u64) -> u64 {
    output::millis_since(output::now().wrapping_sub(ticks))
}

/// Report what the store did and cost this boot: lookups and their causes,
/// shared misses by reason, stored forms kept and checked, and lock waits.
fn report(counters: &pack::Counters) {
    previous::report_now();
    let ahead = AHEAD.with(|queue| queue.as_ref().map(ahead::Queue::counts));
    let ahead = ahead.unwrap_or_default();
    output::arguments(format_args!(
        "TRANSLATIONS ahead queued={} elsewhere={} unwalked={} repeated={} dropped={} \
         prepared={} present={} stale={} refused={}\n",
        ahead.queued,
        ahead.elsewhere,
        ahead.unwalked,
        ahead.repeated,
        ahead.dropped,
        ahead.prepared,
        ahead.present,
        ahead.stale,
        ahead.refused,
    ));
    let costs = &COSTS;
    let load = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
    output::arguments(format_args!(
        "TRANSLATIONS lookups={} repeated={} cached={} decoded={} lookup_ms={} decode_ms={} \
         pages={} page_ms={}\n",
        load(&costs.lookups),
        counters.repeated,
        counters.cached,
        counters.decoded,
        millis(load(&costs.lookup_ticks)),
        millis(load(&costs.decode_ticks)),
        load(&costs.page_reads),
        load(&costs.page_ticks) / (super::host::COUNTER_HZ / 1000),
    ));
    output::arguments(format_args!(
        "TRANSLATIONS shared hits={} rejected={} absent={} other_context={} evicted={} \
         in_flight={}\n",
        counters.shared_hits,
        load(&costs.shared_rejected),
        counters.shared_absent,
        counters.shared_other_context,
        counters.shared_evicted,
        counters.shared_in_flight,
    ));
    let (mut acquired, mut contended, mut spun) = (0, 0, 0);
    for lock in &STORES {
        acquired += load(&lock.acquired);
        contended += load(&lock.contended);
        spun += load(&lock.spun);
    }
    let checks = stela_engine::machine::entry_checks();
    output::arguments(format_args!(
        "TRANSLATIONS added={} unstorable={} unkept={} unkept_entries={} retaken={} shed={} \
         checked={} differed={} loaded={} absent={} invalid={} locks acquired={acquired} \
         contended={contended} spin_ms={}\n",
        counters.added,
        counters.unstorable,
        counters.unkept,
        counters.unkept_entries,
        counters.retaken,
        counters.shed,
        checks.checked,
        checks.differed,
        counters.loaded,
        counters.absent,
        counters.invalid,
        millis(spun),
    ));
    report_reader(counters);
}

/// How long a step may take, in microseconds.
const STEP_MICROS: u64 = 1000;

/// The pack file's writer (`translations::writer`) with what the boot adds:
/// the file's device extents, found while the drivers may be called, the
/// staged commands of the write in flight, and the steps it took.
struct PackWriter {
    writer: Writer<'static>,
    extents: Option<Vec<Extent>>,
    queued: Queued,
    reads: WriterReads,
    steps: u64,
    longest: u64,
}

static WRITER: Held<PackWriter> = Held::new();

/// The store's parts, as the writer sees them.
struct Parts;

impl Entries for Parts {
    fn take_added(&mut self) -> BTreeMap<Digest, Record> {
        let mut added = BTreeMap::new();
        for lock in &STORES {
            let mut part =
                lock.with(|store| store.as_mut().map(Store::take_added).unwrap_or_default());
            added.append(&mut part);
        }
        added
    }

    fn found_keys(&mut self) -> Vec<Digest> {
        let mut keys = Vec::new();
        for lock in &STORES {
            lock.with(|store| {
                if let Some(store) = store {
                    keys.extend(store.used_keys().copied());
                }
            });
        }
        keys
    }

    fn added(&mut self) -> u64 {
        STORES
            .iter()
            .map(|lock| lock.with(|store| store.as_ref().map_or(0, |store| store.counters.added)))
            .sum()
    }

    fn pressed(&mut self) -> bool {
        STORES
            .iter()
            .any(|lock| lock.with(|store| store.as_ref().is_some_and(Store::added_pressed)))
    }

    fn heap(&mut self) -> Option<writer::Heap> {
        Some(writer::Heap {
            total: crate::ALLOCATOR.total() as u64,
            room: crate::ALLOCATOR.room() as u64,
        })
    }

    fn added_bytes(&mut self) -> u64 {
        STORES
            .iter()
            .map(|lock| lock.with(|store| store.as_ref().map_or(0, |store| store.added_bytes())))
            .sum::<usize>() as u64
    }

    fn uses(&mut self) -> BTreeMap<Digest, pack::Use> {
        // Each key belongs to one part: the parts' maps are disjoint.
        let mut uses = BTreeMap::<Digest, pack::Use>::new();
        for lock in &STORES {
            lock.with(|store| {
                if let Some(store) = store {
                    uses.extend(store.uses().iter().map(|(key, uses)| (*key, *uses)));
                }
            });
        }
        uses
    }

    fn shed(&mut self, shed: bool) {
        AVAILABLE.shed.store(shed, Ordering::Relaxed);
        for lock in &STORES {
            lock.with(|store| {
                if let Some(store) = store {
                    store.shed(shed);
                }
            });
        }
    }
}

/// Report what a writer step did.
fn report_event(event: &Event) {
    match event {
        Event::Segment {
            records,
            bytes,
            trigger,
        } => output::arguments(format_args!(
            "TRANSLATIONS segment records={records} bytes={bytes} trigger={trigger:?}\n"
        )),
        Event::Merged {
            full,
            records,
            bytes,
            bound,
            evicted,
            expired,
            invalid,
        } => output::arguments(format_args!(
            "TRANSLATIONS merged full={full} records={records} bytes={bytes} bound={bound} \
             evicted={evicted} expired={expired} invalid={invalid}\n"
        )),
        Event::MergeCommitted { full } => {
            output::arguments(format_args!("TRANSLATIONS merge written full={full}\n"))
        }
        Event::Waiting { records } => output::arguments(format_args!(
            "TRANSLATIONS segment of {records} records waits for a merge to make room\n"
        )),
        Event::Lost { records } => output::arguments(format_args!(
            "TRANSLATIONS segment of {records} records left out: larger than the file holds\n"
        )),
        Event::Shed {
            reason,
            room,
            pack,
            resident,
        } => output::arguments(format_args!(
            "TRANSLATIONS shed reason={reason:?} room={}M pack={}M resident={}M: new \
             translations are not kept {}\n",
            room.map_or(0, |room| room >> 20),
            pack >> 20,
            resident >> 20,
            match reason {
                writer::ShedReason::Heap => "while the heap is short",
                writer::ShedReason::Pack => "past the share of the pack a boot leaves",
            }
        )),
        Event::Abandoned { full } => output::arguments(format_args!(
            "TRANSLATIONS merge full={full} given up: a read of the pack file failed; the file \
             keeps what it held and no merge runs again this boot\n"
        )),
        Event::Retired => {
            output::arguments(format_args!(
                "TRANSLATIONS retired: the file's only room was where the segments read at boot \
                 lay; their records are read no more and compile instead\n"
            ));
            for lock in &STORES {
                lock.with(|store| store.as_mut().map(Store::retire));
            }
        }
        Event::Released { room, bytes, lost } => output::arguments(format_args!(
            "TRANSLATIONS released room={}M bytes={}M lost={lost}: the heap is nearly full; \
             the segments stay in the file and no more are added this boot\n",
            room >> 20,
            bytes >> 20
        )),
    }
}

/// While the guest runs, from a processor whose guest waits and that holds
/// no device lock: carry the pack file's writer forward for a bounded time
/// ([`STEP_MICROS`]). Never waits for the disk.
pub fn step() {
    read_step();
    if !available(WRITE_PACK) {
        return;
    }
    let started = super::host::fresh_counter();
    WRITER.try_with(|held| {
        let Some(extents) = held.extents.as_deref() else {
            return;
        };
        if held.writer.failed() {
            return;
        }
        let idle = held.writer.idle();
        let deadline = started + super::host::COUNTER_HZ * STEP_MICROS / 1_000_000;
        let queued = &mut held.queued;
        let reads = &mut held.reads;
        let stepped = held.writer.step(
            &mut super::host::fresh_counter,
            deadline,
            &mut Parts,
            &mut |execution| {
                lend_staged_partition(|device| {
                    let mut target = Staged {
                        device,
                        extents,
                        queued: &mut *queued,
                    };
                    execution.advance(&mut target, writer::PIECE_BYTES)
                })
            },
            &mut |asked| {
                lend_staged_partition(|device| reads.step(device, extents, asked));
            },
        );
        match stepped {
            Ok(events) => events.iter().for_each(report_event),
            Err(error) => output::arguments(format_args!("TRANSLATIONS not written: {error}\n")),
        }
        if !idle || !held.writer.idle() {
            let took = super::host::fresh_counter().wrapping_sub(started);
            held.steps += 1;
            held.longest = held.longest.max(took);
        }
    });
}

/// The most records being read at once, whatever the disk's depth.
const MAX_FLIGHTS: usize = 16;
/// The most read records a step decodes: a decode takes tens of
/// microseconds, so a step stays far from a PE's exit bound.
const DECODES_PER_STEP: usize = 8;
/// A mean read latency past this (microseconds) is a disk whose reads
/// seek: the reader reads ahead in the order the records lie in the file,
/// and keeps only the reads lookups asked for that end within
/// [`QUEUE_MICROS`].
const SLOW_READ_MICROS: u64 = 3_000;
/// The longest a lookup's read may wait on a disk whose reads seek: the
/// host replay of a sign-in over a seeking disk (9 ms a read, one at a
/// time) serves 97.8 % of the pack's lookups with a 1 s bound, read-ahead in
/// layout order and the records laid out after each asked (1 GiB cache).
const QUEUE_MICROS: u64 = 1_000_000;
/// The records laid out after one a lookup asked for that the reader reads
/// next: those the boot that laid them out used after it, a launch's.
const LAUNCH: usize = 16;
/// The most of those waiting to be read.
const LAUNCH_QUEUE: usize = 256;

/// One read of `bytes` bytes of the pack file at `at` in flight: the
/// blocks' buffer, where the bytes start in it, and the device reads (tag,
/// where each lands in the buffer).
struct FileRead {
    buffer: Vec<u8>,
    skip: usize,
    bytes: usize,
    reads: Vec<(u32, usize)>,
    failed: bool,
}

impl FileRead {
    /// Start the read: `None` when the range is outside the file or the
    /// device refused a command.
    fn start(
        device: &mut dyn BlockDevice,
        extents: &[Extent],
        at: u64,
        bytes: u64,
    ) -> Option<Self> {
        let block = u64::from(device.identity().block_size).max(1);
        let start = at / block * block;
        let end = at.checked_add(bytes)?.next_multiple_of(block);
        let ranges = journal::device_ranges(extents, block, start, end - start)?;
        let mut read = Self {
            buffer: vec![0; (end - start) as usize],
            skip: (at - start) as usize,
            bytes: bytes as usize,
            reads: Vec::new(),
            failed: false,
        };
        for (lba, length, from) in ranges {
            let command = stela_drivers::block::Command::Read {
                lba,
                bytes: length as usize,
            };
            let tag = device.start_more(command).ok()?;
            read.reads.push((tag, from));
        }
        Some(read)
    }

    /// Take the device reads that ended: whether every one did.
    fn poll(&mut self, device: &mut dyn BlockDevice) -> bool {
        let (buffer, failed) = (&mut self.buffer, &mut self.failed);
        self.reads.retain(|&(tag, from)| match device.finish(tag) {
            None => true,
            Some(Ok(Some(data))) => {
                match buffer.get_mut(from..from + data.len()) {
                    Some(into) => into.copy_from_slice(&data),
                    None => *failed = true,
                }
                false
            }
            Some(_) => {
                *failed = true;
                false
            }
        });
        self.reads.is_empty()
    }

    /// The bytes read, once every device read ended: `None` when one failed.
    fn bytes(&self) -> Option<&[u8]> {
        (!self.failed).then(|| &self.buffer[self.skip..self.skip + self.bytes])
    }
}

/// One record being read: what was asked, each piece's read, when it
/// started, and whether a read could not start.
struct Flight {
    wanted: pack::Wanted,
    reads: [Option<FileRead>; pack::MAX_SEGMENTS],
    started: u64,
    failed: bool,
}

/// The reader of a pack read on demand: the segments read (their indexes),
/// each one's offset in the file (`None` for one held whole), the order
/// their records lie in there, the pack file's extents, the records in
/// flight, those laid out after a record a lookup asked for (to read
/// next), how far reading ahead in layout order went, the mean and longest
/// time a record took to read in counter ticks, the part asked next, and
/// the records it gave up.
struct Reader {
    segments: Vec<Pack<'static>>,
    offsets: Vec<Option<u64>>,
    order: alloc::sync::Arc<pack::LayoutOrder>,
    extents: Option<Vec<Extent>>,
    flights: Vec<Flight>,
    launch: alloc::collections::VecDeque<(usize, u32)>,
    cursor: (usize, usize),
    latency: u64,
    longest: u64,
    next: usize,
    failed: u64,
    launched: u64,
    dropped: u64,
}

static READER: Held<Reader> = Held::new();

/// Read what lookups asked for and, while they ask nothing, what boots
/// used most, on the store's staged partition: a bounded step on a PE whose
/// guest waits. A lookup never waits for a read: its block compiles.
fn read_step() {
    if !available(READ_PACK) {
        return;
    }
    READER.try_with(|reader| {
        if reader.extents.is_none() {
            reader.extents = WRITER.try_with(|held| held.extents.clone()).flatten();
        }
        let Some(extents) = reader.extents.take() else {
            return;
        };
        let done = lend_staged_partition(|device| {
            let done = reader.collect(device);
            reader.start(device, &extents);
            done
        });
        reader.extents = Some(extents);
        for flight in done.into_iter().flatten() {
            reader.decode(flight);
        }
    });
}

impl Reader {
    /// The flights whose reads all ended, at most [`DECODES_PER_STEP`].
    fn collect(&mut self, device: &mut dyn BlockDevice) -> Vec<Flight> {
        let mut done = Vec::new();
        let mut at = 0;
        while at < self.flights.len() && done.len() < DECODES_PER_STEP {
            let flight = &mut self.flights[at];
            // Every read polled, so each takes what ended.
            let mut ended = true;
            for read in flight.reads.iter_mut().flatten() {
                ended &= read.poll(device);
            }
            if ended {
                let flight = self.flights.swap_remove(at);
                let took = super::host::fresh_counter().wrapping_sub(flight.started);
                self.latency = (self.latency * 7 + took) / 8;
                self.longest = self.longest.max(took);
                done.push(flight);
            } else {
                at += 1;
            }
        }
        done
    }

    /// Whether the disk's reads seek: lookups stop asking it.
    fn slow(&self) -> bool {
        self.latency > super::host::COUNTER_HZ / 1_000_000 * SLOW_READ_MICROS
    }

    /// The next record to read: what lookups asked (on a seeking disk, those
    /// that can end within [`QUEUE_MICROS`]), with the records laid out
    /// after each queued to follow; else those,
    /// the most used ahead on a disk that does not seek, and on one that
    /// does the next in the order the file holds them.
    fn take(&mut self) -> Option<pack::Wanted> {
        let slow = self.slow();
        // On a seeking disk, the reads that would wait past the bound.
        let keep = match slow {
            true => {
                let latency = (self.latency * 1_000_000 / super::host::COUNTER_HZ).max(1);
                (QUEUE_MICROS / latency) as usize / PARTS + 1
            }
            false => usize::MAX,
        };
        for _ in 0..PARTS {
            let lock = &STORES[self.next % PARTS];
            self.next = self.next.wrapping_add(1);
            let (dropped, wanted) = lock.with(|store| match store.as_mut() {
                Some(store) => (store.trim_wanted(keep), store.take_wanted()),
                None => (0, None),
            });
            self.dropped += dropped as u64;
            if let Some(wanted) = wanted {
                let after = self.order.after(&self.segments, &wanted.key, LAUNCH);
                if self.launch.len() + after.len() <= LAUNCH_QUEUE {
                    self.launch.extend(after);
                }
                return Some(wanted);
            }
        }
        while let Some((segment, index)) = self.launch.pop_front() {
            if let Some(wanted) = self.claim(segment, index) {
                self.launched += 1;
                return Some(wanted);
            }
        }
        if slow {
            // In layout order, a bounded number of records looked at.
            for _ in 0..pack::AHEAD_SCAN {
                let (segment, n) = self.cursor;
                self.segments.get(segment)?;
                match self.order.nth(segment, n) {
                    Some(index) => {
                        self.cursor.1 += 1;
                        if let Some(wanted) = self.claim(segment, index) {
                            return Some(wanted);
                        }
                    }
                    None => self.cursor = (segment + 1, 0),
                }
            }
            return None;
        }
        for _ in 0..PARTS {
            let lock = &STORES[self.next % PARTS];
            self.next = self.next.wrapping_add(1);
            let ahead = lock.with(|store| {
                let store = store.as_mut()?;
                store.ahead_room().then(|| store.take_ahead()).flatten()
            });
            if ahead.is_some() {
                return ahead;
            }
        }
        None
    }

    /// The record of index entry `index` of segment `segment`, to read
    /// ahead, from the part that owns its key, while its cache has room.
    fn claim(&self, segment: usize, index: u32) -> Option<pack::Wanted> {
        let key = self.segments.get(segment)?.key_at(index as usize);
        part(&key).with(|store| {
            let store = store.as_mut()?;
            store.ahead_room().then(|| store.claim(&key)).flatten()
        })
    }

    /// Start reads while the disk has room.
    fn start(&mut self, device: &mut dyn BlockDevice, extents: &[Extent]) {
        let depth = device.depth().clamp(1, MAX_FLIGHTS);
        let mut started = false;
        while self.flights.len() < depth && device.room() {
            let Some(wanted) = self.take() else {
                break;
            };
            match self.flight(device, extents, wanted) {
                Some(flight) => {
                    started = true;
                    self.flights.push(flight);
                }
                None => {
                    self.failed += 1;
                    let evicted = part(&wanted.key).with(|store| {
                        store
                            .as_mut()
                            .map(|store| store.finish_read(&wanted.key, 0, None))
                    });
                    drop(evicted);
                }
            }
        }
        if started {
            device.commit();
        }
    }

    /// The reads of `wanted`'s pieces in the file, started.
    fn flight(
        &self,
        device: &mut dyn BlockDevice,
        extents: &[Extent],
        wanted: pack::Wanted,
    ) -> Option<Flight> {
        let mut flight = Flight {
            wanted,
            reads: Default::default(),
            started: super::host::fresh_counter(),
            failed: false,
        };
        let mut any = false;
        for (segment, piece) in wanted.pieces.iter().enumerate() {
            let (Some((offset, bytes)), Some(Some(base))) = (*piece, self.offsets.get(segment))
            else {
                continue;
            };
            let read = FileRead::start(device, extents, base.checked_add(offset)?, bytes);
            flight.failed |= read.is_none();
            any |= read.is_some();
            flight.reads[segment] = read;
        }
        any.then_some(flight)
    }

    /// Decode a flight's record, outside the store's lock, and give it to
    /// the store; a failed read is given up, to be asked for again.
    fn decode(&mut self, flight: Flight) {
        let mut read: [Option<&[u8]>; pack::MAX_SEGMENTS] = [None; pack::MAX_SEGMENTS];
        let mut failed = flight.failed;
        for (slot, file) in read.iter_mut().zip(&flight.reads) {
            if let Some(file) = file {
                *slot = file.bytes();
                failed |= slot.is_none();
            }
        }
        self.failed += u64::from(failed);
        let key = flight.wanted.key;
        let lock = part(&key);
        let bytes = flight.wanted.bytes();
        let source = (!failed)
            .then(|| {
                lock.with(|store| {
                    let store = store.as_mut()?;
                    Some((
                        store.read_stored(&flight.wanted, &read)?,
                        *store.translator(),
                    ))
                })
            })
            .flatten();
        let decoded = source.map(|(stored, translator)| {
            let mut scratch = pack::DecodeScratch::default();
            DecodedRecord::decode(stored, &translator, &key, &mut scratch)
        });
        let evicted = lock.with(|store| {
            store
                .as_mut()
                .map(|store| store.finish_read(&key, bytes, decoded.as_ref()))
        });
        drop(evicted);
    }
}

/// The reads of the pack file the writer's merges ask for
/// ([`pack::FileReads`]), on the staged partition: those in flight by id.
#[derive(Default)]
struct WriterReads {
    flights: Vec<(u64, FileRead)>,
}

impl WriterReads {
    /// Take the reads that ended and start those asked while the disk has
    /// room, never waiting for it.
    fn step(
        &mut self,
        device: &mut dyn BlockDevice,
        extents: &[Extent],
        reads: &mut pack::FileReads,
    ) {
        let mut at = 0;
        while at < self.flights.len() {
            if self.flights[at].1.poll(device) {
                let (id, read) = self.flights.swap_remove(at);
                reads.finish(id, read.bytes().map(<[u8]>::to_vec));
            } else {
                at += 1;
            }
        }
        let depth = device.depth().clamp(1, MAX_FLIGHTS);
        let mut started = false;
        while self.flights.len() < depth && device.room() {
            let Some((id, offset, bytes)) = reads.take_asked() else {
                break;
            };
            match FileRead::start(device, extents, offset, bytes) {
                Some(read) => {
                    started = true;
                    self.flights.push((id, read));
                }
                None => reads.finish(id, None),
            }
        }
        if started {
            device.commit();
        }
    }
}

/// Report the reader: the records left in the file it read, the time a
/// record took (mean and longest), whether lookups stopped asking a slow
/// disk, and the reads it gave up.
fn report_reader(counters: &pack::Counters) {
    let Some((latency, longest, slow, failed, launched, dropped)) = READER.with(|reader| {
        (
            reader.latency,
            reader.longest,
            reader.slow(),
            reader.failed,
            reader.launched,
            reader.dropped,
        )
    }) else {
        return;
    };
    let micros = |ticks: u64| ticks * 1_000_000 / super::host::COUNTER_HZ;
    output::arguments(format_args!(
        "TRANSLATIONS reader awaited={} read={} read_mb={} ahead={} launched={launched} \
         dropped={dropped} mean_us={} longest_us={} slow={slow} failed={failed} \
         awaited_pages={} awaited_lookups={} awaited_found={}\n",
        counters.awaited,
        counters.read,
        counters.read_bytes >> 20,
        counters.read_ahead,
        micros(latency),
        micros(longest),
        COSTS.awaited_pages.load(Ordering::Relaxed),
        COSTS.awaited_lookups.load(Ordering::Relaxed),
        COSTS.awaited_found.load(Ordering::Relaxed),
    ));
}

/// At a quiescent point (the guest restarts or powers off): the writer's
/// stop ([`Writer::stop`]), carried out on the partition directly.
pub fn write() {
    AVAILABLE.capabilities.store(0, Ordering::SeqCst);
    AVAILABLE.shared_bound.store(0, Ordering::Release);
    let started = output::now();
    let held = WRITER.holding(true, Option::take).flatten();
    let writable = BOOT.with(|boot| boot.as_ref().is_some_and(|boot| boot.partition.is_some()));
    if let Some(mut held) = held.filter(|_| writable) {
        let stats = held.writer.stats();
        output::arguments(format_args!(
            "TRANSLATIONS writer steps={} longest_us={} segments={} records={} bytes={} \
             pressed={} held={} pairs={} merges={} sheds={} resident_peak={}M released={} \
             merge_reads={} merge_read_mb={} abandoned={}\n",
            held.steps,
            held.longest * 1_000_000 / super::host::COUNTER_HZ,
            stats.deltas,
            stats.delta_records,
            stats.delta_bytes,
            stats.pressed,
            stats.held,
            stats.pairs,
            stats.merges,
            stats.sheds,
            stats.resident_peak >> 20,
            held.writer.shedding().1,
            stats.file_reads,
            stats.file_read_bytes >> 20,
            stats.abandoned,
        ));
        let name = pack_name(&translations::translator_id());
        let stopped = held.writer.stop(&mut Parts, &mut |execution| {
            with_store_file(&name, |device, extents| {
                let mut target = Direct { device, extents };
                while !execution.advance(&mut target, writer::PIECE_BYTES)? {}
                Ok::<(), String>(())
            })
            .and_then(|done| done)
        });
        match stopped {
            Ok(stopped) => output::arguments(format_args!(
                "TRANSLATIONS written segments={} lists={} records={} lost={} bytes={} \
                 merge_left={} ms={}\n",
                stopped.segments,
                stopped.lists,
                stopped.records,
                stopped.lost,
                stopped.bytes,
                stopped.merge_left,
                output::millis_since(started)
            )),
            Err(error) => output::arguments(format_args!("TRANSLATIONS not written: {error}\n")),
        }
    }
    let mut parts = STORES.iter().filter_map(|lock| lock.with(Option::take));
    if let Some(mut store) = parts.next() {
        for part in parts {
            store.absorb(part);
        }
        report(&store.counters);
    }
    let boot = BOOT.with(Option::take);
    drop(boot);
}
