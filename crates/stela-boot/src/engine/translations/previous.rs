//! The previous translator's pack, for a new translator's first boot to
//! refill its store from (`stela_engine::machine::engine::refill`).
//!
//! `stela` keeps the most recently written pack when Stela is updated, and
//! the new translator's first boot takes the spare file as its own, blank.
//! The loader then reads the previous pack's segments, the newest first,
//! within [`PREVIOUS_SHARE`] of the pack's budget, into a handoff file of
//! their own ([`FILE`]); the arena the boot writes its own segments to
//! gives up as much, so the boot holds no more than its budget. The engine
//! takes the records hottest first on PEs whose guest waits, in bounded
//! steps after the blocks prepared ahead, while the board's kept forms
//! have room and the heap is not short; then it ends. The pack's pages,
//! outside the heap, stay the boot's, within the budget as said.
use super::{AHEAD, AHEAD_MICROS, DIRECTORY, DOMAIN, Lock, PACK_EXTENSION, SEGMENT};
use super::{open, pack_name, read_at, read_exact};
use crate::{firmware, handoff::Region, output, paging::PAGE};
use alloc::{format, string::String, vec, vec::Vec};
use core::sync::atomic::{AtomicBool, Ordering};
use stela_engine::machine::engine::refill::{Refill, RefillCounts};
use stela_engine::translations::pack::Pack;
use stela_engine::translations::{Digest, Record};
use stela_formats::journal::{self, DATA_START};
use uefi::proto::media::file::{Directory, File as _, FileAttribute, FileMode, RegularFile};

/// The handoff file holding the previous translator's segments.
pub const FILE: &str = "translations-previous";
/// The part of the pack's budget the previous pack may take: the boot's own
/// arena keeps the rest.
pub const PREVIOUS_SHARE: u64 = 2;
/// The handoff file's header: magic, the previous translator's id and the
/// segment count; each segment's length follows, then the segments, oldest
/// first, each from an 8-byte boundary.
const MAGIC: [u8; 8] = *b"STLTPREV";
const HEADER_BYTES: usize = 8 + 32 + 8;

/// Whether `file`, of `size` bytes, commits anything for `translator`: a
/// pack that does not is blank, its translator's first boot.
/// The file's position is its start again after.
pub(super) fn committed(file: &mut RegularFile, translator: &Digest, size: u64) -> bool {
    let mut slots = vec![0; DATA_START as usize];
    let committed = size >= DATA_START
        && read_at(file, 0, &mut slots).is_some()
        && journal::candidates(&slots, DOMAIN, size)
            .first()
            .is_some_and(|commit| commit.key == translator.0 && !commit.entries.is_empty());
    // A file that does not seek reads as blank to the journal too.
    let _ = file.set_position(0);
    committed
}

/// The newest segments of another translator's pack in the store's
/// directory, within `room` bytes, as the handoff file's region and the
/// bytes it holds: `None` when no other pack commits a segment that fits.
pub(super) fn read(root: &mut Directory, translator: &Digest, room: u64) -> Option<(Region, u64)> {
    let ours = pack_name(translator);
    let path = uefi::CString16::try_from(DIRECTORY).ok()?;
    let mut directory = root
        .open(&path, FileMode::Read, FileAttribute::empty())
        .ok()?
        .into_directory()?;
    while let Ok(Some(info)) = directory.read_entry_boxed() {
        let name = format!("{}", info.file_name());
        let lower = name.to_ascii_lowercase();
        if !lower.ends_with(&format!(".{PACK_EXTENSION}")) || lower == ours {
            continue;
        }
        let Some(mut file) = open(root, &format!("{DIRECTORY}\\{name}"), FileMode::Read) else {
            continue;
        };
        if let Some(read) = segments(&mut file, translator, info.file_size(), room) {
            output::arguments(format_args!(
                "TRANSLATIONS previous {name} pack={} for a refill\n",
                read.1
            ));
            return Some(read);
        }
    }
    None
}

/// The newest committed segments of `file`, a pack of another translator
/// than `translator`, within `room`.
fn segments(
    file: &mut RegularFile,
    translator: &Digest,
    size: u64,
    room: u64,
) -> Option<(Region, u64)> {
    let mut slots = vec![0; DATA_START as usize];
    if size < DATA_START {
        return None;
    }
    read_exact(file, &mut slots)?;
    let commit = journal::candidates(&slots, DOMAIN, size)
        .into_iter()
        .next()?;
    drop(slots);
    if commit.key == translator.0 {
        return None;
    }
    let mut left = room;
    let mut held: Vec<&journal::Entry> = Vec::new();
    for entry in commit.entries.iter().rev() {
        if entry.kind != SEGMENT {
            continue;
        }
        if entry.length > left {
            break;
        }
        left -= entry.length;
        held.push(entry);
    }
    if held.is_empty() {
        return None;
    }
    held.reverse();
    let blobs: usize = held
        .iter()
        .map(|entry| (entry.length as usize).next_multiple_of(8))
        .sum();
    let head = HEADER_BYTES + 8 * held.len();
    let bytes = head + blobs;
    let physical = firmware::pages(bytes.div_ceil(PAGE)).ok()?;
    // SAFETY: firmware returned this exclusive allocation of `bytes` or
    // more. Initialize it before taking a mutable slice.
    let region = unsafe {
        let start = physical as *mut u8;
        start.write_bytes(0, bytes);
        core::slice::from_raw_parts_mut(start, bytes)
    };
    region[..8].copy_from_slice(&MAGIC);
    region[8..40].copy_from_slice(&commit.key);
    region[40..44].copy_from_slice(&(held.len() as u32).to_le_bytes());
    let mut at = head;
    for (index, entry) in held.iter().enumerate() {
        let length = entry.length as usize;
        if read_at(file, entry.offset, &mut region[at..at + length]).is_none() {
            // A segment that does not read whole is left out.
            continue;
        }
        let field = HEADER_BYTES + 8 * index;
        region[field..field + 8].copy_from_slice(&(length as u64).to_le_bytes());
        at += length.next_multiple_of(8);
    }
    let read = (at - head) as u64;
    Some((
        Region {
            physical: physical as u64,
            bytes: bytes as u64,
        },
        read,
    ))
}

/// The refill while its records last, and whether a PE takes a step of it.
static REFILL: Lock<Option<Refill<'static>>> = Lock::new(None);
static STEPPING: AtomicBool = AtomicBool::new(false);

/// Start the refill from the handoff file the loader read ([`read`]).
pub fn install(contents: &'static mut [u8]) {
    let contents: &'static [u8] = contents;
    if contents.len() < HEADER_BYTES || contents[..8] != MAGIC {
        return;
    }
    let translator = Digest(contents[8..40].try_into().expect("an id"));
    let count = u32::from_le_bytes(contents[40..44].try_into().expect("a count")) as usize;
    let head = HEADER_BYTES + 8 * count;
    if contents.len() < head {
        return;
    }
    let mut segments = Vec::new();
    let mut at = head;
    for index in 0..count {
        let field = HEADER_BYTES + 8 * index;
        let length =
            u64::from_le_bytes(contents[field..field + 8].try_into().expect("a length")) as usize;
        if length == 0 || at + length > contents.len() {
            continue;
        }
        if let Ok(pack) = Pack::parse(&contents[at..at + length], &translator) {
            segments.push(pack);
        }
        at += length.next_multiple_of(8);
    }
    if segments.is_empty() {
        return;
    }
    let refill = Refill::new(translator, segments);
    output::arguments(format_args!(
        "TRANSLATIONS refill from {} records={}\n",
        hex(&translator.0[..4]),
        refill.left()
    ));
    REFILL.with(|held| *held = Some(refill));
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// A step of the refill on a PE whose guest waits, after the blocks
/// prepared ahead: records hottest first, each compiled by `compile`
/// (`Machine::refill`), for at most [`AHEAD_MICROS`] or until `stop` says
/// the guest has work. Without `room` (the board's kept forms half full,
/// or the heap short) the refill ends; so it does when no record is left.
/// One PE takes a step at a time; the others go on.
pub fn step(
    room: bool,
    compile: &mut dyn FnMut(&Digest, &Record, &mut RefillCounts),
    stop: &dyn Fn() -> bool,
) {
    if REFILL.with(|held| held.is_none())
        || AHEAD.with(|queue| queue.as_ref().is_some_and(|queue| !queue.is_empty()))
        || STEPPING.swap(true, Ordering::Acquire)
    {
        return;
    }
    if !room {
        finish("room");
        STEPPING.store(false, Ordering::Release);
        return;
    }
    let started = crate::engine::host::fresh_counter();
    let deadline = started + crate::engine::host::COUNTER_HZ * AHEAD_MICROS / 1_000_000;
    while crate::engine::host::fresh_counter() < deadline && !stop() {
        let Some(Some((key, record))) = REFILL.with(|held| held.as_mut().map(Refill::next_record))
        else {
            finish("done");
            break;
        };
        let mut counts = RefillCounts::default();
        compile(&key, &record, &mut counts);
        REFILL.with(|held| {
            if let Some(refill) = held {
                let total = refill.counts_mut();
                total.entries += counts.entries;
                total.kept += counts.kept;
                total.unmatched += counts.unmatched;
                total.refused += counts.refused;
            }
        });
    }
    STEPPING.store(false, Ordering::Release);
}

/// End the refill, saying why and what it did.
fn finish(reason: &str) {
    if let Some(refill) = REFILL.with(Option::take) {
        report(&refill, reason);
    }
}

/// The refill's counts, at its end or the run's.
pub fn report_now() {
    REFILL.with(|held| {
        if let Some(refill) = held {
            report(refill, "running");
        }
    });
}

fn report(refill: &Refill<'_>, state: &str) {
    let counts = refill.counts();
    output::arguments(format_args!(
        "TRANSLATIONS refill {state} records={} undecoded={} left={} entries={} kept={} \
         unmatched={} refused={}\n",
        counts.records,
        counts.undecoded,
        refill.left(),
        counts.entries,
        counts.kept,
        counts.unmatched,
        counts.refused,
    ));
}
