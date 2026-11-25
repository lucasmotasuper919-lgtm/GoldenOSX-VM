//! The engine heap's large blocks and the code that made them: each live
//! allocation of at least `heap::TRACKED_BYTES` (`PROFILE_BYTES` in a
//! profile), with the image offsets on its allocating stack, for `HEAP
//! LARGE` lines that `stela.pdb` resolves. The heap calls
//! [`allocated_or_freed`] only for such blocks, a few hundred live at most
//! (a few thousand in a profile), so the cost stays off every ordinary
//! allocation. A profile also samples the smaller blocks, one in each
//! [`SAMPLE_EVERY`] bytes a core allocates, for `HEAP SAMPLED` lines that
//! estimate what holds the size classes and the medium blocks.
use crate::output;
use alloc::{format, string::String, vec::Vec};
use core::fmt::Write;
use core::sync::atomic::{AtomicPtr, AtomicU16, AtomicU64, Ordering};
use stela_boot_core::stack_scan;

/// Live large blocks kept; past them a block is only counted. A profile's
/// blocks from [`PROFILE_BYTES`] are about 1800 in a desktop session.
const SLOTS: usize = 4096;
/// Slots a block's search visits from its hash before it is only counted.
const PROBES: usize = 64;
/// A slot whose block was freed: a search goes past it, a new block takes it.
const VACATED: u64 = 1;
/// The least block a profile tracks: past the size classes' slabs, the
/// bucket of the block records' chunks, which hold most of the heap's
/// free-list bytes beside the larger blocks.
pub(super) const PROFILE_BYTES: usize = 128 << 10;
/// Image offsets kept from each allocating stack.
const WORDS: usize = 12;
/// The stack words searched for them.
const SCANNED: usize = 512;
/// Stacks the report names, the most bytes first; the rest are summed.
const SHOWN: usize = 32;

struct Slot {
    /// The payload's address; zero for a slot never taken, [`VACATED`] for
    /// one whose block was freed.
    pointer: AtomicU64,
    bytes: AtomicU64,
    words: [AtomicU64; WORDS],
}

static BLOCKS: [Slot; SLOTS] = [const {
    Slot {
        pointer: AtomicU64::new(0),
        bytes: AtomicU64::new(0),
        words: [const { AtomicU64::new(0) }; WORDS],
    }
}; SLOTS];

/// `pointer`'s hash: payloads are 16-byte aligned, the multiply spreads the
/// rest into the high bits.
fn hash(pointer: u64) -> u64 {
    (pointer >> 4).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// The slots of `table` a search for `pointer` visits, from its hash.
fn probes(table: &'static [Slot], pointer: u64) -> impl Iterator<Item = &'static Slot> {
    let start = (hash(pointer) >> 40) as usize;
    (0..PROBES).map(move |step| &table[(start + step) % table.len()])
}

/// The bytes each core allocates between sampled blocks in a profile: in a
/// desktop session's 2 GiB of size-class and medium blocks about 8000 are
/// live, each standing for this many bytes.
const SAMPLE_EVERY: usize = 256 << 10;
/// Live sampled blocks kept.
const SAMPLE_SLOTS: usize = 16384;
/// Buckets of the count of live samples by hash, which a free of a block
/// that was not sampled, nearly every free, reads alone.
const FILTER: usize = 1 << 16;

/// A profile's sampled blocks: their slots and the count in each bucket of
/// their hashes. Allocated by [`track_profile`], so a boot without a
/// profile holds none of it.
struct Samples {
    slots: [Slot; SAMPLE_SLOTS],
    filter: [AtomicU16; FILTER],
}

static SAMPLES: AtomicPtr<Samples> = AtomicPtr::new(core::ptr::null_mut());
/// Sampled blocks made while their slots were taken.
static SAMPLES_UNTRACKED: AtomicU64 = AtomicU64::new(0);

fn samples() -> Option<&'static Samples> {
    // SAFETY: only `track_profile` stores a nonzero value, a zeroed
    // allocation of `Samples` that is never freed.
    unsafe { SAMPLES.load(Ordering::Acquire).as_ref() }
}

fn filter_bucket(samples: &'static Samples, pointer: u64) -> &'static AtomicU16 {
    &samples.filter[(hash(pointer) >> 48) as usize % FILTER]
}

/// Tracks blocks from [`PROFILE_BYTES`] on and samples those under it, for
/// a profile's report of what holds the heap: before the engine's own
/// allocations.
pub(super) fn track_profile() {
    stela_boot_core::heap::set_tracked_bytes(PROFILE_BYTES);
    let layout = core::alloc::Layout::new::<Samples>();
    // SAFETY: a nonzero layout; atomics of zero are valid `Samples`.
    let table = unsafe { alloc::alloc::alloc_zeroed(layout) }.cast::<Samples>();
    if table.is_null() {
        return;
    }
    SAMPLES.store(table, Ordering::Release);
    stela_boot_core::heap::set_sample_hook(sampled_or_freed, SAMPLE_EVERY);
}

/// The heap's sample hook, under the same rules as [`allocated_or_freed`].
fn sampled_or_freed(pointer: usize, bytes: usize, allocated: bool) {
    let Some(samples) = samples() else {
        return;
    };
    let pointer = pointer as u64;
    let bucket = filter_bucket(samples, pointer);
    if !allocated {
        if bucket.load(Ordering::Relaxed) != 0 && forget(&samples.slots, pointer) {
            bucket.fetch_sub(1, Ordering::Relaxed);
            CHANGES.fetch_add(1, Ordering::Relaxed);
        }
        return;
    }
    if record(&samples.slots, pointer, bytes) {
        bucket.fetch_add(1, Ordering::Relaxed);
        CHANGES.fetch_add(1, Ordering::Relaxed);
    } else {
        SAMPLES_UNTRACKED.fetch_add(1, Ordering::Relaxed);
    }
}
/// Large blocks made while every slot was taken, which no line names: a
/// count of blocks made, not of live ones.
static UNTRACKED: AtomicU64 = AtomicU64::new(0);
/// Large blocks whose stack scan found no code offset.
static EMPTY: AtomicU64 = AtomicU64::new(0);
/// Blocks made or freed so far, and the count the last report saw.
static CHANGES: AtomicU64 = AtomicU64::new(0);
static REPORTED: AtomicU64 = AtomicU64::new(0);
/// Heartbeats between reports, and the reports asked for so far.
const REPORT_CALLS: u64 = 6;
static CALLS: AtomicU64 = AtomicU64::new(0);

/// Lets the heap tell this module of its large blocks.
pub(super) fn install() {
    stela_boot_core::heap::set_large_hook(allocated_or_freed);
}

/// The heap's large-block hook. It runs on any processor, outside the
/// heap's lock, so it neither allocates nor waits.
fn allocated_or_freed(pointer: usize, bytes: usize, allocated: bool) {
    let pointer = pointer as u64;
    CHANGES.fetch_add(1, Ordering::Relaxed);
    if !allocated {
        forget(&BLOCKS, pointer);
    } else if !record(&BLOCKS, pointer, bytes) {
        UNTRACKED.fetch_add(1, Ordering::Relaxed);
    }
}

/// Frees `pointer`'s slot in `table`, if it has one. The heap calls this
/// before the block can be handed out again. A search ends at a slot never
/// taken: no block went past it.
fn forget(table: &'static [Slot], pointer: u64) -> bool {
    let Some(slot) = probes(table, pointer)
        .take_while(|slot| slot.pointer.load(Ordering::Relaxed) != 0)
        .find(|slot| slot.pointer.load(Ordering::Relaxed) == pointer)
    else {
        return false;
    };
    slot.bytes.store(0, Ordering::Relaxed);
    slot.pointer.store(VACATED, Ordering::Release);
    true
}

/// Takes a slot of `table` for the block at `pointer` standing for `bytes`,
/// with its allocating stack; false with every slot it may take taken.
fn record(table: &'static [Slot], pointer: u64, bytes: usize) -> bool {
    let Some(slot) = probes(table, pointer).find(|slot| {
        [0, VACATED].into_iter().any(|empty| {
            slot.pointer
                .compare_exchange(empty, pointer, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
        })
    }) else {
        return false;
    };
    let mut words = [0; WORDS];
    if !stack_offsets(&mut words) {
        EMPTY.fetch_add(1, Ordering::Relaxed);
    }
    for (word, offset) in slot.words.iter().zip(words) {
        word.store(offset, Ordering::Relaxed);
    }
    slot.bytes.store(bytes as u64, Ordering::Release);
    true
}

/// The image offsets among the allocating stack's words, for its slot.
/// Whether any was found.
fn stack_offsets(out: &mut [u64; WORDS]) -> bool {
    stack_code_offsets(SCANNED, out) != 0
}

/// Fills `out` with the image offsets of code among this stack's words from
/// the stack pointer up, at most `scanned` words, for the count kept: the
/// stack pointer's own page, which is mapped, then the pages above through a
/// read a fault ends ([`stela_boot_core::stack_scan`]), as a stack may end
/// within a page of its pointer, below an absent guard page.
///
/// Only for hooks the engine runs after its host fault handler is active,
/// as the heap's large-block and peak hooks are.
pub(in crate::engine) fn stack_code_offsets(scanned: usize, out: &mut [u64]) -> usize {
    unsafe extern "C" {
        static __ImageBase: u8;
    }
    let base = &raw const __ImageBase as u64;
    // SAFETY: the mapped PE headers of this image: `e_lfanew`, then the
    // optional header's SizeOfCode and BaseOfCode.
    let (code, size) = unsafe {
        let header = u64::from((base as *const u32).byte_add(0x3c).read_unaligned());
        let pe = (base + header) as *const u32;
        (
            u64::from(pe.byte_add(0x2c).read_unaligned()),
            u64::from(pe.byte_add(0x1c).read_unaligned()),
        )
    };
    let stack: u64;
    // SAFETY: reads the stack pointer.
    unsafe {
        core::arch::asm!("mov {}, rsp", out(reg) stack, options(nomem, nostack, preserves_flags));
    }
    let mut kept = 0;
    let read = |address: u64, how| match how {
        // SAFETY: within the stack pointer's own page, which is mapped.
        stack_scan::Read::Own => Some(unsafe { (address as *const u64).read() }),
        stack_scan::Read::Guarded => {
            let mut value = 0u64;
            // SAFETY: the engine's host fault handler is active (this
            // function's contract), and `value` is this frame's own; a fault
            // ends the read with bytes unread.
            let unread = unsafe {
                stela_boot_core::nofault_read::read(
                    (&raw mut value).cast(),
                    address as *const u8,
                    8,
                )
            };
            (unread == 0).then_some(value)
        }
    };
    stack_scan::scan(stack, scanned, read, |value| {
        let offset = value.wrapping_sub(base);
        if offset.wrapping_sub(code) < size {
            out[kept] = offset;
            kept += 1;
        }
        kept < out.len()
    });
    kept
}

/// `HEAP LARGE blocks=31 bytes=612M stacks=12 untracked=0 empty_stacks=0`,
/// then the live blocks of each allocating stack, the most bytes first, as
/// `HEAP LARGE 64M blocks=16 stack 0x… 0x…` (the first [`SHOWN`], then
/// `HEAP LARGE 3M blocks=40 other stacks=9`): every [`REPORT_CALLS`]th
/// call, when blocks were made or freed since the last report.
pub(super) fn report() {
    if !CALLS
        .fetch_add(1, Ordering::Relaxed)
        .is_multiple_of(REPORT_CALLS)
    {
        return;
    }
    let changes = CHANGES.load(Ordering::Relaxed);
    if REPORTED.swap(changes, Ordering::Relaxed) == changes {
        return;
    }
    let blocks = live(&BLOCKS);
    let stacks = by_stack(blocks.iter().copied());
    output::arguments(format_args!("{}\n", summary(&blocks, stacks.len())));
    print_stacks("LARGE", &stacks);
    if let Some(samples) = samples() {
        let blocks = live(&samples.slots);
        let stacks = by_stack(blocks.iter().copied());
        output::arguments(format_args!(
            "{}\n",
            sampled_summary(
                &blocks,
                stacks.len(),
                SAMPLES_UNTRACKED.load(Ordering::Relaxed)
            )
        ));
        print_stacks("SAMPLED", &stacks);
        for frame in by_frame(&blocks).iter().take(FRAMES_SHOWN) {
            output::arguments(format_args!(
                "HEAP SAMPLED FRAME {:#x} {}M blocks={}\n",
                frame.offset,
                frame.bytes >> 20,
                frame.blocks
            ));
        }
    }
}

/// Code offsets the report names by the sampled bytes under them.
const FRAMES_SHOWN: usize = 48;

/// One code offset's sampled blocks: those with it anywhere on their
/// allocating stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Frame {
    offset: u64,
    bytes: u64,
    blocks: u64,
}

/// The sampled bytes under each code offset, counted once a stack, the
/// most first: the owners a long tail of distinct stacks shares (a stack
/// of its own for each caller of a common allocating function), which no
/// single stack's line shows. `HEAP SAMPLED FRAME 0x12c03cb 210M
/// blocks=840`.
fn by_frame(blocks: &[(u64, [u64; WORDS])]) -> Vec<Frame> {
    let mut seen: Vec<(u64, u64)> = Vec::with_capacity(blocks.len() * WORDS);
    for (bytes, words) in blocks {
        for (at, &offset) in words.iter().enumerate() {
            if offset != 0 && !words[..at].contains(&offset) {
                seen.push((offset, *bytes));
            }
        }
    }
    seen.sort_unstable_by_key(|&(offset, _)| offset);
    let mut frames: Vec<Frame> = Vec::new();
    for (offset, bytes) in seen {
        match frames.last_mut() {
            Some(frame) if frame.offset == offset => {
                frame.bytes += bytes;
                frame.blocks += 1;
            }
            _ => frames.push(Frame {
                offset,
                bytes,
                blocks: 1,
            }),
        }
    }
    frames.sort_unstable_by_key(|frame| (core::cmp::Reverse(frame.bytes), frame.offset));
    frames
}

/// The live blocks of `table`: the bytes each stands for and its stack.
fn live(table: &'static [Slot]) -> Vec<(u64, [u64; WORDS])> {
    table
        .iter()
        .filter(|slot| slot.pointer.load(Ordering::Acquire) > VACATED)
        .map(|slot| {
            let bytes = slot.bytes.load(Ordering::Acquire);
            (
                bytes,
                core::array::from_fn(|at| slot.words[at].load(Ordering::Relaxed)),
            )
        })
        .filter(|&(bytes, _)| bytes != 0)
        .collect()
}

/// The first [`SHOWN`] of `stacks` as `HEAP {kind}` lines, then the rest
/// summed.
fn print_stacks(kind: &str, stacks: &[Stack]) {
    for stack in stacks.iter().take(SHOWN) {
        output::arguments(format_args!("{}\n", line(kind, stack)));
    }
    if let Some(rest) = stacks.get(SHOWN..).filter(|rest| !rest.is_empty()) {
        output::arguments(format_args!(
            "HEAP {kind} {}M blocks={} other stacks={}\n",
            rest.iter().map(|stack| stack.bytes).sum::<u64>() >> 20,
            rest.iter().map(|stack| stack.blocks).sum::<u64>(),
            rest.len()
        ));
    }
}

/// `HEAP SAMPLED blocks=8012 bytes=2003M every=256K stacks=410
/// untracked=0`: the sampled blocks under [`PROFILE_BYTES`] live now, the
/// bytes they stand for (an estimate of the live bytes of such blocks), and
/// the stacks that made them.
fn sampled_summary(blocks: &[(u64, [u64; WORDS])], stacks: usize, untracked: u64) -> String {
    let bytes: u64 = blocks.iter().map(|&(bytes, _)| bytes).sum();
    format!(
        "HEAP SAMPLED blocks={} bytes={}M every={}K stacks={stacks} untracked={untracked}",
        blocks.len(),
        bytes >> 20,
        SAMPLE_EVERY >> 10,
    )
}

/// One allocating stack's live blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Stack {
    bytes: u64,
    blocks: u64,
    words: [u64; WORDS],
}

/// The blocks grouped by allocating stack, the most bytes first.
fn by_stack(blocks: impl Iterator<Item = (u64, [u64; WORDS])>) -> Vec<Stack> {
    let mut stacks: Vec<Stack> = Vec::new();
    for (bytes, words) in blocks {
        match stacks.iter_mut().find(|stack| stack.words == words) {
            Some(stack) => {
                stack.bytes += bytes;
                stack.blocks += 1;
            }
            None => stacks.push(Stack {
                bytes,
                blocks: 1,
                words,
            }),
        }
    }
    stacks.sort_unstable_by_key(|stack| core::cmp::Reverse(stack.bytes));
    stacks
}

fn summary(blocks: &[(u64, [u64; WORDS])], stacks: usize) -> String {
    let bytes: u64 = blocks.iter().map(|&(bytes, _)| bytes).sum();
    format!(
        "HEAP LARGE blocks={} bytes={}M stacks={stacks} untracked={} empty_stacks={}",
        blocks.len(),
        bytes >> 20,
        UNTRACKED.load(Ordering::Relaxed),
        EMPTY.load(Ordering::Relaxed)
    )
}

fn line(kind: &str, stack: &Stack) -> String {
    let mut line = format!(
        "HEAP {kind} {}M blocks={} stack",
        stack.bytes >> 20,
        stack.blocks
    );
    for &offset in stack.words.iter().filter(|&&offset| offset != 0) {
        let _ = write!(line, " {offset:#x}");
    }
    line
}
