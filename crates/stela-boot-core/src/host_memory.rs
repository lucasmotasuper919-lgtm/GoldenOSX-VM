//! The parts of Stela's memory sized from the host RAM free at boot (the
//! engine heap, native code arena and shadow page-table pool), the share
//! left for the guest, and what is left of it for the translation pack.
use crate::Result;
use crate::native_resources::{
    CODE_ALIGNMENT, CodeReservation, MAX_WINDOW, SMALL_WINDOW_PER_RUNNER, WINDOW_PER_RUNNER,
    code_reservation_within, parked_reserve, window_share,
};
use stela_contracts::refuse;

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;
/// Each part's alignment: the large page the engine maps it with.
const ALIGNMENT: u64 = 2 * MIB;
/// What the heap needs does not shrink with the RAM, which bounds it from
/// below: a 26A434 desktop boot holds about 1.5 GiB live when WindowServer
/// compiles Core Animation's generic Vfx shader (`fixed_frag_lph_cpf` with
/// `fc_generic`, 42 529 NAK instructions in about 4800 blocks), whose
/// per-block liveness takes NAK another 450 MiB for the compile. That
/// peaked at 2150 MiB, and a boot with a fuller translation pack ran out of
/// a 2304 MiB heap there.
pub const HEAP_FLOOR: u64 = 1408 * MIB;
/// Above the floor, the heap grows with the host to this: a larger host
/// runs more and larger pipelines, apps and translation packs.
pub const HEAP_CEILING: u64 = 16 * GIB;
/// Free host RAM bytes per byte of the engine heap's base share, which
/// passes the floor on hosts with more than about 16 GiB free. The floor
/// is what a boot was measured to need, with little to spare: a PC with 30
/// GiB free ran its first boot after an install on the floor (3072 MiB with
/// a 784 MiB window), carved 2673 MiB of it and failed an allocation with
/// 700 MiB free in extents of at most 192 MiB, while its guest held 25 GiB.
/// An eighth gives that host 4.5 GiB of heap and takes a twentieth of the
/// guest's DRAM; hosts near the guest's floor keep the heap's floor.
const HEAP_SHARE: u64 = 8;
/// The heap a desktop session needs, with room for the large allocations a
/// fragmented heap must still place, at [`FLOOR_WINDOW`]. A macOS desktop
/// holds 2.3 to 2.7 GiB live from about 80 s on, whatever the host, and
/// carved at most 2924 MiB on a 16 GiB PC and 3168 MiB on a 32 GiB PC, both
/// with a 768 MiB window. The 16 GiB PC ran on the floor (3072 MiB): with
/// 450 MiB free in extents of at most 16 MiB it failed a 17 609 526 byte
/// allocation, and an earlier boot a 267 386 880 byte one with 192 MiB
/// extents. The 32 GiB PC's 4666 MiB kept extents of 1.2 GiB. Hosts between
/// the floor and their share take the heap up to this from the guest, as
/// far as [`GUEST_ROOM`] allows. A 16 GiB PC's 8-hour session on its 4352
/// MiB (the target at a 768 MiB window) held 2.8-3.3 GiB live in 1.3-1.8
/// million free extents, the largest 194-272 MiB, and peaked at 4063 MiB
/// when NAK compiled Core Animation's generic shader 8 hours in: defined
/// at that window, the target would leave such a session 33 MiB.
const HEAP_TARGET: u64 = 4 * GIB;
/// The guest DRAM [`HEAP_TARGET`] leaves at least: twice [`GUEST_FLOOR`].
/// A guest under it keeps the heap's share or floor, so 8 GiB hosts boot
/// with what they had.
const GUEST_ROOM: u64 = 2 * GUEST_FLOOR;
/// Heap bytes per byte of the board's code window: the shared block records,
/// successors, recovery rows, index and lookup of the blocks it holds,
/// once for the board since the block store is shared. An 8-PE 26A434
/// boot with a full 448 MiB window (1.56 million of 1.69 million slots
/// live) held 437 MiB of them (`ENGINE MEMORY blocks=231M successors=53M
/// rows=14M index=19M lookup=120M`), 0.98 bytes per window byte. Before
/// the shared store each core kept its own, 2.9 bytes per byte.
const HEAP_PER_WINDOW: u64 = 1;
/// The code window [`HEAP_FLOOR`] was measured with. The floor moves by
/// [`HEAP_PER_WINDOW`] per byte the window differs from it: its block
/// records come on top of the rest of the peak, which does not depend on
/// the window. A 12 GiB VM with a 704 MiB window peaked at 2815 MiB carved,
/// 2623 MiB less the 192 MiB of window past this one.
const FLOOR_WINDOW: u64 = 512 * MIB;
/// Heap each running PE keeps for itself: its machine (about 1 MiB at
/// start), ITLB (512 KiB), fetch and jump caches (about 0.8 MiB) and
/// successor scratch (under 1 MiB); four times that, for its own compiles.
const PE_HEAP: u64 = 16 * MIB;
/// Guest DRAM bytes per byte of a running executor's shadow page-table
/// pool: room for its 8 roots to map all of guest DRAM once in 4 KiB pages
/// (8 bytes per page). An exhausted half resets and refills everything; a
/// 26A434 restore's userland filled 8 MiB halves 20 to 80 times a beat.
const POOL_SHARE: u64 = 64;
/// Runners whose pools together map the guest's DRAM once each. Past them
/// the runners divide that total: each PE's tables are a cache of what it
/// touches, and more PEs divide the guest's working set between them.
const POOL_SHARERS: u64 = 8;
const POOL_FLOOR: u64 = 64 * MIB;
/// The least pool a running executor gets, a 16 MiB half, at any PE count.
const POOL_RUNNER_FLOOR: u64 = 32 * MIB;
/// Each sharer's pool maps at most 64 GiB of guest DRAM once; past that its
/// halves reset as a cache of the guest's working set, rather than hold a
/// tenth of a large host's RAM.
const POOL_PER_RUNNER: u64 = GIB;
/// Past small guests a runner's pool is its two halves' sizes, not a
/// sixty-fourth of the guest: [`LOWER_BASE`] and a [`LOWER_GROWTH`]th of the
/// guest's DRAM for the lower half, [`UPPER_BASE`] and an
/// [`UPPER_GROWTH`]th for the upper. The lower half (TTBR0, user address
/// spaces) is a cache of recent address spaces: it fills whatever it has,
/// and ran out about once in 1.6 heartbeats with 52 MiB as with 64 MiB (74
/// times in 117, 106 in 178), while TLB maintenance reset the shadow about
/// 190 times a heartbeat (`SHADOW exhausted= maintained=`); 48 MiB and a
/// 2048th holds about the 52 MiB of the 12 GiB run. The upper half (TTBR1,
/// the kernel) peaked at 22-29 MiB per runner whatever the guest's size
/// (`BEAT shadow_peak=52M/23M/52M`, `64M/22M/64M`, 29 MiB in a 4-hour 16
/// GiB run): 32 MiB and a 1024th, 42 MiB at 16 GiB.
const LOWER_BASE: u64 = 48 * MIB;
const LOWER_GROWTH: u64 = 2048;
const UPPER_BASE: u64 = 32 * MIB;
const UPPER_GROWTH: u64 = 1024;
/// The least guest DRAM Stela boots macOS with. Below it macOS spends its
/// time paging; 26A434 reaches the desktop with 3.1 GiB.
pub const GUEST_FLOOR: u64 = 1536 * MIB;
/// The step [`Shortfall::needed`] is found in.
const NEEDED_STEP: u64 = 256 * MIB;
/// Free host RAM bytes per byte of the persistent translation pack the
/// loader may hold: a cache, it takes no more than this share and only what
/// the parts and the guest's floor leave.
pub const PACK_SHARE: u64 = 16;
/// The least pack worth reading. Below it the loader leaves the pack on its
/// volume, unread and unwritten, for a PC with more RAM.
pub const PACK_FLOOR: u64 = 64 * MIB;
/// What the loader takes between measuring the free RAM for the pack and
/// reserving the parts, beyond the pack itself: the pack's placement page,
/// the storage drivers' rings and the firmware ranges' rounding to 2 MiB.
const PACK_MARGIN: u64 = 16 * MIB;

/// The engine heap's share (a fraction's denominator) that held compiled
/// code beside the window before one budget held both: the blocks PEs
/// share as they compile and the stored forms of blocks the ring took away
/// (`translations::share`).
pub const CODE_SHARE: u64 = 8;
/// The heap's share the pack's decoded records took (`translations::bounds`
/// before one budget held them).
pub const DECODED_SHARE: u64 = 64;
/// What of those shares code beside the window keeps at least: the blocks
/// shared at compile time took a quarter of the first, and the decoded
/// records keep what they had.
const KEPT_FLOOR_SHARE: u64 = 32;

/// The heap's code beside the window before the budget: the code share and
/// the decoded records' share.
fn kept_share(heap: u64) -> u64 {
    heap / CODE_SHARE + heap / DECODED_SHARE
}

/// The least of it the budget leaves.
fn kept_floor(heap: u64) -> u64 {
    heap / KEPT_FLOOR_SHARE + heap / DECODED_SHARE
}

/// The parts' sizes and the guest DRAM left after them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sizes {
    pub heap: u64,
    pub code: CodeReservation,
    /// The heap bytes for compiled code beside the window: blocks shared
    /// at compile time and the stored forms of blocks the window let go
    /// ([`code_budget`]).
    pub kept: u64,
    pub pool: u64,
    /// Free RAM after the three parts; Stela's stacks, tables and image
    /// take a few more MiB of it.
    pub guest: u64,
}

/// The DRAM the guest Mac reports and the host RAM it does not see.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuestDram {
    /// `/chosen dram-size` and `boot_args.memSizeActual`.
    pub reported: u64,
    /// Installed host RAM that Stela and the firmware keep.
    pub kept: u64,
}

/// The guest reports the DRAM that backs it, `backed` bytes, not the
/// host's `installed` RAM.
///
/// XNU sets `max_mem_actual` from `memSizeActual` and `max_mem` and
/// `sane_size` from `memSize` (the managed range). `hw.memsize` reads
/// `max_mem_actual`; `hw.memsize_usable` reads `max_mem`. User space
/// (`NSProcessInfo.physicalMemory`, Metal's working-set size, jetsam
/// limits, cache sizes) uses `hw.memsize`. A reported size above what
/// exists makes it plan for memory that is not there: a 26A434 boot with
/// 3.9 GB backed and 8 GB reported ran its compressor and pager at full
/// load after login. On a Mac the gap is iBoot's carveouts, a few percent;
/// here those carveouts lie inside the backed range (`memSize` excludes
/// them), so the backed size is the truthful `max_mem_actual`.
pub fn guest_dram(installed: u64, backed: u64) -> GuestDram {
    GuestDram {
        reported: backed,
        kept: installed.saturating_sub(backed),
    }
}

/// Too little RAM for the guest's floor: `free` bytes, of `needed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shortfall {
    pub free: u64,
    pub needed: u64,
}

impl Shortfall {
    /// The refusal the boot stops with, in whole GB rounded up, as RAM is
    /// sold, and the free RAM to a tenth.
    pub fn refusal(self) -> crate::Refusal {
        let needed = self.needed.div_ceil(GIB);
        let free = alloc::format!("{}.{}", self.free / GIB, self.free % GIB * 10 / GIB);
        let floor = GUEST_FLOOR / GIB;
        refuse!(
            "this PC has {free} GB of RAM free; Stela needs at least {needed} GB free: \
             {floor} GB for macOS and the rest for its own memory",
            free = %free,
            needed = %needed,
            floor = %floor
        )
    }
}

/// The sizes for `free` host bytes and `runners` of the `cores` executors
/// running PEs; the others add only their parked reserves.
///
/// Each part takes a share of `free`, between its bounds:
/// - the heap a twenty-fourth, [`HEAP_PER_WINDOW`] byte per byte of the board's
///   code window and [`PE_HEAP`] per runner, within
///   [`HEAP_FLOOR`]..=[`HEAP_CEILING`], the floor moved by the window's
///   difference from [`FLOOR_WINDOW`]; below [`HEAP_TARGET`] (moved the
///   same way), as much more as leaves the guest [`GUEST_ROOM`];
/// - the code arena each executor's own part and the board's window: free
///   RAM / 16 where the guest keeps [`GUEST_ROOM`] beside it and the heap's
///   target, as much of it as does on hosts between, and no less than
///   [`WINDOW_PER_RUNNER`] per runner, or [`SMALL_WINDOW_PER_RUNNER`] where
///   that would leave the guest under its floor;
/// - the pool, for each of up to [`POOL_SHARERS`] runners, 1/[`POOL_SHARE`]
///   of the guest's DRAM, which is what the other parts leave, or its halves'
///   sizes ([`pool_halves`]) if that is less: `pool = sharers * (free - heap -
///   code - pool) / 64`, solved for `pool`; at least [`POOL_RUNNER_FLOOR`] per
///   runner, at most [`POOL_PER_RUNNER`] per sharer.
///
/// Then the window takes what the code kept beside it in the heap held
/// ([`code_budget`]): the guest and the pool stay as decided.
///
/// A guest share under [`GUEST_FLOOR`] is a [`Shortfall`].
pub fn sizes(free: u64, cores: usize, runners: usize) -> Result<Sizes> {
    let estimate = estimate(free, cores, runners)?;
    if estimate.guest >= GUEST_FLOOR {
        return Ok(code_budget(estimate));
    }
    let mut needed = free.next_multiple_of(NEEDED_STEP);
    while estimate_guest(needed, cores, runners)? < GUEST_FLOOR {
        needed += NEEDED_STEP;
    }
    Err(Shortfall { free, needed }.refusal())
}

/// The code window per runner past which a larger one cuts reinstalls
/// steeply no more (the ring replay at 8 PEs: 816 MiB reinstalls 15 % of
/// what 640 MiB does): a window under it takes a pack's room first.
pub const KNEE_PER_RUNNER: u64 = 102 * MIB;
/// The code window per runner up to which the window shares a pack's room
/// with the guest (the replay at 8 PEs: 1999, 1021 and 449 reinstalls per
/// heartbeat at 816, 1200 and 1616 MiB); past it the room is the guest's.
pub const SHARED_PER_RUNNER: u64 = 202 * MIB;

/// [`sizes`] on a host where the translation pack left `room` of its
/// budget ([`pack_budget`]) unused, as a pack read on demand does: it holds
/// its index and the records lookups ask for, not every record. The other
/// parts are what they were while the pack held its budget, and the room
/// goes to the code window (with its block records, [`HEAP_PER_WINDOW`] a
/// window byte) and the guest ([`give_room`]).
pub fn sizes_with_room(free: u64, cores: usize, runners: usize, room: u64) -> Result<Sizes> {
    match sizes(free.saturating_sub(room), cores, runners) {
        Ok(sizes) => Ok(give_room(sizes, room, runners)),
        Err(_) => sizes(free, cores, runners),
    }
}

/// `sizes` with `room` beside them: the window takes it up to the knee
/// ([`KNEE_PER_RUNNER`] per runner), then half of what is left up to
/// [`SHARED_PER_RUNNER`] per runner, within [`MAX_WINDOW`], in whole
/// [`CODE_ALIGNMENT`]s; the guest takes the rest.
fn give_room(sizes: Sizes, room: u64, runners: usize) -> Sizes {
    let runners = runners.max(1) as u64;
    let window = sizes.code.window;
    let per_byte = 1 + HEAP_PER_WINDOW;
    let knee = (KNEE_PER_RUNNER * runners).min(MAX_WINDOW);
    let shared = (SHARED_PER_RUNNER * runners).min(MAX_WINDOW);
    let first = knee.saturating_sub(window).min(room / per_byte);
    let rest = room - first * per_byte;
    let second = shared
        .saturating_sub(window + first)
        .min(rest / 2 / per_byte);
    let grown = (first + second) / CODE_ALIGNMENT * CODE_ALIGNMENT;
    Sizes {
        heap: sizes.heap + HEAP_PER_WINDOW * grown,
        code: CodeReservation {
            bytes: sizes.code.bytes + grown,
            window: window + grown,
            ..sizes.code
        },
        guest: sizes.guest + room - grown * per_byte,
        ..sizes
    }
}

/// One budget for compiled code, the window's first: the window, its
/// block records ([`HEAP_PER_WINDOW`]) and the code kept beside it in the
/// heap (a [`CODE_SHARE`]th of the heap for blocks and stored forms, a
/// [`DECODED_SHARE`]th for the pack's decoded records) are what they were,
/// and of that the window takes as much as leaves the kept code a
/// [`KEPT_FLOOR_SHARE`]th of the heap and the decoded records their share,
/// up to the most a window takes. The heap gives the window's growth back (its records come in,
/// twice as much kept code goes out), so the guest keeps its DRAM.
///
/// Code the window holds needs no stored form to come back from: the
/// ring's tenured parts keep it. A window too small for what the guest
/// runs again and again turns it all over each heartbeat, and the kept
/// forms then turn over too (run 20261119-021855: 512 MiB of window and
/// 504 MiB of kept forms on a 12 GiB host, both every 10 s, kept forms
/// serving 25-190 thousand of 0.6-1.1 million installs a heartbeat).
fn code_budget(sizes: Sizes) -> Sizes {
    let window = sizes.code.window;
    let kept = kept_share(sizes.heap);
    let budget = 2 * HEAP_PER_WINDOW * window + kept;
    let floor = kept_floor(sizes.heap);
    let most = ((budget - floor) / (2 * HEAP_PER_WINDOW)).min(MAX_WINDOW) / CODE_ALIGNMENT
        * CODE_ALIGNMENT;
    if most <= window {
        return Sizes { kept, ..sizes };
    }
    let grown = most - window;
    Sizes {
        heap: sizes.heap - HEAP_PER_WINDOW * grown,
        code: CodeReservation {
            bytes: sizes.code.bytes + grown,
            window: most,
            ..sizes.code
        },
        kept: budget - 2 * HEAP_PER_WINDOW * most,
        ..sizes
    }
}

/// The bytes of persistent translation pack the loader may read for
/// `free` host bytes, measured just before it reads the pack, with
/// `runners` of `cores` executors running PEs: at most a
/// 1/[`PACK_SHARE`] of `free`, and only as much as still leaves [`sizes`]
/// the guest's floor, after the pack and a margin for the loader's other
/// allocations. Zero, below [`PACK_FLOOR`]: the parts and the guest come
/// first, and the pack never makes a boot fail.
pub fn pack_budget(free: u64, cores: usize, runners: usize) -> u64 {
    let mut budget = free / PACK_SHARE / ALIGNMENT * ALIGNMENT;
    while budget >= PACK_FLOOR {
        let left = free.saturating_sub(budget + PACK_MARGIN);
        if estimate_guest(left, cores, runners).is_ok_and(|guest| guest >= GUEST_FLOOR) {
            return budget;
        }
        budget -= ALIGNMENT;
    }
    0
}

fn estimate_guest(free: u64, cores: usize, runners: usize) -> Result<u64> {
    estimate(free, cores, runners).map(|sizes| sizes.guest)
}

/// The sizes with the code window the host's memory allows: the host's
/// share where the guest keeps its room beside it and the heap's target,
/// else the larger window per runner, or the smaller one where the larger
/// would leave the guest under its floor.
fn estimate(free: u64, cores: usize, runners: usize) -> Result<Sizes> {
    let per_runner = |each: u64| estimate_within(free, cores, runners, runners as u64 * each);
    let large = per_runner(WINDOW_PER_RUNNER)?;
    if large.guest < GUEST_FLOOR {
        return per_runner(SMALL_WINDOW_PER_RUNNER);
    }
    // The guest shrinks as the window grows, by the window and its block
    // records: the largest window up to the host's share that leaves the
    // guest its room with the heap on its target, in whole steps.
    let roomy = |window: u64| -> Result<bool> {
        let code = code_reservation_within(free, cores, runners, window)?;
        let shared = shared_heap(free, runners, code.window);
        let heap = shared.max(heap_target(shared, code.window));
        Ok(split(free, heap, code, cores, runners).guest >= GUEST_ROOM)
    };
    let (mut low, mut high) = (
        large.code.window / CODE_ALIGNMENT,
        window_share(free) / CODE_ALIGNMENT,
    );
    if low >= high || !roomy((low + 1) * CODE_ALIGNMENT)? {
        return Ok(large);
    }
    while low < high {
        let middle = (low + high).div_ceil(2);
        if roomy(middle * CODE_ALIGNMENT)? {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    estimate_within(free, cores, runners, low * CODE_ALIGNMENT)
}

/// The heap's target for a `window`, on a host whose share of heap is
/// `shared`: [`HEAP_TARGET`] moved by the window, within the ceiling.
fn heap_target(shared: u64, window: u64) -> u64 {
    at_window(HEAP_TARGET, window).min(HEAP_CEILING.max(shared))
}

/// The sizes with a code window of at most `most` bytes.
fn estimate_within(free: u64, cores: usize, runners: usize, most: u64) -> Result<Sizes> {
    let code = code_reservation_within(free, cores, runners, most)?;
    let shared = shared_heap(free, runners, code.window);
    let base = split(free, shared, code, cores, runners);
    let target = heap_target(shared, code.window);
    if target <= shared || base.guest <= GUEST_ROOM {
        return Ok(base);
    }
    // The guest shrinks as the heap grows: the most heap up to the target
    // that leaves the guest its room, in whole alignments.
    let fits = |heap: u64| split(free, heap, code, cores, runners).guest >= GUEST_ROOM;
    let (mut low, mut high) = (shared / ALIGNMENT, target / ALIGNMENT);
    while low < high {
        let middle = (low + high).div_ceil(2);
        if fits(middle * ALIGNMENT) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    Ok(split(free, low * ALIGNMENT, code, cores, runners))
}

/// `bytes` measured with [`FLOOR_WINDOW`], moved by [`HEAP_PER_WINDOW`] per
/// byte a `window` differs from it, in whole alignments.
fn at_window(bytes: u64, window: u64) -> u64 {
    (bytes + HEAP_PER_WINDOW * window).saturating_sub(HEAP_PER_WINDOW * FLOOR_WINDOW) / ALIGNMENT
        * ALIGNMENT
}

/// The heap's share of `free` for `runners` PEs and a `window`, between its
/// floor and ceiling.
fn shared_heap(free: u64, runners: usize, window: u64) -> u64 {
    let heap = free / HEAP_SHARE + HEAP_PER_WINDOW * window + runners as u64 * PE_HEAP;
    let floor = at_window(HEAP_FLOOR, window);
    (heap / ALIGNMENT * ALIGNMENT).clamp(floor, HEAP_CEILING.max(floor))
}

/// The pool and the guest for `heap` and `code`, of `free`.
fn split(free: u64, heap: u64, code: CodeReservation, cores: usize, runners: usize) -> Sizes {
    let running = runners as u64;
    let rest = free.saturating_sub(heap).saturating_sub(code.bytes);
    let sharers = u128::from(running.min(POOL_SHARERS));
    let rest = u128::from(rest);
    // A sixty-fourth of the guest each, or both halves' sizes if less:
    // pool = sharers * (bases + guest / 2048 + guest / 1024), guest = rest - pool.
    let share = sharers * rest / (u128::from(POOL_SHARE) + sharers);
    let (lower, upper) = (u128::from(LOWER_GROWTH), u128::from(UPPER_GROWTH));
    let base = u128::from(LOWER_BASE + UPPER_BASE);
    let parts = lower / upper + 1;
    let grown = sharers * (lower * base + parts * rest) / (lower + parts * sharers);
    let pool = share.min(grown) as u64;
    let (rest, sharers) = (rest as u64, sharers as u64);
    let pool = (pool / ALIGNMENT * ALIGNMENT).clamp(
        POOL_FLOOR.max(running * POOL_RUNNER_FLOOR),
        sharers * POOL_PER_RUNNER,
    );
    let (_, parked) = parked_reserve(cores, runners);
    let pool = pool + parked;
    Sizes {
        heap,
        code,
        kept: kept_share(heap),
        pool,
        guest: rest.saturating_sub(pool),
    }
}

/// The bytes a runner's pool keeps for each guest half's shadow tables, for
/// `guest` bytes of guest DRAM: the lower half's (user address spaces) and
/// the upper half's (the kernel). [`sizes`] sizes the pool as their sum.
pub fn pool_halves(guest: u64) -> (u64, u64) {
    (
        LOWER_BASE + guest / LOWER_GROWTH,
        UPPER_BASE + guest / UPPER_GROWTH,
    )
}

/// The bytes of a runner's `shadow` frames (its pool less the permanent
/// roots) that the upper half takes, in whole `page`s, for `guest` bytes of
/// guest DRAM: its [`pool_halves`] size, or its share of the part in their
/// sizes' ratio where the part is smaller than both (a pool sized as a
/// sixty-fourth of a small guest, or the pool's rounding); at least a page.
pub fn upper_half(shadow: u64, guest: u64, page: u64) -> u64 {
    let (lower, upper) = pool_halves(guest);
    let share = (u128::from(shadow) * u128::from(upper) / u128::from(lower + upper)) as u64;
    (upper.min(share) / page * page).max(page)
}

#[cfg(test)]
mod tests;
