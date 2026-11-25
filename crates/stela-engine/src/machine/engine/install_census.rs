//! Why blocks are installed, for `--profile`'s heartbeat: a block this
//! machine evicted within the last ten seconds (thrash), one another
//! machine has installed at that moment (elsewhere), or one this machine
//! never installed before (first). The tables are bounded and keyed by a
//! hash of the block's PC, frame address and variant, so the counts are
//! close, not exact: a slot keeps the latest block hashed to it. They exist
//! only while step timing is on: 1 MiB per machine, 16 MiB for a board its
//! PEs share ([`Census::board`]) and 2 MiB per hub, the same on every host.

use alloc::{boxed::Box, vec};
use core::hash::{Hash, Hasher};
use core::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

/// Evictions a machine remembers, by hash slot.
const EVICTED: usize = 1 << 16;
/// Bits of a machine's filter of the blocks it installed (two per block).
const SEEN_BITS: usize = 1 << 22;
/// The same for a board every PE installs into. A macOS boot and sign-in on
/// 8 PEs installed about 4 million distinct blocks and evicted about a
/// million a heartbeat (run 20261119-021855): a 4 Mi-bit filter was 87% set
/// by then, so a first install counted as one only about half the time, and
/// 64 Ki eviction slots were overwritten hundreds of thousands of times
/// within the ten seconds they cover. These keep both under a few percent
/// at that size.
const BOARD_EVICTED: usize = 1 << 20;
const BOARD_SEEN_BITS: usize = 1 << 26;
/// Installed blocks of every machine, by hash slot.
const LIVE: usize = 1 << 18;
/// Counter ticks per unit of a remembered eviction's time.
const TIME_SHIFT: u32 = 10;

/// What a machine's installs and evictions came to.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InstallCauses {
    /// Blocks installed, and of them those this machine evicted within
    /// ten seconds, those installed on another machine at the time, and
    /// those it never installed before. The last three count only while
    /// step timing (`--profile`) is on.
    pub installs: u64,
    pub thrash: u64,
    pub elsewhere: u64,
    pub first: u64,
    /// Code bytes of the blocks installed and of the blocks evicted.
    pub installed_bytes: u64,
    pub evicted_bytes: u64,
    /// Guest instructions of the blocks installed: with `installed_bytes`,
    /// host code bytes per guest instruction. And of those, the code bytes
    /// and guest instructions of blocks in the upper half of the address
    /// space (TTBR1: the kernel and the guarded monitors).
    pub guest_instructions: u64,
    pub upper_bytes: u64,
    pub upper_instructions: u64,
    /// Counter ticks spent evicting ([`super::Machine::evict_code`]).
    pub evict_ticks: u64,
    /// Bytes taken from the board's code window, at its 16-byte
    /// alignment, and the pieces taken: blocks and everything else the
    /// window holds for them (link thunks, tails, dispatchers, PAC and
    /// transition thunks). Less `installed_bytes`, what code beside the
    /// blocks' own takes.
    pub window_bytes: u64,
    pub window_pieces: u64,
    /// Blocks evicted by this machine, by the engine fetches that found
    /// them since their install: none (prepared ahead, or reached only
    /// through links), one (the fetch that compiled or found it), or more.
    /// A block whose only fetch was its own and that no link reached ran
    /// once; links are not counted here, so the first two are an upper
    /// bound on code compiled and never run again. While step timing is on.
    pub departed: [u64; 3],
    /// Installs of a block the census remembers evicting (by this machine,
    /// or on a shared board by any PE), by the time since: within one
    /// second, ten (with the first, `thrash`), a minute, or longer. While
    /// step timing is on.
    pub reinstall_ages: [u64; 4],
}

/// FNV-1a, for a block's identity.
struct Fnv(u64);

impl Hasher for Fnv {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
        }
    }
}

/// The hash of a block's identity, the same on every machine: its PC, the
/// address its code was read from and what it was compiled for.
pub(super) fn identity(pc: u64, physical: u64, variant: &impl Hash) -> u64 {
    let mut hasher = Fnv(0xcbf2_9ce4_8422_2325);
    pc.hash(&mut hasher);
    physical.hash(&mut hasher);
    variant.hash(&mut hasher);
    // Spread the low bits FNV leaves weak.
    let hash = hasher.finish();
    (hash ^ hash >> 29).wrapping_mul(0xbf58_476d_1ce4_e5b9)
}

/// A nonzero tag of `identity`, apart from the bits that pick its slot.
fn tag(identity: u64) -> u32 {
    (identity >> 32) as u32 | 1
}

/// One machine's memory of what it installed and evicted.
pub(super) struct Census {
    /// Per slot: the tag of the block evicted last and when, in
    /// [`TIME_SHIFT`] units.
    evicted: Box<[u64]>,
    seen: Box<[u64]>,
}

impl Census {
    pub(super) fn new() -> Self {
        Self::sized(EVICTED, SEEN_BITS)
    }

    /// The census of a board its PEs share: every install of the boot.
    pub(super) fn board() -> Self {
        Self::sized(BOARD_EVICTED, BOARD_SEEN_BITS)
    }

    fn sized(evicted: usize, seen_bits: usize) -> Self {
        Self {
            evicted: vec![0; evicted].into_boxed_slice(),
            seen: vec![0; seen_bits / 64].into_boxed_slice(),
        }
    }

    /// Note an install of `identity` at counter `now`: whether this
    /// machine evicted it within `window` ticks, whether it is the first
    /// install of it here, and the ticks since its eviction when the
    /// census remembers one.
    pub(super) fn installed(
        &mut self,
        identity: u64,
        now: u64,
        window: u64,
    ) -> (bool, bool, Option<u64>) {
        let slots = self.evicted.len();
        let slot = &mut self.evicted[identity as usize % slots];
        let held = *slot;
        let age = ((held >> 32) as u32 == tag(identity)).then(|| {
            ((now >> TIME_SHIFT).wrapping_sub(held & 0xffff_ffff) & 0xffff_ffff) << TIME_SHIFT
        });
        let thrash = age.is_some_and(|age| age >> TIME_SHIFT <= window >> TIME_SHIFT);
        if age.is_some() {
            *slot = 0;
        }
        let mut first = false;
        let seen_bits = self.seen.len() * 64;
        for bits in [identity, identity.rotate_left(23)] {
            let bit = (bits >> 11) as usize % seen_bits;
            let word = &mut self.seen[bit / 64];
            first |= *word & 1 << (bit % 64) == 0;
            *word |= 1 << (bit % 64);
        }
        (thrash, first, age)
    }

    /// Note an eviction of `identity` at counter `now`.
    pub(super) fn evicted(&mut self, identity: u64, now: u64) {
        let slots = self.evicted.len();
        self.evicted[identity as usize % slots] =
            u64::from(tag(identity)) << 32 | (now >> TIME_SHIFT) & 0xffff_ffff;
    }
}

/// Which machines have each block installed, a bit per PE (modulo 32) in
/// a slot shared by every machine of a [`super::Hub`], allocated at the
/// first use.
#[derive(Default)]
pub struct Live {
    table: AtomicPtr<AtomicU64>,
}

impl Live {
    fn table(&self) -> &[AtomicU64] {
        let mut table = self.table.load(Ordering::Acquire);
        if table.is_null() {
            let made: Box<[AtomicU64]> = (0..LIVE).map(|_| AtomicU64::new(0)).collect();
            let made = Box::into_raw(made).cast::<AtomicU64>();
            table = match self.table.compare_exchange(
                core::ptr::null_mut(),
                made,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => made,
                Err(other) => {
                    // SAFETY: `made` came from `Box::into_raw` of `LIVE`
                    // entries above and was never shared.
                    drop(unsafe { Box::from_raw(core::ptr::slice_from_raw_parts_mut(made, LIVE)) });
                    other
                }
            };
        }
        // SAFETY: a non-null table is `LIVE` entries this owner frees only
        // when dropped.
        unsafe { core::slice::from_raw_parts(table, LIVE) }
    }

    /// Note that PE `pe` installed `identity`: whether another PE had it
    /// installed.
    pub(super) fn install(&self, identity: u64, pe: usize) -> bool {
        let slot = &self.table()[(identity >> 7) as usize % LIVE];
        let (tag, bit) = (u64::from(tag(identity)) << 32, 1u64 << (pe % 32));
        let mut held = slot.load(Ordering::Relaxed);
        loop {
            let (elsewhere, new) = if held & !0xffff_ffff == tag {
                (held & 0xffff_ffff & !bit != 0, held | bit)
            } else {
                (false, tag | bit)
            };
            match slot.compare_exchange_weak(held, new, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return elsewhere,
                Err(now) => held = now,
            }
        }
    }

    /// Note that PE `pe` evicted `identity`.
    pub(super) fn evict(&self, identity: u64, pe: usize) {
        let slot = &self.table()[(identity >> 7) as usize % LIVE];
        let (tag, bit) = (u64::from(tag(identity)) << 32, 1u64 << (pe % 32));
        let mut held = slot.load(Ordering::Relaxed);
        while held & !0xffff_ffff == tag && held & bit != 0 {
            match slot.compare_exchange_weak(
                held,
                held & !bit,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(now) => held = now,
            }
        }
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        let table = *self.table.get_mut();
        if !table.is_null() {
            // SAFETY: as in `Live::table`; no reference outlives `&mut self`.
            drop(unsafe { Box::from_raw(core::ptr::slice_from_raw_parts_mut(table, LIVE)) });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_eviction_is_thrash_for_a_window_and_a_block_is_first_once() {
        let mut census = Census::new();
        let block = identity(0x1000, 0x8000_1000, &3u8);
        let window = 10 << TIME_SHIFT;
        assert_eq!(census.installed(block, 0, window), (false, true, None));
        census.evicted(block, 100 << TIME_SHIFT);
        assert_eq!(
            census.installed(block, 105 << TIME_SHIFT, window),
            (true, false, Some(5 << TIME_SHIFT))
        );
        census.evicted(block, 200 << TIME_SHIFT);
        assert_eq!(
            census.installed(block, 300 << TIME_SHIFT, window),
            (false, false, Some(100 << TIME_SHIFT))
        );
        // Another variant of the same PC and frame is another block.
        let other = identity(0x1000, 0x8000_1000, &4u8);
        assert_ne!(other, block);
        assert!(census.installed(other, 300 << TIME_SHIFT, window).1);
    }

    /// A board's census counts a sign-in's 4 million distinct blocks as
    /// first within a few percent, and a block evicted among a million
    /// others in its window as thrash; a machine's counted about two thirds
    /// and lost most such evictions.
    #[test]
    fn a_board_census_counts_a_sign_ins_blocks_and_evictions() {
        let window = 10 << TIME_SHIFT;
        let count = |mut census: Census| {
            let blocks = 4_000_000u64;
            let firsts = (0..blocks)
                .filter(|&n| {
                    let block = identity(0x1_8000_0000 + n * 20, 0x8_0000_0000 + n * 20, &0u8);
                    census.installed(block, 0, window).1
                })
                .count() as u64;
            for n in 0..1_000_000u64 {
                census.evicted(
                    identity(0x1_8000_0000 + n * 20, 0x8_0000_0000 + n * 20, &0u8),
                    1,
                );
            }
            let thrash = (0..1_000_000u64)
                .filter(|&n| {
                    let block = identity(0x1_8000_0000 + n * 20, 0x8_0000_0000 + n * 20, &0u8);
                    census.installed(block, 2, window).0
                })
                .count() as u64;
            (firsts * 100 / blocks, thrash / 10_000)
        };
        let (firsts, thrash) = count(Census::board());
        assert!(firsts >= 97 && thrash >= 60, "{firsts}% {thrash}%");
        let (firsts, thrash) = count(Census::new());
        assert!(firsts < 80 && thrash < 10, "{firsts}% {thrash}%");
    }

    #[test]
    fn a_block_another_pe_holds_counts_as_elsewhere_until_it_evicts_it() {
        let live = Live::default();
        let block = identity(0x2000, 0x9000_2000, &0u8);
        assert!(!live.install(block, 0));
        assert!(live.install(block, 1));
        live.evict(block, 0);
        assert!(!live.install(block, 1));
        live.evict(block, 1);
        assert!(!live.install(block, 2));
    }
}
