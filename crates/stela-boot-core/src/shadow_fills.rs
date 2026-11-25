//! What the shadow filled recently, region by region: why a fill was
//! needed (`SHADOW first= view= unmapped= lost= upgrade=`) and, after the
//! shadow lost a half, which pages of a region to map again at the
//! region's first fault (`restored=`).
//!
//! A region is the 2 MiB one host page table maps. Each core keeps a
//! direct-mapped history of recently filled regions ([`History`]), keyed by
//! a hash of the half, the address space (the lower half's TTBR0 and TCR
//! fields) and the region: one slot holds the views that filled it, the
//! incarnation of its half then, and a bit for each 16 KiB granule a fill
//! took. A half takes a new incarnation when it is emptied or a context
//! switch attaches a lower half the shadow did not keep, and gets its old
//! one back when the switch finds its half kept. So a refill in the same
//! incarnation follows a scoped unmap (a TLBI by address), and one in
//! another follows the loss of the whole half: a TLBI of everything, a half
//! out of frames, a kept half replaced. A slot another region took counts
//! as a first fill: the history forgets the oldest regions first.
//!
//! The restore is a fault-around that history guides: the pages are walked
//! again at the fault, after whatever TLBI emptied the half, and mapped
//! only as those walks give them, as Arm lets a TLB walk and cache any
//! valid translation before an access needs it.

use alloc::{boxed::Box, vec::Vec};

/// The causes, in the line's order (`first`, `view`, `unmapped`, `lost`,
/// `upgrade`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cause {
    /// No fill of the page is remembered.
    First,
    /// Filled before in this incarnation, in a region only other views
    /// filled.
    View,
    /// Filled before in this incarnation: a TLBI by address unmapped it.
    Unmapped,
    /// Filled before in an earlier incarnation of its half.
    Lost,
    /// A protection fault: a write to a page mapped read-only.
    Upgrade,
}

impl Cause {
    pub const COUNT: usize = 5;
}

/// The lower half and the upper half, as the host numbers them.
pub const HALVES: usize = 2;
/// The bytes one slot covers: one host page table's.
pub const REGION_BITS: u32 = 21;
/// The bytes one bit of a slot stands for: a 16 KiB guest granule.
pub const GRANULE_BITS: u32 = 14;
/// Contexts whose lower-half incarnation the history remembers: more than
/// the host keeps halves of, so a half the host dropped is still known.
const CONTEXTS: usize = 64;
/// Head bits: the tag above, the views' mask and the incarnation below.
const TAG_SHIFT: u32 = 40;
const VIEWS_SHIFT: u32 = 32;
/// Views a slot records.
pub const VIEWS: usize = 8;

/// A region's slot.
#[derive(Debug, Clone, Copy, Default)]
struct Slot {
    head: u64,
    granules: u128,
}

/// What a fill learned: why it was needed, and the other granules of its
/// region that an earlier incarnation of the half filled (bit `n` for the
/// granule at the region's start plus `n << GRANULE_BITS`), which the
/// caller may map again and report with [`History::keep`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fill {
    pub cause: Cause,
    pub restore: u128,
}

/// The fill history of one host core.
pub struct History {
    slots: Box<[Slot]>,
    incarnation: [u32; HALVES],
    next: u32,
    /// The incarnation of each recently detached lower half, by context,
    /// oldest first.
    contexts: Vec<((u64, u64), u32)>,
}

impl History {
    /// A history of `1 << bits` regions.
    pub fn new(bits: u32) -> Self {
        Self {
            slots: alloc::vec![Slot::default(); 1 << bits].into_boxed_slice(),
            incarnation: [1, 2],
            next: 3,
            contexts: Vec::with_capacity(CONTEXTS),
        }
    }

    /// The heap bytes the history takes.
    pub fn bytes(&self) -> usize {
        self.slots.len() * core::mem::size_of::<Slot>()
            + CONTEXTS * core::mem::size_of::<((u64, u64), u32)>()
    }

    fn fresh(&mut self) -> u32 {
        let incarnation = self.next;
        // Zero marks an empty slot.
        self.next = self.next.wrapping_add(1).max(1);
        incarnation
    }

    /// The slot of `page`'s region and its tag.
    fn slot(&self, half: usize, space: u64, page: u64) -> (usize, u64) {
        let region = page >> REGION_BITS;
        let hash = mix(space ^ region.rotate_left(17) ^ (half as u64) << 63);
        ((hash as usize) & (self.slots.len() - 1), hash >> TAG_SHIFT)
    }

    /// A fill of the page at `page` in `half` of address space `space` for
    /// view `view` (below [`VIEWS`]), a protection fault's with `upgrade`:
    /// record it, say why the page was not mapped and, when the region was
    /// filled in an earlier incarnation of the half, which of its other
    /// granules that one took. Those leave the record until [`Self::keep`]
    /// puts them back, so a granule the guest no longer maps is forgotten.
    pub fn fill(&mut self, half: usize, space: u64, page: u64, view: usize, upgrade: bool) -> Fill {
        let (index, tag) = self.slot(half, space, page);
        let incarnation = self.incarnation[half];
        let slot = self.slots[index];
        let known = slot.head != 0 && slot.head >> TAG_SHIFT == tag;
        let same = known && slot.head as u32 == incarnation;
        let bit = 1u128 << ((page >> GRANULE_BITS) & 127);
        let filled = known && slot.granules & bit != 0;
        let views = (slot.head >> VIEWS_SHIFT) as u8;
        let view_bit = 1u8 << (view % VIEWS);
        let cause = if upgrade {
            Cause::Upgrade
        } else if !filled {
            Cause::First
        } else if !same {
            Cause::Lost
        } else if views & view_bit == 0 {
            Cause::View
        } else {
            Cause::Unmapped
        };
        let (views, granules, restore) = match (known, same) {
            (true, true) => (views | view_bit, slot.granules | bit, 0),
            (true, false) => (view_bit, bit, slot.granules & !bit),
            _ => (view_bit, bit, 0),
        };
        self.slots[index] = Slot {
            head: tag << TAG_SHIFT | u64::from(views) << VIEWS_SHIFT | u64::from(incarnation),
            granules,
        };
        Fill { cause, restore }
    }

    /// `half` was emptied: every page it mapped is lost, and for the lower
    /// half every kept half with it.
    pub fn emptied(&mut self, half: usize) {
        self.incarnation[half] = self.fresh();
        if half == 0 {
            self.contexts.clear();
        }
    }

    /// The lower half switched from context `from` to `to`, and `found`
    /// tells whether the host kept `to`'s half: it gets that half's
    /// incarnation back, or a new one.
    pub fn switched(&mut self, from: (u64, u64), to: (u64, u64), found: bool) {
        self.contexts.retain(|(context, _)| *context != from);
        if self.contexts.len() == CONTEXTS {
            self.contexts.remove(0);
        }
        self.contexts.push((from, self.incarnation[0]));
        let kept = self
            .contexts
            .iter()
            .position(|(context, _)| *context == to)
            .filter(|_| found)
            .map(|index| self.contexts.remove(index).1);
        self.incarnation[0] = match kept {
            Some(incarnation) => incarnation,
            None => self.fresh(),
        };
    }
}

/// The address of granule `bit` of the region holding `page`.
pub fn granule(page: u64, bit: u32) -> u64 {
    (page & !((1 << REGION_BITS) - 1)) | u64::from(bit) << GRANULE_BITS
}

/// A 64-bit finalizer (SplitMix64's): every input bit moves every output
/// bit, so the slot index and the tag are independent.
fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOWER: usize = 0;
    const UPPER: usize = 1;
    const A: (u64, u64) = (0x1000_4000, 0x19);
    const B: (u64, u64) = (0x2000_8000, 0x19);

    fn cause(history: &mut History, half: usize, space: u64, page: u64, view: usize) -> Cause {
        history.fill(half, space, page, view, false).cause
    }

    #[test]
    fn a_fill_names_why_the_page_was_not_mapped() {
        let mut history = History::new(12);
        let page = 0x1_8000_4000;
        assert_eq!(cause(&mut history, LOWER, A.0, page, 0), Cause::First);
        // A TLBI unmapped it; the half stayed.
        assert_eq!(cause(&mut history, LOWER, A.0, page, 0), Cause::Unmapped);
        // The kernel's view takes the region for the first time.
        assert_eq!(cause(&mut history, LOWER, A.0, page, 1), Cause::View);
        assert_eq!(cause(&mut history, LOWER, A.0, page, 1), Cause::Unmapped);
        assert_eq!(
            history.fill(LOWER, A.0, page, 0, true).cause,
            Cause::Upgrade
        );
        // Another page of the region, another space's page at the same
        // address, and the upper half's, are first fills.
        assert_eq!(
            cause(&mut history, LOWER, A.0, page + 0x4000, 0),
            Cause::First
        );
        assert_eq!(cause(&mut history, LOWER, B.0, page, 0), Cause::First);
        assert_eq!(cause(&mut history, UPPER, 0, page, 0), Cause::First);
        // The half was emptied: every page of it is lost, for any view.
        history.emptied(LOWER);
        assert_eq!(cause(&mut history, LOWER, A.0, page, 1), Cause::Lost);
        // The upper half's pages stay.
        assert_eq!(cause(&mut history, UPPER, 0, page, 0), Cause::Unmapped);
    }

    #[test]
    fn a_kept_half_keeps_its_incarnation_and_a_dropped_one_is_lost() {
        let mut history = History::new(12);
        let page = 0x1_0000_0000;
        assert_eq!(cause(&mut history, LOWER, A.0, page, 0), Cause::First);
        history.switched(A, B, false);
        assert_eq!(cause(&mut history, LOWER, B.0, page, 0), Cause::First);
        // Back to A, whose half the host kept: a refill there followed a
        // TLBI, not the switch.
        history.switched(B, A, true);
        assert_eq!(cause(&mut history, LOWER, A.0, page, 0), Cause::Unmapped);
        // Back to B, whose half the host dropped meanwhile.
        history.switched(A, B, false);
        assert_eq!(cause(&mut history, LOWER, B.0, page, 0), Cause::Lost);
        // Emptying the lower half forgets the kept halves too.
        history.switched(B, A, true);
        history.emptied(LOWER);
        history.switched(A, B, true);
        assert_eq!(cause(&mut history, LOWER, B.0, page, 0), Cause::Lost);
    }

    /// The first fill of a region after its half was lost names the other
    /// granules the earlier incarnation's faults took, once; a granule
    /// mapped again from that list is not one a fault took, so it is not
    /// named after the next loss unless a fault takes it again.
    #[test]
    fn a_lost_region_names_its_granules_once() {
        let mut history = History::new(12);
        let region = 0x7_0020_0000;
        for granule in [0, 3, 5, 127] {
            let page = region + (granule << GRANULE_BITS) + 0x1000;
            assert_eq!(history.fill(UPPER, 0, page, 1, false).restore, 0);
        }
        history.emptied(UPPER);
        let fault = region + (5 << GRANULE_BITS);
        let fill = history.fill(UPPER, 0, fault, 1, false);
        assert_eq!(fill.cause, Cause::Lost);
        assert_eq!(fill.restore, 1 | 1 << 3 | 1 << 127);
        assert_eq!(granule(fault, 127), region + (127 << GRANULE_BITS));
        assert_eq!(history.fill(UPPER, 0, fault, 1, false).restore, 0);
        // Only granule 5 was a fault's in that incarnation.
        history.emptied(UPPER);
        let fill = history.fill(UPPER, 0, region, 1, false);
        assert_eq!((fill.cause, fill.restore), (Cause::First, 1 << 5));
        history.emptied(UPPER);
        let fill = history.fill(UPPER, 0, region, 1, false);
        assert_eq!((fill.cause, fill.restore), (Cause::Lost, 0));
    }

    #[test]
    fn the_history_is_bounded() {
        let mut history = History::new(10);
        for context in 0..1000u64 {
            history.switched((context, 0), (context + 1, 0), false);
        }
        assert!(history.contexts.len() <= CONTEXTS);
        assert_eq!(history.bytes(), 1024 * 32 + CONTEXTS * 24);
        // A slot another region took reports a first fill.
        let mut first = 0;
        for _ in 0..2 {
            for region in 0..4096u64 {
                let page = region << REGION_BITS;
                first += u64::from(history.fill(UPPER, 0, page, 0, false).cause == Cause::First);
            }
        }
        assert!(first > 4096 + 2048, "{first}");
    }
}
