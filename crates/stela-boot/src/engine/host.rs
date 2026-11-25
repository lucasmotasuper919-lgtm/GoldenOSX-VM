//! The bare-metal [`Host`]: guest RAM, the native code arena, shadow paging,
//! precise native faults, the local APIC timer and the TSC counter.
//!
//! Shadow roots are PML5 pages under LA57 and PML4 pages otherwise, whose
//! entries hold the guest's native ranges and the shared window
//! (`layout::window_entries`), which native code's guest range checks keep
//! every guest access out of; no other entry is ever mapped, which those
//! checks rely on. There is one root per
//! permission view (EL0, EL1, EL1 with PAN, GL1), and a second set for the
//! same views while tag checks keep Tagged memory out (`Regime::tag_checks`),
//! filled on demand by the #PF handler from the software walk of the
//! guest's tables, a window of the guest leaf per fault. The views keep
//! Each half takes its tables from its own part of the pool and records the
//! leaves it was filled from (`mmu::shadow::Extents`), so a TLBI unmaps
//! only the ranges it reaches. The roots come in sets (`Set`), one per
//! lower half's context (TTBR0 and TCR's lower fields) and rights
//! (SPRR_PERM_EL0, SPRR_PERM_EL1 and E0PD1): a set holds its context's
//! lower tables, and the sets of one rights value share the upper half's.
//! A context switch, by the engine or natively (`x86_64::table_switch`),
//! loads another set's root and rewrites none, so each set's views keep
//! their PCIDs' translations.
//!
//! With SMAP, EL1's two views share EL1's root, as they differ only in what
//! PAN takes away, which is all data access or none: a leaf that PAN guards
//! is a user page, and PAN is RFLAGS.AC clear ([`State::root_of`]). PAN
//! changes around each copy to user memory, so this spares a root load, and
//! without PCIDs a TLB flush, on every one.
use super::interrupts::{Frame, SPURIOUS_VECTOR, TIMER_VECTOR};
use crate::Result;
use crate::{
    apic::{self, Apic},
    layout,
    tables::{self, Frames, NX, PAGE, PRESENT, WRITABLE},
};
use alloc::{sync::Arc, vec::Vec};
use core::arch::{asm, global_asm};
use stela_boot_core::native_resources::{CodeExtents, MAX_CODE_PARTS, code_part_bytes};
use stela_compiler::x86_64::table_switch;
use stela_compiler::{GuestState, block::BlockExit};
use stela_contracts::refuse;
use stela_engine::machine::FaultCause;
use stela_engine::machine::Host;
use stela_engine::machine::InstallError;
use stela_engine::machine::NativeCode;
use stela_engine::machine::NativeExit;
use stela_engine::machine::NativeFault;
use stela_engine::machine::NativeView;
use stela_engine::machine::Regime;
use stela_engine::machine::engine::GlobalSpan;
use stela_engine::machine::engine::installed_code::InstalledCodeCache;
use stela_engine::machine::mmu;
use stela_engine::machine::mmu::Access;
use stela_engine::machine::mmu::Outcome;
use stela_engine::machine::mmu::placement::Placement;
use stela_engine::machine::mmu::shadow;
use stela_engine::machine::mmu::shadow::Extents;
use stela_engine::machine::mmu::tlbi::Everywhere;
use stela_engine::machine::mmu::tlbi::Reach;
use stela_engine::machine::mmu::translated::Tlb;

mod bare_host;
pub(crate) mod chain_check;
mod faults;
pub(crate) mod pmu;
#[cfg(feature = "nvk")]
mod process_fault;
pub(crate) mod sampling;
mod signals;
pub use bare_host::BareHost;
use faults::*;
pub(crate) use faults::{gate_fill, native_exception, page_fault};
pub(crate) use signals::{
    clear_interrupted, completion_signal, device_signals, doorbell_counter, doorbells,
    end_stale_in_service, in_interrupt, interrupted, join_shootdown, network_interrupt,
    network_signal, nmi, ring, root_irq_active, shootdown, shootdown_interrupt, signal_interrupt,
    sleep_for, stale_in_service_report, take_network_signal, undelivered_doorbells, wake_interrupt,
};
pub use signals::{spurious_interrupt, stray_interrupt, timer_interrupt};
#[cfg(feature = "diagnostics")]
pub(crate) mod fp_diagnostic;
pub(super) mod local;
#[cfg(feature = "native-profile")]
use stela_boot_core::native_profile as profile;
use stela_boot_core::ram;
use stela_boot_core::shadow_fills;
mod translated;
pub(crate) use local::activated_base;
#[cfg(feature = "nvk")]
pub(crate) use local::context::close_moves;
pub(crate) use local::context::{RootIdentity, moved_here, moves_closed};
use local::state;
#[cfg(feature = "nvk")]
pub(crate) use local::{processor, processors};

/// Calls `action` with each host run behind `bytes` of guest RAM at `guest`, as
/// (host physical, bytes). False, partway, if part of it is not RAM.
#[cfg(any(feature = "nvk", feature = "soft-gpu"))]
pub(crate) fn guest_runs(guest: u64, bytes: u64, action: impl FnMut(u64, u64)) -> bool {
    ram::runs(&local::state().ram, guest, bytes, action)
}
pub(crate) use local::{MmioRoots, Owner as HostOwner};
pub use ram::RamRange;

/// Page-table frames from the handoff pool: a permanent part for roots and a
/// shadow part, split between the guest's two halves, each of which resets
/// wholesale. Between resets a half hands out the frames given back to it
/// (the tables of kept lower halves the shadow let go) before new ones.
struct Pool {
    base: u64,
    bytes: u64,
    /// The upper half's bytes at the shadow part's end
    /// (`host_memory::upper_half`); the lower half has the rest.
    upper: u64,
    permanent: u64,
    /// The next free shadow frame's offset of each half ([`LOWER`],
    /// [`UPPER`]).
    shadow: [u64; 2],
    /// First shadow frame.
    shadow_base: u64,
    /// The half the next frames map.
    half: usize,
    /// The most bytes of shadow frames each half has held at once.
    peak: [u64; 2],
    /// Shadow frames handed out so far.
    taken: u64,
    /// Each half's frames given back since it was last reset, which it
    /// hands out first: the lower half's, whose storage holds every frame
    /// of the half, so a fault never allocates to give one back.
    free: [Vec<u64>; 2],
}

/// The guest halves: VAs with bit 63 clear and set.
const LOWER: usize = 0;
const UPPER: usize = 1;

impl Frames for Pool {
    fn frame(&mut self) -> Option<u64> {
        if let Some(frame) = self.free[self.half].pop() {
            self.taken += 1;
            zero(frame);
            return Some(frame);
        }
        let (_, end) = self.range(self.half);
        if self.shadow[self.half] + PAGE > end {
            return None;
        }
        let frame = self.base + self.shadow[self.half];
        self.shadow[self.half] += PAGE;
        let used = self.shadow[self.half] - self.range(self.half).0;
        self.peak[self.half] = self.peak[self.half].max(used);
        self.taken += 1;
        zero(frame);
        Some(frame)
    }
    fn table(&self, physical: u64) -> *mut u64 {
        layout::physical(physical) as *mut u64
    }
}

fn zero(frame: u64) {
    // SAFETY: a pool frame reached through the physmap.
    unsafe { (layout::physical(frame) as *mut u8).write_bytes(0, PAGE as usize) };
}

impl Pool {
    fn permanent(&mut self) -> Option<u64> {
        if self.permanent + PAGE > self.shadow_base - self.base {
            return None;
        }
        let frame = self.base + self.permanent;
        self.permanent += PAGE;
        zero(frame);
        Some(frame)
    }
    /// The offsets of `half`'s shadow frames: the shadow part's first bytes,
    /// or its last [`Self::upper`].
    fn range(&self, half: usize) -> (u64, u64) {
        let start = self.shadow_base - self.base;
        let middle = self.bytes - self.upper;
        if half == LOWER {
            (start, middle)
        } else {
            (middle, self.bytes)
        }
    }
    fn reset_half(&mut self, half: usize) {
        self.shadow[half] = self.range(half).0;
        self.free[half].clear();
    }
    /// Make room for every frame of the lower half among the frames given
    /// back: only kept lower halves give frames back.
    fn reserve_free(&mut self) {
        let (start, end) = self.range(LOWER);
        self.free[LOWER] = Vec::with_capacity(((end - start) / PAGE) as usize);
    }
}

/// Permission views; the tag-checked roots follow the unchecked ones.
const VIEWS: usize = 4;
/// EL0's view (GL0's too), EL1's, EL1's with PAN, and the GXF guarded
/// level's (GL1).
const EL0_VIEW: usize = 0;
const EL1_VIEW: usize = 1;
const PAN_VIEW: usize = 2;
const GUARDED_VIEW: usize = 3;
/// Why the whole shadow was reset: a half ran out of frames; controls
/// that differ in SPRR state alone (more than its permission registers),
/// in TTBR0 alone (with an ASID the upper half's non-global entries were
/// made under), or otherwise; another
/// pattern of tag-checked halves; [`Host::invalidate_translations`]; or a
/// TLBI that reaches everything.
#[derive(Clone, Copy)]
enum Cause {
    Exhausted,
    Sprr,
    Table,
    Controls,
    Tags,
    Invalidated,
    Maintained,
}
const CAUSES: usize = 7;
/// The TCR fields of the EL1&0 and EL2&0 regimes that configure TTBR0's
/// range alone: T0SZ, EPD0, IRGN0, ORGN0, SH0, TG0 (bits 15:0), TBI0, HPD0,
/// TBID0, NFD0, E0PD0 and TCMA0.
const LOWER_TCR: u64 = 0xffff | 1 << 37 | 1 << 41 | 1 << 51 | 1 << 53 | 1 << 55 | 1 << 57;
/// The TCR field of those regimes that decides EL0's accesses to TTBR1's
/// range alone: E0PD1, which faults them without a walk.
const USER_TCR: u64 = 1 << 56;
/// The [`mmu::Controls`] fields a [`Cause::Controls`] reset counts as
/// differing: SCTLR, TCR, TTBR0, TTBR1, MAIR, SPRR, CTRR, the regime and
/// stage 2 (its requirement or controls).
const FIELDS: usize = 9;

fn differing(from: &mmu::Controls, to: &mmu::Controls) -> [bool; FIELDS] {
    [
        from.sctlr != to.sctlr,
        from.tcr != to.tcr,
        from.ttbr0 != to.ttbr0,
        from.ttbr1 != to.ttbr1,
        from.mair != to.mair || from.xs != to.xs || from.tagged != to.tagged,
        from.sprr != to.sprr,
        from.ctrr != to.ctrr,
        from.regime != to.regime,
        from.stage2 != to.stage2 || from.second_stage != to.second_stage,
    ]
}

/// What decides the views' rights besides the guest's tables and the
/// controls every set shares: SPRR_PERM_EL0, SPRR_PERM_EL1 and the E0PD1 bit
/// of TCR ([`USER_TCR`]), which faults EL0's accesses to the upper half.
type Rights = (u64, u64, u64);
/// What selects a lower half's context: TTBR0 and TCR's lower fields
/// ([`LOWER_TCR`]).
type LowerKey = (u64, u64);

/// One set of roots: a view's each, for one lower half's context and one
/// [`Rights`] value, everything else as the current controls say (a change
/// of any other control empties every set). XNU switches between a few
/// processes' TTBR0s and the kernel's, around its writes to protected data
/// between SPRR_PERM_EL1 values and between threads' EL0 permissions, and
/// the restore kernel toggles E0PD1: each finds its set's tables again
/// instead of filling them anew, and its views' PCIDs their translations.
/// Set `s`'s view `v` root is the permanent frame `s * ROOTS + v` after the
/// host root, loaded in PCID [`root_context`] of that index.
struct Set {
    /// The context and rights it holds, `None` while it is free.
    key: Option<(LowerKey, Rights)>,
    /// Its lower half's leaves and the root entries its views may hold of
    /// it, while it is not the current set: the current set's are
    /// [`State::extents`]`[LOWER]` and [`State::lower_used`].
    extents: Extents,
    lower_used: [u64; ROOT_ENTRIES / 64],
    /// When it was last the current set, for replacement.
    used: u64,
}
const ROOTS: usize = 2 * VIEWS;
const _: () = assert!(ROOTS == table_switch::SET_ROOTS);
const SETS: usize = table_switch::SETS;

/// Root `view`'s TLB context while PCIDE is set.
/// Set in a [`State::native_view`] word, with the view in bits 7:0 and the
/// requester's EL, PAN, guarded level and unprivileged flag above.
const NATIVE_VIEW: u64 = 1 << 63;

/// The [`State::native_view`] word of `view` for `requester`.
fn native_view_word(view: usize, requester: mmu::Requester) -> u64 {
    NATIVE_VIEW
        | view as u64
        | u64::from(requester.el) << 8
        | u64::from(requester.pan) << 10
        | u64::from(requester.guarded) << 11
        | u64::from(requester.unprivileged) << 12
}

/// The view of `regime`: EL0, EL1, EL1 with PAN, or the GXF guarded level,
/// in the tag-checked set when it checks tags.
fn view_of(regime: &Regime) -> usize {
    let view = match (
        regime.requester.el,
        regime.requester.pan,
        regime.requester.guarded,
    ) {
        (0, _, _) => EL0_VIEW,
        (_, _, true) => GUARDED_VIEW,
        (_, false, false) => EL1_VIEW,
        (_, true, false) => PAN_VIEW,
    };
    view + if regime.tag_checked() { VIEWS } else { 0 }
}

/// The PCID of root `index` (a set's first view's index plus the view).
const fn root_context(index: usize) -> u64 {
    table_switch::FIRST_CONTEXT + index as u64
}

/// Every PCID a root loads in while PCIDE is set: the host root's 0, the
/// loader root's 1 (`storage::TaggedRoots`) and every set's views'.
pub(crate) const CONTEXTS: u64 = root_context(SETS * ROOTS);

/// What the views' TLB contexts hold. With a context of its own, a change of
/// view or of set keeps its own translations of the engine, the translated
/// code and the guest, which no mapping marks global. An invalidation
/// reaches the current set's other contexts through INVPCID, or else marks
/// them to flush when loaded, as it marks other sets'
/// ([`table_switch::Sets::stale`]).
struct Contexts {
    /// PCIDE is set ([`tag_views`]). Kept here because a hypervisor may
    /// intercept every CR4 read, and views change at each exception.
    tagged: bool,
    invpcid: bool,
    /// CR4.SMAP is set, and RFLAGS.AC stands for PAN ([`tag_views`]).
    pan_flag: bool,
}

/// Whether this core's views load in their contexts from now on, as
/// [`storage::TaggedRoots`] sets or restores PCIDE, and whether RFLAGS.AC
/// stands for PAN, as it sets or restores SMAP.
pub(super) fn tag_views(tagged: bool, pan_flag: bool) {
    if let Some(state) = local::installed() {
        state.contexts.tagged = tagged;
        state.contexts.pan_flag = pan_flag;
        state.native_sets.tagged = u64::from(tagged);
    }
}

/// Whether this core has the state [`tag_views`] sets.
pub(super) fn views_installed() -> bool {
    local::installed().is_some()
}

/// RFLAGS.AC, which is PSTATE.PAN clear where it stands for it.
const RFLAGS_AC: u64 = 1 << 18;

/// PSTATE.PAN by this core's RFLAGS.AC.
fn flags_pan() -> bool {
    let flags: u64;
    // SAFETY: reads RFLAGS through the stack, which the push and pop balance.
    unsafe { asm!("pushfq", "pop {}", out(reg) flags, options(preserves_flags)) };
    flags & RFLAGS_AC == 0
}

/// Lift SMAP for what runs next on this core, which is not guest code: the
/// loader root's firmware tables may mark any page a user page.
pub(super) fn allow_user_pages() {
    if local::installed().is_some_and(|state| state.contexts.pan_flag) {
        // SAFETY: CPL0 with SMAP, which `pan_flag` says this core has.
        unsafe { asm!("stac", options(nomem, nostack)) };
    }
}

/// Whether this core runs with PCIDE set, as far as its [`Contexts`] know,
/// which a read of CR4 would ask a hypervisor.
pub(super) fn views_tagged() -> Option<bool> {
    local::installed().map(|state| state.contexts.tagged)
}

/// Frames reserved for roots: the host root, then every set's.
const PERMANENT_FRAMES: u64 = stela_boot_core::native_resources::SHADOW_PERMANENT_FRAMES;
const _: () = assert!(PERMANENT_FRAMES == (1 + SETS * ROOTS) as u64);
/// A root's entries.
const ROOT_ENTRIES: usize = 512;
/// log2 of the bytes a four-level root entry maps: 512 GiB.
const PML4_ENTRY_BITS: u32 = 39;
/// The last address of the lower half under LA57 (48 bits) and four-level
/// paging (47 bits).
const FIVE_LEVEL_LOWER_END: u64 = (1 << 48) - 1;
const FOUR_LEVEL_LOWER_END: u64 = (1 << 47) - 1;
/// The aligned bytes of a guest leaf one fault maps.
const FILL_WINDOW: u64 = 64 * 1024;
/// log2 of the 2 MiB regions each core's fill history remembers: 4096,
/// 128 KiB, more than the regions of the kept halves and the kernel.
const HISTORY_BITS: u32 = 12;
/// The causes of a #PF that ended its block: a guest fault, a device
/// access, anything else (a watched page, a Tag Check, an address outside
/// the native ranges, an unsupported translation).
const ABANDONED: usize = 3;
/// Pages a scoped invalidation flushes one by one; more reload CR3.
const INVLPG_PAGES: u64 = 16;
/// The INVPCIDs a TLBI by address of the upper half spends on other sets'
/// views, which share its tables, before it marks the rest stale.
const OTHER_SET_INVPCIDS: usize = 64;
const FAULT_KIND: u64 = 0xffff;
pub(crate) const COUNTER_HZ: u64 = 24_000_000;

/// Fresh ticks from the installed physical Host's configured 24 MHz
/// clock. Native PMP services execute on the BSP owner and share this
/// same source, rather than reusing an Access timestamp or a copied sample.
pub(crate) fn fresh_counter() -> u64 {
    let state = state();
    counter_at_tsc(state, rdtsc())
}

/// Keep this processor's software TSC correction: its raw counter reads
/// `offset` ticks behind the BSP's. From then on [`rdtsc`] reads the BSP's
/// counter here (`tsc::sync::shared_tsc`), and native count reads, which
/// take the raw counter, count from a base moved by it
/// ([`State::native_tsc_base`]). Runs on the AP before it is ready.
pub(crate) fn agree_tsc(offset: i64) {
    local::set_tsc_offset(offset);
    if offset != 0 {
        TSC_OFFSETS.store(true, core::sync::atomic::Ordering::Release);
    }
}

/// Set once an AP agrees with the BSP only through a software offset
/// ([`agree_tsc`]): until then every core's raw counter is the BSP's.
static TSC_OFFSETS: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// This processor's time-stamp counter as the BSP's reads at the same
/// instant: the value every TSC stamp one core stores and another compares
/// must be. A core whose counter agreed only through a software offset
/// reads it moved by that offset. Cores take it only once their local
/// record is active, which each AP's is before the first offset is kept.
pub fn rdtsc() -> u64 {
    shared_tsc(crate::apic::rdtsc())
}

/// This processor's raw counter value `raw` as the BSP's counter read at
/// the same instant ([`rdtsc`]), for a caller that takes the raw value
/// itself, as the native clock does after its LFENCE.
pub(crate) fn shared_tsc(raw: u64) -> u64 {
    if TSC_OFFSETS.load(core::sync::atomic::Ordering::Relaxed) {
        stela_boot_core::tsc::sync::shared_tsc(raw, local::tsc_offset())
    } else {
        raw
    }
}

/// The device clock: [`rdtsc`] on any core, never earlier than a value any
/// core was given (`tsc::sync::Monotonic`).
static DEVICE_COUNTER: stela_boot_core::tsc::sync::Monotonic =
    stela_boot_core::tsc::sync::Monotonic::new();

/// The counter the drivers' clock reads on every core (`on_counter`), and
/// every other host time that one core takes and another compares: the
/// network stack's, a USB4 fabric's, the boot screen's running step. It
/// never goes back, even where a core's counter moved after its
/// synchronization: that core reads the last value given until its own
/// passes it ([`device_counter_behind`]).
pub fn device_counter() -> u64 {
    DEVICE_COUNTER.read(rdtsc())
}

/// The most a core's [`rdtsc`] trailed the device clock, in ticks.
pub(crate) fn device_counter_behind() -> u64 {
    DEVICE_COUNTER.longest_behind()
}

impl State {
    /// The base native count reads, which take this core's raw counter,
    /// subtract: `tsc_base`, the BSP's, moved by this core's software
    /// offset (`tsc::sync::corrected_base`).
    pub(crate) fn native_tsc_base(&self) -> u64 {
        stela_boot_core::tsc::sync::corrected_base(self.tsc_base, local::tsc_offset())
    }
}

fn counter_at_tsc(state: &State, tsc: u64) -> u64 {
    let ticks = tsc.saturating_sub(state.tsc_base);
    ((u128::from(ticks) * u128::from(state.counter_per_tsc)) >> 64) as u64
}

/// Convert an absolute BSP TSC deadline using the very same installed fixed
/// point ratio/base as fresh_counter; no guest-clock or nominal-Hz conversion.
pub(crate) fn counter_deadline(tsc: u64) -> stela_engine::machine::engine::NativeEpochDeadline {
    stela_engine::machine::engine::NativeEpochDeadline(counter_at_tsc(state(), tsc))
}

#[derive(Clone, Copy)]
struct Recorded {
    rip: u64,
    origin: stela_engine::machine::NativeFaultOrigin,
    trap: u16,
    error_code: u64,
    address: u64,
    write: bool,
    cause: FaultCause,
    /// RAX through R15 at the faulting instruction, by x86 register number.
    registers: stela_compiler::block::HostGprSnapshot,
}

pub struct State {
    ram: Vec<RamRange>,
    registered_ram: Option<alloc::sync::Arc<stela_engine::memory::RegisteredSystemRam>>,
    pool: Pool,
    /// Engine root without any guest mapping.
    host_root: u64,
    /// Original loader root kept from install, never caller metadata.
    window_root: u64,
    _loader_tables: &'static crate::paging::loader::LoaderTables,
    /// The current set's roots ([`State::set`]).
    roots: [u64; ROOTS],
    /// Every set, the current one, the uses of sets so far (which order
    /// them for replacement) and the next class a set may take.
    sets: Vec<Set>,
    set: usize,
    set_uses: u64,
    next_class: u64,
    /// The sets as native switches read them, which they move between.
    native_sets: table_switch::Sets,
    /// Switches to another [`Rights`] value's set, and those that made it.
    bank_switches: u64,
    bank_misses: u64,
    /// Root loads for a change of view (EL, PAN, GXF or tag checks).
    view_switches: u64,
    contexts: Contexts,
    /// A shadow-map attempt or software-TLB fill may have published guest
    /// translations. Only a complete reset can prove both halves clean again.
    legacy_translations_dirty: bool,
    current: usize,
    /// The view and requester native code moved to since the host last
    /// followed it ([`NATIVE_VIEW`]), 0 when none: a native transition
    /// loads the view's root itself (`Host::native_view`).
    native_view: u64,
    regime: Option<Regime>,
    /// The halves whose Tagged memory the tag-checked roots keep out.
    tag_checks: [bool; 2],
    code_virtual: u64,
    code_physical: u64,
    code_bytes: u64,
    code_part_bytes: u64,
    code_parts: u64,
    /// The board's code window, which the engine allocates itself: run at
    /// `window_virtual`, written through the physmap. Empty without one.
    window_virtual: u64,
    window_physical: u64,
    window_bytes: u64,
    /// The private arena part new code goes to.
    code_unit: u64,
    /// Successful initialized installs in each arena unit, from its start.
    /// These extents require this State's backing, root and CPU owner.
    installed_extents: [CodeExtents; MAX_CODE_PARTS as usize],
    /// The furthest the arena's code has reached: what the run needed of it
    /// until it first filled and new code reclaimed the oldest.
    code_peak: u64,
    installed_immutable: InstalledCodeCache,
    recorded: Option<Recorded>,
    native: bool,
    tsc_base: u64,
    tsc_hz: u64,
    /// COUNTER_HZ / tsc_hz as a 0.64 fixed-point fraction: the counter is
    /// read several times per block, and a 128-bit division is a library
    /// call.
    counter_per_tsc: u64,
    apic: Apic,
    /// The APIC timer's ticks per second at divide-by-one.
    apic_hz: u64,
    /// LVT_TIMER holds the engine's unmasked one-shot `TIMER_VECTOR`, as
    /// `arm_timer` last wrote it, since the native clock's handover count
    /// ([`crate::native_clock::TIMER_HANDOVERS`]) was this: while it is
    /// unchanged an arm writes only the count. `signals::sleep_for`, the
    /// other writer here, keeps it true.
    timer_lvt: Option<u64>,
    /// This processor's doorbell, when it waits with MWAIT on one
    /// (`native_idle::Ring`): a PE's wake is then a store, not an IPI.
    ring: Option<crate::native_idle::Ring>,
    pub fills: u64,
    pub resets: u64,
    /// Resets because a half's shadow frames ran out.
    pub exhausted: u64,
    /// Resets of the lower half alone.
    pub lower_resets: u64,
    /// Whole resets by [`Cause`].
    reset_causes: [u64; CAUSES],
    /// The [`Cause::Controls`] resets each field differed in ([`differing`]).
    control_fields: [u64; FIELDS],
    /// The controls the last of those resets went from and to.
    last_controls: Option<(mmu::Controls, mmu::Controls)>,
    /// The leaves each half was filled from.
    extents: [Extents; 2],
    /// The span of global lower-half leaves other PEs' broadcasts read.
    lower_globals: Option<Arc<GlobalSpan>>,
    /// The lower half's root entries the current set's roots may hold, by
    /// index: dropping the half visits these, a few, rather than every
    /// entry of the half.
    lower_used: [u64; ROOT_ENTRIES / 64],
    /// The upper half's root entries any set's roots may hold.
    upper_used: [u64; ROOT_ENTRIES / 64],
    /// The roots' contexts that may hold the translated helper's window
    /// since its pages last changed, by root index.
    window_contexts: [u64; (SETS * ROOTS).div_ceil(64)],
    /// Switches of the lower half to another TTBR0, those that found its
    /// set holding a half (natively or not), those native code made, and
    /// the other sets' halves a TLBI or a set's replacement dropped.
    pub lower_switches: u64,
    pub lower_found: u64,
    pub native_switches: u64,
    pub kept_dropped: u64,
    /// TLBIs that unmapped ranges rather than whole halves.
    pub scoped: u64,
    /// Other sets' lower halves whose tables went back to the pool:
    /// replaced, dropped by a TLBI or given up for a half out of frames.
    pub reclaimed: u64,
    /// Pages a region's first fill since its half was lost mapped again
    /// from the fill history (`shadow_fills`).
    pub restored: u64,
    /// Pages mapped beside a fault's from the same leaf.
    pub prefills: u64,
    /// Pages mapped beside a fault's from the window's other leaves
    /// (fault-around), each from its own walk, which starts at the fault's
    /// leaf table when that holds it.
    pub around: u64,
    /// TSC ticks from a filled #PF's handler entry to its return.
    pub fill_ticks: u64,
    /// Every #PF from native code, filled or ending its block (by cause:
    /// a guest fault, a device access, anything else), and the TSC ticks
    /// of all of them.
    pub faults: u64,
    pub abandoned: [u64; ABANDONED],
    pub fault_ticks: u64,
    /// The fills by [`shadow_fills::Cause`], from [`State::history`].
    pub fill_causes: [u64; shadow_fills::Cause::COUNT],
    history: shadow_fills::History,
    /// Guest walks made for fills and fault-around, and of those the ones
    /// that started at the table the fault's own walk read its leaf from.
    pub walks: u64,
    pub table_walks: u64,
    /// Fills the #PF gate made without saving extended state
    /// ([`faults::gate_fill`]).
    pub gate_fills: u64,
    /// TLBIs by address that reached no extent of either half, and the
    /// root walks the others' unmapping took (every view's and every
    /// bank's root, for each range).
    pub empty_tlbis: u64,
    pub unmap_walks: u64,
    /// TSC ticks of the shadow's part of TLBIs that did not reach it all.
    pub scoped_ticks: u64,
    /// The TSC ticks of all fills by segment: the fault's walk, its
    /// mapping (with the leaf's prefill), fault-around and restores.
    pub fill_segments: [u64; 3],
    pub stray: u64,
    /// TSC ticks spent in compiled guest code, its exits and faults.
    pub native_ticks: u64,
    /// The TSC at the last native run's entry and return
    /// (`Host::native_stamps`).
    pub native_stamps: [u64; 2],
    /// The runs of compiled code and the time from their returns to the
    /// next entries, by the kind of exit that ended them (`PES EXIT TIME`).
    outside: stela_boot_core::exit_times::ExitTimes,
    /// This core's clock and thermal samples (`PES CLOCK`).
    clock: stela_boot_core::host_clock::Clock,
    #[cfg(feature = "native-profile")]
    native_profile: profile::Profile,
    /// Where the guest's address space lies beside the window.
    placement: Placement,
    /// The translated variant's software TLB, tagged by root, and the
    /// helper's bounce slot; without the slot the host has no helper.
    tlb: Tlb,
    bounce: Option<translated::Bounce>,
    /// Translation helper calls, the TLB fills they made and the accesses
    /// they ran through the bounce slot; shadow faults in the guard.
    pub helper_calls: u64,
    pub helper_fills: u64,
    pub bounces: u64,
    pub guard_faults: u64,
}

fn load_root(root: u64) {
    // SAFETY: every root maps the window, which holds all code and data in use.
    unsafe { asm!("mov cr3, {}", in(reg) root, options(nostack, preserves_flags)) };
}

fn invlpg(address: u64) {
    // SAFETY: invalidating a TLB entry has no other effect.
    unsafe { asm!("invlpg [{}]", in(reg) address, options(nostack, preserves_flags)) };
}

pub struct Config {
    pub ram: Vec<RamRange>,
    pub pool_physical: u64,
    pub pool_bytes: u64,
    /// The loader's root, whose window entries every root shares.
    pub window_root: u64,
    pub(crate) loader_tables: &'static crate::paging::loader::LoaderTables,
    /// Processor-private executable alias; arena writes use the physmap.
    pub code_virtual: u64,
    pub code_physical: u64,
    pub code_bytes: u64,
    pub code_parts: u32,
    /// The board's code window, the same on every executor: (executable
    /// address, physical address, bytes), or `None`.
    pub window: Option<(u64, u64, u64)>,
    /// One common origin for every physical executor.
    pub tsc_base: u64,
    pub tsc_hz: u64,
    pub apic_base: u64,
    pub placement: Placement,
    /// The translation helper's bounce slot: its first page's address and
    /// the physical addresses of its two leaf entries.
    pub bounce: Option<(u64, [u64; 2])>,
}

/// Install the host state, leave the identity map and start the APIC.
///
/// # Safety
/// Runs once on this host processor, after its descriptor setup, with
/// interrupts disabled. Its pool, code arena and bounce entries are private to
/// this processor; the returned owner outlives every local native call/handler.
/// This owned post-EBS mount adopts stable relevant ISA/vendor/identity
/// declarations as a supported-platform requirement: trusted firmware and any
/// VMM preserve them throughout ownership. Stop and retire the owner before
/// physical reset/resume, microcode update or feature reconfiguration, including
/// SMT-shared changes; reinstall and re-enumerate afterwards. ExitBootServices
/// and CPUID do not prove that external promise or hardware validation.
pub unsafe fn install(config: Config) -> Result<HostOwner> {
    let code_part_bytes = code_part_bytes(config.code_bytes, config.code_parts)?;
    // The kernel's half takes what it needs of the shadow part, the user
    // half's cache the rest; a smaller pool splits in their sizes' ratio.
    let guest: u64 = config.ram.iter().map(|range| range.bytes).sum();
    let shadow = config.pool_bytes.saturating_sub(PERMANENT_FRAMES * PAGE);
    let upper = stela_boot_core::host_memory::upper_half(shadow, guest, PAGE);
    let mut pool = Pool {
        base: config.pool_physical,
        bytes: config.pool_bytes,
        upper,
        permanent: 0,
        shadow: [0; 2],
        shadow_base: config.pool_physical + PERMANENT_FRAMES * PAGE,
        half: LOWER,
        peak: [0; 2],
        taken: 0,
        free: [Vec::new(), Vec::new()],
    };
    pool.reserve_free();
    for half in [LOWER, UPPER] {
        pool.reset_half(half);
    }
    let (first, last) = layout::window_entries();
    if config.loader_tables.root() != config.window_root {
        return Err(refuse!(
            "loader root descriptor differs from its original owner"
        ));
    }
    let mut root = || -> Result<u64> {
        let frame = pool
            .permanent()
            .ok_or(refuse!("page-table pool too small"))?;
        for entry in first..=last {
            let value = config.loader_tables.boot_entry(entry).map_err(|_| {
                refuse!("original loader table context changed before host installation")
            })?;
            // SAFETY: this processor's fresh permanent pool allocation.
            unsafe {
                (layout::physical(frame) as *mut u64)
                    .add(entry)
                    .write(value)
            };
        }
        Ok(frame)
    };
    let host_root = root()?;
    for index in 0..SETS * ROOTS {
        let frame = root()?;
        assert_eq!(frame, host_root + (index as u64 + 1) * PAGE);
    }
    let sets_base = host_root + PAGE;
    let roots = core::array::from_fn(|view| sets_base + view as u64 * PAGE);
    let mut sets = Vec::new();
    sets.try_reserve_exact(SETS)
        .map_err(|_| refuse!("root set allocation failed"))?;
    sets.extend((0..SETS).map(|_| Set {
        key: None,
        extents: Extents::new(),
        lower_used: [0; ROOT_ENTRIES / 64],
        used: 0,
    }));
    let native_sets = table_switch::Sets {
        base: sets_base,
        current: sets_base,
        ..table_switch::Sets::default()
    };
    load_root(host_root);
    let apic = Apic::new(
        config.apic_base,
        layout::physical(config.apic_base & apic::BASE_ADDRESS),
    );
    let ring = crate::native_idle::Ring::of(apic.id());
    // SAFETY: this processor owns its disjoint host state until the owner drops.
    let owner = unsafe {
        local::install(State {
            ram: config.ram,
            registered_ram: None,
            pool,
            host_root,
            window_root: config.window_root,
            _loader_tables: config.loader_tables,
            roots,
            sets,
            set: 0,
            set_uses: 0,
            next_class: 1,
            native_sets,
            bank_switches: 0,
            bank_misses: 0,
            view_switches: 0,
            contexts: Contexts {
                tagged: false,
                invpcid: crate::native_mmio::shootdown::invpcid(),
                pan_flag: false,
            },
            legacy_translations_dirty: false,
            current: usize::MAX,
            native_view: 0,
            regime: None,
            tag_checks: [false; 2],
            code_virtual: config.code_virtual,
            code_physical: config.code_physical,
            code_bytes: config.code_bytes,
            code_part_bytes,
            code_parts: u64::from(config.code_parts),
            window_virtual: config.window.map_or(0, |window| window.0),
            window_physical: config.window.map_or(0, |window| window.1),
            window_bytes: config.window.map_or(0, |window| window.2),
            code_unit: 0,
            installed_extents: core::array::from_fn(|_| CodeExtents::default()),
            code_peak: 0,
            installed_immutable: InstalledCodeCache::default(),
            recorded: None,
            native: false,
            tsc_base: config.tsc_base,
            tsc_hz: config.tsc_hz,
            counter_per_tsc: u64::try_from(
                (u128::from(COUNTER_HZ) << 64) / u128::from(config.tsc_hz),
            )
            .expect("the TSC is faster than the 24 MHz counter"),
            apic,
            apic_hz: 0,
            timer_lvt: None,
            ring,
            fills: 0,
            native_ticks: 0,
            native_stamps: [0; 2],
            outside: stela_boot_core::exit_times::ExitTimes::new(),
            clock: stela_boot_core::host_clock::Clock::new(
                stela_boot_core::host_clock::Features::read(),
            ),
            #[cfg(feature = "native-profile")]
            native_profile: profile::Profile::default(),
            resets: 0,
            exhausted: 0,
            lower_resets: 0,
            reset_causes: [0; CAUSES],
            control_fields: [0; FIELDS],
            last_controls: None,
            extents: [Extents::new(), Extents::new()],
            lower_globals: None,
            lower_used: [0; ROOT_ENTRIES / 64],
            upper_used: [0; ROOT_ENTRIES / 64],
            window_contexts: [0; (SETS * ROOTS).div_ceil(64)],
            lower_switches: 0,
            lower_found: 0,
            native_switches: 0,
            kept_dropped: 0,
            scoped: 0,
            reclaimed: 0,
            restored: 0,
            prefills: 0,
            around: 0,
            fill_ticks: 0,
            faults: 0,
            abandoned: [0; ABANDONED],
            fault_ticks: 0,
            fill_causes: [0; shadow_fills::Cause::COUNT],
            history: shadow_fills::History::new(HISTORY_BITS),
            walks: 0,
            table_walks: 0,
            gate_fills: 0,
            empty_tlbis: 0,
            unmap_walks: 0,
            scoped_ticks: 0,
            fill_segments: [0; 3],
            stray: 0,
            placement: config.placement,
            tlb: Tlb::default(),
            bounce: config
                .bounce
                .map(|(address, entries)| translated::Bounce::new(address, entries)),
            helper_calls: 0,
            helper_fills: 0,
            bounces: 0,
            guard_faults: 0,
        })
    };
    let state = state();
    state.apic_hz = state
        .apic
        .start(SPURIOUS_VECTOR, TIMER_VECTOR, state.tsc_hz);
    Ok(owner)
}

/// The host's counters for the statistics line.
pub struct Statistics {
    /// Shadow faults filled, the pages beside them mapped from the same
    /// leaves, and those mapped from the window's other leaves.
    pub fills: u64,
    pub prefills: u64,
    pub around: u64,
    /// Nanoseconds the filled faults' handlers took, entry to return
    /// (without the exception's delivery and IRETQ).
    pub fill_ns: u64,
    /// Every #PF from native code, those that ended the block by cause
    /// ([`ABANDONED`]) and the nanoseconds of all of them.
    pub faults: u64,
    pub abandoned: [u64; ABANDONED],
    pub fault_ns: u64,
    /// Fills by [`shadow_fills::Cause`].
    pub fill_causes: [u64; shadow_fills::Cause::COUNT],
    /// Guest walks for fills and fault-around, and those that started at
    /// the fault's leaf table; the shadow frames the pool handed out.
    pub walks: u64,
    pub table_walks: u64,
    pub frames: u64,
    /// Kept lower halves whose frames went back to the pool, and pages
    /// mapped again from the fill history.
    pub reclaimed: u64,
    pub restored: u64,
    /// Fills the #PF gate made, and the nanoseconds of all fills by
    /// segment ([`State::fill_segments`]).
    pub gate_fills: u64,
    pub fill_segment_ns: [u64; 3],
    /// [`State::empty_tlbis`] and [`State::unmap_walks`].
    pub empty_tlbis: u64,
    pub unmap_walks: u64,
    /// Nanoseconds of [`State::scoped_ticks`].
    pub scoped_ns: u64,
    /// Resets of the whole shadow, those because a half ran out of frames,
    /// of the lower half alone, and TLBIs that unmapped ranges instead.
    pub resets: u64,
    pub exhausted: u64,
    pub lower_resets: u64,
    /// Switches of the lower half to another TTBR0, those that found its
    /// set holding a half, those native code made, and the other sets'
    /// halves dropped.
    pub lower_switches: u64,
    pub lower_found: u64,
    pub native_switches: u64,
    pub kept_dropped: u64,
    /// CR3 loads that flushed a stale view ([`table_switch::Sets::flushes`]).
    pub flushed_loads: u64,
    /// Whole resets by cause: exhausted frames, SPRR, TTBR0 and other
    /// controls, tag checks, invalidations and TLBIs of everything.
    pub reset_causes: [u64; CAUSES],
    /// The other-controls resets each field differed in: SCTLR, TCR, TTBR0,
    /// TTBR1, MAIR, SPRR, CTRR, the regime and stage 2.
    pub control_fields: [u64; FIELDS],
    /// The controls the last of those went from and to.
    pub last_controls: Option<(mmu::Controls, mmu::Controls)>,
    /// Switches to another SPRR_PERM_EL1 value's kernel roots, and those
    /// that found none kept.
    pub bank_switches: u64,
    pub bank_misses: u64,
    pub view_switches: u64,
    pub scoped: u64,
    pub stray: u64,
    /// The APIC timer frequency.
    pub apic_hz: u64,
    /// Translation helper calls, their TLB fills and bounced accesses, and
    /// shadow faults in the four-level guard.
    pub helper_calls: u64,
    pub helper_fills: u64,
    pub bounces: u64,
    pub guard_faults: u64,
    /// Each half's shadow bytes and the most it has held at once.
    pub shadow_bytes: [u64; 2],
    pub shadow_peak: [u64; 2],
    /// The code arena's bytes and the furthest its code has reached.
    pub code_bytes: u64,
    pub code_peak: u64,
}

pub fn statistics() -> Statistics {
    let state = state();
    Statistics {
        fills: state.fills,
        prefills: state.prefills,
        around: state.around,
        fill_ns: (u128::from(state.fill_ticks) * 1_000_000_000 / u128::from(state.tsc_hz.max(1)))
            as u64,
        faults: state.faults,
        abandoned: state.abandoned,
        fault_ns: (u128::from(state.fault_ticks) * 1_000_000_000 / u128::from(state.tsc_hz.max(1)))
            as u64,
        fill_causes: state.fill_causes,
        walks: state.walks,
        table_walks: state.table_walks,
        frames: state.pool.taken,
        reclaimed: state.reclaimed,
        restored: state.restored,
        gate_fills: state.gate_fills,
        empty_tlbis: state.empty_tlbis,
        unmap_walks: state.unmap_walks,
        scoped_ns: (u128::from(state.scoped_ticks) * 1_000_000_000
            / u128::from(state.tsc_hz.max(1))) as u64,
        fill_segment_ns: state.fill_segments.map(|ticks| {
            (u128::from(ticks) * 1_000_000_000 / u128::from(state.tsc_hz.max(1))) as u64
        }),
        resets: state.resets,
        exhausted: state.exhausted,
        lower_resets: state.lower_resets,
        lower_switches: state.lower_switches,
        lower_found: state.lower_found,
        native_switches: state.native_switches,
        flushed_loads: state.native_sets.flushes,
        kept_dropped: state.kept_dropped,
        reset_causes: state.reset_causes,
        control_fields: state.control_fields,
        last_controls: state.last_controls,
        bank_switches: state.bank_switches,
        bank_misses: state.bank_misses,
        view_switches: state.view_switches,
        scoped: state.scoped,
        stray: state.stray,
        apic_hz: state.apic_hz,
        helper_calls: state.helper_calls,
        helper_fills: state.helper_fills,
        bounces: state.bounces,
        guard_faults: state.guard_faults,
        shadow_bytes: [LOWER, UPPER].map(|half| {
            let (start, end) = state.pool.range(half);
            end - start
        }),
        shadow_peak: state.pool.peak,
        code_bytes: state.code_bytes,
        code_peak: state.code_peak,
    }
}

/// Serialize before the executable alias runs bytes just written through
/// the physmap alias. IRETQ to the next instruction serializes as CPUID
/// does, but never exits to a hypervisor, which intercepts every CPUID: a
/// block install and each link would otherwise pay a VM exit (Linux
/// `iret_to_self`).
fn serialize_code() {
    // SAFETY: CPL0 in long mode, with no red zone. The frame IRETQ pops
    // holds the current SS, RSP, RFLAGS and CS and the next instruction, so
    // it changes nothing but the stack below RSP.
    unsafe {
        core::arch::asm!(
            "mov {frame:e}, ss",
            "push {frame}",
            "lea {frame}, [rsp + 8]",
            "push {frame}",
            "pushfq",
            "mov {frame:e}, cs",
            "push {frame}",
            "lea {frame}, [rip + 2f]",
            "push {frame}",
            "iretq",
            "2:",
            frame = out(reg) _,
        );
    }
}

/// TSC ticks spent in compiled guest code so far.
pub fn native_ticks() -> u64 {
    state().native_ticks
}

/// This core's clock and thermal samples so far, after a sample when the
/// last is old enough (`stela_boot_core::host_clock`).
pub(super) fn clock() -> stela_engine::machine::engine::HostClock {
    // SAFETY: CPL0 on this core, whose own enumeration its State holds.
    state().clock.sample(rdtsc(), |features| unsafe {
        stela_boot_core::host_clock::Reading::take(features)
    })
}

/// From `now` (a TSC reading) this core does the host's work between two
/// slices of guest execution, which is no exit's time (`PES EXIT TIME`),
/// until [`outside_resume`]. Nothing on a core that runs no guest code.
pub(super) fn outside_pause(now: u64) {
    if let Some(state) = local::installed() {
        state.outside.pause(now);
    }
}

/// This core runs its guest again from `now`: the time of the exit that
/// ended its last run of compiled code counts again.
pub(super) fn outside_resume(now: u64) {
    if let Some(state) = local::installed() {
        state.outside.resume(now);
    }
}

/// Serialize both sides of the reference-counter observation. Compiler fences
/// also prevent moving memory work across these opt-in profiling boundaries.
#[cfg(feature = "native-profile")]
fn profile_timestamp() -> u64 {
    use core::sync::atomic::{Ordering, compiler_fence};
    compiler_fence(Ordering::SeqCst);
    core::arch::x86_64::__cpuid(0);
    // SAFETY: the installed native Host runs at CPL0 with its live CPU owner.
    let value = unsafe { core::arch::x86_64::_rdtsc() };
    core::arch::x86_64::__cpuid(0);
    compiler_fence(Ordering::SeqCst);
    value
}

#[cfg(feature = "native-profile")]
pub(crate) fn profile_snapshot() -> profile::Snapshot {
    state().native_profile.snapshot()
}

/// Inclusive work counters keep the outer phase, so native helper calls in
/// stopped services do not masquerade as checked-body execution time.
#[cfg(feature = "native-profile")]
pub(crate) fn profile_work_report() {
    let snapshot = profile_snapshot();
    for work in stela_engine::machine::engine::NativeTimingWork::ALL {
        let count = snapshot.work[work as usize];
        crate::output::arguments(format_args!(
            "NATIVE WORK sampled_bsp category={work:?} calls={} sampled_calls={} ticks={} phase_ticks={:?}\n",
            count.calls, count.sampled_calls, count.sampled_ticks, count.phase_ticks
        ));
    }
}

#[cfg(feature = "native-profile")]
pub(crate) fn profile_word_report() {
    use profile::words::Ranking;
    crate::output::arguments(format_args!(
        "NATIVE WORD coverage_sampled_bsp {:?}\n",
        state().native_profile.snapshot().words,
    ));
    for ranking in [
        Ranking::Frequency,
        Ranking::ObservedBytes,
        Ranking::MaximumBytes,
    ] {
        for row in state()
            .native_profile
            .top_words(ranking)
            .into_iter()
            .flatten()
        {
            crate::output::arguments(format_args!(
                "NATIVE WORD sampled_bsp rank={ranking:?} word={:#010x} event={:?} count_lower={} count_upper={} observed_attributed_bytes={} max_attributed_bytes={}\n",
                row.word,
                row.event,
                row.upper.saturating_sub(row.error),
                row.upper,
                row.observed_bytes,
                row.max_bytes,
            ));
        }
    }
}

impl State {
    fn owns_installed(&self, code: NativeCode, bytes: usize) -> bool {
        if bytes == 0 {
            return false;
        }
        let Some(offset) = (code.0 as u64).checked_sub(self.code_virtual) else {
            return false;
        };
        let Some(end) = offset.checked_add(bytes as u64) else {
            return false;
        };
        let unit = self.code_part_bytes;
        if unit == 0 || end > self.code_bytes {
            return false;
        }
        let Some(extents) = self.installed_extents.get((offset / unit) as usize) else {
            return false;
        };
        let base = offset / unit * unit;
        extents.contains(offset - base..end - base)
    }

    fn host_physical(&self, guest: u64) -> Option<u64> {
        self.run(guest).map(|(host, _)| host)
    }

    /// The longest RAM run starting at `guest`, as (host, bytes).
    fn run(&self, guest: u64) -> Option<(u64, u64)> {
        ram::run(&self.ram, guest)
    }

    fn copy(&self, guest: u64, length: usize, mut action: impl FnMut(u64, usize, usize)) -> bool {
        ram::copy(&self.ram, guest, length, |host, done, chunk| {
            action(layout::physical(host), done, chunk);
        })
    }

    fn reset_shadow(&mut self, cause: Cause) {
        if !self.legacy_translations_dirty {
            // Device translations can exist without any populated RAM view.
            self.tlb.clear_devices();
            return;
        }
        self.clear(LOWER);
        self.clear(UPPER);
        self.tlb.clear();
        self.resets += 1;
        self.reset_causes[cause as usize] += 1;
        self.reload();
        self.legacy_translations_dirty = false;
    }

    /// A TLBI ASIDE1 of the current ASID: it removes the non-global entries
    /// of that ASID alone (Arm `TLBIMatch`), so of the current shadow only
    /// a half that holds a non-global translation goes
    /// (`mmu::shadow::Extents::private`); the kernel's global upper half
    /// stays. Other sets' halves were dropped by their own ASIDs
    /// ([`State::invalidate_other_sets`]).
    fn drop_non_global(&mut self) {
        let mut cleared = false;
        if self.extents[LOWER].private() {
            if let Some((key, _)) = self.sets[self.set].key {
                self.history.switched(key, key, false);
            }
            self.drop_lower(self.set);
            cleared = true;
        }
        if self.extents[UPPER].private() {
            self.clear(UPPER);
            cleared = true;
        }
        self.tlb.clear();
        if cleared {
            self.resets += 1;
            self.reset_causes[Cause::Maintained as usize] += 1;
            self.reload();
        }
    }

    /// Drop every set's lower half.
    fn reset_lower(&mut self) {
        self.clear(LOWER);
        self.tlb.clear();
        self.lower_resets += 1;
        self.reload();
    }

    /// Root `view` of set `set`.
    fn root(&self, set: usize, view: usize) -> u64 {
        self.host_root + (1 + set * ROOTS + view) as u64 * PAGE
    }

    /// Entry `entry` of root `root`, through the physmap.
    fn root_entry(root: u64, entry: usize) -> *mut u64 {
        // SAFETY: permanent root frames, whose 512 entries the physmap maps.
        unsafe { (layout::physical(root) as *mut u64).add(entry) }
    }

    /// Make the set of lower context `key` and `rights` current, from the
    /// least recently used other set where none holds them; whether the
    /// current set changed. The caller loads the view.
    fn select_set(&mut self, key: LowerKey, rights: Rights) -> bool {
        let wanted = Some((key, rights));
        let Some((from, from_rights)) = self.sets[self.set].key else {
            // The first regime: the current set takes it.
            self.label(self.set, key, rights);
            return false;
        };
        if self.sets[self.set].key == wanted {
            return false;
        }
        let found = self.sets.iter().position(|set| set.key == wanted);
        let target = match found {
            Some(target) => target,
            None => self.make_set(key, rights),
        };
        let kept = self.enter_set(target);
        if from != key {
            self.lower_switches += 1;
            self.lower_found += u64::from(kept);
            self.history.switched(from, key, kept);
        }
        if from_rights != rights {
            self.bank_switches += 1;
            self.bank_misses += u64::from(found.is_none());
        }
        true
    }

    /// Make set `target` current: the current one keeps its lower half's
    /// records, `target`'s become the State's. Whether its lower half
    /// holds anything.
    fn enter_set(&mut self, target: usize) -> bool {
        let current = self.set;
        self.sets[current].extents = core::mem::take(&mut self.extents[LOWER]);
        self.sets[current].lower_used = self.lower_used;
        self.extents[LOWER] = core::mem::take(&mut self.sets[target].extents);
        self.lower_used = self.sets[target].lower_used;
        self.set = target;
        self.set_uses += 1;
        self.sets[target].used = self.set_uses;
        self.roots = core::array::from_fn(|view| self.root(target, view));
        self.native_sets.current = self.roots[0];
        self.lower_used != [0; ROOT_ENTRIES / 64]
    }

    /// A set for `key` and `rights`: a free one or the least recently used
    /// other one, whose lower half goes back to the pool, with the upper
    /// half's entries of another set of `rights`, or none.
    fn make_set(&mut self, key: LowerKey, rights: Rights) -> usize {
        let target = (0..SETS)
            .filter(|&set| set != self.set)
            .min_by_key(|&set| (self.sets[set].key.is_some(), self.sets[set].used))
            .expect("more than one set");
        if self.sets[target].key.is_some() {
            self.kept_dropped += 1;
            self.drop_lower(target);
        }
        let source = (0..SETS).find(|&set| {
            set != target && self.sets[set].key.is_some_and(|(_, held)| held == rights)
        });
        let upper = self.upper_used;
        for view in 0..ROOTS {
            let root = self.root(target, view);
            let from = source.map(|set| self.root(set, view));
            for entry in self.used_entries(upper, UPPER) {
                // SAFETY: entries of permanent roots.
                let value = from.map_or(0, |from| unsafe { Self::root_entry(from, entry).read() });
                // SAFETY: as above; the target is no current root.
                unsafe { Self::root_entry(root, entry).write(value) };
            }
        }
        self.label(target, key, rights);
        target
    }

    /// Give set `set` to `key` and `rights`: a new generation, the class
    /// of the sets that differ from it in TTBR0 alone, and stale views.
    fn label(&mut self, set: usize, key: LowerKey, rights: Rights) {
        self.sets[set].key = Some((key, rights));
        let class = (0..SETS)
            .filter(|&other| other != set)
            .find(|&other| {
                self.sets[other]
                    .key
                    .is_some_and(|((_, tcr), held)| held == rights && tcr == key.1)
            })
            .map(|other| self.native_sets.classes[other]);
        self.native_sets.classes[set] = class.unwrap_or_else(|| {
            let class = self.next_class;
            self.next_class += 1;
            class
        });
        self.native_sets.generations[set] += 1;
        self.stale_set(set);
    }

    /// Set `set`'s views may hold what was removed: each flushes as it
    /// next loads.
    fn stale_set(&mut self, set: usize) {
        self.native_sets.stale[set * ROOTS..(set + 1) * ROOTS].fill(1);
    }

    /// Every set but the current one ([`State::stale_set`]).
    fn stale_other_sets(&mut self) {
        let current = self.set;
        for set in (0..SETS).filter(|&set| set != current) {
            self.stale_set(set);
        }
    }

    /// Drop `pages`, of the upper half every set may share, from every
    /// other set's views: those whose context holds anything by INVPCID,
    /// until [`OTHER_SET_INVPCIDS`] are spent, and the rest flush when
    /// next loaded.
    fn forget_other_sets(&mut self, pages: &[u64]) {
        let mut budget = OTHER_SET_INVPCIDS;
        let current = self.set;
        for index in (0..SETS * ROOTS).filter(|index| index / ROOTS != current) {
            if self.native_sets.stale[index] != 0 {
                continue;
            }
            match budget.checked_sub(pages.len()) {
                Some(left) if self.contexts.invpcid => {
                    budget = left;
                    for &page in pages {
                        self.forget_inactive(page, index);
                    }
                }
                _ => self.native_sets.stale[index] = 1,
            }
        }
    }

    /// Give the lower half's pool back the tables of set `set`'s lower half
    /// and detach it from the set's roots, whose views are stale after.
    /// The current set's view must load again before native code runs.
    fn drop_lower(&mut self, set: usize) {
        let current = set == self.set;
        let used = if current {
            self.extents[LOWER].clear();
            core::mem::take(&mut self.lower_used)
        } else {
            self.sets[set].extents.clear();
            core::mem::take(&mut self.sets[set].lower_used)
        };
        let mut free = core::mem::take(&mut self.pool.free[LOWER]);
        for view in 0..ROOTS {
            let root = self.root(set, view);
            for entry in self.used_entries(used, LOWER) {
                let slot = Self::root_entry(root, entry);
                // SAFETY: an entry of a permanent root.
                let value = unsafe { slot.read() };
                if value == 0 {
                    continue;
                }
                tables::tables_under(&self.pool, value, &mut |frame| {
                    // Room for every frame of the half was reserved; a
                    // frame beyond it would be one given back twice.
                    if free.len() < free.capacity() {
                        free.push(frame);
                    }
                });
                // SAFETY: as above.
                unsafe { slot.write(0) };
            }
        }
        self.pool.free[LOWER] = free;
        self.reclaimed += 1;
        self.stale_set(set);
    }

    /// The least recently used set but the current one whose lower half
    /// holds anything, for its frames: what a lower half out of frames
    /// takes before it empties every set's.
    fn oldest_lower(&self) -> Option<usize> {
        (0..SETS)
            .filter(|&set| set != self.set && self.sets[set].lower_used != [0; ROOT_ENTRIES / 64])
            .min_by_key(|&set| self.sets[set].used)
    }

    /// Apply a TLBI that removes `everywhere` to every set's lower half but
    /// the current one's, within `bounds`, under the TCR `tcr` the sets
    /// share outside the lower half's fields: as `mmu::shadow::Kept` did
    /// for detached halves. A TLBI by VA of an ASID reaches the global
    /// pages of a half that may hold no entry of that ASID
    /// ([`shadow::same_asid`]) and every page at its addresses of one that
    /// may; the views of a set that lost a page are stale. A half whose
    /// extents are unknown for the TLBI's addresses, or that a TLBI of
    /// non-global or all entries reaches, goes.
    fn invalidate_other_sets(&mut self, everywhere: Everywhere, tcr: u64, bounds: (u64, u64)) {
        let (reach, asid) = match everywhere {
            Everywhere::Addresses(reach) => (reach, None),
            Everywhere::AsidAddresses(reach, asid) => (reach, Some(asid)),
            Everywhere::NonGlobal(_) | Everywhere::All => (Reach::All, None),
        };
        for set in 0..SETS {
            if set == self.set || self.sets[set].lower_used == [0; ROOT_ENTRIES / 64] {
                continue;
            }
            let Some(((ttbr0, _), _)) = self.sets[set].key else {
                continue;
            };
            let keep = match everywhere {
                Everywhere::NonGlobal(asid) => {
                    !(self.sets[set].extents.private() && shadow::same_asid(ttbr0, tcr, asid))
                }
                _ => {
                    let other = asid.is_some_and(|asid| {
                        self.sets[set].extents.private() && !shadow::same_asid(ttbr0, tcr, asid)
                    });
                    let mut ranges = Vec::new();
                    let extents = &mut self.sets[set].extents;
                    let kept = if other {
                        extents.visit_global_ranges(reach, bounds, |first, last| {
                            ranges.push((first, last));
                        })
                    } else {
                        extents.visit_ranges(reach, bounds, |first, last| {
                            ranges.push((first, last));
                        })
                    };
                    let mut removed = false;
                    for (first, last) in ranges {
                        for view in 0..ROOTS {
                            self.unmap_walks += 1;
                            tables::unmap(&self.pool, self.root(set, view), first, last, |_| {
                                removed = true
                            });
                        }
                    }
                    if removed {
                        self.stale_set(set);
                    }
                    kept
                }
            };
            if !keep {
                self.kept_dropped += 1;
                self.drop_lower(set);
            }
        }
    }

    /// The entries `used` names of `half`'s root entries.
    fn used_entries(
        &self,
        used: [u64; ROOT_ENTRIES / 64],
        half: usize,
    ) -> impl Iterator<Item = usize> + use<> {
        let (first, last) = self.half_entries(half);
        used.into_iter()
            .enumerate()
            .flat_map(|(word, mut bits)| {
                core::iter::from_fn(move || {
                    let bit = (bits != 0).then(|| bits.trailing_zeros() as usize)?;
                    bits &= bits - 1;
                    Some(word * 64 + bit)
                })
            })
            .filter(move |entry| (first..=last).contains(entry))
    }

    /// The current set's root entry `entry` of the lower half may be in use.
    fn record_lower(&mut self, entry: usize) {
        self.lower_used[entry / 64] |= 1 << (entry % 64);
    }

    /// A fill made the current set's upper root entry `entry` of root
    /// `view` `value`: every set of the same rights shares the table.
    fn share_upper(&mut self, view: usize, entry: usize, value: u64) {
        self.upper_used[entry / 64] |= 1 << (entry % 64);
        let Some((_, rights)) = self.sets[self.set].key else {
            return;
        };
        for set in 0..SETS {
            if set != self.set && self.sets[set].key.is_some_and(|(_, held)| held == rights) {
                // SAFETY: an entry of a permanent root.
                unsafe { Self::root_entry(self.root(set, view), entry).write(value) };
            }
        }
    }

    /// Every root a set may load.
    fn all_roots(&self) -> impl Iterator<Item = u64> + use<> {
        let base = self.host_root + PAGE;
        (0..(SETS * ROOTS) as u64).map(move |index| base + index * PAGE)
    }

    /// Unmap `half` from every set and free its frames.
    fn clear(&mut self, half: usize) {
        self.pool.reset_half(half);
        self.extents[half].clear();
        self.history.emptied(half);
        if half == LOWER {
            for set in 0..SETS {
                let used = if set == self.set {
                    core::mem::take(&mut self.lower_used)
                } else {
                    self.sets[set].extents.clear();
                    core::mem::take(&mut self.sets[set].lower_used)
                };
                self.detach(set, used, LOWER);
            }
        } else {
            let used = core::mem::take(&mut self.upper_used);
            for set in (0..SETS).filter(|&set| self.sets[set].key.is_some() || set == self.set) {
                self.detach(set, used, UPPER);
            }
        }
    }

    /// Unmap the entries `used` names of `half` from set `set`'s roots,
    /// leaving their tables' frames taken.
    fn detach(&self, set: usize, used: [u64; ROOT_ENTRIES / 64], half: usize) {
        for view in 0..ROOTS {
            let root = self.root(set, view);
            for entry in self.used_entries(used, half) {
                // SAFETY: an entry of a permanent root.
                unsafe { Self::root_entry(root, entry).write(0) };
            }
        }
    }

    /// The first and last root entries of `half` of the guest's native
    /// ranges: under LA57 PML5 entry 0 or 511; on a four-level host the
    /// PML4 entries of the lower half, or those from `U` up.
    fn half_entries(&self, half: usize) -> (usize, usize) {
        match (self.placement, half) {
            (Placement::FiveLevel, LOWER) => (0, 0),
            (Placement::FiveLevel, _) => (ROOT_ENTRIES - 1, ROOT_ENTRIES - 1),
            (Placement::FourLevel { .. }, LOWER) => (0, ROOT_ENTRIES / 2 - 1),
            (Placement::FourLevel { .. }, _) => (
                ((self.placement.native_upper() >> PML4_ENTRY_BITS) as usize) % ROOT_ENTRIES,
                ROOT_ENTRIES - 1,
            ),
        }
    }

    /// The first and last address of `half` of the native ranges.
    fn half_range(&self, half: usize) -> (u64, u64) {
        match (self.placement, half) {
            (Placement::FiveLevel, LOWER) => (0, FIVE_LEVEL_LOWER_END),
            (Placement::FourLevel { .. }, LOWER) => (0, FOUR_LEVEL_LOWER_END),
            _ => (self.placement.native_upper(), u64::MAX),
        }
    }

    /// The root `view` runs in: its own, or under [`Contexts::pan_flag`] for
    /// EL1 with PAN EL1's, whose tables hold the same leaves.
    fn root_of(&self, view: usize) -> usize {
        if self.contexts.pan_flag && view % VIEWS == PAN_VIEW {
            view - PAN_VIEW + EL1_VIEW
        } else {
            view
        }
    }

    /// The index among every set's roots of the current set's root of the
    /// current view, if there is a view.
    fn current_index(&self) -> Option<usize> {
        (self.current < ROOTS).then(|| self.set * ROOTS + self.root_of(self.current))
    }

    /// What native code left the host: a switch to another set
    /// ([`table_switch::Sets::switched`]), then the view and requester a
    /// native transition moved to (`native_view`). The walks and the view
    /// follow, as the engine's next regime will.
    fn follow_native(&mut self) {
        let switched = core::mem::take(&mut self.native_sets.switched);
        if switched != 0 {
            self.follow_switch(switched as usize - 1);
        }
        let word = core::mem::take(&mut self.native_view);
        if word & NATIVE_VIEW == 0 {
            return;
        }
        self.current = (word & 0xff) as usize;
        if let Some(regime) = &mut self.regime {
            regime.requester = mmu::Requester {
                el: (word >> 8 & 3) as u8,
                pan: word >> 10 & 1 != 0,
                unprivileged: word >> 12 & 1 != 0,
                guarded: word >> 11 & 1 != 0,
            };
        }
    }

    /// Native code switched TTBR0 to set `target`'s context and loaded its
    /// root (`x86_64::table_switch`), which differs from the current set
    /// in TTBR0 alone: it becomes current as an engine-path switch makes
    /// it, and the regime's TTBR0 is its.
    fn follow_switch(&mut self, target: usize) {
        if target >= SETS || target == self.set {
            return;
        }
        let from = self.sets[self.set].key;
        let to = self.sets[target].key;
        let kept = self.enter_set(target);
        self.lower_switches += 1;
        self.lower_found += u64::from(kept);
        self.native_switches += 1;
        if let (Some((from, _)), Some((to, _))) = (from, to) {
            self.history.switched(from, to, kept);
            if let Some(regime) = &mut self.regime {
                regime.controls.ttbr0 = to.0;
            }
        }
        // The helper's software TLB keys no TTBR0.
        self.tlb.clear();
    }

    /// Whether native switches may load other sets: the upper half's
    /// translations must not depend on the ASID a switch may change.
    fn update_open(&mut self) {
        self.native_sets.open = u64::from(self.regime.is_some() && !self.extents[UPPER].private());
    }

    /// PSTATE.PAN is `pan`, as native code set it behind the engine, whose
    /// next regime says the same: the view and the walks' requester follow.
    fn follow_pan(&mut self, pan: bool) {
        let view = self.current % VIEWS;
        if self.contexts.pan_flag && (view == EL1_VIEW || view == PAN_VIEW) {
            self.current = self.current - view + if pan { PAN_VIEW } else { EL1_VIEW };
            if let Some(regime) = &mut self.regime {
                regime.requester.pan = pan;
            }
        }
    }

    /// Set RFLAGS.AC for a native run in the current view, where it only
    /// matters in EL1's root: clear for PAN, which keeps out of user pages.
    fn load_pan(&self) {
        if !self.contexts.pan_flag {
            return;
        }
        // SAFETY: CPL0 with SMAP, which `pan_flag` says this core has. Only
        // AC changes, and no code between here and the guest's reads it.
        unsafe {
            if self.current % VIEWS == PAN_VIEW {
                asm!("clac", options(nomem, nostack));
            } else {
                asm!("stac", options(nomem, nostack));
            }
        }
    }

    /// Load the current set's root of `view` in its context, which keeps
    /// what the context holds unless it is stale.
    fn load_view(&mut self, view: usize) {
        let root_view = self.root_of(view);
        let root = self.roots[root_view];
        if !self.contexts.tagged {
            return load_root(root);
        }
        let index = self.set * ROOTS + root_view;
        let stale = core::mem::replace(&mut self.native_sets.stale[index], 0) != 0;
        let keep = if stale {
            self.native_sets.flushes += 1;
            0
        } else {
            super::storage::KEEP_ENTRIES
        };
        load_root(root | root_context(index) | keep);
    }

    /// Flush the TLB of every view of every set: the current root's context
    /// now, the others' when next loaded.
    fn reload(&mut self) {
        self.native_sets.stale.fill(1);
        if self.current < ROOTS {
            self.load_view(self.current);
        } else {
            load_root(self.host_root);
        }
    }

    /// Drop a page from root `index`'s context while it is inactive, which
    /// keeps its translations across CR3 loads only while tagged and fresh.
    fn forget_inactive(&mut self, page: u64, index: usize) {
        if !self.contexts.tagged || self.native_sets.stale[index] != 0 {
            return;
        }
        if self.contexts.invpcid {
            let descriptor = [root_context(index), page];
            // SAFETY: a valid single-address INVPCID of an inactive PCID.
            unsafe {
                asm!("invpcid {}, [{}]", in(reg) 0u64, in(reg) &descriptor,
                    options(nostack, preserves_flags))
            };
        } else {
            self.native_sets.stale[index] = 1;
        }
    }

    /// The loaded root's context may now hold the window's translations
    /// ([`State::forget_window`]).
    fn window_used(&mut self) {
        if let Some(index) = self.current_index() {
            self.window_contexts[index / 64] |= 1 << (index % 64);
        }
    }

    /// Drop `pages` of the window, which every root maps, from every
    /// context that used the window: the loaded one's now, the others'
    /// through INVPCID or when next loaded. None holds the window after.
    fn forget_window(&mut self, pages: [Option<u64>; 2]) {
        if pages == [None; 2] {
            return;
        }
        let current = self.current_index();
        let used = core::mem::take(&mut self.window_contexts);
        for page in pages.into_iter().flatten() {
            invlpg(page);
            for (word, mut bits) in used.into_iter().enumerate() {
                while bits != 0 {
                    let index = word * 64 + bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    if Some(index) != current {
                        self.forget_inactive(page, index);
                    }
                }
            }
        }
    }
}

/// Guest page tables read through the RAM map.
struct GuestTables<'a> {
    state: &'a State,
    descriptors: Option<stela_engine::memory::system_ram::DescriptorReader<'a>>,
}

impl<'a> GuestTables<'a> {
    fn new(state: &'a State) -> Self {
        Self {
            state,
            descriptors: state
                .registered_ram
                .as_ref()
                .map(|ram| ram.descriptor_reader()),
        }
    }
}

impl mmu::TableMemory for GuestTables<'_> {
    fn read_descriptor(&mut self, physical: u64) -> Option<u64> {
        if let Some(descriptors) = &mut self.descriptors {
            return descriptors.read_descriptor(physical).ok();
        }
        let host = self.state.host_physical(physical)?;
        // SAFETY: guest RAM through the physmap.
        Some(unsafe { (layout::physical(host) as *const u64).read_volatile() })
    }
    fn update_descriptor(&mut self, physical: u64, old: u64, new: u64) -> Option<bool> {
        if let Some(descriptors) = &self.descriptors {
            return descriptors.update_descriptor(physical, old, new).ok();
        }
        let host = self.state.host_physical(physical)?;
        let slot = layout::physical(host) as *mut u64;
        // SAFETY: guest RAM through the physmap; one PE, no concurrent walker.
        unsafe {
            if slot.read_volatile() != old {
                return Some(false);
            }
            slot.write_volatile(new);
        }
        Some(true)
    }
}

/// Complete one real owner-local native return, keeping exact fault GPRs.
fn native_return(code: NativeCode, exit: BlockExit, entered: u64) -> NativeExit {
    crate::native_mmio::shootdown::service();
    let state = state();
    state.native = false;
    let returned = rdtsc();
    state.native_ticks += returned - entered;
    state.native_stamps = [entered, returned];
    // The exit before this run took until `entered`; this one's time runs
    // from here to the next entry.
    let kind = if exit.kind == FAULT_KIND {
        stela_boot_core::exit_times::FAULT
    } else {
        stela_compiler::x86_64::chain::ExitWord::split(exit.kind).0 as usize
    };
    state.outside.run(entered, returned, kind);
    #[cfg(feature = "native-profile")]
    state.native_profile.exit(exit.kind);
    if exit.kind == FAULT_KIND
        && let Some(recorded) = state.recorded.take()
    {
        return NativeExit::Fault(NativeFault {
            origin: recorded.origin,
            // A chained block may lie below the entered one.
            offset: recorded.rip.wrapping_sub(code.0 as u64) as usize,
            address: recorded.address,
            write: recorded.write,
            cause: recorded.cause,
            trap: recorded.trap,
            error_code: recorded.error_code,
            spills: stela_compiler::block::HostSpillSnapshot::new(),
            registers: recorded.registers,
        });
    }
    NativeExit::Exit {
        next_pc: exit.next_pc,
        kind: exit.kind,
    }
}
