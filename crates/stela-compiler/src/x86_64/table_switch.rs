//! Native completion of a TTBR0 write that switches the PE to another
//! address space the engine and the host both keep.
//!
//! An MSR TTBR0 ends its block with a System exit. The thunk the engine
//! installs over it completes a write of the value the register holds as
//! [`super::register_read::unchanged_write_thunk`] does, and a write of
//! another value when the PE's [`Table`] holds an [`Entry`] for that value,
//! of the table's generation, made in the state the PE is in (its upper
//! half's lookup context is the entry's `upper`). Where the host keeps root
//! sets ([`Sets`]), the entry's set must also be open, of the generation
//! the entry names and of the class of the set the current CR3 belongs to:
//! the same rights and lower TCR fields, so the set differs from the
//! current one in TTBR0 alone.
//!
//! On a hit the thunk publishes the entry's translating word where the PE
//! publishes one (the regime and ASID other PEs' TLBIs test, as
//! `Machine::publish_translating` does), and if that changed it and mail is
//! then pending, takes the original exit: the engine reads the mail before
//! the new context is used. Otherwise it writes the register, the lower
//! half's ITLB context and salt, closes the cells a translation change
//! closes (`closes`), moves the transition records' contexts generation
//! (records continue only once the engine gave them contexts again), counts
//! the switch for the engine, which settles the rest at its next exit, and
//! loads CR3 with the target set's root of the view the current CR3 is of:
//! in that view's PCID, without the no-flush bit while the view's stale
//! byte is set, which it clears and counts. Where the host isolates predictions
//! between contexts it issues IBPB and refills the return stack, as the
//! engine's exit would. It clobbers RAX, RCX, RDX, R8, R9 and flags, which
//! no exit keeps live.
use super::asm::{Alu, Asm, Cond, Gpr, Mem, Patch, ShiftOp, Size};
use super::pointer::Thunk;
use super::register_read::Written;
use crate::{GuestAddress, Place, block::ExitKind};
use alloc::vec::Vec;
use core::mem::offset_of;

/// The views a set has a root for: four permission views, unchecked and
/// tag-checked.
pub const SET_ROOTS: usize = 8;
/// The root sets a host keeps.
pub const SETS: usize = 40;
/// The PCID of set `s`'s view `v` is `FIRST_CONTEXT + s * SET_ROOTS + v`.
pub const FIRST_CONTEXT: u64 = 2;
/// CR3's no-flush bit.
pub const KEEP_ENTRIES: u64 = 1 << 63;
/// CR3's root address bits.
pub const ROOT_ADDRESS: u64 = 0x000f_ffff_ffff_f000;

/// A host's root sets as native code reads them: set `s`'s view `v` root
/// lies at `base + (s * SET_ROOTS + v) * 4096`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sets {
    /// Nonzero while a native switch may move to another set.
    pub open: u64,
    pub base: u64,
    /// Nonzero while roots load in their PCIDs.
    pub tagged: u64,
    /// One more than the set a native switch moved to since the host last
    /// followed, 0 for none.
    pub switched: u64,
    /// The current set's first root, which transition records check.
    pub current: u64,
    /// CR3 loads that flushed a stale view's PCID, the host's and native
    /// code's.
    pub flushes: u64,
    /// Each set's generation, which moves whenever it is given to another
    /// context, and class: sets of one class differ in TTBR0 alone.
    pub generations: [u64; SETS],
    pub classes: [u64; SETS],
    /// Each set's views whose PCID may hold what was removed since.
    pub stale: [u8; SETS * SET_ROOTS],
}

impl Default for Sets {
    fn default() -> Self {
        Self {
            open: 0,
            base: 0,
            tagged: 0,
            switched: 0,
            current: 0,
            flushes: 0,
            generations: [0; SETS],
            classes: [0; SETS],
            stale: [1; SETS * SET_ROOTS],
        }
    }
}

/// The entries a [`Table`] holds.
pub const ENTRIES: usize = 8;

/// One TTBR0 value a native switch may write.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Entry {
    /// The table's generation it holds in; 0 never does.
    pub generation: u64,
    /// TTBR0 as the register takes it.
    pub value: u64,
    /// The upper half's lookup context (`ChainControl::contexts[1]`) it
    /// holds under, which a TTBR0 write leaves.
    pub upper: u64,
    /// The lower half's lookup context and its salt after the write.
    pub lower: u64,
    pub salt: u64,
    /// The translating word to publish.
    pub translating: u64,
    /// The host's set of the context and its generation.
    pub set: u64,
    pub set_generation: u64,
}

/// A PE's switch entries. Tables are equal where they hold the same
/// entries and words, whatever generation they count.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Table {
    /// Starts at 1.
    pub generation: u64,
    /// Native switches since the engine last took them.
    pub switches: u64,
    /// The host's [`Sets`], 0 where it keeps none.
    pub sets: u64,
    /// The PE's published translating word (32 bits) and its pending-mail
    /// byte, both 0 where it publishes none.
    pub translating: u64,
    pub pending: u64,
    /// The entry the next one replaces (the engine's alone).
    pub next: u64,
    pub entries: [Entry; ENTRIES],
}

impl Default for Table {
    fn default() -> Self {
        Self {
            generation: 1,
            switches: 0,
            sets: 0,
            translating: 0,
            pending: 0,
            next: 0,
            entries: [Entry::default(); ENTRIES],
        }
    }
}

impl PartialEq for Table {
    fn eq(&self, other: &Self) -> bool {
        (self.switches, self.sets, self.translating, self.pending)
            == (other.switches, other.sets, other.translating, other.pending)
            && self.held().eq(other.held())
    }
}

impl Eq for Table {}

impl Table {
    /// Each entry slot's entry if it holds, without its generation.
    fn held(&self) -> impl Iterator<Item = Option<Entry>> + '_ {
        self.entries.iter().map(|entry| {
            (entry.generation == self.generation).then_some(Entry {
                generation: 0,
                ..*entry
            })
        })
    }

    /// No entry holds any more.
    pub fn forget(&mut self) {
        self.generation += 1;
    }

    /// Hold `entry` from now on, in place of one for the same value or the
    /// next in turn.
    pub fn keep(&mut self, mut entry: Entry) {
        entry.generation = self.generation;
        let index = self
            .entries
            .iter()
            .position(|held| held.value == entry.value && held.generation == self.generation)
            .unwrap_or_else(|| {
                let index = self.next as usize % ENTRIES;
                self.next = self.next.wrapping_add(1);
                index
            });
        self.entries[index] = entry;
    }

    /// The native switches since the last call, which it clears.
    pub fn take_switches(&mut self) -> u64 {
        core::mem::take(&mut self.switches)
    }
}

/// How the thunk reads and loads CR3: the register, or for tests the word
/// at an address that stands for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cr3 {
    Register,
    Word(u64),
}

/// Where a switch thunk finds what it reads and writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Places {
    /// The PE's [`Table`].
    pub table: Place,
    /// `ChainControl::contexts` and `salts`.
    pub contexts: Place,
    pub salts: Place,
    /// The transition table's contexts generation.
    pub contexts_generation: Place,
    /// The `readable` words of cells the switch closes.
    pub closes: Vec<Place>,
    /// Whether the table's sets are loaded through CR3; without, the thunk
    /// switches only where the table names no sets.
    pub cr3: Option<Cr3>,
    /// Issue IBPB and refill the return stack.
    pub isolate: bool,
}

/// IA32_PRED_CMD.
const PRED_CMD: i32 = 0x49;
/// The return stack entries a refill writes, as the host's isolation does.
const RETURN_STACK_ENTRIES: usize = 32;
const XCHG_RCX_EAX: [u8; 2] = [0x87, 0x01];
const MOV_R9_CR3: [u8; 4] = [0x41, 0x0f, 0x20, 0xd9];
const MOV_CR3_RAX: [u8; 3] = [0x0f, 0x22, 0xd8];
const WRMSR: [u8; 2] = [0x0f, 0x30];
const LFENCE: [u8; 3] = [0x0f, 0xae, 0xe8];
/// CALL to the next instruction but one, over an INT3.
const REFILL_CALL: [u8; 6] = [0xe8, 0x01, 0x00, 0x00, 0x00, 0xcc];

/// The thunk completing `MSR TTBR0, Xt` at `pc`, whose register storage the
/// `value` of the cell at `cell` points at, taking `written` of Xt.
pub fn switch_thunk(
    pc: GuestAddress,
    rt: u8,
    cell: Place,
    written: Written,
    places: &Places,
) -> Thunk {
    use Gpr::{R8, R9, Rax, Rcx, Rdx};
    use Size::{Byte, Dword, Qword};
    let mut code = Vec::with_capacity(768);
    let mut asm = Asm::new(&mut code);
    let mut misses: Vec<Patch> = Vec::new();
    let entry = |field: usize| Mem::base(R8, field as i32);
    let table = |field: usize| Mem::base(Rdx, field as i32);
    // The cell, and a write of the value the register holds.
    asm.place(Rax, cell);
    asm.alu_imm8(Alu::Cmp, Qword, Mem::base(Rax, 0), 0);
    misses.push(asm.jcc(Cond::E));
    asm.mov_load(Qword, Rcx, Mem::base_disp8(Rax, 8));
    written.apply(&mut asm, rt);
    asm.alu_load(Alu::Cmp, Qword, Rax, Mem::base(Rcx, 0));
    let unchanged = asm.jcc(Cond::E);
    // The entry for the value, of the table's generation.
    asm.place(Rdx, places.table);
    asm.mov_load(Qword, R9, table(offset_of!(Table, generation)));
    let mut hits = Vec::new();
    for index in 0..ENTRIES {
        let at = offset_of!(Table, entries) + index * size_of::<Entry>();
        asm.alu_load(Alu::Cmp, Qword, Rax, table(at + offset_of!(Entry, value)));
        let other = asm.jcc8(Cond::Ne);
        asm.alu_load(
            Alu::Cmp,
            Qword,
            R9,
            table(at + offset_of!(Entry, generation)),
        );
        let old = asm.jcc8(Cond::Ne);
        asm.lea(Qword, R8, table(at));
        hits.push(asm.jmp());
        asm.bind(other);
        asm.bind(old);
    }
    misses.push(asm.jmp());
    for hit in hits {
        asm.bind(hit);
    }
    // Made in this state: the upper half's context is the entry's.
    asm.place(Rcx, places.contexts);
    asm.mov_load(Qword, Rcx, Mem::base(Rcx, 8));
    asm.alu_load(Alu::Cmp, Qword, Rcx, entry(offset_of!(Entry, upper)));
    misses.push(asm.jcc(Cond::Ne));
    // The host's sets: open, the entry's set of its generation and of the
    // current set's class. R9 keeps the current root's index.
    let sets = |field: usize| Mem::base(Rdx, field as i32);
    if let Some(cr3) = places.cr3 {
        asm.mov_load(Qword, Rdx, table(offset_of!(Table, sets)));
        asm.test(Qword, Rdx, Rdx);
        misses.push(asm.jcc(Cond::E));
        asm.alu_imm8(Alu::Cmp, Qword, sets(offset_of!(Sets, open)), 0);
        misses.push(asm.jcc(Cond::E));
        asm.mov_load(Qword, Rcx, entry(offset_of!(Entry, set)));
        asm.alu_imm32(Alu::Cmp, Qword, Rcx, SETS as i32);
        misses.push(asm.jcc(Cond::Ae));
        asm.mov_load(
            Qword,
            Rax,
            Mem::indexed(Rdx, Rcx, 8, offset_of!(Sets, generations) as i32),
        );
        asm.alu_load(
            Alu::Cmp,
            Qword,
            Rax,
            entry(offset_of!(Entry, set_generation)),
        );
        misses.push(asm.jcc(Cond::Ne));
        read_cr3(&mut asm, cr3);
        asm.mov_imm64(Rax, ROOT_ADDRESS);
        asm.alu(Alu::And, Qword, R9, Rax);
        asm.alu_load(Alu::Sub, Qword, R9, sets(offset_of!(Sets, base)));
        asm.shift_imm(ShiftOp::Shr, Qword, R9, 12);
        asm.alu_imm32(Alu::Cmp, Qword, R9, (SETS * SET_ROOTS) as i32);
        misses.push(asm.jcc(Cond::Ae));
        asm.mov(Qword, Rax, R9);
        asm.shift_imm(ShiftOp::Shr, Qword, Rax, SET_ROOTS.trailing_zeros() as u8);
        asm.mov_load(
            Qword,
            Rax,
            Mem::indexed(Rdx, Rax, 8, offset_of!(Sets, classes) as i32),
        );
        asm.alu_load(
            Alu::Cmp,
            Qword,
            Rax,
            Mem::indexed(Rdx, Rcx, 8, offset_of!(Sets, classes) as i32),
        );
        misses.push(asm.jcc(Cond::Ne));
        asm.place(Rdx, places.table);
    } else {
        asm.alu_imm8(Alu::Cmp, Qword, table(offset_of!(Table, sets)), 0);
        misses.push(asm.jcc(Cond::Ne));
    }
    // Publish the translating word; mail pending after a change goes to
    // the engine first.
    asm.mov_load(Qword, Rcx, table(offset_of!(Table, translating)));
    asm.test(Qword, Rcx, Rcx);
    let unpublished = asm.jcc(Cond::E);
    asm.mov_load(Dword, Rax, entry(offset_of!(Entry, translating)));
    asm.alu_load(Alu::Cmp, Dword, Rax, Mem::base(Rcx, 0));
    let published = asm.jcc(Cond::E);
    asm.fixed(&XCHG_RCX_EAX);
    asm.mov_load(Qword, Rcx, table(offset_of!(Table, pending)));
    asm.movzx(Dword, Rax, Byte, Mem::base(Rcx, 0));
    asm.test(Dword, Rax, Rax);
    misses.push(asm.jcc(Cond::Ne));
    asm.bind(unpublished);
    asm.bind(published);
    // The register, the lower half's context and salt.
    asm.place(Rcx, cell);
    asm.mov_load(Qword, Rcx, Mem::base_disp8(Rcx, 8));
    asm.mov_load(Qword, Rax, entry(offset_of!(Entry, value)));
    asm.mov_store(Qword, Mem::base(Rcx, 0), Rax);
    asm.place(Rcx, places.contexts);
    asm.mov_load(Qword, Rax, entry(offset_of!(Entry, lower)));
    asm.mov_store(Qword, Mem::base(Rcx, 0), Rax);
    asm.place(Rcx, places.salts);
    asm.mov_load(Dword, Rax, entry(offset_of!(Entry, salt)));
    asm.mov_store(Dword, Mem::base(Rcx, 0), Rax);
    for &close in &places.closes {
        asm.place(Rcx, close);
        asm.mov_rm_imm32(Qword, Mem::base(Rcx, 0), 0);
    }
    asm.place(Rcx, places.contexts_generation);
    asm.alu_imm8(Alu::Add, Qword, Mem::base(Rcx, 0), 1);
    asm.alu_imm8(Alu::Add, Qword, table(offset_of!(Table, switches)), 1);
    if let Some(cr3) = places.cr3 {
        // The target root's index: the entry's set, the current view.
        asm.mov_load(Qword, Rdx, table(offset_of!(Table, sets)));
        asm.mov_load(Qword, Rcx, entry(offset_of!(Entry, set)));
        asm.lea(Qword, Rax, Mem::base(Rcx, 1));
        asm.mov_store(Qword, sets(offset_of!(Sets, switched)), Rax);
        asm.shift_imm(ShiftOp::Shl, Qword, Rcx, SET_ROOTS.trailing_zeros() as u8);
        asm.alu_imm8(Alu::And, Qword, R9, (SET_ROOTS - 1) as i8);
        asm.alu(Alu::Or, Qword, Rcx, R9);
        // The set's first root, then the view's.
        asm.mov(Qword, Rax, Rcx);
        asm.alu_imm8(Alu::And, Qword, Rax, !((SET_ROOTS - 1) as i8));
        asm.shift_imm(ShiftOp::Shl, Qword, Rax, 12);
        asm.alu_load(Alu::Add, Qword, Rax, sets(offset_of!(Sets, base)));
        asm.mov_store(Qword, sets(offset_of!(Sets, current)), Rax);
        asm.mov(Qword, Rax, Rcx);
        asm.shift_imm(ShiftOp::Shl, Qword, Rax, 12);
        asm.alu_load(Alu::Add, Qword, Rax, sets(offset_of!(Sets, base)));
        asm.alu_imm8(Alu::Cmp, Qword, sets(offset_of!(Sets, tagged)), 0);
        let untagged = asm.jcc(Cond::E);
        asm.lea(Qword, R9, Mem::base(Rcx, FIRST_CONTEXT as i32));
        asm.alu(Alu::Or, Qword, Rax, R9);
        let stale = Mem::indexed(Rdx, Rcx, 1, offset_of!(Sets, stale) as i32);
        asm.movzx(Dword, R9, Byte, stale);
        asm.test(Dword, R9, R9);
        let flush = asm.jcc8(Cond::Ne);
        asm.mov_imm64(R9, KEEP_ENTRIES);
        asm.alu(Alu::Or, Qword, Rax, R9);
        let kept = asm.jmp8();
        asm.bind(flush);
        asm.alu_imm8(Alu::Add, Qword, sets(offset_of!(Sets, flushes)), 1);
        asm.alu(Alu::Xor, Dword, R9, R9);
        asm.mov_store(Byte, stale, R9);
        asm.bind(kept);
        asm.bind(untagged);
        load_cr3(&mut asm, cr3);
    }
    if places.isolate {
        asm.mov_rm_imm32(Dword, Rcx, PRED_CMD);
        asm.mov_rm_imm32(Dword, Rax, 1);
        asm.alu(Alu::Xor, Dword, Rdx, Rdx);
        asm.fixed(&WRMSR);
        for _ in 0..RETURN_STACK_ENTRIES {
            asm.fixed(&REFILL_CALL);
        }
        asm.alu_imm32(Alu::Add, Qword, Gpr::Rsp, (RETURN_STACK_ENTRIES * 8) as i32);
        asm.fixed(&LFENCE);
    }
    asm.bind(unchanged);
    let next = code.len();
    let mut asm = Asm::new(&mut code);
    super::block::direct_exit_with(
        &mut asm,
        GuestAddress(pc.0.wrapping_add(4)),
        ExitKind::Fallthrough,
    );
    for miss in misses {
        asm.bind(miss);
    }
    super::block::direct_exit_with(&mut asm, pc, ExitKind::System);
    Thunk::new(code, Some(next), None)
}

/// R9 = CR3.
fn read_cr3(asm: &mut Asm<'_>, cr3: Cr3) {
    match cr3 {
        Cr3::Register => asm.fixed(&MOV_R9_CR3),
        Cr3::Word(address) => {
            asm.mov_imm64(Gpr::R9, address);
            asm.mov_load(Size::Qword, Gpr::R9, Mem::base(Gpr::R9, 0));
        }
    }
}

/// CR3 = RAX. Clobbers RCX for a word.
fn load_cr3(asm: &mut Asm<'_>, cr3: Cr3) {
    match cr3 {
        Cr3::Register => asm.fixed(&MOV_CR3_RAX),
        Cr3::Word(address) => {
            asm.mov_imm64(Gpr::Rcx, address);
            asm.mov_store(Size::Qword, Mem::base(Gpr::Rcx, 0), Gpr::Rax);
        }
    }
}

#[cfg(all(
    test,
    target_arch = "x86_64",
    any(target_os = "macos", target_os = "linux")
))]
mod tests;
