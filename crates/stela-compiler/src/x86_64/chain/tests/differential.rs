//! Every native transfer path against the reference's rule, and today's
//! probes against the ones before them (a probe that kept what it needed
//! on the stack and returned to a tail that called it): for random ITLB,
//! block table, cell and control states, with entries put in, taken out,
//! rekeyed and emptied between runs, over several seeds, each path of each
//! BTYPE takes the transfer into the block the reference enters or the
//! exit it takes, and leaves the control as the reference would.
use super::super::*;
use super::Controlled;
use alloc::{format, vec::Vec};
use core::ffi::c_void;
unsafe extern "C" {
    fn mmap(
        address: *mut c_void,
        bytes: usize,
        protection: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(address: *mut c_void, bytes: usize) -> i32;
}

/// The probes before: a call from the tail, PUSH and POP for what they kept,
/// ZF and a return to the tail, which jumped or returned.
fn legacy_found(asm: &mut crate::x86_64::asm::Asm) {
    asm.alu(Alu::Cmp, Qword, R8, R8);
    asm.ret();
}

fn legacy_missed(asm: &mut crate::x86_64::asm::Asm) {
    asm.test(Qword, Gpr::Rsp, Gpr::Rsp);
    asm.ret();
}

/// The tail before: `mov edx, word; call probe; jne exit; jmp r9; exit: ret`,
/// after `movabs r9, cell` for a cell tail.
fn legacy_tail(at: u64, cell: Option<u64>, word: u32, probe: u64) -> Vec<u8> {
    let mut code = Code::new();
    let mut asm = code.asm();
    if let Some(cell) = cell {
        asm.mov_imm64(R9, cell);
    }
    asm.mov_imm32(Rdx, word);
    let call = asm.call();
    let exit = asm.jcc8(Cond::Ne);
    asm.jmp_rm(R9);
    asm.bind(exit);
    asm.ret();
    let distance = relative(at + call.end() as u64, probe).unwrap();
    code.patch(call.offset(), distance.to_le_bytes());
    code.into_bytes()
}

fn legacy_dispatcher(btype: u8, policy: Interruption) -> Option<Vec<u8>> {
    if btype > 3 {
        return None;
    }
    let mut code = Code::new();
    // The cell probe, R9 at the cell.
    let mut asm = code.asm();
    asm.place(Rcx, CONTROL);
    let misses = vec![stop_check(&mut asm, policy, true)];
    asm.push(R9);
    let mut untranslated = Vec::new();
    translate(&mut asm, &mut untranslated);
    asm.pop(R9);
    // The cell is the target's with its page's key, or the table decides.
    // ECX = the cell's last word before its fields ([`Slot`]).
    asm.mov_load(Dword, Rcx, Mem::base(R9, ENTRY_TARGETS));
    asm.alu_load(Alu::Cmp, Qword, Rax, Mem::base(R9, 0));
    let stale = asm.jcc8(Cond::Ne);
    asm.alu_load(Alu::Cmp, Qword, R8, Mem::base(R9, ENTRY_KEY));
    let other = asm.jcc8(Cond::Ne);
    asm.push_rm(Mem::base(R9, ENTRY_BLOCK));
    asm.push_rm(Mem::base(R9, ENTRY_CODE));
    let torn = legacy_seqlock_check(&mut asm);
    asm.pop(R9);
    asm.pop(R8);
    asm.mov(Dword, R8, R8);
    asm.place(Rcx, CONTROL);
    spend_transfer(&mut asm);
    asm.mov_store(Qword, Mem::base(Rcx, BLOCK), R8);
    legacy_found(&mut asm);
    for miss in misses {
        asm.bind(miss);
    }
    legacy_missed(&mut asm);
    for miss in untranslated {
        asm.bind(miss);
    }
    asm.pop(R9);
    legacy_missed(&mut asm);
    // A cell written while it was read leaves the target to the table too,
    // with the key still in R8 and the control in RCX again.
    for miss in torn {
        asm.bind(miss);
    }
    asm.alu_imm8(Alu::Add, Qword, Gpr::Rsp, 16);
    asm.bind(stale);
    asm.bind(other);
    asm.place(Rcx, CONTROL);
    let to_table = asm.jmp();
    let padding = CELL_PROBE_BYTES
        .checked_sub(code.len())
        .expect("the cell probe fits its bytes");
    for _ in 0..padding {
        code.asm().int3();
    }
    debug_assert_eq!(code.len(), PROBE_ENTRY);
    // The table probe.
    let mut asm = code.asm();
    asm.place(Rcx, CONTROL);
    let mut misses = vec![stop_check(&mut asm, policy, true)];
    translate(&mut asm, &mut misses);
    let table = code.len();
    let mut asm = code.asm();
    // R9 = the target's entry of the key in R8.
    asm.mov(Dword, R9, Rax);
    asm.shift_imm(ShiftOp::Shr, Dword, R9, LOOKUP_BITS as u8);
    asm.alu(Alu::Xor, Dword, R9, Rax);
    asm.shift_imm(ShiftOp::Shr, Dword, R9, 2);
    asm.alu(Alu::Xor, Dword, R9, R8);
    asm.alu_load(Alu::And, Dword, R9, Mem::base(Rcx, MASKS));
    asm.shift_imm(ShiftOp::Shl, Dword, R9, 5);
    asm.alu_load(Alu::Add, Qword, R9, Mem::base(Rcx, TABLE));
    // ECX = each entry's last word before its fields ([`Slot`]).
    asm.mov_load(Dword, Rcx, Mem::base(R9, ENTRY_TARGETS));
    asm.alu_load(Alu::Cmp, Qword, Rax, Mem::base(R9, 0));
    let other_pc = asm.jcc8(Cond::Ne);
    asm.alu_load(Alu::Cmp, Qword, R8, Mem::base(R9, ENTRY_KEY));
    let hit = asm.jcc8(Cond::E);
    asm.bind(other_pc);
    asm.alu_imm8(
        Alu::Xor,
        Qword,
        R9,
        core::mem::size_of::<LookupEntry>() as i8,
    );
    asm.mov_load(Dword, Rcx, Mem::base(R9, ENTRY_TARGETS));
    asm.alu_load(Alu::Cmp, Qword, Rax, Mem::base(R9, 0));
    misses.push(asm.jcc(Cond::Ne));
    asm.alu_load(Alu::Cmp, Qword, R8, Mem::base(R9, ENTRY_KEY));
    misses.push(asm.jcc(Cond::Ne));
    asm.bind(hit);
    if btype != 0 {
        asm.test_imm(Size::Byte, Rcx, 1 << btype);
        misses.push(asm.jcc(Cond::E));
    }
    asm.mov_load(Dword, R8, Mem::base(R9, ENTRY_BLOCK));
    asm.push_rm(Mem::base(R9, ENTRY_CODE));
    let torn = legacy_seqlock_check(&mut asm);
    asm.pop(R9);
    asm.place(Rcx, CONTROL);
    spend_transfer(&mut asm);
    asm.alu_imm8(Alu::Add, Qword, Mem::base(Rcx, HITS), 1);
    for bit in 0..2 {
        if btype & 1 << bit != 0 {
            asm.bit_imm(BitTest::Bts, Qword, R8, BLOCK_BTYPE_SHIFT as u8 + bit);
        }
    }
    asm.mov_store(Qword, Mem::base(Rcx, BLOCK), R8);
    legacy_found(&mut asm);
    for miss in torn {
        asm.bind(miss);
    }
    asm.pop(R9);
    for miss in misses {
        asm.bind(miss);
    }
    legacy_missed(&mut asm);
    crate::x86_64::asm::resolve(&mut code, to_table, table);
    Some(code.into_bytes())
}

// The check that ends a probe's read of the entry at R9, whose last word
// it read into ECX before the fields: the entry's word is unchanged and
// no writer was in it ([`Slot`]). The jumps of a torn read.
fn legacy_seqlock_check(asm: &mut crate::x86_64::asm::Asm) -> [crate::x86_64::asm::Patch; 2] {
    asm.alu_load(Alu::Cmp, Dword, Rcx, Mem::base(R9, ENTRY_TARGETS));
    let changed = asm.jcc8(Cond::Ne);
    asm.test_imm(Dword, Rcx, META_BUSY as i32);
    let busy = asm.jcc8(Cond::Ne);
    [changed, busy]
}

const PAGE: usize = 4096;
/// Per variant: the entry stub and tail at 0, the four dispatchers from
/// [`PROBES`]; the blocks are in a page of their own both enter.
const PROBES: usize = 512;
const DISPATCHER_BYTES: usize = 1536;
const REGION: usize = 2 * PAGE;
const BLOCKS: usize = 8;
const CONTEXTS: [u64; 3] = [
    5 << LOOKUP_SALT_BITS,
    9 << LOOKUP_SALT_BITS,
    13 << LOOKUP_SALT_BITS,
];
const KEYS: [u64; 2] = [block_key(0x8_0040_0000, 1), block_key(0x8_0040_0000, 2)];
/// Instructions of the block every transfer leaves.
const RAN: u32 = 5;

/// The BTYPEs a block's first instruction accepts in its entry's targets,
/// by landing pad: none or a plain BTI in a guarded page; BTI c, or PACIASP
/// or PACIBSP with SCTLR_ELx.BT set; BTI j; BTI jc, PACIASP or PACIBSP with
/// BT clear, BRK, HLT, or any instruction outside a guarded page.
const LANDINGS: [u32; 4] = [0b0001, 0b0111, 0b1011, 0b1111];

/// Each seed draws its own order of table changes and transfers.
const SEEDS: [u64; 6] = [
    0x2545_f491_4f6c_dd1d,
    0x9e37_79b9_7f4a_7c15,
    0xd1b5_4a32_d192_ed03,
    0x8cb9_2ba7_2f3d_8dd7,
    0x5851_f42d_4c95_7f2d,
    0x1405_7b7e_f767_814f,
];
/// Transfers each seed runs.
const TRIALS: u32 = 60_000;

fn marker(block: usize) -> u64 {
    0x7700 + block as u64
}

/// Target PCs: two pages in each half, two PCs in each page.
fn pcs() -> [u64; 8] {
    let lower = 0x1_8000_0000_u64;
    let upper = 0xffff_fe00_1234_0000_u64;
    [
        lower,
        lower + 0x40,
        lower + 0x1000,
        lower + 0x1080,
        upper,
        upper + 0x10,
        upper + 0x3000,
        upper + 0x3ff0,
    ]
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

struct Pages(*mut u8, usize);
impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: our mapping, nothing runs it any more.
        assert_eq!(unsafe { munmap(self.0.cast(), self.1) }, 0);
    }
}
fn pages(bytes: usize) -> Pages {
    // SAFETY: private anonymous memory, unmapped on drop.
    let base = unsafe { mmap(core::ptr::null_mut(), bytes, 7, 0x22, -1, 0) };
    assert_ne!(base as usize, usize::MAX);
    Pages(base.cast(), bytes)
}
fn put(pages: &Pages, offset: usize, bytes: &[u8]) {
    assert!(offset + bytes.len() <= pages.1);
    // SAFETY: inside the mapping; nothing runs it while it is written.
    unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), pages.0.add(offset), bytes.len()) };
}

/// Run the stub at `region` (`mov rax, rsi`, then the tail) with `control`
/// and its account: what it returned.
fn run(region: &Pages, control: &mut Controlled, target: u64) -> (u64, u64) {
    // SAFETY: the region holds the stub, a tail and probes with the block
    // ABI; they touch the frame's control, the tables and the blocks, which
    // outlive the call.
    let result = control.run(|frame| unsafe {
        crate::x86_64::native::enter_counted(region.0 as *const u8, frame, target, 0)
    });
    (result as u64, (result >> 64) as u64)
}

/// The native paths of a transfer that a tail starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Path {
    /// A plain tail's call of the table probe.
    Table,
    /// A lookup tail: its jump cache, else the table probe.
    Lookup,
    /// A cell tail's call of the cell probe.
    Cell,
    /// A cell tail's call of the stamping cell probe, as a stamped tail's
    /// miss calls it.
    Stamping,
    /// A stamped branch tail of the cell's entry where the linker gives
    /// one, for an entry that accepts the BTYPE, else the cell tail.
    Stamped,
}
const PATHS: [Path; 5] = [
    Path::Table,
    Path::Lookup,
    Path::Cell,
    Path::Stamping,
    Path::Stamped,
];

/// What the engine's reference path does with a transfer to a target by a
/// branch that set a BTYPE: Arm's Branch Target check of the entered
/// block's first instruction, through the entry's targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reference {
    /// No transfer left, or an interruption pending.
    Stopped,
    /// No translation of the target's page in the contexts the control
    /// names.
    Untranslated,
    /// No entry of the target with its page's key.
    Absent,
    /// The block refuses the BTYPE: the engine takes the exception.
    Refused(LookupEntry),
    Entered(LookupEntry),
}

/// The block key the ITLB holds for the page of `target` in its half's
/// context, or for a lower-half page in the upper half's.
fn translated(itlb: &LookupTable, contexts: [u64; 2], target: u64) -> Option<u64> {
    let page = target & !0xfff;
    let candidates = match half(target) {
        0 => &contexts[..],
        _ => &contexts[1..],
    };
    candidates
        .iter()
        .filter(|&&context| context != 0)
        .find_map(|&context| {
            let resident = itlb.find(itlb.page_slot(target, context), page, context);
            (resident.pc == page && resident.key == context).then_some(resident.entry)
        })
}

/// The reference's answer, with the entry a cell holds first: a cell keeps
/// its entry while the block lives, after the table put it out.
fn reference(
    itlb: &LookupTable,
    table: &LookupTable,
    cell: Option<LookupEntry>,
    control: &Controlled,
    target: u64,
    btype: u8,
) -> Reference {
    if control.quiet.0 & control.account.budget() == 0 {
        return Reference::Stopped;
    }
    let Some(key) = translated(itlb, control.contexts, target) else {
        return Reference::Untranslated;
    };
    let entry = cell
        .filter(|cell| cell.pc == target && cell.key == key)
        .unwrap_or_else(|| table.find(table.slot(target, key), target, key));
    if entry.pc != target || entry.key != key {
        Reference::Absent
    } else if entry.targets & 1 << btype == 0 {
        Reference::Refused(entry)
    } else {
        Reference::Entered(entry)
    }
}

/// Every native transfer path, for each BTYPE and landing pad, against the
/// reference's rule (an entry the target's page translates to, which
/// accepts the BTYPE, else an exit), and the table and cell probes also
/// against the probes before them, over several seeds: random ITLB, block
/// table, cell and control states, with entries put in, put out, rekeyed
/// and emptied between runs. A jump cache or stamp may still enter the
/// block of a translation the ITLB put out for capacity (no drop), as a
/// TLB keeps an entry; never a block the table no longer names for the
/// target, nor one that refuses the BTYPE, and never exit where the
/// reference enters.
#[test]
fn transfers_enter_where_the_reference_enters() {
    let mut seen = alloc::collections::BTreeSet::new();
    for seed in SEEDS {
        differential(seed, &mut seen);
    }
    for (path, path_index) in PATHS.iter().zip(0..) {
        for btype in 0..4u8 {
            // Entered where the landing pad accepts the BTYPE, refused
            // where it does not.
            for (landing, targets) in LANDINGS.iter().enumerate() {
                let entered = targets & 1 << btype != 0;
                assert!(
                    seen.contains(&(path_index, btype, landing, entered)),
                    "{path:?} btype {btype} landing {targets:#06b} entered {entered}"
                );
            }
        }
    }
}

fn differential(seed: u64, seen: &mut alloc::collections::BTreeSet<(usize, u8, usize, bool)>) {
    let pcs = pcs();
    let blocks = pages(PAGE);
    let new = pages(REGION);
    let old = pages(REGION);
    for block in 0..BLOCKS {
        let mut code = vec![0x48, 0xb8];
        code.extend(marker(block).to_le_bytes());
        code.extend([0xba, 99, 0, 0, 0, 0xc3]);
        put(&blocks, 64 * block, &code);
    }
    for btype in 0..4u8 {
        let offset = PROBES + DISPATCHER_BYTES * btype as usize;
        let fresh = dispatcher_with_policy(btype, Interruption::Quiet).unwrap();
        let before = legacy_dispatcher(btype, Interruption::Quiet).unwrap();
        assert!(fresh.len() <= DISPATCHER_BYTES && before.len() <= DISPATCHER_BYTES);
        put(&new, offset, &fresh);
        put(&old, offset, &before);
    }
    let mut table = LookupTable::with_entries(MIN_LOOKUP_ENTRIES);
    let stamps = table.new_stamps();
    let cells = table.stamped_cells();
    let mut itlb = LookupTable::itlb();
    let mut rng = Rng(seed);
    let (mut entered, mut exited, mut cached) = (0u32, 0u32, 0u32);
    // What the page tables map each page to in each context: a key changes
    // only after a drop of the page, as a TLBI precedes a remap.
    let mut keys = alloc::collections::BTreeMap::new();
    for &pc in &pcs {
        for context in CONTEXTS {
            keys.insert((pc & !0xfff, context), KEYS[0]);
        }
    }
    for (block, &pc) in pcs.iter().enumerate() {
        itlb.insert(
            itlb.page_slot(pc, CONTEXTS[half(pc)]),
            LookupEntry::page(pc & !0xfff, CONTEXTS[half(pc)], KEYS[0]),
        );
        let entry = LookupEntry {
            pc,
            key: KEYS[0],
            entry: blocks.0 as u64 + 64 * block as u64,
            block: block as u32,
            targets: LANDINGS[block % LANDINGS.len()],
        };
        table.insert(table.slot(pc, KEYS[0]), entry);
    }
    // Hot exits' stamped cells, each linked to an entry the table holds.
    let mut sites: Vec<(u64, u64)> = (0..4)
        .map(|site| (0x1000 + site, pcs[site as usize]))
        .collect();
    for &(site, pc) in &sites {
        let held = table.find(table.slot(pc, KEYS[0]), pc, KEYS[0]);
        table.link_hot_cell(site, pc, Some(held)).unwrap();
    }
    for trial in 0..TRIALS {
        // Change the tables as the engine does between transfers.
        match rng.below(10) {
            0..=2 => {
                let pc = pcs[rng.below(8) as usize];
                // Mostly the context the control names for the PC's half.
                let context = match rng.below(4) {
                    0 => CONTEXTS[rng.below(3) as usize],
                    _ => CONTEXTS[half(pc)],
                };
                let page = pc & !0xfff;
                let held = keys.get_mut(&(page, context)).unwrap();
                if rng.below(4) == 0 {
                    itlb.forget_range_where(page..=page, |entry| entry.key == context);
                    *held = KEYS[usize::from(rng.below(4) == 0)];
                }
                itlb.insert(
                    itlb.page_slot(pc, context),
                    LookupEntry::page(page, context, *held),
                );
            }
            3..=5 => {
                // A block of the PC put in, or the PC's block again with
                // the targets of another landing pad.
                let pc = pcs[rng.below(8) as usize];
                let block = rng.below(BLOCKS as u64) as usize;
                let entry = LookupEntry {
                    pc,
                    key: KEYS[usize::from(rng.below(4) == 0)],
                    entry: blocks.0 as u64 + 64 * block as u64,
                    block: block as u32,
                    targets: LANDINGS[rng.below(LANDINGS.len() as u64) as usize],
                };
                table.insert(table.slot(entry.pc, entry.key), entry);
            }
            6 => {
                // The engine forgets a frame's entries by block, in the
                // table and in the cells alike.
                let frame = key_frame(KEYS[0]);
                let parity = rng.below(2) as u32;
                table.forget_frames(frame..=frame, |entry| entry.block % 2 == parity);
            }
            7 => {
                let pc = pcs[rng.below(8) as usize] & !0xfff;
                itlb.forget_range_where(pc..=pc, |_| true);
            }
            8 => {
                // Relink a site's cell, to the entry the table holds now or none.
                let index = rng.below(sites.len() as u64) as usize;
                let pc = pcs[rng.below(8) as usize];
                let key = KEYS[rng.below(2) as usize];
                let held = table.find(table.slot(pc, key), pc, key);
                let current =
                    (held.pc == pc && held.key == key && rng.below(4) != 0).then_some(held);
                table.link_hot_cell(sites[index].0, pc, current).unwrap();
                sites[index].1 = pc;
            }
            _ => {
                if rng.below(50) == 0 {
                    itlb.clear();
                }
            }
        }
        // A transfer: every path with every BTYPE in turn.
        let path = PATHS[(trial / 4) as usize % PATHS.len()];
        let path_index = PATHS.iter().position(|&each| each == path).unwrap();
        let btype = (trial % 4) as u8;
        let through_cell = matches!(path, Path::Cell | Path::Stamping | Path::Stamped);
        let (site, cell_pc) = sites[rng.below(sites.len() as u64) as usize];
        let target = match rng.below(8) {
            _ if path == Path::Stamped => cell_pc,
            0 => pcs[rng.below(8) as usize] + 4,
            1..=3 if through_cell => cell_pc,
            _ => pcs[rng.below(8) as usize],
        };
        let kind = ExitWord::from(match btype {
            0 => ExitKind::Direct,
            _ => ExitKind::Indirect,
        });
        let word = kind.encoded() | RAN << ExitWord::RAN_SHIFT;
        let number = table.site_cell_number(site).unwrap();
        let cell = LookupTable::cell_slot(&table.cells, number) as *const Slot as u64;
        let held = LookupTable::cell_slot(&table.cells, number).load();
        let dispatcher = new.0 as u64 + (PROBES + DISPATCHER_BYTES * btype as usize) as u64;
        let at = new.0 as u64 + 3;
        let fresh = match path {
            Path::Table => {
                let mut code = Code::new();
                tail(&mut code, at, kind, RAN, dispatcher + PROBE_ENTRY as u64).unwrap();
                code.into_bytes()
            }
            Path::Lookup => lookup_tail(at, kind, RAN, dispatcher + PROBE_ENTRY as u64, btype)
                .unwrap()
                .to_vec(),
            Path::Cell => {
                rax_cell_tail(at, cell, kind, RAN, dispatcher + CELL_PROBE_ENTRY as u64).unwrap()
            }
            Path::Stamping => rax_cell_tail(
                at,
                cell,
                kind,
                RAN,
                dispatcher + STAMPING_PROBE_ENTRY as u64,
            )
            .unwrap(),
            // As the engine relinks a hot exit: a stamped tail of the entry
            // its cell holds, live again, where the entry accepts the BTYPE.
            Path::Stamped
                if held.key != 0 && held.pc == cell_pc && held.targets & 1 << btype != 0 =>
            {
                table.note_stamped(number);
                assert!(table.revive_stamped(number, held));
                stamped_branch_tail(
                    at,
                    cell,
                    number,
                    cell_pc,
                    kind,
                    RAN,
                    held.block,
                    held.entry,
                    dispatcher + STAMPING_PROBE_ENTRY as u64,
                    btype,
                )
                .unwrap()
                .to_vec()
            }
            Path::Stamped => cell_tail(
                at,
                cell,
                cell_pc,
                kind,
                RAN,
                dispatcher + CELL_PROBE_ENTRY as u64,
            )
            .unwrap()
            .to_vec(),
        };
        let stub = [0x48, 0x89, 0xf0]; // mov rax,rsi
        put(&new, 0, &stub);
        put(&new, 3, &fresh);
        let budget = rng.below(4) as usize;
        let base = ChainControl {
            block: 7,
            contexts: match rng.below(4) {
                0 => [
                    CONTEXTS[rng.below(3) as usize],
                    CONTEXTS[rng.below(3) as usize],
                ],
                _ => [CONTEXTS[0], CONTEXTS[1]],
            },
            table: table.address(),
            hits: 3,
            salts: [0; 2],
            itlb: itlb.address(),
            masks: [table.mask(), itlb.mask()],
            stamps,
            cells,
            ..ChainControl::default()
        };
        let mut base = Controlled {
            control: base,
            account: Account::new([0, 1, 2, 5][budget]).with_retired(100),
        };
        base.salts = base.contexts.map(|context| itlb.salt(context));
        if rng.below(10) == 0 {
            base.interrupt();
        }
        let expected = reference(
            &itlb,
            &table,
            through_cell.then_some(held),
            &base,
            target,
            btype,
        );
        let mut a = base;
        let ra = run(&new, &mut a, target);
        // The probes before, with the same tables, for the table probe of
        // every BTYPE and the cell probe of BTYPE 0, which they had.
        if path == Path::Table || path == Path::Cell && btype == 0 {
            let before = legacy_tail(
                old.0 as u64 + 3,
                through_cell.then_some(cell),
                word,
                old.0 as u64
                    + (PROBES + DISPATCHER_BYTES * btype as usize) as u64
                    + if through_cell {
                        CELL_PROBE_ENTRY
                    } else {
                        PROBE_ENTRY
                    } as u64,
            );
            put(&old, 0, &stub);
            put(&old, 3, &before);
            let mut b = base;
            let rb = run(&old, &mut b, target);
            let mut fresh = a;
            // Only today's cell probes count and sample their transfers.
            (fresh.heat, fresh.sampled) = (0, 0);
            assert_eq!(ra, rb, "trial {trial}: {path:?} btype {btype} {target:#x}");
            assert_eq!(fresh, b, "trial {trial}: the controls differ");
        }
        let context = || {
            format!(
                "seed {seed:#x} trial {trial}: {path:?} btype {btype} target {target:#x} \
                 {expected:?}"
            )
        };
        let named = |block: u32| u64::from(block) | u64::from(btype) << BLOCK_BTYPE_SHIFT;
        let spent = base
            .account
            .with_budget(base.account.budget().saturating_sub(1))
            .with_retired(base.account.retired() + u64::from(RAN));
        // The table probe counts its transfers apart; a cell of the target
        // with its page's key spares it.
        let cell_holds = translated(&itlb, base.contexts, target)
            .is_some_and(|key| held.pc == target && held.key == key);
        let hits = base.hits + u64::from(!(through_cell && cell_holds));
        match expected {
            Reference::Entered(entry) if ra.1 == 99 && ra.0 == marker(entry.block as usize) => {
                assert_eq!(a.block, named(entry.block), "{}", context());
                assert_eq!(a.account, spent, "{}", context());
                assert_eq!(a.hits, hits, "{}", context());
                entered += 1;
            }
            _ if ra.1 != 99 => {
                // An exit where the reference exits, untouched.
                assert!(
                    !matches!(expected, Reference::Entered(_)),
                    "{}: exited where the reference enters",
                    context()
                );
                assert_eq!(ra, (target, u64::from(word)), "{}", context());
                assert_eq!(
                    (a.block, a.account, a.hits),
                    (base.block, base.account, base.hits),
                    "{}",
                    context()
                );
                exited += 1;
            }
            _ => {
                // Entered otherwise: only a jump cache's or stamp's entry
                // of a translation the ITLB put out without a drop, whose
                // block the table or cell still names for the target with a
                // key the page tables map, and which accepts the BTYPE.
                let block = (ra.0 - marker(0)) as u32;
                let translation = translated(&itlb, base.contexts, target);
                let page = target & !0xfff;
                let named_here = |key: u64| {
                    let entry = match through_cell {
                        true => held,
                        false => table.find(table.slot(target, key), target, key),
                    };
                    entry.pc == target
                        && entry.key == key
                        && entry.block == block
                        && entry.targets & 1 << btype != 0
                        && translation != Some(key)
                        && CONTEXTS
                            .into_iter()
                            .any(|context| keys.get(&(page, context)) == Some(&key))
                };
                assert!(
                    matches!(path, Path::Lookup | Path::Stamped)
                        && !matches!(expected, Reference::Stopped)
                        && KEYS.into_iter().any(named_here),
                    "{}: entered block {block}",
                    context()
                );
                assert_eq!(a.block, named(block), "{}", context());
                assert_eq!(a.account, spent, "{}", context());
                cached += 1;
            }
        }
        if let Reference::Entered(entry) | Reference::Refused(entry) = expected {
            let landing = LANDINGS
                .iter()
                .position(|&each| each == entry.targets)
                .unwrap();
            seen.insert((path_index, btype, landing, ra.1 == 99));
        }
    }
    // SAFETY: the table's stamps, which live.
    let counts = unsafe { stamp_counts(stamps) };
    assert!(
        counts[StampCount::Hits as usize] > 0,
        "seed {seed:#x}: no stamped hit"
    );
    assert!(
        counts[StampCount::Filled as usize] > 0,
        "seed {seed:#x}: no jump cache fill"
    );
    // Most transfers exit: no budget, a BTYPE refused, a page or entry
    // the tables lack.
    let trials = TRIALS / 20;
    assert!(
        entered > trials && exited > trials,
        "seed {seed:#x}: {entered} {exited}"
    );
    assert!(cached < trials / 4, "seed {seed:#x}: {cached}");
}

/// A lower-half page the ITLB lacks in the lower half's context is
/// translated through the upper half's, where the engine keeps the pages
/// whose walk ended in a global leaf; the lower half's own entry comes
/// first; an upper-half page is never translated through the lower half's
/// context, nor any page through a context the control does not name.
#[test]
fn a_lower_page_absent_from_its_context_is_translated_through_the_upper_one() {
    let blocks = pages(PAGE);
    let region = pages(REGION);
    let mut code = vec![0x48, 0xb8];
    code.extend(marker(0).to_le_bytes());
    code.extend([0xba, 99, 0, 0, 0, 0xc3]);
    put(&blocks, 0, &code);
    let dispatcher = dispatcher_with_policy(0, Interruption::Quiet).unwrap();
    assert!(dispatcher.len() <= DISPATCHER_BYTES);
    put(&region, PROBES, &dispatcher);
    let tail = lookup_tail(
        region.0 as u64 + 3,
        ExitWord::from(ExitKind::Direct),
        1,
        region.0 as u64 + (PROBES + PROBE_ENTRY) as u64,
        0,
    )
    .unwrap();
    put(&region, 0, &[0x48, 0x89, 0xf0]); // mov rax,rsi
    put(&region, 3, &tail);
    let [lower, upper] = [pcs()[0], pcs()[4]];
    let mut table = LookupTable::with_entries(MIN_LOOKUP_ENTRIES);
    // Each check takes fresh stamps: one page's global and non-global
    // entries together are a TLB conflict, which a jump cache entry filled
    // from either serves until the TLBI that ends it drops the page.
    let fresh: Vec<u64> = (0..8).map(|_| table.new_stamps()).collect();
    let next = std::cell::Cell::new(0);
    for pc in [lower, upper] {
        let entry = LookupEntry {
            pc,
            key: KEYS[0],
            entry: blocks.0 as u64,
            block: 0,
            targets: 0b1111,
        };
        table.insert(table.slot(pc, KEYS[0]), entry);
    }
    let control = |itlb: &LookupTable| {
        let stamps = fresh[next.get()];
        next.set(next.get() + 1);
        let mut control = Controlled {
            control: ChainControl {
                contexts: [CONTEXTS[0], CONTEXTS[1]],
                table: table.address(),
                itlb: itlb.address(),
                masks: [table.mask(), itlb.mask()],
                stamps,
                ..ChainControl::default()
            },
            account: Account::new(5),
        };
        control.salts = control.contexts.map(|context| itlb.salt(context));
        control
    };
    let put_page = |itlb: &mut LookupTable, pc: u64, context: u64, key: u64| {
        let entry = LookupEntry::global_page(pc & !0xfff, context, key);
        itlb.insert(itlb.page_slot(pc, context), entry);
    };
    let entered = |itlb: &LookupTable, pc: u64| run(&region, &mut control(itlb), pc).1 == 99;
    // Only in the upper half's context: entered.
    let mut itlb = LookupTable::itlb();
    put_page(&mut itlb, lower, CONTEXTS[1], KEYS[0]);
    assert!(entered(&itlb, lower));
    // Its own context's entry comes first, here of a key the table lacks.
    put_page(&mut itlb, lower, CONTEXTS[0], KEYS[1]);
    assert!(!entered(&itlb, lower));
    // Of a context the control does not name: not entered.
    let mut itlb = LookupTable::itlb();
    put_page(&mut itlb, lower, CONTEXTS[2], KEYS[0]);
    assert!(!entered(&itlb, lower));
    // An upper-half page in the lower half's context: not entered.
    let mut itlb = LookupTable::itlb();
    put_page(&mut itlb, upper, CONTEXTS[0], KEYS[0]);
    assert!(!entered(&itlb, upper));
    put_page(&mut itlb, upper, CONTEXTS[1], KEYS[0]);
    assert!(entered(&itlb, upper));
    // No upper context (0, as no engine names one): not entered.
    let mut itlb = LookupTable::itlb();
    put_page(&mut itlb, lower, CONTEXTS[1], KEYS[0]);
    let mut none = control(&itlb);
    none.contexts[1] = 0;
    assert_eq!(
        run(&region, &mut none, lower),
        (
            lower,
            u64::from(ExitWord::from(ExitKind::Direct).encoded() | 1 << ExitWord::RAN_SHIFT)
        )
    );
}
