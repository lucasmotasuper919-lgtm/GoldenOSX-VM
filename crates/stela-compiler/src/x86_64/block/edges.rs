//! The internal edges of a region (`block::region`): its branches into
//! itself, which jump there instead of leaving through an exit.
//!
//! Retired instructions. A block's exits and faults count instructions by
//! position: the row of an instruction is the number its block retired
//! before it. A region keeps that count exact with
//! `GuestState::loop_retired` (L), as a self-loop does: at every point,
//! the instructions the entry retired are L plus the row. An edge from row
//! `from` to row `to` retires its branch and moves to `to`, so it adds
//! `from + 1 - to` to L: a forward edge takes the rows it skips off, a back
//! edge adds the rows it runs again. Exits after any landing move L to the
//! chain's retired count first ([`settle`]); the engine adds what is left
//! at faults and at exits it completes.
//!
//! Bound. An entry retires at most `MAX_BLOCK_INSTRUCTIONS`, as a block
//! does, so that the transfer budget and the interrupt poll of the links
//! run as often. Between back edges the region only moves forward, at most
//! [`Region::reach`] rows from a landing. A back edge to `to` jumps there
//! only while `L + to + reach(to)` stays within the bound; otherwise it
//! leaves through a direct exit to its target, whose link polls.
use super::super::asm::{Alu, Asm, Code, Cond, Gpr, Patch, Size};
use super::super::pinned::Dirty;
use crate::{
    GuestAddress,
    aarch64::Instruction,
    block::{MAX_BLOCK_INSTRUCTIONS, Terminator},
};
use alloc::vec;
use alloc::vec::Vec;

/// `GuestState::loop_retired`.
const L: u32 = core::mem::offset_of!(crate::GuestState, loop_retired) as u32;

/// The internal edges of a region and what their emission needs.
pub(in crate::x86_64) struct Region {
    /// Its rows: the body's instructions and the terminal.
    pub rows: usize,
    /// For each body row, the row its branch reaches inside, if it does.
    pub internal: Vec<Option<usize>>,
    /// The row the terminal's branch reaches inside, if it does.
    pub terminal: Option<usize>,
    /// For each row, whether an edge lands there.
    pub landing: Vec<bool>,
    /// For each row, the most rows an entry retires from it before its next
    /// back edge or exit, both included.
    pub reach: Vec<usize>,
    /// The first landing: L is zero at every row before it.
    first_landing: usize,
}

impl Region {
    pub(in crate::x86_64) fn plan(
        instructions: &[Instruction],
        terminal: Option<&(GuestAddress, Terminator)>,
    ) -> Self {
        use crate::block::region::{branch_target, index_of};
        let rows = instructions.len() + usize::from(terminal.is_some());
        let base = instructions
            .first()
            .map(|first| first.pc)
            .or(terminal.map(|(pc, _)| *pc))
            .unwrap_or(GuestAddress(0));
        let inside = |operation| {
            branch_target(operation).and_then(|(target, _)| index_of(base, rows, target))
        };
        let internal: Vec<_> = instructions
            .iter()
            .map(|instruction| inside(&instruction.operation))
            .collect();
        let terminal = match terminal {
            Some((_, Terminator::Branch(operation))) => inside(operation),
            _ => None,
        };
        let mut landing = vec![false; rows];
        for to in internal.iter().flatten().chain(&terminal) {
            landing[*to] = true;
        }
        let mut reach = vec![0; rows];
        for row in (0..rows).rev() {
            let mut further = 0;
            if let Some(instruction) = instructions.get(row) {
                if !crate::block::region::unconditional(instruction) && row + 1 < rows {
                    further = reach[row + 1];
                }
                if let Some(to) = internal[row]
                    && to > row
                {
                    further = further.max(reach[to]);
                }
            }
            reach[row] = 1 + further;
        }
        let first_landing = landing.iter().position(|&at| at).unwrap_or(rows);
        Self {
            rows,
            internal,
            terminal,
            landing,
            reach,
            first_landing,
        }
    }

    /// Whether L may be other than zero at `row`: an edge landed at or
    /// before it.
    pub(in crate::x86_64) fn dirty(&self, row: usize) -> bool {
        row >= self.first_landing
    }

    /// Whether the body's branch at `row` is one the region emits itself:
    /// an internal edge, or a branch that does not fall through.
    pub(in crate::x86_64) fn handles(&self, instructions: &[Instruction], row: usize) -> bool {
        self.internal[row].is_some() || crate::block::region::unconditional(&instructions[row])
    }

    /// The end of the run of locals that may start at `row`: the next
    /// landing after it or branch the region handles from it on, so that no
    /// run spans an edge.
    pub(in crate::x86_64) fn run_end(&self, instructions: &[Instruction], row: usize) -> usize {
        (row..instructions.len())
            .find(|&at| (at > row && self.landing[at]) || self.handles(instructions, at))
            .unwrap_or(instructions.len())
    }

    /// Whether NZCV is dead at `row` on every path of the region from it
    /// ([`super::super::flag_word::dead_from`]): each path reaches an
    /// instruction that defines NZCV, through instructions that leave it
    /// and the frame alone and forward edges, before any reader, exit,
    /// back edge or the terminal.
    pub(in crate::x86_64) fn nzcv_dead(&self, instructions: &[Instruction], row: usize) -> bool {
        use crate::aarch64::Operation;
        let mut seen = vec![false; self.rows];
        let mut work = vec![row];
        while let Some(mut at) = work.pop() {
            loop {
                let Some(instruction) = instructions.get(at) else {
                    return false;
                };
                if core::mem::replace(&mut seen[at], true) {
                    break;
                }
                if let Some((_, conditional)) =
                    crate::block::region::branch_target(&instruction.operation)
                {
                    if matches!(instruction.operation,
                        Operation::BranchCondition { condition, .. } if condition < 14)
                    {
                        return false;
                    }
                    let Some(to) = self.internal[at].filter(|&to| to > at) else {
                        return false;
                    };
                    if conditional {
                        work.push(to);
                        at += 1;
                    } else {
                        at = to;
                    }
                    continue;
                }
                match super::super::flag_word::defines(&instruction.operation) {
                    Some(true) => break,
                    Some(false) => at += 1,
                    None => return false,
                }
            }
        }
        true
    }

    /// The loop that starts at `at` and that a run of locals can cover
    /// alone: the last row of its back edges to `at`, when only the
    /// fallthrough into `at` enters it (or `at` is the entry) and no edge
    /// leaves it for another row of the region, so that the run is entered
    /// once before its landing and left where the loop falls out of its
    /// last branch; its exits are side exits, which store the locals.
    pub(in crate::x86_64) fn inner_loop(&self, at: usize) -> Option<usize> {
        let edges = || {
            self.internal
                .iter()
                .enumerate()
                .filter_map(|(row, to)| Some((row, (*to)?)))
                .chain(self.terminal.map(|to| (self.rows - 1, to)))
        };
        let last = edges()
            .filter(|&(row, to)| to == at && row >= at)
            .map(|(row, _)| row)
            .max()?;
        let inside = |row: usize| (at..=last).contains(&row);
        edges()
            .all(|(row, to)| inside(row) == inside(to) && (!inside(row) || to > row || to == at))
            .then_some(last)
    }

    /// The bound a back edge to `to` compares L with: it jumps while L is
    /// less ([`Self`]'s module).
    fn limit(&self, to: usize) -> i32 {
        (MAX_BLOCK_INSTRUCTIONS + 1 - to - self.reach[to]) as i32
    }
}

/// What a region's edges leave to be bound: forward jumps to landings not
/// yet emitted, and the jumps to the direct exits of back edges at their
/// bound, each with its target and the rows its exit leaves unrun.
#[derive(Default)]
pub(in crate::x86_64) struct Pending {
    pub forward: Vec<(Patch, usize)>,
    pub bounded: Vec<(Patch, GuestAddress)>,
}

/// Emit the edge from row `from` to row `to`, whose instruction is at
/// `target`; `landings` holds where the rows emitted so far start, and
/// `dirty` the locals of a run over the whole region, which an exit at
/// the bound stores.
#[allow(clippy::too_many_arguments)]
pub(in crate::x86_64) fn edge(
    code: &mut Code,
    region: &Region,
    from: usize,
    to: usize,
    target: GuestAddress,
    landings: &[Option<usize>],
    pending: &mut Pending,
    dirty: Dirty,
) {
    let mut asm = code.asm();
    if to > from {
        let skipped = to - from - 1;
        if skipped != 0 {
            asm.alu_imm32(
                Alu::Sub,
                Size::Qword,
                super::super::frame(L),
                skipped as i32,
            );
        }
        let jump = asm.jmp();
        pending.forward.push((jump, to));
        return;
    }
    asm.alu_imm32(
        Alu::Add,
        Size::Qword,
        super::super::frame(L),
        (from + 1 - to) as i32,
    );
    asm.alu_imm32(
        Alu::Cmp,
        Size::Qword,
        super::super::frame(L),
        region.limit(to),
    );
    let again = asm.jcc(Cond::L);
    asm.bind_back(
        again,
        landings[to].expect("a back edge's landing is emitted"),
    );
    // At the bound: the exit to the target counts all rows, so L takes
    // the rows past the target off.
    super::super::pinned::write_back_locals(code, dirty);
    let mut asm = code.asm();
    settle(&mut asm, (region.rows - to) as u32);
    let exit = asm.jmp();
    pending.bounded.push((exit, target));
}

/// Move L, less `less`, to the chain's retired count and clear it, before
/// an exit that may be linked.
pub(in crate::x86_64) fn settle(asm: &mut Asm, less: u32) {
    asm.mov_load(Size::Qword, Gpr::Rax, super::super::frame(L));
    if less != 0 {
        asm.alu_imm32(Alu::Sub, Size::Qword, Gpr::Rax, less as i32);
    }
    super::super::chain::account_add(asm);
    asm.alu_imm8(Alu::And, Size::Qword, super::super::frame(L), 0);
}
