//! Regions: system-mode blocks that run on past the branches a block ends
//! at, so that direct control flow inside them (both sides of B.cond, CBZ
//! and TBZ, B, loops of several blocks) stays in one translation.
//!
//! A region is a block in every contract the engine and the store know:
//! one entry PC, a contiguous source of at most
//! [`MAX_BLOCK_INSTRUCTIONS`] words in one page, one source-map row per
//! word in order, one terminal at its last word. Only its code differs:
//! a branch whose target lies in the region jumps there
//! (`x86_64::block`'s region mode) instead of leaving through an exit.
//!
//! Formation scans a block's words as a block would; where the block
//! would end at a B, a backward branch or its fifth forward branch, the
//! scan goes on to the first terminal that is not a branch, a word it
//! cannot take (which then starts a block of its own) or the end of the
//! source window. It then keeps the words the entry reaches through the
//! region's own branches and fallthroughs, and takes the region only when
//! one of them branches back inside it: a loop. Anything else stays the
//! block it would have been, as does a block that already runs as a
//! self-loop (`x86_64::block::SelfLoop`).
use super::{MAX_BLOCK_INSTRUCTIONS, Terminator, analysis::ParsedBlock};
use crate::{
    GuestAddress,
    aarch64::{Instruction, Operation},
    x86_64::NativeSystem,
};
use alloc::vec;

/// A branch's target, and whether it is conditional.
pub(crate) fn branch_target(operation: &Operation) -> Option<(GuestAddress, bool)> {
    match *operation {
        Operation::Branch { target } => Some((target, false)),
        // AL and NV both always hold.
        Operation::BranchCondition { target, condition } => Some((target, condition < 14)),
        Operation::BranchZero { target, .. } | Operation::BranchBit { target, .. } => {
            Some((target, true))
        }
        _ => None,
    }
}

/// The index of `target` among `count` words from `base`, if it is one.
pub(crate) fn index_of(base: GuestAddress, count: usize, target: GuestAddress) -> Option<usize> {
    let offset = target.0.checked_sub(base.0)?;
    (offset.is_multiple_of(4) && offset / 4 < count as u64).then_some((offset / 4) as usize)
}

/// The block `parsed` ends as without regions: its scan cut at `end`, the
/// branch it would have ended at.
fn block_at(mut parsed: ParsedBlock, end: usize) -> ParsedBlock {
    parsed.instructions.truncate(end + 1);
    let branch = parsed.instructions.pop().expect("the branch at the end");
    parsed.next_pc = GuestAddress(branch.pc.0 + 4);
    parsed.source_bytes = (end + 1) * 4;
    parsed.terminal = Some((branch.pc, Terminator::Branch(branch.operation)));
    parsed.region = false;
    parsed
}

/// The region or block of `parsed`, a region scan of a system-mode block:
/// `end` is where the block would have ended, and `ended` whether the scan
/// stopped before the end of its source (at a terminal, or before a word it
/// could not take). A scan that ran to the end keeps every word, so that a
/// longer source window, which the engine then reads, forms the same region.
pub(super) fn form(
    parsed: ParsedBlock,
    end: usize,
    ended: bool,
    native: NativeSystem,
) -> ParsedBlock {
    let block = |parsed| block_at(parsed, end);
    // A self-loop runs its body twice with checks hoisted: it is better
    // than a region for its shape.
    {
        let words = &parsed.instructions[..end];
        let branch = &parsed.instructions[end];
        let terminal = (branch.pc, Terminator::Branch(branch.operation.clone()));
        if crate::x86_64::block::SelfLoop::admit(words, Some(&terminal), native).is_some() {
            return block(parsed);
        }
    }
    let ParsedBlock {
        instructions,
        terminal,
        ..
    } = &parsed;
    let base = instructions[0].pc;
    let count = instructions.len() + usize::from(terminal.is_some());
    debug_assert!(count <= MAX_BLOCK_INSTRUCTIONS);
    let internal = |target| index_of(base, count, target);
    // The words the entry reaches.
    let mut reached = vec![false; count];
    let mut work = vec![0];
    let mut loops = false;
    while let Some(at) = work.pop() {
        if reached[at] {
            continue;
        }
        reached[at] = true;
        let Some(instruction) = instructions.get(at) else {
            // The terminal leaves.
            continue;
        };
        let mut falls = true;
        if let Some((target, conditional)) = branch_target(&instruction.operation) {
            falls = conditional;
            if let Some(to) = internal(target) {
                loops |= to <= at;
                work.push(to);
            }
        }
        if falls && at + 1 < count {
            work.push(at + 1);
        }
    }
    if !loops {
        return block(parsed);
    }
    let last = reached.iter().rposition(|&reached| reached).unwrap_or(0);
    let mut region = parsed;
    region.region = true;
    if !ended || last + 1 == count {
        // Every word stays. A last word that branches is the terminal.
        if region.terminal.is_none()
            && region
                .instructions
                .last()
                .is_some_and(|last| branch_target(&last.operation).is_some())
        {
            let branch = region.instructions.pop().expect("a last word");
            region.terminal = Some((branch.pc, Terminator::Branch(branch.operation)));
        }
        return region;
    }
    // The words past the last one reached are left out. A word reached
    // only by fallthrough is followed by the next, so the last one reached
    // before them is a branch that does not fall through, or the scan's
    // last word.
    region.instructions.truncate(last + 1);
    let final_word = region.instructions.last().expect("a reached word");
    region.next_pc = GuestAddress(final_word.pc.0 + 4);
    region.source_bytes = (last + 1) * 4;
    region.terminal = None;
    if branch_target(&final_word.operation).is_some() {
        let branch = region.instructions.pop().expect("a last word");
        region.terminal = Some((branch.pc, Terminator::Branch(branch.operation)));
    }
    region
}

/// Whether `instruction` is a branch of a region that does not fall
/// through: a B, or a B.AL or B.NV.
pub(crate) fn unconditional(instruction: &Instruction) -> bool {
    branch_target(&instruction.operation).is_some_and(|(_, conditional)| !conditional)
}

#[cfg(test)]
mod tests {
    use crate::{
        GuestAddress,
        block::{BlockProfile, CompiledBlock, Features},
    };
    use alloc::vec::Vec;

    fn compile(words: &[u32], regions: bool) -> CompiledBlock {
        let source: Vec<u8> = words.iter().copied().flat_map(u32::to_le_bytes).collect();
        let options = crate::Options::new(
            BlockProfile::new(Features::SYSTEM_MODE),
            crate::TargetCpu::detect(),
        )
        .with_memory_checks(crate::x86_64::MemoryChecks::system(true, false, true, true))
        .with_native_system(crate::x86_64::NativeSystem {
            self_loops: true,
            regions,
            zero_block: Some(64),
            ..crate::x86_64::NativeSystem::NONE
        });
        crate::compile(&source, GuestAddress(0x1000), &options).unwrap()
    }

    const RET: u32 = 0xd65f_03c0;
    const ADD: u32 = 0x9100_04e7; // add x7,x7,#1

    /// Loops a block alone cannot hold become regions to their first
    /// terminal that is not a branch, with a direct exit for each back
    /// edge at its bound.
    #[test]
    fn loops_of_several_blocks_run_to_their_terminal() {
        let shapes: [(&[u32], usize, usize, usize); 4] = [
            // 1: ldrb; cmp; b.ne 2f; add; b 3f; 2: add; 3: subs; b.ne 1b; ret
            (
                &[
                    0x3840_1408,
                    0x7100_811f,
                    0x5400_0061,
                    0x9100_054a,
                    0x1400_0002,
                    0x9100_056b,
                    0xf100_0442,
                    0x54ff_ff21,
                    RET,
                ],
                5,
                9,
                1,
            ),
            // b 2f; 1: str; add; 2: subs; b.hs 1b; add; ret
            (
                &[
                    0x1400_0003,
                    0xf800_8403,
                    0x9100_0463,
                    0xf100_0442,
                    0x54ff_ffa2,
                    ADD,
                    RET,
                ],
                1,
                7,
                1,
            ),
            // 1: mov; 2: ldrb; strb; subs; b.ne 2b; add; subs; b.ne 1b; ret
            (
                &[
                    0xaa04_03e5,
                    0x3840_1426,
                    0x3800_1406,
                    0xf100_04a5,
                    0x54ff_ffa1,
                    0x8b03_0000,
                    0xf100_0442,
                    0x54ff_ff21,
                    RET,
                ],
                5,
                9,
                2,
            ),
            // mov; 1: ldr; subs; b.ne 1b; ret
            (
                &[0xaa1f_03e9, 0xf840_8408, 0xf100_0442, 0x54ff_ffc1, RET],
                4,
                5,
                1,
            ),
        ];
        for (words, block, region, backs) in shapes {
            let (alone, formed) = (compile(words, false), compile(words, true));
            assert!(!alone.region && formed.region, "{words:x?}");
            assert_eq!(alone.source_bytes, block * 4, "{words:x?}");
            assert_eq!(formed.source_bytes, region * 4, "{words:x?}");
            assert_eq!(formed.native.source_map.len(), region, "{words:x?}");
            let bounded = formed
                .direct_exits
                .iter()
                .filter(|exit| {
                    exit.kind == crate::block::ExitKind::Direct
                        && exit.skipped == 0
                        && exit.target.0 < 0x1000 + 4 * region as u64
                })
                .count();
            assert_eq!(bounded, backs, "{words:x?}: {:?}", formed.direct_exits);
        }
    }

    /// A self-loop stays one, code without a branch back stays a block,
    /// words past the last one reached are left out, and a word the scan
    /// cannot take ends it before that word.
    #[test]
    fn regions_take_loops_only_and_end_where_they_are_reached() {
        // A self-loop: ldrb; cbnz w8, back; ret.
        let selfloop = compile(&[0x3840_1408, 0x35ff_ffe8, RET], true);
        assert!(selfloop.self_loop && !selfloop.region);
        // No branch back: b +2; add; add; ret stays the block of its b.
        let forward = compile(&[0x1400_0002, ADD, ADD, RET], true);
        assert!(!forward.region && forward.source_bytes == 4);
        // 0: b 2; 1: add; 2: subs; 3: b.ne 1; 4: b 1; 5: add; 6: ret: the
        // words after the b back, reached by nothing, are left out.
        let trimmed = compile(
            &[
                0x1400_0002,
                ADD,
                0xf100_0442,
                0x54ff_ffc1,
                0x17ff_fffd,
                ADD,
                RET,
            ],
            true,
        );
        assert!(trimmed.region);
        assert_eq!(trimmed.source_bytes, 5 * 4);
        // A word the lifter refuses past the block's end: the region ends
        // before it and falls through to it.
        let refused = compile(
            &[0x1400_0002, ADD, 0xf100_0442, 0x54ff_ffc1, 0xffff_ffff],
            true,
        );
        assert!(refused.region);
        assert_eq!(refused.source_bytes, 4 * 4);
    }
}
