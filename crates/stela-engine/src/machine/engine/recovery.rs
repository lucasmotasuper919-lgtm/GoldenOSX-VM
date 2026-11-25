//! Invocation-kept source/profile and installed native recovery owners.
use super::stored::Emission;
use super::{NativeCode, NativeFault, Preparation};
use crate::translations::{Digest, Entry, Relocated, StoredRecord};
use alloc::sync::Arc;
#[cfg(test)]
use alloc::{boxed::Box, vec::Vec};
#[cfg(test)]
use stela_compiler::block::{FaultRecovery, UnqualifiedFaultRecovery};
use stela_compiler::block::{PackedRecovery, RecoveryQualificationError, RecoveryTemplate};
use stela_compiler::x86_64::chain::DirectExit;

/// The process's shared emission or a record's entry, relocated, borrowed
/// through installation. A stored block keeps only its origin once
/// installed.
#[derive(Clone)]
pub(super) enum Template<'a> {
    Compiled(Arc<Emission>),
    Stored {
        entry: &'a Entry,
        relocated: &'a Relocated,
        origin: Arc<StoredOrigin>,
    },
}

/// Where a stored block's record lies: everything but the page offset of its
/// entry, shared by the blocks of a page. The pack outlives every machine,
/// so a block keeps this and not a copy of its entry.
pub(super) struct StoredOrigin {
    pub(super) key: Digest,
    pub(super) context: Digest,
    pub(super) page: u64,
    pub(super) preparation: Arc<Preparation>,
    /// The entry itself, for one proven during the boot
    /// ([`super::rebase`]): no pack holds its record.
    pub(super) kept: Option<Arc<Entry>>,
}

/// An installed block's recovery owner: its code's length and the rows that
/// locate guest state at each faultable site of it.
pub(super) struct CodeRecovery {
    rows: Rows,
    #[cfg(test)]
    qualification_attempts: usize,
}

enum Rows {
    Qualified {
        code_bytes: u32,
        rows: PackedRecovery,
    },
    /// A stored record's entry, whose template and portable rows its first
    /// fault reads and emits again to qualify: replaced by the minted rows.
    /// A record the store no longer holds in memory (a pack read on demand
    /// lets records go) is not read at a fault: the entry's admitted input,
    /// `input_bytes` long and hashing to `input_hash`, is read from guest
    /// memory again and compiled, which gives the entry's template, as the
    /// record's context names every compile input.
    Stored {
        origin: Arc<StoredOrigin>,
        code_bytes: u32,
        offset: u16,
        input_bytes: u16,
        input_hash: u32,
    },
    /// A template and portable rows in hand, as a record's entry gives them.
    #[cfg(test)]
    Pending {
        template: Arc<Emission>,
        rows: Box<[UnqualifiedFaultRecovery]>,
        code_bytes: u32,
    },
}

// The code length shares the per-variant space of the row storage. Every
// installed block uses one slot, including those with no guest fault sites.
#[cfg(not(test))]
const _: () = assert!(core::mem::size_of::<Option<CodeRecovery>>() <= 24);

/// A terminal native implementation failure. Captured data and code owners stay
/// on the stopped Machine; a caller cannot resume a partially committed frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeFailure {
    pub pc: u64,
    pub code: Option<NativeCode>,
    pub native_offset: usize,
    pub address: u64,
    pub origin: super::NativeFaultOrigin,
    pub trap: u16,
    pub error_code: u64,
    pub unsettled_chain: Option<UnsettledChain>,
    pub reason: NativeFailureReason,
    /// Where the last block it ran stood on the board, when the failure
    /// is that its code is unknown or the fault lies outside its code.
    pub board: Option<BoardSlot>,
    /// For a fault outside the last block's code: the slot whose block's
    /// code holds the faulting host address, if any does now.
    pub owner: Option<BoardSlot>,
}

/// The board's state of a block a native failure ran: its slot and
/// whether it is live, its incarnation, whether its install lost to another
/// PE's, the PE that compiled it, whether its recovery rows are held, its
/// part of the arena, the ring's generation and one more than the last
/// generation of that part taken away.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoardSlot {
    pub slot: u64,
    pub held: bool,
    pub live: bool,
    pub incarnation: u32,
    pub lost: bool,
    pub installer: u8,
    pub rows: bool,
    /// Its block's code address and the bytes its rows cover.
    pub code: u64,
    pub code_bytes: u64,
    pub part: Option<usize>,
    pub generation: u32,
    pub retired_through: u32,
}
/// Native control counters copied after return, without granting a resumable
/// architectural boundary or claiming that completed memory effects rolled back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnsettledChain {
    pub reported_retired: u64,
    pub last_block: u64,
    pub lookup_hits: u64,
    pub direct_links: u64,
}
pub(super) struct CapturedFailure {
    pub(super) summary: Arc<NativeFailure>,
    pub(super) capture: NativeFault,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeFailureReason {
    CanonicalDomain {
        pe: usize,
        retired_before: u64,
    },
    UnknownCode,
    CodeExtent,
    InternalSite,
    TrapSiteMismatch,
    Position,
    Qualification(RecoveryQualificationError),
    /// The pack no longer gives a stored block's entry.
    StoredRecord,
    Recovery(stela_compiler::block::FaultRecoveryError),
}

impl<'a> Template<'a> {
    /// The template of a block compiled in this process, here or by a PE
    /// that shared it through the store.
    pub(super) fn compiled(emission: Arc<Emission>) -> Self {
        Self::Compiled(emission)
    }
    /// `entry` of a record, which `relocated` holds as installed at its
    /// origin's page ([`Entry::relocate`]).
    pub(super) fn stored(
        entry: &'a Entry,
        relocated: &'a Relocated,
        origin: Arc<StoredOrigin>,
    ) -> Self {
        Self::Stored {
            entry,
            relocated,
            origin,
        }
    }

    /// The native code.
    pub(super) fn code(&self) -> &[u8] {
        match self {
            Self::Compiled(emission) => &emission.block.native.x86_64_code,
            Self::Stored { relocated, .. } => &relocated.code,
        }
    }
    /// The guest instructions it retires: one source-map entry each.
    pub(super) fn instructions(&self) -> usize {
        match self {
            Self::Compiled(emission) => emission.block.native.source_map.len(),
            Self::Stored { entry, .. } => entry.source_map.len(),
        }
    }
    /// The guest bytes its native template consumed.
    pub(super) fn source_bytes(&self) -> usize {
        match self {
            Self::Compiled(emission) => emission.block.source_bytes,
            Self::Stored { entry, .. } => usize::from(entry.source_bytes),
        }
    }
    /// The code offsets of its cold code (`CompiledBlock::cold`).
    pub(super) fn cold(&self) -> (u32, u32) {
        match self {
            Self::Compiled(emission) => emission.block.cold,
            Self::Stored { entry, .. } => entry.cold,
        }
    }
    /// The native template's direct exits to static successors.
    pub(super) fn direct_exits(&self) -> &[DirectExit] {
        match self {
            Self::Compiled(emission) => &emission.block.direct_exits,
            Self::Stored { relocated, .. } => &relocated.direct_exits,
        }
    }
    /// A portable request may only reject this legacy implementation. Absence
    /// of tags never grants private/APX admission: that must be derived from an
    /// compiler/profile ABI permit before future eager invocation.
    pub(super) fn portable_requires_unavailable_capture(&self) -> bool {
        use stela_compiler::block::{RecoveryBinding, RecoverySource, RetirementPhase};
        fn requires(
            phase: RetirementPhase,
            mut bindings: impl Iterator<Item = RecoveryBinding>,
        ) -> bool {
            phase != RetirementPhase::Before
                || bindings.any(|binding| match binding.source {
                    RecoverySource::HostRegister(register) => register >= 16,
                    RecoverySource::NativeSpill(_) => true,
                })
        }
        // Rebasing a row moves its PC alone, not its phase or bindings.
        match self {
            Self::Stored { entry, .. } => entry
                .recovery
                .iter()
                .any(|row| requires(row.phase, row.bindings())),
            Self::Compiled(emission) => emission
                .block
                .fault_recovery
                .iter()
                .any(|row| requires(row.retirement.phase(), row.bindings.iter().copied())),
        }
    }
    pub(super) fn matches_source(&self, source: &[u8]) -> bool {
        let input = match self {
            Self::Compiled(emission) => emission.input.as_slice(),
            Self::Stored { entry, .. } => entry.input.as_slice(),
        };
        source.get(..input.len()) == Some(input)
    }

    /// The owner of the installed template: its rows, packed, or for a stored
    /// record where they are, until its first fault qualifies them.
    pub(super) fn installed(self) -> CodeRecovery {
        let code_bytes = self.code().len() as u32;
        let rows = match self {
            Self::Stored { origin, entry, .. } => Rows::Stored {
                origin,
                code_bytes,
                offset: entry.offset,
                input_bytes: entry.input.len() as u16,
                input_hash: input_hash(&entry.input),
            },
            Self::Compiled(emission) => Rows::Qualified {
                code_bytes,
                rows: PackedRecovery::new(&emission.block.fault_recovery),
            },
        };
        CodeRecovery {
            rows,
            #[cfg(test)]
            qualification_attempts: 0,
        }
    }
}

/// A word-wise hash of an entry's admitted input: what a stored block's
/// owner keeps to know its bytes in guest memory again ([`Rows::Stored`]).
fn input_hash(bytes: &[u8]) -> u32 {
    let (words, rest) = bytes.as_chunks::<8>();
    let mut hash = (bytes.len() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    for word in words {
        hash = (hash ^ u64::from_le_bytes(*word))
            .wrapping_mul(0x0100_0000_01b3)
            .rotate_left(29);
    }
    for &byte in rest {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
    }
    (hash ^ hash >> 32) as u32
}

impl CodeRecovery {
    /// The installed code's length.
    pub(super) fn code_bytes(&self) -> usize {
        self.rows.code_bytes() as usize
    }

    /// The heap bytes its rows hold.
    pub(super) fn heap_bytes(&self) -> usize {
        match &self.rows {
            Rows::Qualified { rows, .. } => rows.heap_bytes(),
            _ => 0,
        }
    }

    /// Make the owner a stored record's: `template`'s, whose rows are this
    /// owner's own, unqualified.
    #[cfg(test)]
    pub(super) fn test_make_pending(&mut self, template: Arc<Emission>) {
        let rows = self.test_portable_rows().into_boxed_slice();
        self.rows = Rows::Pending {
            template,
            rows,
            code_bytes: self.rows.code_bytes(),
        };
    }
    /// The rows of an actual emitted plan.
    #[cfg(test)]
    pub(super) fn test_rows(&self) -> Vec<FaultRecovery> {
        match &self.rows {
            Rows::Qualified { rows, .. } => rows.rows(),
            _ => panic!("actual emitted plan"),
        }
    }
    /// Change the rows of an actual emitted plan.
    #[cfg(test)]
    pub(super) fn test_edit_rows<T>(&mut self, edit: impl FnOnce(&mut [FaultRecovery]) -> T) -> T {
        let Rows::Qualified { rows: packed, .. } = &mut self.rows else {
            panic!("actual emitted plan");
        };
        let mut rows = packed.rows();
        let result = edit(&mut rows);
        *packed = PackedRecovery::new(&rows);
        result
    }
    #[cfg(test)]
    pub(super) fn test_packed_bytes(&self) -> usize {
        match &self.rows {
            Rows::Qualified { rows, .. } => rows.bytes(),
            _ => panic!("actual emitted plan"),
        }
    }
    #[cfg(test)]
    pub(super) fn test_portable_rows(&self) -> Vec<UnqualifiedFaultRecovery> {
        match &self.rows {
            Rows::Pending { rows, .. } => rows.to_vec(),
            Rows::Stored { .. } => panic!("explicitly pending fixture"),
            Rows::Qualified { rows, .. } => rows
                .rows()
                .iter()
                .map(UnqualifiedFaultRecovery::from)
                .collect(),
        }
    }
    #[cfg(test)]
    pub(super) fn test_pending_rows_mut(&mut self) -> &mut [UnqualifiedFaultRecovery] {
        match &mut self.rows {
            Rows::Pending { rows, .. } => rows,
            _ => panic!("explicitly pending fixture"),
        }
    }
    #[cfg(test)]
    pub(super) fn test_push_pending(&mut self, row: UnqualifiedFaultRecovery) {
        let Rows::Pending { rows, .. } = &mut self.rows else {
            panic!("explicitly pending fixture");
        };
        let mut pushed = rows.to_vec();
        pushed.push(row);
        *rows = pushed.into_boxed_slice();
    }

    /// The bytes of guest memory, from the block's entry, that compile its
    /// rows again when its record is not in memory ([`Self::qualify`]).
    pub(super) fn stored_input_bytes(&self) -> Option<usize> {
        match &self.rows {
            Rows::Stored {
                origin,
                input_bytes,
                ..
            } if origin.kept.is_none() => Some(usize::from(*input_bytes)),
            _ => None,
        }
    }

    /// The key of the record whose entry this owner's rows still are.
    pub(super) fn stored_key(&self) -> Option<Digest> {
        match &self.rows {
            Rows::Stored { origin, .. } if origin.kept.is_none() => Some(origin.key),
            _ => None,
        }
    }

    /// The rows, qualifying a stored record's on its first fault at `pc`, the
    /// block's entry: in the dispatcher, before any guest projection. `record`
    /// is the one [`Self::stored_key`] names; without it, `memory` holds the
    /// guest's bytes at the block's entry ([`Self::stored_input_bytes`]),
    /// whose compile gives the rows when they admit to the entry's input.
    pub(super) fn qualify(
        &mut self,
        pc: u64,
        record: Option<&StoredRecord>,
        memory: Option<&[u8]>,
    ) -> Result<&PackedRecovery, NativeFailureReason> {
        if !matches!(self.rows, Rows::Qualified { .. }) {
            #[cfg(test)]
            {
                self.qualification_attempts += 1;
            }
            let block = match &self.rows {
                Rows::Qualified { .. } => unreachable!("unqualified above"),
                Rows::Stored {
                    origin,
                    offset,
                    code_bytes,
                    input_bytes,
                    input_hash: hash,
                } => {
                    let decoded;
                    let entry = match &origin.kept {
                        Some(kept) => Some(&**kept).filter(|kept| kept.offset == *offset),
                        None => {
                            decoded =
                                record.and_then(|record| record.find(*offset, &origin.context));
                            decoded.as_ref()
                        }
                    };
                    let Some(entry) = entry else {
                        let mut input = memory
                            .filter(|memory| memory.len() == usize::from(*input_bytes))
                            .ok_or(NativeFailureReason::StoredRecord)?
                            .to_vec();
                        origin.preparation.admit(&mut input);
                        if input_hash(&input) != *hash {
                            return Err(NativeFailureReason::StoredRecord);
                        }
                        let block = origin
                            .preparation
                            .recompile(&input, pc)
                            .filter(|block| block.native.x86_64_code.len() == *code_bytes as usize)
                            .ok_or(NativeFailureReason::StoredRecord)?;
                        self.rows = Rows::Qualified {
                            code_bytes: *code_bytes,
                            rows: PackedRecovery::new(&block.fault_recovery),
                        };
                        let Rows::Qualified { rows, .. } = &self.rows else {
                            unreachable!("qualified above");
                        };
                        return Ok(rows);
                    };
                    let stored = entry
                        .rebase(origin.page, &entry.input)
                        .ok_or(NativeFailureReason::StoredRecord)?;
                    origin.preparation.qualify_recovery(
                        &entry.input,
                        pc,
                        RecoveryTemplate {
                            code: &stored.native.x86_64_code,
                            source_map: &stored.native.source_map,
                            source_bytes: stored.source_bytes,
                            direct_exits: &stored.direct_exits,
                            recovery: &stored.fault_recovery,
                        },
                    )
                }
                #[cfg(test)]
                Rows::Pending { template, rows, .. } => template.preparation.qualify_recovery(
                    &template.input,
                    pc,
                    RecoveryTemplate {
                        code: &template.block.native.x86_64_code,
                        source_map: &template.block.native.source_map,
                        source_bytes: template.block.source_bytes,
                        direct_exits: &template.block.direct_exits,
                        recovery: rows,
                    },
                ),
            }
            .map_err(NativeFailureReason::Qualification)?;
            self.rows = Rows::Qualified {
                code_bytes: self.rows.code_bytes(),
                rows: PackedRecovery::new(&block.fault_recovery),
            };
        }
        let Rows::Qualified { rows, .. } = &self.rows else {
            unreachable!("qualified above");
        };
        Ok(rows)
    }
}

impl Rows {
    fn code_bytes(&self) -> u32 {
        match *self {
            Self::Qualified { code_bytes, .. } | Self::Stored { code_bytes, .. } => code_bytes,
            #[cfg(test)]
            Self::Pending { code_bytes, .. } => code_bytes,
        }
    }
}

#[cfg(test)]
mod tests;
