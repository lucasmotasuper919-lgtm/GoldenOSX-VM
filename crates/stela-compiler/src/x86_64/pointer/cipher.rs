//! PAC instructions computed in native code, without the memo.
//!
//! Where the host has AES-NI, the engine's PAC cipher is the whitening key
//! and six full AES rounds of the pointer and modifier under the key
//! (`pauth::keyed`, the PAC algorithm Apple silicon leaves IMPLEMENTATION
//! DEFINED), whose state's low half is the PAC. Native code runs those
//! rounds in fewer instructions than a memo miss spends reaching the
//! service, and a memo hit is a load from a table far larger than the
//! caches. So a block or thunk given a [`Cipher`] computes every signing,
//! successful authentication and PACGA itself, from what the engine keeps
//! there for the current keys, PAC version and TCR: each key's schedule
//! (the whitening key and six round keys) and, for each domain, the PAC
//! field of each range ([`Range`]).
//!
//! Native code completes canonical pointers only, whose extension bits all
//! equal bit 55. For them Arm `AddPAC` and `Auth` take one layout, the one
//! bit 55 selects (a signing that selects by bit 63 sees the same bit
//! there), so the range table is indexed by bit 55 alone. With `E` that
//! range's extension mask, `M` it without bit 55 and `pac` the cipher of
//! the input:
//! - a signing of a canonical pointer (`p & E` is the range's canonical
//!   extension) has the pointer as its input and gives
//!   `p ^ (pac & M) ^ signed`, where `signed` turns the classic insertion,
//!   which replaces an upper pointer's all-ones field, into that XOR;
//! - an authentication's input is `p` stripped, `(p & !E) | canonical`; it
//!   succeeds exactly when `(p ^ pac) & M` is the range's `authenticated`
//!   (the input's field under FEAT_PAuth2's XOR, zero under the classic
//!   comparison), and then gives its input;
//! - PACGA gives the cipher's upper half.
//!
//! A key whose tag word is [`super::DISABLED`] (its SCTLR enable clear)
//! gives the pointer unchanged, as Arm does, from the tag word alone: native
//! transitions switch the tag words between two exception levels of one
//! memo generation (`transition::check_pointer`), whose enables differ while
//! their keys, version and layouts do not, so nothing else in the cipher
//! may depend on an enable. One compare tells an open key from a closed or
//! disabled one; the out-of-line path then tells those two apart
//! (`Lookup::unopened`). Everything else, a closed key (tag word
//! zero), a signing of a non-canonical pointer and a failed authentication,
//! goes to the service as a memo miss does, which computes it the engine's
//! way. The computation clobbers RAX, R8 to R11, XMM0, XMM1 and flags.
use super::{GENERIC_KEY, Lookup, TAG_WORDS, operands};
use crate::Place;
use crate::x86_64::asm::{Alu, Asm, BitTest, Cond, Gpr, Mem, Rm, ShiftOp, Size};
use alloc::vec::Vec;

/// The words of a key's schedule: the whitening key and six round keys,
/// each low quadword first.
pub const SCHEDULE: usize = 7;

/// The bit that selects a canonical pointer's range: Arm keeps it in a
/// signed pointer.
const RANGE_BIT: u8 = 55;

/// Where one range of one domain puts its PAC field, as native code applies
/// it to canonical pointers ([`Range::new`]).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Range {
    /// `E`: the extension bits, from bit 64 - TnSZ to the top PAC bit.
    pub extension: u64,
    /// `!E`: the bits a stripped pointer keeps.
    pub kept: u64,
    /// A canonical pointer's extension in this range: zero in the lower
    /// range, `E` in the upper one.
    pub canonical: u64,
    /// `M`: `E` without bit 55, the PAC field.
    pub field: u64,
    /// What a signing XORs into the field besides the PAC: `M` in the upper
    /// range under the classic insertion, which replaces the field (all
    /// ones there), else zero.
    pub signed: u64,
    /// What `(p ^ pac) & M` is for an authentic `p`: `M` in the upper range
    /// under FEAT_PAuth2, which XORs the PAC into the field (all ones in
    /// the input), else zero.
    pub authenticated: u64,
}

impl Range {
    /// The range whose extension mask is `extension` (it holds bit 55), the
    /// upper one if `upper`, under the classic insertion if `classic`.
    pub const fn new(extension: u64, upper: bool, classic: bool) -> Self {
        let field = extension & !(1 << RANGE_BIT);
        Self {
            extension,
            kept: !extension,
            canonical: if upper { extension } else { 0 },
            field,
            signed: if upper && classic { field } else { 0 },
            authenticated: if upper && !classic { field } else { 0 },
        }
    }
}

/// What native code computes PAC results from, at a fixed host address.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Cipher {
    /// Each tag word's key schedule ([`SCHEDULE`]).
    pub schedules: [[[u64; 2]; SCHEDULE]; TAG_WORDS],
    /// The instruction keys' ranges, then the data keys', each by bit 55.
    pub ranges: [[Range; 2]; 2],
}

const SCHEDULES: usize = core::mem::offset_of!(Cipher, schedules);
const RANGES: usize = core::mem::offset_of!(Cipher, ranges);
/// Bytes of a [`Range`], which bit 55 indexes.
const RANGE_BYTES: i32 = core::mem::size_of::<Range>() as i32;
const _: () = assert!(RANGE_BYTES == 64 && SCHEDULES.is_multiple_of(16));
const EXTENSION: usize = core::mem::offset_of!(Range, extension);
const KEPT: usize = core::mem::offset_of!(Range, kept);
const CANONICAL: usize = core::mem::offset_of!(Range, canonical);
const FIELD: usize = core::mem::offset_of!(Range, field);
const SIGNED: usize = core::mem::offset_of!(Range, signed);
const AUTHENTICATED: usize = core::mem::offset_of!(Range, authenticated);

/// `operation` on XMM0 with ModRM.rm `rm`, a legacy SSE form.
fn sse(asm: &mut Asm, opcode: &[u8], rm: impl Into<Rm>) {
    asm.sse(Some(0x66), opcode, 0, rm);
}

/// The cipher of `data` and `modifier` under the schedule at R8 +
/// `schedule` into XMM0, whose low half is the PAC: the whitening key and
/// the six AES rounds of `pauth::keyed`. Clobbers XMM1.
fn rounds(asm: &mut Asm, data: Gpr, modifier: Gpr, schedule: i32) {
    const PXOR: [u8; 2] = [0x0f, 0xef];
    const AESENC: [u8; 3] = [0x0f, 0x38, 0xdc];
    asm.movq_to_xmm(0, data);
    asm.movq_to_xmm(1, modifier);
    asm.punpcklqdq(0, Rm::Xmm(1));
    let word = |index: i32| Mem::base(Gpr::R8, schedule + 16 * index);
    sse(asm, &PXOR, word(0));
    for round in 1..SCHEDULE as i32 {
        sse(asm, &AESENC, word(round));
    }
}

/// R11 the byte offset of the range bit 55 of RAX selects: 0 or
/// [`RANGE_BYTES`].
fn range_index(asm: &mut Asm) {
    use Gpr::{R11, Rax};
    asm.bit_imm(BitTest::Bt, Size::Qword, Rax, RANGE_BIT);
    asm.alu(Alu::Sbb, Size::Qword, R11, R11);
    asm.alu_imm8(Alu::And, Size::Dword, R11, RANGE_BYTES as i8);
}

/// The result of `form` on its operands, computed from the [`Cipher`] at
/// `cipher` under the tag words at `tags` (one anchor): RAX the result, the
/// misses' jumps with RAX the pointer and the modifier in its register, as
/// [`super::lookup`] leaves them, and the jump taken while the key is not
/// open.
pub(super) fn compute(asm: &mut Asm, form: super::Form, tags: Place, cipher: Place) -> Lookup {
    use Gpr::{R8, R9, R10, R11, Rax};
    use Size::Qword;
    assert_eq!(
        tags.anchor, cipher.anchor,
        "the tag words and cipher share an anchor"
    );
    let modifier = operands(asm, form);
    debug_assert!(!matches!(modifier, R9 | R10 | R11));
    asm.place(R8, Place::new(tags.anchor, 0));
    let at = |offset: usize| i32::try_from(cipher.offset + offset).expect("a cipher field offset");
    // Every open key's tag word is above the disabled one's, and a closed
    // key's is below it.
    let word = i32::try_from(tags.offset + usize::from(form.key) * 8).expect("a tag word offset");
    asm.alu_imm32(Alu::Cmp, Qword, Mem::base(R8, word), super::DISABLED as i32);
    let unopened = asm.jcc(Cond::Be);
    let schedule = at(SCHEDULES + usize::from(form.key) * SCHEDULE * 16);
    if form.key == GENERIC_KEY {
        // The generic key is never disabled: a closed one misses.
        rounds(asm, Rax, modifier, schedule);
        asm.movq_from_xmm(Rax, 0);
        asm.shift_imm(ShiftOp::Shr, Qword, Rax, 32);
        asm.shift_imm(ShiftOp::Shl, Qword, Rax, 32);
        return Lookup {
            misses: alloc::vec![unopened],
            modifier,
            unopened: None,
        };
    }
    let data = form.key >= super::key_index(crate::block::PointerAuthenticationKey::Da);
    let domain = RANGES + usize::from(data) * 2 * RANGE_BYTES as usize;
    let range = |field: usize| Mem::indexed(R8, R11, 1, at(domain + field));
    let mut misses = Vec::new();
    if form.kind != super::AUTHENTICATED {
        // The cipher of the pointer itself, a canonical one's input.
        rounds(asm, Rax, modifier, schedule);
        range_index(asm);
        asm.mov(Qword, R10, Rax);
        asm.alu_load(Alu::And, Qword, R10, range(EXTENSION));
        asm.alu_load(Alu::Cmp, Qword, R10, range(CANONICAL));
        misses.push(asm.jcc(Cond::Ne));
        // p ^ (pac & M) ^ signed.
        asm.movq_from_xmm(R9, 0);
        asm.alu_load(Alu::And, Qword, R9, range(FIELD));
        asm.alu_load(Alu::Xor, Qword, R9, range(SIGNED));
        asm.alu(Alu::Xor, Qword, Rax, R9);
    } else {
        // R9 = the input: (p & !E) | canonical.
        range_index(asm);
        asm.mov(Qword, R9, Rax);
        asm.alu_load(Alu::And, Qword, R9, range(KEPT));
        asm.alu_load(Alu::Or, Qword, R9, range(CANONICAL));
        rounds(asm, R9, modifier, schedule);
        // Authentic when (p ^ pac) & M is the range's `authenticated`.
        asm.movq_from_xmm(R10, 0);
        asm.alu(Alu::Xor, Qword, R10, Rax);
        asm.alu_load(Alu::And, Qword, R10, range(FIELD));
        asm.alu_load(Alu::Cmp, Qword, R10, range(AUTHENTICATED));
        misses.push(asm.jcc(Cond::Ne));
        asm.mov(Qword, Rax, R9);
    }
    Lookup {
        misses,
        modifier,
        unopened: Some(unopened),
    }
}
