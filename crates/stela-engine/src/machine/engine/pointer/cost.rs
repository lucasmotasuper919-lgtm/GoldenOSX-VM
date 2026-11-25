//! What PAC instructions cost in native code, run by hand in a release
//! build:
//!
//!     cargo test --release -p stela-engine --lib -- --ignored --exact \
//!       machine::engine::pointer::cost::pac_costs --nocapture
//!
//! Each case is a system-mode block compiled as the engine compiles it on
//! this host (the cipher in line where the host has AES-NI), run on an
//! open IB key under FEAT_FPAC (the M2's version) and a macOS TCR, against
//! a block of the same shape without the PAC instructions. A branch form
//! that ends its block with a PAC exit runs the thunk the engine installs
//! over that exit instead. `PAC COST` lines give host instructions
//! (single-stepped) and cycles per operation, the run's difference over
//! its operations; cycles come from nanoseconds at the clock a chain of
//! dependent additions measures.
extern crate std;

use super::*;
use std::time::Instant;
use stela_compiler::{
    GuestAddress, GuestState, TargetCpu,
    block::{BlockProfile, CompilationPlan, ExitKind, Features},
    x86_64::{NativeSystem, block_entry, pinned},
};

const PC: u64 = 0x4000;
const RUNS: u64 = 1 << 18;

/// Executable copies of native code, unmapped on drop.
struct Mapped {
    page: *mut core::ffi::c_void,
    bytes: usize,
}

unsafe extern "C" {
    fn mmap(
        address: *mut core::ffi::c_void,
        bytes: usize,
        protection: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut core::ffi::c_void;
    fn mprotect(address: *mut core::ffi::c_void, bytes: usize, protection: i32) -> i32;
    fn munmap(address: *mut core::ffi::c_void, bytes: usize) -> i32;
}

impl Mapped {
    fn new(body: &[u8]) -> Self {
        let code = pinned::entered(body);
        let bytes = code.len().next_multiple_of(4096);
        // SAFETY: private anonymous memory, RX after the copy.
        unsafe {
            let page = mmap(core::ptr::null_mut(), bytes, 3, 0x22, -1, 0);
            assert_ne!(page as usize, usize::MAX);
            core::ptr::copy_nonoverlapping(code.as_ptr(), page.cast(), code.len());
            assert_eq!(mprotect(page, bytes, 5), 0);
            Self { page, bytes }
        }
    }

    fn run(&self, state: &mut GuestState) -> (u64, u64) {
        // SAFETY: block ABI code touching only `state`, its anchors' areas
        // and the stack, calling the service.
        let exit = unsafe { block_entry::enter(self.page.cast(), state, 0, 0) };
        (exit as u64, (exit >> 64) as u64)
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: mapped by `new`, no longer running.
        unsafe { assert_eq!(munmap(self.page, self.bytes), 0) };
    }
}

fn block(words: &[u32]) -> Mapped {
    let source: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let options = stela_compiler::Options::new(
        BlockProfile::new(Features::SYSTEM_MODE),
        TargetCpu::detect(),
    )
    .with_native_system(NativeSystem {
        pointer: Some(cells(TargetCpu::detect())),
        ..NativeSystem::NONE
    });
    let block = CompilationPlan::new(&source, GuestAddress(PC), &options)
        .unwrap()
        .compile()
        .unwrap();
    assert_eq!(block.source_bytes, source.len());
    Mapped::new(&block.native.x86_64_code)
}

/// Cycles per nanosecond: a chain of dependent additions, one a cycle.
fn clock() -> f64 {
    const ITERATIONS: u64 = 1 << 26;
    let started = Instant::now();
    // SAFETY: register arithmetic only.
    unsafe {
        core::arch::asm!(
            "2:",
            "add {x}, 1",
            "add {x}, 1",
            "add {x}, 1",
            "add {x}, 1",
            "dec {n}",
            "jnz 2b",
            x = inout(reg) 0u64 => _,
            n = inout(reg) ITERATIONS => _,
        );
    }
    (4 * ITERATIONS) as f64 / started.elapsed().as_nanos() as f64
}

/// Nanoseconds per run and instructions of one run of `code` from the
/// state `prepare` sets.
fn measure(
    code: &Mapped,
    state: &mut GuestState,
    prepare: &dyn Fn(&mut GuestState, u64),
) -> (f64, usize) {
    for run in 0..RUNS / 8 {
        prepare(state, run);
        core::hint::black_box(code.run(state));
    }
    let started = Instant::now();
    for run in 0..RUNS {
        prepare(state, run);
        core::hint::black_box(code.run(state));
    }
    let ns = started.elapsed().as_nanos() as f64 / RUNS as f64;
    prepare(state, 1);
    let steps = super::tests::steps(|| {
        core::hint::black_box(code.run(state));
    });
    (ns, steps)
}

#[test]
#[ignore = "manual: opt-in"]
fn pac_costs() {
    use stela_contracts::pauth::Key;
    // EL0 47-bit and kernel 39-bit ranges, TBI0 with TBID0, as macOS.
    let tcr = 17 | 25 << 16 | 1 << 37 | 1 << 51;
    let mut service = Box::new(PointerService::new());
    service.open(super::tests::inputs(pauth::Version::Faulting, tcr));
    for key in [Key::Ia, Key::Ib, Key::Da, Key::Db] {
        service.reopen(key, true, false);
    }
    let mut engine = PointerService::new();
    engine.open(super::tests::inputs(pauth::Version::Faulting, tcr));
    let (anchor, memo) = service.anchors();
    let mut state = GuestState::default();
    state.anchors[Anchor::Pointer] = anchor;
    state.anchors[Anchor::Memo] = memo;
    let modifier = 0xffff_ffe0_1234_5670u64;
    let sign = |engine: &mut PointerService, key, pointer| match engine.result(
        PointerOperation::Sign,
        key,
        pointer,
        modifier,
    ) {
        Some(Ok((signed, _))) => signed,
        other => panic!("{other:?}"),
    };
    // Kernel code pointers, a different one for each register.
    let pointer = |register: usize| 0xffff_ffe0_0800_0000 + register as u64 * 0x40;
    let registers = [3usize, 4, 6, 7, 8, 9, 10, 11, 12, 13];
    let signed: Vec<u64> = (0..32)
        .map(|register| sign(&mut engine, Key::Ib, pointer(register)))
        .collect();
    let add = |register: usize| 0x9100_0400 | (register as u32) << 5 | register as u32;
    let pacib = |register: usize| 0xdac1_0400 | 5 << 5 | register as u32;
    let autib = |register: usize| 0xdac1_1400 | 5 << 5 | register as u32;
    let clock = clock();
    let report = |what: &str,
                  operations: usize,
                  code: &Mapped,
                  base: &Mapped,
                  state: &mut GuestState,
                  prepare: &dyn Fn(&mut GuestState, u64)| {
        let (ns, steps) = measure(code, state, prepare);
        let (base_ns, base_steps) = measure(base, state, prepare);
        let per = |value: f64| value / operations as f64;
        std::println!(
            "PAC COST {what}: instructions={:.1} cycles={:.1} (run {steps} vs {base_steps}, {ns:.2} vs {base_ns:.2} ns)",
            per(steps as f64 - base_steps as f64),
            per((ns - base_ns) * clock),
        );
    };
    std::println!("PAC COST clock {clock:.2} cycles/ns");
    let plain = |state: &mut GuestState, _| {
        for register in registers {
            state.x[register] = pointer(register);
        }
        state.x[5] = modifier;
    };
    let adds = block(&registers.map(add));
    report(
        "sign (10 PACIB, independent)",
        10,
        &block(&registers.map(pacib)),
        &adds,
        &mut state,
        &plain,
    );
    let authentic = |state: &mut GuestState, _| {
        for register in registers {
            state.x[register] = signed[register];
        }
        state.x[5] = modifier;
    };
    report(
        "authenticate (10 AUTIB, independent)",
        10,
        &block(&registers.map(autib)),
        &adds,
        &mut state,
        &authentic,
    );
    let pairs: Vec<u32> = (0..8).flat_map(|_| [pacib(3), autib(3)]).collect();
    let pair_adds: Vec<u32> = (0..16).map(|_| add(3)).collect();
    report(
        "sign then authenticate (8 dependent pairs, per operation)",
        16,
        &block(&pairs),
        &block(&pair_adds),
        &mut state,
        &plain,
    );
    // The branch forms against BR, BLR and RET, each alone in its block:
    // in line, or the block's PAC exit and the thunk the engine installs.
    // RETAB's modifier is SP: the same value.
    let sp = modifier;
    let return_signed = sign(&mut engine, Key::Ib, 0xffff_ffe0_0800_1230);
    let branch_target = signed[3];
    use stela_compiler::block::{
        PointerAuthenticationKey as K, PointerAuthenticationModifier as M,
        PointerAuthenticationOperation as Op,
    };
    let cases: [(&str, u32, u32, Op, ExitKind); 3] = [
        (
            "RETAB (against RET)",
            0xd65f_0fff,
            0xd65f_03c0,
            Op::ReturnIbSp,
            ExitKind::Return,
        ),
        (
            "BRAB X3, X5 (against BR X3)",
            0xd71f_0c65,
            0xd61f_0060,
            Op::Branch {
                key: K::Ib,
                target: 3,
                modifier: M::Register(5),
                link: false,
            },
            ExitKind::Indirect,
        ),
        (
            "BLRAB X3, X5 (against BLR X3)",
            0xd73f_0c65,
            0xd63f_0060,
            Op::Branch {
                key: K::Ib,
                target: 3,
                modifier: M::Register(5),
                link: true,
            },
            ExitKind::Call,
        ),
    ];
    for (what, word, plain_word, operation, kind) in cases {
        let prepare = |state: &mut GuestState, _| {
            state.x[30] = return_signed;
            state.sp = sp;
            state.x[3] = branch_target;
            state.x[5] = modifier;
        };
        let mut code = block(&[word]);
        prepare(&mut state, 0);
        let mut through = "in line";
        if code.run(&mut state) == (PC, ExitKind::PointerAuthentication as u64) {
            let form = pointer::Form::of(operation).unwrap();
            let thunk = pointer::thunk(
                form,
                GuestAddress(PC),
                TAGS,
                SERVICE as usize as u64,
                native_cipher(TargetCpu::detect()),
            );
            code = Mapped::new(&thunk.code);
            through = "thunk";
        }
        prepare(&mut state, 0);
        let exit = code.run(&mut state);
        assert_eq!(exit.1, kind as u64, "{what}");
        let expected = if kind == ExitKind::Return {
            0xffff_ffe0_0800_1230
        } else {
            pointer(3)
        };
        assert_eq!(exit.0, expected, "{what}");
        report(
            &alloc::format!("{what}, {through}"),
            1,
            &code,
            &block(&[plain_word]),
            &mut state,
            &prepare,
        );
    }
}
