//! Native MSR TTBR0_EL1 of the value the register holds
//! (`register_read::unchanged_write_thunk`) against the engine path: the
//! same value completes natively and leaves everything as the engine's MSR
//! does; another value exits to the engine.

use super::*;

const MSR: u32 = 0xd518_2003; // msr ttbr0_el1, x3

/// A machine at `el` whose table-base cell a first engine-path write of
/// `first` opened, on the storage of the register the write reached.
fn opened(el: u8, first: u64) -> Machine<FakeHost, NoDevices> {
    let mut machine = machine(&[MSR]);
    machine.cpu.pstate = Pstate::reset(el);
    machine.derived_changed();
    let base = machine.cpu.pc;
    machine.cpu.state.x[3] = first;
    machine.system_instruction(MSR).unwrap();
    let cell = machine.mirrors[TABLE_BASE_KEPT];
    assert_eq!(cell.readable, 1, "a plain write opens the cell");
    let reached = [SystemRegister::TTBR0_EL1, SystemRegister::TTBR0_EL2]
        .map(|register| machine.cpu.registers.address(register, 0));
    assert!(reached.contains(&Some(cell.value)));
    machine.cpu.pc = base + 4;
    machine
}

fn observed(machine: &Machine<FakeHost, NoDevices>) -> alloc::string::String {
    let open: Vec<Vec<u64>> = machine
        .mirrors
        .banks
        .iter()
        .map(|cells| cells.iter().map(|cell| cell.readable).collect())
        .collect();
    alloc::format!(
        "pc={:#x} el1={:#x} el2={:#x} x={:?} open={open:?} isolations={} regimes={} generation={} derived={}",
        machine.cpu.pc,
        machine.cpu.registers.get(SystemRegister::TTBR0_EL1, 0),
        machine.cpu.registers.get(SystemRegister::TTBR0_EL2, 0),
        machine.cpu.state.x,
        machine.host.isolations,
        machine.host.regimes.len(),
        machine.translation_generation,
        machine.native_system.is_some(),
    )
}

#[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
#[test]
fn native_unchanged_table_base_writes_match_the_engine_path() {
    use super::super::transitions::tests::run_native;
    const FIRST: u64 = 1 << 48 | 0x8000;
    const OTHER: u64 = 2 << 48 | 0xc000;
    for el in [1, 2] {
        let (mut engine, mut native) = (opened(el, FIRST), opened(el, FIRST));
        engine.native_system();
        native.native_system();
        let pc = engine.cpu.pc;
        assert_eq!(table_base_write(pc, MSR), Some(3));
        let cell = cell_place(TABLE_BASE_KEPT);
        let written = native.register_fixed_bits(SystemRegister::TTBR0_EL1);
        let thunk = register_read::unchanged_write_thunk(GuestAddress(pc), 3, cell, written);
        assert_eq!(observed(&engine), observed(&native));
        // The same value completes natively, again and again; another value
        // exits and the engine completes it; then the new one is kept.
        let mut exits = 0;
        for value in [FIRST, FIRST, OTHER, OTHER, FIRST] {
            for machine in [&mut engine, &mut native] {
                machine.cpu.pc = pc;
                machine.cpu.state.x[3] = value;
            }
            let storage = native.mirrors[TABLE_BASE_KEPT].value as *const u64;
            // SAFETY: the open cell's value is the address of the register's
            // storage in the machine, which lives and is not written now.
            let held = unsafe { storage.read() };
            engine.system_instruction(MSR).unwrap();
            let (next, kind) = run_native(&mut native, &thunk.code);
            if value == held {
                assert_eq!((next, kind), (pc + 4, ExitKind::Fallthrough as u64));
                native.cpu.pc = next;
            } else {
                assert_eq!((next, kind), (pc, ExitKind::System as u64), "{value:#x}");
                native.system_instruction(MSR).unwrap();
                exits += 1;
            }
            assert_eq!(observed(&engine), observed(&native), "EL{el} {value:#x}");
        }
        assert_eq!(exits, 2, "EL{el}: only the two changes exit");
    }
}

/// Code of the upper half, which a switch thunk serves.
const UPPER_PC: u64 = 0xffff_fe00_0001_0000;
const MSR_TCR: u32 = 0xd518_2049; // msr tcr_el1, x9

/// [`opened`] at the upper half's `UPPER_PC`, the host keeping root sets
/// where `sets` says so.
fn switching(el: u8, first: u64, sets: bool) -> Machine<FakeHost, NoDevices> {
    let mut machine = machine(&[MSR]);
    if sets {
        machine.host.sets = Some(FakeSets::new());
    }
    machine.cpu.pstate = Pstate::reset(el);
    machine.derived_changed();
    machine.sync_regime();
    machine.cpu.pc = UPPER_PC;
    machine.cpu.state.x[3] = first;
    machine.system_instruction(MSR).unwrap();
    assert_eq!(machine.mirrors[TABLE_BASE_KEPT].readable, 1);
    machine.cpu.pc = UPPER_PC + 4;
    machine
}

/// What a TTBR0 write leaves that the two paths must agree on.
fn switched(machine: &mut Machine<FakeHost, NoDevices>) -> alloc::string::String {
    let contexts = machine.switch_contexts();
    let sets = machine.host.sets.as_ref().map(|sets| {
        (
            sets.current,
            sets.cr3 & !table_switch::KEEP_ENTRIES,
            sets.keys.clone(),
        )
    });
    alloc::format!(
        "pc={:#x} el1={:#x} el2={:#x} x={:?} contexts={contexts:?} changes={} \
         prediction={:?} translations={} regime={:?} sets={sets:?}",
        machine.cpu.pc,
        machine.cpu.registers.get(SystemRegister::TTBR0_EL1, 0),
        machine.cpu.registers.get(SystemRegister::TTBR0_EL2, 0),
        machine.cpu.state.x,
        machine.statistics.table_base_changes,
        machine.prediction_context,
        machine.translation_generation,
        machine.host.regimes.last().map(|regime| regime.controls),
    )
}

/// Write `value` with MSR TTBR0 through the engine on `engine` and through
/// `thunk` on `native`, which completes it natively or exits to the
/// engine; whether it completed natively.
#[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
fn write_both(
    engine: &mut Machine<FakeHost, NoDevices>,
    native: &mut Machine<FakeHost, NoDevices>,
    thunk: &[u8],
    value: u64,
) -> bool {
    use super::super::transitions::tests::run_native;
    for machine in [&mut *engine, &mut *native] {
        machine.cpu.pc = UPPER_PC;
        machine.cpu.state.x[3] = value;
    }
    engine.system_instruction(MSR).unwrap();
    engine.cpu.pc = UPPER_PC + 4;
    // Native code runs in the contexts its fetch took.
    let contexts = native.switch_contexts();
    native.chain.set_contexts(contexts);
    let (next, kind) = run_native(native, thunk);
    let completed = kind == ExitKind::Fallthrough as u64;
    if completed {
        assert_eq!(next, UPPER_PC + 4);
        // The thunk wrote the contexts the engine would fetch in.
        let written = native.chain.contexts();
        assert_eq!(written, native.switch_contexts(), "{value:#x}");
    } else {
        assert_eq!(
            (next, kind),
            (UPPER_PC, ExitKind::System as u64),
            "{value:#x}"
        );
        native.system_instruction(MSR).unwrap();
    }
    native.cpu.pc = UPPER_PC + 4;
    completed
}

fn switch_thunk_of(machine: &mut Machine<FakeHost, NoDevices>) -> Vec<u8> {
    let sets = machine.host.native_sets();
    machine.mirrors.table_switch.sets = sets.map_or(0, |(at, _)| at);
    let places = table_switch_places(sets.map(|(_, cr3)| cr3), false);
    let written = machine.register_fixed_bits(SystemRegister::TTBR0_EL1);
    let cell = cell_place(TABLE_BASE_KEPT);
    table_switch::switch_thunk(GuestAddress(UPPER_PC), 3, cell, written, &places).code
}

/// What leaves every lookup context keeps the entries: a TLBI by VA of one
/// ASID or of all, and a change of SPRR permissions; the next native write
/// of a kept value completes as the engine path's does. Engine-path
/// changes count why their thunk would have missed: no entry, then a
/// forgotten one.
#[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
#[test]
fn native_table_switches_survive_what_keeps_contexts() {
    use super::super::table_switches::miss;
    use mmu::tlbi::{Everywhere, Reach};
    const FIRST: u64 = 1 << 48 | 0x8000;
    const OTHER: u64 = 2 << 48 | 0xc000;
    type Change = fn(&mut Machine<FakeHost, NoDevices>);
    let changes: [Change; 3] = [
        |machine| {
            let reach = Reach::Containing(0x4000);
            machine.tlb_maintained(reach, Everywhere::AsidAddresses(reach, 1), Some(1));
        },
        |machine| {
            let reach = Reach::Overlapping {
                first: 0,
                last: 1 << 40,
            };
            machine.tlb_maintained(reach, Everywhere::Addresses(reach), None);
        },
        |machine| machine.permissions_changed(),
    ];
    for sets in [false, true] {
        for (index, change) in changes.into_iter().enumerate() {
            let (mut engine, mut native) = (switching(1, FIRST, sets), switching(1, FIRST, sets));
            let thunk = switch_thunk_of(&mut native);
            // The first write, from the closed cell.
            assert_eq!(native.statistics.switch_misses[miss::CLOSED], 1);
            native.statistics.switch_misses = [0; miss::REASONS];
            assert!(!write_both(&mut engine, &mut native, &thunk, OTHER));
            assert!(write_both(&mut engine, &mut native, &thunk, FIRST));
            change(&mut engine);
            change(&mut native);
            assert_eq!(
                switched(&mut engine),
                switched(&mut native),
                "change {index}"
            );
            assert!(
                write_both(&mut engine, &mut native, &thunk, OTHER),
                "change {index} sets={sets}"
            );
            assert_eq!(
                switched(&mut engine),
                switched(&mut native),
                "change {index}"
            );
            let mut expected = [0; miss::REASONS];
            expected[miss::NEVER] = 1;
            assert_eq!(native.statistics.switch_misses, expected, "change {index}");
            native.mirrors.table_switch.forget();
            assert!(!write_both(&mut engine, &mut native, &thunk, FIRST));
            expected[miss::FORGOTTEN] = 1;
            assert_eq!(native.statistics.switch_misses, expected, "change {index}");
        }
    }
}

/// A process switch and a domain switch and back, through native TTBR0
/// writes, against the engine's: a write to a value the engine wrote or
/// left before completes natively and leaves the register, the lookup
/// contexts, the translation generation, the prediction context, the
/// host's regime and its root set as the engine's write does; any other
/// value exits.
#[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
#[test]
fn native_table_switches_match_the_engine_path() {
    const FIRST: u64 = 1 << 48 | 0x8000;
    const OTHER: u64 = 2 << 48 | 0xc000;
    const DOMAIN: u64 = 0x1_0000;
    // FIRST's ASID with another table.
    const REUSED: u64 = 1 << 48 | 0x2_0000;
    for el in [1, 2] {
        for sets in [false, true] {
            let (mut engine, mut native) = (switching(el, FIRST, sets), switching(el, FIRST, sets));
            let thunk = switch_thunk_of(&mut native);
            assert_eq!(switched(&mut engine), switched(&mut native));
            let mut completed = Vec::new();
            for value in [
                OTHER, FIRST, OTHER, FIRST, DOMAIN, FIRST, DOMAIN, FIRST, OTHER, REUSED, FIRST,
                REUSED,
            ] {
                completed.push(write_both(&mut engine, &mut native, &thunk, value));
                assert_eq!(
                    switched(&mut engine),
                    switched(&mut native),
                    "EL{el} sets={sets} {value:#x}"
                );
            }
            // A new value exits once; going back to the one it replaced,
            // or to it, completes natively from then on.
            assert_eq!(
                completed,
                [
                    false, true, true, true, false, true, true, true, true, false, true, true
                ],
                "EL{el} sets={sets}"
            );
            assert_eq!(native.statistics.native_table_switches, 9);
            if let Some(sets) = &native.host.sets {
                assert_eq!(sets.followed, 9, "the host followed every switch");
            }
        }
    }
}

/// Whatever may have taken a lookup context or changed what the sets
/// hold between a native switch and the next write sends that write to
/// the engine: a TLBI of an ASID, a TCR write, a TLBI of everything and a
/// switch of the PE (which forgets the incoming PE's entries). The two
/// paths stay equal throughout.
#[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
#[test]
fn native_table_switches_stop_where_contexts_may_have_gone() {
    use mmu::tlbi::{Everywhere, Reach};
    const FIRST: u64 = 1 << 48 | 0x8000;
    const OTHER: u64 = 2 << 48 | 0xc000;
    type Change = fn(&mut Machine<FakeHost, NoDevices>);
    let changes: [Change; 4] = [
        |machine| machine.tlb_maintained(Reach::All, Everywhere::NonGlobal(2), Some(2)),
        |machine| {
            let tcr = machine.cpu.registers.get(SystemRegister::TCR_EL1, 0);
            machine.cpu.state.x[9] = tcr ^ 1 << 7;
            machine.cpu.pc = UPPER_PC;
            machine.system_instruction(MSR_TCR).unwrap();
            machine.cpu.pc = UPPER_PC + 4;
        },
        |machine| machine.tlb_maintained(Reach::All, Everywhere::All, None),
        |machine| machine.mirrors.table_switch.forget(),
    ];
    for sets in [false, true] {
        for (index, change) in changes.into_iter().enumerate() {
            let (mut engine, mut native) = (switching(1, FIRST, sets), switching(1, FIRST, sets));
            let thunk = switch_thunk_of(&mut native);
            assert!(!write_both(&mut engine, &mut native, &thunk, OTHER));
            assert!(write_both(&mut engine, &mut native, &thunk, FIRST));
            change(&mut engine);
            change(&mut native);
            assert_eq!(
                switched(&mut engine),
                switched(&mut native),
                "change {index}"
            );
            assert!(
                !write_both(&mut engine, &mut native, &thunk, OTHER),
                "change {index} sets={sets}"
            );
            assert_eq!(
                switched(&mut engine),
                switched(&mut native),
                "change {index}"
            );
            // A change of derived state closes the register's cell too:
            // the next write opens it, and the one after keeps entries.
            write_both(&mut engine, &mut native, &thunk, FIRST);
            assert_eq!(
                switched(&mut engine),
                switched(&mut native),
                "change {index}"
            );
            for value in [OTHER, FIRST] {
                assert!(
                    write_both(&mut engine, &mut native, &thunk, value),
                    "change {index} sets={sets}"
                );
                assert_eq!(
                    switched(&mut engine),
                    switched(&mut native),
                    "change {index}"
                );
            }
        }
    }
}

/// A set the host gave to another context, or closed, sends the write to
/// the engine, as does one of another class.
#[cfg(all(target_arch = "x86_64", any(target_os = "macos", target_os = "linux")))]
#[test]
fn native_table_switches_follow_the_hosts_sets() {
    const FIRST: u64 = 1 << 48 | 0x8000;
    const OTHER: u64 = 2 << 48 | 0xc000;
    type Change = fn(&mut FakeSets);
    // FIRST's set.
    let changes: [Change; 3] = [
        |sets| sets.sets.generations[sets.keys.iter().position(|&key| key == FIRST).unwrap()] += 1,
        |sets| sets.sets.open = 0,
        |sets| sets.sets.classes[sets.keys.iter().position(|&key| key == FIRST).unwrap()] = 2,
    ];
    for (index, change) in changes.into_iter().enumerate() {
        let (mut engine, mut native) = (switching(1, FIRST, true), switching(1, FIRST, true));
        let thunk = switch_thunk_of(&mut native);
        assert!(!write_both(&mut engine, &mut native, &thunk, OTHER));
        change(native.host.sets.as_mut().unwrap());
        assert!(
            !write_both(&mut engine, &mut native, &thunk, FIRST),
            "change {index}"
        );
        assert_eq!(
            switched(&mut engine),
            switched(&mut native),
            "change {index}"
        );
    }
}

/// What a TTBR0 switch between two kept values costs the engine's own
/// path (`system_instruction`, the exit's round trip and the host's work
/// left out) and a native switch thunk (without CR3, which a user process
/// cannot load), with the engine's settling after each run.
///
/// `cargo test -p stela-engine --release --lib table_switch_benchmark --
/// --ignored --nocapture`
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
#[ignore = "manual: opt-in"]
fn table_switch_benchmark() {
    use core::ffi::c_void;
    unsafe extern "C" {
        fn mmap(a: *mut c_void, b: usize, p: i32, f: i32, fd: i32, o: i64) -> *mut c_void;
        fn mprotect(a: *mut c_void, b: usize, p: i32) -> i32;
    }
    const FIRST: u64 = 1 << 48 | 0x8000;
    const OTHER: u64 = 2 << 48 | 0xc000;
    const ROUNDS: u64 = 200_000;
    let (mut engine, mut native) = (switching(1, FIRST, true), switching(1, FIRST, true));
    let thunk = switch_thunk_of(&mut native);
    assert!(!write_both(&mut engine, &mut native, &thunk, OTHER));
    assert!(write_both(&mut engine, &mut native, &thunk, FIRST));
    let start = std::time::Instant::now();
    for round in 0..ROUNDS {
        engine.cpu.pc = UPPER_PC;
        engine.cpu.state.x[3] = if round % 2 == 0 { OTHER } else { FIRST };
        engine.system_instruction(MSR).unwrap();
    }
    let engine_ns = start.elapsed().as_nanos() as f64 / ROUNDS as f64;
    let code = stela_compiler::x86_64::pinned::entered(&thunk);
    // SAFETY: private anonymous memory, RX after the copy, never unmapped:
    // the test's process ends with it.
    let entry: unsafe extern "sysv64" fn(*mut GuestState) -> [u64; 2] = unsafe {
        let page = mmap(core::ptr::null_mut(), 8192, 3, 2 | 0x20, -1, 0);
        assert_ne!(page as usize, usize::MAX);
        core::ptr::copy_nonoverlapping(code.as_ptr(), page.cast(), code.len());
        assert_eq!(mprotect(page, 8192, 5), 0);
        core::mem::transmute(page)
    };
    super::super::transitions::tests::enter_native(&mut native);
    let contexts = native.switch_contexts();
    native.chain.set_contexts(contexts);
    let mut measure = |settle: bool| {
        let start = std::time::Instant::now();
        for round in 0..ROUNDS {
            native.cpu.state.x[3] = if round % 2 == 0 { OTHER } else { FIRST };
            // SAFETY: the thunk's code, on the machine's frame.
            let exit = unsafe { entry(&mut native.cpu.state) };
            assert_eq!(exit[1], ExitKind::Fallthrough as u64);
            if settle {
                native.settle_native_switches();
                native.cpu.state.anchors = native.anchors();
            }
        }
        start.elapsed().as_nanos() as f64 / ROUNDS as f64
    };
    let thunk_ns = measure(false);
    let settled_ns = measure(true);
    println!(
        "TTBR0 switch: engine path {engine_ns:.1} ns, native thunk {thunk_ns:.1} ns, \
         thunk and settle {settled_ns:.1} ns"
    );
}
