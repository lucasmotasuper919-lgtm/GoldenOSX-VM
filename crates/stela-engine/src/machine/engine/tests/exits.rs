//! Host timing of the engine's most frequent System exits: SVC from EL0 and
//! the ERET back, and the GENTER/GEXIT pair, through the same completion as
//! a native System exit (`Machine::completing` and
//! `complete_system_instruction`), followed by the variant the next fetch
//! derives. The manual test prints nanoseconds per transition; the others
//! check that the repeated transitions keep their architectural results.

use super::super::guarded::tests::gxf;
use super::*;
use crate::machine::exception::{DAIF_ALL, Pstate};

const SVC: u32 = 0xd400_0001;
const ERET: u32 = 0xd69f_03e0;
const GENTER: u32 = 0x0020_1420;
const GEXIT: u32 = 0x0020_1400;
const USER: u64 = BASE + 0x4000;
const VECTORS: u64 = BASE + 0x8000;
/// The lower-EL AArch64 synchronous vector.
const LOWER_SYNCHRONOUS: u64 = 0x400;
const HCR_E2H_TGE: u64 = 1 << 34 | 1 << 27;

/// One timed piece of work on a machine.
type Round = fn(&mut Machine<FakeHost, NoDevices>);

/// Complete `word` at the PC as a native System exit does.
fn exit(machine: &mut Machine<FakeHost, NoDevices>, word: u32) {
    let pc = machine.cpu.pc;
    let operation = machine.decode_system_operation(word);
    machine
        .completing(pc, |machine| {
            machine.complete_system_instruction(word, operation)
        })
        .unwrap();
}

/// A machine at EL0 with stage 1 on, whose SVCs go to EL1, or with `host`
/// to EL2 under HCR_EL2.{E2H,TGE}.
fn user(host: bool) -> Machine<FakeHost, NoDevices> {
    user_on(crate::machine::test_chip(), host)
}

/// [`user`] on `chip`.
fn user_on(chip: stela_contracts::Chip, host: bool) -> Machine<FakeHost, NoDevices> {
    let mut machine = machine_on(chip, &[]);
    normal_memory(&mut machine, true);
    let registers = &mut machine.cpu.registers;
    registers.set(SystemRegister::VBAR_EL1, 0, VECTORS);
    registers.set(SystemRegister::VBAR_EL2, 0, VECTORS);
    if host {
        registers.set(SystemRegister::HCR_EL2, 0, HCR_E2H_TGE);
    }
    machine.cpu.set_pstate(Pstate {
        sp: false,
        daif: 0,
        ..Pstate::reset(0)
    });
    machine.sync_regime();
    machine.cpu.pc = USER;
    machine
}

/// One SVC and its ERET, each with the next fetch's variant.
fn system_call(machine: &mut Machine<FakeHost, NoDevices>) {
    machine.cpu.pc = USER;
    exit(machine, SVC);
    machine.current_variant(false, false);
    exit(machine, ERET);
    machine.current_variant(false, false);
    machine.host.events.clear();
    machine.host.regimes.clear();
}

/// One GENTER and its GEXIT at EL1, each with the next fetch's variant.
fn guarded_call(machine: &mut Machine<FakeHost, NoDevices>, at: u64) {
    machine.cpu.pc = at;
    exit(machine, GENTER);
    machine.current_variant(false, false);
    exit(machine, GEXIT);
    machine.current_variant(false, false);
    machine.host.events.clear();
    machine.host.regimes.clear();
}

#[test]
fn repeated_system_calls_enter_and_return_exactly() {
    for host in [false, true] {
        let target = if host { 2 } else { 1 };
        let mut machine = user(host);
        for round in 0..4u64 {
            machine.cpu.pc = USER;
            machine.cpu.state.nzcv = 0x6000_0000;
            let retired = machine.cpu.retired;
            exit(&mut machine, SVC);
            assert_eq!(machine.cpu.pstate.el, target, "{host} {round}");
            assert_eq!(machine.cpu.pstate.daif, DAIF_ALL);
            assert_eq!(machine.cpu.pc, VECTORS + LOWER_SYNCHRONOUS);
            assert_eq!(machine.cpu.retired, retired + 1);
            let (elr, spsr, esr) = if host {
                (
                    SystemRegister::ELR_EL2,
                    SystemRegister::SPSR_EL2,
                    SystemRegister::ESR_EL2,
                )
            } else {
                (
                    SystemRegister::ELR_EL1,
                    SystemRegister::SPSR_EL1,
                    SystemRegister::ESR_EL1,
                )
            };
            assert_eq!(machine.cpu.read(elr), USER + 4);
            assert_eq!(machine.cpu.read(spsr), 0x6000_0000);
            assert_eq!(machine.cpu.read(esr), 0x5600_0000);
            let entered = machine.current_variant(false, false);
            exit(&mut machine, ERET);
            assert_eq!(machine.cpu.pstate.el, 0);
            assert_eq!(machine.cpu.pstate.daif, 0);
            assert_eq!(machine.cpu.pc, USER + 4);
            let returned = machine.current_variant(false, false);
            // EL0 checks SP alignment with SA0, EL1 and EL2 with SA.
            let sctlr = machine.cpu.regime_sctlr();
            assert_eq!(returned.checks.stack_alignment, sctlr & 1 << 4 != 0);
            machine.cpu.pc = USER;
            exit(&mut machine, SVC);
            assert_eq!(machine.current_variant(false, false), entered);
            exit(&mut machine, ERET);
            assert_eq!(machine.current_variant(false, false), returned);
        }
    }
}

/// Nanoseconds per transition of `round`, which makes `transitions` of
/// them, the best of several runs.
fn time(
    machine: &mut Machine<FakeHost, NoDevices>,
    transitions: u64,
    mut round: impl FnMut(&mut Machine<FakeHost, NoDevices>),
) -> u64 {
    const ROUNDS: u64 = 20_000;
    for _ in 0..ROUNDS / 10 {
        round(machine);
    }
    (0..5)
        .map(|_| {
            let started = std::time::Instant::now();
            for _ in 0..ROUNDS {
                round(machine);
            }
            started.elapsed().as_nanos() as u64 / (ROUNDS * transitions)
        })
        .min()
        .unwrap()
}

/// `cargo test -p stela-engine --release --lib exit_transition_costs --
/// --ignored --nocapture`
#[test]
#[ignore = "manual: opt-in"]
fn exit_transition_costs() {
    for host in [false, true] {
        let mut machine = user(host);
        let ns = time(&mut machine, 2, system_call);
        std::println!("EXITS svc+eret host={host} ns_per_transition={ns}");
    }
    let mut machine = gxf(stela_contracts::Chip::AppleM2);
    let at = machine.cpu.pc;
    let ns = time(&mut machine, 2, |machine| guarded_call(machine, at));
    std::println!("EXITS genter+gexit ns_per_transition={ns}");
    // Four PEs of one M2 cluster at once, whose transitions all read the
    // cluster's CTRR groups: unlocked, under its lock; locked, without it.
    for locked in [false, true] {
        let ns = contended(locked);
        std::println!("EXITS svc+eret pes=4 ctrr_locked={locked} ns_per_transition={ns}");
    }
    // The parts of one transition, at EL1 after an SVC.
    let mut machine = user(false);
    exit(&mut machine, SVC);
    let parts = parts();
    for (name, part) in parts {
        let ns = time(&mut machine, 1, part);
        std::println!("EXITS part={name} ns={ns}");
    }
    // The memory-management words a kernel exits on most, at EL1, and a
    // WFE that finds an event (an event stream edge native WFEs now take).
    let words: [(&str, Round); 3] = [
        ("wfe-at-once", |machine| {
            machine.cpu.signal_event();
            machine.cpu.pc = USER;
            exit(machine, 0xd503_205f);
        }),
        ("tlbi-vale1is", |machine| {
            machine.cpu.state.x[8] = 0x4_0000;
            machine.cpu.pc = USER;
            exit(machine, 0xd508_83a8);
        }),
        ("msr-ttbr0", |machine| {
            let ttbr0 = machine.cpu.read(SystemRegister::TTBR0_EL1);
            machine.cpu.state.x[0] = ttbr0 ^ 1 << 48;
            machine.cpu.pc = USER;
            exit(machine, 0xd518_2000);
            machine.current_variant(false, false);
        }),
    ];
    for (name, word) in words {
        let ns = time(&mut machine, 1, word);
        std::println!("EXITS word={name} ns={ns}");
    }
    // A direct exit's whole round trip through the run loop.
    let mut machine = branches();
    let ns = time(&mut machine, 2, direct_round);
    std::println!("EXITS direct round_trip ns_per_exit={ns}");
    // With the step loop's profiler on, as a profile build runs.
    let mut profiled = branches();
    profiled.set_step_timing(true);
    let ns = time(&mut profiled, 2, direct_round);
    std::println!("EXITS direct round_trip profiled ns_per_exit={ns}");
    // Its parts: the schedule, the fetch, and the run with its exit.
    let ns = time(&mut machine, 1, |machine| machine.schedule().unwrap());
    std::println!("EXITS direct part=schedule ns={ns}");
    let ns = time(&mut machine, 1, |machine| {
        let _ = machine.fetch_block(BASE, false).unwrap();
    });
    std::println!("EXITS direct part=fetch ns={ns}");
    let Ok(Fetched::Block(index)) = machine.fetch_block(BASE, false) else {
        panic!("a block");
    };
    let ns = time(&mut machine, 1, |machine| {
        machine.host.exits.push(NativeExit::Exit {
            next_pc: BASE + 8,
            kind: ExitKind::Direct as u64,
        });
        let budget = machine.chain_budget();
        machine.execute(index, budget).unwrap();
    });
    std::println!("EXITS direct part=execute ns={ns}");
    // A System exit and a direct exit through the run loop, fetching at
    // once after the System exit (`continues_directly`), and with every
    // full step, as another PE's interrupt keeps it.
    for full in [false, true] {
        let mut machine = kernel_loop(MSR_TPIDR_EL2_X0, 3);
        machine.set_step_timing(false);
        if full {
            let storage = machine.chain.weak_storage().upgrade().unwrap();
            storage.interrupt_epoch().fetch_add(1, Ordering::Relaxed);
        }
        let ns = time(&mut machine, 2, |machine| {
            switch_exits(machine, 1);
            let limit = machine.statistics.blocks + 2;
            machine.run_blocks(limit);
        });
        std::println!("EXITS system round_trip full_step={full} ns_per_exit={ns}");
    }
}

/// `b .+8`, a word no run reaches, and `b .-8`: two blocks whose direct
/// exits lead to each other.
fn branches() -> Machine<FakeHost, NoDevices> {
    const FORWARD: u32 = 0x1400_0002;
    const BACK: u32 = 0x17ff_fffe;
    machine(&[FORWARD, 0, BACK])
}

/// Each block's run ending at its direct exit to the other, through the
/// whole loop step: the schedule, the fetch, the entry, the native return,
/// the exit and its link.
fn direct_round(machine: &mut Machine<FakeHost, NoDevices>) {
    for next_pc in [BASE + 8, BASE] {
        machine.host.exits.push(NativeExit::Exit {
            next_pc,
            kind: ExitKind::Direct as u64,
        });
        machine.schedule().unwrap();
        machine.step().unwrap();
        assert_eq!(machine.cpu.pc, next_pc);
    }
}

/// The round trip keeps the PC moving between the two blocks and leaves no
/// scripted exit behind.
#[test]
fn direct_rounds_alternate_between_the_blocks() {
    let mut machine = branches();
    for _ in 0..8 {
        direct_round(&mut machine);
    }
    assert!(machine.host.exits.is_empty());
    assert_eq!(machine.statistics.exit_kinds[ExitKind::Direct as usize], 16);
}

/// The parts of one transition, at EL1 after an SVC.
fn parts() -> [(&'static str, Round); 7] {
    [
        // What the engine does after a native run that made an SVC and its
        // ERET (`execute`).
        ("adopt-transitions", |machine| {
            machine.adopt_transitions([1, 1, 0, 0], 0);
            machine.host.regimes.clear();
        }),
        ("cpu-take+return", |machine| {
            let exception = crate::machine::cpu::Exception::synchronous(
                crate::machine::exception::Class::SupervisorCall,
                0,
                USER + 4,
            );
            let pstate = machine.cpu.pstate;
            machine.cpu.take(exception);
            machine.cpu.set_pstate(pstate);
        }),
        ("transitioned", |machine| {
            machine.transitioned();
            machine.host.regimes.clear();
        }),
        ("sync-host-regime", |machine| {
            machine.sync_host_regime(super::super::step_timing::SyncSource::Other);
            machine.host.regimes.clear();
        }),
        ("reopen-pointers", |machine| machine.reopen_pointers()),
        ("forget+variant", |machine| {
            machine.forget_derived();
            machine.current_variant(false, false);
        }),
        ("variant", |machine| {
            machine.current_variant(false, false);
        }),
    ]
}

/// Nanoseconds per transition of `round` (making `transitions`) when each
/// round starts after `evict` bytes of other data went through the caches,
/// as native guest code does between two exits: the median of `rounds`
/// rounds, each timed alone, so the eviction is not counted.
fn time_cold(
    machine: &mut Machine<FakeHost, NoDevices>,
    transitions: u64,
    evict: &mut [u64],
    mut round: impl FnMut(&mut Machine<FakeHost, NoDevices>),
) -> u64 {
    const ROUNDS: usize = 301;
    let mut samples = std::vec::Vec::with_capacity(ROUNDS);
    for index in 0..ROUNDS {
        // One write per line: the lines are dirty and must leave.
        for line in evict.iter_mut().step_by(8) {
            *line = line.wrapping_add(index as u64);
        }
        core::hint::black_box(&mut *evict);
        let started = std::time::Instant::now();
        round(machine);
        samples.push(started.elapsed().as_nanos() as u64);
    }
    samples.sort_unstable();
    samples[ROUNDS / 2] / transitions
}

/// `cargo test -p stela-engine --release --lib exit_transition_cold_costs
/// -- --ignored --nocapture`: [`exit_transition_costs`] with the caches
/// refilled by other data before every pair, which is how a guest's exits
/// find them: hot (none), past L2 (2 MiB), past L3 (256 MiB).
#[test]
#[ignore = "manual: opt-in"]
fn exit_transition_cold_costs() {
    for mib in [0usize, 2, 256] {
        let mut evict = std::vec![0u64; mib * 1024 * 1024 / 8];
        for host in [false, true] {
            let mut machine = user(host);
            let ns = time_cold(&mut machine, 2, &mut evict, system_call);
            std::println!("EXITS cold_mib={mib} svc+eret host={host} ns_per_transition={ns}");
        }
        let mut machine = gxf(stela_contracts::Chip::AppleM2);
        let at = machine.cpu.pc;
        let ns = time_cold(&mut machine, 2, &mut evict, |machine| {
            guarded_call(machine, at)
        });
        std::println!("EXITS cold_mib={mib} genter+gexit ns_per_transition={ns}");
    }
    // Hot code, cold data: SVC+ERET on each of `count` machines in turn,
    // so each finds its own state evicted by the others' but the engine's
    // code warm.
    for count in [1usize, 64] {
        let mut machines: std::vec::Vec<_> = (0..count).map(|_| user(true)).collect();
        let mut samples = std::vec::Vec::new();
        for pass in 0..8 {
            for machine in &mut machines {
                let started = std::time::Instant::now();
                system_call(machine);
                if pass > 0 {
                    samples.push(started.elapsed().as_nanos() as u64);
                }
            }
        }
        samples.sort_unstable();
        let ns = samples[samples.len() / 2] / 2;
        std::println!("EXITS machines={count} svc+eret host=true ns_per_transition={ns}");
    }
    // Each half alone, and each part of a transition (at EL1 after an
    // SVC), past L3.
    let mut evict = std::vec![0u64; 256 * 1024 * 1024 / 8];
    for (name, word) in [("svc", SVC), ("eret", ERET)] {
        let mut machine = user(true);
        let ns = time_cold(&mut machine, 1, &mut evict, |machine| {
            machine.cpu.pc = USER;
            exit(machine, SVC);
            if word == ERET {
                // Only the ERET is timed: the SVC before it runs warm and
                // is taken out by timing it again alone below.
            }
        });
        std::println!("EXITS cold_mib=256 {name}-alone host=true ns={ns}");
        if word == SVC {
            continue;
        }
        let mut samples = std::vec::Vec::new();
        for index in 0..301u64 {
            machine.cpu.pc = USER;
            exit(&mut machine, SVC);
            for line in evict.iter_mut().step_by(8) {
                *line = line.wrapping_add(index);
            }
            core::hint::black_box(&mut evict);
            let started = std::time::Instant::now();
            exit(&mut machine, ERET);
            samples.push(started.elapsed().as_nanos() as u64);
        }
        samples.sort_unstable();
        std::println!(
            "EXITS cold_mib=256 eret-alone host=true ns={}",
            samples[samples.len() / 2]
        );
    }
    let mut machine = user(false);
    exit(&mut machine, SVC);
    for (name, part) in parts() {
        let ns = time_cold(&mut machine, 1, &mut evict, part);
        std::println!("EXITS cold_mib=256 part={name} ns={ns}");
    }
    // Past L3, then the code warmed by another machine's SVC+ERET: what
    // is left of the cold cost is the timed machine's own data.
    let (mut warm, mut timed) = (user(true), user(true));
    let mut samples = std::vec::Vec::new();
    for index in 0..301u64 {
        for line in evict.iter_mut().step_by(8) {
            *line = line.wrapping_add(index);
        }
        core::hint::black_box(&mut evict);
        system_call(&mut warm);
        let started = std::time::Instant::now();
        system_call(&mut timed);
        samples.push(started.elapsed().as_nanos() as u64);
    }
    samples.sort_unstable();
    let ns = samples[samples.len() / 2] / 2;
    std::println!("EXITS cold_mib=256 code_warm svc+eret host=true ns_per_transition={ns}");
}

/// Nanoseconds per SVC or ERET transition of four PEs of one M2 cluster
/// running at once, the slowest PE's.
fn contended(locked: bool) -> u64 {
    use crate::machine::ctrr::Field;
    let cluster = crate::machine::shared::Shared::<crate::machine::ctrr::Cluster>::default();
    if locked {
        for level in 0..2 {
            cluster.borrow_mut().write(level, Field::Lock, 1).unwrap();
        }
    }
    let start = std::sync::Barrier::new(4);
    std::thread::scope(|scope| {
        let runs: std::vec::Vec<_> = (0..4)
            .map(|_| {
                let (cluster, start) = (cluster.clone(), &start);
                scope.spawn(move || {
                    let mut machine = user_on(stela_contracts::Chip::AppleM2, true);
                    machine.cpu.ctrr = cluster;
                    start.wait();
                    time(&mut machine, 2, system_call)
                })
            })
            .collect();
        runs.into_iter()
            .map(|run| run.join().unwrap())
            .max()
            .unwrap()
    })
}

/// `MSR TPIDR_EL1, X0`, then `b .-4`: a System exit and a direct exit back.
fn thread_switches() -> Machine<FakeHost, NoDevices> {
    const MSR_TPIDR_EL1_X0: u32 = 0xd518_d080;
    const BACK: u32 = 0x17ff_ffff;
    let mut machine = machine(&[MSR_TPIDR_EL1_X0, BACK]);
    machine.cpu.set_pstate(Pstate::reset(1));
    machine.host.counter_step = 1;
    machine.set_step_timing(true);
    machine
}

/// The scripted exits of `rounds` of [`thread_switches`]: the System exit
/// at the MSR, then the branch's.
fn switch_exits(machine: &mut Machine<FakeHost, NoDevices>, rounds: usize) {
    for _ in 0..rounds {
        for kind in [ExitKind::System, ExitKind::Direct] {
            machine.host.exits.push(NativeExit::Exit {
                next_pc: BASE,
                kind: kind as u64,
            });
        }
    }
}

/// Under step timing one System exit in `PATH_EVERY` is traced through the
/// run loop, point by point, to the next native entry: each segment counts
/// once and their sum is the round trip. A trace still open when the run
/// loop is left counts as abandoned, and nothing is traced without step
/// timing.
#[test]
fn system_exit_round_trips_are_traced_through_the_run_loop() {
    let mut machine = thread_switches();
    let rounds = step_timing::PATH_EVERY as usize + 1;
    switch_exits(&mut machine, rounds);
    let stop = machine.run_blocks(2 * rounds as u64);
    assert!(matches!(stop, Stop::Budget), "{stop:?}");
    assert!(machine.host.exits.is_empty());
    let paths = machine.take_system_exits().paths;
    assert_eq!((paths.traced, paths.abandoned), (2, 0), "{paths:?}");
    // The fake clock moves at every read, so every segment is short and the
    // service, which reads it, is not empty.
    assert!(paths.ticks.iter().all(|&ticks| ticks < 1000), "{paths:?}");
    assert!(paths.ticks[step_timing::PATH_SERVICE] > 0, "{paths:?}");
    // Left right after the exit, the trace is abandoned.
    let mut machine = thread_switches();
    switch_exits(&mut machine, 1);
    machine.run_blocks(1);
    let paths = machine.take_system_exits().paths;
    assert_eq!((paths.traced, paths.abandoned), (0, 1), "{paths:?}");
    // Without step timing nothing is stamped.
    let mut machine = thread_switches();
    machine.set_step_timing(false);
    switch_exits(&mut machine, 1);
    machine.run_blocks(2);
    assert_eq!(
        machine.take_system_exits().paths,
        step_timing::ExitPaths::default()
    );
}

/// `MSR TPIDR_EL2, X0`.
const MSR_TPIDR_EL2_X0: u32 = 0xd51c_d040;

/// `first` then `b .-4` at EL2 under step timing, with X0 `x0`: the System
/// exit at `first`, then the branch's direct exit back.
fn kernel_loop(first: u32, x0: u64) -> Machine<FakeHost, NoDevices> {
    const BACK: u32 = 0x17ff_ffff;
    let mut machine = machine(&[first, BACK]);
    machine.cpu.set_pstate(Pstate::reset(2));
    machine.cpu.state.x[0] = x0;
    machine.host.counter_step = 1;
    machine.set_step_timing(true);
    machine
}

/// The run loop for `blocks` blocks, or a few thousand host counter
/// ticks: a run that went astray (an exception loop runs no block) stops.
fn bounded_run(machine: &mut Machine<FakeHost, NoDevices>, blocks: u64) -> Stop {
    machine.run_blocks_with_budget(RunBudget {
        blocks,
        host_ticks: Some(100_000),
    })
}

/// `rounds` of [`kernel_loop`] in the run loop: the fetches at once and
/// the architectural state the runs end in.
fn kernel_rounds(
    machine: &mut Machine<FakeHost, NoDevices>,
    rounds: usize,
) -> (u64, (u64, Pstate, u64, u64, u64)) {
    switch_exits(machine, rounds);
    let stop = bounded_run(machine, 2 * rounds as u64);
    assert!(matches!(stop, Stop::Budget), "{stop:?}");
    assert!(machine.host.exits.is_empty());
    let direct = machine.take_system_exits().paths.direct;
    let state = (
        machine.cpu.pc,
        machine.cpu.pstate,
        machine.cpu.retired,
        machine.statistics.blocks,
        machine.cpu.read(SystemRegister::TPIDR_EL2),
    );
    (direct, state)
}

/// After a System exit that leaves PSTATE, the device lines and the timer
/// deadline as the last full step found them, the run loop fetches at
/// once, and a traced one's schedule and devices take no time.
#[test]
fn plain_system_exits_fetch_at_once() {
    let mut machine = kernel_loop(MSR_TPIDR_EL2_X0, 0x1234);
    let rounds = step_timing::PATH_EVERY as usize + 1;
    switch_exits(&mut machine, rounds);
    let stop = bounded_run(&mut machine, 2 * rounds as u64);
    assert!(matches!(stop, Stop::Budget), "{stop:?}");
    assert!(machine.host.exits.is_empty());
    assert_eq!(machine.cpu.read(SystemRegister::TPIDR_EL2), 0x1234);
    let paths = machine.take_system_exits().paths;
    assert_eq!(paths.direct, rounds as u64, "{paths:?}");
    assert_eq!((paths.traced, paths.abandoned), (2, 0), "{paths:?}");
    for name in ["schedule", "devices"] {
        let segment = step_timing::PATH_NAMES.iter().position(|&n| n == name);
        assert_eq!(paths.ticks[segment.unwrap()], 0, "{name} {paths:?}");
    }
}

/// Each source a full step reads stops the fetch at once: an interrupt
/// epoch another PE moved, an EL whose lines no count bounds, a write of
/// PSTATE, and one of a timer; the same write again changes nothing, and
/// fetches at once. A run that fetches at once ends in the architectural
/// state of one that takes every full step.
#[test]
fn system_exits_that_change_what_a_step_reads_take_the_full_step() {
    const MSR_DAIFCLR_I: u32 = 0xd503_42ff;
    const MSR_CNTV_CTL_EL0_X0: u32 = 0xd51b_e320;
    const ROUNDS: usize = 4;
    let (direct, plain) = kernel_rounds(&mut kernel_loop(MSR_TPIDR_EL2_X0, 3), ROUNDS);
    assert_eq!(direct, ROUNDS as u64);
    let mut moved = kernel_loop(MSR_TPIDR_EL2_X0, 3);
    let storage = moved.chain.weak_storage().upgrade().unwrap();
    storage.interrupt_epoch().fetch_add(1, Ordering::Relaxed);
    assert_eq!(kernel_rounds(&mut moved, ROUNDS), (0, plain));
    const MSR_TPIDR_EL1_X0: u32 = 0xd518_d080;
    let mut kernel = kernel_loop(MSR_TPIDR_EL1_X0, 3);
    kernel.cpu.set_pstate(Pstate::reset(1));
    let (direct, _) = kernel_rounds(&mut kernel, ROUNDS);
    assert_eq!(direct, 0, "EL1's lines hold until no count");
    for word in [MSR_DAIFCLR_I, MSR_CNTV_CTL_EL0_X0] {
        let (direct, _) = kernel_rounds(&mut kernel_loop(word, 3), ROUNDS);
        assert_eq!(direct, ROUNDS as u64 - 1, "{word:#010x}");
    }
}
