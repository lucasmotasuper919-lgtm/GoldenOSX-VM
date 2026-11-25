//! Which host instructions real blocks execute, run by hand: each block of
//! the sample `DENSITY_IN` names (lines of hex PC, weight and hex bytes) is
//! compiled as a system-mode block and run once in a child process under
//! the x86 trap flag, which records every host instruction it executes.
//! Guest registers point into a scratch arena whose words point back into
//! it, so loads and pointer chasing stay mapped; the anchors point at zeroed
//! areas with the most chain budget, PAC instructions run the cipher
//! inline where the host has AES-NI, as the engine's blocks do, and the
//! pointer-authentication routine returns its input. `DENSITY_OUT` gets, per block, a `B` line
//! (PC, weight, exit, host instructions executed inline, out of line and
//! outside the block, the loop's count, the instructions retired and
//! whether it is a region) and per guest instruction an `S` line (its word, how
//! often its code was entered, host instructions inline, and those out of
//! line that it branched to), then a `C` line (its code, source map, inline
//! end and how often each host instruction ran). Inline is the block's code
//! up to its last instruction's first unconditional jump or return; the
//! rest (side-exit stubs, cold paths, loop copies) is out of line. At the end
//! `T` lines give the weighted host instructions per guest instruction by
//! cause ([`census`]), also printed.
use crate::{
    GuestAddress, GuestState, Options, TargetCpu,
    block::{BlockProfile, CompilationPlan, Features},
    x86_64::{MemoryChecks, NativeSystem, RangeCheck, pointer},
};
use alloc::{format, string::String, vec, vec::Vec};
use core::ffi::c_void;
use core::sync::atomic::{AtomicUsize, Ordering};

mod census;

const SIGTRAP: i32 = 5;
const SA_SIGINFO: i32 = 4;
const TRAP_FLAG: u64 = 0x100;
/// `uc_mcontext.gregs` in `ucontext_t`, and REG_RIP / REG_EFL in it.
const GREGS: usize = 40;
const RIP: usize = 16;
const EFLAGS: usize = 17;
const STEPS: usize = 1 << 20;
const OUTSIDE: u32 = u32::MAX;
/// Where the pointer anchor's area holds the cipher's schedules and ranges
/// (empty PAC fields: every pointer is canonical, every authentication
/// succeeds and every result is its pointer).
const CIPHER: u32 = 0x400;

#[repr(C)]
struct SigAction {
    handler: usize,
    mask: [u64; 16],
    flags: i32,
    restorer: usize,
}

unsafe extern "C" {
    fn sigaction(signal: i32, action: *const SigAction, old: *mut SigAction) -> i32;
    fn fork() -> i32;
    fn pipe(fds: *mut i32) -> i32;
    fn read(fd: i32, bytes: *mut c_void, count: usize) -> isize;
    fn write(fd: i32, bytes: *const c_void, count: usize) -> isize;
    fn close(fd: i32) -> i32;
    fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
    fn alarm(seconds: u32) -> u32;
    fn _exit(status: i32) -> !;
    fn mmap(
        address: *mut c_void,
        bytes: usize,
        protection: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut c_void;
    fn mprotect(address: *mut c_void, bytes: usize, protection: i32) -> i32;
}

static BASE: AtomicUsize = AtomicUsize::new(0);
static LENGTH: AtomicUsize = AtomicUsize::new(0);
static COUNT: AtomicUsize = AtomicUsize::new(0);
static mut TRACE: [u32; STEPS] = [0; STEPS];

unsafe extern "C" fn step(_: i32, _: *mut c_void, context: *mut c_void) {
    // SAFETY: Linux passes a ucontext_t; gregs holds the interrupted RIP and
    // EFLAGS. The child is single threaded.
    unsafe {
        let gregs = context.cast::<u8>().add(GREGS).cast::<u64>();
        let rip = *gregs.add(RIP) as usize;
        let base = BASE.load(Ordering::Relaxed);
        let count = COUNT.load(Ordering::Relaxed);
        if count >= STEPS {
            *gregs.add(EFLAGS) &= !TRAP_FLAG;
            return;
        }
        let offset = if (base..base + LENGTH.load(Ordering::Relaxed)).contains(&rip) {
            (rip - base) as u32
        } else {
            OUTSIDE
        };
        core::ptr::addr_of_mut!(TRACE)
            .cast::<u32>()
            .add(count)
            .write(offset);
        COUNT.store(count + 1, Ordering::Relaxed);
    }
}

#[repr(C)]
struct Pair(u64, u64);

/// The pointer-authentication routine: success, with its input.
extern "sysv64" fn authenticated(_: *mut u8, _: u32, value: u64, _: u64) -> Pair {
    Pair(1, value)
}

fn options() -> Options {
    // As the engine's blocks; `REGIONS=0` compiles blocks alone.
    options_with(std::env::var("REGIONS").map_or(true, |regions| regions != "0"))
}

fn options_with(regions: bool) -> Options {
    let checks = MemoryChecks::system(true, false, true, false)
        .with_range(RangeCheck::four_level(4).one_sided(true));
    let system = NativeSystem {
        exclusive: true,
        pointer: Some(pointer::inline::Cells {
            tags: 0x40,
            strips: 0x80,
            routine: 0x1c0,
            // As the engine's blocks on an AES-NI host: the cipher inline.
            cipher: TargetCpu::detect()
                .supports(TargetCpu::Aes)
                .then_some(CIPHER),
        }),
        self_loops: true,
        regions,
        ..NativeSystem::NONE
    };
    Options::new(
        BlockProfile::new(Features::SYSTEM_MODE),
        TargetCpu::detect(),
    )
    .with_memory_checks(checks)
    .with_native_system(system)
}

fn map(bytes: usize) -> *mut u8 {
    // SAFETY: private anonymous read-write memory.
    let page = unsafe { mmap(core::ptr::null_mut(), bytes, 3, 0x22, -1, 0) };
    assert_ne!(page as usize, usize::MAX);
    page.cast()
}

/// Run `entry` (the pinned entry wrapper and the block) in a child under
/// the trap flag; the parent gets the trace and the exit.
fn traced(entry: &[u8], body_offset: usize) -> Option<(u64, u64, u64, Vec<u32>)> {
    let mut fds = [0i32; 2];
    // SAFETY: plain POSIX calls; the child only runs the block and writes.
    unsafe {
        assert_eq!(pipe(fds.as_mut_ptr()), 0);
        let pid = fork();
        if pid == 0 {
            close(fds[0]);
            alarm(5);
            let code = map(entry.len().next_multiple_of(4096) + 4096);
            core::ptr::copy_nonoverlapping(entry.as_ptr(), code, entry.len());
            code.add(entry.len()).write(0xc3);
            mprotect(code.cast(), entry.len().next_multiple_of(4096) + 4096, 5);
            BASE.store(code as usize + body_offset, Ordering::Relaxed);
            LENGTH.store(entry.len() - body_offset, Ordering::Relaxed);
            let action = SigAction {
                handler: step as unsafe extern "C" fn(i32, *mut c_void, *mut c_void) as usize,
                mask: [0; 16],
                flags: SA_SIGINFO,
                restorer: 0,
            };
            sigaction(SIGTRAP, &action, core::ptr::null_mut());
            let state = alloc::boxed::Box::leak(alloc::boxed::Box::new(GuestState::default()));
            prepare(state);
            // The exit's target and word, in RAX and RDX.
            let run: unsafe extern "sysv64" fn(*mut GuestState) -> Pair =
                core::mem::transmute(code);
            core::arch::asm!("pushfq", "or qword ptr [rsp], 0x100", "popfq");
            let Pair(exit, word) = run(state);
            core::arch::asm!("pushfq", "and qword ptr [rsp], -257", "popfq");
            let count = COUNT.load(Ordering::Relaxed);
            let trace =
                core::slice::from_raw_parts(core::ptr::addr_of!(TRACE).cast::<u32>(), count);
            write(fds[1], (&raw const exit).cast(), 8);
            write(fds[1], (&raw const word).cast(), 8);
            // The instructions the chain's account counts and the loop's.
            let account = crate::x86_64::chain::Account(state.account);
            let retired = account.retired().wrapping_add(state.loop_retired);
            write(fds[1], (&raw const retired).cast(), 8);
            write(fds[1], (&raw const count).cast(), 8);
            let mut sent = 0;
            while sent < count * 4 {
                let n = write(
                    fds[1],
                    trace.as_ptr().cast::<u8>().add(sent).cast(),
                    count * 4 - sent,
                );
                if n <= 0 {
                    break;
                }
                sent += n as usize;
            }
            _exit(0);
        }
        close(fds[1]);
        let mut header = [0u64; 3];
        let mut bytes = Vec::new();
        let mut buffer = [0u8; 65536];
        loop {
            let n = read(fds[0], buffer.as_mut_ptr().cast(), buffer.len());
            if n <= 0 {
                break;
            }
            bytes.extend_from_slice(&buffer[..n as usize]);
        }
        close(fds[0]);
        let mut status = 0;
        waitpid(pid, &mut status, 0);
        if status != 0 || bytes.len() < 32 {
            return None;
        }
        header[0] = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        header[1] = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        header[2] = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let trace = bytes[32..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect();
        Some((header[0], header[1], header[2], trace))
    }
}

/// Registers and memory for a run: an 8 MiB arena whose words point into
/// its middle, registers spread over it, IXC set (the usual running
/// state), zeroed anchor areas and the most chain budget.
fn prepare(state: &mut GuestState) {
    const ARENA: usize = 8 << 20;
    let arena = map(ARENA);
    let middle = arena as u64 + (ARENA as u64) / 2;
    // SAFETY: the arena is mapped read-write.
    unsafe {
        let words = arena.cast::<u64>();
        for index in 0..ARENA / 8 {
            *words.add(index) = middle + ((index as u64 * 8) & 0xffff);
        }
    }
    for (index, register) in state.x.iter_mut().enumerate() {
        *register = middle + 0x100 * index as u64;
    }
    state.sp = middle + 0x10_0000;
    state.fpsr = 0x10;
    let anchors = map(8 << 16);
    let fault = map(4096);
    state.unreachable = fault as u64;
    state.lower_limit = crate::Limit(u64::MAX);
    state.account = super::chain::Account::new(super::chain::MAX_BUDGET).0;
    let raw = (state as *mut GuestState).cast::<u8>();
    let anchor_offset = core::mem::offset_of!(GuestState, anchors);
    // SAFETY: the anchors are eight words in the state; each area is mapped.
    unsafe {
        for index in 0..8 {
            let area = anchors.add(index << 16);
            raw.add(anchor_offset + 8 * index)
                .cast::<u64>()
                .write(area as u64);
        }
        // Open tag words for the cipher, and the pointer routine.
        for key in 0..pointer::TAG_WORDS {
            anchors
                .add(3 << 16)
                .add(0x40 + 8 * key)
                .cast::<u64>()
                .write(pointer::tag(1, key as u8, 0));
        }
        let ranges = [[false, true].map(|upper| pointer::cipher::Range::new(0, upper, false)); 2];
        anchors
            .add(3 << 16)
            .add(CIPHER as usize + core::mem::offset_of!(pointer::cipher::Cipher, ranges))
            .cast::<[[pointer::cipher::Range; 2]; 2]>()
            .write(ranges);
        anchors.add(3 << 16).add(0x1c0).cast::<u64>().write(
            (authenticated as extern "sysv64" fn(*mut u8, u32, u64, u64) -> Pair) as usize as u64,
        );
    }
}

#[test]
#[ignore = "manual: opt-in"]
fn executed_paths_of_real_blocks() {
    use std::io::Write;
    let input = std::env::var("DENSITY_IN").expect("DENSITY_IN names the sample");
    let output = std::env::var("DENSITY_OUT").expect("DENSITY_OUT names the output");
    let mut out = std::fs::File::create(output).expect("the output");
    let options = options();
    let mut causes = census::Census::default();
    // The causes in the blocks that compile as regions when regions are on,
    // in this run's mode: the same code before and after.
    let with_regions = options_with(true);
    let mut region_causes = census::Census::default();
    // Weighted guest and host instructions of all blocks and of regions
    // (`CompiledBlock::region`), and the weight of each.
    let mut shares = [[0_f64; 3]; 2];
    for line in std::fs::read_to_string(input).expect("the sample").lines() {
        let mut fields = line.split(' ');
        let (Some(pc), Some(weight), Some(hex)) = (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        let pc = u64::from_str_radix(pc, 16).unwrap();
        let source: Vec<u8> = (0..hex.len() / 2)
            .map(|at| u8::from_str_radix(&hex[2 * at..2 * at + 2], 16).unwrap())
            .collect();
        let Ok(plan) = CompilationPlan::new(&source, GuestAddress(pc), &options) else {
            continue;
        };
        let Ok(compiled) = plan.compile() else {
            continue;
        };
        let shaped = compiled.region
            || CompilationPlan::new(&source, GuestAddress(pc), &with_regions)
                .ok()
                .and_then(|plan| plan.compile().ok())
                .is_some_and(|compiled| compiled.region);
        let body = &compiled.native.x86_64_code;
        let entry = crate::x86_64::pinned::entered(body);
        let body_offset = entry.len() - body.len();
        let map = &compiled.native.source_map;
        if map.is_empty() {
            continue;
        }
        // Inline: to the last instruction's first unconditional jump or ret.
        let last = map[map.len() - 1].1;
        let inline_end = inline_end(body, last);
        let Some((exit, word, looped, trace)) = traced(&entry, body_offset) else {
            writeln!(out, "X {pc:x} {weight}").unwrap();
            continue;
        };
        let span_of = |offset: u32| {
            map.iter()
                .rposition(|&(_, start)| start as u32 <= offset)
                .unwrap_or(0)
        };
        let mut entered = vec![0u32; map.len()];
        let mut inline = vec![0u32; map.len()];
        let mut cold = vec![0u32; map.len()];
        let (mut inside, mut outside, mut out_of_line) = (0u32, 0u32, 0u32);
        let mut current = usize::MAX;
        let first = trace
            .iter()
            .position(|&o| o != OUTSIDE)
            .unwrap_or(trace.len());
        let end = trace
            .iter()
            .rposition(|&o| o != OUTSIDE)
            .map_or(first, |at| at + 1);
        for &offset in &trace[first..end] {
            if offset == OUTSIDE {
                outside += 1;
                continue;
            }
            if (offset as usize) < inline_end {
                inside += 1;
                let span = span_of(offset);
                if span != current {
                    entered[span] += 1;
                    current = span;
                }
                inline[span] += 1;
            } else {
                out_of_line += 1;
                if current < map.len() {
                    cold[current] += 1;
                }
            }
        }
        // The instructions retired: what the exit counts (a branch's the
        // rows it did not skip, any other the ones before it) and what the
        // chain and the loop counted besides.
        let (kind, skipped) = crate::x86_64::chain::ExitWord::split(word);
        use crate::block::ExitKind;
        let counted = match ExitKind::try_from(kind) {
            Ok(
                ExitKind::Fallthrough
                | ExitKind::Direct
                | ExitKind::Indirect
                | ExitKind::Call
                | ExitKind::Return,
            ) => (map.len() as u64).saturating_sub(u64::from(skipped)),
            _ => exit.wrapping_sub(pc) / 4,
        };
        let guest = looped.wrapping_add(counted);
        writeln!(
            out,
            "B {pc:x} {weight} {exit:x} {inside} {out_of_line} {outside} {looped} {guest} {}",
            u8::from(compiled.region)
        )
        .unwrap();
        for (index, &(guest, _)) in map.iter().enumerate() {
            let at = (guest.0 - pc) as usize;
            let word = u32::from_le_bytes(source[at..at + 4].try_into().unwrap());
            writeln!(
                out,
                "S {word:08x} {} {} {}",
                entered[index], inline[index], cold[index]
            )
            .unwrap();
        }
        // The code, its source map and how often each host instruction ran,
        // for a per-cause census of the executed instructions.
        let mut runs = alloc::collections::BTreeMap::<u32, u32>::new();
        for &offset in &trace[first..end] {
            if offset != OUTSIDE {
                *runs.entry(offset).or_default() += 1;
            }
        }
        let code: String = body.iter().map(|byte| format!("{byte:02x}")).collect();
        let spans: Vec<String> = map
            .iter()
            .map(|(guest, offset)| format!("{:x}:{offset}", guest.0))
            .collect();
        let counts: Vec<String> = runs
            .iter()
            .map(|(offset, count)| format!("{offset:x}:{count}"))
            .collect();
        writeln!(
            out,
            "C {code} {} {inline_end:x} {}",
            spans.join(","),
            counts.join(",")
        )
        .unwrap();
        let weighted: f64 = weight.parse().unwrap_or(1.0);
        for (index, share) in shares.iter_mut().enumerate() {
            if index == 0 || shaped {
                share[0] += weighted * guest as f64;
                share[1] += weighted * f64::from(inside + out_of_line);
                share[2] += weighted;
            }
        }
        let traced = census::Traced {
            weight: weight.parse().unwrap_or(1),
            guest,
            pc,
            source: &source,
            code: body,
            map,
            inline_end,
            runs: &runs,
        };
        causes.add(&traced);
        if shaped {
            region_causes.add(&traced);
        }
    }
    let [all, regions] = shares;
    let table = format!(
        "{}{}R regions weight={:.1}% guest={:.1}% host_per_guest={:.3} (all {:.3})\n",
        causes.table(),
        region_causes.table().replace("T ", "RT "),
        100.0 * regions[2] / all[2].max(1.0),
        100.0 * regions[0] / all[0].max(1.0),
        regions[1] / regions[0].max(1.0),
        all[1] / all[0].max(1.0),
    );
    write!(out, "{table}").unwrap();
    std::print!("{table}");
}

/// The end of a block's inline code: past the first unconditional jump or
/// return at or after `last` (the last instruction's start); the whole code
/// when there is none.
fn inline_end(body: &[u8], last: usize) -> usize {
    // Decoded by objdump: exact for every form the emitter uses.
    let listing = census::listing(&body[last..]);
    let mut lines = listing.iter().peekable();
    while let Some((_, text)) = lines.next() {
        if text.starts_with("jmp") || text.starts_with("ret") {
            return last + lines.peek().map_or(body.len() - last, |(next, _)| *next);
        }
    }
    body.len()
}

/// What the value allocator would save system-mode blocks, run by hand:
/// each block of `DENSITY_IN` compiled as a system-mode block (pinned set,
/// fault table, no regions) into `DENSITY_OUT`, and as a process block
/// (regions and the value allocator over all twelve hosts, no pinned set
/// or fault table) into `DENSITY_OUT.process`, in `density_dump`'s format
/// (`pc weight source-bytes hex-code pc:offset,...`). The process code
/// bounds what an unconstrained allocator (variant C, before recovery
/// rows) can reach on the same instructions.
#[test]
#[ignore = "manual: opt-in"]
fn allocator_bound_census() {
    use std::io::Write;
    let input = std::env::var("DENSITY_IN").expect("DENSITY_IN names the sample");
    let output = std::env::var("DENSITY_OUT").expect("DENSITY_OUT names the output");
    let process = Options::new(
        BlockProfile::new(Features::POINTER_AUTHENTICATION),
        TargetCpu::detect(),
    );
    for (options, path) in [(options(), output.clone()), (process, output + ".process")] {
        let mut out = std::fs::File::create(path).expect("the output");
        for line in std::fs::read_to_string(&input).expect("the sample").lines() {
            let mut fields = line.split(' ');
            let (Some(pc), Some(weight), Some(hex)) = (fields.next(), fields.next(), fields.next())
            else {
                continue;
            };
            let pc = u64::from_str_radix(pc, 16).unwrap();
            let source: Vec<u8> = (0..hex.len() / 2)
                .map(|at| u8::from_str_radix(&hex[2 * at..2 * at + 2], 16).unwrap())
                .collect();
            let compiled = CompilationPlan::new(&source, GuestAddress(pc), &options)
                .ok()
                .and_then(|plan| plan.compile().ok());
            let Some(compiled) = compiled else {
                writeln!(out, "{pc:x} {weight} refused").unwrap();
                continue;
            };
            let code: String = compiled
                .native
                .x86_64_code
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            let map: Vec<String> = compiled
                .native
                .source_map
                .iter()
                .map(|(pc, offset)| format!("{:x}:{offset}", pc.0))
                .collect();
            writeln!(
                out,
                "{pc:x} {weight} {} {code} {}",
                compiled.source_bytes,
                map.join(",")
            )
            .unwrap();
        }
    }
}
