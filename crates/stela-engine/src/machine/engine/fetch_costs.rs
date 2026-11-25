//! Where a block's fetch time goes, run by hand over real guest code: the
//! phases the engine runs for a block it compiles anew (window admission,
//! the short first window, decode and analysis, lowering and encoding, the
//! shared emission, the stored form) and for one it installs from the
//! store (relocation), in nanoseconds per guest instruction translated.
//! With `FETCH_POLLUTE` set, a sweep through 1 MiB of host code runs
//! between batches of `FETCH_BATCH` blocks, as guest code runs between
//! fetches, so the translator starts each batch with cold caches. An
//! ignored test: `cargo test` never runs it.
use super::prepare::{Preparation, window};
use super::stored::Emission;
use super::{FETCH_PAGE, FIRST_WINDOW_BYTES};
use crate::translations::{Entry, Relocated};
use alloc::{sync::Arc, vec::Vec};
use std::time::{Duration, Instant};
use stela_compiler::{GuestAddress, block};
use stela_crypto::Sha256;
use stela_crypto::digest::Digest as _;

/// The memory checks and native system features a boot's kernel PEs compile
/// with (as the store's `compile_bench`): range checks, exclusives, pointer
/// cells and self loops.
fn system_inputs() -> (
    stela_compiler::x86_64::MemoryChecks,
    stela_compiler::x86_64::NativeSystem,
) {
    use stela_compiler::x86_64::{MemoryChecks, NativeSystem, RangeCheck};
    let checks = MemoryChecks::system(true, false, true, false)
        .with_range(RangeCheck::four_level(4).one_sided(true));
    let system = NativeSystem {
        exclusive: true,
        pointer: Some(super::pointer::CELLS),
        self_loops: true,
        ..NativeSystem::NONE
    };
    (checks, system)
}

pub(super) fn system_preparation() -> Preparation {
    let chip = crate::machine::test_chip();
    let (checks, system) = system_inputs();
    Preparation::new(
        chip,
        Arc::new(chip.admit_table(0).expect("admit")),
        stela_compiler::TargetCpu::detect(),
        true,
        checks,
        None,
        system,
    )
}

/// What [`system_preparation`] compiles blocks with.
fn system_options() -> stela_compiler::Options {
    let (checks, system) = system_inputs();
    stela_compiler::Options::new(super::PROFILE, stela_compiler::TargetCpu::detect())
        .with_memory_checks(checks)
        .with_native_system(system)
}

/// 1 MiB of host code: a jump to the next 64-byte line in each line, and a
/// return at the end. Each call touches every line once.
struct Polluter {
    code: *mut u8,
}

const POLLUTER_BYTES: usize = 1 << 20;

impl Polluter {
    fn new() -> Self {
        // SAFETY: a private anonymous mapping, written before it is run.
        let code = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                POLLUTER_BYTES,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(code, libc::MAP_FAILED, "an executable mapping");
        let code = code.cast::<u8>();
        let lines = POLLUTER_BYTES / 64;
        for line in 0..lines {
            // SAFETY: within the mapping.
            unsafe {
                let at = code.add(line * 64);
                if line + 1 == lines {
                    *at = 0xc3;
                } else {
                    // jmp rel8 +62: to the next line.
                    *at = 0xeb;
                    *at.add(1) = 62;
                }
            }
        }
        Self { code }
    }

    fn run(&self) {
        // SAFETY: the mapping holds code that returns.
        let entry: extern "C" fn() = unsafe { core::mem::transmute(self.code) };
        entry();
    }
}

/// A sampling profile of this process's CPU time: every 50 us of it, the
/// interrupted instruction's address, for `nm` to name offline
/// (`FETCH_PROFILE` names the file: one hex address a line, less the
/// binary's load address).
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
pub(super) mod sampler {
    use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    const SAMPLES: usize = 1 << 20;
    static TAKEN: AtomicUsize = AtomicUsize::new(0);
    static ADDRESSES: [AtomicU64; SAMPLES] = [const { AtomicU64::new(0) }; SAMPLES];

    extern "C" fn take(_: libc::c_int, _: *mut libc::siginfo_t, context: *mut libc::c_void) {
        // SAFETY: the kernel passes the interrupted context.
        let rip = unsafe {
            (*context.cast::<libc::ucontext_t>()).uc_mcontext.gregs[libc::REG_RIP as usize]
        };
        let at = TAKEN.fetch_add(1, Ordering::Relaxed);
        if at < SAMPLES {
            ADDRESSES[at].store(rip as u64, Ordering::Relaxed);
        }
    }

    pub fn start() {
        // SAFETY: a handler that only stores to statics, and a timer of this
        // process's CPU time.
        unsafe {
            let mut action: libc::sigaction = core::mem::zeroed();
            action.sa_sigaction = take as *const () as usize;
            action.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART;
            libc::sigaction(libc::SIGPROF, &action, core::ptr::null_mut());
            let interval = libc::timeval {
                tv_sec: 0,
                tv_usec: 50,
            };
            let timer = libc::itimerval {
                it_interval: interval,
                it_value: interval,
            };
            libc::setitimer(libc::ITIMER_PROF, &timer, core::ptr::null_mut());
        }
    }

    pub fn stop(path: &str) {
        // SAFETY: a zero timer stops it.
        unsafe {
            let timer: libc::itimerval = core::mem::zeroed();
            libc::setitimer(libc::ITIMER_PROF, &timer, core::ptr::null_mut());
        }
        let maps = std::fs::read_to_string("/proc/self/maps").expect("maps");
        let exe = std::fs::read_link("/proc/self/exe").expect("exe");
        let base = maps
            .lines()
            .find(|line| line.ends_with(exe.to_str().expect("a path")))
            .and_then(|line| line.split('-').next())
            .map(|start| u64::from_str_radix(start, 16).expect("hex"))
            .expect("the binary's mapping");
        let taken = TAKEN.load(Ordering::Relaxed).min(SAMPLES);
        let mut text = std::format!("base {base:x}\n");
        // The other executable mappings, for addresses outside the binary.
        for line in maps.lines().filter(|line| line.contains(" r-xp ")) {
            text += &std::format!("map {line}\n");
        }
        for address in &ADDRESSES[..taken] {
            text += &std::format!("{:x}\n", address.load(Ordering::Relaxed).wrapping_sub(base));
        }
        std::fs::write(path, text).expect("the profile");
    }
}

#[derive(Default)]
struct Phases {
    admit: Duration,
    first: Duration,
    decode: Duration,
    emit: Duration,
    emission: Duration,
    stored: Duration,
    relocate: Duration,
    blocks: u64,
    instructions: u64,
    /// Blocks whose first window did not hold them, analysed twice.
    redone: u64,
    refused: u64,
    /// With `FETCH_DIGEST`: the SHA-256 of every block compiled (its PC and
    /// all of `CompiledBlock`) and every refusal, in order, so that a
    /// change of the compiler that must emit the same bytes is checked over
    /// real code.
    digest: Option<Sha256>,
}

/// Compile every block of `image` as the engine compiles one it fetches
/// anew, timing each phase.
fn measure(image: &[u8], base: u64, batch: usize, polluter: Option<&Polluter>) -> Phases {
    let preparation = Arc::new(system_preparation());
    let options = system_options();
    let window_bytes = first_window();
    let mut phases = Phases {
        digest: std::env::var("FETCH_DIGEST").is_ok().then(Sha256::new),
        ..Phases::default()
    };
    let mut relocated = Relocated::default();
    let mut at = 0usize;
    let mut in_batch = 0usize;
    let mut staging = [0u8; super::MAX_BLOCK_BYTES as usize];
    while at + 4 <= image.len() {
        if let Some(polluter) = polluter
            && in_batch == 0
        {
            polluter.run();
        }
        in_batch = (in_batch + 1) % batch.max(1);
        let pc = base + at as u64;
        let page = pc & !(FETCH_PAGE - 1);
        let page_end = (at as u64 | (FETCH_PAGE - 1)) + 1;
        let end = (page_end as usize).min(image.len()).min(at + window(pc));
        let length = end - at;
        // As `compile_window`: the first 32 instructions admitted and
        // analysed, and the whole window only when the block runs past
        // them, the first analysis then wasted.
        let first = length.min(window_bytes);
        let started = Instant::now();
        staging[..first].copy_from_slice(&image[at..at + first]);
        preparation.admit(&mut staging[..first]);
        let decoding = Instant::now();
        phases.admit += decoding - started;
        let mut analyzed = block::analyze_and_plan(&staging[..first], GuestAddress(pc), &options);
        let mut emitting = Instant::now();
        phases.decode += emitting - decoding;
        if first < length
            && analyzed
                .as_ref()
                .is_ok_and(|(analysis, _)| analysis.source_bytes == first)
        {
            phases.redone += 1;
            phases.first += emitting - decoding;
            let admitting = Instant::now();
            staging[..length].copy_from_slice(&image[at..end]);
            preparation.admit(&mut staging[..length]);
            let decoding = Instant::now();
            phases.admit += decoding - admitting;
            analyzed = block::analyze_and_plan(&staging[..length], GuestAddress(pc), &options);
            emitting = Instant::now();
            phases.decode += emitting - decoding;
        }
        let source = &staging[..length];
        let Ok((analysis, Ok(plan))) = analyzed else {
            phases.refused += 1;
            if let Some(digest) = &mut phases.digest {
                digest.update(std::format!("refused {pc:x}").as_bytes());
            }
            at += 4;
            continue;
        };
        let compiled = plan.compile();
        phases.emit += emitting.elapsed();
        if let Some(digest) = &mut phases.digest {
            digest.update(std::format!("{pc:x} {compiled:?}").as_bytes());
        }
        let Ok(compiled) = compiled else {
            phases.refused += 1;
            at += 4;
            continue;
        };
        let input = analysis.source_bytes;
        let source_bytes = compiled.source_bytes;
        phases.blocks += 1;
        phases.instructions += source_bytes as u64 / 4;
        let making = Instant::now();
        let emission = Emission::new(source[..input].to_vec(), compiled, preparation.clone());
        let storing = Instant::now();
        phases.emission += storing - making;
        let context = preparation.context(page);
        let entry = Entry::stored(&emission.block, &emission.input, page, pc, context);
        let relocating = Instant::now();
        phases.stored += relocating - storing;
        if let Some(entry) = entry {
            let moved = page.wrapping_add(0x4000_0000);
            assert!(entry.relocate(moved, &emission.input, &mut relocated));
            phases.relocate += relocating.elapsed();
        }
        at += source_bytes.max(4);
    }
    phases
}

/// `FETCH_IMAGE` names raw guest code (a dyld shared cache subcache's text,
/// kernel code), `FETCH_OFFSET` and `FETCH_BYTES` the part of it compiled.
#[test]
#[ignore = "manual: opt-in"]
fn fetch_phase_costs() {
    let number = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .and_then(|value| {
                value.strip_prefix("0x").map_or_else(
                    || value.parse().ok(),
                    |hex| usize::from_str_radix(hex, 16).ok(),
                )
            })
            .unwrap_or(default)
    };
    let mut image = Vec::new();
    for path in std::env::var("FETCH_IMAGE")
        .expect("FETCH_IMAGE")
        .split(',')
    {
        let offset = number("FETCH_OFFSET", 0);
        let bytes = number("FETCH_BYTES", 8 << 20);
        let file = std::fs::read(path).expect("the code image");
        image.extend_from_slice(&file[offset.min(file.len())..(offset + bytes).min(file.len())]);
    }
    let batch = number("FETCH_BATCH", 1);
    let polluter = std::env::var("FETCH_POLLUTE").is_ok().then(Polluter::new);
    let base = 0x1_8040_0000u64;
    // Warm the compiler's tables and the allocator once.
    measure(
        &image[..image.len().min(256 << 10)],
        base,
        batch,
        polluter.as_ref(),
    );
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    let profile = std::env::var("FETCH_PROFILE").ok();
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    if profile.is_some() {
        sampler::start();
    }
    let started = Instant::now();
    let phases = measure(&image, base, batch, polluter.as_ref());
    let wall = started.elapsed();
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    if let Some(path) = &profile {
        sampler::stop(path);
    }
    if let Some(digest) = phases.digest.clone() {
        let digest: [u8; 32] = digest.finalize().into();
        std::println!("digest {}", hex::encode(digest));
    }
    let per = |time: Duration| time.as_secs_f64() * 1e9 / phases.instructions.max(1) as f64;
    let compile = phases.admit + phases.decode + phases.emit;
    std::println!(
        "blocks={} instructions={} per_block={:.2} redone={} refused={} batch={batch} \
         polluted={} wall_ms={}",
        phases.blocks,
        phases.instructions,
        phases.instructions as f64 / phases.blocks.max(1) as f64,
        phases.redone,
        phases.refused,
        polluter.is_some(),
        wall.as_millis()
    );
    std::println!(
        "ns per guest instruction: admit {:.1} decode {:.1} (first window wasted {:.1}) emit {:.1} \
         compile-total {:.1} emission {:.1} stored-form {:.1} relocate {:.1}",
        per(phases.admit),
        per(phases.decode),
        per(phases.first),
        per(phases.emit),
        per(compile),
        per(phases.emission),
        per(phases.stored),
        per(phases.relocate),
    );
}

/// What DEFLATE does to the pack records of real code: every block of the
/// first `FETCH_IMAGE` file (from `FETCH_OFFSET`, 8 MiB) compiled and
/// stored, one record per 4 KiB page, each encoded and compressed at levels
/// 1 and 6: bytes before and after, and microseconds per record each way.
#[test]
#[ignore = "manual: opt-in"]
fn pack_record_compression() {
    use crate::translations::{Digest, Record, translator_id};
    let path = std::env::var("FETCH_IMAGE").expect("FETCH_IMAGE");
    let path = path.split(',').next().unwrap_or_default();
    let offset = std::env::var("FETCH_OFFSET")
        .ok()
        .and_then(|value| usize::from_str_radix(value.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0);
    let file = std::fs::read(path).expect("the code image");
    let image = &file[offset.min(file.len())..(offset + (8 << 20)).min(file.len())];
    let preparation = Arc::new(system_preparation());
    let options = system_options();
    let translator = translator_id();
    let base = 0x1_8040_0000u64;
    let mut records: alloc::collections::BTreeMap<u64, Record> = Default::default();
    let mut staging = [0u8; super::MAX_BLOCK_BYTES as usize];
    let mut at = 0usize;
    while at + 4 <= image.len() {
        let pc = base + at as u64;
        let page = pc & !(FETCH_PAGE - 1);
        let page_end = (at as u64 | (FETCH_PAGE - 1)) + 1;
        let end = (page_end as usize).min(image.len()).min(at + window(pc));
        let length = end - at;
        staging[..length].copy_from_slice(&image[at..end]);
        preparation.admit(&mut staging[..length]);
        let Ok((analysis, Ok(plan))) =
            block::analyze_and_plan(&staging[..length], GuestAddress(pc), &options)
        else {
            at += 4;
            continue;
        };
        let Ok(compiled) = plan.compile() else {
            at += 4;
            continue;
        };
        let input = analysis.source_bytes;
        let step = compiled.source_bytes.max(4);
        let emission = Emission::new(staging[..input].to_vec(), compiled, preparation.clone());
        let context = preparation.context(page);
        if let Some(entry) = Entry::stored(&emission.block, &emission.input, page, pc, context) {
            records
                .entry(page)
                .or_insert_with(|| {
                    let mut key = [0u8; 32];
                    key[..8].copy_from_slice(&page.to_le_bytes());
                    Record::new(translator, Digest(key))
                })
                .insert(entry);
        }
        at += step;
    }
    let encoded: Vec<Vec<u8>> = records.values().map(Record::encode).collect();
    let raw: usize = encoded.iter().map(Vec::len).sum();
    for level in [1u8, 6] {
        let started = Instant::now();
        let packed: Vec<Vec<u8>> = encoded
            .iter()
            .map(|bytes| miniz_oxide::deflate::compress_to_vec(bytes, level))
            .collect();
        let deflate = started.elapsed();
        let started = Instant::now();
        for (bytes, raw) in packed.iter().zip(&encoded) {
            let out = miniz_oxide::inflate::decompress_to_vec(bytes).expect("inflates");
            assert_eq!(out.len(), raw.len());
        }
        let inflate = started.elapsed();
        let compressed: usize = packed.iter().map(Vec::len).sum();
        let per = |time: Duration| time.as_secs_f64() * 1e6 / encoded.len().max(1) as f64;
        std::println!(
            "records={} raw={raw} bytes_per_record={} level={level} compressed={compressed} \
             ratio={:.2} deflate_us_per_record={:.1} inflate_us_per_record={:.1}",
            encoded.len(),
            raw / encoded.len().max(1),
            raw as f64 / compressed.max(1) as f64,
            per(deflate),
            per(inflate),
        );
    }
}

/// What a stored entry of real code holds in memory: every block of the
/// first `FETCH_IMAGE` file (from `FETCH_OFFSET`, `FETCH_BYTES`, default 8
/// MiB) compiled and stored, the mean bytes an entry retains, its recovery
/// rows' packed bytes and what the rows' own vectors took before
/// ([`crate::translations::Rows`]).
#[test]
#[ignore = "manual: opt-in"]
fn stored_entry_bytes() {
    use stela_compiler::block::{RecoveryBinding, UnqualifiedFaultRecovery};
    let number = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    };
    let path = std::env::var("FETCH_IMAGE").expect("FETCH_IMAGE");
    let path = path.split(',').next().unwrap_or_default();
    let (offset, bytes) = (
        number("FETCH_OFFSET", 1 << 20),
        number("FETCH_BYTES", 8 << 20),
    );
    let file = std::fs::read(path).expect("the code image");
    let image = &file[offset.min(file.len())..(offset + bytes).min(file.len())];
    let preparation = Arc::new(system_preparation());
    let options = system_options();
    let base = 0x1_8040_0000u64;
    let mut staging = [0u8; super::MAX_BLOCK_BYTES as usize];
    let (mut entries, mut retained, mut packed, mut unpacked, mut rows) = (0u64, 0, 0, 0, 0);
    let mut at = 0usize;
    while at + 4 <= image.len() {
        let pc = base + at as u64;
        let page = pc & !(FETCH_PAGE - 1);
        let page_end = (at as u64 | (FETCH_PAGE - 1)) + 1;
        let end = (page_end as usize).min(image.len()).min(at + window(pc));
        let length = end - at;
        staging[..length].copy_from_slice(&image[at..end]);
        preparation.admit(&mut staging[..length]);
        let Ok((analysis, Ok(plan))) =
            block::analyze_and_plan(&staging[..length], GuestAddress(pc), &options)
        else {
            at += 4;
            continue;
        };
        let Ok(compiled) = plan.compile() else {
            at += 4;
            continue;
        };
        let step = compiled.source_bytes.max(4);
        let input = analysis.source_bytes;
        let emission = Emission::new(staging[..input].to_vec(), compiled, preparation.clone());
        let context = preparation.context(page);
        if let Some(entry) = Entry::stored(&emission.block, &emission.input, page, pc, context) {
            entries += 1;
            retained += entry.retained_bytes() as u64;
            packed += entry.recovery.capacity() as u64;
            rows += entry.recovery.len() as u64;
            unpacked += (entry.recovery.len() * size_of::<UnqualifiedFaultRecovery>()) as u64;
            for row in entry.recovery.iter() {
                unpacked += (row.bindings().len() * size_of::<RecoveryBinding>()) as u64;
            }
        }
        at += step;
    }
    let per = |bytes: u64| bytes as f64 / entries.max(1) as f64;
    std::println!(
        "entries={entries} rows_per_entry={:.1} retained_per_entry={:.0} \
         recovery_packed={:.0} recovery_as_vectors={:.0} retained_as_vectors={:.0}",
        per(rows),
        per(retained),
        per(packed),
        per(unpacked),
        per(retained - packed + unpacked + 8 * entries),
    );
}

/// The first window's bytes: the engine's, or `FETCH_FIRST` instructions'.
fn first_window() -> usize {
    std::env::var("FETCH_FIRST")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .map_or(FIRST_WINDOW_BYTES, |words| words * 4)
}
