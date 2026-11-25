//! Host timing of the fetches a ring's turnover makes: the guest's code
//! runs round a working set larger than the code window, so the ring takes
//! blocks away and the guest asks for them again; each comes back as a
//! kept form relocated ([`super::super::rebase`]) or compiled again. Prints
//! the wall time of each kind per block, and with `FETCH_PROFILE` a
//! sampling profile of the rounds for `nm`. Manual:
//! `FETCH_IMAGE=<raw guest code> cargo test -p stela-engine --release --lib
//! reinstall_cost -- --ignored --nocapture` (`FETCH_OFFSET`, `FETCH_BYTES`:
//! the part of the image, default 1 MiB from 1 MiB; `REINSTALL_WINDOW`:
//! the window's bytes, default half the code the first round installs;
//! `REINSTALL_ROUNDS`, default 6). Without `FETCH_IMAGE`, a synthetic
//! image of short blocks.

use super::super::board;
use super::super::reports::StoreSource;
use super::super::stored::tests::MemoryStore;
use super::*;
use std::time::{Duration, Instant};

const ADD: u32 = 0x9100_0400; // add x0, x0, #1
const B8: u32 = 0x1400_0002; // b .+8

pub(super) fn number(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| {
            value.strip_prefix("0x").map_or_else(
                || value.parse().ok(),
                |hex| usize::from_str_radix(hex, 16).ok(),
            )
        })
        .unwrap_or(default)
}

pub(super) fn image() -> Vec<u8> {
    let bytes = number("FETCH_BYTES", 1 << 20);
    match std::env::var("FETCH_IMAGE") {
        Ok(path) => {
            let offset = number("FETCH_OFFSET", 1 << 20);
            let file = std::fs::read(path).expect("the code image");
            file[offset.min(file.len())..(offset + bytes).min(file.len())].to_vec()
        }
        Err(_) => [ADD, ADD, ADD, B8, 0]
            .iter()
            .cycle()
            .take(bytes / 4)
            .flat_map(|word| word.to_le_bytes())
            .collect(),
    }
}

/// A machine over `image` at [`BASE`] with a store, whose code goes to a
/// window of `window` bytes.
fn ring_machine(image: &[u8], memory: &mut [u8]) -> Machine<FakeHost, NoDevices> {
    let mut machine = machine(&[]);
    let mut ram = image.to_vec();
    ram.resize(image.len().next_multiple_of(FETCH_PAGE as usize), 0);
    machine.host.ram = ram;
    machine.host.store = Some(MemoryStore::default());
    let address = memory.as_mut_ptr() as usize;
    machine.code = Arc::new(board::Code::new(
        board::slot_capacity(Some(memory.len())),
        chain::MIN_LOOKUP_ENTRIES,
        Some(CodeWindow {
            exec: address,
            write: address,
            bytes: memory.len(),
        }),
    ));
    machine
}

/// What a fetch did, from the store sources it counted.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// Still installed: nothing to do but find it.
    Found,
    Compiled,
    Rebased,
    Other,
}

const KINDS: usize = 4;

#[derive(Default, Clone, Copy)]
struct Tally {
    fetches: u64,
    time: Duration,
}

/// Fetch the block at every PC of `pcs` once, in order, tallying each by
/// kind.
fn round(machine: &mut Machine<FakeHost, NoDevices>, pcs: &[u64], tallies: &mut [Tally; KINDS]) {
    for &pc in pcs {
        let before = machine.statistics.store_sources;
        let started = Instant::now();
        let fetched = machine.fetch_block(pc, false);
        let time = started.elapsed();
        let after = machine.statistics.store_sources;
        let counted = |source: StoreSource| {
            let at = source as usize;
            after[0][at] + after[1][at] != before[0][at] + before[1][at]
        };
        let kind = if !matches!(fetched, Ok(Fetched::Block(_))) {
            Kind::Other
        } else if counted(StoreSource::Rebased) {
            Kind::Rebased
        } else if counted(StoreSource::Compiled) || counted(StoreSource::Unstorable) {
            Kind::Compiled
        } else if after == before {
            Kind::Found
        } else {
            Kind::Other
        };
        let tally = &mut tallies[kind as usize];
        tally.fetches += 1;
        tally.time += time;
    }
}

#[test]
#[ignore = "manual: opt-in"]
fn reinstall_cost() {
    let image = image();
    // The first round: every block of the image, in order, into a window
    // that holds them all.
    let mut memory = vec![0u8; 64 << 20];
    let mut machine = ring_machine(&image, &mut memory);
    let mut pcs = Vec::new();
    let mut at = 0u64;
    while at + 4 <= image.len() as u64 {
        let pc = BASE + at;
        match machine.fetch_block(pc, false) {
            Ok(Fetched::Block(slot)) => {
                let block = machine.code.blocks.installed(slot).expect("a live block");
                pcs.push(pc);
                at += 4 * u64::from(block.instructions.max(1));
            }
            _ => at += 4,
        }
    }
    let arena = machine.code.arena.as_ref().expect("an arena");
    let used = arena.ring.generation() as usize * arena.part_bytes;
    drop(machine);
    let window = number("REINSTALL_WINDOW", (used / 2).max(64 << 10));
    let rounds = number("REINSTALL_ROUNDS", 6);
    std::println!(
        "image={} blocks={} first-round code={}K window={}K rounds={rounds}",
        image.len(),
        pcs.len(),
        used >> 10,
        window >> 10
    );
    assert!(!pcs.is_empty());
    // Then rounds over the same blocks with a window too small for them.
    let mut memory = vec![0u8; window];
    let mut machine = ring_machine(&image, &mut memory);
    // The kept forms' bound, as a boot gives it: about the window (run
    // 20261119-014905: 504 MiB beside a 528 MiB window).
    machine
        .code
        .rebases
        .set_bound(number("REINSTALL_KEPT", window));
    let mut warm = [Tally::default(); KINDS];
    round(&mut machine, &pcs, &mut warm);
    round(&mut machine, &pcs, &mut warm);
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    let profile = std::env::var("FETCH_PROFILE").ok();
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    if profile.is_some() {
        super::super::fetch_costs::sampler::start();
    }
    let mut tallies = [Tally::default(); KINDS];
    let started = Instant::now();
    for _ in 0..rounds {
        round(&mut machine, &pcs, &mut tallies);
    }
    let wall = started.elapsed();
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    if let Some(path) = &profile {
        super::super::fetch_costs::sampler::stop(path);
    }
    for (kind, tally) in [Kind::Found, Kind::Compiled, Kind::Rebased, Kind::Other]
        .iter()
        .zip(&tallies)
    {
        std::println!(
            "{kind:?}: fetches={} ns_per_fetch={:.0}",
            tally.fetches,
            tally.time.as_nanos() as f64 / tally.fetches.max(1) as f64
        );
    }
    let events = &machine.code.events;
    let rebases = machine.code.rebases.counts();
    let ring = &machine.code.arena.as_ref().expect("an arena").ring;
    std::println!(
        "wall_ms={} evictions={} lost={} redone={} promoted={} kept={} kept_bytes={}K \
         unstorable={} rebase_hits={} parts={} young={} generation={}",
        wall.as_millis(),
        machine.statistics.code_evictions,
        machine.statistics.lost_installs,
        events.redone.load(core::sync::atomic::Ordering::Relaxed),
        events.promoted.load(core::sync::atomic::Ordering::Relaxed),
        rebases.kept,
        rebases.bytes >> 10,
        rebases.unstorable,
        rebases.hits,
        ring.parts,
        ring.young_parts,
        ring.generation(),
    );
}
