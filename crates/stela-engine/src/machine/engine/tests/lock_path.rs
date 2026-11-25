//! Host timing of the board's code lock under several PEs: each PE, on a
//! thread of its own, walks a trace of real guest code round a code window
//! smaller than the code it reaches, as a launch does: it fetches each
//! block (found, relocated from a kept form, or compiled again), links the
//! direct exit it came by, and passes its section's point once per step,
//! while the ring retires parts in the steps installs take. Prints the
//! wall time of each kind of fetch, of the links, and the code lock's
//! acquisitions, contention, waits and holds by site. Manual:
//! `FETCH_IMAGE=<raw guest code> cargo test -p stela-engine --release --lib
//! lock_path_costs -- --ignored --nocapture` (`FETCH_OFFSET`, `FETCH_BYTES`:
//! the part of the image, default 1 MiB from 1 MiB; `LOCK_PES`, default 8;
//! `LOCK_STEPS` per PE, default 150000; `LOCK_WINDOW`, default a third of
//! the code the image compiles to; `LOCK_HOT`, the share of blocks every PE
//! runs, in percent, default 30; `FETCH_PROFILE`, a file for a sampling
//! profile of the measured pass). Without `FETCH_IMAGE`, a synthetic image.

use super::super::board;
use super::super::parallel::{Hub, Kick};
use super::super::reports::StoreSource;
use super::super::stored::tests::MemoryStore;
use super::reinstall_cost::{image, number};
use super::*;
use std::time::{Duration, Instant};

struct NoKick;
impl Kick for NoKick {
    fn kick(&self, _pe: usize) {}
}

/// A PE's section, left however its thread ends.
struct Section<'a> {
    grace: &'a board::Grace,
    pe: usize,
}

impl<'a> Section<'a> {
    fn enter(grace: &'a board::Grace, pe: usize) -> Self {
        grace.enter(pe);
        Self { grace, pe }
    }
}

impl Drop for Section<'_> {
    fn drop(&mut self) {
        self.grace.leave(self.pe);
    }
}

/// What a fetch did.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// The exit it came by is linked: native code enters the block with
    /// no exit to the engine (the fetch is made only to follow the trace).
    Chained,
    /// The block table had it.
    Table,
    /// The PE's jump cache or the board's index had it.
    Found,
    Compiled,
    Rebased,
    Lost,
    Other,
}

const KINDS: [Kind; 7] = [
    Kind::Chained,
    Kind::Table,
    Kind::Found,
    Kind::Compiled,
    Kind::Rebased,
    Kind::Lost,
    Kind::Other,
];

#[derive(Default, Clone, Copy)]
struct Tally {
    count: u64,
    time: Duration,
}

impl Tally {
    fn add(&mut self, other: Tally) {
        self.count += other.count;
        self.time += other.time;
    }
}

#[derive(Default, Clone, Copy)]
struct Tallies {
    fetches: [Tally; KINDS.len()],
    links: Tally,
    linked: u64,
    evictions: u64,
    install_ticks: u64,
    compile_ticks: u64,
    evict_ticks: u64,
}

/// The TSC's ticks per second, against the wall clock.
fn tsc_hz() -> u64 {
    let wall = Instant::now();
    let start = board::timestamp();
    std::thread::sleep(Duration::from_millis(100));
    let ticks = board::timestamp().wrapping_sub(start);
    (ticks as f64 / wall.elapsed().as_secs_f64()) as u64
}

/// A tiny xorshift: the same trace on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound.max(1) as u64) as usize
    }
}

/// What a board's machines share of their numbering.
type Numbering = (
    Arc<numbering::Registry<Variant>>,
    Arc<numbering::Registry<Encoding>>,
    Arc<numbering::Registry<Encoding>>,
);

/// A machine over `image` at [`BASE`] with a store of its own.
fn traced_machine(image: &[u8]) -> Machine<FakeHost, NoDevices> {
    let mut machine = machine(&[]);
    let mut ram = image.to_vec();
    ram.resize(image.len().next_multiple_of(FETCH_PAGE as usize), 0);
    machine.host.ram = ram;
    machine.host.store = Some(MemoryStore::default());
    machine
}

fn windowed(code: &mut [u8]) -> Arc<board::Code> {
    let address = code.as_mut_ptr() as usize;
    Arc::new(board::Code::new(
        board::slot_capacity(Some(code.len())),
        chain::MIN_LOOKUP_ENTRIES,
        Some(CodeWindow {
            exec: address,
            write: address,
            bytes: code.len(),
        }),
    ))
}

/// The blocks of `image` in order, each with its direct successors in the
/// image, and the code the image compiles to.
fn discover(image: &[u8]) -> (Vec<(u64, Vec<u64>)>, usize) {
    let mut memory = vec![0u8; 512 << 20];
    let mut machine = traced_machine(image);
    machine.code = windowed(&mut memory);
    let end = BASE + image.len() as u64;
    let mut blocks = Vec::new();
    let mut at = 0u64;
    while at + 4 <= image.len() as u64 {
        let pc = BASE + at;
        match machine.fetch_block(pc, false) {
            Ok(Fetched::Block(slot)) => {
                let block = machine.code.blocks.installed(slot).expect("a live block");
                let successors = block
                    .successors
                    .iter()
                    .map(|exit| exit.target.0)
                    .filter(|&target| (BASE..end).contains(&target) && target != pc)
                    .collect();
                blocks.push((pc, successors));
                at += 4 * u64::from(block.instructions.max(1));
            }
            _ => at += 4,
        }
        if let Some(store) = &mut machine.host.store {
            store.offers.clear();
        }
    }
    let arena = machine.code.arena.as_ref().expect("an arena");
    let used = arena.ring.generation() as usize * arena.part_bytes;
    (blocks, used)
}

/// One PE's walk: `steps` fetches of the trace its `rng` draws.
#[allow(clippy::too_many_arguments)]
fn walk(
    pe: usize,
    image: &[u8],
    blocks: &[(u64, Vec<u64>)],
    hot: usize,
    code: &Arc<board::Code>,
    numbering: &Numbering,
    hub: &Arc<Hub>,
    (pes, steps): (usize, usize),
    hz: u64,
) -> Tallies {
    let mut machine = traced_machine(image);
    machine.code = Arc::clone(code);
    machine.variants = Arc::clone(&numbering.0);
    machine.stored_reads = Arc::clone(&numbering.1);
    machine.stored_writes = Arc::clone(&numbering.2);
    machine.code_hub = Some(Arc::clone(hub));
    machine.current = pe;
    machine.host.real_counter = true;
    machine.profile.counter_frequency = hz;
    machine
        .code
        .rebases
        .set_bound(code.arena.as_ref().map_or(0, |arena| arena.window.bytes));
    let index: std::collections::HashMap<u64, usize> = blocks
        .iter()
        .enumerate()
        .map(|(at, (pc, _))| (*pc, at))
        .collect();
    // Every PE runs the first `hot` blocks; each its own share of the rest.
    let cold = (blocks.len() - hot) / pes;
    let own = hot + pe * cold..hot + (pe + 1) * cold;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ ((pe as u64 + 1) * 0x1234_5678));
    let grace = &hub.board.grace;
    let section = Section::enter(grace, pe);
    let mut tallies = Tallies::default();
    let mut at = own.start.min(blocks.len() - 1);
    // The block and exit the walk came by.
    let mut came: Option<(usize, usize)> = None;
    for _ in 0..steps {
        grace.pass(pe);
        let pc = blocks[at].0;
        // A linked exit enters its target with no exit to the engine; any
        // other links first, as the run loop links before it fetches the
        // target.
        let mut chained = false;
        if let Some((from, position)) = came.take() {
            let record = machine.code.blocks.record(from);
            chained = record.live() && record.linked(position);
            if !chained {
                let word = machine.code.blocks[from].successors[position].word();
                let started = Instant::now();
                let linked = machine.link(from, pc, word);
                tallies.links.add(Tally {
                    count: 1,
                    time: started.elapsed(),
                });
                tallies.linked += u64::from(linked);
            }
        }
        let sources = machine.statistics.store_sources;
        let fetches = machine.statistics.fetches;
        let lost = machine.statistics.lost_installs;
        let started = Instant::now();
        let fetched = machine.fetch_block(pc, false);
        let time = started.elapsed();
        let delta = |at: StoreSource| {
            let at = at as usize;
            machine.statistics.store_sources[0][at] + machine.statistics.store_sources[1][at]
                != sources[0][at] + sources[1][at]
        };
        let kind = if !matches!(fetched, Ok(Fetched::Block(_))) {
            Kind::Other
        } else if chained {
            Kind::Chained
        } else if machine.statistics.lost_installs != lost {
            Kind::Lost
        } else if delta(StoreSource::Rebased) {
            Kind::Rebased
        } else if delta(StoreSource::Compiled) || delta(StoreSource::Unstorable) {
            Kind::Compiled
        } else if machine.statistics.fetches[0] + machine.statistics.fetches[1]
            != fetches[0] + fetches[1]
        {
            Kind::Table
        } else if machine.statistics.fetches[2] + machine.statistics.fetches[3]
            != fetches[2] + fetches[3]
        {
            Kind::Found
        } else {
            Kind::Other
        };
        let position = KINDS.iter().position(|&k| k == kind).expect("a kind");
        tallies.fetches[position].add(Tally { count: 1, time });
        if let Some(store) = &mut machine.host.store {
            store.offers.clear();
        }
        let Ok(Fetched::Block(slot)) = fetched else {
            at = own.start + rng.below(cold);
            continue;
        };
        // A direct successor three times in four, else a return or call
        // elsewhere: to the shared code mostly, else this PE's own.
        let successors = &blocks[at].1;
        let next = if !successors.is_empty() && rng.below(4) != 0 {
            let target = successors[rng.below(successors.len())];
            let position = machine.code.blocks[slot]
                .successors
                .iter()
                .position(|exit| exit.target.0 == target);
            index.get(&target).map(|&next| {
                came = position.map(|position| (slot, position));
                next
            })
        } else {
            None
        };
        at = next.unwrap_or_else(|| {
            if rng.below(10) < 7 || cold == 0 {
                rng.below(hot.max(1))
            } else {
                own.start + rng.below(cold)
            }
        });
    }
    drop(section);
    tallies.evictions = machine.statistics.code_evictions;
    tallies.install_ticks = machine.statistics.install_ticks;
    tallies.compile_ticks = machine.statistics.compile_ticks;
    tallies.evict_ticks = machine.install_causes.evict_ticks;
    tallies
}

#[test]
#[ignore = "manual: opt-in"]
fn lock_path_costs() {
    let image = image();
    let pes = number("LOCK_PES", 8);
    let steps = number("LOCK_STEPS", 150_000);
    let (blocks, used) = discover(&image);
    let window = number("LOCK_WINDOW", (used / 3).max(1 << 20));
    let hot = blocks.len() * number("LOCK_HOT", 30) / 100;
    let hz = tsc_hz();
    std::println!(
        "image={}K blocks={} code={}K window={}K pes={pes} steps={steps} hot={hot} tsc_mhz={}",
        image.len() >> 10,
        blocks.len(),
        used >> 10,
        window >> 10,
        hz / 1_000_000
    );
    let mut memory = vec![0u8; window];
    let code = windowed(&mut memory);
    let template = traced_machine(&image);
    let numbering: Numbering = (
        Arc::clone(&template.variants),
        Arc::clone(&template.stored_reads),
        Arc::clone(&template.stored_writes),
    );
    drop(template);
    let hub = Hub::new(pes, NoKick);
    // A first pass of every PE warms the kept forms and the ring.
    let run = |steps: usize| {
        let started = Instant::now();
        let all = std::thread::scope(|scope| {
            let threads: Vec<_> = (0..pes)
                .map(|pe| {
                    let (image, blocks, code, numbering, hub) =
                        (&image, &blocks, &code, &numbering, &hub);
                    scope.spawn(move || {
                        walk(
                            pe,
                            image,
                            blocks,
                            hot,
                            code,
                            numbering,
                            hub,
                            (pes, steps),
                            hz,
                        )
                    })
                })
                .collect();
            threads
                .into_iter()
                .map(|thread| thread.join().expect("a PE"))
                .collect::<Vec<_>>()
        });
        (all, started.elapsed())
    };
    let _ = run(steps / 4);
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    let profile = std::env::var("FETCH_PROFILE").ok();
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    if profile.is_some() {
        super::super::fetch_costs::sampler::start();
    }
    let before = code.lock_counts();
    let (all, wall) = run(steps);
    let after = code.lock_counts();
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    if let Some(path) = &profile {
        super::super::fetch_costs::sampler::stop(path);
    }
    let mut sum = Tallies::default();
    for tallies in &all {
        for (total, one) in sum.fetches.iter_mut().zip(&tallies.fetches) {
            total.add(*one);
        }
        sum.links.add(tallies.links);
        sum.linked += tallies.linked;
        sum.evictions += tallies.evictions;
        sum.install_ticks += tallies.install_ticks;
        sum.compile_ticks += tallies.compile_ticks;
        sum.evict_ticks += tallies.evict_ticks;
    }
    let ns = |ticks: u64| ticks as f64 * 1e9 / hz as f64;
    let each = |tally: &Tally| tally.time.as_nanos() as f64 / tally.count.max(1) as f64;
    let fetches: u64 = sum.fetches.iter().map(|tally| tally.count).sum();
    std::println!("wall_ms={} fetches={fetches}", wall.as_millis());
    for (kind, tally) in KINDS.iter().zip(&sum.fetches) {
        std::println!(
            "  {kind:?}: count={} ns_each={:.0} share_of_time={:.1}%",
            tally.count,
            each(tally),
            100.0 * tally.time.as_secs_f64() / (wall.as_secs_f64() * pes as f64)
        );
    }
    let kind = |kind: Kind| sum.fetches[KINDS.iter().position(|&k| k == kind).expect("a kind")];
    let installs = kind(Kind::Compiled).count + kind(Kind::Rebased).count + kind(Kind::Lost).count;
    std::println!(
        "  exits={} of {fetches} steps (the rest chained)",
        fetches - kind(Kind::Chained).count
    );
    std::println!(
        "  links: calls={} linked={} ns_each={:.0}; installs={installs} install_ns_each={:.0} \
         compile_ns_each={:.0} evict_ns_per_install={:.0} evictions={}",
        sum.links.count,
        sum.linked,
        each(&sum.links),
        ns(sum.install_ticks) / installs.max(1) as f64,
        ns(sum.compile_ticks) / kind(Kind::Compiled).count.max(1) as f64,
        ns(sum.evict_ticks) / installs.max(1) as f64,
        sum.evictions,
    );
    let events = &code.events;
    let load = |counter: &core::sync::atomic::AtomicU64| counter.load(Ordering::Relaxed);
    std::println!(
        "  board: grace_waits={} overdue={} redone={} lost={} retired_parts={} live={}",
        load(&events.grace_waits),
        load(&events.overdue),
        load(&events.redone),
        load(&events.lost),
        load(&events.retired),
        code.blocks.live(),
    );
    let acquired = after.acquired - before.acquired;
    let contended = after.contended - before.contended;
    std::println!(
        "  lock: acquired={acquired} contended={contended} ({:.1}%) waited_ms={:.1} \
         per_install_wait_ns={:.0}",
        100.0 * contended as f64 / acquired.max(1) as f64,
        ns(after.waited - before.waited) / 1e6,
        ns(after.waited - before.waited) / installs.max(1) as f64
    );
    for (site, (now, then)) in board::LockSite::ALL
        .iter()
        .zip(after.sites.iter().zip(&before.sites))
    {
        let acquired = now.acquired - then.acquired;
        if acquired == 0 {
            continue;
        }
        std::println!(
            "    {}: acquired={acquired} held_ms={:.1} held_ns_each={:.0} waited_ms={:.1}",
            site.name(),
            ns(now.held - then.held) / 1e6,
            ns(now.held - then.held) / acquired as f64,
            ns(now.waited - then.waited) / 1e6,
        );
    }
}
