//! A host replay of a pack read on demand over a boot shaped as run
//! 20261119-052839's second boot: 174 500 distinct page keys looked up over
//! 142 beats of ten seconds, the first lookups of each beat in proportion
//! to the blocks it installed, each page's 25 blocks fetched over the
//! seconds after its first, and a record of 16 KiB a page (545 MB over
//! 34 000 records in that pack). The pack holds `held` of the pages the
//! boot looks up (the kernel's 82 % is the ceiling the same workload
//! reached).
//!
//! A lookup of a page whose record is not read yet asks for it and the
//! block compiles meanwhile; a reader reads the asked records first, reads
//! ahead while the disk would be idle (the pack's most used records first,
//! which the replay takes as the boot's own order, each page moved by up
//! to a tenth of the boot), and, when a lookup asks for a page, reads the
//! pages the previous boot first used after it: a launch's. The disk
//! serves reads with a latency each, up to its queue depth at once,
//! through one bandwidth.
use alloc::{
    collections::{BTreeSet, BinaryHeap},
    vec,
    vec::Vec,
};
use core::cmp::Reverse;

/// Thousands of blocks the second boot installed per beat (`PES STORE`).
const PROFILE: [u16; 142] = [
    210, 290, 923, 945, 731, 378, 249, 11, 9, 123, 3, 2, 6, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 152, 185,
    29, 325, 1000, 1059, 1022, 1166, 895, 808, 659, 717, 909, 802, 890, 931, 866, 935, 1046, 1080,
    753, 486, 550, 599, 536, 747, 612, 480, 460, 508, 457, 420, 56, 311, 393, 204, 210, 177, 138,
    47, 121, 250, 56, 203, 36, 178, 34, 124, 32, 80, 15, 50, 47, 135, 3, 15, 2, 3, 66, 26, 22, 7,
    7, 15, 62, 117, 530, 410, 517, 579, 546, 675, 490, 841, 323, 374, 514, 633, 237, 621, 684, 624,
    1081, 480, 522, 851, 406, 467, 300, 587, 347, 489, 590, 255, 185, 356, 389, 285, 832, 671, 759,
    575, 658, 408, 307, 741, 704, 1125, 707, 81, 31, 42, 401, 47, 26, 146, 228, 218, 35,
];
pub(super) const PAGES: usize = 174_500;
const BLOCKS_PER_PAGE: u32 = 25;
pub(super) const RECORD_BYTES: u64 = 16 << 10;
const BEAT_US: u64 = 10_000_000;
/// A page's later blocks come this long after its first, on average.
const SPREAD_US: f64 = 1_000_000.0;
/// What a record costs in memory while it stays in the file: its index
/// entry and its read-ahead place.
pub(super) const INDEX_BYTES: u64 = 48 + 5;
/// The longest a lookup's read may wait in the disk's queue.
const MAX_QUEUE_US: u64 = 1_000_000;
/// A record's bytes decoded: about twice its stored bytes.
const DECODED_BYTES: u64 = 2 * RECORD_BYTES;

/// A disk: the latency of a read, its bandwidth and how many reads it
/// serves at once.
#[derive(Clone, Copy, Debug)]
pub(super) struct Disk {
    pub(super) name: &'static str,
    pub(super) latency_us: u64,
    pub(super) mb_per_s: u64,
    pub(super) depth: usize,
}

pub(super) const DISKS: [Disk; 4] = [
    Disk {
        name: "nvme",
        latency_us: 80,
        mb_per_s: 2000,
        depth: 32,
    },
    Disk {
        name: "sata-ssd",
        latency_us: 150,
        mb_per_s: 450,
        depth: 32,
    },
    // Run 20261119-052839 read its 545 MB pack in 1819 ms.
    Disk {
        name: "usb-stick",
        latency_us: 1000,
        mb_per_s: 300,
        depth: 4,
    },
    // A seek a read, one at a time.
    Disk {
        name: "sata-hdd",
        latency_us: 9000,
        mb_per_s: 120,
        depth: 1,
    },
];

/// What the reader does beyond the reads lookups ask for.
#[derive(Clone, Copy, Debug)]
pub(super) struct Reader {
    /// Read the pack's most used records while the disk would be idle.
    pub(super) ahead: bool,
    /// On a page asked for, read this many of the pages the previous boot
    /// first used after it.
    pub(super) launch: usize,
    /// A lookup whose record arrives within this long waits for it.
    pub(super) wait_us: u64,
    /// The pack lays its records out in the order a boot first used them
    /// (a merge's layout); else in key order, by page hash, as merges laid
    /// them out before: reads ahead and a launch's then seek, and the
    /// records after a page in the file are any pages.
    pub(super) first_use: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Outcome {
    /// Every block lookup; those of pages the pack holds, and of those the
    /// ones its record served.
    pub(super) all: u64,
    pub(super) lookups: u64,
    pub(super) hits: u64,
    /// Lookups that found their record being read, and the longest one
    /// waited for it.
    pub(super) in_flight: u64,
    pub(super) longest_wait_us: u64,
    /// Records read, of those read ahead, those the cache let go before a
    /// lookup used them, and when the last read ended.
    pub(super) read: u64,
    pub(super) ahead: u64,
    pub(super) wasted: u64,
    /// Lookups' reads not made: the disk's queue was too long.
    pub(super) refused: u64,
    pub(super) last_read_us: u64,
}

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn uniform(seed: &mut u64) -> f64 {
    (next(seed) >> 11) as f64 / (1u64 << 53) as f64
}

/// Each page's first lookup, in microseconds, in the boot's order.
fn first_lookups(seed: &mut u64) -> Vec<u64> {
    let total: u64 = PROFILE.iter().map(|&count| u64::from(count)).sum();
    let mut times = Vec::with_capacity(PAGES + PROFILE.len());
    for (beat, &count) in PROFILE.iter().enumerate() {
        let pages = (PAGES as u64 * u64::from(count)).div_ceil(total);
        for _ in 0..pages {
            times.push(beat as u64 * BEAT_US + (uniform(seed) * BEAT_US as f64) as u64);
        }
    }
    times.sort_unstable();
    times.truncate(PAGES);
    times
}

/// The disk: the time each read in flight completes, and when its
/// bandwidth is free next.
struct Queue {
    disk: Disk,
    slots: BinaryHeap<Reverse<u64>>,
    pipe: u64,
}

impl Queue {
    /// Issue a read of `bytes` at `now`: when it completes. A read that
    /// follows the last one in the file (`seek` false) pays no latency.
    fn read(&mut self, now: u64, bytes: u64, seek: bool) -> u64 {
        let start = if self.slots.len() < self.disk.depth {
            now
        } else {
            let Reverse(free) = self.slots.pop().expect("a slot");
            now.max(free)
        };
        let transfer = bytes * 1_000_000 / (self.disk.mb_per_s << 20);
        let latency = if seek { self.disk.latency_us } else { 0 };
        let done = (start + latency).max(self.pipe) + transfer;
        self.pipe = done;
        self.slots.push(Reverse(done));
        done
    }
}

struct Run {
    queue: Queue,
    in_pack: Vec<bool>,
    /// When each page's record is (or will be) read, while the decoded
    /// cache holds it.
    ready: Vec<Option<u64>>,
    /// The decoded cache: its bound in records, each page's last use, the
    /// pages by last use, and the records read ahead and not used yet.
    cache: usize,
    touched: Vec<u64>,
    lru: BTreeSet<(u64, usize)>,
    ahead: Vec<bool>,
    unused: usize,
    outcome: Outcome,
}

impl Run {
    fn touch(&mut self, page: usize, now: u64) {
        self.lru.remove(&(self.touched[page], page));
        self.touched[page] = now;
        self.lru.insert((now, page));
        if core::mem::take(&mut self.ahead[page]) {
            self.unused -= 1;
        }
    }

    /// Read `page`'s record at `now`, read `ahead` of a lookup or not,
    /// after the last read in the file or not (`seek`): whether it was.
    /// A lookup's read that would not end within [`MAX_QUEUE_US`] is not
    /// made: the block compiles.
    fn issue(&mut self, page: usize, now: u64, ahead: bool, seek: bool) -> bool {
        if !self.in_pack[page] || self.ready[page].is_some() {
            return false;
        }
        if !ahead && self.queue.pipe > now + MAX_QUEUE_US {
            self.outcome.refused += 1;
            return false;
        }
        while self.lru.len() >= self.cache {
            let (_, gone) = self.lru.pop_first().expect("a record");
            self.ready[gone] = None;
            if core::mem::take(&mut self.ahead[gone]) {
                self.unused -= 1;
                self.outcome.wasted += 1;
            }
        }
        let done = self.queue.read(now, RECORD_BYTES, seek);
        self.ready[page] = Some(done);
        self.touched[page] = now;
        self.lru.insert((now, page));
        if ahead {
            self.ahead[page] = true;
            self.unused += 1;
        }
        self.outcome.read += 1;
        self.outcome.last_read_us = self.outcome.last_read_us.max(done);
        true
    }
}

/// Replay the boot with `held` of its pages in the pack, on `disk`, with
/// a decoded cache of `cache` records.
pub(super) fn replay(held: f64, disk: Disk, reader: Reader, cache: usize) -> Outcome {
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let first = first_lookups(&mut seed);
    let in_pack: Vec<bool> = (0..PAGES).map(|_| uniform(&mut seed) < held).collect();
    let mut previous: Vec<(f64, usize)> = (0..PAGES)
        .map(|page| {
            let moved = (uniform(&mut seed) - 0.5) * 0.2 * PAGES as f64;
            (page as f64 + moved, page)
        })
        .collect();
    previous.sort_by(|one, other| one.0.total_cmp(&other.0));
    let order: Vec<usize> = previous.iter().map(|&(_, page)| page).collect();
    let mut rank = vec![0usize; PAGES];
    for (at, &page) in order.iter().enumerate() {
        rank[page] = at;
    }
    // Key order: the pages by hash, which says nothing of their use.
    let mut key_seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut by_key: Vec<(u64, usize)> =
        (0..PAGES).map(|page| (next(&mut key_seed), page)).collect();
    by_key.sort_unstable();
    let by_key: Vec<usize> = by_key.into_iter().map(|(_, page)| page).collect();
    let mut key_rank = vec![0usize; PAGES];
    for (at, &page) in by_key.iter().enumerate() {
        key_rank[page] = at;
    }
    let mut lookups: Vec<(u64, usize)> = Vec::with_capacity(PAGES * BLOCKS_PER_PAGE as usize);
    for (page, &at) in first.iter().enumerate() {
        lookups.push((at, page));
        for _ in 1..BLOCKS_PER_PAGE {
            let later = -SPREAD_US * (1.0 - uniform(&mut seed)).ln();
            lookups.push((at + later as u64, page));
        }
    }
    lookups.sort_unstable();
    let mut run = Run {
        queue: Queue {
            disk,
            slots: BinaryHeap::new(),
            pipe: 0,
        },
        in_pack,
        ready: vec![None; PAGES],
        cache,
        touched: vec![0; PAGES],
        lru: BTreeSet::new(),
        ahead: vec![false; PAGES],
        unused: 0,
        outcome: Outcome::default(),
    };
    let mut ahead_at = 0;
    for &(now, page) in &lookups {
        // Read ahead while the disk would be idle, and while what was read
        // ahead and not used yet takes at most half the cache.
        while reader.ahead && ahead_at < PAGES && run.unused < cache / 2 {
            // When the bandwidth frees, less the latency a read in flight
            // hides.
            let idle = run.queue.pipe.saturating_sub(disk.latency_us);
            if idle >= now {
                break;
            }
            let next = order[ahead_at];
            ahead_at += 1;
            // Laid out in the order of their first use, reading ahead reads
            // the file in order; in key order, each read seeks.
            if run.issue(next, idle, true, !reader.first_use) {
                run.outcome.ahead += 1;
            }
        }
        run.outcome.all += 1;
        if !run.in_pack[page] {
            continue;
        }
        run.outcome.lookups += 1;
        match run.ready[page] {
            Some(done) if done <= now => {
                run.outcome.hits += 1;
                run.touch(page, now);
            }
            Some(done) => {
                run.outcome.in_flight += 1;
                if done - now <= reader.wait_us {
                    run.outcome.hits += 1;
                    run.outcome.longest_wait_us = run.outcome.longest_wait_us.max(done - now);
                }
                run.touch(page, now);
            }
            None => {
                run.issue(page, now, false, true);
                // The records laid out after it: those the previous boot
                // used next, or in key order any pages.
                let after = match reader.first_use {
                    true => &order[rank[page] + 1..],
                    false => &by_key[key_rank[page] + 1..],
                };
                for &next in after.iter().take(reader.launch) {
                    if run.unused >= cache / 2 {
                        break;
                    }
                    run.issue(next, now, true, false);
                }
            }
        }
    }
    run.outcome
}

/// The table: disks, readers and pack contents.
#[test]
#[ignore = "manual: opt-in"]
fn demand_read_pack_replay() {
    let whole = PAGES as u64 * RECORD_BYTES;
    std::println!(
        "PACK whole={}MB index={}MB (records {}, in memory while in the file)",
        whole >> 20,
        (PAGES as u64 * INDEX_BYTES) >> 20,
        PAGES
    );
    let readers = [
        (
            "demand",
            Reader {
                ahead: false,
                launch: 0,
                wait_us: 0,
                first_use: true,
            },
        ),
        (
            "ahead+launch16",
            Reader {
                ahead: true,
                launch: 16,
                wait_us: 0,
                first_use: true,
            },
        ),
        (
            "ahead+launch16+wait200us",
            Reader {
                ahead: true,
                launch: 16,
                wait_us: 200,
                first_use: true,
            },
        ),
    ];
    for disk in DISKS {
        for cache in [2_048, 8_192, 32_768] {
            for (name, reader) in readers {
                let outcome = replay(0.8, disk, reader, cache);
                std::println!(
                    "DEMAND disk={} cache={}MB reader={name} hit%={:.1} of_all%={:.1} \
                     in_flight={} longest_wait_us={} read={} ahead={} wasted={} refused={} last_read_s={}",
                    disk.name,
                    (cache as u64 * DECODED_BYTES) >> 20,
                    100.0 * outcome.hits as f64 / outcome.lookups.max(1) as f64,
                    100.0 * outcome.hits as f64 / outcome.all.max(1) as f64,
                    outcome.in_flight,
                    outcome.longest_wait_us,
                    outcome.read,
                    outcome.ahead,
                    outcome.wasted,
                    outcome.refused,
                    outcome.last_read_us / 1_000_000,
                );
            }
        }
    }
}

/// The layout's part: the same readers over a pack laid out in key order
/// (merges before) and in the order of first use (a merge's now), and the
/// reader of a disk whose reads seek, which reads ahead only.
#[test]
#[ignore = "manual: opt-in"]
fn demand_read_layout_replay() {
    for disk in DISKS {
        for cache in [8_192, 32_768] {
            for first_use in [false, true] {
                for (name, ahead, launch) in [("ahead+launch16", true, 16), ("ahead", true, 0)] {
                    let reader = Reader {
                        ahead,
                        launch,
                        wait_us: 0,
                        first_use,
                    };
                    let outcome = replay(0.8, disk, reader, cache);
                    std::println!(
                        "LAYOUT disk={} cache={}MB first_use={first_use} reader={name} hit%={:.1} \
                         of_all%={:.1} read={} wasted={} refused={} last_read_s={}",
                        disk.name,
                        (cache as u64 * DECODED_BYTES) >> 20,
                        100.0 * outcome.hits as f64 / outcome.lookups.max(1) as f64,
                        100.0 * outcome.hits as f64 / outcome.all.max(1) as f64,
                        outcome.read,
                        outcome.wasted,
                        outcome.refused,
                        outcome.last_read_us / 1_000_000,
                    );
                }
            }
        }
    }
}

/// The pack's limit: the share of the boot's pages a pack holds when the
/// RAM bounds it (before: the budget of an 8 GiB host, read whole or on
/// demand) and when the file does (now: a 2 GiB file less a merge's room),
/// after a boot (the [`super::writer::LEFT_EIGHTHS`] of the limit it leaves)
/// and after a full merge (the [`super::writer::KEPT_EIGHTHS`] it keeps),
/// with the memory its indexes take on that host. The pack holds pages
/// chosen at random, not by use: a lower bound on its hits.
#[test]
#[ignore = "manual: opt-in"]
fn demand_read_limit_replay() {
    use super::writer::{KEPT_EIGHTHS, LEFT_EIGHTHS, demand_limit, resident_records};
    let footprint = PAGES as u64 * RECORD_BYTES;
    // `pack_budget` of an 8 GiB host at 8 PEs.
    let budget: u64 = 496 << 20;
    let reader = Reader {
        ahead: true,
        launch: 16,
        wait_us: 0,
        first_use: true,
    };
    for (name, limit) in [("ram", budget), ("file", demand_limit(2 << 30))] {
        for (state, eighths) in [("left", LEFT_EIGHTHS), ("kept", KEPT_EIGHTHS)] {
            let bytes = limit / 8 * eighths;
            let pages = (bytes / RECORD_BYTES).min(resident_records(budget)) as usize;
            let held = pages as f64 / PAGES as f64;
            let outcome = replay(held, DISKS[0], reader, 32_768);
            std::println!(
                "LIMIT {name}={}MB {state}={}MB of {}MB held%={:.1} index={}MB of {}MB \
                 hit%={:.1} of_all%={:.1}",
                limit >> 20,
                bytes >> 20,
                footprint >> 20,
                100.0 * held,
                (pages as u64 * super::pack::RESIDENT_RECORD_BYTES) >> 20,
                budget >> 20,
                100.0 * outcome.hits as f64 / outcome.lookups.max(1) as f64,
                100.0 * outcome.hits as f64 / outcome.all.max(1) as f64,
            );
        }
    }
}
