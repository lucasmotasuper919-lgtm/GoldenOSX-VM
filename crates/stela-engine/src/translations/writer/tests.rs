use super::*;
use crate::translations::pack::{FileReads, Pack};
use crate::translations::translator_id;
use alloc::collections::BTreeSet;
use core::cell::RefCell;
use stela_formats::journal::Target;

/// Counter ticks per second, and the steps a second of a boot gives.
const HZ: u64 = 1000;
const STEPS_PER_SECOND: u64 = 10;

/// A pack file in memory whose writes may be cut after a number of bytes,
/// as a power loss cuts a disk's: the cut write lands only partly.
struct Disk {
    bytes: Vec<u8>,
    /// Bytes still written before the cut; `None` for no cut.
    left: Option<u64>,
    cut: bool,
    written: u64,
    /// A commit slot was written since the last flush, and the page keys
    /// each commit left, once durable.
    slot: bool,
    contents: Vec<BTreeSet<Digest>>,
    /// Whether each commit's keys are kept in `contents`.
    track: bool,
    /// Reads of the file a merge made, and the read after which every
    /// read fails (`None`: none does).
    reads: u64,
    fail_after: Option<u64>,
    /// The data writes, offset and bytes.
    writes: Vec<(u64, u64)>,
}

impl Disk {
    fn new(size: u64) -> Self {
        Self {
            bytes: vec![0; size as usize],
            left: None,
            cut: false,
            written: 0,
            slot: false,
            contents: Vec::new(),
            track: true,
            reads: 0,
            fail_after: None,
            writes: Vec::new(),
        }
    }

    /// Carry out every read asked, from the file's bytes now.
    fn serve(&mut self, reads: &mut FileReads) {
        while let Some((id, at, bytes)) = reads.take_asked() {
            self.reads += 1;
            let failed = self.fail_after.is_some_and(|after| self.reads > after);
            let at = at as usize;
            let read = (!failed)
                .then(|| self.bytes.get(at..at + bytes as usize).map(<[u8]>::to_vec))
                .flatten();
            reads.finish(id, read);
        }
    }
}

impl Target for Disk {
    type Error = String;
    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<bool, String> {
        let take = match self.left {
            Some(left) => (left as usize).min(bytes.len()),
            None => bytes.len(),
        };
        let at = offset as usize;
        self.bytes[at..at + take].copy_from_slice(&bytes[..take]);
        self.written += take as u64;
        self.slot |= offset < DATA_START;
        if offset >= DATA_START {
            self.writes.push((offset, take as u64));
        }
        if let Some(left) = &mut self.left {
            *left -= take as u64;
            if take < bytes.len() {
                self.cut = true;
                return Err(String::from("cut"));
            }
        }
        Ok(true)
    }
    fn flush(&mut self) -> Result<bool, String> {
        if self.cut {
            return Err(String::from("cut"));
        }
        if core::mem::take(&mut self.slot) && self.track {
            let found = keys(&self.bytes);
            self.contents.push(found);
        }
        Ok(true)
    }
}

/// What a boot's store hands the writer: entries added, keys found. Like
/// the store, it keeps at most `bound` entries between takes and counts
/// those it refused, and those it shed. Its heap, when measured, holds
/// `heap_total` bytes, of which the rest of the boot takes `filler` and
/// the writer's segments `resident`.
struct Boot {
    added: BTreeMap<Digest, Record>,
    found: Vec<Digest>,
    next: u32,
    total: u64,
    bound: usize,
    unkept: u64,
    heap_total: u64,
    filler: u64,
    resident: u64,
    shedding: bool,
    shed: u64,
    /// The code bytes of each entry added: 0 for records of none.
    entry_bytes: usize,
}

impl Boot {
    /// A boot whose pages are numbered from `seed` millions, that found
    /// every record in `found`.
    fn new(seed: u32, found: Vec<Digest>) -> Self {
        Self {
            added: BTreeMap::new(),
            found,
            next: seed * 1_000_000,
            total: 0,
            bound: usize::MAX,
            unkept: 0,
            heap_total: 0,
            filler: 0,
            resident: 0,
            shedding: false,
            shed: 0,
            entry_bytes: 0,
        }
    }

    /// The heap's room the boot measures: what the rest, the writer's
    /// segments and the entries added leave.
    fn room(&self) -> u64 {
        let added = self.added.len() as u64 * (ENTRY_HEAP_BYTES + self.entry_bytes as u64);
        self.heap_total
            .saturating_sub(self.filler + self.resident + added)
    }

    /// Add `count` entries of new pages: the keys of those kept.
    fn add(&mut self, count: u32) -> Vec<Digest> {
        let mut kept = Vec::new();
        for _ in 0..count {
            self.next += 1;
            kept.extend(self.add_page(self.next));
        }
        kept
    }

    /// Add an entry of page `page`: its key, when kept.
    fn add_page(&mut self, page: u32) -> Option<Digest> {
        if self.shedding {
            self.shed += 1;
            return None;
        }
        if self.added.len() >= self.bound {
            self.unkept += 1;
            return None;
        }
        let key = page_key(page);
        let mut record = Record::new(translator_id(), key);
        if self.entry_bytes != 0 {
            record.insert(entry(0, page as u8, self.entry_bytes));
        }
        self.added.insert(key, record);
        self.total += 1;
        Some(key)
    }
}

impl Entries for Boot {
    fn take_added(&mut self) -> BTreeMap<Digest, Record> {
        core::mem::take(&mut self.added)
    }
    fn found_keys(&mut self) -> Vec<Digest> {
        self.found.clone()
    }
    fn added(&mut self) -> u64 {
        self.total
    }
    fn pressed(&mut self) -> bool {
        !self.added.is_empty() && self.added.len().saturating_mul(2) >= self.bound
    }
    fn heap(&mut self) -> Option<Heap> {
        (self.heap_total != 0).then(|| Heap {
            total: self.heap_total,
            room: self.room(),
        })
    }
    fn shed(&mut self, shed: bool) {
        self.shedding = shed;
    }
    fn added_bytes(&mut self) -> u64 {
        self.added
            .values()
            .map(|record| record.retained_bytes() as u64)
            .sum()
    }
}

/// The key of page `page`'s record.
fn page_key(page: u32) -> Digest {
    Digest::of(&page.to_le_bytes())
}

/// The heap bytes an added entry of [`Boot::add`] takes beside its code.
const ENTRY_HEAP_BYTES: u64 = 256;

/// An entry at page offset `offset` whose code is `bytes` bytes of `fill`.
fn entry(offset: u16, fill: u8, bytes: usize) -> crate::translations::Entry {
    crate::translations::Entry {
        context: Digest::of(b"context"),
        offset,
        source_bytes: 4,
        cold: (0, 0),
        input: vec![0; 4],
        code: vec![fill; bytes],
        relocations: Vec::new(),
        // A merge decodes what two segments hold with every check.
        source_map: vec![(offset, 0)],
        exits: Vec::new(),
        recovery: Default::default(),
    }
}

/// The pack file as a boot's loader reads it: the newest commit, whose
/// blobs must read back whole, and its segments and lists.
fn load(disk: &[u8], limit: u64) -> File<'static> {
    load_holding(disk, limit, false)
}

/// [`load`], each segment's header and index alone when `demand`, as a
/// boot that reads its pack on demand holds them: its records stay in the
/// file, which merges read.
fn load_holding(disk: &[u8], limit: u64, demand: bool) -> File<'static> {
    let translator = translator_id();
    let size = disk.len() as u64;
    let committed = journal::candidates(&disk[..DATA_START as usize], JOURNAL_DOMAIN, size)
        .into_iter()
        .next();
    let mut segments = Vec::new();
    let mut used = Vec::new();
    if let Some(commit) = &committed {
        let blobs = journal::read_entries(commit, |offset, into| {
            let at = offset as usize;
            into.copy_from_slice(disk.get(at..at + into.len())?);
            Some(())
        })
        .expect("the newest commit's blobs read back whole");
        for ((kind, bytes), entry) in blobs.into_iter().zip(&commit.entries) {
            let bytes: &'static [u8] = Vec::leak(bytes);
            match kind {
                SEGMENT_BLOB if demand => {
                    let prefix = crate::translations::pack::prefix_bytes(bytes, &translator);
                    let pack = Pack::index_only(&bytes[..prefix.unwrap()], &translator).unwrap();
                    segments.push(Segment::at(pack, entry.offset));
                }
                SEGMENT_BLOB => {
                    segments.push(Segment::from(Pack::parse(bytes, &translator).unwrap()))
                }
                USED_BLOB => used.push(UsedKeys::parse(bytes, &translator).unwrap()),
                _ => panic!("an unknown blob"),
            }
        }
    }
    File {
        translator,
        limit,
        records: u64::MAX,
        file_bytes: size,
        committed,
        segments,
        used,
        partial: false,
    }
}

/// The page keys the file's newest commit holds.
fn keys(disk: &[u8]) -> BTreeSet<Digest> {
    load(disk, u64::MAX)
        .segments
        .iter()
        .flat_map(Segment::keys)
        .collect()
}

/// A writer's boot: its clock, the store and the disk.
struct Run<'a> {
    writer: Writer<'a>,
    boot: Boot,
    now: u64,
    events: Vec<Event>,
    /// The most heap bytes the writer's segments held after a step.
    resident_peak: u64,
}

impl<'a> Run<'a> {
    /// A boot over `file` whose pages are numbered from `seed` millions,
    /// that uses every record the file holds.
    fn new(file: File<'a>, seed: u32) -> Self {
        let found = file.segments.iter().flat_map(Segment::keys).collect();
        Self {
            writer: Writer::new(file, HZ),
            boot: Boot::new(seed, found),
            now: 0,
            events: Vec::new(),
            resident_peak: 0,
        }
    }

    /// Step for `seconds` of the guest's time.
    fn run(&mut self, disk: &mut Disk, seconds: u64) -> Result<(), String> {
        let disk = RefCell::new(disk);
        for _ in 0..seconds * STEPS_PER_SECOND {
            let now = self.now;
            self.boot.resident = self.writer.resident();
            let events = self.writer.step(
                &mut || now,
                now + 1,
                &mut self.boot,
                &mut |execution| Some(execution.advance(&mut **disk.borrow_mut(), PIECE_BYTES)),
                &mut |reads| disk.borrow_mut().serve(reads),
            )?;
            self.events.extend(events);
            self.resident_peak = self.resident_peak.max(self.writer.resident());
            self.now += HZ / STEPS_PER_SECOND;
        }
        Ok(())
    }

    /// Step for `seconds` while the device takes no write.
    fn stall(&mut self, seconds: u64) {
        for _ in 0..seconds * STEPS_PER_SECOND {
            let now = self.now;
            let events = self
                .writer
                .step(
                    &mut || now,
                    now + 1,
                    &mut self.boot,
                    &mut |_| None,
                    &mut |_| {},
                )
                .unwrap();
            self.events.extend(events);
            self.now += HZ / STEPS_PER_SECOND;
        }
    }

    /// The triggers of the segments committed so far.
    fn triggers(&self) -> Vec<Trigger> {
        self.events
            .iter()
            .filter_map(|event| match event {
                Event::Segment { trigger, .. } => Some(*trigger),
                _ => None,
            })
            .collect()
    }

    fn stop(&mut self, disk: &mut Disk) -> Result<Stopped, String> {
        self.writer.stop(&mut self.boot, &mut |execution| {
            while !execution.advance(disk, PIECE_BYTES)? {}
            Ok(())
        })
    }
}

const FILE_BYTES: u64 = DATA_START + 4096 * journal::ALIGN;

/// A boot `seed` of entries added in bursts, each followed by its
/// seconds, and stopped: the keys it added.
fn boot_of(
    disk: &mut Disk,
    seed: u32,
    bursts: &[(u64, u32)],
) -> Result<(BTreeSet<Digest>, Stopped), String> {
    // Its segments' records stay in the file, which merges read.
    let file: File<'_> = load_holding(&disk.bytes.clone(), u64::MAX, true);
    let mut run = Run::new(file, seed);
    let mut added = BTreeSet::new();
    for &(seconds, count) in bursts {
        added.extend(run.boot.add(count));
        run.run(disk, seconds)?;
    }
    let stopped = run.stop(disk)?;
    Ok((added, stopped))
}

#[test]
fn boots_that_hold_only_indexes_merge_their_segments_from_the_file() {
    let mut disk = Disk::new(FILE_BYTES);
    let mut all = BTreeSet::new();
    let mut merges = 0;
    for boot in 0..8 {
        let file = load_holding(&disk.bytes.clone(), u64::MAX, true);
        assert!(file.segments.iter().all(|segment| !segment.in_memory()));
        let mut run = Run::new(file, boot + 1);
        for (seconds, count) in [(30, 200), (100, 50), (200, 10)] {
            all.extend(run.boot.add(count));
            run.run(&mut disk, seconds).unwrap();
        }
        let stopped = run.stop(&mut disk).unwrap();
        merges += run.writer.stats().merges;
        assert!(stopped.segments < MAX_SEGMENTS, "boot {boot}");
        assert_eq!(keys(&disk.bytes), all, "boot {boot}");
    }
    // Every boot read its pack on demand, and merges ran from the file, so
    // it never ran out of slots.
    assert!(merges >= 2, "{merges}");
    assert!(load(&disk.bytes, u64::MAX).segments.len() < MAX_SEGMENTS);
}

#[test]
fn boots_commit_their_segments_and_lists_and_a_later_boot_merges_them() {
    let mut disk = Disk::new(FILE_BYTES);
    let mut all = BTreeSet::new();
    for boot in 0..5 {
        let bursts = [(30, 200), (100, 50), (200, 10)];
        let (added, stopped) = boot_of(&mut disk, boot + 1, &bursts).unwrap();
        all.extend(added);
        assert_eq!(keys(&disk.bytes), all, "boot {boot}");
        assert!(stopped.segments < MAX_SEGMENTS && stopped.lists <= MAX_USED_LISTS);
        assert!(!stopped.merge_left);
    }
    // A merge ran once segments or lists reached MERGE_AT: the file holds
    // fewer than five boots wrote.
    let file = load(&disk.bytes, u64::MAX);
    assert!(file.segments.len() < 5 * 3, "{}", file.segments.len());
    assert!(file.used.len() < 5);
}

#[test]
fn a_long_session_keeps_a_free_slot_and_commits_every_entry_in_bounded_time() {
    let mut disk = Disk::new(FILE_BYTES);
    let mut run = Run::new(load(&disk.bytes.clone(), u64::MAX), 1);
    let mut pending: Vec<(u64, Vec<Digest>)> = Vec::new();
    // Three days of a burst every ten minutes.
    for _ in 0..3 * 24 * 6 {
        pending.push((run.now, run.boot.add(20)));
        run.run(&mut disk, 600).unwrap();
        let committed = keys(&disk.bytes);
        let file = load(&disk.bytes, u64::MAX);
        assert!(
            file.segments.len() < MAX_SEGMENTS,
            "{}",
            file.segments.len()
        );
        // Whatever was added a last delay and a merge ago is in the file.
        let bound = run.now.saturating_sub(HZ * (LAST_DELTA_SECONDS + 600));
        for (at, added) in &pending {
            if *at < bound {
                assert!(
                    added.iter().all(|key| committed.contains(key)),
                    "added at {at}"
                );
            }
        }
    }
    let stats = run.writer.stats();
    assert!(stats.pairs + stats.merges > 3, "{stats:?}");
    run.stop(&mut disk).unwrap();
    let committed = keys(&disk.bytes);
    assert!(
        pending
            .iter()
            .all(|(_, added)| added.iter().all(|key| committed.contains(key)))
    );
}

#[test]
fn a_stop_finishes_a_segment_in_flight_and_leaves_a_merge() {
    // A segment's write queued and not yet done: the stop finishes it.
    let mut disk = Disk::new(FILE_BYTES);
    let mut run = Run::new(load(&disk.bytes.clone(), u64::MAX), 1);
    let added = run.boot.add(5000);
    run.stall(FIRST_DELTA_SECONDS);
    assert!(!run.writer.idle());
    assert!(keys(&disk.bytes).is_empty());
    let stopped = run.stop(&mut disk).unwrap();
    assert!(!stopped.merge_left);
    assert_eq!(keys(&disk.bytes), added.into_iter().collect());
    assert_eq!(stopped.segments, 1);

    // A merge in progress at the stop: the file keeps its segments, and
    // the stop adds its own beside them.
    let mut disk = Disk::new(FILE_BYTES);
    let mut all = BTreeSet::new();
    for seed in 0..MERGE_AT as u32 {
        all.extend(boot_of(&mut disk, seed + 1, &[(1, 100)]).unwrap().0);
    }
    // The boot's merge waits for its first segment's delay; then its write
    // never lands.
    let mut run = Run::new(load(&disk.bytes.clone(), u64::MAX), 9);
    run.stall(FIRST_DELTA_SECONDS + 1);
    assert!(!run.writer.idle());
    assert!(
        run.events
            .iter()
            .any(|event| matches!(event, Event::Merged { full: true, .. })),
        "{:?}",
        run.events
    );
    all.extend(run.boot.add(30));
    let before = load(&disk.bytes, u64::MAX).segments.len();
    let stopped = run.stop(&mut disk).unwrap();
    assert!(stopped.merge_left);
    assert_eq!(stopped.segments, before + 1);
    assert_eq!(keys(&disk.bytes), all);
}

/// Records whose segment takes about `blocks` journal blocks.
fn records_for(blocks: u64) -> u32 {
    let translator = translator_id();
    let sample: Vec<(Digest, crate::translations::pack::Piece)> = (0u32..100)
        .map(|n| {
            let key = Digest::of(&n.to_le_bytes());
            let piece =
                crate::translations::pack::Piece::New(Record::new(translator, key).encode());
            (key, piece)
        })
        .collect();
    let per_record = Layout::new(&translator, sample).bytes() / 100;
    (blocks * journal::ALIGN / per_record) as u32
}

#[test]
fn a_segment_without_room_waits_for_a_merge_and_one_larger_than_the_file_is_left_out() {
    let mut disk = Disk::new(DATA_START + 64 * journal::ALIGN);
    let mut run = Run::new(load(&disk.bytes.clone(), u64::MAX), 1);
    let first: BTreeSet<Digest> = run.boot.add(records_for(10)).into_iter().collect();
    run.run(&mut disk, FIRST_DELTA_SECONDS + 1).unwrap();
    assert_eq!(keys(&disk.bytes), first);
    // More than the room beside the first, and than any room a merge
    // leaves beside its segment.
    let second = run.boot.add(records_for(60));
    run.run(&mut disk, 2 * FIRST_DELTA_SECONDS + 10).unwrap();
    let position = |wanted: fn(&Event) -> bool| run.events.iter().position(wanted);
    let waiting = position(|event| matches!(event, Event::Waiting { .. })).unwrap();
    let merged = position(|event| *event == Event::MergeCommitted { full: true }).unwrap();
    let lost = position(|event| matches!(event, Event::Lost { .. })).unwrap();
    assert!(waiting < merged && merged < lost, "{:?}", run.events);
    // The writer goes on: the next segment commits.
    let third: BTreeSet<Digest> = run.boot.add(20).into_iter().collect();
    run.run(&mut disk, 4 * FIRST_DELTA_SECONDS + 10).unwrap();
    let committed = keys(&disk.bytes);
    assert!(committed.is_superset(&first) && committed.is_superset(&third));
    assert!(second.iter().all(|key| !committed.contains(key)));
    run.stop(&mut disk).unwrap();
}

/// Boots of a writer, the same each run: a first with two segments, three
/// short ones whose lists bring a merge, and one more.
fn scenario(disk: &mut Disk) -> Result<(), String> {
    let boots: [&[(u64, u32)]; 5] = [
        &[(70, 300), (130, 40)],
        &[(5, 20)],
        &[(5, 20)],
        &[(5, 20)],
        &[(90, 50)],
    ];
    for (seed, bursts) in boots.into_iter().enumerate() {
        boot_of(disk, seed as u32 + 1, bursts)?;
    }
    Ok(())
}

#[test]
fn a_cut_at_any_write_leaves_the_content_of_a_whole_commit() {
    const SIZE: u64 = DATA_START + 512 * journal::ALIGN;
    let mut disk = Disk::new(SIZE);
    scenario(&mut disk).unwrap();
    let contents = core::mem::take(&mut disk.contents);
    assert!(contents.len() > 8, "{}", contents.len());
    assert!(
        contents
            .iter()
            .any(|content| content.len() < contents.last().unwrap().len()),
        "commits before the last hold less"
    );
    let total = disk.written;
    let step = (total / 250).max(1);
    let mut cut = 0;
    while cut <= total {
        let mut disk = Disk::new(SIZE);
        disk.left = Some(cut);
        let _ = scenario(&mut disk);
        // The newest commit reads back whole (`load` checks it) and holds
        // what one of the uncut run's commits held.
        let found = keys(&disk.bytes);
        assert!(
            found.is_empty() || contents.contains(&found),
            "a cut after {cut} of {total} bytes left content no commit held"
        );
        cut += step;
    }
}

#[test]
fn entries_added_faster_than_their_bound_go_to_segments_before_any_is_refused() {
    // A boot's kernel adds entries for its first seconds far faster than the
    // bound the store keeps between segments: the writer takes them once
    // they reach half of it, so the store refuses none, and the next boot
    // finds all of them.
    let mut disk = Disk::new(FILE_BYTES);
    let mut run = Run::new(load(&disk.bytes.clone(), u64::MAX), 1);
    run.boot.bound = 1000;
    let mut added = BTreeSet::new();
    for _ in 0..FIRST_DELTA_SECONDS {
        added.extend(run.boot.add(400));
        run.run(&mut disk, 1).unwrap();
    }
    assert_eq!(run.boot.unkept, 0);
    assert!(run.writer.stats().pressed >= 5, "{:?}", run.writer.stats());
    assert!(run.triggers().contains(&Trigger::Pressed));
    run.stop(&mut disk).unwrap();
    assert_eq!(keys(&disk.bytes), added);
    assert_eq!(added.len() as u64, 400 * FIRST_DELTA_SECONDS);
    // The pressed segments were merged as they came: fewer than the slots.
    assert!(load(&disk.bytes, u64::MAX).segments.len() < MAX_SEGMENTS);
}

#[test]
fn the_first_segment_comes_when_translation_goes_quiet_or_at_its_delay() {
    // A burst, then quiet: the first segment commits a few seconds after.
    let mut disk = Disk::new(FILE_BYTES);
    let mut run = Run::new(load(&disk.bytes.clone(), u64::MAX), 1);
    let burst: BTreeSet<Digest> = run.boot.add(3000).into_iter().collect();
    run.run(&mut disk, QUIET_SECONDS + 2).unwrap();
    assert_eq!(run.triggers(), [Trigger::Quiet]);
    assert_eq!(keys(&disk.bytes), burst);
    run.stop(&mut disk).unwrap();

    // Translation that never goes quiet: the first segment at its delay.
    let mut disk = Disk::new(FILE_BYTES);
    let mut run = Run::new(load(&disk.bytes.clone(), u64::MAX), 1);
    for second in 0..FIRST_DELTA_SECONDS + 2 {
        run.boot.add(2 * QUIET_RATE as u32);
        run.run(&mut disk, 1).unwrap();
        if second + 1 < FIRST_DELTA_SECONDS {
            assert!(run.triggers().is_empty(), "at {second} s");
        }
    }
    assert_eq!(run.triggers(), [Trigger::Due]);
    run.stop(&mut disk).unwrap();
}

#[test]
fn entries_that_press_while_a_write_waits_are_held_for_the_next_segment() {
    let mut disk = Disk::new(FILE_BYTES);
    let mut run = Run::new(load(&disk.bytes.clone(), u64::MAX), 1);
    run.boot.bound = 1000;
    let mut added: BTreeSet<Digest> = run.boot.add(400).into_iter().collect();
    // The first segment starts once translation is quiet; its write waits.
    run.stall(QUIET_SECONDS + 2);
    assert!(!run.writer.idle());
    // More than the bound arrives while it waits: the writer takes a batch
    // once it presses, and the store refuses none.
    for _ in 0..3 {
        added.extend(run.boot.add(400));
        run.stall(1);
    }
    assert_eq!(run.boot.unkept, 0);
    assert_eq!(run.writer.stats().held, 1);
    run.run(&mut disk, 5).unwrap();
    run.stop(&mut disk).unwrap();
    assert_eq!(keys(&disk.bytes), added);
}

#[test]
fn a_filling_heap_sheds_the_stores_entries_and_then_lets_the_segments_go() {
    const HEAP: u64 = 8 << 20;
    let mut disk = Disk::new(FILE_BYTES);
    let mut run = Run::new(load(&disk.bytes.clone(), u64::MAX), 1);
    run.boot.heap_total = HEAP;
    run.boot.filler = HEAP / 4;
    run.boot.bound = 2000;
    let shed = |run: &Run| {
        run.events
            .iter()
            .filter(|event| matches!(event, Event::Shed { .. }))
            .count()
    };

    // The heap's room dips under its shed share: the store keeps no entry
    // until the room is back a release share above it.
    let mut committed: BTreeSet<Digest> = run.boot.add(2000).into_iter().collect();
    run.run(&mut disk, QUIET_SECONDS + 2).unwrap();
    assert_eq!(keys(&disk.bytes), committed);
    run.boot.filler = HEAP - HEAP / SHED_SHARE - run.writer.resident() + 1;
    run.run(&mut disk, 1).unwrap();
    assert!(run.boot.shedding);
    assert_eq!(shed(&run), 1, "{:?}", run.events);
    assert!(run.boot.add(100).is_empty());
    run.boot.filler =
        HEAP - HEAP / SHED_SHARE - HEAP / RELEASE_SHARE - run.writer.resident() - (1 << 20);
    run.run(&mut disk, 1).unwrap();
    assert!(!run.boot.shedding);

    // Entries keep coming with room to spare: the segments' indexes (and
    // those waiting for room, whole) grow to their share of the heap, then
    // the store sheds; they never pass it by more than the segment that
    // reached it.
    run.boot.filler = HEAP / 4;
    let cap = HEAP / RESIDENT_SHARE;
    let mut most = 0;
    for _ in 0..4 * FIRST_DELTA_SECONDS {
        committed.extend(run.boot.add(3000));
        run.run(&mut disk, 1).unwrap();
        most = most.max(run.writer.resident());
    }
    assert!(run.boot.shedding && run.boot.shed > 100);
    assert!(most >= cap / 2 && most < cap * 2, "{most} of {cap}");
    assert_eq!(shed(&run), 1, "reported once: {:?}", run.events);
    assert!(run.writer.stats().sheds >= 2);
    let written = keys(&disk.bytes);
    assert!(!written.is_empty());

    // The rest of the boot fills the heap: the writer lets its segments go
    // and the store sheds for the rest of the boot, room or not.
    run.boot.filler = HEAP - HEAP / RELEASE_SHARE + 1;
    run.run(&mut disk, 1).unwrap();
    assert_eq!(run.writer.resident(), 0);
    assert!(
        run.events
            .iter()
            .any(|event| matches!(event, Event::Released { .. })),
        "{:?}",
        run.events
    );
    run.boot.filler = 0;
    run.run(&mut disk, 2 * FIRST_DELTA_SECONDS).unwrap();
    assert_eq!(run.writer.shedding(), (true, true));
    assert!(run.boot.add(100).is_empty());
    assert_eq!(run.writer.resident(), 0);

    // What the file held stays whole; the stop writes the boot's list.
    let stopped = run.stop(&mut disk).unwrap();
    assert!(keys(&disk.bytes).is_superset(&written));
    assert_eq!(stopped.lists, 1);
}

#[test]
fn a_first_boots_records_all_reach_the_file_while_the_heap_holds_their_index_alone() {
    // A first boot's offers, scaled to a 256 MiB heap as build 96's 600 MB
    // were to its 2.9 GB: about 54 MB of records in 8 KiB entries, added
    // faster than their bound, the store pressing every few seconds.
    const HEAP: u64 = 256 << 20;
    const ENTRY: usize = 8 << 10;
    let mut disk = Disk::new(DATA_START + 50_000 * journal::ALIGN);
    disk.track = false;
    let file: File<'_> = load(&disk.bytes.clone(), u64::MAX);
    let mut run = Run::new(file, 1);
    run.boot.heap_total = HEAP;
    run.boot.filler = HEAP / 2;
    run.boot.bound = (HEAP / 32) as usize / ENTRY;
    run.boot.entry_bytes = ENTRY;
    let mut added = BTreeSet::new();
    for _ in 0..22 {
        added.extend(run.boot.add(300));
        run.run(&mut disk, 1).unwrap();
    }
    run.run(&mut disk, 2 * FIRST_DELTA_SECONDS).unwrap();
    assert_eq!(added.len(), 6600);
    // No entry was refused or shed: the heap never pressed.
    assert_eq!((run.boot.unkept, run.boot.shed), (0, 0));
    assert!(
        !run.events
            .iter()
            .any(|event| matches!(event, Event::Shed { .. } | Event::Released { .. })),
        "{:?}",
        run.events
    );
    // Written, every record stays in the file; the heap held at most a
    // pressed segment being written, and keeps the indexes alone.
    let segment = run.boot.bound as u64 * (ENTRY as u64 + 512);
    // About a segment and a half: the one being written, and the batch
    // held for the next.
    assert!(
        run.resident_peak <= 2 * segment && run.resident_peak < HEAP / RESIDENT_SHARE,
        "{} over {}",
        run.resident_peak,
        2 * segment
    );
    assert!(
        run.writer.resident() < 6600 * 256,
        "{}",
        run.writer.resident()
    );
    assert!(run.writer.stats().pairs >= 2, "{:?}", run.writer.stats());
    run.stop(&mut disk).unwrap();
    assert_eq!(keys(&disk.bytes), added);
}

/// Layouts of `records` new records from `first`, each one entry at
/// `offset` of `bytes` code bytes.
fn layout_of(first: u32, records: u32, offset: u16, bytes: usize) -> Layout<'static> {
    let translator = translator_id();
    let records = (first..first + records)
        .map(|n| {
            let key = Digest::of(&n.to_le_bytes());
            let mut record = Record::new(translator, key);
            record.insert(entry(offset, n as u8, bytes));
            (key, crate::translations::pack::Piece::New(record.encode()))
        })
        .collect();
    Layout::new(&translator, records)
}

/// `layouts` written one after another into a file of their own: the
/// file, and each as a segment left in it.
fn filed(layouts: &[Layout<'static>]) -> (Vec<u8>, Vec<Segment<'static>>) {
    let mut file = vec![0u8; 4096];
    let mut segments = Vec::new();
    for layout in layouts {
        let at = file.len() as u64;
        file.extend(layout.to_vec());
        file.resize(file.len().next_multiple_of(4096), 0);
        let index: &'static [u8] = Vec::leak(layout.prefix().to_vec());
        let pack = Pack::index_only(index, &translator_id()).unwrap();
        segments.push(Segment::at(pack, at));
    }
    (file, segments)
}

/// A merge carried out with its reads served from `file`, each read
/// failing after `fail_after` reads: the merge's layout and the reads.
fn merge_from(
    segments: Vec<Segment<'static>>,
    file: &[u8],
    bound: u64,
    fail_after: Option<u64>,
) -> (Option<Merged<'static>>, u64) {
    let mut merge = Merge::new(translator_id(), segments, Vec::new(), bound);
    let mut disk = Disk::new(0);
    disk.bytes = file.to_vec();
    disk.fail_after = fail_after;
    let mut steps = 0;
    while !merge.step(STEP_WORK) {
        disk.serve(merge.reads());
        steps += 1;
        assert!(steps < 1_000_000, "a merge that never ends");
    }
    let failed = merge.failed();
    let reads = disk.reads;
    let merged = merge.finish();
    assert_eq!(merged.is_none(), failed);
    (merged, reads)
}

/// The bytes a merge's layout writes, its records in the file read from
/// `file` as the writer's window reads them.
fn streamed_bytes(layout: Layout<'static>, file: &[u8]) -> Vec<u8> {
    let layout = Arc::new(layout);
    let streamed = Streamed::new(layout.clone());
    let mut out = vec![0u8; layout.bytes() as usize];
    for offset in (0..layout.bytes()).step_by(PIECE_BYTES as usize) {
        let bytes = PIECE_BYTES.min(layout.bytes() - offset);
        let reads: Vec<(u64, Vec<u8>)> = layout
            .file_runs(offset, bytes)
            .into_iter()
            .map(|(at, length, into)| (into, file[at as usize..(at + length) as usize].to_vec()))
            .collect();
        streamed.fill(offset, bytes, &reads);
        journal::Source::read_at(
            &streamed,
            offset,
            &mut out[offset as usize..(offset + bytes) as usize],
        );
    }
    out
}

#[test]
fn a_merge_of_segments_in_the_file_equals_one_of_them_in_memory() {
    // Two segments sharing a third of their keys, those records holding
    // entries at other offsets: the merge keeps one and merges the shared.
    let translator = translator_id();
    let older = || layout_of(0, 300, 0, 3000);
    let newer = || layout_of(200, 300, 64, 2000);
    let in_memory = Merge::new(
        translator,
        vec![
            Segment::Laid(Arc::new(older())),
            Segment::Laid(Arc::new(newer())),
        ],
        Vec::new(),
        u64::MAX,
    )
    .finish()
    .unwrap()
    .layout
    .to_vec();
    let (file, segments) = filed(&[older(), newer()]);
    let (merged, reads) = merge_from(segments, &file, u64::MAX, None);
    let merged = merged.unwrap();
    assert_eq!(merged.invalid, 0);
    // The kept records stay in the file until written: the layout holds
    // its index and the records merged anew alone.
    assert!(merged.layout.streamed());
    assert!(
        merged.layout.owned() < 500 * 2000,
        "{}",
        merged.layout.owned()
    );
    // Shared records read to merge, the rest read in runs to check them.
    assert!((100..=400).contains(&reads), "{reads}");
    assert_eq!(streamed_bytes(merged.layout, &file), in_memory);
}

/// A record in the file that does not decode whole is left out of the
/// merge, counted invalid; a read that fails ends the merge, which lays
/// nothing out.
#[test]
fn a_merge_from_the_file_leaves_out_a_damaged_record_and_stops_at_a_failed_read() {
    let older = layout_of(0, 300, 0, 3000);
    let newer = layout_of(200, 300, 64, 2000);
    let (mut file, segments) = filed(&[older.clone(), newer.clone()]);
    // Damage a record only the older holds, and one both hold.
    let Segment::Filed { at: older_at, .. } = segments[0] else {
        unreachable!("filed");
    };
    let only = Digest::of(&5u32.to_le_bytes());
    let shared = Digest::of(&250u32.to_le_bytes());
    let index = Pack::index_only(older.prefix(), &translator_id()).unwrap();
    for key in [only, shared] {
        let (offset, bytes) = index.locate(&key).unwrap();
        file[(older_at + offset + bytes / 2) as usize] ^= 0x55;
    }
    let (merged, _) = merge_from(segments.clone(), &file, u64::MAX, None);
    let merged = merged.unwrap();
    assert_eq!(merged.invalid, 2);
    let bytes = streamed_bytes(merged.layout, &file);
    let pack = Pack::parse(&bytes, &translator_id()).unwrap();
    // The damaged record held by one segment alone is gone; the shared one
    // keeps the newer segment's entries.
    assert!(pack.get(&only).is_none());
    assert_eq!(pack.records(), 500 - 1);
    let record = Record::decode(pack.get(&shared).unwrap(), &translator_id(), &shared).unwrap();
    assert_eq!(record.entries().len(), 1);
    // Any read failing ends the merge.
    for fail_after in [0, 3, 50] {
        let (merged, reads) = merge_from(segments.clone(), &file, u64::MAX, Some(fail_after));
        assert!(merged.is_none(), "{fail_after}");
        assert!(reads > fail_after);
    }
}

/// A boot writes nothing where the segments its store reads lie, even once
/// a merge replaced them, while the file has room elsewhere; when it has
/// none, it says so once (the store then reads them no more) and keeps
/// every record.
#[test]
fn a_boot_writes_beside_the_segments_its_store_reads_while_the_file_has_room() {
    let mut retired = 0;
    'sizes: for blocks in [4096u64, 64, 32, 24, 20, 18, 16, 14, 12] {
        let mut disk = Disk::new(DATA_START + blocks * journal::ALIGN);
        let mut all = BTreeSet::new();
        for boot in 0..MERGE_AT as u32 {
            // A file too small for the boots before is no case.
            let Ok((added, stopped)) = boot_of(&mut disk, boot + 1, &[(30, 60)]) else {
                continue 'sizes;
            };
            if stopped.lost != 0 {
                continue 'sizes;
            }
            all.extend(added);
        }
        if keys(&disk.bytes) != all {
            continue;
        }
        let file = load_holding(&disk.bytes.clone(), u64::MAX, true);
        let spans: Vec<(u64, u64)> = file
            .committed
            .iter()
            .flat_map(|commit| commit.entries.iter())
            .filter(|entry| entry.kind == SEGMENT_BLOB)
            .map(|entry| (entry.offset, entry.offset + entry.length))
            .collect();
        disk.writes.clear();
        let mut run = Run::new(file, 9);
        for _ in 0..4 {
            all.extend(run.boot.add(60));
            run.run(&mut disk, 2 * FIRST_DELTA_SECONDS).unwrap();
        }
        let retire = run
            .events
            .iter()
            .filter(|event| matches!(event, Event::Retired))
            .count();
        assert!(retire <= 1, "{blocks}: {:?}", run.events);
        assert!(run.writer.stats().merges >= 1, "{blocks}: {:?}", run.events);
        let overlaps = disk.writes.iter().any(|&(at, bytes)| {
            spans
                .iter()
                .any(|&(start, end)| at < end && start < at + bytes)
        });
        assert_eq!(overlaps, retire == 1, "{blocks}: {:?}", run.events);
        retired += retire;
        let stopped = run.stop(&mut disk).unwrap();
        let kept = keys(&disk.bytes);
        let events: Vec<_> = run
            .events
            .iter()
            .filter(|event| !matches!(event, Event::Segment { .. }))
            .collect();
        // What the merges' bounds (their share of a small file) left out
        // is all that is missing.
        let evicted: u64 = run
            .events
            .iter()
            .map(|event| match event {
                Event::Merged { evicted, .. } => *evicted,
                _ => 0,
            })
            .sum();
        assert!(kept.is_subset(&all), "{blocks}");
        assert!(
            (all.len() - kept.len()) as u64 <= evicted,
            "{blocks}: {} of {} {stopped:?} {events:?}",
            kept.len(),
            all.len()
        );
    }
    assert!(retired >= 1);
}

/// A boot whose merge's read fails keeps the file's content, says so once,
/// merges no more, and still writes its segments.
#[test]
fn a_failed_read_leaves_the_file_as_it_was_and_the_boot_writes_on() {
    let mut disk = Disk::new(FILE_BYTES);
    let mut all = BTreeSet::new();
    for boot in 0..MERGE_AT as u32 {
        let (added, _) = boot_of(&mut disk, boot + 1, &[(30, 100)]).unwrap();
        all.extend(added);
    }
    let before = keys(&disk.bytes);
    assert_eq!(before, all);
    disk.reads = 0;
    disk.fail_after = Some(1);
    let file = load_holding(&disk.bytes.clone(), u64::MAX, true);
    let mut run = Run::new(file, 9);
    all.extend(run.boot.add(100));
    run.run(&mut disk, 3 * FIRST_DELTA_SECONDS).unwrap();
    let abandoned = run
        .events
        .iter()
        .filter(|event| matches!(event, Event::Abandoned { .. }))
        .count();
    assert_eq!(abandoned, 1, "{:?}", run.events);
    assert_eq!(run.writer.stats().merges, 0);
    run.stop(&mut disk).unwrap();
    assert_eq!(keys(&disk.bytes), all);
}

#[test]
fn boots_reading_their_pack_whole_converge_on_the_code_they_use() {
    // Every boot uses a base of 250 pages and the drift of the last two
    // boots, 100 new pages each: a used set of 450 records in a limit of
    // about 1000, which the pack keeps while each boot adds its new 100.
    const ENTRY: usize = 4 << 10;
    let record_bytes = {
        let mut record = Record::new(translator_id(), page_key(0));
        record.insert(entry(0, 0, ENTRY));
        record.encode().len() as u64 + 56
    };
    let limit = 1000 * record_bytes;
    let base: Vec<u32> = (0..250).collect();
    let drift = |boot: u32| -> Vec<u32> { (0..100).map(|n| boot * 1000 + n).collect() };
    let mut disk = Disk::new(DATA_START + 8 * limit.div_ceil(journal::ALIGN) * journal::ALIGN);
    disk.track = false;
    for boot in 1..=5u32 {
        let file = load(&disk.bytes.clone(), limit);
        let read: BTreeSet<Digest> = file.segments.iter().flat_map(Segment::keys).collect();
        let pack: u64 = file.segments.iter().map(Segment::bytes).sum();
        // The loader reads all of it within the limit: no trim.
        assert!(
            pack <= limit / 8 * LEFT_EIGHTHS,
            "boot {boot}: {pack} of {limit}"
        );
        let used: BTreeSet<u32> = base
            .iter()
            .copied()
            .chain(drift(boot))
            .chain(if boot > 1 {
                drift(boot - 1)
            } else {
                Vec::new()
            })
            .collect();
        let found: Vec<Digest> = used
            .iter()
            .map(|&page| page_key(page))
            .filter(|key| read.contains(key))
            .collect();
        let loaded = found.len();
        let missing: Vec<u32> = used
            .iter()
            .copied()
            .filter(|&page| !read.contains(&page_key(page)))
            .collect();
        let mut run = Run::new(file, boot);
        run.boot.found = found.clone();
        run.boot.entry_bytes = ENTRY;
        run.boot.bound = 200;
        for chunk in missing.chunks(50) {
            for &page in chunk {
                run.boot.add_page(page);
            }
            run.run(&mut disk, 2).unwrap();
        }
        run.run(&mut disk, 3 * FIRST_DELTA_SECONDS).unwrap();
        let merged = run
            .events
            .iter()
            .any(|event| matches!(event, Event::Merged { full: true, .. }));
        run.stop(&mut disk).unwrap();
        let kept = keys(&disk.bytes);
        let used_keys: BTreeSet<Digest> = used.iter().map(|&page| page_key(page)).collect();
        // Every page the boot used is in the file it leaves, and the pack
        // within its share of the limit.
        assert!(kept.is_superset(&used_keys), "boot {boot}");
        let left = load(&disk.bytes.clone(), limit);
        let left: u64 = left.segments.iter().map(Segment::bytes).sum();
        assert!(left <= limit / 8 * LEFT_EIGHTHS, "boot {boot}: left {left}");
        // Records of pages it used never went while others stayed.
        let evicted_used = found.iter().any(|key| !kept.contains(key));
        let unused_kept = kept.iter().any(|key| !used_keys.contains(key));
        assert!(!(evicted_used && unused_kept), "boot {boot}");
        assert_eq!(run.boot.shed, 0, "boot {boot}");
        match boot {
            1 => assert_eq!((loaded, missing.len()), (0, 350)),
            2 => assert_eq!((loaded, missing.len()), (350, 100)),
            // From the third boot the pack passes the share a merge keeps:
            // a full merge runs and lets the drift this boot did not use
            // go, the oldest first: the last boot's within what the share
            // leaves, the one before's whole.
            _ => {
                assert_eq!((loaded, missing.len()), (350, 100), "boot {boot}");
                assert!(merged, "boot {boot}: {:?}", run.events);
                let kept_of = |pages: Vec<u32>| {
                    pages
                        .iter()
                        .filter(|&&page| kept.contains(&page_key(page)))
                        .count()
                };
                assert!(kept_of(drift(boot - 2)) < 100, "boot {boot}");
                if boot > 3 {
                    assert_eq!(kept_of(drift(boot - 3)), 0, "boot {boot}");
                }
            }
        }
    }
}

#[test]
fn a_first_boot_leaves_at_most_its_share_of_the_limit_and_sheds_the_rest() {
    const ENTRY: usize = 4 << 10;
    let record_bytes = {
        let mut record = Record::new(translator_id(), page_key(0));
        record.insert(entry(0, 0, ENTRY));
        record.encode().len() as u64 + 56
    };
    let limit = 1000 * record_bytes;
    let mut disk = Disk::new(DATA_START + 8 * limit.div_ceil(journal::ALIGN) * journal::ALIGN);
    disk.track = false;
    let file: File<'_> = load(&disk.bytes.clone(), limit);
    let mut run = Run::new(file, 1);
    run.boot.entry_bytes = ENTRY;
    run.boot.bound = 200;
    // It compiles more than the limit holds.
    for _ in 0..24 {
        run.boot.add(50);
        run.run(&mut disk, 2).unwrap();
    }
    run.run(&mut disk, 2 * FIRST_DELTA_SECONDS).unwrap();
    run.stop(&mut disk).unwrap();
    let left = load(&disk.bytes.clone(), limit);
    let left: u64 = left.segments.iter().map(Segment::bytes).sum();
    assert!(left <= limit / 8 * LEFT_EIGHTHS, "{left} of {limit}");
    assert!(left > limit / 2, "{left} of {limit}");
    assert!(run.boot.shed > 0);
    let pack_sheds = run
        .events
        .iter()
        .filter(|event| {
            matches!(
                event,
                Event::Shed {
                    reason: ShedReason::Pack,
                    ..
                }
            )
        })
        .count();
    assert_eq!(pack_sheds, 1, "{:?}", run.events);
}

/// A pack read on demand grows to its file, not to the RAM: its limit is
/// the file's data area less the share a full merge writes beside it.
#[test]
fn a_pack_read_on_demand_is_bounded_by_its_file_and_a_merges_room() {
    let file = 2u64 << 30;
    let data = file - DATA_START;
    let limit = demand_limit(file);
    assert_eq!(limit, data - data / 8 * MERGED_SHARE);
    // A boot leaves at most 6/8 of it and a full merge keeps 4/8: within
    // the half of the file past which a full merge runs, and the share
    // its segment may take.
    assert!(limit / 8 * LEFT_EIGHTHS <= data / 2);
    assert!(limit / 8 * KEPT_EIGHTHS <= data / 8 * MERGED_SHARE);
    assert_eq!(demand_limit(0), 0);
    // An 8 GiB host's budget holds the indexes of millions of records.
    assert_eq!(
        resident_records(496 << 20),
        (496 << 20) / crate::translations::pack::RESIDENT_RECORD_BYTES
    );
}

/// The records whose indexes the RAM holds bound a pack read on demand as
/// its bytes do: the store sheds once the pack reaches them.
#[test]
fn a_pack_sheds_at_the_records_its_ram_holds_the_indexes_of() {
    const ENTRY: usize = 4 << 10;
    let record_bytes = {
        let mut record = Record::new(translator_id(), page_key(0));
        record.insert(entry(0, 0, ENTRY));
        record.encode().len() as u64 + 56
    };
    let limit = 1000 * record_bytes;
    let mut disk = Disk::new(DATA_START + 8 * limit.div_ceil(journal::ALIGN) * journal::ALIGN);
    disk.track = false;
    let mut file: File<'_> = load(&disk.bytes.clone(), limit);
    file.records = 300;
    let mut run = Run::new(file, 1);
    run.boot.entry_bytes = ENTRY;
    run.boot.bound = 200;
    for _ in 0..24 {
        run.boot.add(50);
        run.run(&mut disk, 2).unwrap();
    }
    run.run(&mut disk, 2 * FIRST_DELTA_SECONDS).unwrap();
    run.stop(&mut disk).unwrap();
    let left = load(&disk.bytes.clone(), limit);
    let records: usize = left.segments.iter().map(Segment::records).sum();
    // Within the bound and what one look's batch adds past it.
    assert!(records <= 300 + 50, "{records}");
    assert!(records >= 300 - 300 / 16, "{records}");
    let left: u64 = left.segments.iter().map(Segment::bytes).sum();
    assert!(left < limit / 2, "{left} of {limit}");
    assert!(run.boot.shed > 0);
    assert!(run.events.iter().any(|event| matches!(
        event,
        Event::Shed {
            reason: ShedReason::Pack,
            ..
        }
    )));
}

/// A second boot of 300 of the 600 pages the first left, each given
/// `phases` more entries of `entry_bytes` (`grow`) or its first again
/// (else), its pack read on demand: the events, the writer's stats, and
/// the keys the file keeps.
fn boot_over_pack(
    grow: bool,
    phases: u16,
    entry_bytes: usize,
) -> (Vec<Event>, Stats, BTreeSet<Digest>, Vec<Digest>) {
    const ENTRY: usize = 4 << 10;
    let record_bytes = {
        let mut record = Record::new(translator_id(), page_key(0));
        record.insert(entry(0, 0, ENTRY));
        record.encode().len() as u64 + 56
    };
    let limit = 1000 * record_bytes;
    let mut disk = Disk::new(DATA_START + 8 * limit.div_ceil(journal::ALIGN) * journal::ALIGN);
    disk.track = false;
    // The first boot leaves 600 pages.
    let file: File<'_> = load(&disk.bytes.clone(), limit);
    let mut run = Run::new(file, 1);
    run.boot.entry_bytes = ENTRY;
    run.boot.bound = 200;
    for chunk in (0..600).collect::<Vec<u32>>().chunks(50) {
        for &page in chunk {
            run.boot.add_page(page);
        }
        run.run(&mut disk, 2).unwrap();
    }
    run.run(&mut disk, 2 * FIRST_DELTA_SECONDS).unwrap();
    run.stop(&mut disk).unwrap();
    drop(run);
    // The second uses 300 of them and compiles each anew.
    let file = load_holding(&disk.bytes.clone(), limit, true);
    let read: u64 = file.segments.iter().map(Segment::bytes).sum();
    assert!(read > limit / 2, "{read} of {limit}");
    let mut run = Run::new(file, 2);
    let used: Vec<Digest> = (0..300).map(page_key).collect();
    run.boot.found = used.clone();
    run.boot.bound = 200;
    for phase in 1..=phases {
        compile_again(&mut run, &mut disk, |page| {
            if grow {
                entry(4 * phase, page as u8 ^ 0xff, entry_bytes)
            } else {
                entry(0, page as u8, ENTRY)
            }
        });
    }
    run.run(&mut disk, 3 * FIRST_DELTA_SECONDS).unwrap();
    let stats = run.writer.stats();
    run.stop(&mut disk).unwrap();
    (run.events, stats, keys(&disk.bytes), used)
}

/// Offer pages 0 to 300 again, an entry each, 50 every two seconds.
fn compile_again(
    run: &mut Run<'_>,
    disk: &mut Disk,
    entry_of: impl Fn(u32) -> crate::translations::Entry,
) {
    for chunk in (0..300).collect::<Vec<u32>>().chunks(50) {
        for &page in chunk {
            let key = page_key(page);
            let mut record = Record::new(translator_id(), key);
            record.insert(entry_of(page));
            run.boot.added.insert(key, record);
            run.boot.total += 1;
        }
        run.run(disk, 2).unwrap();
    }
}

/// A boot that compiles pages of the pack it read anew merges their
/// records anew from the file and sheds nothing.
#[test]
fn records_merged_anew_from_the_file_shed_nothing() {
    let (events, stats, kept, used) = boot_over_pack(true, 3, 1024);
    assert!(
        events
            .iter()
            .any(|event| matches!(event, Event::Merged { full: true, .. })),
        "{events:?}"
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, Event::Shed { .. })),
        "{events:?}"
    );
    assert!(stats.file_reads > 0 && stats.abandoned == 0, "{stats:?}");
    assert!(used.iter().all(|key| kept.contains(key)));
}

/// A merge whose record of a page is one of its holders' takes that
/// holder's bytes: compiled again the same, nothing is copied.
#[test]
fn a_merge_copies_no_record_a_holder_already_has() {
    let (events, stats, kept, used) = boot_over_pack(false, 1, 4);
    assert!(
        events
            .iter()
            .any(|event| matches!(event, Event::Merged { full: true, .. })),
        "{events:?}"
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, Event::Shed { .. })),
        "{events:?}"
    );
    assert_eq!(stats.abandoned, 0, "{stats:?}");
    assert!(
        used.iter().all(|key| kept.contains(key)),
        "{} of {} kept, {} in all {events:?}",
        used.iter().filter(|key| kept.contains(key)).count(),
        used.len(),
        kept.len()
    );
}

/// Batch after batch that presses while a write waits joins the batch held,
/// within its share of the heap: the store refuses none. Without a known
/// heap, one batch is held and the rest wait in the store's bound.
#[test]
fn batches_that_press_through_a_long_wait_join_the_batch_held() {
    for heap_total in [1u64 << 30, 0] {
        let mut disk = Disk::new(FILE_BYTES);
        let mut run = Run::new(load(&disk.bytes.clone(), u64::MAX), 1);
        run.boot.heap_total = heap_total;
        run.boot.bound = 1000;
        let mut added: BTreeSet<Digest> = run.boot.add(400).into_iter().collect();
        run.stall(QUIET_SECONDS + 2);
        assert!(!run.writer.idle());
        for _ in 0..6 {
            added.extend(run.boot.add(600));
            run.stall(1);
        }
        if heap_total == 0 {
            assert_eq!(run.writer.stats().held, 1);
            assert!(run.boot.unkept > 0);
            continue;
        }
        assert_eq!(run.boot.unkept, 0);
        assert_eq!(run.writer.stats().held, 6);
        run.run(&mut disk, 10).unwrap();
        run.stop(&mut disk).unwrap();
        assert_eq!(keys(&disk.bytes), added);
    }
}

/// A boot over a pack of an earlier boot's 600 pages that uses 100 of them
/// and compiles 400 new ones, allowed `refills` full merges once its pack
/// reaches its share: the new pages, used pages, unused earlier pages and
/// full merges in the file it leaves.
fn boot_compiling_past_its_share(refills: u8) -> (usize, usize, usize, u64) {
    const ENTRY: usize = 4 << 10;
    let record_bytes = {
        let mut record = Record::new(translator_id(), page_key(0));
        record.insert(entry(0, 0, ENTRY));
        record.encode().len() as u64 + 56
    };
    let limit = 1000 * record_bytes;
    let mut disk = Disk::new(DATA_START + 8 * limit.div_ceil(journal::ALIGN) * journal::ALIGN);
    disk.track = false;
    let file: File<'_> = load(&disk.bytes.clone(), limit);
    let mut run = Run::new(file, 1);
    run.boot.entry_bytes = ENTRY;
    run.boot.bound = 200;
    for chunk in (0..600).collect::<Vec<u32>>().chunks(50) {
        for &page in chunk {
            run.boot.add_page(page);
        }
        run.run(&mut disk, 2).unwrap();
    }
    run.run(&mut disk, 2 * FIRST_DELTA_SECONDS).unwrap();
    run.stop(&mut disk).unwrap();
    drop(run);
    let file = load_holding(&disk.bytes.clone(), limit, true);
    let mut run = Run::new(file, 2);
    run.writer.refills = refills;
    run.boot.found = (0..100).map(page_key).collect();
    run.boot.entry_bytes = ENTRY;
    run.boot.bound = 200;
    let fresh: Vec<u32> = (10_000..10_400).collect();
    for chunk in fresh.chunks(50) {
        for &page in chunk {
            run.boot.add_page(page);
        }
        run.run(&mut disk, 2).unwrap();
    }
    run.run(&mut disk, 3 * FIRST_DELTA_SECONDS).unwrap();
    let merges = run.writer.stats().merges;
    run.stop(&mut disk).unwrap();
    let kept = keys(&disk.bytes);
    let kept_of = |pages: &mut dyn Iterator<Item = u32>| {
        pages.filter(|&page| kept.contains(&page_key(page))).count()
    };
    let left = load(&disk.bytes.clone(), limit);
    let left: u64 = left.segments.iter().map(Segment::bytes).sum();
    assert!(left <= limit / 8 * LEFT_EIGHTHS, "left {left} of {limit}");
    (
        kept_of(&mut fresh.iter().copied()),
        kept_of(&mut (0..100)),
        kept_of(&mut (100..600)),
        merges,
    )
}

/// Once the pack reaches its share, a full merge lets the earlier pages the
/// boot did not use go and its new ones are kept; before (no refill), the
/// store shed every new page past the first merge's room.
#[test]
fn a_full_pack_replaces_records_no_boot_used_since_with_new_ones() {
    let before = boot_compiling_past_its_share(0);
    let now = boot_compiling_past_its_share(FULL_REFILLS);
    std::println!("REFILL before new/used/unused/merges={before:?} now={now:?}");
    let (new, used, unused, merges) = now;
    assert_eq!(used, 100);
    assert_eq!(new, 400, "{now:?}");
    assert!(before.0 < new, "{before:?}");
    assert!(unused < before.2, "{before:?} {now:?}");
    assert!(merges <= before.3 + u64::from(FULL_REFILLS));
}

/// What a second boot over a full first boot's pack left: the events, the
/// writer's stats, the store's refused and shed entries, and of the pages
/// it compiled, used from the pack, and the earlier pack's unused pages,
/// those the file keeps.
#[derive(Debug)]
struct SecondBoot {
    events: Vec<Event>,
    stats: Stats,
    unkept: u64,
    shed: u64,
    new: (usize, usize),
    used: (usize, usize),
    unused: (usize, usize),
}

/// A replay of run 20261119-052839's second boot, scaled to a limit of
/// 1000 records: the first boot left 720 (545 of 755 MB); the second
/// starts with a quiet second, then compiles past its share in a burst
/// (sign-in) while it uses 500 of the 720, and then more slowly
/// (launches).
fn second_boot_over_a_full_pack() -> SecondBoot {
    const ENTRY: usize = 4 << 10;
    let record_bytes = {
        let mut record = Record::new(translator_id(), page_key(0));
        record.insert(entry(0, 0, ENTRY));
        record.encode().len() as u64 + 56
    };
    let limit = 1000 * record_bytes;
    let mut disk = Disk::new(DATA_START + 8 * limit.div_ceil(journal::ALIGN) * journal::ALIGN);
    disk.track = false;
    let file: File<'_> = load(&disk.bytes.clone(), limit);
    let mut run = Run::new(file, 1);
    run.boot.entry_bytes = ENTRY;
    run.boot.bound = 200;
    for chunk in (0..720).collect::<Vec<u32>>().chunks(40) {
        for &page in chunk {
            run.boot.add_page(page);
        }
        run.run(&mut disk, 2).unwrap();
    }
    run.run(&mut disk, 2 * FIRST_DELTA_SECONDS).unwrap();
    run.stop(&mut disk).unwrap();
    drop(run);
    let file = load_holding(&disk.bytes.clone(), limit, true);
    let mut run = Run::new(file, 2);
    run.boot.entry_bytes = ENTRY;
    run.boot.bound = 200;
    run.boot.heap_total = 64 * limit;
    // Boot: it uses the first 100 pages and compiles a few.
    run.boot.found = (0..100).map(page_key).collect();
    let mut fresh = 10_000u32..;
    for _ in 0..20 {
        run.boot.add_page(fresh.next().unwrap());
    }
    run.run(&mut disk, QUIET_SECONDS + 3).unwrap();
    // Sign-in: 60 new pages a second for ten seconds while it uses pages
    // 100 to 500 of the pack.
    for second in 0..10u32 {
        run.boot.found = (0..100 + 40 * (second + 1)).map(page_key).collect();
        for _ in 0..60 {
            run.boot.add_page(fresh.next().unwrap());
        }
        run.run(&mut disk, 1).unwrap();
    }
    // Launches: 10 a second for a minute.
    for _ in 0..60 {
        for _ in 0..10 {
            run.boot.add_page(fresh.next().unwrap());
        }
        run.run(&mut disk, 1).unwrap();
    }
    run.run(&mut disk, 3 * FIRST_DELTA_SECONDS).unwrap();
    let stats = run.writer.stats();
    run.stop(&mut disk).unwrap();
    let kept = keys(&disk.bytes);
    let of = |pages: &mut dyn Iterator<Item = u32>| {
        let pages: Vec<u32> = pages.collect();
        let kept = pages
            .iter()
            .filter(|&&page| kept.contains(&page_key(page)))
            .count();
        (kept, pages.len())
    };
    let compiled = fresh.next().unwrap();
    SecondBoot {
        events: run.events,
        stats,
        unkept: run.boot.unkept,
        shed: run.boot.shed,
        new: of(&mut (10_000..compiled)),
        used: of(&mut (0..500)),
        unused: of(&mut (500..720)),
    }
}

/// Its pack full, the second boot refills: a merge of its segments that
/// takes the pack just under its share frees no room, and the pack stays
/// full until a refill made room. Before, the store shed every new page
/// past the first segments for the rest of the boot and the file kept 260
/// new pages, 340 used ones and 140 the boot never used (600 of what it
/// ran); now no unused page stays and the file keeps 750 it ran.
#[test]
fn a_second_boot_over_a_full_pack_keeps_the_code_it_uses() {
    let boot = second_boot_over_a_full_pack();
    std::println!(
        "SECOND BOOT unkept={} shed={} new={:?} used={:?} unused={:?} stats={:?}",
        boot.unkept,
        boot.shed,
        boot.new,
        boot.used,
        boot.unused,
        boot.stats
    );
    let shed_at = boot
        .events
        .iter()
        .position(|event| {
            matches!(
                event,
                Event::Shed {
                    reason: ShedReason::Pack,
                    ..
                }
            )
        })
        .expect("the pack fills");
    let refills = boot.events[shed_at..]
        .iter()
        .filter(|event| matches!(event, Event::Merged { full: true, .. }))
        .count();
    assert_eq!(refills, usize::from(FULL_REFILLS), "{:?}", boot.events);
    assert_eq!(boot.unused.0, 0, "{boot:?}");
    assert!(boot.new.0 + boot.used.0 >= 700, "{boot:?}");
}

/// The longest step of a merge from the file and what it read: two
/// segments of 15 000 records of 4 KiB sharing a third of their keys, each
/// step [`STEP_WORK`] units with the reads served between steps.
/// `cargo test -p stela-engine --lib --release merge_from_the_file_step_costs -- --ignored --nocapture`
#[test]
#[ignore = "manual: opt-in"]
fn merge_from_the_file_step_costs() {
    let (file, segments) = filed(&[
        layout_of(0, 15_000, 0, 4096),
        layout_of(10_000, 15_000, 64, 4096),
    ]);
    let mut merge = Merge::new(translator_id(), segments, Vec::new(), u64::MAX);
    let mut disk = Disk::new(0);
    disk.bytes = file;
    let (mut steps, mut longest, mut total) = (0u64, 0u128, 0u128);
    loop {
        let started = std::time::Instant::now();
        let done = merge.step(STEP_WORK);
        let took = started.elapsed().as_micros();
        longest = longest.max(took);
        total += took;
        steps += 1;
        if done {
            break;
        }
        disk.serve(merge.reads());
    }
    let reads = (merge.reads().reads, merge.reads().bytes >> 20);
    let merged = merge.finish().unwrap();
    std::println!(
        "MERGE records={} bytes={}M steps={steps} longest_us={longest} total_ms={} reads={} read_mb={} invalid={}",
        merged.layout.records(),
        merged.layout.bytes() >> 20,
        total / 1000,
        reads.0,
        reads.1,
        merged.invalid
    );
}
