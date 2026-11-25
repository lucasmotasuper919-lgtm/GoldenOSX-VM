use super::super::stored::tests::MemoryStore;
use super::super::tests::{BASE, machine};
use super::super::{Fetched, Host};
use super::*;
use alloc::vec::Vec;

/// `cbz x0, +12; add x1, x1, #1; add x2, x2, #2; add x3, x3, #3; svc #0`:
/// one block, whose side exit leaves for the third ADD.
const WORDS: [u32; 5] = [
    0xb400_0060,
    0x9100_0421,
    0x9100_0842,
    0x9100_0c63,
    0xd400_0001,
];

/// The offer the fetch of `pc` makes in a machine with a store.
fn fetched_offer(pc: u64) -> Offer {
    offers(&WORDS, &[pc]).pop().unwrap()
}

/// The offers the fetches of `pcs`, in order, make in a machine running
/// `words` with a store.
fn offers(words: &[u32], pcs: &[u64]) -> Vec<Offer> {
    let mut fetching = machine(words);
    fetching.host.store = Some(MemoryStore::default());
    for &pc in pcs {
        assert!(matches!(
            fetching.fetch_block(pc, false),
            Ok(Fetched::Block(_))
        ));
    }
    let store = fetching.host.store.as_mut().unwrap();
    assert_eq!(store.offers.len(), pcs.len());
    core::mem::take(&mut store.offers)
}

/// `cbz x0, BASE + 0x1000; svc #0`, NOPs to the next page and `svc #0`
/// there: a block whose side exit leaves for another page.
fn two_pages() -> Vec<u32> {
    let mut words = alloc::vec![0xd503_201f; 0x1000 / 4 + 1];
    words[0] = 0xb400_0000 | (0x1000 / 4) << 5;
    words[1] = 0xd400_0001;
    words[0x1000 / 4] = 0xd400_0001;
    words
}

/// The page at `at`, as a machine running `words` holds it.
fn page_of(words: &[u32], at: u64) -> Box<[u8; PAGE_BYTES]> {
    let mut reader = machine(words);
    let mut page = Box::new([0u8; PAGE_BYTES]);
    assert!(reader.host.read_ram(at, page.as_mut_slice()));
    page
}

/// The page holding `BASE`, as the test machine's RAM holds it.
fn page() -> Box<[u8; PAGE_BYTES]> {
    let mut reader = machine(&WORDS);
    let mut page = Box::new([0u8; PAGE_BYTES]);
    assert!(
        reader
            .host
            .read_ram(BASE & !(FETCH_PAGE - 1), page.as_mut_slice())
    );
    page
}

#[test]
fn a_fetched_block_names_its_successors_in_its_page_as_jobs() {
    let offer = fetched_offer(BASE);
    let mut targets: Vec<u64> = offer.ahead_jobs().map(|job| job.pc()).collect();
    targets.sort_unstable();
    assert_eq!(targets, [BASE + 12]);
    for job in offer.ahead_jobs() {
        assert_eq!(job.key(), Some(&offer.key()));
        assert_eq!(job.context(), &offer.context());
        assert_eq!(job.frame(), BASE & !(FETCH_PAGE - 1));
    }
}

#[test]
fn a_block_prepared_ahead_is_the_block_a_fetch_compiles() {
    let offer = fetched_offer(BASE);
    let page = page();
    for job in offer.ahead_jobs() {
        assert!(page_matches(&job, &page));
        let prepared = prepare(&job, offer.key(), &page).expect("a block");
        assert!(prepared.ahead());
        assert_eq!(prepared.pc(), job.pc());
        // The same PC fetched in a machine of its own compiles it.
        let compiled = fetched_offer(job.pc());
        assert!(!compiled.ahead());
        assert_eq!(prepared.input(), compiled.input());
        assert_eq!(prepared.context(), compiled.context());
        assert_eq!(
            prepared.block().native.x86_64_code,
            compiled.block().native.x86_64_code
        );
        assert_eq!(
            prepared.block().native.source_map,
            compiled.block().native.source_map
        );
        assert_eq!(prepared.successors(), compiled.successors());
    }
}

#[test]
fn a_fetch_installs_a_block_prepared_ahead_and_offers_it_for_the_pack_once() {
    let offer = fetched_offer(BASE);
    let page = page();
    let job = offer
        .ahead_jobs()
        .find(|job| job.pc() == BASE + 12)
        .expect("the CBZ's target");
    let prepared = prepare(&job, offer.key(), &page).expect("a block");
    let code = prepared.block().native.x86_64_code.clone();
    let successors = prepared.successors().to_vec();
    let mut store = MemoryStore::default();
    store.prepared.insert(
        (prepared.key(), prepared.pc(), prepared.context()),
        prepared.shared(),
    );
    let mut receiver = machine(&WORDS);
    receiver.host.store = Some(store);
    assert!(matches!(
        receiver.fetch_block(BASE + 12, false),
        Ok(Fetched::Block(_))
    ));
    let level = usize::from(receiver.cpu.pstate.el != 0);
    assert_eq!(
        (receiver.statistics.shared, receiver.statistics.compiled),
        (1, 0)
    );
    assert_eq!(receiver.statistics.ahead_installs[level], 1);
    assert_eq!(receiver.host.installed.first(), Some(&code));
    // Offered once, as a block that ran, whose successors are jobs as a
    // fetched block's are.
    let store = receiver.host.store.as_mut().unwrap();
    assert_eq!(store.offers.len(), 1);
    let kept = &store.offers[0];
    assert!(!kept.ahead());
    assert_eq!(kept.pc(), BASE + 12);
    assert_eq!(kept.successors(), successors);
    let in_page = successors
        .iter()
        .filter(|&&pc| pc != BASE + 12 && pc & !(FETCH_PAGE - 1) == BASE & !(FETCH_PAGE - 1))
        .count();
    assert_eq!(kept.ahead_jobs().count(), in_page);
    assert!(kept.entry().is_some());
}

#[test]
fn a_frame_that_changed_is_stale() {
    let offer = fetched_offer(BASE);
    let mut page = page();
    let job = offer.ahead_jobs().next().expect("a job");
    page[(job.pc() - BASE) as usize] ^= 1;
    assert!(!page_matches(&job, &page));
}

#[test]
fn jobs_stop_at_the_depth_bound() {
    let offer = fetched_offer(BASE);
    let page = page();
    let mut job = offer
        .ahead_jobs()
        .find(|job| job.pc() == BASE + 12)
        .expect("the side exit's target");
    // Every depth to the bound, by hand.
    for depth in 1..=DEPTH {
        job.depth = depth;
        let prepared = prepare(&job, offer.key(), &page).expect("a block");
        let successors = prepared
            .successors()
            .iter()
            .filter(|&&pc| pc != job.pc())
            .count();
        let jobs = prepared.ahead_jobs().count();
        if depth < DEPTH {
            assert_eq!(jobs, successors);
        } else {
            assert_eq!(jobs, 0);
        }
    }
}

#[test]
fn a_queue_hands_out_the_newest_job_skips_repeats_and_drops_the_oldest() {
    let offer = fetched_offer(BASE);
    let template = offer.ahead_jobs().next().expect("a job");
    let job = |pc: u64| Job {
        pc,
        ..template.clone()
    };
    let mut queue = Queue::new();
    queue.push(job(BASE + 4));
    queue.push(job(BASE + 8));
    queue.push(job(BASE + 8));
    assert_eq!(queue.counts().queued, 2);
    assert_eq!(queue.counts().repeated, 1);
    assert_eq!(queue.pop().map(|job| job.pc()), Some(BASE + 8));
    assert_eq!(queue.pop().map(|job| job.pc()), Some(BASE + 4));
    assert!(queue.pop().is_none());
    for index in 0..CAPACITY as u64 + 3 {
        queue.push(job(BASE + 0x10_0000 + index * 4));
    }
    assert_eq!(queue.len(), CAPACITY);
    assert_eq!(queue.counts().dropped, 3);
    assert_eq!(
        queue.pop().map(|job| job.pc()),
        Some(BASE + 0x10_0000 + (CAPACITY as u64 + 2) * 4)
    );
}

#[test]
fn a_fetched_block_names_its_targets_in_other_pages_whose_walk_its_pe_holds() {
    let words = two_pages();
    let elsewhere = BASE + 0x1000;
    // The other page fetched first: its walk is in the fetch cache.
    let mut walked = offers(&words, &[elsewhere, BASE]);
    let offer = walked.pop().unwrap();
    assert_eq!(offer.elsewhere, [(elsewhere, elsewhere)]);
    let jobs: Vec<Job> = offer.elsewhere_jobs().collect();
    assert_eq!(jobs.len(), 1);
    let job = &jobs[0];
    assert_eq!(
        (job.key(), job.pc(), job.frame()),
        (None, elsewhere, elsewhere)
    );
    assert_eq!(job.context(), &offer.preparation().context(elsewhere));
    // Prepared from the bytes its frame holds, under their key, it is the
    // block a fetch of that page compiles.
    let page = page_of(&words, elsewhere);
    assert!(page_matches(job, &page));
    let prepared = prepare(job, Digest::of(page.as_slice()), &page).expect("a block");
    let compiled = walked.pop().unwrap();
    assert_eq!(prepared.key(), compiled.key());
    assert_eq!(prepared.context(), compiled.context());
    assert_eq!(prepared.input(), compiled.input());
    assert_eq!(
        prepared.block().native.x86_64_code,
        compiled.block().native.x86_64_code
    );
    // Without the walk, the target is counted and left out.
    let unwalked = fetched_offer_of(&words, BASE);
    assert_eq!(unwalked.elsewhere, [(elsewhere, 0)]);
    assert_eq!(unwalked.elsewhere_jobs().count(), 0);
    let mut queue = Queue::new();
    queue.offered(&unwalked);
    assert_eq!((queue.counts().elsewhere, queue.counts().unwalked), (0, 1));
    queue.offered(&offer);
    assert_eq!((queue.counts().elsewhere, queue.counts().unwalked), (1, 1));
}

fn fetched_offer_of(words: &[u32], pc: u64) -> Offer {
    offers(words, &[pc]).pop().unwrap()
}

#[test]
fn jobs_in_other_pages_wait_behind_those_in_their_naming_blocks_page() {
    let offer = fetched_offer(BASE);
    let template = offer.ahead_jobs().next().expect("a job");
    let mut queue = Queue::new();
    queue.push(Job {
        key: None,
        pc: BASE + 0x2000,
        ..template.clone()
    });
    queue.push(Job {
        pc: BASE + 4,
        ..template.clone()
    });
    queue.push(Job {
        key: None,
        pc: BASE + 0x3000,
        ..template.clone()
    });
    let order: Vec<u64> = core::iter::from_fn(|| queue.pop().map(|job| job.pc())).collect();
    assert_eq!(order, [BASE + 4, BASE + 0x3000, BASE + 0x2000]);
    assert_eq!(queue.counts().elsewhere, 2);
    // Each lane drops its own oldest.
    for index in 0..ELSEWHERE_CAPACITY as u64 + 2 {
        queue.push(Job {
            key: None,
            pc: BASE + 0x10_0000 + index * 4,
            ..template.clone()
        });
    }
    queue.push(Job {
        pc: BASE + 8,
        ..template.clone()
    });
    assert_eq!(queue.len(), ELSEWHERE_CAPACITY + 1);
    assert_eq!(queue.counts().dropped, 2);
    assert_eq!(queue.pop().map(|job| job.pc()), Some(BASE + 8));
}
