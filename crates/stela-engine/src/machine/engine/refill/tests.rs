use super::super::stored::tests::MemoryStore;
use super::super::tests::{BASE, machine};
use super::super::{Fetched, Offer, StoreSource};
use super::*;
use crate::translations::pack::{Layout, Piece};

/// `cbz x0, +12; add x1, x1, #1; add x2, x2, #2; add x3, x3, #3; svc #0`.
const WORDS: [u32; 5] = [
    0xb400_0060,
    0x9100_0421,
    0x9100_0842,
    0x9100_0c63,
    0xd400_0001,
];

/// The id of the translator before this one.
const PREVIOUS: Digest = Digest([7; 32]);

/// The offer the fetch of `pc` makes in a machine with a store.
fn fetched_offer(pc: u64) -> Offer {
    let mut fetching = machine(&WORDS);
    fetching.host.store = Some(MemoryStore::default());
    assert!(matches!(
        fetching.fetch_block(pc, false),
        Ok(Fetched::Block(_))
    ));
    fetching.host.store.as_mut().unwrap().offers.pop().unwrap()
}

/// A pack of the previous translator holding `entries` under `key`.
fn previous_pack(key: Digest, entries: &[Entry]) -> Vec<u8> {
    let mut record = Record::new(PREVIOUS, key);
    for entry in entries {
        record.insert(entry.clone());
    }
    Layout::new(&PREVIOUS, alloc::vec![(key, Piece::New(record.encode()))]).to_vec()
}

#[test]
fn a_previous_translators_block_is_compiled_again_and_its_fetch_relocates_it() {
    let offer = fetched_offer(BASE);
    let entry = offer.entry().expect("a stored form");
    let bytes = previous_pack(offer.key(), core::slice::from_ref(&entry));
    let pack = Pack::parse(&bytes, &PREVIOUS).expect("a pack");
    let mut refill = Refill::new(PREVIOUS, alloc::vec![pack]);
    let (key, record) = refill.next_record().expect("its record");
    assert_eq!(key, offer.key());
    assert!(refill.next_record().is_none());
    assert!(refill.done());
    let mut receiver = machine(&WORDS);
    receiver.host.store = Some(MemoryStore::default());
    // The preparation a fetch at this PE's level makes.
    receiver.current_preparation();
    let mut counts = RefillCounts::default();
    receiver.refill(&key, &record, &mut counts);
    assert_eq!((counts.entries, counts.kept), (1, 1));
    // This translator's stored form, the one its own compile keeps.
    let store = receiver.host.store.as_mut().unwrap();
    assert_eq!(store.refilled, [(key, entry.clone())]);
    // The fetch relocates it instead of compiling.
    assert!(matches!(
        receiver.fetch_block(BASE, false),
        Ok(Fetched::Block(_))
    ));
    let level = usize::from(receiver.cpu.pstate.el != 0);
    assert_eq!(receiver.statistics.compiled, 0);
    assert_eq!(
        receiver.statistics.store_sources[level][StoreSource::Rebased as usize],
        1
    );
}

#[test]
fn entries_of_a_context_no_preparation_gives_or_cut_short_are_left_to_their_fetch() {
    let offer = fetched_offer(BASE);
    let entry = offer.entry().expect("a stored form");
    let foreign = Entry {
        context: Digest([9; 32]),
        ..entry.clone()
    };
    let mut receiver = machine(&WORDS);
    receiver.host.store = Some(MemoryStore::default());
    receiver.current_preparation();
    let mut record = Record::new(PREVIOUS, offer.key());
    record.insert(foreign);
    let mut counts = RefillCounts::default();
    receiver.refill(&offer.key(), &record, &mut counts);
    assert_eq!((counts.entries, counts.unmatched, counts.kept), (1, 1, 0));
    assert!(receiver.host.store.as_ref().unwrap().refilled.is_empty());
    // Words that end inside the block: this translator's analysis reaches
    // their end, so they do not hold its block.
    let preparation = receiver.current_preparation();
    let page = PAGES
        .into_iter()
        .find(|&page| preparation.context(page) == entry.context)
        .expect("a page of the entry's half");
    let short = Entry {
        input: entry.input[..8].to_vec(),
        ..entry.clone()
    };
    assert!(refill_entry(&preparation, page, &short).is_none());
    assert_eq!(refill_entry(&preparation, page, &entry), Some(entry));
}

#[test]
fn a_refill_takes_the_newer_segments_records_first_and_each_key_once() {
    let offer = fetched_offer(BASE);
    let entry = offer.entry().expect("a stored form");
    let (older, newer) = (Digest([1; 32]), Digest([2; 32]));
    let first = previous_pack(older, core::slice::from_ref(&entry));
    let second = previous_pack(newer, core::slice::from_ref(&entry));
    let both = previous_pack(older, core::slice::from_ref(&entry));
    let segments = [&first, &second, &both]
        .map(|bytes| Pack::parse(bytes, &PREVIOUS).expect("a pack"))
        .to_vec();
    let mut refill = Refill::new(PREVIOUS, segments);
    assert_eq!(refill.left(), 2);
    let keys: Vec<Digest> =
        core::iter::from_fn(|| refill.next_record().map(|(key, _)| key)).collect();
    assert_eq!(keys, [older, newer]);
    assert_eq!(refill.counts().records, 2);
}

#[test]
fn a_store_keeps_refilled_entries_within_half_its_added_bound() {
    let offer = fetched_offer(BASE);
    let entry = offer.entry().expect("a stored form");
    let bytes = entry.retained_bytes();
    let (mut store, _) = Store::new(Digest([3; 32]), None);
    store.set_bounds(crate::translations::pack::Bounds {
        added: 5 * bytes,
        decoded: 1 << 20,
    });
    for key in 0..4 {
        store.refilled(Digest([key; 32]), entry.clone());
    }
    // Two fit in half of five entries' bytes; the blocks this boot
    // compiles keep the rest.
    assert_eq!(store.take_added().len(), 2);
}
