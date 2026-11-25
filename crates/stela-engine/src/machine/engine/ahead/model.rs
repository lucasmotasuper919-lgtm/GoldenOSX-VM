//! What translation ahead would prepare and what of it runs, over real guest
//! code: the blocks a pack holds (the blocks a boot ran) in a dyld shared
//! cache's text, and the jobs their static successors make, same-page and
//! cross-page, by depth, and the blocks a linear sweep of each page a boot
//! ran prepares at load time. Run by hand: `AHEAD_PACK` names a pack (one
//! segment, no journal), `AHEAD_DSC` the directory of the same guest's
//! shared cache. An ignored test: `cargo test` never runs it.
//!
//! The pack's records name pages by the hash of their bytes, so a page of
//! the cache whose hash a record names is a page the boot ran, at the
//! cache's unslid address; its entries' offsets are the blocks. The model
//! compiles with the kernel's options (`fetch_costs`): a block's static
//! successors are its branch targets, which no option moves.
use super::super::fetch_costs::system_preparation;
use super::super::prepare::{Preparation, Prepared, window};
use super::super::{FETCH_PAGE, MAX_BLOCK_BYTES};
use crate::translations::{Digest, Record, pack::Pack};
use alloc::{string::String, vec::Vec};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::fs::FileExt;
use std::time::Instant;

/// A record's magic, schema, entry count, translator, key and length.
const RECORD_HEADER: usize = 88;

/// An executable mapping of one cache file.
struct Text {
    file: usize,
    address: u64,
    size: u64,
    offset: u64,
}

struct Cache {
    files: Vec<File>,
    texts: Vec<Text>,
}

impl Cache {
    fn open(directory: &str) -> Self {
        let mut names: Vec<String> = std::fs::read_dir(directory)
            .expect("the cache directory")
            .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
            .filter(|name| {
                let Some((_, suffix)) = name.split_once("arm64e") else {
                    return false;
                };
                suffix.is_empty()
                    || suffix
                        .strip_prefix('.')
                        .is_some_and(|digits| digits.bytes().all(|b| b.is_ascii_digit()))
            })
            .collect();
        names.sort();
        let mut files = Vec::new();
        let mut texts = Vec::new();
        for name in names {
            let mut file = File::open(std::format!("{directory}/{name}")).expect("a cache file");
            let mut header = [0u8; 0x20];
            file.read_exact(&mut header).expect("a header");
            assert!(header.starts_with(b"dyld_v1"), "{name}");
            let word = |at: usize| u32::from_le_bytes(header[at..at + 4].try_into().unwrap());
            let (at, count) = (word(0x10), word(0x14));
            let mut mappings = std::vec![0u8; count as usize * 32];
            file.seek(SeekFrom::Start(at.into())).expect("seek");
            file.read_exact(&mut mappings).expect("the mappings");
            for mapping in mappings.as_chunks::<32>().0 {
                let long = |at: usize| u64::from_le_bytes(mapping[at..at + 8].try_into().unwrap());
                let initial = u32::from_le_bytes(mapping[28..32].try_into().unwrap());
                if initial & 4 != 0 {
                    texts.push(Text {
                        file: files.len(),
                        address: long(0),
                        size: long(8),
                        offset: long(16),
                    });
                }
            }
            files.push(file);
        }
        Self { files, texts }
    }

    /// The bytes at `address`, if executable text holds them all.
    fn read(&self, address: u64, into: &mut [u8]) -> bool {
        let Some(text) = self.texts.iter().find(|text| {
            address >= text.address && address + into.len() as u64 <= text.address + text.size
        }) else {
            return false;
        };
        self.files[text.file]
            .read_exact_at(into, text.offset + address - text.address)
            .is_ok()
    }

    fn executable(&self, address: u64) -> bool {
        self.texts
            .iter()
            .any(|text| address >= text.address && address < text.address + text.size)
    }

    /// Each text page whose key `wanted` holds, by key: the page's address,
    /// or none where more than one page has its bytes.
    fn pages(&self, wanted: &HashSet<Digest>) -> HashMap<Digest, Option<u64>> {
        let mut pages = HashMap::new();
        let mut chunk = std::vec![0u8; 16 << 20];
        for text in &self.texts {
            let mut done = 0;
            while done < text.size {
                let bytes = (text.size - done).min(chunk.len() as u64) as usize;
                self.files[text.file]
                    .read_exact_at(&mut chunk[..bytes], text.offset + done)
                    .expect("text");
                for (index, page) in chunk[..bytes]
                    .as_chunks::<{ FETCH_PAGE as usize }>()
                    .0
                    .iter()
                    .enumerate()
                {
                    let key = Digest::of(page);
                    if wanted.contains(&key) {
                        let address = text.address + done + index as u64 * FETCH_PAGE;
                        pages
                            .entry(key)
                            .and_modify(|at| *at = None)
                            .or_insert(Some(address));
                    }
                }
                done += bytes as u64;
            }
        }
        pages
    }
}

/// The successors of the block at `pc`, and its source bytes; none where
/// the cache does not hold its window or it prepares no block.
fn successors(
    cache: &Cache,
    preparation: &Preparation,
    pc: u64,
    compiled: &mut u64,
) -> Option<(Vec<u64>, usize)> {
    let length = window(pc);
    let mut staging = [0u8; MAX_BLOCK_BYTES as usize];
    let source = &mut staging[..length];
    if !cache.read(pc, source) {
        return None;
    }
    preparation.admit(source);
    *compiled += 1;
    match preparation.prepare(source, pc, length) {
        Ok(Prepared::Block {
            compiled,
            successors,
            ..
        }) => Some((successors, compiled.source_bytes)),
        _ => None,
    }
}

/// Jobs by how they came: same-page successors, cross-page targets of a
/// fetched block, cross-page targets of a prepared one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
enum Kind {
    SamePage,
    CrossFromFetched,
    CrossFromPrepared,
}

#[derive(Default, Clone, Copy)]
struct Tally {
    jobs: u64,
    used: u64,
}

#[test]
#[ignore = "manual: opt-in"]
fn ahead_prepared_and_used() {
    let pack_path = std::env::var("AHEAD_PACK").expect("AHEAD_PACK");
    let cache = Cache::open(&std::env::var("AHEAD_DSC").expect("AHEAD_DSC"));
    let depth_limit: u8 = std::env::var("AHEAD_DEPTH")
        .ok()
        .and_then(|depth| depth.parse().ok())
        .unwrap_or(4);
    let mut file = File::open(&pack_path).expect("the pack");
    let mut header = std::vec![0u8; crate::translations::pack::HEADER_BYTES];
    file.read_exact(&mut header).expect("its header");
    let translator = Digest(header[16..48].try_into().unwrap());
    let prefix_bytes =
        crate::translations::pack::prefix_bytes(&header, &translator).expect("a pack header");
    let mut prefix = std::vec![0u8; prefix_bytes];
    file.read_exact_at(&mut prefix, 0).expect("its index");
    let length = Pack::length(&prefix, &translator).expect("its length") as usize;
    let mut bytes = std::vec![0u8; length];
    file.read_exact_at(&mut bytes, 0).expect("the pack");
    let pack = Pack::parse(&bytes, &translator).expect("a pack");
    let keys: HashSet<Digest> = pack.iter().map(|(key, _)| key).collect();
    let started = Instant::now();
    let pages = cache.pages(&keys);
    std::println!(
        "pack: {} records; cache text {} MiB; {} of its pages are records ({} ambiguous); {:?}",
        pack.records(),
        cache.texts.iter().map(|text| text.size).sum::<u64>() >> 20,
        pages.len(),
        pages.values().filter(|at| at.is_none()).count(),
        started.elapsed()
    );
    // The blocks the boot ran in the cache. A record of an older format
    // than this translator decodes is read for its entries' offsets alone:
    // each entry starts with its context digest and its page offset, and
    // the contexts are those the records' first entries name.
    let contexts: HashSet<[u8; 32]> = pack
        .iter()
        .filter_map(|(_, record)| record.get(RECORD_HEADER..RECORD_HEADER + 32))
        .map(|context| context.try_into().unwrap())
        .collect();
    let mut used: HashSet<u64> = HashSet::new();
    let mut scanned = 0;
    for (key, record) in pack.iter() {
        let Some(Some(page)) = pages.get(&key) else {
            continue;
        };
        if let Ok(record) = Record::decode_checked(record, &translator, &key, false) {
            used.extend(
                record
                    .entries()
                    .iter()
                    .map(|entry| page + u64::from(entry.offset)),
            );
            continue;
        }
        scanned += 1;
        let mut at = RECORD_HEADER;
        while at + 34 <= record.len() {
            if contexts.contains(&record[at..at + 32]) {
                let offset = u16::from_le_bytes([record[at + 32], record[at + 33]]);
                if u64::from(offset) < FETCH_PAGE && offset.is_multiple_of(4) {
                    used.insert(page + u64::from(offset));
                }
                at += 34;
            } else {
                at += 1;
            }
        }
    }
    let failed = scanned;
    let used_pages: HashSet<u64> = used.iter().map(|pc| pc & !(FETCH_PAGE - 1)).collect();
    std::println!(
        "used: {} blocks in {} pages ({failed} records of an older format scanned)",
        used.len(),
        used_pages.len()
    );
    let preparation = system_preparation();
    // Every block a boot ran is fetched, so compiled once: its jobs.
    let mut compiled = 0;
    let mut queue: VecDeque<(u64, u8, Kind)> = VecDeque::new();
    let mut fetched: Vec<u64> = used.iter().copied().collect();
    fetched.sort_unstable();
    let started = Instant::now();
    for &pc in &fetched {
        let Some((targets, _)) = successors(&cache, &preparation, pc, &mut compiled) else {
            continue;
        };
        let page = pc & !(FETCH_PAGE - 1);
        for target in targets {
            if target == pc || !target.is_multiple_of(4) || !cache.executable(target) {
                continue;
            }
            let kind = match target & !(FETCH_PAGE - 1) == page {
                true => Kind::SamePage,
                false => Kind::CrossFromFetched,
            };
            queue.push_back((target, 1, kind));
        }
    }
    let fetch_time = started.elapsed();
    // Each job's block once, at the depth and kind it is first queued
    // with: a queue's repeat filter and the store's presence check leave
    // out the others.
    let mut seen: HashMap<u64, (u8, Kind)> = HashMap::new();
    let mut tallies: HashMap<(Kind, u8), Tally> = HashMap::new();
    let started = Instant::now();
    let mut prepared = 0;
    while let Some((pc, depth, kind)) = queue.pop_front() {
        if seen.contains_key(&pc) {
            continue;
        }
        seen.insert(pc, (depth, kind));
        let tally = tallies.entry((kind, depth)).or_default();
        tally.jobs += 1;
        tally.used += u64::from(used.contains(&pc));
        if depth >= depth_limit {
            continue;
        }
        let Some((targets, _)) = successors(&cache, &preparation, pc, &mut prepared) else {
            continue;
        };
        let page = pc & !(FETCH_PAGE - 1);
        for target in targets {
            if target == pc || !target.is_multiple_of(4) || !cache.executable(target) {
                continue;
            }
            let next = match (target & !(FETCH_PAGE - 1) == page, kind) {
                (true, kind) => kind,
                (false, _) => Kind::CrossFromPrepared,
            };
            queue.push_back((target, depth + 1, next));
        }
    }
    let job_time = started.elapsed();
    std::println!(
        "compiled {compiled} fetched blocks in {fetch_time:?}, {prepared} job blocks in {job_time:?}"
    );
    std::println!("kind depth: jobs, of them used, used/jobs; cumulative used of all used blocks");
    let mut rows: Vec<_> = tallies.into_iter().collect();
    rows.sort_by_key(|&((kind, depth), _)| (kind, depth));
    let mut reached = 0;
    for ((kind, depth), tally) in rows {
        reached += tally.used;
        std::println!(
            "{kind:?} {depth}: {} {} {:.3}; {:.3}",
            tally.jobs,
            tally.used,
            tally.used as f64 / tally.jobs.max(1) as f64,
            reached as f64 / used.len().max(1) as f64
        );
    }
    // Load time: a linear sweep of each page the boot ran, each block
    // where the one before ends.
    let (mut swept, mut hits, mut sweep_compiled) = (0u64, 0u64, 0u64);
    let started = Instant::now();
    for &page in &used_pages {
        let mut pc = page;
        while pc < page + FETCH_PAGE {
            swept += 1;
            hits += u64::from(used.contains(&pc));
            let step = successors(&cache, &preparation, pc, &mut sweep_compiled)
                .map_or(4, |(_, bytes)| bytes.max(4) as u64);
            pc += step;
        }
    }
    std::println!(
        "load-time sweep of the used pages: {swept} blocks, {hits} used ({:.3}), {:.3} of the used \
         blocks; {:?}",
        hits as f64 / swept.max(1) as f64,
        hits as f64 / used.len().max(1) as f64,
        started.elapsed()
    );
}
