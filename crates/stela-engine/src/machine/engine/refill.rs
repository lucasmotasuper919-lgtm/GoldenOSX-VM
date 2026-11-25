//! A new translator's first boot refills its store from the pack the
//! translator before it wrote, so that an update of the engine does not
//! cost a cold session.
//!
//! The previous pack's records name pages by the hash of their bytes and
//! hold, for each block, its page offset, its context and the admitted
//! words its compiler was given. A context names every compile input
//! (`Preparation::context`), so an entry whose context a preparation of
//! this boot gives is a block this boot compiles from those words in that
//! context. A PE whose guest waits takes the records hottest first (the
//! most uses a boot made of each, [`Pack::uses`]) in bounded steps and
//! compiles each such entry with this translator at a page of its half of
//! the address space; the block's stored form is page-relative, as any
//! entry's, so it serves every page of that content. The board's kept
//! forms take it ([`super::rebase`]), so a fetch of the page this boot
//! relocates it instead of compiling (after the checks every kept form
//! passes: the admitted input equal to the fetch's bytes), and the store
//! keeps it for this translator's pack ([`super::Translations::refilled`]).
//!
//! The refill decides only which blocks are compiled, never their code:
//! each is what this translator compiles from those words in that context.
//! An entry whose words do not hold the whole block this translator
//! compiles (its analysis reaches their end short of the page's), whose
//! context no preparation of this boot gives, or that has no stored form
//! is left to its fetch.
use super::prepare::{Preparation, Prepared, window};
use super::stored::stored_entry;
use super::{Devices, Host, MAX_BLOCK_BYTES, Machine};
use crate::translations::pack::{DecodeScratch, DecodedRecord, Pack, Store};
use crate::translations::{Digest, Entry, Record};
use alloc::{sync::Arc, vec::Vec};

/// A page of each half of the address space, where the refill compiles an
/// entry of a context made for that half: entries are page-relative.
const PAGES: [u64; 2] = [0x10_0000_0000, 0xffff_fff0_0000_0000];

/// What a refill did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RefillCounts {
    /// Records taken from the previous pack.
    pub records: u64,
    /// Records whose bytes did not decode.
    pub undecoded: u64,
    /// Their entries.
    pub entries: u64,
    /// Entries compiled and kept.
    pub kept: u64,
    /// Entries whose context no preparation of this boot gives.
    pub unmatched: u64,
    /// Entries whose words do not hold this translator's block, or whose
    /// block has no stored form.
    pub refused: u64,
}

/// The previous translator's pack, its records hottest first.
pub struct Refill<'a> {
    translator: Digest,
    source: Store<'a>,
    order: Vec<Digest>,
    next: usize,
    scratch: DecodeScratch,
    counts: RefillCounts,
}

impl<'a> Refill<'a> {
    /// The refill from `segments`, the previous translator's packs, oldest
    /// first: its records by the most uses a boot made of each, the newer
    /// segment's first among equals; a key in several segments once.
    pub fn new(translator: Digest, segments: Vec<Pack<'a>>) -> Self {
        let mut keys: Vec<(u8, usize, Digest)> = segments
            .iter()
            .enumerate()
            .flat_map(|(segment, pack)| pack.uses().map(move |(key, uses)| (uses, segment, key)))
            .collect();
        keys.sort_unstable_by(|a, b| b.cmp(a));
        let mut seen = alloc::collections::BTreeSet::new();
        let order = keys
            .into_iter()
            .filter_map(|(_, _, key)| seen.insert(key).then_some(key))
            .collect();
        Self {
            translator,
            source: Store::over(translator, segments),
            order,
            next: 0,
            scratch: DecodeScratch::default(),
            counts: RefillCounts::default(),
        }
    }

    /// The next record, decoded, or none when every record was taken.
    pub fn next_record(&mut self) -> Option<(Digest, Record)> {
        while let Some(&key) = self.order.get(self.next) {
            self.next += 1;
            self.counts.records += 1;
            let record = self
                .source
                .stored(&key)
                .and_then(|stored| {
                    DecodedRecord::decode(stored, &self.translator, &key, &mut self.scratch).ok()
                })
                .and_then(|decoded| decoded.into_record().record().ok());
            match record {
                Some(record) => return Some((key, record)),
                None => self.counts.undecoded += 1,
            }
        }
        None
    }

    /// Whether every record was taken.
    pub fn done(&self) -> bool {
        self.next >= self.order.len()
    }

    /// The records left.
    pub fn left(&self) -> usize {
        self.order.len() - self.next.min(self.order.len())
    }

    pub fn counts(&self) -> RefillCounts {
        self.counts
    }

    pub fn counts_mut(&mut self) -> &mut RefillCounts {
        &mut self.counts
    }
}

/// A word the analysis reads past the end of an entry's words, which ends
/// no block: NOP.
const PAST: u32 = 0xd503_201f;

/// The stored form this translator gives the block `entry` holds, compiled
/// with `preparation` from its admitted words at its offset of `page`:
/// none where the words do not hold the whole block (with a word past
/// them, its analysis goes on into that word) or the block has no stored
/// form. Words that reach the page's end are the whole window.
pub(super) fn refill_entry(preparation: &Preparation, page: u64, entry: &Entry) -> Option<Entry> {
    let pc = page + u64::from(entry.offset);
    let words = entry.input.as_slice();
    let room = window(pc);
    if words.is_empty() || words.len() > room || !words.len().is_multiple_of(4) {
        return None;
    }
    let mut staging = [0u8; MAX_BLOCK_BYTES as usize + 4];
    let whole = words.len() == room;
    let length = if whole { words.len() } else { words.len() + 4 };
    let source = &mut staging[..length];
    source[..words.len()].copy_from_slice(words);
    if !whole {
        source[words.len()..].copy_from_slice(&PAST.to_le_bytes());
    }
    match preparation.prepare_window(source, pc, length, true)? {
        Ok(Prepared::Block {
            compiled,
            input,
            consumed,
            ..
        }) if consumed <= words.len() => stored_entry(
            &compiled,
            &source[..input],
            page,
            pc,
            entry.context,
            preparation,
        ),
        Ok(_) | Err(_) => None,
    }
}

impl<H: Host, D: Devices> Machine<H, D> {
    /// Whether a refill goes on: the heap is not short, and the board's
    /// kept forms have room for refilled entries (the refill takes at most
    /// half their bound, so that it never pushes out what this boot keeps
    /// of its own).
    pub fn refill_room(&self) -> bool {
        !self.host.memory_low() && self.code.rebases.under_half()
    }

    /// Compile the entries of `record`, the previous translator's record of
    /// the page `key` names, whose contexts this PE's preparations give,
    /// and keep each block's stored form in the board's kept forms and in
    /// the store.
    pub fn refill(&mut self, key: &Digest, record: &Record, counts: &mut RefillCounts) {
        for entry in record.entries() {
            counts.entries += 1;
            let Some((preparation, page)) = self.refill_preparation(&entry.context) else {
                counts.unmatched += 1;
                continue;
            };
            let Some(stored) = refill_entry(&preparation, page, entry) else {
                counts.refused += 1;
                continue;
            };
            counts.kept += 1;
            self.code
                .rebases
                .keep(key, entry.offset, &entry.context, Some(stored.clone()));
            if let Some(translations) = self.host.translations() {
                translations.refilled(key, stored);
            }
        }
    }

    /// A preparation of this PE whose context for a page of one half of the
    /// address space is `context`, and that page.
    fn refill_preparation(&self, context: &Digest) -> Option<(Arc<Preparation>, u64)> {
        self.preparations.values().find_map(|preparation| {
            PAGES
                .iter()
                .find(|&&page| preparation.context(page) == *context)
                .map(|&page| (preparation.clone(), page))
        })
    }
}

#[cfg(test)]
mod tests;
