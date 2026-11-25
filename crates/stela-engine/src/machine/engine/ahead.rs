//! Translation ahead of execution: the static successors of a block a fetch
//! compiled, or installed as a block prepared ahead, are prepared by PEs
//! whose guest waits, before any PE fetches them.
//!
//! A fetch that compiles a block offers it to the store
//! ([`super::Translations::offer`]) with the entry PCs of its static
//! successors, and the frames of those in other pages whose walk the
//! fetching PE's fetch cache holds ([`Offer::elsewhere`]). The store turns
//! them into [`Job`]s and keeps them in a bounded [`Queue`], newest first,
//! the other pages' behind the block's own page's. A PE whose guest waits
//! takes jobs in bounded steps: it reads the job's frame, checks that its
//! bytes still hash to the page key (a job for another page takes the key
//! its bytes hash to then), and [`prepare`]s the block as a fetch would
//! compile it ([`super::Preparation::prepare`] over the same window). The
//! store keeps the result among its shared blocks, where a fetch finds it
//! after its pack record ([`super::Translations::prepared`]) and installs
//! it after the same checks as any shared block (key, PC, context, admitted
//! bytes). The first PE that installs one offers it, so the pack keeps only
//! blocks that ran, and its successors become jobs as a fetched block's do.
//!
//! A job decides only which block is prepared, never its code: a prepared
//! block is the block the engine compiles at that PC from those bytes in
//! that context. Successors of prepared blocks are queued only up to
//! [`DEPTH`] steps from a block that ran.
use super::prepare::{Preparation, Prepared, window};
use super::stored::{Emission, Offer};
use super::{FETCH_PAGE, MAX_BLOCK_BYTES};
use crate::translations::{Digest, PAGE_BYTES};
use alloc::{boxed::Box, collections::VecDeque, sync::Arc, vec};

/// Static successor steps from a block that ran a job may lie: one. Of
/// the blocks a boot ran in the shared cache (the host model in `model.rs`,
/// a pack of a desktop boot: 1.0 million blocks in 27 thousand pages), 81 %
/// of the same-page successors one step from them ran, 16 % at two steps,
/// 2 % at three; of the targets in other pages, 78 % at one step, 3 % at
/// two. One step reaches 89 % of the blocks that ran; steps two to four
/// reach 2.5 % more with 1.35 times the jobs, 94 % of those never run.
pub const DEPTH: u8 = 1;
/// Jobs a queue holds for blocks in their naming block's page; when full,
/// the oldest go.
pub const CAPACITY: usize = 4096;
/// Jobs a queue holds for blocks in other pages, served when none of the
/// others wait: a tenth of a block's successors lie in other pages (the
/// model's 115 thousand to 994 thousand).
pub const ELSEWHERE_CAPACITY: usize = 1024;
/// Slots of the queue's filter of recent jobs.
const SEEN: usize = 1 << 14;

/// A block to prepare: the page key (none for a page the naming block did
/// not read: its bytes give it), guest page and frame of its fetch page,
/// its entry PC, and the context and preparation of the block that names
/// it.
#[derive(Debug, Clone)]
pub struct Job {
    key: Option<Digest>,
    page: u64,
    frame: u64,
    pc: u64,
    context: Digest,
    preparation: Arc<Preparation>,
    depth: u8,
    /// The naming block's [`Offer::quiet`].
    quiet: bool,
}

impl Job {
    /// The key of its page as the naming block read it; none for another
    /// page, whose key is the hash of the bytes its frame holds when the
    /// job is taken.
    pub fn key(&self) -> Option<&Digest> {
        self.key.as_ref()
    }
    pub fn frame(&self) -> u64 {
        self.frame
    }
    pub fn pc(&self) -> u64 {
        self.pc
    }
    pub fn context(&self) -> &Digest {
        &self.context
    }
    /// Its entry PC's offset in its fetch page.
    pub fn offset(&self) -> u16 {
        (self.pc - self.page) as u16
    }
    fn tag(&self) -> u64 {
        let mut tag = match &self.key {
            Some(key) => u64::from_le_bytes(key.0[..8].try_into().expect("8 bytes")),
            None => self.frame.rotate_left(29),
        };
        tag ^= u64::from_le_bytes(self.context.0[..8].try_into().expect("8 bytes")).rotate_left(17);
        tag ^= self.pc.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        tag | 1
    }
}

impl Offer {
    /// Jobs for its static successors in its own fetch page, other than its
    /// own entry, while it lies fewer than [`DEPTH`] steps from a block that
    /// ran. A store with no frame (an offer made off any fetch) has none.
    pub fn ahead_jobs(&self) -> impl Iterator<Item = Job> + '_ {
        let open = self.depth < DEPTH && self.frame != 0;
        self.successors
            .iter()
            .copied()
            .filter(move |&target| {
                open && target != self.pc
                    && target & !(FETCH_PAGE - 1) == self.page
                    && target.is_multiple_of(4)
            })
            .map(|pc| Job {
                key: Some(self.key),
                page: self.page,
                frame: self.frame,
                pc,
                context: self.context,
                preparation: self.emission.preparation.clone(),
                depth: self.depth + 1,
                quiet: self.quiet,
            })
    }

    /// Jobs for its static successors in other pages whose frame the
    /// offering PE's fetch cache held ([`Offer::elsewhere`]), on the same
    /// terms, in the context its preparation gives their page.
    pub fn elsewhere_jobs(&self) -> impl Iterator<Item = Job> + '_ {
        let open = self.depth < DEPTH && self.frame != 0;
        self.elsewhere
            .iter()
            .copied()
            .filter(move |&(pc, frame)| open && frame != 0 && pc.is_multiple_of(4))
            .map(|(pc, frame)| {
                let page = pc & !(FETCH_PAGE - 1);
                Job {
                    key: None,
                    page,
                    frame,
                    pc,
                    context: self.emission.preparation.context(page),
                    preparation: self.emission.preparation.clone(),
                    depth: self.depth + 1,
                    quiet: self.quiet,
                }
            })
    }
}

/// What became of the jobs a queue took and handed out
/// ([`Queue::note`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    /// Jobs taken in.
    pub queued: u64,
    /// Of them, jobs for blocks in other pages than their naming block's.
    pub elsewhere: u64,
    /// Successors in other pages left out: the offering PE's fetch cache
    /// held no walk of their page.
    pub unwalked: u64,
    /// Jobs left out: the same block was queued recently.
    pub repeated: u64,
    /// Jobs that went unprepared because the queue was full.
    pub dropped: u64,
    /// Blocks prepared and kept for the PEs.
    pub prepared: u64,
    /// Jobs whose block the store held already (pack or shared).
    pub present: u64,
    /// Jobs whose frame no longer held the page's bytes.
    pub stale: u64,
    /// Jobs whose window prepares no block (a refused word, an access
    /// check, FP trapped).
    pub refused: u64,
}

/// How one job ended, for [`Counts`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Prepared,
    Present,
    Stale,
    Refused,
}

/// Jobs waiting for a PE, newest first: those in their naming block's page
/// (at most [`CAPACITY`]), then those in other pages (at most
/// [`ELSEWHERE_CAPACITY`]).
pub struct Queue {
    jobs: VecDeque<Job>,
    elsewhere: VecDeque<Job>,
    /// Tags of recent jobs, direct mapped: a block queued again while its
    /// tag is here is left out.
    seen: Box<[u64]>,
    counts: Counts,
}

impl Default for Queue {
    fn default() -> Self {
        Self::new()
    }
}

impl Queue {
    pub fn new() -> Self {
        Self {
            jobs: VecDeque::with_capacity(CAPACITY),
            elsewhere: VecDeque::with_capacity(ELSEWHERE_CAPACITY),
            seen: vec![0; SEEN].into_boxed_slice(),
            counts: Counts::default(),
        }
    }

    /// Take in the jobs `offer` names ([`Offer::ahead_jobs`],
    /// [`Offer::elsewhere_jobs`]). Allocates nothing.
    pub fn offered(&mut self, offer: &Offer) {
        offer.ahead_jobs().for_each(|job| self.push(job));
        if offer.depth < DEPTH && offer.frame != 0 {
            self.counts.unwalked += offer
                .elsewhere
                .iter()
                .filter(|&&(_, frame)| frame == 0)
                .count() as u64;
        }
        offer.elsewhere_jobs().for_each(|job| self.push(job));
    }

    /// Take `job` in; the oldest of its lane goes when the lane is full.
    /// Allocates nothing.
    pub fn push(&mut self, job: Job) {
        let tag = job.tag();
        let slot = (tag as usize >> 1) % SEEN;
        if self.seen[slot] == tag {
            self.counts.repeated += 1;
            return;
        }
        self.seen[slot] = tag;
        let (lane, capacity) = match job.key {
            Some(_) => (&mut self.jobs, CAPACITY),
            None => {
                self.counts.elsewhere += 1;
                (&mut self.elsewhere, ELSEWHERE_CAPACITY)
            }
        };
        if lane.len() == capacity {
            lane.pop_front();
            self.counts.dropped += 1;
        }
        lane.push_back(job);
        self.counts.queued += 1;
    }

    /// The newest job in its naming block's page, else the newest in
    /// another page.
    pub fn pop(&mut self) -> Option<Job> {
        self.jobs.pop_back().or_else(|| self.elsewhere.pop_back())
    }

    pub fn len(&self) -> usize {
        self.jobs.len() + self.elsewhere.len()
    }

    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty() && self.elsewhere.is_empty()
    }

    pub fn note(&mut self, outcome: Outcome) {
        let counter = match outcome {
            Outcome::Prepared => &mut self.counts.prepared,
            Outcome::Present => &mut self.counts.present,
            Outcome::Stale => &mut self.counts.stale,
            Outcome::Refused => &mut self.counts.refused,
        };
        *counter += 1;
    }

    pub fn counts(&self) -> Counts {
        self.counts
    }
}

/// Whether `page`, the bytes of the job's frame as read now, are the bytes
/// its key names; for a job without one, the bytes it takes.
pub fn page_matches(job: &Job, page: &[u8; PAGE_BYTES]) -> bool {
    job.key.is_none_or(|key| Digest::of(page) == key)
}

/// The block `job` names, prepared from `page`, the bytes of its frame,
/// which hash to `key` ([`page_matches`]): what a fetch of its PC in its
/// context compiles from them, as an offer of a block prepared ahead.
pub fn prepare(job: &Job, key: Digest, page: &[u8; PAGE_BYTES]) -> Result<Offer, Outcome> {
    let offset = usize::from(job.offset());
    if job.preparation.context(job.page) != job.context {
        return Err(Outcome::Refused);
    }
    let length = window(job.pc);
    let mut staging = [0u8; MAX_BLOCK_BYTES as usize];
    let source = &mut staging[..length];
    source.copy_from_slice(&page[offset..offset + length]);
    job.preparation.admit(source);
    match job.preparation.prepare(source, job.pc, length) {
        Ok(Prepared::Block {
            compiled,
            input,
            successors,
            ..
        }) => Ok(Offer {
            key,
            page: job.page,
            pc: job.pc,
            context: job.context,
            emission: Arc::new(Emission::new(
                source[..input].to_vec(),
                compiled,
                job.preparation.clone(),
            )),
            successors,
            elsewhere: alloc::vec::Vec::new(),
            quiet: job.quiet,
            frame: job.frame,
            depth: job.depth,
        }),
        Ok(Prepared::Checked(_) | Prepared::FloatingPointTrap) | Err(_) => Err(Outcome::Refused),
    }
}

#[cfg(test)]
mod model;
#[cfg(test)]
mod tests;
