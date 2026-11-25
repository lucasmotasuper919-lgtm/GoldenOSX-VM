//! Stable chaining storage with one local writer and atomic interruption epochs.
use alloc::sync::Arc;
use core::{
    cell::{Cell, UnsafeCell},
    marker::PhantomData,
    sync::atomic::{AtomicU64, Ordering},
};
use stela_compiler::x86_64::chain::{self, Account, ChainControl, itlb_salt};

/// The salts of `contexts` in an ITLB of the set mask in `masks`.
fn salts(contexts: [u64; 2], masks: [u32; 2]) -> [u32; 2] {
    contexts.map(|context| itlb_salt(context, masks[1]))
}

/// By its hub inbox even when no Machine owns the PE. The hub exposes
/// only the interruption epoch; all other fields belong to its admitted Machine.
#[repr(C, align(64))]
pub(super) struct Storage {
    control: UnsafeCell<ChainControl>,
}
// SAFETY: shared storage exposes only an aligned atomic epoch. Other fields
// are accessed only through the unique Owner admitted by the inbox.
unsafe impl Send for Storage {}
// SAFETY: the same single-writer/atomic-epoch contract applies to shared access.
unsafe impl Sync for Storage {}
impl Storage {
    pub(super) fn new() -> Arc<Self> {
        Arc::new(Self {
            control: UnsafeCell::new(ChainControl::default()),
        })
    }
    pub(super) fn interrupt_epoch(&self) -> &AtomicU64 {
        // SAFETY: the control has a naturally aligned epoch field.
        // Every Rust access is atomic; native thunks perform aligned loads.
        unsafe {
            AtomicU64::from_ptr(core::ptr::addr_of_mut!(
                (*self.control.get()).interrupt_epoch
            ))
        }
    }
    /// The control's address, where a notifier posts ([`Self::notify`],
    /// a fast IPI's target).
    pub(super) fn control(&self) -> u64 {
        self.control.get() as u64
    }
    pub(super) fn notify(&self) {
        // Publish every source write before making the corresponding epoch
        // visible. Exhaustion stays permanently interrupted rather than wraps.
        // SAFETY: the epoch and quiet words, which every writer but the
        // owner's acknowledgment accesses atomically.
        unsafe { chain::interrupt(self.control.get()) };
    }
}

/// The sole local-control writer. It can move between threads with its Machine,
/// but cannot be shared; a hub keeps Storage without granting this role.
pub(super) struct Owner {
    storage: Arc<Storage>,
    /// The budget the last native call began with ([`Self::begin`]).
    entered: u64,
    /// The account the last native call left ([`Self::finish`]), which
    /// native code keeps in a host register (`chain::ACCOUNT`).
    left: Account,
    _single_writer: PhantomData<Cell<()>>,
    _attachment: Option<Arc<super::parallel::Attached>>,
}
impl Default for Owner {
    fn default() -> Self {
        Self {
            storage: Storage::new(),
            entered: 0,
            left: Account(0),
            _single_writer: PhantomData,
            _attachment: None,
        }
    }
}
impl Owner {
    /// # Safety
    /// The caller holds the unique live attachment claim for this storage,
    /// until this Owner can no longer access it. No old native code or other
    /// writer may run through its previous owner's installed chain thunks.
    pub(super) unsafe fn from_claimed(
        storage: Arc<Storage>,
        attachment: Arc<super::parallel::Attached>,
    ) -> Self {
        Self {
            storage,
            entered: 0,
            left: Account(0),
            _single_writer: PhantomData,
            _attachment: Some(attachment),
        }
    }
    /// The account the last native call left, or [`Self::begin`] set.
    #[cfg(test)]
    pub(super) fn account(&self) -> Account {
        self.left
    }
    pub(super) fn address(&self) -> u64 {
        self.storage.control.get() as u64
    }

    /// Reset every local field, keeping the inbox's monotonic notification epoch.
    pub(super) fn initialize(
        &mut self,
        table: u64,
        itlb: u64,
        contexts: [u64; 2],
        masks: [u32; 2],
    ) {
        let control = self.storage.control.get();
        // SAFETY: this Owner is the sole local writer, with no native call
        // in progress. Disjoint raw writes never borrow the shared epoch.
        unsafe {
            core::ptr::addr_of_mut!((*control).block).write(0);
            core::ptr::addr_of_mut!((*control).contexts).write(contexts);
            core::ptr::addr_of_mut!((*control).salts).write(salts(contexts, masks));
            core::ptr::addr_of_mut!((*control).masks).write(masks);
            core::ptr::addr_of_mut!((*control).table).write(table);
            core::ptr::addr_of_mut!((*control).itlb).write(itlb);
            core::ptr::addr_of_mut!((*control).hits).write(0);
            core::ptr::addr_of_mut!((*control).linked).write(0);
            chain::acknowledge(control, 0);
        }
        self.left = Account(0);
    }
    /// Acknowledge only the Acquire snapshot taken before a successful drain.
    /// MAX is never acknowledged, so exhausted notification state stays sticky.
    pub(super) fn acknowledge_interrupt(&mut self, epoch: u64) {
        // SAFETY: only the unique owner writes the observed field outside its
        // synchronous native invocation; notifiers touch the epoch and quiet
        // words atomically.
        unsafe { chain::acknowledge(self.storage.control.get(), epoch) };
    }
    /// The epoch the owner last acknowledged: every notification up to it
    /// has been drained ([`Self::acknowledge_interrupt`]).
    pub(super) fn acknowledged_interrupt(&self) -> u64 {
        // SAFETY: only this owner writes the observed field, outside native
        // execution; notifiers touch the separate epoch.
        unsafe {
            core::ptr::addr_of!((*self.storage.control.get()).observed_interrupt_epoch).read()
        }
    }
    /// Whether a notification came since the owner last acknowledged one,
    /// as a native GEXIT compares them.
    pub(super) fn interrupt_moved(&self) -> bool {
        self.storage.interrupt_epoch().load(Ordering::Acquire) != self.acknowledged_interrupt()
    }
    pub(super) fn set_table(&mut self, table: u64, itlb: u64, masks: [u32; 2]) {
        // SAFETY: the sole owner writes these non-budget fields outside native execution.
        unsafe {
            let control = self.storage.control.get();
            core::ptr::addr_of_mut!((*control).table).write(table);
            core::ptr::addr_of_mut!((*control).itlb).write(itlb);
            core::ptr::addr_of_mut!((*control).masks).write(masks);
            let contexts = core::ptr::addr_of!((*control).contexts).read();
            core::ptr::addr_of_mut!((*control).salts).write(salts(contexts, masks));
        }
    }
    /// The block table [`Self::set_table`] gave, or [`Self::set_block_table`].
    pub(super) fn table(&self) -> u64 {
        let control = self.storage.control.get();
        // SAFETY: the sole owner reads this non-budget field outside native execution.
        unsafe { core::ptr::addr_of!((*control).table).read() }
    }
    /// Probe the block table at `table` with slot mask `mask` from now on,
    /// keeping the ITLB's.
    pub(super) fn set_block_table(&mut self, table: u64, mask: u32) {
        // SAFETY: the sole owner writes these non-budget fields outside native execution.
        unsafe {
            let control = self.storage.control.get();
            core::ptr::addr_of_mut!((*control).table).write(table);
            core::ptr::addr_of_mut!((*control).masks[0]).write(mask);
        }
    }
    /// This PE's stamps and the board's stamped cells, which stamped tails
    /// and the cell probe use ([`stela_compiler::x86_64::chain::stamped_tail`]).
    pub(super) fn set_stamps(&mut self, stamps: u64, cells: u64) {
        // SAFETY: the sole owner writes these non-budget fields outside native execution.
        unsafe {
            let control = self.storage.control.get();
            core::ptr::addr_of_mut!((*control).stamps).write(stamps);
            core::ptr::addr_of_mut!((*control).cells).write(cells);
        }
    }
    /// The stamps and stamped cells [`Self::set_stamps`] gave, or zeros.
    pub(super) fn stamps(&self) -> (u64, u64) {
        let control = self.storage.control.get();
        // SAFETY: the sole owner reads these non-budget fields outside native execution.
        unsafe {
            (
                core::ptr::addr_of!((*control).stamps).read(),
                core::ptr::addr_of!((*control).cells).read(),
            )
        }
    }
    /// What this PE's stamped tails and stamping cell probe counted
    /// ([`chain::StampCount`]).
    pub(super) fn stamp_counts(&self) -> [u64; chain::STAMP_COUNTS] {
        // SAFETY: the sole owner reads this non-budget field outside native execution.
        let stamps = unsafe { core::ptr::addr_of!((*self.storage.control.get()).stamps).read() };
        // SAFETY: `set_stamps` stored 0 or the board's table's stamps for
        // this PE, which live as long as the board.
        unsafe { chain::stamp_counts(stamps) }
    }
    pub(super) fn masks(&self) -> [u32; 2] {
        // SAFETY: only this Owner writes the masks, outside native execution.
        unsafe { core::ptr::addr_of!((*self.storage.control.get()).masks).read() }
    }
    pub(super) fn contexts(&self) -> [u64; 2] {
        // SAFETY: only this Owner or its synchronous native call can write contexts.
        unsafe { core::ptr::addr_of!((*self.storage.control.get()).contexts).read() }
    }
    /// The ITLB salts of [`Self::contexts`].
    pub(super) fn salts(&self) -> [u32; 2] {
        // SAFETY: only this Owner or its synchronous native call can write salts.
        unsafe { core::ptr::addr_of!((*self.storage.control.get()).salts).read() }
    }
    /// The ITLB salts `contexts` would take as [`Self::contexts`].
    pub(super) fn salts_of(&self, contexts: [u64; 2]) -> [u32; 2] {
        salts(contexts, self.masks())
    }
    pub(super) fn set_contexts(&mut self, contexts: [u64; 2]) {
        // SAFETY: the sole owner writes this non-budget field outside native execution.
        unsafe {
            let control = self.storage.control.get();
            core::ptr::addr_of_mut!((*control).contexts).write(contexts);
        }
        let masks = self.masks();
        // SAFETY: as above.
        unsafe {
            core::ptr::addr_of_mut!((*self.storage.control.get()).salts)
                .write(salts(contexts, masks));
        }
    }
    /// Prepare a native call into block `block` that allows `budget`
    /// transfers, at most [`chain::MAX_BUDGET`]: the account it starts
    /// with, for `GuestState::account`.
    pub(super) fn begin(&mut self, block: usize, budget: u64) -> u64 {
        let budget = budget.min(chain::MAX_BUDGET);
        let control = self.storage.control.get();
        // SAFETY: the sole owner prepares these disjoint fields before its
        // synchronous native call; remote writers touch only the epoch.
        unsafe {
            core::ptr::addr_of_mut!((*control).block).write(block as u64);
            core::ptr::addr_of_mut!((*control).hits).write(0);
            core::ptr::addr_of_mut!((*control).linked).write(0);
            core::ptr::addr_of_mut!((*control).heat).write(0);
            core::ptr::addr_of_mut!((*control).sampled).write(0);
        }
        self.entered = budget;
        self.left = Account::new(budget);
        self.left.0
    }

    /// The account the native call [`Self::begin`] prepared left: its
    /// `GuestState::account`, or the account register a fault's capture
    /// holds (`x86_64::pinned::settled_account`).
    pub(super) fn finish(&mut self, account: u64) {
        self.left = Account(account);
    }
    /// Let one native run keep up to `bound` paired frames, checked against
    /// the retire epoch at `epoch` ([`ChainControl::retire_epoch`]), which
    /// must live through every native call until the next change: 0 and 0
    /// push none.
    pub(super) fn set_frames(&mut self, bound: u32, epoch: u64) {
        let control = self.storage.control.get();
        // SAFETY: the sole owner writes these non-budget fields outside native execution.
        unsafe {
            core::ptr::addr_of_mut!((*control).frame_room).write(if epoch == 0 {
                0
            } else {
                bound.saturating_mul(chain::FRAME_BYTES as u32)
            });
            core::ptr::addr_of_mut!((*control).retire_epoch).write(epoch);
        }
    }
    /// What paired frames did since the storage was made
    /// ([`chain::PairCount`]).
    pub(super) fn pairs(&self) -> [u64; chain::PAIR_COUNTS] {
        // SAFETY: native code writes them only within a call of this owner.
        unsafe { core::ptr::addr_of!((*self.storage.control.get()).pairs).read() }
    }
    /// The transfers the last native call made.
    pub(super) fn chained(&self) -> u64 {
        self.entered.saturating_sub(self.left.budget())
    }
    /// Whether the last native call used its whole budget: its last
    /// transfer, or one after that, which found none left and stopped it.
    pub(super) fn exhausted(&self) -> bool {
        self.left.budget() == 0
    }
    /// The block that ended the last native call, the instructions of the
    /// blocks it chained out of, the transfers lookup probes made and the
    /// others. Native code counts no other transfers: each spends one of the
    /// budget ([`Self::chained`]).
    pub(super) fn completed(&self) -> (u64, u64, u64, u64) {
        let control = self.storage.control.get();
        // SAFETY: the synchronous native call has returned; no other owner
        // touches these fields, including while a peer posts an interruption.
        let (block, hits) = unsafe {
            (
                core::ptr::addr_of!((*control).block).read(),
                core::ptr::addr_of!((*control).hits).read(),
            )
        };
        let retired = self.left.retired();
        (block, retired, hits, self.chained().saturating_sub(hits))
    }
    /// The transfers the last native call's cell probes made
    /// ([`ChainControl::heat`], whose low 32 bits they count in and
    /// [`Self::begin`] zeroes), and the cell of the last one they named, or
    /// 0 ([`ChainControl::sampled`]).
    pub(super) fn cell_probes(&self) -> (u64, u64) {
        let control = self.storage.control.get();
        // SAFETY: the synchronous native call has returned; only this
        // owner's native code writes the fields.
        let (heat, sampled) = unsafe {
            (
                core::ptr::addr_of!((*control).heat).read(),
                core::ptr::addr_of!((*control).sampled).read(),
            )
        };
        (u64::from(heat as u32), sampled)
    }
    #[cfg(test)]
    pub(super) fn weak_storage(&self) -> alloc::sync::Weak<Storage> {
        Arc::downgrade(&self.storage)
    }
    #[cfg(test)]
    pub(super) fn snapshot(&self) -> ChainControl {
        let (block, _, hits, linked) = self.completed();
        let control = self.storage.control.get();
        ChainControl {
            // SAFETY: the word other cores write, read atomically.
            quiet: chain::Quiet(unsafe {
                AtomicU64::from_ptr(core::ptr::addr_of_mut!((*control).quiet.0))
                    .load(Ordering::Acquire)
            }),
            block,
            hits,
            linked,
            contexts: self.contexts(),
            salts: salts(self.contexts(), self.masks()),
            masks: self.masks(),
            // SAFETY: the sole owner observes a non-budget field outside entry.
            table: unsafe { core::ptr::addr_of!((*control).table).read() },
            // SAFETY: as for `table`.
            itlb: unsafe { core::ptr::addr_of!((*control).itlb).read() },
            interrupt_epoch: self.storage.interrupt_epoch().load(Ordering::Acquire),
            // SAFETY: only the unique owner writes the observed epoch.
            observed_interrupt_epoch: unsafe {
                core::ptr::addr_of!((*control).observed_interrupt_epoch).read()
            },
            // SAFETY: as for `table`.
            stamps: unsafe { core::ptr::addr_of!((*control).stamps).read() },
            // SAFETY: as for `table`.
            cells: unsafe { core::ptr::addr_of!((*control).cells).read() },
            // SAFETY: as for `table`; native code adds to it only within a call.
            heat: unsafe { core::ptr::addr_of!((*control).heat).read() },
            // SAFETY: as for `heat`.
            sampled: unsafe { core::ptr::addr_of!((*control).sampled).read() },
            loop_entries: self.loops().0,
            loop_iterated: self.loops().1,
            // SAFETY: as for `table`.
            frame_room: unsafe { core::ptr::addr_of!((*control).frame_room).read() },
            reserved: 0,
            // SAFETY: as for `table`.
            retire_epoch: unsafe { core::ptr::addr_of!((*control).retire_epoch).read() },
            pairs: self.pairs(),
        }
    }

    /// [`ChainControl::loop_entries`] and [`ChainControl::loop_iterated`]
    /// since the storage was made.
    pub(super) fn loops(&self) -> (u64, u64) {
        let control = self.storage.control.get();
        // SAFETY: the sole owner reads them outside a native call, the only
        // time native code writes them.
        unsafe {
            (
                core::ptr::addr_of!((*control).loop_entries).read(),
                core::ptr::addr_of!((*control).loop_iterated).read(),
            )
        }
    }
}
