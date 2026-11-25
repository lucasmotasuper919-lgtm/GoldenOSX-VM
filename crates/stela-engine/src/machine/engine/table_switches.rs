//! Native TTBR0 switches (`x86_64::table_switch`): the entries an
//! engine-path TTBR0 write leaves for native code, and what the engine
//! settles after native code switched.
//!
//! An engine-path write that changes the translating regime's TTBR0 and
//! nothing else, in the upper half's code, keeps an entry for the value
//! written and one for the value it replaced, each with the lower half's
//! lookup context of the state the write ran in, which the entry holds
//! under (that state's upper context, the same before and after), and the
//! host's set of that TTBR0. A native write of either value later switches
//! there without an exit, as long as the entries' generation holds: it
//! moves whenever a lookup context may have gone or the translation
//! controls changed ([`table_switch::Table::forget`]).
//!
//! What the engine's own write does besides is either done by the thunk
//! (the register, the lookup contexts, the closed cells, the transition
//! records' contexts, the published translating word, the host's CR3 and
//! the prediction barrier) or settled at the next return from native code
//! ([`Machine::settle_native_switches`]): the table generation, the device
//! read generation, the prediction context, the statistics and the host's
//! regime, which the host already follows from its own sets.
use super::*;

/// Why a TTBR0 write that changes the register took the engine path, as
/// its thunk would have decided ([`Machine::note_switch_miss`]): no thunk
/// could serve it (the table-base cell closed, or code of the lower half
/// or of a non-global upper half); no entry for the value was ever kept;
/// one was, but the table forgot it since; it was kept under another upper
/// lookup context; the host's sets were closed or the entry's set was
/// given to another context since; the entry's set is of another class
/// than the current one (other rights or lower TCR fields); or the entry
/// held, so no thunk ran (none installed yet, or the block ran in the
/// engine) or mail was pending.
pub mod miss {
    pub const CLOSED: usize = 0;
    pub const NEVER: usize = 1;
    pub const FORGOTTEN: usize = 2;
    pub const CONTEXT: usize = 3;
    pub const SET: usize = 4;
    pub const CLASS: usize = 5;
    pub const OTHER: usize = 6;
    pub const REASONS: usize = 7;
}

/// What a TTBR0 write leaves of the state before it, for its entry back.
pub(super) struct SwitchPoint {
    controls: mmu::Controls,
    value: u64,
    contexts: [u64; 2],
}

impl<H: Host, D: Devices> Machine<H, D> {
    /// The state an MSR TTBR0 at the current PC switches from, when its
    /// code is the upper half's: a switch thunk may only serve that code.
    pub(super) fn switch_point(&mut self) -> Option<SwitchPoint> {
        if chain::half(self.cpu.pc) != 1 || self.private_fetches[1] {
            return None;
        }
        let cell = self.mirrors[TABLE_BASE_KEPT];
        if cell.readable == 0 {
            return None;
        }
        // SAFETY: an open table-base cell's value is the address of the
        // register's storage in this machine's registers.
        let value = unsafe { (cell.value as *const u64).read() };
        Some(SwitchPoint {
            controls: self.cpu.translating_controls(self.cpu.regime()),
            value,
            contexts: self.switch_contexts(),
        })
    }

    /// The lookup contexts a fetch in the current state takes.
    pub(super) fn switch_contexts(&mut self) -> [u64; 2] {
        let variant = self.current_variant(false, false);
        let el = self.cpu.pstate.el | u8::from(self.cpu.pstate.guarded) << 4;
        self.contexts(el, variant)
    }

    /// The engine is about to complete `MSR TTBR0, Xt` (`word`) from the
    /// state `from` ([`Self::switch_point`]): count why its thunk would not
    /// have, when the write changes the register ([`miss`]).
    pub(super) fn note_switch_miss(&mut self, word: u32, from: Option<&SwitchPoint>) {
        let rt = (word & 0x1f) as usize;
        let value = self.cpu.state.x.get(rt).copied().unwrap_or(0);
        let reason = match from {
            None => {
                let current = self.cpu.translating_controls(self.cpu.regime()).ttbr0;
                if value == current {
                    return;
                }
                miss::CLOSED
            }
            Some(from) if from.value == value => return,
            Some(from) => self.switch_miss_reason(value, from.contexts[1]),
        };
        self.statistics.switch_misses[reason] += 1;
    }

    fn switch_miss_reason(&self, value: u64, upper: u64) -> usize {
        let table = &self.mirrors.table_switch;
        let mut kept = table.entries.iter().filter(|entry| entry.value == value);
        let Some(entry) = kept
            .clone()
            .find(|entry| entry.generation == table.generation)
        else {
            return if kept.next().is_some() {
                miss::FORGOTTEN
            } else {
                miss::NEVER
            };
        };
        if entry.upper != upper {
            return miss::CONTEXT;
        }
        if table.sets == 0 {
            return miss::OTHER;
        }
        // SAFETY: the host's sets live as long as the machine runs on it
        // (`Host::native_sets`); only this PE writes them, and not now.
        let sets = unsafe { &*(table.sets as *const table_switch::Sets) };
        let set = entry.set as usize;
        if sets.open == 0
            || set >= table_switch::SETS
            || sets.generations[set] != entry.set_generation
        {
            return miss::SET;
        }
        let current =
            (sets.current.wrapping_sub(sets.base) >> 12) as usize / table_switch::SET_ROOTS;
        match sets.classes.get(current) {
            Some(&class) if class == sets.classes[set] => miss::OTHER,
            _ => miss::CLASS,
        }
    }

    /// An engine-path write of the register at `storage` that left the
    /// state `from` had but TTBR0: keep entries for the value written and
    /// the one it replaced.
    pub(super) fn keep_table_switches(&mut self, from: SwitchPoint, storage: u64) {
        // SAFETY: as in `switch_point`: the cell the write opened.
        let value = unsafe { (storage as *const u64).read() };
        let to = self.cpu.translating_controls(self.cpu.regime());
        let others = mmu::Controls {
            ttbr0: from.controls.ttbr0,
            ..to
        };
        if value == from.value
            || to.ttbr0 == from.controls.ttbr0
            || others != from.controls
            || self.private_fetches[1]
        {
            return;
        }
        let contexts = self.switch_contexts();
        if contexts[1] != from.contexts[1] {
            return;
        }
        let sets = self.host.native_sets().map(|(at, _)| at);
        self.mirrors.table_switch.sets = sets.unwrap_or(0);
        for (value, controls, contexts) in [
            (value, to, contexts),
            (from.value, from.controls, from.contexts),
        ] {
            let (set, set_generation) = match sets {
                Some(_) => match self.host.native_set(&controls) {
                    Some(set) => set,
                    None => continue,
                },
                None => (0, 0),
            };
            let (translating_at, pending_at, translating) = self.switch_words(&controls);
            self.mirrors.table_switch.translating = translating_at;
            self.mirrors.table_switch.pending = pending_at;
            let salt = self.chain.salts_of(contexts)[0];
            self.mirrors.table_switch.keep(table_switch::Entry {
                generation: 0,
                value,
                upper: contexts[1],
                lower: contexts[0],
                salt: u64::from(salt),
                translating,
                set,
                set_generation,
            });
        }
    }

    /// Native code switched TTBR0 ([`table_switch::Table::switches`]):
    /// what the engine's write would have done besides, done once for all
    /// of them. The thunk issued the prediction barrier the change of
    /// context needed and moved the transition records' contexts.
    pub(super) fn settle_native_switches(&mut self) {
        let switches = self.mirrors.table_switch.take_switches();
        if switches == 0 {
            return;
        }
        self.statistics.table_base_changes += switches;
        self.statistics.native_table_switches += switches;
        if self.host.isolates_predictions() {
            self.statistics.prediction_isolations += switches;
        }
        self.prediction_context = Some(PredictionContext::of(&self.cpu));
        self.advance_table_generation();
        self.sync_host_regime(step_timing::SyncSource::TableBase);
    }
}
