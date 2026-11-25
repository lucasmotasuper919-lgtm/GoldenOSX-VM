//! Where a PE's time outside compiled guest code goes, by the kind of exit
//! that ended each run of guest code, in any boot. A run of guest code ends
//! in an exit to the engine; the time from that return to the next entry is
//! the exit's handling, the lookup or translation of the next block and the
//! waits on other PEs it met. The host reads the TSC at every entry and
//! return already (its share of time in guest code), so this keeps two sums
//! per kind from those same readings and times nothing new per exit. What
//! is no exit's cost is left out: the core's halts and the host's work
//! between slices pause the pending exit's time.
use alloc::{format, string::String, vec::Vec};

use stela_compiler::block::ExitKind;

/// The kinds counted: the compiler's block exit kinds in their order, then
/// a fault taken in guest code.
pub const KINDS: usize = ExitKind::Amx as usize + 2;
/// A fault in guest code, after the block exit kinds.
pub const FAULT: usize = KINDS - 1;
/// System instructions and device register accesses, whose exits also
/// begin a PE's waits.
pub const SYSTEM: usize = ExitKind::System as usize;
/// The kinds' names in the heartbeat line.
pub const NAMES: [&str; KINDS] = [
    "fallthrough",
    "direct",
    "indirect",
    "call",
    "return",
    "trap",
    "pointer-auth",
    "clock",
    "thread-register",
    "floating-point",
    "scalable",
    "system",
    "amx",
    "fault",
];
/// The kinds a PE's part of the line shows, the costliest first.
const SHOWN: usize = 4;

/// One core's sums: for each kind the runs that ended in it and the ticks
/// from their returns to the next entries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExitTimes {
    /// `[runs, ticks]` by kind.
    sums: [[u64; 2]; KINDS],
    /// The kind of the exit whose time is running.
    kind: usize,
    /// An exit's time is running since its return.
    pending: bool,
    /// Whether the time counts now, and since when: not while paused.
    since: Option<u64>,
    /// What the pending exit took before the last pause.
    spent: u64,
}

impl ExitTimes {
    pub const fn new() -> Self {
        Self {
            sums: [[0; 2]; KINDS],
            kind: 0,
            pending: false,
            since: None,
            spent: 0,
        }
    }

    /// A run of guest code entered at `entered` returned at `returned`
    /// with an exit of `kind` (one past the last is a fault's): the exit
    /// before it took until `entered`, and this one's time starts.
    pub fn run(&mut self, entered: u64, returned: u64, kind: usize) {
        if self.pending {
            let open = self.since.map_or(0, |since| entered.wrapping_sub(since));
            let sum = &mut self.sums[self.kind];
            sum[0] += 1;
            sum[1] += self.spent + open;
        }
        self.kind = kind.min(FAULT);
        self.pending = true;
        self.since = Some(returned);
        self.spent = 0;
    }

    /// From `now` the core does what is no exit's cost (it halts, or does
    /// the host's work between slices) until [`Self::resume`].
    pub fn pause(&mut self, now: u64) {
        if let Some(since) = self.since.take() {
            self.spent += now.wrapping_sub(since);
        }
    }

    /// The pending exit's time counts again from `now`.
    pub fn resume(&mut self, now: u64) {
        if self.pending && self.since.is_none() {
            self.since = Some(now);
        }
    }

    /// `[runs, ticks]` by kind so far.
    pub fn sums(&self) -> [[u64; 2]; KINDS] {
        self.sums
    }
}

/// One PE's interval: `[runs, ticks]` by kind, and the ticks it polled for
/// a wake without halting, which its waits' exits (System ones) include.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PeExits {
    pub kinds: [[u64; 2]; KINDS],
    pub polled: u64,
}

impl PeExits {
    /// Its kinds with runs as `(ticks, runs, kind)`, the costliest first.
    fn ranked(&self) -> Vec<(u64, u64, usize)> {
        let mut ranked: Vec<_> = self
            .kinds
            .iter()
            .enumerate()
            .filter(|(_, sum)| sum[0] != 0)
            .map(|(kind, sum)| {
                let ticks = if kind == SYSTEM {
                    sum[1].saturating_sub(self.polled)
                } else {
                    sum[1]
                };
                (ticks, sum[0], kind)
            })
            .collect();
        ranked.sort_unstable_by(|a, b| b.cmp(a));
        ranked
    }
}

/// `PES EXIT TIME kind=runs/ms 0:system=154821/1210,return=128665/420,…
/// all:system=…`: for each PE the kinds of exit that cost it the most time
/// outside guest code since the last beat, each with the runs of guest code
/// that ended in it and the milliseconds from those returns to the next
/// entries (handling, the next block's lookup or translation, waits on
/// other PEs; not its halts, idle polling or host service), then every kind
/// over all PEs. `ticks_per_ms` is the ticks' rate.
pub fn pes_exit_time(ticks_per_ms: u64, pes: &[PeExits]) -> String {
    let item = |&(ticks, runs, kind): &(u64, u64, usize)| {
        format!("{}={runs}/{}", NAMES[kind], ticks / ticks_per_ms.max(1))
    };
    let mut line = String::from("PES EXIT TIME kind=runs/ms");
    let mut all = PeExits::default();
    for (pe, exits) in pes.iter().enumerate() {
        let shown: Vec<_> = exits.ranked().iter().take(SHOWN).map(item).collect();
        line.push_str(&format!(" {pe}:{}", shown.join(",")));
        for (sum, kind) in all.kinds.iter_mut().zip(&exits.kinds) {
            sum[0] += kind[0];
            sum[1] += kind[1];
        }
        // Each PE's polling comes off its own System time.
        let polled = exits.polled.min(exits.kinds[SYSTEM][1]);
        all.kinds[SYSTEM][1] -= polled;
    }
    let shown: Vec<_> = all.ranked().iter().map(item).collect();
    line.push_str(&format!(" all:{}", shown.join(",")));
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_names_follow_the_compilers_exit_kinds() {
        for (kind, name) in [
            (ExitKind::Fallthrough, "fallthrough"),
            (ExitKind::Direct, "direct"),
            (ExitKind::Indirect, "indirect"),
            (ExitKind::Call, "call"),
            (ExitKind::Return, "return"),
            (ExitKind::Trap, "trap"),
            (ExitKind::PointerAuthentication, "pointer-auth"),
            (ExitKind::Clock, "clock"),
            (ExitKind::ThreadRegister, "thread-register"),
            (ExitKind::FloatingPoint, "floating-point"),
            (ExitKind::ScalableNumerical, "scalable"),
            (ExitKind::System, "system"),
            (ExitKind::Amx, "amx"),
        ] {
            assert_eq!(NAMES[kind as usize], name);
        }
        assert_eq!(NAMES[FAULT], "fault");
    }

    #[test]
    fn an_exit_takes_from_its_return_to_the_next_entry() {
        let mut times = ExitTimes::new();
        // The first run has no exit before it.
        times.run(100, 1_000, 4);
        assert_eq!(times.sums(), [[0; 2]; KINDS]);
        // Its return exit took until the next entry; a System exit follows.
        times.run(1_300, 5_000, SYSTEM);
        let mut expected = [[0; 2]; KINDS];
        expected[4] = [1, 300];
        assert_eq!(times.sums(), expected);
        times.run(5_050, 6_000, 4);
        expected[SYSTEM] = [1, 50];
        assert_eq!(times.sums(), expected);
        // A fault, and a kind past the table's, count as faults.
        times.run(6_010, 7_000, FAULT);
        times.run(7_500, 8_000, 99);
        times.run(8_100, 9_000, 0);
        expected[4] = [2, 310];
        expected[FAULT] = [2, 600];
        assert_eq!(times.sums(), expected);
    }

    #[test]
    fn halts_and_host_work_are_no_exits_time() {
        let mut times = ExitTimes::new();
        times.run(0, 1_000, SYSTEM);
        // The PE waits: 40 ticks of handling, a halt, 10 more on the wake.
        times.pause(1_040);
        times.resume(900_000);
        // The slice ends for host work, then the guest runs again.
        times.pause(900_010);
        times.resume(950_000);
        times.run(950_025, 960_000, 1);
        let mut expected = [[0; 2]; KINDS];
        expected[SYSTEM] = [1, 75];
        assert_eq!(times.sums(), expected);
        // A pause with nothing pending, twice, and a resume without one,
        // change nothing.
        let mut idle = ExitTimes::new();
        idle.resume(5);
        idle.pause(10);
        idle.pause(20);
        idle.run(100, 200, 0);
        idle.pause(210);
        idle.pause(260);
        idle.resume(300);
        idle.resume(400);
        idle.run(305, 500, 0);
        let mut expected = [[0; 2]; KINDS];
        expected[0] = [1, 15];
        assert_eq!(idle.sums(), expected);
    }

    #[test]
    fn the_line_names_each_pes_costliest_kinds_and_all_of_them_together() {
        let mut busy = PeExits::default();
        busy.kinds[SYSTEM] = [154_821, 1_210_000];
        busy.kinds[4] = [128_665, 420_000];
        busy.kinds[3] = [67_596, 380_000];
        busy.kinds[1] = [44_687, 200_000];
        busy.kinds[2] = [19_369, 90_000];
        busy.kinds[FAULT] = [900, 4_000];
        // Mostly waiting: its System exits' time is mostly idle polling.
        let mut waiting = PeExits {
            polled: 700_000,
            ..PeExits::default()
        };
        waiting.kinds[SYSTEM] = [9_000, 760_000];
        waiting.kinds[4] = [2_000, 9_000];
        assert_eq!(
            pes_exit_time(1_000, &[busy, waiting, PeExits::default()]),
            "PES EXIT TIME kind=runs/ms \
             0:system=154821/1210,return=128665/420,call=67596/380,direct=44687/200 \
             1:system=9000/60,return=2000/9 2: \
             all:system=163821/1270,return=130665/429,call=67596/380,direct=44687/200,\
             indirect=19369/90,fault=900/4"
        );
        // Polling past the System time leaves none, not a wrapped number.
        waiting.polled = 9_000_000;
        assert_eq!(
            pes_exit_time(1_000, &[waiting]),
            "PES EXIT TIME kind=runs/ms 0:return=2000/9,system=9000/0 \
             all:return=2000/9,system=9000/0"
        );
        assert_eq!(pes_exit_time(0, &[]), "PES EXIT TIME kind=runs/ms all:");
    }
}
