//! What the guest's session does while its screen is black or shows only a
//! pointer: one `GUEST SESSION` line a minute, so that a log of such a boot
//! says whether a process crashed, a daemon restarts in a loop, WindowServer
//! stopped presenting, or macOS is only slow. The processes come from the
//! guest's console: those the StelaGPU kext says opened the accelerator,
//! those the kernel refused a core dump for (each is a crash), and those it
//! names with a process ID, whose new IDs are restarts.
use alloc::{string::String, vec::Vec};
use core::fmt;

/// The most processes kept in each list; the rest are counted, not named.
const NAMES: usize = 24;
/// The longest process name kept: the kext passes up to 63 bytes, the
/// kernel's own lines 16.
const NAME_BYTES: usize = 63;

/// The kext's line for a process's first open of the accelerator, after
/// the process's name.
const OPENED: &str = ": opened the accelerator";
const KEXT: &str = "StelaGPU: ";
/// The kernel's line for a process that crashed where core dumps are
/// refused; it ends without a newline, so the next message follows it on
/// the same console line.
const CRASHED: &str = "AMFI: Denying core dump for pid ";
/// A kernel line's naming of the process that asked: `NAME (PID)` follows.
const ASKED: &str = "requested by: ";

/// How many minutes each get a line; later a line is written only when the
/// processes changed, and each [`QUIET_MINUTES`] otherwise.
pub const EVERY_MINUTES: u64 = 30;
pub const QUIET_MINUTES: u64 = 10;

/// A process the console named with a process ID.
#[derive(Debug)]
struct Seen {
    name: String,
    /// Its highest process ID so far: IDs rise, so a higher one is a new
    /// process of that name.
    highest: u32,
    starts: u32,
}

/// What the guest's console said of its processes since the boot began.
#[derive(Debug, Default)]
pub struct Console {
    clients: Vec<String>,
    crashes: Vec<(String, u32)>,
    seen: Vec<Seen>,
    /// Processes not named for want of room in a list.
    unlisted: u32,
    /// How often any of the above changed.
    changes: u64,
}

/// `NAME (PID)` at the start of `text`, where a name may hold spaces and
/// brackets.
fn name_and_pid(text: &str) -> Option<(&str, u32)> {
    let mut from = 0;
    while let Some(at) = text[from..].find(" (") {
        let name = &text[..from + at];
        if name.len() > NAME_BYTES {
            return None;
        }
        let rest = &text[from + at + 2..];
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        if digits != 0 && rest[digits..].starts_with(')') {
            return Some((name, rest[..digits].parse().ok()?));
        }
        from += at + 2;
    }
    None
}

/// `PID (NAME)` at the start of `text`.
fn pid_and_name(text: &str) -> Option<(u32, &str)> {
    let digits = text.bytes().take_while(u8::is_ascii_digit).count();
    let pid = text[..digits].parse().ok()?;
    let name = text[digits..].strip_prefix(" (")?;
    let name = &name[..name.find(')')?];
    (!name.is_empty() && name.len() <= NAME_BYTES).then_some((pid, name))
}

impl Console {
    pub const fn new() -> Self {
        Self {
            clients: Vec::new(),
            crashes: Vec::new(),
            seen: Vec::new(),
            unlisted: 0,
            changes: 0,
        }
    }

    /// How often the processes changed: new clients, crashes and starts.
    pub fn changes(&self) -> u64 {
        self.changes
    }

    /// Reads one console line.
    pub fn line(&mut self, text: &str) {
        if let Some((_, said)) = text.split_once(KEXT)
            && let Some((name, _)) = said.split_once(OPENED)
            && !name.is_empty()
            && name.len() <= NAME_BYTES
            && !self.clients.iter().any(|client| client == name)
        {
            if self.clients.len() < NAMES {
                self.clients.push(String::from(name));
                self.changes += 1;
            } else {
                self.unlisted = self.unlisted.saturating_add(1);
            }
        }
        for (at, _) in text.match_indices(CRASHED) {
            if let Some((pid, name)) = pid_and_name(&text[at + CRASHED.len()..]) {
                self.crashed(name);
                self.saw(name, pid);
            }
        }
        for (at, _) in text.match_indices(ASKED) {
            if let Some((name, pid)) = name_and_pid(&text[at + ASKED.len()..])
                && !name.is_empty()
            {
                self.saw(name, pid);
            }
        }
    }

    fn crashed(&mut self, name: &str) {
        self.changes += 1;
        if let Some((_, count)) = self.crashes.iter_mut().find(|(known, _)| known == name) {
            *count = count.saturating_add(1);
        } else if self.crashes.len() < NAMES {
            self.crashes.push((String::from(name), 1));
        } else {
            self.unlisted = self.unlisted.saturating_add(1);
        }
    }

    fn saw(&mut self, name: &str, pid: u32) {
        if let Some(seen) = self.seen.iter_mut().find(|seen| seen.name == name) {
            if pid > seen.highest {
                seen.highest = pid;
                seen.starts = seen.starts.saturating_add(1);
                self.changes += 1;
            }
        } else if self.seen.len() < NAMES {
            self.seen.push(Seen {
                name: String::from(name),
                highest: pid,
                starts: 1,
            });
        } else if let Some(once) = self.seen.iter().position(|seen| seen.starts == 1) {
            // A process seen once gives its place to a new name: the ones
            // that restart are the ones a log wants.
            self.seen.remove(once);
            self.seen.push(Seen {
                name: String::from(name),
                highest: pid,
                starts: 1,
            });
        } else {
            self.unlisted = self.unlisted.saturating_add(1);
        }
    }

    fn crashes_of(&self, name: &str) -> u32 {
        self.crashes
            .iter()
            .find(|(known, _)| known == name)
            .map_or(0, |&(_, count)| count)
    }
}

/// One minute's line.
pub struct Minute<'a> {
    pub console: &'a Console,
    /// The accelerator connections open now, if it could be asked.
    pub open: Option<usize>,
    /// The frames presented in the last `seconds`.
    pub presented: u64,
    pub seconds: u64,
    /// Seconds since input last reached the guest, if any did.
    pub last_input: Option<u64>,
}

impl fmt::Display for Minute<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let console = self.console;
        f.write_str("GUEST SESSION clients=[")?;
        for (index, client) in console.clients.iter().enumerate() {
            write!(f, "{}{client}", if index == 0 { "" } else { ", " })?;
        }
        f.write_str("] open=")?;
        match self.open {
            Some(open) => write!(f, "{open}")?,
            None => f.write_str("?")?,
        }
        write!(
            f,
            " presented_{}s={} crashes=",
            self.seconds, self.presented
        )?;
        // Each client with its count, then the others that crashed.
        let clients = console
            .clients
            .iter()
            .map(|client| (client.as_str(), console.crashes_of(client)));
        let others = console
            .crashes
            .iter()
            .filter(|(name, _)| !console.clients.contains(name))
            .map(|(name, count)| (name.as_str(), *count));
        let mut any = false;
        for (name, count) in clients.chain(others) {
            write!(f, "{}{name}:{count}", if any { "," } else { "" })?;
            any = true;
        }
        if !any {
            f.write_str("none")?;
        }
        f.write_str(" restarts=")?;
        any = false;
        for seen in console.seen.iter().filter(|seen| seen.starts > 1) {
            write!(
                f,
                "{}{}:{}",
                if any { "," } else { "" },
                seen.name,
                seen.starts - 1
            )?;
            any = true;
        }
        if !any {
            f.write_str("none")?;
        }
        match self.last_input {
            Some(seconds) => write!(f, " last_input_s={seconds}")?,
            None => f.write_str(" last_input_s=never")?,
        }
        if console.unlisted != 0 {
            write!(f, " unlisted={}", console.unlisted)?;
        }
        Ok(())
    }
}

/// A minute that ended: the seconds its frame count covers, and whether it
/// gets a line.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Tick {
    pub seconds: u64,
    pub line: bool,
}

/// When the lines are written: one for each of the first [`EVERY_MINUTES`]
/// minutes, then one for a minute in which the processes changed and one
/// each [`QUIET_MINUTES`], so that a boot of days keeps its log.
#[derive(Debug, Default)]
pub struct Schedule {
    /// The last minute that ended, and the last that got a line.
    minute: u64,
    written: u64,
    changes: u64,
}

impl Schedule {
    pub const fn new() -> Self {
        Self {
            minute: 0,
            written: 0,
            changes: 0,
        }
    }

    /// The seconds since macOS started at which the next minute ends.
    pub fn next(&self) -> u64 {
        (self.minute + 1) * 60
    }

    /// The minute that ended by `seconds` since macOS started, if one did
    /// since the last call; `changes` is [`Console::changes`] now.
    pub fn tick(&mut self, seconds: u64, changes: u64) -> Option<Tick> {
        let minute = seconds / 60;
        if minute <= self.minute {
            return None;
        }
        let covered = (minute - self.minute) * 60;
        self.minute = minute;
        let line = minute <= EVERY_MINUTES
            || changes != self.changes
            || minute - self.written >= QUIET_MINUTES;
        if line {
            self.written = minute;
            self.changes = changes;
        }
        Some(Tick {
            seconds: covered,
            line,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{format, string::ToString};

    fn line(console: &Console, presented: u64, last_input: Option<u64>) -> String {
        Minute {
            console,
            open: Some(3),
            presented,
            seconds: 60,
            last_input,
        }
        .to_string()
    }

    #[test]
    fn a_boot_that_said_nothing_has_an_empty_line() {
        let mut console = Console::new();
        console.line("apfs_vfsop_unmount:2996: all done.  going home.  (numMountedAPFSVolumes 0)");
        console.line("StelaGPU: this GPU has no universal buffer compression");
        assert_eq!(console.changes(), 0);
        let minute = Minute {
            console: &console,
            open: None,
            presented: 0,
            seconds: 120,
            last_input: None,
        };
        assert_eq!(
            minute.to_string(),
            "GUEST SESSION clients=[] open=? presented_120s=0 crashes=none restarts=none \
             last_input_s=never"
        );
    }

    #[test]
    fn clients_are_the_processes_the_kext_says_opened_the_accelerator() {
        let mut console = Console::new();
        console.line("StelaGPU: WindowServer: opened the accelerator (type 5)");
        console.line("StelaGPU: Language Chooser: opened the accelerator (type 5)");
        console.line("StelaGPU: WindowServer: opened the accelerator (type 6)");
        console.line("StelaGPU: Dock: accelerator open (type 5) refused: 0xe00002c7");
        assert_eq!(console.changes(), 2);
        assert_eq!(
            line(&console, 1440, Some(7)),
            "GUEST SESSION clients=[WindowServer, Language Chooser] open=3 presented_60s=1440 \
             crashes=WindowServer:0,Language Chooser:0 restarts=none last_input_s=7"
        );
    }

    #[test]
    fn a_refused_core_dump_counts_a_crash_though_the_next_message_shares_its_line() {
        let mut console = Console::new();
        console.line("StelaGPU: WindowServer: opened the accelerator (type 5)");
        console.line(
            "AMFI: Denying core dump for pid 412 (WindowServer)\"AppleSEPKeyStore\":pid:455,\
             sel:7:8596: unexpected session uid: -1 requested by: loginwindow (455)",
        );
        console.line("AMFI: Denying core dump for pid 471 (WindowServer)StelaGPU: x");
        console.line("AMFI: Denying core dump for pid 480 (com.apple.dock.e)");
        // Not a crash line's shape.
        console.line("AMFI: Denying core dump for pid (Dock)");
        console.line("AMFI: Denying core dump for pid 12 Dock");
        assert_eq!(
            line(&console, 0, Some(300)),
            "GUEST SESSION clients=[WindowServer] open=3 presented_60s=0 \
             crashes=WindowServer:2,com.apple.dock.e:1 restarts=WindowServer:1 last_input_s=300"
        );
    }

    #[test]
    fn a_daemon_under_new_process_ids_is_counted_as_restarting() {
        let mut console = Console::new();
        for pid in [92, 92, 246, 246, 277, 293, 277] {
            console.line(&format!(
                "\"AppleSEPKeyStore\":pid:{pid},sel:7:8596: unexpected session uid: -1 requested \
                 by: bluetoothd ({pid})"
            ));
        }
        console.line("x requested by: keybagd (61)");
        console.line("x requested by: Language Chooser (301)");
        console.line("x requested by: (5)");
        console.line("x requested by: odd (name) (7) and requested by: odd (name) (9)");
        assert_eq!(console.changes(), 4);
        assert_eq!(
            line(&console, 60, None),
            "GUEST SESSION clients=[] open=3 presented_60s=60 crashes=none \
             restarts=bluetoothd:3,odd (name):1 last_input_s=never"
        );
    }

    #[test]
    fn full_lists_count_what_they_do_not_name_and_keep_the_restarting() {
        let mut console = Console::new();
        for index in 0..NAMES + 2 {
            console.line(&format!(
                "StelaGPU: client{index}: opened the accelerator (type 5)"
            ));
            console.line(&format!(
                "AMFI: Denying core dump for pid {} (crash{index})",
                index + 1
            ));
        }
        console.line("x requested by: daemon (500)");
        console.line("x requested by: daemon (501)");
        let text = line(&console, 0, None);
        assert!(text.contains("client23] open=3"));
        assert!(!text.contains("client24"));
        assert!(text.contains(",crash23:1 restarts=daemon:1 "));
        // Two clients and two crashed names had no room; every crashed
        // process was seen once, so the daemon took such a place.
        assert!(text.ends_with(" unlisted=4"));
        // A name longer than any process's is not a name.
        let long = "n".repeat(NAME_BYTES + 1);
        console.line(&format!(
            "StelaGPU: {long}: opened the accelerator (type 5)"
        ));
        console.line(&format!("x requested by: {long} (9)"));
        console.line(&format!("AMFI: Denying core dump for pid 9 ({long})"));
        assert_eq!(line(&console, 0, None), text);
    }

    #[test]
    fn each_early_minute_gets_a_line_and_later_ones_when_processes_changed() {
        let mut schedule = Schedule::new();
        assert_eq!(schedule.next(), 60);
        assert_eq!(schedule.tick(59, 0), None);
        let line = |seconds| {
            Some(Tick {
                seconds,
                line: true,
            })
        };
        let quiet = |seconds| {
            Some(Tick {
                seconds,
                line: false,
            })
        };
        assert_eq!(schedule.tick(60, 0), line(60));
        assert_eq!(schedule.tick(61, 0), None);
        assert_eq!(schedule.next(), 120);
        // A stalled home PE: the count covers the minutes it missed.
        assert_eq!(schedule.tick(245, 3), line(180));
        for minute in 5..=EVERY_MINUTES {
            assert_eq!(schedule.tick(minute * 60, 3), line(60));
        }
        let after = EVERY_MINUTES + 1;
        assert_eq!(schedule.tick(after * 60, 3), quiet(60));
        assert_eq!(schedule.tick((after + 1) * 60, 4), line(60));
        for minute in after + 2..after + 1 + QUIET_MINUTES {
            assert_eq!(schedule.tick(minute * 60, 4), quiet(60));
        }
        assert_eq!(schedule.tick((after + 1 + QUIET_MINUTES) * 60, 4), line(60));
    }
}
