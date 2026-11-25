//! The `GUEST SESSION` line: once a minute after macOS starts, what the
//! guest's session does (`stela_boot_core::guest_session`), for a boot
//! whose screen stays black or shows only a pointer. The processes come from
//! the guest's console as the home PE writes it; the frames from the display
//! model's pacing reports; the input from what the guest's devices took.
use crate::output;
use core::sync::atomic::{AtomicU64, Ordering};
use stela_boot_core::{
    guest_session::{Console, Minute, Schedule},
    screen::Exclusive,
};

/// The console's processes and the lines' schedule, which the home PE
/// alone reaches.
static SESSION: Exclusive<(Console, Schedule)> = Exclusive::new((Console::new(), Schedule::new()));

/// The counter when macOS started; 0 before it.
static STARTED: AtomicU64 = AtomicU64::new(0);
/// The seconds since then at which the next minute ends.
static NEXT: AtomicU64 = AtomicU64::new(60);
/// Frames presented since the last minute ended.
static PRESENTED: AtomicU64 = AtomicU64::new(0);
/// The counter when input last reached the guest; 0 while none did.
static INPUT: AtomicU64 = AtomicU64::new(0);

/// macOS starts: the minutes count from here.
pub(super) fn begin() {
    STARTED.store(output::now().max(1), Ordering::Relaxed);
}

/// A line of the guest's console.
pub(super) fn console_line(text: &str) {
    SESSION.try_with(|(console, _)| console.line(text));
}

/// A pacing window's presented frames.
pub(super) fn presented(frames: u32) {
    PRESENTED.fetch_add(u64::from(frames), Ordering::Relaxed);
}

/// An input event reached one of the guest's devices.
pub(super) fn input() {
    INPUT.store(output::now().max(1), Ordering::Relaxed);
}

/// Write the line of a minute that ended, on the home PE between slices.
pub(super) fn minute() {
    let (started, hz) = (STARTED.load(Ordering::Relaxed), output::counter_hz());
    if started == 0 || hz == 0 {
        return;
    }
    let now = output::now();
    let seconds = now.wrapping_sub(started) / hz;
    if seconds < NEXT.load(Ordering::Relaxed) {
        return;
    }
    SESSION.try_with(|(console, schedule)| {
        let Some(tick) = schedule.tick(seconds, console.changes()) else {
            return;
        };
        NEXT.store(schedule.next(), Ordering::Relaxed);
        let presented = PRESENTED.swap(0, Ordering::Relaxed);
        if !tick.line {
            return;
        }
        let last_input = match INPUT.load(Ordering::Relaxed) {
            0 => None,
            at => Some(now.wrapping_sub(at) / hz),
        };
        output::arguments(format_args!(
            "{}\n",
            Minute {
                console,
                open: open_connections(),
                presented,
                seconds: tick.seconds,
                last_input,
            }
        ));
    });
}

/// The accelerator connections open now on every GPU service, unless a
/// call holds one of them.
#[cfg(feature = "nvk")]
fn open_connections() -> Option<usize> {
    super::gpu::gate::open_connections()
}

#[cfg(not(feature = "nvk"))]
fn open_connections() -> Option<usize> {
    None
}
