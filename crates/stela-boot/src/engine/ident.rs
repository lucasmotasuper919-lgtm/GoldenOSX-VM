//! The boot's `BOOT IDENT` line (`stela_boot_core::boot_ident`): the code
//! that starts the GPU's display, the guest's network adapter and the
//! disks leaves each a short text here, and the run loop writes them every
//! ten minutes with the version, the boot's number and the processor.
use crate::output;
use core::{
    cell::UnsafeCell,
    fmt,
    sync::atomic::{AtomicBool, Ordering},
};
use stela_boot_core::boot_ident::{Ident, Part, line};

/// The parts other code leaves.
#[derive(Clone, Copy)]
pub(super) enum Kind {
    /// The size and rate macOS draws at, and what shows it.
    Display,
    /// The guest's network adapter, or why it has none.
    Network,
    /// The transports of the disks this boot uses.
    Disk,
}

struct Slot {
    held: AtomicBool,
    part: UnsafeCell<Part>,
}

// SAFETY: `part` is reached only while `held` is taken (`Slot::with`).
unsafe impl Sync for Slot {}

impl Slot {
    /// Run `f` on the part unless another processor holds it now: a text
    /// that is not left this time is left by the next call or reads `-`.
    fn with<R>(&self, f: impl FnOnce(&mut Part) -> R) -> Option<R> {
        if self.held.swap(true, Ordering::Acquire) {
            return None;
        }
        // SAFETY: `held` was clear and is now this processor's.
        let result = f(unsafe { &mut *self.part.get() });
        self.held.store(false, Ordering::Release);
        Some(result)
    }
}

static PARTS: [Slot; 3] = [const {
    Slot {
        held: AtomicBool::new(false),
        part: UnsafeCell::new(Part::new()),
    }
}; 3];

/// Replace `kind`'s text.
pub(super) fn set(kind: Kind, text: fmt::Arguments<'_>) {
    PARTS[kind as usize].with(|part| part.set(text));
}

/// Add `text` to `kind`'s, unless it already holds it.
pub(super) fn add(kind: Kind, text: &str) {
    PARTS[kind as usize].with(|part| {
        if !part.as_str().split(", ").any(|held| held == text) {
            part.add(format_args!("{text}"));
        }
    });
}

/// This processor's brand string, as the loader's `HOST` line gives it.
fn cpu() -> alloc::string::String {
    use core::arch::x86_64::__cpuid;
    let mut words = [0u32; 12];
    if __cpuid(0x8000_0000).eax >= 0x8000_0004 {
        for (index, leaf) in (0x8000_0002..=0x8000_0004u32).enumerate() {
            let registers = __cpuid(leaf);
            words[index * 4..index * 4 + 4].copy_from_slice(&[
                registers.eax,
                registers.ebx,
                registers.ecx,
                registers.edx,
            ]);
        }
    }
    stela_boot_core::host_identity::brand(words)
}

/// Write the line.
pub(super) fn report() {
    let part = |kind: Kind| PARTS[kind as usize].with(|part| *part).unwrap_or_default();
    let (display, network, disk) = (part(Kind::Display), part(Kind::Network), part(Kind::Disk));
    output::arguments(format_args!(
        "{}",
        line(&Ident {
            version: env!("CARGO_PKG_VERSION"),
            boot: crate::stop_record::boot(),
            cpu: &cpu(),
            gpu: &super::gpu::ident(),
            display: display.as_str(),
            network: network.as_str(),
            disk: disk.as_str(),
        })
    ));
}
