//! The display's frames composed by its GPU.
//!
//! Where the display's GPU serves Metal, the boot CPU shares the display's
//! three scanout buffers with that service once, and each composed swap
//! the DCP model offers ([`Scanout::try_present_gpu`]) is committed as a
//! composition of its damaged rows into the hidden buffer, from its
//! layers' guest pages. The hidden buffer's flip waits for the
//! composition: the next present or a flush on the boot CPU queues it once
//! the service shows it complete. A newer frame before then is composed
//! over it, the GPU running both in order, so the older one is replaced,
//! never torn. A frame the CPU draws first waits for the composition, for
//! [`SETTLE_MS`] at most.
//!
//! [`Scanout::try_present_gpu`]: stela_soc::machine::Scanout::try_present_gpu
use super::{super::gate, Node};
use crate::{engine::host, output};
use alloc::{sync::Arc, vec::Vec};
use core::ops::Range;
use stela_contracts::gate::MemoryRange;
use stela_drivers::gpu::metal::{
    binding::ComposeFormat,
    device::display::{DisplayFrame, DisplayLayer, DisplayPlane},
    error::Error,
    kms::Display,
    service::DisplayTarget,
};
use stela_soc::scanout::{GpuPresent, GpuStats, Layer, PixelFormat};

/// The longest a frame the CPU draws waits for the composition before it.
pub(super) const SETTLE_MS: u64 = 50;

/// The GPU's composition of the display's frames.
#[derive(Default)]
pub(super) struct Composition {
    /// The display's buffers shared with its GPU's service, once shared.
    shared: Option<Shared>,
    /// The frame composed into the hidden buffer, whose flip waits for its
    /// composition: its serial and the counter at its commit.
    pending: Option<(u64, u64)>,
    /// Why the GPU composes none of the display's frames.
    refused: Option<&'static str>,
    /// The first refusal's details were reported, and a composition that
    /// outlasted [`SETTLE_MS`].
    reported: bool,
    outlasted: bool,
    stats: GpuStats,
}

struct Shared {
    service: &'static Arc<gate::Shared>,
    target: DisplayTarget,
    /// Each shared buffer's framebuffer and bytes per row.
    buffers: [(u32, u32); 3],
}

impl Composition {
    /// Composes `layers` into `display`'s hidden buffer on the GPU of
    /// `name`, or declines, which leaves the CPU to compose them.
    pub(super) fn offer(
        &mut self,
        name: &str,
        display: &mut Display<Node>,
        layers: &[Layer],
        damage: Option<Range<usize>>,
        physical: &dyn Fn(u64) -> Option<u64>,
        tsc_hz: u64,
    ) -> GpuPresent {
        if let Some(reason) = self.refused {
            return GpuPresent::Declined(reason);
        }
        let frame_layers = match display_layers(layers, physical) {
            Ok(frame_layers) => frame_layers,
            Err(reason) => return GpuPresent::Declined(reason),
        };
        if let Err(reason) = self.share(name, display) {
            return GpuPresent::Declined(reason);
        }
        // The frame before goes to the screen once composed; one still
        // being composed is composed over.
        let replaced = match self.composed() {
            Some(true) => {
                self.flipped(display, tsc_hz);
                false
            }
            Some(false) => true,
            None => {
                // Its device was lost: its buffers are shared again.
                self.pending = None;
                self.shared = None;
                return GpuPresent::Declined("the display's GPU device was lost");
            }
        };
        let Some(shared) = &self.shared else {
            return GpuPresent::Declined("the display's buffers are not shared");
        };
        let (framebuffer, rows) = display.draw_rows(damage);
        let Some(index) = shared
            .buffers
            .iter()
            .position(|&(buffer, _)| buffer == framebuffer)
        else {
            return GpuPresent::Declined("the display's hidden buffer was not shared");
        };
        let frame = DisplayFrame {
            pitch: shared.buffers[index].1,
            width: display.size().0,
            rows: rows.start as u32..rows.end as u32,
            layers: frame_layers,
        };
        match shared
            .service
            .compose_display(&shared.target, index, &frame)
        {
            Ok((serial, copied)) => {
                if let Some((_, committed)) = self.pending.replace((serial, host::rdtsc())) {
                    self.note(committed, tsc_hz, replaced);
                }
                self.stats.copied = self
                    .stats
                    .copied
                    .saturating_add(u32::try_from(copied).unwrap_or(u32::MAX));
                GpuPresent::Taken
            }
            Err(Error::DeviceLost) => {
                self.shared = None;
                self.pending = None;
                GpuPresent::Declined("the display's GPU device was lost")
            }
            Err(Error::Unsupported(_)) => {
                GpuPresent::Declined("the GPU service was busy past its wait bound")
            }
            Err(error) => {
                self.report(name, "refused a display composition", error);
                GpuPresent::Declined("the display's GPU refused a composition")
            }
        }
    }

    /// Whether a composition waits for its flip.
    pub(super) fn pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Queues the flip of the hidden buffer's composed frame once its
    /// composition completed. On the boot CPU, the display's buffers are
    /// shared first, so presents on any processor may compose.
    pub(super) fn flush(&mut self, name: &str, display: &mut Display<Node>, tsc_hz: u64) {
        if self.refused.is_none() {
            let _ = self.share(name, display);
        }
        match self.composed() {
            Some(true) => self.flipped(display, tsc_hz),
            Some(false) => {}
            None => {
                self.pending = None;
                self.shared = None;
            }
        }
    }

    /// Before the CPU draws into the hidden buffer: waits for the
    /// composition there, for [`SETTLE_MS`] at most, and forgets it, as
    /// the CPU's frame replaces it.
    pub(super) fn settle(&mut self, name: &str, tsc_hz: u64) {
        let Some((_, committed)) = self.pending else {
            return;
        };
        let deadline = committed.saturating_add(tsc_hz / 1000 * SETTLE_MS);
        while self.composed() == Some(false) {
            if host::rdtsc() >= deadline {
                if !core::mem::replace(&mut self.outlasted, true) {
                    output::arguments(format_args!(
                        "GPU {name}: a display composition outlasted {SETTLE_MS} ms; the CPU \
                         draws over it\n"
                    ));
                }
                break;
            }
            core::hint::spin_loop();
        }
        self.pending = None;
        self.note(committed, tsc_hz, true);
    }

    pub(super) fn take_stats(&mut self) -> GpuStats {
        core::mem::take(&mut self.stats)
    }

    /// Whether the pending composition completed: true with none, `None`
    /// once its device was lost or replaced.
    fn composed(&self) -> Option<bool> {
        let Some((serial, _)) = self.pending else {
            return Some(true);
        };
        let shared = self.shared.as_ref()?;
        shared.service.display_composed(&shared.target, serial)
    }

    /// The pending composition completed: its flip is queued.
    fn flipped(&mut self, display: &mut Display<Node>, tsc_hz: u64) {
        let Some((_, committed)) = self.pending.take() else {
            return;
        };
        self.note(committed, tsc_hz, false);
        // A refused flip is the display's to report at its next flush.
        let _ = display.present_drawn();
    }

    fn note(&mut self, committed: u64, tsc_hz: u64, replaced: bool) {
        let ticks = host::rdtsc().saturating_sub(committed);
        let ns = u64::try_from(u128::from(ticks) * 1_000_000_000 / u128::from(tsc_hz.max(1)))
            .unwrap_or(u64::MAX);
        let stats = &mut self.stats;
        if replaced {
            stats.replaced += 1;
        } else {
            stats.flipped += 1;
        }
        stats.total_ns = stats.total_ns.saturating_add(ns);
        stats.longest_ns = stats.longest_ns.max(ns);
    }

    /// Shares `display`'s buffers with the service of its GPU `name`, on
    /// the boot CPU, where a DRM request completes at once.
    fn share(&mut self, name: &str, display: &mut Display<Node>) -> Result<(), &'static str> {
        if self.shared.is_some() {
            return Ok(());
        }
        let service = gate::service_of(name).ok_or("no Metal service runs on the display's GPU")?;
        let buffers = match display.share_buffers() {
            None => return Err("the display's buffers are shared from the boot CPU first"),
            Some(Err(error)) => {
                self.report(name, "shares no scanout buffer", error);
                return Err(self.refuse("the display's driver shares no scanout buffer"));
            }
            Some(Ok(buffers)) => buffers,
        };
        let descriptors: Vec<_> = buffers
            .iter()
            .map(|buffer| (buffer.fd, buffer.bytes as u64))
            .collect();
        match service.share_scanouts(&descriptors) {
            Ok(target) => {
                self.shared = Some(Shared {
                    service,
                    target,
                    buffers: buffers.map(|buffer| (buffer.framebuffer, buffer.pitch as u32)),
                });
                output::arguments(format_args!(
                    "GPU {name}: composes the display's frames into its shared scanout buffers\n"
                ));
                Ok(())
            }
            Err(error) => {
                // The descriptors stay open: the driver may have taken some.
                self.report(name, "imports no scanout buffer", error);
                Err(self.refuse("the display's GPU imports no scanout buffer"))
            }
        }
    }

    fn refuse(&mut self, reason: &'static str) -> &'static str {
        self.refused = Some(reason);
        reason
    }

    fn report(&mut self, name: &str, what: &str, error: impl core::fmt::Debug) {
        if !core::mem::replace(&mut self.reported, true) {
            output::arguments(format_args!(
                "GPU {name}: the display's GPU {what}: {error:?}\n"
            ));
        }
    }
}

/// `layers` as the compose kernel reads them: their planes' guest pages
/// by `physical`, or why the kernel cannot read them.
fn display_layers(
    layers: &[Layer],
    physical: &dyn Fn(u64) -> Option<u64>,
) -> Result<Vec<DisplayLayer>, &'static str> {
    if layers.len() > 4 {
        return Err("the GPU composes four layers at most");
    }
    let rect = |rect: stela_soc::scanout::Rect| [rect.x, rect.y, rect.width, rect.height];
    layers
        .iter()
        .map(|layer| {
            let mut planes = layer
                .planes(physical)
                .ok_or("a layer's pages did not resolve")?
                .into_iter()
                .map(|plane| DisplayPlane {
                    ranges: plane
                        .runs
                        .into_iter()
                        .map(|(address, bytes)| MemoryRange { address, bytes })
                        .collect(),
                    origin: plane.origin,
                });
            let color = planes.next().ok_or("a layer has its colour plane")?;
            let surface = layer.surface;
            let (format, alpha_stride) = match surface.format {
                PixelFormat::Bgra8888 => (ComposeFormat::Bgra8, 0),
                PixelFormat::Rgba8888 => (ComposeFormat::Rgba8, 0),
                PixelFormat::Argb2101010 => (ComposeFormat::Argb2101010, 0),
                PixelFormat::Xr10A8 { alpha_stride, .. } => (ComposeFormat::Xr10A8, alpha_stride),
            };
            Ok(DisplayLayer {
                color,
                alpha: planes.next(),
                stride: surface.stride,
                alpha_stride,
                format,
                premultiplied: layer.premultiplied,
                source: rect(layer.source),
                destination: rect(layer.destination),
                surface: layer.id,
            })
        })
        .collect()
}
