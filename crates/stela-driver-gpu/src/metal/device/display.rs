//! Display frames the GPU composes.
//!
//! The display coprocessor model blends a swap's layers into the frame the
//! host's display shows. Where the display's driver shares its scanout
//! buffers as dma-bufs, the device imports them once, and each frame's
//! layers are imported from the guest RAM the display's IOMMU maps for
//! them: the compose kernel writes the frame's damaged rows into a shared
//! buffer, and the CPU only flips it once the command buffer completes.
//! The guest RAM imports are the device's own, so freeing them retires no
//! guest token, and a later frame over the same pages takes the kept
//! binding back. A two-plane extended-range layer the device rendered
//! itself, whose planes it keeps a device copy of, is composed from the
//! copy instead: a WindowServer framebuffer's rows then never cross the bus
//! twice.
use super::{Device, pins::Scope};
use crate::metal::{
    binding::{ComposeArguments, ComposeFormat, ComposeLayer},
    error::{Error, Result, ensure},
    shader::Compiler,
    vulkan::{
        BufferCreate, BufferUsage,
        backend::{Backend, Object},
        object::{Buffer, BufferInfo, Memory},
    },
};
use alloc::{format, vec::Vec};
use core::ops::Range;
use stela_contracts::gate::MemoryRange;

/// The Metal queue of the device's display compositions, which no guest
/// queue's held work delays.
const DISPLAY_QUEUE: u64 = u64::MAX - 1;

/// The usage of the buffers the compose kernel addresses.
const ADDRESSED: BufferUsage =
    BufferUsage::STORAGE_BUFFER.union(BufferUsage::SHADER_DEVICE_ADDRESS);

/// A display's scanout buffer the device imported, which compositions
/// write.
#[derive(Debug)]
pub struct ScanoutBuffer {
    memory: Memory,
    buffer: Buffer,
    address: u64,
    bytes: u64,
}

impl ScanoutBuffer {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

/// A plane of a display layer in guest RAM: its pages in the order the
/// display's IOMMU maps them, and the offset from their first byte to the
/// plane's first pixel, negative when the layer's rows start past the
/// pages' first.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DisplayPlane {
    pub ranges: Vec<MemoryRange>,
    pub origin: i64,
}

/// A layer of a display frame: its color plane and, for
/// [`ComposeFormat::Xr10A8`], its alpha plane, their bytes per row, and its
/// source and destination rectangles as `[x, y, width, height]`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DisplayLayer {
    pub color: DisplayPlane,
    pub alpha: Option<DisplayPlane>,
    pub stride: u32,
    pub alpha_stride: u32,
    pub format: ComposeFormat,
    pub premultiplied: bool,
    pub source: [u32; 4],
    pub destination: [u32; 4],
    /// The layer's IOSurface ID, zero where none is known. A two-plane
    /// layer the device rendered is composed from its planes' device copy
    /// ([`PlaneCopy`](crate::metal::pass::PlaneCopy)), which holds what they
    /// do, rather than from the guest RAM of `color` and `alpha`.
    pub surface: u32,
}

/// A frame's rows to compose into a scanout buffer of `width` BGRX pixels
/// a row, `pitch` bytes apart: black, then each layer bottom first.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DisplayFrame {
    pub pitch: u32,
    pub width: u32,
    pub rows: Range<u32>,
    pub layers: Vec<DisplayLayer>,
}

/// A plane's pages imported for one composition.
#[derive(Clone, Copy)]
struct Imported {
    memory: Memory,
    buffer: Buffer,
}

impl<B: Backend, C: Compiler> Device<B, C> {
    /// Takes the count of display layers composed from a device copy of
    /// their planes since the last call.
    pub fn take_display_copies(&mut self) -> u64 {
        core::mem::take(&mut self.display_copies)
    }

    /// Imports the dma-buf `fd` of `bytes`, a display's scanout buffer.
    /// The driver owns `fd` once this succeeds, and after an error once
    /// the memory was imported but no buffer could be placed on it.
    pub fn import_scanout(&mut self, fd: i32, bytes: u64) -> Result<ScanoutBuffer> {
        self.ensure_admission()?;
        let memory = self.backend.import_dma_buf(fd, bytes)?;
        self.book_memory(memory, bytes, || format!("scanout buffer of {bytes} bytes"));
        let (buffer, address) = self.addressed(memory, bytes)?;
        Ok(ScanoutBuffer {
            memory,
            buffer,
            address,
            bytes,
        })
    }

    /// Frees `scanout` once no composition writes it.
    pub fn release_scanout(&mut self, scanout: ScanoutBuffer) {
        self.release_imported(Imported {
            memory: scanout.memory,
            buffer: scanout.buffer,
        });
    }

    /// Commits the composition of `frame` into `scanout`, answering the
    /// command buffer whose completion makes its rows the display's.
    pub fn compose_display(
        &mut self,
        scanout: &ScanoutBuffer,
        frame: &DisplayFrame,
    ) -> Result<u64> {
        self.ensure_admission()?;
        let rows = frame.rows.end.saturating_sub(frame.rows.start);
        let last = u64::from(frame.rows.end.max(1) - 1) * u64::from(frame.pitch);
        ensure(
            u64::from(frame.width) * 4 <= u64::from(frame.pitch)
                && (rows == 0 || last + u64::from(frame.width) * 4 <= scanout.bytes),
            "a display frame's rows lie in its scanout buffer",
        )?;
        ensure(
            frame.layers.len() <= ComposeArguments::MAX_LAYERS as usize,
            "a display frame has at most four layers",
        )?;
        let count = frame
            .width
            .checked_mul(rows)
            .ok_or(Error::Unsupported("display frames of 4 Gi pixels"))?;
        let mut imported = Vec::new();
        let made = self.compose_layers(scanout, frame, count, &mut imported);
        for plane in imported {
            self.release_imported(plane);
        }
        made
    }

    fn compose_layers(
        &mut self,
        scanout: &ScanoutBuffer,
        frame: &DisplayFrame,
        count: u32,
        imported: &mut Vec<Imported>,
    ) -> Result<u64> {
        let mut layers = Vec::new();
        for layer in &frame.layers {
            let copy = match layer.format {
                ComposeFormat::Xr10A8 if layer.surface != 0 => {
                    self.surface_copy(layer.surface, layer.source)
                }
                _ => None,
            };
            if let Some(copy) = copy {
                self.display_copies += 1;
                layers.push(ComposeLayer {
                    color: copy.color,
                    alpha: copy.alpha,
                    stride: copy.color_bytes_per_row,
                    alpha_stride: copy.alpha_bytes_per_row,
                    format: layer.format,
                    premultiplied: layer.premultiplied,
                    source: layer.source,
                    destination: layer.destination,
                });
                continue;
            }
            let alpha = match (layer.format, &layer.alpha) {
                (ComposeFormat::Xr10A8, Some(alpha)) => Some(alpha),
                (ComposeFormat::Xr10A8, None) => {
                    return Err(Error::Invalid(
                        "a two-plane display layer has its alpha plane",
                    ));
                }
                _ => None,
            };
            let color = self.import_plane(&layer.color, layer.stride, 4, layer.source)?;
            imported.push(color.0);
            let alpha = match alpha {
                Some(plane) => {
                    let alpha = self.import_plane(plane, layer.alpha_stride, 1, layer.source)?;
                    imported.push(alpha.0);
                    alpha.1
                }
                None => 0,
            };
            layers.push(ComposeLayer {
                color: color.1,
                alpha,
                stride: layer.stride,
                alpha_stride: layer.alpha_stride,
                format: layer.format,
                premultiplied: layer.premultiplied,
                source: layer.source,
                destination: layer.destination,
            });
        }
        let arguments = ComposeArguments {
            destination: scanout.address,
            pitch: frame.pitch,
            width: frame.width,
            first_row: frame.rows.start,
            count,
            layers,
        };
        let mut commands = self.new_command_buffer();
        let recorded = commands
            .begin_blit()
            .and_then(|()| commands.compose(&mut self.resolver(), &arguments))
            .and_then(|()| commands.end_encoding(&mut self.resolver()));
        match recorded {
            Ok(()) => self.commit_on(0, DISPLAY_QUEUE, commands),
            Err(error) => {
                self.discard(commands);
                Err(error)
            }
        }
    }

    /// Imports `plane`'s pages, checked to hold every pixel of `source`
    /// at `size` bytes a pixel and `stride` bytes a row, and answers them
    /// with the GPU address of the plane's first pixel.
    fn import_plane(
        &mut self,
        plane: &DisplayPlane,
        stride: u32,
        size: u32,
        source: [u32; 4],
    ) -> Result<(Imported, u64)> {
        let bytes: u64 = plane.ranges.iter().map(|range| range.bytes).sum();
        let [x, y, width, height] = source.map(i128::from);
        let origin = i128::from(plane.origin);
        let first = origin + y * i128::from(stride) + x * i128::from(size);
        let end = origin + (y + height - 1) * i128::from(stride) + (x + width) * i128::from(size);
        ensure(
            width > 0 && height > 0 && first >= 0 && end <= i128::from(bytes),
            "a display layer's source lies in its pages",
        )?;
        let memory = self.backend.import_own(&plane.ranges)?;
        if let Some(key) = self.backend.kept_bound(memory) {
            self.book.retaken(key);
        }
        self.book_memory(memory, bytes, || format!("display layer of {bytes} bytes"));
        let (buffer, address) = self.addressed(memory, bytes)?;
        let imported = Imported { memory, buffer };
        Ok((imported, address.wrapping_add_signed(plane.origin)))
    }

    /// A buffer over all of `memory`, with its GPU address; the memory is
    /// freed when none can be made.
    fn addressed(&mut self, memory: Memory, bytes: u64) -> Result<(Buffer, u64)> {
        let info = BufferInfo {
            size: bytes,
            usage: ADDRESSED,
            flags: BufferCreate::EMPTY,
        };
        self.backend
            .create_placed_buffer(&info, memory, 0)
            .inspect_err(|_| {
                self.release_for(Scope::DEVICE, |device| device.release_memory(memory, 0));
            })
    }

    /// Frees an import once no command buffer submitted so far runs: they
    /// are the device's own work, whoever asked for it.
    fn release_imported(&mut self, imported: Imported) {
        self.release_for(Scope::DEVICE, |device| {
            device.defer(Object::Buffer(imported.buffer));
            device.release_memory(imported.memory, 0);
        });
    }
}

#[cfg(test)]
mod tests;
