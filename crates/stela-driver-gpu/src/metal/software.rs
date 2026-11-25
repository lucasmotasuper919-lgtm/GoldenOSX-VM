//! Pure-Rust software Metal backend (`SoftBackend` + `SoftCompiler`).
//!
//! Backs Stela's `StelaGPU` Metal wire service without Mesa/LLVM so macOS
//! `WindowServer` receives a functional Metal device and renders clears,
//! blits, and copies directly onto guest `IOSurface` RAM (which Stela's DCP
//! compositor presents to the host display).

use crate::metal::{
    abi::FunctionType,
    binding::{ArgumentTable, PushConstants, ResourceId, TableSlot},
    capabilities::{Platform, ShaderFeatures},
    command_buffer::Segment,
    error::{Error, NO_FUNCTION, Result},
    library::{Function, Library},
    nvidia::{Architecture, CHIPS, Chip},
    open::{Attachment as OpenAttachment, PcieLink},
    pass::ClearValue,
    pixel_format::{Compression, Kind, PixelFormat},
    reflection::Binding,
    shader::{
        CompiledShader, Compiler, FunctionConstant, Interface, LinkedFunction, Provenance,
        Reflection, Specialization, Stage, Tessellation, TessellationStages,
    },
    stitching::StitchedGraph,
    vulkan::{
        AttachmentLoadOp, DriverId, Format, FormatFeatures as F, IndexType, MemoryHeapFlags,
        MemoryProperties as M, PhysicalDeviceType, PipelineBindPoint, PrimitiveTopology,
        ResolveModes, SampleCounts, ShaderStages, SubgroupFeatures,
        backend::{Backend, BufferAllocation, Descriptor, Object, Outside, Requirements},
        command::{
            BufferCopy, BufferImageCopy, Command, ImageBlit, ImageCopy, Rect, VertexBuffer,
            Viewport,
        },
        object::{
            Buffer, BufferInfo, BufferView, BufferViewInfo, ComputePipelineInfo, ExecutionSet,
            GraphicsPipelineInfo, Image, ImageInfo, ImageView, ImageViewInfo,
            IndirectCommandsLayout, IndirectCommandsLayoutInfo, Memory, MemoryRequest, Pipeline,
            PipelineCache, QueryPool, QueryPoolInfo, Sampler, SamplerInfo, Semaphore, ShaderCode,
            SparseImageRequirements, VertexInput,
        },
        physical::{
            Features, FormatSupport, FormatTable, GeneratedCommandsLimits, Identity, Limits,
            Memory as PhysicalMemory, MemoryHeap, MemoryType, MeshLimits, PciBusInfo,
            PhysicalDevice, SampleLocationLimits, SparseLimits,
        },
    },
};
use alloc::{
    boxed::Box,
    collections::{BTreeMap, VecDeque},
    string::String,
    sync::Arc,
    vec,
    vec::Vec,
};
use core::num::NonZeroU64;
use stela_contracts::gate::MemoryRange;

/// Translates a guest physical address range `(guest_pa, bytes)` into one or
/// more host virtual slices `(host_ptr, len)`. Returns `true` if the entire
/// range is valid guest RAM.
pub type GuestRunsFn = fn(u64, u64, &mut dyn FnMut(*mut u8, usize)) -> bool;

/// Cap individual CPU-backed device allocations so a large sparse/heap request
/// does not exhaust the host's heap in a 4 GiB VM.
const MAX_OWNED_BYTES: usize = 32 * 1024 * 1024;

const SAMPLE_COUNTS: SampleCounts = SampleCounts::C1
    .union(SampleCounts::C2)
    .union(SampleCounts::C4)
    .union(SampleCounts::C8);
const DESCRIPTORS: u32 = 1 << 20;

enum SoftMemory {
    Owned(Vec<u8>),
    Guest {
        ranges: Vec<MemoryRange>,
        token: Option<NonZeroU64>,
        total: u64,
    },
}

struct SoftBuffer {
    memory: Memory,
    offset: u64,
    size: u64,
}

struct SoftImage {
    info: ImageInfo,
    memory: Memory,
    offset: u64,
    bpp: u32,
    row_pitch: u64,
}

struct SoftPipeline {
    vertex_input: VertexInput,
    topology: PrimitiveTopology,
    blend_enable: bool,
    is_simple_vertex: bool,
}

const SIMPLE_VERTEX_CODE_BIT: u64 = 1 << 60;

#[derive(Clone, Copy)]
struct SoftVertex {
    pos: [f32; 4],
    uv: [f32; 2],
    color: [f32; 4],
}

/// Pure-Rust software Vulkan-like [`Backend`] that executes clears, copies,
/// blits, and 2D textured-quad draws directly on host-allocated buffers and
/// imported guest `IOSurface` RAM, and immediately signals timeline semaphores.
pub struct SoftBackend {
    physical: PhysicalDevice,
    guest_runs: GuestRunsFn,
    next: u64,
    memories: BTreeMap<u64, SoftMemory>,
    buffers: BTreeMap<u64, SoftBuffer>,
    images: BTreeMap<u64, SoftImage>,
    image_views: BTreeMap<u64, Image>,
    pipelines: BTreeMap<u64, SoftPipeline>,
    sampled_images: BTreeMap<u32, ImageView>,
    semaphores: BTreeMap<u64, u64>,
    retired: VecDeque<NonZeroU64>,
    // Active graphics pass state during submit()
    active_target: Option<Image>,
    active_resolve: Option<Image>,
    last_guest_display: Option<Image>,
    guest_display_mems: Vec<(Memory, u64)>,
    active_extent: [u32; 2],
    dirty_y: Option<(u32, u32)>,
    active_pipeline: Option<Pipeline>,
    viewport: Option<Viewport>,
    scissor: Option<Rect>,
    topology: PrimitiveTopology,
    push_constants: PushConstants,
    vertex_buffers: BTreeMap<u32, VertexBuffer>,
    index_buffer: Option<(Buffer, u64, IndexType)>,
    submissions: u64,
    clears: u64,
    copies: u64,
    draws: u64,
    textured_screen_draws: u64,
    dispatches: u64,
    passes: u64,
    log: Option<fn(&str)>,
}

impl SoftBackend {
    /// Create a software backend with the given guest-RAM mapping callback and
    /// advertised VRAM budget in bytes (e.g. `512 << 20`).
    pub fn new(guest_runs: GuestRunsFn, vram_bytes: u64) -> Self {
        let chip = soft_chip();
        Self {
            physical: soft_physical_device(chip, vram_bytes),
            guest_runs,
            next: 1,
            memories: BTreeMap::new(),
            buffers: BTreeMap::new(),
            images: BTreeMap::new(),
            image_views: BTreeMap::new(),
            pipelines: BTreeMap::new(),
            sampled_images: BTreeMap::new(),
            semaphores: BTreeMap::new(),
            retired: VecDeque::new(),
            active_target: None,
            active_resolve: None,
            last_guest_display: None,
            guest_display_mems: Vec::new(),
            active_extent: [0, 0],
            dirty_y: None,
            active_pipeline: None,
            viewport: None,
            scissor: None,
            topology: PrimitiveTopology::TriangleList,
            push_constants: PushConstants::default(),
            vertex_buffers: BTreeMap::new(),
            index_buffer: None,
            submissions: 0,
            clears: 0,
            copies: 0,
            draws: 0,
            textured_screen_draws: 0,
            dispatches: 0,
            passes: 0,
            log: None,
        }
    }

    pub fn with_log(mut self, log: fn(&str)) -> Self {
        self.log = Some(log);
        self
    }

    /// The synthetic platform identity matching [`Self::new`].
    pub fn platform() -> Platform {
        soft_chip().platform(OpenAttachment {
            link: PcieLink {
                speed: 4,
                width: 16,
            },
            external: false,
            location_number: 0,
        })
    }

    /// Counters for diagnostics: `(submissions, clears, copies)`.
    pub fn stats(&self) -> (u64, u64, u64) {
        (self.submissions, self.clears, self.copies)
    }

    fn alloc_id(&mut self) -> u64 {
        let id = self.next;
        self.next += 1;
        id
    }

    fn alloc_owned_memory(&mut self, size: u64) -> Memory {
        let id = self.alloc_id();
        let alloc_len = (size as usize).min(MAX_OWNED_BYTES);
        self.memories
            .insert(id, SoftMemory::Owned(vec![0u8; alloc_len]));
        Memory(id)
    }

    fn read_mem(&self, memory: Memory, offset: u64, out: &mut [u8]) {
        let Some(mem) = self.memories.get(&memory.0) else {
            out.fill(0);
            return;
        };
        match mem {
            SoftMemory::Owned(buf) => {
                let start = (offset as usize).min(buf.len());
                let avail = buf.len().saturating_sub(start).min(out.len());
                if avail > 0 {
                    out[..avail].copy_from_slice(&buf[start..start + avail]);
                }
                if avail < out.len() {
                    out[avail..].fill(0);
                }
            }
            SoftMemory::Guest { ranges, .. } => {
                let mut cursor = offset;
                let mut dst_off = 0usize;
                let mut range_base = 0u64;
                for r in ranges {
                    let range_end = range_base.saturating_add(r.bytes);
                    if dst_off >= out.len() {
                        break;
                    }
                    if cursor < range_end {
                        let in_range = cursor - range_base;
                        let take = ((range_end - cursor) as usize).min(out.len() - dst_off);
                        let guest_pa = r.address.saturating_add(in_range);
                        let mut copied = 0usize;
                        let dst_slice = &mut out[dst_off..dst_off + take];
                        (self.guest_runs)(guest_pa, take as u64, &mut |ptr, len| {
                            let n = len.min(dst_slice.len().saturating_sub(copied));
                            if n > 0 {
                                // SAFETY: `guest_runs` yields valid mapped host RAM for guest pages.
                                unsafe {
                                    core::ptr::copy_nonoverlapping(
                                        ptr as *const u8,
                                        dst_slice.as_mut_ptr().add(copied),
                                        n,
                                    );
                                }
                                copied += n;
                            }
                        });
                        cursor += take as u64;
                        dst_off += take;
                    }
                    range_base = range_end;
                }
                if dst_off < out.len() {
                    out[dst_off..].fill(0);
                }
            }
        }
    }

    fn write_mem(&mut self, memory: Memory, offset: u64, data: &[u8]) {
        let Some(mem) = self.memories.get_mut(&memory.0) else {
            return;
        };
        match mem {
            SoftMemory::Owned(buf) => {
                let start = (offset as usize).min(buf.len());
                let avail = buf.len().saturating_sub(start).min(data.len());
                if avail > 0 {
                    buf[start..start + avail].copy_from_slice(&data[..avail]);
                }
            }
            SoftMemory::Guest { ranges, .. } => {
                let guest_runs = self.guest_runs;
                let mut cursor = offset;
                let mut src_off = 0usize;
                let mut range_base = 0u64;
                for r in ranges.iter() {
                    let range_end = range_base.saturating_add(r.bytes);
                    if src_off >= data.len() {
                        break;
                    }
                    if cursor < range_end {
                        let in_range = cursor - range_base;
                        let take = ((range_end - cursor) as usize).min(data.len() - src_off);
                        let guest_pa = r.address.saturating_add(in_range);
                        let mut copied = 0usize;
                        let src_slice = &data[src_off..src_off + take];
                        (guest_runs)(guest_pa, take as u64, &mut |ptr, len| {
                            let n = len.min(src_slice.len().saturating_sub(copied));
                            if n > 0 {
                                // SAFETY: `guest_runs` yields valid mapped host RAM for guest pages.
                                unsafe {
                                    core::ptr::copy_nonoverlapping(
                                        src_slice.as_ptr().add(copied),
                                        ptr,
                                        n,
                                    );
                                }
                                copied += n;
                            }
                        });
                        cursor += take as u64;
                        src_off += take;
                    }
                    range_base = range_end;
                }
            }
        }
    }

    fn fill_image_color(&mut self, image: Image, value: ClearValue) {
        let Some(img) = self.images.get(&image.0) else {
            return;
        };
        let memory = img.memory;
        let base_offset = img.offset;
        let width = img.info.extent[0] as usize;
        let height = img.info.extent[1].max(1) as usize;
        let bpp = img.bpp as usize;
        let row_pitch = img.row_pitch;
        let format = img.info.format;
        let is_guest = matches!(self.memories.get(&memory.0), Some(SoftMemory::Guest { .. }));

        if width == 0 || height == 0 || (bpp != 4 && bpp != 8) {
            return;
        }
        let pixel = encode_clear_pixel(format, value);
        let mut row = vec![0u8; width.saturating_mul(bpp).min(65536)];
        if bpp == 4 {
            for chunk in row.chunks_exact_mut(4) {
                chunk.copy_from_slice(&pixel);
            }
        } else {
            let px8 = encode_rgba16f(
                f32::from(pixel[0]) / 255.0,
                f32::from(pixel[1]) / 255.0,
                f32::from(pixel[2]) / 255.0,
                f32::from(pixel[3]) / 255.0,
            );
            for chunk in row.chunks_exact_mut(8) {
                chunk.copy_from_slice(&px8);
            }
        }
        let _ = is_guest;
        let rows = height.min(4096);
        for y in 0..rows {
            let off = base_offset.saturating_add((y as u64).saturating_mul(row_pitch));
            self.write_mem(memory, off, &row);
        }
        self.clears += 1;
    }

    fn exec_copy_buffer(&mut self, src: Buffer, dst: Buffer, region: BufferCopy) {
        let (Some(sb), Some(db)) = (self.buffers.get(&src.0), self.buffers.get(&dst.0)) else {
            return;
        };
        let (sm, so) = (sb.memory, sb.offset.saturating_add(region.src_offset));
        let (dm, doff) = (db.memory, db.offset.saturating_add(region.dst_offset));
        let len = (region.size as usize).min(MAX_OWNED_BYTES);
        if len == 0 {
            return;
        }
        let mut tmp = vec![0u8; len];
        self.read_mem(sm, so, &mut tmp);
        self.write_mem(dm, doff, &tmp);
        self.copies += 1;
    }

    fn exec_copy_buffer_to_image(&mut self, src: Buffer, dst: Image, region: &BufferImageCopy) {
        let (Some(sb), Some(di)) = (self.buffers.get(&src.0), self.images.get(&dst.0)) else {
            return;
        };
        let bpp = di.bpp as u64;
        if bpp == 0 {
            return;
        }
        let (sm, sbase) = (sb.memory, sb.offset.saturating_add(region.buffer_offset));
        let (dm, dbase, dpitch) = (di.memory, di.offset, di.row_pitch);
        let src_row_texels = if region.buffer_row_length != 0 {
            region.buffer_row_length as u64
        } else {
            region.extent[0] as u64
        };
        let src_pitch = src_row_texels.saturating_mul(bpp);
        let row_bytes = (region.extent[0] as usize).saturating_mul(bpp as usize);
        if row_bytes == 0 || row_bytes > 65536 {
            return;
        }
        let dx = region.offset[0].max(0) as u64;
        let dy = region.offset[1].max(0) as u64;
        let rows = (region.extent[1] as usize).min(4096);
        let mut line = vec![0u8; row_bytes];
        for y in 0..rows {
            let soff = sbase.saturating_add((y as u64).saturating_mul(src_pitch));
            let doff = dbase
                .saturating_add((dy + y as u64).saturating_mul(dpitch))
                .saturating_add(dx.saturating_mul(bpp));
            self.read_mem(sm, soff, &mut line);
            self.write_mem(dm, doff, &line);
        }
        self.copies += 1;
    }

    fn exec_copy_image_to_buffer(&mut self, src: Image, dst: Buffer, region: &BufferImageCopy) {
        let (Some(si), Some(db)) = (self.images.get(&src.0), self.buffers.get(&dst.0)) else {
            return;
        };
        let bpp = si.bpp as u64;
        if bpp == 0 {
            return;
        }
        let (sm, sbase, spitch) = (si.memory, si.offset, si.row_pitch);
        let (dm, dbase) = (db.memory, db.offset.saturating_add(region.buffer_offset));
        let dst_row_texels = if region.buffer_row_length != 0 {
            region.buffer_row_length as u64
        } else {
            region.extent[0] as u64
        };
        let dst_pitch = dst_row_texels.saturating_mul(bpp);
        let row_bytes = (region.extent[0] as usize).saturating_mul(bpp as usize);
        if row_bytes == 0 || row_bytes > 65536 {
            return;
        }
        let sx = region.offset[0].max(0) as u64;
        let sy = region.offset[1].max(0) as u64;
        let rows = (region.extent[1] as usize).min(4096);
        let mut line = vec![0u8; row_bytes];
        for y in 0..rows {
            let soff = sbase
                .saturating_add((sy + y as u64).saturating_mul(spitch))
                .saturating_add(sx.saturating_mul(bpp));
            let doff = dbase.saturating_add((y as u64).saturating_mul(dst_pitch));
            self.read_mem(sm, soff, &mut line);
            self.write_mem(dm, doff, &line);
        }
        self.copies += 1;
    }

    fn exec_copy_image(&mut self, src: Image, dst: Image, regions: &[ImageCopy]) {
        let (Some(si), Some(di)) = (self.images.get(&src.0), self.images.get(&dst.0)) else {
            return;
        };
        let bpp = si.bpp.min(di.bpp) as u64;
        if bpp == 0 {
            return;
        }
        let (sm, sbase, spitch) = (si.memory, si.offset, si.row_pitch);
        let (dm, dbase, dpitch) = (di.memory, di.offset, di.row_pitch);
        for r in regions {
            let row_bytes = (r.extent[0] as usize).saturating_mul(bpp as usize);
            if row_bytes == 0 || row_bytes > 65536 {
                continue;
            }
            let sx = r.src_offset[0].max(0) as u64;
            let sy = r.src_offset[1].max(0) as u64;
            let dx = r.dst_offset[0].max(0) as u64;
            let dy = r.dst_offset[1].max(0) as u64;
            let rows = (r.extent[1] as usize).min(4096);
            let mut line = vec![0u8; row_bytes];
            for y in 0..rows {
                let soff = sbase
                    .saturating_add((sy + y as u64).saturating_mul(spitch))
                    .saturating_add(sx.saturating_mul(bpp));
                let doff = dbase
                    .saturating_add((dy + y as u64).saturating_mul(dpitch))
                    .saturating_add(dx.saturating_mul(bpp));
                self.read_mem(sm, soff, &mut line);
                self.write_mem(dm, doff, &line);
            }
            self.copies += 1;
        }
    }

    fn exec_blit_image(&mut self, src: Image, dst: Image, regions: &[ImageBlit]) {
        let (Some(si), Some(di)) = (self.images.get(&src.0), self.images.get(&dst.0)) else {
            return;
        };
        if si.bpp != 4 || di.bpp != 4 {
            return;
        }
        let (sm, sbase, spitch) = (si.memory, si.offset, si.row_pitch);
        let (dm, dbase, dpitch) = (di.memory, di.offset, di.row_pitch);
        for r in regions {
            let sx0 = r.src_bounds[0][0].min(r.src_bounds[1][0]).max(0) as usize;
            let sy0 = r.src_bounds[0][1].min(r.src_bounds[1][1]).max(0) as usize;
            let sw = (r.src_bounds[1][0] - r.src_bounds[0][0]).unsigned_abs() as usize;
            let sh = (r.src_bounds[1][1] - r.src_bounds[0][1]).unsigned_abs() as usize;
            let dx0 = r.dst_bounds[0][0].min(r.dst_bounds[1][0]).max(0) as usize;
            let dy0 = r.dst_bounds[0][1].min(r.dst_bounds[1][1]).max(0) as usize;
            let dw = (r.dst_bounds[1][0] - r.dst_bounds[0][0]).unsigned_abs() as usize;
            let dh = (r.dst_bounds[1][1] - r.dst_bounds[0][1]).unsigned_abs() as usize;
            if sw == 0 || sh == 0 || dw == 0 || dh == 0 || sw > 8192 || dw > 8192 {
                continue;
            }
            let mut src_row = vec![0u8; sw * 4];
            let mut dst_row = vec![0u8; dw * 4];
            let rows = dh.min(4096);
            for y in 0..rows {
                let sy = sy0 + (y * sh) / dh;
                let soff = sbase
                    .saturating_add((sy as u64).saturating_mul(spitch))
                    .saturating_add((sx0 as u64) * 4);
                self.read_mem(sm, soff, &mut src_row);
                for x in 0..dw {
                    let sx = (x * sw) / dw;
                    dst_row[x * 4..x * 4 + 4].copy_from_slice(&src_row[sx * 4..sx * 4 + 4]);
                }
                let doff = dbase
                    .saturating_add(((dy0 + y) as u64).saturating_mul(dpitch))
                    .saturating_add((dx0 as u64) * 4);
                self.write_mem(dm, doff, &dst_row);
            }
            self.copies += 1;
        }
    }

    fn read_gpu_address(&self, addr: u64, out: &mut [u8]) -> bool {
        const BASE: u64 = 0x1000_0000_0000;
        if addr < BASE {
            return false;
        }
        let rel = addr - BASE;
        let buf_id = rel >> 32;
        let off = rel & 0xFFFF_FFFF;
        let Some(buf) = self.buffers.get(&buf_id) else {
            return false;
        };
        self.read_mem(buf.memory, buf.offset.saturating_add(off), out);
        true
    }

    fn bound_primary_buffer(&self, slot: u32) -> Option<(u64, u64)> {
        let table_addr = self.push_constants.tables[TableSlot::Primary as usize];
        if table_addr == 0 {
            return None;
        }
        let mut addr_raw = [0u8; 8];
        let mut len_raw = [0u8; 8];
        let aoff = u64::from(ArgumentTable::BUFFER_ADDRESSES_OFFSET) + u64::from(slot) * 8;
        let loff = u64::from(ArgumentTable::BUFFER_LENGTHS_OFFSET) + u64::from(slot) * 8;
        if !self.read_gpu_address(table_addr.saturating_add(aoff), &mut addr_raw)
            || !self.read_gpu_address(table_addr.saturating_add(loff), &mut len_raw)
        {
            return None;
        }
        let addr = u64::from_le_bytes(addr_raw);
        let len = u64::from_le_bytes(len_raw);
        (addr != 0 && len > 0).then_some((addr, len))
    }

    fn bound_fragment_texture(&self) -> Option<(Image, Option<Image>, bool)> {
        for slot_stage in [TableSlot::Fragment as usize, TableSlot::Primary as usize] {
            let table_addr = self.push_constants.tables[slot_stage];
            if table_addr == 0 {
                continue;
            }
            let mut raw = [0u8; 16 * 8];
            if !self.read_gpu_address(
                table_addr.saturating_add(u64::from(ArgumentTable::TEXTURES_OFFSET)),
                &mut raw,
            ) {
                continue;
            }
            for i in 0..16 {
                let res_id = u64::from_le_bytes(raw[i * 8..i * 8 + 8].try_into().unwrap());
                let idx = (res_id & 0xFFFF_FFFF) as u32;
                let high = (res_id >> 32) as u32;
                if idx != 0
                    && let Some(view) = self.sampled_images.get(&idx)
                    && let Some(&img) = self.image_views.get(&view.0)
                {
                    let is_xr10 = (high & ResourceId::XR10) != 0
                        && (high & ResourceId::YCBCR) == 0;
                    let alpha_img = if is_xr10 {
                        self.sampled_images
                            .get(&(idx + 1))
                            .and_then(|av| self.image_views.get(&av.0).copied())
                    } else {
                        None
                    };
                    return Some((img, alpha_img, is_xr10));
                }
            }
        }
        None
    }

    fn fetch_vertex(&self, vertex_index: u32) -> Option<SoftVertex> {
        let pipe = self.active_pipeline.and_then(|p| self.pipelines.get(&p.0))?;
        let mut pos = [0.0f32, 0.0, 0.0, 1.0];
        let mut uv = [0.0f32, 0.0];
        let mut color = [1.0f32, 1.0, 1.0, 1.0];
        let mut got_pos = false;

        for attr in &pipe.vertex_input.attributes {
            let Some(vb) = self.vertex_buffers.get(&attr.binding) else {
                continue;
            };
            let stride = vb.stride.unwrap_or_else(|| {
                pipe.vertex_input
                    .bindings
                    .iter()
                    .find(|b| b.binding == attr.binding)
                    .and_then(|b| b.stride)
                    .map_or(48, u64::from)
            });
            let Some(buf) = self.buffers.get(&vb.buffer.0) else {
                continue;
            };
            let byte_off = buf
                .offset
                .saturating_add(vb.offset)
                .saturating_add(u64::from(vertex_index).saturating_mul(stride))
                .saturating_add(u64::from(attr.offset));
            match attr.location {
                0 => {
                    let mut raw = [0u8; 16];
                    match attr.format {
                        Format::R32G32Sfloat => {
                            self.read_mem(buf.memory, byte_off, &mut raw[..8]);
                            pos[0] = f32::from_le_bytes(raw[0..4].try_into().unwrap());
                            pos[1] = f32::from_le_bytes(raw[4..8].try_into().unwrap());
                            got_pos = true;
                        }
                        Format::R32G32B32Sfloat => {
                            self.read_mem(buf.memory, byte_off, &mut raw[..12]);
                            pos[0] = f32::from_le_bytes(raw[0..4].try_into().unwrap());
                            pos[1] = f32::from_le_bytes(raw[4..8].try_into().unwrap());
                            pos[2] = f32::from_le_bytes(raw[8..12].try_into().unwrap());
                            got_pos = true;
                        }
                        _ => {
                            self.read_mem(buf.memory, byte_off, &mut raw);
                            pos[0] = f32::from_le_bytes(raw[0..4].try_into().unwrap());
                            pos[1] = f32::from_le_bytes(raw[4..8].try_into().unwrap());
                            pos[2] = f32::from_le_bytes(raw[8..12].try_into().unwrap());
                            pos[3] = f32::from_le_bytes(raw[12..16].try_into().unwrap());
                            got_pos = true;
                        }
                    }
                }
                1 => {
                    let mut raw = [0u8; 8];
                    self.read_mem(buf.memory, byte_off, &mut raw);
                    if attr.format == Format::R16G16Sfloat {
                        uv[0] = f16_to_f32(u16::from_le_bytes([raw[0], raw[1]]));
                        uv[1] = f16_to_f32(u16::from_le_bytes([raw[2], raw[3]]));
                    } else {
                        uv[0] = f32::from_le_bytes(raw[0..4].try_into().unwrap());
                        uv[1] = f32::from_le_bytes(raw[4..8].try_into().unwrap());
                    }
                }
                3 => match attr.format {
                    Format::R16G16B16A16Sfloat => {
                        let mut raw = [0u8; 8];
                        self.read_mem(buf.memory, byte_off, &mut raw);
                        for i in 0..4 {
                            color[i] = f16_to_f32(u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]));
                        }
                    }
                    Format::R8G8B8A8Unorm | Format::R8G8B8A8Srgb => {
                        let mut raw = [0u8; 4];
                        self.read_mem(buf.memory, byte_off, &mut raw);
                        for i in 0..4 {
                            color[i] = f32::from(raw[i]) / 255.0;
                        }
                    }
                    Format::B8G8R8A8Unorm | Format::B8G8R8A8Srgb => {
                        let mut raw = [0u8; 4];
                        self.read_mem(buf.memory, byte_off, &mut raw);
                        color = [
                            f32::from(raw[2]) / 255.0,
                            f32::from(raw[1]) / 255.0,
                            f32::from(raw[0]) / 255.0,
                            f32::from(raw[3]) / 255.0,
                        ];
                    }
                    Format::R32G32B32A32Sfloat => {
                        let mut raw = [0u8; 16];
                        self.read_mem(buf.memory, byte_off, &mut raw);
                        for i in 0..4 {
                            color[i] =
                                f32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
                        }
                    }
                    _ => {}
                },
                _ => {}
            }
        }
        if got_pos {
            return Some(SoftVertex { pos, uv, color });
        }
        // Fallback when pipeline.vertex_input was filtered out (empty SoftCompiler
        // reflection):
        // If this pipeline uses SimpleVertex (Language Chooser / AppKit), read slot 0
        // (stride 16: Float2@0 pos, Float2@8 uv) first so a uniform buffer at slot 1
        // is never mistaken for a QuartzCore vertex buffer.
        if pipe.is_simple_vertex
            && let Some((addr, len)) = self.bound_primary_buffer(0)
        {
            let voff = u64::from(vertex_index).saturating_mul(16);
            if voff + 16 <= len {
                let mut raw = [0u8; 16];
                if self.read_gpu_address(addr.saturating_add(voff), &mut raw) {
                    pos = [
                        f32::from_le_bytes(raw[0..4].try_into().unwrap()),
                        f32::from_le_bytes(raw[4..8].try_into().unwrap()),
                        0.0,
                        1.0,
                    ];
                    uv = [
                        f32::from_le_bytes(raw[8..12].try_into().unwrap()),
                        f32::from_le_bytes(raw[12..16].try_into().unwrap()),
                    ];
                    color = [1.0, 1.0, 1.0, 1.0];
                    return Some(SoftVertex { pos, uv, color });
                }
            }
        }
        // 1) QuartzCore (`fixed_vert_lph_spc`): slot 1, stride 48 bytes
        //    (Float4@0, Float2@16, Float2@24, Half4@32).
        if let Some((addr, len)) = self.bound_primary_buffer(1) {
            let voff = u64::from(vertex_index).saturating_mul(48);
            if voff + 40 <= len {
                let mut raw = [0u8; 40];
                if self.read_gpu_address(addr.saturating_add(voff), &mut raw) {
                    pos = [
                        f32::from_le_bytes(raw[0..4].try_into().unwrap()),
                        f32::from_le_bytes(raw[4..8].try_into().unwrap()),
                        f32::from_le_bytes(raw[8..12].try_into().unwrap()),
                        f32::from_le_bytes(raw[12..16].try_into().unwrap()),
                    ];
                    uv = [
                        f32::from_le_bytes(raw[16..20].try_into().unwrap()),
                        f32::from_le_bytes(raw[20..24].try_into().unwrap()),
                    ];
                    color = [
                        f16_to_f32(u16::from_le_bytes([raw[32], raw[33]])),
                        f16_to_f32(u16::from_le_bytes([raw[34], raw[35]])),
                        f16_to_f32(u16::from_le_bytes([raw[36], raw[37]])),
                        f16_to_f32(u16::from_le_bytes([raw[38], raw[39]])),
                    ];
                    return Some(SoftVertex { pos, uv, color });
                }
            }
        }
        // 2) Language Chooser / AppKit (`SimpleVertex` + `SimpleTextureFragment`):
        //    slot 0, stride 16 bytes (Float2@0 pos, Float2@8 uv).
        if let Some((addr, len)) = self.bound_primary_buffer(0) {
            let voff = u64::from(vertex_index).saturating_mul(16);
            if voff + 16 <= len {
                let mut raw = [0u8; 16];
                if self.read_gpu_address(addr.saturating_add(voff), &mut raw) {
                    pos = [
                        f32::from_le_bytes(raw[0..4].try_into().unwrap()),
                        f32::from_le_bytes(raw[4..8].try_into().unwrap()),
                        0.0,
                        1.0,
                    ];
                    uv = [
                        f32::from_le_bytes(raw[8..12].try_into().unwrap()),
                        f32::from_le_bytes(raw[12..16].try_into().unwrap()),
                    ];
                    color = [1.0, 1.0, 1.0, 1.0];
                    return Some(SoftVertex { pos, uv, color });
                }
            }
        }
        None
    }

    fn exec_draw(&mut self, vertex_count: u32, first_vertex: u32) {
        if vertex_count < 3 || vertex_count > 65536 {
            return;
        }
        let mut verts = Vec::with_capacity(vertex_count as usize);
        for i in 0..vertex_count {
            if let Some(v) = self.fetch_vertex(first_vertex.saturating_add(i)) {
                verts.push(v);
            }
        }
        self.rasterize_vertices(&verts);
    }

    fn exec_draw_indexed(&mut self, index_count: u32, first_index: u32, vertex_offset: i32) {
        if index_count < 3 || index_count > 65536 {
            return;
        }
        let Some((ibuf, ioff, itype)) = self.index_buffer else {
            return;
        };
        let Some(buf) = self.buffers.get(&ibuf.0) else {
            return;
        };
        let elem_bytes: u64 = match itype {
            IndexType::Uint16 => 2,
            IndexType::Uint32 => 4,
        };
        let byte_len = (index_count as usize) * (elem_bytes as usize);
        let mut raw = vec![0u8; byte_len];
        let base = buf
            .offset
            .saturating_add(ioff)
            .saturating_add(u64::from(first_index).saturating_mul(elem_bytes));
        self.read_mem(buf.memory, base, &mut raw);

        let mut verts = Vec::with_capacity(index_count as usize);
        for i in 0..(index_count as usize) {
            let idx: u32 = match itype {
                IndexType::Uint16 => u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]) as u32,
                IndexType::Uint32 => {
                    u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap())
                }
            };
            let vidx = (idx as i64 + vertex_offset as i64).max(0) as u32;
            if let Some(v) = self.fetch_vertex(vidx) {
                verts.push(v);
            }
        }
        self.rasterize_vertices(&verts);
    }

    fn rasterize_vertices(&mut self, verts: &[SoftVertex]) {
        if verts.len() < 3 {
            return;
        }
        let Some(target_img) = self.active_target else {
            return;
        };
        let Some(dst) = self.images.get(&target_img.0) else {
            return;
        };
        if dst.bpp != 4 && dst.bpp != 8 {
            return;
        }
        let (dst_mem, dst_base, dst_pitch, dst_fmt, dst_bpp) = (
            dst.memory,
            dst.offset,
            dst.row_pitch,
            dst.info.format,
            dst.bpp as usize,
        );
        let dst_w = self.active_extent[0].min(dst.info.extent[0]) as i32;
        let dst_h = self.active_extent[1].min(dst.info.extent[1]) as i32;
        if dst_w <= 0 || dst_h <= 0 || dst_w > 4096 || dst_h > 4096 {
            return;
        }
        let dst_is_bgra = matches!(dst_fmt, Format::B8G8R8A8Unorm | Format::B8G8R8A8Srgb);

        let blend_enable = self
            .active_pipeline
            .and_then(|p| self.pipelines.get(&p.0))
            .is_some_and(|p| p.blend_enable);

        let vp = self.viewport.unwrap_or(Viewport {
            x: 0.0,
            y: 0.0,
            width: dst_w as f32,
            height: dst_h as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        });

        let (mut clip_x0, mut clip_y0, mut clip_x1, mut clip_y1) = (0i32, 0i32, dst_w, dst_h);
        if let Some(sc) = self.scissor {
            clip_x0 = clip_x0.max(sc.offset[0]);
            clip_y0 = clip_y0.max(sc.offset[1]);
            clip_x1 = clip_x1.min(sc.offset[0].saturating_add(sc.extent[0] as i32));
            clip_y1 = clip_y1.min(sc.offset[1].saturating_add(sc.extent[1] as i32));
        }
        if clip_x0 >= clip_x1 || clip_y0 >= clip_y1 {
            return;
        }

        let is_simple_vertex = self
            .active_pipeline
            .and_then(|p| self.pipelines.get(&p.0))
            .is_some_and(|p| p.is_simple_vertex);

        // Detect whether UVs are normalized [0..1] or pixel-space.
        let uv_normalized = verts
            .iter()
            .all(|v| v.uv[0] >= -0.05 && v.uv[0] <= 1.05 && v.uv[1] >= -0.05 && v.uv[1] <= 1.05);

        // Snapshot only the needed row band of the bound fragment texture (and optional XR10
        // plane-1 alpha) so small cursor/wheel draws don't copy 4-5 MB per draw.
        let mut tex_is_xr10 = false;
        let mut alpha_snap: Option<(Vec<u8>, u32, u32, u64, usize)> = None;
        let tex_snap: Option<(Vec<u8>, u32, u32, u32, u64, Format, usize)> =
            self.bound_fragment_texture().and_then(|(img, alpha_img, is_xr10)| {
                tex_is_xr10 = is_xr10;
                let ti = self.images.get(&img.0)?;
                let tw = ti.info.extent[0];
                let th = ti.info.extent[1].max(1);
                let tbpp = ti.bpp;
                let tpitch = ti.row_pitch;
                if tw == 0 || th == 0 || tpitch == 0 {
                    return None;
                }
                let (mut ty0, mut ty1) = (th as i32 - 1, 0i32);
                for v in verts {
                    let ty = if uv_normalized {
                        ((v.uv[1] * th as f32) as i32).clamp(0, th as i32 - 1)
                    } else {
                        (th as i32 - 1 - v.uv[1] as i32).clamp(0, th as i32 - 1)
                    };
                    ty0 = ty0.min(ty);
                    ty1 = ty1.max(ty);
                }
                let band_y0 = ty0.saturating_sub(1).max(0) as usize;
                let band_y1 = (ty1 + 2).min(th as i32) as usize;
                let band_h = band_y1.saturating_sub(band_y0).max(1);
                let total = (tpitch as usize).saturating_mul(band_h);
                if total == 0 || total > 16 * 1024 * 1024 {
                    return None;
                }
                let mut data = vec![0u8; total];
                self.read_mem(
                    ti.memory,
                    ti.offset.saturating_add((band_y0 as u64).saturating_mul(tpitch)),
                    &mut data,
                );
                if let Some(aimg) = alpha_img
                    && let Some(ai) = self.images.get(&aimg.0)
                {
                    let aw = ai.info.extent[0];
                    let ah = ai.info.extent[1].max(1);
                    let apitch = ai.row_pitch;
                    let aby0 = band_y0.min(ah.saturating_sub(1) as usize);
                    let aby1 = band_y1.min(ah as usize).max(aby0 + 1);
                    let atotal = (apitch as usize).saturating_mul(aby1 - aby0);
                    if aw > 0 && ah > 0 && atotal > 0 && atotal <= 8 * 1024 * 1024 {
                        let mut adata = vec![0u8; atotal];
                        self.read_mem(
                            ai.memory,
                            ai.offset.saturating_add((aby0 as u64).saturating_mul(apitch)),
                            &mut adata,
                        );
                        alpha_snap = Some((adata, aw, ah, apitch, aby0));
                    }
                }
                Some((data, tw, th, tbpp, tpitch, ti.info.format, band_y0))
            });

        // Detect whether vertex positions are normalized NDC [-1..1] or already pixel-space.
        let is_ndc = verts
            .iter()
            .all(|v| v.pos[0].abs() <= 1.5 && v.pos[1].abs() <= 1.5);
        // For pixel-space offscreen layers (e.g. 1280x30 menu bar with y=770..800),
        // rebase coordinates that start past the target's extent into 0..extent.
        let mut shift_x = 0.0f32;
        let mut shift_y = 0.0f32;
        if !is_ndc {
            let vmin_x = verts.iter().map(|v| v.pos[0]).fold(f32::INFINITY, f32::min);
            let vmin_y = verts.iter().map(|v| v.pos[1]).fold(f32::INFINITY, f32::min);
            if vmin_x >= dst_w as f32 {
                shift_x = vmin_x;
            }
            if vmin_y >= dst_h as f32 {
                shift_y = vmin_y;
            }
        }
        let to_screen = |v: &SoftVertex| -> (f32, f32) {
            let inv_w = if v.pos[3].abs() > 1e-6 && is_ndc {
                1.0 / v.pos[3]
            } else {
                1.0
            };
            let x = v.pos[0] * inv_w;
            let y = v.pos[1] * inv_w;
            if is_ndc {
                (
                    vp.x + (x * 0.5 + 0.5) * vp.width,
                    vp.y + (y * 0.5 + 0.5) * vp.height,
                )
            } else {
                // QuartzCore pixel-space coordinates use bottom-left origin (y=0 at bottom,
                // y=dst_h at top), whereas the framebuffer uses top-left origin.
                (x - shift_x, dst_h as f32 - (y - shift_y))
            }
        };

        let mut tris: Vec<(usize, usize, usize)> = Vec::new();
        match self.topology {
            PrimitiveTopology::TriangleStrip => {
                for i in 0..verts.len().saturating_sub(2) {
                    tris.push((i, i + 1, i + 2));
                }
            }
            _ => {
                let mut i = 0;
                while i + 2 < verts.len() {
                    tris.push((i, i + 1, i + 2));
                    i += 3;
                }
            }
        }

        if let Some(log) = self.log
            && (self.draws < 6
                || (is_simple_vertex && self.draws < 100)
                || (dst_w >= 1024 && dst_h >= 768 && self.textured_screen_draws < 25))
        {
            if dst_w >= 1024 && dst_h >= 768 {
                self.textured_screen_draws += 1;
            }
            let v0 = &verts[0];
            let has_tex = tex_snap.as_ref().map(|t| (t.1, t.2, t.3, t.5));
            let nz = tex_snap.as_ref().map_or(0usize, |t| {
                let step = (t.0.len() / 64).max(1);
                (0..64)
                    .filter(|&k| t.0.get(k * step).copied().unwrap_or(0) != 0)
                    .count()
            });
            let msg = alloc::format!(
                "DRAW #{} simple={} verts={} dst={}x{} ndc={} uv_norm={} xr10={} a_plane={} nz={}/64 tex={:?} v0.pos=({:.2},{:.2},{:.2},{:.2}) v0.uv=({:.2},{:.2}) v0.col=({:.2},{:.2},{:.2},{:.2})",
                self.draws + 1,
                is_simple_vertex,
                verts.len(),
                dst_w,
                dst_h,
                is_ndc,
                uv_normalized,
                tex_is_xr10,
                alpha_snap.is_some(),
                nz,
                has_tex,
                v0.pos[0],
                v0.pos[1],
                v0.pos[2],
                v0.pos[3],
                v0.uv[0],
                v0.uv[1],
                v0.color[0],
                v0.color[1],
                v0.color[2],
                v0.color[3],
            );
            log(&msg);
        }

        let mut row_buf = Vec::new();
        for (i0, i1, i2) in tris {
            let (v0, v1, v2) = (&verts[i0], &verts[i1], &verts[i2]);
            let (x0, y0) = to_screen(v0);
            let (x1, y1) = to_screen(v1);
            let (x2, y2) = to_screen(v2);

            let area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
            if area.abs() < 0.01 {
                continue;
            }
            let inv_area = 1.0 / area;

            let min_x = (x0.min(x1).min(x2) as i32).max(clip_x0);
            let max_x = ((x0.max(x1).max(x2) as i32) + 1).min(clip_x1);
            let min_y = (y0.min(y1).min(y2) as i32).max(clip_y0);
            let max_y = ((y0.max(y1).max(y2) as i32) + 1).min(clip_y1);
            if min_x >= max_x || min_y >= max_y {
                continue;
            }
            self.dirty_y = Some(match self.dirty_y {
                Some((y0, y1)) => (y0.min(min_y as u32), y1.max(max_y as u32)),
                None => (min_y as u32, max_y as u32),
            });
            let span_w = (max_x - min_x) as usize;
            row_buf.resize(span_w * dst_bpp, 0u8);

            for py in min_y..max_y {
                let row_off = dst_base
                    .saturating_add((py as u64).saturating_mul(dst_pitch))
                    .saturating_add((min_x as u64) * (dst_bpp as u64));
                self.read_mem(dst_mem, row_off, &mut row_buf);
                let fy = py as f32 + 0.5;
                let mut touched = false;

                for dx in 0..span_w {
                    let fx = (min_x + dx as i32) as f32 + 0.5;
                    let w0 = ((x1 - fx) * (y2 - fy) - (y1 - fy) * (x2 - fx)) * inv_area;
                    let w1 = ((x2 - fx) * (y0 - fy) - (y2 - fy) * (x0 - fx)) * inv_area;
                    let w2 = 1.0 - w0 - w1;
                    if w0 < -1e-4 || w1 < -1e-4 || w2 < -1e-4 {
                        continue;
                    }

                    let mut r = v0.color[0] * w0 + v1.color[0] * w1 + v2.color[0] * w2;
                    let mut g = v0.color[1] * w0 + v1.color[1] * w1 + v2.color[1] * w2;
                    let mut b = v0.color[2] * w0 + v1.color[2] * w1 + v2.color[2] * w2;
                    let mut a = v0.color[3] * w0 + v1.color[3] * w1 + v2.color[3] * w2;

                    if let Some((ref tdata, tw, th, tbpp, tpitch, tfmt, band_y0)) = tex_snap {
                        let u = v0.uv[0] * w0 + v1.uv[0] * w1 + v2.uv[0] * w2;
                        let v = v0.uv[1] * w0 + v1.uv[1] * w1 + v2.uv[1] * w2;
                        let tx = if uv_normalized {
                            ((u * tw as f32) as i32).clamp(0, tw as i32 - 1) as usize
                        } else {
                            (u as i32).clamp(0, tw as i32 - 1) as usize
                        };
                        let ty = if uv_normalized {
                            ((v * th as f32) as i32).clamp(0, th as i32 - 1) as usize
                        } else {
                            (th as i32 - 1 - v as i32).clamp(0, th as i32 - 1) as usize
                        };
                        let toff = ty.saturating_sub(band_y0) * (tpitch as usize) + tx * (tbpp as usize);
                        if tbpp == 4 && toff + 4 <= tdata.len() {
                            let (tr, tg, tb, ta) = if tex_is_xr10 {
                                let w = u32::from_le_bytes([
                                    tdata[toff],
                                    tdata[toff + 1],
                                    tdata[toff + 2],
                                    tdata[toff + 3],
                                ]);
                                let b_v = (((w & 0x3FF) as f32 - 384.0) / 510.0).clamp(0.0, 1.0);
                                let g_v = ((((w >> 10) & 0x3FF) as f32 - 384.0) / 510.0)
                                    .clamp(0.0, 1.0);
                                let r_v = ((((w >> 20) & 0x3FF) as f32 - 384.0) / 510.0)
                                    .clamp(0.0, 1.0);
                                let a_v = if let Some((ref adata, aw, ah, apitch, aby0)) = alpha_snap {
                                    let ax = tx.min(aw.saturating_sub(1) as usize);
                                    let ay = ty.min(ah.saturating_sub(1) as usize);
                                    let aoff = ay.saturating_sub(aby0) * (apitch as usize) + ax;
                                    f32::from(adata.get(aoff).copied().unwrap_or(255)) / 255.0
                                } else {
                                    1.0
                                };
                                (r_v, g_v, b_v, a_v)
                            } else {
                                decode_pixel_4(tfmt, &tdata[toff..toff + 4])
                            };
                            r *= tr;
                            g *= tg;
                            b *= tb;
                            a *= ta;
                        } else if tbpp == 8 && toff + 8 <= tdata.len() {
                            r *= f16_to_f32(u16::from_le_bytes([tdata[toff], tdata[toff + 1]]));
                            g *= f16_to_f32(u16::from_le_bytes([tdata[toff + 2], tdata[toff + 3]]));
                            b *= f16_to_f32(u16::from_le_bytes([tdata[toff + 4], tdata[toff + 5]]));
                            a *= f16_to_f32(u16::from_le_bytes([tdata[toff + 6], tdata[toff + 7]]));
                        } else if tbpp == 1 && toff < tdata.len() {
                            let ta = f32::from(tdata[toff]) / 255.0;
                            a *= ta;
                        }
                    }

                    let po = dx * dst_bpp;
                    if blend_enable {
                        let (dr, dg, db, da) = if dst_bpp == 8 {
                            (
                                f16_to_f32(u16::from_le_bytes([row_buf[po], row_buf[po + 1]])),
                                f16_to_f32(u16::from_le_bytes([row_buf[po + 2], row_buf[po + 3]])),
                                f16_to_f32(u16::from_le_bytes([row_buf[po + 4], row_buf[po + 5]])),
                                f16_to_f32(u16::from_le_bytes([row_buf[po + 6], row_buf[po + 7]])),
                            )
                        } else {
                            decode_pixel_4(dst_fmt, &row_buf[po..po + 4])
                        };
                        let inv_a = 1.0 - a.clamp(0.0, 1.0);
                        r = r + dr * inv_a;
                        g = g + dg * inv_a;
                        b = b + db * inv_a;
                        a = a + da * inv_a;
                    }

                    if dst_bpp == 8 {
                        let px8 = encode_rgba16f(r, g, b, a);
                        row_buf[po..po + 8].copy_from_slice(&px8);
                    } else {
                        let px4 = encode_pixel_4(dst_fmt, r, g, b, a);
                        row_buf[po..po + 4].copy_from_slice(&px4);
                    }
                    let _ = dst_is_bgra;
                    touched = true;
                }
                if touched {
                    self.write_mem(dst_mem, row_off, &row_buf);
                }
            }
        }
        self.draws += 1;
    }
    fn copy_convert_to_display(&mut self, src: Image, dst: Image) {
        let (Some(si), Some(di)) = (self.images.get(&src.0), self.images.get(&dst.0)) else {
            return;
        };
        let w = si.info.extent[0].min(di.info.extent[0]) as usize;
        let h = si.info.extent[1].min(di.info.extent[1]) as usize;
        if w == 0 || h == 0 || w > 4096 || h > 4096 {
            return;
        }
        let (sm, sb, sp, sbpp, sfmt) = (si.memory, si.offset, si.row_pitch, si.bpp as usize, si.info.format);
        let (dm, db, dp, dbpp, dfmt) = (di.memory, di.offset, di.row_pitch, di.bpp as usize, di.info.format);
        if sbpp == dbpp && sfmt == dfmt {
            let mut line = vec![0u8; w * sbpp];
            for y in 0..h {
                self.read_mem(sm, sb + (y as u64) * sp, &mut line);
                self.write_mem(dm, db + (y as u64) * dp, &line);
            }
            return;
        }
        if sbpp == 8 && dbpp == 4 {
            let dst_bgra = matches!(dfmt, Format::B8G8R8A8Unorm | Format::B8G8R8A8Srgb);
            let mut sline = vec![0u8; w * 8];
            let mut dline = vec![0u8; w * 4];
            for y in 0..h {
                self.read_mem(sm, sb + (y as u64) * sp, &mut sline);
                for x in 0..w {
                    let r = clamp_u8(f16_to_f32(u16::from_le_bytes([sline[x * 8], sline[x * 8 + 1]])));
                    let g = clamp_u8(f16_to_f32(u16::from_le_bytes([sline[x * 8 + 2], sline[x * 8 + 3]])));
                    let b = clamp_u8(f16_to_f32(u16::from_le_bytes([sline[x * 8 + 4], sline[x * 8 + 5]])));
                    let a = clamp_u8(f16_to_f32(u16::from_le_bytes([sline[x * 8 + 6], sline[x * 8 + 7]])));
                    if dst_bgra {
                        dline[x * 4] = b;
                        dline[x * 4 + 1] = g;
                        dline[x * 4 + 2] = r;
                        dline[x * 4 + 3] = a;
                    } else {
                        dline[x * 4] = r;
                        dline[x * 4 + 1] = g;
                        dline[x * 4 + 2] = b;
                        dline[x * 4 + 3] = a;
                    }
                }
                self.write_mem(dm, db + (y as u64) * dp, &dline);
            }
        }
    }
    fn copy_convert_to_guest_mem(
        &mut self,
        src: Image,
        dst_mem: Memory,
        dst_bytes: u64,
        y_range: (u32, u32),
    ) {
        let Some(si) = self.images.get(&src.0) else {
            return;
        };
        let w = si.info.extent[0] as usize;
        let h = si.info.extent[1] as usize;
        if w == 0 || h == 0 || w > 4096 || h > 4096 {
            return;
        }
        let y0 = (y_range.0 as usize).min(h);
        let y1 = (y_range.1 as usize).min(h);
        if y0 >= y1 {
            return;
        }
        let (sm, sb, sp, sbpp) = (si.memory, si.offset, si.row_pitch, si.bpp as usize);
        let plane0_bytes = (w as u64) * (h as u64) * 4;
        // Apple two-plane RGB10A8_2P_XR10 ('&b38'): 5,259,264 bytes for 1280x800
        // (plane 0 = 10-bit BGR XR10 at 4 B/px, plane 1 = 8-bit alpha at 1 B/px).
        let is_xr10 = dst_bytes >= plane0_bytes + (w as u64) * (h as u64)
            && dst_bytes < (w as u64) * (h as u64) * 8;
        if is_xr10 && sbpp == 8 {
            const XR10_BLACK: [u8; 4] = (384u32 | (384u32 << 10) | (384u32 << 20)).to_le_bytes();
            let dp = (w * 4) as u64;
            let mut sline = vec![0u8; w * 8];
            let mut dline = vec![0u8; w * 4];
            for y in y0..y1 {
                self.read_mem(sm, sb + (y as u64) * sp, &mut sline);
                for x in 0..w {
                    let o = x * 8;
                    if sline[o] == 0
                        && sline[o + 1] == 0
                        && sline[o + 2] == 0
                        && sline[o + 3] == 0
                        && sline[o + 4] == 0
                        && sline[o + 5] == 0
                    {
                        dline[x * 4..x * 4 + 4].copy_from_slice(&XR10_BLACK);
                        continue;
                    }
                    let r = f16_to_f32(u16::from_le_bytes([sline[o], sline[o + 1]]));
                    let g = f16_to_f32(u16::from_le_bytes([sline[o + 2], sline[o + 3]]));
                    let b = f16_to_f32(u16::from_le_bytes([sline[o + 4], sline[o + 5]]));
                    let b10 = (384.0 + b.clamp(0.0, 1.0) * 510.0 + 0.5) as u32 & 0x3FF;
                    let g10 = (384.0 + g.clamp(0.0, 1.0) * 510.0 + 0.5) as u32 & 0x3FF;
                    let r10 = (384.0 + r.clamp(0.0, 1.0) * 510.0 + 0.5) as u32 & 0x3FF;
                    let word = b10 | (g10 << 10) | (r10 << 20);
                    dline[x * 4..x * 4 + 4].copy_from_slice(&word.to_le_bytes());
                }
                self.write_mem(dst_mem, (y as u64) * dp, &dline);
            }
            return;
        }
        let dst_is_8bpp = dst_bytes >= (w as u64) * (h as u64) * 8;
        if sbpp == 8 && dst_is_8bpp {
            let dp = (w * 8) as u64;
            let mut line = vec![0u8; w * 8];
            for y in y0..y1 {
                self.read_mem(sm, sb + (y as u64) * sp, &mut line);
                self.write_mem(dst_mem, (y as u64) * dp, &line);
            }
        } else if sbpp == 8 {
            let dp = (w * 4) as u64;
            let mut sline = vec![0u8; w * 8];
            let mut dline = vec![0u8; w * 4];
            for y in y0..y1 {
                self.read_mem(sm, sb + (y as u64) * sp, &mut sline);
                for x in 0..w {
                    let r = clamp_u8(f16_to_f32(u16::from_le_bytes([sline[x * 8], sline[x * 8 + 1]])));
                    let g = clamp_u8(f16_to_f32(u16::from_le_bytes([sline[x * 8 + 2], sline[x * 8 + 3]])));
                    let b = clamp_u8(f16_to_f32(u16::from_le_bytes([sline[x * 8 + 4], sline[x * 8 + 5]])));
                    let a = clamp_u8(f16_to_f32(u16::from_le_bytes([sline[x * 8 + 6], sline[x * 8 + 7]])));
                    dline[x * 4] = b;
                    dline[x * 4 + 1] = g;
                    dline[x * 4 + 2] = r;
                    dline[x * 4 + 3] = a;
                }
                self.write_mem(dst_mem, (y as u64) * dp, &dline);
            }
        } else if sbpp == 4 {
            let dp = (w * 4) as u64;
            let mut line = vec![0u8; w * 4];
            for y in y0..y1 {
                self.read_mem(sm, sb + (y as u64) * sp, &mut line);
                self.write_mem(dst_mem, (y as u64) * dp, &line);
            }
        }
    }
}

fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let frac = (bits >> 13) & 0x03FF;
    if exp <= 0 {
        sign
    } else if exp >= 31 {
        sign | 0x7C00
    } else {
        sign | ((exp as u16) << 10) | (frac as u16)
    }
}

fn encode_rgba16f(r: f32, g: f32, b: f32, a: f32) -> [u8; 8] {
    let mut out = [0u8; 8];
    out[0..2].copy_from_slice(&f32_to_f16(r).to_le_bytes());
    out[2..4].copy_from_slice(&f32_to_f16(g).to_le_bytes());
    out[4..6].copy_from_slice(&f32_to_f16(b).to_le_bytes());
    out[6..8].copy_from_slice(&f32_to_f16(a).to_le_bytes());
    out
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = u32::from((bits >> 10) & 0x1F);
    let frac = u32::from(bits & 0x03FF);
    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign)
        } else {
            let mut e = 127 - 14;
            let mut f = frac;
            while (f & 0x0400) == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x03FF;
            f32::from_bits(sign | (e << 23) | (f << 13))
        }
    } else if exp == 31 {
        f32::from_bits(sign | (0xFF << 23) | (frac << 13))
    } else {
        f32::from_bits(sign | ((exp + 127 - 15) << 23) | (frac << 13))
    }
}

fn format_bpp(format: Format) -> u32 {
    match format {
        Format::R8Unorm | Format::R8Snorm | Format::R8Uint | Format::R8Sint | Format::R8Srgb => 1,
        Format::R8G8Unorm
        | Format::R8G8Snorm
        | Format::R8G8Uint
        | Format::R8G8Sint
        | Format::R16Unorm
        | Format::R16Snorm
        | Format::R16Uint
        | Format::R16Sint
        | Format::R16Sfloat
        | Format::R5G6B5UnormPack16
        | Format::R4G4B4A4UnormPack16
        | Format::A1R5G5B5UnormPack16 => 2,
        Format::R16G16B16A16Unorm
        | Format::R16G16B16A16Snorm
        | Format::R16G16B16A16Uint
        | Format::R16G16B16A16Sint
        | Format::R16G16B16A16Sfloat
        | Format::R32G32Uint
        | Format::R32G32Sint
        | Format::R32G32Sfloat => 8,
        Format::R32G32B32A32Uint | Format::R32G32B32A32Sint | Format::R32G32B32A32Sfloat => 16,
        _ => 4,
    }
}

fn clamp_u8(v: f32) -> u8 {
    let clamped = v.clamp(0.0, 1.0);
    (clamped * 255.0 + 0.5) as u8
}

fn encode_clear_pixel(format: Format, value: ClearValue) -> [u8; 4] {
    let (r, g, b, a) = match value {
        ClearValue::Float([r, g, b, a]) => (r, g, b, a),
        ClearValue::Uint([r, g, b, a]) => (
            (r as u8) as f32 / 255.0,
            (g as u8) as f32 / 255.0,
            (b as u8) as f32 / 255.0,
            (a as u8) as f32 / 255.0,
        ),
        ClearValue::Sint([r, g, b, a]) => (
            (r as u8) as f32 / 255.0,
            (g as u8) as f32 / 255.0,
            (b as u8) as f32 / 255.0,
            (a as u8) as f32 / 255.0,
        ),
        ClearValue::DepthStencil { .. } => (0.0, 0.0, 0.0, 1.0),
    };
    encode_pixel_4(format, r, g, b, a)
}

fn encode_pixel_4(format: Format, r: f32, g: f32, b: f32, a: f32) -> [u8; 4] {
    match format {
        Format::B8G8R8A8Unorm | Format::B8G8R8A8Srgb => {
            [clamp_u8(b), clamp_u8(g), clamp_u8(r), clamp_u8(a)]
        }
        Format::A2B10G10R10UnormPack32 => {
            let r10 = (r.clamp(0.0, 1.0) * 1023.0 + 0.5) as u32 & 0x3FF;
            let g10 = (g.clamp(0.0, 1.0) * 1023.0 + 0.5) as u32 & 0x3FF;
            let b10 = (b.clamp(0.0, 1.0) * 1023.0 + 0.5) as u32 & 0x3FF;
            let a2 = (a.clamp(0.0, 1.0) * 3.0 + 0.5) as u32 & 0x3;
            (r10 | (g10 << 10) | (b10 << 20) | (a2 << 30)).to_le_bytes()
        }
        Format::A2R10G10B10UnormPack32 => {
            let b10 = (b.clamp(0.0, 1.0) * 1023.0 + 0.5) as u32 & 0x3FF;
            let g10 = (g.clamp(0.0, 1.0) * 1023.0 + 0.5) as u32 & 0x3FF;
            let r10 = (r.clamp(0.0, 1.0) * 1023.0 + 0.5) as u32 & 0x3FF;
            let a2 = (a.clamp(0.0, 1.0) * 3.0 + 0.5) as u32 & 0x3;
            (b10 | (g10 << 10) | (r10 << 20) | (a2 << 30)).to_le_bytes()
        }
        _ => [clamp_u8(r), clamp_u8(g), clamp_u8(b), clamp_u8(a)],
    }
}

fn decode_pixel_4(format: Format, px: &[u8]) -> (f32, f32, f32, f32) {
    match format {
        Format::B8G8R8A8Unorm | Format::B8G8R8A8Srgb => (
            f32::from(px[2]) / 255.0,
            f32::from(px[1]) / 255.0,
            f32::from(px[0]) / 255.0,
            f32::from(px[3]) / 255.0,
        ),
        Format::A2B10G10R10UnormPack32 => {
            let w = u32::from_le_bytes([px[0], px[1], px[2], px[3]]);
            (
                (w & 0x3FF) as f32 / 1023.0,
                ((w >> 10) & 0x3FF) as f32 / 1023.0,
                ((w >> 20) & 0x3FF) as f32 / 1023.0,
                ((w >> 30) & 0x3) as f32 / 3.0,
            )
        }
        Format::A2R10G10B10UnormPack32 => {
            let w = u32::from_le_bytes([px[0], px[1], px[2], px[3]]);
            (
                ((w >> 20) & 0x3FF) as f32 / 1023.0,
                ((w >> 10) & 0x3FF) as f32 / 1023.0,
                (w & 0x3FF) as f32 / 1023.0,
                ((w >> 30) & 0x3) as f32 / 3.0,
            )
        }
        _ => (
            f32::from(px[0]) / 255.0,
            f32::from(px[1]) / 255.0,
            f32::from(px[2]) / 255.0,
            f32::from(px[3]) / 255.0,
        ),
    }
}

fn image_byte_size(info: &ImageInfo) -> (u32, u64, u64) {
    let bpp = format_bpp(info.format);
    let min_pitch = (info.extent[0].max(1) as u64).saturating_mul(bpp as u64);
    let row_pitch = info
        .linear
        .map_or(min_pitch, |l| l.row_pitch.max(min_pitch));
    let height = info.extent[1].max(1) as u64;
    let depth = info.extent[2].max(1) as u64;
    let layers = info.array_layers.max(1) as u64;
    let size = row_pitch
        .saturating_mul(height)
        .saturating_mul(depth)
        .saturating_mul(layers)
        .max(256);
    (bpp, row_pitch, size)
}

impl Backend for SoftBackend {
    fn physical_device(&self) -> &PhysicalDevice {
        &self.physical
    }

    fn outside(&self) -> Option<Arc<dyn Outside>> {
        None
    }

    fn buffer_requirements(&mut self, info: &BufferInfo) -> Result<Requirements> {
        Ok(Requirements {
            size: info.size.max(256).next_multiple_of(256),
            alignment: 256,
            memory_types: 0b1111,
        })
    }

    fn image_requirements(&mut self, info: &ImageInfo) -> Result<Requirements> {
        let (_, _, size) = image_byte_size(info);
        Ok(Requirements {
            size: size.next_multiple_of(4096),
            alignment: 4096,
            memory_types: 0b1111,
        })
    }

    fn allocate_memory(
        &mut self,
        size: u64,
        _: MemoryRequest,
        _: u32,
    ) -> Result<Memory> {
        let size = size.max(4096).next_multiple_of(4096);
        Ok(self.alloc_owned_memory(size))
    }

    fn import_memory(&mut self, ranges: &[MemoryRange], token: NonZeroU64) -> Result<Memory> {
        let total: u64 = ranges.iter().map(|r| r.bytes).sum();
        let id = self.alloc_id();
        self.memories.insert(
            id,
            SoftMemory::Guest {
                ranges: ranges.to_vec(),
                token: Some(token),
                total,
            },
        );
        if (5_000_000..=6_000_000).contains(&total) {
            self.guest_display_mems.push((Memory(id), total));
            if self.guest_display_mems.len() > 4 {
                self.guest_display_mems.remove(0);
            }
            // Initialize plane 1 (alpha) to 0xFF once on import instead of every frame.
            let alpha_chunk = vec![0xFFu8; 65536];
            let mut off = 4_096_000u64;
            while off < total {
                let n = ((total - off) as usize).min(alpha_chunk.len());
                self.write_mem(Memory(id), off, &alpha_chunk[..n]);
                off += n as u64;
            }
            if let Some(log) = self.log {
                let msg = alloc::format!("GUEST XR10 DISPLAY MEM id={} bytes={}", id, total);
                log(&msg);
            }
        }
        Ok(Memory(id))
    }

    fn import_own(&mut self, ranges: &[MemoryRange]) -> Result<Memory> {
        let total: u64 = ranges.iter().map(|r| r.bytes).sum();
        let id = self.alloc_id();
        self.memories.insert(
            id,
            SoftMemory::Guest {
                ranges: ranges.to_vec(),
                token: None,
                total,
            },
        );
        Ok(Memory(id))
    }

    fn retired(&mut self) -> Option<NonZeroU64> {
        self.retired.pop_front()
    }

    fn retiring(&self) -> bool {
        !self.retired.is_empty()
    }

    fn create_buffer(
        &mut self,
        info: &BufferInfo,
        _: MemoryRequest,
    ) -> Result<BufferAllocation> {
        let allocated = info.size.max(256).next_multiple_of(256);
        let memory = self.alloc_owned_memory(allocated);
        let id = self.alloc_id();
        self.buffers.insert(
            id,
            SoftBuffer {
                memory,
                offset: 0,
                size: info.size,
            },
        );
        Ok(BufferAllocation {
            buffer: Buffer(id),
            memory,
            address: 0x1000_0000_0000 + (id << 32),
            allocated,
        })
    }

    fn create_placed_buffer(
        &mut self,
        info: &BufferInfo,
        memory: Memory,
        offset: u64,
    ) -> Result<(Buffer, u64)> {
        let id = self.alloc_id();
        self.buffers.insert(
            id,
            SoftBuffer {
                memory,
                offset,
                size: info.size,
            },
        );
        Ok((Buffer(id), 0x1000_0000_0000 + (id << 32)))
    }

    fn create_image(
        &mut self,
        info: &ImageInfo,
        _: MemoryRequest,
    ) -> Result<(Image, Memory, u64)> {
        let (bpp, row_pitch, size) = image_byte_size(info);
        let allocated = size.next_multiple_of(4096);
        let memory = self.alloc_owned_memory(allocated);
        let id = self.alloc_id();
        self.images.insert(
            id,
            SoftImage {
                info: info.clone(),
                memory,
                offset: 0,
                bpp,
                row_pitch,
            },
        );
        Ok((Image(id), memory, allocated))
    }

    fn create_placed_image(
        &mut self,
        info: &ImageInfo,
        memory: Memory,
        offset: u64,
    ) -> Result<Image> {
        let (bpp, row_pitch, _) = image_byte_size(info);
        let id = self.alloc_id();
        let is_guest = matches!(self.memories.get(&memory.0), Some(SoftMemory::Guest { .. }));
        if is_guest && bpp >= 4 && info.extent[0] >= 1024 && info.extent[1] >= 768 {
            self.last_guest_display = Some(Image(id));
            if let Some(log) = self.log {
                let msg = alloc::format!(
                    "GUEST DISPLAY IMAGE id={} extent={}x{} fmt={:?} bpp={} pitch={}",
                    id, info.extent[0], info.extent[1], info.format, bpp, row_pitch
                );
                log(&msg);
            }
        }
        self.images.insert(
            id,
            SoftImage {
                info: info.clone(),
                memory,
                offset,
                bpp,
                row_pitch,
            },
        );
        Ok(Image(id))
    }

    fn clear_reused_image(&mut self, image: Image, _: &ImageInfo) {
        self.fill_image_color(image, ClearValue::Float([0.0; 4]));
    }

    fn create_sparse_image(
        &mut self,
        info: &ImageInfo,
    ) -> Result<(Image, SparseImageRequirements)> {
        let (bpp, row_pitch, _) = image_byte_size(info);
        let id = self.alloc_id();
        self.images.insert(
            id,
            SoftImage {
                info: info.clone(),
                memory: Memory(0),
                offset: 0,
                bpp,
                row_pitch,
            },
        );
        Ok((
            Image(id),
            SparseImageRequirements {
                memory_types: 0b1111,
                granularity: [64, 64, 1],
                mip_tail_first_lod: 1,
                mip_tail_size: 4096,
                mip_tail_offset: 0,
                mip_tail_stride: 4096,
                single_mip_tail: true,
            },
        ))
    }

    fn create_image_view(&mut self, image: Image, _: &ImageViewInfo) -> Result<ImageView> {
        let id = self.alloc_id();
        self.image_views.insert(id, image);
        Ok(ImageView(id))
    }

    fn create_buffer_view(&mut self, _: Buffer, _: &BufferViewInfo) -> Result<BufferView> {
        Ok(BufferView(self.alloc_id()))
    }

    fn create_sampler(&mut self, _: &SamplerInfo) -> Result<Sampler> {
        Ok(Sampler(self.alloc_id()))
    }

    fn create_graphics_pipeline(&mut self, info: &GraphicsPipelineInfo) -> Result<Pipeline> {
        let id = self.alloc_id();
        let blend_enable = info.color_blend.first().is_some_and(|b| b.blend_enable);
        let is_simple_vertex = info
            .stages
            .iter()
            .any(|(_, code)| (code.0 & SIMPLE_VERTEX_CODE_BIT) != 0);
        self.pipelines.insert(
            id,
            SoftPipeline {
                vertex_input: info.vertex_input.clone(),
                topology: info.topology,
                blend_enable,
                is_simple_vertex,
            },
        );
        Ok(Pipeline(id))
    }

    fn create_compute_pipeline(&mut self, _: &ComputePipelineInfo) -> Result<Pipeline> {
        Ok(Pipeline(self.alloc_id()))
    }

    fn create_pipeline_cache(&mut self, _: &[u8]) -> Result<PipelineCache> {
        Ok(PipelineCache(self.alloc_id()))
    }

    fn pipeline_cache_data(&mut self, _: PipelineCache) -> Result<Vec<u8>> {
        Ok(Vec::new())
    }

    fn create_query_pool(&mut self, _: &QueryPoolInfo) -> Result<QueryPool> {
        Ok(QueryPool(self.alloc_id()))
    }

    fn create_semaphore(&mut self, initial: u64) -> Result<Semaphore> {
        let id = self.alloc_id();
        self.semaphores.insert(id, initial);
        Ok(Semaphore(id))
    }

    fn export_semaphore(&mut self, _: Semaphore) -> Option<i32> {
        None
    }

    fn create_indirect_commands_layout(
        &mut self,
        _: &IndirectCommandsLayoutInfo,
    ) -> Result<IndirectCommandsLayout> {
        Ok(IndirectCommandsLayout(self.alloc_id()))
    }

    fn create_execution_set(&mut self, _: Pipeline, _: u32) -> Result<ExecutionSet> {
        Ok(ExecutionSet(self.alloc_id()))
    }

    fn write_execution_set(&mut self, _: ExecutionSet, _: u32, _: Pipeline) {}

    fn generated_commands_requirements(
        &mut self,
        _: IndirectCommandsLayout,
        _: Option<ExecutionSet>,
        sequences: u32,
        _: MemoryRequest,
    ) -> Result<Requirements> {
        Ok(Requirements {
            size: u64::from(sequences).max(1) * 64,
            alignment: 64,
            memory_types: 0b1111,
        })
    }

    fn destroy(&mut self, object: Object) {
        match object {
            Object::Memory(m) => {
                self.guest_display_mems.retain(|(mem, _)| *mem != m);
                if let Some(SoftMemory::Guest {
                    token: Some(token), ..
                }) = self.memories.remove(&m.0)
                {
                    self.retired.push_back(token);
                }
            }
            Object::Buffer(b) => {
                self.buffers.remove(&b.0);
            }
            Object::Image(i) => {
                self.images.remove(&i.0);
            }
            Object::ImageView(v) => {
                self.image_views.remove(&v.0);
            }
            Object::Pipeline(p) => {
                self.pipelines.remove(&p.0);
            }
            Object::Semaphore(s) => {
                self.semaphores.remove(&s.0);
            }
            _ => {}
        }
    }

    fn write_descriptor(&mut self, index: u32, descriptor: Descriptor) {
        if let Descriptor::SampledImage(view) = descriptor {
            self.sampled_images.insert(index, view);
        }
    }

    fn write_memory(&mut self, memory: Memory, offset: u64, bytes: &[u8]) -> Result<()> {
        self.write_mem(memory, offset, bytes);
        Ok(())
    }

    fn read_memory(&mut self, memory: Memory, offset: u64, bytes: &mut [u8]) -> Result<()> {
        self.read_mem(memory, offset, bytes);
        Ok(())
    }

    fn flush_memory(&mut self, _: Memory, _: u64, _: u64) -> Result<()> {
        Ok(())
    }

    fn invalidate_memory(&mut self, _: Memory, _: u64, _: u64) -> Result<()> {
        Ok(())
    }

    fn copy_to_image(
        &mut self,
        image: Image,
        region: &BufferImageCopy,
        bytes: &[u8],
    ) -> Result<()> {
        let Some(img) = self.images.get(&image.0) else {
            return Ok(());
        };
        let (memory, base, pitch, bpp) = (img.memory, img.offset, img.row_pitch, img.bpp as u64);
        let row_bytes = (region.extent[0] as usize).saturating_mul(bpp as usize);
        if row_bytes == 0 {
            return Ok(());
        }
        let dx = region.offset[0].max(0) as u64;
        let dy = region.offset[1].max(0) as u64;
        let rows = (region.extent[1] as usize).min(4096);
        for y in 0..rows {
            let src_start = y * row_bytes;
            let src_end = (src_start + row_bytes).min(bytes.len());
            if src_start >= src_end {
                break;
            }
            let doff = base
                .saturating_add((dy + y as u64).saturating_mul(pitch))
                .saturating_add(dx.saturating_mul(bpp));
            self.write_mem(memory, doff, &bytes[src_start..src_end]);
        }
        self.copies += 1;
        Ok(())
    }

    fn copy_from_image(
        &mut self,
        image: Image,
        region: &BufferImageCopy,
        bytes: &mut [u8],
    ) -> Result<()> {
        let Some(img) = self.images.get(&image.0) else {
            bytes.fill(0);
            return Ok(());
        };
        let (memory, base, pitch, bpp) = (img.memory, img.offset, img.row_pitch, img.bpp as u64);
        let row_bytes = (region.extent[0] as usize).saturating_mul(bpp as usize);
        if row_bytes == 0 {
            return Ok(());
        }
        let sx = region.offset[0].max(0) as u64;
        let sy = region.offset[1].max(0) as u64;
        let rows = (region.extent[1] as usize).min(4096);
        for y in 0..rows {
            let dst_start = y * row_bytes;
            let dst_end = (dst_start + row_bytes).min(bytes.len());
            if dst_start >= dst_end {
                break;
            }
            let soff = base
                .saturating_add((sy + y as u64).saturating_mul(pitch))
                .saturating_add(sx.saturating_mul(bpp));
            self.read_mem(memory, soff, &mut bytes[dst_start..dst_end]);
        }
        Ok(())
    }

    fn reset_queries(&mut self, _: QueryPool, _: u32, _: u32) {}

    fn query_results(&mut self, _: QueryPool, _: u32, results: &mut [u64]) -> Result<bool> {
        results.fill(1);
        Ok(true)
    }

    fn semaphore_value(&mut self, semaphore: Semaphore) -> Result<u64> {
        Ok(self.semaphores.get(&semaphore.0).copied().unwrap_or(0))
    }

    fn signal_semaphore(&mut self, semaphore: Semaphore, value: u64) -> Result<()> {
        let entry = self.semaphores.entry(semaphore.0).or_default();
        *entry = (*entry).max(value);
        Ok(())
    }

    fn wait_semaphore(&mut self, semaphore: Semaphore, value: u64, _: u64) -> Result<bool> {
        Ok(self.semaphores.get(&semaphore.0).copied().unwrap_or(0) >= value)
    }

    fn submit(&mut self, segments: &[Segment]) -> Result<()> {
        self.submissions += 1;
        for segment in segments {
            for command in &segment.commands {
                match command {
                    Command::BeginRendering(rendering) => {
                        self.passes += 1;
                        self.active_extent = rendering.extent;
                        self.active_target = rendering
                            .color
                            .first()
                            .and_then(|slot| slot.as_ref())
                            .and_then(|att| self.image_views.get(&att.view.0).copied());
                        self.active_resolve = rendering
                            .color
                            .first()
                            .and_then(|slot| slot.as_ref())
                            .and_then(|att| att.resolve)
                            .and_then(|(_, view)| self.image_views.get(&view.0).copied());
                        if let Some(log) = self.log
                            && (self.passes <= 15
                                || rendering.extent[0] >= 1024
                                    && rendering.extent[1] >= 768
                                    && self.passes <= 30)
                        {
                            let guest = self
                                .active_target
                                .and_then(|i| self.images.get(&i.0))
                                .is_some_and(|im| {
                                    matches!(
                                        self.memories.get(&im.memory.0),
                                        Some(SoftMemory::Guest { .. })
                                    )
                                });
                            let resolve_guest = self
                                .active_resolve
                                .and_then(|i| self.images.get(&i.0))
                                .is_some_and(|im| {
                                    matches!(
                                        self.memories.get(&im.memory.0),
                                        Some(SoftMemory::Guest { .. })
                                    )
                                });
                            let load = rendering
                                .color
                                .first()
                                .and_then(|s| s.as_ref())
                                .map(|a| a.load_op);
                            let (fmt, bpp) = self
                                .active_target
                                .and_then(|i| self.images.get(&i.0))
                                .map_or((None, 0), |im| (Some(im.info.format), im.bpp));
                            let msg = alloc::format!(
                                "PASS #{} extent={}x{} fmt={:?} bpp={} guest={} resolve={}({}) load={:?}",
                                self.passes,
                                rendering.extent[0],
                                rendering.extent[1],
                                fmt,
                                bpp,
                                guest,
                                self.active_resolve.is_some(),
                                resolve_guest,
                                load
                            );
                            log(&msg);
                        }
                        self.dirty_y = None;
                        for att in rendering.color.iter().flatten() {
                            if att.load_op == AttachmentLoadOp::Clear
                                && let Some(&img) = self.image_views.get(&att.view.0)
                            {
                                self.fill_image_color(img, att.clear);
                                self.dirty_y = Some((0, rendering.extent[1]));
                            }
                        }
                    }
                    Command::EndRendering => {
                        if let (Some(src), Some(dst)) = (self.active_target, self.active_resolve) {
                            self.copy_convert_to_display(src, dst);
                        } else if let Some(src) = self.active_target
                            && let Some(y_range) = self.dirty_y
                            && self.active_extent[0] >= 1024
                            && self.active_extent[1] >= 768
                        {
                            if !self.guest_display_mems.is_empty() {
                                let targets = self.guest_display_mems.clone();
                                for (dst_mem, dst_bytes) in targets {
                                    self.copy_convert_to_guest_mem(src, dst_mem, dst_bytes, y_range);
                                }
                            } else if let Some(dst) = self.last_guest_display
                                && src != dst
                            {
                                self.copy_convert_to_display(src, dst);
                            }
                        }
                        self.active_target = None;
                        self.active_resolve = None;
                        self.dirty_y = None;
                    }
                    Command::BindPipeline {
                        bind_point,
                        pipeline,
                    } => {
                        if *bind_point == PipelineBindPoint::Graphics {
                            self.active_pipeline = Some(*pipeline);
                            if let Some(p) = self.pipelines.get(&pipeline.0) {
                                self.topology = p.topology;
                            }
                        }
                    }
                    Command::PushConstants(pc) => {
                        self.push_constants = *pc;
                    }
                    Command::SetViewport(vp) => {
                        self.viewport = Some(*vp);
                    }
                    Command::SetViewports(vps) => {
                        if let Some(vp) = vps.first() {
                            self.viewport = Some(*vp);
                        }
                    }
                    Command::SetScissor(sc) => {
                        self.scissor = Some(*sc);
                    }
                    Command::SetScissors(scs) => {
                        if let Some(sc) = scs.first() {
                            self.scissor = Some(*sc);
                        }
                    }
                    Command::SetPrimitiveTopology(topo) => {
                        self.topology = *topo;
                    }
                    Command::BindVertexBuffers { first, buffers } => {
                        for (i, vb) in buffers.iter().enumerate() {
                            self.vertex_buffers.insert(first + i as u32, *vb);
                        }
                    }
                    Command::BindIndexBuffer {
                        buffer,
                        offset,
                        index_type,
                    } => {
                        self.index_buffer = Some((*buffer, *offset, *index_type));
                    }
                    Command::Draw {
                        vertex_count,
                        first_vertex,
                        ..
                    } => {
                        self.exec_draw(*vertex_count, *first_vertex);
                    }
                    Command::DrawIndexed {
                        index_count,
                        first_index,
                        vertex_offset,
                        ..
                    } => {
                        self.exec_draw_indexed(*index_count, *first_index, *vertex_offset);
                    }
                    Command::Dispatch(grid) => {
                        self.dispatches += 1;
                        if let Some(log) = self.log
                            && self.dispatches <= 15
                        {
                            let msg = alloc::format!(
                                "DISPATCH #{} grid=({},{},{})",
                                self.dispatches,
                                grid[0],
                                grid[1],
                                grid[2]
                            );
                            log(&msg);
                        }
                    }
                    Command::ClearColorImage { image, value } => {
                        self.fill_image_color(*image, *value);
                    }
                    Command::CopyBuffer { src, dst, region } => {
                        self.exec_copy_buffer(*src, *dst, *region);
                    }
                    Command::CopyBufferRegions { src, dst, regions } => {
                        for region in regions {
                            self.exec_copy_buffer(*src, *dst, *region);
                        }
                    }
                    Command::CopyBufferToImage { src, dst, region } => {
                        self.exec_copy_buffer_to_image(*src, *dst, region);
                    }
                    Command::CopyImageToBuffer { src, dst, region } => {
                        self.exec_copy_image_to_buffer(*src, *dst, region);
                    }
                    Command::CopyImage { src, dst, regions } => {
                        self.exec_copy_image(*src, *dst, regions);
                    }
                    Command::BlitImage {
                        src, dst, regions, ..
                    } => {
                        self.exec_blit_image(*src, *dst, regions);
                    }
                    Command::FillBuffer {
                        buffer,
                        offset,
                        size,
                        data,
                    } => {
                        if let Some(buf) = self.buffers.get(&buffer.0) {
                            let (mem, base) = (buf.memory, buf.offset.saturating_add(*offset));
                            let count = (*size as usize / 4).min(MAX_OWNED_BYTES / 4);
                            let word = data.to_le_bytes();
                            let mut tmp = vec![0u8; count * 4];
                            for chunk in tmp.chunks_exact_mut(4) {
                                chunk.copy_from_slice(&word);
                            }
                            self.write_mem(mem, base, &tmp);
                        }
                    }
                    _ => {}
                }
            }
            for &(semaphore, value) in &segment.signals {
                let entry = self.semaphores.entry(semaphore.0).or_default();
                *entry = (*entry).max(value);
            }
        }
        Ok(())
    }

    fn calibrated_timestamps(&mut self) -> Result<(u64, u64)> {
        Ok((self.submissions, self.submissions))
    }
}

/// Pure-Rust software [`Compiler`] that parses real macOS `.metallib` containers
/// using [`Library::parse`] to report exact function names and stages without
/// requiring LLVM or Mesa NIR.
#[derive(Default)]
pub struct SoftCompiler {
    libraries: Vec<Option<Arc<Vec<u8>>>>,
    next_code: u64,
}

fn mtlb_kind_to_function_type(kind: u8) -> Option<FunctionType> {
    match kind {
        0 => Some(FunctionType::Vertex),
        1 => Some(FunctionType::Fragment),
        2 => Some(FunctionType::Kernel),
        4 => Some(FunctionType::Visible),
        6 => Some(FunctionType::Intersection),
        7 => Some(FunctionType::Mesh),
        8 => Some(FunctionType::Object),
        _ => None,
    }
}

fn empty_interface(stage: Stage) -> Interface {
    Interface {
        stage,
        patch: None,
        buffers: Vec::new(),
        textures: Vec::new(),
        samplers: Vec::new(),
        threadgroup: Vec::new(),
        attributes: Vec::new(),
        declared_attributes: Vec::new(),
        varyings: Vec::new(),
        color_outputs: Vec::new(),
        writes_depth: false,
        writes_stencil: false,
        writes_sample_mask: false,
        discards: false,
        early_fragment_tests: false,
        static_threadgroup_bytes: 0,
        imageblock_bytes: 0,
        max_total_threads: None,
        amplification: false,
        command_barriers: false,
        attachment_reads: 0,
        storage_attachments: 0,
        buffer_reads: 0,
        constants: Vec::new(),
        features: ShaderFeatures::EMPTY,
    }
}

impl SoftCompiler {
    fn lookup_function(&self, library: u64, entry: &str) -> Result<FunctionType> {
        let data = self.library_data(library)?;
        if let Ok(parsed) = Library::parse(data) {
            for f in parsed.functions() {
                if f.name == entry {
                    return Ok(mtlb_kind_to_function_type(f.kind).unwrap_or(FunctionType::Kernel));
                }
            }
            return Err(NO_FUNCTION);
        }
        // Raw bitcode/unparsed fallback: guess stage from name.
        if entry.contains("vert") || entry.contains("Vert") || entry.ends_with("_vs") {
            Ok(FunctionType::Vertex)
        } else if entry.contains("frag") || entry.contains("Frag") || entry.ends_with("_fs") {
            Ok(FunctionType::Fragment)
        } else {
            Ok(FunctionType::Kernel)
        }
    }
}

impl Compiler for SoftCompiler {
    fn features(&self) -> ShaderFeatures {
        ShaderFeatures::KNOWN
    }

    fn add_library(&mut self, data: Vec<u8>) -> Result<u64> {
        self.libraries.push(Some(Arc::new(data)));
        Ok(self.libraries.len() as u64 - 1)
    }

    fn share_library(&mut self, library: u64) -> Result<u64> {
        let data = self
            .libraries
            .get(library as usize)
            .and_then(Option::as_ref)
            .cloned()
            .ok_or(Error::Invalid("unknown library"))?;
        self.libraries.push(Some(data));
        Ok(self.libraries.len() as u64 - 1)
    }

    fn functions(&mut self, data: &[u8]) -> Result<Vec<(String, FunctionType)>> {
        if let Ok(parsed) = Library::parse(data) {
            let mut out = Vec::new();
            for f in parsed.functions() {
                if f.kind == Function::EXTERN || f.kind == Function::EXPORTED {
                    continue;
                }
                if let Some(ft) = mtlb_kind_to_function_type(f.kind) {
                    out.push((String::from(f.name), ft));
                }
            }
            return Ok(out);
        }
        Ok(Vec::new())
    }

    fn exports(&mut self, data: &[u8]) -> Result<Vec<String>> {
        if let Ok(parsed) = Library::parse(data) {
            return Ok(parsed
                .functions()
                .iter()
                .filter(|f| f.kind == Function::EXPORTED)
                .map(|f| String::from(f.name))
                .collect());
        }
        Ok(Vec::new())
    }

    fn stitch(&mut self, _: &[LinkedFunction], _: &[StitchedGraph]) -> Result<Vec<u8>> {
        Ok(b"BC\xc0\xde\0\0\0\0".to_vec())
    }

    fn library_data(&self, library: u64) -> Result<&[u8]> {
        self.libraries
            .get(library as usize)
            .and_then(Option::as_deref)
            .map(Vec::as_slice)
            .ok_or(Error::Invalid("unknown library"))
    }

    fn remove_library(&mut self, library: u64) -> Result<()> {
        self.libraries
            .get_mut(library as usize)
            .and_then(Option::take)
            .map(drop)
            .ok_or(Error::Invalid("unknown library"))
    }

    fn constants(&mut self, _: u64, _: &str) -> Result<Vec<FunctionConstant>> {
        Ok(Vec::new())
    }

    fn reflect(&mut self, library: u64, entry: &str, _: &Specialization) -> Result<Reflection> {
        let ft = self.lookup_function(library, entry)?;
        match Stage::from_function_type(ft) {
            Some(stage) => Ok(Reflection::Stage(Box::new(empty_interface(stage)))),
            None => Ok(Reflection::Visible {
                constants: Vec::new(),
                command_barriers: false,
            }),
        }
    }

    fn bindings(&mut self, _: u64, _: &str, _: Option<&Specialization>) -> Result<Vec<Binding>> {
        Ok(Vec::new())
    }

    fn sampler_states(&mut self, _: u64, _: &str, _: &Specialization) -> Result<Vec<u64>> {
        Ok(Vec::new())
    }

    fn compile(
        &mut self,
        library: u64,
        entry: &str,
        _: &Specialization,
    ) -> Result<CompiledShader> {
        let ft = self.lookup_function(library, entry)?;
        let stage = Stage::from_function_type(ft).unwrap_or(Stage::Kernel);
        self.next_code += 1;
        let code_id = if entry.contains("SimpleVertex") {
            self.next_code | SIMPLE_VERTEX_CODE_BIT
        } else {
            self.next_code
        };
        Ok(CompiledShader {
            code: ShaderCode(code_id),
            interface: empty_interface(stage),
            provenance: Provenance::default(),
        })
    }

    fn compile_tessellation(
        &mut self,
        library: u64,
        entry: &str,
        spec: &Specialization,
        _: &Tessellation,
    ) -> Result<TessellationStages> {
        let evaluation = self.compile(library, entry, spec)?;
        self.next_code += 2;
        Ok(TessellationStages {
            vertex: ShaderCode(self.next_code - 1),
            control: ShaderCode(self.next_code),
            evaluation,
        })
    }

    fn release(&mut self, _: ShaderCode) {}
}

fn soft_chip() -> &'static Chip {
    &CHIPS[0]
}

fn soft_physical_device(chip: &Chip, vram: u64) -> PhysicalDevice {
    let budget = vram.saturating_sub(64 << 20).max(vram / 2);
    PhysicalDevice {
        identity: Identity {
            api_version: (1 << 22) | (4 << 12),
            driver_id: DriverId::MesaNvk,
            driver_version: 1,
            vendor_id: 0x1af4,
            device_id: 0x1050,
            pipeline_cache_uuid: [0x53; 16],
            device_type: PhysicalDeviceType::IntegratedGpu,
            device_name: String::from("Stela SoftGPU (QEMU virtio-gpu)"),
            conformance_version: match chip.architecture {
                Architecture::Hopper => [0; 4],
                Architecture::Blackwell if !chip.display => [0; 4],
                _ => [1, 4, 3, 0],
            },
            pci_bus: Some(PciBusInfo {
                domain: 0,
                bus: 0,
                device: 3,
                function: 0,
            }),
        },
        limits: soft_limits(),
        features: soft_features(chip),
        memory: PhysicalMemory {
            heaps: vec![
                MemoryHeap {
                    size: vram,
                    flags: MemoryHeapFlags::DEVICE_LOCAL,
                    budget: Some(budget),
                },
                MemoryHeap {
                    size: 1 << 30,
                    flags: MemoryHeapFlags::EMPTY,
                    budget: Some(768 << 20),
                },
            ],
            types: vec![
                MemoryType {
                    properties: M::DEVICE_LOCAL,
                    heap_index: 0,
                },
                MemoryType {
                    properties: M::HOST_VISIBLE | M::HOST_COHERENT,
                    heap_index: 1,
                },
                MemoryType {
                    properties: M::HOST_VISIBLE | M::HOST_COHERENT | M::HOST_CACHED,
                    heap_index: 1,
                },
                MemoryType {
                    properties: M::DEVICE_LOCAL | M::HOST_VISIBLE | M::HOST_COHERENT,
                    heap_index: 0,
                },
            ],
        },
        formats: soft_formats(),
    }
}

fn soft_limits() -> Limits {
    Limits {
        max_image_dimension_1d: 0x8000,
        max_image_dimension_2d: 0x8000,
        max_image_dimension_3d: 0x4000,
        max_image_dimension_cube: 0x8000,
        max_image_array_layers: 2048,
        max_texel_buffer_elements: 128 << 20,
        max_uniform_buffer_range: 65536,
        max_storage_buffer_range: u32::MAX,
        max_push_constants_size: 256,
        max_memory_allocation_count: 4096,
        max_sampler_allocation_count: 4000,
        buffer_image_granularity: 0x400,
        sparse_address_space_size: 1 << 39,
        max_per_stage_update_after_bind_samplers: DESCRIPTORS,
        max_per_stage_update_after_bind_sampled_images: DESCRIPTORS,
        max_per_stage_update_after_bind_storage_images: DESCRIPTORS,
        max_per_stage_update_after_bind_storage_buffers: DESCRIPTORS,
        max_descriptor_set_update_after_bind_samplers: DESCRIPTORS,
        max_descriptor_set_update_after_bind_sampled_images: DESCRIPTORS,
        max_descriptor_set_update_after_bind_storage_images: DESCRIPTORS,
        max_vertex_input_attributes: 32,
        max_vertex_input_bindings: 32,
        max_vertex_input_attribute_offset: 2047,
        max_vertex_input_binding_stride: 2048,
        max_vertex_output_components: 128,
        max_tessellation_generation_level: 64,
        max_tessellation_patch_size: 32,
        max_fragment_input_components: 128,
        max_fragment_output_attachments: 8,
        max_fragment_dual_src_attachments: 1,
        max_compute_shared_memory_size: 48 << 10,
        max_compute_work_group_count: [0x7fff_ffff, 65535, 65535],
        max_compute_work_group_invocations: 1024,
        max_compute_work_group_size: [1024, 1024, 64],
        max_sampler_lod_bias: 15.0,
        max_sampler_anisotropy: 16.0,
        max_viewports: 16,
        max_viewport_dimensions: [32768, 32768],
        min_texel_buffer_offset_alignment: 16,
        min_uniform_buffer_offset_alignment: 64,
        min_storage_buffer_offset_alignment: 16,
        max_framebuffer_width: 0x8000,
        max_framebuffer_height: 0x8000,
        max_framebuffer_layers: 2048,
        framebuffer_color_sample_counts: SAMPLE_COUNTS,
        framebuffer_depth_sample_counts: SAMPLE_COUNTS,
        framebuffer_stencil_sample_counts: SAMPLE_COUNTS,
        framebuffer_no_attachments_sample_counts: SAMPLE_COUNTS,
        max_color_attachments: 8,
        sampled_image_color_sample_counts: SAMPLE_COUNTS,
        sampled_image_integer_sample_counts: SAMPLE_COUNTS,
        sampled_image_depth_sample_counts: SAMPLE_COUNTS,
        storage_image_sample_counts: SAMPLE_COUNTS,
        max_sample_mask_words: 1,
        timestamp_compute_and_graphics: true,
        timestamp_period: 1.0,
        max_clip_distances: 8,
        max_cull_distances: 8,
        point_size_range: [1.0, 2047.94],
        line_width_range: [1.0, 64.0],
        non_coherent_atom_size: 64,
        subgroup_size: 32,
        min_subgroup_size: 32,
        max_subgroup_size: 32,
        subgroup_supported_stages: ShaderStages::KNOWN,
        subgroup_supported_operations: SubgroupFeatures::KNOWN,
        subgroup_quad_operations_in_all_stages: true,
        max_multiview_view_count: 32,
        max_memory_allocation_size: 1 << 31,
        max_buffer_size: 1 << 32,
        storage_texel_buffer_offset_alignment_bytes: 16,
        storage_texel_buffer_offset_single_texel_alignment: true,
        uniform_texel_buffer_offset_alignment_bytes: 16,
        uniform_texel_buffer_offset_single_texel_alignment: true,
        supported_depth_resolve_modes: ResolveModes::KNOWN,
        supported_stencil_resolve_modes: ResolveModes::SAMPLE_ZERO
            | ResolveModes::MIN
            | ResolveModes::MAX,
        independent_resolve_none: true,
        independent_resolve: true,
        max_vertex_attrib_divisor: u32::MAX,
        sample_locations: Some(SampleLocationLimits {
            sample_counts: SAMPLE_COUNTS.difference(SampleCounts::C1),
            max_grid_size: [1, 1],
            coordinate_range: [0.0, 0.9375],
            sub_pixel_bits: 4,
            variable_sample_locations: true,
        }),
        mesh: Some(MeshLimits {
            max_task_work_group_total_count: 1 << 22,
            max_task_work_group_count: [1 << 22, 65535, 65535],
            max_task_work_group_invocations: 128,
            max_task_payload_size: 1 << 14,
            max_mesh_work_group_total_count: 1 << 22,
            max_mesh_work_group_count: [1 << 22, 65535, 65535],
            max_mesh_work_group_invocations: 128,
            max_mesh_output_vertices: 256,
            max_mesh_output_primitives: 256,
        }),
        generated_commands: Some(GeneratedCommandsLimits {
            max_pipelines: u32::MAX,
            max_sequences: 1 << 20,
            max_tokens: 16,
            max_token_offset: 2047,
            max_stride: 1 << 12,
            pipeline_stages: ShaderStages::KNOWN.difference(ShaderStages::GEOMETRY),
        }),
        dma_buf_import: true,
        stencil_export: false,
        sparse: SparseLimits {
            residency_standard_2d_block_shape: true,
            residency_standard_2d_multisample_block_shape: true,
            residency_standard_3d_block_shape: true,
            residency_aligned_mip_size: false,
            residency_non_resident_strict: true,
        },
    }
}

fn soft_features(chip: &Chip) -> Features {
    Features {
        image_cube_array: true,
        independent_blend: true,
        geometry_shader: true,
        tessellation_shader: true,
        sample_rate_shading: true,
        dual_src_blend: true,
        alpha_to_one: true,
        multi_draw_indirect: true,
        draw_indirect_first_instance: true,
        depth_clamp: true,
        depth_bias_clamp: true,
        fill_mode_non_solid: true,
        wide_lines: true,
        large_points: true,
        multi_viewport: true,
        sampler_anisotropy: true,
        texture_compression_bc: true,
        occlusion_query_precise: true,
        pipeline_statistics_query: true,
        vertex_pipeline_stores_and_atomics: true,
        fragment_stores_and_atomics: true,
        shader_image_gather_extended: true,
        shader_storage_image_multisample: true,
        shader_storage_image_read_without_format: true,
        shader_storage_image_write_without_format: true,
        shader_clip_distance: true,
        shader_cull_distance: true,
        shader_float64: true,
        shader_int64: true,
        shader_int16: true,
        shader_float16: true,
        shader_int8: true,
        shader_resource_residency: true,
        shader_resource_min_lod: true,
        sparse_binding: true,
        sparse_residency_buffer: true,
        sparse_residency_image_2d: true,
        sparse_residency_image_3d: true,
        sparse_residency_samples: SAMPLE_COUNTS.difference(SampleCounts::C1),
        sparse_residency_aliased: true,
        storage_buffer_8bit_access: true,
        storage_buffer_16bit_access: true,
        multiview: true,
        sampler_mirror_clamp_to_edge: true,
        sampler_filter_minmax: true,
        descriptor_indexing: true,
        runtime_descriptor_array: true,
        descriptor_binding_partially_bound: true,
        descriptor_binding_update_unused_while_pending: true,
        descriptor_binding_sampled_image_update_after_bind: true,
        descriptor_binding_storage_image_update_after_bind: true,
        descriptor_binding_uniform_texel_buffer_update_after_bind: true,
        descriptor_binding_storage_texel_buffer_update_after_bind: true,
        shader_sampled_image_array_non_uniform_indexing: true,
        shader_storage_image_array_non_uniform_indexing: true,
        buffer_device_address: true,
        timeline_semaphore: true,
        vulkan_memory_model: true,
        vulkan_memory_model_device_scope: true,
        shader_buffer_int64_atomics: true,
        shader_shared_int64_atomics: true,
        shader_image_int64_atomics: true,
        shader_buffer_float32_atomic_add: true,
        shader_shared_float32_atomic_add: true,
        shader_image_float32_atomics: true,
        shader_demote_to_helper_invocation: true,
        subgroup_size_control: true,
        compute_full_subgroups: true,
        synchronization2: true,
        dynamic_rendering: true,
        dynamic_rendering_local_read: true,
        maintenance4: true,
        maintenance5: true,
        maintenance8: true,
        host_image_copy: true,
        extended_dynamic_state: true,
        extended_dynamic_state2: true,
        extended_dynamic_state3_polygon_mode: true,
        extended_dynamic_state3_depth_clamp_enable: true,
        extended_dynamic_state3_depth_clip_enable: true,
        vertex_input_dynamic_state: true,
        vertex_attribute_instance_rate_divisor: true,
        depth_clip_enable: true,
        bresenham_lines: true,
        rectangular_lines: true,
        image_view_min_lod: true,
        fragment_shader_barycentric: true,
        task_shader: true,
        mesh_shader: true,
        fragment_shader_pixel_interlock: true,
        unified_image_layouts: true,
        null_descriptor: true,
        dynamic_rendering_unused_attachments: true,
        dynamic_primitive_topology_unrestricted: true,
        host_query_reset: true,
        conditional_rendering: true,
        image_drm_format_modifier: true,
        shader_float16_vector_atomics: true,
        memory_budget: true,
        calibrated_timestamps: true,
        cooperative_matrix: chip.chipset != 0x167 && chip.chipset != 0x168,
        ..Features::default()
    }
}

fn soft_formats() -> FormatTable {
    let mut table = FormatTable::default();
    let transfer = F::TRANSFER_SRC | F::TRANSFER_DST | F::BLIT_SRC | F::HOST_IMAGE_TRANSFER;
    for format in PixelFormat::ALL {
        let description = format.describe();
        let Ok(realization) = description.realization else {
            continue;
        };
        let block_bytes = description.block.bytes;
        let support = match (description.kind, description.compression) {
            (_, Compression::Bc) => FormatSupport {
                optimal: F::SAMPLED_IMAGE | F::SAMPLED_IMAGE_FILTER_LINEAR | transfer,
                sample_counts: SampleCounts::C1,
                sparse: SampleCounts::C1,
                ..FormatSupport::default()
            },
            (_, Compression::None)
                if description.is_color() && block_bytes % 3 == 0 && block_bytes != 3 * 4 =>
            {
                FormatSupport {
                    buffer: F::VERTEX_BUFFER | F::UNIFORM_TEXEL_BUFFER,
                    ..FormatSupport::default()
                }
            }
            (Kind::Color(numeric), Compression::None) => {
                let texture_only = matches!(
                    realization.format,
                    Format::R8Srgb | Format::R4G4B4A4UnormPack16 | Format::E5B9G9R9UfloatPack32
                );
                let renderable = block_bytes != 12 && !texture_only;
                let filter = !numeric.is_integer();
                let atomic = matches!(
                    realization.format,
                    Format::R32Uint | Format::R32Sint | Format::R32Sfloat
                );
                let optimal = F::SAMPLED_IMAGE
                    .union(transfer)
                    .with(
                        F::SAMPLED_IMAGE_FILTER_LINEAR | F::SAMPLED_IMAGE_FILTER_MINMAX,
                        filter,
                    )
                    .with(
                        F::STORAGE_IMAGE
                            | F::STORAGE_READ_WITHOUT_FORMAT
                            | F::STORAGE_WRITE_WITHOUT_FORMAT,
                        renderable,
                    )
                    .with(F::COLOR_ATTACHMENT | F::BLIT_DST, renderable)
                    .with(F::COLOR_ATTACHMENT_BLEND, renderable && filter)
                    .with(F::STORAGE_IMAGE_ATOMIC, atomic);
                FormatSupport {
                    optimal,
                    linear: optimal.difference(F::STORAGE_IMAGE_ATOMIC),
                    modifier_linear: optimal.difference(F::STORAGE_IMAGE_ATOMIC),
                    buffer: F::VERTEX_BUFFER | F::UNIFORM_TEXEL_BUFFER | F::STORAGE_TEXEL_BUFFER,
                    sample_counts: if renderable {
                        SAMPLE_COUNTS
                    } else {
                        SampleCounts::C1
                    },
                    sparse: if renderable {
                        SAMPLE_COUNTS
                    } else {
                        SampleCounts::EMPTY
                    },
                }
            }
            (_, Compression::None) => FormatSupport {
                optimal: F::SAMPLED_IMAGE
                    | F::SAMPLED_IMAGE_FILTER_LINEAR
                    | F::SAMPLED_IMAGE_DEPTH_COMPARISON
                    | F::DEPTH_STENCIL_ATTACHMENT
                    | transfer.difference(F::HOST_IMAGE_TRANSFER),
                sample_counts: SAMPLE_COUNTS,
                ..FormatSupport::default()
            },
            _ => continue,
        };
        table.insert(realization.format, support);
    }
    table.insert(
        Format::A2B10G10R10SnormPack32,
        FormatSupport {
            buffer: F::VERTEX_BUFFER,
            ..FormatSupport::default()
        },
    );
    let rgb9e5 = table.get(Format::E5B9G9R9UfloatPack32);
    table.insert(
        Format::E5B9G9R9UfloatPack32,
        FormatSupport {
            buffer: rgb9e5.buffer.difference(F::VERTEX_BUFFER),
            ..rgb9e5
        },
    );
    table
}
