//! Aligned copies of linear textures whose rows the GPU cannot address in
//! place.
//!
//! Metal takes a buffer's or IOSurface's rows at any whole number of pixels
//! a row that the device's linear alignment allows, and macOS lays out
//! IOSurface planes to Apple GPUs' 16 bytes: an '&b38' screen surface's
//! alpha plane is one byte a pixel, its row the screen's width, as 3440 or
//! 1680. NVIDIA's linear images need 32-byte row pitches and offsets
//! ([`resource::linear_layout_fits`]). Such a plane's texture is an image in
//! the GPU's own layout instead, its aligned copy, which copies keep equal
//! to the rows, the guest's memory, where the CPU, the display and other
//! textures read and write them:
//!
//! - a command buffer begins by copying an aligned copy's rows into it when
//!   anything may have written the rows since: the CPU, the display, another
//!   texture over the same surface. Until a command buffer reports the
//!   texture as an IOSurface object it uses (`SurfaceUse`), that is every
//!   command buffer, since nothing tells when the rows change: a buffer's
//!   contents the CPU writes in place. Once reported, the copy is refreshed
//!   only by a command buffer that uses it, and only when the surface's
//!   seed differs from the one it was copied at or a command buffer wrote
//!   the surface through another object since. Before, every command
//!   buffer copied every such plane: 10.4 million texels each on an AMD
//!   iGPU, whose 256-byte linear rows a 4K video wallpaper's planes miss,
//!   half of its time while the desktop animated;
//! - a command buffer that writes a texture (attaches it to a pass, blits
//!   into it, binds it for shader writes) ends by copying the aligned copy
//!   back into its rows;
//! - a pass that encodes a two-plane extended-range texture's planes, which
//!   it writes in the rows themselves, copies the rows into the aligned
//!   copies after it, for the command buffer's later reads; a rendered one
//!   on a GPU of its own memory keeps a device copy of its planes instead
//!   ([`PlaneCopy`](crate::metal::pass::PlaneCopy)), whose encodes write it
//!   with the rows, and which a command buffer that writes it unreported
//!   fills first while nothing filled it yet;
//! - the CPU's copies of such a texture convert row by row in the rows.
//!
//! A texture whose rows fit keeps its linear image placed in the rows: no
//! copy at all.
//!
//! A large IOSurface texture the GPU only samples, of a two-plane format it
//! cannot render (a video frame or a video wallpaper's 10-bit 4:2:0
//! surface), is kept in device memory as well, though its rows fit: sampled
//! in place, every texel cache miss is a bus read of guest RAM with no L2
//! caching, so one 3840x2160 surface cost a Mission Control frame 10-19 ms of
//! GPU time on the 5080. Its copy is refreshed only when the surface
//! changed since it was last copied, whether or not its rows fit, and is
//! current at the start of every command buffer, which may sample it named
//! or through an argument buffer or a residency set:
//! - the guest reports each IOSurface object a command buffer uses with the
//!   surface's ID and seed as committed (`SurfaceUse`), and each other live
//!   texture the service keeps a copy of (`TextureResponse::copied`) whose
//!   seed changed since a commit last reported it; the CPU's writes change
//!   the seed, so the command buffer begins by refreshing those copies;
//! - a command buffer that writes a surface (a pass attachment, a blit
//!   destination, a texture shaders may write, any buffer over it) ends by
//!   refreshing every copy over it but the one texture over the surface it
//!   wrote whose copy then holds the rows: its end copies the copy back
//!   into them, or its encode wrote them and refreshed the copy after;
//! - the service's own writes of the rows for the CPU (`replaceRegion`),
//!   which change no seed, make the next command buffer refresh the copies
//!   over them.
use super::{Device, TextureId};
use crate::metal::{
    abi::TextureUsage,
    command_buffer::CommandBuffer,
    error::{Error, Result},
    hazard::FULL,
    pixel_format::Encoding,
    resource::{self, PlaneLayout, TexturePlan},
    shader::Compiler,
    vulkan::{
        ImageAspects, MemoryProperties,
        backend::Backend,
        command::{BufferImageCopy, Command, Subresource},
        object::{Buffer, Image, Memory, MemoryRequest},
    },
};
use alloc::vec::Vec;

/// A plane's aligned copy: its image and memory, and the copy between it
/// and the plane's rows.
#[derive(Debug)]
pub(in crate::metal::device) struct AlignedCopy {
    memory: Memory,
    bytes: u64,
    rows: RowCopy,
    refresh: Refresh,
    /// The rows' memory and where they start in it, for the CPU's copies.
    rows_memory: Memory,
    place: u64,
    bytes_per_row: u64,
    texel_bytes: u64,
    /// A plane of an extended-range IOSurface texture's device copy
    /// ([`PlaneCopy`]), whose memory the texture keeps and whose encodes
    /// write it with the rows: nothing refreshes it after them.
    ///
    /// [`PlaneCopy`]: crate::metal::pass::PlaneCopy
    device: bool,
    /// The service wrote the rows for the CPU since the copy was filled
    /// ([`Device::write_aligned_rows`]), which changes no seed: the next
    /// command buffer refreshes it, whether or not it names the texture.
    rows_written: bool,
}

/// When an aligned copy is refreshed from its rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Refresh {
    /// By every command buffer: rows the GPU cannot address in place, of a
    /// texture no command buffer reported over an IOSurface yet. The first
    /// report makes it [`Refresh::Changed`].
    Always,
    /// By a command buffer that uses it whose surface's seed differs from
    /// `filled`, the seed it was last copied at (`None`: never copied, or
    /// written by the GPU since); `surface` is the surface's ID once the
    /// guest reported it.
    Changed {
        surface: Option<u32>,
        filled: Option<u32>,
    },
}

/// An IOSurface object a command buffer uses, as the guest reported it:
/// the texture (`None` for a buffer) and the surface's ID and seed at its
/// commit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SurfaceNote {
    pub texture: Option<TextureId>,
    pub surface: u32,
    pub seed: u32,
}

/// The fewest texels of a sampled-only two-plane surface's first plane kept
/// in device memory: a 1920x1080 frame's. Smaller ones cost less sampled in
/// place than their copies.
pub(in crate::metal::device) const DEVICE_COPY_TEXELS: u64 = 1920 * 1080;

/// A copy between a plane's rows in `buffer` and its aligned copy `image`,
/// both ways: `region` names the rows by their offset and their pitch in
/// texels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowCopy {
    pub buffer: Buffer,
    pub image: Image,
    pub region: BufferImageCopy,
}

impl RowCopy {
    /// The copy into the aligned copy, which reads the rows.
    pub fn refresh(&self) -> Command {
        Command::CopyBufferToImage {
            src: self.buffer,
            dst: self.image,
            region: self.region,
        }
    }

    /// The copy back into the rows.
    pub fn write_back(&self) -> Command {
        Command::CopyImageToBuffer {
            src: self.image,
            dst: self.buffer,
            region: self.region,
        }
    }
}

/// Where a plane's rows lie: in `buffer`, bound `memory_offset` bytes into
/// `memory`, at `layout`, of `texel_bytes` pixels.
#[derive(Clone, Copy, Debug)]
pub(in crate::metal::device) struct Rows {
    pub buffer: Buffer,
    pub memory: Memory,
    pub memory_offset: u64,
    pub layout: PlaneLayout,
    pub texel_bytes: u64,
}

impl<B: Backend, C: Compiler> Device<B, C> {
    /// The image of a linear plane `plan` over `rows`: its linear image
    /// placed in the rows when the GPU addresses them in place and `sampled`
    /// is false, else its aligned copy, which the caller keeps with the
    /// texture ([`Self::keep_aligned_copies`]) or releases
    /// ([`Self::release_aligned_copies`]) once it destroyed the image. With
    /// `sampled`, a plane gets a copy, whether or not its rows fit,
    /// refreshed only where used and changed ([`Refresh::Changed`]).
    pub(in crate::metal::device) fn plane_image(
        &mut self,
        plan: &TexturePlan,
        rows: Rows,
        sampled: bool,
    ) -> Result<(Image, Option<AlignedCopy>)> {
        let Rows {
            buffer,
            memory,
            memory_offset,
            layout,
            texel_bytes,
        } = rows;
        let place = memory_offset
            .checked_add(layout.offset)
            .ok_or(Error::Invalid("the rows lie past the memory"))?;
        let fits = resource::linear_layout_fits(&self.capabilities, place, layout.bytes_per_row);
        if fits && !sampled {
            let image = self
                .backend
                .create_placed_image(&plan.image, memory, place)?;
            return Ok((image, None));
        }
        // A sampled plane is an IOSurface's, which every use reports.
        let refresh = if sampled {
            Refresh::Changed {
                surface: None,
                filled: None,
            }
        } else {
            Refresh::Always
        };
        let row_length = u32::try_from(layout.bytes_per_row / texel_bytes.max(1))
            .map_err(|_| Error::Invalid("bytesPerRow exceeds the row pitch range"))?;
        let info = resource::aligned_copy(&plan.image);
        let request = MemoryRequest {
            required: MemoryProperties::DEVICE_LOCAL,
            preferred: MemoryProperties::EMPTY,
            mapped: false,
        };
        let (image, copy_memory, bytes) = self.make_image(&info, request)?;
        self.allocated += bytes;
        self.book_memory(copy_memory, bytes, || {
            alloc::format!(
                "aligned copy of a {}x{} plane",
                info.extent[0],
                info.extent[1]
            )
        });
        let [width, height, _] = info.extent;
        let region = BufferImageCopy {
            buffer_offset: layout.offset,
            buffer_row_length: row_length,
            buffer_image_height: 0,
            subresource: Subresource {
                aspects: ImageAspects::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            offset: [0; 3],
            extent: [width, height, 1],
        };
        Ok((
            image,
            Some(AlignedCopy {
                memory: copy_memory,
                bytes,
                rows: RowCopy {
                    buffer,
                    image,
                    region,
                },
                refresh,
                rows_memory: memory,
                place,
                bytes_per_row: layout.bytes_per_row,
                texel_bytes,
                device: false,
                rows_written: false,
            }),
        ))
    }

    /// The aligned copy of a plane of an extended-range IOSurface texture's
    /// device copy: `image`, placed in device memory the texture keeps,
    /// refreshed from the plane's `rows` where used and changed
    /// ([`Refresh::Changed`]) and written with them by encodes.
    pub(in crate::metal::device) fn device_plane_copy(
        &self,
        image: Image,
        rows: Rows,
        extent: [u32; 2],
    ) -> Result<AlignedCopy> {
        let Rows {
            buffer,
            memory,
            memory_offset,
            layout,
            texel_bytes,
        } = rows;
        let place = memory_offset
            .checked_add(layout.offset)
            .ok_or(Error::Invalid("the rows lie past the memory"))?;
        let row_length = u32::try_from(layout.bytes_per_row / texel_bytes.max(1))
            .map_err(|_| Error::Invalid("bytesPerRow exceeds the row pitch range"))?;
        Ok(AlignedCopy {
            memory,
            bytes: 0,
            rows: RowCopy {
                buffer,
                image,
                region: BufferImageCopy {
                    buffer_offset: layout.offset,
                    buffer_row_length: row_length,
                    buffer_image_height: 0,
                    subresource: Subresource {
                        aspects: ImageAspects::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    offset: [0; 3],
                    extent: [extent[0], extent[1], 1],
                },
            },
            refresh: Refresh::Changed {
                surface: None,
                filled: None,
            },
            rows_memory: memory,
            place,
            bytes_per_row: layout.bytes_per_row,
            texel_bytes,
            device: true,
            rows_written: false,
        })
    }

    /// Releases the memory of aligned copies whose images are destroyed.
    pub(in crate::metal::device) fn release_aligned_copies(
        &mut self,
        copies: impl IntoIterator<Item = AlignedCopy>,
    ) {
        // A device copy's memory is its texture's.
        for copy in copies.into_iter().filter(|copy| !copy.device) {
            self.release_memory(copy.memory, copy.bytes);
        }
    }

    /// Keeps texture `id`'s aligned copies, which commits then refresh.
    pub(in crate::metal::device) fn keep_aligned_copies(
        &mut self,
        id: TextureId,
        copies: Vec<AlignedCopy>,
    ) {
        if copies.is_empty() {
            return;
        }
        // Shader writes reach the rows from command buffers that bind the
        // texture or attach it; one written only through an argument
        // buffer leaves them in the aligned copy.
        let writable = self.resources.textures.get(id.0).is_ok_and(|texture| {
            texture
                .descriptor
                .usage
                .contains(TextureUsage::SHADER_WRITE)
        });
        if writable {
            self.unsupported(
                "shader writes through argument buffers alone to textures in rows the GPU \
                 cannot address in place",
            );
        }
        self.resources.aligned_copies.insert(id.0, copies);
    }

    /// Takes texture `id`'s aligned copies as it is destroyed.
    pub(in crate::metal::device) fn take_aligned_copies(
        &mut self,
        id: TextureId,
    ) -> Option<Vec<AlignedCopy>> {
        self.resources.aligned_copies.remove(&id.0)
    }

    /// Whether texture `id`, or the texture it views, is an aligned copy.
    pub fn has_aligned_copy(&self, id: TextureId) -> bool {
        self.aligned_root(id).is_some()
    }

    /// The texture with aligned copies that `id` is or views.
    fn aligned_root(&self, id: TextureId) -> Option<TextureId> {
        let mut root = id;
        loop {
            if self.resources.aligned_copies.contains_key(&root.0) {
                return Some(root);
            }
            root = self.resources.textures.get(root.0).ok()?.parent?;
        }
    }

    /// Notes that `commands` writes texture `id`, which its end then copies
    /// back into the rows when it is an aligned copy. An extended-range
    /// IOSurface texture writes its rows itself.
    pub fn note_texture_write(&self, commands: &mut CommandBuffer, id: TextureId) {
        if !commands.written.contains(&id) {
            commands.written.push(id);
        }
        let Some(root) = self.aligned_root(id) else {
            return;
        };
        if self.has_extended_planes(root) {
            return;
        }
        if !commands.aligned_writes.contains(&root) {
            commands.aligned_writes.push(root);
        }
    }

    /// Notes a texture bound for shaders: one they may write is written.
    pub fn note_texture_binding(&self, commands: &mut CommandBuffer, id: TextureId) {
        let writes = self.resources.textures.get(id.0).is_ok_and(|texture| {
            texture
                .descriptor
                .usage
                .contains(TextureUsage::SHADER_WRITE)
        });
        if writes {
            self.note_texture_write(commands, id);
        }
    }

    /// The copies a command buffer begins with, and those it ends with; each
    /// group followed by a full barrier, and empty when there are none.
    ///
    /// It begins with every [`Refresh::Always`] copy, and each
    /// [`Refresh::Changed`] copy whose surface's seed differs from the one
    /// it was last filled at: of a texture it uses (`surfaces`), or of one
    /// kept in device memory it may reach unnamed whose seed changed
    /// (`seeds`, which the guest reports of every such live texture). A
    /// device copy no command buffer filled yet, of a texture it writes
    /// unreported, is filled first as well. A reported texture's `Always`
    /// copies become `Changed`.
    ///
    /// It ends by copying back into their rows the copies of the textures
    /// it wrote (`written`), then refreshing every other copy of a surface
    /// it may have written (`writes`: the textures it writes; any buffer
    /// over a surface), but for that surface's one texture it wrote whose
    /// copies hold the rows once it ends ([`Self::holds_rows_at_end`]). A
    /// copy is thus current at every command buffer's start whoever reads
    /// it, named or through an argument buffer: the CPU's writes reach it
    /// through the seed reports, the GPU's through these refreshes.
    pub(in crate::metal::device) fn aligned_copy_commands(
        &mut self,
        written: &[TextureId],
        surfaces: &[SurfaceNote],
        seeds: &[SurfaceNote],
        writes: &[TextureId],
    ) -> (Vec<Command>, Vec<Command>) {
        let used: Vec<(TextureId, SurfaceNote)> = surfaces
            .iter()
            .chain(seeds)
            .filter_map(|note| Some((self.aligned_root(note.texture?)?, *note)))
            .collect();
        // A reported use says the texture is an IOSurface's, whose writes
        // change its seed or come from command buffers, which report them.
        for (root, _) in &used {
            for copy in self
                .resources
                .aligned_copies
                .get_mut(&root.0)
                .into_iter()
                .flatten()
            {
                if copy.refresh == Refresh::Always {
                    copy.refresh = Refresh::Changed {
                        surface: None,
                        filled: None,
                    };
                }
            }
        }
        let mut lead: Vec<Command> = Vec::new();
        for copy in self.resources.aligned_copies.values_mut().flatten() {
            if copy.refresh == Refresh::Always || core::mem::take(&mut copy.rows_written) {
                lead.push(copy.rows.refresh());
            }
        }
        // A device copy no command buffer filled yet, of a texture this one
        // writes without reporting it, is filled first: encodes compare
        // with it.
        for id in writes {
            if used.iter().any(|(root, _)| root == id) {
                continue;
            }
            lead.extend(
                self.resources
                    .aligned_copies
                    .get(&id.0)
                    .into_iter()
                    .flatten()
                    .filter(|copy| {
                        copy.device && matches!(copy.refresh, Refresh::Changed { filled: None, .. })
                    })
                    .map(|copy| copy.rows.refresh()),
            );
        }
        for (root, note) in &used {
            let Some(copies) = self.resources.aligned_copies.get_mut(&root.0) else {
                continue;
            };
            for copy in copies {
                if let Refresh::Changed { surface, filled } = &mut copy.refresh {
                    *surface = Some(note.surface);
                    if *filled != Some(note.seed) {
                        *filled = Some(note.seed);
                        // Unless refreshed above for the CPU's writes.
                        if !lead.contains(&copy.rows.refresh()) {
                            lead.push(copy.rows.refresh());
                        }
                    }
                }
            }
        }
        let copies = &self.resources.aligned_copies;
        let mut tail: Vec<Command> = written
            .iter()
            .filter_map(|id| copies.get(&id.0))
            .flatten()
            .map(|copy| copy.rows.write_back())
            .collect();
        let dirty: Vec<&SurfaceNote> = surfaces
            .iter()
            .filter(|note| note.texture.is_none_or(|texture| writes.contains(&texture)))
            .collect();
        // Each surface written, its seed, and its one texture written whose
        // copies hold the rows once the command buffer ends.
        let mut written_surfaces: Vec<(u32, u32, Option<TextureId>)> = Vec::new();
        for note in &dirty {
            if written_surfaces
                .iter()
                .any(|(surface, ..)| *surface == note.surface)
            {
                continue;
            }
            let mut writers = dirty
                .iter()
                .filter(|other| other.surface == note.surface)
                .map(|other| other.texture.and_then(|texture| self.aligned_root(texture)));
            let first = writers.next().flatten();
            let sole = first.filter(|first| {
                writers.all(|other| other == Some(*first))
                    && self.holds_rows_at_end(*first, written)
            });
            written_surfaces.push((note.surface, note.seed, sole));
        }
        let mut refreshes = Vec::new();
        for (texture, copies) in &mut self.resources.aligned_copies {
            for copy in copies {
                let Refresh::Changed {
                    surface: Some(surface),
                    filled,
                } = &mut copy.refresh
                else {
                    continue;
                };
                let Some(&(_, seed, sole)) = written_surfaces
                    .iter()
                    .find(|(written, ..)| written == surface)
                else {
                    continue;
                };
                if sole.map(|sole| sole.0) != Some(*texture) {
                    refreshes.push(copy.rows.refresh());
                }
                *filled = Some(seed);
            }
        }
        if !refreshes.is_empty() {
            if !tail.is_empty() {
                tail.push(Command::Barrier(FULL));
            }
            tail.extend(refreshes);
        }
        for commands in [&mut lead, &mut tail] {
            if !commands.is_empty() {
                commands.push(Command::Barrier(FULL));
            }
        }
        (lead, tail)
    }

    /// Whether texture `root`'s aligned copies hold its rows once a command
    /// buffer that wrote it ends, `written` the textures whose copies its
    /// end copies back into their rows: such a texture, or an
    /// extended-range one, whose passes encode the rows and refresh the
    /// copies from them after.
    fn holds_rows_at_end(&self, root: TextureId, written: &[TextureId]) -> bool {
        written.contains(&root) || self.has_extended_planes(root)
    }

    /// The copies that refresh an extended-range IOSurface texture's
    /// planes' aligned copies after a pass encodes the planes: the first
    /// plane's and the second's.
    pub(in crate::metal::device) fn plane_refreshes(&self, id: TextureId) -> [Option<RowCopy>; 2] {
        let mut refreshes = [None; 2];
        let Some(copies) = self.resources.aligned_copies.get(&id.0) else {
            return refreshes;
        };
        let Ok(texture) = self.resources.textures.get(id.0) else {
            return refreshes;
        };
        for copy in copies.iter().filter(|copy| !copy.device) {
            if Some(copy.rows.image) == texture.image {
                refreshes[0] = Some(copy.rows);
            } else if Some(copy.rows.image) == texture.second_plane {
                refreshes[1] = Some(copy.rows);
            }
        }
        refreshes
    }

    /// Whether texture `id` keeps a device copy of its planes that holds
    /// their rows as of the surface's seed last reported: filled, and not
    /// written for the CPU by the service since.
    pub(in crate::metal::device) fn device_copy_holds_rows(&self, id: TextureId) -> bool {
        let mut copies = self
            .resources
            .aligned_copies
            .get(&id.0)
            .into_iter()
            .flatten()
            .filter(|copy| copy.device)
            .peekable();
        copies.peek().is_some()
            && copies.all(|copy| {
                !copy.rows_written
                    && matches!(
                        copy.refresh,
                        Refresh::Changed {
                            filled: Some(_),
                            ..
                        }
                    )
            })
    }

    /// The copies between an extended-range IOSurface texture's device
    /// copy and its planes' rows: the first plane's and the second's.
    pub(in crate::metal::device) fn device_copy_rows(&self, id: TextureId) -> [Option<RowCopy>; 2] {
        let mut refreshes = [None; 2];
        for (slot, copy) in self
            .resources
            .aligned_copies
            .get(&id.0)
            .into_iter()
            .flatten()
            .filter(|copy| copy.device)
            .enumerate()
            .take(2)
        {
            refreshes[slot] = Some(copy.rows);
        }
        refreshes
    }

    /// Writes `region` of an aligned copy's texture from tightly packed
    /// `bytes` into its rows, which the next commit copies into it.
    /// `Ok(false)` for a texture without an aligned copy.
    pub(super) fn write_aligned_rows(
        &mut self,
        id: TextureId,
        region: &BufferImageCopy,
        source: &super::SourceRows<'_>,
    ) -> Result<bool> {
        let bytes = source.packed();
        let Some((memory, rows)) = self.aligned_rows(id, region, bytes.len())? else {
            return Ok(false);
        };
        for (offset, packed) in rows {
            self.backend.write_memory(memory, offset, &bytes[packed])?;
        }
        // The next command buffer copies the rows again into its copies and
        // those of other textures over the same surface.
        let surfaces: Vec<u32> = self
            .resources
            .aligned_copies
            .get(&id.0)
            .into_iter()
            .flatten()
            .filter_map(|copy| match copy.refresh {
                Refresh::Changed { surface, .. } => surface,
                Refresh::Always => None,
            })
            .collect();
        for (texture, copies) in &mut self.resources.aligned_copies {
            for copy in copies {
                let over = match copy.refresh {
                    Refresh::Changed { surface, .. } => {
                        *texture == id.0 || surface.is_some_and(|known| surfaces.contains(&known))
                    }
                    Refresh::Always => false,
                };
                if over {
                    copy.rows_written = true;
                }
            }
        }
        Ok(true)
    }

    /// Reads `region` of an aligned copy's texture into tightly packed
    /// `bytes` from its rows. `Ok(false)` for a texture without an aligned
    /// copy.
    pub(super) fn read_aligned_rows(
        &mut self,
        id: TextureId,
        region: &BufferImageCopy,
        bytes: &mut [u8],
    ) -> Result<bool> {
        let Some((memory, rows)) = self.aligned_rows(id, region, bytes.len())? else {
            return Ok(false);
        };
        for (offset, packed) in rows {
            self.backend
                .read_memory(memory, offset, &mut bytes[packed])?;
        }
        Ok(true)
    }

    /// The rows' memory and each row of `region` in it, with its bytes in
    /// a packed copy of `length` bytes; `None` for a texture without an
    /// aligned copy. Only formats whose texels the rows hold as the image
    /// does are copied.
    #[allow(clippy::type_complexity)]
    fn aligned_rows(
        &self,
        id: TextureId,
        region: &BufferImageCopy,
        length: usize,
    ) -> Result<Option<(Memory, Vec<(u64, core::ops::Range<usize>)>)>> {
        if !self.resources.aligned_copies.contains_key(&id.0) {
            return Ok(None);
        }
        let texture = self.resources.textures.get(id.0)?;
        let copy = self
            .resources
            .aligned_copies
            .get(&id.0)
            .and_then(|copies| {
                copies
                    .iter()
                    .find(|copy| Some(copy.rows.image) == texture.image)
            })
            .ok_or(Error::Invalid(
                "the texture's image is not its aligned copy",
            ))?;
        if !matches!(
            texture.descriptor.pixel_format.describe().encoding(),
            Encoding::Exact
        ) {
            return Err(Error::Unsupported(
                "CPU copies of converted formats in rows the GPU cannot address in place",
            ));
        }
        let [x, y, _] = region.offset;
        let [width, height, depth] = region.extent;
        let (x, y) = (
            u64::try_from(x).map_err(|_| Error::Invalid("negative region origin"))?,
            u64::try_from(y).map_err(|_| Error::Invalid("negative region origin"))?,
        );
        let row_bytes = u64::from(width) * copy.texel_bytes;
        let rows = u64::from(height) * u64::from(depth.max(1));
        let packed = row_bytes
            .checked_mul(rows)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or(Error::Invalid("the region overflows"))?;
        if packed > length {
            return Err(Error::Invalid("the bytes end before the region"));
        }
        let ranges = (0..u64::from(height))
            .map(|row| {
                let offset = copy.place + (y + row) * copy.bytes_per_row + x * copy.texel_bytes;
                let start = (row * row_bytes) as usize;
                (offset, start..start + row_bytes as usize)
            })
            .collect();
        Ok(Some((copy.rows_memory, ranges)))
    }
}

#[cfg(test)]
mod tests;
