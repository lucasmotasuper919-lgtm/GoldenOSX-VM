use super::*;
use crate::metal::{
    abi::{
        CpuCacheMode, HazardTrackingMode, LoadAction, MultisampleDepthResolveFilter,
        MultisampleStencilResolveFilter, ResourceOptions, StorageMode, StoreAction,
        StoreActionOptions, SwizzleChannels, TextureType, VisibilityResultType,
    },
    command_buffer::blit::Region,
    device::{
        BufferId,
        fixture::{TestDevice, device},
        queue::{AttachmentRequest, RenderPassRequest},
        resources::Strides,
    },
    pipeline::MAX_COLOR_ATTACHMENTS,
    pixel_format::PixelFormat,
    resource::TextureDescriptor,
};

fn shared() -> ResourceOptions {
    ResourceOptions {
        cpu_cache_mode: CpuCacheMode::DefaultCache,
        storage_mode: StorageMode::Shared,
        hazard_tracking_mode: HazardTrackingMode::Default,
    }
}

fn descriptor(
    pixel_format: PixelFormat,
    width: u64,
    height: u64,
    usage: TextureUsage,
) -> TextureDescriptor {
    TextureDescriptor {
        texture_type: TextureType::D2,
        pixel_format,
        width,
        height,
        depth: 1,
        mipmap_level_count: 1,
        sample_count: 1,
        array_length: 1,
        options: shared(),
        usage,
        swizzle: SwizzleChannels::IDENTITY,
    }
}

fn pass(texture: TextureId, load_action: LoadAction) -> RenderPassRequest {
    let mut colors = [None; MAX_COLOR_ATTACHMENTS];
    colors[0] = Some(AttachmentRequest {
        texture,
        level: 0,
        slice: 0,
        depth_plane: 0,
        resolve: None,
        load_action,
        store_action: StoreAction::Store,
        store_action_options: StoreActionOptions::EMPTY,
        clear: [0.0; 4],
        surface: None,
    });
    RenderPassRequest {
        color_attachments: colors,
        depth_attachment: None,
        depth_resolve_filter: MultisampleDepthResolveFilter::Sample0,
        stencil_attachment: None,
        stencil_resolve_filter: MultisampleStencilResolveFilter::Sample0,
        visibility_result_buffer: None,
        visibility_result_type: VisibilityResultType::Reset,
        render_target_array_length: 0,
        render_target_width: 0,
        render_target_height: 0,
        default_raster_sample_count: 0,
        tile_memory: Default::default(),
        sample_positions: Vec::new(),
        rasterization_rate_map: false,
        counter_samples: Vec::new(),
    }
}

/// The copies between rows and aligned copies in the last submission's
/// first segment, in order: `true` into the copy, `false` back into the
/// rows, with each region's offset, row length and extent.
fn row_copies(device: &mut TestDevice) -> Vec<(bool, u64, u32, [u32; 3])> {
    device
        .backend()
        .submissions
        .last()
        .unwrap()
        .iter()
        .flat_map(|segment| segment.commands.iter())
        .filter_map(|command| match command {
            Command::CopyBufferToImage { region, .. } => Some((
                true,
                region.buffer_offset,
                region.buffer_row_length,
                region.extent,
            )),
            Command::CopyImageToBuffer { region, .. } => Some((
                false,
                region.buffer_offset,
                region.buffer_row_length,
                region.extent,
            )),
            _ => None,
        })
        .collect()
}

fn commit_empty(device: &mut TestDevice) {
    let commands = device.new_command_buffer();
    device.commit(commands).unwrap();
}

fn commit_pass(device: &mut TestDevice, texture: TextureId, load: LoadAction) {
    let mut commands = device.new_command_buffer();
    device
        .begin_render(&mut commands, &pass(texture, load))
        .unwrap();
    commands.end_encoding(&mut device.resolver()).unwrap();
    device.commit(commands).unwrap();
}

fn buffer(device: &mut TestDevice, length: u64) -> BufferId {
    device.new_buffer(length, shared()).unwrap()
}

/// WindowServer's '&b38' screen surfaces of a 3440 or 1680 wide screen,
/// over unified memory: the alpha plane's row is the width in bytes, which
/// an integrated GPU's 256-byte linear rows cannot take. The texture is
/// made, its alpha plane an aligned copy every commit refreshes from the
/// rows and a pass's encode of the planes refreshes again; the color
/// plane's rows fit and stay in place.
#[test]
fn screen_surfaces_of_odd_widths_sample_their_alpha_through_an_aligned_copy() {
    for (width, height, alpha_offset) in
        [(3440u32, 1440u32, 20_076_544u64), (1680, 1050, 7_227_392)]
    {
        let mut device = device();
        device.capabilities.has_unified_memory = true;
        let length = alpha_offset + u64::from(width) * u64::from(height);
        let buffer = buffer(&mut device, length);
        let planes = [
            PlaneLayout {
                offset: 0,
                bytes_per_row: u64::from(width) * 4,
            },
            PlaneLayout {
                offset: alpha_offset,
                bytes_per_row: u64::from(width),
            },
        ];
        let surface = descriptor(
            PixelFormat::Rgb10A8_2pXr10,
            width.into(),
            height.into(),
            TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
        );
        let texture = device
            .new_extended_plane_texture(buffer, &surface, planes)
            .unwrap();
        assert!(device.has_aligned_copy(texture));
        let alpha = (true, alpha_offset, width, [width, height, 1]);
        commit_empty(&mut device);
        assert_eq!(row_copies(&mut device), [alpha], "{width}");
        // The encode writes the rows: they are copied in again after it,
        // and never back, as the planes are what the pass writes.
        commit_pass(&mut device, texture, LoadAction::Clear);
        assert_eq!(row_copies(&mut device), [alpha, alpha], "{width}");
        device.destroy_texture(texture).unwrap();
        commit_empty(&mut device);
        assert_eq!(row_copies(&mut device), [], "{width}");
    }
}

/// A buffer texture whose rows the GPU addresses in place is its linear
/// image, with no copies; one of 1004 BGRA8 pixels a row (4016 bytes) is an
/// aligned copy, refreshed by every commit and copied back by those that
/// write it.
#[test]
fn buffer_textures_of_unaligned_rows_copy_back_what_passes_write() {
    let mut device = device();
    let buffer = buffer(&mut device, 1 << 20);
    let usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    let fits = descriptor(PixelFormat::Bgra8Unorm, 1000, 10, usage);
    let fitting = device
        .new_texture_from_buffer(buffer, &fits, 0, 4096)
        .unwrap();
    assert!(!device.has_aligned_copy(fitting));
    let odd = descriptor(PixelFormat::Bgra8Unorm, 1004, 10, usage);
    let texture = device
        .new_texture_from_buffer(buffer, &odd, 65_536, 4016)
        .unwrap();
    assert!(device.has_aligned_copy(texture));
    let copy = (65_536, 1004, [1004, 10, 1]);
    commit_empty(&mut device);
    assert_eq!(row_copies(&mut device), [(true, copy.0, copy.1, copy.2)]);
    commit_pass(&mut device, texture, LoadAction::Load);
    assert_eq!(
        row_copies(&mut device),
        [
            (true, copy.0, copy.1, copy.2),
            (false, copy.0, copy.1, copy.2)
        ]
    );
    // A pass on the other texture writes only that one.
    commit_pass(&mut device, fitting, LoadAction::Load);
    assert_eq!(row_copies(&mut device), [(true, copy.0, copy.1, copy.2)]);
    // A pixel is a row's unit; an offset of whole pixels need not fit the
    // GPU's 32 bytes either.
    assert!(
        device
            .new_texture_from_buffer(buffer, &odd, 0, 4018)
            .is_err()
    );
    let offset = device
        .new_texture_from_buffer(buffer, &fits, 4100, 4096)
        .unwrap();
    assert!(device.has_aligned_copy(offset));
}

/// The CPU's copies of an aligned copy's texture go to and from its rows,
/// at their own pitch.
#[test]
fn cpu_copies_of_an_aligned_copy_convert_row_by_row_in_the_rows() {
    let mut device = device();
    let buffer = buffer(&mut device, 1 << 16);
    let odd = descriptor(PixelFormat::R8Unorm, 40, 3, TextureUsage::SHADER_READ);
    let texture = device
        .new_texture_from_buffer(buffer, &odd, 16, 48)
        .unwrap();
    assert!(device.has_aligned_copy(texture));
    let region = Region {
        slice: 0,
        level: 0,
        origin: [2, 1, 0],
        size: [3, 2, 1],
    };
    let tight = Strides {
        bytes_per_row: 3,
        bytes_per_image: 0,
    };
    let written = [1, 2, 3, 4, 5, 6];
    device
        .replace_region(texture, &region, &written, tight)
        .unwrap();
    let memory = device.resources.buffers.get(buffer.0).unwrap().memory;
    let rows: Vec<_> = device
        .backend()
        .writes
        .iter()
        .filter(|(written, ..)| *written == memory)
        .map(|(_, offset, bytes)| (*offset, bytes.clone()))
        .collect();
    // Row 1 starts 16 + 48 bytes in, row 2 48 after it; pixel 2 of each.
    assert_eq!(
        rows,
        [(66, alloc::vec![1, 2, 3]), (114, alloc::vec![4, 5, 6])]
    );
    let mut read = [0; 6];
    device
        .get_bytes(texture, &region, &mut read, tight)
        .unwrap();
    assert_eq!(read, written);
}

/// A two-plane surface whose planes' rows do not fit has an aligned copy
/// of each, refreshed by every commit.
#[test]
fn two_plane_surfaces_copy_each_unaligned_plane() {
    let mut device = device();
    let buffer = buffer(&mut device, 1 << 20);
    let surface = descriptor(
        PixelFormat::Rgb8_420_2p,
        1010,
        10,
        TextureUsage::SHADER_READ,
    );
    let texture = device
        .new_two_plane_texture(
            buffer,
            &surface,
            [
                PlaneLayout {
                    offset: 0,
                    bytes_per_row: 1040,
                },
                PlaneLayout {
                    offset: 1 << 16,
                    bytes_per_row: 1010,
                },
            ],
            0,
        )
        .unwrap();
    commit_empty(&mut device);
    assert_eq!(
        row_copies(&mut device),
        [
            (true, 0, 1040, [1010, 10, 1]),
            (true, 1 << 16, 505, [505, 5, 1])
        ]
    );
    device.destroy_texture(texture).unwrap();
}

/// Every connection's command buffer refreshes an aligned copy from its
/// rows, so the rows' buffer outlives the command buffers of other
/// connections that read it, though the buffer is its own connection's:
/// before, connection 1's release of the texture and then the buffer freed
/// the rows while connection 2's command buffer still copied from them.
#[test]
fn rows_another_connections_work_refreshes_from_outlive_it() {
    use crate::metal::{device::pins::Scope, vulkan::backend::Object};
    let mut device = device();
    let made = device.submitted_serial();
    let buffer = buffer(&mut device, 1 << 20);
    let (memory, ..) = device.buffer_memory(buffer).unwrap();
    let usage = TextureUsage::SHADER_READ;
    let odd = descriptor(PixelFormat::Bgra8Unorm, 1004, 10, usage);
    let texture = device
        .new_texture_from_buffer(buffer, &odd, 65_536, 4016)
        .unwrap();
    assert!(device.has_aligned_copy(texture));
    let commands = device.new_command_buffer();
    let other = device.commit_for(2, commands).unwrap();
    assert_eq!(row_copies(&mut device).len(), 1, "it reads the rows");
    let own = Scope::connection(1, made);
    device
        .release_for(own.clone(), |device| device.destroy_texture(texture))
        .unwrap();
    device
        .release_for(own, |device| device.destroy_buffer(buffer))
        .unwrap();
    assert!(
        !device.backend().destroyed.contains(&Object::Memory(memory)),
        "the rows stay while connection 2's command buffer runs"
    );
    device.backend().finish();
    assert_eq!(device.poll(), Ok(other));
    assert!(device.backend().destroyed.contains(&Object::Memory(memory)));
}

/// A video wallpaper's 10-bit 4:2:0 surface, whose rows the GPU could
/// address in place, is sampled from device memory: both planes' copies are
/// refreshed only by a command buffer that uses the texture or may reach it
/// unnamed (a seed report), only when the surface's seed changed since the
/// last copy, and at the end of a command buffer that wrote the surface
/// through another object. Before, it was sampled
/// from guest RAM: 10.6 ms for one 1280x896 quad over a 3840x2160 surface.
#[test]
fn a_large_sampled_surface_is_copied_only_where_used_and_changed() {
    let mut device = device();
    let buffer = buffer(&mut device, 8 << 20);
    let surface = descriptor(
        PixelFormat::Ycbcr10_420_2p,
        1920,
        1080,
        TextureUsage::SHADER_READ,
    );
    let planes = [
        PlaneLayout {
            offset: 0,
            bytes_per_row: 3840,
        },
        PlaneLayout {
            offset: 4 << 20,
            bytes_per_row: 3840,
        },
    ];
    let texture = device
        .new_two_plane_texture(buffer, &surface, planes, 0)
        .unwrap();
    assert!(device.has_aligned_copy(texture));
    let both = [
        (true, 0, 1920, [1920, 1080, 1]),
        (true, 4 << 20, 960, [960, 540, 1]),
    ];
    let note = |texture, seed| SurfaceNote {
        texture,
        surface: 42,
        seed,
    };
    let commit = |device: &mut TestDevice, notes: &[SurfaceNote], written: &[TextureId]| {
        let mut commands = device.new_command_buffer();
        commands.surfaces.extend_from_slice(notes);
        commands.written.extend_from_slice(written);
        device.commit(commands).unwrap();
        row_copies(device)
    };
    // Not used: nothing is copied.
    assert_eq!(commit(&mut device, &[], &[]), []);
    // Used at seed 7: copied once, then not again while the seed holds.
    assert_eq!(commit(&mut device, &[note(Some(texture), 7)], &[]), both);
    assert_eq!(commit(&mut device, &[note(Some(texture), 7)], &[]), []);
    // The CPU wrote it: a new seed.
    assert_eq!(commit(&mut device, &[note(Some(texture), 8)], &[]), both);
    // The CPU wrote it again, and a command buffer may sample it through an
    // argument buffer: the guest reports the new seed, which copies.
    let seeds = |device: &mut TestDevice, notes: &[SurfaceNote]| {
        let mut commands = device.new_command_buffer();
        commands.seeds.extend_from_slice(notes);
        device.commit(commands).unwrap();
        row_copies(device)
    };
    assert_eq!(seeds(&mut device, &[note(Some(texture), 9)]), both);
    assert_eq!(commit(&mut device, &[note(Some(texture), 9)], &[]), []);
    assert_eq!(commit(&mut device, &[note(Some(texture), 8)], &[]), both);
    // Another texture over the same surface (another connection's, as the
    // surface's ID says) written by a command buffer: its end copies, so a
    // later sample through an argument buffer reads what it wrote.
    let other = device
        .new_texture_from_buffer(
            buffer,
            &descriptor(
                PixelFormat::R16Unorm,
                1920,
                1080,
                TextureUsage::RENDER_TARGET,
            ),
            0,
            3840,
        )
        .unwrap();
    assert_eq!(commit(&mut device, &[note(Some(other), 8)], &[other]), both);
    assert_eq!(commit(&mut device, &[note(Some(texture), 8)], &[]), []);
    // Read, not written, it changes nothing.
    assert_eq!(commit(&mut device, &[note(Some(other), 8)], &[]), []);
    assert_eq!(commit(&mut device, &[note(Some(texture), 8)], &[]), []);
    // A buffer over the surface may be written by any shader.
    assert_eq!(commit(&mut device, &[note(None, 8)], &[]), both);
    assert_eq!(commit(&mut device, &[note(Some(texture), 8)], &[]), []);
    device.destroy_texture(other).unwrap();
    device.destroy_texture(texture).unwrap();
}

/// A surface below a 1920x1080 frame keeps sampling its rows in
/// place: no copy at all.
#[test]
fn small_surfaces_stay_in_their_rows() {
    let mut device = device();
    let buffer = buffer(&mut device, 8 << 20);
    let planes = [
        PlaneLayout {
            offset: 0,
            bytes_per_row: 1280,
        },
        PlaneLayout {
            offset: 4 << 20,
            bytes_per_row: 1280,
        },
    ];
    let small = descriptor(
        PixelFormat::Ycbcr10_420_2p,
        640,
        360,
        TextureUsage::SHADER_READ,
    );
    let texture = device
        .new_two_plane_texture(buffer, &small, planes, 0)
        .unwrap();
    assert!(!device.has_aligned_copy(texture));
    device.destroy_texture(texture).unwrap();
}

/// A large sampled two-plane surface whose rows the GPU cannot address in
/// place (a 4K video wallpaper's planes on a GPU of 256-byte linear rows)
/// is copied only where used and changed, as one whose rows fit. Before,
/// its copies were refreshed by every command buffer: 10.4 million texels
/// each on the AMD iGPU, about half of its time.
#[test]
fn a_sampled_surface_of_unaligned_rows_is_copied_only_where_used_and_changed() {
    let mut device = device();
    let buffer = buffer(&mut device, 8 << 20);
    let surface = descriptor(
        PixelFormat::Ycbcr10_420_2p,
        1920,
        1080,
        TextureUsage::SHADER_READ,
    );
    // 1924 and 962 texels a row: no multiple of 32 bytes.
    let planes = [
        PlaneLayout {
            offset: 0,
            bytes_per_row: 3848,
        },
        PlaneLayout {
            offset: 4 << 20,
            bytes_per_row: 3848,
        },
    ];
    let texture = device
        .new_two_plane_texture(buffer, &surface, planes, 0)
        .unwrap();
    assert!(device.has_aligned_copy(texture));
    let both = [
        (true, 0, 1924, [1920, 1080, 1]),
        (true, 4 << 20, 962, [960, 540, 1]),
    ];
    let note = |seed| SurfaceNote {
        texture: Some(texture),
        surface: 7,
        seed,
    };
    let commit = |device: &mut TestDevice, notes: &[SurfaceNote]| {
        let mut commands = device.new_command_buffer();
        commands.surfaces.extend_from_slice(notes);
        device.commit(commands).unwrap();
        row_copies(device)
    };
    assert_eq!(commit(&mut device, &[]), []);
    assert_eq!(commit(&mut device, &[note(3)]), both);
    assert_eq!(commit(&mut device, &[note(3)]), []);
    assert_eq!(commit(&mut device, &[]), []);
    assert_eq!(commit(&mut device, &[note(4)]), both);
    device.destroy_texture(texture).unwrap();
}

/// A screen surface's alpha plane of an odd width over unified memory, an
/// aligned copy: until a command buffer reports the texture, every one
/// refreshes it; once reported, only one that uses it after the surface
/// changed. Its own pass encodes the rows and refreshes the copy after, so
/// the next use copies nothing; the CPU (a new seed) makes the next use
/// copy again, and another texture's write to the surface or a buffer over
/// it the end of the command buffer that writes it.
#[test]
fn a_reported_screen_surface_is_refreshed_only_where_used_and_changed() {
    let (width, height, alpha_offset) = (1680u32, 1050u32, 7_227_392u64);
    let mut device = device();
    device.capabilities.has_unified_memory = true;
    let length = alpha_offset + u64::from(width) * u64::from(height);
    let buffer = buffer(&mut device, length);
    let planes = [
        PlaneLayout {
            offset: 0,
            bytes_per_row: u64::from(width) * 4,
        },
        PlaneLayout {
            offset: alpha_offset,
            bytes_per_row: u64::from(width),
        },
    ];
    let surface = descriptor(
        PixelFormat::Rgb10A8_2pXr10,
        width.into(),
        height.into(),
        TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
    );
    let screen = device
        .new_extended_plane_texture(buffer, &surface, planes)
        .unwrap();
    let alpha = (true, alpha_offset, width, [width, height, 1]);
    let note = |texture, seed| SurfaceNote {
        texture,
        surface: 9,
        seed,
    };
    let commit = |device: &mut TestDevice, notes: &[SurfaceNote], written: &[TextureId]| {
        let mut commands = device.new_command_buffer();
        commands.surfaces.extend_from_slice(notes);
        commands.written.extend_from_slice(written);
        device.commit(commands).unwrap();
        row_copies(device)
    };
    // Unreported: every command buffer refreshes it, as before.
    assert_eq!(commit(&mut device, &[], &[]), [alpha]);
    assert_eq!(commit(&mut device, &[], &[]), [alpha]);
    // Reported: once, then not again while the seed holds, nor by command
    // buffers that do not use it.
    assert_eq!(commit(&mut device, &[note(Some(screen), 1)], &[]), [alpha]);
    assert_eq!(commit(&mut device, &[note(Some(screen), 1)], &[]), []);
    assert_eq!(commit(&mut device, &[], &[]), []);
    // Its own pass: the encode writes the rows and refreshes the copy.
    let mut commands = device.new_command_buffer();
    device
        .begin_render(&mut commands, &pass(screen, LoadAction::Clear))
        .unwrap();
    commands.end_encoding(&mut device.resolver()).unwrap();
    commands.surfaces.push(note(Some(screen), 1));
    device.commit(commands).unwrap();
    assert_eq!(row_copies(&mut device), [alpha]);
    assert_eq!(commit(&mut device, &[note(Some(screen), 1)], &[]), []);
    // The CPU wrote the surface.
    assert_eq!(commit(&mut device, &[note(Some(screen), 2)], &[]), [alpha]);
    // Another texture over the surface written by the GPU.
    let other = device
        .new_texture_from_buffer(
            buffer,
            &descriptor(PixelFormat::R8Unorm, 1024, 16, TextureUsage::RENDER_TARGET),
            alpha_offset,
            u64::from(width),
        )
        .unwrap();
    assert_eq!(
        commit(&mut device, &[note(Some(other), 2)], &[other]).len(),
        3,
        "the other texture's own copy, then both copies after the write"
    );
    assert_eq!(commit(&mut device, &[note(Some(screen), 2)], &[]), []);
    // A buffer over the surface, which any shader may write.
    assert_eq!(commit(&mut device, &[note(None, 2)], &[]).len(), 2);
    assert_eq!(commit(&mut device, &[note(Some(screen), 2)], &[]), []);
    device.destroy_texture(other).unwrap();
    device.destroy_texture(screen).unwrap();
}

/// A reported buffer texture of unaligned rows that a pass writes is copied
/// back into its rows at the end; its copy then holds them, so the next use
/// at the same seed copies nothing in. Written beside another texture over
/// the same surface, which also copies back, both are copied in again once
/// both copied back.
#[test]
fn a_reported_texture_its_command_buffer_wrote_keeps_its_copy() {
    let mut device = device();
    let buffer = buffer(&mut device, 1 << 20);
    let usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    let odd = descriptor(PixelFormat::Bgra8Unorm, 1004, 10, usage);
    let texture = device
        .new_texture_from_buffer(buffer, &odd, 65_536, 4016)
        .unwrap();
    let copy = (65_536, 1004, [1004, 10, 1]);
    let (into, back) = (
        (true, copy.0, copy.1, copy.2),
        (false, copy.0, copy.1, copy.2),
    );
    let note = |texture| SurfaceNote {
        texture: Some(texture),
        surface: 5,
        seed: 1,
    };
    let write = |device: &mut TestDevice, textures: &[TextureId]| {
        let mut commands = device.new_command_buffer();
        for &texture in textures {
            device
                .begin_render(&mut commands, &pass(texture, LoadAction::Load))
                .unwrap();
            commands.end_encoding(&mut device.resolver()).unwrap();
            commands.surfaces.push(note(texture));
        }
        device.commit(commands).unwrap();
        row_copies(device)
    };
    let read = |device: &mut TestDevice| {
        let mut commands = device.new_command_buffer();
        commands.surfaces.push(note(texture));
        device.commit(commands).unwrap();
        row_copies(device)
    };
    assert_eq!(write(&mut device, &[texture]), [into, back]);
    assert_eq!(read(&mut device), []);
    assert_eq!(write(&mut device, &[texture]), [back]);
    assert_eq!(read(&mut device), []);
    let other = device
        .new_texture_from_buffer(buffer, &odd, 65_536, 4016)
        .unwrap();
    assert_eq!(
        write(&mut device, &[texture, other]),
        [into, back, back, into, into]
    );
    assert_eq!(read(&mut device), []);
    device.destroy_texture(other).unwrap();
    device.destroy_texture(texture).unwrap();
}

/// A rendered screen surface on a GPU of its own memory keeps a device
/// copy of both planes, whatever their rows: no command buffer refreshes
/// it before one uses it. One that writes it unreported fills it first,
/// while nothing filled it yet; a reported use fills it when the seed
/// changed; its own pass's encode writes it with the rows, so nothing is
/// copied after, nor back. The CPU makes the next use copy again, a buffer
/// over the surface the end of the command buffer that may write it.
#[test]
fn a_rendered_screen_surface_keeps_a_device_copy_of_both_planes() {
    let (width, height, alpha_offset) = (1680u32, 1050u32, 7_227_392u64);
    let mut device = device();
    let length = alpha_offset + u64::from(width) * u64::from(height);
    let buffer = buffer(&mut device, length);
    let planes = [
        PlaneLayout {
            offset: 0,
            bytes_per_row: u64::from(width) * 4,
        },
        PlaneLayout {
            offset: alpha_offset,
            bytes_per_row: u64::from(width),
        },
    ];
    let surface = descriptor(
        PixelFormat::Rgb10A8_2pXr10,
        width.into(),
        height.into(),
        TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
    );
    let screen = device
        .new_extended_plane_texture(buffer, &surface, planes)
        .unwrap();
    let both = [
        (true, 0, width, [width, height, 1]),
        (true, alpha_offset, width, [width, height, 1]),
    ];
    let note = |texture, seed| SurfaceNote {
        texture,
        surface: 9,
        seed,
    };
    let commit = |device: &mut TestDevice, notes: &[SurfaceNote], written: &[TextureId]| {
        let mut commands = device.new_command_buffer();
        commands.surfaces.extend_from_slice(notes);
        commands.written.extend_from_slice(written);
        device.commit(commands).unwrap();
        row_copies(device)
    };
    assert_eq!(commit(&mut device, &[], &[]), []);
    // Written unreported, never filled: filled first, every time.
    assert_eq!(commit(&mut device, &[], &[screen]), both);
    assert_eq!(commit(&mut device, &[], &[screen]), both);
    // Reported: once, then not while the seed holds.
    assert_eq!(commit(&mut device, &[note(Some(screen), 1)], &[]), both);
    assert_eq!(commit(&mut device, &[note(Some(screen), 1)], &[]), []);
    // Its own pass, of the surface reported: nothing copied before, after
    // or back.
    let mut request = pass(screen, LoadAction::Load);
    if let Some(color) = &mut request.color_attachments[0] {
        color.surface = Some((9, 1));
    }
    let mut commands = device.new_command_buffer();
    device.begin_render(&mut commands, &request).unwrap();
    commands.end_encoding(&mut device.resolver()).unwrap();
    commands.surfaces.push(note(Some(screen), 1));
    device.commit(commands).unwrap();
    assert_eq!(row_copies(&mut device), []);
    assert_eq!(commit(&mut device, &[note(Some(screen), 1)], &[]), []);
    // The CPU wrote the surface.
    assert_eq!(commit(&mut device, &[note(Some(screen), 2)], &[]), both);
    // A buffer over the surface, which any shader may write.
    assert_eq!(commit(&mut device, &[note(None, 2)], &[]), both);
    assert_eq!(commit(&mut device, &[note(Some(screen), 2)], &[]), []);
    device.destroy_texture(screen).unwrap();
}

/// The service's writes for the CPU (`replaceRegion`) change no seed: the
/// next command buffer refreshes the copies over the rows, whether or not
/// it names the texture, so one sampling it through an argument buffer
/// reads them.
#[test]
fn rows_the_service_writes_for_the_cpu_refresh_the_next_command_buffer() {
    let mut device = device();
    let buffer = buffer(&mut device, 1 << 16);
    let odd = descriptor(PixelFormat::R8Unorm, 40, 3, TextureUsage::SHADER_READ);
    let texture = device
        .new_texture_from_buffer(buffer, &odd, 16, 48)
        .unwrap();
    let copy = [(true, 16, 48, [40, 3, 1])];
    // Reported once: refreshed only where used and changed from then on.
    let mut commands = device.new_command_buffer();
    commands.surfaces.push(SurfaceNote {
        texture: Some(texture),
        surface: 3,
        seed: 1,
    });
    device.commit(commands).unwrap();
    assert_eq!(row_copies(&mut device), copy);
    commit_empty(&mut device);
    assert_eq!(row_copies(&mut device), []);
    let region = Region {
        slice: 0,
        level: 0,
        origin: [0; 3],
        size: [4, 1, 1],
    };
    let tight = Strides {
        bytes_per_row: 4,
        bytes_per_image: 0,
    };
    device
        .replace_region(texture, &region, &[1, 2, 3, 4], tight)
        .unwrap();
    commit_empty(&mut device);
    assert_eq!(row_copies(&mut device), copy);
    commit_empty(&mut device);
    assert_eq!(row_copies(&mut device), []);
    device.destroy_texture(texture).unwrap();
}
