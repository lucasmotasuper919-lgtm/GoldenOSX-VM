use super::{DisplayFrame, DisplayLayer, DisplayPlane};
use crate::metal::{
    binding::{ComposeArguments, ComposeFormat},
    device::fixture::{MockBackend, TestDevice, device_on},
    error::Error,
    vulkan::{backend::Object, command::Command, object::Memory},
};
use alloc::{vec, vec::Vec};
use stela_contracts::gate::MemoryRange;

fn page(address: u64) -> MemoryRange {
    MemoryRange {
        address,
        bytes: 0x1000,
    }
}

/// A 4x2 BGRA layer whose pages start 8 bytes before its first pixel
/// and hold its two rows, 16 bytes apart.
fn layer() -> DisplayLayer {
    DisplayLayer {
        color: DisplayPlane {
            ranges: vec![page(0x8000_0000), page(0x9000_0000)],
            origin: 8,
        },
        alpha: None,
        stride: 16,
        alpha_stride: 0,
        format: ComposeFormat::Bgra8,
        premultiplied: true,
        source: [0, 0, 4, 2],
        destination: [2, 1, 8, 4],
        surface: 0,
    }
}

fn frame(layers: Vec<DisplayLayer>) -> DisplayFrame {
    DisplayFrame {
        pitch: 64,
        width: 16,
        rows: 1..5,
        layers,
    }
}

/// The compositions submitted, with the arguments each uploaded.
fn compositions(device: &TestDevice) -> Vec<(Vec<u8>, u32)> {
    device
        .backend
        .submissions
        .iter()
        .flatten()
        .flat_map(|segment| &segment.commands)
        .filter_map(|command| match *command {
            Command::Compose { arguments, count } => Some((uploaded(device, arguments), count)),
            _ => None,
        })
        .collect()
}

/// The bytes a fixture buffer at `address` was written with there.
fn uploaded(device: &TestDevice, address: u64) -> Vec<u8> {
    let (memory, offset) = (Memory(address >> 32), address & 0xffff_ffff);
    let (_, at, bytes) = device
        .backend
        .writes
        .iter()
        .rev()
        .find(|(written, at, bytes)| {
            *written == memory && (*at..*at + bytes.len() as u64).contains(&offset)
        })
        .expect("an upload holds the arguments");
    bytes[(offset - at) as usize..][..ComposeArguments::BYTES as usize].to_vec()
}

fn word(bytes: &[u8], at: u32) -> u32 {
    u32::from_le_bytes(bytes[at as usize..][..4].try_into().unwrap())
}

fn double(bytes: &[u8], at: u32) -> u64 {
    u64::from_le_bytes(bytes[at as usize..][..8].try_into().unwrap())
}

#[test]
fn a_frame_is_composed_into_the_shared_buffer_from_the_layers_guest_pages() {
    let mut device = device_on(MockBackend::new());
    let scanout = device.import_scanout(7, 64 * 8).unwrap();
    let (&memory, &(fd, bytes)) = device.backend.dma_bufs.iter().next().unwrap();
    assert_eq!((fd, bytes), (7, 512), "the dma-buf, whole");
    let serial = device
        .compose_display(&scanout, &frame(vec![layer()]))
        .unwrap();
    let ranges: Vec<_> = device.backend.own_imports.values().cloned().collect();
    assert_eq!(
        ranges,
        [layer().color.ranges],
        "the layer's pages, in order"
    );
    let composed = compositions(&device);
    let [(arguments, count)] = &composed[..] else {
        panic!("one composition: {composed:?}");
    };
    assert_eq!(*count, 16 * 4, "every pixel of the four damaged rows");
    assert_eq!(
        double(arguments, 0),
        scanout.address,
        "into the scanout buffer"
    );
    assert_eq!(
        [8, 12, 16, 20, 24].map(|at| word(arguments, at)),
        [64, 16, 1, 64, 1],
        "pitch, width, first row, count, layers"
    );
    let layer = &arguments[ComposeArguments::LAYERS_OFFSET as usize..];
    // The color plane's buffer, from 8 bytes into its pages.
    assert_eq!(double(layer, 0) & 0xffff_ffff, 8);
    assert_eq!(word(layer, 16), 16, "stride");
    assert_eq!(word(layer, 28), 1, "premultiplied");
    assert_eq!(
        [32, 36, 40, 44, 48, 52, 56, 60].map(|at| word(layer, at)),
        [0, 0, 4, 2, 2, 1, 8, 4]
    );
    assert!(!device.serial_complete(serial));
    device.backend.finish();
    device.poll().unwrap();
    assert!(device.serial_complete(serial));
    assert!(
        device.retired().is_none(),
        "the device's own imports retire no guest token"
    );
    assert!(
        device
            .backend
            .destroyed
            .iter()
            .any(|object| matches!(object, Object::Memory(memory) if device.backend.own_imports.contains_key(&memory.0))),
        "the layer's import goes once its composition completed"
    );
    device.release_scanout(scanout);
    assert!(
        device
            .backend
            .destroyed
            .contains(&Object::Memory(Memory(memory)))
    );
}

#[test]
fn a_two_plane_layer_reads_its_alpha_plane() {
    let mut device = device_on(MockBackend::new());
    let scanout = device.import_scanout(7, 64 * 8).unwrap();
    let mut two_plane = layer();
    two_plane.format = ComposeFormat::Xr10A8;
    assert!(matches!(
        device.compose_display(&scanout, &frame(vec![two_plane.clone()])),
        Err(Error::Invalid(_))
    ));
    two_plane.alpha = Some(DisplayPlane {
        ranges: vec![page(0xa000_0000)],
        origin: 0,
    });
    two_plane.alpha_stride = 4;
    device
        .compose_display(&scanout, &frame(vec![two_plane]))
        .unwrap();
    assert_eq!(device.backend.own_imports.len(), 2, "both planes");
    let composed = compositions(&device);
    let layer = &composed[0].0[ComposeArguments::LAYERS_OFFSET as usize..];
    assert_ne!(double(layer, 8), 0, "the alpha plane's address");
    assert_eq!(word(layer, 20), 4);
    assert_eq!(word(layer, 24), ComposeFormat::Xr10A8.raw());
}

#[test]
fn what_lies_outside_its_memory_is_refused_before_anything_runs() {
    let mut device = device_on(MockBackend::new());
    let scanout = device.import_scanout(7, 64 * 4).unwrap();
    // Row 4 ends past the 256 bytes.
    assert!(
        device
            .compose_display(&scanout, &frame(vec![layer()]))
            .is_err()
    );
    let scanout = device.import_scanout(8, 64 * 8).unwrap();
    let mut early = layer();
    early.color.origin = -1;
    let mut late = layer();
    late.source = [0, 0, 4, 0x200];
    for refused in [early, late] {
        assert!(matches!(
            device.compose_display(&scanout, &frame(vec![refused])),
            Err(Error::Invalid(_))
        ));
    }
    assert!(
        device
            .compose_display(&scanout, &frame(vec![layer(); 5]))
            .is_err(),
        "four layers at most"
    );
    assert!(compositions(&device).is_empty());
    assert!(device.backend.own_imports.is_empty());
}
