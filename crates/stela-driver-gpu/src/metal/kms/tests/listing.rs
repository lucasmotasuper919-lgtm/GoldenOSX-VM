//! The boot log's line for each connector read: every mode it lists and
//! why each was passed over.
use super::outputs::{at, two_monitors};
use super::*;
use alloc::format;
use alloc::string::String;

fn lines(display: &mut Display<&mut Card>) -> Vec<String> {
    display
        .listings()
        .iter()
        .map(|listing| format!("{listing}"))
        .collect()
}

/// A mode whose refresh is its `clock` in kilohertz over a million pixels:
/// `clock` millihertz.
fn timed(width: u16, height: u16, clock: u32) -> ModeInfo {
    ModeInfo {
        clock,
        timing_h: [0, 0, 1000, 0],
        timing_v: [0, 0, 1000, 0],
        ..mode(width, height, 0)
    }
}

#[test]
fn the_line_names_every_mode_the_output_lists_and_why_it_was_passed_over() {
    let interlaced = ModeInfo {
        flags: INTERLACED,
        ..at(8, 4, 0, 60)
    };
    let mut card = Card::new(vec![
        at(8, 4, PREFERRED, 60),
        timed(8, 4, 239_970),
        timed(8, 4, 143_912),
        at(8, 4, 0, 120),
        at(8, 4, 0, 120),
        at(8, 4, 0, 200),
        interlaced,
        at(10, 4, 0, 60),
        at(6, 4, 0, 240),
        at(6, 4, 0, 60),
    ]);
    card.ports = [(10, 1), (11, 1)];
    card.lit = 51;
    // macOS cannot be told of a 10 wide mode, nor of 200 Hz.
    let fits: Fits<'_> = &|width, _, refresh| width != 10 && refresh != 200_000;
    let mut display = Display::take_fitting(&mut card, Wanted::Native((6, 4)), fits).unwrap();
    assert_eq!(display.refresh_millihertz(), 239_970);
    assert_eq!(
        lines(&mut display),
        [
            "HDMI-A-1 connected, lit by the firmware, gives the picture 8x4; modes (* preferred): \
          runs: 8x4 239.97; offered to macOS as a rate: 8x4 60* 143.912 120; a rate listed \
          before: 8x4 120; a rate macOS cannot be told of: 8x4 200; interlaced: 8x4i 60; other \
          size, which macOS cannot be told of: 10x4 60; other size, ranked after the picture's: \
          6x4 240 60"
        ]
    );
    // What the line says is offered is what the display offers.
    assert_eq!(display.refresh_rates(), [239_970, 143_912, 120_000, 60_000]);
    // Each connector is named once.
    assert!(lines(&mut display).is_empty());

    // A size the guest asked for: the others are not that size.
    let mut asked = Card::new(vec![at(8, 4, PREFERRED, 60), at(6, 4, 0, 60)]);
    let mut display = Display::take(&mut asked, (6, 4)).unwrap();
    assert_eq!(
        lines(&mut display),
        [
            "Unknown-0 connected, gives the picture 6x4; modes (* preferred): runs: 6x4 60; not \
          the size asked for: 8x4 60*"
        ]
    );
}

#[test]
fn each_connected_output_says_what_it_does_with_the_picture() {
    // Two monitors that share a size: the one both list wins over the
    // first's own, which fewer outputs list.
    let mut card = two_monitors();
    card.ports = [(10, 1), (11, 1)];
    let mut display =
        Display::take_fitting(&mut card, Wanted::Native((4, 2)), &|_, _, _| true).unwrap();
    assert_eq!(
        lines(&mut display),
        [
            "DP-1 connected, gives the picture 6x4; modes (* preferred): runs: 6x4 75; other \
             size, which fewer outputs list: 8x4 60*; other size, ranked after the picture's: \
             4x2 60",
            "HDMI-A-1 connected, shows the picture; modes (* preferred): runs: 6x4 60*; not the \
             picture's size: 4x2 60",
        ]
    );
    drop(display);

    // A larger monitor without the size is scaled; without a scaling
    // property it is dark, and so is a smaller one.
    let unshared = |lit: u32, scaler: bool| {
        let mut card = Card::new(vec![at(8, 4, PREFERRED, 60)]);
        card.first = vec![at(4, 2, PREFERRED, 60)];
        card.first_connected = true;
        card.scaler = scaler;
        card.lit = lit;
        card
    };
    let mut card = unshared(50, true);
    let mut display =
        Display::take_fitting(&mut card, Wanted::Native((4, 2)), &|_, _, _| true).unwrap();
    assert_eq!(
        lines(&mut display)[1],
        "Unknown-0 connected, shows the picture scaled; modes (* preferred): runs, the picture \
         scaled into it: 8x4 60*"
    );
    drop(display);
    let mut card = unshared(50, false);
    let mut display =
        Display::take_fitting(&mut card, Wanted::Native((4, 2)), &|_, _, _| true).unwrap();
    assert_eq!(
        lines(&mut display)[1],
        "Unknown-0 connected, dark: no mode of the picture's size and no scaling property; \
         modes (* preferred): not the picture's size: 8x4 60*"
    );
    drop(display);
    let mut card = unshared(51, true);
    let mut display =
        Display::take_fitting(&mut card, Wanted::Native((4, 2)), &|_, _, _| true).unwrap();
    assert_eq!(
        lines(&mut display),
        [
            "Unknown-0 connected, lit by the firmware, gives the picture 8x4; modes (* \
             preferred): runs: 8x4 60*",
            "Unknown-0 connected, dark: its native mode is smaller than the picture, which is \
             only scaled up; modes (* preferred): not the picture's size: 4x2 60*",
        ]
    );
    drop(display);

    // An output the driver refuses to set, and one with no free controller.
    let mut card = two_monitors();
    card.refused_crtc = Some(41);
    let mut display = Display::take(&mut card, (6, 4)).unwrap();
    assert_eq!(
        lines(&mut display)[1],
        "Unknown-0 connected, dark: the driver refused to set it: Request(-22); modes (* \
         preferred): refused by the driver: 6x4 60*; not the picture's size: 4x2 60"
    );
    drop(display);
    let mut card = two_monitors();
    card.first_crtcs = 0b01;
    card.possible_crtcs = 0b01;
    let mut display = Display::take(&mut card, (6, 4)).unwrap();
    assert_eq!(
        lines(&mut display)[1],
        "Unknown-0 connected, dark: no controller or encoder of its own is free; modes (* \
         preferred): not run: 6x4 60*; not the picture's size: 4x2 60"
    );
    drop(display);

    // An output the firmware lit that detects no display is named too; one
    // that was never lit and is not connected is not.
    let mut card = Card::new(vec![at(2, 2, 0, 60)]);
    card.lit = 50;
    let mut display = Display::take(&mut card, (2, 2)).unwrap();
    assert_eq!(
        lines(&mut display),
        [
            "Unknown-0 not connected, lit by the firmware, dark: no display detected",
            "Unknown-0 connected, gives the picture 2x2; modes: runs: 2x2 60",
        ]
    );
    drop(display);
    let mut card = Card::new(vec![at(2, 2, 0, 60)]);
    let mut display = Display::take(&mut card, (2, 2)).unwrap();
    assert_eq!(lines(&mut display).len(), 1);
}

#[test]
fn a_look_names_an_output_when_it_shows_the_picture_or_is_given_up_on() {
    // Connected after the take: named when it joins.
    let mut card = two_monitors();
    card.first_connected = false;
    let mut display = Display::take(&mut card, (6, 4)).unwrap();
    assert_eq!(lines(&mut display).len(), 1);
    display.watch();
    assert!(lines(&mut display).is_empty());
    display.drm.first_connected = true;
    assert_eq!(display.watch().joined, 1);
    assert_eq!(
        lines(&mut display),
        [
            "Unknown-0 connected, shows the picture; modes (* preferred): runs: 6x4 75; not the \
          picture's size: 8x4 60*, 4x2 60"
        ]
    );
    // Unplugged and back with another monitor: named when it is set again.
    display.drm.first_connected = false;
    display.watch();
    display.drm.first = vec![at(6, 4, PREFERRED, 50)];
    display.drm.first_connected = true;
    assert_eq!(display.watch().restored, 1);
    assert_eq!(
        lines(&mut display),
        ["Unknown-0 connected, shows the picture; modes (* preferred): runs: 6x4 50*"]
    );
    drop(display);

    // One that cannot show the size is named once, at its last try.
    let mut card = two_monitors();
    card.first = vec![at(8, 4, PREFERRED, 60)];
    card.first_connected = false;
    let mut display = Display::take(&mut card, (6, 4)).unwrap();
    display.listings();
    display.drm.first_connected = true;
    for _ in 1..RESTORE_TRIES {
        display.watch();
        assert!(lines(&mut display).is_empty());
    }
    assert_eq!(display.watch().unshown, 1);
    assert_eq!(
        lines(&mut display),
        [
            "Unknown-0 connected, dark: no mode of the picture's size and no scaling property; \
          modes (* preferred): not the picture's size: 8x4 60*"
        ]
    );
    display.watch();
    assert!(lines(&mut display).is_empty());
    drop(display);

    // The output that gives the picture, back with a monitor that has its
    // size at another rate.
    let mut card = Card::new(vec![at(2, 2, PREFERRED, 60)]);
    let mut display = Display::take(&mut card, (2, 2)).unwrap();
    display.listings();
    display.drm.connected = false;
    display.watch();
    display.drm.modes = vec![at(4, 2, PREFERRED, 60), at(2, 2, 0, 75), at(2, 2, 0, 50)];
    display.drm.connected = true;
    assert_eq!(display.watch().restored, 1);
    assert_eq!(
        lines(&mut display),
        [
            "Unknown-0 connected, gives the picture 2x2; modes (* preferred): runs: 2x2 75; not \
          the picture's size: 4x2 60*; not the rate run: 2x2 50"
        ]
    );
}

/// An EDID of a base block and the extension blocks `extensions`, each
/// padded to 128 bytes.
fn edid_with(extensions: &[&[u8]]) -> Vec<u8> {
    let mut edid = vec![0u8; 128];
    edid[..8].copy_from_slice(&[0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0]);
    edid[126] = extensions.len() as u8;
    for extension in extensions {
        let mut block = extension.to_vec();
        block.resize(128, 0);
        edid.extend(block);
    }
    edid
}

#[test]
fn the_line_shows_the_edid_timing_blocks_linux_lists_no_mode_from() {
    // 2560x1440 as a formula timing: active less one, then the refresh
    // less one.
    let size = [0xff, 0x09, 0x9f, 0x05];
    let mut blocks = Vec::new();
    // Type VII detailed timing, which Linux reads.
    blocks.extend([0x22, 0x00, 20]);
    blocks.extend([0u8; 20]);
    // Type X, six bytes, CVT at 144 Hz: read.
    blocks.extend([0x2a, 0x00, 6, 0x00]);
    blocks.extend(size);
    blocks.push(143);
    // Type X, seven bytes, 320 Hz: its rate needs the seventh byte, and
    // Linux reads no seven-byte descriptor.
    blocks.extend([0x2a, 0x10, 7, 0x00]);
    blocks.extend(size);
    blocks.extend([0x3f, 0x01]);
    // Type X, six bytes, reduced blanking 2 at 240 Hz: not read.
    blocks.extend([0x2a, 0x00, 6, 0x02]);
    blocks.extend(size);
    blocks.push(239);
    // Type III short timings: not read.
    blocks.extend([0x05, 0x00, 3, 0x01, 0x4f, 0x3b]);
    let mut displayid = vec![0x70, 0x20, blocks.len() as u8, 0, 0];
    displayid.extend(&blocks);
    // A CTA extension: a video data block, which is read, and a type X
    // video timing block, which is not.
    let mut cta = vec![0x02, 0x03, 0, 0];
    cta.extend([0x41, 0x10]);
    cta.extend([0xe8, 0x2a, 0x00, 0x02]);
    cta.extend(size);
    cta.push(239);
    cta[2] = cta.len() as u8;
    let mut card = Card::new(vec![at(2, 2, 0, 60)]);
    card.edids[1] = Some(edid_with(&[&displayid, &cta]));
    let mut display = Display::take(&mut card, (2, 2)).unwrap();
    assert_eq!(
        lines(&mut display),
        [
            "Unknown-0 connected, gives the picture 2x2; modes: runs: 2x2 60; EDID timing blocks \
          Linux lists no mode from: DisplayID 0x2a revision 0x10 [00ff099f053f01] DisplayID \
          0x2a revision 0x00 [02ff099f05ef] DisplayID 0x05 revision 0x00 [014f3b] CTA 0x2a \
          [0002ff099f05ef]"
        ]
    );
    drop(display);

    // Blocks that run past their section or their extension end the read,
    // and a long block is shown in part.
    let mut long = vec![0x70, 0x20, 121, 0, 0, 0x05, 0x00, 118];
    long.extend([0x11; 118]);
    let overrun = [0x70, 0x20, 121, 0, 0, 0x05, 0x00, 200];
    let cta_overrun = [0x02, 0x03, 0xff, 0, 0xff];
    for edid in [
        edid_with(&[&long]),
        edid_with(&[&overrun]),
        edid_with(&[&cta_overrun]),
        edid_with(&[&[0x70; 128]]),
        edid_with(&[&[0x02; 128]]),
        vec![0xff; 300],
    ] {
        let mut card = Card::new(vec![at(2, 2, 0, 60)]);
        card.edids[1] = Some(edid);
        let mut display = Display::take(&mut card, (2, 2)).unwrap();
        let line = lines(&mut display).remove(0);
        assert!(line.len() < 400, "{line}");
        assert!(!line.contains("EDID") || line.ends_with("and 22 more bytes]"));
    }
}

/// An 18-byte detailed timing of `width` by `height` active pixels with
/// `h_blank` and `v_blank` more, at `clock` kHz.
fn dtd(clock: u32, (width, h_blank): (u32, u32), (height, v_blank): (u32, u32)) -> [u8; 18] {
    let ten = (clock / 10) as u16;
    let byte = |value: u32| value as u8;
    let mut timing = [0u8; 18];
    timing[..2].copy_from_slice(&ten.to_le_bytes());
    timing[2] = byte(width);
    timing[3] = byte(h_blank);
    timing[4] = byte(width >> 8 << 4 | h_blank >> 8);
    timing[5] = byte(height);
    timing[6] = byte(v_blank);
    timing[7] = byte(height >> 8 << 4 | v_blank >> 8);
    // Sync offsets and widths: 8 and 32 pixels, 3 and 5 lines.
    timing[8..11].copy_from_slice(&[8, 32, 0x35]);
    timing[17] = 0x1e;
    timing
}

/// A DisplayID type I (10 kHz) or type VII (kHz) detailed timing.
fn displayid_timing(
    clock: u32,
    (width, h_blank): (u32, u32),
    (height, v_blank): (u32, u32),
) -> [u8; 20] {
    let mut timing = [0u8; 20];
    timing[..3].copy_from_slice(&(clock - 1).to_le_bytes()[..3]);
    for (at, value) in [
        (4, width),
        (6, h_blank),
        (8, 8),
        (10, 32),
        (12, height),
        (14, v_blank),
        (16, 3),
        (18, 5),
    ] {
        timing[at..at + 2].copy_from_slice(&((value - 1) as u16).to_le_bytes());
    }
    timing
}

/// A mode of `width` by `height` at `clock` kHz with these totals.
fn totalled(width: u16, height: u16, clock: u32, h_total: u16, v_total: u16) -> ModeInfo {
    ModeInfo {
        clock,
        timing_h: [0, 0, h_total, 0],
        timing_v: [0, 0, v_total, 0],
        ..mode(width, height, 0)
    }
}

/// The monitor's EDID: 1920x1080 at 60 Hz in the base block, at 144 Hz in
/// its CTA extension after HDMI's blocks `hdmi`, and at 320 Hz (type I)
/// and 2560x1440 at 360 Hz (type VII) in a DisplayID extension.
fn fast_monitor(hdmi: &[&[u8]]) -> Vec<u8> {
    let mut cta = vec![0x02, 0x03, 0, 0];
    for block in hdmi {
        cta.extend(*block);
    }
    cta[2] = cta.len() as u8;
    cta.extend(dtd(356_400, (1920, 280), (1080, 45)));
    let mut blocks = vec![0x03, 0x00, 20];
    blocks.extend(displayid_timing(74_624, (1920, 80), (1080, 86)));
    blocks.extend([0x22, 0x00, 20]);
    blocks.extend(displayid_timing(1_449_360, (2560, 80), (1440, 85)));
    let mut displayid = vec![0x70, 0x12, blocks.len() as u8, 0, 0];
    displayid.extend(blocks);
    let mut edid = edid_with(&[&cta, &displayid]);
    edid[54..72].copy_from_slice(&dtd(148_500, (1920, 280), (1080, 45)));
    edid
}

/// HDMI's vendor block with a 340 MHz TMDS clock, and the HDMI Forum's
/// with 600 MHz and SCDC.
const HDMI_VSDB: [u8; 8] = [0x67, 0x03, 0x0c, 0x00, 0x10, 0x00, 0x00, 0x44];
const HF_VSDB: [u8; 8] = [0x67, 0xd8, 0x5d, 0xc4, 0x01, 0x78, 0x80, 0x00];

#[test]
fn the_line_names_each_edid_timing_the_driver_lists_no_mode_for_and_the_link_limit_it_passes() {
    let line = |kind: u32, edid: Vec<u8>, driver: &'static [u8]| {
        // Linux listed the 60 and 144 Hz timings only.
        let mut card = Card::new(vec![
            totalled(1920, 1080, 148_500, 2200, 1125),
            totalled(1920, 1080, 356_400, 2200, 1125),
        ]);
        card.ports = [(kind, 1), (kind, 1)];
        card.edids[1] = Some(edid);
        card.driver = driver;
        let mut display = Display::take(&mut card, (1920, 1080)).unwrap();
        let line = lines(&mut display).remove(0);
        let at = line.find("; EDID timings").map_or(line.len(), |at| at + 2);
        String::from(&line[at..])
    };
    let scrambled = fast_monitor(&[&HDMI_VSDB, &HF_VSDB]);
    // HDMI with scrambling: 594 MHz, under the monitor's 600.
    assert_eq!(
        line(11, scrambled.clone(), b"nouveau"),
        "EDID timings Linux lists no mode for: 1920x1080 320 at 746.24 MHz, past the 594 MHz \
         TMDS clock nouveau allows this HDMI monitor; HDMI 2.1's FRL links, which carry more, \
         nouveau does not drive, 2560x1440 360 at 1449.36 MHz, past the 594 MHz TMDS clock \
         nouveau allows this HDMI monitor; HDMI 2.1's FRL links, which carry more, nouveau \
         does not drive"
    );
    // DisplayPort: 1440 MHz at 6 bits a colour on 4 lanes of HBR3.
    assert_eq!(
        line(10, scrambled.clone(), b"nouveau"),
        "EDID timings Linux lists no mode for: 1920x1080 320 at 746.24 MHz, within 4 lanes of \
         HBR3 at 6 bits a colour, so the monitor's link offers fewer lanes or a lower rate, or \
         an adapter limits the clock, 2560x1440 360 at 1449.36 MHz, past what nouveau's \
         DisplayPort carries, 4 lanes of HBR3 at 6 bits a colour (1440 MHz); nouveau has no \
         DSC and no UHBR"
    );
    // Another driver: the timings, no reason.
    assert_eq!(
        line(11, scrambled, b"amdgpu"),
        "EDID timings Linux lists no mode for: 1920x1080 320 at 746.24 MHz, not listed by the \
         driver, 2560x1440 360 at 1449.36 MHz, not listed by the driver"
    );
    // HDMI without the Forum's block: 340 MHz.
    let plain = line(11, fast_monitor(&[&HDMI_VSDB]), b"nouveau");
    assert!(
        plain.contains("1920x1080 320 at 746.24 MHz, past the 340 MHz TMDS clock"),
        "{plain}"
    );
    // No HDMI block: single-link TMDS.
    let dvi = line(3, fast_monitor(&[]), b"nouveau");
    assert!(
        dvi.contains("1920x1080 320 at 746.24 MHz, past single-link TMDS's 165 MHz"),
        "{dvi}"
    );

    // Every timing listed, as on a usual monitor: nothing is added.
    let mut card = Card::new(vec![totalled(1920, 1080, 148_500, 2200, 1125)]);
    let mut edid = edid_with(&[]);
    edid[54..72].copy_from_slice(&dtd(148_500, (1920, 280), (1080, 45)));
    // A stereo timing Linux makes no mode of.
    let mut stereo = dtd(148_500, (1280, 280), (1024, 45));
    stereo[17] |= 0x20;
    edid[72..90].copy_from_slice(&stereo);
    card.edids[1] = Some(edid);
    let mut display = Display::take(&mut card, (1920, 1080)).unwrap();
    assert_eq!(
        lines(&mut display),
        [
            "Unknown-0 connected, gives the picture 1920x1080; modes: runs: 1920x1080 60; EDID \
             timings Linux lists no mode for: 1280x1024 89.048 at 148.5 MHz, a timing Linux \
             takes as invalid (stereo, or no sync width)"
        ]
    );
    assert_eq!(size_of::<Version>(), 64);
}
