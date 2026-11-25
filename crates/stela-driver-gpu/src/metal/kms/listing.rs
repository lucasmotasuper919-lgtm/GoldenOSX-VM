//! One connector as a take or a look read it, for the boot log: every mode
//! it lists, the one it runs and why each other was passed over, and the
//! timing blocks of its EDID that Linux lists no mode from. A report of a
//! size or a refresh rate that macOS was not offered is answered by this
//! line: the mode was not listed, or it was and the line says what it lost
//! to.
use super::{Error, Fits, INTERLACED, ModeInfo, PREFERRED, Port, refresh, size_of_mode};
use alloc::vec::Vec;
use core::fmt;

/// The most modes a line names; the rest are counted.
const LISTED_MODES: usize = 96;
/// The most bytes a line shows of EDID timing blocks read as no mode.
const UNREAD_BYTES: usize = 96;
/// The most EDID timings without a mode a line names; the rest are
/// counted.
const MISSED_TIMINGS: usize = 16;

/// Why a size some output lists is not the picture's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Lost {
    /// No output's mode of it can be told to macOS.
    Untold,
    /// The guest's size is set, and only that size is shown.
    Asked,
    /// Fewer outputs list it than list the picture's.
    Fewer,
    /// It ranks after the picture's: the output's own mode, then the sizes
    /// of that shape, the firmware's, then the others as listed.
    Later,
    /// The picture's size was set before this output was read.
    Fixed,
}

/// What became of one mode an output lists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Verdict {
    /// The output runs it.
    Runs,
    /// The output runs it, and its scaler fills it with the picture.
    Scaled,
    /// The picture's size at another rate, which macOS is offered.
    Offered,
    /// The picture's size at a rate a mode listed before it has.
    Repeated,
    /// The picture's size at a rate macOS cannot be told of.
    Untold,
    /// The picture's size, interlaced.
    Interlaced,
    /// The picture's size on an output that follows the first: not the
    /// rate it runs.
    OtherRate,
    /// Another size than the picture's.
    Size(Lost),
    /// The output runs nothing.
    Unused,
    /// The driver refused to set it.
    Refused,
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Runs => "runs",
            Self::Scaled => "runs, the picture scaled into it",
            Self::Offered => "offered to macOS as a rate",
            Self::Repeated => "a rate listed before",
            Self::Untold => "a rate macOS cannot be told of",
            Self::Interlaced => "interlaced",
            Self::OtherRate => "not the rate run",
            Self::Size(Lost::Untold) => "other size, which macOS cannot be told of",
            Self::Size(Lost::Asked) => "not the size asked for",
            Self::Size(Lost::Fewer) => "other size, which fewer outputs list",
            Self::Size(Lost::Later) => "other size, ranked after the picture's",
            Self::Size(Lost::Fixed) => "not the picture's size",
            Self::Unused => "not run",
            Self::Refused => "refused by the driver",
        })
    }
}

/// Why a connected output, or one the firmware lit, shows nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Dark {
    /// The firmware lit it and no display is detected on it.
    Absent,
    /// It lists no mode.
    NoModes,
    /// None of its encoders reaches a controller.
    Unreachable,
    /// No controller or encoder of its own is free.
    Taken,
    /// Its native mode is narrower or shorter than the picture.
    Smaller,
    /// No mode of the picture's size, and the connector cannot scale.
    NoScaler,
    /// The driver refused its mode set or its scaling.
    Refused(Error),
}

impl Dark {
    /// The output would need the picture made smaller.
    pub(super) fn smaller(self) -> bool {
        self == Self::Smaller
    }
}

impl fmt::Display for Dark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Absent => f.write_str("no display detected"),
            Self::NoModes => f.write_str("it lists no mode"),
            Self::Unreachable => f.write_str("its encoders reach no controller"),
            Self::Taken => f.write_str("no controller or encoder of its own is free"),
            Self::Smaller => {
                f.write_str("its native mode is smaller than the picture, which is only scaled up")
            }
            Self::NoScaler => f.write_str("no mode of the picture's size and no scaling property"),
            Self::Refused(error) => write!(f, "the driver refused to set it: {error:?}"),
        }
    }
}

/// What an output does with the picture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Role {
    /// It gives the picture its size, its rates and the monitor's name.
    Gives,
    /// It shows the picture in its own mode of that size.
    Shows,
    /// It shows the picture scaled into its native mode.
    ShowsScaled,
    Dark(Dark),
}

/// One listed mode, as the line names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Listed {
    size: (u32, u32),
    interlaced: bool,
    preferred: bool,
    /// In millihertz.
    refresh: u32,
    verdict: Verdict,
}

/// An EDID timing block Linux lists no mode from: a DisplayID data block,
/// or a data block of the CTA extension with an extended tag.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Unread {
    cta: bool,
    tag: u8,
    /// A DisplayID block's revision byte, which holds its descriptors' size.
    revision: u8,
    bytes: Vec<u8>,
}

/// A detailed timing of the monitor's EDID that Linux makes a mode of, as
/// its EDID parser computes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Timing {
    size: (u32, u32),
    /// In millihertz.
    refresh: u32,
    /// The pixel clock in kilohertz.
    clock: u32,
    /// Linux makes no mode of it: a stereo timing or one with no sync
    /// width (`drm_mode_detailed`).
    invalid: bool,
}

/// Why the driver lists no mode for a timing of the monitor's EDID, as far
/// as the boot log can tell: nouveau's checks of a mode against its output
/// (`nouveau_connector_mode_valid`), with the monitor's limits read from
/// its EDID as Linux reads them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Missed {
    /// Linux takes the timing as invalid.
    Invalid,
    /// Past the TMDS clock in kilohertz nouveau allows this HDMI monitor:
    /// 594 MHz with scrambling, else 340 MHz, or less where the EDID says
    /// so. HDMI 2.1's FRL links, which carry more, nouveau does not drive.
    Hdmi(u32),
    /// Past single-link TMDS's 165 MHz for a monitor whose EDID has no
    /// HDMI block, twice that on an output with dual-link DVI.
    Dvi,
    /// Past what nouveau's DisplayPort carries at all: the mode at 6 bits
    /// a colour against 4 lanes of HBR3, without DSC.
    DisplayPort,
    /// Within that: the monitor's link offers fewer lanes or a lower rate,
    /// or something else between them limits the clock.
    Link,
    /// The driver is not nouveau, or the connector not one of these kinds.
    Unknown,
}

impl fmt::Display for Missed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid => {
                f.write_str("a timing Linux takes as invalid (stereo, or no sync width)")
            }
            Self::Hdmi(link) => write!(
                f,
                "past the {} MHz TMDS clock nouveau allows this HDMI monitor; HDMI 2.1's FRL \
                 links, which carry more, nouveau does not drive",
                Hertz(*link)
            ),
            Self::Dvi => f.write_str(
                "past single-link TMDS's 165 MHz for a monitor with no HDMI block (twice that \
                 on a dual-link DVI output)",
            ),
            Self::DisplayPort => f.write_str(
                "past what nouveau's DisplayPort carries, 4 lanes of HBR3 at 6 bits a colour \
                 (1440 MHz); nouveau has no DSC and no UHBR",
            ),
            Self::Link => f.write_str(
                "within 4 lanes of HBR3 at 6 bits a colour, so the monitor's link offers fewer \
                 lanes or a lower rate, or an adapter limits the clock",
            ),
            Self::Unknown => f.write_str("not listed by the driver"),
        }
    }
}

/// `DRM_MODE_CONNECTOR_*` kinds nouveau drives as TMDS (DVI-I, DVI-D,
/// HDMI-A, HDMI-B) and as DisplayPort (DP, eDP).
const TMDS_PORTS: [u32; 4] = [2, 3, 11, 12];
const DP_PORTS: [u32; 2] = [10, 14];
/// nouveau's limits from Turing on (`get_tmds_link_bandwidth`) and the
/// most its DisplayPort trains: 4 lanes of HBR3, 810000 kilobytes a second
/// each, checked at 6 bits a colour (`nv50_dp_mode_valid`).
const SCRAMBLED_TMDS: u32 = 594_000;
const PLAIN_TMDS: u32 = 340_000;
const SINGLE_LINK_TMDS: u32 = 165_000;
const DP_BYTES: u64 = 4 * 810_000;
const DP_MIN_BITS: u64 = 6 * 3;

/// One connector as read, shown as its boot log line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Listing {
    pub(super) connector: u32,
    port: Port,
    connected: bool,
    lit: bool,
    picture: (u32, u32),
    role: Role,
    modes: Vec<Listed>,
    /// The EDID's detailed timings the driver lists no mode for, and why.
    missed: Vec<(Timing, Missed)>,
    unread: Vec<Unread>,
}

impl Listing {
    /// Connector `connector` at `port` with `modes`, each with its verdict
    /// in `verdicts`, showing a `picture` as `role` says; `edid` is the
    /// connector's, when it has one, and `nouveau` whether its driver is.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        connector: u32,
        port: Port,
        (connected, lit): (bool, bool),
        picture: (u32, u32),
        role: Role,
        modes: &[ModeInfo],
        verdicts: &[Verdict],
        edid: Option<&[u8]>,
        nouveau: bool,
    ) -> Self {
        let modes: Vec<Listed> = modes
            .iter()
            .zip(verdicts)
            .map(|(mode, &verdict)| Listed {
                size: size_of_mode(mode),
                interlaced: mode.flags & INTERLACED != 0,
                preferred: mode.kind & PREFERRED != 0,
                refresh: refresh(mode),
                verdict,
            })
            .collect();
        let missed = match (edid, connected) {
            (Some(edid), true) => missed(edid, &modes, port.kind, nouveau),
            _ => Vec::new(),
        };
        Self {
            connector,
            port,
            connected,
            lit,
            picture,
            role,
            modes,
            missed,
            unread: edid.map(unread).unwrap_or_default(),
        }
    }

    /// The driver refused to set the output: it is dark, and the mode it
    /// was to run is the one refused.
    pub(super) fn refuse(&mut self, error: Error) {
        self.role = Role::Dark(Dark::Refused(error));
        for mode in &mut self.modes {
            if matches!(mode.verdict, Verdict::Runs | Verdict::Scaled) {
                mode.verdict = Verdict::Refused;
            }
        }
    }
}

/// The verdicts of the `modes` of the output that gives a picture of
/// `size` in the mode `set`, the rates of which `fits` are offered to
/// macOS, the first listed of each; `lost` says why another size is not
/// the picture's.
pub(super) fn gives(
    modes: &[ModeInfo],
    size: (u32, u32),
    set: &ModeInfo,
    fits: Fits<'_>,
    lost: &dyn Fn((u32, u32)) -> Lost,
) -> Vec<Verdict> {
    let mut ran = false;
    let mut rates = alloc::vec![refresh(set)];
    modes
        .iter()
        .map(|mode| {
            let rate = refresh(mode);
            match size_of_mode(mode) {
                other if other != size => Verdict::Size(lost(other)),
                _ if !ran && mode == set => {
                    ran = true;
                    Verdict::Runs
                }
                _ if mode.flags & INTERLACED != 0 => Verdict::Interlaced,
                _ if !fits(size.0, size.1, rate) => Verdict::Untold,
                _ if rates.contains(&rate) => Verdict::Repeated,
                _ => {
                    rates.push(rate);
                    Verdict::Offered
                }
            }
        })
        .collect()
}

/// The verdicts of the `modes` of an output that follows the first, or of
/// one read after the picture's `size` was set: `run` is the mode it runs
/// and whether the picture is scaled into it, `None` for a dark one.
pub(super) fn follows(
    modes: &[ModeInfo],
    size: (u32, u32),
    run: Option<(ModeInfo, bool)>,
) -> Vec<Verdict> {
    let mut ran = false;
    modes
        .iter()
        .map(|mode| match run {
            Some((set, scaled)) if !ran && *mode == set => {
                ran = true;
                match scaled {
                    true => Verdict::Scaled,
                    false => Verdict::Runs,
                }
            }
            _ if size_of_mode(mode) != size => Verdict::Size(Lost::Fixed),
            _ if mode.flags & INTERLACED != 0 => Verdict::Interlaced,
            None => Verdict::Unused,
            Some(_) => Verdict::OtherRate,
        })
        .collect()
}

/// EDID's extension tags: CTA-861 and DisplayID.
const CTA_EXTENSION: u8 = 0x02;
const DISPLAYID_EXTENSION: u8 = 0x70;
/// A CTA data block's tag for an extended tag in its first byte, and for
/// a vendor's block, whose first three bytes are its IEEE OUI, low byte
/// first: HDMI's and the HDMI Forum's.
const CTA_EXTENDED: u8 = 7;
const CTA_VENDOR: u8 = 3;
const HDMI_OUI: [u8; 3] = [0x03, 0x0c, 0x00];
const HDMI_FORUM_OUI: [u8; 3] = [0xd8, 0x5d, 0xc4];
/// The HDMI Forum's sink capability data block's extended tag.
const CTA_HF_SCDB: u8 = 0x79;
/// CTA-861's extended tags of the DisplayID type VII, VIII and X video
/// timing data blocks, none of which Linux reads.
const CTA_TIMINGS: [u8; 3] = [0x22, 0x23, 0x2a];
/// DisplayID's type II detailed, type III short, type IV code and type VIII
/// enumerated timing blocks, none of which Linux reads.
const DISPLAYID_UNREAD: [u8; 4] = [0x04, 0x05, 0x06, 0x23];
/// DisplayID's type IX and type X formula timing blocks.
const DISPLAYID_FORMULA: [u8; 2] = [0x24, 0x2a];
/// DisplayID's type I and type VII detailed timing blocks, which Linux
/// reads.
const DISPLAYID_DETAILED: [u8; 2] = [0x03, 0x22];

/// The timing blocks of `edid`'s extensions that Linux lists no mode from
/// (`drm_edid.c`, `add_displayid_detailed_modes` and `add_cea_modes`): a
/// DisplayID block of a type it does not read; a formula block (types IX
/// and X) whose descriptors are not six bytes, as a rate past 256 Hz needs
/// seven, or one of whose descriptors uses a reduced blanking formula past
/// the first; and the CTA extension's blocks that carry DisplayID timings.
/// Such blocks hold timings alone, no name or serial.
fn unread(edid: &[u8]) -> Vec<Unread> {
    let mut unread = Vec::new();
    for extension in edid.as_chunks::<128>().0.iter().skip(1) {
        match extension[0] {
            DISPLAYID_EXTENSION => displayid_unread(extension, &mut unread),
            CTA_EXTENSION => cta_unread(extension, &mut unread),
            _ => {}
        }
    }
    unread
}

/// Each data block of one DisplayID extension block, as its tag, its
/// revision and its bytes: a section header of four bytes after the
/// extension's tag, then data blocks of a tag, a revision and a length.
/// A block past the section ends the read, as Linux's iterator does.
fn displayid_blocks(extension: &[u8], mut each: impl FnMut(u8, u8, &[u8])) {
    let end = (5 + usize::from(extension[2])).min(extension.len() - 1);
    let mut at = 5;
    while at + 3 <= end {
        let (tag, revision, length) = (extension[at], extension[at + 1], extension[at + 2]);
        let Some(bytes) = extension.get(at + 3..at + 3 + usize::from(length)) else {
            return;
        };
        if at + 3 + bytes.len() > end || (tag == 0 && length == 0) {
            return;
        }
        at += 3 + bytes.len();
        each(tag, revision, bytes);
    }
}

/// Each data block of one CTA-861 extension block, as its tag and its
/// payload: data blocks from its fifth byte to its detailed timings, each
/// a tag and a length in one byte.
fn cta_blocks(extension: &[u8], mut each: impl FnMut(u8, &[u8])) {
    let end = usize::from(extension[2]).min(extension.len() - 1);
    let mut at = 4;
    while at < end {
        let (tag, length) = (extension[at] >> 5, usize::from(extension[at] & 0x1f));
        let Some(bytes) = extension.get(at + 1..at + 1 + length) else {
            return;
        };
        if at + 1 + length > end {
            return;
        }
        at += 1 + length;
        each(tag, bytes);
    }
}

/// The timing of an 18-byte EDID detailed timing descriptor, as Linux's
/// `drm_mode_detailed` reads it; none for a display descriptor, an
/// interlaced timing or one under 64 pixels a side, which Linux ignores.
fn detailed(descriptor: &[u8]) -> Option<Timing> {
    let clock = u32::from(u16::from_le_bytes([descriptor[0], descriptor[1]])) * 10;
    let high = |byte: u8, shift: u32| u32::from(byte) << shift;
    let width = u32::from(descriptor[2]) | high(descriptor[4] & 0xf0, 4);
    let h_blank = u32::from(descriptor[3]) | high(descriptor[4] & 0x0f, 8);
    let height = u32::from(descriptor[5]) | high(descriptor[7] & 0xf0, 4);
    let v_blank = u32::from(descriptor[6]) | high(descriptor[7] & 0x0f, 8);
    let h_sync = u32::from(descriptor[9]) | high(descriptor[11] & 0x30, 4);
    let v_sync = u32::from(descriptor[10] & 0x0f) | high(descriptor[11] & 0x03, 4);
    let misc = descriptor[17];
    if clock == 0 || width < 64 || height < 64 || misc & 0x80 != 0 {
        return None;
    }
    Some(Timing {
        size: (width, height),
        refresh: rate(clock, width + h_blank, height + v_blank),
        clock,
        invalid: misc & 0x20 != 0 || h_sync == 0 || v_sync == 0,
    })
}

/// The timing of a DisplayID type I or type VII detailed timing descriptor
/// of 20 bytes, as Linux's `drm_mode_displayid_detailed` reads it: each
/// field one less than its value, the clock in 10 kHz for type I and in
/// kHz for type VII.
fn displayid_detailed(descriptor: &[u8], type_7: bool) -> Option<Timing> {
    let field = |at: usize| u32::from(u16::from_le_bytes([descriptor[at], descriptor[at + 1]]));
    let clock = (u32::from_le_bytes([descriptor[0], descriptor[1], descriptor[2], 0]) + 1)
        * if type_7 { 1 } else { 10 };
    let (width, h_blank) = (field(4) + 1, field(6) + 1);
    let (height, v_blank) = (field(12) + 1, field(14) + 1);
    if width < 64 || height < 64 {
        return None;
    }
    Some(Timing {
        size: (width, height),
        refresh: rate(clock, width + h_blank, height + v_blank),
        clock,
        invalid: false,
    })
}

/// The refresh in millihertz of a `clock` kilohertz timing of `h_total`
/// by `v_total` pixels, rounded as [`refresh`].
fn rate(clock: u32, h_total: u32, v_total: u32) -> u32 {
    let total = u64::from(h_total) * u64::from(v_total);
    u32::try_from((u64::from(clock) * 1_000_000 + total / 2) / total.max(1)).unwrap_or(u32::MAX)
}

/// The detailed timings of `edid` Linux makes modes of: the base block's,
/// the CTA extension's, and the type I and type VII timings of DisplayID
/// extensions, each once.
fn timings(edid: &[u8]) -> Vec<Timing> {
    let mut timings: Vec<Timing> = Vec::new();
    let mut add = |timing: Option<Timing>| {
        if let Some(timing) = timing
            && !timings
                .iter()
                .any(|seen| (seen.size, seen.refresh) == (timing.size, timing.refresh))
        {
            timings.push(timing);
        }
    };
    let blocks = edid.as_chunks::<128>().0;
    if let Some(base) = blocks.first() {
        for descriptor in base[54..126].as_chunks::<18>().0 {
            add(detailed(descriptor));
        }
    }
    for extension in blocks.iter().skip(1) {
        match extension[0] {
            // `cea_for_each_detailed_block`.
            CTA_EXTENSION if (4..=127).contains(&extension[2]) => {
                let start = usize::from(extension[2]);
                for descriptor in extension[start..127].as_chunks::<18>().0 {
                    add(detailed(descriptor));
                }
            }
            // `add_displayid_detailed_1_modes`: whole descriptors only.
            DISPLAYID_EXTENSION => displayid_blocks(extension, |tag, _, bytes| {
                if DISPLAYID_DETAILED.contains(&tag) && bytes.len() % 20 == 0 {
                    for descriptor in bytes.as_chunks::<20>().0 {
                        add(displayid_detailed(descriptor, tag == DISPLAYID_DETAILED[1]));
                    }
                }
            }),
            _ => {}
        }
    }
    timings
}

/// The TMDS clock in kilohertz nouveau allows a monitor with `edid` on an
/// HDMI or DVI output (`get_tmds_link_bandwidth`), from the CTA blocks
/// Linux reads for it (`drm_parse_hdmi_vsdb_video`,
/// `drm_parse_hdmi_forum_scds`); none for a monitor with no HDMI block.
fn hdmi_tmds(edid: &[u8]) -> Option<u32> {
    let (mut hdmi, mut max, mut scdc, mut scrambling) = (false, 0, false, false);
    for extension in edid.as_chunks::<128>().0.iter().skip(1) {
        if extension[0] != CTA_EXTENSION {
            continue;
        }
        cta_blocks(extension, |tag, payload| {
            let vendor = |oui: [u8; 3]| tag == CTA_VENDOR && payload.get(..3) == Some(&oui);
            if vendor(HDMI_OUI) && payload.len() >= 5 {
                hdmi = true;
                if let Some(&clock) = payload.get(6) {
                    max = u32::from(clock) * 5000;
                }
            } else if (vendor(HDMI_FORUM_OUI)
                || (tag == CTA_EXTENDED && payload.first() == Some(&CTA_HF_SCDB)))
                && payload.len() >= 7
            {
                // The block's bytes 5 and 6, counting its tag byte.
                scdc |= payload[5] & 0x80 != 0;
                if payload[4] != 0 {
                    let clock = u32::from(payload[4]) * 5000;
                    if clock > PLAIN_TMDS {
                        max = clock;
                    }
                    scrambling |= scdc;
                }
            }
        });
    }
    let link = if scrambling {
        SCRAMBLED_TMDS
    } else {
        PLAIN_TMDS
    };
    hdmi.then(|| if max != 0 { max.min(link) } else { link })
}

/// The detailed timings of `edid` for which a connector of `kind` lists
/// none of `modes`, each with why; the reason is nouveau's where `nouveau`
/// drives it. A mode of the timing's size and refresh, to half a hertz,
/// is its own.
fn missed(edid: &[u8], modes: &[Listed], kind: u32, nouveau: bool) -> Vec<(Timing, Missed)> {
    let tmds = hdmi_tmds(edid);
    timings(edid)
        .into_iter()
        .filter(|timing| {
            !modes.iter().any(|mode| {
                !mode.interlaced
                    && mode.size == timing.size
                    && mode.refresh.abs_diff(timing.refresh) <= 500
            })
        })
        .map(|timing| {
            let why = match (nouveau, tmds) {
                _ if timing.invalid => Missed::Invalid,
                (false, _) => Missed::Unknown,
                (true, Some(link)) if TMDS_PORTS.contains(&kind) && timing.clock > link => {
                    Missed::Hdmi(link)
                }
                (true, None) if TMDS_PORTS.contains(&kind) && timing.clock > SINGLE_LINK_TMDS => {
                    Missed::Dvi
                }
                (true, _) if DP_PORTS.contains(&kind) => {
                    match u64::from(timing.clock) * DP_MIN_BITS > DP_BYTES * 8 {
                        true => Missed::DisplayPort,
                        false => Missed::Link,
                    }
                }
                (true, _) => Missed::Unknown,
            };
            (timing, why)
        })
        .collect()
}

/// [`unread`] for one DisplayID extension block.
fn displayid_unread(extension: &[u8], unread: &mut Vec<Unread>) {
    displayid_blocks(extension, |tag, revision, bytes| {
        let descriptor = 6 + usize::from(revision >> 4 & 7);
        let formula = DISPLAYID_FORMULA.contains(&tag)
            && (descriptor != 6
                || bytes.len() % descriptor != 0
                || bytes
                    .chunks_exact(descriptor)
                    .any(|timing| timing[0] & 7 > 1));
        if formula || DISPLAYID_UNREAD.contains(&tag) {
            unread.push(Unread {
                cta: false,
                tag,
                revision,
                bytes: bytes.to_vec(),
            });
        }
    });
}

/// [`unread`] for one CTA-861 extension block.
fn cta_unread(extension: &[u8], unread: &mut Vec<Unread>) {
    cta_blocks(extension, |tag, bytes| {
        if let (CTA_EXTENDED, Some((&extended, bytes))) = (tag, bytes.split_first())
            && CTA_TIMINGS.contains(&extended)
        {
            unread.push(Unread {
                cta: true,
                tag: extended,
                revision: 0,
                bytes: bytes.to_vec(),
            });
        }
    });
}

/// A refresh in millihertz as hertz, with the decimals it has.
struct Hertz(u32);

impl fmt::Display for Hertz {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (whole, mut part) = (self.0 / 1000, self.0 % 1000);
        write!(f, "{whole}")?;
        if part == 0 {
            return Ok(());
        }
        let mut digits = 3;
        while part.is_multiple_of(10) {
            part /= 10;
            digits -= 1;
        }
        write!(f, ".{part:0digits$}")
    }
}

impl fmt::Display for Listing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = match self.connected {
            true => "connected",
            false => "not connected",
        };
        write!(f, "{} {state}", self.port)?;
        if self.lit {
            f.write_str(", lit by the firmware")?;
        }
        let (width, height) = self.picture;
        match self.role {
            Role::Gives => write!(f, ", gives the picture {width}x{height}")?,
            Role::Shows => f.write_str(", shows the picture")?,
            Role::ShowsScaled => f.write_str(", shows the picture scaled")?,
            Role::Dark(dark) => write!(f, ", dark: {dark}")?,
        }
        let named = &self.modes[..self.modes.len().min(LISTED_MODES)];
        if !named.is_empty() {
            f.write_str("; modes")?;
            if named.iter().any(|mode| mode.preferred) {
                f.write_str(" (* preferred)")?;
            }
            f.write_str(": ")?;
        }
        // Each verdict once, what runs first, then as first listed; under
        // it each size once, as first listed, with its rates.
        let mut verdicts: Vec<Verdict> = Vec::new();
        for mode in named {
            if !verdicts.contains(&mode.verdict) {
                verdicts.push(mode.verdict);
            }
        }
        verdicts.sort_by_key(|verdict| !matches!(verdict, Verdict::Runs | Verdict::Scaled));
        for (index, verdict) in verdicts.iter().enumerate() {
            if index != 0 {
                f.write_str("; ")?;
            }
            write!(f, "{verdict}:")?;
            let of = |mode: &&Listed| mode.verdict == *verdict;
            let mut shapes: Vec<((u32, u32), bool)> = Vec::new();
            for mode in named.iter().filter(of) {
                if !shapes.contains(&(mode.size, mode.interlaced)) {
                    shapes.push((mode.size, mode.interlaced));
                }
            }
            for (index, &(size, interlaced)) in shapes.iter().enumerate() {
                if index != 0 {
                    f.write_str(",")?;
                }
                write!(f, " {}x{}", size.0, size.1)?;
                if interlaced {
                    f.write_str("i")?;
                }
                let shaped = named
                    .iter()
                    .filter(of)
                    .filter(|mode| (mode.size, mode.interlaced) == (size, interlaced));
                for mode in shaped {
                    write!(f, " {}", Hertz(mode.refresh))?;
                    if mode.preferred {
                        f.write_str("*")?;
                    }
                }
            }
        }
        if self.modes.len() > named.len() {
            write!(f, "; and {} more", self.modes.len() - named.len())?;
        }
        if !self.missed.is_empty() {
            f.write_str("; EDID timings Linux lists no mode for:")?;
        }
        let missed = &self.missed[..self.missed.len().min(MISSED_TIMINGS)];
        for (index, (timing, why)) in missed.iter().enumerate() {
            if index != 0 {
                f.write_str(",")?;
            }
            let (width, height) = timing.size;
            write!(
                f,
                " {width}x{height} {} at {} MHz, {why}",
                Hertz(timing.refresh),
                Hertz(timing.clock)
            )?;
        }
        if self.missed.len() > missed.len() {
            write!(f, ", and {} more", self.missed.len() - missed.len())?;
        }
        if !self.unread.is_empty() {
            f.write_str("; EDID timing blocks Linux lists no mode from:")?;
        }
        let mut shown = 0;
        for block in &self.unread {
            let kind = match block.cta {
                true => "CTA",
                false => "DisplayID",
            };
            write!(f, " {kind} {:#04x}", block.tag)?;
            if !block.cta {
                write!(f, " revision {:#04x}", block.revision)?;
            }
            let bytes = &block.bytes[..block.bytes.len().min(UNREAD_BYTES - shown)];
            shown += bytes.len();
            f.write_str(" [")?;
            for byte in bytes {
                write!(f, "{byte:02x}")?;
            }
            match block.bytes.len() - bytes.len() {
                0 => f.write_str("]")?,
                more => write!(f, " and {more} more bytes]")?,
            }
        }
        Ok(())
    }
}
