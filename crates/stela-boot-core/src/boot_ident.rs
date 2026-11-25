//! `BOOT IDENT`: one line that repeats what a boot runs on, every
//! `beat_pace::IDENT_SECONDS` while macOS runs. The lines it repeats are
//! the boot's first, and a log read long after them, or one whose start
//! was cut, still names the version, the processor, the GPU and its
//! driver's state, what the display shows on, the guest's network adapter
//! and the disk's transport.
//!
//! The parts come from the code that learns them, each a short text kept
//! without the heap ([`Part`]); a part no code has set reads `-`. None
//! holds a serial number, an address or a name of the person or the PC.
use alloc::string::String;
use core::fmt::{self, Write as _};

/// The bytes of one part's text.
pub const PART_BYTES: usize = 160;

/// One part of the line: text set, replaced or added to as the boot
/// learns it, cut at [`PART_BYTES`].
#[derive(Clone, Copy)]
pub struct Part {
    bytes: [u8; PART_BYTES],
    len: usize,
}

impl Default for Part {
    fn default() -> Self {
        Self::new()
    }
}

impl Part {
    pub const fn new() -> Self {
        Self {
            bytes: [0; PART_BYTES],
            len: 0,
        }
    }

    /// Replace the text.
    pub fn set(&mut self, text: fmt::Arguments<'_>) {
        self.len = 0;
        let _ = self.write_fmt(text);
    }

    /// Add `text` after what is there, as one more of several.
    pub fn add(&mut self, text: fmt::Arguments<'_>) {
        if self.len != 0 {
            let _ = self.write_str(", ");
        }
        let _ = self.write_fmt(text);
    }

    /// The text, `-` for none.
    pub fn as_str(&self) -> &str {
        match core::str::from_utf8(&self.bytes[..self.len]) {
            Ok("") | Err(_) => "-",
            Ok(text) => text,
        }
    }
}

impl fmt::Write for Part {
    /// What fits, in whole characters; a quote or a line end, which would
    /// end the part early for a reader, becomes a space.
    fn write_str(&mut self, text: &str) -> fmt::Result {
        for character in text.chars() {
            let character = match character {
                '"' | '\n' | '\r' => ' ',
                other => other,
            };
            let mut encoded = [0; 4];
            let encoded = character.encode_utf8(&mut encoded).as_bytes();
            let Some(room) = self.bytes.get_mut(self.len..self.len + encoded.len()) else {
                return Ok(());
            };
            room.copy_from_slice(encoded);
            self.len += encoded.len();
        }
        Ok(())
    }
}

/// What the line says.
pub struct Ident<'a> {
    pub version: &'a str,
    /// The boot's number in the USB's log; 0 without one.
    pub boot: u64,
    pub cpu: &'a str,
    pub gpu: &'a str,
    pub display: &'a str,
    pub network: &'a str,
    pub disk: &'a str,
}

/// `BOOT IDENT version=… boot=… cpu="…" gpu="…" display="…" network="…"
/// disk="…"`.
pub fn line(ident: &Ident<'_>) -> String {
    alloc::format!(
        "BOOT IDENT version={} boot={} cpu=\"{}\" gpu=\"{}\" display=\"{}\" network=\"{}\" \
         disk=\"{}\"\n",
        ident.version,
        ident.boot,
        ident.cpu,
        ident.gpu,
        ident.display,
        ident.network,
        ident.disk
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_line_names_each_part_and_an_unset_one_reads_as_none() {
        let mut gpu = Part::new();
        gpu.add(format_args!("0000:01:00.0 NVIDIA ga104 draws"));
        gpu.add(format_args!("0000:00:02.0 shows only"));
        let mut display = Part::new();
        display.set(format_args!("1920x1080 at 60000 mHz, firmware copy"));
        display.set(format_args!("2560x1440 at 143998 mHz on the GPU's output"));
        let line = line(&Ident {
            version: "0.5.0",
            boot: 12,
            cpu: "AMD Ryzen 5 5600 6-Core Processor",
            gpu: gpu.as_str(),
            display: display.as_str(),
            network: Part::new().as_str(),
            disk: "NVMe",
        });
        assert_eq!(
            line,
            "BOOT IDENT version=0.5.0 boot=12 cpu=\"AMD Ryzen 5 5600 6-Core Processor\" \
             gpu=\"0000:01:00.0 NVIDIA ga104 draws, 0000:00:02.0 shows only\" \
             display=\"2560x1440 at 143998 mHz on the GPU's output\" network=\"-\" disk=\"NVMe\"\n"
        );
    }

    #[test]
    fn a_part_is_cut_at_its_room_in_whole_characters_and_keeps_one_line() {
        let mut part = Part::new();
        part.set(format_args!("a \"quoted\"\nname {}", "é".repeat(200)));
        let text = part.as_str();
        assert!(text.starts_with("a  quoted  name éé"));
        assert!(text.len() <= PART_BYTES && text.len() >= PART_BYTES - 1);
        assert!(!text.contains(['"', '\n']));
        let full = text.len();
        part.add(format_args!("more"));
        assert_eq!(part.as_str().len(), full);
    }
}
