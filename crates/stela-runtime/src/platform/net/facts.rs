//! Counted facts about a bridged guest's traffic to Apple's services: name
//! lookups and their answers, HTTPS connection attempts and resets, TLS
//! handshakes and plaintext alerts, and the guest's clock against a time
//! server's. A sign-in that fails while other traffic works then shows
//! whether a name went unanswered, a server refused or reset connections,
//! TLS stopped with an alert, or the guest's clock was off.
//!
//! The observer reads frame headers only, never changes or keeps a frame,
//! and keeps no name, address or payload in its line: only counts, alert
//! codes and the two clocks. Addresses of Apple's servers are kept in a
//! small table, to count connections to them, and never logged. Apple's
//! services are told by a fixed list of their domains, matched against a
//! lookup's question or a TLS client hello's server name.

use alloc::{format, string::String, vec::Vec};

/// Domains of Apple's services, matched as the name or a suffix after a dot.
const APPLE_DOMAINS: [&[u8]; 10] = [
    b"apple.com",
    b"icloud.com",
    b"icloud-content.com",
    b"apple-cloudkit.com",
    b"apple-dns.net",
    b"aaplimg.com",
    b"cdn-apple.com",
    b"mzstatic.com",
    b"itunes.com",
    b"me.com",
];
/// Addresses of Apple's servers kept, the oldest replaced first.
const ADDRESSES: usize = 64;
/// Distinct TLS alert codes counted each way.
const ALERTS: usize = 8;
/// Answer records read from one lookup's reply.
const ANSWERS: usize = 16;
/// Seconds from the NTP era (1900) to the Unix epoch (1970).
const NTP_UNIX: u64 = 2_208_988_800;

#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct Lookups {
    asked: u64,
    answered: u64,
    empty: u64,
    failed: u64,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct Connections {
    /// Connection attempts (SYN) the guest sent to port 443.
    attempts: u64,
    /// Attempts the server accepted (SYN-ACK).
    accepted: u64,
    /// Resets from the server.
    reset: u64,
    /// TLS client hellos the guest sent, and server hellos back.
    hello: u64,
    server_hello: u64,
}

/// A frame's IP endpoints and its transport payload.
struct Packet<'a> {
    source: [u8; 16],
    destination: [u8; 16],
    protocol: u8,
    payload: &'a [u8],
}

/// The facts, counted since the guest network opened.
#[derive(Default)]
pub struct Facts {
    lookups: Lookups,
    apple_lookups: Lookups,
    connections: Connections,
    apple_connections: Connections,
    /// Plaintext TLS alert codes from servers and from the guest, each
    /// with its count.
    server_alerts: Vec<(u8, u64)>,
    guest_alerts: Vec<(u8, u64)>,
    /// The guest's clock in its last time request, and the server's in
    /// the last reply, in Unix seconds.
    guest_clock: Option<u64>,
    server_clock: Option<u64>,
    apple: Vec<[u8; 16]>,
    next_address: usize,
    /// Whether a fact changed since the last [`Facts::line`].
    changed: bool,
}

impl Facts {
    /// A frame the guest sent toward the network.
    pub fn outbound(&mut self, frame: &[u8]) {
        let Some(packet) = packet(frame) else {
            return;
        };
        match packet.protocol {
            17 => {
                let Some((_, destination, data)) = ports(packet.payload, 8) else {
                    return;
                };
                if destination == 53 && data.len() >= 12 && data[2] & 0x80 == 0 {
                    let apple = question(data).is_some_and(|(name, _)| is_apple(&name));
                    self.lookups.asked += 1;
                    self.apple_lookups.asked += u64::from(apple);
                    self.changed = true;
                } else if destination == 123 && data.len() >= 48 && data[0] & 7 == 3 {
                    self.guest_clock = ntp_seconds(&data[40..48]);
                    self.changed = true;
                }
            }
            6 => {
                let Some((_, destination, data, flags)) = segment(packet.payload) else {
                    return;
                };
                if destination != 443 {
                    return;
                }
                if hello(data, 1) && server_name(data).is_some_and(is_apple) {
                    self.remember(packet.destination);
                }
                let apple = self.is_apple_address(&packet.destination);
                let attempt = flags & 0x12 == 0x02;
                let hello = hello(data, 1);
                for (counts, counted) in [
                    (&mut self.connections, true),
                    (&mut self.apple_connections, apple),
                ] {
                    if counted {
                        counts.attempts += u64::from(attempt);
                        counts.hello += u64::from(hello);
                    }
                }
                if let Some(code) = alert(data) {
                    count(&mut self.guest_alerts, code);
                }
                self.changed |= attempt || hello || alert(data).is_some();
            }
            _ => {}
        }
    }

    /// A frame from the network for the guest.
    pub fn inbound(&mut self, frame: &[u8]) {
        let Some(packet) = packet(frame) else {
            return;
        };
        match packet.protocol {
            17 => {
                let Some((source, _, data)) = ports(packet.payload, 8) else {
                    return;
                };
                if source == 53 && data.len() >= 12 && data[2] & 0x80 != 0 {
                    self.reply(data);
                } else if source == 123 && data.len() >= 48 && data[0] & 7 == 4 {
                    self.server_clock = ntp_seconds(&data[40..48]);
                    self.changed = true;
                }
            }
            6 => {
                let Some((source, _, data, flags)) = segment(packet.payload) else {
                    return;
                };
                if source != 443 {
                    return;
                }
                let apple = self.is_apple_address(&packet.source);
                let accepted = flags & 0x12 == 0x12;
                let reset = flags & 0x04 != 0;
                let hello = hello(data, 2);
                for (counts, counted) in [
                    (&mut self.connections, true),
                    (&mut self.apple_connections, apple),
                ] {
                    if counted {
                        counts.accepted += u64::from(accepted);
                        counts.reset += u64::from(reset);
                        counts.server_hello += u64::from(hello);
                    }
                }
                if let Some(code) = alert(data) {
                    count(&mut self.server_alerts, code);
                }
                self.changed |= accepted || reset || hello || alert(data).is_some();
            }
            _ => {}
        }
    }

    /// The line of facts when one changed since the last line, else none.
    pub fn line(&mut self) -> Option<String> {
        if !core::mem::take(&mut self.changed) {
            return None;
        }
        let lookups = |l: Lookups| {
            format!(
                "asked={} answered={} empty={} failed={}",
                l.asked, l.answered, l.empty, l.failed
            )
        };
        let connections = |c: Connections| {
            format!(
                "attempts={} accepted={} reset={} tls_hello={} server_hello={}",
                c.attempts, c.accepted, c.reset, c.hello, c.server_hello
            )
        };
        let alerts = |alerts: &[(u8, u64)]| match alerts.is_empty() {
            true => String::from("none"),
            false => alerts
                .iter()
                .map(|(code, n)| format!("{code}x{n}"))
                .collect::<Vec<_>>()
                .join(","),
        };
        let clock = |seconds: Option<u64>| match seconds {
            Some(seconds) => format!("{seconds}"),
            None => String::from("unseen"),
        };
        Some(format!(
            "dns {} apple_dns {} https {} apple_https {} tls_alerts server={} guest={} \
             time_request_clock guest={} server={}",
            lookups(self.lookups),
            lookups(self.apple_lookups),
            connections(self.connections),
            connections(self.apple_connections),
            alerts(&self.server_alerts),
            alerts(&self.guest_alerts),
            clock(self.guest_clock),
            clock(self.server_clock),
        ))
    }

    /// A lookup's reply: answered, empty or failed, and the addresses an
    /// Apple name's reply gives.
    fn reply(&mut self, data: &[u8]) {
        let code = data[3] & 0x0f;
        let answers = usize::from(u16::from_be_bytes([data[6], data[7]]));
        let question = question(data);
        let apple = question.as_ref().is_some_and(|(name, _)| is_apple(name));
        for (lookups, counted) in [(&mut self.lookups, true), (&mut self.apple_lookups, apple)] {
            if counted {
                match (code, answers) {
                    (0, 0) => lookups.empty += 1,
                    (0, _) => lookups.answered += 1,
                    _ => lookups.failed += 1,
                }
            }
        }
        self.changed = true;
        if let (true, Some((_, mut at))) = (apple && code == 0, question) {
            for _ in 0..answers.min(ANSWERS) {
                let Some(after) = skip_name(data, at) else {
                    return;
                };
                let Some(fixed) = data.get(after..after + 10) else {
                    return;
                };
                let kind = u16::from_be_bytes([fixed[0], fixed[1]]);
                let length = usize::from(u16::from_be_bytes([fixed[8], fixed[9]]));
                let Some(rdata) = data.get(after + 10..after + 10 + length) else {
                    return;
                };
                match (kind, length) {
                    (1, 4) => self.remember(v4_mapped(rdata)),
                    (28, 16) => self.remember(rdata.try_into().expect("16 bytes")),
                    _ => {}
                }
                at = after + 10 + length;
            }
        }
    }

    fn remember(&mut self, address: [u8; 16]) {
        if self.is_apple_address(&address) {
            return;
        }
        if self.apple.len() < ADDRESSES {
            self.apple.push(address);
        } else {
            self.apple[self.next_address] = address;
            self.next_address = (self.next_address + 1) % ADDRESSES;
        }
    }

    fn is_apple_address(&self, address: &[u8; 16]) -> bool {
        self.apple.contains(address)
    }
}

fn count(alerts: &mut Vec<(u8, u64)>, code: u8) {
    if let Some((_, n)) = alerts.iter_mut().find(|(seen, _)| *seen == code) {
        *n += 1;
    } else if alerts.len() < ALERTS {
        alerts.push((code, 1));
    }
}

/// An IPv4 address as an IPv4-mapped IPv6 one.
fn v4_mapped(address: &[u8]) -> [u8; 16] {
    let mut mapped = [0; 16];
    mapped[10..12].fill(0xff);
    mapped[12..].copy_from_slice(address);
    mapped
}

/// The IP packet in an Ethernet frame (one VLAN tag allowed): its
/// addresses, protocol and payload. Fragments after the first are none.
fn packet(frame: &[u8]) -> Option<Packet<'_>> {
    let (kind, ip) = match frame.get(12..14)? {
        [0x81, 0x00] => (frame.get(16..18)?, frame.get(18..)?),
        kind => (kind, frame.get(14..)?),
    };
    match kind {
        [0x08, 0x00] => {
            let header = usize::from(ip.first()? & 0x0f) * 4;
            let total = usize::from(u16::from_be_bytes([*ip.get(2)?, *ip.get(3)?]));
            let fragment = u16::from_be_bytes([*ip.get(6)?, *ip.get(7)?]) & 0x1fff;
            if ip[0] >> 4 != 4 || header < 20 || fragment != 0 || total < header {
                return None;
            }
            Some(Packet {
                source: v4_mapped(ip.get(12..16)?),
                destination: v4_mapped(ip.get(16..20)?),
                protocol: ip[9],
                payload: ip.get(header..total.min(ip.len()))?,
            })
        }
        [0x86, 0xdd] => {
            let length = usize::from(u16::from_be_bytes([*ip.get(4)?, *ip.get(5)?]));
            Some(Packet {
                source: ip.get(8..24)?.try_into().ok()?,
                destination: ip.get(24..40)?.try_into().ok()?,
                protocol: *ip.get(6)?,
                payload: ip.get(40..(40 + length).min(ip.len()))?,
            })
        }
        _ => None,
    }
}

/// A UDP datagram's ports and data (`header` bytes in).
fn ports(payload: &[u8], header: usize) -> Option<(u16, u16, &[u8])> {
    let source = u16::from_be_bytes([*payload.first()?, *payload.get(1)?]);
    let destination = u16::from_be_bytes([*payload.get(2)?, *payload.get(3)?]);
    Some((source, destination, payload.get(header..)?))
}

/// A TCP segment's ports, data and flags.
fn segment(payload: &[u8]) -> Option<(u16, u16, &[u8], u8)> {
    let header = usize::from(payload.get(12)? >> 4) * 4;
    let flags = *payload.get(13)?;
    let (source, destination, data) = ports(payload, header.max(20))?;
    Some((source, destination, data, flags))
}

/// Whether `data` starts a TLS handshake record holding message `kind`
/// (1, client hello; 2, server hello).
fn hello(data: &[u8], kind: u8) -> bool {
    data.len() > 5 && data[0] == 0x16 && data[1] == 3 && data[5] == kind
}

/// A plaintext TLS alert record's description code.
fn alert(data: &[u8]) -> Option<u8> {
    (data.len() >= 7 && data[0] == 0x15 && data[1] == 3 && data[3..5] == [0, 2]).then(|| data[6])
}

/// A client hello's server name (its `server_name` extension).
fn server_name(data: &[u8]) -> Option<&[u8]> {
    // Record (5), handshake type and length (4), version (2), random (32).
    let mut at = 5 + 4 + 2 + 32;
    at += 1 + usize::from(*data.get(at)?); // Session ID.
    at += 2 + usize::from(u16::from_be_bytes([*data.get(at)?, *data.get(at + 1)?])); // Suites.
    at += 1 + usize::from(*data.get(at)?); // Compression methods.
    let end = at + 2 + usize::from(u16::from_be_bytes([*data.get(at)?, *data.get(at + 1)?]));
    at += 2;
    while at + 4 <= end.min(data.len()) {
        let kind = u16::from_be_bytes([data[at], data[at + 1]]);
        let length = usize::from(u16::from_be_bytes([data[at + 2], data[at + 3]]));
        let body = data.get(at + 4..at + 4 + length)?;
        if kind == 0 {
            // List length (2), name type (1), name length (2), name.
            let name_length = usize::from(u16::from_be_bytes([*body.get(3)?, *body.get(4)?]));
            return body.get(5..5 + name_length);
        }
        at += 4 + length;
    }
    None
}

/// A lookup's first question name, lowercase and dot-separated, with the
/// offset after the question.
fn question(data: &[u8]) -> Option<(Vec<u8>, usize)> {
    if u16::from_be_bytes([data[4], data[5]]) == 0 {
        return None;
    }
    let mut name = Vec::new();
    let mut at = 12;
    loop {
        let length = usize::from(*data.get(at)?);
        if length == 0 {
            break;
        }
        if length > 63 || name.len() + length > 253 {
            return None;
        }
        if !name.is_empty() {
            name.push(b'.');
        }
        name.extend(
            data.get(at + 1..at + 1 + length)?
                .iter()
                .map(u8::to_ascii_lowercase),
        );
        at += 1 + length;
    }
    // The root label, type and class.
    Some((name, at + 1 + 4))
}

/// The offset after a record name at `at`: labels, ending in the root or
/// a compression pointer.
fn skip_name(data: &[u8], mut at: usize) -> Option<usize> {
    for _ in 0..128 {
        let length = *data.get(at)?;
        match length {
            0 => return Some(at + 1),
            0xc0..=0xff => return Some(at + 2),
            1..=63 => at += 1 + usize::from(length),
            _ => return None,
        }
    }
    None
}

/// Whether `name` is one of Apple's service domains or under one.
fn is_apple(name: &[u8]) -> bool {
    let name = name.strip_suffix(b".").unwrap_or(name);
    APPLE_DOMAINS.iter().any(|domain| {
        name.len() >= domain.len()
            && name[name.len() - domain.len()..].eq_ignore_ascii_case(domain)
            && (name.len() == domain.len() || name[name.len() - domain.len() - 1] == b'.')
    })
}

/// An NTP timestamp's seconds as Unix seconds, none for a zero one. A
/// value below 2^31 is in the era after 2036, as RFC 4330 reads it.
fn ntp_seconds(timestamp: &[u8]) -> Option<u64> {
    let seconds = u64::from(u32::from_be_bytes(timestamp.get(..4)?.try_into().ok()?));
    let seconds = match seconds {
        0 => return None,
        1..0x8000_0000 => seconds + (1 << 32),
        _ => seconds,
    };
    Some(seconds - NTP_UNIX)
}

#[cfg(test)]
mod tests;
