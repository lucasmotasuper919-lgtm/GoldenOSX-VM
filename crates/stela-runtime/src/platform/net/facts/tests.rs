use super::*;
use alloc::vec;

const GUEST: [u8; 4] = [192, 168, 1, 20];
const RESOLVER: [u8; 4] = [192, 168, 1, 1];
const APPLE: [u8; 4] = [17, 253, 144, 10];
const OTHER: [u8; 4] = [93, 184, 216, 34];

fn ipv4(protocol: u8, source: [u8; 4], destination: [u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut frame = vec![0x02, 0, 0, 0, 0, 1, 0x02, 0, 0, 0, 0, 2, 0x08, 0x00];
    let total = (20 + payload.len()) as u16;
    frame.extend_from_slice(&[0x45, 0]);
    frame.extend_from_slice(&total.to_be_bytes());
    frame.extend_from_slice(&[0, 0, 0x40, 0, 64, protocol, 0, 0]);
    frame.extend_from_slice(&source);
    frame.extend_from_slice(&destination);
    frame.extend_from_slice(payload);
    frame
}

fn udp(source: u16, destination: u16, data: &[u8]) -> Vec<u8> {
    let mut datagram = Vec::new();
    datagram.extend_from_slice(&source.to_be_bytes());
    datagram.extend_from_slice(&destination.to_be_bytes());
    datagram.extend_from_slice(&((8 + data.len()) as u16).to_be_bytes());
    datagram.extend_from_slice(&[0, 0]);
    datagram.extend_from_slice(data);
    datagram
}

fn tcp(source: u16, destination: u16, flags: u8, data: &[u8]) -> Vec<u8> {
    let mut segment = Vec::new();
    segment.extend_from_slice(&source.to_be_bytes());
    segment.extend_from_slice(&destination.to_be_bytes());
    segment.extend_from_slice(&[0; 8]);
    segment.extend_from_slice(&[0x50, flags, 0xff, 0xff, 0, 0, 0, 0]);
    segment.extend_from_slice(data);
    segment
}

fn name(name: &str) -> Vec<u8> {
    let mut encoded = Vec::new();
    for label in name.split('.') {
        encoded.push(label.len() as u8);
        encoded.extend_from_slice(label.as_bytes());
    }
    encoded.push(0);
    encoded
}

fn query(host: &str) -> Vec<u8> {
    let mut data = vec![0x12, 0x34, 0x01, 0x00, 0, 1, 0, 0, 0, 0, 0, 0];
    data.extend(name(host));
    data.extend_from_slice(&[0, 1, 0, 1]);
    data
}

fn reply(host: &str, code: u8, addresses: &[[u8; 4]]) -> Vec<u8> {
    let mut data = vec![0x12, 0x34, 0x81, 0x80 | code, 0, 1];
    data.extend_from_slice(&(addresses.len() as u16).to_be_bytes());
    data.extend_from_slice(&[0, 0, 0, 0]);
    data.extend(name(host));
    data.extend_from_slice(&[0, 1, 0, 1]);
    for address in addresses {
        // A pointer to the question's name, then type A, class IN, TTL.
        data.extend_from_slice(&[0xc0, 12, 0, 1, 0, 1, 0, 0, 0, 60, 0, 4]);
        data.extend_from_slice(address);
    }
    data
}

fn client_hello(server: &str) -> Vec<u8> {
    let mut body = vec![3, 3];
    body.extend_from_slice(&[0x55; 32]);
    body.push(0); // Session ID.
    body.extend_from_slice(&[0, 2, 0x13, 0x01]); // One suite.
    body.extend_from_slice(&[1, 0]); // No compression.
    let mut sni = Vec::new();
    sni.extend_from_slice(&((server.len() + 3) as u16).to_be_bytes());
    sni.push(0);
    sni.extend_from_slice(&(server.len() as u16).to_be_bytes());
    sni.extend_from_slice(server.as_bytes());
    let mut extensions = vec![0, 0];
    extensions.extend_from_slice(&(sni.len() as u16).to_be_bytes());
    extensions.extend(sni);
    body.extend_from_slice(&(extensions.len() as u16).to_be_bytes());
    body.extend(extensions);
    let mut handshake = vec![1, 0];
    handshake.extend_from_slice(&(body.len() as u16).to_be_bytes());
    handshake.extend(body);
    let mut record = vec![0x16, 3, 1];
    record.extend_from_slice(&(handshake.len() as u16).to_be_bytes());
    record.extend(handshake);
    record
}

fn ntp(mode: u8, seconds: u32) -> Vec<u8> {
    let mut data = vec![0; 48];
    data[0] = 0x20 | mode;
    data[40..44].copy_from_slice(&seconds.to_be_bytes());
    data
}

/// Lookups and connections are counted for all traffic and, apart, for
/// Apple's services, told by the lookup's name and the addresses its reply
/// gave; the line holds counts only.
#[test]
fn apple_lookups_and_connections_are_counted_apart_without_names_or_addresses() {
    let mut facts = Facts::default();
    assert_eq!(facts.line(), None);
    facts.outbound(&ipv4(
        17,
        GUEST,
        RESOLVER,
        &udp(5353, 53, &query("gsa.apple.com")),
    ));
    facts.inbound(&ipv4(
        17,
        RESOLVER,
        GUEST,
        &udp(53, 5353, &reply("gsa.apple.com", 0, &[APPLE])),
    ));
    facts.outbound(&ipv4(
        17,
        GUEST,
        RESOLVER,
        &udp(5353, 53, &query("example.org")),
    ));
    facts.inbound(&ipv4(
        17,
        RESOLVER,
        GUEST,
        &udp(53, 5353, &reply("example.org", 0, &[OTHER])),
    ));
    facts.outbound(&ipv4(
        17,
        GUEST,
        RESOLVER,
        &udp(5353, 53, &query("idmsa.apple.com")),
    ));
    facts.inbound(&ipv4(
        17,
        RESOLVER,
        GUEST,
        &udp(53, 5353, &reply("idmsa.apple.com", 2, &[])),
    ));
    facts.outbound(&ipv4(
        17,
        GUEST,
        RESOLVER,
        &udp(5353, 53, &query("notapple.com")),
    ));
    facts.inbound(&ipv4(
        17,
        RESOLVER,
        GUEST,
        &udp(53, 5353, &reply("notapple.com", 0, &[])),
    ));
    facts.outbound(&ipv4(6, GUEST, APPLE, &tcp(50000, 443, 0x02, &[])));
    facts.inbound(&ipv4(6, APPLE, GUEST, &tcp(443, 50000, 0x12, &[])));
    facts.inbound(&ipv4(6, APPLE, GUEST, &tcp(443, 50000, 0x14, &[])));
    facts.outbound(&ipv4(6, GUEST, OTHER, &tcp(50001, 443, 0x02, &[])));
    facts.outbound(&ipv4(6, GUEST, OTHER, &tcp(50002, 80, 0x02, &[])));
    let line = facts.line().unwrap();
    assert_eq!(
        line,
        "dns asked=4 answered=2 empty=1 failed=1 apple_dns asked=2 answered=1 empty=0 failed=1 \
         https attempts=2 accepted=1 reset=1 tls_hello=0 server_hello=0 apple_https attempts=1 \
         accepted=1 reset=1 tls_hello=0 server_hello=0 tls_alerts server=none guest=none \
         time_request_clock guest=unseen server=unseen"
    );
    assert!(!line.contains("apple.com") && !line.contains("17.253"));
    assert_eq!(facts.line(), None);
}

/// A TLS client hello naming an Apple service marks its server as Apple's
/// without a lookup, and a plaintext alert's code is counted.
#[test]
fn a_client_hello_names_apple_servers_and_alerts_are_counted() {
    let mut facts = Facts::default();
    facts.outbound(&ipv4(
        6,
        GUEST,
        APPLE,
        &tcp(50000, 443, 0x18, &client_hello("gsa.apple.com")),
    ));
    facts.inbound(&ipv4(
        6,
        APPLE,
        GUEST,
        &tcp(443, 50000, 0x18, &[0x16, 3, 3, 0, 4, 2, 0, 0, 0]),
    ));
    facts.inbound(&ipv4(
        6,
        APPLE,
        GUEST,
        &tcp(443, 50000, 0x18, &[0x15, 3, 3, 0, 2, 2, 42]),
    ));
    facts.outbound(&ipv4(
        6,
        GUEST,
        OTHER,
        &tcp(50001, 443, 0x18, &client_hello("example.org")),
    ));
    facts.outbound(&ipv4(
        6,
        GUEST,
        OTHER,
        &tcp(50001, 443, 0x18, &[0x15, 3, 3, 0, 2, 2, 48]),
    ));
    let line = facts.line().unwrap();
    assert!(
        line.contains("https attempts=0 accepted=0 reset=0 tls_hello=2 server_hello=1 apple_https attempts=0 accepted=0 reset=0 tls_hello=1 server_hello=1"),
        "{line}"
    );
    assert!(line.contains("tls_alerts server=42x1 guest=48x1"), "{line}");
}

/// The guest's clock in its time request and the server's in the reply,
/// as Unix seconds, the era after 2036 included.
#[test]
fn time_requests_show_the_guest_and_server_clocks() {
    let mut facts = Facts::default();
    // 2026-11-17 and, for the guest, 2052 (past the NTP era's end).
    let server = (1_794_900_000 + NTP_UNIX) as u32;
    let guest = (2_600_000_000 + NTP_UNIX - (1 << 32)) as u32;
    facts.outbound(&ipv4(17, GUEST, OTHER, &udp(123, 123, &ntp(3, guest))));
    facts.inbound(&ipv4(17, OTHER, GUEST, &udp(123, 123, &ntp(4, server))));
    let line = facts.line().unwrap();
    assert!(
        line.ends_with("time_request_clock guest=2600000000 server=1794900000"),
        "{line}"
    );
}

/// Truncated and malformed frames are passed over, never read past.
#[test]
fn truncated_frames_are_passed_over() {
    let frames = [
        ipv4(
            17,
            RESOLVER,
            GUEST,
            &udp(53, 5353, &reply("gsa.apple.com", 0, &[APPLE, APPLE])),
        ),
        ipv4(17, GUEST, RESOLVER, &udp(5353, 53, &query("gsa.apple.com"))),
        ipv4(
            6,
            GUEST,
            APPLE,
            &tcp(50000, 443, 0x18, &client_hello("gsa.apple.com")),
        ),
        ipv4(17, OTHER, GUEST, &udp(123, 123, &ntp(4, 1))),
    ];
    let mut facts = Facts::default();
    for frame in &frames {
        for end in 0..frame.len() {
            facts.inbound(&frame[..end]);
            facts.outbound(&frame[..end]);
            let mut corrupt = frame.clone();
            corrupt[end] ^= 0xff;
            facts.inbound(&corrupt);
            facts.outbound(&corrupt);
        }
    }
}
