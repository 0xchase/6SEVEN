use std::net::Ipv6Addr;

use aes::Aes128;
use aes::cipher::{BlockEncrypt, KeyInit};
use pnet::packet::ip::IpNextHeaderProtocol;

pub mod config;
pub mod probe;

pub use config::{ProbeConfig, ProbeError};
use probe::dns::DnsProbe;
use probe::icmp::IcmpProbe;
use probe::ntp::NtpProbe;
use probe::tcp::TcpProbe;
use probe::udp::UdpProbe;

#[derive(Clone)]
struct Session {
    source: Ipv6Addr,
    key: u128,
    cipher: Aes128,
}

impl Session {
    pub fn new(source: Ipv6Addr, key: u128) -> Self {
        Self {
            source,
            key,
            cipher: validation_cipher(key),
        }
    }

    pub fn validate(&self, dst: Ipv6Addr, token: u8) -> [u32; 4] {
        validate_with(&self.cipher, self.source, dst, token)
    }
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("source", &self.source)
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}

impl PartialEq for Session {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source && self.key == other.key
    }
}

impl Eq for Session {}

fn validation_cipher(key: u128) -> Aes128 {
    Aes128::new((&key.to_be_bytes()).into())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Request {
    pub target: Ipv6Addr,
    pub token: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodedPacket {
    pub protocol: IpNextHeaderProtocol,
    pub payload_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawReply<'a> {
    pub responder: Ipv6Addr,
    pub protocol: IpNextHeaderProtocol,
    pub payload: &'a [u8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ReplyKind {
    Direct,
    IcmpError { icmp_type: u8, icmp_code: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reply {
    pub target: Ipv6Addr,
    pub responder: Ipv6Addr,
    pub token: u8,
    pub kind: ReplyKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decode {
    Reply(Reply),
    NotMine,
    Malformed,
}

#[derive(Debug, Clone)]
enum CompiledProbe {
    Icmp(IcmpProbe),
    Tcp(TcpProbe),
    Udp(UdpProbe),
    Dns(DnsProbe),
    Ntp(NtpProbe),
}

impl CompiledProbe {
    fn max_packet_len(&self) -> usize {
        match self {
            Self::Icmp(probe) => probe.max_packet_len(),
            Self::Tcp(probe) => probe.max_packet_len(),
            Self::Udp(probe) => probe.max_packet_len(),
            Self::Dns(probe) => probe.max_packet_len(),
            Self::Ntp(probe) => probe.max_packet_len(),
        }
    }

    fn bind(self, source: Ipv6Addr) -> PreparedProbe {
        self.bind_with_key(source, rand::random::<u128>())
    }

    fn bind_with_key(self, source: Ipv6Addr, key: u128) -> PreparedProbe {
        PreparedProbe {
            probe: self,
            session: Session::new(source, key),
        }
    }

    fn packet(
        &self,
        buf: &mut [u8],
        session: &Session,
        request: Request,
    ) -> Option<(IpNextHeaderProtocol, usize)> {
        match self {
            Self::Icmp(probe) => probe.packet(buf, session, request),
            Self::Tcp(probe) => probe.packet(buf, session, request),
            Self::Udp(probe) => probe.packet(buf, session, request),
            Self::Dns(probe) => probe.packet(buf, session, request),
            Self::Ntp(probe) => probe.packet(buf, session, request),
        }
    }

    fn decode(&self, session: &Session, reply: RawReply<'_>) -> Decode {
        match self {
            Self::Icmp(probe) => probe.decode(session, reply),
            Self::Tcp(probe) => probe.decode(session, reply),
            Self::Udp(probe) => probe.decode(session, reply),
            Self::Dns(probe) => probe.decode(session, reply),
            Self::Ntp(probe) => probe.decode(session, reply),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PreparedProbe {
    probe: CompiledProbe,
    session: Session,
}

impl PreparedProbe {
    pub fn source_addr(&self) -> Ipv6Addr {
        self.session.source
    }

    pub fn max_packet_len(&self) -> usize {
        self.probe.max_packet_len()
    }

    pub fn encode(&self, buf: &mut [u8], request: Request) -> Option<EncodedPacket> {
        let (protocol, payload_len) = self.probe.packet(buf, &self.session, request)?;
        Some(EncodedPacket {
            protocol,
            payload_len,
        })
    }

    pub fn decode(&self, reply: RawReply<'_>) -> Decode {
        self.probe.decode(&self.session, reply)
    }
}

/// Keyed PRF modeled after ZMap v6:
/// AES-128(key, src ^ dst), which is direction-invariant.
/// `token` is folded in so callers can correlate probe fan-outs.
pub fn validate(key: u128, src: Ipv6Addr, dst: Ipv6Addr, token: u8) -> [u32; 4] {
    validate_with(&validation_cipher(key), src, dst, token)
}

fn validate_with(cipher: &Aes128, src: Ipv6Addr, dst: Ipv6Addr, token: u8) -> [u32; 4] {
    let src_octets = src.octets();
    let dst_octets = dst.octets();
    let mut block = [0u8; 16];
    for i in 0..16 {
        block[i] = src_octets[i] ^ dst_octets[i];
    }
    block[0] ^= token;

    let mut ga = aes::cipher::generic_array::GenericArray::clone_from_slice(&block);
    cipher.encrypt_block(&mut ga);

    [
        u32::from_be_bytes([ga[0], ga[1], ga[2], ga[3]]),
        u32::from_be_bytes([ga[4], ga[5], ga[6], ga[7]]),
        u32::from_be_bytes([ga[8], ga[9], ga[10], ga[11]]),
        u32::from_be_bytes([ga[12], ga[13], ga[14], ga[15]]),
    ]
}

pub(crate) fn encode_token_u16(token: u8) -> Option<u16> {
    (token <= 0x07).then_some((token as u16) << 13)
}

pub(crate) fn decode_token_u16(bits: u16) -> u8 {
    ((bits >> 13) & 0x07) as u8
}

pub(crate) fn encode_token_u32(token: u8) -> Option<u32> {
    (token <= 0x07).then_some((token as u32) << 29)
}

pub(crate) fn decode_token_u32(bits: u32) -> u8 {
    ((bits >> 29) & 0x07) as u8
}

/// Walk IPv6 extension headers to find the actual transport protocol and payload.
pub(crate) trait Probe: Clone + Send + 'static {
    fn max_packet_len(&self) -> usize;

    fn packet(
        &self,
        buf: &mut [u8],
        session: &Session,
        request: Request,
    ) -> Option<(IpNextHeaderProtocol, usize)>;

    fn decode(&self, session: &Session, reply: RawReply<'_>) -> Decode;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::dns::{DnsConfig, DnsQueryType};
    use crate::probe::icmp::IcmpConfig;
    use crate::probe::ntp::NtpConfig;
    use crate::probe::tcp::TcpConfig;
    use crate::probe::udp::UdpConfig;

    #[test]
    fn max_packet_len_matches_encoder_contract() {
        for probe in sample_probes() {
            let packet_len = probe.max_packet_len();
            let prepared =
                probe.bind_with_key("2001:db8::1".parse().unwrap(), 0x1234_5678_90ab_cdef);
            let request = Request {
                target: "2001:db8::2".parse().unwrap(),
                token: 0,
            };

            let mut exact = vec![0u8; packet_len];
            let encoded = prepared.encode(&mut exact, request).unwrap();
            assert_eq!(encoded.payload_len, packet_len);

            let mut short = vec![0u8; packet_len - 1];
            assert_eq!(prepared.encode(&mut short, request), None);
        }
    }

    fn sample_probes() -> Vec<CompiledProbe> {
        vec![
            ProbeConfig::Icmp(IcmpConfig).compile().unwrap(),
            ProbeConfig::Tcp(TcpConfig { port: 443 }).compile().unwrap(),
            ProbeConfig::Udp(UdpConfig {
                port: 53,
                payload: b"payload".to_vec(),
            })
            .compile()
            .unwrap(),
            ProbeConfig::Dns(DnsConfig {
                port: 53,
                domain: "www.example.com".to_string(),
                query_type: DnsQueryType::Aaaa,
            })
            .compile()
            .unwrap(),
            ProbeConfig::Ntp(NtpConfig { port: 123 }).compile().unwrap(),
        ]
    }
}
