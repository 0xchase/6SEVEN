use pnet::packet::MutablePacket;
use pnet::packet::Packet;
use pnet::packet::ip::{IpNextHeaderProtocol, IpNextHeaderProtocols};
use pnet::packet::udp::MutableUdpPacket;
use serde::{Deserialize, Serialize};

use crate::{
    Decode, Probe, RawReply, Reply, ReplyKind, Request, Session, decode_token_u16, encode_token_u16,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum DnsQueryType {
    #[default]
    A,
    Aaaa,
    Mx,
    Txt,
    Ns,
    Soa,
}

impl DnsQueryType {
    fn qtype(self) -> u16 {
        match self {
            Self::A => 1,
            Self::Aaaa => 28,
            Self::Mx => 15,
            Self::Txt => 16,
            Self::Ns => 2,
            Self::Soa => 6,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnsConfig {
    pub port: u16,
    pub domain: String,
    pub query_type: DnsQueryType,
}

impl Default for DnsConfig {
    fn default() -> Self {
        Self {
            port: 53,
            domain: "www.google.com".to_string(),
            query_type: DnsQueryType::A,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DnsProbe {
    port: u16,
    query: Box<[u8]>,
}

impl From<DnsConfig> for DnsProbe {
    fn from(cfg: DnsConfig) -> Self {
        Self {
            port: cfg.port,
            query: build_dns_query(&cfg.domain, cfg.query_type).into_boxed_slice(),
        }
    }
}

impl Probe for DnsProbe {
    fn max_packet_len(&self) -> usize {
        8usize.saturating_add(self.query.len())
    }

    fn packet(
        &self,
        buf: &mut [u8],
        session: &Session,
        request: Request,
    ) -> Option<(IpNextHeaderProtocol, usize)> {
        let total_len = self.max_packet_len();
        let udp_len = u16::try_from(total_len).ok()?;
        let buf = buf.get_mut(..total_len)?;

        let v = session.validate(request.target, request.token);
        let token_bits = encode_token_u16(request.token)?;
        let txid = token_bits | (v[2] as u16 & 0x1FFF);
        let source_port = token_bits | (v[0] as u16 & 0x1FFF);

        let mut udp = MutableUdpPacket::new(buf)?;
        udp.set_source(source_port);
        udp.set_destination(self.port);
        udp.set_length(udp_len);
        udp.set_checksum(0);

        let payload = udp.payload_mut();
        payload[..self.query.len()].copy_from_slice(&self.query);
        payload[0..2].copy_from_slice(&txid.to_be_bytes());

        let checksum =
            pnet::packet::udp::ipv6_checksum(&udp.to_immutable(), &session.source, &request.target);
        udp.set_checksum(checksum);

        Some((IpNextHeaderProtocols::Udp, total_len))
    }

    fn decode(&self, session: &Session, reply: RawReply<'_>) -> Decode {
        if reply.protocol == IpNextHeaderProtocols::Icmpv6 {
            let Some(inner) = link::icmpv6_error_transport(reply.payload, session.source) else {
                return Decode::NotMine;
            };
            if inner.protocol != IpNextHeaderProtocols::Udp {
                return Decode::NotMine;
            }
            let Some(udp) = pnet::packet::udp::UdpPacket::new(inner.payload) else {
                return Decode::Malformed;
            };
            let dns = udp.payload();
            if dns.len() < 2 {
                return Decode::Malformed;
            }

            let txid = u16::from_be_bytes([dns[0], dns[1]]);
            let token = decode_token_u16(txid);
            let v = session.validate(inner.target, token);
            let expected_txid =
                encode_token_u16(token).unwrap_or_default() | (v[2] as u16 & 0x1FFF);
            let expected_source =
                encode_token_u16(token).unwrap_or_default() | (v[0] as u16 & 0x1FFF);
            if udp.get_source() != expected_source || udp.get_destination() != self.port {
                return Decode::NotMine;
            }
            if txid != expected_txid {
                return Decode::NotMine;
            }

            return Decode::Reply(Reply {
                target: inner.target,
                responder: reply.responder,
                token,
                kind: ReplyKind::IcmpError {
                    icmp_type: reply.payload[0],
                    icmp_code: reply.payload[1],
                },
            });
        }

        if reply.protocol != IpNextHeaderProtocols::Udp {
            return Decode::NotMine;
        }
        let Some(udp) = pnet::packet::udp::UdpPacket::new(reply.payload) else {
            return Decode::Malformed;
        };
        let dns = udp.payload();
        if dns.len() < 2 {
            return Decode::Malformed;
        }

        let txid = u16::from_be_bytes([dns[0], dns[1]]);
        let token = decode_token_u16(txid);
        let v = session.validate(reply.responder, token);
        let expected_txid = encode_token_u16(token).unwrap_or_default() | (v[2] as u16 & 0x1FFF);
        let expected_destination =
            encode_token_u16(token).unwrap_or_default() | (v[0] as u16 & 0x1FFF);
        if udp.get_destination() != expected_destination || txid != expected_txid {
            return Decode::NotMine;
        }

        Decode::Reply(Reply {
            target: reply.responder,
            responder: reply.responder,
            token,
            kind: ReplyKind::Direct,
        })
    }
}

fn build_dns_query(domain: &str, query_type: DnsQueryType) -> Vec<u8> {
    let mut packet = Vec::new();
    packet.extend(&[0x00, 0x00]);
    packet.extend(&[0x01, 0x00]);
    packet.extend(&[0x00, 0x01]);
    packet.extend(&[0x00, 0x00]);
    packet.extend(&[0x00, 0x00]);
    packet.extend(&[0x00, 0x00]);

    for label in domain.split('.') {
        packet.push(label.len() as u8);
        packet.extend(label.as_bytes());
    }
    packet.push(0);
    packet.extend(&query_type.qtype().to_be_bytes());
    packet.extend(&[0x00, 0x01]);
    packet
}
