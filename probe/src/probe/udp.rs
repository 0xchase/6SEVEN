use pnet::packet::MutablePacket;
use pnet::packet::ip::{IpNextHeaderProtocol, IpNextHeaderProtocols};
use pnet::packet::udp::MutableUdpPacket;
use serde::{Deserialize, Serialize};

use crate::{
    Decode, Probe, RawReply, Reply, ReplyKind, Request, Session, decode_token_u16, encode_token_u16,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UdpConfig {
    pub port: u16,
    pub payload: Vec<u8>,
}

impl Default for UdpConfig {
    fn default() -> Self {
        Self {
            port: 53,
            payload: b"GET / HTTP/1.1\r\nHost: www\r\n\r\n".to_vec(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UdpProbe {
    port: u16,
    payload: Box<[u8]>,
}

impl From<UdpConfig> for UdpProbe {
    fn from(cfg: UdpConfig) -> Self {
        Self {
            port: cfg.port,
            payload: cfg.payload.into_boxed_slice(),
        }
    }
}

impl Probe for UdpProbe {
    fn max_packet_len(&self) -> usize {
        8usize.saturating_add(self.payload.len())
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
        let source_port = token_bits | (v[0] as u16 & 0x1FFF);

        let mut udp = MutableUdpPacket::new(buf)?;
        udp.set_source(source_port);
        udp.set_destination(self.port);
        udp.set_length(udp_len);
        udp.set_checksum(0);
        udp.payload_mut()[..self.payload.len()].copy_from_slice(&self.payload);

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
            let Ok((header, _)) = etherparse::UdpHeader::from_slice(inner.payload) else {
                return Decode::Malformed;
            };
            let token = decode_token_u16(header.source_port);
            let v = session.validate(inner.target, token);
            let expected_source =
                encode_token_u16(token).unwrap_or_default() | (v[0] as u16 & 0x1FFF);
            if header.source_port != expected_source || header.destination_port != self.port {
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
        let Ok((header, _)) = etherparse::UdpHeader::from_slice(reply.payload) else {
            return Decode::Malformed;
        };
        let token = decode_token_u16(header.destination_port);
        let v = session.validate(reply.responder, token);
        let expected_destination =
            encode_token_u16(token).unwrap_or_default() | (v[0] as u16 & 0x1FFF);
        if header.destination_port != expected_destination {
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
