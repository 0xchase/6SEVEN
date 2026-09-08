use etherparse::TcpHeader;
use pnet::packet::MutablePacket;
use pnet::packet::ip::{IpNextHeaderProtocol, IpNextHeaderProtocols};
use pnet::packet::tcp::MutableTcpPacket;
use serde::{Deserialize, Serialize};

use crate::{
    Decode, Probe, RawReply, Reply, ReplyKind, Request, Session, decode_token_u32, encode_token_u32,
};

const PACKET_LEN: usize = 28;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TcpConfig {
    pub port: u16,
}

impl Default for TcpConfig {
    fn default() -> Self {
        Self { port: 80 }
    }
}

#[derive(Debug, Clone)]
pub struct TcpProbe {
    port: u16,
}

impl From<TcpConfig> for TcpProbe {
    fn from(cfg: TcpConfig) -> Self {
        Self { port: cfg.port }
    }
}

impl Probe for TcpProbe {
    fn max_packet_len(&self) -> usize {
        PACKET_LEN
    }

    fn packet(
        &self,
        buf: &mut [u8],
        session: &Session,
        request: Request,
    ) -> Option<(IpNextHeaderProtocol, usize)> {
        let v = session.validate(request.target, request.token);
        let token_bits = encode_token_u32(request.token)?;
        let source_port = 0x8000 | (v[1] as u16 & 0x7FFF);
        let sequence = token_bits | (v[0] & 0x1FFF_FFFF);

        let mut tcp = MutableTcpPacket::new(buf.get_mut(..PACKET_LEN)?)?;
        tcp.set_source(source_port);
        tcp.set_destination(self.port);
        tcp.set_flags(0x02);
        tcp.set_window(65535);
        tcp.set_data_offset((PACKET_LEN / 4) as u8);
        tcp.set_sequence(sequence);
        tcp.set_checksum(0);
        tcp.packet_mut()[20..28].copy_from_slice(&[0x02, 0x04, 0x05, 0xB4, 0x04, 0x02, 0x01, 0x01]);

        let checksum =
            pnet::packet::tcp::ipv6_checksum(&tcp.to_immutable(), &session.source, &request.target);
        tcp.set_checksum(checksum);

        Some((IpNextHeaderProtocols::Tcp, PACKET_LEN))
    }

    fn decode(&self, session: &Session, reply: RawReply<'_>) -> Decode {
        if reply.protocol == IpNextHeaderProtocols::Icmpv6 {
            let Some(inner) = link::icmpv6_error_transport(reply.payload, session.source) else {
                return Decode::NotMine;
            };
            if inner.protocol != IpNextHeaderProtocols::Tcp {
                return Decode::NotMine;
            }
            let Ok((header, _)) = TcpHeader::from_slice(inner.payload) else {
                return Decode::Malformed;
            };
            let token = decode_token_u32(header.sequence_number);
            let v = session.validate(inner.target, token);
            let expected_source = 0x8000 | (v[1] as u16 & 0x7FFF);
            let expected_sequence =
                encode_token_u32(token).unwrap_or_default() | (v[0] & 0x1FFF_FFFF);
            if header.source_port != expected_source
                || header.destination_port != self.port
                || header.sequence_number != expected_sequence
            {
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

        if reply.protocol != IpNextHeaderProtocols::Tcp {
            return Decode::NotMine;
        }
        let Ok((header, _)) = TcpHeader::from_slice(reply.payload) else {
            return Decode::Malformed;
        };
        if header.source_port != self.port {
            return Decode::NotMine;
        }

        let ack_for_token = if header.rst {
            header.acknowledgment_number
        } else if header.syn && header.ack {
            header.acknowledgment_number.wrapping_sub(1)
        } else {
            return Decode::NotMine;
        };
        let token = decode_token_u32(ack_for_token);
        let v = session.validate(reply.responder, token);
        let expected_destination = 0x8000 | (v[1] as u16 & 0x7FFF);
        if header.destination_port != expected_destination {
            return Decode::NotMine;
        }

        let expected_sequence = encode_token_u32(token).unwrap_or_default() | (v[0] & 0x1FFF_FFFF);
        let expected_ack = expected_sequence.wrapping_add(1);
        let valid = if header.rst {
            header.acknowledgment_number == expected_sequence
                || header.acknowledgment_number == expected_ack
        } else {
            header.syn && header.ack && header.acknowledgment_number == expected_ack
        };
        if !valid {
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
