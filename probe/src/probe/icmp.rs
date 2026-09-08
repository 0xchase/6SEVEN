use pnet::packet::MutablePacket;
use pnet::packet::Packet;
use pnet::packet::{
    icmpv6::echo_reply::EchoReplyPacket as Icmpv6EchoReplyPacket,
    icmpv6::echo_request::EchoRequestPacket,
    icmpv6::echo_request::MutableEchoRequestPacket as MutableIcmpv6EchoRequestPacket,
    icmpv6::{Icmpv6Packet, Icmpv6Types},
    ip::{IpNextHeaderProtocol, IpNextHeaderProtocols},
};
use serde::{Deserialize, Serialize};

use crate::{
    Decode, Probe, RawReply, Reply, ReplyKind, Request, Session, decode_token_u16, encode_token_u16,
};

const VALIDATION_PAYLOAD_LEN: usize = 4;
const PACKET_LEN: usize = 8 + VALIDATION_PAYLOAD_LEN;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct IcmpConfig;

#[derive(Debug, Default, Clone)]
pub struct IcmpProbe;

impl From<IcmpConfig> for IcmpProbe {
    fn from(_: IcmpConfig) -> Self {
        Self
    }
}

impl Probe for IcmpProbe {
    fn max_packet_len(&self) -> usize {
        PACKET_LEN
    }

    fn packet(
        &self,
        buf: &mut [u8],
        session: &Session,
        request: Request,
    ) -> Option<(IpNextHeaderProtocol, usize)> {
        let mut icmp = MutableIcmpv6EchoRequestPacket::new(buf.get_mut(..PACKET_LEN)?)?;
        let v = session.validate(request.target, request.token);
        let token_bits = encode_token_u16(request.token)?;

        icmp.set_icmpv6_type(Icmpv6Types::EchoRequest);
        icmp.set_identifier(token_bits | (v[1] as u16 & 0x1FFF));
        icmp.set_sequence_number((v[2] & 0xFFFF) as u16);
        icmp.payload_mut()[..VALIDATION_PAYLOAD_LEN].copy_from_slice(&v[0].to_be_bytes());
        icmp.set_checksum(0);

        let checksum = pnet::packet::icmpv6::checksum(
            &Icmpv6Packet::new(icmp.packet())?,
            &session.source,
            &request.target,
        );
        icmp.set_checksum(checksum);
        Some((IpNextHeaderProtocols::Icmpv6, PACKET_LEN))
    }

    fn decode(&self, session: &Session, reply: RawReply<'_>) -> Decode {
        if reply.protocol != IpNextHeaderProtocols::Icmpv6 {
            return Decode::NotMine;
        }

        let Some(icmp) = Icmpv6Packet::new(reply.payload) else {
            return Decode::Malformed;
        };
        let icmp_type = icmp.get_icmpv6_type().0;
        let icmp_code = icmp.get_icmpv6_code().0;

        if icmp.get_icmpv6_type() == Icmpv6Types::EchoReply {
            let Some(echo) = Icmpv6EchoReplyPacket::new(reply.payload) else {
                return Decode::Malformed;
            };
            let echo_id = echo.get_identifier();
            let token = decode_token_u16(echo_id);
            let v = session.validate(reply.responder, token);
            let expected_id = encode_token_u16(token).unwrap_or_default() | (v[1] as u16 & 0x1FFF);
            let expected_seq = (v[2] & 0xFFFF) as u16;
            let expected_payload = v[0].to_be_bytes();

            if echo_id != expected_id || echo.get_sequence_number() != expected_seq {
                return Decode::NotMine;
            }

            let payload = echo.payload();
            if payload.len() >= 4 && payload[..4] != expected_payload {
                return Decode::NotMine;
            }

            return Decode::Reply(Reply {
                target: reply.responder,
                responder: reply.responder,
                token,
                kind: ReplyKind::Direct,
            });
        }

        let Some(inner) = link::icmpv6_error_transport(reply.payload, session.source) else {
            return Decode::NotMine;
        };
        if inner.protocol != IpNextHeaderProtocols::Icmpv6 {
            return Decode::NotMine;
        }
        let Some(inner_echo) = EchoRequestPacket::new(inner.payload) else {
            return Decode::Malformed;
        };
        if Icmpv6Packet::new(inner.payload)
            .is_none_or(|pkt| pkt.get_icmpv6_type() != Icmpv6Types::EchoRequest)
        {
            return Decode::NotMine;
        }

        let inner_id = inner_echo.get_identifier();
        let token = decode_token_u16(inner_id);
        let v = session.validate(inner.target, token);
        let expected_id = encode_token_u16(token).unwrap_or_default() | (v[1] as u16 & 0x1FFF);
        let expected_seq = (v[2] & 0xFFFF) as u16;
        if inner_id != expected_id || inner_echo.get_sequence_number() != expected_seq {
            return Decode::NotMine;
        }

        Decode::Reply(Reply {
            target: inner.target,
            responder: reply.responder,
            token,
            kind: ReplyKind::IcmpError {
                icmp_type,
                icmp_code,
            },
        })
    }
}
