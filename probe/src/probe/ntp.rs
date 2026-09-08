use pnet::packet::{
    MutablePacket, Packet,
    ip::{IpNextHeaderProtocol, IpNextHeaderProtocols},
    udp::MutableUdpPacket,
};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{
    Decode, Probe, RawReply, Reply, ReplyKind, Request, Session, decode_token_u16, encode_token_u16,
};

const PACKET_LEN: usize = 8 + 48;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NtpConfig {
    pub port: u16,
}

impl Default for NtpConfig {
    fn default() -> Self {
        Self { port: 123 }
    }
}

#[derive(Debug, Clone)]
pub struct NtpProbe {
    port: u16,
    template: [u8; 48],
}

impl From<NtpConfig> for NtpProbe {
    fn from(cfg: NtpConfig) -> Self {
        let mut template = [0u8; 48];
        template[0] = 0x1b;
        template[1] = 0x00;
        template[2] = 0x04;
        template[3] = 0xfa;
        Self {
            port: cfg.port,
            template,
        }
    }
}

impl Probe for NtpProbe {
    fn max_packet_len(&self) -> usize {
        PACKET_LEN
    }

    fn packet(
        &self,
        buf: &mut [u8],
        session: &Session,
        request: Request,
    ) -> Option<(IpNextHeaderProtocol, usize)> {
        let buf = buf.get_mut(..PACKET_LEN)?;

        let v = session.validate(request.target, request.token);
        let token_bits = encode_token_u16(request.token)?;
        let source_port = token_bits | (v[0] as u16 & 0x1FFF);

        let mut udp = MutableUdpPacket::new(buf)?;
        udp.set_source(source_port);
        udp.set_destination(self.port);
        udp.set_length(PACKET_LEN as u16);
        udp.set_checksum(0);
        udp.payload_mut()[..self.template.len()].copy_from_slice(&self.template);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let ntp_timestamp = now + 2_208_988_800;
        udp.payload_mut()[40..44].copy_from_slice(&(ntp_timestamp as u32).to_be_bytes());

        let checksum =
            pnet::packet::udp::ipv6_checksum(&udp.to_immutable(), &session.source, &request.target);
        udp.set_checksum(checksum);

        Some((IpNextHeaderProtocols::Udp, PACKET_LEN))
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
            let token = decode_token_u16(udp.get_source());
            let v = session.validate(inner.target, token);
            let expected_source =
                encode_token_u16(token).unwrap_or_default() | (v[0] as u16 & 0x1FFF);
            if udp.get_source() != expected_source || udp.get_destination() != self.port {
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
        let token = decode_token_u16(udp.get_destination());
        let v = session.validate(reply.responder, token);
        let expected_destination =
            encode_token_u16(token).unwrap_or_default() | (v[0] as u16 & 0x1FFF);
        if udp.get_destination() != expected_destination || udp.payload().len() < 48 {
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
