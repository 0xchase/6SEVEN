use probe::{Decode, PreparedProbe, RawReply};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::finite::{Counters, Error, map_link_error};

pub(crate) async fn receive_records(
    interface: link::Interface,
    mut receiver: link::Rx,
    probe: PreparedProbe,
    alias_probe: Option<PreparedProbe>,
    frame_len: usize,
    result_tx: mpsc::Sender<super::ReceivedReply>,
    cancel: CancellationToken,
    stats: Counters,
) -> Result<(), Error> {
    let mut frame_buf = vec![0u8; frame_len];
    while !cancel.is_cancelled() {
        while !cancel.is_cancelled() {
            match receiver.try_recv(&mut frame_buf) {
                Ok(Some(n)) => {
                    if let Some(record) =
                        decode_reply(&interface, &probe, alias_probe.as_ref(), &frame_buf[..n])
                    {
                        if record.purpose == super::target_batch::ProbePurpose::Discovery {
                            stats.add_replies(1);
                        }
                        if result_tx.send(record).await.is_err() {
                            return Ok(());
                        }
                    }
                }
                Ok(None) => break,
                Err(err) => {
                    let err = map_link_error(err);
                    stats.add_errors(1);
                    cancel.cancel();
                    return Err(err);
                }
            }
        }

        if cancel.is_cancelled() {
            break;
        }

        let n = match tokio::select! { result = receiver.recv(&mut frame_buf) => result, _ = cancel.cancelled() => return Ok(()) }
        {
            Ok(n) => n,
            Err(err) => {
                let err = map_link_error(err);
                stats.add_errors(1);
                cancel.cancel();
                return Err(err);
            }
        };

        if let Some(record) =
            decode_reply(&interface, &probe, alias_probe.as_ref(), &frame_buf[..n])
        {
            if record.purpose == super::target_batch::ProbePurpose::Discovery {
                stats.add_replies(1);
            }
            if result_tx.send(record).await.is_err() {
                break;
            }
        }
    }

    Ok(())
}

fn decode_reply(
    interface: &link::Interface,
    probe: &PreparedProbe,
    alias_probe: Option<&PreparedProbe>,
    packet: &[u8],
) -> Option<super::ReceivedReply> {
    let reply = match link::decode_ipv6_frame(interface, packet) {
        link::FrameDecode::Packet(reply) => RawReply {
            responder: reply.source,
            protocol: reply.next_header,
            payload: reply.payload,
        },
        link::FrameDecode::NotMine | link::FrameDecode::Malformed => return None,
    };

    decode_probes(probe, alias_probe, reply)
}

fn decode_probes(
    probe: &PreparedProbe,
    alias_probe: Option<&PreparedProbe>,
    reply: RawReply<'_>,
) -> Option<super::ReceivedReply> {
    if let Some(alias) = alias_probe {
        if let Decode::Reply(reply) = alias.decode(reply) {
            return Some(super::ReceivedReply {
                purpose: super::target_batch::ProbePurpose::Alias,
                reply,
            });
        }
    }
    match probe.decode(reply) {
        Decode::Reply(reply) => Some(super::ReceivedReply {
            purpose: super::target_batch::ProbePurpose::Discovery,
            reply,
        }),
        Decode::NotMine | Decode::Malformed => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_validation_contexts_separate_overlapping_icmp_targets() {
        let source = "2001:db8::1".parse().unwrap();
        let target = "2001:db8::2".parse().unwrap();
        let config = probe::ProbeConfig::Icmp(Default::default());
        let discovery = config.clone().prepare_for(source).unwrap();
        let alias = config.prepare_for(source).unwrap();
        for (prepared, expected) in [
            (
                &discovery,
                super::super::target_batch::ProbePurpose::Discovery,
            ),
            (&alias, super::super::target_batch::ProbePurpose::Alias),
        ] {
            let mut payload = vec![0; prepared.max_packet_len()];
            let packet = prepared
                .encode(&mut payload, probe::Request { target, token: 0 })
                .unwrap();
            payload[0] = 129; // Echo Reply: validation data is echoed unchanged.
            let reply = decode_probes(
                &discovery,
                Some(&alias),
                RawReply {
                    responder: target,
                    protocol: packet.protocol,
                    payload: &payload,
                },
            )
            .unwrap();
            assert_eq!(reply.purpose, expected);
            assert_eq!(reply.reply.target, target);
        }
    }
}
