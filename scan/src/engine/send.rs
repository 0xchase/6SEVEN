use std::net::Ipv6Addr;
use std::time::Instant;

use probe::{PreparedProbe, Request};
use tokio_util::sync::CancellationToken;

use super::rate::Rate;
use super::target_batch::TargetBatch;
use crate::finite::Counters;
use crate::finite::Error;

pub(crate) struct SenderWorker {
    interface: link::Interface,
    sender: link::Tx,
    probe: PreparedProbe,
    alias_probe: Option<PreparedProbe>,
    packet_buf: Vec<u8>,
    frame_buf: Vec<u8>,
}

impl SenderWorker {
    pub(crate) fn new(
        interface: link::Interface,
        sender: link::Tx,
        probe: PreparedProbe,
        alias_probe: Option<PreparedProbe>,
    ) -> Self {
        let packet_len = probe.max_packet_len().max(
            alias_probe
                .as_ref()
                .map_or(0, PreparedProbe::max_packet_len),
        );
        let frame_len = link::ETH_HEADER_LEN + link::IPV6_HEADER_LEN + packet_len;
        Self {
            interface,
            sender,
            probe,
            alias_probe,
            packet_buf: vec![0u8; packet_len],
            frame_buf: vec![0u8; frame_len],
        }
    }

    async fn send(
        &mut self,
        target: Ipv6Addr,
        token: u8,
        purpose: super::target_batch::ProbePurpose,
    ) -> Result<(), String> {
        let request = Request { target, token };
        let probe = match purpose {
            super::target_batch::ProbePurpose::Discovery => &self.probe,
            super::target_batch::ProbePurpose::Alias => self
                .alias_probe
                .as_ref()
                .ok_or("alias probe is not configured")?,
        };
        let Some(packet) = probe.encode(&mut self.packet_buf, request) else {
            return Err(format!(
                "probe packet construction failed for target {target}"
            ));
        };
        let Some(frame_len) = link::encode_ipv6_frame(
            &self.interface,
            &mut self.frame_buf,
            link::OutboundIpv6Packet {
                destination: target,
                next_header: packet.protocol,
                payload: &self.packet_buf[..packet.payload_len],
            },
        ) else {
            return Err(format!(
                "IPv6 frame construction failed for target {target}"
            ));
        };

        let queued = self
            .sender
            .try_send(&self.frame_buf[..frame_len])
            .map_err(|e| format!("transmit failed for {target}: {e}"))?;
        if queued {
            return Ok(());
        }

        self.sender
            .send(&self.frame_buf[..frame_len])
            .await
            .map_err(|e| format!("transmit failed for {target}: {e}"))
    }

    async fn flush(&mut self, cancel: &CancellationToken) -> Result<(), Error> {
        tokio::select! { result = self.sender.flush() => result, _ = cancel.cancelled() => return Ok(()) }
            .map_err(|e| Error::NetworkError(format!("transmit flush failed: {e}")))
    }
}

pub(crate) async fn send_targets(
    targets: flume::Receiver<TargetBatch>,
    limiter: usize,
    mut sender: SenderWorker,
    cancel: CancellationToken,
    stats: Counters,
    submitted: Option<tokio::sync::mpsc::Sender<TargetBatch>>,
) -> Result<(), Error> {
    let mut rate = Rate::new(limiter);
    loop {
        let batch = tokio::select! {
            result = targets.recv_async() => match result {
                Ok(batch) => batch,
                Err(_) => return sender.flush(&cancel).await,
            },
            _ = cancel.cancelled() => return Ok(()),
        };
        for burst in batch.as_slice().chunks(rate.burst_size()) {
            let deadline = rate.reserve(burst.len(), Instant::now());
            if deadline > Instant::now() {
                tokio::select! {
                    _ = tokio::time::sleep_until(deadline.into()) => {},
                    _ = cancel.cancelled() => return Ok(()),
                }
            }
            let mut sent = 0;
            let result = tokio::select! {
                biased;
                _ = cancel.cancelled() => None,
                result = async {
                    for &target in burst {
                        sender.send(target, batch.token, batch.purpose).await.map_err(Error::NetworkError)?;
                        sent += 1;
                    }
                    sender.flush(&cancel).await
                } => Some(result),
            };
            stats.add_sent(sent);
            match result {
                None => return Ok(()),
                Some(Ok(())) => {}
                Some(Err(error)) => {
                    stats.add_errors(1);
                    cancel.cancel();
                    return Err(error);
                }
            }
        }
        if let Some(submitted) = &submitted {
            tokio::select! {
                result = submitted.send(batch) => if result.is_err() { return Ok(()); },
                _ = cancel.cancelled() => return Ok(()),
            }
        }
        tokio::task::yield_now().await;
    }
}
