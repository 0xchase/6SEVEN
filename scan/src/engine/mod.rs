pub(crate) mod generate;
mod rate;
mod recv;
mod send;
pub(crate) mod target_batch;
use std::time::Duration;

use probe::PreparedProbe;
use tokio::sync::mpsc;
use tokio::task::{JoinHandle, JoinSet};
use tokio_util::sync::CancellationToken;

use self::generate::stream_targets;
use self::recv::receive_records;
use self::send::{SenderWorker, send_targets};
use self::target_batch::TargetBatch;
use crate::finite::{
    Config, Counters, Error, Plan, Stats, Summary, active_sender_workers, channel_capacity,
    join_summary, map_link_error, split_rate, target_channel_capacity, validate,
};

#[derive(Debug, Clone, Copy)]
pub(crate) struct ReceivedReply {
    pub purpose: target_batch::ProbePurpose,
    pub reply: probe::Reply,
}

pub(crate) struct ScanJob {
    results: mpsc::Receiver<ReceivedReply>,
    task: Option<JoinHandle<Result<Summary, Error>>>,
    cancel: CancellationToken,
    cancelled: CancellationToken,
    plan: Plan,
    stats: Counters,
}

impl ScanJob {
    pub(crate) fn try_next(&mut self) -> Result<ReceivedReply, mpsc::error::TryRecvError> {
        self.results.try_recv()
    }

    pub(crate) async fn next(&mut self) -> Option<ReceivedReply> {
        self.results.recv().await
    }

    pub(crate) async fn completion_error(&mut self) -> Error {
        match self.task.take() {
            Some(task) => match task.await {
                Ok(Err(error)) => error,
                Err(error) => Error::InternalError(error.to_string()),
                Ok(Ok(_)) => Error::NetworkError(
                    "network session ended before outstanding work completed".into(),
                ),
            },
            None => Error::InternalError("network session already joined".into()),
        }
    }

    pub(crate) fn plan(&self) -> Plan {
        self.plan
    }

    pub(crate) fn snapshot(&self) -> Stats {
        self.stats.snapshot()
    }

    pub(crate) fn cancel_handle(&self) -> CancellationToken {
        self.cancel.clone()
    }

    pub(crate) fn cancelled_handle(&self) -> CancellationToken {
        self.cancelled.clone()
    }

    pub(crate) fn cancel(&self) {
        self.cancelled.cancel();
        self.cancel.cancel();
    }

    pub(crate) async fn join(mut self) -> Result<Summary, Error> {
        self.results.close();
        let task = self
            .task
            .take()
            .ok_or_else(|| Error::InternalError("scan task already joined".to_string()))?;
        task.await
            .map_err(|e| Error::InternalError(format!("scan task failed: {e}")))?
    }
}

impl Drop for ScanJob {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            self.cancelled.cancel();
            self.cancel.cancel();
            task.abort();
        }
    }
}

pub(crate) struct BatchInput {
    pub targets: flume::Receiver<TargetBatch>,
    pub submitted: mpsc::Sender<TargetBatch>,
}

pub(crate) async fn spawn(config: Config) -> Result<ScanJob, Error> {
    spawn_inner(config, None).await
}

pub(crate) async fn spawn_inner(
    config: Config,
    batches: Option<BatchInput>,
) -> Result<ScanJob, Error> {
    let Config {
        source,
        target_count,
        interface,
        probe,
        pps,
        network_workers,
        receive_window,
        stats_interval: _,
        mut dealias,
    } = config;

    validate(network_workers)?;
    if let Some(online) = &mut dealias.online {
        online.validate()?;
    }
    let alias_probe = if dealias.online.is_some() {
        Some(
            probe::ProbeConfig::Icmp(Default::default())
                .prepare_for(interface.source_addr())
                .map_err(|e| Error::InvalidConfiguration(e.to_string()))?,
        )
    } else {
        None
    };

    let result_capacity = channel_capacity(pps);
    let (result_tx, result_rx) = mpsc::channel(result_capacity);
    let cancel = CancellationToken::new();
    let cancelled = CancellationToken::new();
    let stats = Counters::default();

    if target_count == Some(0) {
        let plan = Plan {
            target_count: Some(0),
            network_workers,
            sender_workers: 0,
            receiver_workers: 0,
        };
        let stats_handle = stats.clone();
        let task = tokio::spawn(async move { Ok(join_summary(plan, &stats_handle)) });
        return Ok(ScanJob {
            results: result_rx,
            task: Some(task),
            cancel,
            cancelled,
            plan,
            stats,
        });
    }

    let mut prepared =
        PreparedScan::new(source, interface, probe, alias_probe, pps, network_workers)?;
    prepared.batches = batches;
    let plan = Plan {
        target_count,
        network_workers: prepared.sender_workers,
        sender_workers: prepared.sender_workers,
        receiver_workers: prepared.sender_workers,
    };
    let stats_handle = stats.clone();
    let engine_cancel = cancel.clone();
    let task = tokio::spawn(async move {
        run_engine(
            prepared,
            pps,
            receive_window,
            result_tx,
            engine_cancel,
            stats_handle,
            plan,
        )
        .await
    });

    Ok(ScanJob {
        results: result_rx,
        task: Some(task),
        cancel,
        cancelled,
        plan,
        stats,
    })
}

async fn run_engine(
    mut prepared: PreparedScan,
    pps: usize,
    receive_window: Duration,
    result_tx: mpsc::Sender<ReceivedReply>,
    cancel: CancellationToken,
    stats: Counters,
    plan: Plan,
) -> Result<Summary, Error> {
    let receivers = spawn_receivers(&prepared, &result_tx, &cancel, &stats)?;
    let (target_tx, target_rx) = flume::bounded(target_channel_capacity(pps));
    let (target_rx, submitted) = match prepared.batches.take() {
        Some(input) => (input.targets, Some(input.submitted)),
        None => (target_rx, None),
    };
    let external_source = submitted.is_some();
    let senders = match spawn_senders(&prepared, target_rx, submitted, pps, &cancel, &stats) {
        Ok(senders) => senders,
        Err(err) => {
            cancel.cancel();
            let _ = await_workers(receivers, "receiver").await;
            return Err(err);
        }
    };

    let generator = async {
        if external_source {
            drop(target_tx);
            Ok(())
        } else {
            stream_targets(
                prepared.source,
                plan.target_count,
                target_tx,
                cancel.clone(),
                stats.clone(),
            )
            .await
        }
    };
    drop(result_tx);

    let send_phase = async {
        tokio::try_join!(generator, await_workers(senders, "sender"))?;
        if !cancel.is_cancelled() && !receive_window.is_zero() {
            tokio::select! { _ = tokio::time::sleep(receive_window) => {}, _ = cancel.cancelled() => {} }
        }
        cancel.cancel();
        Ok::<_, Error>(())
    };
    tokio::try_join!(send_phase, await_workers(receivers, "receiver"))?;
    Ok(join_summary(plan, &stats))
}

struct PreparedScan {
    alias_probe: Option<PreparedProbe>,
    probe: PreparedProbe,
    link_iface: link::Interface,
    frames: link::FrameConfig,
    sender_workers: usize,
    source: sixseven_core::source::TargetSource,
    batches: Option<BatchInput>,
}

impl PreparedScan {
    fn new(
        source: sixseven_core::source::TargetSource,
        interface: link::Interface,
        probe: PreparedProbe,
        alias_probe: Option<PreparedProbe>,
        pps: usize,
        network_workers: usize,
    ) -> Result<Self, Error> {
        if probe.source_addr() != interface.source_addr() {
            return Err(Error::InvalidConfiguration(format!(
                "prepared probe source {} does not match interface source {}",
                probe.source_addr(),
                interface.source_addr()
            )));
        }
        let frames = if let Some(alias) = &alias_probe {
            validate_probe_network_fit(alias, &interface)?;
            frame_config_for_probe(if alias.max_packet_len() > probe.max_packet_len() {
                alias
            } else {
                &probe
            })?
        } else {
            frame_config_for_probe(&probe)?
        };
        validate_probe_network_fit(&probe, &interface)?;
        let sender_workers = active_sender_workers(interface.sender_shards(network_workers), pps);

        Ok(Self {
            alias_probe,
            probe,
            link_iface: interface,
            frames,
            sender_workers,
            source,
            batches: None,
        })
    }
}

fn spawn_receivers(
    prepared: &PreparedScan,
    result_tx: &mpsc::Sender<ReceivedReply>,
    cancel: &CancellationToken,
    stats: &Counters,
) -> Result<JoinSet<Result<(), Error>>, Error> {
    let mut receivers = JoinSet::new();
    for _ in 0..prepared.sender_workers {
        let receiver = prepared
            .link_iface
            .rx(prepared.frames)
            .map_err(map_link_error)?;
        let interface = prepared.link_iface.clone();
        let result_tx = result_tx.clone();
        let cancel = cancel.clone();
        let probe = prepared.probe.clone();
        let alias_probe = prepared.alias_probe.clone();
        let frame_len = prepared.frames.rx_frame_len();
        let stats = stats.clone();
        receivers.spawn(async move {
            receive_records(
                interface,
                receiver,
                probe,
                alias_probe,
                frame_len,
                result_tx,
                cancel,
                stats,
            )
            .await
        });
    }
    Ok(receivers)
}

fn spawn_senders(
    prepared: &PreparedScan,
    target_rx: flume::Receiver<TargetBatch>,
    submitted: Option<mpsc::Sender<TargetBatch>>,
    pps: usize,
    cancel: &CancellationToken,
    stats: &Counters,
) -> Result<JoinSet<Result<(), Error>>, Error> {
    let mut senders = JoinSet::new();
    for limiter in split_rate(pps, prepared.sender_workers) {
        let sender = SenderWorker::new(
            prepared.link_iface.clone(),
            prepared
                .link_iface
                .tx(prepared.frames)
                .map_err(map_link_error)?,
            prepared.probe.clone(),
            prepared.alias_probe.clone(),
        );
        let cancel = cancel.clone();
        let target_rx = target_rx.clone();
        let stats = stats.clone();
        let submitted = submitted.clone();
        senders.spawn(async move {
            send_targets(target_rx, limiter, sender, cancel, stats, submitted).await
        });
    }
    Ok(senders)
}

async fn await_workers(
    mut workers: JoinSet<Result<(), Error>>,
    label: &'static str,
) -> Result<(), Error> {
    while let Some(result) = workers.join_next().await {
        result
            .map_err(|error| Error::InternalError(format!("{label} worker failed: {error}")))??;
    }
    Ok(())
}

fn frame_config_for_probe(probe: &PreparedProbe) -> Result<link::FrameConfig, Error> {
    let packet_len = probe.max_packet_len();
    validate_ipv6_payload_len(packet_len)?;
    let tx_frame_len = link::ETH_HEADER_LEN + link::IPV6_HEADER_LEN + packet_len;
    let rx_frame_len = link::DEFAULT_RX_FRAME_LEN.max(tx_frame_len);
    Ok(link::FrameConfig::new(tx_frame_len, rx_frame_len))
}

#[cfg(test)]
pub(super) fn test_job(replies: usize, fail: bool) -> ScanJob {
    let (tx, results) = mpsc::channel(1);
    let cancel = CancellationToken::new();
    let stopped = cancel.clone();
    let plan = Plan {
        target_count: Some(replies),
        network_workers: 1,
        sender_workers: 1,
        receiver_workers: 1,
    };
    let stats = Counters::default();
    let counters = stats.clone();
    let task = tokio::spawn(async move {
        for _ in 0..replies {
            let reply = probe::Reply {
                target: std::net::Ipv6Addr::LOCALHOST,
                responder: std::net::Ipv6Addr::LOCALHOST,
                token: 0,
                kind: probe::ReplyKind::Direct,
            };
            tokio::select! { _ = stopped.cancelled() => break, result = tx.send(ReceivedReply { purpose: target_batch::ProbePurpose::Discovery, reply }) => { if result.is_err() { break; } } }
        }
        if fail {
            Err(Error::NetworkError("test failure".into()))
        } else {
            Ok(join_summary(plan, &counters))
        }
    });
    ScanJob {
        results,
        task: Some(task),
        cancel,
        cancelled: CancellationToken::new(),
        plan,
        stats,
    }
}

fn validate_probe_network_fit(probe: &PreparedProbe, iface: &link::Interface) -> Result<(), Error> {
    let packet_len = probe.max_packet_len();
    validate_ipv6_payload_len(packet_len)?;
    let ip_packet_len = link::IPV6_HEADER_LEN
        .checked_add(packet_len)
        .ok_or_else(|| Error::InvalidConfiguration("probe packet size overflows".to_string()))?;
    if let Some(mtu) = iface.mtu()
        && ip_packet_len > mtu
    {
        return Err(Error::InvalidConfiguration(format!(
            "probe packet size {ip_packet_len} exceeds interface MTU {mtu}"
        )));
    }
    Ok(())
}

fn validate_ipv6_payload_len(packet_len: usize) -> Result<(), Error> {
    if u16::try_from(packet_len).is_err() {
        return Err(Error::InvalidConfiguration(format!(
            "probe packet size {packet_len} exceeds IPv6 payload limit"
        )));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn test_session(input: BatchInput) -> ScanJob {
    test_session_replies(input, 3)
}

#[cfg(test)]
pub(crate) fn test_session_replies(input: BatchInput, alias_replies: usize) -> ScanJob {
    let (tx, results) = mpsc::channel(64);
    let cancel = CancellationToken::new();
    let stopped = cancel.clone();
    let plan = Plan {
        target_count: None,
        network_workers: 1,
        sender_workers: 1,
        receiver_workers: 1,
    };
    let stats = Counters::default();
    let counters = stats.clone();
    let task = tokio::spawn(async move {
        loop {
            let batch = tokio::select! {
                result = input.targets.recv_async() => match result { Ok(batch) => batch, Err(_) => break },
                _ = stopped.cancelled() => break,
            };
            counters.add_sent(batch.len() as u64);
            for (index, &target) in batch.as_slice().iter().enumerate() {
                if batch.purpose == target_batch::ProbePurpose::Alias && index >= alias_replies {
                    continue;
                }
                if batch.purpose == target_batch::ProbePurpose::Discovery {
                    counters.add_replies(1);
                }
                tx.send(ReceivedReply {
                    purpose: batch.purpose,
                    reply: probe::Reply {
                        target,
                        responder: target,
                        token: batch.token,
                        kind: probe::ReplyKind::Direct,
                    },
                })
                .await
                .map_err(|error| Error::InternalError(error.to_string()))?;
            }
            input
                .submitted
                .send(batch)
                .await
                .map_err(|error| Error::InternalError(error.to_string()))?;
        }
        Ok(join_summary(plan, &counters))
    });
    ScanJob {
        results,
        task: Some(task),
        cancel,
        cancelled: CancellationToken::new(),
        plan,
        stats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    struct Finished(Arc<AtomicBool>);
    impl Drop for Finished {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn worker_failure_aborts_siblings() {
        let finished = Arc::new(AtomicBool::new(false));
        let flag = finished.clone();
        let (ready, started) = tokio::sync::oneshot::channel();
        let mut workers = JoinSet::new();
        workers.spawn(async move {
            let _finished = Finished(flag);
            ready.send(()).unwrap();
            std::future::pending::<Result<(), Error>>().await
        });
        workers.spawn(async move {
            started.await.unwrap();
            Err(Error::InternalError("failed worker".into()))
        });
        assert!(await_workers(workers, "test").await.is_err());
        tokio::time::timeout(Duration::from_secs(1), async {
            while !finished.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }
}
