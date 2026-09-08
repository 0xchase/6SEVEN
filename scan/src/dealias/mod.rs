//! Shared offline prefix filtering and paper-style ICMPv6 alias detection.
mod detector;
use crate::finite::Error;
pub(crate) use detector::{Detector, Status};
use sixseven_core::PrefixSet;

#[derive(Debug, Clone, Default)]
pub struct Config {
    pub aliases: PrefixSet,
    pub online: Option<OnlineConfig>,
}
#[derive(Debug, Clone)]
pub struct OnlineConfig {
    pub prefix_lengths: Vec<u8>,
    pub seed: u64,
}
impl Default for OnlineConfig {
    fn default() -> Self {
        Self {
            prefix_lengths: vec![48, 64, 96, 112, 116, 120],
            seed: 0,
        }
    }
}
impl OnlineConfig {
    pub fn validate(&mut self) -> Result<(), Error> {
        self.prefix_lengths.sort_unstable();
        self.prefix_lengths.dedup();
        if self.prefix_lengths.is_empty()
            || self.prefix_lengths.len() > 8
            || self.prefix_lengths.iter().any(|&n| n > 126)
        {
            return Err(Error::InvalidConfiguration(
                "alias detection requires 1–8 distinct prefix lengths between 0 and 126".into(),
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    pub aliases: PrefixSet,
    pub probes_sent: u64,
}

/// Classify a stream of candidate addresses using the same session as live scans.
/// This does not send discovery probes or interpret model feedback.
pub async fn run(
    source: sixseven_core::source::TargetSource,
    interface: link::Interface,
    config: Config,
    pps: usize,
    network_workers: usize,
    receive_window: std::time::Duration,
) -> Result<Report, Error> {
    if config.online.is_none() {
        return Ok(Report {
            aliases: config.aliases,
            probes_sent: 0,
        });
    }
    let probe = probe::ProbeConfig::Icmp(Default::default())
        .prepare_for(interface.source_addr())
        .map_err(|e| Error::InvalidConfiguration(e.to_string()))?;
    let transport =
        crate::finite::Config::from_source(sixseven_core::source::addresses([]), interface, probe)
            .dealias(config.clone())
            .pps(pps)
            .network_workers(network_workers)
            .receive_window(std::time::Duration::ZERO);
    let (targets, target_rx) = flume::bounded(64);
    let (submitted, acknowledgements) = tokio::sync::mpsc::channel(64);
    let job = crate::engine::spawn_inner(
        transport,
        Some(crate::engine::BatchInput {
            targets: target_rx,
            submitted,
        }),
    )
    .await?;
    let cancel = job.cancel_handle();
    let guard = cancel.clone().drop_guard();
    let mut session = crate::session::Session::new(
        job,
        targets,
        acknowledgements,
        receive_window,
        262144,
        config,
    )?;
    let (input_tx, input) = flume::bounded(4);
    let producer = tokio::spawn(crate::engine::generate::stream_targets(
        source,
        None,
        input_tx,
        cancel,
        crate::finite::Counters::default(),
    ));
    let mut ended = false;
    loop {
        if ended && session.settled() {
            break;
        }
        tokio::select! {
            batch = input.recv_async(), if !ended && session.capacity() >= crate::engine::target_batch::TARGET_BATCH_SIZE => {
                match batch {
                    Ok(batch) => for &address in batch.as_slice() { session.consider(address); },
                    Err(_) => ended = true,
                }
                session.take_update();
            },
            update = session.next() => { update?; },
        }
    }
    producer
        .await
        .map_err(|e| Error::InternalError(e.to_string()))??;
    let (_, report) = session.finish().await?;
    guard.disarm();
    Ok(report)
}
