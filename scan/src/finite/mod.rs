use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::engine;
use crate::engine::target_batch;

pub struct Config {
    pub(crate) dealias: crate::dealias::Config,
    pub(crate) source: sixseven_core::source::TargetSource,
    pub(crate) target_count: Option<usize>,
    pub(crate) interface: link::Interface,
    pub(crate) probe: probe::PreparedProbe,
    pub(crate) pps: usize,
    pub(crate) network_workers: usize,
    pub(crate) receive_window: Duration,
    pub(crate) stats_interval: Option<Duration>,
}

impl Config {
    pub fn from_source(
        source: sixseven_core::source::TargetSource,
        interface: link::Interface,
        probe: probe::PreparedProbe,
    ) -> Self {
        let mut config = Self::new(source, 0, interface, probe);
        config.target_count = None;
        config
    }

    pub fn new(
        source: sixseven_core::source::TargetSource,
        target_count: usize,
        interface: link::Interface,
        probe: probe::PreparedProbe,
    ) -> Self {
        Self {
            dealias: crate::dealias::Config::default(),
            source,
            target_count: Some(target_count),
            interface,
            probe,
            pps: 10_000,
            network_workers: 1,
            receive_window: Duration::from_secs(8),
            stats_interval: None,
        }
    }

    pub fn dealias(mut self, config: crate::dealias::Config) -> Self {
        self.dealias = config;
        self
    }

    pub fn pps(mut self, pps: usize) -> Self {
        self.pps = pps;
        self
    }

    pub fn network_workers(mut self, network_workers: usize) -> Self {
        self.network_workers = network_workers;
        self
    }

    pub fn receive_window(mut self, receive_window: Duration) -> Self {
        self.receive_window = receive_window;
        self
    }

    pub fn stats_interval(mut self, stats_interval: Option<Duration>) -> Self {
        self.stats_interval = stats_interval;
        self
    }
}

#[derive(thiserror::Error, Debug, Clone)]
pub enum Error {
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String),
    #[error("network error: {0}")]
    NetworkError(String),
    #[error("internal error: {0}")]
    InternalError(String),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    pub generated: u64,
    pub sent: u64,
    pub replies: u64,
    pub errors: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plan {
    pub target_count: Option<usize>,
    pub network_workers: usize,
    pub sender_workers: usize,
    pub receiver_workers: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Summary {
    pub dealias: crate::dealias::Report,
    pub plan: Plan,
    pub stats: Stats,
}

#[derive(Debug, Clone)]
pub enum Outcome {
    Completed(Summary),
    Cancelled(Summary),
    Failed(Error),
}

#[derive(Debug, Clone)]
pub enum Event {
    Reply(probe::Reply),
    Stats(Stats),
}

pub struct Run {
    events: mpsc::Receiver<Event>,
    task: Option<JoinHandle<Outcome>>,
    cancel: CancellationToken,
    cancelled: CancellationToken,
    plan: Plan,
}

impl Run {
    pub async fn next(&mut self) -> Option<Event> {
        self.events.recv().await
    }

    pub fn plan(&self) -> Plan {
        self.plan
    }

    pub fn cancel(&self) {
        self.cancelled.cancel();
        self.cancel.cancel();
    }

    pub async fn join(mut self) -> Result<Outcome, Error> {
        while self.events.recv().await.is_some() {}
        let task = self
            .task
            .take()
            .ok_or_else(|| Error::InternalError("scan run already joined".to_string()))?;
        task.await
            .map_err(|e| Error::InternalError(format!("scan run task failed: {e}")))
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            self.cancelled.cancel();
            self.cancel.cancel();
            task.abort();
        }
    }
}

pub async fn run(config: Config) -> Result<Run, Error> {
    if config.dealias.online.is_some() || !config.dealias.aliases.is_empty() {
        return run_dealias(config).await;
    }
    if config.stats_interval.is_some_and(|value| value.is_zero()) {
        return Err(Error::InvalidConfiguration(
            "stats interval must be positive".into(),
        ));
    }
    let stats_interval = config.stats_interval;
    let event_capacity = channel_capacity(config.pps);
    let job = engine::spawn(config).await?;
    let plan = job.plan();
    let cancel = job.cancel_handle();
    let cancelled = job.cancelled_handle();
    let (event_tx, event_rx) = mpsc::channel(event_capacity);
    let task = tokio::spawn(forward_run(
        job,
        event_tx,
        cancelled.clone(),
        stats_interval,
    ));

    Ok(Run {
        events: event_rx,
        task: Some(task),
        cancel,
        cancelled,
        plan,
    })
}

async fn forward_run(
    mut job: engine::ScanJob,
    event_tx: mpsc::Sender<Event>,
    cancelled: CancellationToken,
    stats_interval: Option<Duration>,
) -> Outcome {
    let mut tick = tokio::time::interval(stats_interval.unwrap_or(Duration::from_secs(3600)));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        let event = tokio::select! {
            biased;
            _ = cancelled.cancelled() => { job.cancel(); break; }
            reply = job.next() => match reply { Some(reply) => Event::Reply(reply.reply), None => break },
            _ = tick.tick(), if stats_interval.is_some() => Event::Stats(job.snapshot()),
        };
        let delivered = tokio::select! {
            _ = cancelled.cancelled() => false,
            result = event_tx.send(event) => result.is_ok(),
        };
        if !delivered {
            job.cancel();
            break;
        }
    }
    match job.join().await {
        Ok(summary) if cancelled.is_cancelled() => Outcome::Cancelled(summary),
        Ok(summary) => Outcome::Completed(summary),
        Err(error) => Outcome::Failed(error),
    }
}

#[derive(Clone, Default)]
pub(crate) struct Counters {
    inner: Arc<CountersInner>,
}

#[derive(Default)]
struct CountersInner {
    generated: AtomicU64,
    sent: AtomicU64,
    replies: AtomicU64,
    errors: AtomicU64,
}

impl std::fmt::Debug for Counters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Counters")
            .field("snapshot", &self.snapshot())
            .finish()
    }
}

impl Counters {
    pub(crate) fn snapshot(&self) -> Stats {
        Stats {
            generated: self.inner.generated.load(Ordering::Relaxed),
            sent: self.inner.sent.load(Ordering::Relaxed),
            replies: self.inner.replies.load(Ordering::Relaxed),
            errors: self.inner.errors.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn add_generated(&self, count: u64) {
        if count != 0 {
            self.inner.generated.fetch_add(count, Ordering::Relaxed);
        }
    }

    pub(crate) fn add_sent(&self, count: u64) {
        if count != 0 {
            self.inner.sent.fetch_add(count, Ordering::Relaxed);
        }
    }

    pub(crate) fn add_replies(&self, count: u64) {
        if count != 0 {
            self.inner.replies.fetch_add(count, Ordering::Relaxed);
        }
    }

    pub(crate) fn add_errors(&self, count: u64) {
        if count != 0 {
            self.inner.errors.fetch_add(count, Ordering::Relaxed);
        }
    }
}

pub(crate) fn join_summary(plan: Plan, counters: &Counters) -> Summary {
    Summary {
        dealias: crate::dealias::Report::default(),
        plan,
        stats: counters.snapshot(),
    }
}

pub(crate) fn map_link_error(err: link::Error) -> Error {
    match err {
        link::Error::InvalidInterface(msg) | link::Error::InvalidConfiguration(msg) => {
            Error::InvalidConfiguration(msg)
        }
        link::Error::BackendUnavailable(msg) => Error::NetworkError(msg),
        link::Error::Io(err) => Error::NetworkError(err.to_string()),
    }
}

pub(crate) fn validate(network_workers: usize) -> Result<(), Error> {
    if network_workers == 0 {
        return Err(Error::InvalidConfiguration(
            "source workers must be at least 1".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn active_sender_workers(requested: usize, pps: usize) -> usize {
    let requested = requested.max(1);
    if pps == 0 {
        requested
    } else {
        requested.min(pps.max(1))
    }
}

pub(crate) fn split_rate(pps: usize, shards: usize) -> Vec<usize> {
    if pps == 0 {
        return vec![0; shards.max(1)];
    }

    let shards = shards.max(1);
    let base = pps / shards;
    let extra = pps % shards;
    (0..shards)
        .map(|shard| base + usize::from(shard < extra))
        .collect()
}

pub(crate) fn channel_capacity(pps: usize) -> usize {
    pps.max(1024).clamp(4096, 1_000_000)
}

pub(crate) const TARGET_CHANNEL_TARGET_CAPACITY: usize = 262_144;

pub(crate) fn target_channel_capacity(pps: usize) -> usize {
    let target_capacity = pps
        .max(TARGET_CHANNEL_TARGET_CAPACITY)
        .clamp(TARGET_CHANNEL_TARGET_CAPACITY, 1_000_000);
    target_capacity.div_ceil(target_batch::TARGET_BATCH_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_run(replies: usize, fail: bool) -> Run {
        let job = engine::test_job(replies, fail);
        let plan = job.plan();
        let cancel = job.cancel_handle();
        let cancelled = job.cancelled_handle();
        let (tx, events) = mpsc::channel(1);
        let task = tokio::spawn(forward_run(job, tx, cancelled.clone(), None));
        Run {
            events,
            task: Some(task),
            cancel,
            cancelled,
            plan,
        }
    }

    #[tokio::test]
    async fn joining_without_reading_events_completes() {
        let outcome = tokio::time::timeout(Duration::from_secs(2), test_run(100, false).join())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(outcome, Outcome::Completed(_)));
    }

    #[tokio::test]
    async fn join_reports_worker_failure() {
        assert!(matches!(
            test_run(3, true).join().await.unwrap(),
            Outcome::Failed(_)
        ));
    }

    #[tokio::test]
    async fn cancellation_interrupts_backpressure() {
        let run = test_run(100_000, false);
        tokio::task::yield_now().await;
        run.cancel();
        let outcome = tokio::time::timeout(Duration::from_secs(2), run.join())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(outcome, Outcome::Cancelled(_)));
    }
}

#[cfg(test)]
mod dealias_tests {
    use super::*;
    #[tokio::test]
    async fn fixed_source_uses_shared_detection_and_offline_filtering() {
        for online in [false, true] {
            let (targets, target_rx) = flume::bounded(2);
            let (submitted, acknowledgements) = mpsc::channel(2);
            let job = engine::test_session(engine::BatchInput {
                targets: target_rx,
                submitted,
            });
            let cancel = job.cancel_handle();
            let stopped = job.cancelled_handle();
            let config = if online {
                crate::dealias::Config {
                    online: Some(crate::dealias::OnlineConfig {
                        prefix_lengths: vec![120],
                        seed: 0,
                    }),
                    ..Default::default()
                }
            } else {
                crate::dealias::Config {
                    aliases: ["::/120".parse().unwrap()].into_iter().collect(),
                    online: None,
                }
            };
            let session = crate::session::Session::new(
                job,
                targets,
                acknowledgements,
                Duration::from_millis(1),
                2048,
                config,
            )
            .unwrap();
            let (events, mut input) = mpsc::channel(32);
            let source =
                sixseven_core::source::addresses((1..=6u128).map(std::net::Ipv6Addr::from));
            let (summary, report) = tokio::time::timeout(
                Duration::from_secs(2),
                drive_source(source, Some(6), session, events, cancel, stopped, None),
            )
            .await
            .unwrap()
            .unwrap();
            assert_eq!(summary.stats.generated, 6);
            assert_eq!(summary.stats.sent, if online { 6 } else { 0 });
            assert_eq!(report.probes_sent, if online { 3 } else { 0 });
            assert!(report.aliases.contains(std::net::Ipv6Addr::LOCALHOST));
            let mut count = 0;
            while let Some(event) = input.recv().await {
                if matches!(event, Event::Reply(_)) {
                    count += 1;
                }
            }
            assert_eq!(count, if online { 6 } else { 0 });
        }
    }
}

async fn run_dealias(mut config: Config) -> Result<Run, Error> {
    if config.stats_interval.is_some_and(|value| value.is_zero()) {
        return Err(Error::InvalidConfiguration(
            "stats interval must be positive".into(),
        ));
    }
    let source = std::mem::replace(&mut config.source, sixseven_core::source::addresses([]));
    let count = config.target_count.take();
    let window = config.receive_window;
    config.receive_window = Duration::ZERO;
    let dealias = config.dealias.clone();
    let interval = config.stats_interval;
    let (targets, target_rx) = flume::bounded(64);
    let (submitted, acknowledgements) = mpsc::channel(64);
    let job = engine::spawn_inner(
        config,
        Some(engine::BatchInput {
            targets: target_rx,
            submitted,
        }),
    )
    .await?;
    let mut plan = job.plan();
    plan.target_count = count;
    let cancel = job.cancel_handle();
    let cancelled = job.cancelled_handle();
    let session =
        crate::session::Session::new(job, targets, acknowledgements, window, 262144, dealias)?;
    let (tx, events) = mpsc::channel(4096);
    let stopped = cancelled.clone();
    let shutdown = cancel.clone();
    let task = tokio::spawn(async move {
        let result = drive_source(
            source,
            count,
            session,
            tx,
            shutdown,
            stopped.clone(),
            interval,
        )
        .await;
        match result {
            Ok((mut summary, report)) => {
                summary.plan = plan;
                summary.dealias = report;
                if stopped.is_cancelled() {
                    Outcome::Cancelled(summary)
                } else {
                    Outcome::Completed(summary)
                }
            }
            Err(_error) if stopped.is_cancelled() => Outcome::Cancelled(Summary {
                plan,
                stats: Stats::default(),
                dealias: crate::dealias::Report::default(),
            }),
            Err(error) => Outcome::Failed(error),
        }
    });
    Ok(Run {
        events,
        task: Some(task),
        cancel,
        cancelled,
        plan,
    })
}

async fn drive_source(
    source: sixseven_core::source::TargetSource,
    count: Option<usize>,
    mut session: crate::session::Session,
    events: mpsc::Sender<Event>,
    cancel: CancellationToken,
    stopped: CancellationToken,
    interval: Option<Duration>,
) -> Result<(Summary, crate::dealias::Report), Error> {
    let guard = cancel.clone().drop_guard();
    // A separate producer keeps synchronous file reads off the network executor.
    let (input_tx, input) = flume::bounded(4);
    let counters = Counters::default();
    let producer = tokio::spawn(engine::generate::stream_targets(
        source,
        count,
        input_tx,
        cancel.clone(),
        counters.clone(),
    ));
    let mut ended = false;
    let mut tick = tokio::time::interval(interval.unwrap_or(Duration::from_secs(3600)));
    loop {
        if ended && session.settled() {
            break;
        }
        tokio::select! {
            _ = stopped.cancelled() => { session.cancel(); return Err(Error::NetworkError("scan cancelled".into())); },
            batch = input.recv_async(), if !ended && session.capacity() >= target_batch::TARGET_BATCH_SIZE => {
                match batch { Ok(batch) => session.submit(batch), Err(_) => ended = true }
            },
            update = session.next() => {
                let update = update?;
                counters.add_replies(update.replies.len() as u64);
                for reply in update.replies {
                    tokio::select! {
                        result = events.send(Event::Reply(reply)) => if result.is_err() { return Err(Error::InternalError("output closed".into())); },
                        _ = stopped.cancelled() => return Err(Error::NetworkError("scan cancelled".into())),
                    }
                }
            },
            _ = tick.tick(), if interval.is_some() => {
                let mut stats = counters.snapshot(); stats.sent = session.sent;
                tokio::select! {
                    result = events.send(Event::Stats(stats)) => if result.is_err() { return Err(Error::InternalError("output closed".into())); },
                    _ = stopped.cancelled() => return Err(Error::NetworkError("scan cancelled".into())),
                }
            },
        }
    }
    producer
        .await
        .map_err(|e| Error::InternalError(e.to_string()))??;
    let (mut summary, report) = session.finish().await?;
    summary.stats.generated = counters.snapshot().generated;
    guard.disarm();
    Ok((summary, report))
}
