//! Shared probe tracking and feedback finalization. Models and files stay outside.
use crate::{
    dealias::{self, Detector, Status},
    engine::{
        self,
        target_batch::{ProbePurpose, TargetBatch},
    },
    finite::Error,
    reply_record,
};
use sixseven_core::{Feedback, Ipv6Prefix, PrefixSet};
use sixseven_formats::csv::scan::{OutcomeClass, ScanRecord};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    net::Ipv6Addr,
    time::Duration,
};
use tokio::time::Instant;

#[derive(Default)]
struct Pending {
    copies: usize,
    active: bool,
    responded: bool,
    reported: bool,
}
#[derive(Default)]
pub(crate) struct Update {
    pub records: Vec<ScanRecord>,
    pub replies: Vec<probe::Reply>,
    pub feedback: Vec<Feedback>,
}
pub(crate) struct Session {
    scan: engine::ScanJob,
    tx: flume::Sender<TargetBatch>,
    submitted: tokio::sync::mpsc::Receiver<TargetBatch>,
    pending: HashMap<Ipv6Addr, Pending>,
    deadlines: VecDeque<(Instant, TargetBatch)>,
    outbox: VecDeque<TargetBatch>,
    waiting: HashMap<Ipv6Prefix, HashSet<Ipv6Addr>>,
    waiting_count: usize,
    detector: Detector,
    tests: HashMap<u64, Ipv6Prefix>,
    next_id: u64,
    wire: usize,
    max_in_flight: usize,
    receive_window: Duration,
    update: Update,
    pub sent: u64,
    pub alias_sent: u64,
}
impl Session {
    pub fn new(
        scan: engine::ScanJob,
        tx: flume::Sender<TargetBatch>,
        submitted: tokio::sync::mpsc::Receiver<TargetBatch>,
        receive_window: Duration,
        max_in_flight: usize,
        config: dealias::Config,
    ) -> Result<Self, Error> {
        if config.online.is_some() && max_in_flight < 3 {
            return Err(Error::InvalidConfiguration(
                "active dealiasing requires max-in-flight of at least 3".into(),
            ));
        }
        let initial: Vec<_> = config.aliases.prefixes().map(Feedback::Aliased).collect();
        Ok(Self {
            scan,
            tx,
            submitted,
            pending: HashMap::new(),
            deadlines: VecDeque::new(),
            outbox: VecDeque::new(),
            waiting: HashMap::new(),
            waiting_count: 0,
            detector: Detector::new(config)?,
            tests: HashMap::new(),
            next_id: 1,
            wire: 0,
            max_in_flight,
            receive_window,
            update: Update {
                feedback: initial,
                ..Update::default()
            },
            sent: 0,
            alias_sent: 0,
        })
    }
    pub fn aliases(&self) -> &PrefixSet {
        &self.detector.aliases
    }
    pub fn capacity(&self) -> usize {
        self.max_in_flight
            .saturating_sub(self.wire + self.waiting_count)
    }
    pub fn settled(&self) -> bool {
        self.wire == 0 && self.waiting.is_empty() && self.outbox.is_empty()
    }
    pub fn cancel(&self) {
        self.scan.cancel();
    }
    pub fn consider(&mut self, address: Ipv6Addr) {
        match self.detector.consider(address) {
            Status::Pending(prefix) => {
                if self.waiting.entry(prefix).or_default().insert(address) {
                    self.waiting_count += 1;
                }
            }
            Status::Aliased => self.report(address, Feedback::Skipped(address)),
            Status::NotDetected => self.report(address, Feedback::Active(address)),
        }
    }
    fn report(&mut self, address: Ipv6Addr, feedback: Feedback) {
        if let Some(pending) = self.pending.get_mut(&address) {
            if pending.reported {
                return;
            }
            pending.reported = true;
        }
        self.update.feedback.push(feedback);
    }
    pub fn submit(&mut self, batch: TargetBatch) {
        let mut accepted = TargetBatch::new();
        for &target in batch.as_slice() {
            if self.aliases().contains(target) {
                self.update.feedback.push(Feedback::Skipped(target));
            } else {
                accepted.push(target);
                self.pending.entry(target).or_default().copies += 1;
                self.wire += 1;
            }
        }
        if !accepted.is_empty() {
            self.outbox.push_back(accepted);
        }
    }
    fn resolve(&mut self, prefix: Ipv6Prefix) {
        if let Some(waiting) = self.waiting.remove(&prefix) {
            self.waiting_count -= waiting.len();
            for address in waiting {
                self.consider(address);
            }
        }
    }
    fn publish_aliases(&mut self) {
        for prefix in self.detector.take_aliases() {
            self.update.feedback.push(Feedback::Aliased(prefix));
            self.resolve(prefix);
        }
    }
    fn pump(&mut self) -> Result<(), Error> {
        while self.wire + 3 <= self.max_in_flight {
            let Some(test) = self.detector.next_test() else {
                break;
            };
            let batch = TargetBatch::alias(test.targets, test.token, self.next_id);
            self.next_id += 1;
            self.tests.insert(batch.id, test.prefix);
            self.wire += 3;
            self.outbox.push_front(batch);
        }
        while let Some(mut batch) = self.outbox.pop_front() {
            if batch.purpose == ProbePurpose::Discovery && !self.aliases().is_empty() {
                let mut retained = TargetBatch::new();
                for &address in batch.as_slice() {
                    if self.aliases().contains(address) {
                        self.wire -= 1;
                        self.report(address, Feedback::Skipped(address));
                        let pending = self.pending.get_mut(&address).unwrap();
                        pending.copies -= 1;
                        if pending.copies == 0 {
                            self.pending.remove(&address);
                        }
                    } else {
                        retained.push(address);
                    }
                }
                batch = retained;
                if batch.is_empty() {
                    continue;
                }
            }
            match self.tx.try_send(batch) {
                Ok(()) => {}
                Err(flume::TrySendError::Full(batch)) => {
                    self.outbox.push_front(batch);
                    break;
                }
                Err(flume::TrySendError::Disconnected(_)) => {
                    return Err(Error::NetworkError("transmit queue closed".into()));
                }
            }
        }
        Ok(())
    }
    fn reply(&mut self, received: engine::ReceivedReply) {
        let reply = received.reply;
        if received.purpose == ProbePurpose::Alias {
            if matches!(reply.kind, probe::ReplyKind::Direct) {
                self.detector.reply(reply.target, reply.token);
                if self.aliases().contains(reply.target) {
                    self.publish_aliases();
                }
            }
            return;
        }
        let Some(pending) = self.pending.get_mut(&reply.target) else {
            return;
        };
        pending.responded = true;
        let active = matches!(reply.kind, probe::ReplyKind::Direct) && !pending.active;
        pending.active |= active;
        self.update.records.push(reply_record(&reply));
        self.update.replies.push(reply);
        if active {
            self.consider(reply.target);
        }
    }
    fn expire(&mut self) {
        let now = Instant::now();
        while self
            .deadlines
            .front()
            .is_some_and(|(deadline, _)| *deadline <= now)
        {
            let (_, batch) = self.deadlines.pop_front().unwrap();
            self.wire -= batch.len();
            if batch.purpose == ProbePurpose::Alias {
                if let Some(prefix) = self.tests.remove(&batch.id) {
                    self.detector.finish(prefix);
                    self.resolve(prefix);
                }
                continue;
            }
            for target in batch.as_slice() {
                let pending = self.pending.get_mut(target).unwrap();
                pending.copies -= 1;
                if pending.copies != 0 {
                    continue;
                }
                let pending = self.pending.remove(target).unwrap();
                if !pending.active && !pending.reported {
                    self.update
                        .feedback
                        .push(if self.aliases().contains(*target) {
                            Feedback::Skipped(*target)
                        } else {
                            Feedback::Inactive(*target)
                        });
                }
                if !pending.responded {
                    self.update.records.push(ScanRecord {
                        target: *target,
                        responder: None,
                        class: OutcomeClass::Timeout,
                        success: false,
                        rtt_ms: None,
                        icmp_type: None,
                        icmp_code: None,
                        router: None,
                    });
                }
            }
        }
    }
    fn drain_replies(&mut self) -> bool {
        for _ in 0..4096 {
            match self.scan.try_next() {
                Ok(reply) => self.reply(reply),
                Err(_) => return true,
            }
        }
        false
    }
    pub fn take_update(&mut self) -> Update {
        std::mem::take(&mut self.update)
    }
    pub async fn next(&mut self) -> Result<Update, Error> {
        self.pump()?;
        if !self.update.feedback.is_empty() || !self.update.records.is_empty() {
            return Ok(self.take_update());
        }
        let deadline = self.deadlines.front().map(|(d, _)| *d);
        tokio::select! {
            reply = self.scan.next() => {
                let Some(reply) = reply else { return Err(self.scan.completion_error().await); };
                self.reply(reply);
                // Drain replies before declaring a timeout.
                self.drain_replies();
            },
            batch = self.submitted.recv() => {
                let batch = batch.ok_or_else(|| Error::NetworkError("transmit acknowledgement queue closed".into()))?;
                if batch.purpose == ProbePurpose::Alias { self.alias_sent += batch.len() as u64; } else { self.sent += batch.len() as u64; }
                self.deadlines.push_back((Instant::now() + self.receive_window, batch));
            },
            _ = async { match deadline { Some(d) => tokio::time::sleep_until(d).await, None => std::future::pending().await } } => {
                if self.drain_replies() { self.expire(); }
            },
            _ = tokio::time::sleep(Duration::from_millis(1)), if !self.outbox.is_empty() => {},
        }
        Ok(self.take_update())
    }
    pub async fn finish(self) -> Result<(crate::finite::Summary, dealias::Report), Error> {
        drop(self.tx);
        let mut summary = self.scan.join().await?;
        if summary.stats.sent != self.sent + self.alias_sent {
            return Err(Error::InternalError(
                "transmit acknowledgements do not match sent probes".into(),
            ));
        }
        summary.stats.sent = self.sent;
        Ok((
            summary,
            dealias::Report {
                aliases: self.detector.aliases,
                probes_sent: self.alias_sent,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn session(config: dealias::Config) -> Session {
        let (tx, targets) = flume::bounded(2);
        let (submitted, acknowledgements) = tokio::sync::mpsc::channel(2);
        let job = engine::test_session(engine::BatchInput { targets, submitted });
        Session::new(
            job,
            tx,
            acknowledgements,
            Duration::from_millis(1),
            32,
            config,
        )
        .unwrap()
    }
    fn reply(target: Ipv6Addr, purpose: ProbePurpose, token: u8) -> engine::ReceivedReply {
        engine::ReceivedReply {
            purpose,
            reply: probe::Reply {
                target,
                responder: target,
                token,
                kind: probe::ReplyKind::Direct,
            },
        }
    }
    #[tokio::test]
    async fn only_acknowledged_batches_expire() {
        let mut s = session(dealias::Config::default());
        let mut batch = TargetBatch::new();
        batch.push(Ipv6Addr::LOCALHOST);
        s.submit(batch);
        s.expire();
        assert!(s.take_update().feedback.is_empty());
        assert_eq!(s.wire, 1);
        let batch = s.outbox.pop_front().unwrap();
        s.deadlines.push_back((Instant::now(), batch));
        s.expire();
        let update = s.take_update();
        assert_eq!(
            update.feedback,
            vec![Feedback::Inactive(Ipv6Addr::LOCALHOST)]
        );
        assert_eq!(update.records.len(), 1);
        assert!(s.settled());
    }
    #[tokio::test]
    async fn reply_before_acknowledgement_prevents_negative_feedback() {
        let mut s = session(dealias::Config::default());
        let mut batch = TargetBatch::new();
        batch.push(Ipv6Addr::LOCALHOST);
        s.submit(batch);
        s.reply(reply(Ipv6Addr::LOCALHOST, ProbePurpose::Discovery, 0));
        s.reply(reply(Ipv6Addr::LOCALHOST, ProbePurpose::Discovery, 0));
        let batch = s.outbox.pop_front().unwrap();
        s.deadlines.push_back((Instant::now(), batch));
        s.expire();
        let update = s.take_update();
        assert_eq!(update.feedback, vec![Feedback::Active(Ipv6Addr::LOCALHOST)]);
        assert!(update.records.iter().all(|r| r.success));
    }
    #[tokio::test]
    async fn alias_samples_are_isolated_and_resolve_to_skipped() {
        let mut s = session(dealias::Config {
            online: Some(dealias::OnlineConfig {
                prefix_lengths: vec![120],
                seed: 0,
            }),
            ..Default::default()
        });
        let address = "2001:db8::42".parse().unwrap();
        let mut batch = TargetBatch::new();
        batch.push(address);
        s.submit(batch);
        s.reply(reply(address, ProbePurpose::Discovery, 0));
        assert!(s.take_update().feedback.is_empty());
        let test = s.detector.next_test().unwrap();
        for target in test.targets {
            s.reply(reply(target, ProbePurpose::Discovery, test.token));
        }
        assert!(s.aliases().is_empty());
        let mut error_reply = reply(test.targets[0], ProbePurpose::Alias, test.token);
        error_reply.reply.kind = probe::ReplyKind::IcmpError {
            icmp_type: 1,
            icmp_code: 0,
        };
        s.reply(error_reply);
        s.reply(reply(test.targets[0], ProbePurpose::Alias, test.token));
        s.reply(reply(test.targets[0], ProbePurpose::Alias, test.token));
        assert!(s.take_update().feedback.is_empty());
        s.reply(reply(test.targets[1], ProbePurpose::Alias, test.token));
        let update = s.take_update();
        assert_eq!(
            update.feedback,
            vec![Feedback::Aliased(test.prefix), Feedback::Skipped(address)]
        );
        assert!(update.records.is_empty());
        assert!(s.waiting.is_empty());
    }
    #[tokio::test]
    async fn shared_session_drains_alias_probes_and_counts_them_separately() {
        let mut s = session(dealias::Config {
            online: Some(dealias::OnlineConfig {
                prefix_lengths: vec![120],
                seed: 0,
            }),
            ..Default::default()
        });
        let mut targets = TargetBatch::new();
        for n in 1..=6u128 {
            targets.push(n.into());
        }
        s.submit(targets);
        let mut feedback = Vec::new();
        let mut records = Vec::new();
        tokio::time::timeout(Duration::from_secs(2), async {
            while !s.settled() {
                let update = s.next().await.unwrap();
                feedback.extend(update.feedback);
                records.extend(update.records);
            }
        })
        .await
        .unwrap();
        let (_, report) = s.finish().await.unwrap();
        assert_eq!(report.probes_sent, 3);
        assert_eq!(records.len(), 6);
        assert_eq!(
            feedback
                .iter()
                .filter(|f| matches!(f, Feedback::Skipped(_)))
                .count(),
            6
        );
        assert!(
            !feedback
                .iter()
                .any(|f| matches!(f, Feedback::Active(_) | Feedback::Inactive(_)))
        );
    }
    #[tokio::test]
    async fn one_reply_per_test_releases_active_after_all_lengths_finish() {
        let (tx, targets) = flume::bounded(2);
        let (submitted, acknowledgements) = tokio::sync::mpsc::channel(2);
        let job = engine::test_session_replies(engine::BatchInput { targets, submitted }, 1);
        let mut s = Session::new(
            job,
            tx,
            acknowledgements,
            Duration::from_millis(1),
            32,
            dealias::Config {
                online: Some(dealias::OnlineConfig {
                    prefix_lengths: vec![112, 120],
                    seed: 0,
                }),
                ..Default::default()
            },
        )
        .unwrap();
        let address = Ipv6Addr::LOCALHOST;
        let mut targets = TargetBatch::new();
        targets.push(address);
        s.submit(targets);
        let mut feedback = Vec::new();
        tokio::time::timeout(Duration::from_secs(2), async {
            while !s.settled() {
                feedback.extend(s.next().await.unwrap().feedback);
            }
        })
        .await
        .unwrap();
        assert_eq!(feedback, vec![Feedback::Active(address)]);
        let (summary, report) = s.finish().await.unwrap();
        assert_eq!(report.probes_sent, 6);
        assert!(report.aliases.is_empty());
        assert_eq!(summary.stats.sent, 1);
        assert_eq!(summary.stats.replies, 1);
    }
    #[tokio::test]
    async fn unexpected_network_failure_is_not_negative_evidence() {
        let (tx, _targets) = flume::bounded(2);
        let (_submitted, acknowledgements) = tokio::sync::mpsc::channel(2);
        let job = engine::test_job(0, true);
        let mut s = Session::new(
            job,
            tx,
            acknowledgements,
            Duration::ZERO,
            32,
            dealias::Config {
                online: Some(dealias::OnlineConfig::default()),
                ..Default::default()
            },
        )
        .unwrap();
        s.consider(Ipv6Addr::LOCALHOST);
        let error = s.next().await.err().unwrap();
        assert!(error.to_string().contains("test failure"));
        assert!(s.take_update().feedback.is_empty());
        assert!(!s.settled());
    }
}
