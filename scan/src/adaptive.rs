use crate::engine::{
    self, BatchInput,
    target_batch::{TARGET_BATCH_SIZE, TargetBatch},
};
use crate::finite;
use sixseven_core::{
    Feedback, GenerationOptions, GenerationState, StoredModel, TgaError,
    generation::{CandidateFilter, validate_generated},
};
use sixseven_formats::csv::scan::ScanRecord;
use std::{net::Ipv6Addr, num::NonZeroUsize, time::Duration};

pub struct Config {
    pub interface: link::Interface,
    pub probe: probe::PreparedProbe,
    pub pps: usize,
    pub network_workers: usize,
    pub receive_window: Duration,
    pub batch_size: NonZeroUsize,
    pub max_in_flight: NonZeroUsize,
    pub generation: GenerationOptions,
    pub dealias: crate::dealias::Config,
}

pub enum Event {
    Records(Vec<ScanRecord>),
    Feedback(Vec<Feedback>),
}

pub struct Result {
    pub model: Box<dyn StoredModel>,
    pub generated: u64,
    pub sent: u64,
    pub dealias: crate::dealias::Report,
}

pub async fn run(
    model: Box<dyn StoredModel>,
    config: Config,
    output: tokio::sync::mpsc::Sender<Event>,
) -> std::result::Result<Result, TgaError> {
    if config.dealias.online.is_some() && config.max_in_flight.get() < 3 {
        return Err(TgaError::Config(
            "active dealiasing requires max-in-flight of at least 3".into(),
        ));
    }
    if config.generation.count == 0 {
        return Ok(Result {
            model,
            generated: 0,
            sent: 0,
            dealias: crate::dealias::Report {
                aliases: config.dealias.aliases,
                probes_sent: 0,
            },
        });
    }
    let scan_config = finite::Config::from_source(
        sixseven_core::source::addresses([]),
        config.interface,
        config.probe,
    )
    .dealias(config.dealias.clone())
    .pps(config.pps)
    .network_workers(config.network_workers)
    .receive_window(Duration::ZERO);
    let (target_tx, target_rx) = flume::bounded(64);
    let (submitted_tx, submitted_rx) = tokio::sync::mpsc::channel(64);
    let scan = engine::spawn_inner(
        scan_config,
        Some(BatchInput {
            targets: target_rx,
            submitted: submitted_tx,
        }),
    )
    .await
    .map_err(error)?;
    drive(
        model,
        Limits {
            receive_window: config.receive_window,
            batch_size: config.batch_size,
            max_in_flight: config.max_in_flight,
            generation: config.generation,
            dealias: config.dealias,
        },
        scan,
        target_tx,
        submitted_rx,
        output,
    )
    .await
}

struct Limits {
    dealias: crate::dealias::Config,
    receive_window: Duration,
    batch_size: NonZeroUsize,
    max_in_flight: NonZeroUsize,
    generation: GenerationOptions,
}

async fn drive(
    mut model: Box<dyn StoredModel>,
    config: Limits,
    scan: engine::ScanJob,
    target_tx: flume::Sender<TargetBatch>,
    submitted_rx: tokio::sync::mpsc::Receiver<TargetBatch>,
    output: tokio::sync::mpsc::Sender<Event>,
) -> std::result::Result<Result, TgaError> {
    model.set_budget(config.generation.count);
    let mut generation = config.generation;
    if model.allows_repeated_probes() {
        generation.unique = false;
    }
    let mut filter = CandidateFilter::new(generation);
    let mut model = Some(model);
    let mut work = tokio::task::JoinSet::new();
    let mut session = crate::session::Session::new(
        scan,
        target_tx,
        submitted_rx,
        config.receive_window,
        config.max_in_flight.get(),
        config.dealias,
    )
    .map_err(error)?;
    let mut feedback = Vec::new();
    let mut state = GenerationState::Ready;
    let mut empty_round = false;
    let mut buffer = Some(vec![
        [0; 16];
        config.batch_size.get().min(TARGET_BATCH_SIZE)
    ]);
    loop {
        let update = session.take_update();
        feedback.extend(update.feedback);
        if !update.records.is_empty() {
            output
                .send(Event::Records(update.records))
                .await
                .map_err(error)?;
        }
        if work.is_empty() {
            if state == GenerationState::AwaitingFeedback && session.settled() {
                feedback.push(Feedback::BatchComplete);
                state = GenerationState::Ready;
            }
            let remaining = filter.remaining();
            if (state == GenerationState::Exhausted || remaining == 0)
                && session.settled()
                && feedback.is_empty()
            {
                break;
            }
            let limit = (state == GenerationState::Ready && feedback.len() <= 4096)
                .then(|| {
                    NonZeroUsize::new(
                        remaining
                            .min(session.capacity())
                            .min(config.batch_size.get())
                            .min(TARGET_BATCH_SIZE),
                    )
                })
                .flatten();
            if limit.is_some() || !feedback.is_empty() {
                let mut owned = model.take().expect("idle model is available");
                let mut addresses = buffer.take().expect("idle generation buffer is available");
                let applied: Vec<_> = feedback.drain(..feedback.len().min(4096)).collect();
                work.spawn(async move {
                    crate::blocking::run(move || {
                        if !applied.is_empty() {
                            owned.apply_feedback(&applied)?;
                        }
                        let batch = limit
                            .map(|limit| {
                                let batch = owned.generate(&mut addresses[..limit.get()])?;
                                validate_generated(batch, limit.get())?;
                                Ok::<_, TgaError>(batch)
                            })
                            .transpose()?;
                        Ok::<_, TgaError>((owned, addresses, batch, applied))
                    })
                    .await
                    .map_err(error)?
                });
            }
        }
        tokio::select! {
            update = session.next() => {
                let update = update.map_err(error)?;
                feedback.extend(update.feedback);
                if !update.records.is_empty() { output.send(Event::Records(update.records)).await.map_err(error)?; }
            },
            completed = work.join_next(), if !work.is_empty() => {
                let (returned, addresses, batch, applied) = completed.expect("model task exists").map_err(error)??;
                model = Some(returned); buffer = Some(addresses);
                if !applied.is_empty() { output.send(Event::Feedback(applied)).await.map_err(error)?; }
                if let Some(batch) = batch {
                    state = batch.state;
                    if batch.written == 0 && state == GenerationState::AwaitingFeedback {
                        if empty_round { return Err(error("model repeatedly requests feedback without targets")); }
                        empty_round = true;
                    } else { empty_round = false; }
                    let mut targets = TargetBatch::new();
                    for &address in &buffer.as_ref().unwrap()[..batch.written] {
                        let target = Ipv6Addr::from(address);
                        if filter.accept_excluding(address, session.aliases())? {
                            targets.push(target); filter.written();
                        } else { feedback.push(Feedback::Skipped(target)); }
                    }
                    if !targets.is_empty() { session.submit(targets); }
                }
            },
        }
    }
    let sent = session.sent;
    let (_, dealias) = session.finish().await.map_err(error)?;
    Ok(Result {
        model: model.expect("completed model is available"),
        generated: filter.stats.attempts,
        sent,
        dealias,
    })
}

fn error(value: impl std::fmt::Display) -> TgaError {
    TgaError::Feedback(value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sixseven_core::{Address, AlgorithmId, Generated, TargetModel};
    use std::sync::{Arc, Mutex};

    struct Model {
        next: u8,
        repeat: bool,
        budget: Option<usize>,
        feedback: Arc<Mutex<Vec<Feedback>>>,
    }
    impl TargetModel for Model {
        fn allows_repeated_probes(&self) -> bool {
            self.repeat
        }
        fn set_budget(&mut self, budget: usize) {
            self.budget = Some(budget);
        }
        fn generate(&mut self, output: &mut [Address]) -> std::result::Result<Generated, TgaError> {
            assert_eq!(self.budget, Some(6));
            let mut written = 0;
            let end = if self.repeat && self.next < 3 { 3 } else { 6 };
            for slot in output {
                if self.next >= end {
                    break;
                }
                let mut address = [0; 16];
                address[15] = if self.repeat {
                    self.next % 3
                } else {
                    self.next
                };
                self.next += 1;
                *slot = address;
                written += 1;
            }
            Ok(Generated {
                written,
                state: if self.next == end {
                    GenerationState::AwaitingFeedback
                } else {
                    GenerationState::Ready
                },
            })
        }
        fn apply_feedback(&mut self, feedback: &[Feedback]) -> std::result::Result<(), TgaError> {
            self.feedback.lock().unwrap().extend_from_slice(feedback);
            Ok(())
        }
    }
    impl StoredModel for Model {
        fn algorithm_id(&self) -> AlgorithmId {
            AlgorithmId::new("test").unwrap()
        }
        fn encode(&self) -> std::result::Result<Vec<u8>, TgaError> {
            Ok(vec![])
        }
    }

    #[tokio::test]
    async fn sixtree_alias_probes_are_repeated_and_budgeted_through_the_adaptive_session() {
        let registry = tga::builtin_registry();
        let artifact = registry
            .train(
                &tga::AlgorithmSpec {
                    algorithm: AlgorithmId::new("6tree").unwrap(),
                    config: "{}".parse().unwrap(),
                },
                &[tga::Observation {
                    address: 1u128.to_be_bytes(),
                    active: true,
                }],
            )
            .unwrap();
        let model = registry.open(&artifact).unwrap();
        let (targets, input) = flume::bounded(2);
        let (submitted, acknowledgements) = tokio::sync::mpsc::channel(2);
        let scan = engine::test_session(BatchInput {
            targets: input,
            submitted,
        });
        let (output, mut events) = tokio::sync::mpsc::channel(32);
        let collected = tokio::spawn(async move {
            let mut records = Vec::new();
            let mut feedback = Vec::new();
            while let Some(event) = events.recv().await {
                match event {
                    Event::Records(batch) => records.extend(batch),
                    Event::Feedback(batch) => feedback.extend(batch),
                }
            }
            (records, feedback)
        });
        let result = tokio::time::timeout(
            Duration::from_secs(5),
            drive(
                model,
                Limits {
                    dealias: crate::dealias::Config::default(),
                    receive_window: Duration::from_millis(5),
                    batch_size: NonZeroUsize::new(16).unwrap(),
                    max_in_flight: NonZeroUsize::new(128).unwrap(),
                    generation: GenerationOptions {
                        count: 1024,
                        max_attempts: 1024,
                        unique: true,
                        exclude: vec![],
                    },
                },
                scan,
                targets,
                acknowledgements,
                output,
            ),
        )
        .await
        .unwrap()
        .unwrap();
        let (records, feedback) = collected.await.unwrap();
        assert_eq!(result.sent, 1024);
        assert_eq!(records.len(), 1024);
        let unique: std::collections::HashSet<_> =
            records.iter().map(|record| record.target).collect();
        assert!(unique.len() < records.len());
        assert_eq!(
            feedback
                .iter()
                .filter(|item| matches!(item, Feedback::Active(_)))
                .count(),
            1024
        );
        assert!(
            !feedback
                .iter()
                .any(|item| matches!(item, Feedback::Skipped(_) | Feedback::Inactive(_)))
        );
        assert!(result.model.detected_aliases().is_empty());
    }

    #[tokio::test]
    async fn persistent_session_delivers_feedback_and_completes_the_round() {
        for repeat in [false, true] {
            let feedback = Arc::new(Mutex::new(Vec::new()));
            let model = Box::new(Model {
                next: 0,
                repeat,
                budget: None,
                feedback: feedback.clone(),
            });
            let (targets, input) = flume::bounded(2);
            let (submitted, acknowledgements) = tokio::sync::mpsc::channel(2);
            let scan = engine::test_session(BatchInput {
                targets: input,
                submitted,
            });
            let (output, mut events) = tokio::sync::mpsc::channel(32);
            let result = tokio::time::timeout(
                Duration::from_secs(2),
                drive(
                    model,
                    Limits {
                        dealias: crate::dealias::Config::default(),
                        receive_window: Duration::from_millis(5),
                        batch_size: NonZeroUsize::new(2).unwrap(),
                        max_in_flight: NonZeroUsize::new(4).unwrap(),
                        generation: GenerationOptions {
                            count: 6,
                            max_attempts: 6,
                            unique: true,
                            exclude: vec![],
                        },
                    },
                    scan,
                    targets,
                    acknowledgements,
                    output,
                ),
            )
            .await
            .unwrap()
            .unwrap();
            assert_eq!(result.sent, 6);
            let feedback = feedback.lock().unwrap().clone();
            assert_eq!(
                feedback
                    .iter()
                    .filter(|value| matches!(value, Feedback::Active(_)))
                    .count(),
                6
            );
            assert_eq!(feedback.last(), Some(&Feedback::BatchComplete));
            assert!(
                !feedback
                    .iter()
                    .any(|value| matches!(value, Feedback::Inactive(_)))
            );
            let mut records = 0;
            while let Some(event) = events.recv().await {
                if let Event::Records(batch) = event {
                    records += batch.len();
                }
            }
            assert_eq!(records, 6);
        }
    }
    #[tokio::test]
    async fn paper_dealiasing_finishes_before_model_feedback_and_journal_replays() {
        let feedback = Arc::new(Mutex::new(Vec::new()));
        let model = Box::new(Model {
            next: 0,
            repeat: false,
            budget: None,
            feedback: feedback.clone(),
        });
        let (targets, input) = flume::bounded(2);
        let (submitted, acknowledgements) = tokio::sync::mpsc::channel(2);
        let scan = engine::test_session(BatchInput {
            targets: input,
            submitted,
        });
        let (output, mut events) = tokio::sync::mpsc::channel(32);
        let result = tokio::time::timeout(
            Duration::from_secs(2),
            drive(
                model,
                Limits {
                    receive_window: Duration::from_millis(1),
                    batch_size: NonZeroUsize::new(6).unwrap(),
                    max_in_flight: NonZeroUsize::new(32).unwrap(),
                    generation: GenerationOptions {
                        count: 6,
                        max_attempts: 6,
                        unique: true,
                        exclude: vec![],
                    },
                    dealias: crate::dealias::Config {
                        online: Some(crate::dealias::OnlineConfig {
                            prefix_lengths: vec![120],
                            seed: 0,
                        }),
                        ..Default::default()
                    },
                },
                scan,
                targets,
                acknowledgements,
                output,
            ),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(result.sent, 6);
        assert_eq!(result.dealias.probes_sent, 3);
        let mut journal = Vec::new();
        while let Some(event) = events.recv().await {
            if let Event::Feedback(batch) = event {
                sixseven_formats::feedback::write_batch(&mut journal, &batch).unwrap();
            }
        }
        let replayed: Vec<Feedback> = String::from_utf8(journal)
            .unwrap()
            .lines()
            .flat_map(|line| serde_json::from_str::<Vec<Feedback>>(line).unwrap())
            .collect();
        assert_eq!(replayed, *feedback.lock().unwrap());
        assert_eq!(replayed.last(), Some(&Feedback::BatchComplete));
        assert_eq!(
            replayed
                .iter()
                .filter(|f| matches!(f, Feedback::Skipped(_)))
                .count(),
            6
        );
        assert!(
            !replayed
                .iter()
                .any(|f| matches!(f, Feedback::Active(_) | Feedback::Inactive(_)))
        );
        assert!(matches!(replayed[0], Feedback::Aliased(_)));
    }
}
