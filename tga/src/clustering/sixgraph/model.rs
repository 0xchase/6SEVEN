use super::Pattern;
use super::generation::sample_round;
use super::pattern_index::PatternIndex;
use crate::{Address, Feedback, Generated, GenerationState, Ipv6Prefix, TargetModel, TgaError};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::Arc;

#[derive(Debug, Clone, Default)]
pub struct SixGraphModel {
    pub(super) patterns: Arc<[Pattern]>,
    pub(super) scan: ScanState,
    pub(super) outlier_count: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(super) struct ScanState {
    pub(super) known: BTreeSet<Address>,
    pub(super) aliases: Vec<Ipv6Prefix>,
    pub(super) initial_counts: Vec<usize>,
    pub(super) sample_seed: u64,
    pub(super) round: usize,
    pub(super) targets: Vec<Address>,
    next_target: usize,
    issued: BTreeSet<Address>,
    active: BTreeSet<Address>,
}

impl SixGraphModel {
    pub(super) fn new(
        patterns: Vec<Pattern>,
        known: BTreeSet<Address>,
        outlier_count: usize,
        sample_seed: u64,
    ) -> Self {
        let initial_counts = patterns.iter().map(Pattern::seed_count).collect();
        let mut model = Self {
            patterns: patterns.into(),
            scan: ScanState {
                known,
                initial_counts,
                sample_seed,
                ..Default::default()
            },
            outlier_count,
        };
        model.prepare_round();
        model
    }

    /// Count the unique active training addresses including final outliers.
    pub fn seed_count(&self) -> usize {
        self.scan
            .initial_counts
            .iter()
            .sum::<usize>()
            .saturating_add(self.outlier_count)
    }

    /// Count the sampled candidates in the current scanning round.
    pub fn sampled_target_count(&self) -> usize {
        self.scan.targets.len()
    }

    fn prepare_round(&mut self) {
        self.scan.targets = sample_round(&self.patterns, &self.scan);
        self.scan.next_target = 0;
    }

    fn complete_round(&mut self) {
        self.scan.known.append(&mut self.scan.issued);
        self.attach_active();
        self.scan.round = self.scan.round.saturating_add(1);
        self.prepare_round();
    }

    fn attach_active(&mut self) {
        if self.scan.active.is_empty() {
            return;
        }
        let index = PatternIndex::new(&self.patterns);
        let mut additions = BTreeMap::<usize, Vec<Address>>::new();
        for &address in &self.scan.active {
            if self.scan.is_aliased(&address) {
                continue;
            }
            for id in index.matches(address) {
                additions.entry(id).or_default().push(address);
            }
        }
        if !additions.is_empty() {
            let patterns = Arc::make_mut(&mut self.patterns);
            for (id, addresses) in additions {
                let pattern = &mut patterns[id];
                let existing = pattern.seeds.iter().copied().collect::<HashSet<_>>();
                pattern
                    .seeds
                    .extend(addresses.into_iter().filter(|ip| !existing.contains(ip)));
            }
        }
        self.scan.active.clear();
    }
}

impl ScanState {
    pub(super) fn is_aliased(&self, address: &Address) -> bool {
        let ip = std::net::Ipv6Addr::from(*address);
        self.aliases.iter().any(|prefix| prefix.contains(&ip))
    }

    pub(super) fn excluded(&self, address: &Address) -> bool {
        self.known.contains(address) || self.issued.contains(address) || self.is_aliased(address)
    }
}

impl TargetModel for SixGraphModel {
    fn generate(&mut self, output: &mut [Address]) -> Result<Generated, TgaError> {
        sixseven_core::generation::validate_output(output)?;
        let mut written = 0;
        while written < output.len() && self.scan.next_target < self.scan.targets.len() {
            let address = self.scan.targets[self.scan.next_target];
            self.scan.next_target += 1;
            if !self.scan.excluded(&address) {
                self.scan.issued.insert(address);
                output[written] = address;
                written += 1;
            }
        }
        let state = if self.scan.targets.is_empty() {
            GenerationState::Exhausted
        } else if self.scan.next_target == self.scan.targets.len() {
            GenerationState::AwaitingFeedback
        } else {
            GenerationState::Ready
        };
        Ok(Generated { written, state })
    }

    fn apply_feedback(&mut self, feedback: &[Feedback]) -> Result<(), TgaError> {
        for event in feedback {
            match event {
                Feedback::Active(ip) => {
                    self.scan.known.insert(ip.octets());
                    self.scan.active.insert(ip.octets());
                }
                Feedback::Inactive(ip) | Feedback::Skipped(ip) => {
                    self.scan.known.insert(ip.octets());
                }
                Feedback::Aliased(prefix) => {
                    if !self.scan.aliases.iter().any(|known| known.contains(prefix)) {
                        self.scan.aliases.retain(|known| !prefix.contains(known));
                        self.scan.aliases.push(*prefix);
                        self.scan.aliases.sort_unstable();
                    }
                }
                Feedback::BatchComplete => self.complete_round(),
            }
        }
        Ok(())
    }
}

#[derive(Deserialize)]
struct StoredModel {
    patterns: Vec<Pattern>,
    scan: ScanState,
    outlier_count: usize,
}

impl Serialize for SixGraphModel {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct ModelRef<'a> {
            patterns: &'a [Pattern],
            scan: &'a ScanState,
            outlier_count: usize,
        }
        ModelRef {
            patterns: &self.patterns,
            scan: &self.scan,
            outlier_count: self.outlier_count,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SixGraphModel {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let stored = StoredModel::deserialize(deserializer)?;
        if stored.scan.next_target > stored.scan.targets.len()
            || stored.scan.initial_counts.len() != stored.patterns.len()
            || stored
                .scan
                .initial_counts
                .iter()
                .zip(&stored.patterns)
                .any(|(&count, pattern)| count < 2 || count > pattern.seeds.len())
            || stored.patterns.iter().any(|pattern| !pattern.is_valid())
        {
            return Err(serde::de::Error::custom("invalid 6Graph model state"));
        }
        Ok(Self {
            patterns: stored.patterns.into(),
            scan: stored.scan,
            outlier_count: stored.outlier_count,
        })
    }
}

impl std::fmt::Display for SixGraphModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "6Graph: {} patterns, {} outliers, {} initial seeds, round {}, {} sampled targets",
            self.patterns.len(),
            self.outlier_count,
            self.seed_count(),
            self.scan.round.saturating_add(1),
            self.sampled_target_count()
        )
    }
}
