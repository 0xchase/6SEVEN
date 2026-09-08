//! Optional prescanning followed by descending hit rate generation.
use super::sampling::{RandomDraws, SplitMix64};
use super::{SixForestStream, SixForestTargetSet};
use crate::{Address, Feedback, Generated, GenerationState};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, net::Ipv6Addr};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct Prescan {
    stream: SixForestStream,
    order: Vec<usize>,
    cursor: usize,
    quotas: Vec<usize>,
    hits: Vec<usize>,
    trials: Vec<usize>,
    pending: BTreeMap<Ipv6Addr, Vec<usize>>,
    aliases: Vec<crate::Ipv6Prefix>,
    phase: Phase,
    draws: Vec<RandomDraws>,
    rng: SplitMix64,
    started: bool,
    sample_limit: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Phase {
    Sampling,
    Ranked,
}

impl Prescan {
    pub(super) fn new(space: &SixForestTargetSet, samples: usize) -> Self {
        let stream = space.stream();
        Self {
            order: stream.region_order(),
            stream,
            cursor: 0,
            quotas: space
                .regions
                .iter()
                .map(|region| {
                    if region.free_nibbles() <= 3 {
                        4096
                    } else {
                        samples
                    }
                })
                .collect(),
            hits: vec![0; space.regions.len()],
            trials: vec![0; space.regions.len()],
            pending: BTreeMap::new(),
            aliases: Vec::new(),
            phase: Phase::Sampling,
            draws: space
                .regions
                .iter()
                .map(|region| RandomDraws::new(region.free_bits()))
                .collect(),
            rng: SplitMix64::new(space.generation_seed),
            started: false,
            sample_limit: samples,
        }
    }

    pub(super) fn set_budget(&mut self, budget: usize) {
        if self.started {
            return;
        }
        let large: Vec<_> = self
            .order
            .iter()
            .copied()
            .filter(|&i| self.stream.needs_prescan(i))
            .collect();
        if large.is_empty() {
            return;
        }
        let allowance = budget.saturating_sub(1) / 100;
        let base = allowance / large.len();
        let extra = allowance % large.len();
        for (position, index) in large.into_iter().enumerate() {
            self.quotas[index] = self.sample_limit.min(base + usize::from(position < extra));
        }
    }

    pub(super) fn generate(&mut self, output: &mut [Address]) -> Generated {
        self.started = true;
        let mut written = 0;
        while written < output.len() {
            if self.cursor == self.order.len() {
                if self.phase == Phase::Sampling {
                    if !self.pending.is_empty() {
                        return Generated {
                            written,
                            state: GenerationState::AwaitingFeedback,
                        };
                    }
                    self.rank();
                } else {
                    return Generated {
                        written,
                        state: GenerationState::Exhausted,
                    };
                }
            }
            let Some(&region) = self.order.get(self.cursor) else {
                return Generated {
                    written,
                    state: GenerationState::Exhausted,
                };
            };
            if self.phase == Phase::Sampling && self.quotas[region] == 0 {
                self.cursor += 1;
                continue;
            }
            let candidate = if self.phase == Phase::Sampling && self.stream.needs_prescan(region) {
                self.stream
                    .sample_region(region, &mut self.draws[region], &mut self.rng)
            } else {
                self.stream.draw_region(region)
            };
            let Some(address) = candidate else {
                self.cursor += 1;
                continue;
            };
            let ip = Ipv6Addr::from(address);
            if self.aliases.iter().any(|prefix| prefix.contains(&ip)) {
                continue;
            }
            if self.phase == Phase::Sampling {
                self.quotas[region] -= 1;
                if self.stream.needs_prescan(region) {
                    self.pending.entry(ip).or_default().push(region);
                }
            }
            output[written] = address;
            written += 1;
        }
        Generated {
            written,
            state: GenerationState::Ready,
        }
    }

    pub(super) fn apply_feedback(&mut self, feedback: &[Feedback]) {
        for item in feedback {
            match item {
                Feedback::Active(ip) | Feedback::Inactive(ip) => {
                    if let Some(regions) = self.pending.remove(ip) {
                        for region in regions {
                            self.trials[region] += 1;
                            self.hits[region] += usize::from(matches!(item, Feedback::Active(_)));
                        }
                    }
                }
                Feedback::Skipped(ip) => {
                    if let Some(regions) = self.pending.remove(ip) {
                        for region in regions {
                            self.quotas[region] += 1;
                        }
                        self.cursor = 0;
                    }
                }
                Feedback::Aliased(prefix) => {
                    self.stream.exclude_prefix(prefix);
                    if !self.aliases.contains(prefix) {
                        self.aliases.push(*prefix);
                    }
                    self.pending.retain(|ip, _| !prefix.contains(ip));
                }
                Feedback::BatchComplete => {}
            }
        }
        // Completion applies after all explicit outcomes, regardless of order.
        if feedback.contains(&Feedback::BatchComplete) {
            for regions in std::mem::take(&mut self.pending).into_values() {
                for region in regions {
                    self.trials[region] += 1;
                }
            }
        }
    }

    fn rank(&mut self) {
        self.order.sort_by(|&a, &b| {
            let left = self.hits[a] as u128 * self.trials[b].max(1) as u128;
            let right = self.hits[b] as u128 * self.trials[a].max(1) as u128;
            right.cmp(&left)
        });
        self.draws.clear();
        self.phase = Phase::Ranked;
        self.cursor = 0;
    }
}

#[derive(Deserialize)]
pub(super) struct LegacyPrescan {
    stream: SixForestStream,
    order: Vec<usize>,
    cursor: usize,
    quotas: Vec<usize>,
    hits: Vec<usize>,
    trials: Vec<usize>,
    pending: BTreeMap<Ipv6Addr, Vec<usize>>,
    aliases: Vec<crate::Ipv6Prefix>,
    phase: Phase,
}

impl LegacyPrescan {
    pub(super) fn migrate(mut self, space: &SixForestTargetSet) -> Prescan {
        let started = self.phase == Phase::Ranked || self.stream.has_drawn();
        let sample_limit = self
            .order
            .iter()
            .copied()
            .filter(|&index| self.stream.needs_prescan(index))
            .map(|index| self.quotas[index])
            .max()
            .unwrap_or(0);
        if self.phase == Phase::Sampling {
            self.stream.exclude_previous_samples();
        }
        Prescan {
            stream: self.stream,
            order: self.order,
            cursor: self.cursor,
            quotas: self.quotas,
            hits: self.hits,
            trials: self.trials,
            pending: self.pending,
            aliases: self.aliases,
            phase: self.phase,
            draws: space
                .regions
                .iter()
                .map(|region| RandomDraws::new(region.free_bits()))
                .collect(),
            rng: SplitMix64::new(space.generation_seed),
            started,
            sample_limit,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Algorithm, SixForest, TargetModel};

    fn space() -> SixForestTargetSet {
        let regions = [0x20, 0x30, 0x40].map(|prefix| {
            let mut base = [0; 16];
            base[0] = prefix;
            super::super::SixForestRegion {
                base,
                free_dims: vec![28, 29, 30, 31],
                seed_count: 2,
            }
        });
        SixForestTargetSet::new(regions.into(), Vec::new(), 0)
    }

    #[test]
    fn sample_allocation_is_balanced_and_strictly_below_one_percent() {
        for budget in [0, 1, 99, 100, 101, 200, 301, 10_000, usize::MAX] {
            let mut prescan = Prescan::new(&space(), 100);
            prescan.set_budget(budget);
            let total: usize = prescan.quotas.iter().sum();
            assert!(total <= budget.saturating_sub(1) / 100);
            assert!(
                prescan.quotas.iter().max().unwrap() - prescan.quotas.iter().min().unwrap() <= 1
            );
            assert!(total <= 300);
        }
    }

    #[test]
    fn budget_changes_do_not_restart_a_pending_prescan() {
        let mut prescan = Prescan::new(&space(), 100);
        prescan.set_budget(1_001);
        let batch = prescan.generate(&mut [[0; 16]; 128]);
        assert_eq!(batch.written, 10);
        assert_eq!(batch.state, GenerationState::AwaitingFeedback);
        prescan.set_budget(100_000);
        assert_eq!(prescan.generate(&mut [[0; 16]; 128]).written, 0);
    }

    #[test]
    fn skipped_samples_are_replaced_before_ranking() {
        let mut prescan = Prescan::new(&space(), 1);
        let mut output = [[0; 16]; 8];
        let batch = prescan.generate(&mut output);
        assert_eq!(batch.written, 3);
        let skipped = Ipv6Addr::from(output[0]);
        prescan.apply_feedback(&[Feedback::Skipped(skipped), Feedback::BatchComplete]);
        let next = prescan.generate(&mut output);
        assert_eq!(next.written, 1);
        assert_eq!(next.state, GenerationState::AwaitingFeedback);
        assert_ne!(Ipv6Addr::from(output[0]), skipped);
        assert_eq!(prescan.trials.iter().sum::<usize>(), 2);
    }

    #[test]
    fn v2_pending_prescan_migrates_without_resampling_its_previous_targets() {
        let space = space();
        let mut legacy = Prescan::new(&space, 2);
        let region = legacy.order[0];
        let previous = legacy.stream.draw_region(region).unwrap();
        legacy.quotas[region] -= 1;
        legacy.pending.insert(previous.into(), vec![region]);
        let fields = (
            &legacy.stream,
            &legacy.order,
            legacy.cursor,
            &legacy.quotas,
            &legacy.hits,
            &legacy.trials,
            &legacy.pending,
            &legacy.aliases,
            legacy.phase,
        );
        let bytes = bincode::serialize(&(&space, Some(fields), 6usize, 0usize)).unwrap();
        let mut model = SixForest::decode_model(2, &bytes).unwrap();
        let mut output = [[0; 16]; 16];
        let batch = model.generate(&mut output).unwrap();
        assert_eq!(batch.written, 5);
        assert_eq!(batch.state, GenerationState::AwaitingFeedback);
        assert!(!output[..batch.written].contains(&previous));
        model
            .apply_feedback(&[Feedback::Active(previous.into()), Feedback::BatchComplete])
            .unwrap();
        model.generate(&mut output).unwrap();
        assert!(output.iter().all(|address| address[0] == previous[0]));
        assert!(!output.contains(&previous));
    }
}
