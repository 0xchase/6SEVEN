use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{Address, Feedback, TgaError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct Allocation {
    seeds: BTreeSet<Address>,
    candidates: BTreeMap<Address, Candidate>,
    rates: Vec<f64>,
    credits: Vec<f64>,
    round_samples: usize,
    remaining: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Candidate {
    sources: BTreeSet<usize>,
    outcome: Outcome,
    draws: usize,
    skipped: usize,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
enum Outcome {
    #[default]
    Pending,
    Inactive,
    Active,
    Aliased,
}

impl Allocation {
    pub fn new(
        seeds: impl IntoIterator<Item = Address>,
        components: usize,
        round_samples: usize,
    ) -> Result<Self, TgaError> {
        let remaining = components
            .checked_mul(round_samples)
            .filter(|_| components > 0 && round_samples > 0)
            .ok_or_else(|| TgaError::Config("invalid 6GCVAE calibration size".into()))?;
        Ok(Self {
            seeds: seeds.into_iter().collect(),
            candidates: BTreeMap::new(),
            rates: vec![1.0; components],
            credits: vec![0.0; components],
            round_samples,
            remaining,
        })
    }

    pub fn select(&mut self) -> usize {
        let total: f64 = self.rates.iter().sum();
        let uniform = total == 0.0;
        let total = if uniform {
            self.rates.len() as f64
        } else {
            total
        };
        let mut selected = 0;
        for index in 0..self.rates.len() {
            self.credits[index] += if uniform { 1.0 } else { self.rates[index] };
            if self.credits[index] > self.credits[selected] {
                selected = index;
            }
        }
        self.credits[selected] -= total;
        selected
    }

    pub fn remaining(&self) -> usize {
        self.remaining
    }

    pub fn record(&mut self, address: Address, source: usize) {
        let candidate = self.candidates.entry(address).or_insert_with(|| Candidate {
            sources: BTreeSet::new(),
            outcome: Outcome::Pending,
            draws: 0,
            skipped: 0,
        });
        candidate.sources.insert(source);
        candidate.draws = candidate.draws.saturating_add(1);
        self.remaining = self.remaining.saturating_sub(1);
    }

    pub fn apply(&mut self, feedback: &[Feedback]) {
        for event in feedback {
            match event {
                Feedback::BatchComplete => self.complete(),
                Feedback::Aliased(prefix) => {
                    for (address, candidate) in &mut self.candidates {
                        if prefix.contains(&std::net::Ipv6Addr::from(*address)) {
                            candidate.outcome = Outcome::Aliased;
                        }
                    }
                }
                Feedback::Active(address)
                | Feedback::Inactive(address)
                | Feedback::Skipped(address) => {
                    let Some(candidate) = self.candidates.get_mut(&address.octets()) else {
                        continue;
                    };
                    if matches!(event, Feedback::Skipped(_)) {
                        candidate.skipped =
                            candidate.skipped.saturating_add(1).min(candidate.draws);
                    } else {
                        candidate.outcome = match (candidate.outcome, event) {
                            (Outcome::Aliased, _) => Outcome::Aliased,
                            (Outcome::Active, _) | (_, Feedback::Active(_)) => Outcome::Active,
                            _ => Outcome::Inactive,
                        };
                    }
                }
            }
        }
    }

    fn complete(&mut self) {
        if self.candidates.is_empty() {
            return;
        }
        self.remaining = self.round_samples;
        let mut candidates = vec![0usize; self.rates.len()];
        let mut new_active = vec![0usize; self.rates.len()];
        for (address, candidate) in std::mem::take(&mut self.candidates) {
            if matches!(candidate.outcome, Outcome::Aliased)
                || (matches!(candidate.outcome, Outcome::Pending)
                    && candidate.skipped == candidate.draws)
            {
                continue;
            }
            let is_new =
                matches!(candidate.outcome, Outcome::Active) && !self.seeds.contains(&address);
            for source in candidate.sources {
                candidates[source] += 1;
                new_active[source] += usize::from(is_new);
            }
        }
        if candidates.iter().all(|&count| count == 0) {
            return;
        }
        for (index, count) in candidates.into_iter().enumerate() {
            if count > 0 {
                self.rates[index] = new_active[index] as f64 / count as f64;
            }
        }
        self.credits.fill(0.0);
    }

    pub fn validate(&self, components: usize) -> Result<(), TgaError> {
        if (self.remaining == 0 && self.candidates.is_empty())
            || self.round_samples == 0
            || self.remaining > self.round_samples.saturating_mul(components)
            || self.rates.len() != components
            || self.credits.len() != components
            || self
                .rates
                .iter()
                .any(|rate| !rate.is_finite() || !(0.0..=1.0).contains(rate))
            || self.credits.iter().any(|credit| !credit.is_finite())
            || self.candidates.values().any(|candidate| {
                candidate.sources.is_empty()
                    || candidate.draws < candidate.sources.len()
                    || candidate.skipped > candidate.draws
                    || candidate.sources.iter().any(|&source| source >= components)
            })
        {
            return Err(TgaError::Model("invalid 6GCVAE feedback allocation".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(value: u128) -> Address {
        value.to_be_bytes()
    }
    fn active(value: u128) -> Feedback {
        Feedback::Active(value.into())
    }

    #[test]
    fn allocation_uses_new_active_rate_and_deduplicates_each_category() {
        let mut allocation = Allocation::new([addr(1)], 2, 10).unwrap();
        for (address, component) in [(1, 0), (2, 0), (2, 0), (3, 1), (4, 1), (5, 1), (6, 1)] {
            allocation.record(addr(address), component);
        }
        allocation.apply(&[
            active(1),
            active(2),
            active(3),
            active(3),
            Feedback::Inactive(3.into()),
            active(999),
            Feedback::BatchComplete,
        ]);
        assert_eq!(allocation.rates, vec![0.5, 0.25]);
        let mut counts = [0; 2];
        for _ in 0..300 {
            counts[allocation.select()] += 1;
        }
        assert_eq!(counts, [200, 100]);
    }

    #[test]
    fn shared_candidates_are_attributed_to_every_source() {
        let mut allocation = Allocation::new([], 2, 10).unwrap();
        allocation.record(addr(1), 0);
        allocation.record(addr(1), 1);
        allocation.apply(&[active(1), Feedback::BatchComplete]);
        assert_eq!(allocation.rates, vec![1.0, 1.0]);
    }

    #[test]
    fn skipped_candidates_are_not_measurements_and_empty_feedback_is_idempotent() {
        let mut allocation = Allocation::new([], 2, 10).unwrap();
        allocation.record(addr(1), 0);
        allocation.record(addr(2), 0);
        allocation.record(addr(3), 1);
        allocation.apply(&[
            active(1),
            Feedback::Skipped(2.into()),
            Feedback::Skipped(3.into()),
            Feedback::BatchComplete,
        ]);
        assert_eq!(allocation.rates, vec![1.0, 1.0]);
        allocation.select();
        let before = bincode::serialize(&allocation).unwrap();
        allocation.apply(&[Feedback::BatchComplete]);
        assert_eq!(before, bincode::serialize(&allocation).unwrap());
    }

    #[test]
    fn zero_rates_preserve_progress_without_assigning_false_hits() {
        let mut allocation = Allocation::new([], 2, 10).unwrap();
        allocation.record(addr(1), 0);
        allocation.record(addr(2), 1);
        allocation.apply(&[Feedback::BatchComplete]);
        assert_eq!(allocation.rates, vec![0.0, 0.0]);
        assert_eq!(
            (0..4).map(|_| allocation.select()).collect::<Vec<_>>(),
            [0, 1, 0, 1]
        );
    }

    #[test]
    fn skipping_a_duplicate_keeps_the_original_measurement() {
        let mut allocation = Allocation::new([], 2, 10).unwrap();
        allocation.record(addr(1), 0);
        allocation.record(addr(1), 1);
        allocation.apply(&[
            Feedback::Skipped(1.into()),
            active(1),
            Feedback::BatchComplete,
        ]);
        assert_eq!(allocation.rates, vec![1.0, 1.0]);
        allocation.record(addr(2), 0);
        allocation.record(addr(2), 0);
        allocation.apply(&[Feedback::Skipped(2.into()), Feedback::BatchComplete]);
        assert_eq!(allocation.rates, vec![0.0, 1.0]);
    }

    #[test]
    fn allocations_change_only_at_completed_rounds() {
        let mut allocation = Allocation::new([], 2, 1).unwrap();
        allocation.record(addr(1), 0);
        allocation.record(addr(2), 1);
        allocation.apply(&[active(1)]);
        assert_eq!(allocation.rates, vec![1.0, 1.0]);
        assert_eq!(allocation.remaining(), 0);
        allocation.apply(&[Feedback::BatchComplete]);
        assert_eq!(allocation.rates, vec![1.0, 0.0]);
        assert_eq!(allocation.remaining(), 1);
        for _ in 0..10 {
            assert_eq!(allocation.select(), 0);
        }
    }

    #[test]
    fn calibration_draws_are_equal_across_categories() {
        let mut allocation = Allocation::new([], 3, 4).unwrap();
        let mut counts = [0; 3];
        while allocation.remaining() > 0 {
            let source = allocation.select();
            counts[source] += 1;
            allocation.record(addr(counts.iter().sum::<u128>()), source);
        }
        assert_eq!(counts, [4, 4, 4]);
    }

    #[test]
    fn aliased_candidates_do_not_count_as_new_hits() {
        let mut allocation = Allocation::new([], 1, 2).unwrap();
        allocation.record(addr(1), 0);
        allocation.record(addr(2), 0);
        allocation.apply(&[
            active(1),
            Feedback::Aliased("::1/128".parse().unwrap()),
            Feedback::Inactive(2.into()),
            Feedback::BatchComplete,
        ]);
        assert_eq!(allocation.rates, vec![0.0]);
    }

    #[test]
    fn pending_attribution_survives_serialization() {
        let mut allocation = Allocation::new([], 2, 10).unwrap();
        allocation.record(addr(1), 1);
        let mut restored: Allocation =
            bincode::deserialize(&bincode::serialize(&allocation).unwrap()).unwrap();
        restored.apply(&[active(1), Feedback::BatchComplete]);
        assert_eq!(restored.rates, vec![1.0, 1.0]);
        assert!(restored.candidates.is_empty());
    }
}
