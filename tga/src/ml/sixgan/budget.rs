use crate::{Address, Feedback, GenerationState, TgaError};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct Budget {
    remaining: Vec<usize>,
    allocated: bool,
    total: usize,
    seeds: BTreeSet<Address>,
    aliases: BTreeMap<u8, BTreeSet<u128>>,
    origins: BTreeMap<Address, BTreeSet<usize>>,
    outcomes: BTreeMap<Address, bool>,
    pub buffer: VecDeque<Address>,
    pub generator: usize,
    pub draw: u64,
}

impl Budget {
    pub fn new(
        classes: usize,
        calibration: usize,
        total: usize,
        seeds: &[Address],
        aliases: &[ipnet::Ipv6Net],
    ) -> Self {
        Self {
            remaining: vec![calibration; classes],
            allocated: false,
            total,
            seeds: seeds.iter().copied().collect(),
            aliases: aliases.iter().fold(BTreeMap::new(), |mut index, prefix| {
                index
                    .entry(prefix.prefix_len())
                    .or_insert_with(BTreeSet::new)
                    .insert(u128::from(prefix.network()));
                index
            }),
            origins: BTreeMap::new(),
            outcomes: BTreeMap::new(),
            buffer: VecDeque::new(),
            generator: 0,
            draw: 0,
        }
    }

    pub fn validate(&self, classes: usize) -> Result<(), TgaError> {
        if self.remaining.len() != classes
            || self.generator >= classes
            || self.buffer.len() > self.remaining[self.generator]
            || self
                .origins
                .values()
                .flatten()
                .any(|&class| class >= classes)
            || self.aliases.keys().any(|&bits| bits > 128)
        {
            return Err(TgaError::Model(
                "6GAN feedback state has invalid dimensions".into(),
            ));
        }
        Ok(())
    }

    pub fn state(&self) -> GenerationState {
        if self.remaining.iter().any(|&count| count > 0) {
            GenerationState::Ready
        } else if self.allocated {
            GenerationState::Exhausted
        } else {
            GenerationState::AwaitingFeedback
        }
    }

    pub fn next_batch(&self, batch_size: usize) -> Option<(usize, usize)> {
        self.remaining
            .iter()
            .position(|&count| count > 0)
            .map(|index| (index, self.remaining[index].min(batch_size)))
    }

    pub fn emit(&mut self) -> Option<Address> {
        let address = self.buffer.pop_front()?;
        self.remaining[self.generator] -= 1;
        if !self.allocated {
            self.origins
                .entry(address)
                .or_default()
                .insert(self.generator);
        }
        Some(address)
    }

    fn known_non_discovery(&self, address: &Address) -> bool {
        let value = u128::from_be_bytes(*address);
        self.seeds.contains(address)
            || self.aliases.iter().any(|(&bits, prefixes)| {
                let mask = u128::MAX.checked_shl(u32::from(128 - bits)).unwrap_or(0);
                prefixes.contains(&(value & mask))
            })
    }

    pub fn feedback(&mut self, feedback: &[Feedback]) -> Result<(), TgaError> {
        if self.allocated {
            return Ok(());
        }
        for item in feedback {
            match item {
                Feedback::Active(ip) | Feedback::Inactive(ip) => {
                    if self.origins.contains_key(&ip.octets()) {
                        let active = matches!(item, Feedback::Active(_));
                        *self.outcomes.entry(ip.octets()).or_default() |= active;
                    }
                }
                Feedback::Aliased(prefix) => {
                    self.aliases
                        .entry(prefix.prefix_len())
                        .or_default()
                        .insert(u128::from(prefix.network()));
                }
                Feedback::Skipped(_) | Feedback::BatchComplete => {}
            }
        }
        if !feedback.contains(&Feedback::BatchComplete)
            || self.state() != GenerationState::AwaitingFeedback
        {
            return Ok(());
        }
        if self.origins.keys().any(|address| {
            !self.outcomes.contains_key(address) && !self.known_non_discovery(address)
        }) {
            return Err(TgaError::Feedback(
                "6GAN calibration still has unresolved candidates".into(),
            ));
        }
        let mut candidates = vec![0usize; self.remaining.len()];
        let mut discoveries = vec![0usize; self.remaining.len()];
        for (address, classes) in &self.origins {
            let novel = self.outcomes.get(address).copied().unwrap_or(false)
                && !self.known_non_discovery(address);
            for &class in classes {
                candidates[class] += 1;
                discoveries[class] += usize::from(novel);
            }
        }
        let rates = discoveries
            .iter()
            .zip(candidates)
            .map(|(&hits, count)| hits as f64 / count.max(1) as f64)
            .collect::<Vec<_>>();
        self.remaining = allocate(&rates, self.total).map_err(TgaError::Feedback)?;
        self.allocated = true;
        self.origins.clear();
        self.outcomes.clear();
        self.seeds.clear();
        self.aliases.clear();
        Ok(())
    }
}

fn allocate(rates: &[f64], total: usize) -> Result<Vec<usize>, String> {
    let sum: f64 = rates.iter().sum();
    if sum == 0.0 {
        return Err(
            "6GAN cannot apply Equation 16 because every measured generation rate is zero".into(),
        );
    }
    let exact = rates
        .iter()
        .map(|rate| rate / sum * total as f64)
        .collect::<Vec<_>>();
    let mut remaining = total;
    let mut counts = exact
        .iter()
        .map(|value| {
            let count = (value.floor() as usize).min(remaining);
            remaining -= count;
            count
        })
        .collect::<Vec<_>>();
    let mut order = (0..rates.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| {
        (exact[b] - counts[b] as f64)
            .total_cmp(&(exact[a] - counts[a] as f64))
            .then(a.cmp(&b))
    });
    for &index in order.iter().cycle().take(remaining) {
        counts[index] += 1;
    }
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn address(last: u8) -> Address {
        let mut address = [0; 16];
        address[15] = last;
        address
    }
    fn emit(budget: &mut Budget, class: usize, addresses: &[Address]) {
        budget.generator = class;
        budget.buffer.extend(addresses);
        for _ in addresses {
            budget.emit().unwrap();
        }
    }

    #[test]
    fn skipped_unmeasured_targets_are_not_assumed_inactive() {
        let mut budget = Budget::new(1, 2, 10, &[address(1)], &[]);
        emit(&mut budget, 0, &[address(1), address(2)]);
        assert!(
            budget
                .feedback(&[
                    Feedback::Skipped(address(1).into()),
                    Feedback::Skipped(address(2).into()),
                    Feedback::BatchComplete
                ])
                .is_err()
        );
        budget
            .feedback(&[Feedback::Active(address(2).into()), Feedback::BatchComplete])
            .unwrap();
        assert_eq!(budget.remaining, vec![10]);
    }

    #[test]
    fn equation_16_allocates_exact_integer_budget() {
        assert_eq!(allocate(&[0.5, 0.25, 0.0], 10).unwrap(), vec![7, 3, 0]);
        assert!(allocate(&[0.0, 0.0, 0.0], 10).is_err());
    }

    #[test]
    fn rates_deduplicate_candidates_and_exclude_seeds_and_aliases() {
        let mut budget = Budget::new(2, 4, 12, &[address(1)], &[]);
        emit(
            &mut budget,
            0,
            &[address(1), address(2), address(2), address(3)],
        );
        emit(
            &mut budget,
            1,
            &[address(2), address(4), address(5), address(6)],
        );
        let mut feedback = (1..=6)
            .map(|i| Feedback::Active(address(i).into()))
            .collect::<Vec<_>>();
        feedback.push(Feedback::Aliased("::3/128".parse().unwrap()));
        feedback.push(Feedback::BatchComplete);
        budget.feedback(&feedback).unwrap();
        assert_eq!(budget.remaining, vec![3, 9]);
    }

    #[test]
    fn incomplete_feedback_waits_and_persistence_preserves_attribution() {
        let mut budget = Budget::new(2, 1, 3, &[], &[]);
        emit(&mut budget, 0, &[address(1)]);
        budget
            .feedback(&[Feedback::Active(address(1).into()), Feedback::BatchComplete])
            .unwrap();
        let mut budget: Budget =
            bincode::deserialize(&bincode::serialize(&budget).unwrap()).unwrap();
        emit(&mut budget, 1, &[address(1)]);
        budget.feedback(&[Feedback::BatchComplete]).unwrap();
        assert_eq!(budget.remaining, vec![2, 1]);
        let mut pending = Budget::new(1, 1, 1, &[], &[]);
        emit(&mut pending, 0, &[address(2)]);
        assert!(pending.feedback(&[Feedback::BatchComplete]).is_err());
        assert_eq!(pending.state(), GenerationState::AwaitingFeedback);
    }
}
