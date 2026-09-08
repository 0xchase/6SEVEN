use super::{Address, ExactSize, Nibbles, cap_count, model::SixGenBlock, range::AddressRange};
use rand::{Rng, rngs::StdRng};
use std::collections::HashSet;

pub(super) fn subtract_from_fragments(
    range: AddressRange,
    fragments: &[AddressRange],
) -> Vec<AddressRange> {
    let mut remaining = vec![range];

    for fragment in fragments {
        let mut next = Vec::new();
        for piece in remaining {
            next.extend(piece.subtract(fragment));
        }
        remaining = next;
        if remaining.is_empty() {
            break;
        }
    }

    remaining
}

#[cfg(test)]
pub(super) fn total_fragment_size(fragments: &[AddressRange]) -> usize {
    fragments
        .iter()
        .fold(0usize, |acc, fragment| acc.saturating_add(fragment.size()))
}

pub(super) fn total_fragment_exact_size(fragments: &[AddressRange]) -> ExactSize {
    fragments.iter().fold(0u128, |acc, fragment| {
        acc.checked_add(fragment.exact_size().finite())
            .expect("disjoint fragments excluding seeds fit u128")
    })
}

#[derive(Debug, Clone)]
pub(super) struct GeneratedCoverage {
    // Coverage includes seeds and emitted targets to avoid charging either twice.
    fragments: Vec<AddressRange>,
    blocks: Vec<SixGenBlock>,
    target_count: usize,
}

impl GeneratedCoverage {
    pub(super) fn from_seed_ranges(seeds: &[Nibbles]) -> Self {
        Self {
            fragments: seeds.iter().map(AddressRange::from_seed).collect(),
            blocks: Vec::new(),
            target_count: 0,
        }
    }

    pub(super) fn target_count(&self) -> usize {
        self.target_count
    }

    pub(super) fn uncovered_fragments(&self, range: AddressRange) -> Vec<AddressRange> {
        subtract_from_fragments(range, &self.fragments)
    }

    pub(super) fn admit_fragments(&mut self, fragments: Vec<AddressRange>) {
        for fragment in fragments {
            self.target_count = self.target_count.saturating_add(fragment.size());
            self.fragments.push(fragment);
            self.blocks.push(SixGenBlock::Range(fragment));
        }
    }

    pub(super) fn admit_final_sample(
        &mut self,
        fragments: &[AddressRange],
        count: usize,
        rng: &mut StdRng,
    ) {
        let sampled = sample_points(fragments, count, rng);
        if sampled.is_empty() {
            return;
        }

        self.target_count = self.target_count.saturating_add(sampled.len());
        self.blocks.push(SixGenBlock::Points(sampled));
    }

    pub(super) fn into_blocks(self) -> (Vec<SixGenBlock>, usize) {
        (self.blocks, self.target_count)
    }
}

// Floyd sampling uses one draw per target without replacement.
pub(super) fn sample_points(
    fragments: &[AddressRange],
    count: usize,
    rng: &mut StdRng,
) -> Vec<Address> {
    let mut total = 0u128;
    let ends: Vec<_> = fragments
        .iter()
        .map(|fragment| {
            total = total
                .checked_add(fragment.exact_size().finite())
                .expect("disjoint fragments excluding seeds fit u128");
            total
        })
        .collect();
    let count = count.min(cap_count(total));
    let mut ranks = HashSet::with_capacity(count);
    let mut selected = Vec::with_capacity(count);
    for upper in (total - count as u128)..total {
        let drawn = rng.gen_range(0..=upper);
        let rank = if ranks.insert(drawn) {
            drawn
        } else {
            ranks.insert(upper);
            upper
        };
        let index = ends.partition_point(|&end| end <= rank);
        let start = if index == 0 { 0 } else { ends[index - 1] };
        selected.push(fragments[index].address_at(rank - start));
    }
    selected
}
