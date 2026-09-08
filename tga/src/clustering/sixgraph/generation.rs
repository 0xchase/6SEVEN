use super::{HEX_RADIX, Pattern};
use crate::address::{Address, AddressExt};
use std::collections::HashSet;

const MIX_INCREMENT: u64 = 0x9e37_79b9_7f4a_7c15;

pub(super) fn sample_round(patterns: &[Pattern], scan: &super::model::ScanState) -> Vec<Address> {
    let mut selected = HashSet::new();
    let mut output = Vec::new();
    for (index, pattern) in patterns.iter().enumerate() {
        let mut seen = HashSet::new();
        let mut candidates = Vec::new();
        for seed in pattern.seeds.iter().filter(|seed| !scan.is_aliased(seed)) {
            for &position in &pattern.variable_nibbles {
                for value in 0..HEX_RADIX as u8 {
                    if value == seed.get_nibble(position) {
                        continue;
                    }
                    let mut target = *seed;
                    target.set_nibble(position, value);
                    if !scan.excluded(&target) && !selected.contains(&target) && seen.insert(target)
                    {
                        candidates.push(target);
                    }
                }
            }
        }
        drop(seen);
        let seed = pattern_sample_seed(scan.sample_seed, index) ^ mix64(scan.round as u64);
        shuffle_items(&mut candidates, seed);
        candidates.truncate(sampling_round_budget(
            scan.round,
            scan.initial_counts[index],
        ));
        selected.extend(candidates.iter().copied());
        output.extend(candidates);
    }
    output
}

pub(super) fn shuffle_items<T>(items: &mut [T], seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for i in (1..items.len()).rev() {
        let j = rng.index_below(i + 1);
        items.swap(i, j);
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(MIX_INCREMENT);
        mix64(self.state)
    }

    fn index_below(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        if upper <= 1 {
            return 0;
        }

        let upper = upper as u64;
        let acceptance_zone = u64::MAX - (u64::MAX % upper);
        loop {
            let value = self.next_u64();
            if value < acceptance_zone {
                return (value % upper) as usize;
            }
        }
    }
}

pub(super) fn sampling_round_budget(round_idx: usize, seed_count: usize) -> usize {
    if round_idx <= 1 {
        return seed_count;
    }

    let shift = round_idx - 1;
    if shift >= usize::BITS as usize {
        return usize::MAX;
    }

    seed_count.saturating_mul(1usize << shift)
}

fn pattern_sample_seed(sample_seed: u64, pattern_idx: usize) -> u64 {
    mix64(sample_seed ^ (pattern_idx as u64).wrapping_mul(MIX_INCREMENT))
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
