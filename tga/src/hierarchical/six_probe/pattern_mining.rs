use super::encoding::{NibbleAddr, PARALLEL_NIBBLE_SCAN_MIN_SEEDS, Subspace, compute_subspace};
use super::split::max_covering;
use rayon::prelude::*;
use std::collections::HashSet;

#[derive(Default)]
pub(super) struct PatternMiner {
    seen_regions: HashSet<Subspace>,
    patterns: Vec<Subspace>,
}

impl PatternMiner {
    pub(super) fn into_patterns(self) -> Vec<Subspace> {
        self.patterns
    }

    pub(super) fn mine(&mut self, seeds: &[NibbleAddr], subspace: Subspace, dimension: usize) {
        if seeds.len() == 2 {
            if dimension == 1 {
                // The release bypasses region deduplication for two-seed leaves.
                self.patterns.push(subspace);
            }
            return;
        }
        if seeds.len() < 2 || dimension == 0 {
            return;
        }
        let weights = isolated_forest_weights(seeds);
        let mut is_outlier = vec![false; seeds.len()];
        for value in four_d(&weights) {
            // Preserve np.where(...)[0][0], including first-match tie behavior.
            if let Some(index) = weights.iter().position(|&weight| weight == value) {
                is_outlier[index] = true;
            }
        }
        let region = seeds
            .iter()
            .zip(is_outlier)
            .filter_map(|(&seed, outlier)| (!outlier).then_some(seed))
            .collect();
        for region in iter_divide(region) {
            self.record_region(&region);
        }
    }

    fn record_region(&mut self, seeds: &[NibbleAddr]) {
        let (subspace, dimension) = compute_subspace(seeds);
        if !(1..=4).contains(&dimension) {
            return;
        }
        // The release records candidates even when the density check rejects them.
        if self.seen_regions.insert(subspace) && is_probeable_low_dim_region(dimension, seeds.len())
        {
            self.patterns.push(subspace);
        }
    }
}

fn is_probeable_low_dim_region(dimension: usize, seed_count: usize) -> bool {
    matches!(dimension, 1..=3) || (dimension == 4 && seed_count >= 3)
}

fn iter_divide(seeds: Vec<NibbleAddr>) -> Vec<Vec<NibbleAddr>> {
    let mut stack = vec![seeds];
    let mut regions = Vec::new();

    while let Some(current) = stack.pop() {
        let splits = max_covering(&current);
        if splits.iter().any(|split| split.len() == 1) {
            regions.push(current);
            continue;
        }

        if splits.len() == 1 && splits[0].len() == current.len() {
            regions.push(current);
            continue;
        }

        for split in splits {
            let next: Vec<NibbleAddr> = split.into_iter().map(|idx| current[idx]).collect();
            stack.push(next);
        }
    }

    regions
}

fn isolated_forest_weights(seeds: &[NibbleAddr]) -> Vec<f64> {
    let mut weights = vec![0.0f64; seeds.len()];

    let contributions = if seeds.len() >= PARALLEL_NIBBLE_SCAN_MIN_SEEDS {
        (0..32)
            .into_par_iter()
            .map(|dim| isolated_forest_dimension_contributions(seeds, dim))
            .collect::<Vec<_>>()
    } else {
        (0..32)
            .map(|dim| isolated_forest_dimension_contributions(seeds, dim))
            .collect::<Vec<_>>()
    };

    for dimension_contributions in contributions {
        for (seed_idx, contribution) in dimension_contributions {
            weights[seed_idx] += contribution;
        }
    }

    weights
}

fn isolated_forest_dimension_contributions(seeds: &[NibbleAddr], dim: usize) -> Vec<(usize, f64)> {
    let mut counts = [0usize; 16];
    let mut first_index = [0usize; 16];

    for (idx, seed) in seeds.iter().enumerate() {
        let nibble = seed[dim] as usize;
        if counts[nibble] == 0 {
            first_index[nibble] = idx;
        }
        counts[nibble] += 1;
    }

    if counts.iter().filter(|&&count| count > 0).count() == 1 {
        return Vec::new();
    }

    let outlier_num = counts.iter().filter(|&&count| count == 1).count();
    if outlier_num == 0 {
        return Vec::new();
    }

    let contribution = 1.0 / outlier_num as f64;
    let mut contributions = Vec::with_capacity(outlier_num);
    for nibble in 0..16 {
        if counts[nibble] == 1 {
            contributions.push((first_index[nibble], contribution));
        }
    }
    contributions
}

fn four_d(weights: &[f64]) -> Vec<f64> {
    if weights.len() <= 2 {
        return Vec::new();
    }

    let mut remaining = weights.to_vec();
    let mut outliers = Vec::new();
    while remaining.len() > 2 {
        let index = argmax(&remaining);
        let candidate = remaining.remove(index);
        if candidate - mean(&remaining) <= 3.0 * variance(&remaining).sqrt() {
            break;
        }
        outliers.push(candidate);
    }
    outliers
}

fn argmax(values: &[f64]) -> usize {
    let mut max_idx = 0usize;
    for (idx, value) in values.iter().enumerate().skip(1) {
        if *value > values[max_idx] {
            max_idx = idx;
        }
    }
    max_idx
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let avg = mean(values);
    values.iter().map(|v| (v - avg).powi(2)).sum::<f64>() / values.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_pattern(pattern: &str) -> Vec<NibbleAddr> {
        let mut seeds = vec![[0u8; 32]];
        let mut nibble_idx = 0usize;
        for ch in pattern.chars() {
            if ch == ':' {
                continue;
            }

            match ch {
                '*' => {
                    let mut next = Vec::with_capacity(seeds.len() * 2);
                    for seed in &seeds {
                        let mut left = *seed;
                        left[nibble_idx] = 0;
                        next.push(left);
                        let mut right = *seed;
                        right[nibble_idx] = 1;
                        next.push(right);
                    }
                    seeds = next;
                }
                hex => {
                    let value = hex.to_digit(16).expect("pattern must contain hex digits") as u8;
                    for seed in &mut seeds {
                        seed[nibble_idx] = value;
                    }
                }
            }

            nibble_idx += 1;
        }

        seeds
    }

    #[test]
    fn non_probeable_four_d_region_still_blocks_later_probeable_duplicate() {
        let duplicate = "2409:40c1:41**:**d4:8000:0000:0000:0000";
        let two_seed_region = vec![
            parse_pattern("2409:40c1:4102:edd4:8000:0000:0000:0000")[0],
            parse_pattern("2409:40c1:4143:64d4:8000:0000:0000:0000")[0],
        ];
        let three_seed_region = parse_pattern(duplicate);

        let mut miner = PatternMiner::default();
        miner.record_region(&two_seed_region);
        miner.record_region(&three_seed_region);
        assert!(miner.patterns.is_empty());
        assert_eq!(miner.seen_regions.len(), 1);
    }
    #[test]
    fn mining_stages_match_released_python() {
        for case in super::super::tests::stage_cases() {
            let seeds = case.nibbles();
            let weights = isolated_forest_weights(&seeds);
            assert_eq!(weights.len(), case.weights.len());
            for (&actual, &expected) in weights.iter().zip(&case.weights) {
                assert!((actual - expected).abs() < 1e-12, "{}", case.name);
            }
            assert_eq!(four_d(&weights), case.outliers, "{}", case.name);
            let (space, dimension) = compute_subspace(&seeds);
            let mut miner = PatternMiner::default();
            miner.mine(&seeds, space, dimension);
            let patterns: Vec<_> = miner
                .into_patterns()
                .iter()
                .map(super::super::encoding::subspace_to_pattern_string)
                .collect();
            assert_eq!(patterns, case.mined, "{}", case.name);
        }
    }
}
