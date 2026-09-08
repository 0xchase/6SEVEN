use super::{
    ExactSize, Nibbles, SixGen, SixGenModel, SixGenRangeMode,
    coverage::{GeneratedCoverage, total_fragment_exact_size},
    index::SeedTrie,
    model::total_block_size,
    range::{AddressRange, RangeSize},
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use std::collections::HashSet;

#[derive(Debug, Clone, Copy)]
pub(super) struct GrowthProposal {
    pub(super) new_range: AddressRange,
    pub(super) new_seed_count: usize,
    pub(super) range_size: RangeSize,
}

pub(super) type GrowthCache = Vec<GrowthProposal>;

#[derive(Debug, Clone)]
pub(super) struct ActiveCluster {
    pub(super) range: AddressRange,
    pub(super) growth_cache: Option<GrowthCache>,
}

impl ActiveCluster {
    pub(super) fn new(range: AddressRange) -> Self {
        Self {
            range,
            growth_cache: None,
        }
    }
}

pub(super) fn compare_proposals(lhs: &GrowthProposal, rhs: &GrowthProposal) -> std::cmp::Ordering {
    // Doubling usize seed counts in u128 keeps full-space density comparisons exact.
    let scaled = lhs.range_size == RangeSize::Full || rhs.range_size == RangeSize::Full;
    let ratio = |proposal: &GrowthProposal| match proposal.range_size {
        RangeSize::Full => (proposal.new_seed_count as u128, 1u128 << 127),
        RangeSize::Finite(size) => ((proposal.new_seed_count as u128) << u32::from(scaled), size),
    };
    let (lhs_count, lhs_size) = ratio(lhs);
    let (rhs_count, rhs_size) = ratio(rhs);
    compare_density(lhs_count, lhs_size, rhs_count, rhs_size)
        .then_with(|| rhs.range_size.cmp(&lhs.range_size))
}

pub(super) fn compare_density(
    lhs_count: ExactSize,
    lhs_range_size: ExactSize,
    rhs_count: ExactSize,
    rhs_range_size: ExactSize,
) -> std::cmp::Ordering {
    debug_assert!(lhs_range_size > 0);
    debug_assert!(rhs_range_size > 0);

    compare_ratio(lhs_count, lhs_range_size, rhs_count, rhs_range_size)
}

// Continued fractions compare densities without overflowing cross products.
fn compare_ratio(
    mut lhs_num: ExactSize,
    mut lhs_den: ExactSize,
    mut rhs_num: ExactSize,
    mut rhs_den: ExactSize,
) -> std::cmp::Ordering {
    debug_assert!(lhs_den > 0);
    debug_assert!(rhs_den > 0);

    let mut inverted = false;
    loop {
        let lhs_quotient = lhs_num / lhs_den;
        let rhs_quotient = rhs_num / rhs_den;
        if lhs_quotient != rhs_quotient {
            let ordering = lhs_quotient.cmp(&rhs_quotient);
            return if inverted {
                ordering.reverse()
            } else {
                ordering
            };
        }

        let lhs_remainder = lhs_num % lhs_den;
        let rhs_remainder = rhs_num % rhs_den;
        let ordering = match (lhs_remainder == 0, rhs_remainder == 0) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (false, false) => {
                lhs_num = lhs_den;
                lhs_den = lhs_remainder;
                rhs_num = rhs_den;
                rhs_den = rhs_remainder;
                inverted = !inverted;
                continue;
            }
        };

        return if inverted {
            ordering.reverse()
        } else {
            ordering
        };
    }
}

pub(super) fn best_growth_for_cluster(
    range: &AddressRange,
    seeds: &[Nibbles],
    seed_trie: &SeedTrie,
    mode: SixGenRangeMode,
) -> GrowthCache {
    let candidates = seed_trie.nearest_external_seed_indices(range);
    if candidates.is_empty() {
        return Vec::new();
    }

    let mut seen_ranges = HashSet::new();
    let mut proposals = Vec::new();

    for candidate_idx in candidates {
        let new_range = range.grow_with_seed(&seeds[candidate_idx], mode);
        if !seen_ranges.insert(new_range) {
            continue;
        }

        let proposal = GrowthProposal {
            new_seed_count: seed_trie.count_in_range(&new_range),
            range_size: new_range.exact_size(),
            new_range,
        };

        match proposals.first() {
            None => proposals.push(proposal),
            Some(best) => match compare_proposals(&proposal, best) {
                std::cmp::Ordering::Greater => {
                    proposals.clear();
                    proposals.push(proposal);
                }
                std::cmp::Ordering::Equal => proposals.push(proposal),
                std::cmp::Ordering::Less => {}
            },
        }
    }

    proposals
}

pub(super) fn prune_subset_clusters(clusters: &mut [Option<ActiveCluster>], kept_idx: usize) {
    let Some(kept_range) = clusters
        .get(kept_idx)
        .and_then(Option::as_ref)
        .map(|cluster| cluster.range)
    else {
        return;
    };

    for (idx, cluster) in clusters.iter_mut().enumerate() {
        if idx == kept_idx {
            continue;
        }

        let remove = cluster
            .as_ref()
            .is_some_and(|cluster| cluster.range.strict_subset_of(&kept_range));

        if remove {
            *cluster = None;
        }
    }
}

pub(super) fn train_sixgen(seeds: &[Nibbles], config: &SixGen) -> SixGenModel {
    SixGenTrainer::new(seeds, config).train()
}

struct SixGenTrainer<'a> {
    seeds: &'a [Nibbles],
    seed_trie: SeedTrie,
    range_mode: SixGenRangeMode,
    budget: usize,
    rng: StdRng,
    clusters: Vec<Option<ActiveCluster>>,
    coverage: GeneratedCoverage,
}

impl<'a> SixGenTrainer<'a> {
    fn new(seeds: &'a [Nibbles], config: &SixGen) -> Self {
        let seed_trie = SeedTrie::new(seeds);

        let clusters = seeds
            .iter()
            .map(|seed| Some(ActiveCluster::new(AddressRange::from_seed(seed))))
            .collect::<Vec<_>>();
        let coverage = GeneratedCoverage::from_seed_ranges(seeds);

        Self {
            seeds,
            seed_trie,
            range_mode: config.range_mode,
            budget: config.budget,
            rng: StdRng::seed_from_u64(config.seed),
            clusters,
            coverage,
        }
    }

    fn train(mut self) -> SixGenModel {
        while self.coverage.target_count() < self.budget {
            self.refresh_growth_caches();
            let Some((cluster_idx, proposal)) = self.select_best_growth() else {
                break;
            };

            let new_seed_count = proposal.new_seed_count;
            if !self.admit_growth(cluster_idx, proposal) {
                break;
            }

            // The text states 6Gen stops once one cluster contains every seed.
            if new_seed_count == self.seeds.len() || self.coverage.target_count() >= self.budget {
                break;
            }
        }

        let (blocks, target_count) = self.coverage.into_blocks();
        let model = SixGenModel::from_blocks(blocks, self.seeds.len(), self.range_mode);
        debug_assert_eq!(target_count, total_block_size(&model.blocks));
        model
    }

    fn refresh_growth_caches(&mut self) {
        let seeds = self.seeds;
        let seed_trie = &self.seed_trie;
        let range_mode = self.range_mode;

        self.clusters.par_iter_mut().for_each(|cluster| {
            let Some(cluster) = cluster else {
                return;
            };
            if cluster.growth_cache.is_none() {
                cluster.growth_cache = Some(best_growth_for_cluster(
                    &cluster.range,
                    seeds,
                    seed_trie,
                    range_mode,
                ));
            }
        });
    }

    fn select_best_growth(&mut self) -> Option<(usize, GrowthProposal)> {
        let mut winners: Vec<(usize, GrowthProposal)> = Vec::new();

        for (cluster_idx, cluster) in self.clusters.iter().enumerate() {
            let Some(cache) = cluster
                .as_ref()
                .and_then(|cluster| cluster.growth_cache.as_ref())
            else {
                continue;
            };
            for proposal in cache {
                match winners.first() {
                    None => winners.push((cluster_idx, *proposal)),
                    Some((_, best)) => match compare_proposals(proposal, best) {
                        std::cmp::Ordering::Greater => {
                            winners.clear();
                            winners.push((cluster_idx, *proposal));
                        }
                        std::cmp::Ordering::Equal => winners.push((cluster_idx, *proposal)),
                        std::cmp::Ordering::Less => {}
                    },
                }
            }
        }

        if winners.is_empty() {
            return None;
        }

        Some(winners[self.rng.gen_range(0..winners.len())])
    }

    fn admit_growth(&mut self, cluster_idx: usize, proposal: GrowthProposal) -> bool {
        let delta_fragments = self.coverage.uncovered_fragments(proposal.new_range);
        let delta_size = total_fragment_exact_size(&delta_fragments);

        if (self.coverage.target_count() as ExactSize).saturating_add(delta_size)
            > self.budget as ExactSize
        {
            let remaining = self.budget.saturating_sub(self.coverage.target_count());
            self.coverage
                .admit_final_sample(&delta_fragments, remaining, &mut self.rng);
            return false;
        }

        let cluster = self.clusters[cluster_idx]
            .as_mut()
            .expect("selected cluster must still be active");
        cluster.range = proposal.new_range;
        cluster.growth_cache = None;

        prune_subset_clusters(&mut self.clusters, cluster_idx);

        self.coverage.admit_fragments(delta_fragments);

        true
    }
}
