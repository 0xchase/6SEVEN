use std::collections::HashSet;

use super::config::{DhcType, SixProbe, SixProbeMode, SplitOrder};
use super::encoding::{NibbleAddr, Subspace, compute_subspace};
use super::pattern_mining::PatternMiner;
use super::split::{
    SplitArray, generate_split_arrays, split_by_dhc_type, split_by_order_with_index,
};

enum Partition<'a> {
    Strategy(DhcType),
    Schedule(&'a SplitArray, SplitOrder),
}

impl Partition<'_> {
    fn split(&self, seeds: &[NibbleAddr], level: usize) -> Vec<Vec<usize>> {
        match *self {
            Self::Strategy(strategy) => split_by_dhc_type(seeds, strategy),
            Self::Schedule(array, order) => {
                split_by_order_with_index(seeds, order, array.relative_position_for_level(level))
            }
        }
    }
}

#[derive(Default)]
struct Forest {
    seen_nodes: HashSet<Subspace>,
    miner: PatternMiner,
}

pub(crate) fn build_patterns(seeds: &[NibbleAddr], config: &SixProbe) -> Vec<Subspace> {
    if seeds.is_empty() {
        return Vec::new();
    }
    let mut forest = Forest::default();
    let (subspace, dimension) = compute_subspace(seeds);
    let strategy = match config.mode {
        SixProbeMode::SingleTree => config.dhc_type,
        SixProbeMode::Forest => DhcType::LeftVdps,
    };
    forest.visit(
        seeds,
        subspace,
        dimension,
        config.beta,
        1,
        &Partition::Strategy(strategy),
    );

    if matches!(config.mode, SixProbeMode::Forest) && dimension > 0 {
        for array in generate_split_arrays(
            dimension,
            config.tree_num,
            config.split_array_type,
            config.random_seed,
        ) {
            forest.visit(
                seeds,
                subspace,
                dimension,
                config.beta,
                1,
                &Partition::Schedule(&array, config.split_order),
            );
        }
    }
    forest.miner.into_patterns()
}

impl Forest {
    fn visit(
        &mut self,
        seeds: &[NibbleAddr],
        subspace: Subspace,
        dimension: usize,
        beta: usize,
        level: usize,
        partition: &Partition<'_>,
    ) {
        // Dimension zero also terminates for repeated identical observations.
        if dimension <= 1 || seeds.len() < beta {
            // Mine in traversal order without retaining copies of all leaves.
            self.miner.mine(seeds, subspace, dimension);
            return;
        }
        for indices in partition.split(seeds, level) {
            let child: Vec<_> = indices.into_iter().map(|index| seeds[index]).collect();
            let (space, dimensions) = compute_subspace(&child);
            if dimensions > 0 && self.seen_nodes.insert(space) {
                self.visit(&child, space, dimensions, beta, level + 1, partition);
            }
        }
    }
}
