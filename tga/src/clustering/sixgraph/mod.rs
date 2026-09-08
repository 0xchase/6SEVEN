//! 6Graph density-based mining and feedback-driven one-Hamming scanning.

mod generation;
mod mining;
mod model;
mod pattern_index;
pub use model::SixGraphModel;
#[cfg(test)]
mod tests;

use mining::{MiningResult, mine_patterns, variable_nibble_indices};

use clap::Args;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashSet};

use crate::address::Address;
use crate::{Algorithm, Observation, TgaError};

const DEFAULT_MIN_REGION_SIZE: usize = 16;
const DEFAULT_DISTANCE_THRESHOLD: usize = 12;
const DEFAULT_ITERATIONS: usize = 3;
const DEFAULT_SAMPLE_SEED: u64 = 0x3647_2612_6f72_6170;
const NIBBLE_COUNT: usize = 32;
const HEX_RADIX: usize = 16;

/// A mined pattern: the variable nibble positions and the seed addresses.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Pattern {
    variable_nibbles: Vec<usize>,
    seeds: Vec<Address>,
}

impl Pattern {
    fn from_component(addresses: &[Address], component: &[usize]) -> Self {
        Self {
            variable_nibbles: variable_nibble_indices(addresses, component),
            seeds: component.iter().map(|&idx| addresses[idx]).collect(),
        }
    }

    fn seed_count(&self) -> usize {
        self.seeds.len()
    }

    fn fixed_mask(&self) -> u128 {
        self.variable_nibbles
            .iter()
            .fold(u128::MAX, |mask, &position| {
                mask & !(0xf << (4 * (NIBBLE_COUNT - 1 - position)))
            })
    }

    fn contains(&self, address: &Address) -> bool {
        let difference = u128::from_be_bytes(self.seeds[0]) ^ u128::from_be_bytes(*address);
        difference & self.fixed_mask() == 0
    }

    fn is_valid(&self) -> bool {
        self.seeds.len() >= 2
            && !self.variable_nibbles.is_empty()
            && self
                .variable_nibbles
                .iter()
                .all(|&position| position < NIBBLE_COUNT)
            && self
                .variable_nibbles
                .windows(2)
                .all(|pair| pair[0] < pair[1])
            && self.seeds.iter().all(|seed| self.contains(seed))
    }
}

struct TrainingAddresses {
    active: Vec<Address>,
    known: BTreeSet<Address>,
}

impl TrainingAddresses {
    fn from_observations(observations: &[Observation]) -> Self {
        let mut active_seen = HashSet::new();
        let mut active = Vec::new();
        let mut known = BTreeSet::new();

        for observation in observations {
            known.insert(observation.address);
            if observation.active && active_seen.insert(observation.address) {
                active.push(observation.address);
            }
        }

        Self { active, known }
    }
}

#[derive(Args, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SixGraph {
    /// Seed-count threshold for stopping DHC region partitioning.
    #[arg(long, default_value_t = DEFAULT_MIN_REGION_SIZE)]
    pub min_region_size: usize,

    /// Maximum nibble Hamming distance for candidate graph edges.
    #[arg(long, default_value_t = DEFAULT_DISTANCE_THRESHOLD)]
    pub distance_threshold: usize,

    /// Number of outlier-refinement rounds after the initial mining pass.
    #[arg(long, alias = "refinement-rounds", default_value_t = DEFAULT_ITERATIONS)]
    pub iterations: usize,

    /// Rejoin contained outliers as described in the paper but omitted by the reference code.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub seed_rejoining: bool,

    /// Seed for the paper-style random target sampling order.
    #[arg(long, default_value_t = DEFAULT_SAMPLE_SEED)]
    pub sample_seed: u64,
}

impl Default for SixGraph {
    fn default() -> Self {
        Self {
            min_region_size: DEFAULT_MIN_REGION_SIZE,
            distance_threshold: DEFAULT_DISTANCE_THRESHOLD,
            iterations: DEFAULT_ITERATIONS,
            sample_seed: DEFAULT_SAMPLE_SEED,
            seed_rejoining: true,
        }
    }
}

impl Algorithm for SixGraph {
    const ID: &'static str = "6graph";
    const DESCRIPTION: &'static str =
        "6Graph density-based clustering with feedback-driven one-Hamming generation";

    const MODEL_VERSION: u32 = 2;

    type Model = SixGraphModel;

    fn train(&self, observations: &[Observation]) -> Result<Self::Model, TgaError> {
        if self.min_region_size == 0 {
            return Err(TgaError::Config(
                "6Graph min_region_size must be positive".into(),
            ));
        }
        if self.distance_threshold > NIBBLE_COUNT {
            return Err(TgaError::Config(
                "6Graph distance_threshold must be in 0..=32".into(),
            ));
        }
        let training = TrainingAddresses::from_observations(observations);
        if training.active.is_empty() {
            return Err(TgaError::Training(
                "No active seed addresses provided".into(),
            ));
        }

        let MiningResult { patterns, outliers } = mine_patterns(training.active, self);
        Ok(SixGraphModel::new(
            patterns,
            training.known,
            outliers.len(),
            self.sample_seed,
        ))
    }
}
