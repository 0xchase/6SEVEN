use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};

use super::bytes_to_nibbles;
use crate::address::Address;
use crate::{Algorithm, Observation, TgaError};

const DIMS: usize = 32;

type Nibbles = [u8; DIMS];
type ExactSize = u128;

fn cap_count(size: ExactSize) -> usize {
    size.min(usize::MAX as ExactSize) as usize
}

mod coverage;
mod growth;
mod index;
mod model;
mod range;
#[cfg(test)]
mod tests;

use growth::train_sixgen;
pub use model::SixGenModel;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
pub enum SixGenRangeMode {
    /// Dynamic nybbles expand to the full wildcard range `0-f`.
    Loose,
    /// Dynamic nybbles retain the exact observed nybble values from clustered seeds.
    Tight,
}

#[derive(Debug, Args, Clone, Serialize, Deserialize)]
pub struct SixGen {
    /// Maximum number of new targets to emit beyond the input seeds.
    #[arg(long, default_value_t = 1_000_000)]
    pub budget: usize,

    /// Cluster range semantics from the paper: loose (`?`) or tight nybble-value sets.
    #[arg(long, value_enum, default_value_t = SixGenRangeMode::Loose)]
    pub range_mode: SixGenRangeMode,

    /// Implementation-defined seed for deterministic tie-breaking and final sampling.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
}

impl Default for SixGen {
    fn default() -> Self {
        Self {
            budget: 1_000_000,
            range_mode: SixGenRangeMode::Loose,
            seed: 0,
        }
    }
}

impl Algorithm for SixGen {
    const ID: &'static str = "6gen";
    const DESCRIPTION: &'static str =
        "Paper-faithful 6Gen dense-cluster growth for IPv6 target generation";

    type Model = SixGenModel;

    fn train(&self, observations: &[Observation]) -> Result<Self::Model, TgaError> {
        let mut seeds: Vec<Address> = observations
            .iter()
            .filter(|obs| obs.active)
            .map(|obs| obs.address)
            .collect();
        if seeds.is_empty() {
            return Err(TgaError::Training("No seed addresses provided".into()));
        }

        seeds.sort_unstable();
        seeds.dedup();

        let nibble_seeds: Vec<Nibbles> = seeds.iter().map(bytes_to_nibbles).collect();
        Ok(train_sixgen(&nibble_seeds, self))
    }
}
