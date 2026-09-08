mod model;
mod pattern;
#[cfg(test)]
mod tests;
mod tree;

use crate::{Algorithm, Observation, TgaError};
use clap::Args;
pub use model::DetModel;
use serde::{Deserialize, Serialize};
use tree::{build_model, resolve_bits_per_dim};

#[derive(Args, Clone, Serialize, Deserialize)]
pub struct Det {
    /// Address vector base.
    #[arg(long = "delta-base", default_value_t = 16)]
    pub delta_base: usize,

    /// Maximum seed vectors per leaf.
    #[arg(long = "leaf-max", default_value_t = 16)]
    pub leaf_max: usize,
}

impl Algorithm for Det {
    const MODEL_VERSION: u32 = 2;
    const ID: &'static str = "det";
    const DESCRIPTION: &'static str =
        "DET entropy-guided IPv6 target generation from feedback-scheduled density-tree batches.";

    type Model = DetModel;

    fn train(&self, observations: &[Observation]) -> Result<Self::Model, TgaError> {
        let bits_per_dimension = resolve_bits_per_dim(self.delta_base)?;
        if self.leaf_max == 0 {
            return Err(TgaError::Training("`leaf_max` must be at least 1".into()));
        }

        let seeds = observations
            .iter()
            .filter(|observation| observation.active)
            .map(|observation| observation.address)
            .collect::<Vec<_>>();

        if seeds.is_empty() {
            return Err(TgaError::Training(
                "DET initialization requires at least one active seed address".into(),
            ));
        }

        Ok(build_model(&seeds, bits_per_dimension, self.leaf_max))
    }
}
