//! 6Forest space partition, isolation weighting, and hit rate prescanning.

mod outlier;
mod partition;
mod prescan;
mod region;
mod sampling;

#[cfg(test)]
use crate::AddressStream;
use crate::{Address, AddressExt, Algorithm, Observation, TgaError};
use clap::Args;
use outlier::detect_outliers;
use partition::partition;
use rayon::prelude::*;
use region::{SixForestRegion, SixForestTargetSet};
use sampling::SixForestStream;
use serde::{Deserialize, Serialize};

const NIBBLE_COUNT: usize = 32;
const HEX_RADIX: usize = 16;
const DEFAULT_MIN_REGION_SIZE: usize = 16;
const DEFAULT_GENERATION_SEED: u64 = 0;
const DEFAULT_PRESCAN_SAMPLES: usize = 100;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SixForestModel {
    #[serde(skip)]
    pub(crate) generation: crate::cursor::GenerationCursor<SixForestStream>,
    space: SixForestTargetSet,
    #[serde(default)]
    prescan: Option<prescan::Prescan>,
    #[serde(default)]
    input_seed_count: usize,
    /// Number of removed seed rows.
    #[serde(default)]
    outlier_count: usize,
}

impl std::fmt::Display for SixForestModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cluster_count = self.space.regions.len();
        let normal_seed_count = self
            .space
            .regions
            .iter()
            .map(|region| region.seed_count)
            .sum::<usize>();
        let total_seed_count = if self.input_seed_count == 0 {
            normal_seed_count + self.outlier_count
        } else {
            self.input_seed_count
        };
        let avg_free_nibbles = if self.space.regions.is_empty() {
            0.0
        } else {
            self.space
                .regions
                .iter()
                .map(|region| region.free_nibbles() as f64)
                .sum::<f64>()
                / self.space.regions.len() as f64
        };

        write!(
            f,
            "6Forest model with {} clusters, {} outliers, {} total seeds, avg free nibbles per pattern: {:.1}",
            cluster_count, self.outlier_count, total_seed_count, avg_free_nibbles
        )
    }
}

#[derive(Args, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SixForest {
    /// Seed count threshold beta for stopping space partition.
    #[arg(long, default_value_t = DEFAULT_MIN_REGION_SIZE)]
    pub min_region_size: usize,

    /// Seed for deterministic pseudo-random sampling from mined regions.
    #[arg(long, default_value_t = DEFAULT_GENERATION_SEED)]
    pub generation_seed: u64,

    /// Prescan samples per large region, or zero for offline generation.
    #[arg(long, default_value_t = DEFAULT_PRESCAN_SAMPLES)]
    pub prescan_samples: usize,
}

impl Default for SixForest {
    fn default() -> Self {
        Self {
            min_region_size: DEFAULT_MIN_REGION_SIZE,
            generation_seed: DEFAULT_GENERATION_SEED,
            prescan_samples: DEFAULT_PRESCAN_SAMPLES,
        }
    }
}

impl Algorithm for SixForest {
    const ID: &'static str = "6forest";
    const DESCRIPTION: &'static str = "6Forest max-coverage DHC, isolation-forest outlier filtering, and region-sampling target emission";

    const MODEL_VERSION: u32 = 3;

    type Model = SixForestModel;

    fn migrate_model(version: u32, bytes: &[u8]) -> Result<Self::Model, TgaError> {
        #[derive(Deserialize)]
        struct LegacyModel {
            space: SixForestTargetSet,
            input_seed_count: usize,
            outlier_count: usize,
        }
        #[derive(Deserialize)]
        struct FeedbackModel {
            space: SixForestTargetSet,
            prescan: Option<prescan::LegacyPrescan>,
            input_seed_count: usize,
            outlier_count: usize,
        }
        let error = |error| TgaError::Model(format!("decode 6Forest v{version} model: {error}"));
        match version {
            1 => {
                let legacy: LegacyModel = bincode::deserialize(bytes).map_err(error)?;
                Ok(SixForestModel {
                    space: legacy.space,
                    input_seed_count: legacy.input_seed_count,
                    outlier_count: legacy.outlier_count,
                    ..Default::default()
                })
            }
            2 => {
                let legacy: FeedbackModel = bincode::deserialize(bytes).map_err(error)?;
                let prescan = legacy.prescan.map(|prescan| prescan.migrate(&legacy.space));
                Ok(SixForestModel {
                    space: legacy.space,
                    prescan,
                    input_seed_count: legacy.input_seed_count,
                    outlier_count: legacy.outlier_count,
                    ..Default::default()
                })
            }
            _ => Err(TgaError::Model(format!(
                "unsupported 6Forest model version {version}"
            ))),
        }
    }

    fn train(&self, observations: &[Observation]) -> Result<Self::Model, TgaError> {
        self.validate()?;
        self.train_from_seeds(observations)
    }
}

impl SixForest {
    fn validate(&self) -> Result<(), TgaError> {
        if self.min_region_size == 0 {
            return Err(TgaError::Config(
                "6Forest min_region_size must be greater than zero".into(),
            ));
        }
        Ok(())
    }

    fn train_from_seeds(&self, observations: &[Observation]) -> Result<SixForestModel, TgaError> {
        let mut seeds: Vec<Address> = observations
            .iter()
            .filter(|obs| obs.active)
            .map(|obs| obs.address)
            .collect();
        seeds.sort_unstable();
        seeds.dedup();
        if seeds.is_empty() {
            return Err(TgaError::Training("No seed addresses provided".into()));
        }

        // Algorithm 1 partitions the sorted seed set in breadth-first order.
        let regions = partition(&seeds, self.min_region_size);

        let mined: Vec<_> = regions
            .into_par_iter()
            .map(|region| {
                let filtered = detect_outliers(&region);
                let pattern = (!filtered.normal.is_empty())
                    .then(|| SixForestRegion::from_addresses(&filtered.normal));
                (pattern, filtered.outlier_count)
            })
            .collect();
        let outlier_count = mined.iter().map(|(_, count)| count).sum();
        let regions = mined
            .into_iter()
            .filter_map(|(pattern, _)| pattern)
            .collect();

        let input_seed_count = seeds.len();
        let space = SixForestTargetSet::new(regions, seeds, self.generation_seed);
        let prescan =
            (self.prescan_samples > 0).then(|| prescan::Prescan::new(&space, self.prescan_samples));
        Ok(SixForestModel {
            generation: Default::default(),
            space,
            prescan,
            input_seed_count,
            outlier_count,
        })
    }
}

impl SixForestModel {
    #[cfg(test)]
    pub(crate) fn stream(&self) -> Result<AddressStream, TgaError> {
        Ok(Box::new(self.space.stream().map(Ok)))
    }
}
impl crate::TargetModel for SixForestModel {
    fn set_budget(&mut self, budget: usize) {
        if let Some(prescan) = &mut self.prescan {
            prescan.set_budget(budget);
        }
    }
    fn generate(&mut self, output: &mut [crate::Address]) -> Result<crate::Generated, TgaError> {
        sixseven_core::generation::validate_output(output)?;
        if let Some(prescan) = &mut self.prescan {
            return Ok(prescan.generate(output));
        }
        if self.generation.0.is_none() {
            self.generation.0 = Some(self.space.stream());
        }
        crate::cursor::fill(
            self.generation.0.as_mut().expect("initialized cursor"),
            output,
            crate::GenerationState::Exhausted,
        )
    }

    fn apply_feedback(&mut self, feedback: &[crate::Feedback]) -> Result<(), TgaError> {
        let prescan = self.prescan.as_mut().ok_or_else(|| {
            TgaError::Unsupported(
                "6Forest feedback requires prescan_samples greater than zero".into(),
            )
        })?;
        prescan.apply_feedback(feedback);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
fn nibble_to_hex(value: u8) -> char {
    debug_assert!((value as usize) < HEX_RADIX);
    char::from_digit(value as u32, 16).unwrap()
}
