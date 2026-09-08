mod budget;
mod classify;
mod config;
mod generation;
mod ipv62vec;
mod models;
mod optimizer;
mod reward;
mod train;

use crate::{Address, Algorithm, Observation, TgaError};
pub use config::{SixGan, SixGanClassification};
pub use generation::SixGanModel;
use std::sync::{Arc, OnceLock};

impl SixGan {
    fn train_inner(&self, seeds: Vec<Address>) -> Result<SixGanModel, String> {
        self.validate()?;
        if seeds.len() < self.batch_size {
            return Err(format!(
                "6GAN requires at least {} active seeds, got {}",
                self.batch_size,
                seeds.len()
            ));
        }

        let training_seeds = seeds
            .into_iter()
            .take(self.total_generation)
            .collect::<Vec<_>>();
        if training_seeds.len() < self.batch_size {
            return Err(format!(
                "6GAN requires at least {} active seeds after the total_generation cap, got {}",
                self.batch_size,
                training_seeds.len()
            ));
        }

        let trained = train::train_model(self, &training_seeds)?;
        let budget = self.feedback_budget.map(|total| {
            budget::Budget::new(
                trained.len(),
                self.calibration_samples,
                total,
                &training_seeds,
                &self.aliased_prefixes,
            )
        });
        Ok(SixGanModel {
            budget,
            generation: Default::default(),
            generators: trained,
            emb_dim: self.emb_dim,
            hidden_dim: self.hidden_dim,
            generation_batch_size: self.batch_size,
            generation_temperature: self.temperature,
            sampling_seed: self.seed,
            classification: self.classification,
            runtime: Arc::new(OnceLock::new()),
        })
    }
}

impl Algorithm for SixGan {
    const MODEL_VERSION: u32 = 3;
    const ID: &'static str = "sixgan";
    const DESCRIPTION: &'static str =
        "6GAN multi-pattern IPv6 target generation with adversarial reinforcement learning";

    type Model = SixGanModel;

    fn migrate_model(version: u32, bytes: &[u8]) -> Result<Self::Model, TgaError> {
        if version == 2 {
            return generation::migrate_v2(bytes);
        }
        Err(TgaError::Model(format!(
            "unsupported 6GAN model version {version}, expected 2 or 3"
        )))
    }

    fn train(&self, observations: &[Observation]) -> Result<Self::Model, TgaError> {
        let positives = observations
            .iter()
            .filter(|obs| obs.active)
            .map(|obs| obs.address)
            .collect::<Vec<_>>();

        if positives.is_empty() {
            return Err(TgaError::Training(
                "6GAN training requires at least one active observation".to_string(),
            ));
        }

        self.train_inner(positives).map_err(TgaError::Training)
    }
}

#[cfg(test)]
mod tests;
