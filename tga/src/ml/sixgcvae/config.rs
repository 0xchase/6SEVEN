use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};

pub const N: usize = 32;
// Dimensions follow the released reference implementation.
pub const LATENT_DIM: usize = 64;
pub const HIDDEN_DIM: usize = 64;
pub const VOCAB_SIZE: usize = 16;

pub const TRAIN_TEST_SPLIT: f64 = 0.2;
pub const TRAIN_TEST_SEED: u64 = 0;

pub const DEFAULT_EPOCHS: usize = 3;
pub const DEFAULT_BATCH_SIZE: usize = 64;

pub const ADAM_LEARNING_RATE: f64 = 1e-3;
pub const ADAM_BETA_1: f32 = 0.9;
pub const ADAM_BETA_2: f32 = 0.999;
pub const ADAM_EPSILON: f32 = 1e-7;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SixGcvaeClassification {
    #[default]
    None,
    Manual,
    Entropy,
}

/// 6GCVAE TGA configuration mirroring gcnn_vae.py defaults.
#[derive(Debug, Clone, Args, Serialize, Deserialize)]
pub struct SixGcvae {
    /// Number of training epochs (reference default: 3)
    #[arg(long, default_value_t = DEFAULT_EPOCHS)]
    #[serde(default = "default_epochs")]
    pub epochs: usize,

    /// Seed for latent sampling during generation
    #[arg(long, default_value_t = 0)]
    #[serde(default)]
    pub generation_seed: u64,

    /// Training batch size (reference default: 64)
    #[arg(long, default_value_t = DEFAULT_BATCH_SIZE)]
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// Seed classification before training separate decoders
    #[arg(long, value_enum, default_value = "none")]
    #[serde(default)]
    pub classification: SixGcvaeClassification,

    /// Number of entropy clusters (paper experiment: 6)
    #[arg(long, default_value_t = default_clusters())]
    #[serde(default = "default_clusters")]
    pub clusters: usize,

    /// Draws per calibration category and per later feedback round
    #[arg(long, default_value_t = default_feedback_samples())]
    #[serde(default = "default_feedback_samples")]
    pub feedback_samples: usize,
}

impl Default for SixGcvae {
    fn default() -> Self {
        Self {
            epochs: DEFAULT_EPOCHS,
            generation_seed: 0,
            batch_size: DEFAULT_BATCH_SIZE,
            classification: SixGcvaeClassification::None,
            clusters: default_clusters(),
            feedback_samples: default_feedback_samples(),
        }
    }
}

impl SixGcvae {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.epochs == 0 {
            return Err("epochs must be at least 1".to_string());
        }
        if self.batch_size == 0 {
            return Err("batch_size must be at least 1".to_string());
        }
        if self.classification == SixGcvaeClassification::Entropy && self.clusters == 0 {
            return Err("clusters must be at least 1".into());
        }
        if self.feedback_samples == 0 {
            return Err("feedback_samples must be at least 1".into());
        }
        Ok(())
    }
}

fn default_feedback_samples() -> usize {
    1_000_000
}

fn default_clusters() -> usize {
    6
}

fn default_epochs() -> usize {
    DEFAULT_EPOCHS
}

fn default_batch_size() -> usize {
    DEFAULT_BATCH_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn architecture_constants_match_reference_code() {
        assert_eq!(N, 32);
        assert_eq!(LATENT_DIM, 64);
        assert_eq!(HIDDEN_DIM, 64);
        assert_eq!(VOCAB_SIZE, 16);
    }

    #[test]
    fn optimizer_defaults_match_keras_adam() {
        assert_eq!(ADAM_LEARNING_RATE, 1e-3);
        assert_eq!(ADAM_BETA_1, 0.9);
        assert_eq!(ADAM_BETA_2, 0.999);
        assert_eq!(ADAM_EPSILON, 1e-7);
    }

    #[test]
    fn split_defaults_match_reference_pipeline() {
        assert_eq!(TRAIN_TEST_SPLIT, 0.2);
        assert_eq!(TRAIN_TEST_SEED, 0);
    }

    #[test]
    fn config_defaults_match_reference_code() {
        let cfg = SixGcvae::default();
        assert_eq!(cfg.epochs, DEFAULT_EPOCHS);
        assert_eq!(cfg.batch_size, DEFAULT_BATCH_SIZE);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_rejects_degenerate_training_loops() {
        assert!(
            SixGcvae {
                epochs: 0,
                ..SixGcvae::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            SixGcvae {
                batch_size: 0,
                ..SixGcvae::default()
            }
            .validate()
            .is_err()
        );
    }
}
