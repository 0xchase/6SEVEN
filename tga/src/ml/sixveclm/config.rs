use super::{
    DEFAULT_EMBEDDING_BATCH_SIZE, DEFAULT_GENERATION_TEMPERATURE, DEFAULT_TRANSFORMER_BATCH_SIZE,
};
use clap::Args;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Args, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SixVecLm {
    #[arg(long, default_value = "1")]
    pub seed: u64,

    #[arg(long, default_value_t = 100)]
    pub embedding_dim: usize,

    #[arg(long, default_value_t = 5)]
    pub embedding_window: usize,

    #[arg(long, default_value_t = 5)]
    pub embedding_epochs: usize,

    #[arg(long, default_value_t = 10)]
    pub transformer_epochs: usize,

    #[arg(long, default_value_t = DEFAULT_EMBEDDING_BATCH_SIZE)]
    pub embedding_batch_size: usize,

    #[arg(long = "transformer-batch-size", alias = "batch-size", default_value_t = DEFAULT_TRANSFORMER_BATCH_SIZE)]
    #[serde(alias = "batch_size")]
    pub transformer_batch_size: usize,

    #[arg(long, default_value_t = 6)]
    pub layers: usize,

    #[arg(long, default_value_t = 10)]
    pub heads: usize,

    #[arg(long, default_value_t = 2048)]
    pub ff_dim: usize,

    #[arg(long, default_value = "0.1")]
    pub dropout: f64,

    #[arg(long, default_value = "0.025")]
    pub embedding_lr: f64,

    #[arg(long, default_value = "0.0001")]
    pub embedding_min_lr: f64,

    #[arg(long, default_value = "1.0")]
    pub noam_factor: f64,

    #[arg(long, default_value_t = 400)]
    pub noam_warmup: usize,

    #[arg(long, default_value_t = DEFAULT_GENERATION_TEMPERATURE)]
    pub generation_temperature: f32,
}

impl Default for SixVecLm {
    fn default() -> Self {
        Self {
            seed: 1,
            embedding_dim: 100,
            embedding_window: 5,
            embedding_epochs: 5,
            transformer_epochs: 10,
            embedding_batch_size: DEFAULT_EMBEDDING_BATCH_SIZE,
            transformer_batch_size: DEFAULT_TRANSFORMER_BATCH_SIZE,
            layers: 6,
            heads: 10,
            ff_dim: 2048,
            dropout: 0.1,
            embedding_lr: 0.025,
            embedding_min_lr: 0.0001,
            noam_factor: 1.0,
            noam_warmup: 400,
            generation_temperature: DEFAULT_GENERATION_TEMPERATURE,
        }
    }
}

impl SixVecLm {
    pub(super) fn validate(&self) -> Result<(), String> {
        if self.embedding_dim < 2 {
            return Err("embedding_dim must be at least 2".into());
        }
        if self.embedding_window < 3
            || self.embedding_window > 2 * super::TOTAL_LEN - 1
            || self.embedding_window.is_multiple_of(2)
        {
            return Err("embedding_window must be an odd width in 3..64".into());
        }
        if self.embedding_epochs == 0 {
            return Err("embedding_epochs must be at least 1".into());
        }
        if self.transformer_epochs == 0 {
            return Err("transformer_epochs must be at least 1".into());
        }
        if self.embedding_batch_size == 0 {
            return Err("embedding_batch_size must be at least 1".into());
        }
        if self.transformer_batch_size == 0 {
            return Err("transformer_batch_size must be at least 1".into());
        }
        if self.layers == 0 {
            return Err("layers must be at least 1".into());
        }
        if self.heads == 0 {
            return Err("heads must be at least 1".into());
        }
        if !self.embedding_dim.is_multiple_of(self.heads) {
            return Err(format!(
                "embedding_dim ({}) must be divisible by heads ({})",
                self.embedding_dim, self.heads
            ));
        }
        if self.ff_dim == 0 {
            return Err("ff_dim must be at least 1".into());
        }
        if !self.dropout.is_finite() || !(0.0..1.0).contains(&self.dropout) {
            return Err("dropout must be a finite value in [0, 1)".into());
        }
        if !self.embedding_lr.is_finite() || self.embedding_lr <= 0.0 {
            return Err("embedding_lr must be a finite value > 0".into());
        }
        if !self.embedding_min_lr.is_finite() || self.embedding_min_lr <= 0.0 {
            return Err("embedding_min_lr must be a finite value > 0".into());
        }
        if self.embedding_min_lr > self.embedding_lr {
            return Err("embedding_min_lr must be less than or equal to embedding_lr".into());
        }
        if !self.noam_factor.is_finite() || self.noam_factor <= 0.0 {
            return Err("noam_factor must be a finite value > 0".into());
        }
        if self.noam_warmup == 0 {
            return Err("noam_warmup must be at least 1".into());
        }
        if !self.generation_temperature.is_finite() || self.generation_temperature <= 0.0 {
            return Err("generation_temperature must be a finite value > 0".into());
        }
        Ok(())
    }
}
