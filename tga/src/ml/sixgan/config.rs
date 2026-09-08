use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
pub enum SixGanClassification {
    None,
    RfcBased,
    EntropyClustering,
    Ipv62Vec,
}

#[derive(Debug, Clone, Args, Serialize, Deserialize)]
#[serde(default)]
pub struct SixGan {
    #[arg(long, default_value = "42")]
    pub seed: u64,

    #[arg(long)]
    pub feedback_budget: Option<usize>,

    #[arg(long, default_value_t = 50_000)]
    pub calibration_samples: usize,

    #[arg(long, default_value = "200")]
    pub emb_dim: usize,

    #[arg(long, default_value = "200")]
    pub hidden_dim: usize,

    #[arg(long, default_value_t = 64)]
    pub discriminator_emb_dim: usize,

    #[arg(long, default_value_t = 100)]
    pub discriminator_filters: usize,

    #[arg(long, default_value = "64")]
    pub batch_size: usize,

    #[arg(long, default_value = "60")]
    pub generator_pretrain_steps: usize,

    #[arg(long, default_value = "20")]
    pub discriminator_pretrain_steps: usize,

    #[arg(long, default_value = "800")]
    pub adversarial_rounds: usize,

    #[arg(long, default_value = "5")]
    pub generator_steps: usize,

    #[arg(long, default_value = "1")]
    pub discriminator_steps: usize,

    #[arg(long, default_value = "15")]
    pub rollout_num: usize,

    #[arg(long, default_value = "50000")]
    pub total_generation: usize,

    #[arg(long, value_enum, default_value_t = SixGanClassification::RfcBased)]
    pub classification: SixGanClassification,

    #[arg(long, default_value = "6")]
    pub entropy_k: usize,

    #[arg(long, default_value = "1.0")]
    pub temperature: f32,

    #[arg(long, value_delimiter = ',')]
    pub aliased_prefixes: Vec<ipnet::Ipv6Net>,

    #[arg(long, default_value_t = 0.9)]
    pub alias_alpha: f32,

    #[arg(long, default_value_t = 10.0)]
    pub alias_strength: f32,
}

impl Default for SixGan {
    fn default() -> Self {
        Self {
            seed: 42,
            feedback_budget: None,
            calibration_samples: 50_000,
            emb_dim: 200,
            hidden_dim: 200,
            discriminator_emb_dim: 64,
            discriminator_filters: 100,
            batch_size: 64,
            generator_pretrain_steps: 60,
            discriminator_pretrain_steps: 20,
            adversarial_rounds: 800,
            generator_steps: 5,
            discriminator_steps: 1,
            rollout_num: 15,
            total_generation: 50_000,
            classification: SixGanClassification::RfcBased,
            entropy_k: 6,
            temperature: 1.0,
            aliased_prefixes: Vec::new(),
            alias_alpha: 0.9,
            alias_strength: 10.0,
        }
    }
}

impl SixGan {
    pub(super) fn validate(&self) -> Result<(), String> {
        if self
            .feedback_budget
            .is_some_and(|budget| budget == 0 || budget as u128 > (1u128 << 53))
            || (self.feedback_budget.is_some() && self.calibration_samples == 0)
        {
            return Err(
                "feedback allocation requires positive calibration and a budget between 1 and 2^53"
                    .into(),
            );
        }
        if self.emb_dim == 0 || self.hidden_dim == 0 {
            return Err("generator dimensions must be positive".into());
        }
        if self.discriminator_emb_dim == 0 || self.discriminator_filters == 0 {
            return Err("discriminator dimensions must be positive".into());
        }
        if self.entropy_k == 0 {
            return Err("entropy_k must be positive".into());
        }
        if !self.alias_alpha.is_finite()
            || self.alias_alpha < 0.0
            || !self.alias_strength.is_finite()
            || self.alias_strength < 0.0
            || !(self.alias_alpha * self.alias_strength).is_finite()
        {
            return Err("alias penalty parameters must be finite and nonnegative".into());
        }
        if self
            .aliased_prefixes
            .iter()
            .any(|prefix| prefix.prefix_len() == 0 || prefix.prefix_len() % 4 != 0)
        {
            return Err("aliased prefixes must have positive nybble-aligned lengths".into());
        }
        if self.batch_size == 0 {
            return Err("batch_size must be at least 1".to_string());
        }
        if self.adversarial_rounds > 0 && self.generator_steps == 0 {
            return Err(
                "generator_steps must be at least 1 when adversarial_rounds is nonzero".to_string(),
            );
        }
        if self.adversarial_rounds > 0 && self.discriminator_steps == 0 {
            return Err(
                "discriminator_steps must be at least 1 when adversarial_rounds is nonzero"
                    .to_string(),
            );
        }
        if self.rollout_num == 0 {
            return Err("rollout_num must be at least 1".to_string());
        }
        if self.total_generation < self.batch_size {
            return Err("total_generation must be at least batch_size".to_string());
        }
        if self.temperature <= 0.0 || !self.temperature.is_finite() {
            return Err("temperature must be finite and greater than 0".to_string());
        }
        Ok(())
    }
}
