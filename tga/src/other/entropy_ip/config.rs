use clap::{ArgAction, Args};
use serde::{Deserialize, Serialize};

use crate::TgaError;

const DEFAULT_SIZE: usize = 32;
const DEFAULT_STEP: usize = 1;
const DEFAULT_ISP_NYBBLES: usize = 8;
const DEFAULT_NET_NYBBLES: usize = 16;
const DEFAULT_THRESHOLDS: [f64; 5] = [0.025, 0.1, 0.3, 0.5, 0.9];
const DEFAULT_HYSTERESIS: f64 = 0.05;
const DEFAULT_SEGMENT_SAMPLE_SIZE: usize = 50_000;
const DEFAULT_BNF_SAMPLE_SIZE: usize = 100_000;
const DEFAULT_BNF_FULL: bool = false;
const DEFAULT_RCODE: bool = false;
const DEFAULT_DROP_UNKNOWN: bool = true;

#[derive(Args, Serialize, Deserialize, Clone)]
pub struct EntropyIp {
    /// Optional maximum number of parents per node in the Bayesian network.
    #[arg(long)]
    pub max_parents: Option<usize>,
    /// Total address length in nybbles.
    #[arg(long, default_value_t = DEFAULT_SIZE)]
    pub size: usize,
    /// Step size in nybbles for entropy analysis.
    #[arg(long, default_value_t = DEFAULT_STEP)]
    pub step: usize,
    /// ISP prefix size in nybbles (hard boundary).
    #[arg(long, default_value_t = DEFAULT_ISP_NYBBLES)]
    pub isp_nybbles: usize,
    /// Network-id prefix size in nybbles (hard boundary).
    #[arg(long, default_value_t = DEFAULT_NET_NYBBLES)]
    pub net_nybbles: usize,
    /// Thresholds used by entropy segmentation.
    #[arg(long, value_delimiter = ',', default_values_t = DEFAULT_THRESHOLDS)]
    pub thresholds: Vec<f64>,
    /// Hysteresis used for threshold crossings.
    #[arg(long, default_value_t = DEFAULT_HYSTERESIS)]
    pub hysteresis: f64,
    /// Per-segment sample size cap used by mining.
    #[arg(long, default_value_t = DEFAULT_SEGMENT_SAMPLE_SIZE)]
    pub segment_sample_size: usize,
    /// BN training sample size cap (rewrite-bnf.py uses 100K by default).
    #[arg(long, default_value_t = DEFAULT_BNF_SAMPLE_SIZE)]
    pub bnf_sample_size: usize,
    /// Use the full encoded dataset instead of sampling for BN training.
    #[arg(long, default_value_t = DEFAULT_BNF_FULL)]
    pub bnf_full: bool,
    /// Encode unknown segment values into an extra code.
    #[arg(long, default_value_t = DEFAULT_RCODE)]
    pub rcode: bool,
    /// Drop rows that contain unknown segment values.
    #[arg(long, default_value_t = DEFAULT_DROP_UNKNOWN, action = ArgAction::Set)]
    pub drop_unknown: bool,
    /// Fixed RNG seed for deterministic sampling/generation.
    #[arg(long)]
    pub rng_seed: Option<u64>,
}

impl Default for EntropyIp {
    fn default() -> Self {
        Self {
            max_parents: None,
            size: DEFAULT_SIZE,
            step: DEFAULT_STEP,
            isp_nybbles: DEFAULT_ISP_NYBBLES,
            net_nybbles: DEFAULT_NET_NYBBLES,
            thresholds: DEFAULT_THRESHOLDS.to_vec(),
            hysteresis: DEFAULT_HYSTERESIS,
            segment_sample_size: DEFAULT_SEGMENT_SAMPLE_SIZE,
            bnf_sample_size: DEFAULT_BNF_SAMPLE_SIZE,
            bnf_full: DEFAULT_BNF_FULL,
            rcode: DEFAULT_RCODE,
            drop_unknown: DEFAULT_DROP_UNKNOWN,
            rng_seed: None,
        }
    }
}

impl EntropyIp {
    pub(super) fn validate(&self) -> Result<(), TgaError> {
        if self.step == 0 {
            return Err(TgaError::Training(
                "step must be greater than zero.".to_string(),
            ));
        }
        if self.size == 0 || self.size > 32 {
            return Err(TgaError::Training(
                "size must be in the range 1..=32 nybbles.".to_string(),
            ));
        }
        if !self.size.is_multiple_of(self.step) {
            return Err(TgaError::Training(
                "size must be divisible by step.".to_string(),
            ));
        }
        if self.isp_nybbles > self.size || self.net_nybbles > self.size {
            return Err(TgaError::Training(
                "isp-nybbles and net-nybbles must be <= size.".to_string(),
            ));
        }
        if self.isp_nybbles > self.net_nybbles {
            return Err(TgaError::Training(
                "isp-nybbles must be <= net-nybbles so Entropy/IP can preserve the paper's /32 then /64 hard boundaries.".to_string(),
            ));
        }
        if !self.isp_nybbles.is_multiple_of(self.step)
            || !self.net_nybbles.is_multiple_of(self.step)
        {
            return Err(TgaError::Training(
                "isp-nybbles and net-nybbles must be divisible by step.".to_string(),
            ));
        }
        if !self.hysteresis.is_finite() || self.hysteresis < 0.0 {
            return Err(TgaError::Training(
                "hysteresis must be a finite non-negative value.".to_string(),
            ));
        }
        if self.thresholds.is_empty() {
            return Err(TgaError::Training(
                "threshold list must contain at least one value.".to_string(),
            ));
        }
        if self
            .thresholds
            .iter()
            .any(|threshold| !threshold.is_finite() || *threshold < 0.0 || *threshold > 1.0)
        {
            return Err(TgaError::Training(
                "thresholds must be finite values in the range 0.0..=1.0.".to_string(),
            ));
        }
        if self.segment_sample_size == 0 || self.bnf_sample_size == 0 {
            return Err(TgaError::Training(
                "segment-sample-size and bnf-sample-size must be greater than zero.".to_string(),
            ));
        }
        if !self.rcode && !self.drop_unknown {
            return Err(TgaError::Training(
                "drop_unknown=false without rcode is unsupported; original Entropy/IP either drops unknown rows or encodes them into an extra state.".to_string(),
            ));
        }
        Ok(())
    }

    pub(super) fn effective_drop_unknown(&self) -> bool {
        if self.rcode { false } else { self.drop_unknown }
    }

    pub(super) fn effective_max_parents(&self, segment_idx: usize) -> usize {
        self.max_parents.unwrap_or(segment_idx).min(segment_idx)
    }
}
