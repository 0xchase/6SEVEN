use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum SixProbeMode {
    Forest,
    SingleTree,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum DhcType {
    LeftVdps,
    RightVdps,
    MinEntropy,
    MaxCover,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum SplitArrayType {
    Random,
    Sequential,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum SplitOrder {
    Left,
    Right,
}

#[derive(Args, Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct SixProbe {
    /// Minimum seeds for DHC recursion, defaulting to the reference value of 12.
    #[arg(long, default_value_t = 12)]
    pub beta: usize,

    /// Additional 6ASTrees after the initial LeftVDPS tree, defaulting to 40.
    #[arg(long, default_value_t = 40)]
    pub tree_num: usize,

    /// Execution mode for the 6Probe forest or individual DHC strategy analysis.
    #[arg(long, value_enum, default_value_t = SixProbeMode::Forest)]
    pub mode: SixProbeMode,

    /// DHC strategy for single-tree mode, defaulting to the reference LeftVDPS strategy.
    #[arg(long, value_enum, default_value_t = DhcType::LeftVdps)]
    pub dhc_type: DhcType,

    /// Forest split-array strategy, defaulting to the paper's optimized random mode.
    #[arg(long, value_enum, default_value_t = SplitArrayType::Random)]
    pub split_array_type: SplitArrayType,

    /// Optional Python-compatible seed for reproducing reference forest split arrays.
    #[arg(long)]
    pub random_seed: Option<u64>,

    /// Forest split direction, defaulting to the paper's optimized right mode.
    #[arg(long, value_enum, default_value_t = SplitOrder::Right)]
    pub split_order: SplitOrder,

    /// Optional path for exporting mined low-dimensional patterns.
    #[arg(long)]
    pub export_patterns: Option<PathBuf>,

    #[arg(
        long,
        help = "File of aliased IPv6 prefixes to exclude from generated targets"
    )]
    #[serde(default)]
    pub aliased_prefixes: Option<PathBuf>,
}

impl Default for SixProbe {
    fn default() -> Self {
        Self {
            beta: 12,
            tree_num: 40,
            mode: SixProbeMode::Forest,
            dhc_type: DhcType::LeftVdps,
            split_array_type: SplitArrayType::Random,
            random_seed: None,
            split_order: SplitOrder::Right,
            export_patterns: None,
            aliased_prefixes: None,
        }
    }
}
