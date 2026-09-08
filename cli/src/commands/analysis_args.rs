use clap::Subcommand;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Subcommand, Serialize, Deserialize, Debug, Clone)]
pub enum AnalyzeCommand {
    /// Address space dispersion metrics
    Dispersion,
    /// Information entropy analysis
    Entropy {
        /// Start bit position (0-127) for entropy calculation
        #[arg(short = 's', long, value_parser = clap::value_parser!(u8).range(0..=127), default_value_t = 0)]
        start_bit: u8,

        /// End bit position (1-128) for entropy calculation
        #[arg(short = 'e', long, value_parser = clap::value_parser!(u8).range(1..=128), default_value_t = 128)]
        end_bit: u8,

        /// Path to scan result CSV files (can be specified multiple times)
        #[arg(long = "scan-results", value_name = "FILE")]
        scan_results: Vec<PathBuf>,

        /// Emit flat per-nibble entropy rows for heatmap generation
        #[arg(long = "output-heatmap")]
        output_heatmap: bool,
    },
    /// Bit-balance bias analysis
    Bias {
        /// Start bit position (0-127) for bias calculation
        #[arg(short = 's', long, value_parser = clap::value_parser!(u8).range(0..=127), default_value_t = 0)]
        start_bit: u8,

        /// End bit position (1-128) for bias calculation
        #[arg(short = 'e', long, value_parser = clap::value_parser!(u8).range(1..=128), default_value_t = 128)]
        end_bit: u8,
    },
    /// Count unique prefixes at a given prefix length
    Prefix {
        /// CIDR prefix length to count unique prefixes
        #[arg(short = 'l', long, value_parser = clap::value_parser!(u8).range(0..=128), default_value_t = 64)]
        prefix_length: u8,
    },
    /// Subnet distribution analysis
    Subnets {
        /// Maximum number of subnets to show (default: 10)
        #[arg(short = 'n', long, value_parser = clap::value_parser!(usize), default_value_t = 10)]
        max_subnets: usize,

        /// CIDR prefix length (default: 64)
        #[arg(short = 'l', long, value_parser = clap::value_parser!(u8).range(0..=128), default_value_t = 64)]
        prefix_length: u8,
    },
    /// Count addresses matching each predicate
    Counts,
    /// Categorize addresses from scan result CSVs
    Categories {
        /// Path to scan result CSV files (can be specified multiple times)
        #[arg(long = "scan-results", value_name = "FILE", required = true)]
        scan_results: Vec<PathBuf>,

        /// Write uncategorized addresses to this file
        #[arg(long = "output-uncategorized", value_name = "FILE")]
        output_uncategorized: Option<PathBuf>,
    },
    /// Plot cumulative success and error response probability by target index
    Rate {
        /// Scan result CSV files (must be paired with --targets in the same order)
        #[arg(long = "results", value_name = "FILE")]
        results: Vec<PathBuf>,

        /// Target CSV files (must be paired with --results in the same order)
        #[arg(long = "targets", value_name = "FILE")]
        targets: Vec<PathBuf>,

        /// Directory of scan result CSV files to pair recursively by relative path
        #[arg(long = "results-dir", value_name = "DIR")]
        results_dir: Option<PathBuf>,

        /// Directory of target CSV files to pair recursively by relative path
        #[arg(long = "targets-dir", value_name = "DIR")]
        targets_dir: Option<PathBuf>,

        /// Directory where PNG plots should be written
        #[arg(long = "output-dir", value_name = "DIR", default_value = "rate-plots")]
        output_dir: PathBuf,

        /// Record one plotted point every N targets
        #[arg(long = "sample-every", default_value_t = 10_000)]
        sample_every: usize,
    },
}
