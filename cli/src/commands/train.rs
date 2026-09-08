use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tga::{AlgorithmConfig, Observation};

use tracing::info;

use crate::commands::Command;

#[derive(Parser, Serialize, Deserialize)]
pub struct TrainCommand {
    /// Path to file containing seed addresses (one per line)
    #[arg(short, long)]
    pub seeds: Option<PathBuf>,

    /// Scan results CSV file to use as training observations
    #[arg(long = "scan-results")]
    pub scan_results: Option<PathBuf>,

    /// Output file to save the trained model
    #[arg(short = 'o', long = "output-model")]
    pub output_model: PathBuf,

    /// Maximum number of seeds to use (randomly samples if dataset is larger)
    #[arg(short = 'm', long)]
    pub max_seeds: Option<usize>,

    /// Seed for reproducible input sampling
    #[arg(long, default_value_t = 0)]
    pub sampling_seed: u64,

    /// TGA algorithm to use for training
    #[command(subcommand)]
    pub algorithm: Option<AlgorithmConfig>,
    #[arg(long = "algorithm")]
    pub algorithm_id: Option<String>,
    #[arg(long)]
    pub config: Option<PathBuf>,
}

impl Command for TrainCommand {
    fn run(&self, registry: &sixseven_core::Registry) -> Result<(), String> {
        match (&self.seeds, &self.scan_results) {
            (Some(seeds), None) => self.run_from_seeds(seeds, registry),
            (None, Some(scan_results)) => self.run_from_scan_results(scan_results, registry),
            (Some(_), Some(_)) => Err(
                "--seeds and --scan-results are mutually exclusive; provide one observation source"
                    .to_string(),
            ),
            (None, None) => Err("either --seeds or --scan-results must be provided".to_string()),
        }
    }
}

impl TrainCommand {
    fn spec(&self) -> Result<tga::AlgorithmSpec, String> {
        match (&self.algorithm, &self.algorithm_id) {
            (Some(algorithm), None) if self.config.is_none() => {
                algorithm.spec().map_err(|e| e.to_string())
            }
            (None, Some(id)) => {
                let config = match &self.config {
                    Some(path) => {
                        serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
                            .map_err(|e| e.to_string())?
                    }
                    None => serde_json::json!({}),
                };
                Ok(tga::AlgorithmSpec {
                    algorithm: tga::AlgorithmId::new(id).map_err(|e| e.to_string())?,
                    config,
                })
            }
            _ => Err("provide a built-in subcommand or --algorithm with optional --config".into()),
        }
    }

    fn run_from_seeds(
        &self,
        seeds_path: &std::path::Path,
        registry: &sixseven_core::Registry,
    ) -> Result<(), String> {
        info!(
            "Training {} from seeds file {}",
            self.spec()?.algorithm,
            seeds_path.display()
        );

        let seeds = load_seeds(seeds_path, self.max_seeds, self.sampling_seed)?;
        let observations = seeds
            .into_iter()
            .map(|address| Observation {
                address,
                active: true,
            })
            .collect();

        self.run_with_observations(observations, registry)
    }

    fn run_from_scan_results(
        &self,
        scan_results: &PathBuf,
        registry: &sixseven_core::Registry,
    ) -> Result<(), String> {
        info!(
            "Training {} from scan results {}",
            self.spec()?.algorithm,
            scan_results.display()
        );

        let observations =
            sixseven_formats::feedback::observations(scan_results).map_err(|e| e.to_string())?;

        self.run_with_observations(observations, registry)
    }

    fn run_with_observations(
        &self,
        observations: Vec<Observation>,
        registry: &sixseven_core::Registry,
    ) -> Result<(), String> {
        let artifact = registry
            .train(&self.spec()?, &observations)
            .map_err(|e| e.to_string())?;

        sixseven_formats::model::save(&self.output_model, &artifact).map_err(|e| e.to_string())?;
        Ok(())
    }
}

fn load_seeds(
    path: &std::path::Path,
    max_seeds: Option<usize>,
    sampling_seed: u64,
) -> Result<Vec<[u8; 16]>, String> {
    use rand::{Rng, SeedableRng};
    let seeds = (|| {
        let reader = sixseven_formats::targets::read(path, None).map_err(|e| e.to_string())?;
        let mut seeds = Vec::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(sampling_seed);
        for (index, address) in reader.enumerate() {
            let address = address.map_err(|e| e.to_string())?.octets();
            match max_seeds {
                Some(limit) if seeds.len() >= limit => {
                    let slot = rng.gen_range(0..=index);
                    if slot < limit {
                        seeds[slot] = address;
                    }
                }
                _ => seeds.push(address),
            }
        }
        Ok::<_, String>(seeds)
    })()?;

    if seeds.is_empty() {
        return Err("no valid seeds were loaded".to_string());
    }

    Ok(seeds)
}
