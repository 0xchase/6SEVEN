use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::commands::{
    AnalyzeCommandArgs, Command, FeedbackCommand, GenerateCommand, ScanCommand, TrainCommand,
};

#[derive(Parser)]
#[command(
    name = "6seven",
    about = "IPv6 network scanning and analysis toolkit",
    version = "0.1.0",
    author = "Chase Kanipe"
)]
pub struct Cli {
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    #[arg(long, value_name = "PATH")]
    pub plugins: Option<PathBuf>,

    #[arg(short, long, value_name = "LOG_FILE")]
    pub log: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Serialize, Deserialize)]
pub enum Commands {
    /// Probe a fixed set of targets
    Scan(ScanCommand),
    /// Filter known aliases or actively detect aliased IPv6 prefixes
    Dealias(crate::commands::dealias::DealiasCommand),
    /// Train a new model of the address space
    Train(TrainCommand),
    /// Update an existing model from scan feedback
    Feedback(FeedbackCommand),
    /// Generate IPv6 targets from trained models or pseudorandom components
    Generate(GenerateCommand),
    /// Analyze data with various metrics
    Analyze(AnalyzeCommandArgs),
}

impl Commands {
    pub fn run(&self, registry: &sixseven_core::Registry) -> Result<(), String> {
        match self {
            Commands::Scan(cmd) => tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|e| e.to_string())?
                .block_on(cmd.execute(registry))
                .map_err(|e| e.to_string()),
            Commands::Dealias(cmd) => tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|e| e.to_string())?
                .block_on(cmd.execute()),
            Commands::Train(cmd) => cmd.run(registry),
            Commands::Feedback(cmd) => cmd.run(registry),
            Commands::Generate(cmd) => cmd.run(registry),
            Commands::Analyze(cmd) => cmd.run(registry),
        }
    }
}
