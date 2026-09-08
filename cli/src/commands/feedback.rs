use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::info;

use crate::commands::Command;

#[derive(Parser, Serialize, Deserialize)]
pub struct FeedbackCommand {
    /// Existing model to update from scan feedback
    #[arg(long = "input-model")]
    pub input_model: PathBuf,

    /// Scan results CSV or feedback JSONL journal
    #[arg(long = "scan-results")]
    pub scan_results: PathBuf,

    /// Output file to save the updated model
    #[arg(short = 'o', long = "output-model")]
    pub output_model: PathBuf,
}

impl Command for FeedbackCommand {
    fn run(&self, registry: &sixseven_core::Registry) -> Result<(), String> {
        info!(
            "Applying scan feedback from {} to model {}",
            self.scan_results.display(),
            self.input_model.display()
        );

        let mut model =
            sixseven_formats::model::load(&self.input_model).map_err(|e| e.to_string())?;
        let batches = sixseven_formats::feedback::load_batches(&self.scan_results)
            .map_err(|e| e.to_string())?;
        let mut loaded = registry.open(&model).map_err(|e| e.to_string())?;
        for feedback in batches {
            loaded
                .apply_feedback(&feedback)
                .map_err(|e| e.to_string())?;
        }
        registry
            .save_model(&mut model, loaded.as_ref())
            .map_err(|e| e.to_string())?;

        sixseven_formats::model::save(&self.output_model, &model).map_err(|e| e.to_string())
    }
}
