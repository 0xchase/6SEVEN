mod analysis_args;
pub mod analyze;
pub mod common;
pub mod feedback;
pub mod generate;
pub mod scan;
pub mod train;

pub use analyze::AnalyzeCommandArgs;
pub use feedback::FeedbackCommand;
pub use generate::GenerateCommand;
pub use scan::ScanCommand;
use serde::{Deserialize, Serialize};
pub use train::TrainCommand;

pub trait Command: Serialize + for<'de> Deserialize<'de> {
    fn run(&self, registry: &sixseven_core::Registry) -> Result<(), String>;
}

pub mod dealias;
mod runtime_args;
