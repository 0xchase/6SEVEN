pub mod csv;
pub mod feedback;
pub mod model;
pub mod targets;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Csv(#[from] ::csv::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("invalid data: {0}")]
    Invalid(String),
}

pub mod prefixes;
