use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use std::collections::BTreeMap;
use std::net::Ipv6Addr;
pub type Address = [u8; 16];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationState {
    Ready,
    AwaitingFeedback,
    Exhausted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Generated {
    pub written: usize,
    pub state: GenerationState,
}

#[derive(Debug, thiserror::Error)]
pub enum TgaError {
    #[error("training failed: {0}")]
    Training(String),
    #[error("feedback update failed: {0}")]
    Feedback(String),
    #[error("generation failed: {0}")]
    Generation(String),
    #[error("invalid config: {0}")]
    Config(String),
    #[error("invalid model: {0}")]
    Model(String),
    #[error("unknown algorithm: {0}")]
    UnknownAlgorithm(String),
    #[error("unsupported operation: {0}")]
    Unsupported(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct AlgorithmId(String);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlgorithmSpec {
    pub algorithm: AlgorithmId,
    pub config: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelArtifact {
    pub algorithm: AlgorithmId,
    pub revision: u32,
    pub model_version: u32,
    pub payload: Vec<u8>,
    pub config: Value,
    pub metadata: BTreeMap<String, String>,
}

impl AlgorithmId {
    pub fn new(value: impl AsRef<str>) -> Result<Self, TgaError> {
        let value = value.as_ref().trim().to_ascii_lowercase();
        let valid = value.split('/').all(|part| {
            !part.is_empty()
                && part.as_bytes()[0].is_ascii_alphanumeric()
                && part.as_bytes()[part.len() - 1].is_ascii_alphanumeric()
        }) && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'/');
        if !valid {
            return Err(TgaError::Config(format!("invalid algorithm id '{value}'")));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for AlgorithmId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

impl std::fmt::Display for AlgorithmId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Observation {
    pub address: Address,
    pub active: bool,
}

pub type Ipv6Prefix = ipnet::Ipv6Net;

/// Scan outcome or aliased prefix reported by a probe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Feedback {
    Skipped(Ipv6Addr),
    BatchComplete,
    Active(Ipv6Addr),
    Inactive(Ipv6Addr),
    Aliased(Ipv6Prefix),
}

impl Feedback {
    pub fn is_address_observation(&self) -> bool {
        matches!(self, Feedback::Active(_) | Feedback::Inactive(_))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct GenerationStats {
    pub attempts: u64,
    pub written: u64,
    pub unique: Option<u64>,
    pub duplicates: u64,
    pub excluded: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationOptions {
    pub max_attempts: u64,
    pub count: usize,
    pub unique: bool,
    pub exclude: Vec<Ipv6Addr>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "(u8, u8)", into = "(u8, u8)")]
pub struct BitRange {
    start: u8,
    end: u8,
}

impl TryFrom<(u8, u8)> for BitRange {
    type Error = TgaError;
    fn try_from(value: (u8, u8)) -> Result<Self, Self::Error> {
        Self::new(value.0, value.1)
    }
}
impl From<BitRange> for (u8, u8) {
    fn from(value: BitRange) -> Self {
        (value.start, value.end)
    }
}
impl BitRange {
    pub fn start(self) -> u8 {
        self.start
    }
    pub fn end(self) -> u8 {
        self.end
    }
    pub fn new(start: u8, end: u8) -> Result<Self, TgaError> {
        if start >= 128 || end > 128 || start >= end {
            return Err(TgaError::Config(format!(
                "invalid bit range {start}..{end}; expected 0 <= start < end <= 128"
            )));
        }
        Ok(Self { start, end })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GenerationInput {
    Model {
        model: ModelArtifact,
        bits: Option<BitRange>,
    },
    Random {
        bits: Option<BitRange>,
    },
    Fixed {
        address: Address,
        bits: Option<BitRange>,
    },
}

pub trait TargetModel: Send {
    fn allows_repeated_probes(&self) -> bool {
        false
    }

    fn detected_aliases(&self) -> &[Ipv6Prefix] {
        &[]
    }

    /// Set the upcoming target budget before generation starts.
    fn set_budget(&mut self, _budget: usize) {}

    /// Fill a nonempty output buffer and report its valid prefix.
    fn generate(&mut self, output: &mut [Address]) -> Result<Generated, TgaError>;

    /// Apply scan feedback to this model.
    fn apply_feedback(&mut self, feedback: &[Feedback]) -> Result<(), TgaError> {
        let _ = feedback;
        Err(TgaError::Unsupported(
            "this model does not accept scan feedback".into(),
        ))
    }
}

pub trait Algorithm: Serialize + DeserializeOwned + Send + Sync + 'static {
    const ID: &'static str;
    const DESCRIPTION: &'static str;
    const MODEL_VERSION: u32 = 1;

    type Model: TargetModel + Serialize + DeserializeOwned + 'static;

    fn train(&self, observations: &[Observation]) -> Result<Self::Model, TgaError>;

    fn encode_model(model: &Self::Model) -> Result<Vec<u8>, TgaError> {
        bincode::serialize(model)
            .map_err(|error| TgaError::Model(format!("encode {} model: {error}", Self::ID)))
    }

    fn decode_model(version: u32, bytes: &[u8]) -> Result<Self::Model, TgaError> {
        if version != Self::MODEL_VERSION {
            return Self::migrate_model(version, bytes);
        }
        bincode::deserialize(bytes)
            .map_err(|error| TgaError::Model(format!("decode {} model: {error}", Self::ID)))
    }

    fn migrate_model(version: u32, _bytes: &[u8]) -> Result<Self::Model, TgaError> {
        Err(TgaError::Model(format!(
            "unsupported {} model version {version}; expected {}",
            Self::ID,
            Self::MODEL_VERSION
        )))
    }
}
