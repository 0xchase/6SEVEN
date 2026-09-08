use super::{InferBackend, TrainBackend};

mod config;
mod corpus;
mod embedding;
mod generation;
#[cfg(test)]
mod tests;
mod tokenizer;
mod train;
mod transformer;

use crate::{Address, TgaError};
use burn::{
    module::Module,
    record::{FullPrecisionSettings, Recorder},
    tensor::{Int, Tensor, TensorData, backend::Backend},
};
pub use config::SixVecLm;
use corpus::*;
use generation::GenerationProgress;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, OnceLock};
pub use tokenizer::{IPv6Tokenizer, TokenKind};
use transformer::{Ipv6Transformer, TransformerArtifacts};

const PREFIX_LEN: usize = 16;
const DECODER_LEN: usize = 16;
const TOTAL_LEN: usize = 32;
const FIRST_SAMPLED_POSITION: usize = PREFIX_LEN + 1;
const DEFAULT_GENERATION_TEMPERATURE: f32 = 0.01;
const DEFAULT_EMBEDDING_BATCH_SIZE: usize = 10_000;
const DEFAULT_TRANSFORMER_BATCH_SIZE: usize = 100;
const INFERENCE_BATCH_SIZE: usize = 64;

mod serde_arc {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::sync::Arc;

    pub fn serialize<T, S>(value: &Arc<T>, serializer: S) -> Result<S::Ok, S::Error>
    where
        T: Serialize,
        S: Serializer,
    {
        value.as_ref().serialize(serializer)
    }

    pub fn deserialize<'de, T, D>(deserializer: D) -> Result<Arc<T>, D::Error>
    where
        T: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        T::deserialize(deserializer).map(Arc::new)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SixVecLmModel {
    #[serde(skip)]
    pub(crate) generation: GenerationProgress,
    tokenizer: IPv6Tokenizer,
    embedding_dim: usize,
    vocab_size: usize,
    #[serde(with = "serde_arc")]
    word_embeddings: Arc<Vec<f32>>,
    #[serde(with = "serde_arc")]
    transformer: Arc<TransformerArtifacts>,
    #[serde(with = "serde_arc")]
    position_candidates: Arc<Vec<Vec<usize>>>,
    #[serde(with = "serde_arc")]
    seed_addresses: Arc<Vec<[u8; 16]>>,
    generation_seed: u64,
    generation_temperature: f32,
    #[serde(skip)]
    runtime: OnceLock<Result<SixVecLmRuntime, String>>,
}

impl Clone for SixVecLmModel {
    fn clone(&self) -> Self {
        Self {
            generation: self.generation.clone(),
            tokenizer: self.tokenizer.clone(),
            embedding_dim: self.embedding_dim,
            vocab_size: self.vocab_size,
            word_embeddings: self.word_embeddings.clone(),
            transformer: self.transformer.clone(),
            position_candidates: self.position_candidates.clone(),
            seed_addresses: self.seed_addresses.clone(),
            generation_seed: self.generation_seed,
            generation_temperature: self.generation_temperature,
            runtime: OnceLock::new(),
        }
    }
}

impl Default for SixVecLmModel {
    fn default() -> Self {
        Self {
            generation: Default::default(),
            tokenizer: IPv6Tokenizer::new(),
            embedding_dim: 0,
            vocab_size: 0,
            word_embeddings: Arc::new(Vec::new()),
            transformer: Arc::new(TransformerArtifacts::default()),
            position_candidates: Arc::new(Vec::new()),
            seed_addresses: Arc::new(Vec::new()),
            generation_seed: 0,
            generation_temperature: DEFAULT_GENERATION_TEMPERATURE,
            runtime: OnceLock::new(),
        }
    }
}

struct SixVecLmRuntime {
    transformer: Mutex<Ipv6Transformer<InferBackend>>,
    word_embeddings: Arc<Vec<f32>>,
    seed_token_rows: Arc<Vec<[u16; TOTAL_LEN]>>,
    device: <InferBackend as Backend>::Device,
}

impl std::fmt::Debug for SixVecLmRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SixVecLmRuntime").finish_non_exhaustive()
    }
}

impl std::fmt::Display for SixVecLmModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "6VecLM(d_model={}, layers={}, heads={}, temp={:.3}, vocab={})",
            self.embedding_dim,
            self.transformer.hyper.n_layers,
            self.transformer.hyper.n_heads,
            self.generation_temperature,
            self.vocab_size
        )
    }
}

impl SixVecLmModel {
    fn validate_ready_for_generation(&self) -> Result<(), TgaError> {
        if self.transformer.bytes.is_empty() {
            return Err(TgaError::Generation("6VecLM model is not trained".into()));
        }
        if self.seed_addresses.is_empty() {
            return Err(TgaError::Model(
                "6VecLM model has no representable seed addresses".into(),
            ));
        }
        if self.embedding_dim < 2 || self.vocab_size == 0 {
            return Err(TgaError::Model(
                "6VecLM model has invalid embedding dimensions".into(),
            ));
        }
        if self.vocab_size != self.tokenizer.vocab_size() {
            return Err(TgaError::Model(format!(
                "6VecLM vocab size {} does not match tokenizer vocab size {}",
                self.vocab_size,
                self.tokenizer.vocab_size()
            )));
        }

        let expected_weights = self
            .vocab_size
            .checked_mul(self.embedding_dim)
            .ok_or_else(|| TgaError::Model("6VecLM embedding table dimensions overflow".into()))?;
        if self.word_embeddings.len() != expected_weights {
            return Err(TgaError::Model(format!(
                "6VecLM embedding table has {} values, expected {}",
                self.word_embeddings.len(),
                expected_weights
            )));
        }

        self.tokenizer.validate().map_err(TgaError::Model)?;
        if self
            .word_embeddings
            .iter()
            .any(|weight| !weight.is_finite())
        {
            return Err(TgaError::Model(
                "6VecLM embedding table contains non-finite weights".into(),
            ));
        }
        let hyper = &self.transformer.hyper;
        if hyper.d_model != self.embedding_dim
            || hyper.prefix_len != PREFIX_LEN
            || hyper.decoder_len != DECODER_LEN
            || hyper.n_layers == 0
            || hyper.d_ff == 0
            || !hyper.dropout.is_finite()
            || !(0.0..1.0).contains(&hyper.dropout)
            || hyper.n_heads == 0
            || !hyper.d_model.is_multiple_of(hyper.n_heads)
        {
            return Err(TgaError::Model(
                "6VecLM transformer metadata does not match the expected 6VecLM shape".into(),
            ));
        }

        if self.position_candidates.len() != TOTAL_LEN {
            return Err(TgaError::Model(format!(
                "6VecLM has candidate vocabularies for {} positions, expected {TOTAL_LEN}",
                self.position_candidates.len()
            )));
        }
        for position in FIRST_SAMPLED_POSITION..TOTAL_LEN {
            let candidates = &self.position_candidates[position];
            if candidates.is_empty() {
                return Err(TgaError::Model(format!(
                    "6VecLM has no generation candidates for position {position}"
                )));
            }
            if candidates.iter().any(|&token| token >= self.vocab_size) {
                return Err(TgaError::Model(format!(
                    "6VecLM candidate vocabulary for position {position} contains an out-of-range token"
                )));
            }
            if candidates.iter().any(|&token| {
                !matches!(
                    self.tokenizer.token_kind(token),
                    TokenKind::Hex {
                        position: token_position,
                        ..
                    } if token_position == position
                )
            }) {
                return Err(TgaError::Model(format!(
                    "6VecLM candidate vocabulary for position {position} contains a token from another position"
                )));
            }
        }

        Ok(())
    }

    fn get_runtime(&self) -> Result<&SixVecLmRuntime, TgaError> {
        let runtime = self.runtime.get_or_init(|| {
            let _rng = super::rng_guard();
            let device = super::infer_device();
            let embedding_tensor = Tensor::<InferBackend, 2>::from_data(
                TensorData::new(
                    self.word_embeddings.as_ref().clone(),
                    [self.vocab_size, self.embedding_dim],
                ),
                &device,
            );

            let mut transformer = Ipv6Transformer::<InferBackend>::new(
                &self.transformer.hyper,
                self.vocab_size,
                &embedding_tensor,
                &device,
            );
            let recorder =
                burn::record::BinBytesRecorder::<FullPrecisionSettings, Vec<u8>>::default();
            let record = recorder
                .load(self.transformer.bytes.clone(), &device)
                .map_err(|err| format!("failed to load 6VecLM transformer weights: {err}"))?;
            transformer = transformer.load_record(record);

            Ok(SixVecLmRuntime {
                transformer: Mutex::new(transformer),
                word_embeddings: self.word_embeddings.clone(),
                seed_token_rows: Arc::new(
                    self.encode_seed_token_rows()
                        .map_err(|err| err.to_string())?,
                ),
                device,
            })
        });

        runtime.as_ref().map_err(|err| TgaError::Model(err.clone()))
    }

    fn encode_seed_token_rows(&self) -> Result<Vec<[u16; TOTAL_LEN]>, TgaError> {
        let mut rows = Vec::with_capacity(self.seed_addresses.len());
        for (row_index, seed_addr) in self.seed_addresses.iter().enumerate() {
            let tokens = self.tokenizer.encode_nybbles(seed_addr);
            if tokens.len() != TOTAL_LEN {
                return Err(TgaError::Model(format!(
                    "seed row {row_index} did not tokenize to {TOTAL_LEN} nybbles"
                )));
            }

            let mut row = [0u16; TOTAL_LEN];
            for (dst, token) in row.iter_mut().zip(tokens) {
                *dst = u16::try_from(token).map_err(|_| {
                    TgaError::Model(format!(
                        "token id {token} at seed row {row_index} exceeds cached generation storage"
                    ))
                })?;
            }
            rows.push(row);
        }
        Ok(rows)
    }
}
