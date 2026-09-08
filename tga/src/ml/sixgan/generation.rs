use super::{
    SixGanClassification,
    models::{Generator, NYBBLE_COUNT, id_to_nybble},
};
use crate::{Address, TgaError, ml::InferBackend};
use burn::{
    module::Module,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    tensor::backend::Backend,
};
use rand::{RngCore, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, OnceLock};

const GENERATION_BATCH_SIZE: usize = 64;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct GeneratorArtifact {
    pub bytes: Vec<u8>,
    pub label: String,
    pub class_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SixGanModel {
    pub(super) budget: Option<super::budget::Budget>,
    #[serde(skip)]
    pub(crate) generation: GenerationProgress,
    pub(super) generators: Vec<GeneratorArtifact>,
    pub(super) emb_dim: usize,
    pub(super) hidden_dim: usize,
    #[serde(default = "default_generation_batch_size")]
    pub(super) generation_batch_size: usize,
    pub(super) generation_temperature: f32,
    pub(super) sampling_seed: u64,
    pub(super) classification: SixGanClassification,
    #[serde(skip, default = "new_runtime_cell")]
    pub(super) runtime: Arc<OnceLock<Result<SixGanRuntime, String>>>,
}

impl Default for SixGanModel {
    fn default() -> Self {
        Self {
            budget: None,
            generation: Default::default(),
            generators: Vec::new(),
            emb_dim: 0,
            hidden_dim: 0,
            generation_batch_size: GENERATION_BATCH_SIZE,
            generation_temperature: 1.0,
            sampling_seed: 0,
            classification: SixGanClassification::RfcBased,
            runtime: new_runtime_cell(),
        }
    }
}

pub(super) struct SixGanRuntime {
    generators: Mutex<Vec<Generator<InferBackend>>>,
    device: <InferBackend as Backend>::Device,
}

fn new_runtime_cell() -> Arc<OnceLock<Result<SixGanRuntime, String>>> {
    Arc::new(OnceLock::new())
}

fn default_generation_batch_size() -> usize {
    GENERATION_BATCH_SIZE
}

impl std::fmt::Debug for SixGanRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SixGanRuntime").finish_non_exhaustive()
    }
}

impl std::fmt::Display for SixGanModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "6GAN(k={}, classification={:?})",
            self.generators.len(),
            self.classification
        )
    }
}

impl SixGanModel {
    fn validate_for_sampling(&self) -> Result<usize, TgaError> {
        let generator_count = self.generators.len();
        if generator_count == 0 {
            return Err(TgaError::Model(
                "6GAN model contains no trained generators".to_string(),
            ));
        }
        if self.emb_dim == 0 || self.hidden_dim == 0 {
            return Err(TgaError::Model(
                "6GAN model has invalid generator dimensions".to_string(),
            ));
        }
        if self.generation_temperature <= 0.0 || !self.generation_temperature.is_finite() {
            return Err(TgaError::Model(
                "6GAN model has invalid generation temperature".to_string(),
            ));
        }
        if self.generation_batch_size == 0
            || self
                .generators
                .iter()
                .any(|generator| generator.class_size == 0 || generator.label.is_empty())
        {
            return Err(TgaError::Model(
                "6GAN model has invalid batch or class sizes".into(),
            ));
        }
        Ok(generator_count)
    }

    fn get_runtime(&self) -> Result<&SixGanRuntime, TgaError> {
        let runtime = self.runtime.get_or_init(|| {
            let _rng = crate::ml::rng_guard();
            let device = crate::ml::infer_device();
            let recorder = BinBytesRecorder::<FullPrecisionSettings, Vec<u8>>::default();

            let generators = self
                .generators
                .iter()
                .enumerate()
                .map(|(index, artifact)| {
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let record =
                            recorder
                                .load(artifact.bytes.clone(), &device)
                                .map_err(|err| {
                                    format!("failed to load 6GAN generator {index}: {err:?}")
                                })?;
                        let generator =
                            Generator::<InferBackend>::new(self.emb_dim, self.hidden_dim, &device)
                                .load_record(record);
                        if !generator.has_valid_dimensions() {
                            return Err(format!(
                                "6GAN generator {index} has incompatible tensor dimensions"
                            ));
                        }
                        Ok(generator)
                    }))
                    .map_err(|_| format!("failed to reconstruct 6GAN generator {index}"))?
                })
                .collect::<Result<Vec<_>, String>>()?;

            Ok(SixGanRuntime {
                generators: Mutex::new(generators),
                device,
            })
        });

        runtime.as_ref().map_err(|err| TgaError::Model(err.clone()))
    }

    fn generate_budgeted(&mut self, output: &mut [Address]) -> Result<crate::Generated, TgaError> {
        let classes = self.validate_for_sampling()?;
        self.budget.as_ref().unwrap().validate(classes)?;
        let mut written = 0;
        while written < output.len() {
            let budget = self
                .budget
                .as_mut()
                .expect("budgeted generation requires a budget");
            if let Some(address) = budget.emit() {
                output[written] = address;
                written += 1;
                continue;
            }
            let Some((index, count)) = budget.next_batch(self.generation_batch_size) else {
                break;
            };
            let mut rng = StdRng::seed_from_u64(self.sampling_seed.wrapping_add(budget.draw));
            let batch = self.sample_generator_batch(index, count, rng.next_u64())?;
            let budget = self.budget.as_mut().unwrap();
            budget.draw = budget.draw.wrapping_add(1);
            budget.generator = index;
            budget.buffer = batch.into();
        }
        Ok(crate::Generated {
            written,
            state: self.budget.as_ref().unwrap().state(),
        })
    }

    fn sample_generator_batch(
        &self,
        generator_index: usize,
        batch_size: usize,
        seed: u64,
    ) -> Result<Vec<Address>, TgaError> {
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let runtime = self.get_runtime()?;
        let generators = runtime
            .generators
            .lock()
            .map_err(|_| TgaError::Generation("6GAN generator lock poisoned".to_string()))?;
        let Some(generator) = generators.get(generator_index) else {
            return Err(TgaError::Generation(format!(
                "6GAN generator index {generator_index} is out of range"
            )));
        };

        let _rng = crate::ml::rng_guard();
        InferBackend::seed(&runtime.device, seed);
        let generated = generator
            .generate_sample(batch_size, &runtime.device, self.generation_temperature)
            .map_err(TgaError::Generation)?;
        let sequences = generated;

        if sequences.len() != batch_size {
            return Err(TgaError::Generation(format!(
                "6GAN generated {} addresses for {} requested samples",
                sequences.len(),
                batch_size
            )));
        }

        sequences
            .iter()
            .map(|sequence| decode_candidate_sequence(sequence))
            .collect()
    }
}

#[derive(Debug, Default, Clone)]
pub(crate) struct GenerationProgress {
    next_generator: usize,
    buffer: std::vec::IntoIter<Address>,
    rng: Option<StdRng>,
}

impl crate::TargetModel for SixGanModel {
    fn generate(&mut self, output: &mut [crate::Address]) -> Result<crate::Generated, TgaError> {
        sixseven_core::generation::validate_output(output)?;
        if self.budget.is_some() {
            return self.generate_budgeted(output);
        }
        let mut written = 0;
        while written < output.len() {
            if self.generation.buffer.len() == 0 {
                let count = self.validate_for_sampling()?;
                let rng = self
                    .generation
                    .rng
                    .get_or_insert_with(|| StdRng::seed_from_u64(self.sampling_seed));
                let seed = rng.next_u64();
                let index = self.generation.next_generator % count;
                let batch = self.sample_generator_batch(index, self.generation_batch_size, seed)?;
                self.generation.next_generator = (index + 1) % count;
                self.generation.buffer = batch.into_iter();
            }
            let count = self.generation.buffer.len().min(output.len() - written);
            output[written..written + count]
                .copy_from_slice(&self.generation.buffer.as_slice()[..count]);
            self.generation.buffer.by_ref().take(count).for_each(drop);
            written += count;
        }
        Ok(crate::Generated {
            written,
            state: crate::GenerationState::Ready,
        })
    }
    fn apply_feedback(&mut self, feedback: &[crate::Feedback]) -> Result<(), TgaError> {
        let budget = self.budget.as_mut().ok_or_else(|| {
            TgaError::Unsupported("6GAN scan feedback requires feedback_budget".into())
        })?;
        budget.validate(self.generators.len())?;
        budget.feedback(feedback)
    }
}

fn decode_candidate_sequence(sequence: &[usize]) -> Result<Address, TgaError> {
    if sequence.len() != NYBBLE_COUNT {
        return Err(TgaError::Generation(
            "6GAN candidate must contain 32 nybbles".into(),
        ));
    }
    let mut address = [0; 16];
    for (byte, tokens) in address.iter_mut().zip(sequence.chunks_exact(2)) {
        let high = id_to_nybble(tokens[0]);
        let low = id_to_nybble(tokens[1]);
        match (high, low) {
            (Some(high), Some(low)) => *byte = high << 4 | low,
            _ => {
                return Err(TgaError::Generation(
                    "6GAN candidate contains an invalid nybble".into(),
                ));
            }
        }
    }
    Ok(address)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoding_rejects_malformed_sequences() {
        assert!(decode_candidate_sequence(&[0; 31]).is_err());
        assert!(decode_candidate_sequence(&[0; 33]).is_err());
        assert!(decode_candidate_sequence(&[16; 32]).is_err());
        let tokens = (0..32).map(|i| i % 16).collect::<Vec<_>>();
        assert_eq!(
            decode_candidate_sequence(&tokens).unwrap(),
            [
                1, 35, 69, 103, 137, 171, 205, 239, 1, 35, 69, 103, 137, 171, 205, 239
            ]
        );
    }
}

#[derive(Deserialize)]
struct LegacyModel {
    generators: Vec<GeneratorArtifact>,
    emb_dim: usize,
    hidden_dim: usize,
    generation_batch_size: usize,
    generation_temperature: f32,
    sampling_seed: u64,
    classification: SixGanClassification,
}

pub(super) fn migrate_v2(bytes: &[u8]) -> Result<SixGanModel, TgaError> {
    let old: LegacyModel = bincode::deserialize(bytes)
        .map_err(|error| TgaError::Model(format!("decode 6GAN v2 model: {error}")))?;
    let model = SixGanModel {
        generators: old.generators,
        emb_dim: old.emb_dim,
        hidden_dim: old.hidden_dim,
        generation_batch_size: old.generation_batch_size,
        generation_temperature: old.generation_temperature,
        sampling_seed: old.sampling_seed,
        classification: old.classification,
        ..Default::default()
    };
    model.validate_for_sampling()?;
    Ok(model)
}
