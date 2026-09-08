mod classify;
mod config;
mod feedback;
mod generate;
mod model;
mod optimizer;
pub mod preprocess;
mod split;
mod train;

use burn::{
    module::Module,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    tensor::backend::Backend,
};
use serde::{Deserialize, Serialize};

use self::{
    config::{HIDDEN_DIM, LATENT_DIM},
    feedback::Allocation,
    generate::SixGcvaeStream,
    model::GcvaeDecoder,
    train::train_decoder,
};
use super::InferBackend;
use crate::{
    Address, Algorithm, Feedback, Generated, GenerationState, Observation, TargetModel, TgaError,
};

pub use config::{SixGcvae, SixGcvaeClassification};

type InferDevice = <InferBackend as Backend>::Device;

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct SixGcvaeModel {
    #[serde(skip)]
    generation: crate::cursor::GenerationCursor<Vec<SixGcvaeStream>>,
    components: Vec<Component>,
    generation_seed: u64,
    allocation: Option<Allocation>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Component {
    label: String,
    decoder: StoredDecoder,
    emitted: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct StoredDecoder {
    record_bytes: Vec<u8>,
    final_loss: f32,
}

impl std::fmt::Display for SixGcvaeModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "6GCVAE(latent={LATENT_DIM}, hidden={HIDDEN_DIM}")?;
        for component in &self.components {
            write!(
                f,
                ", {} loss={:.6}",
                component.label, component.decoder.final_loss
            )?;
        }
        if self.components.is_empty() {
            write!(f, ", untrained")?;
        }
        write!(f, ")")
    }
}

impl StoredDecoder {
    fn load(&self, device: &InferDevice) -> Result<GcvaeDecoder<InferBackend>, TgaError> {
        let _rng = super::rng_guard();
        let recorder = BinBytesRecorder::<FullPrecisionSettings, Vec<u8>>::default();
        let record = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            recorder.load(self.record_bytes.clone(), device)
        }))
        .map_err(|_| TgaError::Generation("failed to decode decoder weights".into()))?
        .map_err(|error| {
            TgaError::Generation(format!("failed to load decoder weights: {error:?}"))
        })?;
        let decoder = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            GcvaeDecoder::<InferBackend>::new(device).load_record(record)
        }))
        .map_err(|_| TgaError::Generation("failed to reconstruct decoder weights".into()))?;
        decoder.validate().map_err(TgaError::Generation)?;
        Ok(decoder)
    }
}

impl SixGcvaeModel {
    fn build_streams(&self) -> Result<Vec<SixGcvaeStream>, TgaError> {
        self.validate()?;
        self.components
            .iter()
            .enumerate()
            .map(|(index, component)| {
                let device = super::infer_device();
                let decoder = component.decoder.load(&device)?;
                let seed = self.generation_seed.wrapping_add(index as u64);
                Ok(SixGcvaeStream::new(decoder, device, seed).resume(
                    if self.allocation.is_some() {
                        component.emitted
                    } else {
                        0
                    },
                ))
            })
            .collect()
    }

    fn validate(&self) -> Result<(), TgaError> {
        if self.components.is_empty() {
            return Err(TgaError::Model(
                "6GCVAE model has no decoder weights".into(),
            ));
        }
        if let Some(allocation) = &self.allocation {
            allocation.validate(self.components.len())?;
        } else if self.components.len() != 1 {
            return Err(TgaError::Model(
                "6GCVAE ensemble has no allocation state".into(),
            ));
        }
        if self
            .components
            .iter()
            .any(|component| !component.decoder.final_loss.is_finite())
        {
            return Err(TgaError::Model(
                "6GCVAE training loss must be finite".into(),
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    fn load_decoder(&self, device: &InferDevice) -> Result<GcvaeDecoder<InferBackend>, TgaError> {
        self.components
            .first()
            .ok_or_else(|| TgaError::Generation("6GCVAE model has no decoder weights".into()))?
            .decoder
            .load(device)
    }

    #[cfg(test)]
    fn stream(&self) -> Result<crate::AddressStream, TgaError> {
        Ok(Box::new(self.build_streams()?.remove(0).with_batch_size(4)))
    }
}

impl Algorithm for SixGcvae {
    const ID: &'static str = "sixgcvae";
    const DESCRIPTION: &'static str =
        "6GCVAE gated convolutional variational autoencoder for IPv6 address generation";
    const MODEL_VERSION: u32 = 2;

    type Model = SixGcvaeModel;

    fn train(&self, observations: &[Observation]) -> Result<Self::Model, TgaError> {
        self.validate().map_err(TgaError::Config)?;
        let seeds: Vec<Address> = observations
            .iter()
            .filter(|obs| obs.active)
            .map(|obs| obs.address)
            .collect();
        if seeds.len() < 2 {
            return Err(TgaError::Training(
                "6GCVAE needs at least two active seeds".into(),
            ));
        }
        let groups = classify::partition(&seeds, self.classification, self.clusters)?;
        let mut components = Vec::with_capacity(groups.len());
        for group in groups {
            let trained = train_decoder(self, &group.seeds, super::train_device())
                .map_err(|error| TgaError::Training(format!("{}: {error}", group.label)))?;
            components.push(Component {
                label: group.label,
                emitted: 0,
                decoder: StoredDecoder {
                    record_bytes: trained.decoder_bytes,
                    final_loss: trained.final_loss,
                },
            });
        }
        let allocation = (self.classification != SixGcvaeClassification::None)
            .then(|| Allocation::new(seeds, components.len(), self.feedback_samples))
            .transpose()?;
        Ok(SixGcvaeModel {
            generation: Default::default(),
            components,
            generation_seed: self.generation_seed,
            allocation,
        })
    }

    fn decode_model(version: u32, bytes: &[u8]) -> Result<Self::Model, TgaError> {
        if version != Self::MODEL_VERSION {
            return Self::migrate_model(version, bytes);
        }
        let model: SixGcvaeModel = bincode::deserialize(bytes)
            .map_err(|error| TgaError::Model(format!("decode 6GCVAE model: {error}")))?;
        model.validate()?;
        Ok(model)
    }

    fn migrate_model(version: u32, bytes: &[u8]) -> Result<Self::Model, TgaError> {
        if version != 1 {
            return Err(TgaError::Model(format!(
                "unsupported 6GCVAE model version {version}"
            )));
        }
        #[derive(Deserialize)]
        struct LegacyModel {
            decoder: Option<StoredDecoder>,
            generation_seed: u64,
        }
        let legacy: LegacyModel = bincode::deserialize(bytes)
            .map_err(|error| TgaError::Model(format!("decode legacy 6GCVAE model: {error}")))?;
        let model = SixGcvaeModel {
            generation: Default::default(),
            components: legacy
                .decoder
                .into_iter()
                .map(|decoder| Component {
                    label: "all".into(),
                    emitted: 0,
                    decoder,
                })
                .collect(),
            generation_seed: legacy.generation_seed,
            allocation: None,
        };
        model.validate()?;
        Ok(model)
    }
}

impl TargetModel for SixGcvaeModel {
    fn generate(&mut self, output: &mut [Address]) -> Result<Generated, TgaError> {
        sixseven_core::generation::validate_output(output)?;
        if self
            .allocation
            .as_ref()
            .is_some_and(|allocation| allocation.remaining() == 0)
        {
            return Ok(Generated {
                written: 0,
                state: GenerationState::AwaitingFeedback,
            });
        }
        if self.generation.0.is_none() {
            self.generation.0 = Some(self.build_streams()?);
        }
        let streams = self.generation.0.as_mut().expect("initialized cursor");
        if self.allocation.is_none() {
            return crate::cursor::fill(&mut streams[0], output, GenerationState::Exhausted);
        }
        let allocation = self.allocation.as_mut().expect("classified allocation");
        let written = output.len().min(allocation.remaining());
        for slot in output[..written].iter_mut() {
            let source = allocation.select();
            let address = streams[source]
                .next()
                .ok_or_else(|| TgaError::Generation("6GCVAE decoder stream terminated".into()))??;
            self.components[source].emitted = self.components[source]
                .emitted
                .checked_add(1)
                .ok_or_else(|| TgaError::Generation("6GCVAE latent sequence exhausted".into()))?;
            allocation.record(address, source);
            *slot = address;
        }
        Ok(Generated {
            written,
            state: if allocation.remaining() == 0 {
                GenerationState::AwaitingFeedback
            } else {
                GenerationState::Ready
            },
        })
    }

    fn apply_feedback(&mut self, feedback: &[Feedback]) -> Result<(), TgaError> {
        let allocation = self.allocation.as_mut().ok_or_else(|| {
            TgaError::Unsupported("unclassified 6GCVAE has no feedback allocation".into())
        })?;
        if self
            .components
            .iter()
            .all(|component| component.emitted == 0)
            && feedback.iter().any(Feedback::is_address_observation)
        {
            return Err(TgaError::Feedback(
                "6GCVAE feedback requires a checkpoint with generated candidates".into(),
            ));
        }
        allocation.apply(feedback);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
