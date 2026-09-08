use burn::{
    module::{AutodiffModule, Module},
    optim::{GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    tensor::{Int, Tensor, TensorData, backend::Backend},
};
use rand::seq::SliceRandom;

use super::{
    config::{ADAM_LEARNING_RATE, N, SixGcvae},
    model::GcvaeNet,
    optimizer::KerasAdam,
    preprocess::{bytes_to_nybble_sequence, split_train_test_indices},
};
use crate::ml::TrainBackend;

#[derive(Debug, thiserror::Error)]
pub enum SixGcvaeTrainError {
    #[error("invalid 6GCVAE config: {0}")]
    InvalidConfig(String),
    #[error("need at least two seed addresses")]
    NotEnoughSeeds,
    #[error("train/test split produced an empty training set")]
    EmptyTrainingSplit,
    #[error("training produced a non-finite loss at epoch {0}")]
    NonFiniteLoss(usize),
    #[error("failed to record decoder weights: {0}")]
    Record(String),
}

#[derive(Debug)]
pub struct TrainedDecoder {
    pub decoder_bytes: Vec<u8>,
    pub final_loss: f32,
}

pub fn train_decoder(
    config: &SixGcvae,
    seeds: &[[u8; 16]],
    device: <TrainBackend as Backend>::Device,
) -> Result<TrainedDecoder, SixGcvaeTrainError> {
    config
        .validate()
        .map_err(SixGcvaeTrainError::InvalidConfig)?;

    if seeds.len() < 2 {
        return Err(SixGcvaeTrainError::NotEnoughSeeds);
    }

    let sequences: Vec<[u8; N]> = seeds.iter().map(bytes_to_nybble_sequence).collect();
    let (mut train_indices, _) = split_train_test_indices(sequences.len());

    if train_indices.is_empty() {
        return Err(SixGcvaeTrainError::EmptyTrainingSplit);
    }

    let _rng = super::super::rng_guard();
    let mut model = GcvaeNet::<TrainBackend>::new(&device);
    let mut optimizer = OptimizerAdaptor::from(KerasAdam);

    let batch_size = config.batch_size;
    let mut final_loss = 0.0f32;
    let mut rng = rand::thread_rng();

    for epoch in 0..config.epochs {
        train_indices.shuffle(&mut rng);
        let mut running_loss = 0.0f64;

        for batch in train_indices.chunks(batch_size) {
            let input = build_training_tensor(&sequences, batch, &device);

            let vae_loss = vae_batch_loss(&model, input);
            let loss = scalar(vae_loss.clone().detach());
            if !loss.is_finite() {
                return Err(SixGcvaeTrainError::NonFiniteLoss(epoch + 1));
            }

            let grads = vae_loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(ADAM_LEARNING_RATE, model, grads);

            running_loss += f64::from(loss) * batch.len() as f64;
        }

        final_loss = (running_loss / train_indices.len() as f64) as f32;
        if !final_loss.is_finite() {
            return Err(SixGcvaeTrainError::NonFiniteLoss(epoch + 1));
        }
        tracing::info!(target: "sixgcvae", "Epoch {epoch} avg loss {final_loss:.6}");
    }

    let inference_model = model.valid();
    inference_model
        .decoder
        .validate()
        .map_err(SixGcvaeTrainError::Record)?;
    let record = inference_model.decoder.into_record();
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let decoder_bytes = recorder
        .record(record, ())
        .map_err(|e| SixGcvaeTrainError::Record(format!("{e:?}")))?;

    Ok(TrainedDecoder {
        decoder_bytes,
        final_loss,
    })
}

fn build_training_tensor<B: Backend>(
    sequences: &[[u8; N]],
    train_indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let mut flat = Vec::with_capacity(train_indices.len() * N);
    for &idx in train_indices {
        for &nybble in &sequences[idx] {
            flat.push(i64::from(nybble));
        }
    }
    Tensor::<B, 2, Int>::from_data(TensorData::new(flat, [train_indices.len(), N]), device)
}

fn vae_batch_loss(
    model: &GcvaeNet<TrainBackend>,
    input: Tensor<TrainBackend, 2, Int>,
) -> Tensor<TrainBackend, 1> {
    let (reconstruction_logits, z_mean, z_log_var) = model.forward(input.clone());
    let xent_loss = sparse_sequence_cross_entropy(reconstruction_logits, input);
    let kl_loss = kl_divergence_loss(z_mean, z_log_var);
    (xent_loss + kl_loss).mean()
}

fn kl_divergence_loss<B: Backend>(z_mean: Tensor<B, 2>, z_log_var: Tensor<B, 2>) -> Tensor<B, 2> {
    let ones = z_log_var.clone().ones_like();
    (ones + z_log_var.clone() - z_mean.powi_scalar(2) - z_log_var.exp())
        .sum_dim(1)
        .mul_scalar(-0.5)
}

fn sparse_sequence_cross_entropy<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 2> {
    let [batch, seq_len, _vocab] = logits.dims();
    let target_indices = targets.reshape([batch, seq_len, 1]);
    // Keras 2.2.4 clips probabilities before sparse cross entropy.
    let probabilities = burn::tensor::activation::softmax(logits, 2).clamp(1e-7, 1.0 - 1e-7);
    let target_log_probs = burn::tensor::activation::log_softmax(probabilities.log(), 2)
        .gather(2, target_indices)
        .reshape([batch, seq_len]);
    target_log_probs.neg().sum_dim(1)
}

fn scalar<B: Backend>(tensor: Tensor<B, 1>) -> f32 {
    let values = tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("6GCVAE scalar loss tensor should be readable");
    debug_assert_eq!(values.len(), 1);
    values[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::CpuBackend;

    fn dense_sequence_cross_entropy<B: Backend>(
        logits: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let [batch, seq_len, vocab] = logits.dims();
        let log_probs = burn::tensor::activation::log_softmax(logits, 2);
        let target_3d = targets.reshape([batch, seq_len, 1]);
        let range = Tensor::<B, 1, Int>::arange(0..vocab as i64, device).reshape([1, 1, vocab]);
        let one_hot = target_3d
            .expand([batch, seq_len, vocab])
            .equal(range.expand([batch, seq_len, vocab]))
            .float();
        (log_probs * one_hot)
            .sum_dim(2)
            .reshape([batch, seq_len])
            .neg()
            .sum_dim(1)
    }

    #[test]
    fn reconstruction_loss_matches_keras_probability_clipping() {
        let device = Default::default();
        let logits = Tensor::<CpuBackend, 3>::from_data([[[100.0, -100.0]]], &device);
        let targets = Tensor::<CpuBackend, 2, Int>::from_data([[1]], &device);
        let loss = scalar(sparse_sequence_cross_entropy(logits, targets).mean());
        assert!((loss - 1e7f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn loss_reductions_match_reference_equations() {
        let device = Default::default();
        let logits = Tensor::<CpuBackend, 3>::zeros([2, N, 16], &device);
        let targets = Tensor::<CpuBackend, 2, Int>::zeros([2, N], &device);
        let loss = scalar(sparse_sequence_cross_entropy(logits, targets).mean());
        assert!((loss - N as f32 * 16f32.ln()).abs() < 1e-4);
        let mean = Tensor::<CpuBackend, 2>::from_data([[0.0, 0.0], [1.0, 2.0]], &device);
        let log_var = Tensor::<CpuBackend, 2>::zeros([2, 2], &device);
        let kl = kl_divergence_loss(mean, log_var)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(kl, vec![0.0, 2.5]);
    }

    #[test]
    fn train_decoder_smoke() {
        let seeds = vec![
            [0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
            [0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
        ];
        let cfg = SixGcvae {
            generation_seed: 0,
            epochs: 1,
            batch_size: 2,
            ..SixGcvae::default()
        };

        let out = train_decoder(&cfg, &seeds, crate::ml::train_device()).unwrap();
        assert!(!out.decoder_bytes.is_empty());
        assert!(out.final_loss.is_finite());
    }

    #[test]
    fn sparse_sequence_cross_entropy_matches_dense_baseline() {
        let device = Default::default();
        let logits = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(
                vec![
                    1.0, 0.5, -0.5, 2.0, 0.2, -0.1, 0.4, 0.0, -1.0, 0.3, 1.2, -0.7, 0.8, -0.2, 0.1,
                    1.5, -0.4, 0.7, 1.1, -0.8, 0.9, 0.3, -0.6, 0.2,
                ],
                [2, 3, 4],
            ),
            &device,
        );
        let targets = Tensor::<CpuBackend, 2, Int>::from_data(
            TensorData::new(vec![3, 0, 2, 1, 2, 0], [2, 3]),
            &device,
        );

        let sparse = sparse_sequence_cross_entropy(logits.clone(), targets.clone());
        let dense = dense_sequence_cross_entropy(logits, targets, &device);

        assert!((scalar(sparse.mean()) - scalar(dense.mean())).abs() < 1e-6);
    }
}
