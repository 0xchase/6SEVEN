use super::*;
use crate::ml::sixveclm::{TOTAL_LEN, TokenCorpus, TrainBackend};
use burn::{
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    tensor::{linalg, module::embedding},
};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
const COS_EPS: f32 = 1e-8;

pub fn train_transformer(
    hyper: &TransformerHyperParams,
    train_cfg: &TransformerTrainConfig,
    corpus: &TokenCorpus,
    embedding_bank: &Tensor<TrainBackend, 2>,
) -> Result<TransformerArtifacts, String> {
    let _rng = crate::ml::rng_guard();
    if corpus.is_empty() {
        return Err("Need at least one training sequence for transformer".into());
    }

    let device = crate::ml::train_device();
    TrainBackend::seed(&device, train_cfg.seed);
    let mut rng = StdRng::seed_from_u64(train_cfg.seed);
    let [vocab_size, _] = embedding_bank.dims();

    let mut model =
        Ipv6Transformer::<TrainBackend>::new(hyper, vocab_size, embedding_bank, &device);
    let mut optimizer = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.98)
        .with_epsilon(1e-9)
        .init();

    let batch_size = train_cfg.batch_size.max(1);
    let decoder_input_len = hyper.decoder_len.saturating_sub(1);
    if decoder_input_len == 0 {
        return Err("Decoder length must be at least 2".into());
    }
    let mut step_num = 0usize;

    for epoch in 0..train_cfg.epochs.max(1) {
        let mut indices: Vec<usize> = (0..corpus.len()).collect();
        if train_cfg.shuffle {
            indices.shuffle(&mut rng);
        }

        let mut running_loss = 0f32;
        let mut batches = 0usize;

        for batch_indices in indices.chunks(batch_size) {
            let batch = build_batch(batch_indices, corpus, hyper)?;
            let src_tensor =
                tensor_from_tokens(&batch.src, batch_indices.len(), hyper.prefix_len, &device);
            let decoder_in = tensor_from_tokens(
                &batch.decoder_in,
                batch_indices.len(),
                decoder_input_len,
                &device,
            );
            let decoder_target = tensor_from_tokens(
                &batch.decoder_target,
                batch_indices.len(),
                decoder_input_len,
                &device,
            );

            let tgt_mask =
                subsequent_mask::<TrainBackend>(batch_indices.len(), decoder_input_len, &device);

            let decoder_out = model.forward(src_tensor, decoder_in, tgt_mask);
            let projected = model.project(decoder_out);

            let target_vectors = embedding(embedding_bank.clone(), decoder_target.clone());
            let cos =
                linalg::cosine_similarity(projected.clone(), target_vectors, 2, Some(COS_EPS));

            let ones = cos.clone().ones_like();
            let loss = (ones - cos).mean();

            let loss_value = loss.clone().into_scalar();
            if !loss_value.is_finite() {
                return Err("6VecLM transformer training produced non-finite loss".into());
            }
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            step_num += 1;
            let learning_rate = noam_rate(
                step_num,
                hyper.d_model,
                train_cfg.noam_factor,
                train_cfg.noam_warmup,
            );
            model = optimizer.step(learning_rate, model, grads);

            running_loss += loss_value;
            batches += 1;
        }

        tracing::info!(
            target: "sixveclm",
            "Transformer epoch {epoch} avg loss {:.6}",
            running_loss / (batches.max(1) as f32)
        );
    }

    let inference_model = model.valid();
    let record = inference_model.into_record();
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let bytes = recorder
        .record(record, ())
        .map_err(|e| format!("Failed to record transformer: {e:?}"))?;

    Ok(TransformerArtifacts {
        bytes,
        hyper: hyper.clone(),
    })
}

#[derive(Default)]
pub(super) struct PreparedBatch {
    pub(super) src: Vec<usize>,
    pub(super) decoder_in: Vec<usize>,
    pub(super) decoder_target: Vec<usize>,
}

impl PreparedBatch {
    fn with_capacity(batch_size: usize, hyper: &TransformerHyperParams) -> Self {
        let decoder_input_len = hyper.decoder_len.saturating_sub(1);
        Self {
            src: Vec::with_capacity(batch_size * hyper.prefix_len),
            decoder_in: Vec::with_capacity(batch_size * decoder_input_len),
            decoder_target: Vec::with_capacity(batch_size * decoder_input_len),
        }
    }
}

fn build_batch(
    batch_indices: &[usize],
    corpus: &TokenCorpus,
    hyper: &TransformerHyperParams,
) -> Result<PreparedBatch, String> {
    let mut batch = PreparedBatch::with_capacity(batch_indices.len(), hyper);
    for &idx in batch_indices {
        let tokens = corpus
            .row(idx)
            .ok_or_else(|| format!("corpus row {idx} was out of bounds"))?;
        append_sequence(&mut batch, tokens, hyper)?;
    }
    if batch.src.is_empty() {
        return Err("Batch contained no valid sequences".into());
    }
    Ok(batch)
}

pub(super) fn append_sequence(
    batch: &mut PreparedBatch,
    tokens: &[u16; TOTAL_LEN],
    hyper: &TransformerHyperParams,
) -> Result<(), String> {
    if hyper.prefix_len + hyper.decoder_len > tokens.len() {
        return Err("Sequence length mismatch in dataset".into());
    }

    let decoder_slice = &tokens[hyper.prefix_len..hyper.prefix_len + hyper.decoder_len];
    if decoder_slice.len() < 2 {
        return Err("Decoder sequence too short".into());
    }

    batch.src.extend(
        tokens[..hyper.prefix_len]
            .iter()
            .map(|&token| token as usize),
    );
    batch.decoder_in.extend(
        decoder_slice[..decoder_slice.len() - 1]
            .iter()
            .map(|&token| token as usize),
    );
    batch
        .decoder_target
        .extend(decoder_slice[1..].iter().map(|&token| token as usize));
    Ok(())
}

fn tensor_from_tokens(
    tokens: &[usize],
    batch: usize,
    seq_len: usize,
    device: &<TrainBackend as Backend>::Device,
) -> Tensor<TrainBackend, 2, Int> {
    let data: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
    Tensor::from_data(TensorData::new(data, [batch, seq_len]), device)
}

pub(super) fn noam_rate(step: usize, d_model: usize, factor: f64, warmup: usize) -> f64 {
    let step = step.max(1) as f64;
    let warmup = warmup.max(1) as f64;
    factor * (d_model as f64).powf(-0.5) * step.powf(-0.5).min(step * warmup.powf(-1.5))
}
