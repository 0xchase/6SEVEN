use super::*;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::HashSet;
use transformer::subsequent_mask;

struct GenerationBatch {
    prefixes: Vec<Vec<usize>>,
    decoder_inputs: Vec<Vec<usize>>,
    rngs: Vec<StdRng>,
}

impl GenerationBatch {
    fn with_capacity(batch_size: usize) -> Self {
        Self {
            prefixes: Vec::with_capacity(batch_size),
            decoder_inputs: Vec::with_capacity(batch_size),
            rngs: Vec::with_capacity(batch_size),
        }
    }
}

impl SixVecLmModel {
    pub fn set_generation_temperature(&mut self, temperature: f32) -> Result<(), TgaError> {
        validate_generation_temperature(temperature)?;
        self.generation_temperature = temperature;
        self.generation = GenerationProgress::default();
        Ok(())
    }

    fn decode_batch_step_from_memory(
        &self,
        transformer: &Ipv6Transformer<InferBackend>,
        memory: &Tensor<InferBackend, 3>,
        decoder_rows: &[Vec<usize>],
        device: &<InferBackend as Backend>::Device,
    ) -> Result<Vec<f32>, TgaError> {
        let batch = decoder_rows.len();
        let decoder_len = decoder_rows.first().map(|row| row.len()).ok_or_else(|| {
            TgaError::Generation("6VecLM decoder batch is unexpectedly empty".into())
        })?;
        let tgt_tensor = tensor_from_token_rows_inference(decoder_rows, device);
        let mask = subsequent_mask::<InferBackend>(batch, decoder_len, device);
        let output = transformer.decode(memory.clone(), tgt_tensor, mask);
        let projected = transformer.project(output);
        let slice = projected
            .slice([
                0..batch,
                decoder_len - 1..decoder_len,
                0..self.embedding_dim,
            ])
            .reshape([batch, self.embedding_dim]);
        let flat = slice
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|err| {
                TgaError::Generation(format!("failed to decode 6VecLM batch tensor: {err:?}"))
            })?;
        if flat.len() != batch * self.embedding_dim {
            return Err(TgaError::Generation(format!(
                "6VecLM batch projection length {} did not match expected {}",
                flat.len(),
                batch * self.embedding_dim
            )));
        }

        Ok(flat)
    }

    fn generate_from_prepared_rows(
        &self,
        rt: &SixVecLmRuntime,
        batch: GenerationBatch,
        temperature: f32,
    ) -> Result<Vec<Address>, TgaError> {
        let device = &rt.device;
        let word_embeddings = rt.word_embeddings.as_slice();
        let GenerationBatch {
            prefixes,
            mut decoder_inputs,
            mut rngs,
        } = batch;
        let src_tensor = tensor_from_token_rows_inference(&prefixes, device);
        let transformer = rt.transformer.lock().map_err(|_| {
            TgaError::Generation("6VecLM inference runtime mutex was poisoned".into())
        })?;
        let memory = transformer.encode(src_tensor);

        for position in FIRST_SAMPLED_POSITION..TOTAL_LEN {
            let predictions =
                self.decode_batch_step_from_memory(&transformer, &memory, &decoder_inputs, device)?;
            for row_index in 0..decoder_inputs.len() {
                let prediction_start = row_index * self.embedding_dim;
                let prediction_end = prediction_start + self.embedding_dim;
                let next = self.sample_token_from_embeddings(
                    &mut rngs[row_index],
                    word_embeddings,
                    &predictions[prediction_start..prediction_end],
                    position,
                    temperature,
                )?;
                decoder_inputs[row_index].push(next);
            }
        }

        let mut out = Vec::with_capacity(prefixes.len());
        for (mut full, tail) in prefixes.into_iter().zip(decoder_inputs) {
            full.extend_from_slice(&tail);
            let address = self
                .tokenizer
                .detokenize_hex_stream(&full)
                .ok_or_else(|| TgaError::Generation("failed to decode 6VecLM target".into()))?;
            out.push(address);
        }
        Ok(out)
    }

    fn prepare_generation_batch(
        &self,
        source_rows: &[[u16; TOTAL_LEN]],
        first_sample: usize,
        sample_count: usize,
    ) -> Result<GenerationBatch, TgaError> {
        let source_count = source_rows.len();
        if source_count == 0 {
            return Err(TgaError::Generation(
                "6VecLM model has no representable source rows".into(),
            ));
        }

        let mut batch = GenerationBatch::with_capacity(sample_count);

        for offset in 0..sample_count {
            let sample_id = first_sample
                .checked_add(offset)
                .ok_or_else(|| TgaError::Generation("6VecLM sample counter exhausted".into()))?;
            let row = source_rows.get(sample_id).ok_or_else(|| {
                TgaError::Generation("6VecLM generation exceeded its seed set".into())
            })?;
            let mut src_tokens = Vec::with_capacity(PREFIX_LEN);
            src_tokens.extend(row[..PREFIX_LEN].iter().map(|&token| token as usize));
            let first_tail_token = row[PREFIX_LEN] as usize;

            batch.prefixes.push(src_tokens);

            let mut decoder_row = Vec::with_capacity(DECODER_LEN);
            decoder_row.push(first_tail_token);
            batch.decoder_inputs.push(decoder_row);

            batch.rngs.push(StdRng::seed_from_u64(mix_generation_seed(
                self.generation_seed,
                sample_id,
            )));
        }

        Ok(batch)
    }

    fn sample_token_from_embeddings(
        &self,
        rng: &mut StdRng,
        word_embeddings: &[f32],
        prediction: &[f32],
        position: usize,
        temperature: f32,
    ) -> Result<usize, TgaError> {
        let tokens = self.position_candidates.get(position).ok_or_else(|| {
            TgaError::Generation(format!(
                "missing candidate vocabulary for 6VecLM position {position}"
            ))
        })?;
        let mut scores = Vec::with_capacity(tokens.len());

        for &token in tokens {
            let embed = embedding_slice(word_embeddings, self.embedding_dim, token);
            scores.push(cosine(prediction, embed));
        }

        if scores.iter().any(|score| !score.is_finite()) {
            return Err(TgaError::Generation(
                "6VecLM produced a non-finite cosine score".into(),
            ));
        }
        let probabilities = temperature_softmax(&scores, temperature);
        let choice = sample_categorical(rng, &probabilities).ok_or_else(|| {
            TgaError::Generation(format!(
                "failed to sample a 6VecLM token at position {position}"
            ))
        })?;
        tokens.get(choice).copied().ok_or_else(|| {
            TgaError::Generation(format!(
                "sampled 6VecLM candidate choice {choice} was out of bounds for position {position}"
            ))
        })
    }

    pub(super) fn generate_sample_batch(
        &self,
        first_sample: usize,
        sample_count: usize,
        temperature: f32,
    ) -> Result<Vec<Address>, TgaError> {
        if sample_count == 0 {
            return Ok(Vec::new());
        }
        validate_generation_temperature(temperature)?;
        self.validate_ready_for_generation()?;
        let rt = self.get_runtime()?;
        let batch = self.prepare_generation_batch(
            rt.seed_token_rows.as_slice(),
            first_sample,
            sample_count,
        )?;

        self.generate_from_prepared_rows(rt, batch, temperature)
    }
}

fn embedding_slice(weights: &[f32], embedding_dim: usize, token: usize) -> &[f32] {
    let start = token * embedding_dim;
    let end = start + embedding_dim;
    &weights[start..end]
}

// Equations (5) and (6) simplify to softmax(cosine / temperature).
pub(super) fn temperature_softmax(scores: &[f32], temperature: f32) -> Vec<f64> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut weights: Vec<_> = scores
        .iter()
        .map(|&score| ((f64::from(score) - max) / f64::from(temperature)).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= total;
    }
    weights
}

fn sample_categorical(rng: &mut StdRng, probabilities: &[f64]) -> Option<usize> {
    if probabilities.is_empty() {
        return None;
    }

    let threshold = rng.r#gen::<f64>();
    let mut cumulative = 0.0;
    for (idx, probability) in probabilities.iter().copied().enumerate() {
        cumulative += probability;
        if threshold <= cumulative || idx + 1 == probabilities.len() {
            return Some(idx);
        }
    }

    Some(probabilities.len() - 1)
}

fn validate_generation_temperature(temperature: f32) -> Result<(), TgaError> {
    if temperature.is_finite() && temperature > 0.0 {
        Ok(())
    } else {
        Err(TgaError::Generation(
            "6VecLM generation temperature must be finite and > 0".into(),
        ))
    }
}

fn mix_generation_seed(seed: u64, sample_id: usize) -> u64 {
    let sample_id = sample_id as u64;
    seed ^ sample_id
        .rotate_left(17)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0xBF58_476D_1CE4_E5B9)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let (mut dot, mut norm_a, mut norm_b) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&a, &b) in a.iter().zip(b) {
        let (a, b) = (f64::from(a), f64::from(b));
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }
    (dot / (norm_a.sqrt().max(1e-8) * norm_b.sqrt().max(1e-8))) as f32
}

#[derive(Debug, Default, Clone)]
pub(crate) struct GenerationProgress {
    next_sample: usize,
    seen: HashSet<Address>,
}

impl crate::TargetModel for SixVecLmModel {
    fn generate(&mut self, output: &mut [Address]) -> Result<crate::Generated, TgaError> {
        sixseven_core::generation::validate_output(output)?;
        self.validate_ready_for_generation()?;
        let mut written = 0;
        while written < output.len() && self.generation.next_sample < self.seed_addresses.len() {
            let first = self.generation.next_sample;
            let count = (self.seed_addresses.len() - first)
                .min(output.len() - written)
                .min(INFERENCE_BATCH_SIZE);
            let batch = self.generate_sample_batch(first, count, self.generation_temperature)?;
            self.generation.next_sample += count;
            for address in batch {
                if self.generation.seen.insert(address) {
                    output[written] = address;
                    written += 1;
                }
            }
        }
        Ok(crate::Generated {
            written,
            state: if self.generation.next_sample == self.seed_addresses.len() {
                crate::GenerationState::Exhausted
            } else {
                crate::GenerationState::Ready
            },
        })
    }
}

fn tensor_from_token_rows_inference(
    rows: &[Vec<usize>],
    device: &<InferBackend as Backend>::Device,
) -> Tensor<InferBackend, 2, Int> {
    let batch = rows.len();
    let seq_len = rows.first().map(|row| row.len()).unwrap_or(0);
    let mut data = Vec::with_capacity(batch * seq_len);
    for row in rows {
        debug_assert_eq!(row.len(), seq_len);
        data.extend(row.iter().map(|&token| token as i64));
    }
    Tensor::from_data(TensorData::new(data, [batch, seq_len]), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_probabilities_match_reference_code() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/sixveclm/transformer.json"
        ))
        .unwrap();
        let vector = |values: &serde_json::Value| {
            values
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let prediction = vector(&fixture["sampling_prediction"]);
        let scores: Vec<_> = fixture["sampling_vectors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|values| cosine(&prediction, &vector(values)))
            .collect();
        for (index, temperature) in fixture["temperatures"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let actual = temperature_softmax(&scores, temperature.as_f64().unwrap() as f32);
            let expected = fixture["sampling"][index].as_array().unwrap();
            for (actual, expected) in actual.into_iter().zip(expected) {
                assert!((actual - expected.as_f64().unwrap()).abs() < 1e-6);
            }
        }
    }
}
