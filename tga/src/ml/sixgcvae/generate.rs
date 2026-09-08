//! Batched draws from the standard normal prior, decoded by per-nybble argmax.

use burn::tensor::{Tensor, TensorData};
use rand::{Rng, SeedableRng, distributions::Open01};
use rand_chacha::ChaCha12Rng;

use super::{
    InferDevice,
    config::{LATENT_DIM, N},
    model::GcvaeDecoder,
    preprocess::nybble_sequence_to_bytes_i64,
};
use crate::{Address, TgaError, ml::InferBackend};

pub(super) const INFERENCE_BATCH_SIZE: usize = 256;

#[derive(Clone)]
pub(super) struct SixGcvaeStream {
    decoder: GcvaeDecoder<InferBackend>,
    device: InferDevice,
    rng: ChaCha12Rng,
    buffer: Vec<Address>,
    buffer_pos: usize,
    done: bool,
    batch_size: usize,
}

impl SixGcvaeStream {
    pub(super) fn new(decoder: GcvaeDecoder<InferBackend>, device: InferDevice, seed: u64) -> Self {
        Self {
            decoder,
            device,
            rng: ChaCha12Rng::seed_from_u64(seed),
            buffer: Vec::with_capacity(INFERENCE_BATCH_SIZE),
            buffer_pos: 0,
            done: false,
            batch_size: INFERENCE_BATCH_SIZE,
        }
    }

    pub(super) fn resume(mut self, emitted: u64) -> Self {
        // Each normal pair consumes two f64 draws, or four ChaCha words.
        self.rng
            .set_word_pos(u128::from(emitted) * LATENT_DIM as u128 * 2);
        self
    }

    #[cfg(test)]
    pub(super) fn with_batch_size(mut self, batch_size: usize) -> Self {
        assert!(batch_size > 0);
        self.batch_size = batch_size;
        self
    }

    fn refill(&mut self) -> Result<(), TgaError> {
        generate_draw_batch_into(
            &self.decoder,
            &self.device,
            &mut self.rng,
            self.batch_size,
            &mut self.buffer,
        )?;
        self.buffer_pos = 0;
        Ok(())
    }
}

impl Iterator for SixGcvaeStream {
    type Item = Result<Address, TgaError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        loop {
            if self.buffer_pos < self.buffer.len() {
                let address = self.buffer[self.buffer_pos];
                self.buffer_pos += 1;
                return Some(Ok(address));
            }

            match self.refill() {
                Ok(()) => continue,
                Err(error) => {
                    self.done = true;
                    return Some(Err(error));
                }
            }
        }
    }
}

fn generate_draw_batch_into<R: Rng + ?Sized>(
    decoder: &GcvaeDecoder<InferBackend>,
    device: &InferDevice,
    rng: &mut R,
    batch_len: usize,
    out: &mut Vec<Address>,
) -> Result<(), TgaError> {
    out.clear();
    if batch_len == 0 {
        return Ok(());
    }

    let mut latents = Vec::with_capacity(batch_len * LATENT_DIM);
    fill_standard_normal_latents(rng, batch_len, &mut latents);
    decode_latent_batch(decoder, device, latents, batch_len, out)
}

pub(super) fn fill_standard_normal_latents<R: Rng + ?Sized>(
    rng: &mut R,
    batch_len: usize,
    out: &mut Vec<f32>,
) {
    out.clear();
    let target_len = batch_len * LATENT_DIM;
    out.reserve(target_len);
    while out.len() < target_len {
        let (a, b) = standard_normal_pair(rng);
        out.push(a);
        if out.len() < target_len {
            out.push(b);
        }
    }
}

pub(super) fn decode_latent_batch(
    decoder: &GcvaeDecoder<InferBackend>,
    device: &InferDevice,
    latents: Vec<f32>,
    batch_len: usize,
    out: &mut Vec<Address>,
) -> Result<(), TgaError> {
    if latents.len() != batch_len * LATENT_DIM {
        return Err(TgaError::Generation(format!(
            "unexpected latent batch size: expected {}, got {}",
            batch_len * LATENT_DIM,
            latents.len()
        )));
    }
    let z = Tensor::<InferBackend, 2>::from_data(
        TensorData::new(latents, [batch_len, LATENT_DIM]),
        device,
    );
    let all_nybbles = decoder
        .forward_logits(z)
        .argmax(2)
        .into_data()
        .convert::<i64>();
    let nybble_slice = all_nybbles
        .as_slice::<i64>()
        .map_err(|e| TgaError::Generation(format!("decode batch output: {e:?}")))?;

    if nybble_slice.len() != batch_len * N {
        return Err(TgaError::Generation(format!(
            "unexpected decoder output size: expected {}, got {}",
            batch_len * N,
            nybble_slice.len()
        )));
    }

    for nybbles in nybble_slice.chunks_exact(N) {
        out.push(nybble_sequence_to_bytes_i64(nybbles).ok_or_else(|| {
            TgaError::Generation("failed to map decoder output into IPv6 address".into())
        })?);
    }

    Ok(())
}

fn standard_normal_pair<R: Rng + ?Sized>(rng: &mut R) -> (f32, f32) {
    let u1 = uniform_open01(rng);
    let u2 = uniform_open01(rng);

    let radius = (-2.0 * u1.ln()).sqrt();
    let theta = core::f64::consts::TAU * u2;
    ((radius * theta.cos()) as f32, (radius * theta.sin()) as f32)
}

pub(super) fn uniform_open01<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    rng.sample(Open01)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;

    #[test]
    fn explicit_chacha_preserves_existing_generation_seeds() {
        let mut old = StdRng::seed_from_u64(42);
        let mut explicit = ChaCha12Rng::seed_from_u64(42);
        let (mut expected, mut actual) = (Vec::new(), Vec::new());
        fill_standard_normal_latents(&mut old, 16, &mut expected);
        fill_standard_normal_latents(&mut explicit, 16, &mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn seeking_skips_exactly_the_emitted_latent_vectors() {
        let mut uninterrupted = ChaCha12Rng::seed_from_u64(42);
        let mut all = Vec::new();
        fill_standard_normal_latents(&mut uninterrupted, 15, &mut all);
        let mut resumed = ChaCha12Rng::seed_from_u64(42);
        resumed.set_word_pos(13 * LATENT_DIM as u128 * 2);
        let mut next = Vec::new();
        fill_standard_normal_latents(&mut resumed, 2, &mut next);
        assert_eq!(next, all[13 * LATENT_DIM..]);
    }
}
