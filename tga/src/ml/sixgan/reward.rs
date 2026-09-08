use super::{
    SixGan,
    models::{Discriminator, GO_ID, Generator, MAX_SEQ_LEN, PolicySample},
    train::token_slices_to_tensor,
};
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use rand::{RngCore, rngs::StdRng};

pub(super) struct RewardEvaluator<'a, B: Backend> {
    pub discriminator: &'a Discriminator<B>,
    pub config: &'a SixGan,
    pub aliases: &'a AliasDetector,
}

impl<B: Backend> RewardEvaluator<'_, B> {
    pub(super) fn compute(
        &self,
        generator: &Generator<B>,
        sample: &PolicySample<B>,
        class_index: usize,
        device: &B::Device,
        rng: &mut StdRng,
    ) -> Result<Vec<Vec<f32>>, String> {
        let samples = &sample.tokens;
        let config = self.config;
        let batch = samples.len();
        let mut rewards = vec![vec![0.0; MAX_SEQ_LEN]; batch];
        // Keep rollout memory proportional to batch size rather than the number of searches.
        for step in 1..MAX_SEQ_LEN {
            for _ in 0..config.rollout_num {
                let mut prefix = Vec::with_capacity(batch * step);
                let mut next = Vec::with_capacity(batch);
                for sample in samples {
                    prefix.push(GO_ID as i64);
                    prefix.extend(sample[..step - 1].iter().map(|&token| token as i64));
                    next.push(sample[step - 1]);
                }
                let prefix = Tensor::from_data(TensorData::new(prefix, [batch, step]), device);
                B::seed(device, rng.next_u64());
                let continuation = generator.rollout_continuation(
                    prefix,
                    &next,
                    sample.initial_state.clone(),
                    MAX_SEQ_LEN - step,
                    config.temperature,
                );
                let fixed = samples
                    .iter()
                    .flat_map(|row| row[..step].iter().map(|&token| token as i64))
                    .collect::<Vec<_>>();
                let fixed = Tensor::from_data(TensorData::new(fixed, [batch, step]), device);
                let completed = Tensor::cat(vec![fixed, continuation], 1);
                let penalties = self.evaluate(completed, class_index, step)?;
                for (row, penalty) in rewards.iter_mut().zip(penalties) {
                    row[step - 1] += penalty / config.rollout_num as f32;
                }
            }
        }
        let full = token_slices_to_tensor(
            &samples.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            device,
        );
        let penalties = self.evaluate(full, class_index, MAX_SEQ_LEN)?;
        for (row, penalty) in rewards.iter_mut().zip(penalties) {
            row[MAX_SEQ_LEN - 1] = penalty;
        }
        Ok(rewards)
    }

    fn evaluate(
        &self,
        samples: Tensor<B, 2, Int>,
        class_index: usize,
        step: usize,
    ) -> Result<Vec<f32>, String> {
        let discriminator = self.discriminator;
        let aliases = self.aliases;
        let probabilities = discriminator.forward_probs(samples.clone(), false);
        let [batch, classes] = probabilities.dims();
        assert!(class_index + 1 < classes);
        let mut penalties = probabilities
            .slice([0..batch, class_index..class_index + 1])
            .reshape([batch])
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| {
                format!("failed to read 6GAN discriminator probabilities: {error:?}")
            })?;
        for penalty in &mut penalties {
            *penalty = 1.0 - *penalty;
        }
        if !aliases.prefixes.is_empty() {
            let tokens = samples
                .into_data()
                .convert::<i64>()
                .to_vec::<i64>()
                .map_err(|error| format!("failed to read 6GAN rollout tokens: {error:?}"))?;
            for (penalty, row) in penalties.iter_mut().zip(tokens.chunks_exact(MAX_SEQ_LEN)) {
                *penalty += aliases.penalty(row, step);
            }
        }
        Ok(penalties)
    }
}

pub(super) struct AliasDetector {
    prefixes: std::collections::BTreeMap<u8, std::collections::HashSet<u128>>,
    strength: f32,
}

impl AliasDetector {
    pub(super) fn new(config: &SixGan) -> Self {
        let mut prefixes = std::collections::BTreeMap::<u8, std::collections::HashSet<u128>>::new();
        for prefix in &config.aliased_prefixes {
            prefixes
                .entry(prefix.prefix_len())
                .or_default()
                .insert(u128::from(prefix.network()));
        }
        Self {
            prefixes,
            strength: config.alias_alpha * config.alias_strength,
        }
    }

    fn penalty(&self, tokens: &[i64], step: usize) -> f32 {
        let address = tokens
            .iter()
            .fold(0u128, |address, &token| address << 4 | token as u128);
        for (&bits, prefixes) in self.prefixes.iter().rev() {
            let mask = u128::MAX << (128 - bits);
            if prefixes.contains(&(address & mask)) {
                let length = bits as usize / 4;
                return if step <= length {
                    self.strength * step as f32 / length as f32
                } else {
                    0.0
                };
            }
        }
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discriminator_penalty_uses_each_generators_own_class() {
        let _guard = crate::ml::rng_guard();
        type B = crate::ml::CpuBackend;
        let device = Default::default();
        B::seed(&device, 42);
        let discriminator = Discriminator::<B>::new(3, 2, &[1], &[2], &device);
        let config = SixGan::default();
        let aliases = AliasDetector::new(&config);
        let evaluator = RewardEvaluator {
            discriminator: &discriminator,
            config: &config,
            aliases: &aliases,
        };
        let samples = Tensor::<B, 2, Int>::zeros([1, MAX_SEQ_LEN], &device);
        let probabilities = discriminator
            .forward_probs(samples.clone(), false)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (class, probability) in probabilities.iter().enumerate().take(2) {
            let penalty = evaluator
                .evaluate(samples.clone(), class, MAX_SEQ_LEN)
                .unwrap()[0];
            assert!((penalty - (1.0 - probability)).abs() < 1e-6);
        }
    }

    #[test]
    fn alias_penalty_increases_within_prefix_and_stops_at_suffix() {
        let config = SixGan {
            aliased_prefixes: vec!["2001:db8::/32".parse().unwrap()],
            ..Default::default()
        };
        let detector = AliasDetector::new(&config);
        let tokens = vec![
            2, 0, 0, 1, 0, 13, 11, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1,
        ];
        assert_eq!(detector.penalty(&tokens, 1), 1.125);
        assert_eq!(detector.penalty(&tokens, 8), 9.0);
        assert_eq!(detector.penalty(&tokens, 9), 0.0);
        assert_eq!(detector.penalty(&[0; 32], 1), 0.0);
    }
}
