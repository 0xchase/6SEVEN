use burn::{
    module::{AutodiffModule, Module},
    optim::{GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    tensor::{
        Int, Tensor, TensorData,
        backend::{AutodiffBackend, Backend},
    },
};
use rand::{RngCore, SeedableRng, rngs::StdRng, seq::SliceRandom};

use super::{
    SixGan,
    classify::{SeedClass, classify_seeds},
    models::{Discriminator, GO_ID, Generator, LstmState, MAX_SEQ_LEN, PolicySample},
    optimizer::{RmsProp, clip_gradients_global_norm, scalar},
    reward::{AliasDetector, RewardEvaluator},
};
use crate::Address;

const DISCRIMINATOR_LR: f64 = 1e-4;
const GENERATOR_LR: f64 = 1e-2;
const GENERATOR_CLIP_NORM: f32 = 5.0;

const DIS_FILTER_SIZES: [usize; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

#[derive(Debug, Clone)]
struct SequenceLoader<'a> {
    rows: &'a [Vec<usize>],
    batch_size: usize,
    pointer: usize,
}

impl<'a> SequenceLoader<'a> {
    fn new(sequences: &'a [Vec<usize>], batch_size: usize) -> Option<Self> {
        let num_batches = sequences.len() / batch_size;
        if num_batches == 0 {
            return None;
        }

        let rows = &sequences[..num_batches * batch_size];
        Some(Self {
            rows,
            batch_size,
            pointer: 0,
        })
    }

    fn reset_pointer(&mut self) {
        self.pointer = 0;
    }

    fn next_batch(&mut self) -> &[Vec<usize>] {
        let index = self.pointer;
        self.pointer = (self.pointer + 1) % (self.rows.len() / self.batch_size);
        &self.rows[index * self.batch_size..(index + 1) * self.batch_size]
    }
}

#[derive(Debug, Clone)]
struct DiscriminatorDataset<'a> {
    positive_rows: Vec<&'a [usize]>,
    positive_labels: Vec<usize>,
    num_classes: usize,
}

#[derive(Debug, Clone, Copy)]
struct DiscriminatorExample<'a> {
    row: &'a [usize],
    label: usize,
}

#[derive(Debug, Clone)]
struct DiscriminatorBatch<'a> {
    inputs: Vec<&'a [usize]>,
    label_ids: Vec<usize>,
    weights: Vec<f32>,
}

impl<'a> DiscriminatorDataset<'a> {
    fn new(classified: &'a [SeedClass]) -> Self {
        let num_classes = classified.len() + 1;
        let positive_rows = classified
            .iter()
            .flat_map(|class| class.rows.iter().map(Vec::as_slice))
            .collect::<Vec<_>>();
        let positive_labels = classified
            .iter()
            .enumerate()
            .flat_map(|(class_index, class)| std::iter::repeat_n(class_index, class.rows.len()))
            .collect::<Vec<_>>();

        Self {
            positive_rows,
            positive_labels,
            num_classes,
        }
    }

    fn build_batches(
        &self,
        fake_samples: &'a [Vec<usize>],
        batch_size: usize,
        rng: &mut StdRng,
    ) -> Vec<DiscriminatorBatch<'a>> {
        let mut examples =
            Vec::with_capacity(self.positive_rows.len().saturating_add(fake_samples.len()));
        examples.extend(
            self.positive_rows
                .iter()
                .zip(self.positive_labels.iter().copied())
                .map(|(&row, label)| DiscriminatorExample { row, label }),
        );
        examples.extend(fake_samples.iter().map(|row| DiscriminatorExample {
            row: row.as_slice(),
            label: self.num_classes - 1,
        }));

        let weights = class_weights(&examples, self.num_classes);
        examples.shuffle(rng);
        let full = examples.len() / batch_size * batch_size;
        examples.truncate(full);

        examples
            .chunks(batch_size)
            .map(|chunk| DiscriminatorBatch {
                inputs: chunk.iter().map(|example| example.row).collect(),
                label_ids: chunk.iter().map(|example| example.label).collect(),
                weights: chunk.iter().map(|example| weights[example.label]).collect(),
            })
            .collect()
    }
}

fn class_weights(examples: &[DiscriminatorExample<'_>], classes: usize) -> Vec<f32> {
    let mut counts = vec![0usize; classes];
    for example in examples {
        counts[example.label] += 1;
    }
    counts
        .into_iter()
        .map(|count| examples.len() as f32 / count.max(1) as f32)
        .collect()
}

pub(crate) fn train_model(
    config: &SixGan,
    seeds: &[Address],
) -> Result<Vec<super::generation::GeneratorArtifact>, String> {
    let _rng = crate::ml::rng_guard();
    let classified = classify_seeds(config, seeds)?;
    let aliases = AliasDetector::new(config);
    let device = crate::ml::train_device();
    let mut rng = StdRng::seed_from_u64(config.seed);
    crate::ml::TrainBackend::seed(&device, config.seed);

    let generator_count = classified.len();
    let mut generators = (0..generator_count)
        .map(|_| {
            Generator::<crate::ml::TrainBackend>::new(config.emb_dim, config.hidden_dim, &device)
        })
        .collect::<Vec<_>>();
    let mut pretrain_opts = (0..generator_count)
        .map(|_| OptimizerAdaptor::from(RmsProp))
        .collect::<Vec<_>>();
    let mut reward_opts = (0..generator_count)
        .map(|_| OptimizerAdaptor::from(RmsProp))
        .collect::<Vec<_>>();

    let mut discriminator =
        init_discriminator::<crate::ml::TrainBackend>(config, generator_count + 1, &device);
    let mut dis_optimizer = OptimizerAdaptor::from(RmsProp);

    let mut pretrain_loaders = classified
        .iter()
        .map(|class| {
            SequenceLoader::new(&class.rows, config.batch_size).ok_or_else(|| {
                format!(
                    "classified seed group contains fewer than {} samples",
                    config.batch_size
                )
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let discriminator_dataset = DiscriminatorDataset::new(&classified);

    tracing::info!(
        target: "sixgan",
        generators = generator_count,
        classification = ?config.classification,
        steps = config.generator_pretrain_steps,
        "pretraining 6GAN generators"
    );

    for generator_index in 0..generator_count {
        let steps = config.generator_pretrain_steps;
        let loss = pretrain_generator_steps(
            &mut generators[generator_index],
            &mut pretrain_opts[generator_index],
            &mut pretrain_loaders[generator_index],
            steps,
            &device,
            &mut rng,
        )?;
        tracing::debug!(
            target: "sixgan",
            generator = generator_index,
            steps,
            loss = loss,
            "generator pretrain complete"
        );
    }

    tracing::info!(
        target: "sixgan",
        steps = config.discriminator_pretrain_steps,
        "pretraining 6GAN discriminator"
    );
    if config.discriminator_pretrain_steps > 0 {
        let fake_samples =
            generate_fake_dataset(&generators, &classified, config, &device, &mut rng)?;
        let steps = config.discriminator_pretrain_steps;
        let loss = train_discriminator_steps(
            &mut discriminator,
            &mut dis_optimizer,
            &discriminator_dataset,
            &fake_samples,
            StepBudget {
                batch_size: config.batch_size,
                steps,
            },
            &device,
            &mut rng,
        )?;
        tracing::debug!(
            target: "sixgan",
            steps,
            loss = loss,
            "discriminator pretrain complete"
        );
    }

    tracing::info!(
        target: "sixgan",
        rounds = config.adversarial_rounds,
        generator_steps = config.generator_steps,
        discriminator_steps = config.discriminator_steps,
        "starting adversarial 6GAN training"
    );

    for round in 1..=config.adversarial_rounds {
        let reward_discriminator = discriminator.valid();
        let evaluator = RewardEvaluator {
            discriminator: &reward_discriminator,
            config,
            aliases: &aliases,
        };
        for _ in 0..config.generator_steps {
            for generator_index in 0..generator_count {
                let reward_generator = generators[generator_index].valid();
                crate::ml::InnerTrainBackend::seed(&device, rng.next_u64());
                let samples = reward_generator.sample_policy(
                    config.batch_size,
                    &device,
                    config.temperature,
                )?;
                let rewards = evaluator.compute(
                    &reward_generator,
                    &samples,
                    generator_index,
                    &device,
                    &mut rng,
                )?;
                let _ = update_generator_with_rewards(
                    &mut generators[generator_index],
                    &mut reward_opts[generator_index],
                    &samples,
                    &rewards,
                    config.temperature,
                    &device,
                )?;
            }
        }

        for _ in 0..config.discriminator_steps {
            let fake_samples =
                generate_fake_dataset(&generators, &classified, config, &device, &mut rng)?;
            let _ = train_discriminator_steps(
                &mut discriminator,
                &mut dis_optimizer,
                &discriminator_dataset,
                &fake_samples,
                StepBudget {
                    batch_size: config.batch_size,
                    steps: 1,
                },
                &device,
                &mut rng,
            )?;
        }

        if round % 5 == 0 {
            tracing::debug!(target: "sixgan", round, "completed adversarial 6GAN round");
        }
    }

    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let artifacts = generators
        .into_iter()
        .zip(classified)
        .map(|(generator, class)| {
            let bytes = recorder
                .record(generator.valid().into_record(), ())
                .map_err(|err| format!("failed to serialize generator: {err:?}"))?;
            Ok(super::generation::GeneratorArtifact {
                bytes,
                label: class.label,
                class_size: class.rows.len(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    Ok(artifacts)
}

fn init_discriminator<B: Backend>(
    config: &SixGan,
    num_classes: usize,
    device: &B::Device,
) -> Discriminator<B> {
    Discriminator::new(
        num_classes,
        config.discriminator_emb_dim,
        &DIS_FILTER_SIZES,
        &[config.discriminator_filters; DIS_FILTER_SIZES.len()],
        device,
    )
}

fn generate_fake_dataset<B: AutodiffBackend>(
    generators: &[Generator<B>],
    classified: &[SeedClass],
    config: &SixGan,
    device: &B::Device,
    rng: &mut StdRng,
) -> Result<Vec<Vec<usize>>, String> {
    let mut out = Vec::new();
    for (generator, class) in generators.iter().zip(classified) {
        let generator = generator.valid();
        let fake_count = class.rows.len() / config.batch_size * config.batch_size;
        let mut remaining = fake_count;
        let sampling_batch = config.batch_size;
        while remaining > 0 {
            let batch = remaining.min(sampling_batch);
            B::InnerBackend::seed(device, rng.next_u64());
            let generated = generator.generate_sample(batch, device, config.temperature)?;
            out.extend(generated);
            remaining -= batch;
        }
    }
    B::InnerBackend::memory_cleanup(device);
    Ok(out)
}

fn pretrain_generator_steps<B: AutodiffBackend>(
    generator: &mut Generator<B>,
    optimizer: &mut impl Optimizer<Generator<B>, B>,
    loader: &mut SequenceLoader<'_>,
    steps: usize,
    device: &B::Device,
    rng: &mut StdRng,
) -> Result<f32, String> {
    if steps == 0 {
        return Ok(0.0);
    }

    let mut total_loss = 0.0;
    loader.reset_pointer();

    for _ in 0..steps {
        let batch = loader.next_batch();
        B::seed(device, rng.next_u64());
        let (input, targets) = build_generator_batch(batch, device);
        let logits = generator.forward_pretrain(input);
        let loss = selected_log_probs(logits, targets).neg().mean();
        let loss_value = scalar(loss.clone().detach());
        if !loss_value.is_finite() {
            return Err("6GAN training loss is nonfinite".into());
        }

        let grads = loss.backward();
        let grads = clip_gradients_global_norm(
            generator,
            GradientsParams::from_grads(grads, generator),
            GENERATOR_CLIP_NORM,
        )?;
        *generator = optimizer.step(GENERATOR_LR, generator.clone(), grads);
        total_loss += loss_value;
    }

    Ok(total_loss / steps as f32)
}

struct StepBudget {
    batch_size: usize,
    steps: usize,
}

fn train_discriminator_steps<'a, B: AutodiffBackend>(
    discriminator: &mut Discriminator<B>,
    optimizer: &mut impl Optimizer<Discriminator<B>, B>,
    dataset: &DiscriminatorDataset<'a>,
    fake_samples: &'a [Vec<usize>],
    budget: StepBudget,
    device: &B::Device,
    rng: &mut StdRng,
) -> Result<f32, String> {
    let StepBudget { batch_size, steps } = budget;
    if steps == 0 {
        return Ok(0.0);
    }

    let mut batches = dataset.build_batches(fake_samples, batch_size, rng);
    if batches.is_empty() {
        return Ok(0.0);
    }

    let mut final_loss = 0.0;
    let mut batch_index = 0;

    for _ in 0..steps {
        if batch_index >= batches.len() {
            batches = dataset.build_batches(fake_samples, batch_size, rng);
            if batches.is_empty() {
                return Ok(final_loss);
            }
            batch_index = 0;
        }

        let batch = &batches[batch_index];
        batch_index += 1;

        let inputs = token_slices_to_tensor::<B>(&batch.inputs, device);
        let labels = Tensor::<B, 1, Int>::from_data(
            TensorData::new(
                batch
                    .label_ids
                    .iter()
                    .copied()
                    .map(|label| label as i64)
                    .collect::<Vec<_>>(),
                [batch.label_ids.len()],
            ),
            device,
        );

        B::seed(device, rng.next_u64());
        let logits = discriminator.forward_logits(inputs, true);
        let weights = Tensor::from_data(
            TensorData::new(batch.weights.clone(), [batch.weights.len(), 1]),
            device,
        );
        let loss = discriminator_loss(logits, labels, weights);
        let loss_value = scalar(loss.clone().detach());
        if !loss_value.is_finite() {
            return Err("6GAN training loss is nonfinite".into());
        }

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, discriminator);
        *discriminator = optimizer.step(DISCRIMINATOR_LR, discriminator.clone(), grads);
        final_loss = loss_value;
    }

    Ok(final_loss)
}

fn discriminator_loss<B: Backend>(
    logits: Tensor<B, 2>,
    labels: Tensor<B, 1, Int>,
    weights: Tensor<B, 2>,
) -> Tensor<B, 1> {
    (burn::tensor::activation::log_softmax(logits, 1)
        .gather(1, labels.unsqueeze_dim(1))
        .neg()
        * weights)
        .mean()
}

fn build_generator_batch<B: Backend>(
    rows: &[Vec<usize>],
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
    let mut inputs = Vec::with_capacity(rows.len() * MAX_SEQ_LEN);
    for row in rows {
        assert_eq!(row.len(), MAX_SEQ_LEN);
        inputs.push(GO_ID as i64);
        inputs.extend(row[..MAX_SEQ_LEN - 1].iter().map(|&token| token as i64));
    }
    let targets =
        token_slices_to_tensor(&rows.iter().map(Vec::as_slice).collect::<Vec<_>>(), device);
    (
        Tensor::from_data(TensorData::new(inputs, [rows.len(), MAX_SEQ_LEN]), device),
        targets,
    )
}

pub(super) fn token_slices_to_tensor<B: Backend>(
    rows: &[&[usize]],
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let flat = rows
        .iter()
        .flat_map(|row| {
            assert_eq!(row.len(), MAX_SEQ_LEN);
            row.iter().map(|&token| token as i64)
        })
        .collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(flat, [rows.len(), MAX_SEQ_LEN]), device)
}

fn selected_log_probs<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 2> {
    let [batch, steps, _] = logits.dims();
    burn::tensor::activation::log_softmax(logits, 2)
        .gather(2, targets.unsqueeze_dim(2))
        .reshape([batch, steps])
}

fn penalty_loss<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    penalties: Tensor<B, 2>,
) -> Tensor<B, 1> {
    (selected_log_probs(logits, targets).exp() * penalties)
        .sum_dim(1)
        .mean()
}

fn update_generator_with_rewards<B: AutodiffBackend>(
    generator: &mut Generator<B>,
    optimizer: &mut impl Optimizer<Generator<B>, B>,
    sample: &PolicySample<B::InnerBackend>,
    rewards: &[Vec<f32>],
    temperature: f32,
    device: &B::Device,
) -> Result<f32, String> {
    let samples = &sample.tokens;
    let (input, targets) = build_generator_batch(samples, device);
    let initial_state = LstmState {
        h: Tensor::from_inner(sample.initial_state.h.clone()),
        c: Tensor::from_inner(sample.initial_state.c.clone()),
    };
    let penalties = Tensor::from_data(
        TensorData::new(rewards.concat(), [samples.len(), MAX_SEQ_LEN]),
        device,
    );
    // Equation 3 minimizes the selected action probability times its penalty.
    let loss = penalty_loss(
        generator.forward_policy(input, initial_state) / temperature,
        targets,
        penalties,
    );
    let value = scalar(loss.clone().detach());
    if !value.is_finite() {
        return Err("6GAN generator penalty is nonfinite".into());
    }
    let grads = GradientsParams::from_grads(loss.backward(), generator);
    let grads = clip_gradients_global_norm(generator, grads, GENERATOR_CLIP_NORM)?;
    *generator = optimizer.step(GENERATOR_LR, generator.clone(), grads);
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    type B = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

    #[test]
    fn discriminator_objective_sums_class_expectations() {
        let labels = [0usize, 1, 1, 2, 2, 2];
        let examples = labels
            .iter()
            .map(|&label| DiscriminatorExample { row: &[], label })
            .collect::<Vec<_>>();
        let weights = class_weights(&examples, 3);
        let device = Default::default();
        let probabilities = [
            [0.8f32, 0.1, 0.1],
            [0.3, 0.4, 0.3],
            [0.3, 0.4, 0.3],
            [0.4, 0.4, 0.2],
            [0.4, 0.4, 0.2],
            [0.4, 0.4, 0.2],
        ];
        let logits = Tensor::<B, 2>::from_data(
            TensorData::new(
                probabilities
                    .into_iter()
                    .flatten()
                    .map(f32::ln)
                    .collect::<Vec<_>>(),
                [6, 3],
            ),
            &device,
        );
        let targets = Tensor::from_data(
            TensorData::new(labels.map(|label| label as i64).to_vec(), [6]),
            &device,
        );
        let weights = Tensor::from_data(
            TensorData::new(labels.map(|label| weights[label]).to_vec(), [6, 1]),
            &device,
        );
        let loss = scalar(discriminator_loss(logits, targets, weights));
        let expected = -(0.8f32.ln() + 0.4f32.ln() + 0.2f32.ln());
        assert!((loss - expected).abs() < 1e-6);
    }

    #[test]
    fn generator_objective_is_invariant_to_duplicate_batch_rows() {
        let device = Default::default();
        for rows in [1, 2, 5] {
            let logits = Tensor::<B, 3>::zeros([rows, 32, 16], &device);
            let targets = Tensor::zeros([rows, 32], &device);
            let penalties = Tensor::full([rows, 32], 0.5, &device);
            assert!((scalar(penalty_loss(logits, targets, penalties)) - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn teacher_forcing_keeps_the_last_nybble() {
        let row = (0..MAX_SEQ_LEN).map(|i| i % 16).collect::<Vec<_>>();
        let (inputs, targets) =
            build_generator_batch::<B>(std::slice::from_ref(&row), &Default::default());
        let inputs = inputs.into_data().to_vec::<i64>().unwrap();
        let targets = targets.into_data().to_vec::<i64>().unwrap();
        assert_eq!(inputs[0], GO_ID as i64);
        assert_eq!(inputs[31], row[30] as i64);
        assert_eq!(targets[31], row[31] as i64);
    }

    #[test]
    fn penalty_gradient_decreases_the_penalized_action() {
        let device = Default::default();
        let logits = Tensor::<B, 3>::zeros([1, 1, 2], &device).require_grad();
        let targets = Tensor::from_data(TensorData::new(vec![0i64], [1, 1]), &device);
        let loss = penalty_loss(logits.clone(), targets, Tensor::full([1, 1], 2.0, &device));
        assert!((scalar(loss.clone().detach()) - 1.0).abs() < 1e-6);
        let grad = logits
            .grad(&loss.backward())
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!((grad[0] - 0.5).abs() < 1e-6);
        assert!((grad[1] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn loader_cycles_only_full_batches() {
        let rows = [vec![1], vec![2], vec![3]];
        let mut loader = SequenceLoader::new(&rows, 2).unwrap();
        assert_eq!(loader.next_batch(), &[vec![1], vec![2]]);
        assert_eq!(loader.next_batch(), &[vec![1], vec![2]]);
    }
}
