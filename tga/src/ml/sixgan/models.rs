use burn::{
    module::{Initializer, Module, Param},
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{Distribution, Int, Tensor, TensorData, backend::Backend},
};

pub const GO_ID: usize = 16;
pub const VOCAB_SIZE: usize = 16;
pub const MAX_SEQ_LEN: usize = 32;
pub const NYBBLE_COUNT: usize = 32;

const GENERATOR_INIT_STD: f64 = 0.1;
const DISCRIMINATOR_EMBED_BOUND: f64 = 1.0;
const DISCRIMINATOR_DROPOUT_PROB: f64 = 0.25;
const DISCRIMINATOR_BIAS_INIT: f64 = 0.1;
const BASIC_LSTM_FORGET_BIAS: f64 = 1.0;

fn normal_0_1() -> Initializer {
    Initializer::Normal {
        mean: 0.0,
        std: GENERATOR_INIT_STD,
    }
}

fn glorot_uniform() -> Initializer {
    Initializer::XavierUniform { gain: 1.0 }
}

pub fn nybble_to_id(nybble: u8) -> usize {
    nybble as usize
}

pub fn id_to_nybble(id: usize) -> Option<u8> {
    if id < VOCAB_SIZE {
        Some(id as u8)
    } else {
        None
    }
}

#[derive(Module, Debug)]
pub struct LstmCell<B: Backend> {
    weight: Linear<B>,
    hidden_dim: usize,
}

impl<B: Backend> LstmCell<B> {
    pub fn new(input_dim: usize, hidden_dim: usize, device: &B::Device) -> Self {
        let mut weight = LinearConfig::new(input_dim + hidden_dim, 4 * hidden_dim)
            .with_initializer(glorot_uniform())
            .init(device);
        // Apply the standard LSTM forget bias in the cell equation.
        weight.bias = Some(Param::from_tensor(Tensor::zeros([4 * hidden_dim], device)));
        Self { weight, hidden_dim }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 2>,
        h: Tensor<B, 2>,
        c: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [batch, _] = x.dims();
        let device = x.device();
        let hidden = self.hidden_dim;

        let combined = Tensor::cat(vec![x, h], 1);
        let gates = self.weight.forward(combined);

        let i_gate = gates.clone().slice([0..batch, 0..hidden]);
        let j_gate = gates.clone().slice([0..batch, hidden..2 * hidden]);
        let f_gate = gates.clone().slice([0..batch, 2 * hidden..3 * hidden]);
        let o_gate = gates.slice([0..batch, 3 * hidden..4 * hidden]);

        let i_gate = burn::tensor::activation::sigmoid(i_gate);
        let f_gate = burn::tensor::activation::sigmoid(
            f_gate + Tensor::ones([batch, hidden], &device) * BASIC_LSTM_FORGET_BIAS,
        );
        let j_gate = j_gate.tanh();
        let o_gate = burn::tensor::activation::sigmoid(o_gate);

        let c_new = f_gate * c + i_gate * j_gate;
        let h_new = o_gate * c_new.clone().tanh();

        (h_new, c_new)
    }
}

#[derive(Module, Debug)]
pub struct GeneratorLstm<B: Backend> {
    cell: LstmCell<B>,
    hidden_dim: usize,
}

impl<B: Backend> GeneratorLstm<B> {
    pub fn new(input_dim: usize, hidden_dim: usize, device: &B::Device) -> Self {
        Self {
            cell: LstmCell::new(input_dim, hidden_dim, device),
            hidden_dim,
        }
    }

    pub fn forward_sequence(
        &self,
        input: Tensor<B, 3>,
        state: LstmState<B>,
    ) -> (Tensor<B, 3>, LstmState<B>) {
        let [batch, seq_len, input_dim] = input.dims();
        let mut h = state.h;
        let mut c = state.c;
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t = input.clone().slice([0..batch, t..t + 1, 0..input_dim]);
            let x_t = x_t.reshape([batch, input_dim]);

            let (h_new, c_new) = self.cell.forward(x_t, h, c);

            h = h_new.clone();
            c = c_new;

            outputs.push(h_new.reshape([batch, 1, self.hidden_dim]));
        }

        (Tensor::cat(outputs, 1), LstmState { h, c })
    }
}

#[derive(Clone)]
pub struct LstmState<B: Backend> {
    pub h: Tensor<B, 2>,
    pub c: Tensor<B, 2>,
}

impl<B: Backend> LstmState<B> {
    pub fn random(batch: usize, hidden_dim: usize, device: &B::Device) -> Self {
        let dist = burn::tensor::Distribution::Normal(0.0, 4.0);
        let h = Tensor::random([batch, hidden_dim], dist, device);
        let c = Tensor::random([batch, hidden_dim], dist, device);
        Self { h, c }
    }
}

pub struct PolicySample<B: Backend> {
    pub tokens: Vec<Vec<usize>>,
    pub initial_state: LstmState<B>,
}

#[derive(Module, Debug)]
pub struct Generator<B: Backend> {
    embedding: Embedding<B>,
    lstm: GeneratorLstm<B>,
    output_proj: Linear<B>,
    emb_dim: usize,
    hidden_dim: usize,
}

impl<B: Backend> Generator<B> {
    pub fn new(emb_dim: usize, hidden_dim: usize, device: &B::Device) -> Self {
        let embedding = EmbeddingConfig::new(VOCAB_SIZE + 1, emb_dim)
            .with_initializer(normal_0_1())
            .init(device);
        let lstm = GeneratorLstm::new(emb_dim, hidden_dim, device);
        let output_proj = LinearConfig::new(hidden_dim, VOCAB_SIZE)
            .with_bias(true)
            .with_initializer(glorot_uniform())
            .init(device);

        Self {
            embedding,
            lstm,
            output_proj,
            emb_dim,
            hidden_dim,
        }
    }

    pub fn has_valid_dimensions(&self) -> bool {
        self.embedding.weight.val().dims() == [VOCAB_SIZE + 1, self.emb_dim]
            && self.lstm.cell.weight.weight.val().dims()
                == [self.emb_dim + self.hidden_dim, 4 * self.hidden_dim]
            && self
                .lstm
                .cell
                .weight
                .bias
                .as_ref()
                .is_some_and(|bias| bias.val().dims() == [4 * self.hidden_dim])
            && self.output_proj.weight.val().dims() == [self.hidden_dim, VOCAB_SIZE]
            && self
                .output_proj
                .bias
                .as_ref()
                .is_some_and(|bias| bias.val().dims() == [VOCAB_SIZE])
    }

    pub fn forward_pretrain(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, _] = input_ids.dims();
        let device = input_ids.device();
        let state = LstmState::random(batch, self.hidden_dim, &device);
        self.forward_policy(input_ids, state)
    }

    pub fn forward_policy(
        &self,
        input_ids: Tensor<B, 2, Int>,
        state: LstmState<B>,
    ) -> Tensor<B, 3> {
        let embedded = self.embedding.forward(input_ids);
        let (hidden_out, _) = self.lstm.forward_sequence(embedded, state);
        self.output_proj.forward(hidden_out)
    }

    pub fn generate_sample(
        &self,
        batch: usize,
        device: &B::Device,
        temperature: f32,
    ) -> Result<Vec<Vec<usize>>, String> {
        Ok(self.sample_policy(batch, device, temperature)?.tokens)
    }

    pub fn sample_policy(
        &self,
        batch: usize,
        device: &B::Device,
        temperature: f32,
    ) -> Result<PolicySample<B>, String> {
        let initial_state = LstmState::random(batch, self.hidden_dim, device);
        let start = Tensor::<B, 2, Int>::full([batch, 1], GO_ID as i64, device);
        let tokens = token_tensor_to_rows(self.generate_from_state(
            start,
            temperature,
            initial_state.clone(),
            MAX_SEQ_LEN,
        ))?;
        Ok(PolicySample {
            tokens,
            initial_state,
        })
    }

    fn generate_from_state(
        &self,
        mut current_ids: Tensor<B, 2, Int>,
        temperature: f32,
        mut state: LstmState<B>,
        steps: usize,
    ) -> Tensor<B, 2, Int> {
        let device = current_ids.device();
        let mut sampled_steps = Vec::with_capacity(steps);

        for _ in 0..steps {
            let (logits, next_state) = self.decode_step(current_ids, state);
            state = next_state;
            let sampled = sample_ids_from_logits(logits, temperature, &device);
            current_ids = sampled.clone();
            sampled_steps.push(sampled);
        }

        Tensor::cat(sampled_steps, 1)
    }

    fn decode_step(
        &self,
        current_ids: Tensor<B, 2, Int>,
        state: LstmState<B>,
    ) -> (Tensor<B, 2>, LstmState<B>) {
        let [batch, _] = current_ids.dims();
        let embedded = self
            .embedding
            .forward(current_ids)
            .reshape([batch, self.emb_dim]);
        let (h, c) = self.lstm.cell.forward(embedded, state.h, state.c);
        let logits = self.output_proj.forward(h.clone());
        (logits, LstmState { h, c })
    }

    pub fn rollout_continuation(
        &self,
        prefix_ids: Tensor<B, 2, Int>,
        next_ids: &[usize],
        state: LstmState<B>,
        remaining_steps: usize,
        temperature: f32,
    ) -> Tensor<B, 2, Int> {
        let device = &prefix_ids.device();
        let [batch, _] = prefix_ids.dims();
        if remaining_steps == 0 {
            return Tensor::<B, 2, Int>::zeros([batch, 0], device);
        }

        let embedded = self.embedding.forward(prefix_ids);
        let (_, final_state) = self.lstm.forward_sequence(embedded, state);

        let current_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(
                next_ids.iter().map(|&id| id as i64).collect::<Vec<_>>(),
                [batch, 1],
            ),
            device,
        );
        self.generate_from_state(current_ids, temperature, final_state, remaining_steps)
    }
}

#[derive(Module, Debug)]
pub struct ConvFilter<B: Backend> {
    conv: Conv1d<B>,
}

impl<B: Backend> ConvFilter<B> {
    pub fn new(emb_dim: usize, num_filters: usize, kernel_size: usize, device: &B::Device) -> Self {
        let mut conv = Conv1dConfig::new(emb_dim, num_filters, kernel_size)
            .with_padding(burn::nn::PaddingConfig1d::Valid)
            .with_initializer(normal_0_1())
            .init(device);
        conv.bias = Some(Param::from_tensor(Tensor::full(
            [num_filters],
            DISCRIMINATOR_BIAS_INIT,
            device,
        )));
        Self { conv }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let activated = burn::tensor::activation::relu(self.conv.forward(input));
        let [batch, num_filters, _] = activated.dims();
        activated.max_dim(2).reshape([batch, num_filters])
    }
}

#[derive(Module, Debug)]
pub struct Highway<B: Backend> {
    transform: Linear<B>,
    gate: Linear<B>,
}

impl<B: Backend> Highway<B> {
    pub fn new(size: usize, device: &B::Device) -> Self {
        let transform = LinearConfig::new(size, size)
            .with_initializer(glorot_uniform())
            .init(device);
        let gate = LinearConfig::new(size, size)
            .with_initializer(glorot_uniform())
            .init(device);
        Self { transform, gate }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let transformed = burn::tensor::activation::relu(self.transform.forward(input.clone()));
        let gate = burn::tensor::activation::sigmoid(self.gate.forward(input.clone()));
        let one = gate.clone().ones_like();
        gate.clone() * transformed + (one - gate) * input
    }
}

#[derive(Module, Debug)]
pub struct Discriminator<B: Backend> {
    embedding: Embedding<B>,
    filters: Vec<ConvFilter<B>>,
    highway: Highway<B>,
    dropout: Dropout,
    output_proj: Linear<B>,
}

impl<B: Backend> Discriminator<B> {
    pub fn new(
        num_classes: usize,
        emb_dim: usize,
        filter_sizes: &[usize],
        num_filters: &[usize],
        device: &B::Device,
    ) -> Self {
        let embedding = EmbeddingConfig::new(VOCAB_SIZE, emb_dim)
            .with_initializer(Initializer::Uniform {
                min: -DISCRIMINATOR_EMBED_BOUND,
                max: DISCRIMINATOR_EMBED_BOUND,
            })
            .init(device);

        let filters = filter_sizes
            .iter()
            .zip(num_filters.iter())
            .map(|(&kernel, &count)| ConvFilter::new(emb_dim, count, kernel, device))
            .collect::<Vec<_>>();
        let num_filters_total = num_filters.iter().sum();
        let highway = Highway::new(num_filters_total, device);
        let dropout = DropoutConfig::new(DISCRIMINATOR_DROPOUT_PROB).init();
        let mut output_proj = LinearConfig::new(num_filters_total, num_classes)
            .with_initializer(normal_0_1())
            .init(device);
        output_proj.bias = Some(Param::from_tensor(Tensor::full(
            [num_classes],
            DISCRIMINATOR_BIAS_INIT,
            device,
        )));

        Self {
            embedding,
            filters,
            highway,
            dropout,
            output_proj,
        }
    }

    fn encode(&self, input_ids: Tensor<B, 2, Int>, training: bool) -> Tensor<B, 2> {
        let embedded = self.embedding.forward(input_ids).swap_dims(1, 2);
        let pooled = self
            .filters
            .iter()
            .map(|filter| filter.forward(embedded.clone()))
            .collect::<Vec<_>>();
        let hidden = self.highway.forward(Tensor::cat(pooled, 1));

        if training {
            self.dropout.forward(hidden)
        } else {
            hidden
        }
    }

    pub fn forward_logits(&self, input_ids: Tensor<B, 2, Int>, training: bool) -> Tensor<B, 2> {
        self.output_proj.forward(self.encode(input_ids, training))
    }

    pub fn forward_probs(&self, input_ids: Tensor<B, 2, Int>, training: bool) -> Tensor<B, 2> {
        burn::tensor::activation::softmax(self.forward_logits(input_ids, training), 1)
    }
}

fn sample_ids_from_logits<B: Backend>(
    logits: Tensor<B, 2>,
    temperature: f32,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let [batch, vocab_size] = logits.dims();
    let logits = if temperature == 1.0 {
        logits
    } else {
        logits / temperature
    };
    let uniform = Tensor::<B, 2>::random(
        [batch, vocab_size],
        Distribution::Uniform(1e-6, 1.0),
        device,
    );
    let gumbel = uniform.log().neg().log().neg();
    (logits + gumbel).argmax(1)
}

fn token_tensor_to_rows<B: Backend>(tokens: Tensor<B, 2, Int>) -> Result<Vec<Vec<usize>>, String> {
    let [batch, steps] = tokens.dims();
    let expected = batch * steps;
    let flat = tokens
        .into_data()
        .convert::<i64>()
        .to_vec::<i64>()
        .map_err(|err| format!("failed to read generated 6GAN token tensor: {err:?}"))?;

    if flat.len() != expected {
        return Err(format!(
            "generated 6GAN token tensor has {} values but shape [{batch}, {steps}] requires {expected}",
            flat.len()
        ));
    }

    Ok(flat
        .chunks(steps)
        .take(batch)
        .map(|row| row.iter().map(|&id| id as usize).collect())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn monte_carlo_penalties_follow_equations_4_8_and_10_at_every_position() {
        use super::super::{
            SixGan,
            reward::{AliasDetector, RewardEvaluator},
        };
        use rand::{SeedableRng, rngs::StdRng};
        let _guard = crate::ml::rng_guard();
        let device = Default::default();
        let mut generator = Generator::<TestBackend>::new(2, 3, &device);
        generator.output_proj.weight = Param::from_tensor(Tensor::zeros([3, 16], &device));
        let mut logits = vec![-10000.0f32; 16];
        logits[0] = 0.0;
        generator.output_proj.bias = Some(Param::from_tensor(Tensor::from_data(
            TensorData::new(logits, [16]),
            &device,
        )));
        let mut discriminator = Discriminator::<TestBackend>::new(3, 2, &[1], &[2], &device);
        discriminator.output_proj.weight = Param::from_tensor(Tensor::zeros([2, 3], &device));
        discriminator.output_proj.bias = Some(Param::from_tensor(Tensor::from_data(
            [0.2f32.ln(), 0.3f32.ln(), 0.5f32.ln()],
            &device,
        )));
        for searches in [1, 3] {
            let config = SixGan {
                rollout_num: searches,
                aliased_prefixes: vec!["::/16".parse().unwrap()],
                ..Default::default()
            };
            let aliases = AliasDetector::new(&config);
            let evaluator = RewardEvaluator {
                discriminator: &discriminator,
                aliases: &aliases,
                config: &config,
            };
            let sample = generator.sample_policy(2, &device, 1.0).unwrap();
            assert!(sample.tokens.iter().flatten().all(|&token| token == 0));
            let penalties = evaluator
                .compute(
                    &generator,
                    &sample,
                    1,
                    &device,
                    &mut StdRng::seed_from_u64(42),
                )
                .unwrap();
            for row in penalties {
                for (position, value) in row.into_iter().enumerate() {
                    let expected = 0.7
                        + if position < 4 {
                            9.0 * (position + 1) as f32 / 4.0
                        } else {
                            0.0
                        };
                    assert!(
                        (value - expected).abs() < 1e-5,
                        "position {position}: {value} != {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn lstm_cell_matches_scalar_gate_equations() {
        let _guard = crate::ml::rng_guard();
        let device = Default::default();
        let mut cell = LstmCell::<TestBackend>::new(1, 1, &device);
        cell.weight.weight = Param::from_tensor(Tensor::from_data(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            &device,
        ));
        let (h, c) = cell.forward(
            Tensor::from_data([[0.2]], &device),
            Tensor::from_data([[0.3]], &device),
            Tensor::from_data([[0.4]], &device),
        );
        let sigmoid = |value: f32| 1.0 / (1.0 + (-value).exp());
        let expected_c = sigmoid(0.2 * 0.3 + 0.3 * 0.7 + 1.0) * 0.4
            + sigmoid(0.2 * 0.1 + 0.3 * 0.5) * (0.2f32 * 0.2 + 0.3 * 0.6).tanh();
        let expected_h = sigmoid(0.2 * 0.4 + 0.3 * 0.8) * expected_c.tanh();
        assert!((c.into_scalar() - expected_c).abs() < 1e-6);
        assert!((h.into_scalar() - expected_h).abs() < 1e-6);
    }

    #[test]
    fn rollout_and_teacher_forcing_reuse_the_sampled_policy_state() {
        let _guard = crate::ml::rng_guard();
        let device = Default::default();
        TestBackend::seed(&device, 42);
        let generator = Generator::<TestBackend>::new(3, 4, &device);
        let sample = generator.sample_policy(2, &device, 0.7).unwrap();
        let mut inputs = Vec::new();
        for row in &sample.tokens {
            inputs.push(GO_ID as i64);
            inputs.extend(row[..31].iter().map(|&token| token as i64));
        }
        let inputs = Tensor::from_data(TensorData::new(inputs, [2, 32]), &device);
        let logits = generator.forward_policy(inputs.clone(), sample.initial_state.clone());
        let mut state = sample.initial_state.clone();
        for step in 0..32 {
            let (actual, next) =
                generator.decode_step(inputs.clone().slice([0..2, step..step + 1]), state);
            state = next;
            let expected = logits
                .clone()
                .slice([0..2, step..step + 1, 0..16])
                .reshape([2, 16]);
            let error = (actual - expected).abs().max().into_scalar();
            assert!(error < 1e-6);
        }
        for seed in 0..10 {
            TestBackend::seed(&device, seed);
            let expected = sample_ids_from_logits(
                logits.clone().slice([0..2, 31..32, 0..16]).reshape([2, 16]),
                0.7,
                &device,
            );
            TestBackend::seed(&device, seed);
            let next = sample.tokens.iter().map(|row| row[30]).collect::<Vec<_>>();
            let actual = generator.rollout_continuation(
                inputs.clone().slice([0..2, 0..31]),
                &next,
                sample.initial_state.clone(),
                1,
                0.7,
            );
            assert_eq!(actual.into_data(), expected.into_data());
        }
    }

    #[test]
    fn lstm_state_uses_paper_shape() {
        let _rng = crate::ml::rng_guard();
        let device = Default::default();
        let state = LstmState::<TestBackend>::random(2, 3, &device);
        assert_eq!(state.h.dims(), [2, 3]);
        assert_eq!(state.c.dims(), [2, 3]);
    }

    #[test]
    fn token_tensor_rows_are_batch_major() {
        let device = Default::default();
        let tokens = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![7, 9, 11, 8, 10, 12], [2, 3]),
            &device,
        );

        assert_eq!(
            token_tensor_to_rows(tokens).unwrap(),
            vec![vec![7, 9, 11], vec![8, 10, 12]]
        );
    }
}
