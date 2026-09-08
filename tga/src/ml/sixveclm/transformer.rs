use burn::{
    module::{Initializer, Module, Param},
    nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig},
    tensor::{Bool, Int, Tensor, TensorData, activation, backend::Backend},
};
use serde::{Deserialize, Serialize};

#[cfg(test)]
use super::TOTAL_LEN;

mod train;
pub(super) use train::train_transformer;

const LAYER_NORM_EPS: f64 = 1e-6;
const MIN_MASK_SCORE: f32 = -1.0e9;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransformerHyperParams {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub n_layers: usize,
    pub dropout: f64,
    pub prefix_len: usize,
    pub decoder_len: usize,
}

#[derive(Clone, Debug)]
pub struct TransformerTrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub noam_factor: f64,
    pub noam_warmup: usize,
    pub seed: u64,
    pub shuffle: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct TransformerArtifacts {
    pub bytes: Vec<u8>,
    pub hyper: TransformerHyperParams,
}

impl Default for TransformerHyperParams {
    fn default() -> Self {
        Self {
            d_model: 0,
            n_heads: 0,
            d_ff: 0,
            n_layers: 0,
            dropout: 0.0,
            prefix_len: 0,
            decoder_len: 0,
        }
    }
}

#[derive(Module, Debug)]
struct ReferenceLayerNorm<B: Backend> {
    gamma: Param<Tensor<B, 1>>,
    beta: Param<Tensor<B, 1>>,
    epsilon: f64,
    d_model: usize,
}

impl<B: Backend> ReferenceLayerNorm<B> {
    fn new(d_model: usize, epsilon: f64, device: &B::Device) -> Self {
        Self {
            gamma: Initializer::Ones.init([d_model], device),
            beta: Initializer::Zeros.init([d_model], device),
            epsilon,
            d_model,
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let mean = input.clone().mean_dim(2);
        let centered = input.sub(mean);
        let denom = self.d_model.saturating_sub(1).max(1) as f32;
        let std = centered
            .clone()
            .powi_scalar(2)
            .sum_dim(2)
            .div_scalar(denom)
            .clamp_min(f32::MIN_POSITIVE)
            .sqrt();
        let normalized = centered.div(std.add_scalar(self.epsilon as f32));

        normalized
            .mul(self.gamma.val().unsqueeze())
            .add(self.beta.val().unsqueeze())
    }
}

#[derive(Module, Debug)]
struct ReferencePositionalEncoding<B: Backend> {
    sinusoids: Tensor<B, 3>,
    dropout: Dropout,
}

impl<B: Backend> ReferencePositionalEncoding<B> {
    fn new(d_model: usize, dropout: f64, max_len: usize, device: &B::Device) -> Self {
        Self {
            sinusoids: generate_sinusoids(d_model, max_len, device),
            dropout: DropoutConfig::new(dropout).init(),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, seq_len, d_model] = input.dims();
        let pe = self.sinusoids.clone().slice([0..1, 0..seq_len, 0..d_model]);
        self.dropout.forward(input.add(pe))
    }
}

#[derive(Module, Debug)]
struct ReferencePositionwiseFeedForward<B: Backend> {
    inner: Linear<B>,
    outer: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> ReferencePositionwiseFeedForward<B> {
    fn new(
        d_model: usize,
        d_ff: usize,
        dropout: f64,
        initializer: Initializer,
        device: &B::Device,
    ) -> Self {
        Self {
            inner: LinearConfig::new(d_model, d_ff)
                .with_initializer(initializer.clone())
                .init(device),
            outer: LinearConfig::new(d_ff, d_model)
                .with_initializer(initializer)
                .init(device),
            dropout: DropoutConfig::new(dropout).init(),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = self.inner.forward(input);
        let hidden = activation::relu(hidden);
        let hidden = self.dropout.forward(hidden);
        self.outer.forward(hidden)
    }
}

#[derive(Module, Debug)]
struct ReferenceMultiHeadAttention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    output: Linear<B>,
    dropout: Dropout,
    d_model: usize,
    n_heads: usize,
    d_k: usize,
}

impl<B: Backend> ReferenceMultiHeadAttention<B> {
    fn new(
        d_model: usize,
        n_heads: usize,
        dropout: f64,
        initializer: Initializer,
        device: &B::Device,
    ) -> Self {
        Self {
            query: LinearConfig::new(d_model, d_model)
                .with_initializer(initializer.clone())
                .init(device),
            key: LinearConfig::new(d_model, d_model)
                .with_initializer(initializer.clone())
                .init(device),
            value: LinearConfig::new(d_model, d_model)
                .with_initializer(initializer.clone())
                .init(device),
            output: LinearConfig::new(d_model, d_model)
                .with_initializer(initializer)
                .init(device),
            dropout: DropoutConfig::new(dropout).init(),
            d_model,
            n_heads,
            d_k: d_model / n_heads,
        }
    }

    fn forward(
        &self,
        query: Tensor<B, 3>,
        key: Tensor<B, 3>,
        value: Tensor<B, 3>,
        mask: Option<Tensor<B, 3, Bool>>,
    ) -> Tensor<B, 3> {
        let [batch_size, query_len, _] = query.dims();

        let query = self.project(query, &self.query);
        let key = self.project(key, &self.key);
        let value = self.project(value, &self.value);

        let mut scores = query
            .matmul(key.transpose())
            .div_scalar((self.d_k as f32).sqrt());
        if let Some(mask) = mask {
            let [batch_size, query_len, key_len] = mask.dims();
            let blocked = mask.bool_not().reshape([batch_size, 1, query_len, key_len]);
            scores = scores.mask_fill(blocked, MIN_MASK_SCORE);
        }

        let attention = activation::softmax(scores, 3);
        let attention = self.dropout.forward(attention);
        let context = attention.matmul(value);
        let context = context
            .swap_dims(1, 2)
            .reshape([batch_size, query_len, self.d_model]);

        self.output.forward(context)
    }

    fn project(&self, tensor: Tensor<B, 3>, linear: &Linear<B>) -> Tensor<B, 4> {
        let [batch_size, seq_len, _] = tensor.dims();
        linear
            .forward(tensor)
            .reshape([batch_size, seq_len, self.n_heads, self.d_k])
            .swap_dims(1, 2)
    }
}

#[derive(Module, Debug)]
struct ReferenceEncoderLayer<B: Backend> {
    self_attn: ReferenceMultiHeadAttention<B>,
    feed_forward: ReferencePositionwiseFeedForward<B>,
    norm_1: ReferenceLayerNorm<B>,
    norm_2: ReferenceLayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> ReferenceEncoderLayer<B> {
    fn new(hyper: &TransformerHyperParams, initializer: Initializer, device: &B::Device) -> Self {
        Self {
            self_attn: ReferenceMultiHeadAttention::new(
                hyper.d_model,
                hyper.n_heads,
                hyper.dropout,
                initializer.clone(),
                device,
            ),
            feed_forward: ReferencePositionwiseFeedForward::new(
                hyper.d_model,
                hyper.d_ff,
                hyper.dropout,
                initializer,
                device,
            ),
            norm_1: ReferenceLayerNorm::new(hyper.d_model, LAYER_NORM_EPS, device),
            norm_2: ReferenceLayerNorm::new(hyper.d_model, LAYER_NORM_EPS, device),
            dropout: DropoutConfig::new(hyper.dropout).init(),
        }
    }

    fn forward(&self, input: Tensor<B, 3>, mask: Option<Tensor<B, 3, Bool>>) -> Tensor<B, 3> {
        let residual = input.clone();
        let normalized = self.norm_1.forward(input);
        let attended =
            self.self_attn
                .forward(normalized.clone(), normalized.clone(), normalized, mask);
        let input = residual.add(self.dropout.forward(attended));

        let residual = input.clone();
        let normalized = self.norm_2.forward(input);
        let forwarded = self.feed_forward.forward(normalized);

        residual.add(self.dropout.forward(forwarded))
    }
}

#[derive(Module, Debug)]
struct ReferenceEncoder<B: Backend> {
    layers: Vec<ReferenceEncoderLayer<B>>,
    norm: ReferenceLayerNorm<B>,
}

impl<B: Backend> ReferenceEncoder<B> {
    fn new(hyper: &TransformerHyperParams, initializer: Initializer, device: &B::Device) -> Self {
        let layers = (0..hyper.n_layers)
            .map(|_| ReferenceEncoderLayer::new(hyper, initializer.clone(), device))
            .collect();
        Self {
            layers,
            norm: ReferenceLayerNorm::new(hyper.d_model, LAYER_NORM_EPS, device),
        }
    }

    fn forward(&self, mut input: Tensor<B, 3>, mask: Option<Tensor<B, 3, Bool>>) -> Tensor<B, 3> {
        for layer in &self.layers {
            input = layer.forward(input, mask.clone());
        }
        self.norm.forward(input)
    }
}

#[derive(Module, Debug)]
struct ReferenceDecoderLayer<B: Backend> {
    self_attn: ReferenceMultiHeadAttention<B>,
    src_attn: ReferenceMultiHeadAttention<B>,
    feed_forward: ReferencePositionwiseFeedForward<B>,
    norm_1: ReferenceLayerNorm<B>,
    norm_2: ReferenceLayerNorm<B>,
    norm_3: ReferenceLayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> ReferenceDecoderLayer<B> {
    fn new(hyper: &TransformerHyperParams, initializer: Initializer, device: &B::Device) -> Self {
        Self {
            self_attn: ReferenceMultiHeadAttention::new(
                hyper.d_model,
                hyper.n_heads,
                hyper.dropout,
                initializer.clone(),
                device,
            ),
            src_attn: ReferenceMultiHeadAttention::new(
                hyper.d_model,
                hyper.n_heads,
                hyper.dropout,
                initializer.clone(),
                device,
            ),
            feed_forward: ReferencePositionwiseFeedForward::new(
                hyper.d_model,
                hyper.d_ff,
                hyper.dropout,
                initializer,
                device,
            ),
            norm_1: ReferenceLayerNorm::new(hyper.d_model, LAYER_NORM_EPS, device),
            norm_2: ReferenceLayerNorm::new(hyper.d_model, LAYER_NORM_EPS, device),
            norm_3: ReferenceLayerNorm::new(hyper.d_model, LAYER_NORM_EPS, device),
            dropout: DropoutConfig::new(hyper.dropout).init(),
        }
    }

    fn forward(
        &self,
        input: Tensor<B, 3>,
        memory: Tensor<B, 3>,
        src_mask: Option<Tensor<B, 3, Bool>>,
        tgt_mask: Option<Tensor<B, 3, Bool>>,
    ) -> Tensor<B, 3> {
        let residual = input.clone();
        let normalized = self.norm_1.forward(input);
        let attended =
            self.self_attn
                .forward(normalized.clone(), normalized.clone(), normalized, tgt_mask);
        let input = residual.add(self.dropout.forward(attended));

        let residual = input.clone();
        let normalized = self.norm_2.forward(input);
        let attended = self
            .src_attn
            .forward(normalized, memory.clone(), memory, src_mask);
        let input = residual.add(self.dropout.forward(attended));

        let residual = input.clone();
        let normalized = self.norm_3.forward(input);
        let forwarded = self.feed_forward.forward(normalized);

        residual.add(self.dropout.forward(forwarded))
    }
}

#[derive(Module, Debug)]
struct ReferenceDecoder<B: Backend> {
    layers: Vec<ReferenceDecoderLayer<B>>,
    norm: ReferenceLayerNorm<B>,
}

impl<B: Backend> ReferenceDecoder<B> {
    fn new(hyper: &TransformerHyperParams, initializer: Initializer, device: &B::Device) -> Self {
        let layers = (0..hyper.n_layers)
            .map(|_| ReferenceDecoderLayer::new(hyper, initializer.clone(), device))
            .collect();
        Self {
            layers,
            norm: ReferenceLayerNorm::new(hyper.d_model, LAYER_NORM_EPS, device),
        }
    }

    fn forward(
        &self,
        mut input: Tensor<B, 3>,
        memory: Tensor<B, 3>,
        src_mask: Option<Tensor<B, 3, Bool>>,
        tgt_mask: Option<Tensor<B, 3, Bool>>,
    ) -> Tensor<B, 3> {
        for layer in &self.layers {
            input = layer.forward(input, memory.clone(), src_mask.clone(), tgt_mask.clone());
        }
        self.norm.forward(input)
    }
}

#[derive(Module, Debug)]
pub struct Ipv6Transformer<B: Backend> {
    encoder: ReferenceEncoder<B>,
    decoder: ReferenceDecoder<B>,
    src_embed: Embedding<B>,
    tgt_embed: Embedding<B>,
    pos_encoding: ReferencePositionalEncoding<B>,
    projection: Linear<B>,
    scale: f32,
}

impl<B: Backend> Ipv6Transformer<B> {
    pub fn new(
        hyper: &TransformerHyperParams,
        vocab_size: usize,
        embedding_bank: &Tensor<B, 2>,
        device: &B::Device,
    ) -> Self {
        assert_eq!(embedding_bank.dims(), [vocab_size, hyper.d_model]);
        let initializer = Initializer::XavierUniform { gain: 1.0 };
        let encoder = ReferenceEncoder::new(hyper, initializer.clone(), device);
        let decoder = ReferenceDecoder::new(hyper, initializer.clone(), device);
        let src_embed = frozen_embedding(embedding_bank, device);
        let tgt_embed = frozen_embedding(embedding_bank, device);
        let pos_encoding = ReferencePositionalEncoding::new(
            hyper.d_model,
            hyper.dropout,
            hyper.prefix_len + hyper.decoder_len,
            device,
        );
        let projection = LinearConfig::new(hyper.d_model, hyper.d_model)
            .with_bias(true)
            .with_initializer(initializer)
            .init(device);

        Self {
            encoder,
            decoder,
            src_embed,
            tgt_embed,
            pos_encoding,
            projection,
            scale: (hyper.d_model as f32).sqrt(),
        }
    }

    pub fn forward(
        &self,
        src: Tensor<B, 2, Int>,
        tgt: Tensor<B, 2, Int>,
        tgt_mask: Tensor<B, 3, Bool>,
    ) -> Tensor<B, 3> {
        let memory = self.encode(src);
        self.decode(memory, tgt, tgt_mask)
    }

    pub fn encode(&self, src: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let src_embed = self.positional_embed(self.src_embed.forward(src));
        self.encoder.forward(src_embed, None)
    }

    pub fn decode(
        &self,
        memory: Tensor<B, 3>,
        tgt: Tensor<B, 2, Int>,
        tgt_mask: Tensor<B, 3, Bool>,
    ) -> Tensor<B, 3> {
        let tgt_embed = self.positional_embed(self.tgt_embed.forward(tgt));
        self.decoder
            .forward(tgt_embed, memory, None, Some(tgt_mask))
    }

    pub fn project(&self, tensor: Tensor<B, 3>) -> Tensor<B, 3> {
        let projected = self.projection.forward(tensor);
        activation::sigmoid(projected)
    }

    fn positional_embed(&self, tensor: Tensor<B, 3>) -> Tensor<B, 3> {
        self.pos_encoding.forward(tensor.mul_scalar(self.scale))
    }
}

fn frozen_embedding<B: Backend>(bank: &Tensor<B, 2>, device: &B::Device) -> Embedding<B> {
    let [vocab, d_model] = bank.dims();
    let mut embed = EmbeddingConfig::new(vocab, d_model)
        .with_initializer(Initializer::Zeros)
        .init(device);
    // Preserve pretrained inputs as specified in section 5.1.
    embed.weight = Param::from_tensor(bank.clone().detach()).set_require_grad(false);
    embed
}

fn generate_sinusoids<B: Backend>(
    d_model: usize,
    max_len: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let mut values = Vec::with_capacity(max_len * d_model);
    for position in 0..max_len {
        for dim in (0..d_model).step_by(2) {
            let div_term = ((dim as f32) * (-(10_000.0f32).ln() / d_model as f32)).exp();
            values.push((position as f32 * div_term).sin());
            if dim + 1 < d_model {
                values.push((position as f32 * div_term).cos());
            }
        }
    }

    Tensor::from_data(TensorData::new(values, [1, max_len, d_model]), device)
}

pub(super) fn subsequent_mask<B: Backend>(
    batch_size: usize,
    seq_len: usize,
    device: &B::Device,
) -> Tensor<B, 3, Bool> {
    let mut mask = Vec::with_capacity(batch_size * seq_len * seq_len);
    for _ in 0..batch_size {
        for row in 0..seq_len {
            for col in 0..seq_len {
                mask.push(col <= row);
            }
        }
    }

    Tensor::from_data(
        TensorData::new(mask, [batch_size, seq_len, seq_len]),
        device,
    )
}

#[cfg(test)]
mod tests {
    use super::train::{PreparedBatch, append_sequence, noam_rate};
    use super::*;

    #[test]
    fn append_sequence_copies_first_tail_token() {
        let hyper = TransformerHyperParams {
            d_model: 100,
            n_heads: 10,
            d_ff: 2048,
            n_layers: 6,
            dropout: 0.1,
            prefix_len: 16,
            decoder_len: 16,
        };

        let mut tokens = [0u16; TOTAL_LEN];
        for (idx, token) in tokens.iter_mut().enumerate() {
            *token = idx as u16;
        }

        let mut batch = PreparedBatch::default();
        append_sequence(&mut batch, &tokens, &hyper).unwrap();

        assert_eq!(batch.src, (0..16).collect::<Vec<_>>());
        assert_eq!(batch.decoder_in, (16..31).collect::<Vec<_>>());
        assert_eq!(batch.decoder_target, (17..32).collect::<Vec<_>>());
    }

    #[test]
    fn noam_schedule_matches_reference_shape() {
        let first = noam_rate(1, 100, 1.0, 400);
        let warm = noam_rate(400, 100, 1.0, 400);
        let late = noam_rate(10_000, 100, 1.0, 400);

        assert!(warm > first);
        assert!(warm > late);
    }

    #[test]
    fn subsequent_mask_matches_reference_semantics() {
        type TestBackend = burn::backend::ndarray::NdArray<f32>;
        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();
        let mask = subsequent_mask::<TestBackend>(1, 4, &device)
            .into_data()
            .to_vec::<bool>()
            .unwrap();

        assert_eq!(
            mask,
            vec![
                true, false, false, false, true, true, false, false, true, true, true, false, true,
                true, true, true
            ]
        );
    }

    #[test]
    fn sinusoid_generation_supports_odd_model_widths() {
        type TestBackend = burn::backend::ndarray::NdArray<f32>;
        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();
        let values = generate_sinusoids::<TestBackend>(3, 2, &device)
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        assert_eq!(values.len(), 6);
    }

    #[test]
    fn transformer_preserves_pretrained_embeddings() {
        let _rng = crate::ml::rng_guard();
        type TestBackend = burn::backend::Autodiff<burn::backend::ndarray::NdArray<f32>>;
        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();
        let bank =
            Tensor::<TestBackend, 2>::from_data(TensorData::new(vec![0.25; 12], [4, 3]), &device);
        let hyper = TransformerHyperParams {
            d_model: 3,
            n_heads: 1,
            d_ff: 4,
            n_layers: 1,
            dropout: 0.0,
            prefix_len: 2,
            decoder_len: 2,
        };

        let model = Ipv6Transformer::<TestBackend>::new(&hyper, 4, &bank, &device);
        let src_weights = model
            .src_embed
            .weight
            .val()
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        assert_eq!(src_weights, vec![0.25; 12]);
        assert_eq!(
            model
                .tgt_embed
                .weight
                .val()
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            src_weights
        );
        assert!(!model.src_embed.weight.val().is_require_grad());
        assert!(!model.tgt_embed.weight.val().is_require_grad());
    }

    #[test]
    fn decoder_predictions_do_not_depend_on_future_tokens() {
        let _rng = crate::ml::rng_guard();
        type B = burn::backend::ndarray::NdArray<f32>;
        let device = Default::default();
        let hyper = TransformerHyperParams {
            d_model: 4,
            n_heads: 2,
            d_ff: 8,
            n_layers: 1,
            dropout: 0.0,
            prefix_len: 2,
            decoder_len: 3,
        };
        let bank = Tensor::<B, 2>::from_floats(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.1, 0.2, 0.3, 0.4],
                [0.4, 0.3, 0.2, 0.1],
                [0.2, 0.1, 0.4, 0.3],
            ],
            &device,
        );
        let model = Ipv6Transformer::<B>::new(&hyper, 4, &bank, &device);
        let memory = model.encode(Tensor::from_ints([[1, 2]], &device));
        let predict = |tail| {
            model
                .decode(
                    memory.clone(),
                    Tensor::from_ints([[1, tail]], &device),
                    subsequent_mask(1, 2, &device),
                )
                .slice([0..1, 0..1, 0..4])
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        };
        assert_eq!(predict(2), predict(3));
    }
}

#[cfg(test)]
mod reference_tests;
