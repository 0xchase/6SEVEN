//! Burn-based GCNN-VAE architecture matching the 6GCVAE reference implementation.

use burn::{
    module::{Initializer, Module, Param},
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{Int, Tensor, backend::Backend},
};

use super::config::{HIDDEN_DIM, LATENT_DIM, N, VOCAB_SIZE};

const KERNEL_SIZE: usize = 3;
const GLOROT_GAIN: f64 = 1.0;
const KERAS_EMBED_INIT_MIN: f64 = -0.05;
const KERAS_EMBED_INIT_MAX: f64 = 0.05;

/// Gated convolution layer used by the original Keras GCNN module.
#[derive(Module, Debug)]
pub struct GatedConv1d<B: Backend> {
    conv: Conv1d<B>,
    output_dim: usize,
    residual: bool,
}

impl<B: Backend> GatedConv1d<B> {
    pub fn new(input_dim: usize, output_dim: usize, residual: bool, device: &B::Device) -> Self {
        if residual {
            assert_eq!(
                input_dim, output_dim,
                "6GCVAE residual gated convolution requires matching input/output dimensions"
            );
        }

        let conv = keras_conv1d(input_dim, output_dim * 2, KERNEL_SIZE, device);

        Self {
            conv,
            output_dim,
            residual,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _channels] = x.dims();

        let x_conv = x.clone().swap_dims(1, 2);
        let conv_out = self.conv.forward(x_conv).swap_dims(1, 2);

        let a = conv_out
            .clone()
            .slice([0..batch, 0..seq_len, 0..self.output_dim]);
        let b = conv_out.slice([0..batch, 0..seq_len, self.output_dim..self.output_dim * 2]);

        let gated = a * burn::tensor::activation::sigmoid(b);
        if self.residual { gated + x } else { gated }
    }
}

#[derive(Module, Debug)]
pub struct GcvaeEncoder<B: Backend> {
    embedding: Embedding<B>,
    gcnn1: GatedConv1d<B>,
    gcnn2: GatedConv1d<B>,
    mean_proj: Linear<B>,
    logvar_proj: Linear<B>,
}

impl<B: Backend> GcvaeEncoder<B> {
    pub fn new(device: &B::Device) -> Self {
        let embedding = keras_embedding(device);
        let gcnn1 = GatedConv1d::new(HIDDEN_DIM, HIDDEN_DIM, true, device);
        let gcnn2 = GatedConv1d::new(HIDDEN_DIM, HIDDEN_DIM, true, device);
        let mean_proj = keras_linear(HIDDEN_DIM, LATENT_DIM, device);
        let logvar_proj = keras_linear(HIDDEN_DIM, LATENT_DIM, device);

        Self {
            embedding,
            gcnn1,
            gcnn2,
            mean_proj,
            logvar_proj,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2, Int>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [batch, _] = input.dims();
        let h = self.embedding.forward(input);
        let h = self.gcnn1.forward(h);
        let h = self.gcnn2.forward(h);

        let pooled = h.mean_dim(1).reshape([batch, HIDDEN_DIM]);
        let z_mean = self.mean_proj.forward(pooled.clone());
        let z_log_var = self.logvar_proj.forward(pooled);

        (z_mean, z_log_var)
    }
}

#[derive(Module, Debug)]
pub struct GcvaeDecoder<B: Backend> {
    latent_proj: Linear<B>,
    gcnn: GatedConv1d<B>,
    output_proj: Linear<B>,
}

impl<B: Backend> GcvaeDecoder<B> {
    pub fn new(device: &B::Device) -> Self {
        let latent_proj = keras_linear(LATENT_DIM, HIDDEN_DIM * N, device);

        let gcnn = GatedConv1d::new(HIDDEN_DIM, HIDDEN_DIM, true, device);

        let output_proj = keras_linear(HIDDEN_DIM, VOCAB_SIZE, device);

        Self {
            latent_proj,
            gcnn,
            output_proj,
        }
    }

    pub fn forward_logits(&self, z: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, _] = z.dims();

        let h = self.latent_proj.forward(z);
        let h = h.reshape([batch, N, HIDDEN_DIM]);
        let h = self.gcnn.forward(h);

        self.output_proj.forward(h)
    }

    /// Validate persisted tensors before constructing an infallible draw stream.
    pub(super) fn validate(&self) -> Result<(), String> {
        validate_tensor(self.latent_proj.weight.val(), [LATENT_DIM, HIDDEN_DIM * N])?;
        validate_bias(&self.latent_proj.bias, HIDDEN_DIM * N)?;
        validate_tensor(
            self.gcnn.conv.weight.val(),
            [HIDDEN_DIM * 2, HIDDEN_DIM, KERNEL_SIZE],
        )?;
        if self.gcnn.conv.bias.is_some() {
            return Err("gated convolution must not have a bias".into());
        }
        validate_tensor(self.output_proj.weight.val(), [HIDDEN_DIM, VOCAB_SIZE])?;
        validate_bias(&self.output_proj.bias, VOCAB_SIZE)
    }
}

#[derive(Module, Debug)]
pub struct GcvaeNet<B: Backend> {
    pub encoder: GcvaeEncoder<B>,
    pub decoder: GcvaeDecoder<B>,
}

impl<B: Backend> GcvaeNet<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            encoder: GcvaeEncoder::new(device),
            decoder: GcvaeDecoder::new(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2, Int>) -> (Tensor<B, 3>, Tensor<B, 2>, Tensor<B, 2>) {
        let (z_mean, z_log_var) = self.encoder.forward(input);

        let std = (z_log_var.clone() / 2.0).exp();
        let epsilon = Tensor::random_like(&std, burn::tensor::Distribution::Normal(0.0, 1.0));
        let z = z_mean.clone() + std * epsilon;

        let reconstruction_logits = self.decoder.forward_logits(z);
        (reconstruction_logits, z_mean, z_log_var)
    }
}

fn validate_bias<B: Backend>(
    bias: &Option<Param<Tensor<B, 1>>>,
    size: usize,
) -> Result<(), String> {
    let bias = bias.as_ref().ok_or("missing dense bias")?;
    validate_tensor(bias.val(), [size])
}

fn validate_tensor<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    expected: [usize; D],
) -> Result<(), String> {
    if tensor.dims() != expected {
        return Err(format!(
            "unexpected decoder tensor shape: {:?}, expected {expected:?}",
            tensor.dims()
        ));
    }
    let data = tensor.into_data().convert::<f32>();
    let values = data
        .as_slice::<f32>()
        .map_err(|error| format!("unreadable decoder tensor: {error:?}"))?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err("decoder weights must be finite".into());
    }
    Ok(())
}

fn keras_embedding<B: Backend>(device: &B::Device) -> Embedding<B> {
    EmbeddingConfig::new(VOCAB_SIZE, HIDDEN_DIM)
        .with_initializer(Initializer::Uniform {
            min: KERAS_EMBED_INIT_MIN,
            max: KERAS_EMBED_INIT_MAX,
        })
        .init(device)
}

fn keras_linear<B: Backend>(input_dim: usize, output_dim: usize, device: &B::Device) -> Linear<B> {
    let mut linear = LinearConfig::new(input_dim, output_dim)
        .with_initializer(Initializer::XavierUniform { gain: GLOROT_GAIN })
        .init(device);
    linear.bias = Some(Param::from_tensor(Tensor::zeros([output_dim], device)));
    linear
}

fn keras_conv1d<B: Backend>(
    input_dim: usize,
    output_dim: usize,
    kernel_size: usize,
    device: &B::Device,
) -> Conv1d<B> {
    // Use bias-free convolutions with explicit Glorot bounds to match Keras.
    let fan_in = (input_dim * kernel_size) as f64;
    let fan_out = (output_dim * kernel_size) as f64;
    let limit = (6.0 / (fan_in + fan_out)).sqrt();

    Conv1dConfig::new(input_dim, output_dim, kernel_size)
        .with_padding(burn::nn::PaddingConfig1d::Same)
        .with_bias(false)
        .with_initializer(Initializer::Uniform {
            min: -limit,
            max: limit,
        })
        .init(device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::CpuBackend;

    fn assert_zero_bias(bias: &Option<Param<Tensor<CpuBackend, 1>>>) {
        let values = bias
            .as_ref()
            .expect("keras layers keep bias enabled")
            .val()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        assert!(values.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn argmax_uses_the_first_nybble_on_equal_probabilities() {
        let _rng = crate::ml::rng_guard();
        let device = Default::default();
        let mut decoder = GcvaeDecoder::<CpuBackend>::new(&device);
        decoder.output_proj.weight =
            Param::from_tensor(Tensor::zeros([HIDDEN_DIM, VOCAB_SIZE], &device));
        decoder.output_proj.bias = Some(Param::from_tensor(Tensor::zeros([VOCAB_SIZE], &device)));
        let tokens = decoder
            .forward_logits(Tensor::zeros([1, LATENT_DIM], &device))
            .argmax(2)
            .into_data()
            .convert::<i64>()
            .to_vec::<i64>()
            .unwrap();
        assert_eq!(tokens, vec![0; N]);
    }

    #[test]
    fn encoder_and_decoder_match_numpy_forward_fixture() {
        fn weights<const D: usize>(shape: [usize; D]) -> Tensor<CpuBackend, D> {
            let values = (0..shape.iter().product())
                .map(|index| (index as i32 % 17 - 8) as f32 / 1000.0)
                .collect::<Vec<_>>();
            Tensor::from_data(
                burn::tensor::TensorData::new(values, shape),
                &Default::default(),
            )
        }
        fn linear(layer: &mut Linear<CpuBackend>) {
            layer.weight = Param::from_tensor(weights(layer.weight.val().dims()));
            let size = layer.bias.as_ref().unwrap().val().dims()[0];
            let values = (0..size)
                .map(|index| (index as i32 % 7 - 3) as f32 / 1000.0)
                .collect::<Vec<_>>();
            layer.bias = Some(Param::from_tensor(Tensor::from_data(
                burn::tensor::TensorData::new(values, [size]),
                &Default::default(),
            )));
        }
        let _rng = crate::ml::rng_guard();
        let device = Default::default();
        let mut net = GcvaeNet::<CpuBackend>::new(&device);
        net.encoder.embedding.weight = Param::from_tensor(weights([16, 64]));
        for layer in [
            &mut net.encoder.gcnn1,
            &mut net.encoder.gcnn2,
            &mut net.decoder.gcnn,
        ] {
            layer.conv.weight = Param::from_tensor(weights([128, 64, 3]));
        }
        for layer in [
            &mut net.encoder.mean_proj,
            &mut net.encoder.logvar_proj,
            &mut net.decoder.latent_proj,
            &mut net.decoder.output_proj,
        ] {
            linear(layer);
        }
        let input: Vec<i64> = (0..32)
            .map(|i| i % 16)
            .chain((0..32).map(|i| i * 3 % 16))
            .collect();
        let input = Tensor::from_data(burn::tensor::TensorData::new(input, [2, 32]), &device);
        let (mean, log_var) = net.encoder.forward(input);
        let values = mean.clone().into_data().to_vec::<f32>().unwrap();
        for (index, expected) in [0, 1, 15, 63, 64, 65, 79, 127].into_iter().zip([
            -0.0029886993,
            -0.0020145514,
            -0.0019645506,
            -0.0029995906,
            -0.0029904144,
            -0.0020146742,
            -0.001_967_194,
            -0.002_999_178,
        ]) {
            assert!((f64::from(values[index]) - expected).abs() < 1e-8);
        }
        let latent = mean + (log_var / 2.0).exp() * weights([2, 64]);
        let values = net
            .decoder
            .forward_logits(latent)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (index, expected) in [0, 1, 15, 511, 512, 513, 527, 1023].into_iter().zip([
            -0.0029286286,
            -0.0018735491,
            -0.0020083204,
            -0.0019281036,
            -0.0029897832,
            -0.0018710409,
            -0.0020348777,
            -0.0019453799,
        ]) {
            assert!((f64::from(values[index]) - expected).abs() < 1e-8);
        }
    }

    #[test]
    fn gated_convolution_matches_hand_computed_same_padding() {
        let _rng = crate::ml::rng_guard();
        let device = Default::default();
        let mut layer = GatedConv1d::<CpuBackend>::new(1, 1, true, &device);
        // A uses [1, 2, 3] and a zero B gives a gate of 1/2.
        layer.conv.weight = Param::from_tensor(Tensor::from_data(
            [[[1.0, 2.0, 3.0]], [[0.0, 0.0, 0.0]]],
            &device,
        ));
        let output = layer.forward(Tensor::from_data([[[1.0], [2.0], [4.0]]], &device));
        assert_eq!(
            output.into_data().to_vec::<f32>().unwrap(),
            vec![5.0, 10.5, 9.0]
        );
    }

    #[test]
    fn decoder_rejects_wrong_shapes_and_non_finite_parameters() {
        let _rng = crate::ml::rng_guard();
        let device = Default::default();
        let mut decoder = GcvaeDecoder::<CpuBackend>::new(&device);
        assert!(decoder.validate().is_ok());
        decoder.output_proj.bias = Some(Param::from_tensor(Tensor::zeros([1], &device)));
        assert!(decoder.validate().is_err());
        decoder.output_proj.bias = Some(Param::from_tensor(Tensor::full(
            [VOCAB_SIZE],
            f32::NAN,
            &device,
        )));
        assert!(decoder.validate().is_err());
    }

    #[test]
    fn keras_initializers_match_reference_defaults() {
        let _rng = crate::ml::rng_guard();
        let device = Default::default();
        let model = GcvaeNet::<CpuBackend>::new(&device);

        let embedding = model
            .encoder
            .embedding
            .weight
            .val()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        assert!(
            embedding
                .iter()
                .all(|value| (*value as f64) >= KERAS_EMBED_INIT_MIN
                    && (*value as f64) <= KERAS_EMBED_INIT_MAX)
        );

        assert!(model.encoder.gcnn1.conv.bias.is_none());
        assert!(model.encoder.gcnn2.conv.bias.is_none());
        assert!(model.decoder.gcnn.conv.bias.is_none());

        assert_zero_bias(&model.encoder.mean_proj.bias);
        assert_zero_bias(&model.encoder.logvar_proj.bias);
        assert_zero_bias(&model.decoder.latent_proj.bias);
        assert_zero_bias(&model.decoder.output_proj.bias);
    }
}
