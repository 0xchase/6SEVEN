use super::*;
use burn::{
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{linalg, module::embedding},
};

type B = burn::backend::Autodiff<burn::backend::ndarray::NdArray<f32>>;

fn initialize_linear(layer: &mut Linear<B>, device: &<B as Backend>::Device) {
    let [input, output] = layer.weight.val().dims();
    let weights = (0..input)
        .flat_map(|col| {
            (0..output).map(move |row| (((row * input + col + 1) as f64 * 0.17).sin() * 0.2) as f32)
        })
        .collect::<Vec<_>>();
    layer.weight = Param::from_tensor(Tensor::from_data(
        TensorData::new(weights, [input, output]),
        device,
    ));
    let bias: Vec<_> = (0..output)
        .map(|row| ((row as f64 - 0.5) * 0.01) as f32)
        .collect();
    layer.bias = Some(Param::from_tensor(Tensor::from_data(
        TensorData::new(bias, [output]),
        device,
    )));
}

fn initialize_attention(
    attention: &mut ReferenceMultiHeadAttention<B>,
    device: &<B as Backend>::Device,
) {
    for linear in [
        &mut attention.query,
        &mut attention.key,
        &mut attention.value,
        &mut attention.output,
    ] {
        initialize_linear(linear, device);
    }
}

fn assert_values<const D: usize>(tensor: Tensor<B, D>, fixture: &serde_json::Value, key: &str) {
    let actual = tensor.into_data().to_vec::<f32>().unwrap();
    let expected = fixture[key].as_array().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let expected = expected.as_f64().unwrap();
        assert!(
            (f64::from(*actual) - expected).abs() < 2e-5,
            "{key}[{index}]: {actual} differs from {expected}"
        );
    }
}

#[test]
fn transformer_forward_gradients_and_adam_step_match_pytorch_reference() {
    let _rng = crate::ml::rng_guard();
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/sixveclm/transformer.json"
    ))
    .unwrap();
    let device = Default::default();
    let bank: Vec<f32> = fixture["bank"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap() as f32)
        .collect();
    let bank = Tensor::<B, 2>::from_data(TensorData::new(bank, [6, 4]), &device);
    let hyper = TransformerHyperParams {
        d_model: 4,
        n_heads: 2,
        d_ff: 8,
        n_layers: 2,
        dropout: 0.0,
        prefix_len: 3,
        decoder_len: 4,
    };
    let mut model = Ipv6Transformer::new(&hyper, 6, &bank, &device);
    for layer in &mut model.encoder.layers {
        initialize_attention(&mut layer.self_attn, &device);
        initialize_linear(&mut layer.feed_forward.inner, &device);
        initialize_linear(&mut layer.feed_forward.outer, &device);
    }
    for layer in &mut model.decoder.layers {
        initialize_attention(&mut layer.self_attn, &device);
        initialize_attention(&mut layer.src_attn, &device);
        initialize_linear(&mut layer.feed_forward.inner, &device);
        initialize_linear(&mut layer.feed_forward.outer, &device);
    }
    initialize_linear(&mut model.projection, &device);
    let src = Tensor::from_ints([[1, 2, 3], [3, 2, 1]], &device);
    let tgt = Tensor::from_ints([[4, 5, 1], [1, 5, 4]], &device);
    let labels = Tensor::from_ints([[5, 1, 2], [5, 4, 3]], &device);
    let mask = subsequent_mask(2, 3, &device);
    let memory = model.encode(src.clone());
    assert_values(memory.clone(), &fixture, "memory");
    let projected = model.project(model.decode(memory, tgt.clone(), mask.clone()));
    assert_values(projected.clone(), &fixture, "projected");
    let cosine = linalg::cosine_similarity(projected, embedding(bank, labels), 2, Some(1e-8));
    let loss = (cosine.clone().ones_like() - cosine).mean();
    assert!(
        (f64::from(loss.clone().into_scalar()) - fixture["loss"].as_f64().unwrap()).abs() < 1e-6
    );
    let gradients = loss.backward();
    assert_values(
        Tensor::from_inner(model.projection.weight.val().grad(&gradients).unwrap()),
        &fixture,
        "projection_gradient",
    );
    assert_values(
        Tensor::from_inner(
            model.encoder.layers[0]
                .self_attn
                .query
                .weight
                .val()
                .grad(&gradients)
                .unwrap(),
        ),
        &fixture,
        "encoder_query_gradient",
    );
    assert!(model.src_embed.weight.val().grad(&gradients).is_none());
    assert!(model.tgt_embed.weight.val().grad(&gradients).is_none());
    let gradients = GradientsParams::from_grads(gradients, &model);
    let mut optimizer = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.98)
        .with_epsilon(1e-9)
        .init();
    model = optimizer.step(0.0001, model, gradients);
    assert_values(
        model.project(model.forward(src, tgt, mask)),
        &fixture,
        "after_step",
    );
}

#[test]
fn normalization_has_finite_gradients_for_constant_inputs() {
    let device = Default::default();
    let norm = ReferenceLayerNorm::<B>::new(4, LAYER_NORM_EPS, &device);
    let input = Tensor::<B, 3>::from_floats([[[0.5; 4]]], &device).require_grad();
    let normalized = norm.forward(input.clone());
    let loss = normalized * Tensor::from_floats([[[1.0, 0.0, 0.0, 0.0]]], &device);
    let gradients = loss.sum().backward();
    let values = input
        .grad(&gradients)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let expected = [750000.0, -250000.0, -250000.0, -250000.0];
    for (value, expected) in values.into_iter().zip(expected) {
        assert!((value - expected).abs() < 1.0);
    }
}
