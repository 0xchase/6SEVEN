use super::*;
use burn::module::Param;

type Cpu = burn::backend::ndarray::NdArray<f32>;
type Autodiff = burn::backend::Autodiff<Cpu>;

#[test]
fn context_pairs_match_paper_figure_two() {
    let row = std::array::from_fn(|index| index as u16 + 1);
    let storage = [row, row];
    let corpus = TokenCorpus::new(&storage);
    let pairs: Vec<_> = training_pairs(&corpus, 5).collect();
    assert_eq!(
        &pairs[..9],
        &[
            (1, 2),
            (1, 3),
            (2, 1),
            (2, 3),
            (2, 4),
            (3, 1),
            (3, 2),
            (3, 4),
            (3, 5)
        ]
    );
    assert_eq!(&pairs[120..124], &[(32, 30), (32, 31), (1, 2), (1, 3)]);
    assert_eq!(pairs.len(), 244);
}

#[test]
fn vocabulary_keeps_single_occurrence_words_and_excludes_padding() {
    let vocabulary = Vocabulary::from_counts(&[0, 10, 1, 0, 5]);
    assert_eq!(vocabulary.tokens, vec![1, 2, 4]);
    assert_eq!(vocabulary.indices[2], 1);
    assert_eq!(vocabulary.indices[0], usize::MAX);
}

#[test]
fn skip_gram_loss_and_gradients_match_full_softmax() {
    let _rng = crate::ml::rng_guard();
    let device = Default::default();
    let mut model = SkipGram::<Autodiff>::new(3, 2, &device);
    model.input.weight = Param::from_tensor(Tensor::from_floats(
        [[0.2, -0.1], [0.4, 0.3], [-0.2, 0.5]],
        &device,
    ));
    model.output.weight = Param::from_tensor(Tensor::from_floats(
        [[0.1, 0.2], [-0.3, 0.4], [0.5, -0.2]],
        &device,
    ));
    let loss = model.loss(
        Tensor::from_ints([[0]], &device),
        Tensor::from_ints([[1]], &device),
    );
    let logits = [0.0_f64, -0.1, 0.12];
    let total: f64 = logits.iter().map(|value| value.exp()).sum();
    let expected = total.ln() - logits[1];
    assert!((f64::from(loss.clone().into_scalar()) - expected).abs() < 1e-6);
    let gradients = loss.backward();
    let input = model
        .input
        .weight
        .val()
        .grad(&gradients)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let output = model
        .output
        .weight
        .val()
        .grad(&gradients)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let coefficients: Vec<_> = logits
        .iter()
        .enumerate()
        .map(|(index, value)| value.exp() / total - f64::from(index == 1))
        .collect();
    let expected_input = [
        coefficients[0] * 0.1 - coefficients[1] * 0.3 + coefficients[2] * 0.5,
        coefficients[0] * 0.2 + coefficients[1] * 0.4 - coefficients[2] * 0.2,
    ];
    for (actual, expected) in input[..2].iter().zip(expected_input) {
        assert!((f64::from(*actual) - expected).abs() < 1e-6);
    }
    assert_eq!(&input[2..], &[0.0; 4]);
    for (row, coefficient) in output.chunks_exact(2).zip(coefficients) {
        assert!((f64::from(row[0]) - coefficient * 0.2).abs() < 1e-6);
        assert!((f64::from(row[1]) + coefficient * 0.1).abs() < 1e-6);
    }
}

#[test]
fn learning_rate_reaches_both_endpoints() {
    let config = Word2VecConfig {
        embedding_dim: 4,
        window_size: 5,
        epochs: 2,
        batch_size: 3,
        learning_rate: 0.025,
        min_learning_rate: 0.0001,
        seed: 1,
    };
    assert_eq!(learning_rate(&config, 0, 20), 0.025);
    assert!((learning_rate(&config, 19, 20) - 0.0001).abs() < 1e-12);
}
