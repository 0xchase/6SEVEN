use super::*;
use std::net::Ipv6Addr;
use std::str::FromStr;

fn addr(input: &str) -> [u8; 16] {
    Ipv6Addr::from_str(input).unwrap().octets()
}

fn tiny_model() -> SixVecLmModel {
    let algorithm = SixVecLm {
        seed: 1,
        embedding_dim: 20,
        embedding_window: 5,
        embedding_epochs: 1,
        transformer_epochs: 1,
        embedding_batch_size: 32,
        transformer_batch_size: 4,
        layers: 1,
        heads: 4,
        ff_dim: 32,
        dropout: 0.0,
        embedding_lr: 0.025,
        embedding_min_lr: 0.0001,
        noam_factor: 1.0,
        noam_warmup: 1,
        generation_temperature: 0.015,
    };
    let seeds = vec![
        addr("2001:db8:1::1"),
        addr("2001:db8:2::2"),
        addr("2001:db8:3::3"),
        addr("2001:db8:4::4"),
        addr("2001:db8:5::5"),
        addr("2001:db8:6::6"),
        addr("2001:db8:7::7"),
        addr("2001:db8:8::8"),
    ];
    algorithm
        .train_inner(seeds)
        .expect("train tiny 6VecLM model")
}

fn generate_one(model: &SixVecLmModel, sample_id: usize) -> Address {
    model
        .generate_sample_batch(sample_id, 1, model.generation_temperature)
        .expect("single generate")
        .into_iter()
        .next()
        .expect("single generate produced an address")
}

fn collect_generated(mut model: SixVecLmModel, batch_size: usize) -> Vec<Address> {
    use crate::{GenerationState, TargetModel};
    let mut result = Vec::new();
    loop {
        let mut output = vec![[0; 16]; batch_size];
        let generated = model.generate(&mut output).unwrap();
        result.extend_from_slice(&output[..generated.written]);
        if generated.state == GenerationState::Exhausted {
            break;
        }
        assert!(generated.written > 0);
        assert_eq!(generated.state, GenerationState::Ready);
    }
    result
}

#[test]
fn tokenizer_roundtrip() {
    let tokenizer = IPv6Tokenizer::new();
    let address = addr("2001:db8:abcd::1");
    let tokens = tokenizer.encode_nybbles(&address);
    let decoded = tokenizer.detokenize_hex_stream(&tokens).unwrap();
    assert_eq!(decoded, address);
}

#[test]
fn tokenizer_rejects_position_mismatched_tokens() {
    let tokenizer = IPv6Tokenizer::new();
    let address = addr("2001:db8:abcd::1");
    let mut tokens = tokenizer.encode_nybbles(&address);
    tokens.swap(0, 1);

    assert!(tokenizer.detokenize_hex_stream(&tokens).is_none());
}

#[test]
fn low_temperature_sharpens_sampling_distribution() {
    let scores = [0.9, 0.1, -0.3];
    let cold = generation::temperature_softmax(&scores, 0.01);
    let warm = generation::temperature_softmax(&scores, 1.0);

    assert!(cold[0] > warm[0]);
    assert!((cold.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    assert!((warm.iter().sum::<f64>() - 1.0).abs() < 1e-9);
}

#[test]
fn validate_rejects_invalid_probability_like_parameters() {
    let mut algorithm = SixVecLm {
        seed: 1,
        embedding_dim: 100,
        embedding_window: 5,
        embedding_epochs: 5,
        transformer_epochs: 10,
        embedding_batch_size: DEFAULT_EMBEDDING_BATCH_SIZE,
        transformer_batch_size: DEFAULT_TRANSFORMER_BATCH_SIZE,
        layers: 6,
        heads: 10,
        ff_dim: 2048,
        dropout: 1.1,
        embedding_lr: 0.025,
        embedding_min_lr: 0.0001,
        noam_factor: 1.0,
        noam_warmup: 400,
        generation_temperature: 0.015,
    };

    assert!(algorithm.validate().is_err());
    algorithm.dropout = 0.1;
    algorithm.embedding_lr = f64::NAN;
    assert!(algorithm.validate().is_err());
}

#[test]
fn sequential_batch_generation_matches_single_samples() {
    let model = tiny_model();
    let len = model.seed_addresses.len();

    let sample_ids = (0..len).collect::<Vec<_>>();
    let singles: Vec<_> = sample_ids
        .iter()
        .copied()
        .map(|sample_id| generate_one(&model, sample_id))
        .collect();
    let batched = model
        .generate_sample_batch(0, len, model.generation_temperature)
        .expect("batch generate");

    assert_eq!(batched, singles);
}

#[test]
fn later_sample_batches_match_single_samples() {
    let model = tiny_model();
    let first_sample = 3;
    let sample_count = 4;

    let singles: Vec<_> = (first_sample..first_sample + sample_count)
        .map(|sample_id| generate_one(&model, sample_id))
        .collect();
    let batched = model
        .generate_sample_batch(first_sample, sample_count, model.generation_temperature)
        .expect("batch generate");

    assert_eq!(batched, singles);
}

#[test]
fn generation_rejects_sampling_beyond_the_seed_set() {
    let model = tiny_model();
    let source_rows = model.seed_addresses.len();
    let sample_count = source_rows + 3;

    let generated = model.generate_sample_batch(0, sample_count, model.generation_temperature);

    assert!(generated.is_err());
}

#[test]
fn target_model_matches_batch_generation() {
    let model = tiny_model();
    let len = model.seed_addresses.len();
    let expected = model
        .generate_sample_batch(0, len, model.generation_temperature)
        .expect("batch generate");
    let actual = collect_generated(model.clone(), 4);

    assert_eq!(actual, expected);
}

#[test]
fn target_model_is_batch_size_invariant() {
    let model = tiny_model();

    let expected = collect_generated(model.clone(), 1);
    let actual = collect_generated(model, 4);

    assert_eq!(actual, expected);
}

#[test]
fn temperature_sampling_respects_small_positive_values() {
    let probabilities = generation::temperature_softmax(&[0.0, 1e-7], 1e-9);
    assert!(probabilities[1] > 0.999999);
    let ties = generation::temperature_softmax(&[0.5, 0.5], f32::MIN_POSITIVE);
    assert_eq!(ties, vec![0.5, 0.5]);
}

#[test]
fn serialized_model_restarts_generation_and_clone_continues() {
    use crate::TargetModel;
    let mut model = tiny_model();
    let all = collect_generated(model.clone(), 4);
    let mut first = [[0; 16]; 2];
    let produced = model.generate(&mut first).unwrap();
    assert_eq!(&all[..produced.written], &first[..produced.written]);
    let cloned = model.clone();
    let bytes = bincode::serialize(&model).unwrap();
    let restored: SixVecLmModel = bincode::deserialize(&bytes).unwrap();
    assert_eq!(collect_generated(cloned, 3), all[produced.written..]);
    assert_eq!(collect_generated(restored, 1), all);
}

#[test]
fn invalid_artifacts_return_errors_before_inference() {
    let model = tiny_model();
    let mut invalid = model.clone();
    Arc::make_mut(&mut invalid.word_embeddings)[0] = f32::NAN;
    assert!(invalid.validate_ready_for_generation().is_err());
    let mut invalid = model.clone();
    Arc::make_mut(&mut invalid.transformer).hyper.n_layers = 0;
    assert!(invalid.validate_ready_for_generation().is_err());
    let mut invalid = model.clone();
    Arc::make_mut(&mut invalid.position_candidates)[17] = vec![1];
    assert!(invalid.validate_ready_for_generation().is_err());
    let mut invalid = model;
    invalid.generation_temperature = f32::NAN;
    assert!(
        invalid
            .generate_sample_batch(0, 1, invalid.generation_temperature)
            .is_err()
    );
}

#[test]
fn generation_preserves_conditioning_nybbles() {
    let model = tiny_model();
    let rows = model
        .generate_sample_batch(0, model.seed_addresses.len(), 0.5)
        .unwrap();
    for (index, generated) in rows.iter().enumerate() {
        let source = model.seed_addresses[index];
        assert_eq!(generated[..8], source[..8]);
        assert_eq!(generated[8] >> 4, source[8] >> 4);
    }
}

#[test]
fn empty_generation_output_is_rejected() {
    use crate::TargetModel;
    assert!(SixVecLmModel::default().generate(&mut []).is_err());
}

#[test]
fn generation_deduplicates_and_exhausts_without_discarding_seed_hits() {
    use crate::{GenerationState, TargetModel};
    let mut model = tiny_model();
    let seed = model.seed_addresses[0];
    model.seed_addresses = Arc::new(vec![seed; 8]);
    let tokens = model.tokenizer.encode_nybbles(&seed);
    for (position, &token) in tokens.iter().enumerate().skip(FIRST_SAMPLED_POSITION) {
        Arc::make_mut(&mut model.position_candidates)[position] = vec![token];
    }
    let split = model.clone();
    let mut output = [[0; 16]; 10];
    let generated = model.generate(&mut output).unwrap();
    assert_eq!(generated.written, 1);
    assert_eq!(generated.state, GenerationState::Exhausted);
    assert_eq!(output[0], seed);
    let end = model.generate(&mut output).unwrap();
    assert_eq!(end.written, 0);
    assert_eq!(end.state, GenerationState::Exhausted);
    assert_eq!(collect_generated(split, 1), vec![seed]);
}

#[test]
fn feedback_is_unsupported_and_cannot_change_the_model() {
    use crate::{Algorithm, Feedback, TargetModel};
    let mut model = tiny_model();
    let original = SixVecLm::encode_model(&model).unwrap();
    let expected = collect_generated(model.clone(), 4);
    let feedback = [
        Feedback::Active(Ipv6Addr::LOCALHOST),
        Feedback::Inactive(Ipv6Addr::UNSPECIFIED),
        Feedback::Skipped(Ipv6Addr::LOCALHOST),
        Feedback::Aliased("2001:db8::/32".parse().unwrap()),
        Feedback::BatchComplete,
    ];
    for item in feedback {
        assert!(matches!(
            model.apply_feedback(&[item]),
            Err(TgaError::Unsupported(_))
        ));
        assert_eq!(SixVecLm::encode_model(&model).unwrap(), original);
    }
    assert!(matches!(
        model.apply_feedback(&[]),
        Err(TgaError::Unsupported(_))
    ));
    assert_eq!(collect_generated(model, 1), expected);
}

#[test]
fn old_training_artifacts_require_retraining() {
    use crate::Algorithm;
    assert!(SixVecLm::decode_model(1, &[]).is_err());
}

#[test]
fn cli_and_json_defaults_agree_with_paper_settings() {
    use clap::Parser;
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        config: SixVecLm,
    }
    let cli = Cli::try_parse_from(["sixveclm"]).unwrap().config;
    let json: SixVecLm = serde_json::from_str("{}").unwrap();
    assert_eq!(
        serde_json::to_value(&cli).unwrap(),
        serde_json::to_value(&json).unwrap()
    );
    assert_eq!(
        (
            json.embedding_dim,
            json.embedding_window,
            json.layers,
            json.heads
        ),
        (100, 5, 6, 10)
    );
    assert_eq!(json.generation_temperature, 0.01);
    json.validate().unwrap();
    assert!(serde_json::from_str::<SixVecLm>(r#"{"negative_samples":5}"#).is_err());
    assert!(Cli::try_parse_from(["sixveclm", "--negative-samples", "5"]).is_err());
}

#[test]
fn training_uses_all_words_from_active_observations() {
    use crate::{Algorithm, Observation, TargetModel};
    let config = SixVecLm {
        embedding_dim: 4,
        heads: 2,
        layers: 1,
        ff_dim: 8,
        embedding_epochs: 1,
        transformer_epochs: 1,
        dropout: 0.0,
        ..SixVecLm::default()
    };
    let active = addr("2001:db8::abcd");
    let observations = [
        Observation {
            address: active,
            active: true,
        },
        Observation {
            address: addr("ffff::1234"),
            active: false,
        },
    ];
    let mut model = config.train(&observations).unwrap();
    assert_eq!(*model.seed_addresses, vec![active]);
    let tokens = model.tokenizer.encode_nybbles(&active);
    for (position, token) in tokens.iter().enumerate().skip(FIRST_SAMPLED_POSITION) {
        assert_eq!(model.position_candidates[position], vec![*token]);
    }
    let mut output = [[0; 16]; 2];
    let generated = model.generate(&mut output).unwrap();
    assert_eq!(generated.written, 1);
    assert_eq!(generated.state, crate::GenerationState::Exhausted);
    assert_eq!(output[0], active);
    assert!(config.train(&observations[1..]).is_err());
}

#[test]
fn changing_temperature_restarts_sampling_without_retraining() {
    use crate::TargetModel;
    let mut model = tiny_model();
    model.generate(&mut [[0; 16]; 3]).unwrap();
    let before = collect_generated(model.clone(), 4);
    assert!(model.set_generation_temperature(f32::NAN).is_err());
    assert_eq!(collect_generated(model.clone(), 4), before);
    let weights = model.transformer.bytes.clone();
    model.set_generation_temperature(0.5).unwrap();
    assert_eq!(model.transformer.bytes, weights);
    let expected = model
        .generate_sample_batch(0, model.seed_addresses.len(), 0.5)
        .unwrap();
    assert_eq!(collect_generated(model, 3), expected);
}
