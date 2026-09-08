use super::generate::{decode_latent_batch, fill_standard_normal_latents, uniform_open01};
use super::*;
use clap::Parser;
use rand::{SeedableRng, rngs::StdRng};

#[derive(Parser)]
struct ParseTga {
    #[command(subcommand)]
    algorithm: crate::AlgorithmConfig,
}

fn model_with_fresh_decoder() -> SixGcvaeModel {
    let _rng = crate::ml::rng_guard();
    let device = crate::ml::infer_device();
    let decoder = GcvaeDecoder::<InferBackend>::new(&device);
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let decoder_bytes = recorder.record(decoder.into_record(), ()).unwrap();
    SixGcvaeModel {
        generation: Default::default(),
        generation_seed: 0,
        components: vec![Component {
            label: "all".into(),
            emitted: 0,
            decoder: StoredDecoder {
                record_bytes: decoder_bytes,
                final_loss: 0.0,
            },
        }],
        allocation: None,
    }
}

#[test]
fn generation_continues_across_calls_clones_and_reloads() {
    use crate::TargetModel;
    let mut model = model_with_fresh_decoder();
    model.generation.0 = Some(
        model
            .build_streams()
            .unwrap()
            .into_iter()
            .map(|stream| stream.with_batch_size(4))
            .collect(),
    );
    let encoded = bincode::serialize(&model).unwrap();
    let mut restored: SixGcvaeModel = bincode::deserialize(&encoded).unwrap();
    restored.generation.0 = Some(
        restored
            .build_streams()
            .unwrap()
            .into_iter()
            .map(|stream| stream.with_batch_size(4))
            .collect(),
    );
    let mut first = vec![[0; 16]; 3];
    model.generate(&mut first).unwrap();
    let mut cloned = model.clone();
    let mut next = vec![[0; 16]; 4];
    model.generate(&mut next).unwrap();
    let mut cloned_next = next.clone();
    cloned.generate(&mut cloned_next).unwrap();
    assert_eq!(next, cloned_next);
    let mut combined = vec![[0; 16]; first.len() + next.len()];
    restored.generate(&mut combined).unwrap();
    first.extend(next);
    assert_eq!(combined, first);
    // Persistence restarts the seeded stream, including after consumption.
    let mut reloaded: SixGcvaeModel =
        bincode::deserialize(&bincode::serialize(&model).unwrap()).unwrap();
    reloaded.generation.0 = Some(
        reloaded
            .build_streams()
            .unwrap()
            .into_iter()
            .map(|stream| stream.with_batch_size(4))
            .collect(),
    );
    let mut restarted = [[0; 16]; 1];
    reloaded.generate(&mut restarted).unwrap();
    assert_eq!(restarted[0], first[0]);
}

#[test]
fn empty_output_is_rejected_before_loading_weights() {
    use crate::TargetModel;
    let mut model = SixGcvaeModel::default();
    let error = model.generate(&mut []).unwrap_err();
    assert!(!error.to_string().contains("decoder"));
    assert!(model.generation.0.is_none());
}

#[test]
fn training_requires_two_active_observations() {
    let observations = [Observation {
        address: [0; 16],
        active: false,
    }];
    assert!(SixGcvae::default().train(&observations).is_err());
}

#[test]
fn six_gcvae_defaults_match_reference() {
    let parsed = ParseTga::try_parse_from(["bin", "six-gcvae"]).unwrap();
    let crate::AlgorithmConfig::SixGcvae(cfg) = parsed.algorithm else {
        panic!("wrong tga parsed");
    };
    assert_eq!(cfg.epochs, 3);
    assert_eq!(cfg.batch_size, 64);
}

#[test]
fn six_gcvae_rejects_removed_seed_flag() {
    let parsed = ParseTga::try_parse_from(["bin", "six-gcvae", "--seed", "42"]);
    assert!(parsed.is_err());
}

#[test]
fn untrained_model_fails_cleanly() {
    let model = SixGcvaeModel::default();
    assert!(model.stream().is_err());
}

#[test]
fn invalid_decoder_bytes_fail_cleanly() {
    let model = SixGcvaeModel {
        generation: Default::default(),
        generation_seed: 0,
        components: vec![Component {
            label: "all".into(),
            emitted: 0,
            decoder: StoredDecoder {
                record_bytes: vec![1, 2, 3],
                final_loss: 0.0,
            },
        }],
        allocation: None,
    };
    let device = crate::ml::infer_device();
    assert!(model.load_decoder(&device).is_err());
    assert!(model.stream().is_err());
}

#[test]
fn batched_generation_matches_single_item_decoding() {
    let model = model_with_fresh_decoder();
    let device = crate::ml::infer_device();
    let decoder = model.load_decoder(&device).unwrap();

    let latents = (0..8 * LATENT_DIM)
        .map(|index| ((index % 17) as f32 - 8.0) / 4.0)
        .collect::<Vec<_>>();
    let mut batched = Vec::new();
    decode_latent_batch(&decoder, &device, latents.clone(), 8, &mut batched).unwrap();

    let scalar = latents
        .chunks_exact(LATENT_DIM)
        .map(|latent| {
            let mut out = Vec::new();
            decode_latent_batch(&decoder, &device, latent.to_vec(), 1, &mut out).unwrap();
            out[0]
        })
        .collect::<Vec<_>>();

    assert_eq!(batched, scalar);
}

#[test]
fn stream_decodes_stochastic_latent_samples() {
    let model = model_with_fresh_decoder();
    let generated = model
        .stream()
        .unwrap()
        .take(8)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(generated.len(), 8);
}

#[test]
fn stochastic_generator_stream_is_unbounded() {
    let model = model_with_fresh_decoder();
    let generated = model
        .stream()
        .unwrap()
        .take(5)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(generated.len(), 5);
}

#[test]
fn latent_draws_follow_standard_normal_support() {
    let mut rng = StdRng::seed_from_u64(7);
    let mut samples = Vec::new();
    fill_standard_normal_latents(&mut rng, 64, &mut samples);

    assert!(samples.iter().any(|value| *value > 1.0));
    assert!(samples.iter().any(|value| *value < -1.0));
}

#[test]
fn uniform_sampler_stays_inside_open_unit_interval() {
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..1024 {
        let sample = uniform_open01(&mut rng);
        assert!(sample > 0.0, "sample should be greater than zero");
        assert!(sample < 1.0, "sample should be less than one");
    }
}

#[test]
fn classified_training_generation_and_feedback_form_complete_rounds() {
    let observations = [
        "2001:db8::1",
        "2001:db8::2",
        "2001:db8::211:22ff:fe33:4455",
        "2001:db8::211:22ff:fe33:4456",
    ]
    .map(|text| Observation {
        address: text.parse::<std::net::Ipv6Addr>().unwrap().octets(),
        active: true,
    });
    let config = SixGcvae {
        epochs: 1,
        classification: SixGcvaeClassification::Manual,
        feedback_samples: 2,
        ..SixGcvae::default()
    };
    let mut model = config.train(&observations).unwrap();
    assert_eq!(model.components.len(), 2);
    let weights_before = model
        .components
        .iter()
        .map(|component| component.decoder.record_bytes.clone())
        .collect::<Vec<_>>();
    model.generation.0 = Some(
        model
            .build_streams()
            .unwrap()
            .into_iter()
            .map(|stream| stream.with_batch_size(2))
            .collect(),
    );
    let mut output = [[0; 16]; 8];
    assert_eq!(
        model.generate(&mut output).unwrap(),
        Generated {
            written: 4,
            state: GenerationState::AwaitingFeedback
        }
    );
    assert_eq!(
        model.generate(&mut output).unwrap(),
        Generated {
            written: 0,
            state: GenerationState::AwaitingFeedback
        }
    );
    let feedback = [
        Feedback::Active(output[0].into()),
        Feedback::Inactive(output[1].into()),
    ];
    model.apply_feedback(&feedback).unwrap();
    assert_eq!(model.generate(&mut output).unwrap().written, 0);
    let bytes = SixGcvae::encode_model(&model).unwrap();
    let mut restored = SixGcvae::decode_model(SixGcvae::MODEL_VERSION, &bytes).unwrap();
    restored.apply_feedback(&[Feedback::BatchComplete]).unwrap();
    model.apply_feedback(&[Feedback::BatchComplete]).unwrap();
    assert_eq!(
        bincode::serialize(&model.allocation).unwrap(),
        bincode::serialize(&restored.allocation).unwrap()
    );
    restored.generation.0 = Some(
        restored
            .build_streams()
            .unwrap()
            .into_iter()
            .map(|stream| stream.with_batch_size(2))
            .collect(),
    );
    let mut restored_output = [[0; 16]; 2];
    restored.generate(&mut restored_output).unwrap();
    assert_eq!(
        model.generate(&mut output).unwrap(),
        Generated {
            written: 2,
            state: GenerationState::AwaitingFeedback
        }
    );
    assert_eq!(&output[..2], &restored_output);
    assert_eq!(
        weights_before,
        model
            .components
            .iter()
            .map(|component| component.decoder.record_bytes.clone())
            .collect::<Vec<_>>()
    );
}

#[test]
fn version_one_artifacts_migrate_without_changing_decoder_weights() {
    #[derive(Serialize)]
    struct Legacy {
        decoder: Option<StoredDecoder>,
        generation_seed: u64,
    }
    let model = model_with_fresh_decoder();
    let legacy = Legacy {
        decoder: Some(model.components[0].decoder.clone()),
        generation_seed: 42,
    };
    let migrated = SixGcvae::decode_model(1, &bincode::serialize(&legacy).unwrap()).unwrap();
    assert_eq!(migrated.generation_seed, 42);
    assert_eq!(
        migrated.components[0].decoder.record_bytes,
        model.components[0].decoder.record_bytes
    );
    assert!(migrated.allocation.is_none());
    assert!(SixGcvae::decode_model(99, &[]).is_err());
}

#[test]
fn unclassified_feedback_is_explicitly_unsupported() {
    assert!(matches!(
        SixGcvaeModel::default().apply_feedback(&[Feedback::BatchComplete]),
        Err(TgaError::Unsupported(_))
    ));
}

#[test]
fn classified_feedback_requires_generated_candidate_attribution() {
    let mut model = model_with_fresh_decoder();
    model.allocation = Some(Allocation::new([], 1, 2).unwrap());
    assert!(matches!(
        model.apply_feedback(&[Feedback::Active(1u128.into())]),
        Err(TgaError::Feedback(_))
    ));
}

#[test]
fn classification_options_are_available_through_cli_and_serde() {
    let parsed = ParseTga::try_parse_from([
        "bin",
        "six-gcvae",
        "--classification",
        "entropy",
        "--clusters",
        "3",
        "--feedback-samples",
        "100",
    ])
    .unwrap();
    let crate::AlgorithmConfig::SixGcvae(config) = parsed.algorithm else {
        panic!("wrong algorithm")
    };
    assert_eq!(config.classification, SixGcvaeClassification::Entropy);
    assert_eq!(config.clusters, 3);
    assert_eq!(config.feedback_samples, 100);
    let old: SixGcvae =
        serde_json::from_str(r#"{"epochs":3,"batch_size":64,"generation_seed":0}"#).unwrap();
    assert_eq!(old.classification, SixGcvaeClassification::None);
    assert_eq!(old.feedback_samples, 1_000_000);
}
