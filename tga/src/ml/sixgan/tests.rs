use super::*;
use crate::TargetModel;

fn config() -> SixGan {
    SixGan {
        emb_dim: 4,
        hidden_dim: 4,
        discriminator_emb_dim: 4,
        discriminator_filters: 2,
        batch_size: 2,
        generator_pretrain_steps: 1,
        discriminator_pretrain_steps: 0,
        adversarial_rounds: 0,
        rollout_num: 1,
        classification: SixGanClassification::None,
        ..Default::default()
    }
}

fn observations() -> Vec<Observation> {
    ["2001:db8::1", "2001:db8::2"]
        .iter()
        .map(|text| Observation {
            address: text.parse::<std::net::Ipv6Addr>().unwrap().octets(),
            active: true,
        })
        .collect()
}

fn take(model: &mut SixGanModel, count: usize) -> Vec<Address> {
    let mut output = vec![[0; 16]; count];
    assert_eq!(model.generate(&mut output).unwrap().written, count);
    output
}

#[test]
fn generation_is_chunk_invariant_and_serialization_replays() {
    let mut model = config().train(&observations()).unwrap();
    let bytes = bincode::serialize(&model).unwrap();
    let mut restored: SixGanModel = bincode::deserialize(&bytes).unwrap();
    let expected = take(&mut model, 7);
    let mut actual = take(&mut restored, 1);
    actual.extend(take(&mut restored, 3));
    actual.extend(take(&mut restored, 3));
    assert_eq!(actual, expected);
    let mut cloned = model.clone();
    assert_eq!(take(&mut model, 5), take(&mut cloned, 5));
}

#[test]
fn invalid_configuration_returns_errors() {
    for cfg in [
        SixGan {
            emb_dim: 0,
            ..config()
        },
        SixGan {
            hidden_dim: 0,
            ..config()
        },
        SixGan {
            entropy_k: 0,
            ..config()
        },
        SixGan {
            batch_size: 0,
            ..config()
        },
        SixGan {
            temperature: f32::NAN,
            ..config()
        },
        SixGan {
            aliased_prefixes: vec!["2001:db8::/33".parse().unwrap()],
            ..config()
        },
    ] {
        assert!(cfg.train(&observations()).is_err());
    }
}

#[test]
fn empty_and_corrupt_models_return_errors() {
    assert!(SixGanModel::default().generate(&mut [[0; 16]]).is_err());
    let mut model = config().train(&observations()).unwrap();
    model.generators[0].bytes = vec![1, 2, 3];
    assert!(model.generate(&mut [[0; 16]]).is_err());
}

#[test]
fn inactive_observations_and_capped_seeds_do_not_affect_training() {
    let cfg = SixGan {
        total_generation: 2,
        ..config()
    };
    let mut seeds = observations();
    let mut expected = cfg.train(&seeds).unwrap();
    seeds.push(Observation {
        address: [255; 16],
        active: true,
    });
    seeds.insert(
        0,
        Observation {
            address: [0; 16],
            active: false,
        },
    );
    let mut actual = cfg.train(&seeds).unwrap();
    assert_eq!(take(&mut expected, 16), take(&mut actual, 16));
}

#[test]
fn adversarial_training_runs_all_phases() {
    let cfg = SixGan {
        discriminator_pretrain_steps: 1,
        adversarial_rounds: 1,
        generator_steps: 1,
        ..config()
    };
    let mut model = cfg.train(&observations()).unwrap();
    assert_eq!(take(&mut model, 2).len(), 2);
}

#[test]
fn legacy_model_versions_require_retraining() {
    assert!(SixGan::decode_model(1, &[]).is_err());
}

#[test]
fn cli_and_serde_defaults_agree() {
    use clap::Parser;
    #[derive(Parser)]
    struct Command {
        #[command(flatten)]
        config: SixGan,
    }
    let parsed = Command::try_parse_from(["sixgan"]).unwrap().config;
    let json = serde_json::to_value(SixGan::default()).unwrap();
    assert_eq!(serde_json::to_value(parsed).unwrap(), json);
    let missing: SixGan = serde_json::from_str("{}").unwrap();
    assert_eq!(serde_json::to_value(missing).unwrap(), json);
}

#[test]
fn inconsistent_artifact_dimensions_return_an_error() {
    let mut model = config().train(&observations()).unwrap();
    model.hidden_dim += 1;
    assert!(model.generate(&mut [[0; 16]]).is_err());
}

#[test]
fn feedback_budget_waits_allocates_and_survives_reload() {
    use crate::{Feedback, GenerationState};
    let cfg = SixGan {
        feedback_budget: Some(5),
        calibration_samples: 3,
        ..config()
    };
    let mut model = cfg.train(&observations()).unwrap();
    let first = take(&mut model, 1);
    let mut restored: SixGanModel =
        bincode::deserialize(&bincode::serialize(&model).unwrap()).unwrap();
    assert_eq!(take(&mut restored, 2), take(&mut model, 2));
    let mut remaining = [[0; 16]; 10];
    let batch = restored.generate(&mut remaining).unwrap();
    assert_eq!(batch.written, 0);
    assert_eq!(batch.state, GenerationState::AwaitingFeedback);
    let mut replay = cfg.train(&observations()).unwrap();
    let calibration = take(&mut replay, 3);
    assert_eq!(calibration[0], first[0]);
    let mut outcomes = calibration
        .iter()
        .map(|address| Feedback::Active((*address).into()))
        .collect::<Vec<_>>();
    outcomes.push(Feedback::BatchComplete);
    restored.apply_feedback(&outcomes).unwrap();
    let _ = take(&mut restored, 1);
    let mut clone: SixGanModel =
        bincode::deserialize(&bincode::serialize(&restored).unwrap()).unwrap();
    let batch = restored.generate(&mut remaining).unwrap();
    assert_eq!(batch.written, 4);
    assert_eq!(batch.state, GenerationState::Exhausted);
    assert_eq!(take(&mut clone, 4), remaining[..4]);
    assert_eq!(restored.generate(&mut remaining).unwrap().written, 0);
}

#[test]
fn v2_models_migrate_without_changing_generation() {
    let mut model = config().train(&observations()).unwrap();
    let bytes = bincode::serialize(&(
        &model.generators,
        model.emb_dim,
        model.hidden_dim,
        model.generation_batch_size,
        model.generation_temperature,
        model.sampling_seed,
        model.classification,
    ))
    .unwrap();
    let mut migrated = SixGan::decode_model(2, &bytes).unwrap();
    assert_eq!(take(&mut model, 5), take(&mut migrated, 5));
}
