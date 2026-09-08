use std::collections::HashSet;
use tga::{AlgorithmId, AlgorithmSpec, Feedback, GenerationState, Observation};

fn exercise(id: &str, config: serde_json::Value) {
    let registry = tga::builtin_registry();
    let seeds = [1, 2, 3].map(|last| {
        let address = format!("2001:db8::{last}")
            .parse::<std::net::Ipv6Addr>()
            .unwrap()
            .octets();
        Observation {
            address,
            active: true,
        }
    });
    let initial = registry
        .train(
            &AlgorithmSpec {
                algorithm: AlgorithmId::new(id).unwrap(),
                config,
            },
            &seeds,
        )
        .unwrap();
    let mut live = registry.open(&initial).unwrap();
    let mut replay = registry.open(&initial).unwrap();
    let mut observed = HashSet::new();
    let mut waiting = false;
    for _ in 0..8 {
        let mut output = [[0; 16]; 5];
        let batch = live.generate(&mut output).unwrap();
        let addresses = &output[..batch.written];
        for address in addresses {
            assert!(observed.insert(*address));
        }
        let mut feedback: Vec<_> = addresses
            .iter()
            .map(|address| Feedback::Active((*address).into()))
            .collect();
        if batch.state == GenerationState::AwaitingFeedback {
            feedback.push(Feedback::BatchComplete);
            waiting = true;
        }
        live.apply_feedback(&feedback).unwrap();
        replay.apply_feedback(&feedback).unwrap();
        if waiting {
            break;
        }
    }
    assert_eq!(observed.len(), 16);
    assert!(waiting);
    let mut live_artifact = initial.clone();
    let mut replay_artifact = initial;
    registry
        .save_model(&mut live_artifact, live.as_ref())
        .unwrap();
    registry
        .save_model(&mut replay_artifact, replay.as_ref())
        .unwrap();
    let mut restored = registry.open(&live_artifact).unwrap();
    let mut replayed = registry.open(&replay_artifact).unwrap();
    let mut output = [[0; 16]; 10];
    let next = restored.generate(&mut output).unwrap();
    assert!(next.written > 0);
    assert!(
        output[..next.written]
            .iter()
            .all(|address| !observed.contains(address))
    );
    let mut replay_output = [[0; 16]; 10];
    let replay_next = replayed.generate(&mut replay_output).unwrap();
    assert_eq!(next, replay_next);
    assert_eq!(output[..next.written], replay_output[..replay_next.written]);
}

#[test]
fn det_incremental_feedback_survives_save_and_replay() {
    exercise("det", serde_json::json!({"delta_base": 16, "leaf_max": 16}));
}

#[test]
fn sixtree_incremental_feedback_survives_save_and_replay() {
    exercise("6tree", serde_json::json!({}));
}

#[test]
fn saving_a_different_algorithm_preserves_the_artifact() {
    let registry = tga::builtin_registry();
    let seeds = [Observation {
        address: std::net::Ipv6Addr::LOCALHOST.octets(),
        active: true,
    }];
    let mut artifact = registry
        .train(
            &AlgorithmSpec {
                algorithm: AlgorithmId::new("det").unwrap(),
                config: serde_json::json!({"delta_base": 16, "leaf_max": 16}),
            },
            &seeds,
        )
        .unwrap();
    let other = registry
        .train(
            &AlgorithmSpec {
                algorithm: AlgorithmId::new("6tree").unwrap(),
                config: serde_json::json!({}),
            },
            &seeds,
        )
        .unwrap();
    let model = registry.open(&other).unwrap();
    let original = artifact.clone();
    assert!(registry.save_model(&mut artifact, model.as_ref()).is_err());
    assert_eq!(artifact, original);
}
