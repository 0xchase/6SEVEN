use tga::{AlgorithmId, AlgorithmSpec, Feedback, GenerationState, Observation};

#[test]
fn registry_preserves_probe_policy_aliases_and_pending_rounds() {
    let registry = tga::builtin_registry();
    let mut artifact = registry
        .train(
            &AlgorithmSpec {
                algorithm: AlgorithmId::new("6tree").unwrap(),
                config: serde_json::json!({}),
            },
            &[Observation {
                address: 1u128.to_be_bytes(),
                active: true,
            }],
        )
        .unwrap();
    let mut model = registry.open(&artifact).unwrap();
    assert!(model.allows_repeated_probes());
    for round in 0..6 {
        let mut targets = Vec::new();
        loop {
            let mut output = [[0; 16]; 23];
            let batch = model.generate(&mut output).unwrap();
            targets.extend_from_slice(&output[..batch.written]);
            if batch.state == GenerationState::AwaitingFeedback {
                break;
            }
            assert_ne!(batch.state, GenerationState::Exhausted);
        }
        assert!(!targets.is_empty());
        if round < 2 {
            assert_eq!(targets.len(), [16, 240][round]);
        } else {
            assert!(targets.len() <= 160);
        }
        let feedback: Vec<_> = targets
            .into_iter()
            .enumerate()
            .map(|(index, address)| {
                if round < 2 || (round < 5 && index == 0) {
                    Feedback::Active(address.into())
                } else {
                    Feedback::Inactive(address.into())
                }
            })
            .collect();
        model.apply_feedback(&feedback[..1]).unwrap();
        registry.save_model(&mut artifact, model.as_ref()).unwrap();
        model = registry.open(&artifact).unwrap();
        assert!(model.allows_repeated_probes());
        for chunk in feedback[1..].chunks(19) {
            model.apply_feedback(chunk).unwrap();
        }
    }
    assert_eq!(
        model.detected_aliases(),
        &["::/104".parse::<tga::Ipv6Prefix>().unwrap()]
    );
    registry.save_model(&mut artifact, model.as_ref()).unwrap();
    let restored = registry.open(&artifact).unwrap();
    assert_eq!(restored.detected_aliases(), model.detected_aliases());
}

#[test]
fn disabling_detection_retains_unique_probe_policy_through_the_registry() {
    let registry = tga::builtin_registry();
    let artifact = registry
        .train(
            &AlgorithmSpec {
                algorithm: AlgorithmId::new("6tree").unwrap(),
                config: serde_json::json!({"alias_detection":false}),
            },
            &[Observation {
                address: 1u128.to_be_bytes(),
                active: true,
            }],
        )
        .unwrap();
    assert!(!registry.open(&artifact).unwrap().allows_repeated_probes());
}
