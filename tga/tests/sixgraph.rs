use tga::{AlgorithmId, AlgorithmSpec, Feedback, GenerationState, Observation};

#[test]
fn registry_saves_and_resumes_sixgraph_feedback_rounds() {
    let registry = tga::builtin_registry();
    let seeds = [1u128, 2].map(|ip| Observation {
        address: ip.to_be_bytes(),
        active: true,
    });
    let mut artifact = registry
        .train(
            &AlgorithmSpec {
                algorithm: AlgorithmId::new("6graph").unwrap(),
                config: serde_json::json!({}),
            },
            &seeds,
        )
        .unwrap();
    assert_eq!(artifact.model_version, 2);
    let mut model = registry.open(&artifact).unwrap();
    let mut seen = std::collections::HashSet::new();
    loop {
        let mut output = [[0; 16]; 32];
        let batch = model.generate(&mut output).unwrap();
        for &address in &output[..batch.written] {
            assert!(seen.insert(address));
        }
        if batch.state == GenerationState::Exhausted {
            break;
        }
        assert_eq!(batch.state, GenerationState::AwaitingFeedback);
        let mut feedback = output[..batch.written]
            .iter()
            .map(|&address| Feedback::Inactive(address.into()))
            .collect::<Vec<_>>();
        feedback.push(Feedback::BatchComplete);
        model.apply_feedback(&feedback).unwrap();
        registry.save_model(&mut artifact, model.as_ref()).unwrap();
        model = registry.open(&artifact).unwrap();
    }
    assert_eq!(seen.len(), 14);
    assert!(!seen.contains(&1u128.to_be_bytes()));
    assert!(!seen.contains(&2u128.to_be_bytes()));
}
