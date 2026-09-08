use tga::{AlgorithmId, AlgorithmSpec, Feedback, GenerationState, Observation};

#[test]
fn default_prescan_ranks_regions_through_the_registry_and_survives_reload() {
    let registry = tga::builtin_registry();
    let observations: Vec<_> = ["2001:db8::", "2001:db8::ffff"]
        .into_iter()
        .map(|ip| Observation {
            address: ip.parse::<std::net::Ipv6Addr>().unwrap().octets(),
            active: true,
        })
        .collect();
    let mut artifact = registry
        .train(
            &AlgorithmSpec {
                algorithm: AlgorithmId::new("6forest").unwrap(),
                config: serde_json::json!({}),
            },
            &observations,
        )
        .unwrap();
    let mut model = registry.open(&artifact).unwrap();
    let mut output = [[0; 16]; 128];
    let batch = model.generate(&mut output).unwrap();
    assert_eq!(batch.written, 100);
    assert_eq!(batch.state, GenerationState::AwaitingFeedback);
    let samples = output[..batch.written].to_vec();
    model
        .apply_feedback(&[Feedback::Active(samples[0].into())])
        .unwrap();
    registry.save_model(&mut artifact, model.as_ref()).unwrap();
    let mut restored = registry.open(&artifact).unwrap();
    assert_eq!(
        restored.generate(&mut output).unwrap().state,
        GenerationState::AwaitingFeedback
    );
    for model in [&mut model, &mut restored] {
        model.apply_feedback(&[Feedback::BatchComplete]).unwrap();
    }
    let mut expected = output;
    let batch = model.generate(&mut expected).unwrap();
    let resumed = restored.generate(&mut output).unwrap();
    assert_eq!(batch, resumed);
    assert_eq!(expected, output);
    assert_eq!(resumed.written, 128);
    assert!(output.iter().all(|address| !samples.contains(address)));
}

#[test]
fn offline_generation_remains_an_explicit_option() {
    use tga::{Algorithm, SixForest, TargetModel};
    let observations: Vec<_> = ["2001:db8::", "2001:db8::ffff"]
        .into_iter()
        .map(|ip| Observation {
            address: ip.parse::<std::net::Ipv6Addr>().unwrap().octets(),
            active: true,
        })
        .collect();
    let mut model = SixForest {
        prescan_samples: 0,
        ..Default::default()
    }
    .train(&observations)
    .unwrap();
    let batch = model.generate(&mut [[0; 16]; 128]).unwrap();
    assert_eq!(batch.written, 128);
    assert_eq!(batch.state, GenerationState::Ready);
}

#[test]
fn generation_budget_reaches_the_registered_model() {
    let registry = tga::builtin_registry();
    let observations: Vec<_> = ["2001:db8::", "2001:db8::ffff"]
        .into_iter()
        .map(|ip| Observation {
            address: ip.parse::<std::net::Ipv6Addr>().unwrap().octets(),
            active: true,
        })
        .collect();
    let artifact = registry
        .train(
            &AlgorithmSpec {
                algorithm: AlgorithmId::new("6forest").unwrap(),
                config: serde_json::json!({}),
            },
            &observations,
        )
        .unwrap();
    let mut model = registry.open(&artifact).unwrap();
    let mut sampled = Vec::new();
    let result = tga::generation::generate(
        model.as_mut(),
        tga::GenerationOptions {
            count: 1001,
            max_attempts: 1001,
            unique: true,
            exclude: Vec::new(),
        },
        |ip| {
            sampled.push(ip);
            Ok(())
        },
    );
    assert!(
        matches!(result, Err(tga::TgaError::Generation(message)) if message.contains("AwaitingFeedback"))
    );
    assert_eq!(sampled.len(), 10);
}
