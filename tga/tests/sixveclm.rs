#![cfg(feature = "ml")]

use std::collections::HashSet;
use tga::{AlgorithmId, AlgorithmSpec, Feedback, GenerationState, Observation, TgaError};

#[test]
fn registry_preserves_offline_training_and_finite_generation() {
    let registry = tga::builtin_registry();
    let spec = AlgorithmSpec {
        algorithm: AlgorithmId::new("sixveclm").unwrap(),
        config: serde_json::json!({
            "embedding_dim": 4, "heads": 2, "layers": 1, "ff_dim": 8,
            "embedding_epochs": 2, "transformer_epochs": 1, "dropout": 0.0
        }),
    };
    let seeds: Vec<_> = ["2001:db8:1::1", "2001:db8:2::2", "2001:db8:3::3"]
        .into_iter()
        .map(|address| Observation {
            address: address.parse::<std::net::Ipv6Addr>().unwrap().octets(),
            active: true,
        })
        .collect();
    let artifact = registry.train(&spec, &seeds).unwrap();
    assert_eq!(artifact.model_version, 2);
    let mut model = registry.open(&artifact).unwrap();
    let mut output = [[0; 16]; 10];
    let generated = model.generate(&mut output).unwrap();
    assert_eq!(generated.state, GenerationState::Exhausted);
    assert_eq!(generated.written, seeds.len());
    assert_eq!(
        output[..generated.written]
            .iter()
            .collect::<HashSet<_>>()
            .len(),
        seeds.len()
    );
    for (generated, seed) in output.iter().zip(&seeds) {
        assert_eq!(generated[..8], seed.address[..8]);
        assert_eq!(generated[8] >> 4, seed.address[8] >> 4);
    }
    assert!(matches!(
        model.apply_feedback(&[Feedback::BatchComplete]),
        Err(TgaError::Unsupported(_))
    ));
    assert_eq!(model.generate(&mut output).unwrap().written, 0);
    let mut reopened = registry.open(&artifact).unwrap();
    let mut again = [[0; 16]; 10];
    assert_eq!(reopened.generate(&mut again).unwrap(), generated);
    assert_eq!(again, output);
    let mut obsolete = artifact;
    obsolete.model_version = 1;
    assert!(registry.open(&obsolete).is_err());
}
