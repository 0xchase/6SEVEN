#![cfg(feature = "ml")]

use tga::{
    AlgorithmId, AlgorithmSpec, Feedback, GenerationState, Observation, SixGan,
    SixGanClassification,
};

#[test]
fn sixgan_multiclass_training_and_feedback_resume_through_registry() {
    let registry = tga::builtin_registry();
    let config = SixGan {
        emb_dim: 3,
        hidden_dim: 4,
        discriminator_emb_dim: 2,
        discriminator_filters: 2,
        batch_size: 1,
        generator_pretrain_steps: 1,
        discriminator_pretrain_steps: 1,
        adversarial_rounds: 2,
        generator_steps: 2,
        discriminator_steps: 2,
        rollout_num: 2,
        classification: SixGanClassification::RfcBased,
        feedback_budget: Some(7),
        calibration_samples: 2,
        ..Default::default()
    };
    let observations = ["2001:db8::1", "2001:db8::250:56ff:fe89:49be"].map(|ip| Observation {
        address: ip.parse::<std::net::Ipv6Addr>().unwrap().octets(),
        active: true,
    });
    let mut artifact = registry
        .train(
            &AlgorithmSpec {
                algorithm: AlgorithmId::new("sixgan").unwrap(),
                config: serde_json::to_value(config).unwrap(),
            },
            &observations,
        )
        .unwrap();
    let mut model = registry.open(&artifact).unwrap();
    let mut first = [[0; 16]; 1];
    assert_eq!(
        model.generate(&mut first).unwrap().state,
        GenerationState::Ready
    );
    model
        .apply_feedback(&[Feedback::Active(first[0].into())])
        .unwrap();
    registry.save_model(&mut artifact, model.as_ref()).unwrap();
    let mut restored = registry.open(&artifact).unwrap();
    let mut original_output = [[0; 16]; 10];
    let mut restored_output = [[0; 16]; 10];
    let original_batch = model.generate(&mut original_output).unwrap();
    let restored_batch = restored.generate(&mut restored_output).unwrap();
    assert_eq!(original_batch, restored_batch);
    assert_eq!(restored_batch.written, 3);
    assert_eq!(restored_batch.state, GenerationState::AwaitingFeedback);
    assert_eq!(&original_output[..3], &restored_output[..3]);
    let mut feedback = restored_output[..3]
        .iter()
        .map(|address| Feedback::Active((*address).into()))
        .collect::<Vec<_>>();
    feedback.push(Feedback::BatchComplete);
    restored.apply_feedback(&feedback).unwrap();
    registry
        .save_model(&mut artifact, restored.as_ref())
        .unwrap();
    let mut allocated = registry.open(&artifact).unwrap();
    let batch = allocated.generate(&mut restored_output).unwrap();
    assert_eq!(batch.written, 7);
    assert_eq!(batch.state, GenerationState::Exhausted);
    assert_eq!(allocated.generate(&mut restored_output).unwrap().written, 0);
}
