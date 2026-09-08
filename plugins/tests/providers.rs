use sixseven_core::{AlgorithmId, AlgorithmSpec, Feedback, GenerationState, Registry};
use sixseven_plugins::{PluginConfig, register};
use std::path::PathBuf;

fn exercise(config: PluginConfig, id: &str) {
    let mut registry = Registry::default();
    register(&mut registry, config).unwrap();
    let spec = AlgorithmSpec {
        algorithm: AlgorithmId::new(id).unwrap(),
        config: serde_json::json!({"start": 1}),
    };
    let initial = registry.train(&spec, &[]).unwrap();
    let mut model = registry.open(&initial).unwrap();
    assert!(model.generate(&mut []).is_err());
    let mut output = [[0; 16]; 4];
    let first = model.generate(&mut output).unwrap();
    assert_eq!(first.written, 1);
    assert_eq!(first.state, GenerationState::AwaitingFeedback);
    let first_address = output[0];
    assert_eq!(model.generate(&mut output).unwrap().written, 0);
    let feedback = [
        Feedback::Active(first_address.into()),
        Feedback::BatchComplete,
    ];
    model.apply_feedback(&feedback).unwrap();
    let mut live = initial.clone();
    registry.save_model(&mut live, model.as_ref()).unwrap();
    let mut offline = initial;
    registry.update(&mut offline, &feedback).unwrap();
    assert_eq!(live.payload, offline.payload);
    let mut restored = registry.open(&live).unwrap();
    let next = restored.generate(&mut output).unwrap();
    assert_eq!(next.written, 1);
    assert_eq!(output[0][15], 2);
}

#[test]
fn python_live_and_offline_feedback_match() {
    exercise(
        PluginConfig::Python {
            interpreter: "python3".into(),
            script: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/adaptive.py"),
            timeout_seconds: 10,
        },
        "example/adaptive",
    );
}

#[test]
fn native_live_and_offline_feedback_match() {
    let status = std::process::Command::new(env!("CARGO"))
        .args([
            "build",
            "--offline",
            "-p",
            "sixseven-plugins",
            "--example",
            "native_tga",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("target"));
    let filename = format!(
        "{}native_tga{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    );
    exercise(
        PluginConfig::Native {
            path: target.join("debug/examples").join(filename),
        },
        "example/native",
    );
}
