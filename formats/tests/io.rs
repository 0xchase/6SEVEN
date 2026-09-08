use sixseven_core::{AlgorithmId, Feedback, ModelArtifact};
use std::{collections::BTreeMap, io::Write};

#[test]
fn feedback_reconciles_duplicate_targets() {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    writeln!(
        file,
        "saddr,class\n::1,timeout\n::1,reply\n::1,unreachable\n::2,error"
    )
    .unwrap();
    assert_eq!(
        sixseven_formats::feedback::load(file.path()).unwrap(),
        vec![
            Feedback::Active("::1".parse().unwrap()),
            Feedback::Skipped("::2".parse().unwrap()),
            Feedback::BatchComplete,
        ]
    );
}

#[test]
fn contradictory_scan_outcomes_are_rejected() {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    writeln!(file, "saddr,class,success\n::1,reply,false").unwrap();
    assert!(sixseven_formats::feedback::load(file.path()).is_err());
}

#[test]
fn csv_selects_named_column_and_handles_quotes() {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    writeln!(file, "label,target\n\"first,host\",2001:db8::1").unwrap();
    let addresses = sixseven_formats::targets::read(file.path(), Some("target"))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(
        addresses,
        vec!["2001:db8::1".parse::<std::net::Ipv6Addr>().unwrap()]
    );
    assert!(sixseven_formats::targets::read(file.path(), Some("absent")).is_err());
}

#[test]
fn malformed_targets_are_reported() {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    writeln!(file, "# input\n\n2001:db8::1\nbroken").unwrap();
    let mut reader = sixseven_formats::targets::read(file.path(), None).unwrap();
    assert!(reader.next().unwrap().is_ok());
    assert!(reader.next().unwrap().is_err());
}

#[test]
fn model_payload_and_config_round_trip() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.bin");
    let mut artifact = ModelArtifact {
        algorithm: AlgorithmId::new("example/test").unwrap(),
        revision: 1,
        model_version: 3,
        payload: vec![0, 255, 6],
        config: serde_json::json!({"seed": 42}),
        metadata: BTreeMap::new(),
    };
    sixseven_formats::model::save(&path, &artifact).unwrap();
    artifact.revision += 1;
    artifact.payload.push(7);
    sixseven_formats::model::save(&path, &artifact).unwrap();
    assert_eq!(sixseven_formats::model::load(path).unwrap(), artifact);
}

#[test]
fn feedback_journal_preserves_batch_boundaries() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("feedback.jsonl");
    let batches = vec![
        vec![Feedback::Active(std::net::Ipv6Addr::LOCALHOST)],
        vec![Feedback::BatchComplete],
    ];
    let mut file = std::fs::File::create(&path).unwrap();
    for batch in &batches {
        sixseven_formats::feedback::write_batch(&mut file, batch).unwrap();
    }
    assert_eq!(
        sixseven_formats::feedback::load_batches(path).unwrap(),
        batches
    );
}
