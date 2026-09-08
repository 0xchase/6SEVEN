use std::{
    fs,
    process::{Command, Output},
};

fn run(root: &std::path::Path, args: &[&str]) -> Output {
    let result = Command::new(env!("CARGO_BIN_EXE_6seven"))
        .current_dir(root)
        .args(args)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    result
}

#[test]
fn builtin_train_generate_and_offline_feedback() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    fs::write(
        root.join("seeds.csv"),
        "saddr\n2001:db8::1\n2001:db8::2\n2001:db8::3\n",
    )
    .unwrap();
    run(
        root,
        &[
            "train",
            "--seeds",
            "seeds.csv",
            "--output-model",
            "model.bin",
            "det",
        ],
    );
    let generated = run(
        root,
        &[
            "generate",
            "--model",
            "model.bin",
            "--count",
            "16",
            "--unique",
        ],
    );
    let rows = String::from_utf8(generated.stdout).unwrap();
    assert_eq!(rows.lines().count(), 17);
    let addresses: Vec<_> = rows
        .lines()
        .skip(1)
        .map(|line| line.parse::<std::net::Ipv6Addr>().unwrap())
        .collect();
    let feedback: Vec<_> = addresses
        .iter()
        .map(|address| sixseven_core::Feedback::Active(*address))
        .chain([sixseven_core::Feedback::BatchComplete])
        .collect();
    fs::write(
        root.join("feedback.jsonl"),
        serde_json::to_vec(&feedback).unwrap(),
    )
    .unwrap();
    run(
        root,
        &[
            "feedback",
            "--input-model",
            "model.bin",
            "--scan-results",
            "feedback.jsonl",
            "--output-model",
            "updated.bin",
        ],
    );
    let updated = run(
        root,
        &[
            "generate",
            "--model",
            "updated.bin",
            "--count",
            "10",
            "--unique",
        ],
    );
    for line in String::from_utf8(updated.stdout).unwrap().lines().skip(1) {
        assert!(!addresses.contains(&line.parse().unwrap()));
    }
}

#[test]
fn python_plugin_uses_cli_registry() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let script =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../plugins/examples/adaptive.py");
    fs::write(
        root.join("plugins.json"),
        serde_json::to_vec(&serde_json::json!([
            {"kind": "python", "interpreter": "python3", "script": script, "timeout_seconds": 10}
        ]))
        .unwrap(),
    )
    .unwrap();
    fs::write(root.join("seeds.csv"), "saddr\n2001:db8::1\n").unwrap();
    run(
        root,
        &[
            "--plugins",
            "plugins.json",
            "train",
            "--seeds",
            "seeds.csv",
            "--algorithm",
            "example/adaptive",
            "--output-model",
            "model.bin",
        ],
    );
    let generated = run(
        root,
        &[
            "--plugins",
            "plugins.json",
            "generate",
            "--model",
            "model.bin",
            "--count",
            "1",
        ],
    );
    assert_eq!(
        String::from_utf8(generated.stdout).unwrap(),
        "saddr\n2001:db8::1\n"
    );
}

#[test]
fn offline_dealiasing_preserves_csv_and_exports_reusable_prefixes() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    fs::write(root.join("results.csv"), "saddr,class,success,note\n2001:db8::1,reply,1,remove\n2001:db9::1,reply,1,\"keep, quoted\"\n").unwrap();
    fs::write(
        root.join("aliases.txt"),
        "# aliases\n2001:db8::/32\n2001:db8::/48\n",
    )
    .unwrap();
    run(
        root,
        &[
            "dealias",
            "results.csv",
            "--aliased-prefixes",
            "aliases.txt",
            "--output-aliases",
            "combined.txt",
            "--interface",
            "not-a-real-interface",
            "-o",
            "clean.csv",
        ],
    );
    assert_eq!(
        fs::read_to_string(root.join("clean.csv")).unwrap(),
        "saddr,class,success,note\n2001:db9::1,reply,1,\"keep, quoted\"\n"
    );
    assert_eq!(
        fs::read_to_string(root.join("combined.txt")).unwrap(),
        "2001:db8::/32\n"
    );
    let second = run(
        root,
        &[
            "dealias",
            "results.csv",
            "--aliased-prefixes",
            "combined.txt",
        ],
    );
    assert_eq!(second.stdout, fs::read(root.join("clean.csv")).unwrap());
}

#[test]
fn dealiasing_rejects_invalid_flags_and_overlapping_files_before_network_io() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    fs::write(root.join("addresses.txt"), "::1\n").unwrap();
    fs::write(root.join("aliases.txt"), "::/120\n").unwrap();
    for args in [
        vec!["dealias", "addresses.txt"],
        vec![
            "dealias",
            "addresses.txt",
            "--aliased-prefixes",
            "aliases.txt",
            "--alias-prefix-lengths",
            "120",
        ],
        vec![
            "dealias",
            "addresses.txt",
            "--online",
            "--alias-prefix-lengths",
            "128",
        ],
        vec![
            "dealias",
            "addresses.txt",
            "--aliased-prefixes",
            "aliases.txt",
            "-o",
            "addresses.txt",
        ],
        vec![
            "dealias",
            "addresses.txt",
            "--aliased-prefixes",
            "aliases.txt",
            "--output-aliases",
            "aliases.txt",
        ],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_6seven"))
            .current_dir(root)
            .args(&args)
            .output()
            .unwrap();
        assert!(!output.status.success(), "{args:?}");
        assert!(!String::from_utf8_lossy(&output.stderr).contains("packet socket"));
    }
    assert_eq!(
        fs::read_to_string(root.join("addresses.txt")).unwrap(),
        "::1\n"
    );
}
