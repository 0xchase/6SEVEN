use super::*;
fn tiny_reference_model() -> SixProbeModel {
    let patterns = vec![
        SixProbePattern::parse("2001:0db8:0000:0000:0000:0000:0000:00**").unwrap(),
        SixProbePattern::parse("2001:0db8:0000:0000:0000:0000:0001:0000").unwrap(),
    ];
    SixProbeModel {
        generation: Default::default(),
        patterns,
        ..SixProbeModel::default()
    }
}

fn pattern_strings(model: &SixProbeModel) -> Vec<String> {
    model
        .patterns
        .iter()
        .map(SixProbePattern::as_pattern_string)
        .collect()
}

fn subspace(pattern: &str) -> PatternSubspace {
    pattern_string_to_subspace(pattern).unwrap()
}

fn subspace_strings(patterns: Vec<PatternSubspace>) -> Vec<String> {
    patterns.iter().map(subspace_to_pattern_string).collect()
}

#[test]
fn forest_finalization_preserves_first_occurrence_order() {
    let patterns = vec![
        subspace("2001:0db8:0000:0000:0000:0000:0000:0001"),
        subspace("2001:0db8:0000:0000:0000:0000:0000:0000"),
        subspace("2001:0db8:0000:0000:0000:0000:0000:0001"),
        subspace("2001:0db8:0000:0000:0000:0000:0000:0002"),
        subspace("2001:0db8:0000:0000:0000:0000:0000:0000"),
    ];
    assert_eq!(
        subspace_strings(finalize_patterns(SixProbeMode::Forest, patterns)),
        vec![
            "2001:0db8:0000:0000:0000:0000:0000:0001".to_string(),
            "2001:0db8:0000:0000:0000:0000:0000:0000".to_string(),
            "2001:0db8:0000:0000:0000:0000:0000:0002".to_string(),
        ]
    );
}

#[test]
fn single_tree_finalization_keeps_upstream_order() {
    let patterns = vec![
        subspace("2001:0db8:0000:0000:0000:0000:0000:0001"),
        subspace("2001:0db8:0000:0000:0000:0000:0000:0000"),
        subspace("2001:0db8:0000:0000:0000:0000:0000:0001"),
    ];
    assert_eq!(
        subspace_strings(finalize_patterns(SixProbeMode::SingleTree, patterns)),
        vec![
            "2001:0db8:0000:0000:0000:0000:0000:0001".to_string(),
            "2001:0db8:0000:0000:0000:0000:0000:0000".to_string(),
            "2001:0db8:0000:0000:0000:0000:0000:0001".to_string(),
        ]
    );
}

#[test]
fn pattern_iteration_matches_generate_address_nibble_order() {
    let pattern = SixProbePattern::parse("2001:0db8:0000:0000:0000:0000:0000:00**").unwrap();
    let addresses: Vec<String> = SixProbePatternIter::new(pattern)
        .map(|address| std::net::Ipv6Addr::from(address).to_string())
        .collect();
    let rendered: Vec<String> = [0usize, 1, 15, 16, 255]
        .into_iter()
        .map(|idx| addresses[idx].clone())
        .collect();
    assert_eq!(
        rendered,
        vec![
            "2001:db8::".to_string(),
            "2001:db8::1".to_string(),
            "2001:db8::f".to_string(),
            "2001:db8::10".to_string(),
            "2001:db8::ff".to_string(),
        ]
    );
}

#[test]
fn stream_crosses_pattern_boundaries() {
    let model = tiny_reference_model();

    let rendered: Vec<String> = model
        .stream()
        .unwrap()
        .skip(250)
        .take(8)
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .into_iter()
        .map(|addr| std::net::Ipv6Addr::from(addr).to_string())
        .collect();

    assert_eq!(
        rendered,
        vec![
            "2001:db8::fa".to_string(),
            "2001:db8::fb".to_string(),
            "2001:db8::fc".to_string(),
            "2001:db8::fd".to_string(),
            "2001:db8::fe".to_string(),
            "2001:db8::ff".to_string(),
            "2001:db8::1:0".to_string(),
        ]
    );
}

#[test]
fn overlapping_patterns_are_streamed_like_generate_address_outputs() {
    let patterns = vec![
        SixProbePattern::parse("2001:0db8:0000:0000:0000:0000:0000:000*").unwrap(),
        SixProbePattern::parse("2001:0db8:0000:0000:0000:0000:0000:0000").unwrap(),
    ];
    let model = SixProbeModel {
        generation: Default::default(),
        patterns,
        ..SixProbeModel::default()
    };

    let rendered: Vec<String> = model
        .stream()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .into_iter()
        .map(|addr| std::net::Ipv6Addr::from(addr).to_string())
        .collect();

    assert_eq!(rendered.len(), 17);
    assert_eq!(rendered[0], "2001:db8::");
    assert_eq!(rendered[16], "2001:db8::");
}

#[test]
fn released_python_fixtures_match_all_strategies_and_seeded_forests() {
    #[derive(Deserialize)]
    struct Tree {
        strategy: String,
        beta: usize,
        patterns: Vec<String>,
    }
    #[derive(Deserialize)]
    struct Forest {
        seed: u64,
        patterns: Vec<String>,
    }
    #[derive(Deserialize)]
    struct Fixture {
        seeds: Vec<String>,
        trees: Vec<Tree>,
        forests: Vec<Forest>,
    }
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../tests/fixtures/six_probe/reference.json"
    ))
    .unwrap();
    let observations: Vec<_> = fixture
        .seeds
        .iter()
        .map(|seed| Observation {
            address: u128::from_str_radix(seed, 16).unwrap().to_be_bytes(),
            active: true,
        })
        .collect();
    for tree in fixture.trees {
        let dhc_type = match tree.strategy.as_str() {
            "LeftVDPS" => DhcType::LeftVdps,
            "RightVDPS" => DhcType::RightVdps,
            "MinEntropy" => DhcType::MinEntropy,
            "MaxCover" => DhcType::MaxCover,
            _ => panic!("unknown strategy"),
        };
        let config = SixProbe {
            mode: SixProbeMode::SingleTree,
            beta: tree.beta,
            dhc_type,
            ..SixProbe::default()
        };
        assert_eq!(
            pattern_strings(&config.train(&observations).unwrap()),
            tree.patterns,
            "{}",
            tree.strategy
        );
    }
    for forest in fixture.forests {
        let config = SixProbe {
            tree_num: 8,
            random_seed: Some(forest.seed),
            ..SixProbe::default()
        };
        let mut actual = pattern_strings(&config.train(&observations).unwrap());
        actual.sort();
        assert_eq!(actual, forest.patterns, "seed {}", forest.seed);
    }
}

#[test]
fn degenerate_inputs_terminate_without_inventing_patterns() {
    let config = SixProbe::default();
    assert!(config.train(&[]).is_err());
    let seed = Observation {
        address: [0; 16],
        active: true,
    };
    for size in [1, 2, 12, 100] {
        let model = config.train(&vec![seed; size]).unwrap();
        assert!(model.patterns.is_empty());
    }
    assert!(
        config
            .train(&[Observation {
                active: false,
                ..seed
            }])
            .is_err()
    );
}

#[test]
fn terminal_root_is_mined_and_training_state_is_local() {
    let config = SixProbe::default();
    let observations: Vec<_> = (0u128..3)
        .map(|address| Observation {
            address: address.to_be_bytes(),
            active: true,
        })
        .collect();
    let first = config.train(&observations).unwrap();
    assert_eq!(
        pattern_strings(&first),
        ["0000:0000:0000:0000:0000:0000:0000:000*"]
    );
    assert_eq!(
        pattern_strings(&first),
        pattern_strings(&config.train(&observations).unwrap())
    );
}

#[test]
fn serialized_patterns_validate_and_generation_resumes_across_batches() {
    use crate::TargetModel;
    let mut model = tiny_reference_model();
    let expected = model
        .stream()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let mut actual = Vec::new();
    loop {
        let mut output = [[0; 16]; 17];
        let result = model.generate(&mut output).unwrap();
        actual.extend_from_slice(&output[..result.written]);
        if result.state == crate::GenerationState::Exhausted {
            break;
        }
    }
    assert_eq!(actual, expected);
    let mut restored: SixProbeModel =
        bincode::deserialize(&bincode::serialize(&model).unwrap()).unwrap();
    let mut output = [[0; 16]; 1];
    assert_eq!(restored.generate(&mut output).unwrap().written, 1);
    assert_eq!(output[0], expected[0]);
    for pattern in [
        "*",
        "0000:0000:0000:0000:0000:0000:000*:****",
        "0000:0000:0000:0000:0000:0000:0000:000g",
    ] {
        assert!(serde_json::from_value::<SixProbePattern>(serde_json::json!(pattern)).is_err());
    }
}

#[test]
fn four_dimensional_patterns_expand_all_combinations() {
    let pattern = SixProbePattern::parse("2001:0db8:*000:000*:0000:00*0:0000:000*").unwrap();
    let addresses: Vec<_> = SixProbePatternIter::new(pattern).collect();
    assert_eq!(addresses.len(), 65_536);
    assert!(addresses.windows(2).all(|pair| pair[0] < pair[1]));
    assert_eq!(
        std::net::Ipv6Addr::from(addresses[0]).to_string(),
        "2001:db8::"
    );
    assert_eq!(
        std::net::Ipv6Addr::from(addresses[65_535]).to_string(),
        "2001:db8:f000:f:0:f0:0:f"
    );
}

#[derive(Deserialize)]
pub(super) struct StageCase {
    pub(super) name: String,
    seeds: Vec<String>,
    pub(super) weights: Vec<f64>,
    pub(super) outliers: Vec<f64>,
    pub(super) splits: Vec<Vec<Vec<usize>>>,
    forest: Option<Vec<String>>,
    pub(super) mined: Vec<String>,
}

impl StageCase {
    pub(super) fn nibbles(&self) -> Vec<encoding::NibbleAddr> {
        self.seeds
            .iter()
            .map(|seed| addr_to_nibbles(&u128::from_str_radix(seed, 16).unwrap().to_be_bytes()))
            .collect()
    }
}

#[derive(Deserialize)]
struct Expansion {
    pattern: String,
    count: usize,
    checksum: u64,
}

#[derive(Deserialize)]
struct StageFixture {
    cases: Vec<StageCase>,
    expansions: Vec<Expansion>,
}

fn stages() -> StageFixture {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/six_probe/stages.json"
    ))
    .unwrap()
}

pub(super) fn stage_cases() -> Vec<StageCase> {
    stages().cases
}

#[test]
fn default_forests_match_reference_for_parallel_and_full_width_inputs() {
    for case in stage_cases() {
        let observations: Vec<_> = case
            .nibbles()
            .iter()
            .map(|seed| Observation {
                address: encoding::nibbles_to_addr(seed),
                active: true,
            })
            .collect();
        if let Some(expected) = case.forest {
            let config = SixProbe {
                random_seed: Some(67),
                ..SixProbe::default()
            };
            let mut actual = pattern_strings(&config.train(&observations).unwrap());
            actual.sort();
            assert_eq!(actual, expected, "{}", case.name);
        }
    }
}

#[test]
fn expansions_match_reference_address_checksums() {
    for expected in stages().expansions {
        let pattern = SixProbePattern::parse(&expected.pattern).unwrap();
        let mut checksum = 14_695_981_039_346_656_037u64;
        let mut count = 0;
        for address in SixProbePatternIter::new(pattern) {
            for byte in address {
                checksum = (checksum ^ u64::from(byte)).wrapping_mul(1_099_511_628_211);
            }
            count += 1;
        }
        assert_eq!(
            (count, checksum),
            (expected.count, expected.checksum),
            "{}",
            expected.pattern
        );
    }
}

#[test]
fn feedback_is_unsupported_and_cannot_change_generation() {
    use crate::{Feedback, TargetModel};
    let mut model = tiny_reference_model();
    let mut buffer = [[0; 16]; 7];
    model.generate(&mut buffer).unwrap();
    let mut expected = model.clone();
    for feedback in [
        vec![],
        vec![Feedback::Active("2001:db8::1".parse().unwrap())],
        vec![Feedback::Inactive("2001:db8::2".parse().unwrap())],
        vec![Feedback::Aliased("2001:db8::/32".parse().unwrap())],
        vec![
            Feedback::Skipped("2001:db8::3".parse().unwrap()),
            Feedback::BatchComplete,
        ],
    ] {
        assert!(matches!(
            model.apply_feedback(&feedback),
            Err(TgaError::Unsupported(_))
        ));
    }
    let mut actual_buffer = [[0; 16]; 300];
    let mut expected_buffer = actual_buffer;
    assert_eq!(
        model.generate(&mut actual_buffer).unwrap(),
        expected.generate(&mut expected_buffer).unwrap()
    );
    assert_eq!(actual_buffer, expected_buffer);
}

#[test]
fn inactive_observations_do_not_influence_training() {
    let mut observations: Vec<_> = (0u128..3)
        .map(|address| Observation {
            address: address.to_be_bytes(),
            active: true,
        })
        .collect();
    let config = SixProbe::default();
    let expected = pattern_strings(&config.train(&observations).unwrap());
    observations.extend((0..100).map(|_| Observation {
        address: [255; 16],
        active: false,
    }));
    assert_eq!(
        pattern_strings(&config.train(&observations).unwrap()),
        expected
    );
}

#[test]
fn empty_models_exhaust_without_requesting_feedback() {
    use crate::TargetModel;
    let observations = [Observation {
        address: [0; 16],
        active: true,
    }];
    let mut model = SixProbe::default().train(&observations).unwrap();
    let mut buffer = [[255; 16]; 8];
    for _ in 0..2 {
        assert_eq!(
            model.generate(&mut buffer).unwrap(),
            crate::Generated {
                written: 0,
                state: crate::GenerationState::Exhausted,
            }
        );
        assert_eq!(buffer, [[255; 16]; 8]);
    }
}

#[test]
fn version_one_models_migrate_without_changing_targets() {
    let old = tiny_reference_model();
    let bytes = bincode::serialize(&(
        &old.patterns,
        old.beta,
        old.tree_num,
        old.mode,
        old.dhc_type,
        old.split_array_type,
        old.random_seed,
        old.split_order,
    ))
    .unwrap();
    let migrated = SixProbe::decode_model(1, &bytes).unwrap();
    assert!(migrated.aliased_prefixes.is_empty());
    assert_eq!(pattern_strings(&migrated), pattern_strings(&old));
    assert_eq!(
        migrated
            .stream()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap(),
        old.stream()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    );
    assert!(SixProbe::decode_model(0, &bytes).is_err());
    assert!(SixProbe::decode_model(1, &[0]).is_err());
}

struct TemporaryFile(std::path::PathBuf);

impl TemporaryFile {
    fn new(contents: &str) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "six-probe-{}-{}",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = fs::File::create_new(&path).unwrap();
        file.write_all(contents.as_bytes()).unwrap();
        Self(path)
    }
}

impl Drop for TemporaryFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[test]
fn alias_list_filters_targets_after_training_and_survives_serialization() {
    use crate::TargetModel;
    let file = TemporaryFile::new("# known aliases\n2001:db8::8/125\n2001:db8::a/128\n\n");
    let mut config = SixProbe {
        aliased_prefixes: Some(file.0.clone()),
        ..SixProbe::default()
    };
    let observations: Vec<_> = ["2001:db8::1", "2001:db8::a", "2001:db8::f"]
        .iter()
        .map(|address| Observation {
            address: address.parse::<std::net::Ipv6Addr>().unwrap().octets(),
            active: true,
        })
        .collect();
    let model = config.train(&observations).unwrap();
    config.aliased_prefixes = None;
    assert_eq!(
        pattern_strings(&model),
        pattern_strings(&config.train(&observations).unwrap())
    );
    let bytes = SixProbe::encode_model(&model).unwrap();
    drop(file);
    let mut restored = SixProbe::decode_model(SixProbe::MODEL_VERSION, &bytes).unwrap();
    let mut output = [[0; 16]; 32];
    let generated = restored.generate(&mut output).unwrap();
    assert_eq!(
        generated,
        crate::Generated {
            written: 8,
            state: crate::GenerationState::Exhausted
        }
    );
    assert_eq!(
        output[..8],
        (0..8u128)
            .map(|suffix| (0x20010db8000000000000000000000000u128 | suffix).to_be_bytes())
            .collect::<Vec<_>>()
    );
}

#[test]
fn alias_configuration_and_export_failures_are_reported() {
    let file = TemporaryFile::new("2001:db8::/129\n");
    let mut config = SixProbe {
        aliased_prefixes: Some(file.0.clone()),
        ..SixProbe::default()
    };
    let observations = [Observation {
        address: [0; 16],
        active: true,
    }];
    assert!(matches!(
        config.train(&observations),
        Err(TgaError::Config(_))
    ));
    config.aliased_prefixes = None;
    config.export_patterns = Some(file.0.join("patterns"));
    assert!(matches!(
        config.train(&observations),
        Err(TgaError::Training(_))
    ));
}

#[test]
fn fully_aliased_models_exhaust_and_never_request_feedback() {
    use crate::TargetModel;
    let mut model = tiny_reference_model();
    model.aliased_prefixes = vec!["::/0".parse().unwrap()];
    let mut buffer = [[255; 16]; 16];
    for _ in 0..2 {
        assert_eq!(
            model.generate(&mut buffer).unwrap(),
            crate::Generated {
                written: 0,
                state: crate::GenerationState::Exhausted,
            }
        );
        assert_eq!(buffer, [[255; 16]; 16]);
    }
}

#[test]
fn registry_preserves_alias_filtering_and_reports_feedback_as_unsupported() {
    let file = TemporaryFile::new("::8/125\n");
    let config = SixProbe {
        aliased_prefixes: Some(file.0.clone()),
        ..SixProbe::default()
    };
    let mut registry = crate::Registry::default();
    registry.register_algorithm::<SixProbe>().unwrap();
    let artifact = registry
        .train(
            &crate::AlgorithmSpec {
                algorithm: crate::AlgorithmId::new("6probe").unwrap(),
                config: serde_json::to_value(config).unwrap(),
            },
            &(1u128..4)
                .map(|address| Observation {
                    address: address.to_be_bytes(),
                    active: true,
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();
    assert_eq!(artifact.model_version, 2);
    drop(file);
    let mut model = registry.open(&artifact).unwrap();
    let mut output = [[0; 16]; 32];
    let batch = model.generate(&mut output).unwrap();
    assert_eq!(batch.written, 8);
    assert_eq!(batch.state, crate::GenerationState::Exhausted);
    assert_eq!(
        output[..8],
        (0u128..8).map(u128::to_be_bytes).collect::<Vec<_>>()
    );
    assert!(matches!(
        model.apply_feedback(&[crate::Feedback::BatchComplete]),
        Err(TgaError::Unsupported(_))
    ));
}

#[test]
fn cli_defaults_match_library_defaults_and_accept_alias_lists() {
    use clap::Parser;
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        config: SixProbe,
    }
    let json_config: SixProbe = serde_json::from_value(serde_json::json!({})).unwrap();
    assert_eq!(
        serde_json::to_value(json_config).unwrap(),
        serde_json::to_value(SixProbe::default()).unwrap()
    );
    let cli = Cli::try_parse_from(["6probe"]).unwrap();
    assert_eq!(
        serde_json::to_value(cli.config).unwrap(),
        serde_json::to_value(SixProbe::default()).unwrap()
    );
    let cli = Cli::try_parse_from(["6probe", "--aliased-prefixes", "aliases.txt"]).unwrap();
    assert_eq!(
        cli.config.aliased_prefixes.unwrap(),
        std::path::PathBuf::from("aliases.txt")
    );
}

#[test]
fn pattern_exports_match_mined_patterns_including_empty_results() {
    let file = TemporaryFile::new("stale output\n");
    let config = SixProbe {
        export_patterns: Some(file.0.clone()),
        ..SixProbe::default()
    };
    let observations: Vec<_> = (0u128..3)
        .map(|address| Observation {
            address: address.to_be_bytes(),
            active: true,
        })
        .collect();
    let model = config.train(&observations).unwrap();
    assert_eq!(
        fs::read_to_string(&file.0).unwrap(),
        format!("{}\n", pattern_strings(&model).join("\n"))
    );
    config.train(&observations[..1]).unwrap();
    assert!(fs::read(&file.0).unwrap().is_empty());
}

#[test]
fn two_seed_regions_only_expand_when_one_nibble_varies() {
    use crate::TargetModel;
    for (last, count) in [(1, 16), (0x11, 0), (0x111, 0)] {
        let observations = [0u128, last].map(|value| Observation {
            address: value.to_be_bytes(),
            active: true,
        });
        let mut model = SixProbe::default().train(&observations).unwrap();
        let mut output = [[0; 16]; 32];
        let batch = model.generate(&mut output).unwrap();
        assert_eq!(batch.written, count);
        assert_eq!(batch.state, crate::GenerationState::Exhausted);
    }
}
