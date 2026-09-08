use super::*;
use crate::address::AddressExt;
use crate::{Feedback, GenerationState, TargetModel};
use clap::Parser;
use generation::sampling_round_budget;
use mining::{graph_cut_components, outlier_detect, space_partition_regions};
use std::net::Ipv6Addr;
use std::sync::Arc;

fn address(text: &str) -> Address {
    text.parse::<Ipv6Addr>().unwrap().octets()
}

fn train(seeds: &[&str]) -> SixGraphModel {
    SixGraph::default()
        .train(
            &seeds
                .iter()
                .map(|text| Observation {
                    address: address(text),
                    active: true,
                })
                .collect::<Vec<_>>(),
        )
        .unwrap()
}

fn next_round(model: &mut SixGraphModel, active: bool) -> Vec<Address> {
    let mut targets = Vec::new();
    loop {
        let mut output = [[0; 16]; 7];
        let batch = model.generate(&mut output).unwrap();
        targets.extend_from_slice(&output[..batch.written]);
        if batch.state != GenerationState::Ready {
            break;
        }
    }
    let mut feedback = targets
        .iter()
        .map(|&ip| {
            if active {
                Feedback::Active(ip.into())
            } else {
                Feedback::Inactive(ip.into())
            }
        })
        .collect::<Vec<_>>();
    feedback.push(Feedback::BatchComplete);
    model.apply_feedback(&feedback).unwrap();
    targets
}

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    config: SixGraph,
}

#[test]
fn defaults_and_validation_cover_cli_serde_and_training() {
    let cli = Cli::parse_from(["6graph"]).config;
    let serde: SixGraph = serde_json::from_str("{}").unwrap();
    for config in [cli, serde, SixGraph::default()] {
        assert_eq!(config.min_region_size, 16);
        assert_eq!(config.distance_threshold, 12);
        assert_eq!(config.iterations, 3);
        assert_eq!(config.sample_seed, DEFAULT_SAMPLE_SEED);
        assert!(config.seed_rejoining);
    }
    for config in [
        SixGraph {
            min_region_size: 0,
            ..Default::default()
        },
        SixGraph {
            distance_threshold: 33,
            ..Default::default()
        },
    ] {
        assert!(matches!(config.train(&[]), Err(TgaError::Config(_))));
    }
    assert!(matches!(
        SixGraph::default().train(&[]),
        Err(TgaError::Training(_))
    ));
}

#[test]
fn partition_preserves_breadth_first_order() {
    let seeds = ["1000::1", "1000::2", "2000::1", "2000::2", "2000::3"].map(address);
    assert_eq!(
        space_partition_regions(&seeds, 2),
        vec![
            seeds[..2].to_vec(),
            vec![seeds[2]],
            vec![seeds[3]],
            vec![seeds[4]],
        ]
    );
}

#[test]
fn density_must_improve_strictly_for_both_components() {
    let seeds = ["::", "::1", "::10", "::11"].map(address);
    assert_eq!(
        graph_cut_components(&seeds, 32),
        vec![vec![0, 1], vec![2, 3]]
    );
    let mined = outlier_detect(&[address("::1"), address("::2"), address("1000::")], 1);
    assert_eq!(mined.patterns[0].variable_nibbles, vec![31]);
    assert_eq!(mined.outliers, vec![address("1000::")]);
}

#[test]
fn mining_matches_original_python_fixtures() {
    #[derive(Deserialize)]
    struct Fixtures {
        cases: Vec<Case>,
    }
    #[derive(Deserialize)]
    struct Case {
        name: String,
        seeds: Vec<String>,
        threshold: usize,
        iterations: usize,
        patterns: Vec<Vec<String>>,
        outliers: Vec<String>,
        paper_patterns: Vec<Vec<String>>,
        paper_outliers: Vec<String>,
    }
    fn decode(hex: &str) -> Address {
        u128::from_str_radix(hex, 16).unwrap().to_be_bytes()
    }
    let fixtures: Fixtures =
        serde_json::from_str(include_str!("../../../tests/fixtures/sixgraph/mining.json")).unwrap();
    for case in fixtures.cases {
        let seeds = case.seeds.iter().map(|hex| decode(hex)).collect::<Vec<_>>();
        for (seed_rejoining, patterns, outliers) in [
            (false, &case.patterns, &case.outliers),
            (true, &case.paper_patterns, &case.paper_outliers),
        ] {
            let mined = mine_patterns(
                seeds.clone(),
                &SixGraph {
                    distance_threshold: case.threshold,
                    iterations: case.iterations,
                    seed_rejoining,
                    ..Default::default()
                },
            );
            let actual = mined
                .patterns
                .iter()
                .map(|pattern| {
                    let mut seeds = pattern.seeds.clone();
                    seeds.sort_unstable();
                    seeds
                })
                .collect::<Vec<_>>();
            let expected = patterns
                .iter()
                .map(|pattern| pattern.iter().map(|hex| decode(hex)).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "{} patterns", case.name);
            assert_eq!(
                mined.outliers,
                outliers.iter().map(|hex| decode(hex)).collect::<Vec<_>>(),
                "{} outliers",
                case.name
            );
            assert_eq!(
                actual.iter().map(Vec::len).sum::<usize>() + mined.outliers.len(),
                seeds.len()
            );
        }
    }
}

#[test]
fn observations_preserve_active_order_and_exclude_every_known_address() {
    let observations =
        [("::1", true), ("::2", false), ("::1", true), ("::3", true)].map(|(text, active)| {
            Observation {
                address: address(text),
                active,
            }
        });
    let training = TrainingAddresses::from_observations(&observations);
    assert_eq!(training.active, vec![address("::1"), address("::3")]);
    let mut model = SixGraph::default().train(&observations).unwrap();
    assert_eq!(model.seed_count(), 2);
    let mut all = Vec::new();
    loop {
        let round = next_round(&mut model, false);
        if round.is_empty() {
            break;
        }
        all.extend(round);
    }
    assert_eq!(all.len(), 13);
    assert_eq!(all.iter().copied().collect::<HashSet<_>>().len(), all.len());
    assert!(all.iter().all(|target| !training.known.contains(target)));
}

#[test]
fn round_budgets_follow_equation_four_without_overflow() {
    assert_eq!(
        (0..5)
            .map(|round| sampling_round_budget(round, 3))
            .collect::<Vec<_>>(),
        vec![3, 3, 6, 12, 24]
    );
    assert_eq!(sampling_round_budget(usize::MAX, 3), usize::MAX);
    let mut model = train(&["::1", "::2"]);
    let sizes = (0..5)
        .map(|_| next_round(&mut model, false).len())
        .collect::<Vec<_>>();
    assert_eq!(sizes, vec![2, 2, 4, 6, 0]);
}

#[test]
fn feedback_is_required_at_round_boundaries() {
    let mut model = train(&["::1", "::2"]);
    assert!(model.generate(&mut []).is_err());
    let mut output = [[0; 16]; 2];
    let first = model.generate(&mut output).unwrap();
    assert_eq!(first.written, 2);
    assert_eq!(first.state, GenerationState::AwaitingFeedback);
    let waiting = model.generate(&mut output).unwrap();
    assert_eq!(waiting.written, 0);
    assert_eq!(waiting.state, GenerationState::AwaitingFeedback);
    model.apply_feedback(&[Feedback::BatchComplete]).unwrap();
    assert_eq!(model.generate(&mut output).unwrap().written, 2);
}

#[test]
fn positive_feedback_expands_support_within_the_original_pattern() {
    let mut model = train(&["::", "::11"]);
    let initial = model.patterns[0].seeds.clone();
    let mut feedback = Vec::new();
    for &seed in &initial {
        for position in [30, 31] {
            for value in 0..16 {
                let mut candidate = seed;
                candidate.set_nibble(position, value);
                if !initial.contains(&candidate) {
                    feedback.push(Feedback::Inactive(candidate.into()));
                }
            }
        }
    }
    let discovered = address("::2");
    feedback.push(Feedback::Active(discovered.into()));
    feedback.push(Feedback::BatchComplete);
    let mut inactive_only = model.clone();
    inactive_only
        .apply_feedback(&feedback[..feedback.len() - 2])
        .unwrap();
    inactive_only
        .apply_feedback(&[Feedback::BatchComplete])
        .unwrap();
    assert_eq!(
        inactive_only.generate(&mut [[0; 16]; 1]).unwrap().state,
        GenerationState::Exhausted
    );
    model.apply_feedback(&feedback).unwrap();
    let targets = next_round(&mut model, false);
    assert_eq!(targets.len(), 2);
    for target in targets {
        assert_eq!(target.nibble_distance(&discovered), 1);
        assert!(initial.iter().all(|seed| target.nibble_distance(seed) > 1));
        assert!(model.patterns[0].contains(&target));
    }
}

#[test]
fn feedback_excludes_skipped_targets_and_aliased_prefixes() {
    let mut model = train(&["::1", "::2"]);
    let skipped = model.scan.targets[0];
    model
        .apply_feedback(&[Feedback::Skipped(skipped.into())])
        .unwrap();
    assert!(!next_round(&mut model, false).contains(&skipped));
    model
        .apply_feedback(&[
            Feedback::Aliased("::/124".parse().unwrap()),
            Feedback::BatchComplete,
        ])
        .unwrap();
    assert_eq!(
        model.generate(&mut [[0; 16]; 1]).unwrap().state,
        GenerationState::Exhausted
    );
}

#[test]
fn overlapping_patterns_never_repeat_targets() {
    let patterns = vec![
        Pattern {
            variable_nibbles: vec![30],
            seeds: vec![address("::2"), address("::22")],
        },
        Pattern {
            variable_nibbles: vec![31],
            seeds: vec![address("::10"), address("::11")],
        },
    ];
    let known = patterns
        .iter()
        .flat_map(|pattern| pattern.seeds.iter().copied())
        .collect();
    let mut model = SixGraphModel::new(patterns, known, 0, DEFAULT_SAMPLE_SEED);
    let mut all = Vec::new();
    loop {
        let round = next_round(&mut model, false);
        if round.is_empty() {
            break;
        }
        all.extend(round);
    }
    assert_eq!(all.len(), 27);
    assert_eq!(all.iter().copied().collect::<HashSet<_>>().len(), 27);
    assert!(all.contains(&address("::12")));
}

#[test]
fn save_resume_and_feedback_replay_preserve_the_next_round() {
    let mut live = train(&["::", "::11"]);
    let mut replay = live.clone();
    assert!(Arc::ptr_eq(&live.patterns, &replay.patterns));
    let mut first = [[0; 16]; 1];
    live.generate(&mut first).unwrap();
    let mut resumed = SixGraph::decode_model(2, &SixGraph::encode_model(&live).unwrap()).unwrap();
    let mut rest = [[0; 16]; 10];
    let mut resumed_rest = rest;
    assert_eq!(
        live.generate(&mut rest).unwrap(),
        resumed.generate(&mut resumed_rest).unwrap()
    );
    assert_eq!(rest, resumed_rest);
    let feedback = [
        Feedback::Active(first[0].into()),
        Feedback::Inactive(rest[0].into()),
        Feedback::BatchComplete,
    ];
    live.apply_feedback(&feedback).unwrap();
    replay.apply_feedback(&feedback).unwrap();
    assert_eq!(
        SixGraph::encode_model(&live).unwrap(),
        SixGraph::encode_model(&replay).unwrap()
    );
    assert_eq!(next_round(&mut live, false), next_round(&mut replay, false));
    assert!(SixGraph::decode_model(1, &[]).is_err());
}

#[test]
fn buffer_size_does_not_change_sampling() {
    let model = train(&["::", "::11", "::22"]);
    let mut small = model.clone();
    let mut large = model;
    let mut expected = [[0; 16]; 20];
    let batch = large.generate(&mut expected).unwrap();
    let mut actual = Vec::new();
    loop {
        let mut output = [[0; 16]; 1];
        let next = small.generate(&mut output).unwrap();
        actual.extend_from_slice(&output[..next.written]);
        if next.state != GenerationState::Ready {
            break;
        }
    }
    assert_eq!(actual, expected[..batch.written]);
}

#[test]
fn malformed_model_is_rejected_and_singletons_exhaust() {
    let mut value = serde_json::to_value(train(&["::1", "::2"])).unwrap();
    value["patterns"][0]["variable_nibbles"] = serde_json::json!([32]);
    assert!(serde_json::from_value::<SixGraphModel>(value).is_err());
    let mut model = train(&["::1"]);
    assert_eq!(model.seed_count(), 1);
    assert_eq!(
        model.generate(&mut [[0; 16]; 1]).unwrap().state,
        GenerationState::Exhausted
    );
}

#[test]
fn feedback_order_and_duplicate_observations_do_not_change_sampling() {
    let mut forward = train(&["::", "::11"]);
    let mut reverse = forward.clone();
    let observations = [
        Feedback::Inactive(address("::2").into()),
        Feedback::Active(address("::2").into()),
        Feedback::Active(address("::12").into()),
        Feedback::Active(address("::2").into()),
    ];
    forward.apply_feedback(&observations).unwrap();
    for event in observations.iter().rev() {
        reverse.apply_feedback(std::slice::from_ref(event)).unwrap();
    }
    let encoded = SixGraph::encode_model(&forward).unwrap();
    let mut restored = SixGraph::decode_model(2, &encoded).unwrap();
    for model in [&mut forward, &mut reverse, &mut restored] {
        model.apply_feedback(&[Feedback::BatchComplete]).unwrap();
    }
    assert_eq!(
        SixGraph::encode_model(&forward).unwrap(),
        SixGraph::encode_model(&reverse).unwrap()
    );
    assert_eq!(
        SixGraph::encode_model(&forward).unwrap(),
        SixGraph::encode_model(&restored).unwrap()
    );
    assert_eq!(forward.patterns[0].seeds.len(), 4);
}

#[test]
fn unchanged_outliers_stop_even_with_unbounded_refinement() {
    let mined = mine_patterns(
        vec![address("::1")],
        &SixGraph {
            iterations: usize::MAX,
            ..Default::default()
        },
    );
    assert_eq!(mined.outliers, vec![address("::1")]);
}

#[test]
fn paper_rejoining_recovers_contained_outliers_before_generation() {
    let observations = [
        "3120", "1300", "3333", "1302", "2022", "3032", "2032", "2021", "0202", "0010", "1023",
        "2133", "0111", "1121", "2000", "0100",
    ]
    .map(|suffix| Observation {
        address: address(&format!("::{suffix}")),
        active: true,
    });
    let mut config = SixGraph {
        distance_threshold: 2,
        iterations: 0,
        seed_rejoining: false,
        ..Default::default()
    };
    let reference = config.train(&observations).unwrap();
    let outlier = address("::2133");
    assert!(
        !reference
            .patterns
            .iter()
            .any(|pattern| pattern.seeds.contains(&outlier))
    );
    config.seed_rejoining = true;
    let paper = config.train(&observations).unwrap();
    assert_eq!(paper.seed_count(), observations.len());
    assert!(paper.outlier_count < reference.outlier_count);
    assert_eq!(
        paper
            .patterns
            .iter()
            .filter(|pattern| pattern.seeds.contains(&outlier))
            .count(),
        1
    );
    for (before, after) in reference.patterns.iter().zip(paper.patterns.iter()) {
        assert_eq!(before.variable_nibbles, after.variable_nibbles);
        assert!(after.seeds.iter().all(|seed| before.contains(seed)));
    }
    let parsed = Cli::parse_from(["6graph", "--seed-rejoining", "false"]);
    assert!(!parsed.config.seed_rejoining);
    let parsed: SixGraph = serde_json::from_str(r#"{"seed_rejoining":false}"#).unwrap();
    assert!(!parsed.seed_rejoining);
}

#[test]
fn rejoining_preserves_outlier_order_and_unique_seed_ownership() {
    let mut patterns = vec![
        Pattern {
            variable_nibbles: vec![30],
            seeds: vec![address("::2"), address("::22")],
        },
        Pattern {
            variable_nibbles: vec![31],
            seeds: vec![address("::10"), address("::11")],
        },
    ];
    let mut outliers = vec![address("1000::"), address("::12"), address("2000::")];
    mining::rejoin_outliers(&mut patterns, &mut outliers);
    assert_eq!(outliers, vec![address("1000::"), address("2000::")]);
    assert_eq!(patterns[0].seeds.last(), Some(&address("::12")));
    assert_eq!(patterns[1].seeds.len(), 2);
}

#[test]
fn complete_scanning_matches_an_exhaustive_feedback_oracle() {
    fn adjacent(left: u8, right: u8) -> bool {
        (left >> 4 == right >> 4) != (left & 15 == right & 15)
    }
    for outcome in 0..3 {
        for capacity in [1, 3, 256] {
            let observations = [
                Observation {
                    address: 0u128.to_be_bytes(),
                    active: true,
                },
                Observation {
                    address: 17u128.to_be_bytes(),
                    active: true,
                },
                Observation {
                    address: 255u128.to_be_bytes(),
                    active: false,
                },
            ];
            let mut model = SixGraph::default().train(&observations).unwrap();
            let mut replay = model.clone();
            let mut active = BTreeSet::from([0u8, 17]);
            let mut known = BTreeSet::from([0u8, 17, 255]);
            let mut budgets = Vec::<usize>::new();
            let mut rounds = 0;
            loop {
                let candidates = (0..=255u8)
                    .filter(|target| {
                        !known.contains(target)
                            && active.iter().any(|&seed| adjacent(seed, *target))
                    })
                    .collect::<BTreeSet<_>>();
                let budget = if budgets.is_empty() {
                    2
                } else {
                    budgets.iter().sum()
                };
                budgets.push(budget);
                let mut observed = Vec::new();
                loop {
                    let mut buffer = vec![[0; 16]; capacity];
                    let batch = model.generate(&mut buffer).unwrap();
                    let mut feedback = Vec::new();
                    for &target in &buffer[..batch.written] {
                        assert_eq!(&target[..15], &[0; 15]);
                        let value = target[15];
                        assert!(candidates.contains(&value));
                        assert!(known.insert(value));
                        observed.push(value);
                        let positive =
                            outcome == 1 || (outcome == 2 && value.count_ones() % 3 != 0);
                        if positive {
                            active.insert(value);
                            feedback.push(Feedback::Active(target.into()));
                        } else {
                            feedback.push(Feedback::Inactive(target.into()));
                        }
                    }
                    model.apply_feedback(&feedback).unwrap();
                    replay.apply_feedback(&feedback).unwrap();
                    model = SixGraph::decode_model(2, &SixGraph::encode_model(&model).unwrap())
                        .unwrap();
                    if batch.state != GenerationState::Ready {
                        assert_eq!(
                            batch.state,
                            if candidates.is_empty() {
                                GenerationState::Exhausted
                            } else {
                                GenerationState::AwaitingFeedback
                            }
                        );
                        break;
                    }
                }
                assert_eq!(observed.len(), candidates.len().min(budget));
                if candidates.is_empty() {
                    if outcome == 1 {
                        assert_eq!(known.len(), 256);
                    }
                    break;
                }
                model.apply_feedback(&[Feedback::BatchComplete]).unwrap();
                replay.apply_feedback(&[Feedback::BatchComplete]).unwrap();
                assert_eq!(
                    SixGraph::encode_model(&model).unwrap(),
                    SixGraph::encode_model(&replay).unwrap()
                );
                rounds += 1;
                assert!(rounds < 20);
            }
        }
    }
}

#[test]
fn pattern_index_matches_nibble_constraints_at_every_ipv6_position() {
    let baseline = address("2001:db8:abcd:1234:9876:fedc:4567:3210");
    let patterns = (0..32)
        .map(|position| {
            let mut other = baseline;
            other.set_nibble(position, (baseline.get_nibble(position) + 1) % 16);
            Pattern::from_component(&[baseline, other], &[0, 1])
        })
        .collect::<Vec<_>>();
    let index = pattern_index::PatternIndex::new(&patterns);
    for position in 0..32 {
        for value in 0..16 {
            let mut target = baseline;
            target.set_nibble(position, value);
            let expected = patterns
                .iter()
                .enumerate()
                .filter_map(|(id, pattern)| {
                    (0..32)
                        .all(|nibble| {
                            pattern.variable_nibbles.contains(&nibble)
                                || target.get_nibble(nibble) == baseline.get_nibble(nibble)
                        })
                        .then_some(id)
                })
                .collect::<Vec<_>>();
            let mut actual = index.matches(target).collect::<Vec<_>>();
            actual.sort_unstable();
            assert_eq!(actual, expected);
        }
    }
}
