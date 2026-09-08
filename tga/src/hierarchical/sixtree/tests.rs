use super::*;
use crate::GenerationState;
use clap::Parser;
use std::net::Ipv6Addr;

fn address(value: u128) -> Address {
    value.to_be_bytes()
}

fn train(seeds: &[u128]) -> SixTreeModel {
    SixTree::default()
        .train(
            &seeds
                .iter()
                .map(|&value| Observation {
                    address: address(value),
                    active: true,
                })
                .collect::<Vec<_>>(),
        )
        .unwrap()
}

fn frontier(model: &SixTreeModel) -> Vec<Address> {
    model.batch_frontier().cursor().collect()
}

fn feedback(addresses: &[Address]) -> Vec<Feedback> {
    addresses
        .iter()
        .map(|&value| {
            let ip = Ipv6Addr::from(value);
            if matches!(u128::from_be_bytes(value) % 7, 1 | 2) {
                Feedback::Active(ip)
            } else {
                Feedback::Inactive(ip)
            }
        })
        .collect()
}

fn digest(mut addresses: Vec<Address>) -> u64 {
    addresses.sort_unstable();
    addresses
        .into_iter()
        .flatten()
        .fold(14695981039346656037, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(1099511628211)
        })
}

#[test]
fn matches_released_python_prescan_and_dynamic_rounds() {
    // Traces from DHC.py, ScanPre.py and DynamicScan.py with deterministic scan responses.
    let cases = [
        (
            vec![1, 2, 3],
            vec![
                (16, 3922634469851264805),
                (240, 6569405165423464229),
                (3840, 390527482091174181),
                (61440, 9829188798630556965),
            ],
        ),
        (
            (0..4)
                .flat_map(|i| (1..6).map(move |j| 16 * i + j))
                .collect(),
            vec![
                (64, 10470470765215462181),
                (192, 17406705578105295653),
                (3840, 390527482091174181),
                (61440, 9829188798630556965),
            ],
        ),
        (
            (0..3)
                .flat_map(|i| (0..=i).flat_map(move |j| (1..5).map(move |k| 256 * i + 16 * j + k)))
                .collect(),
            vec![
                (96, 10351349973131196453),
                (240, 6569405165423464229),
                (208, 11618928687181283141),
                (3552, 2134408150478205445),
            ],
        ),
    ];
    for (seeds, rounds) in cases {
        let mut model = train(&seeds);
        let mut seen = HashSet::new();
        for (count, expected_digest) in rounds {
            let targets = frontier(&model);
            assert_eq!(targets.len(), count);
            assert_eq!(digest(targets.clone()), expected_digest);
            assert!(targets.iter().all(|&target| seen.insert(target)));
            model.apply_feedback(&feedback(&targets)).unwrap();
        }
    }
}

#[test]
fn promotion_retires_grandchildren_and_inherits_their_history() {
    let mut model = train(&[1]);
    let mut parent = SixTreeNode::new(None, vec![0]);
    parent.children = vec![1, 2];
    let mut child = SixTreeNode::new(Some(0), vec![0]);
    child.targets = RegionSet::from_regions([Region::prefix("::/120".parse().unwrap())]);
    child.scanned = RegionSet::from_regions([Region::singleton(address(1))]);
    child.probed = child.scanned.clone();
    child.active = HashSet::from([address(1)]);
    let mut sibling = SixTreeNode::new(Some(0), vec![0, 1]);
    sibling.children = vec![3];
    let mut grandchild = SixTreeNode::new(Some(2), vec![0, 1, 2]);
    grandchild.targets = RegionSet::from_regions([Region::singleton(address(2))]);
    grandchild.scanned = grandchild.targets.clone();
    grandchild.probed = grandchild.scanned.clone();
    grandchild.active = HashSet::from([address(2)]);
    model.nodes = vec![parent, child, sibling, grandchild];
    model.current_batch = vec![1];
    model.queue = vec![3];
    model.replace_descendants();
    assert_eq!(model.current_batch, vec![0]);
    assert!(model.queue.is_empty());
    assert_eq!(model.nodes[0].probe_count(), 2);
    assert_eq!(model.nodes[0].active.len(), 2);
    assert_eq!(frontier(&model).len(), 254);
    assert!(!frontier(&model).contains(&address(2)));
}

#[test]
fn partial_and_duplicate_feedback_does_not_advance_or_double_count() {
    let mut model = train(&[1, 2, 3]);
    model
        .apply_feedback(&[
            Feedback::Active(1.into()),
            Feedback::Active(1.into()),
            Feedback::BatchComplete,
        ])
        .unwrap();
    model.apply_feedback(&[Feedback::Active(1.into())]).unwrap();
    assert_eq!(model.completed_rounds, 0);
    assert_eq!(model.nodes[0].active.len(), 1);
    assert_eq!(model.nodes[0].probe_count(), 1);
    assert_eq!(frontier(&model).len(), 15);
    model
        .apply_feedback(&[Feedback::Active(1000.into())])
        .unwrap();
    assert_eq!(model.nodes[0].active.len(), 1);
}

#[test]
fn skipped_results_do_not_count_as_probes_even_when_repeated_after_a_response() {
    let mut model = train(&[1]);
    model.apply_feedback(&[Feedback::Active(1.into())]).unwrap();
    model
        .apply_feedback(&[Feedback::Skipped(1.into()), Feedback::Skipped(2.into())])
        .unwrap();
    assert_eq!(model.nodes[0].density(), 1.0);
    assert_eq!(frontier(&model).len(), 14);
    model.apply_feedback(&feedback(&frontier(&model))).unwrap();
    assert_eq!(model.nodes[0].probe_count(), 15);
}

#[test]
fn aliases_do_not_subtract_already_observed_addresses_twice() {
    let mut model = train(&[1]);
    model
        .apply_feedback(&feedback(&(0..8).map(address).collect::<Vec<_>>()))
        .unwrap();
    model
        .apply_feedback(&[Feedback::Aliased("::/125".parse().unwrap())])
        .unwrap();
    assert_eq!(model.completed_rounds, 0);
    assert_eq!(frontier(&model).len(), 8);
    model.apply_feedback(&feedback(&frontier(&model))).unwrap();
    assert_eq!(model.completed_rounds, 1);
    assert_eq!(frontier(&model).len(), 240);
}

#[test]
fn alias_updates_prune_live_generation_without_reissuing_outstanding_targets() {
    let mut model = train(&[1]);
    let mut first = [[0; 16]; 1];
    model.generate(&mut first).unwrap();
    model
        .apply_feedback(&[Feedback::Aliased("::8/125".parse().unwrap())])
        .unwrap();
    let mut output = [[0; 16]; 16];
    let generated = model.generate(&mut output).unwrap();
    assert_eq!(generated.state, GenerationState::AwaitingFeedback);
    assert_eq!(generated.written, 7);
    assert!(!output[..generated.written].contains(&first[0]));
    assert!(
        output[..generated.written]
            .iter()
            .all(|&a| u128::from_be_bytes(a) < 8)
    );
}

#[test]
fn full_ipv6_alias_exhausts_without_enumeration() {
    let mut model = train(&[1]);
    model
        .apply_feedback(&[Feedback::Aliased("::/0".parse().unwrap())])
        .unwrap();
    let result = model.generate(&mut [[0; 16]; 1]).unwrap();
    assert_eq!(result.written, 0);
    assert_eq!(result.state, GenerationState::Exhausted);
}

#[test]
fn serialization_and_cloning_preserve_outstanding_targets_and_cursor() {
    let mut model = train(&[1]);
    let mut first = [[0; 16]; 3];
    model.generate(&mut first).unwrap();
    model.apply_feedback(&feedback(&first[..1])).unwrap();
    let encoded = SixTree::encode_model(&model).unwrap();
    let mut restored = SixTree::decode_model(SixTree::MODEL_VERSION, &encoded).unwrap();
    let mut cloned = model.clone();
    for other in [&mut restored, &mut cloned] {
        let mut output = [[0; 16]; 16];
        let generated = other.generate(&mut output).unwrap();
        assert_eq!(generated.written, 13);
        assert!(
            output[..generated.written]
                .iter()
                .all(|a| !first.contains(a))
        );
        other
            .apply_feedback(&feedback(&output[..generated.written]))
            .unwrap();
        assert_eq!(other.completed_rounds, 0);
        other.apply_feedback(&feedback(&first[1..])).unwrap();
        assert_eq!(other.completed_rounds, 1);
    }
    assert!(SixTree::decode_model(1, &encoded).is_err());
    assert!(SixTree::decode_model(2, &encoded).is_err());
}

#[test]
fn public_generation_is_independent_of_buffer_and_feedback_chunk_sizes() {
    let mut model = train(&[1, 2, 3]);
    let mut seen = HashSet::new();
    for expected in [16, 240, 3840] {
        let mut targets = Vec::new();
        loop {
            let mut output = [[0; 16]; 7];
            let batch = model.generate(&mut output).unwrap();
            targets.extend_from_slice(&output[..batch.written]);
            if batch.state == GenerationState::AwaitingFeedback {
                break;
            }
        }
        assert_eq!(targets.len(), expected);
        assert!(targets.iter().all(|&a| seen.insert(a)));
        for chunk in targets.chunks(11) {
            model.apply_feedback(&feedback(chunk)).unwrap();
        }
    }
}

#[test]
fn config_validation_and_duplicate_seeds() {
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        config: SixTree,
    }
    let defaults = Cli::parse_from(["test"]).config;
    assert_eq!(defaults.leaf_max, None);
    assert_eq!(defaults.batch_percent, 10);
    assert!(defaults.alias_detection);
    assert_eq!(defaults.alias_probe_seed, 0);
    let custom = Cli::parse_from([
        "test",
        "--leaf-max",
        "8",
        "--batch-percent",
        "25",
        "--dedup-seeds",
        "true",
        "--alias-detection",
        "false",
        "--alias-probe-seed",
        "42",
    ])
    .config;
    assert_eq!(custom.leaf_max, Some(8));
    assert_eq!(custom.batch_percent, 25);
    assert!(custom.dedup_seeds);
    assert!(!custom.alias_detection);
    assert_eq!(custom.alias_probe_seed, 42);
    let config: SixTree = serde_json::from_str("{}").unwrap();
    assert_eq!(config.leaf_max, None);
    assert!(config.train(&[]).is_err());
    assert!(
        config
            .train(&[Observation {
                address: address(1),
                active: false
            }])
            .is_err()
    );
    for invalid in [
        SixTree {
            leaf_max: Some(0),
            ..Default::default()
        },
        SixTree {
            batch_percent: 0,
            ..Default::default()
        },
        SixTree {
            batch_percent: 101,
            ..Default::default()
        },
    ] {
        assert!(
            invalid
                .train(&[Observation {
                    address: address(1),
                    active: true
                }])
                .is_err()
        );
    }
    assert_eq!(frontier(&train(&[1; 100])).len(), 16);
    assert!(train(&[1]).generate(&mut []).is_err());
}

#[test]
fn overlapping_duplicate_seed_branches_keep_density_at_most_one() {
    let mut seeds = vec![1; 16];
    seeds.push(2);
    let mut model = train(&seeds);
    let targets = frontier(&model);
    assert_eq!(targets.len(), 16);
    model
        .apply_feedback(
            &targets
                .iter()
                .map(|&a| Feedback::Active(a.into()))
                .collect::<Vec<_>>(),
        )
        .unwrap();
    assert_eq!(model.current_batch, vec![0]);
    assert_eq!(model.nodes[0].probe_count(), 16);
    assert_eq!(model.nodes[0].active.len(), 16);
    assert_eq!(model.nodes[0].density(), 1.0);
    assert_eq!(frontier(&model).len(), 240);
}

#[test]
fn empty_transport_boundaries_preserve_pending_feedback() {
    let mut model = train(&[1]);
    model.apply_feedback(&[]).unwrap();
    model.apply_feedback(&[Feedback::BatchComplete]).unwrap();
    assert_eq!(model.completed_rounds, 0);
    assert_eq!(frontier(&model).len(), 16);
    let mut output = [[0; 16]; 17];
    assert_eq!(
        model.generate(&mut output).unwrap().state,
        GenerationState::AwaitingFeedback
    );
    model.apply_feedback(&[Feedback::BatchComplete]).unwrap();
    assert_eq!(model.generate(&mut output).unwrap().written, 0);
    assert_eq!(model.completed_rounds, 0);
}

#[test]
fn partial_alias_feedback_preserves_actual_probe_history() {
    let mut model = train(&[1]);
    model.apply_feedback(&[Feedback::Active(0.into())]).unwrap();
    model
        .apply_feedback(&[Feedback::Aliased("::/125".parse().unwrap())])
        .unwrap();
    model.apply_feedback(&feedback(&frontier(&model))).unwrap();
    assert_eq!(model.nodes[0].probe_count(), 9);
    assert!(model.nodes[0].active.contains(&address(0)));
}

#[test]
fn released_reference_matches_training_and_dynamic_rounds_in_each_base() {
    #[derive(Deserialize)]
    struct NodeSnapshot {
        parent: Option<usize>,
        children: Vec<usize>,
        stack: Vec<usize>,
    }
    #[derive(Deserialize)]
    struct RoundSnapshot {
        count: usize,
        fingerprint: u64,
        active: usize,
    }
    #[derive(Deserialize)]
    struct Case {
        name: String,
        base: u8,
        seeds: Vec<String>,
        nodes: Vec<NodeSnapshot>,
        rounds: Vec<RoundSnapshot>,
    }
    let cases: Vec<Case> = serde_json::from_str(include_str!(
        "../../../tests/fixtures/sixtree/reference.json"
    ))
    .unwrap();
    for case in cases {
        let observations: Vec<_> = case
            .seeds
            .iter()
            .map(|seed| Observation {
                address: u128::from_str_radix(seed, 16).unwrap().to_be_bytes(),
                active: true,
            })
            .collect();
        let mut model = SixTree {
            base: case.base,
            ..Default::default()
        }
        .train(&observations)
        .unwrap();
        assert_eq!(model.nodes.len(), case.nodes.len(), "{}", case.name);
        for (actual, expected) in model.nodes.iter().zip(&case.nodes) {
            assert_eq!(actual.parent, expected.parent, "{}", case.name);
            assert_eq!(actual.children, expected.children, "{}", case.name);
            assert_eq!(actual.dimension_stack, expected.stack, "{}", case.name);
        }
        let mut seen = HashSet::new();
        for (index, expected) in case.rounds.iter().enumerate() {
            if expected.count == 0 {
                assert!(
                    model.completed_rounds > index,
                    "{} round {index}",
                    case.name
                );
                continue;
            }
            assert_eq!(model.completed_rounds, index, "{}", case.name);
            assert!(
                matches!(model.phase, SearchPhase::Scanning),
                "{} round {index}",
                case.name
            );
            let mut targets = Vec::new();
            loop {
                let mut output = [[0; 16]; 127];
                let batch = model.generate(&mut output).unwrap();
                targets.extend_from_slice(&output[..batch.written]);
                if batch.state == GenerationState::AwaitingFeedback {
                    break;
                }
                assert_ne!(batch.state, GenerationState::Exhausted);
            }
            assert_eq!(targets.len(), expected.count, "{} round {index}", case.name);
            assert_eq!(
                digest(targets.clone()),
                expected.fingerprint,
                "{} round {index}",
                case.name
            );
            assert!(
                targets.iter().all(|&address| seen.insert(address)),
                "{} round {index}",
                case.name
            );
            let feedback = feedback(&targets);
            assert_eq!(
                feedback
                    .iter()
                    .filter(|item| matches!(item, Feedback::Active(_)))
                    .count(),
                expected.active
            );
            for chunk in feedback.chunks(109) {
                model.apply_feedback(chunk).unwrap();
            }
        }
    }
}

#[test]
fn paper_defaults_treat_seeds_as_a_set_and_leaf_capacity_as_the_base() {
    for base in [2, 4, 8, 16, 32] {
        let config = SixTree {
            base,
            ..Default::default()
        };
        let observations: Vec<_> = (0..=u128::from(base))
            .map(|value| Observation {
                address: address(value),
                active: true,
            })
            .collect();
        let original = config.train(&observations).unwrap();
        let mut repeated = observations.repeat(4);
        repeated.reverse();
        let duplicate = config.train(&repeated).unwrap();
        assert!(original.nodes.len() > 1);
        assert_eq!(original.nodes.len(), duplicate.nodes.len());
        assert_eq!(frontier(&original), frontier(&duplicate));
        for (left, right) in original.nodes.iter().zip(&duplicate.nodes) {
            assert_eq!(left.dimension_stack, right.dimension_stack);
        }
        let small = config.train(&observations[..usize::from(base)]).unwrap();
        assert_eq!(small.nodes.len(), 1);
    }
    for base in [0, 1, 3, 64, 255] {
        assert!(
            SixTree {
                base,
                ..Default::default()
            }
            .train(&[Observation {
                address: address(1),
                active: true
            }])
            .is_err()
        );
    }
}

#[test]
fn omitted_leading_bits_stay_fixed_and_separate_seed_scopes() {
    for (base, omitted) in [(8, 2), (32, 3)] {
        let left = 1u128 << (128 - omitted);
        let right = 2u128 << (128 - omitted);
        let observations: Vec<_> = [left | 1, right | 2]
            .map(|value| Observation {
                address: address(value),
                active: true,
            })
            .into();
        let mut model = SixTree {
            base,
            alias_detection: false,
            ..Default::default()
        }
        .train(&observations)
        .unwrap();
        assert_eq!(
            model
                .nodes
                .iter()
                .filter(|node| node.parent.is_none())
                .count(),
            2
        );
        assert_eq!(frontier(&model).len(), usize::from(base) * 2);
        for value in frontier(&model) {
            assert!([left, right].contains(&model.nodes[0].layout.scope(value)));
        }
        let excluded = Ipv6Prefix::new(left.into(), omitted).unwrap();
        model
            .apply_feedback(&[Feedback::Aliased(excluded)])
            .unwrap();
        assert_eq!(frontier(&model).len(), usize::from(base));
        model.apply_feedback(&feedback(&frontier(&model))).unwrap();
        assert!(
            frontier(&model)
                .iter()
                .all(|&value| model.nodes[0].layout.scope(value) == right)
        );
    }
}

#[test]
fn exact_buffer_boundaries_request_feedback_before_another_probe_round() {
    let mut model = train(&[1]);
    let mut buffer = [[0; 16]; 16];
    let generated = model.generate(&mut buffer).unwrap();
    assert_eq!(generated.written, 16);
    assert_eq!(generated.state, GenerationState::AwaitingFeedback);
    model.apply_feedback(&feedback(&buffer)).unwrap();
    let mut buffer = [[0; 16]; 240];
    assert_eq!(
        model.generate(&mut buffer).unwrap().state,
        GenerationState::AwaitingFeedback
    );
}

#[test]
fn normal_feedback_preserves_positive_evidence_across_partial_updates() {
    let mut model = train(&[1]);
    model
        .apply_feedback(&[Feedback::Inactive(1.into()), Feedback::Skipped(2.into())])
        .unwrap();
    assert_eq!(model.nodes[0].probe_count(), 1);
    model
        .apply_feedback(&[Feedback::Active(1.into()), Feedback::Inactive(2.into())])
        .unwrap();
    assert_eq!(model.nodes[0].probe_count(), 2);
    assert_eq!(model.nodes[0].active.len(), 1);
    model
        .apply_feedback(&[Feedback::Inactive(1.into()), Feedback::Skipped(2.into())])
        .unwrap();
    assert_eq!(model.nodes[0].density(), 0.5);
    assert_eq!(frontier(&model).len(), 14);
}
