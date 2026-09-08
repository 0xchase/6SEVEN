use super::model::*;
use super::pattern::*;
use super::tree::*;
use super::*;
use crate::{Address, Feedback};
use clap::Parser;
use std::collections::BTreeSet;
use std::collections::HashSet;
use std::net::Ipv6Addr;
use std::str::FromStr;

fn parse_address(address: &str) -> Address {
    Ipv6Addr::from_str(address).unwrap().octets()
}

fn full_length_pattern(bits_per_dimension: usize, leading_digits: &[u16]) -> TargetPattern {
    let mut digits = vec![0u16; 128 / bits_per_dimension];
    for (idx, digit) in leading_digits.iter().enumerate() {
        digits[idx] = *digit;
    }
    TargetPattern { digits }
}

fn address_with_leading_digits(bits_per_dimension: usize, leading_digits: &[u8]) -> Address {
    let mut digits = vec![0u8; 128 / bits_per_dimension];
    for (idx, digit) in leading_digits.iter().enumerate() {
        digits[idx] = *digit;
    }
    vector_to_address(&digits, bits_per_dimension)
}

fn collect_frontier_addresses(model: &DetModel) -> Vec<Address> {
    let mut model = model.clone();
    let mut result = Vec::new();
    loop {
        let mut output = [[0; 16]; 127];
        let generated = crate::TargetModel::generate(&mut model, &mut output).unwrap();
        result.extend_from_slice(&output[..generated.written]);
        if generated.state != crate::GenerationState::Ready {
            break;
        }
    }
    result
}

fn learn_det_model(seeds: &[Address], leaf_max: usize) -> DetModel {
    let config = Det {
        delta_base: 16,
        leaf_max,
    };
    let observations = seeds
        .iter()
        .copied()
        .map(|address| Observation {
            address,
            active: true,
        })
        .collect::<Vec<_>>();
    config.train(&observations).unwrap()
}

#[derive(Parser)]
struct DetCli {
    #[command(flatten)]
    det: Det,
}

#[test]
fn input_vectors_are_sorted_like_reference_inputaddrs() {
    let seeds = [
        address_with_leading_digits(4, &[0x2, 0x0]),
        address_with_leading_digits(4, &[0x0, 0xf]),
        address_with_leading_digits(4, &[0x1, 0x0]),
        address_with_leading_digits(4, &[0x0, 0x1]),
    ];

    let vectors = sorted_seed_vectors(&seeds, 4)
        .into_iter()
        .map(|vector| vector[..2].to_vec())
        .collect::<Vec<_>>();

    assert_eq!(
        vectors,
        vec![
            vec![0x0, 0x1],
            vec![0x0, 0xf],
            vec![0x1, 0x0],
            vec![0x2, 0x0]
        ]
    );
}

#[test]
fn partition_order_matches_reference_first_seen_dictionary_order() {
    let vectors = vec![vec![1, 2], vec![0, 0], vec![1, 0], vec![0, 1]];
    let mut seed_indices = vec![0, 1, 2, 3];
    let mut scratch = Vec::new();

    let ranges =
        partition_seed_indices_by_dimension(&vectors, &mut seed_indices, 1, 3, &mut scratch);

    assert_eq!(ranges, vec![(0, 1), (1, 3), (3, 4)]);
    assert_eq!(seed_indices, vec![0, 1, 2, 3]);
}

#[test]
fn cli_defaults_match_reference_settings() {
    let cli = DetCli::parse_from(["det-test"]);
    assert_eq!(cli.det.delta_base, 16);
    assert_eq!(cli.det.leaf_max, 16);
    assert_eq!(resolve_bits_per_dim(cli.det.delta_base).unwrap(), 4);
}

#[test]
fn ds_ordering_pushes_parent_split_dim() {
    let vectors = vec![vec![0, 0, 0], vec![0, 1, 0], vec![1, 0, 0], vec![1, 1, 0]];
    let mut seed_indices = (0..vectors.len()).collect::<Vec<_>>();
    let mut scratch = Vec::new();
    let mut root = BuildNode::new(0, vectors.len());
    split_node(
        &mut root,
        &vectors,
        &mut seed_indices,
        &mut scratch,
        1,
        3,
        2,
    );
    assert_eq!(root.split_dimension, Some(0));

    initialize_dimension_stacks(&mut root, &vectors, &seed_indices, &[], None, 3);
    assert_eq!(root.children.len(), 2);
    for child in &root.children {
        assert_eq!(child.children.len(), 2);
        for leaf in &child.children {
            assert_eq!(leaf.dimension_stack, vec![2, 0, 1]);
        }
    }
}

#[test]
fn initial_frontier_is_first_expansion_only() {
    let seeds = ["2001:db8::1", "2001:db8::2", "2001:db8::3"]
        .into_iter()
        .map(parse_address)
        .collect::<Vec<_>>();

    let config = Det {
        delta_base: 16,
        leaf_max: 16,
    };

    let observations = seeds
        .iter()
        .copied()
        .map(|address| Observation {
            address,
            active: true,
        })
        .collect::<Vec<_>>();

    let model = config.train(&observations).unwrap();

    let frontier = collect_frontier_addresses(&model);
    assert_eq!(frontier.len(), 16);
    assert!(frontier.contains(&parse_address("2001:db8::0")));
    assert!(frontier.contains(&parse_address("2001:db8::f")));
    assert!(!frontier.contains(&parse_address("2001:db8::10")));
}

#[test]
fn feedback_advances_to_next_dynamic_batch() {
    let seeds = ["2001:db8::1", "2001:db8::2", "2001:db8::3"]
        .into_iter()
        .map(parse_address)
        .collect::<Vec<_>>();
    let mut model = learn_det_model(&seeds, 16);
    let frontier = collect_frontier_addresses(&model);

    let feedback = frontier
        .iter()
        .copied()
        .map(|address| Feedback::Inactive(Ipv6Addr::from(address)))
        .collect::<Vec<_>>();

    let mut feedback = feedback;
    feedback.push(Feedback::BatchComplete);
    crate::TargetModel::apply_feedback(&mut model, &feedback).unwrap();
    let next = collect_frontier_addresses(&model)[0];

    assert_eq!(model.completed_rounds, 1);
    assert_eq!(next, parse_address("2001:db8::10"));
}

#[test]
fn pattern_address_iter_expands_patterns_in_order() {
    let patterns = [
        full_length_pattern(1, &[WILDCARD, 0, 1]),
        full_length_pattern(1, &[0, 1, WILDCARD]),
    ];

    let actual = patterns
        .iter()
        .flat_map(|pattern| PatternExpansion::new(pattern, 2, 1))
        .collect::<Vec<_>>();

    assert_eq!(
        actual,
        vec![
            address_with_leading_digits(1, &[0, 0, 1]),
            address_with_leading_digits(1, &[1, 0, 1]),
            address_with_leading_digits(1, &[0, 1, 0]),
            address_with_leading_digits(1, &[0, 1, 1]),
        ]
    );
}

#[test]
fn model_frontier_is_globally_disjoint_across_leaves() {
    let nodes = vec![
        DetNode {
            parent: None,
            children: Vec::new(),
            dimension_stack: Vec::new(),
            target_patterns: vec![full_length_pattern(1, &[WILDCARD, 0, 0, 0])],
            scanned_addresses: Vec::new(),
            skipped_addresses: HashSet::new(),
            active_hits: 0,
            active_density: 0.0,
        },
        DetNode {
            parent: None,
            children: Vec::new(),
            dimension_stack: Vec::new(),
            target_patterns: vec![full_length_pattern(1, &[0, WILDCARD, 0, 0])],
            scanned_addresses: Vec::new(),
            skipped_addresses: HashSet::new(),
            active_hits: 0,
            active_density: 0.0,
        },
    ];

    let model = DetModel {
        nodes,
        current_batch: vec![0, 1],
        bits_per_dimension: 1,
        ..DetModel::default()
    };
    let addresses = collect_frontier_addresses(&model)
        .into_iter()
        .collect::<BTreeSet<_>>();

    assert_eq!(addresses.len(), 3);
    assert!(addresses.contains(&address_with_leading_digits(1, &[0, 0, 0, 0])),);
    assert!(addresses.contains(&address_with_leading_digits(1, &[1, 0, 0, 0])),);
    assert!(addresses.contains(&address_with_leading_digits(1, &[0, 1, 0, 0])),);
}

#[test]
fn stream_ends_after_current_batch_without_feedback() {
    let seeds = ["2001:db8::1", "2001:db8::2", "2001:db8::3"]
        .into_iter()
        .map(parse_address)
        .collect::<Vec<_>>();
    let model = learn_det_model(&seeds, 16);

    let generated = collect_frontier_addresses(&model);

    assert_eq!(generated.len(), 16);
    assert!(!generated.contains(&parse_address("2001:db8::10")));
}
fn complete(model: &mut DetModel, addresses: &[Address], active: bool) {
    let mut feedback = addresses
        .iter()
        .map(|&address| {
            if active {
                Feedback::Active(address.into())
            } else {
                Feedback::Inactive(address.into())
            }
        })
        .collect::<Vec<_>>();
    feedback.push(Feedback::BatchComplete);
    crate::TargetModel::apply_feedback(model, &feedback).unwrap();
}

#[derive(Deserialize)]
struct ReferenceFixture {
    cases: Vec<ReferenceCase>,
}

#[derive(Deserialize)]
struct ReferenceCase {
    name: String,
    bits: usize,
    leaf_max: usize,
    seeds: Vec<String>,
    tree: Vec<ReferenceNode>,
    rounds: Vec<ReferenceRound>,
}

#[derive(Deserialize)]
struct ReferenceNode {
    id: usize,
    parent: Option<usize>,
    children: Vec<usize>,
    stack: Vec<usize>,
    patterns: Vec<Vec<u16>>,
    scanned: Vec<[String; 2]>,
    hits: usize,
    density: f64,
}

#[derive(Deserialize)]
struct ReferenceRound {
    batch: Vec<ReferenceNode>,
    queue: Vec<ReferenceNode>,
    targets: Vec<[String; 2]>,
    active: Vec<[String; 2]>,
}

fn reference_addresses(ranges: &[[String; 2]]) -> BTreeSet<Address> {
    ranges
        .iter()
        .flat_map(|[start, end]| {
            let start = u128::from_str_radix(start, 16).unwrap();
            let end = u128::from_str_radix(end, 16).unwrap();
            (start..=end).map(u128::to_be_bytes)
        })
        .collect()
}

fn compare_reference_nodes(model: &DetModel, expected: &[ReferenceNode], context: &str) {
    for node in expected {
        let actual = &model.nodes[node.id];
        assert_eq!(
            actual.parent, node.parent,
            "{context}, node {} parent",
            node.id
        );
        assert_eq!(
            actual.children, node.children,
            "{context}, node {} children",
            node.id
        );
        assert_eq!(
            actual.dimension_stack, node.stack,
            "{context}, node {} stack",
            node.id
        );
        let patterns: Vec<_> = actual
            .target_patterns
            .iter()
            .map(|p| p.digits.clone())
            .collect();
        assert_eq!(
            patterns, node.patterns,
            "{context}, node {} patterns",
            node.id
        );
        assert_eq!(
            actual
                .scanned_addresses
                .iter()
                .copied()
                .collect::<BTreeSet<_>>(),
            reference_addresses(&node.scanned),
            "{context}, node {} scanned",
            node.id
        );
        assert_eq!(
            actual.active_hits, node.hits,
            "{context}, node {} hits",
            node.id
        );
        assert!(
            (actual.active_density - node.density).abs() < 1e-14,
            "{context}, node {} density",
            node.id
        );
    }
}

#[test]
fn training_and_dynamic_rounds_match_reference_with_paper_corrections() {
    let fixture: ReferenceFixture =
        serde_json::from_str(include_str!("../../../tests/fixtures/det/reference.json")).unwrap();
    for case in fixture.cases {
        let observations: Vec<_> = case
            .seeds
            .iter()
            .map(|s| Observation {
                address: u128::from_str_radix(s, 16).unwrap().to_be_bytes(),
                active: true,
            })
            .collect();
        let mut model = Det {
            delta_base: 1 << case.bits,
            leaf_max: case.leaf_max,
        }
        .train(&observations)
        .unwrap();
        assert_eq!(
            model.nodes.len(),
            case.tree.len(),
            "{} tree size",
            case.name
        );
        compare_reference_nodes(&model, &case.tree, &case.name);
        let mut seen = BTreeSet::new();
        for (index, round) in case.rounds.iter().enumerate() {
            let context = format!("{}, round {index}", case.name);
            assert_eq!(
                model.current_batch,
                round.batch.iter().map(|n| n.id).collect::<Vec<_>>(),
                "{context} batch"
            );
            assert_eq!(
                model.queue,
                round.queue.iter().map(|n| n.id).collect::<Vec<_>>(),
                "{context} queue"
            );
            compare_reference_nodes(&model, &round.batch, &context);
            compare_reference_nodes(&model, &round.queue, &context);
            let expected = reference_addresses(&round.targets);
            let active = reference_addresses(&round.active);
            let mut targets = Vec::new();
            let mut buffer = vec![[0; 16]; [1, 17, 1024][index % 3]];
            let mut checkpointed = false;
            loop {
                let generated = crate::TargetModel::generate(&mut model, &mut buffer).unwrap();
                targets.extend_from_slice(&buffer[..generated.written]);
                let mut feedback: Vec<_> = buffer[..generated.written]
                    .iter()
                    .map(|&a| {
                        if active.contains(&a) {
                            Feedback::Active(a.into())
                        } else {
                            Feedback::Inactive(a.into())
                        }
                    })
                    .collect();
                let complete = generated.state == crate::GenerationState::AwaitingFeedback;
                if complete {
                    feedback.push(Feedback::BatchComplete);
                }
                crate::TargetModel::apply_feedback(&mut model, &feedback).unwrap();
                if complete {
                    break;
                }
                assert_eq!(generated.state, crate::GenerationState::Ready, "{context}");
                if !checkpointed {
                    model =
                        Det::decode_model(Det::MODEL_VERSION, &Det::encode_model(&model).unwrap())
                            .unwrap();
                    checkpointed = true;
                }
            }
            assert_eq!(
                targets.iter().copied().collect::<BTreeSet<_>>(),
                expected,
                "{context} targets"
            );
            assert_eq!(targets.len(), expected.len(), "{context} unique targets");
            assert!(targets.iter().all(|a| seen.insert(*a)), "{context} rescans");
        }
    }
}

#[test]
fn partial_feedback_retires_only_observed_addresses() {
    let mut model = learn_det_model(&[parse_address("2001:db8::1")], 16);
    let initial = collect_frontier_addresses(&model);
    complete(&mut model, &initial[..3], true);
    assert_eq!(model.completed_rounds, 0);
    assert_eq!(collect_frontier_addresses(&model), initial[3..]);
    assert_eq!(model.nodes[0].active_density, 1.0);
    complete(&mut model, &initial[3..], false);
    assert_eq!(model.completed_rounds, 1);
    assert_eq!(model.nodes[0].active_hits, 3);
    assert_eq!(model.nodes[0].active_density, 3.0 / 16.0);
    assert_eq!(collect_frontier_addresses(&model).len(), 240);
}

#[test]
fn checkpoint_preserves_cursor_and_pending_feedback() {
    let mut model = learn_det_model(&[parse_address("2001:db8::1")], 16);
    let mut output = [[0; 16]; 5];
    crate::TargetModel::generate(&mut model, &mut output).unwrap();
    let feedback = output.map(|a| Feedback::Active(a.into()));
    crate::TargetModel::apply_feedback(&mut model, &feedback).unwrap();
    let encoded = Det::encode_model(&model).unwrap();
    let mut restored = Det::decode_model(Det::MODEL_VERSION, &encoded).unwrap();
    assert_eq!(
        collect_frontier_addresses(&model),
        collect_frontier_addresses(&restored)
    );
    let rest = collect_frontier_addresses(&restored);
    assert_eq!(rest.len(), 11);
    complete(&mut restored, &rest, false);
    assert_eq!(restored.nodes[0].active_hits, 5);
    assert_eq!(restored.completed_rounds, 1);
    assert!(Det::decode_model(1, &encoded).is_err());
}

#[test]
fn replay_without_generation_matches_live_feedback() {
    let mut live = learn_det_model(&[parse_address("2001:db8::1")], 16);
    let mut replay = live.clone();
    let mut output = [[0; 16]; 32];
    let generated = crate::TargetModel::generate(&mut live, &mut output).unwrap();
    let addresses = &output[..generated.written];
    complete(&mut live, addresses, true);
    complete(&mut replay, addresses, true);
    assert_eq!(
        collect_frontier_addresses(&live),
        collect_frontier_addresses(&replay)
    );
}

#[test]
fn duplicate_and_skipped_feedback_do_not_inflate_density() {
    let mut model = learn_det_model(&[parse_address("2001:db8::1")], 16);
    let targets = collect_frontier_addresses(&model);
    let mut feedback = targets
        .iter()
        .map(|&a| Feedback::Skipped(a.into()))
        .collect::<Vec<_>>();
    feedback.extend([
        Feedback::Inactive(targets[0].into()),
        Feedback::Active(targets[0].into()),
        Feedback::Active(targets[0].into()),
        Feedback::BatchComplete,
    ]);
    crate::TargetModel::apply_feedback(&mut model, &feedback).unwrap();
    assert_eq!(model.nodes[0].active_hits, 1);
    assert_eq!(model.nodes[0].active_density, 1.0);
    assert_eq!(model.nodes[0].skipped_addresses.len(), 15);
    assert_eq!(collect_frontier_addresses(&model).len(), 240);
}

#[test]
fn feedback_does_not_expand_huge_target_regions() {
    let mut model = learn_det_model(&[[0; 16]], 16);
    model.nodes[0].target_patterns = vec![TargetPattern {
        digits: vec![WILDCARD; 32],
    }];
    complete(&mut model, &[[0; 16]], false);
    assert_eq!(model.nodes[0].scanned_addresses.len(), 1);
    assert_eq!(model.completed_rounds, 0);
}

#[test]
fn full_width_pattern_ends_by_carry_without_cardinality_overflow() {
    let pattern = TargetPattern {
        digits: vec![WILDCARD; 32],
    };
    assert_eq!(pattern.size(16), None);
    let mut expansion = PatternExpansion::new(&pattern, 16, 4);
    assert_eq!(expansion.next(), Some([0; 16]));
    expansion.address = u128::MAX;
    for field in &mut expansion.wildcard_fields {
        field.value = 15;
    }
    assert_eq!(expansion.next(), Some([255; 16]));
    assert_eq!(expansion.next(), None);
    assert_eq!(expansion.next(), None);
}

#[test]
fn fully_observed_root_exhausts_cleanly() {
    let mut model = learn_det_model(&[[0; 16]], 16);
    model.nodes[0].dimension_stack.clear();
    let targets = collect_frontier_addresses(&model);
    complete(&mut model, &targets, false);
    let mut output = [[0; 16]; 1];
    let generated = crate::TargetModel::generate(&mut model, &mut output).unwrap();
    assert_eq!(generated.written, 0);
    assert_eq!(generated.state, crate::GenerationState::Exhausted);
}

#[test]
fn all_supported_bases_expand_one_dimension() {
    for base in [2, 4, 16, 256] {
        let model = Det {
            delta_base: base,
            leaf_max: 16,
        }
        .train(&[Observation {
            address: [0; 16],
            active: true,
        }])
        .unwrap();
        assert_eq!(collect_frontier_addresses(&model).len(), base);
    }
    for base in [0, 1, 3, 8, 32, 512] {
        assert!(resolve_bits_per_dim(base).is_err());
    }
    assert!(
        Det {
            delta_base: 16,
            leaf_max: 0
        }
        .train(&[Observation {
            address: [0; 16],
            active: true
        }])
        .is_err()
    );
    assert!(
        Det {
            delta_base: 16,
            leaf_max: 16
        }
        .train(&[])
        .is_err()
    );
}

#[test]
fn entropy_chooses_smallest_nonzero_and_leftmost_ties() {
    let vectors = vec![
        vec![0, 0, 0, 0],
        vec![0, 0, 0, 1],
        vec![0, 0, 0, 0],
        vec![0, 1, 1, 1],
    ];
    assert_eq!(find_split_dimension(&vectors, &[0, 1, 2, 3], 4, 2), Some(1));
}

#[test]
fn promotion_retires_frontier_nodes_below_an_unexpanded_sibling() {
    let template = learn_det_model(&[[0; 16]], 16).nodes.remove(0);
    let mut nodes = vec![template; 5];
    nodes[0].children = vec![1, 2];
    nodes[2].children = vec![3, 4];
    nodes[1].parent = Some(0);
    nodes[2].parent = Some(0);
    nodes[3].parent = Some(2);
    nodes[4].parent = Some(2);
    for (id, value) in [(1, 1u128), (3, 2), (4, 3)] {
        nodes[id].scanned_addresses = vec![value.to_be_bytes()];
        nodes[id].active_hits = 1;
    }
    let mut model = DetModel {
        nodes,
        current_batch: vec![1],
        queue: vec![3, 4],
        ..DetModel::default()
    };
    model.replace_descendants();
    assert_eq!(model.current_batch, vec![0]);
    assert!(model.queue.is_empty());
    assert_eq!(model.nodes[0].scanned_addresses.len(), 3);
    assert_eq!(model.nodes[0].active_hits, 3);
    assert_eq!(model.nodes[0].active_density, 1.0);
    for id in [1, 3, 4] {
        assert!(model.nodes[id].scanned_addresses.is_empty());
        assert!(model.nodes[id].target_patterns.is_empty());
    }
    assert_eq!(collect_frontier_addresses(&model).len(), 13);
}

#[test]
fn invalid_feedback_does_not_poison_later_batches() {
    let mut model = learn_det_model(&[[0; 16]], 16);
    let invalid = Feedback::Active(parse_address("2001:db8::1").into());
    assert!(crate::TargetModel::apply_feedback(&mut model, &[invalid]).is_err());
    let targets = collect_frontier_addresses(&model);
    complete(&mut model, &targets, true);
    assert_eq!(model.completed_rounds, 1);
    assert_eq!(model.nodes[0].active_hits, 16);
}

#[test]
fn empty_buffer_does_not_advance_generation() {
    let mut model = learn_det_model(&[[0; 16]], 16);
    let expected = collect_frontier_addresses(&model);
    assert!(crate::TargetModel::generate(&mut model, &mut []).is_err());
    assert_eq!(collect_frontier_addresses(&model), expected);
}

#[test]
fn equal_entropy_histograms_choose_the_leftmost_dimension() {
    let vectors = vec![
        vec![0, 0],
        vec![1, 0],
        vec![1, 1],
        vec![2, 1],
        vec![2, 1],
        vec![2, 2],
    ];
    let indices: Vec<_> = (0..vectors.len()).collect();
    assert_eq!(find_split_dimension(&vectors, &indices, 2, 4), Some(0));
}

#[test]
fn different_histograms_with_equal_entropy_choose_the_leftmost_dimension() {
    let left = [0, 1, 2, 3, 4, 5, 5, 5, 5];
    let right = [0, 1, 1, 2, 2, 3, 3, 4, 4];
    let vectors: Vec<_> = left
        .into_iter()
        .zip(right)
        .map(|(a, b)| vec![a, b])
        .collect();
    let indices: Vec<_> = (0..vectors.len()).collect();
    assert_eq!(find_split_dimension(&vectors, &indices, 2, 16), Some(0));
}

#[test]
fn parallel_training_preserves_entropy_ties_and_dimension_stacks() {
    let small: Vec<_> = [0, 4, 5, 9, 9, 10]
        .map(|value| Observation {
            address: (value as u128).to_be_bytes(),
            active: true,
        })
        .into_iter()
        .collect();
    let large: Vec<_> = small
        .iter()
        .flat_map(|seed| std::iter::repeat_n(*seed, 683))
        .collect();
    let serial = Det {
        delta_base: 4,
        leaf_max: 2,
    }
    .train(&small)
    .unwrap();
    let parallel = Det {
        delta_base: 4,
        leaf_max: 1366,
    }
    .train(&large)
    .unwrap();
    assert_eq!(serial.nodes.len(), parallel.nodes.len());
    for (left, right) in serial.nodes.iter().zip(&parallel.nodes) {
        assert_eq!(left.parent, right.parent);
        assert_eq!(left.children, right.children);
        assert_eq!(left.dimension_stack, right.dimension_stack);
        assert_eq!(left.target_patterns, right.target_patterns);
    }
    assert_eq!(
        collect_frontier_addresses(&serial),
        collect_frontier_addresses(&parallel)
    );
}

#[test]
fn inactive_observations_do_not_change_the_seed_tree() {
    let seed = Observation {
        address: parse_address("2001:db8::1"),
        active: true,
    };
    let config = Det {
        delta_base: 16,
        leaf_max: 1,
    };
    let expected = config.train(&[seed]).unwrap();
    let actual = config
        .train(&[
            Observation {
                address: [255; 16],
                active: false,
            },
            seed,
        ])
        .unwrap();
    assert_eq!(
        collect_frontier_addresses(&actual),
        collect_frontier_addresses(&expected)
    );
    assert_eq!(actual.nodes.len(), expected.nodes.len());
}

#[test]
fn retraining_with_discovered_seeds_can_change_the_split_dimension() {
    let mut observations: Vec<_> = [0, 1, 2, 3]
        .map(|n| Observation {
            address: (n as u128).to_be_bytes(),
            active: true,
        })
        .into_iter()
        .collect();
    let config = Det {
        delta_base: 16,
        leaf_max: 2,
    };
    let original = config.train(&observations).unwrap();
    observations.extend([16, 32].map(|n| Observation {
        address: (n as u128).to_be_bytes(),
        active: true,
    }));
    let retrained = config.train(&observations).unwrap();
    assert_eq!(original.nodes[0].children.len(), 4);
    assert_eq!(retrained.nodes[0].children.len(), 3);
    assert_ne!(
        original.nodes[0].dimension_stack,
        retrained.nodes[0].dimension_stack
    );
}
