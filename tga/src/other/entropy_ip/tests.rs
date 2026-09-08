use super::mining::mine_segment_states;
use super::segmentation::define_segments;
use super::train::bnfinder_ordered_parent_candidates;
use crate::{Address, TgaError};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::HashMap;
use std::net::Ipv6Addr;
use std::str::FromStr;

use super::types::{BayesianNetwork, CptEntry, Segment, SegmentState};
use super::*;
use super::{encode::sample_bnf_rows, types::build_segment_lookup};
use crate::{Algorithm, Observation};

fn load_ipv6_seeds(path: &str) -> Vec<Address> {
    use std::io::BufRead;
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("Cannot open {}: {}", path, e));
    let reader = std::io::BufReader::new(file);
    reader
        .lines()
        .filter_map(|line| {
            let line = line.ok()?;
            let line = line.trim().to_string();
            if line.is_empty() {
                return None;
            }
            Some(Ipv6Addr::from_str(&line).unwrap().octets())
        })
        .collect()
}

fn single(value: u128) -> SegmentState {
    SegmentState::Single(value)
}

fn range(min: u128, max: u128) -> SegmentState {
    SegmentState::Range(min, max)
}

fn legacy_next_address<R: rand::Rng + ?Sized>(
    model: &EntropyIpModel,
    rng: &mut R,
) -> Result<Address, TgaError> {
    if model.segments.is_empty() {
        return Err(TgaError::Generation(
            "Entropy/IP model has no reachable targets".into(),
        ));
    }

    let mut address_nybbles = [0u8; 32];
    let mut sampled_states_indices = vec![0usize; model.segments.len()];

    for (segment_idx, segment) in model.segments.iter().enumerate() {
        let num_states = model
            .segment_cardinalities
            .get(segment_idx)
            .copied()
            .unwrap_or(segment.states.len());
        if num_states == 0 {
            return Err(TgaError::Generation(format!(
                "Entropy/IP segment {segment_idx} has no encodable states"
            )));
        }

        let parent_state_key = model.network.parents[segment_idx]
            .iter()
            .map(|&parent_idx| sampled_states_indices[parent_idx])
            .collect::<Vec<_>>();
        let entry = model.network.cpts[segment_idx].get(parent_state_key.as_slice());
        let bn_state_idx = super::generate::sample_state(entry, num_states, rng);
        sampled_states_indices[segment_idx] = bn_state_idx;
        let segment_state_idx = *model
            .bn_values
            .get(segment_idx)
            .and_then(|codes| codes.get(bn_state_idx))
            .ok_or_else(|| {
                TgaError::Generation(format!(
                    "Entropy/IP segment {segment_idx} is missing a BN value mapping for state {bn_state_idx}"
                ))
            })?;
        let value = sample_state_value(segment, segment_state_idx, rng)?;
        write_segment_value(&mut address_nybbles, segment, value);
    }

    let mut address_bytes = [0u8; 16];
    for i in 0..16 {
        address_bytes[i] = (address_nybbles[i * 2] << 4) | address_nybbles[i * 2 + 1];
    }
    Ok(address_bytes)
}

fn synthetic_stream_test_model() -> EntropyIpModel {
    let wide_states = (0u128..35).map(SegmentState::Single).collect::<Vec<_>>();

    let mut dense_cpt = HashMap::new();
    dense_cpt.insert(
        vec![0, 0],
        CptEntry {
            explicit_probs: vec![(0, 1.0)],
            default_prob: 0.0,
        },
    );

    let mut sparse_cpt = HashMap::new();
    sparse_cpt.insert(
        vec![0, 0, 0, 0],
        CptEntry {
            explicit_probs: vec![(1, 1.0)],
            default_prob: 0.0,
        },
    );

    EntropyIpModel {
        generation: Default::default(),
        generation_seed: 11,
        segments: vec![
            Segment {
                start_nybble: 0,
                end_nybble: 1,
                states: wide_states.clone(),
                min_value: 0,
                max_value: 34,
            },
            Segment {
                start_nybble: 2,
                end_nybble: 3,
                states: wide_states.clone(),
                min_value: 0,
                max_value: 34,
            },
            Segment {
                start_nybble: 4,
                end_nybble: 5,
                states: wide_states.clone(),
                min_value: 0,
                max_value: 34,
            },
            Segment {
                start_nybble: 6,
                end_nybble: 7,
                states: wide_states,
                min_value: 0,
                max_value: 34,
            },
            Segment {
                start_nybble: 8,
                end_nybble: 8,
                states: vec![SegmentState::Single(0), SegmentState::Single(15)],
                min_value: 0,
                max_value: 15,
            },
            Segment {
                start_nybble: 9,
                end_nybble: 9,
                states: vec![SegmentState::Range(0, 7), SegmentState::Single(15)],
                min_value: 0,
                max_value: 15,
            },
        ],
        bn_values: vec![
            (0usize..35).collect(),
            (0usize..35).collect(),
            (0usize..35).collect(),
            (0usize..35).collect(),
            vec![1, 0],
            vec![0, 1],
        ],
        segment_cardinalities: vec![35, 35, 35, 35, 2, 2],
        network: BayesianNetwork {
            parents: vec![
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                vec![0, 1],
                vec![0, 1, 2, 3],
            ],
            cpts: vec![
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                dense_cpt,
                sparse_cpt,
            ],
        },
        runtime: Default::default(),
    }
}

fn zero_prefix_segment() -> Segment {
    Segment {
        start_nybble: 0,
        end_nybble: 30,
        states: vec![SegmentState::Single(0)],
        min_value: 0,
        max_value: 0,
    }
}

#[test]
#[ignore = "requires local ntp-10k.txt reference dataset"]
fn verify_against_python() {
    let seed_path = concat!(env!("CARGO_MANIFEST_DIR"), "/ntp-10k.txt");
    let seeds = load_ipv6_seeds(seed_path);
    assert_eq!(seeds.len(), 10000);

    let config = EntropyIp {
        rng_seed: Some(0),
        ..EntropyIp::default()
    };

    let observations: Vec<Observation> = seeds
        .iter()
        .copied()
        .map(|address| Observation {
            address,
            active: true,
        })
        .collect();
    let model = config.train(&observations).unwrap();

    let expected_segments = vec![
        Segment {
            start_nybble: 0,
            end_nybble: 7,
            states: vec![
                single(604064000),
                single(604125824),
                single(604143872),
                single(604586211),
                single(604586178),
                single(604586177),
                single(604586113),
                single(604307969),
                single(604586212),
                single(604586224),
                range(536938472, 536940504),
                range(536942096, 536942264),
                range(536953936, 536954056),
                range(537067530, 537067780),
                range(604023872, 604024268),
                range(604428800, 604428831),
                range(604537364, 604538098),
                range(604586048, 604586228),
                range(604604679, 604605052),
                range(604898081, 604898430),
                range(637542160, 637542174),
                range(637600069, 637601351),
                range(671088976, 671089840),
                range(671350808, 671357584),
                range(671359432, 671361476),
                range(671363324, 671365224),
                range(704643073, 704643104),
                range(704646952, 704647888),
                range(704650550, 704651171),
                range(704652228, 704652270),
                range(704709485, 704710176),
                range(704760576, 704760602),
                range(704775392, 704775864),
                range(704776456, 704777440),
                range(704807024, 704807430),
                range(704815890, 704816239),
                range(705024000, 705024037),
                range(739242912, 739243672),
                range(536937024, 739245193),
            ],
            min_value: 536937024,
            max_value: 739245193,
        },
        Segment {
            start_nybble: 8,
            end_nybble: 9,
            states: vec![
                single(0),
                single(16),
                single(32),
                single(1),
                single(64),
                single(48),
                single(17),
                single(128),
                single(28),
                single(49),
                range(5, 174),
                range(2, 255),
            ],
            min_value: 0,
            max_value: 255,
        },
        Segment {
            start_nybble: 10,
            end_nybble: 15,
            states: vec![
                single(0),
                single(6),
                single(1),
                single(71),
                single(25),
                single(26),
                single(8523776),
                single(22),
                single(153),
                single(131072),
                range(2, 580),
                range(66824, 8388620),
                range(8454656, 16697956),
                range(807, 16777212),
            ],
            min_value: 0,
            max_value: 16777212,
        },
        Segment {
            start_nybble: 16,
            end_nybble: 16,
            states: vec![
                single(0),
                single(8),
                single(1),
                single(3),
                single(14),
                single(12),
                range(2, 15),
            ],
            min_value: 0,
            max_value: 15,
        },
        Segment {
            start_nybble: 17,
            end_nybble: 25,
            states: vec![
                single(0),
                single(44030230526),
                single(12),
                single(9268232190),
                single(16),
                single(63587680254),
                single(46698397694),
                single(4),
                single(14),
                single(43185930238),
                range(1, 8782),
                range(16777218, 16778476),
                range(33554691, 33555197),
                range(9614, 27420),
                range(28101, 68575891810),
            ],
            min_value: 0,
            max_value: 68575891810,
        },
        Segment {
            start_nybble: 26,
            end_nybble: 31,
            states: vec![
                single(0),
                single(1),
                single(3298422),
                single(2),
                single(894),
                single(14463092),
                single(571),
                single(1517703),
                single(5094943),
                single(305),
                range(67313, 16709698),
                range(4, 16776293),
            ],
            min_value: 0,
            max_value: 16776293,
        },
    ];

    assert_eq!(model.generation_seed, 0);
    assert_eq!(model.segments, expected_segments, "segment mining drifted");
    assert_eq!(model.bn_values.len(), model.segments.len());
    assert!(
        model
            .bn_values
            .iter()
            .zip(model.segments.iter())
            .all(|(codes, segment)| !codes.is_empty()
                && codes.iter().all(|&code| code < segment.states.len()))
    );

    let state_counts: Vec<usize> = model.segments.iter().map(|s| s.states.len()).collect();
    assert_eq!(model.segments.len(), 6, "segment count must be 6");
    assert_eq!(
        state_counts,
        vec![39, 12, 14, 7, 15, 12],
        "state counts must match"
    );

    let expected_parents: Vec<Vec<usize>> =
        vec![vec![], vec![0], vec![0], vec![0], vec![3], vec![3, 4]];
    for (i, expected) in expected_parents.iter().enumerate() {
        assert_eq!(
            &model.network.parents[i], expected,
            "Segment {} parents mismatch: got {:?}, expected {:?}",
            i, model.network.parents[i], expected
        );
    }
}

#[test]
fn reference_defaults_match_paper_and_scripts() {
    let config = EntropyIp::default();
    assert_eq!(config.max_parents, None);
    assert_eq!(config.size, 32);
    assert_eq!(config.step, 1);
    assert_eq!(config.isp_nybbles, 8);
    assert_eq!(config.net_nybbles, 16);
    assert_eq!(config.thresholds, vec![0.025, 0.1, 0.3, 0.5, 0.9]);
    assert_eq!(config.hysteresis, 0.05);
    assert_eq!(config.segment_sample_size, 50_000);
    assert_eq!(config.bnf_sample_size, 100_000);
    assert!(!config.bnf_full);
    assert!(!config.rcode);
    assert!(config.drop_unknown);
    assert_eq!(config.rng_seed, None);
    assert_eq!(config.effective_max_parents(0), 0);
    assert_eq!(config.effective_max_parents(6), 6);
    assert_eq!(config.effective_max_parents(10), 10);

    let bounded = EntropyIp {
        max_parents: Some(3),
        ..EntropyIp::default()
    };
    assert_eq!(bounded.effective_max_parents(2), 2);
    assert_eq!(bounded.effective_max_parents(6), 3);
}

#[test]
fn rejects_keep_unknown_without_rcode() {
    let config = EntropyIp {
        drop_unknown: false,
        rcode: false,
        ..EntropyIp::default()
    };
    let observations = vec![Observation {
        address: Ipv6Addr::from_str("2001:db8::1").unwrap().octets(),
        active: true,
    }];
    let Err(err) = config.train(&observations) else {
        panic!("invalid configuration should fail");
    };
    assert!(
        err.to_string()
            .contains("drop_unknown=false without rcode is unsupported"),
        "unexpected error: {err}"
    );
}

#[test]
fn rejects_rcode_when_placeholder_states_are_observed() {
    let config = EntropyIp {
        rcode: true,
        segment_sample_size: 1,
        rng_seed: Some(0),
        ..EntropyIp::default()
    };
    let observations = vec![
        Observation {
            address: Ipv6Addr::from_str("2001:db8::1").unwrap().octets(),
            active: true,
        },
        Observation {
            address: Ipv6Addr::from_str("2001:db8::2").unwrap().octets(),
            active: true,
        },
    ];
    let Err(err) = config.train(&observations) else {
        panic!("invalid configuration should fail");
    };
    assert!(
        err.to_string()
            .contains("rcode produced placeholder segment states"),
        "unexpected error: {err}"
    );
}

#[test]
fn preserves_range_order_for_encoding() {
    let segment = Segment {
        start_nybble: 0,
        end_nybble: 0,
        states: vec![SegmentState::Range(0, 10), SegmentState::Range(5, 5)],
        min_value: 0,
        max_value: 10,
    };
    let lookup = build_segment_lookup(&segment);
    assert_eq!(lookup.state_index_for(5), Some(0));
}

#[test]
fn bnf_sampling_matches_rewrite_bnf_generator_rows() {
    let rows = (0usize..16).map(|value| vec![value]).collect::<Vec<_>>();
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let sampled = sample_bnf_rows(&rows, 10, &mut rng);
    assert!(sampled.len() <= 10);
    assert!(sampled.iter().all(|row| rows.contains(row)));

    let unique = sampled.iter().collect::<std::collections::HashSet<_>>();
    assert_eq!(sampled.len(), unique.len());
}

#[test]
fn cpt_matches_bnfinder_dirichlet_smoothing() {
    let cpts = super::bn::calculate_cpt(1, &[0], &[vec![0, 0], vec![0, 0], vec![0, 1]], &[1, 3]);
    let entry = cpts.get(&vec![0]).expect("missing parent configuration");

    assert_eq!(
        entry
            .explicit_probs
            .iter()
            .map(|&(state, _)| state)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(entry.explicit_probs.len(), 2);
    assert!((entry.explicit_probs[0].1 - 0.5).abs() < 1e-12);
    assert!((entry.explicit_probs[1].1 - (2.0 / 6.0)).abs() < 1e-12);
    assert!((entry.default_prob - (1.0 / 6.0)).abs() < 1e-12);
}

#[test]
fn cpt_uses_bnfinder_minimum_binary_cardinality() {
    let cpts = super::bn::calculate_cpt(0, &[], &[vec![0], vec![0], vec![0]], &[1]);
    let entry = cpts.get(&Vec::<usize>::new()).expect("missing root cpt");

    assert_eq!(entry.explicit_probs, vec![(0, 4.0 / 5.0)]);
    assert!((entry.default_prob - (1.0 / 5.0)).abs() < 1e-12);
}

#[test]
fn bde_score_matches_bnfinder_objective_sign_and_penalty() {
    let rows = vec![vec![0, 0], vec![0, 0], vec![0, 1]];
    let state_counts = vec![1, 3];

    let without_parent = super::bn::calculate_bde(&rows, 1, &[], &state_counts);
    let with_parent = super::bn::calculate_bde(&rows, 1, &[0], &state_counts);

    let data_score = (3.0_f64 * 4.0 * 5.0).log2() - 2.0_f64.log2();
    let parent_graph_penalty = 1.5_f64.log2() * 4.0_f64.log2();
    assert!((without_parent + data_score).abs() < 1e-12);
    assert!((with_parent + data_score + parent_graph_penalty).abs() < 1e-12);
}

#[test]
fn parent_candidates_follow_bnfinder_weight_order() {
    let candidates = bnfinder_ordered_parent_candidates(4, &[10, 2, 5, 3, 1]);
    assert_eq!(candidates, vec![1, 3, 2, 0]);
}

#[test]
fn sampler_respects_cpt_probabilities_across_stream() {
    let mut cpts = HashMap::new();
    cpts.insert(
        Vec::new(),
        CptEntry {
            explicit_probs: vec![(1, 1.0)],
            default_prob: 0.0,
        },
    );
    let model = EntropyIpModel {
        generation: Default::default(),
        generation_seed: 42,
        segments: vec![
            zero_prefix_segment(),
            Segment {
                start_nybble: 31,
                end_nybble: 31,
                states: vec![SegmentState::Single(0), SegmentState::Single(15)],
                min_value: 0,
                max_value: 15,
            },
        ],
        bn_values: vec![vec![0], vec![0, 1]],
        segment_cardinalities: vec![1, 2],
        network: BayesianNetwork {
            parents: vec![Vec::new(), Vec::new()],
            cpts: vec![HashMap::new(), cpts],
        },
        runtime: Default::default(),
    };

    let generated = model
        .stream()
        .unwrap()
        .take(2)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let first = Ipv6Addr::from(generated[0]);
    let second = Ipv6Addr::from(generated[1]);
    assert_eq!(first, Ipv6Addr::from_str("::f").unwrap());
    assert_eq!(second, Ipv6Addr::from_str("::f").unwrap());
}

#[test]
fn generation_uses_bn_value_mapping_for_sparse_observed_codes() {
    let mut cpts = HashMap::new();
    cpts.insert(
        Vec::new(),
        CptEntry {
            explicit_probs: vec![(0, 1.0)],
            default_prob: 0.0,
        },
    );
    let model = EntropyIpModel {
        generation: Default::default(),
        generation_seed: 7,
        segments: vec![
            zero_prefix_segment(),
            Segment {
                start_nybble: 31,
                end_nybble: 31,
                states: vec![
                    SegmentState::Single(0),
                    SegmentState::Single(1),
                    SegmentState::Single(15),
                ],
                min_value: 0,
                max_value: 15,
            },
        ],
        bn_values: vec![vec![0], vec![2]],
        segment_cardinalities: vec![1, 1],
        network: BayesianNetwork {
            parents: vec![Vec::new(), Vec::new()],
            cpts: vec![HashMap::new(), cpts],
        },
        runtime: Default::default(),
    };

    let generated = Ipv6Addr::from(model.stream().unwrap().next().unwrap().unwrap());
    assert_eq!(generated, Ipv6Addr::from_str("::f").unwrap());
}

#[test]
fn compiled_runtime_matches_legacy_stochastic_generation_path() {
    let model = synthetic_stream_test_model();
    let generated = model
        .stream()
        .unwrap()
        .take(1024)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let mut legacy_rng = rand::rngs::StdRng::seed_from_u64(model.generation_seed);

    for (index, fast) in generated.into_iter().enumerate() {
        let legacy = legacy_next_address(&model, &mut legacy_rng).unwrap();
        assert_eq!(fast, legacy, "compiled path diverged at sample {index}");
    }
}

#[test]
fn target_model_stream_matches_compiled_generation() {
    let model = synthetic_stream_test_model();
    let generated = model
        .stream()
        .unwrap()
        .take(64)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let mut legacy_rng = rand::rngs::StdRng::seed_from_u64(model.generation_seed);
    let expected = (0..64)
        .map(|_| legacy_next_address(&model, &mut legacy_rng).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(generated, expected);
}

#[test]
fn rejects_segment_states_that_exceed_segment_width() {
    let model = EntropyIpModel {
        generation: Default::default(),
        generation_seed: 1,
        segments: vec![
            zero_prefix_segment(),
            Segment {
                start_nybble: 31,
                end_nybble: 31,
                states: vec![SegmentState::Single(0x10)],
                min_value: 0,
                max_value: 0x10,
            },
        ],
        bn_values: vec![vec![0], vec![0]],
        segment_cardinalities: vec![1, 1],
        network: BayesianNetwork {
            parents: vec![Vec::new(), Vec::new()],
            cpts: vec![HashMap::new(), HashMap::new()],
        },
        runtime: Default::default(),
    };

    let err = match model.stream() {
        Ok(_) => panic!("expected invalid segment-state width to be rejected"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("outside its nybble width"),
        "unexpected error: {err}"
    );
}

#[cfg(test)]
fn sample_state_value<R: Rng + ?Sized>(
    segment: &Segment,
    state_idx: usize,
    rng: &mut R,
) -> Result<u128, TgaError> {
    match segment.states.get(state_idx) {
        Some(SegmentState::Single(value)) => Ok(*value),
        Some(SegmentState::Range(min, max)) => Ok(rng.gen_range(*min..=*max)),
        None => Err(TgaError::Generation(format!(
            "Entropy/IP sampled state index {state_idx} outside mined segment-state table"
        ))),
    }
}

#[cfg(test)]
fn write_segment_value(address_nybbles: &mut [u8; 32], segment: &Segment, value: u128) {
    let num_nybbles = segment.nybble_width();
    for offset in 0..num_nybbles {
        let shift = (num_nybbles - 1 - offset) * 4;
        address_nybbles[segment.start_nybble + offset] = ((value >> shift) & 0xF) as u8;
    }
}

#[test]
fn mining_short_remainders_follow_reference_frequency_order() {
    let mut rng = StdRng::seed_from_u64(0);
    let mined = mine_segment_states(&[1, 2, 2, 3], 8, 50_000, &mut rng);
    assert_eq!(mined.states, vec![single(2), single(3), single(1)]);
}

#[test]
fn segmentation_preserves_hard_boundaries_and_strict_hysteresis() {
    let mut entropies = vec![0.0; 32];
    entropies[1] = 1.0;
    entropies[9] = 0.025;
    entropies[10] = 0.1;
    let config = EntropyIp::default();
    assert_eq!(
        define_segments(&entropies, 8, 16, &config.thresholds, config.hysteresis),
        vec![(0, 7), (8, 9), (10, 10), (11, 15), (16, 31)]
    );
}

#[test]
fn generation_rejects_malformed_model_tables() {
    for mutation in 0..5 {
        let mut model = synthetic_stream_test_model();
        match mutation {
            0 => model.segment_cardinalities[0] += 1,
            1 => model.bn_values[1][1] = model.bn_values[1][0],
            2 => model.network.parents[2] = vec![0, 0],
            3 => {
                model.network.cpts[4]
                    .get_mut(&vec![0, 0])
                    .unwrap()
                    .default_prob = f64::NAN
            }
            4 => {
                model.network.cpts[4]
                    .get_mut(&vec![0, 0])
                    .unwrap()
                    .explicit_probs = vec![(0, 1.1)]
            }
            _ => unreachable!(),
        }
        assert!(model.stream().is_err(), "mutation {mutation}");
    }
}

#[test]
fn seeded_training_and_generation_are_reproducible() {
    use crate::TargetModel;
    let observations: Vec<Observation> = (0u128..128)
        .map(|value| Observation {
            address: ((0x20010db8u128 << 96) | (value << 32) | (value % 4)).to_be_bytes(),
            active: value % 7 != 0,
        })
        .collect();
    let config = EntropyIp {
        rng_seed: Some(123),
        ..EntropyIp::default()
    };
    let mut first = config.train(&observations).unwrap();
    let mut expected = [[0; 16]; 100];
    first.generate(&mut expected).unwrap();
    for _ in 0..4 {
        let model = config.train(&observations).unwrap();
        let encoded = EntropyIp::encode_model(&model).unwrap();
        let mut restored = EntropyIp::decode_model(EntropyIp::MODEL_VERSION, &encoded).unwrap();
        let mut output = [[0; 16]; 100];
        for chunk in output.chunks_mut(7) {
            restored.generate(chunk).unwrap();
        }
        assert_eq!(output, expected);
        let mut cloned = restored.clone();
        restored.generate(&mut output).unwrap();
        let mut continuation = [[0; 16]; 100];
        cloned.generate(&mut continuation).unwrap();
        assert_eq!(continuation, output);
    }
}

#[test]
fn mining_matches_reference_script_fixtures() {
    #[derive(serde::Deserialize)]
    struct Fixture {
        name: String,
        counts: Vec<(u128, usize)>,
        bits: usize,
        states: Vec<SegmentState>,
    }
    // Generated with a2-mining.py after converting Python 2 syntax and integer division.
    let fixtures: Vec<Fixture> = serde_json::from_str(include_str!(
        "../../../tests/fixtures/entropy_ip_mining.json"
    ))
    .unwrap();
    for fixture in fixtures {
        let values: Vec<u128> = fixture
            .counts
            .into_iter()
            .flat_map(|(value, count)| std::iter::repeat_n(value, count))
            .collect();
        let mut rng = StdRng::seed_from_u64(0);
        let result = mine_segment_states(&values, fixture.bits, 50_000, &mut rng);
        assert_eq!(result.states, fixture.states, "{}", fixture.name);
    }
}

fn correlated_model() -> EntropyIpModel {
    EntropyIpModel {
        generation: Default::default(),
        generation_seed: 17,
        segments: vec![
            Segment {
                start_nybble: 0,
                end_nybble: 15,
                states: vec![single(1), single(2)],
                min_value: 1,
                max_value: 2,
            },
            Segment {
                start_nybble: 16,
                end_nybble: 31,
                states: vec![single(0), single(1)],
                min_value: 0,
                max_value: 1,
            },
        ],
        bn_values: vec![vec![0, 1], vec![0, 1]],
        segment_cardinalities: vec![2, 2],
        network: BayesianNetwork {
            parents: vec![vec![], vec![0]],
            cpts: vec![
                HashMap::from([(
                    vec![],
                    CptEntry {
                        explicit_probs: vec![(0, 0.5), (1, 0.5)],
                        default_prob: 0.0,
                    },
                )]),
                HashMap::from([
                    (
                        vec![0],
                        CptEntry {
                            explicit_probs: vec![(0, 0.9), (1, 0.1)],
                            default_prob: 0.0,
                        },
                    ),
                    (
                        vec![1],
                        CptEntry {
                            explicit_probs: vec![(0, 0.1), (1, 0.9)],
                            default_prob: 0.0,
                        },
                    ),
                ]),
            ],
        },
        runtime: Default::default(),
    }
}

#[test]
fn conditioning_on_later_segments_changes_earlier_probabilities() {
    let model = correlated_model();
    let mut sampler = model
        .conditioned_sampler(&std::collections::BTreeMap::from([(1, vec![1])]), 91)
        .unwrap();
    let mut output = vec![[0; 16]; 10_000];
    let result = sampler.sample(&mut output, 30_000).unwrap();
    assert_eq!(result.written, output.len());
    assert!(output.iter().all(|address| address[15] == 1));
    let posterior =
        output.iter().filter(|address| address[7] == 2).count() as f64 / output.len() as f64;
    assert!((posterior - 0.9).abs() < 0.02, "posterior {posterior}");
}

#[test]
fn conditioning_filters_latent_codes_in_overlapping_ranges() {
    let mut model = correlated_model();
    model.segments[0].states = vec![range(0, 1), single(0)];
    let mut sampler = model
        .conditioned_sampler(&std::collections::BTreeMap::from([(0, vec![1])]), 92)
        .unwrap();
    let mut output = vec![[0xff; 16]; 10_000];
    let result = sampler.sample(&mut output, 10_000).unwrap();
    assert!((4_800..5_200).contains(&result.written));
    assert!(
        output[..result.written]
            .iter()
            .all(|address| address[7] == 0)
    );
    assert!(
        output[result.written..]
            .iter()
            .all(|address| *address == [0xff; 16])
    );
}

#[test]
fn conditioning_is_bounded_and_rejects_invalid_codes() {
    use std::collections::BTreeMap;
    let mut model = correlated_model();
    model.network.cpts[1]
        .get_mut(&vec![0])
        .unwrap()
        .explicit_probs = vec![(0, 1.0)];
    let mut sampler = model
        .conditioned_sampler(&BTreeMap::from([(0, vec![0]), (1, vec![1])]), 17)
        .unwrap();
    let mut output = [[0xff; 16]; 3];
    assert_eq!(
        sampler.sample(&mut output, 100).unwrap(),
        EntropyIpSampleBatch {
            written: 0,
            attempts: 100
        }
    );
    assert_eq!(output, [[0xff; 16]; 3]);
    assert!(
        model
            .conditioned_sampler(&BTreeMap::from([(2, vec![0])]), 0)
            .is_err()
    );
    assert!(
        model
            .conditioned_sampler(&BTreeMap::from([(0, vec![2])]), 0)
            .is_err()
    );
    assert!(sampler.sample(&mut [], 10).is_err());
}

#[test]
fn empty_evidence_matches_unconditional_generation_and_chunking() {
    use crate::TargetModel;
    let mut model = correlated_model();
    let mut sampler = model
        .conditioned_sampler(
            &std::collections::BTreeMap::from([(0, vec![])]),
            model.generation_seed,
        )
        .unwrap();
    let mut expected = [[0; 16]; 100];
    model.generate(&mut expected).unwrap();
    let mut output = [[0; 16]; 100];
    for chunk in output.chunks_mut(7) {
        let result = sampler.sample(chunk, chunk.len()).unwrap();
        assert_eq!(result.written, chunk.len());
        assert_eq!(result.attempts, chunk.len());
    }
    assert_eq!(output, expected);
    assert_eq!(model.segments().len(), 2);
    assert_eq!(
        model.segments().next().unwrap().modeled_state_indices,
        [0, 1]
    );
}

#[test]
fn scan_feedback_is_unsupported_and_does_not_change_generation() {
    use crate::{Feedback, TargetModel};
    let mut model = correlated_model();
    let mut unchanged = model.clone();
    for event in [
        Feedback::Active(Ipv6Addr::LOCALHOST),
        Feedback::Inactive(Ipv6Addr::LOCALHOST),
        Feedback::Skipped(Ipv6Addr::LOCALHOST),
        Feedback::BatchComplete,
        Feedback::Aliased("2001:db8::/32".parse().unwrap()),
    ] {
        assert!(matches!(
            model.apply_feedback(&[event]),
            Err(TgaError::Unsupported(_))
        ));
    }
    let mut expected = [[0; 16]; 100];
    let mut actual = [[0; 16]; 100];
    model.generate(&mut actual).unwrap();
    unchanged.generate(&mut expected).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn prefix_prediction_models_only_the_top_64_bits() {
    use crate::TargetModel;
    let observations: Vec<Observation> = (0u128..16)
        .map(|value| Observation {
            address: ((0x20010db800000000u128 << 64) | (value << 64) | value).to_be_bytes(),
            active: true,
        })
        .collect();
    let config = EntropyIp {
        size: 16,
        rng_seed: Some(42),
        ..EntropyIp::default()
    };
    let mut model = config.train(&observations).unwrap();
    assert_eq!(model.segments.last().unwrap().end_nybble, 15);
    let mut output = [[0; 16]; 100];
    model.generate(&mut output).unwrap();
    assert!(output.iter().all(|address| address[8..] == [0; 8]));
    assert!(
        output
            .iter()
            .all(|address| address[..4] == [0x20, 1, 0x0d, 0xb8])
    );
}

#[test]
fn network_learning_matches_bnfinder_reference_fixtures() {
    #[derive(serde::Deserialize)]
    struct Fixture {
        name: String,
        observations: Vec<(Vec<usize>, usize)>,
        parents: Vec<Vec<usize>>,
        scores: Vec<f64>,
        cpts: Vec<Vec<ReferenceCpt>>,
    }
    // Generated from BNFinder 2.1.1 BDE, learn_1, and to_cpd with Python 3 syntax adaptations.
    let fixtures: Vec<Fixture> =
        serde_json::from_str(include_str!("../../../tests/fixtures/entropy_ip_bn.json")).unwrap();
    for fixture in fixtures {
        let rows: Vec<_> = fixture
            .observations
            .into_iter()
            .flat_map(|(row, count)| std::iter::repeat_n(row, count))
            .collect();
        let config = EntropyIp {
            bnf_full: true,
            ..EntropyIp::default()
        };
        let (_, cardinalities, network) = super::train::learn_bayesian_network(
            &config,
            &rows,
            fixture.parents.len(),
            &mut StdRng::seed_from_u64(0),
        )
        .unwrap();
        assert_eq!(network.parents, fixture.parents, "{}", fixture.name);
        for (node, expected_cpts) in fixture.cpts.iter().enumerate() {
            let score =
                super::bn::calculate_bde(&rows, node, &network.parents[node], &cardinalities);
            assert!(
                (score - fixture.scores[node]).abs() < 1e-9,
                "{} node {node}",
                fixture.name
            );
            assert_eq!(network.cpts[node].len(), expected_cpts.len());
            for expected in expected_cpts {
                let actual = &network.cpts[node][&expected.parents];
                assert!((actual.default_prob - expected.default).abs() < 1e-14);
                assert_eq!(actual.explicit_probs.len(), expected.explicit.len());
                for ((state, probability), (expected_state, expected_probability)) in
                    actual.explicit_probs.iter().zip(&expected.explicit)
                {
                    assert_eq!(state, expected_state);
                    assert!((probability - expected_probability).abs() < 1e-14);
                }
            }
        }
    }
}

#[derive(serde::Deserialize)]
struct ReferenceCpt {
    parents: Vec<usize>,
    explicit: Vec<(usize, f64)>,
    default: f64,
}

#[test]
fn complete_training_pipeline_matches_reference_scripts() {
    #[derive(serde::Deserialize)]
    struct ReferenceSegment {
        start_nybble: usize,
        end_nybble: usize,
        states: Vec<SegmentState>,
    }
    #[derive(serde::Deserialize)]
    struct Fixture {
        observations: Vec<(String, usize)>,
        segments: Vec<ReferenceSegment>,
        bn_values: Vec<Vec<usize>>,
        parents: Vec<Vec<usize>>,
        cpts: Vec<Vec<ReferenceCpt>>,
    }
    // Generated by a1, a2, a3, and BNFinder 2.1.1 without dataset sampling.
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../tests/fixtures/entropy_ip_pipeline.json"
    ))
    .unwrap();
    let observations: Vec<_> = fixture
        .observations
        .iter()
        .flat_map(|(address, count)| {
            std::iter::repeat_n(
                Observation {
                    address: u128::from_str_radix(address, 16).unwrap().to_be_bytes(),
                    active: true,
                },
                *count,
            )
        })
        .collect();
    let model = EntropyIp {
        bnf_full: true,
        rng_seed: Some(0),
        ..EntropyIp::default()
    }
    .train(&observations)
    .unwrap();
    assert_eq!(model.segments.len(), fixture.segments.len());
    for (actual, expected) in model.segments.iter().zip(&fixture.segments) {
        assert_eq!(actual.start_nybble, expected.start_nybble);
        assert_eq!(actual.end_nybble, expected.end_nybble);
        assert_eq!(actual.states, expected.states);
    }
    assert_eq!(model.bn_values, fixture.bn_values);
    assert_eq!(model.network.parents, fixture.parents);
    for (actual, expected) in model.network.cpts.iter().zip(&fixture.cpts) {
        assert_eq!(actual.len(), expected.len());
        for entry in expected {
            let actual = &actual[&entry.parents];
            assert!((actual.default_prob - entry.default).abs() < 1e-14);
            assert_eq!(actual.explicit_probs.len(), entry.explicit.len());
            for ((state, probability), (expected_state, expected_probability)) in
                actual.explicit_probs.iter().zip(&entry.explicit)
            {
                assert_eq!(state, expected_state);
                assert!((probability - expected_probability).abs() < 1e-14);
            }
        }
    }
}

#[test]
fn conditioning_uses_mined_codes_after_bn_remapping() {
    let mut model = correlated_model();
    model.bn_values[0] = vec![1, 0];
    let mut sampler = model
        .conditioned_sampler(&std::collections::BTreeMap::from([(0, vec![1, 1])]), 42)
        .unwrap();
    let mut output = [[0; 16]; 100];
    assert_eq!(sampler.sample(&mut output, 1000).unwrap().written, 100);
    assert!(output.iter().all(|address| address[7] == 2));
    let mut cloned = sampler.clone();
    assert_eq!(sampler.sample(&mut output, 0).unwrap().attempts, 0);
    let mut expected = [[0; 16]; 100];
    let left = sampler.sample(&mut output, 1000).unwrap();
    let right = cloned.sample(&mut expected, 1000).unwrap();
    assert_eq!(left, right);
    assert_eq!(output, expected);
}
