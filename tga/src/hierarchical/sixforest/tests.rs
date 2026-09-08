use std::collections::HashSet;
const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;
fn fnv1a_update(state: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *state ^= *byte as u64;
        *state = state.wrapping_mul(FNV_PRIME);
    }
}

use super::*;
use serde_json::json;
use std::net::Ipv6Addr;
use std::str::FromStr;

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

fn addr(hex: &str) -> Address {
    assert_eq!(hex.len(), 32);
    let mut out = [0u8; 16];
    for idx in 0..16 {
        out[idx] = u8::from_str_radix(&hex[idx * 2..idx * 2 + 2], 16).unwrap();
    }
    out
}

fn hash_regions(regions: &[Vec<Address>]) -> u64 {
    let mut state = FNV_OFFSET_BASIS;
    for region in regions {
        for address in region {
            let hex = address
                .iter()
                .flat_map(|byte| [nibble_to_hex(byte >> 4), nibble_to_hex(byte & 0x0f)])
                .collect::<String>();
            fnv1a_update(&mut state, hex.as_bytes());
            fnv1a_update(&mut state, b"|");
        }
        fnv1a_update(&mut state, b"\n");
    }
    state
}

fn hash_patterns(regions: &[SixForestRegion]) -> u64 {
    let mut state = FNV_OFFSET_BASIS;
    for region in regions {
        let line = format!("{}\t{}\n", region.pattern_string(), region.seed_count);
        fnv1a_update(&mut state, line.as_bytes());
    }
    state
}

fn singleton_region(address: Address) -> SixForestRegion {
    SixForestRegion {
        base: address,
        free_dims: Vec::new(),
        seed_count: 0,
    }
}

fn stream_space(space: &SixForestTargetSet, count: usize) -> Vec<Address> {
    space.stream().take(count).collect()
}

fn stream_targets(space: &SixForestModel, count: usize) -> Vec<Address> {
    space
        .stream()
        .unwrap()
        .take(count)
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

#[test]
#[ignore = "requires local ntp-10k.txt reference dataset"]
fn full_dataset_matches_paper_partition_and_reference_threshold() {
    let mut seeds = load_ipv6_seeds(concat!(env!("CARGO_MANIFEST_DIR"), "/ntp-10k.txt"));
    seeds.sort_unstable();
    seeds.dedup();
    assert_eq!(seeds.len(), 10_000);
    let regions = partition(&seeds, 16);
    assert_eq!(regions.len(), 1950);
    assert_eq!(hash_regions(&regions), 7416550444442601621);
    let observations: Vec<_> = seeds
        .iter()
        .map(|&address| Observation {
            address,
            active: true,
        })
        .collect();
    let model = SixForest::default().train(&observations).unwrap();
    assert_eq!(model.space.regions.len(), 1950);
    assert_eq!(model.outlier_count, 928);
    assert_eq!(hash_patterns(&model.space.regions), 1250876033382813827);
}

#[test]
fn generated_targets_exclude_training_seeds() {
    let seeds = vec![
        addr("20010db8000000000000000000000000"),
        addr("20010db8000000000000000000000001"),
    ];
    let excluded = seeds.iter().copied().collect::<HashSet<_>>();
    let region = SixForestRegion::from_addresses(&seeds);

    assert_eq!(region.pattern_string(), "20010db800000000000000000000000*");
    assert_eq!(region.finite_span(), Some(16));
    assert_eq!(region.finite_target_count(&excluded), Some(14));

    let targets = stream_space(
        &SixForestTargetSet::new(vec![region], seeds.clone(), DEFAULT_GENERATION_SEED),
        14,
    );
    assert_eq!(targets.len(), 14);
    assert!(targets.iter().all(|target| !excluded.contains(target)));
}

#[test]
fn seed_exclusion_is_global_not_per_region() {
    let seeds = vec![
        addr("20010db8000000000000000000000000"),
        addr("20010db8000000000000000000000001"),
        addr("20010db8000000000000000000000002"),
    ];
    let region = SixForestRegion::from_addresses(&seeds[..2]);
    let excluded = seeds.iter().copied().collect::<HashSet<_>>();

    assert_eq!(region.finite_target_count(&excluded), Some(13));
    let targets = stream_space(
        &SixForestTargetSet::new(vec![region], seeds.clone(), DEFAULT_GENERATION_SEED),
        13,
    );

    assert_eq!(targets.len(), 13);
    assert!(targets.iter().all(|target| !excluded.contains(target)));
}

#[test]
fn stream_can_run_beyond_seed_neighborhood_size() {
    let broad_region = SixForestRegion {
        base: addr("20010db8000000000000000000000000"),
        free_dims: vec![26, 27, 28, 29, 30, 31],
        seed_count: 2,
    };
    let space = SixForestTargetSet::new(vec![broad_region], Vec::new(), 99);
    let generated = stream_space(&space, 1_000);
    let unique = generated.iter().copied().collect::<HashSet<_>>();

    assert_eq!(generated.len(), 1_000);
    assert_eq!(unique.len(), generated.len());
}

#[test]
#[ignore = "requires local ntp-10k.txt reference dataset"]
fn iterator_never_reemits_seed_addresses() {
    let seed_path = concat!(env!("CARGO_MANIFEST_DIR"), "/ntp-10k.txt");
    let seeds = load_ipv6_seeds(seed_path);
    let seed_set = seeds.iter().copied().collect::<HashSet<_>>();

    let observations = seeds
        .iter()
        .copied()
        .map(|address| Observation {
            address,
            active: true,
        })
        .collect::<Vec<_>>();

    let algorithm = SixForest {
        min_region_size: 16,
        ..SixForest::default()
    };
    let model = algorithm.train(&observations).unwrap();

    let generated = stream_targets(&model, 10_000);
    assert!(
        generated.iter().all(|address| !seed_set.contains(address)),
        "generated candidates must exclude the original training seeds",
    );
}

#[test]
fn fixed_seed_regions_are_exhausted_by_streaming() {
    let seed = addr("20010db8000000000000000000000001");
    let exhausted = singleton_region(seed);
    let live = singleton_region(addr("20010db8000000000000000000000002"));
    let space = SixForestTargetSet::new(vec![exhausted, live], vec![seed], 0);

    assert_eq!(space.regions.len(), 2);
    assert_eq!(
        stream_space(&space, 1),
        vec![addr("20010db8000000000000000000000002")]
    );
}

#[test]
fn generate_returns_none_when_no_clusters() {
    let model = SixForestModel::default();
    assert!(model.stream().unwrap().next().is_none());
}

#[test]
fn serialized_space_contains_only_model_state() {
    let region = SixForestRegion::from_addresses(&[
        addr("20010db8000000000000000000000000"),
        addr("20010db8000000000000000000000001"),
    ]);
    let space = SixForestTargetSet::new(
        vec![region],
        vec![
            addr("20010db8000000000000000000000000"),
            addr("20010db8000000000000000000000001"),
        ],
        7,
    );
    let serialized = serde_json::to_value(&space).unwrap();

    assert!(serialized.get("regions").is_some());
    assert!(serialized.get("excluded_seeds").is_some());
    assert!(serialized.get("generation_seed").is_some());
    assert!(serialized.get("region_ends").is_none());
    assert!(serialized.get("target_count").is_none());

    let decoded: SixForestTargetSet = serde_json::from_value(serialized).unwrap();
    assert_eq!(stream_space(&decoded, 14).len(), 14);
}

#[test]
fn sixforest_models_do_not_claim_unique_targets() {
    let duplicate = addr("20010db8000000000000000000000007");
    let model = SixForestModel {
        generation: Default::default(),
        prescan: None,
        space: SixForestTargetSet::new(
            vec![singleton_region(duplicate), singleton_region(duplicate)],
            Vec::new(),
            0,
        ),
        input_seed_count: 0,
        outlier_count: 0,
    };

    assert_eq!(stream_targets(&model, 2), vec![duplicate, duplicate]);
}

#[test]
fn serde_default_matches_reference_beta() {
    let config: SixForest = serde_json::from_value(json!({})).unwrap();
    assert_eq!(config.min_region_size, DEFAULT_MIN_REGION_SIZE);
    assert_eq!(config.generation_seed, DEFAULT_GENERATION_SEED);
    assert_eq!(config.prescan_samples, DEFAULT_PRESCAN_SAMPLES);
}

#[test]
fn zero_min_region_size_is_rejected() {
    let algorithm = SixForest {
        min_region_size: 0,
        ..SixForest::default()
    };
    let observations = vec![Observation {
        address: addr("20010db8000000000000000000000001"),
        active: true,
    }];
    let err = algorithm.train(&observations).unwrap_err();
    assert!(matches!(err, TgaError::Config(message) if message.contains("min_region_size")));
}

#[test]
fn tied_outliers_retain_distinct_row_identities() {
    let mut weights = vec![10.0, 10.0];
    weights.extend(std::iter::repeat_n(0.0, 98));

    assert_eq!(outlier::outlier_indices(&weights), vec![0, 1]);
}

#[test]
fn display_uses_input_seed_count_over_reference_outlier_rows() {
    let model = SixForestModel {
        generation: Default::default(),
        prescan: None,
        space: SixForestTargetSet::new(
            vec![SixForestRegion {
                base: addr("20010db8000000000000000000000000"),
                free_dims: vec![31],
                seed_count: 80_500,
            }],
            Vec::new(),
            0,
        ),
        input_seed_count: 100_000,
        outlier_count: 19_503,
    };

    let display = model.to_string();
    assert!(display.contains("100000 total seeds"));
    assert!(!display.contains("100003 total seeds"));
}
#[test]
fn malformed_region_dimensions_are_rejected() {
    for dimensions in [vec![32], vec![1, 1], vec![2, 1], (0..33).collect()] {
        let value = json!({"base": ([0; 16]), "free_dims": dimensions, "seed_count": 2});
        assert!(serde_json::from_value::<SixForestRegion>(value).is_err());
    }
    let value = json!({"base": ([1; 16]), "free_dims": [31], "seed_count": 2});
    assert!(serde_json::from_value::<SixForestRegion>(value).is_err());
}

fn prescan_model() -> SixForestModel {
    let regions = [0x20, 0x30].map(|prefix| {
        let mut base = [0; 16];
        base[0] = prefix;
        SixForestRegion {
            base,
            free_dims: vec![28, 29, 30, 31],
            seed_count: 2,
        }
    });
    let space = SixForestTargetSet::new(regions.into(), Vec::new(), 42);
    let prescan = Some(prescan::Prescan::new(&space, 2));
    SixForestModel {
        space,
        prescan,
        ..Default::default()
    }
}

#[test]
fn prescan_waits_for_partial_feedback_and_prioritizes_hits() {
    use crate::{Feedback, GenerationState, TargetModel};
    let mut model = prescan_model();
    let mut output = [[0; 16]; 8];
    let batch = model.generate(&mut output).unwrap();
    assert_eq!(batch.written, 4);
    assert_eq!(batch.state, GenerationState::AwaitingFeedback);
    let samples = output[..4].to_vec();
    model
        .apply_feedback(&[Feedback::Active(samples[0].into())])
        .unwrap();
    assert_eq!(
        model.generate(&mut output).unwrap().state,
        GenerationState::AwaitingFeedback
    );
    model
        .apply_feedback(&[Feedback::Active(samples[0].into()), Feedback::BatchComplete])
        .unwrap();
    assert_eq!(model.generate(&mut output).unwrap().written, 8);
    assert!(output.iter().all(|address| address[0] == samples[0][0]));
    assert!(output.iter().all(|address| !samples.contains(address)));
}

#[test]
fn prescan_round_trip_preserves_pending_feedback_and_position() {
    use crate::{Feedback, TargetModel};
    let mut model = prescan_model();
    model.generate(&mut [[0; 16]; 3]).unwrap();
    let bytes = SixForest::encode_model(&model).unwrap();
    let mut decoded = SixForest::decode_model(SixForest::MODEL_VERSION, &bytes).unwrap();
    let mut a = [[0; 16]; 7];
    let mut b = a;
    assert_eq!(
        model.generate(&mut a).unwrap(),
        decoded.generate(&mut b).unwrap()
    );
    assert_eq!(a, b);
    for model in [&mut model, &mut decoded] {
        model.apply_feedback(&[Feedback::BatchComplete]).unwrap();
    }
    assert_eq!(
        model.generate(&mut a).unwrap(),
        decoded.generate(&mut b).unwrap()
    );
    assert_eq!(a, b);
}

#[test]
fn small_prescan_regions_are_fully_scanned_without_reemission() {
    use crate::{GenerationState, TargetModel};
    let seeds = [
        addr("20010db8000000000000000000000000"),
        addr("20010db8000000000000000000000001"),
    ];
    let observations: Vec<_> = seeds
        .iter()
        .map(|&address| Observation {
            address,
            active: true,
        })
        .collect();
    let mut model = SixForest {
        prescan_samples: 2,
        ..Default::default()
    }
    .train(&observations)
    .unwrap();
    let mut output = [[0; 16]; 20];
    let batch = model.generate(&mut output).unwrap();
    assert_eq!(batch.written, 14);
    assert_eq!(batch.state, GenerationState::Exhausted);
    assert_eq!(
        model.generate(&mut output).unwrap().state,
        GenerationState::Exhausted
    );
}

#[test]
fn inactive_only_input_is_rejected_and_inactive_rows_do_not_train() {
    let active = addr("20010db8000000000000000000000001");
    let inactive = addr("20010db8000000000000000000000002");
    assert!(
        SixForest::default()
            .train(&[Observation {
                address: inactive,
                active: false
            }])
            .is_err()
    );
    let model = SixForest::default()
        .train(&[
            Observation {
                address: active,
                active: true,
            },
            Observation {
                address: inactive,
                active: false,
            },
        ])
        .unwrap();
    assert_eq!(model.input_seed_count, 1);
    assert_eq!(model.outlier_count, 0);
}

#[test]
fn mining_matches_paper_fixture_with_reference_outlier_threshold() {
    #[derive(Deserialize)]
    struct Fixture {
        seeds: Vec<String>,
        partitions: Vec<Vec<String>>,
        patterns: Vec<(String, usize)>,
        outliers: usize,
    }
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../tests/fixtures/sixforest/mining.json"
    ))
    .unwrap();
    let seeds: Vec<_> = fixture.seeds.iter().map(|s| addr(s)).collect();
    let expected: Vec<Vec<_>> = fixture
        .partitions
        .iter()
        .map(|rows| rows.iter().map(|s| addr(s)).collect())
        .collect();
    assert_eq!(partition(&seeds, 16), expected);
    let observations: Vec<_> = seeds
        .iter()
        .map(|&address| Observation {
            address,
            active: true,
        })
        .collect();
    let model = SixForest::default().train(&observations).unwrap();
    let patterns: Vec<_> = model
        .space
        .regions
        .iter()
        .map(|region| (region.pattern_string(), region.seed_count))
        .collect();
    assert_eq!(patterns, fixture.patterns);
    assert_eq!(model.outlier_count, fixture.outliers);
    assert_eq!(
        model
            .space
            .regions
            .iter()
            .map(|region| region.seed_count)
            .sum::<usize>()
            + model.outlier_count,
        seeds.len()
    );
}

#[test]
fn aliases_covering_large_regions_exhaust_without_enumerating_them() {
    use crate::{Feedback, GenerationState, TargetModel};
    let mut model = prescan_model();
    model.generate(&mut [[0; 16]; 8]).unwrap();
    model
        .apply_feedback(&[Feedback::Aliased("::/0".parse().unwrap())])
        .unwrap();
    let batch = model.generate(&mut [[0; 16]; 8]).unwrap();
    assert_eq!(batch.written, 0);
    assert_eq!(batch.state, GenerationState::Exhausted);
}

#[test]
fn skipped_and_unrelated_feedback_do_not_count_as_hits() {
    use crate::{Feedback, TargetModel};
    let mut model = prescan_model();
    let mut output = [[0; 16]; 8];
    model.generate(&mut output).unwrap();
    let samples = output[..4].to_vec();
    model
        .apply_feedback(&[
            Feedback::Active("ffff::1".parse().unwrap()),
            Feedback::Skipped(samples[0].into()),
            Feedback::Active(samples[0].into()),
            Feedback::Active(samples[2].into()),
            Feedback::BatchComplete,
        ])
        .unwrap();
    let replacement = model.generate(&mut output).unwrap();
    assert_eq!(replacement.written, 1);
    model.apply_feedback(&[Feedback::BatchComplete]).unwrap();
    model.generate(&mut output).unwrap();
    assert!(output.iter().all(|address| address[0] == samples[2][0]));
}

#[test]
fn generation_is_independent_of_buffer_size() {
    use crate::TargetModel;
    let seeds = [
        addr("20010db8000000000000000000000000"),
        addr("20010db8000000000000000000000001"),
    ];
    let observations: Vec<_> = seeds
        .iter()
        .map(|&address| Observation {
            address,
            active: true,
        })
        .collect();
    let mut one = SixForest::default().train(&observations).unwrap();
    let mut many = one.clone();
    let mut output = [[0; 16]; 14];
    many.generate(&mut output).unwrap();
    for expected in output {
        let mut single = [[0; 16]; 1];
        one.generate(&mut single).unwrap();
        assert_eq!(single[0], expected);
    }
    assert!(one.generate(&mut []).is_err());
}

#[test]
fn legacy_models_migrate_without_changing_regions() {
    let seeds = [
        addr("20010db8000000000000000000000000"),
        addr("20010db8000000000000000000000001"),
    ];
    let space = SixForestTargetSet::new(
        vec![SixForestRegion::from_addresses(&seeds)],
        seeds.to_vec(),
        7,
    );
    let bytes = bincode::serialize(&(space.clone(), 2usize, 0usize)).unwrap();
    let decoded = SixForest::decode_model(1, &bytes).unwrap();
    assert_eq!(stream_targets(&decoded, 14), stream_space(&space, 14));
    assert_eq!(decoded.input_seed_count, 2);
    assert!(decoded.prescan.is_none());
    assert!(SixForest::decode_model(99, &bytes).is_err());
}

#[test]
fn repeated_seeds_and_input_order_do_not_change_training() {
    let seeds = [
        addr("20010db8000000000000000000000000"),
        addr("20010db800000000000000000000ffff"),
    ];
    let observations: Vec<_> = seeds
        .iter()
        .map(|&address| Observation {
            address,
            active: true,
        })
        .collect();
    let expected = SixForest::default().train(&observations).unwrap();
    let repeated = [
        observations[1],
        observations[0],
        observations[1],
        observations[0],
    ];
    let actual = SixForest::default().train(&repeated).unwrap();
    assert_eq!(actual.input_seed_count, 2);
    assert_eq!(
        hash_patterns(&actual.space.regions),
        hash_patterns(&expected.space.regions)
    );
    assert_eq!(actual.outlier_count, 0);
}
