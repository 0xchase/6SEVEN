use super::{coverage::*, growth::*, index::*, model::*, range::*, *};
use crate::TargetModel;
use crate::{Algorithm, Observation};
use clap::Parser;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::HashSet;
use std::net::Ipv6Addr;
use std::str::FromStr;

fn observations(addrs: &[&str]) -> Vec<Observation> {
    addrs
        .iter()
        .map(|addr| Observation {
            address: Ipv6Addr::from_str(addr).unwrap().octets(),
            active: true,
        })
        .collect()
}

fn generated_addrs(model: &SixGenModel) -> Vec<Ipv6Addr> {
    let mut model = model.clone();
    let mut addresses = Vec::new();
    loop {
        let mut buffer = [[0; 16]; 17];
        let generated = model.generate(&mut buffer).unwrap();
        addresses.extend(
            buffer[..generated.written]
                .iter()
                .copied()
                .map(Ipv6Addr::from),
        );
        if generated.state == crate::GenerationState::Exhausted {
            break;
        }
    }
    addresses
}

fn model_from_blocks(blocks: Vec<SixGenBlock>) -> SixGenModel {
    SixGenModel::from_blocks(blocks, 0, SixGenRangeMode::Loose)
}

#[derive(Parser)]
struct SixGenCliWrap {
    #[command(flatten)]
    sixgen: SixGen,
}

#[test]
fn cli_defaults_match_paper_baseline() {
    let parsed = SixGenCliWrap::parse_from(["sixgen"]);
    assert_eq!(parsed.sixgen.budget, 1_000_000);
    assert_eq!(parsed.sixgen.range_mode, SixGenRangeMode::Loose);
    assert_eq!(parsed.sixgen.seed, 0);
}

#[test]
fn subtract_preserves_exact_domain_difference() {
    let lhs = AddressRange {
        dims: std::array::from_fn(|idx| {
            if idx == 31 {
                NibbleDomain::contiguous(0, 0xF)
            } else {
                NibbleDomain::singleton(0)
            }
        }),
    };
    let rhs = AddressRange {
        dims: std::array::from_fn(|idx| {
            if idx == 31 {
                NibbleDomain::contiguous(4, 7)
            } else {
                NibbleDomain::singleton(0)
            }
        }),
    };

    let pieces = lhs.subtract(&rhs);
    assert_eq!(pieces.len(), 1);
    assert_eq!(pieces[0].size(), 12);
    assert!(pieces[0].contains(&bytes_to_nibbles(
        &Ipv6Addr::from_str("::3").unwrap().octets()
    )));
    assert!(pieces[0].contains(&bytes_to_nibbles(
        &Ipv6Addr::from_str("::8").unwrap().octets()
    )));
    assert!(!pieces[0].contains(&bytes_to_nibbles(
        &Ipv6Addr::from_str("::5").unwrap().octets()
    )));
    assert_eq!(total_fragment_size(&pieces), 12);
}

#[test]
fn proposal_density_comparison_uses_exact_rational_ordering() {
    assert_eq!(compare_density(1, 2, 2, 3), std::cmp::Ordering::Less);
    assert_eq!(compare_density(2, 3, 1, 2), std::cmp::Ordering::Greater);
    assert_eq!(
        compare_density(3, u128::MAX, 2, u128::MAX),
        std::cmp::Ordering::Greater
    );
}

#[test]
fn exact_range_size_is_not_truncated_to_stream_count() {
    let range = AddressRange {
        dims: std::array::from_fn(|idx| {
            if idx < 20 {
                NibbleDomain::full()
            } else {
                NibbleDomain::singleton(0)
            }
        }),
    };

    assert_eq!(range.size(), usize::MAX);
    assert_eq!(range.exact_size(), RangeSize::Finite(1u128 << 80));
}

#[test]
fn tight_range_tracks_exact_observed_values() {
    let first = bytes_to_nibbles(&Ipv6Addr::from_str("2001:db8::1").unwrap().octets());
    let second = bytes_to_nibbles(&Ipv6Addr::from_str("2001:db8::3").unwrap().octets());
    let missing = bytes_to_nibbles(&Ipv6Addr::from_str("2001:db8::2").unwrap().octets());

    let range = AddressRange::from_seed(&first).grow_with_seed(&second, SixGenRangeMode::Tight);
    assert_eq!(range.size(), 2);
    assert!(range.contains(&first));
    assert!(range.contains(&second));
    assert!(!range.contains(&missing));
}

#[test]
fn distance_ignores_already_dynamic_tight_nybbles() {
    let first = bytes_to_nibbles(&Ipv6Addr::from_str("2001:db8::1").unwrap().octets());
    let second = bytes_to_nibbles(&Ipv6Addr::from_str("2001:db8::3").unwrap().octets());
    let third = bytes_to_nibbles(&Ipv6Addr::from_str("2001:db8::2").unwrap().octets());

    let range = AddressRange::from_seed(&first).grow_with_seed(&second, SixGenRangeMode::Tight);
    assert_eq!(range.distance_to(&third), 0);
}

#[test]
fn tight_mode_does_not_fill_unobserved_gap_values() {
    let cfg = SixGen {
        budget: 16,
        range_mode: SixGenRangeMode::Tight,
        seed: 0,
    };
    let model = cfg
        .train(&observations(&["2001:db8::1", "2001:db8::3"]))
        .unwrap();

    let generated = generated_addrs(&model);
    assert!(generated.is_empty());
}

#[test]
fn tight_mode_generates_exact_cross_product_values() {
    let cfg = SixGen {
        budget: 16,
        range_mode: SixGenRangeMode::Tight,
        seed: 0,
    };
    let model = cfg
        .train(&observations(&[
            "2001:db8::11",
            "2001:db8::12",
            "2001:db8::21",
        ]))
        .unwrap();

    let generated = generated_addrs(&model);
    assert_eq!(generated, vec![Ipv6Addr::from_str("2001:db8::22").unwrap()]);
}

#[test]
fn loose_mode_uses_full_wildcard_range() {
    let cfg = SixGen {
        budget: 32,
        range_mode: SixGenRangeMode::Loose,
        seed: 0,
    };
    let model = cfg
        .train(&observations(&["2001:db8::1", "2001:db8::2"]))
        .unwrap();

    let generated = generated_addrs(&model);
    assert_eq!(generated.len(), 14);
    assert!(!generated.contains(&Ipv6Addr::from_str("2001:db8::1").unwrap()));
    assert!(!generated.contains(&Ipv6Addr::from_str("2001:db8::2").unwrap()));
    assert!(generated.contains(&Ipv6Addr::from_str("2001:db8::0").unwrap()));
    assert!(generated.contains(&Ipv6Addr::from_str("2001:db8::f").unwrap()));
}

#[test]
fn partial_budget_sampling_is_exact_and_seeded() {
    let cfg = SixGen {
        budget: 5,
        range_mode: SixGenRangeMode::Loose,
        seed: 7,
    };
    let model = cfg
        .train(&observations(&["2001:db8::1", "2001:db8::2"]))
        .unwrap();

    let generated = generated_addrs(&model);
    assert_eq!(total_block_size(&model.blocks), 5);
    assert_eq!(generated.len(), 5);

    let second = cfg
        .train(&observations(&["2001:db8::1", "2001:db8::2"]))
        .unwrap();
    assert_eq!(generated, generated_addrs(&second));
}

#[test]
fn loose_mode_fills_budget_when_range_capacity_is_available() {
    let cfg = SixGen {
        budget: 1_000,
        range_mode: SixGenRangeMode::Loose,
        seed: 11,
    };
    let model = cfg
        .train(&observations(&["2001:db8::1", "2001:db8::2222"]))
        .unwrap();

    let generated = generated_addrs(&model);
    assert_eq!(total_block_size(&model.blocks), cfg.budget);
    assert_eq!(generated.len(), cfg.budget);
}

#[test]
fn stream_expands_ranges_in_pattern_order() {
    let range = AddressRange {
        dims: std::array::from_fn(|idx| {
            if idx == 31 {
                NibbleDomain::full()
            } else {
                NibbleDomain::singleton(0)
            }
        }),
    };
    let model = model_from_blocks(vec![SixGenBlock::Range(range)]);

    let generated = generated_addrs(&model);
    let sequential = (0..16)
        .map(|value| Ipv6Addr::from_str(&format!("::{value:x}")).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(generated.len(), sequential.len());
    assert_eq!(generated, sequential);
}

#[test]
fn generated_model_emits_unique_targets() {
    let cfg = SixGen {
        budget: 32,
        range_mode: SixGenRangeMode::Loose,
        seed: 0,
    };
    let model = cfg
        .train(&observations(&["2001:db8::1", "2001:db8::2"]))
        .unwrap();

    let generated = generated_addrs(&model);
    assert_eq!(
        generated
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        generated.len()
    );
}

#[test]
fn subset_pruning_removes_strictly_contained_clusters() {
    let seed = bytes_to_nibbles(&Ipv6Addr::from_str("2001:db8::1").unwrap().octets());
    let superset = ActiveCluster {
        range: AddressRange {
            dims: std::array::from_fn(|idx| {
                if idx == 31 {
                    NibbleDomain::contiguous(0, 0xF)
                } else {
                    NibbleDomain::singleton(seed[idx])
                }
            }),
        },
        growth_cache: Some(Vec::new()),
    };
    let subset = ActiveCluster {
        range: AddressRange::from_seed(&seed),
        growth_cache: Some(Vec::new()),
    };

    let mut clusters = vec![Some(superset), Some(subset)];
    prune_subset_clusters(&mut clusters, 0);

    assert!(clusters[0].is_some());
    assert!(clusters[1].is_none());
}

#[test]
fn trie_queries_and_growth_match_brute_force() {
    let mut rng = StdRng::seed_from_u64(91);
    for mode in [SixGenRangeMode::Loose, SixGenRangeMode::Tight] {
        for _ in 0..128 {
            let mut seeds: Vec<Nibbles> = (0..24)
                .map(|_| std::array::from_fn(|dim| if dim < 28 { 0 } else { rng.gen_range(0..8) }))
                .collect();
            seeds.sort_unstable();
            seeds.dedup();
            let trie = SeedTrie::new(&seeds);
            let mut range = AddressRange::from_seed(&seeds[0]);
            for seed in seeds.iter().take(rng.gen_range(1..=4)) {
                range = range.grow_with_seed(seed, mode);
            }
            let external: Vec<_> = seeds
                .iter()
                .enumerate()
                .filter(|(_, seed)| !range.contains(seed))
                .collect();
            let min_distance = external
                .iter()
                .map(|(_, seed)| range.distance_to(seed))
                .min();
            let nearest: Vec<_> = external
                .iter()
                .filter(|(_, seed)| Some(range.distance_to(seed)) == min_distance)
                .map(|(index, _)| *index)
                .collect();
            assert_eq!(trie.nearest_external_seed_indices(&range), nearest);
            assert_eq!(trie.count_in_range(&range), seeds.len() - external.len());

            let candidates: HashSet<_> = nearest
                .iter()
                .map(|&index| range.grow_with_seed(&seeds[index], mode))
                .collect();
            let mut expected = Vec::new();
            let mut best = (0u128, 1u128);
            for candidate in candidates {
                let count = seeds.iter().filter(|seed| candidate.contains(seed)).count() as u128;
                let size: u128 = candidate
                    .dims
                    .iter()
                    .map(|domain| domain.mask.count_ones() as u128)
                    .product();
                match (count * best.1)
                    .cmp(&(best.0 * size))
                    .then_with(|| best.1.cmp(&size))
                {
                    std::cmp::Ordering::Greater => {
                        best = (count, size);
                        expected.clear();
                        expected.push(candidate);
                    }
                    std::cmp::Ordering::Equal => expected.push(candidate),
                    std::cmp::Ordering::Less => {}
                }
            }
            let actual = best_growth_for_cluster(&range, &seeds, &trie, mode);
            assert_eq!(
                actual
                    .iter()
                    .map(|proposal| proposal.new_range)
                    .collect::<HashSet<_>>(),
                expected.into_iter().collect()
            );
        }
    }
}

#[test]
fn subtraction_matches_set_difference_for_sparse_domains() {
    let mut rng = StdRng::seed_from_u64(32);
    for _ in 0..128 {
        let mut make_range = || AddressRange {
            dims: std::array::from_fn(|dim| {
                if dim < 30 {
                    NibbleDomain::singleton(0)
                } else {
                    NibbleDomain {
                        mask: rng.gen_range(1..=u16::MAX),
                    }
                }
            }),
        };
        let lhs = make_range();
        let rhs = make_range();
        let expected: HashSet<_> = lhs
            .iter()
            .filter(|address| !rhs.contains(&bytes_to_nibbles(address)))
            .collect();
        let pieces = lhs.subtract(&rhs);
        let actual: Vec<_> = pieces.iter().flat_map(|range| range.iter()).collect();
        assert_eq!(actual.len(), expected.len());
        assert_eq!(actual.into_iter().collect::<HashSet<_>>(), expected);
    }
}

#[test]
fn overlap_is_charged_once_and_seeds_are_excluded() {
    let seeds = [[0; 32]];
    let mut horizontal = AddressRange::from_seed(&seeds[0]);
    horizontal.dims[31] = NibbleDomain::full();
    let mut vertical = AddressRange::from_seed(&seeds[0]);
    vertical.dims[30] = NibbleDomain::full();
    let mut coverage = GeneratedCoverage::from_seed_ranges(&seeds);
    for range in [horizontal, vertical, horizontal] {
        coverage.admit_fragments(coverage.uncovered_fragments(range));
    }
    assert_eq!(coverage.target_count(), 30);
    let (blocks, _) = coverage.into_blocks();
    let actual = generated_addrs(&model_from_blocks(blocks));
    assert_eq!(actual.iter().collect::<HashSet<_>>().len(), 30);
    assert!(!actual.contains(&Ipv6Addr::UNSPECIFIED));
}

#[test]
fn sampling_handles_dense_sparse_and_empty_budgets() {
    let mut range = AddressRange::from_seed(&[0; 32]);
    range.dims[31] = NibbleDomain { mask: 0b101101 };
    let fragments = [range];
    for count in [0, 1, 3, 4, 20] {
        let sampled = sample_points(&fragments, count, &mut StdRng::seed_from_u64(7));
        assert_eq!(sampled.len(), count.min(4));
        assert_eq!(sampled.iter().collect::<HashSet<_>>().len(), sampled.len());
        assert!(
            sampled
                .iter()
                .all(|address| range.contains(&bytes_to_nibbles(address)))
        );
        assert_eq!(
            sampled,
            sample_points(&fragments, count, &mut StdRng::seed_from_u64(7))
        );
    }
    assert!(sample_points(&[], 10, &mut StdRng::seed_from_u64(0)).is_empty());
}

#[test]
fn sampling_weights_fragments_by_address_count() {
    let first = AddressRange::from_seed(&[0; 32]);
    let mut second = first;
    second.dims[31] = NibbleDomain { mask: 0b1110 };
    let mut counts = [0usize; 4];
    let mut rng = StdRng::seed_from_u64(84);
    for _ in 0..4000 {
        let sampled = sample_points(&[first, second], 1, &mut rng);
        counts[sampled[0][15] as usize] += 1;
    }
    assert!(counts.into_iter().all(|count| (850..1150).contains(&count)));
}

#[test]
fn full_ipv6_space_has_exact_density_and_rank_arithmetic() {
    let full = AddressRange {
        dims: [NibbleDomain::full(); 32],
    };
    assert_eq!(full.exact_size(), RangeSize::Full);
    assert_eq!(full.address_at(u128::MAX), [255; 16]);
    let full_proposal = GrowthProposal {
        new_range: full,
        new_seed_count: 2,
        range_size: RangeSize::Full,
    };
    let almost_full = GrowthProposal {
        range_size: RangeSize::Finite(u128::MAX),
        ..full_proposal
    };
    assert_eq!(
        compare_proposals(&almost_full, &full_proposal),
        std::cmp::Ordering::Greater
    );
    let half_density = GrowthProposal {
        new_seed_count: 1,
        ..almost_full
    };
    assert_eq!(
        compare_proposals(&half_density, &full_proposal),
        std::cmp::Ordering::Less
    );
    let fragments = full.subtract(&AddressRange::from_seed(&[0; 32]));
    assert_eq!(total_fragment_exact_size(&fragments), u128::MAX);
    let sample = sample_points(&fragments, 32, &mut StdRng::seed_from_u64(0));
    assert_eq!(sample.iter().collect::<HashSet<_>>().len(), 32);
    assert!(!sample.contains(&[0; 16]));
    let config = SixGen {
        budget: 32,
        ..SixGen::default()
    };
    let model = config
        .train(&[
            Observation {
                address: [0; 16],
                active: true,
            },
            Observation {
                address: [255; 16],
                active: true,
            },
        ])
        .unwrap();
    let generated = generated_addrs(&model);
    assert_eq!(generated.len(), 32);
    assert!(!generated.contains(&Ipv6Addr::UNSPECIFIED));
    assert!(!generated.contains(&Ipv6Addr::from([255; 16])));
}

#[test]
fn input_order_duplicates_and_inactive_observations_do_not_change_training() {
    let config = SixGen {
        budget: 20,
        ..SixGen::default()
    };
    let clean = observations(&["::1", "::2", "::11"]);
    let mut noisy = clean.clone();
    noisy.reverse();
    noisy.push(clean[0]);
    noisy.push(Observation {
        address: [255; 16],
        active: false,
    });
    assert_eq!(
        generated_addrs(&config.train(&clean).unwrap()),
        generated_addrs(&config.train(&noisy).unwrap())
    );
    assert!(config.train(&[]).is_err());
    assert!(
        config
            .train(&[Observation {
                address: [0; 16],
                active: false
            }])
            .is_err()
    );
    assert!(generated_addrs(&config.train(&clean[..1]).unwrap()).is_empty());
    assert!(
        generated_addrs(
            &SixGen {
                budget: 0,
                ..config
            }
            .train(&clean)
            .unwrap()
        )
        .is_empty()
    );
}

#[test]
fn buffers_clones_and_serialization_preserve_generation_contract() {
    let config = SixGen {
        budget: 20,
        ..SixGen::default()
    };
    let mut model = config
        .train(&observations(&["::1", "::2", "::11"]))
        .unwrap();
    assert!(
        model
            .blocks
            .iter()
            .any(|block| matches!(block, SixGenBlock::Range(_)))
    );
    assert!(
        model
            .blocks
            .iter()
            .any(|block| matches!(block, SixGenBlock::Points(_)))
    );
    let expected = generated_addrs(&model);
    let encoded = SixGen::encode_model(&model).unwrap();
    assert!(model.generate(&mut []).is_err());
    let mut prefix = [[0; 16]; 3];
    assert_eq!(model.generate(&mut prefix).unwrap().written, 3);
    let cloned = model.clone();
    assert_eq!(generated_addrs(&cloned), expected[3..]);
    assert_eq!(SixGen::encode_model(&model).unwrap(), encoded);
    let restored = SixGen::decode_model(SixGen::MODEL_VERSION, &encoded).unwrap();
    assert_eq!(generated_addrs(&restored), expected);
    let mut rest = [[0; 16]; 32];
    let result = model.generate(&mut rest).unwrap();
    assert_eq!(result.written, 17);
    assert_eq!(result.state, crate::GenerationState::Exhausted);
    assert_eq!(model.generate(&mut rest).unwrap().written, 0);
}

#[test]
fn serialized_empty_domains_are_rejected() {
    let domain = NibbleDomain::singleton(3);
    let encoded = bincode::serialize(&domain).unwrap();
    assert_eq!(encoded, bincode::serialize(&8u16).unwrap());
    assert_eq!(
        bincode::deserialize::<NibbleDomain>(&encoded).unwrap(),
        domain
    );
    assert!(bincode::deserialize::<NibbleDomain>(&[0, 0]).is_err());
    assert!(serde_json::from_str::<NibbleDomain>(r#"{"mask":0}"#).is_err());
}
