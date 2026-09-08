use std::collections::BTreeMap;

use linfa::traits::{Fit, Predict};
use linfa_clustering::KMeans;
use ndarray::Array2;
use rand::{SeedableRng, rngs::StdRng};

use super::{config::SixGcvaeClassification, preprocess::bytes_to_nybble_sequence};
use crate::{Address, TgaError};

pub(super) struct SeedGroup {
    pub label: String,
    pub seeds: Vec<Address>,
}

pub(super) fn partition(
    seeds: &[Address],
    classification: SixGcvaeClassification,
    clusters: usize,
) -> Result<Vec<SeedGroup>, TgaError> {
    if classification == SixGcvaeClassification::None {
        return Ok(vec![SeedGroup {
            label: "all".into(),
            seeds: seeds.to_vec(),
        }]);
    }
    let labels = match classification {
        SixGcvaeClassification::None => unreachable!("unclassified seeds returned above"),
        SixGcvaeClassification::Manual => seeds.iter().map(|seed| manual(seed).into()).collect(),
        SixGcvaeClassification::Entropy => entropy_labels(seeds, clusters)?,
    };
    let mut groups = BTreeMap::<String, Vec<Address>>::new();
    for (&seed, label) in seeds.iter().zip(labels) {
        groups.entry(label).or_default().push(seed);
    }
    groups
        .into_iter()
        .map(|(label, seeds)| {
            if seeds.len() < 2 {
                return Err(TgaError::Training(format!(
                    "6GCVAE category {label} needs at least two seeds"
                )));
            }
            Ok(SeedGroup { label, seeds })
        })
        .collect()
}

fn manual(seed: &Address) -> &'static str {
    let nybbles = bytes_to_nybble_sequence(seed);
    let iid = &nybbles[16..];
    if nybbles[22..26] == [15, 15, 15, 14] {
        return "slaac_eui64";
    }
    if normalized_entropy(iid.iter().copied()) > 0.8 {
        return "slaac_privacy";
    }
    let zero_runs = iid
        .split(|&nybble| nybble != 0)
        .filter(|run| run.len() >= 2)
        .count();
    if zero_runs == 1 {
        "fixed_iid"
    } else {
        // The reference script assigns unmatched addresses to the low-subnet category.
        "low_64bit_subnet"
    }
}

fn normalized_entropy(values: impl Iterator<Item = u8>) -> f64 {
    let mut counts = [0usize; 16];
    let mut total = 0;
    for value in values {
        counts[usize::from(value)] += 1;
        total += 1;
    }
    counts
        .into_iter()
        .filter(|&count| count != 0)
        .map(|count| {
            let probability = count as f64 / total as f64;
            -probability * probability.log2() / 4.0
        })
        .sum()
}

fn entropy_labels(seeds: &[Address], clusters: usize) -> Result<Vec<String>, TgaError> {
    if seeds.is_empty() {
        return Ok(Vec::new());
    }
    let mut prefixes = BTreeMap::<[u8; 4], Vec<[u8; 32]>>::new();
    for seed in seeds {
        prefixes
            .entry(seed[..4].try_into().expect("four prefix bytes"))
            .or_default()
            .push(bytes_to_nybble_sequence(seed));
    }
    let prefix_seeds: Vec<_> = prefixes.values().collect();
    let features = Array2::from_shape_fn((prefixes.len(), 24), |(row, column)| {
        normalized_entropy(prefix_seeds[row].iter().map(|seed| seed[column + 8]))
    });
    let dataset = linfa::DatasetBase::from(features);
    let model = KMeans::params_with_rng(clusters.min(prefixes.len()), StdRng::seed_from_u64(0))
        .max_n_iterations(300)
        .fit(&dataset)
        .map_err(|error| TgaError::Training(format!("6GCVAE entropy clustering: {error}")))?;
    let assignments = model.predict(&dataset);
    let labels: BTreeMap<_, _> = prefixes
        .keys()
        .zip(assignments.iter())
        .map(|(prefix, cluster)| (*prefix, format!("cluster_{cluster}")))
        .collect();
    Ok(seeds
        .iter()
        .map(|seed| labels[&<[u8; 4]>::try_from(&seed[..4]).expect("four prefix bytes")].clone())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv6Addr;

    fn address(text: &str) -> Address {
        text.parse::<Ipv6Addr>().unwrap().octets()
    }

    #[test]
    fn manual_categories_follow_iid_structure() {
        for (text, expected) in [
            ("2001:db8::1", "fixed_iid"),
            ("2001:db8::1:0:2", "low_64bit_subnet"),
            ("2001:db8::211:22ff:fe33:4455", "slaac_eui64"),
            ("2001:db8::123:4567:89ab:cdef", "slaac_privacy"),
        ] {
            assert_eq!(manual(&address(text)), expected);
        }
    }

    #[test]
    fn entropy_is_normalized_in_base_two() {
        assert_eq!(normalized_entropy([0; 16].into_iter()), 0.0);
        assert_eq!(normalized_entropy(0..16), 1.0);
        assert_eq!(normalized_entropy([0, 0, 1, 1].into_iter()), 0.25);
    }

    #[test]
    fn identical_prefix_fingerprints_do_not_drop_seeds() {
        let seeds: Vec<_> = (1..=8u32)
            .flat_map(|prefix| {
                let mut seed = [0; 16];
                seed[..4].copy_from_slice(&prefix.to_be_bytes());
                [seed, seed]
            })
            .collect();
        let groups = partition(&seeds, SixGcvaeClassification::Entropy, 6).unwrap();
        assert_eq!(
            groups.iter().map(|group| group.seeds.len()).sum::<usize>(),
            seeds.len()
        );
    }

    #[test]
    fn clustering_groups_prefix_fingerprints_instead_of_addresses() {
        let mut seeds = Vec::new();
        for prefix in [1u32, 2, 3, 4] {
            for suffix in 0..16u8 {
                let mut seed = [0; 16];
                seed[..4].copy_from_slice(&prefix.to_be_bytes());
                if prefix >= 3 {
                    seed[15] = suffix;
                }
                seeds.push(seed);
            }
        }
        let labels = entropy_labels(&seeds, 2).unwrap();
        assert!(labels[..32].iter().all(|label| label == &labels[0]));
        assert!(labels[32..].iter().all(|label| label == &labels[32]));
        assert_ne!(labels[0], labels[32]);
    }
}
