use crate::Address;
use linfa::prelude::{Fit, Predict};
use linfa_clustering::KMeans;
use ndarray::Array2;
use rand::{SeedableRng, rngs::StdRng};
use std::collections::BTreeMap;

use super::models::{NYBBLE_COUNT, nybble_to_id};
use super::{SixGan, SixGanClassification};

#[cfg(test)]
use super::models::MAX_SEQ_LEN;

#[derive(Debug, Clone)]
pub(super) struct SeedClass {
    pub label: String,
    pub rows: Vec<Vec<usize>>,
}

pub(crate) fn classify_seeds(config: &SixGan, seeds: &[Address]) -> Result<Vec<SeedClass>, String> {
    let training_sequences = seeds
        .iter()
        .map(address_to_token_sequence)
        .collect::<Vec<_>>();

    match config.classification {
        SixGanClassification::None => {
            if seeds.len() < config.batch_size {
                return Err(format!(
                    "6GAN requires at least {} active seeds; got {}",
                    config.batch_size,
                    seeds.len()
                ));
            }
            Ok(vec![SeedClass {
                label: "all".into(),
                rows: training_sequences,
            }])
        }
        SixGanClassification::RfcBased => classify_by_labels(
            training_sequences,
            classify_rfc_groups(seeds),
            config.batch_size,
        ),
        SixGanClassification::EntropyClustering => classify_by_labels(
            training_sequences,
            classify_entropy_clusters(seeds, config.entropy_k, config.seed)?,
            config.batch_size,
        ),
        SixGanClassification::Ipv62Vec => classify_by_labels(
            training_sequences,
            super::ipv62vec::classify(seeds, config.batch_size, config.seed)?,
            config.batch_size,
        ),
    }
}

pub(crate) fn address_to_token_sequence(addr: &Address) -> Vec<usize> {
    let mut seq = Vec::with_capacity(NYBBLE_COUNT);
    for &byte in addr {
        seq.push(nybble_to_id(byte >> 4));
        seq.push(nybble_to_id(byte & 0x0f));
    }
    seq
}

fn classify_by_labels(
    training_sequences: Vec<Vec<usize>>,
    labels_by_address: Vec<String>,
    batch_size: usize,
) -> Result<Vec<SeedClass>, String> {
    let mut groups = BTreeMap::<String, Vec<Vec<usize>>>::new();
    for (row, label) in training_sequences.into_iter().zip(labels_by_address) {
        groups.entry(label).or_default().push(row);
    }
    let classes = groups
        .into_iter()
        .filter(|(_, rows)| rows.len() >= batch_size)
        .map(|(label, rows)| SeedClass { label, rows })
        .collect::<Vec<_>>();

    if classes.is_empty() {
        return Err(format!(
            "seed classification produced no classes with at least {} seeds",
            batch_size
        ));
    }

    Ok(classes)
}

fn classify_rfc_groups(seeds: &[Address]) -> Vec<String> {
    seeds.iter().map(classify_rfc_label).collect()
}

pub(crate) fn classify_rfc_label(addr: &Address) -> String {
    let iid = &addr[8..16];

    if is_ieee_derived(iid) {
        "ieee-derived".to_string()
    } else if is_isatap(iid) {
        "isatap".to_string()
    } else if is_embedded_ipv4_32(iid) {
        "embedded-ipv4".to_string()
    } else if is_embedded_port(iid) {
        "embedded-port".to_string()
    } else if is_low_byte(iid) {
        "low-byte".to_string()
    } else if is_embedded_ipv4_64(iid) {
        "embedded-ipv4".to_string()
    } else if zero_byte_iid(iid) > 2 {
        "pattern-bytes".to_string()
    } else {
        "randomized".to_string()
    }
}

fn is_ieee_derived(iid: &[u8]) -> bool {
    iid.len() == 8 && (iid[0] & 0x02) == 0x02 && iid[3] == 0xff && iid[4] == 0xfe
}

fn is_isatap(iid: &[u8]) -> bool {
    iid.len() == 8 && (iid[0] & 0xfd) == 0x00 && iid[1] == 0x00 && iid[2] == 0x5e && iid[3] == 0xfe
}

fn is_embedded_port(iid: &[u8]) -> bool {
    if iid.len() != 8 || iid[..4].iter().any(|byte| *byte != 0) {
        return false;
    }

    let forward = iid[4] == 0 && is_service_port(u16::from_be_bytes([iid[6], iid[7]]));
    let reverse = iid[6] == 0 && is_service_port(u16::from_be_bytes([iid[4], iid[5]]));
    forward || reverse
}

fn is_low_byte(iid: &[u8]) -> bool {
    iid.len() == 8
        && iid[..4].iter().all(|byte| *byte == 0)
        && iid[4] == 0
        && u16::from_be_bytes([iid[6], iid[7]]) != 0
}

fn is_embedded_ipv4_32(iid: &[u8]) -> bool {
    iid.len() == 8
        && iid[..4].iter().all(|byte| *byte == 0)
        && iid[4] != 0
        && u16::from_be_bytes([iid[6], iid[7]]) != 0
}

fn is_embedded_ipv4_64(iid: &[u8]) -> bool {
    if iid.len() != 8 {
        return false;
    }
    let hextets = [
        u16::from_be_bytes([iid[0], iid[1]]),
        u16::from_be_bytes([iid[2], iid[3]]),
        u16::from_be_bytes([iid[4], iid[5]]),
        u16::from_be_bytes([iid[6], iid[7]]),
    ];
    hextets.iter().all(|value| *value <= 0x0255)
}

fn zero_byte_iid(iid: &[u8]) -> usize {
    iid.iter().filter(|byte| **byte == 0).count()
}

fn is_service_port(port: u16) -> bool {
    matches!(
        port,
        0x21 | 0x22
            | 0x23
            | 0x25
            | 0x49
            | 0x53
            | 0x80
            | 0x110
            | 0x123
            | 0x179
            | 0x220
            | 0x389
            | 0x443
            | 0x547
            | 0x993
            | 0x995
            | 0x1194
            | 0x3306
            | 0x5060
            | 0x5061
            | 0x5432
            | 0x6446
            | 0x8080
            | 21
            | 22
            | 23
            | 25
            | 49
            | 53
            | 80
            | 110
            | 123
            | 179
            | 220
            | 389
            | 443
            | 547
            | 993
            | 995
            | 1194
            | 3306
            | 5060
            | 5061
            | 5432
            | 6446
            | 8080
    )
}

fn classify_entropy_clusters(
    seeds: &[Address],
    k: usize,
    seed: u64,
) -> Result<Vec<String>, String> {
    let mut groups: BTreeMap<[u8; 4], Vec<usize>> = BTreeMap::new();
    for (index, addr) in seeds.iter().enumerate() {
        let prefix = [addr[0], addr[1], addr[2], addr[3]];
        groups.entry(prefix).or_default().push(index);
    }

    if groups.len() < k {
        return Err(format!(
            "entropy clustering requires at least {k} distinct /32 prefixes; got {}",
            groups.len()
        ));
    }

    let mut prefixes = Vec::with_capacity(groups.len());
    let mut flat_features = Vec::with_capacity(groups.len() * 24);
    for (prefix, indices) in &groups {
        prefixes.push(*prefix);
        flat_features.extend(compute_entropy_profile(seeds, indices));
    }

    let features = Array2::from_shape_vec((prefixes.len(), 24), flat_features)
        .map_err(|err| format!("failed to assemble entropy feature matrix: {err}"))?;
    let dataset = linfa::DatasetBase::from(features);
    let rng = StdRng::seed_from_u64(seed);
    let model = KMeans::params_with_rng(k, rng)
        .max_n_iterations(300)
        .tolerance(1e-4)
        .fit(&dataset)
        .map_err(|err| format!("entropy clustering k-means failed: {err:?}"))?;
    let assignments = model.predict(&dataset);

    let mut prefix_to_cluster = BTreeMap::new();
    for (prefix, cluster_id) in prefixes.into_iter().zip(assignments.iter()) {
        prefix_to_cluster.insert(prefix, format!("cluster_{cluster_id}"));
    }

    let mut labels = vec![String::new(); seeds.len()];
    for (prefix, indices) in groups {
        let cluster = prefix_to_cluster
            .get(&prefix)
            .cloned()
            .ok_or_else(|| format!("missing entropy cluster assignment for prefix {prefix:?}"))?;
        for index in indices {
            labels[index] = cluster.clone();
        }
    }
    Ok(labels)
}

fn compute_entropy_profile(seeds: &[Address], indices: &[usize]) -> [f64; 24] {
    let mut out = [0.0; 24];
    if indices.is_empty() {
        return out;
    }

    for (feature_index, entropy) in out.iter_mut().enumerate() {
        let nibble_index = 8 + feature_index;
        let mut counts = [0usize; 16];
        for &seed_index in indices {
            let byte = seeds[seed_index][nibble_index / 2];
            let nibble = if nibble_index % 2 == 0 {
                byte >> 4
            } else {
                byte & 15
            };
            counts[nibble as usize] += 1;
        }
        *entropy = normalized_entropy(&counts, indices.len());
    }

    out
}

fn normalized_entropy(counts: &[usize; 16], total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let total = total as f64;
    let entropy = counts
        .iter()
        .filter(|count| **count > 0)
        .map(|count| {
            let probability = *count as f64 / total;
            -probability * probability.log2()
        })
        .sum::<f64>();
    entropy / 4.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        collections::BTreeMap,
        fs::File,
        io::{BufRead, BufReader},
        net::Ipv6Addr,
        str::FromStr,
    };

    fn bundled_seed_path() -> &'static str {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../prior-tgas/tgas/2021-6gan/6GAN/data/source_data/responsive-addresses.txt"
        )
    }

    fn bundled_rfc_profile_path() -> &'static str {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../prior-tgas/tgas/2021-6gan/6GAN/data/save_data/rfc_profile.txt"
        )
    }

    fn load_addresses(path: &str) -> Vec<Address> {
        let file = File::open(path).unwrap();
        BufReader::new(file)
            .lines()
            .take(51_200)
            .map(|line| Ipv6Addr::from_str(&line.unwrap()).unwrap().octets())
            .collect()
    }

    #[test]
    #[ignore = "requires local prior-tgas 6GAN reference datasets"]
    fn rfc_classifier_matches_bundled_profile_counts() {
        let seeds = load_addresses(bundled_seed_path());
        let produced = seeds.iter().map(classify_rfc_label).fold(
            BTreeMap::<String, usize>::new(),
            |mut map, label| {
                *map.entry(label).or_default() += 1;
                map
            },
        );

        let expected = BufReader::new(File::open(bundled_rfc_profile_path()).unwrap())
            .lines()
            .map(|line| {
                let line = line.unwrap();
                line.split('=').nth(3).unwrap().to_string()
            })
            .fold(BTreeMap::<String, usize>::new(), |mut map, label| {
                *map.entry(label).or_default() += 1;
                map
            });

        assert_eq!(produced, expected);
    }

    #[test]
    #[ignore = "requires local prior-tgas 6GAN reference datasets"]
    fn entropy_profile_matches_bundled_prefix() {
        let seeds = load_addresses(bundled_seed_path());
        let first_prefix_indices = seeds
            .iter()
            .enumerate()
            .filter(|(_, addr)| addr[..4] == [0x20, 0x01, 0x04, 0x28])
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let profile = compute_entropy_profile(&seeds, &first_prefix_indices);

        let expected = [
            0.121, 0.121, 0.000, 0.125, 0.000, 0.581, 0.507, 0.983, 0.013, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.117, 0.123, 0.106, 0.130, 0.536, 0.894, 0.817, 0.836,
        ];

        for (actual, expected) in profile.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 0.01, "{actual} vs {expected}");
        }
    }

    #[test]
    #[ignore = "requires local prior-tgas 6GAN reference datasets"]
    fn seed_classification_filters_small_rfc_classes() {
        let seeds = load_addresses(bundled_seed_path());
        let cfg = SixGan {
            batch_size: 64,
            classification: SixGanClassification::RfcBased,
            ..SixGan::default()
        };
        let classified = classify_seeds(&cfg, &seeds).unwrap();
        assert_eq!(
            classified
                .iter()
                .map(|class| class.label.clone())
                .collect::<Vec<_>>(),
            vec![
                "embedded-ipv4".to_string(),
                "embedded-port".to_string(),
                "ieee-derived".to_string(),
                "low-byte".to_string(),
                "pattern-bytes".to_string(),
                "randomized".to_string(),
            ]
        );
    }

    #[test]
    fn rfc_classifier_preserves_historical_addr6_edge_cases() {
        let cases = [
            ("2001:7fe::53", "embedded-port"),
            ("2001:8000:100::40:123", "embedded-port"),
            ("2001:a60::123:1", "embedded-ipv4"),
            ("2a00:1940:107::6:0", "embedded-ipv4"),
            ("2001:310:6000:f::516b:0", "pattern-bytes"),
        ];

        for (addr, expected) in cases {
            let addr = Ipv6Addr::from_str(addr).unwrap().octets();
            assert_eq!(classify_rfc_label(&addr), expected);
        }
    }

    #[test]
    fn token_sequences_are_32_nybbles() {
        let addr = Ipv6Addr::from_str("2001:db8::1").unwrap().octets();
        let tokens = address_to_token_sequence(&addr);
        assert_eq!(tokens.len(), NYBBLE_COUNT);
        assert_eq!(tokens.len(), MAX_SEQ_LEN);
    }
}
