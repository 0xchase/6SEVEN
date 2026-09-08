use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::address::Address;
use crate::pattern::AddressPattern;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cluster {
    pub pattern: AddressPattern,
    pub seed_count: usize,
}

impl Cluster {
    pub fn new(seed_addr: &Address) -> Self {
        Self {
            pattern: AddressPattern::new(*seed_addr),
            seed_count: 1,
        }
    }

    pub fn with_pattern(pattern: AddressPattern, seed_count: usize) -> Self {
        Self {
            pattern,
            seed_count,
        }
    }

    pub fn density(&self) -> f64 {
        self.pattern.density(self.seed_count)
    }

    pub fn size(&self) -> usize {
        self.pattern.size()
    }

    pub fn contains(&self, addr: &Address) -> bool {
        self.pattern.matches(addr)
    }

    pub fn merge_address(&mut self, addr: &Address) {
        self.pattern.merge_address(addr);
        self.seed_count += 1;
    }
}

#[derive(Debug, Clone)]
pub struct UnifiedDhcConfig {
    pub min_region_size: usize,
    pub max_depth: Option<usize>,
    pub strategy: DhcStrategy,
    pub stop_exclusive: bool,
    pub use_lifo: bool,
}

impl Default for UnifiedDhcConfig {
    fn default() -> Self {
        Self {
            min_region_size: 16,
            max_depth: None,
            strategy: DhcStrategy::LeftmostDifference,
            stop_exclusive: false,
            use_lifo: false,
        }
    }
}

impl UnifiedDhcConfig {
    pub fn with_strategy(min_region_size: usize, strategy: DhcStrategy) -> Self {
        Self {
            min_region_size,
            strategy,
            ..Default::default()
        }
    }

    fn should_stop(&self, len: usize) -> bool {
        if self.stop_exclusive {
            len < self.min_region_size
        } else {
            len <= self.min_region_size
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DhcStrategy {
    LeftmostDifference,
    RightmostDifference,
    MinEntropy,
    MaxCoverage,
    MaxSeparation,
}

#[derive(Debug, Clone)]
pub struct DhcResult {
    pub regions: Vec<Vec<Address>>,
    pub clusters: Vec<Cluster>,
}

#[derive(Debug, Clone)]
pub struct DhcProgress {
    pub processed_regions: usize,
    pub queue_len: usize,
    pub current_region_size: usize,
    pub split_dimension: Option<usize>,
}

pub fn perform_unified_dhc(addresses: &[Address], config: &UnifiedDhcConfig) -> DhcResult {
    perform_unified_dhc_with_progress(addresses, config, |_| {})
}

pub fn perform_unified_dhc_with_progress<F>(
    addresses: &[Address],
    config: &UnifiedDhcConfig,
    mut progress: F,
) -> DhcResult
where
    F: FnMut(DhcProgress),
{
    assert!(
        addresses.len() <= u32::MAX as usize,
        "DHC currently supports up to u32::MAX addresses"
    );

    let mut regions: Vec<Vec<Address>> = Vec::new();
    let mut queue = std::collections::VecDeque::new();

    let initial_indices: Vec<u32> = (0..addresses.len() as u32).collect();
    queue.push_back((initial_indices, 0usize));

    progress(DhcProgress {
        processed_regions: 0,
        queue_len: queue.len(),
        current_region_size: addresses.len(),
        split_dimension: None,
    });

    while let Some((indices, depth)) = if config.use_lifo {
        queue.pop_back()
    } else {
        queue.pop_front()
    } {
        let reached_max_depth = config.max_depth.is_some_and(|max| depth >= max);
        if config.should_stop(indices.len()) || reached_max_depth {
            let region = indices
                .iter()
                .map(|&idx| addresses[idx as usize])
                .collect::<Vec<_>>();
            regions.push(region);
            progress(DhcProgress {
                processed_regions: regions.len(),
                queue_len: queue.len(),
                current_region_size: indices.len(),
                split_dimension: None,
            });
            continue;
        }

        let split_dim = match config.strategy {
            DhcStrategy::LeftmostDifference => find_leftmost_diff(addresses, &indices),
            DhcStrategy::RightmostDifference => find_rightmost_diff(addresses, &indices),
            DhcStrategy::MinEntropy => find_min_entropy(addresses, &indices),
            DhcStrategy::MaxCoverage => find_max_coverage(addresses, &indices),
            DhcStrategy::MaxSeparation => find_max_separation(addresses, &indices),
        };

        if let Some(dim) = split_dim {
            let mut groups: [Vec<u32>; 16] = std::array::from_fn(|_| Vec::new());
            for &idx in &indices {
                let val = get_nibble(&addresses[idx as usize], dim) as usize;
                groups[val].push(idx);
            }

            for group in groups.into_iter().filter(|g| !g.is_empty()) {
                queue.push_back((group, depth + 1));
            }
            progress(DhcProgress {
                processed_regions: regions.len(),
                queue_len: queue.len(),
                current_region_size: indices.len(),
                split_dimension: Some(dim),
            });
        } else {
            let region = indices
                .iter()
                .map(|&idx| addresses[idx as usize])
                .collect::<Vec<_>>();
            regions.push(region);
            progress(DhcProgress {
                processed_regions: regions.len(),
                queue_len: queue.len(),
                current_region_size: indices.len(),
                split_dimension: None,
            });
        }
    }

    let clusters = regions_to_clusters(&regions);

    DhcResult { regions, clusters }
}

fn regions_to_clusters(regions: &[Vec<Address>]) -> Vec<Cluster> {
    regions
        .iter()
        .filter_map(|group| {
            let addresses: Vec<Address> = group.to_vec();

            if addresses.is_empty() {
                None
            } else {
                let pattern = if addresses.len() == 1 {
                    AddressPattern::new(addresses[0])
                } else {
                    let mut pattern = AddressPattern::new(addresses[0]);
                    for addr in &addresses[1..] {
                        pattern.merge_address(addr);
                    }
                    pattern
                };
                Some(Cluster::with_pattern(pattern, addresses.len()))
            }
        })
        .collect()
}

fn find_leftmost_diff(addresses: &[Address], indices: &[u32]) -> Option<usize> {
    if indices.len() < 2 {
        return None;
    }

    for dim in 0..32 {
        let first_val = get_nibble(&addresses[indices[0] as usize], dim);
        let all_same = indices
            .iter()
            .all(|&idx| get_nibble(&addresses[idx as usize], dim) == first_val);

        if !all_same {
            return Some(dim);
        }
    }

    None
}

fn find_rightmost_diff(addresses: &[Address], indices: &[u32]) -> Option<usize> {
    if indices.len() < 2 {
        return None;
    }

    for dim in (0..32).rev() {
        let first_val = get_nibble(&addresses[indices[0] as usize], dim);
        let all_same = indices
            .iter()
            .all(|&idx| get_nibble(&addresses[idx as usize], dim) == first_val);

        if !all_same {
            return Some(dim);
        }
    }

    None
}

fn dimension_scores(count: usize, score: impl Fn(usize) -> f64 + Sync) -> [f64; 32] {
    if count < 4096 {
        return std::array::from_fn(score);
    }
    let mut scores = [0.0; 32];
    scores
        .par_iter_mut()
        .enumerate()
        .for_each(|(dim, value)| *value = score(dim));
    scores
}

fn find_min_entropy(addresses: &[Address], indices: &[u32]) -> Option<usize> {
    if indices.len() < 2 {
        return None;
    }

    let (dim, entropy) = dimension_scores(indices.len(), |dim| {
        calculate_entropy(addresses, indices, dim)
    })
    .into_iter()
    .enumerate()
    .filter(|(_, entropy)| *entropy > 0.0)
    .fold((0, f64::INFINITY), |a, b| if b.1 < a.1 { b } else { a });

    if entropy < f64::INFINITY {
        Some(dim)
    } else {
        None
    }
}

fn find_max_coverage(addresses: &[Address], indices: &[u32]) -> Option<usize> {
    if indices.len() < 2 {
        return None;
    }

    let scores: Vec<(usize, f64)> = dimension_scores(indices.len(), |dim| {
        calculate_coverage_score(addresses, indices, dim)
    })
    .into_iter()
    .enumerate()
    .filter(|(_, score)| *score >= 0.0)
    .collect();

    if scores.is_empty() {
        return None;
    }

    let leftmost_index = scores[0].0;
    let leftmost_covering = scores[0].1;

    let (best_index, best_covering) = scores
        .iter()
        .copied()
        .reduce(|a, b| if b.1 > a.1 { b } else { a })
        .unwrap();

    if best_covering - leftmost_covering <= (best_index as f64 - leftmost_index as f64) {
        return Some(leftmost_index);
    }

    Some(best_index)
}

/// Find the max coverage split dimension directly from a slice of addresses.
pub fn find_max_coverage_dim(addresses: &[Address]) -> Option<usize> {
    let indices: Vec<u32> = (0..addresses.len() as u32).collect();
    find_max_coverage(addresses, &indices)
}

fn find_max_separation(addresses: &[Address], indices: &[u32]) -> Option<usize> {
    if indices.len() < 2 {
        return None;
    }

    let (dim, separation) = dimension_scores(indices.len(), |dim| {
        let mut values: Vec<u8> = indices
            .iter()
            .map(|&idx| get_nibble(&addresses[idx as usize], dim))
            .collect();
        values.sort();
        values.dedup();
        values.len() as f64
    })
    .into_iter()
    .enumerate()
    .fold((0, 0.0f64), |a, b| if b.1 > a.1 { b } else { a });

    if separation > 1.0 { Some(dim) } else { None }
}

fn calculate_entropy(addresses: &[Address], indices: &[u32], dimension: usize) -> f64 {
    let mut counts: HashMap<u8, usize> = HashMap::new();

    for &idx in indices {
        let val = get_nibble(&addresses[idx as usize], dimension);
        *counts.entry(val).or_insert(0) += 1;
    }

    let total = indices.len() as f64;
    let mut entropy = 0.0;

    for &count in counts.values() {
        if count > 0 {
            let p = count as f64 / total;
            entropy -= p * p.log2();
        }
    }

    entropy
}

fn calculate_coverage_score(addresses: &[Address], indices: &[u32], dimension: usize) -> f64 {
    let mut counts = [0usize; 16];
    for &idx in indices {
        let val = get_nibble(&addresses[idx as usize], dimension) as usize;
        counts[val] += 1;
    }

    let unique = counts.iter().filter(|&&c| c > 0).count();
    if unique <= 1 {
        return -1.0;
    }

    let non_singleton_sum: usize = counts.iter().filter(|&&c| c != 1).copied().sum();
    non_singleton_sum as f64
}

fn get_nibble(addr: &Address, dim: usize) -> u8 {
    let byte_index = dim / 2;
    let byte = addr[byte_index];
    if dim.is_multiple_of(2) {
        byte >> 4 // High nibble
    } else {
        byte & 0x0F // Low nibble
    }
}

use kodama::Method;
use linfa::prelude::*;
use linfa_clustering::{Dbscan, KMeans};
use linfa_hierarchical::HierarchicalCluster;
use linfa_kernel::{Kernel, KernelMethod};
use ndarray::Array2;

#[derive(Debug, Clone)]
pub struct KMeansConfig {
    pub n_clusters: usize,
    pub max_iter: usize,
    pub tolerance: f32,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            n_clusters: 10,
            max_iter: 300,
            tolerance: 1e-4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DbscanConfig {
    pub tolerance: f64,
    pub min_points: usize,
}

impl Default for DbscanConfig {
    fn default() -> Self {
        Self {
            tolerance: 3.0,
            min_points: 5,
        }
    }
}

fn addresses_to_features(addresses: &[Address]) -> Array2<f64> {
    let n_samples = addresses.len();
    let n_features = 16;

    let mut features = Array2::<f64>::zeros((n_samples, n_features));

    for (i, addr) in addresses.iter().enumerate() {
        for (j, &byte) in addr.iter().enumerate() {
            features[[i, j]] = byte as f64;
        }
    }

    features
}

pub fn kmeans_cluster(
    addresses: &[Address],
    config: &KMeansConfig,
) -> Result<Vec<Cluster>, String> {
    if addresses.is_empty() {
        return Ok(Vec::new());
    }

    if addresses.len() < config.n_clusters {
        return Err(format!(
            "Not enough addresses ({}) for {} clusters",
            addresses.len(),
            config.n_clusters
        ));
    }

    let features = addresses_to_features(addresses);
    let dataset = DatasetBase::from(features);

    let model = KMeans::params(config.n_clusters)
        .max_n_iterations(config.max_iter as u64)
        .tolerance(config.tolerance as f64)
        .fit(&dataset)
        .map_err(|e| format!("K-means clustering failed: {:?}", e))?;

    let predictions = model.predict(&dataset);

    let mut clusters_map: HashMap<usize, Vec<Address>> = HashMap::new();
    for (i, &cluster_id) in predictions.iter().enumerate() {
        clusters_map
            .entry(cluster_id)
            .or_default()
            .push(addresses[i]);
    }

    let clusters: Vec<Cluster> = clusters_map
        .into_values()
        .map(|addrs| {
            let mut pattern = AddressPattern::new(addrs[0]);
            for addr in &addrs[1..] {
                pattern.merge_address(addr);
            }
            Cluster::with_pattern(pattern, addrs.len())
        })
        .collect();

    Ok(clusters)
}

pub fn dbscan_cluster(
    addresses: &[Address],
    config: &DbscanConfig,
) -> Result<Vec<Cluster>, String> {
    if addresses.is_empty() {
        return Ok(Vec::new());
    }

    let features = addresses_to_features(addresses);

    let clusters = Dbscan::params(config.min_points)
        .tolerance(config.tolerance)
        .transform(&features)
        .map_err(|e| format!("DBSCAN clustering failed: {:?}", e))?;

    let mut clusters_map: HashMap<usize, Vec<Address>> = HashMap::new();
    for (i, cluster_id) in clusters.iter().enumerate() {
        if let Some(id) = cluster_id {
            clusters_map.entry(*id).or_default().push(addresses[i]);
        }
    }

    let result: Vec<Cluster> = clusters_map
        .into_values()
        .map(|addrs| {
            let mut pattern = AddressPattern::new(addrs[0]);
            for addr in &addrs[1..] {
                pattern.merge_address(addr);
            }
            Cluster::with_pattern(pattern, addrs.len())
        })
        .collect();

    Ok(result)
}

#[derive(Debug, Clone)]
pub struct HierarchicalConfig {
    pub n_clusters: usize,
    pub method: Method,
    pub kernel_method: KernelMethod<f64>,
    pub kernel_param: f64,
}

impl Default for HierarchicalConfig {
    fn default() -> Self {
        Self {
            n_clusters: 10,
            method: Method::Average,
            kernel_method: KernelMethod::Gaussian(1.0),
            kernel_param: 1.0,
        }
    }
}

impl HierarchicalConfig {
    pub fn with_method(n_clusters: usize, method: Method) -> Self {
        Self {
            n_clusters,
            method,
            ..Default::default()
        }
    }

    pub fn with_kernel(n_clusters: usize, kernel_method: KernelMethod<f64>) -> Self {
        Self {
            n_clusters,
            kernel_method,
            ..Default::default()
        }
    }
}

pub fn hierarchical_cluster(
    addresses: &[Address],
    config: &HierarchicalConfig,
) -> Result<Vec<Cluster>, String> {
    if addresses.is_empty() {
        return Ok(Vec::new());
    }

    if addresses.len() < config.n_clusters {
        return Err(format!(
            "Not enough addresses ({}) for {} clusters",
            addresses.len(),
            config.n_clusters
        ));
    }

    let features = addresses_to_features(addresses);

    let kernel = Kernel::params()
        .method(config.kernel_method.clone())
        .transform(features.view());

    let predictions = HierarchicalCluster::default()
        .with_method(config.method)
        .num_clusters(config.n_clusters)
        .transform(kernel)
        .map_err(|e| format!("Hierarchical clustering failed: {:?}", e))?;

    let mut clusters_map: HashMap<usize, Vec<Address>> = HashMap::new();
    for (i, &cluster_id) in predictions.targets().iter().enumerate() {
        clusters_map
            .entry(cluster_id)
            .or_default()
            .push(addresses[i]);
    }

    let result: Vec<Cluster> = clusters_map
        .into_values()
        .map(|addrs| {
            let mut pattern = AddressPattern::new(addrs[0]);
            for addr in &addrs[1..] {
                pattern.merge_address(addr);
            }
            Cluster::with_pattern(pattern, addrs.len())
        })
        .collect();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv6Addr;
    use std::str::FromStr;

    fn load_ipv6_seeds(path: &str) -> Vec<Address> {
        use std::io::BufRead;
        let file =
            std::fs::File::open(path).unwrap_or_else(|e| panic!("Cannot open {}: {}", path, e));
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

    #[test]
    #[ignore = "requires local ntp-10k.txt reference dataset"]
    fn verify_against_python() {
        let seed_path = concat!(env!("CARGO_MANIFEST_DIR"), "/ntp-10k.txt");
        let seeds = load_ipv6_seeds(seed_path);
        assert_eq!(seeds.len(), 10000);

        let graph_config = UnifiedDhcConfig::with_strategy(16, DhcStrategy::LeftmostDifference);
        let graph_result = perform_unified_dhc(&seeds, &graph_config);
        let graph_total: usize = graph_result.regions.iter().map(|r| r.len()).sum();

        eprintln!(
            "6Graph DHC: {} regions, {} total seeds",
            graph_result.regions.len(),
            graph_total
        );

        assert_eq!(graph_total, 10000);
        assert_eq!(graph_result.regions.len(), 2149);

        let forest_config = UnifiedDhcConfig {
            min_region_size: 16,
            strategy: DhcStrategy::MaxCoverage,
            stop_exclusive: true,
            use_lifo: true,
            ..Default::default()
        };
        let forest_result = perform_unified_dhc(&seeds, &forest_config);
        let forest_total: usize = forest_result.regions.iter().map(|r| r.len()).sum();

        eprintln!(
            "6Forest DHC: {} regions, {} total seeds",
            forest_result.regions.len(),
            forest_total
        );

        assert_eq!(forest_total, 10000);
        assert_eq!(forest_result.regions.len(), 2147);
    }

    fn addr(hex: &str) -> Address {
        assert_eq!(hex.len(), 32);
        let mut out = [0u8; 16];
        for i in 0..16 {
            out[i] = u8::from_str_radix(&hex[(i * 2)..(i * 2 + 2)], 16).unwrap();
        }
        out
    }

    #[test]
    fn max_depth_limits_tree_depth_not_region_count() {
        let seeds = vec![
            addr("00000000000000000000000000000000"),
            addr("10000000000000000000000000000000"),
            addr("11000000000000000000000000000000"),
            addr("11100000000000000000000000000000"),
        ];

        let base = UnifiedDhcConfig {
            min_region_size: 1,
            max_depth: None,
            strategy: DhcStrategy::LeftmostDifference,
            stop_exclusive: true,
            use_lifo: false,
        };

        let unlimited = perform_unified_dhc(&seeds, &base);
        assert_eq!(unlimited.regions.len(), 4);

        let mut depth0 = base.clone();
        depth0.max_depth = Some(0);
        assert_eq!(perform_unified_dhc(&seeds, &depth0).regions.len(), 1);

        let mut depth1 = base.clone();
        depth1.max_depth = Some(1);
        assert_eq!(perform_unified_dhc(&seeds, &depth1).regions.len(), 2);

        let mut depth2 = base;
        depth2.max_depth = Some(2);
        assert_eq!(perform_unified_dhc(&seeds, &depth2).regions.len(), 3);
    }
}
