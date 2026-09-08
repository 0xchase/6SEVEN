use super::pattern_index::PatternIndex;
use super::{HEX_RADIX, NIBBLE_COUNT, Pattern, SixGraph};
use crate::address::{Address, AddressExt};
use rayon::prelude::*;
use std::collections::VecDeque;

#[derive(Debug, Default)]
pub(super) struct MiningResult {
    pub(super) patterns: Vec<Pattern>,
    pub(super) outliers: Vec<Address>,
}

pub(super) fn mine_patterns(initial_addresses: Vec<Address>, config: &SixGraph) -> MiningResult {
    let mut addresses = initial_addresses;
    let mut mined = MiningResult::default();

    // Refine the previous round of outliers after the initial partition.
    for round_idx in 0..=config.iterations {
        if addresses.is_empty() {
            break;
        }

        let MiningResult {
            mut patterns,
            mut outliers,
        } = mine_one_round(
            &addresses,
            config.min_region_size,
            config.distance_threshold,
        );
        mined.patterns.append(&mut patterns);
        if config.seed_rejoining {
            rejoin_outliers(&mut mined.patterns, &mut outliers);
        }

        if round_idx == config.iterations || outliers == addresses {
            mined.outliers = outliers;
            break;
        }

        if outliers.is_empty() {
            mined.outliers = Vec::new();
            break;
        }
        addresses = outliers;
    }

    mined
}

pub(super) fn rejoin_outliers(patterns: &mut [Pattern], outliers: &mut Vec<Address>) {
    if patterns.is_empty() || outliers.is_empty() {
        return;
    }
    let index = PatternIndex::new(patterns);
    outliers.retain(|&address| {
        // Resolve overlapping patterns by mining order to keep seed ownership unique.
        if let Some(id) = index.matches(address).min() {
            patterns[id].seeds.push(address);
            false
        } else {
            true
        }
    });
}

fn mine_one_round(
    addresses: &[Address],
    min_region_size: usize,
    distance_threshold: usize,
) -> MiningResult {
    let round_results = space_partition_regions(addresses, min_region_size)
        .into_par_iter()
        .with_min_len(64)
        .map(|region| outlier_detect(&region, distance_threshold))
        .collect::<Vec<_>>();

    let mut result = MiningResult::default();
    for MiningResult {
        mut patterns,
        mut outliers,
    } in round_results
    {
        result.patterns.append(&mut patterns);
        result.outliers.append(&mut outliers);
    }
    result
}

/// Partition breadth-first using the leftmost varying nibble.
pub(super) fn space_partition_regions(
    addresses: &[Address],
    min_region_size: usize,
) -> Vec<Vec<Address>> {
    let mut regions = Vec::new();
    if addresses.is_empty() {
        return regions;
    }

    let mut queue = VecDeque::new();
    queue.push_back((0..addresses.len()).collect::<Vec<_>>());

    while let Some(indices) = queue.pop_front() {
        if indices.len() <= min_region_size {
            regions.push(materialize_region(addresses, &indices));
            continue;
        }

        let Some(split_dim) = leftmost_varying_nibble(addresses, &indices) else {
            regions.push(materialize_region(addresses, &indices));
            continue;
        };

        let mut groups: [Vec<usize>; HEX_RADIX] = std::array::from_fn(|_| Vec::new());
        for idx in indices {
            groups[addresses[idx].get_nibble(split_dim) as usize].push(idx);
        }

        for group in groups.into_iter().filter(|group| !group.is_empty()) {
            queue.push_back(group);
        }
    }

    regions
}

fn materialize_region(addresses: &[Address], indices: &[usize]) -> Vec<Address> {
    indices.iter().map(|&idx| addresses[idx]).collect()
}

fn leftmost_varying_nibble(addresses: &[Address], indices: &[usize]) -> Option<usize> {
    if indices.len() < 2 {
        return None;
    }

    for nibble_idx in 0..NIBBLE_COUNT {
        let first = addresses[indices[0]].get_nibble(nibble_idx);
        if indices
            .iter()
            .skip(1)
            .any(|&idx| addresses[idx].get_nibble(nibble_idx) != first)
        {
            return Some(nibble_idx);
        }
    }

    None
}

/// Extract patterns using the reference density-gated Kruskal forest.
pub(super) fn outlier_detect(addresses: &[Address], distance_threshold: usize) -> MiningResult {
    if addresses.len() < 2 {
        return MiningResult {
            patterns: Vec::new(),
            outliers: addresses.to_vec(),
        };
    }

    let components = graph_cut_components(addresses, distance_threshold);

    let mut patterns: Vec<Pattern> = Vec::new();
    let mut outliers: Vec<Address> = Vec::new();

    for comp in components {
        if comp.len() > 1 {
            patterns.push(Pattern::from_component(addresses, &comp));
        } else {
            outliers.push(addresses[comp[0]]);
        }
    }

    MiningResult { patterns, outliers }
}

/// Merge distance-bounded edges when component density improves.
pub(super) fn graph_cut_components(
    addresses: &[Address],
    distance_threshold: usize,
) -> Vec<Vec<usize>> {
    let n = addresses.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![vec![0]];
    }

    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let distance = addresses[i].nibble_distance(&addresses[j]) as usize;
            if distance <= distance_threshold {
                edges.push(GraphEdge { i, j, distance });
            }
        }
    }
    edges.sort_by_key(|edge| (edge.distance, edge.i, edge.j));

    let mut forest = ComponentForest::new(addresses);
    for edge in edges {
        forest.try_merge(edge.i, edge.j);
    }
    forest.components()
}

/// Cached component summaries make density checks independent of component size.
struct ComponentForest {
    parent: Vec<usize>,
    summaries: Vec<ComponentSummary>,
}

#[derive(Clone, Copy)]
struct ComponentSummary {
    representative: Address,
    size: usize,
    variable: u32,
}

impl ComponentSummary {
    fn merged(self, other: Self) -> Self {
        let mut variable = self.variable | other.variable;
        for nibble in 0..NIBBLE_COUNT {
            if self.representative.get_nibble(nibble) != other.representative.get_nibble(nibble) {
                variable |= 1 << nibble;
            }
        }
        Self {
            representative: self.representative,
            size: self.size + other.size,
            variable,
        }
    }

    /// Compare densities exactly with zero for singletons and infinity for duplicate-only groups.
    fn denser_than(self, other: Self) -> bool {
        if self.size <= 1 {
            return false;
        }
        if other.size <= 1 {
            return true;
        }
        let free = self.variable.count_ones() as u128;
        let other_free = other.variable.count_ones() as u128;
        (self.size as u128) * other_free > (other.size as u128) * free
    }
}

impl ComponentForest {
    fn new(addresses: &[Address]) -> Self {
        Self {
            parent: (0..addresses.len()).collect(),
            summaries: addresses
                .iter()
                .map(|&representative| ComponentSummary {
                    representative,
                    size: 1,
                    variable: 0,
                })
                .collect(),
        }
    }

    fn root(&mut self, mut index: usize) -> usize {
        while self.parent[index] != index {
            self.parent[index] = self.parent[self.parent[index]];
            index = self.parent[index];
        }
        index
    }

    fn try_merge(&mut self, i: usize, j: usize) {
        let mut left = self.root(i);
        let mut right = self.root(j);
        if left == right {
            return;
        }
        let merged = self.summaries[left].merged(self.summaries[right]);
        if !merged.denser_than(self.summaries[left]) || !merged.denser_than(self.summaries[right]) {
            return;
        }
        // Union by size bounds tree depth without changing edge acceptance or output order.
        if self.summaries[left].size < self.summaries[right].size {
            std::mem::swap(&mut left, &mut right);
        }
        self.parent[right] = left;
        self.summaries[left] = merged;
    }

    fn components(mut self) -> Vec<Vec<usize>> {
        let mut components = Vec::<Vec<usize>>::new();
        let mut slots = vec![None; self.parent.len()];
        for index in 0..self.parent.len() {
            let root = self.root(index);
            let slot = *slots[root].get_or_insert_with(|| {
                components.push(Vec::new());
                components.len() - 1
            });
            components[slot].push(index);
        }
        components
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GraphEdge {
    i: usize,
    j: usize,
    distance: usize,
}

/// Return sorted indices of nibble positions where addresses differ.
pub(super) fn variable_nibble_indices(addresses: &[Address], idxs: &[usize]) -> Vec<usize> {
    if idxs.len() <= 1 {
        return Vec::new();
    }
    let first = &addresses[idxs[0]];
    let mut result = Vec::new();
    for nib in 0..NIBBLE_COUNT {
        let baseline = first.get_nibble(nib);
        for &idx in idxs.iter().skip(1) {
            if addresses[idx].get_nibble(nib) != baseline {
                result.push(nib);
                break;
            }
        }
    }
    result
}
