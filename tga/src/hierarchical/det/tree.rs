use super::model::{DetModel, DetNode};
use super::pattern::TargetPattern;
use crate::{Address, TgaError};
use rayon::prelude::*;
use std::collections::HashSet;
use std::collections::VecDeque;

const PARALLEL_DIMENSION_SCAN_MIN_SEEDS: usize = 4096;
const PARALLEL_VECTORIZE_MIN_SEEDS: usize = 4096;

#[derive(Clone)]
pub(super) struct BuildNode {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) children: Vec<BuildNode>,
    pub(super) split_dimension: Option<usize>,
    pub(super) dimension_stack: Vec<usize>,
}

impl BuildNode {
    pub(super) fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            children: Vec::new(),
            split_dimension: None,
            dimension_stack: Vec::new(),
        }
    }

    pub(super) fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    pub(super) fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }
}

pub(super) fn build_model(
    seeds: &[Address],
    bits_per_dimension: usize,
    leaf_max: usize,
) -> DetModel {
    let dimension_count = 128 / bits_per_dimension;
    let base = 1usize << bits_per_dimension;

    let vectors = sorted_seed_vectors(seeds, bits_per_dimension);

    let mut seed_indices = (0..vectors.len()).collect::<Vec<_>>();
    let mut scratch = Vec::new();
    let mut root = BuildNode::new(0, seed_indices.len());
    split_node(
        &mut root,
        &vectors,
        &mut seed_indices,
        &mut scratch,
        leaf_max,
        dimension_count,
        base,
    );
    initialize_dimension_stacks(
        &mut root,
        &vectors,
        &seed_indices,
        &[],
        None,
        dimension_count,
    );

    let mut nodes = Vec::new();
    let root_id = flatten_tree(&root, None, &vectors, &seed_indices, &mut nodes);
    debug_assert_eq!(root_id, 0);
    for node in &mut nodes {
        node.initialize_leaf_targets();
    }
    let current_batch = initial_leaf_queue(&nodes, root_id);

    DetModel {
        generation: Default::default(),
        pending_feedback: Default::default(),
        bits_per_dimension,
        nodes,
        queue: Vec::new(),
        current_batch,
        completed_rounds: 0,
    }
}

pub(super) fn sorted_seed_vectors(seeds: &[Address], bits_per_dimension: usize) -> Vec<Vec<u8>> {
    let mut vectors = if seeds.len() >= PARALLEL_VECTORIZE_MIN_SEEDS {
        seeds
            .par_iter()
            .map(|address| address_to_vector(address, bits_per_dimension))
            .collect::<Vec<_>>()
    } else {
        seeds
            .iter()
            .map(|address| address_to_vector(address, bits_per_dimension))
            .collect::<Vec<_>>()
    };

    if vectors.len() >= PARALLEL_VECTORIZE_MIN_SEEDS {
        vectors.par_sort();
    } else {
        vectors.sort();
    }

    vectors
}

pub(super) fn flatten_tree(
    node: &BuildNode,
    parent: Option<usize>,
    vectors: &[Vec<u8>],
    seed_indices: &[usize],
    out: &mut Vec<DetNode>,
) -> usize {
    let node_id = out.len();
    out.push(DetNode::from_build_node(
        node,
        vectors,
        seed_indices,
        parent,
        Vec::new(),
    ));

    let children = node
        .children
        .iter()
        .map(|child| flatten_tree(child, Some(node_id), vectors, seed_indices, out))
        .collect::<Vec<_>>();
    out[node_id].children = children;
    node_id
}

pub(super) fn initial_leaf_queue(nodes: &[DetNode], root_id: usize) -> Vec<usize> {
    let mut queue = VecDeque::from([root_id]);
    let mut leaves = Vec::new();

    while let Some(node_id) = queue.pop_front() {
        let node = &nodes[node_id];
        if node.children.is_empty() {
            leaves.push(node_id);
        } else {
            queue.extend(node.children.iter().copied());
        }
    }

    leaves
}

pub(super) fn split_node(
    node: &mut BuildNode,
    vectors: &[Vec<u8>],
    seed_indices: &mut [usize],
    scratch: &mut Vec<usize>,
    leaf_max: usize,
    dimension_count: usize,
    base: usize,
) {
    if node.len() <= leaf_max {
        return;
    }

    let Some(split_dim) = find_split_dimension(
        vectors,
        &seed_indices[node.start..node.end],
        dimension_count,
        base,
    ) else {
        return;
    };

    node.split_dimension = Some(split_dim);
    let child_offsets = partition_seed_indices_by_dimension(
        vectors,
        &mut seed_indices[node.start..node.end],
        split_dim,
        base,
        scratch,
    );
    node.children = child_offsets
        .into_iter()
        .map(|(start, end)| BuildNode::new(node.start + start, node.start + end))
        .collect();

    for child in &mut node.children {
        split_node(
            child,
            vectors,
            seed_indices,
            scratch,
            leaf_max,
            dimension_count,
            base,
        );
    }
}

pub(super) fn find_split_dimension(
    vectors: &[Vec<u8>],
    seed_indices: &[usize],
    dimension_count: usize,
    base: usize,
) -> Option<usize> {
    let candidate = |dim| EntropyCandidate::new(vectors, seed_indices, dim, base);
    let best = if seed_indices.len() >= PARALLEL_DIMENSION_SCAN_MIN_SEEDS && dimension_count > 1 {
        (0..dimension_count)
            .into_par_iter()
            .filter_map(candidate)
            .min_by(EntropyCandidate::compare)
    } else {
        (0..dimension_count)
            .filter_map(candidate)
            .min_by(EntropyCandidate::compare)
    };
    best.map(|candidate| candidate.dimension)
}

struct EntropyCandidate {
    dimension: usize,
    counts: Vec<usize>,
    entropy: f64,
}

impl EntropyCandidate {
    fn new(vectors: &[Vec<u8>], indices: &[usize], dimension: usize, base: usize) -> Option<Self> {
        let mut counts = vec![0usize; base];
        for &index in indices {
            counts[vectors[index][dimension] as usize] += 1;
        }
        counts.retain(|&count| count != 0);
        if counts.len() < 2 {
            return None;
        }
        // Equal histograms must have equal entropy regardless of digit labels.
        counts.sort_unstable();
        let total = indices.len() as f64;
        let entropy = counts
            .iter()
            .map(|&count| {
                let probability = count as f64 / total;
                -probability * probability.ln()
            })
            .sum();
        Some(Self {
            dimension,
            counts,
            entropy,
        })
    }

    fn compare(left: &Self, right: &Self) -> std::cmp::Ordering {
        let error = f64::EPSILON
            * (left.counts.len() + right.counts.len()) as f64
            * left.entropy.max(right.entropy);
        let tied = left.counts == right.counts
            || ((left.entropy - right.entropy).abs() <= error
                && entropy_product(&left.counts) == entropy_product(&right.counts));
        if tied {
            left.dimension.cmp(&right.dimension)
        } else {
            left.entropy
                .total_cmp(&right.entropy)
                .then_with(|| left.dimension.cmp(&right.dimension))
        }
    }
}

// Equal sample sizes have equal entropy exactly when their products of count^count agree.
fn entropy_product(counts: &[usize]) -> std::collections::BTreeMap<usize, u128> {
    let mut exponents = std::collections::BTreeMap::new();
    for &count in counts {
        let mut remaining = count;
        let mut prime = 2;
        while prime <= remaining / prime {
            while remaining.is_multiple_of(prime) {
                *exponents.entry(prime).or_default() += count as u128;
                remaining /= prime;
            }
            prime += if prime == 2 { 1 } else { 2 };
        }
        if remaining > 1 {
            *exponents.entry(remaining).or_default() += count as u128;
        }
    }
    exponents
}

pub(super) fn initialize_dimension_stacks(
    node: &mut BuildNode,
    vectors: &[Vec<u8>],
    seed_indices: &[usize],
    parent_stack: &[usize],
    parent_split_dim: Option<usize>,
    dimension_count: usize,
) {
    let mut stack = parent_stack.to_vec();
    let node_seed_indices = &seed_indices[node.start..node.end];

    if let Some(split_dim) = parent_split_dim {
        stack.push(split_dim);
    }

    stack.extend(steady_dimensions(
        vectors,
        node_seed_indices,
        dimension_count,
        &stack,
    ));

    if node.is_leaf() {
        for dim in 0..dimension_count {
            if !stack.contains(&dim) {
                stack.push(dim);
            }
        }
    } else {
        for child in &mut node.children {
            initialize_dimension_stacks(
                child,
                vectors,
                seed_indices,
                &stack,
                node.split_dimension,
                dimension_count,
            );
        }
    }

    node.dimension_stack = stack;
}

pub(super) fn steady_dimensions(
    vectors: &[Vec<u8>],
    seed_indices: &[usize],
    dimension_count: usize,
    stack: &[usize],
) -> Vec<usize> {
    if seed_indices.len() >= PARALLEL_DIMENSION_SCAN_MIN_SEEDS && dimension_count > 1 {
        let mut dims = (0..dimension_count)
            .into_par_iter()
            .filter(|&dim| is_dimension_steady(vectors, seed_indices, dim) && !stack.contains(&dim))
            .collect::<Vec<_>>();
        dims.sort_unstable();
        return dims;
    }

    (0..dimension_count)
        .filter(|&dim| is_dimension_steady(vectors, seed_indices, dim) && !stack.contains(&dim))
        .collect()
}

pub(super) fn is_dimension_steady(vectors: &[Vec<u8>], seed_indices: &[usize], dim: usize) -> bool {
    let Some((&first_index, remaining_indices)) = seed_indices.split_first() else {
        return true;
    };
    let first = vectors[first_index][dim];
    remaining_indices
        .iter()
        .all(|&seed_idx| vectors[seed_idx][dim] == first)
}

pub(super) fn partition_seed_indices_by_dimension(
    vectors: &[Vec<u8>],
    seed_indices: &mut [usize],
    split_dim: usize,
    base: usize,
    scratch: &mut Vec<usize>,
) -> Vec<(usize, usize)> {
    let mut counts = vec![0usize; base];
    let mut bucket_order = Vec::new();
    for &seed_idx in seed_indices.iter() {
        let bucket = vectors[seed_idx][split_dim] as usize;
        if counts[bucket] == 0 {
            bucket_order.push(bucket);
        }
        counts[bucket] += 1;
    }

    scratch.clear();
    scratch.resize(seed_indices.len(), 0);

    let mut starts = vec![0usize; base];
    let mut running = 0usize;
    let mut ranges = Vec::new();
    for &bucket in &bucket_order {
        starts[bucket] = running;
        let count = counts[bucket];
        running += count;
        ranges.push((starts[bucket], running));
    }
    let mut cursor = starts;
    for &seed_idx in seed_indices.iter() {
        let bucket = vectors[seed_idx][split_dim] as usize;
        scratch[cursor[bucket]] = seed_idx;
        cursor[bucket] += 1;
    }
    seed_indices.copy_from_slice(&scratch[..seed_indices.len()]);

    ranges
}

pub(super) fn resolve_bits_per_dim(delta_base: usize) -> Result<usize, TgaError> {
    if delta_base == 0 || !delta_base.is_power_of_two() {
        return Err(TgaError::Training(
            "`delta_base` must be a non-zero power of two".into(),
        ));
    }

    let bits = delta_base.trailing_zeros() as usize;
    if bits == 0 || 128 % bits != 0 {
        return Err(TgaError::Training(
            "bits per dimension derived from `delta_base` must be a non-zero divisor of 128".into(),
        ));
    }
    if bits > 8 {
        return Err(TgaError::Training(
            "bits per dimension derived from `delta_base` must be <= 8 so digits fit in a byte"
                .into(),
        ));
    }

    Ok(bits)
}

pub(super) fn address_to_vector(address: &Address, bits_per_dimension: usize) -> Vec<u8> {
    let dimension_count = 128 / bits_per_dimension;
    let mut vector = vec![0u8; dimension_count];
    let mut value = u128::from_be_bytes(*address);
    let mask = (1u128 << bits_per_dimension) - 1;

    for index in (0..dimension_count).rev() {
        vector[index] = (value & mask) as u8;
        value >>= bits_per_dimension;
    }

    vector
}

#[cfg(test)]
pub(super) fn vector_to_address(vector: &[u8], bits_per_dimension: usize) -> Address {
    let mut value = 0u128;
    for digit in vector {
        value <<= bits_per_dimension;
        value |= *digit as u128;
    }
    value.to_be_bytes()
}

impl DetNode {
    pub(super) fn from_build_node(
        node: &BuildNode,
        vectors: &[Vec<u8>],
        seed_indices: &[usize],
        parent: Option<usize>,
        children: Vec<usize>,
    ) -> Self {
        let target_patterns = if node.is_leaf() {
            seed_indices[node.start..node.end]
                .iter()
                .map(|&seed_idx| TargetPattern {
                    digits: vectors[seed_idx]
                        .iter()
                        .map(|&digit| digit as u16)
                        .collect(),
                })
                .collect()
        } else {
            Vec::new()
        };

        Self {
            parent,
            children,
            dimension_stack: node.dimension_stack.clone(),
            target_patterns,
            scanned_addresses: Vec::new(),
            skipped_addresses: HashSet::new(),
            active_hits: 0,
            active_density: 0.0,
        }
    }
}
