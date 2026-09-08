use super::{DIMS, Nibbles, range::AddressRange};

#[derive(Debug, Clone)]
struct SeedTrieNode {
    depth: usize,
    suffix: Box<[u8]>,
    children: Box<[(u8, usize)]>,
    leaf_seed: Option<usize>,
}

#[derive(Debug, Clone)]
pub(super) struct SeedTrie {
    nodes: Vec<SeedTrieNode>,
    root: usize,
}

#[derive(Debug, Clone, Copy)]
struct CandidateState {
    node_idx: usize,
    cost: usize,
    outside_range: bool,
}

impl SeedTrie {
    pub(super) fn new(seeds: &[Nibbles]) -> Self {
        debug_assert!(!seeds.is_empty(), "6Gen requires at least one seed");
        debug_assert!(
            seeds.windows(2).all(|pair| pair[0] < pair[1]),
            "6Gen seed trie requires sorted, deduplicated seeds"
        );

        let mut nodes = Vec::new();
        let root = Self::build_node(seeds, 0, seeds.len(), 0, &mut nodes);
        Self { nodes, root }
    }

    pub(super) fn count_in_range(&self, range: &AddressRange) -> usize {
        self.count_in_range_from(self.root, range)
    }

    fn count_in_range_from(&self, node_idx: usize, range: &AddressRange) -> usize {
        let node = &self.nodes[node_idx];
        for (offset, value) in node.suffix.iter().copied().enumerate() {
            if !range.dims[node.depth + offset].contains(value) {
                return 0;
            }
        }

        if node.leaf_seed.is_some() {
            return 1;
        }

        let next_dim = node.depth + node.suffix.len();
        node.children
            .iter()
            .filter(|(value, _)| range.dims[next_dim].contains(*value))
            .map(|(_, child_idx)| self.count_in_range_from(*child_idx, range))
            .sum()
    }

    pub(super) fn nearest_external_seed_indices(&self, range: &AddressRange) -> Vec<usize> {
        // Find the closest external seeds with a Hamming-cost trie walk.
        let mut buckets = vec![Vec::new(); DIMS + 1];
        buckets[0].push(CandidateState {
            node_idx: self.root,
            cost: 0,
            outside_range: false,
        });

        let mut best_cost = None;
        let mut candidates = Vec::new();

        for current_cost in 0..=DIMS {
            if best_cost.is_some_and(|best| current_cost > best) {
                break;
            }

            while let Some(state) = buckets[current_cost].pop() {
                let mut cost = state.cost;
                let mut outside_range = state.outside_range;
                let node = &self.nodes[state.node_idx];

                for (offset, value) in node.suffix.iter().copied().enumerate() {
                    let domain = range.dims[node.depth + offset];
                    let contains_value = domain.contains(value);
                    if !contains_value {
                        outside_range = true;
                        if !domain.is_dynamic() {
                            cost += 1;
                        }
                    }
                }

                if best_cost.is_some_and(|best| cost > best) {
                    continue;
                }

                if let Some(seed_idx) = node.leaf_seed {
                    if outside_range {
                        match best_cost {
                            None => {
                                best_cost = Some(cost);
                                candidates.push(seed_idx);
                            }
                            Some(best) if cost < best => {
                                best_cost = Some(cost);
                                candidates.clear();
                                candidates.push(seed_idx);
                            }
                            Some(best) if cost == best => candidates.push(seed_idx),
                            Some(_) => {}
                        }
                    }
                    continue;
                }

                let next_dim = node.depth + node.suffix.len();
                for (value, child_idx) in node.children.iter().copied() {
                    let domain = range.dims[next_dim];
                    let contains_value = domain.contains(value);
                    let child_outside_range = outside_range || !contains_value;

                    let child_cost = cost + usize::from(!domain.is_dynamic() && !contains_value);
                    if best_cost.is_some_and(|best| child_cost > best) {
                        continue;
                    }
                    buckets[child_cost].push(CandidateState {
                        node_idx: child_idx,
                        cost: child_cost,
                        outside_range: child_outside_range,
                    });
                }
            }
        }

        candidates.sort_unstable();
        candidates
    }

    fn build_node(
        seeds: &[Nibbles],
        start: usize,
        end: usize,
        depth: usize,
        nodes: &mut Vec<SeedTrieNode>,
    ) -> usize {
        debug_assert!(start < end, "seed trie nodes must cover at least one seed");

        let node_idx = nodes.len();
        nodes.push(SeedTrieNode {
            depth,
            suffix: Box::default(),
            children: Box::default(),
            leaf_seed: None,
        });

        let mut common_len = 0usize;
        while depth + common_len < DIMS
            && seeds[start][depth + common_len] == seeds[end - 1][depth + common_len]
        {
            common_len += 1;
        }

        let suffix = seeds[start][depth..depth + common_len]
            .to_vec()
            .into_boxed_slice();
        let next_dim = depth + common_len;
        if next_dim == DIMS {
            debug_assert_eq!(end, start + 1);
            nodes[node_idx] = SeedTrieNode {
                depth,
                suffix,
                children: Box::default(),
                leaf_seed: Some(start),
            };
            return node_idx;
        }

        let mut children = Vec::new();
        let mut cursor = start;
        while cursor < end {
            let value = seeds[cursor][next_dim];
            let mut child_end = cursor + 1;
            while child_end < end && seeds[child_end][next_dim] == value {
                child_end += 1;
            }
            let child_idx = Self::build_node(seeds, cursor, child_end, next_dim + 1, nodes);
            children.push((value, child_idx));
            cursor = child_end;
        }

        nodes[node_idx] = SeedTrieNode {
            depth,
            suffix,
            children: children.into_boxed_slice(),
            leaf_seed: None,
        };
        node_idx
    }
}
