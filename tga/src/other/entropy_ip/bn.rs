use std::collections::{BTreeMap, HashMap};

use super::types::CptEntry;

#[cfg(test)]
pub(super) fn visit_combinations(items: &[usize], k: usize, mut visit: impl FnMut(&[usize])) {
    if items.len() < k {
        return;
    }

    fn helper(
        items: &[usize],
        k: usize,
        start: usize,
        current: &mut Vec<usize>,
        visit: &mut impl FnMut(&[usize]),
    ) {
        if current.len() == k {
            visit(current);
            return;
        }
        for i in start..=items.len() - (k - current.len()) {
            current.push(items[i]);
            helper(items, k, i + 1, current, visit);
            current.pop();
        }
    }

    let mut current = Vec::with_capacity(k);
    helper(items, k, 0, &mut current, &mut visit);
}

pub(super) fn calculate_bde(
    data: &[Vec<usize>],
    node_idx: usize,
    parents: &[usize],
    state_counts: &[usize],
) -> f64 {
    let r_i = state_counts[node_idx];
    if r_i == 0 || data.is_empty() {
        return 0.0;
    }

    // Negate the BNFinder BDE objective for maximization.
    let counts = count_states(data, node_idx, parents);

    let hp = r_i.max(2);
    let mut data_score = 0.0_f64;
    for sc in counts.values() {
        let parent_count = sc.values().sum::<usize>();
        for offset in 0..parent_count {
            data_score += ((hp + offset) as f64).log2();
        }
        for &n_jk in sc.values() {
            for offset in 0..n_jk {
                data_score -= ((offset + 1) as f64).log2();
            }
        }
    }

    let graph_score = graph_penalty(parents, state_counts, data.len());

    -(graph_score + data_score)
}

pub(super) fn calculate_cpt(
    node_idx: usize,
    parents: &[usize],
    data: &[Vec<usize>],
    state_counts: &[usize],
) -> HashMap<Vec<usize>, CptEntry> {
    let observed_states = state_counts[node_idx];
    if observed_states == 0 {
        return HashMap::new();
    }
    let smoothed_states = observed_states.max(2);

    let counts = count_states(data, node_idx, parents);

    // Smooth with a unit Dirichlet pseudocount per child state.
    let mut result = HashMap::with_capacity(counts.len());
    for (parent_config, observed_counts) in counts {
        let total: usize = observed_counts.values().sum();
        let denom = total as f64 + smoothed_states as f64;

        let mut explicit_probs = Vec::new();
        for (&state, count) in &observed_counts {
            if *count > 0 {
                explicit_probs.push((state, (*count as f64 + 1.0) / denom));
            }
        }
        result.insert(
            parent_config,
            CptEntry {
                explicit_probs,
                default_prob: 1.0 / denom,
            },
        );
    }

    result
}

// The nonnegative data term makes the graph penalty a lower bound on total cost.
pub(super) fn graph_penalty(parents: &[usize], state_counts: &[usize], rows: usize) -> f64 {
    parents
        .iter()
        .map(|&p| (state_counts[p] as f64).max(1.5).log2())
        .sum::<f64>()
        * (rows as f64 + 1.0).log2()
}

fn count_states(
    data: &[Vec<usize>],
    node: usize,
    parents: &[usize],
) -> BTreeMap<Vec<usize>, BTreeMap<usize, usize>> {
    let mut counts: BTreeMap<Vec<usize>, BTreeMap<usize, usize>> = BTreeMap::new();
    for row in data {
        let key = parents.iter().map(|&parent| row[parent]).collect();
        *counts.entry(key).or_default().entry(row[node]).or_default() += 1;
    }
    counts
}
