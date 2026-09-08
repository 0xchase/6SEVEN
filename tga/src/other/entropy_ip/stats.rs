use std::collections::HashSet;

pub(super) fn percentile(sorted_vals: &[usize], pct: f64) -> f64 {
    if sorted_vals.is_empty() {
        return 0.0;
    }
    if sorted_vals.len() == 1 {
        return sorted_vals[0] as f64;
    }
    let rank = pct / 100.0 * (sorted_vals.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let frac = rank - lo as f64;
    sorted_vals[lo] as f64 * (1.0 - frac) + sorted_vals[hi] as f64 * frac
}

pub(super) fn histogram_weighted(
    values: &[u128],
    weights: &[usize],
    bincount: usize,
) -> (Vec<usize>, Vec<f64>) {
    if values.is_empty() || bincount == 0 {
        return (vec![], vec![]);
    }
    let vmin = values.iter().copied().min().unwrap_or(0) as f64;
    let vmax = values.iter().copied().max().unwrap_or(0) as f64;
    let range = if vmax > vmin { vmax - vmin } else { 1.0 };
    let step = range / bincount as f64;

    let mut bins = Vec::with_capacity(bincount + 1);
    for i in 0..=bincount {
        bins.push(vmin + i as f64 * step);
    }

    let mut hist = vec![0usize; bincount];
    for (&v, &w) in values.iter().zip(weights.iter()) {
        let idx = ((v as f64 - vmin) / step).floor() as usize;
        let idx = idx.min(bincount - 1);
        hist[idx] += w;
    }

    (hist, bins)
}

pub(super) fn dbscan_1d(values: &[u128], eps: f64, min_samples: usize) -> Vec<i64> {
    if values.is_empty() {
        return Vec::new();
    }
    // Sorted core components avoid quadratic neighborhoods while preserving border precedence.
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_unstable_by_key(|&i| (values[i], i));
    let mut core = vec![false; values.len()];
    let (mut left, mut right) = (0, 0);
    for (pos, &index) in order.iter().enumerate() {
        while (values[index] - values[order[left]]) as f64 > eps {
            left += 1;
        }
        right = right.max(pos);
        while right < order.len() && (values[order[right]] - values[index]) as f64 <= eps {
            right += 1;
        }
        core[pos] = right - left >= min_samples;
    }

    let mut components = vec![usize::MAX; values.len()];
    let mut first_indices = Vec::new();
    let mut previous_core = None;
    for (pos, &index) in order.iter().enumerate().filter(|(pos, _)| core[*pos]) {
        let connected = previous_core
            .is_some_and(|previous: usize| (values[index] - values[order[previous]]) as f64 <= eps);
        if !connected {
            first_indices.push(index);
        }
        let component = first_indices.len() - 1;
        first_indices[component] = first_indices[component].min(index);
        components[pos] = component;
        previous_core = Some(pos);
    }
    let mut component_order: Vec<usize> = (0..first_indices.len()).collect();
    component_order.sort_unstable_by_key(|&i| first_indices[i]);
    let mut labels_by_component = vec![0; first_indices.len()];
    for (label, component) in component_order.into_iter().enumerate() {
        labels_by_component[component] = label as i64;
    }

    let mut labels = vec![-1; values.len()];
    let mut nearest = None;
    for pos in 0..order.len() {
        if core[pos] {
            nearest = Some(pos);
        }
        if let Some(other) = nearest
            && (values[order[pos]] - values[order[other]]) as f64 <= eps
        {
            labels[order[pos]] = labels_by_component[components[other]];
        }
    }
    nearest = None;
    for pos in (0..order.len()).rev() {
        if core[pos] {
            nearest = Some(pos);
        }
        if let Some(other) = nearest
            && (values[order[other]] - values[order[pos]]) as f64 <= eps
        {
            let label = labels_by_component[components[other]];
            let current = &mut labels[order[pos]];
            if *current == -1 || label < *current {
                *current = label;
            }
        }
    }
    labels
}

pub(super) fn dbscan_custom_metric(data: &[[f64; 3]], eps: f64, min_samples: usize) -> Vec<i64> {
    dbscan(
        data.len(),
        |i, j| custom_metric(&data[i], &data[j]),
        eps,
        min_samples,
    )
}

fn custom_metric(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
    let bdiff = (p2[0] - p1[0]).abs();
    let pdiff = (p2[2].log(13.0) - p1[2].log(13.0)).abs();
    bdiff * 0.25 + pdiff * 50.0
}

fn dbscan(
    n: usize,
    distance: impl Fn(usize, usize) -> f64,
    eps: f64,
    min_samples: usize,
) -> Vec<i64> {
    let mut labels = vec![-1i64; n];
    let mut visited = vec![false; n];
    let mut cluster_id: i64 = 0;

    for i in 0..n {
        if visited[i] {
            continue;
        }
        visited[i] = true;

        let neighbors: Vec<usize> = (0..n).filter(|&j| distance(i, j) <= eps).collect();

        if neighbors.len() < min_samples {
            continue;
        }

        labels[i] = cluster_id;
        let mut seed_set: Vec<usize> = neighbors.into_iter().filter(|&j| j != i).collect();
        let mut in_seed_set: HashSet<usize> = seed_set.iter().copied().collect();
        in_seed_set.insert(i);
        let mut k = 0;
        while k < seed_set.len() {
            let j = seed_set[k];
            if !visited[j] {
                visited[j] = true;
                let j_neighbors: Vec<usize> = (0..n).filter(|&m| distance(j, m) <= eps).collect();
                if j_neighbors.len() >= min_samples {
                    for &m in &j_neighbors {
                        if !in_seed_set.contains(&m) {
                            seed_set.push(m);
                            in_seed_set.insert(m);
                        }
                    }
                }
            }
            if labels[j] == -1 {
                labels[j] = cluster_id;
            }
            k += 1;
        }
        cluster_id += 1;
    }

    labels
}

#[cfg(test)]
mod tests {
    use super::{dbscan_1d, dbscan_custom_metric};

    #[test]
    fn dbscan_1d_matches_sklearn_for_two_clusters_and_noise() {
        let values = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 30];
        let labels = dbscan_1d(&values, 2.0, 5);
        assert_eq!(labels, vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1]);
    }

    #[test]
    fn dbscan_1d_expands_density_reachable_points_like_sklearn() {
        let values = [0, 1, 2, 3, 4, 5, 6, 10];
        let labels = dbscan_1d(&values, 2.0, 5);
        assert_eq!(labels, vec![0, 0, 0, 0, 0, 0, 0, -1]);
    }

    #[test]
    fn custom_metric_dbscan_matches_entropy_ip_histogram_clustering() {
        let data = [
            [0.0, 0.0, 0.10],
            [1.0, 1.0, 0.11],
            [2.0, 2.0, 0.10],
            [3.0, 3.0, 0.09],
            [4.0, 4.0, 0.10],
            [30.0, 30.0, 0.001],
        ];
        let labels = dbscan_custom_metric(&data, 5.0, 5);
        assert_eq!(labels, vec![0, 0, 0, 0, 0, -1]);
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    #[test]
    fn sorted_dbscan_matches_exhaustive_neighborhoods() {
        let mut rng = StdRng::seed_from_u64(91);
        for _ in 0..500 {
            let values: Vec<u128> = (0..rng.gen_range(1..100))
                .map(|_| rng.gen_range(0..200))
                .collect();
            let eps = rng.gen_range(0..20) as f64;
            let min_samples = rng.gen_range(1..10);
            let expected = dbscan(
                values.len(),
                |i, j| values[i].abs_diff(values[j]) as f64,
                eps,
                min_samples,
            );
            assert_eq!(dbscan_1d(&values, eps, min_samples), expected);
        }
    }

    #[test]
    fn dense_dbscan_handles_full_mining_sample() {
        let values: Vec<u128> = (0..50_000).map(|i| u128::MAX - i).collect();
        assert!(
            dbscan_1d(&values, 32_768.0, 5)
                .iter()
                .all(|&label| label == 0)
        );
    }
}
