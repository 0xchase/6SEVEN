use rand::Rng;
use std::collections::HashMap;

use super::argsort::numpy_argsort;
use super::stats::{dbscan_1d, dbscan_custom_metric, histogram_weighted, percentile};
use super::types::{MiningResult, SegmentState};

fn find_frequency_outliers(counts: &[usize], n: usize, p: f64) -> (Vec<usize>, Vec<usize>) {
    if counts.len() > 10 {
        let mut sorted_counts: Vec<usize> = counts.to_vec();
        sorted_counts.sort_unstable();
        let q1 = percentile(&sorted_counts, 25.0);
        let q3 = percentile(&sorted_counts, 75.0);
        let iqr = q3 - q1;
        let threshold = (0.1 * n as f64).min((q3 + 1.5 * iqr).max(p * n as f64));

        let mut outliers: Vec<usize> = (0..counts.len())
            .filter(|&i| counts[i] as f64 > threshold)
            .collect();
        let mut remaining: Vec<usize> = (0..counts.len())
            .filter(|&i| counts[i] as f64 <= threshold)
            .collect();

        // Too many outliers? Use 10th highest as threshold
        if outliers.len() > 10 {
            let ascending = numpy_argsort(counts);
            let idx_by_count: Vec<usize> = ascending.into_iter().rev().collect();
            let t10 = counts[idx_by_count[9]].max(2);
            outliers = (0..counts.len()).filter(|&i| counts[i] >= t10).collect();
            remaining = (0..counts.len()).filter(|&i| counts[i] < t10).collect();

            // Still too many? Just use top 10
            if outliers.len() > 10 {
                outliers = idx_by_count[0..10].to_vec();
                remaining = idx_by_count[10..].to_vec();
            }
        }
        (outliers, remaining)
    } else {
        // Frequency table very short: take all with count > max(2, 0.001*N)
        let min_count = 2.max((0.001 * n as f64) as usize);
        let outliers: Vec<usize> = (0..counts.len())
            .filter(|&i| counts[i] > min_count)
            .collect();
        let remaining: Vec<usize> = (0..counts.len())
            .filter(|&i| counts[i] <= min_count)
            .collect();
        (outliers, remaining)
    }
}

fn mine_dense_clusters(
    unique: Vec<u128>,
    counts: Vec<usize>,
    l_bits: usize,
    n: usize,
    states: &mut Vec<SegmentState>,
) -> (Vec<u128>, Vec<usize>) {
    if l_bits < 8 {
        return (unique, counts);
    }

    let eps = (l_bits as f64 / 4.0).powi(3);
    let labels = dbscan_1d(&unique, eps, 5);
    let mut label_set: Vec<i64> = labels.to_vec();
    label_set.sort_unstable();
    label_set.dedup();

    let left: usize = counts.iter().sum();
    let mut noise_mask = vec![true; unique.len()]; // true = noise (not claimed)

    for &label in &label_set {
        if label == -1 {
            continue;
        }
        let r_indices: Vec<usize> = (0..unique.len()).filter(|&i| labels[i] == label).collect();
        let r_counts_sum: usize = r_indices.iter().map(|&i| counts[i]).sum();

        // Significance check
        if (r_counts_sum as f64) < 0.001 * n as f64 {
            continue;
        }

        // Density check
        let Some(r_min) = r_indices.iter().map(|&i| unique[i]).min() else {
            continue;
        };
        let Some(r_max) = r_indices.iter().map(|&i| unique[i]).max() else {
            continue;
        };
        let observed = r_counts_sum as f64;
        let range_span = (r_max - r_min) as f64;
        let total_space = (2.0f64).powi(l_bits as i32) - 1.0;
        let expected = range_span / total_space * left as f64;
        let density = if expected > 0.0 {
            observed / expected
        } else {
            f64::INFINITY
        };
        if density < 100.0 {
            continue;
        }

        let r_vals: Vec<u128> = r_indices.iter().map(|&i| unique[i]).collect();
        let r_cnts: Vec<usize> = r_indices.iter().map(|&i| counts[i]).collect();
        emit_as_states(&r_vals, &r_cnts, n, states);

        for &i in &r_indices {
            noise_mask[i] = false;
        }
    }

    // Collect noise (unclaimed) points
    let noise_vals: Vec<u128> = (0..unique.len())
        .filter(|&i| noise_mask[i])
        .map(|i| unique[i])
        .collect();
    let noise_counts: Vec<usize> = (0..unique.len())
        .filter(|&i| noise_mask[i])
        .map(|i| counts[i])
        .collect();
    (noise_vals, noise_counts)
}

fn mine_histogram_clusters(
    unique: &mut Vec<u128>,
    counts: &mut Vec<usize>,
    l_bits: usize,
    n: usize,
    states: &mut Vec<SegmentState>,
) {
    if l_bits < 8 || counts.len() <= 1 {
        return;
    }

    let bincount = if l_bits >= 8 { 256 } else { 1usize << l_bits };
    let (hist, bins) = histogram_weighted(unique, counts, bincount);
    let step = if bins.len() >= 2 {
        bins[1] - bins[0]
    } else {
        1.0
    };

    // Build data matrix: (bin_index, bin_start, normalized_freq) Filter to non-zero bins.
    let mut data: Vec<[f64; 3]> = Vec::new();
    for (i, &h) in hist.iter().enumerate() {
        let freq = h as f64 / n as f64;
        if freq > 0.0 {
            data.push([i as f64, bins[i], freq]);
        }
    }

    if data.len() <= 1 {
        return;
    }

    let labels = dbscan_custom_metric(&data, 5.0, 5);
    let mut label_set: Vec<i64> = labels.to_vec();
    label_set.sort_unstable();
    label_set.dedup();

    let mut cregions: Vec<(f64, f64, f64)> = Vec::new();

    for &label in &label_set {
        if label == -1 {
            continue;
        }
        let r_indices: Vec<usize> = (0..data.len()).filter(|&i| labels[i] == label).collect();

        let r_bins: Vec<f64> = r_indices.iter().map(|&i| data[i][1]).collect();
        let r_freqs: Vec<f64> = r_indices.iter().map(|&i| data[i][2]).collect();

        if r_bins.len() < 5 || r_freqs.iter().sum::<f64>() < 0.1 {
            continue;
        }

        let start = r_bins.iter().cloned().fold(f64::INFINITY, f64::min);
        let stop = r_bins.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + step;
        let avg = r_freqs.iter().sum::<f64>() / r_freqs.len() as f64;
        cregions.push((start, stop, avg));
    }

    // Sort regions by start, then resolve overlaps (matching original)
    cregions.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut i = 0;
    while i + 1 < cregions.len() {
        let nxt_start = cregions[i + 1].0;
        let nxt_stop = cregions[i + 1].1;
        let nxt_avg = cregions[i + 1].2;
        let cur_stop = cregions[i].1;
        let cur_avg = cregions[i].2;

        if nxt_start < cur_stop {
            if nxt_avg > cur_avg {
                if cur_stop > nxt_stop {
                    cregions.insert(i + 2, (nxt_stop, cur_stop, cur_avg));
                }
                cregions[i].1 = nxt_start;
            } else {
                cregions[i + 1].0 = cur_stop;
            }
        }
        i += 1;
    }

    // Convert regions to states
    for &(start, stop, _) in &cregions {
        let r_vals: Vec<u128> = unique
            .iter()
            .copied()
            .filter(|&v| (v as f64) >= start && (v as f64) <= stop)
            .collect();
        let r_cnts: Vec<usize> = unique
            .iter()
            .zip(counts.iter())
            .filter(|(v, _)| (**v as f64) >= start && (**v as f64) <= stop)
            .map(|(_, c)| *c)
            .collect();

        if emit_as_states(&r_vals, &r_cnts, n, states) {
            let before_len = unique.len();
            let mut new_u = Vec::with_capacity(before_len);
            let mut new_c = Vec::with_capacity(before_len);
            for (&v, &c) in unique.iter().zip(counts.iter()) {
                if !((v as f64) >= start && (v as f64) <= stop) {
                    new_u.push(v);
                    new_c.push(c);
                }
            }
            *unique = new_u;
            *counts = new_c;
        }
    }
}

pub(super) fn mine_segment_states<R: Rng + ?Sized>(
    values: &[u128],
    l_bits: usize,
    sample_size: usize,
    rng: &mut R,
) -> MiningResult {
    if values.is_empty() {
        return MiningResult {
            states: Vec::new(),
            min_value: 0,
            max_value: 0,
        };
    }

    let min_value = values.iter().copied().min().unwrap_or(0);
    let max_value = values.iter().copied().max().unwrap_or(0);

    // Match the reference sampling with replacement when reducing the dataset.
    let sampled: Vec<u128>;
    let vals: &[u128] = if values.len() > sample_size {
        sampled = (0..sample_size)
            .map(|_| values[rng.gen_range(0..values.len())])
            .collect();
        &sampled
    } else {
        values
    };

    let n = vals.len();
    let p: f64 = 1.0 / (2.0f64).powi(l_bits as i32);

    // Build frequency table: unique values and their counts
    let mut freq_map: HashMap<u128, usize> = HashMap::new();
    for &v in vals {
        *freq_map.entry(v).or_insert(0) += 1;
    }
    let mut unique: Vec<u128> = freq_map.keys().copied().collect();
    unique.sort_unstable();
    let counts: Vec<usize> = unique.iter().map(|v| freq_map[v]).collect();

    let mut states = Vec::new();

    // Phase 1: Frequency outliers
    let (outlier_idx, remaining_idx) = find_frequency_outliers(&counts, n, p);

    // Emit outliers as Single states (sorted by count descending) Use numpy-compatible argsort on the outlier counts subset.
    let outlier_counts: Vec<usize> = outlier_idx.iter().map(|&i| counts[i]).collect();
    let outlier_asc = numpy_argsort(&outlier_counts);
    let outlier_sorted: Vec<usize> = outlier_asc.iter().rev().map(|&i| outlier_idx[i]).collect();
    for &i in &outlier_sorted {
        if (100.0 * counts[i] as f64 / n as f64) >= 0.005 {
            states.push(SegmentState::Single(unique[i]));
        }
    }

    // Preserve the reference ordering of non-outlier values.
    let remaining_vals: Vec<u128> = remaining_idx.iter().map(|&i| unique[i]).collect();
    let remaining_counts: Vec<usize> = remaining_idx.iter().map(|&i| counts[i]).collect();

    let sum2: usize = remaining_counts.iter().sum();
    if (sum2 as f64) < 0.001 * n as f64 {
        // Nothing significant left
        return MiningResult {
            states,
            min_value,
            max_value,
        };
    } else if remaining_vals.len() < 5 {
        emit_singles(&remaining_vals, &remaining_counts, n, &mut states);
        return MiningResult {
            states,
            min_value,
            max_value,
        };
    }

    // Phase 2: Dense regions via DBSCAN
    let (mut remaining_vals, mut remaining_counts) =
        mine_dense_clusters(remaining_vals, remaining_counts, l_bits, n, &mut states);

    // Phase 3: Histogram-based DBSCAN
    mine_histogram_clusters(
        &mut remaining_vals,
        &mut remaining_counts,
        l_bits,
        n,
        &mut states,
    );

    // Phase 4: Remainder
    emit_as_states(&remaining_vals, &remaining_counts, n, &mut states);

    MiningResult {
        states,
        min_value,
        max_value,
    }
}

fn emit_as_states(
    vals: &[u128],
    counts: &[usize],
    n: usize,
    states: &mut Vec<SegmentState>,
) -> bool {
    if counts.is_empty() {
        return false;
    }
    let mut rv = false;
    let mut remaining_vals = vals.to_vec();
    let mut remaining_counts = counts.to_vec();

    // Extract sub-heavy-hitters if more than 4 values
    if remaining_counts.len() > 4 {
        // Use the reference threshold with its zero interquartile term.
        let mut sorted_c = remaining_counts.clone();
        sorted_c.sort_unstable();
        let q3 = percentile(&sorted_c, 75.0);
        let threshold = (0.1 * n as f64).min(q3.max(0.02 * n as f64));

        let mut hh_pairs: Vec<(u128, usize)> = remaining_vals
            .iter()
            .zip(remaining_counts.iter())
            .filter(|(_, c)| (**c as f64) > threshold)
            .map(|(&v, &c)| (v, c))
            .collect();
        // Sort by count descending using numpy-compatible argsort
        let rpp_counts: Vec<usize> = hh_pairs.iter().map(|&(_, c)| c).collect();
        let rpp_ascending = numpy_argsort(&rpp_counts);
        let sorted_pairs: Vec<(u128, usize)> =
            rpp_ascending.iter().rev().map(|&i| hh_pairs[i]).collect();
        hh_pairs = sorted_pairs;

        for &(v, c) in &hh_pairs {
            if (100.0 * c as f64 / n as f64) >= 0.005 {
                states.push(SegmentState::Single(v));
                rv = true;
            }
        }

        // Remove heavy-hitters from remaining
        let hh_vals: std::collections::HashSet<u128> = hh_pairs.iter().map(|&(v, _)| v).collect();
        let mut new_v = Vec::new();
        let mut new_c = Vec::new();
        for (&v, &c) in remaining_vals.iter().zip(remaining_counts.iter()) {
            if !hh_vals.contains(&v) {
                new_v.push(v);
                new_c.push(c);
            }
        }
        remaining_vals = new_v;
        remaining_counts = new_c;
    }

    let pcnt: f64 = 100.0 * remaining_counts.iter().sum::<usize>() as f64 / n as f64;
    if pcnt < 0.05 {
        return rv;
    }

    if remaining_vals.len() < 5 {
        rv |= emit_singles(&remaining_vals, &remaining_counts, n, states);
    } else {
        // Emit as range
        let Some(rmin) = remaining_vals.iter().copied().min() else {
            return rv;
        };
        let Some(rmax) = remaining_vals.iter().copied().max() else {
            return rv;
        };
        states.push(SegmentState::Range(rmin, rmax));
        rv = true;
    }

    rv
}

// Match descending frequency and NumPy tie ordering from pp in a2-mining.py.
fn emit_singles(vals: &[u128], counts: &[usize], n: usize, states: &mut Vec<SegmentState>) -> bool {
    let before = states.len();
    for i in numpy_argsort(counts).into_iter().rev() {
        if 100.0 * counts[i] as f64 / n as f64 >= 0.005 {
            states.push(SegmentState::Single(vals[i]));
        }
    }
    states.len() != before
}
