use super::{Address, AddressExt, HEX_RADIX, NIBBLE_COUNT};

/// Filter isolation weights using the released three-sigma rule.
pub(super) fn detect_outliers(addresses: &[Address]) -> FilteredRegion {
    let weights = calculate_isolation_weights(addresses);
    let outlier_indices = outlier_indices(&weights);
    let mut is_outlier = vec![false; addresses.len()];
    for &idx in &outlier_indices {
        is_outlier[idx] = true;
    }

    let mut normal = Vec::new();
    for (idx, addr) in addresses.iter().enumerate() {
        if !is_outlier[idx] {
            normal.push(*addr);
        }
    }
    FilteredRegion {
        normal,
        outlier_count: outlier_indices.len(),
    }
}

pub(super) struct FilteredRegion {
    pub(super) normal: Vec<Address>,
    pub(super) outlier_count: usize,
}

fn calculate_isolation_weights(addresses: &[Address]) -> Vec<f64> {
    let mut weights = vec![0.0; addresses.len()];

    for dim in 0..NIBBLE_COUNT {
        let mut counts = [0usize; HEX_RADIX];
        let mut first_index = [0usize; HEX_RADIX];
        for (idx, addr) in addresses.iter().enumerate() {
            let value = addr.get_nibble(dim) as usize;
            if counts[value] == 0 {
                first_index[value] = idx;
            }
            counts[value] += 1;
        }

        let non_empty_bucket_count = counts.iter().filter(|&&count| count > 0).count();
        let singleton_bucket_count = counts.iter().filter(|&&count| count == 1).count();
        if non_empty_bucket_count <= 1 || singleton_bucket_count == 0 {
            continue;
        }

        let increment = 1.0 / singleton_bucket_count as f64;
        for value in 0..HEX_RADIX {
            if counts[value] == 1 {
                weights[first_index[value]] += increment;
            }
        }
    }

    weights
}

/// Track row identities so tied outliers are each removed once.
pub(super) fn outlier_indices(weights: &[f64]) -> Vec<usize> {
    let mut remaining: Vec<_> = weights.iter().copied().enumerate().collect();
    let mut result = Vec::new();

    while remaining.len() > 2 {
        let mut max_idx = 0usize;
        let mut max_weight = remaining[0].1;
        for (idx, &(_, weight)) in remaining.iter().enumerate().skip(1) {
            if weight > max_weight {
                max_weight = weight;
                max_idx = idx;
            }
        }

        let (original_index, _) = remaining.remove(max_idx);

        let count = remaining.len() as f64;
        let mean = remaining.iter().map(|(_, weight)| weight).sum::<f64>() / count;
        let variance = remaining
            .iter()
            .map(|(_, value)| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / count;
        let stddev = variance.sqrt();

        if max_weight - mean > 3.0 * stddev {
            result.push(original_index);
        } else {
            break;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_singleton_bucket_receives_one_over_the_singleton_count() {
        let mut addresses = [[0; 16]; 3];
        addresses[0][15] = 0x10;
        addresses[1][15] = 0x11;
        addresses[2][15] = 0x22;
        let weights = calculate_isolation_weights(&addresses);
        assert_eq!(weights, vec![1.0 / 3.0, 1.0 / 3.0, 1.0 + 1.0 / 3.0]);
    }

    #[test]
    fn equal_weight_pairs_are_retained_regardless_of_distance() {
        let addresses = [[0; 16], [255; 16]];
        let filtered = detect_outliers(&addresses);
        assert_eq!(filtered.normal, addresses);
        assert_eq!(filtered.outlier_count, 0);
    }
}
