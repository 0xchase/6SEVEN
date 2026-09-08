const NIBBLE_POSITIONS: usize = 32;
const NIBBLE_VALUES: usize = 16;

pub fn compute_nibble_entropy(samples: &[u128], start_nibble: u8, end_nibble: u8) -> Vec<f64> {
    if start_nibble >= end_nibble {
        return Vec::new();
    }

    let start = start_nibble as usize;
    let end = end_nibble.min(NIBBLE_POSITIONS as u8) as usize;
    if start >= end {
        return Vec::new();
    }

    let width = end - start;
    let mut counts = vec![[0u64; NIBBLE_VALUES]; width];
    for &addr in samples {
        for nibble_idx in start..end {
            let shift = ((NIBBLE_POSITIONS - 1 - nibble_idx) * 4) as u32;
            let value = ((addr >> shift) & 0xFu128) as usize;
            counts[nibble_idx - start][value] = counts[nibble_idx - start][value].saturating_add(1);
        }
    }

    let total = samples.len() as f64;
    counts
        .into_iter()
        .map(|nibble_counts| {
            let mut entropy = 0f64;
            for count in nibble_counts.iter() {
                if *count == 0 {
                    continue;
                }
                let probability = *count as f64 / total;
                entropy -= probability * probability.log2();
            }
            entropy
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::compute_nibble_entropy;

    #[test]
    fn nibble_entropy_is_zero_for_constant_nibble() {
        let samples = vec![0u128; 4];
        let entropy = compute_nibble_entropy(&samples, 0, 1);
        assert_eq!(entropy, vec![0.0]);
    }

    #[test]
    fn nibble_entropy_reaches_one_bit_for_two_balanced_values() {
        let samples = vec![0u128, 1u128 << 124];
        let entropy = compute_nibble_entropy(&samples, 0, 1);
        assert_eq!(entropy.len(), 1);
        assert!((entropy[0] - 1.0).abs() < 1e-9);
    }
}
