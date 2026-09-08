use super::{Address, AddressExt, HEX_RADIX, NIBBLE_COUNT};
use rayon::prelude::*;

const PARALLEL_DIM_SCORE_MIN_SEEDS: usize = 4096;

pub(super) fn partition(addresses: &[Address], min_region_size: usize) -> Vec<Vec<Address>> {
    if addresses.is_empty() {
        return Vec::new();
    }
    let mut leaves = Vec::new();
    let mut queue = std::collections::VecDeque::from([(0..addresses.len()).collect::<Vec<_>>()]);
    while let Some(indices) = queue.pop_front() {
        let dimension = if indices.len() < min_region_size {
            None
        } else {
            split_dimension(addresses, &indices)
        };
        if let Some(dimension) = dimension {
            let mut buckets: [Vec<usize>; HEX_RADIX] = std::array::from_fn(|_| Vec::new());
            for index in indices {
                buckets[addresses[index].get_nibble(dimension) as usize].push(index);
            }
            queue.extend(buckets.into_iter().filter(|bucket| !bucket.is_empty()));
        } else {
            leaves.push(indices.into_iter().map(|index| addresses[index]).collect());
        }
    }
    leaves
}

fn split_dimension(addresses: &[Address], indices: &[usize]) -> Option<usize> {
    let score = |dimension| {
        let mut counts = [0usize; HEX_RADIX];
        for &index in indices {
            counts[addresses[index].get_nibble(dimension) as usize] += 1;
        }
        (counts.iter().filter(|&&count| count > 0).count() > 1)
            .then(|| counts.iter().filter(|&&count| count > 1).sum())
    };
    let scores: Vec<Option<usize>> = if indices.len() >= PARALLEL_DIM_SCORE_MIN_SEEDS {
        (0..NIBBLE_COUNT).into_par_iter().map(score).collect()
    } else {
        (0..NIBBLE_COUNT).map(score).collect()
    };
    choose_dimension(&scores)
}

fn choose_dimension(scores: &[Option<usize>]) -> Option<usize> {
    let mut free = scores
        .iter()
        .enumerate()
        .filter_map(|(dim, &score)| score.map(|s| (dim, s)));
    let leftmost = free.next()?;
    let best = free.fold(
        leftmost,
        |best, next| if next.1 > best.1 { next } else { best },
    );
    Some(best.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maximum_coverage_wins_without_a_distance_penalty() {
        assert_eq!(choose_dimension(&[None, Some(4), Some(5)]), Some(2));
        assert_eq!(choose_dimension(&[Some(4), None, Some(7)]), Some(2));
        assert_eq!(choose_dimension(&[Some(7), None, Some(7)]), Some(0));
        assert_eq!(choose_dimension(&[None; 32]), None);
    }

    #[test]
    fn identical_rows_terminate_at_any_threshold() {
        let seeds = vec![[0; 16]; 16];
        for threshold in [1, 16, 17] {
            assert_eq!(partition(&seeds, threshold), vec![seeds.clone()]);
        }
    }

    #[test]
    fn threshold_is_strict_and_children_are_visited_in_nibble_order() {
        let seeds: Vec<_> = (0..16)
            .map(|value| {
                let mut a = [0; 16];
                a[15] = value;
                a
            })
            .collect();
        assert_eq!(partition(&seeds, 17), vec![seeds.clone()]);
        assert_eq!(
            partition(&seeds, 16),
            seeds.iter().map(|&a| vec![a]).collect::<Vec<_>>()
        );
    }
}
