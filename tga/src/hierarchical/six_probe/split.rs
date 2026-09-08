use super::config::{DhcType, SplitArrayType, SplitOrder};
use super::encoding::{NibbleAddr, PARALLEL_NIBBLE_SCAN_MIN_SEEDS};
use super::python_random::PythonRandom;
use rand::Rng;
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SplitArray {
    positions: Vec<usize>,
}

impl SplitArray {
    /// The released Python indexes split arrays by `node.level`, where the root starts at level 1.
    pub(crate) fn relative_position_for_level(&self, level: usize) -> usize {
        self.positions
            .get(level)
            .copied()
            .expect("6Probe split array must cover every recursive DHC level")
    }
}

pub(crate) fn split_by_dhc_type(seeds: &[NibbleAddr], dhc_type: DhcType) -> Vec<Vec<usize>> {
    match dhc_type {
        DhcType::LeftVdps => leftmost(seeds),
        DhcType::RightVdps => rightmost(seeds, 1),
        DhcType::MinEntropy => min_entropy(seeds),
        DhcType::MaxCover => max_covering(seeds),
    }
}

pub(crate) fn split_by_order_with_index(
    seeds: &[NibbleAddr],
    split_order: SplitOrder,
    index: usize,
) -> Vec<Vec<usize>> {
    match split_order {
        SplitOrder::Left => leftmost_k(seeds, index),
        SplitOrder::Right => rightmost(seeds, index),
    }
}

pub(crate) fn generate_split_arrays(
    dimension: usize,
    number: usize,
    split_array_type: SplitArrayType,
    random_seed: Option<u64>,
) -> Vec<SplitArray> {
    match split_array_type {
        SplitArrayType::Random => generate_random_split_arrays(dimension, number, random_seed),
        SplitArrayType::Sequential => generate_sequential_split_arrays(dimension, number),
    }
}

enum SplitArrayRng {
    Thread(rand::rngs::ThreadRng),
    Python(Box<PythonRandom>),
}

impl SplitArrayRng {
    fn new(random_seed: Option<u64>) -> Self {
        match random_seed {
            Some(seed) => Self::Python(Box::new(PythonRandom::seed_from_u64(seed))),
            None => Self::Thread(rand::thread_rng()),
        }
    }

    fn randint_inclusive(&mut self, lower: usize, upper: usize) -> usize {
        debug_assert!(lower <= upper);
        match self {
            Self::Thread(rng) => rng.gen_range(lower..=upper),
            Self::Python(rng) => rng.randint_inclusive(lower, upper),
        }
    }
}

fn generate_random_split_arrays(
    dimension: usize,
    number: usize,
    random_seed: Option<u64>,
) -> Vec<SplitArray> {
    let mut rng = SplitArrayRng::new(random_seed);
    let mut split_index_array = Vec::with_capacity(number);

    for _ in 0..number {
        let mut arr = Vec::with_capacity(dimension);
        for j in (1..=dimension).rev() {
            arr.push(rng.randint_inclusive(1, j));
        }
        split_index_array.push(SplitArray { positions: arr });
    }

    split_index_array
}

fn generate_sequential_split_arrays(dimension: usize, number: usize) -> Vec<SplitArray> {
    if number == 0 {
        return Vec::new();
    }

    if dimension == 0 {
        return (0..number)
            .map(|_| SplitArray {
                positions: Vec::new(),
            })
            .collect();
    }

    let split_count = dimension.saturating_sub(1);
    if split_count == 0 {
        return (0..number)
            .map(|_| SplitArray { positions: vec![1] })
            .collect();
    }

    let mut all = Vec::with_capacity(number);
    for offset_idx in 0..number {
        let offset = offset_idx % split_count;
        let permutation: Vec<usize> = (offset..split_count).chain(0..offset).collect();
        let mut active_dims: Vec<usize> = (0..split_count).collect();
        let mut positions = Vec::with_capacity(dimension);

        // Slot zero is unused because the release starts at level one.
        positions.push(1);

        for selected in permutation {
            let selected_pos = active_dims
                .iter()
                .position(|candidate| *candidate == selected)
                .expect("sequential split array selected an unavailable dimension");
            positions.push(selected_pos + 1);
            active_dims.remove(selected_pos);
        }

        debug_assert_eq!(positions.len(), dimension);
        all.push(SplitArray { positions });
    }

    all
}

pub(crate) fn max_covering(seeds: &[NibbleAddr]) -> Vec<Vec<usize>> {
    let covering = covering_scores_by_dim(seeds);
    let mut leftmost_index: Option<usize> = None;
    let mut leftmost_covering = -1i64;

    for (dim, &score) in covering.iter().enumerate() {
        if score < 0 {
            continue;
        }

        if leftmost_index.is_none() {
            leftmost_index = Some(dim);
            leftmost_covering = score;
        }
    }

    let mut index = 0usize;
    let mut max_cov = covering[0];
    for (dim, &score) in covering.iter().enumerate().skip(1) {
        if score > max_cov {
            index = dim;
            max_cov = score;
        }
    }

    let chosen = if let Some(left_idx) = leftmost_index {
        if max_cov - leftmost_covering <= index as i64 - left_idx as i64 {
            left_idx
        } else {
            index
        }
    } else {
        31
    };

    group_by_dim(seeds, chosen)
}

fn leftmost(seeds: &[NibbleAddr]) -> Vec<Vec<usize>> {
    leftmost_k(seeds, 1)
}

fn leftmost_k(seeds: &[NibbleAddr], rank: usize) -> Vec<Vec<usize>> {
    split_at_rank(seeds, 0..32, rank)
}

fn rightmost(seeds: &[NibbleAddr], rank: usize) -> Vec<Vec<usize>> {
    split_at_rank(seeds, (0..32).rev(), rank)
}

fn split_at_rank(
    seeds: &[NibbleAddr],
    dimensions: impl Iterator<Item = usize>,
    rank: usize,
) -> Vec<Vec<usize>> {
    let mut remaining = rank.max(1);
    let mut selected = 0;
    for dim in dimensions {
        if unique_count(seeds, dim) > 1 {
            selected = dim;
            remaining -= 1;
            if remaining == 0 {
                break;
            }
        }
    }
    // The release clamps an oversized rank to the last variable dimension.
    group_by_dim(seeds, selected)
}

fn min_entropy(seeds: &[NibbleAddr]) -> Vec<Vec<usize>> {
    // The released MinEntropy minimizes cardinality, not Shannon entropy.
    let dim = (0..32)
        .map(|dim| (dim, unique_count(seeds, dim)))
        .filter(|&(_, count)| count > 1)
        .min_by_key(|&(_, count)| count)
        .map_or(0, |(dim, _)| dim);
    group_by_dim(seeds, dim)
}

fn unique_count(seeds: &[NibbleAddr], dim: usize) -> usize {
    seeds
        .iter()
        .fold(0u16, |seen, seed| seen | (1 << seed[dim]))
        .count_ones() as usize
}

fn covering_scores_by_dim(seeds: &[NibbleAddr]) -> Vec<i64> {
    if seeds.len() >= PARALLEL_NIBBLE_SCAN_MIN_SEEDS {
        (0..32)
            .into_par_iter()
            .map(|dim| covering_score(seeds, dim))
            .collect()
    } else {
        (0..32).map(|dim| covering_score(seeds, dim)).collect()
    }
}

fn covering_score(seeds: &[NibbleAddr], dim: usize) -> i64 {
    let mut counts = [0i64; 16];
    for seed in seeds {
        counts[seed[dim] as usize] += 1;
    }

    if counts.iter().filter(|&&count| count > 0).count() == 1 {
        -1
    } else {
        counts.iter().copied().filter(|&count| count != 1).sum()
    }
}

fn group_by_dim(seeds: &[NibbleAddr], dim: usize) -> Vec<Vec<usize>> {
    let mut groups: [Vec<usize>; 16] = std::array::from_fn(|_| Vec::new());
    for (idx, seed) in seeds.iter().enumerate() {
        groups[seed[dim] as usize].push(idx);
    }

    groups
        .into_iter()
        .filter(|group| !group.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr_with_dims(d0: u8, d2: u8, d3: u8) -> NibbleAddr {
        let mut addr = [0u8; 32];
        addr[0] = d0;
        addr[2] = d2;
        addr[3] = d3;
        addr
    }

    #[test]
    fn split_array_lookup_uses_published_one_based_levels() {
        let split_array = SplitArray {
            positions: vec![9, 2, 1],
        };
        assert_eq!(split_array.relative_position_for_level(1), 2);
        assert_eq!(split_array.relative_position_for_level(2), 1);
    }

    #[test]
    fn sequential_split_arrays_follow_paper_cyclic_construction() {
        let arrays = generate_split_arrays(4, 3, SplitArrayType::Sequential, None);
        let rendered: Vec<Vec<usize>> = arrays.into_iter().map(|array| array.positions).collect();
        assert_eq!(
            rendered,
            vec![vec![1, 1, 1, 1], vec![1, 2, 2, 1], vec![1, 3, 1, 1]]
        );
    }

    #[test]
    fn seeded_random_split_arrays_match_python_reference() {
        let cases = [
            (
                0,
                vec![vec![4, 2, 1, 1], vec![4, 2, 2, 1], vec![3, 3, 1, 1]],
            ),
            (
                7,
                vec![vec![3, 1, 2, 1], vec![1, 3, 1, 1], vec![1, 3, 1, 1]],
            ),
            (
                42,
                vec![vec![1, 1, 2, 1], vec![2, 1, 1, 1], vec![4, 1, 1, 1]],
            ),
        ];

        for (seed, expected) in cases {
            let arrays = generate_split_arrays(4, 3, SplitArrayType::Random, Some(seed));
            let rendered: Vec<Vec<usize>> =
                arrays.into_iter().map(|array| array.positions).collect();
            assert_eq!(rendered, expected, "seed {seed}");
        }
    }

    #[test]
    fn seeded_random_split_arrays_match_python_reference_for_full_forest_schedule() {
        let arrays = generate_split_arrays(31, 40, SplitArrayType::Random, Some(0));
        let mut checksum = 14_695_981_039_346_656_037u64;
        for array in arrays {
            for position in array.positions {
                checksum ^= position as u64;
                checksum = checksum.wrapping_mul(1_099_511_628_211);
            }
        }

        assert_eq!(checksum, 9_426_613_658_120_596_933);
    }

    #[test]
    fn max_covering_prefers_first_maximum_like_numpy_argmax() {
        let seeds = vec![
            addr_with_dims(0, 0, 0),
            addr_with_dims(1, 0, 1),
            addr_with_dims(2, 1, 0),
            addr_with_dims(3, 1, 1),
        ];

        assert_eq!(max_covering(&seeds), vec![vec![0, 1], vec![2, 3]]);
    }
    #[test]
    fn split_strategies_match_released_python() {
        for case in super::super::tests::stage_cases() {
            let seeds = case.nibbles();
            let actual = vec![
                leftmost(&seeds),
                rightmost(&seeds, 1),
                min_entropy(&seeds),
                max_covering(&seeds),
                rightmost(&seeds, 32),
            ];
            assert_eq!(actual, case.splits, "{}", case.name);
        }
    }
    #[test]
    fn sequential_schedules_follow_cyclic_dimensions_from_both_directions() {
        let seeds: Vec<NibbleAddr> = (0..16)
            .map(|value| {
                let mut seed = [0; 32];
                for bit in 0..4 {
                    seed[28 + bit] = (value >> bit) & 1;
                }
                seed
            })
            .collect();
        for order in [SplitOrder::Left, SplitOrder::Right] {
            for (offset, array) in generate_split_arrays(4, 3, SplitArrayType::Sequential, None)
                .iter()
                .enumerate()
            {
                let mut current = seeds.clone();
                for level in 1..4 {
                    let groups = split_by_order_with_index(
                        &current,
                        order,
                        array.relative_position_for_level(level),
                    );
                    let rank = (offset + level - 1) % 3;
                    let dim = match order {
                        SplitOrder::Left => 28 + rank,
                        SplitOrder::Right => 31 - rank,
                    };
                    assert_eq!(groups.len(), 2);
                    for (value, group) in groups.iter().enumerate() {
                        assert!(
                            group
                                .iter()
                                .all(|&index| current[index][dim] == value as u8)
                        );
                    }
                    current = groups[0].iter().map(|&index| current[index]).collect();
                }
            }
        }
    }
}
