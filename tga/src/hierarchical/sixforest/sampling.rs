use super::{Address, SixForestRegion};
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SixForestStream {
    regions: Vec<SixForestRegion>,
    excluded_seeds: HashSet<Address>,
    states: Vec<RegionDrawState>,
    active_regions: VecDeque<usize>,
}

impl SixForestStream {
    pub(super) fn new(
        regions: Vec<SixForestRegion>,
        excluded_seeds: HashSet<Address>,
        generation_seed: u64,
    ) -> Self {
        let mut active_regions = (0..regions.len()).collect::<Vec<_>>();
        shuffle_slice(
            &mut active_regions,
            splitmix64(generation_seed ^ 0x481c_9dc5_101b_9f6d),
        );

        let states = regions
            .iter()
            .enumerate()
            .map(|(idx, region)| {
                RegionDrawState::new(
                    region,
                    splitmix64(generation_seed ^ (idx as u64).wrapping_mul(0xd6e8_feb8_6659_fd93)),
                )
            })
            .collect();

        Self {
            regions,
            excluded_seeds,
            states,
            active_regions: active_regions.into(),
        }
    }
    pub(super) fn exclude_prefix(&mut self, prefix: &crate::Ipv6Prefix) {
        for (region, state) in self.regions.iter().zip(&mut self.states) {
            if prefix.contains(&std::net::Ipv6Addr::from(region.base))
                && region
                    .free_dims
                    .iter()
                    .all(|&dim| dim * 4 >= prefix.prefix_len() as usize)
            {
                state.exhausted = true;
            }
        }
    }

    pub(super) fn sample_region(
        &mut self,
        index: usize,
        draws: &mut RandomDraws,
        rng: &mut SplitMix64,
    ) -> Option<Address> {
        if self.states[index].exhausted {
            return None;
        }
        while let Some(offset) = draws.next(rng) {
            let address = self.regions[index].address_at_offset(offset);
            if self.excluded_seeds.insert(address) {
                return Some(address);
            }
        }
        self.states[index].exhausted = true;
        None
    }

    pub(super) fn has_drawn(&self) -> bool {
        self.states.iter().any(|state| state.attempts != 0)
    }

    pub(super) fn exclude_previous_samples(&mut self) {
        for (region, state) in self.regions.iter().zip(&self.states) {
            if region.free_nibbles() > 3 {
                for attempt in 0..state.attempts {
                    self.excluded_seeds
                        .insert(region.address_at_offset(state.sampler.draw(attempt)));
                }
            }
        }
    }

    pub(super) fn needs_prescan(&self, index: usize) -> bool {
        self.regions[index].free_nibbles() > 3
    }

    pub(super) fn region_order(&self) -> Vec<usize> {
        self.active_regions.iter().copied().collect()
    }

    pub(super) fn draw_region(&mut self, index: usize) -> Option<Address> {
        self.states[index].next_address(&self.regions[index], &self.excluded_seeds)
    }
}

impl Iterator for SixForestStream {
    type Item = Address;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(region_idx) = self.active_regions.pop_front() {
            if let Some(address) = self.draw_region(region_idx) {
                self.active_regions.push_back(region_idx);
                return Some(address);
            }
        }

        None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RegionDrawState {
    sampler: RegionSampler,
    attempts: u128,
    exhausted: bool,
    finite_span: Option<u128>,
}

impl RegionDrawState {
    fn new(region: &SixForestRegion, seed: u64) -> Self {
        Self {
            sampler: RegionSampler::new(region.free_bits(), seed),
            attempts: 0,
            exhausted: false,
            finite_span: region.finite_span(),
        }
    }

    fn next_address(
        &mut self,
        region: &SixForestRegion,
        excluded_seeds: &HashSet<Address>,
    ) -> Option<Address> {
        while !self.exhausted {
            let offset = self.sampler.draw(self.attempts);
            let (next, wrapped) = self.attempts.overflowing_add(1);
            self.attempts = next;
            self.exhausted = wrapped || self.finite_span == Some(next);

            let address = region.address_at_offset(offset);
            if excluded_seeds.contains(&address) {
                continue;
            }

            return Some(address);
        }

        None
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct RegionSampler {
    multiplier: u128,
    offset: u128,
    mask: u128,
}

impl RegionSampler {
    fn new(free_bits: u32, seed: u64) -> Self {
        debug_assert!(free_bits <= 128);
        let mask = domain_mask(free_bits);
        if free_bits == 0 {
            return Self {
                multiplier: 1,
                offset: 0,
                mask,
            };
        }

        Self {
            multiplier: splitmix128(seed ^ 0xa5a5_3c3c_9669_5a5a) | 1,
            offset: splitmix128(seed ^ 0x517c_c1b7_2722_0a95) & mask,
            mask,
        }
    }

    fn draw(&self, attempt: u128) -> u128 {
        attempt
            .wrapping_mul(self.multiplier)
            .wrapping_add(self.offset)
            & self.mask
    }
}

pub(super) fn finite_span_for_bits(bits: u32) -> Option<u128> {
    debug_assert!(bits <= 128);
    if bits == 128 {
        None
    } else {
        Some(1u128 << bits)
    }
}

fn domain_mask(bits: u32) -> u128 {
    debug_assert!(bits <= 128);
    match bits {
        0 => 0,
        128 => u128::MAX,
        bits => (1u128 << bits) - 1,
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn splitmix128(seed: u64) -> u128 {
    ((splitmix64(seed) as u128) << 64) | splitmix64(seed ^ 0x8cb9_2ba7_2f3d_8dd7) as u128
}

fn shuffle_slice<T>(items: &mut [T], seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for idx in (1..items.len()).rev() {
        let swap = rng.index_below(idx + 1);
        items.swap(idx, swap);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(super) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        let value = splitmix64(self.state);
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        value
    }

    fn index_below(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        if upper <= 1 {
            return 0;
        }

        let upper = upper as u64;
        let acceptance_zone = u64::MAX - (u64::MAX % upper);
        loop {
            let value = self.next_u64();
            if value < acceptance_zone {
                return (value % upper) as usize;
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct RandomDraws {
    remaining: Option<u128>,
    swaps: std::collections::BTreeMap<u128, u128>,
}

impl RandomDraws {
    pub(super) fn new(bits: u32) -> Self {
        Self {
            remaining: finite_span_for_bits(bits),
            swaps: Default::default(),
        }
    }

    fn next(&mut self, rng: &mut SplitMix64) -> Option<u128> {
        let remaining = self.remaining;
        if remaining == Some(0) {
            return None;
        }
        let index = loop {
            let random = (u128::from(rng.next_u64()) << 64) | u128::from(rng.next_u64());
            match remaining {
                None => break random,
                Some(limit) if random >= limit.wrapping_neg() % limit => break random % limit,
                _ => {}
            }
        };
        let last = remaining.unwrap_or(0).wrapping_sub(1);
        let value = self.swaps.remove(&index).unwrap_or(index);
        let replacement = self.swaps.remove(&last).unwrap_or(last);
        if index != last {
            self.swaps.insert(index, replacement);
        }
        self.remaining = Some(last);
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_draws_cover_small_domains_once_and_exhaust() {
        for bits in [0, 4, 8, 12] {
            let mut draws = RandomDraws::new(bits);
            let mut rng = SplitMix64::new(42);
            let mut values = Vec::new();
            while let Some(value) = draws.next(&mut rng) {
                values.push(value);
            }
            values.sort_unstable();
            assert_eq!(values, (0..1u128 << bits).collect::<Vec<_>>());
            assert!(draws.next(&mut rng).is_none());
        }
    }

    #[test]
    fn random_draws_support_the_entire_ipv6_domain() {
        let mut draws = RandomDraws::new(128);
        let mut rng = SplitMix64::new(42);
        let values: HashSet<_> = (0..1000).map(|_| draws.next(&mut rng).unwrap()).collect();
        assert_eq!(values.len(), 1000);
        assert_eq!(draws.remaining, Some(u128::MAX - 999));
    }

    #[test]
    fn first_random_draw_has_no_low_nibble_bias() {
        let mut counts = [0; 16];
        for seed in 0..10_000 {
            let mut draws = RandomDraws::new(4);
            counts[draws.next(&mut SplitMix64::new(seed)).unwrap() as usize] += 1;
        }
        assert!(counts.iter().all(|&count| (450..800).contains(&count)));
    }

    #[test]
    fn affine_sampler_is_a_permutation_of_small_domains() {
        for bits in [0, 4, 8, 12] {
            let sampler = RegionSampler::new(bits, 42);
            let mut values: Vec<_> = (0..1u128 << bits).map(|i| sampler.draw(i)).collect();
            values.sort_unstable();
            assert_eq!(values, (0..1u128 << bits).collect::<Vec<_>>());
        }
    }

    #[test]
    fn full_ipv6_domain_stops_after_counter_wrap() {
        let region = SixForestRegion {
            base: [0; 16],
            free_dims: (0..32).collect(),
            seed_count: 0,
        };
        let mut state = RegionDrawState::new(&region, 0);
        state.attempts = u128::MAX;
        assert!(state.next_address(&region, &HashSet::new()).is_some());
        assert!(state.next_address(&region, &HashSet::new()).is_none());
    }
}
