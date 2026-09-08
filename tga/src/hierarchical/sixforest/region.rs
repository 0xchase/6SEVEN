#[cfg(test)]
use super::nibble_to_hex;
use super::sampling::finite_span_for_bits;
use super::{Address, AddressExt, NIBBLE_COUNT, SixForestStream};
use crate::pattern::Region;
use serde::{Deserialize, Serialize};
#[cfg(test)]
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "StoredRegion")]
pub(super) struct SixForestRegion {
    /// Canonical base address with every free nibble cleared to zero.
    pub(super) base: Address,
    /// Free nibble indexes in IPv6 text order.
    pub(super) free_dims: Vec<usize>,
    /// Number of normal seeds mined into this region.
    pub(super) seed_count: usize,
}

impl SixForestRegion {
    pub(super) fn from_addresses(addresses: &[Address]) -> Self {
        let mut pattern = Region::create_pattern_from_addresses(addresses);
        pattern.expand_to_nibble_boundaries();

        Self {
            base: pattern.address,
            free_dims: wildcard_nibble_dims(&pattern),
            seed_count: addresses.len(),
        }
    }

    pub(super) fn free_nibbles(&self) -> usize {
        self.free_dims.len()
    }

    pub(super) fn free_bits(&self) -> u32 {
        (self.free_dims.len() * 4) as u32
    }

    pub(super) fn finite_span(&self) -> Option<u128> {
        finite_span_for_bits(self.free_bits())
    }

    pub(super) fn address_at_offset(&self, offset: u128) -> Address {
        let mut address = self.base;

        for (slot, dim) in self.free_dims.iter().enumerate() {
            let value = ((offset >> (slot * 4)) & 0x0f) as u8;
            address.set_nibble(*dim, value);
        }

        address
    }

    #[cfg(test)]
    pub(super) fn finite_target_count(&self, excluded_seeds: &HashSet<Address>) -> Option<u128> {
        let span = self.finite_span()?;
        let excluded_in_region = (0..span)
            .map(|offset| self.address_at_offset(offset))
            .filter(|address| excluded_seeds.contains(address))
            .count() as u128;

        Some(span.saturating_sub(excluded_in_region))
    }

    #[cfg(test)]
    pub(super) fn pattern_string(&self) -> String {
        let mut out = String::with_capacity(NIBBLE_COUNT);
        for dim in 0..NIBBLE_COUNT {
            if self.free_dims.contains(&dim) {
                out.push('*');
            } else {
                out.push(nibble_to_hex(self.base.get_nibble(dim)));
            }
        }
        out
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(super) struct SixForestTargetSet {
    pub(super) regions: Vec<SixForestRegion>,
    /// Exclude training seeds across all regions.
    #[serde(default)]
    pub(super) excluded_seeds: Vec<Address>,
    #[serde(default)]
    pub(super) generation_seed: u64,
}

impl SixForestTargetSet {
    pub(super) fn new(
        regions: Vec<SixForestRegion>,
        excluded_seeds: Vec<Address>,
        generation_seed: u64,
    ) -> Self {
        let mut excluded_seeds = excluded_seeds;
        excluded_seeds.sort_unstable();
        excluded_seeds.dedup();

        Self {
            regions,
            excluded_seeds,
            generation_seed,
        }
    }

    pub(super) fn stream(&self) -> SixForestStream {
        SixForestStream::new(
            self.regions.clone(),
            self.excluded_seeds.iter().copied().collect(),
            self.generation_seed,
        )
    }
}

fn wildcard_nibble_dims(pattern: &crate::pattern::AddressPattern) -> Vec<usize> {
    (0..NIBBLE_COUNT)
        .filter(|&dim| {
            let shift = (dim / 2) * 8 + if dim % 2 == 0 { 4 } else { 0 };
            ((pattern.mask >> shift) & 0x0f) == 0
        })
        .collect()
}

// Validate at the serialization boundary before any nibble indexing or shifts.
#[derive(Deserialize)]
struct StoredRegion {
    base: Address,
    free_dims: Vec<usize>,
    seed_count: usize,
}

impl TryFrom<StoredRegion> for SixForestRegion {
    type Error = String;

    fn try_from(value: StoredRegion) -> Result<Self, Self::Error> {
        if value.free_dims.iter().any(|&dim| dim >= NIBBLE_COUNT)
            || value.free_dims.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err("6Forest free dimensions must be strictly increasing and below 32".into());
        }
        if value
            .free_dims
            .iter()
            .any(|&dim| value.base.get_nibble(dim) != 0)
        {
            return Err("6Forest free dimensions must be zero in the base address".into());
        }
        Ok(Self {
            base: value.base,
            free_dims: value.free_dims,
            seed_count: value.seed_count,
        })
    }
}
