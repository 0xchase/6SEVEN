use super::DigitLayout;
use crate::{Address, Ipv6Prefix};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct Region {
    value: u128,
    fixed: u128,
}

impl Region {
    pub(super) fn singleton(address: Address) -> Self {
        Self {
            value: u128::from_be_bytes(address),
            fixed: u128::MAX,
        }
    }

    pub(super) fn prefix(prefix: Ipv6Prefix) -> Self {
        let fixed = u128::MAX
            .checked_shl(u32::from(128 - prefix.prefix_len()))
            .unwrap_or(0);
        Self {
            value: u128::from(prefix.network()) & fixed,
            fixed,
        }
    }

    pub(super) fn contains(self, address: Address) -> bool {
        u128::from_be_bytes(address) & self.fixed == self.value
    }

    pub(super) fn expand(&mut self, dimension: usize, layout: DigitLayout) {
        self.fixed &= !layout.mask(dimension);
        self.value &= self.fixed;
    }

    pub(super) fn wildcard_dimensions(self, layout: DigitLayout) -> u32 {
        layout.wildcard_dimensions(self.fixed.count_zeros())
    }

    pub(super) fn covering_prefix(self) -> Ipv6Prefix {
        let length = self.fixed.leading_ones() as u8;
        Ipv6Prefix::new(self.value.into(), length)
            .expect("valid prefix length")
            .trunc()
    }

    pub(super) fn probe(
        self,
        dimension: usize,
        value: u8,
        random: u128,
        layout: DigitLayout,
    ) -> Address {
        let shift = layout.shift(dimension);
        let address = self.value | (random & !self.fixed);
        ((address & !layout.mask(dimension)) | (u128::from(value) << shift)).to_be_bytes()
    }

    fn intersects(self, other: Self) -> bool {
        (self.value ^ other.value) & self.fixed & other.fixed == 0
    }

    fn subtract(self, other: Self, output: &mut Vec<Self>) {
        if !self.intersects(other) {
            output.push(self);
            return;
        }
        let mut remainder = self;
        let mut split = other.fixed & !self.fixed;
        while split != 0 {
            let bit = 1u128 << (127 - split.leading_zeros());
            split &= !bit;
            remainder.fixed |= bit;
            remainder.value = (remainder.value & !bit) | (other.value & bit);
            output.push(Self {
                value: remainder.value ^ bit,
                fixed: remainder.fixed,
            });
        }
    }

    fn size_capped(self) -> usize {
        1usize
            .checked_shl(self.fixed.count_zeros())
            .unwrap_or(usize::MAX)
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub(super) struct RegionSet(Vec<Region>);

impl RegionSet {
    pub(super) fn from_regions(regions: impl IntoIterator<Item = Region>) -> Self {
        let mut regions: Vec<_> = regions.into_iter().collect();
        regions.sort_unstable_by_key(|region| (region.fixed, region.value));
        regions.dedup();
        let mut set = Self::default();
        let mut start = 0;
        while start < regions.len() {
            let mut end = start + 1;
            while end < regions.len() && regions[end].fixed == regions[start].fixed {
                end += 1;
            }
            // Distinct cubes with the same fixed bits are disjoint.
            let mut group = Self(regions[start..end].to_vec());
            group.subtract(&set);
            set.0.extend(group.0);
            start = end;
        }
        set
    }

    pub(super) fn regions(&self) -> impl Iterator<Item = Region> + '_ {
        self.0.iter().copied()
    }

    pub(super) fn insert(&mut self, region: Region) {
        let mut additions = Self(vec![region]);
        additions.subtract(self);
        self.0.extend(additions.0);
    }

    pub(super) fn union(&mut self, other: &Self) {
        *self = Self::from_regions(self.regions().chain(other.regions()));
    }

    pub(super) fn subtract(&mut self, other: &Self) {
        for &excluded in &other.0 {
            self.exclude(excluded);
        }
    }

    fn exclude(&mut self, excluded: Region) {
        let mut remaining = Vec::new();
        for region in self.0.drain(..) {
            region.subtract(excluded, &mut remaining);
        }
        self.0 = remaining;
    }

    pub(super) fn contains(&self, address: Address) -> bool {
        self.0.iter().any(|region| region.contains(address))
    }

    pub(super) fn expand(&mut self, dimension: usize, layout: DigitLayout) {
        *self = Self::from_regions(self.0.iter().map(|&region| {
            let mut expanded = region;
            expanded.expand(dimension, layout);
            expanded
        }));
    }

    pub(super) fn size_capped(&self) -> usize {
        self.0.iter().fold(0usize, |total, region| {
            total.saturating_add(region.size_capped())
        })
    }

    pub(super) fn cursor(self) -> RegionCursor {
        RegionCursor {
            pending: self.0.into_iter().rev().collect(),
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub(super) struct RegionCursor {
    pending: Vec<Region>,
}

impl RegionCursor {
    pub(super) fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    pub(super) fn exclude(&mut self, region: Region) {
        let mut remaining = Vec::new();
        for pending in self.pending.drain(..) {
            pending.subtract(region, &mut remaining);
        }
        self.pending = remaining;
    }
}

impl Iterator for RegionCursor {
    type Item = Address;

    fn next(&mut self) -> Option<Self::Item> {
        let mut region = self.pending.pop()?;
        while region.fixed != u128::MAX {
            let bit = 1u128 << (127 - (!region.fixed).leading_zeros());
            region.fixed |= bit;
            self.pending.push(Region {
                value: region.value | bit,
                fixed: region.fixed,
            });
        }
        Some(region.value.to_be_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn symbolic_union_and_subtraction_match_exhaustive_sets() {
        for fixed_a in [0, 1, 3, 15, 51, 85, 170, 240, 255] {
            for fixed_b in [0, 1, 3, 15, 51, 85, 170, 240, 255] {
                for value in [0, 1, 5, 17, 85, 170, 255] {
                    let a = Region {
                        value: value & fixed_a,
                        fixed: (u128::MAX ^ 255) | fixed_a,
                    };
                    let b = Region {
                        value: value & fixed_b,
                        fixed: (u128::MAX ^ 255) | fixed_b,
                    };
                    let enumerate = |region: Region| {
                        (0u128..256)
                            .filter(move |&n| region.contains(n.to_be_bytes()))
                            .map(u128::to_be_bytes)
                            .collect::<HashSet<_>>()
                    };
                    let left = enumerate(a);
                    let right = enumerate(b);
                    let mut difference = RegionSet::from_regions([a]);
                    difference.subtract(&RegionSet::from_regions([b]));
                    let actual: Vec<_> = difference.clone().cursor().collect();
                    assert_eq!(actual.len(), difference.size_capped());
                    assert_eq!(
                        actual.into_iter().collect::<HashSet<_>>(),
                        left.difference(&right).copied().collect()
                    );
                    let union = RegionSet::from_regions([a, b]);
                    let actual: Vec<_> = union.clone().cursor().collect();
                    assert_eq!(actual.len(), union.size_capped());
                    assert_eq!(
                        actual.into_iter().collect::<HashSet<_>>(),
                        left.union(&right).copied().collect()
                    );
                }
            }
        }
    }

    #[test]
    fn prefixes_support_every_bit_length() {
        for length in 0..=128 {
            let prefix: Ipv6Prefix = format!("2001:db8:abcd:1234:5678:9abc:def0:1234/{length}")
                .parse()
                .unwrap();
            let region = Region::prefix(prefix);
            for value in [
                0,
                u128::MAX,
                u128::from(prefix.network()),
                u128::from(prefix.network()) ^ 1,
                u128::from(prefix.network()) ^ (1u128 << 100),
            ] {
                assert_eq!(
                    region.contains(value.to_be_bytes()),
                    prefix.contains(&std::net::Ipv6Addr::from(value))
                );
            }
        }
    }

    #[test]
    fn full_space_cursor_reaches_addresses_above_machine_word_limits() {
        let mut all = RegionSet::from_regions([Region::prefix("::/0".parse().unwrap())]);
        all.subtract(&RegionSet::from_regions([Region::prefix(
            "::/1".parse().unwrap(),
        )]));
        assert_eq!(all.size_capped(), usize::MAX);
        let mut cursor = all.cursor();
        assert_eq!(cursor.next(), Some((1u128 << 127).to_be_bytes()));
        cursor.exclude(Region::prefix("8000::/2".parse().unwrap()));
        assert_eq!(cursor.next(), Some((3u128 << 126).to_be_bytes()));
        let mut end = RegionSet::from_regions([Region::prefix(
            "ffff:ffff:ffff:ffff:ffff:ffff:ffff:fffe/127"
                .parse()
                .unwrap(),
        )])
        .cursor();
        assert_eq!(end.next(), Some((u128::MAX - 1).to_be_bytes()));
        assert_eq!(end.next(), Some(u128::MAX.to_be_bytes()));
        assert_eq!(end.next(), None);
    }
}
