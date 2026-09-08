//! IPv6 prefix coverage, independent of scanning and file formats.
use crate::Ipv6Prefix;
use std::{collections::BTreeMap, net::Ipv6Addr};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PrefixSet {
    ranges: BTreeMap<u128, u128>,
}

impl PrefixSet {
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }
    pub fn contains(&self, address: Ipv6Addr) -> bool {
        let address = u128::from(address);
        self.ranges
            .range(..=address)
            .next_back()
            .is_some_and(|(_, end)| address <= *end)
    }
    pub fn covers(&self, prefix: Ipv6Prefix) -> bool {
        self.ranges
            .range(..=u128::from(prefix.network()))
            .next_back()
            .is_some_and(|(_, end)| u128::from(prefix.broadcast()) <= *end)
    }
    /// Returns whether address coverage changed.
    pub fn insert(&mut self, prefix: Ipv6Prefix) -> bool {
        if self.covers(prefix) {
            return false;
        }
        let mut start = u128::from(prefix.network());
        let mut end = u128::from(prefix.broadcast());
        if let Some((&left, &right)) = self.ranges.range(..=start).next_back() {
            if right.saturating_add(1) >= start {
                start = left;
                end = end.max(right);
                self.ranges.remove(&left);
            }
        }
        while let Some((&left, &right)) = self.ranges.range(start..).next() {
            if left > end.saturating_add(1) {
                break;
            }
            end = end.max(right);
            self.ranges.remove(&left);
        }
        self.ranges.insert(start, end);
        true
    }
    pub fn prefixes(&self) -> impl Iterator<Item = Ipv6Prefix> + '_ {
        self.ranges.iter().flat_map(|(&start, &end)| {
            let mut next = Some(start);
            std::iter::from_fn(move || {
                let start = next?;
                let mut bits = start.trailing_zeros();
                while bits > 0 && (start | (u128::MAX >> (128 - bits))) > end {
                    if bits == 0 {
                        break;
                    }
                    bits -= 1;
                }
                // Avoid shifts by 128, including the singleton case.
                let mask = if bits == 128 {
                    u128::MAX
                } else if bits == 0 {
                    0
                } else {
                    (1u128 << bits) - 1
                };
                let last = start | mask;
                next = if last == end {
                    None
                } else {
                    last.checked_add(1)
                };
                Some(Ipv6Prefix::new(start.into(), (128 - bits) as u8).unwrap())
            })
        })
    }
}
impl FromIterator<Ipv6Prefix> for PrefixSet {
    fn from_iter<T: IntoIterator<Item = Ipv6Prefix>>(iter: T) -> Self {
        let mut result = Self::default();
        for prefix in iter {
            result.insert(prefix);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn merges_and_exports_exact_coverage() {
        let mut set: PrefixSet = ["::8/125", "::10/125", "::a/128", "::/127"]
            .into_iter()
            .map(|s| s.parse().unwrap())
            .collect();
        for n in 0..40u128 {
            assert_eq!(set.contains(n.into()), n < 2 || (8..24).contains(&n));
        }
        let roundtrip: PrefixSet = set.prefixes().collect();
        assert_eq!(set, roundtrip);
        assert!(!set.insert("::8/126".parse().unwrap()));
        assert!(set.insert("::/0".parse().unwrap()));
        assert_eq!(
            set.prefixes().collect::<Vec<_>>(),
            vec!["::/0".parse::<Ipv6Prefix>().unwrap()]
        );
        assert!(set.contains(u128::MAX.into()));
    }
    #[test]
    fn singleton_and_last_address() {
        for n in [0u128, 1, 2, 3, 128, u128::MAX] {
            let prefix = Ipv6Prefix::new(n.into(), 128).unwrap();
            let set: PrefixSet = [prefix].into_iter().collect();
            assert_eq!(set.prefixes().collect::<Vec<_>>(), vec![prefix]);
        }
    }
}
