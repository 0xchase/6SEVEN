use super::Pattern;
use crate::Address;
use std::collections::{BTreeMap, HashMap};

pub(super) struct PatternIndex {
    signatures: BTreeMap<u128, HashMap<u128, Vec<usize>>>,
}

impl PatternIndex {
    pub(super) fn new(patterns: &[Pattern]) -> Self {
        let mut signatures = BTreeMap::<u128, HashMap<u128, Vec<usize>>>::new();
        for (id, pattern) in patterns.iter().enumerate() {
            let mask = pattern.fixed_mask();
            signatures
                .entry(mask)
                .or_default()
                .entry(u128::from_be_bytes(pattern.seeds[0]) & mask)
                .or_default()
                .push(id);
        }
        Self { signatures }
    }

    pub(super) fn matches(&self, address: Address) -> impl Iterator<Item = usize> + '_ {
        let value = u128::from_be_bytes(address);
        self.signatures
            .iter()
            .filter_map(move |(&mask, signatures)| signatures.get(&(value & mask)))
            .flat_map(|ids| ids.iter().copied())
    }
}
