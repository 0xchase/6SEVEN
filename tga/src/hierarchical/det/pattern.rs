use crate::Address;
use serde::{Deserialize, Serialize};

pub(super) const WILDCARD: u16 = u16::MAX;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(super) struct TargetPattern {
    pub(super) digits: Vec<u16>,
}

impl TargetPattern {
    // None represents a cardinality of 2^128.
    pub(super) fn size(&self, base: usize) -> Option<u128> {
        self.digits
            .iter()
            .filter(|&&digit| digit == WILDCARD)
            .try_fold(1u128, |size, _| size.checked_mul(base as u128))
    }

    pub(super) fn matches(&self, address: &Address, bits: usize) -> bool {
        let value = u128::from_be_bytes(*address);
        let mask = (1u128 << bits) - 1;
        self.digits.iter().enumerate().all(|(index, &digit)| {
            digit == WILDCARD || ((value >> (128 - (index + 1) * bits)) & mask) == digit as u128
        })
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(super) struct WildcardField {
    pub(super) shift: usize,
    pub(super) mask: u128,
    pub(super) value: usize,
}

#[derive(Clone, Serialize, Deserialize)]
pub(super) struct PatternExpansion {
    pub(super) address: u128,
    pub(super) wildcard_fields: Vec<WildcardField>,
    pub(super) base: usize,
    pub(super) exhausted: bool,
}

impl PatternExpansion {
    pub(super) fn new(pattern: &TargetPattern, base: usize, bits_per_dimension: usize) -> Self {
        let dimension_count = pattern.digits.len();
        let mut address = 0u128;
        let mut wildcard_fields = Vec::new();
        let digit_mask = (1u128 << bits_per_dimension) - 1;

        for (idx, &digit) in pattern.digits.iter().enumerate() {
            let shift = (dimension_count - idx - 1) * bits_per_dimension;
            if digit == WILDCARD {
                wildcard_fields.push(WildcardField {
                    shift,
                    mask: digit_mask << shift,
                    value: 0,
                });
            } else {
                address |= (digit as u128) << shift;
            }
        }

        Self {
            address,
            wildcard_fields,
            base,
            exhausted: false,
        }
    }

    pub(super) fn increment(&mut self) {
        for field in self.wildcard_fields.iter_mut().rev() {
            field.value += 1;

            if field.value == self.base {
                field.value = 0;
                self.address &= !field.mask;
            } else {
                self.address &= !field.mask;
                self.address |= (field.value as u128) << field.shift;
                return;
            }
        }
        self.exhausted = true;
    }
}

impl Iterator for PatternExpansion {
    type Item = Address;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        let address = self.address.to_be_bytes();
        self.increment();

        Some(address)
    }
}
