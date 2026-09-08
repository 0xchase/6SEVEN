use ipnet::Ipv6Net;
use serde::{Deserialize, Serialize};

use crate::{Address, AddressExt};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NibbleSet(u16);

impl NibbleSet {
    pub const EMPTY: Self = Self(0);
    pub const FULL: Self = Self(u16::MAX);

    pub fn empty() -> Self {
        Self::EMPTY
    }

    pub fn full() -> Self {
        Self::FULL
    }

    pub fn single(value: u8) -> Self {
        Self(1u16 << value)
    }

    pub fn matching(value: u8, mask: u8) -> Self {
        let value = value & 0x0f;
        let mask = mask & 0x0f;
        let mut bits = 0u16;
        for candidate in 0..16u8 {
            if (candidate & mask) == (value & mask) {
                bits |= 1u16 << candidate;
            }
        }
        Self(bits)
    }

    pub fn contains(self, value: u8) -> bool {
        value < 16 && (self.0 & (1u16 << value)) != 0
    }

    pub fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn intersect(self, other: Self) -> Option<Self> {
        let bits = self.0 & other.0;
        if bits == 0 { None } else { Some(Self(bits)) }
    }

    fn nth(self, index: usize) -> Option<u8> {
        let mut seen = 0usize;
        for value in 0..16u8 {
            if self.contains(value) {
                if seen == index {
                    return Some(value);
                }
                seen += 1;
            }
        }
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Subspace {
    pub(crate) dims: [NibbleSet; 32],
}

impl Subspace {
    pub fn empty() -> Self {
        Self {
            dims: [NibbleSet::EMPTY; 32],
        }
    }

    pub fn full() -> Self {
        Self {
            dims: [NibbleSet::FULL; 32],
        }
    }

    pub fn from_addr(address: Address) -> Self {
        let nibbles = address.to_nibbles();
        Self {
            dims: std::array::from_fn(|idx| NibbleSet::single(nibbles[idx])),
        }
    }

    pub fn from_prefix(prefix: Ipv6Net) -> Self {
        let mut dims = [NibbleSet::FULL; 32];
        let nibbles = prefix.network().octets().to_nibbles();
        let full_nibbles = (prefix.prefix_len() / 4) as usize;
        let partial_bits = prefix.prefix_len() % 4;

        for idx in 0..full_nibbles.min(32) {
            dims[idx] = NibbleSet::single(nibbles[idx]);
        }

        if partial_bits > 0 && full_nibbles < 32 {
            let nibble_mask = (!((1u8 << (4 - partial_bits)) - 1)) & 0x0f;
            dims[full_nibbles] = NibbleSet::matching(nibbles[full_nibbles], nibble_mask);
        }

        Self { dims }
    }

    pub fn contains(&self, address: &Address) -> bool {
        let nibbles = address.to_nibbles();
        self.dims
            .iter()
            .zip(nibbles)
            .all(|(allowed, value)| allowed.contains(value))
    }

    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let mut dims = [NibbleSet::EMPTY; 32];
        for (idx, slot) in dims.iter_mut().enumerate() {
            *slot = self.dims[idx].intersect(other.dims[idx])?;
        }
        Some(Self { dims })
    }

    pub fn is_empty(&self) -> bool {
        self.dims.iter().any(|dim| dim.is_empty())
    }

    pub fn size(&self) -> usize {
        if self.is_empty() {
            return 0;
        }

        self.dims
            .iter()
            .fold(1usize, |acc, dim| acc.saturating_mul(dim.len()))
    }

    pub fn address_at(&self, index: usize) -> Address {
        let mut remaining = index;
        let mut nibbles = [0u8; 32];

        for idx in (0..32).rev() {
            let radix = self.dims[idx].len();
            let digit = if radix == 0 { 0 } else { remaining % radix };
            remaining = remaining.checked_div(radix).unwrap_or(0);
            nibbles[idx] = self.dims[idx].nth(digit).unwrap_or(0);
        }

        Address::from_nibbles(&nibbles)
    }
}
