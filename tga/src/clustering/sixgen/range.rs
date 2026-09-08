use super::{Address, DIMS, ExactSize, Nibbles, SixGenRangeMode, cap_count};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, Hash)]
pub(super) struct NibbleDomain {
    pub(super) mask: u16,
}

// Reject empty domains without changing their serialized format.
impl<'de> Deserialize<'de> for NibbleDomain {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Domain {
            mask: u16,
        }
        let Domain { mask } = Domain::deserialize(deserializer)?;
        if mask == 0 {
            return Err(serde::de::Error::custom(
                "6Gen nybble domain must not be empty",
            ));
        }
        Ok(Self { mask })
    }
}

// The only IPv6 cardinality that does not fit in u128 is 2^128 itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum RangeSize {
    Finite(u128),
    Full,
}

impl RangeSize {
    pub(super) fn capped(self) -> usize {
        match self {
            Self::Finite(size) => cap_count(size),
            Self::Full => usize::MAX,
        }
    }

    pub(super) fn finite(self) -> u128 {
        match self {
            Self::Finite(size) => size,
            Self::Full => panic!("uncovered fragments must exclude the nonempty seed set"),
        }
    }
}

impl NibbleDomain {
    const FULL_MASK: u16 = 0xFFFF;

    pub(super) fn singleton(value: u8) -> Self {
        Self {
            mask: 1u16 << value,
        }
    }

    pub(super) fn full() -> Self {
        Self {
            mask: Self::FULL_MASK,
        }
    }

    #[cfg(test)]
    pub(super) fn contiguous(lo: u8, hi: u8) -> Self {
        debug_assert!(lo <= hi && hi < 16);

        if lo == 0 && hi == 0xF {
            return Self::full();
        }

        let width = hi - lo + 1;
        Self {
            mask: ((1u16 << width) - 1) << lo,
        }
    }

    pub(super) fn exact_len(self) -> ExactSize {
        self.mask.count_ones() as ExactSize
    }

    pub(super) fn contains(self, value: u8) -> bool {
        self.mask & (1u16 << value) != 0
    }

    pub(super) fn is_dynamic(self) -> bool {
        self.mask.count_ones() > 1
    }

    pub(super) fn with_value(self, value: u8) -> Self {
        Self {
            mask: self.mask | (1u16 << value),
        }
    }

    fn intersection(self, other: Self) -> Option<Self> {
        let mask = self.mask & other.mask;
        (mask != 0).then_some(Self { mask })
    }

    fn difference(self, other: Self) -> Option<Self> {
        let mask = self.mask & !other.mask;
        (mask != 0).then_some(Self { mask })
    }

    fn first_value(self) -> u8 {
        self.mask.trailing_zeros() as u8
    }

    fn next_value_after(self, current: u8) -> Option<u8> {
        ((current + 1)..16).find(|&value| self.contains(value))
    }

    fn value_at(self, rank: u32) -> u8 {
        debug_assert!(rank < self.mask.count_ones());
        let mut mask = self.mask;
        for _ in 0..rank {
            mask &= mask - 1;
        }
        mask.trailing_zeros() as u8
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub(super) struct AddressRange {
    pub(super) dims: [NibbleDomain; DIMS],
}

impl AddressRange {
    pub(super) fn from_seed(seed: &Nibbles) -> Self {
        Self {
            dims: std::array::from_fn(|idx| NibbleDomain::singleton(seed[idx])),
        }
    }

    #[cfg(test)]
    pub(super) fn contains(&self, seed: &Nibbles) -> bool {
        self.dims
            .iter()
            .zip(seed.iter())
            .all(|(interval, value)| interval.contains(*value))
    }

    #[cfg(test)]
    pub(super) fn distance_to(&self, seed: &Nibbles) -> usize {
        // Count only fixed nibbles that become dynamic.
        self.dims
            .iter()
            .zip(seed.iter())
            .filter(|(interval, value)| !interval.is_dynamic() && !interval.contains(**value))
            .count()
    }

    pub(super) fn grow_with_seed(&self, seed: &Nibbles, mode: SixGenRangeMode) -> Self {
        let mut next = *self;
        for (domain, value) in next.dims.iter_mut().zip(seed.iter().copied()) {
            if domain.contains(value) {
                continue;
            }

            *domain = match mode {
                SixGenRangeMode::Loose => NibbleDomain::full(),
                // Tight ranges retain only observed nybble values.
                SixGenRangeMode::Tight => domain.with_value(value),
            };
        }
        next
    }

    pub(super) fn size(&self) -> usize {
        self.exact_size().capped()
    }

    pub(super) fn exact_size(&self) -> RangeSize {
        match self
            .dims
            .iter()
            .try_fold(1u128, |total, domain| total.checked_mul(domain.exact_len()))
        {
            Some(size) => RangeSize::Finite(size),
            None => RangeSize::Full,
        }
    }

    pub(super) fn strict_subset_of(&self, other: &Self) -> bool {
        let mut any_strict = false;
        for (lhs, rhs) in self.dims.iter().zip(other.dims.iter()) {
            if lhs.mask & !rhs.mask != 0 {
                return false;
            }
            if lhs.mask != rhs.mask {
                any_strict = true;
            }
        }
        any_strict
    }

    fn intersection(&self, other: &Self) -> Option<Self> {
        let mut dims = [NibbleDomain { mask: 0 }; DIMS];
        for (idx, (lhs, rhs)) in self.dims.iter().zip(other.dims.iter()).enumerate() {
            dims[idx] = lhs.intersection(*rhs)?;
        }
        Some(Self { dims })
    }

    pub(super) fn subtract(self, other: &Self) -> Vec<Self> {
        let Some(overlap) = self.intersection(other) else {
            return vec![self];
        };
        if overlap == self {
            return Vec::new();
        }

        let mut pieces = Vec::new();
        let mut core = self;

        for dim in 0..DIMS {
            if let Some(extra) = core.dims[dim].difference(overlap.dims[dim]) {
                let mut left = core;
                left.dims[dim] = extra;
                pieces.push(left);
                core.dims[dim] = overlap.dims[dim];
            }
        }

        pieces
    }

    // Decode a range rank without enumeration.
    pub(super) fn address_at(&self, mut rank: u128) -> Address {
        let mut address = [0; 16];
        for (dim, domain) in self.dims.iter().enumerate().rev() {
            let radix = domain.exact_len();
            set_nibble_at(&mut address, dim, domain.value_at((rank % radix) as u32));
            rank /= radix;
        }
        debug_assert_eq!(rank, 0, "rank must be within the range");
        address
    }

    pub(super) fn iter(self) -> AddressRangeIter {
        AddressRangeIter::new(self)
    }
}

fn set_nibble_at(address: &mut Address, dim: usize, value: u8) {
    let byte_idx = dim / 2;
    match dim & 1 {
        0 => address[byte_idx] = (address[byte_idx] & 0x0F) | (value << 4),
        _ => address[byte_idx] = (address[byte_idx] & 0xF0) | (value & 0x0F),
    }
}

#[derive(Clone)]
pub(super) struct AddressRangeIter {
    range: AddressRange,
    exhausted: bool,
    values: [u8; DIMS],
    address: Address,
}

impl AddressRangeIter {
    pub(super) fn new(range: AddressRange) -> Self {
        let values = std::array::from_fn(|dim| range.dims[dim].first_value());
        let mut address = [0; 16];
        for (dim, value) in values.iter().copied().enumerate() {
            set_nibble_at(&mut address, dim, value);
        }

        Self {
            range,
            exhausted: false,
            values,
            address,
        }
    }

    fn advance(&mut self) {
        for dim in (0..DIMS).rev() {
            let domain = self.range.dims[dim];
            if let Some(next) = domain.next_value_after(self.values[dim]) {
                self.values[dim] = next;
                set_nibble_at(&mut self.address, dim, next);
                return;
            }

            let first = domain.first_value();
            self.values[dim] = first;
            set_nibble_at(&mut self.address, dim, first);
        }
        self.exhausted = true;
    }
}

impl Iterator for AddressRangeIter {
    type Item = Address;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        let address = self.address;
        self.advance();
        Some(address)
    }
}
