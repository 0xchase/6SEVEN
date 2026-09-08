use serde::{Deserialize, Serialize};

use crate::address::{Address, AddressVector};
use crate::index::{MAX_INDEX_DOMAIN, cap_index_size};
use crate::subspace::{NibbleSet, Subspace};

/// Unified address pattern representation with mask
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct AddressPattern {
    /// The pattern address
    pub address: Address,
    /// Mask indicating which bits/nibbles are fixed (1) or wildcard (0)
    pub mask: u128,
}

impl AddressPattern {
    /// Create a new pattern from a single address (all bits fixed)
    pub fn new(address: Address) -> Self {
        Self {
            address,
            mask: u128::MAX, // All bits fixed
        }
    }

    /// Create a pattern with a specific mask
    pub fn with_mask(address: Address, mask: u128) -> Self {
        Self { address, mask }
    }

    /// Check if an address matches this pattern
    pub fn matches(&self, addr: &Address) -> bool {
        for (i, &byte) in addr.iter().enumerate() {
            let byte_mask = ((self.mask >> (i * 8)) & 0xFF) as u8;
            if (byte & byte_mask) != (self.address[i] & byte_mask) {
                return false;
            }
        }
        true
    }

    /// Merge another address into this pattern, expanding wildcards as needed
    pub fn merge_address(&mut self, addr: &Address) {
        for (i, &byte) in addr.iter().enumerate() {
            let byte_diff = self.address[i] ^ byte;
            if byte_diff != 0 {
                // Clear mask bits where addresses differ
                let diff_mask = (byte_diff as u128) << (i * 8);
                self.mask &= !diff_mask;
            }
        }
    }

    /// Get the number of wildcard bits
    pub fn wildcard_bits(&self) -> u32 {
        128 - self.mask.count_ones()
    }

    /// Expand wildcard bits to nibble boundaries.
    pub fn expand_to_nibble_boundaries(&mut self) {
        let mut mask = self.mask;
        for i in 0..32 {
            let shift = i * 4;
            let nibble_mask = (mask >> shift) & 0xF;
            if nibble_mask != 0xF {
                // Clear the nibble when any bit is a wildcard.
                mask &= !(0xFu128 << shift);
            }
        }
        self.mask = mask;
        // Clear address bits in wildcard nibbles for canonical form
        for i in 0..16 {
            let byte_mask = ((self.mask >> (i * 8)) & 0xFF) as u8;
            self.address[i] &= byte_mask;
        }
    }

    /// Count nibble positions where all 4 bits are wildcard (free dimensions).
    pub fn wildcard_nibbles(&self) -> u32 {
        let mut count = 0;
        for i in 0..32 {
            let shift = i * 4;
            let nibble_mask = (self.mask >> shift) & 0xF;
            if nibble_mask == 0 {
                count += 1;
            }
        }
        count
    }

    /// Get the number of addresses this pattern represents
    pub fn size(&self) -> usize {
        match self.wildcard_bits() {
            0 => 1,
            bits if bits >= 64 => MAX_INDEX_DOMAIN,
            bits => 1usize << bits,
        }
    }

    /// Return the `index`th address in the pattern's numeric ordering.
    pub fn address_at(&self, index: usize) -> Address {
        let mut addr = self.address;
        let mut remaining = index;

        for byte_idx in (0..16).rev() {
            let byte_mask = ((self.mask >> (byte_idx * 8)) & 0xFF) as u8;
            if byte_mask == 0xFF {
                continue;
            }

            let mut byte = addr[byte_idx] & byte_mask;
            for bit in 0..8 {
                let bit_mask = 1u8 << bit;
                if byte_mask & bit_mask != 0 {
                    continue;
                }
                if remaining & 1 == 1 {
                    byte |= bit_mask;
                }
                remaining >>= 1;
            }
            addr[byte_idx] = byte;
        }

        addr
    }

    /// Calculate density given a seed count
    pub fn density(&self, seed_count: usize) -> f64 {
        seed_count as f64 / self.size() as f64
    }

    /// Calculate Hamming distance between this pattern and an address at the bit level.
    pub fn distance(&self, addr: &Address) -> u32 {
        let mut dist = 0;
        for (i, &byte) in addr.iter().enumerate() {
            let byte_mask = ((self.mask >> (i * 8)) & 0xFF) as u8;
            let pattern_byte = self.address[i] & byte_mask;
            let addr_byte = byte & byte_mask;
            if pattern_byte != addr_byte {
                dist += (pattern_byte ^ addr_byte).count_ones();
            }
        }
        dist
    }

    /// Check if this pattern is a subset of another pattern.
    pub fn is_subset_of(&self, other: &Self) -> bool {
        // A pattern is a subset if every fixed bit in `other` is also fixed to the same value in `self`.
        if (other.mask & !self.mask) != 0 {
            return false;
        }

        let relevant_mask = other.mask;
        for i in 0..16 {
            let byte_mask = ((relevant_mask >> (i * 8)) & 0xFF) as u8;
            if byte_mask != 0 {
                let self_byte = self.address[i] & byte_mask;
                let other_byte = other.address[i] & byte_mask;
                if self_byte != other_byte {
                    return false;
                }
            }
        }
        true
    }

    /// Generate a random address from this pattern
    pub fn generate_address(&self, rng: &mut rand::rngs::StdRng) -> Address {
        use rand::Rng;
        let mut addr = self.address;

        for (i, byte) in addr.iter_mut().enumerate() {
            let byte_mask = ((self.mask >> (i * 8)) & 0xFF) as u8;
            if byte_mask != 0xFF {
                // Generate random bits for wildcard positions
                let random_bits = rng.gen_range(0..=255u8);
                let fixed_bits = *byte & byte_mask;
                let wildcard_bits = random_bits & !byte_mask;
                *byte = fixed_bits | wildcard_bits;
            }
        }

        addr
    }

    pub fn to_subspace(&self) -> Subspace {
        let mut dims = [NibbleSet::EMPTY; 32];

        for (idx, dim) in dims.iter_mut().enumerate() {
            let byte_idx = idx / 2;
            let high = idx % 2 == 0;
            let shift = byte_idx * 8 + if high { 4 } else { 0 };
            let nibble_mask = ((self.mask >> shift) & 0x0f) as u8;
            let nibble_value = if high {
                self.address[byte_idx] >> 4
            } else {
                self.address[byte_idx] & 0x0f
            };
            *dim = NibbleSet::matching(nibble_value, nibble_mask);
        }

        Subspace { dims }
    }
}

/// Unified address vector pattern with mask for multi-dimensional analysis Generic over dimension size N.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AddressVectorPattern<const N: usize> {
    /// The pattern vector
    pub vector: AddressVector<N>,
    /// Mask positions are fixed when set and wildcard otherwise.
    pub mask: [bool; N],
}

/// Manual serialization/deserialization for AddressVectorPattern to handle generic arrays
impl<const N: usize> Serialize for AddressVectorPattern<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("AddressVectorPattern", 2)?;
        state.serialize_field("vector", &self.vector)?;
        state.serialize_field("mask", &self.mask.as_slice())?;
        state.end()
    }
}

impl<'de, const N: usize> Deserialize<'de> for AddressVectorPattern<N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper<const N: usize> {
            vector: AddressVector<N>,
            mask: Vec<bool>,
        }

        let helper: Helper<N> = Helper::deserialize(deserializer)?;
        if helper.mask.len() != N {
            return Err(serde::de::Error::custom(format!(
                "Expected {} mask values, got {}",
                N,
                helper.mask.len()
            )));
        }
        let mut mask_array = [false; N];
        mask_array.copy_from_slice(&helper.mask);
        Ok(AddressVectorPattern {
            vector: helper.vector,
            mask: mask_array,
        })
    }
}

impl<const N: usize> AddressVectorPattern<N> {
    /// Create a new pattern from a vector (all positions fixed)
    pub fn new(vector: AddressVector<N>) -> Self {
        let mask = [true; N];
        Self { vector, mask }
    }

    /// Create a pattern with a specific mask
    pub fn with_mask(vector: AddressVector<N>, mask: [bool; N]) -> Self {
        Self { vector, mask }
    }

    /// Check if a vector matches this pattern
    pub fn matches(&self, other: &AddressVector<N>) -> bool {
        for (i, (&pattern_val, &other_val)) in self
            .vector
            .values
            .iter()
            .zip(other.values.iter())
            .enumerate()
        {
            if self.mask[i] && pattern_val != other_val {
                return false;
            }
        }
        true
    }

    /// Merge another vector into this pattern, expanding wildcards as needed
    pub fn merge_vector(&mut self, other: &AddressVector<N>) {
        for (i, (&pattern_val, &other_val)) in self
            .vector
            .values
            .iter()
            .zip(other.values.iter())
            .enumerate()
        {
            if pattern_val != other_val {
                self.mask[i] = false; // Make this position a wildcard
            }
        }
    }

    /// Get the number of wildcard positions
    pub fn wildcard_count(&self) -> usize {
        self.mask.iter().filter(|&&fixed| !fixed).count()
    }

    /// Get the number of addresses this pattern represents
    pub fn size(&self) -> usize {
        let wildcard_count = self.wildcard_count();
        if wildcard_count >= 8 {
            MAX_INDEX_DOMAIN
        } else {
            cap_index_size(256u128.pow(wildcard_count as u32))
        }
    }

    /// Calculate density given a seed count
    pub fn density(&self, seed_count: usize) -> f64 {
        let size = self.size();
        if size == MAX_INDEX_DOMAIN {
            0.0 // Infinite pattern space
        } else {
            seed_count as f64 / size as f64
        }
    }

    /// Generate a random vector from this pattern
    pub fn generate_vector(&self, rng: &mut rand::rngs::StdRng) -> AddressVector<N> {
        use rand::Rng;
        let mut values = self.vector.values;

        for (i, &is_fixed) in self.mask.iter().enumerate() {
            if !is_fixed {
                values[i] = rng.r#gen(); // Generate random value for wildcard positions
            }
        }

        AddressVector { values }
    }

    /// Create pattern from multiple vectors
    pub fn from_vectors(vectors: &[AddressVector<N>]) -> Self {
        if vectors.is_empty() {
            return Self::new(AddressVector::new([0u8; N]));
        }

        let mut pattern = Self::new(vectors[0].clone());
        for vector in &vectors[1..] {
            pattern.merge_vector(vector);
        }
        pattern
    }
}

// 32-dimensional nibble pattern.
pub type NibbleVectorPattern = AddressVectorPattern<32>;
/// 16-dimensional pattern (bytes)
pub type ByteVectorPattern = AddressVectorPattern<16>;
/// 128-dimensional pattern (bits)
pub type BitVectorPattern = AddressVectorPattern<128>;
/// 64-dimensional pattern (quaternary, 2 bits each)
pub type QuaternaryVectorPattern = AddressVectorPattern<64>;
/// 26-dimensional pattern (base-32, 5 bits each)
pub type Base32VectorPattern = AddressVectorPattern<26>;
/// 43-dimensional pattern (octal, 3 bits each)
pub type OctalVectorPattern = AddressVectorPattern<43>;

/// Unified region representation for clustering algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Region {
    /// The addresses in this region
    pub addresses: Vec<Address>,
    /// The pattern that represents this region
    pub pattern: AddressPattern,
}

impl Region {
    /// Create a new region from a list of addresses, automatically generating the pattern
    pub fn new(addresses: Vec<Address>) -> Self {
        let pattern = Self::create_pattern_from_addresses(&addresses);
        Self { addresses, pattern }
    }

    /// Create a new region with a specific pattern
    pub fn with_pattern(addresses: Vec<Address>, pattern: AddressPattern) -> Self {
        Self { addresses, pattern }
    }

    /// Create a pattern from the given addresses
    pub fn create_pattern_from_addresses(addresses: &[Address]) -> AddressPattern {
        if addresses.is_empty() {
            return AddressPattern::new([0u8; 16]);
        }

        let mut pattern = AddressPattern::new(addresses[0]);
        for addr in &addresses[1..] {
            pattern.merge_address(addr);
        }
        pattern
    }

    /// Get the number of addresses in this region
    pub fn size(&self) -> usize {
        self.addresses.len()
    }

    /// Check if this region is empty
    pub fn is_empty(&self) -> bool {
        self.addresses.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_to_nibble_boundaries_partial_nibble() {
        // Two addresses differing in 1 bit within a nibble
        let a: Address = [0x10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let b: Address = [0x30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        let mut pattern = AddressPattern::new(a);
        pattern.merge_address(&b);
        // Before expansion: only the 1 differing bit is wildcard
        assert_eq!(pattern.wildcard_bits(), 1);

        pattern.expand_to_nibble_boundaries();
        // After expansion: entire high nibble of byte 0 is wildcard (4 bits)
        assert_eq!(pattern.wildcard_bits(), 4);
        assert_eq!(pattern.wildcard_nibbles(), 1);
    }

    #[test]
    fn expand_to_nibble_boundaries_full_nibble_unchanged() {
        // Four differing bits already form a full nibble wildcard.
        let a: Address = [0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let b: Address = [0x0F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        let mut pattern = AddressPattern::new(a);
        pattern.merge_address(&b);
        let bits_before = pattern.wildcard_bits();

        pattern.expand_to_nibble_boundaries();
        assert_eq!(pattern.wildcard_bits(), bits_before);
        assert_eq!(pattern.wildcard_nibbles(), 1);
    }

    #[test]
    fn expand_to_nibble_boundaries_no_wildcards() {
        let a: Address = [0x12, 0x34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut pattern = AddressPattern::new(a);
        pattern.expand_to_nibble_boundaries();
        assert_eq!(pattern.wildcard_bits(), 0);
        assert_eq!(pattern.wildcard_nibbles(), 0);
    }

    #[test]
    fn wildcard_nibbles_multiple() {
        let a: Address = [0x12, 0x34, 0x56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let b: Address = [0x13, 0x44, 0x56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        let mut pattern = AddressPattern::new(a);
        pattern.merge_address(&b);
        pattern.expand_to_nibble_boundaries();
        // Nibbles that differ: byte0 low nibble (2 vs 3), byte1 high nibble (3 vs 4)
        assert_eq!(pattern.wildcard_nibbles(), 2);
        assert_eq!(pattern.wildcard_bits(), 8);
    }
}
