use serde::{Deserialize, Serialize};

// Import nibble conversion functions from clustering module
use crate::clustering::{bytes_to_nibbles, nibbles_to_bytes};

/// IPv6 address representation used throughout TGA algorithms
pub use sixseven_core::Address;

/// Extension trait for Address type providing common operations
pub trait AddressExt {
    /// Convert to nibbles (32 4-bit values)
    fn to_nibbles(&self) -> [u8; 32];

    /// Create from nibbles
    fn from_nibbles(nibbles: &[u8; 32]) -> Self;

    /// Get a specific nibble (0-31)
    fn get_nibble(&self, index: usize) -> u8;

    /// Set a specific nibble (0-31)
    fn set_nibble(&mut self, index: usize, value: u8);

    /// Calculate Hamming distance to another address
    fn hamming_distance(&self, other: &Address) -> u32;

    /// Calculate nibble-level Hamming distance
    fn nibble_distance(&self, other: &Address) -> u32;
}

impl AddressExt for Address {
    fn to_nibbles(&self) -> [u8; 32] {
        bytes_to_nibbles(self)
    }

    fn from_nibbles(nibbles: &[u8; 32]) -> Self {
        nibbles_to_bytes(nibbles)
    }

    fn get_nibble(&self, index: usize) -> u8 {
        let byte_index = index / 2;
        if index.is_multiple_of(2) {
            self[byte_index] >> 4
        } else {
            self[byte_index] & 0x0F
        }
    }

    fn set_nibble(&mut self, index: usize, value: u8) {
        let byte_index = index / 2;
        if index.is_multiple_of(2) {
            self[byte_index] = (self[byte_index] & 0x0F) | (value << 4);
        } else {
            self[byte_index] = (self[byte_index] & 0xF0) | (value & 0x0F);
        }
    }

    fn hamming_distance(&self, other: &Address) -> u32 {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }

    fn nibble_distance(&self, other: &Address) -> u32 {
        self.iter()
            .zip(other.iter())
            .flat_map(|(a, b)| {
                let high_diff = if (a >> 4) != (b >> 4) { 1 } else { 0 };
                let low_diff = if (a & 0x0F) != (b & 0x0F) { 1 } else { 0 };
                [high_diff, low_diff]
            })
            .sum()
    }
}

/// Unified address vector representation (for multi-dimensional analysis) Generic over dimension size N.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AddressVector<const N: usize> {
    /// The values in the vector (e.g., nibbles, bytes, or encoded values)
    pub values: [u8; N],
}

/// Manual serialization/deserialization for AddressVector to handle generic arrays
impl<const N: usize> Serialize for AddressVector<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.values.as_slice().serialize(serializer)
    }
}

impl<'de, const N: usize> Deserialize<'de> for AddressVector<N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let values: Vec<u8> = Vec::deserialize(deserializer)?;
        if values.len() != N {
            return Err(serde::de::Error::custom(format!(
                "Expected {} values, got {}",
                N,
                values.len()
            )));
        }
        let mut array = [0u8; N];
        array.copy_from_slice(&values);
        Ok(AddressVector { values: array })
    }
}

impl<const N: usize> AddressVector<N> {
    /// Create a vector from raw values
    pub fn new(values: [u8; N]) -> Self {
        Self { values }
    }

    /// Get the dimensionality of this vector
    pub fn dimensions(&self) -> usize {
        N
    }

    /// Get value at a specific dimension
    pub fn get(&self, dim: usize) -> Option<u8> {
        self.values.get(dim).copied()
    }

    /// Convert back to an address (if possible)
    pub fn to_address(&self) -> Option<Address> {
        if N == 16 {
            // Convert the byte representation to an address.
            let mut addr = [0u8; 16];
            if self.values.len() == 16 {
                for (i, &val) in self.values.iter().enumerate() {
                    if i < 16 {
                        addr[i] = val;
                    }
                }
                Some(addr)
            } else {
                None
            }
        } else if N == 32 {
            // Convert the nibble representation to an address.
            let mut nibbles = [0u8; 32];
            if self.values.len() == 32 {
                for (i, &val) in self.values.iter().enumerate() {
                    if i < 32 {
                        nibbles[i] = val;
                    }
                }
                Some(Address::from_nibbles(&nibbles))
            } else {
                None
            }
        } else {
            None
        }
    }
}

// Convenience type aliases and constructors for common sizes
impl AddressVector<32> {
    /// Create a vector from an address as nibbles (32 dimensions)
    pub fn from_nibbles(addr: &Address) -> Self {
        Self {
            values: addr.to_nibbles(),
        }
    }
}

impl AddressVector<16> {
    /// Create a vector from an address as bytes (16 dimensions)
    pub fn from_bytes(addr: &Address) -> Self {
        Self { values: *addr }
    }
}
