// Clustering methods, probably using linfa
mod sixgen;
mod sixgraph;

pub use sixgen::{SixGen, SixGenModel, SixGenRangeMode};
pub use sixgraph::{SixGraph, SixGraphModel};

/// Convert nibble array to byte array
pub fn nibbles_to_bytes(nibbles: &[u8; 32]) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    for i in 0..16 {
        bytes[i] = (nibbles[i * 2] << 4) | (nibbles[i * 2 + 1] & 0x0F);
    }
    bytes
}

/// Convert byte array to nibble array
pub fn bytes_to_nibbles(bytes: &[u8; 16]) -> [u8; 32] {
    let mut nibbles = [0u8; 32];
    for i in 0..16 {
        nibbles[i * 2] = (bytes[i] >> 4) & 0x0F;
        nibbles[i * 2 + 1] = bytes[i] & 0x0F;
    }
    nibbles
}
