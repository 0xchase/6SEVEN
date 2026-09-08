use crate::{Address, TgaError};
use rayon::prelude::*;

pub(crate) const WILD: u8 = 16;
pub(crate) const PARALLEL_NIBBLE_SCAN_MIN_SEEDS: usize = 1024;
pub(crate) type NibbleAddr = [u8; 32];
pub(crate) type Subspace = [u8; 32];

pub(crate) fn addr_to_nibbles(addr: &Address) -> NibbleAddr {
    let mut nibbles = [0u8; 32];
    for i in 0..16 {
        nibbles[i * 2] = addr[i] >> 4;
        nibbles[i * 2 + 1] = addr[i] & 0x0f;
    }
    nibbles
}

pub(crate) fn nibbles_to_addr(nibbles: &NibbleAddr) -> Address {
    debug_assert!(nibbles.iter().all(|nibble| *nibble < 16));

    let mut addr = [0u8; 16];
    for i in 0..16 {
        addr[i] = (nibbles[i * 2] << 4) | nibbles[i * 2 + 1];
    }
    addr
}

pub(crate) fn compute_subspace(seeds: &[NibbleAddr]) -> (Subspace, usize) {
    if seeds.is_empty() {
        return ([0u8; 32], 0);
    }

    let mut subspace = seeds[0];
    let scan = |(dim, slot): (usize, &mut u8)| {
        if seeds.iter().any(|seed| seed[dim] != *slot) {
            *slot = WILD;
        }
    };
    if seeds.len() >= PARALLEL_NIBBLE_SCAN_MIN_SEEDS {
        subspace.par_iter_mut().enumerate().for_each(scan);
    } else {
        subspace.iter_mut().enumerate().for_each(scan);
    }
    let dimension = subspace.iter().filter(|&&nibble| nibble == WILD).count();
    (subspace, dimension)
}

pub(crate) fn subspace_to_pattern_string(subspace: &Subspace) -> String {
    let mut out = String::with_capacity(39);
    for (idx, nibble) in subspace.iter().enumerate() {
        if idx > 0 && idx % 4 == 0 {
            out.push(':');
        }

        if *nibble == WILD {
            out.push('*');
        } else {
            out.push(
                char::from_digit(*nibble as u32, 16)
                    .expect("6Probe pattern nibbles must be hexadecimal digits or WILD"),
            );
        }
    }
    out
}

pub(crate) fn pattern_string_to_subspace(pattern: &str) -> Result<Subspace, TgaError> {
    let nibbles: Vec<char> = pattern.chars().filter(|ch| *ch != ':').collect();
    if nibbles.len() != 32 {
        return Err(TgaError::Training(format!(
            "Invalid six-probe pattern '{}': expected 32 nibbles",
            pattern
        )));
    }

    let mut subspace = [0u8; 32];
    for (idx, nibble_char) in nibbles.into_iter().enumerate() {
        subspace[idx] = if nibble_char == '*' {
            WILD
        } else {
            nibble_char.to_digit(16).ok_or_else(|| {
                TgaError::Training(format!(
                    "Invalid hex nibble '{}' in six-probe pattern '{}'",
                    nibble_char, pattern
                ))
            })? as u8
        };
    }

    Ok(subspace)
}
