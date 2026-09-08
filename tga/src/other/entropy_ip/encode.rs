use rand::Rng;
use std::collections::HashSet;

use crate::Address;

pub(super) fn get_nybble_window_value(nybbles: &[u8], start: usize, len: usize) -> u128 {
    let mut val = 0u128;
    for j in 0..len {
        val = (val << 4) | (nybbles[start + j] as u128);
    }
    val
}

pub(super) fn sample_bnf_rows<R: Rng + ?Sized>(
    rows: &[Vec<usize>],
    sample_size: usize,
    rng: &mut R,
) -> Vec<Vec<usize>> {
    if rows.len() <= sample_size {
        return rows.to_vec();
    }

    // Keep each sampled generator once because repeated generators are exhausted.
    let mut selected = Vec::with_capacity(sample_size.min(rows.len()));
    let mut seen = HashSet::with_capacity(sample_size.min(rows.len()));
    for _ in 0..sample_size {
        let index = rng.gen_range(0..rows.len());
        if seen.insert(index) {
            selected.push(rows[index].clone());
        }
    }
    selected
}

pub(super) fn address_to_nybbles(addr: &Address) -> [u8; 32] {
    let mut nybbles = [0u8; 32];
    for (i, &byte) in addr.iter().enumerate() {
        nybbles[i * 2] = byte >> 4;
        nybbles[i * 2 + 1] = byte & 0x0F;
    }
    nybbles
}
