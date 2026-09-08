use serde::Serialize;
use sixseven_core::BitRange;
use std::{collections::BTreeMap, net::Ipv6Addr};

#[derive(Debug, Serialize)]
pub struct Dispersion {
    pub min_distance: u32,
    pub max_distance: u32,
    pub avg_distance: f64,
    pub total_pairs: u64,
}

pub fn dispersion(addresses: &[Ipv6Addr]) -> Dispersion {
    let mut result = Dispersion {
        min_distance: u32::MAX,
        max_distance: 0,
        avg_distance: 0.0,
        total_pairs: 0,
    };
    let mut total = 0u128;
    for (i, a) in addresses.iter().enumerate() {
        for b in &addresses[i + 1..] {
            let distance = (u128::from(*a) ^ u128::from(*b)).count_ones();
            result.min_distance = result.min_distance.min(distance);
            result.max_distance = result.max_distance.max(distance);
            total += u128::from(distance);
            result.total_pairs += 1;
        }
    }
    if result.total_pairs == 0 {
        result.min_distance = 0;
    } else {
        result.avg_distance = total as f64 / result.total_pairs as f64;
    }
    result
}

#[derive(Debug, Serialize)]
pub struct BitStatistics {
    pub one_counts: Vec<usize>,
    pub addresses: usize,
}

impl BitStatistics {
    pub fn one_bits(&self) -> usize {
        self.one_counts.iter().sum()
    }
    pub fn total_bits(&self) -> usize {
        self.addresses * self.one_counts.len()
    }
    pub fn entropy(&self) -> f64 {
        if self.total_bits() == 0 {
            return 0.0;
        }
        let p = self.one_bits() as f64 / self.total_bits() as f64;
        [p, 1.0 - p]
            .into_iter()
            .filter(|p| *p > 0.0)
            .map(|p| -p * p.log2())
            .sum()
    }
}

pub fn bit_statistics(addresses: &[Ipv6Addr], range: BitRange) -> BitStatistics {
    let mut one_counts = vec![0; usize::from(range.end() - range.start())];
    for address in addresses {
        let bytes = address.octets();
        for bit in range.start()..range.end() {
            one_counts[usize::from(bit - range.start())] +=
                usize::from((bytes[usize::from(bit / 8)] >> (7 - bit % 8)) & 1);
        }
    }
    BitStatistics {
        one_counts,
        addresses: addresses.len(),
    }
}

pub fn subnets(addresses: &[Ipv6Addr], prefix: u8) -> Result<BTreeMap<Ipv6Addr, usize>, String> {
    if prefix > 128 {
        return Err("prefix length must be at most 128".into());
    }
    let mask = if prefix == 0 {
        0
    } else {
        u128::MAX << (128 - prefix)
    };
    let mut counts = BTreeMap::new();
    for address in addresses {
        *counts
            .entry(Ipv6Addr::from(u128::from(*address) & mask))
            .or_default() += 1;
    }
    Ok(counts)
}
