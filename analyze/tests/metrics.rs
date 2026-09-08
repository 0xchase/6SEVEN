use analyze::metrics::{bit_statistics, dispersion, subnets};
use sixseven_core::BitRange;
use std::net::Ipv6Addr;

#[test]
fn bits_use_network_order() {
    let stats = bit_statistics(&["8000::1".parse().unwrap()], BitRange::new(0, 2).unwrap());
    assert_eq!(stats.one_counts, [1, 0]);
    assert_eq!(stats.entropy(), 1.0);
}

#[test]
fn prefixes_include_partial_bytes_and_boundary_lengths() {
    let addresses = [
        "2001:db8::1".parse().unwrap(),
        "2001:db8:8000::1".parse().unwrap(),
    ];
    assert_eq!(subnets(&addresses, 33).unwrap().len(), 2);
    assert_eq!(subnets(&addresses, 32).unwrap().len(), 1);
    assert_eq!(subnets(&addresses, 0).unwrap()[&Ipv6Addr::UNSPECIFIED], 2);
    assert_eq!(subnets(&addresses, 128).unwrap().len(), 2);
    assert!(subnets(&addresses, 129).is_err());
}

#[test]
fn dispersion_handles_empty_and_opposite_addresses() {
    assert_eq!(dispersion(&[]).min_distance, 0);
    let result = dispersion(&[Ipv6Addr::UNSPECIFIED, Ipv6Addr::from(u128::MAX)]);
    assert_eq!(result.total_pairs, 1);
    assert_eq!(result.avg_distance, 128.0);
}
