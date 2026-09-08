use super::split::NumpyRandomState;
use std::net::{AddrParseError, Ipv6Addr};

use super::config::{N, TRAIN_TEST_SEED, TRAIN_TEST_SPLIT};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ManualClassification {
    pub fixed_iid: Vec<String>,
    pub low_64bit_subnet: Vec<String>,
    pub slaac_eui64: Vec<String>,
    pub slaac_privacy: Vec<String>,
}

pub fn bytes_to_nybble_sequence(addr: &[u8; 16]) -> [u8; N] {
    let mut nybbles = [0u8; N];
    for (i, byte) in addr.iter().enumerate() {
        nybbles[i * 2] = byte >> 4;
        nybbles[i * 2 + 1] = byte & 0x0f;
    }
    nybbles
}

pub fn nybble_sequence_to_bytes_i64(nybbles: &[i64]) -> Option<[u8; 16]> {
    if nybbles.len() != N {
        return None;
    }

    let mut addr = [0u8; 16];
    for i in 0..16 {
        let high = u8::try_from(nybbles[i * 2]).ok()?;
        let low = u8::try_from(nybbles[i * 2 + 1]).ok()?;
        if high > 0x0f || low > 0x0f {
            return None;
        }
        addr[i] = (high << 4) | low;
    }
    Some(addr)
}

/// Match the reference split with the NumPy MT19937 permutation.
pub fn split_train_test_indices(len: usize) -> (Vec<usize>, Vec<usize>) {
    if len == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut indices: Vec<usize> = (0..len).collect();
    let mut rng = NumpyRandomState::new(TRAIN_TEST_SEED as u32);
    rng.shuffle(&mut indices);

    let mut test_len = ((len as f64) * TRAIN_TEST_SPLIT).ceil() as usize;
    if len > 1 {
        test_len = test_len.clamp(1, len - 1);
    } else {
        test_len = 0;
    }
    let train = indices[test_len..].to_vec();
    let test = indices[..test_len].to_vec();
    (train, test)
}

/// Expand valid IPv6 text to eight lowercase, four-digit groups.
pub fn flatten_ipv6_text(addresses: &[String]) -> Result<Vec<String>, AddrParseError> {
    addresses
        .iter()
        .map(|address| {
            let address: Ipv6Addr = address.parse()?;
            Ok(address
                .segments()
                .map(|segment| format!("{segment:04x}"))
                .join(":"))
        })
        .collect()
}

/// Convert valid IPv6 text to exactly 32 lowercase hexadecimal digits.
pub fn to_training_rows(addresses: &[String]) -> Result<Vec<String>, AddrParseError> {
    addresses
        .iter()
        .map(|address| {
            let address: Ipv6Addr = address.parse()?;
            Ok(format!("{:032x}", u128::from(address)))
        })
        .collect()
}

/// Reproduce the order-dependent text classifier from `data_process.py`.
pub fn classify_manual(addresses: &[String]) -> ManualClassification {
    let mut out = ManualClassification::default();
    let mut previous = "";

    for address in addresses {
        if address.contains("::") {
            let last_string = address.rsplit("::").next().unwrap_or_default();
            if !last_string.contains(':') {
                out.fixed_iid.push(address.clone());
                previous = address;
                continue;
            }
        } else {
            let parts: Vec<&str> = address.split(':').collect();
            if parts.len() >= 3 {
                let p2 = parts[parts.len() - 2];
                let p3 = parts[parts.len() - 3];

                if p2.len() >= 2
                    && p3.len() >= 4
                    && p2.starts_with("fe")
                    && p3.get(2..4) == Some("ff")
                {
                    out.slaac_eui64.push(address.clone());
                    previous = address;
                    continue;
                }

                let long_tail = parts.len() >= 4
                    && parts[parts.len() - 1].len() >= 3
                    && parts[parts.len() - 2].len() >= 3
                    && parts[parts.len() - 3].len() >= 3
                    && parts[parts.len() - 4].len() >= 3;

                let previous_parts: Vec<&str> = previous.split(':').collect();
                let has_previous_tail = previous_parts.len() >= 4;

                if long_tail
                    && has_previous_tail
                    && previous_parts[previous_parts.len() - 2] != parts[parts.len() - 2]
                    && previous_parts[previous_parts.len() - 3] != parts[parts.len() - 3]
                    && previous_parts[previous_parts.len() - 4] != parts[parts.len() - 4]
                {
                    out.slaac_privacy.push(address.clone());
                    previous = address;
                    continue;
                }
            }
        }

        out.low_64bit_subnet.push(address.clone());
        previous = address;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_handles_mapped_addresses_and_rejects_invalid_text() {
        let inputs = vec!["::ffff:192.0.2.1".into(), "2001:DB8::ABCD".into()];
        let flattened = flatten_ipv6_text(&inputs).unwrap();
        let rows = to_training_rows(&inputs).unwrap();
        for ((original, expanded), row) in inputs.iter().zip(&flattened).zip(&rows) {
            assert_eq!(
                original.parse::<Ipv6Addr>().unwrap(),
                expanded.parse::<Ipv6Addr>().unwrap()
            );
            assert_eq!(row.len(), 32);
            assert!(
                row.bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
            );
        }
        for invalid in ["", "2001:::1", "gggg::1", "127.0.0.1"] {
            assert!(flatten_ipv6_text(&[invalid.into()]).is_err());
            assert!(to_training_rows(&[invalid.into()]).is_err());
        }
    }

    #[test]
    fn invalid_nybbles_are_rejected() {
        assert!(nybble_sequence_to_bytes_i64(&[0; 31]).is_none());
        for invalid in [-1, 16, 256] {
            let mut nybbles = [0; N];
            nybbles[17] = invalid;
            assert!(nybble_sequence_to_bytes_i64(&nybbles).is_none());
        }
    }

    #[test]
    fn text_classifier_does_not_slice_inside_unicode() {
        let _ = classify_manual(&["a:b:c:d:e:aéa:fe12:3456".into()]);
    }

    #[test]
    fn nybble_roundtrip() {
        let input = [
            0x20, 0x01, 0x0d, 0xb8, 0xab, 0xcd, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x01,
        ];
        let nybbles = bytes_to_nybble_sequence(&input);
        let as_i64: Vec<i64> = nybbles.iter().map(|&n| i64::from(n)).collect();
        let output = nybble_sequence_to_bytes_i64(&as_i64).unwrap();
        assert_eq!(input, output);
    }

    #[test]
    fn train_test_split_is_deterministic() {
        let (train_a, test_a) = split_train_test_indices(10);
        let (train_b, test_b) = split_train_test_indices(10);
        assert_eq!(train_a, train_b);
        assert_eq!(test_a, test_b);
        assert_eq!(train_a.len(), 8);
        assert_eq!(test_a.len(), 2);
        assert_eq!(train_a, vec![4, 9, 1, 6, 7, 3, 0, 5]);
        assert_eq!(test_a, vec![2, 8]);
    }

    #[test]
    fn split_matches_numpy_after_multiple_rng_state_cycles() {
        let (train, test) = split_train_test_indices(1000);
        assert_eq!(&test[..8], &[993, 859, 298, 553, 672, 971, 27, 231]);
        assert_eq!(&train[..8], &[687, 500, 332, 979, 817, 620, 814, 516]);
        assert_eq!(&train[792..], &[359, 707, 763, 835, 192, 629, 559, 684]);
        let mut all = train;
        all.extend(test);
        all.sort_unstable();
        assert_eq!(all, (0..1000).collect::<Vec<_>>());
    }

    #[test]
    fn train_test_split_matches_reference_on_small_inputs() {
        let expected = [
            (2usize, vec![0], vec![1]),
            (3, vec![1, 0], vec![2]),
            (4, vec![3, 1, 0], vec![2]),
            (5, vec![0, 1, 3, 4], vec![2]),
        ];

        for (len, train, test) in expected {
            let (actual_train, actual_test) = split_train_test_indices(len);
            assert_eq!(actual_train, train);
            assert_eq!(actual_test, test);
        }
    }

    #[test]
    fn flatten_and_row_conversion() {
        let in_addrs = vec!["2001:db8::1".to_string()];
        let flattened = flatten_ipv6_text(&in_addrs).unwrap();
        assert_eq!(flattened, vec!["2001:0db8:0000:0000:0000:0000:0000:0001"]);
        let rows = to_training_rows(&flattened).unwrap();
        assert_eq!(rows, vec!["20010db8000000000000000000000001"]);
    }

    #[test]
    fn flatten_preserves_ipv6_addresses() {
        let in_addrs = vec![
            "2001:db8::1".to_string(),
            "::1".to_string(),
            "2001:db8::".to_string(),
            "::".to_string(),
            "2001:0:0:1::1".to_string(),
        ];
        let flattened = flatten_ipv6_text(&in_addrs).unwrap();
        assert_eq!(
            flattened,
            vec![
                "2001:0db8:0000:0000:0000:0000:0000:0001",
                "0000:0000:0000:0000:0000:0000:0000:0001",
                "2001:0db8:0000:0000:0000:0000:0000:0000",
                "0000:0000:0000:0000:0000:0000:0000:0000",
                "2001:0000:0000:0001:0000:0000:0000:0001",
            ]
        );
    }

    #[test]
    fn manual_classification_matches_reference_python_fixture() {
        let addrs = vec![
            "2001:db8::1".to_string(),
            "2001:db8:0:1:0211:22ff:fe33:4455".to_string(),
            "2001:db8:0:1:abcd:1234:5678:9abc".to_string(),
            "2001:db8:0:1:abcd:1235:5679:9abd".to_string(),
            "2001:db8:0:1:abcd:1236:567a:9abe".to_string(),
        ];
        let out = classify_manual(&addrs);
        assert_eq!(out.fixed_iid.len(), 1);
        assert_eq!(out.fixed_iid, vec!["2001:db8::1"]);
        assert_eq!(out.slaac_eui64.len(), 1);
        assert_eq!(out.slaac_eui64, vec!["2001:db8:0:1:0211:22ff:fe33:4455"]);
        assert_eq!(out.slaac_privacy, vec!["2001:db8:0:1:abcd:1234:5678:9abc"]);
        assert_eq!(
            out.low_64bit_subnet,
            vec![
                "2001:db8:0:1:abcd:1235:5679:9abd",
                "2001:db8:0:1:abcd:1236:567a:9abe",
            ]
        );
    }
}
