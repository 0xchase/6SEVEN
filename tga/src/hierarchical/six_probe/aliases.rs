use crate::{Address, Ipv6Prefix, TgaError};
use std::io::{BufRead, BufReader};
use std::path::Path;

pub(super) fn load(path: &Path) -> Result<Vec<Ipv6Prefix>, TgaError> {
    let file = std::fs::File::open(path)
        .map_err(|error| TgaError::Config(format!("{}: {error}", path.display())))?;
    let mut prefixes = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|error| {
            TgaError::Config(format!("{}:{}: {error}", path.display(), index + 1))
        })?;
        let value = line.split('#').next().unwrap_or_default().trim();
        if value.is_empty() {
            continue;
        }
        let prefix = value.parse::<Ipv6Prefix>().map_err(|error| {
            TgaError::Config(format!("{}:{}: {error}", path.display(), index + 1))
        })?;
        prefixes.push(prefix.trunc());
    }
    prefixes.sort_unstable();
    prefixes.dedup();
    Ok(prefixes)
}

#[derive(Clone)]
pub(super) struct AliasFilter(sixseven_core::PrefixSet);
impl AliasFilter {
    pub(super) fn new(prefixes: &[Ipv6Prefix]) -> Self {
        Self(prefixes.iter().copied().collect())
    }
    pub(super) fn contains(&self, address: Address) -> bool {
        self.0.contains(address.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_lookup_matches_prefix_membership_at_boundaries() {
        let prefixes: Vec<Ipv6Prefix> = [
            "2001:db8::8/125",
            "2001:db8::a/128",
            "2001:db8::10/125",
            "2001:db8::/127",
            "8000::/1",
            "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/128",
        ]
        .iter()
        .map(|prefix| prefix.parse().unwrap())
        .collect();
        let filter = AliasFilter::new(&prefixes);
        let mut addresses = vec![0, 1, u128::MAX];
        for prefix in &prefixes {
            for boundary in [u128::from(prefix.network()), u128::from(prefix.broadcast())] {
                addresses.extend([
                    boundary.saturating_sub(1),
                    boundary,
                    boundary.saturating_add(1),
                ]);
            }
        }
        for address in addresses {
            let expected = prefixes
                .iter()
                .any(|prefix| prefix.contains(&std::net::Ipv6Addr::from(address)));
            assert_eq!(
                filter.contains(address.to_be_bytes()),
                expected,
                "{address:x}"
            );
        }
        let all = AliasFilter::new(&["::/0".parse().unwrap()]);
        assert!(all.contains([0; 16]));
        assert!(all.contains([255; 16]));
        assert!(!AliasFilter::new(&[]).contains([0; 16]));
    }
}
