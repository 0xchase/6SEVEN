use super::aliases::AliasFilter;
use super::encoding::{
    Subspace as PatternSubspace, WILD, nibbles_to_addr, pattern_string_to_subspace,
    subspace_to_pattern_string,
};
use crate::{Address, Ipv6Prefix, TgaError};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Debug, Clone, Copy)]
pub(super) struct SixProbePattern {
    pub(super) subspace: PatternSubspace,
    wildcard_dims: [u8; 4],
    wildcard_count: u8,
}

impl SixProbePattern {
    pub(super) fn parse(pattern: &str) -> Result<Self, TgaError> {
        let subspace = pattern_string_to_subspace(pattern)?;
        Self::from_subspace(subspace).map_err(|error| match error {
            TgaError::Training(message) => TgaError::Training(format!(
                "Invalid six-probe pattern '{}': {}",
                pattern, message
            )),
            error => error,
        })
    }

    pub(super) fn from_subspace(subspace: PatternSubspace) -> Result<Self, TgaError> {
        let mut wildcard_dims = [0u8; 4];
        let mut wildcard_count = 0usize;

        for (idx, &nibble) in subspace.iter().enumerate() {
            if nibble == WILD {
                if wildcard_count == wildcard_dims.len() {
                    return Err(TgaError::Training(format!(
                        "expected at most {} wildcard nibbles",
                        wildcard_dims.len()
                    )));
                }
                wildcard_dims[wildcard_count] = idx as u8;
                wildcard_count += 1;
            } else if nibble > 0x0f {
                return Err(TgaError::Training(format!(
                    "invalid six-probe nibble value {nibble}"
                )));
            }
        }

        Ok(Self {
            subspace,
            wildcard_dims,
            wildcard_count: wildcard_count as u8,
        })
    }

    fn wildcard_dims(&self) -> &[u8] {
        &self.wildcard_dims[..self.wildcard_count as usize]
    }

    #[cfg(test)]
    pub(super) fn as_pattern_string(&self) -> String {
        subspace_to_pattern_string(&self.subspace)
    }
}

impl Serialize for SixProbePattern {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&subspace_to_pattern_string(&self.subspace))
    }
}

impl<'de> Deserialize<'de> for SixProbePattern {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let pattern = String::deserialize(deserializer)?;
        Self::parse(&pattern).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone)]
pub(super) struct SixProbePatternIter {
    wildcard_dims: [u8; 4],
    wildcard_count: u8,
    next_nibbles: PatternSubspace,
    done: bool,
}

impl SixProbePatternIter {
    pub(super) fn new(pattern: SixProbePattern) -> Self {
        let mut next_nibbles = pattern.subspace;
        for &dim in pattern.wildcard_dims() {
            next_nibbles[dim as usize] = 0;
        }

        Self {
            wildcard_dims: pattern.wildcard_dims,
            wildcard_count: pattern.wildcard_count,
            next_nibbles,
            done: false,
        }
    }

    fn advance(&mut self) {
        for idx in (0..self.wildcard_count as usize).rev() {
            let dim = self.wildcard_dims[idx];
            let slot = &mut self.next_nibbles[dim as usize];
            if *slot < 0x0f {
                *slot += 1;
                return;
            }
            *slot = 0;
        }

        self.done = true;
    }
}

impl Iterator for SixProbePatternIter {
    type Item = Address;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let address = nibbles_to_addr(&self.next_nibbles);
        self.advance();
        Some(address)
    }
}

#[derive(Clone)]
pub(crate) struct SixProbeIter {
    patterns: std::vec::IntoIter<SixProbePattern>,
    current: Option<SixProbePatternIter>,
    aliases: AliasFilter,
}

impl SixProbeIter {
    pub(super) fn new(patterns: Vec<SixProbePattern>, aliases: &[Ipv6Prefix]) -> Self {
        Self {
            patterns: patterns.into_iter(),
            current: None,
            aliases: AliasFilter::new(aliases),
        }
    }
}

impl Iterator for SixProbeIter {
    type Item = Result<Address, TgaError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(current) = &mut self.current
                && let Some(address) = current.next()
            {
                if !self.aliases.contains(address) {
                    return Some(Ok(address));
                }
                continue;
            }

            self.current = Some(SixProbePatternIter::new(self.patterns.next()?));
        }
    }
}
