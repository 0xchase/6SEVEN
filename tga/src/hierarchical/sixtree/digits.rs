use crate::{Address, TgaError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub(super) struct DigitLayout {
    bits: u8,
}

impl Default for DigitLayout {
    fn default() -> Self {
        Self { bits: 4 }
    }
}

impl TryFrom<u8> for DigitLayout {
    type Error = TgaError;

    fn try_from(bits: u8) -> Result<Self, Self::Error> {
        if (1..=5).contains(&bits) {
            Ok(Self { bits })
        } else {
            Err(TgaError::Config(
                "6Tree requires base 2, 4, 8, 16, or 32".into(),
            ))
        }
    }
}

impl From<DigitLayout> for u8 {
    fn from(layout: DigitLayout) -> Self {
        layout.bits
    }
}

impl DigitLayout {
    pub(super) fn from_base(base: u8) -> Result<Self, TgaError> {
        if !base.is_power_of_two() {
            return Err(TgaError::Config("6Tree base must be a power of two".into()));
        }
        Self::try_from(base.trailing_zeros() as u8)
    }

    pub(super) fn radix(self) -> usize {
        1 << self.bits
    }

    pub(super) fn dimensions(self) -> usize {
        128 / usize::from(self.bits)
    }

    pub(super) fn scope(self, address: Address) -> u128 {
        let width = self.dimensions() * usize::from(self.bits);
        let mask = u128::MAX.checked_shl(width as u32).unwrap_or(0);
        u128::from_be_bytes(address) & mask
    }

    pub(super) fn theoretical_dimensions(self) -> f64 {
        128.0 / f64::from(self.bits)
    }

    pub(super) fn shift(self, dimension: usize) -> usize {
        (self.dimensions() - dimension - 1) * usize::from(self.bits)
    }

    pub(super) fn mask(self, dimension: usize) -> u128 {
        (self.radix() as u128 - 1) << self.shift(dimension)
    }

    pub(super) fn digit(self, address: Address, dimension: usize) -> u8 {
        ((u128::from_be_bytes(address) & self.mask(dimension)) >> self.shift(dimension)) as u8
    }

    pub(super) fn wildcard_dimensions(self, bits: u32) -> u32 {
        bits / u32::from(self.bits)
    }
}
