use serde::{Deserialize, Serialize};

pub const LOCATION_ALPHABET: &[u8] = b"0123456789abcdefghijklmnopqrstuv";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IPv6Tokenizer {
    token_positions: Vec<TokenKind>,
    position_lookup: Vec<[usize; 16]>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TokenKind {
    Pad,
    Hex { position: usize, value: char },
}

impl Default for IPv6Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl IPv6Tokenizer {
    pub(super) fn validate(&self) -> Result<(), String> {
        let canonical = Self::new();
        if self.token_positions != canonical.token_positions
            || self.position_lookup != canonical.position_lookup
        {
            return Err("6VecLM tokenizer does not match the positional nybble vocabulary".into());
        }
        Ok(())
    }

    pub fn new() -> Self {
        let mut token_positions = Vec::new();
        let mut position_lookup = vec![[usize::MAX; 16]; LOCATION_ALPHABET.len()];

        let mut push_token = |kind: TokenKind| -> usize {
            let idx = token_positions.len();
            token_positions.push(kind);
            idx
        };

        let pad_id = push_token(TokenKind::Pad);
        debug_assert_eq!(pad_id, 0);

        for (pos, lookup) in position_lookup.iter_mut().enumerate() {
            for hex_char in "0123456789abcdef".chars() {
                let idx = push_token(TokenKind::Hex {
                    position: pos,
                    value: hex_char,
                });
                let digit = hex_char.to_digit(16).unwrap() as usize;
                lookup[digit] = idx;
            }
        }

        Self {
            token_positions,
            position_lookup,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.token_positions.len()
    }

    pub fn token_for_position(&self, position: usize, hex_value: u8) -> Option<usize> {
        let row = self.position_lookup.get(position)?;
        let idx = *row.get(hex_value as usize)?;
        if idx == usize::MAX { None } else { Some(idx) }
    }

    pub fn token_kind(&self, token_id: usize) -> TokenKind {
        self.token_positions
            .get(token_id)
            .copied()
            .unwrap_or(TokenKind::Pad)
    }

    pub fn encode_nybbles(&self, addr: &[u8; 16]) -> Vec<usize> {
        let mut tokens = Vec::with_capacity(32);
        for byte in addr {
            for nybble in [byte >> 4, byte & 0x0f] {
                let position = tokens.len();
                if let Some(token_id) = self.token_for_position(position, nybble) {
                    tokens.push(token_id);
                }
            }
        }
        tokens
    }

    pub fn detokenize_hex_stream(&self, tokens: &[usize]) -> Option<[u8; 16]> {
        if tokens.len() != 32 {
            return None;
        }

        let mut bytes = [0u8; 16];
        for (byte_idx, chunk) in tokens.chunks_exact(2).enumerate() {
            let high = self.hex_value_at(chunk[0], byte_idx * 2)?;
            let low = self.hex_value_at(chunk[1], byte_idx * 2 + 1)?;
            bytes[byte_idx] = (high << 4) | low;
        }
        Some(bytes)
    }

    fn hex_value_at(&self, token_id: usize, expected_position: usize) -> Option<u8> {
        match self.token_kind(token_id) {
            TokenKind::Hex { position, value } if position == expected_position => {
                value.to_digit(16).map(|digit| digit as u8)
            }
            _ => None,
        }
    }
}
