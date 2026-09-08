use super::{FIRST_SAMPLED_POSITION, IPv6Tokenizer, TOTAL_LEN};

pub(super) fn encode_tokens(
    tokenizer: &IPv6Tokenizer,
    addr: &[u8; 16],
) -> Option<[u16; TOTAL_LEN]> {
    let encoded = tokenizer.encode_nybbles(addr);
    if encoded.len() != TOTAL_LEN {
        return None;
    }

    let mut tokens = [0u16; TOTAL_LEN];
    for (idx, token) in encoded.into_iter().enumerate() {
        tokens[idx] = token as u16;
    }
    Some(tokens)
}

pub(super) fn count_tokens(tokens: &[u16; TOTAL_LEN], token_counts: &mut [usize]) {
    for &token in tokens {
        let token = token as usize;
        if token < token_counts.len() {
            token_counts[token] += 1;
        }
    }
}

pub(super) fn build_position_candidates(
    tokenizer: &IPv6Tokenizer,
    vocab_token_mask: &[bool],
) -> Result<Vec<Vec<usize>>, String> {
    let mut candidates = vec![Vec::new(); TOTAL_LEN];
    for (position, candidate) in candidates
        .iter_mut()
        .enumerate()
        .skip(FIRST_SAMPLED_POSITION)
    {
        let mut tokens = Vec::new();
        for digit in 0..16 {
            if let Some(token) = tokenizer.token_for_position(position, digit as u8)
                && vocab_token_mask.get(token).copied().unwrap_or(false)
            {
                tokens.push(token);
            }
        }

        if tokens.is_empty() {
            return Err(format!(
                "No IPv62Vec vocabulary remains for nybble position {position}; \
                 original 6VecLM generation cannot continue without at least one sampled token at every generated position"
            ));
        }

        *candidate = tokens;
    }
    Ok(candidates)
}

pub(super) struct TokenCorpus<'a> {
    tokens: &'a [[u16; TOTAL_LEN]],
}

impl<'a> TokenCorpus<'a> {
    pub(super) fn new(tokens: &'a [[u16; TOTAL_LEN]]) -> Self {
        Self { tokens }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn row(&self, corpus_idx: usize) -> Option<&[u16; TOTAL_LEN]> {
        self.tokens.get(corpus_idx)
    }
}
