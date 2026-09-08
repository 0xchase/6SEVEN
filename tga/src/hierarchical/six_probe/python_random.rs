const MT_N: usize = 624;
const MT_M: usize = 397;
const MT_MATRIX_A: u32 = 0x9908_b0df;
const MT_UPPER_MASK: u32 = 0x8000_0000;
const MT_LOWER_MASK: u32 = 0x7fff_ffff;

#[derive(Clone)]
pub(super) struct PythonRandom {
    mt: [u32; MT_N],
    index: usize,
}

impl PythonRandom {
    pub(super) fn seed_from_u64(seed: u64) -> Self {
        let bytes = seed.to_le_bytes();
        let key = [
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        ];
        Self::init_by_array(trim_trailing_zero_words(&key))
    }

    fn init_genrand(seed: u32) -> Self {
        let mut mt = [0u32; MT_N];
        mt[0] = seed;
        for i in 1..MT_N {
            mt[i] = 1_812_433_253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }

        Self {
            mt,
            index: MT_N + 1,
        }
    }

    fn init_by_array(key: &[u32]) -> Self {
        let mut rng = Self::init_genrand(19_650_218);
        let mut i = 1usize;
        let mut j = 0usize;

        for _ in (1..=key.len().max(MT_N)).rev() {
            rng.mt[i] = (rng.mt[i]
                ^ ((rng.mt[i - 1] ^ (rng.mt[i - 1] >> 30)).wrapping_mul(1_664_525)))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;

            if i >= MT_N {
                rng.mt[0] = rng.mt[MT_N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }

        for _ in (1..MT_N).rev() {
            rng.mt[i] = (rng.mt[i]
                ^ ((rng.mt[i - 1] ^ (rng.mt[i - 1] >> 30)).wrapping_mul(1_566_083_941)))
            .wrapping_sub(i as u32);
            i += 1;

            if i >= MT_N {
                rng.mt[0] = rng.mt[MT_N - 1];
                i = 1;
            }
        }

        rng.mt[0] = 0x8000_0000;
        rng
    }

    fn genrand_u32(&mut self) -> u32 {
        if self.index >= MT_N {
            let mag01 = [0, MT_MATRIX_A];
            let mut y;

            for kk in 0..(MT_N - MT_M) {
                y = (self.mt[kk] & MT_UPPER_MASK) | (self.mt[kk + 1] & MT_LOWER_MASK);
                self.mt[kk] = self.mt[kk + MT_M] ^ (y >> 1) ^ mag01[(y & 0x1) as usize];
            }

            for kk in (MT_N - MT_M)..(MT_N - 1) {
                y = (self.mt[kk] & MT_UPPER_MASK) | (self.mt[kk + 1] & MT_LOWER_MASK);
                self.mt[kk] = self.mt[kk + MT_M - MT_N] ^ (y >> 1) ^ mag01[(y & 0x1) as usize];
            }

            y = (self.mt[MT_N - 1] & MT_UPPER_MASK) | (self.mt[0] & MT_LOWER_MASK);
            self.mt[MT_N - 1] = self.mt[MT_M - 1] ^ (y >> 1) ^ mag01[(y & 0x1) as usize];
            self.index = 0;
        }

        let mut y = self.mt[self.index];
        self.index += 1;

        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    fn getrandbits(&mut self, bits: u32) -> u32 {
        debug_assert!(bits <= 32);
        if bits == 0 {
            0
        } else {
            self.genrand_u32() >> (32 - bits)
        }
    }

    fn randbelow(&mut self, upper_exclusive: usize) -> usize {
        assert!(
            upper_exclusive > 0,
            "randbelow upper bound must be positive"
        );
        let upper = upper_exclusive as u32;
        let bits = u32::BITS - upper.leading_zeros();
        loop {
            let sample = self.getrandbits(bits);
            if sample < upper {
                return sample as usize;
            }
        }
    }

    pub(super) fn randint_inclusive(&mut self, lower: usize, upper: usize) -> usize {
        lower + self.randbelow(upper - lower + 1)
    }
}

fn trim_trailing_zero_words(words: &[u32]) -> &[u32] {
    let trimmed_len = words
        .iter()
        .rposition(|&word| word != 0)
        .map(|idx| idx + 1)
        .unwrap_or(1);
    &words[..trimmed_len]
}
