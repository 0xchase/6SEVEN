//! NumPy RandomState compatibility for the released sklearn split.

pub(super) struct NumpyRandomState {
    key: [u32; 624],
    pos: usize,
}

impl NumpyRandomState {
    pub(super) fn new(seed: u32) -> Self {
        let mut key = [0u32; 624];
        key[0] = seed;
        for i in 1..624 {
            let prev = key[i - 1];
            key[i] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }

        Self { key, pos: 624 }
    }

    pub(super) fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let max = u32::try_from(i).expect("6GCVAE preprocessing does not support >u32 samples");
            let j = self.interval(max) as usize;
            values.swap(i, j);
        }
    }

    fn interval(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }

        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;

        loop {
            let value = self.next_u32() & mask;
            if value <= max {
                return value;
            }
        }
    }

    fn next_u32(&mut self) -> u32 {
        if self.pos >= self.key.len() {
            self.twist();
        }

        let mut y = self.key[self.pos];
        self.pos += 1;

        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    fn twist(&mut self) {
        const UPPER_MASK: u32 = 0x8000_0000;
        const LOWER_MASK: u32 = 0x7fff_ffff;
        const MATRIX_A: u32 = 0x9908_b0df;
        const PERIOD: usize = 397;

        for i in 0..624 {
            let next = (i + 1) % 624;
            let mid = (i + PERIOD) % 624;
            let y = (self.key[i] & UPPER_MASK) | (self.key[next] & LOWER_MASK);

            self.key[i] = self.key[mid] ^ (y >> 1);
            if (y & 1) != 0 {
                self.key[i] ^= MATRIX_A;
            }
        }

        self.pos = 0;
    }
}
