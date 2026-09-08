use crate::{
    Address, BitRange, Generated, GenerationInput, GenerationOptions, GenerationState,
    GenerationStats, Registry, TargetModel, TgaError,
};
use std::{
    collections::{HashSet, VecDeque},
    net::Ipv6Addr,
};

pub struct CandidateFilter {
    options: GenerationOptions,
    seen: HashSet<Ipv6Addr>,
    excluded: HashSet<Ipv6Addr>,
    pub stats: GenerationStats,
}

impl CandidateFilter {
    pub fn new(options: GenerationOptions) -> Self {
        let stats = GenerationStats {
            unique: options.unique.then_some(0),
            ..Default::default()
        };
        Self {
            excluded: options.exclude.iter().copied().collect(),
            options,
            seen: HashSet::new(),
            stats,
        }
    }

    pub fn remaining(&self) -> usize {
        self.options
            .count
            .saturating_sub(self.stats.written as usize)
    }

    pub fn accept(&mut self, address: Address) -> Result<bool, TgaError> {
        self.accept_excluding(address, &crate::PrefixSet::default())
    }

    pub fn accept_excluding(
        &mut self,
        address: Address,
        prefixes: &crate::PrefixSet,
    ) -> Result<bool, TgaError> {
        if self.stats.attempts >= self.options.max_attempts {
            return Err(TgaError::Generation(
                "candidate attempt budget exhausted".into(),
            ));
        }
        self.stats.attempts += 1;
        let ip = Ipv6Addr::from(address);
        if self.excluded.contains(&ip) || prefixes.contains(ip) {
            self.stats.excluded += 1;
            return Ok(false);
        }
        if self.options.unique && !self.seen.insert(ip) {
            self.stats.duplicates += 1;
            return Ok(false);
        }
        Ok(true)
    }

    pub fn written(&mut self) {
        self.stats.written += 1;
        if self.options.unique {
            self.stats.unique = Some(self.stats.written);
        }
    }
}

pub fn generate(
    model: &mut dyn TargetModel,
    options: GenerationOptions,
    mut emit: impl FnMut(Ipv6Addr) -> Result<(), TgaError>,
) -> Result<GenerationStats, TgaError> {
    model.set_budget(options.count);
    let mut filter = CandidateFilter::new(options);
    let mut output = vec![[0; 16]; filter.remaining().min(4096)];
    while filter.remaining() > 0 {
        let capacity = filter.remaining().min(output.len());
        let batch = model.generate(&mut output[..capacity])?;
        validate_generated(batch, capacity)?;
        for &address in &output[..batch.written] {
            if filter.accept(address)? {
                emit(Ipv6Addr::from(address))?;
                filter.written();
            }
        }
        if filter.remaining() > 0 && batch.state != GenerationState::Ready {
            return Err(TgaError::Generation(format!(
                "generation {:?} after {} targets",
                batch.state, filter.stats.written
            )));
        }
    }
    Ok(filter.stats)
}

pub fn validate_output(output: &[Address]) -> Result<(), TgaError> {
    if output.is_empty() {
        return Err(TgaError::Generation(
            "output buffer must not be empty".into(),
        ));
    }
    Ok(())
}

pub fn validate_generated(generated: Generated, capacity: usize) -> Result<(), TgaError> {
    if capacity == 0
        || generated.written > capacity
        || (generated.written == 0 && generated.state == GenerationState::Ready)
    {
        return Err(TgaError::Generation(
            "invalid generated count or empty ready output".into(),
        ));
    }
    Ok(())
}

pub fn compose(
    registry: &Registry,
    inputs: Vec<GenerationInput>,
) -> Result<Box<dyn TargetModel>, TgaError> {
    if inputs.is_empty() {
        return Err(TgaError::Config(
            "at least one generation input is required".into(),
        ));
    }
    let single = inputs.len() == 1;
    let mut occupied = [false; 128];
    let mut parts = Vec::new();
    for input in inputs {
        let (source, bits): (Box<dyn TargetModel>, _) = match input {
            GenerationInput::Model { model, bits } => (registry.open(&model)?, bits),
            GenerationInput::Random { bits } => (Box::new(RandomTargets::default()), bits),
            GenerationInput::Fixed { address, bits } => (Box::new(FixedTarget(address)), bits),
        };
        if single && bits.is_none() {
            return Ok(source);
        }
        let bits = bits
            .or_else(|| single.then(|| BitRange::new(0, 128).unwrap()))
            .ok_or_else(|| TgaError::Config("composition requires bit ranges".into()))?;
        for bit in bits.start()..bits.end() {
            if std::mem::replace(&mut occupied[bit as usize], true) {
                return Err(TgaError::Config("overlapping bit ranges".into()));
            }
        }
        parts.push(Component {
            model: source,
            bits,
            pending: VecDeque::new(),
            state: GenerationState::Ready,
        });
    }
    Ok(Box::new(Composed { parts }))
}

struct Composed {
    parts: Vec<Component>,
}

struct Component {
    model: Box<dyn TargetModel>,
    bits: BitRange,
    pending: VecDeque<Address>,
    state: GenerationState,
}

impl TargetModel for Composed {
    fn set_budget(&mut self, budget: usize) {
        for part in &mut self.parts {
            part.model.set_budget(budget);
        }
    }
    fn generate(&mut self, output: &mut [Address]) -> Result<Generated, TgaError> {
        validate_output(output)?;
        for part in &mut self.parts {
            if part.pending.is_empty() && part.state == GenerationState::Ready {
                let batch = part.model.generate(output)?;
                validate_generated(batch, output.len())?;
                part.pending.extend(output[..batch.written].iter().copied());
                part.state = batch.state;
            }
        }
        let count = self
            .parts
            .iter()
            .map(|part| part.pending.len())
            .min()
            .unwrap_or(0)
            .min(output.len());
        output[..count].fill([0; 16]);
        for part in &mut self.parts {
            for out in &mut output[..count] {
                let input = part
                    .pending
                    .pop_front()
                    .expect("component has enough buffered addresses");
                for bit in part.bits.start()..part.bits.end() {
                    let index = usize::from(bit / 8);
                    let mask = 1 << (7 - bit % 8);
                    out[index] = (out[index] & !mask) | (input[index] & mask);
                }
            }
        }
        let state = self
            .parts
            .iter()
            .find(|part| part.pending.is_empty() && part.state != GenerationState::Ready)
            .map_or(GenerationState::Ready, |part| part.state);
        Ok(Generated {
            written: count,
            state,
        })
    }
}

struct FixedTarget(Address);

impl TargetModel for FixedTarget {
    fn generate(&mut self, output: &mut [Address]) -> Result<Generated, TgaError> {
        validate_output(output)?;
        output.fill(self.0);
        Ok(Generated {
            written: output.len(),
            state: GenerationState::Ready,
        })
    }
}

#[derive(Default)]
pub struct RandomTargets {
    index: u64,
}

impl TargetModel for RandomTargets {
    fn generate(&mut self, output: &mut [Address]) -> Result<Generated, TgaError> {
        validate_output(output)?;
        let capacity = output.len();
        for (written, slot) in output.iter_mut().enumerate() {
            let Some(next) = self.index.checked_add(1) else {
                return Ok(Generated {
                    written,
                    state: GenerationState::Exhausted,
                });
            };
            let mut address = [0; 16];
            address[..8].copy_from_slice(&mix(self.index ^ 0x243f6a8885a308d3).to_be_bytes());
            address[8..].copy_from_slice(&mix(self.index ^ 0x13198a2e03707344).to_be_bytes());
            *slot = address;
            self.index = next;
        }
        Ok(Generated {
            written: capacity,
            state: GenerationState::Ready,
        })
    }
}

fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Sequence {
        next: u8,
        chunk: usize,
    }
    impl TargetModel for Sequence {
        fn generate(&mut self, output: &mut [Address]) -> Result<Generated, TgaError> {
            validate_output(output)?;
            let mut written = 0;
            for slot in output.iter_mut().take(self.chunk) {
                if self.next == 5 {
                    break;
                }
                *slot = [self.next; 16];
                written += 1;
                self.next += 1;
            }
            Ok(Generated {
                written,
                state: if self.next == 5 {
                    GenerationState::Exhausted
                } else {
                    GenerationState::Ready
                },
            })
        }
    }

    #[test]
    fn composition_preserves_candidates_from_short_ready_batches() {
        let mut model = Composed {
            parts: vec![
                Component {
                    model: Box::new(Sequence { next: 0, chunk: 4 }),
                    bits: BitRange::new(0, 64).unwrap(),
                    pending: VecDeque::new(),
                    state: GenerationState::Ready,
                },
                Component {
                    model: Box::new(Sequence { next: 0, chunk: 1 }),
                    bits: BitRange::new(64, 128).unwrap(),
                    pending: VecDeque::new(),
                    state: GenerationState::Ready,
                },
            ],
        };
        let mut output = Vec::new();
        let mut buffer = [[0; 16]; 4];
        loop {
            let batch = model.generate(&mut buffer).unwrap();
            output.extend_from_slice(&buffer[..batch.written]);
            if batch.state == GenerationState::Exhausted {
                break;
            }
        }
        assert_eq!(output, (0..5).map(|value| [value; 16]).collect::<Vec<_>>());
    }
}
