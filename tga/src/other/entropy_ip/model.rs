use super::generate::{self, EntropyIpGenerationScratch};
use super::types::EntropyIpModel;
#[cfg(test)]
use crate::AddressStream;
use crate::{Address, TgaError};
use rand::{SeedableRng, rngs::StdRng};
use std::sync::Arc;

impl EntropyIpModel {
    pub(super) fn get_runtime(&self) -> Result<&Arc<generate::EntropyIpRuntime>, TgaError> {
        let result = self.runtime.get_or_init(|| {
            generate::EntropyIpRuntime::compile(
                &self.segments,
                &self.bn_values,
                &self.segment_cardinalities,
                &self.network,
            )
            .map(Arc::new)
        });

        result
            .as_ref()
            .map_err(|error| TgaError::Generation(error.clone()))
    }

    #[cfg(test)]
    pub(super) fn stream(&self) -> Result<AddressStream, TgaError> {
        if self.segments.is_empty() {
            return Ok(Box::new(std::iter::empty()));
        }

        Ok(entropy_ip_stream(
            self.get_runtime()?.clone(),
            self.generation_seed,
        ))
    }
}
impl crate::TargetModel for EntropyIpModel {
    fn generate(&mut self, output: &mut [crate::Address]) -> Result<crate::Generated, TgaError> {
        if self.generation.0.is_none() {
            self.generation.0 = Some(
                if self.segments.is_empty() {
                    None
                } else {
                    Some(EntropyIpIter::new(
                        self.get_runtime()?.clone(),
                        self.generation_seed,
                    ))
                }
                .into_iter()
                .flatten(),
            );
        }
        crate::cursor::fill(
            self.generation.0.as_mut().expect("initialized cursor"),
            output,
            crate::GenerationState::Exhausted,
        )
    }
}

#[derive(Clone)]
pub(crate) struct EntropyIpIter {
    runtime: Arc<generate::EntropyIpRuntime>,
    rng: StdRng,
    scratch: EntropyIpGenerationScratch,
}

impl EntropyIpIter {
    fn new(runtime: Arc<generate::EntropyIpRuntime>, generation_seed: u64) -> Self {
        let scratch = runtime.new_scratch();
        Self {
            runtime,
            rng: StdRng::seed_from_u64(generation_seed),
            scratch,
        }
    }
}

impl Iterator for EntropyIpIter {
    type Item = Result<Address, TgaError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut address = [0u8; 16];
        match self
            .runtime
            .generate_address(&mut self.scratch, &mut address, &mut self.rng)
        {
            Ok(()) => Some(Ok(address)),
            Err(error) => Some(Err(TgaError::Generation(error))),
        }
    }
}

#[cfg(test)]
fn entropy_ip_stream(
    runtime: Arc<generate::EntropyIpRuntime>,
    generation_seed: u64,
) -> AddressStream {
    Box::new(EntropyIpIter::new(runtime, generation_seed))
}

impl std::fmt::Display for EntropyIpModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total_states: usize = self.segment_cardinalities.iter().sum();
        let boundaries: Vec<String> = self
            .segments
            .iter()
            .map(|s| format!("[{}..{}]", s.start_nybble, s.end_nybble))
            .collect();
        write!(
            f,
            "EntropyIp Model - {} segments {}, {} total states",
            self.segments.len(),
            boundaries.join(" "),
            total_states
        )
    }
}
