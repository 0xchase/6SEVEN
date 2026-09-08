use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use rand::{SeedableRng, rngs::StdRng};

use super::generate::{EntropyIpGenerationScratch, EntropyIpRuntime};
use super::types::{EntropyIpModel, SegmentState};
use crate::{Address, TgaError};

#[derive(Debug, Clone)]
pub struct EntropyIpSegment<'a> {
    pub nybble_range: Range<usize>,
    pub states: &'a [SegmentState],
    pub modeled_state_indices: &'a [usize],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntropyIpSampleBatch {
    pub written: usize,
    pub attempts: usize,
}

#[derive(Clone)]
pub struct EntropyIpConditionedSampler {
    runtime: Arc<EntropyIpRuntime>,
    scratch: EntropyIpGenerationScratch,
    rng: StdRng,
    accepted_states: Vec<(usize, Vec<bool>)>,
}

impl EntropyIpModel {
    pub fn segments(&self) -> impl ExactSizeIterator<Item = EntropyIpSegment<'_>> {
        self.segments
            .iter()
            .zip(&self.bn_values)
            .map(|(segment, codes)| EntropyIpSegment {
                nybble_range: segment.start_nybble..segment.end_nybble + 1,
                states: &segment.states,
                modeled_state_indices: codes,
            })
    }

    pub fn conditioned_sampler(
        &self,
        evidence: &BTreeMap<usize, Vec<usize>>,
        seed: u64,
    ) -> Result<EntropyIpConditionedSampler, TgaError> {
        let runtime = self.get_runtime()?.clone();
        let mut accepted_states = Vec::with_capacity(evidence.len());
        for (&segment_index, state_indices) in evidence {
            let codes = self.bn_values.get(segment_index).ok_or_else(|| {
                TgaError::Config(format!("unknown Entropy/IP segment index {segment_index}"))
            })?;
            // An empty selection leaves the segment unconstrained in bngen.js.
            if state_indices.is_empty() {
                continue;
            }
            let mut accepted = vec![false; codes.len()];
            for state in state_indices {
                let index = codes.iter().position(|code| code == state).ok_or_else(|| {
                    TgaError::Config(format!(
                        "Entropy/IP segment {segment_index} has no modeled state index {state}"
                    ))
                })?;
                accepted[index] = true;
            }
            accepted_states.push((segment_index, accepted));
        }
        Ok(EntropyIpConditionedSampler {
            scratch: runtime.new_scratch(),
            runtime,
            rng: StdRng::seed_from_u64(seed),
            accepted_states,
        })
    }
}

impl EntropyIpConditionedSampler {
    pub fn sample(
        &mut self,
        output: &mut [Address],
        max_attempts: usize,
    ) -> Result<EntropyIpSampleBatch, TgaError> {
        sixseven_core::generation::validate_output(output)?;
        let mut batch = EntropyIpSampleBatch {
            written: 0,
            attempts: 0,
        };
        while batch.written < output.len() && batch.attempts < max_attempts {
            let mut address = [0; 16];
            self.runtime
                .generate_address(&mut self.scratch, &mut address, &mut self.rng)
                .map_err(TgaError::Generation)?;
            batch.attempts += 1;
            // Filter latent states so overlapping mined ranges keep their distinct probabilities.
            if self
                .accepted_states
                .iter()
                .all(|(segment, accepted)| accepted[self.scratch.sampled_state(*segment)])
            {
                output[batch.written] = address;
                batch.written += 1;
            }
        }
        Ok(batch)
    }
}
