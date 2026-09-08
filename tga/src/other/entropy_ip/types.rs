use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use super::generate::EntropyIpRuntime;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum SegmentState {
    Single(u128),
    Range(u128, u128),
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub(super) struct BayesianNetwork {
    pub(super) parents: Vec<Vec<usize>>,
    pub(super) cpts: Vec<HashMap<Vec<usize>, CptEntry>>,
}

#[derive(Serialize, Deserialize)]
pub struct EntropyIpModel {
    #[serde(skip)]
    pub(crate) generation: crate::cursor::GenerationCursor<
        std::iter::Flatten<std::option::IntoIter<super::model::EntropyIpIter>>,
    >,
    #[serde(default)]
    pub(super) generation_seed: u64,
    pub(super) segments: Vec<Segment>,
    #[serde(default)]
    pub(super) bn_values: Vec<Vec<usize>>,
    pub(super) segment_cardinalities: Vec<usize>,
    pub(super) network: BayesianNetwork,
    #[serde(skip, default)]
    pub(super) runtime: OnceLock<Result<Arc<EntropyIpRuntime>, String>>,
}

impl Clone for EntropyIpModel {
    fn clone(&self) -> Self {
        Self {
            generation: self.generation.clone(),
            generation_seed: self.generation_seed,
            segments: self.segments.clone(),
            bn_values: self.bn_values.clone(),
            segment_cardinalities: self.segment_cardinalities.clone(),
            network: self.network.clone(),
            runtime: self.runtime.clone(),
        }
    }
}

impl Default for EntropyIpModel {
    fn default() -> Self {
        Self {
            generation: Default::default(),
            generation_seed: 0,
            segments: Vec::new(),
            bn_values: Vec::new(),
            segment_cardinalities: Vec::new(),
            network: BayesianNetwork::default(),
            runtime: OnceLock::new(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub(super) struct Segment {
    pub(super) start_nybble: usize,
    pub(super) end_nybble: usize,
    pub(super) states: Vec<SegmentState>,
    pub(super) min_value: u128,
    pub(super) max_value: u128,
}

impl Segment {
    pub(super) fn nybble_width(&self) -> usize {
        self.end_nybble - self.start_nybble + 1
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub(super) struct CptEntry {
    pub(super) explicit_probs: Vec<(usize, f64)>,
    pub(super) default_prob: f64,
}

pub(super) struct SegmentLookup {
    exact: HashMap<u128, usize>,
    ranges: Vec<(u128, u128, usize)>,
}

pub(super) struct MiningResult {
    pub(super) states: Vec<SegmentState>,
    pub(super) min_value: u128,
    pub(super) max_value: u128,
}

impl SegmentLookup {
    pub(super) fn state_index_for(&self, value: u128) -> Option<usize> {
        if let Some(&idx) = self.exact.get(&value) {
            return Some(idx);
        }

        // Keep original emission order from segment mining (matches a3-encode.py).
        for &(min, max, idx) in &self.ranges {
            if value >= min && value <= max {
                return Some(idx);
            }
        }
        None
    }
}

pub(super) fn build_segment_lookup(segment: &Segment) -> SegmentLookup {
    let mut exact = HashMap::new();
    let mut ranges = Vec::new();
    for (state_idx, state) in segment.states.iter().enumerate() {
        match state {
            SegmentState::Single(value) => {
                exact.insert(*value, state_idx);
            }
            SegmentState::Range(min, max) => {
                ranges.push((*min, *max, state_idx));
            }
        }
    }
    SegmentLookup { exact, ranges }
}
