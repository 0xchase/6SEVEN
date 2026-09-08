use rand::Rng;
use std::collections::HashMap;

use crate::Address;

use super::types::{BayesianNetwork, CptEntry, Segment, SegmentState};

const DENSE_LOOKUP_MAX_SLOTS: usize = 1 << 20;
const MISSING_ENTRY_INDEX: u32 = u32::MAX;

pub(super) struct EntropyIpRuntime {
    segments: Box<[CompiledSegment]>,
    nodes: Box<[CompiledNode]>,
    max_parent_count: usize,
}

#[derive(Clone)]
pub(super) struct EntropyIpGenerationScratch {
    sampled_states: Vec<usize>,
    parent_key: Vec<usize>,
}

impl EntropyIpGenerationScratch {
    pub(super) fn sampled_state(&self, segment: usize) -> usize {
        self.sampled_states[segment]
    }

    pub(super) fn new(num_segments: usize, max_parent_count: usize) -> Self {
        Self {
            sampled_states: vec![0; num_segments],
            parent_key: Vec::with_capacity(max_parent_count),
        }
    }
}

struct CompiledSegment {
    write_steps: Box<[SegmentWriteStep]>,
    bn_state_to_segment_state: Box<[usize]>,
    states: Box<[CompiledSegmentState]>,
}

struct SegmentWriteStep {
    shift: u32,
    byte_index: usize,
    high_nybble: bool,
}

enum CompiledSegmentState {
    Single(u128),
    Range { min: u128, max: u128 },
}

struct CompiledNode {
    parents: Box<[usize]>,
    num_states: usize,
    lookup: CompiledLookup,
}

enum CompiledLookup {
    Dense(DenseLookup),
    Sparse(SparseLookup),
}

struct DenseLookup {
    parent_multipliers: Box<[usize]>,
    entry_indices: Box<[u32]>,
    entries: Box<[CompiledCptEntry]>,
}

struct SparseLookup {
    entries: HashMap<Box<[usize]>, CompiledCptEntry>,
}

struct CompiledCptEntry {
    explicit_thresholds: Box<[(usize, f64)]>,
    remaining_states: Box<[usize]>,
}

impl EntropyIpRuntime {
    pub(super) fn compile(
        segments: &[Segment],
        bn_values: &[Vec<usize>],
        segment_cardinalities: &[usize],
        network: &BayesianNetwork,
    ) -> Result<Self, String> {
        let num_segments = segments.len();
        if num_segments == 0 {
            return Err("Entropy/IP model has no segments".to_string());
        }
        validate_model_dimensions(num_segments, bn_values, segment_cardinalities, network)?;
        validate_segment_layout(segments)?;

        let mut compiled_segments = Vec::with_capacity(num_segments);
        for (segment_idx, segment) in segments.iter().enumerate() {
            let bn_state_to_segment_state = bn_values
                .get(segment_idx)
                .ok_or_else(|| format!("Entropy/IP segment {segment_idx} is missing BN values"))?;
            if bn_state_to_segment_state.len() != segment_cardinalities[segment_idx] {
                return Err(format!(
                    "Entropy/IP segment {segment_idx} has inconsistent BN cardinality"
                ));
            }
            let mut codes = bn_state_to_segment_state.clone();
            codes.sort_unstable();
            if codes.windows(2).any(|pair| pair[0] == pair[1]) {
                return Err(format!(
                    "Entropy/IP segment {segment_idx} has duplicate BN state mappings"
                ));
            }
            let states = compile_segment_states(segment_idx, segment)?;
            for &segment_state_idx in bn_state_to_segment_state {
                if segment_state_idx >= states.len() {
                    return Err(format!(
                        "Entropy/IP segment {segment_idx} maps BN state to missing segment state {segment_state_idx}"
                    ));
                }
            }

            compiled_segments.push(CompiledSegment {
                write_steps: compile_write_steps(segment.start_nybble, segment.end_nybble),
                bn_state_to_segment_state: bn_state_to_segment_state.clone().into_boxed_slice(),
                states,
            });
        }

        let mut max_parent_count = 0usize;
        let mut compiled_nodes = Vec::with_capacity(num_segments);
        for segment_idx in 0..num_segments {
            let parents = compile_parent_indices(segment_idx, &network.parents[segment_idx])?;
            max_parent_count = max_parent_count.max(parents.len());

            let num_states = segment_cardinalities[segment_idx];
            if num_states == 0 {
                return Err(format!(
                    "Entropy/IP segment {segment_idx} has zero encodable BN states"
                ));
            }

            let lookup = compile_lookup(
                segment_idx,
                &parents,
                segment_cardinalities,
                &network.cpts[segment_idx],
                num_states,
            )?;
            compiled_nodes.push(CompiledNode {
                parents,
                num_states,
                lookup,
            });
        }

        Ok(Self {
            segments: compiled_segments.into_boxed_slice(),
            nodes: compiled_nodes.into_boxed_slice(),
            max_parent_count,
        })
    }

    pub(super) fn new_scratch(&self) -> EntropyIpGenerationScratch {
        EntropyIpGenerationScratch::new(self.nodes.len(), self.max_parent_count)
    }

    pub(super) fn generate_address<R: Rng + ?Sized>(
        &self,
        scratch: &mut EntropyIpGenerationScratch,
        address: &mut Address,
        rng: &mut R,
    ) -> Result<(), String> {
        if self.nodes.is_empty() {
            return Err("Entropy/IP model has no reachable targets".to_string());
        }

        debug_assert_eq!(scratch.sampled_states.len(), self.nodes.len());
        address.fill(0);

        for (segment_idx, (segment, node)) in
            self.segments.iter().zip(self.nodes.iter()).enumerate()
        {
            let bn_state_idx =
                node.sample_state(&scratch.sampled_states, &mut scratch.parent_key, rng);
            scratch.sampled_states[segment_idx] = bn_state_idx;

            let segment_state_idx = *segment.bn_state_to_segment_state.get(bn_state_idx).ok_or_else(
                || {
                    format!(
                        "Entropy/IP segment {segment_idx} is missing a BN value mapping for state {bn_state_idx}"
                    )
                },
            )?;
            let value = segment.states[segment_state_idx].sample(rng);
            write_segment_value(&segment.write_steps, address, value);
        }
        Ok(())
    }
}

impl CompiledSegmentState {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u128 {
        match self {
            Self::Single(value) => *value,
            Self::Range { min, max } => rng.gen_range(*min..=*max),
        }
    }
}

impl CompiledNode {
    fn sample_state<R: Rng + ?Sized>(
        &self,
        sampled_states: &[usize],
        parent_key_scratch: &mut Vec<usize>,
        rng: &mut R,
    ) -> usize {
        let entry = match &self.lookup {
            CompiledLookup::Dense(lookup) => lookup.entry_for(&self.parents, sampled_states),
            CompiledLookup::Sparse(lookup) => {
                parent_key_scratch.clear();
                parent_key_scratch.extend(
                    self.parents
                        .iter()
                        .map(|&parent_idx| sampled_states[parent_idx]),
                );
                lookup.entry_for(parent_key_scratch.as_slice())
            }
        };
        sample_compiled_state(entry, self.num_states, rng)
    }
}

impl DenseLookup {
    fn entry_for<'a>(
        &'a self,
        parents: &[usize],
        sampled_states: &[usize],
    ) -> Option<&'a CompiledCptEntry> {
        if parents.is_empty() {
            return self.entry_from_flat_index(0);
        }

        let mut flat_index = 0usize;
        for (parent_idx, multiplier) in parents.iter().zip(self.parent_multipliers.iter()) {
            flat_index += sampled_states[*parent_idx] * multiplier;
        }
        self.entry_from_flat_index(flat_index)
    }

    fn entry_from_flat_index(&self, flat_index: usize) -> Option<&CompiledCptEntry> {
        let entry_index = *self.entry_indices.get(flat_index)?;
        if entry_index == MISSING_ENTRY_INDEX {
            None
        } else {
            self.entries.get(entry_index as usize)
        }
    }
}

impl SparseLookup {
    fn entry_for(&self, parent_key: &[usize]) -> Option<&CompiledCptEntry> {
        self.entries.get(parent_key)
    }
}

#[cfg(test)]
pub(super) fn sample_state<R: Rng + ?Sized>(
    entry: Option<&CptEntry>,
    num_states: usize,
    rng: &mut R,
) -> usize {
    if num_states == 0 {
        return 0;
    }
    let Some(entry) = entry else {
        return rng.gen_range(0..num_states);
    };

    let mut cprob: f64 = rng.r#gen();
    for &(state, prob) in &entry.explicit_probs {
        cprob -= prob;
        if cprob <= 0.0 {
            return state;
        }
    }

    let num_explicit = entry.explicit_probs.len();
    let num_remaining = num_states - num_explicit;
    if num_remaining == 0 {
        return rng.gen_range(0..num_states);
    }

    let k = rng.gen_range(0..num_remaining);
    let mut explicit_sorted = entry
        .explicit_probs
        .iter()
        .map(|&(state, _)| state)
        .collect::<Vec<_>>();
    explicit_sorted.sort_unstable();

    let mut skipped = 0;
    let mut prev = 0;
    for &ex in &explicit_sorted {
        let gap = ex - prev;
        if skipped + gap > k {
            return prev + (k - skipped);
        }
        skipped += gap;
        prev = ex + 1;
    }
    prev + (k - skipped)
}

fn sample_compiled_state<R: Rng + ?Sized>(
    entry: Option<&CompiledCptEntry>,
    num_states: usize,
    rng: &mut R,
) -> usize {
    if num_states == 0 {
        return 0;
    }
    let Some(entry) = entry else {
        return rng.gen_range(0..num_states);
    };

    let draw: f64 = rng.r#gen();
    for &(state, threshold) in entry.explicit_thresholds.iter() {
        if draw <= threshold {
            return state;
        }
    }
    if entry.remaining_states.is_empty() {
        return rng.gen_range(0..num_states);
    }
    entry.remaining_states[rng.gen_range(0..entry.remaining_states.len())]
}

fn validate_model_dimensions(
    num_segments: usize,
    bn_values: &[Vec<usize>],
    segment_cardinalities: &[usize],
    network: &BayesianNetwork,
) -> Result<(), String> {
    if bn_values.len() != num_segments
        || segment_cardinalities.len() != num_segments
        || network.parents.len() != num_segments
        || network.cpts.len() != num_segments
    {
        Err(
            "Entropy/IP model dimensions disagree across segments, BN mappings, and CPTs"
                .to_string(),
        )
    } else {
        Ok(())
    }
}

fn validate_segment_layout(segments: &[Segment]) -> Result<(), String> {
    let mut next_start = 0usize;
    for (segment_idx, segment) in segments.iter().enumerate() {
        validate_segment_bounds(segment_idx, segment)?;
        if segment.start_nybble != next_start {
            return Err(format!(
                "Entropy/IP segment {segment_idx} starts at nibble {}, expected contiguous coverage from nibble {next_start}",
                segment.start_nybble
            ));
        }
        next_start = segment.end_nybble.saturating_add(1);
    }
    Ok(())
}

fn compile_parent_indices(segment_idx: usize, parents: &[usize]) -> Result<Box<[usize]>, String> {
    for (offset, &parent_idx) in parents.iter().enumerate() {
        if parents[..offset].contains(&parent_idx) {
            return Err(format!(
                "Entropy/IP segment {segment_idx} has duplicate parent index {parent_idx}"
            ));
        }
        if parent_idx >= segment_idx {
            return Err(format!(
                "Entropy/IP segment {segment_idx} has non-causal parent index {parent_idx}"
            ));
        }
    }
    Ok(parents.to_vec().into_boxed_slice())
}

fn validate_segment_bounds(segment_idx: usize, segment: &Segment) -> Result<(), String> {
    if segment.start_nybble > segment.end_nybble || segment.end_nybble >= 32 {
        Err(format!(
            "Entropy/IP segment {segment_idx} has invalid nybble bounds [{}..{}]",
            segment.start_nybble, segment.end_nybble
        ))
    } else {
        Ok(())
    }
}

fn compile_segment_states(
    segment_idx: usize,
    segment: &Segment,
) -> Result<Box<[CompiledSegmentState]>, String> {
    if segment.states.is_empty() {
        return Err(format!(
            "Entropy/IP segment {segment_idx} has no mined segment states"
        ));
    }

    let max_value = max_segment_value(segment.nybble_width());
    segment
        .states
        .iter()
        .map(|state| match state {
            SegmentState::Single(value) => {
                validate_segment_state_value(segment_idx, *value, max_value)?;
                Ok(CompiledSegmentState::Single(*value))
            }
            SegmentState::Range(min, max) => {
                if min > max {
                    Err(format!(
                        "Entropy/IP segment {segment_idx} contains an invalid range state [{min}, {max}]"
                    ))
                } else {
                    validate_segment_state_value(segment_idx, *min, max_value)?;
                    validate_segment_state_value(segment_idx, *max, max_value)?;
                    Ok(CompiledSegmentState::Range {
                        min: *min,
                        max: *max,
                    })
                }
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Vec::into_boxed_slice)
}

fn max_segment_value(num_nybbles: usize) -> u128 {
    if num_nybbles >= 32 {
        u128::MAX
    } else {
        (1u128 << (num_nybbles * 4)) - 1
    }
}

fn validate_segment_state_value(
    segment_idx: usize,
    value: u128,
    max_value: u128,
) -> Result<(), String> {
    if value > max_value {
        Err(format!(
            "Entropy/IP segment {segment_idx} contains state value {value:#x} outside its nybble width"
        ))
    } else {
        Ok(())
    }
}

fn compile_write_steps(start_nybble: usize, end_nybble: usize) -> Box<[SegmentWriteStep]> {
    let mut steps = Vec::with_capacity(end_nybble - start_nybble + 1);
    for nybble in start_nybble..=end_nybble {
        steps.push(SegmentWriteStep {
            shift: ((end_nybble - nybble) * 4) as u32,
            byte_index: nybble / 2,
            high_nybble: nybble % 2 == 0,
        });
    }
    steps.into_boxed_slice()
}

fn compile_lookup(
    segment_idx: usize,
    parents: &[usize],
    segment_cardinalities: &[usize],
    cpts: &HashMap<Vec<usize>, CptEntry>,
    num_states: usize,
) -> Result<CompiledLookup, String> {
    let mut parent_dims = Vec::with_capacity(parents.len());
    for &parent_idx in parents {
        let cardinality = *segment_cardinalities.get(parent_idx).ok_or_else(|| {
            format!("Entropy/IP segment {segment_idx} references missing parent {parent_idx}")
        })?;
        if cardinality == 0 {
            return Err(format!(
                "Entropy/IP segment {segment_idx} references parent {parent_idx} with zero states"
            ));
        }
        parent_dims.push(cardinality);
    }

    let dense_capacity = dense_lookup_capacity(&parent_dims);
    if let Some(capacity) = dense_capacity.filter(|capacity| *capacity <= DENSE_LOOKUP_MAX_SLOTS) {
        return compile_dense_lookup(segment_idx, cpts, &parent_dims, capacity, num_states)
            .map(CompiledLookup::Dense);
    }

    compile_sparse_lookup(segment_idx, cpts, &parent_dims, num_states).map(CompiledLookup::Sparse)
}

fn dense_lookup_capacity(parent_dims: &[usize]) -> Option<usize> {
    parent_dims
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

fn compile_dense_lookup(
    segment_idx: usize,
    cpts: &HashMap<Vec<usize>, CptEntry>,
    parent_dims: &[usize],
    capacity: usize,
    num_states: usize,
) -> Result<DenseLookup, String> {
    let parent_multipliers = compute_parent_multipliers(parent_dims);
    let mut entry_indices = vec![MISSING_ENTRY_INDEX; capacity];
    let mut entries = Vec::with_capacity(cpts.len());

    for (parent_config, entry) in cpts {
        let flat_index =
            flatten_parent_config(segment_idx, parent_config, parent_dims, &parent_multipliers)?;
        if entry_indices[flat_index] != MISSING_ENTRY_INDEX {
            return Err(format!(
                "Entropy/IP segment {segment_idx} contains duplicate CPT rows for parent configuration {:?}",
                parent_config
            ));
        }
        entry_indices[flat_index] = entries.len() as u32;
        entries.push(compile_cpt_entry(segment_idx, num_states, entry)?);
    }

    Ok(DenseLookup {
        parent_multipliers: parent_multipliers.into_boxed_slice(),
        entry_indices: entry_indices.into_boxed_slice(),
        entries: entries.into_boxed_slice(),
    })
}

fn compile_sparse_lookup(
    segment_idx: usize,
    cpts: &HashMap<Vec<usize>, CptEntry>,
    parent_dims: &[usize],
    num_states: usize,
) -> Result<SparseLookup, String> {
    let mut entries = HashMap::with_capacity(cpts.len());
    for (parent_config, entry) in cpts {
        validate_parent_config(segment_idx, parent_config, parent_dims)?;
        let key = parent_config.clone().into_boxed_slice();
        if entries.contains_key(key.as_ref()) {
            return Err(format!(
                "Entropy/IP segment {segment_idx} contains duplicate CPT rows for parent configuration {:?}",
                parent_config
            ));
        }
        entries.insert(key, compile_cpt_entry(segment_idx, num_states, entry)?);
    }
    Ok(SparseLookup { entries })
}

fn compute_parent_multipliers(parent_dims: &[usize]) -> Vec<usize> {
    let mut multipliers = vec![1usize; parent_dims.len()];
    let mut stride = 1usize;
    for (idx, dim) in parent_dims.iter().enumerate().rev() {
        multipliers[idx] = stride;
        stride = stride.saturating_mul(*dim);
    }
    multipliers
}

fn flatten_parent_config(
    segment_idx: usize,
    parent_config: &[usize],
    parent_dims: &[usize],
    parent_multipliers: &[usize],
) -> Result<usize, String> {
    validate_parent_config(segment_idx, parent_config, parent_dims)?;
    Ok(parent_config
        .iter()
        .zip(parent_multipliers.iter())
        .map(|(value, multiplier)| value * multiplier)
        .sum())
}

fn validate_parent_config(
    segment_idx: usize,
    parent_config: &[usize],
    parent_dims: &[usize],
) -> Result<(), String> {
    if parent_config.len() != parent_dims.len() {
        return Err(format!(
            "Entropy/IP segment {segment_idx} has CPT key of width {}, expected {}",
            parent_config.len(),
            parent_dims.len()
        ));
    }
    for (offset, (&value, &dim)) in parent_config.iter().zip(parent_dims.iter()).enumerate() {
        if value >= dim {
            return Err(format!(
                "Entropy/IP segment {segment_idx} has out-of-range parent state {value} at parent offset {offset}"
            ));
        }
    }
    Ok(())
}

fn compile_cpt_entry(
    segment_idx: usize,
    num_states: usize,
    entry: &CptEntry,
) -> Result<CompiledCptEntry, String> {
    if !entry.default_prob.is_finite() || !(0.0..=1.0).contains(&entry.default_prob) {
        return Err(format!(
            "Entropy/IP segment {segment_idx} has invalid default CPT probability"
        ));
    }
    let mut seen_states = vec![false; num_states];
    let mut cumulative = 0.0_f64;
    let mut explicit_thresholds = Vec::with_capacity(entry.explicit_probs.len());

    for &(state, prob) in &entry.explicit_probs {
        if state >= num_states {
            return Err(format!(
                "Entropy/IP segment {segment_idx} references child state {state} outside 0..{num_states}"
            ));
        }
        if !prob.is_finite() || prob < 0.0 {
            return Err(format!(
                "Entropy/IP segment {segment_idx} contains non-finite or negative CPT probability {prob}"
            ));
        }
        if seen_states[state] {
            return Err(format!(
                "Entropy/IP segment {segment_idx} contains duplicate explicit CPT state {state}"
            ));
        }
        seen_states[state] = true;
        cumulative += prob;
        explicit_thresholds.push((state, cumulative));
    }

    if cumulative > 1.0 + 1e-10 {
        return Err(format!(
            "Entropy/IP segment {segment_idx} has CPT probability mass greater than one"
        ));
    }

    let remaining_states = seen_states
        .into_iter()
        .enumerate()
        .filter_map(|(state, seen)| (!seen).then_some(state))
        .collect::<Vec<_>>()
        .into_boxed_slice();

    Ok(CompiledCptEntry {
        explicit_thresholds: explicit_thresholds.into_boxed_slice(),
        remaining_states,
    })
}

fn write_segment_value(write_steps: &[SegmentWriteStep], address: &mut Address, value: u128) {
    for step in write_steps.iter() {
        let nybble = ((value >> step.shift) & 0xF) as u8;
        let byte = &mut address[step.byte_index];
        if step.high_nybble {
            *byte = (*byte & 0x0F) | (nybble << 4);
        } else {
            *byte = (*byte & 0xF0) | nybble;
        }
    }
}
