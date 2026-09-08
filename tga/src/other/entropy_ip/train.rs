use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeSet, BinaryHeap, HashMap};

#[cfg(test)]
use super::bn::visit_combinations;
use super::bn::{calculate_bde, calculate_cpt, graph_penalty};
use super::config::EntropyIp;
use super::encode::{address_to_nybbles, get_nybble_window_value, sample_bnf_rows};
use super::mining::mine_segment_states;
use super::segmentation::{compute_position_entropies, define_segments};
use super::types::EntropyIpModel;
use super::types::{BayesianNetwork, CptEntry, Segment, SegmentLookup, build_segment_lookup};
use crate::{Algorithm, Observation, TgaError};

pub(super) fn active_seed_nybbles(observations: &[Observation]) -> Result<Vec<[u8; 32]>, TgaError> {
    let seeds: Vec<[u8; 32]> = observations
        .iter()
        .filter(|obs| obs.active)
        .map(|obs| address_to_nybbles(&obs.address))
        .collect();

    if seeds.is_empty() {
        Err(TgaError::Training(
            "Training data cannot be empty.".to_string(),
        ))
    } else {
        Ok(seeds)
    }
}

pub(super) fn mine_training_segments(
    config: &EntropyIp,
    nybbles_list: &[[u8; 32]],
    rng: &mut StdRng,
) -> Vec<Segment> {
    let al = config.size / config.step;
    let isp_pos = config.isp_nybbles / config.step;
    let net_pos = config.net_nybbles / config.step;
    let entropies = compute_position_entropies(nybbles_list, config.step, al);
    let segment_bounds = define_segments(
        &entropies,
        isp_pos,
        net_pos,
        &config.thresholds,
        config.hysteresis,
    );

    segment_bounds
        .into_iter()
        .map(|(start_pos, end_pos)| {
            let start = start_pos * config.step;
            let end = ((end_pos + 1) * config.step) - 1;
            let num_nybbles = end - start + 1;
            let segment_values: Vec<u128> = nybbles_list
                .iter()
                .map(|nybbles| get_nybble_window_value(nybbles, start, num_nybbles))
                .collect();
            let mining_result = mine_segment_states(
                &segment_values,
                num_nybbles * 4,
                config.segment_sample_size,
                rng,
            );
            Segment {
                start_nybble: start,
                end_nybble: end,
                states: mining_result.states,
                min_value: mining_result.min_value,
                max_value: mining_result.max_value,
            }
        })
        .collect()
}

pub(super) fn encode_training_rows(
    config: &EntropyIp,
    nybbles_list: &[[u8; 32]],
    segments: &[Segment],
) -> Result<Vec<Vec<usize>>, TgaError> {
    let lookups: Vec<SegmentLookup> = segments.iter().map(build_segment_lookup).collect();
    let drop_unknown = config.effective_drop_unknown();

    let rows: Vec<Vec<usize>> = nybbles_list
        .par_iter()
        .map(|nybbles| encode_training_row(config, segments, &lookups, drop_unknown, nybbles))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect();

    if rows.is_empty() {
        return Err(TgaError::Training(
            "No encodable rows remained after unknown-value filtering.".to_string(),
        ));
    }

    Ok(rows)
}

fn encode_training_row(
    config: &EntropyIp,
    segments: &[Segment],
    lookups: &[SegmentLookup],
    drop_unknown: bool,
    nybbles: &[u8; 32],
) -> Result<Option<Vec<usize>>, TgaError> {
    let mut row = Vec::with_capacity(segments.len());

    for (segment, lookup) in segments.iter().zip(lookups) {
        let val = get_nybble_window_value(nybbles, segment.start_nybble, segment.nybble_width());
        match lookup.state_index_for(val) {
            Some(state_idx) => row.push(state_idx),
            None if config.rcode => {
                return Err(TgaError::Training(
                    "rcode produced placeholder segment states that cannot be decoded back into concrete IPv6 nybbles faithfully; disable rcode or provide seeds that avoid unknown states.".to_string(),
                ));
            }
            None if drop_unknown => return Ok(None),
            None => {
                return Err(TgaError::Training(
                    "drop_unknown=false without rcode is unsupported; original Entropy/IP either drops unknown rows or encodes them into an extra state.".to_string(),
                ));
            }
        }
    }

    Ok(Some(row))
}

type LearnedNetwork = (Vec<Vec<usize>>, Vec<usize>, BayesianNetwork);
type StateCodes = (Vec<Vec<usize>>, Vec<Vec<usize>>);

pub(super) fn learn_bayesian_network(
    config: &EntropyIp,
    encoded_rows: &[Vec<usize>],
    num_segments: usize,
    rng: &mut StdRng,
) -> Result<LearnedNetwork, TgaError> {
    let sampled_rows = if config.bnf_full {
        encoded_rows.to_vec()
    } else {
        sample_bnf_rows(encoded_rows, config.bnf_sample_size, rng)
    };
    let (bn_values, bn_rows) = remap_observed_state_codes(&sampled_rows, num_segments)?;

    let segment_cardinalities: Vec<usize> = bn_values.iter().map(Vec::len).collect();
    let parents = learn_parent_sets(config, &bn_rows, &segment_cardinalities);
    let cpts: Vec<HashMap<Vec<usize>, CptEntry>> = (0..num_segments)
        .into_par_iter()
        .map(|i| calculate_cpt(i, &parents[i], &bn_rows, &segment_cardinalities))
        .collect();

    Ok((
        bn_values,
        segment_cardinalities,
        BayesianNetwork { parents, cpts },
    ))
}

fn learn_parent_sets(
    config: &EntropyIp,
    bn_rows: &[Vec<usize>],
    segment_cardinalities: &[usize],
) -> Vec<Vec<usize>> {
    let num_segments = segment_cardinalities.len();
    let mut parents = vec![vec![]; num_segments];
    parents
        .par_iter_mut()
        .enumerate()
        .skip(1)
        .for_each(|(segment_idx, parent_set)| {
            *parent_set = best_parent_set(config, bn_rows, segment_idx, segment_cardinalities);
        });
    parents
}

fn best_parent_set(
    config: &EntropyIp,
    bn_rows: &[Vec<usize>],
    segment_idx: usize,
    segment_cardinalities: &[usize],
) -> Vec<usize> {
    let max_parents = config.effective_max_parents(segment_idx);
    if max_parents == 0 {
        return Vec::new();
    }

    let mut best_score = calculate_bde(bn_rows, segment_idx, &[], segment_cardinalities);
    let mut best_parents = Vec::new();
    let candidates = bnfinder_ordered_parent_candidates(segment_idx, segment_cardinalities);

    let mut pending = BinaryHeap::new();
    for &parent in &candidates {
        pending.push(Reverse(ParentCandidate::new(
            vec![parent],
            segment_cardinalities,
            bn_rows.len(),
        )));
    }
    while let Some(Reverse(candidate)) = pending.pop() {
        if candidate.penalty > -best_score {
            break;
        }
        let score = calculate_bde(
            bn_rows,
            segment_idx,
            &candidate.parents,
            segment_cardinalities,
        );
        if score > best_score {
            best_score = score;
            best_parents.clone_from(&candidate.parents);
        }
        if candidate.parents.len() < max_parents {
            let last = *candidate.parents.last().expect("nonempty candidate");
            let offset = candidates
                .iter()
                .position(|&parent| parent == last)
                .expect("candidate parent")
                + 1;
            for &parent in &candidates[offset..] {
                let mut successor = candidate.parents.clone();
                successor.push(parent);
                pending.push(Reverse(ParentCandidate::new(
                    successor,
                    segment_cardinalities,
                    bn_rows.len(),
                )));
            }
        }
    }

    best_parents
}

#[derive(Debug)]
struct ParentCandidate {
    penalty: f64,
    cardinalities: Vec<usize>,
    parents: Vec<usize>,
}

impl ParentCandidate {
    fn new(parents: Vec<usize>, cardinalities: &[usize], rows: usize) -> Self {
        Self {
            penalty: graph_penalty(&parents, cardinalities, rows),
            cardinalities: parents
                .iter()
                .map(|&parent| cardinalities[parent])
                .collect(),
            parents,
        }
    }
}

impl Ord for ParentCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.penalty
            .total_cmp(&other.penalty)
            .then_with(|| self.cardinalities.cmp(&other.cardinalities))
            .then_with(|| self.parents.cmp(&other.parents))
    }
}

impl PartialOrd for ParentCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ParentCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for ParentCandidate {}

fn remap_observed_state_codes(
    rows: &[Vec<usize>],
    num_segments: usize,
) -> Result<StateCodes, TgaError> {
    let mut observed_codes: Vec<BTreeSet<usize>> =
        (0..num_segments).map(|_| BTreeSet::new()).collect();
    for row in rows {
        if row.len() != num_segments {
            return Err(TgaError::Training(
                "encoded Entropy/IP rows have inconsistent segment widths".to_string(),
            ));
        }
        for (segment_idx, &code) in row.iter().enumerate() {
            observed_codes[segment_idx].insert(code);
        }
    }

    let bn_values: Vec<Vec<usize>> = observed_codes
        .into_iter()
        .map(|codes| codes.into_iter().collect())
        .collect();
    if bn_values.iter().any(Vec::is_empty) {
        return Err(TgaError::Training(
            "at least one Entropy/IP segment had no observed categorical states".to_string(),
        ));
    }

    let code_to_bn_state: Vec<HashMap<usize, usize>> = bn_values
        .iter()
        .map(|codes| {
            codes
                .iter()
                .enumerate()
                .map(|(bn_state, &code)| (code, bn_state))
                .collect()
        })
        .collect();

    let bn_rows = rows
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(segment_idx, code)| {
                    code_to_bn_state[segment_idx].get(code).copied().ok_or_else(|| {
                        TgaError::Training(format!(
                            "failed to remap Entropy/IP segment code {code} at segment {segment_idx}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok((bn_values, bn_rows))
}

pub(super) fn bnfinder_ordered_parent_candidates(
    segment_idx: usize,
    segment_cardinalities: &[usize],
) -> Vec<usize> {
    let mut candidates = (0..segment_idx).collect::<Vec<_>>();
    candidates.sort_by(|&left, &right| {
        let left_weight = bnfinder_parent_weight(segment_cardinalities[left]);
        let right_weight = bnfinder_parent_weight(segment_cardinalities[right]);
        left_weight
            .total_cmp(&right_weight)
            .then_with(|| left.cmp(&right))
    });
    candidates
}

fn bnfinder_parent_weight(cardinality: usize) -> f64 {
    (cardinality as f64).max(1.5)
}

impl Algorithm for EntropyIp {
    const ID: &'static str = "entropy";
    const DESCRIPTION: &'static str = "Entropy/IP algorithm for IPv6 address generation based on entropy analysis and segment mining";

    type Model = EntropyIpModel;

    fn train(&self, observations: &[Observation]) -> Result<Self::Model, TgaError> {
        self.validate()?;
        let nybbles_list = active_seed_nybbles(observations)?;
        let rng_seed = self.rng_seed.unwrap_or_else(rand::random);
        let mut training_rng = StdRng::seed_from_u64(rng_seed);
        let segments = mine_training_segments(self, &nybbles_list, &mut training_rng);
        let encoded_rows = encode_training_rows(self, &nybbles_list, &segments)?;
        let (bn_values, segment_cardinalities, network) =
            learn_bayesian_network(self, &encoded_rows, segments.len(), &mut training_rng)?;

        Ok(EntropyIpModel {
            generation: Default::default(),
            generation_seed: rng_seed,
            segments,
            bn_values,
            segment_cardinalities,
            network,
            runtime: Default::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn bounded_search_matches_exhaustive_parent_sets() {
        let mut rng = StdRng::seed_from_u64(8);
        for _ in 0..40 {
            let cardinalities: Vec<usize> = (0..6).map(|_| rng.gen_range(1..5)).collect();
            let rows: Vec<Vec<usize>> = (0..80)
                .map(|_| cardinalities.iter().map(|&r| rng.gen_range(0..r)).collect())
                .collect();
            for node in 1..6 {
                let candidates = bnfinder_ordered_parent_candidates(node, &cardinalities);
                let mut best = Vec::new();
                let mut best_score = calculate_bde(&rows, node, &best, &cardinalities);
                for size in 1..=node {
                    visit_combinations(&candidates, size, |parents| {
                        let score = calculate_bde(&rows, node, parents, &cardinalities);
                        if score > best_score {
                            best_score = score;
                            best = parents.to_vec();
                        }
                    });
                }
                assert_eq!(
                    best_parent_set(&EntropyIp::default(), &rows, node, &cardinalities),
                    best
                );
            }
        }
    }

    #[test]
    fn ordered_network_learns_joint_xor_dependency() {
        let rows: Vec<Vec<usize>> = (0..1000)
            .map(|i| {
                let a = i % 2;
                let b = (i / 2) % 2;
                vec![a, b, a ^ b]
            })
            .collect();
        assert_eq!(
            best_parent_set(&EntropyIp::default(), &rows, 2, &[2, 2, 2]),
            vec![0, 1]
        );
    }
}
