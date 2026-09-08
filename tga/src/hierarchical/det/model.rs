use super::pattern::*;
use crate::{Address, Feedback, TgaError};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct DetNode {
    pub(super) parent: Option<usize>,
    pub(super) children: Vec<usize>,
    pub(super) dimension_stack: Vec<usize>,
    pub(super) target_patterns: Vec<TargetPattern>,
    pub(super) scanned_addresses: Vec<Address>,
    pub(super) skipped_addresses: HashSet<Address>,
    pub(super) active_hits: usize,
    pub(super) active_density: f64,
}

impl DetNode {
    pub(super) fn fully_observed(&self, base: usize) -> bool {
        let size = self
            .target_patterns
            .iter()
            .try_fold(0u128, |sum, pattern| sum.checked_add(pattern.size(base)?));
        size == Some(self.scanned_addresses.len() as u128)
    }

    pub(super) fn initialize_leaf_targets(&mut self) {
        if !self.children.is_empty() {
            return;
        }

        if let Some(dim) = self.dimension_stack.pop() {
            self.expand_targets(dim);
        }
    }

    pub(super) fn expand_next_dimension(&mut self) {
        if let Some(dim) = self.dimension_stack.pop() {
            self.expand_targets(dim);
        }
    }

    pub(super) fn expand_targets(&mut self, dim: usize) {
        for pattern in &mut self.target_patterns {
            pattern.digits[dim] = WILDCARD;
        }
        self.target_patterns.sort_unstable();
        self.target_patterns.dedup();
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(super) enum Outcome {
    Skipped,
    Inactive,
    Active,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DetModel {
    pub(super) generation: GenerationCursor,
    pub(super) pending_feedback: BTreeMap<Address, Outcome>,
    pub(super) bits_per_dimension: usize,
    pub(super) nodes: Vec<DetNode>,
    pub(super) queue: Vec<usize>,
    pub(super) current_batch: Vec<usize>,
    pub(super) completed_rounds: usize,
}

impl Default for DetModel {
    fn default() -> Self {
        Self {
            generation: Default::default(),
            pending_feedback: BTreeMap::new(),
            bits_per_dimension: 4,
            nodes: Vec::new(),
            queue: Vec::new(),
            current_batch: Vec::new(),
            completed_rounds: 0,
        }
    }
}

impl DetModel {
    pub(super) fn leaf_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|node| node.children.is_empty())
            .count()
    }

    pub(super) fn dimension_count(&self) -> usize {
        128 / self.bits_per_dimension
    }

    pub(super) fn base(&self) -> usize {
        1usize << self.bits_per_dimension
    }

    pub(super) fn finish_batch(
        &mut self,
        outcomes: &BTreeMap<Address, Outcome>,
    ) -> Result<(), TgaError> {
        if self.current_batch.is_empty() {
            return Err(TgaError::Feedback(
                "DET has no pending target batch to update".into(),
            ));
        }

        let bits = self.bits_per_dimension;
        for &address in outcomes.keys() {
            if !self.current_batch.iter().any(|&id| {
                self.nodes[id]
                    .target_patterns
                    .iter()
                    .any(|p| p.matches(&address, bits))
            }) {
                return Err(TgaError::Feedback(
                    "DET feedback address is outside the current target batch".into(),
                ));
            }
        }
        for &id in &self.current_batch {
            let node = &mut self.nodes[id];
            let mut scanned: HashSet<_> = node.scanned_addresses.iter().copied().collect();
            for (&address, &outcome) in outcomes {
                if node
                    .target_patterns
                    .iter()
                    .any(|p| p.matches(&address, bits))
                    && scanned.insert(address)
                {
                    if outcome == Outcome::Active {
                        node.active_hits += 1;
                    }
                    if outcome == Outcome::Skipped {
                        node.skipped_addresses.insert(address);
                    }
                }
            }
            node.scanned_addresses = sorted_addresses(&mut scanned);
            let probes = node.scanned_addresses.len() - node.skipped_addresses.len();
            node.active_density = if probes == 0 {
                0.0
            } else {
                node.active_hits as f64 / probes as f64
            };
        }
        // Partial feedback leaves unobserved targets available in the same round.
        self.generation = GenerationCursor::default();
        let base = self.base();
        if !self
            .current_batch
            .iter()
            .all(|&id| self.nodes[id].fully_observed(base))
        {
            return Ok(());
        }
        self.current_batch.retain(|&id| {
            let node = &mut self.nodes[id];
            if node.parent.is_none() && node.dimension_stack.is_empty() {
                return false;
            }
            node.expand_next_dimension();
            true
        });

        sort_nodes_by_density(&mut self.current_batch, &self.nodes);
        if self.completed_rounds == 0 {
            self.queue = self.current_batch.clone();
        } else {
            self.queue = merge_sorted_nodes(&self.current_batch, &self.queue, &self.nodes);
        }
        self.completed_rounds = self.completed_rounds.saturating_add(1);
        self.prepare_next_batch();
        Ok(())
    }

    pub(super) fn prepare_next_batch(&mut self) {
        if self.queue.is_empty() {
            self.current_batch.clear();
            return;
        }

        let batch_len = self.queue.len() / 10 + 1;
        let batch_len = batch_len.min(self.queue.len());
        self.current_batch = self.queue.drain(..batch_len).collect();
        self.replace_descendants();
    }

    pub(super) fn replace_descendants(&mut self) {
        let current_batch = self.current_batch.clone();
        let mut promoted = Vec::new();
        {
            let nodes = &mut self.nodes;
            let mut seen_promoted = HashSet::new();
            for &node_id in &current_batch {
                let Some(parent_id) = nodes[node_id].parent else {
                    continue;
                };

                if nodes[parent_id].dimension_stack == nodes[node_id].dimension_stack {
                    nodes[parent_id].target_patterns = nodes[node_id].target_patterns.clone();
                    if seen_promoted.insert(parent_id) {
                        promoted.push(parent_id);
                    }
                }
            }

            promoted.sort_unstable(); // Preorder IDs put ancestors first.
            for parent_id in promoted.clone() {
                if !promoted.contains(&parent_id) {
                    continue;
                }
                retire_children_into_parent(
                    nodes,
                    parent_id,
                    &mut self.current_batch,
                    &mut self.queue,
                    &mut promoted,
                );
            }
        }

        for node_id in promoted {
            if !self.current_batch.contains(&node_id) {
                self.current_batch.push(node_id);
            }
        }
    }
}

pub(super) fn sorted_addresses(addresses: &mut HashSet<Address>) -> Vec<Address> {
    let mut out = addresses.drain().collect::<Vec<_>>();
    out.sort_unstable();
    out
}

pub(super) fn sort_nodes_by_density(nodes: &mut [usize], model_nodes: &[DetNode]) {
    nodes.sort_by(|left, right| compare_node_density(model_nodes, *left, *right));
}

pub(super) fn compare_node_density(nodes: &[DetNode], left: usize, right: usize) -> Ordering {
    nodes[right]
        .active_density
        .total_cmp(&nodes[left].active_density)
}

pub(super) fn merge_sorted_nodes(left: &[usize], right: &[usize], nodes: &[DetNode]) -> Vec<usize> {
    let mut merged = Vec::with_capacity(left.len() + right.len());
    let mut left_idx = 0;
    let mut right_idx = 0;

    while left_idx < left.len() || right_idx < right.len() {
        if left_idx >= left.len() {
            merged.extend_from_slice(&right[right_idx..]);
            break;
        }
        if right_idx >= right.len() {
            merged.extend_from_slice(&left[left_idx..]);
            break;
        }

        let ordering = compare_node_density(nodes, left[left_idx], right[right_idx]);
        if ordering != Ordering::Greater {
            merged.push(left[left_idx]);
            left_idx += 1;
        } else {
            merged.push(right[right_idx]);
            right_idx += 1;
        }
    }

    merged
}

pub(super) fn retire_children_into_parent(
    nodes: &mut [DetNode],
    parent_id: usize,
    current_batch: &mut Vec<usize>,
    queue: &mut Vec<usize>,
    promoted: &mut Vec<usize>,
) {
    // Retire deeper frontier nodes to keep promoted regions disjoint.
    let mut descendants = nodes[parent_id].children.clone();
    let mut child_ids = HashSet::new();
    while let Some(id) = descendants.pop() {
        child_ids.insert(id);
        descendants.extend_from_slice(&nodes[id].children);
    }
    let mut scanned = nodes[parent_id]
        .scanned_addresses
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let mut active_hits = nodes[parent_id].active_hits;
    let mut skipped = nodes[parent_id].skipped_addresses.clone();

    for child_id in current_batch
        .iter()
        .chain(queue.iter())
        .copied()
        .filter(|node_id| child_ids.contains(node_id))
        .collect::<Vec<_>>()
    {
        scanned.extend(nodes[child_id].scanned_addresses.iter().copied());
        skipped.extend(nodes[child_id].skipped_addresses.iter().copied());
        active_hits = active_hits.saturating_add(nodes[child_id].active_hits);
        nodes[child_id].scanned_addresses.clear();
        nodes[child_id].skipped_addresses.clear();
        nodes[child_id].target_patterns.clear();
        nodes[child_id].active_hits = 0;
    }

    nodes[parent_id].scanned_addresses = sorted_addresses(&mut scanned);
    nodes[parent_id].active_hits = active_hits;
    nodes[parent_id].skipped_addresses = skipped;
    let scanned_count = nodes[parent_id]
        .scanned_addresses
        .len()
        .saturating_sub(nodes[parent_id].skipped_addresses.len());
    nodes[parent_id].active_density = if scanned_count == 0 {
        0.0
    } else {
        active_hits as f64 / scanned_count as f64
    };

    current_batch.retain(|node_id| !child_ids.contains(node_id));
    queue.retain(|node_id| !child_ids.contains(node_id));
    promoted.retain(|node_id| !child_ids.contains(node_id));
}

impl std::fmt::Display for DetModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DET model (base {}, {} dims) with {} leaves, {} queued nodes, {} active batch nodes",
            self.base(),
            self.dimension_count(),
            self.leaf_count(),
            self.queue.len(),
            self.current_batch.len()
        )
    }
}

#[derive(Default, Clone, Serialize, Deserialize)]
pub(super) struct GenerationCursor {
    pub(super) next_node: usize,
    pub(super) next_pattern: usize,
    pub(super) current: Option<PatternExpansion>,
    pub(super) emitted: HashSet<Address>,
}

impl GenerationCursor {
    pub(super) fn next(
        &mut self,
        nodes: &[DetNode],
        batch: &[usize],
        base: usize,
        bits_per_dimension: usize,
    ) -> Option<Address> {
        loop {
            let node = &nodes[*batch.get(self.next_node)?];
            if let Some(current) = &mut self.current {
                for address in current.by_ref() {
                    let retired = batch
                        .iter()
                        .any(|&id| nodes[id].scanned_addresses.binary_search(&address).is_ok());
                    if !retired && self.emitted.insert(address) {
                        return Some(address);
                    }
                }
            }
            if let Some(pattern) = node.target_patterns.get(self.next_pattern) {
                self.next_pattern += 1;
                self.current = Some(PatternExpansion::new(pattern, base, bits_per_dimension));
            } else {
                self.next_node += 1;
                self.next_pattern = 0;
                self.current = None;
            }
        }
    }
}

impl crate::TargetModel for DetModel {
    fn generate(&mut self, output: &mut [crate::Address]) -> Result<crate::Generated, TgaError> {
        let terminal = if self.current_batch.is_empty() {
            crate::GenerationState::Exhausted
        } else {
            crate::GenerationState::AwaitingFeedback
        };
        let base = self.base();
        let bits_per_dimension = self.bits_per_dimension;
        crate::cursor::fill(
            std::iter::from_fn(|| {
                self.generation
                    .next(&self.nodes, &self.current_batch, base, bits_per_dimension)
            }),
            output,
            terminal,
        )
    }
    fn apply_feedback(&mut self, feedback: &[Feedback]) -> Result<(), TgaError> {
        for item in feedback {
            if matches!(item, Feedback::BatchComplete) {
                let items = std::mem::take(&mut self.pending_feedback);
                if let Err(error) = self.finish_batch(&items) {
                    self.pending_feedback = items;
                    return Err(error);
                }
            } else {
                let (address, outcome) = match item {
                    Feedback::Active(address) => (address.octets(), Outcome::Active),
                    Feedback::Inactive(address) => (address.octets(), Outcome::Inactive),
                    Feedback::Skipped(address) => (address.octets(), Outcome::Skipped),
                    Feedback::Aliased(_) | Feedback::BatchComplete => continue,
                };
                if !self.current_batch.iter().any(|&id| {
                    self.nodes[id]
                        .target_patterns
                        .iter()
                        .any(|p| p.matches(&address, self.bits_per_dimension))
                }) {
                    return Err(TgaError::Feedback(
                        "DET feedback address is outside the current target batch".into(),
                    ));
                }
                // A response takes precedence over an inactive or skipped result.
                self.pending_feedback
                    .entry(address)
                    .and_modify(|old| *old = (*old).max(outcome))
                    .or_insert(outcome);
            }
        }
        Ok(())
    }
}
