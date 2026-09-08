mod alias;
mod digits;
mod region;
#[cfg(test)]
mod tests;
mod tree;

use crate::address::Address;
use crate::{Algorithm, Feedback, Ipv6Prefix, Observation, TargetModel, TgaError};
use alias::SearchPhase;
use clap::Args;
use digits::DigitLayout;
use rayon::prelude::*;
use region::{Region, RegionCursor, RegionSet};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

const DEFAULT_BASE: u8 = 16;
const DEFAULT_BATCH_PERCENT: u8 = 10;
const PARALLEL_NODE_SEED_THRESHOLD: usize = 16_384;
type NodeId = usize;

#[derive(Args, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SixTree {
    #[arg(long, default_value_t = DEFAULT_BASE, help = "Address vector base")]
    pub base: u8,

    #[arg(
        long = "leaf-max",
        help = "Maximum seeds per leaf with the selected base as the default"
    )]
    pub leaf_max: Option<usize>,

    #[arg(long = "batch-percent", default_value_t = DEFAULT_BATCH_PERCENT, help = "Queue percentage selected per round with a one-node minimum")]
    pub batch_percent: u8,

    #[arg(
        long = "dedup-seeds",
        default_value_t = true,
        action = clap::ArgAction::Set,
        help = "Remove duplicate seeds before building the tree"
    )]
    pub dedup_seeds: bool,

    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, help = "Probe suspicious regions for aliases before normal scanning")]
    pub alias_detection: bool,

    #[arg(
        long,
        default_value_t = 0,
        help = "Random seed for alias probe selection"
    )]
    pub alias_probe_seed: u64,
}

impl Default for SixTree {
    fn default() -> Self {
        Self {
            base: DEFAULT_BASE,
            leaf_max: None,
            batch_percent: DEFAULT_BATCH_PERCENT,
            dedup_seeds: true,
            alias_detection: true,
            alias_probe_seed: 0,
        }
    }
}

impl Algorithm for SixTree {
    const MODEL_VERSION: u32 = 4;
    const ID: &'static str = "6tree";
    const DESCRIPTION: &'static str =
        "6Tree DHC target generation with scan-feedback-driven dynamic expansion.";

    type Model = SixTreeModel;

    fn train(&self, observations: &[Observation]) -> Result<Self::Model, TgaError> {
        let layout = self.validate()?;
        self.train_from_seeds(observations, layout)
    }
}

impl SixTree {
    fn validate(&self) -> Result<DigitLayout, TgaError> {
        let layout = DigitLayout::from_base(self.base)?;
        if self.leaf_max == Some(0) {
            return Err(TgaError::Training("`leaf_max` must be at least 1".into()));
        }
        if !(1..=100).contains(&self.batch_percent) {
            return Err(TgaError::Training(
                "`batch-percent` must be in the range 1..=100".into(),
            ));
        }
        Ok(layout)
    }

    fn train_from_seeds(
        &self,
        observations: &[Observation],
        layout: DigitLayout,
    ) -> Result<SixTreeModel, TgaError> {
        let mut vectors: Vec<Address> = observations
            .par_iter()
            .filter_map(|obs| obs.active.then_some(obs.address))
            .collect();
        if vectors.is_empty() {
            return Err(TgaError::Training(
                "No active seed addresses provided".into(),
            ));
        }

        vectors.par_sort_unstable();
        if self.dedup_seeds {
            vectors.dedup();
        }

        let mut model = tree::build_model(
            &vectors,
            self.leaf_max.unwrap_or(layout.radix()),
            self.batch_percent,
            layout,
        );
        model.alias_detection = self.alias_detection;
        model.alias_probe_seed = self.alias_probe_seed;
        Ok(model)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Outcome {
    Active,
    Inactive,
    Skipped,
}

impl Outcome {
    fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::Active, _) | (_, Self::Active) => Self::Active,
            (Self::Inactive, _) | (_, Self::Inactive) => Self::Inactive,
            _ => Self::Skipped,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SixTreeNode {
    layout: DigitLayout,
    parent: Option<NodeId>,
    children: Vec<NodeId>,
    dimension_stack: Vec<usize>,
    targets: RegionSet,
    scanned: RegionSet,
    partial: HashMap<Address, Outcome>,
    probed: RegionSet,
    active: HashSet<Address>,
    last_expanded: Option<usize>,
    alias_checked: bool,
}

impl SixTreeNode {
    fn new(parent: Option<NodeId>, dimension_stack: Vec<usize>) -> Self {
        Self {
            layout: DigitLayout::default(),
            parent,
            children: Vec::new(),
            dimension_stack,
            targets: RegionSet::default(),
            scanned: RegionSet::default(),
            partial: HashMap::new(),
            probed: RegionSet::default(),
            active: HashSet::new(),
            last_expanded: None,
            alias_checked: false,
        }
    }

    fn expand(&mut self) {
        if let Some(dimension) = self.dimension_stack.pop() {
            self.targets.expand(dimension, self.layout);
            self.last_expanded = Some(dimension);
            self.alias_checked = false;
        }
    }

    fn frontier(&self, aliases: &RegionSet) -> RegionSet {
        let mut frontier = self.targets.clone();
        frontier.subtract(&self.scanned);
        frontier.subtract(aliases);
        frontier
    }

    fn has_frontier(&self, aliases: &RegionSet) -> bool {
        let frontier = self.frontier(aliases);
        let observed = self
            .partial
            .keys()
            .filter(|&&address| frontier.contains(address))
            .count();
        frontier.size_capped() > observed
    }

    fn probe_count(&self) -> usize {
        self.probed.size_capped().saturating_add(
            self.partial
                .values()
                .filter(|outcome| !matches!(outcome, Outcome::Skipped))
                .count(),
        )
    }

    fn density(&self) -> f64 {
        let probes = self.probe_count();
        if probes == 0 {
            0.0
        } else {
            self.active.len() as f64 / probes as f64
        }
    }

    fn finish_round(&mut self, aliases: &RegionSet) {
        let mut probed = self.frontier(aliases);
        for (&address, outcome) in &self.partial {
            if matches!(outcome, Outcome::Skipped) {
                probed.subtract(&RegionSet::from_regions([Region::singleton(address)]));
            } else if aliases.contains(address) {
                probed.insert(Region::singleton(address));
            }
        }
        self.probed.union(&probed);
        self.scanned = self.targets.clone();
        self.partial.clear();
        self.expand();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SixTreeModel {
    nodes: Vec<SixTreeNode>,
    queue: Vec<NodeId>,
    current_batch: Vec<NodeId>,
    completed_rounds: usize,
    batch_percent: u8,
    aliases: RegionSet,
    generation: Option<RegionCursor>,
    phase: SearchPhase,
    alias_detection: bool,
    alias_probe_seed: u64,
    detected_aliases: Vec<Ipv6Prefix>,
}

impl Default for SixTreeModel {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            queue: Vec::new(),
            current_batch: Vec::new(),
            completed_rounds: 0,
            batch_percent: DEFAULT_BATCH_PERCENT,
            aliases: RegionSet::default(),
            generation: None,
            phase: SearchPhase::Scanning,
            alias_detection: true,
            alias_probe_seed: 0,
            detected_aliases: Vec::new(),
        }
    }
}

impl SixTreeModel {
    pub fn detected_aliases(&self) -> &[Ipv6Prefix] {
        &self.detected_aliases
    }

    fn batch_frontier(&self) -> RegionSet {
        if let SearchPhase::Detecting(round) = &self.phase {
            return RegionSet::from_regions(round.pending().map(Region::singleton));
        }
        let mut targets = RegionSet::from_regions(
            self.current_batch
                .iter()
                .flat_map(|&id| self.nodes[id].targets.regions()),
        );
        let retired = RegionSet::from_regions(self.aliases.regions().chain(
            self.current_batch.iter().flat_map(|&id| {
                let node = &self.nodes[id];
                node.scanned
                    .regions()
                    .chain(node.partial.keys().copied().map(Region::singleton))
            }),
        ));
        targets.subtract(&retired);
        targets
    }

    fn is_descendant(&self, mut node: NodeId, ancestor: NodeId) -> bool {
        while let Some(parent) = self.nodes[node].parent {
            if parent == ancestor {
                return true;
            }
            node = parent;
        }
        false
    }

    fn replace_descendants(&mut self) {
        let mut promotions = Vec::new();
        let mut seen = HashSet::new();
        for &id in &self.current_batch {
            if let Some(parent) = self.nodes[id].parent
                && self.nodes[id].dimension_stack == self.nodes[parent].dimension_stack
                && seen.insert(parent)
            {
                promotions.push((parent, id));
            }
        }
        let candidates = promotions.clone();
        promotions.retain(|&(parent, _)| {
            !candidates
                .iter()
                .any(|&(other, _)| self.is_descendant(parent, other))
        });
        for (parent, source) in promotions {
            let retired: Vec<_> = self
                .current_batch
                .iter()
                .chain(&self.queue)
                .copied()
                .filter(|&id| self.is_descendant(id, parent))
                .collect();
            let mut scanned = RegionSet::default();
            let mut probed = RegionSet::default();
            let mut active = HashSet::new();
            for &id in &retired {
                let node = &self.nodes[id];
                scanned.union(&node.scanned);
                for &address in node.partial.keys() {
                    scanned.insert(Region::singleton(address));
                }
                probed.union(&node.probed);
                for (&address, outcome) in &node.partial {
                    if !matches!(outcome, Outcome::Skipped) {
                        probed.insert(Region::singleton(address));
                    }
                }
                active.extend(node.active.iter().copied());
            }
            self.nodes[parent].targets = self.nodes[source].targets.clone();
            self.nodes[parent].last_expanded = self.nodes[source].last_expanded;
            self.nodes[parent].alias_checked = self.nodes[source].alias_checked;
            self.nodes[parent].scanned = scanned;
            self.nodes[parent].partial.clear();
            self.nodes[parent].probed = probed;
            self.nodes[parent].active = active;
            for &id in &retired {
                let node = &mut self.nodes[id];
                node.targets = RegionSet::default();
                node.scanned = RegionSet::default();
                node.probed = RegionSet::default();
                node.partial = HashMap::new();
                node.active = HashSet::new();
            }
            let retired: HashSet<_> = retired.into_iter().collect();
            self.current_batch.retain(|id| !retired.contains(id));
            self.queue.retain(|id| !retired.contains(id));
            self.current_batch.push(parent);
        }
    }

    fn advance_completed_batches(&mut self) {
        loop {
            if let SearchPhase::Detecting(round) = &self.phase {
                if !round.is_complete() {
                    return;
                }
                self.finish_alias_round();
                continue;
            }
            if self.current_batch.is_empty()
                || self
                    .current_batch
                    .iter()
                    .any(|&id| self.nodes[id].has_frontier(&self.aliases))
            {
                return;
            }
            for &id in &self.current_batch {
                self.nodes[id].finish_round(&self.aliases);
            }
            self.merge_scanned_nodes();
            self.completed_rounds += 1;
            self.schedule_next_batch();
        }
    }

    fn merge_scanned_nodes(&mut self) {
        self.current_batch
            .sort_by(|&a, &b| self.nodes[b].density().total_cmp(&self.nodes[a].density()));
        let scanned = std::mem::take(&mut self.current_batch);
        let queued = std::mem::take(&mut self.queue);
        let mut scanned = scanned.into_iter().peekable();
        let mut queued = queued.into_iter().peekable();
        while let (Some(&left), Some(&right)) = (scanned.peek(), queued.peek()) {
            if self.nodes[left].density() >= self.nodes[right].density() {
                self.queue.extend(scanned.next());
            } else {
                self.queue.extend(queued.next());
            }
        }
        self.queue.extend(scanned);
        self.queue.extend(queued);
    }

    fn schedule_next_batch(&mut self) {
        self.generation = None;
        self.queue.retain(|&id| {
            let node = &self.nodes[id];
            !node.dimension_stack.is_empty() || node.has_frontier(&self.aliases)
        });
        if self.alias_detection
            && let Some(index) = self.queue.iter().position(|&id| {
                let node = &self.nodes[id];
                node.is_abnormal() && !node.alias_checked && node.has_frontier(&self.aliases)
            })
        {
            self.current_batch = vec![self.queue.remove(index)];
            self.start_alias_round();
            return;
        }
        self.phase = SearchPhase::Scanning;
        let count = (self
            .queue
            .len()
            .saturating_mul(usize::from(self.batch_percent))
            / 100)
            .max(1)
            .min(self.queue.len());
        self.current_batch = self.queue.drain(..count).collect();
        self.replace_descendants();
    }
}

impl TargetModel for SixTreeModel {
    fn allows_repeated_probes(&self) -> bool {
        self.alias_detection
    }

    fn detected_aliases(&self) -> &[Ipv6Prefix] {
        &self.detected_aliases
    }

    fn generate(&mut self, output: &mut [Address]) -> Result<crate::Generated, TgaError> {
        sixseven_core::generation::validate_output(output)?;
        self.advance_completed_batches();
        if self.generation.is_none() {
            self.generation = Some(self.batch_frontier().cursor());
        }
        let terminal = if self.current_batch.is_empty() {
            crate::GenerationState::Exhausted
        } else {
            crate::GenerationState::AwaitingFeedback
        };
        let cursor = self.generation.as_mut().expect("cursor initialized");
        let mut generated = crate::cursor::fill(&mut *cursor, output, terminal)?;
        if cursor.is_empty() {
            generated.state = terminal;
        }
        Ok(generated)
    }

    fn apply_feedback(&mut self, feedback: &[Feedback]) -> Result<(), TgaError> {
        for item in feedback {
            if let Feedback::Aliased(prefix) = item {
                let region = Region::prefix(*prefix);
                self.aliases.insert(region);
                if let SearchPhase::Detecting(round) = &mut self.phase {
                    round.exclude(region);
                }
                if let Some(cursor) = &mut self.generation {
                    cursor.exclude(region);
                }
            }
        }
        let mut observations: HashMap<Address, Outcome> = HashMap::new();
        for item in feedback {
            let (address, outcome) = match item {
                Feedback::Active(ip) => (ip.octets(), Outcome::Active),
                Feedback::Inactive(ip) => (ip.octets(), Outcome::Inactive),
                Feedback::Skipped(ip) => (ip.octets(), Outcome::Skipped),
                Feedback::Aliased(_) | Feedback::BatchComplete => continue,
            };
            observations
                .entry(address)
                .and_modify(|previous| *previous = previous.merge(outcome))
                .or_insert(outcome);
        }
        if let SearchPhase::Detecting(round) = &mut self.phase {
            for (address, outcome) in observations {
                if !self.aliases.contains(address)
                    && round.record(address, outcome)
                    && let Some(cursor) = &mut self.generation
                {
                    cursor.exclude(Region::singleton(address));
                }
            }
            self.advance_completed_batches();
            return Ok(());
        }
        for (address, outcome) in observations {
            if self.aliases.contains(address) {
                continue;
            }
            let mut recorded = false;
            for &id in &self.current_batch {
                let node = &mut self.nodes[id];
                if node.targets.contains(address) && !node.scanned.contains(address) {
                    let observed = node.partial.entry(address).or_insert(Outcome::Skipped);
                    *observed = observed.merge(outcome);
                    if matches!(observed, Outcome::Active) {
                        node.active.insert(address);
                    }
                    recorded = true;
                }
            }
            if recorded && let Some(cursor) = &mut self.generation {
                cursor.exclude(Region::singleton(address));
            }
        }
        self.advance_completed_batches();
        Ok(())
    }
}

impl std::fmt::Display for SixTreeModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "6Tree model with {} leaves, {} queued nodes, {} active batch nodes, {} completed feedback rounds",
            self.nodes
                .iter()
                .filter(|node| node.children.is_empty())
                .count(),
            self.queue.len(),
            self.current_batch.len(),
            self.completed_rounds
        )
    }
}
