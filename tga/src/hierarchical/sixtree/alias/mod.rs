use super::*;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::BTreeMap;

const PROBES_PER_VALUE: usize = 10;
const MIN_SUSPICIOUS_SPACE: usize = 1 << 10;
const MAX_NORMAL_SPACE: usize = 1 << 20;
const DENSITY_THRESHOLD: f64 = 0.9;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) enum SearchPhase {
    Scanning,
    Detecting(AliasProbeRound),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct AliasProbeRound {
    probes: BTreeMap<Address, Option<Outcome>>,
}

impl AliasProbeRound {
    fn sample(node: &SixTreeNode, seed: u64, aliases: &RegionSet) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut probes = BTreeMap::new();
        if let Some(dimension) = node.last_expanded {
            for region in node.targets.regions() {
                for index in 0..node.layout.radix() * PROBES_PER_VALUE {
                    let address = region.probe(
                        dimension,
                        (index % node.layout.radix()) as u8,
                        rng.r#gen(),
                        node.layout,
                    );
                    let outcome = aliases.contains(address).then_some(Outcome::Skipped);
                    probes.insert(address, outcome);
                }
            }
        }
        Self { probes }
    }

    pub(super) fn pending(&self) -> impl Iterator<Item = Address> + '_ {
        self.probes
            .iter()
            .filter_map(|(&address, outcome)| outcome.is_none().then_some(address))
    }

    pub(super) fn is_complete(&self) -> bool {
        self.pending().next().is_none()
    }

    pub(super) fn record(&mut self, address: Address, outcome: Outcome) -> bool {
        let Some(previous) = self.probes.get_mut(&address) else {
            return false;
        };
        let merged = previous.map_or(outcome, |old| old.merge(outcome));
        let changed = *previous != Some(merged);
        *previous = Some(merged);
        changed
    }

    pub(super) fn exclude(&mut self, region: Region) {
        for (&address, outcome) in &mut self.probes {
            if region.contains(address) {
                *outcome = Some(Outcome::Skipped);
            }
        }
    }

    fn has_response(&self) -> bool {
        self.probes
            .values()
            .any(|outcome| matches!(outcome, Some(Outcome::Active)))
    }

    fn all_inactive(&self) -> bool {
        !self.probes.is_empty()
            && self
                .probes
                .values()
                .all(|outcome| matches!(outcome, Some(Outcome::Inactive)))
    }
}

impl SixTreeNode {
    pub(super) fn is_abnormal(&self) -> bool {
        if self.targets.size_capped() < MIN_SUSPICIOUS_SPACE {
            return false;
        }
        let Some(region) = self.targets.regions().next() else {
            return false;
        };
        let dimensions = f64::from(region.wildcard_dimensions(self.layout));
        if dimensions == 0.0 {
            return false;
        }
        let threshold = DENSITY_THRESHOLD / (128.0 / 10.0 - 1.0)
            * (self.layout.theoretical_dimensions() - dimensions)
            / dimensions;
        self.density() >= threshold
    }
}

impl SixTreeModel {
    pub(super) fn start_alias_round(&mut self) {
        let node = &self.nodes[self.current_batch[0]];
        let round = AliasProbeRound::sample(node, self.alias_probe_seed, &self.aliases);
        self.alias_probe_seed = self.alias_probe_seed.wrapping_add(0x9e3779b97f4a7c15);
        self.phase = SearchPhase::Detecting(round);
        self.generation = None;
    }

    pub(super) fn finish_alias_round(&mut self) {
        let SearchPhase::Detecting(round) =
            std::mem::replace(&mut self.phase, SearchPhase::Scanning)
        else {
            return;
        };
        let id = self.current_batch[0];
        if round.has_response() && !self.nodes[id].dimension_stack.is_empty() {
            self.nodes[id].expand();
            self.replace_descendants();
            self.start_alias_round();
            return;
        }
        self.generation = None;
        self.nodes[id].alias_checked = true;
        // Skipped probes provide no evidence that a region is unresponsive.
        if round.all_inactive() && self.nodes[id].targets.size_capped() > MAX_NORMAL_SPACE {
            for prefix in self.nodes[id]
                .targets
                .regions()
                .map(Region::covering_prefix)
            {
                if !self.detected_aliases.contains(&prefix) {
                    self.detected_aliases.push(prefix);
                }
            }
            let density = 1.0 / self.nodes[id].probe_count().max(1) as f64;
            let index = self
                .queue
                .iter()
                .position(|&other| self.nodes[other].density() <= density)
                .unwrap_or(self.queue.len());
            self.queue.insert(index, id);
            self.current_batch.clear();
            self.schedule_next_batch();
        }
    }
}

#[cfg(test)]
mod tests;
