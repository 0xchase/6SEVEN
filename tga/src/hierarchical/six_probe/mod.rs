mod aliases;
mod config;
mod encoding;
mod forest;
mod generation;
mod pattern_mining;
mod python_random;
mod split;

pub use config::{DhcType, SixProbe, SixProbeMode, SplitArrayType, SplitOrder};

#[cfg(test)]
use crate::AddressStream;
use crate::{Algorithm, Observation, TgaError};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::io::{BufWriter, Write};

#[cfg(test)]
use encoding::pattern_string_to_subspace;
use encoding::{Subspace as PatternSubspace, addr_to_nibbles, subspace_to_pattern_string};
use forest::build_patterns;
#[cfg(test)]
use generation::SixProbePatternIter;
use generation::{SixProbeIter, SixProbePattern};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SixProbeModel {
    #[serde(skip)]
    pub(crate) generation: crate::cursor::GenerationCursor<SixProbeIter>,
    patterns: Vec<SixProbePattern>,
    beta: usize,
    tree_num: usize,
    mode: SixProbeMode,
    dhc_type: DhcType,
    split_array_type: SplitArrayType,
    random_seed: Option<u64>,
    split_order: SplitOrder,
    aliased_prefixes: Vec<crate::Ipv6Prefix>,
}

impl Default for SixProbeModel {
    fn default() -> Self {
        Self {
            generation: Default::default(),
            patterns: Vec::new(),
            beta: 0,
            tree_num: 0,
            mode: SixProbeMode::Forest,
            dhc_type: DhcType::LeftVdps,
            split_array_type: SplitArrayType::Random,
            random_seed: None,
            split_order: SplitOrder::Right,
            aliased_prefixes: Vec::new(),
        }
    }
}

fn write_export_patterns(
    path: &std::path::Path,
    patterns: &[SixProbePattern],
) -> Result<(), TgaError> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|err| {
            TgaError::Training(format!(
                "Failed creating export directory '{}': {}",
                parent.display(),
                err
            ))
        })?;
    }

    let write = || -> std::io::Result<()> {
        let mut output = BufWriter::new(fs::File::create(path)?);
        for pattern in patterns {
            writeln!(output, "{}", subspace_to_pattern_string(&pattern.subspace))?;
        }
        output.flush()
    };
    write().map_err(|err| {
        TgaError::Training(format!(
            "Failed writing six-probe patterns to '{}': {}",
            path.display(),
            err
        ))
    })
}

/// Keep first-seen forest patterns and preserve single-tree traversal order.
fn finalize_patterns(mode: SixProbeMode, patterns: Vec<PatternSubspace>) -> Vec<PatternSubspace> {
    match mode {
        SixProbeMode::SingleTree => patterns,
        SixProbeMode::Forest => {
            let mut seen = HashSet::with_capacity(patterns.len());
            let mut deduped = Vec::with_capacity(patterns.len());
            for pattern in patterns {
                if seen.insert(pattern) {
                    deduped.push(pattern);
                }
            }
            deduped
        }
    }
}

impl std::fmt::Display for SixProbeModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "6Probe model with {} patterns, beta: {}, tree-num: {}, mode: {:?}, dhc-type: {:?}, split-array-type: {:?}, random-seed: {:?}, split-order: {:?}",
            self.patterns.len(),
            self.beta,
            self.tree_num,
            self.mode,
            self.dhc_type,
            self.split_array_type,
            self.random_seed,
            self.split_order
        )
    }
}

impl Algorithm for SixProbe {
    const ID: &'static str = "6probe";
    const DESCRIPTION: &'static str = "6Probe implementation faithful to the released 6Probe codebase (6ASForest + pattern mining)";

    const MODEL_VERSION: u32 = 2;

    type Model = SixProbeModel;

    fn migrate_model(version: u32, bytes: &[u8]) -> Result<Self::Model, TgaError> {
        if version != 1 {
            return Err(TgaError::Model(format!(
                "unsupported 6probe model version {version}"
            )));
        }
        #[derive(Deserialize)]
        struct LegacyModel {
            patterns: Vec<SixProbePattern>,
            beta: usize,
            tree_num: usize,
            mode: SixProbeMode,
            dhc_type: DhcType,
            split_array_type: SplitArrayType,
            random_seed: Option<u64>,
            split_order: SplitOrder,
        }
        let old: LegacyModel = bincode::deserialize(bytes)
            .map_err(|error| TgaError::Model(format!("decode 6probe v1 model: {error}")))?;
        Ok(SixProbeModel {
            patterns: old.patterns,
            beta: old.beta,
            tree_num: old.tree_num,
            mode: old.mode,
            dhc_type: old.dhc_type,
            split_array_type: old.split_array_type,
            random_seed: old.random_seed,
            split_order: old.split_order,
            ..SixProbeModel::default()
        })
    }

    fn train(&self, observations: &[Observation]) -> Result<Self::Model, TgaError> {
        let aliased_prefixes = self
            .aliased_prefixes
            .as_deref()
            .map(aliases::load)
            .transpose()?
            .unwrap_or_default();
        let nibble_seeds: Vec<_> = observations
            .iter()
            .filter(|obs| obs.active)
            .map(|obs| addr_to_nibbles(&obs.address))
            .collect();
        if nibble_seeds.is_empty() {
            return Err(TgaError::Training("Training data cannot be empty".into()));
        }

        let pattern_subspaces = finalize_patterns(self.mode, build_patterns(&nibble_seeds, self));

        let stored_patterns = pattern_subspaces
            .into_iter()
            .map(SixProbePattern::from_subspace)
            .collect::<Result<Vec<_>, _>>()?;

        if let Some(export_path) = &self.export_patterns {
            write_export_patterns(export_path, &stored_patterns)?;
        }

        Ok(SixProbeModel {
            generation: Default::default(),
            patterns: stored_patterns,
            beta: self.beta,
            tree_num: self.tree_num,
            mode: self.mode,
            dhc_type: self.dhc_type,
            split_array_type: self.split_array_type,
            random_seed: self.random_seed,
            split_order: self.split_order,
            aliased_prefixes,
        })
    }
}

impl SixProbeModel {
    #[cfg(test)]
    pub(crate) fn stream(&self) -> Result<AddressStream, TgaError> {
        Ok(Box::new(SixProbeIter::new(
            self.patterns.clone(),
            &self.aliased_prefixes,
        )))
    }
}
impl crate::TargetModel for SixProbeModel {
    fn generate(&mut self, output: &mut [crate::Address]) -> Result<crate::Generated, TgaError> {
        if self.generation.0.is_none() {
            self.generation.0 = Some(SixProbeIter::new(
                self.patterns.clone(),
                &self.aliased_prefixes,
            ));
        }
        crate::cursor::fill(
            self.generation.0.as_mut().expect("initialized cursor"),
            output,
            crate::GenerationState::Exhausted,
        )
    }
}

#[cfg(test)]
mod tests;
