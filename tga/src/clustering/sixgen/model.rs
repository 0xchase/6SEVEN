use super::{
    Address, SixGenRangeMode, TgaError,
    range::{AddressRange, AddressRangeIter},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) enum SixGenBlock {
    Range(AddressRange),
    Points(Vec<Address>),
}

impl SixGenBlock {
    pub(super) fn size(&self) -> usize {
        match self {
            Self::Range(range) => range.size(),
            Self::Points(points) => points.len(),
        }
    }
}

pub(super) fn total_block_size(blocks: &[SixGenBlock]) -> usize {
    blocks
        .iter()
        .fold(0usize, |total, block| total.saturating_add(block.size()))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SixGenModel {
    #[serde(skip)]
    generation: crate::cursor::GenerationCursor<SixGenCursor>,
    pub(super) blocks: Vec<SixGenBlock>,
    seed_count: usize,
    range_mode: SixGenRangeMode,
}

impl Default for SixGenModel {
    fn default() -> Self {
        Self {
            generation: Default::default(),
            blocks: Vec::new(),
            seed_count: 0,
            range_mode: SixGenRangeMode::Loose,
        }
    }
}

impl std::fmt::Display for SixGenModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let range_blocks = self
            .blocks
            .iter()
            .filter(|block| matches!(block, SixGenBlock::Range(_)))
            .count();
        let point_blocks = self.blocks.len().saturating_sub(range_blocks);
        let target_count = total_block_size(&self.blocks);
        write!(
            f,
            "6Gen model with {} targets across {} blocks ({} range, {} sampled), trained from {} seeds, {:?} mode",
            target_count,
            self.blocks.len(),
            range_blocks,
            point_blocks,
            self.seed_count,
            self.range_mode
        )
    }
}

impl SixGenModel {
    pub(super) fn from_blocks(
        blocks: Vec<SixGenBlock>,
        seed_count: usize,
        range_mode: SixGenRangeMode,
    ) -> Self {
        Self {
            generation: Default::default(),
            blocks,
            seed_count,
            range_mode,
        }
    }
}

impl crate::TargetModel for SixGenModel {
    fn generate(&mut self, output: &mut [crate::Address]) -> Result<crate::Generated, TgaError> {
        let cursor = self.generation.0.get_or_insert_with(SixGenCursor::default);
        crate::cursor::fill(
            std::iter::from_fn(|| cursor.next_from(&self.blocks)),
            output,
            crate::GenerationState::Exhausted,
        )
    }
}

// Generation borrows blocks and stores only cursor positions.
#[derive(Clone, Default)]
struct SixGenCursor {
    block: usize,
    point: usize,
    range: Option<AddressRangeIter>,
}

impl SixGenCursor {
    fn next_from(&mut self, blocks: &[SixGenBlock]) -> Option<Address> {
        loop {
            let address = match blocks.get(self.block)? {
                SixGenBlock::Range(range) => self.range.get_or_insert_with(|| range.iter()).next(),
                SixGenBlock::Points(points) => {
                    points.get(self.point).copied().inspect(|_| self.point += 1)
                }
            };
            if address.is_some() {
                return address;
            }
            self.block += 1;
            self.point = 0;
            self.range = None;
        }
    }
}
