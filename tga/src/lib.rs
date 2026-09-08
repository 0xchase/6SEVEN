//! Target generation algorithms used to construct IPv6 scan hitlists.

mod address;
mod cluster;
mod clustering;
mod hierarchical;
mod index;
#[cfg(feature = "ml")]
mod ml;
mod other;
mod pattern;
mod registry;
mod subspace;
#[allow(unused_imports)]
#[cfg(feature = "ml")]
use bincode_v2 as _;

pub use address::*;
pub use sixseven_core::*;
mod cursor;

#[cfg(test)]
type AddressStream = Box<dyn Iterator<Item = Result<Address, TgaError>> + Send>;

pub use cluster::*;
pub use clustering::{
    SixGen, SixGenModel, SixGenRangeMode, SixGraph, SixGraphModel,
    bytes_to_nibbles as clustering_bytes_to_nibbles,
    nibbles_to_bytes as clustering_nibbles_to_bytes,
};
pub use hierarchical::{
    Det, DetModel, DhcType, SixForest, SixForestModel, SixProbe, SixProbeMode, SixProbeModel,
    SixTree, SixTreeModel, SplitArrayType, SplitOrder,
};
pub use index::MAX_INDEX_DOMAIN;
#[cfg(feature = "ml")]
pub use ml::{
    SixGan, SixGanClassification, SixGanModel, SixGcvae, SixGcvaeClassification,
    SixGcvaeManualClassification, SixGcvaeModel, SixVecLm, SixVecLmModel,
    sixgcvae_bytes_to_nybble_sequence, sixgcvae_classify_manual, sixgcvae_flatten_ipv6_text,
    sixgcvae_split_train_test_indices, sixgcvae_to_training_rows,
};
pub use other::{
    EntropyIp, EntropyIpConditionedSampler, EntropyIpModel, EntropyIpSampleBatch, EntropyIpSegment,
    EntropyIpState,
};
pub use pattern::*;
pub use registry::{AlgorithmConfig, builtin_registry};
pub use subspace::{NibbleSet, Subspace};
