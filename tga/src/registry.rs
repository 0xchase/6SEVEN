use crate::{
    Algorithm, AlgorithmId, AlgorithmSpec, Det, EntropyIp, SixForest, SixGen, SixGraph, SixProbe,
    SixTree, TgaError,
};
#[cfg(feature = "ml")]
use crate::{SixGan, SixGcvae, SixVecLm};
use serde::{Deserialize, Serialize};

macro_rules! define_algorithms {
    ($($(#[$meta:meta])* $variant:ident($algorithm:ty);)*) => {
        #[derive(clap::Subcommand, Serialize, Deserialize)]
        pub enum AlgorithmConfig {
            $($(#[$meta])* $variant($algorithm),)*
        }

        impl AlgorithmConfig {
            pub fn id(&self) -> AlgorithmId {
                let id = match self {
                    $( $(#[$meta])* Self::$variant(_) => <$algorithm as Algorithm>::ID,)*
                };
                AlgorithmId::new(id).expect("registered algorithm IDs must be valid")
            }

            pub fn description(&self) -> &'static str {
                match self {
                    $( $(#[$meta])* Self::$variant(_) => <$algorithm as Algorithm>::DESCRIPTION,)*
                }
            }

            pub fn spec(&self) -> Result<AlgorithmSpec, TgaError> {
                let config = match self {
                    $( $(#[$meta])* Self::$variant(config) => serde_json::to_value(config),)*
                }
                .map_err(|error| TgaError::Config(format!("encode algorithm config: {error}")))?;
                Ok(AlgorithmSpec {
                    algorithm: self.id(),
                    config,
                })
            }
        }

        pub fn builtin_registry() -> crate::Registry {
            let mut registry = crate::Registry::default();
            $( $(#[$meta])* registry.register_algorithm::<$algorithm>().expect("unique algorithm id");)*
            registry
        }

    };
}

define_algorithms! {
    EntropyIp(EntropyIp);
    SixGen(SixGen);
    SixForest(SixForest);
    SixGraph(SixGraph);
    SixTree(SixTree);
    Det(Det);
    #[cfg(feature = "ml")]
    SixGcvae(SixGcvae);
    #[cfg(feature = "ml")]
    SixVecLm(SixVecLm);
    SixProbe(SixProbe);
    #[cfg(feature = "ml")]
    SixGan(SixGan);
}
