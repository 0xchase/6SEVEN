#[cfg(any(test, not(feature = "gpu")))]
pub(crate) type CpuBackend = burn::backend::ndarray::NdArray<f32>;

/// The inner (non-autodiff) backend used for training.
#[cfg(feature = "gpu")]
pub(crate) type InnerTrainBackend = burn::backend::wgpu::Wgpu;
#[cfg(not(feature = "gpu"))]
pub(crate) type InnerTrainBackend = CpuBackend;

pub(crate) type TrainBackend = burn::backend::Autodiff<InnerTrainBackend>;

/// Backend used for inference.
#[cfg(feature = "gpu")]
pub(crate) type InferBackend = burn::backend::wgpu::Wgpu;
#[cfg(not(feature = "gpu"))]
pub(crate) type InferBackend = CpuBackend;

/// Returns the device to use for training.
pub(crate) fn train_device() -> <TrainBackend as burn::tensor::backend::Backend>::Device {
    #[cfg(feature = "gpu")]
    {
        tracing::info!(target: "ml", "Training with wgpu GPU backend");
        Default::default()
    }
    #[cfg(not(feature = "gpu"))]
    {
        Default::default()
    }
}

/// Returns the device to use for inference.
pub(crate) fn infer_device() -> <InferBackend as burn::tensor::backend::Backend>::Device {
    #[cfg(feature = "gpu")]
    {
        tracing::info!(target: "ml", "Inference with wgpu GPU backend");
        Default::default()
    }
    #[cfg(not(feature = "gpu"))]
    {
        Default::default()
    }
}

mod sixgan;
mod sixgcvae;
mod sixveclm;

pub use sixgan::{SixGan, SixGanClassification, SixGanModel};
pub use sixgcvae::preprocess::{
    ManualClassification as SixGcvaeManualClassification,
    bytes_to_nybble_sequence as sixgcvae_bytes_to_nybble_sequence,
    classify_manual as sixgcvae_classify_manual, flatten_ipv6_text as sixgcvae_flatten_ipv6_text,
    split_train_test_indices as sixgcvae_split_train_test_indices,
    to_training_rows as sixgcvae_to_training_rows,
};
pub use sixgcvae::{SixGcvae, SixGcvaeClassification, SixGcvaeModel};
pub use sixveclm::{SixVecLm, SixVecLmModel};

// Burn shares random state across model instances.
pub(crate) fn rng_guard() -> std::sync::MutexGuard<'static, ()> {
    static RNG: std::sync::Mutex<()> = std::sync::Mutex::new(());
    RNG.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
