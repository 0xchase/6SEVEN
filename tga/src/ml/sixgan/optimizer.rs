use burn::{
    module::{AutodiffModule, ModuleVisitor, Param},
    optim::GradientsParams,
    tensor::{
        Tensor,
        backend::{AutodiffBackend, Backend},
    },
};
use std::marker::PhantomData;
pub(super) fn clip_gradients_global_norm<M, B>(
    module: &M,
    mut grads: GradientsParams,
    max_norm: f32,
) -> Result<GradientsParams, String>
where
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    if max_norm <= 0.0 || grads.is_empty() {
        return Ok(grads);
    }

    let mut norm_visitor = GradientNormVisitor::<B> {
        grads: &grads,
        sum_sq: 0.0,
        _backend: PhantomData,
    };
    module.visit(&mut norm_visitor);

    let total_norm = norm_visitor.sum_sq.sqrt();
    if !total_norm.is_finite() {
        return Err("6GAN gradient norm is nonfinite".into());
    }
    if total_norm <= max_norm {
        return Ok(grads);
    }

    let scale = max_norm / total_norm;
    let mut scale_visitor = GradientScaleVisitor::<B> {
        grads: &mut grads,
        scale,
        _backend: PhantomData,
    };
    module.visit(&mut scale_visitor);
    Ok(grads)
}

struct GradientNormVisitor<'a, B: AutodiffBackend> {
    grads: &'a GradientsParams,
    sum_sq: f32,
    _backend: PhantomData<B>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for GradientNormVisitor<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if let Some(grad) = self.grads.get::<B::InnerBackend, D>(param.id) {
            self.sum_sq += scalar(grad.powi_scalar(2).sum());
        }
    }
}

struct GradientScaleVisitor<'a, B: AutodiffBackend> {
    grads: &'a mut GradientsParams,
    scale: f32,
    _backend: PhantomData<B>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for GradientScaleVisitor<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if let Some(grad) = self.grads.remove::<B::InnerBackend, D>(param.id) {
            self.grads.register(param.id, grad.mul_scalar(self.scale));
        }
    }
}

pub(super) fn scalar<B: Backend>(tensor: Tensor<B, 1>) -> f32 {
    tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("6GAN scalar tensor must contain f32 data")[0]
}

#[derive(Clone)]
pub(super) struct RmsProp;

#[derive(burn::record::Record, Clone)]
pub(super) struct RmsPropState<B: Backend, const D: usize> {
    mean_square: Tensor<B, D>,
}

impl<B: Backend> burn::optim::SimpleOptimizer<B> for RmsProp {
    type State<const D: usize> = RmsPropState<B, D>;

    fn step<const D: usize>(
        &self,
        rate: f64,
        parameter: Tensor<B, D>,
        gradient: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        // TensorFlow initializes the accumulator to one and adds epsilon inside the square root.
        let previous = state
            .map(|state| state.mean_square)
            .unwrap_or_else(|| gradient.ones_like());
        let mean_square = previous * 0.9 + gradient.clone().powi_scalar(2) * 0.1;
        let update = gradient * rate / (mean_square.clone() + 1e-10).sqrt();
        (parameter - update, Some(RmsPropState { mean_square }))
    }

    fn to_device<const D: usize>(state: Self::State<D>, device: &B::Device) -> Self::State<D> {
        RmsPropState {
            mean_square: state.mean_square.to_device(device),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::CpuBackend;
    use burn::optim::SimpleOptimizer;

    #[test]
    fn clipping_scales_all_gradient_components_by_the_global_norm() {
        use burn::{nn::LinearConfig, optim::GradientsParams};
        type B = burn::backend::Autodiff<CpuBackend>;
        let _guard = crate::ml::rng_guard();
        let device = Default::default();
        let module = LinearConfig::new(1, 2).with_bias(false).init::<B>(&device);
        let weight = module.weight.val();
        let loss = (weight * Tensor::from_data([[3.0, 4.0]], &device)).sum();
        let gradients = GradientsParams::from_grads(loss.backward(), &module);
        let gradients = clip_gradients_global_norm(&module, gradients, 1.0).unwrap();
        let values = gradients
            .get::<CpuBackend, 2>(module.weight.id)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!((values[0] - 0.6).abs() < 1e-6);
        assert!((values[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn rmsprop_matches_reference_scalar_updates() {
        let device = Default::default();
        let mut parameter = Tensor::<CpuBackend, 1>::from_data([1.0], &device);
        let mut state = None;
        let (mut expected, mut mean_square) = (1.0f32, 1.0f32);
        for gradient in [0.2f32, -0.3, 0.0, 1e-8] {
            mean_square = 0.9 * mean_square + 0.1 * gradient * gradient;
            expected -= 0.01 * gradient / (mean_square + 1e-10).sqrt();
            (parameter, state) = RmsProp.step(
                0.01,
                parameter,
                Tensor::from_data([gradient], &device),
                state,
            );
            assert!((parameter.clone().into_scalar() - expected).abs() < 1e-7);
        }
    }
}
