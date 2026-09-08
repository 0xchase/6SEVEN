//! Keras 2.2.4 Adam adds epsilon before bias correction.

use burn::{
    optim::SimpleOptimizer,
    record::Record,
    tensor::{Tensor, backend::Backend},
};

use super::config::{ADAM_BETA_1, ADAM_BETA_2, ADAM_EPSILON};

#[derive(Clone)]
pub(super) struct KerasAdam;

#[derive(Record, Clone)]
pub(super) struct KerasAdamState<B: Backend, const D: usize> {
    step: usize,
    first: Tensor<B, D>,
    second: Tensor<B, D>,
}

impl<B: Backend> SimpleOptimizer<B> for KerasAdam {
    type State<const D: usize> = KerasAdamState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: f64,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let state = state.unwrap_or_else(|| KerasAdamState {
            step: 0,
            first: grad.zeros_like(),
            second: grad.zeros_like(),
        });
        let step = state.step + 1;
        let first = state.first * ADAM_BETA_1 + grad.clone() * (1.0 - ADAM_BETA_1);
        let second = state.second * ADAM_BETA_2 + grad.powi_scalar(2) * (1.0 - ADAM_BETA_2);
        let correction =
            (1.0 - ADAM_BETA_2.powf(step as f32)).sqrt() / (1.0 - ADAM_BETA_1.powf(step as f32));
        let update = first.clone().mul_scalar(lr * f64::from(correction))
            / (second.clone().sqrt() + ADAM_EPSILON);
        (
            tensor - update,
            Some(KerasAdamState {
                step,
                first,
                second,
            }),
        )
    }

    fn to_device<const D: usize>(state: Self::State<D>, device: &B::Device) -> Self::State<D> {
        KerasAdamState {
            step: state.step,
            first: state.first.to_device(device),
            second: state.second.to_device(device),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::CpuBackend;

    #[test]
    fn updates_match_keras_scalar_equations_including_small_gradients() {
        let device = Default::default();
        let mut parameter = Tensor::<CpuBackend, 1>::zeros([1], &device);
        let mut state = None;
        let (mut expected, mut first, mut second) = (0.0f32, 0.0f32, 0.0f32);
        // Small gradients expose incorrect epsilon placement.
        for (index, gradient) in [1e-8f32, 0.2, 0.0, -0.3].into_iter().enumerate() {
            first = 0.9 * first + (1.0 - 0.9f32) * gradient;
            second = 0.999 * second + (1.0 - 0.999f32) * gradient * gradient;
            let t = (index + 1) as f32;
            let rate = 0.001 * (1.0 - 0.999f32.powf(t)).sqrt() / (1.0 - 0.9f32.powf(t));
            expected -= rate * first / (second.sqrt() + 1e-7);
            (parameter, state) = KerasAdam.step(
                0.001,
                parameter,
                Tensor::from_data([gradient], &device),
                state,
            );
            let actual = parameter.clone().into_scalar();
            assert!(
                (actual - expected).abs() < 1e-9,
                "step {t}: {actual} != {expected}"
            );
        }
    }
}
