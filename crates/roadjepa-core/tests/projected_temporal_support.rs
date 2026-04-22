#[allow(dead_code)]
#[path = "../examples/support/projected_temporal.rs"]
mod projected_temporal;
#[allow(dead_code)]
#[path = "../examples/support/temporal_vision.rs"]
mod temporal_vision;

use projected_temporal::{
    combine_projection_grads, gaussian_moment_regularizer, gaussian_moment_regularizer_grad,
    projection_stats,
};
use roadjepa_core::{mse_loss, mse_loss_grad, EmbeddingEncoder, Linear, Predictor, Tensor};
use temporal_vision::{make_frozen_encoder, make_train_batch};

const PROJECTION_DIM: usize = 4;
const PROJECTOR_LR: f32 = 0.005;
const PREDICTOR_LR: f32 = 0.02;
const REGULARIZER_WEIGHT: f32 = 1e-4;

fn make_projector() -> Linear {
    Linear::new(
        Tensor::new(
            vec![
                1.0, 0.0, 1.0, 0.0, //
                0.0, 1.0, 0.0, 1.0, //
                0.0, 0.0, 1.0, 1.0,
            ],
            vec![3, PROJECTION_DIM],
        ),
        Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![PROJECTION_DIM]),
    )
}

fn make_predictor() -> Predictor {
    Predictor::new(
        Linear::randn(PROJECTION_DIM, 8, 0.1, 21_000),
        Linear::randn(8, PROJECTION_DIM, 0.1, 21_001),
    )
}

fn projected_target(
    encoder: &EmbeddingEncoder,
    target_projector: &Linear,
    x_t1: &Tensor,
) -> Tensor {
    let z_t1 = encoder.forward(x_t1);
    target_projector.forward(&z_t1)
}

fn projected_batch_losses(
    encoder: &EmbeddingEncoder,
    online_projector: &Linear,
    target_projector: &Linear,
    predictor: &Predictor,
    x_t: &Tensor,
    x_t1: &Tensor,
) -> (f32, f32, f32) {
    let z_t = encoder.forward(x_t);
    let projection_t = online_projector.forward(&z_t);
    let pred = predictor.forward(&projection_t);
    let target = projected_target(encoder, target_projector, x_t1);
    let prediction_loss = mse_loss(&pred, &target);
    let regularizer_loss = gaussian_moment_regularizer(&projection_t);
    let total_loss = prediction_loss + REGULARIZER_WEIGHT * regularizer_loss;

    (prediction_loss, regularizer_loss, total_loss)
}

fn projected_step(
    encoder: &EmbeddingEncoder,
    online_projector: &mut Linear,
    target_projector: &Linear,
    predictor: &mut Predictor,
    x_t: &Tensor,
    x_t1: &Tensor,
) {
    let z_t = encoder.forward(x_t);
    let projection_t = online_projector.forward(&z_t);
    let target = projected_target(encoder, target_projector, x_t1);
    let pred = predictor.forward(&projection_t);
    let prediction_grad = mse_loss_grad(&pred, &target);
    let reg_grad = gaussian_moment_regularizer_grad(&projection_t);
    let pred_grads = predictor.backward(&projection_t, &prediction_grad);
    let projection_grad =
        combine_projection_grads(&pred_grads.grad_input, &reg_grad, REGULARIZER_WEIGHT);

    let projector_grads = online_projector.backward(&z_t, &projection_grad);

    predictor.sgd_step(&pred_grads, PREDICTOR_LR);
    online_projector.sgd_step(&projector_grads, PROJECTOR_LR);
}

fn finite_difference_regularizer_grad(latents: &Tensor, index: usize, epsilon: f32) -> f32 {
    let mut plus = latents.clone();
    plus.data[index] += epsilon;

    let mut minus = latents.clone();
    minus.data[index] -= epsilon;

    (gaussian_moment_regularizer(&plus) - gaussian_moment_regularizer(&minus)) / (2.0 * epsilon)
}

#[test]
fn gaussian_moment_regularizer_is_zero_for_zero_mean_unit_variance_latents() {
    let latents = Tensor::new(vec![-1.0, 1.0, 1.0, -1.0], vec![2, 2]);

    assert_eq!(gaussian_moment_regularizer(&latents), 0.0);
}

#[test]
fn gaussian_moment_regularizer_grad_matches_finite_difference() {
    let latents = Tensor::new(vec![-0.3, 0.7, 1.2, -1.1, 0.4, 0.2], vec![3, 2]);
    let analytic = gaussian_moment_regularizer_grad(&latents);
    let epsilon = 1e-3;

    for index in 0..latents.data.len() {
        let numerical = finite_difference_regularizer_grad(&latents, index, epsilon);

        assert!(
            (analytic.data[index] - numerical).abs() < 1e-3,
            "grad mismatch at flat index {}: analytic {:.6} vs numerical {:.6}",
            index,
            analytic.data[index],
            numerical
        );
    }
}

#[test]
fn projection_stats_report_feature_mean_abs_and_variance_mean() {
    let latents = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let (mean_abs, variance_mean) = projection_stats(&latents);

    assert!((mean_abs - 2.5).abs() < 1e-6);
    assert!((variance_mean - 1.0).abs() < 1e-6);
}

#[test]
fn combine_projection_grads_applies_regularizer_weight() {
    let prediction_grad = Tensor::new(vec![1.0, -2.0, 0.5, 3.0], vec![2, 2]);
    let regularizer_grad = Tensor::new(vec![0.2, 0.4, -1.0, 2.0], vec![2, 2]);
    let combined = combine_projection_grads(&prediction_grad, &regularizer_grad, 0.5);

    assert_eq!(combined, Tensor::new(vec![1.1, -1.8, 0.0, 4.0], vec![2, 2]));
}

#[test]
fn projected_training_step_reduces_total_loss_on_fixed_batch() {
    let encoder = make_frozen_encoder();
    let mut online_projector = make_projector();
    let target_projector = online_projector.clone();
    let mut predictor = make_predictor();
    let (x_t, x_t1) = make_train_batch(11_000, 0);

    let initial_total = projected_batch_losses(
        &encoder,
        &online_projector,
        &target_projector,
        &predictor,
        &x_t,
        &x_t1,
    )
    .2;

    projected_step(
        &encoder,
        &mut online_projector,
        &target_projector,
        &mut predictor,
        &x_t,
        &x_t1,
    );

    let final_total = projected_batch_losses(
        &encoder,
        &online_projector,
        &target_projector,
        &predictor,
        &x_t,
        &x_t1,
    )
    .2;

    assert!(
        final_total + 1e-6 < initial_total,
        "one projected step did not reduce total loss: {:.6} -> {:.6}",
        initial_total,
        final_total
    );
}
