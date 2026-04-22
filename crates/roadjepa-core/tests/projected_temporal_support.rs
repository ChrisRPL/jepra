#[allow(dead_code)]
#[path = "../examples/support/projected_temporal.rs"]
mod projected_temporal;
#[allow(dead_code)]
#[path = "../examples/support/temporal_vision.rs"]
mod temporal_vision;

use projected_temporal::{
    combine_projection_grads, gaussian_moment_regularizer, gaussian_moment_regularizer_grad,
    projected_batch_losses, projected_step, projection_stats,
};
use roadjepa_core::{Linear, Predictor, ProjectedVisionJepa, Tensor};
use temporal_vision::{make_frozen_encoder, make_train_batch};

const PROJECTION_DIM: usize = 4;
const PROJECTOR_LR: f32 = 0.005;
const PREDICTOR_LR: f32 = 0.02;
const REGULARIZER_WEIGHT: f32 = 1e-4;
const TRAIN_BASE_SEED: u64 = 11_000;
const VALIDATION_BASE_SEED: u64 = 111_000;
const VALIDATION_BATCHES: usize = 8;

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

fn finite_difference_regularizer_grad(latents: &Tensor, index: usize, epsilon: f32) -> f32 {
    let mut plus = latents.clone();
    plus.data[index] += epsilon;

    let mut minus = latents.clone();
    minus.data[index] -= epsilon;

    (gaussian_moment_regularizer(&plus) - gaussian_moment_regularizer(&minus)) / (2.0 * epsilon)
}

fn projected_validation_losses(
    encoder: &roadjepa_core::EmbeddingEncoder,
    online_projector: &Linear,
    target_projector: &Linear,
    predictor: &Predictor,
) -> (f32, f32, f32) {
    let mut prediction_total = 0.0;
    let mut regularizer_total = 0.0;
    let mut total = 0.0;

    for batch_idx in 0..VALIDATION_BATCHES {
        let (x_t, x_t1) = make_train_batch(VALIDATION_BASE_SEED, batch_idx as u64);
        let (prediction_loss, regularizer_loss, total_loss) = projected_batch_losses(
            encoder,
            online_projector,
            target_projector,
            predictor,
            &x_t,
            &x_t1,
            REGULARIZER_WEIGHT,
        );
        prediction_total += prediction_loss;
        regularizer_total += regularizer_loss;
        total += total_loss;
    }

    let batches = VALIDATION_BATCHES as f32;
    (
        prediction_total / batches,
        regularizer_total / batches,
        total / batches,
    )
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
    let target_projector_weight_snapshot = target_projector.weight.clone();
    let target_projector_bias_snapshot = target_projector.bias.clone();
    let mut predictor = make_predictor();
    let (x_t, x_t1) = make_train_batch(11_000, 0);

    let initial_total = projected_batch_losses(
        &encoder,
        &online_projector,
        &target_projector,
        &predictor,
        &x_t,
        &x_t1,
        REGULARIZER_WEIGHT,
    )
    .2;

    projected_step(
        &encoder,
        &mut online_projector,
        &target_projector,
        &mut predictor,
        &x_t,
        &x_t1,
        REGULARIZER_WEIGHT,
        PREDICTOR_LR,
        PROJECTOR_LR,
    );

    assert_eq!(
        target_projector.weight, target_projector_weight_snapshot,
        "target projector weight mutated during projected step"
    );
    assert_eq!(
        target_projector.bias, target_projector_bias_snapshot,
        "target projector bias mutated during projected step"
    );

    let final_total = projected_batch_losses(
        &encoder,
        &online_projector,
        &target_projector,
        &predictor,
        &x_t,
        &x_t1,
        REGULARIZER_WEIGHT,
    )
    .2;

    assert!(
        final_total + 1e-6 < initial_total,
        "one projected step did not reduce total loss: {:.6} -> {:.6}",
        initial_total,
        final_total
    );
}

#[test]
fn projected_vision_jepa_step_reduces_total_loss_on_fixed_batch() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let target_projector_weight_snapshot = target_projector.weight.clone();
    let target_projector_bias_snapshot = target_projector.bias.clone();
    let predictor = make_predictor();
    let (x_t, x_t1) = make_train_batch(11_000, 0);

    let mut model = ProjectedVisionJepa::new(
        encoder.clone(),
        projector,
        target_projector.clone(),
        predictor,
    );
    let initial_total = projected_batch_losses(
        &encoder,
        &model.projector,
        &target_projector,
        &model.predictor,
        &x_t,
        &x_t1,
        REGULARIZER_WEIGHT,
    )
    .2;

    model.step(&x_t, &x_t1, REGULARIZER_WEIGHT, PREDICTOR_LR, PROJECTOR_LR);

    assert_eq!(
        model.target_projector.weight, target_projector_weight_snapshot,
        "target projector weight mutated during projected step"
    );
    assert_eq!(
        model.target_projector.bias, target_projector_bias_snapshot,
        "target projector bias mutated during projected step"
    );

    let final_total = projected_batch_losses(
        &encoder,
        &model.projector,
        &model.target_projector,
        &model.predictor,
        &x_t,
        &x_t1,
        REGULARIZER_WEIGHT,
    )
    .2;

    assert!(
        final_total + 1e-6 < initial_total,
        "one projected VisionJePA step did not reduce total loss: {:.6} -> {:.6}",
        initial_total,
        final_total
    );
}

#[test]
fn projected_training_steps_preserve_target_projector_after_multiple_batches() {
    let encoder = make_frozen_encoder();
    let mut online_projector = make_projector();
    let target_projector = online_projector.clone();
    let target_projector_weight_snapshot = target_projector.weight.clone();
    let target_projector_bias_snapshot = target_projector.bias.clone();
    let mut predictor = make_predictor();
    let (x_t_step0, x_t1_step0) = make_train_batch(11_000, 0);
    let (x_t_step1, x_t1_step1) = make_train_batch(11_000, 1);

    projected_step(
        &encoder,
        &mut online_projector,
        &target_projector,
        &mut predictor,
        &x_t_step0,
        &x_t1_step0,
        REGULARIZER_WEIGHT,
        PREDICTOR_LR,
        PROJECTOR_LR,
    );

    projected_step(
        &encoder,
        &mut online_projector,
        &target_projector,
        &mut predictor,
        &x_t_step1,
        &x_t1_step1,
        REGULARIZER_WEIGHT,
        PREDICTOR_LR,
        PROJECTOR_LR,
    );

    assert_eq!(
        target_projector.weight, target_projector_weight_snapshot,
        "target projector weight mutated during multi-step projected training"
    );
    assert_eq!(
        target_projector.bias, target_projector_bias_snapshot,
        "target projector bias mutated during multi-step projected training"
    );
}

#[test]
fn projected_random_temporal_loop_reduces_train_and_validation_loss() {
    let encoder = make_frozen_encoder();
    let mut online_projector = make_projector();
    let target_projector = online_projector.clone();
    let mut predictor = make_predictor();
    let (probe_t, probe_t1) = make_train_batch(TRAIN_BASE_SEED, 0);
    let steps = 120;
    let initial_losses = projected_batch_losses(
        &encoder,
        &online_projector,
        &target_projector,
        &predictor,
        &probe_t,
        &probe_t1,
        REGULARIZER_WEIGHT,
    );
    let initial_validation =
        projected_validation_losses(&encoder, &online_projector, &target_projector, &predictor);

    for step in 1..=steps {
        let (x_t, x_t1) = make_train_batch(TRAIN_BASE_SEED, step as u64);
        projected_step(
            &encoder,
            &mut online_projector,
            &target_projector,
            &mut predictor,
            &x_t,
            &x_t1,
            REGULARIZER_WEIGHT,
            PREDICTOR_LR,
            PROJECTOR_LR,
        );

        if step == 1 || step == steps {
            println!(
                "projected temporal step {:03} | train total {:.6} | val total {:.6}",
                step,
                projected_batch_losses(
                    &encoder,
                    &online_projector,
                    &target_projector,
                    &predictor,
                    &probe_t,
                    &probe_t1,
                    REGULARIZER_WEIGHT,
                )
                .2,
                projected_validation_losses(
                    &encoder,
                    &online_projector,
                    &target_projector,
                    &predictor,
                )
                .2
            );
        }
    }

    let final_losses = projected_batch_losses(
        &encoder,
        &online_projector,
        &target_projector,
        &predictor,
        &probe_t,
        &probe_t1,
        REGULARIZER_WEIGHT,
    );
    let final_validation =
        projected_validation_losses(&encoder, &online_projector, &target_projector, &predictor);

    assert!(
        final_losses.2 < initial_losses.2,
        "train total loss did not improve: {:.6} -> {:.6}",
        initial_losses.2,
        final_losses.2
    );
    assert!(
        final_validation.2 < initial_validation.2,
        "validation total loss did not improve: {:.6} -> {:.6}",
        initial_validation.2,
        final_validation.2
    );
    assert!(
        final_losses.2 < 0.18,
        "train total loss too high after {} steps: {:.6}",
        steps,
        final_losses.2
    );
    assert!(
        final_validation.2 < 0.18,
        "validation total loss too high after {} steps: {:.6}",
        steps,
        final_validation.2
    );
}
