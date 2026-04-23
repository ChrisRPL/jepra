#[allow(dead_code)]
#[path = "../examples/support/projected_temporal.rs"]
mod projected_temporal;
#[allow(dead_code)]
#[path = "../examples/support/temporal_vision.rs"]
mod temporal_vision;

use jepra_core::{Linear, Predictor, ProjectedVisionJepa, Tensor};
use projected_temporal::{
    PROJECTED_TRAIN_LOSS_MAX_REDUCTION_RATIO, PROJECTED_VALIDATION_BASE_SEED,
    PROJECTED_VALIDATION_BATCHES, PROJECTED_VALIDATION_LOSS_MAX_REDUCTION_RATIO,
    combine_projection_grads, gaussian_moment_regularizer, gaussian_moment_regularizer_grad,
    projected_batch_losses, projected_step, projected_validation_batch_losses,
    projected_validation_batch_losses_from_base_seed, projection_stats,
};
use temporal_vision::{
    BATCH_SIZE, assert_seed_range_has_both_motion_modes,
    assert_seed_range_has_single_and_double_square_batch_examples,
    assert_temporal_experiment_improved, make_frozen_encoder, make_temporal_batch,
    make_train_batch, CompactEncoderMode, TemporalRunConfig,
};

const PROJECTION_DIM: usize = 4;
const PROJECTOR_LR: f32 = 0.005;
const PREDICTOR_LR: f32 = 0.02;
const REGULARIZER_WEIGHT: f32 = 1e-4;
const TRAIN_BASE_SEED: u64 = 11_000;

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

fn make_predictor_with_seed(seed_offset: u64) -> Predictor {
    Predictor::new(
        Linear::randn(PROJECTION_DIM, 8, 0.1, 22_000 + seed_offset),
        Linear::randn(8, PROJECTION_DIM, 0.1, 22_001 + seed_offset),
    )
}

fn projected_short_run_convergence(train_base_seed: u64, predictor_seed: u64) -> (f32, f32) {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let mut model = ProjectedVisionJepa::new(
        encoder,
        projector,
        target_projector,
        make_predictor_with_seed(predictor_seed),
    );
    let steps = 60;

    let (probe_t, probe_t1) = make_train_batch(train_base_seed, 0);
    let initial_train_loss = model.losses(&probe_t, &probe_t1, REGULARIZER_WEIGHT).2;
    let initial_val_loss = projected_validation_losses_model(&model).2;

    for step in 1..=steps {
        let (x_t, x_t1) = make_train_batch(train_base_seed, step as u64);
        model.step(&x_t, &x_t1, REGULARIZER_WEIGHT, PREDICTOR_LR, PROJECTOR_LR);
    }

    let final_train_loss = model.losses(&probe_t, &probe_t1, REGULARIZER_WEIGHT).2;
    let final_val_loss = projected_validation_losses_model(&model).2;

    assert!(
        final_train_loss < initial_train_loss * PROJECTED_TRAIN_LOSS_MAX_REDUCTION_RATIO,
        "short projected run did not shrink train total loss enough: {:.6} -> {:.6}",
        initial_train_loss,
        final_train_loss
    );
    assert!(
        final_val_loss < initial_val_loss * PROJECTED_VALIDATION_LOSS_MAX_REDUCTION_RATIO,
        "short projected run did not shrink validation total loss enough: {:.6} -> {:.6}",
        initial_val_loss,
        final_val_loss
    );

    (
        final_train_loss / initial_train_loss,
        final_val_loss / initial_val_loss,
    )
}

#[test]
fn projected_short_runs_have_stable_convergence_ratios() {
    let train_base_seed = 11_000;
    let runs = [22_010u64, 22_011u64, 22_012u64];
    let mut train_ratios = Vec::<f32>::new();
    let mut val_ratios = Vec::<f32>::new();

    for seed in runs {
        let (train_ratio, val_ratio) = projected_short_run_convergence(train_base_seed, seed);
        train_ratios.push(train_ratio);
        val_ratios.push(val_ratio);
    }

    let train_min = train_ratios.iter().copied().fold(f32::INFINITY, f32::min);
    let train_max = train_ratios
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let val_min = val_ratios.iter().copied().fold(f32::INFINITY, f32::min);
    let val_max = val_ratios.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    assert!(
        train_max - train_min < 0.20,
        "projected train convergence spread too large: [{:.6}, {:.6}]",
        train_min,
        train_max
    );
    assert!(
        val_max - val_min < 0.20,
        "projected validation convergence spread too large: [{:.6}, {:.6}]",
        val_min,
        val_max
    );
}

fn finite_difference_regularizer_grad(latents: &Tensor, index: usize, epsilon: f32) -> f32 {
    let mut plus = latents.clone();
    plus.data[index] += epsilon;

    let mut minus = latents.clone();
    minus.data[index] -= epsilon;

    (gaussian_moment_regularizer(&plus) - gaussian_moment_regularizer(&minus)) / (2.0 * epsilon)
}

fn projected_validation_losses_model(model: &ProjectedVisionJepa) -> (f32, f32, f32) {
    projected_validation_batch_losses_from_base_seed(
        model,
        REGULARIZER_WEIGHT,
        PROJECTED_VALIDATION_BASE_SEED,
        PROJECTED_VALIDATION_BATCHES,
    )
}

fn projected_validation_losses_projection_support(model: &ProjectedVisionJepa) -> (f32, f32, f32) {
    projected_validation_batch_losses(
        &model.encoder,
        &model.projector,
        &model.target_projector,
        &model.predictor,
        REGULARIZER_WEIGHT,
        PROJECTED_VALIDATION_BASE_SEED,
        PROJECTED_VALIDATION_BATCHES,
        |seed| make_temporal_batch(BATCH_SIZE, seed),
    )
}

#[test]
#[should_panic(expected = "validation_batches must be greater than 0")]
fn projected_validation_batch_losses_zero_batch_count_panics() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let model = ProjectedVisionJepa::new(encoder, projector, target_projector, make_predictor());

    let _ = projected_validation_batch_losses(
        &model.encoder,
        &model.projector,
        &model.target_projector,
        &model.predictor,
        REGULARIZER_WEIGHT,
        PROJECTED_VALIDATION_BASE_SEED,
        0,
        |seed| make_train_batch(seed, 0),
    );
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
fn projected_temporal_batch_contains_expected_square_counts_and_decays_mass() {
    assert_seed_range_has_single_and_double_square_batch_examples(128, |seed| {
        make_train_batch(seed, 0)
    });
}

#[test]
fn projected_generator_exposes_both_motion_modes_across_seed_range() {
    assert_seed_range_has_both_motion_modes(64, |seed| make_train_batch(seed, 0));
}

#[test]
fn projected_training_step_reduces_total_loss_on_fixed_batch() {
    let encoder = make_frozen_encoder();
    let online_projector = make_projector();
    let target_projector = online_projector.clone();
    let target_projector_weight_snapshot = target_projector.weight.clone();
    let target_projector_bias_snapshot = target_projector.bias.clone();
    let predictor = make_predictor();
    let mut model = ProjectedVisionJepa::new(
        encoder.clone(),
        online_projector,
        target_projector.clone(),
        predictor,
    );
    let (x_t, x_t1) = make_train_batch(11_000, 0);

    let initial_total = projected_batch_losses(
        &encoder,
        &model.projector,
        &model.target_projector,
        &model.predictor,
        &x_t,
        &x_t1,
        REGULARIZER_WEIGHT,
    )
    .2;

    projected_step(
        &mut model,
        &x_t,
        &x_t1,
        REGULARIZER_WEIGHT,
        PREDICTOR_LR,
        PROJECTOR_LR,
    );

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
        "one projected step did not reduce total loss: {:.6} -> {:.6}",
        initial_total,
        final_total
    );
}

#[test]
fn projected_step_updates_target_projector_when_momentum_is_enabled() {
    let encoder = make_frozen_encoder();
    let online_projector = make_projector();
    let target_projector = make_projector();
    let predictor = make_predictor();
    let mut model =
        ProjectedVisionJepa::new(encoder, online_projector, target_projector, predictor)
            .with_target_projection_momentum(0.5);
    let (x_t, x_t1) = make_train_batch(11_000, 0);

    assert_eq!(
        model.target_projection_drift(),
        0.0,
        "target_projection_drift should start at zero when target mirrors projector"
    );

    let initial_target_weight = model.target_projector.weight.clone();
    let initial_target_bias = model.target_projector.bias.clone();

    model.step(&x_t, &x_t1, REGULARIZER_WEIGHT, PREDICTOR_LR, PROJECTOR_LR);

    assert_ne!(model.target_projector.weight, initial_target_weight);
    assert_ne!(model.target_projector.bias, initial_target_bias);
    let momentum = model.target_projection_momentum();
    let one_minus_momentum = 1.0 - momentum;
    for i in 0..model.projector.weight.len() {
        let expected = momentum * initial_target_weight.data[i]
            + one_minus_momentum * model.projector.weight.data[i];
        let actual = model.target_projector.weight.data[i];
        assert!(
            (expected - actual).abs() < 1e-6,
            "target projector weight did not track EMA at index {}: expected {:.6}, got {:.6}",
            i,
            expected,
            actual
        );
    }

    for i in 0..model.projector.bias.len() {
        let expected = momentum * initial_target_bias.data[i]
            + one_minus_momentum * model.projector.bias.data[i];
        let actual = model.target_projector.bias.data[i];
        assert!(
            (expected - actual).abs() < 1e-6,
            "target projector bias did not track EMA at index {}: expected {:.6}, got {:.6}",
            i,
            expected,
            actual
        );
    }

    let mut expected_drift = 0.0f32;
    let mut parameter_count = 0usize;

    for (i, initial_target_value) in initial_target_weight.data.iter().enumerate() {
        expected_drift += momentum * (model.projector.weight.data[i] - initial_target_value).abs();
        parameter_count += 1;
    }

    for (i, initial_target_value) in initial_target_bias.data.iter().enumerate() {
        expected_drift += momentum * (model.projector.bias.data[i] - initial_target_value).abs();
        parameter_count += 1;
    }

    let expected_drift = if parameter_count == 0 {
        0.0
    } else {
        expected_drift / parameter_count as f32
    };

    assert!(
        (model.target_projection_drift() - expected_drift).abs() < 1e-6,
        "target_projection_drift should match EMA lag metric: {:.6} vs {:.6}",
        model.target_projection_drift(),
        expected_drift
    );
}

#[test]
fn projected_target_projector_warmup_transition_is_deterministic() {
    let config = TemporalRunConfig {
        train_base_seed: 11_000u64,
        total_steps: 2,
        log_every: 1,
        encoder_learning_rate: 0.0,
        compact_encoder_mode: CompactEncoderMode::Disabled,
        target_projection_momentum: 0.0,
        target_projection_momentum_start: 1.0,
        target_projection_momentum_end: 0.0,
        target_projection_momentum_warmup_steps: 2,
    };

    assert!(
        (config.target_projection_momentum_at_step(1) - 0.5).abs() < 1e-6,
        "warmup should start at 0.5 momentum for step 1 (start=1.0, end=0.0, warmup=2)"
    );
    assert!(
        (config.target_projection_momentum_at_step(2) - 0.0).abs() < 1e-6,
        "warmup should finish at 0.0 momentum for step 2 (end after warmup)"
    );

    let mut model = ProjectedVisionJepa::new(
        make_frozen_encoder(),
        make_projector(),
        make_projector(),
        make_predictor(),
    );

    let (x_t_1, x_t1_1) = make_train_batch(config.train_base_seed, 1);
    let (x_t_2, x_t1_2) = make_train_batch(config.train_base_seed, 2);
    let initial_target = model.target_projector.clone();

    model.set_target_projection_momentum(config.target_projection_momentum_at_step(1));
    model.step_with_trainable_encoder(
        &x_t_1,
        &x_t1_1,
        REGULARIZER_WEIGHT,
        PREDICTOR_LR,
        PROJECTOR_LR,
        config.encoder_learning_rate,
    );
    let step1_projector = model.projector.clone();
    let step1_target = model.target_projector.clone();

    assert_ne!(
        step1_target.weight, initial_target.weight,
        "step 1 should move target projector during warmup"
    );
    for i in 0..step1_target.weight.len() {
        let expected = 0.5 * initial_target.weight.data[i] + 0.5 * step1_projector.weight.data[i];
        assert!(
            (expected - step1_target.weight.data[i]).abs() < 1e-6,
            "step 1 target weight EMA not aligned at index {}: expected {:.6}, got {:.6}",
            i,
            expected,
            step1_target.weight.data[i]
        );
    }

    model.set_target_projection_momentum(config.target_projection_momentum_at_step(2));
    model.step_with_trainable_encoder(
        &x_t_2,
        &x_t1_2,
        REGULARIZER_WEIGHT,
        PREDICTOR_LR,
        PROJECTOR_LR,
        config.encoder_learning_rate,
    );

    assert_eq!(
        model.target_projector.weight, model.projector.weight,
        "step 2 should hard-copy target projector once warmup reaches momentum 0.0"
    );
    assert_eq!(
        model.target_projector.bias, model.projector.bias,
        "step 2 should hard-copy target bias once warmup reaches momentum 0.0"
    );

    assert!(
        model.target_projection_momentum() < 1e-6,
        "step 2 momentum must be near zero after warmup transition"
    );
}

#[test]
fn projected_target_projector_ema_edges_with_trainable_encoder_lr() {
    let encoder = make_frozen_encoder();
    let (x_t, x_t1) = make_train_batch(11_100, 3);
    let projector = make_projector();
    let predictor = make_predictor();
    let trainable_lr = 0.004;

    let mut zero_momentum_model = ProjectedVisionJepa::new(
        encoder.clone(),
        projector.clone(),
        projector.clone(),
        predictor.clone(),
    )
    .with_target_projection_momentum(0.0);

    let zero_initial_projector = zero_momentum_model.projector.clone();
    let zero_initial_target_projector = zero_momentum_model.target_projector.clone();
    let zero_initial_encoder = zero_momentum_model.encoder.clone();

    zero_momentum_model.step_with_trainable_encoder(
        &x_t,
        &x_t1,
        REGULARIZER_WEIGHT,
        PREDICTOR_LR,
        PROJECTOR_LR,
        trainable_lr,
    );

    assert_ne!(
        zero_momentum_model.projector, zero_initial_projector,
        "projector should change after optimizer step at momentum 0.0"
    );
    assert_eq!(
        zero_momentum_model.target_projector.weight, zero_momentum_model.projector.weight,
        "momentum 0.0 should hard-copy target projector weight"
    );
    assert_eq!(
        zero_momentum_model.target_projector.bias, zero_momentum_model.projector.bias,
        "momentum 0.0 should hard-copy target projector bias"
    );
    assert_ne!(
        zero_momentum_model.encoder, zero_initial_encoder,
        "encoder should update when encoder_lr is non-zero"
    );

    let mut full_momentum_model =
        ProjectedVisionJepa::new(encoder, projector, zero_initial_target_projector, predictor)
            .with_target_projection_momentum(1.0);

    let one_initial_projector = full_momentum_model.projector.clone();
    let one_initial_target_projector = full_momentum_model.target_projector.clone();

    full_momentum_model.step_with_trainable_encoder(
        &x_t,
        &x_t1,
        REGULARIZER_WEIGHT,
        PREDICTOR_LR,
        PROJECTOR_LR,
        trainable_lr,
    );

    assert_ne!(
        full_momentum_model.projector, one_initial_projector,
        "projector should change when running trainable projected step at momentum 1.0"
    );
    assert_eq!(
        full_momentum_model.target_projector, one_initial_target_projector,
        "momentum 1.0 should keep target projector frozen"
    );
}

#[test]
#[should_panic(expected = "target projection momentum must be in [0.0, 1.0]")]
fn projected_model_rejects_invalid_target_projection_momentum() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = make_projector();
    let predictor = make_predictor();
    let _ = ProjectedVisionJepa::new(encoder, projector, target_projector, predictor)
        .with_target_projection_momentum(-0.25);
}

#[test]
fn projected_step_with_trainable_encoder_updates_encoder_parameters() {
    let encoder = make_frozen_encoder();
    let online_projector = make_projector();
    let target_projector = make_projector();
    let predictor = make_predictor();
    let mut model =
        ProjectedVisionJepa::new(encoder, online_projector, target_projector, predictor);
    let (x_t, x_t1) = make_train_batch(11_000, 0);

    let initial_encoder = model.encoder.clone();
    let initial_projector = model.projector.clone();
    let initial_predictor = model.predictor.clone();
    let initial_total = model.losses(&x_t, &x_t1, REGULARIZER_WEIGHT).2;

    let encoder_lr = 0.004;
    let _ = model.step_with_trainable_encoder(
        &x_t,
        &x_t1,
        REGULARIZER_WEIGHT,
        PREDICTOR_LR,
        PROJECTOR_LR,
        encoder_lr,
    );

    assert_ne!(
        model.encoder, initial_encoder,
        "encoder did not change with trainable projected step"
    );
    assert_ne!(
        model.projector, initial_projector,
        "projector did not change with projected step"
    );
    assert_ne!(
        model.predictor, initial_predictor,
        "predictor did not change with projected step"
    );

    let final_total = model.losses(&x_t, &x_t1, REGULARIZER_WEIGHT).2;
    assert!(
        final_total + 1e-6 < initial_total,
        "projected trainable-step did not reduce total loss: {:.6} -> {:.6}",
        initial_total,
        final_total
    );
}

#[test]
fn projected_step_with_zero_encoder_lr_matches_frozen_projector_step() {
    let encoder = make_frozen_encoder();
    let online_projector = make_projector();
    let target_projector = make_projector();
    let predictor = make_predictor();
    let mut step_trainable = ProjectedVisionJepa::new(
        encoder.clone(),
        online_projector.clone(),
        target_projector.clone(),
        predictor.clone(),
    );
    let mut step_frozen =
        ProjectedVisionJepa::new(encoder, online_projector, target_projector, predictor);
    let (x_t, x_t1) = make_train_batch(11_000, 0);

    let trainable_losses = step_trainable.step_with_trainable_encoder(
        &x_t,
        &x_t1,
        REGULARIZER_WEIGHT,
        PREDICTOR_LR,
        PROJECTOR_LR,
        0.0,
    );
    let frozen_losses =
        step_frozen.step(&x_t, &x_t1, REGULARIZER_WEIGHT, PREDICTOR_LR, PROJECTOR_LR);

    assert!(
        (trainable_losses.0 - frozen_losses.0).abs() < 1e-6,
        "trainable projected step prediction loss diverged from frozen step: {:.6} vs {:.6}",
        trainable_losses.0,
        frozen_losses.0
    );
    assert!(
        (trainable_losses.1 - frozen_losses.1).abs() < 1e-6,
        "trainable projected step regularizer loss diverged from frozen step: {:.6} vs {:.6}",
        trainable_losses.1,
        frozen_losses.1
    );
    assert!(
        (trainable_losses.2 - frozen_losses.2).abs() < 1e-6,
        "trainable projected step total loss diverged from frozen step: {:.6} vs {:.6}",
        trainable_losses.2,
        frozen_losses.2
    );
    assert_eq!(step_trainable, step_frozen);
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
    let initial_total = model.losses(&x_t, &x_t1, REGULARIZER_WEIGHT).2;

    model.step(&x_t, &x_t1, REGULARIZER_WEIGHT, PREDICTOR_LR, PROJECTOR_LR);

    assert_eq!(
        model.target_projector.weight, target_projector_weight_snapshot,
        "target projector weight mutated during projected step"
    );
    assert_eq!(
        model.target_projector.bias, target_projector_bias_snapshot,
        "target projector bias mutated during projected step"
    );

    let final_total = model.losses(&x_t, &x_t1, REGULARIZER_WEIGHT).2;

    assert!(
        final_total + 1e-6 < initial_total,
        "one projected ProjectedVisionJepa step did not reduce total loss: {:.6} -> {:.6}",
        initial_total,
        final_total
    );
}

#[test]
fn projected_step_and_model_step_are_equivalent() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let predictor = make_predictor();
    let (x_t, x_t1) = make_train_batch(11_000, 0);

    let mut step_free = ProjectedVisionJepa::new(
        encoder.clone(),
        projector.clone(),
        target_projector.clone(),
        predictor.clone(),
    );
    let mut step_method = ProjectedVisionJepa::new(encoder, projector, target_projector, predictor);

    projected_step(
        &mut step_free,
        &x_t,
        &x_t1,
        REGULARIZER_WEIGHT,
        PREDICTOR_LR,
        PROJECTOR_LR,
    );
    step_method.step(&x_t, &x_t1, REGULARIZER_WEIGHT, PREDICTOR_LR, PROJECTOR_LR);

    assert_eq!(
        step_free, step_method,
        "projected helper and model API step diverged after same input and hyperparams"
    );
}

#[test]
fn projected_step_and_model_step_stable_over_two_steps() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let predictor = make_predictor();

    let mut step_free = ProjectedVisionJepa::new(
        encoder.clone(),
        projector.clone(),
        target_projector.clone(),
        predictor.clone(),
    );
    let mut step_method = ProjectedVisionJepa::new(encoder, projector, target_projector, predictor);

    let encoder_snapshot = step_method.encoder.clone();
    let target_weight_snapshot = step_method.target_projector.weight.clone();
    let target_bias_snapshot = step_method.target_projector.bias.clone();
    let mut prev_projector_weight = step_method.projector.weight.clone();
    let mut prev_projector_bias = step_method.projector.bias.clone();
    let mut prev_predictor_fc1_weight = step_method.predictor.fc1.weight.clone();
    let mut prev_predictor_fc1_bias = step_method.predictor.fc1.bias.clone();
    let mut prev_predictor_fc2_weight = step_method.predictor.fc2.weight.clone();
    let mut prev_predictor_fc2_bias = step_method.predictor.fc2.bias.clone();

    for batch_idx in 0..2 {
        let (x_t, x_t1) = make_train_batch(11_000, batch_idx as u64);
        let expected = projected_batch_losses(
            &step_free.encoder,
            &step_free.projector,
            &step_free.target_projector,
            &step_free.predictor,
            &x_t,
            &x_t1,
            REGULARIZER_WEIGHT,
        );

        projected_step(
            &mut step_free,
            &x_t,
            &x_t1,
            REGULARIZER_WEIGHT,
            PREDICTOR_LR,
            PROJECTOR_LR,
        );
        let method_step =
            step_method.step(&x_t, &x_t1, REGULARIZER_WEIGHT, PREDICTOR_LR, PROJECTOR_LR);

        assert!(
            (method_step.0 - expected.0).abs() < 1e-6,
            "projected API step mismatch at batch {} (pred): {:.6} vs {:.6}",
            batch_idx,
            method_step.0,
            expected.0
        );
        assert!(
            (method_step.1 - expected.1).abs() < 1e-6,
            "projected API step mismatch at batch {} (reg): {:.6} vs {:.6}",
            batch_idx,
            method_step.1,
            expected.1
        );
        assert!(
            (method_step.2 - expected.2).abs() < 1e-6,
            "projected API step mismatch at batch {} (total): {:.6} vs {:.6}",
            batch_idx,
            method_step.2,
            expected.2
        );

        assert_eq!(step_free, step_method);
        assert_eq!(step_method.encoder, encoder_snapshot);
        assert_eq!(step_method.target_projector.weight, target_weight_snapshot);
        assert_eq!(step_method.target_projector.bias, target_bias_snapshot);

        assert_ne!(step_method.projector.weight, prev_projector_weight);
        assert_ne!(step_method.projector.bias, prev_projector_bias);
        assert_ne!(step_method.predictor.fc1.weight, prev_predictor_fc1_weight);
        assert_ne!(step_method.predictor.fc1.bias, prev_predictor_fc1_bias);
        assert_ne!(step_method.predictor.fc2.weight, prev_predictor_fc2_weight);
        assert_ne!(step_method.predictor.fc2.bias, prev_predictor_fc2_bias);

        prev_projector_weight = step_method.projector.weight.clone();
        prev_projector_bias = step_method.projector.bias.clone();
        prev_predictor_fc1_weight = step_method.predictor.fc1.weight.clone();
        prev_predictor_fc1_bias = step_method.predictor.fc1.bias.clone();
        prev_predictor_fc2_weight = step_method.predictor.fc2.weight.clone();
        prev_predictor_fc2_bias = step_method.predictor.fc2.bias.clone();
    }
}

#[test]
fn projected_step_helper_matches_model_api_over_short_trajectory() {
    let encoder = make_frozen_encoder();

    let mut helper_model = ProjectedVisionJepa::new(
        encoder.clone(),
        make_projector(),
        make_projector(),
        make_predictor(),
    );
    let mut api_model = ProjectedVisionJepa::new(
        encoder,
        make_projector(),
        make_projector(),
        make_predictor(),
    );

    for batch_idx in 0..4 {
        let (x_t, x_t1) = make_train_batch(TRAIN_BASE_SEED, batch_idx as u64);
        let expected = projected_batch_losses(
            &helper_model.encoder,
            &helper_model.projector,
            &helper_model.target_projector,
            &helper_model.predictor,
            &x_t,
            &x_t1,
            REGULARIZER_WEIGHT,
        );
        let api_step = api_model.step(&x_t, &x_t1, REGULARIZER_WEIGHT, PREDICTOR_LR, PROJECTOR_LR);

        projected_step(
            &mut helper_model,
            &x_t,
            &x_t1,
            REGULARIZER_WEIGHT,
            PREDICTOR_LR,
            PROJECTOR_LR,
        );

        assert_eq!(helper_model, api_model);
        assert!(
            (api_step.0 - expected.0).abs() < 1e-6,
            "api prediction loss mismatch at batch {}: {:.6} vs {:.6}",
            batch_idx,
            api_step.0,
            expected.0
        );
        assert!(
            (api_step.1 - expected.1).abs() < 1e-6,
            "api regularizer loss mismatch at batch {}: {:.6} vs {:.6}",
            batch_idx,
            api_step.1,
            expected.1
        );
        assert!(
            (api_step.2 - expected.2).abs() < 1e-6,
            "api total loss mismatch at batch {}: {:.6} vs {:.6}",
            batch_idx,
            api_step.2,
            expected.2
        );
    }
}

#[test]
fn projected_random_temporal_training_trajectory_is_reproducible() {
    let encoder = make_frozen_encoder();
    let mut model_a = ProjectedVisionJepa::new(
        encoder.clone(),
        make_projector(),
        make_projector(),
        make_predictor(),
    );
    let mut model_b = ProjectedVisionJepa::new(
        encoder,
        make_projector(),
        make_projector(),
        make_predictor(),
    );
    let steps = 6u64;
    let (probe_t, probe_t1) = make_train_batch(11_100, 0);
    let train_seed = 11_100;

    let mut trajectory_a = Vec::<((f32, f32, f32), (f32, f32, f32))>::new();
    let mut trajectory_b = Vec::<((f32, f32, f32), (f32, f32, f32))>::new();

    for step in 1..=steps {
        let (x_t, x_t1) = make_train_batch(train_seed, step);
        let step_a = model_a.step(&x_t, &x_t1, REGULARIZER_WEIGHT, PREDICTOR_LR, PROJECTOR_LR);
        let step_b = model_b.step(&x_t, &x_t1, REGULARIZER_WEIGHT, PREDICTOR_LR, PROJECTOR_LR);

        assert!(
            (step_a.0 - step_b.0).abs() < 1e-7,
            "prediction loss diverged at trajectory batch {}: {:.6} vs {:.6}",
            step,
            step_a.0,
            step_b.0
        );
        assert!(
            (step_a.1 - step_b.1).abs() < 1e-7,
            "regularizer loss diverged at trajectory batch {}: {:.6} vs {:.6}",
            step,
            step_a.1,
            step_b.1
        );
        assert!(
            (step_a.2 - step_b.2).abs() < 1e-7,
            "total loss diverged at trajectory batch {}: {:.6} vs {:.6}",
            step,
            step_a.2,
            step_b.2
        );
        assert_eq!(
            model_a, model_b,
            "models diverged at trajectory batch {}",
            step
        );

        let train_loss_a = model_a.losses(&probe_t, &probe_t1, REGULARIZER_WEIGHT);
        let train_loss_b = model_b.losses(&probe_t, &probe_t1, REGULARIZER_WEIGHT);
        assert!(
            (train_loss_a.0 - train_loss_b.0).abs() < 1e-7,
            "probe train prediction loss diverged at trajectory batch {}: {:.6} vs {:.6}",
            step,
            train_loss_a.0,
            train_loss_b.0
        );
        assert!(
            (train_loss_a.1 - train_loss_b.1).abs() < 1e-7,
            "probe train regularizer loss diverged at trajectory batch {}: {:.6} vs {:.6}",
            step,
            train_loss_a.1,
            train_loss_b.1
        );
        assert!(
            (train_loss_a.2 - train_loss_b.2).abs() < 1e-7,
            "probe train total loss diverged at trajectory batch {}: {:.6} vs {:.6}",
            step,
            train_loss_a.2,
            train_loss_b.2
        );

        let validation_loss_a = projected_validation_losses_model(&model_a);
        let validation_loss_b = projected_validation_losses_model(&model_b);
        assert!(
            (validation_loss_a.0 - validation_loss_b.0).abs() < 1e-7,
            "validation prediction loss diverged at trajectory batch {}: {:.6} vs {:.6}",
            step,
            validation_loss_a.0,
            validation_loss_b.0
        );
        assert!(
            (validation_loss_a.1 - validation_loss_b.1).abs() < 1e-7,
            "validation regularizer loss diverged at trajectory batch {}: {:.6} vs {:.6}",
            step,
            validation_loss_a.1,
            validation_loss_b.1
        );
        assert!(
            (validation_loss_a.2 - validation_loss_b.2).abs() < 1e-7,
            "validation total loss diverged at trajectory batch {}: {:.6} vs {:.6}",
            step,
            validation_loss_a.2,
            validation_loss_b.2
        );

        trajectory_a.push((train_loss_a, validation_loss_a));
        trajectory_b.push((train_loss_b, validation_loss_b));
    }

    assert_eq!(trajectory_a, trajectory_b);
}

#[test]
fn projected_step_reported_losses_match_batch_losses() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let (x_t, x_t1) = make_train_batch(11_000, 2);
    let mut model = ProjectedVisionJepa::new(
        encoder.clone(),
        projector.clone(),
        target_projector,
        make_predictor(),
    );
    let (expected_prediction_loss, expected_regularizer_loss, expected_total_loss) =
        projected_batch_losses(
            &encoder,
            &model.projector,
            &model.target_projector,
            &model.predictor,
            &x_t,
            &x_t1,
            REGULARIZER_WEIGHT,
        );
    let (step_prediction_loss, step_regularizer_loss, step_total_loss) =
        model.step(&x_t, &x_t1, REGULARIZER_WEIGHT, PREDICTOR_LR, PROJECTOR_LR);
    let (_, _, actual_total_loss) = model.losses(&x_t, &x_t1, REGULARIZER_WEIGHT);

    assert!(
        (step_prediction_loss - expected_prediction_loss).abs() < 1e-6,
        "reported prediction loss mismatch: {:.6} vs {:.6}",
        step_prediction_loss,
        expected_prediction_loss
    );
    assert!(
        (step_regularizer_loss - expected_regularizer_loss).abs() < 1e-6,
        "reported regularizer loss mismatch: {:.6} vs {:.6}",
        step_regularizer_loss,
        expected_regularizer_loss
    );
    assert!(
        (step_total_loss - expected_total_loss).abs() < 1e-6,
        "reported total loss mismatch: {:.6} vs {:.6}",
        step_total_loss,
        expected_total_loss
    );
    assert!(
        actual_total_loss < expected_total_loss,
        "step did not reduce total loss: {:.6} -> {:.6}",
        expected_total_loss,
        actual_total_loss
    );
}

#[test]
fn projected_step_reduces_total_loss_over_two_steps_on_same_batch() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let mut model =
        ProjectedVisionJepa::new(encoder, projector, target_projector, make_predictor());
    let (x_t, x_t1) = make_train_batch(11_000, 4);
    let mut prev_total = model.losses(&x_t, &x_t1, REGULARIZER_WEIGHT).2;

    for step_idx in 0..2 {
        let expected = projected_batch_losses(
            &model.encoder,
            &model.projector,
            &model.target_projector,
            &model.predictor,
            &x_t,
            &x_t1,
            REGULARIZER_WEIGHT,
        );
        let (step_prediction_loss, step_regularizer_loss, step_total_loss) =
            model.step(&x_t, &x_t1, REGULARIZER_WEIGHT, PREDICTOR_LR, PROJECTOR_LR);
        let (_, _, actual_total_loss) = model.losses(&x_t, &x_t1, REGULARIZER_WEIGHT);

        assert!(
            (step_prediction_loss - expected.0).abs() < 1e-6,
            "projected step prediction mismatch at step {}: {:.6} vs {:.6}",
            step_idx,
            step_prediction_loss,
            expected.0
        );
        assert!(
            (step_regularizer_loss - expected.1).abs() < 1e-6,
            "projected step regularizer mismatch at step {}: {:.6} vs {:.6}",
            step_idx,
            step_regularizer_loss,
            expected.1
        );
        assert!(
            (step_total_loss - expected.2).abs() < 1e-6,
            "projected step total mismatch at step {}: {:.6} vs {:.6}",
            step_idx,
            step_total_loss,
            expected.2
        );

        assert!(
            actual_total_loss + 1e-6 < prev_total,
            "same-batch projected total loss did not decrease at step {}: {:.6} -> {:.6}",
            step_idx,
            prev_total,
            actual_total_loss
        );
        prev_total = actual_total_loss;
    }
}

#[test]
fn projected_step_updates_trainable_parameters() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let predictor = make_predictor();
    let mut model = ProjectedVisionJepa::new(encoder, projector, target_projector, predictor);
    let (x_t, x_t1) = make_train_batch(11_000, 0);
    let encoder_snapshot = model.encoder.clone();
    let target_weight_snapshot = model.target_projector.weight.clone();
    let target_bias_snapshot = model.target_projector.bias.clone();

    let projector_weight_snapshot = model.projector.weight.clone();
    let projector_bias_snapshot = model.projector.bias.clone();
    let predictor_fc1_weight_snapshot = model.predictor.fc1.weight.clone();
    let predictor_fc1_bias_snapshot = model.predictor.fc1.bias.clone();
    let predictor_fc2_weight_snapshot = model.predictor.fc2.weight.clone();
    let predictor_fc2_bias_snapshot = model.predictor.fc2.bias.clone();

    model.step(&x_t, &x_t1, REGULARIZER_WEIGHT, PREDICTOR_LR, PROJECTOR_LR);

    assert_eq!(model.target_projector.weight, target_weight_snapshot);
    assert_eq!(model.target_projector.bias, target_bias_snapshot);
    assert_eq!(model.encoder, encoder_snapshot);

    assert_ne!(model.projector.weight, projector_weight_snapshot);
    assert_ne!(model.projector.bias, projector_bias_snapshot);
    assert_ne!(model.predictor.fc1.weight, predictor_fc1_weight_snapshot);
    assert_ne!(model.predictor.fc1.bias, predictor_fc1_bias_snapshot);
    assert_ne!(model.predictor.fc2.weight, predictor_fc2_weight_snapshot);
    assert_ne!(model.predictor.fc2.bias, predictor_fc2_bias_snapshot);
}

#[test]
fn projected_training_steps_preserve_target_projector_after_multiple_batches() {
    let encoder = make_frozen_encoder();
    let online_projector = make_projector();
    let target_projector = online_projector.clone();
    let target_projector_weight_snapshot = target_projector.weight.clone();
    let target_projector_bias_snapshot = target_projector.bias.clone();
    let predictor = make_predictor();
    let mut model = ProjectedVisionJepa::new(
        encoder.clone(),
        online_projector,
        target_projector.clone(),
        predictor,
    );
    let (x_t_step0, x_t1_step0) = make_train_batch(11_000, 0);
    let (x_t_step1, x_t1_step1) = make_train_batch(11_000, 1);

    projected_step(
        &mut model,
        &x_t_step0,
        &x_t1_step0,
        REGULARIZER_WEIGHT,
        PREDICTOR_LR,
        PROJECTOR_LR,
    );

    projected_step(
        &mut model,
        &x_t_step1,
        &x_t1_step1,
        REGULARIZER_WEIGHT,
        PREDICTOR_LR,
        PROJECTOR_LR,
    );

    assert_eq!(
        model.target_projector.weight, target_projector_weight_snapshot,
        "target projector weight mutated during multi-step projected training"
    );
    assert_eq!(
        model.target_projector.bias, target_projector_bias_snapshot,
        "target projector bias mutated during multi-step projected training"
    );
}

#[test]
fn projected_vision_jepa_losses_matches_projection_support() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let predictor = make_predictor();
    let (x_t, x_t1) = make_train_batch(11_000, 0);
    let model = ProjectedVisionJepa::new(encoder.clone(), projector, target_projector, predictor);
    let support = projected_batch_losses(
        &encoder,
        &model.projector,
        &model.target_projector,
        &model.predictor,
        &x_t,
        &x_t1,
        REGULARIZER_WEIGHT,
    );
    let model_losses = model.losses(&x_t, &x_t1, REGULARIZER_WEIGHT);

    assert!(
        (support.0 - model_losses.0).abs() < 1e-6,
        "prediction loss mismatch between core and support: {:.6} vs {:.6}",
        support.0,
        model_losses.0
    );
    assert!(
        (support.1 - model_losses.1).abs() < 1e-6,
        "regularizer loss mismatch between core and support: {:.6} vs {:.6}",
        support.1,
        model_losses.1
    );
    assert!(
        (support.2 - model_losses.2).abs() < 1e-6,
        "total loss mismatch between core and support: {:.6} vs {:.6}",
        support.2,
        model_losses.2
    );
}

#[test]
fn projected_validation_losses_matches_projection_support() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let predictor = make_predictor();
    let model = ProjectedVisionJepa::new(encoder.clone(), projector, target_projector, predictor);

    let support = projected_validation_losses_projection_support(&model);
    let model_losses = projected_validation_losses_model(&model);

    assert!(
        (support.0 - model_losses.0).abs() < 1e-6,
        "validation prediction loss mismatch between core and support: {:.6} vs {:.6}",
        support.0,
        model_losses.0
    );
    assert!(
        (support.1 - model_losses.1).abs() < 1e-6,
        "validation regularizer loss mismatch between core and support: {:.6} vs {:.6}",
        support.1,
        model_losses.1
    );
    assert!(
        (support.2 - model_losses.2).abs() < 1e-6,
        "validation total loss mismatch between core and support: {:.6} vs {:.6}",
        support.2,
        model_losses.2
    );
}

#[test]
fn projected_random_temporal_loop_reduces_train_and_validation_loss() {
    let encoder = make_frozen_encoder();
    let online_projector = make_projector();
    let target_projector = online_projector.clone();
    let mut model = ProjectedVisionJepa::new(
        encoder,
        online_projector,
        target_projector,
        make_predictor(),
    );
    let (probe_t, probe_t1) = make_train_batch(TRAIN_BASE_SEED, 0);
    let steps = 120;
    let initial_losses = model.losses(&probe_t, &probe_t1, REGULARIZER_WEIGHT);
    let initial_validation = projected_validation_losses_model(&model);

    for step in 1..=steps {
        let (x_t, x_t1) = make_train_batch(TRAIN_BASE_SEED, step as u64);
        model.step(&x_t, &x_t1, REGULARIZER_WEIGHT, PREDICTOR_LR, PROJECTOR_LR);

        if step == 1 || step == steps {
            println!(
                "projected temporal step {:03} | train total {:.6} | val total {:.6}",
                step,
                model.losses(&probe_t, &probe_t1, REGULARIZER_WEIGHT).2,
                projected_validation_losses_model(&model).2
            );
        }
    }

    let final_losses = model.losses(&probe_t, &probe_t1, REGULARIZER_WEIGHT);
    let final_validation = projected_validation_losses_model(&model);

    assert_temporal_experiment_improved(
        "projected",
        initial_losses.2,
        final_losses.2,
        initial_validation.2,
        final_validation.2,
        PROJECTED_TRAIN_LOSS_MAX_REDUCTION_RATIO,
        PROJECTED_VALIDATION_LOSS_MAX_REDUCTION_RATIO,
    );
}
