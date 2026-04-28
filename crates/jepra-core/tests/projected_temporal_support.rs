#[allow(dead_code)]
#[path = "../examples/support/projected_temporal.rs"]
mod projected_temporal;
#[allow(dead_code)]
#[path = "../examples/support/temporal_vision.rs"]
mod temporal_vision;

use jepra_core::{
    Linear, LinearGrads, Predictor, ProjectedVisionJepa, SignedAngularRadialObjectiveConfig,
    SignedBankSoftmaxObjectiveConfig, SignedMarginObjectiveConfig, Tensor,
    combine_projection_grads, gaussian_moment_regularizer, gaussian_moment_regularizer_grad,
    mse_loss, mse_loss_grad, projection_stats, projector_drift_regularizer,
    projector_drift_regularizer_grads,
};
use projected_temporal::{
    PROJECTED_TRAIN_LOSS_MAX_REDUCTION_RATIO, PROJECTED_VALIDATION_BASE_SEED,
    PROJECTED_VALIDATION_BATCHES, PROJECTED_VALIDATION_LOSS_MAX_REDUCTION_RATIO,
    projected_batch_losses, projected_signed_angular_radial_objective_report_from_base_seed,
    projected_signed_candidate_selector_active_normalized_stable_hard_full_output_coupling_loss_and_grad,
    projected_signed_candidate_selector_stable_hard_full_output_coupling_loss_and_grad,
    projected_signed_direct_candidate_margin_objective_loss_and_grad,
    projected_signed_margin_objective_loss_and_grad,
    projected_signed_objective_error_breakdown_from_base_seed,
    projected_signed_prediction_bank_margin_from_base_seed,
    projected_signed_prediction_bank_unit_geometry_from_base_seed,
    projected_signed_prediction_geometry_counterfactual_from_base_seed,
    projected_signed_prediction_ray_boundary_from_base_seed,
    projected_signed_radial_calibration_report_from_base_seed,
    projected_signed_state_separability_from_base_seed,
    projected_signed_target_bank_separability_from_base_seed,
    projected_signed_true_target_mse_amplification_loss_and_grad,
    projected_signed_velocity_bank_breakdown_from_base_seed, projected_step,
    projected_validation_batch_losses, projected_validation_batch_losses_from_base_seed,
    projected_validation_batch_losses_from_base_seed_for_task,
    projected_velocity_bank_ranking_from_base_seed,
};
use temporal_vision::{
    BATCH_SIZE, CompactEncoderMode, PredictorMode, TemporalExperimentSummary, TemporalRunConfig,
    TemporalTaskMode, assert_seed_range_has_both_motion_modes,
    assert_seed_range_has_single_and_double_square_batch_examples,
    assert_temporal_experiment_improved, make_compact_frozen_encoder,
    make_compact_frozen_encoder_signed_direction,
    make_compact_frozen_encoder_signed_direction_magnitude, make_frozen_encoder,
    make_temporal_batch, make_train_batch, run_temporal_experiment_with_summary,
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

fn symmetric_relative_difference(lhs: f32, rhs: f32) -> f32 {
    let scale = lhs.abs() + rhs.abs();
    if scale == 0.0 {
        0.0
    } else {
        2.0 * (lhs - rhs).abs() / scale
    }
}

fn tensor_symmetric_relative_difference(lhs: &Tensor, rhs: &Tensor) -> f32 {
    assert_eq!(lhs.shape, rhs.shape);
    let diff_sum = lhs
        .data
        .iter()
        .zip(&rhs.data)
        .map(|(lhs_value, rhs_value)| (lhs_value - rhs_value).abs())
        .sum::<f32>();
    let scale_sum = lhs
        .data
        .iter()
        .zip(&rhs.data)
        .map(|(lhs_value, rhs_value)| lhs_value.abs() + rhs_value.abs())
        .sum::<f32>();

    if scale_sum == 0.0 {
        0.0
    } else {
        2.0 * diff_sum / scale_sum
    }
}

fn projected_run_with_encoder(
    train_base_seed: u64,
    predictor_seed: u64,
    predictor_lr: f32,
    encoder_lr: f32,
    compact_encoder_mode: CompactEncoderMode,
    make_encoder: fn() -> jepra_core::EmbeddingEncoder,
) -> TemporalExperimentSummary {
    let encoder = make_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let mut model = ProjectedVisionJepa::new(
        encoder,
        projector,
        target_projector,
        make_predictor_with_seed(predictor_seed),
    );
    let config = TemporalRunConfig {
        train_base_seed,
        total_steps: 120,
        log_every: 120,
        encoder_learning_rate: encoder_lr,
        temporal_task_mode: TemporalTaskMode::RandomSpeed,
        compact_encoder_mode,
        predictor_mode: PredictorMode::Baseline,
        residual_delta_scale: 1.0,
        projector_drift_weight: 0.0,
        signed_margin_weight: 0.0,
        signed_margin_config: SignedMarginObjectiveConfig::default(),
        signed_bank_softmax_weight: 0.0,
        signed_bank_softmax_config: SignedBankSoftmaxObjectiveConfig::default(),
        signed_radial_weight: 0.0,
        signed_angular_radial_weight: 0.0,
        signed_angular_radial_config: SignedAngularRadialObjectiveConfig::default(),
        target_projection_momentum: 0.5,
        target_projection_momentum_start: 1.0,
        target_projection_momentum_end: 0.5,
        target_projection_momentum_warmup_steps: 8,
    };

    let (probe_t, probe_t1) = make_train_batch(config.train_base_seed, 0);
    let initial_train_loss = model.losses(&probe_t, &probe_t1, REGULARIZER_WEIGHT).2;
    let initial_validation_loss = projected_validation_losses_model(&model).2;

    run_temporal_experiment_with_summary(
        config,
        &mut model,
        initial_train_loss,
        initial_validation_loss,
        |model, step, _| {
            model.set_target_projection_momentum(config.target_projection_momentum_at_step(step));
            let (x_t, x_t1) = make_train_batch(config.train_base_seed, step as u64);
            let (_, _, train_loss) = model.step_with_trainable_encoder(
                &x_t,
                &x_t1,
                REGULARIZER_WEIGHT,
                predictor_lr,
                PROJECTOR_LR,
                config.encoder_learning_rate,
            );
            train_loss
        },
        |model| projected_validation_losses_model(model).2,
    )
}

fn assert_projected_frozen_vs_trainable_protocol(
    label: &str,
    train_base_seed: u64,
    predictor_seed: u64,
    predictor_lr: f32,
    encoder_lr: f32,
    compact_encoder_mode: CompactEncoderMode,
    make_encoder: fn() -> jepra_core::EmbeddingEncoder,
) {
    let frozen = projected_run_with_encoder(
        train_base_seed,
        predictor_seed,
        predictor_lr,
        0.0,
        compact_encoder_mode,
        make_encoder,
    );
    let trainable = projected_run_with_encoder(
        train_base_seed,
        predictor_seed,
        predictor_lr,
        encoder_lr,
        compact_encoder_mode,
        make_encoder,
    );

    assert_temporal_experiment_improved(
        &format!("{label} frozen projected"),
        frozen.initial_train_loss,
        frozen.final_train_loss,
        frozen.initial_validation_loss,
        frozen.final_validation_loss,
        PROJECTED_TRAIN_LOSS_MAX_REDUCTION_RATIO,
        PROJECTED_VALIDATION_LOSS_MAX_REDUCTION_RATIO,
    );
    assert_temporal_experiment_improved(
        &format!("{label} trainable projected"),
        trainable.initial_train_loss,
        trainable.final_train_loss,
        trainable.initial_validation_loss,
        trainable.final_validation_loss,
        PROJECTED_TRAIN_LOSS_MAX_REDUCTION_RATIO,
        PROJECTED_VALIDATION_LOSS_MAX_REDUCTION_RATIO,
    );

    assert!(
        trainable.final_train_loss <= frozen.final_train_loss * 2.0,
        "trainable final train loss regressed too far vs frozen for {label}: frozen {:.6} vs trainable {:.6}",
        frozen.final_train_loss,
        trainable.final_train_loss
    );
    assert!(
        trainable.final_validation_loss <= frozen.final_validation_loss * 2.0,
        "trainable final validation loss regressed too far vs frozen for {label}: frozen {:.6} vs trainable {:.6}",
        frozen.final_validation_loss,
        trainable.final_validation_loss
    );
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
fn projected_velocity_trail_validation_losses_are_finite() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let model = ProjectedVisionJepa::new(encoder, projector, target_projector, make_predictor());

    let losses = projected_validation_batch_losses_from_base_seed_for_task(
        &model,
        REGULARIZER_WEIGHT,
        PROJECTED_VALIDATION_BASE_SEED,
        2,
        TemporalTaskMode::VelocityTrail,
    );

    assert!(losses.0.is_finite() && losses.0 > 0.0);
    assert!(losses.1.is_finite() && losses.1 >= 0.0);
    assert!(losses.2.is_finite() && losses.2 > 0.0);
}

#[test]
fn projected_signed_velocity_trail_validation_losses_are_finite() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let model = ProjectedVisionJepa::new(encoder, projector, target_projector, make_predictor());

    let losses = projected_validation_batch_losses_from_base_seed_for_task(
        &model,
        REGULARIZER_WEIGHT,
        PROJECTED_VALIDATION_BASE_SEED,
        2,
        TemporalTaskMode::SignedVelocityTrail,
    );

    assert!(losses.0.is_finite() && losses.0 > 0.0);
    assert!(losses.1.is_finite() && losses.1 >= 0.0);
    assert!(losses.2.is_finite() && losses.2 > 0.0);
}

#[test]
fn projected_signed_objective_error_breakdown_reconciles_signed_validation_prediction_loss() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let model = ProjectedVisionJepa::new(encoder, projector, target_projector, make_predictor());

    let breakdown = projected_signed_objective_error_breakdown_from_base_seed(
        &model,
        PROJECTED_VALIDATION_BASE_SEED,
        2,
    );
    let validation_losses = projected_validation_batch_losses_from_base_seed_for_task(
        &model,
        REGULARIZER_WEIGHT,
        PROJECTED_VALIDATION_BASE_SEED,
        2,
        TemporalTaskMode::SignedVelocityTrail,
    );

    for value in [
        breakdown.all_loss,
        breakdown.dx_neg2_loss,
        breakdown.dx_neg1_loss,
        breakdown.dx_pos1_loss,
        breakdown.dx_pos2_loss,
        breakdown.neg_loss,
        breakdown.pos_loss,
        breakdown.slow_loss,
        breakdown.fast_loss,
        breakdown.sign_gap,
        breakdown.speed_gap,
    ] {
        assert!(value.is_finite());
    }

    assert!(breakdown.all_loss > 0.0);
    assert!(breakdown.dx_neg2_loss >= 0.0);
    assert!(breakdown.dx_neg1_loss >= 0.0);
    assert!(breakdown.dx_pos1_loss >= 0.0);
    assert!(breakdown.dx_pos2_loss >= 0.0);
    assert_eq!(breakdown.samples, BATCH_SIZE * 2);
    assert_eq!(
        breakdown.dx_neg2_samples
            + breakdown.dx_neg1_samples
            + breakdown.dx_pos1_samples
            + breakdown.dx_pos2_samples,
        breakdown.samples
    );
    assert_eq!(breakdown.dx_neg2_samples, breakdown.dx_neg1_samples);
    assert_eq!(breakdown.dx_neg1_samples, breakdown.dx_pos1_samples);
    assert_eq!(breakdown.dx_pos1_samples, breakdown.dx_pos2_samples);

    let bucket_weighted_loss = (breakdown.dx_neg2_loss * breakdown.dx_neg2_samples as f32
        + breakdown.dx_neg1_loss * breakdown.dx_neg1_samples as f32
        + breakdown.dx_pos1_loss * breakdown.dx_pos1_samples as f32
        + breakdown.dx_pos2_loss * breakdown.dx_pos2_samples as f32)
        / breakdown.samples as f32;
    let sign_weighted_loss = (breakdown.neg_loss
        * (breakdown.dx_neg2_samples + breakdown.dx_neg1_samples) as f32
        + breakdown.pos_loss * (breakdown.dx_pos1_samples + breakdown.dx_pos2_samples) as f32)
        / breakdown.samples as f32;
    let speed_weighted_loss = (breakdown.slow_loss
        * (breakdown.dx_neg1_samples + breakdown.dx_pos1_samples) as f32
        + breakdown.fast_loss * (breakdown.dx_neg2_samples + breakdown.dx_pos2_samples) as f32)
        / breakdown.samples as f32;

    assert!(symmetric_relative_difference(breakdown.all_loss, validation_losses.0) < 1e-5);
    assert!(symmetric_relative_difference(breakdown.all_loss, bucket_weighted_loss) < 1e-6);
    assert!(symmetric_relative_difference(breakdown.all_loss, sign_weighted_loss) < 1e-6);
    assert!(symmetric_relative_difference(breakdown.all_loss, speed_weighted_loss) < 1e-6);
    assert!(
        symmetric_relative_difference(breakdown.sign_gap, breakdown.pos_loss - breakdown.neg_loss)
            < 1e-6
    );
    assert!(
        symmetric_relative_difference(
            breakdown.speed_gap,
            breakdown.fast_loss - breakdown.slow_loss
        ) < 1e-6
    );
}

#[test]
fn projected_signed_margin_objective_grad_is_finite_for_signed_batch() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let model = ProjectedVisionJepa::new(encoder, projector, target_projector, make_predictor());
    let (x_t, x_t1) = temporal_vision::make_temporal_batch_for_task(
        BATCH_SIZE,
        PROJECTED_VALIDATION_BASE_SEED,
        TemporalTaskMode::SignedVelocityTrail,
    );

    let (report, grad) = projected_signed_margin_objective_loss_and_grad(
        &model,
        &x_t,
        &x_t1,
        SignedMarginObjectiveConfig::default(),
    );

    assert!(report.bank_loss.is_finite() && report.bank_loss >= 0.0);
    assert!(report.sign_loss.is_finite() && report.sign_loss >= 0.0);
    assert!(report.speed_loss.is_finite() && report.speed_loss >= 0.0);
    assert!(report.weighted_loss.is_finite() && report.weighted_loss >= 0.0);
    assert_eq!(report.samples, BATCH_SIZE);
    assert_eq!(grad.shape, vec![BATCH_SIZE, PROJECTION_DIM]);
    assert!(grad.data.iter().all(|value| value.is_finite()));
}

#[test]
fn projected_signed_direct_candidate_margin_objective_grad_is_finite_for_signed_batch() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let model = ProjectedVisionJepa::new(encoder, projector, target_projector, make_predictor());
    let (x_t, x_t1) = temporal_vision::make_temporal_batch_for_task(
        BATCH_SIZE,
        PROJECTED_VALIDATION_BASE_SEED,
        TemporalTaskMode::SignedVelocityTrail,
    );

    let (report, grad) =
        projected_signed_direct_candidate_margin_objective_loss_and_grad(&model, &x_t, &x_t1, 0.05);

    assert!(report.loss.is_finite() && report.loss >= 0.0);
    assert!((0.0..=1.0).contains(&report.active_rate));
    assert!(report.true_distance.is_finite() && report.true_distance >= 0.0);
    assert!(report.wrong_distance.is_finite() && report.wrong_distance >= 0.0);
    assert!(report.margin.is_finite());
    assert!((0.0..=1.0).contains(&report.positive_margin_rate));
    assert!((0.0..=1.0).contains(&report.top1));
    assert_eq!(report.samples, BATCH_SIZE);
    assert_eq!(grad.shape, vec![BATCH_SIZE, PROJECTION_DIM]);
    assert!(grad.data.iter().all(|value| value.is_finite()));
}

#[test]
fn projected_stable_selector_output_coupling_zeroes_inactive_gate() {
    let encoder = make_compact_frozen_encoder_signed_direction_magnitude();
    let projector = make_projector();
    let target_projector = projector.clone();
    let model = ProjectedVisionJepa::new(encoder, projector, target_projector, make_predictor());
    let (x_t, x_t1) = temporal_vision::make_temporal_batch_for_task(
        BATCH_SIZE,
        PROJECTED_VALIDATION_BASE_SEED,
        TemporalTaskMode::SignedVelocityTrail,
    );
    let selector_logits = Tensor::new(vec![0.0; BATCH_SIZE * 4], vec![BATCH_SIZE, 4]);

    let (report, grad) =
        projected_signed_candidate_selector_stable_hard_full_output_coupling_loss_and_grad(
            &model,
            &x_t,
            &x_t1,
            &selector_logits,
            1.0,
        );

    assert_eq!(report.active_samples, 0);
    assert_eq!(report.active_rate, 0.0);
    assert_eq!(report.samples, BATCH_SIZE);
    assert_eq!(report.candidates, 4);
    assert_eq!(report.loss, 0.0);
    assert_eq!(grad.shape, vec![BATCH_SIZE, PROJECTION_DIM]);
    assert!(grad.data.iter().all(|value| value.abs() < 1e-7));
}

#[test]
fn projected_active_normalized_selector_output_coupling_scales_active_gate() {
    let encoder = make_compact_frozen_encoder_signed_direction_magnitude();
    let projector = make_projector();
    let target_projector = projector.clone();
    let model = ProjectedVisionJepa::new(encoder, projector, target_projector, make_predictor());
    let (x_t, x_t1) = temporal_vision::make_temporal_batch_for_task(
        BATCH_SIZE,
        PROJECTED_VALIDATION_BASE_SEED,
        TemporalTaskMode::SignedVelocityTrail,
    );
    let selector_logits = Tensor::new(vec![0.0; BATCH_SIZE * 4], vec![BATCH_SIZE, 4]);

    let (stable_report, stable_grad) =
        projected_signed_candidate_selector_stable_hard_full_output_coupling_loss_and_grad(
            &model,
            &x_t,
            &x_t1,
            &selector_logits,
            0.25,
        );
    let (normalized_report, normalized_grad) =
        projected_signed_candidate_selector_active_normalized_stable_hard_full_output_coupling_loss_and_grad(
            &model,
            &x_t,
            &x_t1,
            &selector_logits,
            0.25,
        );

    assert!(stable_report.active_samples > 0);
    assert!(stable_report.active_samples < BATCH_SIZE);
    assert_eq!(
        normalized_report.active_samples,
        stable_report.active_samples
    );
    assert_eq!(normalized_report.samples, stable_report.samples);

    let scale = BATCH_SIZE as f32 / stable_report.active_samples as f32;
    assert!((normalized_report.loss - stable_report.loss * scale).abs() < 1e-6);
    for (normalized, stable) in normalized_grad.data.iter().zip(&stable_grad.data) {
        assert!((normalized - stable * scale).abs() < 1e-6);
    }
}

#[test]
fn projected_extra_prediction_grad_zero_path_matches_existing_step() {
    let (x_t, x_t1) = make_train_batch(TRAIN_BASE_SEED, 1);
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let model = ProjectedVisionJepa::new(encoder, projector, target_projector, make_predictor());
    let mut existing_step_model = model.clone();
    let mut extra_step_model = model;

    let existing_losses = existing_step_model.step_with_projector_drift_regularizer(
        &x_t,
        &x_t1,
        REGULARIZER_WEIGHT,
        0.0,
        PREDICTOR_LR,
        PROJECTOR_LR,
    );
    let extra_losses = extra_step_model.step_with_extra_prediction_grad(
        &x_t,
        &x_t1,
        REGULARIZER_WEIGHT,
        0.0,
        0.0,
        None,
        PREDICTOR_LR,
        PROJECTOR_LR,
        0.0,
    );

    assert_eq!(existing_losses, extra_losses);
    assert_eq!(existing_step_model, extra_step_model);
}

#[test]
fn projected_true_target_mse_amplification_matches_base_prediction_loss_and_grad() {
    let (x_t, x_t1) = make_train_batch(TRAIN_BASE_SEED, 1);
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let model = ProjectedVisionJepa::new(encoder, projector, target_projector, make_predictor());

    let prediction = model.predict_next_projection(&x_t);
    let target = model.target_projection(&x_t1);
    let expected_loss = mse_loss(&prediction, &target);
    let expected_grad = mse_loss_grad(&prediction, &target);
    let (loss, grad) =
        projected_signed_true_target_mse_amplification_loss_and_grad(&model, &x_t, &x_t1);

    assert!((loss - expected_loss).abs() < 1e-6);
    assert_eq!(grad.shape, expected_grad.shape);
    for (actual, expected) in grad.data.iter().zip(&expected_grad.data) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn projected_velocity_bank_ranking_is_finite_and_bounded() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let model = ProjectedVisionJepa::new(encoder, projector, target_projector, make_predictor());

    let ranking = projected_velocity_bank_ranking_from_base_seed(
        &model,
        PROJECTED_VALIDATION_BASE_SEED,
        2,
        TemporalTaskMode::VelocityTrail,
    );

    assert!(ranking.mrr.is_finite());
    assert!(
        (0.5..=1.0).contains(&ranking.mrr),
        "two-candidate MRR should be in [0.5, 1.0], got {:.6}",
        ranking.mrr
    );
    assert!(
        (0.0..=1.0).contains(&ranking.top1),
        "top1 should be in [0.0, 1.0], got {:.6}",
        ranking.top1
    );
    assert!(
        (1.0..=2.0).contains(&ranking.mean_rank),
        "mean rank should be in [1.0, 2.0], got {:.6}",
        ranking.mean_rank
    );
    assert_eq!(ranking.samples, BATCH_SIZE * 2);
    assert_eq!(ranking.candidates, 2);
}

#[test]
fn projected_signed_velocity_bank_ranking_is_finite_and_bounded() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let model = ProjectedVisionJepa::new(encoder, projector, target_projector, make_predictor());

    let ranking = projected_velocity_bank_ranking_from_base_seed(
        &model,
        PROJECTED_VALIDATION_BASE_SEED,
        2,
        TemporalTaskMode::SignedVelocityTrail,
    );

    assert!(ranking.mrr.is_finite());
    assert!(
        (0.25..=1.0).contains(&ranking.mrr),
        "four-candidate MRR should be in [0.25, 1.0], got {:.6}",
        ranking.mrr
    );
    assert!(
        (0.0..=1.0).contains(&ranking.top1),
        "top1 should be in [0.0, 1.0], got {:.6}",
        ranking.top1
    );
    assert!(
        (1.0..=4.0).contains(&ranking.mean_rank),
        "mean rank should be in [1.0, 4.0], got {:.6}",
        ranking.mean_rank
    );
    assert_eq!(ranking.samples, BATCH_SIZE * 2);
    assert_eq!(ranking.candidates, 4);
}

#[test]
fn projected_signed_velocity_bank_breakdown_is_finite_and_balanced() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let model = ProjectedVisionJepa::new(encoder, projector, target_projector, make_predictor());

    let breakdown = projected_signed_velocity_bank_breakdown_from_base_seed(
        &model,
        PROJECTED_VALIDATION_BASE_SEED,
        2,
    );

    for value in [
        breakdown.negative_mrr,
        breakdown.positive_mrr,
        breakdown.slow_mrr,
        breakdown.fast_mrr,
        breakdown.sign_top1,
        breakdown.speed_top1,
    ] {
        assert!(value.is_finite());
        assert!(
            (0.0..=1.0).contains(&value),
            "breakdown metric should be in [0.0, 1.0], got {:.6}",
            value
        );
    }

    assert_eq!(breakdown.samples, BATCH_SIZE * 2);
    assert_eq!(
        breakdown.negative_samples + breakdown.positive_samples,
        breakdown.samples
    );
    assert_eq!(
        breakdown.slow_samples + breakdown.fast_samples,
        breakdown.samples
    );
    assert_eq!(breakdown.negative_samples, breakdown.positive_samples);
    assert_eq!(breakdown.slow_samples, breakdown.fast_samples);
    assert_eq!(
        breakdown.true_neg_best_neg + breakdown.true_neg_best_pos,
        breakdown.negative_samples
    );
    assert_eq!(
        breakdown.true_pos_best_neg + breakdown.true_pos_best_pos,
        breakdown.positive_samples
    );
    assert_eq!(
        breakdown.true_slow_best_slow + breakdown.true_slow_best_fast,
        breakdown.slow_samples
    );
    assert_eq!(
        breakdown.true_fast_best_slow + breakdown.true_fast_best_fast,
        breakdown.fast_samples
    );
}

#[test]
fn projected_signed_state_separability_is_finite_and_bounded() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let model = ProjectedVisionJepa::new(encoder, projector, target_projector, make_predictor());

    let separability = projected_signed_state_separability_from_base_seed(
        &model,
        PROJECTED_VALIDATION_BASE_SEED,
        2,
    );

    for value in [
        separability.latent_mrr,
        separability.latent_top1,
        separability.latent_sign_top1,
        separability.latent_mean_rank,
        separability.projection_mrr,
        separability.projection_top1,
        separability.projection_sign_top1,
        separability.projection_mean_rank,
    ] {
        assert!(value.is_finite());
    }

    assert!((0.25..=1.0).contains(&separability.latent_mrr));
    assert!((0.0..=1.0).contains(&separability.latent_top1));
    assert!((0.0..=1.0).contains(&separability.latent_sign_top1));
    assert!((1.0..=4.0).contains(&separability.latent_mean_rank));
    assert!((0.25..=1.0).contains(&separability.projection_mrr));
    assert!((0.0..=1.0).contains(&separability.projection_top1));
    assert!((0.0..=1.0).contains(&separability.projection_sign_top1));
    assert!((1.0..=4.0).contains(&separability.projection_mean_rank));
    assert_eq!(separability.support_samples, BATCH_SIZE);
    assert_eq!(separability.query_samples, BATCH_SIZE);
    assert_eq!(separability.candidates, 4);
}

#[test]
fn signed_direction_compact_encoder_keeps_spatial_features_and_three_latents() {
    let encoder = make_compact_frozen_encoder_signed_direction();
    let (x_t, _) = temporal_vision::make_temporal_batch_for_task(
        BATCH_SIZE,
        PROJECTED_VALIDATION_BASE_SEED,
        TemporalTaskMode::SignedVelocityTrail,
    );

    let feature_map = encoder.backbone.forward(&x_t);
    let latents = encoder.forward(&x_t);

    assert_eq!(feature_map.shape[0], BATCH_SIZE);
    assert_eq!(feature_map.shape[1], 3);
    assert!(
        feature_map.shape[2] > 1 && feature_map.shape[3] > 1,
        "signed-direction encoder should preserve spatial maps before pooling, got {:?}",
        feature_map.shape
    );
    assert_eq!(latents.shape, vec![BATCH_SIZE, 3]);
    assert!(feature_map.data.iter().all(|value| value.is_finite()));
    assert!(latents.data.iter().all(|value| value.is_finite()));
}

#[test]
fn signed_direction_magnitude_compact_encoder_keeps_three_latents() {
    let encoder = make_compact_frozen_encoder_signed_direction_magnitude();
    let (x_t, _) = temporal_vision::make_temporal_batch_for_task(
        BATCH_SIZE,
        PROJECTED_VALIDATION_BASE_SEED,
        TemporalTaskMode::SignedVelocityTrail,
    );

    let feature_map = encoder.backbone.forward(&x_t);
    let latents = encoder.forward(&x_t);

    assert_eq!(feature_map.shape[0], BATCH_SIZE);
    assert_eq!(feature_map.shape[1], 3);
    assert!(
        feature_map.shape[2] > 1 && feature_map.shape[3] > 1,
        "signed-direction-magnitude encoder should preserve spatial maps before pooling, got {:?}",
        feature_map.shape
    );
    assert_eq!(latents.shape, vec![BATCH_SIZE, 3]);
    assert!(feature_map.data.iter().all(|value| value.is_finite()));
    assert!(latents.data.iter().all(|value| value.is_finite()));
}

#[test]
fn projected_signed_target_bank_separability_is_finite_and_balanced() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let model = ProjectedVisionJepa::new(encoder, projector, target_projector, make_predictor());

    let separability = projected_signed_target_bank_separability_from_base_seed(
        &model,
        PROJECTED_VALIDATION_BASE_SEED,
        2,
    );

    for value in [
        separability.oracle_mrr,
        separability.oracle_top1,
        separability.true_distance,
        separability.max_true_distance,
        separability.nearest_wrong_distance,
        separability.min_nearest_wrong_distance,
        separability.margin,
        separability.min_margin,
        separability.negative_nearest_wrong_distance,
        separability.positive_nearest_wrong_distance,
        separability.slow_nearest_wrong_distance,
        separability.fast_nearest_wrong_distance,
        separability.sign_margin,
        separability.speed_margin,
    ] {
        assert!(value.is_finite());
    }

    assert!((0.0..=1.0).contains(&separability.oracle_mrr));
    assert!((0.0..=1.0).contains(&separability.oracle_top1));
    assert!(separability.true_distance >= 0.0);
    assert!(separability.max_true_distance >= separability.true_distance);
    assert!(separability.nearest_wrong_distance >= 0.0);
    assert!(separability.min_nearest_wrong_distance >= 0.0);
    assert!(separability.margin >= 0.0);
    assert!(separability.min_margin >= 0.0);
    assert!(separability.negative_nearest_wrong_distance >= 0.0);
    assert!(separability.positive_nearest_wrong_distance >= 0.0);
    assert!(separability.slow_nearest_wrong_distance >= 0.0);
    assert!(separability.fast_nearest_wrong_distance >= 0.0);
    assert_eq!(separability.samples, BATCH_SIZE * 2);
    assert_eq!(
        separability.negative_samples + separability.positive_samples,
        separability.samples
    );
    assert_eq!(
        separability.slow_samples + separability.fast_samples,
        separability.samples
    );
    assert_eq!(separability.negative_samples, separability.positive_samples);
    assert_eq!(separability.slow_samples, separability.fast_samples);
}

#[test]
fn projected_signed_prediction_bank_margin_is_finite_and_balanced() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let model = ProjectedVisionJepa::new(encoder, projector, target_projector, make_predictor());

    let margin = projected_signed_prediction_bank_margin_from_base_seed(
        &model,
        PROJECTED_VALIDATION_BASE_SEED,
        2,
    );

    for value in [
        margin.true_distance,
        margin.nearest_wrong_distance,
        margin.margin,
        margin.min_margin,
        margin.positive_margin_rate,
        margin.sign_margin,
        margin.speed_margin,
    ] {
        assert!(value.is_finite());
    }

    assert!(margin.true_distance >= 0.0);
    assert!(margin.nearest_wrong_distance >= 0.0);
    assert!((0.0..=1.0).contains(&margin.positive_margin_rate));
    assert!(
        margin.min_margin <= margin.margin,
        "minimum margin should not exceed mean margin: {:.6} > {:.6}",
        margin.min_margin,
        margin.margin
    );
    assert_eq!(margin.samples, BATCH_SIZE * 2);
    assert_eq!(
        margin.negative_samples + margin.positive_samples,
        margin.samples
    );
    assert_eq!(margin.slow_samples + margin.fast_samples, margin.samples);
    assert_eq!(margin.negative_samples, margin.positive_samples);
    assert_eq!(margin.slow_samples, margin.fast_samples);
}

#[test]
fn projected_signed_prediction_bank_unit_geometry_is_finite_and_bounded() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let model = ProjectedVisionJepa::new(encoder, projector, target_projector, make_predictor());

    let geometry = projected_signed_prediction_bank_unit_geometry_from_base_seed(
        &model,
        PROJECTED_VALIDATION_BASE_SEED,
        2,
    );

    for value in [
        geometry.mrr,
        geometry.top1,
        geometry.true_distance,
        geometry.nearest_wrong_distance,
        geometry.margin,
        geometry.positive_margin_rate,
        geometry.sign_margin,
        geometry.speed_margin,
        geometry.prediction_center_norm,
        geometry.true_target_center_norm,
    ] {
        assert!(value.is_finite());
    }

    assert!((0.25..=1.0).contains(&geometry.mrr));
    assert!((0.0..=1.0).contains(&geometry.top1));
    assert!(geometry.true_distance >= 0.0);
    assert!(geometry.nearest_wrong_distance >= 0.0);
    assert!((0.0..=1.0).contains(&geometry.positive_margin_rate));
    assert!(geometry.prediction_center_norm >= 0.0);
    assert!(geometry.true_target_center_norm >= 0.0);
    assert_eq!(geometry.samples, BATCH_SIZE * 2);
    assert_eq!(geometry.candidates, 4);
}

#[test]
fn projected_signed_prediction_ray_boundary_is_finite_and_bounded() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let model = ProjectedVisionJepa::new(encoder, projector, target_projector, make_predictor());

    let boundary = projected_signed_prediction_ray_boundary_from_base_seed(
        &model,
        PROJECTED_VALIDATION_BASE_SEED,
        2,
    );

    for value in [
        boundary.current_radius,
        boundary.required_radius,
        boundary.upper_radius,
        boundary.radius_margin,
        boundary.radius_shortfall,
        boundary.radius_overshoot,
        boundary.satisfied_rate,
        boundary.infeasible_rate,
    ] {
        assert!(value.is_finite());
    }

    assert!(boundary.current_radius >= 0.0);
    assert!(boundary.required_radius >= 0.0);
    assert!(boundary.upper_radius >= 0.0);
    assert!(boundary.radius_shortfall >= 0.0);
    assert!(boundary.radius_overshoot >= 0.0);
    assert!((0.0..=1.0).contains(&boundary.satisfied_rate));
    assert!((0.0..=1.0).contains(&boundary.infeasible_rate));
    assert!(boundary.finite_upper_samples <= boundary.feasible_samples);
    assert!(boundary.feasible_samples <= boundary.samples);
    assert_eq!(
        boundary.satisfied_by_dx.iter().sum::<usize>(),
        (boundary.satisfied_rate * boundary.samples as f32).round() as usize
    );
    assert_eq!(
        boundary.infeasible_by_dx.iter().sum::<usize>(),
        (boundary.infeasible_rate * boundary.samples as f32).round() as usize
    );
    assert_eq!(
        boundary.satisfied_by_dx.iter().sum::<usize>()
            + boundary.infeasible_by_dx.iter().sum::<usize>()
            + boundary.below_lower_by_dx.iter().sum::<usize>()
            + boundary.upper_overshoot_by_dx.iter().sum::<usize>(),
        boundary.samples
    );
    assert_eq!(boundary.samples, BATCH_SIZE * 2);
    assert_eq!(boundary.candidates, 4);
}

#[test]
fn projected_signed_prediction_geometry_counterfactual_is_finite_and_bounded() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let model = ProjectedVisionJepa::new(encoder, projector, target_projector, make_predictor());

    let counterfactual = projected_signed_prediction_geometry_counterfactual_from_base_seed(
        &model,
        PROJECTED_VALIDATION_BASE_SEED,
        2,
    );

    for metrics in [
        counterfactual.oracle_radius,
        counterfactual.oracle_angle,
        counterfactual.support_global_rescale,
    ] {
        for value in [
            metrics.mrr,
            metrics.top1,
            metrics.margin,
            metrics.positive_margin_rate,
            metrics.sign_margin,
            metrics.speed_margin,
            metrics.norm_ratio,
        ] {
            assert!(value.is_finite());
        }

        assert!((0.25..=1.0).contains(&metrics.mrr));
        assert!((0.0..=1.0).contains(&metrics.top1));
        assert!((0.0..=1.0).contains(&metrics.positive_margin_rate));
        assert!(metrics.norm_ratio >= 0.0);
    }

    assert!(counterfactual.support_norm_ratio.is_finite());
    assert!(counterfactual.support_norm_ratio >= 0.0);
    assert_eq!(counterfactual.support_samples, BATCH_SIZE);
    assert_eq!(counterfactual.query_samples, BATCH_SIZE);
    assert_eq!(counterfactual.candidates, 4);
}

#[test]
fn projected_signed_radial_calibration_report_is_finite_and_bounded() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let model = ProjectedVisionJepa::new(encoder, projector, target_projector, make_predictor());

    let report = projected_signed_radial_calibration_report_from_base_seed(
        &model,
        PROJECTED_VALIDATION_BASE_SEED,
        2,
    );

    for value in [
        report.loss,
        report.prediction_norm,
        report.target_norm,
        report.norm_ratio,
    ] {
        assert!(value.is_finite());
    }

    assert!(report.loss >= 0.0);
    assert!(report.prediction_norm >= 0.0);
    assert!(report.target_norm >= 0.0);
    assert!(report.norm_ratio >= 0.0);
    assert_eq!(report.samples, BATCH_SIZE * 2);
}

#[test]
fn projected_signed_angular_radial_report_is_finite_and_bounded() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let model = ProjectedVisionJepa::new(encoder, projector, target_projector, make_predictor());

    let report = projected_signed_angular_radial_objective_report_from_base_seed(
        &model,
        PROJECTED_VALIDATION_BASE_SEED,
        2,
        SignedAngularRadialObjectiveConfig::default(),
    );

    for value in [
        report.loss,
        report.angular_loss,
        report.radial_loss,
        report.cosine,
        report.prediction_norm,
        report.target_norm,
        report.norm_ratio,
    ] {
        assert!(value.is_finite());
    }

    assert!(report.loss >= 0.0);
    assert!(report.angular_loss >= 0.0);
    assert!(report.radial_loss >= 0.0);
    assert!((-1.0..=1.0).contains(&report.cosine));
    assert!(report.prediction_norm >= 0.0);
    assert!(report.target_norm >= 0.0);
    assert!(report.norm_ratio >= 0.0);
    assert_eq!(report.samples, BATCH_SIZE * 2);
}

#[test]
#[should_panic(expected = "velocity-bank ranking only supports velocity-trail")]
fn projected_velocity_bank_ranking_rejects_non_velocity_trail_task() {
    let encoder = make_frozen_encoder();
    let projector = make_projector();
    let target_projector = projector.clone();
    let model = ProjectedVisionJepa::new(encoder, projector, target_projector, make_predictor());

    let _ = projected_velocity_bank_ranking_from_base_seed(
        &model,
        PROJECTED_VALIDATION_BASE_SEED,
        1,
        TemporalTaskMode::RandomSpeed,
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
fn projector_drift_regularizer_is_half_mean_squared_parameter_drift() {
    let online = Linear::new(
        Tensor::new(vec![2.0, 0.0, -1.0, 3.0], vec![2, 2]),
        Tensor::new(vec![1.0, -2.0], vec![2]),
    );
    let target = Linear::new(
        Tensor::new(vec![1.0, 1.0, -1.0, 5.0], vec![2, 2]),
        Tensor::new(vec![0.0, -4.0], vec![2]),
    );

    let loss = projector_drift_regularizer(&online, &target);

    assert!((loss - (0.5 * 11.0 / 6.0)).abs() < 1e-6);
}

#[test]
fn projector_drift_regularizer_grads_match_online_minus_target_over_parameter_count() {
    let online = Linear::new(
        Tensor::new(vec![2.0, 0.0, -1.0, 3.0], vec![2, 2]),
        Tensor::new(vec![1.0, -2.0], vec![2]),
    );
    let target = Linear::new(
        Tensor::new(vec![1.0, 1.0, -1.0, 5.0], vec![2, 2]),
        Tensor::new(vec![0.0, -4.0], vec![2]),
    );

    let (grad_weight, grad_bias) = projector_drift_regularizer_grads(&online, &target);

    assert_eq!(
        grad_weight,
        Tensor::new(vec![1.0 / 6.0, -1.0 / 6.0, 0.0, -2.0 / 6.0], vec![2, 2])
    );
    assert_eq!(grad_bias, Tensor::new(vec![1.0 / 6.0, 2.0 / 6.0], vec![2]));
}

#[test]
fn projector_drift_regularizer_updates_projector_grads_without_touching_grad_input() {
    let online = Linear::new(
        Tensor::new(vec![2.0, 0.0, -1.0, 3.0], vec![2, 2]),
        Tensor::new(vec![1.0, -2.0], vec![2]),
    );
    let target = Linear::new(
        Tensor::new(vec![1.0, 1.0, -1.0, 5.0], vec![2, 2]),
        Tensor::new(vec![0.0, -4.0], vec![2]),
    );
    let mut grads = LinearGrads {
        grad_input: Tensor::new(vec![9.0, 8.0], vec![1, 2]),
        grad_weight: Tensor::zeros(vec![2, 2]),
        grad_bias: Tensor::zeros(vec![2]),
    };

    jepra_core::add_projector_drift_regularizer_grad(&mut grads, &online, &target, 3.0);

    assert_eq!(grads.grad_input, Tensor::new(vec![9.0, 8.0], vec![1, 2]));
    assert_eq!(
        grads.grad_weight,
        Tensor::new(vec![0.5, -0.5, 0.0, -1.0], vec![2, 2])
    );
    assert_eq!(grads.grad_bias, Tensor::new(vec![0.5, 1.0], vec![2]));
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
fn projected_frozen_vs_trainable_projection_protocol_has_stable_behavior() {
    assert_projected_frozen_vs_trainable_protocol(
        "base projected encoder",
        30_000u64,
        22_500u64,
        0.02f32,
        0.004f32,
        CompactEncoderMode::Disabled,
        make_frozen_encoder,
    );
}

#[test]
fn projected_compact_frozen_vs_trainable_projection_protocol_has_stable_behavior() {
    assert_projected_frozen_vs_trainable_protocol(
        "compact projected encoder",
        30_010u64,
        22_520u64,
        0.02f32,
        0.004f32,
        CompactEncoderMode::Base,
        make_compact_frozen_encoder,
    );
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
fn projected_target_projector_warmup_schedule_matches_frozen_and_trainable_protocols() {
    let config = TemporalRunConfig {
        train_base_seed: 11_000u64,
        total_steps: 2,
        log_every: 1,
        encoder_learning_rate: 0.0,
        temporal_task_mode: TemporalTaskMode::RandomSpeed,
        compact_encoder_mode: CompactEncoderMode::Disabled,
        predictor_mode: PredictorMode::Baseline,
        residual_delta_scale: 1.0,
        projector_drift_weight: 0.0,
        signed_margin_weight: 0.0,
        signed_margin_config: SignedMarginObjectiveConfig::default(),
        signed_bank_softmax_weight: 0.0,
        signed_bank_softmax_config: SignedBankSoftmaxObjectiveConfig::default(),
        signed_radial_weight: 0.0,
        signed_angular_radial_weight: 0.0,
        signed_angular_radial_config: SignedAngularRadialObjectiveConfig::default(),
        target_projection_momentum: 0.0,
        target_projection_momentum_start: 1.0,
        target_projection_momentum_end: 0.0,
        target_projection_momentum_warmup_steps: 2,
    };
    let step1_momentum = config.target_projection_momentum_at_step(1);
    let step2_momentum = config.target_projection_momentum_at_step(2);

    assert!(
        step1_momentum / config.target_projection_momentum_start > 0.49
            && step1_momentum / config.target_projection_momentum_start < 0.51,
        "warmup step 1 should land near the midpoint ratio: start {:.6}, step1 {:.6}",
        config.target_projection_momentum_start,
        step1_momentum
    );
    assert!(
        step2_momentum <= step1_momentum * 1e-3,
        "warmup should collapse momentum relative to step 1 by the end: step1 {:.6}, step2 {:.6}",
        step1_momentum,
        step2_momentum
    );

    let mut frozen = ProjectedVisionJepa::new(
        make_frozen_encoder(),
        make_projector(),
        make_projector(),
        make_predictor(),
    );
    let mut trainable = frozen.clone();
    let mut previous_frozen_drift = None;

    for step in 1..=config.total_steps {
        let scheduled_momentum = config.target_projection_momentum_at_step(step);
        let (x_t, x_t1) = make_train_batch(config.train_base_seed, step as u64);

        frozen.set_target_projection_momentum(scheduled_momentum);
        trainable.set_target_projection_momentum(scheduled_momentum);

        let frozen_losses =
            frozen.step(&x_t, &x_t1, REGULARIZER_WEIGHT, PREDICTOR_LR, PROJECTOR_LR);
        let trainable_losses = trainable.step_with_trainable_encoder(
            &x_t,
            &x_t1,
            REGULARIZER_WEIGHT,
            PREDICTOR_LR,
            PROJECTOR_LR,
            config.encoder_learning_rate,
        );

        assert!(
            symmetric_relative_difference(frozen_losses.2, trainable_losses.2) < 1e-6,
            "scheduled warmup total loss drifted across frozen/trainable protocols at step {}: frozen {:.6}, trainable {:.6}",
            step,
            frozen_losses.2,
            trainable_losses.2
        );
        assert!(
            symmetric_relative_difference(
                frozen.target_projection_drift(),
                trainable.target_projection_drift()
            ) < 1e-6,
            "scheduled warmup target drift diverged across protocols at step {}: frozen {:.6}, trainable {:.6}",
            step,
            frozen.target_projection_drift(),
            trainable.target_projection_drift()
        );

        if let Some(previous_drift) = previous_frozen_drift {
            assert!(
                frozen.target_projection_drift() <= previous_drift * 1e-3,
                "end-of-warmup target drift should collapse relative to the prior step: previous {:.6}, current {:.6}",
                previous_drift,
                frozen.target_projection_drift()
            );
        } else {
            assert!(
                frozen.target_projection_drift() > 0.0,
                "mid-warmup step should leave non-zero target drift"
            );
        }

        previous_frozen_drift = Some(frozen.target_projection_drift());
    }

    assert!(
        tensor_symmetric_relative_difference(&frozen.projector.weight, &trainable.projector.weight)
            < 1e-6,
        "projector weights diverged across frozen/trainable scheduled warmup runs"
    );
    assert!(
        tensor_symmetric_relative_difference(
            &frozen.target_projector.weight,
            &trainable.target_projector.weight
        ) < 1e-6,
        "target projector weights diverged across frozen/trainable scheduled warmup runs"
    );
    assert!(
        tensor_symmetric_relative_difference(
            &frozen.target_projector.bias,
            &trainable.target_projector.bias
        ) < 1e-6,
        "target projector biases diverged across frozen/trainable scheduled warmup runs"
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
fn projected_momentum_sweep_trajectory_is_stable_and_expected_monotonic() {
    let train_base_seed = 11_000u64;
    let predictor_seed = 22_300u64;
    let steps = 60usize;
    let stability_tolerance = 1e-7f32;
    let momenta = [1.0f32, 0.5f32, 0.0f32];

    let run_protocol = |momentum: f32| {
        let mut model = ProjectedVisionJepa::new(
            make_frozen_encoder(),
            make_projector(),
            make_projector(),
            make_predictor_with_seed(predictor_seed),
        )
        .with_target_projection_momentum(momentum);

        let config = TemporalRunConfig {
            train_base_seed,
            total_steps: steps,
            log_every: steps,
            encoder_learning_rate: 0.0,
            temporal_task_mode: TemporalTaskMode::RandomSpeed,
            compact_encoder_mode: CompactEncoderMode::Disabled,
            predictor_mode: PredictorMode::Baseline,
            residual_delta_scale: 1.0,
            projector_drift_weight: 0.0,
            signed_margin_weight: 0.0,
            signed_margin_config: SignedMarginObjectiveConfig::default(),
            signed_bank_softmax_weight: 0.0,
            signed_bank_softmax_config: SignedBankSoftmaxObjectiveConfig::default(),
            signed_radial_weight: 0.0,
            signed_angular_radial_weight: 0.0,
            signed_angular_radial_config: SignedAngularRadialObjectiveConfig::default(),
            target_projection_momentum: momentum,
            target_projection_momentum_start: momentum,
            target_projection_momentum_end: momentum,
            target_projection_momentum_warmup_steps: 0,
        };

        let (probe_t, probe_t1) = make_train_batch(train_base_seed, 0);
        let initial_train_loss = model.losses(&probe_t, &probe_t1, REGULARIZER_WEIGHT).2;
        let initial_validation_loss = projected_validation_losses_model(&model).2;
        let mut train_trajectory = Vec::<f32>::new();
        let mut validation_trajectory = Vec::<f32>::new();
        let mut drift_trajectory = Vec::<f32>::new();

        let summary = run_temporal_experiment_with_summary(
            config,
            &mut model,
            initial_train_loss,
            initial_validation_loss,
            |model, step, _| {
                let (x_t, x_t1) = make_train_batch(train_base_seed, step as u64);
                let (_, _, train_loss) =
                    model.step(&x_t, &x_t1, REGULARIZER_WEIGHT, PREDICTOR_LR, PROJECTOR_LR);
                let validation_loss = projected_validation_losses_model(model).2;

                train_trajectory.push(train_loss);
                validation_trajectory.push(validation_loss);
                drift_trajectory.push(model.target_projection_drift());

                train_loss
            },
            |model| projected_validation_losses_model(model).2,
        );

        assert_temporal_experiment_improved(
            &format!("projected sweep momentum {momentum}"),
            summary.initial_train_loss,
            summary.final_train_loss,
            summary.initial_validation_loss,
            summary.final_validation_loss,
            PROJECTED_TRAIN_LOSS_MAX_REDUCTION_RATIO,
            PROJECTED_VALIDATION_LOSS_MAX_REDUCTION_RATIO,
        );

        assert_eq!(train_trajectory.len(), steps);
        assert_eq!(validation_trajectory.len(), steps);
        assert_eq!(drift_trajectory.len(), steps);

        (train_trajectory, validation_trajectory, drift_trajectory)
    };

    let assert_trajectory_stability = |name: &str, lhs: &[f32], rhs: &[f32]| {
        assert_eq!(
            lhs.len(),
            rhs.len(),
            "{name} trajectory lengths drifted under identical seeds: {} vs {}",
            lhs.len(),
            rhs.len()
        );

        for (step, (lhs_value, rhs_value)) in lhs.iter().zip(rhs.iter()).enumerate() {
            assert!(
                (lhs_value - rhs_value).abs() < stability_tolerance,
                "{name} trajectory diverged at step {}: {:.9} vs {:.9}",
                step + 1,
                lhs_value,
                rhs_value
            );
        }
    };

    for momentum in momenta {
        let (train_a, validation_a, drift_a) = run_protocol(momentum);
        let (train_b, validation_b, drift_b) = run_protocol(momentum);

        assert_trajectory_stability(&format!("train (momentum {momentum})"), &train_a, &train_b);
        assert_trajectory_stability(
            &format!("validation (momentum {momentum})"),
            &validation_a,
            &validation_b,
        );

        if momentum == 0.0f32 {
            assert_trajectory_stability(
                &format!("target drift (momentum {momentum})"),
                &drift_a,
                &drift_b,
            );
            for (step, (&lhs_drift, &rhs_drift)) in drift_a.iter().zip(drift_b.iter()).enumerate() {
                assert!(
                    lhs_drift < 1e-6,
                    "target drift should stay near zero for momentum 0 at step {}: {:.6}",
                    step + 1,
                    lhs_drift
                );
                assert!(
                    rhs_drift < 1e-6,
                    "target drift should stay near zero for momentum 0 at step {}: {:.6}",
                    step + 1,
                    rhs_drift
                );
            }
        }
    }
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
