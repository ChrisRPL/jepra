#[path = "support/projected_temporal.rs"]
mod projected_temporal;
#[path = "support/temporal_vision.rs"]
mod temporal_vision;

use jepra_core::{
    BottleneckPredictor, Linear, Predictor, PredictorModule, ProjectedVisionJepa,
    ResidualBottleneckPredictor, SignedMarginObjectiveReport, Tensor, projection_stats,
    representation_stats,
};
use projected_temporal::{
    PROJECTED_TRAIN_LOSS_MAX_REDUCTION_RATIO, PROJECTED_VALIDATION_BASE_SEED,
    PROJECTED_VALIDATION_BATCHES, PROJECTED_VALIDATION_LOSS_MAX_REDUCTION_RATIO,
    ProjectedSignedObjectiveErrorBreakdown, ProjectedSignedPredictionBankMargin,
    ProjectedSignedTargetBankSeparability, ProjectedSignedVelocityBankBreakdown,
    ProjectedVelocityBankRanking, projected_signed_margin_objective_loss_and_grad,
    projected_signed_margin_objective_report_from_base_seed,
    projected_signed_objective_error_breakdown_from_base_seed,
    projected_signed_prediction_bank_margin_from_base_seed,
    projected_signed_target_bank_separability_from_base_seed,
    projected_signed_velocity_bank_breakdown_from_base_seed,
    projected_validation_batch_losses_from_base_seed_for_task,
    projected_velocity_bank_ranking_from_base_seed,
};
use temporal_vision::{
    CompactEncoderMode, PredictorMode, TemporalTaskMode, assert_required_motion_modes_for_task,
    assert_temporal_contract, assert_temporal_experiment_improved, make_compact_frozen_encoder,
    make_compact_frozen_encoder_stronger, make_frozen_encoder, make_train_batch_for_config,
    make_validation_batch_for_config, make_validation_batch_with_required_motion_modes_for_config,
    print_batch_summary_for_task, print_motion_mode_summary_for_task, print_representation_stats,
};

const PROJECTION_DIM: usize = 4;
const TRAIN_BASE_SEED: u64 = 11_000;
const NUM_STEPS: usize = 300;
const LOG_EVERY: usize = 25;
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

fn make_bottleneck_predictor() -> BottleneckPredictor {
    BottleneckPredictor::new(
        Linear::randn(PROJECTION_DIM, 8, 0.1, 21_100),
        Linear::randn(8, 2, 0.1, 21_101),
        Linear::randn(2, PROJECTION_DIM, 0.1, 21_102),
    )
}

fn make_residual_bottleneck_predictor(residual_delta_scale: f32) -> ResidualBottleneckPredictor {
    ResidualBottleneckPredictor::new_scaled(
        BottleneckPredictor::new(
            Linear::randn(PROJECTION_DIM, 8, 0.1, 21_100),
            Linear::randn(8, 2, 0.1, 21_101),
            Linear::new(
                Tensor::zeros(vec![2, PROJECTION_DIM]),
                Tensor::zeros(vec![PROJECTION_DIM]),
            ),
        ),
        residual_delta_scale,
    )
}

fn reduction_thresholds_for_run_config(
    run_config: temporal_vision::TemporalRunConfig,
) -> (f32, f32) {
    match (run_config.compact_encoder_mode, run_config.predictor_mode) {
        (CompactEncoderMode::Disabled, PredictorMode::Baseline) => (
            PROJECTED_TRAIN_LOSS_MAX_REDUCTION_RATIO,
            PROJECTED_VALIDATION_LOSS_MAX_REDUCTION_RATIO,
        ),
        _ => (1.0, 1.0),
    }
}

fn main() {
    let run_config =
        temporal_vision::TemporalRunConfig::from_args(TRAIN_BASE_SEED, NUM_STEPS, LOG_EVERY, 0.0);

    println!(
        "temporal run config | train_base_seed {} | steps {} | log_every {}",
        run_config.train_base_seed, run_config.total_steps, run_config.log_every
    );
    println!(
        "temporal run config | target projection momentum {} -> {} | warmup {} steps",
        run_config.target_projection_momentum_start,
        run_config.target_projection_momentum_end,
        run_config.target_projection_momentum_warmup_steps
    );
    println!(
        "temporal run config | task {}",
        run_config.temporal_task_mode.as_str()
    );
    println!(
        "temporal run config | encoder variant {}",
        run_config.compact_encoder_mode.as_str()
    );
    println!(
        "temporal run config | predictor mode {}",
        run_config.predictor_mode.as_str()
    );
    println!(
        "temporal run config | residual delta scale {}",
        run_config.residual_delta_scale
    );
    println!(
        "temporal run config | projector drift weight {}",
        run_config.projector_drift_weight
    );
    println!(
        "temporal run config | signed margin weight {} | bank_gap {} | sign_gap {} | speed_gap {} | bank_weight {} | sign_weight {} | speed_weight {}",
        run_config.signed_margin_weight,
        run_config.signed_margin_config.bank_gap,
        run_config.signed_margin_config.sign_gap,
        run_config.signed_margin_config.speed_gap,
        run_config.signed_margin_config.bank_weight,
        run_config.signed_margin_config.sign_weight,
        run_config.signed_margin_config.speed_weight
    );

    match run_config.predictor_mode {
        PredictorMode::Baseline => run_with_predictor(run_config, make_predictor()),
        PredictorMode::Bottleneck => run_with_predictor(run_config, make_bottleneck_predictor()),
        PredictorMode::ResidualBottleneck => run_with_predictor(
            run_config,
            make_residual_bottleneck_predictor(run_config.residual_delta_scale),
        ),
    }
}

fn run_with_predictor<P>(run_config: temporal_vision::TemporalRunConfig, predictor: P)
where
    P: PredictorModule,
{
    let (train_probe_t, train_probe_t1) = make_train_batch_for_config(run_config, 0);
    let (train_probe_next_t, train_probe_next_t1) = make_train_batch_for_config(run_config, 1);
    let (val_probe_t, val_probe_t1) =
        make_validation_batch_for_config(run_config, PROJECTED_VALIDATION_BASE_SEED, 0);
    let (mixed_val_probe_t, mixed_val_probe_t1, mixed_val_probe_seed) =
        make_validation_batch_with_required_motion_modes_for_config(
            run_config,
            PROJECTED_VALIDATION_BASE_SEED,
            1,
        );

    assert_temporal_contract(&train_probe_t, &train_probe_t1);
    assert_temporal_contract(&train_probe_next_t, &train_probe_next_t1);
    assert_temporal_contract(&val_probe_t, &val_probe_t1);
    assert_temporal_contract(&mixed_val_probe_t, &mixed_val_probe_t1);

    assert_ne!(train_probe_t.data, train_probe_next_t.data);
    assert_ne!(train_probe_t.data, val_probe_t.data);
    assert_ne!(val_probe_t.data, mixed_val_probe_t.data);
    assert_required_motion_modes_for_task(
        run_config.temporal_task_mode,
        &mixed_val_probe_t,
        &mixed_val_probe_t1,
    );

    print_batch_summary_for_task(
        "train probe",
        run_config.temporal_task_mode,
        &train_probe_t,
        &train_probe_t1,
    );
    print_batch_summary_for_task(
        "validation probe",
        run_config.temporal_task_mode,
        &val_probe_t,
        &val_probe_t1,
    );
    print_batch_summary_for_task(
        "validation mixed probe",
        run_config.temporal_task_mode,
        &mixed_val_probe_t,
        &mixed_val_probe_t1,
    );
    print_motion_mode_summary_for_task(
        "validation mixed probe",
        mixed_val_probe_seed,
        run_config.temporal_task_mode,
        &mixed_val_probe_t,
        &mixed_val_probe_t1,
    );

    let encoder = match run_config.compact_encoder_mode {
        CompactEncoderMode::Disabled => make_frozen_encoder(),
        CompactEncoderMode::Base => make_compact_frozen_encoder(),
        CompactEncoderMode::Stronger => make_compact_frozen_encoder_stronger(),
    };
    let online_projector = make_projector();
    let target_projector = online_projector.clone();
    let mut model =
        ProjectedVisionJepa::new(encoder, online_projector, target_projector, predictor)
            .with_target_projection_momentum(run_config.target_projection_momentum_at_step(1));

    let _initial_mixed_val_z_t = model.encode(&mixed_val_probe_t);
    let initial_projection_t = model.project_latent(&train_probe_t);
    let initial_prediction = model.predict_next_projection(&train_probe_t);
    let initial_target = model.target_projection(&train_probe_t1);
    let (initial_train_prediction_loss, initial_train_regularizer_loss, initial_train_total_loss) =
        model.losses(&train_probe_t, &train_probe_t1, REGULARIZER_WEIGHT);
    let (initial_val_prediction_loss, initial_val_regularizer_loss, initial_val_total_loss) =
        projected_validation_batch_losses_from_base_seed_for_task(
            &model,
            REGULARIZER_WEIGHT,
            PROJECTED_VALIDATION_BASE_SEED,
            PROJECTED_VALIDATION_BATCHES,
            run_config.temporal_task_mode,
        );
    let initial_projection_drift = model.target_projection_drift();
    let (initial_projection_mean_abs, initial_projection_var_mean) =
        projection_stats(&initial_projection_t);
    let initial_velocity_bank_ranking = maybe_projected_velocity_bank_ranking(&model, run_config);
    let initial_signed_velocity_bank_breakdown =
        maybe_projected_signed_velocity_bank_breakdown(&model, run_config);
    let initial_signed_target_bank_separability =
        maybe_projected_signed_target_bank_separability(&model, run_config);
    let initial_signed_prediction_bank_margin =
        maybe_projected_signed_prediction_bank_margin(&model, run_config);
    let initial_signed_margin_objective_report =
        maybe_projected_signed_margin_objective_report(&model, run_config);

    println!(
        "initial | projection sample0 {:?} | target {:?}",
        &initial_projection_t.data[0..PROJECTION_DIM],
        &initial_target.data[0..PROJECTION_DIM]
    );
    println!(
        "initial | proj mean_abs {:.6} | var_mean {:.6}",
        initial_projection_mean_abs, initial_projection_var_mean
    );
    print_representation_stats(
        "initial prediction",
        &representation_stats(&initial_prediction),
    );
    print_representation_stats("initial target", &representation_stats(&initial_target));
    println!(
        "initial | train pred {:.6} | reg {:.6} | total {:.6}",
        initial_train_prediction_loss, initial_train_regularizer_loss, initial_train_total_loss
    );
    println!(
        "initial | val pred {:.6} | reg {:.6} | total {:.6}",
        initial_val_prediction_loss, initial_val_regularizer_loss, initial_val_total_loss
    );
    println!("initial | target drift {:.6}", initial_projection_drift);
    print_velocity_bank_ranking("initial", initial_velocity_bank_ranking);
    print_signed_velocity_bank_breakdown("initial", initial_signed_velocity_bank_breakdown);
    print_signed_target_bank_separability("initial", initial_signed_target_bank_separability);
    print_signed_prediction_bank_margin("initial", initial_signed_prediction_bank_margin);
    print_signed_margin_objective_report("initial", initial_signed_margin_objective_report);

    let experiment_summary = temporal_vision::run_temporal_experiment_with_summary(
        run_config,
        &mut model,
        initial_train_total_loss,
        initial_val_total_loss,
        |model, step, should_log| {
            let (x_t, x_t1) = make_train_batch_for_config(run_config, step as u64);
            let momentum = run_config.target_projection_momentum_at_step(step);
            model.set_target_projection_momentum(momentum);
            let signed_margin_step = maybe_projected_signed_margin_objective_loss_and_grad(
                model, run_config, &x_t, &x_t1,
            );
            let signed_margin_extra_loss = signed_margin_step
                .as_ref()
                .map(|(report, _)| run_config.signed_margin_weight * report.weighted_loss)
                .unwrap_or(0.0);
            let signed_margin_extra_grad = signed_margin_step
                .as_ref()
                .map(|(_, grad)| scale_tensor(grad, run_config.signed_margin_weight));
            let (prediction_loss, regularizer_loss, projector_drift_loss, total_loss) =
                if run_config.encoder_learning_rate > 0.0 {
                    model.step_with_extra_prediction_grad(
                        &x_t,
                        &x_t1,
                        REGULARIZER_WEIGHT,
                        run_config.projector_drift_weight,
                        signed_margin_extra_loss,
                        signed_margin_extra_grad.as_ref(),
                        PREDICTOR_LR,
                        PROJECTOR_LR,
                        run_config.encoder_learning_rate,
                    )
                } else {
                    model.step_with_extra_prediction_grad(
                        &x_t,
                        &x_t1,
                        REGULARIZER_WEIGHT,
                        run_config.projector_drift_weight,
                        signed_margin_extra_loss,
                        signed_margin_extra_grad.as_ref(),
                        PREDICTOR_LR,
                        PROJECTOR_LR,
                        0.0,
                    )
                };

            if should_log {
                let current_momentum = model.target_projection_momentum();
                let (val_prediction_loss, val_regularizer_loss, val_total_loss) =
                    projected_validation_batch_losses_from_base_seed_for_task(
                        model,
                        REGULARIZER_WEIGHT,
                        PROJECTED_VALIDATION_BASE_SEED,
                        PROJECTED_VALIDATION_BATCHES,
                        run_config.temporal_task_mode,
                    );

                temporal_vision::print_projected_temporal_train_val_metrics(
                    step,
                    prediction_loss,
                    regularizer_loss,
                    total_loss,
                    current_momentum,
                    model.target_projection_drift(),
                    val_prediction_loss,
                    val_regularizer_loss,
                    val_total_loss,
                );
                if run_config.projector_drift_weight > 0.0 {
                    println!(
                        "step {:03} | projector drift regularizer {:.6}",
                        step, projector_drift_loss
                    );
                }
                if let Some((report, _)) = signed_margin_step {
                    print_signed_margin_objective_report(
                        &format!("step {:03} train", step),
                        Some(report),
                    );
                }
            }

            total_loss
        },
        |model| {
            projected_validation_batch_losses_from_base_seed_for_task(
                model,
                REGULARIZER_WEIGHT,
                PROJECTED_VALIDATION_BASE_SEED,
                PROJECTED_VALIDATION_BATCHES,
                run_config.temporal_task_mode,
            )
            .2
        },
    );

    let final_projection_t = model.project_latent(&train_probe_t);
    let final_pred = model.predict_next_projection(&train_probe_t);
    let final_target = model.target_projection(&train_probe_t1);
    let final_projection_drift = model.target_projection_drift();
    let (final_train_prediction_loss, final_train_regularizer_loss, final_train_total_loss) =
        model.losses(&train_probe_t, &train_probe_t1, REGULARIZER_WEIGHT);
    let (final_val_prediction_loss, final_val_regularizer_loss, final_val_total_loss) =
        projected_validation_batch_losses_from_base_seed_for_task(
            &model,
            REGULARIZER_WEIGHT,
            PROJECTED_VALIDATION_BASE_SEED,
            PROJECTED_VALIDATION_BATCHES,
            run_config.temporal_task_mode,
        );
    let (final_projection_mean_abs, final_projection_var_mean) =
        projection_stats(&final_projection_t);
    let final_velocity_bank_ranking = maybe_projected_velocity_bank_ranking(&model, run_config);
    let final_signed_velocity_bank_breakdown =
        maybe_projected_signed_velocity_bank_breakdown(&model, run_config);
    let final_signed_target_bank_separability =
        maybe_projected_signed_target_bank_separability(&model, run_config);
    let final_signed_prediction_bank_margin =
        maybe_projected_signed_prediction_bank_margin(&model, run_config);
    let final_signed_objective_error_breakdown =
        maybe_projected_signed_objective_error_breakdown(&model, run_config);
    let final_signed_margin_objective_report =
        maybe_projected_signed_margin_objective_report(&model, run_config);
    let (train_reduction_threshold, validation_reduction_threshold) =
        reduction_thresholds_for_run_config(run_config);

    assert_temporal_experiment_improved(
        "projected",
        initial_train_total_loss,
        final_train_total_loss,
        initial_val_total_loss,
        final_val_total_loss,
        train_reduction_threshold,
        validation_reduction_threshold,
    );

    println!(
        "projected run summary | steps {} | train {:.6} -> {:.6} (Δ {:+.6}, improved={}) | val {:.6} -> {:.6} (Δ {:+.6}, improved={}) | target drift {:.6} -> {:.6} (Δ {:+.6})",
        experiment_summary.config.total_steps,
        experiment_summary.initial_train_loss,
        experiment_summary.final_train_loss,
        experiment_summary.train_delta(),
        experiment_summary.train_improved(),
        experiment_summary.initial_validation_loss,
        experiment_summary.final_validation_loss,
        experiment_summary.validation_delta(),
        experiment_summary.validation_improved(),
        initial_projection_drift,
        final_projection_drift,
        final_projection_drift - initial_projection_drift,
    );

    println!(
        "\nfinal | projection sample0 {:?} | pred {:?} | target {:?}",
        &final_projection_t.data[0..PROJECTION_DIM],
        &final_pred.data[0..PROJECTION_DIM],
        &final_target.data[0..PROJECTION_DIM]
    );
    println!(
        "final | proj mean_abs {:.6} | var_mean {:.6}",
        final_projection_mean_abs, final_projection_var_mean
    );
    print_representation_stats("final prediction", &representation_stats(&final_pred));
    print_representation_stats("final target", &representation_stats(&final_target));
    println!(
        "final | train pred {:.6} | reg {:.6} | total {:.6}",
        final_train_prediction_loss, final_train_regularizer_loss, final_train_total_loss
    );
    println!(
        "final | val pred {:.6} | reg {:.6} | total {:.6}",
        final_val_prediction_loss, final_val_regularizer_loss, final_val_total_loss
    );
    println!("final | target drift {:.6}", final_projection_drift);
    print_velocity_bank_ranking("final", final_velocity_bank_ranking);
    print_signed_velocity_bank_breakdown("final", final_signed_velocity_bank_breakdown);
    print_signed_target_bank_separability("final", final_signed_target_bank_separability);
    print_signed_prediction_bank_margin("final", final_signed_prediction_bank_margin);
    print_signed_objective_error_breakdown("final", final_signed_objective_error_breakdown);
    print_signed_margin_objective_report("final", final_signed_margin_objective_report);
}

fn scale_tensor(tensor: &Tensor, scale: f32) -> Tensor {
    Tensor {
        data: tensor.data.iter().map(|value| value * scale).collect(),
        shape: tensor.shape.clone(),
    }
}

fn maybe_projected_velocity_bank_ranking<P>(
    model: &ProjectedVisionJepa<P>,
    run_config: temporal_vision::TemporalRunConfig,
) -> Option<ProjectedVelocityBankRanking>
where
    P: PredictorModule,
{
    if !matches!(
        run_config.temporal_task_mode,
        TemporalTaskMode::VelocityTrail | TemporalTaskMode::SignedVelocityTrail
    ) {
        return None;
    }

    Some(projected_velocity_bank_ranking_from_base_seed(
        model,
        PROJECTED_VALIDATION_BASE_SEED,
        PROJECTED_VALIDATION_BATCHES,
        run_config.temporal_task_mode,
    ))
}

fn print_velocity_bank_ranking(tag: &str, ranking: Option<ProjectedVelocityBankRanking>) {
    if let Some(ranking) = ranking {
        println!(
            "{} | velocity bank mrr {:.6} | top1 {:.6} | mean_rank {:.6} | samples {} | candidates {}",
            tag, ranking.mrr, ranking.top1, ranking.mean_rank, ranking.samples, ranking.candidates
        );
    }
}

fn maybe_projected_signed_velocity_bank_breakdown<P>(
    model: &ProjectedVisionJepa<P>,
    run_config: temporal_vision::TemporalRunConfig,
) -> Option<ProjectedSignedVelocityBankBreakdown>
where
    P: PredictorModule,
{
    if run_config.temporal_task_mode != TemporalTaskMode::SignedVelocityTrail {
        return None;
    }

    Some(projected_signed_velocity_bank_breakdown_from_base_seed(
        model,
        PROJECTED_VALIDATION_BASE_SEED,
        PROJECTED_VALIDATION_BATCHES,
    ))
}

fn print_signed_velocity_bank_breakdown(
    tag: &str,
    breakdown: Option<ProjectedSignedVelocityBankBreakdown>,
) {
    if let Some(breakdown) = breakdown {
        println!(
            "{} | signed velocity bank neg_mrr {:.6} | pos_mrr {:.6} | slow_mrr {:.6} | fast_mrr {:.6} | sign_top1 {:.6} | speed_top1 {:.6} | samples {} | true_neg_best_neg {} | true_neg_best_pos {} | true_pos_best_neg {} | true_pos_best_pos {} | true_slow_best_slow {} | true_slow_best_fast {} | true_fast_best_slow {} | true_fast_best_fast {}",
            tag,
            breakdown.negative_mrr,
            breakdown.positive_mrr,
            breakdown.slow_mrr,
            breakdown.fast_mrr,
            breakdown.sign_top1,
            breakdown.speed_top1,
            breakdown.samples,
            breakdown.true_neg_best_neg,
            breakdown.true_neg_best_pos,
            breakdown.true_pos_best_neg,
            breakdown.true_pos_best_pos,
            breakdown.true_slow_best_slow,
            breakdown.true_slow_best_fast,
            breakdown.true_fast_best_slow,
            breakdown.true_fast_best_fast
        );
    }
}

fn maybe_projected_signed_target_bank_separability<P>(
    model: &ProjectedVisionJepa<P>,
    run_config: temporal_vision::TemporalRunConfig,
) -> Option<ProjectedSignedTargetBankSeparability>
where
    P: PredictorModule,
{
    if run_config.temporal_task_mode != TemporalTaskMode::SignedVelocityTrail {
        return None;
    }

    Some(projected_signed_target_bank_separability_from_base_seed(
        model,
        PROJECTED_VALIDATION_BASE_SEED,
        PROJECTED_VALIDATION_BATCHES,
    ))
}

fn print_signed_target_bank_separability(
    tag: &str,
    separability: Option<ProjectedSignedTargetBankSeparability>,
) {
    if let Some(separability) = separability {
        println!(
            "{} | target bank separability oracle_mrr {:.6} | top1 {:.6} | true_dist {:.6} | true_dist_max {:.6} | nearest_wrong {:.6} | nearest_wrong_min {:.6} | margin {:.6} | margin_min {:.6} | neg_nearest_wrong {:.6} | pos_nearest_wrong {:.6} | slow_nearest_wrong {:.6} | fast_nearest_wrong {:.6} | sign_margin {:.6} | speed_margin {:.6} | samples {}",
            tag,
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
            separability.samples,
        );
    }
}

fn maybe_projected_signed_prediction_bank_margin<P>(
    model: &ProjectedVisionJepa<P>,
    run_config: temporal_vision::TemporalRunConfig,
) -> Option<ProjectedSignedPredictionBankMargin>
where
    P: PredictorModule,
{
    if run_config.temporal_task_mode != TemporalTaskMode::SignedVelocityTrail {
        return None;
    }

    Some(projected_signed_prediction_bank_margin_from_base_seed(
        model,
        PROJECTED_VALIDATION_BASE_SEED,
        PROJECTED_VALIDATION_BATCHES,
    ))
}

fn print_signed_prediction_bank_margin(
    tag: &str,
    margin: Option<ProjectedSignedPredictionBankMargin>,
) {
    if let Some(margin) = margin {
        println!(
            "{} | signed prediction bank margin true_distance {:.6} | nearest_wrong_distance {:.6} | margin {:.6} | min_margin {:.6} | positive_margin_rate {:.6} | sign_margin {:.6} | speed_margin {:.6} | samples {}",
            tag,
            margin.true_distance,
            margin.nearest_wrong_distance,
            margin.margin,
            margin.min_margin,
            margin.positive_margin_rate,
            margin.sign_margin,
            margin.speed_margin,
            margin.samples,
        );
    }
}

fn maybe_projected_signed_objective_error_breakdown<P>(
    model: &ProjectedVisionJepa<P>,
    run_config: temporal_vision::TemporalRunConfig,
) -> Option<ProjectedSignedObjectiveErrorBreakdown>
where
    P: PredictorModule,
{
    if run_config.temporal_task_mode != TemporalTaskMode::SignedVelocityTrail {
        return None;
    }

    Some(projected_signed_objective_error_breakdown_from_base_seed(
        model,
        PROJECTED_VALIDATION_BASE_SEED,
        PROJECTED_VALIDATION_BATCHES,
    ))
}

fn maybe_projected_signed_margin_objective_loss_and_grad<P>(
    model: &ProjectedVisionJepa<P>,
    run_config: temporal_vision::TemporalRunConfig,
    x_t: &Tensor,
    x_t1: &Tensor,
) -> Option<(SignedMarginObjectiveReport, Tensor)>
where
    P: PredictorModule,
{
    if run_config.signed_margin_weight == 0.0 {
        return None;
    }
    assert_eq!(
        run_config.temporal_task_mode,
        TemporalTaskMode::SignedVelocityTrail,
        "signed margin objective is only supported for signed-velocity-trail projected runs"
    );

    Some(projected_signed_margin_objective_loss_and_grad(
        model,
        x_t,
        x_t1,
        run_config.signed_margin_config,
    ))
}

fn maybe_projected_signed_margin_objective_report<P>(
    model: &ProjectedVisionJepa<P>,
    run_config: temporal_vision::TemporalRunConfig,
) -> Option<SignedMarginObjectiveReport>
where
    P: PredictorModule,
{
    if run_config.signed_margin_weight == 0.0 {
        return None;
    }
    assert_eq!(
        run_config.temporal_task_mode,
        TemporalTaskMode::SignedVelocityTrail,
        "signed margin objective is only supported for signed-velocity-trail projected runs"
    );

    Some(projected_signed_margin_objective_report_from_base_seed(
        model,
        PROJECTED_VALIDATION_BASE_SEED,
        PROJECTED_VALIDATION_BATCHES,
        run_config.signed_margin_config,
    ))
}

fn print_signed_objective_error_breakdown(
    tag: &str,
    breakdown: Option<ProjectedSignedObjectiveErrorBreakdown>,
) {
    if let Some(breakdown) = breakdown {
        println!(
            "{} | signed objective error breakdown all_loss {:.6} | dx_neg2_loss {:.6} | dx_neg1_loss {:.6} | dx_pos1_loss {:.6} | dx_pos2_loss {:.6} | neg_loss {:.6} | pos_loss {:.6} | slow_loss {:.6} | fast_loss {:.6} | sign_gap {:.6} | speed_gap {:.6} | samples {} | dx_neg2_samples {} | dx_neg1_samples {} | dx_pos1_samples {} | dx_pos2_samples {}",
            tag,
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
            breakdown.samples,
            breakdown.dx_neg2_samples,
            breakdown.dx_neg1_samples,
            breakdown.dx_pos1_samples,
            breakdown.dx_pos2_samples,
        );
    }
}

fn print_signed_margin_objective_report(tag: &str, report: Option<SignedMarginObjectiveReport>) {
    if let Some(report) = report {
        println!(
            "{} | signed margin objective bank_loss {:.6} | sign_loss {:.6} | speed_loss {:.6} | weighted_loss {:.6} | active_bank_rate {:.6} | active_sign_rate {:.6} | active_speed_rate {:.6} | samples {}",
            tag,
            report.bank_loss,
            report.sign_loss,
            report.speed_loss,
            report.weighted_loss,
            report.active_bank_pairs as f32 / report.bank_pairs as f32,
            report.active_sign_pairs as f32 / report.sign_pairs as f32,
            report.active_speed_pairs as f32 / report.speed_pairs as f32,
            report.samples,
        );
    }
}
