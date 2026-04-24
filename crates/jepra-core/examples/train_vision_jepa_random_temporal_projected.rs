#[path = "support/projected_temporal.rs"]
mod projected_temporal;
#[path = "support/temporal_vision.rs"]
mod temporal_vision;

use jepra_core::{
    BottleneckPredictor, Linear, Predictor, PredictorModule, ProjectedVisionJepa,
    ResidualBottleneckPredictor, Tensor, projection_stats, representation_stats,
};
use projected_temporal::{
    PROJECTED_TRAIN_LOSS_MAX_REDUCTION_RATIO, PROJECTED_VALIDATION_BASE_SEED,
    PROJECTED_VALIDATION_BATCHES, PROJECTED_VALIDATION_LOSS_MAX_REDUCTION_RATIO,
    projected_validation_batch_losses_from_base_seed,
};
use temporal_vision::{
    CompactEncoderMode, MIN_MIXED_MODE_COUNT, PredictorMode, assert_temporal_contract,
    assert_temporal_experiment_improved, make_compact_frozen_encoder,
    make_compact_frozen_encoder_stronger, make_frozen_encoder, make_train_batch,
    make_validation_batch, make_validation_batch_with_both_motion_modes, motion_mode_counts,
    print_batch_summary, print_representation_stats,
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
    let (train_probe_t, train_probe_t1) = make_train_batch(run_config.train_base_seed, 0);
    let (train_probe_next_t, train_probe_next_t1) = make_train_batch(run_config.train_base_seed, 1);
    let (val_probe_t, val_probe_t1) = make_validation_batch(PROJECTED_VALIDATION_BASE_SEED, 0);
    let (mixed_val_probe_t, mixed_val_probe_t1, mixed_val_probe_seed) =
        make_validation_batch_with_both_motion_modes(PROJECTED_VALIDATION_BASE_SEED, 1);

    assert_temporal_contract(&train_probe_t, &train_probe_t1);
    assert_temporal_contract(&train_probe_next_t, &train_probe_next_t1);
    assert_temporal_contract(&val_probe_t, &val_probe_t1);
    assert_temporal_contract(&mixed_val_probe_t, &mixed_val_probe_t1);

    assert_ne!(train_probe_t.data, train_probe_next_t.data);
    assert_ne!(train_probe_t.data, val_probe_t.data);
    assert_ne!(val_probe_t.data, mixed_val_probe_t.data);
    let (mixed_slow_count, mixed_fast_count) =
        motion_mode_counts(&mixed_val_probe_t, &mixed_val_probe_t1);
    assert!(
        mixed_slow_count >= MIN_MIXED_MODE_COUNT,
        "mixed validation probe slow count too small: {} < {}",
        mixed_slow_count,
        MIN_MIXED_MODE_COUNT
    );
    assert!(
        mixed_fast_count >= MIN_MIXED_MODE_COUNT,
        "mixed validation probe fast count too small: {} < {}",
        mixed_fast_count,
        MIN_MIXED_MODE_COUNT
    );

    print_batch_summary("train probe", &train_probe_t, &train_probe_t1);
    print_batch_summary("validation probe", &val_probe_t, &val_probe_t1);
    print_batch_summary(
        "validation mixed probe",
        &mixed_val_probe_t,
        &mixed_val_probe_t1,
    );
    println!(
        "validation mixed probe seed {} | slow {} | fast {}",
        mixed_val_probe_seed, mixed_slow_count, mixed_fast_count
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
        projected_validation_batch_losses_from_base_seed(
            &model,
            REGULARIZER_WEIGHT,
            PROJECTED_VALIDATION_BASE_SEED,
            PROJECTED_VALIDATION_BATCHES,
        );
    let initial_projection_drift = model.target_projection_drift();
    let (initial_projection_mean_abs, initial_projection_var_mean) =
        projection_stats(&initial_projection_t);

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

    let experiment_summary = temporal_vision::run_temporal_experiment_with_summary(
        run_config,
        &mut model,
        initial_train_total_loss,
        initial_val_total_loss,
        |model, step, should_log| {
            let (x_t, x_t1) = make_train_batch(run_config.train_base_seed, step as u64);
            let momentum = run_config.target_projection_momentum_at_step(step);
            model.set_target_projection_momentum(momentum);
            let (prediction_loss, regularizer_loss, projector_drift_loss, total_loss) =
                if run_config.encoder_learning_rate > 0.0 {
                    model.step_with_trainable_encoder_and_projector_drift_regularizer(
                        &x_t,
                        &x_t1,
                        REGULARIZER_WEIGHT,
                        run_config.projector_drift_weight,
                        PREDICTOR_LR,
                        PROJECTOR_LR,
                        run_config.encoder_learning_rate,
                    )
                } else {
                    model.step_with_projector_drift_regularizer(
                        &x_t,
                        &x_t1,
                        REGULARIZER_WEIGHT,
                        run_config.projector_drift_weight,
                        PREDICTOR_LR,
                        PROJECTOR_LR,
                    )
                };

            if should_log {
                let current_momentum = model.target_projection_momentum();
                let (val_prediction_loss, val_regularizer_loss, val_total_loss) =
                    projected_validation_batch_losses_from_base_seed(
                        model,
                        REGULARIZER_WEIGHT,
                        PROJECTED_VALIDATION_BASE_SEED,
                        PROJECTED_VALIDATION_BATCHES,
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
            }

            total_loss
        },
        |model| {
            projected_validation_batch_losses_from_base_seed(
                model,
                REGULARIZER_WEIGHT,
                PROJECTED_VALIDATION_BASE_SEED,
                PROJECTED_VALIDATION_BATCHES,
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
        projected_validation_batch_losses_from_base_seed(
            &model,
            REGULARIZER_WEIGHT,
            PROJECTED_VALIDATION_BASE_SEED,
            PROJECTED_VALIDATION_BATCHES,
        );
    let (final_projection_mean_abs, final_projection_var_mean) =
        projection_stats(&final_projection_t);
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
}
