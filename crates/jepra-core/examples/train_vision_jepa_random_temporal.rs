#[path = "support/temporal_validation.rs"]
mod temporal_validation;
#[path = "support/temporal_vision.rs"]
mod temporal_vision;

use jepra_core::{
    BottleneckPredictor, Linear, Predictor, PredictorModule, ResidualBottleneckPredictor, Tensor,
    VisionJepa, representation_stats,
};
use temporal_validation::{
    UNPROJECTED_TRAIN_LOSS_MAX_REDUCTION_RATIO, UNPROJECTED_VALIDATION_BASE_SEED,
    UNPROJECTED_VALIDATION_BATCHES, UNPROJECTED_VALIDATION_LOSS_MAX_REDUCTION_RATIO,
    temporal_validation_batch_loss_from_base_seed_for_task,
};
use temporal_vision::{
    CompactEncoderMode, PredictorMode, assert_required_motion_modes_for_task,
    assert_temporal_contract, assert_temporal_experiment_improved, make_compact_frozen_encoder,
    make_compact_frozen_encoder_signed_direction,
    make_compact_frozen_encoder_signed_direction_magnitude, make_compact_frozen_encoder_stronger,
    make_frozen_encoder, make_train_batch_for_config, make_validation_batch_for_config,
    make_validation_batch_with_required_motion_modes_for_config, print_batch_summary_for_task,
    print_motion_mode_summary_for_task, print_representation_stats,
};

const TRAIN_BASE_SEED: u64 = 1_000;
const NUM_STEPS: usize = 300;
const LOG_EVERY: usize = 25;
const LR: f32 = 0.02;
const ENCODER_LR: f32 = 0.0;

fn make_predictor() -> Predictor {
    let fc1 = Linear::new(
        Tensor::new(
            vec![
                1.0, 0.0, 0.0, 1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
            ],
            vec![3, 6],
        ),
        Tensor::new(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![6]),
    );

    let fc2 = Linear::new(
        Tensor::zeros(vec![6, 3]),
        Tensor::new(vec![0.0, 0.0, 0.0], vec![3]),
    );

    Predictor::new(fc1, fc2)
}

fn make_bottleneck_predictor() -> BottleneckPredictor {
    let fc1 = Linear::new(
        Tensor::new(
            vec![
                1.0, 0.0, 0.0, 1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
            ],
            vec![3, 6],
        ),
        Tensor::zeros(vec![6]),
    );
    let fc2 = Linear::new(
        Tensor::new(
            vec![
                1.0, 0.0, //
                0.0, 1.0, //
                0.0, 0.0, //
                1.0, 0.0, //
                0.0, 1.0, //
                0.0, 0.0,
            ],
            vec![6, 2],
        ),
        Tensor::zeros(vec![2]),
    );
    let fc3 = Linear::new(
        Tensor::new(
            vec![
                1.0, 0.0, 0.5, //
                0.0, 1.0, 0.5,
            ],
            vec![2, 3],
        ),
        Tensor::zeros(vec![3]),
    );

    BottleneckPredictor::new(fc1, fc2, fc3)
}

fn make_residual_bottleneck_predictor(residual_delta_scale: f32) -> ResidualBottleneckPredictor {
    let fc1 = Linear::new(
        Tensor::new(
            vec![
                1.0, 0.0, 0.0, 1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
            ],
            vec![3, 6],
        ),
        Tensor::zeros(vec![6]),
    );
    let fc2 = Linear::new(
        Tensor::new(
            vec![
                1.0, 0.0, //
                0.0, 1.0, //
                0.0, 0.0, //
                1.0, 0.0, //
                0.0, 1.0, //
                0.0, 0.0,
            ],
            vec![6, 2],
        ),
        Tensor::zeros(vec![2]),
    );
    let fc3 = Linear::new(Tensor::zeros(vec![2, 3]), Tensor::zeros(vec![3]));

    ResidualBottleneckPredictor::new_scaled(
        BottleneckPredictor::new(fc1, fc2, fc3),
        residual_delta_scale,
    )
}

fn reduction_thresholds_for_run_config(
    run_config: temporal_vision::TemporalRunConfig,
) -> (f32, f32) {
    match (run_config.compact_encoder_mode, run_config.predictor_mode) {
        (CompactEncoderMode::Disabled, PredictorMode::Baseline) => (
            UNPROJECTED_TRAIN_LOSS_MAX_REDUCTION_RATIO,
            UNPROJECTED_VALIDATION_LOSS_MAX_REDUCTION_RATIO,
        ),
        _ => (1.0, 1.0),
    }
}

pub fn main() {
    let run_config = temporal_vision::TemporalRunConfig::from_args(
        TRAIN_BASE_SEED,
        NUM_STEPS,
        LOG_EVERY,
        ENCODER_LR,
    );
    assert!(
        run_config.signed_margin_weight == 0.0,
        "signed margin objective is only supported by projected signed-velocity-trail runs"
    );
    assert!(
        run_config.signed_bank_softmax_weight == 0.0,
        "signed bank softmax objective is only supported by projected signed-velocity-trail runs"
    );

    println!(
        "temporal run config | train_base_seed {} | steps {} | log_every {}",
        run_config.train_base_seed, run_config.total_steps, run_config.log_every,
    );
    println!(
        "temporal run config | encoder learning rate {}",
        run_config.encoder_learning_rate
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
        make_validation_batch_for_config(run_config, UNPROJECTED_VALIDATION_BASE_SEED, 0);
    let (mixed_val_probe_t, mixed_val_probe_t1, mixed_val_probe_seed) =
        make_validation_batch_with_required_motion_modes_for_config(
            run_config,
            UNPROJECTED_VALIDATION_BASE_SEED,
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
        CompactEncoderMode::SignedDirection => make_compact_frozen_encoder_signed_direction(),
        CompactEncoderMode::SignedDirectionMagnitude => {
            make_compact_frozen_encoder_signed_direction_magnitude()
        }
    };
    let mut model = VisionJepa::new(encoder, predictor);

    let initial_z_t = model.encode(&train_probe_t);
    let initial_z_t1 = model.target_latent(&train_probe_t1);
    let initial_pred = model.predict_next_latent(&train_probe_t);
    let initial_train_loss = model.losses(&train_probe_t, &train_probe_t1).0;
    let initial_val_loss = temporal_validation_batch_loss_from_base_seed_for_task(
        &model,
        UNPROJECTED_VALIDATION_BASE_SEED,
        UNPROJECTED_VALIDATION_BATCHES,
        run_config.temporal_task_mode,
    );

    println!(
        "initial | latent sample0 {:?} -> {:?}",
        &initial_z_t.data[0..3],
        &initial_z_t1.data[0..3]
    );
    println!(
        "initial | probe train {:.6} | val {:.6}",
        initial_train_loss, initial_val_loss
    );
    print_representation_stats("initial prediction", &representation_stats(&initial_pred));
    print_representation_stats("initial target", &representation_stats(&initial_z_t1));

    let experiment_summary = temporal_vision::run_temporal_experiment_with_summary(
        run_config,
        &mut model,
        initial_train_loss,
        initial_val_loss,
        |model, step, should_log| {
            let (x_t, x_t1) = make_train_batch_for_config(run_config, step as u64);
            let (train_loss, _) = model.step_with_trainable_encoder(
                &x_t,
                &x_t1,
                LR,
                run_config.encoder_learning_rate,
            );
            if should_log {
                temporal_vision::print_temporal_train_val_metrics(
                    step,
                    train_loss,
                    temporal_validation_batch_loss_from_base_seed_for_task(
                        model,
                        UNPROJECTED_VALIDATION_BASE_SEED,
                        UNPROJECTED_VALIDATION_BATCHES,
                        run_config.temporal_task_mode,
                    ),
                );
            }

            train_loss
        },
        |model| {
            temporal_validation_batch_loss_from_base_seed_for_task(
                model,
                UNPROJECTED_VALIDATION_BASE_SEED,
                UNPROJECTED_VALIDATION_BATCHES,
                run_config.temporal_task_mode,
            )
        },
    );
    temporal_vision::print_temporal_experiment_summary("unprojected", &experiment_summary);

    let final_train_loss = model.losses(&train_probe_t, &train_probe_t1).0;
    let final_val_loss = temporal_validation_batch_loss_from_base_seed_for_task(
        &model,
        UNPROJECTED_VALIDATION_BASE_SEED,
        UNPROJECTED_VALIDATION_BATCHES,
        run_config.temporal_task_mode,
    );
    let final_z_t = model.encode(&train_probe_t);
    let final_pred = model.predict_next_latent(&train_probe_t);
    let final_target = model.target_latent(&train_probe_t1);
    let (train_reduction_threshold, validation_reduction_threshold) =
        reduction_thresholds_for_run_config(run_config);

    assert_temporal_experiment_improved(
        "unprojected",
        initial_train_loss,
        final_train_loss,
        initial_val_loss,
        final_val_loss,
        train_reduction_threshold,
        validation_reduction_threshold,
    );

    println!(
        "\nfinal | latent sample0 {:?} | pred {:?} | target {:?}",
        &final_z_t.data[0..3],
        &final_pred.data[0..3],
        &final_target.data[0..3]
    );
    print_representation_stats("final prediction", &representation_stats(&final_pred));
    print_representation_stats("final target", &representation_stats(&final_target));
    println!(
        "final | probe train {:.6} | val {:.6}",
        final_train_loss, final_val_loss
    );
}
