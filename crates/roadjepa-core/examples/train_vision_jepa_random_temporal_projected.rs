#[path = "support/projected_temporal.rs"]
mod projected_temporal;
#[path = "support/temporal_vision.rs"]
mod temporal_vision;

use projected_temporal::{projected_batch_losses, projection_stats};
use roadjepa_core::{Linear, Predictor, ProjectedVisionJepa, Tensor};
use temporal_vision::{
    MIN_MIXED_MODE_COUNT, assert_temporal_contract, fast_mode_channel_summary, make_frozen_encoder,
    make_train_batch, make_validation_batch, make_validation_batch_with_both_motion_modes,
    motion_mode_counts, print_batch_summary,
};

const PROJECTION_DIM: usize = 4;
const TRAIN_BASE_SEED: u64 = 11_000;
const VALIDATION_BASE_SEED: u64 = 111_000;
const VALIDATION_BATCHES: usize = 8;
const NUM_STEPS: usize = 300;
const LOG_EVERY: usize = 25;
const PROJECTOR_LR: f32 = 0.005;
const PREDICTOR_LR: f32 = 0.02;
const REGULARIZER_WEIGHT: f32 = 1e-4;
const TARGET_FINAL_TRAIN_TOTAL_LOSS: f32 = 0.15;
const TARGET_FINAL_VAL_TOTAL_LOSS: f32 = 0.15;

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

fn validation_losses(model: &ProjectedVisionJepa) -> (f32, f32, f32) {
    let mut prediction_total = 0.0;
    let mut regularizer_total = 0.0;
    let mut total = 0.0;

    for batch_idx in 0..VALIDATION_BATCHES {
        let (x_t, x_t1) = make_validation_batch(VALIDATION_BASE_SEED, batch_idx as u64);
        let (prediction_loss, regularizer_loss, total_loss) = projected_batch_losses(
            &model.encoder,
            &model.projector,
            &model.target_projector,
            &model.predictor,
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

fn assert_fast_mode_channel_coverage(label: &str, z: &Tensor) {
    let (inactive_count, active_count, mean_value, max_value) = fast_mode_channel_summary(z);

    assert!(
        inactive_count >= MIN_MIXED_MODE_COUNT,
        "{} inactive fast-mode coverage too small: {} < {}",
        label,
        inactive_count,
        MIN_MIXED_MODE_COUNT
    );
    assert!(
        active_count >= MIN_MIXED_MODE_COUNT,
        "{} active fast-mode coverage too small: {} < {}",
        label,
        active_count,
        MIN_MIXED_MODE_COUNT
    );

    println!(
        "{} | inactive {} | active {} | mean {:.6} | max {:.6}",
        label, inactive_count, active_count, mean_value, max_value
    );
}

fn main() {
    let (train_probe_t, train_probe_t1) = make_train_batch(TRAIN_BASE_SEED, 0);
    let (train_probe_next_t, train_probe_next_t1) = make_train_batch(TRAIN_BASE_SEED, 1);
    let (val_probe_t, val_probe_t1) = make_validation_batch(VALIDATION_BASE_SEED, 0);
    let (mixed_val_probe_t, mixed_val_probe_t1, mixed_val_probe_seed) =
        make_validation_batch_with_both_motion_modes(VALIDATION_BASE_SEED, 1);

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

    let encoder = make_frozen_encoder();
    let online_projector = make_projector();
    let target_projector = online_projector.clone();
    let predictor = make_predictor();
    let mut model =
        ProjectedVisionJepa::new(encoder, online_projector, target_projector, predictor);

    let initial_mixed_val_z_t = model.encode(&mixed_val_probe_t);
    let initial_projection_t = model.project_latent(&train_probe_t);
    let initial_target = model.target_projection(&train_probe_t1);
    let (initial_train_prediction_loss, initial_train_regularizer_loss, initial_train_total_loss) =
        projected_batch_losses(
            &model.encoder,
            &model.projector,
            &model.target_projector,
            &model.predictor,
            &train_probe_t,
            &train_probe_t1,
            REGULARIZER_WEIGHT,
        );
    let (initial_val_prediction_loss, initial_val_regularizer_loss, initial_val_total_loss) =
        validation_losses(&model);
    let (initial_projection_mean_abs, initial_projection_var_mean) =
        projection_stats(&initial_projection_t);

    assert_fast_mode_channel_coverage("validation mixed probe latent", &initial_mixed_val_z_t);

    println!(
        "initial | projection sample0 {:?} | target {:?}",
        &initial_projection_t.data[0..PROJECTION_DIM],
        &initial_target.data[0..PROJECTION_DIM]
    );
    println!(
        "initial | proj mean_abs {:.6} | var_mean {:.6}",
        initial_projection_mean_abs, initial_projection_var_mean
    );
    println!(
        "initial | train pred {:.6} | reg {:.6} | total {:.6}",
        initial_train_prediction_loss, initial_train_regularizer_loss, initial_train_total_loss
    );
    println!(
        "initial | val pred {:.6} | reg {:.6} | total {:.6}",
        initial_val_prediction_loss, initial_val_regularizer_loss, initial_val_total_loss
    );

    for step in 1..=NUM_STEPS {
        let (x_t, x_t1) = make_train_batch(TRAIN_BASE_SEED, step as u64);
        let (prediction_loss, regularizer_loss, total_loss) =
            model.step(&x_t, &x_t1, REGULARIZER_WEIGHT, PREDICTOR_LR, PROJECTOR_LR);

        if step == 1 || step % LOG_EVERY == 0 {
            let (val_prediction_loss, val_regularizer_loss, val_total_loss) =
                validation_losses(&model);

            println!(
                "step {:03} | train pred {:.6} | reg {:.6} | total {:.6} | val total {:.6}",
                step, prediction_loss, regularizer_loss, total_loss, val_total_loss
            );
            println!(
                "step {:03} | val pred {:.6} | reg {:.6}",
                step, val_prediction_loss, val_regularizer_loss
            );
        }
    }

    let final_projection_t = model.project_latent(&train_probe_t);
    let final_pred = model.predict_next_projection(&train_probe_t);
    let final_target = model.target_projection(&train_probe_t1);
    let (final_train_prediction_loss, final_train_regularizer_loss, final_train_total_loss) =
        projected_batch_losses(
            &model.encoder,
            &model.projector,
            &model.target_projector,
            &model.predictor,
            &train_probe_t,
            &train_probe_t1,
            REGULARIZER_WEIGHT,
        );
    let (final_val_prediction_loss, final_val_regularizer_loss, final_val_total_loss) =
        validation_losses(&model);
    let (final_projection_mean_abs, final_projection_var_mean) =
        projection_stats(&final_projection_t);

    assert!(
        final_train_total_loss < initial_train_total_loss,
        "probe total loss did not improve: {:.6} -> {:.6}",
        initial_train_total_loss,
        final_train_total_loss
    );
    assert!(
        final_val_total_loss < initial_val_total_loss,
        "validation total loss did not improve: {:.6} -> {:.6}",
        initial_val_total_loss,
        final_val_total_loss
    );
    assert!(
        final_train_total_loss < TARGET_FINAL_TRAIN_TOTAL_LOSS,
        "probe total loss stayed too high: {:.6} >= {:.6}",
        final_train_total_loss,
        TARGET_FINAL_TRAIN_TOTAL_LOSS
    );
    assert!(
        final_val_total_loss < TARGET_FINAL_VAL_TOTAL_LOSS,
        "validation total loss stayed too high: {:.6} >= {:.6}",
        final_val_total_loss,
        TARGET_FINAL_VAL_TOTAL_LOSS
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
    println!(
        "final | train pred {:.6} | reg {:.6} | total {:.6}",
        final_train_prediction_loss, final_train_regularizer_loss, final_train_total_loss
    );
    println!(
        "final | val pred {:.6} | reg {:.6} | total {:.6}",
        final_val_prediction_loss, final_val_regularizer_loss, final_val_total_loss
    );
}
