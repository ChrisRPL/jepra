#[path = "support/temporal_vision.rs"]
mod temporal_vision;

use roadjepa_core::{Linear, Predictor, Tensor, VisionJepa};
use temporal_vision::{
    MIN_MIXED_MODE_COUNT, assert_temporal_contract, make_frozen_encoder, make_train_batch,
    make_validation_batch, make_validation_batch_with_both_motion_modes, motion_mode_counts,
    print_batch_summary,
};

const TRAIN_BASE_SEED: u64 = 1_000;
const VALIDATION_BASE_SEED: u64 = 99_001;
const VALIDATION_BATCHES: usize = 8;
const NUM_STEPS: usize = 300;
const LOG_EVERY: usize = 25;
const LR: f32 = 0.02;
const TARGET_FINAL_TRAIN_LOSS: f32 = 0.05;
const TARGET_FINAL_VAL_LOSS: f32 = 0.05;

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

fn batch_loss(model: &VisionJepa, x_t: &Tensor, x_t1: &Tensor) -> f32 {
    model.losses(x_t, x_t1).0
}

fn validation_loss(model: &VisionJepa) -> f32 {
    let mut total = 0.0;

    for batch_idx in 0..VALIDATION_BATCHES {
        let (x_t, x_t1) = make_validation_batch(VALIDATION_BASE_SEED, batch_idx as u64);
        total += batch_loss(model, &x_t, &x_t1);
    }

    total / VALIDATION_BATCHES as f32
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
    let predictor = make_predictor();
    let mut model = VisionJepa::new(encoder, predictor);

    let initial_z_t = model.encode(&train_probe_t);
    let initial_z_t1 = model.target_latent(&train_probe_t1);
    let initial_train_loss = batch_loss(&model, &train_probe_t, &train_probe_t1);
    let initial_val_loss = validation_loss(&model);

    println!(
        "initial | latent sample0 {:?} -> {:?}",
        &initial_z_t.data[0..3],
        &initial_z_t1.data[0..3]
    );
    println!(
        "initial | probe train {:.6} | val {:.6}",
        initial_train_loss, initial_val_loss
    );

    for step in 1..=NUM_STEPS {
        let (x_t, x_t1) = make_train_batch(TRAIN_BASE_SEED, step as u64);
        let (train_loss, _) = model.step(&x_t, &x_t1, LR);

        if step == 1 || step % LOG_EVERY == 0 {
            println!(
                "step {:03} | train {:.6} | val {:.6}",
                step,
                train_loss,
                validation_loss(&model)
            );
        }
    }

    let final_train_loss = batch_loss(&model, &train_probe_t, &train_probe_t1);
    let final_val_loss = validation_loss(&model);
    let final_z_t = model.encode(&train_probe_t);
    let final_pred = model.predict_next_latent(&train_probe_t);
    let final_target = model.target_latent(&train_probe_t1);

    assert!(
        final_train_loss < initial_train_loss,
        "probe train loss did not improve: {:.6} -> {:.6}",
        initial_train_loss,
        final_train_loss
    );
    assert!(
        final_val_loss < initial_val_loss,
        "validation loss did not improve: {:.6} -> {:.6}",
        initial_val_loss,
        final_val_loss
    );
    assert!(
        final_train_loss < TARGET_FINAL_TRAIN_LOSS,
        "probe train loss stayed too high: {:.6} >= {:.6}",
        final_train_loss,
        TARGET_FINAL_TRAIN_LOSS
    );
    assert!(
        final_val_loss < TARGET_FINAL_VAL_LOSS,
        "validation loss stayed too high: {:.6} >= {:.6}",
        final_val_loss,
        TARGET_FINAL_VAL_LOSS
    );

    println!(
        "\nfinal | latent sample0 {:?} | pred {:?} | target {:?}",
        &final_z_t.data[0..3],
        &final_pred.data[0..3],
        &final_target.data[0..3]
    );
    println!(
        "final | probe train {:.6} | val {:.6}",
        final_train_loss, final_val_loss
    );
}
