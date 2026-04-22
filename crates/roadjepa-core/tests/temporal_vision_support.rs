#[allow(dead_code)]
#[path = "../examples/support/temporal_vision.rs"]
mod temporal_vision;

use roadjepa_core::Tensor;
use temporal_vision::{
    assert_temporal_contract, make_temporal_batch, make_train_batch, make_validation_batch,
    square_center_x, BATCH_SIZE, IMAGE_SIZE, MOTION_DX,
};

fn total_mass(tensor: &Tensor, sample: usize) -> f32 {
    let mut total = 0.0;

    for row in 0..IMAGE_SIZE {
        for col in 0..IMAGE_SIZE {
            total += tensor.get(&[sample, 0, row, col]);
        }
    }

    total
}

#[test]
fn temporal_batch_is_deterministic_for_same_seed_and_changes_for_nearby_seed() {
    let (x_t_a, x_t1_a) = make_temporal_batch(BATCH_SIZE, 7_001);
    let (x_t_b, x_t1_b) = make_temporal_batch(BATCH_SIZE, 7_001);
    let (x_t_c, x_t1_c) = make_temporal_batch(BATCH_SIZE, 7_002);

    assert_eq!(x_t_a, x_t_b);
    assert_eq!(x_t1_a, x_t1_b);
    assert_ne!(x_t_a, x_t_c);
    assert_ne!(x_t1_a, x_t1_c);
}

#[test]
fn helper_batches_follow_seed_offsets() {
    let train_base_seed = 10_000;
    let validation_base_seed = 20_000;
    let train_step = 17;
    let validation_batch_idx = 9;

    let expected_train = make_temporal_batch(BATCH_SIZE, train_base_seed + train_step);
    let expected_validation =
        make_temporal_batch(BATCH_SIZE, validation_base_seed + validation_batch_idx);

    assert_eq!(
        make_train_batch(train_base_seed, train_step),
        expected_train
    );
    assert_eq!(
        make_validation_batch(validation_base_seed, validation_batch_idx),
        expected_validation
    );
}

#[test]
fn generated_temporal_batch_moves_right_and_decays_mass() {
    let (x_t, x_t1) = make_temporal_batch(BATCH_SIZE, 9_123);

    for sample in 0..BATCH_SIZE {
        let delta_x = square_center_x(&x_t1, sample) - square_center_x(&x_t, sample);

        assert!(
            (delta_x - MOTION_DX as f32).abs() < 1e-6,
            "sample {} moved by {:.6}, expected {}",
            sample,
            delta_x,
            MOTION_DX
        );
        assert!(
            total_mass(&x_t1, sample) < total_mass(&x_t, sample),
            "sample {} mass did not decay: {:.6} -> {:.6}",
            sample,
            total_mass(&x_t, sample),
            total_mass(&x_t1, sample)
        );
    }
}

#[test]
#[should_panic(expected = "mass did not follow the deterministic intensity rule")]
fn assert_temporal_contract_panics_when_mass_does_not_decay() {
    let (x_t, mut x_t1) = make_temporal_batch(BATCH_SIZE, 42);

    for sample in 0..BATCH_SIZE {
        for row in 0..IMAGE_SIZE {
            for col in 0..IMAGE_SIZE {
                if x_t1.get(&[sample, 0, row, col]) > 0.0 {
                    x_t1.set(&[sample, 0, row, col], 1.0);
                }
            }
        }
    }

    assert_temporal_contract(&x_t, &x_t1);
}
