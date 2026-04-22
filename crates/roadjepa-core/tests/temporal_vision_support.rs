#[allow(dead_code)]
#[path = "../examples/support/temporal_vision.rs"]
mod temporal_vision;

use roadjepa_core::Tensor;
use temporal_vision::{
    assert_temporal_contract, batch_has_both_motion_modes, fast_motion_feature_for_sample,
    make_frozen_encoder, make_temporal_batch, make_train_batch, make_validation_batch,
    make_validation_batch_with_both_motion_modes, motion_dx_for_sample, square_center_x,
    BATCH_SIZE, FAST_MOTION_DX, IMAGE_SIZE, SLOW_MOTION_DX,
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
        let expected_dx = motion_dx_for_sample(&x_t, sample) as f32;

        assert!(
            (delta_x - expected_dx).abs() < 1e-6,
            "sample {} moved by {:.6}, expected {:.6}",
            sample,
            delta_x,
            expected_dx
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
fn generator_exposes_both_motion_modes_across_seed_range() {
    let mut saw_slow_motion = false;
    let mut saw_fast_motion = false;

    for seed in 0..64 {
        let (x_t, _) = make_temporal_batch(BATCH_SIZE, seed);

        for sample in 0..BATCH_SIZE {
            match motion_dx_for_sample(&x_t, sample) {
                SLOW_MOTION_DX => saw_slow_motion = true,
                FAST_MOTION_DX => saw_fast_motion = true,
                dx => panic!("unexpected motion dx {}", dx),
            }
        }
    }

    assert!(saw_slow_motion, "never observed slow motion");
    assert!(saw_fast_motion, "never observed fast motion");
}

#[test]
fn mixed_mode_validation_probe_is_deterministic_and_contains_both_modes() {
    let batch_a = make_validation_batch_with_both_motion_modes(20_000, 1);
    let batch_b = make_validation_batch_with_both_motion_modes(20_000, 1);

    assert_eq!(batch_a.0, batch_b.0);
    assert_eq!(batch_a.1, batch_b.1);
    assert_eq!(batch_a.2, batch_b.2);
    assert!(batch_a.2 >= 20_001);
    assert!(batch_has_both_motion_modes(&batch_a.0));
}

#[test]
fn frozen_encoder_exposes_fast_motion_mode_feature() {
    let encoder = make_frozen_encoder();
    let mut checked_samples = 0;

    for seed in 0..64 {
        let (x_t, _) = make_temporal_batch(BATCH_SIZE, seed);
        let z_t = encoder.forward(&x_t);

        assert_eq!(z_t.shape, vec![BATCH_SIZE, 3]);

        for sample in 0..BATCH_SIZE {
            let expected = fast_motion_feature_for_sample(&x_t, sample);
            let actual = z_t.get(&[sample, 2]);

            assert!(
                (actual - expected).abs() < 1e-6,
                "sample {} feature mismatch: {:.6} vs {:.6}",
                sample,
                actual,
                expected
            );
            checked_samples += 1;
        }
    }

    assert!(checked_samples > 0);
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
