#[allow(dead_code)]
#[path = "../examples/support/temporal_vision.rs"]
mod temporal_vision;

use roadjepa_core::{Linear, Predictor, Tensor, VisionJepa};
use temporal_vision::{
    BATCH_SIZE, FAST_MOTION_DX, IMAGE_SIZE, MIN_MIXED_MODE_COUNT, SLOW_MOTION_DX,
    assert_temporal_contract, batch_has_both_motion_modes, batch_has_min_motion_mode_counts,
    fast_mode_channel_summary, fast_motion_feature_for_sample, make_frozen_encoder,
    make_temporal_batch, make_train_batch, make_validation_batch,
    make_validation_batch_with_both_motion_modes, motion_dx_for_sample, motion_mode_counts,
    square_center_x,
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

fn batch_loss(model: &VisionJepa, x_t: &Tensor, x_t1: &Tensor) -> f32 {
    model.losses(x_t, x_t1).0
}

fn validation_loss(model: &VisionJepa) -> f32 {
    let mut total = 0.0;
    let validation_base_seed = 99_001u64;
    let validation_batches = 8usize;

    for batch_idx in 0..validation_batches {
        let (x_t, x_t1) = make_validation_batch(validation_base_seed, batch_idx as u64);
        total += batch_loss(model, &x_t, &x_t1);
    }

    total / validation_batches as f32
}

fn make_predictor() -> Predictor {
    Predictor::new(
        Linear::new(
            Tensor::new(
                vec![
                    1.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, //
                    0.0, 0.0, 1.0,
                ],
                vec![3, 3],
            ),
            Tensor::new(vec![0.0, 0.0, 0.0], vec![3]),
        ),
        Linear::new(
            Tensor::new(
                vec![
                    1.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, //
                    0.0, 0.0, 1.0,
                ],
                vec![3, 3],
            ),
            Tensor::new(vec![0.0, 0.0, 0.0], vec![3]),
        ),
    )
}

#[test]
fn unprojected_random_temporal_training_reduces_train_and_validation_loss() {
    let encoder = make_frozen_encoder();
    let mut model = VisionJepa::new(encoder, make_predictor());
    let train_base_seed = 1_000u64;
    let steps = 120;
    let lr = 0.02;

    let (probe_t, probe_t1) = make_train_batch(train_base_seed, 0);
    let initial_train_loss = batch_loss(&model, &probe_t, &probe_t1);
    let initial_val_loss = validation_loss(&model);

    for step in 1..=steps {
        let (x_t, x_t1) = make_train_batch(train_base_seed, step as u64);
        let (train_loss, _) = model.step(&x_t, &x_t1, lr);

        if step == 1 || step == steps {
            println!(
                "unprojected random temporal step {:03} | train {:.6} | val {:.6}",
                step,
                train_loss,
                validation_loss(&model)
            );
        }
    }

    let final_train_loss = batch_loss(&model, &probe_t, &probe_t1);
    let final_val_loss = validation_loss(&model);

    assert!(
        final_train_loss < initial_train_loss,
        "train loss did not improve: {:.6} -> {:.6}",
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
        final_train_loss < 0.1,
        "train loss too high after {} steps: {:.6}",
        steps,
        final_train_loss
    );
    assert!(
        final_val_loss < 0.08,
        "validation loss too high after {} steps: {:.6}",
        steps,
        final_val_loss
    );
}

#[test]
fn unprojected_random_temporal_training_trajectory_is_reproducible() {
    let encoder = make_frozen_encoder();
    let mut model_a = VisionJepa::new(encoder.clone(), make_predictor());
    let mut model_b = VisionJepa::new(encoder, make_predictor());
    let train_base_seed = 1_100u64;
    let steps = 6;
    let lr = 0.02;

    let (probe_t, probe_t1) = make_train_batch(train_base_seed, 0);

    let mut trajectory_a = Vec::<(f32, f32)>::new();
    let mut trajectory_b = Vec::<(f32, f32)>::new();

    for step in 1..=steps {
        let (x_t, x_t1) = make_train_batch(train_base_seed, step as u64);
        let batch_step_a = model_a.step(&x_t, &x_t1, lr);
        let batch_step_b = model_b.step(&x_t, &x_t1, lr);

        assert!(
            (batch_step_a.0 - batch_step_b.0).abs() < 1e-7,
            "step loss diverged at trajectory batch {}: {:.6} vs {:.6}",
            step,
            batch_step_a.0,
            batch_step_b.0
        );
        assert!(
            (batch_step_a.1 - batch_step_b.1).abs() < 1e-7,
            "total loss diverged at trajectory batch {}: {:.6} vs {:.6}",
            step,
            batch_step_a.1,
            batch_step_b.1
        );
        assert_eq!(
            model_a, model_b,
            "models diverged at trajectory batch {}",
            step
        );

        let train_loss_a = batch_loss(&model_a, &probe_t, &probe_t1);
        let train_loss_b = batch_loss(&model_b, &probe_t, &probe_t1);
        assert!(
            (train_loss_a - train_loss_b).abs() < 1e-7,
            "probe train loss diverged at trajectory batch {}: {:.6} vs {:.6}",
            step,
            train_loss_a,
            train_loss_b
        );

        let validation_loss_a = validation_loss(&model_a);
        let validation_loss_b = validation_loss(&model_b);
        assert!(
            (validation_loss_a - validation_loss_b).abs() < 1e-7,
            "validation loss diverged at trajectory batch {}: {:.6} vs {:.6}",
            step,
            validation_loss_a,
            validation_loss_b
        );

        trajectory_a.push((train_loss_a, validation_loss_a));
        trajectory_b.push((train_loss_b, validation_loss_b));
    }

    assert_eq!(trajectory_a, trajectory_b);
}

#[test]
fn unprojected_temporal_step_reduces_loss_on_random_batch() {
    let encoder = make_frozen_encoder();
    let (x_t, x_t1) = make_train_batch(31_000, 2);
    let mut model = VisionJepa::new(encoder, make_predictor());
    let lr = 0.02;

    let initial_loss = batch_loss(&model, &x_t, &x_t1);
    model.step(&x_t, &x_t1, lr);
    let final_loss = batch_loss(&model, &x_t, &x_t1);

    assert!(
        final_loss + 1e-6 < initial_loss,
        "one step did not reduce prediction loss: {:.6} -> {:.6}",
        initial_loss,
        final_loss
    );
}

#[test]
fn unprojected_temporal_step_reported_loss_matches_batch_loss() {
    let encoder = make_frozen_encoder();
    let (x_t, x_t1) = make_train_batch(31_001, 0);
    let mut model = VisionJepa::new(encoder, make_predictor());
    let lr = 0.015;

    let expected_loss = batch_loss(&model, &x_t, &x_t1);
    let (step_loss, total_loss) = model.step(&x_t, &x_t1, lr);
    let actual_loss = batch_loss(&model, &x_t, &x_t1);

    assert!(
        (step_loss - expected_loss).abs() < 1e-6,
        "step loss should match batch loss before update: {:.6} vs {:.6}",
        expected_loss,
        step_loss
    );
    assert!(
        (total_loss - expected_loss).abs() < 1e-6,
        "step total should match step loss for unprojected JEPA: {:.6} vs {:.6}",
        expected_loss,
        total_loss
    );
    assert!(
        actual_loss < expected_loss,
        "one step did not decrease loss: {:.6} -> {:.6}",
        expected_loss,
        actual_loss
    );
}

#[test]
fn unprojected_temporal_step_stable_over_two_steps() {
    let encoder = make_frozen_encoder();
    let mut model = VisionJepa::new(encoder.clone(), make_predictor());
    let lr = 0.02;
    let encoder_snapshot = model.encoder.clone();
    let mut prev_predictor_fc1_weight = model.predictor.fc1.weight.clone();
    let mut prev_predictor_fc1_bias = model.predictor.fc1.bias.clone();
    let mut prev_predictor_fc2_weight = model.predictor.fc2.weight.clone();
    let mut prev_predictor_fc2_bias = model.predictor.fc2.bias.clone();

    for batch_idx in 0..2 {
        let (x_t, x_t1) = make_train_batch(31_100, batch_idx as u64);
        let expected = batch_loss(&model, &x_t, &x_t1);
        let (step_loss, total_loss) = model.step(&x_t, &x_t1, lr);

        assert!(
            (step_loss - expected).abs() < 1e-6,
            "step loss mismatch at batch {}: {:.6} vs {:.6}",
            batch_idx,
            step_loss,
            expected
        );
        assert!(
            (total_loss - expected).abs() < 1e-6,
            "total loss mismatch at batch {}: {:.6} vs {:.6}",
            batch_idx,
            total_loss,
            expected
        );
        assert_eq!(model.encoder, encoder_snapshot);
        assert_ne!(
            model.predictor.fc1.weight, prev_predictor_fc1_weight,
            "fc1.weight did not change at batch {}",
            batch_idx
        );
        assert_ne!(
            model.predictor.fc1.bias, prev_predictor_fc1_bias,
            "fc1.bias did not change at batch {}",
            batch_idx
        );
        assert_ne!(
            model.predictor.fc2.weight, prev_predictor_fc2_weight,
            "fc2.weight did not change at batch {}",
            batch_idx
        );
        assert_ne!(
            model.predictor.fc2.bias, prev_predictor_fc2_bias,
            "fc2.bias did not change at batch {}",
            batch_idx
        );

        prev_predictor_fc1_weight = model.predictor.fc1.weight.clone();
        prev_predictor_fc1_bias = model.predictor.fc1.bias.clone();
        prev_predictor_fc2_weight = model.predictor.fc2.weight.clone();
        prev_predictor_fc2_bias = model.predictor.fc2.bias.clone();
    }
}

#[test]
fn unprojected_temporal_step_reproducible_over_fixed_batch_schedule() {
    let encoder = make_frozen_encoder();
    let mut model_a = VisionJepa::new(encoder.clone(), make_predictor());
    let mut model_b = VisionJepa::new(encoder, make_predictor());
    let lr = 0.02;

    for step_idx in 0..3 {
        let (x_t, x_t1) = make_train_batch(31_200, step_idx as u64);
        let expected_a = batch_loss(&model_a, &x_t, &x_t1);
        let expected_b = batch_loss(&model_b, &x_t, &x_t1);

        assert!(
            (expected_a - expected_b).abs() < 1e-7,
            "pre-step loss mismatch at batch {}: {:.6} vs {:.6}",
            step_idx,
            expected_a,
            expected_b
        );

        let result_a = model_a.step(&x_t, &x_t1, lr);
        let result_b = model_b.step(&x_t, &x_t1, lr);

        assert!(
            (result_a.0 - result_b.0).abs() < 1e-7,
            "step loss mismatch at batch {}: {:.6} vs {:.6}",
            step_idx,
            result_a.0,
            result_b.0
        );
        assert!(
            (result_a.1 - result_b.1).abs() < 1e-7,
            "total loss mismatch at batch {}: {:.6} vs {:.6}",
            step_idx,
            result_a.1,
            result_b.1
        );
        assert_eq!(model_a, model_b, "models diverged at batch {}", step_idx);

        let actual_a = batch_loss(&model_a, &x_t, &x_t1);
        let actual_b = batch_loss(&model_b, &x_t, &x_t1);

        assert!(
            (actual_a - actual_b).abs() < 1e-7,
            "post-step loss mismatch at batch {}: {:.6} vs {:.6}",
            step_idx,
            actual_a,
            actual_b
        );
        assert!(
            (result_a.0 - expected_a).abs() < 1e-6,
            "step loss mismatch vs pre-step expectation at batch {}: {:.6} vs {:.6}",
            step_idx,
            result_a.0,
            expected_a
        );
    }
}

#[test]
fn unprojected_temporal_step_reduces_loss_over_two_steps_on_same_batch() {
    let encoder = make_frozen_encoder();
    let mut model = VisionJepa::new(encoder, make_predictor());
    let (x_t, x_t1) = make_train_batch(31_101, 0);
    let lr = 0.01;
    let mut prev_loss = batch_loss(&model, &x_t, &x_t1);

    for step_idx in 0..2 {
        let expected = batch_loss(&model, &x_t, &x_t1);
        let (step_loss, total_loss) = model.step(&x_t, &x_t1, lr);
        let current_loss = batch_loss(&model, &x_t, &x_t1);

        assert!(
            (step_loss - expected).abs() < 1e-6,
            "step loss mismatch at step {}: {:.6} vs {:.6}",
            step_idx,
            step_loss,
            expected
        );
        assert!(
            (total_loss - expected).abs() < 1e-6,
            "total loss mismatch at step {}: {:.6} vs {:.6}",
            step_idx,
            total_loss,
            expected
        );
        assert!(
            current_loss + 1e-6 < prev_loss,
            "same batch loss did not decrease at step {}: {:.6} -> {:.6}",
            step_idx,
            prev_loss,
            current_loss
        );
        prev_loss = current_loss;
    }
}

#[test]
fn unprojected_step_reduces_total_loss_over_two_steps_on_same_batch() {
    let encoder = make_frozen_encoder();
    let mut model = VisionJepa::new(encoder, make_predictor());
    let (x_t, x_t1) = make_train_batch(31_201, 1);
    let lr = 0.01;
    let mut prev_total = batch_loss(&model, &x_t, &x_t1);

    for step_idx in 0..2 {
        let expected = batch_loss(&model, &x_t, &x_t1);
        let (step_loss, total_loss) = model.step(&x_t, &x_t1, lr);
        let (_, post_total) = model.losses(&x_t, &x_t1);

        assert!(
            (step_loss - expected).abs() < 1e-6,
            "unprojected step total mismatch at step {}: {:.6} vs {:.6}",
            step_idx,
            step_loss,
            expected
        );
        assert!(
            (total_loss - expected).abs() < 1e-6,
            "unprojected total loss mismatch at step {}: {:.6} vs {:.6}",
            step_idx,
            total_loss,
            expected
        );
        assert!(
            post_total + 1e-6 < prev_total,
            "unprojected same-batch total loss did not decrease at step {}: {:.6} -> {:.6}",
            step_idx,
            prev_total,
            post_total
        );

        prev_total = post_total;
    }
}

#[test]
fn unprojected_temporal_step_updates_predictor_parameters() {
    let encoder = make_frozen_encoder();
    let mut model = VisionJepa::new(encoder, make_predictor());
    let (x_t, x_t1) = make_train_batch(31_002, 0);
    let lr = 0.02;
    let encoder_snapshot = model.encoder.clone();

    let predictor_fc1_weight_snapshot = model.predictor.fc1.weight.clone();
    let predictor_fc1_bias_snapshot = model.predictor.fc1.bias.clone();
    let predictor_fc2_weight_snapshot = model.predictor.fc2.weight.clone();
    let predictor_fc2_bias_snapshot = model.predictor.fc2.bias.clone();

    model.step(&x_t, &x_t1, lr);

    assert_eq!(model.encoder, encoder_snapshot);
    assert_ne!(model.predictor.fc1.weight, predictor_fc1_weight_snapshot);
    assert_ne!(model.predictor.fc1.bias, predictor_fc1_bias_snapshot);
    assert_ne!(model.predictor.fc2.weight, predictor_fc2_weight_snapshot);
    assert_ne!(model.predictor.fc2.bias, predictor_fc2_bias_snapshot);
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
        let expected_dx = motion_dx_for_sample(&x_t, &x_t1, sample) as f32;

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
        let (x_t, x_t1) = make_temporal_batch(BATCH_SIZE, seed);

        for sample in 0..BATCH_SIZE {
            match motion_dx_for_sample(&x_t, &x_t1, sample) {
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
    let (slow_count, fast_count) = motion_mode_counts(&batch_a.0, &batch_a.1);

    assert_eq!(batch_a.0, batch_b.0);
    assert_eq!(batch_a.1, batch_b.1);
    assert_eq!(batch_a.2, batch_b.2);
    assert!(batch_a.2 >= 20_001);
    assert!(batch_has_both_motion_modes(&batch_a.0, &batch_a.1));
    assert!(batch_has_min_motion_mode_counts(
        &batch_a.0,
        &batch_a.1,
        MIN_MIXED_MODE_COUNT,
        MIN_MIXED_MODE_COUNT
    ));
    assert!(slow_count >= MIN_MIXED_MODE_COUNT);
    assert!(fast_count >= MIN_MIXED_MODE_COUNT);
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
fn fast_mode_channel_summary_matches_mixed_probe_counts() {
    let encoder = make_frozen_encoder();
    let (x_t, _, _) = make_validation_batch_with_both_motion_modes(20_000, 1);
    let z_t = encoder.forward(&x_t);
    let (inactive_count, active_count, mean_value, max_value) = fast_mode_channel_summary(&z_t);

    assert_eq!(inactive_count + active_count, BATCH_SIZE);
    assert!(inactive_count >= MIN_MIXED_MODE_COUNT);
    assert!(active_count >= MIN_MIXED_MODE_COUNT);
    assert!(mean_value > 0.0);
    assert!(max_value > 0.0);
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
