use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use roadjepa_core::{Conv2d, ConvEncoder, EmbeddingEncoder, Tensor};

pub const BATCH_SIZE: usize = 8;
pub const IMAGE_SIZE: usize = 8;
pub const CHANNELS: usize = 1;
pub const SQUARE_SIZE: usize = 2;
pub const SLOW_MOTION_DX: usize = 1;
pub const FAST_MOTION_DX: usize = 2;
pub const FAST_MOTION_MASS_THRESHOLD: f32 = 0.8f32 * (SQUARE_SIZE * SQUARE_SIZE) as f32;
pub const MIXED_MODE_SEARCH_LIMIT: u64 = 64;
pub const MIN_MIXED_MODE_COUNT: usize = 2;

pub fn motion_dx_for_pair(x_t: &Tensor, x_t1: &Tensor, sample: usize) -> usize {
    let center_x_t = square_center_x(x_t, sample);
    let center_x_t1 = square_center_x(x_t1, sample);
    let delta_x = (center_x_t1 - center_x_t).round();

    if (delta_x - 1.0).abs() < 1e-6 {
        SLOW_MOTION_DX
    } else if (delta_x - 2.0).abs() < 1e-6 {
        FAST_MOTION_DX
    } else {
        panic!("unexpected motion delta {:.6}", delta_x);
    }
}

pub fn motion_dx_for_sample(x_t: &Tensor, x_t1: &Tensor, sample: usize) -> usize {
    motion_dx_for_pair(x_t, x_t1, sample)
}

fn random_motion_dx(rng: &mut StdRng) -> usize {
    if rng.gen_bool(0.5) {
        SLOW_MOTION_DX
    } else {
        FAST_MOTION_DX
    }
}

pub fn make_temporal_batch(batch_size: usize, seed: u64) -> (Tensor, Tensor) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x_t = Tensor::zeros(vec![batch_size, CHANNELS, IMAGE_SIZE, IMAGE_SIZE]);
    let mut x_t1 = Tensor::zeros(vec![batch_size, CHANNELS, IMAGE_SIZE, IMAGE_SIZE]);

    let max_row = IMAGE_SIZE - SQUARE_SIZE;
    let max_col_t = IMAGE_SIZE - SQUARE_SIZE - FAST_MOTION_DX;

    for sample in 0..batch_size {
        let row = rng.gen_range(0..=max_row);
        let col_t = rng.gen_range(0..=max_col_t);
        let intensity_t = rng.gen_range(0.65f32..0.95f32);
        let motion_dx = random_motion_dx(&mut rng);
        let col_t1 = col_t + motion_dx;
        let intensity_t1 = (0.9f32 * intensity_t + 0.05f32).clamp(0.5f32, 1.0f32);

        draw_square(&mut x_t, sample, row, col_t, intensity_t);
        draw_square(&mut x_t1, sample, row, col_t1, intensity_t1);
    }

    (x_t, x_t1)
}

pub fn make_train_batch(train_base_seed: u64, step: u64) -> (Tensor, Tensor) {
    make_temporal_batch(BATCH_SIZE, train_base_seed + step)
}

pub fn make_validation_batch(validation_base_seed: u64, batch_idx: u64) -> (Tensor, Tensor) {
    make_temporal_batch(BATCH_SIZE, validation_base_seed + batch_idx)
}

pub fn motion_mode_counts(x_t: &Tensor, x_t1: &Tensor) -> (usize, usize) {
    let mut slow_count = 0;
    let mut fast_count = 0;

    for sample in 0..BATCH_SIZE {
        match motion_dx_for_sample(x_t, x_t1, sample) {
            SLOW_MOTION_DX => slow_count += 1,
            FAST_MOTION_DX => fast_count += 1,
            dx => panic!("unexpected motion dx {}", dx),
        }
    }

    (slow_count, fast_count)
}

#[cfg(test)]
#[allow(dead_code)]
pub fn batch_has_motion_mode(x_t: &Tensor, x_t1: &Tensor, motion_dx: usize) -> bool {
    let (slow_count, fast_count) = motion_mode_counts(x_t, x_t1);

    match motion_dx {
        SLOW_MOTION_DX => slow_count > 0,
        FAST_MOTION_DX => fast_count > 0,
        _ => false,
    }
}

#[cfg(test)]
#[allow(dead_code)]
pub fn batch_has_both_motion_modes(x_t: &Tensor, x_t1: &Tensor) -> bool {
    batch_has_motion_mode(x_t, x_t1, SLOW_MOTION_DX)
        && batch_has_motion_mode(x_t, x_t1, FAST_MOTION_DX)
}

pub fn batch_has_min_motion_mode_counts(
    x_t: &Tensor,
    x_t1: &Tensor,
    min_slow_count: usize,
    min_fast_count: usize,
) -> bool {
    let (slow_count, fast_count) = motion_mode_counts(x_t, x_t1);
    slow_count >= min_slow_count && fast_count >= min_fast_count
}

pub fn make_validation_batch_with_both_motion_modes(
    validation_base_seed: u64,
    start_batch_idx: u64,
) -> (Tensor, Tensor, u64) {
    for offset in 0..MIXED_MODE_SEARCH_LIMIT {
        let batch_idx = start_batch_idx + offset;
        let seed = validation_base_seed + batch_idx;
        let (x_t, x_t1) = make_temporal_batch(BATCH_SIZE, seed);

        if batch_has_min_motion_mode_counts(&x_t, &x_t1, MIN_MIXED_MODE_COUNT, MIN_MIXED_MODE_COUNT)
        {
            return (x_t, x_t1, seed);
        }
    }

    panic!(
        "did not find a mixed-mode validation batch with at least {} slow and {} fast samples within {} seeds from base {} and start batch {}",
        MIN_MIXED_MODE_COUNT,
        MIN_MIXED_MODE_COUNT,
        MIXED_MODE_SEARCH_LIMIT,
        validation_base_seed,
        start_batch_idx
    );
}

pub fn square_center_x(tensor: &Tensor, sample: usize) -> f32 {
    let mut weighted_sum = 0.0;
    let mut total_mass = 0.0;

    for row in 0..IMAGE_SIZE {
        for col in 0..IMAGE_SIZE {
            let value = tensor.get(&[sample, 0, row, col]);
            weighted_sum += value * col as f32;
            total_mass += value;
        }
    }

    weighted_sum / total_mass
}

#[allow(dead_code)]
pub fn fast_motion_feature_from_mass(mass: f32) -> f32 {
    (mass - FAST_MOTION_MASS_THRESHOLD).max(0.0)
}

#[allow(dead_code)]
pub fn fast_motion_feature_for_sample(x_t: &Tensor, sample: usize) -> f32 {
    fast_motion_feature_from_mass(total_mass(x_t, sample))
}

#[allow(dead_code)]
pub fn fast_mode_channel_summary(z: &Tensor) -> (usize, usize, f32, f32) {
    assert_eq!(
        z.shape,
        vec![BATCH_SIZE, 3],
        "fast_mode_channel_summary expects [B, 3], got {:?}",
        z.shape
    );

    let mut inactive_count = 0;
    let mut active_count = 0;
    let mut sum = 0.0;
    let mut max_value: f32 = 0.0;

    for sample in 0..BATCH_SIZE {
        let value = z.get(&[sample, 2]);
        sum += value;
        max_value = max_value.max(value);

        if value > 0.0 {
            active_count += 1;
        } else {
            inactive_count += 1;
        }
    }

    (
        inactive_count,
        active_count,
        sum / BATCH_SIZE as f32,
        max_value,
    )
}

pub fn assert_temporal_contract(x_t: &Tensor, x_t1: &Tensor) {
    assert_eq!(
        x_t.shape,
        vec![BATCH_SIZE, CHANNELS, IMAGE_SIZE, IMAGE_SIZE]
    );
    assert_eq!(
        x_t1.shape,
        vec![BATCH_SIZE, CHANNELS, IMAGE_SIZE, IMAGE_SIZE]
    );

    for sample in 0..BATCH_SIZE {
        let center_x_t = square_center_x(x_t, sample);
        let center_x_t1 = square_center_x(x_t1, sample);
        let delta_x = center_x_t1 - center_x_t;
        let expected_motion_dx = motion_dx_for_sample(x_t, x_t1, sample) as f32;
        let mass_t = total_mass(x_t, sample);
        let mass_t1 = total_mass(x_t1, sample);

        assert!(
            (delta_x - expected_motion_dx).abs() < 1e-6,
            "sample {} moved by {:.3}, expected {:.3}",
            sample,
            delta_x,
            expected_motion_dx
        );
        assert!(
            mass_t1 < mass_t,
            "sample {} mass did not follow the deterministic intensity rule: {:.3} -> {:.3}",
            sample,
            mass_t,
            mass_t1
        );
    }
}

pub fn print_batch_summary(name: &str, x_t: &Tensor, x_t1: &Tensor) {
    let (slow_count, fast_count) = motion_mode_counts(x_t, x_t1);

    println!(
        "{} | shape {:?} | sample0 dx {} | slow {} | fast {} | center_x {:.3} -> {:.3} | mass {:.3} -> {:.3}",
        name,
        x_t.shape,
        motion_dx_for_sample(x_t, x_t1, 0),
        slow_count,
        fast_count,
        square_center_x(x_t, 0),
        square_center_x(x_t1, 0),
        total_mass(x_t, 0),
        total_mass(x_t1, 0)
    );
}

pub fn make_frozen_encoder() -> EmbeddingEncoder {
    let mut conv1_weights = Vec::with_capacity(3 * IMAGE_SIZE * IMAGE_SIZE);

    for _row in 0..IMAGE_SIZE {
        for _col in 0..IMAGE_SIZE {
            conv1_weights.push(1.0);
        }
    }

    for _row in 0..IMAGE_SIZE {
        for col in 0..IMAGE_SIZE {
            conv1_weights.push((col as f32 + 1.0) / IMAGE_SIZE as f32);
        }
    }

    for _row in 0..IMAGE_SIZE {
        for _col in 0..IMAGE_SIZE {
            conv1_weights.push(1.0);
        }
    }

    let conv1 = Conv2d::new(
        Tensor::new(conv1_weights, vec![3, 1, IMAGE_SIZE, IMAGE_SIZE]),
        Tensor::new(vec![0.0, 0.0, -FAST_MOTION_MASS_THRESHOLD], vec![3]),
        1,
        0,
    );

    let conv2 = Conv2d::new(
        Tensor::new(
            vec![
                1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0,
            ],
            vec![3, 3, 1, 1],
        ),
        Tensor::new(vec![0.0, 0.0, 0.0], vec![3]),
        1,
        0,
    );

    EmbeddingEncoder::new(ConvEncoder::new(conv1, conv2))
}

fn draw_square(tensor: &mut Tensor, sample: usize, row: usize, col: usize, intensity: f32) {
    for dy in 0..SQUARE_SIZE {
        for dx in 0..SQUARE_SIZE {
            tensor.set(&[sample, 0, row + dy, col + dx], intensity);
        }
    }
}

fn total_mass(tensor: &Tensor, sample: usize) -> f32 {
    let mut total = 0.0;

    for row in 0..IMAGE_SIZE {
        for col in 0..IMAGE_SIZE {
            total += tensor.get(&[sample, 0, row, col]);
        }
    }

    total
}
