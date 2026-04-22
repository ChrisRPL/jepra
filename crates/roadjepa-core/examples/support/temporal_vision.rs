use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use roadjepa_core::{Conv2d, ConvEncoder, EmbeddingEncoder, Tensor};

pub const BATCH_SIZE: usize = 8;
pub const IMAGE_SIZE: usize = 8;
pub const CHANNELS: usize = 1;
pub const SQUARE_SIZE: usize = 2;
pub const MOTION_DX: usize = 1;

pub fn make_temporal_batch(batch_size: usize, seed: u64) -> (Tensor, Tensor) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x_t = Tensor::zeros(vec![batch_size, CHANNELS, IMAGE_SIZE, IMAGE_SIZE]);
    let mut x_t1 = Tensor::zeros(vec![batch_size, CHANNELS, IMAGE_SIZE, IMAGE_SIZE]);

    let max_row = IMAGE_SIZE - SQUARE_SIZE;
    let max_col_t = IMAGE_SIZE - SQUARE_SIZE - MOTION_DX;

    for sample in 0..batch_size {
        let row = rng.gen_range(0..=max_row);
        let col_t = rng.gen_range(0..=max_col_t);
        let col_t1 = col_t + MOTION_DX;
        let intensity_t = rng.gen_range(0.65f32..0.95f32);
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
        let mass_t = total_mass(x_t, sample);
        let mass_t1 = total_mass(x_t1, sample);

        assert!(
            center_x_t1 > center_x_t,
            "sample {} did not move right: {:.3} -> {:.3}",
            sample,
            center_x_t,
            center_x_t1
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
    println!(
        "{} | shape {:?} | sample0 center_x {:.3} -> {:.3} | mass {:.3} -> {:.3}",
        name,
        x_t.shape,
        square_center_x(x_t, 0),
        square_center_x(x_t1, 0),
        total_mass(x_t, 0),
        total_mass(x_t1, 0)
    );
}

pub fn make_frozen_encoder() -> EmbeddingEncoder {
    let mut conv1_weights = Vec::with_capacity(2 * IMAGE_SIZE * IMAGE_SIZE);

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

    let conv1 = Conv2d::new(
        Tensor::new(conv1_weights, vec![2, 1, IMAGE_SIZE, IMAGE_SIZE]),
        Tensor::new(vec![0.0, 0.0], vec![2]),
        1,
        0,
    );

    let conv2 = Conv2d::new(
        Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2, 1, 1]),
        Tensor::new(vec![0.0, 0.0], vec![2]),
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
