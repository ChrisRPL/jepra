use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use roadjepa_core::Tensor;

const BATCH_SIZE: usize = 8;
const IMAGE_SIZE: usize = 8;
const CHANNELS: usize = 1;
const SQUARE_SIZE: usize = 2;
const MOTION_DX: usize = 1;
const TRAIN_BASE_SEED: u64 = 1_000;
const VALIDATION_SEED: u64 = 99_001;

fn make_temporal_batch(batch_size: usize, seed: u64) -> (Tensor, Tensor) {
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
        let intensity_t1 = (intensity_t + rng.gen_range(-0.05f32..0.05f32)).clamp(0.5f32, 1.0f32);

        draw_square(&mut x_t, sample, row, col_t, intensity_t);
        draw_square(&mut x_t1, sample, row, col_t1, intensity_t1);
    }

    (x_t, x_t1)
}

fn make_train_batch(step: u64) -> (Tensor, Tensor) {
    make_temporal_batch(BATCH_SIZE, TRAIN_BASE_SEED + step)
}

fn make_validation_batch() -> (Tensor, Tensor) {
    make_temporal_batch(BATCH_SIZE, VALIDATION_SEED)
}

fn draw_square(tensor: &mut Tensor, sample: usize, row: usize, col: usize, intensity: f32) {
    for dy in 0..SQUARE_SIZE {
        for dx in 0..SQUARE_SIZE {
            tensor.set(&[sample, 0, row + dy, col + dx], intensity);
        }
    }
}

fn square_center_x(tensor: &Tensor, sample: usize) -> f32 {
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

fn assert_temporal_contract(x_t: &Tensor, x_t1: &Tensor) {
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

        assert!(
            center_x_t1 > center_x_t,
            "sample {} did not move right: {:.3} -> {:.3}",
            sample,
            center_x_t,
            center_x_t1
        );
    }
}

fn print_batch_summary(name: &str, x_t: &Tensor, x_t1: &Tensor) {
    let center_x_t = square_center_x(x_t, 0);
    let center_x_t1 = square_center_x(x_t1, 0);

    println!(
        "{} | shape {:?} | sample0 center_x {:.3} -> {:.3}",
        name, x_t.shape, center_x_t, center_x_t1
    );
}

fn main() {
    // Keep motion simple and inspectable: every square moves exactly one pixel to the right.
    let (train_step0_t, train_step0_t1) = make_train_batch(0);
    let (train_step1_t, train_step1_t1) = make_train_batch(1);
    let (val_t, val_t1) = make_validation_batch();

    assert_temporal_contract(&train_step0_t, &train_step0_t1);
    assert_temporal_contract(&train_step1_t, &train_step1_t1);
    assert_temporal_contract(&val_t, &val_t1);

    assert_ne!(train_step0_t.data, train_step1_t.data);
    assert_ne!(train_step0_t.data, val_t.data);

    print_batch_summary("train step 0", &train_step0_t, &train_step0_t1);
    print_batch_summary("train step 1", &train_step1_t, &train_step1_t1);
    print_batch_summary("validation", &val_t, &val_t1);
}
