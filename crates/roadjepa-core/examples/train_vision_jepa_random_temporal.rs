use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use roadjepa_core::{
    mse_loss, mse_loss_grad, Conv2d, ConvEncoder, EmbeddingEncoder, Linear, Predictor, Tensor,
    VisionJepa,
};

const BATCH_SIZE: usize = 8;
const IMAGE_SIZE: usize = 8;
const CHANNELS: usize = 1;
const SQUARE_SIZE: usize = 2;
const MOTION_DX: usize = 1;
const TRAIN_BASE_SEED: u64 = 1_000;
const VALIDATION_BASE_SEED: u64 = 99_001;
const VALIDATION_BATCHES: usize = 8;
const NUM_STEPS: usize = 300;
const LOG_EVERY: usize = 25;
const LR: f32 = 0.02;

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
        let intensity_t1 = (0.9f32 * intensity_t + 0.05f32).clamp(0.5f32, 1.0f32);

        draw_square(&mut x_t, sample, row, col_t, intensity_t);
        draw_square(&mut x_t1, sample, row, col_t1, intensity_t1);
    }

    (x_t, x_t1)
}

fn make_train_batch(step: u64) -> (Tensor, Tensor) {
    make_temporal_batch(BATCH_SIZE, TRAIN_BASE_SEED + step)
}

fn make_validation_batch(batch_idx: u64) -> (Tensor, Tensor) {
    make_temporal_batch(BATCH_SIZE, VALIDATION_BASE_SEED + batch_idx)
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

fn print_batch_summary(name: &str, x_t: &Tensor, x_t1: &Tensor) {
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

fn make_frozen_encoder() -> EmbeddingEncoder {
    // Full-image asymmetric kernels leak absolute horizontal position before pooling.
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

fn make_predictor() -> Predictor {
    let fc1 = Linear::new(
        Tensor::new(
            vec![
                1.0, 0.0, 1.0, 0.0, //
                0.0, 1.0, 0.0, 1.0,
            ],
            vec![2, 4],
        ),
        Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![4]),
    );

    let fc2 = Linear::new(
        Tensor::zeros(vec![4, 2]),
        Tensor::new(vec![0.0, 0.0], vec![2]),
    );

    Predictor::new(fc1, fc2)
}

fn batch_loss(model: &VisionJepa, x_t: &Tensor, x_t1: &Tensor) -> f32 {
    let pred = model.predict_next_latent(x_t);
    let target = model.target_latent(x_t1);
    mse_loss(&pred, &target)
}

fn validation_loss(model: &VisionJepa) -> f32 {
    let mut total = 0.0;

    for batch_idx in 0..VALIDATION_BATCHES {
        let (x_t, x_t1) = make_validation_batch(batch_idx as u64);
        total += batch_loss(model, &x_t, &x_t1);
    }

    total / VALIDATION_BATCHES as f32
}

fn main() {
    let (train_probe_t, train_probe_t1) = make_train_batch(0);
    let (train_probe_next_t, train_probe_next_t1) = make_train_batch(1);
    let (val_probe_t, val_probe_t1) = make_validation_batch(0);

    assert_temporal_contract(&train_probe_t, &train_probe_t1);
    assert_temporal_contract(&train_probe_next_t, &train_probe_next_t1);
    assert_temporal_contract(&val_probe_t, &val_probe_t1);

    assert_ne!(train_probe_t.data, train_probe_next_t.data);
    assert_ne!(train_probe_t.data, val_probe_t.data);

    print_batch_summary("train probe", &train_probe_t, &train_probe_t1);
    print_batch_summary("validation probe", &val_probe_t, &val_probe_t1);

    let encoder = make_frozen_encoder();
    let predictor = make_predictor();
    let mut model = VisionJepa::new(encoder, predictor);

    let initial_z_t = model.encode(&train_probe_t);
    let initial_z_t1 = model.target_latent(&train_probe_t1);
    let initial_train_loss = batch_loss(&model, &train_probe_t, &train_probe_t1);
    let initial_val_loss = validation_loss(&model);

    println!(
        "initial | latent sample0 {:?} -> {:?}",
        &initial_z_t.data[0..2],
        &initial_z_t1.data[0..2]
    );
    println!(
        "initial | probe train {:.6} | val {:.6}",
        initial_train_loss, initial_val_loss
    );

    for step in 1..=NUM_STEPS {
        let (x_t, x_t1) = make_train_batch(step as u64);
        let z_t = model.encode(&x_t);
        let z_t1 = model.target_latent(&x_t1);
        let pred = model.predictor.forward(&z_t);
        let train_loss = mse_loss(&pred, &z_t1);

        let grad_out = mse_loss_grad(&pred, &z_t1);
        let grads = model.predictor.backward(&z_t, &grad_out);
        model.predictor.sgd_step(&grads, LR);

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

    println!(
        "\nfinal | latent sample0 {:?} | pred {:?} | target {:?}",
        &final_z_t.data[0..2],
        &final_pred.data[0..2],
        &final_target.data[0..2]
    );
    println!(
        "final | probe train {:.6} | val {:.6}",
        final_train_loss, final_val_loss
    );
}
