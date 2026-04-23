use jepra_core::{Conv2d, ConvEncoder, EmbeddingEncoder, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub const BATCH_SIZE: usize = 8;
pub const IMAGE_SIZE: usize = 8;
pub const CHANNELS: usize = 1;
pub const SQUARE_SIZE: usize = 2;
pub const SLOW_MOTION_DX: usize = 1;
pub const FAST_MOTION_DX: usize = 2;
pub const FAST_MOTION_MASS_THRESHOLD: f32 = 0.8f32 * (SQUARE_SIZE * SQUARE_SIZE) as f32;
pub const EXTRA_SQUARE_CHANCE: f64 = 0.5;
pub const MIXED_MODE_SEARCH_LIMIT: u64 = 64;
pub const MIN_MIXED_MODE_COUNT: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemporalRunConfig {
    pub train_base_seed: u64,
    pub total_steps: usize,
    pub log_every: usize,
}

impl TemporalRunConfig {
    pub fn from_args(
        default_train_base_seed: u64,
        default_total_steps: usize,
        default_log_every: usize,
    ) -> Self {
        let args: Vec<String> = std::env::args().collect();

        let train_base_seed = parse_u64_arg(&args, "--train-base-seed")
            .or_else(|| parse_u64_arg(&args, "--seed"))
            .unwrap_or(default_train_base_seed);

        let total_steps = parse_usize_arg(&args, "--train-steps")
            .or_else(|| parse_usize_arg(&args, "--steps"))
            .or_else(|| Some(training_steps(default_total_steps)))
            .unwrap_or(default_total_steps);

        let log_every = parse_usize_arg(&args, "--log-every")
            .or_else(|| parse_usize_arg(&args, "--log"))
            .unwrap_or(default_log_every);

        assert!(
            total_steps > 0,
            "train steps must be greater than 0, got {}",
            total_steps
        );
        assert!(
            log_every > 0,
            "log_every must be greater than 0, got {}",
            log_every
        );

        Self {
            train_base_seed,
            total_steps,
            log_every,
        }
    }
}

pub fn training_steps(default_steps: usize) -> usize {
    std::env::var("JEPRA_TRAIN_STEPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&steps| steps > 0)
        .unwrap_or(default_steps)
}

fn parse_arg_value<'a>(args: &'a [String], flag: &'a str) -> Option<&'a str> {
    args.windows(2).find_map(|window| {
        if window[0] == flag {
            Some(window[1].as_str())
        } else {
            None
        }
    })
}

fn parse_u64_arg(args: &[String], flag: &str) -> Option<u64> {
    parse_arg_value(args, flag).and_then(|value| value.parse::<u64>().ok())
}

fn parse_usize_arg(args: &[String], flag: &str) -> Option<usize> {
    parse_arg_value(args, flag)
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
}

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

        if rng.gen_bool(EXTRA_SQUARE_CHANCE) {
            let mut secondary_row = rng.gen_range(0..=max_row);
            let mut secondary_col_t = rng.gen_range(0..=max_col_t);
            let mut attempts = 0usize;

            while squares_overlap(row, col_t, secondary_row, secondary_col_t) && attempts < 16 {
                secondary_row = rng.gen_range(0..=max_row);
                secondary_col_t = rng.gen_range(0..=max_col_t);
                attempts += 1;
            }

            let secondary_intensity_t = intensity_t;
            let secondary_intensity_t1 =
                (0.9f32 * secondary_intensity_t + 0.05f32).clamp(0.5f32, 1.0f32);

            draw_square(
                &mut x_t,
                sample,
                secondary_row,
                secondary_col_t,
                secondary_intensity_t,
            );
            draw_square(
                &mut x_t1,
                sample,
                secondary_row,
                secondary_col_t + motion_dx,
                secondary_intensity_t1,
            );
        }
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

pub fn run_temporal_training_loop<TStep>(total_steps: usize, log_every: usize, mut step_fn: TStep)
where
    TStep: FnMut(usize, bool),
{
    assert!(total_steps > 0, "total_steps must be greater than 0");
    assert!(log_every > 0, "log_every must be greater than 0");

    for step in 1..=total_steps {
        let should_log = should_log_step(step, log_every);
        step_fn(step, should_log);
    }
}

#[allow(dead_code)]
pub fn print_temporal_train_val_metrics(step: usize, train_loss: f32, val_loss: f32) {
    println!(
        "step {:03} | train {:.6} | val {:.6}",
        step, train_loss, val_loss
    );
}

#[allow(dead_code)]
pub fn print_projected_temporal_train_val_metrics(
    step: usize,
    prediction_loss: f32,
    regularizer_loss: f32,
    total_loss: f32,
    val_prediction_loss: f32,
    val_regularizer_loss: f32,
    val_total_loss: f32,
) {
    println!(
        "step {:03} | train pred {:.6} | reg {:.6} | total {:.6} | val total {:.6}",
        step, prediction_loss, regularizer_loss, total_loss, val_total_loss
    );
    println!(
        "step {:03} | val pred {:.6} | reg {:.6}",
        step, val_prediction_loss, val_regularizer_loss
    );
}

pub fn should_log_step(step: usize, log_every: usize) -> bool {
    assert!(log_every > 0, "log_every must be greater than 0");
    step == 1 || step % log_every == 0
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

fn squares_overlap(row_a: usize, col_a: usize, row_b: usize, col_b: usize) -> bool {
    let row_a_end = row_a + SQUARE_SIZE;
    let col_a_end = col_a + SQUARE_SIZE;
    let row_b_end = row_b + SQUARE_SIZE;
    let col_b_end = col_b + SQUARE_SIZE;

    row_a < row_b_end && row_b < row_a_end && col_a < col_b_end && col_b < col_a_end
}

pub fn total_mass(tensor: &Tensor, sample: usize) -> f32 {
    let mut total = 0.0;

    for row in 0..IMAGE_SIZE {
        for col in 0..IMAGE_SIZE {
            total += tensor.get(&[sample, 0, row, col]);
        }
    }

    total
}

#[cfg(test)]
pub fn active_cell_count(tensor: &Tensor, sample: usize) -> usize {
    let mut count = 0usize;

    for row in 0..IMAGE_SIZE {
        for col in 0..IMAGE_SIZE {
            if tensor.get(&[sample, 0, row, col]).abs() > 1e-6 {
                count += 1;
            }
        }
    }

    count
}

#[cfg(test)]
pub fn assert_square_footprint_and_decay_invariants(
    x_t: &Tensor,
    x_t1: &Tensor,
    seed: u64,
) -> (usize, usize) {
    let mut single_square_samples = 0usize;
    let mut double_square_samples = 0usize;

    for sample in 0..BATCH_SIZE {
        let cells_t = active_cell_count(x_t, sample);
        let cells_t1 = active_cell_count(x_t1, sample);

        match cells_t {
            4 => single_square_samples += 1,
            8 => double_square_samples += 1,
            cells => panic!(
                "unexpected non-zero footprint at seed {} sample {}: {}",
                seed, sample, cells
            ),
        }

        assert!(
            cells_t == cells_t1,
            "non-zero footprint changed across temporal step at seed {} sample {}: {} -> {}",
            seed,
            sample,
            cells_t,
            cells_t1
        );

        let mass_t = total_mass(x_t, sample);
        let mass_t1 = total_mass(x_t1, sample);
        assert!(
            mass_t1 < mass_t,
            "sample {} mass did not decay at seed {}: {:.6} -> {:.6}",
            sample,
            seed,
            mass_t,
            mass_t1
        );
    }

    (single_square_samples, double_square_samples)
}

#[cfg(test)]
pub fn assert_seed_range_has_single_and_double_square_batch_examples(
    seed_count: u64,
    mut make_batch: impl FnMut(u64) -> (Tensor, Tensor),
) -> (usize, usize) {
    let mut saw_single_square_batches = 0usize;
    let mut saw_double_square_batches = 0usize;

    for seed in 0..seed_count {
        let (x_t, x_t1) = make_batch(seed);
        let (single_count, double_count) =
            assert_square_footprint_and_decay_invariants(&x_t, &x_t1, seed);

        saw_single_square_batches += usize::from(single_count > 0);
        saw_double_square_batches += usize::from(double_count > 0);
    }

    assert!(
        saw_single_square_batches > 0,
        "never saw single-square sample in {} seeds",
        seed_count
    );
    assert!(
        saw_double_square_batches > 0,
        "never saw double-square sample in {} seeds",
        seed_count
    );

    (saw_single_square_batches, saw_double_square_batches)
}

#[cfg(test)]
pub fn assert_seed_range_has_both_motion_modes(
    seed_count: u64,
    mut make_batch: impl FnMut(u64) -> (Tensor, Tensor),
) -> (bool, bool) {
    let mut saw_slow_motion = false;
    let mut saw_fast_motion = false;

    for seed in 0..seed_count {
        let (x_t, x_t1) = make_batch(seed);

        for sample in 0..BATCH_SIZE {
            match motion_dx_for_sample(&x_t, &x_t1, sample) {
                SLOW_MOTION_DX => saw_slow_motion = true,
                FAST_MOTION_DX => saw_fast_motion = true,
                dx => panic!(
                    "unexpected motion dx {} in seed {} sample {}",
                    dx, seed, sample
                ),
            }
        }
    }

    assert!(
        saw_slow_motion,
        "never observed slow motion in {} seeds",
        seed_count
    );
    assert!(
        saw_fast_motion,
        "never observed fast motion in {} seeds",
        seed_count
    );

    (saw_slow_motion, saw_fast_motion)
}
