#[path = "support/temporal_vision.rs"]
mod temporal_vision;

use roadjepa_core::{mse_loss, mse_loss_grad, EmbeddingEncoder, Linear, Predictor, Tensor};
use temporal_vision::{
    assert_temporal_contract, make_frozen_encoder, make_train_batch, make_validation_batch,
    print_batch_summary,
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

fn make_projector() -> Linear {
    Linear::new(
        Tensor::new(
            vec![
                1.0, 0.0, 1.0, 0.0, //
                0.0, 1.0, 0.0, 1.0,
            ],
            vec![2, PROJECTION_DIM],
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

fn projected_target(
    encoder: &EmbeddingEncoder,
    target_projector: &Linear,
    x_t1: &Tensor,
) -> Tensor {
    let z_t1 = encoder.forward(x_t1);
    target_projector.forward(&z_t1)
}

fn gaussian_moment_regularizer(latents: &Tensor) -> f32 {
    assert!(
        latents.shape.len() == 2,
        "gaussian_moment_regularizer expects [B, D], got {:?}",
        latents.shape
    );

    let batch_size = latents.shape[0];
    let dim = latents.shape[1];
    let batch_scale = batch_size as f32;
    let dim_scale = dim as f32;
    let mut loss = 0.0;

    for feature in 0..dim {
        let mut mean = 0.0;
        let mut variance = 0.0;

        for sample in 0..batch_size {
            mean += latents.get(&[sample, feature]);
        }

        mean /= batch_scale;

        for sample in 0..batch_size {
            let centered = latents.get(&[sample, feature]) - mean;
            variance += centered * centered;
        }

        variance /= batch_scale;
        loss += mean * mean + (variance - 1.0) * (variance - 1.0);
    }

    loss / dim_scale
}

fn gaussian_moment_regularizer_grad(latents: &Tensor) -> Tensor {
    assert!(
        latents.shape.len() == 2,
        "gaussian_moment_regularizer_grad expects [B, D], got {:?}",
        latents.shape
    );

    let batch_size = latents.shape[0];
    let dim = latents.shape[1];
    let batch_scale = batch_size as f32;
    let dim_scale = dim as f32;
    let mut grad = Tensor::zeros(latents.shape.clone());

    for feature in 0..dim {
        let mut mean = 0.0;
        let mut variance = 0.0;

        for sample in 0..batch_size {
            mean += latents.get(&[sample, feature]);
        }

        mean /= batch_scale;

        for sample in 0..batch_size {
            let centered = latents.get(&[sample, feature]) - mean;
            variance += centered * centered;
        }

        variance /= batch_scale;

        for sample in 0..batch_size {
            let centered = latents.get(&[sample, feature]) - mean;
            let grad_value =
                (2.0 / (batch_scale * dim_scale)) * (mean + 2.0 * centered * (variance - 1.0));
            grad.set(&[sample, feature], grad_value);
        }
    }

    grad
}

fn projection_stats(latents: &Tensor) -> (f32, f32) {
    assert!(
        latents.shape.len() == 2,
        "projection_stats expects [B, D], got {:?}",
        latents.shape
    );

    let batch_size = latents.shape[0];
    let dim = latents.shape[1];
    let batch_scale = batch_size as f32;
    let dim_scale = dim as f32;
    let mut mean_abs_acc = 0.0;
    let mut variance_acc = 0.0;

    for feature in 0..dim {
        let mut mean = 0.0;
        let mut variance = 0.0;

        for sample in 0..batch_size {
            mean += latents.get(&[sample, feature]);
        }

        mean /= batch_scale;

        for sample in 0..batch_size {
            let centered = latents.get(&[sample, feature]) - mean;
            variance += centered * centered;
        }

        variance /= batch_scale;
        mean_abs_acc += mean.abs();
        variance_acc += variance;
    }

    (mean_abs_acc / dim_scale, variance_acc / dim_scale)
}

fn combine_projection_grads(prediction_grad: &Tensor, regularizer_grad: &Tensor) -> Tensor {
    assert_eq!(
        prediction_grad.shape, regularizer_grad.shape,
        "projection grad shape mismatch: {:?} vs {:?}",
        prediction_grad.shape, regularizer_grad.shape
    );

    let data = prediction_grad
        .data
        .iter()
        .zip(regularizer_grad.data.iter())
        .map(|(prediction, regularizer)| prediction + REGULARIZER_WEIGHT * regularizer)
        .collect();

    Tensor::new(data, prediction_grad.shape.clone())
}

fn batch_losses(
    encoder: &EmbeddingEncoder,
    online_projector: &Linear,
    target_projector: &Linear,
    predictor: &Predictor,
    x_t: &Tensor,
    x_t1: &Tensor,
) -> (f32, f32, f32) {
    let z_t = encoder.forward(x_t);
    let projection_t = online_projector.forward(&z_t);
    let pred = predictor.forward(&projection_t);
    let target = projected_target(encoder, target_projector, x_t1);
    let prediction_loss = mse_loss(&pred, &target);
    let regularizer_loss = gaussian_moment_regularizer(&projection_t);
    let total_loss = prediction_loss + REGULARIZER_WEIGHT * regularizer_loss;

    (prediction_loss, regularizer_loss, total_loss)
}

fn validation_losses(
    encoder: &EmbeddingEncoder,
    online_projector: &Linear,
    target_projector: &Linear,
    predictor: &Predictor,
) -> (f32, f32, f32) {
    let mut prediction_total = 0.0;
    let mut regularizer_total = 0.0;
    let mut total = 0.0;

    for batch_idx in 0..VALIDATION_BATCHES {
        let (x_t, x_t1) = make_validation_batch(VALIDATION_BASE_SEED, batch_idx as u64);
        let (prediction_loss, regularizer_loss, total_loss) = batch_losses(
            encoder,
            online_projector,
            target_projector,
            predictor,
            &x_t,
            &x_t1,
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

fn main() {
    let (train_probe_t, train_probe_t1) = make_train_batch(TRAIN_BASE_SEED, 0);
    let (train_probe_next_t, train_probe_next_t1) = make_train_batch(TRAIN_BASE_SEED, 1);
    let (val_probe_t, val_probe_t1) = make_validation_batch(VALIDATION_BASE_SEED, 0);

    assert_temporal_contract(&train_probe_t, &train_probe_t1);
    assert_temporal_contract(&train_probe_next_t, &train_probe_next_t1);
    assert_temporal_contract(&val_probe_t, &val_probe_t1);

    assert_ne!(train_probe_t.data, train_probe_next_t.data);
    assert_ne!(train_probe_t.data, val_probe_t.data);

    print_batch_summary("train probe", &train_probe_t, &train_probe_t1);
    print_batch_summary("validation probe", &val_probe_t, &val_probe_t1);

    let encoder = make_frozen_encoder();
    let mut online_projector = make_projector();
    let target_projector = online_projector.clone();
    let mut predictor = make_predictor();

    let initial_z_t = encoder.forward(&train_probe_t);
    let initial_projection_t = online_projector.forward(&initial_z_t);
    let initial_target = projected_target(&encoder, &target_projector, &train_probe_t1);
    let (initial_train_prediction_loss, initial_train_regularizer_loss, initial_train_total_loss) =
        batch_losses(
            &encoder,
            &online_projector,
            &target_projector,
            &predictor,
            &train_probe_t,
            &train_probe_t1,
        );
    let (initial_val_prediction_loss, initial_val_regularizer_loss, initial_val_total_loss) =
        validation_losses(&encoder, &online_projector, &target_projector, &predictor);
    let (initial_projection_mean_abs, initial_projection_var_mean) =
        projection_stats(&initial_projection_t);

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
        let z_t = encoder.forward(&x_t);
        let projection_t = online_projector.forward(&z_t);
        let target = projected_target(&encoder, &target_projector, &x_t1);
        let pred = predictor.forward(&projection_t);
        let prediction_loss = mse_loss(&pred, &target);
        let regularizer_loss = gaussian_moment_regularizer(&projection_t);
        let total_loss = prediction_loss + REGULARIZER_WEIGHT * regularizer_loss;

        let prediction_grad = mse_loss_grad(&pred, &target);
        let predictor_grads = predictor.backward(&projection_t, &prediction_grad);
        let regularizer_grad = gaussian_moment_regularizer_grad(&projection_t);
        let projection_grad =
            combine_projection_grads(&predictor_grads.grad_input, &regularizer_grad);
        let projector_grads = online_projector.backward(&z_t, &projection_grad);

        predictor.sgd_step(&predictor_grads, PREDICTOR_LR);
        online_projector.sgd_step(&projector_grads, PROJECTOR_LR);

        if step == 1 || step % LOG_EVERY == 0 {
            let (val_prediction_loss, val_regularizer_loss, val_total_loss) =
                validation_losses(&encoder, &online_projector, &target_projector, &predictor);

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

    let final_projection_t = online_projector.forward(&initial_z_t);
    let final_pred = predictor.forward(&final_projection_t);
    let final_target = projected_target(&encoder, &target_projector, &train_probe_t1);
    let (final_train_prediction_loss, final_train_regularizer_loss, final_train_total_loss) =
        batch_losses(
            &encoder,
            &online_projector,
            &target_projector,
            &predictor,
            &train_probe_t,
            &train_probe_t1,
        );
    let (final_val_prediction_loss, final_val_regularizer_loss, final_val_total_loss) =
        validation_losses(&encoder, &online_projector, &target_projector, &predictor);
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
