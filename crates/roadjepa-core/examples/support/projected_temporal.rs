use roadjepa_core::{EmbeddingEncoder, Linear, Predictor, Tensor, mse_loss, mse_loss_grad};

pub fn gaussian_moment_regularizer(latents: &Tensor) -> f32 {
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

pub fn gaussian_moment_regularizer_grad(latents: &Tensor) -> Tensor {
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

pub fn projection_stats(latents: &Tensor) -> (f32, f32) {
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

pub fn combine_projection_grads(
    prediction_grad: &Tensor,
    regularizer_grad: &Tensor,
    regularizer_weight: f32,
) -> Tensor {
    assert_eq!(
        prediction_grad.shape, regularizer_grad.shape,
        "projection grad shape mismatch: {:?} vs {:?}",
        prediction_grad.shape, regularizer_grad.shape
    );

    let data = prediction_grad
        .data
        .iter()
        .zip(regularizer_grad.data.iter())
        .map(|(prediction, regularizer)| prediction + regularizer_weight * regularizer)
        .collect();

    Tensor::new(data, prediction_grad.shape.clone())
}

#[allow(dead_code)]
pub fn projected_target(
    encoder: &EmbeddingEncoder,
    target_projector: &Linear,
    x_t1: &Tensor,
) -> Tensor {
    let z_t1 = encoder.forward(x_t1);
    target_projector.forward(&z_t1)
}

#[allow(dead_code)]
pub fn projected_batch_losses(
    encoder: &EmbeddingEncoder,
    online_projector: &Linear,
    target_projector: &Linear,
    predictor: &Predictor,
    x_t: &Tensor,
    x_t1: &Tensor,
    regularizer_weight: f32,
) -> (f32, f32, f32) {
    let z_t = encoder.forward(x_t);
    let projection_t = online_projector.forward(&z_t);
    let pred = predictor.forward(&projection_t);
    let target = projected_target(encoder, target_projector, x_t1);
    let prediction_loss = mse_loss(&pred, &target);
    let regularizer_loss = gaussian_moment_regularizer(&projection_t);
    let total_loss = prediction_loss + regularizer_weight * regularizer_loss;

    (prediction_loss, regularizer_loss, total_loss)
}

#[allow(dead_code)]
pub fn projected_step(
    encoder: &EmbeddingEncoder,
    online_projector: &mut Linear,
    target_projector: &Linear,
    predictor: &mut Predictor,
    x_t: &Tensor,
    x_t1: &Tensor,
    regularizer_weight: f32,
    predictor_lr: f32,
    projector_lr: f32,
) {
    let z_t = encoder.forward(x_t);
    let projection_t = online_projector.forward(&z_t);
    let target = projected_target(encoder, target_projector, x_t1);
    let pred = predictor.forward(&projection_t);
    let prediction_grad = mse_loss_grad(&pred, &target);
    let reg_grad = gaussian_moment_regularizer_grad(&projection_t);
    let pred_grads = predictor.backward(&projection_t, &prediction_grad);
    let projection_grad =
        combine_projection_grads(&pred_grads.grad_input, &reg_grad, regularizer_weight);

    let projector_grads = online_projector.backward(&z_t, &projection_grad);

    predictor.sgd_step(&pred_grads, predictor_lr);
    online_projector.sgd_step(&projector_grads, projector_lr);
}
