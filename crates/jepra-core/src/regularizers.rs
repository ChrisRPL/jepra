use crate::tensor::Tensor;

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
