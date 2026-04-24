use crate::tensor::Tensor;

#[derive(Debug, Clone, PartialEq)]
pub struct RepresentationHealthStats {
    pub per_dim_std: Vec<f32>,
    pub mean_abs: f32,
    pub mean_std: f32,
    pub min_std: f32,
    pub mean_abs_offdiag_cov: f32,
    pub max_abs_offdiag_cov: f32,
}

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

pub fn representation_health_stats(latents: &Tensor) -> RepresentationHealthStats {
    assert!(
        latents.shape.len() == 2,
        "representation_health_stats expects [B, D], got {:?}",
        latents.shape
    );

    let batch_size = latents.shape[0];
    let dim = latents.shape[1];
    assert!(
        batch_size > 0,
        "representation_health_stats expects batch size > 0"
    );
    assert!(
        dim > 0,
        "representation_health_stats expects feature dim > 0"
    );

    let batch_scale = batch_size as f32;
    let dim_scale = dim as f32;
    let mut means = vec![0.0f32; dim];
    let mut variances = vec![0.0f32; dim];

    for feature in 0..dim {
        for sample in 0..batch_size {
            means[feature] += latents.get(&[sample, feature]);
        }
        means[feature] /= batch_scale;

        for sample in 0..batch_size {
            let centered = latents.get(&[sample, feature]) - means[feature];
            variances[feature] += centered * centered;
        }
        variances[feature] /= batch_scale;
    }

    let mean_abs = means.iter().map(|mean| mean.abs()).sum::<f32>() / dim_scale;
    let per_dim_std = variances
        .iter()
        .map(|variance| variance.sqrt())
        .collect::<Vec<_>>();
    let mean_std = per_dim_std.iter().sum::<f32>() / dim_scale;
    let min_std = per_dim_std.iter().copied().fold(f32::INFINITY, f32::min);
    let mut offdiag_cov_abs_sum = 0.0f32;
    let mut offdiag_cov_count = 0usize;
    let mut max_abs_offdiag_cov = 0.0f32;

    for left in 0..dim {
        for right in (left + 1)..dim {
            let mut covariance = 0.0f32;
            for sample in 0..batch_size {
                let left_centered = latents.get(&[sample, left]) - means[left];
                let right_centered = latents.get(&[sample, right]) - means[right];
                covariance += left_centered * right_centered;
            }
            covariance /= batch_scale;
            let abs_covariance = covariance.abs();
            offdiag_cov_abs_sum += abs_covariance;
            max_abs_offdiag_cov = max_abs_offdiag_cov.max(abs_covariance);
            offdiag_cov_count += 1;
        }
    }

    let mean_abs_offdiag_cov = if offdiag_cov_count == 0 {
        0.0
    } else {
        offdiag_cov_abs_sum / offdiag_cov_count as f32
    };

    RepresentationHealthStats {
        per_dim_std,
        mean_abs,
        mean_std,
        min_std,
        mean_abs_offdiag_cov,
        max_abs_offdiag_cov,
    }
}

pub fn representation_stats(latents: &Tensor) -> RepresentationHealthStats {
    representation_health_stats(latents)
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
