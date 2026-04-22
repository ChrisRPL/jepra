use crate::encoder::EmbeddingEncoder;
use crate::linear::Linear;
use crate::losses::{mse_loss, mse_loss_grad};
use crate::predictor::Predictor;
use crate::tensor::Tensor;

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

fn combine_projection_grads(
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

#[derive(Debug, Clone, PartialEq)]
pub struct VisionJepa {
    pub encoder: EmbeddingEncoder,
    pub predictor: Predictor,
}

impl VisionJepa {
    pub fn new(encoder: EmbeddingEncoder, predictor: Predictor) -> Self {
        Self { encoder, predictor }
    }

    pub fn encode(&self, x: &Tensor) -> Tensor {
        self.encoder.forward(x)
    }

    pub fn predict_next_latent(&self, x_t: &Tensor) -> Tensor {
        let z_t = self.encode(x_t);
        self.predictor.forward(&z_t)
    }

    pub fn target_latent(&self, x_t1: &Tensor) -> Tensor {
        self.encode(x_t1)
    }

    pub fn forward_pair(&self, x_t: &Tensor, x_t1: &Tensor) -> (Tensor, Tensor) {
        let pred = self.predict_next_latent(x_t);
        let target = self.target_latent(x_t1);
        (pred, target)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedVisionJepa {
    pub encoder: EmbeddingEncoder,
    pub projector: Linear,
    pub target_projector: Linear,
    pub predictor: Predictor,
}

impl ProjectedVisionJepa {
    pub fn new(
        encoder: EmbeddingEncoder,
        projector: Linear,
        target_projector: Linear,
        predictor: Predictor,
    ) -> Self {
        Self {
            encoder,
            projector,
            target_projector,
            predictor,
        }
    }

    pub fn encode(&self, x: &Tensor) -> Tensor {
        self.encoder.forward(x)
    }

    pub fn project_latent(&self, x_t: &Tensor) -> Tensor {
        let z_t = self.encode(x_t);
        self.projector.forward(&z_t)
    }

    pub fn target_projection(&self, x_t1: &Tensor) -> Tensor {
        let z_t1 = self.encode(x_t1);
        self.target_projector.forward(&z_t1)
    }

    pub fn predict_next_projection(&self, x_t: &Tensor) -> Tensor {
        let projected = self.project_latent(x_t);
        self.predictor.forward(&projected)
    }

    pub fn forward_projection_pair(&self, x_t: &Tensor, x_t1: &Tensor) -> (Tensor, Tensor) {
        let prediction = self.predict_next_projection(x_t);
        let target = self.target_projection(x_t1);
        (prediction, target)
    }

    pub fn step(
        &mut self,
        x_t: &Tensor,
        x_t1: &Tensor,
        regularizer_weight: f32,
        predictor_lr: f32,
        projector_lr: f32,
    ) -> (f32, f32, f32) {
        let z_t = self.encoder.forward(x_t);
        let projection_t = self.projector.forward(&z_t);
        let target = self.target_projector.forward(&self.encoder.forward(x_t1));
        let pred = self.predictor.forward(&projection_t);

        let prediction_loss = mse_loss(&pred, &target);
        let regularizer_loss = gaussian_moment_regularizer(&projection_t);
        let total_loss = prediction_loss + regularizer_weight * regularizer_loss;

        let prediction_grad = mse_loss_grad(&pred, &target);
        let reg_grad = gaussian_moment_regularizer_grad(&projection_t);
        let pred_grads = self.predictor.backward(&projection_t, &prediction_grad);
        let projection_grad =
            combine_projection_grads(&pred_grads.grad_input, &reg_grad, regularizer_weight);
        let projector_grads = self.projector.backward(&z_t, &projection_grad);

        self.predictor.sgd_step(&pred_grads, predictor_lr);
        self.projector.sgd_step(&projector_grads, projector_lr);

        (prediction_loss, regularizer_loss, total_loss)
    }

    pub fn losses(&self, x_t: &Tensor, x_t1: &Tensor, regularizer_weight: f32) -> (f32, f32, f32) {
        let prediction = self.predict_next_projection(x_t);
        let target = self.target_projection(x_t1);
        let prediction_loss = mse_loss(&prediction, &target);
        let regularizer_loss = gaussian_moment_regularizer(&self.project_latent(x_t));
        let total_loss = prediction_loss + regularizer_weight * regularizer_loss;

        (prediction_loss, regularizer_loss, total_loss)
    }
}
