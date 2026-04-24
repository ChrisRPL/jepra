use crate::encoder::EmbeddingEncoder;
use crate::linear::Linear;
use crate::losses::{mse_loss, mse_loss_grad};
use crate::predictor::{Predictor, PredictorModule};
use crate::regularizers::{
    add_projector_drift_regularizer_grad, combine_projection_grads, gaussian_moment_regularizer,
    gaussian_moment_regularizer_grad, projector_drift_regularizer,
};
use crate::tensor::Tensor;

#[derive(Debug, Clone, PartialEq)]
pub struct VisionJepa<P = Predictor> {
    pub encoder: EmbeddingEncoder,
    pub predictor: P,
}

impl<P> VisionJepa<P>
where
    P: PredictorModule,
{
    pub fn new(encoder: EmbeddingEncoder, predictor: P) -> Self {
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

    pub fn step(&mut self, x_t: &Tensor, x_t1: &Tensor, lr: f32) -> (f32, f32) {
        self.step_with_trainable_encoder(x_t, x_t1, lr, 0.0)
    }

    pub fn step_with_trainable_encoder(
        &mut self,
        x_t: &Tensor,
        x_t1: &Tensor,
        predictor_lr: f32,
        encoder_lr: f32,
    ) -> (f32, f32) {
        let z_t = self.encode(x_t);
        let pred = self.predictor.forward(&z_t);
        let target = self.target_latent(x_t1);

        let prediction_loss = mse_loss(&pred, &target);
        let total_loss = prediction_loss;

        let grad_out = mse_loss_grad(&pred, &target);
        let grads = self.predictor.backward(&z_t, &grad_out);
        let encoder_grads = self.encoder.backward(x_t, P::grad_input(&grads));
        self.predictor.sgd_step(&grads, predictor_lr);
        self.encoder.sgd_step(&encoder_grads, encoder_lr);

        (prediction_loss, total_loss)
    }

    pub fn losses(&self, x_t: &Tensor, x_t1: &Tensor) -> (f32, f32) {
        let pred = self.predict_next_latent(x_t);
        let target = self.target_latent(x_t1);
        let loss = mse_loss(&pred, &target);

        (loss, loss)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedVisionJepa<P = Predictor> {
    pub encoder: EmbeddingEncoder,
    pub projector: Linear,
    pub target_projector: Linear,
    pub predictor: P,
    pub target_projection_momentum: f32,
}

impl<P> ProjectedVisionJepa<P>
where
    P: PredictorModule,
{
    pub fn new(
        encoder: EmbeddingEncoder,
        projector: Linear,
        target_projector: Linear,
        predictor: P,
    ) -> Self {
        Self {
            encoder,
            projector,
            target_projector,
            predictor,
            target_projection_momentum: 1.0,
        }
    }

    pub fn encode(&self, x: &Tensor) -> Tensor {
        self.encoder.forward(x)
    }

    pub fn with_target_projection_momentum(mut self, momentum: f32) -> Self {
        Self::assert_target_projection_momentum_in_range(momentum);
        self.target_projection_momentum = momentum;
        self
    }

    pub fn set_target_projection_momentum(&mut self, momentum: f32) {
        Self::assert_target_projection_momentum_in_range(momentum);
        self.target_projection_momentum = momentum;
    }

    pub fn target_projection_momentum(&self) -> f32 {
        self.target_projection_momentum
    }

    pub fn project_latent(&self, x_t: &Tensor) -> Tensor {
        let z_t = self.encode(x_t);
        self.projector.forward(&z_t)
    }

    pub fn target_projection(&self, x_t1: &Tensor) -> Tensor {
        let z_t1 = self.encode(x_t1);
        self.target_projector.forward(&z_t1)
    }

    pub fn target_projection_drift(&self) -> f32 {
        let mut drift = 0.0f32;
        let mut parameter_count = 0usize;

        for (target_weight, online_weight) in self
            .target_projector
            .weight
            .data
            .iter()
            .zip(self.projector.weight.data.iter())
        {
            drift += (target_weight - online_weight).abs();
            parameter_count += 1;
        }

        for (target_bias, online_bias) in self
            .target_projector
            .bias
            .data
            .iter()
            .zip(self.projector.bias.data.iter())
        {
            drift += (target_bias - online_bias).abs();
            parameter_count += 1;
        }

        if parameter_count == 0 {
            0.0
        } else {
            drift / parameter_count as f32
        }
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
        self.step_with_trainable_encoder(
            x_t,
            x_t1,
            regularizer_weight,
            predictor_lr,
            projector_lr,
            0.0,
        )
    }

    pub fn step_with_trainable_encoder(
        &mut self,
        x_t: &Tensor,
        x_t1: &Tensor,
        regularizer_weight: f32,
        predictor_lr: f32,
        projector_lr: f32,
        encoder_lr: f32,
    ) -> (f32, f32, f32) {
        let (prediction_loss, regularizer_loss, _, total_loss) = self
            .step_with_trainable_encoder_and_projector_drift_regularizer(
                x_t,
                x_t1,
                regularizer_weight,
                0.0,
                predictor_lr,
                projector_lr,
                encoder_lr,
            );
        (prediction_loss, regularizer_loss, total_loss)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_with_projector_drift_regularizer(
        &mut self,
        x_t: &Tensor,
        x_t1: &Tensor,
        regularizer_weight: f32,
        projector_drift_weight: f32,
        predictor_lr: f32,
        projector_lr: f32,
    ) -> (f32, f32, f32, f32) {
        self.step_with_trainable_encoder_and_projector_drift_regularizer(
            x_t,
            x_t1,
            regularizer_weight,
            projector_drift_weight,
            predictor_lr,
            projector_lr,
            0.0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_with_trainable_encoder_and_projector_drift_regularizer(
        &mut self,
        x_t: &Tensor,
        x_t1: &Tensor,
        regularizer_weight: f32,
        projector_drift_weight: f32,
        predictor_lr: f32,
        projector_lr: f32,
        encoder_lr: f32,
    ) -> (f32, f32, f32, f32) {
        self.step_with_extra_prediction_grad(
            x_t,
            x_t1,
            regularizer_weight,
            projector_drift_weight,
            0.0,
            None,
            predictor_lr,
            projector_lr,
            encoder_lr,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_with_extra_prediction_grad(
        &mut self,
        x_t: &Tensor,
        x_t1: &Tensor,
        regularizer_weight: f32,
        projector_drift_weight: f32,
        extra_prediction_loss: f32,
        extra_prediction_grad: Option<&Tensor>,
        predictor_lr: f32,
        projector_lr: f32,
        encoder_lr: f32,
    ) -> (f32, f32, f32, f32) {
        assert_projector_drift_weight(projector_drift_weight);
        assert!(
            extra_prediction_loss.is_finite() && extra_prediction_loss >= 0.0,
            "extra prediction loss must be finite and non-negative, got {}",
            extra_prediction_loss
        );

        let z_t = self.encoder.forward(x_t);
        let projection_t = self.projector.forward(&z_t);
        let target = self.target_projector.forward(&self.encoder.forward(x_t1));
        let pred = self.predictor.forward(&projection_t);

        let prediction_loss = mse_loss(&pred, &target);
        let regularizer_loss = gaussian_moment_regularizer(&projection_t);
        let projector_drift_loss =
            projector_drift_regularizer(&self.projector, &self.target_projector);
        let total_loss = prediction_loss
            + regularizer_weight * regularizer_loss
            + projector_drift_weight * projector_drift_loss
            + extra_prediction_loss;

        let mut prediction_grad = mse_loss_grad(&pred, &target);
        if let Some(extra_prediction_grad) = extra_prediction_grad {
            assert_eq!(
                extra_prediction_grad.shape, prediction_grad.shape,
                "extra prediction grad shape mismatch: got {:?}, expected {:?}",
                extra_prediction_grad.shape, prediction_grad.shape
            );
            prediction_grad.add_inplace(extra_prediction_grad);
        }
        let reg_grad = gaussian_moment_regularizer_grad(&projection_t);
        let pred_grads = self.predictor.backward(&projection_t, &prediction_grad);
        let projection_grad =
            combine_projection_grads(P::grad_input(&pred_grads), &reg_grad, regularizer_weight);
        let mut projector_grads = self.projector.backward(&z_t, &projection_grad);
        add_projector_drift_regularizer_grad(
            &mut projector_grads,
            &self.projector,
            &self.target_projector,
            projector_drift_weight,
        );
        let encoder_grads = self.encoder.backward(x_t, &projector_grads.grad_input);

        self.predictor.sgd_step(&pred_grads, predictor_lr);
        self.projector.sgd_step(&projector_grads, projector_lr);
        self.encoder.sgd_step(&encoder_grads, encoder_lr);
        self.update_target_projector();

        (
            prediction_loss,
            regularizer_loss,
            projector_drift_loss,
            total_loss,
        )
    }

    fn assert_target_projection_momentum_in_range(momentum: f32) {
        assert!(
            (0.0..=1.0).contains(&momentum),
            "target projection momentum must be in [0.0, 1.0], got {}",
            momentum
        );
    }

    fn update_target_projector(&mut self) {
        let momentum = self.target_projection_momentum;
        Self::assert_target_projection_momentum_in_range(momentum);

        if momentum == 1.0 {
            return;
        }

        let one_minus_momentum = 1.0 - momentum;
        for (target_weight, online_weight) in self
            .target_projector
            .weight
            .data
            .iter_mut()
            .zip(self.projector.weight.data.iter())
        {
            *target_weight = momentum * *target_weight + one_minus_momentum * online_weight;
        }

        for (target_bias, online_bias) in self
            .target_projector
            .bias
            .data
            .iter_mut()
            .zip(self.projector.bias.data.iter())
        {
            *target_bias = momentum * *target_bias + one_minus_momentum * online_bias;
        }
    }

    pub fn losses(&self, x_t: &Tensor, x_t1: &Tensor, regularizer_weight: f32) -> (f32, f32, f32) {
        let (prediction_loss, regularizer_loss, _, total_loss) =
            self.losses_with_projector_drift_regularizer(x_t, x_t1, regularizer_weight, 0.0);
        (prediction_loss, regularizer_loss, total_loss)
    }

    pub fn losses_with_projector_drift_regularizer(
        &self,
        x_t: &Tensor,
        x_t1: &Tensor,
        regularizer_weight: f32,
        projector_drift_weight: f32,
    ) -> (f32, f32, f32, f32) {
        assert_projector_drift_weight(projector_drift_weight);

        let prediction = self.predict_next_projection(x_t);
        let target = self.target_projection(x_t1);
        let prediction_loss = mse_loss(&prediction, &target);
        let regularizer_loss = gaussian_moment_regularizer(&self.project_latent(x_t));
        let projector_drift_loss =
            projector_drift_regularizer(&self.projector, &self.target_projector);
        let total_loss = prediction_loss
            + regularizer_weight * regularizer_loss
            + projector_drift_weight * projector_drift_loss;

        (
            prediction_loss,
            regularizer_loss,
            projector_drift_loss,
            total_loss,
        )
    }
}

fn assert_projector_drift_weight(projector_drift_weight: f32) {
    assert!(
        projector_drift_weight.is_finite() && projector_drift_weight >= 0.0,
        "projector drift weight must be finite and non-negative, got {}",
        projector_drift_weight
    );
}
