use crate::linear::{Linear, LinearGrads};
use crate::tensor::Tensor;

#[derive(Debug, Clone, PartialEq)]
pub struct Predictor {
    pub fc1: Linear,
    pub fc2: Linear,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PredictorGrads {
    pub grad_input: Tensor,
    pub grad_fc1: LinearGrads,
    pub grad_fc2: LinearGrads,
}

pub trait PredictorModule {
    type Grads;

    fn forward(&self, x: &Tensor) -> Tensor;
    fn backward(&self, x: &Tensor, grad_out: &Tensor) -> Self::Grads;
    fn grad_input(grads: &Self::Grads) -> &Tensor;
    fn sgd_step(&mut self, grads: &Self::Grads, lr: f32);
}

impl Predictor {
    pub fn new(fc1: Linear, fc2: Linear) -> Self {
        assert!(
            fc1.weight.shape[1] == fc2.weight.shape[0],
            "Predictor layer mismatch: fc1 output {} != fc2 input {}",
            fc1.weight.shape[1],
            fc2.weight.shape[0]
        );

        Self { fc1, fc2 }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.fc1.forward(x);
        let h = h.relu();
        self.fc2.forward(&h)
    }

    pub fn backward(&self, x: &Tensor, grad_out: &Tensor) -> PredictorGrads {
        let h_pre = self.fc1.forward(x);
        let h = h_pre.relu();

        let grad_fc2 = self.fc2.backward(&h, grad_out);
        let grad_h_pre = h_pre.relu_backward(&grad_fc2.grad_input);
        let grad_fc1 = self.fc1.backward(x, &grad_h_pre);

        PredictorGrads {
            grad_input: grad_fc1.grad_input.clone(),
            grad_fc1,
            grad_fc2,
        }
    }

    pub fn sgd_step(&mut self, grads: &PredictorGrads, lr: f32) {
        self.fc1.sgd_step(&grads.grad_fc1, lr);
        self.fc2.sgd_step(&grads.grad_fc2, lr);
    }
}

impl PredictorModule for Predictor {
    type Grads = PredictorGrads;

    fn forward(&self, x: &Tensor) -> Tensor {
        Predictor::forward(self, x)
    }

    fn backward(&self, x: &Tensor, grad_out: &Tensor) -> Self::Grads {
        Predictor::backward(self, x, grad_out)
    }

    fn grad_input(grads: &Self::Grads) -> &Tensor {
        &grads.grad_input
    }

    fn sgd_step(&mut self, grads: &Self::Grads, lr: f32) {
        Predictor::sgd_step(self, grads, lr);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BottleneckPredictor {
    pub fc1: Linear,
    pub fc2: Linear,
    pub fc3: Linear,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BottleneckPredictorGrads {
    pub grad_input: Tensor,
    pub grad_fc1: LinearGrads,
    pub grad_fc2: LinearGrads,
    pub grad_fc3: LinearGrads,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResidualBottleneckPredictor {
    pub delta: BottleneckPredictor,
    pub residual_scale: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResidualBottleneckPredictorGrads {
    pub grad_input: Tensor,
    pub grad_delta: BottleneckPredictorGrads,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StateRadiusPredictor {
    pub direction: Predictor,
    pub radius_fc1: Linear,
    pub radius_fc2: Linear,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StateRadiusPredictorGrads {
    pub grad_input: Tensor,
    pub grad_direction: PredictorGrads,
    pub grad_radius_fc1: LinearGrads,
    pub grad_radius_fc2: LinearGrads,
}

impl BottleneckPredictor {
    pub fn new(fc1: Linear, fc2: Linear, fc3: Linear) -> Self {
        assert!(
            fc1.weight.shape[1] == fc2.weight.shape[0],
            "BottleneckPredictor layer mismatch: fc1 output {} != fc2 input {}",
            fc1.weight.shape[1],
            fc2.weight.shape[0]
        );
        assert!(
            fc2.weight.shape[1] == fc3.weight.shape[0],
            "BottleneckPredictor layer mismatch: fc2 output {} != fc3 input {}",
            fc2.weight.shape[1],
            fc3.weight.shape[0]
        );

        Self { fc1, fc2, fc3 }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h1 = self.fc1.forward(x).relu();
        let h2 = self.fc2.forward(&h1).relu();
        self.fc3.forward(&h2)
    }

    pub fn backward(&self, x: &Tensor, grad_out: &Tensor) -> BottleneckPredictorGrads {
        let h1_pre = self.fc1.forward(x);
        let h1 = h1_pre.relu();
        let h2_pre = self.fc2.forward(&h1);
        let h2 = h2_pre.relu();

        let grad_fc3 = self.fc3.backward(&h2, grad_out);
        let grad_h2_pre = h2_pre.relu_backward(&grad_fc3.grad_input);
        let grad_fc2 = self.fc2.backward(&h1, &grad_h2_pre);
        let grad_h1_pre = h1_pre.relu_backward(&grad_fc2.grad_input);
        let grad_fc1 = self.fc1.backward(x, &grad_h1_pre);

        BottleneckPredictorGrads {
            grad_input: grad_fc1.grad_input.clone(),
            grad_fc1,
            grad_fc2,
            grad_fc3,
        }
    }

    pub fn sgd_step(&mut self, grads: &BottleneckPredictorGrads, lr: f32) {
        self.fc1.sgd_step(&grads.grad_fc1, lr);
        self.fc2.sgd_step(&grads.grad_fc2, lr);
        self.fc3.sgd_step(&grads.grad_fc3, lr);
    }
}

impl PredictorModule for BottleneckPredictor {
    type Grads = BottleneckPredictorGrads;

    fn forward(&self, x: &Tensor) -> Tensor {
        BottleneckPredictor::forward(self, x)
    }

    fn backward(&self, x: &Tensor, grad_out: &Tensor) -> Self::Grads {
        BottleneckPredictor::backward(self, x, grad_out)
    }

    fn grad_input(grads: &Self::Grads) -> &Tensor {
        &grads.grad_input
    }

    fn sgd_step(&mut self, grads: &Self::Grads, lr: f32) {
        BottleneckPredictor::sgd_step(self, grads, lr);
    }
}

impl ResidualBottleneckPredictor {
    pub fn new(delta: BottleneckPredictor) -> Self {
        Self::new_scaled(delta, 1.0)
    }

    pub fn new_scaled(delta: BottleneckPredictor, residual_scale: f32) -> Self {
        let input_dim = delta.fc1.weight.shape[0];
        let output_dim = delta.fc3.weight.shape[1];
        assert!(
            input_dim == output_dim,
            "ResidualBottleneckPredictor requires matching input/output dims, got {} -> {}",
            input_dim,
            output_dim
        );
        assert!(
            residual_scale.is_finite() && residual_scale >= 0.0,
            "ResidualBottleneckPredictor residual scale must be finite and non-negative, got {}",
            residual_scale
        );

        Self {
            delta,
            residual_scale,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.add(&scaled_tensor(&self.delta.forward(x), self.residual_scale))
    }

    pub fn backward(&self, x: &Tensor, grad_out: &Tensor) -> ResidualBottleneckPredictorGrads {
        let scaled_grad_out = scaled_tensor(grad_out, self.residual_scale);
        let grad_delta = self.delta.backward(x, &scaled_grad_out);
        let grad_input = grad_out.add(&grad_delta.grad_input);

        ResidualBottleneckPredictorGrads {
            grad_input,
            grad_delta,
        }
    }

    pub fn sgd_step(&mut self, grads: &ResidualBottleneckPredictorGrads, lr: f32) {
        self.delta.sgd_step(&grads.grad_delta, lr);
    }
}

impl StateRadiusPredictor {
    pub fn new(direction: Predictor, radius_fc1: Linear, radius_fc2: Linear) -> Self {
        let input_dim = direction.fc1.weight.shape[0];
        let output_dim = direction.fc2.weight.shape[1];
        assert!(
            input_dim == output_dim,
            "StateRadiusPredictor requires matching input/output dims, got {} -> {}",
            input_dim,
            output_dim
        );
        assert!(
            radius_fc1.weight.shape[0] == input_dim,
            "StateRadiusPredictor radius input dim mismatch: got {}, expected {}",
            radius_fc1.weight.shape[0],
            input_dim
        );
        assert!(
            radius_fc1.weight.shape[1] == radius_fc2.weight.shape[0],
            "StateRadiusPredictor radius layer mismatch: fc1 output {} != fc2 input {}",
            radius_fc1.weight.shape[1],
            radius_fc2.weight.shape[0]
        );
        assert!(
            radius_fc2.weight.shape[1] == 1,
            "StateRadiusPredictor radius head must emit one gain per sample, got {}",
            radius_fc2.weight.shape[1]
        );

        Self {
            direction,
            radius_fc1,
            radius_fc2,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let direction = self.direction.forward(x);
        let log_radius = self.log_radius(x);
        let batch_size = x.shape[0];
        let features = x.shape[1];
        let mut output = Tensor::zeros(x.shape.clone());

        for sample in 0..batch_size {
            let scale = positive_radius_scale(log_radius.get(&[sample, 0]));
            for feature_idx in 0..features {
                let input_value = x.get(&[sample, feature_idx]);
                let direction_value = direction.get(&[sample, feature_idx]);
                output.set(
                    &[sample, feature_idx],
                    input_value + scale * (direction_value - input_value),
                );
            }
        }

        output
    }

    pub fn backward(&self, x: &Tensor, grad_out: &Tensor) -> StateRadiusPredictorGrads {
        assert!(
            x.ndim() == 2,
            "StateRadiusPredictor backward input must be 2D, got shape {:?}",
            x.shape
        );
        assert!(
            grad_out.shape == x.shape,
            "StateRadiusPredictor backward grad shape mismatch: got {:?}, expected {:?}",
            grad_out.shape,
            x.shape
        );

        let direction = self.direction.forward(x);
        let radius_pre = self.radius_fc1.forward(x);
        let radius_hidden = radius_pre.relu();
        let log_radius = self.radius_fc2.forward(&radius_hidden);
        let batch_size = x.shape[0];
        let features = x.shape[1];
        let mut grad_direction_out = Tensor::zeros(x.shape.clone());
        let mut grad_input_direct = Tensor::zeros(x.shape.clone());
        let mut grad_log_radius = Tensor::zeros(vec![batch_size, 1]);

        for sample in 0..batch_size {
            let raw_log_radius = log_radius.get(&[sample, 0]);
            let scale = positive_radius_scale(raw_log_radius);
            let scale_grad = positive_radius_scale_grad(raw_log_radius);
            let mut grad_scale = 0.0f32;

            for feature_idx in 0..features {
                let grad_value = grad_out.get(&[sample, feature_idx]);
                let input_value = x.get(&[sample, feature_idx]);
                let direction_value = direction.get(&[sample, feature_idx]);
                let delta = direction_value - input_value;

                grad_direction_out.set(&[sample, feature_idx], grad_value * scale);
                grad_input_direct.set(&[sample, feature_idx], grad_value * (1.0 - scale));
                grad_scale += grad_value * delta;
            }

            grad_log_radius.set(&[sample, 0], grad_scale * scale_grad);
        }

        let grad_direction = self.direction.backward(x, &grad_direction_out);
        let grad_radius_fc2 = self.radius_fc2.backward(&radius_hidden, &grad_log_radius);
        let grad_radius_hidden_pre = radius_pre.relu_backward(&grad_radius_fc2.grad_input);
        let grad_radius_fc1 = self.radius_fc1.backward(x, &grad_radius_hidden_pre);
        let grad_input = grad_input_direct
            .add(&grad_direction.grad_input)
            .add(&grad_radius_fc1.grad_input);

        StateRadiusPredictorGrads {
            grad_input,
            grad_direction,
            grad_radius_fc1,
            grad_radius_fc2,
        }
    }

    pub fn sgd_step(&mut self, grads: &StateRadiusPredictorGrads, lr: f32) {
        self.direction.sgd_step(&grads.grad_direction, lr);
        self.radius_fc1.sgd_step(&grads.grad_radius_fc1, lr);
        self.radius_fc2.sgd_step(&grads.grad_radius_fc2, lr);
    }

    fn log_radius(&self, x: &Tensor) -> Tensor {
        self.radius_fc2.forward(&self.radius_fc1.forward(x).relu())
    }
}

fn scaled_tensor(tensor: &Tensor, scale: f32) -> Tensor {
    Tensor::new(
        tensor.data.iter().map(|value| value * scale).collect(),
        tensor.shape.clone(),
    )
}

fn positive_radius_scale(raw_log_radius: f32) -> f32 {
    raw_log_radius.clamp(-4.0, 4.0).exp()
}

fn positive_radius_scale_grad(raw_log_radius: f32) -> f32 {
    if (-4.0..=4.0).contains(&raw_log_radius) {
        positive_radius_scale(raw_log_radius)
    } else {
        0.0
    }
}

impl PredictorModule for ResidualBottleneckPredictor {
    type Grads = ResidualBottleneckPredictorGrads;

    fn forward(&self, x: &Tensor) -> Tensor {
        ResidualBottleneckPredictor::forward(self, x)
    }

    fn backward(&self, x: &Tensor, grad_out: &Tensor) -> Self::Grads {
        ResidualBottleneckPredictor::backward(self, x, grad_out)
    }

    fn grad_input(grads: &Self::Grads) -> &Tensor {
        &grads.grad_input
    }

    fn sgd_step(&mut self, grads: &Self::Grads, lr: f32) {
        ResidualBottleneckPredictor::sgd_step(self, grads, lr);
    }
}

impl PredictorModule for StateRadiusPredictor {
    type Grads = StateRadiusPredictorGrads;

    fn forward(&self, x: &Tensor) -> Tensor {
        StateRadiusPredictor::forward(self, x)
    }

    fn backward(&self, x: &Tensor, grad_out: &Tensor) -> Self::Grads {
        StateRadiusPredictor::backward(self, x, grad_out)
    }

    fn grad_input(grads: &Self::Grads) -> &Tensor {
        &grads.grad_input
    }

    fn sgd_step(&mut self, grads: &Self::Grads, lr: f32) {
        StateRadiusPredictor::sgd_step(self, grads, lr);
    }
}
