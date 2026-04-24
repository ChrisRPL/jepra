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
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResidualBottleneckPredictorGrads {
    pub grad_input: Tensor,
    pub grad_delta: BottleneckPredictorGrads,
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
        let input_dim = delta.fc1.weight.shape[0];
        let output_dim = delta.fc3.weight.shape[1];
        assert!(
            input_dim == output_dim,
            "ResidualBottleneckPredictor requires matching input/output dims, got {} -> {}",
            input_dim,
            output_dim
        );

        Self { delta }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.add(&self.delta.forward(x))
    }

    pub fn backward(&self, x: &Tensor, grad_out: &Tensor) -> ResidualBottleneckPredictorGrads {
        let grad_delta = self.delta.backward(x, grad_out);
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
