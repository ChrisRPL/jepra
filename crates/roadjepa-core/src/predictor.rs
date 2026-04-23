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
