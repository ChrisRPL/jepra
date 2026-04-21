use crate::tensor::Tensor;

#[derive(Debug, Clone, PartialEq)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearGrads {
    pub grad_input: Tensor,
    pub grad_weight: Tensor,
    pub grad_bias: Tensor,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        assert!(
            weight.ndim() == 2,
            "Linear weight must be 2D, got shape {:?}",
            weight.shape
        );

        assert!(
            bias.ndim() == 1,
            "Linear bias must be 1D, got shape {:?}",
            bias.shape
        );

        let out_features = weight.shape[1];
        assert!(
            bias.shape[0] == out_features,
            "Linear bias shape mismatch: bias {:?}, expected [{}]",
            bias.shape,
            out_features
        );

        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        assert!(
            x.ndim() == 2,
            "Linear input must be 2D, got shape {:?}",
            x.shape
        );

        let in_features = self.weight.shape[0];
        let out_features = self.weight.shape[1];

        assert!(
            x.shape[1] == in_features,
            "Linear input feature mismatch: input {:?}, weight {:?}",
            x.shape,
            self.weight.shape
        );

        let mut y = x.matmul(&self.weight);
        let batch_size = x.shape[0];

        for i in 0..batch_size {
            for j in 0..out_features {
                let value = y.get(&[i, j]) + self.bias.get(&[j]);
                y.set(&[i, j], value);
            }
        }

        y
    }

    pub fn backward(&self, x: &Tensor, grad_out: &Tensor) -> LinearGrads {
        assert!(
            x.ndim() == 2,
            "Linear backward input must be 2D, got shape {:?}",
            x.shape
        );

        assert!(
            grad_out.ndim() == 2,
            "Linear backward grad_out must be 2D, got shape {:?}",
            grad_out.shape
        );

        let batch_size = x.shape[0];
        let in_features = x.shape[1];
        let out_features = self.weight.shape[1];

        assert!(
            grad_out.shape == vec![batch_size, out_features],
            "Linear backward grad_out shape mismatch: got {:?}, expected [{}, {}]",
            grad_out.shape,
            batch_size,
            out_features
        );

        assert!(
            self.weight.shape == vec![in_features, out_features],
            "Linear backward weight shape mismatch: weight {:?}, expected [{}, {}]",
            self.weight.shape,
            in_features,
            out_features
        );

        let grad_weight = x.transpose().matmul(grad_out);
        let grad_bias = grad_out.sum_axis0();
        let grad_input = grad_out.matmul(&self.weight.transpose());

        LinearGrads {
            grad_input,
            grad_weight,
            grad_bias,
        }
    }

    pub fn sgd_step(&mut self, grads: &LinearGrads, lr: f32) {
        assert!(
            self.weight.shape == grads.grad_weight.shape,
            "sgd_step weight shape mismatch: weight {:?}, grad {:?}",
            self.weight.shape,
            grads.grad_weight.shape
        );

        assert!(
            self.bias.shape == grads.grad_bias.shape,
            "sgd_step bias shape mismatch: bias {:?}, grad {:?}",
            self.bias.shape,
            grads.grad_bias.shape
        );

        self.weight.sub_scaled_inplace(&grads.grad_weight, lr);
        self.bias.sub_scaled_inplace(&grads.grad_bias, lr);
    }
}