use crate::conv::{Conv2d, Conv2dGrads};
use crate::tensor::Tensor;

#[derive(Debug, Clone, PartialEq)]
pub struct ConvEncoder {
    pub conv1: Conv2d,
    pub conv2: Conv2d,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConvEncoderGrads {
    pub grad_conv1: Conv2dGrads,
    pub grad_conv2: Conv2dGrads,
}

impl ConvEncoder {
    pub fn new(conv1: Conv2d, conv2: Conv2d) -> Self {
        Self { conv1, conv2 }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h1 = self.conv1.forward(x).relu();
        self.conv2.forward(&h1).relu()
    }

    pub fn sgd_step(&mut self, grads: &ConvEncoderGrads, lr: f32) {
        self.conv1.sgd_step(&grads.grad_conv1, lr);
        self.conv2.sgd_step(&grads.grad_conv2, lr);
    }

    pub fn backward(&self, x: &Tensor, grad_latent: &Tensor) -> ConvEncoderGrads {
        let h1 = self.conv1.forward(x);
        let h1_relu = h1.relu();
        let h2 = self.conv2.forward(&h1_relu);
        let h2_relu = h2.relu();

        assert!(
            grad_latent.shape == vec![x.shape[0], h2_relu.shape[1]],
            "ConvEncoder backward grad shape mismatch: got {:?}, expected [{}, {}]",
            grad_latent.shape,
            x.shape[0],
            h2_relu.shape[1]
        );

        let grad_h2 = grad_latent.global_avg_pool2d_backward(h2.shape[2], h2.shape[3]);
        let grad_h2_relu = h2.relu_backward(&grad_h2);
        let grad_conv2 = self.conv2.backward(&h1_relu, &grad_h2_relu);
        let grad_h1_pre = h1_relu.relu_backward(&grad_conv2.grad_input);
        let grad_conv1 = self.conv1.backward(x, &grad_h1_pre);

        ConvEncoderGrads {
            grad_conv1,
            grad_conv2,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingEncoder {
    pub backbone: ConvEncoder,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingEncoderGrads {
    pub grad_backbone: ConvEncoderGrads,
}

impl EmbeddingEncoder {
    pub fn new(backbone: ConvEncoder) -> Self {
        Self { backbone }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.backbone.forward(x).global_avg_pool2d()
    }

    pub fn sgd_step(&mut self, grads: &EmbeddingEncoderGrads, lr: f32) {
        self.backbone.sgd_step(&grads.grad_backbone, lr);
    }

    pub fn backward(&self, x: &Tensor, grad_latent: &Tensor) -> EmbeddingEncoderGrads {
        let grad_backbone = self.backbone.backward(x, grad_latent);
        EmbeddingEncoderGrads { grad_backbone }
    }
}
