use crate::encoder::EmbeddingEncoder;
use crate::predictor::Predictor;
use crate::tensor::Tensor;

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