use crate::encoder::EmbeddingEncoder;
use crate::linear::Linear;
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
}
