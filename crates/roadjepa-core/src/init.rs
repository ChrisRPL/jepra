use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

use crate::tensor::Tensor;

pub fn randn(shape: Vec<usize>, mean: f32, std: f32, seed: u64) -> Tensor {
    assert!(std > 0.0, "randn std must be > 0, got {}", std);

    let len: usize = shape.iter().product();
    let normal = Normal::new(mean as f64, std as f64).unwrap();
    let mut rng = StdRng::seed_from_u64(seed);

    let data = (0..len)
        .map(|_| normal.sample(&mut rng) as f32)
        .collect();

    Tensor::new(data, shape)
}

pub fn zeros(shape: Vec<usize>) -> Tensor {
    Tensor::zeros(shape)
}