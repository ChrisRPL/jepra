use crate::tensor::Tensor;

pub fn mse_loss(pred: &Tensor, target: &Tensor) -> f32 {
    assert!(
        pred.shape == target.shape,
        "mse_loss shape mismatch: pred {:?}, target {:?}",
        pred.shape,
        target.shape
    );

    let sum_sq: f32 = pred
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(a, b)| {
            let diff = a - b;
            diff * diff
        })
        .sum();

    sum_sq / pred.len() as f32
}

pub fn mse_loss_grad(pred: &Tensor, target: &Tensor) -> Tensor {
    assert!(
        pred.shape == target.shape,
        "mse_loss_grad shape mismatch: pred {:?}, target {:?}",
        pred.shape,
        target.shape
    );

    let scale = 2.0 / pred.len() as f32;

    let data = pred
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(p, t)| scale * (p - t))
        .collect();

    Tensor {
        data,
        shape: pred.shape.clone(),
    }
}
