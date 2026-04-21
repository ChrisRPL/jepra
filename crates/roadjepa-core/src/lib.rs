#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected_len: usize = shape.iter().product();

        assert!(
            data.len() == expected_len,
            "data length ({}) does not match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_len
        );

        Self { data, shape }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        Self {
            data: vec![0.0; len],
            shape,
        }
    }

    pub fn zeros_like(&self) -> Tensor {
        Tensor::zeros(self.shape.clone())
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        assert!(
            self.shape == other.shape,
            "shape mismatch in add: left {:?}, right {:?}",
            self.shape,
            other.shape
        );

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn relu(&self) -> Tensor {
        let data = self
            .data
            .iter()
            .map(|x| if *x > 0.0 { *x } else { 0.0 })
            .collect();

        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn offset(&self, indices: &[usize]) -> usize {
        assert!(
            indices.len() == self.ndim(),
            "index rank mismatch: got {}, expected {}",
            indices.len(),
            self.ndim()
        );

        let mut offset = 0;
        let mut stride = 1;

        for (idx, dim) in indices.iter().rev().zip(self.shape.iter().rev()) {
            assert!(
                *idx < *dim,
                "index out of bounds: index {} for dim size {}",
                idx,
                dim
            );
            offset += idx * stride;
            stride *= dim;
        }

        offset
    }

    pub fn get(&self, indices: &[usize]) -> f32 {
        let offset = self.offset(indices);
        self.data[offset]
    }

    pub fn set(&mut self, indices: &[usize], value: f32) {
        let offset = self.offset(indices);
        self.data[offset] = value;
    }

    pub fn add_inplace(&mut self, other: &Tensor) {
        assert!(
            self.shape == other.shape,
            "add_inplace shape mismatch: left {:?}, right {:?}",
            self.shape,
            other.shape
        );

        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += *b;
        }
    }

    pub fn sub_scaled_inplace(&mut self, other: &Tensor, scale: f32) {
        assert!(
            self.shape == other.shape,
            "sub_scaled_inplace shape mismatch: left {:?}, right {:?}",
            self.shape,
            other.shape
        );

        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a -= scale * *b;
        }
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert!(
            self.ndim() == 2 && other.ndim() == 2,
            "matmul currently supports only 2D tensors, got {:?} and {:?}",
            self.shape,
            other.shape
        );

        let m = self.shape[0];
        let k_left = self.shape[1];
        let k_right = other.shape[0];
        let n = other.shape[1];

        assert!(
            k_left == k_right,
            "matmul inner dimension mismatch: left {:?}, right {:?}",
            self.shape,
            other.shape
        );

        let mut out = Tensor::zeros(vec![m, n]);

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..k_left {
                    sum += self.get(&[i, k]) * other.get(&[k, j]);
                }
                out.set(&[i, j], sum);
            }
        }

        out
    }

    pub fn transpose(&self) -> Tensor {
        assert!(
            self.ndim() == 2,
            "transpose currently supports only 2D tensors, got shape {:?}",
            self.shape
        );

        let rows = self.shape[0];
        let cols = self.shape[1];

        let mut out = Tensor::zeros(vec![cols, rows]);

        for i in 0..rows {
            for j in 0..cols {
                out.set(&[j, i], self.get(&[i, j]));
            }
        }

        out
    }

    pub fn sum_axis0(&self) -> Tensor {
        assert!(
            self.ndim() == 2,
            "sum_axis0 currently supports only 2D tensors, got shape {:?}",
            self.shape
        );

        let rows = self.shape[0];
        let cols = self.shape[1];

        let mut out = Tensor::zeros(vec![cols]);

        for j in 0..cols {
            let mut sum = 0.0;
            for i in 0..rows {
                sum += self.get(&[i, j]);
            }
            out.set(&[j], sum);
        }

        out
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Linear {
    pub weight: Tensor, // shape: [in_features, out_features]
    pub bias: Tensor,   // shape: [out_features]
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
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearGrads {
    pub grad_input: Tensor,
    pub grad_weight: Tensor,
    pub grad_bias: Tensor,
}

impl Linear {
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

#[derive(Debug, Clone, PartialEq)]
pub struct Predictor {
    pub fc1: Linear,
    pub fc2: Linear,
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
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_tensor_with_valid_shape() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.len(), 4);
        assert_eq!(t.ndim(), 2);
    }

    #[test]
    fn creates_zero_tensor() {
        let t = Tensor::zeros(vec![2, 3]);

        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.len(), 6);
        assert_eq!(t.data, vec![0.0; 6]);
    }

    #[test]
    #[should_panic]
    fn panics_on_invalid_shape() {
        let _ = Tensor::new(vec![1.0, 2.0, 3.0], vec![2, 2]);
    }

    #[test]
    fn adds_two_tensors() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);

        let c = a.add(&b);

        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    #[should_panic]
    fn panics_on_add_shape_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![1.0, 2.0], vec![2]);

        let _ = a.add(&b);
    }

    #[test]
    fn computes_offset_for_2d_tensor() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        assert_eq!(t.offset(&[0, 0]), 0);
        assert_eq!(t.offset(&[0, 2]), 2);
        assert_eq!(t.offset(&[1, 0]), 3);
        assert_eq!(t.offset(&[1, 2]), 5);
    }

    #[test]
    fn gets_value_from_2d_tensor() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        assert_eq!(t.get(&[0, 0]), 1.0);
        assert_eq!(t.get(&[0, 2]), 3.0);
        assert_eq!(t.get(&[1, 1]), 5.0);
    }

    #[test]
    fn gets_value_from_3d_tensor() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);

        assert_eq!(t.get(&[0, 0, 0]), 1.0);
        assert_eq!(t.get(&[0, 1, 1]), 4.0);
        assert_eq!(t.get(&[1, 0, 0]), 5.0);
        assert_eq!(t.get(&[1, 1, 1]), 8.0);
    }

    #[test]
    fn sets_value_in_2d_tensor() {
        let mut t = Tensor::zeros(vec![2, 2]);

        t.set(&[0, 1], 3.5);
        t.set(&[1, 0], 7.0);

        assert_eq!(t.get(&[0, 1]), 3.5);
        assert_eq!(t.get(&[1, 0]), 7.0);
        assert_eq!(t.data, vec![0.0, 3.5, 7.0, 0.0]);
    }

    #[test]
    fn matmul_works_for_2d_tensors() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);

        let c = a.matmul(&b);

        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    #[should_panic]
    fn panics_on_matmul_dimension_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

        let _ = a.matmul(&b);
    }

    #[test]
    #[should_panic]
    fn panics_on_index_rank_mismatch() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let _ = t.get(&[0]);
    }

    #[test]
    #[should_panic]
    fn panics_on_out_of_bounds_index() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let _ = t.get(&[0, 2]);
    }

    #[test]
    #[should_panic]
    fn panics_on_set_out_of_bounds_index() {
        let mut t = Tensor::zeros(vec![2, 2]);
        t.set(&[2, 0], 1.0);
    }

    #[test]
    fn linear_forward_works() {
        let weight = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

        let bias = Tensor::new(vec![0.5, -1.0], vec![2]);

        let linear = Linear::new(weight, bias);

        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        let y = linear.forward(&x);

        assert_eq!(y.shape, vec![2, 2]);
        assert_eq!(y.data, vec![22.5, 27.0, 49.5, 63.0]);
    }

    #[test]
    #[should_panic]
    fn linear_panics_on_bias_shape_mismatch() {
        let weight = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let bias = Tensor::new(vec![1.0], vec![1]);

        let _ = Linear::new(weight, bias);
    }

    #[test]
    #[should_panic]
    fn linear_panics_on_input_feature_mismatch() {
        let weight = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

        let bias = Tensor::new(vec![0.0, 0.0], vec![2]);
        let linear = Linear::new(weight, bias);

        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let _ = linear.forward(&x);
    }

    #[test]
    fn relu_works() {
        let t = Tensor::new(vec![-2.0, -0.5, 0.0, 1.0, 3.5, -4.0], vec![2, 3]);

        let y = t.relu();

        assert_eq!(y.shape, vec![2, 3]);
        assert_eq!(y.data, vec![0.0, 0.0, 0.0, 1.0, 3.5, 0.0]);
    }

    #[test]
    fn relu_preserves_shape() {
        let t = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![2, 2]);
        let y = t.relu();

        assert_eq!(y.shape, vec![2, 2]);
    }

    #[test]
    fn tiny_predictor_forward_works() {
        let fc1 = Linear::new(
            Tensor::new(
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                vec![3, 4],
            ),
            Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![4]),
        );

        let fc2 = Linear::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], vec![4, 2]),
            Tensor::new(vec![0.5, -0.5], vec![2]),
        );

        let predictor = Predictor::new(fc1, fc2);

        let x = Tensor::new(vec![1.0, -2.0, 3.0, -1.0, 2.0, -3.0], vec![2, 3]);

        let y = predictor.forward(&x);

        assert_eq!(y.shape, vec![2, 2]);
        assert_eq!(y.data, vec![4.5, 2.5, 0.5, 1.5]);
    }

    #[test]
    #[should_panic]
    fn tiny_predictor_panics_on_layer_mismatch() {
        let fc1 = Linear::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![2, 3]),
            Tensor::new(vec![0.0, 0.0, 0.0], vec![3]),
        );

        let fc2 = Linear::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], vec![4, 2]),
            Tensor::new(vec![0.0, 0.0], vec![2]),
        );

        let _ = Predictor::new(fc1, fc2);
    }

    #[test]
    fn mse_loss_works() {
        let pred = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let target = Tensor::new(vec![1.0, 4.0, 2.0], vec![3]);

        let loss = mse_loss(&pred, &target);

        let expected = (0.0_f32.powi(2) + (-2.0_f32).powi(2) + 1.0_f32.powi(2)) / 3.0;
        assert!((loss - expected).abs() < 1e-6);
    }

    #[test]
    fn mse_loss_is_zero_for_identical_tensors() {
        let pred = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let target = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let loss = mse_loss(&pred, &target);

        assert!((loss - 0.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic]
    fn mse_loss_panics_on_shape_mismatch() {
        let pred = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let target = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let _ = mse_loss(&pred, &target);
    }

    #[test]
    fn end_to_end_predictor_and_loss_works() {
        let fc1 = Linear::new(
            Tensor::new(
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                vec![3, 4],
            ),
            Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![4]),
        );

        let fc2 = Linear::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], vec![4, 2]),
            Tensor::new(vec![0.5, -0.5], vec![2]),
        );

        let predictor = Predictor::new(fc1, fc2);

        let x = Tensor::new(vec![1.0, -2.0, 3.0, -1.0, 2.0, -3.0], vec![2, 3]);

        let target = Tensor::new(vec![4.0, 3.0, 1.0, 1.0], vec![2, 2]);

        let pred = predictor.forward(&x);
        let loss = mse_loss(&pred, &target);

        assert_eq!(pred.shape, vec![2, 2]);
        assert_eq!(pred.data, vec![4.5, 2.5, 0.5, 1.5]);

        let expected = ((4.5_f32 - 4.0).powi(2)
            + (2.5_f32 - 3.0).powi(2)
            + (0.5_f32 - 1.0).powi(2)
            + (1.5_f32 - 1.0).powi(2))
            / 4.0;

        assert!((loss - expected).abs() < 1e-6);
    }

    #[test]
    fn add_inplace_works() {
        let mut a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);

        a.add_inplace(&b);

        assert_eq!(a.shape, vec![2, 2]);
        assert_eq!(a.data, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn sub_scaled_inplace_works() {
        let mut param = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let grad = Tensor::new(vec![0.1, 0.2, 0.3], vec![3]);

        param.sub_scaled_inplace(&grad, 0.5);

        assert_eq!(param.data, vec![0.95, 1.9, 2.85]);
    }

    #[test]
    #[should_panic]
    fn add_inplace_panics_on_shape_mismatch() {
        let mut a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![1.0, 2.0], vec![2]);

        a.add_inplace(&b);
    }

    #[test]
    #[should_panic]
    fn sub_scaled_inplace_panics_on_shape_mismatch() {
        let mut param = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let grad = Tensor::new(vec![0.1, 0.2], vec![2]);

        param.sub_scaled_inplace(&grad, 0.1);
    }

    #[test]
    fn zeros_like_works() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let z = t.zeros_like();

        assert_eq!(z.shape, vec![2, 2]);
        assert_eq!(z.data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn transpose_works_for_2d_tensor() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        let y = t.transpose();

        assert_eq!(y.shape, vec![3, 2]);
        assert_eq!(y.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_of_square_tensor_works() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let y = t.transpose();

        assert_eq!(y.shape, vec![2, 2]);
        assert_eq!(y.data, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    #[should_panic]
    fn transpose_panics_for_non_2d_tensor() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2, 1]);
        let _ = t.transpose();
    }

    #[test]
    fn sum_axis0_works_for_2d_tensor() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        let y = t.sum_axis0();

        assert_eq!(y.shape, vec![3]);
        assert_eq!(y.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn sum_axis0_works_for_single_row() {
        let t = Tensor::new(vec![1.5, -2.0, 4.0], vec![1, 3]);

        let y = t.sum_axis0();

        assert_eq!(y.shape, vec![3]);
        assert_eq!(y.data, vec![1.5, -2.0, 4.0]);
    }

    #[test]
    #[should_panic]
    fn sum_axis0_panics_for_non_2d_tensor() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2, 1]);
        let _ = t.sum_axis0();
    }

    #[test]
    fn linear_backward_works() {
        let linear = Linear::new(
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]),
            Tensor::new(vec![0.1, -0.2], vec![2]),
        );

        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        let grad_out = Tensor::new(vec![0.5, 1.0, 1.5, 2.0], vec![2, 2]);

        let grads = linear.backward(&x, &grad_out);

        assert_eq!(grads.grad_weight.shape, vec![3, 2]);
        assert_eq!(
            grads.grad_weight.data,
            vec![6.5, 9.0, 8.5, 12.0, 10.5, 15.0]
        );

        assert_eq!(grads.grad_bias.shape, vec![2]);
        assert_eq!(grads.grad_bias.data, vec![2.0, 3.0]);

        assert_eq!(grads.grad_input.shape, vec![2, 3]);
        assert_eq!(grads.grad_input.data, vec![2.5, 5.5, 8.5, 5.5, 12.5, 19.5]);
    }

    #[test]
    #[should_panic]
    fn linear_backward_panics_on_grad_out_shape_mismatch() {
        let linear = Linear::new(
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]),
            Tensor::new(vec![0.0, 0.0], vec![2]),
        );

        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        let grad_out = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]);

        let _ = linear.backward(&x, &grad_out);
    }

    #[test]
    fn linear_sgd_step_updates_parameters() {
        let mut linear = Linear::new(
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]),
            Tensor::new(vec![0.1, -0.2], vec![2]),
        );

        let grads = LinearGrads {
            grad_input: Tensor::zeros(vec![2, 3]),
            grad_weight: Tensor::new(vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0], vec![3, 2]),
            grad_bias: Tensor::new(vec![0.25, 0.75], vec![2]),
        };

        linear.sgd_step(&grads, 0.1);

        assert_eq!(linear.weight.data, vec![0.95, 1.9, 2.85, 3.8, 4.75, 5.7,]);

        assert_eq!(linear.bias.data, vec![0.075, -0.275]);
    }

    #[test]
    #[should_panic]
    fn linear_sgd_step_panics_on_weight_shape_mismatch() {
        let mut linear = Linear::new(
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]),
            Tensor::new(vec![0.0, 0.0], vec![2]),
        );

        let grads = LinearGrads {
            grad_input: Tensor::zeros(vec![2, 3]),
            grad_weight: Tensor::zeros(vec![2, 2]),
            grad_bias: Tensor::zeros(vec![2]),
        };

        linear.sgd_step(&grads, 0.1);
    }

    #[test]
    fn mse_loss_grad_works() {
        let pred = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let target = Tensor::new(vec![1.0, 4.0, 2.0], vec![3]);

        let grad = mse_loss_grad(&pred, &target);

        assert_eq!(grad.shape, vec![3]);
        assert!((grad.data[0] - 0.0).abs() < 1e-6);
        assert!((grad.data[1] - (-1.3333334)).abs() < 1e-6);
        assert!((grad.data[2] - 0.6666667).abs() < 1e-6);
    }

    #[test]
    fn linear_training_step_reduces_mse_loss() {
        let mut linear = Linear::new(
            Tensor::new(vec![0.0, 0.0], vec![2, 1]),
            Tensor::new(vec![0.0], vec![1]),
        );

        let x = Tensor::new(vec![1.0, 2.0], vec![1, 2]);
        let target = Tensor::new(vec![3.0], vec![1, 1]);

        let pred_before = linear.forward(&x);
        let loss_before = mse_loss(&pred_before, &target);

        let grad_out = mse_loss_grad(&pred_before, &target);
        let grads = linear.backward(&x, &grad_out);
        linear.sgd_step(&grads, 0.1);

        let pred_after = linear.forward(&x);
        let loss_after = mse_loss(&pred_after, &target);

        assert!(loss_after < loss_before);

        assert!((linear.weight.data[0] - 0.6).abs() < 1e-6);
        assert!((linear.weight.data[1] - 1.2).abs() < 1e-6);
        assert!((linear.bias.data[0] - 0.6).abs() < 1e-6);

        assert!((loss_before - 9.0).abs() < 1e-6);
        assert!((loss_after - 0.36).abs() < 1e-6);
    }

    #[test]
    #[should_panic]
    fn mse_loss_grad_panics_on_shape_mismatch() {
        let pred = Tensor::new(vec![1.0, 2.0], vec![2]);
        let target = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);

        let _ = mse_loss_grad(&pred, &target);
    }
}
