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

    pub fn relu_backward(&self, grad_out: &Tensor) -> Tensor {
        assert!(
            self.shape == grad_out.shape,
            "relu_backward shape mismatch: input {:?}, grad_out {:?}",
            self.shape,
            grad_out.shape
        );

        let data = self
            .data
            .iter()
            .zip(grad_out.data.iter())
            .map(|(x, g)| if *x > 0.0 { *g } else { 0.0 })
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

    pub fn global_avg_pool2d(&self) -> Tensor {
        assert!(
            self.ndim() == 4,
            "global_avg_pool2d expects 4D tensor [B, C, H, W], got shape {:?}",
            self.shape
        );

        let batch = self.shape[0];
        let channels = self.shape[1];
        let height = self.shape[2];
        let width = self.shape[3];

        let mut out = Tensor::zeros(vec![batch, channels]);

        let area = (height * width) as f32;

        for b in 0..batch {
            for c in 0..channels {
                let mut sum = 0.0;
                for h in 0..height {
                    for w in 0..width {
                        sum += self.get(&[b, c, h, w]);
                    }
                }
                out.set(&[b, c], sum / area);
            }
        }

        out
    }

    pub fn global_avg_pool2d_backward(&self, out_h: usize, out_w: usize) -> Tensor {
        assert!(
            self.ndim() == 2,
            "global_avg_pool2d_backward expects 2D grad tensor [B, C], got shape {:?}",
            self.shape
        );

        let batch = self.shape[0];
        let channels = self.shape[1];
        let out_area = out_h * out_w;

        assert!(
            out_area > 0,
            "global_avg_pool2d_backward expects non-empty output spatial size, got {}x{}",
            out_h,
            out_w
        );

        let mut out = Tensor::zeros(vec![batch, channels, out_h, out_w]);
        let scale = 1.0 / out_area as f32;

        for b in 0..batch {
            for c in 0..channels {
                let grad = self.get(&[b, c]) * scale;

                for h in 0..out_h {
                    for w in 0..out_w {
                        out.set(&[b, c, h, w], grad);
                    }
                }
            }
        }

        out
    }
}
