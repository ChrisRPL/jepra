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

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
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
}