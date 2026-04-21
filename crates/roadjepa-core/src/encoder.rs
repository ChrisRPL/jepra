use crate::conv::Conv2d;
use crate::tensor::Tensor;

#[derive(Debug, Clone, PartialEq)]
pub struct ConvEncoder {
    pub conv1: Conv2d,
    pub conv2: Conv2d,
}

impl ConvEncoder {
    pub fn new(conv1: Conv2d, conv2: Conv2d) -> Self {
        Self { conv1, conv2 }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h1 = self.conv1.forward(x).relu();
        self.conv2.forward(&h1).relu()
    }
}