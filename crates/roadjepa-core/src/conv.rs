use crate::tensor::Tensor;

#[derive(Debug, Clone, PartialEq)]
pub struct Conv2d {
    pub weight: Tensor, // [out_channels, in_channels, kernel_h, kernel_w]
    pub bias: Tensor,   // [out_channels]
    pub stride: usize,
    pub padding: usize,
}

impl Conv2d {
    pub fn new(weight: Tensor, bias: Tensor, stride: usize, padding: usize) -> Self {
        assert!(
            weight.ndim() == 4,
            "Conv2d weight must be 4D, got shape {:?}",
            weight.shape
        );
        assert!(
            bias.ndim() == 1,
            "Conv2d bias must be 1D, got shape {:?}",
            bias.shape
        );
        assert!(
            bias.shape[0] == weight.shape[0],
            "Conv2d bias shape mismatch: bias {:?}, expected [{}]",
            bias.shape,
            weight.shape[0]
        );
        assert!(stride >= 1, "Conv2d stride must be >= 1");
        Self {
            weight,
            bias,
            stride,
            padding,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        assert!(
            x.ndim() == 4,
            "Conv2d input must be 4D [B, C, H, W], got shape {:?}",
            x.shape
        );

        let batch = x.shape[0];
        let in_channels = x.shape[1];
        let in_h = x.shape[2];
        let in_w = x.shape[3];

        let out_channels = self.weight.shape[0];
        let weight_in_channels = self.weight.shape[1];
        let kernel_h = self.weight.shape[2];
        let kernel_w = self.weight.shape[3];

        assert!(
            in_channels == weight_in_channels,
            "Conv2d input channel mismatch: input {:?}, weight {:?}",
            x.shape,
            self.weight.shape
        );

        let out_h = (in_h + 2 * self.padding - kernel_h) / self.stride + 1;
        let out_w = (in_w + 2 * self.padding - kernel_w) / self.stride + 1;

        let mut out = Tensor::zeros(vec![batch, out_channels, out_h, out_w]);

        for b in 0..batch {
            for oc in 0..out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = self.bias.get(&[oc]);

                        for ic in 0..in_channels {
                            for kh in 0..kernel_h {
                                for kw in 0..kernel_w {
                                    let ih = oh * self.stride + kh;
                                    let iw = ow * self.stride + kw;

                                    let ih_padded = ih as isize - self.padding as isize;
                                    let iw_padded = iw as isize - self.padding as isize;

                                    if ih_padded >= 0
                                        && iw_padded >= 0
                                        && (ih_padded as usize) < in_h
                                        && (iw_padded as usize) < in_w
                                    {
                                        let x_val = x.get(&[
                                            b,
                                            ic,
                                            ih_padded as usize,
                                            iw_padded as usize,
                                        ]);
                                        let w_val = self.weight.get(&[oc, ic, kh, kw]);
                                        sum += x_val * w_val;
                                    }
                                }
                            }
                        }

                        out.set(&[b, oc, oh, ow], sum);
                    }
                }
            }
        }

        out
    }
}