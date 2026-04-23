use crate::tensor::Tensor;

#[derive(Debug, Clone, PartialEq)]
pub struct Conv2d {
    pub weight: Tensor, // [out_channels, in_channels, kernel_h, kernel_w]
    pub bias: Tensor,   // [out_channels]
    pub stride: usize,
    pub padding: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Conv2dGrads {
    pub grad_input: Tensor,
    pub grad_weight: Tensor,
    pub grad_bias: Tensor,
}

impl Conv2d {
    fn output_height_and_width(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let kernel_h = self.weight.shape[2];
        let kernel_w = self.weight.shape[3];
        let effective_h = input_h as isize + 2 * self.padding as isize - kernel_h as isize;
        let effective_w = input_w as isize + 2 * self.padding as isize - kernel_w as isize;
        assert!(
            effective_h >= 0,
            "conv output height negative: input {} stride {} kernel {} padding {}",
            input_h,
            self.stride,
            kernel_h,
            self.padding
        );
        assert!(
            effective_w >= 0,
            "conv output width negative: input {} stride {} kernel {} padding {}",
            input_w,
            self.stride,
            kernel_w,
            self.padding
        );

        let out_h = effective_h as usize / self.stride + 1;
        let out_w = effective_w as usize / self.stride + 1;
        (out_h, out_w)
    }

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

        let (out_h, out_w) = self.output_height_and_width(in_h, in_w);

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
                                        let x_val =
                                            x.get(&[b, ic, ih_padded as usize, iw_padded as usize]);
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

    pub fn backward(&self, x: &Tensor, grad_out: &Tensor) -> Conv2dGrads {
        assert!(
            x.ndim() == 4,
            "Conv2d backward input must be 4D [B, C, H, W], got shape {:?}",
            x.shape
        );

        let batch = x.shape[0];
        let in_channels = x.shape[1];
        let in_h = x.shape[2];
        let in_w = x.shape[3];
        let (out_h, out_w) = self.output_height_and_width(in_h, in_w);

        assert!(
            grad_out.shape == vec![batch, self.weight.shape[0], out_h, out_w],
            "Conv2d backward grad_out shape mismatch: got {:?}, expected {:?}",
            grad_out.shape,
            vec![batch, self.weight.shape[0], out_h, out_w]
        );

        let mut grad_input = Tensor::zeros(vec![batch, in_channels, in_h, in_w]);
        let mut grad_weight = Tensor::zeros(self.weight.shape.clone());
        let mut grad_bias = Tensor::zeros(vec![self.weight.shape[0]]);

        for b in 0..batch {
            for oc in 0..self.weight.shape[0] {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let output_grad = grad_out.get(&[b, oc, oh, ow]);
                        let current_bias_grad = grad_bias.get(&[oc]);
                        grad_bias.set(&[oc], current_bias_grad + output_grad);

                        let h_origin = oh * self.stride;
                        let w_origin = ow * self.stride;

                        for ic in 0..in_channels {
                            for kh in 0..self.weight.shape[2] {
                                let in_h = h_origin + kh;
                                if in_h < self.padding {
                                    continue;
                                }
                                let ih = in_h - self.padding;
                                if ih >= x.shape[2] {
                                    continue;
                                }

                                for kw in 0..self.weight.shape[3] {
                                    let in_w = w_origin + kw;
                                    if in_w < self.padding {
                                        continue;
                                    }
                                    let iw = in_w - self.padding;
                                    if iw >= x.shape[3] {
                                        continue;
                                    }

                                    let input = x.get(&[b, ic, ih, iw]);
                                    let weight = self.weight.get(&[oc, ic, kh, kw]);
                                    let grad_weight_index =
                                        grad_weight.get(&[oc, ic, kh, kw]) + output_grad * input;

                                    grad_weight.set(&[oc, ic, kh, kw], grad_weight_index);

                                    let grad_input_index =
                                        grad_input.get(&[b, ic, ih, iw]) + output_grad * weight;
                                    grad_input.set(&[b, ic, ih, iw], grad_input_index);
                                }
                            }
                        }
                    }
                }
            }
        }

        Conv2dGrads {
            grad_input,
            grad_weight,
            grad_bias,
        }
    }

    pub fn sgd_step(&mut self, grads: &Conv2dGrads, lr: f32) {
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
