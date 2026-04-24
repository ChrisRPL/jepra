pub mod conv;
pub mod encoder;
pub mod init;
pub mod linear;
pub mod losses;
pub mod predictor;
pub mod regularizers;
pub mod tensor;
pub mod vision_jepa;

pub use conv::{Conv2d, Conv2dGrads};
pub use encoder::{ConvEncoder, ConvEncoderGrads, EmbeddingEncoder, EmbeddingEncoderGrads};
pub use init::{randn, zeros};
pub use linear::{Linear, LinearGrads};
pub use losses::{mse_loss, mse_loss_grad};
pub use predictor::{
    BottleneckPredictor, BottleneckPredictorGrads, Predictor, PredictorGrads, PredictorModule,
};
pub use regularizers::{
    combine_projection_grads, gaussian_moment_regularizer, gaussian_moment_regularizer_grad,
    projection_stats,
};
pub use tensor::Tensor;
pub use vision_jepa::{ProjectedVisionJepa, VisionJepa};

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

    #[test]
    fn relu_backward_works() {
        let x = Tensor::new(vec![-2.0, -0.5, 0.0, 1.0, 3.5, -4.0], vec![2, 3]);

        let grad_out = Tensor::new(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]);

        let grad_in = x.relu_backward(&grad_out);

        assert_eq!(grad_in.shape, vec![2, 3]);
        assert_eq!(grad_in.data, vec![0.0, 0.0, 0.0, 40.0, 50.0, 0.0]);
    }

    #[test]
    fn relu_backward_preserves_positive_entries() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let grad_out = Tensor::new(vec![0.1, 0.2, 0.3], vec![3]);

        let grad_in = x.relu_backward(&grad_out);

        assert_eq!(grad_in.data, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    #[should_panic]
    fn relu_backward_panics_on_shape_mismatch() {
        let x = Tensor::new(vec![1.0, -1.0, 2.0], vec![3]);
        let grad_out = Tensor::new(vec![1.0, 2.0], vec![2]);

        let _ = x.relu_backward(&grad_out);
    }

    #[test]
    fn tiny_predictor_backward_works() {
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

        let grad_out = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let grads = predictor.backward(&x, &grad_out);

        assert_eq!(grads.grad_fc2.grad_weight.shape, vec![4, 2]);
        assert_eq!(
            grads.grad_fc2.grad_weight.data,
            vec![1.0, 2.0, 6.0, 8.0, 3.0, 6.0, 0.0, 0.0,]
        );
        assert_eq!(grads.grad_fc2.grad_bias.data, vec![4.0, 6.0]);

        assert_eq!(grads.grad_fc1.grad_weight.shape, vec![3, 4]);
        assert_eq!(
            grads.grad_fc1.grad_weight.data,
            vec![
                1.0, -4.0, 3.0, 0.0, -2.0, 8.0, -6.0, 0.0, 3.0, -12.0, 9.0, 0.0,
            ]
        );
        assert_eq!(grads.grad_fc1.grad_bias.data, vec![1.0, 4.0, 3.0, 0.0]);

        assert_eq!(grads.grad_input.shape, vec![2, 3]);
        assert_eq!(grads.grad_input.data, vec![1.0, 0.0, 3.0, 0.0, 4.0, 0.0]);
    }

    #[test]
    #[should_panic]
    fn tiny_predictor_backward_panics_on_grad_shape_mismatch() {
        let fc1 = Linear::new(
            Tensor::new(
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                vec![3, 4],
            ),
            Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![4]),
        );

        let fc2 = Linear::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], vec![4, 2]),
            Tensor::new(vec![0.0, 0.0], vec![2]),
        );

        let predictor = Predictor::new(fc1, fc2);

        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let grad_out = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]);

        let _ = predictor.backward(&x, &grad_out);
    }

    #[test]
    fn tiny_predictor_sgd_step_updates_parameters() {
        let fc1 = Linear::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
            Tensor::new(vec![0.0, 0.0], vec![2]),
        );

        let fc2 = Linear::new(
            Tensor::new(vec![1.0, 1.0], vec![2, 1]),
            Tensor::new(vec![0.0], vec![1]),
        );

        let mut predictor = Predictor::new(fc1, fc2);

        let grads = PredictorGrads {
            grad_input: Tensor::zeros(vec![1, 2]),
            grad_fc1: LinearGrads {
                grad_input: Tensor::zeros(vec![1, 2]),
                grad_weight: Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]),
                grad_bias: Tensor::new(vec![0.5, 0.6], vec![2]),
            },
            grad_fc2: LinearGrads {
                grad_input: Tensor::zeros(vec![1, 2]),
                grad_weight: Tensor::new(vec![0.7, 0.8], vec![2, 1]),
                grad_bias: Tensor::new(vec![0.9], vec![1]),
            },
        };

        predictor.sgd_step(&grads, 0.1);

        assert!((predictor.fc1.weight.data[0] - 0.99).abs() < 1e-6);
        assert!((predictor.fc1.weight.data[1] - (-0.02)).abs() < 1e-6);
        assert!((predictor.fc1.weight.data[2] - (-0.03)).abs() < 1e-6);
        assert!((predictor.fc1.weight.data[3] - 0.96).abs() < 1e-6);

        assert!((predictor.fc1.bias.data[0] - (-0.05)).abs() < 1e-6);
        assert!((predictor.fc1.bias.data[1] - (-0.06)).abs() < 1e-6);

        assert!((predictor.fc2.weight.data[0] - 0.93).abs() < 1e-6);
        assert!((predictor.fc2.weight.data[1] - 0.92).abs() < 1e-6);

        assert!((predictor.fc2.bias.data[0] - (-0.09)).abs() < 1e-6);
    }

    #[test]
    fn tiny_predictor_training_step_reduces_mse_loss() {
        let fc1 = Linear::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
            Tensor::new(vec![0.0, 0.0], vec![2]),
        );

        let fc2 = Linear::new(
            Tensor::new(vec![1.0, 1.0], vec![2, 1]),
            Tensor::new(vec![0.0], vec![1]),
        );

        let mut predictor = Predictor::new(fc1, fc2);

        let x = Tensor::new(vec![1.0, 2.0], vec![1, 2]);
        let target = Tensor::new(vec![5.0], vec![1, 1]);

        let pred_before = predictor.forward(&x);
        let loss_before = mse_loss(&pred_before, &target);

        let grad_out = mse_loss_grad(&pred_before, &target);
        let grads = predictor.backward(&x, &grad_out);
        predictor.sgd_step(&grads, 0.01);

        let pred_after = predictor.forward(&x);
        let loss_after = mse_loss(&pred_after, &target);

        assert!(loss_after < loss_before);
    }

    #[test]
    fn randn_is_reproducible_for_same_seed() {
        let a = randn(vec![2, 3], 0.0, 1.0, 42);
        let b = randn(vec![2, 3], 0.0, 1.0, 42);

        assert_eq!(a.shape, vec![2, 3]);
        assert_eq!(b.shape, vec![2, 3]);
        assert_eq!(a.data, b.data);
    }

    #[test]
    fn linear_randn_creates_valid_shapes() {
        let linear = Linear::randn(3, 4, 0.02, 123);

        assert_eq!(linear.weight.shape, vec![3, 4]);
        assert_eq!(linear.bias.shape, vec![4]);
    }

    #[test]
    fn conv2d_forward_works_without_padding() {
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![1, 1, 3, 3],
        );

        let weight = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![1, 1, 2, 2]);

        let bias = Tensor::new(vec![0.0], vec![1]);

        let conv = Conv2d::new(weight, bias, 1, 0);
        let y = conv.forward(&x);

        assert_eq!(y.shape, vec![1, 1, 2, 2]);
        assert_eq!(y.data, vec![6.0, 8.0, 12.0, 14.0]);
    }

    #[test]
    fn conv2d_forward_works_with_padding() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);

        let weight = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![1, 1, 2, 2]);

        let bias = Tensor::new(vec![0.0], vec![1]);

        let conv = Conv2d::new(weight, bias, 1, 1);
        let y = conv.forward(&x);

        assert_eq!(y.shape, vec![1, 1, 3, 3]);
        assert_eq!(y.data, vec![1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0]);
    }

    #[test]
    #[should_panic]
    fn conv2d_panics_on_channel_mismatch() {
        let x = Tensor::zeros(vec![1, 2, 4, 4]);
        let weight = Tensor::zeros(vec![1, 1, 3, 3]);
        let bias = Tensor::zeros(vec![1]);

        let conv = Conv2d::new(weight, bias, 1, 0);
        let _ = conv.forward(&x);
    }

    #[test]
    fn conv_encoder_forward_works() {
        let conv1 = Conv2d::new(
            Tensor::new(
                vec![
                    1.0, // out channel 0, in channel 0, 1x1 kernel
                    2.0, // out channel 1, in channel 0, 1x1 kernel
                ],
                vec![2, 1, 1, 1],
            ),
            Tensor::new(vec![0.0, 0.0], vec![2]),
            1,
            0,
        );

        let conv2 = Conv2d::new(
            Tensor::new(
                vec![
                    1.0, // out channel 0, in channel 0
                    1.0, // out channel 0, in channel 1
                ],
                vec![1, 2, 1, 1],
            ),
            Tensor::new(vec![0.0], vec![1]),
            1,
            0,
        );

        let encoder = ConvEncoder::new(conv1, conv2);

        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);

        let y = encoder.forward(&x);

        assert_eq!(y.shape, vec![1, 1, 2, 2]);
        assert_eq!(y.data, vec![3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn conv_encoder_relu_blocks_negative_values() {
        let conv1 = Conv2d::new(
            Tensor::new(vec![1.0], vec![1, 1, 1, 1]),
            Tensor::new(vec![0.0], vec![1]),
            1,
            0,
        );

        let conv2 = Conv2d::new(
            Tensor::new(vec![1.0], vec![1, 1, 1, 1]),
            Tensor::new(vec![0.0], vec![1]),
            1,
            0,
        );

        let encoder = ConvEncoder::new(conv1, conv2);

        let x = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![1, 1, 2, 2]);

        let y = encoder.forward(&x);

        assert_eq!(y.shape, vec![1, 1, 2, 2]);
        assert_eq!(y.data, vec![0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn global_avg_pool2d_works() {
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            vec![1, 2, 2, 2],
        );

        let y = x.global_avg_pool2d();

        assert_eq!(y.shape, vec![1, 2]);
        assert_eq!(y.data, vec![2.5, 25.0]);
    }

    #[test]
    fn global_avg_pool2d_works_for_batched_input() {
        let x = Tensor::new(
            vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0],
            vec![2, 1, 2, 2],
        );

        let y = x.global_avg_pool2d();

        assert_eq!(y.shape, vec![2, 1]);
        assert_eq!(y.data, vec![4.0, 5.0]);
    }

    #[test]
    #[should_panic]
    fn global_avg_pool2d_panics_for_non_4d_tensor() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let _ = x.global_avg_pool2d();
    }

    #[test]
    fn embedding_encoder_forward_works() {
        let conv1 = Conv2d::new(
            Tensor::new(
                vec![
                    1.0, // out channel 0, in channel 0, 1x1
                    2.0, // out channel 1, in channel 0, 1x1
                ],
                vec![2, 1, 1, 1],
            ),
            Tensor::new(vec![0.0, 0.0], vec![2]),
            1,
            0,
        );

        let conv2 = Conv2d::new(
            Tensor::new(
                vec![
                    1.0, 0.0, // out channel 0 from channels 0,1
                    0.0, 1.0, // out channel 1 from channels 0,1
                ],
                vec![2, 2, 1, 1],
            ),
            Tensor::new(vec![0.0, 0.0], vec![2]),
            1,
            0,
        );

        let backbone = ConvEncoder::new(conv1, conv2);
        let encoder = EmbeddingEncoder::new(backbone);

        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);

        let y = encoder.forward(&x);

        assert_eq!(y.shape, vec![1, 2]);
        assert_eq!(y.data, vec![2.5, 5.0]);
    }

    #[test]
    fn embedding_encoder_outputs_batched_embeddings() {
        let conv1 = Conv2d::new(
            Tensor::new(vec![1.0], vec![1, 1, 1, 1]),
            Tensor::new(vec![0.0], vec![1]),
            1,
            0,
        );

        let conv2 = Conv2d::new(
            Tensor::new(vec![1.0], vec![1, 1, 1, 1]),
            Tensor::new(vec![0.0], vec![1]),
            1,
            0,
        );

        let backbone = ConvEncoder::new(conv1, conv2);
        let encoder = EmbeddingEncoder::new(backbone);

        let x = Tensor::new(
            vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0],
            vec![2, 1, 2, 2],
        );

        let y = encoder.forward(&x);

        assert_eq!(y.shape, vec![2, 1]);
        assert_eq!(y.data, vec![4.0, 5.0]);
    }

    #[test]
    fn vision_jepa_forward_pair_works() {
        let conv1 = Conv2d::new(
            Tensor::new(
                vec![
                    1.0, // out channel 0
                    2.0, // out channel 1
                ],
                vec![2, 1, 1, 1],
            ),
            Tensor::new(vec![0.0, 0.0], vec![2]),
            1,
            0,
        );

        let conv2 = Conv2d::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2, 1, 1]),
            Tensor::new(vec![0.0, 0.0], vec![2]),
            1,
            0,
        );

        let encoder = EmbeddingEncoder::new(ConvEncoder::new(conv1, conv2));

        let fc1 = Linear::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
            Tensor::new(vec![0.0, 0.0], vec![2]),
        );

        let fc2 = Linear::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
            Tensor::new(vec![0.0, 0.0], vec![2]),
        );

        let predictor = Predictor::new(fc1, fc2);
        let model = VisionJepa::new(encoder, predictor);

        let x_t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);

        let x_t1 = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![1, 1, 2, 2]);

        let (pred, target) = model.forward_pair(&x_t, &x_t1);

        assert_eq!(pred.shape, vec![1, 2]);
        assert_eq!(target.shape, vec![1, 2]);

        assert_eq!(pred.data, vec![2.5, 5.0]);
        assert_eq!(target.data, vec![5.0, 10.0]);
    }

    #[test]
    fn vision_jepa_losses_matches_forward_pair_mse() {
        let conv1 = Conv2d::new(
            Tensor::new(
                vec![
                    1.0, // out channel 0
                    2.0, // out channel 1
                ],
                vec![2, 1, 1, 1],
            ),
            Tensor::new(vec![0.0, 0.0], vec![2]),
            1,
            0,
        );

        let conv2 = Conv2d::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2, 1, 1]),
            Tensor::new(vec![0.0, 0.0], vec![2]),
            1,
            0,
        );

        let encoder = EmbeddingEncoder::new(ConvEncoder::new(conv1, conv2));

        let fc1 = Linear::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
            Tensor::new(vec![0.0, 0.0], vec![2]),
        );

        let fc2 = Linear::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
            Tensor::new(vec![0.0, 0.0], vec![2]),
        );

        let predictor = Predictor::new(fc1, fc2);
        let model = VisionJepa::new(encoder, predictor);

        let x_t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);

        let x_t1 = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![1, 1, 2, 2]);

        let (prediction, target) = model.forward_pair(&x_t, &x_t1);
        let (prediction_loss, total_loss) = model.losses(&x_t, &x_t1);
        let expected_prediction_loss = mse_loss(&prediction, &target);

        assert_eq!(prediction_loss, expected_prediction_loss);
        assert_eq!(total_loss, expected_prediction_loss);
    }

    #[test]
    fn vision_jepa_step_reduces_unprojected_loss() {
        let conv1 = Conv2d::new(
            Tensor::new(
                vec![
                    1.0, // out channel 0
                    2.0, // out channel 1
                ],
                vec![2, 1, 1, 1],
            ),
            Tensor::new(vec![0.0, 0.0], vec![2]),
            1,
            0,
        );

        let conv2 = Conv2d::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2, 1, 1]),
            Tensor::new(vec![0.0, 0.0], vec![2]),
            1,
            0,
        );

        let encoder = EmbeddingEncoder::new(ConvEncoder::new(conv1, conv2));

        let fc1 = Linear::new(
            Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]),
            Tensor::new(vec![0.0, 0.0], vec![2]),
        );

        let fc2 = Linear::new(
            Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]),
            Tensor::new(vec![0.0, 0.0], vec![2]),
        );

        let predictor = Predictor::new(fc1, fc2);
        let mut model = VisionJepa::new(encoder, predictor);

        let x_t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let x_t1 = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![1, 1, 2, 2]);

        let (initial_loss, _) = model.losses(&x_t, &x_t1);
        let (step_loss, _) = model.step(&x_t, &x_t1, 0.2);
        let (final_loss, _) = model.losses(&x_t, &x_t1);

        assert_eq!(step_loss, initial_loss);
        assert!(final_loss + 1e-6 < initial_loss);
    }

    #[test]
    fn vision_jepa_step_with_trainable_encoder_updates_encoder_and_predictor() {
        let conv1 = Conv2d::new(
            Tensor::new(
                vec![
                    1.0, // out channel 0
                    2.0, // out channel 1
                ],
                vec![2, 1, 1, 1],
            ),
            Tensor::new(vec![0.0, 0.0], vec![2]),
            1,
            0,
        );

        let conv2 = Conv2d::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2, 1, 1]),
            Tensor::new(vec![0.0, 0.0], vec![2]),
            1,
            0,
        );

        let encoder = EmbeddingEncoder::new(ConvEncoder::new(conv1, conv2));

        let fc1 = Linear::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
            Tensor::new(vec![0.0, 0.0], vec![2]),
        );

        let fc2 = Linear::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
            Tensor::new(vec![0.0, 0.0], vec![2]),
        );

        let predictor = Predictor::new(fc1, fc2);
        let mut model = VisionJepa::new(encoder, predictor);

        let x_t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let x_t1 = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![1, 1, 2, 2]);

        let encoder_snapshot = model.encoder.clone();
        let predictor_fc1_weight_snapshot = model.predictor.fc1.weight.clone();
        let predictor_fc1_bias_snapshot = model.predictor.fc1.bias.clone();
        let predictor_fc2_weight_snapshot = model.predictor.fc2.weight.clone();
        let predictor_fc2_bias_snapshot = model.predictor.fc2.bias.clone();
        let initial_loss = model.losses(&x_t, &x_t1).0;

        let mut best_loss = initial_loss;

        for _ in 0..4 {
            model.step_with_trainable_encoder(&x_t, &x_t1, 0.02, 0.005);
            best_loss = best_loss.min(model.losses(&x_t, &x_t1).0);
        }

        assert_ne!(model.encoder, encoder_snapshot);
        assert_ne!(model.predictor.fc1.weight, predictor_fc1_weight_snapshot);
        assert_ne!(model.predictor.fc1.bias, predictor_fc1_bias_snapshot);
        assert_ne!(model.predictor.fc2.weight, predictor_fc2_weight_snapshot);
        assert_ne!(model.predictor.fc2.bias, predictor_fc2_bias_snapshot);
        assert!(best_loss < initial_loss);
    }

    #[test]
    fn vision_jepa_step_with_trainable_encoder_can_disable_encoder_update_with_zero_lr() {
        let conv1 = Conv2d::new(
            Tensor::new(
                vec![
                    1.0, // out channel 0
                    2.0, // out channel 1
                ],
                vec![2, 1, 1, 1],
            ),
            Tensor::new(vec![0.0, 0.0], vec![2]),
            1,
            0,
        );

        let conv2 = Conv2d::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2, 1, 1]),
            Tensor::new(vec![0.0, 0.0], vec![2]),
            1,
            0,
        );

        let encoder = EmbeddingEncoder::new(ConvEncoder::new(conv1, conv2));

        let fc1 = Linear::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
            Tensor::new(vec![0.0, 0.0], vec![2]),
        );

        let fc2 = Linear::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
            Tensor::new(vec![0.0, 0.0], vec![2]),
        );

        let predictor = Predictor::new(fc1, fc2);
        let mut model = VisionJepa::new(encoder, predictor);

        let x_t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let x_t1 = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![1, 1, 2, 2]);

        let encoder_snapshot = model.encoder.clone();

        model.step_with_trainable_encoder(&x_t, &x_t1, 0.2, 0.0);

        assert_eq!(model.encoder, encoder_snapshot);
    }

    #[test]
    fn vision_jepa_predict_next_latent_outputs_batched_embeddings() {
        let conv1 = Conv2d::new(
            Tensor::new(vec![1.0], vec![1, 1, 1, 1]),
            Tensor::new(vec![0.0], vec![1]),
            1,
            0,
        );

        let conv2 = Conv2d::new(
            Tensor::new(vec![1.0], vec![1, 1, 1, 1]),
            Tensor::new(vec![0.0], vec![1]),
            1,
            0,
        );

        let encoder = EmbeddingEncoder::new(ConvEncoder::new(conv1, conv2));

        let fc1 = Linear::new(
            Tensor::new(vec![1.0], vec![1, 1]),
            Tensor::new(vec![0.0], vec![1]),
        );

        let fc2 = Linear::new(
            Tensor::new(vec![1.0], vec![1, 1]),
            Tensor::new(vec![0.0], vec![1]),
        );

        let predictor = Predictor::new(fc1, fc2);
        let model = VisionJepa::new(encoder, predictor);

        let x_t = Tensor::new(
            vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0],
            vec![2, 1, 2, 2],
        );

        let pred = model.predict_next_latent(&x_t);

        assert_eq!(pred.shape, vec![2, 1]);
        assert_eq!(pred.data, vec![4.0, 5.0]);
    }

    #[test]
    fn projected_vision_jepa_forward_projection_pair_works() {
        let conv1 = Conv2d::new(
            Tensor::new(
                vec![
                    1.0, // out channel 0
                    2.0, // out channel 1
                ],
                vec![2, 1, 1, 1],
            ),
            Tensor::new(vec![0.0, 0.0], vec![2]),
            1,
            0,
        );

        let conv2 = Conv2d::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2, 1, 1]),
            Tensor::new(vec![0.0, 0.0], vec![2]),
            1,
            0,
        );

        let encoder = EmbeddingEncoder::new(ConvEncoder::new(conv1, conv2));

        let projector = Linear::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
            Tensor::new(vec![0.0, 0.0], vec![2]),
        );

        let target_projector = Linear::new(
            Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]),
            Tensor::new(vec![0.0, 0.0], vec![2]),
        );

        let predictor = Predictor::new(
            Linear::new(
                Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
                Tensor::new(vec![0.0, 0.0], vec![2]),
            ),
            Linear::new(
                Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
                Tensor::new(vec![0.0, 0.0], vec![2]),
            ),
        );

        let model = ProjectedVisionJepa::new(encoder, projector, target_projector, predictor);

        let x_t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let x_t1 = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![1, 1, 2, 2]);

        let (prediction, target) = model.forward_projection_pair(&x_t, &x_t1);

        let expected_projection = model.project_latent(&x_t);
        let expected_prediction = model.predictor.forward(&expected_projection);
        let expected_target = model.target_projection(&x_t1);

        assert_eq!(prediction.shape, vec![1, 2]);
        assert_eq!(target.shape, vec![1, 2]);
        assert_eq!(prediction, expected_prediction);
        assert_eq!(target, expected_target);
    }
}
