pub mod init;
pub mod linear;
pub mod losses;
pub mod predictor;
pub mod tensor;

pub use init::{randn, zeros};
pub use linear::{Linear, LinearGrads};
pub use losses::{mse_loss, mse_loss_grad};
pub use predictor::{Predictor, PredictorGrads};
pub use tensor::Tensor;

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
            vec![1.0, -4.0, 3.0, 0.0, -2.0, 8.0, -6.0, 0.0, 3.0, -12.0, 9.0, 0.0,]
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
}
