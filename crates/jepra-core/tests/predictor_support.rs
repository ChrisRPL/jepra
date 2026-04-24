use jepra_core::{
    BottleneckPredictor, Linear, ResidualBottleneckPredictor, Tensor, mse_loss, mse_loss_grad,
};

fn exact_bottleneck_predictor() -> BottleneckPredictor {
    BottleneckPredictor::new(
        Linear::new(
            Tensor::new(
                vec![
                    1.0, -1.0, 0.5, //
                    2.0, 1.0, -2.0,
                ],
                vec![2, 3],
            ),
            Tensor::new(vec![0.0, 0.5, 1.0], vec![3]),
        ),
        Linear::new(
            Tensor::new(
                vec![
                    1.0, 0.0, //
                    0.5, 2.0, //
                    3.0, 1.0,
                ],
                vec![3, 2],
            ),
            Tensor::new(vec![0.0, -1.0], vec![2]),
        ),
        Linear::new(
            Tensor::new(vec![1.0, -2.0], vec![2, 1]),
            Tensor::new(vec![0.25], vec![1]),
        ),
    )
}

#[test]
fn bottleneck_predictor_forward_matches_expected_values() {
    let predictor = exact_bottleneck_predictor();
    let x = Tensor::new(vec![1.0, 2.0], vec![1, 2]);

    let y = predictor.forward(&x);

    assert_eq!(y.shape, vec![1, 1]);
    assert!((y.data[0] - 2.0).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "BottleneckPredictor layer mismatch")]
fn bottleneck_predictor_panics_on_layer_mismatch() {
    let _ = BottleneckPredictor::new(
        Linear::new(Tensor::zeros(vec![2, 3]), Tensor::zeros(vec![3])),
        Linear::new(Tensor::zeros(vec![2, 2]), Tensor::zeros(vec![2])),
        Linear::new(Tensor::zeros(vec![2, 1]), Tensor::zeros(vec![1])),
    );
}

#[test]
fn bottleneck_predictor_backward_shapes_match_layers() {
    let predictor = exact_bottleneck_predictor();
    let x = Tensor::new(vec![1.0, 2.0, 0.5, 1.5], vec![2, 2]);
    let grad_out = Tensor::new(vec![0.25, -0.5], vec![2, 1]);

    let grads = predictor.backward(&x, &grad_out);

    assert_eq!(grads.grad_input.shape, vec![2, 2]);
    assert_eq!(grads.grad_fc1.grad_weight.shape, vec![2, 3]);
    assert_eq!(grads.grad_fc1.grad_bias.shape, vec![3]);
    assert_eq!(grads.grad_fc2.grad_weight.shape, vec![3, 2]);
    assert_eq!(grads.grad_fc2.grad_bias.shape, vec![2]);
    assert_eq!(grads.grad_fc3.grad_weight.shape, vec![2, 1]);
    assert_eq!(grads.grad_fc3.grad_bias.shape, vec![1]);
}

#[test]
fn bottleneck_predictor_sgd_step_reduces_mse_loss() {
    let mut predictor = BottleneckPredictor::new(
        Linear::new(
            Tensor::new(
                vec![
                    1.0, 0.0, //
                    0.0, 1.0,
                ],
                vec![2, 2],
            ),
            Tensor::zeros(vec![2]),
        ),
        Linear::new(
            Tensor::new(
                vec![
                    1.0, 0.0, //
                    0.0, 1.0,
                ],
                vec![2, 2],
            ),
            Tensor::zeros(vec![2]),
        ),
        Linear::new(Tensor::zeros(vec![2, 1]), Tensor::zeros(vec![1])),
    );
    let x = Tensor::new(
        vec![
            1.0, 0.0, //
            0.0, 1.0,
        ],
        vec![2, 2],
    );
    let target = Tensor::new(vec![1.0, 1.0], vec![2, 1]);

    let loss_before = mse_loss(&predictor.forward(&x), &target);
    let grad = mse_loss_grad(&predictor.forward(&x), &target);
    let grads = predictor.backward(&x, &grad);
    predictor.sgd_step(&grads, 0.1);
    let loss_after = mse_loss(&predictor.forward(&x), &target);

    assert!(
        loss_after < loss_before,
        "bottleneck predictor step did not reduce loss: {:.6} -> {:.6}",
        loss_before,
        loss_after
    );
}

#[test]
fn residual_bottleneck_predictor_adds_delta_to_input() {
    let predictor = ResidualBottleneckPredictor::new(BottleneckPredictor::new(
        Linear::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
            Tensor::zeros(vec![2]),
        ),
        Linear::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
            Tensor::zeros(vec![2]),
        ),
        Linear::new(
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
            Tensor::zeros(vec![2]),
        ),
    ));
    let x = Tensor::new(vec![1.0, 2.0], vec![1, 2]);

    let out = predictor.forward(&x);

    assert_eq!(out, Tensor::new(vec![2.0, 4.0], vec![1, 2]));
}

#[test]
fn residual_bottleneck_predictor_backward_preserves_skip_gradient() {
    let predictor = ResidualBottleneckPredictor::new(BottleneckPredictor::new(
        Linear::new(Tensor::zeros(vec![2, 2]), Tensor::zeros(vec![2])),
        Linear::new(Tensor::zeros(vec![2, 1]), Tensor::zeros(vec![1])),
        Linear::new(Tensor::zeros(vec![1, 2]), Tensor::zeros(vec![2])),
    ));
    let x = Tensor::new(vec![1.0, 2.0], vec![1, 2]);
    let grad_out = Tensor::new(vec![0.25, -0.75], vec![1, 2]);

    let grads = predictor.backward(&x, &grad_out);

    assert_eq!(grads.grad_input, grad_out);
}

#[test]
fn residual_bottleneck_predictor_scale_controls_delta_path() {
    let predictor = ResidualBottleneckPredictor::new_scaled(
        BottleneckPredictor::new(
            Linear::new(
                Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
                Tensor::zeros(vec![2]),
            ),
            Linear::new(
                Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
                Tensor::zeros(vec![2]),
            ),
            Linear::new(
                Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
                Tensor::zeros(vec![2]),
            ),
        ),
        0.25,
    );
    let x = Tensor::new(vec![1.0, 2.0], vec![1, 2]);

    let out = predictor.forward(&x);

    assert_eq!(out, Tensor::new(vec![1.25, 2.5], vec![1, 2]));
}
