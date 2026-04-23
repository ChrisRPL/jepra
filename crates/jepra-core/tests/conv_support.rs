use jepra_core::{Conv2d, Conv2dGrads, ConvEncoder, ConvEncoderGrads, EmbeddingEncoder, Tensor};

fn weighted_output_sum(conv: &Conv2d, x: &Tensor, grad_out: &Tensor) -> f32 {
    let output = conv.forward(x);
    assert!(
        output.shape == grad_out.shape,
        "weighted output expected shape {:?}, got {:?}",
        output.shape,
        grad_out.shape
    );

    output
        .data
        .iter()
        .zip(grad_out.data.iter())
        .map(|(value, weight)| value * weight)
        .sum()
}

fn finite_difference_weight(
    conv: &Conv2d,
    x: &Tensor,
    grad_out: &Tensor,
    weight_index: usize,
    epsilon: f32,
) -> f32 {
    let mut plus = conv.clone();
    plus.weight.data[weight_index] += epsilon;
    let plus_sum = weighted_output_sum(&plus, x, grad_out);

    let mut minus = conv.clone();
    minus.weight.data[weight_index] -= epsilon;
    let minus_sum = weighted_output_sum(&minus, x, grad_out);

    (plus_sum - minus_sum) / (2.0 * epsilon)
}

fn finite_difference_bias(
    conv: &Conv2d,
    x: &Tensor,
    grad_out: &Tensor,
    bias_index: usize,
    epsilon: f32,
) -> f32 {
    let mut plus = conv.clone();
    plus.bias.data[bias_index] += epsilon;
    let plus_sum = weighted_output_sum(&plus, x, grad_out);

    let mut minus = conv.clone();
    minus.bias.data[bias_index] -= epsilon;
    let minus_sum = weighted_output_sum(&minus, x, grad_out);

    (plus_sum - minus_sum) / (2.0 * epsilon)
}

fn finite_difference_input(
    conv: &Conv2d,
    x: &Tensor,
    grad_out: &Tensor,
    input_index: usize,
    epsilon: f32,
) -> f32 {
    let mut plus_input = x.clone();
    plus_input.data[input_index] += epsilon;
    let plus_sum = weighted_output_sum(conv, &plus_input, grad_out);

    let mut minus_input = x.clone();
    minus_input.data[input_index] -= epsilon;
    let minus_sum = weighted_output_sum(conv, &minus_input, grad_out);

    (plus_sum - minus_sum) / (2.0 * epsilon)
}

#[test]
fn conv2d_backward_single_position_kernel_matches_expected_values() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let conv = Conv2d::new(
        Tensor::new(vec![1.5], vec![1, 1, 1, 1]),
        Tensor::new(vec![0.25], vec![1]),
        1,
        0,
    );
    let grad_out = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let grads = conv.backward(&x, &grad_out);

    assert_eq!(
        grads.grad_input,
        Tensor::new(vec![1.5, 3.0, 4.5, 6.0], vec![1, 1, 2, 2])
    );
    assert_eq!(grads.grad_weight, Tensor::new(vec![30.0], vec![1, 1, 1, 1]));
    assert_eq!(grads.grad_bias, Tensor::new(vec![10.0], vec![1]));
}

#[test]
fn conv2d_backward_matches_finite_difference_checks() {
    let conv = Conv2d::new(
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]),
        Tensor::new(vec![0.15], vec![1]),
        1,
        0,
    );
    let x = Tensor::new(
        (1..10).map(|value| value as f32).collect(),
        vec![1, 1, 3, 3],
    );
    let grad_out = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let grads = conv.backward(&x, &grad_out);
    let epsilon = 1e-2;

    for i in 0..conv.weight.data.len() {
        let numerical = finite_difference_weight(&conv, &x, &grad_out, i, epsilon);
        assert!(
            (grads.grad_weight.data[i] - numerical).abs() < 1e-2,
            "weight grad mismatch idx {}: expected {:.6}, numerical {:.6}",
            i,
            grads.grad_weight.data[i],
            numerical
        );
    }

    for i in 0..conv.bias.data.len() {
        let numerical = finite_difference_bias(&conv, &x, &grad_out, i, epsilon);
        assert!(
            (grads.grad_bias.data[i] - numerical).abs() < 1e-2,
            "bias grad mismatch idx {}: expected {:.6}, numerical {:.6}",
            i,
            grads.grad_bias.data[i],
            numerical
        );
    }

    for i in 0..x.data.len() {
        let numerical = finite_difference_input(&conv, &x, &grad_out, i, epsilon);
        assert!(
            (grads.grad_input.data[i] - numerical).abs() < 1e-2,
            "input grad mismatch idx {}: expected {:.6}, numerical {:.6}",
            i,
            grads.grad_input.data[i],
            numerical
        );
    }
}

#[test]
fn conv2d_sgd_step_reduces_parameters_given_gradient_signal() {
    let weight = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let bias = Tensor::new(vec![0.5], vec![1]);
    let mut conv = Conv2d::new(weight, bias, 1, 0);
    let grads = Conv2dGrads {
        grad_input: Tensor::zeros(vec![1, 1, 2, 2]),
        grad_weight: Tensor::new(vec![0.4, 0.5, 0.6, 0.7], vec![1, 1, 2, 2]),
        grad_bias: Tensor::new(vec![1.1], vec![1]),
    };
    let mut expected_weight = vec![1.0, 2.0, 3.0, 4.0];
    let mut expected_bias = vec![0.5];
    for (value, step) in expected_weight
        .iter_mut()
        .zip(grads.grad_weight.data.iter())
    {
        *value -= 0.1 * step;
    }
    expected_bias[0] -= 0.1 * grads.grad_bias.data[0];

    conv.sgd_step(&grads, 0.1);

    assert_eq!(conv.weight, Tensor::new(expected_weight, vec![1, 1, 2, 2]));
    assert_eq!(conv.bias, Tensor::new(expected_bias, vec![1]));
}

#[test]
fn conv_encoder_backward_with_1x1_layers_matches_expected_shapes_and_values() {
    let conv1 = Conv2d::new(
        Tensor::new(vec![1.0], vec![1, 1, 1, 1]),
        Tensor::new(vec![0.0], vec![1]),
        1,
        0,
    );
    let conv2 = Conv2d::new(
        Tensor::new(vec![2.0], vec![1, 1, 1, 1]),
        Tensor::new(vec![0.0], vec![1]),
        1,
        0,
    );
    let encoder = ConvEncoder::new(conv1, conv2);
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let grad_latent = Tensor::new(vec![1.0], vec![1, 1]);
    let ConvEncoderGrads {
        grad_conv1,
        grad_conv2,
    } = encoder.backward(&x, &grad_latent);

    assert_eq!(
        grad_conv1.grad_weight,
        Tensor::new(vec![5.0], vec![1, 1, 1, 1])
    );
    assert_eq!(grad_conv1.grad_bias, Tensor::new(vec![2.0], vec![1]));
    assert_eq!(
        grad_conv2.grad_weight,
        Tensor::new(vec![2.5], vec![1, 1, 1, 1])
    );
    assert_eq!(grad_conv2.grad_bias, Tensor::new(vec![1.0], vec![1]));
}

#[test]
fn embedding_encoder_backward_compiles_with_conv_chain() {
    let conv1 = Conv2d::new(
        Tensor::new(vec![1.0], vec![1, 1, 1, 1]),
        Tensor::new(vec![0.0], vec![1]),
        1,
        0,
    );
    let conv2 = Conv2d::new(
        Tensor::new(vec![2.0], vec![1, 1, 1, 1]),
        Tensor::new(vec![0.0], vec![1]),
        1,
        0,
    );
    let encoder = EmbeddingEncoder::new(ConvEncoder::new(conv1, conv2));
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let grad_latent = Tensor::new(vec![1.0], vec![1, 1]);
    let grads = encoder.backward(&x, &grad_latent);

    assert_eq!(
        grads.grad_backbone.grad_conv1.grad_bias,
        Tensor::new(vec![2.0], vec![1])
    );
    assert_eq!(
        grads.grad_backbone.grad_conv2.grad_bias,
        Tensor::new(vec![1.0], vec![1])
    );
}
