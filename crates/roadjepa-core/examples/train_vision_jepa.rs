use roadjepa_core::{Conv2d, ConvEncoder, EmbeddingEncoder, Linear, Predictor, Tensor, VisionJepa};

fn make_toy_temporal_batch() -> (Tensor, Tensor) {
    let x_t = Tensor::new(
        vec![
            // sample 1
            1.0, 2.0, 3.0, 4.0, // sample 2
            2.0, 1.0, 0.0, 3.0, // sample 3
            4.0, 2.0, 1.0, 5.0, // sample 4
            3.0, 3.0, 2.0, 1.0,
        ],
        vec![4, 1, 2, 2],
    );

    let mut x_t1 = x_t.clone();
    for v in x_t1.data.iter_mut() {
        *v = 1.2 * *v + 0.3;
    }

    (x_t, x_t1)
}

fn main() {
    // Frozen encoder: simple deterministic mapping from image -> 2D embedding
    let conv1 = Conv2d::new(
        Tensor::new(
            vec![
                1.0, // out channel 0
                0.5, // out channel 1
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

    // Trainable predictor: start simple so learning is stable
    let fc1 = Linear::new(
        Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
        Tensor::new(vec![0.0, 0.0], vec![2]),
    );

    let fc2 = Linear::new(
        Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]),
        Tensor::new(vec![0.0, 0.0], vec![2]),
    );

    let predictor = Predictor::new(fc1, fc2);
    let mut model = VisionJepa::new(encoder, predictor);

    let (x_t, x_t1) = make_toy_temporal_batch();

    let num_steps = 200;
    let lr = 0.05;

    for step in 1..=num_steps {
        let (loss, _) = model.step(&x_t, &x_t1, lr);

        if step == 1 || step % 20 == 0 {
            println!("step {:03} | loss {:.6}", step, loss);
        }
    }

    let (pred, target) = model.forward_pair(&x_t, &x_t1);
    let final_loss = model.losses(&x_t, &x_t1).0;

    println!("\nFinal loss: {:.6}", final_loss);
    println!("Predicted next latents: {:?}", pred.data);
    println!("Target next latents:    {:?}", target.data);
}
