use roadjepa_core::{mse_loss, mse_loss_grad, Linear, Tensor, Predictor};

fn main() {
    let fc1 = Linear::new(
        Tensor::new(
            vec![
                1.0, 0.0,
                0.0, 1.0,
            ],
            vec![2, 2],
        ),
        Tensor::new(vec![0.0, 0.0], vec![2]),
    );

    let fc2 = Linear::new(
        Tensor::new(
            vec![
                0.1,
                0.1,
            ],
            vec![2, 1],
        ),
        Tensor::new(vec![0.0], vec![1]),
    );

    let mut predictor = Predictor::new(fc1, fc2);

    // Simple toy regression task:
    // target = x1 + 2 * x2
    let x = Tensor::new(
        vec![
            1.0, 2.0,
            2.0, 1.0,
            3.0, 1.0,
            1.0, 3.0,
        ],
        vec![4, 2],
    );

    let target = Tensor::new(
        vec![
            5.0,
            4.0,
            5.0,
            7.0,
        ],
        vec![4, 1],
    );

    let num_steps = 100;
    let lr = 0.01;

    for step in 1..=num_steps {
        let pred = predictor.forward(&x);
        let loss = mse_loss(&pred, &target);

        let grad_out = mse_loss_grad(&pred, &target);
        let grads = predictor.backward(&x, &grad_out);
        predictor.sgd_step(&grads, lr);

        if step == 1 || step % 10 == 0 {
            println!("step {:03} | loss {:.6}", step, loss);
        }
    }

    let final_pred = predictor.forward(&x);
    let final_loss = mse_loss(&final_pred, &target);

    println!("\nFinal loss: {:.6}", final_loss);
    println!("Final predictions: {:?}", final_pred.data);
    println!("Targets:           {:?}", target.data);
}