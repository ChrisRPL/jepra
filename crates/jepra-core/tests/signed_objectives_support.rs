use jepra_core::{
    SignedBankSoftmaxObjectiveConfig, SignedMarginObjectiveConfig, Tensor,
    signed_bank_softmax_objective_loss_and_grad, signed_margin_objective_loss_and_grad,
    signed_radial_calibration_loss_and_grad,
};

fn signed_objective_fixture_prediction(value: f32) -> Tensor {
    Tensor::new(vec![value, 0.0], vec![1, 2])
}

fn signed_objective_fixture_candidates() -> Vec<Tensor> {
    vec![
        Tensor::new(vec![0.0, 0.0], vec![1, 2]),
        Tensor::new(vec![0.6, 0.0], vec![1, 2]),
        Tensor::new(vec![0.7, 0.0], vec![1, 2]),
        Tensor::new(vec![1.0, 0.0], vec![1, 2]),
    ]
}

#[test]
fn signed_margin_objective_grad_matches_finite_difference() {
    let config = SignedMarginObjectiveConfig::default();
    let candidate_targets = signed_objective_fixture_candidates();
    let candidate_dx = [-2, -1, 1, 2];
    let true_indices = [0usize];
    let prediction = signed_objective_fixture_prediction(0.4);
    let (report, grad) = signed_margin_objective_loss_and_grad(
        &prediction,
        &candidate_targets,
        &true_indices,
        &candidate_dx,
        config,
    );

    assert!(report.weighted_loss.is_finite() && report.weighted_loss > 0.0);
    assert!(grad.data.iter().all(|value| value.is_finite()));

    let epsilon = 1e-3;
    let plus = signed_objective_fixture_prediction(0.4 + epsilon);
    let minus = signed_objective_fixture_prediction(0.4 - epsilon);
    let plus_loss = signed_margin_objective_loss_and_grad(
        &plus,
        &candidate_targets,
        &true_indices,
        &candidate_dx,
        config,
    )
    .0
    .weighted_loss;
    let minus_loss = signed_margin_objective_loss_and_grad(
        &minus,
        &candidate_targets,
        &true_indices,
        &candidate_dx,
        config,
    )
    .0
    .weighted_loss;
    let finite_difference = (plus_loss - minus_loss) / (2.0 * epsilon);

    assert!(
        (finite_difference - grad.data[0]).abs() < 1e-3,
        "finite difference {:.6} != grad {:.6}",
        finite_difference,
        grad.data[0]
    );
}

#[test]
#[should_panic(expected = "requires at least one positive component weight")]
fn signed_margin_objective_rejects_zero_component_weights() {
    let config = SignedMarginObjectiveConfig {
        bank_weight: 0.0,
        sign_weight: 0.0,
        speed_weight: 0.0,
        ..SignedMarginObjectiveConfig::default()
    };
    let _ = signed_margin_objective_loss_and_grad(
        &signed_objective_fixture_prediction(0.4),
        &signed_objective_fixture_candidates(),
        &[0usize],
        &[-2, -1, 1, 2],
        config,
    );
}

#[test]
fn signed_bank_softmax_objective_grad_matches_finite_difference() {
    let config = SignedBankSoftmaxObjectiveConfig { temperature: 0.7 };
    let candidate_targets = signed_objective_fixture_candidates();
    let true_indices = [1usize];
    let prediction = signed_objective_fixture_prediction(0.4);
    let (report, grad) = signed_bank_softmax_objective_loss_and_grad(
        &prediction,
        &candidate_targets,
        &true_indices,
        config,
    );

    assert!(report.loss.is_finite() && report.loss > 0.0);
    assert!((0.0..=1.0).contains(&report.mean_true_probability));
    assert!(grad.data.iter().all(|value| value.is_finite()));

    let epsilon = 1e-3;
    let plus = signed_objective_fixture_prediction(0.4 + epsilon);
    let minus = signed_objective_fixture_prediction(0.4 - epsilon);
    let plus_loss = signed_bank_softmax_objective_loss_and_grad(
        &plus,
        &candidate_targets,
        &true_indices,
        config,
    )
    .0
    .loss;
    let minus_loss = signed_bank_softmax_objective_loss_and_grad(
        &minus,
        &candidate_targets,
        &true_indices,
        config,
    )
    .0
    .loss;
    let finite_difference = (plus_loss - minus_loss) / (2.0 * epsilon);

    assert!(
        (finite_difference - grad.data[0]).abs() < 1e-3,
        "finite difference {:.6} != grad {:.6}",
        finite_difference,
        grad.data[0]
    );
}

#[test]
#[should_panic(expected = "temperature must be finite and positive")]
fn signed_bank_softmax_objective_rejects_non_positive_temperature() {
    let _ = signed_bank_softmax_objective_loss_and_grad(
        &signed_objective_fixture_prediction(0.4),
        &signed_objective_fixture_candidates(),
        &[0usize],
        SignedBankSoftmaxObjectiveConfig { temperature: 0.0 },
    );
}

#[test]
fn signed_radial_calibration_grad_matches_finite_difference() {
    let candidate_targets = signed_objective_fixture_candidates();
    let true_indices = [3usize];
    let prediction = signed_objective_fixture_prediction(0.4);
    let (report, grad) =
        signed_radial_calibration_loss_and_grad(&prediction, &candidate_targets, &true_indices);

    assert!(report.loss.is_finite() && report.loss > 0.0);
    assert!(report.prediction_norm.is_finite() && report.prediction_norm >= 0.0);
    assert!(report.target_norm.is_finite() && report.target_norm >= 0.0);
    assert!(report.norm_ratio.is_finite() && report.norm_ratio >= 0.0);
    assert_eq!(report.samples, 1);
    assert!(grad.data.iter().all(|value| value.is_finite()));

    let epsilon = 1e-3;
    let plus = signed_objective_fixture_prediction(0.4 + epsilon);
    let minus = signed_objective_fixture_prediction(0.4 - epsilon);
    let plus_loss =
        signed_radial_calibration_loss_and_grad(&plus, &candidate_targets, &true_indices)
            .0
            .loss;
    let minus_loss =
        signed_radial_calibration_loss_and_grad(&minus, &candidate_targets, &true_indices)
            .0
            .loss;
    let finite_difference = (plus_loss - minus_loss) / (2.0 * epsilon);

    assert!(
        (finite_difference - grad.data[0]).abs() < 1e-3,
        "finite difference {:.6} != grad {:.6}",
        finite_difference,
        grad.data[0]
    );
}
