use jepra_core::signed_objectives::signed_direct_candidate_margin_objective_loss_and_grad;
use jepra_core::{
    SignedAngularRadialObjectiveConfig, SignedBankSoftmaxObjectiveConfig,
    SignedMarginObjectiveConfig, Tensor, signed_angular_radial_objective_loss_and_grad,
    signed_bank_softmax_objective_loss_and_grad, signed_candidate_centered_radius_targets,
    signed_centered_radius_scalar_loss_and_grad, signed_margin_objective_loss_and_grad,
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

fn angular_radial_fixture_prediction(first_value: f32) -> Tensor {
    Tensor::new(vec![first_value, 0.25], vec![1, 2])
}

fn angular_radial_fixture_candidates() -> Vec<Tensor> {
    vec![
        Tensor::new(vec![0.0, 0.0], vec![1, 2]),
        Tensor::new(vec![0.6, 0.2], vec![1, 2]),
        Tensor::new(vec![0.7, -0.1], vec![1, 2]),
        Tensor::new(vec![1.0, 0.8], vec![1, 2]),
    ]
}

fn centered_radius_fixture_candidates() -> Vec<Tensor> {
    vec![
        Tensor::new(vec![0.0, 0.0, 1.0, 1.0], vec![2, 2]),
        Tensor::new(vec![2.0, 0.0, 1.0, 3.0], vec![2, 2]),
        Tensor::new(vec![0.0, 2.0, 4.0, 1.0], vec![2, 2]),
    ]
}

fn direct_candidate_margin_fixture_prediction(value: f32) -> Tensor {
    Tensor::new(vec![value, 0.2], vec![1, 2])
}

fn direct_candidate_margin_fixture_candidates() -> Vec<Tensor> {
    vec![
        Tensor::new(vec![0.0, 0.0], vec![1, 2]),
        Tensor::new(vec![1.0, 0.2], vec![1, 2]),
        Tensor::new(vec![3.0, 3.0], vec![1, 2]),
    ]
}

fn direct_candidate_margin_batch_prediction() -> Tensor {
    Tensor::new(vec![0.4, 0.2, 1.05, 0.0], vec![2, 2])
}

fn direct_candidate_margin_batch_candidates() -> Vec<Tensor> {
    vec![
        Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]),
        Tensor::new(vec![1.0, 0.2, 1.0, 0.0], vec![2, 2]),
        Tensor::new(vec![3.0, 3.0, 1.2, 0.0], vec![2, 2]),
    ]
}

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 1e-6,
        "actual {:.8} != expected {:.8}",
        actual,
        expected
    );
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
fn signed_direct_candidate_margin_reports_hard_wrong_metrics() {
    let candidate_targets = direct_candidate_margin_batch_candidates();
    let true_indices = [0usize, 1usize];
    let prediction = direct_candidate_margin_batch_prediction();
    let (report, grad) = signed_direct_candidate_margin_objective_loss_and_grad(
        &prediction,
        &candidate_targets,
        &true_indices,
        0.05,
    );

    assert_close(report.loss, 0.02);
    assert_close(report.active_rate, 0.5);
    assert_close(report.true_distance, 0.050625);
    assert_close(report.wrong_distance, 0.095625);
    assert_close(report.margin, 0.045);
    assert_close(report.positive_margin_rate, 1.0);
    assert_close(report.top1, 1.0);
    assert_eq!(report.samples, 2);
    assert_eq!(report.active_count, 1);
    assert_eq!(grad.shape, vec![2, 2]);
    assert_close(grad.data[0], 0.0);
    assert_close(grad.data[1], 0.0);
    assert_close(grad.data[2], 0.1);
    assert_close(grad.data[3], 0.0);
}

#[test]
fn signed_direct_candidate_margin_uses_deterministic_ties() {
    let prediction = Tensor::new(vec![0.0, 0.0], vec![1, 2]);
    let candidate_targets = vec![
        Tensor::new(vec![-1.0, 0.0], vec![1, 2]),
        Tensor::new(vec![1.0, 0.0], vec![1, 2]),
        Tensor::new(vec![0.0, 1.0], vec![1, 2]),
    ];
    let (report, grad) = signed_direct_candidate_margin_objective_loss_and_grad(
        &prediction,
        &candidate_targets,
        &[1usize],
        0.5,
    );

    assert_close(report.loss, 0.5);
    assert_close(report.true_distance, 0.5);
    assert_close(report.wrong_distance, 0.5);
    assert_close(report.margin, 0.0);
    assert_close(report.positive_margin_rate, 0.0);
    assert_close(report.top1, 0.0);
    assert_eq!(report.active_count, 1);
    assert_close(grad.data[0], -2.0);
    assert_close(grad.data[1], 0.0);
}

#[test]
fn signed_direct_candidate_margin_grad_matches_finite_difference() {
    let candidate_targets = direct_candidate_margin_fixture_candidates();
    let true_indices = [0usize];
    let prediction = direct_candidate_margin_fixture_prediction(0.4);
    let (report, grad) = signed_direct_candidate_margin_objective_loss_and_grad(
        &prediction,
        &candidate_targets,
        &true_indices,
        0.5,
    );

    assert!(report.loss.is_finite() && report.loss > 0.0);
    assert_eq!(report.active_count, 1);
    assert!(grad.data.iter().all(|value| value.is_finite()));

    let epsilon = 1e-3;
    let plus = direct_candidate_margin_fixture_prediction(0.4 + epsilon);
    let minus = direct_candidate_margin_fixture_prediction(0.4 - epsilon);
    let plus_loss = signed_direct_candidate_margin_objective_loss_and_grad(
        &plus,
        &candidate_targets,
        &true_indices,
        0.5,
    )
    .0
    .loss;
    let minus_loss = signed_direct_candidate_margin_objective_loss_and_grad(
        &minus,
        &candidate_targets,
        &true_indices,
        0.5,
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
fn signed_direct_candidate_margin_inactive_hinge_has_zero_grad() {
    let candidate_targets = direct_candidate_margin_fixture_candidates();
    let true_indices = [0usize];
    let prediction = Tensor::new(vec![0.01, 0.0], vec![1, 2]);
    let (report, grad) = signed_direct_candidate_margin_objective_loss_and_grad(
        &prediction,
        &candidate_targets,
        &true_indices,
        0.05,
    );

    assert_close(report.loss, 0.0);
    assert_close(report.active_rate, 0.0);
    assert_eq!(report.active_count, 0);
    assert!(grad.data.iter().all(|value| value.abs() < 1e-6));
}

#[test]
#[should_panic(expected = "margin must be finite and non-negative")]
fn signed_direct_candidate_margin_rejects_negative_margin() {
    let _ = signed_direct_candidate_margin_objective_loss_and_grad(
        &direct_candidate_margin_fixture_prediction(0.4),
        &direct_candidate_margin_fixture_candidates(),
        &[0usize],
        -0.1,
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

#[test]
fn signed_candidate_centered_radius_targets_match_bank_centroid_geometry() {
    let candidate_targets = centered_radius_fixture_candidates();
    let true_indices = [1usize, 2usize];
    let targets = signed_candidate_centered_radius_targets(&candidate_targets, &true_indices);

    assert_eq!(targets.shape, vec![2, 1]);
    assert!((targets.get(&[0, 0]) - 20.0f32.sqrt() / 3.0).abs() < 1e-6);
    assert!((targets.get(&[1, 0]) - 40.0f32.sqrt() / 3.0).abs() < 1e-6);
}

#[test]
fn signed_centered_radius_scalar_grad_matches_finite_difference() {
    let candidate_targets = centered_radius_fixture_candidates();
    let true_indices = [1usize, 2usize];
    let prediction = Tensor::new(vec![1.0, 2.0], vec![2, 1]);
    let (report, grad) =
        signed_centered_radius_scalar_loss_and_grad(&prediction, &candidate_targets, &true_indices);

    assert!(report.loss.is_finite() && report.loss > 0.0);
    assert!(report.prediction_radius.is_finite());
    assert!(report.target_radius.is_finite() && report.target_radius >= 0.0);
    assert!(report.radius_ratio.is_finite());
    assert_eq!(report.samples, 2);
    assert_eq!(grad.shape, vec![2, 1]);
    assert!(grad.data.iter().all(|value| value.is_finite()));

    let epsilon = 1e-3;
    let plus = Tensor::new(vec![1.0 + epsilon, 2.0], vec![2, 1]);
    let minus = Tensor::new(vec![1.0 - epsilon, 2.0], vec![2, 1]);
    let plus_loss =
        signed_centered_radius_scalar_loss_and_grad(&plus, &candidate_targets, &true_indices)
            .0
            .loss;
    let minus_loss =
        signed_centered_radius_scalar_loss_and_grad(&minus, &candidate_targets, &true_indices)
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
fn signed_centered_radius_scalar_accepts_rank_one_predictions() {
    let candidate_targets = centered_radius_fixture_candidates();
    let true_indices = [1usize, 2usize];
    let prediction = Tensor::new(vec![1.0, 2.0], vec![2]);
    let (report, grad) =
        signed_centered_radius_scalar_loss_and_grad(&prediction, &candidate_targets, &true_indices);

    assert!(report.loss.is_finite());
    assert_eq!(grad.shape, vec![2]);
    assert!(grad.data.iter().all(|value| value.is_finite()));
}

#[test]
fn signed_angular_radial_objective_grad_matches_finite_difference() {
    let config = SignedAngularRadialObjectiveConfig {
        angular_weight: 0.7,
        radial_weight: 0.3,
    };
    let candidate_targets = angular_radial_fixture_candidates();
    let true_indices = [3usize];
    let prediction = angular_radial_fixture_prediction(0.4);
    let (report, grad) = signed_angular_radial_objective_loss_and_grad(
        &prediction,
        &candidate_targets,
        &true_indices,
        config,
    );

    assert!(report.loss.is_finite() && report.loss > 0.0);
    assert!(report.angular_loss.is_finite() && report.angular_loss >= 0.0);
    assert!(report.radial_loss.is_finite() && report.radial_loss >= 0.0);
    assert!((-1.0..=1.0).contains(&report.cosine));
    assert!(report.prediction_norm.is_finite() && report.prediction_norm >= 0.0);
    assert!(report.target_norm.is_finite() && report.target_norm >= 0.0);
    assert!(report.norm_ratio.is_finite() && report.norm_ratio >= 0.0);
    assert_eq!(report.samples, 1);
    assert!(grad.data.iter().all(|value| value.is_finite()));

    let epsilon = 1e-3;
    let plus = angular_radial_fixture_prediction(0.4 + epsilon);
    let minus = angular_radial_fixture_prediction(0.4 - epsilon);
    let plus_loss = signed_angular_radial_objective_loss_and_grad(
        &plus,
        &candidate_targets,
        &true_indices,
        config,
    )
    .0
    .loss;
    let minus_loss = signed_angular_radial_objective_loss_and_grad(
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
#[should_panic(expected = "requires at least one positive component weight")]
fn signed_angular_radial_objective_rejects_zero_component_weights() {
    let _ = signed_angular_radial_objective_loss_and_grad(
        &angular_radial_fixture_prediction(0.4),
        &angular_radial_fixture_candidates(),
        &[3usize],
        SignedAngularRadialObjectiveConfig {
            angular_weight: 0.0,
            radial_weight: 0.0,
        },
    );
}
