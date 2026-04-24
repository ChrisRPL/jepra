use crate::tensor::Tensor;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignedMarginObjectiveConfig {
    pub bank_gap: f32,
    pub sign_gap: f32,
    pub speed_gap: f32,
    pub bank_weight: f32,
    pub sign_weight: f32,
    pub speed_weight: f32,
}

impl Default for SignedMarginObjectiveConfig {
    fn default() -> Self {
        Self {
            bank_gap: 0.05,
            sign_gap: 0.05,
            speed_gap: 0.05,
            bank_weight: 1.0,
            sign_weight: 1.0,
            speed_weight: 1.0,
        }
    }
}

impl SignedMarginObjectiveConfig {
    pub fn assert_valid(&self) {
        for (name, value) in [
            ("bank_gap", self.bank_gap),
            ("sign_gap", self.sign_gap),
            ("speed_gap", self.speed_gap),
            ("bank_weight", self.bank_weight),
            ("sign_weight", self.sign_weight),
            ("speed_weight", self.speed_weight),
        ] {
            assert!(
                value.is_finite() && value >= 0.0,
                "signed margin {} must be finite and non-negative, got {}",
                name,
                value
            );
        }

        assert!(
            self.bank_weight > 0.0 || self.sign_weight > 0.0 || self.speed_weight > 0.0,
            "signed margin objective requires at least one positive component weight"
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignedMarginObjectiveReport {
    pub bank_loss: f32,
    pub sign_loss: f32,
    pub speed_loss: f32,
    pub weighted_loss: f32,
    pub samples: usize,
    pub bank_pairs: usize,
    pub sign_pairs: usize,
    pub speed_pairs: usize,
    pub active_bank_pairs: usize,
    pub active_sign_pairs: usize,
    pub active_speed_pairs: usize,
}

pub fn mse_loss(pred: &Tensor, target: &Tensor) -> f32 {
    assert!(
        pred.shape == target.shape,
        "mse_loss shape mismatch: pred {:?}, target {:?}",
        pred.shape,
        target.shape
    );

    let sum_sq: f32 = pred
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(a, b)| {
            let diff = a - b;
            diff * diff
        })
        .sum();

    sum_sq / pred.len() as f32
}

pub fn signed_margin_objective_loss_and_grad(
    prediction: &Tensor,
    candidate_targets: &[Tensor],
    true_candidate_indices: &[usize],
    candidate_dx: &[isize],
    config: SignedMarginObjectiveConfig,
) -> (SignedMarginObjectiveReport, Tensor) {
    config.assert_valid();
    assert!(
        prediction.ndim() == 2,
        "signed margin prediction must be rank-2, got {:?}",
        prediction.shape
    );
    assert!(
        candidate_targets.len() == candidate_dx.len(),
        "signed margin candidate target/dx mismatch: {} targets, {} labels",
        candidate_targets.len(),
        candidate_dx.len()
    );
    assert!(
        candidate_targets.len() >= 2,
        "signed margin objective requires at least two candidates"
    );
    assert!(
        true_candidate_indices.len() == prediction.shape[0],
        "signed margin true-index count {} must match batch {}",
        true_candidate_indices.len(),
        prediction.shape[0]
    );

    for target in candidate_targets {
        assert!(
            target.shape == prediction.shape,
            "signed margin candidate target shape mismatch: prediction {:?}, candidate {:?}",
            prediction.shape,
            target.shape
        );
    }

    let batch_size = prediction.shape[0];
    let projection_dim = prediction.shape[1];
    assert!(
        projection_dim > 0,
        "signed margin objective requires non-empty projection dim"
    );

    let mut grad = Tensor::zeros(prediction.shape.clone());
    let mut bank_loss_sum = 0.0f32;
    let mut sign_loss_sum = 0.0f32;
    let mut speed_loss_sum = 0.0f32;
    let bank_pairs = batch_size * (candidate_targets.len() - 1);
    let sign_pairs = batch_size;
    let speed_pairs = batch_size;
    let mut active_bank_pairs = 0usize;
    let mut active_sign_pairs = 0usize;
    let mut active_speed_pairs = 0usize;

    for sample in 0..batch_size {
        let true_index = true_candidate_indices[sample];
        assert!(
            true_index < candidate_targets.len(),
            "signed margin true index {} out of bounds for {} candidates",
            true_index,
            candidate_targets.len()
        );
        let true_dx = candidate_dx[true_index];
        let true_target = &candidate_targets[true_index];
        let distances = candidate_targets
            .iter()
            .map(|target| sample_mean_squared_distance(prediction, target, sample))
            .collect::<Vec<_>>();
        let true_distance = distances[true_index];

        for wrong_index in 0..candidate_targets.len() {
            if wrong_index == true_index {
                continue;
            }

            let hinge = config.bank_gap + true_distance - distances[wrong_index];
            if hinge > 0.0 {
                active_bank_pairs += 1;
                bank_loss_sum += hinge;
                add_signed_margin_pair_grad(
                    &mut grad,
                    true_target,
                    &candidate_targets[wrong_index],
                    sample,
                    config.bank_weight / bank_pairs as f32,
                );
            }
        }

        let opposite_sign_index =
            best_candidate_index(candidate_dx, &distances, |candidate_index| {
                candidate_index != true_index
                    && candidate_dx[candidate_index].signum() != true_dx.signum()
            });
        let hinge = config.sign_gap + true_distance - distances[opposite_sign_index];
        if hinge > 0.0 {
            active_sign_pairs += 1;
            sign_loss_sum += hinge;
            add_signed_margin_pair_grad(
                &mut grad,
                true_target,
                &candidate_targets[opposite_sign_index],
                sample,
                config.sign_weight / sign_pairs as f32,
            );
        }

        let same_sign_speed_index =
            best_candidate_index(candidate_dx, &distances, |candidate_index| {
                candidate_index != true_index
                    && candidate_dx[candidate_index].signum() == true_dx.signum()
                    && candidate_dx[candidate_index].abs() != true_dx.abs()
            });
        let hinge = config.speed_gap + true_distance - distances[same_sign_speed_index];
        if hinge > 0.0 {
            active_speed_pairs += 1;
            speed_loss_sum += hinge;
            add_signed_margin_pair_grad(
                &mut grad,
                true_target,
                &candidate_targets[same_sign_speed_index],
                sample,
                config.speed_weight / speed_pairs as f32,
            );
        }
    }

    let bank_loss = bank_loss_sum / bank_pairs as f32;
    let sign_loss = sign_loss_sum / sign_pairs as f32;
    let speed_loss = speed_loss_sum / speed_pairs as f32;
    let weighted_loss = config.bank_weight * bank_loss
        + config.sign_weight * sign_loss
        + config.speed_weight * speed_loss;

    (
        SignedMarginObjectiveReport {
            bank_loss,
            sign_loss,
            speed_loss,
            weighted_loss,
            samples: batch_size,
            bank_pairs,
            sign_pairs,
            speed_pairs,
            active_bank_pairs,
            active_sign_pairs,
            active_speed_pairs,
        },
        grad,
    )
}

fn sample_mean_squared_distance(lhs: &Tensor, rhs: &Tensor, sample: usize) -> f32 {
    assert_eq!(
        lhs.shape, rhs.shape,
        "signed margin distance expects matching shapes, got {:?} and {:?}",
        lhs.shape, rhs.shape
    );
    let dim = lhs.shape[1];
    let mut distance = 0.0f32;

    for feature_idx in 0..dim {
        let diff = lhs.get(&[sample, feature_idx]) - rhs.get(&[sample, feature_idx]);
        distance += diff * diff;
    }

    distance / dim as f32
}

fn add_signed_margin_pair_grad(
    grad: &mut Tensor,
    true_target: &Tensor,
    wrong_target: &Tensor,
    sample: usize,
    scale: f32,
) {
    let projection_dim = grad.shape[1];
    let pair_scale = scale * 2.0 / projection_dim as f32;

    for feature_idx in 0..projection_dim {
        let value = grad.get(&[sample, feature_idx])
            + pair_scale
                * (wrong_target.get(&[sample, feature_idx])
                    - true_target.get(&[sample, feature_idx]));
        grad.set(&[sample, feature_idx], value);
    }
}

fn best_candidate_index(
    candidate_dx: &[isize],
    distances: &[f32],
    mut include_candidate: impl FnMut(usize) -> bool,
) -> usize {
    let mut best_index = None;
    let mut best_distance = f32::INFINITY;

    for candidate_index in 0..candidate_dx.len() {
        if include_candidate(candidate_index) && distances[candidate_index] < best_distance {
            best_index = Some(candidate_index);
            best_distance = distances[candidate_index];
        }
    }

    best_index.unwrap_or_else(|| {
        panic!(
            "signed margin objective missing required candidate group in {:?}",
            candidate_dx
        )
    })
}

#[cfg(test)]
mod tests {
    use super::{SignedMarginObjectiveConfig, signed_margin_objective_loss_and_grad};
    use crate::tensor::Tensor;

    fn signed_margin_fixture_prediction(value: f32) -> Tensor {
        Tensor::new(vec![value, 0.0], vec![1, 2])
    }

    fn signed_margin_fixture_candidates() -> Vec<Tensor> {
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
        let candidate_targets = signed_margin_fixture_candidates();
        let candidate_dx = [-2, -1, 1, 2];
        let true_indices = [0usize];
        let prediction = signed_margin_fixture_prediction(0.4);
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
        let plus = signed_margin_fixture_prediction(0.4 + epsilon);
        let minus = signed_margin_fixture_prediction(0.4 - epsilon);
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
            &signed_margin_fixture_prediction(0.4),
            &signed_margin_fixture_candidates(),
            &[0usize],
            &[-2, -1, 1, 2],
            config,
        );
    }
}

pub fn mse_loss_grad(pred: &Tensor, target: &Tensor) -> Tensor {
    assert!(
        pred.shape == target.shape,
        "mse_loss_grad shape mismatch: pred {:?}, target {:?}",
        pred.shape,
        target.shape
    );

    let scale = 2.0 / pred.len() as f32;

    let data = pred
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(p, t)| scale * (p - t))
        .collect();

    Tensor {
        data,
        shape: pred.shape.clone(),
    }
}
