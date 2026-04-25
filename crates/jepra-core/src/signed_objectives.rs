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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignedBankSoftmaxObjectiveConfig {
    pub temperature: f32,
}

impl Default for SignedBankSoftmaxObjectiveConfig {
    fn default() -> Self {
        Self { temperature: 1.0 }
    }
}

impl SignedBankSoftmaxObjectiveConfig {
    pub fn assert_valid(&self) {
        assert!(
            self.temperature.is_finite() && self.temperature > 0.0,
            "signed bank softmax temperature must be finite and positive, got {}",
            self.temperature
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignedBankSoftmaxObjectiveReport {
    pub loss: f32,
    pub top1: f32,
    pub mean_true_probability: f32,
    pub samples: usize,
}

pub fn signed_bank_softmax_objective_loss_and_grad(
    prediction: &Tensor,
    candidate_targets: &[Tensor],
    true_candidate_indices: &[usize],
    config: SignedBankSoftmaxObjectiveConfig,
) -> (SignedBankSoftmaxObjectiveReport, Tensor) {
    config.assert_valid();
    assert!(
        prediction.ndim() == 2,
        "signed bank softmax prediction must be rank-2, got {:?}",
        prediction.shape
    );
    assert!(
        candidate_targets.len() >= 2,
        "signed bank softmax objective requires at least two candidates"
    );
    assert!(
        true_candidate_indices.len() == prediction.shape[0],
        "signed bank softmax true-index count {} must match batch {}",
        true_candidate_indices.len(),
        prediction.shape[0]
    );

    for target in candidate_targets {
        assert!(
            target.shape == prediction.shape,
            "signed bank softmax candidate target shape mismatch: prediction {:?}, candidate {:?}",
            prediction.shape,
            target.shape
        );
    }

    let batch_size = prediction.shape[0];
    let projection_dim = prediction.shape[1];
    assert!(
        projection_dim > 0,
        "signed bank softmax objective requires non-empty projection dim"
    );

    let mut grad = Tensor::zeros(prediction.shape.clone());
    let mut loss_sum = 0.0f32;
    let mut top1_count = 0usize;
    let mut true_probability_sum = 0.0f32;

    for sample in 0..batch_size {
        let true_index = true_candidate_indices[sample];
        assert!(
            true_index < candidate_targets.len(),
            "signed bank softmax true index {} out of bounds for {} candidates",
            true_index,
            candidate_targets.len()
        );

        let logits = candidate_targets
            .iter()
            .map(|target| {
                -sample_mean_squared_distance(prediction, target, sample) / config.temperature
            })
            .collect::<Vec<_>>();
        let max_logit = logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, |max_value, value| max_value.max(value));
        let exp_logits = logits
            .iter()
            .map(|logit| (logit - max_logit).exp())
            .collect::<Vec<_>>();
        let exp_sum: f32 = exp_logits.iter().sum();
        assert!(
            exp_sum.is_finite() && exp_sum > 0.0,
            "signed bank softmax produced invalid normalizer {}",
            exp_sum
        );

        let log_sum_exp = max_logit + exp_sum.ln();
        let true_probability = exp_logits[true_index] / exp_sum;
        let loss = log_sum_exp - logits[true_index];
        assert!(
            loss.is_finite() && true_probability.is_finite(),
            "signed bank softmax produced invalid loss/probability"
        );

        loss_sum += loss;
        true_probability_sum += true_probability;

        let best_index = logits
            .iter()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
            .map(|(index, _)| index)
            .expect("signed bank softmax logits cannot be empty");
        top1_count += usize::from(best_index == true_index);

        let grad_scale = 2.0 / (batch_size as f32 * projection_dim as f32 * config.temperature);
        for feature_idx in 0..projection_dim {
            let expected_target = candidate_targets
                .iter()
                .zip(exp_logits.iter())
                .map(|(target, exp_logit)| {
                    let probability = exp_logit / exp_sum;
                    probability * target.get(&[sample, feature_idx])
                })
                .sum::<f32>();
            let value = grad.get(&[sample, feature_idx])
                + grad_scale
                    * (expected_target - candidate_targets[true_index].get(&[sample, feature_idx]));
            grad.set(&[sample, feature_idx], value);
        }
    }

    (
        SignedBankSoftmaxObjectiveReport {
            loss: loss_sum / batch_size as f32,
            top1: top1_count as f32 / batch_size as f32,
            mean_true_probability: true_probability_sum / batch_size as f32,
            samples: batch_size,
        },
        grad,
    )
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
        "signed objective distance expects matching shapes, got {:?} and {:?}",
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
