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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignedAngularRadialObjectiveConfig {
    pub angular_weight: f32,
    pub radial_weight: f32,
}

impl Default for SignedAngularRadialObjectiveConfig {
    fn default() -> Self {
        Self {
            angular_weight: 1.0,
            radial_weight: 1.0,
        }
    }
}

impl SignedAngularRadialObjectiveConfig {
    pub fn assert_valid(&self) {
        for (name, value) in [
            ("angular_weight", self.angular_weight),
            ("radial_weight", self.radial_weight),
        ] {
            assert!(
                value.is_finite() && value >= 0.0,
                "signed angular-radial {} must be finite and non-negative, got {}",
                name,
                value
            );
        }

        assert!(
            self.angular_weight > 0.0 || self.radial_weight > 0.0,
            "signed angular-radial objective requires at least one positive component weight"
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignedAngularRadialObjectiveReport {
    pub loss: f32,
    pub angular_loss: f32,
    pub radial_loss: f32,
    pub cosine: f32,
    pub prediction_norm: f32,
    pub target_norm: f32,
    pub norm_ratio: f32,
    pub samples: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignedRadialCalibrationReport {
    pub loss: f32,
    pub prediction_norm: f32,
    pub target_norm: f32,
    pub norm_ratio: f32,
    pub samples: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignedCenteredRadiusScalarObjectiveReport {
    pub loss: f32,
    pub prediction_radius: f32,
    pub target_radius: f32,
    pub radius_ratio: f32,
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

pub fn signed_radial_calibration_loss_and_grad(
    prediction: &Tensor,
    candidate_targets: &[Tensor],
    true_candidate_indices: &[usize],
) -> (SignedRadialCalibrationReport, Tensor) {
    assert!(
        prediction.ndim() == 2,
        "signed radial calibration prediction must be rank-2, got {:?}",
        prediction.shape
    );
    assert!(
        candidate_targets.len() >= 2,
        "signed radial calibration requires at least two candidates"
    );
    assert!(
        true_candidate_indices.len() == prediction.shape[0],
        "signed radial calibration true-index count {} must match batch {}",
        true_candidate_indices.len(),
        prediction.shape[0]
    );

    for target in candidate_targets {
        assert!(
            target.shape == prediction.shape,
            "signed radial calibration candidate target shape mismatch: prediction {:?}, candidate {:?}",
            prediction.shape,
            target.shape
        );
    }

    let batch_size = prediction.shape[0];
    let projection_dim = prediction.shape[1];
    assert!(
        projection_dim > 0,
        "signed radial calibration requires non-empty projection dim"
    );

    let mut grad = Tensor::zeros(prediction.shape.clone());
    let mut loss_sum = 0.0f32;
    let mut prediction_norm_sum = 0.0f32;
    let mut target_norm_sum = 0.0f32;
    let mut norm_ratio_sum = 0.0f32;

    for sample in 0..batch_size {
        let true_index = true_candidate_indices[sample];
        assert!(
            true_index < candidate_targets.len(),
            "signed radial calibration true index {} out of bounds for {} candidates",
            true_index,
            candidate_targets.len()
        );

        let centroid = candidate_target_centroid(candidate_targets, sample, projection_dim);
        let prediction_centered =
            centered_sample_features(prediction, sample, projection_dim, &centroid);
        let target_centered = centered_sample_features(
            &candidate_targets[true_index],
            sample,
            projection_dim,
            &centroid,
        );
        let prediction_norm = vector_norm(&prediction_centered);
        let target_norm = vector_norm(&target_centered);
        let norm_error = prediction_norm - target_norm;
        let loss = norm_error * norm_error;

        assert!(
            loss.is_finite() && prediction_norm.is_finite() && target_norm.is_finite(),
            "signed radial calibration produced non-finite metrics"
        );

        loss_sum += loss;
        prediction_norm_sum += prediction_norm;
        target_norm_sum += target_norm;
        norm_ratio_sum += prediction_norm / target_norm.max(RADIAL_EPSILON);

        if prediction_norm > RADIAL_EPSILON {
            let grad_scale = 2.0 * norm_error / (batch_size as f32 * prediction_norm);
            for feature_idx in 0..projection_dim {
                let value = grad.get(&[sample, feature_idx])
                    + grad_scale * prediction_centered[feature_idx];
                grad.set(&[sample, feature_idx], value);
            }
        }
    }

    (
        SignedRadialCalibrationReport {
            loss: loss_sum / batch_size as f32,
            prediction_norm: prediction_norm_sum / batch_size as f32,
            target_norm: target_norm_sum / batch_size as f32,
            norm_ratio: norm_ratio_sum / batch_size as f32,
            samples: batch_size,
        },
        grad,
    )
}

pub fn signed_candidate_centered_radius_targets(
    candidate_targets: &[Tensor],
    true_candidate_indices: &[usize],
) -> Tensor {
    let (batch_size, projection_dim) = validate_signed_candidate_bank(
        "signed candidate centered radius targets",
        candidate_targets,
        true_candidate_indices,
    );
    let mut targets = Vec::with_capacity(batch_size);

    for sample in 0..batch_size {
        let true_index = true_candidate_indices[sample];
        assert!(
            true_index < candidate_targets.len(),
            "signed candidate centered radius true index {} out of bounds for {} candidates",
            true_index,
            candidate_targets.len()
        );

        let centroid = candidate_target_centroid(candidate_targets, sample, projection_dim);
        let target_centered = centered_sample_features(
            &candidate_targets[true_index],
            sample,
            projection_dim,
            &centroid,
        );
        let target_radius = vector_norm(&target_centered);
        assert!(
            target_radius.is_finite(),
            "signed candidate centered radius produced non-finite target"
        );
        targets.push(target_radius);
    }

    Tensor::new(targets, vec![batch_size, 1])
}

pub fn signed_centered_radius_scalar_loss_and_grad(
    predicted_centered_radius: &Tensor,
    candidate_targets: &[Tensor],
    true_candidate_indices: &[usize],
) -> (SignedCenteredRadiusScalarObjectiveReport, Tensor) {
    let (batch_size, _) = validate_signed_candidate_bank(
        "signed centered radius scalar objective",
        candidate_targets,
        true_candidate_indices,
    );
    assert_scalar_prediction_shape(
        predicted_centered_radius,
        batch_size,
        "signed centered radius scalar objective",
    );

    let target_centered_radius =
        signed_candidate_centered_radius_targets(candidate_targets, true_candidate_indices);
    let mut grad = Tensor::zeros(predicted_centered_radius.shape.clone());
    let mut loss_sum = 0.0f32;
    let mut prediction_radius_sum = 0.0f32;
    let mut target_radius_sum = 0.0f32;
    let mut radius_ratio_sum = 0.0f32;

    for sample in 0..batch_size {
        let prediction_radius = scalar_prediction_value(predicted_centered_radius, sample);
        let target_radius = target_centered_radius.get(&[sample, 0]);
        assert!(
            prediction_radius.is_finite(),
            "signed centered radius scalar prediction must be finite"
        );

        let radius_error = prediction_radius - target_radius;
        let loss = radius_error * radius_error;
        assert!(
            loss.is_finite(),
            "signed centered radius scalar produced non-finite loss"
        );

        loss_sum += loss;
        prediction_radius_sum += prediction_radius;
        target_radius_sum += target_radius;
        radius_ratio_sum += prediction_radius / target_radius.max(RADIAL_EPSILON);

        set_scalar_prediction_value(&mut grad, sample, 2.0 * radius_error / batch_size as f32);
    }

    (
        SignedCenteredRadiusScalarObjectiveReport {
            loss: loss_sum / batch_size as f32,
            prediction_radius: prediction_radius_sum / batch_size as f32,
            target_radius: target_radius_sum / batch_size as f32,
            radius_ratio: radius_ratio_sum / batch_size as f32,
            samples: batch_size,
        },
        grad,
    )
}

pub fn signed_angular_radial_objective_loss_and_grad(
    prediction: &Tensor,
    candidate_targets: &[Tensor],
    true_candidate_indices: &[usize],
    config: SignedAngularRadialObjectiveConfig,
) -> (SignedAngularRadialObjectiveReport, Tensor) {
    config.assert_valid();
    assert!(
        prediction.ndim() == 2,
        "signed angular-radial prediction must be rank-2, got {:?}",
        prediction.shape
    );
    assert!(
        candidate_targets.len() >= 2,
        "signed angular-radial objective requires at least two candidates"
    );
    assert!(
        true_candidate_indices.len() == prediction.shape[0],
        "signed angular-radial true-index count {} must match batch {}",
        true_candidate_indices.len(),
        prediction.shape[0]
    );

    for target in candidate_targets {
        assert!(
            target.shape == prediction.shape,
            "signed angular-radial candidate target shape mismatch: prediction {:?}, candidate {:?}",
            prediction.shape,
            target.shape
        );
    }

    let batch_size = prediction.shape[0];
    let projection_dim = prediction.shape[1];
    assert!(
        batch_size > 0,
        "signed angular-radial objective requires a non-empty batch"
    );
    assert!(
        projection_dim > 0,
        "signed angular-radial objective requires non-empty projection dim"
    );

    let mut grad = Tensor::zeros(prediction.shape.clone());
    let mut angular_loss_sum = 0.0f32;
    let mut radial_loss_sum = 0.0f32;
    let mut cosine_sum = 0.0f32;
    let mut prediction_norm_sum = 0.0f32;
    let mut target_norm_sum = 0.0f32;
    let mut norm_ratio_sum = 0.0f32;

    for sample in 0..batch_size {
        let true_index = true_candidate_indices[sample];
        assert!(
            true_index < candidate_targets.len(),
            "signed angular-radial true index {} out of bounds for {} candidates",
            true_index,
            candidate_targets.len()
        );

        let centroid = candidate_target_centroid(candidate_targets, sample, projection_dim);
        let prediction_centered =
            centered_sample_features(prediction, sample, projection_dim, &centroid);
        let target_centered = centered_sample_features(
            &candidate_targets[true_index],
            sample,
            projection_dim,
            &centroid,
        );
        let prediction_norm = vector_norm(&prediction_centered);
        let target_norm = vector_norm(&target_centered);
        let norm_error = prediction_norm - target_norm;
        let radial_loss = norm_error * norm_error;
        let cosine = if prediction_norm > RADIAL_EPSILON && target_norm > RADIAL_EPSILON {
            (vector_dot(&prediction_centered, &target_centered) / (prediction_norm * target_norm))
                .clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let angular_loss = 1.0 - cosine;

        assert!(
            angular_loss.is_finite()
                && radial_loss.is_finite()
                && cosine.is_finite()
                && prediction_norm.is_finite()
                && target_norm.is_finite(),
            "signed angular-radial produced non-finite metrics"
        );

        angular_loss_sum += angular_loss;
        radial_loss_sum += radial_loss;
        cosine_sum += cosine;
        prediction_norm_sum += prediction_norm;
        target_norm_sum += target_norm;
        norm_ratio_sum += prediction_norm / target_norm.max(RADIAL_EPSILON);

        if prediction_norm > RADIAL_EPSILON {
            let radial_grad_scale =
                config.radial_weight * 2.0 * norm_error / (batch_size as f32 * prediction_norm);
            for feature_idx in 0..projection_dim {
                let value = grad.get(&[sample, feature_idx])
                    + radial_grad_scale * prediction_centered[feature_idx];
                grad.set(&[sample, feature_idx], value);
            }
        }

        if prediction_norm > RADIAL_EPSILON && target_norm > RADIAL_EPSILON {
            for feature_idx in 0..projection_dim {
                let angular_grad = -target_centered[feature_idx] / (prediction_norm * target_norm)
                    + cosine * prediction_centered[feature_idx]
                        / (prediction_norm * prediction_norm);
                let value = grad.get(&[sample, feature_idx])
                    + config.angular_weight * angular_grad / batch_size as f32;
                grad.set(&[sample, feature_idx], value);
            }
        }
    }

    let angular_loss = angular_loss_sum / batch_size as f32;
    let radial_loss = radial_loss_sum / batch_size as f32;
    (
        SignedAngularRadialObjectiveReport {
            loss: config.angular_weight * angular_loss + config.radial_weight * radial_loss,
            angular_loss,
            radial_loss,
            cosine: cosine_sum / batch_size as f32,
            prediction_norm: prediction_norm_sum / batch_size as f32,
            target_norm: target_norm_sum / batch_size as f32,
            norm_ratio: norm_ratio_sum / batch_size as f32,
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

const RADIAL_EPSILON: f32 = 1e-6;

fn validate_signed_candidate_bank(
    context: &str,
    candidate_targets: &[Tensor],
    true_candidate_indices: &[usize],
) -> (usize, usize) {
    assert!(
        candidate_targets.len() >= 2,
        "{} requires at least two candidates",
        context
    );
    assert!(
        candidate_targets[0].ndim() == 2,
        "{} candidate targets must be rank-2, got {:?}",
        context,
        candidate_targets[0].shape
    );

    let batch_size = candidate_targets[0].shape[0];
    let projection_dim = candidate_targets[0].shape[1];
    assert!(batch_size > 0, "{} requires a non-empty batch", context);
    assert!(
        projection_dim > 0,
        "{} requires non-empty projection dim",
        context
    );
    assert!(
        true_candidate_indices.len() == batch_size,
        "{} true-index count {} must match batch {}",
        context,
        true_candidate_indices.len(),
        batch_size
    );

    for target in candidate_targets {
        assert!(
            target.shape == candidate_targets[0].shape,
            "{} candidate target shape mismatch: first {:?}, candidate {:?}",
            context,
            candidate_targets[0].shape,
            target.shape
        );
    }

    (batch_size, projection_dim)
}

fn assert_scalar_prediction_shape(prediction: &Tensor, batch_size: usize, context: &str) {
    match prediction.ndim() {
        1 => assert!(
            prediction.shape[0] == batch_size,
            "{} prediction batch {} must match target batch {}",
            context,
            prediction.shape[0],
            batch_size
        ),
        2 => assert!(
            prediction.shape[0] == batch_size && prediction.shape[1] == 1,
            "{} prediction must have shape [batch] or [batch, 1], got {:?}",
            context,
            prediction.shape
        ),
        _ => panic!(
            "{} prediction must have shape [batch] or [batch, 1], got {:?}",
            context, prediction.shape
        ),
    }
}

fn scalar_prediction_value(prediction: &Tensor, sample: usize) -> f32 {
    match prediction.ndim() {
        1 => prediction.get(&[sample]),
        2 => prediction.get(&[sample, 0]),
        _ => unreachable!("scalar prediction rank validated before use"),
    }
}

fn set_scalar_prediction_value(tensor: &mut Tensor, sample: usize, value: f32) {
    match tensor.ndim() {
        1 => tensor.set(&[sample], value),
        2 => tensor.set(&[sample, 0], value),
        _ => unreachable!("scalar prediction rank validated before use"),
    }
}

fn candidate_target_centroid(
    candidate_targets: &[Tensor],
    sample: usize,
    projection_dim: usize,
) -> Vec<f32> {
    let mut centroid = vec![0.0f32; projection_dim];

    for target in candidate_targets {
        for (feature_idx, value) in centroid.iter_mut().enumerate() {
            *value += target.get(&[sample, feature_idx]);
        }
    }

    for value in &mut centroid {
        *value /= candidate_targets.len() as f32;
    }

    centroid
}

fn centered_sample_features(
    tensor: &Tensor,
    sample: usize,
    projection_dim: usize,
    centroid: &[f32],
) -> Vec<f32> {
    assert_eq!(
        centroid.len(),
        projection_dim,
        "signed radial calibration centroid dim mismatch"
    );

    (0..projection_dim)
        .map(|feature_idx| tensor.get(&[sample, feature_idx]) - centroid[feature_idx])
        .collect()
}

fn vector_norm(features: &[f32]) -> f32 {
    features
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
}

fn vector_dot(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(
        left.len(),
        right.len(),
        "signed angular-radial dot expects matching lengths"
    );

    left.iter()
        .zip(right.iter())
        .map(|(left_value, right_value)| left_value * right_value)
        .sum()
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
