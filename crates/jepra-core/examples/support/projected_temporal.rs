use super::temporal_vision::{
    BATCH_SIZE, SIGNED_VELOCITY_BANK_CANDIDATE_DX, TemporalTaskMode, VELOCITY_BANK_CANDIDATE_DX,
    make_signed_velocity_trail_candidate_target_batch, make_temporal_batch_for_task,
    make_velocity_trail_candidate_target_batch, signed_motion_dx_for_sample,
};
use jepra_core::{
    EmbeddingEncoder, Linear, PredictorModule, ProjectedVisionJepa,
    SignedAngularRadialObjectiveConfig, SignedAngularRadialObjectiveReport,
    SignedBankSoftmaxObjectiveConfig, SignedBankSoftmaxObjectiveReport,
    SignedMarginObjectiveConfig, SignedMarginObjectiveReport, SignedRadialCalibrationReport,
    Tensor, gaussian_moment_regularizer, mse_loss, signed_angular_radial_objective_loss_and_grad,
    signed_bank_softmax_objective_loss_and_grad, signed_margin_objective_loss_and_grad,
    signed_radial_calibration_loss_and_grad,
};

pub const PROJECTED_VALIDATION_BASE_SEED: u64 = 111_000;
pub const PROJECTED_VALIDATION_BATCHES: usize = 8;
pub const PROJECTED_TRAIN_LOSS_MAX_REDUCTION_RATIO: f32 = 0.2;
pub const PROJECTED_VALIDATION_LOSS_MAX_REDUCTION_RATIO: f32 = 0.2;
const VELOCITY_BANK_TIE_EPSILON: f32 = 1e-7;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedVelocityBankRanking {
    pub mrr: f32,
    pub top1: f32,
    pub mean_rank: f32,
    pub samples: usize,
    pub candidates: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedSignedStateSeparability {
    pub latent_mrr: f32,
    pub latent_top1: f32,
    pub latent_sign_top1: f32,
    pub latent_mean_rank: f32,
    pub projection_mrr: f32,
    pub projection_top1: f32,
    pub projection_sign_top1: f32,
    pub projection_mean_rank: f32,
    pub support_samples: usize,
    pub query_samples: usize,
    pub candidates: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedSignedVelocityBankBreakdown {
    pub negative_mrr: f32,
    pub positive_mrr: f32,
    pub slow_mrr: f32,
    pub fast_mrr: f32,
    pub sign_top1: f32,
    pub speed_top1: f32,
    pub samples: usize,
    pub negative_samples: usize,
    pub positive_samples: usize,
    pub slow_samples: usize,
    pub fast_samples: usize,
    pub true_neg_best_neg: usize,
    pub true_neg_best_pos: usize,
    pub true_pos_best_neg: usize,
    pub true_pos_best_pos: usize,
    pub true_slow_best_slow: usize,
    pub true_slow_best_fast: usize,
    pub true_fast_best_slow: usize,
    pub true_fast_best_fast: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedSignedTargetBankSeparability {
    pub oracle_mrr: f32,
    pub oracle_top1: f32,
    pub true_distance: f32,
    pub max_true_distance: f32,
    pub nearest_wrong_distance: f32,
    pub min_nearest_wrong_distance: f32,
    pub margin: f32,
    pub min_margin: f32,
    pub negative_nearest_wrong_distance: f32,
    pub positive_nearest_wrong_distance: f32,
    pub slow_nearest_wrong_distance: f32,
    pub fast_nearest_wrong_distance: f32,
    pub sign_margin: f32,
    pub speed_margin: f32,
    pub samples: usize,
    pub negative_samples: usize,
    pub positive_samples: usize,
    pub slow_samples: usize,
    pub fast_samples: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedSignedPredictionBankMargin {
    pub true_distance: f32,
    pub nearest_wrong_distance: f32,
    pub margin: f32,
    pub min_margin: f32,
    pub positive_margin_rate: f32,
    pub sign_margin: f32,
    pub speed_margin: f32,
    pub samples: usize,
    pub negative_samples: usize,
    pub positive_samples: usize,
    pub slow_samples: usize,
    pub fast_samples: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedSignedPredictionBankUnitGeometry {
    pub mrr: f32,
    pub top1: f32,
    pub true_distance: f32,
    pub nearest_wrong_distance: f32,
    pub margin: f32,
    pub positive_margin_rate: f32,
    pub sign_margin: f32,
    pub speed_margin: f32,
    pub prediction_center_norm: f32,
    pub true_target_center_norm: f32,
    pub samples: usize,
    pub candidates: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedSignedPredictionCounterfactualMetrics {
    pub mrr: f32,
    pub top1: f32,
    pub margin: f32,
    pub positive_margin_rate: f32,
    pub sign_margin: f32,
    pub speed_margin: f32,
    pub norm_ratio: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedSignedPredictionGeometryCounterfactual {
    pub oracle_radius: ProjectedSignedPredictionCounterfactualMetrics,
    pub oracle_angle: ProjectedSignedPredictionCounterfactualMetrics,
    pub support_global_rescale: ProjectedSignedPredictionCounterfactualMetrics,
    pub support_norm_ratio: f32,
    pub support_samples: usize,
    pub query_samples: usize,
    pub candidates: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedSignedObjectiveErrorBreakdown {
    pub all_loss: f32,
    pub dx_neg2_loss: f32,
    pub dx_neg1_loss: f32,
    pub dx_pos1_loss: f32,
    pub dx_pos2_loss: f32,
    pub neg_loss: f32,
    pub pos_loss: f32,
    pub slow_loss: f32,
    pub fast_loss: f32,
    pub sign_gap: f32,
    pub speed_gap: f32,
    pub samples: usize,
    pub dx_neg2_samples: usize,
    pub dx_neg1_samples: usize,
    pub dx_pos1_samples: usize,
    pub dx_pos2_samples: usize,
}

#[derive(Debug, Clone, Copy)]
struct VelocityBankSampleOutcome {
    true_dx: isize,
    best_dx: isize,
    rank: usize,
    sign_correct: bool,
    speed_correct: bool,
}

#[derive(Debug, Clone, Copy)]
struct SignedTargetBankSampleOutcome {
    true_dx: isize,
    true_distance: f32,
    nearest_wrong_distance: f32,
    rank: usize,
    sign_margin: f32,
    speed_margin: f32,
}

#[derive(Debug, Clone, Copy)]
struct SignedPredictionBankMarginSampleOutcome {
    true_dx: isize,
    true_distance: f32,
    nearest_wrong_distance: f32,
    sign_margin: f32,
    speed_margin: f32,
}

#[derive(Debug, Clone, Copy)]
struct SignedPredictionBankUnitGeometrySampleOutcome {
    true_dx: isize,
    true_distance: f32,
    nearest_wrong_distance: f32,
    rank: usize,
    sign_margin: f32,
    speed_margin: f32,
    prediction_center_norm: f32,
    true_target_center_norm: f32,
}

#[derive(Debug, Clone, Copy)]
struct SignedPredictionCounterfactualSampleOutcome {
    true_dx: isize,
    true_distance: f32,
    nearest_wrong_distance: f32,
    rank: usize,
    sign_margin: f32,
    speed_margin: f32,
    norm_ratio: f32,
}

#[derive(Debug, Clone, Copy)]
struct SignedPredictionGeometryCounterfactualSampleOutcome {
    oracle_radius: SignedPredictionCounterfactualSampleOutcome,
    oracle_angle: SignedPredictionCounterfactualSampleOutcome,
    support_global_rescale: SignedPredictionCounterfactualSampleOutcome,
}

#[derive(Debug, Clone, Copy)]
struct SignedStateSeparabilitySampleOutcome {
    true_dx: isize,
    best_dx: isize,
    rank: usize,
}

#[allow(dead_code)]
#[derive(Debug, Default)]
struct SignedObjectiveErrorBreakdownTotals {
    all_loss: f32,
    dx_neg2_loss: f32,
    dx_neg1_loss: f32,
    dx_pos1_loss: f32,
    dx_pos2_loss: f32,
    samples: usize,
    dx_neg2_samples: usize,
    dx_neg1_samples: usize,
    dx_pos1_samples: usize,
    dx_pos2_samples: usize,
}

#[derive(Debug, Default)]
struct SignedVelocityBankBreakdownTotals {
    negative_rr: f32,
    positive_rr: f32,
    slow_rr: f32,
    fast_rr: f32,
    sign_correct: usize,
    speed_correct: usize,
    samples: usize,
    negative_samples: usize,
    positive_samples: usize,
    slow_samples: usize,
    fast_samples: usize,
    true_neg_best_neg: usize,
    true_neg_best_pos: usize,
    true_pos_best_neg: usize,
    true_pos_best_pos: usize,
    true_slow_best_slow: usize,
    true_slow_best_fast: usize,
    true_fast_best_slow: usize,
    true_fast_best_fast: usize,
}

#[derive(Debug, Default)]
struct SignedTargetBankSeparabilityTotals {
    reciprocal_rank: f32,
    top1: usize,
    true_distance: f32,
    max_true_distance: f32,
    nearest_wrong_distance: f32,
    min_nearest_wrong_distance: f32,
    margin: f32,
    min_margin: f32,
    negative_nearest_wrong_distance: f32,
    positive_nearest_wrong_distance: f32,
    slow_nearest_wrong_distance: f32,
    fast_nearest_wrong_distance: f32,
    sign_margin: f32,
    speed_margin: f32,
    samples: usize,
    negative_samples: usize,
    positive_samples: usize,
    slow_samples: usize,
    fast_samples: usize,
}

#[derive(Debug, Default)]
struct SignedPredictionBankMarginTotals {
    true_distance: f32,
    nearest_wrong_distance: f32,
    margin: f32,
    min_margin: f32,
    positive_margin_count: usize,
    sign_margin: f32,
    speed_margin: f32,
    samples: usize,
    negative_samples: usize,
    positive_samples: usize,
    slow_samples: usize,
    fast_samples: usize,
}

#[derive(Debug, Default)]
struct SignedPredictionBankUnitGeometryTotals {
    reciprocal_rank: f32,
    top1: usize,
    true_distance: f32,
    nearest_wrong_distance: f32,
    margin: f32,
    positive_margin_count: usize,
    sign_margin: f32,
    speed_margin: f32,
    prediction_center_norm: f32,
    true_target_center_norm: f32,
    samples: usize,
    negative_samples: usize,
    positive_samples: usize,
    slow_samples: usize,
    fast_samples: usize,
}

#[derive(Debug, Default)]
struct SignedPredictionCounterfactualMetricTotals {
    reciprocal_rank: f32,
    top1: usize,
    margin: f32,
    positive_margin_count: usize,
    sign_margin: f32,
    speed_margin: f32,
    norm_ratio: f32,
    samples: usize,
    negative_samples: usize,
    positive_samples: usize,
    slow_samples: usize,
    fast_samples: usize,
}

#[derive(Debug)]
struct SignedStateCentroids {
    sums: Vec<Vec<f32>>,
    counts: Vec<usize>,
    dim: usize,
}

#[derive(Debug, Default)]
struct SignedStateSeparabilityTotals {
    reciprocal_rank: f32,
    top1: usize,
    sign_top1: usize,
    rank: usize,
    samples: usize,
}

#[allow(dead_code)]
impl SignedObjectiveErrorBreakdownTotals {
    fn observe(&mut self, true_dx: isize, loss: f32) {
        assert!(
            loss.is_finite(),
            "signed objective error loss is non-finite"
        );

        self.all_loss += loss;
        self.samples += 1;

        match true_dx {
            -2 => {
                self.dx_neg2_loss += loss;
                self.dx_neg2_samples += 1;
            }
            -1 => {
                self.dx_neg1_loss += loss;
                self.dx_neg1_samples += 1;
            }
            1 => {
                self.dx_pos1_loss += loss;
                self.dx_pos1_samples += 1;
            }
            2 => {
                self.dx_pos2_loss += loss;
                self.dx_pos2_samples += 1;
            }
            dx => panic!("unexpected signed objective error dx {}", dx),
        }
    }

    fn into_breakdown(self) -> ProjectedSignedObjectiveErrorBreakdown {
        assert!(self.samples > 0, "signed objective error has no samples");
        assert!(
            self.dx_neg2_samples > 0
                && self.dx_neg1_samples > 0
                && self.dx_pos1_samples > 0
                && self.dx_pos2_samples > 0,
            "signed objective error requires all signed dx buckets"
        );

        let dx_neg2_loss = self.dx_neg2_loss / self.dx_neg2_samples as f32;
        let dx_neg1_loss = self.dx_neg1_loss / self.dx_neg1_samples as f32;
        let dx_pos1_loss = self.dx_pos1_loss / self.dx_pos1_samples as f32;
        let dx_pos2_loss = self.dx_pos2_loss / self.dx_pos2_samples as f32;
        let neg_samples = self.dx_neg2_samples + self.dx_neg1_samples;
        let pos_samples = self.dx_pos1_samples + self.dx_pos2_samples;
        let slow_samples = self.dx_neg1_samples + self.dx_pos1_samples;
        let fast_samples = self.dx_neg2_samples + self.dx_pos2_samples;
        let neg_loss = (self.dx_neg2_loss + self.dx_neg1_loss) / neg_samples as f32;
        let pos_loss = (self.dx_pos1_loss + self.dx_pos2_loss) / pos_samples as f32;
        let slow_loss = (self.dx_neg1_loss + self.dx_pos1_loss) / slow_samples as f32;
        let fast_loss = (self.dx_neg2_loss + self.dx_pos2_loss) / fast_samples as f32;

        ProjectedSignedObjectiveErrorBreakdown {
            all_loss: self.all_loss / self.samples as f32,
            dx_neg2_loss,
            dx_neg1_loss,
            dx_pos1_loss,
            dx_pos2_loss,
            neg_loss,
            pos_loss,
            slow_loss,
            fast_loss,
            sign_gap: pos_loss - neg_loss,
            speed_gap: fast_loss - slow_loss,
            samples: self.samples,
            dx_neg2_samples: self.dx_neg2_samples,
            dx_neg1_samples: self.dx_neg1_samples,
            dx_pos1_samples: self.dx_pos1_samples,
            dx_pos2_samples: self.dx_pos2_samples,
        }
    }
}

impl SignedStateCentroids {
    fn new(dim: usize) -> Self {
        assert!(dim > 0, "signed state centroid dim must be non-empty");

        Self {
            sums: vec![vec![0.0; dim]; SIGNED_VELOCITY_BANK_CANDIDATE_DX.len()],
            counts: vec![0; SIGNED_VELOCITY_BANK_CANDIDATE_DX.len()],
            dim,
        }
    }

    fn observe(&mut self, features: &Tensor, sample: usize, true_dx: isize) {
        assert!(
            features.shape.len() == 2,
            "signed state centroids expect rank-2 features, got {:?}",
            features.shape
        );
        assert_eq!(
            features.shape[1], self.dim,
            "signed state centroid dim mismatch"
        );

        let class_index = signed_velocity_candidate_index(true_dx);
        self.counts[class_index] += 1;

        for feature_idx in 0..self.dim {
            self.sums[class_index][feature_idx] += features.get(&[sample, feature_idx]);
        }
    }

    fn finalize(mut self) -> Self {
        self.assert_all_classes_present("support");

        for class_index in 0..self.sums.len() {
            let count = self.counts[class_index] as f32;
            for feature in &mut self.sums[class_index] {
                *feature /= count;
            }
        }

        self
    }

    fn distance_to_class(&self, features: &Tensor, sample: usize, class_index: usize) -> f32 {
        assert!(
            features.shape.len() == 2,
            "signed state separability expects rank-2 features, got {:?}",
            features.shape
        );
        assert_eq!(
            features.shape[1], self.dim,
            "signed state separability dim mismatch"
        );

        let mut distance = 0.0f32;

        for feature_idx in 0..self.dim {
            let diff = features.get(&[sample, feature_idx]) - self.sums[class_index][feature_idx];
            distance += diff * diff;
        }

        distance
    }

    fn sample_outcome(
        &self,
        features: &Tensor,
        sample: usize,
        true_dx: isize,
    ) -> SignedStateSeparabilitySampleOutcome {
        let true_index = signed_velocity_candidate_index(true_dx);
        let distances = (0..SIGNED_VELOCITY_BANK_CANDIDATE_DX.len())
            .map(|class_index| self.distance_to_class(features, sample, class_index))
            .collect::<Vec<_>>();
        let true_distance = distances[true_index];
        let mut rank = 1usize;
        let mut best_index = 0usize;
        let mut best_distance = distances[0];

        for (class_index, distance) in distances.iter().enumerate() {
            if class_index != true_index && *distance <= true_distance + VELOCITY_BANK_TIE_EPSILON {
                rank += 1;
            }
            if *distance < best_distance - VELOCITY_BANK_TIE_EPSILON {
                best_index = class_index;
                best_distance = *distance;
            }
        }

        SignedStateSeparabilitySampleOutcome {
            true_dx,
            best_dx: SIGNED_VELOCITY_BANK_CANDIDATE_DX[best_index],
            rank,
        }
    }

    fn assert_all_classes_present(&self, label: &str) {
        for (class_index, count) in self.counts.iter().enumerate() {
            assert!(
                *count > 0,
                "{} signed state separability is missing dx {}",
                label,
                SIGNED_VELOCITY_BANK_CANDIDATE_DX[class_index]
            );
        }
    }
}

impl SignedStateSeparabilityTotals {
    fn observe(&mut self, outcome: SignedStateSeparabilitySampleOutcome) {
        assert!(outcome.rank > 0, "signed state rank must be positive");

        self.samples += 1;
        self.reciprocal_rank += 1.0 / outcome.rank as f32;
        self.top1 += usize::from(outcome.rank == 1);
        self.sign_top1 += usize::from(outcome.best_dx.signum() == outcome.true_dx.signum());
        self.rank += outcome.rank;
    }

    fn mrr(&self) -> f32 {
        assert!(self.samples > 0, "signed state separability has no samples");
        self.reciprocal_rank / self.samples as f32
    }

    fn top1(&self) -> f32 {
        assert!(self.samples > 0, "signed state separability has no samples");
        self.top1 as f32 / self.samples as f32
    }

    fn sign_top1(&self) -> f32 {
        assert!(self.samples > 0, "signed state separability has no samples");
        self.sign_top1 as f32 / self.samples as f32
    }

    fn mean_rank(&self) -> f32 {
        assert!(self.samples > 0, "signed state separability has no samples");
        self.rank as f32 / self.samples as f32
    }
}

impl SignedVelocityBankBreakdownTotals {
    fn observe(&mut self, outcome: VelocityBankSampleOutcome) {
        let reciprocal_rank = 1.0 / outcome.rank as f32;
        self.samples += 1;
        self.sign_correct += usize::from(outcome.sign_correct);
        self.speed_correct += usize::from(outcome.speed_correct);

        if outcome.true_dx < 0 {
            self.negative_rr += reciprocal_rank;
            self.negative_samples += 1;
            if outcome.best_dx < 0 {
                self.true_neg_best_neg += 1;
            } else {
                self.true_neg_best_pos += 1;
            }
        } else {
            self.positive_rr += reciprocal_rank;
            self.positive_samples += 1;
            if outcome.best_dx < 0 {
                self.true_pos_best_neg += 1;
            } else {
                self.true_pos_best_pos += 1;
            }
        }

        if outcome.true_dx.abs() == 1 {
            self.slow_rr += reciprocal_rank;
            self.slow_samples += 1;
            if outcome.best_dx.abs() == 1 {
                self.true_slow_best_slow += 1;
            } else {
                self.true_slow_best_fast += 1;
            }
        } else {
            self.fast_rr += reciprocal_rank;
            self.fast_samples += 1;
            if outcome.best_dx.abs() == 1 {
                self.true_fast_best_slow += 1;
            } else {
                self.true_fast_best_fast += 1;
            }
        }
    }

    fn into_breakdown(self) -> ProjectedSignedVelocityBankBreakdown {
        assert!(self.samples > 0, "signed velocity breakdown has no samples");
        assert!(
            self.negative_samples > 0
                && self.positive_samples > 0
                && self.slow_samples > 0
                && self.fast_samples > 0,
            "signed velocity breakdown requires all sign/speed groups"
        );

        ProjectedSignedVelocityBankBreakdown {
            negative_mrr: self.negative_rr / self.negative_samples as f32,
            positive_mrr: self.positive_rr / self.positive_samples as f32,
            slow_mrr: self.slow_rr / self.slow_samples as f32,
            fast_mrr: self.fast_rr / self.fast_samples as f32,
            sign_top1: self.sign_correct as f32 / self.samples as f32,
            speed_top1: self.speed_correct as f32 / self.samples as f32,
            samples: self.samples,
            negative_samples: self.negative_samples,
            positive_samples: self.positive_samples,
            slow_samples: self.slow_samples,
            fast_samples: self.fast_samples,
            true_neg_best_neg: self.true_neg_best_neg,
            true_neg_best_pos: self.true_neg_best_pos,
            true_pos_best_neg: self.true_pos_best_neg,
            true_pos_best_pos: self.true_pos_best_pos,
            true_slow_best_slow: self.true_slow_best_slow,
            true_slow_best_fast: self.true_slow_best_fast,
            true_fast_best_slow: self.true_fast_best_slow,
            true_fast_best_fast: self.true_fast_best_fast,
        }
    }
}

impl SignedTargetBankSeparabilityTotals {
    fn observe(&mut self, outcome: SignedTargetBankSampleOutcome) {
        let margin = outcome.nearest_wrong_distance - outcome.true_distance;
        assert!(
            outcome.true_distance.is_finite()
                && margin.is_finite()
                && outcome.rank > 0
                && outcome.nearest_wrong_distance.is_finite()
                && outcome.sign_margin.is_finite()
                && outcome.speed_margin.is_finite(),
            "signed target-bank separability produced non-finite metrics"
        );

        if self.samples == 0 {
            self.min_nearest_wrong_distance = outcome.nearest_wrong_distance;
            self.min_margin = margin;
        } else {
            self.min_nearest_wrong_distance = self
                .min_nearest_wrong_distance
                .min(outcome.nearest_wrong_distance);
            self.min_margin = self.min_margin.min(margin);
        }

        self.samples += 1;
        self.reciprocal_rank += 1.0 / outcome.rank as f32;
        self.top1 += usize::from(outcome.rank == 1);
        self.true_distance += outcome.true_distance;
        self.max_true_distance = self.max_true_distance.max(outcome.true_distance);
        self.nearest_wrong_distance += outcome.nearest_wrong_distance;
        self.margin += margin;
        self.sign_margin += outcome.sign_margin;
        self.speed_margin += outcome.speed_margin;

        if outcome.true_dx < 0 {
            self.negative_nearest_wrong_distance += outcome.nearest_wrong_distance;
            self.negative_samples += 1;
        } else {
            self.positive_nearest_wrong_distance += outcome.nearest_wrong_distance;
            self.positive_samples += 1;
        }

        if outcome.true_dx.abs() == 1 {
            self.slow_nearest_wrong_distance += outcome.nearest_wrong_distance;
            self.slow_samples += 1;
        } else {
            self.fast_nearest_wrong_distance += outcome.nearest_wrong_distance;
            self.fast_samples += 1;
        }
    }

    fn into_separability(self) -> ProjectedSignedTargetBankSeparability {
        assert!(
            self.samples > 0,
            "signed target-bank separability has no samples"
        );
        assert!(
            self.negative_samples > 0
                && self.positive_samples > 0
                && self.slow_samples > 0
                && self.fast_samples > 0,
            "signed target-bank separability requires all sign/speed groups"
        );

        ProjectedSignedTargetBankSeparability {
            oracle_mrr: self.reciprocal_rank / self.samples as f32,
            oracle_top1: self.top1 as f32 / self.samples as f32,
            true_distance: self.true_distance / self.samples as f32,
            max_true_distance: self.max_true_distance,
            nearest_wrong_distance: self.nearest_wrong_distance / self.samples as f32,
            min_nearest_wrong_distance: self.min_nearest_wrong_distance,
            margin: self.margin / self.samples as f32,
            min_margin: self.min_margin,
            negative_nearest_wrong_distance: self.negative_nearest_wrong_distance
                / self.negative_samples as f32,
            positive_nearest_wrong_distance: self.positive_nearest_wrong_distance
                / self.positive_samples as f32,
            slow_nearest_wrong_distance: self.slow_nearest_wrong_distance
                / self.slow_samples as f32,
            fast_nearest_wrong_distance: self.fast_nearest_wrong_distance
                / self.fast_samples as f32,
            sign_margin: self.sign_margin / self.samples as f32,
            speed_margin: self.speed_margin / self.samples as f32,
            samples: self.samples,
            negative_samples: self.negative_samples,
            positive_samples: self.positive_samples,
            slow_samples: self.slow_samples,
            fast_samples: self.fast_samples,
        }
    }
}

impl SignedPredictionBankMarginTotals {
    fn observe(&mut self, outcome: SignedPredictionBankMarginSampleOutcome) {
        let margin = outcome.nearest_wrong_distance - outcome.true_distance;
        assert!(
            outcome.true_distance.is_finite()
                && outcome.nearest_wrong_distance.is_finite()
                && margin.is_finite()
                && outcome.sign_margin.is_finite()
                && outcome.speed_margin.is_finite(),
            "signed prediction-bank margin produced non-finite metrics"
        );

        if self.samples == 0 {
            self.min_margin = margin;
        } else {
            self.min_margin = self.min_margin.min(margin);
        }

        self.samples += 1;
        self.true_distance += outcome.true_distance;
        self.nearest_wrong_distance += outcome.nearest_wrong_distance;
        self.margin += margin;
        self.positive_margin_count += usize::from(margin > VELOCITY_BANK_TIE_EPSILON);
        self.sign_margin += outcome.sign_margin;
        self.speed_margin += outcome.speed_margin;

        if outcome.true_dx < 0 {
            self.negative_samples += 1;
        } else {
            self.positive_samples += 1;
        }

        if outcome.true_dx.abs() == 1 {
            self.slow_samples += 1;
        } else {
            self.fast_samples += 1;
        }
    }

    fn into_margin(self) -> ProjectedSignedPredictionBankMargin {
        assert!(
            self.samples > 0,
            "signed prediction-bank margin has no samples"
        );
        assert!(
            self.negative_samples > 0
                && self.positive_samples > 0
                && self.slow_samples > 0
                && self.fast_samples > 0,
            "signed prediction-bank margin requires all sign/speed groups"
        );

        ProjectedSignedPredictionBankMargin {
            true_distance: self.true_distance / self.samples as f32,
            nearest_wrong_distance: self.nearest_wrong_distance / self.samples as f32,
            margin: self.margin / self.samples as f32,
            min_margin: self.min_margin,
            positive_margin_rate: self.positive_margin_count as f32 / self.samples as f32,
            sign_margin: self.sign_margin / self.samples as f32,
            speed_margin: self.speed_margin / self.samples as f32,
            samples: self.samples,
            negative_samples: self.negative_samples,
            positive_samples: self.positive_samples,
            slow_samples: self.slow_samples,
            fast_samples: self.fast_samples,
        }
    }
}

impl SignedPredictionBankUnitGeometryTotals {
    fn observe(&mut self, outcome: SignedPredictionBankUnitGeometrySampleOutcome) {
        let margin = outcome.nearest_wrong_distance - outcome.true_distance;
        assert!(
            outcome.true_distance.is_finite()
                && outcome.nearest_wrong_distance.is_finite()
                && margin.is_finite()
                && outcome.rank > 0
                && outcome.sign_margin.is_finite()
                && outcome.speed_margin.is_finite()
                && outcome.prediction_center_norm.is_finite()
                && outcome.true_target_center_norm.is_finite(),
            "signed prediction-bank unit geometry produced non-finite metrics"
        );

        self.samples += 1;
        self.reciprocal_rank += 1.0 / outcome.rank as f32;
        self.top1 += usize::from(outcome.rank == 1);
        self.true_distance += outcome.true_distance;
        self.nearest_wrong_distance += outcome.nearest_wrong_distance;
        self.margin += margin;
        self.positive_margin_count += usize::from(margin > VELOCITY_BANK_TIE_EPSILON);
        self.sign_margin += outcome.sign_margin;
        self.speed_margin += outcome.speed_margin;
        self.prediction_center_norm += outcome.prediction_center_norm;
        self.true_target_center_norm += outcome.true_target_center_norm;

        if outcome.true_dx < 0 {
            self.negative_samples += 1;
        } else {
            self.positive_samples += 1;
        }

        if outcome.true_dx.abs() == 1 {
            self.slow_samples += 1;
        } else {
            self.fast_samples += 1;
        }
    }

    fn into_geometry(self) -> ProjectedSignedPredictionBankUnitGeometry {
        assert!(
            self.samples > 0,
            "signed prediction-bank unit geometry has no samples"
        );
        assert!(
            self.negative_samples > 0
                && self.positive_samples > 0
                && self.slow_samples > 0
                && self.fast_samples > 0,
            "signed prediction-bank unit geometry requires all sign/speed groups"
        );

        ProjectedSignedPredictionBankUnitGeometry {
            mrr: self.reciprocal_rank / self.samples as f32,
            top1: self.top1 as f32 / self.samples as f32,
            true_distance: self.true_distance / self.samples as f32,
            nearest_wrong_distance: self.nearest_wrong_distance / self.samples as f32,
            margin: self.margin / self.samples as f32,
            positive_margin_rate: self.positive_margin_count as f32 / self.samples as f32,
            sign_margin: self.sign_margin / self.samples as f32,
            speed_margin: self.speed_margin / self.samples as f32,
            prediction_center_norm: self.prediction_center_norm / self.samples as f32,
            true_target_center_norm: self.true_target_center_norm / self.samples as f32,
            samples: self.samples,
            candidates: SIGNED_VELOCITY_BANK_CANDIDATE_DX.len(),
        }
    }
}

impl SignedPredictionCounterfactualMetricTotals {
    fn observe(&mut self, outcome: SignedPredictionCounterfactualSampleOutcome) {
        let margin = outcome.nearest_wrong_distance - outcome.true_distance;
        assert!(
            outcome.true_distance.is_finite()
                && outcome.nearest_wrong_distance.is_finite()
                && margin.is_finite()
                && outcome.rank > 0
                && outcome.sign_margin.is_finite()
                && outcome.speed_margin.is_finite()
                && outcome.norm_ratio.is_finite(),
            "signed prediction counterfactual produced non-finite metrics"
        );

        self.samples += 1;
        self.reciprocal_rank += 1.0 / outcome.rank as f32;
        self.top1 += usize::from(outcome.rank == 1);
        self.margin += margin;
        self.positive_margin_count += usize::from(margin > VELOCITY_BANK_TIE_EPSILON);
        self.sign_margin += outcome.sign_margin;
        self.speed_margin += outcome.speed_margin;
        self.norm_ratio += outcome.norm_ratio;

        if outcome.true_dx < 0 {
            self.negative_samples += 1;
        } else {
            self.positive_samples += 1;
        }

        if outcome.true_dx.abs() == 1 {
            self.slow_samples += 1;
        } else {
            self.fast_samples += 1;
        }
    }

    fn into_metrics(self) -> ProjectedSignedPredictionCounterfactualMetrics {
        assert!(
            self.samples > 0,
            "signed prediction counterfactual has no samples"
        );
        assert!(
            self.negative_samples > 0
                && self.positive_samples > 0
                && self.slow_samples > 0
                && self.fast_samples > 0,
            "signed prediction counterfactual requires all sign/speed groups"
        );

        ProjectedSignedPredictionCounterfactualMetrics {
            mrr: self.reciprocal_rank / self.samples as f32,
            top1: self.top1 as f32 / self.samples as f32,
            margin: self.margin / self.samples as f32,
            positive_margin_rate: self.positive_margin_count as f32 / self.samples as f32,
            sign_margin: self.sign_margin / self.samples as f32,
            speed_margin: self.speed_margin / self.samples as f32,
            norm_ratio: self.norm_ratio / self.samples as f32,
        }
    }
}

#[allow(dead_code)]
pub fn projected_target(
    encoder: &EmbeddingEncoder,
    target_projector: &Linear,
    x_t1: &Tensor,
) -> Tensor {
    let z_t1 = encoder.forward(x_t1);
    target_projector.forward(&z_t1)
}

#[allow(dead_code)]
pub fn projected_batch_losses<P>(
    encoder: &EmbeddingEncoder,
    online_projector: &Linear,
    target_projector: &Linear,
    predictor: &P,
    x_t: &Tensor,
    x_t1: &Tensor,
    regularizer_weight: f32,
) -> (f32, f32, f32)
where
    P: PredictorModule,
{
    let z_t = encoder.forward(x_t);
    let projection_t = online_projector.forward(&z_t);
    let pred = predictor.forward(&projection_t);
    let target = projected_target(encoder, target_projector, x_t1);
    let prediction_loss = mse_loss(&pred, &target);
    let regularizer_loss = gaussian_moment_regularizer(&projection_t);
    let total_loss = prediction_loss + regularizer_weight * regularizer_loss;

    (prediction_loss, regularizer_loss, total_loss)
}

#[allow(dead_code)]
pub fn projected_step(
    model: &mut ProjectedVisionJepa,
    x_t: &Tensor,
    x_t1: &Tensor,
    regularizer_weight: f32,
    predictor_lr: f32,
    projector_lr: f32,
) {
    model.step(x_t, x_t1, regularizer_weight, predictor_lr, projector_lr);
}

#[allow(clippy::too_many_arguments)]
pub fn projected_validation_batch_losses<P, F>(
    encoder: &EmbeddingEncoder,
    online_projector: &Linear,
    target_projector: &Linear,
    predictor: &P,
    regularizer_weight: f32,
    base_seed: u64,
    validation_batches: usize,
    make_batch: F,
) -> (f32, f32, f32)
where
    P: PredictorModule,
    F: Fn(u64) -> (Tensor, Tensor),
{
    assert!(
        validation_batches > 0,
        "validation_batches must be greater than 0"
    );

    let mut prediction_total = 0.0;
    let mut regularizer_total = 0.0;
    let mut total = 0.0;

    for batch_idx in 0..validation_batches {
        let (x_t, x_t1) = make_batch(base_seed + batch_idx as u64);
        let (prediction_loss, regularizer_loss, total_loss) = projected_batch_losses(
            encoder,
            online_projector,
            target_projector,
            predictor,
            &x_t,
            &x_t1,
            regularizer_weight,
        );
        prediction_total += prediction_loss;
        regularizer_total += regularizer_loss;
        total += total_loss;
    }

    let batches = validation_batches as f32;
    (
        prediction_total / batches,
        regularizer_total / batches,
        total / batches,
    )
}

#[allow(dead_code)]
pub fn projected_validation_batch_losses_from_base_seed<P>(
    model: &ProjectedVisionJepa<P>,
    regularizer_weight: f32,
    validation_base_seed: u64,
    validation_batches: usize,
) -> (f32, f32, f32)
where
    P: PredictorModule,
{
    projected_validation_batch_losses_from_base_seed_for_task(
        model,
        regularizer_weight,
        validation_base_seed,
        validation_batches,
        TemporalTaskMode::RandomSpeed,
    )
}

pub fn projected_validation_batch_losses_from_base_seed_for_task<P>(
    model: &ProjectedVisionJepa<P>,
    regularizer_weight: f32,
    validation_base_seed: u64,
    validation_batches: usize,
    temporal_task_mode: TemporalTaskMode,
) -> (f32, f32, f32)
where
    P: PredictorModule,
{
    projected_validation_batch_losses(
        &model.encoder,
        &model.projector,
        &model.target_projector,
        &model.predictor,
        regularizer_weight,
        validation_base_seed,
        validation_batches,
        |seed| make_temporal_batch_for_task(BATCH_SIZE, seed, temporal_task_mode),
    )
}

pub fn projected_signed_state_separability_from_base_seed<P>(
    model: &ProjectedVisionJepa<P>,
    validation_base_seed: u64,
    validation_batches: usize,
) -> ProjectedSignedStateSeparability
where
    P: PredictorModule,
{
    assert!(
        validation_batches >= 2,
        "signed state separability requires at least one support and one query batch"
    );

    let support_batches = validation_batches / 2;
    let (support_x_t, support_x_t1) = make_temporal_batch_for_task(
        BATCH_SIZE,
        validation_base_seed,
        TemporalTaskMode::SignedVelocityTrail,
    );
    let support_latent = model.encode(&support_x_t);
    let support_projection = model.project_latent(&support_x_t);
    let mut latent_centroids = SignedStateCentroids::new(support_latent.shape[1]);
    let mut projection_centroids = SignedStateCentroids::new(support_projection.shape[1]);
    let mut support_samples = 0usize;

    observe_signed_state_centroids(
        &support_x_t,
        &support_x_t1,
        &support_latent,
        &support_projection,
        &mut latent_centroids,
        &mut projection_centroids,
        &mut support_samples,
    );

    for batch_idx in 1..support_batches {
        let (x_t, x_t1) = make_temporal_batch_for_task(
            BATCH_SIZE,
            validation_base_seed + batch_idx as u64,
            TemporalTaskMode::SignedVelocityTrail,
        );
        let latent = model.encode(&x_t);
        let projection = model.project_latent(&x_t);

        observe_signed_state_centroids(
            &x_t,
            &x_t1,
            &latent,
            &projection,
            &mut latent_centroids,
            &mut projection_centroids,
            &mut support_samples,
        );
    }

    let latent_centroids = latent_centroids.finalize();
    let projection_centroids = projection_centroids.finalize();
    let mut latent_totals = SignedStateSeparabilityTotals::default();
    let mut projection_totals = SignedStateSeparabilityTotals::default();
    let mut query_counts = vec![0usize; SIGNED_VELOCITY_BANK_CANDIDATE_DX.len()];

    for batch_idx in support_batches..validation_batches {
        let (x_t, x_t1) = make_temporal_batch_for_task(
            BATCH_SIZE,
            validation_base_seed + batch_idx as u64,
            TemporalTaskMode::SignedVelocityTrail,
        );
        let latent = model.encode(&x_t);
        let projection = model.project_latent(&x_t);

        observe_signed_state_query(
            &x_t,
            &x_t1,
            &latent,
            &projection,
            &latent_centroids,
            &projection_centroids,
            &mut latent_totals,
            &mut projection_totals,
            &mut query_counts,
        );
    }

    assert_all_signed_state_query_classes_present(&query_counts);
    assert_eq!(
        latent_totals.samples, projection_totals.samples,
        "latent/projection query sample mismatch"
    );

    ProjectedSignedStateSeparability {
        latent_mrr: latent_totals.mrr(),
        latent_top1: latent_totals.top1(),
        latent_sign_top1: latent_totals.sign_top1(),
        latent_mean_rank: latent_totals.mean_rank(),
        projection_mrr: projection_totals.mrr(),
        projection_top1: projection_totals.top1(),
        projection_sign_top1: projection_totals.sign_top1(),
        projection_mean_rank: projection_totals.mean_rank(),
        support_samples,
        query_samples: latent_totals.samples,
        candidates: SIGNED_VELOCITY_BANK_CANDIDATE_DX.len(),
    }
}

#[allow(dead_code)]
pub fn projected_velocity_bank_ranking<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
) -> ProjectedVelocityBankRanking
where
    P: PredictorModule,
{
    projected_velocity_bank_ranking_for_task(model, x_t, x_t1, TemporalTaskMode::VelocityTrail)
}

pub fn projected_velocity_bank_ranking_for_task<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
    temporal_task_mode: TemporalTaskMode,
) -> ProjectedVelocityBankRanking
where
    P: PredictorModule,
{
    assert_eq!(
        x_t.shape, x_t1.shape,
        "velocity-bank ranking expects matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "velocity-bank ranking expects rank-4 temporal batches, got {:?}",
        x_t.shape
    );

    let candidate_dx_bank = velocity_bank_candidates_for_task(temporal_task_mode);
    let prediction = model.predict_next_projection(x_t);
    let candidate_targets = candidate_dx_bank
        .iter()
        .map(|candidate_dx| {
            let candidate_x_t1 =
                make_velocity_bank_candidate_target_batch(x_t, *candidate_dx, temporal_task_mode);
            model.target_projection(&candidate_x_t1)
        })
        .collect::<Vec<_>>();
    let batch_size = x_t.shape[0];
    let mut reciprocal_rank_total = 0.0f32;
    let mut top1_total = 0usize;
    let mut rank_total = 0usize;

    for sample in 0..batch_size {
        let true_dx = signed_motion_dx_for_sample(x_t, x_t1, sample);
        let true_index = candidate_dx_bank
            .iter()
            .position(|candidate_dx| *candidate_dx == true_dx)
            .unwrap_or_else(|| panic!("true dx {} is missing from velocity bank", true_dx));
        let distances = candidate_targets
            .iter()
            .map(|candidate_target| sample_squared_distance(&prediction, candidate_target, sample))
            .collect::<Vec<_>>();
        let true_distance = distances[true_index];
        let mut rank = 1usize;

        for (candidate_index, distance) in distances.iter().enumerate() {
            if candidate_index != true_index
                && *distance <= true_distance + VELOCITY_BANK_TIE_EPSILON
            {
                rank += 1;
            }
        }

        if rank == 1 {
            top1_total += 1;
        }
        rank_total += rank;
        reciprocal_rank_total += 1.0 / rank as f32;
    }

    ProjectedVelocityBankRanking {
        mrr: reciprocal_rank_total / batch_size as f32,
        top1: top1_total as f32 / batch_size as f32,
        mean_rank: rank_total as f32 / batch_size as f32,
        samples: batch_size,
        candidates: candidate_dx_bank.len(),
    }
}

pub fn projected_velocity_bank_ranking_from_base_seed<P>(
    model: &ProjectedVisionJepa<P>,
    validation_base_seed: u64,
    validation_batches: usize,
    temporal_task_mode: TemporalTaskMode,
) -> ProjectedVelocityBankRanking
where
    P: PredictorModule,
{
    assert!(
        matches!(
            temporal_task_mode,
            TemporalTaskMode::VelocityTrail | TemporalTaskMode::SignedVelocityTrail
        ),
        "velocity-bank ranking only supports velocity-trail or signed-velocity-trail task"
    );
    assert!(
        validation_batches > 0,
        "validation_batches must be greater than 0"
    );

    let mut reciprocal_rank_total = 0.0f32;
    let mut top1_total = 0.0f32;
    let mut rank_total = 0.0f32;
    let mut sample_total = 0usize;
    let mut candidate_count = velocity_bank_candidates_for_task(temporal_task_mode).len();

    for batch_idx in 0..validation_batches {
        let (x_t, x_t1) = make_temporal_batch_for_task(
            BATCH_SIZE,
            validation_base_seed + batch_idx as u64,
            temporal_task_mode,
        );
        let ranking =
            projected_velocity_bank_ranking_for_task(model, &x_t, &x_t1, temporal_task_mode);

        reciprocal_rank_total += ranking.mrr * ranking.samples as f32;
        top1_total += ranking.top1 * ranking.samples as f32;
        rank_total += ranking.mean_rank * ranking.samples as f32;
        sample_total += ranking.samples;
        candidate_count = ranking.candidates;
    }

    ProjectedVelocityBankRanking {
        mrr: reciprocal_rank_total / sample_total as f32,
        top1: top1_total / sample_total as f32,
        mean_rank: rank_total / sample_total as f32,
        samples: sample_total,
        candidates: candidate_count,
    }
}

#[allow(dead_code)]
pub fn projected_signed_velocity_bank_breakdown<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
) -> ProjectedSignedVelocityBankBreakdown
where
    P: PredictorModule,
{
    let outcomes = projected_velocity_bank_sample_outcomes(
        model,
        x_t,
        x_t1,
        TemporalTaskMode::SignedVelocityTrail,
    );
    let mut totals = SignedVelocityBankBreakdownTotals::default();

    for outcome in outcomes {
        totals.observe(outcome);
    }

    totals.into_breakdown()
}

pub fn projected_signed_velocity_bank_breakdown_from_base_seed<P>(
    model: &ProjectedVisionJepa<P>,
    validation_base_seed: u64,
    validation_batches: usize,
) -> ProjectedSignedVelocityBankBreakdown
where
    P: PredictorModule,
{
    assert!(
        validation_batches > 0,
        "validation_batches must be greater than 0"
    );

    let mut totals = SignedVelocityBankBreakdownTotals::default();

    for batch_idx in 0..validation_batches {
        let (x_t, x_t1) = make_temporal_batch_for_task(
            BATCH_SIZE,
            validation_base_seed + batch_idx as u64,
            TemporalTaskMode::SignedVelocityTrail,
        );
        let outcomes = projected_velocity_bank_sample_outcomes(
            model,
            &x_t,
            &x_t1,
            TemporalTaskMode::SignedVelocityTrail,
        );

        for outcome in outcomes {
            totals.observe(outcome);
        }
    }

    totals.into_breakdown()
}

#[allow(dead_code)]
pub fn projected_signed_target_bank_separability<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
) -> ProjectedSignedTargetBankSeparability
where
    P: PredictorModule,
{
    let outcomes = projected_signed_target_bank_sample_outcomes(model, x_t, x_t1);
    let mut totals = SignedTargetBankSeparabilityTotals::default();

    for outcome in outcomes {
        totals.observe(outcome);
    }

    totals.into_separability()
}

pub fn projected_signed_target_bank_separability_from_base_seed<P>(
    model: &ProjectedVisionJepa<P>,
    validation_base_seed: u64,
    validation_batches: usize,
) -> ProjectedSignedTargetBankSeparability
where
    P: PredictorModule,
{
    assert!(
        validation_batches > 0,
        "validation_batches must be greater than 0"
    );

    let mut totals = SignedTargetBankSeparabilityTotals::default();

    for batch_idx in 0..validation_batches {
        let (x_t, x_t1) = make_temporal_batch_for_task(
            BATCH_SIZE,
            validation_base_seed + batch_idx as u64,
            TemporalTaskMode::SignedVelocityTrail,
        );
        let outcomes = projected_signed_target_bank_sample_outcomes(model, &x_t, &x_t1);

        for outcome in outcomes {
            totals.observe(outcome);
        }
    }

    totals.into_separability()
}

#[allow(dead_code)]
pub fn projected_signed_prediction_bank_margin<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
) -> ProjectedSignedPredictionBankMargin
where
    P: PredictorModule,
{
    let outcomes = projected_signed_prediction_bank_margin_sample_outcomes(model, x_t, x_t1);
    let mut totals = SignedPredictionBankMarginTotals::default();

    for outcome in outcomes {
        totals.observe(outcome);
    }

    totals.into_margin()
}

pub fn projected_signed_prediction_bank_margin_from_base_seed<P>(
    model: &ProjectedVisionJepa<P>,
    validation_base_seed: u64,
    validation_batches: usize,
) -> ProjectedSignedPredictionBankMargin
where
    P: PredictorModule,
{
    assert!(
        validation_batches > 0,
        "validation_batches must be greater than 0"
    );

    let mut totals = SignedPredictionBankMarginTotals::default();

    for batch_idx in 0..validation_batches {
        let (x_t, x_t1) = make_temporal_batch_for_task(
            BATCH_SIZE,
            validation_base_seed + batch_idx as u64,
            TemporalTaskMode::SignedVelocityTrail,
        );
        let outcomes = projected_signed_prediction_bank_margin_sample_outcomes(model, &x_t, &x_t1);

        for outcome in outcomes {
            totals.observe(outcome);
        }
    }

    totals.into_margin()
}

#[allow(dead_code)]
pub fn projected_signed_prediction_bank_unit_geometry<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
) -> ProjectedSignedPredictionBankUnitGeometry
where
    P: PredictorModule,
{
    let outcomes = projected_signed_prediction_bank_unit_geometry_sample_outcomes(model, x_t, x_t1);
    let mut totals = SignedPredictionBankUnitGeometryTotals::default();

    for outcome in outcomes {
        totals.observe(outcome);
    }

    totals.into_geometry()
}

pub fn projected_signed_prediction_bank_unit_geometry_from_base_seed<P>(
    model: &ProjectedVisionJepa<P>,
    validation_base_seed: u64,
    validation_batches: usize,
) -> ProjectedSignedPredictionBankUnitGeometry
where
    P: PredictorModule,
{
    assert!(
        validation_batches > 0,
        "validation_batches must be greater than 0"
    );

    let mut totals = SignedPredictionBankUnitGeometryTotals::default();

    for batch_idx in 0..validation_batches {
        let (x_t, x_t1) = make_temporal_batch_for_task(
            BATCH_SIZE,
            validation_base_seed + batch_idx as u64,
            TemporalTaskMode::SignedVelocityTrail,
        );
        let outcomes =
            projected_signed_prediction_bank_unit_geometry_sample_outcomes(model, &x_t, &x_t1);

        for outcome in outcomes {
            totals.observe(outcome);
        }
    }

    totals.into_geometry()
}

pub fn projected_signed_prediction_geometry_counterfactual_from_base_seed<P>(
    model: &ProjectedVisionJepa<P>,
    validation_base_seed: u64,
    validation_batches: usize,
) -> ProjectedSignedPredictionGeometryCounterfactual
where
    P: PredictorModule,
{
    assert!(
        validation_batches >= 2,
        "geometry counterfactual requires at least one support and one query batch"
    );

    let support_batches = validation_batches / 2;
    let query_batches = validation_batches - support_batches;
    assert!(
        support_batches > 0 && query_batches > 0,
        "geometry counterfactual requires non-empty support and query splits"
    );

    let mut support_prediction_norm = 0.0f32;
    let mut support_target_norm = 0.0f32;
    let mut support_samples = 0usize;

    for batch_idx in 0..support_batches {
        let (x_t, x_t1) = make_temporal_batch_for_task(
            BATCH_SIZE,
            validation_base_seed + batch_idx as u64,
            TemporalTaskMode::SignedVelocityTrail,
        );
        let (prediction_norm, target_norm, samples) =
            signed_prediction_bank_center_norm_sums(model, &x_t, &x_t1);
        support_prediction_norm += prediction_norm;
        support_target_norm += target_norm;
        support_samples += samples;
    }

    let support_norm_ratio =
        support_target_norm / support_prediction_norm.max(VELOCITY_BANK_TIE_EPSILON);
    let mut oracle_radius_totals = SignedPredictionCounterfactualMetricTotals::default();
    let mut oracle_angle_totals = SignedPredictionCounterfactualMetricTotals::default();
    let mut support_global_rescale_totals = SignedPredictionCounterfactualMetricTotals::default();

    for query_idx in 0..query_batches {
        let batch_idx = support_batches + query_idx;
        let (x_t, x_t1) = make_temporal_batch_for_task(
            BATCH_SIZE,
            validation_base_seed + batch_idx as u64,
            TemporalTaskMode::SignedVelocityTrail,
        );
        let outcomes = projected_signed_prediction_geometry_counterfactual_sample_outcomes(
            model,
            &x_t,
            &x_t1,
            support_norm_ratio,
        );

        for outcome in outcomes {
            oracle_radius_totals.observe(outcome.oracle_radius);
            oracle_angle_totals.observe(outcome.oracle_angle);
            support_global_rescale_totals.observe(outcome.support_global_rescale);
        }
    }

    let query_samples = support_global_rescale_totals.samples;

    ProjectedSignedPredictionGeometryCounterfactual {
        oracle_radius: oracle_radius_totals.into_metrics(),
        oracle_angle: oracle_angle_totals.into_metrics(),
        support_global_rescale: support_global_rescale_totals.into_metrics(),
        support_norm_ratio,
        support_samples,
        query_samples,
        candidates: SIGNED_VELOCITY_BANK_CANDIDATE_DX.len(),
    }
}

#[allow(dead_code)]
pub fn projected_signed_objective_error_breakdown_from_base_seed<P>(
    model: &ProjectedVisionJepa<P>,
    validation_base_seed: u64,
    validation_batches: usize,
) -> ProjectedSignedObjectiveErrorBreakdown
where
    P: PredictorModule,
{
    assert!(
        validation_batches > 0,
        "validation_batches must be greater than 0"
    );

    let mut totals = SignedObjectiveErrorBreakdownTotals::default();

    for batch_idx in 0..validation_batches {
        let (x_t, x_t1) = make_temporal_batch_for_task(
            BATCH_SIZE,
            validation_base_seed + batch_idx as u64,
            TemporalTaskMode::SignedVelocityTrail,
        );

        observe_signed_objective_error_batch(model, &x_t, &x_t1, &mut totals);
    }

    totals.into_breakdown()
}

pub fn projected_signed_margin_objective_loss_and_grad<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
    config: SignedMarginObjectiveConfig,
) -> (SignedMarginObjectiveReport, Tensor)
where
    P: PredictorModule,
{
    assert_eq!(
        x_t.shape, x_t1.shape,
        "signed margin objective expects matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "signed margin objective expects rank-4 temporal batches, got {:?}",
        x_t.shape
    );

    let prediction = model.predict_next_projection(x_t);
    let candidate_targets = signed_velocity_candidate_target_projections(model, x_t);
    let true_candidate_indices = signed_velocity_true_candidate_indices(x_t, x_t1);

    signed_margin_objective_loss_and_grad(
        &prediction,
        &candidate_targets,
        &true_candidate_indices,
        &SIGNED_VELOCITY_BANK_CANDIDATE_DX,
        config,
    )
}

pub fn projected_signed_bank_softmax_objective_loss_and_grad<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
    config: SignedBankSoftmaxObjectiveConfig,
) -> (SignedBankSoftmaxObjectiveReport, Tensor)
where
    P: PredictorModule,
{
    assert_eq!(
        x_t.shape, x_t1.shape,
        "signed bank softmax objective expects matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "signed bank softmax objective expects rank-4 temporal batches, got {:?}",
        x_t.shape
    );

    let prediction = model.predict_next_projection(x_t);
    let candidate_targets = signed_velocity_candidate_target_projections(model, x_t);
    let true_candidate_indices = signed_velocity_true_candidate_indices(x_t, x_t1);

    signed_bank_softmax_objective_loss_and_grad(
        &prediction,
        &candidate_targets,
        &true_candidate_indices,
        config,
    )
}

pub fn projected_signed_radial_calibration_loss_and_grad<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
) -> (SignedRadialCalibrationReport, Tensor)
where
    P: PredictorModule,
{
    assert_eq!(
        x_t.shape, x_t1.shape,
        "signed radial calibration expects matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "signed radial calibration expects rank-4 temporal batches, got {:?}",
        x_t.shape
    );

    let prediction = model.predict_next_projection(x_t);
    let candidate_targets = signed_velocity_candidate_target_projections(model, x_t);
    let true_candidate_indices = signed_velocity_true_candidate_indices(x_t, x_t1);

    signed_radial_calibration_loss_and_grad(
        &prediction,
        &candidate_targets,
        &true_candidate_indices,
    )
}

pub fn projected_signed_angular_radial_objective_loss_and_grad<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
    config: SignedAngularRadialObjectiveConfig,
) -> (SignedAngularRadialObjectiveReport, Tensor)
where
    P: PredictorModule,
{
    assert_eq!(
        x_t.shape, x_t1.shape,
        "signed angular-radial objective expects matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "signed angular-radial objective expects rank-4 temporal batches, got {:?}",
        x_t.shape
    );

    let prediction = model.predict_next_projection(x_t);
    let candidate_targets = signed_velocity_candidate_target_projections(model, x_t);
    let true_candidate_indices = signed_velocity_true_candidate_indices(x_t, x_t1);

    signed_angular_radial_objective_loss_and_grad(
        &prediction,
        &candidate_targets,
        &true_candidate_indices,
        config,
    )
}

pub fn projected_signed_margin_objective_report_from_base_seed<P>(
    model: &ProjectedVisionJepa<P>,
    validation_base_seed: u64,
    validation_batches: usize,
    config: SignedMarginObjectiveConfig,
) -> SignedMarginObjectiveReport
where
    P: PredictorModule,
{
    assert!(
        validation_batches > 0,
        "validation_batches must be greater than 0"
    );

    let mut totals = SignedMarginObjectiveReportTotals::default();

    for batch_idx in 0..validation_batches {
        let (x_t, x_t1) = make_temporal_batch_for_task(
            BATCH_SIZE,
            validation_base_seed + batch_idx as u64,
            TemporalTaskMode::SignedVelocityTrail,
        );
        let (report, _) =
            projected_signed_margin_objective_loss_and_grad(model, &x_t, &x_t1, config);
        totals.observe(report);
    }

    totals.into_report()
}

pub fn projected_signed_bank_softmax_objective_report_from_base_seed<P>(
    model: &ProjectedVisionJepa<P>,
    validation_base_seed: u64,
    validation_batches: usize,
    config: SignedBankSoftmaxObjectiveConfig,
) -> SignedBankSoftmaxObjectiveReport
where
    P: PredictorModule,
{
    assert!(
        validation_batches > 0,
        "validation_batches must be greater than 0"
    );

    let mut totals = SignedBankSoftmaxObjectiveReportTotals::default();

    for batch_idx in 0..validation_batches {
        let (x_t, x_t1) = make_temporal_batch_for_task(
            BATCH_SIZE,
            validation_base_seed + batch_idx as u64,
            TemporalTaskMode::SignedVelocityTrail,
        );
        let (report, _) =
            projected_signed_bank_softmax_objective_loss_and_grad(model, &x_t, &x_t1, config);
        totals.observe(report);
    }

    totals.into_report()
}

pub fn projected_signed_radial_calibration_report_from_base_seed<P>(
    model: &ProjectedVisionJepa<P>,
    validation_base_seed: u64,
    validation_batches: usize,
) -> SignedRadialCalibrationReport
where
    P: PredictorModule,
{
    assert!(
        validation_batches > 0,
        "validation_batches must be greater than 0"
    );

    let mut totals = SignedRadialCalibrationReportTotals::default();

    for batch_idx in 0..validation_batches {
        let (x_t, x_t1) = make_temporal_batch_for_task(
            BATCH_SIZE,
            validation_base_seed + batch_idx as u64,
            TemporalTaskMode::SignedVelocityTrail,
        );
        let (report, _) = projected_signed_radial_calibration_loss_and_grad(model, &x_t, &x_t1);
        totals.observe(report);
    }

    totals.into_report()
}

pub fn projected_signed_angular_radial_objective_report_from_base_seed<P>(
    model: &ProjectedVisionJepa<P>,
    validation_base_seed: u64,
    validation_batches: usize,
    config: SignedAngularRadialObjectiveConfig,
) -> SignedAngularRadialObjectiveReport
where
    P: PredictorModule,
{
    assert!(
        validation_batches > 0,
        "validation_batches must be greater than 0"
    );

    let mut totals = SignedAngularRadialObjectiveReportTotals::default();

    for batch_idx in 0..validation_batches {
        let (x_t, x_t1) = make_temporal_batch_for_task(
            BATCH_SIZE,
            validation_base_seed + batch_idx as u64,
            TemporalTaskMode::SignedVelocityTrail,
        );
        let (report, _) =
            projected_signed_angular_radial_objective_loss_and_grad(model, &x_t, &x_t1, config);
        totals.observe(report);
    }

    totals.into_report()
}

#[derive(Debug, Default)]
struct SignedMarginObjectiveReportTotals {
    bank_loss: f32,
    sign_loss: f32,
    speed_loss: f32,
    weighted_loss: f32,
    samples: usize,
    bank_pairs: usize,
    sign_pairs: usize,
    speed_pairs: usize,
    active_bank_pairs: usize,
    active_sign_pairs: usize,
    active_speed_pairs: usize,
}

impl SignedMarginObjectiveReportTotals {
    fn observe(&mut self, report: SignedMarginObjectiveReport) {
        self.bank_loss += report.bank_loss * report.bank_pairs as f32;
        self.sign_loss += report.sign_loss * report.sign_pairs as f32;
        self.speed_loss += report.speed_loss * report.speed_pairs as f32;
        self.weighted_loss += report.weighted_loss * report.samples as f32;
        self.samples += report.samples;
        self.bank_pairs += report.bank_pairs;
        self.sign_pairs += report.sign_pairs;
        self.speed_pairs += report.speed_pairs;
        self.active_bank_pairs += report.active_bank_pairs;
        self.active_sign_pairs += report.active_sign_pairs;
        self.active_speed_pairs += report.active_speed_pairs;
    }

    fn into_report(self) -> SignedMarginObjectiveReport {
        assert!(self.samples > 0, "signed margin objective has no samples");
        assert!(
            self.bank_pairs > 0 && self.sign_pairs > 0 && self.speed_pairs > 0,
            "signed margin objective has empty pair counts"
        );

        SignedMarginObjectiveReport {
            bank_loss: self.bank_loss / self.bank_pairs as f32,
            sign_loss: self.sign_loss / self.sign_pairs as f32,
            speed_loss: self.speed_loss / self.speed_pairs as f32,
            weighted_loss: self.weighted_loss / self.samples as f32,
            samples: self.samples,
            bank_pairs: self.bank_pairs,
            sign_pairs: self.sign_pairs,
            speed_pairs: self.speed_pairs,
            active_bank_pairs: self.active_bank_pairs,
            active_sign_pairs: self.active_sign_pairs,
            active_speed_pairs: self.active_speed_pairs,
        }
    }
}

#[derive(Debug, Default)]
struct SignedBankSoftmaxObjectiveReportTotals {
    loss: f32,
    top1: f32,
    mean_true_probability: f32,
    samples: usize,
}

#[derive(Debug, Default)]
struct SignedRadialCalibrationReportTotals {
    loss: f32,
    prediction_norm: f32,
    target_norm: f32,
    norm_ratio: f32,
    samples: usize,
}

#[derive(Debug, Default)]
struct SignedAngularRadialObjectiveReportTotals {
    loss: f32,
    angular_loss: f32,
    radial_loss: f32,
    cosine: f32,
    prediction_norm: f32,
    target_norm: f32,
    norm_ratio: f32,
    samples: usize,
}

impl SignedBankSoftmaxObjectiveReportTotals {
    fn observe(&mut self, report: SignedBankSoftmaxObjectiveReport) {
        self.loss += report.loss * report.samples as f32;
        self.top1 += report.top1 * report.samples as f32;
        self.mean_true_probability += report.mean_true_probability * report.samples as f32;
        self.samples += report.samples;
    }

    fn into_report(self) -> SignedBankSoftmaxObjectiveReport {
        assert!(
            self.samples > 0,
            "signed bank softmax objective has no samples"
        );

        SignedBankSoftmaxObjectiveReport {
            loss: self.loss / self.samples as f32,
            top1: self.top1 / self.samples as f32,
            mean_true_probability: self.mean_true_probability / self.samples as f32,
            samples: self.samples,
        }
    }
}

impl SignedRadialCalibrationReportTotals {
    fn observe(&mut self, report: SignedRadialCalibrationReport) {
        self.loss += report.loss * report.samples as f32;
        self.prediction_norm += report.prediction_norm * report.samples as f32;
        self.target_norm += report.target_norm * report.samples as f32;
        self.norm_ratio += report.norm_ratio * report.samples as f32;
        self.samples += report.samples;
    }

    fn into_report(self) -> SignedRadialCalibrationReport {
        assert!(self.samples > 0, "signed radial calibration has no samples");

        SignedRadialCalibrationReport {
            loss: self.loss / self.samples as f32,
            prediction_norm: self.prediction_norm / self.samples as f32,
            target_norm: self.target_norm / self.samples as f32,
            norm_ratio: self.norm_ratio / self.samples as f32,
            samples: self.samples,
        }
    }
}

impl SignedAngularRadialObjectiveReportTotals {
    fn observe(&mut self, report: SignedAngularRadialObjectiveReport) {
        self.loss += report.loss * report.samples as f32;
        self.angular_loss += report.angular_loss * report.samples as f32;
        self.radial_loss += report.radial_loss * report.samples as f32;
        self.cosine += report.cosine * report.samples as f32;
        self.prediction_norm += report.prediction_norm * report.samples as f32;
        self.target_norm += report.target_norm * report.samples as f32;
        self.norm_ratio += report.norm_ratio * report.samples as f32;
        self.samples += report.samples;
    }

    fn into_report(self) -> SignedAngularRadialObjectiveReport {
        assert!(
            self.samples > 0,
            "signed angular-radial objective has no samples"
        );

        SignedAngularRadialObjectiveReport {
            loss: self.loss / self.samples as f32,
            angular_loss: self.angular_loss / self.samples as f32,
            radial_loss: self.radial_loss / self.samples as f32,
            cosine: self.cosine / self.samples as f32,
            prediction_norm: self.prediction_norm / self.samples as f32,
            target_norm: self.target_norm / self.samples as f32,
            norm_ratio: self.norm_ratio / self.samples as f32,
            samples: self.samples,
        }
    }
}

#[allow(dead_code)]
fn observe_signed_objective_error_batch<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
    totals: &mut SignedObjectiveErrorBreakdownTotals,
) where
    P: PredictorModule,
{
    assert_eq!(
        x_t.shape, x_t1.shape,
        "signed objective error expects matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "signed objective error expects rank-4 temporal batches, got {:?}",
        x_t.shape
    );

    let prediction = model.predict_next_projection(x_t);
    let actual_target = model.target_projection(x_t1);
    let batch_size = x_t.shape[0];

    for sample in 0..batch_size {
        let true_dx = signed_motion_dx_for_sample(x_t, x_t1, sample);
        let loss = sample_mean_squared_distance(&prediction, &actual_target, sample);

        totals.observe(true_dx, loss);
    }
}

fn signed_velocity_candidate_target_projections<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
) -> Vec<Tensor>
where
    P: PredictorModule,
{
    SIGNED_VELOCITY_BANK_CANDIDATE_DX
        .iter()
        .map(|candidate_dx| {
            let candidate_x_t1 =
                make_signed_velocity_trail_candidate_target_batch(x_t, *candidate_dx);
            model.target_projection(&candidate_x_t1)
        })
        .collect()
}

fn signed_velocity_true_candidate_indices(x_t: &Tensor, x_t1: &Tensor) -> Vec<usize> {
    (0..x_t.shape[0])
        .map(|sample| {
            let true_dx = signed_motion_dx_for_sample(x_t, x_t1, sample);
            signed_velocity_candidate_index(true_dx)
        })
        .collect()
}

fn signed_velocity_candidate_index(true_dx: isize) -> usize {
    SIGNED_VELOCITY_BANK_CANDIDATE_DX
        .iter()
        .position(|candidate_dx| *candidate_dx == true_dx)
        .unwrap_or_else(|| panic!("true dx {} is missing from signed target bank", true_dx))
}

fn observe_signed_state_centroids(
    x_t: &Tensor,
    x_t1: &Tensor,
    latent: &Tensor,
    projection: &Tensor,
    latent_centroids: &mut SignedStateCentroids,
    projection_centroids: &mut SignedStateCentroids,
    support_samples: &mut usize,
) {
    assert_eq!(
        x_t.shape, x_t1.shape,
        "signed state separability expects matching pair shapes"
    );

    for sample in 0..x_t.shape[0] {
        let true_dx = signed_motion_dx_for_sample(x_t, x_t1, sample);

        latent_centroids.observe(latent, sample, true_dx);
        projection_centroids.observe(projection, sample, true_dx);
        *support_samples += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn observe_signed_state_query(
    x_t: &Tensor,
    x_t1: &Tensor,
    latent: &Tensor,
    projection: &Tensor,
    latent_centroids: &SignedStateCentroids,
    projection_centroids: &SignedStateCentroids,
    latent_totals: &mut SignedStateSeparabilityTotals,
    projection_totals: &mut SignedStateSeparabilityTotals,
    query_counts: &mut [usize],
) {
    assert_eq!(
        x_t.shape, x_t1.shape,
        "signed state separability expects matching pair shapes"
    );

    for sample in 0..x_t.shape[0] {
        let true_dx = signed_motion_dx_for_sample(x_t, x_t1, sample);
        let class_index = signed_velocity_candidate_index(true_dx);

        latent_totals.observe(latent_centroids.sample_outcome(latent, sample, true_dx));
        projection_totals.observe(projection_centroids.sample_outcome(projection, sample, true_dx));
        query_counts[class_index] += 1;
    }
}

fn assert_all_signed_state_query_classes_present(query_counts: &[usize]) {
    for (class_index, count) in query_counts.iter().enumerate() {
        assert!(
            *count > 0,
            "query signed state separability is missing dx {}",
            SIGNED_VELOCITY_BANK_CANDIDATE_DX[class_index]
        );
    }
}

fn projected_velocity_bank_sample_outcomes<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
    temporal_task_mode: TemporalTaskMode,
) -> Vec<VelocityBankSampleOutcome>
where
    P: PredictorModule,
{
    assert_eq!(
        x_t.shape, x_t1.shape,
        "velocity-bank breakdown expects matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "velocity-bank breakdown expects rank-4 temporal batches, got {:?}",
        x_t.shape
    );

    let candidate_dx_bank = velocity_bank_candidates_for_task(temporal_task_mode);
    let prediction = model.predict_next_projection(x_t);
    let candidate_targets = candidate_dx_bank
        .iter()
        .map(|candidate_dx| {
            let candidate_x_t1 =
                make_velocity_bank_candidate_target_batch(x_t, *candidate_dx, temporal_task_mode);
            model.target_projection(&candidate_x_t1)
        })
        .collect::<Vec<_>>();
    let batch_size = x_t.shape[0];
    let mut outcomes = Vec::with_capacity(batch_size);

    for sample in 0..batch_size {
        let true_dx = signed_motion_dx_for_sample(x_t, x_t1, sample);
        let true_index = candidate_dx_bank
            .iter()
            .position(|candidate_dx| *candidate_dx == true_dx)
            .unwrap_or_else(|| panic!("true dx {} is missing from velocity bank", true_dx));
        let distances = candidate_targets
            .iter()
            .map(|candidate_target| sample_squared_distance(&prediction, candidate_target, sample))
            .collect::<Vec<_>>();
        let true_distance = distances[true_index];
        let mut rank = 1usize;
        let mut best_index = 0usize;
        let mut best_distance = distances[0];

        for (candidate_index, distance) in distances.iter().enumerate() {
            if candidate_index != true_index
                && *distance <= true_distance + VELOCITY_BANK_TIE_EPSILON
            {
                rank += 1;
            }
            if *distance < best_distance - VELOCITY_BANK_TIE_EPSILON {
                best_index = candidate_index;
                best_distance = *distance;
            }
        }

        let best_dx = candidate_dx_bank[best_index];
        let true_sign_distance =
            best_group_distance(candidate_dx_bank, &distances, |candidate_dx| {
                candidate_dx.signum() == true_dx.signum()
            });
        let other_sign_distance =
            best_group_distance(candidate_dx_bank, &distances, |candidate_dx| {
                candidate_dx.signum() != true_dx.signum()
            });
        let true_speed_distance =
            best_group_distance(candidate_dx_bank, &distances, |candidate_dx| {
                candidate_dx.abs() == true_dx.abs()
            });
        let other_speed_distance =
            best_group_distance(candidate_dx_bank, &distances, |candidate_dx| {
                candidate_dx.abs() != true_dx.abs()
            });

        outcomes.push(VelocityBankSampleOutcome {
            true_dx,
            best_dx,
            rank,
            sign_correct: true_sign_distance < other_sign_distance - VELOCITY_BANK_TIE_EPSILON,
            speed_correct: true_speed_distance < other_speed_distance - VELOCITY_BANK_TIE_EPSILON,
        });
    }

    outcomes
}

fn projected_signed_prediction_bank_margin_sample_outcomes<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
) -> Vec<SignedPredictionBankMarginSampleOutcome>
where
    P: PredictorModule,
{
    assert_eq!(
        x_t.shape, x_t1.shape,
        "prediction-bank margin expects matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "prediction-bank margin expects rank-4 temporal batches, got {:?}",
        x_t.shape
    );

    let candidate_dx_bank = &SIGNED_VELOCITY_BANK_CANDIDATE_DX;
    let prediction = model.predict_next_projection(x_t);
    let candidate_targets = candidate_dx_bank
        .iter()
        .map(|candidate_dx| {
            let candidate_x_t1 =
                make_signed_velocity_trail_candidate_target_batch(x_t, *candidate_dx);
            model.target_projection(&candidate_x_t1)
        })
        .collect::<Vec<_>>();
    let batch_size = x_t.shape[0];
    let mut outcomes = Vec::with_capacity(batch_size);

    for sample in 0..batch_size {
        let true_dx = signed_motion_dx_for_sample(x_t, x_t1, sample);
        let true_index = candidate_dx_bank
            .iter()
            .position(|candidate_dx| *candidate_dx == true_dx)
            .unwrap_or_else(|| panic!("true dx {} is missing from signed target bank", true_dx));
        let distances = candidate_targets
            .iter()
            .map(|candidate_target| sample_squared_distance(&prediction, candidate_target, sample))
            .collect::<Vec<_>>();
        let true_distance = distances[true_index];
        let nearest_wrong_distance =
            best_indexed_group_distance(candidate_dx_bank, &distances, |candidate_index, _| {
                candidate_index != true_index
            });
        let true_sign_distance =
            best_indexed_group_distance(candidate_dx_bank, &distances, |_, candidate_dx| {
                candidate_dx.signum() == true_dx.signum()
            });
        let other_sign_distance =
            best_indexed_group_distance(candidate_dx_bank, &distances, |_, candidate_dx| {
                candidate_dx.signum() != true_dx.signum()
            });
        let true_speed_distance =
            best_indexed_group_distance(candidate_dx_bank, &distances, |_, candidate_dx| {
                candidate_dx.abs() == true_dx.abs()
            });
        let other_speed_distance =
            best_indexed_group_distance(candidate_dx_bank, &distances, |_, candidate_dx| {
                candidate_dx.abs() != true_dx.abs()
            });

        outcomes.push(SignedPredictionBankMarginSampleOutcome {
            true_dx,
            true_distance,
            nearest_wrong_distance,
            sign_margin: other_sign_distance - true_sign_distance,
            speed_margin: other_speed_distance - true_speed_distance,
        });
    }

    outcomes
}

fn projected_signed_prediction_bank_unit_geometry_sample_outcomes<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
) -> Vec<SignedPredictionBankUnitGeometrySampleOutcome>
where
    P: PredictorModule,
{
    assert_eq!(
        x_t.shape, x_t1.shape,
        "prediction-bank unit geometry expects matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "prediction-bank unit geometry expects rank-4 temporal batches, got {:?}",
        x_t.shape
    );

    let candidate_dx_bank = &SIGNED_VELOCITY_BANK_CANDIDATE_DX;
    let prediction = model.predict_next_projection(x_t);
    let candidate_targets = candidate_dx_bank
        .iter()
        .map(|candidate_dx| {
            let candidate_x_t1 =
                make_signed_velocity_trail_candidate_target_batch(x_t, *candidate_dx);
            model.target_projection(&candidate_x_t1)
        })
        .collect::<Vec<_>>();
    let batch_size = x_t.shape[0];
    let projection_dim = prediction.shape[1];
    let mut outcomes = Vec::with_capacity(batch_size);

    for sample in 0..batch_size {
        let true_dx = signed_motion_dx_for_sample(x_t, x_t1, sample);
        let true_index = signed_velocity_candidate_index(true_dx);
        let mut centroid = vec![0.0f32; projection_dim];

        for candidate_target in &candidate_targets {
            for (feature_idx, centroid_value) in centroid.iter_mut().enumerate() {
                *centroid_value += candidate_target.get(&[sample, feature_idx]);
            }
        }
        for centroid_value in &mut centroid {
            *centroid_value /= candidate_targets.len() as f32;
        }

        let prediction_centered = centered_sample_vector(&prediction, sample, &centroid);
        let prediction_center_norm = vector_l2_norm(&prediction_centered);
        let prediction_unit = unit_vector(&prediction_centered);
        let candidate_units = candidate_targets
            .iter()
            .map(|candidate_target| {
                let centered = centered_sample_vector(candidate_target, sample, &centroid);
                unit_vector(&centered)
            })
            .collect::<Vec<_>>();
        let true_target_center_norm = vector_l2_norm(&centered_sample_vector(
            &candidate_targets[true_index],
            sample,
            &centroid,
        ));
        let distances = candidate_units
            .iter()
            .map(|candidate_unit| vector_squared_distance(&prediction_unit, candidate_unit))
            .collect::<Vec<_>>();
        let true_distance = distances[true_index];
        let nearest_wrong_distance =
            best_indexed_group_distance(candidate_dx_bank, &distances, |candidate_index, _| {
                candidate_index != true_index
            });
        let mut rank = 1usize;

        for (candidate_index, distance) in distances.iter().enumerate() {
            if candidate_index != true_index
                && *distance <= true_distance + VELOCITY_BANK_TIE_EPSILON
            {
                rank += 1;
            }
        }

        let true_sign_distance =
            best_indexed_group_distance(candidate_dx_bank, &distances, |_, candidate_dx| {
                candidate_dx.signum() == true_dx.signum()
            });
        let other_sign_distance =
            best_indexed_group_distance(candidate_dx_bank, &distances, |_, candidate_dx| {
                candidate_dx.signum() != true_dx.signum()
            });
        let true_speed_distance =
            best_indexed_group_distance(candidate_dx_bank, &distances, |_, candidate_dx| {
                candidate_dx.abs() == true_dx.abs()
            });
        let other_speed_distance =
            best_indexed_group_distance(candidate_dx_bank, &distances, |_, candidate_dx| {
                candidate_dx.abs() != true_dx.abs()
            });

        outcomes.push(SignedPredictionBankUnitGeometrySampleOutcome {
            true_dx,
            true_distance,
            nearest_wrong_distance,
            rank,
            sign_margin: other_sign_distance - true_sign_distance,
            speed_margin: other_speed_distance - true_speed_distance,
            prediction_center_norm,
            true_target_center_norm,
        });
    }

    outcomes
}

fn signed_prediction_bank_center_norm_sums<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
) -> (f32, f32, usize)
where
    P: PredictorModule,
{
    assert_eq!(
        x_t.shape, x_t1.shape,
        "prediction-bank center norm sums expect matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "prediction-bank center norm sums expect rank-4 temporal batches, got {:?}",
        x_t.shape
    );

    let prediction = model.predict_next_projection(x_t);
    let candidate_targets = signed_velocity_candidate_target_projections(model, x_t);
    let batch_size = x_t.shape[0];
    let projection_dim = prediction.shape[1];
    let mut prediction_norm_sum = 0.0f32;
    let mut target_norm_sum = 0.0f32;

    for sample in 0..batch_size {
        let true_dx = signed_motion_dx_for_sample(x_t, x_t1, sample);
        let true_index = signed_velocity_candidate_index(true_dx);
        let centroid = signed_candidate_centroid(&candidate_targets, sample, projection_dim);
        let prediction_centered = centered_sample_vector(&prediction, sample, &centroid);
        let target_centered =
            centered_sample_vector(&candidate_targets[true_index], sample, &centroid);

        prediction_norm_sum += vector_l2_norm(&prediction_centered);
        target_norm_sum += vector_l2_norm(&target_centered);
    }

    (prediction_norm_sum, target_norm_sum, batch_size)
}

fn projected_signed_prediction_geometry_counterfactual_sample_outcomes<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
    support_norm_ratio: f32,
) -> Vec<SignedPredictionGeometryCounterfactualSampleOutcome>
where
    P: PredictorModule,
{
    assert!(
        support_norm_ratio.is_finite() && support_norm_ratio >= 0.0,
        "support norm ratio must be finite and non-negative"
    );
    assert_eq!(
        x_t.shape, x_t1.shape,
        "prediction geometry counterfactual expects matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "prediction geometry counterfactual expects rank-4 temporal batches, got {:?}",
        x_t.shape
    );

    let candidate_dx_bank = &SIGNED_VELOCITY_BANK_CANDIDATE_DX;
    let prediction = model.predict_next_projection(x_t);
    let candidate_targets = signed_velocity_candidate_target_projections(model, x_t);
    let batch_size = x_t.shape[0];
    let projection_dim = prediction.shape[1];
    let mut outcomes = Vec::with_capacity(batch_size);

    for sample in 0..batch_size {
        let true_dx = signed_motion_dx_for_sample(x_t, x_t1, sample);
        let true_index = signed_velocity_candidate_index(true_dx);
        let centroid = signed_candidate_centroid(&candidate_targets, sample, projection_dim);
        let prediction_centered = centered_sample_vector(&prediction, sample, &centroid);
        let target_centered =
            centered_sample_vector(&candidate_targets[true_index], sample, &centroid);
        let prediction_norm = vector_l2_norm(&prediction_centered);
        let target_norm = vector_l2_norm(&target_centered);
        let prediction_unit = unit_vector(&prediction_centered);
        let target_unit = unit_vector(&target_centered);
        let oracle_radius_centered = scale_vector(&prediction_unit, target_norm);
        let oracle_angle_centered = scale_vector(&target_unit, prediction_norm);
        let support_global_rescale_centered =
            scale_vector(&prediction_centered, support_norm_ratio);

        let oracle_radius_norm_ratio =
            vector_l2_norm(&oracle_radius_centered) / target_norm.max(VELOCITY_BANK_TIE_EPSILON);
        let oracle_angle_norm_ratio =
            vector_l2_norm(&oracle_angle_centered) / target_norm.max(VELOCITY_BANK_TIE_EPSILON);
        let support_global_rescale_norm_ratio = vector_l2_norm(&support_global_rescale_centered)
            / target_norm.max(VELOCITY_BANK_TIE_EPSILON);

        outcomes.push(SignedPredictionGeometryCounterfactualSampleOutcome {
            oracle_radius: signed_prediction_counterfactual_outcome(
                candidate_dx_bank,
                &candidate_targets,
                sample,
                true_dx,
                true_index,
                &centroid,
                &oracle_radius_centered,
                oracle_radius_norm_ratio,
            ),
            oracle_angle: signed_prediction_counterfactual_outcome(
                candidate_dx_bank,
                &candidate_targets,
                sample,
                true_dx,
                true_index,
                &centroid,
                &oracle_angle_centered,
                oracle_angle_norm_ratio,
            ),
            support_global_rescale: signed_prediction_counterfactual_outcome(
                candidate_dx_bank,
                &candidate_targets,
                sample,
                true_dx,
                true_index,
                &centroid,
                &support_global_rescale_centered,
                support_global_rescale_norm_ratio,
            ),
        });
    }

    outcomes
}

fn projected_signed_target_bank_sample_outcomes<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
) -> Vec<SignedTargetBankSampleOutcome>
where
    P: PredictorModule,
{
    assert_eq!(
        x_t.shape, x_t1.shape,
        "target-bank separability expects matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "target-bank separability expects rank-4 temporal batches, got {:?}",
        x_t.shape
    );

    let candidate_dx_bank = &SIGNED_VELOCITY_BANK_CANDIDATE_DX;
    let candidate_targets = candidate_dx_bank
        .iter()
        .map(|candidate_dx| {
            let candidate_x_t1 =
                make_signed_velocity_trail_candidate_target_batch(x_t, *candidate_dx);
            model.target_projection(&candidate_x_t1)
        })
        .collect::<Vec<_>>();
    let actual_target = model.target_projection(x_t1);
    let batch_size = x_t.shape[0];
    let mut outcomes = Vec::with_capacity(batch_size);

    for sample in 0..batch_size {
        let true_dx = signed_motion_dx_for_sample(x_t, x_t1, sample);
        let true_index = candidate_dx_bank
            .iter()
            .position(|candidate_dx| *candidate_dx == true_dx)
            .unwrap_or_else(|| panic!("true dx {} is missing from signed target bank", true_dx));
        let distances = candidate_targets
            .iter()
            .map(|candidate_target| {
                sample_squared_distance(&actual_target, candidate_target, sample)
            })
            .collect::<Vec<_>>();
        let true_distance = distances[true_index];
        let mut rank = 1usize;

        for (candidate_index, distance) in distances.iter().enumerate() {
            if candidate_index != true_index
                && *distance <= true_distance + VELOCITY_BANK_TIE_EPSILON
            {
                rank += 1;
            }
        }

        let nearest_wrong_distance =
            best_indexed_group_distance(candidate_dx_bank, &distances, |candidate_index, _| {
                candidate_index != true_index
            });
        let same_sign_wrong_distance = best_indexed_group_distance(
            candidate_dx_bank,
            &distances,
            |candidate_index, candidate_dx| {
                candidate_index != true_index && candidate_dx.signum() == true_dx.signum()
            },
        );
        let opposite_sign_distance =
            best_indexed_group_distance(candidate_dx_bank, &distances, |_, candidate_dx| {
                candidate_dx.signum() != true_dx.signum()
            });
        let same_speed_wrong_distance = best_indexed_group_distance(
            candidate_dx_bank,
            &distances,
            |candidate_index, candidate_dx| {
                candidate_index != true_index && candidate_dx.abs() == true_dx.abs()
            },
        );
        let different_speed_distance =
            best_indexed_group_distance(candidate_dx_bank, &distances, |_, candidate_dx| {
                candidate_dx.abs() != true_dx.abs()
            });

        outcomes.push(SignedTargetBankSampleOutcome {
            true_dx,
            true_distance,
            nearest_wrong_distance,
            rank,
            sign_margin: opposite_sign_distance - same_sign_wrong_distance,
            speed_margin: different_speed_distance - same_speed_wrong_distance,
        });
    }

    outcomes
}

fn signed_prediction_counterfactual_outcome(
    candidate_dx_bank: &[isize],
    candidate_targets: &[Tensor],
    sample: usize,
    true_dx: isize,
    true_index: usize,
    centroid: &[f32],
    prediction_centered: &[f32],
    norm_ratio: f32,
) -> SignedPredictionCounterfactualSampleOutcome {
    let distances = counterfactual_prediction_distances(
        candidate_targets,
        sample,
        centroid,
        prediction_centered,
    );
    let true_distance = distances[true_index];
    let nearest_wrong_distance =
        best_indexed_group_distance(candidate_dx_bank, &distances, |candidate_index, _| {
            candidate_index != true_index
        });
    let mut rank = 1usize;

    for (candidate_index, distance) in distances.iter().enumerate() {
        if candidate_index != true_index && *distance <= true_distance + VELOCITY_BANK_TIE_EPSILON {
            rank += 1;
        }
    }

    let true_sign_distance =
        best_indexed_group_distance(candidate_dx_bank, &distances, |_, candidate_dx| {
            candidate_dx.signum() == true_dx.signum()
        });
    let other_sign_distance =
        best_indexed_group_distance(candidate_dx_bank, &distances, |_, candidate_dx| {
            candidate_dx.signum() != true_dx.signum()
        });
    let true_speed_distance =
        best_indexed_group_distance(candidate_dx_bank, &distances, |_, candidate_dx| {
            candidate_dx.abs() == true_dx.abs()
        });
    let other_speed_distance =
        best_indexed_group_distance(candidate_dx_bank, &distances, |_, candidate_dx| {
            candidate_dx.abs() != true_dx.abs()
        });

    SignedPredictionCounterfactualSampleOutcome {
        true_dx,
        true_distance,
        nearest_wrong_distance,
        rank,
        sign_margin: other_sign_distance - true_sign_distance,
        speed_margin: other_speed_distance - true_speed_distance,
        norm_ratio,
    }
}

fn counterfactual_prediction_distances(
    candidate_targets: &[Tensor],
    sample: usize,
    centroid: &[f32],
    prediction_centered: &[f32],
) -> Vec<f32> {
    assert_eq!(
        centroid.len(),
        prediction_centered.len(),
        "counterfactual prediction dim mismatch"
    );

    candidate_targets
        .iter()
        .map(|candidate_target| {
            assert_eq!(
                candidate_target.shape[1],
                centroid.len(),
                "counterfactual target dim mismatch"
            );

            let mut distance = 0.0f32;
            for feature_idx in 0..centroid.len() {
                let prediction_value = centroid[feature_idx] + prediction_centered[feature_idx];
                let diff = prediction_value - candidate_target.get(&[sample, feature_idx]);
                distance += diff * diff;
            }
            distance
        })
        .collect()
}

fn signed_candidate_centroid(
    candidate_targets: &[Tensor],
    sample: usize,
    projection_dim: usize,
) -> Vec<f32> {
    let mut centroid = vec![0.0f32; projection_dim];

    for candidate_target in candidate_targets {
        for (feature_idx, centroid_value) in centroid.iter_mut().enumerate() {
            *centroid_value += candidate_target.get(&[sample, feature_idx]);
        }
    }
    for centroid_value in &mut centroid {
        *centroid_value /= candidate_targets.len() as f32;
    }

    centroid
}

fn centered_sample_vector(tensor: &Tensor, sample: usize, centroid: &[f32]) -> Vec<f32> {
    assert!(
        tensor.shape.len() == 2,
        "centered sample vector expects rank-2 tensor, got {:?}",
        tensor.shape
    );
    assert_eq!(
        tensor.shape[1],
        centroid.len(),
        "centered sample vector dim mismatch"
    );

    centroid
        .iter()
        .enumerate()
        .map(|(feature_idx, centroid_value)| tensor.get(&[sample, feature_idx]) - centroid_value)
        .collect()
}

fn scale_vector(vector: &[f32], scale: f32) -> Vec<f32> {
    vector.iter().map(|value| value * scale).collect()
}

fn vector_l2_norm(vector: &[f32]) -> f32 {
    vector.iter().map(|value| value * value).sum::<f32>().sqrt()
}

fn unit_vector(vector: &[f32]) -> Vec<f32> {
    let norm = vector_l2_norm(vector);

    if norm <= VELOCITY_BANK_TIE_EPSILON {
        return vec![0.0; vector.len()];
    }

    vector.iter().map(|value| value / norm).collect()
}

fn vector_squared_distance(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(
        left.len(),
        right.len(),
        "vector distance expects matching lengths"
    );

    left.iter()
        .zip(right.iter())
        .map(|(left_value, right_value)| {
            let diff = left_value - right_value;
            diff * diff
        })
        .sum()
}

fn best_group_distance(
    candidate_dx_bank: &[isize],
    distances: &[f32],
    mut include_candidate: impl FnMut(isize) -> bool,
) -> f32 {
    let mut best_distance = f32::INFINITY;

    for (candidate_index, candidate_dx) in candidate_dx_bank.iter().enumerate() {
        if include_candidate(*candidate_dx) {
            best_distance = best_distance.min(distances[candidate_index]);
        }
    }

    best_distance
}

fn best_indexed_group_distance(
    candidate_dx_bank: &[isize],
    distances: &[f32],
    mut include_candidate: impl FnMut(usize, isize) -> bool,
) -> f32 {
    let mut best_distance = f32::INFINITY;

    for (candidate_index, candidate_dx) in candidate_dx_bank.iter().enumerate() {
        if include_candidate(candidate_index, *candidate_dx) {
            best_distance = best_distance.min(distances[candidate_index]);
        }
    }

    best_distance
}

fn velocity_bank_candidates_for_task(temporal_task_mode: TemporalTaskMode) -> &'static [isize] {
    match temporal_task_mode {
        TemporalTaskMode::VelocityTrail => &VELOCITY_BANK_CANDIDATE_DX,
        TemporalTaskMode::SignedVelocityTrail => &SIGNED_VELOCITY_BANK_CANDIDATE_DX,
        TemporalTaskMode::RandomSpeed => {
            panic!(
                "velocity-bank ranking only supports velocity-trail or signed-velocity-trail task"
            )
        }
    }
}

fn make_velocity_bank_candidate_target_batch(
    x_t: &Tensor,
    candidate_dx: isize,
    temporal_task_mode: TemporalTaskMode,
) -> Tensor {
    match temporal_task_mode {
        TemporalTaskMode::VelocityTrail => {
            make_velocity_trail_candidate_target_batch(x_t, candidate_dx)
        }
        TemporalTaskMode::SignedVelocityTrail => {
            make_signed_velocity_trail_candidate_target_batch(x_t, candidate_dx)
        }
        TemporalTaskMode::RandomSpeed => {
            panic!(
                "velocity-bank ranking only supports velocity-trail or signed-velocity-trail task"
            )
        }
    }
}

fn sample_squared_distance(lhs: &Tensor, rhs: &Tensor, sample: usize) -> f32 {
    assert_eq!(
        lhs.shape, rhs.shape,
        "distance expects matching shapes, got {:?} and {:?}",
        lhs.shape, rhs.shape
    );
    assert!(
        lhs.shape.len() == 2,
        "distance expects projected rank-2 tensors, got {:?}",
        lhs.shape
    );

    let dim = lhs.shape[1];
    let mut distance = 0.0f32;

    for feature_idx in 0..dim {
        let diff = lhs.get(&[sample, feature_idx]) - rhs.get(&[sample, feature_idx]);
        distance += diff * diff;
    }

    distance
}

#[allow(dead_code)]
fn sample_mean_squared_distance(lhs: &Tensor, rhs: &Tensor, sample: usize) -> f32 {
    let distance = sample_squared_distance(lhs, rhs, sample);
    let dim = lhs.shape[1];

    assert!(dim > 0, "distance expects non-empty projection dim");

    distance / dim as f32
}
