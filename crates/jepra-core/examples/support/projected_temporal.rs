use super::temporal_vision::{
    BATCH_SIZE, SIGNED_VELOCITY_BANK_CANDIDATE_DX, TemporalTaskMode, VELOCITY_BANK_CANDIDATE_DX,
    make_signed_velocity_trail_candidate_target_batch, make_temporal_batch_for_task,
    make_velocity_trail_candidate_target_batch, signed_motion_dx_for_sample,
};
use jepra_core::{
    EmbeddingEncoder, Linear, PredictorModule, ProjectedVisionJepa,
    SignedAngularRadialObjectiveConfig, SignedAngularRadialObjectiveReport,
    SignedBankSoftmaxObjectiveConfig, SignedBankSoftmaxObjectiveReport,
    SignedCenteredRadiusScalarObjectiveReport, SignedMarginObjectiveConfig,
    SignedMarginObjectiveReport, SignedRadialCalibrationReport, Tensor,
    gaussian_moment_regularizer, mse_loss, mse_loss_grad,
    signed_angular_radial_objective_loss_and_grad, signed_bank_softmax_objective_loss_and_grad,
    signed_centered_radius_scalar_loss_and_grad, signed_margin_objective_loss_and_grad,
    signed_objectives::{
        SignedDirectCandidateMarginObjectiveReport,
        signed_direct_candidate_margin_objective_loss_and_grad,
    },
    signed_radial_calibration_loss_and_grad,
};

pub const PROJECTED_VALIDATION_BASE_SEED: u64 = 111_000;
pub const PROJECTED_VALIDATION_BATCHES: usize = 8;
pub const PROJECTED_TRAIN_LOSS_MAX_REDUCTION_RATIO: f32 = 0.2;
pub const PROJECTED_VALIDATION_LOSS_MAX_REDUCTION_RATIO: f32 = 0.2;
const VELOCITY_BANK_TIE_EPSILON: f32 = 1e-7;
const CANDIDATE_RADIUS_DELTA_CLAMP: f32 = 2.0;
const RAY_DIRECTION_REPAIR_GRAD_CLIP: f32 = 10.0;

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
pub struct ProjectedSignedPredictionRayBoundary {
    pub current_radius: f32,
    pub required_radius: f32,
    pub upper_radius: f32,
    pub radius_margin: f32,
    pub radius_shortfall: f32,
    pub radius_overshoot: f32,
    pub satisfied_rate: f32,
    pub infeasible_rate: f32,
    pub finite_upper_samples: usize,
    pub feasible_samples: usize,
    pub samples: usize,
    pub candidates: usize,
    pub satisfied_by_dx: [usize; 4],
    pub infeasible_by_dx: [usize; 4],
    pub below_lower_by_dx: [usize; 4],
    pub upper_overshoot_by_dx: [usize; 4],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedSignedRayDirectionRepairReport {
    pub loss: f32,
    pub gap_loss: f32,
    pub parallel_loss: f32,
    pub active_rate: f32,
    pub cosine: f32,
    pub current_radius: f32,
    pub target_radius: f32,
    pub samples: usize,
    pub active_count: usize,
    pub gap_active_count: usize,
    pub parallel_active_count: usize,
    pub zero_direction_skipped: usize,
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedSignedCandidateCentroidIntegration {
    pub mean_radius: ProjectedSignedPredictionCounterfactualMetrics,
    pub nearest_unit_radius: ProjectedSignedPredictionCounterfactualMetrics,
    pub softmax_radius: ProjectedSignedPredictionCounterfactualMetrics,
    pub softmax_temperature: f32,
    pub samples: usize,
    pub candidates: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedSignedCandidateRadiusHeadIntegration {
    pub learned_radius: ProjectedSignedPredictionCounterfactualMetrics,
    pub scalar_report: SignedCenteredRadiusScalarObjectiveReport,
    pub softmax_temperature: f32,
    pub samples: usize,
    pub candidates: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedSignedCandidateUnitMixObjectiveReport {
    pub loss: f32,
    pub prediction_radius: f32,
    pub target_radius: f32,
    pub radius_ratio: f32,
    pub entropy: f32,
    pub max_weight: f32,
    pub true_weight: f32,
    pub samples: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedSignedCandidateUnitMixHeadIntegration {
    pub learned_mix: ProjectedSignedPredictionCounterfactualMetrics,
    pub objective_report: ProjectedSignedCandidateUnitMixObjectiveReport,
    pub softmax_temperature: f32,
    pub samples: usize,
    pub candidates: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedSignedCandidateSelectorProbeMetrics {
    pub loss: f32,
    pub mrr: f32,
    pub top1: f32,
    pub true_probability: f32,
    pub entropy: f32,
    pub samples: usize,
    pub candidates: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedSignedCandidateSelectorProbe {
    pub prior_anchor: ProjectedSignedCandidateSelectorProbeMetrics,
    pub support_trained_probe: ProjectedSignedCandidateSelectorProbeMetrics,
    pub trained_probe: ProjectedSignedCandidateSelectorProbeMetrics,
    pub softmax_temperature: f32,
    pub probe_steps: usize,
    pub learning_rate: f32,
    pub support_samples: usize,
    pub query_samples: usize,
    pub candidates: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedSignedCandidateSelectorHeadObjectiveReport {
    pub loss: f32,
    pub cross_entropy_loss: f32,
    pub entropy_regularization_loss: f32,
    pub kl_to_prior_loss: f32,
    pub entropy: f32,
    pub true_probability: f32,
    pub max_probability: f32,
    pub samples: usize,
    pub candidates: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedSignedCandidateSelectorOutputCouplingReport {
    pub loss: f32,
    pub selector_max_probability: f32,
    pub active_rate: f32,
    pub base_prediction_top1: f32,
    pub base_prediction_margin: f32,
    pub base_prediction_norm_ratio: f32,
    pub selector_hard_full_top1: f32,
    pub selector_hard_full_margin: f32,
    pub selector_hard_full_norm_ratio: f32,
    pub active_samples: usize,
    pub samples: usize,
    pub candidates: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedSignedCandidateSelectorReadoutDiagnostics {
    pub base_prediction: ProjectedSignedPredictionCounterfactualMetrics,
    pub selector_soft_unit_mix: ProjectedSignedPredictionCounterfactualMetrics,
    pub selector_soft_full: ProjectedSignedPredictionCounterfactualMetrics,
    pub selector_hard_full: ProjectedSignedPredictionCounterfactualMetrics,
    pub selector_soft_radius: ProjectedSignedPredictionCounterfactualMetrics,
    pub selector_hard_radius: ProjectedSignedPredictionCounterfactualMetrics,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedSignedCandidateSelectorHeadIntegration {
    pub readout_diagnostics: ProjectedSignedCandidateSelectorReadoutDiagnostics,
    pub objective_report: ProjectedSignedCandidateSelectorHeadObjectiveReport,
    pub softmax_temperature: f32,
    pub selector_steps: usize,
    pub learning_rate: f32,
    pub entropy_regularization_weight: f32,
    pub entropy_floor: f32,
    pub kl_to_prior_weight: f32,
    pub support_samples: usize,
    pub query_samples: usize,
    pub samples: usize,
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
struct SignedPredictionRayBoundarySampleOutcome {
    true_dx: isize,
    current_radius: f32,
    required_radius: f32,
    upper_radius: f32,
    feasible: bool,
    satisfied: bool,
    failure_kind: SignedPredictionRayBoundaryFailureKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SignedPredictionRayBoundaryFailureKind {
    Satisfied,
    Infeasible,
    BelowLower,
    UpperOvershoot,
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
struct SignedCandidateCentroidIntegrationSampleOutcome {
    mean_radius: SignedPredictionCounterfactualSampleOutcome,
    nearest_unit_radius: SignedPredictionCounterfactualSampleOutcome,
    softmax_radius: SignedPredictionCounterfactualSampleOutcome,
}

#[derive(Debug, Clone, Copy)]
struct SignedCandidateRadiusHeadIntegrationSampleOutcome {
    learned_radius: SignedPredictionCounterfactualSampleOutcome,
}

#[derive(Debug, Clone, Copy)]
struct SignedCandidateUnitMixHeadIntegrationSampleOutcome {
    learned_mix: SignedPredictionCounterfactualSampleOutcome,
}

#[derive(Debug, Clone, Copy)]
struct SignedCandidateSelectorHeadIntegrationSampleOutcome {
    base_prediction: SignedPredictionCounterfactualSampleOutcome,
    selector_soft_unit_mix: SignedPredictionCounterfactualSampleOutcome,
    selector_soft_full: SignedPredictionCounterfactualSampleOutcome,
    selector_hard_full: SignedPredictionCounterfactualSampleOutcome,
    selector_soft_radius: SignedPredictionCounterfactualSampleOutcome,
    selector_hard_radius: SignedPredictionCounterfactualSampleOutcome,
}

#[derive(Debug, Default)]
struct SignedCandidateSelectorProbeMetricTotals {
    loss: f32,
    reciprocal_rank: f32,
    top1: usize,
    true_probability: f32,
    entropy: f32,
    samples: usize,
}

#[derive(Debug, Default)]
struct SignedCandidateSelectorHeadObjectiveReportTotals {
    loss: f32,
    cross_entropy_loss: f32,
    entropy_regularization_loss: f32,
    kl_to_prior_loss: f32,
    entropy: f32,
    true_probability: f32,
    max_probability: f32,
    samples: usize,
}

#[derive(Debug, Default)]
struct SignedCandidateSelectorOutputCouplingReportTotals {
    loss: f32,
    selector_max_probability: f32,
    active_rate: f32,
    base_prediction_top1: f32,
    base_prediction_margin: f32,
    base_prediction_norm_ratio: f32,
    selector_hard_full_top1: f32,
    selector_hard_full_margin: f32,
    selector_hard_full_norm_ratio: f32,
    active_samples: usize,
    samples: usize,
}

#[derive(Debug, Clone)]
struct SignedCandidateUnitMixComponents {
    centered_prediction: Vec<f32>,
    mixed_unit: Vec<f32>,
    mixed_unit_norm: f32,
    mixed_radius: f32,
    predicted_radius: f32,
    weights: Vec<f32>,
    candidate_radii: Vec<f32>,
    candidate_units: Vec<Vec<f32>>,
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
struct SignedPredictionRayBoundaryTotals {
    current_radius: f32,
    required_radius: f32,
    upper_radius: f32,
    radius_margin: f32,
    radius_shortfall: f32,
    radius_overshoot: f32,
    finite_upper_samples: usize,
    satisfied_count: usize,
    infeasible_count: usize,
    feasible_samples: usize,
    samples: usize,
    satisfied_by_dx: [usize; 4],
    infeasible_by_dx: [usize; 4],
    below_lower_by_dx: [usize; 4],
    upper_overshoot_by_dx: [usize; 4],
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

impl SignedPredictionRayBoundaryTotals {
    fn observe(&mut self, outcome: SignedPredictionRayBoundarySampleOutcome) {
        assert!(
            outcome.current_radius.is_finite()
                && outcome.current_radius >= 0.0
                && outcome.required_radius.is_finite()
                && outcome.required_radius >= 0.0
                && (outcome.upper_radius.is_finite()
                    || outcome.upper_radius.is_infinite()
                        && outcome.upper_radius.is_sign_positive()),
            "signed prediction ray-boundary produced non-finite metrics"
        );

        self.samples += 1;
        self.current_radius += outcome.current_radius;
        let dx_index = signed_velocity_candidate_index(outcome.true_dx);
        match outcome.failure_kind {
            SignedPredictionRayBoundaryFailureKind::Satisfied => {
                self.satisfied_by_dx[dx_index] += 1;
            }
            SignedPredictionRayBoundaryFailureKind::Infeasible => {
                self.infeasible_by_dx[dx_index] += 1;
            }
            SignedPredictionRayBoundaryFailureKind::BelowLower => {
                self.below_lower_by_dx[dx_index] += 1;
            }
            SignedPredictionRayBoundaryFailureKind::UpperOvershoot => {
                self.upper_overshoot_by_dx[dx_index] += 1;
            }
        }

        if outcome.feasible {
            let radius_margin = outcome.current_radius - outcome.required_radius;
            self.feasible_samples += 1;
            self.required_radius += outcome.required_radius;
            self.radius_margin += radius_margin;
            self.radius_shortfall += (-radius_margin).max(0.0);
            if outcome.upper_radius.is_finite() {
                self.upper_radius += outcome.upper_radius;
                self.radius_overshoot += (outcome.current_radius - outcome.upper_radius).max(0.0);
                self.finite_upper_samples += 1;
            }
            self.satisfied_count += usize::from(outcome.satisfied);
        } else {
            self.infeasible_count += 1;
        }
    }

    fn into_boundary(self) -> ProjectedSignedPredictionRayBoundary {
        assert!(
            self.samples > 0,
            "signed prediction ray-boundary has no samples"
        );

        let feasible_samples = self.feasible_samples as f32;
        let feasible_denominator = feasible_samples.max(1.0);
        let finite_upper_samples = self.finite_upper_samples as f32;
        let finite_upper_denominator = finite_upper_samples.max(1.0);

        ProjectedSignedPredictionRayBoundary {
            current_radius: self.current_radius / self.samples as f32,
            required_radius: self.required_radius / feasible_denominator,
            upper_radius: self.upper_radius / finite_upper_denominator,
            radius_margin: self.radius_margin / feasible_denominator,
            radius_shortfall: self.radius_shortfall / feasible_denominator,
            radius_overshoot: self.radius_overshoot / finite_upper_denominator,
            satisfied_rate: self.satisfied_count as f32 / self.samples as f32,
            infeasible_rate: self.infeasible_count as f32 / self.samples as f32,
            finite_upper_samples: self.finite_upper_samples,
            feasible_samples: self.feasible_samples,
            samples: self.samples,
            candidates: SIGNED_VELOCITY_BANK_CANDIDATE_DX.len(),
            satisfied_by_dx: self.satisfied_by_dx,
            infeasible_by_dx: self.infeasible_by_dx,
            below_lower_by_dx: self.below_lower_by_dx,
            upper_overshoot_by_dx: self.upper_overshoot_by_dx,
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

impl SignedCandidateSelectorProbeMetricTotals {
    fn observe_logits(&mut self, logits: &Tensor, true_candidate_indices: &[usize]) {
        assert!(
            logits.shape.len() == 2,
            "candidate selector probe logits must be rank-2, got {:?}",
            logits.shape
        );
        let batch_size = logits.shape[0];
        let candidates = logits.shape[1];
        assert_eq!(
            true_candidate_indices.len(),
            batch_size,
            "candidate selector probe label batch mismatch"
        );
        assert!(candidates > 0, "candidate selector probe has no candidates");

        for (sample, true_index) in true_candidate_indices.iter().copied().enumerate() {
            assert!(
                true_index < candidates,
                "candidate selector probe true index {} exceeds candidate count {}",
                true_index,
                candidates
            );

            let sample_logits = (0..candidates)
                .map(|candidate_idx| logits.get(&[sample, candidate_idx]))
                .collect::<Vec<_>>();
            let probabilities = stable_softmax(&sample_logits, "candidate selector probe");
            let true_probability = probabilities[true_index];
            let loss_probability = true_probability.max(VELOCITY_BANK_TIE_EPSILON);
            let true_logit = sample_logits[true_index];
            let mut rank = 1usize;
            let mut entropy = 0.0f32;

            for (candidate_idx, probability) in probabilities.iter().enumerate() {
                if *probability > VELOCITY_BANK_TIE_EPSILON {
                    entropy -= probability * probability.ln();
                }
                if candidate_idx != true_index
                    && sample_logits[candidate_idx] >= true_logit - VELOCITY_BANK_TIE_EPSILON
                {
                    rank += 1;
                }
            }

            self.loss += -loss_probability.ln();
            self.reciprocal_rank += 1.0 / rank as f32;
            self.top1 += usize::from(rank == 1);
            self.true_probability += true_probability;
            self.entropy += entropy;
            self.samples += 1;
        }
    }

    fn into_metrics(self, candidates: usize) -> ProjectedSignedCandidateSelectorProbeMetrics {
        assert!(
            self.samples > 0,
            "candidate selector probe metrics have no samples"
        );

        ProjectedSignedCandidateSelectorProbeMetrics {
            loss: self.loss / self.samples as f32,
            mrr: self.reciprocal_rank / self.samples as f32,
            top1: self.top1 as f32 / self.samples as f32,
            true_probability: self.true_probability / self.samples as f32,
            entropy: self.entropy / self.samples as f32,
            samples: self.samples,
            candidates,
        }
    }
}

impl SignedCandidateSelectorHeadObjectiveReportTotals {
    fn observe(&mut self, report: ProjectedSignedCandidateSelectorHeadObjectiveReport) {
        self.loss += report.loss * report.samples as f32;
        self.cross_entropy_loss += report.cross_entropy_loss * report.samples as f32;
        self.entropy_regularization_loss +=
            report.entropy_regularization_loss * report.samples as f32;
        self.kl_to_prior_loss += report.kl_to_prior_loss * report.samples as f32;
        self.entropy += report.entropy * report.samples as f32;
        self.true_probability += report.true_probability * report.samples as f32;
        self.max_probability += report.max_probability * report.samples as f32;
        self.samples += report.samples;
    }

    fn into_report(self, candidates: usize) -> ProjectedSignedCandidateSelectorHeadObjectiveReport {
        assert!(
            self.samples > 0,
            "candidate selector head objective has no samples"
        );

        ProjectedSignedCandidateSelectorHeadObjectiveReport {
            loss: self.loss / self.samples as f32,
            cross_entropy_loss: self.cross_entropy_loss / self.samples as f32,
            entropy_regularization_loss: self.entropy_regularization_loss / self.samples as f32,
            kl_to_prior_loss: self.kl_to_prior_loss / self.samples as f32,
            entropy: self.entropy / self.samples as f32,
            true_probability: self.true_probability / self.samples as f32,
            max_probability: self.max_probability / self.samples as f32,
            samples: self.samples,
            candidates,
        }
    }
}

impl SignedCandidateSelectorOutputCouplingReportTotals {
    fn observe(&mut self, report: ProjectedSignedCandidateSelectorOutputCouplingReport) {
        assert!(
            report.loss.is_finite()
                && report.selector_max_probability.is_finite()
                && report.active_rate.is_finite()
                && report.base_prediction_top1.is_finite()
                && report.base_prediction_margin.is_finite()
                && report.base_prediction_norm_ratio.is_finite()
                && report.selector_hard_full_top1.is_finite()
                && report.selector_hard_full_margin.is_finite()
                && report.selector_hard_full_norm_ratio.is_finite(),
            "candidate selector output coupling report has non-finite fields"
        );
        assert!(
            report.samples > 0,
            "candidate selector output coupling report has no samples"
        );
        assert!(
            report.active_samples <= report.samples,
            "candidate selector output coupling report active samples exceed sample count"
        );

        let samples = report.samples as f32;
        self.loss += report.loss * samples;
        self.selector_max_probability += report.selector_max_probability * samples;
        self.active_rate += report.active_rate * samples;
        self.base_prediction_top1 += report.base_prediction_top1 * samples;
        self.base_prediction_margin += report.base_prediction_margin * samples;
        self.base_prediction_norm_ratio += report.base_prediction_norm_ratio * samples;
        self.selector_hard_full_top1 += report.selector_hard_full_top1 * samples;
        self.selector_hard_full_margin += report.selector_hard_full_margin * samples;
        self.selector_hard_full_norm_ratio += report.selector_hard_full_norm_ratio * samples;
        self.active_samples += report.active_samples;
        self.samples += report.samples;
    }

    fn into_report(
        self,
        candidates: usize,
    ) -> ProjectedSignedCandidateSelectorOutputCouplingReport {
        assert!(
            self.samples > 0,
            "candidate selector output coupling report has no samples"
        );

        let samples = self.samples as f32;
        ProjectedSignedCandidateSelectorOutputCouplingReport {
            loss: self.loss / samples,
            selector_max_probability: self.selector_max_probability / samples,
            active_rate: self.active_rate / samples,
            base_prediction_top1: self.base_prediction_top1 / samples,
            base_prediction_margin: self.base_prediction_margin / samples,
            base_prediction_norm_ratio: self.base_prediction_norm_ratio / samples,
            selector_hard_full_top1: self.selector_hard_full_top1 / samples,
            selector_hard_full_margin: self.selector_hard_full_margin / samples,
            selector_hard_full_norm_ratio: self.selector_hard_full_norm_ratio / samples,
            active_samples: self.active_samples,
            samples: self.samples,
            candidates,
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

#[allow(dead_code)]
pub fn projected_signed_prediction_ray_boundary<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
) -> ProjectedSignedPredictionRayBoundary
where
    P: PredictorModule,
{
    let outcomes = projected_signed_prediction_ray_boundary_sample_outcomes(model, x_t, x_t1);
    let mut totals = SignedPredictionRayBoundaryTotals::default();

    for outcome in outcomes {
        totals.observe(outcome);
    }

    totals.into_boundary()
}

pub fn projected_signed_prediction_ray_boundary_from_base_seed<P>(
    model: &ProjectedVisionJepa<P>,
    validation_base_seed: u64,
    validation_batches: usize,
) -> ProjectedSignedPredictionRayBoundary
where
    P: PredictorModule,
{
    assert!(
        validation_batches > 0,
        "validation_batches must be greater than 0"
    );

    let mut totals = SignedPredictionRayBoundaryTotals::default();

    for batch_idx in 0..validation_batches {
        let (x_t, x_t1) = make_temporal_batch_for_task(
            BATCH_SIZE,
            validation_base_seed + batch_idx as u64,
            TemporalTaskMode::SignedVelocityTrail,
        );
        let outcomes = projected_signed_prediction_ray_boundary_sample_outcomes(model, &x_t, &x_t1);

        for outcome in outcomes {
            totals.observe(outcome);
        }
    }

    totals.into_boundary()
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

pub fn projected_signed_candidate_centroid_integration_from_base_seed<P>(
    model: &ProjectedVisionJepa<P>,
    validation_base_seed: u64,
    validation_batches: usize,
    softmax_temperature: f32,
) -> ProjectedSignedCandidateCentroidIntegration
where
    P: PredictorModule,
{
    assert!(
        validation_batches > 0,
        "candidate-centroid integration requires at least one validation batch"
    );
    assert!(
        softmax_temperature.is_finite() && softmax_temperature > 0.0,
        "candidate-centroid integration temperature must be finite and positive"
    );

    let mut mean_radius_totals = SignedPredictionCounterfactualMetricTotals::default();
    let mut nearest_unit_radius_totals = SignedPredictionCounterfactualMetricTotals::default();
    let mut softmax_radius_totals = SignedPredictionCounterfactualMetricTotals::default();

    for batch_idx in 0..validation_batches {
        let (x_t, x_t1) = make_temporal_batch_for_task(
            BATCH_SIZE,
            validation_base_seed + batch_idx as u64,
            TemporalTaskMode::SignedVelocityTrail,
        );
        let outcomes = projected_signed_candidate_centroid_integration_sample_outcomes(
            model,
            &x_t,
            &x_t1,
            softmax_temperature,
        );

        for outcome in outcomes {
            mean_radius_totals.observe(outcome.mean_radius);
            nearest_unit_radius_totals.observe(outcome.nearest_unit_radius);
            softmax_radius_totals.observe(outcome.softmax_radius);
        }
    }

    let samples = mean_radius_totals.samples;
    assert_eq!(
        samples, nearest_unit_radius_totals.samples,
        "candidate-centroid nearest radius sample mismatch"
    );
    assert_eq!(
        samples, softmax_radius_totals.samples,
        "candidate-centroid softmax radius sample mismatch"
    );

    ProjectedSignedCandidateCentroidIntegration {
        mean_radius: mean_radius_totals.into_metrics(),
        nearest_unit_radius: nearest_unit_radius_totals.into_metrics(),
        softmax_radius: softmax_radius_totals.into_metrics(),
        softmax_temperature,
        samples,
        candidates: SIGNED_VELOCITY_BANK_CANDIDATE_DX.len(),
    }
}

pub fn projected_signed_candidate_radius_delta_loss_and_grad<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
    radius_delta: &Tensor,
    softmax_temperature: f32,
) -> (SignedCenteredRadiusScalarObjectiveReport, Tensor)
where
    P: PredictorModule,
{
    assert_eq!(
        x_t.shape, x_t1.shape,
        "candidate radius delta objective expects matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "candidate radius delta objective expects rank-4 temporal batches, got {:?}",
        x_t.shape
    );
    assert!(
        softmax_temperature.is_finite() && softmax_temperature > 0.0,
        "candidate radius delta objective temperature must be finite and positive"
    );

    let prediction = model.predict_next_projection(x_t);
    let candidate_targets = signed_velocity_candidate_target_projections(model, x_t);
    let true_candidate_indices = signed_velocity_true_candidate_indices(x_t, x_t1);
    let anchor_radius = signed_candidate_softmax_anchor_radius(
        &prediction,
        &candidate_targets,
        softmax_temperature,
    );
    let predicted_radius = radius_delta_to_centered_radius(radius_delta, &anchor_radius);
    let (report, grad_radius) = signed_centered_radius_scalar_loss_and_grad(
        &predicted_radius,
        &candidate_targets,
        &true_candidate_indices,
    );
    let grad_delta =
        centered_radius_grad_to_delta_grad(radius_delta, &predicted_radius, &grad_radius);

    (report, grad_delta)
}

pub fn projected_signed_candidate_radius_head_features<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    softmax_temperature: f32,
) -> Tensor
where
    P: PredictorModule,
{
    assert!(
        x_t.shape.len() == 4,
        "candidate radius head features expect rank-4 temporal batches, got {:?}",
        x_t.shape
    );
    assert!(
        softmax_temperature.is_finite() && softmax_temperature > 0.0,
        "candidate radius head feature temperature must be finite and positive"
    );

    let projection = model.project_latent(x_t);
    let prediction = model.predict_next_projection(x_t);
    let candidate_targets = signed_velocity_candidate_target_projections(model, x_t);

    signed_candidate_radius_head_features_from_prediction(
        &projection,
        &prediction,
        &candidate_targets,
        softmax_temperature,
    )
}

pub fn projected_signed_candidate_radius_logit_head_features<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    softmax_temperature: f32,
) -> Tensor
where
    P: PredictorModule,
{
    assert!(
        x_t.shape.len() == 4,
        "candidate radius logit head features expect rank-4 temporal batches, got {:?}",
        x_t.shape
    );
    assert!(
        softmax_temperature.is_finite() && softmax_temperature > 0.0,
        "candidate radius logit head feature temperature must be finite and positive"
    );

    let projection = model.project_latent(x_t);
    let prediction = model.predict_next_projection(x_t);
    let candidate_targets = signed_velocity_candidate_target_projections(model, x_t);

    signed_candidate_radius_logit_head_features_from_prediction(
        &projection,
        &prediction,
        &candidate_targets,
        softmax_temperature,
    )
}

pub fn projected_signed_candidate_unit_mix_head_features<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    softmax_temperature: f32,
) -> Tensor
where
    P: PredictorModule,
{
    assert!(
        x_t.shape.len() == 4,
        "candidate unit-mix head features expect rank-4 temporal batches, got {:?}",
        x_t.shape
    );
    assert!(
        softmax_temperature.is_finite() && softmax_temperature > 0.0,
        "candidate unit-mix head feature temperature must be finite and positive"
    );

    let projection = model.project_latent(x_t);
    let prediction = model.predict_next_projection(x_t);
    let candidate_targets = signed_velocity_candidate_target_projections(model, x_t);

    signed_candidate_radius_logit_head_features_from_prediction(
        &projection,
        &prediction,
        &candidate_targets,
        softmax_temperature,
    )
}

pub fn projected_signed_candidate_selector_head_features<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    softmax_temperature: f32,
) -> Tensor
where
    P: PredictorModule,
{
    assert!(
        x_t.shape.len() == 4,
        "candidate selector head features expect rank-4 temporal batches, got {:?}",
        x_t.shape
    );

    let candidates = SIGNED_VELOCITY_BANK_CANDIDATE_DX.len();
    assert_eq!(
        candidates, 4,
        "candidate selector head expects signed-velocity-trail K=4"
    );

    let features =
        projected_signed_candidate_unit_mix_head_features(model, x_t, softmax_temperature);
    normalize_signed_candidate_selector_batch_features(&features)
}

pub fn projected_signed_candidate_radius_logit_mixing_loss_and_grad<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
    residual_logits: &Tensor,
    softmax_temperature: f32,
) -> (SignedCenteredRadiusScalarObjectiveReport, Tensor)
where
    P: PredictorModule,
{
    assert_eq!(
        x_t.shape, x_t1.shape,
        "candidate radius logit objective expects matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "candidate radius logit objective expects rank-4 temporal batches, got {:?}",
        x_t.shape
    );
    assert!(
        softmax_temperature.is_finite() && softmax_temperature > 0.0,
        "candidate radius logit objective temperature must be finite and positive"
    );

    let prediction = model.predict_next_projection(x_t);
    let candidate_targets = signed_velocity_candidate_target_projections(model, x_t);
    let true_candidate_indices = signed_velocity_true_candidate_indices(x_t, x_t1);
    let (predicted_radius, weights, candidate_radii) = radius_logits_to_centered_radius_and_weights(
        residual_logits,
        &prediction,
        &candidate_targets,
        softmax_temperature,
    );
    let (report, grad_radius) = signed_centered_radius_scalar_loss_and_grad(
        &predicted_radius,
        &candidate_targets,
        &true_candidate_indices,
    );
    let grad_logits = centered_radius_grad_to_residual_logits_grad(
        &predicted_radius,
        &grad_radius,
        &weights,
        &candidate_radii,
    );

    (report, grad_logits)
}

pub fn projected_signed_candidate_unit_mix_loss_and_grad<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
    residual_logits: &Tensor,
    softmax_temperature: f32,
) -> (ProjectedSignedCandidateUnitMixObjectiveReport, Tensor)
where
    P: PredictorModule,
{
    assert_eq!(
        x_t.shape, x_t1.shape,
        "candidate unit-mix objective expects matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "candidate unit-mix objective expects rank-4 temporal batches, got {:?}",
        x_t.shape
    );
    assert!(
        softmax_temperature.is_finite() && softmax_temperature > 0.0,
        "candidate unit-mix objective temperature must be finite and positive"
    );

    let prediction = model.predict_next_projection(x_t);
    let candidate_targets = signed_velocity_candidate_target_projections(model, x_t);
    let true_candidate_indices = signed_velocity_true_candidate_indices(x_t, x_t1);
    candidate_unit_mix_loss_and_grad_from_prediction(
        &prediction,
        &candidate_targets,
        &true_candidate_indices,
        residual_logits,
        softmax_temperature,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn projected_signed_candidate_selector_head_loss_and_grad<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
    selector_logits: &Tensor,
    softmax_temperature: f32,
    entropy_floor: f32,
    entropy_regularization_weight: f32,
    kl_to_prior_weight: f32,
) -> (ProjectedSignedCandidateSelectorHeadObjectiveReport, Tensor)
where
    P: PredictorModule,
{
    assert_eq!(
        x_t.shape, x_t1.shape,
        "candidate selector head objective expects matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "candidate selector head objective expects rank-4 temporal batches, got {:?}",
        x_t.shape
    );
    assert!(
        softmax_temperature.is_finite() && softmax_temperature > 0.0,
        "candidate selector head objective temperature must be finite and positive"
    );
    assert!(
        entropy_floor.is_finite() && entropy_floor >= 0.0,
        "candidate selector head entropy floor must be finite and non-negative"
    );
    assert!(
        entropy_regularization_weight.is_finite() && entropy_regularization_weight >= 0.0,
        "candidate selector head entropy regularization weight must be finite and non-negative"
    );
    assert!(
        kl_to_prior_weight.is_finite() && kl_to_prior_weight >= 0.0,
        "candidate selector head KL-to-prior weight must be finite and non-negative"
    );

    let candidates = SIGNED_VELOCITY_BANK_CANDIDATE_DX.len();
    assert_eq!(
        candidates, 4,
        "candidate selector head expects signed-velocity-trail K=4"
    );
    assert_eq!(
        selector_logits.shape,
        vec![x_t.shape[0], candidates],
        "candidate selector head logits shape mismatch"
    );

    let prior_features =
        projected_signed_candidate_unit_mix_head_features(model, x_t, softmax_temperature);
    let prior_logits =
        signed_candidate_selector_prior_logits_from_features(&prior_features, candidates);
    let true_candidate_indices = signed_velocity_true_candidate_indices(x_t, x_t1);

    signed_candidate_selector_head_objective_loss_and_grad_from_logits(
        selector_logits,
        &prior_logits,
        &true_candidate_indices,
        entropy_floor,
        entropy_regularization_weight,
        kl_to_prior_weight,
    )
}

pub fn projected_signed_candidate_selector_hard_full_output_coupling_loss_and_grad<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
    selector_logits: &Tensor,
    min_confidence: f32,
) -> (ProjectedSignedCandidateSelectorOutputCouplingReport, Tensor)
where
    P: PredictorModule,
{
    assert_eq!(
        x_t.shape, x_t1.shape,
        "candidate selector output coupling expects matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "candidate selector output coupling expects rank-4 temporal batches, got {:?}",
        x_t.shape
    );
    assert!(
        min_confidence.is_finite() && (0.0..=1.0).contains(&min_confidence),
        "candidate selector output coupling min confidence must be in [0, 1]"
    );

    let prediction = model.predict_next_projection(x_t);
    let (hard_full_target, active_samples) = signed_candidate_selector_gated_hard_full_targets(
        model,
        x_t,
        selector_logits,
        &prediction,
        None,
        false,
        min_confidence,
    );
    let loss = mse_loss(&prediction, &hard_full_target);
    let grad = mse_loss_grad(&prediction, &hard_full_target);
    let report = projected_signed_candidate_selector_hard_full_output_coupling_report_from_logits(
        model,
        x_t,
        x_t1,
        selector_logits,
        loss,
        active_samples,
    );

    (report, grad)
}

pub fn projected_signed_true_target_mse_amplification_loss_and_grad<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
) -> (f32, Tensor)
where
    P: PredictorModule,
{
    assert_eq!(
        x_t.shape, x_t1.shape,
        "signed true-target MSE amplification expects matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "signed true-target MSE amplification expects rank-4 temporal batches, got {:?}",
        x_t.shape
    );

    let prediction = model.predict_next_projection(x_t);
    let target = model.target_projection(x_t1);
    (
        mse_loss(&prediction, &target),
        mse_loss_grad(&prediction, &target),
    )
}

pub fn projected_signed_ray_direction_repair_loss_and_grad<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
    denominator_margin: f32,
) -> (ProjectedSignedRayDirectionRepairReport, Tensor)
where
    P: PredictorModule,
{
    assert_eq!(
        x_t.shape, x_t1.shape,
        "signed ray-direction repair expects matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "signed ray-direction repair expects rank-4 temporal batches, got {:?}",
        x_t.shape
    );
    assert!(
        denominator_margin.is_finite() && denominator_margin >= 0.0,
        "signed ray-direction repair margin must be finite and non-negative"
    );

    let prediction = model.predict_next_projection(x_t);
    let candidate_targets = signed_velocity_candidate_target_projections(model, x_t);
    let batch_size = prediction.shape[0];
    let projection_dim = prediction.shape[1];
    let mut grad = Tensor::zeros(prediction.shape.clone());
    let mut gap_loss_sum = 0.0f32;
    let mut parallel_loss_sum = 0.0f32;
    let mut cosine_sum = 0.0f32;
    let mut current_radius_sum = 0.0f32;
    let mut target_radius_sum = 0.0f32;
    let mut active_count = 0usize;
    let mut gap_active_count = 0usize;
    let mut parallel_active_count = 0usize;
    let mut zero_direction_skipped = 0usize;

    for sample in 0..batch_size {
        let true_dx = signed_motion_dx_for_sample(x_t, x_t1, sample);
        let true_index = signed_velocity_candidate_index(true_dx);
        let centroid = signed_candidate_centroid(&candidate_targets, sample, projection_dim);
        let prediction_centered = centered_sample_vector(&prediction, sample, &centroid);
        let prediction_unit = unit_vector(&prediction_centered);
        let current_radius = vector_l2_norm(&prediction_centered);
        let true_centered =
            centered_sample_vector(&candidate_targets[true_index], sample, &centroid);
        let true_radius = vector_l2_norm(&true_centered);
        let true_unit = unit_vector(&true_centered);

        assert!(
            current_radius.is_finite()
                && true_radius.is_finite()
                && vector_squared_norm(&true_centered).is_finite(),
            "signed ray-direction repair produced non-finite geometry"
        );

        cosine_sum += vector_dot(&prediction_unit, &true_unit).clamp(-1.0, 1.0);
        current_radius_sum += current_radius;
        target_radius_sum += true_radius;

        if current_radius <= VELOCITY_BANK_TIE_EPSILON {
            zero_direction_skipped += 1;
            continue;
        }

        let true_norm_sq = vector_squared_norm(&true_centered);
        let denominator_threshold = denominator_margin.max(VELOCITY_BANK_TIE_EPSILON);
        let mut lower_radius = 0.0f32;
        let mut upper_radius = f32::INFINITY;
        let mut lower_grad = vec![0.0f32; projection_dim];
        let mut upper_grad = vec![0.0f32; projection_dim];
        let mut parallel_grad = vec![0.0f32; projection_dim];
        let mut sample_parallel_loss = 0.0f32;
        let mut sample_parallel_active_count = 0usize;
        let wrong_count = (candidate_targets.len() - 1) as f32;

        for (candidate_index, candidate_target) in candidate_targets.iter().enumerate() {
            if candidate_index == true_index {
                continue;
            }

            let wrong_centered = centered_sample_vector(candidate_target, sample, &centroid);
            let wrong_norm_sq = vector_squared_norm(&wrong_centered);
            let true_minus_wrong = true_centered
                .iter()
                .zip(&wrong_centered)
                .map(|(true_value, wrong_value)| true_value - wrong_value)
                .collect::<Vec<_>>();
            let denominator = 2.0 * vector_dot(&prediction_unit, &true_minus_wrong);
            let numerator = true_norm_sq - wrong_norm_sq;

            if denominator > denominator_threshold {
                let candidate_lower = numerator / denominator;
                if candidate_lower > lower_radius {
                    lower_radius = candidate_lower;
                    for (feature_idx, direction_delta) in true_minus_wrong.iter().enumerate() {
                        lower_grad[feature_idx] =
                            -2.0 * numerator * direction_delta / (denominator * denominator);
                    }
                }
            } else if denominator < -denominator_threshold {
                let candidate_upper = numerator / denominator;
                if candidate_upper < upper_radius {
                    upper_radius = candidate_upper;
                    for (feature_idx, direction_delta) in true_minus_wrong.iter().enumerate() {
                        upper_grad[feature_idx] =
                            -2.0 * numerator * direction_delta / (denominator * denominator);
                    }
                }
            } else if numerator >= -VELOCITY_BANK_TIE_EPSILON {
                let hinge = denominator_margin - denominator;
                if hinge > 0.0 {
                    sample_parallel_active_count += 1;
                    sample_parallel_loss += hinge * hinge / wrong_count;
                    for (feature_idx, direction_delta) in true_minus_wrong.iter().enumerate() {
                        parallel_grad[feature_idx] += -4.0 * hinge * direction_delta / wrong_count;
                    }
                }
            }
        }

        let mut sample_grad_u = parallel_grad;
        let mut sample_active = sample_parallel_active_count > 0;
        if sample_parallel_active_count > 0 {
            parallel_active_count += sample_parallel_active_count;
            parallel_loss_sum += sample_parallel_loss;
        }

        if upper_radius.is_finite() && lower_radius + VELOCITY_BANK_TIE_EPSILON >= upper_radius {
            let interval_gap = lower_radius - upper_radius;
            let interval_scale = 1.0 + lower_radius.abs() + upper_radius.abs();
            let hinge = interval_gap / interval_scale;
            gap_active_count += 1;
            sample_active = true;
            gap_loss_sum += hinge * hinge;
            for feature_idx in 0..projection_dim {
                sample_grad_u[feature_idx] +=
                    2.0 * hinge * (lower_grad[feature_idx] - upper_grad[feature_idx])
                        / interval_scale;
            }
        }

        if !sample_active {
            continue;
        }

        active_count += 1;
        let sample_grad_norm = vector_l2_norm(&sample_grad_u);
        if sample_grad_norm > RAY_DIRECTION_REPAIR_GRAD_CLIP {
            let clip_scale = RAY_DIRECTION_REPAIR_GRAD_CLIP / sample_grad_norm;
            for value in &mut sample_grad_u {
                *value *= clip_scale;
            }
        }

        let radial_grad = vector_dot(&sample_grad_u, &prediction_unit);
        for feature_idx in 0..projection_dim {
            let tangent_grad = (sample_grad_u[feature_idx]
                - radial_grad * prediction_unit[feature_idx])
                / current_radius;
            let grad_value = grad.get(&[sample, feature_idx]) + tangent_grad / batch_size as f32;
            grad.set(&[sample, feature_idx], grad_value);
        }
    }

    let loss = (gap_loss_sum + parallel_loss_sum) / batch_size as f32;
    (
        ProjectedSignedRayDirectionRepairReport {
            loss,
            gap_loss: gap_loss_sum / batch_size as f32,
            parallel_loss: parallel_loss_sum / batch_size as f32,
            active_rate: active_count as f32 / batch_size as f32,
            cosine: cosine_sum / batch_size as f32,
            current_radius: current_radius_sum / batch_size as f32,
            target_radius: target_radius_sum / batch_size as f32,
            samples: batch_size,
            active_count,
            gap_active_count,
            parallel_active_count,
            zero_direction_skipped,
        },
        grad,
    )
}

pub fn projected_signed_candidate_selector_stable_hard_full_output_coupling_loss_and_grad<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
    selector_logits: &Tensor,
    min_confidence: f32,
) -> (ProjectedSignedCandidateSelectorOutputCouplingReport, Tensor)
where
    P: PredictorModule,
{
    assert_eq!(
        x_t.shape, x_t1.shape,
        "stable candidate selector output coupling expects matching pair shapes"
    );
    assert!(
        min_confidence.is_finite() && (0.0..=1.0).contains(&min_confidence),
        "stable candidate selector output coupling min confidence must be in [0, 1]"
    );

    let prediction = model.predict_next_projection(x_t);
    let (hard_full_target, active_samples) = signed_candidate_selector_gated_hard_full_targets(
        model,
        x_t,
        selector_logits,
        &prediction,
        Some(x_t1),
        true,
        min_confidence,
    );
    let loss = mse_loss(&prediction, &hard_full_target);
    let grad = mse_loss_grad(&prediction, &hard_full_target);
    let report = projected_signed_candidate_selector_hard_full_output_coupling_report_from_logits(
        model,
        x_t,
        x_t1,
        selector_logits,
        loss,
        active_samples,
    );

    (report, grad)
}

pub fn projected_signed_candidate_selector_active_normalized_stable_hard_full_output_coupling_loss_and_grad<
    P,
>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
    selector_logits: &Tensor,
    min_confidence: f32,
) -> (ProjectedSignedCandidateSelectorOutputCouplingReport, Tensor)
where
    P: PredictorModule,
{
    assert_eq!(
        x_t.shape, x_t1.shape,
        "active-normalized stable selector output coupling expects matching pair shapes"
    );
    assert!(
        min_confidence.is_finite() && (0.0..=1.0).contains(&min_confidence),
        "active-normalized stable selector output coupling min confidence must be in [0, 1]"
    );

    let prediction = model.predict_next_projection(x_t);
    let (hard_full_target, active_samples) = signed_candidate_selector_gated_hard_full_targets(
        model,
        x_t,
        selector_logits,
        &prediction,
        Some(x_t1),
        true,
        min_confidence,
    );
    let normalization = if active_samples > 0 {
        prediction.shape[0] as f32 / active_samples as f32
    } else {
        0.0
    };
    let loss = mse_loss(&prediction, &hard_full_target) * normalization;
    let grad = scale_tensor_values(
        &mse_loss_grad(&prediction, &hard_full_target),
        normalization,
    );
    let report = projected_signed_candidate_selector_hard_full_output_coupling_report_from_logits(
        model,
        x_t,
        x_t1,
        selector_logits,
        loss,
        active_samples,
    );

    (report, grad)
}

pub fn projected_signed_candidate_selector_hard_full_output_coupling_report_from_base_seed<P>(
    model: &ProjectedVisionJepa<P>,
    selector_head: &Linear,
    validation_base_seed: u64,
    validation_batches: usize,
    softmax_temperature: f32,
    min_confidence: f32,
    require_correct_selector: bool,
    active_normalized: bool,
) -> ProjectedSignedCandidateSelectorOutputCouplingReport
where
    P: PredictorModule,
{
    assert!(
        validation_batches > 0,
        "candidate selector output coupling requires at least one validation batch"
    );
    assert!(
        softmax_temperature.is_finite() && softmax_temperature > 0.0,
        "candidate selector output coupling temperature must be finite and positive"
    );
    assert!(
        min_confidence.is_finite() && (0.0..=1.0).contains(&min_confidence),
        "candidate selector output coupling min confidence must be in [0, 1]"
    );
    assert!(
        !active_normalized || require_correct_selector,
        "active-normalized selector output coupling requires correct-selector gating"
    );

    let candidates = SIGNED_VELOCITY_BANK_CANDIDATE_DX.len();
    assert_eq!(
        candidates, 4,
        "candidate selector output coupling expects signed-velocity-trail K=4"
    );

    let (sample_x_t, _) = make_temporal_batch_for_task(
        BATCH_SIZE,
        validation_base_seed,
        TemporalTaskMode::SignedVelocityTrail,
    );
    let sample_features =
        projected_signed_candidate_selector_head_features(model, &sample_x_t, softmax_temperature);
    assert_eq!(
        selector_head.weight.shape,
        vec![sample_features.shape[1], candidates],
        "candidate selector output coupling head weight shape mismatch"
    );
    assert_eq!(
        selector_head.bias.shape,
        vec![candidates],
        "candidate selector output coupling head bias shape mismatch"
    );

    let mut totals = SignedCandidateSelectorOutputCouplingReportTotals::default();

    for batch_idx in 0..validation_batches {
        let (x_t, x_t1) = make_temporal_batch_for_task(
            BATCH_SIZE,
            validation_base_seed + batch_idx as u64,
            TemporalTaskMode::SignedVelocityTrail,
        );
        let features =
            projected_signed_candidate_selector_head_features(model, &x_t, softmax_temperature);
        let selector_logits = selector_head.forward(&features);
        let (report, _) = if active_normalized {
            projected_signed_candidate_selector_active_normalized_stable_hard_full_output_coupling_loss_and_grad(
                model,
                &x_t,
                &x_t1,
                &selector_logits,
                min_confidence,
            )
        } else if require_correct_selector {
            projected_signed_candidate_selector_stable_hard_full_output_coupling_loss_and_grad(
                model,
                &x_t,
                &x_t1,
                &selector_logits,
                min_confidence,
            )
        } else {
            projected_signed_candidate_selector_hard_full_output_coupling_loss_and_grad(
                model,
                &x_t,
                &x_t1,
                &selector_logits,
                min_confidence,
            )
        };

        totals.observe(report);
    }

    totals.into_report(candidates)
}

pub fn projected_signed_candidate_radius_head_integration_from_base_seed<P>(
    model: &ProjectedVisionJepa<P>,
    radius_head: &Linear,
    validation_base_seed: u64,
    validation_batches: usize,
    softmax_temperature: f32,
) -> ProjectedSignedCandidateRadiusHeadIntegration
where
    P: PredictorModule,
{
    assert!(
        validation_batches > 0,
        "candidate radius head integration requires at least one validation batch"
    );
    assert!(
        softmax_temperature.is_finite() && softmax_temperature > 0.0,
        "candidate radius head integration temperature must be finite and positive"
    );

    let mut learned_radius_totals = SignedPredictionCounterfactualMetricTotals::default();
    let mut loss_sum = 0.0f32;
    let mut prediction_radius_sum = 0.0f32;
    let mut target_radius_sum = 0.0f32;
    let mut radius_ratio_sum = 0.0f32;
    let mut samples = 0usize;

    for batch_idx in 0..validation_batches {
        let (x_t, x_t1) = make_temporal_batch_for_task(
            BATCH_SIZE,
            validation_base_seed + batch_idx as u64,
            TemporalTaskMode::SignedVelocityTrail,
        );
        let features =
            projected_signed_candidate_radius_head_features(model, &x_t, softmax_temperature);
        let radius_delta = radius_head.forward(&features);
        let (report, _) = projected_signed_candidate_radius_delta_loss_and_grad(
            model,
            &x_t,
            &x_t1,
            &radius_delta,
            softmax_temperature,
        );
        let outcomes = projected_signed_candidate_radius_head_integration_sample_outcomes(
            model,
            &x_t,
            &x_t1,
            &radius_delta,
            softmax_temperature,
        );

        loss_sum += report.loss * report.samples as f32;
        prediction_radius_sum += report.prediction_radius * report.samples as f32;
        target_radius_sum += report.target_radius * report.samples as f32;
        radius_ratio_sum += report.radius_ratio * report.samples as f32;
        samples += report.samples;

        for outcome in outcomes {
            learned_radius_totals.observe(outcome.learned_radius);
        }
    }

    assert!(samples > 0, "candidate radius head report has no samples");
    assert_eq!(
        samples, learned_radius_totals.samples,
        "candidate radius head sample mismatch"
    );

    ProjectedSignedCandidateRadiusHeadIntegration {
        learned_radius: learned_radius_totals.into_metrics(),
        scalar_report: SignedCenteredRadiusScalarObjectiveReport {
            loss: loss_sum / samples as f32,
            prediction_radius: prediction_radius_sum / samples as f32,
            target_radius: target_radius_sum / samples as f32,
            radius_ratio: radius_ratio_sum / samples as f32,
            samples,
        },
        softmax_temperature,
        samples,
        candidates: SIGNED_VELOCITY_BANK_CANDIDATE_DX.len(),
    }
}

pub fn projected_signed_candidate_radius_logit_head_integration_from_base_seed<P>(
    model: &ProjectedVisionJepa<P>,
    radius_head: &Linear,
    validation_base_seed: u64,
    validation_batches: usize,
    softmax_temperature: f32,
) -> ProjectedSignedCandidateRadiusHeadIntegration
where
    P: PredictorModule,
{
    assert!(
        validation_batches > 0,
        "candidate radius logit head integration requires at least one validation batch"
    );
    assert!(
        softmax_temperature.is_finite() && softmax_temperature > 0.0,
        "candidate radius logit head integration temperature must be finite and positive"
    );

    let mut learned_radius_totals = SignedPredictionCounterfactualMetricTotals::default();
    let mut loss_sum = 0.0f32;
    let mut prediction_radius_sum = 0.0f32;
    let mut target_radius_sum = 0.0f32;
    let mut radius_ratio_sum = 0.0f32;
    let mut samples = 0usize;

    for batch_idx in 0..validation_batches {
        let (x_t, x_t1) = make_temporal_batch_for_task(
            BATCH_SIZE,
            validation_base_seed + batch_idx as u64,
            TemporalTaskMode::SignedVelocityTrail,
        );
        let features =
            projected_signed_candidate_radius_logit_head_features(model, &x_t, softmax_temperature);
        let residual_logits = radius_head.forward(&features);
        let (report, _) = projected_signed_candidate_radius_logit_mixing_loss_and_grad(
            model,
            &x_t,
            &x_t1,
            &residual_logits,
            softmax_temperature,
        );
        let outcomes = projected_signed_candidate_radius_logit_head_integration_sample_outcomes(
            model,
            &x_t,
            &x_t1,
            &residual_logits,
            softmax_temperature,
        );

        loss_sum += report.loss * report.samples as f32;
        prediction_radius_sum += report.prediction_radius * report.samples as f32;
        target_radius_sum += report.target_radius * report.samples as f32;
        radius_ratio_sum += report.radius_ratio * report.samples as f32;
        samples += report.samples;

        for outcome in outcomes {
            learned_radius_totals.observe(outcome.learned_radius);
        }
    }

    assert!(
        samples > 0,
        "candidate radius logit head report has no samples"
    );
    assert_eq!(
        samples, learned_radius_totals.samples,
        "candidate radius logit head sample mismatch"
    );

    ProjectedSignedCandidateRadiusHeadIntegration {
        learned_radius: learned_radius_totals.into_metrics(),
        scalar_report: SignedCenteredRadiusScalarObjectiveReport {
            loss: loss_sum / samples as f32,
            prediction_radius: prediction_radius_sum / samples as f32,
            target_radius: target_radius_sum / samples as f32,
            radius_ratio: radius_ratio_sum / samples as f32,
            samples,
        },
        softmax_temperature,
        samples,
        candidates: SIGNED_VELOCITY_BANK_CANDIDATE_DX.len(),
    }
}

pub fn projected_signed_candidate_unit_mix_head_integration_from_base_seed<P>(
    model: &ProjectedVisionJepa<P>,
    unit_mix_head: &Linear,
    validation_base_seed: u64,
    validation_batches: usize,
    softmax_temperature: f32,
) -> ProjectedSignedCandidateUnitMixHeadIntegration
where
    P: PredictorModule,
{
    assert!(
        validation_batches > 0,
        "candidate unit-mix head integration requires at least one validation batch"
    );
    assert!(
        softmax_temperature.is_finite() && softmax_temperature > 0.0,
        "candidate unit-mix head integration temperature must be finite and positive"
    );

    let mut learned_mix_totals = SignedPredictionCounterfactualMetricTotals::default();
    let mut loss_sum = 0.0f32;
    let mut prediction_radius_sum = 0.0f32;
    let mut target_radius_sum = 0.0f32;
    let mut radius_ratio_sum = 0.0f32;
    let mut entropy_sum = 0.0f32;
    let mut max_weight_sum = 0.0f32;
    let mut true_weight_sum = 0.0f32;
    let mut samples = 0usize;

    for batch_idx in 0..validation_batches {
        let (x_t, x_t1) = make_temporal_batch_for_task(
            BATCH_SIZE,
            validation_base_seed + batch_idx as u64,
            TemporalTaskMode::SignedVelocityTrail,
        );
        let features =
            projected_signed_candidate_unit_mix_head_features(model, &x_t, softmax_temperature);
        let residual_logits = unit_mix_head.forward(&features);
        let (report, _) = projected_signed_candidate_unit_mix_loss_and_grad(
            model,
            &x_t,
            &x_t1,
            &residual_logits,
            softmax_temperature,
        );
        let outcomes = projected_signed_candidate_unit_mix_head_integration_sample_outcomes(
            model,
            &x_t,
            &x_t1,
            &residual_logits,
            softmax_temperature,
        );

        loss_sum += report.loss * report.samples as f32;
        prediction_radius_sum += report.prediction_radius * report.samples as f32;
        target_radius_sum += report.target_radius * report.samples as f32;
        radius_ratio_sum += report.radius_ratio * report.samples as f32;
        entropy_sum += report.entropy * report.samples as f32;
        max_weight_sum += report.max_weight * report.samples as f32;
        true_weight_sum += report.true_weight * report.samples as f32;
        samples += report.samples;

        for outcome in outcomes {
            learned_mix_totals.observe(outcome.learned_mix);
        }
    }

    assert!(samples > 0, "candidate unit-mix head report has no samples");
    assert_eq!(
        samples, learned_mix_totals.samples,
        "candidate unit-mix head sample mismatch"
    );

    ProjectedSignedCandidateUnitMixHeadIntegration {
        learned_mix: learned_mix_totals.into_metrics(),
        objective_report: ProjectedSignedCandidateUnitMixObjectiveReport {
            loss: loss_sum / samples as f32,
            prediction_radius: prediction_radius_sum / samples as f32,
            target_radius: target_radius_sum / samples as f32,
            radius_ratio: radius_ratio_sum / samples as f32,
            entropy: entropy_sum / samples as f32,
            max_weight: max_weight_sum / samples as f32,
            true_weight: true_weight_sum / samples as f32,
            samples,
        },
        softmax_temperature,
        samples,
        candidates: SIGNED_VELOCITY_BANK_CANDIDATE_DX.len(),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn projected_signed_candidate_selector_head_integration_from_base_seed<P>(
    model: &ProjectedVisionJepa<P>,
    selector_head: &Linear,
    validation_base_seed: u64,
    validation_batches: usize,
    softmax_temperature: f32,
    selector_steps: usize,
    learning_rate: f32,
    entropy_floor: f32,
    entropy_regularization_weight: f32,
    kl_to_prior_weight: f32,
) -> ProjectedSignedCandidateSelectorHeadIntegration
where
    P: PredictorModule,
{
    assert!(
        validation_batches > 0,
        "candidate selector head integration requires at least one validation batch"
    );
    assert!(
        softmax_temperature.is_finite() && softmax_temperature > 0.0,
        "candidate selector head integration temperature must be finite and positive"
    );
    assert!(
        learning_rate.is_finite() && learning_rate >= 0.0,
        "candidate selector head integration learning rate must be finite and non-negative"
    );
    assert!(
        entropy_floor.is_finite() && entropy_floor >= 0.0,
        "candidate selector head integration entropy floor must be finite and non-negative"
    );
    assert!(
        entropy_regularization_weight.is_finite() && entropy_regularization_weight >= 0.0,
        "candidate selector head integration entropy regularization weight must be finite and non-negative"
    );
    assert!(
        kl_to_prior_weight.is_finite() && kl_to_prior_weight >= 0.0,
        "candidate selector head integration KL-to-prior weight must be finite and non-negative"
    );

    let candidates = SIGNED_VELOCITY_BANK_CANDIDATE_DX.len();
    assert_eq!(
        candidates, 4,
        "candidate selector head expects signed-velocity-trail K=4"
    );

    let (sample_x_t, _) = make_temporal_batch_for_task(
        BATCH_SIZE,
        validation_base_seed,
        TemporalTaskMode::SignedVelocityTrail,
    );
    let sample_features =
        projected_signed_candidate_selector_head_features(model, &sample_x_t, softmax_temperature);
    assert_eq!(
        selector_head.weight.shape,
        vec![sample_features.shape[1], candidates],
        "candidate selector head weight shape mismatch"
    );
    assert_eq!(
        selector_head.bias.shape,
        vec![candidates],
        "candidate selector head bias shape mismatch"
    );

    let mut base_prediction_totals = SignedPredictionCounterfactualMetricTotals::default();
    let mut selector_soft_unit_mix_totals = SignedPredictionCounterfactualMetricTotals::default();
    let mut selector_soft_full_totals = SignedPredictionCounterfactualMetricTotals::default();
    let mut selector_hard_full_totals = SignedPredictionCounterfactualMetricTotals::default();
    let mut selector_soft_radius_totals = SignedPredictionCounterfactualMetricTotals::default();
    let mut selector_hard_radius_totals = SignedPredictionCounterfactualMetricTotals::default();
    let mut objective_totals = SignedCandidateSelectorHeadObjectiveReportTotals::default();

    for batch_idx in 0..validation_batches {
        let (x_t, x_t1) = make_temporal_batch_for_task(
            BATCH_SIZE,
            validation_base_seed + batch_idx as u64,
            TemporalTaskMode::SignedVelocityTrail,
        );
        let features =
            projected_signed_candidate_selector_head_features(model, &x_t, softmax_temperature);
        let selector_logits = selector_head.forward(&features);
        let (report, _) = projected_signed_candidate_selector_head_loss_and_grad(
            model,
            &x_t,
            &x_t1,
            &selector_logits,
            softmax_temperature,
            entropy_floor,
            entropy_regularization_weight,
            kl_to_prior_weight,
        );
        let outcomes = projected_signed_candidate_selector_head_integration_sample_outcomes(
            model,
            &x_t,
            &x_t1,
            &selector_logits,
        );

        objective_totals.observe(report);
        for outcome in outcomes {
            base_prediction_totals.observe(outcome.base_prediction);
            selector_soft_unit_mix_totals.observe(outcome.selector_soft_unit_mix);
            selector_soft_full_totals.observe(outcome.selector_soft_full);
            selector_hard_full_totals.observe(outcome.selector_hard_full);
            selector_soft_radius_totals.observe(outcome.selector_soft_radius);
            selector_hard_radius_totals.observe(outcome.selector_hard_radius);
        }
    }

    let objective_report = objective_totals.into_report(candidates);
    let samples = base_prediction_totals.samples;
    assert_eq!(
        objective_report.samples, samples,
        "candidate selector head base readout sample mismatch"
    );
    assert_eq!(
        samples, selector_soft_unit_mix_totals.samples,
        "candidate selector head soft unit-mix readout sample mismatch"
    );
    assert_eq!(
        samples, selector_soft_full_totals.samples,
        "candidate selector head soft/full readout sample mismatch"
    );
    assert_eq!(
        samples, selector_hard_full_totals.samples,
        "candidate selector head hard/full readout sample mismatch"
    );
    assert_eq!(
        samples, selector_soft_radius_totals.samples,
        "candidate selector head soft-radius readout sample mismatch"
    );
    assert_eq!(
        samples, selector_hard_radius_totals.samples,
        "candidate selector head hard-radius readout sample mismatch"
    );

    ProjectedSignedCandidateSelectorHeadIntegration {
        readout_diagnostics: ProjectedSignedCandidateSelectorReadoutDiagnostics {
            base_prediction: base_prediction_totals.into_metrics(),
            selector_soft_unit_mix: selector_soft_unit_mix_totals.into_metrics(),
            selector_soft_full: selector_soft_full_totals.into_metrics(),
            selector_hard_full: selector_hard_full_totals.into_metrics(),
            selector_soft_radius: selector_soft_radius_totals.into_metrics(),
            selector_hard_radius: selector_hard_radius_totals.into_metrics(),
        },
        objective_report,
        softmax_temperature,
        selector_steps,
        learning_rate,
        entropy_regularization_weight,
        entropy_floor,
        kl_to_prior_weight,
        support_samples: 0,
        query_samples: objective_report.samples,
        samples: objective_report.samples,
        candidates,
    }
}

pub fn projected_signed_candidate_selector_probe_from_base_seed<P>(
    model: &ProjectedVisionJepa<P>,
    validation_base_seed: u64,
    validation_batches: usize,
    softmax_temperature: f32,
    probe_steps: usize,
    learning_rate: f32,
) -> ProjectedSignedCandidateSelectorProbe
where
    P: PredictorModule,
{
    assert!(
        validation_batches >= 2,
        "candidate selector probe requires at least one support and one query batch"
    );
    assert!(
        softmax_temperature.is_finite() && softmax_temperature > 0.0,
        "candidate selector probe temperature must be finite and positive"
    );
    assert!(
        learning_rate.is_finite() && learning_rate >= 0.0,
        "candidate selector probe learning rate must be finite and non-negative"
    );

    let candidates = SIGNED_VELOCITY_BANK_CANDIDATE_DX.len();
    assert_eq!(
        candidates, 4,
        "candidate selector probe expects signed-velocity-trail K=4"
    );

    let support_batches = validation_batches / 2;
    let query_batches = validation_batches - support_batches;
    assert!(
        support_batches > 0 && query_batches > 0,
        "candidate selector probe requires non-empty support and query splits"
    );

    let (sample_x_t, _) = make_temporal_batch_for_task(
        BATCH_SIZE,
        validation_base_seed,
        TemporalTaskMode::SignedVelocityTrail,
    );
    let sample_features =
        projected_signed_candidate_unit_mix_head_features(model, &sample_x_t, softmax_temperature);
    let feature_dim = sample_features.shape[1];
    let (feature_mean, feature_inv_std) = signed_candidate_selector_feature_normalizer(
        model,
        validation_base_seed,
        support_batches,
        softmax_temperature,
        feature_dim,
    );
    let mut selector = Linear::new(
        Tensor::zeros(vec![feature_dim, candidates]),
        Tensor::zeros(vec![candidates]),
    );
    let support_samples = support_batches * BATCH_SIZE;

    for _ in 0..probe_steps {
        for batch_idx in 0..support_batches {
            let (x_t, x_t1) = make_temporal_batch_for_task(
                BATCH_SIZE,
                validation_base_seed + batch_idx as u64,
                TemporalTaskMode::SignedVelocityTrail,
            );
            let features =
                projected_signed_candidate_unit_mix_head_features(model, &x_t, softmax_temperature);
            let normalized_features = normalize_signed_candidate_selector_features(
                &features,
                &feature_mean,
                &feature_inv_std,
            );
            let true_candidate_indices = signed_velocity_true_candidate_indices(&x_t, &x_t1);
            let logits = selector.forward(&normalized_features);
            let (_, grad_logits) = signed_candidate_selector_softmax_ce_loss_and_grad(
                &logits,
                &true_candidate_indices,
            );
            let grads = selector.backward(&normalized_features, &grad_logits);
            selector.sgd_step(&grads, learning_rate);
        }
    }

    let mut prior_totals = SignedCandidateSelectorProbeMetricTotals::default();
    let mut support_trained_totals = SignedCandidateSelectorProbeMetricTotals::default();
    let mut query_trained_totals = SignedCandidateSelectorProbeMetricTotals::default();

    for batch_idx in 0..support_batches {
        let (x_t, x_t1) = make_temporal_batch_for_task(
            BATCH_SIZE,
            validation_base_seed + batch_idx as u64,
            TemporalTaskMode::SignedVelocityTrail,
        );
        let features =
            projected_signed_candidate_unit_mix_head_features(model, &x_t, softmax_temperature);
        let normalized_features = normalize_signed_candidate_selector_features(
            &features,
            &feature_mean,
            &feature_inv_std,
        );
        let true_candidate_indices = signed_velocity_true_candidate_indices(&x_t, &x_t1);
        let trained_logits = selector.forward(&normalized_features);

        support_trained_totals.observe_logits(&trained_logits, &true_candidate_indices);
    }

    for query_idx in 0..query_batches {
        let batch_idx = support_batches + query_idx;
        let (x_t, x_t1) = make_temporal_batch_for_task(
            BATCH_SIZE,
            validation_base_seed + batch_idx as u64,
            TemporalTaskMode::SignedVelocityTrail,
        );
        let features =
            projected_signed_candidate_unit_mix_head_features(model, &x_t, softmax_temperature);
        let normalized_features = normalize_signed_candidate_selector_features(
            &features,
            &feature_mean,
            &feature_inv_std,
        );
        let true_candidate_indices = signed_velocity_true_candidate_indices(&x_t, &x_t1);
        let prior_logits =
            signed_candidate_selector_prior_logits_from_features(&features, candidates);
        let trained_logits = selector.forward(&normalized_features);

        prior_totals.observe_logits(&prior_logits, &true_candidate_indices);
        query_trained_totals.observe_logits(&trained_logits, &true_candidate_indices);
    }

    let prior_anchor = prior_totals.into_metrics(candidates);
    let support_trained_probe = support_trained_totals.into_metrics(candidates);
    let trained_probe = query_trained_totals.into_metrics(candidates);
    assert_eq!(
        prior_anchor.samples, trained_probe.samples,
        "candidate selector probe query sample mismatch"
    );

    ProjectedSignedCandidateSelectorProbe {
        prior_anchor,
        support_trained_probe,
        trained_probe,
        softmax_temperature,
        probe_steps,
        learning_rate,
        support_samples,
        query_samples: trained_probe.samples,
        candidates,
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

pub fn projected_signed_direct_candidate_margin_objective_loss_and_grad<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
    required_margin: f32,
) -> (SignedDirectCandidateMarginObjectiveReport, Tensor)
where
    P: PredictorModule,
{
    assert_eq!(
        x_t.shape, x_t1.shape,
        "signed direct candidate-margin objective expects matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "signed direct candidate-margin objective expects rank-4 temporal batches, got {:?}",
        x_t.shape
    );

    let prediction = model.predict_next_projection(x_t);
    let candidate_targets = signed_velocity_candidate_target_projections(model, x_t);
    let true_candidate_indices = signed_velocity_true_candidate_indices(x_t, x_t1);

    signed_direct_candidate_margin_objective_loss_and_grad(
        &prediction,
        &candidate_targets,
        &true_candidate_indices,
        required_margin,
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

pub fn projected_signed_direct_candidate_margin_objective_report_from_base_seed<P>(
    model: &ProjectedVisionJepa<P>,
    validation_base_seed: u64,
    validation_batches: usize,
    required_margin: f32,
) -> SignedDirectCandidateMarginObjectiveReport
where
    P: PredictorModule,
{
    assert!(
        validation_batches > 0,
        "validation_batches must be greater than 0"
    );

    let mut loss = 0.0f32;
    let mut active_rate = 0.0f32;
    let mut true_distance = 0.0f32;
    let mut wrong_distance = 0.0f32;
    let mut margin = 0.0f32;
    let mut positive_margin_rate = 0.0f32;
    let mut top1 = 0.0f32;
    let mut samples = 0usize;
    let mut active_count = 0usize;

    for batch_idx in 0..validation_batches {
        let (x_t, x_t1) = make_temporal_batch_for_task(
            BATCH_SIZE,
            validation_base_seed + batch_idx as u64,
            TemporalTaskMode::SignedVelocityTrail,
        );
        let (report, _) = projected_signed_direct_candidate_margin_objective_loss_and_grad(
            model,
            &x_t,
            &x_t1,
            required_margin,
        );
        let report_samples = report.samples;
        loss += report.loss * report_samples as f32;
        active_rate += report.active_rate * report_samples as f32;
        true_distance += report.true_distance * report_samples as f32;
        wrong_distance += report.wrong_distance * report_samples as f32;
        margin += report.margin * report_samples as f32;
        positive_margin_rate += report.positive_margin_rate * report_samples as f32;
        top1 += report.top1 * report_samples as f32;
        samples += report_samples;
        active_count += report.active_count;
    }

    assert!(
        samples > 0,
        "signed direct candidate-margin objective has no samples"
    );

    SignedDirectCandidateMarginObjectiveReport {
        loss: loss / samples as f32,
        active_rate: active_rate / samples as f32,
        true_distance: true_distance / samples as f32,
        wrong_distance: wrong_distance / samples as f32,
        margin: margin / samples as f32,
        positive_margin_rate: positive_margin_rate / samples as f32,
        top1: top1 / samples as f32,
        samples,
        active_count,
    }
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

fn projected_signed_prediction_ray_boundary_sample_outcomes<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
) -> Vec<SignedPredictionRayBoundarySampleOutcome>
where
    P: PredictorModule,
{
    assert_eq!(
        x_t.shape, x_t1.shape,
        "prediction ray-boundary expects matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "prediction ray-boundary expects rank-4 temporal batches, got {:?}",
        x_t.shape
    );

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
        let current_radius = vector_l2_norm(&prediction_centered);
        let prediction_unit = unit_vector(&prediction_centered);
        let boundary = signed_ray_boundary_interval(
            &candidate_targets,
            sample,
            &centroid,
            &prediction_unit,
            true_index,
        );

        outcomes.push(match boundary {
            Some((required_radius, upper_radius)) => {
                let failure_kind = if current_radius + VELOCITY_BANK_TIE_EPSILON < required_radius {
                    SignedPredictionRayBoundaryFailureKind::BelowLower
                } else if current_radius >= upper_radius - VELOCITY_BANK_TIE_EPSILON {
                    SignedPredictionRayBoundaryFailureKind::UpperOvershoot
                } else {
                    SignedPredictionRayBoundaryFailureKind::Satisfied
                };

                SignedPredictionRayBoundarySampleOutcome {
                    true_dx,
                    current_radius,
                    required_radius,
                    upper_radius,
                    feasible: true,
                    satisfied: failure_kind == SignedPredictionRayBoundaryFailureKind::Satisfied,
                    failure_kind,
                }
            }
            None => SignedPredictionRayBoundarySampleOutcome {
                true_dx,
                current_radius,
                required_radius: 0.0,
                upper_radius: 0.0,
                feasible: false,
                satisfied: false,
                failure_kind: SignedPredictionRayBoundaryFailureKind::Infeasible,
            },
        });
    }

    outcomes
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

fn projected_signed_candidate_centroid_integration_sample_outcomes<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
    softmax_temperature: f32,
) -> Vec<SignedCandidateCentroidIntegrationSampleOutcome>
where
    P: PredictorModule,
{
    assert!(
        softmax_temperature.is_finite() && softmax_temperature > 0.0,
        "candidate-centroid integration temperature must be finite and positive"
    );
    assert_eq!(
        x_t.shape, x_t1.shape,
        "candidate-centroid integration expects matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "candidate-centroid integration expects rank-4 temporal batches, got {:?}",
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
        let prediction_unit = unit_vector(&prediction_centered);
        let target_centered =
            centered_sample_vector(&candidate_targets[true_index], sample, &centroid);
        let target_norm = vector_l2_norm(&target_centered).max(VELOCITY_BANK_TIE_EPSILON);
        let mut candidate_units = Vec::with_capacity(candidate_targets.len());
        let mut candidate_radii = Vec::with_capacity(candidate_targets.len());

        for candidate_target in &candidate_targets {
            let candidate_centered = centered_sample_vector(candidate_target, sample, &centroid);
            candidate_radii.push(vector_l2_norm(&candidate_centered));
            candidate_units.push(unit_vector(&candidate_centered));
        }

        let mean_radius = candidate_radii.iter().sum::<f32>() / candidate_radii.len() as f32;
        let nearest_index = nearest_unit_candidate_index(&prediction_unit, &candidate_units);
        let nearest_radius = candidate_radii[nearest_index];
        let softmax_radius = softmax_weighted_radius(
            &prediction_unit,
            &candidate_units,
            &candidate_radii,
            softmax_temperature,
        );

        let mean_radius_centered = scale_vector(&prediction_unit, mean_radius);
        let nearest_unit_radius_centered = scale_vector(&prediction_unit, nearest_radius);
        let softmax_radius_centered = scale_vector(&prediction_unit, softmax_radius);

        outcomes.push(SignedCandidateCentroidIntegrationSampleOutcome {
            mean_radius: signed_prediction_counterfactual_outcome(
                candidate_dx_bank,
                &candidate_targets,
                sample,
                true_dx,
                true_index,
                &centroid,
                &mean_radius_centered,
                mean_radius / target_norm,
            ),
            nearest_unit_radius: signed_prediction_counterfactual_outcome(
                candidate_dx_bank,
                &candidate_targets,
                sample,
                true_dx,
                true_index,
                &centroid,
                &nearest_unit_radius_centered,
                nearest_radius / target_norm,
            ),
            softmax_radius: signed_prediction_counterfactual_outcome(
                candidate_dx_bank,
                &candidate_targets,
                sample,
                true_dx,
                true_index,
                &centroid,
                &softmax_radius_centered,
                softmax_radius / target_norm,
            ),
        });
    }

    outcomes
}

fn projected_signed_candidate_radius_head_integration_sample_outcomes<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
    radius_delta: &Tensor,
    softmax_temperature: f32,
) -> Vec<SignedCandidateRadiusHeadIntegrationSampleOutcome>
where
    P: PredictorModule,
{
    assert!(
        softmax_temperature.is_finite() && softmax_temperature > 0.0,
        "candidate radius head integration temperature must be finite and positive"
    );
    assert_eq!(
        x_t.shape, x_t1.shape,
        "candidate radius head integration expects matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "candidate radius head integration expects rank-4 temporal batches, got {:?}",
        x_t.shape
    );

    let candidate_dx_bank = &SIGNED_VELOCITY_BANK_CANDIDATE_DX;
    let prediction = model.predict_next_projection(x_t);
    let candidate_targets = signed_velocity_candidate_target_projections(model, x_t);
    let batch_size = x_t.shape[0];
    let projection_dim = prediction.shape[1];
    let anchor_radius = signed_candidate_softmax_anchor_radius(
        &prediction,
        &candidate_targets,
        softmax_temperature,
    );
    let predicted_radius = radius_delta_to_centered_radius(radius_delta, &anchor_radius);
    let mut outcomes = Vec::with_capacity(batch_size);

    for sample in 0..batch_size {
        let true_dx = signed_motion_dx_for_sample(x_t, x_t1, sample);
        let true_index = signed_velocity_candidate_index(true_dx);
        let centroid = signed_candidate_centroid(&candidate_targets, sample, projection_dim);
        let prediction_centered = centered_sample_vector(&prediction, sample, &centroid);
        let prediction_unit = unit_vector(&prediction_centered);
        let target_centered =
            centered_sample_vector(&candidate_targets[true_index], sample, &centroid);
        let target_norm = vector_l2_norm(&target_centered).max(VELOCITY_BANK_TIE_EPSILON);
        let learned_radius = scalar_tensor_value(&predicted_radius, sample);
        let learned_radius_centered = scale_vector(&prediction_unit, learned_radius);

        outcomes.push(SignedCandidateRadiusHeadIntegrationSampleOutcome {
            learned_radius: signed_prediction_counterfactual_outcome(
                candidate_dx_bank,
                &candidate_targets,
                sample,
                true_dx,
                true_index,
                &centroid,
                &learned_radius_centered,
                learned_radius / target_norm,
            ),
        });
    }

    outcomes
}

fn projected_signed_candidate_radius_logit_head_integration_sample_outcomes<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
    residual_logits: &Tensor,
    softmax_temperature: f32,
) -> Vec<SignedCandidateRadiusHeadIntegrationSampleOutcome>
where
    P: PredictorModule,
{
    assert!(
        softmax_temperature.is_finite() && softmax_temperature > 0.0,
        "candidate radius logit head integration temperature must be finite and positive"
    );
    assert_eq!(
        x_t.shape, x_t1.shape,
        "candidate radius logit head integration expects matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "candidate radius logit head integration expects rank-4 temporal batches, got {:?}",
        x_t.shape
    );

    let candidate_dx_bank = &SIGNED_VELOCITY_BANK_CANDIDATE_DX;
    let prediction = model.predict_next_projection(x_t);
    let candidate_targets = signed_velocity_candidate_target_projections(model, x_t);
    let batch_size = x_t.shape[0];
    let projection_dim = prediction.shape[1];
    let (predicted_radius, _, _) = radius_logits_to_centered_radius_and_weights(
        residual_logits,
        &prediction,
        &candidate_targets,
        softmax_temperature,
    );
    let mut outcomes = Vec::with_capacity(batch_size);

    for sample in 0..batch_size {
        let true_dx = signed_motion_dx_for_sample(x_t, x_t1, sample);
        let true_index = signed_velocity_candidate_index(true_dx);
        let centroid = signed_candidate_centroid(&candidate_targets, sample, projection_dim);
        let prediction_centered = centered_sample_vector(&prediction, sample, &centroid);
        let prediction_unit = unit_vector(&prediction_centered);
        let target_centered =
            centered_sample_vector(&candidate_targets[true_index], sample, &centroid);
        let target_norm = vector_l2_norm(&target_centered).max(VELOCITY_BANK_TIE_EPSILON);
        let learned_radius = scalar_tensor_value(&predicted_radius, sample);
        let learned_radius_centered = scale_vector(&prediction_unit, learned_radius);

        outcomes.push(SignedCandidateRadiusHeadIntegrationSampleOutcome {
            learned_radius: signed_prediction_counterfactual_outcome(
                candidate_dx_bank,
                &candidate_targets,
                sample,
                true_dx,
                true_index,
                &centroid,
                &learned_radius_centered,
                learned_radius / target_norm,
            ),
        });
    }

    outcomes
}

fn projected_signed_candidate_unit_mix_head_integration_sample_outcomes<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
    residual_logits: &Tensor,
    softmax_temperature: f32,
) -> Vec<SignedCandidateUnitMixHeadIntegrationSampleOutcome>
where
    P: PredictorModule,
{
    assert!(
        softmax_temperature.is_finite() && softmax_temperature > 0.0,
        "candidate unit-mix head integration temperature must be finite and positive"
    );
    assert_eq!(
        x_t.shape, x_t1.shape,
        "candidate unit-mix head integration expects matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "candidate unit-mix head integration expects rank-4 temporal batches, got {:?}",
        x_t.shape
    );

    let candidate_dx_bank = &SIGNED_VELOCITY_BANK_CANDIDATE_DX;
    let prediction = model.predict_next_projection(x_t);
    let candidate_targets = signed_velocity_candidate_target_projections(model, x_t);
    let batch_size = x_t.shape[0];
    let projection_dim = prediction.shape[1];
    let mut outcomes = Vec::with_capacity(batch_size);

    assert_eq!(
        residual_logits.shape,
        vec![batch_size, candidate_targets.len()],
        "candidate unit-mix residual logits shape mismatch"
    );

    for sample in 0..batch_size {
        let true_dx = signed_motion_dx_for_sample(x_t, x_t1, sample);
        let true_index = signed_velocity_candidate_index(true_dx);
        let centroid = signed_candidate_centroid(&candidate_targets, sample, projection_dim);
        let target_centered =
            centered_sample_vector(&candidate_targets[true_index], sample, &centroid);
        let target_norm = vector_l2_norm(&target_centered).max(VELOCITY_BANK_TIE_EPSILON);
        let components = signed_candidate_unit_mix_components(
            residual_logits,
            &prediction,
            &candidate_targets,
            sample,
            &centroid,
            softmax_temperature,
        );

        outcomes.push(SignedCandidateUnitMixHeadIntegrationSampleOutcome {
            learned_mix: signed_prediction_counterfactual_outcome(
                candidate_dx_bank,
                &candidate_targets,
                sample,
                true_dx,
                true_index,
                &centroid,
                &components.centered_prediction,
                components.predicted_radius / target_norm,
            ),
        });
    }

    outcomes
}

fn projected_signed_candidate_selector_head_integration_sample_outcomes<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
    selector_logits: &Tensor,
) -> Vec<SignedCandidateSelectorHeadIntegrationSampleOutcome>
where
    P: PredictorModule,
{
    assert_eq!(
        x_t.shape, x_t1.shape,
        "candidate selector head integration expects matching pair shapes"
    );
    assert!(
        x_t.shape.len() == 4,
        "candidate selector head integration expects rank-4 temporal batches, got {:?}",
        x_t.shape
    );

    let candidate_dx_bank = &SIGNED_VELOCITY_BANK_CANDIDATE_DX;
    let prediction = model.predict_next_projection(x_t);
    let candidate_targets = signed_velocity_candidate_target_projections(model, x_t);
    let batch_size = x_t.shape[0];
    let projection_dim = prediction.shape[1];
    let mut outcomes = Vec::with_capacity(batch_size);

    assert_eq!(
        selector_logits.shape,
        vec![batch_size, candidate_targets.len()],
        "candidate selector head logits shape mismatch"
    );

    for sample in 0..batch_size {
        let true_dx = signed_motion_dx_for_sample(x_t, x_t1, sample);
        let true_index = signed_velocity_candidate_index(true_dx);
        let centroid = signed_candidate_centroid(&candidate_targets, sample, projection_dim);
        let prediction_centered = centered_sample_vector(&prediction, sample, &centroid);
        let prediction_unit = unit_vector(&prediction_centered);
        let prediction_norm = vector_l2_norm(&prediction_centered);
        let target_centered =
            centered_sample_vector(&candidate_targets[true_index], sample, &centroid);
        let target_norm = vector_l2_norm(&target_centered).max(VELOCITY_BANK_TIE_EPSILON);
        let sample_logits = (0..candidate_targets.len())
            .map(|candidate_idx| selector_logits.get(&[sample, candidate_idx]))
            .collect::<Vec<_>>();
        let weights = stable_softmax(&sample_logits, "candidate selector readout");
        let hard_index = max_logit_index(&sample_logits);
        let mut candidate_centered = Vec::with_capacity(candidate_targets.len());
        let mut candidate_units = Vec::with_capacity(candidate_targets.len());
        let mut candidate_radii = Vec::with_capacity(candidate_targets.len());

        for candidate_target in &candidate_targets {
            let centered = centered_sample_vector(candidate_target, sample, &centroid);
            candidate_radii.push(vector_l2_norm(&centered));
            candidate_units.push(unit_vector(&centered));
            candidate_centered.push(centered);
        }

        let mut selector_soft_unit_sum = vec![0.0f32; projection_dim];
        let mut selector_soft_full_centered = vec![0.0f32; projection_dim];
        let selector_soft_radius = weights
            .iter()
            .zip(candidate_radii.iter())
            .map(|(weight, radius)| weight * radius)
            .sum::<f32>();
        for ((weight, centered), candidate_unit) in weights
            .iter()
            .zip(candidate_centered.iter())
            .zip(candidate_units.iter())
        {
            for feature_idx in 0..projection_dim {
                selector_soft_full_centered[feature_idx] += weight * centered[feature_idx];
                selector_soft_unit_sum[feature_idx] += weight * candidate_unit[feature_idx];
            }
        }

        let selector_soft_unit_norm = vector_l2_norm(&selector_soft_unit_sum);
        let selector_soft_unit_mix_centered =
            if selector_soft_unit_norm <= VELOCITY_BANK_TIE_EPSILON {
                vec![0.0f32; projection_dim]
            } else {
                selector_soft_unit_sum
                    .iter()
                    .map(|value| value / selector_soft_unit_norm * selector_soft_radius)
                    .collect::<Vec<_>>()
            };
        let selector_hard_radius = candidate_radii[hard_index];
        let selector_soft_radius_centered = scale_vector(&prediction_unit, selector_soft_radius);
        let selector_hard_radius_centered = scale_vector(&prediction_unit, selector_hard_radius);

        outcomes.push(SignedCandidateSelectorHeadIntegrationSampleOutcome {
            base_prediction: signed_prediction_counterfactual_outcome(
                candidate_dx_bank,
                &candidate_targets,
                sample,
                true_dx,
                true_index,
                &centroid,
                &prediction_centered,
                prediction_norm / target_norm,
            ),
            selector_soft_unit_mix: signed_prediction_counterfactual_outcome(
                candidate_dx_bank,
                &candidate_targets,
                sample,
                true_dx,
                true_index,
                &centroid,
                &selector_soft_unit_mix_centered,
                vector_l2_norm(&selector_soft_unit_mix_centered) / target_norm,
            ),
            selector_soft_full: signed_prediction_counterfactual_outcome(
                candidate_dx_bank,
                &candidate_targets,
                sample,
                true_dx,
                true_index,
                &centroid,
                &selector_soft_full_centered,
                vector_l2_norm(&selector_soft_full_centered) / target_norm,
            ),
            selector_hard_full: signed_prediction_counterfactual_outcome(
                candidate_dx_bank,
                &candidate_targets,
                sample,
                true_dx,
                true_index,
                &centroid,
                &candidate_centered[hard_index],
                candidate_radii[hard_index] / target_norm,
            ),
            selector_soft_radius: signed_prediction_counterfactual_outcome(
                candidate_dx_bank,
                &candidate_targets,
                sample,
                true_dx,
                true_index,
                &centroid,
                &selector_soft_radius_centered,
                selector_soft_radius / target_norm,
            ),
            selector_hard_radius: signed_prediction_counterfactual_outcome(
                candidate_dx_bank,
                &candidate_targets,
                sample,
                true_dx,
                true_index,
                &centroid,
                &selector_hard_radius_centered,
                selector_hard_radius / target_norm,
            ),
        });
    }

    outcomes
}

fn projected_signed_candidate_selector_hard_full_output_coupling_report_from_logits<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    x_t1: &Tensor,
    selector_logits: &Tensor,
    loss: f32,
    active_samples: usize,
) -> ProjectedSignedCandidateSelectorOutputCouplingReport
where
    P: PredictorModule,
{
    assert!(
        loss.is_finite() && loss >= 0.0,
        "candidate selector output coupling loss must be finite and non-negative"
    );
    assert!(
        selector_logits.shape.len() == 2,
        "candidate selector output coupling logits must be rank-2, got {:?}",
        selector_logits.shape
    );

    let batch_size = selector_logits.shape[0];
    let candidates = selector_logits.shape[1];
    assert!(
        batch_size > 0,
        "candidate selector output coupling has no samples"
    );
    assert!(
        active_samples <= batch_size,
        "candidate selector output coupling active sample count exceeds batch size"
    );
    assert_eq!(
        candidates,
        SIGNED_VELOCITY_BANK_CANDIDATE_DX.len(),
        "candidate selector output coupling candidate count mismatch"
    );

    let outcomes = projected_signed_candidate_selector_head_integration_sample_outcomes(
        model,
        x_t,
        x_t1,
        selector_logits,
    );
    assert_eq!(
        outcomes.len(),
        batch_size,
        "candidate selector output coupling outcome batch mismatch"
    );

    let mut selector_max_probability_sum = 0.0f32;
    let mut base_prediction_top1 = 0usize;
    let mut base_prediction_margin_sum = 0.0f32;
    let mut base_prediction_norm_ratio_sum = 0.0f32;
    let mut selector_hard_full_top1 = 0usize;
    let mut selector_hard_full_margin_sum = 0.0f32;
    let mut selector_hard_full_norm_ratio_sum = 0.0f32;

    for (sample, outcome) in outcomes.into_iter().enumerate() {
        let sample_logits = (0..candidates)
            .map(|candidate_idx| selector_logits.get(&[sample, candidate_idx]))
            .collect::<Vec<_>>();
        let probabilities = stable_softmax(&sample_logits, "candidate selector output coupling");
        let selector_max_probability = probabilities
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, |best, value| best.max(value));
        let base_prediction_margin =
            outcome.base_prediction.nearest_wrong_distance - outcome.base_prediction.true_distance;
        let selector_hard_full_margin = outcome.selector_hard_full.nearest_wrong_distance
            - outcome.selector_hard_full.true_distance;

        assert!(
            selector_max_probability.is_finite()
                && base_prediction_margin.is_finite()
                && outcome.base_prediction.norm_ratio.is_finite()
                && selector_hard_full_margin.is_finite()
                && outcome.selector_hard_full.norm_ratio.is_finite(),
            "candidate selector output coupling produced non-finite report metrics"
        );

        selector_max_probability_sum += selector_max_probability;
        base_prediction_top1 += usize::from(outcome.base_prediction.rank == 1);
        base_prediction_margin_sum += base_prediction_margin;
        base_prediction_norm_ratio_sum += outcome.base_prediction.norm_ratio;
        selector_hard_full_top1 += usize::from(outcome.selector_hard_full.rank == 1);
        selector_hard_full_margin_sum += selector_hard_full_margin;
        selector_hard_full_norm_ratio_sum += outcome.selector_hard_full.norm_ratio;
    }

    let samples = batch_size as f32;
    ProjectedSignedCandidateSelectorOutputCouplingReport {
        loss,
        selector_max_probability: selector_max_probability_sum / samples,
        active_rate: active_samples as f32 / samples,
        base_prediction_top1: base_prediction_top1 as f32 / samples,
        base_prediction_margin: base_prediction_margin_sum / samples,
        base_prediction_norm_ratio: base_prediction_norm_ratio_sum / samples,
        selector_hard_full_top1: selector_hard_full_top1 as f32 / samples,
        selector_hard_full_margin: selector_hard_full_margin_sum / samples,
        selector_hard_full_norm_ratio: selector_hard_full_norm_ratio_sum / samples,
        active_samples,
        samples: batch_size,
        candidates,
    }
}

fn scale_tensor_values(tensor: &Tensor, scale: f32) -> Tensor {
    assert!(
        scale.is_finite(),
        "tensor scale must be finite, got {}",
        scale
    );
    Tensor::new(
        tensor.data.iter().map(|value| value * scale).collect(),
        tensor.shape.clone(),
    )
}

fn signed_candidate_selector_gated_hard_full_targets<P>(
    model: &ProjectedVisionJepa<P>,
    x_t: &Tensor,
    selector_logits: &Tensor,
    prediction: &Tensor,
    x_t1: Option<&Tensor>,
    require_correct_selector: bool,
    min_confidence: f32,
) -> (Tensor, usize)
where
    P: PredictorModule,
{
    assert!(
        selector_logits.shape.len() == 2,
        "candidate selector hard-full targets logits must be rank-2, got {:?}",
        selector_logits.shape
    );
    assert!(
        x_t.shape.len() == 4,
        "candidate selector hard-full targets expect rank-4 temporal batches, got {:?}",
        x_t.shape
    );
    assert!(
        min_confidence.is_finite() && (0.0..=1.0).contains(&min_confidence),
        "candidate selector hard-full target min confidence must be in [0, 1]"
    );

    let batch_size = x_t.shape[0];
    let candidates = SIGNED_VELOCITY_BANK_CANDIDATE_DX.len();
    assert_eq!(
        selector_logits.shape,
        vec![batch_size, candidates],
        "candidate selector hard-full targets logit shape mismatch"
    );

    let candidate_targets = signed_velocity_candidate_target_projections(model, x_t);
    assert_eq!(
        candidate_targets.len(),
        candidates,
        "candidate selector hard-full target count mismatch"
    );
    let target_shape = candidate_targets
        .first()
        .expect("candidate selector hard-full targets require non-empty candidates")
        .shape
        .clone();
    assert!(
        target_shape.len() == 2 && target_shape[0] == batch_size,
        "candidate selector hard-full targets expect rank-2 projected targets, got {:?}",
        target_shape
    );
    for candidate_target in &candidate_targets {
        assert_eq!(
            candidate_target.shape, target_shape,
            "candidate selector hard-full target shape mismatch"
        );
    }
    assert_eq!(
        prediction.shape, target_shape,
        "candidate selector hard-full target prediction shape mismatch"
    );
    let true_candidate_indices = if require_correct_selector {
        let x_t1 =
            x_t1.expect("correct-gated selector hard-full targets require the true future batch");
        Some(signed_velocity_true_candidate_indices(x_t, x_t1))
    } else {
        None
    };

    let projection_dim = target_shape[1];
    let mut data = Vec::with_capacity(batch_size * projection_dim);
    let mut active_samples = 0usize;
    for sample in 0..batch_size {
        let sample_logits = (0..candidates)
            .map(|candidate_idx| selector_logits.get(&[sample, candidate_idx]))
            .collect::<Vec<_>>();
        let probabilities = stable_softmax(&sample_logits, "candidate selector hard-full targets");
        let hard_index = max_logit_index(&sample_logits);
        let max_probability = probabilities[hard_index];
        let selector_is_correct = true_candidate_indices
            .as_ref()
            .map_or(true, |indices| hard_index == indices[sample]);
        let sample_active = max_probability >= min_confidence && selector_is_correct;
        active_samples += usize::from(sample_active);
        for feature_idx in 0..projection_dim {
            if sample_active {
                data.push(candidate_targets[hard_index].get(&[sample, feature_idx]));
            } else {
                data.push(prediction.get(&[sample, feature_idx]));
            }
        }
    }

    (Tensor::new(data, target_shape), active_samples)
}

fn signed_candidate_radius_head_features_from_prediction(
    projection: &Tensor,
    prediction: &Tensor,
    candidate_targets: &[Tensor],
    softmax_temperature: f32,
) -> Tensor {
    assert!(
        projection.shape.len() == 2,
        "candidate radius head projection features must be rank-2, got {:?}",
        projection.shape
    );
    assert!(
        prediction.shape.len() == 2,
        "candidate radius head prediction must be rank-2, got {:?}",
        prediction.shape
    );
    assert_eq!(
        projection.shape[0], prediction.shape[0],
        "candidate radius head feature batch mismatch"
    );
    assert!(
        !candidate_targets.is_empty(),
        "candidate radius head features require non-empty candidate targets"
    );
    assert!(
        softmax_temperature.is_finite() && softmax_temperature > 0.0,
        "candidate radius head feature temperature must be finite and positive"
    );

    let batch_size = prediction.shape[0];
    let projection_dim = prediction.shape[1];
    let feature_dim = projection.shape[1] + projection_dim + 4;
    let mut features = Tensor::zeros(vec![batch_size, feature_dim]);

    for candidate_target in candidate_targets {
        assert_eq!(
            candidate_target.shape, prediction.shape,
            "candidate radius head target shape mismatch"
        );
    }

    for sample in 0..batch_size {
        let centroid = signed_candidate_centroid(candidate_targets, sample, projection_dim);
        let prediction_centered = centered_sample_vector(prediction, sample, &centroid);
        let prediction_unit = unit_vector(&prediction_centered);
        let prediction_norm = vector_l2_norm(&prediction_centered);
        let mut candidate_units = Vec::with_capacity(candidate_targets.len());
        let mut candidate_radii = Vec::with_capacity(candidate_targets.len());

        for candidate_target in candidate_targets {
            let candidate_centered = centered_sample_vector(candidate_target, sample, &centroid);
            candidate_radii.push(vector_l2_norm(&candidate_centered));
            candidate_units.push(unit_vector(&candidate_centered));
        }

        let anchor_radius = softmax_weighted_radius(
            &prediction_unit,
            &candidate_units,
            &candidate_radii,
            softmax_temperature,
        );
        let mean_radius = candidate_radii.iter().sum::<f32>() / candidate_radii.len() as f32;
        let min_radius = candidate_radii
            .iter()
            .copied()
            .fold(f32::INFINITY, |best, radius| best.min(radius));
        let max_radius = candidate_radii
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, |best, radius| best.max(radius));

        for feature_idx in 0..projection.shape[1] {
            features.set(
                &[sample, feature_idx],
                projection.get(&[sample, feature_idx]),
            );
        }

        let centered_offset = projection.shape[1];
        for (feature_idx, value) in prediction_centered.iter().enumerate() {
            features.set(&[sample, centered_offset + feature_idx], *value);
        }

        let scalar_offset = centered_offset + projection_dim;
        features.set(&[sample, scalar_offset], prediction_norm);
        features.set(&[sample, scalar_offset + 1], anchor_radius);
        features.set(&[sample, scalar_offset + 2], mean_radius);
        features.set(&[sample, scalar_offset + 3], max_radius - min_radius);
    }

    features
}

fn signed_candidate_radius_logit_head_features_from_prediction(
    projection: &Tensor,
    prediction: &Tensor,
    candidate_targets: &[Tensor],
    softmax_temperature: f32,
) -> Tensor {
    assert!(
        projection.shape.len() == 2,
        "candidate radius logit head projection features must be rank-2, got {:?}",
        projection.shape
    );
    assert!(
        prediction.shape.len() == 2,
        "candidate radius logit head prediction must be rank-2, got {:?}",
        prediction.shape
    );
    assert_eq!(
        projection.shape[0], prediction.shape[0],
        "candidate radius logit head feature batch mismatch"
    );
    assert!(
        !candidate_targets.is_empty(),
        "candidate radius logit head features require non-empty candidate targets"
    );
    assert!(
        softmax_temperature.is_finite() && softmax_temperature > 0.0,
        "candidate radius logit head feature temperature must be finite and positive"
    );

    let batch_size = prediction.shape[0];
    let projection_dim = prediction.shape[1];
    let candidates = candidate_targets.len();
    let feature_dim = projection.shape[1] + candidates * 2;
    let mut features = Tensor::zeros(vec![batch_size, feature_dim]);

    for candidate_target in candidate_targets {
        assert_eq!(
            candidate_target.shape, prediction.shape,
            "candidate radius logit head target shape mismatch"
        );
    }

    for sample in 0..batch_size {
        let (prior_logits, candidate_radii) = signed_candidate_prior_logits_and_radii(
            prediction,
            candidate_targets,
            sample,
            projection_dim,
            softmax_temperature,
        );

        for feature_idx in 0..projection.shape[1] {
            features.set(
                &[sample, feature_idx],
                projection.get(&[sample, feature_idx]),
            );
        }

        let prior_offset = projection.shape[1];
        for (candidate_idx, prior_logit) in prior_logits.iter().enumerate() {
            features.set(&[sample, prior_offset + candidate_idx], *prior_logit);
        }

        let radius_offset = prior_offset + candidates;
        for (candidate_idx, radius) in candidate_radii.iter().enumerate() {
            features.set(&[sample, radius_offset + candidate_idx], *radius);
        }
    }

    features
}

fn signed_candidate_selector_feature_normalizer<P>(
    model: &ProjectedVisionJepa<P>,
    validation_base_seed: u64,
    support_batches: usize,
    softmax_temperature: f32,
    feature_dim: usize,
) -> (Vec<f32>, Vec<f32>)
where
    P: PredictorModule,
{
    assert!(support_batches > 0, "selector probe needs support batches");
    assert!(
        feature_dim > 0,
        "selector probe feature dim must be positive"
    );

    let mut feature_sum = vec![0.0f32; feature_dim];
    let mut feature_square_sum = vec![0.0f32; feature_dim];
    let mut samples = 0usize;

    for batch_idx in 0..support_batches {
        let (x_t, _) = make_temporal_batch_for_task(
            BATCH_SIZE,
            validation_base_seed + batch_idx as u64,
            TemporalTaskMode::SignedVelocityTrail,
        );
        let features =
            projected_signed_candidate_selector_head_features(model, &x_t, softmax_temperature);
        assert_eq!(
            features.shape,
            vec![BATCH_SIZE, feature_dim],
            "candidate selector probe support feature shape mismatch"
        );

        for sample in 0..BATCH_SIZE {
            for feature_idx in 0..feature_dim {
                let value = features.get(&[sample, feature_idx]);
                feature_sum[feature_idx] += value;
                feature_square_sum[feature_idx] += value * value;
            }
            samples += 1;
        }
    }

    assert!(
        samples > 1,
        "selector probe needs at least two support samples"
    );
    let mut feature_mean = vec![0.0f32; feature_dim];
    let mut feature_inv_std = vec![0.0f32; feature_dim];

    for feature_idx in 0..feature_dim {
        let mean = feature_sum[feature_idx] / samples as f32;
        let square_mean = feature_square_sum[feature_idx] / samples as f32;
        let variance = (square_mean - mean * mean).max(0.0);
        let std = variance.sqrt();

        feature_mean[feature_idx] = mean;
        feature_inv_std[feature_idx] = if std > VELOCITY_BANK_TIE_EPSILON {
            1.0 / std
        } else {
            0.0
        };
    }

    (feature_mean, feature_inv_std)
}

fn normalize_signed_candidate_selector_features(
    features: &Tensor,
    feature_mean: &[f32],
    feature_inv_std: &[f32],
) -> Tensor {
    assert!(
        features.shape.len() == 2,
        "candidate selector probe normalized features must be rank-2, got {:?}",
        features.shape
    );
    assert_eq!(
        feature_mean.len(),
        features.shape[1],
        "candidate selector probe mean feature mismatch"
    );
    assert_eq!(
        feature_inv_std.len(),
        features.shape[1],
        "candidate selector probe std feature mismatch"
    );

    let mut normalized = Tensor::zeros(features.shape.clone());

    for sample in 0..features.shape[0] {
        for feature_idx in 0..features.shape[1] {
            let value = (features.get(&[sample, feature_idx]) - feature_mean[feature_idx])
                * feature_inv_std[feature_idx];
            normalized.set(&[sample, feature_idx], value);
        }
    }

    normalized
}

fn normalize_signed_candidate_selector_batch_features(features: &Tensor) -> Tensor {
    assert!(
        features.shape.len() == 2,
        "candidate selector head normalized features must be rank-2, got {:?}",
        features.shape
    );

    let batch_size = features.shape[0];
    let feature_dim = features.shape[1];
    assert!(
        batch_size > 0 && feature_dim > 0,
        "candidate selector head normalized features require non-empty tensors"
    );

    let mut feature_mean = vec![0.0f32; feature_dim];
    let mut feature_square_mean = vec![0.0f32; feature_dim];
    for sample in 0..batch_size {
        for feature_idx in 0..feature_dim {
            let value = features.get(&[sample, feature_idx]);
            feature_mean[feature_idx] += value;
            feature_square_mean[feature_idx] += value * value;
        }
    }

    for feature_idx in 0..feature_dim {
        feature_mean[feature_idx] /= batch_size as f32;
        feature_square_mean[feature_idx] /= batch_size as f32;
    }

    let mut normalized = Tensor::zeros(features.shape.clone());
    for sample in 0..batch_size {
        for feature_idx in 0..feature_dim {
            let variance =
                (feature_square_mean[feature_idx] - feature_mean[feature_idx].powi(2)).max(0.0);
            let std = variance.sqrt();
            let inv_std = if std > VELOCITY_BANK_TIE_EPSILON {
                1.0 / std
            } else {
                0.0
            };
            let value =
                (features.get(&[sample, feature_idx]) - feature_mean[feature_idx]) * inv_std;
            normalized.set(&[sample, feature_idx], value);
        }
    }

    normalized
}

fn signed_candidate_selector_prior_logits_from_features(
    features: &Tensor,
    candidates: usize,
) -> Tensor {
    assert!(
        features.shape.len() == 2,
        "candidate selector probe features must be rank-2, got {:?}",
        features.shape
    );
    assert!(candidates > 0, "candidate selector probe has no candidates");
    assert!(
        features.shape[1] >= candidates * 2,
        "candidate selector probe feature dim {} cannot contain prior logits and radii for {} candidates",
        features.shape[1],
        candidates
    );

    let batch_size = features.shape[0];
    let prior_offset = features.shape[1] - candidates * 2;
    let mut prior_logits = Tensor::zeros(vec![batch_size, candidates]);

    for sample in 0..batch_size {
        for candidate_idx in 0..candidates {
            prior_logits.set(
                &[sample, candidate_idx],
                features.get(&[sample, prior_offset + candidate_idx]),
            );
        }
    }

    prior_logits
}

fn signed_candidate_softmax_anchor_radius(
    prediction: &Tensor,
    candidate_targets: &[Tensor],
    softmax_temperature: f32,
) -> Vec<f32> {
    assert!(
        prediction.shape.len() == 2,
        "candidate radius anchor prediction must be rank-2, got {:?}",
        prediction.shape
    );
    assert!(
        !candidate_targets.is_empty(),
        "candidate radius anchor requires non-empty candidate targets"
    );
    assert!(
        softmax_temperature.is_finite() && softmax_temperature > 0.0,
        "candidate radius anchor temperature must be finite and positive"
    );

    let batch_size = prediction.shape[0];
    let projection_dim = prediction.shape[1];
    let mut anchor_radius = Vec::with_capacity(batch_size);

    for candidate_target in candidate_targets {
        assert_eq!(
            candidate_target.shape, prediction.shape,
            "candidate radius anchor target shape mismatch"
        );
    }

    for sample in 0..batch_size {
        let centroid = signed_candidate_centroid(candidate_targets, sample, projection_dim);
        let prediction_centered = centered_sample_vector(prediction, sample, &centroid);
        let prediction_unit = unit_vector(&prediction_centered);
        let mut candidate_units = Vec::with_capacity(candidate_targets.len());
        let mut candidate_radii = Vec::with_capacity(candidate_targets.len());

        for candidate_target in candidate_targets {
            let candidate_centered = centered_sample_vector(candidate_target, sample, &centroid);
            candidate_radii.push(vector_l2_norm(&candidate_centered));
            candidate_units.push(unit_vector(&candidate_centered));
        }

        anchor_radius.push(softmax_weighted_radius(
            &prediction_unit,
            &candidate_units,
            &candidate_radii,
            softmax_temperature,
        ));
    }

    anchor_radius
}

fn radius_logits_to_centered_radius_and_weights(
    residual_logits: &Tensor,
    prediction: &Tensor,
    candidate_targets: &[Tensor],
    softmax_temperature: f32,
) -> (Tensor, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    assert!(
        prediction.shape.len() == 2,
        "candidate radius logit prediction must be rank-2, got {:?}",
        prediction.shape
    );
    assert!(
        !candidate_targets.is_empty(),
        "candidate radius logit mixing requires non-empty candidate targets"
    );
    assert_eq!(
        residual_logits.shape,
        vec![prediction.shape[0], candidate_targets.len()],
        "candidate radius residual logits shape mismatch"
    );
    assert!(
        softmax_temperature.is_finite() && softmax_temperature > 0.0,
        "candidate radius logit mixing temperature must be finite and positive"
    );

    for candidate_target in candidate_targets {
        assert_eq!(
            candidate_target.shape, prediction.shape,
            "candidate radius logit target shape mismatch"
        );
    }

    let batch_size = prediction.shape[0];
    let projection_dim = prediction.shape[1];
    let candidates = candidate_targets.len();
    let mut predicted_radius = Tensor::zeros(vec![batch_size, 1]);
    let mut all_weights = Vec::with_capacity(batch_size);
    let mut all_candidate_radii = Vec::with_capacity(batch_size);

    for sample in 0..batch_size {
        let (prior_logits, candidate_radii) = signed_candidate_prior_logits_and_radii(
            prediction,
            candidate_targets,
            sample,
            projection_dim,
            softmax_temperature,
        );
        let scores = (0..candidates)
            .map(|candidate_idx| {
                prior_logits[candidate_idx] + residual_logits.get(&[sample, candidate_idx])
            })
            .collect::<Vec<_>>();
        let weights = stable_softmax(&scores, "candidate radius logit mixing");
        let radius = weights
            .iter()
            .zip(candidate_radii.iter())
            .map(|(weight, candidate_radius)| weight * candidate_radius)
            .sum::<f32>();

        assert!(
            radius.is_finite() && radius >= 0.0,
            "candidate radius logit mixing produced non-finite radius"
        );
        predicted_radius.set(&[sample, 0], radius);
        all_weights.push(weights);
        all_candidate_radii.push(candidate_radii);
    }

    (predicted_radius, all_weights, all_candidate_radii)
}

fn signed_candidate_selector_softmax_ce_loss_and_grad(
    logits: &Tensor,
    true_candidate_indices: &[usize],
) -> (ProjectedSignedCandidateSelectorProbeMetrics, Tensor) {
    assert!(
        logits.shape.len() == 2,
        "candidate selector probe logits must be rank-2, got {:?}",
        logits.shape
    );

    let batch_size = logits.shape[0];
    let candidates = logits.shape[1];
    assert_eq!(
        true_candidate_indices.len(),
        batch_size,
        "candidate selector probe label batch mismatch"
    );
    assert!(batch_size > 0, "candidate selector probe has no samples");
    assert!(candidates > 0, "candidate selector probe has no candidates");

    let mut totals = SignedCandidateSelectorProbeMetricTotals::default();
    let mut grad_logits = Tensor::zeros(vec![batch_size, candidates]);

    for (sample, true_index) in true_candidate_indices.iter().copied().enumerate() {
        assert!(
            true_index < candidates,
            "candidate selector probe true index {} exceeds candidate count {}",
            true_index,
            candidates
        );

        let sample_logits = (0..candidates)
            .map(|candidate_idx| logits.get(&[sample, candidate_idx]))
            .collect::<Vec<_>>();
        let probabilities = stable_softmax(&sample_logits, "candidate selector probe train");

        for (candidate_idx, probability) in probabilities.iter().enumerate() {
            let target = if candidate_idx == true_index {
                1.0
            } else {
                0.0
            };
            grad_logits.set(
                &[sample, candidate_idx],
                (*probability - target) / batch_size as f32,
            );
        }
    }

    totals.observe_logits(logits, true_candidate_indices);

    (totals.into_metrics(candidates), grad_logits)
}

fn signed_candidate_selector_head_objective_loss_and_grad_from_logits(
    selector_logits: &Tensor,
    prior_logits: &Tensor,
    true_candidate_indices: &[usize],
    entropy_floor: f32,
    entropy_regularization_weight: f32,
    kl_to_prior_weight: f32,
) -> (ProjectedSignedCandidateSelectorHeadObjectiveReport, Tensor) {
    assert!(
        selector_logits.shape.len() == 2,
        "candidate selector head logits must be rank-2, got {:?}",
        selector_logits.shape
    );
    assert_eq!(
        selector_logits.shape, prior_logits.shape,
        "candidate selector head prior/logit shape mismatch"
    );
    assert!(
        entropy_floor.is_finite() && entropy_floor >= 0.0,
        "candidate selector head entropy floor must be finite and non-negative"
    );
    assert!(
        entropy_regularization_weight.is_finite() && entropy_regularization_weight >= 0.0,
        "candidate selector head entropy regularization weight must be finite and non-negative"
    );
    assert!(
        kl_to_prior_weight.is_finite() && kl_to_prior_weight >= 0.0,
        "candidate selector head KL-to-prior weight must be finite and non-negative"
    );

    let batch_size = selector_logits.shape[0];
    let candidates = selector_logits.shape[1];
    assert_eq!(
        true_candidate_indices.len(),
        batch_size,
        "candidate selector head label batch mismatch"
    );
    assert!(batch_size > 0, "candidate selector head has no samples");
    assert_eq!(
        candidates, 4,
        "candidate selector head expects signed-velocity-trail K=4"
    );

    let mut grad_logits = Tensor::zeros(vec![batch_size, candidates]);
    let mut cross_entropy_sum = 0.0f32;
    let mut entropy_regularization_sum = 0.0f32;
    let mut entropy_sum = 0.0f32;
    let mut kl_sum = 0.0f32;
    let mut true_probability_sum = 0.0f32;
    let mut max_probability_sum = 0.0f32;

    for (sample, true_index) in true_candidate_indices.iter().copied().enumerate() {
        assert!(
            true_index < candidates,
            "candidate selector head true index {} exceeds candidate count {}",
            true_index,
            candidates
        );

        let sample_logits = (0..candidates)
            .map(|candidate_idx| selector_logits.get(&[sample, candidate_idx]))
            .collect::<Vec<_>>();
        let sample_prior_logits = (0..candidates)
            .map(|candidate_idx| prior_logits.get(&[sample, candidate_idx]))
            .collect::<Vec<_>>();
        let probabilities = stable_softmax(&sample_logits, "candidate selector head");
        let prior_probabilities =
            stable_softmax(&sample_prior_logits, "candidate selector head prior");
        let true_probability = probabilities[true_index];
        let cross_entropy = -true_probability.max(VELOCITY_BANK_TIE_EPSILON).ln();
        let mut entropy = 0.0f32;
        let mut kl_to_prior = 0.0f32;
        let mut max_probability = f32::NEG_INFINITY;

        for candidate_idx in 0..candidates {
            let probability = probabilities[candidate_idx].max(VELOCITY_BANK_TIE_EPSILON);
            let prior_probability =
                prior_probabilities[candidate_idx].max(VELOCITY_BANK_TIE_EPSILON);
            let log_probability = probability.ln();
            let log_prior_probability = prior_probability.ln();

            entropy -= probabilities[candidate_idx] * log_probability;
            kl_to_prior += probabilities[candidate_idx] * (log_probability - log_prior_probability);
            max_probability = max_probability.max(probabilities[candidate_idx]);
        }

        let entropy_gap = (entropy_floor - entropy).max(0.0);
        cross_entropy_sum += cross_entropy;
        entropy_regularization_sum += entropy_gap * entropy_gap;
        entropy_sum += entropy;
        kl_sum += kl_to_prior;
        true_probability_sum += true_probability;
        max_probability_sum += max_probability;

        for candidate_idx in 0..candidates {
            let target = if candidate_idx == true_index {
                1.0
            } else {
                0.0
            };
            let probability = probabilities[candidate_idx].max(VELOCITY_BANK_TIE_EPSILON);
            let prior_probability =
                prior_probabilities[candidate_idx].max(VELOCITY_BANK_TIE_EPSILON);
            let entropy_hinge_grad = if entropy_gap > 0.0 {
                2.0 * entropy_regularization_weight
                    * entropy_gap
                    * probabilities[candidate_idx]
                    * (probability.ln() + entropy)
            } else {
                0.0
            };
            let kl_grad = probabilities[candidate_idx]
                * (probability.ln() - prior_probability.ln() - kl_to_prior);
            let value = (probabilities[candidate_idx] - target
                + entropy_hinge_grad
                + kl_to_prior_weight * kl_grad)
                / batch_size as f32;

            assert!(
                value.is_finite(),
                "candidate selector head produced non-finite logit gradient"
            );
            grad_logits.set(&[sample, candidate_idx], value);
        }
    }

    let entropy_regularization_loss = entropy_regularization_weight * entropy_regularization_sum;
    let kl_to_prior_loss = kl_to_prior_weight * kl_sum;
    let total_loss = cross_entropy_sum + entropy_regularization_loss + kl_to_prior_loss;

    (
        ProjectedSignedCandidateSelectorHeadObjectiveReport {
            loss: total_loss / batch_size as f32,
            cross_entropy_loss: cross_entropy_sum / batch_size as f32,
            entropy_regularization_loss: entropy_regularization_loss / batch_size as f32,
            kl_to_prior_loss: kl_to_prior_loss / batch_size as f32,
            entropy: entropy_sum / batch_size as f32,
            true_probability: true_probability_sum / batch_size as f32,
            max_probability: max_probability_sum / batch_size as f32,
            samples: batch_size,
            candidates,
        },
        grad_logits,
    )
}

fn centered_radius_grad_to_residual_logits_grad(
    predicted_radius: &Tensor,
    grad_radius: &Tensor,
    weights: &[Vec<f32>],
    candidate_radii: &[Vec<f32>],
) -> Tensor {
    assert_eq!(
        predicted_radius.shape, grad_radius.shape,
        "candidate radius logit grad radius shape mismatch"
    );
    assert_eq!(
        weights.len(),
        predicted_radius.shape[0],
        "candidate radius logit weight batch mismatch"
    );
    assert_eq!(
        candidate_radii.len(),
        predicted_radius.shape[0],
        "candidate radius logit radius batch mismatch"
    );

    let batch_size = predicted_radius.shape[0];
    let candidates = weights
        .first()
        .map(|row| row.len())
        .expect("candidate radius logit grad requires non-empty weights");
    let mut grad_logits = Tensor::zeros(vec![batch_size, candidates]);

    for sample in 0..batch_size {
        assert_eq!(
            weights[sample].len(),
            candidates,
            "candidate radius logit weight candidate count mismatch"
        );
        assert_eq!(
            candidate_radii[sample].len(),
            candidates,
            "candidate radius logit radius candidate count mismatch"
        );

        let grad = scalar_tensor_value(grad_radius, sample);
        let radius = scalar_tensor_value(predicted_radius, sample);
        for candidate_idx in 0..candidates {
            let value = grad
                * weights[sample][candidate_idx]
                * (candidate_radii[sample][candidate_idx] - radius);
            grad_logits.set(&[sample, candidate_idx], value);
        }
    }

    grad_logits
}

fn candidate_unit_mix_loss_and_grad_from_prediction(
    prediction: &Tensor,
    candidate_targets: &[Tensor],
    true_candidate_indices: &[usize],
    residual_logits: &Tensor,
    softmax_temperature: f32,
) -> (ProjectedSignedCandidateUnitMixObjectiveReport, Tensor) {
    assert!(
        prediction.shape.len() == 2,
        "candidate unit-mix prediction must be rank-2, got {:?}",
        prediction.shape
    );
    assert!(
        !candidate_targets.is_empty(),
        "candidate unit-mix requires non-empty candidate targets"
    );
    assert!(
        softmax_temperature.is_finite() && softmax_temperature > 0.0,
        "candidate unit-mix temperature must be finite and positive"
    );

    let batch_size = prediction.shape[0];
    let projection_dim = prediction.shape[1];
    let candidates = candidate_targets.len();
    assert_eq!(
        true_candidate_indices.len(),
        batch_size,
        "candidate unit-mix true index batch mismatch"
    );
    assert_eq!(
        residual_logits.shape,
        vec![batch_size, candidates],
        "candidate unit-mix residual logits shape mismatch"
    );
    for candidate_target in candidate_targets {
        assert_eq!(
            candidate_target.shape, prediction.shape,
            "candidate unit-mix target shape mismatch"
        );
    }

    let mut grad_logits = Tensor::zeros(vec![batch_size, candidates]);
    let mut loss_sum = 0.0f32;
    let mut prediction_radius_sum = 0.0f32;
    let mut target_radius_sum = 0.0f32;
    let mut radius_ratio_sum = 0.0f32;
    let mut entropy_sum = 0.0f32;
    let mut max_weight_sum = 0.0f32;
    let mut true_weight_sum = 0.0f32;

    for (sample, true_index) in true_candidate_indices.iter().copied().enumerate() {
        assert!(
            true_index < candidates,
            "candidate unit-mix true index {} exceeds candidate count {}",
            true_index,
            candidates
        );

        let centroid = signed_candidate_centroid(candidate_targets, sample, projection_dim);
        let target_centered =
            centered_sample_vector(&candidate_targets[true_index], sample, &centroid);
        let components = signed_candidate_unit_mix_components(
            residual_logits,
            prediction,
            candidate_targets,
            sample,
            &centroid,
            softmax_temperature,
        );
        let mut sample_loss = 0.0f32;
        let mut grad_centered = vec![0.0f32; projection_dim];

        for feature_idx in 0..projection_dim {
            let diff = components.centered_prediction[feature_idx] - target_centered[feature_idx];
            sample_loss += diff * diff;
            grad_centered[feature_idx] = 2.0 * diff / (batch_size * projection_dim) as f32;
        }

        let target_radius = vector_l2_norm(&target_centered);
        let target_radius_safe = target_radius.max(VELOCITY_BANK_TIE_EPSILON);
        let entropy = components
            .weights
            .iter()
            .map(|weight| {
                if *weight <= VELOCITY_BANK_TIE_EPSILON {
                    0.0
                } else {
                    -weight * weight.ln()
                }
            })
            .sum::<f32>();
        let max_weight = components
            .weights
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, |best, weight| best.max(weight));

        loss_sum += sample_loss / projection_dim as f32;
        prediction_radius_sum += components.predicted_radius;
        target_radius_sum += target_radius;
        radius_ratio_sum += components.predicted_radius / target_radius_safe;
        entropy_sum += entropy;
        max_weight_sum += max_weight;
        true_weight_sum += components.weights[true_index];

        if components.mixed_unit_norm > VELOCITY_BANK_TIE_EPSILON {
            let grad_dot_unit = vector_dot(&grad_centered, &components.mixed_unit);
            let grad_radius = grad_dot_unit;
            let grad_mixed_sum = (0..projection_dim)
                .map(|feature_idx| {
                    components.mixed_radius / components.mixed_unit_norm
                        * (grad_centered[feature_idx]
                            - components.mixed_unit[feature_idx] * grad_dot_unit)
                })
                .collect::<Vec<_>>();
            let candidate_scores = (0..candidates)
                .map(|candidate_idx| {
                    grad_radius * components.candidate_radii[candidate_idx]
                        + vector_dot(&grad_mixed_sum, &components.candidate_units[candidate_idx])
                })
                .collect::<Vec<_>>();
            let expected_score = components
                .weights
                .iter()
                .zip(candidate_scores.iter())
                .map(|(weight, score)| weight * score)
                .sum::<f32>();

            for (candidate_idx, score) in candidate_scores.iter().enumerate() {
                let value = components.weights[candidate_idx] * (*score - expected_score);
                assert!(
                    value.is_finite(),
                    "candidate unit-mix produced non-finite logit gradient"
                );
                grad_logits.set(&[sample, candidate_idx], value);
            }
        }
    }

    assert!(batch_size > 0, "candidate unit-mix report has no samples");

    (
        ProjectedSignedCandidateUnitMixObjectiveReport {
            loss: loss_sum / batch_size as f32,
            prediction_radius: prediction_radius_sum / batch_size as f32,
            target_radius: target_radius_sum / batch_size as f32,
            radius_ratio: radius_ratio_sum / batch_size as f32,
            entropy: entropy_sum / batch_size as f32,
            max_weight: max_weight_sum / batch_size as f32,
            true_weight: true_weight_sum / batch_size as f32,
            samples: batch_size,
        },
        grad_logits,
    )
}

fn signed_candidate_unit_mix_components(
    residual_logits: &Tensor,
    prediction: &Tensor,
    candidate_targets: &[Tensor],
    sample: usize,
    centroid: &[f32],
    softmax_temperature: f32,
) -> SignedCandidateUnitMixComponents {
    assert!(
        softmax_temperature.is_finite() && softmax_temperature > 0.0,
        "candidate unit-mix component temperature must be finite and positive"
    );
    assert!(
        !candidate_targets.is_empty(),
        "candidate unit-mix components require non-empty candidate targets"
    );
    assert_eq!(
        residual_logits.shape,
        vec![prediction.shape[0], candidate_targets.len()],
        "candidate unit-mix residual logits shape mismatch"
    );

    let projection_dim = prediction.shape[1];
    let candidates = candidate_targets.len();
    let prediction_centered = centered_sample_vector(prediction, sample, centroid);
    let prediction_unit = unit_vector(&prediction_centered);
    let mut candidate_radii = Vec::with_capacity(candidates);
    let mut candidate_units = Vec::with_capacity(candidates);
    let mut prior_logits = Vec::with_capacity(candidates);

    for candidate_target in candidate_targets {
        let candidate_centered = centered_sample_vector(candidate_target, sample, centroid);
        let candidate_radius = vector_l2_norm(&candidate_centered);
        let candidate_unit = unit_vector(&candidate_centered);

        prior_logits.push(
            -vector_squared_distance(&prediction_unit, &candidate_unit) / softmax_temperature,
        );
        candidate_radii.push(candidate_radius);
        candidate_units.push(candidate_unit);
    }

    let scores = (0..candidates)
        .map(|candidate_idx| {
            prior_logits[candidate_idx] + residual_logits.get(&[sample, candidate_idx])
        })
        .collect::<Vec<_>>();
    let weights = stable_softmax(&scores, "candidate unit-mix");
    let mixed_radius = weights
        .iter()
        .zip(candidate_radii.iter())
        .map(|(weight, radius)| weight * radius)
        .sum::<f32>();
    let mut mixed_unit_sum = vec![0.0f32; projection_dim];

    for (weight, candidate_unit) in weights.iter().zip(candidate_units.iter()) {
        for feature_idx in 0..projection_dim {
            mixed_unit_sum[feature_idx] += weight * candidate_unit[feature_idx];
        }
    }

    let mixed_unit_norm = vector_l2_norm(&mixed_unit_sum);
    let mixed_unit = if mixed_unit_norm <= VELOCITY_BANK_TIE_EPSILON {
        vec![0.0f32; projection_dim]
    } else {
        mixed_unit_sum
            .iter()
            .map(|value| value / mixed_unit_norm)
            .collect::<Vec<_>>()
    };
    let centered_prediction = scale_vector(&mixed_unit, mixed_radius);
    let predicted_radius = vector_l2_norm(&centered_prediction);

    assert!(
        predicted_radius.is_finite() && predicted_radius >= 0.0,
        "candidate unit-mix produced non-finite radius"
    );

    SignedCandidateUnitMixComponents {
        centered_prediction,
        mixed_unit,
        mixed_unit_norm,
        mixed_radius,
        predicted_radius,
        weights,
        candidate_radii,
        candidate_units,
    }
}

fn radius_delta_to_centered_radius(radius_delta: &Tensor, anchor_radius: &[f32]) -> Tensor {
    assert_scalar_tensor_shape(radius_delta, anchor_radius.len(), "candidate radius delta");

    let mut predicted_radius = Tensor::zeros(vec![anchor_radius.len(), 1]);

    for (sample, anchor_radius) in anchor_radius.iter().enumerate() {
        let delta = scalar_tensor_value(radius_delta, sample);
        let multiplier = clamped_radius_delta_multiplier(delta);
        let radius = (*anchor_radius).max(VELOCITY_BANK_TIE_EPSILON) * multiplier;

        assert!(
            radius.is_finite() && radius >= 0.0,
            "candidate radius delta produced non-finite radius"
        );
        predicted_radius.set(&[sample, 0], radius);
    }

    predicted_radius
}

fn centered_radius_grad_to_delta_grad(
    radius_delta: &Tensor,
    predicted_radius: &Tensor,
    grad_radius: &Tensor,
) -> Tensor {
    assert_eq!(
        predicted_radius.shape, grad_radius.shape,
        "candidate radius grad shape mismatch"
    );
    assert_scalar_tensor_shape(
        radius_delta,
        predicted_radius.shape[0],
        "candidate radius delta",
    );

    let mut grad_delta = Tensor::zeros(radius_delta.shape.clone());

    for sample in 0..predicted_radius.shape[0] {
        let delta = scalar_tensor_value(radius_delta, sample);
        let grad = if delta.abs() <= CANDIDATE_RADIUS_DELTA_CLAMP {
            scalar_tensor_value(grad_radius, sample) * scalar_tensor_value(predicted_radius, sample)
        } else {
            0.0
        };

        set_scalar_tensor_value(&mut grad_delta, sample, grad);
    }

    grad_delta
}

fn clamped_radius_delta_multiplier(delta: f32) -> f32 {
    assert!(delta.is_finite(), "candidate radius delta must be finite");

    delta
        .clamp(-CANDIDATE_RADIUS_DELTA_CLAMP, CANDIDATE_RADIUS_DELTA_CLAMP)
        .exp()
}

fn assert_scalar_tensor_shape(tensor: &Tensor, batch_size: usize, context: &str) {
    assert!(
        tensor.shape == vec![batch_size, 1] || tensor.shape == vec![batch_size],
        "{} expects scalar tensor shape [{}, 1] or [{}], got {:?}",
        context,
        batch_size,
        batch_size,
        tensor.shape
    );
}

fn scalar_tensor_value(tensor: &Tensor, sample: usize) -> f32 {
    match tensor.shape.as_slice() {
        [_, 1] => tensor.get(&[sample, 0]),
        [_] => tensor.get(&[sample]),
        shape => panic!("scalar tensor expected rank 1 or 2, got {:?}", shape),
    }
}

fn set_scalar_tensor_value(tensor: &mut Tensor, sample: usize, value: f32) {
    match tensor.shape.as_slice() {
        [_, 1] => tensor.set(&[sample, 0], value),
        [_] => tensor.set(&[sample], value),
        shape => panic!("scalar tensor expected rank 1 or 2, got {:?}", shape),
    }
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

#[allow(clippy::too_many_arguments)]
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

fn nearest_unit_candidate_index(prediction_unit: &[f32], candidate_units: &[Vec<f32>]) -> usize {
    assert!(
        !candidate_units.is_empty(),
        "candidate-centroid integration requires non-empty candidate units"
    );

    let mut best_index = 0usize;
    let mut best_distance = vector_squared_distance(prediction_unit, &candidate_units[0]);

    for (candidate_index, candidate_unit) in candidate_units.iter().enumerate().skip(1) {
        let distance = vector_squared_distance(prediction_unit, candidate_unit);
        if distance < best_distance - VELOCITY_BANK_TIE_EPSILON {
            best_index = candidate_index;
            best_distance = distance;
        }
    }

    best_index
}

fn max_logit_index(logits: &[f32]) -> usize {
    assert!(!logits.is_empty(), "selector readout requires logits");

    let mut best_index = 0usize;
    let mut best_logit = logits[0];

    for (candidate_idx, logit) in logits.iter().copied().enumerate().skip(1) {
        if logit > best_logit + VELOCITY_BANK_TIE_EPSILON {
            best_index = candidate_idx;
            best_logit = logit;
        }
    }

    best_index
}

fn softmax_weighted_radius(
    prediction_unit: &[f32],
    candidate_units: &[Vec<f32>],
    candidate_radii: &[f32],
    temperature: f32,
) -> f32 {
    assert_eq!(
        candidate_units.len(),
        candidate_radii.len(),
        "candidate-centroid integration unit/radius count mismatch"
    );
    assert!(
        !candidate_units.is_empty(),
        "candidate-centroid integration requires non-empty candidates"
    );
    assert!(
        temperature.is_finite() && temperature > 0.0,
        "candidate-centroid integration temperature must be finite and positive"
    );

    let logits = candidate_units
        .iter()
        .map(|candidate_unit| {
            -vector_squared_distance(prediction_unit, candidate_unit) / temperature
        })
        .collect::<Vec<_>>();
    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |best, value| best.max(value));
    let mut weight_sum = 0.0f32;
    let mut radius_sum = 0.0f32;

    for (logit, radius) in logits.iter().zip(candidate_radii.iter()) {
        let weight = (*logit - max_logit).exp();
        weight_sum += weight;
        radius_sum += weight * radius;
    }

    if weight_sum <= VELOCITY_BANK_TIE_EPSILON {
        candidate_radii.iter().sum::<f32>() / candidate_radii.len() as f32
    } else {
        radius_sum / weight_sum
    }
}

fn signed_candidate_prior_logits_and_radii(
    prediction: &Tensor,
    candidate_targets: &[Tensor],
    sample: usize,
    projection_dim: usize,
    softmax_temperature: f32,
) -> (Vec<f32>, Vec<f32>) {
    let centroid = signed_candidate_centroid(candidate_targets, sample, projection_dim);
    let prediction_centered = centered_sample_vector(prediction, sample, &centroid);
    let prediction_unit = unit_vector(&prediction_centered);
    let mut prior_logits = Vec::with_capacity(candidate_targets.len());
    let mut candidate_radii = Vec::with_capacity(candidate_targets.len());

    for candidate_target in candidate_targets {
        let candidate_centered = centered_sample_vector(candidate_target, sample, &centroid);
        let candidate_unit = unit_vector(&candidate_centered);

        prior_logits.push(
            -vector_squared_distance(&prediction_unit, &candidate_unit) / softmax_temperature,
        );
        candidate_radii.push(vector_l2_norm(&candidate_centered));
    }

    (prior_logits, candidate_radii)
}

fn stable_softmax(logits: &[f32], context: &str) -> Vec<f32> {
    assert!(!logits.is_empty(), "{} requires non-empty logits", context);

    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |best, value| best.max(value));
    let mut weight_sum = 0.0f32;
    let mut weights = Vec::with_capacity(logits.len());

    for logit in logits {
        let weight = (*logit - max_logit).exp();
        weight_sum += weight;
        weights.push(weight);
    }

    assert!(
        weight_sum.is_finite() && weight_sum > VELOCITY_BANK_TIE_EPSILON,
        "{} produced invalid softmax normalizer {}",
        context,
        weight_sum
    );

    for weight in &mut weights {
        *weight /= weight_sum;
    }

    weights
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
    vector_squared_norm(vector).sqrt()
}

fn vector_squared_norm(vector: &[f32]) -> f32 {
    vector.iter().map(|value| value * value).sum()
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

fn vector_dot(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(
        left.len(),
        right.len(),
        "vector dot expects matching lengths"
    );

    left.iter()
        .zip(right.iter())
        .map(|(left_value, right_value)| left_value * right_value)
        .sum()
}

fn signed_ray_boundary_interval(
    candidate_targets: &[Tensor],
    sample: usize,
    centroid: &[f32],
    prediction_unit: &[f32],
    true_index: usize,
) -> Option<(f32, f32)> {
    assert!(
        true_index < candidate_targets.len(),
        "ray-boundary true index exceeds candidate count"
    );
    assert_eq!(
        centroid.len(),
        prediction_unit.len(),
        "ray-boundary unit dim mismatch"
    );

    let true_centered = centered_sample_vector(&candidate_targets[true_index], sample, centroid);
    let true_norm_sq = vector_squared_norm(&true_centered);
    let mut lower_radius = 0.0f32;
    let mut upper_radius = f32::INFINITY;

    for (candidate_index, candidate_target) in candidate_targets.iter().enumerate() {
        if candidate_index == true_index {
            continue;
        }

        let wrong_centered = centered_sample_vector(candidate_target, sample, centroid);
        let wrong_norm_sq = vector_squared_norm(&wrong_centered);
        let true_minus_wrong = true_centered
            .iter()
            .zip(&wrong_centered)
            .map(|(true_value, wrong_value)| true_value - wrong_value)
            .collect::<Vec<_>>();
        let denominator = 2.0 * vector_dot(prediction_unit, &true_minus_wrong);
        let numerator = true_norm_sq - wrong_norm_sq;

        if denominator > VELOCITY_BANK_TIE_EPSILON {
            lower_radius = lower_radius.max(numerator / denominator);
        } else if denominator < -VELOCITY_BANK_TIE_EPSILON {
            upper_radius = upper_radius.min(numerator / denominator);
        } else if numerator >= -VELOCITY_BANK_TIE_EPSILON {
            return None;
        }
    }

    let required_radius = lower_radius.max(0.0);
    if required_radius + VELOCITY_BANK_TIE_EPSILON >= upper_radius {
        return None;
    }

    Some((required_radius, upper_radius))
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
