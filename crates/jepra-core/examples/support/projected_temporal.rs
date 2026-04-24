use super::temporal_vision::{
    BATCH_SIZE, SIGNED_VELOCITY_BANK_CANDIDATE_DX, TemporalTaskMode, VELOCITY_BANK_CANDIDATE_DX,
    make_signed_velocity_trail_candidate_target_batch, make_temporal_batch_for_task,
    make_velocity_trail_candidate_target_batch, signed_motion_dx_for_sample,
};
use jepra_core::{
    EmbeddingEncoder, Linear, PredictorModule, ProjectedVisionJepa, Tensor,
    gaussian_moment_regularizer, mse_loss,
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
