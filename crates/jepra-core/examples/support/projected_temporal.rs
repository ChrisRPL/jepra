use super::temporal_vision::{BATCH_SIZE, make_temporal_batch};
use jepra_core::{
    EmbeddingEncoder, Linear, PredictorModule, ProjectedVisionJepa, Tensor,
    gaussian_moment_regularizer, mse_loss,
};

pub const PROJECTED_VALIDATION_BASE_SEED: u64 = 111_000;
pub const PROJECTED_VALIDATION_BATCHES: usize = 8;
pub const PROJECTED_TRAIN_LOSS_MAX_REDUCTION_RATIO: f32 = 0.2;
pub const PROJECTED_VALIDATION_LOSS_MAX_REDUCTION_RATIO: f32 = 0.2;

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

pub fn projected_validation_batch_losses_from_base_seed<P>(
    model: &ProjectedVisionJepa<P>,
    regularizer_weight: f32,
    validation_base_seed: u64,
    validation_batches: usize,
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
        |seed| make_temporal_batch(BATCH_SIZE, seed),
    )
}
