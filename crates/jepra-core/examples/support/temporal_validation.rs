use super::temporal_vision::make_validation_batch;
use jepra_core::{PredictorModule, Tensor, VisionJepa};

pub const UNPROJECTED_TRAIN_LOSS_MAX_REDUCTION_RATIO: f32 = 0.5;
pub const UNPROJECTED_VALIDATION_LOSS_MAX_REDUCTION_RATIO: f32 = 0.9;
pub const UNPROJECTED_VALIDATION_BASE_SEED: u64 = 99_001;
pub const UNPROJECTED_VALIDATION_BATCHES: usize = 8;

pub fn temporal_validation_batch_loss<P, F>(
    model: &VisionJepa<P>,
    base_seed: u64,
    validation_batches: usize,
    make_batch: F,
) -> f32
where
    P: PredictorModule,
    F: Fn(u64) -> (Tensor, Tensor),
{
    assert!(
        validation_batches > 0,
        "validation_batches must be greater than 0"
    );

    let mut total = 0.0;

    for batch_idx in 0..validation_batches {
        let (x_t, x_t1) = make_batch(base_seed + batch_idx as u64);
        total += model.losses(&x_t, &x_t1).0;
    }

    total / validation_batches as f32
}

pub fn temporal_validation_batch_loss_from_base_seed<P>(
    model: &VisionJepa<P>,
    validation_base_seed: u64,
    validation_batches: usize,
) -> f32
where
    P: PredictorModule,
{
    temporal_validation_batch_loss(
        model,
        validation_base_seed,
        validation_batches,
        |batch_idx| make_validation_batch(validation_base_seed, batch_idx),
    )
}
