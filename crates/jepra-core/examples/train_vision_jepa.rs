#[path = "train_vision_jepa_random_temporal.rs"]
mod train_vision_jepa_random_temporal;

/// Legacy entrypoint maintained for compatibility.
///
/// Canonical random-temporal JEPA proof now lives in:
/// `train_vision_jepa_random_temporal.rs`
#[deprecated(
    since = "0.1.0",
    note = "train_vision_jepa is legacy; use train_vision_jepa_random_temporal for the current JEPA path."
)]
fn main() {
    eprintln!(
        "DEPRECATED: train_vision_jepa is legacy and delegates to the hardened \
         train_vision_jepa_random_temporal path."
    );
    train_vision_jepa_random_temporal::main();
}
