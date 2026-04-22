#[path = "train_vision_jepa_random_temporal.rs"]
mod train_vision_jepa_random_temporal;

fn main() {
    println!("train_vision_jepa.rs is legacy and delegates to the hardened random-temporal path.");
    train_vision_jepa_random_temporal::run();
}
