use std::fs;

#[test]
fn legacy_vision_jepa_example_delegates_to_random_temporal_entrypoint() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let legacy_path = format!("{manifest_dir}/examples/train_vision_jepa.rs");
    let random_path = format!("{manifest_dir}/examples/train_vision_jepa_random_temporal.rs");

    let legacy_src = fs::read_to_string(legacy_path).expect("legacy example source must exist");
    let random_src =
        fs::read_to_string(random_path).expect("random-temporal example source must exist");

    assert!(
        legacy_src.contains("train_vision_jepa_random_temporal::main()"),
        "legacy example should delegate to random-temporal main entrypoint"
    );
    assert!(
        !legacy_src.contains("Conv2d::new("),
        "legacy example should not contain duplicated Conv2d setup"
    );
    assert!(
        !legacy_src.contains("Predictor::new("),
        "legacy example should not contain duplicated predictor construction"
    );
    assert!(
        random_src.contains("fn main()"),
        "random-temporal example should still expose a canonical main entrypoint"
    );
}
