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
        legacy_src.contains("DEPRECATED: train_vision_jepa is legacy"),
        "legacy example should keep deprecation messaging"
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

#[test]
fn temporal_examples_keep_predictor_mode_wiring_explicit() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let unprojected_path = format!("{manifest_dir}/examples/train_vision_jepa_random_temporal.rs");
    let projected_path =
        format!("{manifest_dir}/examples/train_vision_jepa_random_temporal_projected.rs");

    let unprojected_src =
        fs::read_to_string(unprojected_path).expect("unprojected temporal source must exist");
    let projected_src =
        fs::read_to_string(projected_path).expect("projected temporal source must exist");

    assert!(
        unprojected_src.contains("run_config.predictor_mode"),
        "unprojected temporal example should route predictor mode through run config"
    );
    assert!(
        projected_src.contains("run_config.predictor_mode"),
        "projected temporal example should route predictor mode through run config"
    );
    assert!(
        unprojected_src.contains("run_config.temporal_task_mode"),
        "unprojected temporal example should route temporal task through run config"
    );
    assert!(
        projected_src.contains("run_config.temporal_task_mode"),
        "projected temporal example should route temporal task through run config"
    );
}
