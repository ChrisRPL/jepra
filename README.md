# RoadJEPA

RoadJEPA is a Rust-first framework for compact JEPA-style latent predictive modeling, with a current focus on temporal prediction (vision-based synthetic tasks) rather than broad framework abstraction.

## Current Scope (from `VISION.md`)

- frozen-encoder `VisionJepa` and `ProjectedVisionJepa` training paths
- synthetic temporal batch generation and temporal training examples with held-out validation
- deterministic regression test coverage for step, trajectory, and loss-contract behavior

## Core Verification Commands

```bash
cargo test --manifest-path crates/roadjepa-core/Cargo.toml --test temporal_vision_support
cargo test --manifest-path crates/roadjepa-core/Cargo.toml --test projected_temporal_support
cargo clippy --all-targets --all-features
```

## Current Tracked Scope

- 102 tests pass via `cargo test --manifest-path crates/roadjepa-core/Cargo.toml --all-targets`
- `cargo test --test temporal_vision_support` and `--test projected_temporal_support` should both pass with deterministic loss reductions and reproducible trajectories
- `train_vision_jepa_random_temporal.rs` and `train_vision_jepa_random_temporal_projected.rs` are the hardening examples for the JEPA proof path
- `train_vision_jepa.rs` remains legacy and delegates to `train_vision_jepa_random_temporal.rs` via `train_vision_jepa_random_temporal::main()`.

## Useful Check Commands

```bash
cargo fmt --manifest-path crates/roadjepa-core/Cargo.toml
cargo test --manifest-path crates/roadjepa-core/Cargo.toml --all-targets
cargo run --manifest-path crates/roadjepa-core/Cargo.toml --example train_vision_jepa_random_temporal
cargo run --manifest-path crates/roadjepa-core/Cargo.toml --example train_vision_jepa
cargo run --manifest-path crates/roadjepa-core/Cargo.toml --example train_vision_jepa_random_temporal_projected
```

### Example Entrypoints

- `train_vision_jepa_random_temporal.rs` is the canonical hardening path for the current JEPA temporal proof.
- `train_vision_jepa.rs` remains as a legacy entrypoint and now delegates to the same random-temporal training flow.
