# JEPRA

JEPRA is a Rust-first framework for compact JEPA-style latent predictive modeling, with a current focus on temporal prediction (vision-based synthetic tasks) rather than broad framework abstraction.
The current crate is still published as `roadjepa-core` while project branding is migrating to `JEPRA`.

## Current Scope (from `VISION.md`)

- frozen-encoder `VisionJepa` and `ProjectedVisionJepa` training paths
- synthetic temporal batch generation and temporal training examples with held-out validation
- deterministic regression test coverage for step, trajectory, and loss-contract behavior
- temporal data now supports one or two moving squares per sample in the synthetic generator

## Core Verification Commands

```bash
cargo test --manifest-path crates/roadjepa-core/Cargo.toml --test temporal_vision_support
cargo test --manifest-path crates/roadjepa-core/Cargo.toml --test projected_temporal_support
cargo clippy --all-targets --all-features
```

## Current Tracked Scope

- Use `cargo test --manifest-path crates/roadjepa-core/Cargo.toml --all-targets` to validate core suite; expected test count may vary with added coverage.
- CI in this repo runs the core checks from `.github/workflows/ci.yml`.
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

## Fast Feedback Loop

For quick iteration:

- `cargo fmt --manifest-path crates/roadjepa-core/Cargo.toml --all -- --check`
- `cargo test --manifest-path crates/roadjepa-core/Cargo.toml --test example_entrypoint_guard`
- `cargo test --manifest-path crates/roadjepa-core/Cargo.toml --test temporal_vision_support`

Use these first; run the full `--all-targets`/clippy suite before PR handoff or CI-sensitive changes.

### Example Entrypoints

- `train_vision_jepa_random_temporal.rs` is the canonical hardening path for the current JEPA temporal proof.
- `train_vision_jepa.rs` remains as a legacy entrypoint and now delegates to the same random-temporal training flow.
