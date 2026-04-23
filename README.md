# JEPRA

JEPRA is a Rust-first framework for compact JEPA-style latent predictive modeling, with a current focus on temporal prediction (vision-based synthetic tasks) rather than broad framework abstraction.
The crate is published as `jepra-core`.

## Current Scope (from `VISION.md`)

- `VisionJepa` and `ProjectedVisionJepa` training paths with a frozen baseline, a compact frozen-encoder option, and optional trainable-encoder updates in temporal JEPA examples
- synthetic temporal batch generation and temporal training examples with held-out validation
- deterministic regression test coverage for step, trajectory, and loss-contract behavior
- temporal data now supports one or two moving squares per sample in the synthetic generator
- unprojected validation helpers and reduction thresholds are centralized in `crates/jepra-core/examples/support/temporal_validation.rs`
- temporal validation helpers now explicitly panic when validation batches is configured as zero

## Core Verification Commands

```bash
cargo test --manifest-path crates/jepra-core/Cargo.toml --test temporal_vision_support
cargo test --manifest-path crates/jepra-core/Cargo.toml --test projected_temporal_support
cargo clippy --manifest-path crates/jepra-core/Cargo.toml --all-targets --all-features
```

## Current Tracked Scope

- Use `cargo test --manifest-path crates/jepra-core/Cargo.toml --all-targets` to validate core suite; expected test count may vary with added coverage.
- CI in this repo runs the core checks from `.github/workflows/ci.yml`.
- `cargo test --manifest-path crates/jepra-core/Cargo.toml --test temporal_vision_support` and `cargo test --manifest-path crates/jepra-core/Cargo.toml --test projected_temporal_support` should both pass with deterministic loss reductions and reproducible trajectories
- `train_vision_jepa_random_temporal.rs` and `train_vision_jepa_random_temporal_projected.rs` are the hardening examples for the JEPA proof path
- `train_vision_jepa.rs` remains legacy and delegates to `train_vision_jepa_random_temporal.rs` via `train_vision_jepa_random_temporal::main()`.

## Useful Check Commands

```bash
cargo fmt --manifest-path crates/jepra-core/Cargo.toml
cargo test --manifest-path crates/jepra-core/Cargo.toml --all-targets
JEPRA_TRAIN_STEPS=12 cargo run --manifest-path crates/jepra-core/Cargo.toml --example train_vision_jepa_random_temporal
JEPRA_TRAIN_STEPS=12 cargo run --manifest-path crates/jepra-core/Cargo.toml --example train_vision_jepa
JEPRA_TRAIN_STEPS=12 cargo run --manifest-path crates/jepra-core/Cargo.toml --example train_vision_jepa_random_temporal_projected
# explicit flags now supported:
# cargo run --manifest-path crates/jepra-core/Cargo.toml --example train_vision_jepa_random_temporal -- --train-base-seed 1400 --train-steps 40 --log-every 10
# cargo run --manifest-path crates/jepra-core/Cargo.toml --example train_vision_jepa_random_temporal -- --encoder-lr 0.005 --train-steps 60 --train-base-seed 1400
# cargo run --manifest-path crates/jepra-core/Cargo.toml --example train_vision_jepa_random_temporal_projected -- --seed 21000 --steps 80 --log 20
# cargo run --manifest-path crates/jepra-core/Cargo.toml --example train_vision_jepa_random_temporal_projected -- --target-momentum 0.95 --train-steps 120
```

### Temporal Example CLI

Temporal examples accept shared args via `TemporalRunConfig`:

- `--train-base-seed` (`--seed`) sets the base training seed
- `--train-steps` (`--steps`) sets total training steps
- `--log-every` (`--log`) sets log cadence (must be > 0)
- `--encoder-lr` (or `--encoder-learning-rate`) enables encoder updates in temporal JEPA runs; `0.0` keeps a frozen encoder baseline
- `--compact-encoder` enables compact frozen encoder mode
- `--compact-encoder-mode <base|stronger>` selects compact mode explicitly (`--compact-encoder` defaults to `stronger`, `--compact-encoder-mode base` opts into the original compact variant)
- `--target-momentum` (or `--target-projection-momentum`) sets EMA momentum for the projected path target projector (`1.0` keeps target projector frozen)
- `--target-momentum-start` sets the starting EMA momentum when warmup is enabled
- `--target-momentum-end` sets the final EMA momentum target (defaults to `--target-momentum`)
- `--target-momentum-end` can also be passed as `--target-projection-momentum-end`
- `--target-momentum-warmup-steps` linearly interpolates momentum from start to end over the first N steps (alias: `--target-projection-warmup-steps`)
- `JEPRA_TRAIN_STEPS` is the environment fallback when step flags are not passed
- `JEPRA_ENCODER_LR` is an environment fallback when encoder-learning flags are not passed
- `JEPRA_TARGET_MOMENTUM` is an environment fallback for projected target-projector momentum

### Evidence Snapshot

- [ ] Projected momentum hardening protocol (fixed-seed sweeps) using the projected temporal entrypoint:
  - warmup: `for SEED in 21000 21001 21002; do cargo run --manifest-path crates/jepra-core/Cargo.toml --example train_vision_jepa_random_temporal_projected -- --train-base-seed "$SEED" --train-steps 80 --encoder-lr 0.0 --target-momentum-start 1.0 --target-momentum-end 0.5 --target-momentum-warmup-steps 24 --log 20; done`
  - frozen: `for SEED in 21000 21001 21002; do cargo run --manifest-path crates/jepra-core/Cargo.toml --example train_vision_jepa_random_temporal_projected -- --train-base-seed "$SEED" --train-steps 80 --encoder-lr 0.0 --target-momentum 1.0 --log 20; done`
  - trainable: `for SEED in 21000 21001 21002; do cargo run --manifest-path crates/jepra-core/Cargo.toml --example train_vision_jepa_random_temporal_projected -- --train-base-seed "$SEED" --train-steps 80 --encoder-lr 0.004 --target-momentum 0.5 --log 20; done`

## Fast Feedback Loop

For quick iteration:

- `cargo fmt --manifest-path crates/jepra-core/Cargo.toml --all -- --check`
- `cargo test --manifest-path crates/jepra-core/Cargo.toml --test example_entrypoint_guard`
- `cargo test --manifest-path crates/jepra-core/Cargo.toml --test temporal_vision_support`
- `cargo test --manifest-path crates/jepra-core/Cargo.toml --test temporal_vision_support unprojected_validation_batch_losses_zero_batch_count_panics`
- `cargo test --manifest-path crates/jepra-core/Cargo.toml --test projected_temporal_support projected_validation_batch_losses_zero_batch_count_panics`

Use these first; run the full `--all-targets`/clippy suite before PR handoff or CI-sensitive changes.

### Example Entrypoints

- `train_vision_jepa_random_temporal.rs` and `train_vision_jepa_random_temporal_projected.rs` form the core JEPA temporal proof pair.
- `train_vision_jepa.rs` remains as a legacy entrypoint and now delegates to the same random-temporal training flow.
