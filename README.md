# JEPRA

JEPRA is a Rust-first framework for compact JEPA-style latent predictive modeling, with a current focus on temporal prediction (vision-based synthetic tasks) rather than broad framework abstraction.
The crate is published as `jepra-core`.

## Current Scope (from `VISION.md`)

- `VisionJepa` and `ProjectedVisionJepa` training paths with a frozen baseline, a compact frozen-encoder option, and optional trainable-encoder updates in temporal JEPA examples
- baseline two-layer predictors plus an opt-in `BottleneckPredictor` for compact-capacity experiments
- synthetic temporal batch generation and temporal training examples with held-out validation
- deterministic regression test coverage for step, trajectory, and loss-contract behavior
- core JEPA projection regularizer utilities for Gaussian moment regularization and projection statistics
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
- `--predictor-mode <baseline|bottleneck>` selects the predictor topology (`baseline` is the default; `bottleneck` is experimental and non-default)
- `--target-momentum` (or `--target-projection-momentum`) sets EMA momentum for the projected path target projector (`1.0` keeps target projector frozen)
- `--target-momentum-start` sets the starting EMA momentum when warmup is enabled
- `--target-momentum-end` sets the final EMA momentum target (defaults to `--target-momentum`)
- `--target-momentum-end` can also be passed as `--target-projection-momentum-end`
- `--target-momentum-warmup-steps` linearly interpolates momentum from start to end over the first N steps (alias: `--target-projection-warmup-steps`)
- `JEPRA_TRAIN_STEPS` is the environment fallback when step flags are not passed
- `JEPRA_ENCODER_LR` is an environment fallback when encoder-learning flags are not passed
- `JEPRA_TARGET_MOMENTUM` is an environment fallback for projected target-projector momentum

### Evidence Snapshot

Projected momentum hardening protocol (fixed-seed sweeps) for `train_vision_jepa_random_temporal_projected`:

```bash
./run-projected-momentum-sweep.sh                  # run warmup+frozen+trainable+zero
./run-projected-momentum-sweep.sh warmup           # run warmup only
./run-projected-momentum-sweep.sh frozen           # run frozen only
./run-projected-momentum-sweep.sh trainable        # run trainable only
./run-projected-momentum-sweep.sh zero            # run 0.0 momentum only
JEPRA_MOMENTUM_SWEEP_REPORT=./artifacts/projected-momentum-sweep.csv ./run-projected-momentum-sweep.sh all # emit parsed CSV
```

The script prints one structured line per seed/momentum:

```text
seed=<seed> momentum=<1.0|0.5|0.0|1.0->0.5> profile=<warmup|frozen|trainable|zero> status=ok | projected run summary | steps 80 | train <start> -> <end> (Δ <delta>, improved=<true|false>) | val <start> -> <end> (Δ <delta>, improved=<true|false>) | target drift <start> -> <end> (Δ <delta>)
```

Latest captured fixed-seed hardening evidence (`2026-04-23`, 80 steps, log 20):

```text
seed=21000 momentum=1.0->0.5 profile=warmup status=ok | projected run summary | steps 80 | train 22.758507 -> 0.376958 (Δ -22.381550, improved=true) | val 24.749554 -> 0.427223 (Δ -24.322330, improved=true) | target drift 0.000000 -> 0.000641 (Δ +0.000641)
seed=21001 momentum=1.0->0.5 profile=warmup status=ok | projected run summary | steps 80 | train 17.251961 -> 0.253679 (Δ -16.998281, improved=true) | val 24.749554 -> 0.349486 (Δ -24.400068, improved=true) | target drift 0.000000 -> 0.000310 (Δ +0.000310)
seed=21002 momentum=1.0->0.5 profile=warmup status=ok | projected run summary | steps 80 | train 32.680893 -> 0.320816 (Δ -32.360077, improved=true) | val 24.749554 -> 0.325427 (Δ -24.424128, improved=true) | target drift 0.000000 -> 0.000134 (Δ +0.000134)
seed=21000 momentum=1.0 profile=frozen status=ok | projected run summary | steps 80 | train 22.758507 -> 0.492315 (Δ -22.266191, improved=true) | val 24.749554 -> 0.589146 (Δ -24.160408, improved=true) | target drift 0.000000 -> 0.016285 (Δ +0.016285)
seed=21001 momentum=1.0 profile=frozen status=ok | projected run summary | steps 80 | train 17.251961 -> 0.388103 (Δ -16.863857, improved=true) | val 24.749554 -> 0.427732 (Δ -24.321821, improved=true) | target drift 0.000000 -> 0.018250 (Δ +0.018250)
seed=21002 momentum=1.0 profile=frozen status=ok | projected run summary | steps 80 | train 32.680893 -> 0.366556 (Δ -32.314335, improved=true) | val 24.749554 -> 0.327721 (Δ -24.421833, improved=true) | target drift 0.000000 -> 0.014827 (Δ +0.014827)
seed=21000 momentum=0.5 profile=trainable status=ok | projected run summary | steps 80 | train 22.758507 -> 0.284548 (Δ -22.473959, improved=true) | val 24.749554 -> 0.313719 (Δ -24.435835, improved=true) | target drift 0.000000 -> 0.000065 (Δ +0.000065)
seed=21001 momentum=0.5 profile=trainable status=ok | projected run summary | steps 80 | train 17.251961 -> 0.194428 (Δ -17.057533, improved=true) | val 24.749554 -> 0.351039 (Δ -24.398514, improved=true) | target drift 0.000000 -> 0.000210 (Δ +0.000210)
seed=21002 momentum=0.5 profile=trainable status=ok | projected run summary | steps 80 | train 32.680893 -> 0.326627 (Δ -32.354267, improved=true) | val 24.749554 -> 0.315798 (Δ -24.433756, improved=true) | target drift 0.000000 -> 0.000216 (Δ +0.000216)
seed=21000 momentum=0.0 profile=zero status=ok | projected run summary | steps 80 | train 22.758507 -> 0.347984 (Δ -22.410522, improved=true) | val 24.749554 -> 0.389587 (Δ -24.359966, improved=true) | target drift 0.000000 -> 0.000000 (Δ +0.000000)
seed=21001 momentum=0.0 profile=zero status=ok | projected run summary | steps 80 | train 17.251961 -> 0.223455 (Δ -17.028505, improved=true) | val 24.749554 -> 0.345468 (Δ -24.404085, improved=true) | target drift 0.000000 -> 0.000000 (Δ +0.000000)
seed=21002 momentum=0.0 profile=zero status=ok | projected run summary | steps 80 | train 32.680893 -> 0.322963 (Δ -32.357929, improved=true) | val 24.749554 -> 0.327485 (Δ -24.422068, improved=true) | target drift 0.000000 -> 0.000000 (Δ +0.000000)
```

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
