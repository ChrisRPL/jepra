# JEPRA

JEPRA is a Rust-first framework for compact JEPA-style latent predictive modeling, with a current focus on temporal prediction (vision-based synthetic tasks) rather than broad framework abstraction.
The crate is published as `jepra-core`.

## Current Scope (from `VISION.md`)

- `VisionJepa` and `ProjectedVisionJepa` training paths with a frozen baseline, a compact frozen-encoder option, and optional trainable-encoder updates in temporal JEPA examples
- baseline two-layer predictors plus opt-in `BottleneckPredictor` and `ResidualBottleneckPredictor` variants for compact-capacity experiments
- lightweight representation-health telemetry for predictor/target comparisons (`mean_abs`, `mean_std`, `min_std`, mean/max off-diagonal covariance)
- synthetic temporal batch generation and temporal training examples with held-out validation
- deterministic regression test coverage for step, trajectory, and loss-contract behavior
- core JEPA projection regularizer utilities for Gaussian moment regularization and projection statistics
- temporal data now supports one or two moving squares per sample in the synthetic generator
- temporal examples expose opt-in `velocity-trail` and `signed-velocity-trail` tasks that add previous-position cues while preserving the default `random-speed` task
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
# cargo run --manifest-path crates/jepra-core/Cargo.toml --example train_vision_jepa_random_temporal_projected -- --temporal-task velocity-trail --train-steps 40
# cargo run --manifest-path crates/jepra-core/Cargo.toml --example train_vision_jepa_random_temporal_projected -- --temporal-task signed-velocity-trail --train-steps 40
./run-predictor-mode-comparison.sh all # compare baseline/bottleneck/residual predictor modes
```

### Temporal Example CLI

Temporal examples accept shared args via `TemporalRunConfig`:

- `--train-base-seed` (`--seed`) sets the base training seed
- `--train-steps` (`--steps`) sets total training steps
- `--log-every` (`--log`) sets log cadence (must be > 0)
- `--temporal-task <random-speed|velocity-trail|signed-velocity-trail>` selects the synthetic temporal task (`random-speed` remains the default; trail tasks are opt-in diagnostics)
- `--encoder-lr` (or `--encoder-learning-rate`) enables encoder updates in temporal JEPA runs; `0.0` keeps a frozen encoder baseline
- `--compact-encoder` enables compact frozen encoder mode
- `--compact-encoder-mode <base|stronger>` selects compact mode explicitly (`--compact-encoder` defaults to `stronger`, `--compact-encoder-mode base` opts into the original compact variant)
- `--predictor-mode <baseline|bottleneck|residual-bottleneck>` selects the predictor topology (`baseline` is the default; the others are experimental and non-default)
- `--residual-delta-scale <float>` scales only the residual-bottleneck delta branch (`1.0` preserves the unscaled identity-plus-delta predictor)
- `--projector-drift-weight <float>` adds an opt-in L2 online-projector-to-target-projector drift regularizer in projected runs (`0.0` disables it)
- `--target-momentum` (or `--target-projection-momentum`) sets EMA momentum for the projected path target projector (`1.0` keeps target projector frozen)
- `--target-momentum-start` sets the starting EMA momentum when warmup is enabled
- `--target-momentum-end` sets the final EMA momentum target (defaults to `--target-momentum`)
- `--target-momentum-end` can also be passed as `--target-projection-momentum-end`
- `--target-momentum-warmup-steps` linearly interpolates momentum from start to end over the first N steps (alias: `--target-projection-warmup-steps`)
- `JEPRA_TRAIN_STEPS` is the environment fallback when step flags are not passed
- `JEPRA_TEMPORAL_TASK` is the environment fallback for the temporal task
- `JEPRA_ENCODER_LR` is an environment fallback when encoder-learning flags are not passed
- `JEPRA_RESIDUAL_DELTA_SCALE` is an environment fallback for residual-bottleneck delta scale
- `JEPRA_PROJECTOR_DRIFT_WEIGHT` is an environment fallback for projected online-projector drift regularization
- `JEPRA_TARGET_MOMENTUM` is an environment fallback for projected target-projector momentum

### Evidence Snapshot

Predictor-mode comparison protocol:

```bash
./run-predictor-mode-comparison.sh all
JEPRA_TEMPORAL_TASK=velocity-trail ./run-predictor-mode-comparison.sh projected
JEPRA_TEMPORAL_TASK=signed-velocity-trail ./run-predictor-mode-comparison.sh projected
JEPRA_PREDICTOR_COMPARISON_REPORT=/tmp/jepra-predictor-compare.csv ./run-predictor-mode-comparison.sh all
```

The script prints one structured row per path/seed/predictor:

```text
schema=jepra_predictor_compare_v9 temporal_task=<random-speed|velocity-trail|signed-velocity-trail> path=<unprojected|projected> predictor=<baseline|bottleneck|residual-bottleneck> residual_delta_scale=<n> projector_drift_weight=<n> seed=<seed> steps=<steps> ... pred_min_std_final=<n> target_min_std_final=<n> velocity_bank_mrr_end=<n|na> velocity_bank_top1_end=<n|na> signed_bank_neg_mrr_end=<n|na> signed_bank_sign_top1_end=<n|na> signed_bank_speed_top1_end=<n|na> target_bank_oracle_mrr_end=<n|na> target_bank_margin_end=<n|na> prediction_bank_margin_end=<n|na> prediction_bank_positive_margin_rate_end=<n|na> signed_objective_all_loss_end=<n|na> signed_objective_sign_gap_end=<n|na> signed_objective_speed_gap_end=<n|na> status=<ok|accept_failed|run_failed|parse_failed>
```

Latest predictor comparison evidence (`2026-04-24`, `random-speed` task, 300 steps, frozen-base encoder, projected target momentum `1.0`, residual delta scale `1.0`, projector drift weight `0.0`):

```text
schema=jepra_predictor_compare_v5 temporal_task=random-speed path=unprojected predictor=baseline residual_delta_scale=1.0 projector_drift_weight=0.0 seed=1000 steps=300 val_pred_end=0.040150 pred_min_std_final=1.254614 velocity_bank_mrr_end=na status=ok
schema=jepra_predictor_compare_v5 temporal_task=random-speed path=unprojected predictor=bottleneck residual_delta_scale=1.0 projector_drift_weight=0.0 seed=1000 steps=300 val_pred_end=0.127445 pred_min_std_final=1.245264 velocity_bank_mrr_end=na status=ok
schema=jepra_predictor_compare_v5 temporal_task=random-speed path=unprojected predictor=residual-bottleneck residual_delta_scale=1.0 projector_drift_weight=0.0 seed=1000 steps=300 val_pred_end=0.074866 pred_min_std_final=1.063676 velocity_bank_mrr_end=na status=ok
schema=jepra_predictor_compare_v5 temporal_task=random-speed path=projected predictor=baseline residual_delta_scale=1.0 projector_drift_weight=0.0 seed=11000 steps=300 val_pred_end=0.216322 pred_min_std_final=0.990177 target_drift_end=0.036044 velocity_bank_mrr_end=na status=ok
schema=jepra_predictor_compare_v5 temporal_task=random-speed path=projected predictor=bottleneck residual_delta_scale=1.0 projector_drift_weight=0.0 seed=11000 steps=300 val_pred_end=5.042953 pred_min_std_final=0.000000 target_drift_end=0.002192 velocity_bank_mrr_end=na status=accept_failed
schema=jepra_predictor_compare_v5 temporal_task=random-speed path=projected predictor=residual-bottleneck residual_delta_scale=1.0 projector_drift_weight=0.0 seed=11000 steps=300 val_pred_end=0.045684 pred_min_std_final=0.972417 target_drift_end=0.031384 velocity_bank_mrr_end=na status=ok
```

Interpretation: residual-bottleneck is the current projected-path candidate because it keeps prediction health close to target health and materially beats projected baseline on this seed. It is not promoted as a default because unprojected baseline still has the lower final validation loss.

Compact-stronger projected predictor evidence (`2026-04-24`, 300 steps, seeds `11000..11002`, target momentum `1.0`, zero-init residual delta head):

```text
baseline mean val_pred_end=1.158821 | mean pred_min_std=0.467862 | mean target_drift=0.006282 | status=ok all seeds
bottleneck mean val_pred_end=1.299992 | mean pred_min_std=0.000000 | mean target_drift=0.000701 | status=accept_failed all seeds
residual-bottleneck mean val_pred_end=1.119540 | mean pred_min_std=0.473802 | mean target_drift=0.140347 | status=ok all seeds
```

Interpretation: residual-bottleneck remains healthy and modestly improves compact-stronger projected validation on 2/3 seeds, but target drift is much higher than baseline. Treat it as a projected candidate with drift confound, not a default promotion or a trigger for depthwise work.

Residual delta-scale hardening evidence (`2026-04-24`, compact-stronger projected, seeds `11000..11002`):

```text
scale=0.25 target_momentum=1.0 residual mean val_pred_end=1.137108 | mean pred_min_std=0.517087 | mean target_drift=0.133765
scale=0.25 target_momentum=0.5 residual mean val_pred_end=0.004085 | mean pred_min_std=0.028745 | mean target_min_std=0.023829 | status=accept_failed by health gate
```

Interpretation: delta scaling is now an explicit experiment knob, not a promoted topology change. Scale `0.25` preserves compact-stronger prediction health under frozen target momentum but does not solve projector drift; pairing it with low target momentum creates a loss-only collapse. Defaults remain `--predictor-mode baseline`, residual scale `1.0`, `--target-momentum 1.0`.

Projector drift regularizer evidence (`2026-04-24`, compact-stronger projected, residual-bottleneck, seed `11000`, target momentum `1.0`):

```text
projector_drift_weight=1.0 val_pred_end=1.126885 | pred_min_std=0.425927 | target_drift=0.137658 | status=ok
projector_drift_weight=5.0 val_pred_end=1.165685 | pred_min_std=0.462189 | target_drift=0.126054 | status=ok
projector_drift_weight=10.0 val_pred_end=1.216697 | pred_min_std=0.502000 | target_drift=0.113993 | status=ok
```

Interpretation: parameter-space projector drift regularization is a valid opt-in control knob and reduces drift monotonically on this seed, but the observed tradeoff is not yet good enough for promotion. Keep it as evidence tooling; trail-task evidence remains the active architecture filter.

Velocity-trail compact-stronger projected evidence (`2026-04-24`, 300 steps, seeds `11000..11002`, target momentum `1.0`, residual delta scale `1.0`, projector drift weight `0.0`):

```text
baseline mean val_pred_end=0.119354 | mean pred_min_std=0.101571 | mean target_drift=0.002187 | status=ok all seeds
residual-bottleneck mean val_pred_end=0.178396 | mean pred_min_std=0.226671 | mean target_drift=0.159557 | status=ok all seeds
baseline mean velocity_bank_mrr=0.817708 | mean velocity_bank_top1=0.635417 | mean velocity_bank_rank=1.364583
residual-bottleneck mean velocity_bank_mrr=0.835938 | mean velocity_bank_top1=0.671875 | mean velocity_bank_rank=1.328125
```

Interpretation: baseline wins the harder `velocity-trail` task on all three seeds and ranks the two-speed bank above random (`MRR=0.75`, `top1=0.5`). Residual-bottleneck has slightly higher speed-bank ranking but remains validation-worse and drift-confounded, so this evidence still blocks residual promotion and depthwise/spatial primitive work.

Signed velocity-trail compact-stronger projected evidence (`2026-04-24`, 300 steps, seeds `11000..11002`, target momentum `1.0`, residual delta scale `1.0`, projector drift weight `0.0`):

```text
baseline mean val_pred_end=1.305638 | mean pred_min_std=0.177438 | mean target_drift=0.008615 | status=ok all seeds
residual-bottleneck mean val_pred_end=1.385723 | mean pred_min_std=0.312411 | mean target_drift=0.136066 | status=ok all seeds
baseline mean velocity_bank_mrr=0.519965 | mean velocity_bank_top1=0.239583 | mean velocity_bank_rank=2.484375
residual-bottleneck mean velocity_bank_mrr=0.489583 | mean velocity_bank_top1=0.203125 | mean velocity_bank_rank=2.593750
baseline signed breakdown neg_mrr=0.386285 | pos_mrr=0.653646 | slow_mrr=0.489583 | fast_mrr=0.550347 | sign_top1=0.479167 | speed_top1=0.494792
residual signed breakdown neg_mrr=0.309896 | pos_mrr=0.669271 | slow_mrr=0.489583 | fast_mrr=0.489583 | sign_top1=0.375000 | speed_top1=0.635417
target bank oracle_mrr=1.000000 | oracle_top1=1.000000 | true_distance=0.000000 | margin=1.463823 | min_margin=0.001335
target bank neg_nearest_wrong=1.938479 | pos_nearest_wrong=0.989167 | sign_margin=7.331476 | speed_margin=-12.402091
baseline prediction margin true_distance=5.222552 | nearest_wrong=1.063438 | margin=-4.159113 | positive_margin_rate=0.239583 | sign_margin=-0.981234 | speed_margin=0.185016
residual prediction margin true_distance=5.542892 | nearest_wrong=0.854323 | margin=-4.688569 | positive_margin_rate=0.203125 | sign_margin=-1.366664 | speed_margin=0.422940
baseline signed objective all=1.305638 | neg=2.118653 | pos=0.492622 | slow=1.349482 | fast=1.261794 | sign_gap=-1.626031 | speed_gap=-0.087688
residual signed objective all=1.385723 | neg=2.204694 | pos=0.566752 | slow=1.327650 | fast=1.443796 | sign_gap=-1.637942 | speed_gap=0.116146
```

Interpretation: baseline wins validation on all three signed-task seeds. Residual-bottleneck has higher prediction spread but ~16x higher target drift and weaker four-candidate signed velocity-bank ranking. The v7 target-bank oracle is clean (`oracle_mrr=1.0`, true distance `0.0`), so signed failure is not candidate-target construction. The v8 prediction-margin diagnostic shows the actual prediction is much closer to wrong signed candidates than the true target on most samples (`positive_margin_rate` near top1). The v9 objective decomposition shows the main signed loss is negative-direction error (`neg` is ~4x `pos`) and residual does not fix it. This blocks residual promotion, depthwise convolution, and spatial predictor work; the next valid build step is a default-off signed-margin objective probe.

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
