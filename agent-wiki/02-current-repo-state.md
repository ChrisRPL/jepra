# Current Repo State

Snapshot from local files only. Internal working note for the repo state today.

## Status
- JEPA proof currently uses temporal random-video tasks as the forcing function.
- Projected path now has EMA target projector + optional encoder training (`step_with_trainable_encoder`).
- Regression coverage is in `crates/jepra-core/tests/projected_temporal_support.rs`; fixed-seed hardening evidence is captured in README and reproducible via `run-projected-momentum-sweep.sh`.

## Next Action
- Current phase focus has shifted from projected hardening to compact model capacity:
  - keep default path conservative (`--encoder-lr=0.0`, `target_projection_momentum=1.0`, `--predictor-mode=baseline`),
  - add and compare opt-in Rust-native predictor variants without changing defaults,
  - use lightweight representation-health stats to compare prediction and target behavior,
  - keep the fixed-seed projected sweep as the regression baseline before default or projected-policy changes.

## Anti-Goals
- no implicit trainable projected encoder defaults
- no untracked config defaults
- no broad API expansion to support non-JEPA workloads
- no bypass of explicit target-projector momentum intent

## High-Level Read

- Vision has moved to `JEPRA`, a Rust-first JEPA systems framework (`VISION.md:1-10`).
- Naming is now consistently `JEPRA` at strategy level and `jepra-core` at crate level (`crates/jepra-core/Cargo.toml`, `README.md:1-4`).
- Current implementation is one Rust crate, `crates/jepra-core`, with tensor/linear/predictor/conv/encoder/projection core and temporal JEPA training loops.
- Core suite includes temporal and projected regression coverage for optimizer, target-momentum, and encoder-training protocol behavior.

## Current Modules

- `tensor.rs`: shape/rank validation, indexing, `get`/`set`, elementwise add, inplace updates, `matmul`, `transpose`, `sum_axis0`, `relu`, `relu_backward`, `global_avg_pool2d`, and `global_avg_pool2d_backward`.
- `linear.rs`: `Linear`, `LinearGrads`, forward, backward, SGD step.
- `predictor.rs`: baseline two-layer predictor plus bottleneck and residual-bottleneck predictors, with ReLU, backward pass, and SGD step.
- `conv.rs`: `Conv2d` with forward, backward, and SGD step APIs.
- `encoder.rs`: `ConvEncoder` and `EmbeddingEncoder` on top of conv + global average pooling.
- `regularizers.rs`: JEPA projection regularizer utilities (`gaussian_moment_regularizer`, gradient, projection stats, representation stats, projection-grad combination).
- `vision_jepa.rs`: `VisionJepa` and `ProjectedVisionJepa` wrappers that compose encoders + predictor/projector and expose core encode/predict/loss/update entry points.
- `losses.rs` and `init.rs`: MSE loss/grad and deterministic random init helpers.

## Examples And Tests

- `crates/jepra-core/examples/train_predictor.rs:1-37` is a tiny supervised regression demo.
- `crates/jepra-core/examples/train_vision_jepa.rs` is now a compatibility wrapper.
- `crates/jepra-core/examples/train_vision_jepa_random_temporal.rs` is the canonical JEPA temporal example with fresh batch generation, held-out validation, and motion constraints.
- `crates/jepra-core/examples/train_vision_jepa_random_temporal_projected.rs` is the projected-path counterpart.
- Unit tests and temporal suites enforce relative convergence from own-batch baselines.
- Projected suite (`tests/projected_temporal_support.rs`) now covers:
  - target projector EMA edge cases (`momentum = 0.0`, `1.0`, and interior values),
  - invalid momentum rejection,
  - projected encoder-training step coverage,
- projected compact-mode frozen vs trainable parity (`CompactEncoderMode::Base`) and stronger compact-mode coverage in protocol checks.

## What Is Already JEPA-Capable

- The framework already has:
  - unprojected JEPA scaffold (`VisionJepa` with optional encoder updates),
  - projected JEPA scaffold with target-projector EMA and optional projected encoder updates.
- `VisionJepa` is the right conceptual boundary for latent prediction.
- The current conv encoder path supports batched image-like tensors and latent pooling, so the code is already past pure scalar toy math.

## What Is Still Toy Or Trivial

- `train_vision_jepa.rs` is intentionally a compatibility wrapper that delegates to the canonical random-temporal training flow.
- A dedicated example entrypoint regression test now enforces that delegation in `crates/jepra-core/tests/example_entrypoint_guard.rs`.
- Temporal data uses mixed-speed synthetic motion, with deterministic mixed-mode validation probes and held-out schedules.
- CLI/env config for runs lives in `examples/support/temporal_vision.rs` and now includes `--predictor-mode`, `--target-momentum`/`--target-momentum-start`/`--target-momentum-end` plus `--target-momentum-warmup-steps` and `JEPRA_TARGET_MOMENTUM`.
- `Conv2d` has backprop support and optimizer updates; encoder-level backward/grad plumbing is now in place.
- There is no real evaluation harness, no benchmark runner, and no model checkpointing or serialization path.

## Strengths

- Small surface area, easy to inspect.
- Core tensor API is consistent and aggressively shape-checked.
- Deterministic init makes examples and tests reproducible.
- The repo already expresses the intended JEPA shape instead of hiding it behind generic ML abstractions.

## Limitations And Technical Debt

- Naming drift: no active `RoadJEPA`-vs-`JEPRA` naming split in tracked metadata (`crates/jepra-core/Cargo.toml`).
- Scope drift: reduced; this wiki now tracks active temporal JEPA work and validation-focused paths.
- Missing infrastructure: no workspace-level organization, no docs folder policy, no benchmarks, no dataset layer, and no dedicated temporal batching abstraction.
- Implementation debt: manual loops everywhere, no automatic differentiation, no dataset layer, no temporal batching abstraction.

## Naming And Drift Summary

- Canonical direction in vision: `JEPRA`.
- Current crate/package names: `jepra-core`.
- Current README message: JEPRA temporal JEPA framework, aligned with crate naming.
- Current VISION message: Rust-first JEPA systems framework with broader temporal predictive latent scope.
- Practical takeaway: code and docs now both reflect the same JEPA-first temporal direction.

## Promotion Baseline

- Maintain projected-path hardening sweep baseline:
  - re-run `run-projected-momentum-sweep.sh all` before default changes,
  - compare against captured fixed-seed rows,
  - emit CSV via `JEPRA_MOMENTUM_SWEEP_REPORT` when updating evidence,
  - keep defaults conservative (`--encoder-lr=0.0`, `--target-momentum=1.0`) and require explicit intent for departures,
  - only expand trainable projected defaults if trajectory checks are stable.
  - script enforcement requires a parserable `projected run summary` and both train/val segments reporting `improved=true`.
