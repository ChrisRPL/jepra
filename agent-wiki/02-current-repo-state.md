# Current Repo State

Snapshot from local files only. Internal working note for the repo state today.

## Status
- JEPA proof currently uses temporal random-video tasks as the forcing function.
- Projected path now has EMA target projector + optional encoder training (`step_with_trainable_encoder`).
- Regression coverage is in `crates/jepra-core/tests/projected_temporal_support.rs`; fixed-seed hardening evidence is captured in README and reproducible via `run-projected-momentum-sweep.sh`.
- Predictor comparison evidence is reproducible via `run-predictor-mode-comparison.sh`; current 300-step evidence favors residual-bottleneck only on the projected path, with compact-stronger runs showing a drift confound.
- Residual-bottleneck now has an explicit residual-delta scale control for ablations. Scale `0.25` did not solve compact-stronger projected drift, and low target momentum produced low-loss representation collapse.
- Projected path now has an opt-in L2 online-projector drift regularizer. It reduces target drift on a seed-level probe, but current weights are a tradeoff rather than a default-ready fix.
- Temporal examples now expose a task axis: default `random-speed` plus opt-in `velocity-trail` and `signed-velocity-trail`, which add previous-position trail cues without changing model defaults.
- Latest `velocity-trail` compact-stronger projected sweep: baseline beats residual-bottleneck on validation across seeds `11000..11002`; residual remains health-ok but target drift is ~73x higher.
- Velocity-bank ranking/MRR is now implemented for projected `velocity-trail`; both baseline and residual rank speed above random, but residual is still blocked by validation loss and drift.
- Latest `signed-velocity-trail` compact-stronger projected sweep: baseline beats residual-bottleneck on validation across seeds `11000..11002`; residual remains health-ok but target drift is ~16x higher.
- Signed velocity-bank v6 breakdown shows direction failure: both predictors are weak on negative motion, and residual has speed signal (`speed_top1=0.635417`) but worse sign accuracy (`sign_top1=0.375000`) than baseline.
- Signed target-bank v7 oracle diagnostics are clean (`oracle_mrr=1.0`, true distance `0.0`, mean margin `1.463823`), so signed failure is not candidate-target construction; it points back to predictor/objective learning under drift.
- Signed prediction-bank v8 margin diagnostics show predictions are closer to wrong signed futures than true targets on most samples: baseline mean margin `-4.159113`, residual mean margin `-4.688569`.
- Signed objective v9 diagnostics decompose validation loss by signed dx bucket: baseline mean `neg=2.118653`, `pos=0.492622`; residual mean `neg=2.204694`, `pos=0.566752`. The dominant error is negative-direction prediction, and residual does not fix it.
- Default-off signed-margin objective v10 is implemented as an opt-in projected signed-task probe. Weight `0.0` avoids candidate-bank construction and emits `na` report fields; weight `>0` adds hinge gradients against wrong signed futures and emits margin-objective losses/active rates.
- V10 signed-margin weight grid rejects all candidates. Best baseline weight `0.1` has small validation/sign gains but misses the margin, positive-rate, and MRR gates by a wide margin.
- Signed state separability v11 is implemented as a report-only projected signed-task probe. It builds support/query nearest-centroid classifiers over current latent state and projected state.
- V11 evidence shows state separability is near random for both baseline and residual: latent/projection MRR `0.518229` and sign-top1 `0.468750`, with a four-candidate random MRR reference of `0.520833`. This shifts the next build target from more loss shaping to representation/conditioning.
- Opt-in compact encoder mode `signed-direction` now adds scaled local horizontal trail-orientation filters while preserving the existing 3D latent interface. Three-seed projected signed baseline evidence is health-ok (`pred_min_std=0.158783`, `target_min_std=0.750253`) and keeps state MRR `0.619792`, but prediction-bank margin remains negative, so it is not promoted.
- Default-off signed-bank softmax objective is implemented for projected `signed-velocity-trail` probes. It keeps the candidate-bank ranking loss explicit and reportable, but narrow evidence does not close the signed-direction margin gate (`ppr=0.281250` at weight `0.5`; high-weight single-seed still `ppr=0.281250`), so it is not promoted.
- Opt-in `signed-direction-magnitude` and prediction-bank unit-geometry diagnostics are implemented. Combined 3-seed evidence improves validation (`0.387471`) and raw margin (`-0.818459`) while preserving health, and unit geometry reaches top1/PPR `0.453125`, but raw PPR is still pinned at `0.281250`.
- Default-off signed radial calibration is implemented for projected `signed-velocity-trail` probes. The first bounded `signed-direction-magnitude` 3-seed probe at weight `0.1` is health-ok and raises centered norm ratio to `0.485350`, but raw PPR remains pinned at `0.281250`, so this is evidence/tooling rather than a promotable fix.
- Report-only signed geometry counterfactual v16 is implemented. On `signed-direction-magnitude`, true-radius snapping raises PPR to `0.447917`, while true-angle and support global rescale stay at `0.218750`; the next bottleneck is trainable state-conditioned radius/speed geometry, not missing angle or simple global gain.
- Opt-in `StateRadiusPredictor` is implemented as the first trainable radius/speed geometry path. It learns a positive per-sample gain over the predicted projected-state displacement, but the first bounded signed-direction-magnitude probe rejects it as a fix: validation improves slightly (`0.377447` vs `0.387471`) while raw PPR drops to `0.140625` and unit PPR collapses to `0.114583`.

## Next Action
- Current phase focus has shifted from projected hardening to compact model capacity:
  - keep default path conservative (`--encoder-lr=0.0`, `target_projection_momentum=1.0`, `--predictor-mode=baseline`),
  - add and compare opt-in Rust-native predictor variants without changing defaults,
  - use lightweight representation-health stats to compare prediction and target behavior,
  - control residual/projector drift on stronger compact projected runs before any new primitive work,
  - keep residual, depthwise, and spatial primitive work blocked by signed-task evidence,
  - do not expand the signed-margin grid; v11 shows the current signed state representation is not direction-separable enough for another loss-only pass,
  - next high-value build step is candidate-centroid-aware signed geometry, because simple state-displacement gain damages the bank-centered angular signal,
  - evaluate that build with existing v16 parser fields: same-run baseline versus candidate, raw PPR closing at least half of the oracle-radius gap, unit PPR/MRR within `0.03` of baseline, centered norm-ratio error reduced by at least `25%`, and health/drift intact,
  - keep projector drift regularization as a narrow opt-in drift-control probe,
  - reject loss-only wins when prediction/target health collapses,
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
- `predictor.rs`: baseline two-layer predictor plus bottleneck and residual-bottleneck predictors, with ReLU, scaled residual delta, backward pass, and SGD step.
- `conv.rs`: `Conv2d` with forward, backward, and SGD step APIs.
- `encoder.rs`: `ConvEncoder` and `EmbeddingEncoder` on top of conv + global average pooling.
- `regularizers.rs`: JEPA projection regularizer utilities (`gaussian_moment_regularizer`, gradient, projection stats, representation stats, projection-grad combination).
- `vision_jepa.rs`: `VisionJepa` and `ProjectedVisionJepa` wrappers that compose encoders + predictor/projector and expose core encode/predict/loss/update entry points, including opt-in projected drift regularization.
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
- Temporal data uses mixed-speed synthetic motion, with deterministic mixed-mode validation probes and held-out schedules; `velocity-trail` and `signed-velocity-trail` are available as harder opt-in tasks.
- CLI/env config for runs lives in `examples/support/temporal_vision.rs` and now includes `--temporal-task random-speed|velocity-trail|signed-velocity-trail`, `--predictor-mode`, `--residual-delta-scale`, `--projector-drift-weight`, `--target-momentum`/`--target-momentum-start`/`--target-momentum-end` plus `--target-momentum-warmup-steps`, `JEPRA_TEMPORAL_TASK`, `JEPRA_RESIDUAL_DELTA_SCALE`, `JEPRA_PROJECTOR_DRIFT_WEIGHT`, and `JEPRA_TARGET_MOMENTUM`.
- `Conv2d` has backprop support and optimizer updates; encoder-level backward/grad plumbing is now in place.
- There is no real evaluation harness, no benchmark runner, and no model checkpointing or serialization path.

## Strengths

- Small surface area, easy to inspect.
- Core tensor API is consistent and aggressively shape-checked.
- Deterministic init makes examples and tests reproducible.
- The repo already expresses the intended JEPA shape instead of hiding it behind generic ML abstractions.

## Limitations And Technical Debt

- Naming drift: no active legacy-vs-`JEPRA` naming split in tracked metadata (`crates/jepra-core/Cargo.toml`).
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
