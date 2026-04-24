# Implementation Guidelines and Next Steps

This note is the working implementation policy for the current JEPRA direction.
Read with `VISION.md`.

## Status
- Core JEPA stack is stable for temporal proof runs with shared CLI/config surfaces (`TemporalRunConfig` in `examples/support/temporal_vision.rs`).
- Unprojected path: `VisionJepa::step`, `VisionJepa::step_with_trainable_encoder`, and `VisionJepa::losses` are in core and used by the canonical temporal example.
- Projected path: `ProjectedVisionJepa` now has:
  - target-projector momentum (`target_projection_momentum`), builder API, and EMA update in `step_with_trainable_encoder`.
  - optional encoder update mode (`step_with_trainable_encoder(..., encoder_lr)`).
  - CLI/env control via `--target-momentum`, `--target-momentum-start`, `--target-momentum-end`, `--target-momentum-warmup-steps`, optional projection aliases (`--target-projection-momentum`, `--target-projection-momentum-end`), and `JEPRA_TARGET_MOMENTUM`.
- Projection regularization utilities now live in core (`regularizers.rs`) instead of example-only support code.
- Representation-health telemetry now lives in core and is printed by both temporal examples for prediction/target comparison.
- `ResidualBottleneckPredictor` is available as the compact-capacity identity-skip variant: identity plus scaled bottleneck delta, opt-in only.
- `TemporalRunConfig` exposes `--residual-delta-scale` / `JEPRA_RESIDUAL_DELTA_SCALE` for residual-delta ablations without changing defaults.
- `TemporalRunConfig` exposes `--projector-drift-weight` / `JEPRA_PROJECTOR_DRIFT_WEIGHT` for opt-in online-projector trust-region ablations without changing defaults.
- `TemporalRunConfig` exposes `--temporal-task` / `JEPRA_TEMPORAL_TASK`; default `random-speed` is unchanged, and `velocity-trail` is the harder opt-in diagnostic task.
- Current defaults preserve hard target-projector behavior (`momentum = 1.0`) unless explicit tuning is passed.
- Regression posture:
  - one-step projected loss reduction,
  - target-projector EMA math edges,
  - momentum validation,
  - projected encoder-learning test coverage,
  - fixed + multi-seed convergence gates.

## Next Action

Current high-value implementation path:

1. Run `run-predictor-mode-comparison.sh` before predictor-topology policy changes.
2. Treat `residual-bottleneck` as the current projected-path candidate, not a default, because 300-step frozen-base evidence is strong for projected but not unprojected.
3. Compact-stronger evidence is healthy but drift-confounded; residual delta scaling is now the explicit control knob for ablations, not a hidden topology change.
4. Treat the `velocity-trail` sweep as blocking residual promotion: baseline wins validation on all three compact-stronger projected seeds and residual has much higher target drift.
5. Treat velocity-bank ranking/MRR as implemented: both baseline and residual rank speed above random, but residual remains blocked by validation loss and drift.
6. Keep projector drift regularization as an opt-in control knob; use it to test exact drift confounds, not as a promoted default.
7. Keep projected momentum/default policy locked unless the established sweep gate remains clean.

## Predictor Evidence Snapshot

Latest paired evidence (`2026-04-24`, 300 steps, frozen-base encoder):

- Unprojected baseline remains best on final validation prediction loss: `0.040150` vs residual `0.074866` vs bottleneck `0.127445`.
- Projected residual-bottleneck is strongest and healthy: final validation prediction loss `0.045684`, prediction `min_std=0.972417`, target `min_std=0.941715`, target drift `0.031384`.
- Projected baseline is healthy but worse: final validation prediction loss `0.216322`.
- Projected bottleneck collapses prediction spread: final validation prediction loss `5.042953`, prediction `min_std=0.000000`, `status=accept_failed`.
- Decision: residual is worth building on for projected compact-capacity work; bottleneck alone should not drive projected topology.

Compact-stronger projected evidence (`2026-04-24`, 300 steps, seeds `11000..11002`, zero-init residual delta head):

- Baseline: mean final validation prediction loss `1.158821`, mean prediction `min_std=0.467862`, mean target drift `0.006282`, all rows `ok`.
- Bottleneck: mean final validation prediction loss `1.299992`, prediction `min_std=0.000000`, all rows `accept_failed`.
- Residual-bottleneck: mean final validation prediction loss `1.119540`, mean prediction `min_std=0.473802`, mean target drift `0.140347`, all rows `ok`.
- Decision: residual is still the projected candidate, but the compact-stronger advantage is modest and drift-confounded. Do not promote defaults; do not implement depthwise yet.

Residual delta-scale hardening evidence (`2026-04-24`, compact-stronger projected, seeds `11000..11002`):

- Scale `0.25` with target momentum `1.0`: residual mean validation prediction loss `1.137108`, mean prediction `min_std=0.517087`, mean target drift `0.133765`.
- Scale `0.25` with target momentum `0.5`: residual mean validation prediction loss `0.004085`, but mean prediction `min_std=0.028745` and mean target `min_std=0.023829`; treat as collapse, not a win.
- Decision: keep `residual_delta_scale=1.0` as default, keep target momentum `1.0`, and use `--residual-delta-scale` only for explicit evidence runs. The next modeling step should control residual/projector drift through a better mechanism than low target momentum.

Projector drift regularizer evidence (`2026-04-24`, compact-stronger projected, residual-bottleneck, seed `11000`, target momentum `1.0`):

- Weight `1.0`: final validation prediction loss `1.126885`, prediction `min_std=0.425927`, target drift `0.137658`.
- Weight `5.0`: final validation prediction loss `1.165685`, prediction `min_std=0.462189`, target drift `0.126054`.
- Weight `10.0`: final validation prediction loss `1.216697`, prediction `min_std=0.502000`, target drift `0.113993`.
- Decision: the L2 parameter-space drift regularizer works as an opt-in trust-region knob and keeps health intact, but current weights trade validation loss for only partial drift reduction. Do not promote defaults; use `velocity-trail` as the harder diagnostic before depthwise.

Velocity-trail task axis:

- `--temporal-task velocity-trail` adds a previous-position trail to each moving square while preserving deterministic mass decay and motion-mode contracts.
- `random-speed` remains the default and the historical evidence baseline.
- Current compact-stronger projected evidence (`2026-04-24`, seeds `11000..11002`, 300 steps): baseline mean validation prediction loss `0.119354`, residual-bottleneck mean `0.178396`; baseline wins 3/3.
- Residual-bottleneck remains health-ok but drift-confounded on this task: mean target drift `0.159557` vs baseline `0.002187`.
- Velocity-bank ranking/MRR (`2026-04-24`, same sweep): baseline mean MRR `0.817708`, top1 `0.635417`; residual mean MRR `0.835938`, top1 `0.671875`. Two-candidate random reference is MRR `0.75`, top1 `0.5`.
- Decision: current models learn some ordered speed structure; residual's ranking edge does not override worse validation loss and much higher drift. Do not promote residual, depthwise, or spatial primitives from this result.

## Focused Review (Projected Path Hardening)

- `TemporalRunConfig` already supports `--target-momentum`, `--target-momentum-start`, `--target-momentum-end`, and warmup args in `examples/support/temporal_vision.rs`, with a linear schedule tested in `target_projection_momentum_warms_linearly_to_end`.
- `ProjectedVisionJepa` already has bounded `target_projection_momentum`, explicit EMA updates in `step_with_trainable_encoder`, and `target_projection_drift()` for protocol introspection.
- `tests/projected_temporal_support.rs` already validates:
  - midpoint warmup to final target behavior,
  - zero/one momentum edge behavior,
  - frozen/trainable protocol parity while warmup is active.
- Protocol evidence is now established for fixed-seed projected behavior across `{1.0, 0.5, 0.0}` momentum under the explicit entrypoint path.
- Predictor comparison now uses schema `jepra_predictor_compare_v5`, emits `temporal_task`, velocity-bank fields for projected `velocity-trail`, `residual_delta_scale`, and `projector_drift_weight`, and rejects low-std representation collapse by default (`JEPRA_MIN_STD_THRESHOLD=0.05`).
- Projected hardening remains a regression gate, not the main build target for the next implementation step.

## Promotion/Regression Gate

1. Reconfirm warmup contract before projected-path policy changes.
   - Run `cargo test --manifest-path crates/jepra-core/Cargo.toml --test temporal_vision_support target_projection_momentum_warms_linearly_to_end`.
   - Run `cargo test --manifest-path crates/jepra-core/Cargo.toml --test projected_temporal_support projected_target_projector_warmup_schedule_matches_frozen_and_trainable_protocols`.
   - Run `cargo test --manifest-path crates/jepra-core/Cargo.toml --test projected_temporal_support projected_target_projector_ema_edges_with_trainable_encoder_lr`.
   - Acceptance: all three tests pass, including: warmup midpoint and collapse assertions pass, EMA update math assertions stay exact (`target_projection_drift` consistency) for frozen/trainable branches.
   - Stop condition: if any assertion fails, remain in warmup contract hardening and do not change projected defaults.

2. Keep protocol evidence clean against the fixed-seed baseline at `target_projection_momentum ∈ {1.0, 0.5, 0.0}`.
   - Use `run-projected-momentum-sweep.sh` profiles (`all|warmup|frozen|trainable|zero`) with fixed seeds `21000, 21001, 21002` and capture parser-readable results.
   - `run-projected-momentum-sweep.sh` now enforces summary shape, checks `improved=true` for both train and validation segments, and optionally emits per-row CSV via `JEPRA_MOMENTUM_SWEEP_REPORT`.
   - Warmup controls are part of the profile contract:
     - `--target-momentum-start 1.0`
     - `--target-momentum-end 0.5`
     - `--target-momentum-warmup-steps 24`
   - Track acceptance on:
     - no command failures,
     - parser-recognized output for all 12 rows (3 seeds × 4 profiles: warmup/frozen/trainable/zero),
     - no deterministic regressions against prior wiki evidence at fixed seeds.
   - Stop condition: any profile missing data, any run failure, or any deterministic mismatch blocks default-momentum changes.

3. Promote only after explicit hardening gate closure and protocol baseline lock.
   - Run `cargo test --manifest-path crates/jepra-core/Cargo.toml --test projected_temporal_support` and confirm:
     - frozen vs trainable parity tests still hold at `encoder_lr=0.0`,
     - trajectory determinism tests pass for projected step and validation helpers,
     - momentum edge tests remain green.
   - Update evidence only when final `(train_loss, val_loss, target_projection_drift)` rows are parser-clean and comparable to the captured baseline.
   - Stop condition: all targeted tests pass and recorded evidence has no regression against prior phase; then and only then allow explicit non-default momentum and encoder-learning experiments as opt-in follow-up.

### Immediate Phase Stop Matrix

1. Warmup contract not stable: halt at step 1 and record failing assertion.
2. Evidence sweep mismatch on any momentum branch or seed: keep momentum and defaults frozen at `1.0`, encoder-lr zero, until re-runs are clean.
3. Regression in parity/determinism/edge tests: stop immediately; no follow-up changes until regression source is isolated and fixed.

## Cross-Document Alignment Check

- `README.md` is aligned for this step on:
  - script modes (`all|warmup|frozen|trainable|zero`),
  - profile momentum set (`1.0`, `0.5`, `0.0`),
  - output format (`seed`, `momentum`, `profile`, `status`, `projected run summary`).
- `VISION.md` now explicitly names this phase and stop rules in section 7.

## Anti-Goals
- no new abstractions for their own sake
- no generic framework expansion
- no CUDA/benchmark/perf rewrites before JEPA proof stability
- no model-family expansion or reconstruction objectives
- no implicit trainable-encoder defaults
- no hidden target-projector training bypass (all projected encoder updates stay explicit)

## Stale Guidance Audit
- `VISION.md` now frames compact predictor-capacity work as the active implementation path.
- Projected hardening remains a regression gate, not the main build loop.

## Approved Implementation Sequence

1. Keep the core Rust surface small and understandable.
2. Use a harder signed/bounce temporal task or objective diagnostic as the next proof step before widening the model.
3. Keep regression coverage focused on task shape, determinism, and loss behavior.
4. Only after `random-speed` and `velocity-trail` evidence are credible, widen the model or data path.
5. Only after the JEPA proof is stable, consider performance work or lower-level acceleration.

This is the sequence because JEPRA is a framework with a thesis, not a framework-first abstraction exercise.
The proof comes from a compact model that learns useful temporal structure.

## Current State Snapshot

- `VisionJepa` and projected core APIs are now the canonical training entrypoint for both paths.
- `run_temporal_experiment_with_summary` remains the standard control flow for short reproducibility runs.
- Validation is still held out and deterministic; no fixed toy table replay.
- Default project posture is conservative:
  - unprojected `--encoder-lr = 0.0`
  - projected `target_projection_momentum = 1.0`

## Implementation Notes

- keep files small; keep core-path APIs explicit
- keep projected regression coverage in `tests/projected_temporal_support.rs`
- keep JEPA projection regularizer math in core `regularizers.rs`
- keep one source of truth for training/validation math in core and shared support helpers

## Decision Rule

If a change does not improve JEPA learning signal, stability, or proof-readiness for compact temporal models, it is out of scope.
