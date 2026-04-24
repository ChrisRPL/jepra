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
- Current defaults preserve hard target-projector behavior (`momentum = 1.0`) unless explicit tuning is passed.
- Regression posture:
  - one-step projected loss reduction,
  - target-projector EMA math edges,
  - momentum validation,
  - projected encoder-learning test coverage,
  - fixed + multi-seed convergence gates.

## Next Action

Current high-value implementation path:

1. Finish `BottleneckPredictor` as an opt-in Rust-native predictor variant.
2. Compare `baseline` vs `bottleneck` with the current temporal examples using identical seeds.
3. Keep projected momentum/default policy locked unless the established sweep gate remains clean.

## Focused Review (Projected Path Hardening)

- `TemporalRunConfig` already supports `--target-momentum`, `--target-momentum-start`, `--target-momentum-end`, and warmup args in `examples/support/temporal_vision.rs`, with a linear schedule tested in `target_projection_momentum_warms_linearly_to_end`.
- `ProjectedVisionJepa` already has bounded `target_projection_momentum`, explicit EMA updates in `step_with_trainable_encoder`, and `target_projection_drift()` for protocol introspection.
- `tests/projected_temporal_support.rs` already validates:
  - midpoint warmup to final target behavior,
  - zero/one momentum edge behavior,
  - frozen/trainable protocol parity while warmup is active.
- Protocol evidence is now established for fixed-seed projected behavior across `{1.0, 0.5, 0.0}` momentum under the explicit entrypoint path.
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
- `VISION.md` alignment is now up to date for the active projected-hardening phase; no immediate-step framing patch is pending.

## Approved Implementation Sequence

1. Keep the core Rust surface small and understandable.
2. Build one harder temporal Vision JEPA example that forces real prediction, not memorization.
3. Add regression coverage for that example’s shape, determinism, and loss behavior.
4. Only after the temporal example is credible, widen the model or data path.
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
