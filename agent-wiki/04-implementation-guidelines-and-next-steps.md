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
- Current defaults preserve hard target-projector behavior (`momentum = 1.0`) unless explicit tuning is passed.
- Regression posture:
  - one-step projected loss reduction,
  - target-projector EMA math edges,
  - momentum validation,
  - projected encoder-learning test coverage,
  - fixed + multi-seed convergence gates.

## Next Action
## Focused Review (Projected Path Hardening)

- `TemporalRunConfig` already supports `--target-momentum`, `--target-momentum-start`, `--target-momentum-end`, and warmup args in `examples/support/temporal_vision.rs`, with a linear schedule tested in `target_projection_momentum_warms_linearly_to_end`.
- `ProjectedVisionJepa` already has bounded `target_projection_momentum`, explicit EMA updates in `step_with_trainable_encoder`, and `target_projection_drift()` for protocol introspection.
- `tests/projected_temporal_support.rs` already validates:
  - midpoint warmup to final target behavior,
  - zero/one momentum edge behavior,
  - frozen/trainable protocol parity while warmup is active.
- Evidence gap: no committed protocol sweep currently verifies fixed-seed projected behavior across `{1.0, 0.5, 0.0}` momentum under the explicit entrypoint path and logs it as phase evidence.

## Next 3-Step Plan (Current Phase)

1. Confirm warmup contract before widening protocol permutations.
   - Run `cargo test --manifest-path crates/jepra-core/Cargo.toml --test temporal_vision_support target_projection_momentum_warms_linearly_to_end`.
   - Run `cargo test --manifest-path crates/jepra-core/Cargo.toml --test projected_temporal_support projected_target_projector_warmup_schedule_matches_frozen_and_trainable_protocols`.
   - Run `cargo test --manifest-path crates/jepra-core/Cargo.toml --test projected_temporal_support projected_target_projector_ema_edges_with_trainable_encoder_lr`.
   - Acceptance: all three tests pass, including: warmup midpoint and collapse assertions pass, EMA update math assertions stay exact (`target_projection_drift` consistency) for frozen/trainable branches.
   - Stop condition: if any assertion fails, remain in warmup contract hardening and do not start new protocol sweeps.

2. Execute projected protocol evidence sweep with fixed seeds at `target_projection_momentum ∈ {1.0, 0.5, 0.0}`.
   - For each target momentum value in the set, run the projected example with fixed seeds `21000, 21001, 21002` and explicit CLI:
     - `--train-base-seed <seed> --train-steps 80 --log 20`
     - `--encoder-lr 0.0`
     - warmup controls for non-trivial schedule: when testing `0.5`, use `--target-momentum-start 1.0 --target-momentum-end 0.5 --target-momentum-warmup-steps 24`.
   - Acceptance per seed: final train/validation totals must strictly improve, and meet deterministic reduction floors (same ratios as suite constants: train ≤ 0.2×initial, validation ≤ 0.2×initial where supported).
   - Stop condition: any momentum value with one seed failing deterministic reproducibility or reduction gate blocks promotion to next step.

3. Promote only after explicit hardening gate closure and freeze protocol baseline lock.
   - Run `cargo test --manifest-path crates/jepra-core/Cargo.toml --test projected_temporal_support` and confirm:
     - frozen vs trainable parity tests still hold at `encoder_lr=0.0`,
     - trajectory determinism tests pass for projected step and validation helpers,
     - momentum edge tests remain green.
   - Capture in wiki: final `(train_loss, val_loss, target_projection_drift)` summary for each seed+momentum row, and note warmup-step behavior.
   - Stop condition: all targeted tests pass and recorded evidence has no regression against prior phase; then and only then allow explicit non-default momentum and encoder-learning experiments as opt-in follow-up.

### Immediate Phase Stop Matrix

1. Warmup contract not stable: halt at step 1 and record failing assertion.
2. Evidence sweep mismatch on any momentum branch or seed: keep momentum and defaults frozen at `1.0`, encoder-lr zero, until re-runs are clean.
3. Regression in parity/determinism/edge tests: stop immediately; no follow-up changes until regression source is isolated and fixed.

## Anti-Goals
- no new abstractions for their own sake
- no generic framework expansion
- no CUDA/benchmark/perf rewrites before JEPA proof stability
- no model-family expansion or reconstruction objectives
- no implicit trainable-encoder defaults
- no hidden target-projector training bypass (all projected encoder updates stay explicit)

## Stale Guidance Audit
- README has one remaining narrow phrasing point about trainable encoder defaults; tracked roadmap text should match `ProjectedVisionJepa` trainable encoder capability.
- VISION is consistent on JEPA-first scope, projection head presence, and momentum-control direction.

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
- keep projected hardening evidence in `tests/projected_temporal_support.rs`
- keep one source of truth for training/validation math in core and shared support helpers

## Decision Rule

If a change does not improve JEPA learning signal, stability, or proof-readiness for compact temporal models, it is out of scope.
