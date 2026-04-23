# Implementation Guidelines and Next Steps

This note is the working implementation policy for the current JEPRA direction.
Read with `VISION.md`.

## Status
- Core JEPA stack is stable for temporal proof runs with shared CLI/config surfaces (`TemporalRunConfig` in `examples/support/temporal_vision.rs`).
- Unprojected path: `VisionJepa::step`, `VisionJepa::step_with_trainable_encoder`, and `VisionJepa::losses` are in core and used by the canonical temporal example.
- Projected path: `ProjectedVisionJepa` now has:
  - target-projector momentum (`target_projection_momentum`), builder API, and EMA update in `step_with_trainable_encoder`.
  - optional encoder update mode (`step_with_trainable_encoder(..., encoder_lr)`).
  - CLI/env control via `--target-momentum` / `--target-projection-momentum` and `JEPRA_TARGET_MOMENTUM`.
- Current defaults preserve hard target-projector behavior (`momentum = 1.0`) unless explicit tuning is passed.
- Regression posture:
  - one-step projected loss reduction,
  - target-projector EMA math edges,
  - momentum validation,
  - projected encoder-learning test coverage,
  - fixed + multi-seed convergence gates.

## Next Action
- Run a projected-path encoder-learning hardening sweep and lock it in:
  - fixed seed schedule comparison for `momentum ∈ {1.0, 0.5, 0.0}` and a small non-zero `encoder_lr`.
  - keep default path frozen for examples; make non-default configs explicit and documented.
  - add/maintain protocol checks only if validation behavior improves or stays stable.

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
