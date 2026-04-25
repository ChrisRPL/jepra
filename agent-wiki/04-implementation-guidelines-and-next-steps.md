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
- `StateRadiusPredictor` is available as the signed radius/speed-geometry probe: baseline direction head plus a learned positive per-sample gain on projected-state displacement, opt-in only.
- `TemporalRunConfig` exposes `--residual-delta-scale` / `JEPRA_RESIDUAL_DELTA_SCALE` for residual-delta ablations without changing defaults.
- `TemporalRunConfig` exposes `--projector-drift-weight` / `JEPRA_PROJECTOR_DRIFT_WEIGHT` for opt-in online-projector trust-region ablations without changing defaults.
- `TemporalRunConfig` exposes `--temporal-task` / `JEPRA_TEMPORAL_TASK`; default `random-speed` is unchanged, and `velocity-trail` plus `signed-velocity-trail` are harder opt-in diagnostic tasks.
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
6. Treat the `signed-velocity-trail` sweep as blocking residual promotion: baseline wins validation on all three compact-stronger projected seeds, residual has much higher target drift, and signed velocity-bank ranking is not above random.
7. Treat target-bank v7 oracle diagnostics as implemented: candidate-target construction is clean, so signed failure is a predictor/objective problem under drift rather than target-bank ambiguity.
8. Treat prediction-bank v8 margin diagnostics as implemented: predictions sit closer to wrong signed futures than true targets on most samples.
9. Treat signed objective v9 decomposition as implemented: the dominant signed loss is negative-direction error, not missing candidate construction or raw speed ranking.
10. Treat signed-margin objective v10 as implemented and rejected by the narrow grid: do not expand weights or promote it.
11. Treat signed state separability v11 as implemented: current latent and projected state are near random on signed direction, so another loss-only pass is not the next best build step.
12. Treat `--compact-encoder-mode signed-direction` as the active opt-in representation probe: scaled responses preserve state separability and pass prediction health, but prediction-bank margin remains negative, so it is not promotable yet.
13. Treat signed-bank softmax v12 as implemented and blocked: it gives a direct candidate-bank cross-entropy probe, but narrow signed-direction evidence leaves positive-margin rate unchanged, so do not promote or tune it broadly.
14. Treat `signed-direction-magnitude` and unit-geometry v13 as implemented: magnitude conditioning improves validation/raw margin, and bank-centered unit geometry exposes useful signed direction, but raw PPR remains pinned.
15. Treat signed radial calibration v14 as implemented but not sufficient: weight `0.1` raises centered norm ratio while preserving health, but raw PPR remains pinned.
16. Keep projector drift regularization as an opt-in control knob; use it to test exact drift confounds, not as a promoted default.
17. Keep projected momentum/default policy locked unless the established sweep gate remains clean.

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
- Decision: the L2 parameter-space drift regularizer works as an opt-in trust-region knob and keeps health intact, but current weights trade validation loss for only partial drift reduction. Do not promote defaults; use trail-task diagnostics before depthwise.

Velocity-trail task axis:

- `--temporal-task velocity-trail` adds a previous-position trail to each moving square while preserving deterministic mass decay and motion-mode contracts.
- `--temporal-task signed-velocity-trail` adds the same trail structure with balanced signed velocities `dx ∈ {-2,-1,+1,+2}` per batch.
- `random-speed` remains the default and the historical evidence baseline.
- Current compact-stronger projected evidence (`2026-04-24`, seeds `11000..11002`, 300 steps): baseline mean validation prediction loss `0.119354`, residual-bottleneck mean `0.178396`; baseline wins 3/3.
- Residual-bottleneck remains health-ok but drift-confounded on this task: mean target drift `0.159557` vs baseline `0.002187`.
- Velocity-bank ranking/MRR (`2026-04-24`, same sweep): baseline mean MRR `0.817708`, top1 `0.635417`; residual mean MRR `0.835938`, top1 `0.671875`. Two-candidate random reference is MRR `0.75`, top1 `0.5`.
- Decision: current models learn some ordered speed structure; residual's ranking edge does not override worse validation loss and much higher drift. Do not promote residual, depthwise, or spatial primitives from this result.
- Signed velocity-trail compact-stronger projected evidence (`2026-04-24`, seeds `11000..11002`, 300 steps): baseline mean validation prediction loss `1.305638`, residual-bottleneck mean `1.385723`; baseline wins 3/3.
- Residual-bottleneck remains health-ok but drift-confounded on this task: mean prediction `min_std=0.312411` vs baseline `0.177438`, mean target drift `0.136066` vs baseline `0.008615`.
- Signed velocity-bank ranking/MRR (`2026-04-24`, same sweep): baseline mean MRR `0.519965`, top1 `0.239583`; residual mean MRR `0.489583`, top1 `0.203125`. Four-candidate random reference is MRR `0.520833`, top1 `0.25`.
- Signed v6 breakdown (`2026-04-24`, same sweep): baseline `neg_mrr=0.386285`, `pos_mrr=0.653646`, `sign_top1=0.479167`, `speed_top1=0.494792`; residual `neg_mrr=0.309896`, `pos_mrr=0.669271`, `sign_top1=0.375000`, `speed_top1=0.635417`.
- Signed target-bank v7 oracle diagnostics (`2026-04-24`, same sweep): `oracle_mrr=1.000000`, `oracle_top1=1.000000`, `true_distance=0.000000`, mean margin `1.463823`, minimum margin `0.001335`, target sign margin `7.331476`, target speed margin `-12.402091`.
- Signed prediction-bank v8 diagnostics (`2026-04-24`, same sweep): baseline prediction margin `-4.159113`, positive margin rate `0.239583`, sign margin `-0.981234`; residual prediction margin `-4.688569`, positive margin rate `0.203125`, sign margin `-1.366664`.
- Signed objective v9 diagnostics (`2026-04-24`, same sweep): baseline `all=1.305638`, `neg=2.118653`, `pos=0.492622`, `slow=1.349482`, `fast=1.261794`, `sign_gap=-1.626031`, `speed_gap=-0.087688`; residual `all=1.385723`, `neg=2.204694`, `pos=0.566752`, `slow=1.327650`, `fast=1.443796`, `sign_gap=-1.637942`, `speed_gap=0.116146`.
- Signed-margin objective v10 status: default-off hinge objective is wired for projected `signed-velocity-trail` only. Disabled rows keep signed-margin report fields `na`; enabled rows emit signed-margin loss and active hinge rates.
- Signed-margin objective v10 grid (`2026-04-24`, weights `0,0.003,0.01,0.03,0.1`): all candidates rejected. Best baseline weight `0.1` gives `val_ratio=0.989425`, `sign_top1=0.552083`, `mrr=0.531250`, `margin_gain=0.084920`, and `ppr_gain=0.010417`.
- Signed state separability v11 (`2026-04-24`, same sweep): baseline and residual both have latent MRR `0.518229`, latent sign-top1 `0.468750`, projection MRR `0.518229`, and projection sign-top1 `0.468750`; four-candidate random references are MRR `0.520833` and top1 `0.25`.
- Signed-direction compact encoder probe (`2026-04-24`, projected signed 300-step baseline, seeds `11000..11002`): response scaling keeps all rows health-ok with mean `pred_min_std=0.158783`, `target_min_std=0.750253`, state MRR `0.619792`, sign-top1 `0.609375`, target drift `0.008791`, but prediction-bank positive-margin rate is only `0.281250` and margins remain negative.
- Signed-bank softmax v12 (`2026-04-25`, signed-direction projected baseline): weight `0.5` keeps all rows health-ok but leaves `ppr=0.281250`, `margin=-1.200875`, and softmax top1 `0.281250`; a high-weight seed-11000 diagnostic lowers softmax loss but still leaves `ppr=0.281250` and worsens signed-bank MRR.
- Signed-direction-magnitude + unit geometry v13 (`2026-04-25`, projected signed baseline): all rows health-ok; validation improves to `0.387471`, raw margin improves to `-0.818459`, raw PPR remains `0.281250`, unit MRR is `0.631944`, unit top1/PPR is `0.453125`, and unit speed margin is near zero (`0.001397`).
- Signed radial calibration v14 (`2026-04-25`, `signed-direction-magnitude`, projected signed baseline, seeds `11000..11002`, weight `0.1`): all rows health-ok; validation is `0.396301`, centered norm ratio rises to `0.485350`, but raw PPR remains `0.281250`, raw margin is `-0.860106`, and unit top1 is `0.437500`.
- Signed angular-radial v15 (`2026-04-25`, `signed-direction-magnitude`, projected signed baseline, seeds `11000..11002`, weight `0.1`): all rows health-ok, but raw PPR remains `0.281250`.
- Signed geometry counterfactual v16 (`2026-04-25`, same baseline): raw PPR remains `0.281250`, unit PPR is `0.453125`, oracle-radius PPR is `0.447917`, oracle-angle PPR is `0.218750`, and support-global-rescale PPR is `0.218750`.
- State-radius predictor probe (`2026-04-25`, same baseline): all rows health-ok and validation improves slightly (`0.377447` vs `0.387471`), but raw PPR drops to `0.140625`, unit PPR drops to `0.114583`, and oracle-radius PPR drops to `0.000000`.
- Candidate-centroid integration probe (`2026-04-25`, same baseline, report-only, temperature `0.05`): raw PPR is `0.281250`, unit PPR is `0.453125`, nearest/softmax candidate-radius PPR is `0.348958`, and nearest/softmax MRR is `0.573350`. This preserves direction and improves over raw, but misses the `~0.364583` proof gate.
- Decision: signed evidence blocks residual promotion, depthwise convolution, and spatial predictor work. Target-bank construction is valid, prediction-bank margins show predictions closer to wrong signed futures than true targets on most samples, v9 shows negative-direction error dominates, v10 shows uniform signed-margin shaping is too weak, v11 shows current state representation is not direction-separable, signed-direction proves local orientation features can be health-ok, v12 shows candidate cross-entropy alone does not close ranking, v13 shows angular signal is usable but raw radial calibration is wrong, v14/v15 show scalar radius/angular-radial matching alone is not enough, v16 shows true-radius snapping recovers ranking while global rescale does not, state-radius shows simple projected-state displacement gain damages the angular signal, and candidate-centroid report-only radius selection helps but is not sufficient. Next valid step: train a candidate-centered radius residual/logit head with the centered-radius scalar primitive while preserving bank-centered direction.

### Candidate-Centroid-Aware Geometry Gate

- Schema/parser: keep `jepra_predictor_compare_v16` for the first implementation. Existing fields already cover raw bank ranking, unit bank-centered angular ranking, prediction/target centered norms, oracle-radius ceiling, support global rescale, validation, drift, and health. Only bump the schema if a future patch emits a new diagnostic line that cannot be derived from these fields.
- Required run shape: projected `signed-velocity-trail`, `signed-direction-magnitude`, same-run `baseline` versus the candidate predictor mode, seeds `11000 11001 11002`, `300` train steps, CSV report enabled.
- Health gate: all candidate rows `status=ok`; mean `val_pred_end <= 1.05 * baseline_val_pred_end`; mean `target_drift_end <= max(0.02, 2.5 * baseline_target_drift_end)`.
- Raw-ranking gate: mean `prediction_bank_positive_margin_rate_end >= baseline_raw_ppr + 0.5 * (baseline_oracle_radius_ppr - baseline_raw_ppr)`. Current evidence makes this approximately `>= 0.364583`.
- Angular-preservation gate: mean `prediction_unit_positive_margin_rate_end >= baseline_unit_ppr - 0.03` and mean `prediction_unit_mrr_end >= baseline_unit_mrr - 0.03`; current evidence makes this roughly `unit_ppr >= 0.423125`, `unit_mrr >= 0.601944`.
- Radius gate: mean centered norm ratio `prediction_unit_prediction_center_norm_end / prediction_unit_true_target_center_norm_end` must reduce absolute error from `1.0` by at least `25%` versus same-run baseline and stay within `[0.50, 1.50]`.
- Seed-level sanity: `prediction_bank_margin_end`, `prediction_bank_sign_margin_end`, and `prediction_bank_speed_margin_end` should improve versus baseline on at least two of three seeds. Treat this as secondary to raw PPR and radius because oracle-radius improved PPR while mean margin remained negative.
- Stop rules: if raw PPR improves but unit PPR/MRR collapses, reject the head as direction-destructive; if unit metrics stay strong but norm ratio does not improve, keep the centroid frame and change radius parameterization/loss; if health or drift fails, add anchoring/regularization before adding capacity.

## Focused Review (Projected Path Hardening)

- `TemporalRunConfig` already supports `--target-momentum`, `--target-momentum-start`, `--target-momentum-end`, and warmup args in `examples/support/temporal_vision.rs`, with a linear schedule tested in `target_projection_momentum_warms_linearly_to_end`.
- `ProjectedVisionJepa` already has bounded `target_projection_momentum`, explicit EMA updates in `step_with_trainable_encoder`, and `target_projection_drift()` for protocol introspection.
- `tests/projected_temporal_support.rs` already validates:
  - midpoint warmup to final target behavior,
  - zero/one momentum edge behavior,
  - frozen/trainable protocol parity while warmup is active.
- Protocol evidence is now established for fixed-seed projected behavior across `{1.0, 0.5, 0.0}` momentum under the explicit entrypoint path.
- Predictor comparison now uses schema `jepra_predictor_compare_v16`, emits `temporal_task`, velocity-bank fields for projected trail tasks, signed-task breakdown fields, target-bank oracle fields, prediction-bank margin fields, prediction-bank unit geometry and counterfactual fields, signed objective error decomposition fields, signed-margin objective fields, signed-bank softmax objective fields, signed radial/angular-radial fields, signed state separability fields, `residual_delta_scale`, and `projector_drift_weight`, and rejects low-std representation collapse by default (`JEPRA_MIN_STD_THRESHOLD=0.05`).
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
2. Use signed state separability, raw prediction-bank margin, unit prediction geometry, radial calibration, and representation health as the current proof gate; improve angular-radial signed candidate geometry before another objective grid, bounce, or widening the model.
3. Keep regression coverage focused on task shape, determinism, and loss behavior.
4. Only after `random-speed`, `velocity-trail`, and `signed-velocity-trail` evidence are credible, widen the model or data path.
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
