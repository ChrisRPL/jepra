#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_PATH="${JEPRA_MANIFEST_PATH:-$ROOT_DIR/crates/jepra-core/Cargo.toml}"
SCHEMA="jepra_predictor_compare_v7"
TRAIN_STEPS="${JEPRA_TRAIN_STEPS:-300}"
LOG_EVERY="${JEPRA_LOG_EVERY:-25}"
TEMPORAL_TASK="${JEPRA_TEMPORAL_TASK:-random-speed}"
PREDICTOR_MODES_CSV="${JEPRA_PREDICTOR_MODES:-baseline bottleneck residual-bottleneck}"
UNPROJECTED_SEEDS_CSV="${JEPRA_UNPROJECTED_PREDICTOR_SEEDS:-1000}"
PROJECTED_SEEDS_CSV="${JEPRA_PROJECTED_PREDICTOR_SEEDS:-11000}"
UNPROJECTED_ENCODER_LR="${JEPRA_UNPROJECTED_ENCODER_LR:-0.0}"
PROJECTED_ENCODER_LR="${JEPRA_PROJECTED_ENCODER_LR:-0.0}"
PROJECTED_TARGET_MOMENTUM="${JEPRA_PROJECTED_TARGET_MOMENTUM:-1.0}"
PROJECTED_TARGET_MOMENTUM_START="${JEPRA_PROJECTED_TARGET_MOMENTUM_START:-$PROJECTED_TARGET_MOMENTUM}"
PROJECTED_TARGET_MOMENTUM_END="${JEPRA_PROJECTED_TARGET_MOMENTUM_END:-$PROJECTED_TARGET_MOMENTUM}"
PROJECTED_TARGET_MOMENTUM_WARMUP_STEPS="${JEPRA_PROJECTED_TARGET_MOMENTUM_WARMUP_STEPS:-0}"
COMPACT_ENCODER_MODE="${JEPRA_COMPACT_ENCODER_MODE:-}"
RESIDUAL_DELTA_SCALE="${JEPRA_RESIDUAL_DELTA_SCALE:-1.0}"
PROJECTOR_DRIFT_WEIGHT="${JEPRA_PROJECTOR_DRIFT_WEIGHT:-${JEPRA_PROJECTOR_ANCHOR_WEIGHT:-0.0}}"
MIN_STD_THRESHOLD="${JEPRA_MIN_STD_THRESHOLD:-0.05}"
REPORT_PATH="${JEPRA_PREDICTOR_COMPARISON_REPORT:-}"
SCENARIO="${1:-all}"

read -r -a PREDICTOR_MODES <<< "$PREDICTOR_MODES_CSV"
read -r -a UNPROJECTED_SEEDS <<< "$UNPROJECTED_SEEDS_CSV"
read -r -a PROJECTED_SEEDS <<< "$PROJECTED_SEEDS_CSV"
failures=0

usage() {
  cat <<'EOF'
Usage:
  ./run-predictor-mode-comparison.sh [all|unprojected|projected]

Environment:
  JEPRA_PREDICTOR_MODES                         Space-separated modes (default: "baseline bottleneck residual-bottleneck")
  JEPRA_TRAIN_STEPS                             Train steps per run (default: 300)
  JEPRA_LOG_EVERY                               Log interval passed to examples (default: 25)
  JEPRA_TEMPORAL_TASK                           Temporal task: random-speed|velocity-trail|signed-velocity-trail (default: random-speed)
  JEPRA_UNPROJECTED_PREDICTOR_SEEDS             Space-separated unprojected seeds (default: "1000")
  JEPRA_PROJECTED_PREDICTOR_SEEDS               Space-separated projected seeds (default: "11000")
  JEPRA_UNPROJECTED_ENCODER_LR                  Unprojected encoder LR (default: 0.0)
  JEPRA_PROJECTED_ENCODER_LR                    Projected encoder LR (default: 0.0)
  JEPRA_PROJECTED_TARGET_MOMENTUM               Projected target momentum shorthand (default: 1.0)
  JEPRA_PROJECTED_TARGET_MOMENTUM_START         Projected warmup start (default: target momentum)
  JEPRA_PROJECTED_TARGET_MOMENTUM_END           Projected warmup end (default: target momentum)
  JEPRA_PROJECTED_TARGET_MOMENTUM_WARMUP_STEPS  Projected warmup steps (default: 0)
  JEPRA_COMPACT_ENCODER_MODE                    Optional compact encoder mode: base|stronger
  JEPRA_RESIDUAL_DELTA_SCALE                    Residual-bottleneck delta scale (default: 1.0)
  JEPRA_PROJECTOR_DRIFT_WEIGHT                  Projected online-projector drift regularizer weight (default: 0.0)
  JEPRA_MIN_STD_THRESHOLD                       Minimum final prediction/target min-std for ok rows (default: 0.05)
  JEPRA_PREDICTOR_COMPARISON_REPORT             Optional CSV path for parsed rows
EOF
}

if [[ "$SCENARIO" == "-h" || "$SCENARIO" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$SCENARIO" != "all" && "$SCENARIO" != "unprojected" && "$SCENARIO" != "projected" ]]; then
  echo "Unknown scenario: $SCENARIO" >&2
  usage >&2
  exit 2
fi

if [[ "${#PREDICTOR_MODES[@]}" -eq 0 ]]; then
  echo "No predictor modes configured in JEPRA_PREDICTOR_MODES" >&2
  exit 2
fi

if [[ -n "$REPORT_PATH" ]]; then
  mkdir -p "$(dirname "$REPORT_PATH")"
  printf 'schema,temporal_task,path,predictor,residual_delta_scale,projector_drift_weight,seed,steps,encoder_mode,encoder_lr,target_momentum_start,target_momentum_end,target_momentum_warmup_steps,train_pred_start,train_pred_end,val_pred_start,val_pred_end,train_obj_start,train_obj_end,val_obj_start,val_obj_end,pred_min_std_final,target_min_std_final,proj_var_mean_final,target_drift_end,velocity_bank_mrr_start,velocity_bank_mrr_end,velocity_bank_top1_start,velocity_bank_top1_end,velocity_bank_mean_rank_start,velocity_bank_mean_rank_end,velocity_bank_samples,velocity_bank_candidates,signed_bank_neg_mrr_end,signed_bank_pos_mrr_end,signed_bank_slow_mrr_end,signed_bank_fast_mrr_end,signed_bank_sign_top1_end,signed_bank_speed_top1_end,signed_bank_samples,signed_bank_true_neg_best_neg,signed_bank_true_neg_best_pos,signed_bank_true_pos_best_neg,signed_bank_true_pos_best_pos,signed_bank_true_slow_best_slow,signed_bank_true_slow_best_fast,signed_bank_true_fast_best_slow,signed_bank_true_fast_best_fast,target_bank_oracle_mrr_end,target_bank_oracle_top1_end,target_bank_true_distance_end,target_bank_true_distance_max_end,target_bank_nearest_wrong_end,target_bank_nearest_wrong_min_end,target_bank_margin_end,target_bank_margin_min_end,target_bank_neg_nearest_wrong_end,target_bank_pos_nearest_wrong_end,target_bank_slow_nearest_wrong_end,target_bank_fast_nearest_wrong_end,target_bank_sign_margin_end,target_bank_speed_margin_end,target_bank_samples,status\n' > "$REPORT_PATH"
fi

encoder_mode_label() {
  case "$COMPACT_ENCODER_MODE" in
    "") printf '%s' "frozen-base" ;;
    "base") printf '%s' "compact-base" ;;
    "stronger") printf '%s' "compact-stronger" ;;
    *) printf '%s' "unknown-${COMPACT_ENCODER_MODE}" ;;
  esac
}

delta() {
  awk -v start="$1" -v end="$2" 'BEGIN { printf "%.6f", end - start }'
}

lt_bool() {
  awk -v lhs="$1" -v rhs="$2" 'BEGIN { if (lhs < rhs) print "true"; else print "false" }'
}

gt_threshold_bool() {
  awk -v value="$1" -v threshold="$2" 'BEGIN { if (value > threshold) print "true"; else print "false" }'
}

is_velocity_bank_task() {
  [[ "$TEMPORAL_TASK" == "velocity-trail" || "$TEMPORAL_TASK" == "signed-velocity-trail" ]]
}

is_signed_velocity_bank_task() {
  [[ "$TEMPORAL_TASK" == "signed-velocity-trail" ]]
}

parse_unprojected_probe_line() {
  local line="$1"
  local parsed
  parsed="$(sed -E 's/^(initial|final)[[:space:]]+\|[[:space:]]+probe train[[:space:]]+([^[:space:]]+)[[:space:]]+\|[[:space:]]+val[[:space:]]+([^[:space:]]+)$/\2 \3/' <<< "$line")"
  if [[ "$parsed" == "$line" ]]; then
    return 1
  fi
  printf '%s' "$parsed"
}

parse_projected_loss_line() {
  local line="$1"
  local parsed
  parsed="$(sed -E 's/^(initial|final)[[:space:]]+\|[[:space:]]+(train|val) pred[[:space:]]+([^[:space:]]+)[[:space:]]+\|[[:space:]]+reg[[:space:]]+([^[:space:]]+)[[:space:]]+\|[[:space:]]+total[[:space:]]+([^[:space:]]+)$/\3 \5/' <<< "$line")"
  if [[ "$parsed" == "$line" ]]; then
    return 1
  fi
  printf '%s' "$parsed"
}

parse_health_line() {
  local line="$1"
  local parsed
  parsed="$(sed -E 's/^final[[:space:]]+(prediction|target)[[:space:]]+health[[:space:]]+\|[[:space:]]+mean_abs[[:space:]]+([^[:space:]]+)[[:space:]]+\|[[:space:]]+mean_std[[:space:]]+([^[:space:]]+)[[:space:]]+\|[[:space:]]+min_std[[:space:]]+([^[:space:]]+)[[:space:]]+\|[[:space:]]+offdiag_cov_abs[[:space:]]+([^[:space:]]+)[[:space:]]+\|[[:space:]]+offdiag_cov_max[[:space:]]+([^[:space:]]+)$/\3 \4/' <<< "$line")"
  if [[ "$parsed" == "$line" ]]; then
    return 1
  fi
  printf '%s' "$parsed"
}

parse_projected_var_line() {
  local line="$1"
  local parsed
  parsed="$(sed -E 's/^final[[:space:]]+\|[[:space:]]+proj mean_abs[[:space:]]+([^[:space:]]+)[[:space:]]+\|[[:space:]]+var_mean[[:space:]]+([^[:space:]]+)$/\2/' <<< "$line")"
  if [[ "$parsed" == "$line" ]]; then
    return 1
  fi
  printf '%s' "$parsed"
}

parse_target_drift_line() {
  local line="$1"
  local parsed
  parsed="$(sed -E 's/^final[[:space:]]+\|[[:space:]]+target drift[[:space:]]+([^[:space:]]+)$/\1/' <<< "$line")"
  if [[ "$parsed" == "$line" ]]; then
    return 1
  fi
  printf '%s' "$parsed"
}

parse_velocity_bank_line() {
  local line="$1"
  local parsed
  parsed="$(sed -E 's/^(initial|final)[[:space:]]+\|[[:space:]]+velocity bank mrr[[:space:]]+([^[:space:]]+)[[:space:]]+\|[[:space:]]+top1[[:space:]]+([^[:space:]]+)[[:space:]]+\|[[:space:]]+mean_rank[[:space:]]+([^[:space:]]+)[[:space:]]+\|[[:space:]]+samples[[:space:]]+([^[:space:]]+)[[:space:]]+\|[[:space:]]+candidates[[:space:]]+([^[:space:]]+)$/\2 \3 \4 \5 \6/' <<< "$line")"
  if [[ "$parsed" == "$line" ]]; then
    return 1
  fi
  printf '%s' "$parsed"
}

parse_signed_velocity_bank_line() {
  local line="$1"
  if [[ ! "$line" =~ ^(initial|final)[[:space:]]+\|[[:space:]]+signed[[:space:]]+velocity[[:space:]]+bank[[:space:]]+neg_mrr[[:space:]] ]]; then
    return 1
  fi
  awk '{print $7, $10, $13, $16, $19, $22, $25, $28, $31, $34, $37, $40, $43, $46, $49}' <<< "$line"
}

parse_target_bank_separability_line() {
  local line="$1"
  if [[ ! "$line" =~ ^(initial|final)[[:space:]]+\|[[:space:]]+target[[:space:]]+bank[[:space:]]+separability[[:space:]]+oracle_mrr[[:space:]] ]]; then
    return 1
  fi
  awk '
    {
      for (i = 1; i <= NF; i++) {
        values[$i] = $(i + 1)
      }
      print values["oracle_mrr"], values["top1"], values["true_dist"], values["true_dist_max"], values["nearest_wrong"], values["nearest_wrong_min"], values["margin"], values["margin_min"], values["neg_nearest_wrong"], values["pos_nearest_wrong"], values["slow_nearest_wrong"], values["fast_nearest_wrong"], values["sign_margin"], values["speed_margin"], values["samples"]
    }
  ' <<< "$line"
}

row_status() {
  local train_pred_start="$1"
  local train_pred_end="$2"
  local val_pred_start="$3"
  local val_pred_end="$4"
  local train_obj_start="$5"
  local train_obj_end="$6"
  local val_obj_start="$7"
  local val_obj_end="$8"
  local pred_min_std="$9"
  local target_min_std="${10}"
  local proj_var_mean="${11}"

  if [[ "$(lt_bool "$train_pred_end" "$train_pred_start")" != "true" ]]; then
    printf '%s' "accept_failed"
  elif [[ "$(lt_bool "$val_pred_end" "$val_pred_start")" != "true" ]]; then
    printf '%s' "accept_failed"
  elif [[ "$(lt_bool "$train_obj_end" "$train_obj_start")" != "true" ]]; then
    printf '%s' "accept_failed"
  elif [[ "$(lt_bool "$val_obj_end" "$val_obj_start")" != "true" ]]; then
    printf '%s' "accept_failed"
  elif [[ "$(gt_threshold_bool "$pred_min_std" "$MIN_STD_THRESHOLD")" != "true" ]]; then
    printf '%s' "accept_failed"
  elif [[ "$(gt_threshold_bool "$target_min_std" "$MIN_STD_THRESHOLD")" != "true" ]]; then
    printf '%s' "accept_failed"
  elif [[ "$proj_var_mean" != "na" && "$(gt_threshold_bool "$proj_var_mean" "0.000001")" != "true" ]]; then
    printf '%s' "accept_failed"
  else
    printf '%s' "ok"
  fi
}

emit_row() {
  local path="$1"
  local predictor="$2"
  local seed="$3"
  local encoder_lr="$4"
  local target_momentum_start="$5"
  local target_momentum_end="$6"
  local target_momentum_warmup_steps="$7"
  local train_pred_start="$8"
  local train_pred_end="$9"
  local val_pred_start="${10}"
  local val_pred_end="${11}"
  local train_obj_start="${12}"
  local train_obj_end="${13}"
  local val_obj_start="${14}"
  local val_obj_end="${15}"
  local pred_min_std="${16}"
  local target_min_std="${17}"
  local proj_var_mean="${18}"
  local target_drift_end="${19}"
  local status="${20}"
  local encoder_mode
  encoder_mode="$(encoder_mode_label)"
  local velocity_bank_mrr_start_value="${velocity_bank_mrr_start:-na}"
  local velocity_bank_mrr_end_value="${velocity_bank_mrr_end:-na}"
  local velocity_bank_top1_start_value="${velocity_bank_top1_start:-na}"
  local velocity_bank_top1_end_value="${velocity_bank_top1_end:-na}"
  local velocity_bank_mean_rank_start_value="${velocity_bank_mean_rank_start:-na}"
  local velocity_bank_mean_rank_end_value="${velocity_bank_mean_rank_end:-na}"
  local velocity_bank_samples_value="${velocity_bank_samples:-na}"
  local velocity_bank_candidates_value="${velocity_bank_candidates:-na}"
  local signed_bank_neg_mrr_end_value="${signed_bank_neg_mrr_end:-na}"
  local signed_bank_pos_mrr_end_value="${signed_bank_pos_mrr_end:-na}"
  local signed_bank_slow_mrr_end_value="${signed_bank_slow_mrr_end:-na}"
  local signed_bank_fast_mrr_end_value="${signed_bank_fast_mrr_end:-na}"
  local signed_bank_sign_top1_end_value="${signed_bank_sign_top1_end:-na}"
  local signed_bank_speed_top1_end_value="${signed_bank_speed_top1_end:-na}"
  local signed_bank_samples_value="${signed_bank_samples:-na}"
  local signed_bank_true_neg_best_neg_value="${signed_bank_true_neg_best_neg:-na}"
  local signed_bank_true_neg_best_pos_value="${signed_bank_true_neg_best_pos:-na}"
  local signed_bank_true_pos_best_neg_value="${signed_bank_true_pos_best_neg:-na}"
  local signed_bank_true_pos_best_pos_value="${signed_bank_true_pos_best_pos:-na}"
  local signed_bank_true_slow_best_slow_value="${signed_bank_true_slow_best_slow:-na}"
  local signed_bank_true_slow_best_fast_value="${signed_bank_true_slow_best_fast:-na}"
  local signed_bank_true_fast_best_slow_value="${signed_bank_true_fast_best_slow:-na}"
  local signed_bank_true_fast_best_fast_value="${signed_bank_true_fast_best_fast:-na}"
  local target_bank_oracle_mrr_end_value="${target_bank_oracle_mrr_end:-na}"
  local target_bank_oracle_top1_end_value="${target_bank_oracle_top1_end:-na}"
  local target_bank_true_distance_end_value="${target_bank_true_distance_end:-na}"
  local target_bank_true_distance_max_end_value="${target_bank_true_distance_max_end:-na}"
  local target_bank_nearest_wrong_end_value="${target_bank_nearest_wrong_end:-na}"
  local target_bank_nearest_wrong_min_end_value="${target_bank_nearest_wrong_min_end:-na}"
  local target_bank_margin_end_value="${target_bank_margin_end:-na}"
  local target_bank_margin_min_end_value="${target_bank_margin_min_end:-na}"
  local target_bank_neg_nearest_wrong_end_value="${target_bank_neg_nearest_wrong_end:-na}"
  local target_bank_pos_nearest_wrong_end_value="${target_bank_pos_nearest_wrong_end:-na}"
  local target_bank_slow_nearest_wrong_end_value="${target_bank_slow_nearest_wrong_end:-na}"
  local target_bank_fast_nearest_wrong_end_value="${target_bank_fast_nearest_wrong_end:-na}"
  local target_bank_sign_margin_end_value="${target_bank_sign_margin_end:-na}"
  local target_bank_speed_margin_end_value="${target_bank_speed_margin_end:-na}"
  local target_bank_samples_value="${target_bank_samples:-na}"

  printf 'schema=%s temporal_task=%s path=%s predictor=%s residual_delta_scale=%s projector_drift_weight=%s seed=%s steps=%s encoder_mode=%s encoder_lr=%s target_momentum_start=%s target_momentum_end=%s target_momentum_warmup_steps=%s train_pred_start=%s train_pred_end=%s val_pred_start=%s val_pred_end=%s train_obj_start=%s train_obj_end=%s val_obj_start=%s val_obj_end=%s pred_min_std_final=%s target_min_std_final=%s proj_var_mean_final=%s target_drift_end=%s velocity_bank_mrr_start=%s velocity_bank_mrr_end=%s velocity_bank_top1_start=%s velocity_bank_top1_end=%s velocity_bank_mean_rank_start=%s velocity_bank_mean_rank_end=%s velocity_bank_samples=%s velocity_bank_candidates=%s signed_bank_neg_mrr_end=%s signed_bank_pos_mrr_end=%s signed_bank_slow_mrr_end=%s signed_bank_fast_mrr_end=%s signed_bank_sign_top1_end=%s signed_bank_speed_top1_end=%s signed_bank_samples=%s signed_bank_true_neg_best_neg=%s signed_bank_true_neg_best_pos=%s signed_bank_true_pos_best_neg=%s signed_bank_true_pos_best_pos=%s signed_bank_true_slow_best_slow=%s signed_bank_true_slow_best_fast=%s signed_bank_true_fast_best_slow=%s signed_bank_true_fast_best_fast=%s target_bank_oracle_mrr_end=%s target_bank_oracle_top1_end=%s target_bank_true_distance_end=%s target_bank_true_distance_max_end=%s target_bank_nearest_wrong_end=%s target_bank_nearest_wrong_min_end=%s target_bank_margin_end=%s target_bank_margin_min_end=%s target_bank_neg_nearest_wrong_end=%s target_bank_pos_nearest_wrong_end=%s target_bank_slow_nearest_wrong_end=%s target_bank_fast_nearest_wrong_end=%s target_bank_sign_margin_end=%s target_bank_speed_margin_end=%s target_bank_samples=%s status=%s\n' \
    "$SCHEMA" "$TEMPORAL_TASK" "$path" "$predictor" "$RESIDUAL_DELTA_SCALE" "$PROJECTOR_DRIFT_WEIGHT" "$seed" "$TRAIN_STEPS" "$encoder_mode" "$encoder_lr" \
    "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" \
    "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" \
    "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" \
    "$pred_min_std" "$target_min_std" "$proj_var_mean" "$target_drift_end" \
    "$velocity_bank_mrr_start_value" "$velocity_bank_mrr_end_value" "$velocity_bank_top1_start_value" "$velocity_bank_top1_end_value" \
    "$velocity_bank_mean_rank_start_value" "$velocity_bank_mean_rank_end_value" "$velocity_bank_samples_value" "$velocity_bank_candidates_value" \
    "$signed_bank_neg_mrr_end_value" "$signed_bank_pos_mrr_end_value" "$signed_bank_slow_mrr_end_value" "$signed_bank_fast_mrr_end_value" \
    "$signed_bank_sign_top1_end_value" "$signed_bank_speed_top1_end_value" "$signed_bank_samples_value" \
    "$signed_bank_true_neg_best_neg_value" "$signed_bank_true_neg_best_pos_value" "$signed_bank_true_pos_best_neg_value" "$signed_bank_true_pos_best_pos_value" \
    "$signed_bank_true_slow_best_slow_value" "$signed_bank_true_slow_best_fast_value" "$signed_bank_true_fast_best_slow_value" "$signed_bank_true_fast_best_fast_value" \
    "$target_bank_oracle_mrr_end_value" "$target_bank_oracle_top1_end_value" "$target_bank_true_distance_end_value" "$target_bank_true_distance_max_end_value" \
    "$target_bank_nearest_wrong_end_value" "$target_bank_nearest_wrong_min_end_value" "$target_bank_margin_end_value" "$target_bank_margin_min_end_value" \
    "$target_bank_neg_nearest_wrong_end_value" "$target_bank_pos_nearest_wrong_end_value" \
    "$target_bank_slow_nearest_wrong_end_value" "$target_bank_fast_nearest_wrong_end_value" "$target_bank_sign_margin_end_value" \
    "$target_bank_speed_margin_end_value" "$target_bank_samples_value" "$status"

  if [[ -n "$REPORT_PATH" ]]; then
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$SCHEMA" "$TEMPORAL_TASK" "$path" "$predictor" "$RESIDUAL_DELTA_SCALE" "$PROJECTOR_DRIFT_WEIGHT" "$seed" "$TRAIN_STEPS" "$encoder_mode" "$encoder_lr" \
      "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" \
      "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" \
      "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" \
      "$pred_min_std" "$target_min_std" "$proj_var_mean" "$target_drift_end" \
      "$velocity_bank_mrr_start_value" "$velocity_bank_mrr_end_value" "$velocity_bank_top1_start_value" "$velocity_bank_top1_end_value" \
      "$velocity_bank_mean_rank_start_value" "$velocity_bank_mean_rank_end_value" "$velocity_bank_samples_value" "$velocity_bank_candidates_value" \
      "$signed_bank_neg_mrr_end_value" "$signed_bank_pos_mrr_end_value" "$signed_bank_slow_mrr_end_value" "$signed_bank_fast_mrr_end_value" \
      "$signed_bank_sign_top1_end_value" "$signed_bank_speed_top1_end_value" "$signed_bank_samples_value" \
      "$signed_bank_true_neg_best_neg_value" "$signed_bank_true_neg_best_pos_value" "$signed_bank_true_pos_best_neg_value" "$signed_bank_true_pos_best_pos_value" \
      "$signed_bank_true_slow_best_slow_value" "$signed_bank_true_slow_best_fast_value" "$signed_bank_true_fast_best_slow_value" "$signed_bank_true_fast_best_fast_value" \
      "$target_bank_oracle_mrr_end_value" "$target_bank_oracle_top1_end_value" "$target_bank_true_distance_end_value" "$target_bank_true_distance_max_end_value" \
      "$target_bank_nearest_wrong_end_value" "$target_bank_nearest_wrong_min_end_value" "$target_bank_margin_end_value" "$target_bank_margin_min_end_value" \
      "$target_bank_neg_nearest_wrong_end_value" "$target_bank_pos_nearest_wrong_end_value" \
      "$target_bank_slow_nearest_wrong_end_value" "$target_bank_fast_nearest_wrong_end_value" "$target_bank_sign_margin_end_value" \
      "$target_bank_speed_margin_end_value" "$target_bank_samples_value" "$status" >> "$REPORT_PATH"
  fi
}

run_one() {
  local path="$1"
  local seed="$2"
  local predictor="$3"
  local example=""
  local encoder_lr=""
  local target_momentum_start="na"
  local target_momentum_end="na"
  local target_momentum_warmup_steps="na"
  local log_file
  local -a extra_args=()
  local velocity_bank_mrr_start="na"
  local velocity_bank_mrr_end="na"
  local velocity_bank_top1_start="na"
  local velocity_bank_top1_end="na"
  local velocity_bank_mean_rank_start="na"
  local velocity_bank_mean_rank_end="na"
  local velocity_bank_samples="na"
  local velocity_bank_candidates="na"
  local signed_bank_neg_mrr_end="na"
  local signed_bank_pos_mrr_end="na"
  local signed_bank_slow_mrr_end="na"
  local signed_bank_fast_mrr_end="na"
  local signed_bank_sign_top1_end="na"
  local signed_bank_speed_top1_end="na"
  local signed_bank_samples="na"
  local signed_bank_true_neg_best_neg="na"
  local signed_bank_true_neg_best_pos="na"
  local signed_bank_true_pos_best_neg="na"
  local signed_bank_true_pos_best_pos="na"
  local signed_bank_true_slow_best_slow="na"
  local signed_bank_true_slow_best_fast="na"
  local signed_bank_true_fast_best_slow="na"
  local signed_bank_true_fast_best_fast="na"
  local target_bank_oracle_mrr_end="na"
  local target_bank_oracle_top1_end="na"
  local target_bank_true_distance_end="na"
  local target_bank_true_distance_max_end="na"
  local target_bank_nearest_wrong_end="na"
  local target_bank_nearest_wrong_min_end="na"
  local target_bank_margin_end="na"
  local target_bank_margin_min_end="na"
  local target_bank_neg_nearest_wrong_end="na"
  local target_bank_pos_nearest_wrong_end="na"
  local target_bank_slow_nearest_wrong_end="na"
  local target_bank_fast_nearest_wrong_end="na"
  local target_bank_sign_margin_end="na"
  local target_bank_speed_margin_end="na"
  local target_bank_samples="na"

  if [[ "$path" == "unprojected" ]]; then
    example="train_vision_jepa_random_temporal"
    encoder_lr="$UNPROJECTED_ENCODER_LR"
  else
    example="train_vision_jepa_random_temporal_projected"
    encoder_lr="$PROJECTED_ENCODER_LR"
    target_momentum_start="$PROJECTED_TARGET_MOMENTUM_START"
    target_momentum_end="$PROJECTED_TARGET_MOMENTUM_END"
    target_momentum_warmup_steps="$PROJECTED_TARGET_MOMENTUM_WARMUP_STEPS"
    if [[ "$PROJECTED_TARGET_MOMENTUM_WARMUP_STEPS" != "0" || "$PROJECTED_TARGET_MOMENTUM_START" != "$PROJECTED_TARGET_MOMENTUM_END" ]]; then
      extra_args+=(--target-momentum-start "$PROJECTED_TARGET_MOMENTUM_START")
      extra_args+=(--target-momentum-end "$PROJECTED_TARGET_MOMENTUM_END")
      extra_args+=(--target-momentum-warmup-steps "$PROJECTED_TARGET_MOMENTUM_WARMUP_STEPS")
    else
      extra_args+=(--target-momentum "$PROJECTED_TARGET_MOMENTUM_END")
    fi
  fi

  if [[ -n "$COMPACT_ENCODER_MODE" ]]; then
    extra_args+=(--compact-encoder-mode "$COMPACT_ENCODER_MODE")
  fi
  extra_args+=(--residual-delta-scale "$RESIDUAL_DELTA_SCALE")
  extra_args+=(--projector-drift-weight "$PROJECTOR_DRIFT_WEIGHT")
  extra_args+=(--temporal-task "$TEMPORAL_TASK")

  log_file="$(mktemp)"
  local run_failed="false"
  if (( ${#extra_args[@]} > 0 )); then
    if ! cargo run --manifest-path "$MANIFEST_PATH" --example "$example" -- \
      --train-base-seed "$seed" \
      --train-steps "$TRAIN_STEPS" \
      --log "$LOG_EVERY" \
      --encoder-lr "$encoder_lr" \
      --predictor-mode "$predictor" \
      "${extra_args[@]}" >"$log_file" 2>&1; then
      run_failed="true"
    fi
  elif ! cargo run --manifest-path "$MANIFEST_PATH" --example "$example" -- \
    --train-base-seed "$seed" \
    --train-steps "$TRAIN_STEPS" \
    --log "$LOG_EVERY" \
    --encoder-lr "$encoder_lr" \
    --predictor-mode "$predictor" >"$log_file" 2>&1; then
    run_failed="true"
  fi

  if [[ "$run_failed" == "true" ]]; then
    failures=$((failures + 1))
    emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" \
      "na" "na" "na" "na" "na" "na" "na" "na" "na" "na" "na" "na" "run_failed"
    sed -n '1,40p' "$log_file" >&2
    rm -f "$log_file"
    return 0
  fi

  local train_pred_start train_pred_end val_pred_start val_pred_end
  local train_obj_start train_obj_end val_obj_start val_obj_end
  local pred_mean_std pred_min_std target_mean_std target_min_std
  local proj_var_mean="na"
  local target_drift_end="na"
  local parsed initial_train_line initial_val_line final_train_line final_val_line
  local pred_health_line target_health_line
  local initial_velocity_bank_line final_velocity_bank_line
  local final_signed_velocity_bank_line
  local final_target_bank_separability_line

  pred_health_line="$(grep -m1 '^final prediction health |' "$log_file" || true)"
  target_health_line="$(grep -m1 '^final target health |' "$log_file" || true)"

  if [[ "$path" == "unprojected" ]]; then
    initial_train_line="$(grep -m1 '^initial | probe train ' "$log_file" || true)"
    final_train_line="$(grep -m1 '^final | probe train ' "$log_file" || true)"
    if ! parsed="$(parse_unprojected_probe_line "$initial_train_line")"; then
      failures=$((failures + 1))
      emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" \
        "na" "na" "na" "na" "na" "na" "na" "na" "na" "na" "na" "na" "parse_failed"
      rm -f "$log_file"
      return 0
    fi
    read -r train_pred_start val_pred_start <<< "$parsed"
    if ! parsed="$(parse_unprojected_probe_line "$final_train_line")"; then
      failures=$((failures + 1))
      emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" \
        "na" "na" "na" "na" "na" "na" "na" "na" "na" "na" "na" "na" "parse_failed"
      rm -f "$log_file"
      return 0
    fi
    read -r train_pred_end val_pred_end <<< "$parsed"
    train_obj_start="$train_pred_start"
    train_obj_end="$train_pred_end"
    val_obj_start="$val_pred_start"
    val_obj_end="$val_pred_end"
  else
    initial_train_line="$(grep -m1 '^initial | train pred ' "$log_file" || true)"
    initial_val_line="$(grep -m1 '^initial | val pred ' "$log_file" || true)"
    final_train_line="$(grep -m1 '^final | train pred ' "$log_file" || true)"
    final_val_line="$(grep -m1 '^final | val pred ' "$log_file" || true)"

    if ! parsed="$(parse_projected_loss_line "$initial_train_line")"; then failures=$((failures + 1)); emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "na" "na" "na" "na" "na" "na" "na" "na" "na" "na" "na" "na" "parse_failed"; rm -f "$log_file"; return 0; fi
    read -r train_pred_start train_obj_start <<< "$parsed"
    if ! parsed="$(parse_projected_loss_line "$initial_val_line")"; then failures=$((failures + 1)); emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "na" "na" "na" "na" "na" "na" "na" "na" "na" "na" "na" "na" "parse_failed"; rm -f "$log_file"; return 0; fi
    read -r val_pred_start val_obj_start <<< "$parsed"
    if ! parsed="$(parse_projected_loss_line "$final_train_line")"; then failures=$((failures + 1)); emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "na" "na" "na" "na" "na" "na" "na" "na" "na" "na" "na" "na" "parse_failed"; rm -f "$log_file"; return 0; fi
    read -r train_pred_end train_obj_end <<< "$parsed"
    if ! parsed="$(parse_projected_loss_line "$final_val_line")"; then failures=$((failures + 1)); emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "na" "na" "na" "na" "na" "na" "na" "na" "na" "na" "na" "na" "parse_failed"; rm -f "$log_file"; return 0; fi
    read -r val_pred_end val_obj_end <<< "$parsed"

    if ! proj_var_mean="$(parse_projected_var_line "$(grep -m1 '^final | proj mean_abs ' "$log_file" || true)")"; then proj_var_mean="na"; fi
    if ! target_drift_end="$(parse_target_drift_line "$(grep -m1 '^final | target drift ' "$log_file" || true)")"; then target_drift_end="na"; fi
    if is_velocity_bank_task; then
      initial_velocity_bank_line="$(grep -m1 '^initial | velocity bank mrr ' "$log_file" || true)"
      final_velocity_bank_line="$(grep -m1 '^final | velocity bank mrr ' "$log_file" || true)"

      if ! parsed="$(parse_velocity_bank_line "$initial_velocity_bank_line")"; then
        failures=$((failures + 1))
        emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "na" "na" "$proj_var_mean" "$target_drift_end" "parse_failed"
        rm -f "$log_file"
        return 0
      fi
      read -r velocity_bank_mrr_start velocity_bank_top1_start velocity_bank_mean_rank_start velocity_bank_samples velocity_bank_candidates <<< "$parsed"

      if ! parsed="$(parse_velocity_bank_line "$final_velocity_bank_line")"; then
        failures=$((failures + 1))
        emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "na" "na" "$proj_var_mean" "$target_drift_end" "parse_failed"
        rm -f "$log_file"
        return 0
      fi
      read -r velocity_bank_mrr_end velocity_bank_top1_end velocity_bank_mean_rank_end velocity_bank_samples velocity_bank_candidates <<< "$parsed"
    fi
    if is_signed_velocity_bank_task; then
      final_signed_velocity_bank_line="$(grep -m1 '^final | signed velocity bank neg_mrr ' "$log_file" || true)"

      if ! parsed="$(parse_signed_velocity_bank_line "$final_signed_velocity_bank_line")"; then
        failures=$((failures + 1))
        emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "na" "na" "$proj_var_mean" "$target_drift_end" "parse_failed"
        rm -f "$log_file"
        return 0
      fi
      read -r signed_bank_neg_mrr_end signed_bank_pos_mrr_end signed_bank_slow_mrr_end signed_bank_fast_mrr_end signed_bank_sign_top1_end signed_bank_speed_top1_end signed_bank_samples signed_bank_true_neg_best_neg signed_bank_true_neg_best_pos signed_bank_true_pos_best_neg signed_bank_true_pos_best_pos signed_bank_true_slow_best_slow signed_bank_true_slow_best_fast signed_bank_true_fast_best_slow signed_bank_true_fast_best_fast <<< "$parsed"

      final_target_bank_separability_line="$(grep -m1 '^final | target bank separability oracle_mrr ' "$log_file" || true)"

      if ! parsed="$(parse_target_bank_separability_line "$final_target_bank_separability_line")"; then
        failures=$((failures + 1))
        emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "na" "na" "$proj_var_mean" "$target_drift_end" "parse_failed"
        rm -f "$log_file"
        return 0
      fi
      read -r target_bank_oracle_mrr_end target_bank_oracle_top1_end target_bank_true_distance_end target_bank_true_distance_max_end target_bank_nearest_wrong_end target_bank_nearest_wrong_min_end target_bank_margin_end target_bank_margin_min_end target_bank_neg_nearest_wrong_end target_bank_pos_nearest_wrong_end target_bank_slow_nearest_wrong_end target_bank_fast_nearest_wrong_end target_bank_sign_margin_end target_bank_speed_margin_end target_bank_samples <<< "$parsed"
    fi
  fi

  if ! parsed="$(parse_health_line "$pred_health_line")"; then
    failures=$((failures + 1))
    emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" \
      "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "na" "na" "$proj_var_mean" "$target_drift_end" "parse_failed"
    rm -f "$log_file"
    return 0
  fi
  read -r pred_mean_std pred_min_std <<< "$parsed"

  if ! parsed="$(parse_health_line "$target_health_line")"; then
    failures=$((failures + 1))
    emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" \
      "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "$pred_min_std" "na" "$proj_var_mean" "$target_drift_end" "parse_failed"
    rm -f "$log_file"
    return 0
  fi
  read -r target_mean_std target_min_std <<< "$parsed"

  local status
  status="$(row_status "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "$pred_min_std" "$target_min_std" "$proj_var_mean")"
  emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" \
    "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "$pred_min_std" "$target_min_std" "$proj_var_mean" "$target_drift_end" "$status"
  rm -f "$log_file"
}

run_path() {
  local path="$1"
  shift
  local -a seeds=("$@")

  if [[ "$SCENARIO" != "all" && "$SCENARIO" != "$path" ]]; then
    return 0
  fi

  for seed in "${seeds[@]}"; do
    for predictor in "${PREDICTOR_MODES[@]}"; do
      run_one "$path" "$seed" "$predictor"
    done
  done
}

run_path "unprojected" "${UNPROJECTED_SEEDS[@]}"
run_path "projected" "${PROJECTED_SEEDS[@]}"

exit "$failures"
