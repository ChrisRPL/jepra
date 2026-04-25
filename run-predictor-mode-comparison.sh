#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_PATH="${JEPRA_MANIFEST_PATH:-$ROOT_DIR/crates/jepra-core/Cargo.toml}"
SCHEMA="jepra_predictor_compare_v12"
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
SIGNED_MARGIN_WEIGHT="${JEPRA_SIGNED_MARGIN_WEIGHT:-${JEPRA_MARGIN_OBJECTIVE_WEIGHT:-0.0}}"
SIGNED_MARGIN_BANK_GAP="${JEPRA_SIGNED_MARGIN_BANK_GAP:-0.05}"
SIGNED_MARGIN_SIGN_GAP="${JEPRA_SIGNED_MARGIN_SIGN_GAP:-0.05}"
SIGNED_MARGIN_SPEED_GAP="${JEPRA_SIGNED_MARGIN_SPEED_GAP:-0.05}"
SIGNED_MARGIN_BANK_WEIGHT="${JEPRA_SIGNED_MARGIN_BANK_WEIGHT:-1.0}"
SIGNED_MARGIN_SIGN_WEIGHT="${JEPRA_SIGNED_MARGIN_SIGN_WEIGHT:-1.0}"
SIGNED_MARGIN_SPEED_WEIGHT="${JEPRA_SIGNED_MARGIN_SPEED_WEIGHT:-1.0}"
SIGNED_BANK_SOFTMAX_WEIGHT="${JEPRA_SIGNED_BANK_SOFTMAX_WEIGHT:-0.0}"
SIGNED_BANK_SOFTMAX_TEMPERATURE="${JEPRA_SIGNED_BANK_SOFTMAX_TEMPERATURE:-1.0}"
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
  JEPRA_COMPACT_ENCODER_MODE                    Optional compact encoder mode: base|stronger|signed-direction
  JEPRA_RESIDUAL_DELTA_SCALE                    Residual-bottleneck delta scale (default: 1.0)
  JEPRA_PROJECTOR_DRIFT_WEIGHT                  Projected online-projector drift regularizer weight (default: 0.0)
  JEPRA_SIGNED_MARGIN_WEIGHT                    Signed-margin objective weight (default: 0.0)
  JEPRA_SIGNED_MARGIN_BANK_GAP                  Signed-margin all-wrong hinge gap (default: 0.05)
  JEPRA_SIGNED_MARGIN_SIGN_GAP                  Signed-margin opposite-sign hinge gap (default: 0.05)
  JEPRA_SIGNED_MARGIN_SPEED_GAP                 Signed-margin same-sign speed hinge gap (default: 0.05)
  JEPRA_SIGNED_MARGIN_BANK_WEIGHT               Signed-margin all-wrong component weight (default: 1.0)
  JEPRA_SIGNED_MARGIN_SIGN_WEIGHT               Signed-margin sign component weight (default: 1.0)
  JEPRA_SIGNED_MARGIN_SPEED_WEIGHT              Signed-margin speed component weight (default: 1.0)
  JEPRA_SIGNED_BANK_SOFTMAX_WEIGHT              Signed-bank softmax objective weight (default: 0.0)
  JEPRA_SIGNED_BANK_SOFTMAX_TEMPERATURE         Signed-bank softmax objective temperature (default: 1.0)
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
  printf 'schema,temporal_task,path,predictor,residual_delta_scale,projector_drift_weight,signed_margin_weight,signed_margin_bank_gap,signed_margin_sign_gap,signed_margin_speed_gap,signed_margin_bank_weight,signed_margin_sign_weight,signed_margin_speed_weight,signed_bank_softmax_weight,signed_bank_softmax_temperature,seed,steps,encoder_mode,encoder_lr,target_momentum_start,target_momentum_end,target_momentum_warmup_steps,train_pred_start,train_pred_end,val_pred_start,val_pred_end,train_obj_start,train_obj_end,val_obj_start,val_obj_end,pred_min_std_final,target_min_std_final,proj_var_mean_final,target_drift_end,velocity_bank_mrr_start,velocity_bank_mrr_end,velocity_bank_top1_start,velocity_bank_top1_end,velocity_bank_mean_rank_start,velocity_bank_mean_rank_end,velocity_bank_samples,velocity_bank_candidates,signed_bank_neg_mrr_end,signed_bank_pos_mrr_end,signed_bank_slow_mrr_end,signed_bank_fast_mrr_end,signed_bank_sign_top1_end,signed_bank_speed_top1_end,signed_bank_samples,signed_bank_true_neg_best_neg,signed_bank_true_neg_best_pos,signed_bank_true_pos_best_neg,signed_bank_true_pos_best_pos,signed_bank_true_slow_best_slow,signed_bank_true_slow_best_fast,signed_bank_true_fast_best_slow,signed_bank_true_fast_best_fast,target_bank_oracle_mrr_end,target_bank_oracle_top1_end,target_bank_true_distance_end,target_bank_true_distance_max_end,target_bank_nearest_wrong_end,target_bank_nearest_wrong_min_end,target_bank_margin_end,target_bank_margin_min_end,target_bank_neg_nearest_wrong_end,target_bank_pos_nearest_wrong_end,target_bank_slow_nearest_wrong_end,target_bank_fast_nearest_wrong_end,target_bank_sign_margin_end,target_bank_speed_margin_end,target_bank_samples,prediction_bank_true_distance_end,prediction_bank_nearest_wrong_distance_end,prediction_bank_margin_end,prediction_bank_min_margin_end,prediction_bank_positive_margin_rate_end,prediction_bank_sign_margin_end,prediction_bank_speed_margin_end,prediction_bank_samples,signed_objective_all_loss_end,signed_objective_dx_neg2_loss_end,signed_objective_dx_neg1_loss_end,signed_objective_dx_pos1_loss_end,signed_objective_dx_pos2_loss_end,signed_objective_neg_loss_end,signed_objective_pos_loss_end,signed_objective_slow_loss_end,signed_objective_fast_loss_end,signed_objective_sign_gap_end,signed_objective_speed_gap_end,signed_objective_samples,signed_objective_dx_neg2_samples,signed_objective_dx_neg1_samples,signed_objective_dx_pos1_samples,signed_objective_dx_pos2_samples,signed_margin_bank_loss_end,signed_margin_sign_loss_end,signed_margin_speed_loss_end,signed_margin_weighted_loss_end,signed_margin_active_bank_rate_end,signed_margin_active_sign_rate_end,signed_margin_active_speed_rate_end,signed_margin_samples,signed_bank_softmax_loss_end,signed_bank_softmax_top1_end,signed_bank_softmax_true_probability_end,signed_bank_softmax_samples,state_latent_mrr_end,state_latent_top1_end,state_latent_sign_top1_end,state_latent_mean_rank_end,state_projection_mrr_end,state_projection_top1_end,state_projection_sign_top1_end,state_projection_mean_rank_end,state_support_samples,state_query_samples,state_candidates,status\n' > "$REPORT_PATH"
fi

encoder_mode_label() {
  case "$COMPACT_ENCODER_MODE" in
    "") printf '%s' "frozen-base" ;;
    "base") printf '%s' "compact-base" ;;
    "stronger") printf '%s' "compact-stronger" ;;
    "signed-direction") printf '%s' "compact-signed-direction" ;;
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

signed_margin_enabled() {
  awk -v value="$SIGNED_MARGIN_WEIGHT" 'BEGIN { if (value + 0 > 0) print "true"; else print "false" }'
}

signed_bank_softmax_enabled() {
  awk -v value="$SIGNED_BANK_SOFTMAX_WEIGHT" 'BEGIN { if (value + 0 > 0) print "true"; else print "false" }'
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

parse_prediction_bank_margin_line() {
  local line="$1"
  if [[ ! "$line" =~ ^(initial|final)[[:space:]]+\|[[:space:]]+signed[[:space:]]+prediction[[:space:]]+bank[[:space:]]+margin[[:space:]]+true_distance[[:space:]] ]]; then
    return 1
  fi
  awk '
    {
      for (i = 1; i <= NF; i++) {
        values[$i] = $(i + 1)
      }
      print values["true_distance"], values["nearest_wrong_distance"], values["margin"], values["min_margin"], values["positive_margin_rate"], values["sign_margin"], values["speed_margin"], values["samples"]
    }
  ' <<< "$line"
}

parse_signed_objective_error_breakdown_line() {
  local line="$1"
  if [[ ! "$line" =~ ^(initial|final)[[:space:]]+\|[[:space:]]+signed[[:space:]]+objective[[:space:]]+error[[:space:]]+breakdown[[:space:]]+all_loss[[:space:]] ]]; then
    return 1
  fi
  awk '
    {
      for (i = 1; i <= NF; i++) {
        values[$i] = $(i + 1)
      }
      print values["all_loss"], values["dx_neg2_loss"], values["dx_neg1_loss"], values["dx_pos1_loss"], values["dx_pos2_loss"], values["neg_loss"], values["pos_loss"], values["slow_loss"], values["fast_loss"], values["sign_gap"], values["speed_gap"], values["samples"], values["dx_neg2_samples"], values["dx_neg1_samples"], values["dx_pos1_samples"], values["dx_pos2_samples"]
    }
  ' <<< "$line"
}

parse_signed_margin_objective_line() {
  local line="$1"
  if [[ ! "$line" =~ ^(initial|final)[[:space:]]+\|[[:space:]]+signed[[:space:]]+margin[[:space:]]+objective[[:space:]]+bank_loss[[:space:]] ]]; then
    return 1
  fi
  awk '
    {
      for (i = 1; i <= NF; i++) {
        values[$i] = $(i + 1)
      }
      print values["bank_loss"], values["sign_loss"], values["speed_loss"], values["weighted_loss"], values["active_bank_rate"], values["active_sign_rate"], values["active_speed_rate"], values["samples"]
    }
  ' <<< "$line"
}

parse_signed_bank_softmax_objective_line() {
  local line="$1"
  if [[ ! "$line" =~ ^(initial|final)[[:space:]]+\|[[:space:]]+signed[[:space:]]+bank[[:space:]]+softmax[[:space:]]+objective[[:space:]]+loss[[:space:]] ]]; then
    return 1
  fi
  awk '
    {
      for (i = 1; i <= NF; i++) {
        values[$i] = $(i + 1)
      }
      print values["loss"], values["top1"], values["true_probability"], values["samples"]
    }
  ' <<< "$line"
}

parse_signed_state_separability_line() {
  local line="$1"
  if [[ ! "$line" =~ ^(initial|final)[[:space:]]+\|[[:space:]]+signed[[:space:]]+state[[:space:]]+separability[[:space:]]+latent_mrr[[:space:]] ]]; then
    return 1
  fi
  awk '
    {
      for (i = 1; i <= NF; i++) {
        values[$i] = $(i + 1)
      }
      print values["latent_mrr"], values["latent_top1"], values["latent_sign_top1"], values["latent_mean_rank"], values["projection_mrr"], values["projection_top1"], values["projection_sign_top1"], values["projection_mean_rank"], values["support_samples"], values["query_samples"], values["candidates"]
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
  local prediction_bank_true_distance_end_value="${prediction_bank_true_distance_end:-na}"
  local prediction_bank_nearest_wrong_distance_end_value="${prediction_bank_nearest_wrong_distance_end:-na}"
  local prediction_bank_margin_end_value="${prediction_bank_margin_end:-na}"
  local prediction_bank_min_margin_end_value="${prediction_bank_min_margin_end:-na}"
  local prediction_bank_positive_margin_rate_end_value="${prediction_bank_positive_margin_rate_end:-na}"
  local prediction_bank_sign_margin_end_value="${prediction_bank_sign_margin_end:-na}"
  local prediction_bank_speed_margin_end_value="${prediction_bank_speed_margin_end:-na}"
  local prediction_bank_samples_value="${prediction_bank_samples:-na}"
  local signed_objective_all_loss_end_value="${signed_objective_all_loss_end:-na}"
  local signed_objective_dx_neg2_loss_end_value="${signed_objective_dx_neg2_loss_end:-na}"
  local signed_objective_dx_neg1_loss_end_value="${signed_objective_dx_neg1_loss_end:-na}"
  local signed_objective_dx_pos1_loss_end_value="${signed_objective_dx_pos1_loss_end:-na}"
  local signed_objective_dx_pos2_loss_end_value="${signed_objective_dx_pos2_loss_end:-na}"
  local signed_objective_neg_loss_end_value="${signed_objective_neg_loss_end:-na}"
  local signed_objective_pos_loss_end_value="${signed_objective_pos_loss_end:-na}"
  local signed_objective_slow_loss_end_value="${signed_objective_slow_loss_end:-na}"
  local signed_objective_fast_loss_end_value="${signed_objective_fast_loss_end:-na}"
  local signed_objective_sign_gap_end_value="${signed_objective_sign_gap_end:-na}"
  local signed_objective_speed_gap_end_value="${signed_objective_speed_gap_end:-na}"
  local signed_objective_samples_value="${signed_objective_samples:-na}"
  local signed_objective_dx_neg2_samples_value="${signed_objective_dx_neg2_samples:-na}"
  local signed_objective_dx_neg1_samples_value="${signed_objective_dx_neg1_samples:-na}"
  local signed_objective_dx_pos1_samples_value="${signed_objective_dx_pos1_samples:-na}"
  local signed_objective_dx_pos2_samples_value="${signed_objective_dx_pos2_samples:-na}"
  local signed_margin_bank_loss_end_value="${signed_margin_bank_loss_end:-na}"
  local signed_margin_sign_loss_end_value="${signed_margin_sign_loss_end:-na}"
  local signed_margin_speed_loss_end_value="${signed_margin_speed_loss_end:-na}"
  local signed_margin_weighted_loss_end_value="${signed_margin_weighted_loss_end:-na}"
  local signed_margin_active_bank_rate_end_value="${signed_margin_active_bank_rate_end:-na}"
  local signed_margin_active_sign_rate_end_value="${signed_margin_active_sign_rate_end:-na}"
  local signed_margin_active_speed_rate_end_value="${signed_margin_active_speed_rate_end:-na}"
  local signed_margin_samples_value="${signed_margin_samples:-na}"
  local signed_bank_softmax_loss_end_value="${signed_bank_softmax_loss_end:-na}"
  local signed_bank_softmax_top1_end_value="${signed_bank_softmax_top1_end:-na}"
  local signed_bank_softmax_true_probability_end_value="${signed_bank_softmax_true_probability_end:-na}"
  local signed_bank_softmax_samples_value="${signed_bank_softmax_samples:-na}"
  local state_latent_mrr_end_value="${state_latent_mrr_end:-na}"
  local state_latent_top1_end_value="${state_latent_top1_end:-na}"
  local state_latent_sign_top1_end_value="${state_latent_sign_top1_end:-na}"
  local state_latent_mean_rank_end_value="${state_latent_mean_rank_end:-na}"
  local state_projection_mrr_end_value="${state_projection_mrr_end:-na}"
  local state_projection_top1_end_value="${state_projection_top1_end:-na}"
  local state_projection_sign_top1_end_value="${state_projection_sign_top1_end:-na}"
  local state_projection_mean_rank_end_value="${state_projection_mean_rank_end:-na}"
  local state_support_samples_value="${state_support_samples:-na}"
  local state_query_samples_value="${state_query_samples:-na}"
  local state_candidates_value="${state_candidates:-na}"

  printf 'schema=%s temporal_task=%s path=%s predictor=%s residual_delta_scale=%s projector_drift_weight=%s signed_margin_weight=%s signed_margin_bank_gap=%s signed_margin_sign_gap=%s signed_margin_speed_gap=%s signed_margin_bank_weight=%s signed_margin_sign_weight=%s signed_margin_speed_weight=%s signed_bank_softmax_weight=%s signed_bank_softmax_temperature=%s seed=%s steps=%s encoder_mode=%s encoder_lr=%s target_momentum_start=%s target_momentum_end=%s target_momentum_warmup_steps=%s train_pred_start=%s train_pred_end=%s val_pred_start=%s val_pred_end=%s train_obj_start=%s train_obj_end=%s val_obj_start=%s val_obj_end=%s pred_min_std_final=%s target_min_std_final=%s proj_var_mean_final=%s target_drift_end=%s velocity_bank_mrr_start=%s velocity_bank_mrr_end=%s velocity_bank_top1_start=%s velocity_bank_top1_end=%s velocity_bank_mean_rank_start=%s velocity_bank_mean_rank_end=%s velocity_bank_samples=%s velocity_bank_candidates=%s signed_bank_neg_mrr_end=%s signed_bank_pos_mrr_end=%s signed_bank_slow_mrr_end=%s signed_bank_fast_mrr_end=%s signed_bank_sign_top1_end=%s signed_bank_speed_top1_end=%s signed_bank_samples=%s signed_bank_true_neg_best_neg=%s signed_bank_true_neg_best_pos=%s signed_bank_true_pos_best_neg=%s signed_bank_true_pos_best_pos=%s signed_bank_true_slow_best_slow=%s signed_bank_true_slow_best_fast=%s signed_bank_true_fast_best_slow=%s signed_bank_true_fast_best_fast=%s target_bank_oracle_mrr_end=%s target_bank_oracle_top1_end=%s target_bank_true_distance_end=%s target_bank_true_distance_max_end=%s target_bank_nearest_wrong_end=%s target_bank_nearest_wrong_min_end=%s target_bank_margin_end=%s target_bank_margin_min_end=%s target_bank_neg_nearest_wrong_end=%s target_bank_pos_nearest_wrong_end=%s target_bank_slow_nearest_wrong_end=%s target_bank_fast_nearest_wrong_end=%s target_bank_sign_margin_end=%s target_bank_speed_margin_end=%s target_bank_samples=%s prediction_bank_true_distance_end=%s prediction_bank_nearest_wrong_distance_end=%s prediction_bank_margin_end=%s prediction_bank_min_margin_end=%s prediction_bank_positive_margin_rate_end=%s prediction_bank_sign_margin_end=%s prediction_bank_speed_margin_end=%s prediction_bank_samples=%s signed_objective_all_loss_end=%s signed_objective_dx_neg2_loss_end=%s signed_objective_dx_neg1_loss_end=%s signed_objective_dx_pos1_loss_end=%s signed_objective_dx_pos2_loss_end=%s signed_objective_neg_loss_end=%s signed_objective_pos_loss_end=%s signed_objective_slow_loss_end=%s signed_objective_fast_loss_end=%s signed_objective_sign_gap_end=%s signed_objective_speed_gap_end=%s signed_objective_samples=%s signed_objective_dx_neg2_samples=%s signed_objective_dx_neg1_samples=%s signed_objective_dx_pos1_samples=%s signed_objective_dx_pos2_samples=%s signed_margin_bank_loss_end=%s signed_margin_sign_loss_end=%s signed_margin_speed_loss_end=%s signed_margin_weighted_loss_end=%s signed_margin_active_bank_rate_end=%s signed_margin_active_sign_rate_end=%s signed_margin_active_speed_rate_end=%s signed_margin_samples=%s signed_bank_softmax_loss_end=%s signed_bank_softmax_top1_end=%s signed_bank_softmax_true_probability_end=%s signed_bank_softmax_samples=%s state_latent_mrr_end=%s state_latent_top1_end=%s state_latent_sign_top1_end=%s state_latent_mean_rank_end=%s state_projection_mrr_end=%s state_projection_top1_end=%s state_projection_sign_top1_end=%s state_projection_mean_rank_end=%s state_support_samples=%s state_query_samples=%s state_candidates=%s status=%s\n' \
    "$SCHEMA" "$TEMPORAL_TASK" "$path" "$predictor" "$RESIDUAL_DELTA_SCALE" "$PROJECTOR_DRIFT_WEIGHT" \
    "$SIGNED_MARGIN_WEIGHT" "$SIGNED_MARGIN_BANK_GAP" "$SIGNED_MARGIN_SIGN_GAP" "$SIGNED_MARGIN_SPEED_GAP" \
    "$SIGNED_MARGIN_BANK_WEIGHT" "$SIGNED_MARGIN_SIGN_WEIGHT" "$SIGNED_MARGIN_SPEED_WEIGHT" \
    "$SIGNED_BANK_SOFTMAX_WEIGHT" "$SIGNED_BANK_SOFTMAX_TEMPERATURE" \
    "$seed" "$TRAIN_STEPS" "$encoder_mode" "$encoder_lr" \
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
    "$target_bank_speed_margin_end_value" "$target_bank_samples_value" \
    "$prediction_bank_true_distance_end_value" "$prediction_bank_nearest_wrong_distance_end_value" "$prediction_bank_margin_end_value" \
    "$prediction_bank_min_margin_end_value" "$prediction_bank_positive_margin_rate_end_value" "$prediction_bank_sign_margin_end_value" \
    "$prediction_bank_speed_margin_end_value" "$prediction_bank_samples_value" \
    "$signed_objective_all_loss_end_value" "$signed_objective_dx_neg2_loss_end_value" "$signed_objective_dx_neg1_loss_end_value" \
    "$signed_objective_dx_pos1_loss_end_value" "$signed_objective_dx_pos2_loss_end_value" "$signed_objective_neg_loss_end_value" \
    "$signed_objective_pos_loss_end_value" "$signed_objective_slow_loss_end_value" "$signed_objective_fast_loss_end_value" \
    "$signed_objective_sign_gap_end_value" "$signed_objective_speed_gap_end_value" "$signed_objective_samples_value" \
    "$signed_objective_dx_neg2_samples_value" "$signed_objective_dx_neg1_samples_value" "$signed_objective_dx_pos1_samples_value" \
    "$signed_objective_dx_pos2_samples_value" \
    "$signed_margin_bank_loss_end_value" "$signed_margin_sign_loss_end_value" "$signed_margin_speed_loss_end_value" \
    "$signed_margin_weighted_loss_end_value" "$signed_margin_active_bank_rate_end_value" "$signed_margin_active_sign_rate_end_value" \
    "$signed_margin_active_speed_rate_end_value" "$signed_margin_samples_value" \
    "$signed_bank_softmax_loss_end_value" "$signed_bank_softmax_top1_end_value" "$signed_bank_softmax_true_probability_end_value" \
    "$signed_bank_softmax_samples_value" \
    "$state_latent_mrr_end_value" "$state_latent_top1_end_value" "$state_latent_sign_top1_end_value" \
    "$state_latent_mean_rank_end_value" "$state_projection_mrr_end_value" "$state_projection_top1_end_value" \
    "$state_projection_sign_top1_end_value" "$state_projection_mean_rank_end_value" "$state_support_samples_value" \
    "$state_query_samples_value" "$state_candidates_value" "$status"

  if [[ -n "$REPORT_PATH" ]]; then
    local row_values=(
      "$SCHEMA" "$TEMPORAL_TASK" "$path" "$predictor" "$RESIDUAL_DELTA_SCALE" "$PROJECTOR_DRIFT_WEIGHT" \
      "$SIGNED_MARGIN_WEIGHT" "$SIGNED_MARGIN_BANK_GAP" "$SIGNED_MARGIN_SIGN_GAP" "$SIGNED_MARGIN_SPEED_GAP" \
      "$SIGNED_MARGIN_BANK_WEIGHT" "$SIGNED_MARGIN_SIGN_WEIGHT" "$SIGNED_MARGIN_SPEED_WEIGHT" \
      "$SIGNED_BANK_SOFTMAX_WEIGHT" "$SIGNED_BANK_SOFTMAX_TEMPERATURE" \
      "$seed" "$TRAIN_STEPS" "$encoder_mode" "$encoder_lr" \
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
      "$target_bank_speed_margin_end_value" "$target_bank_samples_value" \
      "$prediction_bank_true_distance_end_value" "$prediction_bank_nearest_wrong_distance_end_value" "$prediction_bank_margin_end_value" \
      "$prediction_bank_min_margin_end_value" "$prediction_bank_positive_margin_rate_end_value" "$prediction_bank_sign_margin_end_value" \
      "$prediction_bank_speed_margin_end_value" "$prediction_bank_samples_value" \
      "$signed_objective_all_loss_end_value" "$signed_objective_dx_neg2_loss_end_value" "$signed_objective_dx_neg1_loss_end_value" \
      "$signed_objective_dx_pos1_loss_end_value" "$signed_objective_dx_pos2_loss_end_value" "$signed_objective_neg_loss_end_value" \
      "$signed_objective_pos_loss_end_value" "$signed_objective_slow_loss_end_value" "$signed_objective_fast_loss_end_value" \
      "$signed_objective_sign_gap_end_value" "$signed_objective_speed_gap_end_value" "$signed_objective_samples_value" \
      "$signed_objective_dx_neg2_samples_value" "$signed_objective_dx_neg1_samples_value" "$signed_objective_dx_pos1_samples_value" \
      "$signed_objective_dx_pos2_samples_value" \
      "$signed_margin_bank_loss_end_value" "$signed_margin_sign_loss_end_value" "$signed_margin_speed_loss_end_value" \
      "$signed_margin_weighted_loss_end_value" "$signed_margin_active_bank_rate_end_value" "$signed_margin_active_sign_rate_end_value" \
      "$signed_margin_active_speed_rate_end_value" "$signed_margin_samples_value" \
      "$signed_bank_softmax_loss_end_value" "$signed_bank_softmax_top1_end_value" "$signed_bank_softmax_true_probability_end_value" \
      "$signed_bank_softmax_samples_value" \
      "$state_latent_mrr_end_value" "$state_latent_top1_end_value" "$state_latent_sign_top1_end_value" \
      "$state_latent_mean_rank_end_value" "$state_projection_mrr_end_value" "$state_projection_top1_end_value" \
      "$state_projection_sign_top1_end_value" "$state_projection_mean_rank_end_value" "$state_support_samples_value" \
      "$state_query_samples_value" "$state_candidates_value" "$status"
    )
    local IFS=,
    printf '%s\n' "${row_values[*]}" >> "$REPORT_PATH"
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
  local prediction_bank_true_distance_end="na"
  local prediction_bank_nearest_wrong_distance_end="na"
  local prediction_bank_margin_end="na"
  local prediction_bank_min_margin_end="na"
  local prediction_bank_positive_margin_rate_end="na"
  local prediction_bank_sign_margin_end="na"
  local prediction_bank_speed_margin_end="na"
  local prediction_bank_samples="na"
  local signed_objective_all_loss_end="na"
  local signed_objective_dx_neg2_loss_end="na"
  local signed_objective_dx_neg1_loss_end="na"
  local signed_objective_dx_pos1_loss_end="na"
  local signed_objective_dx_pos2_loss_end="na"
  local signed_objective_neg_loss_end="na"
  local signed_objective_pos_loss_end="na"
  local signed_objective_slow_loss_end="na"
  local signed_objective_fast_loss_end="na"
  local signed_objective_sign_gap_end="na"
  local signed_objective_speed_gap_end="na"
  local signed_objective_samples="na"
  local signed_objective_dx_neg2_samples="na"
  local signed_objective_dx_neg1_samples="na"
  local signed_objective_dx_pos1_samples="na"
  local signed_objective_dx_pos2_samples="na"
  local signed_margin_bank_loss_end="na"
  local signed_margin_sign_loss_end="na"
  local signed_margin_speed_loss_end="na"
  local signed_margin_weighted_loss_end="na"
  local signed_margin_active_bank_rate_end="na"
  local signed_margin_active_sign_rate_end="na"
  local signed_margin_active_speed_rate_end="na"
  local signed_margin_samples="na"
  local signed_bank_softmax_loss_end="na"
  local signed_bank_softmax_top1_end="na"
  local signed_bank_softmax_true_probability_end="na"
  local signed_bank_softmax_samples="na"
  local state_latent_mrr_end="na"
  local state_latent_top1_end="na"
  local state_latent_sign_top1_end="na"
  local state_latent_mean_rank_end="na"
  local state_projection_mrr_end="na"
  local state_projection_top1_end="na"
  local state_projection_sign_top1_end="na"
  local state_projection_mean_rank_end="na"
  local state_support_samples="na"
  local state_query_samples="na"
  local state_candidates="na"

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
  if [[ "$(signed_margin_enabled)" == "true" ]]; then
    extra_args+=(--signed-margin-weight "$SIGNED_MARGIN_WEIGHT")
    extra_args+=(--signed-margin-bank-gap "$SIGNED_MARGIN_BANK_GAP")
    extra_args+=(--signed-margin-sign-gap "$SIGNED_MARGIN_SIGN_GAP")
    extra_args+=(--signed-margin-speed-gap "$SIGNED_MARGIN_SPEED_GAP")
    extra_args+=(--signed-margin-bank-weight "$SIGNED_MARGIN_BANK_WEIGHT")
    extra_args+=(--signed-margin-sign-weight "$SIGNED_MARGIN_SIGN_WEIGHT")
    extra_args+=(--signed-margin-speed-weight "$SIGNED_MARGIN_SPEED_WEIGHT")
  fi
  if [[ "$(signed_bank_softmax_enabled)" == "true" ]]; then
    extra_args+=(--signed-bank-softmax-weight "$SIGNED_BANK_SOFTMAX_WEIGHT")
    extra_args+=(--signed-bank-softmax-temperature "$SIGNED_BANK_SOFTMAX_TEMPERATURE")
  fi

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
  local final_prediction_bank_margin_line
  local final_signed_objective_error_breakdown_line
  local final_signed_margin_objective_line
  local final_signed_bank_softmax_objective_line
  local final_signed_state_separability_line

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

      final_prediction_bank_margin_line="$(grep -m1 '^final | signed prediction bank margin true_distance ' "$log_file" || true)"

      if ! parsed="$(parse_prediction_bank_margin_line "$final_prediction_bank_margin_line")"; then
        failures=$((failures + 1))
        emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "na" "na" "$proj_var_mean" "$target_drift_end" "parse_failed"
        rm -f "$log_file"
        return 0
      fi
      read -r prediction_bank_true_distance_end prediction_bank_nearest_wrong_distance_end prediction_bank_margin_end prediction_bank_min_margin_end prediction_bank_positive_margin_rate_end prediction_bank_sign_margin_end prediction_bank_speed_margin_end prediction_bank_samples <<< "$parsed"

      final_signed_objective_error_breakdown_line="$(grep -m1 '^final | signed objective error breakdown all_loss ' "$log_file" || true)"

      if ! parsed="$(parse_signed_objective_error_breakdown_line "$final_signed_objective_error_breakdown_line")"; then
        failures=$((failures + 1))
        emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "na" "na" "$proj_var_mean" "$target_drift_end" "parse_failed"
        rm -f "$log_file"
        return 0
      fi
      read -r signed_objective_all_loss_end signed_objective_dx_neg2_loss_end signed_objective_dx_neg1_loss_end signed_objective_dx_pos1_loss_end signed_objective_dx_pos2_loss_end signed_objective_neg_loss_end signed_objective_pos_loss_end signed_objective_slow_loss_end signed_objective_fast_loss_end signed_objective_sign_gap_end signed_objective_speed_gap_end signed_objective_samples signed_objective_dx_neg2_samples signed_objective_dx_neg1_samples signed_objective_dx_pos1_samples signed_objective_dx_pos2_samples <<< "$parsed"

      if [[ "$(signed_margin_enabled)" == "true" ]]; then
        final_signed_margin_objective_line="$(grep -m1 '^final | signed margin objective bank_loss ' "$log_file" || true)"

        if ! parsed="$(parse_signed_margin_objective_line "$final_signed_margin_objective_line")"; then
          failures=$((failures + 1))
          emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "na" "na" "$proj_var_mean" "$target_drift_end" "parse_failed"
          rm -f "$log_file"
          return 0
        fi
        read -r signed_margin_bank_loss_end signed_margin_sign_loss_end signed_margin_speed_loss_end signed_margin_weighted_loss_end signed_margin_active_bank_rate_end signed_margin_active_sign_rate_end signed_margin_active_speed_rate_end signed_margin_samples <<< "$parsed"
      fi

      if [[ "$(signed_bank_softmax_enabled)" == "true" ]]; then
        final_signed_bank_softmax_objective_line="$(grep -m1 '^final | signed bank softmax objective loss ' "$log_file" || true)"

        if ! parsed="$(parse_signed_bank_softmax_objective_line "$final_signed_bank_softmax_objective_line")"; then
          failures=$((failures + 1))
          emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "na" "na" "$proj_var_mean" "$target_drift_end" "parse_failed"
          rm -f "$log_file"
          return 0
        fi
        read -r signed_bank_softmax_loss_end signed_bank_softmax_top1_end signed_bank_softmax_true_probability_end signed_bank_softmax_samples <<< "$parsed"
      fi

      final_signed_state_separability_line="$(grep -m1 '^final | signed state separability latent_mrr ' "$log_file" || true)"

      if ! parsed="$(parse_signed_state_separability_line "$final_signed_state_separability_line")"; then
        failures=$((failures + 1))
        emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "na" "na" "$proj_var_mean" "$target_drift_end" "parse_failed"
        rm -f "$log_file"
        return 0
      fi
      read -r state_latent_mrr_end state_latent_top1_end state_latent_sign_top1_end state_latent_mean_rank_end state_projection_mrr_end state_projection_top1_end state_projection_sign_top1_end state_projection_mean_rank_end state_support_samples state_query_samples state_candidates <<< "$parsed"
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
