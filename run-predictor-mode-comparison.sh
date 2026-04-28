#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_PATH="${JEPRA_MANIFEST_PATH:-$ROOT_DIR/crates/jepra-core/Cargo.toml}"
SCHEMA="jepra_predictor_compare_v19"
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
SIGNED_RADIAL_WEIGHT="${JEPRA_SIGNED_RADIAL_WEIGHT:-0.0}"
SIGNED_ANGULAR_RADIAL_WEIGHT="${JEPRA_SIGNED_ANGULAR_RADIAL_WEIGHT:-0.0}"
SIGNED_ANGULAR_WEIGHT="${JEPRA_SIGNED_ANGULAR_WEIGHT:-1.0}"
SIGNED_ANGULAR_RADIAL_RADIUS_WEIGHT="${JEPRA_SIGNED_ANGULAR_RADIAL_RADIUS_WEIGHT:-1.0}"
SIGNED_CANDIDATE_SELECTOR_HEAD="${JEPRA_SIGNED_CANDIDATE_SELECTOR_HEAD:-0}"
SIGNED_CANDIDATE_SELECTOR_HEAD_TEMPERATURE="${JEPRA_SIGNED_CANDIDATE_SELECTOR_HEAD_TEMPERATURE:-0.05}"
SIGNED_CANDIDATE_SELECTOR_HEAD_LR="${JEPRA_SIGNED_CANDIDATE_SELECTOR_HEAD_LR:-0.05}"
SIGNED_CANDIDATE_SELECTOR_HEAD_WEIGHT="${JEPRA_SIGNED_CANDIDATE_SELECTOR_HEAD_WEIGHT:-1.0}"
SIGNED_CANDIDATE_SELECTOR_HEAD_ENTROPY_FLOOR="${JEPRA_SIGNED_CANDIDATE_SELECTOR_HEAD_ENTROPY_FLOOR:-1.0}"
SIGNED_CANDIDATE_SELECTOR_HEAD_ENTROPY_WEIGHT="${JEPRA_SIGNED_CANDIDATE_SELECTOR_HEAD_ENTROPY_WEIGHT:-0.1}"
SIGNED_CANDIDATE_SELECTOR_HEAD_KL_WEIGHT="${JEPRA_SIGNED_CANDIDATE_SELECTOR_HEAD_KL_WEIGHT:-0.0}"
SIGNED_CANDIDATE_SELECTOR_OUTPUT="${JEPRA_SIGNED_CANDIDATE_SELECTOR_OUTPUT:-off}"
SIGNED_CANDIDATE_SELECTOR_OUTPUT_COUPLING="${JEPRA_SIGNED_CANDIDATE_SELECTOR_OUTPUT_COUPLING:-0}"
SIGNED_CANDIDATE_SELECTOR_OUTPUT_COUPLING_WEIGHT="${JEPRA_SIGNED_CANDIDATE_SELECTOR_OUTPUT_COUPLING_WEIGHT:-1.0}"
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
  JEPRA_PREDICTOR_MODES                         Space-separated modes (default: "baseline bottleneck residual-bottleneck"; also supports state-radius)
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
  JEPRA_COMPACT_ENCODER_MODE                    Optional compact encoder mode: base|stronger|signed-direction|signed-direction-magnitude
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
  JEPRA_SIGNED_RADIAL_WEIGHT                    Signed radial calibration objective weight (default: 0.0)
  JEPRA_SIGNED_ANGULAR_RADIAL_WEIGHT            Signed angular-radial objective weight (default: 0.0)
  JEPRA_SIGNED_ANGULAR_WEIGHT                   Signed angular component weight (default: 1.0)
  JEPRA_SIGNED_ANGULAR_RADIAL_RADIUS_WEIGHT     Signed angular-radial radius component weight (default: 1.0)
  JEPRA_SIGNED_CANDIDATE_SELECTOR_HEAD          Enable signed candidate selector head reporting/training (default: 0)
  JEPRA_SIGNED_CANDIDATE_SELECTOR_HEAD_TEMPERATURE Signed candidate selector head temperature (default: 0.05)
  JEPRA_SIGNED_CANDIDATE_SELECTOR_HEAD_LR       Signed candidate selector head learning rate (default: 0.05)
  JEPRA_SIGNED_CANDIDATE_SELECTOR_HEAD_WEIGHT   Signed candidate selector head loss weight (default: 1.0)
  JEPRA_SIGNED_CANDIDATE_SELECTOR_HEAD_ENTROPY_FLOOR Signed candidate selector head entropy floor (default: 1.0)
  JEPRA_SIGNED_CANDIDATE_SELECTOR_HEAD_ENTROPY_WEIGHT Signed candidate selector head entropy weight (default: 0.1)
  JEPRA_SIGNED_CANDIDATE_SELECTOR_HEAD_KL_WEIGHT Signed candidate selector head KL-to-prior weight (default: 0.0)
  JEPRA_SIGNED_CANDIDATE_SELECTOR_OUTPUT        Selector-to-output mode: off|hard-full (default: off)
  JEPRA_SIGNED_CANDIDATE_SELECTOR_OUTPUT_COUPLING Legacy boolean alias for hard-full output coupling (default: 0)
  JEPRA_SIGNED_CANDIDATE_SELECTOR_OUTPUT_COUPLING_WEIGHT Selector output coupling weight (default: 1.0)
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
  printf 'schema,temporal_task,path,predictor,residual_delta_scale,projector_drift_weight,signed_margin_weight,signed_margin_bank_gap,signed_margin_sign_gap,signed_margin_speed_gap,signed_margin_bank_weight,signed_margin_sign_weight,signed_margin_speed_weight,signed_bank_softmax_weight,signed_bank_softmax_temperature,signed_radial_weight,signed_angular_radial_weight,signed_angular_weight,signed_angular_radial_radius_weight,seed,steps,encoder_mode,encoder_lr,target_momentum_start,target_momentum_end,target_momentum_warmup_steps,train_pred_start,train_pred_end,val_pred_start,val_pred_end,train_obj_start,train_obj_end,val_obj_start,val_obj_end,pred_min_std_final,target_min_std_final,proj_var_mean_final,target_drift_end,velocity_bank_mrr_start,velocity_bank_mrr_end,velocity_bank_top1_start,velocity_bank_top1_end,velocity_bank_mean_rank_start,velocity_bank_mean_rank_end,velocity_bank_samples,velocity_bank_candidates,signed_bank_neg_mrr_end,signed_bank_pos_mrr_end,signed_bank_slow_mrr_end,signed_bank_fast_mrr_end,signed_bank_sign_top1_end,signed_bank_speed_top1_end,signed_bank_samples,signed_bank_true_neg_best_neg,signed_bank_true_neg_best_pos,signed_bank_true_pos_best_neg,signed_bank_true_pos_best_pos,signed_bank_true_slow_best_slow,signed_bank_true_slow_best_fast,signed_bank_true_fast_best_slow,signed_bank_true_fast_best_fast,target_bank_oracle_mrr_end,target_bank_oracle_top1_end,target_bank_true_distance_end,target_bank_true_distance_max_end,target_bank_nearest_wrong_end,target_bank_nearest_wrong_min_end,target_bank_margin_end,target_bank_margin_min_end,target_bank_neg_nearest_wrong_end,target_bank_pos_nearest_wrong_end,target_bank_slow_nearest_wrong_end,target_bank_fast_nearest_wrong_end,target_bank_sign_margin_end,target_bank_speed_margin_end,target_bank_samples,prediction_bank_true_distance_end,prediction_bank_nearest_wrong_distance_end,prediction_bank_margin_end,prediction_bank_min_margin_end,prediction_bank_positive_margin_rate_end,prediction_bank_sign_margin_end,prediction_bank_speed_margin_end,prediction_bank_samples,prediction_unit_mrr_end,prediction_unit_top1_end,prediction_unit_true_distance_end,prediction_unit_nearest_wrong_distance_end,prediction_unit_margin_end,prediction_unit_positive_margin_rate_end,prediction_unit_sign_margin_end,prediction_unit_speed_margin_end,prediction_unit_prediction_center_norm_end,prediction_unit_true_target_center_norm_end,prediction_unit_samples,prediction_unit_candidates,prediction_counterfactual_oracle_radius_mrr_end,prediction_counterfactual_oracle_radius_top1_end,prediction_counterfactual_oracle_radius_margin_end,prediction_counterfactual_oracle_radius_positive_margin_rate_end,prediction_counterfactual_oracle_radius_sign_margin_end,prediction_counterfactual_oracle_radius_speed_margin_end,prediction_counterfactual_oracle_radius_norm_ratio_end,prediction_counterfactual_oracle_angle_mrr_end,prediction_counterfactual_oracle_angle_top1_end,prediction_counterfactual_oracle_angle_margin_end,prediction_counterfactual_oracle_angle_positive_margin_rate_end,prediction_counterfactual_oracle_angle_sign_margin_end,prediction_counterfactual_oracle_angle_speed_margin_end,prediction_counterfactual_oracle_angle_norm_ratio_end,prediction_counterfactual_support_global_rescale_mrr_end,prediction_counterfactual_support_global_rescale_top1_end,prediction_counterfactual_support_global_rescale_margin_end,prediction_counterfactual_support_global_rescale_positive_margin_rate_end,prediction_counterfactual_support_global_rescale_sign_margin_end,prediction_counterfactual_support_global_rescale_speed_margin_end,prediction_counterfactual_support_global_rescale_norm_ratio_end,prediction_counterfactual_support_norm_ratio_end,prediction_counterfactual_support_samples,prediction_counterfactual_query_samples,prediction_counterfactual_candidates,signed_objective_all_loss_end,signed_objective_dx_neg2_loss_end,signed_objective_dx_neg1_loss_end,signed_objective_dx_pos1_loss_end,signed_objective_dx_pos2_loss_end,signed_objective_neg_loss_end,signed_objective_pos_loss_end,signed_objective_slow_loss_end,signed_objective_fast_loss_end,signed_objective_sign_gap_end,signed_objective_speed_gap_end,signed_objective_samples,signed_objective_dx_neg2_samples,signed_objective_dx_neg1_samples,signed_objective_dx_pos1_samples,signed_objective_dx_pos2_samples,signed_margin_bank_loss_end,signed_margin_sign_loss_end,signed_margin_speed_loss_end,signed_margin_weighted_loss_end,signed_margin_active_bank_rate_end,signed_margin_active_sign_rate_end,signed_margin_active_speed_rate_end,signed_margin_samples,signed_bank_softmax_loss_end,signed_bank_softmax_top1_end,signed_bank_softmax_true_probability_end,signed_bank_softmax_samples,signed_radial_loss_end,signed_radial_prediction_norm_end,signed_radial_target_norm_end,signed_radial_norm_ratio_end,signed_radial_samples,signed_angular_radial_loss_end,signed_angular_radial_angular_loss_end,signed_angular_radial_radial_loss_end,signed_angular_radial_cosine_end,signed_angular_radial_prediction_norm_end,signed_angular_radial_target_norm_end,signed_angular_radial_norm_ratio_end,signed_angular_radial_samples,state_latent_mrr_end,state_latent_top1_end,state_latent_sign_top1_end,state_latent_mean_rank_end,state_projection_mrr_end,state_projection_top1_end,state_projection_sign_top1_end,state_projection_mean_rank_end,state_support_samples,state_query_samples,state_candidates,selector_head_enabled,selector_head_temperature,selector_head_lr,selector_head_weight,selector_head_entropy_floor,selector_head_entropy_weight,selector_head_kl_weight,selector_head_mrr_end,selector_head_top1_end,selector_head_margin_end,selector_head_positive_margin_rate_end,selector_head_sign_margin_end,selector_head_speed_margin_end,selector_head_norm_ratio_end,selector_head_objective_loss_end,selector_head_objective_ce_end,selector_head_objective_entropy_reg_end,selector_head_objective_kl_to_prior_end,selector_head_entropy_end,selector_head_true_probability_end,selector_head_max_probability_end,selector_head_temperature_end,selector_head_steps,selector_head_lr_end,selector_head_entropy_floor_end,selector_head_entropy_weight_end,selector_head_kl_weight_end,selector_head_support_samples,selector_head_query_samples,selector_head_candidates,selector_readout_base_prediction_mrr_end,selector_readout_base_prediction_top1_end,selector_readout_base_prediction_margin_end,selector_readout_base_prediction_positive_margin_rate_end,selector_readout_base_prediction_sign_margin_end,selector_readout_base_prediction_speed_margin_end,selector_readout_base_prediction_norm_ratio_end,selector_readout_soft_unit_mix_mrr_end,selector_readout_soft_unit_mix_top1_end,selector_readout_soft_unit_mix_margin_end,selector_readout_soft_unit_mix_positive_margin_rate_end,selector_readout_soft_unit_mix_sign_margin_end,selector_readout_soft_unit_mix_speed_margin_end,selector_readout_soft_unit_mix_norm_ratio_end,selector_readout_soft_full_mrr_end,selector_readout_soft_full_top1_end,selector_readout_soft_full_margin_end,selector_readout_soft_full_positive_margin_rate_end,selector_readout_soft_full_sign_margin_end,selector_readout_soft_full_speed_margin_end,selector_readout_soft_full_norm_ratio_end,selector_readout_hard_full_mrr_end,selector_readout_hard_full_top1_end,selector_readout_hard_full_margin_end,selector_readout_hard_full_positive_margin_rate_end,selector_readout_hard_full_sign_margin_end,selector_readout_hard_full_speed_margin_end,selector_readout_hard_full_norm_ratio_end,selector_readout_soft_radius_mrr_end,selector_readout_soft_radius_top1_end,selector_readout_soft_radius_margin_end,selector_readout_soft_radius_positive_margin_rate_end,selector_readout_soft_radius_sign_margin_end,selector_readout_soft_radius_speed_margin_end,selector_readout_soft_radius_norm_ratio_end,selector_readout_hard_radius_mrr_end,selector_readout_hard_radius_top1_end,selector_readout_hard_radius_margin_end,selector_readout_hard_radius_positive_margin_rate_end,selector_readout_hard_radius_sign_margin_end,selector_readout_hard_radius_speed_margin_end,selector_readout_hard_radius_norm_ratio_end,selector_output_mode,selector_output_coupling_enabled,selector_output_coupling_weight,selector_output_coupling_loss_end,selector_output_coupling_selector_max_probability_end,selector_output_coupling_base_prediction_top1_end,selector_output_coupling_base_prediction_margin_end,selector_output_coupling_base_prediction_norm_ratio_end,selector_output_coupling_hard_full_top1_end,selector_output_coupling_hard_full_margin_end,selector_output_coupling_hard_full_norm_ratio_end,selector_output_coupling_samples,selector_output_coupling_candidates,status\n' > "$REPORT_PATH"
fi

encoder_mode_label() {
  case "$COMPACT_ENCODER_MODE" in
    "") printf '%s' "frozen-base" ;;
    "base") printf '%s' "compact-base" ;;
    "stronger") printf '%s' "compact-stronger" ;;
    "signed-direction") printf '%s' "compact-signed-direction" ;;
    "signed-direction-magnitude") printf '%s' "compact-signed-direction-magnitude" ;;
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

signed_radial_enabled() {
  awk -v value="$SIGNED_RADIAL_WEIGHT" 'BEGIN { if (value + 0 > 0) print "true"; else print "false" }'
}

signed_angular_radial_enabled() {
  awk -v value="$SIGNED_ANGULAR_RADIAL_WEIGHT" 'BEGIN { if (value + 0 > 0) print "true"; else print "false" }'
}

signed_candidate_selector_head_enabled() {
  case "$SIGNED_CANDIDATE_SELECTOR_HEAD" in
    1|true|TRUE|yes|YES|on|ON) printf '%s' "true" ;;
    *) printf '%s' "false" ;;
  esac
}

signed_candidate_selector_output_coupling_alias_enabled() {
  case "$SIGNED_CANDIDATE_SELECTOR_OUTPUT_COUPLING" in
    1|true|TRUE|yes|YES|on|ON) printf '%s' "true" ;;
    *) printf '%s' "false" ;;
  esac
}

signed_candidate_selector_output_mode() {
  local mode
  mode="$(tr '[:upper:]' '[:lower:]' <<< "$SIGNED_CANDIDATE_SELECTOR_OUTPUT")"
  case "$mode" in
    hard-full) printf '%s' "hard-full" ;;
    off)
      if [[ "$(signed_candidate_selector_output_coupling_alias_enabled)" == "true" ]]; then
        printf '%s' "hard-full"
      else
        printf '%s' "off"
      fi
      ;;
    *)
      echo "Unsupported JEPRA_SIGNED_CANDIDATE_SELECTOR_OUTPUT: $SIGNED_CANDIDATE_SELECTOR_OUTPUT (expected off|hard-full)" >&2
      exit 2
      ;;
  esac
}

signed_candidate_selector_output_coupling_enabled() {
  if [[ "$(signed_candidate_selector_output_mode)" == "hard-full" ]]; then
    printf '%s' "true"
    return
  fi
  printf '%s' "false"
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

parse_prediction_bank_unit_geometry_line() {
  local line="$1"
  if [[ ! "$line" =~ ^(initial|final)[[:space:]]+\|[[:space:]]+signed[[:space:]]+prediction[[:space:]]+bank[[:space:]]+unit[[:space:]]+geometry[[:space:]]+mrr[[:space:]] ]]; then
    return 1
  fi
  awk '
    {
      for (i = 1; i <= NF; i++) {
        values[$i] = $(i + 1)
      }
      print values["mrr"], values["top1"], values["true_distance"], values["nearest_wrong_distance"], values["margin"], values["positive_margin_rate"], values["sign_margin"], values["speed_margin"], values["prediction_center_norm"], values["true_target_center_norm"], values["samples"], values["candidates"]
    }
  ' <<< "$line"
}

parse_prediction_geometry_counterfactual_line() {
  local line="$1"
  if [[ ! "$line" =~ ^(initial|final)[[:space:]]+\|[[:space:]]+signed[[:space:]]+prediction[[:space:]]+geometry[[:space:]]+counterfactual[[:space:]]+oracle_radius_mrr[[:space:]] ]]; then
    return 1
  fi
  awk '
    {
      for (i = 1; i <= NF; i++) {
        values[$i] = $(i + 1)
      }
      print values["oracle_radius_mrr"], values["oracle_radius_top1"], values["oracle_radius_margin"], values["oracle_radius_positive_margin_rate"], values["oracle_radius_sign_margin"], values["oracle_radius_speed_margin"], values["oracle_radius_norm_ratio"], values["oracle_angle_mrr"], values["oracle_angle_top1"], values["oracle_angle_margin"], values["oracle_angle_positive_margin_rate"], values["oracle_angle_sign_margin"], values["oracle_angle_speed_margin"], values["oracle_angle_norm_ratio"], values["support_global_rescale_mrr"], values["support_global_rescale_top1"], values["support_global_rescale_margin"], values["support_global_rescale_positive_margin_rate"], values["support_global_rescale_sign_margin"], values["support_global_rescale_speed_margin"], values["support_global_rescale_norm_ratio"], values["support_norm_ratio"], values["support_samples"], values["query_samples"], values["candidates"]
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

parse_signed_radial_calibration_line() {
  local line="$1"
  if [[ ! "$line" =~ ^(initial|final)[[:space:]]+\|[[:space:]]+signed[[:space:]]+radial[[:space:]]+calibration[[:space:]]+loss[[:space:]] ]]; then
    return 1
  fi
  awk '
    {
      for (i = 1; i <= NF; i++) {
        values[$i] = $(i + 1)
      }
      print values["loss"], values["prediction_norm"], values["target_norm"], values["norm_ratio"], values["samples"]
    }
  ' <<< "$line"
}

parse_signed_angular_radial_objective_line() {
  local line="$1"
  if [[ ! "$line" =~ ^(initial|final)[[:space:]]+\|[[:space:]]+signed[[:space:]]+angular-radial[[:space:]]+objective[[:space:]]+loss[[:space:]] ]]; then
    return 1
  fi
  awk '
    {
      for (i = 1; i <= NF; i++) {
        values[$i] = $(i + 1)
      }
      print values["loss"], values["angular_loss"], values["radial_loss"], values["cosine"], values["prediction_norm"], values["target_norm"], values["norm_ratio"], values["samples"]
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

parse_signed_candidate_selector_head_integration_line() {
  local line="$1"
  if [[ ! "$line" =~ ^final[[:space:]]+\|[[:space:]]+signed[[:space:]]+candidate[[:space:]]+selector[[:space:]]+head[[:space:]]+integration[[:space:]]+learned_selector_mrr[[:space:]] ]]; then
    return 1
  fi
  awk '
    {
      for (i = 1; i <= NF; i++) {
        values[$i] = $(i + 1)
      }
      print values["learned_selector_mrr"], values["learned_selector_top1"], values["learned_selector_margin"], values["learned_selector_positive_margin_rate"], values["learned_selector_sign_margin"], values["learned_selector_speed_margin"], values["learned_selector_norm_ratio"], values["objective_loss"], values["objective_ce"], values["objective_entropy_reg"], values["objective_kl_to_prior"], values["objective_entropy"], values["objective_true_probability"], values["objective_max_probability"], values["softmax_temperature"], values["selector_steps"], values["lr"], values["entropy_floor"], values["entropy_weight"], values["kl_weight"], values["support_samples"], values["query_samples"], values["candidates"]
    }
  ' <<< "$line"
}

parse_signed_candidate_selector_readout_line() {
  local line="$1"
  local field="$2"
  if [[ ! "$line" =~ ^final[[:space:]]+\|[[:space:]]+signed[[:space:]]+candidate[[:space:]]+selector[[:space:]]+readout[[:space:]]+${field}_mrr[[:space:]] ]]; then
    return 1
  fi
  awk -v prefix="$field" '
    {
      for (i = 1; i <= NF; i++) {
        values[$i] = $(i + 1)
      }
      print values[prefix "_mrr"], values[prefix "_top1"], values[prefix "_margin"], values[prefix "_positive_margin_rate"], values[prefix "_sign_margin"], values[prefix "_speed_margin"], values[prefix "_norm_ratio"]
    }
  ' <<< "$line"
}

parse_signed_candidate_selector_output_coupling_line() {
  local line="$1"
  if [[ ! "$line" =~ ^final[[:space:]]+\|[[:space:]]+signed[[:space:]]+candidate[[:space:]]+selector[[:space:]]+output[[:space:]]+coupling[[:space:]]+loss[[:space:]] ]]; then
    return 1
  fi
  awk '
    {
      for (i = 1; i <= NF; i++) {
        values[$i] = $(i + 1)
      }
      print values["loss"], values["selector_max_probability"], values["base_prediction_top1"], values["base_prediction_margin"], values["base_prediction_norm_ratio"], values["selector_hard_full_top1"], values["selector_hard_full_margin"], values["selector_hard_full_norm_ratio"], values["samples"], values["candidates"]
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
  local prediction_unit_mrr_end_value="${prediction_unit_mrr_end:-na}"
  local prediction_unit_top1_end_value="${prediction_unit_top1_end:-na}"
  local prediction_unit_true_distance_end_value="${prediction_unit_true_distance_end:-na}"
  local prediction_unit_nearest_wrong_distance_end_value="${prediction_unit_nearest_wrong_distance_end:-na}"
  local prediction_unit_margin_end_value="${prediction_unit_margin_end:-na}"
  local prediction_unit_positive_margin_rate_end_value="${prediction_unit_positive_margin_rate_end:-na}"
  local prediction_unit_sign_margin_end_value="${prediction_unit_sign_margin_end:-na}"
  local prediction_unit_speed_margin_end_value="${prediction_unit_speed_margin_end:-na}"
  local prediction_unit_prediction_center_norm_end_value="${prediction_unit_prediction_center_norm_end:-na}"
  local prediction_unit_true_target_center_norm_end_value="${prediction_unit_true_target_center_norm_end:-na}"
  local prediction_unit_samples_value="${prediction_unit_samples:-na}"
  local prediction_unit_candidates_value="${prediction_unit_candidates:-na}"
  local prediction_counterfactual_oracle_radius_mrr_end_value="${prediction_counterfactual_oracle_radius_mrr_end:-na}"
  local prediction_counterfactual_oracle_radius_top1_end_value="${prediction_counterfactual_oracle_radius_top1_end:-na}"
  local prediction_counterfactual_oracle_radius_margin_end_value="${prediction_counterfactual_oracle_radius_margin_end:-na}"
  local prediction_counterfactual_oracle_radius_positive_margin_rate_end_value="${prediction_counterfactual_oracle_radius_positive_margin_rate_end:-na}"
  local prediction_counterfactual_oracle_radius_sign_margin_end_value="${prediction_counterfactual_oracle_radius_sign_margin_end:-na}"
  local prediction_counterfactual_oracle_radius_speed_margin_end_value="${prediction_counterfactual_oracle_radius_speed_margin_end:-na}"
  local prediction_counterfactual_oracle_radius_norm_ratio_end_value="${prediction_counterfactual_oracle_radius_norm_ratio_end:-na}"
  local prediction_counterfactual_oracle_angle_mrr_end_value="${prediction_counterfactual_oracle_angle_mrr_end:-na}"
  local prediction_counterfactual_oracle_angle_top1_end_value="${prediction_counterfactual_oracle_angle_top1_end:-na}"
  local prediction_counterfactual_oracle_angle_margin_end_value="${prediction_counterfactual_oracle_angle_margin_end:-na}"
  local prediction_counterfactual_oracle_angle_positive_margin_rate_end_value="${prediction_counterfactual_oracle_angle_positive_margin_rate_end:-na}"
  local prediction_counterfactual_oracle_angle_sign_margin_end_value="${prediction_counterfactual_oracle_angle_sign_margin_end:-na}"
  local prediction_counterfactual_oracle_angle_speed_margin_end_value="${prediction_counterfactual_oracle_angle_speed_margin_end:-na}"
  local prediction_counterfactual_oracle_angle_norm_ratio_end_value="${prediction_counterfactual_oracle_angle_norm_ratio_end:-na}"
  local prediction_counterfactual_support_global_rescale_mrr_end_value="${prediction_counterfactual_support_global_rescale_mrr_end:-na}"
  local prediction_counterfactual_support_global_rescale_top1_end_value="${prediction_counterfactual_support_global_rescale_top1_end:-na}"
  local prediction_counterfactual_support_global_rescale_margin_end_value="${prediction_counterfactual_support_global_rescale_margin_end:-na}"
  local prediction_counterfactual_support_global_rescale_positive_margin_rate_end_value="${prediction_counterfactual_support_global_rescale_positive_margin_rate_end:-na}"
  local prediction_counterfactual_support_global_rescale_sign_margin_end_value="${prediction_counterfactual_support_global_rescale_sign_margin_end:-na}"
  local prediction_counterfactual_support_global_rescale_speed_margin_end_value="${prediction_counterfactual_support_global_rescale_speed_margin_end:-na}"
  local prediction_counterfactual_support_global_rescale_norm_ratio_end_value="${prediction_counterfactual_support_global_rescale_norm_ratio_end:-na}"
  local prediction_counterfactual_support_norm_ratio_end_value="${prediction_counterfactual_support_norm_ratio_end:-na}"
  local prediction_counterfactual_support_samples_value="${prediction_counterfactual_support_samples:-na}"
  local prediction_counterfactual_query_samples_value="${prediction_counterfactual_query_samples:-na}"
  local prediction_counterfactual_candidates_value="${prediction_counterfactual_candidates:-na}"
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
  local signed_radial_loss_end_value="${signed_radial_loss_end:-na}"
  local signed_radial_prediction_norm_end_value="${signed_radial_prediction_norm_end:-na}"
  local signed_radial_target_norm_end_value="${signed_radial_target_norm_end:-na}"
  local signed_radial_norm_ratio_end_value="${signed_radial_norm_ratio_end:-na}"
  local signed_radial_samples_value="${signed_radial_samples:-na}"
  local signed_angular_radial_loss_end_value="${signed_angular_radial_loss_end:-na}"
  local signed_angular_radial_angular_loss_end_value="${signed_angular_radial_angular_loss_end:-na}"
  local signed_angular_radial_radial_loss_end_value="${signed_angular_radial_radial_loss_end:-na}"
  local signed_angular_radial_cosine_end_value="${signed_angular_radial_cosine_end:-na}"
  local signed_angular_radial_prediction_norm_end_value="${signed_angular_radial_prediction_norm_end:-na}"
  local signed_angular_radial_target_norm_end_value="${signed_angular_radial_target_norm_end:-na}"
  local signed_angular_radial_norm_ratio_end_value="${signed_angular_radial_norm_ratio_end:-na}"
  local signed_angular_radial_samples_value="${signed_angular_radial_samples:-na}"
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
  local selector_head_mrr_end_value="${selector_head_mrr_end:-na}"
  local selector_head_top1_end_value="${selector_head_top1_end:-na}"
  local selector_head_margin_end_value="${selector_head_margin_end:-na}"
  local selector_head_positive_margin_rate_end_value="${selector_head_positive_margin_rate_end:-na}"
  local selector_head_sign_margin_end_value="${selector_head_sign_margin_end:-na}"
  local selector_head_speed_margin_end_value="${selector_head_speed_margin_end:-na}"
  local selector_head_norm_ratio_end_value="${selector_head_norm_ratio_end:-na}"
  local selector_head_objective_loss_end_value="${selector_head_objective_loss_end:-na}"
  local selector_head_objective_ce_end_value="${selector_head_objective_ce_end:-na}"
  local selector_head_objective_entropy_reg_end_value="${selector_head_objective_entropy_reg_end:-na}"
  local selector_head_objective_kl_to_prior_end_value="${selector_head_objective_kl_to_prior_end:-na}"
  local selector_head_entropy_end_value="${selector_head_entropy_end:-na}"
  local selector_head_true_probability_end_value="${selector_head_true_probability_end:-na}"
  local selector_head_max_probability_end_value="${selector_head_max_probability_end:-na}"
  local selector_head_temperature_end_value="${selector_head_temperature_end:-na}"
  local selector_head_steps_value="${selector_head_steps:-na}"
  local selector_head_lr_end_value="${selector_head_lr_end:-na}"
  local selector_head_entropy_floor_end_value="${selector_head_entropy_floor_end:-na}"
  local selector_head_entropy_weight_end_value="${selector_head_entropy_weight_end:-na}"
  local selector_head_kl_weight_end_value="${selector_head_kl_weight_end:-na}"
  local selector_head_support_samples_value="${selector_head_support_samples:-na}"
  local selector_head_query_samples_value="${selector_head_query_samples:-na}"
  local selector_head_candidates_value="${selector_head_candidates:-na}"
  local selector_readout_base_prediction_mrr_end_value="${selector_readout_base_prediction_mrr_end:-na}"
  local selector_readout_base_prediction_top1_end_value="${selector_readout_base_prediction_top1_end:-na}"
  local selector_readout_base_prediction_margin_end_value="${selector_readout_base_prediction_margin_end:-na}"
  local selector_readout_base_prediction_positive_margin_rate_end_value="${selector_readout_base_prediction_positive_margin_rate_end:-na}"
  local selector_readout_base_prediction_sign_margin_end_value="${selector_readout_base_prediction_sign_margin_end:-na}"
  local selector_readout_base_prediction_speed_margin_end_value="${selector_readout_base_prediction_speed_margin_end:-na}"
  local selector_readout_base_prediction_norm_ratio_end_value="${selector_readout_base_prediction_norm_ratio_end:-na}"
  local selector_readout_soft_unit_mix_mrr_end_value="${selector_readout_soft_unit_mix_mrr_end:-na}"
  local selector_readout_soft_unit_mix_top1_end_value="${selector_readout_soft_unit_mix_top1_end:-na}"
  local selector_readout_soft_unit_mix_margin_end_value="${selector_readout_soft_unit_mix_margin_end:-na}"
  local selector_readout_soft_unit_mix_positive_margin_rate_end_value="${selector_readout_soft_unit_mix_positive_margin_rate_end:-na}"
  local selector_readout_soft_unit_mix_sign_margin_end_value="${selector_readout_soft_unit_mix_sign_margin_end:-na}"
  local selector_readout_soft_unit_mix_speed_margin_end_value="${selector_readout_soft_unit_mix_speed_margin_end:-na}"
  local selector_readout_soft_unit_mix_norm_ratio_end_value="${selector_readout_soft_unit_mix_norm_ratio_end:-na}"
  local selector_readout_soft_full_mrr_end_value="${selector_readout_soft_full_mrr_end:-na}"
  local selector_readout_soft_full_top1_end_value="${selector_readout_soft_full_top1_end:-na}"
  local selector_readout_soft_full_margin_end_value="${selector_readout_soft_full_margin_end:-na}"
  local selector_readout_soft_full_positive_margin_rate_end_value="${selector_readout_soft_full_positive_margin_rate_end:-na}"
  local selector_readout_soft_full_sign_margin_end_value="${selector_readout_soft_full_sign_margin_end:-na}"
  local selector_readout_soft_full_speed_margin_end_value="${selector_readout_soft_full_speed_margin_end:-na}"
  local selector_readout_soft_full_norm_ratio_end_value="${selector_readout_soft_full_norm_ratio_end:-na}"
  local selector_readout_hard_full_mrr_end_value="${selector_readout_hard_full_mrr_end:-na}"
  local selector_readout_hard_full_top1_end_value="${selector_readout_hard_full_top1_end:-na}"
  local selector_readout_hard_full_margin_end_value="${selector_readout_hard_full_margin_end:-na}"
  local selector_readout_hard_full_positive_margin_rate_end_value="${selector_readout_hard_full_positive_margin_rate_end:-na}"
  local selector_readout_hard_full_sign_margin_end_value="${selector_readout_hard_full_sign_margin_end:-na}"
  local selector_readout_hard_full_speed_margin_end_value="${selector_readout_hard_full_speed_margin_end:-na}"
  local selector_readout_hard_full_norm_ratio_end_value="${selector_readout_hard_full_norm_ratio_end:-na}"
  local selector_readout_soft_radius_mrr_end_value="${selector_readout_soft_radius_mrr_end:-na}"
  local selector_readout_soft_radius_top1_end_value="${selector_readout_soft_radius_top1_end:-na}"
  local selector_readout_soft_radius_margin_end_value="${selector_readout_soft_radius_margin_end:-na}"
  local selector_readout_soft_radius_positive_margin_rate_end_value="${selector_readout_soft_radius_positive_margin_rate_end:-na}"
  local selector_readout_soft_radius_sign_margin_end_value="${selector_readout_soft_radius_sign_margin_end:-na}"
  local selector_readout_soft_radius_speed_margin_end_value="${selector_readout_soft_radius_speed_margin_end:-na}"
  local selector_readout_soft_radius_norm_ratio_end_value="${selector_readout_soft_radius_norm_ratio_end:-na}"
  local selector_readout_hard_radius_mrr_end_value="${selector_readout_hard_radius_mrr_end:-na}"
  local selector_readout_hard_radius_top1_end_value="${selector_readout_hard_radius_top1_end:-na}"
  local selector_readout_hard_radius_margin_end_value="${selector_readout_hard_radius_margin_end:-na}"
  local selector_readout_hard_radius_positive_margin_rate_end_value="${selector_readout_hard_radius_positive_margin_rate_end:-na}"
  local selector_readout_hard_radius_sign_margin_end_value="${selector_readout_hard_radius_sign_margin_end:-na}"
  local selector_readout_hard_radius_speed_margin_end_value="${selector_readout_hard_radius_speed_margin_end:-na}"
  local selector_readout_hard_radius_norm_ratio_end_value="${selector_readout_hard_radius_norm_ratio_end:-na}"
  local selector_output_coupling_loss_end_value="${selector_output_coupling_loss_end:-na}"
  local selector_output_coupling_selector_max_probability_end_value="${selector_output_coupling_selector_max_probability_end:-na}"
  local selector_output_coupling_base_prediction_top1_end_value="${selector_output_coupling_base_prediction_top1_end:-na}"
  local selector_output_coupling_base_prediction_margin_end_value="${selector_output_coupling_base_prediction_margin_end:-na}"
  local selector_output_coupling_base_prediction_norm_ratio_end_value="${selector_output_coupling_base_prediction_norm_ratio_end:-na}"
  local selector_output_coupling_hard_full_top1_end_value="${selector_output_coupling_hard_full_top1_end:-na}"
  local selector_output_coupling_hard_full_margin_end_value="${selector_output_coupling_hard_full_margin_end:-na}"
  local selector_output_coupling_hard_full_norm_ratio_end_value="${selector_output_coupling_hard_full_norm_ratio_end:-na}"
  local selector_output_coupling_samples_value="${selector_output_coupling_samples:-na}"
  local selector_output_coupling_candidates_value="${selector_output_coupling_candidates:-na}"

  printf 'schema=%s temporal_task=%s path=%s predictor=%s residual_delta_scale=%s projector_drift_weight=%s signed_margin_weight=%s signed_margin_bank_gap=%s signed_margin_sign_gap=%s signed_margin_speed_gap=%s signed_margin_bank_weight=%s signed_margin_sign_weight=%s signed_margin_speed_weight=%s signed_bank_softmax_weight=%s signed_bank_softmax_temperature=%s signed_radial_weight=%s signed_angular_radial_weight=%s signed_angular_weight=%s signed_angular_radial_radius_weight=%s seed=%s steps=%s encoder_mode=%s encoder_lr=%s target_momentum_start=%s target_momentum_end=%s target_momentum_warmup_steps=%s train_pred_start=%s train_pred_end=%s val_pred_start=%s val_pred_end=%s train_obj_start=%s train_obj_end=%s val_obj_start=%s val_obj_end=%s pred_min_std_final=%s target_min_std_final=%s proj_var_mean_final=%s target_drift_end=%s velocity_bank_mrr_start=%s velocity_bank_mrr_end=%s velocity_bank_top1_start=%s velocity_bank_top1_end=%s velocity_bank_mean_rank_start=%s velocity_bank_mean_rank_end=%s velocity_bank_samples=%s velocity_bank_candidates=%s signed_bank_neg_mrr_end=%s signed_bank_pos_mrr_end=%s signed_bank_slow_mrr_end=%s signed_bank_fast_mrr_end=%s signed_bank_sign_top1_end=%s signed_bank_speed_top1_end=%s signed_bank_samples=%s signed_bank_true_neg_best_neg=%s signed_bank_true_neg_best_pos=%s signed_bank_true_pos_best_neg=%s signed_bank_true_pos_best_pos=%s signed_bank_true_slow_best_slow=%s signed_bank_true_slow_best_fast=%s signed_bank_true_fast_best_slow=%s signed_bank_true_fast_best_fast=%s target_bank_oracle_mrr_end=%s target_bank_oracle_top1_end=%s target_bank_true_distance_end=%s target_bank_true_distance_max_end=%s target_bank_nearest_wrong_end=%s target_bank_nearest_wrong_min_end=%s target_bank_margin_end=%s target_bank_margin_min_end=%s target_bank_neg_nearest_wrong_end=%s target_bank_pos_nearest_wrong_end=%s target_bank_slow_nearest_wrong_end=%s target_bank_fast_nearest_wrong_end=%s target_bank_sign_margin_end=%s target_bank_speed_margin_end=%s target_bank_samples=%s prediction_bank_true_distance_end=%s prediction_bank_nearest_wrong_distance_end=%s prediction_bank_margin_end=%s prediction_bank_min_margin_end=%s prediction_bank_positive_margin_rate_end=%s prediction_bank_sign_margin_end=%s prediction_bank_speed_margin_end=%s prediction_bank_samples=%s prediction_unit_mrr_end=%s prediction_unit_top1_end=%s prediction_unit_true_distance_end=%s prediction_unit_nearest_wrong_distance_end=%s prediction_unit_margin_end=%s prediction_unit_positive_margin_rate_end=%s prediction_unit_sign_margin_end=%s prediction_unit_speed_margin_end=%s prediction_unit_prediction_center_norm_end=%s prediction_unit_true_target_center_norm_end=%s prediction_unit_samples=%s prediction_unit_candidates=%s prediction_counterfactual_oracle_radius_mrr_end=%s prediction_counterfactual_oracle_radius_top1_end=%s prediction_counterfactual_oracle_radius_margin_end=%s prediction_counterfactual_oracle_radius_positive_margin_rate_end=%s prediction_counterfactual_oracle_radius_sign_margin_end=%s prediction_counterfactual_oracle_radius_speed_margin_end=%s prediction_counterfactual_oracle_radius_norm_ratio_end=%s prediction_counterfactual_oracle_angle_mrr_end=%s prediction_counterfactual_oracle_angle_top1_end=%s prediction_counterfactual_oracle_angle_margin_end=%s prediction_counterfactual_oracle_angle_positive_margin_rate_end=%s prediction_counterfactual_oracle_angle_sign_margin_end=%s prediction_counterfactual_oracle_angle_speed_margin_end=%s prediction_counterfactual_oracle_angle_norm_ratio_end=%s prediction_counterfactual_support_global_rescale_mrr_end=%s prediction_counterfactual_support_global_rescale_top1_end=%s prediction_counterfactual_support_global_rescale_margin_end=%s prediction_counterfactual_support_global_rescale_positive_margin_rate_end=%s prediction_counterfactual_support_global_rescale_sign_margin_end=%s prediction_counterfactual_support_global_rescale_speed_margin_end=%s prediction_counterfactual_support_global_rescale_norm_ratio_end=%s prediction_counterfactual_support_norm_ratio_end=%s prediction_counterfactual_support_samples=%s prediction_counterfactual_query_samples=%s prediction_counterfactual_candidates=%s signed_objective_all_loss_end=%s signed_objective_dx_neg2_loss_end=%s signed_objective_dx_neg1_loss_end=%s signed_objective_dx_pos1_loss_end=%s signed_objective_dx_pos2_loss_end=%s signed_objective_neg_loss_end=%s signed_objective_pos_loss_end=%s signed_objective_slow_loss_end=%s signed_objective_fast_loss_end=%s signed_objective_sign_gap_end=%s signed_objective_speed_gap_end=%s signed_objective_samples=%s signed_objective_dx_neg2_samples=%s signed_objective_dx_neg1_samples=%s signed_objective_dx_pos1_samples=%s signed_objective_dx_pos2_samples=%s signed_margin_bank_loss_end=%s signed_margin_sign_loss_end=%s signed_margin_speed_loss_end=%s signed_margin_weighted_loss_end=%s signed_margin_active_bank_rate_end=%s signed_margin_active_sign_rate_end=%s signed_margin_active_speed_rate_end=%s signed_margin_samples=%s signed_bank_softmax_loss_end=%s signed_bank_softmax_top1_end=%s signed_bank_softmax_true_probability_end=%s signed_bank_softmax_samples=%s signed_radial_loss_end=%s signed_radial_prediction_norm_end=%s signed_radial_target_norm_end=%s signed_radial_norm_ratio_end=%s signed_radial_samples=%s signed_angular_radial_loss_end=%s signed_angular_radial_angular_loss_end=%s signed_angular_radial_radial_loss_end=%s signed_angular_radial_cosine_end=%s signed_angular_radial_prediction_norm_end=%s signed_angular_radial_target_norm_end=%s signed_angular_radial_norm_ratio_end=%s signed_angular_radial_samples=%s state_latent_mrr_end=%s state_latent_top1_end=%s state_latent_sign_top1_end=%s state_latent_mean_rank_end=%s state_projection_mrr_end=%s state_projection_top1_end=%s state_projection_sign_top1_end=%s state_projection_mean_rank_end=%s state_support_samples=%s state_query_samples=%s state_candidates=%s selector_head_enabled=%s selector_head_temperature=%s selector_head_lr=%s selector_head_weight=%s selector_head_entropy_floor=%s selector_head_entropy_weight=%s selector_head_kl_weight=%s selector_head_mrr_end=%s selector_head_top1_end=%s selector_head_margin_end=%s selector_head_positive_margin_rate_end=%s selector_head_sign_margin_end=%s selector_head_speed_margin_end=%s selector_head_norm_ratio_end=%s selector_head_objective_loss_end=%s selector_head_objective_ce_end=%s selector_head_objective_entropy_reg_end=%s selector_head_objective_kl_to_prior_end=%s selector_head_entropy_end=%s selector_head_true_probability_end=%s selector_head_max_probability_end=%s selector_head_temperature_end=%s selector_head_steps=%s selector_head_lr_end=%s selector_head_entropy_floor_end=%s selector_head_entropy_weight_end=%s selector_head_kl_weight_end=%s selector_head_support_samples=%s selector_head_query_samples=%s selector_head_candidates=%s status=%s\n' \
    "$SCHEMA" "$TEMPORAL_TASK" "$path" "$predictor" "$RESIDUAL_DELTA_SCALE" "$PROJECTOR_DRIFT_WEIGHT" \
    "$SIGNED_MARGIN_WEIGHT" "$SIGNED_MARGIN_BANK_GAP" "$SIGNED_MARGIN_SIGN_GAP" "$SIGNED_MARGIN_SPEED_GAP" \
    "$SIGNED_MARGIN_BANK_WEIGHT" "$SIGNED_MARGIN_SIGN_WEIGHT" "$SIGNED_MARGIN_SPEED_WEIGHT" \
    "$SIGNED_BANK_SOFTMAX_WEIGHT" "$SIGNED_BANK_SOFTMAX_TEMPERATURE" "$SIGNED_RADIAL_WEIGHT" \
    "$SIGNED_ANGULAR_RADIAL_WEIGHT" "$SIGNED_ANGULAR_WEIGHT" "$SIGNED_ANGULAR_RADIAL_RADIUS_WEIGHT" \
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
    "$prediction_unit_mrr_end_value" "$prediction_unit_top1_end_value" "$prediction_unit_true_distance_end_value" \
    "$prediction_unit_nearest_wrong_distance_end_value" "$prediction_unit_margin_end_value" "$prediction_unit_positive_margin_rate_end_value" \
    "$prediction_unit_sign_margin_end_value" "$prediction_unit_speed_margin_end_value" "$prediction_unit_prediction_center_norm_end_value" \
    "$prediction_unit_true_target_center_norm_end_value" "$prediction_unit_samples_value" "$prediction_unit_candidates_value" \
    "$prediction_counterfactual_oracle_radius_mrr_end_value" "$prediction_counterfactual_oracle_radius_top1_end_value" \
    "$prediction_counterfactual_oracle_radius_margin_end_value" "$prediction_counterfactual_oracle_radius_positive_margin_rate_end_value" \
    "$prediction_counterfactual_oracle_radius_sign_margin_end_value" "$prediction_counterfactual_oracle_radius_speed_margin_end_value" \
    "$prediction_counterfactual_oracle_radius_norm_ratio_end_value" "$prediction_counterfactual_oracle_angle_mrr_end_value" \
    "$prediction_counterfactual_oracle_angle_top1_end_value" "$prediction_counterfactual_oracle_angle_margin_end_value" \
    "$prediction_counterfactual_oracle_angle_positive_margin_rate_end_value" "$prediction_counterfactual_oracle_angle_sign_margin_end_value" \
    "$prediction_counterfactual_oracle_angle_speed_margin_end_value" "$prediction_counterfactual_oracle_angle_norm_ratio_end_value" \
    "$prediction_counterfactual_support_global_rescale_mrr_end_value" "$prediction_counterfactual_support_global_rescale_top1_end_value" \
    "$prediction_counterfactual_support_global_rescale_margin_end_value" "$prediction_counterfactual_support_global_rescale_positive_margin_rate_end_value" \
    "$prediction_counterfactual_support_global_rescale_sign_margin_end_value" "$prediction_counterfactual_support_global_rescale_speed_margin_end_value" \
    "$prediction_counterfactual_support_global_rescale_norm_ratio_end_value" "$prediction_counterfactual_support_norm_ratio_end_value" \
    "$prediction_counterfactual_support_samples_value" "$prediction_counterfactual_query_samples_value" "$prediction_counterfactual_candidates_value" \
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
    "$signed_radial_loss_end_value" "$signed_radial_prediction_norm_end_value" "$signed_radial_target_norm_end_value" \
    "$signed_radial_norm_ratio_end_value" "$signed_radial_samples_value" \
    "$signed_angular_radial_loss_end_value" "$signed_angular_radial_angular_loss_end_value" "$signed_angular_radial_radial_loss_end_value" \
    "$signed_angular_radial_cosine_end_value" "$signed_angular_radial_prediction_norm_end_value" "$signed_angular_radial_target_norm_end_value" \
    "$signed_angular_radial_norm_ratio_end_value" "$signed_angular_radial_samples_value" \
    "$state_latent_mrr_end_value" "$state_latent_top1_end_value" "$state_latent_sign_top1_end_value" \
    "$state_latent_mean_rank_end_value" "$state_projection_mrr_end_value" "$state_projection_top1_end_value" \
    "$state_projection_sign_top1_end_value" "$state_projection_mean_rank_end_value" "$state_support_samples_value" \
    "$state_query_samples_value" "$state_candidates_value" "$(signed_candidate_selector_head_enabled)" \
    "$SIGNED_CANDIDATE_SELECTOR_HEAD_TEMPERATURE" "$SIGNED_CANDIDATE_SELECTOR_HEAD_LR" "$SIGNED_CANDIDATE_SELECTOR_HEAD_WEIGHT" \
    "$SIGNED_CANDIDATE_SELECTOR_HEAD_ENTROPY_FLOOR" "$SIGNED_CANDIDATE_SELECTOR_HEAD_ENTROPY_WEIGHT" "$SIGNED_CANDIDATE_SELECTOR_HEAD_KL_WEIGHT" \
    "$selector_head_mrr_end_value" "$selector_head_top1_end_value" "$selector_head_margin_end_value" \
    "$selector_head_positive_margin_rate_end_value" "$selector_head_sign_margin_end_value" "$selector_head_speed_margin_end_value" \
    "$selector_head_norm_ratio_end_value" "$selector_head_objective_loss_end_value" "$selector_head_objective_ce_end_value" \
    "$selector_head_objective_entropy_reg_end_value" "$selector_head_objective_kl_to_prior_end_value" "$selector_head_entropy_end_value" \
    "$selector_head_true_probability_end_value" "$selector_head_max_probability_end_value" "$selector_head_temperature_end_value" \
    "$selector_head_steps_value" "$selector_head_lr_end_value" "$selector_head_entropy_floor_end_value" \
    "$selector_head_entropy_weight_end_value" "$selector_head_kl_weight_end_value" "$selector_head_support_samples_value" \
    "$selector_head_query_samples_value" "$selector_head_candidates_value" "$status"

  if [[ -n "$REPORT_PATH" ]]; then
    local row_values=(
      "$SCHEMA" "$TEMPORAL_TASK" "$path" "$predictor" "$RESIDUAL_DELTA_SCALE" "$PROJECTOR_DRIFT_WEIGHT" \
      "$SIGNED_MARGIN_WEIGHT" "$SIGNED_MARGIN_BANK_GAP" "$SIGNED_MARGIN_SIGN_GAP" "$SIGNED_MARGIN_SPEED_GAP" \
      "$SIGNED_MARGIN_BANK_WEIGHT" "$SIGNED_MARGIN_SIGN_WEIGHT" "$SIGNED_MARGIN_SPEED_WEIGHT" \
      "$SIGNED_BANK_SOFTMAX_WEIGHT" "$SIGNED_BANK_SOFTMAX_TEMPERATURE" "$SIGNED_RADIAL_WEIGHT" \
      "$SIGNED_ANGULAR_RADIAL_WEIGHT" "$SIGNED_ANGULAR_WEIGHT" "$SIGNED_ANGULAR_RADIAL_RADIUS_WEIGHT" \
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
      "$prediction_unit_mrr_end_value" "$prediction_unit_top1_end_value" "$prediction_unit_true_distance_end_value" \
      "$prediction_unit_nearest_wrong_distance_end_value" "$prediction_unit_margin_end_value" "$prediction_unit_positive_margin_rate_end_value" \
      "$prediction_unit_sign_margin_end_value" "$prediction_unit_speed_margin_end_value" "$prediction_unit_prediction_center_norm_end_value" \
      "$prediction_unit_true_target_center_norm_end_value" "$prediction_unit_samples_value" "$prediction_unit_candidates_value" \
      "$prediction_counterfactual_oracle_radius_mrr_end_value" "$prediction_counterfactual_oracle_radius_top1_end_value" \
      "$prediction_counterfactual_oracle_radius_margin_end_value" "$prediction_counterfactual_oracle_radius_positive_margin_rate_end_value" \
      "$prediction_counterfactual_oracle_radius_sign_margin_end_value" "$prediction_counterfactual_oracle_radius_speed_margin_end_value" \
      "$prediction_counterfactual_oracle_radius_norm_ratio_end_value" "$prediction_counterfactual_oracle_angle_mrr_end_value" \
      "$prediction_counterfactual_oracle_angle_top1_end_value" "$prediction_counterfactual_oracle_angle_margin_end_value" \
      "$prediction_counterfactual_oracle_angle_positive_margin_rate_end_value" "$prediction_counterfactual_oracle_angle_sign_margin_end_value" \
      "$prediction_counterfactual_oracle_angle_speed_margin_end_value" "$prediction_counterfactual_oracle_angle_norm_ratio_end_value" \
      "$prediction_counterfactual_support_global_rescale_mrr_end_value" "$prediction_counterfactual_support_global_rescale_top1_end_value" \
      "$prediction_counterfactual_support_global_rescale_margin_end_value" "$prediction_counterfactual_support_global_rescale_positive_margin_rate_end_value" \
      "$prediction_counterfactual_support_global_rescale_sign_margin_end_value" "$prediction_counterfactual_support_global_rescale_speed_margin_end_value" \
      "$prediction_counterfactual_support_global_rescale_norm_ratio_end_value" "$prediction_counterfactual_support_norm_ratio_end_value" \
      "$prediction_counterfactual_support_samples_value" "$prediction_counterfactual_query_samples_value" "$prediction_counterfactual_candidates_value" \
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
      "$signed_radial_loss_end_value" "$signed_radial_prediction_norm_end_value" "$signed_radial_target_norm_end_value" \
      "$signed_radial_norm_ratio_end_value" "$signed_radial_samples_value" \
      "$signed_angular_radial_loss_end_value" "$signed_angular_radial_angular_loss_end_value" "$signed_angular_radial_radial_loss_end_value" \
      "$signed_angular_radial_cosine_end_value" "$signed_angular_radial_prediction_norm_end_value" "$signed_angular_radial_target_norm_end_value" \
      "$signed_angular_radial_norm_ratio_end_value" "$signed_angular_radial_samples_value" \
      "$state_latent_mrr_end_value" "$state_latent_top1_end_value" "$state_latent_sign_top1_end_value" \
      "$state_latent_mean_rank_end_value" "$state_projection_mrr_end_value" "$state_projection_top1_end_value" \
      "$state_projection_sign_top1_end_value" "$state_projection_mean_rank_end_value" "$state_support_samples_value" \
      "$state_query_samples_value" "$state_candidates_value" "$(signed_candidate_selector_head_enabled)" \
      "$SIGNED_CANDIDATE_SELECTOR_HEAD_TEMPERATURE" "$SIGNED_CANDIDATE_SELECTOR_HEAD_LR" "$SIGNED_CANDIDATE_SELECTOR_HEAD_WEIGHT" \
      "$SIGNED_CANDIDATE_SELECTOR_HEAD_ENTROPY_FLOOR" "$SIGNED_CANDIDATE_SELECTOR_HEAD_ENTROPY_WEIGHT" "$SIGNED_CANDIDATE_SELECTOR_HEAD_KL_WEIGHT" \
      "$selector_head_mrr_end_value" "$selector_head_top1_end_value" "$selector_head_margin_end_value" \
      "$selector_head_positive_margin_rate_end_value" "$selector_head_sign_margin_end_value" "$selector_head_speed_margin_end_value" \
      "$selector_head_norm_ratio_end_value" "$selector_head_objective_loss_end_value" "$selector_head_objective_ce_end_value" \
      "$selector_head_objective_entropy_reg_end_value" "$selector_head_objective_kl_to_prior_end_value" "$selector_head_entropy_end_value" \
      "$selector_head_true_probability_end_value" "$selector_head_max_probability_end_value" "$selector_head_temperature_end_value" \
      "$selector_head_steps_value" "$selector_head_lr_end_value" "$selector_head_entropy_floor_end_value" \
      "$selector_head_entropy_weight_end_value" "$selector_head_kl_weight_end_value" "$selector_head_support_samples_value" \
      "$selector_head_query_samples_value" "$selector_head_candidates_value" \
      "$selector_readout_base_prediction_mrr_end_value" "$selector_readout_base_prediction_top1_end_value" \
      "$selector_readout_base_prediction_margin_end_value" "$selector_readout_base_prediction_positive_margin_rate_end_value" \
      "$selector_readout_base_prediction_sign_margin_end_value" "$selector_readout_base_prediction_speed_margin_end_value" \
      "$selector_readout_base_prediction_norm_ratio_end_value" "$selector_readout_soft_unit_mix_mrr_end_value" \
      "$selector_readout_soft_unit_mix_top1_end_value" "$selector_readout_soft_unit_mix_margin_end_value" \
      "$selector_readout_soft_unit_mix_positive_margin_rate_end_value" "$selector_readout_soft_unit_mix_sign_margin_end_value" \
      "$selector_readout_soft_unit_mix_speed_margin_end_value" "$selector_readout_soft_unit_mix_norm_ratio_end_value" \
      "$selector_readout_soft_full_mrr_end_value" "$selector_readout_soft_full_top1_end_value" \
      "$selector_readout_soft_full_margin_end_value" "$selector_readout_soft_full_positive_margin_rate_end_value" \
      "$selector_readout_soft_full_sign_margin_end_value" "$selector_readout_soft_full_speed_margin_end_value" \
      "$selector_readout_soft_full_norm_ratio_end_value" "$selector_readout_hard_full_mrr_end_value" \
      "$selector_readout_hard_full_top1_end_value" "$selector_readout_hard_full_margin_end_value" \
      "$selector_readout_hard_full_positive_margin_rate_end_value" "$selector_readout_hard_full_sign_margin_end_value" \
      "$selector_readout_hard_full_speed_margin_end_value" "$selector_readout_hard_full_norm_ratio_end_value" \
      "$selector_readout_soft_radius_mrr_end_value" "$selector_readout_soft_radius_top1_end_value" \
      "$selector_readout_soft_radius_margin_end_value" "$selector_readout_soft_radius_positive_margin_rate_end_value" \
      "$selector_readout_soft_radius_sign_margin_end_value" "$selector_readout_soft_radius_speed_margin_end_value" \
      "$selector_readout_soft_radius_norm_ratio_end_value" "$selector_readout_hard_radius_mrr_end_value" \
      "$selector_readout_hard_radius_top1_end_value" "$selector_readout_hard_radius_margin_end_value" \
      "$selector_readout_hard_radius_positive_margin_rate_end_value" "$selector_readout_hard_radius_sign_margin_end_value" \
      "$selector_readout_hard_radius_speed_margin_end_value" "$selector_readout_hard_radius_norm_ratio_end_value" \
      "$(signed_candidate_selector_output_mode)" "$(signed_candidate_selector_output_coupling_enabled)" \
      "$SIGNED_CANDIDATE_SELECTOR_OUTPUT_COUPLING_WEIGHT" "$selector_output_coupling_loss_end_value" \
      "$selector_output_coupling_selector_max_probability_end_value" "$selector_output_coupling_base_prediction_top1_end_value" \
      "$selector_output_coupling_base_prediction_margin_end_value" "$selector_output_coupling_base_prediction_norm_ratio_end_value" \
      "$selector_output_coupling_hard_full_top1_end_value" "$selector_output_coupling_hard_full_margin_end_value" \
      "$selector_output_coupling_hard_full_norm_ratio_end_value" "$selector_output_coupling_samples_value" \
      "$selector_output_coupling_candidates_value" "$status"
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
  local prediction_unit_mrr_end="na"
  local prediction_unit_top1_end="na"
  local prediction_unit_true_distance_end="na"
  local prediction_unit_nearest_wrong_distance_end="na"
  local prediction_unit_margin_end="na"
  local prediction_unit_positive_margin_rate_end="na"
  local prediction_unit_sign_margin_end="na"
  local prediction_unit_speed_margin_end="na"
  local prediction_unit_prediction_center_norm_end="na"
  local prediction_unit_true_target_center_norm_end="na"
  local prediction_unit_samples="na"
  local prediction_unit_candidates="na"
  local prediction_counterfactual_oracle_radius_mrr_end="na"
  local prediction_counterfactual_oracle_radius_top1_end="na"
  local prediction_counterfactual_oracle_radius_margin_end="na"
  local prediction_counterfactual_oracle_radius_positive_margin_rate_end="na"
  local prediction_counterfactual_oracle_radius_sign_margin_end="na"
  local prediction_counterfactual_oracle_radius_speed_margin_end="na"
  local prediction_counterfactual_oracle_radius_norm_ratio_end="na"
  local prediction_counterfactual_oracle_angle_mrr_end="na"
  local prediction_counterfactual_oracle_angle_top1_end="na"
  local prediction_counterfactual_oracle_angle_margin_end="na"
  local prediction_counterfactual_oracle_angle_positive_margin_rate_end="na"
  local prediction_counterfactual_oracle_angle_sign_margin_end="na"
  local prediction_counterfactual_oracle_angle_speed_margin_end="na"
  local prediction_counterfactual_oracle_angle_norm_ratio_end="na"
  local prediction_counterfactual_support_global_rescale_mrr_end="na"
  local prediction_counterfactual_support_global_rescale_top1_end="na"
  local prediction_counterfactual_support_global_rescale_margin_end="na"
  local prediction_counterfactual_support_global_rescale_positive_margin_rate_end="na"
  local prediction_counterfactual_support_global_rescale_sign_margin_end="na"
  local prediction_counterfactual_support_global_rescale_speed_margin_end="na"
  local prediction_counterfactual_support_global_rescale_norm_ratio_end="na"
  local prediction_counterfactual_support_norm_ratio_end="na"
  local prediction_counterfactual_support_samples="na"
  local prediction_counterfactual_query_samples="na"
  local prediction_counterfactual_candidates="na"
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
  local signed_radial_loss_end="na"
  local signed_radial_prediction_norm_end="na"
  local signed_radial_target_norm_end="na"
  local signed_radial_norm_ratio_end="na"
  local signed_radial_samples="na"
  local signed_angular_radial_loss_end="na"
  local signed_angular_radial_angular_loss_end="na"
  local signed_angular_radial_radial_loss_end="na"
  local signed_angular_radial_cosine_end="na"
  local signed_angular_radial_prediction_norm_end="na"
  local signed_angular_radial_target_norm_end="na"
  local signed_angular_radial_norm_ratio_end="na"
  local signed_angular_radial_samples="na"
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
  local selector_head_mrr_end="na"
  local selector_head_top1_end="na"
  local selector_head_margin_end="na"
  local selector_head_positive_margin_rate_end="na"
  local selector_head_sign_margin_end="na"
  local selector_head_speed_margin_end="na"
  local selector_head_norm_ratio_end="na"
  local selector_head_objective_loss_end="na"
  local selector_head_objective_ce_end="na"
  local selector_head_objective_entropy_reg_end="na"
  local selector_head_objective_kl_to_prior_end="na"
  local selector_head_entropy_end="na"
  local selector_head_true_probability_end="na"
  local selector_head_max_probability_end="na"
  local selector_head_temperature_end="na"
  local selector_head_steps="na"
  local selector_head_lr_end="na"
  local selector_head_entropy_floor_end="na"
  local selector_head_entropy_weight_end="na"
  local selector_head_kl_weight_end="na"
  local selector_head_support_samples="na"
  local selector_head_query_samples="na"
  local selector_head_candidates="na"
  local selector_readout_base_prediction_mrr_end="na"
  local selector_readout_base_prediction_top1_end="na"
  local selector_readout_base_prediction_margin_end="na"
  local selector_readout_base_prediction_positive_margin_rate_end="na"
  local selector_readout_base_prediction_sign_margin_end="na"
  local selector_readout_base_prediction_speed_margin_end="na"
  local selector_readout_base_prediction_norm_ratio_end="na"
  local selector_readout_soft_unit_mix_mrr_end="na"
  local selector_readout_soft_unit_mix_top1_end="na"
  local selector_readout_soft_unit_mix_margin_end="na"
  local selector_readout_soft_unit_mix_positive_margin_rate_end="na"
  local selector_readout_soft_unit_mix_sign_margin_end="na"
  local selector_readout_soft_unit_mix_speed_margin_end="na"
  local selector_readout_soft_unit_mix_norm_ratio_end="na"
  local selector_readout_soft_full_mrr_end="na"
  local selector_readout_soft_full_top1_end="na"
  local selector_readout_soft_full_margin_end="na"
  local selector_readout_soft_full_positive_margin_rate_end="na"
  local selector_readout_soft_full_sign_margin_end="na"
  local selector_readout_soft_full_speed_margin_end="na"
  local selector_readout_soft_full_norm_ratio_end="na"
  local selector_readout_hard_full_mrr_end="na"
  local selector_readout_hard_full_top1_end="na"
  local selector_readout_hard_full_margin_end="na"
  local selector_readout_hard_full_positive_margin_rate_end="na"
  local selector_readout_hard_full_sign_margin_end="na"
  local selector_readout_hard_full_speed_margin_end="na"
  local selector_readout_hard_full_norm_ratio_end="na"
  local selector_readout_soft_radius_mrr_end="na"
  local selector_readout_soft_radius_top1_end="na"
  local selector_readout_soft_radius_margin_end="na"
  local selector_readout_soft_radius_positive_margin_rate_end="na"
  local selector_readout_soft_radius_sign_margin_end="na"
  local selector_readout_soft_radius_speed_margin_end="na"
  local selector_readout_soft_radius_norm_ratio_end="na"
  local selector_readout_hard_radius_mrr_end="na"
  local selector_readout_hard_radius_top1_end="na"
  local selector_readout_hard_radius_margin_end="na"
  local selector_readout_hard_radius_positive_margin_rate_end="na"
  local selector_readout_hard_radius_sign_margin_end="na"
  local selector_readout_hard_radius_speed_margin_end="na"
  local selector_readout_hard_radius_norm_ratio_end="na"
  local selector_output_coupling_loss_end="na"
  local selector_output_coupling_selector_max_probability_end="na"
  local selector_output_coupling_base_prediction_top1_end="na"
  local selector_output_coupling_base_prediction_margin_end="na"
  local selector_output_coupling_base_prediction_norm_ratio_end="na"
  local selector_output_coupling_hard_full_top1_end="na"
  local selector_output_coupling_hard_full_margin_end="na"
  local selector_output_coupling_hard_full_norm_ratio_end="na"
  local selector_output_coupling_samples="na"
  local selector_output_coupling_candidates="na"

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
  if [[ "$(signed_radial_enabled)" == "true" ]]; then
    extra_args+=(--signed-radial-weight "$SIGNED_RADIAL_WEIGHT")
  fi
  if [[ "$(signed_angular_radial_enabled)" == "true" ]]; then
    extra_args+=(--signed-angular-radial-weight "$SIGNED_ANGULAR_RADIAL_WEIGHT")
    extra_args+=(--signed-angular-weight "$SIGNED_ANGULAR_WEIGHT")
    extra_args+=(--signed-angular-radial-radius-weight "$SIGNED_ANGULAR_RADIAL_RADIUS_WEIGHT")
  fi
  if [[ "$(signed_candidate_selector_head_enabled)" == "true" ]]; then
    extra_args+=(--signed-candidate-selector-head)
    extra_args+=(--signed-candidate-selector-head-temperature "$SIGNED_CANDIDATE_SELECTOR_HEAD_TEMPERATURE")
    extra_args+=(--signed-candidate-selector-head-lr "$SIGNED_CANDIDATE_SELECTOR_HEAD_LR")
    extra_args+=(--signed-candidate-selector-head-weight "$SIGNED_CANDIDATE_SELECTOR_HEAD_WEIGHT")
    extra_args+=(--signed-candidate-selector-head-entropy-floor "$SIGNED_CANDIDATE_SELECTOR_HEAD_ENTROPY_FLOOR")
    extra_args+=(--signed-candidate-selector-head-entropy-weight "$SIGNED_CANDIDATE_SELECTOR_HEAD_ENTROPY_WEIGHT")
    extra_args+=(--signed-candidate-selector-head-kl-weight "$SIGNED_CANDIDATE_SELECTOR_HEAD_KL_WEIGHT")
  fi
  if [[ "$(signed_candidate_selector_output_coupling_enabled)" == "true" ]]; then
    extra_args+=(--signed-candidate-selector-output hard-full)
    extra_args+=(--signed-candidate-selector-output-coupling-weight "$SIGNED_CANDIDATE_SELECTOR_OUTPUT_COUPLING_WEIGHT")
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
  local final_prediction_bank_unit_geometry_line
  local final_prediction_geometry_counterfactual_line
  local final_signed_objective_error_breakdown_line
  local final_signed_margin_objective_line
  local final_signed_bank_softmax_objective_line
  local final_signed_radial_calibration_line
  local final_signed_angular_radial_objective_line
  local final_signed_state_separability_line
  local final_signed_candidate_selector_head_integration_line
  local final_signed_candidate_selector_output_coupling_line

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

      final_prediction_bank_unit_geometry_line="$(grep -m1 '^final | signed prediction bank unit geometry mrr ' "$log_file" || true)"

      if ! parsed="$(parse_prediction_bank_unit_geometry_line "$final_prediction_bank_unit_geometry_line")"; then
        failures=$((failures + 1))
        emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "na" "na" "$proj_var_mean" "$target_drift_end" "parse_failed"
        rm -f "$log_file"
        return 0
      fi
      read -r prediction_unit_mrr_end prediction_unit_top1_end prediction_unit_true_distance_end prediction_unit_nearest_wrong_distance_end prediction_unit_margin_end prediction_unit_positive_margin_rate_end prediction_unit_sign_margin_end prediction_unit_speed_margin_end prediction_unit_prediction_center_norm_end prediction_unit_true_target_center_norm_end prediction_unit_samples prediction_unit_candidates <<< "$parsed"

      final_prediction_geometry_counterfactual_line="$(grep -m1 '^final | signed prediction geometry counterfactual oracle_radius_mrr ' "$log_file" || true)"

      if ! parsed="$(parse_prediction_geometry_counterfactual_line "$final_prediction_geometry_counterfactual_line")"; then
        failures=$((failures + 1))
        emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "na" "na" "$proj_var_mean" "$target_drift_end" "parse_failed"
        rm -f "$log_file"
        return 0
      fi
      read -r prediction_counterfactual_oracle_radius_mrr_end prediction_counterfactual_oracle_radius_top1_end prediction_counterfactual_oracle_radius_margin_end prediction_counterfactual_oracle_radius_positive_margin_rate_end prediction_counterfactual_oracle_radius_sign_margin_end prediction_counterfactual_oracle_radius_speed_margin_end prediction_counterfactual_oracle_radius_norm_ratio_end prediction_counterfactual_oracle_angle_mrr_end prediction_counterfactual_oracle_angle_top1_end prediction_counterfactual_oracle_angle_margin_end prediction_counterfactual_oracle_angle_positive_margin_rate_end prediction_counterfactual_oracle_angle_sign_margin_end prediction_counterfactual_oracle_angle_speed_margin_end prediction_counterfactual_oracle_angle_norm_ratio_end prediction_counterfactual_support_global_rescale_mrr_end prediction_counterfactual_support_global_rescale_top1_end prediction_counterfactual_support_global_rescale_margin_end prediction_counterfactual_support_global_rescale_positive_margin_rate_end prediction_counterfactual_support_global_rescale_sign_margin_end prediction_counterfactual_support_global_rescale_speed_margin_end prediction_counterfactual_support_global_rescale_norm_ratio_end prediction_counterfactual_support_norm_ratio_end prediction_counterfactual_support_samples prediction_counterfactual_query_samples prediction_counterfactual_candidates <<< "$parsed"

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

      if [[ "$(signed_radial_enabled)" == "true" ]]; then
        final_signed_radial_calibration_line="$(grep -m1 '^final | signed radial calibration loss ' "$log_file" || true)"

        if ! parsed="$(parse_signed_radial_calibration_line "$final_signed_radial_calibration_line")"; then
          failures=$((failures + 1))
          emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "na" "na" "$proj_var_mean" "$target_drift_end" "parse_failed"
          rm -f "$log_file"
          return 0
        fi
        read -r signed_radial_loss_end signed_radial_prediction_norm_end signed_radial_target_norm_end signed_radial_norm_ratio_end signed_radial_samples <<< "$parsed"
      fi

      if [[ "$(signed_angular_radial_enabled)" == "true" ]]; then
        final_signed_angular_radial_objective_line="$(grep -m1 '^final | signed angular-radial objective loss ' "$log_file" || true)"

        if ! parsed="$(parse_signed_angular_radial_objective_line "$final_signed_angular_radial_objective_line")"; then
          failures=$((failures + 1))
          emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "na" "na" "$proj_var_mean" "$target_drift_end" "parse_failed"
          rm -f "$log_file"
          return 0
        fi
        read -r signed_angular_radial_loss_end signed_angular_radial_angular_loss_end signed_angular_radial_radial_loss_end signed_angular_radial_cosine_end signed_angular_radial_prediction_norm_end signed_angular_radial_target_norm_end signed_angular_radial_norm_ratio_end signed_angular_radial_samples <<< "$parsed"
      fi

      final_signed_state_separability_line="$(grep -m1 '^final | signed state separability latent_mrr ' "$log_file" || true)"

      if ! parsed="$(parse_signed_state_separability_line "$final_signed_state_separability_line")"; then
        failures=$((failures + 1))
        emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "na" "na" "$proj_var_mean" "$target_drift_end" "parse_failed"
        rm -f "$log_file"
        return 0
      fi
      read -r state_latent_mrr_end state_latent_top1_end state_latent_sign_top1_end state_latent_mean_rank_end state_projection_mrr_end state_projection_top1_end state_projection_sign_top1_end state_projection_mean_rank_end state_support_samples state_query_samples state_candidates <<< "$parsed"

      if [[ "$(signed_candidate_selector_head_enabled)" == "true" ]]; then
        final_signed_candidate_selector_head_integration_line="$(grep -m1 '^final | signed candidate selector head integration learned_selector_mrr ' "$log_file" || true)"

        if ! parsed="$(parse_signed_candidate_selector_head_integration_line "$final_signed_candidate_selector_head_integration_line")"; then
          failures=$((failures + 1))
          emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "na" "na" "$proj_var_mean" "$target_drift_end" "parse_failed"
          rm -f "$log_file"
          return 0
        fi
        read -r selector_head_mrr_end selector_head_top1_end selector_head_margin_end selector_head_positive_margin_rate_end selector_head_sign_margin_end selector_head_speed_margin_end selector_head_norm_ratio_end selector_head_objective_loss_end selector_head_objective_ce_end selector_head_objective_entropy_reg_end selector_head_objective_kl_to_prior_end selector_head_entropy_end selector_head_true_probability_end selector_head_max_probability_end selector_head_temperature_end selector_head_steps selector_head_lr_end selector_head_entropy_floor_end selector_head_entropy_weight_end selector_head_kl_weight_end selector_head_support_samples selector_head_query_samples selector_head_candidates <<< "$parsed"

        final_signed_candidate_selector_readout_line="$(grep -m1 '^final | signed candidate selector readout base_prediction_mrr ' "$log_file" || true)"
        if ! parsed="$(parse_signed_candidate_selector_readout_line "$final_signed_candidate_selector_readout_line" "base_prediction")"; then
          failures=$((failures + 1))
          emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "na" "na" "$proj_var_mean" "$target_drift_end" "parse_failed"
          rm -f "$log_file"
          return 0
        fi
        read -r selector_readout_base_prediction_mrr_end selector_readout_base_prediction_top1_end selector_readout_base_prediction_margin_end selector_readout_base_prediction_positive_margin_rate_end selector_readout_base_prediction_sign_margin_end selector_readout_base_prediction_speed_margin_end selector_readout_base_prediction_norm_ratio_end <<< "$parsed"

        final_signed_candidate_selector_readout_line="$(grep -m1 '^final | signed candidate selector readout selector_soft_unit_mix_mrr ' "$log_file" || true)"
        if ! parsed="$(parse_signed_candidate_selector_readout_line "$final_signed_candidate_selector_readout_line" "selector_soft_unit_mix")"; then
          failures=$((failures + 1))
          emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "na" "na" "$proj_var_mean" "$target_drift_end" "parse_failed"
          rm -f "$log_file"
          return 0
        fi
        read -r selector_readout_soft_unit_mix_mrr_end selector_readout_soft_unit_mix_top1_end selector_readout_soft_unit_mix_margin_end selector_readout_soft_unit_mix_positive_margin_rate_end selector_readout_soft_unit_mix_sign_margin_end selector_readout_soft_unit_mix_speed_margin_end selector_readout_soft_unit_mix_norm_ratio_end <<< "$parsed"

        final_signed_candidate_selector_readout_line="$(grep -m1 '^final | signed candidate selector readout selector_soft_full_mrr ' "$log_file" || true)"
        if ! parsed="$(parse_signed_candidate_selector_readout_line "$final_signed_candidate_selector_readout_line" "selector_soft_full")"; then
          failures=$((failures + 1))
          emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "na" "na" "$proj_var_mean" "$target_drift_end" "parse_failed"
          rm -f "$log_file"
          return 0
        fi
        read -r selector_readout_soft_full_mrr_end selector_readout_soft_full_top1_end selector_readout_soft_full_margin_end selector_readout_soft_full_positive_margin_rate_end selector_readout_soft_full_sign_margin_end selector_readout_soft_full_speed_margin_end selector_readout_soft_full_norm_ratio_end <<< "$parsed"

        final_signed_candidate_selector_readout_line="$(grep -m1 '^final | signed candidate selector readout selector_hard_full_mrr ' "$log_file" || true)"
        if ! parsed="$(parse_signed_candidate_selector_readout_line "$final_signed_candidate_selector_readout_line" "selector_hard_full")"; then
          failures=$((failures + 1))
          emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "na" "na" "$proj_var_mean" "$target_drift_end" "parse_failed"
          rm -f "$log_file"
          return 0
        fi
        read -r selector_readout_hard_full_mrr_end selector_readout_hard_full_top1_end selector_readout_hard_full_margin_end selector_readout_hard_full_positive_margin_rate_end selector_readout_hard_full_sign_margin_end selector_readout_hard_full_speed_margin_end selector_readout_hard_full_norm_ratio_end <<< "$parsed"

        final_signed_candidate_selector_readout_line="$(grep -m1 '^final | signed candidate selector readout selector_soft_radius_mrr ' "$log_file" || true)"
        if ! parsed="$(parse_signed_candidate_selector_readout_line "$final_signed_candidate_selector_readout_line" "selector_soft_radius")"; then
          failures=$((failures + 1))
          emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "na" "na" "$proj_var_mean" "$target_drift_end" "parse_failed"
          rm -f "$log_file"
          return 0
        fi
        read -r selector_readout_soft_radius_mrr_end selector_readout_soft_radius_top1_end selector_readout_soft_radius_margin_end selector_readout_soft_radius_positive_margin_rate_end selector_readout_soft_radius_sign_margin_end selector_readout_soft_radius_speed_margin_end selector_readout_soft_radius_norm_ratio_end <<< "$parsed"

        final_signed_candidate_selector_readout_line="$(grep -m1 '^final | signed candidate selector readout selector_hard_radius_mrr ' "$log_file" || true)"
        if ! parsed="$(parse_signed_candidate_selector_readout_line "$final_signed_candidate_selector_readout_line" "selector_hard_radius")"; then
          failures=$((failures + 1))
          emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "na" "na" "$proj_var_mean" "$target_drift_end" "parse_failed"
          rm -f "$log_file"
          return 0
        fi
        read -r selector_readout_hard_radius_mrr_end selector_readout_hard_radius_top1_end selector_readout_hard_radius_margin_end selector_readout_hard_radius_positive_margin_rate_end selector_readout_hard_radius_sign_margin_end selector_readout_hard_radius_speed_margin_end selector_readout_hard_radius_norm_ratio_end <<< "$parsed"

        if [[ "$(signed_candidate_selector_output_coupling_enabled)" == "true" ]]; then
          final_signed_candidate_selector_output_coupling_line="$(grep -m1 '^final | signed candidate selector output coupling loss ' "$log_file" || true)"
          if ! parsed="$(parse_signed_candidate_selector_output_coupling_line "$final_signed_candidate_selector_output_coupling_line")"; then
            failures=$((failures + 1))
            emit_row "$path" "$predictor" "$seed" "$encoder_lr" "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" "na" "na" "$proj_var_mean" "$target_drift_end" "parse_failed"
            rm -f "$log_file"
            return 0
          fi
          read -r selector_output_coupling_loss_end selector_output_coupling_selector_max_probability_end selector_output_coupling_base_prediction_top1_end selector_output_coupling_base_prediction_margin_end selector_output_coupling_base_prediction_norm_ratio_end selector_output_coupling_hard_full_top1_end selector_output_coupling_hard_full_margin_end selector_output_coupling_hard_full_norm_ratio_end selector_output_coupling_samples selector_output_coupling_candidates <<< "$parsed"
        fi
      fi
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
