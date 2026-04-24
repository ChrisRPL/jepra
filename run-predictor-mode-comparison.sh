#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_PATH="${JEPRA_MANIFEST_PATH:-$ROOT_DIR/crates/jepra-core/Cargo.toml}"
SCHEMA="jepra_predictor_compare_v2"
TRAIN_STEPS="${JEPRA_TRAIN_STEPS:-300}"
LOG_EVERY="${JEPRA_LOG_EVERY:-25}"
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
  printf 'schema,path,predictor,residual_delta_scale,seed,steps,encoder_mode,encoder_lr,target_momentum_start,target_momentum_end,target_momentum_warmup_steps,train_pred_start,train_pred_end,val_pred_start,val_pred_end,train_obj_start,train_obj_end,val_obj_start,val_obj_end,pred_min_std_final,target_min_std_final,proj_var_mean_final,target_drift_end,status\n' > "$REPORT_PATH"
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

  printf 'schema=%s path=%s predictor=%s residual_delta_scale=%s seed=%s steps=%s encoder_mode=%s encoder_lr=%s target_momentum_start=%s target_momentum_end=%s target_momentum_warmup_steps=%s train_pred_start=%s train_pred_end=%s val_pred_start=%s val_pred_end=%s train_obj_start=%s train_obj_end=%s val_obj_start=%s val_obj_end=%s pred_min_std_final=%s target_min_std_final=%s proj_var_mean_final=%s target_drift_end=%s status=%s\n' \
    "$SCHEMA" "$path" "$predictor" "$RESIDUAL_DELTA_SCALE" "$seed" "$TRAIN_STEPS" "$encoder_mode" "$encoder_lr" \
    "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" \
    "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" \
    "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" \
    "$pred_min_std" "$target_min_std" "$proj_var_mean" "$target_drift_end" "$status"

  if [[ -n "$REPORT_PATH" ]]; then
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$SCHEMA" "$path" "$predictor" "$RESIDUAL_DELTA_SCALE" "$seed" "$TRAIN_STEPS" "$encoder_mode" "$encoder_lr" \
      "$target_momentum_start" "$target_momentum_end" "$target_momentum_warmup_steps" \
      "$train_pred_start" "$train_pred_end" "$val_pred_start" "$val_pred_end" \
      "$train_obj_start" "$train_obj_end" "$val_obj_start" "$val_obj_end" \
      "$pred_min_std" "$target_min_std" "$proj_var_mean" "$target_drift_end" "$status" >> "$REPORT_PATH"
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
