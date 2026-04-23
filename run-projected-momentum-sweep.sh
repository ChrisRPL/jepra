#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_PATH="${JEPRA_MANIFEST_PATH:-$ROOT_DIR/crates/jepra-core/Cargo.toml}"
EXAMPLE="train_vision_jepa_random_temporal_projected"
TRAIN_STEPS="${JEPRA_TRAIN_STEPS:-80}"
LOG_EVERY="${JEPRA_LOG_EVERY:-20}"
WARMUP_STEPS="${JEPRA_WARMUP_STEPS:-24}"
REPORT_PATH="${JEPRA_MOMENTUM_SWEEP_REPORT:-}"
SEEDS_CSV="${JEPRA_MOMENTUM_SEEDS:-21000 21001 21002}"
SCENARIO="${1:-all}"

read -r -a SEEDS <<< "$SEEDS_CSV"
failures=0

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

append_report() {
  local seed="$1"
  local momentum="$2"
  local profile="$3"
  local status="$4"
  local train_tuple="$5"
  local val_tuple="$6"
  local drift_tuple="$7"

  local train_start
  local train_end
  local train_delta
  local train_improved
  local val_start
  local val_end
  local val_delta
  local val_improved
  local drift_start
  local drift_end
  local drift_delta

  read -r train_start train_end train_delta train_improved <<< "$train_tuple"
  read -r val_start val_end val_delta val_improved <<< "$val_tuple"
  read -r drift_start drift_end drift_delta <<< "$drift_tuple"

  {
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
      "$seed" \
      "$momentum" \
      "$profile" \
      "$status" \
      "$train_start" \
      "$train_end" \
      "$train_delta" \
      "$train_improved" \
      "$val_start" \
      "$val_end" \
      "$val_delta" \
      "$val_improved" \
      "$drift_start,$drift_end,$drift_delta"
  } >> "$REPORT_PATH"
}

validate_and_extract_summary() {
  local summary_line="$1"
  local -a segments=()
  local train_segment=""
  local val_segment=""
  local drift_segment=""
  local train_start=""
  local train_end=""
  local train_delta=""
  local train_improved=""
  local val_start=""
  local val_end=""
  local val_delta=""
  local val_improved=""
  local drift_start=""
  local drift_end=""
  local drift_delta=""
  local raw=""

  if [[ "$summary_line" != *"| train "* || "$summary_line" != *"| val "* || "$summary_line" != *"| target drift "* ]]; then
    printf '%s' "summary_format_unexpected"
    return 1
  fi

  local old_ifs="$IFS"
  IFS='|'
  read -r -a segments <<< "$summary_line"
  IFS="$old_ifs"
  if (( ${#segments[@]} < 5 )); then
    printf '%s' "summary_parse_failed"
    return 1
  fi

  train_segment="$(trim "${segments[2]}")"
  val_segment="$(trim "${segments[3]}")"
  drift_segment="$(trim "${segments[4]}")"

  train_tuple="$(sed -E 's/^train[[:space:]]+([^[:space:]]+)[[:space:]]+->[[:space:]]+([^[:space:]]+)[[:space:]]+\(Δ[[:space:]]+([^,]+),[[:space:]]+improved=(true|false)\)$/\1 \2 \3 \4/' <<< "$train_segment")"
  if [[ "$train_tuple" == "$train_segment" ]]; then
    printf '%s' "train_parse_failed"
    return 1
  fi
  train_start="${train_tuple%% *}"
  raw="${train_tuple#* }"
  train_end="${raw%% *}"
  raw="${raw#* }"
  train_delta="${raw%% *}"
  raw="${raw#* }"
  train_improved="${raw}"

  val_tuple="$(sed -E 's/^val[[:space:]]+([^[:space:]]+)[[:space:]]+->[[:space:]]+([^[:space:]]+)[[:space:]]+\(Δ[[:space:]]+([^,]+),[[:space:]]+improved=(true|false)\)$/\1 \2 \3 \4/' <<< "$val_segment")"
  if [[ "$val_tuple" == "$val_segment" ]]; then
    printf '%s' "validation_parse_failed"
    return 1
  fi
  val_start="${val_tuple%% *}"
  raw="${val_tuple#* }"
  val_end="${raw%% *}"
  raw="${raw#* }"
  val_delta="${raw%% *}"
  raw="${raw#* }"
  val_improved="${raw}"

  drift_tuple="$(sed -E 's/^target[[:space:]]+drift[[:space:]]+([^[:space:]]+)[[:space:]]+->[[:space:]]+([^[:space:]]+)[[:space:]]+\(Δ[[:space:]]+([^)]+)\)$/\1 \2 \3/' <<< "$drift_segment")"
  if [[ "$drift_tuple" == "$drift_segment" ]]; then
    printf '%s' "target_drift_parse_failed"
    return 1
  fi
  drift_start="${drift_tuple%% *}"
  raw="${drift_tuple#* }"
  drift_end="${raw%% *}"
  drift_delta="${raw#* }"

  if [[ "$train_improved" != "true" || "$val_improved" != "true" ]]; then
    printf '%s' "improvement_regression_detected"
    return 1
  fi

  printf '%s %s %s %s|%s %s %s %s|%s %s %s' \
    "$train_start" "$train_end" "$train_delta" "$train_improved" \
    "$val_start" "$val_end" "$val_delta" "$val_improved" \
    "$drift_start" "$drift_end" "$drift_delta"
  return 0
}

if [[ "$SCENARIO" == "-h" || "$SCENARIO" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./run-projected-momentum-sweep.sh [all|warmup|frozen|trainable|zero]

Environment:
  JEPRA_MOMENTUM_SEEDS  Space-separated seed list (default: "21000 21001 21002")
  JEPRA_TRAIN_STEPS      Train steps per run (default: 80)
  JEPRA_LOG_EVERY        Log interval passed to example (default: 20)
  JEPRA_WARMUP_STEPS     Warmup steps for warmup profile (default: 24)
  JEPRA_MANIFEST_PATH    Cargo manifest override (default: crates/jepra-core/Cargo.toml)
  JEPRA_MOMENTUM_SWEEP_REPORT
                        Optional CSV path for parsed sweep rows
Supported profiles:
  warmup    -> momentum 1.0 -> 0.5 with linear warmup
  frozen    -> momentum 1.0, encoder-lr 0.0
  trainable -> momentum 0.5, encoder-lr 0.004
  zero      -> momentum 0.0, encoder-lr 0.0
EOF
  exit 0
fi

if [[ "$SCENARIO" != "all" && "$SCENARIO" != "warmup" && "$SCENARIO" != "frozen" && "$SCENARIO" != "trainable" && "$SCENARIO" != "zero" ]]; then
  echo "Unknown scenario: $SCENARIO" >&2
  exit 2
fi

if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  echo "No seeds configured in JEPRA_MOMENTUM_SEEDS" >&2
  exit 2
fi

if [[ -n "$REPORT_PATH" ]]; then
  mkdir -p "$(dirname "$REPORT_PATH")"
  {
    printf 'timestamp,seed,momentum,profile,status,train_start,train_end,train_delta,train_improved,val_start,val_end,val_delta,val_improved,drift_start,drift_end,drift_delta\n'
  } > "$REPORT_PATH"
fi

run_profile() {
  local name="$1"
  local momentum_label="$2"
  local encoder_lr="$3"
  shift 3

  if [[ "$SCENARIO" != "all" && "$SCENARIO" != "$name" ]]; then
    return 0
  fi

  for seed in "${SEEDS[@]}"; do
    local log_file
    log_file="$(mktemp)"
    local status="ok"
    local summary_line
    local parsed_tuple=""

    if ! cargo run --manifest-path "$MANIFEST_PATH" --example "$EXAMPLE" -- \
      --train-base-seed "$seed" \
      --train-steps "$TRAIN_STEPS" \
      --encoder-lr "$encoder_lr" \
      --log "$LOG_EVERY" \
      "$@" >"$log_file" 2>&1; then
      status="run_failed"
      failures=$((failures + 1))
      summary_line="run_failed"
    else
      summary_line="$(grep -m1 '^projected run summary |' "$log_file" || true)"
      if [[ -z "$summary_line" ]]; then
        summary_line="summary_missing"
        if [[ "$status" == "ok" ]]; then
          status="summary_missing"
          failures=$((failures + 1))
        fi
      else
        if ! parsed_tuple="$(validate_and_extract_summary "$summary_line")"; then
          summary_line="${summary_line} | ${parsed_tuple}"
          status="summary_invalid"
          failures=$((failures + 1))
        fi

        if [[ "$status" == "ok" && -n "$REPORT_PATH" ]]; then
          local old_ifs="$IFS"
          IFS='|'
          read -r train_fields val_fields drift_fields <<< "$parsed_tuple"
          IFS="$old_ifs"
          append_report "$seed" "$momentum_label" "$name" "$status" "$train_fields" "$val_fields" "$drift_fields"
        fi
      fi
    fi

    echo "seed=${seed} momentum=${momentum_label} profile=${name} status=${status} | ${summary_line}"
    rm -f "$log_file"
  done
}

run_profile "warmup" "1.0->0.5" "0.0" \
  --target-momentum-start 1.0 \
  --target-momentum-end 0.5 \
  --target-momentum-warmup-steps "$WARMUP_STEPS"

run_profile "frozen" "1.0" "0.0" \
  --target-momentum 1.0

run_profile "trainable" "0.5" "0.004" \
  --target-momentum 0.5

run_profile "zero" "0.0" "0.0" \
  --target-momentum 0.0

if ((failures > 0)); then
  echo "Projected momentum sweep finished with ${failures} failure(s)." >&2
  exit 1
fi

echo "Projected momentum sweep completed."
