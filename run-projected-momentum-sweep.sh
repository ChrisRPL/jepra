#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_PATH="${JEPRA_MANIFEST_PATH:-$ROOT_DIR/crates/jepra-core/Cargo.toml}"
EXAMPLE="train_vision_jepa_random_temporal_projected"
TRAIN_STEPS="${JEPRA_TRAIN_STEPS:-80}"
LOG_EVERY="${JEPRA_LOG_EVERY:-20}"
WARMUP_STEPS="${JEPRA_WARMUP_STEPS:-24}"
SEEDS_CSV="${JEPRA_MOMENTUM_SEEDS:-21000 21001 21002}"
SCENARIO="${1:-all}"

read -r -a SEEDS <<< "$SEEDS_CSV"

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

failures=0

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
