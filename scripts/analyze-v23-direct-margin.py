#!/usr/bin/env python3
"""Gate/analyze jepra_predictor_compare_v23 direct-margin CSV evidence."""

import argparse
import csv
import math
import sys
from collections import defaultdict

SCHEMA = "jepra_predictor_compare_v23"
EPS = 1e-12

PAIR_FIELDS = (
    "temporal_task",
    "path",
    "predictor",
    "seed",
    "steps",
    "encoder_mode",
    "selector_output_mode",
)
GROUP_FIELDS = tuple(field for field in PAIR_FIELDS if field != "seed") + (
    "signed_direct_candidate_margin_weight",
    "signed_direct_candidate_margin",
)


def is_missing(value):
    return value is None or value.strip().lower() in {"", "na", "nan", "none"}


def text(row, name, default=""):
    value = row.get(name)
    if value is None or value == "":
        return default
    return value


def number(row, name):
    value = row.get(name)
    if is_missing(value):
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name}={value!r} is not numeric") from exc
    if not math.isfinite(parsed):
        return None
    return parsed


def direct_weight(row):
    value = number(row, "signed_direct_candidate_margin_weight")
    return 0.0 if value is None else value


def key(row, fields):
    return tuple(text(row, field, "off" if field == "selector_output_mode" else "") for field in fields)


def mean(values):
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def metric(rows, name):
    return mean(number(row, name) for row in rows)


def ratio(numerator, denominator):
    if numerator is None or denominator is None or abs(denominator) <= EPS:
        return None
    return numerator / denominator


def norm_ratio(row):
    return ratio(
        number(row, "prediction_unit_prediction_center_norm_end"),
        number(row, "prediction_unit_true_target_center_norm_end"),
    )


def fmt(value):
    if value is None:
        return "na"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.6f}"


def ok_status(row):
    return text(row, "status") == "ok"


def wanted(row, args):
    if not args.all_schemas and text(row, "schema") != SCHEMA:
        return False
    if args.predictor != "any" and text(row, "predictor") != args.predictor:
        return False
    if args.selector_output_mode != "any":
        mode = text(row, "selector_output_mode", "off")
        if mode != args.selector_output_mode:
            return False
    return True


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("empty CSV or missing header")
        return list(reader)


def collect_pairs(rows, args):
    baselines = defaultdict(list)
    candidates = []
    for row in rows:
        if not wanted(row, args):
            continue
        weight = direct_weight(row)
        if abs(weight) <= EPS:
            baselines[key(row, PAIR_FIELDS)].append(row)
        elif args.candidate_weight is None or abs(weight - args.candidate_weight) <= EPS:
            candidates.append(row)

    groups = defaultdict(list)
    missing = defaultdict(list)
    for candidate in candidates:
        baseline_rows = baselines.get(key(candidate, PAIR_FIELDS), [])
        group_key = key(candidate, GROUP_FIELDS)
        if not baseline_rows:
            missing[group_key].append(text(candidate, "seed", "unknown"))
            continue
        groups[group_key].append((baseline_rows[0], candidate))
    return groups, missing


def gate_group(group_key, pairs, missing_seeds):
    baseline_rows = [baseline for baseline, _ in pairs]
    candidate_rows = [candidate for _, candidate in pairs]

    base_val = metric(baseline_rows, "val_pred_end")
    cand_val = metric(candidate_rows, "val_pred_end")
    val_ratio = ratio(cand_val, base_val)

    base_drift = metric(baseline_rows, "target_drift_end")
    cand_drift = metric(candidate_rows, "target_drift_end")
    drift_limit = None if base_drift is None else max(0.02, 2.5 * base_drift)

    base_raw = metric(baseline_rows, "prediction_bank_positive_margin_rate_end")
    cand_raw = metric(candidate_rows, "prediction_bank_positive_margin_rate_end")
    base_oracle = metric(
        baseline_rows,
        "prediction_counterfactual_oracle_radius_positive_margin_rate_end",
    )
    raw_gate = None
    if base_raw is not None and base_oracle is not None:
        raw_gate = base_raw + 0.5 * (base_oracle - base_raw)

    base_unit_ppr = metric(baseline_rows, "prediction_unit_positive_margin_rate_end")
    cand_unit_ppr = metric(candidate_rows, "prediction_unit_positive_margin_rate_end")
    base_unit_mrr = metric(baseline_rows, "prediction_unit_mrr_end")
    cand_unit_mrr = metric(candidate_rows, "prediction_unit_mrr_end")
    unit_ppr_gate = None if base_unit_ppr is None else base_unit_ppr - 0.03
    unit_mrr_gate = None if base_unit_mrr is None else base_unit_mrr - 0.03

    base_norm = mean(norm_ratio(row) for row in baseline_rows)
    cand_norm = mean(norm_ratio(row) for row in candidate_rows)
    base_norm_err = None if base_norm is None else abs(base_norm - 1.0)
    cand_norm_err = None if cand_norm is None else abs(cand_norm - 1.0)
    if base_norm_err is None or cand_norm_err is None:
        norm_shrink = None
    elif base_norm_err <= EPS:
        norm_shrink = 1.0 if cand_norm_err <= EPS else -math.inf
    else:
        norm_shrink = (base_norm_err - cand_norm_err) / base_norm_err

    baseline_ok = sum(1 for row in baseline_rows if ok_status(row))
    candidate_ok = sum(1 for row in candidate_rows if ok_status(row))
    all_status_ok = baseline_ok == len(baseline_rows) and candidate_ok == len(candidate_rows)

    health_ok = (
        all_status_ok
        and val_ratio is not None
        and val_ratio <= 1.05
        and cand_drift is not None
        and drift_limit is not None
        and cand_drift <= drift_limit
        and not missing_seeds
    )
    raw_ok = raw_gate is not None and cand_raw is not None and cand_raw >= raw_gate
    unit_ok = (
        unit_ppr_gate is not None
        and unit_mrr_gate is not None
        and cand_unit_ppr is not None
        and cand_unit_mrr is not None
        and cand_unit_ppr >= unit_ppr_gate
        and cand_unit_mrr >= unit_mrr_gate
    )
    norm_ok = (
        cand_norm is not None
        and norm_shrink is not None
        and 0.50 <= cand_norm <= 1.50
        and norm_shrink >= 0.25
    )
    passed = health_ok and raw_ok and unit_ok and norm_ok

    print(("PASS" if passed else "REJECT") + f" direct-margin v23 pairs={len(pairs)}")
    print("group " + " ".join(f"{field}={value}" for field, value in zip(GROUP_FIELDS, group_key)))
    if missing_seeds:
        print("missing_baseline_seeds=" + ",".join(sorted(set(missing_seeds))))
    print(
        "health "
        f"baseline_ok={baseline_ok}/{len(baseline_rows)} "
        f"direct_ok={candidate_ok}/{len(candidate_rows)} "
        f"val_ratio={fmt(val_ratio)}<=1.050000 "
        f"drift={fmt(cand_drift)}<={fmt(drift_limit)}"
    )
    print(
        "raw_ppr "
        f"baseline={fmt(base_raw)} direct={fmt(cand_raw)} "
        f"oracle_radius={fmt(base_oracle)} gate={fmt(raw_gate)}"
    )
    print(
        "unit "
        f"ppr_baseline={fmt(base_unit_ppr)} ppr_direct={fmt(cand_unit_ppr)} "
        f"ppr_gate={fmt(unit_ppr_gate)} "
        f"mrr_baseline={fmt(base_unit_mrr)} mrr_direct={fmt(cand_unit_mrr)} "
        f"mrr_gate={fmt(unit_mrr_gate)}"
    )
    print(
        "centered_norm_ratio "
        f"baseline={fmt(base_norm)} direct={fmt(cand_norm)} "
        f"error_shrink={fmt(norm_shrink)} gate=0.250000 range=[0.500000,1.500000]"
    )
    print(
        "direct_margin "
        f"loss={fmt(metric(candidate_rows, 'signed_direct_candidate_margin_loss_end'))} "
        f"active_rate={fmt(metric(candidate_rows, 'signed_direct_candidate_margin_active_rate_end'))} "
        f"true_distance={fmt(metric(candidate_rows, 'signed_direct_candidate_margin_true_distance_end'))} "
        f"wrong_distance={fmt(metric(candidate_rows, 'signed_direct_candidate_margin_wrong_distance_end'))} "
        f"margin={fmt(metric(candidate_rows, 'signed_direct_candidate_margin_margin_end'))} "
        f"ppr={fmt(metric(candidate_rows, 'signed_direct_candidate_margin_positive_margin_rate_end'))} "
        f"top1={fmt(metric(candidate_rows, 'signed_direct_candidate_margin_top1_end'))}"
    )
    print("secondary " + secondary_summary(pairs))
    for baseline, candidate in pairs:
        seed = text(candidate, "seed", "unknown")
        seed_val_ratio = ratio(number(candidate, "val_pred_end"), number(baseline, "val_pred_end"))
        seed_drift_limit = None
        base_seed_drift = number(baseline, "target_drift_end")
        if base_seed_drift is not None:
            seed_drift_limit = max(0.02, 2.5 * base_seed_drift)
        print(
            f"seed={seed} "
            f"status={text(baseline, 'status')}/{text(candidate, 'status')} "
            f"val_ratio={fmt(seed_val_ratio)} "
            f"drift={fmt(number(candidate, 'target_drift_end'))}<={fmt(seed_drift_limit)} "
            f"raw_ppr={fmt(number(baseline, 'prediction_bank_positive_margin_rate_end'))}->{fmt(number(candidate, 'prediction_bank_positive_margin_rate_end'))} "
            f"unit_ppr={fmt(number(baseline, 'prediction_unit_positive_margin_rate_end'))}->{fmt(number(candidate, 'prediction_unit_positive_margin_rate_end'))} "
            f"unit_mrr={fmt(number(baseline, 'prediction_unit_mrr_end'))}->{fmt(number(candidate, 'prediction_unit_mrr_end'))} "
            f"norm_ratio={fmt(norm_ratio(baseline))}->{fmt(norm_ratio(candidate))} "
            f"direct_ppr={fmt(number(candidate, 'signed_direct_candidate_margin_positive_margin_rate_end'))}"
        )
    return passed


def secondary_summary(pairs):
    labels = (
        ("margin", "prediction_bank_margin_end"),
        ("sign_margin", "prediction_bank_sign_margin_end"),
        ("speed_margin", "prediction_bank_speed_margin_end"),
    )
    parts = []
    for label, field in labels:
        comparable = 0
        improved = 0
        for baseline, candidate in pairs:
            base = number(baseline, field)
            direct = number(candidate, field)
            if base is None or direct is None:
                continue
            comparable += 1
            if direct > base:
                improved += 1
        parts.append(f"{label}_improved={improved}/{comparable}")
    return " ".join(parts)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Gate v23 direct-margin evidence against same-seed baseline/off CSV rows.",
    )
    parser.add_argument("csv_path", help="jepra_predictor_compare_v23 CSV report")
    parser.add_argument(
        "--candidate-weight",
        type=float,
        help="Only analyze this signed_direct_candidate_margin_weight",
    )
    parser.add_argument(
        "--predictor",
        default="baseline",
        help="Predictor filter, or 'any' (default: baseline)",
    )
    parser.add_argument(
        "--selector-output-mode",
        default="off",
        help="Selector output filter, or 'any' (default: off)",
    )
    parser.add_argument(
        "--all-schemas",
        action="store_true",
        help=f"Do not require schema={SCHEMA}",
    )
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    try:
        rows = load_rows(args.csv_path)
        groups, missing = collect_pairs(rows, args)
        if not groups and not missing:
            raise ValueError("no matching direct-margin candidate rows found")
        overall_pass = True
        for index, group_key in enumerate(sorted(set(groups) | set(missing))):
            if index:
                print()
            group_pass = gate_group(group_key, groups.get(group_key, []), missing.get(group_key, []))
            overall_pass = overall_pass and group_pass
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
