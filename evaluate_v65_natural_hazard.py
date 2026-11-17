#!/usr/bin/env python3
"""Frozen three-seed evaluation and promotion gate for V65 hazard forecasts."""

import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parent
SEEDS = (42, 73, 2026)
FOLDS = ("A", "B", "C")
HORIZONS = (5, 10, 15)
THRESHOLD = 0.20
BOOTSTRAP_DRAWS = 1000
ALIGNMENT_KEYS = ("forecast_origin", "start", "winter", "target_class")
MERGE_STEPS = 20
OUTPUT_JSON = ROOT / "v65_natural_hazard_results.json"
OUTPUT_NPZ = ROOT / "v65_natural_hazard_predictions.npz"
OUTPUT_CSV = ROOT / "v65_natural_hazard_source.csv"


def average_precision(target, probability):
    order = np.argsort(-probability, kind="mergesort")
    target = target[order]
    probability = probability[order]
    true_positive = np.cumsum(target)
    false_positive = np.cumsum(1 - target)
    distinct = np.r_[np.where(np.diff(probability))[0], target.size - 1]
    recall = true_positive[distinct] / true_positive[-1]
    precision = true_positive[distinct] / (
        true_positive[distinct] + false_positive[distinct])
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc(target, probability):
    order = np.argsort(probability, kind="mergesort")
    sorted_probability = probability[order]
    ranks = np.empty(target.size, dtype=float)
    left = 0
    while left < target.size:
        right = left + 1
        while (right < target.size and
               sorted_probability[right] == sorted_probability[left]):
            right += 1
        ranks[order[left:right]] = 0.5 * (left + 1 + right)
        left = right
    positive = target == 1
    n_positive = int(positive.sum())
    n_negative = int(target.size - n_positive)
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative))


def decision_metrics(predicted, target):
    tp = int(np.sum(predicted & (target == 1)))
    fp = int(np.sum(predicted & (target == 0)))
    fn = int(np.sum(~predicted & (target == 1)))
    tn = int(np.sum(~predicted & (target == 0)))
    return {
        "CSI": float(tp / max(1, tp + fp + fn)),
        "POD": float(tp / max(1, tp + fn)),
        "FAR": float(fp / max(1, tp + fp)),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def event_metrics(predicted, target, origin, winter, events):
    valid_winters = set(np.unique(winter).tolist())
    selected = [row for row in events if row["winter"] in valid_winters]
    rows = []
    for event in selected:
        onset = int(event["onset_step"])
        eligible = (origin < onset) & (onset <= origin + event["horizon"] * 4)
        alarms = eligible & predicted
        rows.append({
            "winter": int(event["winter"]), "onset_step": onset,
            "hit": bool(np.any(alarms)),
            "first_alarm_lead_days": (
                float(np.max((onset - origin[alarms]) / 4.0))
                if np.any(alarms) else None),
            "n_eligible_initializations": int(eligible.sum()),
        })
    false_by_winter = {}
    for held_winter in sorted(valid_winters):
        false_origins = np.sort(origin[
            (winter == held_winter) & (target == 0) & predicted])
        episodes, previous = 0, None
        for current in false_origins:
            if previous is None or current - previous > MERGE_STEPS:
                episodes += 1
            previous = current
        false_by_winter[str(int(held_winter))] = int(episodes)
    leads = [row["first_alarm_lead_days"] for row in rows
             if row["first_alarm_lead_days"] is not None]
    return {
        "events_hit": int(sum(row["hit"] for row in rows)),
        "n_events": len(rows),
        "mean_first_alarm_lead_days": float(np.mean(leads)) if leads else None,
        "false_alarm_episodes_total": int(sum(false_by_winter.values())),
        "false_alarm_episodes_by_winter": false_by_winter,
        "events": rows,
    }


def reliability(probability, target):
    edges = np.linspace(0.0, 1.0, 11)
    rows = []
    for index in range(10):
        mask = ((probability >= edges[index]) &
                (probability <= edges[index + 1] if index == 9
                 else probability < edges[index + 1]))
        rows.append({
            "lower": float(edges[index]), "upper": float(edges[index + 1]),
            "count": int(mask.sum()),
            "mean_probability": float(probability[mask].mean()) if np.any(mask) else None,
            "observed_frequency": float(target[mask].mean()) if np.any(mask) else None,
        })
    return rows


def sorted_arrays(path):
    with np.load(path) as source:
        arrays = {key: source[key] for key in source.files}
    order = np.lexsort(tuple(arrays[key] for key in reversed(ALIGNMENT_KEYS)))
    return {key: value[order] if value.shape[:1] == order.shape else value
            for key, value in arrays.items()}


def calibration_fit(probability, target):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    target = np.asarray(target, dtype=int)
    if np.unique(target).size < 2:
        return {"intercept": None, "slope": None}
    logit = np.log(probability / (1.0 - probability))

    def objective(parameters):
        fitted = np.clip(
            expit(parameters[0] + parameters[1] * logit), 1e-12, 1 - 1e-12)
        return float(-np.sum(
            target * np.log(fitted) + (1 - target) * np.log(1 - fitted)))

    fitted = minimize(objective, np.asarray([0.0, 1.0]), method="BFGS")
    if not fitted.success and not np.isfinite(fitted.fun):
        return {"intercept": None, "slope": None}
    return {"intercept": float(fitted.x[0]), "slope": float(fitted.x[1])}


def metric_block(probability, target, reference):
    probability = np.asarray(probability, dtype=float)
    target = np.asarray(target, dtype=int)
    reference = np.asarray(reference, dtype=float)
    brier = float(np.mean((probability - target) ** 2))
    reference_brier = float(np.mean((reference - target) ** 2))
    calibration = calibration_fit(probability, target)
    return {
        "pr_auc": average_precision(target, probability),
        "roc_auc": roc_auc(target, probability),
        "brier": brier,
        "reference_brier": reference_brier,
        "bss": float(1.0 - brier / reference_brier),
        "prevalence": float(target.mean()),
        "calibration_intercept": calibration["intercept"],
        "calibration_slope": calibration["slope"],
        "sharpness_sd": float(np.std(probability, ddof=1)),
        "mean_probability": float(np.mean(probability)),
        "reliability": reliability(probability, target),
    }


def load_hazard_seed(seed):
    blocks = []
    for fold in FOLDS:
        result_path = ROOT / f"v65_natural_hazard_fold{fold}_seed{seed}_results.json"
        result = json.loads(result_path.read_text())
        if result.get("status") != "V65_NATURAL_HAZARD_FOLD_OK":
            raise RuntimeError(f"V65 result failed status: {result_path}")
        path = ROOT / (
            f"v65_natural_hazard_fold{fold}_seed{seed}_fixed_epoch21_predictions.npz")
        block = sorted_arrays(path)
        if block["horizon_probability"].shape[1] != 3:
            raise RuntimeError(f"hazard horizon shape changed: {path}")
        if block["daily_hazard_probability"].shape[1] != 15:
            raise RuntimeError(f"daily hazard shape changed: {path}")
        if not np.isfinite(block["horizon_probability"]).all():
            raise RuntimeError(f"non-finite hazard output: {path}")
        if np.any(np.diff(block["horizon_probability"], axis=1) < -1e-7):
            raise RuntimeError(f"non-monotone hazard output: {path}")
        blocks.append(block)
    combined = {
        key: np.concatenate([block[key] for block in blocks])
        for key in ALIGNMENT_KEYS + ("horizon_probability",)
    }
    fold = np.concatenate([
        np.full(block["start"].shape, name) for block, name in zip(blocks, FOLDS)
    ])
    order = np.lexsort(tuple(combined[key] for key in reversed(ALIGNMENT_KEYS)))
    arrays = {key: value[order] for key, value in combined.items()}
    arrays["fold"] = fold[order]
    return arrays


def load_reference():
    with np.load(ROOT / "v55_p0_multiseed_scratch_predictions.npz") as source:
        arrays = {key: source[key] for key in source.files}
    order = np.lexsort(tuple(arrays[key] for key in reversed(ALIGNMENT_KEYS)))
    return {key: value[order] if value.shape[:1] == order.shape else value
            for key, value in arrays.items()}


def event_templates():
    result = json.loads((ROOT / "v55_p0_multiseed_scratch_results.json").read_text())
    return {
        horizon: [{
            "winter": int(row["winter"]),
            "onset_step": int(row["onset_step"]),
            "horizon": horizon,
        } for row in result["metrics"]["three_seed_equal_mean"][str(horizon)]
            ["event_metrics"]["events"]]
        for horizon in HORIZONS
    }


def evaluate(probability, arrays, reference, events, mask=None):
    if mask is None:
        mask = np.ones(len(arrays["winter"]), dtype=bool)
    output = {}
    for class_index, horizon in enumerate(HORIZONS):
        target = (arrays["target_class"][mask] <= class_index).astype(int)
        prob = probability[mask, class_index]
        block = metric_block(prob, target, reference[mask, class_index])
        predicted = prob >= THRESHOLD
        block["fixed_0.20"] = decision_metrics(predicted, target)
        block["event_metrics"] = event_metrics(
            predicted, target, arrays["forecast_origin"][mask],
            arrays["winter"][mask], events[horizon])
        output[str(horizon)] = block
    return output


def paired_bootstrap(hazard, scratch, arrays, reference):
    rng = np.random.default_rng(65065)
    winters = arrays["winter"]
    unique = np.unique(winters)
    samples = {str(h): {name: [] for name in
                        ("delta_pr_auc", "delta_brier", "delta_bss")}
               for h in HORIZONS}
    for _ in range(BOOTSTRAP_DRAWS):
        selected = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([np.where(winters == winter)[0] for winter in selected])
        for class_index, horizon in enumerate(HORIZONS):
            target = (arrays["target_class"][index] <= class_index).astype(int)
            if np.unique(target).size < 2:
                continue
            hb = metric_block(
                hazard[index, class_index], target, reference[index, class_index])
            sb = metric_block(
                scratch[index, class_index], target, reference[index, class_index])
            samples[str(horizon)]["delta_pr_auc"].append(
                hb["pr_auc"] - sb["pr_auc"])
            samples[str(horizon)]["delta_brier"].append(
                hb["brier"] - sb["brier"])
            samples[str(horizon)]["delta_bss"].append(
                hb["bss"] - sb["bss"])
    return {
        horizon: {
            name: {
                "lower_2.5": float(np.percentile(values, 2.5)),
                "median": float(np.percentile(values, 50)),
                "upper_97.5": float(np.percentile(values, 97.5)),
            } for name, values in blocks.items()
        } for horizon, blocks in samples.items()
    }


def promotion_gate(hazard, scratch, by_fold):
    checks = {}
    for horizon in HORIZONS:
        key = str(horizon)
        required_bss = 0.05 if horizon in (5, 10) else 0.03
        checks[f"delta_bss_{horizon}"] = {
            "value": hazard[key]["bss"] - scratch[key]["bss"],
            "required": required_bss,
            "passed": hazard[key]["bss"] - scratch[key]["bss"] >= required_bss,
        }
        checks[f"delta_pr_auc_{horizon}"] = {
            "value": hazard[key]["pr_auc"] - scratch[key]["pr_auc"],
            "required": -0.02,
            "passed": hazard[key]["pr_auc"] - scratch[key]["pr_auc"] >= -0.02,
        }
        checks[f"hits_{horizon}"] = {
            "value": hazard[key]["event_metrics"]["events_hit"],
            "required": scratch[key]["event_metrics"]["events_hit"],
            "passed": hazard[key]["event_metrics"]["events_hit"] >=
                      scratch[key]["event_metrics"]["events_hit"],
        }
        checks[f"false_episodes_{horizon}"] = {
            "value": hazard[key]["event_metrics"]["false_alarm_episodes_total"],
            "required": scratch[key]["event_metrics"]["false_alarm_episodes_total"],
            "passed": hazard[key]["event_metrics"]["false_alarm_episodes_total"] <=
                      scratch[key]["event_metrics"]["false_alarm_episodes_total"],
        }
        checks[f"foldC_bss_{horizon}"] = {
            "value": by_fold["C"]["hazard"][key]["bss"],
            "required": 0.0,
            "passed": by_fold["C"]["hazard"][key]["bss"] >= 0.0,
        }
    passed = all(block["passed"] for block in checks.values())
    return {
        "passed": passed,
        "decision": "retain_regular_AIES_article" if passed else
                    "convert_to_AIES_lessons_learned",
        "checks": checks,
        "no_posthoc_rescue": True,
    }


def self_test():
    target = np.asarray([1, 0, 1, 0])
    probability = np.asarray([0.8, 0.2, 0.7, 0.1])
    assert average_precision(target, probability) == 1.0
    assert roc_auc(target, probability) == 1.0
    fitted = calibration_fit(probability, target)
    assert np.isfinite(fitted["intercept"]) and np.isfinite(fitted["slope"])
    print("V65_NATURAL_HAZARD_EVALUATION_SELF_TEST_OK")


def main():
    seed_arrays = {seed: load_hazard_seed(seed) for seed in SEEDS}
    arrays = seed_arrays[SEEDS[0]]
    for seed in SEEDS[1:]:
        for key in ALIGNMENT_KEYS + ("fold",):
            np.testing.assert_array_equal(arrays[key], seed_arrays[seed][key])
    reference = load_reference()
    for key in ALIGNMENT_KEYS + ("fold",):
        np.testing.assert_array_equal(arrays[key], reference[key])

    hazard_probabilities = {
        f"seed{seed}": seed_arrays[seed]["horizon_probability"]
        for seed in SEEDS
    }
    hazard_probabilities["three_seed_equal_mean"] = np.mean(
        [hazard_probabilities[f"seed{seed}"] for seed in SEEDS], axis=0)
    scratch_probability = reference["three_seed_equal_mean_probability"]
    climatology = reference["climatology_probability"]
    events = event_templates()

    seed_metrics = {
        name: evaluate(probability, arrays, climatology, events)
        for name, probability in hazard_probabilities.items()
    }
    hazard_metrics = seed_metrics["three_seed_equal_mean"]
    scratch_metrics = evaluate(
        scratch_probability, arrays, climatology, events)
    by_fold = {}
    for fold in FOLDS:
        mask = arrays["fold"] == fold
        by_fold[fold] = {
            "hazard": evaluate(
                hazard_probabilities["three_seed_equal_mean"], arrays,
                climatology, events, mask),
            "scratch": evaluate(
                scratch_probability, arrays, climatology, events, mask),
        }

    dispersion = {}
    for horizon in map(str, HORIZONS):
        dispersion[horizon] = {}
        for metric in ("pr_auc", "roc_auc", "brier", "bss"):
            values = [seed_metrics[f"seed{seed}"][horizon][metric]
                      for seed in SEEDS]
            dispersion[horizon][metric] = {
                "minimum": float(min(values)),
                "maximum": float(max(values)),
                "range": float(max(values) - min(values)),
                "standard_deviation": float(np.std(values)),
            }

    gate = promotion_gate(hazard_metrics, scratch_metrics, by_fold)
    result = {
        "status": "V65_NATURAL_HAZARD_EVALUATION_OK",
        "protocol": {
            "registration": "v65_natural_hazard_registration.md",
            "seeds": list(SEEDS), "folds": list(FOLDS),
            "fixed_epochs": 21, "fixed_threshold": THRESHOLD,
            "ensemble": "equal arithmetic probability mean; no seed selection",
            "alignment_keys": list(ALIGNMENT_KEYS),
            "bootstrap": "1000 paired winter-cluster resamples; seed 65065",
        },
        "alignment": {"status": "PASS", "n_samples": len(arrays["winter"]),
                      "n_events": 16},
        "hazard_seed_metrics": seed_metrics,
        "hazard_three_seed": hazard_metrics,
        "scratch_three_seed": scratch_metrics,
        "fold_metrics": by_fold,
        "seed_dispersion": dispersion,
        "paired_winter_bootstrap_95ci": paired_bootstrap(
            hazard_probabilities["three_seed_equal_mean"], scratch_probability,
            arrays, climatology),
        "promotion_gate": gate,
        "claim_boundary": (
            "Frozen retrospective comparison; passing does not establish operational skill. "
            "Failure stops model optimization on A/B/C and triggers Lessons Learned framing."
        ),
    }
    OUTPUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        OUTPUT_NPZ,
        forecast_origin=arrays["forecast_origin"], start=arrays["start"],
        winter=arrays["winter"], target_class=arrays["target_class"],
        fold=arrays["fold"], horizons_days=np.asarray(HORIZONS),
        seed42_probability=hazard_probabilities["seed42"],
        seed73_probability=hazard_probabilities["seed73"],
        seed2026_probability=hazard_probabilities["seed2026"],
        three_seed_equal_mean_probability=hazard_probabilities["three_seed_equal_mean"],
        scratch_three_seed_probability=scratch_probability,
        climatology_probability=climatology,
    )
    rows = []
    for scope, blocks in (("pooled_hazard", hazard_metrics),
                          ("pooled_scratch", scratch_metrics)):
        for horizon, block in blocks.items():
            rows.append({
                "scope": scope, "fold": "pooled", "horizon_days": horizon,
                "pr_auc": block["pr_auc"], "roc_auc": block["roc_auc"],
                "brier": block["brier"], "bss": block["bss"],
                "calibration_intercept": block["calibration_intercept"],
                "calibration_slope": block["calibration_slope"],
                "events_hit": block["event_metrics"]["events_hit"],
                "false_episodes": block["event_metrics"]["false_alarm_episodes_total"],
            })
    for fold, systems in by_fold.items():
        for system, blocks in systems.items():
            for horizon, block in blocks.items():
                rows.append({
                    "scope": system, "fold": fold, "horizon_days": horizon,
                    "pr_auc": block["pr_auc"], "roc_auc": block["roc_auc"],
                    "brier": block["brier"], "bss": block["bss"],
                    "calibration_intercept": block["calibration_intercept"],
                    "calibration_slope": block["calibration_slope"],
                    "events_hit": block["event_metrics"]["events_hit"],
                    "false_episodes": block["event_metrics"]["false_alarm_episodes_total"],
                })
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({
        "status": result["status"], "alignment": result["alignment"],
        "promotion_gate": gate,
        "hazard_three_seed": hazard_metrics,
    }, indent=2))


if __name__ == "__main__":
    self_test() if "--self-test" in sys.argv else main()
