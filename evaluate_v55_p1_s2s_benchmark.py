#!/usr/bin/env python3
"""Construct and evaluate the frozen ECMWF S2S major-SSW probabilities."""

import csv
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from eccodes import codes_get, codes_grib_new_from_file, codes_release


ROOT = Path(__file__).resolve().parent
GRIB_DIR = ROOT / "v55_p1_s2s_cy48r1_grib"
COMMON = ROOT / "v55_p1_s2s_common_origins.csv"
DOWNLOAD_MANIFEST = ROOT / "v55_p1_s2s_download_manifest.json"
AUDIT = ROOT / "v55_p1_s2s_full_audit.json"
P3 = ROOT / "v52_p3_physical_baselines_predictions.npz"
OLD = ROOT / "v52_paired21_seed42_pooled_results.json"
OUTPUT_NPZ = ROOT / "v55_p1_s2s_probabilities.npz"
OUTPUT_JSON = ROOT / "v55_p1_s2s_benchmark_results.json"
OUTPUT_CSV = ROOT / "v55_p1_s2s_benchmark_source.csv"
HORIZONS = (5, 10, 15)
FOLDS = ("A", "B", "C")
MODELS = (
    "climatology", "persistence", "physical_logistic",
    "scratch", "v51a", "ecmwf_s2s",
)
THRESHOLD = 0.20
MERGE_STEPS = 20
BOOTSTRAP_DRAWS = 1000


def average_precision(target, probability):
    order = np.argsort(-probability, kind="mergesort")
    target = target[order]
    probability = probability[order]
    true_positive = np.cumsum(target)
    false_positive = np.cumsum(1 - target)
    distinct = np.r_[np.where(np.diff(probability))[0], target.size - 1]
    recall = true_positive[distinct] / true_positive[-1]
    precision = true_positive[distinct] / (
        true_positive[distinct] + false_positive[distinct]
    )
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc(target, probability):
    order = np.argsort(probability, kind="mergesort")
    sorted_probability = probability[order]
    ranks = np.empty(target.size, dtype=float)
    left = 0
    while left < target.size:
        right = left + 1
        while right < target.size and sorted_probability[right] == sorted_probability[left]:
            right += 1
        ranks[order[left:right]] = 0.5 * (left + 1 + right)
        left = right
    positive = target == 1
    n_positive = int(positive.sum())
    n_negative = int(target.size - n_positive)
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def metric_block(probability, target, reference):
    brier = float(np.mean((probability - target) ** 2))
    reference_brier = float(np.mean((reference - target) ** 2))
    return {
        "pr_auc": average_precision(target, probability),
        "roc_auc": roc_auc(target, probability),
        "brier": brier,
        "reference_brier": reference_brier,
        "bss": float(1.0 - brier / reference_brier),
        "prevalence": float(target.mean()),
    }


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
            "winter": int(event["winter"]),
            "onset_step": onset,
            "hit": bool(np.any(alarms)),
            "first_alarm_lead_days": (
                float(np.max((onset - origin[alarms]) / 4.0))
                if np.any(alarms) else None
            ),
            "n_eligible_initializations": int(eligible.sum()),
        })
    false_by_winter = {}
    for held_winter in sorted(valid_winters):
        false_origins = np.sort(origin[
            (winter == held_winter) & (target == 0) & predicted
        ])
        episodes = 0
        previous = None
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
        mask = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == 9
            else probability < edges[index + 1]
        )
        rows.append({
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "count": int(mask.sum()),
            "mean_probability": float(probability[mask].mean()) if np.any(mask) else None,
            "observed_frequency": float(target[mask].mean()) if np.any(mask) else None,
        })
    return rows


def sorted_arrays(path):
    with np.load(path) as source:
        arrays = {key: source[key] for key in source.files}
    keys = ("forecast_origin", "start", "winter", "target_class")
    order = np.lexsort(tuple(arrays[key] for key in reversed(keys)))
    return {
        key: value[order] if value.shape[:1] == order.shape else value
        for key, value in arrays.items()
    }


def load_neural(arm):
    blocks = []
    for fold in FOLDS:
        blocks.append(sorted_arrays(ROOT / (
            f"v52_paired21_{arm}_fold{fold}_seed42_fixed_epoch21_predictions.npz"
        )))
    keys = ("forecast_origin", "start", "winter", "target_class", "horizon_probability")
    combined = {key: np.concatenate([block[key] for block in blocks]) for key in keys}
    order = np.lexsort(tuple(combined[key] for key in reversed(keys[:4])))
    return {key: value[order] for key, value in combined.items()}


def common_rows():
    with COMMON.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 454:
        raise ValueError("common-origin count changed")
    return rows


def load_u60(rows):
    date_to_index = {
        int(row["forecast_date"].replace("-", "")): index
        for index, row in enumerate(rows)
    }
    values = np.full((len(rows), 11, 47), np.nan, dtype=np.float32)
    manifest = json.loads(DOWNLOAD_MANIFEST.read_text(encoding="utf-8"))
    for file_index, item in enumerate(manifest["requests"], start=1):
        path = GRIB_DIR / item["output"]
        with path.open("rb") as handle:
            while True:
                gid = codes_grib_new_from_file(handle)
                if gid is None:
                    break
                try:
                    hindcast = int(codes_get(gid, "dataDate"))
                    step_index = int(codes_get(gid, "step")) // 24
                    kind = codes_get(gid, "type")
                    number = int(codes_get(gid, "number"))
                    member_index = 0 if kind == "cf" else number
                    index = date_to_index[hindcast]
                    if np.isfinite(values[index, member_index, step_index]):
                        raise ValueError("duplicate U60 value")
                    values[index, member_index, step_index] = float(codes_get(gid, "average"))
                finally:
                    codes_release(gid)
        print(f"U60 {file_index}/30", flush=True)
    if not np.isfinite(values).all():
        raise ValueError("incomplete U60 cube")
    return values


def major_onset_lead(series, origin):
    candidates = [day for day in range(1, 16)
                  if series[day - 1] >= 0.0 and series[day] < 0.0]
    for onset in candidates:
        onset_date = origin + timedelta(days=onset)
        if onset_date.month not in (11, 12, 1, 2, 3):
            continue
        for start in range(onset + 1, 47 - 9):
            recovery_end = origin + timedelta(days=start + 9)
            if recovery_end > date(onset_date.year, 4, 30):
                break
            if np.all(series[start:start + 10] > 0.0):
                return onset
    return -1


def s2s_probabilities(u60, rows):
    leads = np.full((len(rows), 11), -1, dtype=np.int16)
    for index, row in enumerate(rows):
        origin = date.fromisoformat(row["forecast_date"])
        for member in range(11):
            leads[index, member] = major_onset_lead(u60[index, member], origin)
    probability = np.column_stack([
        np.mean((leads > 0) & (leads <= horizon), axis=1)
        for horizon in HORIZONS
    ])
    probability_minus1 = np.column_stack([
        np.mean((leads > 0) & (leads <= horizon - 1), axis=1)
        for horizon in HORIZONS
    ])
    probability_plus1 = np.column_stack([
        np.mean((leads > 0) & (leads <= horizon + 1), axis=1)
        for horizon in HORIZONS
    ])
    return leads, probability, probability_minus1, probability_plus1


def bootstrap(probabilities, target_class, reference, winters):
    generator = np.random.default_rng(55101)
    unique = np.unique(winters)
    samples = {
        str(horizon): {model: {key: [] for key in ("pr_auc", "brier", "bss")}
                       for model in MODELS}
        for horizon in HORIZONS
    }
    differences = {
        str(horizon): {key: [] for key in ("pr_auc", "brier", "bss")}
        for horizon in HORIZONS
    }
    for _ in range(BOOTSTRAP_DRAWS):
        selected = generator.choice(unique, size=unique.size, replace=True)
        indices = np.concatenate([np.where(winters == winter)[0] for winter in selected])
        for class_index, horizon in enumerate(HORIZONS):
            target = (target_class[indices] <= class_index).astype(int)
            if np.unique(target).size < 2:
                continue
            blocks = {}
            for model in MODELS:
                blocks[model] = metric_block(
                    probabilities[model][indices, class_index], target,
                    reference[indices, class_index],
                )
                for key in samples[str(horizon)][model]:
                    samples[str(horizon)][model][key].append(blocks[model][key])
            for key in differences[str(horizon)]:
                differences[str(horizon)][key].append(
                    blocks["ecmwf_s2s"][key] - blocks["scratch"][key]
                )
    def intervals(block):
        return {
            key: {
                "lower_2.5": float(np.percentile(values, 2.5)),
                "upper_97.5": float(np.percentile(values, 97.5)),
            }
            for key, values in block.items()
        }
    return {
        "models": {
            horizon: {model: intervals(block) for model, block in models.items()}
            for horizon, models in samples.items()
        },
        "ecmwf_minus_scratch": {
            horizon: intervals(block) for horizon, block in differences.items()
        },
    }


def self_test():
    origin = date(2021, 1, 1)
    series = np.ones(47)
    series[4:8] = -1.0
    assert major_onset_lead(series, origin) == 4
    series[:] = 1.0
    series[4:] = -1.0
    assert major_onset_lead(series, origin) == -1
    target = np.asarray([1, 0, 1, 0])
    assert average_precision(target, np.asarray([0.8, 0.2, 0.7, 0.1])) == 1.0
    print("V55_P1_S2S_BENCHMARK_SELF_TEST_OK")


def main():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit["status"] != "V55_P1_S2S_FULL_AUDIT_OK":
        raise RuntimeError("full S2S audit has not passed")
    rows = common_rows()
    u60 = load_u60(rows)
    leads, s2s, s2s_minus1, s2s_plus1 = s2s_probabilities(u60, rows)

    p3 = sorted_arrays(P3)
    scratch = load_neural("scratch")
    v51a = load_neural("v51a")
    index = {int(value): position for position, value in enumerate(p3["forecast_origin"])}
    selected = np.asarray([index[int(row["forecast_origin"])] for row in rows])
    arrays = {key: p3[key][selected] for key in (
        "forecast_origin", "start", "winter", "target_class", "fold"
    )}
    for key in ("start", "winter", "target_class"):
        expected = np.asarray([int(row[key]) for row in rows])
        np.testing.assert_array_equal(arrays[key], expected)
    np.testing.assert_array_equal(arrays["fold"], np.asarray([row["fold"] for row in rows]))
    for neural in (scratch, v51a):
        neural_index = {int(value): position for position, value in enumerate(neural["forecast_origin"])}
        order = np.asarray([neural_index[int(row["forecast_origin"])] for row in rows])
        for key in ("start", "winter", "target_class"):
            np.testing.assert_array_equal(neural[key][order], arrays[key])
        neural["selected_probability"] = neural["horizon_probability"][order]

    probabilities = {
        "climatology": p3["climatology_probability"][selected],
        "persistence": p3["persistence_probability"][selected],
        "physical_logistic": p3["physical_logistic_probability"][selected],
        "scratch": scratch["selected_probability"],
        "v51a": v51a["selected_probability"],
        "ecmwf_s2s": s2s,
    }
    for model, probability in probabilities.items():
        if probability.shape != (454, 3) or not np.isfinite(probability).all():
            raise ValueError(f"invalid probability for {model}")

    old = json.loads(OLD.read_text(encoding="utf-8"))
    events = [{
        "winter": int(row["winter"]), "onset_step": int(row["onset_step"]),
        "horizon": horizon,
    } for horizon in HORIZONS for row in
        old["pooled_outer_test"]["scratch"][str(horizon)]["event_metrics"]["events"]]

    result = {
        "status": "V55_P1_S2S_BENCHMARK_OK",
        "alignment": {"status": "PASS", "n_samples": 454, "covered_events": 11},
        "protocol": {
            "s2s_model": "ECMWF IFS cy48r1",
            "members": 11, "threshold": THRESHOLD,
            "onset": "first 00UTC U60N 10hPa westerly-to-easterly transition",
            "major_event": "onset in Nov-Mar followed by 10 consecutive westerly days before 30 April within 46-day forecast",
            "prior_event_separation": "frozen outer origins already exclude observed recovery windows",
            "date_sensitivity": "fixed horizon minus/plus one day; diagnostic only",
            "bootstrap": "1000 winter-cluster resamples",
            "claim_boundary": "454 common origins and 11/16 events only; no model or rule selection",
        },
        "folds": {}, "pooled": {}, "date_sensitivity": {},
    }
    source_rows = []
    for fold in (*FOLDS, "pooled"):
        mask = np.ones(454, dtype=bool) if fold == "pooled" else arrays["fold"] == fold
        destination = result["pooled"] if fold == "pooled" else result["folds"].setdefault(fold, {})
        for class_index, horizon in enumerate(HORIZONS):
            target = (arrays["target_class"][mask] <= class_index).astype(int)
            reference = probabilities["climatology"][mask, class_index]
            selected_events = [event for event in events if event["horizon"] == horizon]
            destination[str(horizon)] = {}
            for model in MODELS:
                probability = probabilities[model][mask, class_index]
                block = metric_block(probability, target, reference)
                predicted = probability >= THRESHOLD
                block["fixed_0.20"] = decision_metrics(predicted, target)
                block["event_metrics"] = event_metrics(
                    predicted, target, arrays["forecast_origin"][mask],
                    arrays["winter"][mask], selected_events,
                )
                block["reliability"] = reliability(probability, target)
                destination[str(horizon)][model] = block
                source_rows.append({
                    "record_type": "metric", "fold": fold,
                    "horizon_days": horizon, "model": model,
                    "pr_auc": block["pr_auc"], "roc_auc": block["roc_auc"],
                    "brier": block["brier"], "bss": block["bss"],
                    "CSI": block["fixed_0.20"]["CSI"],
                    "POD": block["fixed_0.20"]["POD"],
                    "FAR": block["fixed_0.20"]["FAR"],
                    "events_hit": block["event_metrics"]["events_hit"],
                    "n_events": block["event_metrics"]["n_events"],
                    "mean_lead_days": block["event_metrics"]["mean_first_alarm_lead_days"],
                    "false_episodes": block["event_metrics"]["false_alarm_episodes_total"],
                })

    for class_index, horizon in enumerate(HORIZONS):
        target = (arrays["target_class"] <= class_index).astype(int)
        reference = probabilities["climatology"][:, class_index]
        result["date_sensitivity"][str(horizon)] = {}
        for label, probability in (("minus_1_day", s2s_minus1), ("plus_1_day", s2s_plus1)):
            result["date_sensitivity"][str(horizon)][label] = metric_block(
                probability[:, class_index], target, reference
            )

    result["winter_cluster_bootstrap_95ci"] = bootstrap(
        probabilities, arrays["target_class"], probabilities["climatology"], arrays["winter"]
    )
    np.savez_compressed(
        OUTPUT_NPZ,
        forecast_origin=arrays["forecast_origin"], start=arrays["start"],
        winter=arrays["winter"], target_class=arrays["target_class"], fold=arrays["fold"],
        horizons_days=np.asarray(HORIZONS), u60=u60, member_onset_lead_days=leads,
        ecmwf_s2s_probability=s2s,
        ecmwf_s2s_probability_minus_1_day=s2s_minus1,
        ecmwf_s2s_probability_plus_1_day=s2s_plus1,
    )
    OUTPUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    fields = sorted({key for row in source_rows for key in row})
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(source_rows)
    print(json.dumps({
        "status": result["status"], "alignment": result["alignment"],
        "pooled": {horizon: {
            model: {key: result["pooled"][horizon][model][key]
                    for key in ("pr_auc", "roc_auc", "brier", "bss")}
            for model in MODELS
        } for horizon in map(str, HORIZONS)},
    }, indent=2))


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
