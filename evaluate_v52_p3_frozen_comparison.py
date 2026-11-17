#!/usr/bin/env python3
"""Recompute frozen P3 metrics and compare with aligned v52 predictions."""

import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
HORIZONS = (5, 10, 15)
FOLDS = ("A", "B", "C")
ALIGNMENT_KEYS = ("forecast_origin", "start", "winter", "target_class")
THRESHOLD = 0.20
MERGE_STEPS = 20
BOOTSTRAP_DRAWS = 1000
MODELS = ("climatology", "persistence", "physical_logistic", "scratch", "v51a")
DISPLAY = {
    "climatology": "Climatology", "persistence": "U10 persistence",
    "physical_logistic": "Physical logistic", "scratch": "Deep scratch",
    "v51a": "Deep pretrained",
}
COLORS = {
    "climatology": "#9D9D9D", "persistence": "#59A14F",
    "physical_logistic": "#F28E2B", "scratch": "#4C78A8",
    "v51a": "#E45756",
}


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
    positives = target == 1
    n_positive = int(positives.sum())
    n_negative = int(target.size - n_positive)
    return float((ranks[positives].sum() - n_positive * (n_positive + 1) / 2) /
                 (n_positive * n_negative))


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
    selected_events = [row for row in events if row["winter"] in valid_winters]
    rows = []
    for event in selected_events:
        onset = int(event["onset_step"])
        eligible = (origin < onset) & (onset <= origin + event["horizon"] * 4)
        alarms = eligible & predicted
        rows.append({
            "winter": int(event["winter"]),
            "onset_step": onset,
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


def sorted_arrays(path):
    with np.load(path) as source:
        arrays = {key: source[key] for key in source.files}
    order = np.lexsort(tuple(arrays[key] for key in reversed(ALIGNMENT_KEYS)))
    return {key: value[order] if value.shape[:1] == order.shape else value
            for key, value in arrays.items()}


def load_neural(arm):
    blocks = []
    for fold in FOLDS:
        blocks.append(sorted_arrays(ROOT / (
            f"v52_paired21_{arm}_fold{fold}_seed42_fixed_epoch21_predictions.npz")))
    return {
        key: np.concatenate([block[key] for block in blocks])
        for key in ALIGNMENT_KEYS + ("horizon_probability",)
    }


def paired_bootstrap(physical_probability, scratch_probability, target_class,
                     reference_probability, winters):
    generator = np.random.default_rng(53042)
    unique = np.unique(winters)
    samples = {str(horizon): {key: [] for key in ("pr_auc", "brier", "bss")}
               for horizon in HORIZONS}
    for _ in range(BOOTSTRAP_DRAWS):
        selected = generator.choice(unique, size=unique.size, replace=True)
        indices = np.concatenate([np.where(winters == winter)[0]
                                  for winter in selected])
        for class_index, horizon in enumerate(HORIZONS):
            target = (target_class[indices] <= class_index).astype(int)
            if np.unique(target).size < 2:
                continue
            first = metric_block(physical_probability[indices, class_index],
                                 target, reference_probability[indices, class_index])
            second = metric_block(scratch_probability[indices, class_index],
                                  target, reference_probability[indices, class_index])
            for key in samples[str(horizon)]:
                samples[str(horizon)][key].append(first[key] - second[key])
    return {
        horizon: {
            key: {
                "lower_2.5": float(np.percentile(values, 2.5)),
                "upper_97.5": float(np.percentile(values, 97.5)),
            }
            for key, values in block.items()
        }
        for horizon, block in samples.items()
    }


def plot(result, source_rows):
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42,
    })
    figure, axes = plt.subplots(2, 2, figsize=(7.4, 6.0))
    x = np.arange(len(HORIZONS))
    for model in MODELS:
        style = "--" if model == "climatology" else "-"
        axes[0, 0].plot(x, [result["pooled"][str(h)][model]["pr_auc"]
                            for h in HORIZONS], marker="o", linestyle=style,
                        color=COLORS[model], label=DISPLAY[model])
        axes[0, 1].plot(x, [result["pooled"][str(h)][model]["bss"]
                            for h in HORIZONS], marker="o", linestyle=style,
                        color=COLORS[model])
    axes[0, 0].set_ylabel("PR-AUC")
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper right")
    axes[0, 1].axhline(0, color="0.5", linewidth=1)
    axes[0, 1].set_ylabel("Brier skill score")
    for axis in axes[0]:
        axis.set_xticks(x, [str(h) for h in HORIZONS])
        axis.set_xlabel("Forecast horizon (days)")

    metrics = ("pr_auc", "bss")
    offsets = (-0.12, 0.12)
    for metric, offset, color in zip(metrics, offsets, ("#9467BD", "#8C564B")):
        point = np.asarray([
            result["physical_minus_scratch"][str(h)][metric]
            for h in HORIZONS])
        lower = np.asarray([
            result["paired_winter_bootstrap_95ci"][str(h)][metric]["lower_2.5"]
            for h in HORIZONS])
        upper = np.asarray([
            result["paired_winter_bootstrap_95ci"][str(h)][metric]["upper_97.5"]
            for h in HORIZONS])
        axes[1, 0].errorbar(x + offset, point,
                            yerr=np.vstack((point - lower, upper - point)),
                            marker="o", capsize=3, linewidth=1.4, color=color,
                            label=("delta PR-AUC" if metric == "pr_auc"
                                   else "delta BSS"))
    axes[1, 0].axhline(0, color="0.5", linewidth=1)
    axes[1, 0].set_xticks(x, [str(h) for h in HORIZONS])
    axes[1, 0].set_xlabel("Forecast horizon (days)")
    axes[1, 0].set_ylabel("Physical logistic minus deep scratch")
    axes[1, 0].legend(frameon=False)

    width = 0.36
    for offset, model in zip((-width / 2, width / 2),
                             ("scratch", "physical_logistic")):
        hits = [result["pooled"][str(h)][model]["event_metrics"]["events_hit"]
                for h in HORIZONS]
        false = [-result["pooled"][str(h)][model]["event_metrics"]
                 ["false_alarm_episodes_total"] for h in HORIZONS]
        axes[1, 1].bar(x + offset, hits, width, color=COLORS[model],
                       label=DISPLAY[model])
        axes[1, 1].bar(x + offset, false, width, color=COLORS[model], alpha=0.45)
    axes[1, 1].axhline(0, color="0.4", linewidth=1)
    axes[1, 1].set_xticks(x, [str(h) for h in HORIZONS])
    axes[1, 1].set_xlabel("Forecast horizon (days)")
    axes[1, 1].set_ylabel("Events hit (+) / false episodes (-)")
    axes[1, 1].legend(frameon=False)

    for label, axis in zip("abcd", axes.flat):
        axis.text(0.01, 0.98, label, transform=axis.transAxes,
                  fontsize=10, fontweight="bold", va="top",
                  bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(w_pad=1.8, h_pad=1.8)
    figure.savefig(ROOT / "v52_p3_incremental_skill_comparison.png", dpi=300,
                   bbox_inches="tight")
    figure.savefig(ROOT / "v52_p3_incremental_skill_comparison.pdf",
                   bbox_inches="tight")
    plt.close(figure)


def main():
    p3 = sorted_arrays(ROOT / "v52_p3_physical_baselines_predictions.npz")
    scratch = load_neural("scratch")
    v51a = load_neural("v51a")
    for key in ALIGNMENT_KEYS:
        np.testing.assert_array_equal(p3[key], scratch[key],
                                      err_msg=f"P3/scratch mismatch {key}")
        np.testing.assert_array_equal(p3[key], v51a[key],
                                      err_msg=f"P3/v51a mismatch {key}")
    probabilities = {
        "climatology": p3["climatology_probability"],
        "persistence": p3["persistence_probability"],
        "physical_logistic": p3["physical_logistic_probability"],
        "scratch": scratch["horizon_probability"],
        "v51a": v51a["horizon_probability"],
    }
    for model, probability in probabilities.items():
        if (probability.shape != (1918, 3) or not np.isfinite(probability).all()
                or np.any((probability < 0) | (probability > 1))):
            raise RuntimeError(f"invalid probability array for {model}")

    old = json.loads((ROOT / "v52_paired21_seed42_pooled_results.json").read_text())
    events = [{"winter": int(row["winter"]),
               "onset_step": int(row["onset_step"]), "horizon": horizon}
              for horizon in HORIZONS
              for row in old["pooled_outer_test"]["scratch"][str(horizon)]
              ["event_metrics"]["events"]]
    result = {
        "status": "V52_P3_FROZEN_COMPARISON_OK",
        "alignment": {"status": "PASS", "n_samples": 1918,
                      "keys": list(ALIGNMENT_KEYS)},
        "protocol": {
            "threshold": THRESHOLD,
            "bootstrap": "1000 paired winter-cluster resamples",
            "outer_use": "reporting and uncertainty only; no selection",
            "supersedes": (
                "event_metrics fields in v52_p3_physical_baselines_results.json; "
                "that file inherited a 2018--2021 global event filter"
            ),
        },
        "folds": {}, "pooled": {},
        "physical_minus_scratch": {},
    }
    source_rows = []
    for fold in (*FOLDS, "pooled"):
        mask = np.ones(1918, dtype=bool) if fold == "pooled" else p3["fold"] == fold
        destination = result["pooled"] if fold == "pooled" else result["folds"].setdefault(fold, {})
        for class_index, horizon in enumerate(HORIZONS):
            target = (p3["target_class"][mask] <= class_index).astype(int)
            reference = probabilities["climatology"][mask, class_index]
            horizon_result = {}
            selected_events = [event for event in events if event["horizon"] == horizon]
            for model in MODELS:
                probability = probabilities[model][mask, class_index]
                block = metric_block(probability, target, reference)
                predicted = probability >= THRESHOLD
                block["fixed_0.20"] = decision_metrics(predicted, target)
                block["event_metrics"] = event_metrics(
                    predicted, target, p3["forecast_origin"][mask],
                    p3["winter"][mask], selected_events)
                horizon_result[model] = block
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
            destination[str(horizon)] = horizon_result

    for horizon in HORIZONS:
        result["physical_minus_scratch"][str(horizon)] = {
            key: (result["pooled"][str(horizon)]["physical_logistic"][key] -
                  result["pooled"][str(horizon)]["scratch"][key])
            for key in ("pr_auc", "roc_auc", "brier", "bss")
        }
    result["paired_winter_bootstrap_95ci"] = paired_bootstrap(
        probabilities["physical_logistic"], probabilities["scratch"],
        p3["target_class"], probabilities["climatology"], p3["winter"])
    for horizon in HORIZONS:
        for metric, interval in result["paired_winter_bootstrap_95ci"][str(horizon)].items():
            source_rows.append({
                "record_type": "bootstrap_difference",
                "fold": "pooled", "horizon_days": horizon,
                "model": "physical_logistic_minus_scratch", "metric": metric,
                "point_difference": result["physical_minus_scratch"][str(horizon)][metric],
                **interval,
            })

    path = ROOT / "v52_p3_frozen_comparison.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    fields = sorted({key for row in source_rows for key in row})
    with (ROOT / "v52_p3_incremental_skill_source.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(source_rows)
    plot(result, source_rows)
    print(json.dumps({
        "status": result["status"], "alignment": result["alignment"],
        "physical_minus_scratch": result["physical_minus_scratch"],
        "bootstrap_95ci": result["paired_winter_bootstrap_95ci"],
    }, indent=2))


def self_test():
    target = np.asarray([1, 0, 1, 0])
    good = np.asarray([0.8, 0.2, 0.7, 0.1])
    bad = np.asarray([0.4, 0.6, 0.3, 0.7])
    assert average_precision(target, good) > average_precision(target, bad)
    assert roc_auc(target, good) == 1.0
    assert decision_metrics(good >= 0.5, target)["CSI"] == 1.0
    print("V52_P3_FROZEN_COMPARISON_SELF_TEST_OK")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
