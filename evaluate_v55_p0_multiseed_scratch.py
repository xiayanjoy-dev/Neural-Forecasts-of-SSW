#!/usr/bin/env python3
"""Frozen three-seed stability evaluation for v55 scratch predictions."""

import csv
import json
from pathlib import Path

import numpy as np

from evaluate_v55_p1_s2s_benchmark import (
    average_precision,
    decision_metrics,
    event_metrics,
    metric_block,
    reliability,
    roc_auc,
    sorted_arrays,
)


ROOT = Path(__file__).resolve().parent
SEEDS = (42, 73, 2026)
FOLDS = ("A", "B", "C")
HORIZONS = (5, 10, 15)
THRESHOLD = 0.20
BOOTSTRAP_DRAWS = 1000
ALIGNMENT_KEYS = ("forecast_origin", "start", "winter", "target_class")
OUTPUT_JSON = ROOT / "v55_p0_multiseed_scratch_results.json"
OUTPUT_NPZ = ROOT / "v55_p0_multiseed_scratch_predictions.npz"
OUTPUT_CSV = ROOT / "v55_p0_multiseed_scratch_source.csv"


def load_seed(seed):
    blocks = [sorted_arrays(ROOT / (
        f"v52_paired21_scratch_fold{fold}_seed{seed}_fixed_epoch21_predictions.npz"
    )) for fold in FOLDS]
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
    if np.unique(arrays["forecast_origin"]).size != arrays["forecast_origin"].size:
        raise RuntimeError(f"seed {seed} has overlapping outer-test origins")
    return arrays


def aligned_reference(seed_arrays):
    reference = seed_arrays[SEEDS[0]]
    for seed in SEEDS[1:]:
        for key in ALIGNMENT_KEYS + ("fold",):
            np.testing.assert_array_equal(
                reference[key], seed_arrays[seed][key],
                err_msg=f"seed {seed} misaligned {key}",
            )
    p3 = sorted_arrays(ROOT / "v52_p3_physical_baselines_predictions.npz")
    index = {int(value): position for position, value in enumerate(p3["forecast_origin"])}
    selected = np.asarray([index[int(value)] for value in reference["forecast_origin"]])
    for key in ALIGNMENT_KEYS:
        np.testing.assert_array_equal(reference[key], p3[key][selected])
    np.testing.assert_array_equal(reference["fold"], p3["fold"][selected])
    return reference, p3["climatology_probability"][selected]


def event_table():
    old = json.loads((ROOT / "v52_paired21_seed42_pooled_results.json").read_text())
    return {
        horizon: [{
            "winter": int(row["winter"]),
            "onset_step": int(row["onset_step"]),
            "horizon": horizon,
        } for row in old["pooled_outer_test"]["scratch"][str(horizon)][
            "event_metrics"
        ]["events"]]
        for horizon in HORIZONS
    }


def evaluate_probability(probability, arrays, climatology, events):
    output = {}
    for class_index, horizon in enumerate(HORIZONS):
        target = (arrays["target_class"] <= class_index).astype(int)
        prob = probability[:, class_index]
        block = metric_block(prob, target, climatology[:, class_index])
        predicted = prob >= THRESHOLD
        block["fixed_0.20"] = decision_metrics(predicted, target)
        block["event_metrics"] = event_metrics(
            predicted, target, arrays["forecast_origin"], arrays["winter"],
            events[horizon],
        )
        block["reliability"] = reliability(prob, target)
        output[str(horizon)] = block
    return output


def winter_bootstrap(probabilities, arrays, climatology):
    generator = np.random.default_rng(55042)
    winters = arrays["winter"]
    unique = np.unique(winters)
    metrics = ("pr_auc", "roc_auc", "brier", "bss")
    samples = {
        str(horizon): {model: {metric: [] for metric in metrics}
                       for model in probabilities}
        for horizon in HORIZONS
    }
    for _ in range(BOOTSTRAP_DRAWS):
        selected = generator.choice(unique, size=unique.size, replace=True)
        indices = np.concatenate([np.where(winters == winter)[0] for winter in selected])
        for class_index, horizon in enumerate(HORIZONS):
            target = (arrays["target_class"][indices] <= class_index).astype(int)
            if np.unique(target).size < 2:
                continue
            reference = climatology[indices, class_index]
            for model, probability in probabilities.items():
                block = metric_block(probability[indices, class_index], target, reference)
                for metric in metrics:
                    samples[str(horizon)][model][metric].append(block[metric])
    return {
        horizon: {
            model: {
                metric: {
                    "lower_2.5": float(np.percentile(values, 2.5)),
                    "upper_97.5": float(np.percentile(values, 97.5)),
                }
                for metric, values in block.items()
            }
            for model, block in models.items()
        }
        for horizon, models in samples.items()
    }


def self_test():
    target = np.asarray([1, 0, 1, 0])
    probability = np.asarray([0.8, 0.2, 0.7, 0.1])
    assert average_precision(target, probability) == 1.0
    assert roc_auc(target, probability) == 1.0
    print("V55_P0_MULTISEED_SELF_TEST_OK")


def main():
    seed_arrays = {seed: load_seed(seed) for seed in SEEDS}
    arrays, climatology = aligned_reference(seed_arrays)
    probabilities = {
        f"seed{seed}": seed_arrays[seed]["horizon_probability"] for seed in SEEDS
    }
    probabilities["three_seed_equal_mean"] = np.mean(
        [probabilities[f"seed{seed}"] for seed in SEEDS], axis=0
    )
    for name, probability in probabilities.items():
        if probability.shape != (1918, 3) or not np.isfinite(probability).all():
            raise ValueError(f"invalid probability array: {name}")

    events = event_table()
    metrics = {
        name: evaluate_probability(probability, arrays, climatology, events)
        for name, probability in probabilities.items()
    }
    dispersion = {}
    for horizon in map(str, HORIZONS):
        dispersion[horizon] = {}
        for metric in ("pr_auc", "roc_auc", "brier", "bss"):
            values = [metrics[f"seed{seed}"][horizon][metric] for seed in SEEDS]
            dispersion[horizon][metric] = {
                "minimum": float(min(values)), "maximum": float(max(values)),
                "range": float(max(values) - min(values)),
                "standard_deviation": float(np.std(values)),
            }

    result = {
        "status": "V55_P0_MULTISEED_SCRATCH_OK",
        "protocol": {
            "seeds": list(SEEDS), "folds": list(FOLDS),
            "fixed_epochs": 21, "fixed_threshold": THRESHOLD,
            "ensemble": "equal arithmetic mean; no seed selection",
            "alignment_keys": list(ALIGNMENT_KEYS),
            "bootstrap": "1000 winter-cluster resamples",
            "outer_test_selection": "forbidden",
        },
        "alignment": {"status": "PASS", "n_samples": 1918, "n_events": 16},
        "metrics": metrics,
        "seed_dispersion": dispersion,
        "winter_cluster_bootstrap_95ci": winter_bootstrap(
            probabilities, arrays, climatology
        ),
    }
    np.savez_compressed(
        OUTPUT_NPZ,
        forecast_origin=arrays["forecast_origin"], start=arrays["start"],
        winter=arrays["winter"], target_class=arrays["target_class"],
        fold=arrays["fold"], horizons_days=np.asarray(HORIZONS),
        seed42_probability=probabilities["seed42"],
        seed73_probability=probabilities["seed73"],
        seed2026_probability=probabilities["seed2026"],
        three_seed_equal_mean_probability=probabilities["three_seed_equal_mean"],
        climatology_probability=climatology,
    )
    OUTPUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    rows = []
    for model, horizon_blocks in metrics.items():
        for horizon, block in horizon_blocks.items():
            rows.append({
                "model": model, "horizon_days": horizon,
                "pr_auc": block["pr_auc"], "roc_auc": block["roc_auc"],
                "brier": block["brier"], "bss": block["bss"],
                "CSI": block["fixed_0.20"]["CSI"],
                "POD": block["fixed_0.20"]["POD"],
                "FAR": block["fixed_0.20"]["FAR"],
                "events_hit": block["event_metrics"]["events_hit"],
                "mean_lead_days": block["event_metrics"]["mean_first_alarm_lead_days"],
                "false_episodes": block["event_metrics"]["false_alarm_episodes_total"],
            })
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({
        "status": result["status"], "alignment": result["alignment"],
        "metrics": metrics, "seed_dispersion": dispersion,
    }, indent=2))


if __name__ == "__main__":
    import sys
    self_test() if "--self-test" in sys.argv else main()
