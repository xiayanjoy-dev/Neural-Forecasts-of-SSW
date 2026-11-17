#!/usr/bin/env python3
"""Frozen paper-1 revision diagnostics; no fitting or model selection."""

import csv
import json
from pathlib import Path

import numpy as np

from analyze_v59_state_dependent_predictability import (
    FEATURES,
    false_episodes,
    load_aligned,
)
ROOT = Path(__file__).resolve().parent
V55_PRED = ROOT / "v55_p0_multiseed_scratch_predictions.npz"
V55_RESULT = ROOT / "v55_p0_multiseed_scratch_results.json"
S2S_PRED = ROOT / "v55_p1_s2s_probabilities.npz"
P3_PRED = ROOT / "v52_p3_physical_baselines_predictions.npz"
V63_RESULT = ROOT / "v63_common_origin_probability_diagnostic.json"
V59_RESULT = ROOT / "v59_state_dependent_predictability.json"
P7_RESULT = ROOT / "v57_p7_forward_evaluation.json"

OUT_JSON = ROOT / "ssw_paper1_revision_p0_results.json"
OUT_COMMON_CSV = ROOT / "ssw_paper1_revision_common_origin_metrics.csv"
OUT_THRESHOLD_CSV = ROOT / "ssw_paper1_revision_threshold_sensitivity.csv"
OUT_ASSOC_CSV = ROOT / "ssw_paper1_revision_event_associations.csv"
OUT_CONTRAST_CSV = ROOT / "ssw_paper1_revision_state_contrasts.csv"
OUT_NPZ = ROOT / "ssw_paper1_revision_common_origin_predictions.npz"

KEYS = ("forecast_origin", "start", "winter", "target_class")
HORIZONS = (5, 10, 15)
THRESHOLD = 0.20
THRESHOLD_RANGE = (0.10, 0.15, 0.20, 0.25, 0.30)
BOOTSTRAP_DRAWS = 1000
CONTRAST_DRAWS = 10000
PERMUTATION_DRAWS = 100000
MERGE_STEPS = 20
FORWARD_TRAIN_PRIOR = {
    "A": np.asarray([0.032526671870934165, 0.06505334374186833,
                     0.09679937548790007]),
    "B": np.asarray([0.031440891124685594, 0.06288178224937119,
                     0.09378368666906216]),
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
        while right < target.size and sorted_probability[right] == sorted_probability[left]:
            right += 1
        ranks[order[left:right]] = 0.5 * (left + 1 + right)
        left = right
    positive = target == 1
    n_positive = int(positive.sum())
    n_negative = int(target.size - n_positive)
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
                 / (n_positive * n_negative))


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
    rows_ = []
    for event in selected:
        onset = int(event["onset_step"])
        eligible = (origin < onset) & (onset <= origin + event["horizon"] * 4)
        alarms = eligible & predicted
        rows_.append({
            "winter": int(event["winter"]), "onset_step": onset,
            "hit": bool(np.any(alarms)),
            "first_alarm_lead_days": (
                float(np.max((onset - origin[alarms]) / 4.0)) if np.any(alarms) else None),
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
    leads = [row["first_alarm_lead_days"] for row in rows_
             if row["first_alarm_lead_days"] is not None]
    return {
        "events_hit": int(sum(row["hit"] for row in rows_)),
        "n_events": len(rows_),
        "mean_first_alarm_lead_days": float(np.mean(leads)) if leads else None,
        "false_alarm_episodes_total": int(sum(false_by_winter.values())),
        "false_alarm_episodes_by_winter": false_by_winter,
        "events": rows_,
    }


def reliability(probability, target):
    edges = np.linspace(0.0, 1.0, 11)
    output = []
    for index in range(10):
        mask = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == 9
            else probability < edges[index + 1])
        output.append({
            "lower": float(edges[index]), "upper": float(edges[index + 1]),
            "count": int(mask.sum()),
            "mean_probability": float(probability[mask].mean()) if np.any(mask) else None,
            "observed_frequency": float(target[mask].mean()) if np.any(mask) else None,
        })
    return output


def load_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def rows(arrays):
    return [tuple(int(arrays[key][index]) for key in KEYS)
            for index in range(len(arrays[KEYS[0]]))]


def select_by_keys(arrays, target_rows):
    index = {row: position for position, row in enumerate(rows(arrays))}
    if len(index) != len(arrays[KEYS[0]]):
        raise RuntimeError("duplicate alignment key")
    missing = [row for row in target_rows if row not in index]
    if missing:
        raise RuntimeError(f"missing alignment row: {missing[0]}")
    position = np.asarray([index[row] for row in target_rows])
    length = len(arrays[KEYS[0]])
    selected = {
        key: value[position] for key, value in arrays.items()
        if value.ndim and value.shape[0] == length
    }
    for key, expected in zip(KEYS, zip(*target_rows)):
        np.testing.assert_array_equal(selected[key], np.asarray(expected))
    return selected


def brier_decomposition(probability, target):
    bins = reliability(probability, target)
    prevalence = float(target.mean())
    rel = 0.0
    res = 0.0
    for row in bins:
        if row["count"] == 0:
            continue
        weight = row["count"] / len(target)
        rel += weight * (row["mean_probability"] - row["observed_frequency"]) ** 2
        res += weight * (row["observed_frequency"] - prevalence) ** 2
    return {
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": float(prevalence * (1.0 - prevalence)),
        "bins": bins,
    }


def events_for_horizon(benchmark, horizon):
    source = benchmark["pooled"][str(horizon)]["ecmwf_s2s"]["event_metrics"]["events"]
    return [{"winter": int(row["winter"]), "onset_step": int(row["onset_step"]),
             "horizon": int(horizon)} for row in source]


def evaluate(probability, arrays, reference, events, threshold=THRESHOLD):
    output = {}
    for class_index, horizon in enumerate(HORIZONS):
        target = (arrays["target_class"] <= class_index).astype(int)
        prob = probability[:, class_index]
        block = metric_block(prob, target, reference[:, class_index])
        block["fixed_threshold"] = decision_metrics(prob >= threshold, target)
        block["event_metrics"] = event_metrics(
            prob >= threshold, target, arrays["forecast_origin"], arrays["winter"],
            events[str(horizon)],
        )
        block["probability_decomposition"] = brier_decomposition(prob, target)
        output[str(horizon)] = block
    return output


def event_contributions(block, winters):
    output = {}
    for winter in winters:
        event_rows = [row for row in block["events"] if int(row["winter"]) == int(winter)]
        leads = [float(row["first_alarm_lead_days"]) for row in event_rows
                 if row["first_alarm_lead_days"] is not None]
        output[int(winter)] = {
            "events_hit": int(sum(bool(row["hit"]) for row in event_rows)),
            "n_events": int(len(event_rows)),
            "leads": leads,
            "false_episodes": int(block["false_alarm_episodes_by_winter"].get(str(int(winter)), 0)),
        }
    return output


def percentile(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]


def paired_winter_bootstrap(probabilities, arrays, reference, events, evaluations):
    rng = np.random.default_rng(6421)
    winters = np.unique(arrays["winter"])
    metrics = ("pr_auc", "roc_auc", "brier", "bss")
    samples = {
        str(h): {model: {metric: [] for metric in metrics} | {
            "events_hit": [], "event_hit_fraction": [], "mean_lead_days": [],
            "false_episodes": [],
        } for model in probabilities} for h in HORIZONS
    }
    differences = {str(h): {name: [] for name in (
        "ecmwf_minus_neural_pr_auc", "ecmwf_minus_neural_brier",
        "ecmwf_minus_neural_bss", "ecmwf_minus_neural_events_hit",
        "ecmwf_minus_neural_false_episodes",
    )} for h in HORIZONS}
    contributions = {
        str(h): {model: event_contributions(
            evaluations[model][str(h)]["event_metrics"], winters)
            for model in probabilities}
        for h in HORIZONS
    }
    for _ in range(BOOTSTRAP_DRAWS):
        chosen = rng.choice(winters, size=len(winters), replace=True)
        index = np.concatenate([np.where(arrays["winter"] == winter)[0] for winter in chosen])
        for class_index, horizon in enumerate(HORIZONS):
            target = (arrays["target_class"][index] <= class_index).astype(int)
            if np.unique(target).size < 2:
                continue
            per_model = {}
            for model, probability in probabilities.items():
                block = metric_block(
                    probability[index, class_index], target, reference[index, class_index])
                counts = [contributions[str(horizon)][model][int(w)] for w in chosen]
                n_event = sum(row["n_events"] for row in counts)
                n_hit = sum(row["events_hit"] for row in counts)
                leads = [lead for row in counts for lead in row["leads"]]
                false = sum(row["false_episodes"] for row in counts)
                block.update({
                    "events_hit": n_hit,
                    "event_hit_fraction": n_hit / n_event if n_event else np.nan,
                    "mean_lead_days": np.mean(leads) if leads else np.nan,
                    "false_episodes": false,
                })
                per_model[model] = block
                for name in metrics + ("events_hit", "event_hit_fraction",
                                       "mean_lead_days", "false_episodes"):
                    samples[str(horizon)][model][name].append(block[name])
            ec = per_model["ecmwf_s2s"]
            neural = per_model["three_seed_equal_mean"]
            differences[str(horizon)]["ecmwf_minus_neural_pr_auc"].append(
                ec["pr_auc"] - neural["pr_auc"])
            differences[str(horizon)]["ecmwf_minus_neural_brier"].append(
                ec["brier"] - neural["brier"])
            differences[str(horizon)]["ecmwf_minus_neural_bss"].append(
                ec["bss"] - neural["bss"])
            differences[str(horizon)]["ecmwf_minus_neural_events_hit"].append(
                ec["events_hit"] - neural["events_hit"])
            differences[str(horizon)]["ecmwf_minus_neural_false_episodes"].append(
                ec["false_episodes"] - neural["false_episodes"])
    return {
        "model_95ci": {h: {model: {metric: percentile(value)
                                     for metric, value in block.items()}
                               for model, block in models.items()}
                         for h, models in samples.items()},
        "paired_difference_95ci": {h: {name: percentile(value)
                                        for name, value in block.items()}
                                      for h, block in differences.items()},
    }


def full_event_bootstrap(evaluation, arrays):
    rng = np.random.default_rng(6422)
    winters = np.unique(arrays["winter"])
    output = {}
    for horizon in HORIZONS:
        contributions = event_contributions(evaluation[str(horizon)]["event_metrics"], winters)
        values = {name: [] for name in (
            "events_hit", "event_hit_fraction", "mean_lead_days", "false_episodes")}
        for _ in range(BOOTSTRAP_DRAWS):
            chosen = rng.choice(winters, size=len(winters), replace=True)
            rows_ = [contributions[int(w)] for w in chosen]
            n_event = sum(row["n_events"] for row in rows_)
            n_hit = sum(row["events_hit"] for row in rows_)
            leads = [lead for row in rows_ for lead in row["leads"]]
            values["events_hit"].append(n_hit)
            values["event_hit_fraction"].append(n_hit / n_event if n_event else np.nan)
            values["mean_lead_days"].append(np.mean(leads) if leads else np.nan)
            values["false_episodes"].append(sum(row["false_episodes"] for row in rows_))
        output[str(horizon)] = {name: percentile(value) for name, value in values.items()}
    return output


def threshold_sensitivity(v55, results):
    probability = v55["three_seed_equal_mean_probability"]
    archived = results["metrics"]["three_seed_equal_mean"]
    event_templates = {
        str(h): [{"winter": int(row["winter"]), "onset_step": int(row["onset_step"]),
                  "horizon": h} for row in archived[str(h)]["event_metrics"]["events"]]
        for h in HORIZONS
    }
    output = {}
    for threshold in THRESHOLD_RANGE:
        output[f"{threshold:.2f}"] = {}
        for class_index, horizon in enumerate(HORIZONS):
            target = (v55["target_class"] <= class_index).astype(int)
            predicted = probability[:, class_index] >= threshold
            event = event_metrics(predicted, target, v55["forecast_origin"], v55["winter"],
                                  event_templates[str(horizon)])
            output[f"{threshold:.2f}"][str(horizon)] = {
                "decision": decision_metrics(predicted, target),
                "events_hit": event["events_hit"],
                "n_events": event["n_events"],
                "mean_first_alarm_lead_days": event["mean_first_alarm_lead_days"],
                "false_alarm_episodes": event["false_alarm_episodes_total"],
            }
    for horizon in HORIZONS:
        current = output["0.20"][str(horizon)]
        expected = archived[str(horizon)]
        assert current["events_hit"] == expected["event_metrics"]["events_hit"]
        assert current["false_alarm_episodes"] == expected["event_metrics"]["false_alarm_episodes_total"]
        for metric in ("CSI", "POD", "FAR"):
            np.testing.assert_allclose(current["decision"][metric], expected["fixed_0.20"][metric])
    return output


def outer_fold_inventory(v55, results):
    archived = results["metrics"]["three_seed_equal_mean"]
    output = {}
    for fold in ("A", "B", "C"):
        mask = v55["fold"] == fold
        output[fold] = {}
        for class_index, horizon in enumerate(HORIZONS):
            target = (v55["target_class"][mask] <= class_index).astype(int)
            event_rows = archived[str(horizon)]["event_metrics"]["events"]
            fold_winters = set(v55["winter"][mask].tolist())
            output[fold][str(horizon)] = {
                "forecast_origins": int(mask.sum()),
                "positive_origin_count": int(target.sum()),
                "prevalence": float(target.mean()),
                "events": int(sum(int(row["winter"]) in fold_winters for row in event_rows)),
            }
    return output


def forward_training_reference():
    archived = json.loads(P7_RESULT.read_text())
    output = {}
    for arm in ("A", "B"):
        seed_arrays = []
        reference = None
        for seed in (42, 73, 2026):
            path = ROOT / "p7_artifacts" / (
                f"v57_p7_arm{arm}_forward_seed{seed}_fixed_epoch21_predictions.npz")
            arrays = load_npz(path)
            if reference is None:
                reference = arrays
            else:
                arrays = select_by_keys(arrays, rows(reference))
                for key in KEYS:
                    np.testing.assert_array_equal(reference[key], arrays[key])
            seed_arrays.append(arrays["horizon_probability"])
        probability = np.mean(seed_arrays, axis=0)
        output[arm] = {}
        for class_index, horizon in enumerate(HORIZONS):
            target = (reference["target_class"] <= class_index).astype(int)
            ref_probability = np.full(len(target), FORWARD_TRAIN_PRIOR[arm][class_index])
            block = metric_block(probability[:, class_index], target, ref_probability)
            old = archived["arms"][arm]["metrics"]["three_seed_equal_mean"][str(horizon)]
            np.testing.assert_allclose(block["pr_auc"], old["pr_auc"], atol=1e-12)
            np.testing.assert_allclose(block["brier"], old["brier"], atol=1e-9)
            output[arm][str(horizon)] = {
                **block,
                "training_climatology_probability": float(FORWARD_TRAIN_PRIOR[arm][class_index]),
                "old_test_prevalence_bss": float(old["bss"]),
                "old_test_prevalence_reference": "diagnostic only; replaced in manuscript",
            }
    return output


def average_ranks(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    left = 0
    while left < len(values):
        right = left + 1
        while right < len(values) and values[order[right]] == values[order[left]]:
            right += 1
        ranks[order[left:right]] = 0.5 * (left + 1 + right)
        left = right
    return ranks


def bh_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 1.0
    for rank_index in range(len(pvalues) - 1, -1, -1):
        item = order[rank_index]
        rank = rank_index + 1
        running = min(running, pvalues[item] * len(pvalues) / rank)
        adjusted[item] = running
    return adjusted


def association_multiplicity(v59):
    event_probability = np.asarray([row["max_probability"] for row in v59["events"]])
    matrix = np.asarray([[row[f"{feature}_z"] for feature in FEATURES]
                         for row in v59["events"]], dtype=float)
    ranked_x = np.column_stack([average_ranks(matrix[:, i]) for i in range(matrix.shape[1])])
    ranked_y = average_ranks(event_probability)
    ranked_x = (ranked_x - ranked_x.mean(axis=0)) / ranked_x.std(axis=0, ddof=1)
    ranked_y = (ranked_y - ranked_y.mean()) / ranked_y.std(ddof=1)
    observed = ranked_x.T @ ranked_y / (len(ranked_y) - 1)
    rng = np.random.default_rng(6423)
    exceed = np.zeros(len(FEATURES), dtype=np.int64)
    max_exceed = np.zeros(len(FEATURES), dtype=np.int64)
    processed = 0
    chunk = 5000
    while processed < PERMUTATION_DRAWS:
        size = min(chunk, PERMUTATION_DRAWS - processed)
        permuted = np.column_stack([rng.permutation(ranked_y) for _ in range(size)])
        rho = ranked_x.T @ permuted / (len(ranked_y) - 1)
        exceed += np.sum(np.abs(rho) >= np.abs(observed)[:, None], axis=1)
        maximum = np.max(np.abs(rho), axis=0)
        max_exceed += np.sum(maximum[None, :] >= np.abs(observed)[:, None], axis=1)
        processed += size
    raw = (exceed + 1) / (PERMUTATION_DRAWS + 1)
    family = (max_exceed + 1) / (PERMUTATION_DRAWS + 1)
    adjusted = bh_adjust(raw)
    old = {row["feature"]: row for row in v59["event_probability_associations"]}
    return [{
        "feature": feature,
        "spearman": float(observed[index]),
        "bootstrap_ci95": old[feature]["ci95"],
        "permutation_p_two_sided": float(raw[index]),
        "bh_q": float(adjusted[index]),
        "max_abs_rho_fwer_p": float(family[index]),
    } for index, feature in enumerate(FEATURES)]


def cluster_contrast(values_a, winter_a, values_b, winter_b, seed):
    rng = np.random.default_rng(seed)
    winters = np.unique(np.concatenate([winter_a, winter_b]))
    samples = []
    for _ in range(CONTRAST_DRAWS):
        chosen = rng.choice(winters, size=len(winters), replace=True)
        a = np.concatenate([values_a[winter_a == winter] for winter in chosen])
        b = np.concatenate([values_b[winter_b == winter] for winter in chosen])
        if len(a) and len(b):
            samples.append(a.mean(axis=0) - b.mean(axis=0))
    samples = np.asarray(samples)
    return np.percentile(samples, [2.5, 97.5], axis=0).T


def state_contrasts(v59):
    prediction, _, z = load_aligned()
    events = v59["events"]
    event_values = np.asarray([[row[f"{feature}_z"] for feature in FEATURES]
                               for row in events])
    event_winters = np.asarray([row["winter"] for row in events])
    hit = np.asarray([row["hit"] for row in events], dtype=bool)
    false = false_episodes(prediction)
    false_values = z[np.asarray([row["peak_index"] for row in false])]
    false_winters = np.asarray([row["winter"] for row in false])
    hm_ci = cluster_contrast(event_values[hit], event_winters[hit],
                             event_values[~hit], event_winters[~hit], 6424)
    fh_ci = cluster_contrast(false_values, false_winters,
                             event_values[hit], event_winters[hit], 6425)
    output = []
    for index, feature in enumerate(FEATURES):
        output.append({
            "feature": feature,
            "hit_mean_z": float(event_values[hit, index].mean()),
            "miss_mean_z": float(event_values[~hit, index].mean()),
            "hit_minus_miss": float(event_values[hit, index].mean() -
                                    event_values[~hit, index].mean()),
            "hit_minus_miss_ci95": hm_ci[index].tolist(),
            "false_mean_z": float(false_values[:, index].mean()),
            "false_minus_hit": float(false_values[:, index].mean() -
                                     event_values[hit, index].mean()),
            "false_minus_hit_ci95": fh_ci[index].tolist(),
        })
    return output


def write_csv(path, data):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def main():
    v55 = load_npz(V55_PRED)
    s2s = load_npz(S2S_PRED)
    p3 = load_npz(P3_PRED)
    v55_result = json.loads(V55_RESULT.read_text())
    v63 = json.loads(V63_RESULT.read_text())
    v59 = json.loads(V59_RESULT.read_text())
    benchmark = json.loads((ROOT / "v55_p1_s2s_benchmark_results.json").read_text())
    target_rows = rows(s2s)
    if len(target_rows) != 454 or len(set(target_rows)) != 454:
        raise RuntimeError("common-origin inventory changed")
    v55_common = select_by_keys(v55, target_rows)
    p3_common = select_by_keys(p3, target_rows)
    for key in KEYS:
        np.testing.assert_array_equal(s2s[key], v55_common[key])
        np.testing.assert_array_equal(s2s[key], p3_common[key])
    np.testing.assert_array_equal(s2s["fold"], v55_common["fold"])
    np.testing.assert_array_equal(s2s["fold"], p3_common["fold"])
    probabilities = {
        "ecmwf_s2s": s2s["ecmwf_s2s_probability"],
        "three_seed_equal_mean": v55_common["three_seed_equal_mean_probability"],
        "seed42": v55_common["seed42_probability"],
        "physical_logistic": p3_common["physical_logistic_probability"],
    }
    for name, probability in probabilities.items():
        if probability.shape != (454, 3) or not np.isfinite(probability).all():
            raise RuntimeError(f"invalid common-origin probabilities: {name}")
    reference = v55_common["climatology_probability"]
    events = {str(h): events_for_horizon(benchmark, h) for h in HORIZONS}
    common = {name: evaluate(prob, s2s, reference, events)
              for name, prob in probabilities.items()}
    for index, horizon in enumerate(HORIZONS):
        archived = v63["probability_reports"][str(horizon)]["scratch"]
        current = common["seed42"][str(horizon)]
        np.testing.assert_allclose(current["brier"], archived["direct_brier"], atol=1e-12)
        np.testing.assert_allclose(current["pr_auc"], archived["pr_auc"], atol=1e-12)
        np.testing.assert_allclose(current["bss"], archived["bss"], atol=1e-12)
    bootstrap = paired_winter_bootstrap(probabilities, s2s, reference, events, common)
    threshold = threshold_sensitivity(v55, v55_result)
    forward_reference = forward_training_reference()
    associations = association_multiplicity(v59)
    contrasts = state_contrasts(v59)
    full_ensemble = v55_result["metrics"]["three_seed_equal_mean"]
    full_event_ci = full_event_bootstrap(full_ensemble, v55)
    result = {
        "status": "SSW_PAPER1_REVISION_P0_OK",
        "contract": "ssw_paper1_revision_analysis_contract.md",
        "alignment": {"status": "PASS", "common_origins": 454,
                      "covered_events": 11, "keys": list(KEYS)},
        "common_origin": common,
        "common_origin_winter_bootstrap": bootstrap,
        "full_outer_prevalence": {
            str(h): full_ensemble[str(h)]["prevalence"] for h in HORIZONS},
        "outer_fold_inventory": outer_fold_inventory(v55, v55_result),
        "full_outer_event_metric_95ci": full_event_ci,
        "threshold_sensitivity": threshold,
        "forward_training_climatology_reference": forward_reference,
        "event_association_multiplicity": associations,
        "state_contrasts": contrasts,
        "claim_boundary": (
            "Frozen post hoc diagnostics only: no fitting, threshold selection, seed "
            "selection, calibration, feature selection, or causal attribution."
        ),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    common_rows = []
    for model, horizon_blocks in common.items():
        for horizon, block in horizon_blocks.items():
            event = block["event_metrics"]
            common_rows.append({
                "model": model, "horizon_days": horizon,
                "prevalence": block["prevalence"], "pr_auc": block["pr_auc"],
                "roc_auc": block["roc_auc"], "brier": block["brier"],
                "bss": block["bss"], "CSI": block["fixed_threshold"]["CSI"],
                "POD": block["fixed_threshold"]["POD"],
                "FAR": block["fixed_threshold"]["FAR"],
                "events_hit": event["events_hit"], "n_events": event["n_events"],
                "mean_lead_days": event["mean_first_alarm_lead_days"],
                "false_episodes": event["false_alarm_episodes_total"],
                "reliability": block["probability_decomposition"]["reliability"],
                "resolution": block["probability_decomposition"]["resolution"],
            })
    threshold_rows = []
    for threshold_value, horizon_blocks in threshold.items():
        for horizon, block in horizon_blocks.items():
            threshold_rows.append({"threshold": threshold_value, "horizon_days": horizon,
                                   **block["decision"], "events_hit": block["events_hit"],
                                   "n_events": block["n_events"],
                                   "mean_lead_days": block["mean_first_alarm_lead_days"],
                                   "false_episodes": block["false_alarm_episodes"]})
    association_rows = [{**row, "bootstrap_ci95": json.dumps(row["bootstrap_ci95"])}
                        for row in associations]
    contrast_rows = [{**row,
                      "hit_minus_miss_ci95": json.dumps(row["hit_minus_miss_ci95"]),
                      "false_minus_hit_ci95": json.dumps(row["false_minus_hit_ci95"])}
                     for row in contrasts]
    write_csv(OUT_COMMON_CSV, common_rows)
    write_csv(OUT_THRESHOLD_CSV, threshold_rows)
    write_csv(OUT_ASSOC_CSV, association_rows)
    write_csv(OUT_CONTRAST_CSV, contrast_rows)
    np.savez_compressed(
        OUT_NPZ,
        forecast_origin=s2s["forecast_origin"], start=s2s["start"],
        winter=s2s["winter"], target_class=s2s["target_class"], fold=s2s["fold"],
        horizons_days=np.asarray(HORIZONS),
        ecmwf_s2s_probability=probabilities["ecmwf_s2s"],
        three_seed_equal_mean_probability=probabilities["three_seed_equal_mean"],
        seed42_probability=probabilities["seed42"],
        physical_logistic_probability=probabilities["physical_logistic"],
        climatology_probability=reference,
    )
    print(json.dumps({
        "status": result["status"],
        "common_origin_summary": common_rows,
        "association_summary": associations,
    }, indent=2))


def self_test():
    adjusted = bh_adjust([0.01, 0.02, 0.8])
    np.testing.assert_allclose(adjusted, [0.03, 0.03, 0.8])
    print("SSW_PAPER1_REVISION_P0_SELF_TEST_OK")


if __name__ == "__main__":
    import sys
    self_test() if "--self-test" in sys.argv else main()
