#!/usr/bin/env python3
"""Winter-block temporal dataset-shift diagnostic for SSW Paper 1."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
PHYSICAL = ROOT / "v52_p2_physical_context.json"
PREDICTIONS = ROOT / "v55_p0_multiseed_scratch_predictions.npz"
METRICS = ROOT / "ssw_paper1_revision_v4_results.json"
OUT_JSON = ROOT / "ssw_paper1_temporal_shift_results.json"
OUT_CSV = ROOT / "ssw_paper1_temporal_shift_source.csv"
OUT_PNG = ROOT / "ssw_paper1_figS3_temporal_shift.png"
OUT_PDF = ROOT / "ssw_paper1_figS3_temporal_shift.pdf"

FOLDS = ("A", "B", "C")
HORIZONS = (5, 10, 15)
FEATURES = (
    "u10_now", "u10_delta5", "u10_delta10",
    "t10_pcap_delta5", "t30_pcap_delta5", "t50_pcap_delta5",
    "heat_flux100_5d", "heat_flux100_10d",
    "epdiv_upper_5d", "epdiv_lower_5d", "ep_f_p_100_5d", "qbo30_5d",
)
LABELS = (
    "U10", "dU10 5 d", "dU10 10 d", "dT10 5 d", "dT30 5 d",
    "dT50 5 d", "HF100 5 d", "HF100 10 d", "EPdiv upper",
    "EPdiv lower", "EP Fp 100", "QBO30",
)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 6510


def bh_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    for reverse_rank in range(len(order) - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = running
    return adjusted


def pairwise_distance(values):
    delta = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def energy_statistic(distance, first, second):
    cross = distance[np.ix_(first, second)].mean()
    within_first = distance[np.ix_(first, first)].mean()
    within_second = distance[np.ix_(second, second)].mean()
    return float(2.0 * cross - within_first - within_second)


def winter_summary(records, reducer="mean", background_only=False):
    rows = [row for row in records if not background_only or row["target_class"] == 4]
    output = []
    for fold in FOLDS:
        winters = sorted({int(row["winter"]) for row in rows if row["fold"] == fold})
        for winter in winters:
            selected = [row for row in rows
                        if row["fold"] == fold and int(row["winter"]) == winter]
            values = np.asarray([[row[name] for name in FEATURES] for row in selected],
                                dtype=float)
            if not np.isfinite(values).all() or values.size == 0:
                raise RuntimeError(f"invalid winter feature block {fold} {winter}")
            aggregate = values.mean(axis=0) if reducer == "mean" else np.median(values, axis=0)
            output.append({"fold": fold, "winter": winter, "n_origins": len(selected),
                           "values": aggregate})
    return output


def compare(summary, first_fold, second_fold, seed):
    first_rows = [row for row in summary if row["fold"] == first_fold]
    second_rows = [row for row in summary if row["fold"] == second_fold]
    first = np.asarray([row["values"] for row in first_rows], dtype=float)
    second = np.asarray([row["values"] for row in second_rows], dtype=float)
    combined = np.vstack([first, second])
    scale = combined.std(axis=0, ddof=1)
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise RuntimeError("invalid pooled winter scale")
    effect = (second.mean(axis=0) - first.mean(axis=0)) / scale
    z = (combined - combined.mean(axis=0)) / scale
    distance = pairwise_distance(z)
    n_first = len(first)
    all_index = np.arange(len(combined))
    observed_first = np.arange(n_first)
    observed_second = np.arange(n_first, len(combined))
    observed_energy = energy_statistic(distance, observed_first, observed_second)

    perm_effects = []
    perm_energy = []
    for choice in itertools.combinations(all_index, n_first):
        group_first = np.asarray(choice, dtype=int)
        group_second = np.setdiff1d(all_index, group_first, assume_unique=True)
        perm_effects.append(
            (combined[group_second].mean(axis=0) - combined[group_first].mean(axis=0)) / scale
        )
        perm_energy.append(energy_statistic(distance, group_first, group_second))
    perm_effects = np.asarray(perm_effects)
    perm_energy = np.asarray(perm_energy)
    p_value = np.mean(np.abs(perm_effects) >= np.abs(effect), axis=0)
    max_abs = np.max(np.abs(perm_effects), axis=1)
    family_p = np.mean(max_abs[:, None] >= np.abs(effect)[None, :], axis=0)
    q_value = bh_adjust(p_value)
    energy_p = float(np.mean(perm_energy >= observed_energy))

    rng = np.random.default_rng(seed)
    bootstrap = np.empty((BOOTSTRAP_DRAWS, len(FEATURES)), dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        a = first[rng.integers(0, len(first), len(first))]
        b = second[rng.integers(0, len(second), len(second))]
        bootstrap[draw] = (b.mean(axis=0) - a.mean(axis=0)) / scale
    lower, upper = np.quantile(bootstrap, [0.025, 0.975], axis=0)

    features = {}
    for index, name in enumerate(FEATURES):
        features[name] = {
            "second_minus_first_smd": float(effect[index]),
            "bootstrap_ci95": [float(lower[index]), float(upper[index])],
            "exact_p_two_sided": float(p_value[index]),
            "bh_q": float(q_value[index]),
            "max_abs_family_p": float(family_p[index]),
        }
    return {
        "first_fold": first_fold,
        "second_fold": second_fold,
        "n_first_winters": len(first),
        "n_second_winters": len(second),
        "n_exact_allocations": int(len(perm_energy)),
        "energy_distance": observed_energy,
        "energy_exact_p": energy_p,
        "features": features,
    }


def aligned_predictions(records):
    ordered = sorted(records, key=lambda row: (
        int(row["forecast_origin"]), int(row["start"]), int(row["winter"]),
        int(row["target_class"])))
    with np.load(PREDICTIONS, allow_pickle=False) as source:
        order = np.lexsort((source["target_class"], source["winter"], source["start"],
                            source["forecast_origin"]))
        for key in ("forecast_origin", "start", "winter", "target_class"):
            actual = np.asarray([row[key] for row in ordered])
            np.testing.assert_array_equal(actual, source[key][order])
        probability = source["three_seed_equal_mean_probability"][order]
        fold = source["fold"][order].astype(str)
        climatology = source["climatology_probability"][order]
    return ordered, fold, probability, climatology


def forecast_composition(records):
    ordered, fold, probability, climatology = aligned_predictions(records)
    target_class = np.asarray([row["target_class"] for row in ordered], dtype=int)
    frozen = json.loads(METRICS.read_text())["metrics"]
    output = {}
    for fold_name in FOLDS:
        mask = fold == fold_name
        output[fold_name] = {}
        for index, horizon in enumerate(HORIZONS):
            target = (target_class[mask] <= index).astype(float)
            prob = probability[mask, index]
            ref = climatology[mask, index]
            brier = float(np.mean((prob - target) ** 2))
            reference_brier = float(np.mean((ref - target) ** 2))
            block = frozen[fold_name][str(horizon)]["three_seed_equal_mean"]
            output[fold_name][str(horizon)] = {
                "n_origins": int(mask.sum()),
                "positive_origins": int(target.sum()),
                "prevalence": float(target.mean()),
                "mean_probability": float(prob.mean()),
                "mean_probability_minus_prevalence": float(prob.mean() - target.mean()),
                "brier": brier,
                "bss": float(1.0 - brier / reference_brier),
                "calibration_intercept": float(block["calibration_intercept"]),
                "calibration_slope": float(block["calibration_slope"]),
                "events_hit": int(block["event"]["events_hit"]),
                "n_events": int(block["event"]["n_events"]),
                "false_alarm_episodes": int(block["event"]["false_alarm_episodes_total"]),
            }
    return output


def write_source(primary, robustness, secondary, composition):
    rows = []
    for feature, block in primary["features"].items():
        rows.append({"panel": "a", "comparison": "C-minus-A", "subset": "all",
                     "aggregation": "winter_mean", "feature": feature,
                     **block})
    for name, block in {**robustness, **secondary}.items():
        rows.append({"panel": "b", "comparison": name, "subset": "summary",
                     "aggregation": "multivariate", "feature": "energy_distance",
                     "energy_distance": block["energy_distance"],
                     "energy_exact_p": block["energy_exact_p"],
                     "n_first_winters": block["n_first_winters"],
                     "n_second_winters": block["n_second_winters"]})
    for fold, horizons in composition.items():
        for horizon, block in horizons.items():
            rows.append({"panel": "c", "comparison": fold, "subset": "all",
                         "aggregation": "origin_probability", "feature": "forecast",
                         "horizon_days": horizon, **block})
    fields = sorted({key for row in rows for key in row})
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_figure(primary, robustness, secondary, composition):
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.8))
    axis = axes[0]
    effect = np.asarray([primary["features"][name]["second_minus_first_smd"]
                         for name in FEATURES])
    lower = np.asarray([primary["features"][name]["bootstrap_ci95"][0]
                        for name in FEATURES])
    upper = np.asarray([primary["features"][name]["bootstrap_ci95"][1]
                        for name in FEATURES])
    family = np.asarray([primary["features"][name]["max_abs_family_p"]
                         for name in FEATURES])
    y = np.arange(len(FEATURES))
    colors = np.where(family < 0.05, "#C44E52", "#4C78A8")
    axis.errorbar(effect, y, xerr=np.vstack([effect - lower, upper - effect]),
                  fmt="none", ecolor="#777777", elinewidth=0.9, capsize=2)
    axis.scatter(effect, y, c=colors, s=28, zorder=3)
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_yticks(y, LABELS)
    axis.invert_yaxis()
    axis.set_xlabel("Fold C minus A standardized difference")
    axis.set_title("Winter-mean physical states")

    axis = axes[1]
    blocks = {
        "A-C mean": primary,
        "A-C median": robustness["A-C_winter_median"],
        "A-C background": robustness["A-C_background_only"],
        "A-B mean": secondary["A-B_winter_mean"],
        "B-C mean": secondary["B-C_winter_mean"],
    }
    labels = list(blocks)
    values = [blocks[name]["energy_distance"] for name in labels]
    p_values = [blocks[name]["energy_exact_p"] for name in labels]
    bars = axis.barh(np.arange(len(labels)), values, color="#72A0C1")
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.invert_yaxis()
    axis.set_xlabel("Energy distance")
    axis.set_title("Exact winter-label tests")
    for bar, p_value in zip(bars, p_values):
        axis.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                  f"  p={p_value:.3f}", va="center", fontsize=8)

    axis = axes[2]
    colors = {5: "#4C78A8", 10: "#F28E2B", 15: "#59A14F"}
    markers = {"A": "o", "B": "s", "C": "^"}
    for fold in FOLDS:
        for horizon in HORIZONS:
            block = composition[fold][str(horizon)]
            axis.scatter(block["prevalence"], block["mean_probability"],
                         color=colors[horizon], marker=markers[fold], s=50,
                         edgecolor="black", linewidth=0.4)
            axis.text(block["prevalence"], block["mean_probability"],
                      f" {fold}{horizon}", fontsize=7, va="center")
    limit = max(max(block["prevalence"], block["mean_probability"])
                for horizons in composition.values() for block in horizons.values()) * 1.08
    axis.plot([0, limit], [0, limit], "--", color="#777777", linewidth=0.9)
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_xlabel("Observed target prevalence")
    axis.set_ylabel("Mean neural probability")
    axis.set_title("Fold probability level")

    for label, axis in zip("abc", axes):
        axis.text(-0.16, 1.04, label, transform=axis.transAxes,
                  fontweight="bold", va="bottom")
        axis.spines[["top", "right"]].set_visible(False)
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.86, wspace=0.55)
    figure.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    figure.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(figure)


def self_test():
    rows = []
    for fold, winters, shift in (("A", range(2000, 2005), 0.0),
                                 ("C", range(2011, 2022), 1.0)):
        for winter in winters:
            rows.append({"fold": fold, "winter": winter, "n_origins": 1,
                         "values": np.arange(len(FEATURES), dtype=float) + shift})
    result = compare(rows, "A", "C", BOOTSTRAP_SEED)
    assert result["energy_distance"] > 0
    assert result["n_exact_allocations"] == 4368
    print("SSW_PAPER1_TEMPORAL_SHIFT_SELF_TEST_OK")


def main():
    physical = json.loads(PHYSICAL.read_text())
    if physical.get("status") != "V52_P2_PHYSICAL_CONTEXT_OK":
        raise RuntimeError("physical context audit did not pass")
    if physical.get("n_forecast_origins") != 1918 or tuple(physical["features"]) != FEATURES:
        raise RuntimeError("physical feature contract changed")
    records = physical["records"]
    if not all(np.isfinite([row[name] for name in FEATURES]).all() for row in records):
        raise RuntimeError("non-finite physical descriptor")

    mean_all = winter_summary(records, "mean", False)
    median_all = winter_summary(records, "median", False)
    mean_background = winter_summary(records, "mean", True)
    primary = compare(mean_all, "A", "C", BOOTSTRAP_SEED)
    robustness = {
        "A-C_winter_median": compare(median_all, "A", "C", BOOTSTRAP_SEED + 1),
        "A-C_background_only": compare(mean_background, "A", "C", BOOTSTRAP_SEED + 2),
    }
    secondary = {
        "A-B_winter_mean": compare(mean_all, "A", "B", BOOTSTRAP_SEED + 3),
        "B-C_winter_mean": compare(mean_all, "B", "C", BOOTSTRAP_SEED + 4),
    }
    composition = forecast_composition(records)
    result = {
        "status": "SSW_PAPER1_TEMPORAL_SHIFT_OK",
        "registration": "ssw_paper1_temporal_shift_registration.md",
        "inventory": {"origins": 1918, "features": list(FEATURES),
                      "winters_by_fold": {fold: len({row["winter"] for row in records
                                                     if row["fold"] == fold})
                                          for fold in FOLDS}},
        "primary": primary,
        "robustness": robustness,
        "secondary": secondary,
        "forecast_composition": composition,
        "interpretation": {
            "dataset_shift_supported": bool(primary["energy_exact_p"] < 0.05),
            "family_controlled_features": [name for name, block in primary["features"].items()
                                            if block["max_abs_family_p"] < 0.05],
            "allowed": "Temporal dataset shift across frozen outer-test periods.",
            "forbidden": "Confirmed covariate shift or attribution to a physical forcing.",
        },
        "claim_boundary": (
            "Post hoc winter-block outer-period diagnostic; it does not test invariant P(Y|X), "
            "does not contain complete fold-training feature distributions, and cannot select "
            "a model, threshold or physical cause."
        ),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    write_source(primary, robustness, secondary, composition)
    make_figure(primary, robustness, secondary, composition)
    print(json.dumps({
        "status": result["status"],
        "primary_energy": primary["energy_distance"],
        "primary_energy_p": primary["energy_exact_p"],
        "family_controlled_features": result["interpretation"]["family_controlled_features"],
        "outputs": [OUT_JSON.name, OUT_CSV.name, OUT_PNG.name, OUT_PDF.name],
    }, indent=2))


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
