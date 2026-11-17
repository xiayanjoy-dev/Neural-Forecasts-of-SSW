#!/usr/bin/env python3
"""Lead-dependent physical trajectories for all frozen v55 SSW events."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "v55_p0_multiseed_scratch_predictions.npz"
RESULTS = ROOT / "v55_p0_multiseed_scratch_results.json"
PHYSICAL = ROOT / "v52_p2_physical_context.json"
V59 = ROOT / "v59_state_dependent_predictability.json"
OUT_JSON = ROOT / "v62_all_event_lead_trajectory.json"
OUT_EVENTS = ROOT / "v62_all_event_lead_trajectory_events.csv"
OUT_SOURCE = ROOT / "v62_all_event_lead_trajectory_source.csv"
OUT_PNG = ROOT / "v62_all_event_lead_trajectory.png"
OUT_PDF = ROOT / "v62_all_event_lead_trajectory.pdf"
OUT_CAPTION = ROOT / "v62_all_event_lead_trajectory_caption.md"

KEYS = ("forecast_origin", "start", "winter", "target_class")
LEADS = (15, 10, 5)
FEATURES = (
    ("u10_now", "U10"),
    ("u10_delta5", "dU10 (5 d)"),
    ("t10_pcap_delta5", "dT10"),
    ("heat_flux100_5d", "HF100 (5 d)"),
    ("epdiv_lower_5d", "EP div. lower"),
    ("qbo30_5d", "QBO30"),
)
THRESHOLD = 0.20
BOOTSTRAP_DRAWS = 1000


def key_from_prediction(prediction, index):
    return tuple(int(prediction[key][index]) for key in KEYS)


def trajectory_lengths(z_by_lead):
    z15, z10, z5 = (z_by_lead[lead] for lead in LEADS)
    direct = float(np.linalg.norm(z5 - z15))
    first = float(np.linalg.norm(z10 - z15))
    second = float(np.linalg.norm(z5 - z10))
    return {
        "direct_15_to_5": direct,
        "path_15_to_10_to_5": first + second,
        "segment_15_to_10": first,
        "segment_10_to_5": second,
    }


def bootstrap_group_difference(values, hit, seed=6200):
    values = np.asarray(values, dtype=float)
    hit = np.asarray(hit, dtype=bool)
    if hit.sum() == 0 or (~hit).sum() == 0:
        return None
    rng = np.random.default_rng(seed)
    hit_values, miss_values = values[hit], values[~hit]
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        draws[draw] = (
            rng.choice(miss_values, size=miss_values.size, replace=True).mean()
            - rng.choice(hit_values, size=hit_values.size, replace=True).mean()
        )
    return {
        "miss_minus_hit": float(miss_values.mean() - hit_values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "n_hit": int(hit_values.size),
        "n_miss": int(miss_values.size),
        "unit": "standardized-state Euclidean distance",
    }


def event_labels(events):
    counts = {}
    labels = []
    for event in events:
        winter = event["winter"]
        counts[winter] = counts.get(winter, 0) + 1
        labels.append(str(winter) if counts[winter] == 1 else f"{winter}-{counts[winter]}")
    return labels


def make_figure(events):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 7.5,
        "ytick.labelsize": 8, "legend.fontsize": 7.5, "pdf.fonttype": 42,
    })
    figure, axes = plt.subplots(2, 2, figsize=(7.35, 6.3))
    hit_color, miss_color, highlight = "#3977B8", "#C44E52", "#202020"
    labels = event_labels(events)
    hits = np.asarray([event["five_day_hit"] for event in events], dtype=bool)
    paths = np.asarray([event["trajectory"]["direct_15_to_5"] for event in events])
    probability5 = np.asarray([event["states"]["5"]["matching_probability"] for event in events])
    target_index = next(index for index, event in enumerate(events) if event["winter"] == 2021)

    axis = axes[0, 0]
    colors = [hit_color if hit else miss_color for hit in hits]
    axis.bar(np.arange(len(events)), paths, color=colors, width=0.78)
    axis.bar(target_index, paths[target_index], color="none", edgecolor=highlight,
             linewidth=1.5, width=0.78)
    axis.set_xticks(np.arange(len(events)), labels, rotation=60, ha="right")
    axis.set_ylabel("15-to-5 d state distance (z units)")
    axis.text(target_index, paths[target_index], "  2021", rotation=90,
              va="bottom", ha="left", fontsize=7)

    axis = axes[0, 1]
    event = events[target_index]
    x = np.asarray(LEADS)
    palette = ("#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377")
    for (feature, label), color in zip(FEATURES, palette):
        y = [event["states"][str(lead)]["z"][feature] for lead in LEADS]
        axis.plot(x, y, marker="o", lw=1.45, color=color, label=label)
    axis.axhline(0, color="#606060", lw=0.75)
    axis.set_xlim(16, 4)
    axis.set_xticks(LEADS)
    axis.set_xlabel("Days before onset")
    axis.set_ylabel("Within-fold standardized state")
    axis.legend(frameon=False, ncol=2, loc="best", handlelength=1.6)

    axis = axes[1, 0]
    for index, event in enumerate(events):
        probability = [event["states"][str(lead)]["matching_probability"] for lead in LEADS]
        color = hit_color if event["five_day_hit"] else miss_color
        lw = 2.2 if index == target_index else 0.85
        alpha = 1.0 if index == target_index else 0.43
        label = "2021" if index == target_index else None
        axis.plot(x, np.asarray(probability) * 100, marker="o", lw=lw,
                  color=highlight if index == target_index else color,
                  alpha=alpha, label=label)
    axis.set_yscale("log")
    axis.set_xlim(16, 4)
    axis.set_xticks(LEADS)
    axis.set_xlabel("Matching forecast lead (days)")
    axis.set_ylabel("Forecast probability (%)")
    axis.legend(frameon=False, loc="lower left")

    axis = axes[1, 1]
    axis.scatter(paths[hits], probability5[hits] * 100, color=hit_color,
                 edgecolor="white", linewidth=0.5, s=38, label="5-d hit")
    axis.scatter(paths[~hits], probability5[~hits] * 100, color=miss_color,
                 edgecolor="white", linewidth=0.5, s=38, label="5-d miss")
    axis.scatter(paths[target_index], probability5[target_index] * 100,
                 facecolor="none", edgecolor=highlight, linewidth=1.5, s=70)
    axis.annotate("2021", (paths[target_index], probability5[target_index] * 100),
                  xytext=(4, 5), textcoords="offset points", fontsize=7)
    axis.set_yscale("log")
    axis.set_xlabel("15-to-5 d state distance (z units)")
    axis.set_ylabel("5-d probability (%)")
    axis.legend(frameon=False, loc="best")

    for letter, axis in zip("abcd", axes.flat):
        axis.text(-0.12, 1.06, letter, transform=axis.transAxes,
                  fontweight="bold", va="top")
        axis.spines[["top", "right"]].set_visible(False)
    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.14, top=0.95,
                            wspace=0.34, hspace=0.42)
    figure.savefig(OUT_PNG, dpi=300, facecolor="white")
    figure.savefig(OUT_PDF, facecolor="white")
    plt.close(figure)


def main():
    with np.load(PREDICTIONS, allow_pickle=False) as source:
        prediction = {key: source[key] for key in source.files}
    physical = json.loads(PHYSICAL.read_text())
    if physical["status"] != "V52_P2_PHYSICAL_CONTEXT_OK":
        raise RuntimeError("physical context did not pass its audit")
    if len(prediction["forecast_origin"]) != physical["n_forecast_origins"]:
        raise RuntimeError("forecast-origin inventory mismatch")
    context = {tuple(row[key] for key in KEYS): row for row in physical["records"]}
    prediction_keys = {key_from_prediction(prediction, index)
                       for index in range(len(prediction["forecast_origin"]))}
    if prediction_keys != set(context):
        raise RuntimeError("physical-context keys are not one-to-one with predictions")

    values = np.asarray([
        [context[key_from_prediction(prediction, index)][feature] for feature, _ in FEATURES]
        for index in range(len(prediction["forecast_origin"]))
    ], dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite physical state")
    z = np.empty_like(values)
    for fold in ("A", "B", "C"):
        mask = prediction["fold"] == fold
        scale = values[mask].std(axis=0, ddof=1)
        if not mask.any() or np.any(scale <= 0):
            raise RuntimeError(f"invalid physical standardization for fold {fold}")
        z[mask] = (values[mask] - values[mask].mean(axis=0)) / scale

    results = json.loads(RESULTS.read_text())
    events = results["metrics"]["three_seed_equal_mean"]["5"]["event_metrics"]["events"]
    v59 = json.loads(V59.read_text())
    if v59["status"] != "V59_STATE_DEPENDENT_PREDICTABILITY_OK":
        raise RuntimeError("v59 event diagnostic did not pass")
    hit_lookup = {(int(event["winter"]), int(event["onset_step"])): bool(event["hit"])
                  for event in v59["events"]}

    available, unavailable, source_rows = [], [], []
    for base_event in events:
        winter, onset = int(base_event["winter"]), int(base_event["onset_step"])
        key = (winter, onset)
        if key not in hit_lookup:
            raise RuntimeError(f"missing v59 hit status for {key}")
        indices = {}
        for lead in LEADS:
            candidate = np.where((prediction["winter"] == winter) &
                                 (prediction["forecast_origin"] == onset - lead * 4))[0]
            if len(candidate) == 1:
                indices[lead] = int(candidate[0])
            elif len(candidate) > 1:
                raise RuntimeError(f"duplicate origin for {winter}/{onset}/{lead}")
        if set(indices) != set(LEADS):
            unavailable.append({"winter": winter, "onset_step": onset,
                                "missing_leads_days": [lead for lead in LEADS if lead not in indices]})
            continue
        folds = {str(prediction["fold"][index]) for index in indices.values()}
        if len(folds) != 1:
            raise RuntimeError(f"event trajectory crossed folds for {winter}/{onset}")
        states, z_by_lead = {}, {}
        for lead, index in indices.items():
            raw = values[index]
            z_value = z[index]
            z_by_lead[lead] = z_value
            matching_column = {5: 0, 10: 1, 15: 2}[lead]
            state = {
                "forecast_origin": int(prediction["forecast_origin"][index]),
                "start": int(prediction["start"][index]),
                "target_class": int(prediction["target_class"][index]),
                "matching_probability": float(
                    prediction["three_seed_equal_mean_probability"][index, matching_column]),
                "raw": {feature: float(value) for (feature, _), value in zip(FEATURES, raw)},
                "z": {feature: float(value) for (feature, _), value in zip(FEATURES, z_value)},
            }
            states[str(lead)] = state
            for (feature, label), raw_value, z_value_item in zip(FEATURES, raw, z_value):
                source_rows.append({
                    "record_type": "state", "winter": winter, "onset_step": onset,
                    "fold": next(iter(folds)), "five_day_hit": hit_lookup[key],
                    "lead_days": lead, "feature": feature, "feature_label": label,
                    "raw_value": float(raw_value), "z_value": float(z_value_item),
                    "matching_probability": state["matching_probability"],
                    "direct_15_to_5": "", "path_15_to_10_to_5": "",
                })
        trajectory = trajectory_lengths(z_by_lead)
        event = {"winter": winter, "onset_step": onset, "fold": next(iter(folds)),
                 "five_day_hit": hit_lookup[key], "states": states, "trajectory": trajectory}
        available.append(event)
        source_rows.append({
            "record_type": "trajectory", "winter": winter, "onset_step": onset,
            "fold": event["fold"], "five_day_hit": event["five_day_hit"],
            "lead_days": "", "feature": "", "feature_label": "",
            "raw_value": "", "z_value": "", "matching_probability": "",
            "direct_15_to_5": trajectory["direct_15_to_5"],
            "path_15_to_10_to_5": trajectory["path_15_to_10_to_5"],
        })

    if not available:
        raise RuntimeError("no complete event lead trajectories")
    event_2021 = [event for event in available if event["winter"] == 2021]
    if len(event_2021) != 1:
        raise RuntimeError("2021 does not have exactly one complete trajectory")
    path_values = np.asarray([event["trajectory"]["direct_15_to_5"] for event in available])
    hit_values = np.asarray([event["five_day_hit"] for event in available], dtype=bool)
    rank = int(np.sum(path_values <= event_2021[0]["trajectory"]["direct_15_to_5"]))
    component_ranks = {"at_lead": {}, "change_15_to_5": {}}
    for lead in LEADS:
        component_ranks["at_lead"][str(lead)] = {}
        for feature, _ in FEATURES:
            sample = np.asarray([event["states"][str(lead)]["z"][feature] for event in available])
            value = event_2021[0]["states"][str(lead)]["z"][feature]
            rank_value = int(np.sum(sample <= value))
            component_ranks["at_lead"][str(lead)][feature] = {
                "z_value": value, "rank_of_n": [rank_value, int(sample.size)],
                "percentile_including_2021": float(100 * rank_value / sample.size),
            }
    for feature, _ in FEATURES:
        sample = np.asarray([
            event["states"]["5"]["z"][feature] - event["states"]["15"]["z"][feature]
            for event in available
        ])
        value = (event_2021[0]["states"]["5"]["z"][feature]
                 - event_2021[0]["states"]["15"]["z"][feature])
        rank_value = int(np.sum(sample <= value))
        component_ranks["change_15_to_5"][feature] = {
            "z_change": value, "rank_of_n": [rank_value, int(sample.size)],
            "percentile_including_2021": float(100 * rank_value / sample.size),
        }
    result = {
        "status": "V62_ALL_EVENT_LEAD_TRAJECTORY_OK",
        "registration": "v62_all_event_lead_trajectory_registration.md",
        "alignment": {
            "status": "PASS", "n_prediction_origins": int(len(prediction["forecast_origin"])),
            "n_registered_events": int(len(events)), "n_complete_trajectories": int(len(available)),
            "n_unavailable_trajectories": int(len(unavailable)),
            "keys": list(KEYS), "leads_days": list(LEADS),
        },
        "unavailable_events": unavailable,
        "events": available,
        "five_day_miss_minus_hit_path_difference": bootstrap_group_difference(path_values, hit_values),
        "event_2021_direct_path_rank": {
            "rank_of_n": [rank, int(len(available))],
            "percentile_including_2021": float(100 * rank / len(available)),
            "direct_15_to_5": event_2021[0]["trajectory"]["direct_15_to_5"],
        },
        "event_2021_component_ranks": component_ranks,
        "claim_boundary": (
            "Descriptive all-event lead trajectories only; no causal attribution, model "
            "selection, recalibration, threshold change or forecast-skill claim."
        ),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    with OUT_EVENTS.open("w", newline="") as handle:
        fields = ("winter", "onset_step", "fold", "five_day_hit", "direct_15_to_5",
                  "path_15_to_10_to_5", "segment_15_to_10", "segment_10_to_5")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in available:
            writer.writerow({key: event["trajectory"].get(key, event.get(key)) for key in fields})
    with OUT_SOURCE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    make_figure(available)
    OUT_CAPTION.write_text(
        "# v62 all-event lead-trajectory diagnostic\n\n"
        "**Claim.** Among rolling-test SSWs with complete 15/10/5-day records, this "
        "figure tests whether the 2021 five-day miss is an unusual lead-dependent "
        "physical-state trajectory rather than a single weak-vortex snapshot. "
        "(a) Direct Euclidean distance between the 15- and 5-day antecedent states "
        "in six within-fold standardized descriptors; blue/red denote the existing "
        "fixed-threshold five-day event hit/miss policy and the outlined bar is 2021. "
        "(b) The six 2021 standardized descriptors across leads. (c) Equal-weight "
        "three-seed matching-horizon probabilities for every complete event; 2021 is "
        "black. (d) The descriptive relation between trajectory distance and the "
        "matching five-day probability. Values are derived solely from frozen v55 "
        "predictions and v52 physical context. Missing 15-day event records are not "
        "imputed. This is a state/lead diagnostic, not causal attribution or a model "
        "selection analysis.\n"
    )
    print(json.dumps({
        "status": result["status"], "n_complete": len(available),
        "n_unavailable": len(unavailable), "2021_rank": result["event_2021_direct_path_rank"],
        "miss_minus_hit": result["five_day_miss_minus_hit_path_difference"],
    }, indent=2))


def self_test():
    z = {15: np.asarray([0.0, 0.0]), 10: np.asarray([3.0, 4.0]),
         5: np.asarray([6.0, 8.0])}
    lengths = trajectory_lengths(z)
    assert lengths["direct_15_to_5"] == 10.0
    assert lengths["path_15_to_10_to_5"] == 10.0
    diff = bootstrap_group_difference([1.0, 2.0, 4.0, 5.0], [True, True, False, False])
    assert diff["miss_minus_hit"] == 3.0
    print("V62_ALL_EVENT_LEAD_TRAJECTORY_SELF_TEST_OK")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
