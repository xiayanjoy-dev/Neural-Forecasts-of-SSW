#!/usr/bin/env python3
"""All-event state-dependent diagnosis of frozen v55 five-day forecasts."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
PREDICTION = ROOT / "v55_p0_multiseed_scratch_predictions.npz"
PHYSICAL = ROOT / "v52_p2_physical_context.json"
METRICS = ROOT / "v55_p0_multiseed_scratch_results.json"
OUT_JSON = ROOT / "v59_state_dependent_predictability.json"
OUT_EVENT_CSV = ROOT / "v59_state_dependent_event_states.csv"
OUT_PROFILE_CSV = ROOT / "v59_state_dependent_profiles.csv"
OUT_PNG = ROOT / "v59_state_dependent_predictability.png"
OUT_PDF = ROOT / "v59_state_dependent_predictability.pdf"

FEATURES = (
    "u10_now", "u10_delta5", "u10_delta10", "t10_pcap_delta5",
    "t30_pcap_delta5", "t50_pcap_delta5", "heat_flux100_5d",
    "heat_flux100_10d", "epdiv_upper_5d", "epdiv_lower_5d",
    "ep_f_p_100_5d", "qbo30_5d",
)
SHORT_NAMES = ("U10", "dU10-5", "dU10-10", "dT10", "dT30", "dT50",
               "HF100-5", "HF100-10", "EPdiv-up", "EPdiv-low", "EPF100", "QBO30")
KEYS = ("forecast_origin", "start", "winter", "target_class")
THRESHOLD = 0.20
MERGE_STEPS = 20
BOOTSTRAP_DRAWS = 1000


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


def spearman(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1])


def bootstrap_spearman(x, y, seed):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAP_DRAWS):
        idx = rng.integers(0, len(x), len(x))
        value = spearman(x[idx], y[idx])
        if np.isfinite(value):
            values.append(value)
    return [float(np.percentile(values, q)) for q in (2.5, 97.5)]


def profile(z, mask):
    return {feature: {
        "n": int(mask.sum()),
        "mean_z": float(z[mask, index].mean()) if mask.any() else None,
        "median_z": float(np.median(z[mask, index])) if mask.any() else None,
    } for index, feature in enumerate(FEATURES)}


def load_aligned():
    with np.load(PREDICTION) as source:
        prediction = {key: source[key] for key in source.files}
    keys = list(zip(*(prediction[key].astype(object) for key in KEYS)))
    physical = json.loads(PHYSICAL.read_text())
    if physical["status"] != "V52_P2_PHYSICAL_CONTEXT_OK":
        raise RuntimeError("physical context is not passing")
    context = {tuple(row[key] for key in KEYS): row for row in physical["records"]}
    if set(keys) != set(context):
        raise RuntimeError("v55/P2 alignment key sets differ")
    values = np.asarray([[context[key][feature] for feature in FEATURES] for key in keys],
                        dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite physical feature")
    z = np.empty_like(values)
    for fold in ("A", "B", "C"):
        mask = prediction["fold"] == fold
        mean = values[mask].mean(axis=0)
        std = values[mask].std(axis=0, ddof=1)
        if np.any(std <= 0):
            raise RuntimeError("constant physical feature")
        z[mask] = (values[mask] - mean) / std
    return prediction, values, z


def false_episodes(prediction):
    probability = prediction["three_seed_equal_mean_probability"][:, 0]
    negative = prediction["target_class"] != 0
    alarm = (probability >= THRESHOLD) & negative
    rows = []
    for winter in sorted(np.unique(prediction["winter"])):
        idx = np.where((prediction["winter"] == winter) & alarm)[0]
        if not len(idx):
            continue
        start = 0
        for position in range(1, len(idx) + 1):
            if position == len(idx) or (prediction["forecast_origin"][idx[position]] -
                                        prediction["forecast_origin"][idx[position - 1]]) > MERGE_STEPS:
                episode = idx[start:position]
                peak = episode[np.argmax(probability[episode])]
                rows.append({"winter": int(winter), "peak_index": int(peak),
                             "peak_probability": float(probability[peak]),
                             "n_origins": int(len(episode))})
                start = position
    return rows


def make_figure(events, correlations, hit_profile, miss_profile, false_profile):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.labelsize": 9, "axes.titlesize": 9,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 8,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.1))
    blue, red, orange, grey = "#3977B8", "#B84A4A", "#D9772D", "#59636E"
    ordered = sorted(events, key=lambda row: row["max_probability"], reverse=True)
    ax = axes[0, 0]
    colors = [blue if row["hit"] else red for row in ordered]
    ax.bar(range(len(ordered)), [row["max_probability"] for row in ordered], color=colors)
    ax.axhline(THRESHOLD, color=orange, ls=":", lw=1)
    ax.set_xticks(range(len(ordered)), [str(row["winter"]) for row in ordered], rotation=65, ha="right")
    ax.set_ylabel("Maximum eligible 5-day probability")
    ax.set_title("Event-level hit and miss states")

    ax = axes[0, 1]
    strongest = sorted(correlations, key=lambda row: abs(row["spearman"]), reverse=True)[:6]
    y = np.arange(len(strongest))
    vals = [row["spearman"] for row in strongest]
    lower = [row["spearman"] - row["ci95"][0] for row in strongest]
    upper = [row["ci95"][1] - row["spearman"] for row in strongest]
    ax.errorbar(vals, y, xerr=[lower, upper], fmt="o", color=blue, capsize=2)
    ax.axvline(0, color=grey, lw=0.8)
    ax.set_yticks(y, [SHORT_NAMES[FEATURES.index(row["feature"])] for row in strongest])
    ax.set_xlabel("Event-level Spearman correlation")
    ax.set_title("Association with forecast probability")

    selected = ("u10_now", "u10_delta5", "heat_flux100_5d", "epdiv_lower_5d", "qbo30_5d")
    ax = axes[1, 0]
    x = np.arange(len(selected)); width = 0.35
    ax.bar(x - width / 2, [hit_profile[f]["mean_z"] for f in selected], width, color=blue, label="Hit events")
    ax.bar(x + width / 2, [miss_profile[f]["mean_z"] for f in selected], width, color=red, label="Missed events")
    ax.axhline(0, color=grey, lw=0.8)
    ax.set_xticks(x, [SHORT_NAMES[FEATURES.index(f)] for f in selected], rotation=25, ha="right")
    ax.set_ylabel("Within-fold standardized state")
    ax.set_title("Five-day antecedent context")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.bar(x - width / 2, [hit_profile[f]["mean_z"] for f in selected], width, color=blue, label="Hit-event state")
    ax.bar(x + width / 2, [false_profile[f]["mean_z"] for f in selected], width, color=orange, label="False-alarm peak")
    ax.axhline(0, color=grey, lw=0.8)
    ax.set_xticks(x, [SHORT_NAMES[FEATURES.index(f)] for f in selected], rotation=25, ha="right")
    ax.set_ylabel("Within-fold standardized state")
    ax.set_title("False-alarm state comparison")
    ax.legend(frameon=False)

    for label, axis in zip("abcd", axes.flat):
        axis.text(-0.12, 1.05, label, transform=axis.transAxes, fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.15, top=0.94, wspace=0.42, hspace=0.48)
    fig.savefig(OUT_PNG, dpi=300, facecolor="white")
    fig.savefig(OUT_PDF, facecolor="white")
    plt.close(fig)


def main():
    prediction, values, z = load_aligned()
    metrics = json.loads(METRICS.read_text())
    event_table = metrics["metrics"]["three_seed_equal_mean"]["5"]["event_metrics"]["events"]
    probability = prediction["three_seed_equal_mean_probability"][:, 0]
    events, event_indices = [], []
    for event in event_table:
        onset = int(event["onset_step"])
        winter = int(event["winter"])
        eligible = np.where((prediction["winter"] == winter) &
                            (prediction["forecast_origin"] < onset) &
                            (onset <= prediction["forecast_origin"] + 20))[0]
        if len(eligible) != int(event["n_eligible_initializations"]):
            raise RuntimeError("eligible initialization count changed")
        peak = eligible[np.argmax(probability[eligible])]
        fixed = eligible[np.argmin(np.abs(prediction["forecast_origin"][eligible] - (onset - 20)))]
        if abs(int(prediction["forecast_origin"][fixed]) - (onset - 20)) > 4:
            raise RuntimeError("missing nominal five-day state")
        hit = bool(np.any(probability[eligible] >= THRESHOLD))
        alarm = eligible[probability[eligible] >= THRESHOLD]
        row = {"winter": winter, "onset_step": onset,
               "fixed_forecast_origin": int(prediction["forecast_origin"][fixed]),
               "max_forecast_origin": int(prediction["forecast_origin"][peak]),
               "max_probability": float(probability[peak]), "hit": hit,
               "first_alarm_lead_days": (
                   float(np.max((onset - prediction["forecast_origin"][alarm]) / 4.0))
                   if len(alarm) else None)}
        for index, feature in enumerate(FEATURES):
            row[feature] = float(values[fixed, index])
            row[feature + "_z"] = float(z[fixed, index])
        events.append(row); event_indices.append(fixed)

    event_z = z[np.asarray(event_indices)]
    event_probability = np.asarray([row["max_probability"] for row in events])
    correlations = []
    for index, feature in enumerate(FEATURES):
        corr = spearman(event_z[:, index], event_probability)
        correlations.append({"feature": feature, "spearman": corr,
                             "ci95": bootstrap_spearman(event_z[:, index], event_probability, 5900 + index)})
    hit_mask = np.asarray([row["hit"] for row in events])
    hit_profile = profile(event_z, hit_mask)
    miss_profile = profile(event_z, ~hit_mask)

    false = false_episodes(prediction)
    false_z = z[np.asarray([row["peak_index"] for row in false])]
    false_profile = profile(false_z, np.ones(len(false_z), dtype=bool))
    result = {
        "status": "V59_STATE_DEPENDENT_PREDICTABILITY_OK",
        "alignment": {"status": "PASS", "n_forecast_origins": int(len(probability)), "n_events": len(events)},
        "protocol": "v59_state_dependent_predictability_registration.md",
        "events": events,
        "event_probability_associations": correlations,
        "hit_profile": hit_profile,
        "miss_profile": miss_profile,
        "false_alarm_profile": false_profile,
        "false_alarm_episodes": false,
        "summary": {"events_hit": int(hit_mask.sum()), "events_missed": int((~hit_mask).sum()),
                    "false_alarm_episodes": len(false)},
        "claim_boundary": "Descriptive state associations only; no causal attribution or model selection.",
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    with OUT_EVENT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=events[0].keys())
        writer.writeheader(); writer.writerows(events)
    profiles = []
    for group, block in (("hit", hit_profile), ("miss", miss_profile), ("false_alarm", false_profile)):
        for feature, value in block.items():
            profiles.append({"group": group, "feature": feature, **value})
    with OUT_PROFILE_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=profiles[0].keys())
        writer.writeheader(); writer.writerows(profiles)
    make_figure(events, correlations, hit_profile, miss_profile, false_profile)
    print(json.dumps({"status": result["status"], "summary": result["summary"],
                      "top_associations": sorted(correlations, key=lambda row: abs(row["spearman"]), reverse=True)[:4]}, indent=2))


def self_test():
    assert round(spearman(np.asarray([1., 2., 3.]), np.asarray([3., 2., 1.])), 6) == -1.0
    assert average_ranks(np.asarray([1., 1., 2.])).tolist() == [1.5, 1.5, 3.0]
    print("V59_STATE_DEPENDENT_SELF_TEST_OK")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
