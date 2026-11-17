#!/usr/bin/env python3
"""Paper-1 Figure 5: ECMWF versus prespecified three-seed neural benchmark."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "ssw_paper1_revision_p0_results.json"
PREDICTION = ROOT / "ssw_paper1_revision_common_origin_predictions.npz"
OUT_BASE = ROOT / "ssw_paper1_fig5_common_origin_benchmark_v4"
OUT_SOURCE = ROOT / "ssw_paper1_fig5_common_origin_benchmark_v4_source.csv"
OUT_CAPTION = ROOT / "ssw_paper1_fig5_common_origin_benchmark_v4_caption.md"

HORIZONS = (5, 10, 15)
MODELS = ("ecmwf_s2s", "three_seed_equal_mean", "physical_logistic")
LABELS = {
    "ecmwf_s2s": "ECMWF S2S",
    "three_seed_equal_mean": "ERA5 neural (3-seed mean)",
    "physical_logistic": "Physical logistic",
}
COLORS = {
    "ecmwf_s2s": "#3A7D44",
    "three_seed_equal_mean": "#3C78B4",
    "physical_logistic": "#C96F27",
}


def main():
    result = json.loads(RESULT.read_text())
    if result["status"] != "SSW_PAPER1_REVISION_P0_OK":
        raise RuntimeError("revision result is not passing")
    reports = result["common_origin"]
    with np.load(PREDICTION, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    probability = {
        "ecmwf_s2s": arrays["ecmwf_s2s_probability"],
        "three_seed_equal_mean": arrays["three_seed_equal_mean_probability"],
        "physical_logistic": arrays["physical_logistic_probability"],
    }
    source_rows = []
    for model in MODELS:
        for horizon in HORIZONS:
            block = reports[model][str(horizon)]
            for row in block["probability_decomposition"]["bins"]:
                source_rows.append({
                    "record_type": "reliability_bin", "model": model,
                    "horizon_days": horizon, **row,
                    "pr_auc": block["pr_auc"], "brier": block["brier"],
                    "bss": block["bss"],
                    "reliability": block["probability_decomposition"]["reliability"],
                    "resolution": block["probability_decomposition"]["resolution"],
                    "events_hit": block["event_metrics"]["events_hit"],
                    "false_episodes": block["event_metrics"]["false_alarm_episodes_total"],
                })
    with OUT_SOURCE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader(); writer.writerows(source_rows)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.3, "axes.labelsize": 8.8,
        "axes.titlesize": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.1, "pdf.fonttype": 42,
    })
    figure, axes = plt.subplots(2, 3, figsize=(10.2, 6.4))
    for axis, class_index, horizon in zip(axes[0], range(3), HORIZONS):
        axis.plot([0, 1], [0, 1], color="#777777", lw=0.8, ls=":")
        for model in MODELS:
            rows = reports[model][str(horizon)]["probability_decomposition"]["bins"]
            nonempty = [row for row in rows if row["count"]]
            x = [row["mean_probability"] for row in nonempty]
            y = [row["observed_frequency"] for row in nonempty]
            axis.scatter(x, y, s=[16 + 5 * np.sqrt(row["count"]) for row in nonempty],
                         color=COLORS[model], alpha=0.78, label=LABELS[model])
            for row in nonempty:
                if row["count"] < 10:
                    axis.annotate(str(row["count"]),
                                  (row["mean_probability"], row["observed_frequency"]),
                                  xytext=(2, 2), textcoords="offset points", fontsize=6,
                                  color=COLORS[model])
        axis.set_xlim(-0.02, 1.02); axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("Mean forecast probability")
        axis.set_ylabel("Observed frequency")
        axis.set_title(f"{horizon}-day horizon")
        if horizon == 5:
            axis.legend(frameon=False, loc="upper left")
        axis.spines[["top", "right"]].set_visible(False)

    axis = axes[1, 0]
    for model in MODELS:
        x = [reports[model][str(h)]["probability_decomposition"]["reliability"]
             for h in HORIZONS]
        y = [reports[model][str(h)]["probability_decomposition"]["resolution"]
             for h in HORIZONS]
        axis.plot(x, y, marker="o", lw=1.15, color=COLORS[model], label=LABELS[model])
        for xx, yy, horizon in zip(x, y, HORIZONS):
            axis.annotate(str(horizon), (xx, yy), xytext=(3, 3),
                          textcoords="offset points", fontsize=7)
    axis.set_xlabel("Binned reliability (lower is better)")
    axis.set_ylabel("Binned resolution (higher is better)")
    axis.legend(frameon=False, loc="best")

    axis = axes[1, 1]
    markers = {5: "o", 10: "s", 15: "^"}
    for horizon in HORIZONS:
        for model in MODELS:
            event = reports[model][str(horizon)]["event_metrics"]
            axis.scatter(event["false_alarm_episodes_total"], event["events_hit"],
                         marker=markers[horizon], color=COLORS[model], s=40,
                         edgecolor="white", linewidth=0.45)
            axis.annotate(str(horizon),
                          (event["false_alarm_episodes_total"], event["events_hit"]),
                          xytext=(3, 3), textcoords="offset points", fontsize=7)
    axis.set_xlabel("False-alarm episodes")
    axis.set_ylabel("Events hit (of 11)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[m], label=LABELS[m])
               for m in MODELS]
    legend = axis.legend(handles=handles, frameon=False, loc="upper left")
    axis.add_artist(legend)
    lead_handles = [plt.Line2D([], [], marker=markers[h], ls="", color="#555555",
                               label=f"{h} d") for h in HORIZONS]
    axis.legend(handles=lead_handles, frameon=False, loc="lower right")

    axis = axes[1, 2]
    target = (arrays["target_class"] <= 0).astype(int)
    x = probability["three_seed_equal_mean"][:, 0]
    y = probability["ecmwf_s2s"][:, 0]
    axis.scatter(x[target == 0], y[target == 0], color="#8D99A7", alpha=0.55,
                 s=17, label="No onset")
    axis.scatter(x[target == 1], y[target == 1], color="#C44E52", edgecolor="white",
                 linewidth=0.4, s=30, label="Observed onset")
    axis.plot([0, 1], [0, 1], color="#555555", lw=0.75, ls=":")
    axis.set_xlim(-0.02, 1.02); axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("ERA5 neural probability (3-seed mean)")
    axis.set_ylabel("ECMWF S2S probability")
    axis.legend(frameon=False, loc="upper left")

    for letter, axis in zip("abcdef", axes.flat):
        axis.text(-0.14, 1.08, letter, transform=axis.transAxes,
                  fontweight="bold", fontsize=10, va="top")
        axis.spines[["top", "right"]].set_visible(False)
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.12, top=0.94,
                           wspace=0.35, hspace=0.42)
    figure.savefig(OUT_BASE.with_suffix(".png"), dpi=300, facecolor="white")
    figure.savefig(OUT_BASE.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    OUT_CAPTION.write_text(
        "# Figure 5 | ECMWF reforecasts have greater probability resolution and event coverage\n\n"
        "**a--c,** Fixed 0.1-bin reliability diagrams at 5, 10 and 15 days for ECMWF "
        "IFS cy48r1, the prespecified three-seed neural mean and the physical logistic "
        "baseline. Marker area is proportional to bin count and bins with fewer than ten "
        "origins are numerically labelled; sparse high-probability bins should not be "
        "interpreted as stable calibration estimates. **d,** Fixed-bin reliability and "
        "resolution. ECMWF resolution is 0.023/0.044/0.055, versus "
        "0.003/0.003/0.011 for the neural mean. **e,** Fixed-0.20 event hits and "
        "false-alarm episodes; ECMWF detects 11/11 events at each horizon, versus "
        "5/5/8 for the neural mean. **f,** Paired five-day probabilities on all 454 "
        "origins. ECMWF PR-AUC/BSS is 0.742/+0.626, 0.745/+0.611 and 0.700/+0.530; "
        "neural values are 0.237/+0.018, 0.212/-0.040 and 0.271/+0.031. Direct Brier "
        "score remains primary because the fixed-bin decomposition is approximate. "
        "The common 00 UTC subset covers 11 events from one ECMWF cycle and was not used for model, "
        "threshold or calibration selection.\n"
    )
    print("SSW_PAPER1_FIG5_REVISED_OK")


def self_test():
    result = json.loads(RESULT.read_text())["common_origin"]
    assert result["ecmwf_s2s"]["5"]["event_metrics"]["events_hit"] == 11
    assert result["three_seed_equal_mean"]["5"]["event_metrics"]["events_hit"] == 5
    print("SSW_PAPER1_FIG5_REVISED_SELF_TEST_OK")


if __name__ == "__main__":
    import sys
    self_test() if "--self-test" in sys.argv else main()
