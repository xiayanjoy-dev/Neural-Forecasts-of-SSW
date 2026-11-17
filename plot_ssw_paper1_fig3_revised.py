#!/usr/bin/env python3
"""Paper-1 Figure 3: multiplicity-aware event-state diagnosis."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
V59 = ROOT / "v59_state_dependent_predictability.json"
REVISION = ROOT / "ssw_paper1_revision_p0_results.json"
REVISION_V4 = ROOT / "ssw_paper1_revision_v4_results.json"
OUT_BASE = ROOT / "ssw_paper1_fig3_state_diagnostic_v4"
OUT_SOURCE = ROOT / "ssw_paper1_fig3_state_diagnostic_v4_source.csv"
OUT_CAPTION = ROOT / "ssw_paper1_fig3_state_diagnostic_v4_caption.md"

FEATURES = (
    "u10_now", "u10_delta5", "u10_delta10", "t10_pcap_delta5",
    "t30_pcap_delta5", "t50_pcap_delta5", "heat_flux100_5d",
    "heat_flux100_10d", "epdiv_upper_5d", "epdiv_lower_5d",
    "ep_f_p_100_5d", "qbo30_5d",
)
LABELS = {
    "u10_now": "U10", "u10_delta5": "dU10 (5 d)",
    "u10_delta10": "dU10 (10 d)", "t10_pcap_delta5": "dT10 polar cap",
    "t30_pcap_delta5": "dT30 polar cap", "t50_pcap_delta5": "dT50 polar cap",
    "heat_flux100_5d": "HF100 (5 d)", "heat_flux100_10d": "HF100 (10 d)",
    "epdiv_upper_5d": "EPdiv upper", "epdiv_lower_5d": "EPdiv lower",
    "ep_f_p_100_5d": "vertical EP flux 100", "qbo30_5d": "QBO30",
}
BLUE = "#3977B8"
RED = "#B84A4A"
ORANGE = "#D9772D"
GREY = "#68727D"


def interval_plot(axis, values, intervals, y, colors, xlabel):
    lower = np.asarray(values) - np.asarray([row[0] for row in intervals])
    upper = np.asarray([row[1] for row in intervals]) - np.asarray(values)
    for index, (value, low, high, color) in enumerate(zip(values, lower, upper, colors)):
        axis.errorbar(value, y[index], xerr=[[low], [high]], fmt="o", color=color,
                      ecolor=color, ms=4.0, capsize=2, lw=1.0)
    axis.axvline(0.0, color="#777777", lw=0.8, ls="--")
    axis.set_yticks(y, [LABELS[feature] for feature in FEATURES])
    axis.set_xlabel(xlabel)
    axis.grid(axis="x", color="#D9E0E6", lw=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def main():
    v59 = json.loads(V59.read_text())
    revision = json.loads(REVISION.read_text())
    revision_v4 = json.loads(REVISION_V4.read_text())
    if v59["status"] != "V59_STATE_DEPENDENT_PREDICTABILITY_OK":
        raise RuntimeError("v59 diagnostic is not passing")
    if revision["status"] != "SSW_PAPER1_REVISION_P0_OK":
        raise RuntimeError("revision diagnostic is not passing")
    association = revision_v4["aligned_event_associations"]["features"]
    contrast = {row["feature"]: row for row in revision["state_contrasts"]}
    family_contrast = revision_v4["state_contrasts_family_control"]["features"]
    rows = []
    for feature in FEATURES:
        rows.append({"record_type": "association", **association[feature]})
        rows.append({"record_type": "state_contrast", **contrast[feature]})
        rows.append({"record_type": "false_hit_family_contrast", "feature": feature,
                     **family_contrast[feature]})
    with OUT_SOURCE.open("w", newline="") as handle:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value
                             for key, value in row.items()})

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2, "axes.labelsize": 8.8,
        "axes.titlesize": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 7.4,
        "pdf.fonttype": 42,
    })
    figure, axes = plt.subplots(2, 2, figsize=(8.0, 6.5))
    ordered = sorted(v59["events"], key=lambda row: row["max_probability"], reverse=True)
    axis = axes[0, 0]
    x = np.arange(len(ordered))
    axis.bar(x, [row["max_probability"] for row in ordered],
             color=[BLUE if row["hit"] else RED for row in ordered])
    axis.axhline(0.20, color=ORANGE, lw=1.0, ls=":")
    axis.set_xticks(x, [str(row["winter"]) for row in ordered], rotation=65, ha="right")
    axis.set_ylabel("Maximum eligible 5-day probability")
    axis.set_title("Six hits and ten misses")
    axis.spines[["top", "right"]].set_visible(False)

    y = np.arange(len(FEATURES))
    values = [association[f]["spearman"] for f in FEATURES]
    intervals = [association[f]["bootstrap_ci95"] for f in FEATURES]
    interval_plot(axes[0, 1], values, intervals, y, [BLUE] * len(FEATURES),
                  "Spearman correlation with event probability")
    axes[0, 1].set_title("No association survives multiplicity control")

    hm_values = [contrast[f]["hit_minus_miss"] for f in FEATURES]
    hm_intervals = [contrast[f]["hit_minus_miss_ci95"] for f in FEATURES]
    hm_colors = [BLUE if row[0] > 0 or row[1] < 0 else GREY for row in hm_intervals]
    interval_plot(axes[1, 0], hm_values, hm_intervals, y, hm_colors,
                  "Hit minus missed event (standardized units)")
    axes[1, 0].set_title("Hit and missed antecedent states overlap")

    fh_values = [family_contrast[f]["false_minus_hit"] for f in FEATURES]
    fh_intervals = [family_contrast[f]["simultaneous_ci95"] for f in FEATURES]
    fh_colors = [ORANGE if family_contrast[f]["survives_family_control"] else GREY
                 for f in FEATURES]
    interval_plot(axes[1, 1], fh_values, fh_intervals, y, fh_colors,
                  "False alarm minus hit (standardized units)")
    axes[1, 1].set_title("No false-alarm contrast survives family control")

    for letter, axis in zip("abcd", axes.flat):
        axis.text(-0.15, 1.07, letter, transform=axis.transAxes,
                  fontweight="bold", fontsize=10, va="top")
    figure.subplots_adjust(left=0.15, right=0.985, bottom=0.12, top=0.94,
                           wspace=0.46, hspace=0.43)
    figure.savefig(OUT_BASE.with_suffix(".png"), dpi=300, facecolor="white")
    figure.savefig(OUT_BASE.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    OUT_CAPTION.write_text(
        "# Figure 3 | Response-conditioned state diagnostics show unresolved hit, miss and false-alarm overlap\n\n"
        "**a,** Maximum eligible five-day probability for all 16 outer-test SSWs; blue and red "
        "bars denote the six fixed-0.20 hits and ten misses. **b,** Event-level Spearman "
        "associations between each exact day-minus-5 state and the aligned five-day-head "
        "probability, with 95% intervals from 1,000 event resamples. The largest absolute "
        "association is recent U10 change (rho = -0.535), but no association survives "
        "Benjamini--Hochberg control (minimum q = 0.184). "
        "**c,** Hit-minus-miss mean standardized-state contrasts and 95% winter-cluster "
        "intervals. **d,** False-alarm-minus-hit contrasts for the 20 false-alarm peaks with "
        "simultaneous 95% winter-cluster intervals across all 12 descriptors; none excludes "
        "zero. Standardization was performed within each fold for this post hoc diagnostic. "
        "These response-conditioned comparisons describe forecast "
        "associations and do not establish causal drivers or select model inputs.\n"
    )
    print("SSW_PAPER1_FIG3_REVISED_OK")


def self_test():
    revision_v4 = json.loads(REVISION_V4.read_text())
    association = revision_v4["aligned_event_associations"]["features"]
    assert min(row["bh_q"] for row in association.values()) > 0.18
    assert not any(row["survives_family_control"] for row in
                   revision_v4["state_contrasts_family_control"]["features"].values())
    print("SSW_PAPER1_FIG3_REVISED_SELF_TEST_OK")


if __name__ == "__main__":
    import sys
    self_test() if "--self-test" in sys.argv else main()
