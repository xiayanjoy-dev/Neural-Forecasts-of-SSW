#!/usr/bin/env python3
"""Build the integrated AIES Lessons Learned Figure 3 from frozen artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parent
P0 = ROOT / "ssw_paper1_revision_p0_results.json"
PAIRED = ROOT / "v52_paired21_seed42_pooled_results.json"
PLATT = ROOT / "v53_p4_nested_calibration_results.json"
EARLY = ROOT / "v57_p7_forward_evaluation.json"
XGB = ROOT / "v58_oof_latent_xgb_results_corrected.json"
TRAJECTORY = ROOT / "v64_s2s_committor_c_pilot.json"
EVENT_TRAJECTORIES = ROOT / "ssw_paper1_revision_v4_event_trajectories.csv"
REVISION_V4 = ROOT / "ssw_paper1_revision_v4_results.json"
ECMWF_WARNINGS = ROOT / "v63_common_origin_event_warnings.csv"
V65_RESULTS = ROOT / "v65_natural_hazard_results.json"

OUT_PNG = ROOT / "ssw_paper1_fig3_lessons_v3.png"
OUT_PDF = ROOT / "ssw_paper1_fig3_lessons_v3.pdf"
OUT_CSV = ROOT / "ssw_paper1_fig3_lessons_v3_source.csv"
OUT_JSON = ROOT / "ssw_paper1_fig3_lessons_v3_audit.json"

HORIZONS = (5, 10, 15)
MODEL_ORDER = ("ecmwf_s2s", "three_seed_equal_mean", "physical_logistic")
MODEL_LABEL = {
    "ecmwf_s2s": "ECMWF IFS",
    "three_seed_equal_mean": "ERA5 neural",
    "physical_logistic": "Physical logistic",
}
MODEL_COLOR = {
    "ecmwf_s2s": "#3A7D44",
    "three_seed_equal_mean": "#3C78B4",
    "physical_logistic": "#C96F27",
}
MARKER = {5: "o", 10: "s", 15: "^"}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def intervention_rows():
    paired = load_json(PAIRED)
    platt = load_json(PLATT)
    early = load_json(EARLY)
    xgb = load_json(XGB)
    trajectory = load_json(TRAJECTORY)
    v65 = load_json(V65_RESULTS)

    early_a = early["arms"]["A"]["metrics"]["three_seed_equal_mean"]["5"]["event_metrics"]
    early_b = early["arms"]["B"]["metrics"]["three_seed_equal_mean"]["5"]["event_metrics"]
    xgb_t = xgb["pooled_BC"]["5"]["transformer"]["event_metrics"]
    xgb_b = xgb["pooled_BC"]["5"]["blend"]["event_metrics"]
    raw = trajectory["metrics"]["5"]["raw_ecmwf"]
    hazard = trajectory["metrics"]["5"]["trajectory_hazard"]
    v65_hazard = v65["hazard_three_seed"]["5"]
    v65_scratch = v65["scratch_three_seed"]["5"]

    rows = [
        {
            "name": "Label-free pretraining", "scope": "A/B/C, seed 42",
            "dpr": paired["v51a_minus_scratch"]["5"]["pr_auc"],
            "dbss": paired["v51a_minus_scratch"]["5"]["brier_skill_score"],
            "dhit": paired["v51a_minus_scratch"]["5"]["event_hits"],
            "dfa": paired["v51a_minus_scratch"]["5"]["false_alarm_episodes"],
            "decision": "STOP", "source": PAIRED.name,
        },
        {
            "name": "Nested Platt", "scope": "A/B/C, seed 42",
            "dpr": platt["calibrated_minus_raw"]["5"]["pr_auc"],
            "dbss": platt["calibrated_minus_raw"]["5"]["bss"],
            "dhit": platt["calibrated_minus_raw"]["5"]["event_hits"],
            "dfa": platt["calibrated_minus_raw"]["5"]["false_episodes"],
            "decision": "STOP", "source": PLATT.name,
        },
        {
            "name": "1960–78 labels", "scope": "2022–25, 3 seeds",
            "dpr": early["arm_B_minus_A"]["5"]["pr_auc"],
            "dbss": early["arm_B_minus_A"]["5"]["bss"],
            "dhit": early_b["events_hit"] - early_a["events_hit"],
            "dfa": early_b["false_alarm_episodes_total"] - early_a["false_alarm_episodes_total"],
            "decision": "STOP", "source": EARLY.name,
        },
        {
            "name": "Out-of-fold XGBoost blend", "scope": "B+C, seed 42",
            "dpr": xgb["pooled_BC"]["5"]["blend"]["pr_auc"] - xgb["pooled_BC"]["5"]["transformer"]["pr_auc"],
            "dbss": xgb["pooled_BC"]["5"]["blend"]["bss"] - xgb["pooled_BC"]["5"]["transformer"]["bss"],
            "dhit": xgb_b["events_hit"] - xgb_t["events_hit"],
            "dfa": xgb_b["false_alarm_episodes_total"] - xgb_t["false_alarm_episodes_total"],
            "decision": "STOP", "source": XGB.name,
        },
        {
            "name": "ECMWF trajectory hazard", "scope": "C fold, 4 events",
            "dpr": hazard["pr_auc"] - raw["pr_auc"],
            "dbss": hazard["bss"] - raw["bss"],
            "dhit": hazard["event_metrics"]["events_hit"] - raw["event_metrics"]["events_hit"],
            "dfa": hazard["event_metrics"]["false_alarm_episodes_total"] - raw["event_metrics"]["false_alarm_episodes_total"],
            "decision": "PILOT", "source": TRAJECTORY.name,
        },
        {
            "name": "Natural-frequency hazard", "scope": "A/B/C, 3 seeds",
            "dpr": v65_hazard["pr_auc"] - v65_scratch["pr_auc"],
            "dbss": v65_hazard["bss"] - v65_scratch["bss"],
            "dhit": v65_hazard["event_metrics"]["events_hit"] - v65_scratch["event_metrics"]["events_hit"],
            "dfa": v65_hazard["event_metrics"]["false_alarm_episodes_total"] - v65_scratch["event_metrics"]["false_alarm_episodes_total"],
            "decision": "STOP", "source": V65_RESULTS.name,
        },
    ]
    return rows


def case_2021():
    with EVENT_TRAJECTORIES.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    neural = {
        int(row["head_horizon_days"]): float(row["probability"])
        for row in rows
        if row["lead_days"] and int(row["winter"]) == 2021 and int(row["lead_days"]) == 5
    }
    revision = load_json(REVISION_V4)["trajectory_diagnostics"]["event_2021"]
    with ECMWF_WARNINGS.open(newline="", encoding="utf-8") as handle:
        warning_rows = list(csv.DictReader(handle))
    ecmwf = {
        int(row["horizon_days"]): float(row["first_alarm_lead_days"])
        for row in warning_rows if row["model"] == "ecmwf_s2s" and int(row["winter"]) == 2021
    }
    v65 = load_json(V65_RESULTS)
    v65_first_alert = {
        horizon: float(next(
            event["first_alarm_lead_days"]
            for event in v65["hazard_three_seed"][str(horizon)]["event_metrics"]["events"]
            if event["winter"] == 2021
        ))
        for horizon in HORIZONS
    }
    return {
        "neural": neural,
        "ecmwf_first_alert": ecmwf,
        "v65_first_alert": v65_first_alert,
        "euclidean": float(revision["euclidean"]),
        "euclidean_rank": tuple(revision["euclidean_rank_of_n"]),
        "mahalanobis": float(revision["mahalanobis"]),
        "mahalanobis_rank": tuple(revision["mahalanobis_rank_of_n"]),
    }


def write_source(p0, interventions, case):
    rows = []
    for model in MODEL_ORDER:
        for horizon in HORIZONS:
            block = p0["common_origin"][model][str(horizon)]
            event = block["event_metrics"]
            for metric, value in (
                ("pr_auc", block["pr_auc"]), ("bss", block["bss"]),
                ("events_hit", event["events_hit"]),
                ("false_alarm_episodes", event["false_alarm_episodes_total"]),
            ):
                rows.append({
                    "panel": "a" if metric in ("pr_auc", "bss") else "b",
                    "category": model, "scope": "454 origins; 11 events; IFS cy48r1",
                    "horizon_days": horizon, "metric": metric, "value": value,
                    "note": "common-origin point estimate", "source": P0.name,
                })
    for item in interventions:
        for metric in ("dpr", "dbss", "dhit", "dfa"):
            rows.append({
                "panel": "c", "category": item["name"], "scope": item["scope"],
                "horizon_days": 5, "metric": metric, "value": item[metric],
                "note": item["decision"], "source": item["source"],
            })
    for horizon, probability in case["neural"].items():
        rows.append({
            "panel": "d", "category": "2021 three-seed mean", "scope": "exact day -5 origin",
            "horizon_days": horizon, "metric": "probability", "value": probability,
            "note": "fixed threshold 0.20", "source": EVENT_TRAJECTORIES.name,
        })
    for horizon, lead in case["v65_first_alert"].items():
        rows.append({
            "panel": "d", "category": "2021 natural-frequency hazard",
            "scope": "frozen A/B/C three-seed evaluation", "horizon_days": horizon,
            "metric": "first_alarm_lead_days", "value": lead,
            "note": "case recovered but pooled event hits decreased", "source": V65_RESULTS.name,
        })
    rows.extend([
        {"panel": "d", "category": "2021", "scope": "14 complete trajectories", "horizon_days": "",
         "metric": "euclidean_distance", "value": case["euclidean"],
         "note": f"rank {case['euclidean_rank'][0]}/{case['euclidean_rank'][1]}", "source": REVISION_V4.name},
        {"panel": "d", "category": "2021", "scope": "14 complete trajectories", "horizon_days": "",
         "metric": "mahalanobis_distance", "value": case["mahalanobis"],
         "note": f"rank {case['mahalanobis_rank'][0]}/{case['mahalanobis_rank'][1]}", "source": REVISION_V4.name},
    ])
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def signed(value, digits=3):
    if value is None:
        return "—"
    return f"{value:+.{digits}f}"


def plot(p0, interventions, case):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11.0,
        "axes.labelsize": 11.5, "axes.titlesize": 11.5,
        "xtick.labelsize": 10.2, "ytick.labelsize": 10.2,
        "legend.fontsize": 10.2, "pdf.fonttype": 42,
    })
    figure = plt.figure(figsize=(7.6, 6.35))
    grid = figure.add_gridspec(3, 2, height_ratios=(1.0, 1.05, 0.72), hspace=0.56, wspace=0.38)
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, :])
    ax_d = figure.add_subplot(grid[2, :])

    for model in MODEL_ORDER:
        xs, ys = [], []
        for horizon in HORIZONS:
            block = p0["common_origin"][model][str(horizon)]
            x, y = block["pr_auc"], block["bss"]
            xs.append(x); ys.append(y)
            ax_a.scatter(x, y, s=48, marker=MARKER[horizon], color=MODEL_COLOR[model],
                         edgecolor="white", linewidth=0.6, zorder=4)
        ax_a.plot(xs, ys, color=MODEL_COLOR[model], lw=1.0, alpha=0.55, label=MODEL_LABEL[model])
    ax_a.axhline(0, color="#6B7280", lw=0.8, ls="--")
    ax_a.set(xlabel="PR-AUC", ylabel="Brier skill score", xlim=(0.1, 0.8), ylim=(-0.11, 0.70))
    ax_a.set_title("Common-origin scores")
    model_legend = ax_a.legend(frameon=False, loc="upper left", handlelength=1.4)
    ax_a.add_artist(model_legend)
    horizon_handles = [Line2D([], [], color="#4B5563", marker=MARKER[h], ls="none", ms=5,
                              label=f"{h} d") for h in HORIZONS]
    ax_a.legend(handles=horizon_handles, frameon=False, loc="lower right", ncol=3,
                handletextpad=0.2, columnspacing=0.6)

    for model in MODEL_ORDER:
        xs, ys = [], []
        for horizon in HORIZONS:
            event = p0["common_origin"][model][str(horizon)]["event_metrics"]
            x, y = event["false_alarm_episodes_total"], event["events_hit"]
            xs.append(x); ys.append(y)
            ax_b.scatter(x, y, s=48, marker=MARKER[horizon], color=MODEL_COLOR[model],
                         edgecolor="white", linewidth=0.6, zorder=4)
        ax_b.plot(xs, ys, color=MODEL_COLOR[model], lw=1.0, alpha=0.55)
    ax_b.set(xlabel="False-alarm episodes", ylabel="Events hit (of 11)", xlim=(0, 25), ylim=(0, 12))
    ax_b.set_title("Fixed-0.20 event warnings")
    ax_b.text(0.03, 0.06, "better: upper left", transform=ax_b.transAxes,
              ha="left", va="bottom", fontsize=8, color="#4B5563")

    ax_c.set_axis_off(); ax_c.set_xlim(0, 1); ax_c.set_ylim(-0.7, 6.65)
    columns = ((0.015, "Intervention and scope"), (0.46, r"$Δ$PR$_5$"),
               (0.57, r"$Δ$BSS$_5$"), (0.68, r"$Δ$hit$_5$"),
               (0.78, r"$Δ$FA$_5$"), (0.90, "Decision"))
    for x, label in columns:
        ax_c.text(x, 6.25, label, fontsize=10.5, fontweight="bold", va="center")
    fills = {"STOP": "#FBE9E7", "PILOT": "#FFF3CD", "PENDING": "#ECEFF1"}
    status_color = {"STOP": "#A33A32", "PILOT": "#8A6500", "PENDING": "#5F6B73"}
    short_name = {
        "Label-free pretraining": "Label-free (A/B/C, s42)",
        "Nested Platt": "Nested Platt (A/B/C, s42)",
        "1960–78 labels": "1960–78 labels (2022–25, 3s)",
        "Out-of-fold XGBoost blend": "OOF XGBoost (B+C, s42)",
        "ECMWF trajectory hazard": "ECMWF hazard (C, 4 events)",
        "Natural-frequency hazard": "Natural hazard (A/B/C, 3s)",
    }
    for index, item in enumerate(interventions):
        y = 5.45 - index
        ax_c.axhspan(y - 0.42, y + 0.42, color=fills[item["decision"]], zorder=0)
        ax_c.text(0.015, y, short_name[item["name"]], fontsize=9.8,
                  fontweight="bold", va="center")
        values = (signed(item["dpr"]), signed(item["dbss"]), signed(item["dhit"], 0), signed(item["dfa"], 0))
        for x, value in zip((0.46, 0.57, 0.68, 0.78), values):
            ax_c.text(x, y, value, fontsize=10.2, ha="center", va="center")
        decision = "PILOT ONLY" if item["decision"] == "PILOT" else item["decision"]
        ax_c.text(0.92, y, decision, fontsize=9.8, fontweight="bold", ha="center", va="center",
                  color=status_color[item["decision"]])
    ax_c.text(0.015, -0.38, "Rows use different registered estimands; deltas are intervention minus its own baseline.",
              fontsize=9.0, color="#4B5563", va="center")
    ax_c.set_title("Registered interventions", pad=4)

    x = np.arange(3)
    probabilities = np.asarray([case["neural"][h] for h in HORIZONS])
    ax_d.scatter(x, probabilities, color="#3C78B4", s=52, zorder=4)
    ax_d.plot(x, probabilities, color="#3C78B4", lw=1.1, alpha=0.65)
    ax_d.axhline(0.20, color="#A33A32", lw=1.0, ls="--")
    for i, value in enumerate(probabilities):
        ax_d.annotate(f"{100*value:.2f}%", (i, value), xytext=(0, 6), textcoords="offset points",
                      ha="center", fontsize=10.0, color="#2D5F91")
    ax_d.set_yscale("log"); ax_d.set_ylim(4e-4, 0.35); ax_d.set_xlim(-0.45, 6.2)
    ax_d.set_xticks(x, ["5 d", "10 d", "15 d"])
    ax_d.set_ylabel("Three-seed P at day -5")
    ax_d.axvspan(2.75, 6.15, color="#F5F6F7", zorder=0)
    ax_d.text(0.56, 0.83, "IFS first alerts: 4 / 8 / 15 d\n(no same-origin probability)",
              transform=ax_d.transAxes, fontsize=9.6, va="top", color="#2F6F44")
    ax_d.text(0.56, 0.55, "Natural-frequency hazard: 2 / 4 / 9 d\n(2021 recovered; pooled hits 4 / 4 / 5)",
              transform=ax_d.transAxes, fontsize=9.6, va="top", color="#7A5195")
    ax_d.text(0.56, 0.25, "State displacement: Euclidean 3.67 (9/14);\nMahalanobis 3.12 (6/14)",
              transform=ax_d.transAxes, fontsize=9.6, va="top", color="#374151")
    ax_d.set_title("January 2021 context")

    for letter, axis in (("a", ax_a), ("b", ax_b), ("c", ax_c), ("d", ax_d)):
        axis.text(-0.10 if letter in ("a", "b") else -0.025, 1.08, letter,
                  transform=axis.transAxes, fontsize=12, fontweight="bold", va="top")
    for axis in (ax_a, ax_b, ax_d):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color="#D9E0E6", lw=0.55, alpha=0.7)
        axis.set_axisbelow(True)
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.07, top=0.965)
    figure.savefig(OUT_PNG, dpi=300, facecolor="white")
    figure.savefig(OUT_PDF, facecolor="white")
    plt.close(figure)


def main():
    p0 = load_json(P0)
    if p0["status"] != "SSW_PAPER1_REVISION_P0_OK":
        raise RuntimeError("Paper-1 common-origin source is not passing")
    interventions = intervention_rows()
    case = case_2021()
    checks = {
        "common_origins_454": p0["alignment"]["common_origins"] == 454,
        "covered_events_11": p0["alignment"]["covered_events"] == 11,
        "ecmwf_pr5": abs(p0["common_origin"]["ecmwf_s2s"]["5"]["pr_auc"] - 0.7424338624338623) < 1e-12,
        "neural_bss10": abs(p0["common_origin"]["three_seed_equal_mean"]["10"]["bss"] + 0.04017230687147788) < 1e-12,
        "six_interventions": len(interventions) == 6,
        "v65_final_stop": interventions[-1]["decision"] == "STOP" and np.allclose(
            [interventions[-1][k] for k in ("dpr", "dbss", "dhit", "dfa")],
            [-0.07040653093340932, 0.027098083222371172, -2.0, -16.0],
        ),
        "case_probabilities": np.allclose([case["neural"][h] for h in HORIZONS],
                                           [0.0007343457, 0.0264024828, 0.0844369233], atol=1e-10),
        "case_ranks": case["euclidean_rank"] == (9, 14) and case["mahalanobis_rank"] == (6, 14),
        "ecmwf_alerts": case["ecmwf_first_alert"] == {5: 4.0, 10: 8.0, 15: 15.0},
        "v65_alerts": case["v65_first_alert"] == {5: 2.0, 10: 4.0, 15: 9.0},
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks, indent=2))
    write_source(p0, interventions, case)
    plot(p0, interventions, case)
    result = {
        "status": "SSW_PAPER1_FIG3_LESSONS_V3_OK",
        "checks": checks,
        "outputs": [OUT_PNG.name, OUT_PDF.name, OUT_CSV.name],
        "claim_boundary": (
            "Panels a-b use 454 common origins and 11 events from one IFS cycle; "
            "panel c rows have different registered estimands and cannot be ranked across rows; "
            "panel d is descriptive and does not identify a causal failure mechanism."
        ),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
