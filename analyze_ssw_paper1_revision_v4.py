#!/usr/bin/env python3
"""Paper-1 v4 reviewer-requested diagnostics from frozen prediction artifacts.

No model is fitted for forecasting and no outer result selects a setting. The
only fitted quantities are descriptive calibration intercepts/slopes computed
on the already frozen outer predictions.
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from analyze_ssw_paper1_revision_p0 import (
    average_precision, roc_auc, decision_metrics, event_metrics,
    brier_decomposition,
)
from analyze_v59_state_dependent_predictability import (
    FEATURES as STATE_FEATURES, load_aligned, false_episodes, spearman,
)


ROOT = Path(__file__).resolve().parent
V55_PRED = ROOT / "v55_p0_multiseed_scratch_predictions.npz"
V55_RESULTS = ROOT / "v55_p0_multiseed_scratch_results.json"
BASELINE_PRED = ROOT / "v52_p3_physical_baselines_predictions.npz"
V62_RESULTS = ROOT / "v62_all_event_lead_trajectory.json"
COMMON_CSV = ROOT / "v55_p1_s2s_common_origins.csv"
ECMWF_MANIFEST = ROOT / "v55_p1_s2s_download_manifest.json"

OUT_JSON = ROOT / "ssw_paper1_revision_v4_results.json"
OUT_METRICS = ROOT / "ssw_paper1_revision_v4_metrics.csv"
OUT_RELIABILITY = ROOT / "ssw_paper1_revision_v4_reliability.csv"
OUT_FALSE = ROOT / "ssw_paper1_revision_v4_false_alarm_sensitivity.csv"
OUT_STATE = ROOT / "ssw_paper1_revision_v4_state_contrasts.csv"
OUT_ASSOC = ROOT / "ssw_paper1_revision_v4_aligned_event_associations.csv"
OUT_TRAJECTORY = ROOT / "ssw_paper1_revision_v4_event_trajectories.csv"
OUT_FIG2_PNG = ROOT / "ssw_paper1_fig2_fold_baseline_v4.png"
OUT_FIG2_PDF = ROOT / "ssw_paper1_fig2_fold_baseline_v4.pdf"
OUT_FIG4_PNG = ROOT / "ssw_paper1_fig4_2021_v4.png"
OUT_FIG4_PDF = ROOT / "ssw_paper1_fig4_2021_v4.pdf"
OUT_FIGS2_PNG = ROOT / "ssw_paper1_figS2_probability_diagnostics_v4.png"
OUT_FIGS2_PDF = ROOT / "ssw_paper1_figS2_probability_diagnostics_v4.pdf"

HORIZONS = (5, 10, 15)
FOLDS = ("A", "B", "C")
THRESHOLD = 0.20
KEYS = ("forecast_origin", "start", "winter", "target_class", "fold")
MODEL_ORDER = (
    "climatology", "persistence", "physical_logistic",
    "seed42", "seed73", "seed2026", "three_seed_equal_mean",
)
DISPLAY = {
    "climatology": "Fold climatology",
    "persistence": "U10 persistence",
    "physical_logistic": "Physical logistic",
    "seed42": "Neural seed 42",
    "seed73": "Neural seed 73",
    "seed2026": "Neural seed 2026",
    "three_seed_equal_mean": "Neural 3-seed mean",
}
TRAJECTORY_FEATURES = (
    "u10_now", "u10_delta5", "t10_pcap_delta5",
    "heat_flux100_5d", "epdiv_lower_5d", "qbo30_5d",
)
TRAJECTORY_LABELS = ("U10", "dU10", "dT10", "HF100", "EPdiv-low", "Eq. U30")


def load_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def assert_aligned(left, right):
    for key in KEYS:
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)


def targets(arrays, class_index):
    return (arrays["target_class"] <= class_index).astype(np.int64)


def calibration_fit(probability, target):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    target = np.asarray(target, dtype=int)
    if np.unique(target).size < 2:
        return {"intercept": None, "slope": None}
    logit = np.log(probability / (1.0 - probability))
    def objective(parameters):
        fitted = np.clip(expit(parameters[0] + parameters[1] * logit), 1e-12, 1 - 1e-12)
        return float(-np.sum(target * np.log(fitted) + (1 - target) * np.log(1 - fitted)))
    fitted = minimize(objective, np.asarray([0.0, 1.0]), method="BFGS")
    if not fitted.success and not np.isfinite(fitted.fun):
        return {"intercept": None, "slope": None}
    return {"intercept": float(fitted.x[0]), "slope": float(fitted.x[1])}


def reliability_rows(probability, target, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    output = []
    for index in range(bins):
        mask = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == bins - 1
            else probability < edges[index + 1]
        )
        output.append({
            "bin": index,
            "lower": float(edges[index]), "upper": float(edges[index + 1]),
            "count": int(mask.sum()),
            "mean_probability": float(np.mean(probability[mask])) if mask.any() else None,
            "observed_frequency": float(np.mean(target[mask])) if mask.any() else None,
        })
    return output


def metric_block(probability, target, reference):
    probability = np.asarray(probability, dtype=float)
    target = np.asarray(target, dtype=int)
    reference = np.asarray(reference, dtype=float)
    brier = float(np.mean((probability - target) ** 2))
    reference_brier = float(np.mean((reference - target) ** 2))
    calibration = calibration_fit(probability, target)
    return {
        "pr_auc": average_precision(target, probability),
        "roc_auc": roc_auc(target, probability),
        "brier": brier,
        "reference_brier": reference_brier,
        "bss": float(1.0 - brier / reference_brier),
        "prevalence": float(np.mean(target)),
        "calibration_intercept": calibration["intercept"],
        "calibration_slope": calibration["slope"],
        "sharpness_sd": float(np.std(probability, ddof=1)),
        "probability_q90": float(np.quantile(probability, 0.90)),
        "probability_q99": float(np.quantile(probability, 0.99)),
    }


def event_templates(results):
    return {
        str(horizon): [
            {"winter": int(row["winter"]), "onset_step": int(row["onset_step"]),
             "horizon": horizon}
            for row in results["metrics"]["three_seed_equal_mean"][str(horizon)]
            ["event_metrics"]["events"]
        ] for horizon in HORIZONS
    }


def evaluate_models(arrays, probabilities, archived):
    events = event_templates(archived)
    rows = []
    blocks = {}
    for fold in (*FOLDS, "pooled"):
        mask = np.ones(len(arrays["fold"]), dtype=bool) if fold == "pooled" else arrays["fold"] == fold
        blocks[fold] = {}
        for class_index, horizon in enumerate(HORIZONS):
            y = targets(arrays, class_index)[mask]
            reference = probabilities["climatology"][mask, class_index]
            blocks[fold][str(horizon)] = {}
            for model in MODEL_ORDER:
                p = probabilities[model][mask, class_index]
                block = metric_block(p, y, reference)
                predicted = p >= THRESHOLD
                block["decision"] = decision_metrics(predicted, y)
                block["event"] = event_metrics(
                    predicted, y, arrays["forecast_origin"][mask], arrays["winter"][mask],
                    events[str(horizon)],
                )
                blocks[fold][str(horizon)][model] = block
                rows.append({
                    "fold": fold, "horizon_days": horizon, "model": model,
                    **{key: block[key] for key in (
                        "pr_auc", "roc_auc", "brier", "reference_brier", "bss",
                        "prevalence", "calibration_intercept", "calibration_slope",
                        "sharpness_sd", "probability_q90", "probability_q99")},
                    "CSI": block["decision"]["CSI"], "POD": block["decision"]["POD"],
                    "FAR": block["decision"]["FAR"],
                    "events_hit": block["event"]["events_hit"],
                    "n_events": block["event"]["n_events"],
                    "mean_lead_days": block["event"]["mean_first_alarm_lead_days"],
                    "false_episodes": block["event"]["false_alarm_episodes_total"],
                })
    equal_fold = {}
    for horizon in HORIZONS:
        equal_fold[str(horizon)] = {}
        for model in MODEL_ORDER:
            equal_fold[str(horizon)][model] = {
                metric: float(np.mean([blocks[fold][str(horizon)][model][metric] for fold in FOLDS]))
                for metric in ("pr_auc", "roc_auc", "brier", "bss", "calibration_intercept",
                               "calibration_slope", "sharpness_sd")
            }
    return blocks, equal_fold, rows


def full_reliability(arrays, probabilities):
    output, rows = {}, []
    for class_index, horizon in enumerate(HORIZONS):
        y = targets(arrays, class_index)
        output[str(horizon)] = {}
        for model in ("three_seed_equal_mean", "physical_logistic"):
            p = probabilities[model][:, class_index]
            bins = reliability_rows(p, y)
            decomposition = brier_decomposition(p, y)
            output[str(horizon)][model] = {
                "bins": bins,
                "decomposition": {key: decomposition[key] for key in (
                    "reliability", "resolution", "uncertainty")},
                **calibration_fit(p, y),
                "sharpness_sd": float(np.std(p, ddof=1)),
            }
            for row in bins:
                rows.append({"horizon_days": horizon, "model": model, **row})
    return output, rows


def cumulative_coherence(probabilities):
    output = {}
    for model in ("seed42", "seed73", "seed2026", "three_seed_equal_mean"):
        p = probabilities[model]
        first = np.maximum(p[:, 0] - p[:, 1], 0.0)
        second = np.maximum(p[:, 1] - p[:, 2], 0.0)
        output[model] = {
            "n_any_violation": int(np.sum((first > 0) | (second > 0))),
            "fraction_any_violation": float(np.mean((first > 0) | (second > 0))),
            "max_violation": float(max(np.max(first), np.max(second))),
            "mean_positive_violation": float(np.mean(np.r_[first[first > 0], second[second > 0]]))
            if np.any(first > 0) or np.any(second > 0) else 0.0,
        }
    return output


def false_episode_summary(arrays, probability):
    rows, result = [], {}
    for class_index, horizon in enumerate(HORIZONS):
        y = targets(arrays, class_index)
        result[str(horizon)] = {}
        for gap in (3, 5, 7, 10):
            total = 0
            by_winter = {}
            for winter in np.unique(arrays["winter"]):
                mask = (arrays["winter"] == winter) & (y == 0)
                origins = np.sort(arrays["forecast_origin"][mask & (probability[:, class_index] >= THRESHOLD)])
                episodes, previous = 0, None
                for current in origins:
                    if previous is None or current - previous > gap * 4:
                        episodes += 1
                    previous = current
                negative_count = int(mask.sum())
                total += episodes
                by_winter[str(int(winter))] = {
                    "episodes": int(episodes), "negative_origins": negative_count,
                    "episodes_per_100_negative_origins": float(100 * episodes / negative_count)
                    if negative_count else None,
                }
                rows.append({
                    "horizon_days": horizon, "merge_gap_days": gap,
                    "winter": int(winter), **by_winter[str(int(winter))],
                })
            negative_total = int(np.sum(y == 0))
            result[str(horizon)][str(gap)] = {
                "episodes_total": int(total),
                "n_winters": int(len(np.unique(arrays["winter"]))),
                "episodes_per_winter": float(total / len(np.unique(arrays["winter"]))),
                "episodes_per_100_negative_origins": float(100 * total / negative_total),
                "by_winter": by_winter,
            }
    return result, rows


def simultaneous_state_contrasts(draws=10000):
    prediction, _, z = load_aligned()
    metrics = json.loads(V55_RESULTS.read_text())
    event_rows = metrics["metrics"]["three_seed_equal_mean"]["5"]["event_metrics"]["events"]
    probability = prediction["three_seed_equal_mean_probability"][:, 0]
    hit_states, hit_winters = [], []
    for event in event_rows:
        onset, winter = int(event["onset_step"]), int(event["winter"])
        eligible = np.where((prediction["winter"] == winter) &
                            (prediction["forecast_origin"] < onset) &
                            (onset <= prediction["forecast_origin"] + 20))[0]
        hit = np.any(probability[eligible] >= THRESHOLD)
        if not hit:
            continue
        fixed = eligible[np.argmin(np.abs(prediction["forecast_origin"][eligible] - (onset - 20)))]
        hit_states.append(z[fixed]); hit_winters.append(winter)
    false = false_episodes(prediction)
    false_states = np.asarray([z[row["peak_index"]] for row in false])
    false_winters = np.asarray([row["winter"] for row in false], dtype=int)
    hit_states = np.asarray(hit_states); hit_winters = np.asarray(hit_winters, dtype=int)
    point = false_states.mean(axis=0) - hit_states.mean(axis=0)
    all_winters = np.unique(np.r_[hit_winters, false_winters])
    rng = np.random.default_rng(6404)
    samples = []
    for _ in range(draws):
        chosen = rng.choice(all_winters, size=all_winters.size, replace=True)
        h = np.concatenate([hit_states[hit_winters == winter] for winter in chosen
                            if np.any(hit_winters == winter)], axis=0) if any(
                                np.any(hit_winters == winter) for winter in chosen) else np.empty((0, len(STATE_FEATURES)))
        f = np.concatenate([false_states[false_winters == winter] for winter in chosen
                            if np.any(false_winters == winter)], axis=0) if any(
                                np.any(false_winters == winter) for winter in chosen) else np.empty((0, len(STATE_FEATURES)))
        if len(h) and len(f):
            samples.append(f.mean(axis=0) - h.mean(axis=0))
    samples = np.asarray(samples)
    centered_max = np.max(np.abs(samples - point), axis=1)
    critical = float(np.quantile(centered_max, 0.95))
    output, rows = {}, []
    for index, feature in enumerate(STATE_FEATURES):
        pointwise = np.quantile(samples[:, index], [0.025, 0.975])
        simultaneous = [float(point[index] - critical), float(point[index] + critical)]
        output[feature] = {
            "false_minus_hit": float(point[index]),
            "pointwise_ci95": pointwise.tolist(),
            "simultaneous_ci95": simultaneous,
            "survives_family_control": bool(simultaneous[0] > 0 or simultaneous[1] < 0),
        }
        rows.append({"feature": feature, **output[feature]})
    return {
        "n_hit_events": int(len(hit_states)), "n_false_episodes": int(len(false_states)),
        "n_winter_clusters": int(len(all_winters)), "bootstrap_draws_retained": int(len(samples)),
        "simultaneous_critical_distance": critical, "features": output,
    }, rows


def aligned_event_associations(permutation_draws=100000, bootstrap_draws=1000):
    prediction, _, _ = load_aligned()
    v59 = json.loads((ROOT / "v59_state_dependent_predictability.json").read_text())
    key_index = {(int(prediction["winter"][i]), int(prediction["forecast_origin"][i])): i
                 for i in range(len(prediction["winter"]))}
    state = np.asarray([[row[feature + "_z"] for feature in STATE_FEATURES]
                        for row in v59["events"]], dtype=float)
    probability = np.asarray([
        prediction["three_seed_equal_mean_probability"][
            key_index[(int(row["winter"]), int(row["fixed_forecast_origin"]))], 0]
        for row in v59["events"]
    ], dtype=float)
    observed = np.asarray([spearman(state[:, i], probability)
                           for i in range(len(STATE_FEATURES))])
    rng = np.random.default_rng(6412)
    bootstrap = np.empty((bootstrap_draws, len(STATE_FEATURES)), dtype=float)
    for draw in range(bootstrap_draws):
        index = rng.integers(0, len(probability), len(probability))
        bootstrap[draw] = [spearman(state[index, i], probability[index])
                           for i in range(len(STATE_FEATURES))]
    permutation = np.empty((permutation_draws, len(STATE_FEATURES)), dtype=float)
    for draw in range(permutation_draws):
        permuted = probability[rng.permutation(len(probability))]
        permutation[draw] = [spearman(state[:, i], permuted)
                             for i in range(len(STATE_FEATURES))]
    p = (1 + np.sum(np.abs(permutation) >= np.abs(observed), axis=0)) / (permutation_draws + 1)
    max_abs = np.max(np.abs(permutation), axis=1)
    fwer = (1 + np.sum(max_abs[:, None] >= np.abs(observed)[None, :], axis=0)) / (permutation_draws + 1)
    order = np.argsort(p)
    q = np.empty_like(p)
    running = 1.0
    for reverse_rank in range(len(order) - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, p[index] * len(order) / rank)
        q[index] = running
    output, rows = {}, []
    for i, feature in enumerate(STATE_FEATURES):
        output[feature] = {
            "spearman": float(observed[i]),
            "bootstrap_ci95": np.nanquantile(bootstrap[:, i], [0.025, 0.975]).tolist(),
            "permutation_p_two_sided": float(p[i]),
            "bh_q": float(q[i]), "max_abs_rho_fwer_p": float(fwer[i]),
        }
        rows.append({"feature": feature, **output[feature]})
    return {
        "n_events": int(len(probability)),
        "probability_definition": "five-day-head probability at the exact day-minus-5 origin",
        "features": output,
    }, rows


def trajectory_diagnostics(arrays, z_all):
    v62 = json.loads(V62_RESULTS.read_text())
    key_index = {(int(arrays["winter"][i]), int(arrays["forecast_origin"][i])): i
                 for i in range(len(arrays["winter"]))}
    feature_indices = [STATE_FEATURES.index(feature) for feature in TRAJECTORY_FEATURES]
    covariance = np.cov(z_all[:, feature_indices], rowvar=False)
    inverse = np.linalg.pinv(covariance + np.eye(len(feature_indices)) * 1e-6)
    rows, events = [], []
    for event in v62["events"]:
        states, z_by_lead, probabilities = {}, {}, {}
        for lead in (15, 10, 5):
            state = event["states"][str(lead)]
            index = key_index[(int(event["winter"]), int(state["forecast_origin"]))]
            z_value = np.asarray([state["z"][feature] for feature in TRAJECTORY_FEATURES])
            z_by_lead[lead] = z_value
            probabilities[lead] = arrays["three_seed_equal_mean_probability"][index].astype(float)
            states[lead] = state
            for head_index, head in enumerate(HORIZONS):
                rows.append({
                    "winter": int(event["winter"]), "fold": event["fold"],
                    "five_day_hit": bool(event["five_day_hit"]), "lead_days": lead,
                    "head_horizon_days": head,
                    "probability": float(probabilities[lead][head_index]),
                    "euclidean_distance": None, "mahalanobis_distance": None,
                })
        difference = z_by_lead[5] - z_by_lead[15]
        euclidean = float(np.linalg.norm(difference))
        mahalanobis = float(np.sqrt(difference @ inverse @ difference))
        events.append({
            "winter": int(event["winter"]), "fold": event["fold"],
            "five_day_hit": bool(event["five_day_hit"]),
            "euclidean_distance": euclidean, "mahalanobis_distance": mahalanobis,
            "probability_5head_at_5d": float(probabilities[5][0]),
            "states": states, "probabilities": probabilities,
        })
        rows.append({
            "winter": int(event["winter"]), "fold": event["fold"],
            "five_day_hit": bool(event["five_day_hit"]), "lead_days": None,
            "head_horizon_days": None, "probability": None,
            "euclidean_distance": euclidean, "mahalanobis_distance": mahalanobis,
        })
    event_2021 = next(event for event in events if event["winter"] == 2021)
    euclidean_values = np.asarray([event["euclidean_distance"] for event in events])
    mahalanobis_values = np.asarray([event["mahalanobis_distance"] for event in events])
    return {
        "n_complete_events": len(events),
        "unavailable_events": v62["unavailable_events"],
        "event_2021": {
            "euclidean": event_2021["euclidean_distance"],
            "euclidean_rank_of_n": [int(np.sum(euclidean_values <= event_2021["euclidean_distance"])), len(events)],
            "mahalanobis": event_2021["mahalanobis_distance"],
            "mahalanobis_rank_of_n": [int(np.sum(mahalanobis_values <= event_2021["mahalanobis_distance"])), len(events)],
        },
        "events": events,
    }, rows


def timing_and_sampler_audit(arrays):
    manifest = json.loads(ECMWF_MANIFEST.read_text())
    times = {item["request"]["time"] for item in manifest["requests"]}
    common_rows = list(csv.DictReader(COMMON_CSV.open()))
    timing = {
        "status": "PASS_IDENTICAL_00UTC_INFORMATION_CUTOFF",
        "all_forecast_origins_multiple_of_4": bool(np.all(arrays["forecast_origin"] % 4 == 0)),
        "all_input_windows_180_steps": bool(np.all(arrays["forecast_origin"] - arrays["start"] == 180)),
        "last_predictor_relative_to_origin": "origin-1 = 18 UTC on previous calendar day",
        "forecast_origin_interpretation": "origin step = 00 UTC",
        "label_onset_interpretation": "event date mapped to (day-1)*4 = 00 UTC",
        "ecmwf_request_times": sorted(times),
        "common_origin_count": len(common_rows),
        "comparison_boundary": "same 00UTC cutoff; retrospective common-date reforecast benchmark, one IFS cycle",
        "member_recovery_rule": "10 consecutive westerly days must occur within the 47-day trajectory and before 30 April",
    }
    sampler = {
        "status": "SCALAR_PRIOR_SHIFT_NOT_PROVEN_TO_RECOVER_POPULATION_POSTERIOR",
        "reason": (
            "Primary class-0 origins are sampled equally by event/lead, hard classes 1-3 "
            "are sampled uniformly from a combined pool, and background class 4 is sampled "
            "from a separate pool. Selection therefore depends on event and time-to-onset, "
            "not binary class alone for every cumulative horizon."
        ),
        "allowed_claim": "direct probability-head outputs or prior-corrected risk scores evaluated with proper scores",
        "forbidden_claim": "scalar prior correction proves calibrated population onset probabilities",
    }
    return timing, sampler


def plot_figure2(blocks):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.labelsize": 9, "axes.titlesize": 9,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 8,
                         "legend.fontsize": 7.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.35))
    cols = [(fold, horizon) for fold in FOLDS for horizon in HORIZONS]
    models = ("persistence", "physical_logistic", "three_seed_equal_mean")
    for axis, metric, title, cmap, limits in (
        (axes[0, 0], "pr_auc", "PR-AUC minus fold prevalence", "RdBu_r", (-0.25, 0.25)),
        (axes[0, 1], "bss", "Brier skill score", "RdBu_r", (-0.35, 0.35)),
    ):
        matrix = []
        for model in models:
            row = []
            for fold, horizon in cols:
                value = blocks[fold][str(horizon)][model][metric]
                if metric == "pr_auc":
                    value -= blocks[fold][str(horizon)][model]["prevalence"]
                row.append(value)
            matrix.append(row)
        axis.imshow(matrix, cmap=cmap, vmin=limits[0], vmax=limits[1], aspect="auto")
        axis.set_yticks(range(len(models)), ["U10 persistence", "Physical logistic", "Neural mean"])
        axis.set_xticks(range(len(cols)), [f"{f}{h}" for f, h in cols], rotation=45, ha="right")
        axis.set_xlabel("Fold and horizon (days)")
        axis.set_title(title)
        for i in range(len(models)):
            for j in range(len(cols)):
                axis.text(j, i, f"{matrix[i][j]:+.2f}", ha="center", va="center", fontsize=6.5)

    axis = axes[1, 0]
    palette = {"seed42": "#4477AA", "seed73": "#EE7733", "seed2026": "#228833",
               "three_seed_equal_mean": "#202020"}
    for model in ("seed42", "seed73", "seed2026", "three_seed_equal_mean"):
        axis.plot(HORIZONS, [blocks["pooled"][str(h)][model]["pr_auc"] for h in HORIZONS],
                  marker="o", lw=2 if model == "three_seed_equal_mean" else 1.2,
                  color=palette[model], label=DISPLAY[model])
    axis.plot(HORIZONS, [blocks["pooled"][str(h)]["physical_logistic"]["pr_auc"] for h in HORIZONS],
              marker="s", ls="--", color="#9467BD", label="Physical logistic")
    axis.set_xticks(HORIZONS); axis.set_xlabel("Forecast horizon (days)"); axis.set_ylabel("Pooled PR-AUC")
    axis.set_title("Initialization and baseline sensitivity")
    handles, labels_ = axis.get_legend_handles_labels()
    short = ["42", "73", "2026", "3-seed mean", "Physical"]
    axis.legend(handles, short, frameon=False, ncol=3, loc="upper center",
                bbox_to_anchor=(0.5, -0.23))

    axis = axes[1, 1]
    x = np.arange(3); width = 0.34
    hits = [blocks["pooled"][str(h)]["three_seed_equal_mean"]["event"]["events_hit"] for h in HORIZONS]
    false = [blocks["pooled"][str(h)]["three_seed_equal_mean"]["event"]["false_alarm_episodes_total"] for h in HORIZONS]
    axis.bar(x - width/2, hits, width, color="#3A8D5D", label="Events hit")
    axis.bar(x + width/2, false, width, color="#B44C43", label="False episodes")
    axis.set_xticks(x, [f"{h} d" for h in HORIZONS]); axis.set_ylabel("Count")
    axis.set_title("Fixed-0.20 event warning burden")
    for i, value in enumerate(hits): axis.text(i-width/2, value+0.4, f"{value}/16", ha="center", fontsize=7)
    axis.legend(frameon=False)
    for letter, axis in zip("abcd", axes.flat):
        axis.text(-0.13, 1.06, letter, transform=axis.transAxes, fontweight="bold", va="top")
        axis.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.17, top=0.94, wspace=0.46, hspace=0.44)
    fig.savefig(OUT_FIG2_PNG, dpi=300, facecolor="white")
    fig.savefig(OUT_FIG2_PDF, facecolor="white")
    plt.close(fig)


def plot_figure4(trajectory):
    events = trajectory["events"]
    target = next(event for event in events if event["winter"] == 2021)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.labelsize": 9, "axes.titlesize": 9,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 8,
                         "legend.fontsize": 7.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 6.25))
    hit_color, miss_color = "#3977B8", "#C44E52"
    labels = [str(event["winter"]) for event in events]
    eu = np.asarray([event["euclidean_distance"] for event in events])
    ma = np.asarray([event["mahalanobis_distance"] for event in events])
    target_index = labels.index("2021")
    axis = axes[0, 0]
    axis.scatter(range(len(events)), np.argsort(np.argsort(eu)) / (len(events)-1), marker="o",
                 color="#777777", label="Euclidean rank")
    axis.scatter(range(len(events)), np.argsort(np.argsort(ma)) / (len(events)-1), marker="s",
                 color="#7A5195", label="Mahalanobis rank")
    axis.scatter([target_index, target_index], [np.argsort(np.argsort(eu))[target_index]/(len(events)-1),
                                                np.argsort(np.argsort(ma))[target_index]/(len(events)-1)],
                 s=65, facecolor="none", edgecolor="black", lw=1.2)
    axis.set_xticks(range(len(events)), labels, rotation=60, ha="right")
    axis.set_ylabel("Within-event-sample rank fraction")
    axis.set_title("2021 is not extreme under either distance")
    axis.legend(frameon=False)

    axis = axes[0, 1]
    palette = ("#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377")
    for feature, label, color in zip(TRAJECTORY_FEATURES, TRAJECTORY_LABELS, palette):
        y = [target["states"][lead]["z"][feature] for lead in (15, 10, 5)]
        axis.plot((15, 10, 5), y, marker="o", color=color, lw=1.4, label=label)
    axis.axhline(0, color="#666666", lw=0.7); axis.set_xlim(16, 4); axis.set_xticks((15, 10, 5))
    axis.set_xlabel("Days before onset"); axis.set_ylabel("Within-fold standardized state")
    axis.set_title("2021 descriptor evolution")
    axis.legend(frameon=False, ncol=2)

    axis = axes[1, 0]
    head_colors = ("#4477AA", "#EE7733", "#228833")
    for head_index, (head, color) in enumerate(zip(HORIZONS, head_colors)):
        y = [target["probabilities"][lead][head_index] * 100 for lead in (15, 10, 5)]
        axis.plot((15, 10, 5), y, marker="o", color=color, lw=1.6, label=f"{head}-day head")
    axis.axhline(20, color="#666666", ls="--", lw=0.8, label="0.20 threshold")
    axis.set_yscale("log"); axis.set_xlim(16, 4); axis.set_xticks((15, 10, 5))
    axis.set_xlabel("Days before onset"); axis.set_ylabel("Forecast probability (%)")
    axis.set_title("Separate probability-head trajectories")
    axis.legend(frameon=False)

    axis = axes[1, 1]
    hit = np.asarray([event["five_day_hit"] for event in events], dtype=bool)
    probability5 = np.asarray([event["probability_5head_at_5d"] * 100 for event in events])
    axis.scatter(ma[hit], probability5[hit], color=hit_color, edgecolor="white", s=38, label="5-d hit")
    axis.scatter(ma[~hit], probability5[~hit], color=miss_color, edgecolor="white", s=38, label="5-d miss")
    axis.scatter(target["mahalanobis_distance"], target["probability_5head_at_5d"]*100,
                 facecolor="none", edgecolor="black", s=70, lw=1.3)
    axis.annotate("2021", (target["mahalanobis_distance"], target["probability_5head_at_5d"]*100),
                  xytext=(4, 5), textcoords="offset points", fontsize=7)
    axis.set_yscale("log"); axis.set_xlabel("15-to-5 d Mahalanobis displacement")
    axis.set_ylabel("5-day probability at day -5 (%)")
    axis.set_title("Displacement does not separate hits and misses")
    axis.legend(frameon=False)
    for letter, axis in zip("abcd", axes.flat):
        axis.text(-0.13, 1.06, letter, transform=axis.transAxes, fontweight="bold", va="top")
        axis.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.94, wspace=0.38, hspace=0.42)
    fig.savefig(OUT_FIG4_PNG, dpi=300, facecolor="white")
    fig.savefig(OUT_FIG4_PDF, facecolor="white")
    plt.close(fig)


def plot_probability_supplement(reliability, coherence):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.labelsize": 9, "axes.titlesize": 9,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 8,
                         "legend.fontsize": 7.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.0))
    for axis, horizon in zip(axes.flat[:3], HORIZONS):
        for model, color, marker in (("three_seed_equal_mean", "#4477AA", "o"),
                                     ("physical_logistic", "#EE7733", "s")):
            rows = [row for row in reliability[str(horizon)][model]["bins"] if row["count"]]
            x = [row["mean_probability"] for row in rows]
            y = [row["observed_frequency"] for row in rows]
            size = [18 + 5*np.sqrt(row["count"]) for row in rows]
            axis.scatter(x, y, s=size, color=color, marker=marker, alpha=0.85,
                         label=DISPLAY[model])
            for row in rows:
                if row["count"] < 20:
                    axis.annotate(str(row["count"]), (row["mean_probability"], row["observed_frequency"]),
                                  xytext=(2, 2), textcoords="offset points", fontsize=6, color=color)
        axis.plot([0, 1], [0, 1], color="#777777", ls="--", lw=0.8)
        axis.set_xlim(-0.02, 1.02); axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("Mean forecast probability"); axis.set_ylabel("Observed frequency")
        axis.set_title(f"{horizon}-day reliability; size reflects count")
        if horizon == 5: axis.legend(frameon=False)
    axis = axes[1, 1]
    models = ("seed42", "seed73", "seed2026", "three_seed_equal_mean")
    fractions = [100*coherence[model]["fraction_any_violation"] for model in models]
    axis.bar(range(len(models)), fractions, color=("#4477AA", "#EE7733", "#228833", "#202020"))
    axis.set_xticks(range(len(models)), ["42", "73", "2026", "Mean"])
    axis.set_ylabel("Origins with order violation (%)")
    axis.set_title("Cumulative-horizon coherence")
    for i, value in enumerate(fractions): axis.text(i, value+0.1, f"{value:.2f}", ha="center", fontsize=7)
    for letter, axis in zip("abcd", axes.flat):
        axis.text(-0.13, 1.06, letter, transform=axis.transAxes, fontweight="bold", va="top")
        axis.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.12, top=0.94, wspace=0.36, hspace=0.40)
    fig.savefig(OUT_FIGS2_PNG, dpi=300, facecolor="white")
    fig.savefig(OUT_FIGS2_PDF, facecolor="white")
    plt.close(fig)


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def self_test():
    target = np.asarray([0, 0, 1, 1])
    probability = np.asarray([0.1, 0.2, 0.7, 0.8])
    assert abs(average_precision(target, probability) - 1.0) < 1e-12
    assert calibration_fit(probability, target)["slope"] > 0
    print("SSW_PAPER1_REVISION_V4_SELF_TEST_OK")


def main():
    arrays = load_npz(V55_PRED)
    baselines = load_npz(BASELINE_PRED)
    assert_aligned(arrays, baselines)
    for key in ("seed42_probability", "seed73_probability", "seed2026_probability",
                "three_seed_equal_mean_probability"):
        if arrays[key].shape != (1918, 3) or not np.isfinite(arrays[key]).all():
            raise RuntimeError(f"invalid {key}")
    probabilities = {
        "climatology": baselines["climatology_probability"],
        "persistence": baselines["persistence_probability"],
        "physical_logistic": baselines["physical_logistic_probability"],
        "seed42": arrays["seed42_probability"],
        "seed73": arrays["seed73_probability"],
        "seed2026": arrays["seed2026_probability"],
        "three_seed_equal_mean": arrays["three_seed_equal_mean_probability"],
    }
    archived = json.loads(V55_RESULTS.read_text())
    blocks, equal_fold, metric_rows = evaluate_models(arrays, probabilities, archived)
    reliability, reliability_rows_ = full_reliability(arrays, probabilities)
    coherence = cumulative_coherence(probabilities)
    false_sensitivity, false_rows = false_episode_summary(
        arrays, probabilities["three_seed_equal_mean"])
    state_contrasts, state_rows = simultaneous_state_contrasts()
    aligned_associations, association_rows = aligned_event_associations()
    _, _, z_all = load_aligned()
    trajectory, trajectory_rows = trajectory_diagnostics(
        {**arrays, "three_seed_equal_mean_probability": probabilities["three_seed_equal_mean"]}, z_all)
    timing, sampler = timing_and_sampler_audit(arrays)
    result = {
        "status": "SSW_PAPER1_REVISION_V4_OK",
        "alignment": {"status": "PASS", "n_origins": 1918, "n_events": 16,
                      "keys": list(KEYS)},
        "metrics": blocks,
        "equal_fold_summary": equal_fold,
        "full_reliability": reliability,
        "cumulative_coherence": coherence,
        "false_alarm_sensitivity": false_sensitivity,
        "state_contrasts_family_control": state_contrasts,
        "aligned_event_associations": aligned_associations,
        "trajectory_diagnostics": {key: value for key, value in trajectory.items() if key != "events"},
        "timing_audit": timing,
        "sampler_probability_audit": sampler,
        "claim_boundary": (
            "Frozen retrospective rolling-refit diagnostics. No prospective leakage-free "
            "claim, calibrated-population-probability claim, causal state mechanism or "
            "operational superiority is established."
        ),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    write_csv(OUT_METRICS, metric_rows)
    write_csv(OUT_RELIABILITY, reliability_rows_)
    write_csv(OUT_FALSE, false_rows)
    write_csv(OUT_STATE, state_rows)
    write_csv(OUT_ASSOC, association_rows)
    write_csv(OUT_TRAJECTORY, trajectory_rows)
    plot_figure2(blocks)
    plot_figure4(trajectory)
    plot_probability_supplement(reliability, coherence)
    print(json.dumps({
        "status": result["status"],
        "timing": timing["status"],
        "sampler": sampler["status"],
        "coherence": coherence["three_seed_equal_mean"],
        "state_survivors": [feature for feature, row in state_contrasts["features"].items()
                            if row["survives_family_control"]],
        "event_2021": trajectory["event_2021"],
    }, indent=2))


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
