#!/usr/bin/env python3
"""V65 matched-update natural-frequency discrete-time SSW hazard model."""

import argparse
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

import ssw_train_v50_1deg_rolling as v50


CFG = v50.CFG
FOLDS = v50.FOLDS
SEEDS = (42, 73, 2026)
HAZARD_DAYS = 15
FIXED_TRAIN_EPOCHS = 21
FIXED_ALARM_THRESHOLD = 0.20
OUT_DIR = Path(CFG["out_dir"])


def protocol():
    return {
        "name": "v65_natural_frequency_discrete_hazard",
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "hazard_days": HAZARD_DAYS,
        "sampling": "uniform without replacement from all eligible training origins",
        "epoch_budget": "matched to v52 event-balanced samples and updates",
        "hazard_loss": "unweighted at-risk discrete-time binary log likelihood",
        "prior_shift": "none",
        "probability_calibration": "none",
        "fixed_train_epochs": FIXED_TRAIN_EPOCHS,
        "optimizer_schedule_epochs": 45,
        "fixed_alarm_threshold": FIXED_ALARM_THRESHOLD,
        "checkpoint_selection": "fixed epoch; outer test instantiated after training",
        "outer_test_selection": "forbidden",
    }


def configure_fold(fold_name):
    if fold_name not in FOLDS:
        raise ValueError("Unknown fold %s" % fold_name)
    fold = FOLDS[fold_name]
    CFG.update({
        "version": "v65_natural_hazard_fold%s" % fold_name,
        "train_end_year": fold["train_end_year"],
        "val_end_year": fold["test_end_year"],
        "fixed_train_epochs": FIXED_TRAIN_EPOCHS,
        "fixed_alarm_threshold": FIXED_ALARM_THRESHOLD,
        "rolling_fold": fold_name,
        "epochs": 45,
    })
    return fold


def first_event_day(start, ssw_idx):
    origin = int(start) + CFG["input_steps"]
    future = ssw_idx[
        (ssw_idx > origin) &
        (ssw_idx <= origin + HAZARD_DAYS * 4)
    ]
    if len(future) == 0:
        return 0
    delta = int(future.min() - origin)
    day = (delta + 3) // 4
    if not 1 <= day <= HAZARD_DAYS:
        raise ValueError("Invalid hazard event day %d" % day)
    return day


class HazardDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.starts = base_dataset.starts
        self.labels = base_dataset.labels
        self.winters = base_dataset.winters
        self.ssw_idx = base_dataset.ssw_idx
        self.event_days = np.asarray([
            first_event_day(start, self.ssw_idx) for start in self.starts
        ], dtype=np.int64)
        for class_id, horizon in enumerate((5, 10, 15)):
            expected = self.labels <= class_id
            actual = (self.event_days > 0) & (self.event_days <= horizon)
            np.testing.assert_array_equal(
                expected, actual,
                err_msg="hazard/cumulative target mismatch at %d days" % horizon)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        item["event_day"] = torch.tensor(self.event_days[index]).long()
        return item


class UniformNaturalEpochSampler(Sampler):
    def __init__(self, dataset_size, epoch_samples, seed):
        if epoch_samples > dataset_size:
            raise ValueError(
                "Matched epoch samples %d exceed eligible origins %d" %
                (epoch_samples, dataset_size))
        self.dataset_size = int(dataset_size)
        self.epoch_samples = int(epoch_samples)
        self.generator = torch.Generator().manual_seed(int(seed) + 65000)
        self.last_indices = None

    def __len__(self):
        return self.epoch_samples

    def __iter__(self):
        indices = torch.randperm(
            self.dataset_size, generator=self.generator)[:self.epoch_samples]
        self.last_indices = indices.numpy()
        return iter(indices.tolist())


class NaturalHazardModel(v50.pilot.base.SSW3DNetV43):
    def __init__(self):
        super().__init__()
        input_dim = self.time_head[0].in_features
        hidden_dim = self.time_head[0].out_features
        self.time_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Dropout(CFG["dropout"]),
            nn.Linear(hidden_dim, HAZARD_DAYS),
        )


def hazard_targets(event_day, device):
    event_day = event_day.to(device)
    day = torch.arange(1, HAZARD_DAYS + 1, device=device).unsqueeze(0)
    observed = event_day.unsqueeze(1)
    target = (day == observed).float()
    at_risk = ((observed == 0) | (day <= observed)).float()
    return target, at_risk


def hazard_loss(logits, event_day):
    target, at_risk = hazard_targets(event_day, logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * at_risk).sum() / at_risk.sum().clamp_min(1.0)


def cumulative_probability(hazard_logits):
    hazard = torch.sigmoid(hazard_logits.float()).clamp(1e-7, 1.0 - 1e-7)
    survival = torch.cumprod(1.0 - hazard, dim=1)
    daily = 1.0 - survival
    return daily, daily[:, [4, 9, 14]], hazard


def training_hazard_prior(event_days):
    event_days = np.asarray(event_days, dtype=np.int64)
    events = int(np.sum(event_days > 0))
    risk_days = int(np.where(event_days > 0, event_days, HAZARD_DAYS).sum())
    probability = events / max(1, risk_days)
    probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    return probability, float(np.log(probability / (1.0 - probability))), events, risk_days


def evaluate_outer(model, loader, ssw_idx, season_of_step, fold_name,
                   stem, device, amp_enabled):
    model.eval()
    daily_probability, daily_hazard = [], []
    targets, starts, winters = [], [], []
    vortex_predictions, vortex_targets = [], []
    change_predictions, change_targets = [], []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["x"].to(device)
            qbo = batch["q"].to(device)
            ep_input = batch["ep_in"].to(device) if CFG["ep_input"] else None
            with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=amp_enabled):
                logits, vortex, change = model(
                    inputs, qbo, ep_input, return_aux=True)
            cumulative, _, hazard = cumulative_probability(logits)
            daily_probability.append(cumulative.cpu())
            daily_hazard.append(hazard.cpu())
            targets.append(batch["y"])
            starts.append(batch["start"])
            winters.append(batch["winter"])
            vortex_predictions.append(vortex.float().cpu())
            vortex_targets.append(batch["vortex_target"])
            change_predictions.append(change.float().cpu())
            change_targets.append(batch["vortex_change_target"])

    cumulative_daily = torch.cat(daily_probability).numpy()
    hazard_daily = torch.cat(daily_hazard).numpy()
    probability = cumulative_daily[:, [4, 9, 14]]
    target_class = torch.cat(targets).numpy()
    starts_array = torch.cat(starts).numpy()
    winters_array = torch.cat(winters).numpy()
    vortex_prediction = torch.cat(vortex_predictions).numpy()
    vortex_target = torch.cat(vortex_targets).numpy()
    change_prediction = torch.cat(change_predictions).numpy()
    change_target = torch.cat(change_targets).numpy()

    if not (np.isfinite(probability).all() and
            np.all(np.diff(probability, axis=1) >= -1e-7)):
        raise RuntimeError("Hazard cumulative probabilities are invalid")
    metrics = v50.pilot.base.horizon_probability_metrics(
        probability, target_class)
    for class_id, horizon in enumerate((5, 10, 15)):
        entry = metrics[str(horizon)]
        entry.pop("best_threshold_metrics", None)
        binary_target = (target_class <= class_id).astype(np.int64)
        predicted = probability[:, class_id] >= FIXED_ALARM_THRESHOLD
        decision = v50.pilot.base.probability_metrics(
            predicted.astype(np.float32), binary_target, 0.5)
        decision.pop("threshold", None)
        thresholds = {
            str(int(winter)): FIXED_ALARM_THRESHOLD
            for winter in np.unique(winters_array)
        }
        entry["fixed_threshold_metrics"] = decision
        event = v50.pilot.base.event_diagnostics(
            predicted, binary_target, starts_array, winters_array,
            ssw_idx, season_of_step, horizon, thresholds)
        event["note"] = (
            "Fixed 0.20 threshold; no outer-test threshold, calibration, "
            "checkpoint, seed or model selection.")
        entry["event_metrics"] = event

    prediction_path = OUT_DIR / (
        "%s_fixed_epoch%d_predictions.npz" % (stem, FIXED_TRAIN_EPOCHS))
    np.savez_compressed(
        prediction_path,
        horizon_probability=probability,
        daily_cumulative_probability=cumulative_daily,
        daily_hazard_probability=hazard_daily,
        target_class=target_class,
        start=starts_array,
        forecast_origin=starts_array + CFG["input_steps"],
        winter=winters_array,
        vortex_prediction=vortex_prediction,
        vortex_target=vortex_target,
        vortex_change_prediction=change_prediction,
        vortex_change_target=change_target,
        horizons_days=np.asarray([5, 10, 15]),
        hazard_days=np.arange(1, HAZARD_DAYS + 1),
        fixed_alarm_threshold=np.asarray([FIXED_ALARM_THRESHOLD]),
    )
    return {
        "fold": fold_name,
        "prediction_file": str(prediction_path),
        "n_test_samples": int(len(target_class)),
        "horizons": metrics,
        "cumulative_order_violations": int(
            np.sum(np.any(np.diff(probability, axis=1) < -1e-7, axis=1))),
        "vortex_aux_rmse": float(np.sqrt(np.mean(
            (vortex_prediction - vortex_target) ** 2))),
        "vortex_change_aux_rmse": float(np.sqrt(np.mean(
            (change_prediction - change_target) ** 2))),
    }


def run_fold(fold_name, seed):
    if seed not in SEEDS:
        raise ValueError("Unregistered seed %d" % seed)
    fold = configure_fold(fold_name)
    v50.pilot.verify_core_source()
    v50.pilot.base.set_seed(seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("V65 requires a healthy H100")
    stem = "%s_seed%d" % (CFG["version"], seed)
    print(json.dumps({"fold": fold, "seed": seed, "protocol": protocol()},
                     indent=2), flush=True)

    all_files, all_steps, all_cum, winter_mask = v50.pilot._build_all_file_list()
    data, qbo, steps_pf, files, norm_stats = v50.pilot.preload_uvt()
    cumulative = np.cumsum(np.concatenate([[0], steps_pf]))
    ep, ep_steps, ep_cumulative, ep_mean, ep_std = v50.pilot.preload_ep_aligned(
        steps_pf, files)
    norm_stats["ep_mean"] = ep_mean
    norm_stats["ep_std"] = ep_std
    np.savez(OUT_DIR / ("norm_stats_%s_seed%d.npz" % (CFG["version"], seed)),
             **norm_stats)
    ssw_idx = v50._labels_in_winter_space(all_cum, all_steps, winter_mask)
    season_of_step, year_of_step, _ = v50.pilot.base.assign_step_seasons(
        steps_pf, files)

    base_train = v50.pilot.base.SSWDatasetV43(
        data, qbo, steps_pf, files, ep, ep_steps, ep_cumulative,
        ssw_idx, season_of_step, year_of_step, cumulative,
        split="train", train_end_year=fold["train_end_year"],
        val_end_year=fold["test_end_year"])
    matched_index = v50.PrimaryEventIndex(base_train)
    train_dataset = HazardDataset(base_train)
    sampler = UniformNaturalEpochSampler(
        len(train_dataset), len(matched_index), seed)
    train_loader = DataLoader(
        train_dataset, CFG["batch_size"], sampler=sampler,
        num_workers=CFG["num_workers"], drop_last=True)
    if len(train_loader) != len(matched_index) // CFG["batch_size"]:
        raise RuntimeError("Optimizer-update matching failed")

    model = NaturalHazardModel().to(device)
    hazard_prior, hazard_bias, n_events, risk_days = training_hazard_prior(
        train_dataset.event_days)
    nn.init.constant_(model.time_head[-1].bias, hazard_bias)
    ema_model = copy.deepcopy(model).to(device).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = v50._scheduler(optimizer)
    amp_enabled = CFG["use_bf16"] and torch.cuda.is_bf16_supported()
    print(
        "GPU=%s eligible=%d matched_samples=%d updates=%d train_events=%d "
        "hazard_prior=%.8f risk_days=%d" % (
            torch.cuda.get_device_name(0), len(train_dataset), len(sampler),
            len(train_loader), len(matched_index.onsets), hazard_prior, risk_days),
        flush=True)

    log_path = OUT_DIR / ("train_%s_log.txt" % stem)
    checkpoint_path = OUT_DIR / (
        "ssw_%s_fixed_epoch%d.pt" % (stem, FIXED_TRAIN_EPOCHS))
    history = []
    started = time.time()
    with log_path.open("w") as log_file:
        for epoch in range(FIXED_TRAIN_EPOCHS):
            model.train()
            total_loss = total_hazard = 0.0
            for batch_index, batch in enumerate(train_loader):
                inputs = batch["x"].to(device, non_blocking=True)
                qbo_input = batch["q"].to(device, non_blocking=True)
                ep_input = (batch["ep_in"].to(device, non_blocking=True)
                            if CFG["ep_input"] else None)
                optimizer.zero_grad()
                with torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16,
                        enabled=amp_enabled):
                    logits, vortex, change = model(
                        inputs, qbo_input, ep_input, return_aux=True)
                    likelihood = hazard_loss(logits, batch["event_day"])
                    vortex_loss = F.smooth_l1_loss(
                        vortex.float(), batch["vortex_target"].to(device))
                    change_loss = F.smooth_l1_loss(
                        change.float(), batch["vortex_change_target"].to(device))
                    loss = (
                        likelihood +
                        CFG["vortex_level_aux_loss_weight"] * vortex_loss +
                        CFG["vortex_change_aux_loss_weight"] * change_loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), CFG["grad_clip_norm"])
                optimizer.step()
                v50.pilot.base._update_ema(ema_model, model)
                total_loss += loss.item()
                total_hazard += likelihood.item()
                if batch_index > 0 and batch_index % 50 == 0:
                    print("  epoch %d batch %d/%d" % (
                        epoch, batch_index, len(train_loader)), flush=True)
            sampled_counts = np.bincount(
                train_dataset.labels[sampler.last_indices], minlength=5).tolist()
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "loss": total_loss / len(train_loader),
                "hazard_loss": total_hazard / len(train_loader),
                "sampled_time_class_counts": sampled_counts,
                "elapsed_minutes": int((time.time() - started) / 60),
            }
            history.append(row)
            line = json.dumps(row)
            print(line, flush=True)
            log_file.write(line + "\n")
            log_file.flush()
            scheduler.step()

        torch.save({
            "model_state": ema_model.state_dict(),
            "epoch": FIXED_TRAIN_EPOCHS - 1,
            "cfg": {key: value for key, value in CFG.items()
                    if isinstance(value, (int, float, str, bool, list, tuple))},
            "protocol": protocol(),
            "fold": fold,
            "seed": int(seed),
            "eligible_train_origins": len(train_dataset),
            "matched_epoch_samples": len(sampler),
            "optimizer_updates_per_epoch": len(train_loader),
            "training_hazard_prior": hazard_prior,
            "training_hazard_bias": hazard_bias,
            "training_events_within_15_days": n_events,
            "training_risk_days": risk_days,
            "history": history,
        }, checkpoint_path)

    base_test = v50.pilot.base.SSWDatasetV43(
        data, qbo, steps_pf, files, ep, ep_steps, ep_cumulative,
        ssw_idx, season_of_step, year_of_step, cumulative,
        split="val", train_end_year=fold["train_end_year"],
        val_end_year=fold["test_end_year"])
    test_dataset = HazardDataset(base_test)
    test_loader = DataLoader(
        test_dataset, CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"])
    outer = evaluate_outer(
        ema_model, test_loader, ssw_idx, season_of_step,
        fold_name, stem, device, amp_enabled)
    result = {
        "status": "V65_NATURAL_HAZARD_FOLD_OK",
        "protocol": protocol(),
        "fold": fold,
        "seed": int(seed),
        "checkpoint": str(checkpoint_path),
        "eligible_train_origins": len(train_dataset),
        "matched_epoch_samples": len(sampler),
        "optimizer_updates_per_epoch": len(train_loader),
        "training_hazard_prior": hazard_prior,
        "outer_test": outer,
    }
    result_path = OUT_DIR / ("%s_seed%d_results.json" % (CFG["version"], seed))
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print("V65_OUTER_TEST_DONE %s" % result_path, flush=True)


def run_self_test():
    v50.pilot.run_self_test()
    for fold_name, fold in FOLDS.items():
        configure_fold(fold_name)
        assert fold["train_end_year"] < fold["test_end_year"]
    origin = CFG["input_steps"]
    assert first_event_day(0, np.asarray([origin + 1])) == 1
    assert first_event_day(0, np.asarray([origin + 5 * 4])) == 5
    assert first_event_day(0, np.asarray([origin + 15 * 4])) == 15
    assert first_event_day(0, np.asarray([origin + 15 * 4 + 1])) == 0
    logits = torch.zeros((2, HAZARD_DAYS))
    daily, horizons, _ = cumulative_probability(logits)
    assert daily.shape == (2, HAZARD_DAYS) and horizons.shape == (2, 3)
    assert torch.all(torch.diff(horizons, dim=1) >= 0)
    targets, risk = hazard_targets(torch.tensor([3, 0]), torch.device("cpu"))
    assert targets[0].sum() == 1 and risk[0].sum() == 3
    assert targets[1].sum() == 0 and risk[1].sum() == HAZARD_DAYS
    sampler = UniformNaturalEpochSampler(100, 60, 42)
    values = list(iter(sampler))
    assert len(values) == len(set(values)) == 60
    model = NaturalHazardModel()
    assert model.time_head[-1].out_features == HAZARD_DAYS
    print("V65_NATURAL_HAZARD_SELF_TEST_OK", flush=True)


def run_data_check():
    v50.pilot.verify_core_source()
    files, steps, cumulative, winter_mask = v50.pilot._build_all_file_list()
    labels = v50._labels_in_winter_space(cumulative, steps, winter_mask)
    seasons, _, _ = v50.pilot.base.assign_step_seasons(steps, files)
    folds = {}
    for fold_name, fold in FOLDS.items():
        test_events = seasons[labels][
            (seasons[labels] > fold["train_end_year"]) &
            (seasons[labels] <= fold["test_end_year"])]
        if len(test_events) != fold["expected_test_events"]:
            raise RuntimeError("Fold %s event count changed" % fold_name)
        folds[fold_name] = {
            "train_end_year": fold["train_end_year"],
            "test_end_year": fold["test_end_year"],
            "test_event_count": int(len(test_events)),
            "test_event_winters": test_events.tolist(),
        }
    report = {
        "status": "V65_NATURAL_HAZARD_DATA_CHECK_OK",
        "protocol": protocol(),
        "n_months": len(files),
        "n_time_steps": int(np.sum(steps)),
        "source_audit": CFG["audit_json"],
        "folds": folds,
        "runtime_assertions": [
            "hazard targets equal cumulative 5/10/15 labels",
            "uniform sample unique within epoch",
            "matched sample count <= eligible train origins",
            "optimizer updates match v52 sampler",
            "outer test instantiated after checkpoint",
        ],
    }
    path = OUT_DIR / "v65_natural_hazard_data_check.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=tuple(FOLDS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--data-check", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
    elif args.data_check:
        run_data_check()
    else:
        if args.fold is None:
            parser.error("--fold is required for training")
        run_fold(args.fold, args.seed)


if __name__ == "__main__":
    main()
