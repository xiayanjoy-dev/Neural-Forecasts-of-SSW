"""Leakage-controlled rolling external test for the audited 1-degree v43 control.

Each fold refits the unchanged v43_1deg_e2021 model through a fixed ten
epochs.  The outer test winter block is never constructed or evaluated until
training has stopped.  The fixed 0.20 alarm probability is a pre-registered
decision policy, not a CSI threshold selected on the outer test data.
"""
import copy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import ssw_train_v43_1deg_e2021 as pilot


CFG = pilot.CFG
FIXED_TRAIN_EPOCHS = 10
FIXED_ALARM_THRESHOLD = 0.20
FOLDS = {
    "A": {"train_end_year": 1999, "test_end_year": 2004,
          "expected_test_events": 6},
    "B": {"train_end_year": 2004, "test_end_year": 2010,
          "expected_test_events": 6},
    "C": {"train_end_year": 2010, "test_end_year": 2021,
          "expected_test_events": 4},
}
PROTOCOL = {
    "name": "v50_1deg_fixed10_rolling",
    "fixed_train_epochs": FIXED_TRAIN_EPOCHS,
    "optimizer_schedule_epochs": 45,
    "alarm_probability_threshold": FIXED_ALARM_THRESHOLD,
    "checkpoint_selection": "fixed epoch; outer test never evaluated during training",
    "calibration": "none; raw prior-corrected probabilities",
    "outer_test_threshold_selection": "forbidden",
}


def configure_fold(fold_name):
    if fold_name not in FOLDS:
        raise ValueError("Unknown fold %s" % fold_name)
    fold = FOLDS[fold_name]
    CFG.update({
        "version": "v50_1deg_fixed10_fold%s" % fold_name,
        "train_end_year": fold["train_end_year"],
        "val_end_year": fold["test_end_year"],
        "fixed_train_epochs": FIXED_TRAIN_EPOCHS,
        "fixed_alarm_threshold": FIXED_ALARM_THRESHOLD,
        "rolling_fold": fold_name,
    })
    # Keep v43's 45-epoch scheduler shape.  Only the stopping epoch is fixed.
    CFG["epochs"] = 45
    return fold


def _labels_in_winter_space(all_cum, all_steps, winter_mask):
    full_labels = pilot.load_ssw_labels()
    labels = [
        pilot.base._full2winter_step(label, all_cum, all_steps, winter_mask)
        for label in full_labels
    ]
    if any(label < 0 for label in labels):
        raise ValueError("An onset was lost when mapping to winter space")
    return np.asarray(sorted(labels), dtype=np.int64)


class PrimaryEventIndex(Dataset):
    """Unchanged v43 event-balanced sampling policy."""
    def __init__(self, dataset):
        self.dataset = dataset
        positive_indices = np.where(dataset.labels == 0)[0]
        self.hard_indices = np.where(
            (dataset.labels >= 1) &
            (dataset.labels < CFG["n_time_classes"] - 1)
        )[0]
        self.background_indices = np.where(
            dataset.labels == CFG["n_time_classes"] - 1
        )[0]
        self.positive_by_onset = {}
        for index in positive_indices:
            origin = dataset.starts[index] + CFG["input_steps"]
            future = dataset.ssw_idx[
                (dataset.ssw_idx > origin) &
                (dataset.ssw_idx <= origin + CFG["lead_steps"])
            ]
            if len(future) != 1:
                raise ValueError("Each primary positive needs one future onset")
            self.positive_by_onset.setdefault(int(future[0]), []).append(index)
        self.onsets = sorted(self.positive_by_onset)
        if (not self.onsets or len(self.hard_indices) == 0 or
                len(self.background_indices) == 0):
            raise ValueError("Primary, hard-negative and background pools are required")
        self.blocks = len(self.onsets) * CFG["positive_draws_per_event"]

    def __len__(self):
        return self.blocks * (CFG["negative_per_positive"] + 1)

    def __getitem__(self, index):
        block, offset = divmod(index, CFG["negative_per_positive"] + 1)
        if offset == 0:
            onset = self.onsets[block % len(self.onsets)]
            candidates = self.positive_by_onset[onset]
            if CFG["deterministic_positive_leads"]:
                draw = block // len(self.onsets)
                source_index = candidates[draw % len(candidates)]
            else:
                source_index = np.random.choice(candidates)
        elif offset <= CFG["hard_negatives_per_positive"]:
            source_index = np.random.choice(self.hard_indices)
        else:
            source_index = np.random.choice(self.background_indices)
        return self.dataset[source_index]


def _prior_shift(train_dataset, device):
    class_counts = np.bincount(
        train_dataset.labels, minlength=CFG["n_time_classes"]
    ).astype(np.float64)
    train_prior = (
        np.cumsum(class_counts)[:len(CFG["lead_times"])] / class_counts.sum()
    )
    hard_slice = slice(1, CFG["n_time_classes"] - 1)
    sample_prior = np.zeros(CFG["n_time_classes"])
    block_size = CFG["negative_per_positive"] + 1
    sample_prior[0] = 1.0 / block_size
    hard_fraction = CFG["hard_negatives_per_positive"] / block_size
    hard_count = class_counts[hard_slice].sum()
    if hard_count <= 0:
        raise ValueError("No hard negatives in this fold")
    sample_prior[hard_slice] = (
        hard_fraction * class_counts[hard_slice] / hard_count
    )
    sample_prior[-1] = (
        (CFG["negative_per_positive"] - CFG["hard_negatives_per_positive"])
        / block_size
    )
    sampled_prior = np.cumsum(sample_prior)[:len(CFG["lead_times"])]

    def logit(probability):
        probability = np.clip(probability, 1e-7, 1.0 - 1e-7)
        return np.log(probability / (1.0 - probability))

    shift = logit(train_prior) - logit(sampled_prior)
    return (train_prior, sampled_prior, shift,
            torch.tensor(shift, dtype=torch.float32, device=device))


def _scheduler(optimizer):
    warmup = CFG["warmup_epochs"]
    eta_ratio = CFG["lr_eta_min"] / CFG["lr"]

    def factor(epoch):
        if epoch < warmup:
            return float(epoch + 1) / float(warmup)
        denominator = max(1, CFG["epochs"] - warmup - 1)
        progress = min(1.0, (epoch - warmup) / denominator)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return eta_ratio + (1.0 - eta_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def _evaluate_outer_test(model, loader, prior_shift, ssw_idx, season_of_step,
                         fold_name, stem, device, amp_enabled):
    """Evaluate once after fixed training; do not select any setting here."""
    model.eval()
    logits, targets, starts, winters = [], [], [], []
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
                raw_logits, vortex, change = model(
                    inputs, qbo, ep_input, return_aux=True)
            logits.append(raw_logits.float().cpu())
            targets.append(batch["y"])
            starts.append(batch["start"])
            winters.append(batch["winter"])
            vortex_predictions.append(vortex.float().cpu())
            vortex_targets.append(batch["vortex_target"])
            change_predictions.append(change.float().cpu())
            change_targets.append(batch["vortex_change_target"])

    probability = torch.sigmoid(torch.cat(logits) + prior_shift.cpu()).numpy()
    target_class = torch.cat(targets).numpy()
    starts_array = torch.cat(starts).numpy()
    winters_array = torch.cat(winters).numpy()
    vortex_prediction = torch.cat(vortex_predictions).numpy()
    vortex_target = torch.cat(vortex_targets).numpy()
    change_prediction = torch.cat(change_predictions).numpy()
    change_target = torch.cat(change_targets).numpy()

    metrics = pilot.base.horizon_probability_metrics(probability, target_class)
    for class_id, horizon in enumerate(CFG["lead_times"]):
        entry = metrics[str(horizon)]
        # A CSI scan on the outer test is explicitly prohibited.
        entry.pop("best_threshold_metrics", None)
        binary_target = (target_class <= class_id).astype(np.int64)
        predicted = probability[:, class_id] >= FIXED_ALARM_THRESHOLD
        decision = pilot.base.probability_metrics(
            predicted.astype(np.float32), binary_target, 0.5)
        decision.pop("threshold", None)
        thresholds = {
            str(int(winter)): FIXED_ALARM_THRESHOLD
            for winter in np.unique(winters_array)
        }
        entry["fixed_threshold_metrics"] = decision
        event_metrics = pilot.base.event_diagnostics(
            predicted, binary_target, starts_array, winters_array,
            ssw_idx, season_of_step, horizon, thresholds)
        event_metrics["note"] = (
            "Fixed pre-registered 0.20 decision threshold; no outer-test "
            "threshold, calibration, or checkpoint selection.")
        entry["event_metrics"] = event_metrics

    prediction_path = os.path.join(
        CFG["out_dir"], "%s_fixed_epoch%d_predictions.npz" % (
            stem, FIXED_TRAIN_EPOCHS))
    np.savez_compressed(
        prediction_path,
        horizon_probability=probability,
        target_class=target_class,
        start=starts_array,
        forecast_origin=starts_array + CFG["input_steps"],
        winter=winters_array,
        vortex_prediction=vortex_prediction,
        vortex_target=vortex_target,
        vortex_change_prediction=change_prediction,
        vortex_change_target=change_target,
        horizons_days=np.asarray(CFG["lead_times"]),
        fixed_alarm_threshold=np.asarray([FIXED_ALARM_THRESHOLD]),
    )
    return {
        "fold": fold_name,
        "prediction_file": prediction_path,
        "n_test_samples": int(len(target_class)),
        "horizons": metrics,
        "vortex_aux_rmse": float(np.sqrt(np.mean(
            (vortex_prediction - vortex_target) ** 2))),
        "vortex_aux_correlation": float(np.corrcoef(
            vortex_prediction, vortex_target)[0, 1]),
        "vortex_change_aux_rmse": float(np.sqrt(np.mean(
            (change_prediction - change_target) ** 2))),
        "vortex_change_aux_correlation": float(np.corrcoef(
            change_prediction, change_target)[0, 1]),
    }


def run_fold(fold_name, seed):
    fold = configure_fold(fold_name)
    pilot.verify_core_source()
    pilot.base.set_seed(seed)
    os.makedirs(CFG["out_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("v50 external benchmark must run on H100")
    stem = "%s_seed%d" % (CFG["version"], seed)
    print("=" * 60, flush=True)
    print("V50 1deg fixed-epoch rolling fold %s, seed %d" % (
        fold_name, seed), flush=True)
    print(json.dumps({"fold": fold, "protocol": PROTOCOL}, indent=2), flush=True)

    all_files, all_steps, all_cum, winter_mask = pilot._build_all_file_list()
    data, qbo, steps_pf, files, norm_stats = pilot.preload_uvt()
    cumulative = np.cumsum(np.concatenate([[0], steps_pf]))
    ep, ep_steps, ep_cumulative, ep_mean, ep_std = pilot.preload_ep_aligned(
        steps_pf, files)
    norm_stats["ep_mean"] = ep_mean
    norm_stats["ep_std"] = ep_std
    np.savez(os.path.join(
        CFG["out_dir"], "norm_stats_%s_seed%d.npz" % (CFG["version"], seed)),
        **norm_stats)
    ssw_idx = _labels_in_winter_space(all_cum, all_steps, winter_mask)
    season_of_step, year_of_step, _ = pilot.base.assign_step_seasons(
        steps_pf, files)

    train_dataset = pilot.base.SSWDatasetV43(
        data, qbo, steps_pf, files, ep, ep_steps, ep_cumulative,
        ssw_idx, season_of_step, year_of_step, cumulative,
        split="train", train_end_year=fold["train_end_year"],
        val_end_year=fold["test_end_year"])
    train_index = PrimaryEventIndex(train_dataset)
    train_loader = DataLoader(
        train_index, CFG["batch_size"], shuffle=True,
        num_workers=CFG["num_workers"], drop_last=True)

    model = pilot.base.SSW3DNetV43().to(device)
    ema_model = copy.deepcopy(model).to(device).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = _scheduler(optimizer)
    train_prior, sampled_prior, shift, shift_tensor = _prior_shift(
        train_dataset, device)
    amp_enabled = CFG["use_bf16"] and torch.cuda.is_bf16_supported()
    horizon_weights = torch.tensor(
        CFG["horizon_loss_weights"], dtype=torch.float32, device=device)
    print("GPU: %s; train events=%d; fixed epochs=%d" % (
        torch.cuda.get_device_name(0), len(train_index.onsets),
        FIXED_TRAIN_EPOCHS), flush=True)
    print("Train prior=%s sampled prior=%s" % (
        np.round(train_prior, 6).tolist(),
        np.round(sampled_prior, 6).tolist()), flush=True)

    log_path = os.path.join(CFG["out_dir"], "train_%s_log.txt" % stem)
    checkpoint_path = os.path.join(
        CFG["out_dir"], "ssw_%s_fixed_epoch%d.pt" % (
            stem, FIXED_TRAIN_EPOCHS))
    started = time.time()
    with open(log_path, "w") as log_file:
        for epoch in range(FIXED_TRAIN_EPOCHS):
            model.train()
            train_loss = 0.0
            for batch_index, batch in enumerate(train_loader):
                inputs = batch["x"].to(device, non_blocking=True)
                qbo_input = batch["q"].to(device, non_blocking=True)
                ep_input = (
                    batch["ep_in"].to(device, non_blocking=True)
                    if CFG["ep_input"] else None)
                target_class = batch["y"].to(device, non_blocking=True)
                targets = torch.stack([
                    (target_class <= class_id).float()
                    for class_id in range(len(CFG["lead_times"]))
                ], dim=1)
                optimizer.zero_grad()
                with torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16,
                        enabled=amp_enabled):
                    logits, vortex, change = model(
                        inputs, qbo_input, ep_input, return_aux=True)
                    loss_by_horizon = F.binary_cross_entropy_with_logits(
                        logits, targets, reduction="none")
                    forecast_loss = (
                        loss_by_horizon * horizon_weights).sum(dim=1).mean()
                    forecast_loss /= horizon_weights.sum()
                    corrected_probability = torch.sigmoid(
                        logits.float() + shift_tensor)
                    monotonic_loss = F.relu(
                        corrected_probability[:, :-1] -
                        corrected_probability[:, 1:]).mean()
                    vortex_loss = F.smooth_l1_loss(
                        vortex.float(), batch["vortex_target"].to(device))
                    change_loss = F.smooth_l1_loss(
                        change.float(),
                        batch["vortex_change_target"].to(device))
                    loss = (
                        forecast_loss +
                        CFG["monotonic_loss_weight"] * monotonic_loss +
                        CFG["vortex_level_aux_loss_weight"] * vortex_loss +
                        CFG["vortex_change_aux_loss_weight"] * change_loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), CFG["grad_clip_norm"])
                optimizer.step()
                pilot.base._update_ema(ema_model, model)
                train_loss += loss.item()
                if batch_index > 0 and batch_index % 50 == 0:
                    print("  epoch %d batch %d/%d" % (
                        epoch, batch_index, len(train_loader)), flush=True)
            line = "E%3d lr=%.1e train=%.4f [%dm]" % (
                epoch, optimizer.param_groups[0]["lr"],
                train_loss / len(train_loader),
                int((time.time() - started) / 60))
            print(line, flush=True)
            log_file.write(line + "\n")
            log_file.flush()
            scheduler.step()

        # The outer test data is instantiated only after all optimisation ends.
        state = {
            "model_state": ema_model.state_dict(),
            "epoch": FIXED_TRAIN_EPOCHS - 1,
            "cfg": {k: v for k, v in CFG.items()
                    if isinstance(v, (int, float, str, bool, list, tuple))},
            "protocol": PROTOCOL,
            "fold": fold,
            "train_horizon_prior": train_prior.tolist(),
            "sampled_horizon_prior": sampled_prior.tolist(),
            "prior_logit_shift": shift.tolist(),
        }
        torch.save(state, checkpoint_path)

    test_dataset = pilot.base.SSWDatasetV43(
        data, qbo, steps_pf, files, ep, ep_steps, ep_cumulative,
        ssw_idx, season_of_step, year_of_step, cumulative,
        split="val", train_end_year=fold["train_end_year"],
        val_end_year=fold["test_end_year"])
    test_loader = DataLoader(
        test_dataset, CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"])
    outer_result = _evaluate_outer_test(
        ema_model, test_loader, shift_tensor.detach().cpu(), ssw_idx,
        season_of_step, fold_name, stem, device, amp_enabled)
    result = {
        "protocol": PROTOCOL,
        "fold": fold,
        "seed": int(seed),
        "checkpoint": checkpoint_path,
        "outer_test": outer_result,
    }
    result_path = os.path.join(
        CFG["out_dir"], "%s_seed%d_results.json" % (CFG["version"], seed))
    with open(result_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print("OUTER_TEST_DONE %s" % result_path, flush=True)


def run_self_test():
    pilot.run_self_test()
    for fold_name, fold in FOLDS.items():
        configure_fold(fold_name)
        assert fold["train_end_year"] < fold["test_end_year"]
        assert CFG["epochs"] == 45
        assert FIXED_TRAIN_EPOCHS == 10
    print("V50_SELF_TEST_OK", flush=True)


def run_data_check():
    pilot.verify_core_source()
    files, steps, cumulative, winter_mask = pilot._build_all_file_list()
    labels = _labels_in_winter_space(cumulative, steps, winter_mask)
    season, _, _ = pilot.base.assign_step_seasons(steps, files)
    report = {"protocol": PROTOCOL, "folds": {}}
    for fold_name, fold in FOLDS.items():
        test_events = season[labels][
            (season[labels] > fold["train_end_year"]) &
            (season[labels] <= fold["test_end_year"])]
        if len(test_events) != fold["expected_test_events"]:
            raise ValueError("Fold %s event count %d != %d" % (
                fold_name, len(test_events), fold["expected_test_events"]))
        report["folds"][fold_name] = {
            "train_end_year": fold["train_end_year"],
            "test_years": [fold["train_end_year"] + 1,
                           fold["test_end_year"]],
            "test_events": int(len(test_events)),
            "test_event_winters": test_events.tolist(),
        }
    report_path = Path(CFG["out_dir"]) / "v50_1deg_fixed10_data_check.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print("V50_DATA_CHECK_OK", flush=True)


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        run_self_test()
    elif "--data-check" in sys.argv:
        run_data_check()
    else:
        if "--fold" not in sys.argv:
            raise ValueError("--fold A|B|C is required")
        fold_position = sys.argv.index("--fold") + 1
        if fold_position >= len(sys.argv):
            raise ValueError("--fold requires A, B, or C")
        seed = CFG["seed"]
        if "--seed" in sys.argv:
            seed_position = sys.argv.index("--seed") + 1
            if seed_position >= len(sys.argv):
                raise ValueError("--seed requires an integer")
            seed = int(sys.argv[seed_position])
        run_fold(sys.argv[fold_position], seed)
