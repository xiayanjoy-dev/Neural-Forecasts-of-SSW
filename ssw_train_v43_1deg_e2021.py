"""Minimal v43 control on the audited unified 1-degree ERA5 corpus.

This is a data/architecture pilot only.  It preserves v43 supervision,
sampling, losses, auxiliary targets, optimizer, and 45-day input.  The only
scientific input changes are the audited native 1-degree UVT source, fixed
5-degree zonal mean pooling, and corrected instantaneous QG EP divergence.
Validation ends in winter 2021 because the new corpus ends in April 2021; this
run must not be presented as a replacement for the 2018--2025 v43 benchmark.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from netCDF4 import Dataset as NcFile
import torch

import ssw_train_v43_core_versioned as base


EXPECTED_CORE_SHA256 = (
    "7e6f75f8846b2b8e4c2ac9d14ca729b1b9600c6975c97026e4b35fc46abcc2d7"
)
CFG = base.CFG
CFG.update({
    "version": "v43_1deg_e2021",
    "data_dir": "/share/home/xiayan/ssw_phase1/era5_1deg_full/uvt",
    "ep_dir": "/share/home/xiayan/ssw_phase1/era5_1deg_full/ep_qg",
    "label_json": "/share/home/xiayan/ssw_phase1/ssw_labels_final.json",
    "audit_json": (
        "/share/home/xiayan/ssw_phase1/era5_1deg_full/"
        "audit_era5_1deg_full.json"
    ),
    "source_start_winter": 1979,
    "source_end_winter": 2021,
    "val_end_year": 2021,
    "longitude_pool_degrees": 5,
    "native_latitude_degrees": 1,
    "compressed_n_lat": 71,
    "compressed_n_lon": 72,
    "idx_60n_downsampled": 30,
    "polar_cap_last_index": 30,
    # Preserve v43's inclusive 75N--45N eddy heat-flux band.
    "heat_flux_lat_slice": [15, 46],
    "n_ep_lat": 36,
    "n_ep_lev": 11,
    "pilot_scope": "ending-2021 data/architecture pilot",
})


def verify_core_source():
    digest = hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()
    if digest != EXPECTED_CORE_SHA256:
        raise RuntimeError(
            "Versioned v43 core changed: %s != %s" %
            (digest, EXPECTED_CORE_SHA256)
        )


def _selected_uvt_files():
    files = sorted(Path(CFG["data_dir"]).glob("era5_uvt_1deg_*.nc"))
    selected = []
    for path in files:
        winter_year = base._winter_year_from_file(str(path))
        if CFG["source_start_winter"] <= winter_year <= CFG["source_end_winter"]:
            selected.append(str(path))
    return selected


def _build_all_file_list():
    files = _selected_uvt_files()
    steps, valid = [], []
    for path in files:
        try:
            with NcFile(path, "r") as ds:
                steps.append(ds.dimensions["valid_time"].size)
                valid.append(path)
        except Exception:
            print("  SKIP corrupted: %s" % os.path.basename(path), flush=True)
    steps = np.asarray(steps, dtype=np.int32)
    cumulative = np.cumsum(np.concatenate([[0], steps]))
    winter_mask = np.ones(len(valid), dtype=bool)
    return valid, steps, cumulative, winter_mask


def _pool_longitude_5deg(variable, prefix):
    """Average five native 1-degree longitude offsets into 72 fixed bins."""
    pooled = None
    for offset in range(CFG["longitude_pool_degrees"]):
        part = np.asarray(variable[prefix + (slice(offset, None, 5),)],
                          dtype=np.float32)
        if part.shape[-1] != CFG["compressed_n_lon"]:
            raise ValueError("Unexpected pooled longitude shape %s" % (part.shape,))
        if pooled is None:
            pooled = part
        else:
            pooled += part
    pooled /= float(CFG["longitude_pool_degrees"])
    return pooled


def preload_uvt():
    """Load 1-degree UVT, retaining latitude and applying fixed zonal pooling."""
    files = _selected_uvt_files()
    print("Supervised winter files: %d" % len(files), flush=True)
    if not files:
        raise FileNotFoundError("No unified 1-degree UVT files")

    with NcFile(files[0], "r") as ds:
        lat = np.asarray(ds.variables["latitude"][:])
        plev = np.asarray(ds.variables["pressure_level"][:])
    lat_idx = np.where(
        (lat >= CFG["lat_range"][0]) & (lat <= CFG["lat_range"][1])
    )[0]
    if len(lat_idx) != CFG["compressed_n_lat"] or not np.all(np.diff(lat_idx) == 1):
        raise ValueError("Expected 71 contiguous native 1-degree latitudes")
    lat_slice = slice(int(lat_idx[0]), int(lat_idx[-1]) + 1)
    if not np.isclose(lat[CFG["idx_60n_downsampled"]], 60.0):
        raise ValueError("Configured 60N index does not select 60N")

    eq_idx = np.where(np.abs(lat) <= 5)[0]
    if not np.all(np.diff(eq_idx) == 1):
        raise ValueError("Equatorial latitude band is not contiguous")
    eq_slice = slice(int(eq_idx[0]), int(eq_idx[-1]) + 1)
    n_lev = CFG["n_lev"]
    idx_200 = int(np.abs(plev - 200).argmin())
    idx_50 = int(np.abs(plev - 50).argmin())
    CFG["u_dz_lev_hi"] = idx_50
    CFG["u_dz_lev_lo"] = idx_200
    print("  U_dz: U(%d hPa) - U(%d hPa), idx %d - %d" % (
        plev[idx_50], plev[idx_200], idx_50, idx_200), flush=True)

    steps, valid_files = [], []
    for path in files:
        try:
            with NcFile(path, "r") as ds:
                steps.append(ds.dimensions["valid_time"].size)
                valid_files.append(path)
        except Exception:
            print("  SKIP corrupted: %s" % os.path.basename(path), flush=True)
    files = valid_files
    steps_pf = np.asarray(steps, dtype=np.int32)
    total = int(steps_pf.sum())
    data = np.empty((
        total, CFG["n_input_channels"], n_lev,
        CFG["compressed_n_lat"], CFG["compressed_n_lon"]
    ), dtype=np.float32)
    qbo_u = np.empty((total, n_lev), dtype=np.float32)

    started, offset = time.time(), 0
    for file_index, path in enumerate(files):
        with NcFile(path, "r") as ds:
            n_steps = int(steps_pf[file_index])
            target = slice(offset, offset + n_steps)
            prefix = (slice(None), slice(0, n_lev), lat_slice)
            data[target, 0] = _pool_longitude_5deg(ds.variables["t"], prefix)
            u_3d = _pool_longitude_5deg(ds.variables["u"], prefix)
            data[target, 1] = u_3d
            data[target, 2] = _pool_longitude_5deg(ds.variables["v"], prefix)
            u_dz = u_3d[:, idx_50] - u_3d[:, idx_200]
            data[target, 3] = u_dz[:, None, :, :]
            u_equatorial = np.asarray(
                ds.variables["u"][:, :n_lev, eq_slice, :], dtype=np.float32)
            qbo_u[target] = u_equatorial.mean(axis=(2, 3))
        offset += n_steps
        if (file_index + 1) % 25 == 0:
            print("  %d/%d (%d steps, %ds)" % (
                file_index + 1, len(files), offset,
                time.time() - started), flush=True)

    train_stop = base._training_stop_step(
        steps_pf, files, CFG["train_end_year"])
    print("  Normalization fit: first %d/%d winter steps (winter <= %d)" % (
        train_stop, total, CFG["train_end_year"]), flush=True)
    norm_stats = {}
    names = ["t", "u", "v", "u_dz"]
    for channel, name in enumerate(names):
        values = data[:, channel]
        missing = int(np.isnan(values).sum())
        mean = float(np.nanmean(values[:train_stop]))
        std = float(np.nanstd(values[:train_stop]))
        if not np.isfinite(std) or std <= 0:
            raise ValueError("Invalid %s standard deviation: %s" % (name, std))
        norm_stats[name + "_mean"] = mean
        norm_stats[name + "_std"] = std
        print("  %s norm: mean=%.4f std=%.4f missing=%d" % (
            name, mean, std, missing), flush=True)
    qbo_mean = np.nanmean(qbo_u[:train_stop], axis=0)
    qbo_std = np.nanstd(qbo_u[:train_stop], axis=0)
    norm_stats["qbo_mean"] = qbo_mean.tolist()
    norm_stats["qbo_std"] = qbo_std.tolist()

    for channel, name in enumerate(names):
        data[:, channel] = (
            data[:, channel] - norm_stats[name + "_mean"]
        ) / norm_stats[name + "_std"]
    qbo_u[:] = (qbo_u - qbo_mean) / (qbo_std + 1e-8)
    np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(qbo_u, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return data, qbo_u, steps_pf, files, norm_stats


def preload_ep_aligned(steps_pf, files):
    """Load corrected instantaneous QG EP divergence and align by month."""
    ep_files = sorted(Path(CFG["ep_dir"]).glob("era5_ep_qg_1deg_*.nc"))
    ep_by_month = {
        "%04d%02d" % base._year_month_from_file(str(path)): str(path)
        for path in ep_files
    }
    if not ep_by_month:
        raise FileNotFoundError("No corrected 1-degree QG EP files")
    with NcFile(next(iter(ep_by_month.values())), "r") as ds:
        ep_lat = np.asarray(ds.variables["lat"][:])
        ep_level = np.asarray(ds.variables["level"][:])
    ep_lat_idx = np.where(
        (ep_lat >= CFG["ep_lat_range"][0]) &
        (ep_lat <= CFG["ep_lat_range"][1])
    )[0]
    ep_lev_idx = np.where(
        (ep_level >= CFG["ep_lev_range"][1]) &
        (ep_level <= CFG["ep_lev_range"][0])
    )[0]
    CFG["n_ep_lat"] = len(ep_lat_idx)
    CFG["n_ep_lev"] = len(ep_lev_idx)
    if (CFG["n_ep_lat"], CFG["n_ep_lev"]) != (36, 11):
        raise ValueError("Unexpected corrected EP grid")

    total = int(np.sum(steps_pf))
    ep_data = np.full((total, CFG["n_ep_lat"], CFG["n_ep_lev"]),
                      np.nan, dtype=np.float32)
    offset, started = 0, time.time()
    for file_index, path in enumerate(files):
        year, month = base._year_month_from_file(path)
        yyyymm = "%04d%02d" % (year, month)
        n_steps = int(steps_pf[file_index])
        if yyyymm not in ep_by_month:
            raise FileNotFoundError("Missing corrected EP month %s" % yyyymm)
        with NcFile(ep_by_month[yyyymm], "r") as ds:
            epdiv = np.asarray(ds.variables["epdiv"][:], dtype=np.float32)
        if epdiv.shape[0] != n_steps:
            raise ValueError("EP/UVT time mismatch for %s" % yyyymm)
        ep_data[offset:offset + n_steps] = (
            epdiv[:, ep_lev_idx, :][:, :, ep_lat_idx].transpose(0, 2, 1)
        )
        offset += n_steps
        if (file_index + 1) % 50 == 0:
            print("  EP aligned %d/%d (%ds)" % (
                file_index + 1, len(files), time.time() - started), flush=True)

    train_stop = base._training_stop_step(
        steps_pf, files, CFG["train_end_year"])
    ep_mean = np.nanmean(ep_data[:train_stop], axis=(0, 1), keepdims=True)
    ep_std = np.nanstd(ep_data[:train_stop], axis=(0, 1), keepdims=True)
    if not np.all(np.isfinite(ep_std)) or np.any(ep_std <= 0):
        raise ValueError("Invalid corrected EP normalization standard deviation")
    ep_data = (ep_data - ep_mean) / ep_std
    np.nan_to_num(ep_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    cumulative = np.cumsum(np.concatenate([[0], steps_pf]))
    return (ep_data, steps_pf, cumulative,
            ep_mean.reshape(-1).tolist(), ep_std.reshape(-1).tolist())


def load_ssw_labels():
    """Map unchanged v43 onset dates onto the winter-only 1-degree timeline."""
    files, steps, cumulative, _ = _build_all_file_list()
    file_by_month = {
        "%04d%02d" % base._year_month_from_file(path): index
        for index, path in enumerate(files)
    }
    payload = json.loads(Path(CFG["label_json"]).read_text())
    mapped = []
    for event in payload["events"]:
        date = datetime.strptime(event["date"], "%Y-%m-%d")
        yyyymm = date.strftime("%Y%m")
        if yyyymm not in file_by_month:
            continue
        file_index = file_by_month[yyyymm]
        step_in_month = (date.day - 1) * 4
        if step_in_month >= int(steps[file_index]):
            raise ValueError("SSW date outside monthly time axis: %s" % event["date"])
        mapped.append(int(cumulative[file_index] + step_in_month))
    labels = np.asarray(sorted(mapped), dtype=np.int64)
    print("SSW onsets mapped to 1-degree timeline: %d" % len(labels), flush=True)
    return labels


def run_self_test():
    verify_core_source()
    origin = CFG["input_steps"]
    background = CFG["n_time_classes"] - 1
    assert base.time_class_for_start(0, np.array([origin])) == background
    previous = 0
    for class_id, horizon in enumerate(CFG["lead_times"]):
        assert base.time_class_for_start(
            0, np.array([origin + previous * 4 + 1])) == class_id
        assert base.time_class_for_start(
            0, np.array([origin + horizon * 4])) == class_id
        previous = horizon

    onset = np.array([181], dtype=np.int64)
    total = 320
    values = np.arange(total, dtype=np.float32)
    data = np.broadcast_to(
        values[:, None, None, None, None],
        (total, 4, 20, CFG["compressed_n_lat"], 1)
    ).copy()
    qbo = np.zeros((total, CFG["n_lev"]), dtype=np.float32)
    ep = np.zeros((total, 36, 11), dtype=np.float32)
    steps = np.array([total], dtype=np.int32)
    cumulative = np.array([0, total], dtype=np.int64)
    season = np.full(total, 2018, dtype=np.int32)
    year = np.full(total, 2018, dtype=np.int32)
    dataset = base.SSWDatasetV43(
        data, qbo, steps, ["synthetic_201801.nc"],
        ep, steps, cumulative, onset, season, year, cumulative,
        split="val", train_end_year=2017, val_end_year=2021
    )
    item = dataset[dataset.starts.index(0)]
    assert item["x"].shape[0] == CFG["input_steps"]
    assert item["x"][-1, 0, 0, 0, 0].item() == 179.0
    assert item["y"].item() == 0
    assert item["vortex_target"].item() == 182.5
    assert item["vortex_change_target"].item() == 5.0

    model = base.SSW3DNetV43().eval()
    with torch.no_grad():
        output, vortex, change = model(
            torch.zeros(1, 24, 4, 20, 71, 72),
            torch.zeros(1, 24, 20),
            torch.zeros(1, 24, 36, 11),
            return_aux=True,
        )
    assert output.shape == (1, len(CFG["lead_times"]))
    assert vortex.shape == (1,) and change.shape == (1,)
    print("SELF_TEST_OK", flush=True)


def run_data_check():
    verify_core_source()
    audit = json.loads(Path(CFG["audit_json"]).read_text())
    summary = audit.get("summary", audit)
    if not summary.get("pass", False) or summary.get("months_passed") != 372:
        raise ValueError("The 372-month source audit is not passing")

    files, steps, cumulative, winter_mask = _build_all_file_list()
    if len(files) != 258 or not np.all(winter_mask):
        raise ValueError("Expected 258 supervised winter months, got %d" % len(files))
    first_month = "%04d%02d" % base._year_month_from_file(files[0])
    last_month = "%04d%02d" % base._year_month_from_file(files[-1])
    if (first_month, last_month) != ("197811", "202104"):
        raise ValueError("Unexpected supervised corpus bounds")

    ep_files = sorted(Path(CFG["ep_dir"]).glob("era5_ep_qg_1deg_*.nc"))
    ep_by_month = {
        "%04d%02d" % base._year_month_from_file(str(path)): str(path)
        for path in ep_files
    }
    for path, n_steps in zip(files, steps):
        yyyymm = "%04d%02d" % base._year_month_from_file(path)
        if yyyymm not in ep_by_month:
            raise FileNotFoundError("Missing corrected EP file for %s" % yyyymm)
        with NcFile(ep_by_month[yyyymm], "r") as ds:
            if ds.variables["epdiv"].shape[0] != int(n_steps):
                raise ValueError("EP/UVT time mismatch for %s" % yyyymm)

    with NcFile(files[0], "r") as ds:
        lat = np.asarray(ds.variables["latitude"][:])
        lon = np.asarray(ds.variables["longitude"][:])
        lev = np.asarray(ds.variables["pressure_level"][:])
    selected_lat = lat[(lat >= 20) & (lat <= 90)]
    if (len(lat), len(lon), len(lev)) != (96, 360, 20):
        raise ValueError("Unexpected native UVT grid")
    if len(selected_lat) != 71 or len(lon) // 5 != 72:
        raise ValueError("Unexpected deterministic compressed grid")

    labels = load_ssw_labels()
    seasons, _, _ = base.assign_step_seasons(steps, files)
    onset_winters = seasons[labels]
    train_events = int(np.sum(onset_winters <= CFG["train_end_year"]))
    validation_events = int(np.sum(
        (onset_winters > CFG["train_end_year"]) &
        (onset_winters <= CFG["val_end_year"])
    ))
    if (len(labels), train_events, validation_events) != (28, 25, 3):
        raise ValueError("Unexpected label split %s" % (
            (len(labels), train_events, validation_events),
        ))

    report = {
        "status": "DATA_CHECK_OK",
        "pilot_scope": CFG["pilot_scope"],
        "source_audit_months": 372,
        "supervised_months": len(files),
        "supervised_steps": int(cumulative[-1]),
        "winter_bounds": [1979, 2021],
        "native_grid": [20, 96, 360],
        "model_grid": [20, 71, 72],
        "labels": len(labels),
        "train_events": train_events,
        "validation_events": validation_events,
        "validation_winters": [2018, 2019, 2020, 2021],
        "core_sha256": EXPECTED_CORE_SHA256,
    }
    report_path = Path(CFG["out_dir"]) / "v43_1deg_e2021_data_check.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


base._build_all_file_list = _build_all_file_list
base.preload_uvt = preload_uvt
base.preload_ep_aligned = preload_ep_aligned
base.load_ssw_labels = load_ssw_labels


def main(seed):
    verify_core_source()
    base.main(seed)


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        run_self_test()
    elif "--data-check" in sys.argv:
        run_data_check()
    else:
        selected_seed = CFG["seed"]
        if "--seed" in sys.argv:
            selected_seed = int(sys.argv[sys.argv.index("--seed") + 1])
        main(selected_seed)
