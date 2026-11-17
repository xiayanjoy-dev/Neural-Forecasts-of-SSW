#!/usr/bin/env python
"""Audit the complete native 1-degree ERA5 winter corpus for SSW work."""

import argparse
import json
import os
from calendar import monthrange
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np


PRESSURE_LEVELS = np.array([
    1000, 925, 850, 700, 500, 400, 300, 200, 150, 100,
    70, 50, 30, 20, 10, 7, 5, 3, 2, 1,
], dtype=np.float64)
LATITUDE = np.arange(90.0, -6.0, -1.0)
LONGITUDE = np.arange(360.0)
EP_LEVELS = PRESSURE_LEVELS[
    (PRESSURE_LEVELS >= 10.0) & (PRESSURE_LEVELS <= 500.0)
]
EP_LATITUDE = LATITUDE[(LATITUDE >= 40.0) & (LATITUDE <= 75.0)]
UVT_VARIABLES = ("u", "v", "t")
EP_VARIABLES = ("f_phi", "f_p", "epdiv", "ep_accel")


def expected_months():
    result = []
    for year in range(1959, 2022):
        for month in (1, 2, 3, 4, 11, 12):
            if (year, month) < (1959, 11) or (year, month) > (2021, 4):
                continue
            result.append(f"{year:04d}{month:02d}")
    return result


def expected_datetimes(year, month):
    return [
        datetime(year, month, day, hour)
        for day in range(1, monthrange(year, month)[1] + 1)
        for hour in (0, 6, 12, 18)
    ]


def variable_stats(variable, block):
    missing = 0
    total = 0
    minimum = np.inf
    maximum = -np.inf
    for start in range(0, variable.shape[0], block):
        values = np.ma.asarray(variable[start:start + block])
        data = np.asarray(np.ma.getdata(values), dtype=np.float64)
        invalid = np.ma.getmaskarray(values) | ~np.isfinite(data)
        missing += int(invalid.sum())
        total += int(data.size)
        valid = data[~invalid]
        if valid.size:
            minimum = min(minimum, float(valid.min()))
            maximum = max(maximum, float(valid.max()))
    return {
        "count": total,
        "nonfinite_or_masked": missing,
        "min": None if not np.isfinite(minimum) else minimum,
        "max": None if not np.isfinite(maximum) else maximum,
    }


def check_time(variable, expected):
    from netCDF4 import date2num

    calendar = getattr(variable, "calendar", "standard")
    wanted = np.asarray(date2num(expected, variable.units, calendar=calendar))
    actual = np.asarray(variable[:], dtype=np.float64)
    return bool(
        actual.shape == wanted.shape
        and np.allclose(actual, wanted, rtol=0.0, atol=1e-7)
    )


def audit_month(base_dir, yyyymm):
    from netCDF4 import Dataset

    base = Path(base_dir)
    year, month = int(yyyymm[:4]), int(yyyymm[4:])
    expected_time = expected_datetimes(year, month)
    paths = {
        "uvt": base / "uvt" / f"era5_uvt_1deg_{yyyymm}.nc",
        "manifest": base / "manifest" / f"era5_uvt_1deg_{yyyymm}.json",
        "qg_ep": base / "ep_qg" / f"era5_ep_qg_1deg_{yyyymm}.nc",
    }
    result = {
        "pass": False,
        "paths": {name: str(path) for name, path in paths.items()},
        "errors": [],
    }
    for name, path in paths.items():
        if not path.is_file():
            result["errors"].append(f"missing_{name}")
    if result["errors"]:
        return yyyymm, result

    try:
        manifest = json.loads(paths["manifest"].read_text())
        expected_manifest = {
            "year": year,
            "month": month,
            "output": str(paths["uvt"]),
            "dimensions": {
                "valid_time": len(expected_time),
                "pressure_level": 20,
                "latitude": 96,
                "longitude": 360,
            },
            "time_start": expected_time[0].isoformat(),
            "time_end": expected_time[-1].isoformat(),
            "pressure_levels_hpa": PRESSURE_LEVELS.astype(int).tolist(),
            "latitude_range": [90.0, -5.0],
            "longitude_range": [0.0, 359.0],
        }
        manifest_checks = {
            key: manifest.get(key) == value
            for key, value in expected_manifest.items()
        }
        manifest_checks["variables"] = set(manifest.get("variables", {})) == {
            "u", "v", "t"
        }
        manifest_checks["source"] = str(manifest.get("source", "")).startswith(
            "gs://gcp-public-data-arco-era5/"
        )
        if not all(manifest_checks.values()):
            result["errors"].append("manifest_mismatch")
        result["manifest_checks"] = manifest_checks
    except Exception as exc:
        result["errors"].append(f"manifest_error:{type(exc).__name__}:{exc}")

    try:
        with Dataset(paths["uvt"]) as dataset:
            dimensions = {
                name: len(value) for name, value in dataset.dimensions.items()
            }
            coordinate_checks = {
                "time": check_time(dataset.variables["valid_time"], expected_time),
                "pressure_level": np.array_equal(
                    np.asarray(dataset.variables["pressure_level"][:]),
                    PRESSURE_LEVELS,
                ),
                "latitude": np.allclose(
                    np.asarray(dataset.variables["latitude"][:]), LATITUDE,
                    rtol=0.0, atol=1e-6,
                ),
                "longitude": np.allclose(
                    np.asarray(dataset.variables["longitude"][:]), LONGITUDE,
                    rtol=0.0, atol=1e-6,
                ),
            }
            expected_dimensions = {
                "valid_time": len(expected_time),
                "pressure_level": 20,
                "latitude": 96,
                "longitude": 360,
            }
            if dimensions != expected_dimensions:
                result["errors"].append("uvt_dimension_mismatch")
            if not all(coordinate_checks.values()):
                result["errors"].append("uvt_coordinate_or_time_mismatch")
            if set(dataset.variables) != {
                    "valid_time", "pressure_level", "latitude", "longitude",
                    "u", "v", "t"}:
                result["errors"].append("uvt_variable_inventory_mismatch")
            variables = {
                name: variable_stats(dataset.variables[name], block=8)
                for name in UVT_VARIABLES
            }
            if any(value["nonfinite_or_masked"] for value in variables.values()):
                result["errors"].append("uvt_nonfinite_or_masked")
            result["uvt"] = {
                "bytes": paths["uvt"].stat().st_size,
                "dimensions": dimensions,
                "coordinate_checks": coordinate_checks,
                "variables": variables,
            }
    except Exception as exc:
        result["errors"].append(f"uvt_error:{type(exc).__name__}:{exc}")

    try:
        with Dataset(paths["qg_ep"]) as dataset:
            dimensions = {
                name: len(value) for name, value in dataset.dimensions.items()
            }
            coordinate_checks = {
                "time": check_time(dataset.variables["time"], expected_time),
                "level": np.array_equal(
                    np.asarray(dataset.variables["level"][:]), EP_LEVELS
                ),
                "latitude": np.allclose(
                    np.asarray(dataset.variables["lat"][:]), EP_LATITUDE,
                    rtol=0.0, atol=1e-6,
                ),
            }
            expected_dimensions = {
                "time": len(expected_time),
                "level": len(EP_LEVELS),
                "lat": len(EP_LATITUDE),
            }
            if dimensions != expected_dimensions:
                result["errors"].append("qg_ep_dimension_mismatch")
            if not all(coordinate_checks.values()):
                result["errors"].append("qg_ep_coordinate_or_time_mismatch")
            if not set(EP_VARIABLES).issubset(dataset.variables):
                result["errors"].append("qg_ep_variable_inventory_mismatch")
            variables = {
                name: variable_stats(dataset.variables[name], block=32)
                for name in EP_VARIABLES
            }
            if any(value["nonfinite_or_masked"] for value in variables.values()):
                result["errors"].append("qg_ep_nonfinite_or_masked")
            recorded_source = Path(getattr(dataset, "source_file", ""))
            source_file_matches = bool(
                recorded_source == paths["uvt"]
                or (
                    recorded_source.is_file()
                    and os.path.samefile(recorded_source, paths["uvt"])
                )
            )
            result["qg_ep"] = {
                "bytes": paths["qg_ep"].stat().st_size,
                "dimensions": dimensions,
                "coordinate_checks": coordinate_checks,
                "variables": variables,
                "recorded_source_file": str(recorded_source),
                "source_file_matches": source_file_matches,
            }
            if not result["qg_ep"]["source_file_matches"]:
                result["errors"].append("qg_ep_source_file_mismatch")
    except Exception as exc:
        result["errors"].append(f"qg_ep_error:{type(exc).__name__}:{exc}")

    result["pass"] = not result["errors"]
    return yyyymm, result


def inventory(base_dir, months):
    base = Path(base_dir)
    expected = {
        "uvt": {f"era5_uvt_1deg_{month}.nc" for month in months},
        "manifest": {f"era5_uvt_1deg_{month}.json" for month in months},
        "ep_qg": {f"era5_ep_qg_1deg_{month}.nc" for month in months},
    }
    patterns = {"uvt": "*.nc", "manifest": "*.json", "ep_qg": "*.nc"}
    result = {}
    for name, wanted in expected.items():
        found = {path.name for path in (base / name).glob(patterns[name])}
        result[name] = {
            "expected": len(wanted),
            "found": len(found & wanted),
            "missing": sorted(wanted - found),
        }
    partials = sorted(str(path) for path in base.rglob("*.partial"))
    partials += sorted(str(path) for path in base.rglob("*.part"))
    partials += sorted(str(path) for path in base.rglob("*.tmp"))
    result["partial_files"] = partials
    result["pass"] = bool(
        all(not value["missing"] for key, value in result.items()
            if key in expected)
        and not partials
    )
    return result


def run_audit(base_dir, output_path, months, workers):
    report = {
        "scope": {
            "base_dir": str(Path(base_dir)),
            "months": months,
            "month_count": len(months),
            "period": [months[0], months[-1]],
        },
        "inventory": inventory(base_dir, months),
        "months": {},
    }
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(audit_month, base_dir, month): month
            for month in months
        }
        for index, future in enumerate(as_completed(futures), start=1):
            month, result = future.result()
            report["months"][month] = result
            if index % 20 == 0 or index == len(futures):
                print(f"Audited {index}/{len(futures)} months", flush=True)
    report["months"] = dict(sorted(report["months"].items()))
    failed = [
        month for month, result in report["months"].items()
        if not result["pass"]
    ]
    report["summary"] = {
        "months_passed": len(months) - len(failed),
        "months_failed": len(failed),
        "failed_months": failed,
        "pass": bool(report["inventory"]["pass"] and not failed),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report["summary"], indent=2), flush=True)
    if not report["summary"]["pass"]:
        raise SystemExit(1)


def self_test():
    months = expected_months()
    assert len(months) == 372
    assert months[0] == "195911"
    assert months[-1] == "202104"
    assert len(expected_datetimes(2000, 2)) == 116
    assert len(expected_datetimes(2001, 2)) == 112
    assert len(LATITUDE) == 96 and len(LONGITUDE) == 360
    assert len(PRESSURE_LEVELS) == 20
    assert len(EP_LEVELS) == 11 and len(EP_LATITUDE) == 36
    print("SELF_TEST_OK", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir")
    parser.add_argument("--output")
    parser.add_argument("--months", help="comma-separated YYYYMM subset")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.base_dir or not args.output:
        parser.error("--base-dir and --output are required")
    months = args.months.split(",") if args.months else expected_months()
    run_audit(args.base_dir, args.output, months, args.workers)


if __name__ == "__main__":
    main()
