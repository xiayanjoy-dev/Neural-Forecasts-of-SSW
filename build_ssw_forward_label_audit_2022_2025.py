#!/usr/bin/env python3
"""Build the frozen 2022--2025 major-SSW label audit from ERA5.

Primary labels use the existing SSWC/Charlton--Polvani rule. Published event
dates are an independent comparison only and never create or remove a label.
"""

import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import re

import numpy as np

from build_ssw_early_label_audit import detect_major_ssw


DATA_START = date(2021, 11, 1)
DATA_END = date(2025, 4, 30)
WINTERS = (2022, 2023, 2024, 2025)
EXPECTED_MONTHS = 24

INDEPENDENT_RECORDS = (
    {
        "date": "2022-03-19",
        "event_type": "final_warming_not_major",
        "source": "Liu et al. (2023), Dynamics of Atmospheres and Oceans",
        "doi": "10.1016/j.dynatmoce.2023.101373",
    },
    {
        "date": "2023-02-16",
        "event_type": "major_ssw",
        "source": "NOAA Sudden Stratospheric Warming Compendium",
        "url": "https://csl.noaa.gov/groups/csl8/sswcompendium/majorevents.html",
    },
    {
        "date": "2024-01-16",
        "event_type": "major_ssw",
        "source": "Lee, Butler and Manney (2024), Weather",
        "doi": "10.1002/wea.7656",
    },
    {
        "date": "2024-03-04",
        "event_type": "major_ssw",
        "source": "Lee, Butler and Manney (2024), Weather",
        "doi": "10.1002/wea.7656",
    },
)


def _winter_year(day):
    return day.year + 1 if day.month >= 11 else day.year


def load_daily_u60_10(data_dir):
    from netCDF4 import Dataset, num2date

    pattern = re.compile(r"era5_uvt_1deg_(\d{6})\.nc$")
    files = []
    for path in sorted(Path(data_dir).glob("era5_uvt_1deg_*.nc")):
        match = pattern.match(path.name)
        if match is None:
            continue
        year_month = match.group(1)
        month_start = date(int(year_month[:4]), int(year_month[4:]), 1)
        if DATA_START <= month_start <= DATA_END:
            files.append(path)
    if len(files) != EXPECTED_MONTHS:
        raise ValueError("Expected %d UVT months, found %d" % (
            EXPECTED_MONTHS, len(files)))

    by_day = defaultdict(list)
    file_rows = []
    for path in files:
        with Dataset(path) as dataset:
            levels = np.asarray(dataset.variables["pressure_level"][:], dtype=float)
            latitudes = np.asarray(dataset.variables["latitude"][:], dtype=float)
            longitudes = np.asarray(dataset.variables["longitude"][:], dtype=float)
            level_index = int(np.abs(levels - 10.0).argmin())
            latitude_index = int(np.abs(latitudes - 60.0).argmin())
            if not np.isclose(levels[level_index], 10.0):
                raise ValueError("No exact 10 hPa in %s" % path)
            if not np.isclose(latitudes[latitude_index], 60.0):
                raise ValueError("No exact 60N in %s" % path)
            if len(longitudes) != 360:
                raise ValueError("Expected 360 longitudes in %s" % path)
            time_var = dataset.variables["valid_time"]
            raw_times = num2date(
                time_var[:], time_var.units,
                calendar=getattr(time_var, "calendar", "standard"),
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=False,
            )
            wind = np.ma.filled(
                dataset.variables["u"][:, level_index, latitude_index, :],
                np.nan,
            )
            zonal_mean = np.nanmean(np.asarray(wind, dtype=float), axis=1)
            if not np.all(np.isfinite(zonal_mean)):
                raise ValueError("Non-finite U60/10 in %s" % path)
            for timestamp, value in zip(raw_times, zonal_mean):
                day = date(timestamp.year, timestamp.month, timestamp.day)
                by_day[day].append(float(value))
            file_rows.append({"file": path.name, "time_steps": len(raw_times)})

    dates = []
    cursor = DATA_START
    while cursor <= DATA_END:
        if cursor.month in {11, 12, 1, 2, 3, 4}:
            dates.append(cursor)
        cursor += timedelta(days=1)
    failures = [day.isoformat() for day in dates if len(by_day[day]) != 4]
    if failures:
        raise ValueError("Expected four samples per day: %s" % failures[:10])
    daily_wind = np.asarray([np.mean(by_day[day]) for day in dates])
    return dates, daily_wind, file_rows


def compare_records(events):
    event_days = [datetime.strptime(row["date"], "%Y-%m-%d").date()
                  for row in events]
    rows = []
    for record in INDEPENDENT_RECORDS:
        expected = datetime.strptime(record["date"], "%Y-%m-%d").date()
        differences = [abs((candidate - expected).days) for candidate in event_days]
        nearest = int(np.argmin(differences)) if differences else None
        row = dict(record)
        row["nearest_era5_rule_date"] = (
            event_days[nearest].isoformat() if nearest is not None else None)
        row["difference_days"] = (
            differences[nearest] if nearest is not None else None)
        row["within_plus_minus_1_day"] = bool(
            nearest is not None and differences[nearest] <= 1)
        if record["event_type"] != "major_ssw":
            row["within_plus_minus_1_day"] = False
        rows.append(row)
    return rows


def build_report(data_dir, source_audit):
    audit = json.loads(Path(source_audit).read_text())
    if audit.get("status") != "PASS" or audit.get("completed_months") != 24:
        raise ValueError("The 24-month source audit is not passing")
    dates, winds, files = load_daily_u60_10(data_dir)
    events = detect_major_ssw(dates, winds)
    if any(event["winter"] not in WINTERS for event in events):
        raise ValueError("Detected event outside frozen forward winters")
    comparisons = compare_records(events)
    major_records = [row for row in comparisons if row["event_type"] == "major_ssw"]
    return {
        "status": "FORWARD_SSW_LABEL_AUDIT_OK",
        "protocol": {
            "definition": "daily zonal-mean U(60N,10hPa) westerly-to-easterly reversal",
            "onset_months": ["November", "December", "January", "February", "March"],
            "inter_event_westerly_days": 20,
            "pre_30_april_westerly_days": 10,
            "daily_samples": "mean of 00/06/12/18 UTC",
            "primary_label_source": "ERA5 rule only",
            "published_records_role": "independent verification only",
        },
        "input": {
            "data_dir": str(Path(data_dir).resolve()),
            "source_audit": str(Path(source_audit).resolve()),
            "first_day": dates[0].isoformat(),
            "last_day": dates[-1].isoformat(),
            "months": len(files),
            "days": len(dates),
            "u60_10_daily_min": float(np.min(winds)),
            "u60_10_daily_max": float(np.max(winds)),
            "files": files,
        },
        "events": events,
        "independent_record_comparison": comparisons,
        "summary": {
            "forward_winters": list(WINTERS),
            "major_ssw_labels": len(events),
            "published_major_records": len(major_records),
            "published_major_matches_within_1_day": int(sum(
                row["within_plus_minus_1_day"] for row in major_records)),
        },
        "claim_boundary": (
            "These labels define the untouched 2022--2025 forward test. "
            "They must not be changed after any model prediction is viewed."
        ),
    }


def self_test():
    synthetic = [{"date": "2023-02-16", "winter": 2023}]
    rows = compare_records(synthetic)
    assert rows[1]["within_plus_minus_1_day"]
    assert not rows[0]["within_plus_minus_1_day"]
    print("FORWARD_LABEL_SELF_TEST_OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument("--source-audit")
    parser.add_argument("--out")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.data_dir, args.source_audit, args.out)):
        parser.error("--data-dir, --source-audit and --out are required")
    report = build_report(args.data_dir, args.source_audit)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
