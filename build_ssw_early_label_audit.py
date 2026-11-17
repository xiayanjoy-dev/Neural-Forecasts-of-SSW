"""Audit 1959--1978 major-SSW labels against unified 1-degree ERA5.

The primary label rule is the Charlton--Polvani/SSW Compendium rule: daily
zonal-mean U(60N, 10 hPa) first changes from westerly to easterly between
November and March; a later event requires a 20-day westerly recovery, and
the event must be followed by 10 consecutive westerly days before 30 April.

This script treats the NOAA SSW Compendium as an independent event catalogue.
It does not create labels from a catalogue alone: a primary early label needs
an ERA5 reversal at the listed date (within three days) and agreement by at
least two additional reanalyses in the Compendium table.  Catalogue rows that
fail either condition remain explicitly rejected in the audit output.
"""
import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import re

import numpy as np


DATA_START = date(1959, 11, 1)
DATA_END = date(1978, 4, 30)
ALLOWED_ONSET_MONTHS = {11, 12, 1, 2, 3}
CONSENSUS_TOLERANCE_DAYS = 3
REPEAT_RECOVERY_DAYS = 20
FINAL_RECOVERY_DAYS = 10
SSWC_URL = "https://csl.noaa.gov/groups/csl8/sswcompendium/majorevents.html"
SSWC_DOI = "https://doi.org/10.5194/essd-9-63-2017"

# Transcribed from the NOAA SSWC current table of major mid-winter NH events.
# None represents an available reanalysis in which the table reports no event.
# The catalogue also contains 1958 events, excluded here because the unified
# corpus begins in November 1959.
SSWC_EARLY_ROWS = (
    {"name": "JAN 1960", "NCEP-NCAR": "1960-01-16", "ERA-40": "1960-01-17",
     "JRA-55": "1960-01-17", "ERA5": "1960-01-17"},
    {"name": "JAN 1963", "NCEP-NCAR": None, "ERA-40": "1963-01-28",
     "JRA-55": "1963-01-30", "ERA5": "1963-01-27"},
    {"name": "MAR 1965", "NCEP-NCAR": "1965-03-23", "ERA-40": None,
     "JRA-55": None, "ERA5": None},
    {"name": "DEC 1965", "NCEP-NCAR": "1965-12-08", "ERA-40": "1965-12-16",
     "JRA-55": "1965-12-18", "ERA5": "1965-12-16"},
    {"name": "FEB 1966", "NCEP-NCAR": "1966-02-24", "ERA-40": "1966-02-23",
     "JRA-55": "1966-02-23", "ERA5": "1966-02-22"},
    {"name": "JAN 1968", "NCEP-NCAR": None, "ERA-40": "1968-01-07",
     "JRA-55": "1968-01-07", "ERA5": "1968-01-07"},
    {"name": "NOV 1968", "NCEP-NCAR": "1968-11-27", "ERA-40": "1968-11-28",
     "JRA-55": "1968-11-29", "ERA5": "1968-11-28"},
    {"name": "MAR 1969", "NCEP-NCAR": "1969-03-13", "ERA-40": "1969-03-13",
     "JRA-55": None, "ERA5": "1969-03-13"},
    {"name": "JAN 1970", "NCEP-NCAR": "1970-01-02", "ERA-40": "1970-01-02",
     "JRA-55": "1970-01-02", "ERA5": "1970-01-02"},
    {"name": "JAN 1971", "NCEP-NCAR": "1971-01-17", "ERA-40": "1971-01-18",
     "JRA-55": "1971-01-18", "ERA5": "1971-01-18"},
    {"name": "MAR 1971", "NCEP-NCAR": "1971-03-20", "ERA-40": "1971-03-20",
     "JRA-55": "1971-03-20", "ERA5": "1971-03-20"},
    {"name": "JAN 1973", "NCEP-NCAR": "1973-02-02", "ERA-40": "1973-01-31",
     "JRA-55": "1973-01-31", "ERA5": "1973-01-31"},
    {"name": "JAN 1977", "NCEP-NCAR": None, "ERA-40": "1977-01-09",
     "JRA-55": "1977-01-09", "ERA5": "1977-01-09"},
)


def _as_date(text):
    return None if text is None else datetime.strptime(text, "%Y-%m-%d").date()


def _winter_year(day):
    return day.year + 1 if day.month >= 11 else day.year


def _daily_westerly_run(dates, winds, start_index, minimum_days, last_day):
    """Return the first eligible all-westerly run after one candidate onset."""
    last_start = len(dates) - minimum_days
    for i in range(start_index + 1, last_start + 1):
        if dates[i + minimum_days - 1] > last_day:
            break
        if np.all(winds[i:i + minimum_days] > 0.0):
            return i
    return None


def _detect_one_winter(dates, winds):
    """Implement the SSWC/CP07 rule for one contiguous November--April season."""
    dates = np.asarray(dates, dtype=object)
    winds = np.asarray(winds, dtype=np.float64)
    if len(dates) != len(winds) or len(dates) == 0:
        raise ValueError("dates and winds must be non-empty and aligned")
    if not np.all(np.isfinite(winds)):
        raise ValueError("daily wind series contains non-finite values")
    if any((dates[i] - dates[i - 1]).days != 1 for i in range(1, len(dates))):
        raise ValueError("daily wind series must be contiguous")

    accepted = []
    prior_event_index = None
    for i in range(1, len(dates)):
        day = dates[i]
        if day.month not in ALLOWED_ONSET_MONTHS:
            continue
        if not (winds[i - 1] > 0.0 and winds[i] <= 0.0):
            continue
        season_end = date(_winter_year(day), 4, 30)
        final_recovery = _daily_westerly_run(
            dates, winds, i, FINAL_RECOVERY_DAYS, season_end
        )
        if final_recovery is None:
            continue
        if prior_event_index is not None:
            repeat_recovery = _daily_westerly_run(
                dates, winds, prior_event_index, REPEAT_RECOVERY_DAYS, season_end
            )
            if repeat_recovery is None or repeat_recovery >= i:
                continue
        accepted.append({
            "date": day.isoformat(),
            "winter": _winter_year(day),
            "previous_day_u60_10": float(winds[i - 1]),
            "onset_day_u60_10": float(winds[i]),
            "final_recovery_start": dates[final_recovery].isoformat(),
        })
        prior_event_index = i
    return accepted


def detect_major_ssw(dates, winds):
    """Apply the event rule independently to each November--April winter."""
    dates = np.asarray(dates, dtype=object)
    winds = np.asarray(winds, dtype=np.float64)
    if len(dates) != len(winds) or len(dates) == 0:
        raise ValueError("dates and winds must be non-empty and aligned")
    events = []
    for winter in sorted({_winter_year(day) for day in dates}):
        selected = np.asarray([_winter_year(day) == winter for day in dates])
        events.extend(_detect_one_winter(dates[selected], winds[selected]))
    return events


def _source_consensus(row):
    """Return all SSWC detections within three days of its ERA5 date."""
    era5_day = _as_date(row["ERA5"])
    if era5_day is None:
        return [], False
    agreeing = []
    for source in ("NCEP-NCAR", "ERA-40", "JRA-55"):
        candidate = _as_date(row[source])
        if candidate is not None and abs((candidate - era5_day).days) <= CONSENSUS_TOLERANCE_DAYS:
            agreeing.append(source)
    return agreeing, len(agreeing) >= 2


def _find_matching_candidate(candidates, expected_day):
    matches = []
    for candidate in candidates:
        difference = abs((_as_date(candidate["date"]) - expected_day).days)
        if difference <= CONSENSUS_TOLERANCE_DAYS:
            match = dict(candidate)
            match["abs_difference_days"] = difference
            matches.append(match)
    return min(matches, key=lambda item: item["abs_difference_days"], default=None)


def audit_catalogue(candidates):
    """Combine independently published SSWC rows with ERA5-rule candidates."""
    rows = []
    labels = []
    for source_row in SSWC_EARLY_ROWS:
        row = dict(source_row)
        agreeing_sources, catalogue_consensus = _source_consensus(row)
        era5_day = _as_date(row["ERA5"])
        matched = None if era5_day is None else _find_matching_candidate(candidates, era5_day)
        quality = "rejected"
        rejection_reason = None
        if era5_day is None:
            rejection_reason = "SSWC_has_no_ERA5_detection"
        elif not catalogue_consensus:
            rejection_reason = "fewer_than_two_independent_reanalyses_agree_with_ERA5"
        elif matched is None:
            rejection_reason = "ERA5_rule_did_not_reproduce_SSWC_date_within_3_days"
        else:
            quality = "consensus"
            labels.append({
                "date": matched["date"],
                "winter": matched["winter"],
                "source": "SSWC_ERA5_plus_%s" % "+".join(agreeing_sources),
                "quality": quality,
                "sswc_event": row["name"],
            })
        row.update({
            "agreeing_non_ERA5_sources": agreeing_sources,
            "catalogue_consensus": catalogue_consensus,
            "era5_rule_match": matched,
            "quality": quality,
            "rejection_reason": rejection_reason,
        })
        rows.append(row)
    matched_dates = {label["date"] for label in labels}
    unmatched_era5_candidates = [
        candidate for candidate in candidates if candidate["date"] not in matched_dates
    ]
    return rows, labels, unmatched_era5_candidates


def load_era5_daily_u60_10(data_dir):
    """Read daily mean zonal U at exactly 60N and 10 hPa from all early months."""
    from netCDF4 import Dataset, num2date

    data_dir = Path(data_dir)
    pattern = re.compile(r"era5_uvt_1deg_(\d{6})\.nc$")
    files = []
    for path in sorted(data_dir.glob("era5_uvt_1deg_*.nc")):
        match = pattern.match(path.name)
        if match is None:
            continue
        yyyymm = match.group(1)
        month_start = date(int(yyyymm[:4]), int(yyyymm[4:]), 1)
        if DATA_START <= month_start <= DATA_END:
            files.append(path)
    if not files:
        raise FileNotFoundError("No early unified-ERA5 UVT files found")

    by_day = defaultdict(list)
    selected_files = []
    for path in files:
        with Dataset(path) as ds:
            for required in ("valid_time", "pressure_level", "latitude", "longitude", "u"):
                if required not in ds.variables:
                    raise KeyError("%s lacks %s" % (path, required))
            levels = np.asarray(ds.variables["pressure_level"][:], dtype=float)
            latitudes = np.asarray(ds.variables["latitude"][:], dtype=float)
            level_index = int(np.abs(levels - 10.0).argmin())
            latitude_index = int(np.abs(latitudes - 60.0).argmin())
            if not np.isclose(levels[level_index], 10.0) or not np.isclose(latitudes[latitude_index], 60.0):
                raise ValueError("%s does not contain exact 10 hPa / 60N" % path)
            time_var = ds.variables["valid_time"]
            calendar = getattr(time_var, "calendar", "standard")
            raw_times = num2date(time_var[:], time_var.units, calendar=calendar,
                                 only_use_cftime_datetimes=False,
                                 only_use_python_datetimes=False)
            u60 = np.ma.filled(
                ds.variables["u"][:, level_index, latitude_index, :], np.nan
            )
            if u60.shape != (len(raw_times), len(ds.dimensions["longitude"])):
                raise ValueError("Unexpected U slice shape in %s" % path)
            zonal_u60 = np.nanmean(np.asarray(u60, dtype=float), axis=1)
            for timestamp, value in zip(raw_times, zonal_u60):
                day = date(timestamp.year, timestamp.month, timestamp.day)
                if DATA_START <= day <= DATA_END:
                    by_day[day].append(float(value))
        selected_files.append(path.name)

    all_days = []
    cursor = DATA_START
    while cursor <= DATA_END:
        if cursor.month in {11, 12, 1, 2, 3, 4}:
            all_days.append(cursor)
        cursor += timedelta(days=1)
    missing = [day.isoformat() for day in all_days if len(by_day[day]) != 4]
    if missing:
        raise ValueError("Expected four 6-hourly samples per day; first failures %s" % missing[:10])
    daily = np.asarray([np.mean(by_day[day]) for day in all_days], dtype=float)
    return all_days, daily, selected_files


def build_report(data_dir):
    dates, daily_u, files = load_era5_daily_u60_10(data_dir)
    candidates = detect_major_ssw(dates, daily_u)
    rows, labels, unmatched_candidates = audit_catalogue(candidates)
    return {
        "status": "EARLY_SSW_LABEL_AUDIT_OK",
        "protocol": {
            "definition": "SSWC/Charlton-Polvani daily U(60N,10hPa) first westerly-to-easterly reversal",
            "onset_months": ["November", "December", "January", "February", "March"],
            "inter_event_westerly_days": REPEAT_RECOVERY_DAYS,
            "pre_30_april_westerly_days": FINAL_RECOVERY_DAYS,
            "catalogue_agreement_tolerance_days": CONSENSUS_TOLERANCE_DAYS,
            "supervised_label_rule": "ERA5 rule match plus >=2 other SSWC reanalyses within tolerance",
        },
        "sources": {
            "sswc_table": SSWC_URL,
            "sswc_dataset_doi": SSWC_DOI,
            "source_accessed": date.today().isoformat(),
            "early_era5_input": str(Path(data_dir)),
        },
        "input": {
            "first_day": dates[0].isoformat(),
            "last_day": dates[-1].isoformat(),
            "daily_samples": len(dates),
            "monthly_files": len(files),
            "files": files,
            "u60_10_daily_min": float(np.min(daily_u)),
            "u60_10_daily_max": float(np.max(daily_u)),
        },
        "era5_rule_candidates": candidates,
        "sswc_rows": rows,
        "consensus_labels": labels,
        "unmatched_era5_rule_candidates": unmatched_candidates,
        "summary": {
            "sswc_rows_considered": len(rows),
            "era5_rule_candidates": len(candidates),
            "consensus_labels": len(labels),
            "rejected_catalogue_rows": int(sum(row["quality"] != "consensus" for row in rows)),
            "unmatched_era5_rule_candidates": len(unmatched_candidates),
        },
    }


def run_self_test():
    dates = [date(2000, 11, 1) + timedelta(days=offset) for offset in range(181)]
    winds = np.full(len(dates), 15.0)
    first = dates.index(date(2001, 1, 10))
    winds[first:first + 12] = -5.0
    winds[first + 12:first + 40] = 10.0
    second = dates.index(date(2001, 3, 1))
    winds[second:second + 12] = -4.0
    events = detect_major_ssw(dates, winds)
    assert [event["date"] for event in events] == ["2001-01-10", "2001-03-01"]

    no_final_recovery = winds.copy()
    no_final_recovery[second:] = -4.0
    events = detect_major_ssw(dates, no_final_recovery)
    assert [event["date"] for event in events] == ["2001-01-10"]

    rows, labels, extras = audit_catalogue([
        {"date": "1960-01-17", "winter": 1960,
         "previous_day_u60_10": 1.0, "onset_day_u60_10": -1.0,
         "final_recovery_start": "1960-02-01"},
    ])
    assert labels[0]["date"] == "1960-01-17"
    assert any(row["name"] == "MAR 1965" and row["quality"] == "rejected" for row in rows)
    assert extras == []
    print("SELF_TEST_OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    report = build_report(args.data_dir)
    if report["summary"]["consensus_labels"] == 0:
        raise RuntimeError("No consensus early labels; do not write a training label file")
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
