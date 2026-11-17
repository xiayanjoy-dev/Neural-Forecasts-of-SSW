#!/usr/bin/env python
"""Stage one native 1-degree ARCO-ERA5 month for the SSW pipeline."""

import argparse
import json
import os
from calendar import monthrange
from pathlib import Path

import numpy as np
import xarray as xr


SOURCE_URL = (
    "gs://gcp-public-data-arco-era5/ar/"
    "1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr"
)
PRESSURE_LEVELS = np.array(
    [1000, 925, 850, 700, 500, 400, 300, 200, 150, 100,
     70, 50, 30, 20, 10, 7, 5, 3, 2, 1],
    dtype=np.int64,
)
VARIABLES = {
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "t": "temperature",
}


def expected_times(year, month):
    end_day = monthrange(year, month)[1]
    start = np.datetime64(f"{year:04d}-{month:02d}-01T00:00:00")
    stop = np.datetime64(
        f"{year:04d}-{month:02d}-{end_day:02d}T18:00:00"
    )
    return np.arange(start, stop + np.timedelta64(6, "h"), np.timedelta64(6, "h"))


def validate_existing(path, year, month):
    try:
        with xr.open_dataset(path) as dataset:
            times = expected_times(year, month)
            return bool(
                dict(dataset.sizes) == {
                    "valid_time": len(times),
                    "pressure_level": len(PRESSURE_LEVELS),
                    "latitude": 96,
                    "longitude": 360,
                }
                and np.array_equal(dataset.valid_time.values, times)
                and np.array_equal(dataset.pressure_level.values, PRESSURE_LEVELS)
                and np.isclose(float(dataset.latitude.values[0]), 90.0)
                and np.isclose(float(dataset.latitude.values[-1]), -5.0)
                and np.isclose(float(dataset.longitude.values[0]), 0.0)
                and np.isclose(float(dataset.longitude.values[-1]), 359.0)
                and set(dataset.data_vars) == set(VARIABLES)
            )
    except Exception:
        return False


def write_manifest(path, output, year, month, source_url):
    times = expected_times(year, month)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "source": source_url,
        "output": str(output),
        "year": year,
        "month": month,
        "dimensions": {
            "valid_time": len(times),
            "pressure_level": len(PRESSURE_LEVELS),
            "latitude": 96,
            "longitude": 360,
        },
        "time_start": str(times[0]),
        "time_end": str(times[-1]),
        "pressure_levels_hpa": PRESSURE_LEVELS.tolist(),
        "latitude_range": [90.0, -5.0],
        "longitude_range": [0.0, 359.0],
        "variables": VARIABLES,
    }, indent=2) + "\n")


def extract_month(year, month, output, manifest, source_url, overwrite=False):
    output = Path(output)
    manifest = Path(manifest)
    if output.exists() and not overwrite:
        if not validate_existing(output, year, month):
            raise ValueError(f"Existing output failed validation: {output}")
        write_manifest(manifest, output, year, month, source_url)
        print(f"Reuse validated output: {output}", flush=True)
        return

    times = expected_times(year, month)
    source = xr.open_zarr(
        source_url,
        storage_options={"token": "anon"},
        consolidated=True,
    )
    selected = source[list(VARIABLES.values())].sel(
        time=times,
        level=PRESSURE_LEVELS,
        latitude=slice(-5, 90),
    )
    target_latitude = selected.latitude.values[::-1]
    selected = selected.sel(latitude=target_latitude)

    dataset = xr.Dataset(
        coords={
            "valid_time": times,
            "pressure_level": PRESSURE_LEVELS.astype(np.float64),
            "latitude": target_latitude.astype(np.float64),
            "longitude": selected.longitude.values.astype(np.float64),
        }
    )
    for short_name, source_name in VARIABLES.items():
        dataset[short_name] = (
            selected[source_name]
            .rename({"time": "valid_time", "level": "pressure_level"})
            .transpose(
                "valid_time", "pressure_level", "latitude", "longitude"
            )
            .astype("float32")
        )
        dataset[short_name].attrs = {
            "units": selected[source_name].attrs["units"]
        }

    dataset.attrs.update(
        source=source_url,
        source_grid="1 degree regular latitude-longitude conservative ARCO-ERA5",
        temporal_sampling="00, 06, 12, 18 UTC",
        domain="5S-90N, global longitude",
        purpose="native-grid SSW v43-1deg pilot",
    )
    encoding = {
        name: {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "chunksizes": (1, len(PRESSURE_LEVELS), 96, 360),
        }
        for name in VARIABLES
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    dataset.to_netcdf(temporary, engine="netcdf4", encoding=encoding)
    os.replace(temporary, output)
    if not validate_existing(output, year, month):
        raise ValueError(f"Written output failed validation: {output}")

    write_manifest(manifest, output, year, month, source_url)
    print(f"Wrote and validated: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    extract_month(
        args.year,
        args.month,
        args.output,
        args.manifest,
        args.source_url,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
