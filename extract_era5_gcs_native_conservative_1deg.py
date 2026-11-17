#!/usr/bin/env python3
"""Area-conservatively remap native ARCO-ERA5 to the frozen one-degree grid."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
import xarray as xr


SOURCE = (
    "gs://gcp-public-data-arco-era5/ar/"
    "full_37-1h-0p25deg-chunk-1.zarr-v3"
)
LEVELS = np.asarray(
    [1000, 925, 850, 700, 500, 400, 300, 200, 150, 100,
     70, 50, 30, 20, 10, 7, 5, 3, 2, 1], dtype=np.int64,
)
VARIABLES = {
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "t": "temperature",
}
TARGET_LAT_ASCENDING = np.arange(-5.0, 91.0, 1.0)
TARGET_LONGITUDE = np.arange(0.0, 360.0, 1.0)


def bounds(centers, lower=None, upper=None):
    centers = np.asarray(centers, dtype=float)
    edges = np.empty(centers.size + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    if lower is not None:
        edges = np.maximum(edges, lower)
    if upper is not None:
        edges = np.minimum(edges, upper)
    return edges


def latitude_weights(source_centers, target_centers):
    source_edges = np.sin(np.deg2rad(bounds(source_centers, -90.0, 90.0)))
    target_edges = np.sin(np.deg2rad(bounds(target_centers, -90.0, 90.0)))
    rows, columns, values = [], [], []
    for row in range(target_centers.size):
        low, high = target_edges[row], target_edges[row + 1]
        overlap = np.maximum(
            0.0,
            np.minimum(high, source_edges[1:]) - np.maximum(low, source_edges[:-1]),
        )
        selected = np.flatnonzero(overlap > 0)
        rows.extend([row] * selected.size)
        columns.extend(selected.tolist())
        values.extend((overlap[selected] / (high - low)).tolist())
    return csr_matrix(
        (values, (rows, columns)),
        shape=(target_centers.size, source_centers.size),
    )


def longitude_weights(source_centers, target_centers):
    source_half = 0.5 * (source_centers[1] - source_centers[0])
    target_half = 0.5 * (target_centers[1] - target_centers[0])
    rows, columns, values = [], [], []
    for row, center in enumerate(target_centers):
        low, high = center - target_half, center + target_half
        overlap = np.zeros(source_centers.size)
        for shift in (-360.0, 0.0, 360.0):
            source_low = source_centers + shift - source_half
            source_high = source_centers + shift + source_half
            overlap += np.maximum(
                0.0, np.minimum(high, source_high) - np.maximum(low, source_low)
            )
        selected = np.flatnonzero(overlap > 0)
        rows.extend([row] * selected.size)
        columns.extend(selected.tolist())
        values.extend((overlap[selected] / (2.0 * target_half)).tolist())
    return csr_matrix(
        (values, (rows, columns)),
        shape=(target_centers.size, source_centers.size),
    )


def remap(field, latitude_matrix, longitude_matrix):
    output = np.empty(
        (field.shape[0], latitude_matrix.shape[0], longitude_matrix.shape[0]),
        dtype=np.float32,
    )
    for level in range(field.shape[0]):
        latitude_reduced = latitude_matrix.dot(field[level])
        output[level] = longitude_matrix.dot(latitude_reduced.T).T
    return output


def expected_times(start, end):
    start, end = np.datetime64(start), np.datetime64(end)
    return np.arange(start, end + np.timedelta64(6, "h"), np.timedelta64(6, "h"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    times = expected_times(args.start, args.end)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)

    source = xr.open_zarr(
        SOURCE, storage_options={"token": "anon"}, consolidated=True, chunks=None
    )
    source_latitude = source.latitude.sel(latitude=slice(90.0, -5.5)).values[::-1]
    source_longitude = source.longitude.values
    lat_weights = latitude_weights(source_latitude, TARGET_LAT_ASCENDING)
    lon_weights = longitude_weights(source_longitude, TARGET_LONGITUDE)
    if not np.allclose(np.asarray(lat_weights.sum(axis=1)).ravel(), 1.0):
        raise ValueError("latitude weights do not sum to one")
    if not np.allclose(np.asarray(lon_weights.sum(axis=1)).ravel(), 1.0):
        raise ValueError("longitude weights do not sum to one")

    arrays = {
        name: np.empty((len(times), 20, 96, 360), dtype=np.float32)
        for name in VARIABLES
    }
    for time_index, time in enumerate(times):
        selected = source[list(VARIABLES.values())].sel(
            time=time, level=LEVELS, latitude=slice(90.0, -5.5)
        ).isel(latitude=slice(None, None, -1))
        for short_name, source_name in VARIABLES.items():
            field = np.asarray(selected[source_name].values, dtype=np.float32)
            arrays[short_name][time_index] = remap(
                field, lat_weights, lon_weights
            )[:, ::-1, :]
        print(f"Remapped {time_index + 1}/{len(times)}: {time}", flush=True)

    dataset = xr.Dataset(
        {
            name: (("valid_time", "pressure_level", "latitude", "longitude"), value)
            for name, value in arrays.items()
        },
        coords={
            "valid_time": times,
            "pressure_level": LEVELS.astype(np.float64),
            "latitude": TARGET_LAT_ASCENDING[::-1],
            "longitude": TARGET_LONGITUDE,
        },
        attrs={
            "source": SOURCE,
            "source_grid": "native 0.25 degree ARCO-ERA5 pressure-level Zarr",
            "processing": (
                "Exact cell-overlap area weights in sin(latitude) and periodic "
                "longitude; target centers 90N-5S and 0-359E at one degree."
            ),
        },
    )
    for short_name, source_name in VARIABLES.items():
        dataset[short_name].attrs["units"] = source[source_name].attrs["units"]
    encoding = {
        name: {
            "dtype": "float32", "zlib": True, "complevel": 4, "shuffle": True,
            "chunksizes": (1, 20, 96, 360),
        }
        for name in VARIABLES
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    dataset.to_netcdf(temporary, engine="netcdf4", encoding=encoding)
    os.replace(temporary, output)
    if not all(np.isfinite(dataset[name].values).all() for name in VARIABLES):
        raise ValueError("non-finite remapped field")

    report = {
        "status": "ERA5_GCS_NATIVE_CONSERVATIVE_1DEG_OK",
        "source": SOURCE,
        "output": str(output),
        "time_start": str(times[0]), "time_end": str(times[-1]),
        "n_times": len(times), "levels_hpa": LEVELS.tolist(),
        "method": "separable exact cell-overlap conservative remapping",
        "latitude_weight_row_sum_range": [
            float(lat_weights.sum(axis=1).min()), float(lat_weights.sum(axis=1).max())
        ],
        "longitude_weight_row_sum_range": [
            float(lon_weights.sum(axis=1).min()), float(lon_weights.sum(axis=1).max())
        ],
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
