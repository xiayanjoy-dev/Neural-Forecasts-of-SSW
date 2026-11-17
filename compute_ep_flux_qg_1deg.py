#!/usr/bin/env python
"""Compute leakage-safe instantaneous QG EP flux from native 1-degree UVT."""

import argparse
import os
from pathlib import Path

import numpy as np


A = 6.371e6
OMEGA = 7.2921e-5
KAPPA = 0.286
P0_HPA = 1000.0


def qg_ep_flux(u, v, t, latitude, pressure_hpa):
    """Return F_phi, F_p, div(F), and div(F)/(a cos(phi))."""
    phi = np.deg2rad(np.asarray(latitude, dtype=np.float64))
    pressure_pa = np.asarray(pressure_hpa, dtype=np.float64) * 100.0
    cos_phi = np.cos(phi)
    f = 2.0 * OMEGA * np.sin(phi)

    theta_factor = (P0_HPA / pressure_hpa) ** KAPPA
    theta = t * theta_factor[:, None, None]
    ubar = np.mean(u, axis=-1)
    vbar = np.mean(v, axis=-1)
    thetabar = np.mean(theta, axis=-1)
    u_prime = u - ubar[:, :, None]
    v_prime = v - vbar[:, :, None]
    theta_prime = theta - thetabar[:, :, None]
    uv_eddy = np.mean(u_prime * v_prime, axis=-1)
    vtheta_eddy = np.mean(v_prime * theta_prime, axis=-1)

    theta_p = np.gradient(thetabar, pressure_pa, axis=0, edge_order=2)
    stable = np.where(np.abs(theta_p) >= 1e-8, theta_p, np.nan)
    f_phi = -A * cos_phi[None, :] * uv_eddy
    f_p = (
        A * f[None, :] * cos_phi[None, :]
        * vtheta_eddy / stable
    )

    meridional = np.gradient(
        f_phi * cos_phi[None, :], phi, axis=1, edge_order=2
    ) / (A * cos_phi[None, :])
    vertical = np.gradient(f_p, pressure_pa, axis=0, edge_order=2)
    divergence = meridional + vertical
    acceleration = divergence / (A * cos_phi[None, :])
    return f_phi, f_p, divergence, acceleration


def compute_file(input_path, output_path, overwrite=False):
    from netCDF4 import Dataset

    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        print(f"Reuse existing EP output: {output_path}", flush=True)
        return

    with Dataset(input_path) as source:
        latitude = np.asarray(source.variables["latitude"][:], dtype=np.float64)
        pressure = np.asarray(
            source.variables["pressure_level"][:], dtype=np.float64
        )
        time_values = source.variables["valid_time"][:]
        time_units = source.variables["valid_time"].units
        time_calendar = getattr(source.variables["valid_time"], "calendar", None)
        time_count = len(time_values)

        lat_index = np.where((latitude >= 40) & (latitude <= 75))[0]
        level_index = np.where((pressure >= 10) & (pressure <= 500))[0]
        shape = (time_count, len(level_index), len(lat_index))
        f_phi_out = np.empty(shape, dtype=np.float32)
        f_p_out = np.empty(shape, dtype=np.float32)
        divergence_out = np.empty(shape, dtype=np.float32)
        acceleration_out = np.empty(shape, dtype=np.float32)

        for time_index in range(time_count):
            u = np.asarray(source.variables["u"][time_index], dtype=np.float64)
            v = np.asarray(source.variables["v"][time_index], dtype=np.float64)
            t = np.asarray(source.variables["t"][time_index], dtype=np.float64)
            f_phi, f_p, divergence, acceleration = qg_ep_flux(
                u, v, t, latitude, pressure
            )
            selection = np.ix_(level_index, lat_index)
            f_phi_out[time_index] = f_phi[selection]
            f_p_out[time_index] = f_p[selection]
            divergence_out[time_index] = divergence[selection]
            acceleration_out[time_index] = acceleration[selection]

    if not all(np.all(np.isfinite(values)) for values in (
        f_phi_out, f_p_out, divergence_out, acceleration_out
    )):
        raise ValueError("Non-finite values in retained QG EP domain")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    with Dataset(temporary, "w", format="NETCDF4") as output:
        output.createDimension("time", time_count)
        output.createDimension("level", len(level_index))
        output.createDimension("lat", len(lat_index))

        time_var = output.createVariable("time", "f8", ("time",))
        time_var.units = time_units
        if time_calendar is not None:
            time_var.calendar = time_calendar
        time_var[:] = time_values
        level_var = output.createVariable("level", "f8", ("level",))
        level_var.units = "hPa"
        level_var[:] = pressure[level_index]
        lat_var = output.createVariable("lat", "f8", ("lat",))
        lat_var.units = "degrees_north"
        lat_var[:] = latitude[lat_index]

        definitions = {
            "f_phi": (f_phi_out, "m3 s-2", "meridional QG EP flux component"),
            "f_p": (f_p_out, "m2 Pa s-2", "vertical QG EP flux component"),
            "epdiv": (divergence_out, "m2 s-2", "QG EP flux divergence"),
            "ep_accel": (
                acceleration_out,
                "m s-2",
                "zonal acceleration from QG EP flux divergence",
            ),
        }
        for name, (values, units, long_name) in definitions.items():
            variable = output.createVariable(
                name,
                "f4",
                ("time", "level", "lat"),
                zlib=True,
                complevel=3,
                shuffle=True,
            )
            variable.units = units
            variable.long_name = long_name
            variable[:] = values
        output.formulation = (
            "Instantaneous quasi-geostrophic EP flux in pressure coordinates: "
            "F_phi=-a*cos(phi)*mean(u_prime*v_prime); "
            "F_p=a*f*cos(phi)*mean(v_prime*theta_prime)/d(theta_bar)/dp."
        )
        output.leakage_control = (
            "Static stability and eddy covariances use only the current 6-hour state."
        )
        output.source_file = str(input_path)
    os.replace(temporary, output_path)
    print(f"Wrote QG EP diagnostics: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    compute_file(args.input, args.output, args.overwrite)


if __name__ == "__main__":
    main()
