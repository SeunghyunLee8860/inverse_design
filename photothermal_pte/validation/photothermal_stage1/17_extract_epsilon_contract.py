#!/usr/bin/env python3
"""Extract raw, fitted, and index-monitor epsilon at 4 um.

All reported optical material values are relative permittivity epsilon.  A
complex refractive index is included only as an explicitly separate field.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

import config_stage1 as config
from lumerical_api import select_installation, write_json


C0 = 299792458.0
WAVELENGTH_M = 4.0e-6


def complex_json(value: complex) -> dict[str, float]:
    value = complex(value)
    return {"real": float(value.real), "imag": float(value.imag)}


def scalar(value: Any) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise RuntimeError(f"expected scalar, got {array.shape}")
    return float(np.real(array[0]))


def interp_complex(x: np.ndarray, y: np.ndarray, target: float) -> complex:
    order = np.argsort(x)
    x = np.asarray(x, float)[order]
    y = np.asarray(y, complex)[order]
    return complex(
        np.interp(target, x, y.real)
        + 1j * np.interp(target, x, y.imag)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lumerical-version", choices=("v251", "v261"), default="v261")
    parser.add_argument("--material-name", default="TaIrTe4_ani")
    args = parser.parse_args()

    project = Path(args.project).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    installation = select_installation(args.lumerical_version)
    os.environ["VC_LUMERICAL_ROOT"] = str(installation.root)
    os.environ["LUMERICAL_ROOT"] = str(installation.root)
    for path in (config.REPOSITORY_ROOT, config.REPOSITORY_ROOT / "bundle"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import eqc_lib as runtime

    fdtd = runtime.open_control(project)
    try:
        target_frequency = C0 / WAVELENGTH_M
        source_start = scalar(fdtd.getglobalsource("wavelength start"))
        source_stop = scalar(fdtd.getglobalsource("wavelength stop"))
        frequency_ends = C0 / np.asarray([source_start, source_stop], float)
        fmin = float(np.min(frequency_ends))
        fmax = float(np.max(frequency_ends))
        single_frequency_limit_used = bool(np.isclose(fmin, fmax))
        if single_frequency_limit_used:
            # getfdtdindex requires an ordered range.  This symmetric 2e-9
            # relative window is the explicit limit of the actual zero-span
            # source range; the monitor value below remains the direct record
            # of what the completed FDTD run used.
            fmin = target_frequency * (1.0 - 1e-9)
            fmax = target_frequency * (1.0 + 1e-9)

        fitted_epsilon: list[complex] = []
        for component in (1, 2, 3):
            fitted_index = np.asarray(
                fdtd.getfdtdindex(
                    args.material_name,
                    np.asarray([target_frequency]),
                    fmin,
                    fmax,
                    component,
                )
            ).reshape(-1)[0]
            fitted_epsilon.append(complex(fitted_index) ** 2)

        sampled = np.asarray(
            fdtd.getmaterial(args.material_name, "sampled data")
        )
        if sampled.ndim != 2 or sampled.shape[1] < 4:
            raise RuntimeError(
                f"{args.material_name} sampled data shape is {sampled.shape}, "
                "expected frequency plus epsilon_x/y/z"
            )
        raw_epsilon = [
            interp_complex(sampled[:, 0].real, sampled[:, component], target_frequency)
            for component in (1, 2, 3)
        ]

        monitor = "stage1_pabs_adv::index"
        x = np.asarray(fdtd.getdata(monitor, "x", 1), float).reshape(-1)
        y = np.asarray(fdtd.getdata(monitor, "y", 1), float).reshape(-1)
        z = np.asarray(fdtd.getdata(monitor, "z", 1), float).reshape(-1)
        ix = int(np.argmin(np.abs(x)))
        iy = int(np.argmin(np.abs(y)))
        iz = int(np.argmin(np.abs(z + 50e-9)))
        monitor_epsilon: list[complex] = []
        for axis in "xyz":
            index = np.asarray(
                fdtd.getdata(monitor, f"index_{axis}", 1)
            ).squeeze()
            monitor_epsilon.append(complex(index[ix, iy, iz]) ** 2)
    finally:
        fdtd.close()

    axes = []
    for axis, raw, fitted, monitored in zip(
        "xyz", raw_epsilon, fitted_epsilon, monitor_epsilon
    ):
        axes.append(
            {
                "axis": axis,
                "quantity": "complex relative permittivity epsilon_r",
                "raw_imported_epsilon": complex_json(raw),
                "fitted_epsilon_getfdtdindex": complex_json(fitted),
                "index_monitor_epsilon_index_squared": complex_json(monitored),
                "fitted_refractive_index_sqrt_epsilon": complex_json(
                    np.sqrt(fitted)
                ),
                "fit_relative_error_vs_raw_epsilon": float(
                    abs(fitted - raw) / max(abs(raw), np.finfo(float).tiny)
                ),
                "index_monitor_relative_error_vs_fitted_epsilon": float(
                    abs(monitored - fitted)
                    / max(abs(fitted), np.finfo(float).tiny)
                ),
            }
        )
    result = {
        "project": str(project),
        "solver_version": args.lumerical_version,
        "material_name": args.material_name,
        "wavelength_m": WAVELENGTH_M,
        "frequency_Hz": target_frequency,
        "source_wavelength_range_m": [source_start, source_stop],
        "getfdtdindex_frequency_range_Hz": [fmin, fmax],
        "single_frequency_limit_used": single_frequency_limit_used,
        "axes": axes,
        "quantity_warning": (
            "raw/fitted/index-monitor values are epsilon_r, not n; the only "
            "n field is explicitly named fitted_refractive_index_sqrt_epsilon"
        ),
        "heat_run": False,
        "clipping": False,
        "flux_gain": False,
    }
    write_json(output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
