#!/usr/bin/env python3
"""Solver-free v261 fit-order sweep for the Kitamura silica model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from photothermal_pte.validation.paper_ir_sanity import (
    run_lumerical_device_a_ir_q as model,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum", type=int, default=20)
    args = parser.parse_args()
    sys.path.insert(0, str(model.APPROVED_API))
    os.environ["VC_LUMERICAL_ROOT"] = str(model.APPROVED_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(model.APPROVED_ROOT)
    import lumapi

    fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
    wavelengths_data = np.linspace(model.SOURCE_START_M, model.SOURCE_STOP_M, 1201)
    frequencies_data = model.C0 / wavelengths_data
    requested_data = model.kitamura_2007_sio2_epsilon(wavelengths_data)
    sampled = np.column_stack((frequencies_data, requested_data))
    wavelengths_check = np.linspace(model.SOURCE_START_M, model.SOURCE_STOP_M, 241)
    frequencies_check = model.C0 / wavelengths_check
    requested_check = model.kitamura_2007_sio2_epsilon(wavelengths_check)
    fmin = float(np.min(frequencies_check))
    fmax = float(np.max(frequencies_check))
    target_index = int(np.argmin(abs(wavelengths_check - model.WAVELENGTH_M)))
    cases = []
    for coefficients in range(1, args.maximum + 1):
        name = f"Kitamura_fit_sweep_mc{coefficients}"
        material = fdtd.addmaterial("Sampled data")
        fdtd.setmaterial(material, "name", name)
        fdtd.setmaterial(name, "max coefficients", coefficients)
        fdtd.setmaterial(name, "tolerance", 0.0)
        fdtd.setmaterial(name, "sampled data", sampled)
        fitted_n = np.asarray(
            fdtd.getfdtdindex(
                name,
                frequencies_check,
                fmin,
                fmax,
                1,
            )
        ).reshape(-1)
        fitted_epsilon = fitted_n**2
        difference = fitted_epsilon - requested_check
        cases.append(
            {
                "max_coefficients": coefficients,
                "target_11um_relative_error": float(
                    abs(difference[target_index])
                    / abs(requested_check[target_index])
                ),
                "band_complex_epsilon_NRMSE": float(
                    np.linalg.norm(difference)
                    / np.linalg.norm(requested_check)
                ),
                "band_maximum_pointwise_relative_error": float(
                    np.max(abs(difference) / abs(requested_check))
                ),
                "minimum_fitted_imaginary_epsilon": float(
                    np.min(np.imag(fitted_epsilon))
                ),
                "fitted_n_at_11um": {
                    "real": float(np.real(fitted_n[target_index])),
                    "imag": float(np.imag(fitted_n[target_index])),
                },
                "fitted_epsilon_at_11um": {
                    "real": float(np.real(fitted_epsilon[target_index])),
                    "imag": float(np.imag(fitted_epsilon[target_index])),
                },
            }
        )
    fdtd.close()
    selected = min(cases, key=lambda item: item["target_11um_relative_error"])
    payload = {
        "status": (
            "VALIDATED_KITAMURA_FIT_ORDER_CANDIDATE"
            if selected["target_11um_relative_error"] < 0.005
            and selected["minimum_fitted_imaginary_epsilon"] >= 0.0
            else "FAILED_KITAMURA_FIT_ORDER_SWEEP"
        ),
        "hostname_context": os.uname().nodename,
        "fit_band_m": [model.SOURCE_START_M, model.SOURCE_STOP_M],
        "tolerance": 0.0,
        "selected_by": "minimum relative complex-epsilon error at 11 um",
        "selected": selected,
        "cases": cases,
        "FDTD_solve_run": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    return 0 if payload["status"].startswith("VALIDATED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
