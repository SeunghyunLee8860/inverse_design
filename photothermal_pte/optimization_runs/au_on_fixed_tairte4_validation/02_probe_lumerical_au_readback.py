#!/usr/bin/env python3
"""License/API-only v261 readback of Ordal Au and built-in CRC Au at 10 um.

This script performs no FDTD solve and acquires no GPU engine.  It validates
only the material fit used by a newly opened FDTD design session.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
C0 = 299792458.0
TARGET_WAVELENGTH_M = 10.0e-6
DEFAULT_API_ROOT = Path("/opt/lumerical/v261")
ORDAL_NAME = "Au_Ordal_1987_sampled_10um_validation"
EXACT_NK_NAME = "Au_Ordal_1987_exact_10um_nk"
CRC_NAME = "Au (Gold) - CRC"


def complex_record(value: complex) -> dict[str, float]:
    number = complex(value)
    return {"real": float(number.real), "imag": float(number.imag)}


def load_ordal() -> tuple[np.ndarray, np.ndarray]:
    table = np.genfromtxt(HERE / "data" / "au_ordal_1987_nk.csv", delimiter=",", names=True)
    wavelength_m = np.asarray(table["wavelength_um"], dtype=float) * 1e-6
    index = np.asarray(table["n"], dtype=float) + 1j * np.asarray(table["k"], dtype=float)
    return wavelength_m, index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "results" / "lumerical_au_readback.json")
    parser.add_argument("--max-coefficients", type=int, default=20)
    parser.add_argument("--lumerical-root", type=Path, default=DEFAULT_API_ROOT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    api_root = args.lumerical_root.resolve()
    sys.path.insert(0, str(api_root / "api" / "python"))
    os.environ["VC_LUMERICAL_ROOT"] = str(api_root)
    os.environ["LUMERICAL_ROOT"] = str(api_root)
    import lumapi

    wavelength_m, index = load_ordal()
    frequency = C0 / wavelength_m
    epsilon = index**2
    target_frequency = C0 / TARGET_WAVELENGTH_M
    fit_wavelengths = np.linspace(9.5e-6, 10.5e-6, 101)
    fit_frequencies = C0 / fit_wavelengths
    fmin, fmax = float(np.min(fit_frequencies)), float(np.max(fit_frequencies))

    try:
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
    except Exception as exc:
        payload = {
            "status": "BLOCKED_LUMERICAL_LICENSE_SESSION_STARTUP",
            "FDTD_solve_run": False,
            "GPU_engine_acquired": False,
            "lumerical_root": str(api_root),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        return 3
    try:
        requested = complex(index[np.argmin(abs(wavelength_m - TARGET_WAVELENGTH_M))])
        exact_material = fdtd.addmaterial("(n,k) Material")
        fdtd.setmaterial(exact_material, "name", EXACT_NK_NAME)
        fdtd.setmaterial(EXACT_NK_NAME, "Refractive Index", float(requested.real))
        fdtd.setmaterial(
            EXACT_NK_NAME,
            "Imaginary Refractive Index",
            float(requested.imag),
        )
        material = fdtd.addmaterial("Sampled data")
        fdtd.setmaterial(material, "name", ORDAL_NAME)
        fdtd.setmaterial(ORDAL_NAME, "max coefficients", args.max_coefficients)
        fdtd.setmaterial(ORDAL_NAME, "tolerance", 0.0)
        fdtd.setmaterial(ORDAL_NAME, "sampled data", np.column_stack((frequency, epsilon)))
        direct_exact = complex(
            np.asarray(fdtd.getindex(EXACT_NK_NAME, target_frequency)).reshape(-1)[0]
        )
        fitted_exact = complex(
            np.asarray(fdtd.getfdtdindex(EXACT_NK_NAME, np.asarray([target_frequency]), fmin, fmax, 1)).reshape(-1)[0]
        )
        fitted_sampled = complex(
            np.asarray(fdtd.getfdtdindex(ORDAL_NAME, np.asarray([target_frequency]), fmin, fmax, 1)).reshape(-1)[0]
        )
        fitted_crc = complex(
            np.asarray(fdtd.getfdtdindex(CRC_NAME, np.asarray([target_frequency]), fmin, fmax, 1)).reshape(-1)[0]
        )
        direct_crc = complex(np.asarray(fdtd.getindex(CRC_NAME, target_frequency)).reshape(-1)[0])
        version = str(fdtd.version())
    finally:
        fdtd.close()

    relative_error = abs(fitted_exact**2 - requested**2) / abs(requested**2)
    sampled_relative_error = abs(fitted_sampled**2 - requested**2) / abs(requested**2)
    passed = bool(relative_error < 0.005 and (fitted_exact**2).imag >= 0.0)
    payload = {
        "status": "VALIDATED_LUMERICAL_AU_MATERIAL_READBACK" if passed else "FAILED_LUMERICAL_AU_MATERIAL_READBACK",
        "FDTD_solve_run": False,
        "GPU_engine_acquired": False,
        "lumerical_root": str(api_root),
        "lumerical_product_info": version,
        "wavelength_m": TARGET_WAVELENGTH_M,
        "fit_band_m": [9.5e-6, 10.5e-6],
        "production_single_frequency_nk": {
            "requested_n_plus_ik": complex_record(requested),
            "requested_epsilon": complex_record(requested**2),
            "direct_getindex_n_plus_ik": complex_record(direct_exact),
            "fitted_n_plus_ik": complex_record(fitted_exact),
            "fitted_epsilon": complex_record(fitted_exact**2),
            "relative_complex_epsilon_error": float(relative_error),
        },
        "full_Ordal_sampled_table_diagnostic_only": {
            "fitted_n_plus_ik": complex_record(fitted_sampled),
            "fitted_epsilon": complex_record(fitted_sampled**2),
            "relative_complex_epsilon_error": float(sampled_relative_error),
            "note": "The global 0.667-286 um fit is not the single-wavelength production endpoint.",
        },
        "CRC_sensitivity_only": {
            "direct_getindex_n_plus_ik": complex_record(direct_crc),
            "fitted_getfdtdindex_n_plus_ik": complex_record(fitted_crc),
            "fitted_epsilon": complex_record(fitted_crc**2),
        },
        "gate": "Exact 10-um (n,k) fitted complex-epsilon error <0.5% and passive imaginary epsilon",
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
