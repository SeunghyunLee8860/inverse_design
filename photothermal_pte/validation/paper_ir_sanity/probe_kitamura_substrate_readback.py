#!/usr/bin/env python3
"""Open v261, install the paper-consistent substrate, and save readback."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from photothermal_pte.validation.paper_ir_sanity import (
    run_lumerical_device_a_ir_q as model,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-coefficients", type=int, default=20)
    parser.add_argument(
        "--model",
        choices=("sampled", "single-frequency-nk"),
        default="single-frequency-nk",
    )
    args = parser.parse_args()
    root = model.APPROVED_ROOT
    api = model.APPROVED_API
    sys.path.insert(0, str(api))
    os.environ["VC_LUMERICAL_ROOT"] = str(root)
    os.environ["LUMERICAL_ROOT"] = str(root)
    import lumapi

    fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
    single_frequency = args.model == "single-frequency-nk"
    parsed = SimpleNamespace(
        substrate_optical_model=(
            "paper-kitamura-palik-nk-11um"
            if single_frequency
            else "paper-kitamura-11um"
        ),
        sio2_material_name=(
            model.KITAMURA_SIO2_NK_MATERIAL
            if single_frequency
            else model.KITAMURA_SIO2_MATERIAL
        ),
        si_material_name=(
            model.PALIK_SI_NK_MATERIAL
            if single_frequency
            else model.PALIK_SI_MATERIAL
        ),
        sio2_max_coefficients=args.max_coefficients,
        source_start_m=(
            model.SOURCE_CENTERED_START_M
            if single_frequency
            else model.SOURCE_START_M
        ),
        source_stop_m=(
            model.SOURCE_CENTERED_STOP_M
            if single_frequency
            else model.SOURCE_STOP_M
        ),
        source_center_wavelength_m=model.WAVELENGTH_M,
    )
    contract = model.add_substrate_materials(fdtd, parsed)
    readback = model.substrate_epsilon_readback(fdtd, parsed)
    fdtd.close()
    payload = {
        "status": "VALIDATED_KITAMURA_SUBSTRATE_MATERIAL_READBACK",
        "hostname_context": os.uname().nodename,
        "Lumerical_version": "v261 / solver 8.35 series",
        "contract": contract,
        "readback": readback,
        "FDTD_solve_run": False,
        "thermal_run": False,
        "PTE_run": False,
    }
    fit_error = readback["materials"]["SiO2"][
        "FDTD_fit_relative_error_vs_requested_epsilon"
    ]
    payload["acceptance"] = {
        "SiO2_fit_relative_error_lt_0p5_percent": bool(fit_error < 0.005),
        "SiO2_loss_positive": bool(
            readback["materials"]["SiO2"]["n_complex"]["imag"] > 0.0
        ),
        "Si_loss_positive": bool(
            readback["materials"]["Si"]["n_complex"]["imag"] > 0.0
        ),
    }
    if not all(payload["acceptance"].values()):
        payload["status"] = "FAILED_KITAMURA_SUBSTRATE_MATERIAL_READBACK"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    return 0 if all(payload["acceptance"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
