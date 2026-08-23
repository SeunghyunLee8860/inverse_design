#!/usr/bin/env python3
"""Calibrate the 4 um Gaussian incident power on the identical all-air grid.

The detector lies below the source.  Removing every material eliminates the
reflected/scattered field that contaminated the full-device ``incident_plane``
diagnostic.  Ea and Eb use the same grid, time window, aperture, and amplitude.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import time

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_4um_model import (
    build_model,
    source_calibration_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.validation_provenance import (
    require_single_visible_gpu,
)


HERE = Path(__file__).resolve().parent
OUT = HERE / "results_fdtdx_4um_source_calibration"


def run_case(polarization: str) -> dict[str, object]:
    model = build_model(
        polarization,
        include_adjoint_source=False,
        air_only_source_calibration=True,
    )
    arrays = (
        model["base"]
        .reset()
        .aset("dispersive_c1", model["fixed_c1"])
        .aset("dispersive_c2", model["fixed_c2"])
        .aset("dispersive_c3", model["fixed_c3"])
    )
    start = time.perf_counter()
    _, output = model["fdtdx"].run_fdtd(
        arrays,
        model["placed"],
        model["config"],
        model["key"],
        show_progress=False,
    )
    runtime_s = time.perf_counter() - start
    eta0 = float(model["fdtdx"].constants.eta0)
    incident_W = float(
        eta0
        * np.asarray(
            model["placed"]["incident_plane"].compute_poynting_flux(
                output.detector_states["incident_plane"]
            )
        )[0]
    )
    target = np.asarray(output.detector_states["target_field"]["phasor"])
    finite = bool(np.all(np.isfinite(target)) and np.isfinite(incident_W))
    return {
        "polarization": polarization,
        "incident_power_W": incident_W,
        "runtime_s": runtime_s,
        "finite": finite,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def main() -> int:
    require_single_visible_gpu()
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [run_case(pol) for pol in ("Ea", "Eb")]
    powers = [float(case["incident_power_W"]) for case in cases]
    mismatch = abs(powers[0] - powers[1]) / max(abs(powers[0]), abs(powers[1]))
    status = (
        "VALIDATED_FDTDX_4UM_SOURCE_POWER_CALIBRATION"
        if all(bool(case["finite"]) and float(case["incident_power_W"]) > 0 for case in cases)
        and mismatch < 5.0e-3
        else "FAILED_FDTDX_4UM_SOURCE_POWER_CALIBRATION"
    )
    summary = {
        "status": status,
        "scope": "all-air source-only calibration on the identical optical grid",
        "cases": cases,
        "Ea_Eb_incident_power_mismatch_relative": mismatch,
        "common_reference_incident_power_W": float(np.mean(powers)),
        "reporting_incident_power_W": 285.0e-6,
        "normalization_contract": (
            "each physical Q is multiplied by the one common ratio "
            "285uW/P_source-only; no polarization matching or Q rescaling"
        ),
        "source_calibration_contract": source_calibration_contract(),
    }
    (OUT / "fdtdx_4um_source_calibration.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (OUT / "fdtdx_4um_source_calibration.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    lines = [
        "# FDTDX 4 um source-only incident-power calibration",
        "",
        f"Status: **{status}**",
        "",
        "This is an all-air run on the identical nonuniform grid and source aperture.",
        "The full-device plane detector is not used as a pure-incident calibration because it contains reflection.",
        "",
        "| polarization | incident power (W) | runtime (s) |",
        "|---|---:|---:|",
    ]
    for case in cases:
        lines.append(
            f"| {case['polarization']} | {case['incident_power_W']:.9e} | {case['runtime_s']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Ea/Eb mismatch: `{100*mismatch:.6f}%`.",
            "All later 285 µW values use the same source-only scale factor; the two polarizations are never matched to one another.",
        ]
    )
    (OUT / "FDTDX_4UM_SOURCE_CALIBRATION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if status.startswith("VALIDATED_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
