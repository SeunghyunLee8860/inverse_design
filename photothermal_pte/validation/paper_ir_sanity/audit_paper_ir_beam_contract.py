#!/usr/bin/env python3
"""Build the paper-IR beam audit and one source-only candidate contract.

This module is deliberately solver-free.  It separates values stated in the
paper/SI from inferences and numerical assumptions, then derives a finite
scalar-Gaussian aperture for the first source-only certification.  The scalar
case is a candidate only: the selected waist is comparable to the wavelength,
so a matched fully-vectorial thin-lens comparison remains a production gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import erf


REPOSITORY = Path(__file__).resolve().parents[3]
MAIN_PAPER = Path(
    "/home/seunghyun/tairte4/papers/"
    "Adv Funct Materials - 2026 - Blevins - Large Transverse "
    "Thermoelectric Effect in Weyl Semimetal TaIrTe4 Engineered for-2.pdf"
)
SUPPORTING_INFORMATION = Path(
    "/home/seunghyun/tairte4/papers/adfm75986-sup-0001-suppmat-2.pdf"
)

WAVELENGTH_M = 11.0e-6
OBJECTIVE_NA = 0.4
REPORTED_POWER_W = 285.0e-6
REPORTED_QCL_RANGE_M = (7.0e-6, 13.0e-6)
REPORTED_SPOT_RANGE_M = (9.0e-6, 16.0e-6)
AIRY_FWHM_DIAMETER_FACTOR = 0.514
SELECTED_W0_M = 12.0e-6
FOCUS_Z_M = -65.0e-9
SOURCE_Z_M = 5.0e-6
SOURCE_SPAN_M = 50.0e-6
LATERAL_DOMAIN_M = 60.0e-6
FDTD_Z_MIN_M = FOCUS_Z_M - WAVELENGTH_M
FDTD_Z_MAX_M = SOURCE_Z_M + WAVELENGTH_M
CAPTURE_GATE = 0.999
BOUNDARY_MAX_GATE = 1.0e-3
BOUNDARY_MEAN_GATE = 1.0e-4


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=False,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def square_metrics(half_span_m: float, radius_m: float) -> dict[str, float]:
    t = half_span_m / radius_m
    boundary_max = math.exp(-2.0 * t**2)
    boundary_mean = (
        boundary_max
        * math.sqrt(math.pi / 2.0)
        * float(erf(math.sqrt(2.0) * t))
        / (2.0 * t)
    )
    return {
        "half_span_over_radius": t,
        "square_captured_fraction": float(erf(math.sqrt(2.0) * t) ** 2),
        "inscribed_circle_fraction": 1.0 - math.exp(-2.0 * t**2),
        "boundary_max_intensity_over_peak": boundary_max,
        "boundary_mean_intensity_over_peak": boundary_mean,
    }


def required_half_span_ratio() -> dict[str, float]:
    capture = float(
        brentq(
            lambda t: float(erf(math.sqrt(2.0) * t) ** 2) - CAPTURE_GATE,
            0.1,
            5.0,
        )
    )
    maximum = math.sqrt(-0.5 * math.log(BOUNDARY_MAX_GATE))
    mean = float(
        brentq(
            lambda t: (
                math.exp(-2.0 * t**2)
                * math.sqrt(math.pi / 2.0)
                * float(erf(math.sqrt(2.0) * t))
                / (2.0 * t)
                - BOUNDARY_MEAN_GATE
            ),
            0.1,
            5.0,
        )
    )
    return {
        "capture_gate": capture,
        "boundary_max_gate": maximum,
        "boundary_mean_gate": mean,
        "governing": max(capture, maximum, mean),
    }


def beam_scenario(
    name: str,
    classification: str,
    w0_m: float,
    definition: str,
) -> dict[str, Any]:
    eta = WAVELENGTH_M / (math.pi * w0_m)
    rayleigh = math.pi * w0_m**2 / WAVELENGTH_M
    distance = SOURCE_Z_M - FOCUS_Z_M
    source_radius = w0_m * math.sqrt(1.0 + (distance / rayleigh) ** 2)
    return {
        "name": name,
        "classification": classification,
        "w0_m": w0_m,
        "definition": definition,
        "eta_paraxial": eta,
        "Rayleigh_range_m": rayleigh,
        "source_to_focus_distance_m": distance,
        "source_plane_radius_m": source_radius,
        "selected_aperture": square_metrics(0.5 * SOURCE_SPAN_M, source_radius),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    interpolated_spot = REPORTED_SPOT_RANGE_M[0] + (
        (WAVELENGTH_M - REPORTED_QCL_RANGE_M[0])
        / (REPORTED_QCL_RANGE_M[1] - REPORTED_QCL_RANGE_M[0])
        * (REPORTED_SPOT_RANGE_M[1] - REPORTED_SPOT_RANGE_M[0])
    )
    inferred_fwhm = AIRY_FWHM_DIAMETER_FACTOR * WAVELENGTH_M / OBJECTIVE_NA
    inferred_w0 = inferred_fwhm / math.sqrt(2.0 * math.log(2.0))
    scenarios = [
        {
            "name": "paper_reported_range_only",
            "classification": "PAPER_REPORTED",
            "wavelength_m": WAVELENGTH_M,
            "spot_range_m": list(REPORTED_SPOT_RANGE_M),
            "spot_definition": "not published",
            "w0_m": None,
        },
        beam_scenario(
            "reported_range_interpolated_as_1e2_diameter",
            "EXPLICIT_ASSUMPTION",
            0.5 * interpolated_spot,
            (
                "linear interpolation of the approximate 9-16 um range at "
                "11 um, then assuming it is a 1/e^2 intensity diameter"
            ),
        ),
        beam_scenario(
            "reported_range_interpolated_as_FWHM_diameter",
            "PAPER_INFERRED",
            interpolated_spot / math.sqrt(2.0 * math.log(2.0)),
            (
                "linear interpolation of the approximate range, interpreted "
                "as Gaussian intensity FWHM diameter"
            ),
        ),
        beam_scenario(
            "NA0p4_Airy_FWHM_Gaussian_equivalent",
            "PAPER_INFERRED",
            inferred_w0,
            (
                "0.514 lambda/NA Airy FWHM diameter converted to the "
                "same-FWHM Gaussian 1/e^2 intensity radius"
            ),
        ),
        beam_scenario(
            "selected_scalar_source_only_candidate",
            "EXPLICIT_ASSUMPTION",
            SELECTED_W0_M,
            (
                "rounded Gaussian-equivalent radius for the first scalar "
                "source-only certification; not yet production approved"
            ),
        ),
    ]
    selected = scenarios[-1]
    ratios = required_half_span_ratio()
    selected_radius = selected["source_plane_radius_m"]
    minimum_span = 2.0 * ratios["governing"] * selected_radius
    source_metrics = square_metrics(0.5 * SOURCE_SPAN_M, selected_radius)

    paper_records = [
        {
            "classification": "PAPER_REPORTED",
            "item": "LWIR source",
            "value": "Block LaserTune QCL, 7-13 um",
            "location": "main paper PDF p.9, Methods 2.3",
        },
        {
            "classification": "PAPER_REPORTED",
            "item": "objective",
            "value": "40x reflective objective, NA=0.4",
            "location": "main paper PDF pp.8-9, Methods 2.3",
        },
        {
            "classification": "PAPER_REPORTED",
            "item": "IR spot",
            "value": "approximately 9-16 um diffraction-limited spot size",
            "location": "main paper PDF p.9, Methods 2.3",
        },
        {
            "classification": "PAPER_REPORTED",
            "item": "Figure 3H/I wavelength and power",
            "value": "11 um and 285 uW time-averaged incident power",
            "location": "main paper PDF p.6, Figure 3 caption",
        },
        {
            "classification": "PAPER_REPORTED",
            "item": "LWIR scan wavelengths",
            "value": "7.84, 9.17, 10, 11, 12.5 um; E parallel a and b",
            "location": "SI PDF p.5, Figure S5 caption",
        },
        {
            "classification": "PAPER_REPORTED",
            "item": "thermal source equation",
            "value": "Gaussian intensity; w0=2 sigma is called beam radius",
            "location": "SI PDF p.6, Equations S1-S2",
        },
        {
            "classification": "PAPER_REPORTED",
            "item": "Figure 3F/G geometry",
            "value": "130 nm flake, 285 nm SiO2/Si, 45-degree insulating edge",
            "location": "main paper PDF pp.5-6 and Figure 3 caption",
        },
        {
            "classification": "PAPER_INFERRED",
            "item": "incidence angle for LWIR source model",
            "value": "normal incidence",
            "location": (
                "SI PDF p.7 normal-incidence symmetry discussion and shared "
                "SPCM optical axis; no numerical alignment tolerance reported"
            ),
        },
        {
            "classification": "EXPLICIT_ASSUMPTION",
            "item": "waist plane",
            "value": "TaIrTe4 midplane z=-65 nm",
            "location": "numerical source contract; not stated by paper",
        },
        {
            "classification": "EXPLICIT_ASSUMPTION",
            "item": "first source model",
            "value": (
                "scalar Gaussian candidate w0=12 um; vector thin-lens match "
                "required before production"
            ),
            "location": "numerical source contract",
        },
    ]

    payload = {
        "status": "PROPOSED_PAPER_IR_SOURCE_CONTRACT_READY_FOR_GPU_PROBE",
        "generation_commit": git_commit(),
        "paper_sources": {
            "main": {
                "path": str(MAIN_PAPER),
                "size_bytes": MAIN_PAPER.stat().st_size,
                "sha256": sha256(MAIN_PAPER),
            },
            "supporting_information": {
                "path": str(SUPPORTING_INFORMATION),
                "size_bytes": SUPPORTING_INFORMATION.stat().st_size,
                "sha256": sha256(SUPPORTING_INFORMATION),
            },
        },
        "paper_audit": paper_records,
        "beam_scenarios": scenarios,
        "selected_source_only_candidate": {
            **selected,
            "wavelength_m": WAVELENGTH_M,
            "incident_power_for_later_physical_scaling_W": REPORTED_POWER_W,
            "source_model": "scalar Gaussian candidate",
            "production_approval": False,
            "production_blocker": (
                "w0 is only about 1.09 wavelengths; matched vectorial "
                "thin-lens flake-plane field comparison remains required"
            ),
            "propagation_direction": "-z",
            "distance_from_waist_property_m": -(SOURCE_Z_M - FOCUS_Z_M),
            "distance_sign_note": (
                "negative for a backward source converging to the lower-z "
                "waist; the legacy positive sign is not reused"
            ),
            "source_span_m": SOURCE_SPAN_M,
            "minimum_source_span_m_from_all_gates": minimum_span,
            "source_aperture_metrics": source_metrics,
            "lateral_domain_m": LATERAL_DOMAIN_M,
            "lateral_source_to_PML_margin_m": (
                0.5 * (LATERAL_DOMAIN_M - SOURCE_SPAN_M)
            ),
            "FDTD_z_bounds_m": [FDTD_Z_MIN_M, FDTD_Z_MAX_M],
            "source_to_top_PML_margin_m": FDTD_Z_MAX_M - SOURCE_Z_M,
            "focus_to_bottom_PML_margin_m": FOCUS_Z_M - FDTD_Z_MIN_M,
            "all_optical_boundaries": "PML",
            "periodic_or_Bloch": False,
        },
        "aperture_gate_ratios": ratios,
        "scalar_vector_match_contract": {
            "required": True,
            "not_implemented_by_boolean_only": True,
            "vector_source": "fully vectorial thin-lens, objective NA=0.4",
            "must_match": [
                "flake-plane incident power",
                "beam center",
                "fitted and second-moment widths",
                "focus",
                "aperture/NA",
                "normalized flake-plane intensity",
            ],
            "gates": {
                "incident_power_relative_difference": 0.005,
                "beam_width_relative_difference": 0.005,
                "normalized_intensity_NRMSE": 0.005,
            },
        },
        "prohibited": {
            "thermal": True,
            "PTE": True,
            "adjoint": True,
            "gradient": True,
            "optimization": True,
            "CPU_FDTD_fallback": True,
        },
    }
    summary = output / "paper_ir_beam_contract_summary.json"
    summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (output / "paper_ir_beam_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paper_records[0]))
        writer.writeheader()
        writer.writerows(paper_records)
    with (output / "paper_ir_beam_scenarios.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "name",
            "classification",
            "w0_um",
            "eta_paraxial",
            "Rayleigh_range_um",
            "source_plane_radius_um",
            "definition",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in scenarios:
            writer.writerow(
                {
                    "name": row["name"],
                    "classification": row["classification"],
                    "w0_um": (
                        "" if row.get("w0_m") is None else row["w0_m"] * 1e6
                    ),
                    "eta_paraxial": row.get("eta_paraxial", ""),
                    "Rayleigh_range_um": (
                        ""
                        if row.get("Rayleigh_range_m") is None
                        else row["Rayleigh_range_m"] * 1e6
                    ),
                    "source_plane_radius_um": (
                        ""
                        if row.get("source_plane_radius_m") is None
                        else row["source_plane_radius_m"] * 1e6
                    ),
                    "definition": row.get("definition", row.get("spot_definition")),
                }
            )

    width = np.linspace(0.1, 2.4, 500)
    capture = erf(np.sqrt(2.0) * width) ** 2
    boundary_max = np.exp(-2.0 * width**2)
    boundary_mean = (
        boundary_max
        * np.sqrt(np.pi / 2.0)
        * erf(np.sqrt(2.0) * width)
        / (2.0 * width)
    )
    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    axis.semilogy(width, 1.0 - capture, label="square power outside")
    axis.semilogy(width, boundary_max, label="boundary maximum / peak")
    axis.semilogy(width, boundary_mean, label="boundary mean / peak")
    axis.axvline(
        0.5 * SOURCE_SPAN_M / selected_radius,
        color="black",
        linestyle="--",
        label="selected 50 µm span",
    )
    axis.axhline(1.0 - CAPTURE_GATE, color="tab:blue", linestyle=":")
    axis.axhline(BOUNDARY_MAX_GATE, color="tab:orange", linestyle=":")
    axis.axhline(BOUNDARY_MEAN_GATE, color="tab:green", linestyle=":")
    axis.set(
        xlabel="source half-span / source-plane Gaussian radius",
        ylabel="fraction",
        title="Scalar source-aperture contract at λ=11 µm",
        ylim=(1e-7, 1.0),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output / "paper_ir_source_aperture_contract.png", dpi=180)
    plt.close(figure)

    report = f"""# Paper-like IR beam audit and proposed source contract

Status: `PROPOSED_PAPER_IR_SOURCE_CONTRACT_READY_FOR_GPU_PROBE`

No FDTD, thermal, PTE, weighting-potential, adjoint, gradient, or
optimization solve was run by this audit.

## Paper-reported facts

- Main paper PDF p.9, Methods 2.3: 40x reflective objective with NA=0.4,
  Block LaserTune QCL covering 7–13 µm, and an approximately 9–16 µm
  diffraction-limited spot size.
- Main paper PDF p.6, Figure 3: the 11 µm maps use 285 µW time-averaged
  incident power; Device A is 130 nm TaIrTe4 on 285 nm SiO2/Si.
- SI PDF p.5, Figure S5: 7.84, 9.17, 10, 11 and 12.5 µm were measured for
  E parallel a and b.
- SI PDF p.6, Equations S1–S2: the thermal model uses Gaussian intensity and
  calls `w0=2 sigma` the beam radius.

The paper does **not** state whether the 9–16 µm spot is a radius, diameter,
FWHM, or 1/e^2 width.  It does not publish the exact wavelength-specific
11 µm spot, waist-plane location, objective pupil fill, or alignment error.

## Selected first source-only candidate

The 0.4-NA Airy FWHM estimate at 11 µm is
`{inferred_fwhm*1e6:.6f} µm`; its same-FWHM Gaussian radius is
`{inferred_w0*1e6:.6f} µm`.  The first candidate therefore uses a rounded
`w0=12.0 µm`, explicitly as an assumption rather than a paper value.

- eta=lambda/(pi*w0): `{selected['eta_paraxial']:.6f}`
- Rayleigh range: `{selected['Rayleigh_range_m']*1e6:.6f} µm`
- source-to-focus distance: `{selected['source_to_focus_distance_m']*1e6:.6f} µm`
- expected source-plane radius: `{selected_radius*1e6:.6f} µm`
- source span: `50 µm`; lateral domain: `60 µm`
- fitted-Gaussian square capture: `{source_metrics['square_captured_fraction']:.8%}`
- expected boundary maximum/peak:
  `{source_metrics['boundary_max_intensity_over_peak']:.8e}`
- expected boundary mean/peak:
  `{source_metrics['boundary_mean_intensity_over_peak']:.8e}`

The backward source must use
`distance from waist = {-selected['source_to_focus_distance_m']*1e6:.6f} µm`.
The legacy positive sign is not reused.

## Scalar/vector status

Since `w0` is only about 1.09 wavelengths, the scalar source is a
source-only calibration candidate, not a production approval.  Before any
planar/edge material case, it must be matched against a fully vectorial
thin-lens source using NA=0.4 by realized flake-plane power, center, fitted
and second-moment widths, focus, and normalized spatial intensity.  Merely
toggling the scalar Boolean with identical numeric parameters is prohibited.
"""
    (output / "PAPER_IR_BEAM_CONTRACT_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(payload["selected_source_only_candidate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
