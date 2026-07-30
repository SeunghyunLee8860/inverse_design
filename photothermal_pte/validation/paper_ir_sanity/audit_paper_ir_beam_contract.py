#!/usr/bin/env python3
"""Build the paper-IR beam audit and fixed scalar source contract.

This module is deliberately solver-free.  It separates values stated in the
paper/SI from inferences and numerical assumptions, then derives a finite
scalar-Gaussian aperture for the first source-only certification.  The user
has selected this explicitly assumed scalar scenario for the present sanity
check.  A vectorial thin-lens comparison is optional and is not a gate.
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
            "fixed_paper_like_scalar_Gaussian_assumed_waist",
            "EXPLICIT_ASSUMPTION",
            SELECTED_W0_M,
            (
                "fixed scalar-Gaussian scenario with an explicitly assumed "
                "1/e^2 intensity waist radius; not an experimentally "
                "reproduced or paper-certified beam"
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
            "item": "Figure 3F wavelength and geometry",
            "value": (
                "11 um Gaussian heat source focused near a straight crystal "
                "edge 45 degrees to the a axis; E parallel a and b"
            ),
            "location": "main paper PDF pp.5-6, Figure 3F and surrounding text",
        },
        {
            "classification": "EXPLICIT_ASSUMPTION",
            "item": "Figure 3F exact beam center and power",
            "value": (
                "not published numerically; no exact edge-normal beam-center "
                "coordinate or Figure-3F incident power was identified"
            ),
            "location": "publication omission recorded by this audit",
        },
        {
            "classification": "PAPER_REPORTED",
            "item": "Figure 3H wavelength, power, polarization and scan region",
            "value": (
                "11 um, 285 uW time-averaged incident power, E parallel a/b, "
                "SPCM map including the off-axis crystal edge"
            ),
            "location": "main paper PDF p.6, Figure 3 caption",
        },
        {
            "classification": "PAPER_REPORTED",
            "item": "Figure 3I extraction",
            "value": (
                "line profile extracted along the black dashed lines in the "
                "11-um Figure-3H SPCM maps"
            ),
            "location": "main paper PDF p.6, Figure 3 caption",
        },
        {
            "classification": "EXPLICIT_ASSUMPTION",
            "item": "Figure 3H/I scan coordinates and step",
            "value": (
                "exact absolute beam coordinates, raster step, and dashed-line "
                "coordinates are not published"
            ),
            "location": "publication omission recorded by this audit",
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
                "scalar Gaussian w0=12 um fixed for this sanity check; "
                "thin-lens comparison is optional and not a gate"
            ),
            "location": "numerical source contract",
        },
    ]

    payload = {
        "status": "FIXED_PAPER_LIKE_SCALAR_GAUSSIAN_CONTRACT_READY_FOR_GPU_PROBE",
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
        "selected_source_only_scenario": {
            **selected,
            "wavelength_m": WAVELENGTH_M,
            "incident_power_for_later_physical_scaling_W": REPORTED_POWER_W,
            "source_model": "scalar Gaussian; waist size and position",
            "scenario_label": (
                "paper-like scalar-Gaussian scenario with an explicitly "
                "assumed waist"
            ),
            "approved_for_current_sanity_check": True,
            "experimentally_reproduced_beam": False,
            "paper_certified_beam": False,
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
        "optional_future_diagnostic": {
            "name": "scalar versus vectorial thin-lens comparison",
            "required_for_current_sanity_check": False,
            "blocks_source_only_or_material_cases": False,
            "execute_now": False,
        },
        "successor_optical_sequence": {
            "condition": "only after the scalar source-only gate passes",
            "cases": [
                "planar TaIrTe4, E parallel a",
                "planar TaIrTe4, E parallel b",
                "straight 45-degree finite edge, E parallel a",
                "straight 45-degree finite edge, E parallel b",
            ],
            "same_scalar_source_geometry": True,
            "same_incident_power_normalization": True,
            "polarization_specific_raw_Q_rescaling_prohibited": True,
        },
        "mesh_contract": {
            "global_mesh": "auto non-uniform, conformal variant 1, accuracy 5",
            "uniform_global_fine_mesh": False,
            "material_case_local_fine_regions": [
                "TaIrTe4",
                "illuminated straight edge",
                "Q extraction region",
            ],
            "far_air_SiO2_Si": "wavelength-appropriate nonuniform coarse mesh",
            "TaIrTe4_thickness_resolution": "separate local z override",
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
        writer = csv.DictWriter(
            handle, fieldnames=list(paper_records[0]), lineterminator="\n"
        )
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
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
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

    report = f"""# Paper-like IR beam audit and fixed scalar source contract

Status: `FIXED_PAPER_LIKE_SCALAR_GAUSSIAN_CONTRACT_READY_FOR_GPU_PROBE`

No FDTD, thermal, PTE, weighting-potential, adjoint, gradient, or
optimization solve was run by this audit.

## Paper-reported facts

- Main paper PDF p.9, Methods 2.3: 40x reflective objective with NA=0.4,
  Block LaserTune QCL covering 7–13 µm, and an approximately 9–16 µm
  diffraction-limited spot size.
- Main paper PDF p.6, Figure 3: the 11 µm maps use 285 µW time-averaged
  incident power; Device A is 130 nm TaIrTe4 on 285 nm SiO2/Si.
- Main paper PDF pp.5–6, Figure 3F: a Gaussian heat source is focused near
  a crystal edge 45 degrees to the a axis at 11 µm for E parallel a/b.
  Its exact beam-center coordinate and incident power are not published.
- Main paper PDF p.6, Figure 3H/I: the 11 µm, 285 µW E-parallel-a/b SPCM
  maps include the off-axis edge; Figure 3I is extracted along the plotted
  dashed lines.  Absolute raster coordinates, scan step, and line coordinates
  are not published.
- SI PDF p.5, Figure S5: 7.84, 9.17, 10, 11 and 12.5 µm were measured for
  E parallel a and b.
- SI PDF p.6, Equations S1–S2: the thermal model uses Gaussian intensity and
  calls `w0=2 sigma` the beam radius.

The paper does **not** state whether the 9–16 µm spot is a radius, diameter,
FWHM, or 1/e^2 width.  It does not publish the exact wavelength-specific
11 µm spot, waist-plane location, objective pupil fill, or alignment error.

## Fixed source-only scenario

The 0.4-NA Airy FWHM estimate at 11 µm is
`{inferred_fwhm*1e6:.6f} µm`; its same-FWHM Gaussian radius is
`{inferred_w0*1e6:.6f} µm`.  The fixed scenario uses a rounded
`w0=12.0 µm`, explicitly as an assumption rather than a paper value.
Its required label is **paper-like scalar-Gaussian scenario with an
explicitly assumed waist**.  It is not an experimentally reproduced beam or
a paper-certified beam.

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

## Source and mesh decision

The scalar Gaussian is fixed for this sanity check.  A vectorial thin-lens
comparison is an optional future diagnostic and is neither executed nor kept
as a blocker.  After one homogeneous-air GPU source-only case passes, the
next cases are planar a/b and straight-45-degree-edge a/b with the identical
scalar geometry and incident-power normalization.

The optical solver uses auto non-uniform mesh, conformal variant 1, accuracy
5.  Material cases must use local fine regions at TaIrTe4, the illuminated
edge, and Q extraction volume, plus a separate TaIrTe4 z override.  A uniform
fine mesh over the 60-µm domain is prohibited, as are Q clipping, smoothing,
gain, and rescaling.
"""
    (output / "PAPER_IR_BEAM_CONTRACT_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(payload["selected_source_only_scenario"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
