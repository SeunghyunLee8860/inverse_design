#!/usr/bin/env python3
"""GPU-only 11 um Lumerical optical-Q sanity check for paper Device A.

This is deliberately separate from the inverse-design geometry.  It reuses
the certified finite-Gaussian absorption extraction, but replaces the square
2 um flake with an explicitly documented, approximate Device-A polygon and
swaps the optical material axes into the paper image coordinates:

    lab x = crystal b, lab y = crystal a.

The exact Device-A CAD and wavelength-specific spot radius are not published.
Those quantities therefore remain named physical scenarios, not hidden fits.
No thermal solve, adjoint, gradient, or optimization is performed here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath
import numpy as np
from scipy.optimize import least_squares


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
from photothermal_pte.validation.paper_ir_sanity import (
    audit_paper_ir_beam_contract as source_contract,
)

STAGE1 = HERE.parent / "photothermal_stage1"
BASE_SCRIPT = STAGE1 / "27_validate_finite_2um_optical_q.py"
APPROVED_ROOT = Path("/home/seunghyun/lumerical_r12/opt/lumerical/v261")
APPROVED_API = APPROVED_ROOT / "api" / "python"

C0 = 299792458.0
WAVELENGTH_M = 11.0e-6
SOURCE_START_M = 7.0e-6
SOURCE_STOP_M = 13.0e-6
FLAKE_THICKNESS_M = 130.0e-9
SIO2_THICKNESS_M = 285.0e-9
SI_DEPTH_M = 3.0e-6
FDTD_Z_MIN_M = -(FLAKE_THICKNESS_M + SIO2_THICKNESS_M + SI_DEPTH_M)
FDTD_Z_MAX_M = 10.0e-6
SOURCE_Z_M = 5.0e-6
FOCUS_Z_M = -0.5 * FLAKE_THICKNESS_M
INCIDENT_Z_M = 0.60e-6
W2_INCIDENT_PLANE_Z_M = 50.0e-9
W2_FLAKE_MIDPLANE_Z_M = -0.5 * FLAKE_THICKNESS_M
W2_INCIDENT_MONITOR = "w2_incident_plane"
W2_FLAKE_FIELD_MONITOR = "w2_flake_midplane"
PABS_PADDING_M = 50.0e-9
PRODUCTION_MATERIAL_NAME = "TaIrTe4_DeviceA_lab_x_b_y_a_z_b_closure"
LEGACY_MATERIAL_NAME = "TaIrTe4_DeviceA_lab_x_b_y_a_legacy_z16"
MATERIAL_NAME = PRODUCTION_MATERIAL_NAME
SIO2_MATERIAL = "paper_ir_SiO2_n1p38"

# Approximation digitized from the scale bar and outline in paper Fig. 2A.
# It preserves the two essential edge orientations: a 45-degree upper-left
# off-axis edge and a right edge parallel to crystal a.
FLAKE_VERTICES_UM = np.asarray(
    [
        [-11.0, -10.0],
        [12.0, -10.0],
        [12.0, 10.0],
        [-6.0, 10.0],
        [-6.0, 6.0],
        [-11.0, 1.0],
    ],
    dtype=float,
)
FLAKE_BOUNDS_M = {
    "x": (
        float(np.min(FLAKE_VERTICES_UM[:, 0])) * 1e-6,
        float(np.max(FLAKE_VERTICES_UM[:, 0])) * 1e-6,
    ),
    "y": (
        float(np.min(FLAKE_VERTICES_UM[:, 1])) * 1e-6,
        float(np.max(FLAKE_VERTICES_UM[:, 1])) * 1e-6,
    ),
    "z": (-FLAKE_THICKNESS_M, 0.0),
}
INNER_BOX = {
    "x": (FLAKE_BOUNDS_M["x"][0] - 0.75e-6, FLAKE_BOUNDS_M["x"][1] + 0.75e-6),
    "y": (FLAKE_BOUNDS_M["y"][0] - 0.75e-6, FLAKE_BOUNDS_M["y"][1] + 0.75e-6),
    "z": (-0.75e-6, 1.25e-6),
}


def polygon_area(vertices_um: np.ndarray) -> float:
    x = vertices_um[:, 0]
    y = vertices_um[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--case", choices=("empty-stack", "finite-flake"), required=True
    )
    parser.add_argument(
        "--polarization",
        choices=("a", "b"),
        required=True,
        help="crystal-axis polarization; lab x=b and lab y=a",
    )
    parser.add_argument(
        "--geometry",
        choices=("device-a-polygon", "straight-45-edge", "planar-stack"),
        default="device-a-polygon",
    )
    parser.add_argument("--domain-um", type=float, default=60.0)
    parser.add_argument("--pml-layers", type=int, default=24)
    parser.add_argument("--flake-dz-nm", type=float, default=5.0)
    parser.add_argument(
        "--local-xy-mesh-nm",
        type=float,
        default=100.0,
        help=(
            "material-case local x/y mesh over the illuminated/Q region; "
            "50 nm is the required refinement comparison"
        ),
    )
    parser.add_argument(
        "--refinement-half-span-um",
        type=float,
        default=None,
        help=(
            "optional nested fine-mesh half span about the beam centre; "
            "the full Q/closure region remains on the fixed 100-nm outer "
            "mesh so no source support is cropped or deleted"
        ),
    )
    parser.add_argument("--simulation-time-ps", type=float, default=1.2)
    parser.add_argument("--auto-shutoff-min", type=float, default=1.0e-5)
    parser.add_argument("--source-span-um", type=float, default=50.0)
    parser.add_argument("--waist-um", type=float, default=12.0)
    parser.add_argument(
        "--source-object-waist-um",
        type=float,
        default=None,
        help=(
            "Lumerical source-object input; production defaults to the "
            "SHA-pinned calibration that realizes the physical 12-um target"
        ),
    )
    parser.add_argument("--beam-x-um", type=float, default=0.0)
    parser.add_argument("--beam-y-um", type=float, default=0.0)
    parser.add_argument("--incident-reference")
    parser.add_argument(
        "--execution-contract",
        choices=("production", "diagnostic-smoke", "edge-isolation-smoke"),
        default="production",
        help=(
            "production uses the fixed paper-like scalar-Gaussian scenario "
            "with an explicitly assumed waist; diagnostic-smoke "
            "is a separately labeled reduced-cost engine/material/Q check; "
            "edge-isolation-smoke is the approved w0=2 um planar/edge "
            "observable-Q diagnostic"
        ),
    )
    parser.add_argument("--gpu-device", default="GPU 2")
    parser.add_argument("--threads", default="8")
    parser.add_argument(
        "--epsilon-c-model",
        choices=("paper-b-closure", "legacy-lossless-16"),
        default="paper-b-closure",
        help=(
            "production paper-consistent epsilon_c=epsilon_b closure or the "
            "historical lossless epsilon_c=16 diagnostic"
        ),
    )
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    reduced_smoke = args.execution_contract in (
        "diagnostic-smoke",
        "edge-isolation-smoke",
    )
    if args.source_object_waist_um is None:
        args.source_object_waist_um = (
            source_contract.CALIBRATED_SOURCE_OBJECT_W0_M * 1.0e6
            if not reduced_smoke
            else args.waist_um
        )
    minimum_domain_um = 60.0 if not reduced_smoke else 10.0
    if args.domain_um < minimum_domain_um:
        parser.error(
            f"{args.execution_contract} optical domain must be at least "
            f"{minimum_domain_um:g} um"
        )
    if args.source_span_um >= args.domain_um - 2.0:
        parser.error("source aperture needs at least 1 um PML clearance per side")
    if (
        args.execution_contract == "production"
        and not (
            np.isclose(args.domain_um, 60.0)
            and np.isclose(args.source_span_um, 50.0)
            and np.isclose(args.waist_um, 12.0)
            and np.isclose(
                args.source_object_waist_um,
                source_contract.CALIBRATED_SOURCE_OBJECT_W0_M * 1.0e6,
            )
        )
    ):
        parser.error(
            "production scalar source contract is fixed at domain=60 um, "
            "source span=50 um, physical target w0=12 um, and the "
            "SHA-pinned calibrated Lumerical source-object input"
        )
    if (
        args.waist_um <= 0
        or args.source_object_waist_um <= 0
        or args.flake_dz_nm <= 0
        or args.local_xy_mesh_nm <= 0
        or args.simulation_time_ps <= 0
        or args.auto_shutoff_min <= 0
    ):
        parser.error("waist, flake dz, simulation time, and shutoff must be positive")
    args.outer_local_xy_mesh_nm = (
        100.0
        if args.refinement_half_span_um is not None
        else args.local_xy_mesh_nm
    )
    if args.refinement_half_span_um is not None:
        if args.refinement_half_span_um <= 0.0:
            parser.error("refinement half span must be positive")
        if args.local_xy_mesh_nm >= args.outer_local_xy_mesh_nm:
            parser.error(
                "nested refinement requires a fine local x/y mesh below "
                "the fixed 100-nm outer mesh"
            )
    if (
        args.execution_contract == "production"
        and args.case == "finite-flake"
        and not args.incident_reference
    ):
        parser.error("finite-flake requires a matching empty-stack reference")
    if (
        reduced_smoke
        and args.case != "finite-flake"
    ):
        parser.error("reduced smoke contracts are defined only for finite-flake")
    if (
        args.execution_contract == "diagnostic-smoke"
        and args.simulation_time_ps < 4.0
    ):
        parser.error(
            "diagnostic rerun requires at least 4 ps to permit source decay"
        )
    if (
        reduced_smoke
        and args.auto_shutoff_min > 1.0e-5
    ):
        parser.error(
            "diagnostic rerun requires auto-shutoff min <= 1e-5"
        )
    args.polarization_deg = 90.0 if args.polarization == "a" else 0.0
    args.material_name = (
        PRODUCTION_MATERIAL_NAME
        if args.epsilon_c_model == "paper-b-closure"
        else LEGACY_MATERIAL_NAME
    )
    args.design_radius_um = 0.0
    args.require_design_inside_flake = False
    if args.geometry == "device-a-polygon":
        args.flake_vertices_um = np.array(FLAKE_VERTICES_UM, copy=True)
        absorption_bounds_um = {
            "x": (
                float(np.min(args.flake_vertices_um[:, 0])) - 0.05,
                float(np.max(args.flake_vertices_um[:, 0])) + 0.05,
            ),
            "y": (
                float(np.min(args.flake_vertices_um[:, 1])) - 0.05,
                float(np.max(args.flake_vertices_um[:, 1])) + 0.05,
            ),
        }
    elif args.geometry == "straight-45-edge":
        # TaIrTe4 occupies y <= x.  The other two triangle faces remain 1 um
        # beyond the FDTD x/y bounds, so only one straight 45-degree edge is
        # present in the physical calculation region.
        outer = 0.5 * args.domain_um + 1.0
        args.flake_vertices_um = np.asarray(
            [[-outer, -outer], [outer, -outer], [outer, outer]],
            dtype=float,
        )
        # The fixed 60/50-um production contract keeps 2 um analysis padding.
        # The separate smoke contract
        # uses 1.5 um only to exercise engine/material/Q/closure at lower cost.
        analysis_padding_um = (
            2.0 if args.execution_contract == "production" else 1.5
        )
        half_analysis = 0.5 * args.source_span_um + analysis_padding_um
        absorption_bounds_um = {
            "x": (
                args.beam_x_um - half_analysis,
                args.beam_x_um + half_analysis,
            ),
            "y": (
                args.beam_y_um - half_analysis,
                args.beam_y_um + half_analysis,
            ),
        }
    else:
        # The layer is deliberately continued beyond every lateral PML
        # interface to remove a physical TaIrTe4 edge from the calculation.
        outer = 0.5 * args.domain_um + 1.0
        args.flake_vertices_um = np.asarray(
            [
                [-outer, -outer],
                [outer, -outer],
                [outer, outer],
                [-outer, outer],
            ],
            dtype=float,
        )
        analysis_padding_um = (
            2.0 if args.execution_contract == "production" else 1.5
        )
        half_analysis = 0.5 * args.source_span_um + analysis_padding_um
        absorption_bounds_um = {
            "x": (
                args.beam_x_um - half_analysis,
                args.beam_x_um + half_analysis,
            ),
            "y": (
                args.beam_y_um - half_analysis,
                args.beam_y_um + half_analysis,
            ),
        }
    args.absorption_bounds_m = {
        axis: tuple(value * 1e-6 for value in bounds)
        for axis, bounds in absorption_bounds_um.items()
    }
    args.absorption_bounds_m["z"] = (-FLAKE_THICKNESS_M, 0.0)
    if args.refinement_half_span_um is not None:
        outer_mesh_half_span_um = max(
            max(abs(value) for value in absorption_bounds_um["x"]),
            max(abs(value) for value in absorption_bounds_um["y"]),
        ) + 0.5
        if args.refinement_half_span_um >= outer_mesh_half_span_um:
            parser.error(
                "nested refinement half span must be smaller than the "
                f"{outer_mesh_half_span_um:g}-um outer mesh half span"
            )
    if reduced_smoke:
        # Use one nominal control volume for both pabs_adv and all six power
        # faces.  Post-run gates independently read the realized coordinates
        # and close the volume quadrature on the realized face locations.
        args.inner_box = {
            axis: (
                args.absorption_bounds_m[axis][0] - PABS_PADDING_M,
                args.absorption_bounds_m[axis][1] + PABS_PADDING_M,
            )
            for axis in "xyz"
        }
    else:
        flux_padding_m = 0.5e-6
        args.inner_box = {
            "x": (
                args.absorption_bounds_m["x"][0] - flux_padding_m,
                args.absorption_bounds_m["x"][1] + flux_padding_m,
            ),
            "y": (
                args.absorption_bounds_m["y"][0] - flux_padding_m,
                args.absorption_bounds_m["y"][1] + flux_padding_m,
            ),
            "z": (-1.2e-6, SOURCE_Z_M - 0.5e-6),
        }
    inner_limit_um = max(
        abs(value) * 1e6
        for axis in ("x", "y")
        for value in args.inner_box[axis]
    )
    if inner_limit_um >= 0.5 * args.domain_um - 0.5:
        parser.error(
            "absorption six-face box needs at least 0.5 um nominal "
            "clearance from the lateral domain boundary"
        )
    return args


def load_base() -> Any:
    for path in (STAGE1, REPOSITORY / "photothermal_pte", REPOSITORY / "photothermal_pte" / "bundle"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("paper_ir_finite_q_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import base validator: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def complex_json(value: complex) -> dict[str, float]:
    return {"real": float(np.real(value)), "imag": float(np.imag(value))}


def add_device_a_material(
    fdtd: Any,
    model: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    wavelengths_nm = np.linspace(2700.0, 13200.0, 600)
    frequencies_hz = C0 / (wavelengths_nm * 1e-9)
    eps_a = model.eps_flake(wavelengths_nm, "a")
    eps_b = model.eps_flake(wavelengths_nm, "b")
    eps_c = (
        model.eps_flake(wavelengths_nm, "c")
        if args.epsilon_c_model == "paper-b-closure"
        else np.full_like(eps_a, model.eps_c_flake_legacy_diagnostic)
    )
    material = fdtd.addmaterial("Sampled 3D data")
    fdtd.setmaterial(material, "name", args.material_name)
    fdtd.setmaterial(args.material_name, "anisotropy", 1)
    fdtd.setmaterial(args.material_name, "max coefficients", model.MAX_COEFFS)
    # Sampled-3D columns follow lab x,y,z.  Paper image: x=b and y=a.
    fdtd.setmaterial(
        args.material_name,
        "sampled data",
        np.column_stack((frequencies_hz, eps_b, eps_a, eps_c)),
    )
    target_nm = WAVELENGTH_M * 1e9
    requested = {
        "x": complex(model.eps_flake(target_nm, "b")),
        "y": complex(model.eps_flake(target_nm, "a")),
        "z": complex(
            model.eps_flake(target_nm, "c")
            if args.epsilon_c_model == "paper-b-closure"
            else model.eps_c_flake_legacy_diagnostic
        ),
    }
    return {
        "name": args.material_name,
        "epsilon_c_model": args.epsilon_c_model,
        "axis_mapping": {
            "x": "epsilon_b",
            "y": "epsilon_a",
            "z": (
                "epsilon_c=epsilon_b paper-consistent 3D closure"
                if args.epsilon_c_model == "paper-b-closure"
                else "legacy lossless epsilon_c=16 diagnostic"
            ),
        },
        "requested_epsilon_at_11um": {
            axis: complex_json(value)
            for axis, value in requested.items()
        },
        "provenance": {
            "in_plane": (
                "paper Supporting Information Fig. S3b / Ref. 11, stored "
                "in paper-derived perm_data.txt"
            ),
            "out_of_plane": (
                "explicit finite-edge 3D closure epsilon_c=epsilon_b; not "
                "an independently measured c-axis property"
                if args.epsilon_c_model == "paper-b-closure"
                else "historical repository constant; diagnostic only"
            ),
            "sample_count": int(wavelengths_nm.size),
            "wavelength_range_m": [
                float(wavelengths_nm[0] * 1e-9),
                float(wavelengths_nm[-1] * 1e-9),
            ],
        },
    }


def strict_gpu_run(fdtd: Any, run_name: str) -> str:
    """Run only a GPU resource; never attempt a CPU resource."""
    errors: list[str] = []
    # The third argument is a session resource *name*.  Some v261 installs call
    # their GPU resource "Local Host" or "Local Computer".  The second argument
    # remains "GPU" in every attempt, so these are aliases, not CPU fallbacks.
    for resource in (
        "Local GPU",
        "local GPU",
        "Local Host",
        "localhost",
        "Local Computer",
    ):
        try:
            print(f"[gpu-only] {run_name} on {resource!r}", flush=True)
            fdtd.run("FDTD", "GPU", resource)
            return resource
        except Exception as exc:
            errors.append(f"{resource}: {type(exc).__name__}: {exc}")
    raise RuntimeError("GPU-only FDTD failed; CPU fallback prohibited: " + " | ".join(errors))


def material_epsilon_readback(
    fdtd: Any,
    args: argparse.Namespace,
    requested_contract: dict[str, Any],
    *,
    dt_s: float,
) -> dict[str, Any]:
    frequency_hz = C0 / WAVELENGTH_M
    frequency_ends = C0 / np.asarray(
        [SOURCE_START_M, SOURCE_STOP_M],
        float,
    )
    fmin = float(np.min(frequency_ends))
    fmax = float(np.max(frequency_ends))
    axes: dict[str, Any] = {}
    for component_index, axis in zip((1, 2, 3), "xyz"):
        requested_json = requested_contract[
            "requested_epsilon_at_11um"
        ][axis]
        requested = complex(
            requested_json["real"],
            requested_json["imag"],
        )
        fitted_n = np.asarray(
            fdtd.getfdtdindex(
                args.material_name,
                np.asarray([frequency_hz]),
                fmin,
                fmax,
                component_index,
            )
        ).reshape(-1)[0]
        fitted = complex(fitted_n) ** 2
        numerical = complex(
            np.asarray(
                fdtd.getnumericalpermittivity(
                    args.material_name,
                    np.asarray([frequency_hz]),
                    fmin,
                    fmax,
                    dt_s,
                    component_index,
                )
            ).reshape(-1)[0]
        )
        axes[axis] = {
            "requested_epsilon": complex_json(requested),
            "fitted_epsilon_getfdtdindex": complex_json(fitted),
            "finite_dt_numerical_permittivity": complex_json(numerical),
            "fitted_relative_error_vs_requested": float(
                abs(fitted - requested)
                / max(abs(requested), np.finfo(float).tiny)
            ),
            "finite_dt_relative_difference_vs_fitted": float(
                abs(numerical - fitted)
                / max(abs(fitted), np.finfo(float).tiny)
            ),
        }
    fitted_x = complex(
        axes["x"]["fitted_epsilon_getfdtdindex"]["real"],
        axes["x"]["fitted_epsilon_getfdtdindex"]["imag"],
    )
    fitted_z = complex(
        axes["z"]["fitted_epsilon_getfdtdindex"]["real"],
        axes["z"]["fitted_epsilon_getfdtdindex"]["imag"],
    )
    numerical_x = complex(
        axes["x"]["finite_dt_numerical_permittivity"]["real"],
        axes["x"]["finite_dt_numerical_permittivity"]["imag"],
    )
    numerical_z = complex(
        axes["z"]["finite_dt_numerical_permittivity"]["real"],
        axes["z"]["finite_dt_numerical_permittivity"]["imag"],
    )
    return {
        "quantity": "complex relative permittivity epsilon_r",
        "analysis_wavelength_m": WAVELENGTH_M,
        "fit_frequency_range_Hz": [fmin, fmax],
        "dt_s": dt_s,
        "axes": axes,
        "epsilon_z_equals_epsilon_b_contract": (
            args.epsilon_c_model == "paper-b-closure"
        ),
        "fitted_z_vs_x_relative_difference": float(
            abs(fitted_z - fitted_x)
            / max(abs(fitted_x), np.finfo(float).tiny)
        ),
        "finite_dt_z_vs_x_relative_difference": float(
            abs(numerical_z - numerical_x)
            / max(abs(numerical_x), np.finfo(float).tiny)
        ),
    }


def _mesh_coordinate(fdtd: Any, axis: str) -> np.ndarray:
    raw = fdtd.getresult("FDTD", axis)
    if isinstance(raw, dict):
        raw = raw[axis]
    coordinate = np.asarray(raw, float).reshape(-1)
    if coordinate.size < 3 or np.any(np.diff(coordinate) <= 0.0):
        raise RuntimeError(f"invalid native FDTD {axis} mesh coordinate")
    return coordinate


def _coordinate_summary(values: np.ndarray) -> dict[str, Any]:
    coordinate = np.asarray(values, float).reshape(-1)
    steps = np.diff(coordinate)
    return {
        "shape": [int(coordinate.size)],
        "bounds_m": [float(coordinate[0]), float(coordinate[-1])],
        "minimum_step_m": float(np.min(steps)),
        "median_step_m": float(np.median(steps)),
        "maximum_step_m": float(np.max(steps)),
    }


def _monitor_coordinate(
    fdtd: Any,
    monitor: str,
    axis: str,
) -> np.ndarray:
    coordinate = np.asarray(
        fdtd.getdata(monitor, axis, 1),
        float,
    ).reshape(-1)
    if coordinate.size == 0 or np.any(~np.isfinite(coordinate)):
        raise RuntimeError(
            f"invalid {monitor}.{axis} realized coordinate"
        )
    if coordinate.size > 1 and np.any(np.diff(coordinate) <= 0.0):
        raise RuntimeError(
            f"non-increasing {monitor}.{axis} realized coordinate"
        )
    return coordinate


def bounded_dual_cell_weights(
    coordinate: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    """Intersect every sample's dual cell with realized control-volume faces."""
    values = np.asarray(coordinate, float).reshape(-1)
    if (
        values.size == 0
        or not np.isfinite(low)
        or not np.isfinite(high)
        or high <= low
        or np.any(~np.isfinite(values))
        or (values.size > 1 and np.any(np.diff(values) <= 0.0))
    ):
        raise RuntimeError("invalid bounded dual-cell coordinate contract")
    if values.size == 1:
        if values[0] < low or values[0] > high:
            raise RuntimeError(
                "single Q sample does not cover the realized flux volume"
            )
        return np.asarray([high - low], float)
    midpoints = 0.5 * (values[:-1] + values[1:])
    edges = np.concatenate(
        (
            [values[0] - 0.5 * (values[1] - values[0])],
            midpoints,
            [values[-1] + 0.5 * (values[-1] - values[-2])],
        )
    )
    weights = np.maximum(
        0.0,
        np.minimum(edges[1:], high) - np.maximum(edges[:-1], low),
    )
    if not np.any(weights > 0.0):
        raise RuntimeError(
            "Q sampling support does not intersect the realized flux volume"
        )
    if not np.isclose(
        float(np.sum(weights)),
        high - low,
        rtol=1.0e-13,
        atol=1.0e-18,
    ):
        raise RuntimeError("bounded dual-cell weights do not close")
    return weights


def integrate_xyz_bounded(
    values: np.ndarray,
    coordinates: dict[str, np.ndarray],
    bounds: dict[str, tuple[float, float] | list[float]],
) -> float:
    weights = {
        axis: bounded_dual_cell_weights(
            coordinates[axis],
            float(bounds[axis][0]),
            float(bounds[axis][1]),
        )
        for axis in "xyz"
    }
    return float(
        np.einsum(
            "i,j,k,ijk->",
            weights["x"],
            weights["y"],
            weights["z"],
            np.asarray(values, float),
            optimize=True,
        )
    )


def native_component_absorption_bounded(
    base: Any,
    fdtd: Any,
    frequency_hz: float,
    bounds: dict[str, tuple[float, float] | list[float]],
) -> dict[str, Any]:
    field_common = {
        axis: _monitor_coordinate(fdtd, base.PABS_FIELD, axis)
        for axis in "xyz"
    }
    index_common = {
        axis: _monitor_coordinate(fdtd, base.PABS_INDEX, axis)
        for axis in "xyz"
    }
    field_delta = {
        axis: np.asarray(
            fdtd.getdata(base.PABS_FIELD, f"delta_{axis}", 1),
            float,
        ).reshape(-1)
        for axis in "xyz"
    }
    index_delta = {
        axis: np.asarray(
            fdtd.getdata(base.PABS_INDEX, f"delta_{axis}", 1),
            float,
        ).reshape(-1)
        for axis in "xyz"
    }
    omega = 2.0 * np.pi * frequency_hz
    component_power: dict[str, float] = {}
    pairing: dict[str, Any] = {}
    for component in "xyz":
        electric = np.asarray(
            fdtd.getdata(base.PABS_FIELD, f"E{component}", 1)
        ).squeeze()
        epsilon = (
            np.asarray(
                fdtd.getdata(
                    base.PABS_INDEX,
                    f"index_{component}",
                    1,
                )
            ).squeeze()
            ** 2
        )
        field_coordinates = {
            axis: np.array(field_common[axis], copy=True)
            for axis in "xyz"
        }
        index_coordinates = {
            axis: np.array(index_common[axis], copy=True)
            for axis in "xyz"
        }
        field_coordinates[component] += field_delta[component]
        index_coordinates[component] += index_delta[component]
        if electric.shape != epsilon.shape:
            raise RuntimeError(
                f"E/index shape mismatch for {component}: "
                f"{electric.shape} != {epsilon.shape}"
            )
        coordinate_mismatch = {}
        for axis in "xyz":
            if (
                field_coordinates[axis].shape
                != index_coordinates[axis].shape
            ):
                raise RuntimeError(
                    f"E/index {component}:{axis} coordinate shape mismatch"
                )
            coordinate_mismatch[axis] = float(
                np.max(
                    np.abs(
                        field_coordinates[axis]
                        - index_coordinates[axis]
                    )
                )
            )
        maximum_mismatch = max(coordinate_mismatch.values())
        if maximum_mismatch > 1.0e-15:
            raise RuntimeError(
                "independent E/index coordinate mismatch exceeds 1 fm: "
                f"{component}={maximum_mismatch}"
            )
        q_native = (
            0.5
            * base.EPS0
            * omega
            * np.abs(electric) ** 2
            * np.imag(epsilon)
        )
        component_power[component] = integrate_xyz_bounded(
            q_native,
            field_coordinates,
            bounds,
        )
        pairing[component] = {
            "field_shape": list(electric.shape),
            "index_shape": list(epsilon.shape),
            "coordinate_mismatch_m": coordinate_mismatch,
            "maximum_coordinate_mismatch_m": maximum_mismatch,
            "field_coordinates_read_from": base.PABS_FIELD,
            "index_coordinates_read_from": base.PABS_INDEX,
            "field_delta_read_independently": True,
            "index_delta_read_independently": True,
        }
    return {
        "component_power_W": component_power,
        "total_power_W": float(sum(component_power.values())),
        "independent_field_index_pairing": pairing,
        "maximum_coordinate_mismatch_m": max(
            value["maximum_coordinate_mismatch_m"]
            for value in pairing.values()
        ),
    }


def final_logged_auto_shutoff(
    output: Path,
) -> dict[str, Any]:
    logs = sorted(output.glob("*_p0.log"))
    if not logs:
        raise RuntimeError("FDTD engine log missing after completed run")
    log_path = logs[-1]
    text = log_path.read_text(encoding="utf-8", errors="replace")
    values = re.findall(r"Auto Shutoff: ([0-9.eE+-]+)", text)
    if not values:
        raise RuntimeError("FDTD engine log has no auto-shutoff samples")
    return {
        "log_path": str(log_path.resolve()),
        "final_value": float(values[-1]),
        "simulation_completed_successfully": (
            "Simulation completed successfully" in text
        ),
    }


def realized_six_face_control_volume(
    fdtd: Any,
    faces: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    coordinates = {
        key: {
            axis: _monitor_coordinate(fdtd, face["name"], axis)
            for axis in "xyz"
        }
        for key, face in faces.items()
    }
    bounds = {
        axis: [
            float(coordinates[f"{axis}_min"][axis][0]),
            float(coordinates[f"{axis}_max"][axis][0]),
        ]
        for axis in "xyz"
    }
    transverse_mismatches: dict[str, float] = {}
    for face_key, face_coordinates in coordinates.items():
        normal_axis = face_key[0]
        for axis in "xyz":
            if axis == normal_axis:
                continue
            mismatch = max(
                abs(float(face_coordinates[axis][0]) - bounds[axis][0]),
                abs(float(face_coordinates[axis][-1]) - bounds[axis][1]),
            )
            transverse_mismatches[f"{face_key}:{axis}"] = mismatch
    return {
        "bounds_m": bounds,
        "face_coordinates": {
            face: {
                axis: _coordinate_summary(values)
                if values.size > 1
                else {
                    "shape": [1],
                    "bounds_m": [float(values[0]), float(values[0])],
                }
                for axis, values in axes.items()
            }
            for face, axes in coordinates.items()
        },
        "maximum_transverse_boundary_mismatch_m": float(
            max(transverse_mismatches.values(), default=0.0)
        ),
        "transverse_boundary_mismatch_m": transverse_mismatches,
    }


def post_run_native_mesh_audit(
    base: Any,
    fdtd: Any,
    output: Path,
    run_result: dict[str, Any],
    faces: dict[str, dict[str, Any]],
    q_quadrature_bounds: dict[
        str, tuple[float, float] | list[float]
    ],
) -> dict[str, Any]:
    solver = {axis: _mesh_coordinate(fdtd, axis) for axis in "xyz"}
    field_common = {
        axis: np.asarray(
            fdtd.getdata(base.PABS_FIELD, axis, 1),
            float,
        ).reshape(-1)
        for axis in "xyz"
    }
    index_common = {
        axis: np.asarray(
            fdtd.getdata(base.PABS_INDEX, axis, 1),
            float,
        ).reshape(-1)
        for axis in "xyz"
    }
    field_delta = {
        axis: np.asarray(
            fdtd.getdata(base.PABS_FIELD, f"delta_{axis}", 1),
            float,
        ).reshape(-1)
        for axis in "xyz"
    }
    index_delta = {
        axis: np.asarray(
            fdtd.getdata(base.PABS_INDEX, f"delta_{axis}", 1),
            float,
        ).reshape(-1)
        for axis in "xyz"
    }
    realized_control_volume = realized_six_face_control_volume(
        fdtd,
        faces,
    )
    realized_bounds = realized_control_volume["bounds_m"]
    common_Q_bounded_weights = {
        axis: bounded_dual_cell_weights(
            field_common[axis],
            float(q_quadrature_bounds[axis][0]),
            float(q_quadrature_bounds[axis][1]),
        )
        for axis in "xyz"
    }
    common_Q_support = {
        axis: {
            "sample_bounds_m": [
                float(field_common[axis][0]),
                float(field_common[axis][-1]),
            ],
            "Q_quadrature_bounds_m": q_quadrature_bounds[axis],
            "low_bound_to_first_sample_m": float(
                field_common[axis][0] - q_quadrature_bounds[axis][0]
            ),
            "last_sample_to_high_bound_m": float(
                q_quadrature_bounds[axis][1] - field_common[axis][-1]
            ),
            "bounded_weight_sum_m": float(
                np.sum(common_Q_bounded_weights[axis])
            ),
            "Q_quadrature_span_m": float(
                q_quadrature_bounds[axis][1]
                - q_quadrature_bounds[axis][0]
            ),
        }
        for axis in "xyz"
    }
    field_component_coordinates: dict[
        str, dict[str, np.ndarray]
    ] = {}
    index_component_coordinates: dict[
        str, dict[str, np.ndarray]
    ] = {}
    for component in "xyz":
        field_coordinates = {
            axis: np.array(field_common[axis], copy=True)
            for axis in "xyz"
        }
        index_coordinates = {
            axis: np.array(index_common[axis], copy=True)
            for axis in "xyz"
        }
        field_coordinates[component] = (
            field_coordinates[component] + field_delta[component]
        )
        index_coordinates[component] = (
            index_coordinates[component] + index_delta[component]
        )
        field_component_coordinates[component] = field_coordinates
        index_component_coordinates[component] = index_coordinates

    artifact_path = output / "native_yee_mesh_coordinates.npz"
    np.savez_compressed(
        artifact_path,
        **{f"solver_{axis}_m": values for axis, values in solver.items()},
        **{
            f"field_common_{axis}_m": values
            for axis, values in field_common.items()
        },
        **{
            f"index_common_{axis}_m": values
            for axis, values in index_common.items()
        },
        **{
            f"field_delta_{axis}_m": values
            for axis, values in field_delta.items()
        },
        **{
            f"index_delta_{axis}_m": values
            for axis, values in index_delta.items()
        },
        **{
            f"realized_control_volume_{axis}_bounds_m": np.asarray(
                bounds,
                float,
            )
            for axis, bounds in realized_bounds.items()
        },
        **{
            f"common_Q_bounded_{axis}_weights_m": values
            for axis, values in common_Q_bounded_weights.items()
        },
        **{
            f"E{component}_{axis}_m": values
            for component, coordinates in (
                field_component_coordinates.items()
            )
            for axis, values in coordinates.items()
        },
        **{
            f"index_{component}_{axis}_m": values
            for component, coordinates in (
                index_component_coordinates.items()
            )
            for axis, values in coordinates.items()
        },
    )

    hotspot = run_result.get("Q_hotspot", {})
    local_step: dict[str, Any] = {}
    edge_beam_step: dict[str, Any] = {}
    for axis in "xyz":
        coordinate = solver[axis]
        centre = float(hotspot.get(f"{axis}_m", 0.0))
        midpoint = 0.5 * (coordinate[:-1] + coordinate[1:])
        half_window = 2.0e-6 if axis in "xy" else 0.5e-6
        selected = np.abs(midpoint - centre) <= half_window
        steps = np.diff(coordinate)[selected]
        local_step[axis] = {
            "window_center_m": centre,
            "window_half_width_m": half_window,
            "minimum_step_m": float(np.min(steps)),
            "maximum_step_m": float(np.max(steps)),
            "step_count": int(steps.size),
        }
        edge_beam_centre = 0.0 if axis in "xy" else -0.5 * FLAKE_THICKNESS_M
        edge_beam_half_window = 2.0e-6 if axis in "xy" else 0.5e-6
        edge_beam_selected = (
            np.abs(midpoint - edge_beam_centre) <= edge_beam_half_window
        )
        edge_beam_steps = np.diff(coordinate)[edge_beam_selected]
        edge_beam_step[axis] = {
            "window_center_m": edge_beam_centre,
            "window_half_width_m": edge_beam_half_window,
            "minimum_step_m": float(np.min(edge_beam_steps)),
            "median_step_m": float(np.median(edge_beam_steps)),
            "maximum_step_m": float(np.max(edge_beam_steps)),
            "step_count": int(edge_beam_steps.size),
        }

    component_summary: dict[str, Any] = {}
    independent_pairing_maximum_mismatch = 0.0
    for component, field_coordinates in (
        field_component_coordinates.items()
    ):
        index_coordinates = index_component_coordinates[component]
        electric_shape = list(
            np.asarray(
                fdtd.getdata(base.PABS_FIELD, f"E{component}", 1)
            ).squeeze().shape
        )
        index_shape = list(
            np.asarray(
                fdtd.getdata(base.PABS_INDEX, f"index_{component}", 1)
            ).squeeze().shape
        )
        coordinate_mismatch = {
            axis: (
                float(
                    np.max(
                        np.abs(
                            field_coordinates[axis]
                            - index_coordinates[axis]
                        )
                    )
                )
                if (
                    field_coordinates[axis].shape
                    == index_coordinates[axis].shape
                )
                else None
            )
            for axis in "xyz"
        }
        finite_mismatch = [
            value
            for value in coordinate_mismatch.values()
            if value is not None
        ]
        component_maximum_mismatch = (
            max(finite_mismatch) if finite_mismatch else float("inf")
        )
        independent_pairing_maximum_mismatch = max(
            independent_pairing_maximum_mismatch,
            component_maximum_mismatch,
        )
        component_summary[component] = {
            "field_coordinates": {
                axis: _coordinate_summary(values)
                for axis, values in field_coordinates.items()
            },
            "permittivity_coordinates": {
                axis: _coordinate_summary(values)
                for axis, values in index_coordinates.items()
            },
            "field_delta_read_independently": True,
            "permittivity_delta_read_independently": True,
            "electric_field_shape": electric_shape,
            "permittivity_shape": index_shape,
            "field_permittivity_shape_pairing_exact": (
                electric_shape == index_shape
            ),
            "field_permittivity_coordinate_mismatch_m": (
                coordinate_mismatch
            ),
            "field_permittivity_maximum_coordinate_mismatch_m": (
                component_maximum_mismatch
            ),
            "staggering_axis": component,
            "maximum_coordinate_mismatch_from_common_m": {
                axis: float(
                    np.max(np.abs(values - field_common[axis]))
                )
                for axis, values in field_coordinates.items()
            },
        }
    return {
        "native_solver_mesh": {
            axis: _coordinate_summary(values)
            for axis, values in solver.items()
        },
        "native_solver_coordinate_counts": {
            axis: int(values.size) for axis, values in solver.items()
        },
        "native_solver_gridpoint_product": int(
            np.prod([values.size for values in solver.values()])
        ),
        "native_Yee_cell_count": int(
            np.prod([max(values.size - 1, 0) for values in solver.values()])
        ),
        "hotspot_neighbourhood_solver_steps": local_step,
        "edge_beam_neighbourhood_solver_steps": edge_beam_step,
        "component_specific_Yee_coordinates": component_summary,
        "independent_field_index_pairing": {
            "method": (
                "field-detail and index-detail x/y/z plus delta_x/delta_y/"
                "delta_z are read separately; no field coordinate is copied "
                "into the permittivity contract"
            ),
            "maximum_coordinate_mismatch_m": (
                independent_pairing_maximum_mismatch
            ),
            "all_component_shapes_match": all(
                value[
                    "field_permittivity_shape_pairing_exact"
                ]
                for value in component_summary.values()
            ),
        },
        "index_monitor_common_coordinates": {
            axis: _coordinate_summary(values)
            for axis, values in index_common.items()
        },
        "common_Q_output_coordinates": {
            axis: _coordinate_summary(values)
            for axis, values in field_common.items()
        },
        "realized_six_face_control_volume": realized_control_volume,
        "Q_quadrature_control_volume_bounds_m": q_quadrature_bounds,
        "common_Q_support_in_Q_quadrature_control_volume": (
            common_Q_support
        ),
        "common_Q_quadrature_contract": (
            "dual-cell weights use the pabs/Q analysis control-volume "
            "bounds; in matched diagnostic cases these are identical to "
            "the independently read six-face bounds, while production "
            "uses a larger flux box around the thin lossy-Q volume"
        ),
        "component_to_common_Q_contract": (
            "pabs_adv E/index component loss is evaluated on each native "
            "component grid x+delta_x, y+delta_y, or z+delta_z and then "
            "linearly interpolated to the field-monitor common x/y/z grid"
        ),
        "mesh_override_contract": {
            "x": (
                "local flake/illuminated-region override plus automatic "
                "nonuniform mesh outside"
            ),
            "y": (
                "local flake/illuminated-region override plus automatic "
                "nonuniform mesh outside"
            ),
            "z": "flake-region override plus automatic nonuniform mesh outside",
        },
        "coordinate_artifact": str(artifact_path.resolve()),
        "coordinate_artifact_size_bytes": artifact_path.stat().st_size,
        "not_claimed": (
            "the common Q spacing is not called the native solver mesh spacing"
        ),
    }


def add_geometry_and_monitors(
    base: Any, fdtd: Any, model: Any, args: argparse.Namespace
) -> dict[str, Any]:
    vertices_um = np.asarray(args.flake_vertices_um, float)
    absorption_bounds = args.absorption_bounds_m
    inner_box = args.inner_box
    domain_m = args.domain_um * 1e-6
    half_source = 0.5 * args.source_span_um * 1e-6
    beam_x = args.beam_x_um * 1e-6
    beam_y = args.beam_y_um * 1e-6
    source_bounds = {
        "x": (beam_x - half_source, beam_x + half_source),
        "y": (beam_y - half_source, beam_y + half_source),
        "z": SOURCE_Z_M,
    }
    for axis in ("x", "y"):
        if max(abs(v) for v in source_bounds[axis]) >= 0.5 * domain_m - 0.5e-6:
            raise RuntimeError(f"{axis} source aperture is too close to PML")
    outer_bounds = {
        "x": (-0.5 * domain_m + 1.0e-6, 0.5 * domain_m - 1.0e-6),
        "y": (-0.5 * domain_m + 1.0e-6, 0.5 * domain_m - 1.0e-6),
        "z": (-1.2e-6, SOURCE_Z_M - 0.5e-6),
    }

    fdtd.addfdtd()
    for prop, value in (
        ("dimension", "3D"),
        ("x", 0.0),
        ("x span", domain_m),
        ("y", 0.0),
        ("y span", domain_m),
        ("z min", FDTD_Z_MIN_M),
        ("z max", FDTD_Z_MAX_M),
        ("pml layers", int(args.pml_layers)),
        ("mesh type", "auto non-uniform"),
        ("mesh refinement", "conformal variant 1"),
        ("mesh accuracy", 5),
        ("simulation time", args.simulation_time_ps * 1e-12),
        ("auto shutoff min", args.auto_shutoff_min),
        ("min mesh step", 1.0e-9),
    ):
        base.safe_set(fdtd, "FDTD", prop, value)
    for axis in "xyz":
        base.safe_set(fdtd, "FDTD", f"{axis} min bc", "PML")
        base.safe_set(fdtd, "FDTD", f"{axis} max bc", "PML")

    material = fdtd.addmaterial("Dielectric")
    fdtd.setmaterial(material, "name", SIO2_MATERIAL)
    fdtd.setmaterial(SIO2_MATERIAL, "Refractive Index", 1.38)
    material_contract = add_device_a_material(fdtd, model, args)

    lateral_material_span = domain_m + 2.0e-6
    base.add_rect(
        fdtd,
        "Si_substrate",
        {
            "x": (-0.5 * lateral_material_span, 0.5 * lateral_material_span),
            "y": (-0.5 * lateral_material_span, 0.5 * lateral_material_span),
            "z": (
                FDTD_Z_MIN_M - 0.5e-6,
                -FLAKE_THICKNESS_M - SIO2_THICKNESS_M,
            ),
        },
        index=3.425,
    )
    base.add_rect(
        fdtd,
        "SiO2_spacer",
        {
            "x": (-0.5 * lateral_material_span, 0.5 * lateral_material_span),
            "y": (-0.5 * lateral_material_span, 0.5 * lateral_material_span),
            "z": (
                -FLAKE_THICKNESS_M - SIO2_THICKNESS_M,
                -FLAKE_THICKNESS_M,
            ),
        },
        material=SIO2_MATERIAL,
    )
    if args.case == "finite-flake":
        if args.geometry == "planar-stack":
            base.add_rect(
                fdtd,
                "TaIrTe4_flake",
                {
                    "x": (
                        -0.5 * lateral_material_span,
                        0.5 * lateral_material_span,
                    ),
                    "y": (
                        -0.5 * lateral_material_span,
                        0.5 * lateral_material_span,
                    ),
                    "z": (-FLAKE_THICKNESS_M, 0.0),
                },
                material=args.material_name,
            )
        else:
            polygon = fdtd.addpoly()
            polygon["name"] = "TaIrTe4_flake"
            polygon["vertices"] = vertices_um * 1e-6
            polygon["z min"] = -FLAKE_THICKNESS_M
            polygon["z max"] = 0.0
            polygon["material"] = args.material_name

    mesh_z_min = -FLAKE_THICKNESS_M - 10 * args.flake_dz_nm * 1e-9
    mesh_z_max = 10 * args.flake_dz_nm * 1e-9
    outer_mesh_bounds = {
        "x": (
            absorption_bounds["x"][0] - 0.5e-6,
            absorption_bounds["x"][1] + 0.5e-6,
        ),
        "y": (
            absorption_bounds["y"][0] - 0.5e-6,
            absorption_bounds["y"][1] + 0.5e-6,
        ),
    }
    if args.refinement_half_span_um is not None:
        outer_mesh = fdtd.addmesh()
        outer_mesh["name"] = "flake_outer_mesh"
        outer_mesh["x min"], outer_mesh["x max"] = outer_mesh_bounds["x"]
        outer_mesh["y min"], outer_mesh["y max"] = outer_mesh_bounds["y"]
        outer_mesh["z min"] = mesh_z_min
        outer_mesh["z max"] = mesh_z_max
        outer_mesh["override x mesh"] = 1
        outer_mesh["override y mesh"] = 1
        outer_mesh["override z mesh"] = 1
        outer_mesh["dx"] = args.outer_local_xy_mesh_nm * 1e-9
        outer_mesh["dy"] = args.outer_local_xy_mesh_nm * 1e-9
        outer_mesh["dz"] = args.flake_dz_nm * 1e-9
        fine_half_span = args.refinement_half_span_um * 1e-6
        fine_mesh_bounds = {
            "x": (beam_x - fine_half_span, beam_x + fine_half_span),
            "y": (beam_y - fine_half_span, beam_y + fine_half_span),
        }
    else:
        fine_mesh_bounds = outer_mesh_bounds

    mesh = fdtd.addmesh()
    mesh["name"] = "flake_mesh"
    mesh["x min"], mesh["x max"] = fine_mesh_bounds["x"]
    mesh["y min"], mesh["y max"] = fine_mesh_bounds["y"]
    mesh["z min"] = mesh_z_min
    mesh["z max"] = mesh_z_max
    mesh["override x mesh"] = 1
    mesh["override y mesh"] = 1
    mesh["override z mesh"] = 1
    mesh["dx"] = args.local_xy_mesh_nm * 1e-9
    mesh["dy"] = args.local_xy_mesh_nm * 1e-9
    mesh["dz"] = args.flake_dz_nm * 1e-9

    source = fdtd.addgaussian()
    source["name"] = base.SOURCE_NAME
    source["injection axis"] = "z"
    source["direction"] = "backward"
    source["polarization angle"] = float(args.polarization_deg)
    source["source shape"] = "Gaussian"
    source["use scalar approximation"] = True
    source["beam parameters"] = "Waist size and position"
    source["waist radius w0"] = args.source_object_waist_um * 1e-6
    source["distance from waist"] = -(SOURCE_Z_M - FOCUS_Z_M)
    source["x min"], source["x max"] = source_bounds["x"]
    source["y min"], source["y max"] = source_bounds["y"]
    source["z"] = SOURCE_Z_M
    source["use global source settings"] = True
    source["override global source settings"] = False

    fdtd.setglobalsource("wavelength start", SOURCE_START_M)
    fdtd.setglobalsource("wavelength stop", SOURCE_STOP_M)
    fdtd.setglobalmonitor("use source limits", False)
    fdtd.setglobalmonitor("use wavelength spacing", True)
    fdtd.setglobalmonitor("wavelength center", WAVELENGTH_M)
    fdtd.setglobalmonitor("wavelength span", 0.0)
    fdtd.setglobalmonitor("frequency points", 1)

    pabs = fdtd.addobject("pabs_adv")
    pabs["name"] = base.PABS_GROUP
    pabs_bounds = (
        inner_box
        if args.execution_contract
        in ("diagnostic-smoke", "edge-isolation-smoke")
        else {
            axis: (
                absorption_bounds[axis][0] - PABS_PADDING_M,
                absorption_bounds[axis][1] + PABS_PADDING_M,
            )
            for axis in "xyz"
        }
    )
    for axis in "xyz":
        pabs[axis] = 0.5 * sum(pabs_bounds[axis])
        pabs[f"{axis} span"] = (
            pabs_bounds[axis][1] - pabs_bounds[axis][0]
        )

    inner_faces = base.add_flux_box(fdtd, "paper_ir_abs", inner_box)
    if args.execution_contract == "production":
        outer_faces = base.add_flux_box(
            fdtd,
            "paper_ir_outer",
            outer_bounds,
        )
        base.add_field_monitor(
            fdtd,
            base.INCIDENT_REFERENCE_MONITOR,
            "2D Z-normal",
            {
                "x": source_bounds["x"],
                "y": source_bounds["y"],
                "z": (INCIDENT_Z_M, INCIDENT_Z_M),
            },
        )
        base.add_field_monitor(
            fdtd,
            "finite_E_xy_inside",
            "2D Z-normal",
            {
                "x": inner_box["x"],
                "y": inner_box["y"],
                "z": (0.5e-6, 0.5e-6),
            },
        )
        base.add_field_monitor(
            fdtd,
            "finite_E_yz_outside_x",
            "2D X-normal",
            {
                "x": (
                    outer_bounds["x"][1],
                    outer_bounds["x"][1],
                ),
                "y": outer_bounds["y"],
                "z": outer_bounds["z"],
            },
        )
        base.add_field_monitor(
            fdtd,
            "finite_E_xz_outside_y",
            "2D Y-normal",
            {
                "x": outer_bounds["x"],
                "y": (
                    outer_bounds["y"][1],
                    outer_bounds["y"][1],
                ),
                "z": outer_bounds["z"],
            },
        )
        optional_monitor_names = (
            *(face["name"] for face in outer_faces.values()),
            base.INCIDENT_REFERENCE_MONITOR,
            "finite_E_xy_inside",
            "finite_E_yz_outside_x",
            "finite_E_xz_outside_y",
        )
    else:
        outer_faces = {}
        if args.execution_contract == "edge-isolation-smoke":
            base.add_field_monitor(
                fdtd,
                W2_INCIDENT_MONITOR,
                "2D Z-normal",
                {
                    "x": inner_box["x"],
                    "y": inner_box["y"],
                    "z": (
                        W2_INCIDENT_PLANE_Z_M,
                        W2_INCIDENT_PLANE_Z_M,
                    ),
                },
            )
            base.add_field_monitor(
                fdtd,
                W2_FLAKE_FIELD_MONITOR,
                "2D Z-normal",
                {
                    "x": inner_box["x"],
                    "y": inner_box["y"],
                    "z": (
                        W2_FLAKE_MIDPLANE_Z_M,
                        W2_FLAKE_MIDPLANE_Z_M,
                    ),
                },
            )
            optional_monitor_names = (
                W2_INCIDENT_MONITOR,
                W2_FLAKE_FIELD_MONITOR,
            )
        else:
            optional_monitor_names = ()
    for name in (
        base.PABS_FIELD,
        base.PABS_INDEX,
        *(face["name"] for face in inner_faces.values()),
        *optional_monitor_names,
    ):
        base.configure_single_frequency(fdtd, name)

    geometry = {
        "geometry_name": args.geometry,
        "geometry_source": (
            "matching empty SiO2/Si layered-stack incident reference; "
            "no TaIrTe4 object"
            if args.case == "empty-stack"
            else (
                "approximation digitized from paper Figure 2A, not exact CAD"
                if args.geometry == "device-a-polygon"
                else (
                    "paper Figure 3F straight 45-degree half-plane control; "
                    "TaIrTe4 occupies lab y<=x and remote triangle faces lie "
                    "outside the PML-bounded physical domain"
                    if args.geometry == "straight-45-edge"
                    else (
                        "edge-free planar TaIrTe4 layer extended 1 um beyond "
                        "each lateral PML-bounded domain face"
                    )
                )
            )
        ),
        "coordinate_contract": {"lab_x": "crystal b", "lab_y": "crystal a"},
        "flake_vertices_um": vertices_um.tolist(),
        "flake_thickness_m": FLAKE_THICKNESS_M,
        "flake_area_m2": (
            None
            if args.case == "empty-stack"
            else polygon_area(vertices_um) * 1e-12
        ),
        "absorption_analysis_bounds_m": absorption_bounds,
        "pabs_nominal_control_volume_bounds_m": pabs_bounds,
        "six_face_absorption_box_bounds_m": inner_box,
        "outer_flux_box_bounds_m": (
            outer_bounds if outer_faces else None
        ),
        "exact_flake_mask_kind": (
            "no TaIrTe4; lossless matched-volume control"
            if args.case == "empty-stack"
            else (
                "digitized polygon"
                if args.geometry == "device-a-polygon"
                else "all lateral samples inside the matched control volume"
                if args.geometry == "planar-stack"
                else "analytic half-plane lab_y<=lab_x"
            )
        ),
        "substrate": "285 nm SiO2 on Si",
        "electrodes_in_optical_model": False,
        "electrode_note": (
            "Off-axis-edge optical certificate intentionally excludes Au/Ti; "
            "contacts enter the separate weighting-potential geometry. A contact "
            "optical scenario is required before interpreting contact hotspots."
        ),
        "source": {
            "wavelength_m": WAVELENGTH_M,
            "experimental_band_m": [SOURCE_START_M, SOURCE_STOP_M],
            "physical_target_waist_radius_m": args.waist_um * 1e-6,
            "Lumerical_source_object_waist_radius_m": (
                args.source_object_waist_um * 1e-6
            ),
            "source_object_calibration_field_NPZ_sha256": (
                source_contract.CALIBRATION_BASELINE_FIELD_SHA256
                if args.execution_contract == "production"
                else None
            ),
            "beam_center_m": [beam_x, beam_y],
            "source_span_m": args.source_span_um * 1e-6,
            "normal_incidence": True,
            "polarization_axis": args.polarization,
            "simulation_time_s": args.simulation_time_ps * 1e-12,
            "auto_shutoff_min": args.auto_shutoff_min,
            "scenario_label": (
                "paper-like scalar-Gaussian scenario with an explicitly "
                "assumed waist"
            ),
            "experimentally_reproduced_beam": False,
            "paper_certified_beam": False,
        },
        "domain_bounds_m": {
            "x": [-0.5 * domain_m, 0.5 * domain_m],
            "y": [-0.5 * domain_m, 0.5 * domain_m],
            "z": [FDTD_Z_MIN_M, FDTD_Z_MAX_M],
        },
        "all_six_boundaries": "PML",
        "periodic": False,
        "Q_processing": {
            "clipping": False,
            "smoothing": False,
            "gain": False,
            "global_rescaling": False,
            "polarization_matching_rescaling": False,
        },
        "large_domain_mesh_policy": {
            "uniform_global_fine_mesh": False,
            "global_mesh": "auto non-uniform, conformal variant 1, accuracy 5",
            "outer_local_region_bounds_m": {
                **outer_mesh_bounds,
                "z": (mesh_z_min, mesh_z_max),
            },
            "outer_local_xy_mesh_m": args.outer_local_xy_mesh_nm * 1e-9,
            "fine_refinement_region_bounds_m": {
                **fine_mesh_bounds,
                "z": (mesh_z_min, mesh_z_max),
            },
            "fine_local_xy_mesh_m": args.local_xy_mesh_nm * 1e-9,
            "local_xy_baseline_m": 100.0e-9,
            "required_local_xy_refinement_m": 50.0e-9,
            "local_xy_refinement_status": (
                "ACTIVE_NESTED_50NM_REFINEMENT"
                if args.refinement_half_span_um is not None
                and args.local_xy_mesh_nm <= 50.0
                else "REQUIRED_BEFORE_MATERIAL_Q_PROMOTION"
            ),
            "refinement_half_span_m": (
                None
                if args.refinement_half_span_um is None
                else args.refinement_half_span_um * 1e-6
            ),
            "source_support_outside_fine_region_is_preserved": True,
            "TaIrTe4_z_override_m": args.flake_dz_nm * 1e-9,
            "far_air_SiO2_Si": "wavelength-appropriate automatic coarse mesh",
        },
        "execution_contract": args.execution_contract,
        "monitor_contract": {
            "pabs_field_and_index": True,
            "six_face_absorption_box": True,
            "outer_flux_box": bool(outer_faces),
            "incident_reference_monitor": (
                args.execution_contract in (
                    "production",
                    "edge-isolation-smoke",
                )
            ),
            "diagnostic_field_slice_monitors": (
                args.execution_contract in (
                    "production",
                    "edge-isolation-smoke",
                )
            ),
            "frequency_points": 1,
        },
        "diagnostic_difference_from_production": (
            None
            if args.execution_contract == "production"
            else {
                "classification": (
                    "reduced-cost one-polarization diagnostic only; not a "
                    "paper-like optical result and not a replacement for the "
                    "48 um production contract"
                ),
                "production_lateral_domain_um": 60.0,
                "diagnostic_lateral_domain_um": args.domain_um,
                "production_source_span_um": 50.0,
                "diagnostic_source_span_um": args.source_span_um,
                "production_waist_um": 12.0,
                "diagnostic_waist_um": args.waist_um,
                "production_absorption_padding_um": 2.0,
                "diagnostic_absorption_padding_um": 1.5,
                "removed_monitors": (
                    [
                        "outer six-face box",
                        "three production diagnostic field slices",
                    ]
                    if args.execution_contract == "edge-isolation-smoke"
                    else [
                        "outer six-face box",
                        "incident-reference plane",
                        "three diagnostic field slices",
                    ]
                ),
                "unchanged": [
                    "11 um analysis wavelength and 7-13 um source band",
                    "normal incidence",
                    (
                        "straight 45-degree TaIrTe4 edge"
                        if args.geometry == "straight-45-edge"
                        else "edge-free planar TaIrTe4 layer"
                    ),
                    "130 nm TaIrTe4 thickness",
                    "285 nm SiO2 on Si stack",
                    "epsilon_x=epsilon_b, epsilon_y=epsilon_a, epsilon_z=epsilon_b",
                    "six PML boundaries",
                    "auto non-uniform mesh accuracy 5",
                    "conformal variant 1",
                    "10 nm flake-region z override",
                    "pabs_adv component-resolved Q extraction",
                ],
            }
        ),
        "material_contract": material_contract,
    }
    def exact_flake_mask_builder(
        x_m: np.ndarray,
        y_m: np.ndarray,
        z_m: np.ndarray,
    ) -> np.ndarray:
        if args.geometry == "straight-45-edge":
            xy = y_m[None, :] <= x_m[:, None] + 1e-15
        elif args.geometry == "device-a-polygon":
            xx, yy = np.meshgrid(x_m, y_m, indexing="ij")
            xy = PolygonPath(vertices_um * 1e-6).contains_points(
                np.column_stack((xx.ravel(), yy.ravel())),
                radius=1e-15,
            ).reshape(xx.shape)
        else:
            xy = np.ones((x_m.size, y_m.size), dtype=bool)
        zz = (z_m >= -FLAKE_THICKNESS_M) & (z_m <= 0.0)
        return xy[:, :, None] & zz[None, None, :]

    return {
        "source_bounds": source_bounds,
        "outer_bounds": outer_bounds,
        "inner_faces": inner_faces,
        "outer_faces": outer_faces,
        "geometry": geometry,
        "exact_flake_mask_builder": exact_flake_mask_builder,
    }


def assert_contract(
    base: Any,
    fdtd: Any,
    runtime: Any,
    args: argparse.Namespace,
    setup: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    boundaries: dict[str, str] = {}
    for axis in "xyz":
        for side in ("min", "max"):
            key = f"{axis}_{side}"
            value = str(fdtd.getnamed("FDTD", f"{axis} {side} bc")).strip()
            boundaries[key] = value
            checks[f"{key}_pml"] = value.lower() == "pml"
    checks["one_TaIrTe4_geometry_object"] = int(
        fdtd.getnamednumber("TaIrTe4_flake")
    ) == (
        1 if args.case == "finite-flake" else 0
    )
    checks["no_periodic_boundary"] = all(v.lower() == "pml" for v in boundaries.values())
    injection_axis = str(
        fdtd.getnamed(base.SOURCE_NAME, "injection axis")
    ).strip().lower()
    checks["normal_incidence"] = (
        injection_axis in ("z", "z axis", "3", "3.0")
        or "z" in injection_axis
    )
    checks["correct_polarization"] = np.isclose(
        base.scalar(fdtd.getnamed(base.SOURCE_NAME, "polarization angle"), "polarization"),
        args.polarization_deg,
    )
    checks["correct_waist"] = np.isclose(
        base.scalar(fdtd.getnamed(base.SOURCE_NAME, "waist radius w0"), "waist"),
        args.source_object_waist_um * 1e-6,
        atol=1e-15,
    )
    checks["scalar_Gaussian_source"] = bool(
        round(
            base.scalar(
                fdtd.getnamed(base.SOURCE_NAME, "use scalar approximation"),
                "source.use scalar approximation",
            )
        )
    )
    checks["backward_focus_distance_from_waist"] = np.isclose(
        base.scalar(
            fdtd.getnamed(base.SOURCE_NAME, "distance from waist"),
            "source.distance from waist",
        ),
        -(SOURCE_Z_M - FOCUS_Z_M),
        rtol=0.0,
        atol=1e-15,
    )
    checks["correct_dz"] = np.isclose(
        base.scalar(fdtd.getnamed("flake_mesh", "dz"), "flake dz"),
        args.flake_dz_nm * 1e-9,
        atol=1e-15,
    )
    checks["correct_local_dx"] = np.isclose(
        base.scalar(fdtd.getnamed("flake_mesh", "dx"), "local dx"),
        args.local_xy_mesh_nm * 1e-9,
        atol=1e-15,
    )
    checks["correct_local_dy"] = np.isclose(
        base.scalar(fdtd.getnamed("flake_mesh", "dy"), "local dy"),
        args.local_xy_mesh_nm * 1e-9,
        atol=1e-15,
    )
    checks["correct_nested_outer_mesh"] = (
        args.refinement_half_span_um is None
        or (
            int(fdtd.getnamednumber("flake_outer_mesh")) == 1
            and np.isclose(
                base.scalar(
                    fdtd.getnamed("flake_outer_mesh", "dx"),
                    "outer local dx",
                ),
                args.outer_local_xy_mesh_nm * 1e-9,
                atol=1e-15,
            )
            and np.isclose(
                base.scalar(
                    fdtd.getnamed("flake_outer_mesh", "dy"),
                    "outer local dy",
                ),
                args.outer_local_xy_mesh_nm * 1e-9,
                atol=1e-15,
            )
            and np.isclose(
                base.scalar(
                    fdtd.getnamed("flake_outer_mesh", "dz"),
                    "outer local dz",
                ),
                args.flake_dz_nm * 1e-9,
                atol=1e-15,
            )
        )
    )
    checks["correct_thickness"] = args.case == "empty-stack" or np.isclose(
        base.scalar(fdtd.getnamed("TaIrTe4_flake", "z span"), "flake thickness"),
        FLAKE_THICKNESS_M,
        atol=1e-15,
    )
    checks["diagnostic_nominal_Q_and_six_face_control_volume_match"] = (
        args.execution_contract
        not in ("diagnostic-smoke", "edge-isolation-smoke")
        or all(
            np.allclose(
                setup["geometry"][
                    "pabs_nominal_control_volume_bounds_m"
                ][axis],
                setup["geometry"][
                    "six_face_absorption_box_bounds_m"
                ][axis],
                rtol=0.0,
                atol=1.0e-18,
            )
            for axis in "xyz"
        )
    )
    runtime.configure_session_resources(fdtd)
    fdtd.runsetup()
    dt_s = base.scalar(fdtd.getnamed("FDTD", "dt"), "FDTD.dt")
    epsilon_readback = material_epsilon_readback(
        fdtd,
        args,
        setup["geometry"]["material_contract"],
        dt_s=dt_s,
    )
    checks["requested_z_equals_x_for_paper_closure"] = (
        args.epsilon_c_model != "paper-b-closure"
        or setup["geometry"]["material_contract"][
            "requested_epsilon_at_11um"
        ]["z"]
        == setup["geometry"]["material_contract"][
            "requested_epsilon_at_11um"
        ]["x"]
    )
    checks["fitted_z_equals_x_for_paper_closure"] = (
        args.epsilon_c_model != "paper-b-closure"
        or epsilon_readback["fitted_z_vs_x_relative_difference"] < 1e-12
    )
    checks["finite_dt_z_equals_x_for_paper_closure"] = (
        args.epsilon_c_model != "paper-b-closure"
        or epsilon_readback["finite_dt_z_vs_x_relative_difference"] < 1e-12
    )
    resources = runtime.resource_contract(fdtd)
    checks["requested_gpu_resource_active"] = resources["2"]["active"].strip() == "1"
    checks["all"] = all(checks.values())
    if not checks["all"]:
        raise RuntimeError(
            "paper IR contract failed: "
            f"{[k for k,v in checks.items() if not v]}; "
            f"injection_axis_readback={injection_axis!r}"
        )
    fdtd_bounds = {
        axis: [
            base.scalar(
                fdtd.getnamed("FDTD", f"{axis} min"),
                f"FDTD.{axis} min",
            ),
            base.scalar(
                fdtd.getnamed("FDTD", f"{axis} max"),
                f"FDTD.{axis} max",
            ),
        ]
        for axis in "xyz"
    }
    source_bounds = {
        axis: [
            base.scalar(
                fdtd.getnamed(base.SOURCE_NAME, f"{axis} min"),
                f"source.{axis} min",
            ),
            base.scalar(
                fdtd.getnamed(base.SOURCE_NAME, f"{axis} max"),
                f"source.{axis} max",
            ),
        ]
        if axis in "xy"
        else [
            base.scalar(
                fdtd.getnamed(base.SOURCE_NAME, "z"),
                "source.z",
            ),
            base.scalar(
                fdtd.getnamed(base.SOURCE_NAME, "z"),
                "source.z",
            ),
        ]
        for axis in "xyz"
    }
    if args.case == "empty-stack":
        flake_vertices_readback = None
        flake_bounds = None
    elif args.geometry == "planar-stack":
        flake_vertices_readback = None
        flake_bounds = {
            axis: [
                base.scalar(
                    fdtd.getnamed("TaIrTe4_flake", f"{axis} min"),
                    f"flake.{axis} min",
                ),
                base.scalar(
                    fdtd.getnamed("TaIrTe4_flake", f"{axis} max"),
                    f"flake.{axis} max",
                ),
            ]
            for axis in "xyz"
        }
        for axis in "xy":
            if not (
                flake_bounds[axis][0] < fdtd_bounds[axis][0]
                and flake_bounds[axis][1] > fdtd_bounds[axis][1]
            ):
                raise RuntimeError(
                    f"planar TaIrTe4 does not extend through {axis}-PML: "
                    f"{flake_bounds[axis]} versus {fdtd_bounds[axis]}"
                )
    else:
        flake_vertices_readback = np.asarray(
            fdtd.getnamed("TaIrTe4_flake", "vertices"),
            float,
        )
        if (
            flake_vertices_readback.ndim != 2
            or flake_vertices_readback.shape[1] != 2
        ):
            raise RuntimeError(
                "unexpected polygon vertices readback shape: "
                f"{flake_vertices_readback.shape}"
            )
        flake_bounds = {
            "x": [
                float(np.min(flake_vertices_readback[:, 0])),
                float(np.max(flake_vertices_readback[:, 0])),
            ],
            "y": [
                float(np.min(flake_vertices_readback[:, 1])),
                float(np.max(flake_vertices_readback[:, 1])),
            ],
            "z": [
                base.scalar(
                    fdtd.getnamed("TaIrTe4_flake", "z min"),
                    "flake.z min",
                ),
                base.scalar(
                    fdtd.getnamed("TaIrTe4_flake", "z max"),
                    "flake.z max",
                ),
            ],
        }
    def mesh_override_readback(name: str) -> dict[str, Any]:
        return {
            "name": name,
            "bounds_m": {
                axis: [
                    base.scalar(
                        fdtd.getnamed(name, f"{axis} min"),
                        f"{name}.{axis} min",
                    ),
                    base.scalar(
                        fdtd.getnamed(name, f"{axis} max"),
                        f"{name}.{axis} max",
                    ),
                ]
                for axis in "xyz"
            },
            "override_x_mesh": bool(
                base.scalar(
                    fdtd.getnamed(name, "override x mesh"),
                    f"{name}.override x",
                )
            ),
            "override_y_mesh": bool(
                base.scalar(
                    fdtd.getnamed(name, "override y mesh"),
                    f"{name}.override y",
                )
            ),
            "override_z_mesh": bool(
                base.scalar(
                    fdtd.getnamed(name, "override z mesh"),
                    f"{name}.override z",
                )
            ),
            "dz_m": base.scalar(
                fdtd.getnamed(name, "dz"),
                f"{name}.dz",
            ),
            "dx_m": base.scalar(
                fdtd.getnamed(name, "dx"),
                f"{name}.dx",
            ),
            "dy_m": base.scalar(
                fdtd.getnamed(name, "dy"),
                f"{name}.dy",
            ),
        }

    mesh_overrides = [mesh_override_readback("flake_mesh")]
    if args.refinement_half_span_um is not None:
        mesh_overrides.append(
            mesh_override_readback("flake_outer_mesh")
        )
    return {
        "checks": checks,
        "boundaries": boundaries,
        "solver": {
            "version": str(fdtd.version()),
            "root": str(runtime.R12_ROOT),
            "resources": resources,
        },
        "geometry": setup["geometry"],
        "material": {
            **setup["geometry"]["material_contract"],
            "epsilon_readback": epsilon_readback,
        },
        "mesh": {
            "type": str(fdtd.getnamed("FDTD", "mesh type")),
            "refinement": str(fdtd.getnamed("FDTD", "mesh refinement")),
            "accuracy": base.scalar(fdtd.getnamed("FDTD", "mesh accuracy"), "accuracy"),
            "minimum_mesh_step_m": base.scalar(
                fdtd.getnamed("FDTD", "min mesh step"),
                "FDTD.min mesh step",
            ),
            "flake_dz_m": args.flake_dz_nm * 1e-9,
            "local_xy_mesh_m": args.local_xy_mesh_nm * 1e-9,
            "outer_local_xy_mesh_m": (
                args.outer_local_xy_mesh_nm * 1e-9
            ),
            "refinement_half_span_m": (
                None
                if args.refinement_half_span_um is None
                else args.refinement_half_span_um * 1e-6
            ),
            "override_objects": mesh_overrides,
            "uniform_global_fine_mesh_present": False,
            "local_x_or_y_override_present": True,
        },
        "object_bounds_readback_m": {
            "FDTD_nominal_outer_bounds": fdtd_bounds,
            "PML_inner_bounds": (
                "not available before native post-run mesh readback"
            ),
            "source": source_bounds,
            "flake": flake_bounds,
            "flake_vertices_readback_m": (
                None
                if flake_vertices_readback is None
                else flake_vertices_readback.tolist()
            ),
            "absorption_analysis": setup["geometry"][
                "absorption_analysis_bounds_m"
            ],
            "pabs_nominal_control_volume": setup["geometry"][
                "pabs_nominal_control_volume_bounds_m"
            ],
            "six_face_absorption_box": setup["geometry"][
                "six_face_absorption_box_bounds_m"
            ],
        },
        "execution_contract": args.execution_contract,
        "source_readback": {
            "injection_axis": injection_axis,
            "direction": str(fdtd.getnamed(base.SOURCE_NAME, "direction")),
            "polarization_angle_deg": args.polarization_deg,
        },
    }


def plot_geometry(output: Path, args: argparse.Namespace, setup: dict[str, Any]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    flake_vertices = np.asarray(args.flake_vertices_um, float)
    vertices = np.vstack((flake_vertices, flake_vertices[0]))
    if args.case == "finite-flake":
        ax.fill(
            vertices[:, 0],
            vertices[:, 1],
            color="#d89023",
            alpha=0.75,
            label="130 nm TaIrTe4",
        )
    half = 0.5 * args.source_span_um
    ax.add_patch(
        plt.Rectangle(
            (args.beam_x_um - half, args.beam_y_um - half),
            2 * half,
            2 * half,
            fill=False,
            ls="--",
            color="tab:blue",
            label="Gaussian aperture",
        )
    )
    ax.add_patch(
        plt.Circle(
            (args.beam_x_um, args.beam_y_um),
            args.waist_um,
            fill=False,
            color="tab:red",
            label="waist radius w0",
        )
    )
    ax.arrow(-1, -3, 5 if args.polarization == "b" else 0, 5 if args.polarization == "a" else 0,
             width=0.15, color="black", length_includes_head=True)
    ax.set_aspect("equal")
    ax.set(
        xlabel="lab x = crystal b (um)",
        ylabel="lab y = crystal a (um)",
        title=(
            "Approximate Device A optical geometry"
            if args.geometry == "device-a-polygon"
            else (
                "Corner-free straight 45-degree edge control"
                if args.geometry == "straight-45-edge"
                else "Edge-free planar TaIrTe4 control"
            )
        ),
    )
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.axhspan(FDTD_Z_MIN_M * 1e6, -(FLAKE_THICKNESS_M + SIO2_THICKNESS_M) * 1e6, color="silver", label="Si")
    ax.axhspan(-(FLAKE_THICKNESS_M + SIO2_THICKNESS_M) * 1e6, -FLAKE_THICKNESS_M * 1e6, color="lightblue", label="285 nm SiO2")
    if args.case == "finite-flake":
        ax.axhspan(
            -FLAKE_THICKNESS_M * 1e6,
            0,
            color="#d89023",
            label="130 nm TaIrTe4",
        )
    ax.axhline(SOURCE_Z_M * 1e6, color="tab:green", ls="--", label="Gaussian source")
    ax.plot([0], [FOCUS_Z_M * 1e6], marker="x", color="tab:green", label="focus")
    ax.set(xlabel="lateral (schematic)", ylabel="z (um)", title="Normal-incidence stack")
    ax.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "paper_ir_device_a_geometry.png", dpi=200)
    plt.close(figure)


def dual_cell_weights_1d(coordinate: np.ndarray) -> np.ndarray:
    values = np.asarray(coordinate, float).reshape(-1)
    if values.size < 2 or np.any(np.diff(values) <= 0.0):
        raise ValueError("beam-plane coordinates must be strictly increasing")
    boundaries = np.empty(values.size + 1, float)
    boundaries[1:-1] = 0.5 * (values[:-1] + values[1:])
    boundaries[0] = values[0] - 0.5 * (values[1] - values[0])
    boundaries[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.diff(boundaries)


def fit_elliptical_gaussian(
    x_m: np.ndarray,
    y_m: np.ndarray,
    intensity_W_m2: np.ndarray,
) -> dict[str, Any]:
    """Fit I=A exp[-2((x-x0)^2/wx^2+(y-y0)^2/wy^2)]."""
    x = np.asarray(x_m, float).reshape(-1)
    y = np.asarray(y_m, float).reshape(-1)
    intensity = np.asarray(intensity_W_m2, float).reshape(x.size, y.size)
    if not np.all(np.isfinite(intensity)) or float(np.max(intensity)) <= 0.0:
        raise RuntimeError("beam intensity is not finite and positive")
    dx = dual_cell_weights_1d(x)
    dy = dual_cell_weights_1d(y)
    area = dx[:, None] * dy[None, :]
    total = float(np.sum(intensity * area))
    x0_moment = float(np.sum(intensity * area * x[:, None]) / total)
    y0_moment = float(np.sum(intensity * area * y[None, :]) / total)
    var_x = float(
        np.sum(intensity * area * (x[:, None] - x0_moment) ** 2) / total
    )
    var_y = float(
        np.sum(intensity * area * (y[None, :] - y0_moment) ** 2) / total
    )
    waist_x_moment = 2.0 * np.sqrt(max(var_x, np.finfo(float).tiny))
    waist_y_moment = 2.0 * np.sqrt(max(var_y, np.finfo(float).tiny))
    xx, yy = np.meshgrid(x, y, indexing="ij")
    peak = float(np.max(intensity))
    selected = intensity >= 1.0e-3 * peak
    observed = intensity[selected] / peak

    def residual(parameter: np.ndarray) -> np.ndarray:
        amplitude = np.exp(parameter[0])
        centre_x = parameter[1]
        centre_y = parameter[2]
        waist_x = np.exp(parameter[3])
        waist_y = np.exp(parameter[4])
        model = amplitude * np.exp(
            -2.0
            * (
                (xx[selected] - centre_x) ** 2 / waist_x**2
                + (yy[selected] - centre_y) ** 2 / waist_y**2
            )
        )
        return model - observed

    initial = np.asarray(
        [
            0.0,
            x0_moment,
            y0_moment,
            np.log(waist_x_moment),
            np.log(waist_y_moment),
        ]
    )
    solved = least_squares(
        residual,
        initial,
        bounds=(
            np.asarray(
                [
                    np.log(0.05),
                    float(x[0]),
                    float(y[0]),
                    np.log(max(np.min(np.diff(x)), 1.0e-12)),
                    np.log(max(np.min(np.diff(y)), 1.0e-12)),
                ]
            ),
            np.asarray(
                [
                    np.log(20.0),
                    float(x[-1]),
                    float(y[-1]),
                    np.log(2.0 * (x[-1] - x[0])),
                    np.log(2.0 * (y[-1] - y[0])),
                ]
            ),
        ),
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=500,
    )
    fitted = {
        "amplitude_over_observed_peak": float(np.exp(solved.x[0])),
        "center_x_m": float(solved.x[1]),
        "center_y_m": float(solved.x[2]),
        "waist_x_m": float(np.exp(solved.x[3])),
        "waist_y_m": float(np.exp(solved.x[4])),
    }
    fitted["waist_effective_geometric_mean_m"] = float(
        np.sqrt(fitted["waist_x_m"] * fitted["waist_y_m"])
    )
    fitted.update(
        {
            "fit_success": bool(solved.success),
            "fit_message": str(solved.message),
            "fit_selected_point_count": int(np.count_nonzero(selected)),
            "fit_relative_RMS_over_peak": float(
                np.sqrt(np.mean(residual(solved.x) ** 2))
            ),
            "integrated_incident_power_W": total,
            "moment_center_x_m": x0_moment,
            "moment_center_y_m": y0_moment,
            "moment_waist_x_m": float(waist_x_moment),
            "moment_waist_y_m": float(waist_y_moment),
        }
    )
    return fitted


def monitor_fields(fdtd: Any, monitor: str) -> dict[str, Any]:
    coordinates = {
        axis: np.asarray(fdtd.getdata(monitor, axis, 1), float).reshape(-1)
        for axis in "xyz"
    }
    electric = {
        axis: np.asarray(fdtd.getdata(monitor, f"E{axis}", 1)).squeeze()
        for axis in "xyz"
    }
    magnetic = {
        axis: np.asarray(fdtd.getdata(monitor, f"H{axis}", 1)).squeeze()
        for axis in "xyz"
    }
    return {
        "coordinates": coordinates,
        "electric": electric,
        "magnetic": magnetic,
    }


def extract_w2_beam_and_fields(
    base: Any,
    fdtd: Any,
    output: Path,
    source_power_native_W: float,
) -> dict[str, Any]:
    incident_plane = monitor_fields(fdtd, W2_INCIDENT_MONITOR)
    flake_plane = monitor_fields(fdtd, W2_FLAKE_FIELD_MONITOR)
    x = incident_plane["coordinates"]["x"]
    y = incident_plane["coordinates"]["y"]
    eta0 = float(base.ETA0)
    ex_down = 0.5 * (
        incident_plane["electric"]["x"]
        - eta0 * incident_plane["magnetic"]["y"]
    )
    ey_down = 0.5 * (
        incident_plane["electric"]["y"]
        + eta0 * incident_plane["magnetic"]["x"]
    )
    downward_intensity = (
        np.abs(ex_down) ** 2 + np.abs(ey_down) ** 2
    ) / (2.0 * eta0)
    downward_intensity = np.asarray(downward_intensity, float).reshape(
        x.size, y.size
    )
    beam_fit = fit_elliptical_gaussian(x, y, downward_intensity)
    total_flake_e2 = sum(
        np.abs(flake_plane["electric"][axis]) ** 2 for axis in "xyz"
    )
    total_flake_fit = fit_elliptical_gaussian(
        flake_plane["coordinates"]["x"],
        flake_plane["coordinates"]["y"],
        np.asarray(total_flake_e2, float),
    )
    source_profile: dict[str, Any]
    try:
        profile = fdtd.getresult(base.SOURCE_NAME, "fields")
        source_profile = {
            "x_m": np.asarray(profile["x"], float).reshape(-1),
            "y_m": np.asarray(profile["y"], float).reshape(-1),
            "z_m": np.asarray(profile["z"], float).reshape(-1),
            "E": np.asarray(profile["E"]),
        }
    except Exception as exc:
        source_profile = {"error": f"{type(exc).__name__}: {exc}"}
    artifact_path = output / "w2_beam_and_field_components.npz"
    payload: dict[str, Any] = {
        "incident_x_m": x,
        "incident_y_m": y,
        "incident_z_m": incident_plane["coordinates"]["z"],
        "downward_Ex": ex_down,
        "downward_Ey": ey_down,
        "downward_intensity_W_m2": downward_intensity,
        "flake_x_m": flake_plane["coordinates"]["x"],
        "flake_y_m": flake_plane["coordinates"]["y"],
        "flake_z_m": flake_plane["coordinates"]["z"],
        "source_power_native_W": np.asarray([source_power_native_W]),
        "beam_fit_json": np.asarray([json.dumps(base.jsonable(beam_fit))]),
        "total_flake_fit_json": np.asarray(
            [json.dumps(base.jsonable(total_flake_fit))]
        ),
    }
    for prefix, fields in (
        ("incident", incident_plane),
        ("flake", flake_plane),
    ):
        for field_kind, short in (("electric", "E"), ("magnetic", "H")):
            for axis, values in fields[field_kind].items():
                payload[f"{prefix}_{short}{axis}"] = values
    if "error" not in source_profile:
        payload.update(
            {
                "source_profile_x_m": source_profile["x_m"],
                "source_profile_y_m": source_profile["y_m"],
                "source_profile_z_m": source_profile["z_m"],
                "source_profile_E": source_profile["E"],
            }
        )
    np.savez_compressed(artifact_path, **payload)
    component_norms = {
        prefix: {
            f"{short}{axis}_L2": float(np.linalg.norm(values))
            for field_kind, short in (("electric", "E"), ("magnetic", "H"))
            for axis, values in fields[field_kind].items()
        }
        for prefix, fields in (
            ("incident_plane_total_field", incident_plane),
            ("flake_midplane_total_field", flake_plane),
        )
    }
    return {
        "incident_plane_z_m": W2_INCIDENT_PLANE_Z_M,
        "incident_decomposition": (
            "downward Ex/Ey from total E/H in homogeneous air using "
            "Ex_down=(Ex-eta0 Hy)/2 and Ey_down=(Ey+eta0 Hx)/2"
        ),
        "beam_fit_from_downward_incident_intensity": beam_fit,
        "flake_midplane_z_m": W2_FLAKE_MIDPLANE_Z_M,
        "total_field_flake_midplane_fit": total_flake_fit,
        "field_component_norms": component_norms,
        "source_profile_readback": (
            {"available": True, "E_shape": list(source_profile["E"].shape)}
            if "error" not in source_profile
            else {"available": False, "error": source_profile["error"]}
        ),
        "source_power_native_W": source_power_native_W,
        "incident_plane_power_over_source_power": (
            beam_fit["integrated_incident_power_W"] / source_power_native_W
        ),
        "all_saved_fields_finite": all(
            np.all(np.isfinite(values))
            for fields in (incident_plane, flake_plane)
            for field_kind in ("electric", "magnetic")
            for values in fields[field_kind].values()
        ),
        "artifact": {
            "path": str(artifact_path.resolve()),
            "size_bytes": artifact_path.stat().st_size,
            "sha256": base.sha256(artifact_path),
        },
    }


def run_diagnostic_gpu_smoke_case(
    base: Any,
    fdtd: Any,
    runtime: Any,
    args: argparse.Namespace,
    output: Path,
    setup: dict[str, Any],
    pre_run_contract: dict[str, Any],
) -> dict[str, Any]:
    """Run the reduced diagnostic without incident normalization or Q edits."""
    resource = runtime.run_session(
        fdtd,
        (
            f"paper_ir_diagnostic_{args.polarization}_"
            f"L{args.domain_um:g}_pml{args.pml_layers}_"
            f"dz{args.flake_dz_nm:g}"
        ),
    )
    source_power_native = base.scalar(
        fdtd.sourcepower(
            base.TARGET_FREQUENCY_HZ,
            2,
            base.SOURCE_NAME,
        ),
        "native source power at 11 um",
    )
    if not np.isfinite(source_power_native) or source_power_native <= 0.0:
        raise RuntimeError(
            f"invalid native Gaussian source power: {source_power_native}"
        )
    beam_and_fields = (
        extract_w2_beam_and_fields(
            base,
            fdtd,
            output,
            source_power_native,
        )
        if args.execution_contract == "edge-isolation-smoke"
        else None
    )

    # The diagnostic deliberately remains in native source-amplitude units.
    # The common multiplicative scale cancels from the volume/flux closure.
    inner_flux = base.face_fluxes(
        fdtd,
        setup["inner_faces"],
        source_power_native,
        1.0,
    )
    realized_control_volume = realized_six_face_control_volume(
        fdtd,
        setup["inner_faces"],
    )
    realized_bounds = realized_control_volume["bounds_m"]
    auto_shutoff = final_logged_auto_shutoff(output)
    fdtd.runanalysis(base.PABS_GROUP)
    q_data = base.common_grid_component_q(
        fdtd,
        base.TARGET_FREQUENCY_HZ,
    )
    artifact = {
        "x_m": np.asarray(q_data["x_m"], float),
        "y_m": np.asarray(q_data["y_m"], float),
        "z_m": np.asarray(q_data["z_m"], float),
        # The base pabs helper historically calls these arrays "native".
        # They have already been interpolated to the field-monitor common
        # grid, so the published artifact uses the physically correct name.
        "Qx_common_grid_W_m3": np.asarray(
            q_data["Qx_native_W_m3"],
            float,
        ),
        "Qy_common_grid_W_m3": np.asarray(
            q_data["Qy_native_W_m3"],
            float,
        ),
        "Qz_common_grid_W_m3": np.asarray(
            q_data["Qz_native_W_m3"],
            float,
        ),
        "Q_common_grid_W_m3": np.asarray(
            q_data["Q_native_W_m3"],
            float,
        ),
    }
    finite_arrays = all(
        np.all(np.isfinite(value))
        for key, value in artifact.items()
        if key.endswith("_m3")
    )
    if not finite_arrays:
        raise RuntimeError("diagnostic Q contains NaN or Inf")

    exact_flake_mask = np.asarray(
        setup["exact_flake_mask_builder"](
            artifact["x_m"],
            artifact["y_m"],
            artifact["z_m"],
        ),
        dtype=bool,
    )
    if exact_flake_mask.shape != artifact["Q_common_grid_W_m3"].shape:
        raise RuntimeError(
            "exact flake mask shape mismatch: "
            f"{exact_flake_mask.shape} != "
            f"{artifact['Q_common_grid_W_m3'].shape}"
        )

    common_coordinates = {
        axis: artifact[f"{axis}_m"]
        for axis in "xyz"
    }
    component_power_common = {
        axis: integrate_xyz_bounded(
            artifact[f"Q{axis}_common_grid_W_m3"],
            common_coordinates,
            realized_bounds,
        )
        for axis in "xyz"
    }
    native_absorption = native_component_absorption_bounded(
        base,
        fdtd,
        base.TARGET_FREQUENCY_HZ,
        realized_bounds,
    )
    component_power_native = native_absorption["component_power_W"]
    p_q_common = float(sum(component_power_common.values()))
    p_q_native = float(native_absorption["total_power_W"])
    p_six = float(inner_flux["net_inward_power_W"])
    common_closure = abs(p_q_common - p_six) / max(
        abs(p_six),
        np.finfo(float).tiny,
    )
    native_closure = abs(p_q_native - p_six) / max(
        abs(p_six),
        np.finfo(float).tiny,
    )
    q_total = artifact["Q_common_grid_W_m3"]
    support_power = integrate_xyz_bounded(
        np.where(exact_flake_mask, q_total, 0.0),
        common_coordinates,
        realized_bounds,
    )
    outside_support_power = integrate_xyz_bounded(
        np.where(~exact_flake_mask, q_total, 0.0),
        common_coordinates,
        realized_bounds,
    )
    common_native_component_difference = {
        axis: abs(
            component_power_common[axis]
            - component_power_native[axis]
        )
        / max(
            abs(component_power_native[axis]),
            np.finfo(float).tiny,
        )
        for axis in "xyz"
    }
    common_native_total_difference = abs(
        p_q_common - p_q_native
    ) / max(abs(p_q_native), np.finfo(float).tiny)
    q_support_contract = {}
    for axis in "xyz":
        coordinate = common_coordinates[axis]
        step = np.diff(coordinate)
        low_gap = float(coordinate[0] - realized_bounds[axis][0])
        high_gap = float(realized_bounds[axis][1] - coordinate[-1])
        q_support_contract[axis] = {
            "common_Q_sample_bounds_m": [
                float(coordinate[0]),
                float(coordinate[-1]),
            ],
            "realized_flux_face_bounds_m": realized_bounds[axis],
            "low_face_to_first_sample_m": low_gap,
            "last_sample_to_high_face_m": high_gap,
            "maximum_common_grid_step_m": float(np.max(step)),
            "samples_inside_realized_faces": (
                low_gap >= -1.0e-15 and high_gap >= -1.0e-15
            ),
            "outer_gap_no_more_than_one_grid_step": (
                low_gap <= float(np.max(step)) + 1.0e-15
                and high_gap <= float(np.max(step)) + 1.0e-15
            ),
        }
    hotspot_index = np.unravel_index(
        int(np.argmax(q_total)),
        q_total.shape,
    )
    hotspot = {
        "x_m": float(artifact["x_m"][hotspot_index[0]]),
        "y_m": float(artifact["y_m"][hotspot_index[1]]),
        "z_m": float(artifact["z_m"][hotspot_index[2]]),
        "Q_common_grid_W_m3": float(q_total[hotspot_index]),
    }
    metadata = {
        "classification": (
            "reduced-cost one-polarization GPU diagnostic; not a paper-like "
            "optical result, not production Q, and not normalized to 1 W/m2"
        ),
        "generation_command": shlex.join([sys.executable, *sys.argv]),
        "generation_commit": base.git_commit(),
        "execution_contract": args.execution_contract,
        "geometry": setup["geometry"],
        "pre_run_contract": pre_run_contract,
        "source_amplitude_contract": (
            "native Lumerical source amplitude; no incident-intensity "
            "normalization and no empirical flux gain"
        ),
        "beam_and_field_readback": beam_and_fields,
        "array_axis_order": ["x", "y", "z"],
        "Q_array_grid": (
            "field-monitor common x/y/z grid after component-specific "
            "native Yee interpolation"
        ),
        "Q_units": "W/m^3 at native source amplitude",
        "realized_control_volume": realized_control_volume,
        "quadrature_contract": (
            "bounded dual-cell weights close exactly on independently read "
            "six-face locations; common-grid arrays are not called native"
        ),
        "independent_field_index_pairing": native_absorption[
            "independent_field_index_pairing"
        ],
        "exact_flake_mask_is_analysis_only": True,
        "Q_operations": {
            "clipped": False,
            "smoothed": False,
            "gain_applied": False,
            "globally_rescaled": False,
            "tiled": False,
            "source_deleted_outside_support": False,
        },
    }
    artifact_path = output / "diagnostic_q_common_grid_artifact.npz"
    np.savez(
        artifact_path,
        **artifact,
        exact_flake_mask=exact_flake_mask,
        source_power_native_W=np.asarray([source_power_native]),
        P_Q_common_grid_bounded_W=np.asarray([p_q_common]),
        P_Q_native_component_bounded_W=np.asarray([p_q_native]),
        P_six_native_W=np.asarray([p_six]),
        **{
            f"realized_control_volume_{axis}_bounds_m": np.asarray(
                bounds,
                float,
            )
            for axis, bounds in realized_bounds.items()
        },
        metadata_json=np.asarray([json.dumps(base.jsonable(metadata))]),
    )
    base.plot_q_slices(
        output,
        {
            "x_m": artifact["x_m"],
            "y_m": artifact["y_m"],
            "z_m": artifact["z_m"],
            "Qx_W_m3": artifact["Qx_common_grid_W_m3"],
            "Qy_W_m3": artifact["Qy_common_grid_W_m3"],
            "Qz_W_m3": artifact["Qz_common_grid_W_m3"],
            "Q_on_W_m3": artifact["Q_common_grid_W_m3"],
        },
    )

    return {
        "classification": (
            "DIAGNOSTIC_ONE_POL_GPU_SMOKE_NOT_PRODUCTION_OR_PAPER_RESULT"
        ),
        "resource": resource,
        "polarization": args.polarization,
        "normalization": {
            "measured_source_power_native_W": source_power_native,
            "incident_intensity_normalization_applied": False,
            "empirical_flux_gain": False,
        },
        "beam_and_field_readback": beam_and_fields,
        "component_power_common_grid_bounded_W": component_power_common,
        "component_power_native_Yee_bounded_W": component_power_native,
        "P_Q_common_grid_bounded_W": p_q_common,
        "P_Q_native_Yee_bounded_W": p_q_native,
        "P_six_face_native_W": p_six,
        "common_grid_six_face_relative_closure": common_closure,
        "native_Yee_six_face_relative_closure": native_closure,
        "common_native_component_relative_difference": (
            common_native_component_difference
        ),
        "common_native_total_relative_difference": (
            common_native_total_difference
        ),
        "realized_control_volume": realized_control_volume,
        "Q_support_contract": q_support_contract,
        "independent_field_index_pairing": native_absorption[
            "independent_field_index_pairing"
        ],
        "maximum_independent_field_index_coordinate_mismatch_m": (
            native_absorption["maximum_coordinate_mismatch_m"]
        ),
        "auto_shutoff": auto_shutoff,
        "Q_hotspot": hotspot,
        "support_analysis": {
            "P_Q_inside_exact_flake_mask_native_W": support_power,
            "P_Q_outside_exact_flake_mask_native_W": outside_support_power,
            "outside_fraction_of_total": abs(outside_support_power)
            / max(abs(p_q_common), np.finfo(float).tiny),
            "note": (
                "mask is used only for auditing; the saved full-grid Q was "
                "not cropped or altered"
            ),
        },
        "minimum_Q_common_grid_W_m3": float(np.min(q_total)),
        "maximum_Q_common_grid_W_m3": float(np.max(q_total)),
        "six_face_native_source_amplitude": {
            "faces": {
                key: {
                    "monitor": value["monitor"],
                    "normalized_signed_axis_flux": value[
                        "normalized_signed_axis_flux"
                    ],
                    "signed_axis_power_native_W": value[
                        "signed_axis_power_W_at_1_W_m2"
                    ],
                    "outward_power_native_W": value[
                        "outward_power_W_at_1_W_m2"
                    ],
                }
                for key, value in inner_flux["faces"].items()
            },
            "net_inward_power_native_W": p_six,
            "sum_absolute_face_power_native_W": inner_flux[
                "sum_absolute_face_power_W"
            ],
        },
        "artifact": artifact_path.name,
        "artifact_metadata": metadata,
        "acceptance": {
            "solver_returned_normally": True,
            "solver_log_reports_successful_completion": auto_shutoff[
                "simulation_completed_successfully"
            ],
            "auto_shutoff_reached_requested_threshold": (
                auto_shutoff["final_value"] <= args.auto_shutoff_min
            ),
            "source_power_positive": source_power_native > 0.0,
            "flake_plane_beam_fit_and_field_components_saved": (
                args.execution_contract != "edge-isolation-smoke"
                or (
                    beam_and_fields is not None
                    and beam_and_fields["all_saved_fields_finite"]
                    and beam_and_fields[
                        "beam_fit_from_downward_incident_intensity"
                    ]["fit_success"]
                    and beam_and_fields[
                        "beam_fit_from_downward_incident_intensity"
                    ]["integrated_incident_power_W"]
                    > 0.0
                )
            ),
            "Q_arrays_finite": finite_arrays,
            "Qx_Qy_Qz_exported": all(
                artifact[f"Q{axis}_common_grid_W_m3"].size > 0
                for axis in "xyz"
            ),
            "lossy_epsilon_z_produces_nonzero_integrated_Qz": (
                abs(component_power_native["z"])
                > np.finfo(float).tiny
            ),
            "common_grid_six_face_closure_lt_0p5_percent": (
                common_closure < base.POWER_CLOSURE_LIMIT
            ),
            "native_Yee_six_face_closure_lt_0p5_percent": (
                native_closure < base.POWER_CLOSURE_LIMIT
            ),
            "independent_field_index_coordinate_mismatch_lt_1fm": (
                native_absorption["maximum_coordinate_mismatch_m"]
                < 1.0e-15
            ),
            "realized_six_face_surfaces_form_one_box_lt_1fm": (
                realized_control_volume[
                    "maximum_transverse_boundary_mismatch_m"
                ]
                < 1.0e-15
            ),
            "common_Q_samples_inside_realized_faces": all(
                value["samples_inside_realized_faces"]
                and value["outer_gap_no_more_than_one_grid_step"]
                for value in q_support_contract.values()
            ),
            "no_Q_clipping_smoothing_gain_rescaling_tiling_or_deletion": True,
        },
    }


def main() -> int:
    args = parse_args()
    if not APPROVED_API.joinpath("lumapi.py").is_file():
        raise FileNotFoundError(f"approved lumapi missing: {APPROVED_API}")
    os.environ["VC_LUMERICAL_ROOT"] = str(APPROVED_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(APPROVED_ROOT)
    os.environ["LUMERICAL_PYTHONPATH"] = str(APPROVED_API)
    os.environ["PYTHONPATH"] = ":".join(
        value
        for value in (str(APPROVED_API), os.environ.get("PYTHONPATH", ""))
        if value and "/opt/lumerical/" not in value
    )
    os.environ["PATH"] = f"{APPROVED_ROOT / 'bin'}:{os.environ.get('PATH','')}"

    base = load_base()
    base.select_installation = lambda requested="v261": SimpleNamespace(
        version_key="v261",
        root=APPROVED_ROOT.resolve(),
        lumapi_path=(APPROVED_API / "lumapi.py").resolve(),
        device_executable=(APPROVED_ROOT / "bin" / "device").resolve(),
    )
    base.TARGET_WAVELENGTH_M = WAVELENGTH_M
    base.TARGET_FREQUENCY_HZ = C0 / WAVELENGTH_M
    base.SOURCE_START_M = SOURCE_START_M
    base.SOURCE_STOP_M = SOURCE_STOP_M
    base.FLAKE_THICKNESS_M = FLAKE_THICKNESS_M
    base.FLAKE_BOUNDS_M = args.absorption_bounds_m
    if args.geometry == "straight-45-edge":
        span_x = (
            args.absorption_bounds_m["x"][1]
            - args.absorption_bounds_m["x"][0]
        )
        span_y = (
            args.absorption_bounds_m["y"][1]
            - args.absorption_bounds_m["y"][0]
        )
        base.GEOMETRIC_AREA_M2 = 0.5 * span_x * span_y
    elif args.geometry == "planar-stack":
        span_x = (
            args.absorption_bounds_m["x"][1]
            - args.absorption_bounds_m["x"][0]
        )
        span_y = (
            args.absorption_bounds_m["y"][1]
            - args.absorption_bounds_m["y"][0]
        )
        base.GEOMETRIC_AREA_M2 = span_x * span_y
    else:
        base.GEOMETRIC_AREA_M2 = polygon_area(args.flake_vertices_um) * 1e-12
    base.SIO2_THICKNESS_M = SIO2_THICKNESS_M
    base.SI_DEPTH_M = SI_DEPTH_M
    base.PABS_PADDING_M = PABS_PADDING_M
    base.FDTD_Z_MIN_M = FDTD_Z_MIN_M
    base.FDTD_Z_MAX_M = FDTD_Z_MAX_M
    base.GAUSSIAN_SOURCE_Z_M = SOURCE_Z_M
    base.GAUSSIAN_FOCUS_Z_M = FOCUS_Z_M
    base.INCIDENT_REFERENCE_Z_M = INCIDENT_Z_M
    base.INNER_BOX = args.inner_box
    base.MATERIAL_NAME = args.material_name
    base.SIO2_MATERIAL = SIO2_MATERIAL

    # Let the audited base main own artifact hashing/provenance and Q extraction.
    base.parse_args = lambda: args
    base.add_geometry_and_monitors = lambda fdtd, model, parsed: add_geometry_and_monitors(base, fdtd, model, parsed)
    base.assert_pre_run_contract = lambda fdtd, runtime, parsed, setup: assert_contract(base, fdtd, runtime, parsed, setup)
    base.plot_geometry = plot_geometry

    original_run_case = base.run_case

    def gpu_only_case(fdtd: Any, runtime: Any, parsed: argparse.Namespace, output: Path, setup: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
        original = runtime.run_session
        runtime.run_session = strict_gpu_run
        try:
            if parsed.execution_contract in (
                "diagnostic-smoke",
                "edge-isolation-smoke",
            ):
                result = run_diagnostic_gpu_smoke_case(
                    base,
                    fdtd,
                    runtime,
                    parsed,
                    output,
                    setup,
                    contract,
                )
            else:
                result = original_run_case(
                    fdtd,
                    runtime,
                    parsed,
                    output,
                    setup,
                    contract,
                )
            auto_shutoff = final_logged_auto_shutoff(output)
            result["auto_shutoff"] = auto_shutoff
            result.setdefault("acceptance", {})[
                "solver_log_reports_successful_completion"
            ] = auto_shutoff["simulation_completed_successfully"]
            result["acceptance"][
                "auto_shutoff_reached_requested_threshold"
            ] = auto_shutoff["final_value"] <= parsed.auto_shutoff_min
            result["native_Yee_mesh_audit"] = post_run_native_mesh_audit(
                base,
                fdtd,
                output,
                result,
                setup["inner_faces"],
                setup["geometry"][
                    "pabs_nominal_control_volume_bounds_m"
                ],
            )
            if parsed.execution_contract in (
                "diagnostic-smoke",
                "edge-isolation-smoke",
            ):
                result["acceptance"][
                    "native_solver_and_component_Yee_coordinates_saved"
                ] = bool(
                    result["native_Yee_mesh_audit"].get(
                        "coordinate_artifact"
                    )
                )
            result["material_epsilon_readback"] = material_epsilon_readback(
                fdtd,
                parsed,
                setup["geometry"]["material_contract"],
                dt_s=float(contract["material"]["epsilon_readback"]["dt_s"]),
            )
            return result
        finally:
            runtime.run_session = original

    base.run_case = gpu_only_case
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
