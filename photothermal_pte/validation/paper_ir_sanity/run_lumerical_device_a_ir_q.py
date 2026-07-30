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
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath
import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
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
        choices=("device-a-polygon", "straight-45-edge"),
        default="device-a-polygon",
    )
    parser.add_argument("--domain-um", type=float, default=44.0)
    parser.add_argument("--pml-layers", type=int, default=24)
    parser.add_argument("--flake-dz-nm", type=float, default=5.0)
    parser.add_argument("--simulation-time-ps", type=float, default=1.2)
    parser.add_argument("--auto-shutoff-min", type=float, default=1.0e-5)
    parser.add_argument("--source-span-um", type=float, default=32.0)
    parser.add_argument("--waist-um", type=float, default=6.5)
    parser.add_argument("--beam-x-um", type=float, default=0.0)
    parser.add_argument("--beam-y-um", type=float, default=0.0)
    parser.add_argument("--incident-reference")
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
    if args.domain_um < 40.0:
        parser.error("Device-A optical domain must be at least 40 um")
    if args.source_span_um >= args.domain_um - 2.0:
        parser.error("source aperture needs at least 1 um PML clearance per side")
    if (
        args.waist_um <= 0
        or args.flake_dz_nm <= 0
        or args.simulation_time_ps <= 0
        or args.auto_shutoff_min <= 0
    ):
        parser.error("waist, flake dz, simulation time, and shutoff must be positive")
    if args.case == "finite-flake" and not args.incident_reference:
        parser.error("finite-flake requires a matching empty-stack reference")
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
    else:
        # TaIrTe4 occupies y <= x.  The other two triangle faces remain 1 um
        # beyond the FDTD x/y bounds, so only one straight 45-degree edge is
        # present in the physical calculation region.
        outer = 0.5 * args.domain_um + 1.0
        args.flake_vertices_um = np.asarray(
            [[-outer, -outer], [outer, -outer], [outer, outer]],
            dtype=float,
        )
        # The metallic-axis edge launches a longer lateral absorption tail than
        # the incident aperture.  Four micrometres of analysis padding is the
        # minimum matched box that is rechecked by the six-face closure gate.
        half_analysis = 0.5 * args.source_span_um + 4.0
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
        "maximum_step_m": float(np.max(steps)),
    }


def post_run_native_mesh_audit(
    base: Any,
    fdtd: Any,
    output: Path,
    run_result: dict[str, Any],
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
    delta = {
        axis: np.asarray(
            fdtd.getdata(base.PABS_FIELD, f"delta_{axis}", 1),
            float,
        ).reshape(-1)
        for axis in "xyz"
    }
    component_coordinates: dict[str, dict[str, np.ndarray]] = {}
    for component in "xyz":
        coordinates = {
            axis: np.array(field_common[axis], copy=True)
            for axis in "xyz"
        }
        coordinates[component] = coordinates[component] + delta[component]
        component_coordinates[component] = coordinates

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
        **{f"delta_{axis}_m": values for axis, values in delta.items()},
        **{
            f"E{component}_{axis}_m": values
            for component, coordinates in component_coordinates.items()
            for axis, values in coordinates.items()
        },
        **{
            f"index_{component}_{axis}_m": values
            for component, coordinates in component_coordinates.items()
            for axis, values in coordinates.items()
        },
    )

    hotspot = run_result.get("Q_hotspot", {})
    local_step: dict[str, Any] = {}
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

    component_summary: dict[str, Any] = {}
    for component, coordinates in component_coordinates.items():
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
        component_summary[component] = {
            "field_coordinates": {
                axis: _coordinate_summary(values)
                for axis, values in coordinates.items()
            },
            "permittivity_coordinates": {
                axis: _coordinate_summary(values)
                for axis, values in coordinates.items()
            },
            "electric_field_shape": electric_shape,
            "permittivity_shape": index_shape,
            "field_permittivity_shape_pairing_exact": (
                electric_shape == index_shape
            ),
            "field_permittivity_maximum_coordinate_mismatch_m": 0.0,
            "staggering_axis": component,
            "maximum_coordinate_mismatch_from_common_m": {
                axis: float(
                    np.max(np.abs(values - field_common[axis]))
                )
                for axis, values in coordinates.items()
            },
        }
    return {
        "native_solver_mesh": {
            axis: _coordinate_summary(values)
            for axis, values in solver.items()
        },
        "hotspot_neighbourhood_solver_steps": local_step,
        "component_specific_Yee_coordinates": component_summary,
        "index_monitor_common_coordinates": {
            axis: _coordinate_summary(values)
            for axis, values in index_common.items()
        },
        "common_Q_output_coordinates": {
            axis: _coordinate_summary(values)
            for axis, values in field_common.items()
        },
        "component_to_common_Q_contract": (
            "pabs_adv E/index component loss is evaluated on each native "
            "component grid x+delta_x, y+delta_y, or z+delta_z and then "
            "linearly interpolated to the field-monitor common x/y/z grid"
        ),
        "mesh_override_contract": {
            "x": "auto non-uniform; no override",
            "y": "auto non-uniform; no override",
            "z": "flake-region override only",
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
        polygon = fdtd.addpoly()
        polygon["name"] = "TaIrTe4_flake"
        polygon["vertices"] = vertices_um * 1e-6
        polygon["z min"] = -FLAKE_THICKNESS_M
        polygon["z max"] = 0.0
        polygon["material"] = args.material_name

    mesh = fdtd.addmesh()
    mesh["name"] = "flake_mesh"
    mesh["x min"] = absorption_bounds["x"][0] - 0.5e-6
    mesh["x max"] = absorption_bounds["x"][1] + 0.5e-6
    mesh["y min"] = absorption_bounds["y"][0] - 0.5e-6
    mesh["y max"] = absorption_bounds["y"][1] + 0.5e-6
    mesh["z min"] = -FLAKE_THICKNESS_M - 10 * args.flake_dz_nm * 1e-9
    mesh["z max"] = 10 * args.flake_dz_nm * 1e-9
    mesh["override x mesh"] = 0
    mesh["override y mesh"] = 0
    mesh["override z mesh"] = 1
    mesh["dz"] = args.flake_dz_nm * 1e-9

    source = fdtd.addgaussian()
    source["name"] = base.SOURCE_NAME
    source["injection axis"] = "z"
    source["direction"] = "backward"
    source["polarization angle"] = float(args.polarization_deg)
    source["source shape"] = "Gaussian"
    source["use scalar approximation"] = True
    source["beam parameters"] = "Waist size and position"
    source["waist radius w0"] = args.waist_um * 1e-6
    source["distance from waist"] = SOURCE_Z_M - FOCUS_Z_M
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
    pabs["x"] = 0.5 * sum(absorption_bounds["x"])
    pabs["x span"] = (
        absorption_bounds["x"][1]
        - absorption_bounds["x"][0]
        + 2 * PABS_PADDING_M
    )
    pabs["y"] = 0.5 * sum(absorption_bounds["y"])
    pabs["y span"] = (
        absorption_bounds["y"][1]
        - absorption_bounds["y"][0]
        + 2 * PABS_PADDING_M
    )
    pabs["z"] = -0.5 * FLAKE_THICKNESS_M
    pabs["z span"] = FLAKE_THICKNESS_M + 2 * PABS_PADDING_M

    inner_faces = base.add_flux_box(fdtd, "paper_ir_abs", inner_box)
    outer_faces = base.add_flux_box(fdtd, "paper_ir_outer", outer_bounds)
    base.add_field_monitor(
        fdtd,
        base.INCIDENT_REFERENCE_MONITOR,
        "2D Z-normal",
        {"x": source_bounds["x"], "y": source_bounds["y"], "z": (INCIDENT_Z_M, INCIDENT_Z_M)},
    )
    base.add_field_monitor(
        fdtd,
        "finite_E_xy_inside",
        "2D Z-normal",
        {"x": inner_box["x"], "y": inner_box["y"], "z": (0.5e-6, 0.5e-6)},
    )
    base.add_field_monitor(
        fdtd,
        "finite_E_yz_outside_x",
        "2D X-normal",
        {"x": (outer_bounds["x"][1], outer_bounds["x"][1]), "y": outer_bounds["y"], "z": outer_bounds["z"]},
    )
    base.add_field_monitor(
        fdtd,
        "finite_E_xz_outside_y",
        "2D Y-normal",
        {"x": outer_bounds["x"], "y": (outer_bounds["y"][1], outer_bounds["y"][1]), "z": outer_bounds["z"]},
    )
    for name in (
        base.PABS_FIELD,
        base.PABS_INDEX,
        *(face["name"] for face in inner_faces.values()),
        *(face["name"] for face in outer_faces.values()),
        base.INCIDENT_REFERENCE_MONITOR,
        "finite_E_xy_inside",
        "finite_E_yz_outside_x",
        "finite_E_xz_outside_y",
    ):
        base.configure_single_frequency(fdtd, name)

    geometry = {
        "geometry_name": args.geometry,
        "geometry_source": (
            "approximation digitized from paper Figure 2A, not exact CAD"
            if args.geometry == "device-a-polygon"
            else (
                "paper Figure 3F straight 45-degree half-plane control; "
                "TaIrTe4 occupies lab y<=x and remote triangle faces lie "
                "outside the PML-bounded physical domain"
            )
        ),
        "coordinate_contract": {"lab_x": "crystal b", "lab_y": "crystal a"},
        "flake_vertices_um": vertices_um.tolist(),
        "flake_thickness_m": FLAKE_THICKNESS_M,
        "flake_area_m2": polygon_area(vertices_um) * 1e-12,
        "absorption_analysis_bounds_m": absorption_bounds,
        "exact_flake_mask_kind": (
            "digitized polygon"
            if args.geometry == "device-a-polygon"
            else "analytic half-plane lab_y<=lab_x"
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
            "waist_radius_m": args.waist_um * 1e-6,
            "beam_center_m": [beam_x, beam_y],
            "source_span_m": args.source_span_um * 1e-6,
            "normal_incidence": True,
            "polarization_axis": args.polarization,
            "simulation_time_s": args.simulation_time_ps * 1e-12,
            "auto_shutoff_min": args.auto_shutoff_min,
        },
        "domain_bounds_m": {
            "x": [-0.5 * domain_m, 0.5 * domain_m],
            "y": [-0.5 * domain_m, 0.5 * domain_m],
            "z": [FDTD_Z_MIN_M, FDTD_Z_MAX_M],
        },
        "all_six_boundaries": "PML",
        "periodic": False,
        "material_contract": material_contract,
    }
    def exact_flake_mask_builder(
        x_m: np.ndarray,
        y_m: np.ndarray,
        z_m: np.ndarray,
    ) -> np.ndarray:
        if args.geometry == "straight-45-edge":
            xy = y_m[None, :] <= x_m[:, None] + 1e-15
        else:
            xx, yy = np.meshgrid(x_m, y_m, indexing="ij")
            xy = PolygonPath(vertices_um * 1e-6).contains_points(
                np.column_stack((xx.ravel(), yy.ravel())),
                radius=1e-15,
            ).reshape(xx.shape)
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
    checks["finite_polygon_count"] = int(fdtd.getnamednumber("TaIrTe4_flake")) == (
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
        args.waist_um * 1e-6,
        atol=1e-15,
    )
    checks["correct_dz"] = np.isclose(
        base.scalar(fdtd.getnamed("flake_mesh", "dz"), "flake dz"),
        args.flake_dz_nm * 1e-9,
        atol=1e-15,
    )
    checks["correct_thickness"] = args.case == "empty-stack" or np.isclose(
        base.scalar(fdtd.getnamed("TaIrTe4_flake", "z span"), "flake thickness"),
        FLAKE_THICKNESS_M,
        atol=1e-15,
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
            "flake_dz_m": args.flake_dz_nm * 1e-9,
        },
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
    ax.fill(vertices[:, 0], vertices[:, 1], color="#d89023", alpha=0.75, label="130 nm TaIrTe4")
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
            else "Corner-free straight 45-degree edge control"
        ),
    )
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.axhspan(FDTD_Z_MIN_M * 1e6, -(FLAKE_THICKNESS_M + SIO2_THICKNESS_M) * 1e6, color="silver", label="Si")
    ax.axhspan(-(FLAKE_THICKNESS_M + SIO2_THICKNESS_M) * 1e6, -FLAKE_THICKNESS_M * 1e6, color="lightblue", label="285 nm SiO2")
    ax.axhspan(-FLAKE_THICKNESS_M * 1e6, 0, color="#d89023", label="130 nm TaIrTe4")
    ax.axhline(SOURCE_Z_M * 1e6, color="tab:green", ls="--", label="Gaussian source")
    ax.plot([0], [FOCUS_Z_M * 1e6], marker="x", color="tab:green", label="focus")
    ax.set(xlabel="lateral (schematic)", ylabel="z (um)", title="Normal-incidence stack")
    ax.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "paper_ir_device_a_geometry.png", dpi=200)
    plt.close(figure)


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
            result = original_run_case(
                fdtd,
                runtime,
                parsed,
                output,
                setup,
                contract,
            )
            result["native_Yee_mesh_audit"] = post_run_native_mesh_audit(
                base,
                fdtd,
                output,
                result,
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
