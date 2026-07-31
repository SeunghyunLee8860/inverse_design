#!/usr/bin/env python3
"""Compare 50-nm Maxwell and analytic sources in the paper-reduced model.

This command does not launch Lumerical.  It consumes the two completed
``w0=12 um`` straight-edge optical artifacts, conservatively maps their raw
Maxwell absorption to one common sheet grid, constructs the paper
Gaussian--Beer--Lambert source at the measured target-plane beam position and
width, and solves the exact thickness-integrated form of Supplement Eq. S4.

The result is a paper-like sanity comparison.  The assumed 12-um waist is not
published by the paper, and the output is not an experimental reproduction or
an optical mesh-convergence certificate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
THERMAL_PATH = HERE / "run_device_a_explicit_thermal_pte.py"
ANALYTIC_PATH = HERE / "run_straight_edge_analytic_q_control.py"
ROBUST_PATH = HERE / "audit_straight_edge_robust_gradient.py"
RUNNER_PATH = HERE / "run_lumerical_device_a_ir_q.py"

STATUS_PASS = (
    "COMPLETED_W12_50NM_MAXWELL_VS_ANALYTIC_"
    "PAPER_REDUCED_THERMAL_SANITY"
)
STATUS_BLOCKED = "BLOCKED_W12_50NM_MAXWELL_ANALYTIC_OPTICAL_GATE"
INCIDENT_POWER_W = 285.0e-6
WAVELENGTH_M = 11.0e-6
THICKNESS_M = 130.0e-9
TMM_ABSORPTION = {"a": 0.17673296, "b": 0.26328721}
SQRT2 = np.sqrt(2.0)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


thermal = load_module("w12_comparison_thermal", THERMAL_PATH)
analytic_base = load_module("w12_comparison_analytic", ANALYTIC_PATH)
robust = load_module("w12_comparison_robust", ROBUST_PATH)
runner = load_module("w12_comparison_runner", RUNNER_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-a-dir", type=Path, required=True)
    parser.add_argument("--edge-b-dir", type=Path, required=True)
    parser.add_argument("--incident-reference-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument(
        "--thermal-step-nm",
        type=float,
        default=50.0,
        help="Common paper-reduced sheet-grid step; not an optical refinement.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        text=True,
    ).strip()


@dataclass
class OpticalInput:
    polarization: str
    directory: Path
    result_path: Path
    q_path: Path
    fsp_path: Path
    manifest_path: Path
    result: dict[str, Any]
    q_summary: dict[str, Any]
    gates: dict[str, bool]
    artifacts: list[dict[str, Any]]


def inspect_optical(directory: Path, polarization: str) -> OpticalInput:
    result_path = directory / "case_result.json"
    q_path = directory / "finite_q_on_artifact.npz"
    manifest_path = directory / "RAW_ARTIFACT_MANIFEST.json"
    for path in (result_path, q_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = json.loads(result_path.read_text())
    fsp_path = directory / "finite_2um_optical_q.fsp"
    if not fsp_path.is_file():
        fsp_path = Path(result["project"]).expanduser().resolve()
    if not fsp_path.is_file():
        raise FileNotFoundError(fsp_path)
    run = result["run_result"]
    component_power = {
        axis: float(run["component_power_W"][axis]) for axis in "xyz"
    }
    with np.load(q_path, allow_pickle=False) as raw:
        q = np.asarray(raw["Q_on_W_m3"], float)
        qx = np.asarray(raw["Qx_W_m3"], float)
        qy = np.asarray(raw["Qy_W_m3"], float)
        qz = np.asarray(raw["Qz_W_m3"], float)
        finite = all(np.all(np.isfinite(value)) for value in (q, qx, qy, qz))
        negative_count = int(np.count_nonzero(q < 0.0))
        q_shape = list(q.shape)
        x = np.asarray(raw["x_m"], float)
        y = np.asarray(raw["y_m"], float)
        z = np.asarray(raw["z_m"], float)
        stored_pq = float(np.asarray(raw["P_abs_volume_W"]).reshape(-1)[0])
        stored_psix = float(
            np.asarray(raw["P_abs_six_face_W"]).reshape(-1)[0]
        )
    source = result["pre_run_contract"]["geometry"]["source"]
    mesh = result["pre_run_contract"]["mesh"]
    native = run["native_Yee_mesh_audit"]
    pairing = native["independent_field_index_pairing"]
    expected_axis = polarization
    contract = {
        "status_completed": result.get("status") == "COMPLETED",
        "polarization": source["polarization_axis"] == expected_axis,
        "wavelength_11um": np.isclose(
            result["pre_run_contract"]["geometry"]["source"]["wavelength_m"],
            WAVELENGTH_M,
        ),
        "scalar_Gaussian": result.get("source_type") == "Gaussian",
        "waist_12um": np.isclose(result["waist_um"], 12.0),
        "source_span_50um": np.isclose(result["source_span_um"], 50.0),
        "domain_60um": np.isclose(result["domain_um"], 60.0),
        "six_PML_nonperiodic": (
            all(
                value == "PML"
                for value in result["pre_run_contract"]["boundaries"].values()
            )
            and result["pre_run_contract"]["checks"]["no_periodic_boundary"]
        ),
        "flake_thickness_130nm": np.isclose(
            result["pre_run_contract"]["geometry"]["flake_thickness_m"],
            THICKNESS_M,
        ),
        "substrate_285nm_SiO2_on_Si": (
            result["pre_run_contract"]["geometry"]["substrate"]
            == "285 nm SiO2 on Si"
        ),
        "epsilon_c_equals_b": result["pre_run_contract"]["material"][
            "epsilon_readback"
        ]["epsilon_z_equals_epsilon_b_contract"],
        "axis_mapping_x_b_y_a": (
            result["pre_run_contract"]["geometry"]["coordinate_contract"]
            == {"lab_x": "crystal b", "lab_y": "crystal a"}
        ),
        "straight_45_edge": (
            result["pre_run_contract"]["geometry"]["geometry_name"]
            == "straight-45-edge"
        ),
        "local_xy_50nm": np.isclose(mesh["local_xy_mesh_m"], 50.0e-9),
        "flake_dz_5nm": np.isclose(mesh["flake_dz_m"], 5.0e-9),
        "GPU_only_resource": str(run["resource"]).lower() != "cpu",
        "raw_Q_unmodified": (
            result["Q_clipped"] is False
            and result["Q_rescaled"] is False
            and result["flux_gain"] is False
            and result["periodic_Q_used"] is False
        ),
    }
    contract = {key: bool(value) for key, value in contract.items()}
    gates = {
        "contract_exact": all(contract.values()),
        "closure_lt_0p5_percent": (
            float(run["six_face_relative_closure"]) < 0.005
        ),
        "auto_shutoff_le_1e_minus_5": (
            float(run["auto_shutoff"]["final_value"]) <= 1.0e-5
        ),
        "solver_completed": bool(
            run["auto_shutoff"]["simulation_completed_successfully"]
        ),
        "finite_Q": bool(finite),
        "negative_Q_voxel_count_zero": negative_count == 0,
        "field_index_shapes_match": bool(
            pairing["all_component_shapes_match"]
        ),
        "field_index_coordinate_mismatch_roundoff": (
            float(pairing["maximum_coordinate_mismatch_m"]) < 1.0e-15
        ),
        "artifact_PQ_matches_result": np.isclose(
            stored_pq,
            float(run["P_Q_W"]),
            rtol=1.0e-12,
            atol=0.0,
        ),
        "artifact_Psix_matches_result": np.isclose(
            stored_psix,
            float(run["P_six_face_W"]),
            rtol=1.0e-12,
            atol=0.0,
        ),
    }
    gates = {key: bool(value) for key, value in gates.items()}
    gates["all_before_remap"] = all(gates.values())
    q_summary = {
        "P_Q_W_at_1_W_m2_central_intensity": float(run["P_Q_W"]),
        "P_six_W_at_1_W_m2_central_intensity": float(
            run["P_six_face_W"]
        ),
        "six_face_relative_closure": float(
            run["six_face_relative_closure"]
        ),
        "Q_component_power_W_at_1_W_m2": component_power,
        "Q_component_fraction": {
            axis: value / float(run["P_Q_W"])
            for axis, value in component_power.items()
        },
        "negative_Q_voxel_count": negative_count,
        "minimum_Q_W_m3": float(run["minimum_Q_W_m3"]),
        "Q_shape": q_shape,
        "Q_coordinate_bounds_m": {
            "x": [float(x[0]), float(x[-1])],
            "y": [float(y[0]), float(y[-1])],
            "z": [float(z[0]), float(z[-1])],
        },
        "incident_power_W_at_1_W_m2": float(
            run["normalization"]["incident_power_W_at_1_W_m2"]
        ),
        "physical_incident_power_W": INCIDENT_POWER_W,
        "physical_power_scale": (
            INCIDENT_POWER_W
            / float(run["normalization"]["incident_power_W_at_1_W_m2"])
        ),
        "auto_shutoff_final": float(run["auto_shutoff"]["final_value"]),
        "native_field_index_pairing": pairing,
        "component_specific_Yee_coordinates": native[
            "component_specific_Yee_coordinates"
        ],
        "contract_checks": contract,
    }
    artifacts = [
        artifact_record(result_path, f"edge_{polarization}_case_result"),
        artifact_record(q_path, f"edge_{polarization}_raw_Q_NPZ"),
        artifact_record(fsp_path, f"edge_{polarization}_FSP"),
        artifact_record(manifest_path, f"edge_{polarization}_manifest"),
    ]
    return OpticalInput(
        polarization,
        directory,
        result_path,
        q_path,
        fsp_path,
        manifest_path,
        result,
        q_summary,
        gates,
        artifacts,
    )


def measured_beam(reference_path: Path) -> dict[str, Any]:
    with np.load(reference_path, allow_pickle=False) as raw:
        x = np.asarray(raw["x_m"], float)
        y = np.asarray(raw["y_m"], float)
        intensity = np.asarray(raw["downward_intensity_W_m2"], float)
    fitted = runner.fit_elliptical_gaussian(x, y, intensity)
    return {
        "reference_artifact": artifact_record(
            reference_path,
            "empty_stack_target_plane_incident_reference",
        ),
        "fit": fitted,
        "requested_waist_m": 12.0e-6,
        "waist_convention": (
            "Lumerical scalar-Gaussian 1/e^2 intensity radius; in the "
            "paper form exp(-r^2/(2 sigma^2)), w0=2 sigma"
        ),
        "evidence_scope": (
            "empty-layered-stack downward E/H decomposition at z=+50 nm; "
            "the same stored target-plane profile is used by both analytic "
            "polarizations"
        ),
    }


def common_sheet_geometry(
    bounds_m: dict[str, list[float]],
    step_m: float,
) -> Any:
    axes = []
    for axis in ("x", "y"):
        lo, hi = map(float, bounds_m[axis])
        count = int(round((hi - lo) / step_m))
        if count <= 0 or not np.isclose(count * step_m, hi - lo):
            raise ValueError(f"{axis} bounds are not divisible by step")
        axes.append(np.linspace(lo, hi, count + 1))
    x_edges, y_edges = axes
    z_edges = np.asarray([-THICKNESS_M, 0.0])
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    flake_xy = y[None, :] <= x[:, None]
    flake = flake_xy[:, :, None]
    material = np.zeros(flake.shape, np.uint8)
    material[flake] = 3
    kappa = np.ones((*flake.shape, 3), float)
    kappa[flake] = thermal.KAPPA_TAIRTE4_LAB_W_MK
    resistances = {
        "x": np.zeros((flake.shape[0] - 1, flake.shape[1], 1)),
        "y": np.zeros((flake.shape[0], flake.shape[1] - 1, 1)),
        "z": np.zeros((flake.shape[0], flake.shape[1], 0)),
    }
    return thermal.Geometry(
        x_edges,
        y_edges,
        z_edges,
        material,
        flake,
        kappa,
        resistances,
    )


def source_coordinate_bounds(
    optical: OpticalInput,
) -> dict[str, list[float]]:
    with np.load(optical.q_path, allow_pickle=False) as raw:
        return {
            axis: [
                float(np.asarray(raw[f"{axis}_m"], float)[0]),
                float(np.asarray(raw[f"{axis}_m"], float)[-1]),
            ]
            for axis in "xyz"
        }


def embedding_geometry(sheet: Any, source_bounds: dict[str, list[float]]) -> Any:
    z_edges = np.asarray(
        sorted(
            {
                float(source_bounds["z"][0]),
                -THICKNESS_M,
                0.0,
                float(source_bounds["z"][1]),
            }
        ),
        float,
    )
    if (
        z_edges[0] > source_bounds["z"][0]
        or z_edges[-1] < source_bounds["z"][1]
        or not np.any(np.isclose(z_edges, -THICKNESS_M))
        or not np.any(np.isclose(z_edges, 0.0))
    ):
        raise RuntimeError("embedding z grid does not contain source")
    shape = (
        sheet.x_edges_m.size - 1,
        sheet.y_edges_m.size - 1,
        z_edges.size - 1,
    )
    z = 0.5 * (z_edges[:-1] + z_edges[1:])
    flake_z = (z > -THICKNESS_M) & (z < 0.0)
    if np.count_nonzero(flake_z) != 1:
        raise RuntimeError("embedding grid must have exactly one sheet cell")
    flake_xy = np.any(sheet.flake_mask, axis=2)
    flake = flake_xy[:, :, None] & flake_z[None, None, :]
    material = np.zeros(shape, np.uint8)
    material[flake] = 3
    kappa = np.ones((*shape, 3), float)
    kappa[flake] = thermal.KAPPA_TAIRTE4_LAB_W_MK
    resistances = {
        "x": np.zeros((shape[0] - 1, shape[1], shape[2])),
        "y": np.zeros((shape[0], shape[1] - 1, shape[2])),
        "z": np.zeros((shape[0], shape[1], shape[2] - 1)),
    }
    return thermal.Geometry(
        sheet.x_edges_m,
        sheet.y_edges_m,
        z_edges,
        material,
        flake,
        kappa,
        resistances,
    )


def map_Maxwell_q_to_sheet(
    optical: OpticalInput,
    sheet: Any,
    source_bounds: dict[str, list[float]],
) -> tuple[np.ndarray, dict[str, Any]]:
    embedding = embedding_geometry(sheet, source_bounds)
    mapped, audit = thermal.load_and_map_q(
        optical.q_path,
        optical.result_path,
        embedding,
    )
    flake_z = np.flatnonzero(np.any(embedding.flake_mask, axis=(0, 1)))
    if flake_z.size != 1:
        raise RuntimeError("mapped embedding has no unique sheet cell")
    q_sheet = mapped[:, :, flake_z[0] : flake_z[0] + 1]
    power_sheet = integrate_q(q_sheet, sheet)
    audit.update(
        {
            "embedding_target_z_edges_m": embedding.z_edges_m.tolist(),
            "sheet_extraction_z_index": int(flake_z[0]),
            "P_Q_sheet_W": power_sheet,
            "embedding_to_sheet_relative_power_error": abs(
                power_sheet - audit["P_Q_target_W"]
            )
            / abs(audit["P_Q_target_W"]),
            "two_stage_contract": (
                "the full stored common-Q coordinate box is first contained "
                "by a 3D target; nearest-support projection preserves every "
                "source-cell energy in the sole -130..0-nm TaIrTe4 cell; "
                "that cell is then viewed as the exact sheet source"
            ),
        }
    )
    return q_sheet, audit


def assemble_sheet_system(geometry: Any) -> Any:
    return thermal.assemble_steady_diagonal_kappa(
        x_edges_m=geometry.x_edges_m,
        y_edges_m=geometry.y_edges_m,
        z_edges_m=geometry.z_edges_m,
        kappa_W_mK=geometry.kappa_W_mK,
        dirichlet_temperature_K={},
        interface_resistance_m2K_W=geometry.interface_resistance_m2K_W,
        active_mask=geometry.flake_mask,
        surface_robin_heat_transfer_W_m2K={
            "z_min": thermal.G_TAIRTE4_SIO2_W_M2K,
            "z_max": thermal.G_TAIRTE4_AIR_W_M2K,
        },
        surface_robin_temperature_K={"z_min": 0.0, "z_max": 0.0},
    )


def optical_constants(polarization: str) -> dict[str, Any]:
    constants = analytic_base.optical_constants(polarization)
    return {
        **constants,
        "TMM_full_plane_absorption": TMM_ABSORPTION[polarization],
        "TMM_provenance": (
            "existing repository paper/TMM provenance for 130-nm TaIrTe4 "
            "on 285-nm SiO2/Si at 11 um; not re-estimated here"
        ),
    }


def analytic_sheet_q(
    geometry: Any,
    polarization: str,
    beam: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    fit = beam["fit"]
    x = 0.5 * (geometry.x_edges_m[:-1] + geometry.x_edges_m[1:])
    y = 0.5 * (geometry.y_edges_m[:-1] + geometry.y_edges_m[1:])
    wx = float(fit["waist_x_m"])
    wy = float(fit["waist_y_m"])
    x0 = float(fit["center_x_m"])
    y0 = float(fit["center_y_m"])
    lateral_per_m2 = 2.0 / (np.pi * wx * wy) * np.exp(
        -2.0
        * (
            (x[:, None] - x0) ** 2 / wx**2
            + (y[None, :] - y0) ** 2 / wy**2
        )
    )
    q_areal = (
        INCIDENT_POWER_W
        * TMM_ABSORPTION[polarization]
        * lateral_per_m2
    )
    flake_xy = np.any(geometry.flake_mask, axis=2)
    q_areal = np.where(flake_xy, q_areal, 0.0)
    q = q_areal[:, :, None] / THICKNESS_M
    area = (
        np.diff(geometry.x_edges_m)[:, None]
        * np.diff(geometry.y_edges_m)[None, :]
    )
    constants = optical_constants(polarization)
    return q, {
        "equation": (
            "P_inc*A_p*f_Gaussian(x,y)*f_depth(z); the normalized "
            "Beer-Lambert depth law integrates to one in this exact "
            "thickness-integrated sheet equation"
        ),
        "polarization": polarization,
        "incident_power_W": INCIDENT_POWER_W,
        "TMM_absorption": TMM_ABSORPTION[polarization],
        "requested_full_plane_absorbed_power_W": (
            INCIDENT_POWER_W * TMM_ABSORPTION[polarization]
        ),
        "finite_half_plane_absorbed_power_W": float(
            np.sum(q_areal * area)
        ),
        "beam_center_m": [x0, y0],
        "realized_waist_x_m": wx,
        "realized_waist_y_m": wy,
        "requested_waist_m": 12.0e-6,
        "depth_law": {
            "kind": "normalized Beer-Lambert over 130 nm",
            "beta_m_inv": constants["beta_m_inv"],
            "penetration_depth_m": constants["penetration_depth_m"],
            "absorbed_depth_fraction_over_130nm": constants[
                "absorbed_depth_fraction_over_130nm"
            ],
        },
        "half_plane_discretization": (
            "common 50-nm Cartesian active-cell support with cell centres "
            "satisfying y<=x; this is the same stair-step material support "
            "used for both Maxwell and analytic thermal inputs"
        ),
    }


def integrate_q(q: np.ndarray, geometry: Any) -> float:
    volume = (
        np.diff(geometry.x_edges_m)[:, None, None]
        * np.diff(geometry.y_edges_m)[None, :, None]
        * np.diff(geometry.z_edges_m)[None, None, :]
    )
    return float(np.sum(np.asarray(q, float) * volume))


def weighted_nrmse(
    value: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    weight: np.ndarray | float = 1.0,
) -> float:
    a = np.asarray(value, float)
    b = np.asarray(reference, float)
    selected = np.asarray(mask, bool) & np.isfinite(a) & np.isfinite(b)
    weights = np.broadcast_to(np.asarray(weight, float), a.shape)
    numerator = float(np.sum(weights[selected] * (a[selected] - b[selected]) ** 2))
    denominator = float(np.sum(weights[selected] * b[selected] ** 2))
    return float(
        np.sqrt(numerator / max(denominator, np.finfo(float).tiny))
    )


def source_moments(q_areal: np.ndarray, geometry: Any) -> dict[str, Any]:
    x = 0.5 * (geometry.x_edges_m[:-1] + geometry.x_edges_m[1:])
    y = 0.5 * (geometry.y_edges_m[:-1] + geometry.y_edges_m[1:])
    area = (
        np.diff(geometry.x_edges_m)[:, None]
        * np.diff(geometry.y_edges_m)[None, :]
    )
    mask = np.any(geometry.flake_mask, axis=2)
    weights = np.asarray(q_areal, float) * area
    weights = np.where(mask, weights, 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("source moments need positive power")
    cx = float(np.sum(weights * x[:, None]) / total)
    cy = float(np.sum(weights * y[None, :]) / total)
    var_x = float(np.sum(weights * (x[:, None] - cx) ** 2) / total)
    var_y = float(np.sum(weights * (y[None, :] - cy) ** 2) / total)
    cov_xy = float(
        np.sum(
            weights
            * (x[:, None] - cx)
            * (y[None, :] - cy)
        )
        / total
    )
    peak_flat = int(np.argmax(np.where(mask, q_areal, -np.inf)))
    peak = np.unravel_index(peak_flat, q_areal.shape)
    return {
        "power_W": total,
        "centroid_m": {"x": cx, "y": cy},
        "second_central_moment_m2": {
            "xx": var_x,
            "yy": var_y,
            "xy": cov_xy,
        },
        "equivalent_1e2_waist_m": {
            "x": 2.0 * np.sqrt(max(var_x, 0.0)),
            "y": 2.0 * np.sqrt(max(var_y, 0.0)),
        },
        "hotspot_m": {
            "x": float(x[peak[0]]),
            "y": float(y[peak[1]]),
            "value_W_m2": float(q_areal[peak]),
        },
    }


def edge_profile(
    values: np.ndarray,
    geometry: Any,
    *,
    tangent_window_m: float = 8.0e-6,
    normal_bounds_m: tuple[float, float] = (-12.0e-6, 3.0e-6),
    bin_width_m: float = 50.0e-9,
) -> tuple[np.ndarray, np.ndarray]:
    field = np.asarray(values, float)
    if field.ndim != 2:
        raise ValueError("edge profile expects a 2D field")
    x = 0.5 * (geometry.x_edges_m[:-1] + geometry.x_edges_m[1:])
    y = 0.5 * (geometry.y_edges_m[:-1] + geometry.y_edges_m[1:])
    normal = (y[None, :] - x[:, None]) / SQRT2
    tangent = (x[:, None] + y[None, :]) / SQRT2
    mask = (
        np.any(geometry.flake_mask, axis=2)
        & (np.abs(tangent) <= tangent_window_m)
        & np.isfinite(field)
    )
    bins = np.arange(
        normal_bounds_m[0],
        normal_bounds_m[1] + bin_width_m,
        bin_width_m,
    )
    centers = 0.5 * (bins[:-1] + bins[1:])
    indices = np.digitize(normal[mask], bins) - 1
    valid = (indices >= 0) & (indices < centers.size)
    area = (
        np.diff(geometry.x_edges_m)[:, None]
        * np.diff(geometry.y_edges_m)[None, :]
    )
    selected_area = np.broadcast_to(area, field.shape)[mask][valid]
    numerator = np.bincount(
        indices[valid],
        weights=field[mask][valid] * selected_area,
        minlength=centers.size,
    )
    denominator = np.bincount(
        indices[valid],
        weights=selected_area,
        minlength=centers.size,
    )
    profile = np.full(centers.shape, np.nan)
    present = denominator > 0.0
    profile[present] = numerator[present] / denominator[present]
    return centers, profile


def profile_nrmse(a: np.ndarray, b: np.ndarray) -> float:
    selected = np.isfinite(a) & np.isfinite(b)
    return float(
        np.linalg.norm(a[selected] - b[selected])
        / max(np.linalg.norm(b[selected]), np.finfo(float).tiny)
    )


def component_metrics(
    fields: dict[str, np.ndarray],
    edge_mask: np.ndarray,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    names = {
        "x": "grad_T_x_K_m",
        "y": "grad_T_y_K_m",
        "n": "grad_T_normal_K_m",
        "t": "grad_T_tangent_K_m",
        "magnitude": "grad_T_magnitude_K_m",
    }
    for label, name in names.items():
        selected = np.abs(np.asarray(fields[name], float)[edge_mask])
        output[label] = {
            "raw_max_abs_K_m": float(np.max(selected)),
            "raw_p99_abs_K_m": float(np.percentile(selected, 99.0)),
        }
    return output


def robust_metrics(
    geometry: Any,
    temperature: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    x = 0.5 * (geometry.x_edges_m[:-1] + geometry.x_edges_m[1:])
    y = 0.5 * (geometry.y_edges_m[:-1] + geometry.y_edges_m[1:])
    fitted = robust.quadratic_edge_fit(
        x,
        y,
        temperature,
        robust.N_BANDS_UM["primary"],
    )
    components = {
        "x": fitted["dT_dx_K_m"],
        "y": fitted["dT_dy_K_m"],
        "n": fitted["dT_dn_K_m"],
        "t": fitted["dT_dt_K_m"],
        "magnitude": np.hypot(
            fitted["dT_dx_K_m"],
            fitted["dT_dy_K_m"],
        ),
    }
    metrics = {
        label: robust.aggregate(fitted["t_m"], value)
        for label, value in components.items()
    }
    metrics["fit_relative_residual_p99"] = float(
        np.percentile(fitted["fit_relative_residual"], 99.0)
    )
    return metrics, {
        "robust_t_m": fitted["t_m"],
        **{f"robust_grad_{key}_K_m": value for key, value in components.items()},
        "robust_fit_relative_residual": fitted["fit_relative_residual"],
    }


def solve_case(
    system: Any,
    geometry: Any,
    q: np.ndarray,
    *,
    source_model: str,
    polarization: str,
    normalization: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    solved = thermal.solve_assembled_thermal_system(
        system,
        source_W_m3=q,
        relative_tolerance=1.0e-10,
        max_iterations=12000,
    )
    straight, fields = thermal.straight_edge_temperature_metrics(
        solved.temperature_K,
        geometry,
    )
    raw = component_metrics(fields, fields["edge_window_mask"])
    robust_summary, robust_fields = robust_metrics(
        geometry,
        fields["temperature_flake_average_K"],
    )
    x = 0.5 * (geometry.x_edges_m[:-1] + geometry.x_edges_m[1:])
    y = 0.5 * (geometry.y_edges_m[:-1] + geometry.y_edges_m[1:])
    flake_xy = np.any(geometry.flake_mask, axis=2)
    temperature = fields["temperature_flake_average_K"]
    temperature_peak_flat = int(
        np.argmax(np.where(flake_xy, temperature, -np.inf))
    )
    temperature_peak = np.unravel_index(
        temperature_peak_flat,
        temperature.shape,
    )
    q_areal = np.asarray(q[:, :, 0], float) * THICKNESS_M
    q_n, q_profile = edge_profile(q_areal, geometry)
    t_n, t_profile = edge_profile(
        fields["temperature_flake_average_K"],
        geometry,
    )
    g_n, g_profile = edge_profile(
        fields["grad_T_magnitude_K_m"],
        geometry,
    )
    if not np.array_equal(q_n, t_n) or not np.array_equal(q_n, g_n):
        raise RuntimeError("edge-profile coordinates differ")
    case = {
        "case_id": f"{source_model}_{polarization}_{normalization}",
        "source_model": source_model,
        "polarization": polarization,
        "normalization": normalization,
        "source_power_W": float(solved.source_power_W),
        "linear_residual_relative": float(
            solved.linear_residual_relative
        ),
        "energy_balance_relative_error": float(
            solved.energy_balance_relative_error
        ),
        "iterations": int(solved.iterations),
        "boundary_power_out_W": {
            key: float(value)
            for key, value in solved.boundary_power_out_W.items()
        },
        "Tmax_rise_K": straight["Tmax_rise_K"],
        "TaIrTe4_area_average_rise_K": straight[
            "TaIrTe4_area_average_rise_K"
        ],
        "temperature_hotspot_m": {
            "x": float(x[temperature_peak[0]]),
            "y": float(y[temperature_peak[1]]),
            "value_K": float(temperature[temperature_peak]),
        },
        "peak_edge_gradient_location_m": straight[
            "peak_edge_gradient_location_m"
        ],
        "raw_fixed_edge_window": raw,
        "robust_exact_edge_window": robust_summary,
        "source_spatial_moments": source_moments(q_areal, geometry),
        "paper_Fig3G_comparator_note": straight[
            "paper_Fig3G_comparator"
        ],
    }
    arrays = {
        "Q_W_m3": q,
        "Q_areal_W_m2": q_areal,
        "temperature_rise_K": fields["temperature_flake_average_K"],
        "grad_T_x_K_m": fields["grad_T_x_K_m"],
        "grad_T_y_K_m": fields["grad_T_y_K_m"],
        "grad_T_normal_K_m": fields["grad_T_normal_K_m"],
        "grad_T_tangent_K_m": fields["grad_T_tangent_K_m"],
        "grad_T_magnitude_K_m": fields["grad_T_magnitude_K_m"],
        "edge_window_mask": fields["edge_window_mask"],
        "edge_profile_n_m": q_n,
        "edge_profile_Q_areal_W_m2": q_profile,
        "edge_profile_temperature_K": t_profile,
        "edge_profile_grad_magnitude_K_m": g_profile,
        **robust_fields,
    }
    return case, arrays


def case_ratios(
    cases: dict[tuple[str, str, str], dict[str, Any]],
    source_model: str,
    normalization: str,
) -> dict[str, float]:
    a = cases[(source_model, "a", normalization)]
    b = cases[(source_model, "b", normalization)]
    return {
        "absorbed_power_b_over_a": b["source_power_W"] / a["source_power_W"],
        "Tmax_b_over_a": b["Tmax_rise_K"] / a["Tmax_rise_K"],
        "raw_max_abs_grad_x_b_over_a": (
            b["raw_fixed_edge_window"]["x"]["raw_max_abs_K_m"]
            / a["raw_fixed_edge_window"]["x"]["raw_max_abs_K_m"]
        ),
        "raw_max_abs_grad_y_b_over_a": (
            b["raw_fixed_edge_window"]["y"]["raw_max_abs_K_m"]
            / a["raw_fixed_edge_window"]["y"]["raw_max_abs_K_m"]
        ),
        "raw_max_abs_grad_n_b_over_a": (
            b["raw_fixed_edge_window"]["n"]["raw_max_abs_K_m"]
            / a["raw_fixed_edge_window"]["n"]["raw_max_abs_K_m"]
        ),
        "raw_max_grad_magnitude_b_over_a": (
            b["raw_fixed_edge_window"]["magnitude"]["raw_max_abs_K_m"]
            / a["raw_fixed_edge_window"]["magnitude"]["raw_max_abs_K_m"]
        ),
        "robust_max_abs_grad_x_b_over_a": (
            b["robust_exact_edge_window"]["x"]["maximum_abs_K_m"]
            / a["robust_exact_edge_window"]["x"]["maximum_abs_K_m"]
        ),
        "robust_max_abs_grad_n_b_over_a": (
            b["robust_exact_edge_window"]["n"]["maximum_abs_K_m"]
            / a["robust_exact_edge_window"]["n"]["maximum_abs_K_m"]
        ),
        "robust_max_grad_magnitude_b_over_a": (
            b["robust_exact_edge_window"]["magnitude"][
                "maximum_abs_K_m"
            ]
            / a["robust_exact_edge_window"]["magnitude"][
                "maximum_abs_K_m"
            ]
        ),
    }


def map_extent_um(geometry: Any) -> list[float]:
    return [
        float(geometry.x_edges_m[0] * 1.0e6),
        float(geometry.x_edges_m[-1] * 1.0e6),
        float(geometry.y_edges_m[0] * 1.0e6),
        float(geometry.y_edges_m[-1] * 1.0e6),
    ]


def masked_for_plot(values: np.ndarray, geometry: Any) -> np.ndarray:
    mask = np.any(geometry.flake_mask, axis=2)
    return np.where(mask, np.asarray(values, float), np.nan).T


def plot_triplet(
    output_path: Path,
    geometry: Any,
    arrays_a: dict[str, np.ndarray],
    arrays_b: dict[str, np.ndarray],
    field: str,
    title_prefix: str,
    colorbar_label: str,
) -> None:
    a = np.asarray(arrays_a[field], float)
    b = np.asarray(arrays_b[field], float)
    difference = a - b
    values = (a, b, difference)
    titles = (
        f"{title_prefix}: E || a",
        f"{title_prefix}: E || b",
        f"{title_prefix}: a - b",
    )
    extent = map_extent_um(geometry)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 4.7),
        constrained_layout=True,
    )
    for index, (axis, value, title) in enumerate(zip(axes, values, titles)):
        plotted = masked_for_plot(value, geometry)
        if index == 2:
            limit = float(np.nanmax(np.abs(plotted)))
            image = axis.imshow(
                plotted,
                origin="lower",
                extent=extent,
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
            )
        else:
            image = axis.imshow(
                plotted,
                origin="lower",
                extent=extent,
                cmap="inferno",
                interpolation="nearest",
            )
        axis.plot(
            [extent[0], extent[1]],
            [extent[0], extent[1]],
            "w--",
            linewidth=0.8,
        )
        axis.set(
            xlabel="lab x = crystal b (µm)",
            ylabel="lab y = crystal a (µm)",
            title=title,
            xlim=(-18, 18),
            ylim=(-18, 18),
        )
        figure.colorbar(image, ax=axis, label=colorbar_label)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_cross_model(
    output_path: Path,
    geometry: Any,
    arrays: dict[tuple[str, str, str], dict[str, np.ndarray]],
    field: str,
    colorbar_label: str,
) -> None:
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15.5, 9.0),
        constrained_layout=True,
    )
    extent = map_extent_um(geometry)
    for row, polarization in enumerate(("a", "b")):
        maxwell = arrays[("Maxwell", polarization, "same_incident_power")][
            field
        ]
        analytic = arrays[("analytic", polarization, "same_incident_power")][
            field
        ]
        difference = maxwell - analytic
        for column, (value, label) in enumerate(
            (
                (maxwell, "Maxwell"),
                (analytic, "analytic"),
                (difference, "Maxwell - analytic"),
            )
        ):
            plotted = masked_for_plot(value, geometry)
            if column == 2:
                limit = float(np.nanmax(np.abs(plotted)))
                image = axes[row, column].imshow(
                    plotted,
                    origin="lower",
                    extent=extent,
                    cmap="coolwarm",
                    vmin=-limit,
                    vmax=limit,
                    interpolation="nearest",
                )
            else:
                image = axes[row, column].imshow(
                    plotted,
                    origin="lower",
                    extent=extent,
                    cmap="inferno",
                    interpolation="nearest",
                )
            axes[row, column].plot(
                [extent[0], extent[1]],
                [extent[0], extent[1]],
                "w--",
                linewidth=0.8,
            )
            axes[row, column].set(
                xlabel="x=b (µm)",
                ylabel="y=a (µm)",
                title=f"E || {polarization}: {label}",
                xlim=(-18, 18),
                ylim=(-18, 18),
            )
            figure.colorbar(
                image,
                ax=axes[row, column],
                label=colorbar_label,
            )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_profiles(
    output_path: Path,
    arrays: dict[tuple[str, str, str], dict[str, np.ndarray]],
) -> None:
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(12.5, 12.0),
        constrained_layout=True,
    )
    fields = (
        ("edge_profile_Q_areal_W_m2", "areal Q (W/m²)"),
        ("edge_profile_temperature_K", "temperature rise (K)"),
        ("edge_profile_grad_magnitude_K_m", "|gradient T| (K/m)"),
    )
    for column, polarization in enumerate(("a", "b")):
        for row, (field, ylabel) in enumerate(fields):
            for model, style in (("Maxwell", "-"), ("analytic", "--")):
                values = arrays[
                    (model, polarization, "same_incident_power")
                ]
                axes[row, column].plot(
                    values["edge_profile_n_m"] * 1.0e6,
                    values[field],
                    style,
                    label=model,
                    linewidth=1.5,
                )
            axes[row, column].axvline(
                0.0,
                color="black",
                linestyle=":",
                linewidth=0.8,
            )
            axes[row, column].set(
                xlabel="edge-normal n=(y-x)/sqrt(2) (µm)",
                ylabel=ylabel,
                title=f"E || {polarization}",
            )
            axes[row, column].grid(alpha=0.25)
            axes[row, column].legend()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_case_csv(
    path: Path,
    cases: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    rows = []
    for key in sorted(cases):
        case = cases[key]
        rows.append(
            {
                "case_id": case["case_id"],
                "source_model": case["source_model"],
                "polarization": case["polarization"],
                "normalization": case["normalization"],
                "source_power_W": case["source_power_W"],
                "Tmax_rise_K": case["Tmax_rise_K"],
                "TaIrTe4_area_average_rise_K": case[
                    "TaIrTe4_area_average_rise_K"
                ],
                "raw_max_abs_grad_x_K_m": case[
                    "raw_fixed_edge_window"
                ]["x"]["raw_max_abs_K_m"],
                "raw_max_abs_grad_y_K_m": case[
                    "raw_fixed_edge_window"
                ]["y"]["raw_max_abs_K_m"],
                "raw_max_abs_grad_n_K_m": case[
                    "raw_fixed_edge_window"
                ]["n"]["raw_max_abs_K_m"],
                "raw_max_grad_magnitude_K_m": case[
                    "raw_fixed_edge_window"
                ]["magnitude"]["raw_max_abs_K_m"],
                "robust_max_abs_grad_x_K_m": case[
                    "robust_exact_edge_window"
                ]["x"]["maximum_abs_K_m"],
                "robust_max_abs_grad_n_K_m": case[
                    "robust_exact_edge_window"
                ]["n"]["maximum_abs_K_m"],
                "robust_max_grad_magnitude_K_m": case[
                    "robust_exact_edge_window"
                ]["magnitude"]["maximum_abs_K_m"],
                "linear_residual_relative": case[
                    "linear_residual_relative"
                ],
                "energy_balance_relative_error": case[
                    "energy_balance_relative_error"
                ],
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    summary: dict[str, Any],
) -> None:
    ratios = summary["polarization_ratios_b_over_a"]
    cross = summary["Maxwell_vs_analytic_same_incident_power"]
    optical = summary["optical_inputs"]
    trend = summary["paper_trend_assessment"]
    cases = summary["cases"]
    text = f"""# W12 50-nm Maxwell vs analytic paper-reduced thermal sanity

Status: `{summary['status']}`

This is a **paper-like scalar-Gaussian sanity comparison with an explicitly
assumed 12-µm waist**. It is not a paper reproduction, an experimentally
certified beam result, or an optical mesh-convergence certificate. Optical
lateral mesh is fixed at 50 nm; no 25-nm or 12.5-nm FDTD was run for this
stage.

## Contracts

- wavelength: 11 µm
- source: scalar Gaussian, requested 1/e²-intensity waist 12 µm
- incident power used by both physical comparisons: 285 µW
- straight edge: TaIrTe4 `y<=x`; lab `x=b`, `y=a`
- flake thickness: 130 nm; optical stack: 285-nm SiO2/Si
- optical boundaries: six PML, nonperiodic
- optical material: `epsilon_c=epsilon_b` paper-consistent 3D closure
- thermal equation: exact one-cell thickness-integrated Supplement Eq. S4
- thermal conductivity: `kx=kb=3.8`, `ky=ka=14.4 W/(m K)`
- Robin loss: top air `G=1`, bottom thermal-SiO2 `G=7.37e6 W/(m² K)`
- lateral TaIrTe4/air edge: insulating
- no explicit Si/SiO2 bulk domain, far Dirichlet, PTE, adjoint, AD-FD, or
  optimization

The requested 12-µm waist and the fitted target-plane widths are both stored
in the summary. In the paper Gaussian form, `w0=2 sigma`; this is the same
1/e²-intensity radius convention used here.

## Optical gates

| polarization | P_Q at unit central intensity (W) | P_six (W) | closure | auto-shutoff | all pre-remap gates |
|---|---:|---:|---:|---:|---|
| a | {optical['a']['Q']['P_Q_W_at_1_W_m2_central_intensity']:.9e} | {optical['a']['Q']['P_six_W_at_1_W_m2_central_intensity']:.9e} | {optical['a']['Q']['six_face_relative_closure']:.6%} | {optical['a']['Q']['auto_shutoff_final']:.9e} | {optical['a']['gates']['all_before_remap']} |
| b | {optical['b']['Q']['P_Q_W_at_1_W_m2_central_intensity']:.9e} | {optical['b']['Q']['P_six_W_at_1_W_m2_central_intensity']:.9e} | {optical['b']['Q']['six_face_relative_closure']:.6%} | {optical['b']['Q']['auto_shutoff_final']:.9e} | {optical['b']['gates']['all_before_remap']} |

All `Qx/Qy/Qz` components are included. Native component-specific field and
permittivity coordinates were read independently; their maximum mismatch is
reported in the JSON. Raw Q was not clipped, smoothed, gained, rescaled,
tiled, or deleted.

The independently measured a/b empty-stack incident powers differ by
`{summary['a_b_incident_normalization_audit']['relative_difference']:.3e}`
relative, passing the `<1e-5` same-normalization gate. No polarization-
matching gain was applied.

## Same-incident-power results

| case | absorbed power (W) | Tmax (K) | raw max |dT/dn| (K/m) | raw max |grad T| (K/m) | robust max |dT/dn| (K/m) | robust max |grad T| (K/m) |
|---|---:|---:|---:|---:|---:|---:|
| Maxwell a | {cases['Maxwell_a_same_incident_power']['source_power_W']:.9e} | {cases['Maxwell_a_same_incident_power']['Tmax_rise_K']:.9e} | {cases['Maxwell_a_same_incident_power']['raw_fixed_edge_window']['n']['raw_max_abs_K_m']:.9e} | {cases['Maxwell_a_same_incident_power']['raw_fixed_edge_window']['magnitude']['raw_max_abs_K_m']:.9e} | {cases['Maxwell_a_same_incident_power']['robust_exact_edge_window']['n']['maximum_abs_K_m']:.9e} | {cases['Maxwell_a_same_incident_power']['robust_exact_edge_window']['magnitude']['maximum_abs_K_m']:.9e} |
| Maxwell b | {cases['Maxwell_b_same_incident_power']['source_power_W']:.9e} | {cases['Maxwell_b_same_incident_power']['Tmax_rise_K']:.9e} | {cases['Maxwell_b_same_incident_power']['raw_fixed_edge_window']['n']['raw_max_abs_K_m']:.9e} | {cases['Maxwell_b_same_incident_power']['raw_fixed_edge_window']['magnitude']['raw_max_abs_K_m']:.9e} | {cases['Maxwell_b_same_incident_power']['robust_exact_edge_window']['n']['maximum_abs_K_m']:.9e} | {cases['Maxwell_b_same_incident_power']['robust_exact_edge_window']['magnitude']['maximum_abs_K_m']:.9e} |
| analytic a | {cases['analytic_a_same_incident_power']['source_power_W']:.9e} | {cases['analytic_a_same_incident_power']['Tmax_rise_K']:.9e} | {cases['analytic_a_same_incident_power']['raw_fixed_edge_window']['n']['raw_max_abs_K_m']:.9e} | {cases['analytic_a_same_incident_power']['raw_fixed_edge_window']['magnitude']['raw_max_abs_K_m']:.9e} | {cases['analytic_a_same_incident_power']['robust_exact_edge_window']['n']['maximum_abs_K_m']:.9e} | {cases['analytic_a_same_incident_power']['robust_exact_edge_window']['magnitude']['maximum_abs_K_m']:.9e} |
| analytic b | {cases['analytic_b_same_incident_power']['source_power_W']:.9e} | {cases['analytic_b_same_incident_power']['Tmax_rise_K']:.9e} | {cases['analytic_b_same_incident_power']['raw_fixed_edge_window']['n']['raw_max_abs_K_m']:.9e} | {cases['analytic_b_same_incident_power']['raw_fixed_edge_window']['magnitude']['raw_max_abs_K_m']:.9e} | {cases['analytic_b_same_incident_power']['robust_exact_edge_window']['n']['maximum_abs_K_m']:.9e} | {cases['analytic_b_same_incident_power']['robust_exact_edge_window']['magnitude']['maximum_abs_K_m']:.9e} |

| source | Pabs b/a | Tmax b/a | raw max |dT/dn| b/a | raw max |grad T| b/a | robust max |dT/dn| b/a | robust max |grad T| b/a |
|---|---:|---:|---:|---:|---:|---:|
| Maxwell | {ratios['Maxwell_same_incident_power']['absorbed_power_b_over_a']:.6f} | {ratios['Maxwell_same_incident_power']['Tmax_b_over_a']:.6f} | {ratios['Maxwell_same_incident_power']['raw_max_abs_grad_n_b_over_a']:.6f} | {ratios['Maxwell_same_incident_power']['raw_max_grad_magnitude_b_over_a']:.6f} | {ratios['Maxwell_same_incident_power']['robust_max_abs_grad_n_b_over_a']:.6f} | {ratios['Maxwell_same_incident_power']['robust_max_grad_magnitude_b_over_a']:.6f} |
| analytic | {ratios['analytic_same_incident_power']['absorbed_power_b_over_a']:.6f} | {ratios['analytic_same_incident_power']['Tmax_b_over_a']:.6f} | {ratios['analytic_same_incident_power']['raw_max_abs_grad_n_b_over_a']:.6f} | {ratios['analytic_same_incident_power']['raw_max_grad_magnitude_b_over_a']:.6f} | {ratios['analytic_same_incident_power']['robust_max_abs_grad_n_b_over_a']:.6f} | {ratios['analytic_same_incident_power']['robust_max_grad_magnitude_b_over_a']:.6f} |

Paper trend assessment:

- analytic: `DeltaT_b > DeltaT_a` = `{trend['analytic_Tmax_b_gt_a']}`,
  raw `|grad T|_b > |grad T|_a` =
  `{ratios['analytic_same_incident_power']['raw_max_grad_magnitude_b_over_a'] > 1.0}`,
  robust = `{trend['analytic_gradient_b_gt_a']}`
- Maxwell: `DeltaT_b > DeltaT_a` = `{trend['Maxwell_Tmax_b_gt_a']}`,
  raw `|grad T|_b > |grad T|_a` =
  `{ratios['Maxwell_same_incident_power']['raw_max_grad_magnitude_b_over_a'] > 1.0}`,
  robust = `{trend['Maxwell_gradient_b_gt_a']}`

Thus the analytic path reproduces both paper trends. The Maxwell path does
not reproduce the Tmax trend, and its gradient ordering depends on the
comparator: the one-cell raw maximum gives `b<a`, whereas the fixed-physical-
window exact-edge fit gives `b>a`. This disagreement is retained as a
diagnostic result, not collapsed into one unconditional pass.

Failure to match the paper numerically is not by itself a solver failure,
because the beam width and exact position are unpublished. Differences are
separated into total absorption, Maxwell edge redistribution, analytic
Gaussian approximation, the explicit beam assumption, and thermal-model
scope.

## Maxwell vs analytic

| polarization | absorbed-power ratio M/A | Q NRMSE | T NRMSE | gradient-vector NRMSE | edge Q profile NRMSE |
|---|---:|---:|---:|---:|---:|
| a | {cross['a']['absorbed_power_ratio_Maxwell_over_analytic']:.6f} | {cross['a']['Q_areal_NRMSE']:.6%} | {cross['a']['temperature_NRMSE']:.6%} | {cross['a']['gradient_vector_NRMSE']:.6%} | {cross['a']['edge_Q_profile_NRMSE']:.6%} |
| b | {cross['b']['absorbed_power_ratio_Maxwell_over_analytic']:.6f} | {cross['b']['Q_areal_NRMSE']:.6%} | {cross['b']['temperature_NRMSE']:.6%} | {cross['b']['gradient_vector_NRMSE']:.6%} | {cross['b']['edge_Q_profile_NRMSE']:.6%} |

Equal-absorbed-power copies are diagnostic only and are stored separately.
They do not modify the raw Lumerical artifact or the primary physical result.

## Provenance

- generation command: `{summary['generation_command']}`
- generation commit: `{summary['generation_commit']}`
- raw NPZ/FSP artifacts are external and are listed with byte size and
  SHA-256 in `RAW_ARTIFACT_MANIFEST.json`
- raw artifacts are not committed to Git
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() or args.report_dir.exists():
        raise FileExistsError("output and report directories must be new")
    if not np.isclose(args.thermal_step_nm, 50.0):
        raise ValueError(
            "this checkpoint fixes the common sheet grid at 50 nm"
        )

    optical = {
        "a": inspect_optical(args.edge_a_dir, "a"),
        "b": inspect_optical(args.edge_b_dir, "b"),
    }
    optical_prepass = all(
        item.gates["all_before_remap"] for item in optical.values()
    )
    incident_power_a = optical["a"].q_summary[
        "incident_power_W_at_1_W_m2"
    ]
    incident_power_b = optical["b"].q_summary[
        "incident_power_W_at_1_W_m2"
    ]
    incident_power_relative_difference = abs(
        incident_power_a - incident_power_b
    ) / max(abs(incident_power_a), abs(incident_power_b))
    same_incident_normalization = incident_power_relative_difference < 1.0e-5
    optical_prepass = optical_prepass and same_incident_normalization
    if not optical_prepass:
        args.report_dir.mkdir(parents=True)
        blocked = {
            "status": STATUS_BLOCKED,
            "optical_inputs": {
                key: {"Q": value.q_summary, "gates": value.gates}
                for key, value in optical.items()
            },
            "thermal_run": False,
            "PTE_run": False,
            "adjoint_run": False,
            "optimization_run": False,
            "generation_commit": git_commit(),
            "generation_command": shlex.join([sys.executable, *sys.argv]),
        }
        (args.report_dir / "w12_50nm_maxwell_analytic_summary.json").write_text(
            json.dumps(jsonable(blocked), indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(jsonable(blocked), indent=2))
        return 2

    args.output_dir.mkdir(parents=True)
    args.report_dir.mkdir(parents=True)
    beam = measured_beam(args.incident_reference_npz)

    bounds_a = source_coordinate_bounds(optical["a"])
    bounds_b = source_coordinate_bounds(optical["b"])
    if any(
        not np.allclose(
            bounds_a[axis],
            bounds_b[axis],
            rtol=0.0,
            atol=1.0e-15,
        )
        for axis in "xyz"
    ):
        raise RuntimeError("a/b saved Q coordinate bounds differ")
    geometry = common_sheet_geometry(
        {"x": bounds_a["x"], "y": bounds_a["y"]},
        args.thermal_step_nm * 1.0e-9,
    )
    system = assemble_sheet_system(geometry)
    flake_mask = np.any(geometry.flake_mask, axis=2)
    area = (
        np.diff(geometry.x_edges_m)[:, None]
        * np.diff(geometry.y_edges_m)[None, :]
    )

    q_inputs: dict[tuple[str, str, str], np.ndarray] = {}
    source_contracts: dict[str, Any] = {}
    mapping: dict[str, Any] = {}
    for polarization in ("a", "b"):
        item = optical[polarization]
        q_maxwell, mapped = map_Maxwell_q_to_sheet(
            item,
            geometry,
            bounds_a,
        )
        mapped["mapping_pass"] = (
            mapped["mapping_relative_power_error"] < 0.005
            and mapped["embedding_to_sheet_relative_power_error"] < 0.005
            and abs(mapped["mapped_power_outside_flake_W"])
            <= np.finfo(float).eps
            * max(abs(mapped["P_Q_target_W"]), 1.0)
        )
        mapping[polarization] = mapped
        q_analytic, contract = analytic_sheet_q(
            geometry,
            polarization,
            beam,
        )
        source_contracts[polarization] = contract
        q_inputs[("Maxwell", polarization, "same_incident_power")] = (
            q_maxwell
        )
        q_inputs[("analytic", polarization, "same_incident_power")] = (
            q_analytic
        )
        p_maxwell = integrate_q(q_maxwell, geometry)
        p_analytic = integrate_q(q_analytic, geometry)
        q_inputs[
            ("analytic", polarization, "equal_absorbed_power_diagnostic")
        ] = q_analytic * (p_maxwell / p_analytic)
        q_inputs[
            ("Maxwell", polarization, "equal_absorbed_power_diagnostic")
        ] = q_maxwell.copy()

    if not all(value["mapping_pass"] for value in mapping.values()):
        raise RuntimeError("conservative Maxwell-to-sheet remap gate failed")

    cases: dict[tuple[str, str, str], dict[str, Any]] = {}
    arrays: dict[
        tuple[str, str, str],
        dict[str, np.ndarray],
    ] = {}
    for key, q in q_inputs.items():
        model, polarization, normalization = key
        cases[key], arrays[key] = solve_case(
            system,
            geometry,
            q,
            source_model=model,
            polarization=polarization,
            normalization=normalization,
        )

    ratios = {
        f"{model}_{normalization}": case_ratios(
            cases,
            model,
            normalization,
        )
        for model in ("Maxwell", "analytic")
        for normalization in (
            "same_incident_power",
            "equal_absorbed_power_diagnostic",
        )
    }

    comparisons: dict[str, Any] = {}
    shape_comparisons: dict[str, Any] = {}
    for polarization in ("a", "b"):
        maxwell = arrays[
            ("Maxwell", polarization, "same_incident_power")
        ]
        analytic = arrays[
            ("analytic", polarization, "same_incident_power")
        ]
        grad_difference = (
            (maxwell["grad_T_x_K_m"] - analytic["grad_T_x_K_m"]) ** 2
            + (maxwell["grad_T_y_K_m"] - analytic["grad_T_y_K_m"]) ** 2
        )
        grad_reference = (
            analytic["grad_T_x_K_m"] ** 2
            + analytic["grad_T_y_K_m"] ** 2
        )
        comparisons[polarization] = {
            "absorbed_power_ratio_Maxwell_over_analytic": (
                cases[
                    ("Maxwell", polarization, "same_incident_power")
                ]["source_power_W"]
                / cases[
                    ("analytic", polarization, "same_incident_power")
                ]["source_power_W"]
            ),
            "Q_areal_NRMSE": weighted_nrmse(
                maxwell["Q_areal_W_m2"],
                analytic["Q_areal_W_m2"],
                flake_mask,
                area,
            ),
            "temperature_NRMSE": weighted_nrmse(
                maxwell["temperature_rise_K"],
                analytic["temperature_rise_K"],
                flake_mask,
                area,
            ),
            "gradient_vector_NRMSE": float(
                np.sqrt(
                    np.sum(area[flake_mask] * grad_difference[flake_mask])
                    / max(
                        np.sum(
                            area[flake_mask]
                            * grad_reference[flake_mask]
                        ),
                        np.finfo(float).tiny,
                    )
                )
            ),
            "edge_Q_profile_NRMSE": profile_nrmse(
                maxwell["edge_profile_Q_areal_W_m2"],
                analytic["edge_profile_Q_areal_W_m2"],
            ),
            "edge_temperature_profile_NRMSE": profile_nrmse(
                maxwell["edge_profile_temperature_K"],
                analytic["edge_profile_temperature_K"],
            ),
            "edge_gradient_profile_NRMSE": profile_nrmse(
                maxwell["edge_profile_grad_magnitude_K_m"],
                analytic["edge_profile_grad_magnitude_K_m"],
            ),
            "Maxwell_source_moments": cases[
                ("Maxwell", polarization, "same_incident_power")
            ]["source_spatial_moments"],
            "analytic_source_moments": cases[
                ("analytic", polarization, "same_incident_power")
            ]["source_spatial_moments"],
        }
        maxwell_equal = arrays[
            (
                "Maxwell",
                polarization,
                "equal_absorbed_power_diagnostic",
            )
        ]
        analytic_equal = arrays[
            (
                "analytic",
                polarization,
                "equal_absorbed_power_diagnostic",
            )
        ]
        shape_comparisons[polarization] = {
            "scope": (
                "diagnostic copy only; raw Maxwell Q is unchanged and the "
                "result is not the physical/paper-like comparison"
            ),
            "Q_areal_NRMSE": weighted_nrmse(
                maxwell_equal["Q_areal_W_m2"],
                analytic_equal["Q_areal_W_m2"],
                flake_mask,
                area,
            ),
            "temperature_NRMSE": weighted_nrmse(
                maxwell_equal["temperature_rise_K"],
                analytic_equal["temperature_rise_K"],
                flake_mask,
                area,
            ),
        }

    numerical_gates = {
        "optical_prepass": optical_prepass,
        "same_a_b_incident_normalization": same_incident_normalization,
        "mapping_power_error_lt_0p5_percent": all(
            item["mapping_relative_power_error"] < 0.005
            for item in mapping.values()
        ),
        "thermal_residual_lt_1e_minus_8": all(
            value["linear_residual_relative"] < 1.0e-8
            for value in cases.values()
        ),
        "thermal_energy_balance_lt_1_percent": all(
            value["energy_balance_relative_error"] < 0.01
            for value in cases.values()
        ),
    }
    numerical_gates["all"] = all(numerical_gates.values())
    status = STATUS_PASS if numerical_gates["all"] else STATUS_BLOCKED
    paper_trend = {
        "analytic_Tmax_b_gt_a": (
            ratios["analytic_same_incident_power"]["Tmax_b_over_a"] > 1.0
        ),
        "analytic_gradient_b_gt_a": (
            ratios["analytic_same_incident_power"][
                "robust_max_grad_magnitude_b_over_a"
            ]
            > 1.0
        ),
        "Maxwell_Tmax_b_gt_a": (
            ratios["Maxwell_same_incident_power"]["Tmax_b_over_a"] > 1.0
        ),
        "Maxwell_gradient_b_gt_a": (
            ratios["Maxwell_same_incident_power"][
                "robust_max_grad_magnitude_b_over_a"
            ]
            > 1.0
        ),
        "interpretation": (
            "trend comparison only; exact numerical agreement is not a gate "
            "because waist/position are unpublished and w0=12 um is explicit "
            "assumption"
        ),
    }

    raw_path = args.output_dir / (
        "w12_50nm_maxwell_analytic_paper_reduced_fields.npz"
    )
    payload: dict[str, np.ndarray] = {
        "x_edges_m": geometry.x_edges_m,
        "y_edges_m": geometry.y_edges_m,
        "z_edges_m": geometry.z_edges_m,
        "flake_mask": flake_mask,
    }
    for key, values in arrays.items():
        prefix = "_".join(key)
        for name, value in values.items():
            payload[f"{prefix}__{name}"] = np.asarray(value)
    np.savez_compressed(raw_path, **payload)

    figure_paths = {
        "Maxwell_temperature": args.report_dir
        / "maxwell_temperature_fig3f.png",
        "Maxwell_gradient": args.report_dir
        / "maxwell_gradient_fig3g.png",
        "analytic_temperature": args.report_dir
        / "analytic_temperature_fig3f.png",
        "analytic_gradient": args.report_dir
        / "analytic_gradient_fig3g.png",
        "temperature_cross_model": args.report_dir
        / "maxwell_vs_analytic_temperature.png",
        "gradient_cross_model": args.report_dir
        / "maxwell_vs_analytic_gradient.png",
        "edge_profiles": args.report_dir / "edge_normal_profiles.png",
    }
    for model, title_temperature, title_gradient in (
        ("Maxwell", "Maxwell ΔT", "Maxwell |∇T|"),
        ("analytic", "Analytic ΔT", "Analytic |∇T|"),
    ):
        a_values = arrays[(model, "a", "same_incident_power")]
        b_values = arrays[(model, "b", "same_incident_power")]
        plot_triplet(
            figure_paths[f"{model}_temperature"],
            geometry,
            a_values,
            b_values,
            "temperature_rise_K",
            title_temperature,
            "ΔT (K)",
        )
        plot_triplet(
            figure_paths[f"{model}_gradient"],
            geometry,
            a_values,
            b_values,
            "grad_T_magnitude_K_m",
            title_gradient,
            "|∇T| (K/m)",
        )
    plot_cross_model(
        figure_paths["temperature_cross_model"],
        geometry,
        arrays,
        "temperature_rise_K",
        "ΔT (K)",
    )
    plot_cross_model(
        figure_paths["gradient_cross_model"],
        geometry,
        arrays,
        "grad_T_magnitude_K_m",
        "|∇T| (K/m)",
    )
    plot_profiles(figure_paths["edge_profiles"], arrays)

    summary = {
        "status": status,
        "scope": (
            "paper-like scalar-Gaussian scenario with an explicitly assumed "
            "waist; 50-nm optical Maxwell Q versus paper analytic Q in the "
            "same exact thickness-integrated paper-reduced thermal operator"
        ),
        "not_claimed": [
            "paper reproduction",
            "experimentally certified result",
            "paper-certified beam",
            "mesh-converged optical certificate",
        ],
        "optical_mesh_contract": {
            "lateral_mesh_nm": 50.0,
            "edge_a_reused": True,
            "edge_b_new_GPU_only": True,
            "25nm_used_in_this_stage": False,
            "12p5nm_FDTD_run": False,
        },
        "beam": beam,
        "optical_inputs": {
            key: {
                "directory": str(value.directory.resolve()),
                "Q": value.q_summary,
                "gates": value.gates,
            }
            for key, value in optical.items()
        },
        "a_b_incident_normalization_audit": {
            "incident_power_a_W_at_1_W_m2": incident_power_a,
            "incident_power_b_W_at_1_W_m2": incident_power_b,
            "relative_difference": incident_power_relative_difference,
            "gate_lt_1e_minus_5": same_incident_normalization,
            "interpretation": (
                "independent matching empty-stack references use the same "
                "central-intensity normalization without polarization "
                "matching gain or rescaling"
            ),
        },
        "Maxwell_mapping": mapping,
        "analytic_source_contracts": source_contracts,
        "thermal_contract": {
            "equation": (
                "-dx(kb*t*dx theta)-dy(ka*t*dy theta)"
                "+(Gtop+Gbottom)*theta=q_areal"
            ),
            "implementation": (
                "one 130-nm z cell; conservative Cartesian finite volume is "
                "algebraically the thickness-integrated 2D equation"
            ),
            "grid_step_nm": args.thermal_step_nm,
            "grid_shape_xyz": list(system.shape),
            "domain_bounds_m": {
                "x": [
                    float(geometry.x_edges_m[0]),
                    float(geometry.x_edges_m[-1]),
                ],
                "y": [
                    float(geometry.y_edges_m[0]),
                    float(geometry.y_edges_m[-1]),
                ],
            },
            "kx_equals_kb_W_mK": 3.8,
            "ky_equals_ka_W_mK": 14.4,
            "thickness_m": THICKNESS_M,
            "G_top_air_W_m2K": thermal.G_TAIRTE4_AIR_W_M2K,
            "G_bottom_thermal_SiO2_W_m2K": (
                thermal.G_TAIRTE4_SIO2_W_M2K
            ),
            "lateral_edge": "insulating TaIrTe4/air y=x",
            "explicit_Si_or_SiO2_bulk": False,
            "far_Dirichlet": False,
        },
        "cases": {
            value["case_id"]: value for value in cases.values()
        },
        "polarization_ratios_b_over_a": ratios,
        "Maxwell_vs_analytic_same_incident_power": comparisons,
        "equal_absorbed_power_shape_diagnostic": shape_comparisons,
        "paper_trend_assessment": paper_trend,
        "acceptance": numerical_gates,
        "raw_output": artifact_record(
            raw_path,
            "external_comparison_fields_NPZ",
        ),
        "figures": {
            key: str(value.resolve()) for key, value in figure_paths.items()
        },
        "Q_processing": {
            "clipping": False,
            "smoothing": False,
            "gain": False,
            "global_rescaling": False,
            "tiling": False,
            "source_deletion": False,
            "equal_power_copy_scope": "diagnostic only",
        },
        "PTE_run": False,
        "adjoint_run": False,
        "AD_FD_run": False,
        "optimization_run": False,
        "generation_commit": git_commit(),
        "generation_command": shlex.join([sys.executable, *sys.argv]),
    }

    summary_path = (
        args.report_dir / "w12_50nm_maxwell_analytic_summary.json"
    )
    summary_path.write_text(
        json.dumps(jsonable(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = (
        args.report_dir / "w12_50nm_maxwell_analytic_cases.csv"
    )
    write_case_csv(csv_path, cases)
    report_path = args.report_dir / (
        "W12_50NM_MAXWELL_VS_ANALYTIC_"
        "PAPER_REDUCED_THERMAL_SANITY_REPORT.md"
    )
    write_report(report_path, summary)

    manifest_records = []
    for item in optical.values():
        manifest_records.extend(item.artifacts)
    manifest_records.extend(
        [
            beam["reference_artifact"],
            summary["raw_output"],
            artifact_record(summary_path, "published_summary_JSON"),
            artifact_record(csv_path, "published_cases_CSV"),
            artifact_record(report_path, "published_report"),
        ]
    )
    manifest_records.extend(
        artifact_record(path, f"figure_{name}")
        for name, path in figure_paths.items()
    )
    manifest = {
        "status": status,
        "generation_command": summary["generation_command"],
        "generation_commit": summary["generation_commit"],
        "raw_NPZ_and_FSP_committed_to_Git": False,
        "artifacts": manifest_records,
    }
    manifest_path = args.report_dir / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(jsonable(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(jsonable(summary), indent=2))
    return 0 if status == STATUS_PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
