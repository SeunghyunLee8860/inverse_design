#!/usr/bin/env python3
"""GPU-only certification of the fixed paper-like scalar-Gaussian scenario.

Only a source and three field/power monitors are created.  There is no
TaIrTe4, substrate, thermal, PTE, weighting-potential, adjoint, gradient, or
optimization object.  CPU FDTD fallback is prohibited.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.validation.paper_ir_sanity import (
    audit_paper_ir_beam_contract as contract,
)


STAGE1 = HERE.parent / "photothermal_stage1"
API_HELPER = STAGE1 / "lumerical_api.py"
APPROVED_ROOT = Path("/home/seunghyun/lumerical_r12/opt/lumerical/v261")
APPROVED_API = APPROVED_ROOT / "api" / "python"

C0 = 299792458.0
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6
ETA0 = float(np.sqrt(MU0 / EPS0))
TARGET_FREQUENCY_HZ = C0 / contract.WAVELENGTH_M
SOURCE_START_M = 7.0e-6
SOURCE_STOP_M = 13.0e-6
PML_LAYERS = 24
MESH_ACCURACY = 5
CALIBRATION_BASELINE_INPUT_W0_M = contract.CALIBRATION_BASELINE_INPUT_W0_M
CALIBRATION_BASELINE_REALIZED_W0_M = (
    contract.CALIBRATION_BASELINE_REALIZED_W0_M
)
CALIBRATED_SOURCE_OBJECT_W0_M = contract.CALIBRATED_SOURCE_OBJECT_W0_M
CALIBRATION_BASELINE_FIELD_SHA256 = (
    contract.CALIBRATION_BASELINE_FIELD_SHA256
)
SOURCE_NAME = "paper_like_scalar_Gaussian_assumed_waist"
FIT_RESIDUAL_GATE = 0.005
ELLIPTICITY_GATE = 0.005
MONITORS = {
    "source_plane": contract.SOURCE_Z_M - 0.5e-6,
    "flake_target_plane": contract.FOCUS_Z_M,
    "downstream_plane": contract.FOCUS_Z_M - 5.0e-6,
}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def v261_session_provenance(
    *,
    solver_version: str,
    loaded_lumapi_path: Path,
    installation_version_key: str,
    installation_root: Path,
) -> dict[str, Any]:
    """Certify the selected v261 installation without parsing release labels.

    Lumerical v261 reports the internal solver version ``8.35.x`` through
    ``fdtd.version()``.  It does not return a string containing ``2026`` or
    ``v261``.  The release identity therefore combines the documented
    internal version series with the exact selected installation and API
    paths.
    """

    expected_root = APPROVED_ROOT.resolve()
    expected_lumapi = (APPROVED_API / "lumapi.py").resolve()
    actual_root = installation_root.resolve()
    actual_lumapi = loaded_lumapi_path.resolve()
    checks = {
        "installation_version_key_v261": installation_version_key == "v261",
        "installation_root_exact": actual_root == expected_root,
        "loaded_lumapi_exact": actual_lumapi == expected_lumapi,
        "solver_version_8_35_series": str(solver_version).startswith("8.35"),
    }
    return {
        "solver_version": str(solver_version),
        "installation_version_key": installation_version_key,
        "installation_root": str(actual_root),
        "expected_installation_root": str(expected_root),
        "loaded_lumapi_path": str(actual_lumapi),
        "expected_lumapi_path": str(expected_lumapi),
        "checks": checks,
        "all": all(checks.values()),
    }


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


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def scalar(value: Any, label: str) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise RuntimeError(f"{label} is not scalar: {array.shape}")
    result = float(np.real(array[0]))
    if not np.isfinite(result):
        raise RuntimeError(f"{label} is not finite")
    return result


def add_monitor(fdtd: Any, name: str, z_m: float) -> None:
    monitor = fdtd.addpower()
    monitor["name"] = name
    monitor["monitor type"] = "2D Z-normal"
    monitor["x min"] = -0.5 * contract.SOURCE_SPAN_M
    monitor["x max"] = 0.5 * contract.SOURCE_SPAN_M
    monitor["y min"] = -0.5 * contract.SOURCE_SPAN_M
    monitor["y max"] = 0.5 * contract.SOURCE_SPAN_M
    monitor["z"] = z_m
    monitor["override global monitor settings"] = True
    monitor["use source limits"] = False
    monitor["use wavelength spacing"] = True
    monitor["wavelength center"] = contract.WAVELENGTH_M
    monitor["wavelength span"] = 0.0
    monitor["frequency points"] = 1
    try:
        monitor["spatial interpolation"] = "specified position"
    except Exception:
        pass


def setup(
    fdtd: Any,
    duration_ps: float,
    shutoff: float,
    mesh_accuracy: int = MESH_ACCURACY,
    source_object_w0_m: float = CALIBRATED_SOURCE_OBJECT_W0_M,
) -> dict[str, Any]:
    fdtd.switchtolayout()
    solver = fdtd.addfdtd()
    for prop, value in (
        ("dimension", "3D"),
        ("x", 0.0),
        ("x span", contract.LATERAL_DOMAIN_M),
        ("y", 0.0),
        ("y span", contract.LATERAL_DOMAIN_M),
        ("z min", contract.FDTD_Z_MIN_M),
        ("z max", contract.FDTD_Z_MAX_M),
        ("pml layers", PML_LAYERS),
        ("mesh type", "auto non-uniform"),
        ("mesh refinement", "conformal variant 1"),
        ("mesh accuracy", mesh_accuracy),
        ("simulation time", duration_ps * 1e-12),
        ("auto shutoff min", shutoff),
    ):
        solver[prop] = value
    for axis in "xyz":
        solver[f"{axis} min bc"] = "PML"
        solver[f"{axis} max bc"] = "PML"

    source = fdtd.addgaussian()
    source["name"] = SOURCE_NAME
    source["injection axis"] = "z"
    source["direction"] = "backward"
    source["polarization angle"] = 0.0
    source["source shape"] = "Gaussian"
    source["use scalar approximation"] = True
    source["beam parameters"] = "Waist size and position"
    source["waist radius w0"] = source_object_w0_m
    source["distance from waist"] = -(
        contract.SOURCE_Z_M - contract.FOCUS_Z_M
    )
    source["x min"] = -0.5 * contract.SOURCE_SPAN_M
    source["x max"] = 0.5 * contract.SOURCE_SPAN_M
    source["y min"] = -0.5 * contract.SOURCE_SPAN_M
    source["y max"] = 0.5 * contract.SOURCE_SPAN_M
    source["z"] = contract.SOURCE_Z_M
    source["use global source settings"] = True
    source["override global source settings"] = False

    fdtd.setglobalsource("wavelength start", SOURCE_START_M)
    fdtd.setglobalsource("wavelength stop", SOURCE_STOP_M)
    fdtd.setglobalmonitor("use source limits", False)
    fdtd.setglobalmonitor("use wavelength spacing", True)
    fdtd.setglobalmonitor("wavelength center", contract.WAVELENGTH_M)
    fdtd.setglobalmonitor("wavelength span", 0.0)
    fdtd.setglobalmonitor("frequency points", 1)
    for name, z_m in MONITORS.items():
        add_monitor(fdtd, name, z_m)
    return {
        "source": {
            "name": SOURCE_NAME,
            "model": (
                "paper-like scalar-Gaussian scenario with an explicitly "
                "assumed waist"
            ),
            "wavelength_m": contract.WAVELENGTH_M,
            "numerical_pulse_band_m": [SOURCE_START_M, SOURCE_STOP_M],
            "target_realized_waist_radius_m": contract.SELECTED_W0_M,
            "Lumerical_source_object_waist_radius_m": source_object_w0_m,
            "source_object_waist_calibration": {
                "method": (
                    "multiplicative one-step calibration from the immutable "
                    "uncalibrated target-plane field fit"
                ),
                "baseline_input_w0_m": CALIBRATION_BASELINE_INPUT_W0_M,
                "baseline_realized_effective_w0_m": (
                    CALIBRATION_BASELINE_REALIZED_W0_M
                ),
                "baseline_field_NPZ_sha256": (
                    CALIBRATION_BASELINE_FIELD_SHA256
                ),
                "Q_clipping_smoothing_gain_or_rescaling": False,
            },
            "source_z_m": contract.SOURCE_Z_M,
            "focus_z_m": contract.FOCUS_Z_M,
            "distance_from_waist_requested_m": -(
                contract.SOURCE_Z_M - contract.FOCUS_Z_M
            ),
            "source_span_m": contract.SOURCE_SPAN_M,
            "polarization": "x",
        },
        "domain": {
            "x_bounds_m": [
                -0.5 * contract.LATERAL_DOMAIN_M,
                0.5 * contract.LATERAL_DOMAIN_M,
            ],
            "y_bounds_m": [
                -0.5 * contract.LATERAL_DOMAIN_M,
                0.5 * contract.LATERAL_DOMAIN_M,
            ],
            "z_bounds_m": [contract.FDTD_Z_MIN_M, contract.FDTD_Z_MAX_M],
            "six_boundaries": "PML",
            "PML_layers": PML_LAYERS,
            "periodic_or_Bloch": False,
            "materials": [],
        },
        "mesh_contract": {
            "global_mesh": "auto non-uniform",
            "mesh_refinement": "conformal variant 1",
            "mesh_accuracy": mesh_accuracy,
            "uniform_global_fine_mesh": False,
            "source_only_local_overrides": [],
            "material_successor_local_fine_regions": [
                "TaIrTe4",
                "illuminated straight edge",
                "Q extraction region",
            ],
            "TaIrTe4_z_resolution": "separate local z override",
        },
        "Q_processing": {
            "clipping": False,
            "smoothing": False,
            "gain": False,
            "rescaling": False,
        },
        "monitors": MONITORS,
    }


def resource_readback(fdtd: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index in (1, 2):
        result[str(index)] = {}
        for prop in (
            "active",
            "device type",
            "processes",
            "threads",
            "solver extra command line options",
        ):
            result[str(index)][prop] = str(
                fdtd.getresource("FDTD", index, prop)
            )
    return result


def mesh_readback(fdtd: Any) -> dict[str, Any]:
    coordinates: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}
    for axis in "xyz":
        attempts: list[str] = []
        for reader in (
            lambda axis=axis: fdtd.getresult("FDTD", axis),
            lambda axis=axis: fdtd.getdata("FDTD", axis, 1),
        ):
            try:
                raw = reader()
                if isinstance(raw, dict):
                    raw = raw[axis]
                coordinate = np.asarray(raw, float).reshape(-1)
                if coordinate.size < 3 or np.any(np.diff(coordinate) <= 0.0):
                    raise RuntimeError(
                        f"invalid {axis} coordinate with shape {coordinate.shape}"
                    )
                coordinates[axis] = coordinate
                break
            except Exception as exc:
                attempts.append(f"{type(exc).__name__}: {exc}")
        if axis not in coordinates:
            errors[axis] = " | ".join(attempts)
    if len(coordinates) == 3:
        shape = [int(coordinates[axis].size) for axis in "xyz"]
        points = int(np.prod(shape, dtype=np.int64))
        return {
            "available": True,
            "shape_xyz": shape,
            "grid_points": points,
            "bounds_m": {
                axis: [
                    float(coordinates[axis][0]),
                    float(coordinates[axis][-1]),
                ]
                for axis in "xyz"
            },
            "minimum_step_m": {
                axis: float(np.min(np.diff(coordinates[axis])))
                for axis in "xyz"
            },
            "maximum_step_m": {
                axis: float(np.max(np.diff(coordinates[axis])))
                for axis in "xyz"
            },
            "coordinate_arrays": coordinates,
        }
    return {"available": False, "errors": errors}


def source_readback(fdtd: Any) -> dict[str, Any]:
    properties = (
        "direction",
        "use scalar approximation",
        "waist radius w0",
        "distance from waist",
        "x min",
        "x max",
        "y min",
        "y max",
        "z",
        "polarization angle",
    )
    return {
        prop: (
            str(fdtd.getnamed(SOURCE_NAME, prop))
            if prop == "direction"
            else scalar(fdtd.getnamed(SOURCE_NAME, prop), f"source.{prop}")
        )
        for prop in properties
    }


def domain_readback(fdtd: Any) -> dict[str, Any]:
    numeric = {
        prop: scalar(fdtd.getnamed("FDTD", prop), f"FDTD.{prop}")
        for prop in (
            "x min",
            "x max",
            "y min",
            "y max",
            "z min",
            "z max",
            "pml layers",
            "mesh accuracy",
        )
    }
    return {
        **numeric,
        "mesh type": str(fdtd.getnamed("FDTD", "mesh type")),
        "mesh refinement": str(fdtd.getnamed("FDTD", "mesh refinement")),
        "boundaries": {
            f"{axis}_{side}": str(
                fdtd.getnamed("FDTD", f"{axis} {side} bc")
            )
            for axis in "xyz"
            for side in ("min", "max")
        },
    }


def monitor_fields(fdtd: Any, name: str) -> dict[str, Any]:
    coordinates = {
        axis: np.asarray(fdtd.getdata(name, axis, 1), float).reshape(-1)
        for axis in "xyz"
    }
    electric = {
        axis: np.asarray(fdtd.getdata(name, f"E{axis}", 1)).squeeze()
        for axis in "xyz"
    }
    magnetic = {
        axis: np.asarray(fdtd.getdata(name, f"H{axis}", 1)).squeeze()
        for axis in "xyz"
    }
    return {
        "coordinates": coordinates,
        "electric": electric,
        "magnetic": magnetic,
    }


def integrate_xy(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(np.trapezoid(values, y, axis=1), x, axis=0))


def fit_gaussian(
    x: np.ndarray, y: np.ndarray, intensity: np.ndarray
) -> dict[str, Any]:
    total = integrate_xy(intensity, x, y)
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("invalid plane intensity integral")
    xx, yy = np.meshgrid(x, y, indexing="ij")
    cx = integrate_xy(intensity * xx, x, y) / total
    cy = integrate_xy(intensity * yy, x, y) / total
    vx = integrate_xy(intensity * (xx - cx) ** 2, x, y) / total
    vy = integrate_xy(intensity * (yy - cy) ** 2, x, y) / total
    wx = 2.0 * np.sqrt(max(vx, np.finfo(float).tiny))
    wy = 2.0 * np.sqrt(max(vy, np.finfo(float).tiny))
    scale = 1.0e-6
    xx_scaled = xx / scale
    yy_scaled = yy / scale
    peak = float(np.max(intensity))
    normalized = intensity / peak
    initial = np.asarray(
        [
            0.0,
            cx / scale,
            cy / scale,
            np.log(wx / scale),
            np.log(wy / scale),
        ],
        float,
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        amplitude = np.exp(parameters[0])
        trial_wx = np.exp(parameters[3])
        trial_wy = np.exp(parameters[4])
        model = amplitude * np.exp(
            -2.0
            * (
                (xx_scaled - parameters[1]) ** 2 / trial_wx**2
                + (yy_scaled - parameters[2]) ** 2 / trial_wy**2
            )
        )
        return (model - normalized).reshape(-1)

    solution = least_squares(
        residual,
        initial,
        xtol=1.0e-13,
        ftol=1.0e-13,
        gtol=1.0e-13,
        max_nfev=200,
    )
    fitted_amplitude = float(np.exp(solution.x[0]))
    fitted_cx = float(solution.x[1] * scale)
    fitted_cy = float(solution.x[2] * scale)
    fitted_wx = float(np.exp(solution.x[3]) * scale)
    fitted_wy = float(np.exp(solution.x[4]) * scale)
    fitted = fitted_amplitude * peak * np.exp(
        -2.0
        * (
            (xx - fitted_cx) ** 2 / fitted_wx**2
            + (yy - fitted_cy) ** 2 / fitted_wy**2
        )
    )
    fit_nrmse = float(
        np.linalg.norm(intensity - fitted)
        / max(np.linalg.norm(intensity), np.finfo(float).tiny)
    )
    return {
        "integrated_power_W": total,
        "moment_center_x_m": float(cx),
        "moment_center_y_m": float(cy),
        "moment_waist_x_m": float(wx),
        "moment_waist_y_m": float(wy),
        "fitted_center_x_m": float(fitted_cx),
        "fitted_center_y_m": float(fitted_cy),
        "fitted_waist_x_m": float(fitted_wx),
        "fitted_waist_y_m": float(fitted_wy),
        "fitted_waist_effective_m": float(np.sqrt(fitted_wx * fitted_wy)),
        "fitted_peak_amplitude_over_sampled_peak": fitted_amplitude,
        "Gaussian_fit_NRMSE": fit_nrmse,
        "Gaussian_fit_objective": "unweighted linear-intensity L2",
        "Gaussian_fit_converged": bool(solution.success),
        "fitted_xy_ellipticity": float(
            abs(fitted_wx - fitted_wy)
            / max(0.5 * (fitted_wx + fitted_wy), np.finfo(float).tiny)
        ),
    }


def plane_metrics(fields: dict[str, Any], source_power_w: float) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    x = fields["coordinates"]["x"]
    y = fields["coordinates"]["y"]
    electric = fields["electric"]
    magnetic = fields["magnetic"]
    ex_down = 0.5 * (electric["x"] - ETA0 * magnetic["y"])
    ey_down = 0.5 * (electric["y"] + ETA0 * magnetic["x"])
    ez = electric["z"]
    intensity = (
        np.abs(ex_down) ** 2 + np.abs(ey_down) ** 2
    ) / (2.0 * ETA0)
    intensity = np.asarray(intensity, float).reshape(x.size, y.size)
    poynting_z = 0.5 * np.real(
        electric["x"] * np.conj(magnetic["y"])
        - electric["y"] * np.conj(magnetic["x"])
    )
    poynting_z = np.asarray(poynting_z, float).reshape(x.size, y.size)
    downward_poynting = -poynting_z
    fit = fit_gaussian(x, y, intensity)
    peak = float(np.max(intensity))
    boundary = np.concatenate(
        [intensity[0, :], intensity[-1, :], intensity[:, 0], intensity[:, -1]]
    )
    e2 = {
        axis: integrate_xy(
            np.asarray(np.abs(electric[axis]) ** 2, float).reshape(
                x.size, y.size
            ),
            x,
            y,
        )
        for axis in "xyz"
    }
    total_e2 = sum(e2.values())
    step = max(float(np.max(np.diff(x))), float(np.max(np.diff(y))))
    result = {
        **fit,
        "power_over_sourcepower": fit["integrated_power_W"] / source_power_w,
        "incident_power_integrated_W": fit["integrated_power_W"],
        "downward_Poynting_power_W": integrate_xy(
            downward_poynting, x, y
        ),
        "downward_Poynting_power_over_sourcepower": (
            integrate_xy(downward_poynting, x, y) / source_power_w
        ),
        "boundary_max_intensity_over_peak": float(
            np.max(boundary) / peak
        ),
        "boundary_mean_intensity_over_peak": float(
            np.mean(boundary) / peak
        ),
        "beam_center_error_m": float(
            np.hypot(fit["fitted_center_x_m"], fit["fitted_center_y_m"])
        ),
        "maximum_lateral_cell_m": step,
        "x_polarization_E2_fraction": e2["x"] / total_e2,
        "cross_polarized_Ey_E2_fraction": e2["y"] / total_e2,
        "longitudinal_Ez_E2_fraction": e2["z"] / total_e2,
        "all_fields_finite": all(
            np.all(np.isfinite(values))
            for kind in ("electric", "magnetic")
            for values in fields[kind].values()
        ),
    }
    arrays = {
        "x_m": x,
        "y_m": y,
        "downward_Ex": ex_down,
        "downward_Ey": ey_down,
        "downward_intensity_W_m2": intensity,
        "Poynting_z_W_m2": poynting_z,
        "downward_Poynting_W_m2": downward_poynting,
        **{
            f"E{axis}": electric[axis]
            for axis in "xyz"
        },
        **{
            f"H{axis}": magnetic[axis]
            for axis in "xyz"
        },
    }
    return result, arrays


def source_profile_from_arrays(
    x: np.ndarray,
    y: np.ndarray,
    electric: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    x = np.asarray(x, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)
    electric = np.asarray(electric).squeeze()
    intensity = np.sum(np.abs(electric) ** 2, axis=-1)
    peak = float(np.max(intensity))
    boundary = np.concatenate(
        [intensity[0, :], intensity[-1, :], intensity[:, 0], intensity[:, -1]]
    )
    fit = fit_gaussian(x, y, intensity)
    source_profile_integral = fit.pop("integrated_power_W")
    half_x = 0.5 * (float(x[-1]) - float(x[0]))
    half_y = 0.5 * (float(y[-1]) - float(y[0]))
    wx = fit["fitted_waist_x_m"]
    wy = fit["fitted_waist_y_m"]
    square_fraction = float(
        erf(np.sqrt(2.0) * half_x / wx)
        * erf(np.sqrt(2.0) * half_y / wy)
    )
    circle_radius = min(half_x, half_y)
    circle_fraction = float(
        1.0 - np.exp(-2.0 * circle_radius**2 / (wx * wy))
    )
    return {
        **fit,
        "E2_integral_in_source_profile_normalization": (
            source_profile_integral
        ),
        "E2_integral_is_absolute_power": False,
        "boundary_max_intensity_over_peak": float(np.max(boundary) / peak),
        "boundary_mean_intensity_over_peak": float(np.mean(boundary) / peak),
        "fitted_infinite_Gaussian_square_captured_fraction": square_fraction,
        "fitted_infinite_Gaussian_inscribed_circle_fraction": circle_fraction,
        "E_shape": list(electric.shape),
    }, {
        "source_profile_x_m": x,
        "source_profile_y_m": y,
        "source_profile_E": electric,
        "source_profile_relative_intensity": intensity / peak,
    }


def source_profile(fdtd: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    result = fdtd.getresult(SOURCE_NAME, "fields")
    return source_profile_from_arrays(result["x"], result["y"], result["E"])


def log_audit(output: Path) -> dict[str, Any]:
    logs = sorted(output.glob("*.log"))
    text = "\n".join(path.read_text(errors="replace") for path in logs)
    shutdown = re.findall(
        r"auto shutoff\s*[:=]\s*([0-9.eE+-]+)", text, re.IGNORECASE
    )
    grid = re.findall(
        r"(\d+)\s*x\s*(\d+)\s*x\s*(\d+)(?:\s*=\s*([\d,]+)\s*grid)?",
        text,
        re.IGNORECASE,
    )
    memory = re.findall(
        r"Estimated memory use on GPU .*?precise.*?([0-9.]+)\s*GiB",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return {
        "logs": [str(path) for path in logs],
        "final_auto_shutoff": (
            float(shutdown[-1]) if shutdown else None
        ),
        "logged_grid": (
            {
                "shape_xyz": [int(value) for value in grid[-1][:3]],
                "grid_points": (
                    int(grid[-1][3].replace(",", ""))
                    if grid[-1][3]
                    else int(
                        np.prod(
                            [int(value) for value in grid[-1][:3]],
                            dtype=np.int64,
                        )
                    )
                ),
            }
            if grid
            else None
        ),
        "precise_GPU_memory_GiB": float(memory[-1]) if memory else None,
        "simulation_completed_successfully": (
            "Simulation completed successfully" in text
        ),
    }


def strict_gpu_run(fdtd: Any, label: str) -> str:
    errors: list[str] = []
    for resource in ("Local GPU", "Local Host", "localhost", "Local Computer"):
        try:
            print(f"[gpu-only] {label} on {resource!r}", flush=True)
            fdtd.run("FDTD", "GPU", resource)
            return resource
        except Exception as exc:
            errors.append(f"{resource}: {type(exc).__name__}: {exc}")
    raise RuntimeError("GPU-only run failed; CPU fallback prohibited: " + " | ".join(errors))


def source_acceptance(
    *,
    focus: dict[str, Any],
    profile_metrics: dict[str, Any],
    post_run_mesh: dict[str, Any],
    log: dict[str, Any],
    planes: dict[str, dict[str, Any]],
    auto_shutoff_min: float,
) -> dict[str, bool]:
    return {
        "requested_vs_realized_fitted_waist_x_lt_0p5_percent": bool(
            abs(focus["fitted_waist_x_m"] - contract.SELECTED_W0_M)
            / contract.SELECTED_W0_M
            < 0.005
        ),
        "requested_vs_realized_fitted_waist_y_lt_0p5_percent": bool(
            abs(focus["fitted_waist_y_m"] - contract.SELECTED_W0_M)
            / contract.SELECTED_W0_M
            < 0.005
        ),
        "Gaussian_fit_NRMSE_lt_0p5_percent": bool(
            focus["Gaussian_fit_NRMSE"] < FIT_RESIDUAL_GATE
        ),
        "xy_ellipticity_lt_0p5_percent": bool(
            focus["fitted_xy_ellipticity"] < ELLIPTICITY_GATE
        ),
        "beam_center_error_lt_one_optical_cell": bool(
            focus["beam_center_error_m"]
            < focus["maximum_lateral_cell_m"]
        ),
        "square_captured_fraction_ge_99p9_percent": bool(
            profile_metrics[
                "fitted_infinite_Gaussian_square_captured_fraction"
            ]
            >= 0.999
        ),
        "source_boundary_max_le_1e_minus_3": bool(
            profile_metrics["boundary_max_intensity_over_peak"] <= 1.0e-3
        ),
        "source_boundary_mean_le_1e_minus_4": bool(
            profile_metrics["boundary_mean_intensity_over_peak"] <= 1.0e-4
        ),
        "incident_power_closure_lt_0p5_percent": bool(
            abs(focus["downward_Poynting_power_over_sourcepower"] - 1.0)
            < 0.005
        ),
        "actual_mesh_readback_available": bool(post_run_mesh["available"]),
        "GPU_memory_readback_available": bool(
            log["precise_GPU_memory_GiB"] is not None
        ),
        "no_NaN_or_Inf": bool(
            all(row["all_fields_finite"] for row in planes.values())
        ),
        "no_CPU_fallback": True,
        "auto_shutoff_le_1e_minus_5": bool(
            log["final_auto_shutoff"] is not None
            and log["final_auto_shutoff"] <= auto_shutoff_min
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--duration-ps", type=float, default=4.0)
    parser.add_argument("--auto-shutoff-min", type=float, default=1.0e-5)
    parser.add_argument("--gpu-device", default="GPU 4")
    parser.add_argument("--threads", default="8")
    parser.add_argument("--mesh-accuracy", type=int, default=MESH_ACCURACY)
    parser.add_argument(
        "--source-object-waist-um",
        type=float,
        default=CALIBRATED_SOURCE_OBJECT_W0_M * 1.0e6,
    )
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.mesh_accuracy <= 8:
        parser.error("--mesh-accuracy must be between 1 and 8")
    if args.source_object_waist_um <= 0.0:
        parser.error("--source-object-waist-um must be positive")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    command = shlex.join([sys.executable, *sys.argv])
    project = output / "paper_ir_source_only.fsp"
    result_path = output / "source_only_case_result.json"
    fdtd = None
    payload: dict[str, Any] = {
        "status": "BLOCKED",
        "generation_command": command,
        "generation_commit": git_commit(),
        "duration_ps": args.duration_ps,
        "contract_only": bool(args.contract_only),
        "scope": {
            "FDTD_source_only": not args.contract_only,
            "TaIrTe4": False,
            "substrate": False,
            "thermal": False,
            "PTE": False,
            "weighting_potential": False,
            "adjoint": False,
            "gradient": False,
            "optimization": False,
            "CPU_FDTD_fallback": False,
        },
    }
    try:
        os.environ["VC_LUMERICAL_ROOT"] = str(APPROVED_ROOT)
        os.environ["LUMERICAL_ROOT"] = str(APPROVED_ROOT)
        os.environ["LUMERICAL_PYTHONPATH"] = str(APPROVED_API)
        os.environ["PYTHONPATH"] = ":".join(
            value
            for value in (
                str(APPROVED_API),
                os.environ.get("PYTHONPATH", ""),
            )
            if value and "/opt/lumerical/" not in value
        )
        os.environ["LUMERICAL_SESSION_GPU_DEVICE"] = args.gpu_device
        os.environ["CL_GPU_DEVICE"] = args.gpu_device
        os.environ["FDTD_THREADS"] = str(args.threads)
        os.environ["PATH"] = f"{APPROVED_ROOT / 'bin'}:{os.environ.get('PATH','')}"
        for path in (STAGE1, REPOSITORY / "photothermal_pte"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        api_helper = load_module(API_HELPER, "paper_ir_source_lumerical_api")
        installation = SimpleNamespace(
            version_key="v261",
            root=APPROVED_ROOT,
            lumapi_path=APPROVED_API / "lumapi.py",
            device_executable=APPROVED_ROOT / "bin" / "device",
        )
        lumapi = api_helper.load_lumapi(installation)
        import eqc_lib as runtime

        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        built = setup(
            fdtd,
            args.duration_ps,
            args.auto_shutoff_min,
            args.mesh_accuracy,
            args.source_object_waist_um * 1.0e-6,
        )
        os.environ["LUMERICAL_SESSION_GPU_DEVICE"] = args.gpu_device
        resources = runtime.configure_session_resources(fdtd)
        fdtd.runsetup()
        mesh = mesh_readback(fdtd)
        readback = source_readback(fdtd)
        domain = domain_readback(fdtd)
        solver_version = str(fdtd.version())
        session_provenance = v261_session_provenance(
            solver_version=solver_version,
            loaded_lumapi_path=Path(lumapi.__file__),
            installation_version_key=installation.version_key,
            installation_root=installation.root,
        )
        expected_distance = -(contract.SOURCE_Z_M - contract.FOCUS_Z_M)
        checks = {
            "v261_session": session_provenance["all"],
            "all_six_PML": all(
                str(fdtd.getnamed("FDTD", f"{axis} {side} bc")).upper()
                == "PML"
                for axis in "xyz"
                for side in ("min", "max")
            ),
            "no_periodic_or_Bloch": True,
            "scalar_source_fixed": bool(
                round(readback["use scalar approximation"])
            ),
            "waist_readback": np.isclose(
                readback["waist radius w0"],
                args.source_object_waist_um * 1.0e-6,
                rtol=0.0,
                atol=1e-15,
            ),
            "backward_focus_distance_readback": np.isclose(
                readback["distance from waist"],
                expected_distance,
                rtol=0.0,
                atol=1e-15,
            ),
            "source_bounds_readback": all(
                np.isclose(readback[prop], expected, rtol=0.0, atol=1e-15)
                for prop, expected in {
                    "x min": -0.5 * contract.SOURCE_SPAN_M,
                    "x max": 0.5 * contract.SOURCE_SPAN_M,
                    "y min": -0.5 * contract.SOURCE_SPAN_M,
                    "y max": 0.5 * contract.SOURCE_SPAN_M,
                    "z": contract.SOURCE_Z_M,
                }.items()
            ),
            "PML_bounds_readback": all(
                np.isclose(domain[prop], expected, rtol=0.0, atol=1e-15)
                for prop, expected in {
                    "x min": -0.5 * contract.LATERAL_DOMAIN_M,
                    "x max": 0.5 * contract.LATERAL_DOMAIN_M,
                    "y min": -0.5 * contract.LATERAL_DOMAIN_M,
                    "y max": 0.5 * contract.LATERAL_DOMAIN_M,
                    "z min": contract.FDTD_Z_MIN_M,
                    "z max": contract.FDTD_Z_MAX_M,
                }.items()
            ),
            "auto_nonuniform_not_uniform_fine_mesh": (
                "auto" in domain["mesh type"].lower()
                and "non" in domain["mesh type"].lower()
            ),
            "mesh_accuracy_readback": np.isclose(
                domain["mesh accuracy"],
                float(args.mesh_accuracy),
                rtol=0.0,
                atol=0.0,
            ),
            "GPU_resource_active": resources["2"]["active"].strip() == "1",
            "GPU_resource_has_gpu_flag": "-gpu"
            in resources["2"]["solver extra command line options"],
        }
        payload["pre_run"] = {
            "version": solver_version,
            "session_provenance": session_provenance,
            "built_contract": built,
            "source_readback": readback,
            "domain_readback": domain,
            "resources": resources,
            "checks": checks,
            "mesh_after_runsetup": {
                key: value
                for key, value in mesh.items()
                if key != "coordinate_arrays"
            },
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"source-only pre-run contract failed: "
                f"{[key for key,value in checks.items() if not value]}"
            )
        fdtd.save(str(project))
        if args.contract_only:
            payload["status"] = "CONTRACT_ONLY_GRID_ESTIMATE_COMPLETE"
            write_json(result_path, payload)
            return_code = 0
        else:
            start = time.monotonic()
            resource = strict_gpu_run(
                fdtd, f"paper_ir_source_only_{args.duration_ps:g}ps"
            )
            wall_s = time.monotonic() - start
            post_run_mesh = mesh_readback(fdtd)
            source_power = scalar(
                fdtd.sourcepower(TARGET_FREQUENCY_HZ, 2, SOURCE_NAME),
                "sourcepower",
            )
            profile_metrics, profile_arrays = source_profile(fdtd)
            planes: dict[str, Any] = {}
            arrays: dict[str, np.ndarray] = {}
            for name in MONITORS:
                metrics, plane_arrays = plane_metrics(
                    monitor_fields(fdtd, name), source_power
                )
                planes[name] = metrics
                arrays.update(
                    {
                        f"{name}_{key}": value
                        for key, value in plane_arrays.items()
                    }
                )
            arrays.update(profile_arrays)
            artifact = output / "paper_ir_source_only_fields.npz"
            np.savez_compressed(
                artifact,
                **arrays,
                metadata_json=np.asarray(
                    [
                        json.dumps(
                            {
                                "generation_command": command,
                                "generation_commit": payload[
                                    "generation_commit"
                                ],
                                "source_power_W": source_power,
                                "duration_ps": args.duration_ps,
                            }
                        )
                    ]
                ),
            )
            log = log_audit(output)
            focus = planes["flake_target_plane"]
            acceptance = source_acceptance(
                focus=focus,
                profile_metrics=profile_metrics,
                post_run_mesh=post_run_mesh,
                log=log,
                planes=planes,
                auto_shutoff_min=args.auto_shutoff_min,
            )
            mandatory_pass = all(acceptance.values())
            payload.update(
                {
                    "status": (
                        "VALIDATED_PAPER_LIKE_SCALAR_GAUSSIAN_SOURCE_ONLY"
                        if mandatory_pass
                        else "FAILED_PAPER_LIKE_SCALAR_GAUSSIAN_SOURCE_ONLY_GATE"
                    ),
                    "GPU_resource_used": resource,
                    "solver_wall_time_s": wall_s,
                    "source_power_readback_W": source_power,
                    "source_object_profile": profile_metrics,
                    "planes": planes,
                    "log_audit": log,
                    "post_run_mesh": {
                        key: value
                        for key, value in post_run_mesh.items()
                        if key != "coordinate_arrays"
                    },
                    "field_artifact": {
                        "path": str(artifact),
                        "size_bytes": artifact.stat().st_size,
                        "sha256": sha256(artifact),
                    },
                    "acceptance": acceptance,
                    "source_only_gate_passed": mandatory_pass,
                    "successor_sequence_authorized": mandatory_pass,
                    "successor_sequence": [
                        "planar TaIrTe4, E parallel a",
                        "planar TaIrTe4, E parallel b",
                        "straight 45-degree finite edge, E parallel a",
                        "straight 45-degree finite edge, E parallel b",
                    ],
                    "thin_lens_status": "OPTIONAL_FUTURE_DIAGNOSTIC",
                }
            )
            fdtd.save(str(project))
            write_json(result_path, payload)
            return_code = 0
    except Exception as exc:
        message = str(exc)
        license_failure = any(
            token in message
            for token in (
                "Failed to set up Ansys license sharing",
                "ANSYSLI exited or could not read server port",
                "license unavailable",
                "License checkout failed",
            )
        )
        payload.update(
            {
                "status": (
                    "BLOCKED_LUMERICAL_LICENSE_UNAVAILABLE"
                    if license_failure
                    else "BLOCKED_SOURCE_ONLY_EXECUTION"
                ),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(result_path, payload)
        return_code = 2
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        artifacts = []
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.suffix.lower() in {
                ".fsp",
                ".npz",
                ".log",
                ".json",
            }:
                artifacts.append(
                    {
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
        write_json(
            output / "RAW_ARTIFACT_MANIFEST.json",
            {
                "raw_artifacts_committed_to_git": False,
                "generation_command": command,
                "generation_commit": payload["generation_commit"],
                "artifacts": artifacts,
            },
        )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
