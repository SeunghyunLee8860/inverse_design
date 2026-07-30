#!/usr/bin/env python3
"""Separate half-plane support removal from finite-edge EM redistribution.

Only saved NPZ/FSP-derived arrays are read.  This script performs no FDTD,
thermal, PTE, weighting-potential, adjoint, gradient, or optimization solve.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import erf

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.validation.paper_ir_sanity import (
    run_lumerical_device_a_ir_q as runner,
)


EPS0 = 8.8541878128e-12
ETA0 = 376.730313668
WAVELENGTH_M = 11.0e-6
W0_M = 2.0e-6
SOURCE_Z_M = 5.0e-6
FOCUS_Z_M = -65.0e-9
SOURCE_HALF_SPAN_M = 3.0e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()


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


def bounded_dual_cells(
    coordinate: np.ndarray,
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(coordinate, float).reshape(-1)
    if values.size < 2 or np.any(np.diff(values) <= 0.0):
        raise ValueError("coordinate must be strictly increasing")
    midpoints = 0.5 * (values[:-1] + values[1:])
    raw_edges = np.concatenate(
        (
            [values[0] - 0.5 * (values[1] - values[0])],
            midpoints,
            [values[-1] + 0.5 * (values[-1] - values[-2])],
        )
    )
    lower = np.maximum(raw_edges[:-1], low)
    upper = np.minimum(raw_edges[1:], high)
    width = np.maximum(upper - lower, 0.0)
    if not np.isclose(np.sum(width), high - low, rtol=1e-13, atol=1e-18):
        raise RuntimeError("bounded dual cells do not close")
    return lower, upper, width


def half_plane_cut_fraction(
    x: np.ndarray,
    y: np.ndarray,
    x_bounds: tuple[float, float] | list[float],
    y_bounds: tuple[float, float] | list[float],
) -> np.ndarray:
    """Exact dual-cell area fraction for the half-plane y <= x."""
    x_low, x_high, x_width = bounded_dual_cells(
        x, float(x_bounds[0]), float(x_bounds[1])
    )
    y_low, y_high, y_width = bounded_dual_cells(
        y, float(y_bounds[0]), float(y_bounds[1])
    )
    xl = x_low[:, None]
    xu = x_high[:, None]
    yl = y_low[None, :]
    yu = y_high[None, :]

    def primitive(value: np.ndarray) -> np.ndarray:
        return 0.5 * np.maximum(value - yl, 0.0) ** 2 - 0.5 * np.maximum(
            value - yu, 0.0
        ) ** 2

    area = primitive(xu) - primitive(xl)
    cell_area = x_width[:, None] * y_width[None, :]
    fraction = np.zeros_like(area)
    active = cell_area > 0.0
    fraction[active] = area[active] / cell_area[active]
    # The antiderivative subtracts O(length^2) terms; allow only machine-scale
    # cancellation in the dimensionless fraction before endpoint rounding.
    tolerance = 1.0e-9
    if np.min(fraction[active]) < -tolerance or np.max(
        fraction[active]
    ) > 1.0 + tolerance:
        raise RuntimeError("analytic cut-cell fraction lies outside [0,1]")
    # Only round floating-point endpoint noise in the geometric fraction.
    fraction[np.abs(fraction) < tolerance] = 0.0
    fraction[np.abs(fraction - 1.0) < tolerance] = 1.0
    return fraction


def xyz_weights(
    coordinates: dict[str, np.ndarray],
    bounds: dict[str, list[float]],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    weights = {
        axis: bounded_dual_cells(
            coordinates[axis],
            float(bounds[axis][0]),
            float(bounds[axis][1]),
        )[2]
        for axis in "xyz"
    }
    volume = (
        weights["x"][:, None, None]
        * weights["y"][None, :, None]
        * weights["z"][None, None, :]
    )
    return weights, volume


def integrate(values: np.ndarray, volume: np.ndarray) -> float:
    return float(np.sum(np.asarray(values, float) * volume))


def normalized_spatial_metrics(
    first: np.ndarray,
    second: np.ndarray,
    volume: np.ndarray,
) -> dict[str, float]:
    p_first = integrate(first, volume)
    p_second = integrate(second, volume)
    a = first / p_first
    b = second / p_second
    nrmse = float(
        np.sqrt(np.sum(volume * (b - a) ** 2) / np.sum(volume * a**2))
    )
    weight = volume / np.sum(volume)
    mean_a = float(np.sum(weight * a))
    mean_b = float(np.sum(weight * b))
    da = a - mean_a
    db = b - mean_b
    pearson = float(
        np.sum(weight * da * db)
        / np.sqrt(np.sum(weight * da**2) * np.sum(weight * db**2))
    )
    cosine = float(
        np.sum(volume * a * b)
        / np.sqrt(np.sum(volume * a**2) * np.sum(volume * b**2))
    )
    return {
        "first_power_W": p_first,
        "second_power_W": p_second,
        "equal_power_NRMSE": nrmse,
        "Pearson_correlation": pearson,
        "cosine_similarity": cosine,
    }


def signed_power_decomposition(
    full: np.ndarray,
    masked: np.ndarray,
    edge: np.ndarray,
    volume: np.ndarray,
) -> dict[str, float]:
    p_full = integrate(full, volume)
    p_masked = integrate(masked, volume)
    p_edge = integrate(edge, volume)
    d_total = p_full - p_edge
    d_support = p_full - p_masked
    d_em = p_masked - p_edge
    closure = d_total - (d_support + d_em)
    return {
        "P_planar_W": p_full,
        "P_masked_W": p_masked,
        "P_edge_W": p_edge,
        "P_edge_over_P_planar": p_edge / p_full,
        "P_edge_over_P_masked": p_edge / p_masked,
        "D_total_W": d_total,
        "D_support_W": d_support,
        "D_EM_W_signed": d_em,
        "D_support_over_P_planar": d_support / p_full,
        "D_EM_over_P_planar_signed": d_em / p_full,
        "D_support_over_D_total": d_support / d_total,
        "D_EM_over_D_total_signed": d_em / d_total,
        "decomposition_closure_W": closure,
        "decomposition_closure_relative_to_P_planar": closure / p_full,
    }


def overlap_matrix(
    source: np.ndarray,
    target: np.ndarray,
    low: float,
    high: float,
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    s_low, s_high, s_width = bounded_dual_cells(source, low, high)
    t_low, t_high, t_width = bounded_dual_cells(target, low, high)
    row: list[int] = []
    column: list[int] = []
    data: list[float] = []
    i = 0
    j = 0
    while i < target.size and j < source.size:
        overlap = min(t_high[i], s_high[j]) - max(t_low[i], s_low[j])
        if overlap > 0.0:
            row.append(i)
            column.append(j)
            data.append(float(overlap))
        if s_high[j] <= t_high[i]:
            j += 1
        else:
            i += 1
    matrix = csr_matrix(
        (np.asarray(data), (np.asarray(row), np.asarray(column))),
        shape=(target.size, source.size),
    )
    return matrix, s_width, t_width


def apply_sparse_axis(
    values: np.ndarray,
    matrix: csr_matrix,
    axis: int,
) -> np.ndarray:
    moved = np.moveaxis(values, axis, 0)
    mapped = matrix @ moved.reshape(moved.shape[0], -1)
    reshaped = np.asarray(mapped).reshape(
        (matrix.shape[0],) + moved.shape[1:]
    )
    return np.moveaxis(reshaped, 0, axis)


def conservative_remap(
    values: np.ndarray,
    source_coordinates: dict[str, np.ndarray],
    target_coordinates: dict[str, np.ndarray],
    bounds: dict[str, list[float]],
) -> tuple[np.ndarray, dict[str, float]]:
    matrices: dict[str, csr_matrix] = {}
    source_width: dict[str, np.ndarray] = {}
    target_width: dict[str, np.ndarray] = {}
    for axis in "xyz":
        matrices[axis], source_width[axis], target_width[axis] = overlap_matrix(
            source_coordinates[axis],
            target_coordinates[axis],
            float(bounds[axis][0]),
            float(bounds[axis][1]),
        )
    mass = np.asarray(values, float)
    for axis_index, axis in enumerate("xyz"):
        mass = apply_sparse_axis(mass, matrices[axis], axis_index)
    target_volume = (
        target_width["x"][:, None, None]
        * target_width["y"][None, :, None]
        * target_width["z"][None, None, :]
    )
    mapped = mass / target_volume
    source_volume = (
        source_width["x"][:, None, None]
        * source_width["y"][None, :, None]
        * source_width["z"][None, None, :]
    )
    p_source = integrate(values, source_volume)
    p_target = integrate(mapped, target_volume)
    return mapped, {
        "source_power_W": p_source,
        "target_power_W": p_target,
        "signed_power_error_W": p_target - p_source,
        "relative_power_error": abs(p_target - p_source)
        / max(abs(p_source), np.finfo(float).tiny),
    }


def line_profile(
    values: np.ndarray,
    coordinates: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
    coordinate_kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    z_integral = np.sum(values * weights["z"][None, None, :], axis=2)
    xy_mass = z_integral * weights["x"][:, None] * weights["y"][None, :]
    xx, yy = np.meshgrid(
        coordinates["x"], coordinates["y"], indexing="ij"
    )
    if coordinate_kind == "normal":
        projected = (yy - xx) / np.sqrt(2.0)
    elif coordinate_kind == "tangent":
        projected = (xx + yy) / np.sqrt(2.0)
    else:
        raise ValueError(coordinate_kind)
    dx = float(np.median(np.diff(coordinates["x"])))
    dy = float(np.median(np.diff(coordinates["y"])))
    if not np.allclose(np.diff(coordinates["x"]), dx, rtol=1e-8) or not np.allclose(
        np.diff(coordinates["y"]), dy, rtol=1e-8
    ):
        raise RuntimeError("exact diagonal profile requires uniform x/y output grid")
    quantum = min(dx, dy) / np.sqrt(2.0)
    origin = float(np.min(projected))
    index = np.rint((projected - origin) / quantum).astype(int)
    mass = np.bincount(
        index.ravel(), weights=xy_mass.ravel(), minlength=int(np.max(index)) + 1
    )
    center = origin + np.arange(mass.size) * quantum
    width = runner.dual_cell_weights_1d(center)
    return center, mass / width


def residual_decay(
    residual: np.ndarray,
    reference: np.ndarray,
    coordinates: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
) -> list[dict[str, float]]:
    xx, yy = np.meshgrid(
        coordinates["x"], coordinates["y"], indexing="ij"
    )
    normal = (yy - xx) / np.sqrt(2.0)
    z_abs = np.sum(
        np.abs(residual) * weights["z"][None, None, :], axis=2
    )
    z_reference = np.sum(
        np.abs(reference) * weights["z"][None, None, :], axis=2
    )
    area = weights["x"][:, None] * weights["y"][None, :]
    bands_um = [
        (-0.25, 0.0),
        (-0.50, -0.25),
        (-1.0, -0.50),
        (-2.0, -1.0),
        (-4.0, -2.0),
        (0.0, 0.25),
        (0.25, 0.50),
        (0.50, 1.0),
        (1.0, 2.0),
        (2.0, 4.0),
    ]
    rows: list[dict[str, float]] = []
    for low_um, high_um in bands_um:
        selected = (normal >= low_um * 1e-6) & (normal < high_um * 1e-6)
        residual_l1 = float(np.sum(z_abs[selected] * area[selected]))
        reference_l1 = float(
            np.sum(z_reference[selected] * area[selected])
        )
        rows.append(
            {
                "normal_low_um": low_um,
                "normal_high_um": high_um,
                "absolute_residual_power_W": residual_l1,
                "reference_absolute_power_W": reference_l1,
                "absolute_residual_over_reference": residual_l1
                / max(reference_l1, np.finfo(float).tiny),
            }
        )
    return rows


def ideal_symmetric_gaussian_control() -> dict[str, float]:
    coordinate = np.linspace(-6.0e-6, 6.0e-6, 401)
    bounds = [-6.0e-6, 6.0e-6]
    fraction = half_plane_cut_fraction(
        coordinate, coordinate, bounds, bounds
    )
    _, _, width = bounded_dual_cells(coordinate, *bounds)
    xx, yy = np.meshgrid(coordinate, coordinate, indexing="ij")
    full = np.exp(-2.0 * (xx**2 + yy**2) / (2.0e-6) ** 2)[
        :, :, None
    ]
    masked = full * fraction[:, :, None]
    volume = width[:, None, None] * width[None, :, None] * np.ones(
        (1, 1, 1)
    )
    p_full = integrate(full, volume)
    p_half = integrate(masked, volume)
    metrics = normalized_spatial_metrics(full, masked, volume)
    return {
        "P_half_over_P_full": p_half / p_full,
        "equal_power_full_half_NRMSE": metrics["equal_power_NRMSE"],
        "Pearson_correlation": metrics["Pearson_correlation"],
        "cosine_similarity": metrics["cosine_similarity"],
    }


def fitted_square_fraction(
    center_x: float,
    center_y: float,
    waist_x: float,
    waist_y: float,
    half_span: float,
) -> float:
    x_term = erf(np.sqrt(2.0) * (half_span - center_x) / waist_x) - erf(
        np.sqrt(2.0) * (-half_span - center_x) / waist_x
    )
    y_term = erf(np.sqrt(2.0) * (half_span - center_y) / waist_y) - erf(
        np.sqrt(2.0) * (-half_span - center_y) / waist_y
    )
    return float(0.25 * x_term * y_term)


def fitted_circle_fraction(
    center_x: float,
    center_y: float,
    waist_x: float,
    waist_y: float,
    radius: float,
) -> float:
    radial = (np.arange(800) + 0.5) / 800.0 * radius
    theta = (np.arange(1440) + 0.5) / 1440.0 * 2.0 * np.pi
    rr, tt = np.meshgrid(radial, theta, indexing="ij")
    xx = rr * np.cos(tt)
    yy = rr * np.sin(tt)
    intensity = np.exp(
        -2.0
        * (
            (xx - center_x) ** 2 / waist_x**2
            + (yy - center_y) ** 2 / waist_y**2
        )
    )
    integral = float(
        np.sum(intensity * rr)
        * (radius / radial.size)
        * (2.0 * np.pi / theta.size)
    )
    infinite = 0.5 * np.pi * waist_x * waist_y
    return integral / infinite


def source_audit(field_path: Path) -> dict[str, Any]:
    with np.load(field_path, allow_pickle=False) as raw:
        x = np.asarray(raw["source_profile_x_m"], float)
        y = np.asarray(raw["source_profile_y_m"], float)
        electric = np.asarray(raw["source_profile_E"]).squeeze()
        source_power = float(raw["source_power_native_W"][0])
        near_field_fit = json.loads(str(raw["beam_fit_json"][0]))
    if electric.shape != (x.size, y.size, 3):
        raise RuntimeError(f"unexpected source-profile shape {electric.shape}")
    transverse_intensity = (
        np.abs(electric[:, :, 0]) ** 2
        + np.abs(electric[:, :, 1]) ** 2
    ) / (2.0 * ETA0)
    longitudinal_fraction = float(
        np.sum(np.abs(electric[:, :, 2]) ** 2)
        / np.sum(np.abs(electric) ** 2)
    )
    x_width = bounded_dual_cells(
        x, -SOURCE_HALF_SPAN_M, SOURCE_HALF_SPAN_M
    )[2]
    y_width = bounded_dual_cells(
        y, -SOURCE_HALF_SPAN_M, SOURCE_HALF_SPAN_M
    )[2]
    square_power = float(
        np.sum(transverse_intensity * x_width[:, None] * y_width[None, :])
    )
    fitted = runner.fit_elliptical_gaussian(x, y, transverse_intensity)
    peak = float(np.max(transverse_intensity))
    boundary = np.concatenate(
        (
            transverse_intensity[0, :],
            transverse_intensity[-1, :],
            transverse_intensity[:, 0],
            transverse_intensity[:, -1],
        )
    )
    fitted_square = fitted_square_fraction(
        fitted["center_x_m"],
        fitted["center_y_m"],
        fitted["waist_x_m"],
        fitted["waist_y_m"],
        SOURCE_HALF_SPAN_M,
    )
    fitted_circle = fitted_circle_fraction(
        fitted["center_x_m"],
        fitted["center_y_m"],
        fitted["waist_x_m"],
        fitted["waist_y_m"],
        SOURCE_HALF_SPAN_M,
    )
    distance = SOURCE_Z_M - FOCUS_Z_M
    rayleigh = np.pi * W0_M**2 / WAVELENGTH_M
    paraxial_source_radius = W0_M * np.sqrt(1.0 + (distance / rayleigh) ** 2)
    theoretical_square = float(
        erf(np.sqrt(2.0) * SOURCE_HALF_SPAN_M / paraxial_source_radius)
        ** 2
    )
    theoretical_circle = float(
        1.0
        - np.exp(
            -2.0 * SOURCE_HALF_SPAN_M**2 / paraxial_source_radius**2
        )
    )
    return {
        "primary_evidence": {
            "path": str(field_path.resolve()),
            "size_bytes": field_path.stat().st_size,
            "sha256": sha256(field_path),
            "source_profile_shape": list(electric.shape),
            "source_profile_bounds_m": {
                "x": [float(x[0]), float(x[-1])],
                "y": [float(y[0]), float(y[-1])],
            },
        },
        "nominal_contract": {
            "wavelength_m": WAVELENGTH_M,
            "requested_waist_m": W0_M,
            "source_z_m": SOURCE_Z_M,
            "focus_z_m": FOCUS_Z_M,
            "distance_from_waist_m": distance,
            "square_aperture_full_span_m": 2.0 * SOURCE_HALF_SPAN_M,
            "use_scalar_approximation": True,
        },
        "stored_source_object_profile": {
            "native_sourcepower_readback_W": source_power,
            "native_sourcepower_is_primary_absolute_power": True,
            "transverse_E_plane_wave_integral_in_source_profile_normalization": (
                square_power
            ),
            "E_integral_over_native_sourcepower": square_power / source_power,
            "E_integral_is_absolute_power_calibration": False,
            "normalization_warning": (
                "The getresult(source,'fields') E profile does not share the "
                "absolute spectral-amplitude normalization of sourcepower. "
                "Its spatial shape is primary evidence, but its naive E-only "
                "plane-wave integral is not used as launched watts."
            ),
            "boundary_max_intensity_over_peak": float(
                np.max(boundary) / peak
            ),
            "boundary_mean_intensity_over_peak": float(
                np.mean(boundary) / peak
            ),
            "longitudinal_E2_fraction": longitudinal_fraction,
            "Gaussian_fit": fitted,
            "fitted_infinite_Gaussian_square_captured_fraction": fitted_square,
            "fitted_infinite_Gaussian_inscribed_circle_fraction": fitted_circle,
        },
        "paraxial_formula_failure_diagnostic": {
            "Rayleigh_range_m": rayleigh,
            "source_plane_expected_radius_m": paraxial_source_radius,
            "lambda_over_pi_w0": WAVELENGTH_M / (np.pi * W0_M),
            "square_captured_fraction": theoretical_square,
            "inscribed_circle_fraction": theoretical_circle,
            "aperture_mid_edge_intensity_over_peak": float(
                np.exp(
                    -2.0
                    * SOURCE_HALF_SPAN_M**2
                    / paraxial_source_radius**2
                )
            ),
            "aperture_corner_intensity_over_peak": float(
                np.exp(
                    -4.0
                    * SOURCE_HALF_SPAN_M**2
                    / paraxial_source_radius**2
                )
            ),
            "interpretation": (
                "lambda/(pi*w0) exceeds unity, so this paraxial expression "
                "is not a physical beam certificate. It diagnoses severe "
                "aperture truncation/inconsistent scalar-source parameters."
            ),
        },
        "z50nm_total_field_downward_decomposition": {
            **near_field_fit,
            "pure_incident_beam_waist": False,
            "warning": (
                "reflection, edge scattering, and evanescent fields may be "
                "present at this monitor"
            ),
        },
        "fit_vs_second_moment_explanation": (
            "The nonlinear fit extrapolates an infinite Gaussian from the "
            "truncated square samples, while the second moment is calculated "
            "only from power retained inside that square. Truncation therefore "
            "biases the moment waist downward."
        ),
    }


def load_common_q(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as raw:
        coordinates = {
            axis: np.asarray(raw[f"{axis}_m"], float) for axis in "xyz"
        }
        components = {
            axis: np.asarray(raw[f"Q{axis}_common_grid_W_m3"], float)
            for axis in "xyz"
        }
        total = np.asarray(raw["Q_common_grid_W_m3"], float)
        bounds = {
            axis: np.asarray(
                raw[f"realized_control_volume_{axis}_bounds_m"], float
            ).tolist()
            for axis in "xyz"
        }
    return {
        "coordinates": coordinates,
        "components": components,
        "total": total,
        "bounds": bounds,
    }


def artifact_record(
    path: Path, role: str, generation_command: str | None = None
) -> dict[str, Any]:
    value = {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if generation_command:
        value["generation_command"] = generation_command
    return value


def observed_runtime_projection(log_paths: list[Path]) -> dict[str, Any]:
    wall_seconds: dict[str, float] = {}
    pattern = re.compile(
        r"Overall wall time measurements in seconds:\s*([0-9.eE+-]+)"
    )
    for path in log_paths:
        match = pattern.search(path.read_text(encoding="utf-8", errors="replace"))
        if match is None:
            raise RuntimeError(f"overall wall time missing from {path}")
        wall_seconds[path.parent.name] = float(match.group(1))
    values = np.asarray(list(wall_seconds.values()), float)
    return {
        "observed_4ps_12um_wall_seconds": wall_seconds,
        "observed_minimum_s": float(np.min(values)),
        "observed_maximum_s": float(np.max(values)),
        "observed_mean_s": float(np.mean(values)),
        "five_case_same_grid_solver_wall_lower_bound_s": float(
            5.0 * np.mean(values)
        ),
        "interpretation": (
            "This is only a lower-bound projection for five cases on the "
            "current 12 um diagnostic grid. A physically adequate paper-like "
            "aperture/domain can be much slower; run a contract-only "
            "grid/memory probe before approval."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planar-q", type=Path, required=True)
    parser.add_argument("--edge-q", type=Path, required=True)
    parser.add_argument("--native-field-index", type=Path, required=True)
    parser.add_argument("--extraction-summary", type=Path, required=True)
    parser.add_argument("--source-fields", type=Path, required=True)
    parser.add_argument(
        "--runtime-log", type=Path, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--derived-artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    derived_dir = args.derived_artifact_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    planar = load_common_q(args.planar_q)
    edge = load_common_q(args.edge_q)
    for axis in "xyz":
        if not np.array_equal(
            planar["coordinates"][axis], edge["coordinates"][axis]
        ):
            raise RuntimeError(f"common {axis} coordinates differ")
        if planar["bounds"][axis] != edge["bounds"][axis]:
            raise RuntimeError(f"control-volume {axis} bounds differ")
    coordinates = planar["coordinates"]
    bounds = planar["bounds"]
    weights, volume = xyz_weights(coordinates, bounds)
    cut_fraction = half_plane_cut_fraction(
        coordinates["x"], coordinates["y"], bounds["x"], bounds["y"]
    )
    analytic_masked_components = {
        axis: planar["components"][axis] * cut_fraction[:, :, None]
        for axis in "xyz"
    }
    analytic_masked = sum(analytic_masked_components.values())
    primary_decomposition = signed_power_decomposition(
        planar["total"], analytic_masked, edge["total"], volume
    )
    primary_metrics = {
        "full_planar_vs_analytic_masked": normalized_spatial_metrics(
            planar["total"], analytic_masked, volume
        ),
        "analytic_masked_vs_finite_edge": normalized_spatial_metrics(
            analytic_masked, edge["total"], volume
        ),
        "full_planar_vs_finite_edge": normalized_spatial_metrics(
            planar["total"], edge["total"], volume
        ),
    }
    primary_component: dict[str, Any] = {}
    for axis in "xyz":
        primary_component[axis] = {
            "power_decomposition": signed_power_decomposition(
                planar["components"][axis],
                analytic_masked_components[axis],
                edge["components"][axis],
                volume,
            ),
            "spatial_metrics": {
                "full_planar_vs_analytic_masked": normalized_spatial_metrics(
                    planar["components"][axis],
                    analytic_masked_components[axis],
                    volume,
                ),
                "analytic_masked_vs_finite_edge": normalized_spatial_metrics(
                    analytic_masked_components[axis],
                    edge["components"][axis],
                    volume,
                ),
                "full_planar_vs_finite_edge": normalized_spatial_metrics(
                    planar["components"][axis],
                    edge["components"][axis],
                    volume,
                ),
            },
        }

    extraction = json.loads(args.extraction_summary.read_text())
    native_arrays: dict[str, np.ndarray] = {}
    with np.load(args.native_field_index, allow_pickle=False) as raw:
        for key in raw.files:
            if key != "metadata_json":
                native_arrays[key] = np.asarray(raw[key])
    omega = 2.0 * np.pi * runner.C0 / WAVELENGTH_M
    target_coordinates = coordinates
    native_results: dict[str, Any] = {}
    mapped: dict[str, dict[str, np.ndarray]] = {
        name: {} for name in ("full", "analytic", "loss_proxy", "edge")
    }
    for component in "xyz":
        source_coordinates = {
            axis: np.asarray(
                native_arrays[f"planar_E{component}_{axis}_m"], float
            )
            for axis in "xyz"
        }
        planar_e = native_arrays[f"planar_E{component}"]
        edge_e = native_arrays[f"edge_E{component}"]
        planar_epsilon = native_arrays[f"planar_epsilon_{component}"]
        edge_epsilon = native_arrays[f"edge_epsilon_{component}"]
        im_planar = np.imag(planar_epsilon)
        im_edge = np.imag(edge_epsilon)
        q_planar = 0.5 * EPS0 * omega * np.abs(planar_e) ** 2 * im_planar
        q_edge = 0.5 * EPS0 * omega * np.abs(edge_e) ** 2 * im_edge
        native_weights, native_volume = xyz_weights(
            source_coordinates, bounds
        )
        native_fraction = half_plane_cut_fraction(
            source_coordinates["x"],
            source_coordinates["y"],
            bounds["x"],
            bounds["y"],
        )
        q_analytic = q_planar * native_fraction[:, :, None]
        volume_active = native_volume > 0.0
        denominator_floor = (
            float(np.max(np.abs(im_planar[volume_active]))) * 1.0e-12
        )
        denominator_near_floor = volume_active & (
            np.abs(im_planar) <= denominator_floor
        )
        denominator_active = volume_active & ~denominator_near_floor
        participation = np.full(im_planar.shape, np.nan, float)
        participation[denominator_active] = (
            im_edge[denominator_active] / im_planar[denominator_active]
        )
        q_loss_proxy = np.zeros_like(q_planar)
        q_loss_proxy[denominator_active] = (
            q_planar[denominator_active]
            * participation[denominator_active]
        )
        direct_floor_contribution = (
            0.5
            * EPS0
            * omega
            * np.abs(planar_e[denominator_near_floor]) ** 2
            * im_edge[denominator_near_floor]
        )
        proxy_stats = {
            "definition": (
                "Im(epsilon_edge,c)/Im(epsilon_planar,c); diagnostic "
                "loss-participation proxy, not geometric occupancy"
            ),
            "clipping_applied": False,
            "denominator_floor": denominator_floor,
            "control_volume_cell_count": int(np.count_nonzero(volume_active)),
            "denominator_near_floor_cell_count": int(
                np.count_nonzero(denominator_near_floor)
            ),
            "denominator_active_cell_count": int(
                np.count_nonzero(denominator_active)
            ),
            "f_less_than_zero_cell_count": int(
                np.count_nonzero(
                    participation[denominator_active] < 0.0
                )
            ),
            "f_greater_than_one_cell_count": int(
                np.count_nonzero(
                    participation[denominator_active] > 1.0
                )
            ),
            "f_min": float(np.nanmin(participation)),
            "f_max": float(np.nanmax(participation)),
            "near_floor_direct_proxy_signed_power_W": float(
                np.sum(
                    direct_floor_contribution
                    * native_volume[denominator_near_floor]
                )
            ),
            "near_floor_cells_excluded_from_proxy": True,
        }
        remap_audit: dict[str, Any] = {}
        for name, values in (
            ("full", q_planar),
            ("analytic", q_analytic),
            ("loss_proxy", q_loss_proxy),
            ("edge", q_edge),
        ):
            mapped[name][component], remap_audit[name] = conservative_remap(
                values, source_coordinates, target_coordinates, bounds
            )
        native_results[component] = {
            "field_index_coordinate_mismatch_m": extraction["audit"][
                "planar"
            ]["pairing"][component]["maximum_coordinate_mismatch_m"],
            "planar_edge_coordinate_mismatch_m": max(
                extraction["audit"]["planar_edge_coordinate_pairing"][
                    component
                ].values()
            ),
            "analytic_cut_fraction": {
                "minimum": float(np.min(native_fraction[volume_active[:, :, 0]])),
                "maximum": float(np.max(native_fraction[volume_active[:, :, 0]])),
                "fractional_xy_cell_count": int(
                    np.count_nonzero(
                        (native_fraction > 0.0)
                        & (native_fraction < 1.0)
                    )
                ),
            },
            "loss_participation_proxy": proxy_stats,
            "component_to_common_conservative_remap": remap_audit,
        }

    mapped_total = {
        name: sum(components.values()) for name, components in mapped.items()
    }
    native_analytic_decomposition = signed_power_decomposition(
        mapped_total["full"],
        mapped_total["analytic"],
        mapped_total["edge"],
        volume,
    )
    loss_proxy_decomposition = signed_power_decomposition(
        mapped_total["full"],
        mapped_total["loss_proxy"],
        mapped_total["edge"],
        volume,
    )
    loss_proxy_metrics = {
        "full_planar_vs_loss_proxy_masked": normalized_spatial_metrics(
            mapped_total["full"], mapped_total["loss_proxy"], volume
        ),
        "loss_proxy_masked_vs_finite_edge": normalized_spatial_metrics(
            mapped_total["loss_proxy"], mapped_total["edge"], volume
        ),
        "full_planar_vs_finite_edge_native_remap": normalized_spatial_metrics(
            mapped_total["full"], mapped_total["edge"], volume
        ),
    }

    residual = edge["total"] - analytic_masked
    loss_residual = mapped_total["edge"] - mapped_total["loss_proxy"]
    decay_rows = []
    for kind, values, reference in (
        ("analytic_cut_cell", residual, analytic_masked),
        ("loss_participation_proxy", loss_residual, mapped_total["loss_proxy"]),
    ):
        for row in residual_decay(
            values, reference, coordinates, weights
        ):
            decay_rows.append({"mask_kind": kind, **row})

    ideal = ideal_symmetric_gaussian_control()
    source = source_audit(args.source_fields)
    runtime_projection = observed_runtime_projection(args.runtime_log)
    arrays_path = derived_dir / "w2_masked_planar_offline_derived_arrays.npz"
    normal_coordinate, planar_normal = line_profile(
        planar["total"], coordinates, weights, "normal"
    )
    _, masked_normal = line_profile(
        analytic_masked, coordinates, weights, "normal"
    )
    _, edge_normal = line_profile(
        edge["total"], coordinates, weights, "normal"
    )
    tangent_coordinate, planar_tangent = line_profile(
        planar["total"], coordinates, weights, "tangent"
    )
    _, masked_tangent = line_profile(
        analytic_masked, coordinates, weights, "tangent"
    )
    _, edge_tangent = line_profile(
        edge["total"], coordinates, weights, "tangent"
    )
    np.savez_compressed(
        arrays_path,
        x_m=coordinates["x"],
        y_m=coordinates["y"],
        z_m=coordinates["z"],
        analytic_cut_fraction=cut_fraction,
        Q_planar_W_m3=planar["total"],
        Q_analytic_masked_W_m3=analytic_masked,
        Q_edge_W_m3=edge["total"],
        residual_edge_minus_analytic_masked_W_m3=residual,
        Q_loss_proxy_masked_common_W_m3=mapped_total["loss_proxy"],
        residual_edge_minus_loss_proxy_W_m3=loss_residual,
        edge_normal_coordinate_m=normal_coordinate,
        planar_edge_normal_profile_W_m=planar_normal,
        masked_edge_normal_profile_W_m=masked_normal,
        edge_edge_normal_profile_W_m=edge_normal,
        edge_tangent_coordinate_m=tangent_coordinate,
        planar_edge_tangent_profile_W_m=planar_tangent,
        masked_edge_tangent_profile_W_m=masked_tangent,
        edge_edge_tangent_profile_W_m=edge_tangent,
    )

    payload = {
        "status": "VALIDATED_OFFLINE_MASKED_PLANAR_AND_SOURCE_AUDIT",
        "scope": {
            "FDTD_run": False,
            "thermal_run": False,
            "PTE_run": False,
            "weighting_potential_run": False,
            "adjoint_run": False,
            "gradient_run": False,
            "optimization_run": False,
            "w0_6p5um_run": False,
        },
        "geometry": {
            "edge_half_plane": "y <= x",
            "normal_coordinate": "n=(y-x)/sqrt(2); material side n<=0",
            "tangent_coordinate": "t=(x+y)/sqrt(2)",
            "analytic_mask": (
                "exact overlap of each bounded dual cell with y<=x; "
                "not a center Boolean mask"
            ),
            "common_grid_bounds_m": bounds,
            "common_grid_shape": list(planar["total"].shape),
            "common_grid_quadrature": (
                "bounded tensor-product dual-cell volumes closing exactly "
                "on the independently realized six-face control volume"
            ),
        },
        "primary_common_grid_analytic_cut_cell": {
            "power_decomposition": primary_decomposition,
            "spatial_metrics": primary_metrics,
            "components": primary_component,
            "residual_definition": (
                "Q_finite_edge_b - Q_planar_b*analytic_cut_fraction"
            ),
        },
        "component_specific_native_controls": {
            "coordinate_pairing": extraction["audit"],
            "components": native_results,
            "analytic_cut_cell_power_decomposition": (
                native_analytic_decomposition
            ),
            "loss_participation_proxy_power_decomposition": (
                loss_proxy_decomposition
            ),
            "loss_participation_proxy_spatial_metrics": loss_proxy_metrics,
            "remap": (
                "piecewise-constant bounded-dual-cell overlap remap; no gain "
                "or global rescaling"
            ),
        },
        "ideal_symmetric_Gaussian_half_plane_control": ideal,
        "source_audit": source,
        "proposed_GPU_runtime_projection": runtime_projection,
        "residual_decay": decay_rows,
        "derived_array_artifact": artifact_record(
            arrays_path,
            "offline analytic/loss-proxy masked Q and profiles",
            " ".join(sys.argv),
        ),
        "generation_commit": git_commit(),
    }
    summary_path = output_dir / "w2_masked_planar_offline_summary.json"
    summary_path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n"
    )

    with (output_dir / "w2_masked_planar_power_cases.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "control",
            "component",
            "P_planar_W",
            "P_masked_W",
            "P_edge_W",
            "D_total_W",
            "D_support_W",
            "D_EM_W_signed",
            "D_support_over_P_planar",
            "D_EM_over_P_planar_signed",
            "D_support_over_D_total",
            "D_EM_over_D_total_signed",
            "decomposition_closure_W",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for control, component, decomposition in [
            ("common_analytic_cut_cell", "total", primary_decomposition),
            *[
                (
                    "common_analytic_cut_cell",
                    axis,
                    primary_component[axis]["power_decomposition"],
                )
                for axis in "xyz"
            ],
            (
                "native_analytic_cut_cell_conservative_remap",
                "total",
                native_analytic_decomposition,
            ),
            (
                "native_loss_participation_conservative_remap",
                "total",
                loss_proxy_decomposition,
            ),
        ]:
            writer.writerow(
                {
                    "control": control,
                    "component": component,
                    **{
                        key: decomposition[key]
                        for key in fields
                        if key not in ("control", "component")
                    },
                }
            )
    with (output_dir / "w2_edge_residual_decay.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decay_rows[0]))
        writer.writeheader()
        writer.writerows(decay_rows)
    spatial_rows = [
        {
            "control": "analytic_cut_cell",
            "comparison": comparison,
            **values,
        }
        for comparison, values in primary_metrics.items()
    ] + [
        {
            "control": "loss_participation_proxy",
            "comparison": comparison,
            **values,
        }
        for comparison, values in loss_proxy_metrics.items()
    ]
    with (output_dir / "w2_masked_planar_spatial_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(spatial_rows[0]))
        writer.writeheader()
        writer.writerows(spatial_rows)

    # Figure 1: raw maps and signed residual.
    z_weight = weights["z"]
    maps = [
        np.sum(values * z_weight[None, None, :], axis=2)
        for values in (planar["total"], analytic_masked, edge["total"], residual)
    ]
    extent = [
        coordinates["x"][0] * 1e6,
        coordinates["x"][-1] * 1e6,
        coordinates["y"][0] * 1e6,
        coordinates["y"][-1] * 1e6,
    ]
    figure, axes = plt.subplots(1, 4, figsize=(16.2, 4.1), constrained_layout=True)
    titles = ["planar-b", "analytic masked planar", "finite-edge-b", "edge − masked"]
    positive_max = max(float(np.max(item)) for item in maps[:3])
    residual_max = float(np.max(np.abs(maps[3])))
    for index, (axis, values, title) in enumerate(zip(axes, maps, titles)):
        image = axis.imshow(
            values.T,
            origin="lower",
            extent=extent,
            cmap="coolwarm" if index == 3 else "magma",
            vmin=-residual_max if index == 3 else 0.0,
            vmax=residual_max if index == 3 else positive_max,
            aspect="equal",
        )
        axis.set(title=title, xlabel="x (µm)", ylabel="y (µm)")
        figure.colorbar(image, ax=axis, label="∫Q dz (W/m²)")
    figure.savefig(output_dir / "w2_masked_planar_raw_Q_and_residual.png", dpi=180)
    plt.close(figure)

    # Figure 2: equal-power normal/tangent profiles.
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)
    for axis, coordinate_line, profiles, title in (
        (
            axes[0],
            normal_coordinate,
            (planar_normal, masked_normal, edge_normal),
            "edge-normal",
        ),
        (
            axes[1],
            tangent_coordinate,
            (planar_tangent, masked_tangent, edge_tangent),
            "edge-tangential",
        ),
    ):
        for label, profile, power in zip(
            ("planar", "analytic masked", "finite edge"),
            profiles,
            (
                primary_decomposition["P_planar_W"],
                primary_decomposition["P_masked_W"],
                primary_decomposition["P_edge_W"],
            ),
        ):
            axis.plot(
                coordinate_line * 1e6,
                profile / power * 1e-6,
                label=label,
            )
        axis.set(
            xlabel=f"{title} coordinate (µm)",
            ylabel="equal-power line density (1/µm)",
            title=f"{title} profiles",
        )
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(output_dir / "w2_masked_planar_equal_power_profiles.png", dpi=180)
    plt.close(figure)

    # Figure 3: residual decay and un-clipped loss-participation audit.
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)
    for kind, label in (
        ("analytic_cut_cell", "analytic cut-cell"),
        ("loss_participation_proxy", "loss-participation proxy"),
    ):
        rows = [
            row
            for row in decay_rows
            if row["mask_kind"] == kind and row["normal_high_um"] <= 0.0
        ]
        distance = [
            -0.5 * (row["normal_low_um"] + row["normal_high_um"])
            for row in rows
        ]
        ratio = [row["absolute_residual_over_reference"] for row in rows]
        axes[0].plot(distance, np.asarray(ratio) * 100.0, "o-", label=label)
    axes[0].set(
        xlabel="distance into material from edge (µm)",
        ylabel="∫|residual| / ∫|masked reference| (%)",
        title="edge-normal residual decay",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    components = list("xyz")
    f_min = [
        native_results[axis]["loss_participation_proxy"]["f_min"]
        for axis in components
    ]
    f_max = [
        native_results[axis]["loss_participation_proxy"]["f_max"]
        for axis in components
    ]
    axes[1].vlines(components, f_min, f_max, linewidth=7, color="tab:purple")
    axes[1].scatter(components, f_min, color="black", marker="_")
    axes[1].scatter(components, f_max, color="black", marker="_")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].axhline(1.0, color="red", linestyle="--", linewidth=1.0)
    for index, axis_name in enumerate(components):
        stats = native_results[axis_name]["loss_participation_proxy"]
        axes[1].text(
            index,
            f_max[index] + 0.05,
            f">1: {stats['f_greater_than_one_cell_count']}",
            ha="center",
            fontsize=8,
        )
    axes[1].set(
        ylabel="un-clipped f_c range",
        title="diagnostic loss participation, not occupancy",
    )
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(output_dir / "w2_residual_decay_and_loss_proxy.png", dpi=180)
    plt.close(figure)

    # Figure 4: source-object profile and paraxial failure diagnostic.
    with np.load(args.source_fields, allow_pickle=False) as raw:
        sx = np.asarray(raw["source_profile_x_m"], float)
        sy = np.asarray(raw["source_profile_y_m"], float)
        se = np.asarray(raw["source_profile_E"]).squeeze()
    source_intensity = (
        np.abs(se[:, :, 0]) ** 2 + np.abs(se[:, :, 1]) ** 2
    ) / (2.0 * ETA0)
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), constrained_layout=True)
    image = axes[0].imshow(
        (source_intensity / np.max(source_intensity)).T,
        origin="lower",
        extent=[sx[0] * 1e6, sx[-1] * 1e6, sy[0] * 1e6, sy[-1] * 1e6],
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        aspect="equal",
    )
    axes[0].set(
        title="saved source-object |Eₜ|² profile",
        xlabel="x (µm)",
        ylabel="y (µm)",
    )
    figure.colorbar(image, ax=axes[0], label="intensity / peak")
    centre_y = int(np.argmin(np.abs(sy)))
    axes[1].plot(
        sx * 1e6,
        source_intensity[:, centre_y] / np.max(source_intensity),
        label="saved source profile",
    )
    paraxial_radius = source["paraxial_formula_failure_diagnostic"][
        "source_plane_expected_radius_m"
    ]
    axes[1].plot(
        sx * 1e6,
        np.exp(-2.0 * sx**2 / paraxial_radius**2),
        "--",
        label="paraxial formula (diagnostic only)",
    )
    axes[1].axvline(-3.0, color="black", linestyle=":")
    axes[1].axvline(3.0, color="black", linestyle=":")
    axes[1].set(
        xlabel="x at source object (µm)",
        ylabel="intensity / peak",
        title="6 µm square-aperture truncation",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.savefig(output_dir / "w2_source_object_aperture_audit.png", dpi=180)
    plt.close(figure)

    extraction_command = extraction.get("generation_command")
    manifest = {
        "raw_artifacts_committed_to_git": False,
        "offline_analysis_only": True,
        "generation_commit": git_commit(),
        "artifacts": [
            artifact_record(
                args.planar_q,
                "existing planar-b 4 ps raw common-grid Q",
            ),
            artifact_record(
                args.edge_q,
                "existing finite-edge-b 4 ps raw common-grid Q",
            ),
            artifact_record(
                args.native_field_index,
                "read-only extracted planar/edge native E and epsilon",
                extraction_command,
            ),
            artifact_record(
                args.extraction_summary,
                "read-only extraction audit",
                extraction_command,
            ),
            artifact_record(
                args.source_fields,
                "existing saved source-object and z=50nm fields",
            ),
            *[
                artifact_record(path, "existing 4 ps GPU solver runtime log")
                for path in args.runtime_log
            ],
            artifact_record(
                arrays_path,
                "offline derived masked/residual arrays",
                " ".join(sys.argv),
            ),
        ],
    }
    (output_dir / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    support = primary_decomposition
    metrics = primary_metrics
    source_stored = source["stored_source_object_profile"]
    source_formula = source["paraxial_formula_failure_diagnostic"]
    proxy_table_rows = "\n".join(
        "| {axis} | {floor:.6e} | {near_floor:,} | {active:,} | "
        "{below:,} | {above:,} | {minimum:.6g} | {maximum:.6g} | "
        "{floor_power:.3e} |".format(
            axis=axis,
            floor=native_results[axis]["loss_participation_proxy"][
                "denominator_floor"
            ],
            near_floor=native_results[axis]["loss_participation_proxy"][
                "denominator_near_floor_cell_count"
            ],
            active=native_results[axis]["loss_participation_proxy"][
                "denominator_active_cell_count"
            ],
            below=native_results[axis]["loss_participation_proxy"][
                "f_less_than_zero_cell_count"
            ],
            above=native_results[axis]["loss_participation_proxy"][
                "f_greater_than_one_cell_count"
            ],
            minimum=native_results[axis]["loss_participation_proxy"][
                "f_min"
            ],
            maximum=native_results[axis]["loss_participation_proxy"][
                "f_max"
            ],
            floor_power=native_results[axis]["loss_participation_proxy"][
                "near_floor_direct_proxy_signed_power_W"
            ],
        )
        for axis in "xyz"
    )
    maximum_remap_error = max(
        audit["relative_power_error"]
        for component in native_results.values()
        for audit in component[
            "component_to_common_conservative_remap"
        ].values()
    )
    analytic_decay = [
        row
        for row in decay_rows
        if row["mask_kind"] == "analytic_cut_cell"
        and row["normal_high_um"] <= 0.0
    ]
    decay_table_rows = "\n".join(
        "| {low:g} to {high:g} | {ratio:.6%} | {power:.6e} |".format(
            low=row["normal_low_um"],
            high=row["normal_high_um"],
            ratio=row["absolute_residual_over_reference"],
            power=row["absolute_residual_power_W"],
        )
        for row in analytic_decay
    )
    report = f"""# Offline masked-planar and source-contract audit

Status: `VALIDATED_OFFLINE_MASKED_PLANAR_AND_SOURCE_AUDIT`

No new FDTD, thermal, PTE, weighting-potential, adjoint, gradient, or
optimization solve was executed.

## Answers first

1. **Support removal.**  The exact analytic dual-cell half-plane removes
   `{support['D_support_over_P_planar']:.6%}` of planar-b power, accounting
   for `{support['D_support_over_D_total']:.6%}` of the observed total drop.
2. **Signed EM residual.**  After that support removal,
   `D_EM/P_planar = {support['D_EM_over_P_planar_signed']:.6%}` and
   `D_EM = {support['D_EM_W_signed']:.9e} W`, or
   `{support['D_EM_over_D_total_signed']:.6%}` of the total drop.
3. **Spatial shape.**  Equal-power NRMSE is
   `{metrics['full_planar_vs_analytic_masked']['equal_power_NRMSE']:.6%}`
   for full→masked,
   `{metrics['analytic_masked_vs_finite_edge']['equal_power_NRMSE']:.6%}`
   for masked→edge, and
   `{metrics['full_planar_vs_finite_edge']['equal_power_NRMSE']:.6%}`
   for full→edge.  These are not treated as additive quantities.
4. **Why nominal w0=2 µm is not realized.**
   `lambda/(pi*w0)={source_formula['lambda_over_pi_w0']:.6f}` is greater
   than one.  The scalar/paraxial contract is outside its validity range
   and the 6 µm aperture severely truncates its requested source-plane
   profile.
5. **Paper-like GPU value.**  A new calculation is worth considering only
   after selecting a physically realizable beam definition and certifying
   a source-only/background reference.  This report proposes cases but does
   not execute them.

## Signed power decomposition

- P_planar = `{support['P_planar_W']:.9e} W`
- P_masked = `{support['P_masked_W']:.9e} W`
- P_edge = `{support['P_edge_W']:.9e} W`
- D_total = `{support['D_total_W']:.9e} W`
- D_support = `{support['D_support_W']:.9e} W`
- signed D_EM = `{support['D_EM_W_signed']:.9e} W`
- D_support/P_planar = `{support['D_support_over_P_planar']:.9e}`
- signed D_EM/P_planar = `{support['D_EM_over_P_planar_signed']:.9e}`
- D_total − (D_support + D_EM) =
  `{support['decomposition_closure_W']:.3e} W`

The primary mask is the exact overlap of every bounded dual cell with
`y<=x`, not a center Boolean mask.  Component-specific native Yee cut-cell
fractions are also evaluated.

## Loss-participation proxy

The component quantity
`Im(epsilon_edge,c)/Im(epsilon_planar,c)` is retained only as a diagnostic
loss-participation proxy, never as geometric occupancy.  No clipping is
applied.  Cells below the recorded denominator floor, proxy ranges, and
counts below zero or above one are listed component-by-component in the
summary JSON.  Native field/index coordinate mismatch is at most
`{max(extraction['audit']['planar'][ 'maximum_field_index_coordinate_mismatch_m'], extraction['audit']['edge']['maximum_field_index_coordinate_mismatch_m']):.3e} m`;
planar/edge component coordinates are identical.  Component-to-common
mapping uses exact bounded dual-cell overlaps and reports its power error.

| component | denominator floor | near-floor cells | active-ratio cells | f<0 | f>1 | f min | f max | excluded-cell direct signed power (W) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{proxy_table_rows}

The `x` and `y` ratios reach about 2 in 268 cells each.  This is retained
without clipping and is direct evidence that `f_c` is not occupancy.  The
maximum component-to-common conservative-remap power error is
`{maximum_remap_error:.3e}`.

The analytic-cut residual on the material side is:

| edge-normal band n (µm) | absolute-residual integral / masked reference | absolute residual power (W) |
|---|---:|---:|
{decay_table_rows}

It is smaller in the 2–4 µm band than in the 0–0.25 µm band, but the trend
is not monotonic and remains nonzero far from the edge.  It is therefore
reported as a diagnostic profile, not fitted to an edge-decay law.

## Spatial comparisons

| comparison | equal-power NRMSE | Pearson correlation | cosine similarity |
|---|---:|---:|---:|
| full planar ↔ analytic masked planar | {metrics['full_planar_vs_analytic_masked']['equal_power_NRMSE']:.6%} | {metrics['full_planar_vs_analytic_masked']['Pearson_correlation']:.9f} | {metrics['full_planar_vs_analytic_masked']['cosine_similarity']:.9f} |
| analytic masked planar ↔ finite edge | {metrics['analytic_masked_vs_finite_edge']['equal_power_NRMSE']:.6%} | {metrics['analytic_masked_vs_finite_edge']['Pearson_correlation']:.9f} | {metrics['analytic_masked_vs_finite_edge']['cosine_similarity']:.9f} |
| full planar ↔ finite edge | {metrics['full_planar_vs_finite_edge']['equal_power_NRMSE']:.6%} | {metrics['full_planar_vs_finite_edge']['Pearson_correlation']:.9f} | {metrics['full_planar_vs_finite_edge']['cosine_similarity']:.9f} |
| loss-participation masked ↔ finite edge | {loss_proxy_metrics['loss_proxy_masked_vs_finite_edge']['equal_power_NRMSE']:.6%} | {loss_proxy_metrics['loss_proxy_masked_vs_finite_edge']['Pearson_correlation']:.9f} | {loss_proxy_metrics['loss_proxy_masked_vs_finite_edge']['cosine_similarity']:.9f} |

NRMSE values are independent pairwise comparisons and are not decomposed
or added.

## Source-object audit

- requested waist: `2 µm`
- source-to-focus distance: `{(SOURCE_Z_M-FOCUS_Z_M)*1e6:.6f} µm`
- paraxial-formula source-plane radius:
  `{source_formula['source_plane_expected_radius_m']*1e6:.6f} µm`
- native absolute `sourcepower` readback:
  `{source_stored['native_sourcepower_readback_W']:.9e} W`
- saved square-aperture E-only transverse integral:
  `{source_stored['transverse_E_plane_wave_integral_in_source_profile_normalization']:.9e}`
- saved square-boundary max intensity/peak:
  `{source_stored['boundary_max_intensity_over_peak']:.6f}`
- saved square-boundary mean intensity/peak:
  `{source_stored['boundary_mean_intensity_over_peak']:.6f}`
- fitted infinite-Gaussian waist:
  `{source_stored['Gaussian_fit']['waist_effective_geometric_mean_m']*1e6:.6f} µm`
- retained-square second-moment waist:
  `{np.sqrt(source_stored['Gaussian_fit']['moment_waist_x_m']*source_stored['Gaussian_fit']['moment_waist_y_m'])*1e6:.6f} µm`
- fitted infinite-Gaussian square captured fraction:
  `{source_stored['fitted_infinite_Gaussian_square_captured_fraction']:.6%}`
- fitted infinite-Gaussian inscribed-circle fraction:
  `{source_stored['fitted_infinite_Gaussian_inscribed_circle_fraction']:.6%}`
- paraxial diagnostic square/circle fractions:
  `{source_formula['square_captured_fraction']:.6%}` /
  `{source_formula['inscribed_circle_fraction']:.6%}`

The saved source-object `source_profile_E` is the primary evidence.  The
paraxial formula is used only as an aperture-truncation failure diagnostic.
The source-object E array and `sourcepower` use different spectral-amplitude
normalizations: the E-only plane-wave integral is retained as a shape proxy,
not called launched watts; `sourcepower` is the primary absolute power.
The z=50 nm total-field downward decomposition may contain reflection,
edge-scattered, and evanescent fields and is not called a pure incident-beam
waist.  The Gaussian fit extrapolates an infinite profile, whereas the
second moment uses only retained square-aperture power; truncation makes the
two widths differ.

## Proposed next GPU contract — not executed

First verify the paper/SI definitions of wavelength, spot radius versus
diameter versus FWHM, power, location, and polarization.  Unpublished
choices remain named scenarios.  The minimum optical set is:

1. source-only/background reference for the selected beam and domain;
2. planar-a and planar-b;
3. finite-edge-a and finite-edge-b.

The aperture/domain must contain at least 99.9% fitted incident power, have
small aperture-edge intensity and sufficient PML margin, and certify the
flake-plane incident profile in the background reference.  Production
material remains `epsilon_c=epsilon_b`, with `x=b, y=a, z=c=b`.

The current 12 µm diagnostic grid required
`{runtime_projection['observed_minimum_s']/60:.2f}`–
`{runtime_projection['observed_maximum_s']/60:.2f}` minutes of logged solver
wall time per 4 ps case.  Five cases on that same grid therefore give only a
lower bound of about
`{runtime_projection['five_case_same_grid_solver_wall_lower_bound_s']/60:.2f}`
minutes.  A physically adequate paper-like aperture/domain can be much
slower, so a contract-only grid/memory probe is required before approval;
no reliable paper-like runtime is claimed from the truncated w0=2 µm runs.
"""
    (output_dir / "W2_MASKED_PLANAR_SOURCE_OFFLINE_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "primary_power_decomposition": primary_decomposition,
                "primary_spatial_metrics": primary_metrics,
                "ideal_control": ideal,
                "source_formula": source_formula,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
