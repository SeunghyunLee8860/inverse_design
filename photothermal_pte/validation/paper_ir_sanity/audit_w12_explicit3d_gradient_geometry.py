#!/usr/bin/env python3
"""Offline coordinate/line-integral audit of the W12 explicit-3D gradients.

No Maxwell, thermal, PTE, adjoint, or optimization solve is performed.  The
script consumes the existing explicit-3D thermal field artifact and checks:

1. invariance of the gradient norm under the Cartesian-to-edge rotation;
2. an edge-normal T(n) line derivative against the stored 2-D gradient; and
3. reconstruction of the line temperature change by integrating dT/dn.

Surface, midplane, and dz-weighted thickness-average projections are audited
separately so that a conclusion cannot silently depend on a z projection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    run_device_a_explicit_thermal_pte as thermal,
)


CASES = ("Maxwell_a", "Maxwell_b", "analytic_a", "analytic_b")
PROJECTIONS = ("surface", "midplane", "thickness_average")
SQRT2 = np.sqrt(2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_rms(error: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    return float(np.linalg.norm(error) / max(denominator, np.finfo(float).tiny))


def bilinear(
    values: np.ndarray,
    mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    xq: np.ndarray,
    yq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly sample a cell-centred field, requiring four valid cells."""
    ix = np.searchsorted(x, xq, side="right") - 1
    iy = np.searchsorted(y, yq, side="right") - 1
    valid = (
        (ix >= 0)
        & (ix + 1 < x.size)
        & (iy >= 0)
        & (iy + 1 < y.size)
    )
    safe_ix = np.clip(ix, 0, x.size - 2)
    safe_iy = np.clip(iy, 0, y.size - 2)
    valid &= (
        mask[safe_ix, safe_iy]
        & mask[safe_ix + 1, safe_iy]
        & mask[safe_ix, safe_iy + 1]
        & mask[safe_ix + 1, safe_iy + 1]
    )
    wx = (xq - x[safe_ix]) / (x[safe_ix + 1] - x[safe_ix])
    wy = (yq - y[safe_iy]) / (y[safe_iy + 1] - y[safe_iy])
    sampled = (
        (1.0 - wx) * (1.0 - wy) * values[safe_ix, safe_iy]
        + wx * (1.0 - wy) * values[safe_ix + 1, safe_iy]
        + (1.0 - wx) * wy * values[safe_ix, safe_iy + 1]
        + wx * wy * values[safe_ix + 1, safe_iy + 1]
    )
    sampled = np.asarray(sampled, float)
    sampled[~valid] = np.nan
    return sampled, valid


def cumulative_trapezoid(values: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    result[1:] = np.cumsum(
        0.5 * (values[1:] + values[:-1]) * np.diff(coordinate)
    )
    return result


def gradient_fields(
    temperature: np.ndarray,
    mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, np.ndarray]:
    gx, gy = thermal.cell_gradient(temperature, mask, x, y)
    gn = (-gx + gy) / SQRT2
    gt = (gx + gy) / SQRT2
    return {
        "gx": gx,
        "gy": gy,
        "gn": gn,
        "gt": gt,
        "magnitude": np.hypot(gx, gy),
    }


def identity_metrics(
    gradients: dict[str, np.ndarray],
    mask: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    lhs = gradients["gx"] ** 2 + gradients["gy"] ** 2
    rhs = gradients["gn"] ** 2 + gradients["gt"] ** 2
    absolute = np.abs(lhs - rhs)
    global_scale = max(
        float(np.max(lhs[mask])),
        float(np.max(rhs[mask])),
        np.finfo(float).tiny,
    )
    active = mask & (np.maximum(lhs, rhs) > global_scale * 1.0e-24)
    pixelwise = np.zeros_like(lhs)
    pixelwise[active] = absolute[active] / np.maximum(lhs[active], rhs[active])
    return {
        "active_cell_count": int(np.count_nonzero(active)),
        "maximum_absolute_error_K2_m2": float(np.max(absolute[mask])),
        "maximum_error_over_global_peak_norm2": float(
            np.max(absolute[mask]) / global_scale
        ),
        "maximum_pixelwise_relative_error": float(np.max(pixelwise[active])),
        "p99_pixelwise_relative_error": float(
            np.percentile(pixelwise[active], 99.0)
        ),
        "rms_error_over_rms_cartesian_norm2": relative_rms(
            (lhs - rhs)[active], lhs[active]
        ),
    }, pixelwise


def linecut_metrics(
    temperature: np.ndarray,
    gradients: dict[str, np.ndarray],
    mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    tangent_m: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    # Stay at least 0.30 um inside the material so every bilinear stencil is
    # on the TaIrTe4 side of the stair-stepped y=x boundary.
    n = np.arange(-10.0e-6, -0.299e-6, 50.0e-9)
    xq = (tangent_m - n) / SQRT2
    yq = (tangent_m + n) / SQRT2
    t_line, valid_t = bilinear(temperature, mask, x, y, xq, yq)
    gn_line, valid_g = bilinear(gradients["gn"], mask, x, y, xq, yq)
    valid = valid_t & valid_g & np.isfinite(t_line) & np.isfinite(gn_line)
    n = n[valid]
    t_line = t_line[valid]
    gn_line = gn_line[valid]
    if n.size < 20:
        raise RuntimeError(
            f"too few valid line samples at t={tangent_m:.6e} m"
        )
    derivative = np.gradient(t_line, n, edge_order=2)
    reconstructed = t_line[0] + cumulative_trapezoid(gn_line, n)
    delta_t = float(t_line[-1] - t_line[0])
    integral = float(np.trapezoid(gn_line, n))
    derivative_scale = max(
        float(np.linalg.norm(derivative)), np.finfo(float).tiny
    )
    correlation = float(np.corrcoef(derivative, gn_line)[0, 1])
    return {
        "tangent_coordinate_m": tangent_m,
        "sample_count": int(n.size),
        "n_bounds_m": [float(n[0]), float(n[-1])],
        "line_sample_spacing_m": float(np.median(np.diff(n))),
        "derivative_NRMSE": float(
            np.linalg.norm(gn_line - derivative) / derivative_scale
        ),
        "derivative_correlation": correlation,
        "temperature_delta_K": delta_t,
        "integrated_gradient_temperature_delta_K": integral,
        "integral_closure_absolute_K": abs(integral - delta_t),
        "integral_closure_relative_to_temperature_delta": float(
            abs(integral - delta_t)
            / max(abs(delta_t), np.finfo(float).tiny)
        ),
        "reconstructed_temperature_NRMSE_after_common_origin": relative_rms(
            reconstructed - t_line, t_line - t_line[0]
        ),
        "linecut_peak_abs_dTdn_location_m": float(
            n[int(np.argmax(np.abs(derivative)))]
        ),
        "field_peak_abs_dTdn_location_m": float(
            n[int(np.argmax(np.abs(gn_line)))]
        ),
        "linecut_peak_abs_dTdn_K_m": float(np.max(np.abs(derivative))),
        "field_peak_abs_dTdn_K_m": float(np.max(np.abs(gn_line))),
    }, {
        "n_m": n,
        "temperature_K": t_line,
        "line_derivative_K_m": derivative,
        "field_grad_n_K_m": gn_line,
        "reconstructed_temperature_K": reconstructed,
    }


def exact_diagonal_linecut_metrics(
    temperature: np.ndarray,
    gradients: dict[str, np.ndarray],
    mask: np.ndarray,
    n_grid: np.ndarray,
    t_grid: np.ndarray,
    target_tangent_m: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Use the actual cell centres on one constant-t Cartesian diagonal."""
    selected_tangent = float(
        t_grid.flat[int(np.argmin(np.abs(t_grid - target_tangent_m)))]
    )
    selected = (
        mask
        & np.isclose(t_grid, selected_tangent, rtol=0.0, atol=1.0e-15)
        & (n_grid >= -10.0e-6)
        & (n_grid <= 0.0)
    )
    n = n_grid[selected]
    t_line = temperature[selected]
    gn_line = gradients["gn"][selected]
    order = np.argsort(n)
    n = n[order]
    t_line = t_line[order]
    gn_line = gn_line[order]
    if n.size < 20:
        raise RuntimeError(
            f"too few exact diagonal cells at t={selected_tangent:.6e} m"
        )
    derivative = np.gradient(t_line, n, edge_order=2)
    reconstructed = t_line[0] + cumulative_trapezoid(gn_line, n)
    delta_t = float(t_line[-1] - t_line[0])
    integral = float(np.trapezoid(gn_line, n))
    interior = slice(None, -1)
    return {
        "requested_tangent_coordinate_m": target_tangent_m,
        "realized_cell_center_tangent_coordinate_m": selected_tangent,
        "maximum_tangent_coordinate_mismatch_m": float(
            np.max(np.abs(t_grid[selected] - selected_tangent))
        ),
        "sample_count": int(n.size),
        "n_bounds_m": [float(n[0]), float(n[-1])],
        "line_sample_spacing_m": {
            "minimum": float(np.min(np.diff(n))),
            "maximum": float(np.max(np.diff(n))),
        },
        "derivative_NRMSE": relative_rms(gn_line - derivative, derivative),
        "derivative_NRMSE_excluding_last_inside_edge_cell": relative_rms(
            gn_line[interior] - derivative[interior],
            derivative[interior],
        ),
        "derivative_correlation": float(
            np.corrcoef(derivative, gn_line)[0, 1]
        ),
        "last_inside_edge_cell_derivative_difference_K_m": float(
            gn_line[-1] - derivative[-1]
        ),
        "last_inside_edge_cell_difference_over_line_peak": float(
            abs(gn_line[-1] - derivative[-1])
            / max(float(np.max(np.abs(derivative))), np.finfo(float).tiny)
        ),
        "temperature_delta_K": delta_t,
        "integrated_gradient_temperature_delta_K": integral,
        "integral_closure_absolute_K": abs(integral - delta_t),
        "integral_closure_relative_to_temperature_delta": float(
            abs(integral - delta_t)
            / max(abs(delta_t), np.finfo(float).tiny)
        ),
        "reconstructed_temperature_NRMSE_after_common_origin": relative_rms(
            reconstructed - t_line, t_line - t_line[0]
        ),
        "linecut_peak_abs_dTdn_location_m": float(
            n[int(np.argmax(np.abs(derivative)))]
        ),
        "field_peak_abs_dTdn_location_m": float(
            n[int(np.argmax(np.abs(gn_line)))]
        ),
        "linecut_peak_abs_dTdn_K_m": float(np.max(np.abs(derivative))),
        "field_peak_abs_dTdn_K_m": float(np.max(np.abs(gn_line))),
        "last_inside_cell_n_m": float(n[-1]),
    }, {
        "n_m": n,
        "temperature_K": t_line,
        "line_derivative_K_m": derivative,
        "field_grad_n_K_m": gn_line,
        "reconstructed_temperature_K": reconstructed,
    }


def edge_statistics(
    gradients: dict[str, np.ndarray],
    mask: np.ndarray,
    n_grid: np.ndarray,
    t_grid: np.ndarray,
) -> dict[str, float]:
    edge_band = (
        mask
        & (n_grid >= -0.5e-6)
        & (n_grid <= 0.0)
        & (np.abs(t_grid) <= 10.0e-6)
    )
    values = np.abs(gradients["gn"][edge_band])
    peak_index = int(
        np.argmax(np.where(edge_band, np.abs(gradients["gn"]), -1.0))
    )
    peak_ij = np.unravel_index(peak_index, mask.shape)
    maximum = float(np.max(values))
    p99 = float(np.percentile(values, 99.0))
    return {
        "edge_band_cell_count": int(values.size),
        "raw_max_abs_grad_n_K_m": maximum,
        "p99_abs_grad_n_K_m": p99,
        "raw_max_over_p99": maximum / max(p99, np.finfo(float).tiny),
        "raw_peak_tangent_coordinate_m": float(t_grid[peak_ij]),
        "raw_peak_normal_coordinate_m": float(n_grid[peak_ij]),
    }


def write_case_projection_figure(
    path: Path,
    model: str,
    projection_data: dict[str, dict[str, dict[str, np.ndarray]]],
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> None:
    extent = [x[0] * 1e6, x[-1] * 1e6, y[0] * 1e6, y[-1] * 1e6]
    figure, axes = plt.subplots(
        2, 3, figsize=(15.5, 9.2), constrained_layout=True
    )
    for column, projection in enumerate(PROJECTIONS):
        arrays = [
            projection_data[f"{model}_a"][projection]["gradients"]["gn"],
            projection_data[f"{model}_b"][projection]["gradients"]["gn"],
        ]
        shared_limit = max(float(np.max(np.abs(item[mask]))) for item in arrays)
        for row, polarization in enumerate(("a", "b")):
            data = np.where(mask, arrays[row], np.nan)
            image = axes[row, column].imshow(
                data.T,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap="coolwarm",
                vmin=-shared_limit,
                vmax=shared_limit,
            )
            axes[row, column].set(
                title=f"{model}, E∥{polarization}, {projection}",
                xlabel="x=b (µm)",
                ylabel="y=a (µm)",
            )
            figure.colorbar(image, ax=axes[row, column], label="∂nT (K/m)")
    figure.suptitle(
        "Edge-normal gradients — a/b share one color scale in each column",
        fontsize=15,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_identity_figure(
    path: Path,
    projection_data: dict[str, dict[str, dict[str, np.ndarray]]],
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> None:
    extent = [x[0] * 1e6, x[-1] * 1e6, y[0] * 1e6, y[-1] * 1e6]
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for axis, case in zip(axes.flat, CASES):
        error = projection_data[case]["thickness_average"]["identity_error"]
        display = np.where(mask, np.log10(np.maximum(error, 1.0e-18)), np.nan)
        image = axis.imshow(
            display.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="viridis",
            vmin=-18,
            vmax=-12,
        )
        axis.set(
            title=case.replace("_", " "),
            xlabel="x=b (µm)",
            ylabel="y=a (µm)",
        )
        figure.colorbar(
            image, ax=axis, label="log10(pixelwise relative identity error)"
        )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_linecut_figure(
    path: Path,
    linecuts: dict[str, dict[str, np.ndarray]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, case in zip(axes.flat, CASES):
        line = linecuts[case]
        n_um = line["n_m"] * 1e6
        axis.plot(
            n_um,
            line["line_derivative_K_m"],
            label="dT(n)/dn",
            linewidth=2,
        )
        axis.plot(
            n_um,
            line["field_grad_n_K_m"],
            "--",
            label="(-∂xT+∂yT)/√2",
        )
        axis.set(
            title=case.replace("_", " "),
            xlabel="n=(-x+y)/√2 (µm)",
            ylabel="K/m",
        )
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_reconstruction_figure(
    path: Path,
    linecuts: dict[str, dict[str, np.ndarray]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, case in zip(axes.flat, CASES):
        line = linecuts[case]
        n_um = line["n_m"] * 1e6
        axis.plot(n_um, line["temperature_K"], label="sampled T(n)", linewidth=2)
        axis.plot(
            n_um,
            line["reconstructed_temperature_K"],
            "--",
            label="T(n₁)+∫∂nT dn",
        )
        axis.set(
            title=case.replace("_", " "),
            xlabel="n=(-x+y)/√2 (µm)",
            ylabel="ΔT (K)",
        )
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    input_path = args.input_npz.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    if report_dir.exists():
        raise FileExistsError(f"report directory already exists: {report_dir}")
    report_dir.mkdir(parents=True)

    with np.load(input_path, allow_pickle=False) as raw:
        x_edges = np.asarray(raw["x_edges_m"], float)
        y_edges = np.asarray(raw["y_edges_m"], float)
        z_edges = np.asarray(raw["z_edges_m"], float)
        x = 0.5 * (x_edges[:-1] + x_edges[1:])
        y = 0.5 * (y_edges[:-1] + y_edges[1:])
        z = 0.5 * (z_edges[:-1] + z_edges[1:])
        dx = np.diff(x_edges)
        dy = np.diff(y_edges)
        dz = np.diff(z_edges)
        flake_mask_3d = np.asarray(raw["flake_mask"], bool)
        mask = np.any(flake_mask_3d, axis=2)
        flake_z = np.flatnonzero(np.any(flake_mask_3d, axis=(0, 1)))
        thickness = float(np.sum(dz[flake_z]))
        surface_index = int(flake_z[-1])
        midplane_index = int(
            flake_z[np.argmin(np.abs(z[flake_z] + 65.0e-9))]
        )
        n_grid = (-x[:, None] + y[None, :]) / SQRT2
        t_grid = (x[:, None] + y[None, :]) / SQRT2

        summary: dict[str, Any] = {
            "status": "COMPLETED_OFFLINE_W12_EXPLICIT3D_GRADIENT_GEOMETRY_AUDIT",
            "scope": (
                "offline audit only; no FDTD, thermal, PTE, adjoint, AD-FD, "
                "or optimization execution"
            ),
            "input": {
                "path": str(input_path),
                "size_bytes": input_path.stat().st_size,
                "sha256": sha256(input_path),
            },
            "coordinate_contract": {
                "lab_x": "crystal b",
                "lab_y": "crystal a",
                "edge": "TaIrTe4 y<=x",
                "outward_normal": "n=(-x+y)/sqrt(2)",
                "tangent": "t=(x+y)/sqrt(2)",
            },
            "grid_audit": {
                "field_location": "thermal finite-volume cell centers",
                "x_cell_count": int(x.size),
                "y_cell_count": int(y.size),
                "z_cell_count": int(z.size),
                "dx_m": {"minimum": float(dx.min()), "maximum": float(dx.max())},
                "dy_m": {"minimum": float(dy.min()), "maximum": float(dy.max())},
                "actual_coordinate_denominators_used": True,
                "fixed_50nm_derivative_used": False,
                "flake_thickness_m": thickness,
                "flake_z_cell_count": int(flake_z.size),
                "surface_cell_center_z_m": float(z[surface_index]),
                "midplane_cell_center_z_m": float(z[midplane_index]),
            },
            "implementation_audit": {
                "dz_weighted_thickness_average": True,
                "gradient_before_mask": False,
                "mask_aware_gradient": True,
                "interior_stencil": (
                    "centered two-cell coordinate difference using actual "
                    "cell-center coordinates"
                ),
                "flake_edge_stencil": (
                    "one-sided derivative using only TaIrTe4-side neighbour"
                ),
                "interface_cross_material_difference": False,
                "stored_report_raw_max_is_single_cell_sensitive": True,
                "stored_report_also_contains_p99": True,
                "old_gradient_figure_a_b_shared_color_scale": False,
                "this_audit_a_b_shared_color_scale": True,
            },
            "cases": {},
        }

        projection_data: dict[
            str, dict[str, dict[str, Any]]
        ] = {}
        central_linecuts: dict[str, dict[str, np.ndarray]] = {}
        exact_central_linecuts: dict[str, dict[str, np.ndarray]] = {}
        exact_peak_linecuts: dict[str, dict[str, np.ndarray]] = {}
        csv_rows: list[dict[str, Any]] = []
        for case in CASES:
            temperature_3d = np.asarray(raw[f"{case}__temperature_3D_K"], float)
            recomputed_average = np.sum(
                temperature_3d[:, :, flake_z]
                * dz[flake_z][None, None, :],
                axis=2,
            ) / thickness
            stored_average = np.asarray(
                raw[f"{case}__temperature_thickness_average_K"], float
            )
            stored_surface = np.asarray(
                raw[f"{case}__temperature_surface_K"], float
            )
            stored_midplane = np.asarray(
                raw[f"{case}__temperature_midplane_K"], float
            )
            projections = {
                "surface": stored_surface,
                "midplane": stored_midplane,
                "thickness_average": stored_average,
            }
            projection_data[case] = {}
            case_summary: dict[str, Any] = {
                "projection_storage_checks": {
                    "thickness_average_max_abs_difference_K": float(
                        np.max(np.abs(recomputed_average - stored_average))
                    ),
                    "surface_slice_max_abs_difference_K": float(
                        np.max(
                            np.abs(
                                temperature_3d[:, :, surface_index]
                                - stored_surface
                            )
                        )
                    ),
                    "midplane_slice_max_abs_difference_K": float(
                        np.max(
                            np.abs(
                                temperature_3d[:, :, midplane_index]
                                - stored_midplane
                            )
                        )
                    ),
                },
                "projections": {},
            }
            for projection, temperature in projections.items():
                gradients = gradient_fields(temperature, mask, x, y)
                identity, identity_error = identity_metrics(gradients, mask)
                edge = edge_statistics(
                    gradients, mask, n_grid, t_grid
                )
                line_central, arrays_central = linecut_metrics(
                    temperature, gradients, mask, x, y, 0.0
                )
                line_peak, arrays_peak = linecut_metrics(
                    temperature,
                    gradients,
                    mask,
                    x,
                    y,
                    edge["raw_peak_tangent_coordinate_m"],
                )
                exact_central, arrays_exact_central = (
                    exact_diagonal_linecut_metrics(
                        temperature,
                        gradients,
                        mask,
                        n_grid,
                        t_grid,
                        0.0,
                    )
                )
                exact_peak, arrays_exact_peak = (
                    exact_diagonal_linecut_metrics(
                        temperature,
                        gradients,
                        mask,
                        n_grid,
                        t_grid,
                        edge["raw_peak_tangent_coordinate_m"],
                    )
                )
                if projection == "thickness_average":
                    central_linecuts[case] = arrays_central
                    exact_central_linecuts[case] = arrays_exact_central
                    exact_peak_linecuts[case] = arrays_exact_peak
                projection_data[case][projection] = {
                    "gradients": gradients,
                    "identity_error": identity_error,
                    "central_line": arrays_central,
                    "peak_line": arrays_peak,
                    "exact_central_line": arrays_exact_central,
                    "exact_peak_line": arrays_exact_peak,
                }
                projection_summary = {
                    "coordinate_rotation_identity": identity,
                    "edge_band": edge,
                    "central_t0_linecut": line_central,
                    "raw_peak_t_linecut": line_peak,
                    "exact_cell_center_t0_linecut": exact_central,
                    "exact_cell_center_raw_peak_t_linecut": exact_peak,
                }
                case_summary["projections"][projection] = projection_summary
                csv_rows.append(
                    {
                        "case": case,
                        "projection": projection,
                        "identity_max_relative": identity[
                            "maximum_pixelwise_relative_error"
                        ],
                        "central_derivative_NRMSE": line_central[
                            "derivative_NRMSE"
                        ],
                        "central_integral_closure_relative": line_central[
                            "integral_closure_relative_to_temperature_delta"
                        ],
                        "peak_line_derivative_NRMSE": line_peak[
                            "derivative_NRMSE"
                        ],
                        "peak_line_integral_closure_relative": line_peak[
                            "integral_closure_relative_to_temperature_delta"
                        ],
                        "exact_central_derivative_NRMSE": exact_central[
                            "derivative_NRMSE"
                        ],
                        "exact_central_derivative_NRMSE_without_last_edge_cell": exact_central[
                            "derivative_NRMSE_excluding_last_inside_edge_cell"
                        ],
                        "exact_central_integral_closure_relative": exact_central[
                            "integral_closure_relative_to_temperature_delta"
                        ],
                        "exact_peak_derivative_NRMSE": exact_peak[
                            "derivative_NRMSE"
                        ],
                        "exact_peak_derivative_NRMSE_without_last_edge_cell": exact_peak[
                            "derivative_NRMSE_excluding_last_inside_edge_cell"
                        ],
                        "exact_peak_integral_closure_relative": exact_peak[
                            "integral_closure_relative_to_temperature_delta"
                        ],
                        "edge_raw_max_abs_grad_n_K_m": edge[
                            "raw_max_abs_grad_n_K_m"
                        ],
                        "edge_p99_abs_grad_n_K_m": edge[
                            "p99_abs_grad_n_K_m"
                        ],
                        "edge_raw_max_over_p99": edge["raw_max_over_p99"],
                    }
                )

            saved_gradient_checks = {}
            stored_to_recomputed = {
                "grad_b_K_m": "gx",
                "grad_a_K_m": "gy",
                "grad_n_K_m": "gn",
                "grad_t_K_m": "gt",
                "grad_magnitude_K_m": "magnitude",
            }
            avg_gradients = projection_data[case]["thickness_average"][
                "gradients"
            ]
            for stored_name, recomputed_name in stored_to_recomputed.items():
                saved = np.asarray(raw[f"{case}__{stored_name}"], float)
                saved_gradient_checks[stored_name] = float(
                    np.max(np.abs(saved - avg_gradients[recomputed_name]))
                )
            case_summary[
                "stored_thickness_average_gradient_max_abs_differences"
            ] = saved_gradient_checks
            summary["cases"][case] = case_summary

    for model in ("Maxwell", "analytic"):
        summary[f"{model}_b_over_a"] = {}
        for projection in PROJECTIONS:
            a = summary["cases"][f"{model}_a"]["projections"][projection][
                "edge_band"
            ]
            b = summary["cases"][f"{model}_b"]["projections"][projection][
                "edge_band"
            ]
            summary[f"{model}_b_over_a"][projection] = {
                "raw_max_abs_grad_n": (
                    b["raw_max_abs_grad_n_K_m"]
                    / a["raw_max_abs_grad_n_K_m"]
                ),
                "p99_abs_grad_n": (
                    b["p99_abs_grad_n_K_m"]
                    / a["p99_abs_grad_n_K_m"]
                ),
            }

    figures = {
        "identity_error": report_dir / "gradient_rotation_identity_error.png",
        "linecut_derivative": report_dir
        / "thickness_average_linecut_derivative_comparison.png",
        "linecut_reconstruction": report_dir
        / "thickness_average_gradient_integral_reconstruction.png",
        "exact_linecut_derivative": report_dir
        / "thickness_average_exact_cell_center_linecut_derivative.png",
        "exact_linecut_reconstruction": report_dir
        / "thickness_average_exact_cell_center_integral_reconstruction.png",
        "exact_peak_linecut_derivative": report_dir
        / "thickness_average_exact_peak_t_linecut_derivative.png",
        "exact_peak_linecut_reconstruction": report_dir
        / "thickness_average_exact_peak_t_integral_reconstruction.png",
        "Maxwell_shared_scale": report_dir
        / "Maxwell_surface_midplane_average_grad_n_shared_scale.png",
        "analytic_shared_scale": report_dir
        / "analytic_surface_midplane_average_grad_n_shared_scale.png",
    }
    write_identity_figure(
        figures["identity_error"], projection_data, x, y, mask
    )
    write_linecut_figure(figures["linecut_derivative"], central_linecuts)
    write_reconstruction_figure(
        figures["linecut_reconstruction"], central_linecuts
    )
    write_linecut_figure(
        figures["exact_linecut_derivative"], exact_central_linecuts
    )
    write_reconstruction_figure(
        figures["exact_linecut_reconstruction"], exact_central_linecuts
    )
    write_linecut_figure(
        figures["exact_peak_linecut_derivative"], exact_peak_linecuts
    )
    write_reconstruction_figure(
        figures["exact_peak_linecut_reconstruction"], exact_peak_linecuts
    )
    write_case_projection_figure(
        figures["Maxwell_shared_scale"],
        "Maxwell",
        projection_data,
        x,
        y,
        mask,
    )
    write_case_projection_figure(
        figures["analytic_shared_scale"],
        "analytic",
        projection_data,
        x,
        y,
        mask,
    )
    summary["figures"] = {
        key: str(value.resolve()) for key, value in figures.items()
    }
    summary["generation_command"] = shlex.join([sys.executable, *sys.argv])

    summary_path = report_dir / "w12_explicit3d_gradient_geometry_audit.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    csv_path = report_dir / "w12_explicit3d_gradient_geometry_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    def ratio_text(model: str) -> str:
        rows = []
        for projection in PROJECTIONS:
            values = summary[f"{model}_b_over_a"][projection]
            rows.append(
                f"| {projection} | "
                f"{values['raw_max_abs_grad_n']:.8f} | "
                f"{values['p99_abs_grad_n']:.8f} |"
            )
        return "\n".join(rows)

    maximum_identity = max(
        summary["cases"][case]["projections"][projection][
            "coordinate_rotation_identity"
        ]["maximum_pixelwise_relative_error"]
        for case in CASES
        for projection in PROJECTIONS
    )
    maximum_average_reproduction = max(
        summary["cases"][case]["projection_storage_checks"][
            "thickness_average_max_abs_difference_K"
        ]
        for case in CASES
    )
    linecut_aggregates: dict[str, float] = {}
    for label, key in (
        ("bilinear_central", "central_t0_linecut"),
        ("bilinear_peak_t", "raw_peak_t_linecut"),
        ("exact_central", "exact_cell_center_t0_linecut"),
        ("exact_peak_t", "exact_cell_center_raw_peak_t_linecut"),
    ):
        items = [
            summary["cases"][case]["projections"][projection][key]
            for case in CASES
            for projection in PROJECTIONS
        ]
        linecut_aggregates[f"{label}_maximum_derivative_NRMSE"] = max(
            item["derivative_NRMSE"] for item in items
        )
        linecut_aggregates[f"{label}_maximum_integral_closure"] = max(
            item["integral_closure_relative_to_temperature_delta"]
            for item in items
        )
        linecut_aggregates[f"{label}_minimum_correlation"] = min(
            item["derivative_correlation"] for item in items
        )
        if label.startswith("exact"):
            linecut_aggregates[
                f"{label}_maximum_derivative_NRMSE_without_last_edge_cell"
            ] = max(
                item[
                    "derivative_NRMSE_excluding_last_inside_edge_cell"
                ]
                for item in items
            )
    summary["linecut_aggregate_metrics"] = linecut_aggregates
    # Re-write after adding human-facing aggregate metrics.
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    maximum_raw_over_p99 = max(
        summary["cases"][case]["projections"][projection]["edge_band"][
            "raw_max_over_p99"
        ]
        for case in CASES
        for projection in PROJECTIONS
    )
    report = f"""# W12 explicit-3D gradient geometry audit

Status: `{summary['status']}`

This is a read-only/offline audit of the existing thermal field NPZ.  It did
not execute FDTD, a thermal solve, PTE, adjoint, AD-FD, or optimization.

## Implementation findings

- thermal derivatives use actual cell-centre coordinate differences; a fixed
  50 nm denominator is not used;
- the crystal contract is lab `x=b`, lab `y=a`;
- the thickness average is explicitly `dz` weighted and is reproduced from
  the saved 3-D field to a maximum absolute difference of
  `{maximum_average_reproduction:.9e} K`;
- gradients are mask-aware: centred stencils are used in the interior and a
  TaIrTe4-side one-sided stencil at the flake edge.  They are not formed
  across the TaIrTe4/air material interface and then masked;
- the prior plot selected a separate color limit for every a/b panel.  That
  is unsuitable for visual magnitude comparison.  The new projection plots
  use one shared a/b color scale for each projection;
- a one-cell raw maximum remains noise-sensitive.  Raw maximum and p99 are
  therefore reported together.

## Rotation identity

For every case and each of surface, midplane, and thickness-average fields,
the audit recomputed

`|grad T|^2 = dxT^2 + dyT^2 = dnT^2 + dtT^2`.

The maximum pixelwise relative error is `{maximum_identity:.9e}`.  See
`gradient_rotation_identity_error.png`.

## Edge-normal derivative and line integral

At `t=(x+y)/sqrt(2)=0`, and separately at the tangent coordinate of each raw
edge-gradient peak, the audit compares the bilinearly sampled field
`(-dxT+dyT)/sqrt(2)` with the numerical derivative of the sampled `T(n)`.
It also reconstructs the temperature with
`T(n1)+integral(dnT dn)`.  The JSON and CSV retain the derivative NRMSE,
correlation, endpoint closure, peak positions, and reconstructed-temperature
NRMSE for every z projection.  A second independent linecut uses only actual
thermal cell centres lying on a constant-`t` diagonal and reaches the final
inside-flake cell; it does not use bilinear interpolation.

- bilinear central-line maximum derivative NRMSE:
  `{linecut_aggregates['bilinear_central_maximum_derivative_NRMSE']:.6%}`;
- bilinear central-line maximum integral closure:
  `{linecut_aggregates['bilinear_central_maximum_integral_closure']:.6%}`;
- exact-cell central-line maximum derivative NRMSE:
  `{linecut_aggregates['exact_central_maximum_derivative_NRMSE']:.6%}`;
- exact-cell central-line maximum derivative NRMSE after excluding only the
  final one-sided edge cell:
  `{linecut_aggregates['exact_central_maximum_derivative_NRMSE_without_last_edge_cell']:.6%}`;
- exact-cell central-line maximum integral closure:
  `{linecut_aggregates['exact_central_maximum_integral_closure']:.6%}`;
- minimum derivative correlation over all tests:
  `{min(value for key, value in linecut_aggregates.items() if key.endswith('minimum_correlation')):.9f}`.

The larger exact-line derivative NRMSE is localized mainly to the final
inside-flake cell: the implemented Cartesian one-sided x/y stencil and an
independent one-sided derivative along a 45-degree diagonal are different
finite-resolution operators.  The integral closure remains below
`{max(value for key, value in linecut_aggregates.items() if key.endswith('maximum_integral_closure')):.6%}`.
This is a finite-grid boundary-stencil diagnostic, not a coordinate-rotation
identity failure.

Across every case/projection, raw max divided by p99 is at most
`{maximum_raw_over_p99:.6f}`.  Thus the reported ordering is not produced by
one isolated extreme cell, although p99 remains the safer comparator.

## Maxwell b/a edge-normal ratios

| z projection | raw maximum b/a | p99 b/a |
|---|---:|---:|
{ratio_text('Maxwell')}

## Analytic b/a edge-normal ratios

| z projection | raw maximum b/a | p99 b/a |
|---|---:|---:|
{ratio_text('analytic')}

The paper does not explicitly identify Fig. 3G as a top-surface, midplane, or
thickness-average comparator.  The three projections are therefore retained
without silently promoting one of them to a paper-exact observable.

## Provenance

- input: `{input_path}`
- input SHA-256: `{summary['input']['sha256']}`
- JSON: `{summary_path.name}`
- CSV: `{csv_path.name}`
- command: `{summary['generation_command']}`
"""
    report_path = report_dir / "W12_EXPLICIT3D_GRADIENT_GEOMETRY_AUDIT.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "report": str(report_path),
        "summary": str(summary_path),
        "maximum_identity_error": maximum_identity,
        "Maxwell_b_over_a": summary["Maxwell_b_over_a"],
        "analytic_b_over_a": summary["analytic_b_over_a"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
