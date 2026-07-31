#!/usr/bin/env python3
"""Compare 50-nm Maxwell and analytic volumetric Q in one explicit 3D FVM.

This command never opens Lumerical.  It consumes the completed 50-nm
straight-edge a/b optical artifacts, preserves all three absorption
components, and applies the exact explicit-3D geometry, material, interface,
boundary, and conservative-remap contract used by the earlier edge-a
downstream audit.

The analytic source is cell-integrated from the Gaussian--Beer--Lambert
volume law.  The paper-reduced thickness-integrated sheet calculation is a
separate optional control and is not used here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import erf, sqrt
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
FVM_DIR = HERE.parent / "photothermal_stage1"
for location in (REPOSITORY, FVM_DIR):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from anisotropic_heat_fvm import solve_assembled_thermal_system  # noqa: E402
from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    analyze_w12_interface_downstream as downstream,
)
from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    compare_w12_50nm_maxwell_analytic_paper_reduced as audit,
)
from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    run_device_a_explicit_thermal_pte as thermal,
)
from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    run_straight_edge_analytic_q_control as analytic_base,
)


STATUS_PASS = (
    "COMPLETED_W12_50NM_MAXWELL_ANALYTIC_EXPLICIT3D_THERMAL_SANITY"
)
STATUS_BLOCKED = "BLOCKED_W12_50NM_MAXWELL_ANALYTIC_EXPLICIT3D_GATE"
INCIDENT_POWER_W = 285.0e-6
TMM_ABSORPTION = {"a": 0.17673296, "b": 0.26328721}
GATE = 5.0e-3
RESIDUAL_GATE = 1.0e-8
ENERGY_GATE = 1.0e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-a-dir", type=Path, required=True)
    parser.add_argument("--edge-b-dir", type=Path, required=True)
    parser.add_argument("--incident-reference-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
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


def cell_volume(geometry: thermal.Geometry) -> np.ndarray:
    return (
        np.diff(geometry.x_edges_m)[:, None, None]
        * np.diff(geometry.y_edges_m)[None, :, None]
        * np.diff(geometry.z_edges_m)[None, None, :]
    )


def cell_area(geometry: thermal.Geometry) -> np.ndarray:
    return (
        np.diff(geometry.x_edges_m)[:, None]
        * np.diff(geometry.y_edges_m)[None, :]
    )


def relative(left: float, right: float) -> float:
    return abs(left - right) / max(abs(right), np.finfo(float).tiny)


def weighted_nrmse(
    left: np.ndarray,
    right: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    a = np.asarray(left, float)
    b = np.asarray(right, float)
    w = np.broadcast_to(np.asarray(weights, float), a.shape)
    selected = np.ones(a.shape, bool) if mask is None else np.asarray(mask, bool)
    numerator = float(np.sum(w[selected] * (a[selected] - b[selected]) ** 2))
    denominator = float(np.sum(w[selected] * b[selected] ** 2))
    return float(
        np.sqrt(numerator / max(denominator, np.finfo(float).tiny))
    )


def exact_gaussian_cell_fractions(
    edges_m: np.ndarray,
    center_m: float,
    sigma_m: float,
) -> np.ndarray:
    scaled = (np.asarray(edges_m, float) - center_m) / (sqrt(2.0) * sigma_m)
    cdf = 0.5 * np.asarray([1.0 + erf(float(value)) for value in scaled])
    return np.diff(cdf)


def analytic_volumetric_q(
    geometry: thermal.Geometry,
    polarization: str,
    beam: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Exact cell-integral of eta*P*beta*exp(-beta*d)*Gaussian."""
    constants = analytic_base.optical_constants(polarization)
    beta = float(constants["beta_m_inv"])
    absorbed_depth_fraction = float(
        constants["absorbed_depth_fraction_over_130nm"]
    )
    eta = TMM_ABSORPTION[polarization] / absorbed_depth_fraction
    fit = beam["fit"]
    sigma_x = float(fit["waist_x_m"]) / 2.0
    sigma_y = float(fit["waist_y_m"]) / 2.0
    x0 = float(fit["center_x_m"])
    y0 = float(fit["center_y_m"])
    fraction_x = exact_gaussian_cell_fractions(
        geometry.x_edges_m, x0, sigma_x
    )
    fraction_y = exact_gaussian_cell_fractions(
        geometry.y_edges_m, y0, sigma_y
    )
    z_lower = geometry.z_edges_m[:-1]
    z_upper = geometry.z_edges_m[1:]
    in_flake_depth = (z_lower >= -thermal.THICKNESS_M) & (z_upper <= 0.0)
    depth_fraction = np.zeros_like(z_lower)
    depth_near = -z_upper[in_flake_depth]
    depth_far = -z_lower[in_flake_depth]
    depth_fraction[in_flake_depth] = (
        np.exp(-beta * depth_near) - np.exp(-beta * depth_far)
    )
    energy = (
        eta
        * INCIDENT_POWER_W
        * fraction_x[:, None, None]
        * fraction_y[None, :, None]
        * depth_fraction[None, None, :]
    )
    energy = np.where(geometry.flake_mask, energy, 0.0)
    volume = cell_volume(geometry)
    q = np.divide(energy, volume, out=np.zeros_like(energy), where=volume > 0)
    power = float(np.sum(energy))
    full_plane_power = INCIDENT_POWER_W * TMM_ABSORPTION[polarization]
    return q, {
        "equation": (
            "Q=eta*P*beta*exp(-beta*d)/(2*pi*sigma_x*sigma_y)"
            "*exp(-(x-x0)^2/(2*sigma_x^2)-(y-y0)^2/(2*sigma_y^2)); "
            "d=-z is depth from the top TaIrTe4 surface"
        ),
        "discretization": (
            "analytic Gaussian and Beer-Lambert factors integrated exactly "
            "over every explicit-3D target control volume; no sheet collapse"
        ),
        "polarization": polarization,
        "incident_power_W": INCIDENT_POWER_W,
        "TMM_absorption": TMM_ABSORPTION[polarization],
        "eta_entrance_factor": eta,
        "beta_m_inv": beta,
        "absorbed_depth_fraction_over_130nm": absorbed_depth_fraction,
        "sigma_x_m": sigma_x,
        "sigma_y_m": sigma_y,
        "realized_waist_x_m": 2.0 * sigma_x,
        "realized_waist_y_m": 2.0 * sigma_y,
        "beam_center_m": [x0, y0],
        "requested_full_plane_absorbed_power_W": full_plane_power,
        "finite_half_plane_absorbed_power_W": power,
        "finite_half_plane_fraction_of_full_plane": power / full_plane_power,
        "minimum_Q_W_m3": float(np.min(q)),
        "negative_Q_cell_count": int(np.count_nonzero(q < 0.0)),
        "nonfinite_Q_cell_count": int(np.count_nonzero(~np.isfinite(q))),
    }


def map_maxwell_components(
    optical: audit.OpticalInput,
    geometry: thermal.Geometry,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Use the exact edge-a downstream remap for Qx, Qy, and Qz."""
    result = optical.result
    target_edges = (
        geometry.x_edges_m,
        geometry.y_edges_m,
        geometry.z_edges_m,
    )
    target_volume = downstream.volume(target_edges)
    incident_at_unit = float(
        result["run_result"]["normalization"]["incident_power_W_at_1_W_m2"]
    )
    physical_scale = INCIDENT_POWER_W / incident_at_unit
    mapped: dict[str, np.ndarray] = {}
    component_audit: dict[str, Any] = {}
    with np.load(optical.q_path, allow_pickle=False) as artifact:
        indices, source_edges, _ = downstream.source_grid(result, artifact)
        source_volume = downstream.volume(source_edges)
        operators = tuple(
            downstream.overlap_fraction(target, source)
            for target, source in zip(target_edges, source_edges)
        )
        for component, key in (
            ("x", "Qx_W_m3"),
            ("y", "Qy_W_m3"),
            ("z", "Qz_W_m3"),
        ):
            q_source = (
                np.asarray(artifact[key][np.ix_(*indices)], float)
                * physical_scale
            )
            source_energy = q_source * source_volume
            target_energy = downstream.remap_energy(source_energy, operators)
            projected_energy, projection = downstream.project_energy_to_support(
                target_energy,
                target_edges,
                geometry.flake_mask,
            )
            q_target = np.divide(
                projected_energy,
                target_volume,
                out=np.zeros_like(projected_energy),
                where=target_volume > 0,
            )
            p_source = float(np.sum(source_energy))
            p_target = float(np.sum(projected_energy))
            mapped[component] = q_target
            component_audit[component] = {
                "source_power_W": p_source,
                "target_power_W": p_target,
                "exact_overlap_power_error": relative(
                    float(np.sum(target_energy)), p_source
                ),
                "final_mapping_power_error": relative(p_target, p_source),
                "projection": projection,
            }
    mapped["total"] = mapped["x"] + mapped["y"] + mapped["z"]
    sum_components = sum(
        component_audit[component]["target_power_W"] for component in "xyz"
    )
    total_power = float(np.sum(mapped["total"] * target_volume))
    return mapped, {
        "optical_artifact": artifact_record(
            optical.q_path, f"edge_{optical.polarization}_raw_Q_NPZ"
        ),
        "optical_result": artifact_record(
            optical.result_path, f"edge_{optical.polarization}_case_result"
        ),
        "incident_power_W": INCIDENT_POWER_W,
        "unit_central_intensity_incident_power_W": incident_at_unit,
        "physical_scale": physical_scale,
        "components": component_audit,
        "sum_component_target_power_W": sum_components,
        "reintegrated_total_target_power_W": total_power,
        "component_sum_relative_error": relative(total_power, sum_components),
        "outside_flake_nonzero_count": int(
            np.count_nonzero(mapped["total"][~geometry.flake_mask])
        ),
        "Q_operations": {
            "clipping": False,
            "smoothing": False,
            "gain": False,
            "global_rescaling": False,
            "polarization_matching_rescaling": False,
            "tiling": False,
            "source_deletion": False,
        },
        "remap_contract": (
            "identical to the existing edge-a downstream audit: bounded "
            "source dual cells, exact Cartesian overlap energy remap, then "
            "one physical-3D nearest TaIrTe4-support projection with exact "
            "distance ties split uniformly"
        ),
    }


def solve_cases(
    system: Any,
    sources: dict[str, np.ndarray],
) -> dict[str, Any]:
    solved: dict[str, Any] = {}
    previous = None
    for case_id in (
        "Maxwell_b",
        "Maxwell_a",
        "analytic_b",
        "analytic_a",
    ):
        print(f"THERMAL_SOLVE_START case={case_id}", flush=True)
        if previous is None:
            result = solve_assembled_thermal_system(
                system,
                source_W_m3=sources[case_id],
                relative_tolerance=1.0e-9,
                max_iterations=12000,
            )
        else:
            result = downstream.solve_warm_started(
                system, sources[case_id], previous.temperature_K
            )
        print(
            f"THERMAL_SOLVE_DONE case={case_id} "
            f"iterations={result.iterations} "
            f"residual={result.linear_residual_relative:.3e}",
            flush=True,
        )
        solved[case_id] = result
        previous = result
    return solved


def thermal_fields(
    solved: Any,
    geometry: thermal.Geometry,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    edge_metrics, edge_fields = thermal.straight_edge_temperature_metrics(
        solved.temperature_K, geometry
    )
    flake_z = np.flatnonzero(np.any(geometry.flake_mask, axis=(0, 1)))
    z = 0.5 * (geometry.z_edges_m[:-1] + geometry.z_edges_m[1:])
    surface_index = int(flake_z[-1])
    mid_index = int(flake_z[np.argmin(np.abs(z[flake_z] + 65.0e-9))])
    volume = cell_volume(geometry)
    return {
        "source_power_W": solved.source_power_W,
        "Tmax_rise_K": float(np.max(solved.temperature_K)),
        "TaIrTe4_Tmax_rise_K": float(
            np.max(solved.temperature_K[geometry.flake_mask])
        ),
        "TaIrTe4_volume_average_rise_K": thermal.measure_weighted_mean(
            solved.temperature_K,
            geometry.flake_mask,
            volume,
        ),
        "surface_z_cell_center_m": float(z[surface_index]),
        "midplane_z_cell_center_m": float(z[mid_index]),
        "linear_residual_relative": solved.linear_residual_relative,
        "energy_balance_relative_error": solved.energy_balance_relative_error,
        "boundary_power_out_W": solved.boundary_power_out_W,
        "iterations": solved.iterations,
        "solver": solved.solver,
        "straight_edge_metrics": edge_metrics,
    }, {
        "temperature_3D_K": solved.temperature_K,
        "temperature_surface_K": solved.temperature_K[:, :, surface_index],
        "temperature_midplane_K": solved.temperature_K[:, :, mid_index],
        "temperature_thickness_average_K": edge_fields[
            "temperature_flake_average_K"
        ],
        "grad_b_K_m": edge_fields["grad_T_x_K_m"],
        "grad_a_K_m": edge_fields["grad_T_y_K_m"],
        "grad_n_K_m": edge_fields["grad_T_normal_K_m"],
        "grad_t_K_m": edge_fields["grad_T_tangent_K_m"],
        "grad_magnitude_K_m": edge_fields["grad_T_magnitude_K_m"],
        "edge_window_mask": edge_fields["edge_window_mask"],
    }


def source_moments(q: np.ndarray, geometry: thermal.Geometry) -> dict[str, Any]:
    energy = q * cell_volume(geometry)
    power = float(np.sum(energy))
    x = 0.5 * (geometry.x_edges_m[:-1] + geometry.x_edges_m[1:])
    y = 0.5 * (geometry.y_edges_m[:-1] + geometry.y_edges_m[1:])
    z = 0.5 * (geometry.z_edges_m[:-1] + geometry.z_edges_m[1:])
    cx = float(np.sum(energy * x[:, None, None]) / power)
    cy = float(np.sum(energy * y[None, :, None]) / power)
    cz = float(np.sum(energy * z[None, None, :]) / power)
    return {
        "power_W": power,
        "centroid_m": {"x": cx, "y": cy, "z": cz},
        "second_central_moment_m2": {
            "xx": float(np.sum(energy * (x[:, None, None] - cx) ** 2) / power),
            "yy": float(np.sum(energy * (y[None, :, None] - cy) ** 2) / power),
            "zz": float(np.sum(energy * (z[None, None, :] - cz) ** 2) / power),
        },
    }


def comparison_metrics(
    source_a: np.ndarray,
    fields_a: dict[str, np.ndarray],
    source_b: np.ndarray,
    fields_b: dict[str, np.ndarray],
    geometry: thermal.Geometry,
) -> dict[str, float]:
    volume = cell_volume(geometry)
    area = cell_area(geometry)
    flake_xy = np.any(geometry.flake_mask, axis=2)
    vector_numerator = float(
        np.sum(
            area[flake_xy]
            * (
                (fields_a["grad_b_K_m"][flake_xy] - fields_b["grad_b_K_m"][flake_xy]) ** 2
                + (fields_a["grad_a_K_m"][flake_xy] - fields_b["grad_a_K_m"][flake_xy]) ** 2
            )
        )
    )
    vector_denominator = float(
        np.sum(
            area[flake_xy]
            * (
                fields_b["grad_b_K_m"][flake_xy] ** 2
                + fields_b["grad_a_K_m"][flake_xy] ** 2
            )
        )
    )
    return {
        "Q_volume_weighted_NRMSE": weighted_nrmse(
            source_a, source_b, volume, geometry.flake_mask
        ),
        "temperature_3D_volume_weighted_NRMSE": weighted_nrmse(
            fields_a["temperature_3D_K"],
            fields_b["temperature_3D_K"],
            volume,
        ),
        "TaIrTe4_temperature_3D_volume_weighted_NRMSE": weighted_nrmse(
            fields_a["temperature_3D_K"],
            fields_b["temperature_3D_K"],
            volume,
            geometry.flake_mask,
        ),
        "temperature_surface_area_weighted_NRMSE": weighted_nrmse(
            fields_a["temperature_surface_K"],
            fields_b["temperature_surface_K"],
            area,
            flake_xy,
        ),
        "temperature_midplane_area_weighted_NRMSE": weighted_nrmse(
            fields_a["temperature_midplane_K"],
            fields_b["temperature_midplane_K"],
            area,
            flake_xy,
        ),
        "gradient_vector_area_weighted_NRMSE": float(
            np.sqrt(
                vector_numerator
                / max(vector_denominator, np.finfo(float).tiny)
            )
        ),
    }


def profile(
    values: np.ndarray,
    geometry: thermal.Geometry,
    *,
    n_min_m: float = -12.0e-6,
    n_max_m: float = 0.5e-6,
    bin_width_m: float = 0.2e-6,
    tangent_window_m: float = 0.5e-6,
) -> tuple[np.ndarray, np.ndarray]:
    x = 0.5 * (geometry.x_edges_m[:-1] + geometry.x_edges_m[1:])
    y = 0.5 * (geometry.y_edges_m[:-1] + geometry.y_edges_m[1:])
    normal = (y[None, :] - x[:, None]) / np.sqrt(2.0)
    tangent = (x[:, None] + y[None, :]) / np.sqrt(2.0)
    bins = np.arange(n_min_m, n_max_m + bin_width_m, bin_width_m)
    centers = 0.5 * (bins[:-1] + bins[1:])
    result = np.full(centers.shape, np.nan)
    selected = np.abs(tangent) <= tangent_window_m
    indices = np.digitize(normal[selected], bins) - 1
    data = np.asarray(values, float)[selected]
    weights = cell_area(geometry)[selected]
    for index in range(centers.size):
        take = indices == index
        if np.any(take):
            result[index] = float(
                np.sum(data[take] * weights[take]) / np.sum(weights[take])
            )
    return centers, result


def map_figure(
    path: Path,
    geometry: thermal.Geometry,
    arrays: dict[str, dict[str, np.ndarray]],
    key: str,
    label: str,
) -> None:
    x = 0.5 * (geometry.x_edges_m[:-1] + geometry.x_edges_m[1:]) * 1e6
    y = 0.5 * (geometry.y_edges_m[:-1] + geometry.y_edges_m[1:]) * 1e6
    extent = [x[0], x[-1], y[0], y[-1]]
    mask = np.any(geometry.flake_mask, axis=2)
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for axis, case_id in zip(
        axes.flat, ("Maxwell_a", "Maxwell_b", "analytic_a", "analytic_b")
    ):
        data = np.where(mask, arrays[case_id][key], np.nan)
        handle = axis.imshow(
            data.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="magma",
        )
        axis.set(
            title=case_id.replace("_", " "),
            xlabel="lab x = b (µm)",
            ylabel="lab y = a (µm)",
        )
        figure.colorbar(handle, ax=axis, label=label)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def gradient_figure(
    path: Path,
    geometry: thermal.Geometry,
    fields: dict[str, dict[str, np.ndarray]],
    model: str,
) -> None:
    x = 0.5 * (geometry.x_edges_m[:-1] + geometry.x_edges_m[1:]) * 1e6
    y = 0.5 * (geometry.y_edges_m[:-1] + geometry.y_edges_m[1:]) * 1e6
    extent = [x[0], x[-1], y[0], y[-1]]
    mask = np.any(geometry.flake_mask, axis=2)
    columns = (
        ("grad_a_K_m", "∂aT"),
        ("grad_b_K_m", "∂bT"),
        ("grad_n_K_m", "∂nT"),
        ("grad_t_K_m", "∂tT"),
        ("grad_magnitude_K_m", "|∇T|"),
    )
    figure, axes = plt.subplots(2, 5, figsize=(22, 8), constrained_layout=True)
    for row, polarization in enumerate(("a", "b")):
        case_id = f"{model}_{polarization}"
        for column, (key, title) in enumerate(columns):
            data = np.where(mask, fields[case_id][key], np.nan)
            if key == "grad_magnitude_K_m":
                kwargs = {"cmap": "magma", "vmin": 0.0}
            else:
                limit = float(np.nanmax(np.abs(data)))
                kwargs = {"cmap": "coolwarm", "vmin": -limit, "vmax": limit}
            handle = axes[row, column].imshow(
                data.T,
                origin="lower",
                extent=extent,
                aspect="equal",
                **kwargs,
            )
            axes[row, column].set(
                title=f"{model}, E∥{polarization}: {title}",
                xlabel="x=b (µm)",
                ylabel="y=a (µm)",
            )
            figure.colorbar(handle, ax=axes[row, column], label="K/m")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def profile_figure(
    path: Path,
    geometry: thermal.Geometry,
    sources: dict[str, np.ndarray],
    fields: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    dz = np.diff(geometry.z_edges_m)
    profiles: dict[str, Any] = {}
    figure, axes = plt.subplots(3, 2, figsize=(13, 13), constrained_layout=True)
    for column, polarization in enumerate(("a", "b")):
        profiles[polarization] = {}
        for model, style in (("Maxwell", "-"), ("analytic", "--")):
            case_id = f"{model}_{polarization}"
            q_areal = np.sum(sources[case_id] * dz[None, None, :], axis=2)
            values = (
                ("q_areal", q_areal),
                (
                    "temperature",
                    fields[case_id]["temperature_thickness_average_K"],
                ),
                ("grad_n", fields[case_id]["grad_n_K_m"]),
            )
            profiles[polarization][model] = {}
            for row, (name, value) in enumerate(values):
                n, data = profile(value, geometry)
                profiles[polarization][model][name] = {
                    "n_m": n,
                    "value": data,
                }
                axes[row, column].plot(n * 1e6, data, style, label=model)
        for row, ylabel in enumerate(
            ("depth-integrated q'' (W/m²)", "ΔT (K)", "∂nT (K/m)")
        ):
            axes[row, column].axvline(0.0, color="k", linestyle=":", linewidth=1)
            axes[row, column].set(
                title=f"E∥{polarization}",
                xlabel="edge-normal n=(y-x)/√2 (µm)",
                ylabel=ylabel,
            )
            axes[row, column].grid(alpha=0.25)
            axes[row, column].legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return profiles


def depth_figure(
    path: Path,
    geometry: thermal.Geometry,
    sources: dict[str, np.ndarray],
) -> None:
    volume = cell_volume(geometry)
    z = 0.5 * (geometry.z_edges_m[:-1] + geometry.z_edges_m[1:])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, polarization in zip(axes, ("a", "b")):
        for model, style in (("Maxwell", "-o"), ("analytic", "--s")):
            case_id = f"{model}_{polarization}"
            power_z = np.sum(sources[case_id] * volume, axis=(0, 1))
            selected = power_z > 0.0
            axis.plot(
                z[selected] * 1e9,
                power_z[selected] / np.sum(power_z),
                style,
                markersize=3,
                label=model,
            )
        axis.set(
            title=f"E∥{polarization}",
            xlabel="z cell center (nm)",
            ylabel="fraction of absorbed power per z cell",
        )
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_cases_csv(
    path: Path,
    metrics: dict[str, dict[str, Any]],
) -> None:
    rows = []
    for case_id, item in metrics.items():
        edge = item["straight_edge_metrics"]
        rows.append(
            {
                "case_id": case_id,
                "source_power_W": item["source_power_W"],
                "Tmax_rise_K": item["Tmax_rise_K"],
                "TaIrTe4_volume_average_rise_K": item[
                    "TaIrTe4_volume_average_rise_K"
                ],
                "max_abs_grad_a_K_m": edge["max_abs_grad_T_y_K_m"],
                "max_abs_grad_b_K_m": edge["max_abs_grad_T_x_K_m"],
                "max_abs_grad_n_K_m": edge[
                    "max_abs_edge_normal_gradient_K_m"
                ],
                "max_abs_grad_t_K_m": edge[
                    "max_abs_edge_tangent_gradient_K_m"
                ],
                "max_grad_magnitude_K_m": edge[
                    "max_inplane_gradient_K_m"
                ],
                "linear_residual_relative": item[
                    "linear_residual_relative"
                ],
                "energy_balance_relative_error": item[
                    "energy_balance_relative_error"
                ],
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    ratio = summary["polarization_ratios_b_over_a"]
    cross = summary["Maxwell_vs_analytic_same_incident_power"]
    cases = summary["cases"]
    text = f"""# W12 50-nm Maxwell vs analytic explicit-3D thermal sanity

Status: `{summary['status']}`

This is the primary **existing inverse-design explicit-3D thermal FVM
sanity comparison**. It is not a paper reproduction. The separately
checkpointed thickness-integrated paper-reduced sheet calculation is an
optional control only and is not used by any result below.

## Fixed contracts

- completed 50-nm straight-edge a/b optical artifacts; no new 25-nm or
  12.5-nm optical refinement
- full Maxwell `Qx+Qy+Qz` retained on the 3D source grid
- same 285-µW incident power; no polarization matching or Q rescaling
- exact-overlap plus nearest-TaIrTe4-support conservative 3D remap used by
  the existing edge-a downstream audit
- one common explicit-3D operator: 60-µm lateral domain, 20-µm Si depth,
  100-nm core x/y, 10-nm TaIrTe4 dz, 285-nm SiO2, 600-nm air
- `kTaIrTe4=(3.8,14.4,1.0) W/(m K)` in lab `(x=b,y=a,z=c)`;
  `kSiO2=1.38`, `kSi=145`, `kair=0.026 W/(m K)`
- `G(TaIrTe4/air)=1`, `G(TaIrTe4/SiO2)=7.37e6`,
  `G(SiO2/Si)=1.1e9 W/(m² K)`
- far x/y and bottom fixed `DeltaT=0`; exposed-surface `h=10 W/(m² K)`
- no PTE, weighting potential, adjoint, AD-FD, or optimization

The analytic source is the volumetric Gaussian--Beer--Lambert law integrated
exactly over every target cell. It is not collapsed to a 130-nm sheet.

## Primary same-incident-power results

| case | Pabs (W) | Tmax (K) | TaIrTe4 mean (K) | max |dT/dn| (K/m) | max |grad T| (K/m) |
|---|---:|---:|---:|---:|---:|
| Maxwell a | {cases['Maxwell_a']['source_power_W']:.9e} | {cases['Maxwell_a']['Tmax_rise_K']:.9e} | {cases['Maxwell_a']['TaIrTe4_volume_average_rise_K']:.9e} | {cases['Maxwell_a']['straight_edge_metrics']['max_abs_edge_normal_gradient_K_m']:.9e} | {cases['Maxwell_a']['straight_edge_metrics']['max_inplane_gradient_K_m']:.9e} |
| Maxwell b | {cases['Maxwell_b']['source_power_W']:.9e} | {cases['Maxwell_b']['Tmax_rise_K']:.9e} | {cases['Maxwell_b']['TaIrTe4_volume_average_rise_K']:.9e} | {cases['Maxwell_b']['straight_edge_metrics']['max_abs_edge_normal_gradient_K_m']:.9e} | {cases['Maxwell_b']['straight_edge_metrics']['max_inplane_gradient_K_m']:.9e} |
| analytic a | {cases['analytic_a']['source_power_W']:.9e} | {cases['analytic_a']['Tmax_rise_K']:.9e} | {cases['analytic_a']['TaIrTe4_volume_average_rise_K']:.9e} | {cases['analytic_a']['straight_edge_metrics']['max_abs_edge_normal_gradient_K_m']:.9e} | {cases['analytic_a']['straight_edge_metrics']['max_inplane_gradient_K_m']:.9e} |
| analytic b | {cases['analytic_b']['source_power_W']:.9e} | {cases['analytic_b']['Tmax_rise_K']:.9e} | {cases['analytic_b']['TaIrTe4_volume_average_rise_K']:.9e} | {cases['analytic_b']['straight_edge_metrics']['max_abs_edge_normal_gradient_K_m']:.9e} | {cases['analytic_b']['straight_edge_metrics']['max_inplane_gradient_K_m']:.9e} |

| model | Pabs b/a | Tmax b/a | mean T b/a | max |dT/dn| b/a | max |grad T| b/a |
|---|---:|---:|---:|---:|---:|
| Maxwell | {ratio['Maxwell']['absorbed_power']:.6f} | {ratio['Maxwell']['Tmax']:.6f} | {ratio['Maxwell']['TaIrTe4_mean']:.6f} | {ratio['Maxwell']['max_abs_grad_n']:.6f} | {ratio['Maxwell']['max_grad_magnitude']:.6f} |
| analytic | {ratio['analytic']['absorbed_power']:.6f} | {ratio['analytic']['Tmax']:.6f} | {ratio['analytic']['TaIrTe4_mean']:.6f} | {ratio['analytic']['max_abs_grad_n']:.6f} | {ratio['analytic']['max_grad_magnitude']:.6f} |

## Maxwell--analytic differences

| polarization | P ratio M/A | volumetric-Q NRMSE | 3D T NRMSE | flake 3D T NRMSE | gradient-vector NRMSE |
|---|---:|---:|---:|---:|---:|
| a | {cross['a']['absorbed_power_ratio_Maxwell_over_analytic']:.6f} | {cross['a']['Q_volume_weighted_NRMSE']:.6%} | {cross['a']['temperature_3D_volume_weighted_NRMSE']:.6%} | {cross['a']['TaIrTe4_temperature_3D_volume_weighted_NRMSE']:.6%} | {cross['a']['gradient_vector_area_weighted_NRMSE']:.6%} |
| b | {cross['b']['absorbed_power_ratio_Maxwell_over_analytic']:.6f} | {cross['b']['Q_volume_weighted_NRMSE']:.6%} | {cross['b']['temperature_3D_volume_weighted_NRMSE']:.6%} | {cross['b']['TaIrTe4_temperature_3D_volume_weighted_NRMSE']:.6%} | {cross['b']['gradient_vector_area_weighted_NRMSE']:.6%} |

Equal-absorbed-power comparisons are stored as separately named linearity
diagnostics. They do not modify either primary source.

All remap errors, residuals, energy balances, source moments, boundary powers,
surface/midplane maps, all five in-plane derivative fields, depth profiles,
and edge-normal profiles are in the JSON/NPZ/figures.

## Provenance

- generation commit: `{summary['generation_commit']}`
- generation command: `{summary['generation_command']}`
- raw NPZ/FSP files remain external and are path/size/SHA-256 inventoried
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    if output_dir.exists() or report_dir.exists():
        raise FileExistsError("output and report directories must be new")

    optical = {
        "a": audit.inspect_optical(args.edge_a_dir.resolve(), "a"),
        "b": audit.inspect_optical(args.edge_b_dir.resolve(), "b"),
    }
    beam = audit.measured_beam(args.incident_reference_npz.resolve())
    incident_a = optical["a"].q_summary["incident_power_W_at_1_W_m2"]
    incident_b = optical["b"].q_summary["incident_power_W_at_1_W_m2"]
    incident_difference = relative(incident_a, incident_b)
    optical_gate = (
        all(item.gates["all_before_remap"] for item in optical.values())
        and incident_difference < 1.0e-5
    )
    if not optical_gate:
        raise RuntimeError("50-nm a/b optical input gate failed")

    output_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    outer_um = 31.0
    thermal.FLAKE_VERTICES_UM = np.asarray(
        [
            [-outer_um, -outer_um],
            [outer_um, -outer_um],
            [outer_um, outer_um],
        ],
        float,
    )
    geometry = thermal.build_geometry(
        domain_m=60.0e-6,
        si_depth_m=20.0e-6,
        core_step_m=100.0e-9,
        flake_dz_m=10.0e-9,
    )
    system = downstream.assemble_downstream_system(geometry)

    sources: dict[str, np.ndarray] = {}
    mappings: dict[str, Any] = {}
    components: dict[str, dict[str, np.ndarray]] = {}
    for polarization in ("a", "b"):
        components[polarization], mappings[polarization] = (
            map_maxwell_components(optical[polarization], geometry)
        )
        sources[f"Maxwell_{polarization}"] = components[polarization]["total"]

    analytic_contracts: dict[str, Any] = {}
    for polarization in ("a", "b"):
        source, contract = analytic_volumetric_q(
            geometry, polarization, beam
        )
        sources[f"analytic_{polarization}"] = source
        analytic_contracts[polarization] = contract

    solved = solve_cases(system, sources)
    metrics: dict[str, dict[str, Any]] = {}
    fields: dict[str, dict[str, np.ndarray]] = {}
    for case_id, result in solved.items():
        metrics[case_id], fields[case_id] = thermal_fields(result, geometry)
        metrics[case_id]["source_moments"] = source_moments(
            sources[case_id], geometry
        )

    ratios: dict[str, dict[str, float]] = {}
    for model in ("Maxwell", "analytic"):
        a = metrics[f"{model}_a"]
        b = metrics[f"{model}_b"]
        ratios[model] = {
            "absorbed_power": b["source_power_W"] / a["source_power_W"],
            "Tmax": b["Tmax_rise_K"] / a["Tmax_rise_K"],
            "TaIrTe4_mean": (
                b["TaIrTe4_volume_average_rise_K"]
                / a["TaIrTe4_volume_average_rise_K"]
            ),
            "max_abs_grad_a": (
                b["straight_edge_metrics"]["max_abs_grad_T_y_K_m"]
                / a["straight_edge_metrics"]["max_abs_grad_T_y_K_m"]
            ),
            "max_abs_grad_b": (
                b["straight_edge_metrics"]["max_abs_grad_T_x_K_m"]
                / a["straight_edge_metrics"]["max_abs_grad_T_x_K_m"]
            ),
            "max_abs_grad_n": (
                b["straight_edge_metrics"][
                    "max_abs_edge_normal_gradient_K_m"
                ]
                / a["straight_edge_metrics"][
                    "max_abs_edge_normal_gradient_K_m"
                ]
            ),
            "max_abs_grad_t": (
                b["straight_edge_metrics"][
                    "max_abs_edge_tangent_gradient_K_m"
                ]
                / a["straight_edge_metrics"][
                    "max_abs_edge_tangent_gradient_K_m"
                ]
            ),
            "max_grad_magnitude": (
                b["straight_edge_metrics"]["max_inplane_gradient_K_m"]
                / a["straight_edge_metrics"]["max_inplane_gradient_K_m"]
            ),
        }

    cross: dict[str, Any] = {}
    equal_power: dict[str, Any] = {}
    for polarization in ("a", "b"):
        maxwell = f"Maxwell_{polarization}"
        analytic = f"analytic_{polarization}"
        cross[polarization] = {
            "absorbed_power_ratio_Maxwell_over_analytic": (
                metrics[maxwell]["source_power_W"]
                / metrics[analytic]["source_power_W"]
            ),
            **comparison_metrics(
                sources[maxwell],
                fields[maxwell],
                sources[analytic],
                fields[analytic],
                geometry,
            ),
        }
        scale = (
            metrics[maxwell]["source_power_W"]
            / metrics[analytic]["source_power_W"]
        )
        scaled_fields = {
            key: value * scale
            for key, value in fields[analytic].items()
            if key != "edge_window_mask"
        }
        scaled_fields["edge_window_mask"] = fields[analytic][
            "edge_window_mask"
        ]
        equal_power[polarization] = {
            "scope": (
                "diagnostic copy derived by linearity; primary Maxwell and "
                "analytic Q remain unchanged"
            ),
            "analytic_scale_to_Maxwell_power": scale,
            **comparison_metrics(
                sources[maxwell],
                fields[maxwell],
                sources[analytic] * scale,
                scaled_fields,
                geometry,
            ),
        }

    figure_paths = {
        "Q_areal": report_dir / "explicit3d_depth_integrated_Q.png",
        "Q_midplane": report_dir / "explicit3d_volumetric_Q_midplane.png",
        "temperature_surface": report_dir / "explicit3d_temperature_surface.png",
        "temperature_midplane": report_dir
        / "explicit3d_temperature_midplane.png",
        "gradient_magnitude": report_dir
        / "explicit3d_gradient_magnitude.png",
        "Maxwell_gradients": report_dir / "explicit3d_Maxwell_gradients.png",
        "analytic_gradients": report_dir / "explicit3d_analytic_gradients.png",
        "edge_profiles": report_dir / "explicit3d_edge_normal_profiles.png",
        "depth_profiles": report_dir / "explicit3d_Q_depth_profiles.png",
    }
    dz = np.diff(geometry.z_edges_m)
    flake_z = np.flatnonzero(np.any(geometry.flake_mask, axis=(0, 1)))
    mid_index = int(
        flake_z[
            np.argmin(
                np.abs(
                    0.5
                    * (
                        geometry.z_edges_m[flake_z]
                        + geometry.z_edges_m[flake_z + 1]
                    )
                    + 65.0e-9
                )
            )
        ]
    )
    q_maps = {
        case_id: {
            "Q_areal_W_m2": np.sum(source * dz[None, None, :], axis=2),
            "Q_midplane_W_m3": source[:, :, mid_index],
        }
        for case_id, source in sources.items()
    }
    map_figure(
        figure_paths["Q_areal"],
        geometry,
        q_maps,
        "Q_areal_W_m2",
        "q'' (W/m²)",
    )
    map_figure(
        figure_paths["Q_midplane"],
        geometry,
        q_maps,
        "Q_midplane_W_m3",
        "Q (W/m³)",
    )
    map_figure(
        figure_paths["temperature_surface"],
        geometry,
        fields,
        "temperature_surface_K",
        "ΔT (K)",
    )
    map_figure(
        figure_paths["temperature_midplane"],
        geometry,
        fields,
        "temperature_midplane_K",
        "ΔT (K)",
    )
    map_figure(
        figure_paths["gradient_magnitude"],
        geometry,
        fields,
        "grad_magnitude_K_m",
        "|∇T| (K/m)",
    )
    gradient_figure(
        figure_paths["Maxwell_gradients"], geometry, fields, "Maxwell"
    )
    gradient_figure(
        figure_paths["analytic_gradients"], geometry, fields, "analytic"
    )
    profiles = profile_figure(
        figure_paths["edge_profiles"], geometry, sources, fields
    )
    depth_figure(figure_paths["depth_profiles"], geometry, sources)

    raw_path = output_dir / "w12_50nm_maxwell_analytic_explicit3d_fields.npz"
    raw_payload: dict[str, np.ndarray] = {
        "x_edges_m": geometry.x_edges_m,
        "y_edges_m": geometry.y_edges_m,
        "z_edges_m": geometry.z_edges_m,
        "material_id": geometry.material_id,
        "flake_mask": geometry.flake_mask,
    }
    for case_id, source in sources.items():
        raw_payload[f"{case_id}__Q_total_W_m3"] = source
        raw_payload[f"{case_id}__Q_areal_W_m2"] = q_maps[case_id][
            "Q_areal_W_m2"
        ]
        for key, value in fields[case_id].items():
            raw_payload[f"{case_id}__{key}"] = value
    for polarization in ("a", "b"):
        for component in "xyz":
            raw_payload[
                f"Maxwell_{polarization}__Q{component}_W_m3"
            ] = components[polarization][component]
    np.savez_compressed(raw_path, **raw_payload)

    mapping_gate = all(
        item["final_mapping_power_error"] < GATE
        for polarization in mappings.values()
        for item in polarization["components"].values()
    )
    source_gate = all(
        np.all(np.isfinite(source)) and np.min(source) >= 0.0
        for source in sources.values()
    )
    residual_gate = all(
        item["linear_residual_relative"] < RESIDUAL_GATE
        for item in metrics.values()
    )
    energy_gate = all(
        item["energy_balance_relative_error"] < ENERGY_GATE
        for item in metrics.values()
    )
    gates = {
        "optical_50nm_a_b_contract": optical_gate,
        "same_incident_normalization_relative_difference_lt_1e_minus_5": (
            incident_difference < 1.0e-5
        ),
        "component_mapping_power_error_lt_0p5_percent": mapping_gate,
        "all_sources_finite_nonnegative": source_gate,
        "linear_residual_lt_1e_minus_8": residual_gate,
        "energy_balance_lt_1_percent": energy_gate,
    }
    gates["all"] = all(gates.values())
    status = STATUS_PASS if gates["all"] else STATUS_BLOCKED

    summary = {
        "status": status,
        "scope": (
            "primary existing-inverse-design explicit-3D FVM comparison of "
            "full Maxwell volumetric Q and analytic volumetric "
            "Gaussian-Beer-Lambert Q"
        ),
        "not_claimed": [
            "paper reproduction",
            "paper-reduced sheet result",
            "optical 25-nm or 12.5-nm mesh convergence",
            "PTE or terminal-current prediction",
        ],
        "optional_sheet_control": {
            "status": "preserved separately; not executed as this primary result",
            "used_by_primary_result": False,
            "primary_Q_collapsed_to_sheet": False,
        },
        "optical_mesh_contract": {
            "lateral_mesh_nm": 50.0,
            "edge_a_reused": True,
            "edge_b_GPU_solve_count_for_this_checkpoint": 1,
            "edge_b_postprocess": (
                "completed FSP reopened read-only with correct b incident "
                "reference; no second solve"
            ),
            "25nm_run_for_this_stage": False,
            "12p5nm_run_for_this_stage": False,
        },
        "beam": beam,
        "same_incident_power": {
            "physical_incident_power_W": INCIDENT_POWER_W,
            "unit_reference_a_W": incident_a,
            "unit_reference_b_W": incident_b,
            "relative_difference": incident_difference,
            "polarization_matching_rescaling": False,
        },
        "optical_inputs": {
            key: {
                "directory": str(item.directory.resolve()),
                "Q": item.q_summary,
                "gates": item.gates,
            }
            for key, item in optical.items()
        },
        "Maxwell_mapping": mappings,
        "analytic_source_contracts": analytic_contracts,
        "thermal_contract": {
            "identity": (
                "existing edge-a downstream explicit anisotropic/material/"
                "interface Cartesian FVM; not paper reduced Robin model"
            ),
            "lateral_domain_um": 60.0,
            "Si_depth_um": 20.0,
            "core_xy_cell_size_nm": 100.0,
            "flake_dz_nm": 10.0,
            "grid_shape": list(geometry.material_id.shape),
            "axis_mapping": "lab x=b, lab y=a, lab z=c",
            "TaIrTe4_thickness_nm": 130.0,
            "SiO2_thickness_nm": 285.0,
            "air_height_nm": 600.0,
            "kappa_TaIrTe4_lab_W_mK": [3.8, 14.4, 1.0],
            "kappa_SiO2_W_mK": thermal.KAPPA_SIO2_W_MK,
            "kappa_Si_W_mK": thermal.KAPPA_SI_W_MK,
            "kappa_air_W_mK": thermal.KAPPA_AIR_W_MK,
            "G_TaIrTe4_air_W_m2K": thermal.G_TAIRTE4_AIR_W_M2K,
            "G_TaIrTe4_SiO2_W_m2K": thermal.G_TAIRTE4_SIO2_W_M2K,
            "G_SiO2_Si_W_m2K": thermal.G_SIO2_SI_W_M2K,
            "far_xy_boundary": "fixed DeltaT=0 numerical truncation",
            "bottom_boundary": "fixed DeltaT=0 numerical truncation",
            "exposed_h_W_m2K": thermal.H_EXPOSED_W_M2K,
            "straight_edge": "TaIrTe4 y<=x; lateral material boundary explicit",
        },
        "cases": metrics,
        "polarization_ratios_b_over_a": ratios,
        "Maxwell_vs_analytic_same_incident_power": cross,
        "equal_absorbed_power_shape_diagnostic": equal_power,
        "edge_normal_profiles": profiles,
        "acceptance": gates,
        "raw_output": artifact_record(
            raw_path, "external_explicit3D_fields_NPZ"
        ),
        "figures": {
            key: str(path.resolve()) for key, path in figure_paths.items()
        },
        "PTE_run": False,
        "weighting_potential_run": False,
        "adjoint_run": False,
        "AD_FD_run": False,
        "optimization_run": False,
        "generation_commit": git_commit(),
        "generation_command": shlex.join([sys.executable, *sys.argv]),
    }

    summary_path = report_dir / "w12_50nm_maxwell_analytic_explicit3d_summary.json"
    csv_path = report_dir / "w12_50nm_maxwell_analytic_explicit3d_cases.csv"
    report_path = report_dir / (
        "W12_50NM_MAXWELL_VS_ANALYTIC_EXPLICIT3D_THERMAL_SANITY_REPORT.md"
    )
    summary_path.write_text(
        json.dumps(jsonable(summary), indent=2) + "\n", encoding="utf-8"
    )
    write_cases_csv(csv_path, metrics)
    write_report(report_path, summary)

    manifest_records = [
        artifact_record(
            args.incident_reference_npz.resolve(),
            "empty_stack_target_plane_incident_reference",
        ),
        summary["raw_output"],
        artifact_record(summary_path, "published_summary_JSON"),
        artifact_record(csv_path, "published_cases_CSV"),
        artifact_record(report_path, "published_report"),
    ]
    for item in optical.values():
        manifest_records.extend(item.artifacts)
    manifest_records.extend(
        artifact_record(path, f"figure_{name}")
        for name, path in figure_paths.items()
    )
    manifest = {
        "status": status,
        "generation_commit": summary["generation_commit"],
        "generation_command": summary["generation_command"],
        "raw_NPZ_and_FSP_committed_to_Git": False,
        "artifacts": manifest_records,
    }
    manifest_path = report_dir / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(jsonable(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(jsonable(summary), indent=2))
    return 0 if gates["all"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
