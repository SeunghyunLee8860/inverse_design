#!/usr/bin/env python3
"""Compare direct analytic Q with analytic Q passed through the current remap."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
BASE_PATH = HERE / "run_straight_edge_analytic_q_control.py"
MAPPING_PATH = (
    REPOSITORY
    / "photothermal_pte"
    / "finite_inverse_design"
    / "finite_q_mapping.py"
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("analytic_q_remap_base", BASE_PATH)
mapping = load_module("analytic_q_remap_mapping", MAPPING_PATH)


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        text=True,
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def cell_volume(edges: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    return (
        np.diff(edges[0])[:, None, None]
        * np.diff(edges[1])[None, :, None]
        * np.diff(edges[2])[None, None, :]
    )


def yee_like_edges(
    *,
    half_span_m: float,
    requested_step_m: float,
) -> np.ndarray:
    cells = int(np.ceil(2.0 * half_span_m / requested_step_m))
    return np.linspace(-half_span_m, half_span_m, cells + 1)


def analytic_q_on_edges(
    edges: tuple[np.ndarray, np.ndarray, np.ndarray],
    polarization: str,
) -> np.ndarray:
    constants = base.optical_constants(polarization)
    beta = constants["beta_m_inv"]
    root_two_over_waist = np.sqrt(2.0) / base.WAIST_M
    one_dimensional_integral = [
        0.5
        * (
            erf(root_two_over_waist * axis[1:])
            - erf(root_two_over_waist * axis[:-1])
        )
        for axis in edges[:2]
    ]
    lateral_cell_average = (
        one_dimensional_integral[0][:, None]
        * one_dimensional_integral[1][None, :]
        / (
            np.diff(edges[0])[:, None]
            * np.diff(edges[1])[None, :]
        )
    )
    depth_cell_average = (
        np.exp(beta * edges[2][1:])
        - np.exp(beta * edges[2][:-1])
    ) / (
        constants["absorbed_depth_fraction_over_130nm"]
        * np.diff(edges[2])
    )
    # Both lateral axes use identical edges. The Gaussian is symmetric under
    # x<->y, so the exact integral over y<=x is one half of a diagonal cell.
    x = 0.5 * (edges[0][:-1] + edges[0][1:])
    y = 0.5 * (edges[1][:-1] + edges[1][1:])
    inside_fraction = np.where(
        y[None, :] < x[:, None],
        1.0,
        np.where(
            np.isclose(y[None, :], x[:, None], rtol=0.0, atol=1.0e-18),
            0.5,
            0.0,
        ),
    )
    q = (
        base.INCIDENT_POWER_W
        * base.TMM_ABSORPTION[polarization]
        * lateral_cell_average[:, :, None]
        * depth_cell_average[None, None, :]
        * inside_fraction[:, :, None]
    )
    return q


def relative_l2(
    first: np.ndarray,
    second: np.ndarray,
    weight: np.ndarray,
) -> float:
    return float(
        np.sqrt(
            np.sum(weight * (second - first) ** 2)
            / np.sum(weight * first**2)
        )
    )


def field_nrmse(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
) -> float:
    selected_first = np.asarray(first, float)[mask]
    selected_second = np.asarray(second, float)[mask]
    return float(
        np.linalg.norm(selected_second - selected_first)
        / np.linalg.norm(selected_first)
    )


def solve_case(
    *,
    polarization: str,
    expanded_geometry: Any,
    reduced_geometry: Any,
    system: Any,
    direct_q: np.ndarray,
    source_half_span_m: float,
    source_step_m: float,
) -> dict[str, Any]:
    source_edges = (
        yee_like_edges(
            half_span_m=source_half_span_m,
            requested_step_m=source_step_m,
        ),
        yee_like_edges(
            half_span_m=source_half_span_m,
            requested_step_m=source_step_m,
        ),
        np.linspace(-base.THICKNESS_M, 0.0, 14),
    )
    source_q = analytic_q_on_edges(source_edges, polarization)
    target_edges = (
        reduced_geometry.x_edges_m,
        reduced_geometry.y_edges_m,
        reduced_geometry.z_edges_m,
    )
    raw_remap = mapping.build_conservative_embedding_remap(
        source_edges_m=source_edges,
        target_edges_m=target_edges,
    )
    remap = mapping.project_remap_to_nearest_material_support(
        raw_remap,
        target_edges_m=target_edges,
        target_support_mask=reduced_geometry.flake_mask,
    )
    remapped_q = remap.apply(source_q)
    source_power = remap.power_source(source_q)
    remapped_power = remap.power_target(remapped_q)
    target_volume = cell_volume(target_edges)
    direct_power = float(np.sum(target_volume * direct_q))

    solved_direct = base.thermal.solve_assembled_thermal_system(
        system,
        source_W_m3=direct_q,
        relative_tolerance=1e-10,
        max_iterations=12000,
    )
    solved_remapped = base.thermal.solve_assembled_thermal_system(
        system,
        source_W_m3=remapped_q,
        relative_tolerance=1e-10,
        max_iterations=12000,
    )
    metrics_direct, fields_direct = (
        base.thermal.straight_edge_temperature_metrics(
            solved_direct.temperature_K,
            reduced_geometry,
        )
    )
    metrics_remapped, fields_remapped = (
        base.thermal.straight_edge_temperature_metrics(
            solved_remapped.temperature_K,
            reduced_geometry,
        )
    )
    mask_2d = np.any(reduced_geometry.flake_mask, axis=2)
    q_nrmse = relative_l2(
        direct_q,
        remapped_q,
        target_volume,
    )
    temperature_nrmse = field_nrmse(
        fields_direct["temperature_flake_average_K"],
        fields_remapped["temperature_flake_average_K"],
        mask_2d,
    )
    gradient_nrmse = {
        name: field_nrmse(
            fields_direct[name],
            fields_remapped[name],
            mask_2d,
        )
        for name in (
            "grad_T_x_K_m",
            "grad_T_y_K_m",
            "grad_T_normal_K_m",
            "grad_T_tangent_K_m",
            "grad_T_magnitude_K_m",
        )
    }
    metric_change = {
        name: abs(metrics_remapped[name] - metrics_direct[name])
        / max(abs(metrics_direct[name]), np.finfo(float).tiny)
        for name in (
            "Tmax_rise_K",
            "TaIrTe4_area_average_rise_K",
            "fixed_24um_ROI_area_average_rise_K",
            "max_abs_grad_T_x_K_m",
            "max_abs_grad_T_y_K_m",
            "max_abs_edge_normal_gradient_K_m",
            "max_abs_edge_tangent_gradient_K_m",
            "max_inplane_gradient_K_m",
        )
    }
    return {
        "polarization": polarization,
        "source_grid": {
            "shape_xyz": list(source_q.shape),
            "bounds_m": {
                axis: [
                    float(source_edges[index][0]),
                    float(source_edges[index][-1]),
                ]
                for index, axis in enumerate("xyz")
            },
            "step_m": {
                axis: {
                    "minimum": float(np.min(np.diff(source_edges[index]))),
                    "maximum": float(np.max(np.diff(source_edges[index]))),
                }
                for index, axis in enumerate("xyz")
            },
            "role": (
                "nonperiodic Yee-like Cartesian layout surrogate; no Maxwell "
                "solve and no claim of exact v261 native mesh"
            ),
        },
        "power": {
            "direct_thermal_grid_W": direct_power,
            "Yee_like_source_W": source_power,
            "remapped_thermal_grid_W": remapped_power,
            "remap_conservation_relative_error": (
                abs(remapped_power - source_power) / abs(source_power)
            ),
            "direct_vs_remapped_relative_difference": (
                abs(remapped_power - direct_power) / abs(direct_power)
            ),
        },
        "Q_thermal_grid_NRMSE": q_nrmse,
        "temperature_field_NRMSE": temperature_nrmse,
        "gradient_field_NRMSE": gradient_nrmse,
        "metric_relative_change": metric_change,
        "direct_metrics": metrics_direct,
        "remapped_metrics": metrics_remapped,
        "direct_solver": {
            "linear_residual_relative": (
                solved_direct.linear_residual_relative
            ),
            "energy_balance_relative_error": (
                solved_direct.energy_balance_relative_error
            ),
        },
        "remapped_solver": {
            "linear_residual_relative": (
                solved_remapped.linear_residual_relative
            ),
            "energy_balance_relative_error": (
                solved_remapped.energy_balance_relative_error
            ),
        },
        "_arrays": {
            "direct_Q_W_m3": direct_q,
            "remapped_Q_W_m3": remapped_q,
            "direct_T_K": fields_direct["temperature_flake_average_K"],
            "remapped_T_K": fields_remapped[
                "temperature_flake_average_K"
            ],
            "direct_grad_magnitude_K_m": fields_direct[
                "grad_T_magnitude_K_m"
            ],
            "remapped_grad_magnitude_K_m": fields_remapped[
                "grad_T_magnitude_K_m"
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--core-step-nm", type=float, default=100.0)
    parser.add_argument("--flake-dz-nm", type=float, default=26.0)
    parser.add_argument("--thermal-domain-um", type=float, default=48.0)
    parser.add_argument("--source-half-span-um", type=float, default=12.0)
    parser.add_argument(
        "--yee-like-step-nm",
        type=float,
        default=67.79661016949153,
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    geometry_args = SimpleNamespace(
        thermal_domain_um=args.thermal_domain_um,
        si_depth_um=20.0,
        core_step_nm=args.core_step_nm,
        flake_dz_nm=args.flake_dz_nm,
    )
    expanded_geometry = base.build_straight_geometry(geometry_args)
    zero_q = np.zeros(expanded_geometry.material_id.shape, float)
    reduced_geometry, _ = base.select_thermal_model(
        expanded_geometry,
        zero_q,
        "paper-reduced",
    )
    system = base.assemble_system(reduced_geometry, "paper-reduced")
    cases = []
    arrays: dict[str, np.ndarray] = {}
    for polarization in ("a", "b"):
        direct_q = analytic_q_on_edges(
            (
                reduced_geometry.x_edges_m,
                reduced_geometry.y_edges_m,
                reduced_geometry.z_edges_m,
            ),
            polarization,
        )
        case = solve_case(
            polarization=polarization,
            expanded_geometry=expanded_geometry,
            reduced_geometry=reduced_geometry,
            system=system,
            direct_q=direct_q,
            source_half_span_m=args.source_half_span_um * 1e-6,
            source_step_m=args.yee_like_step_nm * 1e-9,
        )
        for name, values in case.pop("_arrays").items():
            arrays[f"{polarization}_{name}"] = values
        cases.append(case)

    worst_temperature = max(
        case["temperature_field_NRMSE"] for case in cases
    )
    worst_gradient = max(
        max(case["gradient_field_NRMSE"].values()) for case in cases
    )
    primary_metric_names = (
        "Tmax_rise_K",
        "TaIrTe4_area_average_rise_K",
        "fixed_24um_ROI_area_average_rise_K",
        "max_abs_grad_T_x_K_m",
        "max_abs_edge_tangent_gradient_K_m",
        "max_inplane_gradient_K_m",
    )
    diagnostic_peak_names = (
        "max_abs_grad_T_y_K_m",
        "max_abs_edge_normal_gradient_K_m",
    )
    worst_metric = max(
        case["metric_relative_change"][name]
        for case in cases
        for name in primary_metric_names
    )
    worst_diagnostic_peak = max(
        case["metric_relative_change"][name]
        for case in cases
        for name in diagnostic_peak_names
    )
    worst_power = max(
        case["power"]["direct_vs_remapped_relative_difference"]
        for case in cases
    )
    worst_conservation = max(
        case["power"]["remap_conservation_relative_error"]
        for case in cases
    )
    passed = (
        worst_power < 0.01
        and worst_temperature < 0.01
        and worst_gradient < 0.01
        and worst_metric < 0.01
        and worst_conservation < 1e-12
    )
    summary = {
        "status": (
            "VALIDATED_ANALYTIC_Q_YEE_LIKE_REMAP_CONTROL"
            if passed
            else "FAILED_ANALYTIC_Q_YEE_LIKE_REMAP_CONTROL"
        ),
        "validated": passed,
        "FDTD_run": False,
        "thermal_model": (
            "paper Supplement Eq. S4 reduced flake-only Robin"
        ),
        "geometry": {
            "straight_edge": "TaIrTe4 y<=x",
            "thermal_domain_um": args.thermal_domain_um,
            "core_step_nm": args.core_step_nm,
            "flake_dz_nm": args.flake_dz_nm,
            "TaIrTe4_thickness_nm": 130.0,
            "wavelength_um": 11.0,
            "Gaussian_waist_um": base.WAIST_M * 1e6,
        },
        "comparison": {
            "direct": "analytic Q sampled directly on thermal cells",
            "remapped": (
                "same analytic law sampled on a finer nonperiodic Yee-like "
                "Cartesian grid, then passed through the current conservative "
                "embedding and physical-nearest support projection"
            ),
            "no_equal_power_rescaling": True,
        },
        "cases": cases,
        "worst": {
            "direct_vs_remapped_power_relative_difference": worst_power,
            "remap_conservation_relative_error": worst_conservation,
            "temperature_field_NRMSE": worst_temperature,
            "gradient_field_NRMSE": worst_gradient,
            "primary_metric_relative_change": worst_metric,
            "diagnostic_raw_peak_relative_change": worst_diagnostic_peak,
        },
        "metric_roles": {
            "primary": list(primary_metric_names),
            "diagnostic_raw_cell_maxima": list(diagnostic_peak_names),
            "note": (
                "raw cell maxima remain visible but are not the acceptance "
                "gate; field NRMSE and the paper Fig.3G x-gradient comparator "
                "test the remap without one-cell peak-location instability"
            ),
        },
        "acceptance": {
            "power_lt_1_percent": worst_power < 0.01,
            "temperature_lt_1_percent": worst_temperature < 0.01,
            "gradient_field_lt_1_percent": worst_gradient < 0.01,
            "primary_metrics_lt_1_percent": worst_metric < 0.01,
            "diagnostic_raw_peaks_lt_1_percent": (
                worst_diagnostic_peak < 0.01
            ),
            "remap_conservation_lt_1e_minus_12": (
                worst_conservation < 1e-12
            ),
            "all": passed,
        },
        "generation_commit": git_commit(),
    }
    npz_path = args.output_dir / "analytic_q_remap_control_fields.npz"
    np.savez_compressed(
        npz_path,
        x_edges_m=reduced_geometry.x_edges_m,
        y_edges_m=reduced_geometry.y_edges_m,
        z_edges_m=reduced_geometry.z_edges_m,
        **arrays,
    )
    summary["raw_fields"] = {
        "path": str(npz_path.resolve()),
        "size_bytes": npz_path.stat().st_size,
        "sha256": sha256(npz_path),
    }
    (args.output_dir / "analytic_q_remap_control_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    x = 0.5 * (
        reduced_geometry.x_edges_m[:-1]
        + reduced_geometry.x_edges_m[1:]
    )
    y = 0.5 * (
        reduced_geometry.y_edges_m[:-1]
        + reduced_geometry.y_edges_m[1:]
    )
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(13.5, 8.0),
        constrained_layout=True,
    )
    for row, polarization in enumerate(("a", "b")):
        direct_t = arrays[f"{polarization}_direct_T_K"]
        remapped_t = arrays[f"{polarization}_remapped_T_K"]
        direct_g = arrays[
            f"{polarization}_direct_grad_magnitude_K_m"
        ]
        remapped_g = arrays[
            f"{polarization}_remapped_grad_magnitude_K_m"
        ]
        for axis, values, title, unit in (
            (axes[row, 0], direct_t, f"E || {polarization}: direct T", "K"),
            (
                axes[row, 1],
                remapped_t - direct_t,
                f"E || {polarization}: remapped - direct T",
                "K",
            ),
            (
                axes[row, 2],
                remapped_g - direct_g,
                f"E || {polarization}: gradient difference",
                "K/m",
            ),
        ):
            image = axis.pcolormesh(
                x * 1e6,
                y * 1e6,
                values.T,
                shading="nearest",
            )
            axis.set(
                xlabel="x (µm)",
                ylabel="y (µm)",
                title=title,
                aspect="equal",
            )
            figure.colorbar(image, ax=axis, label=unit)
    figure.savefig(
        args.output_dir / "analytic_q_remap_control.png",
        dpi=170,
    )
    plt.close(figure)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
