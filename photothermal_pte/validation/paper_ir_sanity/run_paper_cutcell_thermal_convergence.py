#!/usr/bin/env python3
"""Paper-reduced straight-edge thermal convergence with exact cut-cell Q."""

from __future__ import annotations

import argparse
import csv
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


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
REMAP_CONTROL_PATH = HERE / "run_analytic_q_remap_control.py"
ROBUST_PATH = HERE / "audit_straight_edge_robust_gradient.py"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


control = load_module("paper_cutcell_control", REMAP_CONTROL_PATH)
base = control.base
robust = load_module("paper_cutcell_robust", ROBUST_PATH)


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


def relative_change(first: float, second: float) -> float:
    return abs(second - first) / max(abs(first), np.finfo(float).tiny)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thermal-domain-um", type=float, default=48.0)
    parser.add_argument("--flake-dz-nm", type=float, default=26.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = args.output_dir / "cases"
    raw_dir.mkdir()

    cases: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    refined_fields: dict[str, np.ndarray] = {}
    refined_edges: tuple[np.ndarray, np.ndarray] | None = None
    for mesh_nm in (200, 100, 50):
        geometry_args = SimpleNamespace(
            thermal_domain_um=args.thermal_domain_um,
            si_depth_um=20.0,
            core_step_nm=float(mesh_nm),
            flake_dz_nm=args.flake_dz_nm,
        )
        expanded = base.build_straight_geometry(geometry_args)
        zero_q = np.zeros(expanded.material_id.shape, float)
        geometry, _ = base.select_thermal_model(
            expanded,
            zero_q,
            "paper-reduced",
        )
        system = base.assemble_system(geometry, "paper-reduced")
        x = 0.5 * (geometry.x_edges_m[:-1] + geometry.x_edges_m[1:])
        y = 0.5 * (geometry.y_edges_m[:-1] + geometry.y_edges_m[1:])
        flake_xy = np.any(geometry.flake_mask, axis=2)
        for polarization in ("a", "b"):
            q = control.analytic_q_on_edges(
                (
                    geometry.x_edges_m,
                    geometry.y_edges_m,
                    geometry.z_edges_m,
                ),
                polarization,
            )
            solved = base.thermal.solve_assembled_thermal_system(
                system,
                source_W_m3=q,
                relative_tolerance=1e-10,
                max_iterations=12000,
            )
            metrics, fields = (
                base.thermal.straight_edge_temperature_metrics(
                    solved.temperature_K,
                    geometry,
                )
            )
            fitted = robust.quadratic_edge_fit(
                x,
                y,
                fields["temperature_flake_average_K"],
                robust.N_BANDS_UM["primary"],
            )
            robust_metrics = {
                component: robust.aggregate(
                    fitted["t_m"],
                    fitted[f"dT_d{component}_K_m"],
                )
                for component in ("x", "n")
            }
            key = f"{polarization}_{mesh_nm}nm"
            case_path = raw_dir / f"{key}.npz"
            np.savez_compressed(
                case_path,
                x_edges_m=geometry.x_edges_m,
                y_edges_m=geometry.y_edges_m,
                z_edges_m=geometry.z_edges_m,
                Q_W_m3=q,
                temperature_flake_average_K=fields[
                    "temperature_flake_average_K"
                ],
                grad_T_x_K_m=fields["grad_T_x_K_m"],
                grad_T_y_K_m=fields["grad_T_y_K_m"],
                grad_T_normal_K_m=fields["grad_T_normal_K_m"],
                grad_T_tangent_K_m=fields["grad_T_tangent_K_m"],
                grad_T_magnitude_K_m=fields["grad_T_magnitude_K_m"],
                robust_t_m=fitted["t_m"],
                robust_dT_dx_K_m=fitted["dT_dx_K_m"],
                robust_dT_dn_K_m=fitted["dT_dn_K_m"],
                robust_fit_relative_residual=fitted[
                    "fit_relative_residual"
                ],
            )
            case = {
                "polarization": polarization,
                "mesh_nm": mesh_nm,
                "grid_shape_xyz": list(system.shape),
                "source_power_W": solved.source_power_W,
                "linear_residual_relative": (
                    solved.linear_residual_relative
                ),
                "energy_balance_relative_error": (
                    solved.energy_balance_relative_error
                ),
                "metrics": metrics,
                "robust_exact_edge_fit": robust_metrics,
                "robust_fit_residual_p99": float(
                    np.percentile(
                        fitted["fit_relative_residual"],
                        99.0,
                    )
                ),
                "artifact": {
                    "path": str(case_path.resolve()),
                    "size_bytes": case_path.stat().st_size,
                    "sha256": sha256(case_path),
                },
            }
            cases[key] = case
            rows.append(
                {
                    "polarization": polarization,
                    "mesh_nm": mesh_nm,
                    "source_power_W": solved.source_power_W,
                    "Tmax_K": metrics["Tmax_rise_K"],
                    "max_abs_grad_T_x_K_m": metrics[
                        "max_abs_grad_T_x_K_m"
                    ],
                    "max_grad_magnitude_K_m": metrics[
                        "max_inplane_gradient_K_m"
                    ],
                    "robust_x_strip_mean_abs_K_m": robust_metrics["x"][
                        "edge_strip_mean_abs_K_m"
                    ],
                    "robust_n_strip_mean_abs_K_m": robust_metrics["n"][
                        "edge_strip_mean_abs_K_m"
                    ],
                    "linear_residual_relative": (
                        solved.linear_residual_relative
                    ),
                    "energy_balance_relative_error": (
                        solved.energy_balance_relative_error
                    ),
                }
            )
            if mesh_nm == 50:
                refined_fields[f"{polarization}_T_K"] = fields[
                    "temperature_flake_average_K"
                ]
                refined_fields[f"{polarization}_grad_magnitude_K_m"] = fields[
                    "grad_T_magnitude_K_m"
                ]
                refined_fields[f"{polarization}_flake_mask"] = flake_xy
                refined_edges = (
                    geometry.x_edges_m,
                    geometry.y_edges_m,
                )

    convergence: dict[str, Any] = {}
    for polarization in ("a", "b"):
        convergence[polarization] = {}
        for coarse, refined in ((200, 100), (100, 50), (200, 50)):
            first = cases[f"{polarization}_{coarse}nm"]
            second = cases[f"{polarization}_{refined}nm"]
            convergence[polarization][f"{coarse}_to_{refined}"] = {
                "source_power": relative_change(
                    first["source_power_W"],
                    second["source_power_W"],
                ),
                "Tmax": relative_change(
                    first["metrics"]["Tmax_rise_K"],
                    second["metrics"]["Tmax_rise_K"],
                ),
                "raw_max_abs_grad_T_x": relative_change(
                    first["metrics"]["max_abs_grad_T_x_K_m"],
                    second["metrics"]["max_abs_grad_T_x_K_m"],
                ),
                "robust_x_strip_mean": relative_change(
                    first["robust_exact_edge_fit"]["x"][
                        "edge_strip_mean_abs_K_m"
                    ],
                    second["robust_exact_edge_fit"]["x"][
                        "edge_strip_mean_abs_K_m"
                    ],
                ),
                "robust_n_strip_mean": relative_change(
                    first["robust_exact_edge_fit"]["n"][
                        "edge_strip_mean_abs_K_m"
                    ],
                    second["robust_exact_edge_fit"]["n"][
                        "edge_strip_mean_abs_K_m"
                    ],
                ),
            }
    ratios: dict[str, Any] = {}
    for mesh_nm in (200, 100, 50):
        a = cases[f"a_{mesh_nm}nm"]
        b = cases[f"b_{mesh_nm}nm"]
        ratios[f"{mesh_nm}nm"] = {
            "source_power_b_over_a": (
                b["source_power_W"] / a["source_power_W"]
            ),
            "raw_max_abs_grad_T_x_b_over_a": (
                b["metrics"]["max_abs_grad_T_x_K_m"]
                / a["metrics"]["max_abs_grad_T_x_K_m"]
            ),
            "robust_x_strip_mean_b_over_a": (
                b["robust_exact_edge_fit"]["x"][
                    "edge_strip_mean_abs_K_m"
                ]
                / a["robust_exact_edge_fit"]["x"][
                    "edge_strip_mean_abs_K_m"
                ]
            ),
            "robust_n_strip_mean_b_over_a": (
                b["robust_exact_edge_fit"]["n"][
                    "edge_strip_mean_abs_K_m"
                ]
                / a["robust_exact_edge_fit"]["n"][
                    "edge_strip_mean_abs_K_m"
                ]
            ),
        }
    numerical_gates = all(
        case["linear_residual_relative"] < 1e-8
        and case["energy_balance_relative_error"] < 0.01
        for case in cases.values()
    )
    b_greater_a = all(
        ratios[f"{mesh_nm}nm"]["raw_max_abs_grad_T_x_b_over_a"] > 1.0
        and ratios[f"{mesh_nm}nm"]["robust_x_strip_mean_b_over_a"] > 1.0
        for mesh_nm in (200, 100, 50)
    )
    robust_x_converged = all(
        convergence[polarization]["100_to_50"][
            "robust_x_strip_mean"
        ]
        < 0.01
        for polarization in ("a", "b")
    )
    source_power_converged = all(
        convergence[polarization]["100_to_50"]["source_power"] < 0.01
        for polarization in ("a", "b")
    )
    passed = (
        numerical_gates
        and b_greater_a
        and robust_x_converged
        and source_power_converged
    )
    summary = {
        "status": (
            "VALIDATED_PAPER_REDUCED_CUTCELL_THERMAL_TREND"
            if passed
            else "FAILED_PAPER_REDUCED_CUTCELL_THERMAL_TREND"
        ),
        "validated": passed,
        "FDTD_run": False,
        "scope": (
            "paper-like analytic Gaussian-Beer-Lambert heat source and "
            "Supplement Eq. S4 reduced thermal model; not full experimental "
            "or optical reproduction"
        ),
        "source": {
            "TaIrTe4_thickness_nm": 130.0,
            "wavelength_um": 11.0,
            "Gaussian_waist_um": base.WAIST_M * 1e6,
            "incident_power_W": base.INCIDENT_POWER_W,
            "TMM_absorption": base.TMM_ABSORPTION,
            "cut_cell_contract": (
                "analytic Gaussian and Beer-Lambert cell averages; exact "
                "one-half integral in diagonal cells intersected by y=x"
            ),
        },
        "thermal": {
            "model": "paper Supplement Eq. S4 reduced flake-only Robin",
            "G_bottom_W_m2K": base.thermal.G_TAIRTE4_SIO2_W_M2K,
            "G_top_W_m2K": base.thermal.G_TAIRTE4_AIR_W_M2K,
            "lateral_boundary": "insulating material edge; no lateral bath",
            "thermal_domain_um": args.thermal_domain_um,
            "flake_dz_nm": args.flake_dz_nm,
        },
        "cases": cases,
        "convergence": convergence,
        "ratios_b_over_a": ratios,
        "map_interpretation": {
            "smooth_temperature_map_saved": True,
            "edge_clipped_ring_like_gradient_map_saved": True,
            "qualitative_only": True,
        },
        "acceptance": {
            "residual_and_energy_gates": numerical_gates,
            "max_and_robust_grad_x_b_greater_a_all_meshes": b_greater_a,
            "robust_x_100_to_50_lt_1_percent": robust_x_converged,
            "source_power_100_to_50_lt_1_percent": (
                source_power_converged
            ),
            "all": passed,
        },
        "generation_commit": git_commit(),
    }
    (args.output_dir / "paper_cutcell_thermal_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (
        args.output_dir / "paper_cutcell_thermal_cases.csv"
    ).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if refined_edges is None:
        raise RuntimeError("50 nm fields were not retained")
    x = 0.5 * (refined_edges[0][:-1] + refined_edges[0][1:])
    y = 0.5 * (refined_edges[1][:-1] + refined_edges[1][1:])
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.5, 9.0),
        constrained_layout=True,
    )
    for row, polarization in enumerate(("a", "b")):
        mask = refined_fields[f"{polarization}_flake_mask"]
        for column, (key, title, unit) in enumerate(
            (
                ("T_K", "temperature rise", "K"),
                (
                    "grad_magnitude_K_m",
                    "in-plane |gradient T|",
                    "K/m",
                ),
            )
        ):
            values = np.where(
                mask,
                refined_fields[f"{polarization}_{key}"],
                np.nan,
            )
            image = axes[row, column].pcolormesh(
                x * 1e6,
                y * 1e6,
                values.T,
                shading="nearest",
            )
            axes[row, column].plot(
                [-12, 12],
                [-12, 12],
                "w--",
                linewidth=0.8,
            )
            axes[row, column].set(
                xlim=(-10, 10),
                ylim=(-10, 10),
                xlabel="x (µm)",
                ylabel="y (µm)",
                title=f"E || {polarization}: {title}, 50 nm",
                aspect="equal",
            )
            figure.colorbar(
                image,
                ax=axes[row, column],
                label=unit,
            )
    figure.savefig(
        args.output_dir / "paper_cutcell_thermal_maps_50nm.png",
        dpi=180,
    )
    plt.close(figure)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.2),
        constrained_layout=True,
    )
    for axis, metric, title in (
        (
            axes[0],
            "raw_max_abs_grad_T_x_K_m",
            "Raw max |dT/dx|",
        ),
        (
            axes[1],
            "robust_x_strip_mean_abs_K_m",
            "Exact-edge fitted |dT/dx| strip mean",
        ),
    ):
        for polarization in ("a", "b"):
            values = []
            for mesh_nm in (200, 100, 50):
                case = cases[f"{polarization}_{mesh_nm}nm"]
                if metric.startswith("raw"):
                    values.append(
                        case["metrics"]["max_abs_grad_T_x_K_m"]
                    )
                else:
                    values.append(
                        case["robust_exact_edge_fit"]["x"][
                            "edge_strip_mean_abs_K_m"
                        ]
                    )
            axis.plot(
                (200, 100, 50),
                values,
                "o-",
                label=f"E || {polarization}",
            )
        axis.set(
            xlabel="thermal lateral cell size (nm)",
            ylabel="K/m",
            title=title,
        )
        axis.invert_xaxis()
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(
        args.output_dir / "paper_cutcell_gradient_convergence.png",
        dpi=180,
    )
    plt.close(figure)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
