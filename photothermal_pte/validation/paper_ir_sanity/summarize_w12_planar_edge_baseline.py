#!/usr/bin/env python3
"""Summarize the saved w0=12 um planar/straight-edge GPU baseline.

This is an offline consumer of completed FDTD and read-only field-audit
artifacts.  It performs no FDTD, thermal, PTE, adjoint, or optimization solve.
Raw absorbed powers are retained; equal-power normalization is used only for
spatial-shape metrics and figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.validation.paper_ir_sanity.compare_paper_ir_smoke_q_convergence import (  # noqa: E402
    edge_profile,
    trapezoid_weights,
)


CASE_ORDER = ("planar_a", "planar_b", "edge_a", "edge_b")
LABELS = {
    "planar_a": r"planar, $E\parallel a$",
    "planar_b": r"planar, $E\parallel b$",
    "edge_a": r"45° edge, $E\parallel a$",
    "edge_b": r"45° edge, $E\parallel b$",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()


def load_q(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as raw:
        coordinates = {
            axis: np.asarray(raw[f"{axis}_m"], float) for axis in "xyz"
        }
        total = np.asarray(raw["Q_on_W_m3"], float)
    shape = tuple(coordinates[axis].size for axis in "xyz")
    if total.shape != shape or np.any(~np.isfinite(total)):
        raise RuntimeError(f"invalid Q artifact {path}: {total.shape} != {shape}")
    return {"coordinates": coordinates, "total": total}


def volume_weights(coordinates: dict[str, np.ndarray]) -> np.ndarray:
    return (
        trapezoid_weights(coordinates["x"])[:, None, None]
        * trapezoid_weights(coordinates["y"])[None, :, None]
        * trapezoid_weights(coordinates["z"])[None, None, :]
    )


def normalized_metrics(
    first: np.ndarray,
    second: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    first_power = float(np.sum(weights * first))
    second_power = float(np.sum(weights * second))
    a = first / first_power
    b = second / second_power
    nrmse = float(
        np.sqrt(np.sum(weights * (a - b) ** 2) / np.sum(weights * a**2))
    )
    weight = weights / np.sum(weights)
    da = a - float(np.sum(weight * a))
    db = b - float(np.sum(weight * b))
    correlation = float(
        np.sum(weight * da * db)
        / np.sqrt(np.sum(weight * da**2) * np.sum(weight * db**2))
    )
    cosine = float(
        np.sum(weights * a * b)
        / np.sqrt(np.sum(weights * a**2) * np.sum(weights * b**2))
    )
    return {
        "first_raw_power_W": first_power,
        "second_raw_power_W": second_power,
        "raw_power_signed_difference_W": second_power - first_power,
        "raw_power_ratio_second_over_first": second_power / first_power,
        "equal_power_spatial_Q_NRMSE": nrmse,
        "equal_power_spatial_Q_Pearson_correlation": correlation,
        "equal_power_spatial_Q_cosine_similarity": cosine,
    }


def spatial_moments(
    q: np.ndarray,
    coordinates: dict[str, np.ndarray],
    weights: np.ndarray,
) -> dict[str, Any]:
    power = float(np.sum(weights * q))
    normalized_weight = weights * q / power
    result: dict[str, Any] = {}
    for index, axis in enumerate("xyz"):
        shape = [1, 1, 1]
        shape[index] = coordinates[axis].size
        coordinate = coordinates[axis].reshape(shape)
        center = float(np.sum(normalized_weight * coordinate))
        variance = float(
            np.sum(normalized_weight * (coordinate - center) ** 2)
        )
        result[axis] = {
            "centroid_m": center,
            "second_moment_sigma_m": float(np.sqrt(max(variance, 0.0))),
        }
    hotspot = np.unravel_index(int(np.argmax(q)), q.shape)
    result["hotspot"] = {
        "x_m": float(coordinates["x"][hotspot[0]]),
        "y_m": float(coordinates["y"][hotspot[1]]),
        "z_m": float(coordinates["z"][hotspot[2]]),
        "Q_W_m3": float(q[hotspot]),
    }
    return result


def wall_time(log_path: Path) -> float | None:
    match = re.search(
        r"Overall wall time measurements in seconds:\s*([0-9.eE+-]+)",
        log_path.read_text(encoding="utf-8", errors="replace"),
    )
    return float(match.group(1)) if match else None


def auto_shutoff(
    run: dict[str, Any],
    log_path: Path,
) -> dict[str, Any]:
    if run.get("auto_shutoff") is not None:
        return run["auto_shutoff"]
    text = log_path.read_text(encoding="utf-8", errors="replace")
    values = re.findall(r"Auto Shutoff:\s*([0-9.eE+-]+)", text)
    if not values:
        raise RuntimeError(f"no auto-shutoff value in {log_path}")
    return {
        "log_path": str(log_path.resolve()),
        "final_value": float(values[-1]),
        "simulation_completed_successfully": (
            "Simulation completed successfully" in text
        ),
        "promoted_metadata_source": (
            "independent immutable solver-log parse; original case JSON "
            "predates the explicit auto-shutoff result key"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in CASE_ORDER:
        parser.add_argument(f"--{name.replace('_', '-')}-dir", type=Path, required=True)
        parser.add_argument(
            f"--{name.replace('_', '-')}-field-dir", type=Path, required=True
        )
    parser.add_argument("--failed-edge-a-dir", type=Path)
    parser.add_argument("--failed-edge-b-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = {
        name: getattr(args, f"{name}_dir").resolve() for name in CASE_ORDER
    }
    field_dirs = {
        name: getattr(args, f"{name}_field_dir").resolve()
        for name in CASE_ORDER
    }
    cases: dict[str, Any] = {}
    q_data: dict[str, Any] = {}
    manifest: list[dict[str, Any]] = []

    for name in CASE_ORDER:
        case_dir = case_dirs[name]
        case_path = case_dir / "case_result.json"
        q_path = case_dir / "finite_q_on_artifact.npz"
        log_path = case_dir / "finite_2um_optical_q_p0.log"
        raw_manifest_path = case_dir / "RAW_ARTIFACT_MANIFEST.json"
        field_path = field_dirs[name] / "saved_field_monitor_audit.json"
        case = read_json(case_path)
        field = read_json(field_path)
        q_data[name] = load_q(q_path)
        run = case["run_result"]
        reintegrated = float(
            np.sum(
                volume_weights(q_data[name]["coordinates"])
                * q_data[name]["total"]
            )
        )
        cases[name] = {
            "geometry": case["pre_run_contract"]["geometry"],
            "polarization_deg": case["polarization_deg"],
            "P_Q_W": run["P_Q_W"],
            "P_Q_reintegrated_W": reintegrated,
            "P_Q_reintegration_relative_error": abs(
                reintegrated - run["P_Q_W"]
            )
            / abs(run["P_Q_W"]),
            "P_six_face_W": run["P_six_face_W"],
            "six_face_relative_closure": run["six_face_relative_closure"],
            "component_power_W": run["component_power_W"],
            "Q_hotspot": run["Q_hotspot"],
            "negative_Q_voxel_count": run["negative_Q_voxel_count"],
            "auto_shutoff": auto_shutoff(run, log_path),
            "acceptance": run["acceptance"],
            "incident_power_W_at_1_W_m2_center": run["normalization"][
                "incident_power_W_at_1_W_m2"
            ],
            "source_intensity_scale": run["normalization"][
                "scale_to_1_W_m2"
            ],
            "empirical_flux_gain": run["normalization"][
                "empirical_flux_gain"
            ],
            "field_readback": field,
            "spatial_moments": spatial_moments(
                q_data[name]["total"],
                q_data[name]["coordinates"],
                volume_weights(q_data[name]["coordinates"]),
            ),
            "solver_wall_time_s": wall_time(log_path),
            "generation_commit": case["generation_commit"],
        }
        for path, role in (
            (case_path, f"{name} case result"),
            (raw_manifest_path, f"{name} raw artifact manifest"),
            (field_path, f"{name} read-only field audit"),
        ):
            manifest.append(record(path, role))
        manifest.append(
            {
                "role": f"{name} raw Q NPZ",
                **read_json(raw_manifest_path)["raw_artifacts"][
                    "finite_q_on_artifact.npz"
                ],
                "hash_source": (
                    "case-local immutable raw manifest generated with the Q"
                ),
            }
        )
        manifest.append(
            {
                "role": f"{name} raw FSP",
                **read_json(raw_manifest_path)["raw_artifacts"][
                    "finite_2um_optical_q.fsp"
                ],
                "hash_source": (
                    "case-local immutable raw manifest generated with the FSP"
                ),
            }
        )
        field_npz = Path(field["field_artifact"]["path"])
        manifest.append(
            {
                "role": f"{name} read-only field NPZ",
                **field["field_artifact"],
                "hash_source": "read-only field audit",
            }
        )

    for name in CASE_ORDER[1:]:
        for axis in "xyz":
            if not np.array_equal(
                q_data["planar_a"]["coordinates"][axis],
                q_data[name]["coordinates"][axis],
            ):
                raise RuntimeError(
                    f"{name} {axis} coordinates differ; comparison is fail-closed"
                )
    coordinates = q_data["planar_a"]["coordinates"]
    weights = volume_weights(coordinates)
    comparisons = {
        "planar_a_vs_edge_a": normalized_metrics(
            q_data["planar_a"]["total"], q_data["edge_a"]["total"], weights
        ),
        "planar_b_vs_edge_b": normalized_metrics(
            q_data["planar_b"]["total"], q_data["edge_b"]["total"], weights
        ),
        "planar_a_vs_planar_b": normalized_metrics(
            q_data["planar_a"]["total"], q_data["planar_b"]["total"], weights
        ),
        "edge_a_vs_edge_b": normalized_metrics(
            q_data["edge_a"]["total"], q_data["edge_b"]["total"], weights
        ),
    }

    closure_pass = all(
        case["six_face_relative_closure"] < 0.005
        for case in cases.values()
    )
    shutoff_pass = all(
        case["auto_shutoff"]["simulation_completed_successfully"]
        and case["auto_shutoff"]["final_value"] <= 1.0e-5
        for case in cases.values()
    )
    finite_pass = all(
        case["negative_Q_voxel_count"] == 0 for case in cases.values()
    )
    reintegration_pass = all(
        case["P_Q_reintegration_relative_error"] < 0.005
        for case in cases.values()
    )
    no_modification = all(
        not read_json(case_dirs[name] / "case_result.json")[key]
        for name in CASE_ORDER
        for key in ("Q_clipped", "flux_gain", "Q_rescaled", "periodic_Q_used")
    )
    baseline_pass = all(
        (
            closure_pass,
            shutoff_pass,
            finite_pass,
            reintegration_pass,
            no_modification,
        )
    )

    failure_dirs = {
        "edge_a_license_start_race": args.failed_edge_a_dir,
        "edge_b_license_start_race": args.failed_edge_b_dir,
    }
    failures: dict[str, Any] = {}
    for name, directory in failure_dirs.items():
        if directory is None:
            continue
        path = directory.resolve() / "case_result.json"
        failure = read_json(path)
        failures[name] = {
            "status": failure["status"],
            "exception": failure["exception"],
            "FDTD_stepping_started": False,
            "CPU_fallback_used": False,
        }
        manifest.append(record(path, f"preserved {name} diagnostic"))

    payload = {
        "status": (
            "BASELINE_PAPER_LIKE_W12_SCALAR_GAUSSIAN_OPTICAL_GATES_PASSED_REFINEMENT_PENDING"
            if baseline_pass
            else "FAILED_PAPER_LIKE_W12_SCALAR_GAUSSIAN_BASELINE_OPTICAL_GATE"
        ),
        "scope": {
            "wavelength_m": 11.0e-6,
            "source": "scalar Gaussian",
            "assumed_waist_m": 12.0e-6,
            "waist_provenance": "EXPLICIT_ASSUMPTION_NOT_PUBLISHED_BY_PAPER",
            "source_span_m": 50.0e-6,
            "FDTD_lateral_span_m": 60.0e-6,
            "local_xy_mesh_m": 100.0e-9,
            "flake_dz_m": 5.0e-9,
            "TaIrTe4_thickness_m": 130.0e-9,
            "substrate": "285 nm SiO2 on Si",
            "axis_mapping": {"x": "b", "y": "a", "z": "c"},
            "epsilon_c": (
                "epsilon_b paper-consistent 3D closure; not a directly "
                "measured c-axis property"
            ),
            "six_boundaries": "PML",
            "periodic": False,
            "GPU_FDTD_only": True,
            "CPU_FDTD_fallback": False,
            "thermal_run": False,
            "PTE_run": False,
            "adjoint_run": False,
            "optimization_run": False,
        },
        "gates": {
            "all_four_six_face_closure_lt_0p5_percent": closure_pass,
            "all_four_auto_shutoff_le_1e_minus_5": shutoff_pass,
            "all_four_Q_reintegration_error_lt_0p5_percent": reintegration_pass,
            "all_four_no_negative_Q_voxels": finite_pass,
            "no_clipping_smoothing_gain_rescaling_tiling_or_periodic_Q": (
                no_modification
            ),
            "baseline_all": baseline_pass,
            "promotion_gate": (
                "NOT_EVALUATED: 50 nm local-xy refinement is required before "
                "material-Q promotion"
            ),
        },
        "interpretation": {
            "scenario_label": (
                "paper-like scalar-Gaussian scenario with an explicitly "
                "assumed waist"
            ),
            "not_claimed": [
                "experimentally reproduced beam",
                "paper-certified beam",
                "paper reproduction",
                "promoted production Q",
            ],
            "raw_vs_equal_power": (
                "Raw Q and absorbed power are never changed. Equal-power "
                "normalization is used only for spatial-shape comparison."
            ),
            "near_stack_field": (
                "The total-field downward decomposition at the realized "
                "z=0.5136 um plane can contain reflection, scattering and "
                "evanescent contributions; it is not a pure incident waist."
            ),
            "edge_field_fit": (
                "A Gaussian fit to total fields in the finite-edge case is a "
                "shape diagnostic only; edge scattering makes it non-Gaussian."
            ),
        },
        "cases": cases,
        "comparisons": comparisons,
        "preserved_license_start_failures": failures,
        "generation_commit": git_commit(),
    }

    summary_path = args.output_dir / "w12_planar_edge_baseline_summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with (args.output_dir / "w12_planar_edge_baseline_cases.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "case",
            "P_Q_W",
            "P_six_W",
            "closure",
            "auto_shutoff",
            "Qx_W",
            "Qy_W",
            "Qz_W",
            "hotspot_x_um",
            "hotspot_y_um",
            "hotspot_z_nm",
            "near_stack_downward_fit_waist_x_um",
            "near_stack_downward_fit_waist_y_um",
            "near_stack_fit_residual",
            "flake_total_E2_fit_waist_x_um",
            "flake_total_E2_fit_waist_y_um",
            "flake_total_E2_fit_residual",
            "solver_wall_time_s",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in CASE_ORDER:
            case = cases[name]
            near = case["field_readback"]["near_stack_total_field_plane"][
                "downward_decomposition_fit_at_1_W_m2_center"
            ]
            flake = case["field_readback"]["flake_midplane_total_field"][
                "total_E2_spatial_fit_at_1_W_m2_center"
            ]
            writer.writerow(
                {
                    "case": name,
                    "P_Q_W": case["P_Q_W"],
                    "P_six_W": case["P_six_face_W"],
                    "closure": case["six_face_relative_closure"],
                    "auto_shutoff": case["auto_shutoff"]["final_value"],
                    "Qx_W": case["component_power_W"]["x"],
                    "Qy_W": case["component_power_W"]["y"],
                    "Qz_W": case["component_power_W"]["z"],
                    "hotspot_x_um": case["Q_hotspot"]["x_m"] * 1e6,
                    "hotspot_y_um": case["Q_hotspot"]["y_m"] * 1e6,
                    "hotspot_z_nm": case["Q_hotspot"]["z_m"] * 1e9,
                    "near_stack_downward_fit_waist_x_um": near["waist_x_m"] * 1e6,
                    "near_stack_downward_fit_waist_y_um": near["waist_y_m"] * 1e6,
                    "near_stack_fit_residual": near["fit_relative_RMS_over_peak"],
                    "flake_total_E2_fit_waist_x_um": flake["waist_x_m"] * 1e6,
                    "flake_total_E2_fit_waist_y_um": flake["waist_y_m"] * 1e6,
                    "flake_total_E2_fit_residual": flake[
                        "fit_relative_RMS_over_peak"
                    ],
                    "solver_wall_time_s": case["solver_wall_time_s"],
                }
            )

    with (args.output_dir / "w12_planar_edge_spatial_comparisons.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = ["comparison", *next(iter(comparisons.values())).keys()]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, values in comparisons.items():
            writer.writerow({"comparison": name, **values})

    z_weight = trapezoid_weights(coordinates["z"])
    maps = {
        name: np.sum(q_data[name]["total"] * z_weight[None, None, :], axis=2)
        for name in CASE_ORDER
    }
    normalized_maps = {
        name: maps[name] / cases[name]["P_Q_W"] * 1.0e-12
        for name in CASE_ORDER
    }
    extent = [
        coordinates["x"][0] * 1e6,
        coordinates["x"][-1] * 1e6,
        coordinates["y"][0] * 1e6,
        coordinates["y"][-1] * 1e6,
    ]
    figure, axes = plt.subplots(2, 4, figsize=(16.5, 7.8), constrained_layout=True)
    raw_max = max(float(np.max(value)) for value in maps.values())
    normalized_max = max(float(np.max(value)) for value in normalized_maps.values())
    for column, name in enumerate(CASE_ORDER):
        raw_image = axes[0, column].imshow(
            maps[name].T,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=raw_max,
            aspect="equal",
        )
        norm_image = axes[1, column].imshow(
            normalized_maps[name].T,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=0.0,
            vmax=normalized_max,
            aspect="equal",
        )
        axes[0, column].set_title(f"{LABELS[name]}\nraw $\\int Q dz$")
        axes[1, column].set_title(f"{LABELS[name]}\nequal-power shape")
        for row in range(2):
            axes[row, column].set(xlabel="x (µm)", ylabel="y (µm)")
    figure.colorbar(raw_image, ax=axes[0, :], label="W/m²", shrink=0.75)
    figure.colorbar(norm_image, ax=axes[1, :], label="1/µm²", shrink=0.75)
    figure.savefig(args.output_dir / "W12_RAW_AND_EQUAL_POWER_Q_MAPS.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.7), constrained_layout=True)
    powers = np.asarray([cases[name]["P_Q_W"] for name in CASE_ORDER])
    components = {
        axis: np.asarray(
            [cases[name]["component_power_W"][axis] for name in CASE_ORDER]
        )
        for axis in "xyz"
    }
    bottom = np.zeros(len(CASE_ORDER))
    for axis, color in zip("xyz", ("tab:blue", "tab:orange", "tab:green")):
        axes[0].bar(CASE_ORDER, components[axis], bottom=bottom, label=f"$Q_{axis}$", color=color)
        bottom += components[axis]
    axes[0].scatter(
        CASE_ORDER,
        [cases[name]["P_six_face_W"] for name in CASE_ORDER],
        marker="x",
        color="black",
        label="$P_{six}$",
    )
    axes[0].set(ylabel="Power (W)", title="Raw absorption components")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        CASE_ORDER,
        [100 * cases[name]["six_face_relative_closure"] for name in CASE_ORDER],
        color="tab:purple",
    )
    axes[1].axhline(0.5, color="red", linestyle="--", label="0.5% gate")
    axes[1].set(ylabel="closure (%)", title="Matched-volume closure")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(args.output_dir / "W12_POWER_COMPONENT_CLOSURE.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), constrained_layout=True)
    for polarization, color in (("a", "tab:blue"), ("b", "tab:orange")):
        for geometry, linestyle in (("planar", "--"), ("edge", "-")):
            name = f"{geometry}_{polarization}"
            coordinate, profile = edge_profile(
                q_data[name]["total"],
                coordinates,
                tangent_window_m=10.0e-6,
            )
            axes[0].plot(
                coordinate * 1e6,
                profile * 1e-6,
                color=color,
                linestyle=linestyle,
                label=LABELS[name],
            )
    axes[0].set(
        xlabel=r"edge-normal $n=(y-x)/\sqrt{2}$ (µm)",
        ylabel="equal-power profile (1/µm)",
        title="Spatial shape only",
        xlim=(-25, 25),
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].bar(
        ("a", "b"),
        (
            cases["edge_a"]["P_Q_W"] / cases["planar_a"]["P_Q_W"],
            cases["edge_b"]["P_Q_W"] / cases["planar_b"]["P_Q_W"],
        ),
        color=("tab:blue", "tab:orange"),
    )
    axes[1].set(
        ylabel="$P_{Q,edge}/P_{Q,planar}$",
        title="Raw power retained by finite-edge case",
        ylim=(0, 1),
    )
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(args.output_dir / "W12_EDGE_NORMAL_PROFILES.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.7), constrained_layout=True)
    near_waists = []
    near_residuals = []
    flake_waists = []
    flake_residuals = []
    flake_center = []
    for name in CASE_ORDER:
        near = cases[name]["field_readback"]["near_stack_total_field_plane"][
            "downward_decomposition_fit_at_1_W_m2_center"
        ]
        flake = cases[name]["field_readback"]["flake_midplane_total_field"][
            "total_E2_spatial_fit_at_1_W_m2_center"
        ]
        near_waists.append(near["waist_effective_geometric_mean_m"] * 1e6)
        near_residuals.append(100 * near["fit_relative_RMS_over_peak"])
        flake_waists.append(flake["waist_effective_geometric_mean_m"] * 1e6)
        flake_residuals.append(100 * flake["fit_relative_RMS_over_peak"])
        flake_center.append(
            np.hypot(flake["center_x_m"], flake["center_y_m"]) * 1e6
        )
    axes[0].bar(CASE_ORDER, near_waists, color="tab:cyan")
    axes[0].axhline(12.0, color="black", linestyle="--", label="requested 12 µm")
    axes[0].set(ylabel="effective fit width (µm)", title="Near-stack downward diagnostic")
    axes[0].legend()
    axes[1].bar(CASE_ORDER, flake_waists, color="tab:olive")
    axes[1].set(ylabel="effective total-$E^2$ fit width (µm)", title="TaIrTe₄ midplane")
    axes[2].bar(CASE_ORDER, flake_residuals, color="tab:red", label="fit residual")
    axes[2].plot(CASE_ORDER, flake_center, "ko--", label="center displacement (µm)")
    axes[2].set(ylabel="percent or µm", title="Edge-induced non-Gaussianity")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(args.output_dir / "W12_FIELD_READBACK.png", dpi=180)
    plt.close(figure)

    manifest_path = args.output_dir / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "raw_NPZ_or_FSP_committed_to_git": False,
                "raw_artifact_hash_policy": (
                    "Large raw Q/FSP hashes are copied from their immutable "
                    "case-local manifests; JSON/manifests are rehashed here."
                ),
                "summary_generation_command": " ".join(sys.argv),
                "artifacts": manifest,
                "generation_commit": git_commit(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = f"""# Paper-like w0=12 µm scalar-Gaussian planar/edge baseline

Status: `{payload['status']}`

This is a **paper-like scalar-Gaussian scenario with an explicitly assumed
waist**.  The 12 µm waist is not published by the paper.  This is not an
experimentally reproduced beam, paper-certified beam, paper reproduction, or
promoted production heat source.

No thermal, PTE, adjoint, gradient, or optimization calculation was run.

## Fixed optical contract

- wavelength: 11 µm
- scalar Gaussian, assumed waist radius 12 µm
- source span: 50×50 µm²
- FDTD span: 60×60 µm², six PML, no periodic boundaries
- TaIrTe₄: 130 nm; local baseline mesh 100 nm in x/y and 5 nm in z
- 285 nm SiO₂ on Si
- lab x=b, lab y=a, epsilon_z=epsilon_c=epsilon_b closure
- GPU FDTD only; no CPU fallback
- no Q clipping, smoothing, gain, rescaling, tiling, or deletion

## Baseline gates

- closure <0.5% for all four cases: **{closure_pass}**
- auto-shutoff ≤1e-5 for all four: **{shutoff_pass}**
- Q reintegration error <0.5%: **{reintegration_pass}**
- no negative-Q voxels: **{finite_pass}**
- no Q/source modification: **{no_modification}**

| case | P_Q (W) | P_six (W) | closure | auto-shutoff | Qx/Qy/Qz (W) |
|---|---:|---:|---:|---:|---|
"""
    for name in CASE_ORDER:
        case = cases[name]
        components = case["component_power_W"]
        report += (
            f"| {name} | {case['P_Q_W']:.9e} | "
            f"{case['P_six_face_W']:.9e} | "
            f"{case['six_face_relative_closure']:.6%} | "
            f"{case['auto_shutoff']['final_value']:.6e} | "
            f"{components['x']:.3e} / {components['y']:.3e} / "
            f"{components['z']:.3e} |\n"
        )
    report += f"""

Raw edge/planar absorbed-power ratios are
`{cases['edge_a']['P_Q_W']/cases['planar_a']['P_Q_W']:.6f}` for a-polarization
and `{cases['edge_b']['P_Q_W']/cases['planar_b']['P_Q_W']:.6f}` for
b-polarization.  These raw powers were not equalized.  Equal-power
normalization appears only in the spatial-shape metrics and plots.

## Saved-field readback

The requested 0.6 µm monitor is realized on the Yee plane at approximately
0.5136 µm.  Its downward E/H decomposition remains a total-field diagnostic
that can contain reflection, scattering, and evanescent fields; it is not
called a pure incident waist.  Component-specific E fields inside TaIrTe₄ at
z=-65 nm were independently read on their staggered coordinates and
interpolated only to their exact common support.  Same-index component pairing
and extrapolation were not used.

The planar flake-midplane total-E² fit widths remain near 12 µm.  The
finite-edge fits have larger residuals and shifted centers, which is reported
as edge-induced non-Gaussian field redistribution rather than a shifted
incident beam.

## Remaining gate

The current 100 nm x/y mesh is the four-case baseline.  A 50 nm local-x/y
comparison remains required before promoting a material-Q artifact.  No
refinement run was started in this checkpoint.
"""
    (args.output_dir / "W12_PLANAR_EDGE_BASELINE_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(payload["gates"], indent=2))
    return 0 if baseline_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
