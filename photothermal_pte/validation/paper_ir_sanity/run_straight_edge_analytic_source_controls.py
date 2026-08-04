#!/usr/bin/env python3
"""Separate absorbed-power and source-shape controls for the paper IR edge.

This command never opens Lumerical.  It solves the same paper-reduced
TaIrTe4 thermal operator with three explicitly different source contracts:

* ``paper_like_absorbed_power_control`` uses the polarization-dependent TMM
  absorbed power and Beer--Lambert depth profile;
* ``equal_absorbed_power_shape_control`` gives the two Beer--Lambert profiles
  exactly the same finite-half-plane absorbed power; and
* ``identical_Q_symmetry_control`` supplies bitwise-identical Q arrays under
  the two polarization labels.

The equal-power operation is confined to this analytic control.  It does not
modify, normalize, or rescale a saved Lumerical Q artifact.
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
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
BASE_PATH = HERE / "run_straight_edge_analytic_q_control.py"
STATUS = "VALIDATED_ANALYTIC_THERMAL_SOURCE_CONTROLS"


def load_base() -> Any:
    spec = importlib.util.spec_from_file_location(
        "straight_edge_analytic_control_base",
        BASE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--core-steps-nm",
        type=float,
        nargs="+",
        default=(200.0, 100.0, 50.0),
    )
    parser.add_argument("--thermal-domain-um", type=float, default=48.0)
    parser.add_argument("--flake-dz-nm", type=float, default=26.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        text=True,
    ).strip()


def integrate_volume(
    values: np.ndarray,
    x_edges_m: np.ndarray,
    y_edges_m: np.ndarray,
    z_edges_m: np.ndarray,
) -> float:
    volume = (
        np.diff(x_edges_m)[:, None, None]
        * np.diff(y_edges_m)[None, :, None]
        * np.diff(z_edges_m)[None, None, :]
    )
    return float(np.sum(np.asarray(values, float) * volume))


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        raise ValueError("symmetry arrays have no common finite support")
    numerator = float(np.linalg.norm((a - b)[finite]))
    denominator = max(
        float(np.linalg.norm(a[finite])),
        float(np.linalg.norm(b[finite])),
        np.finfo(float).tiny,
    )
    return numerator / denominator


def source_for_control(
    expanded_geometry: Any,
    *,
    control: str,
    polarization: str,
    equal_power_W: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    profile_polarization = (
        "a" if control == "identical_Q_symmetry_control" else polarization
    )
    q, source = base.analytic_q(
        expanded_geometry,
        profile_polarization,
        0.0,
    )
    before = integrate_volume(
        q,
        expanded_geometry.x_edges_m,
        expanded_geometry.y_edges_m,
        expanded_geometry.z_edges_m,
    )
    scale = 1.0
    if control == "equal_absorbed_power_shape_control":
        scale = equal_power_W / before
        q = q * scale
    after = integrate_volume(
        q,
        expanded_geometry.x_edges_m,
        expanded_geometry.y_edges_m,
        expanded_geometry.z_edges_m,
    )
    return q, {
        **source,
        "polarization_label": polarization,
        "Beer_Lambert_profile_polarization": profile_polarization,
        "analytic_control": control,
        "finite_half_plane_power_before_control_W": before,
        "finite_half_plane_power_after_control_W": after,
        "analytic_only_scale_factor": scale,
        "raw_Lumerical_Q_modified": False,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cases_dir = args.output_dir / "cases"
    cases_dir.mkdir()
    equal_power_W = (
        0.5
        * base.INCIDENT_POWER_W
        * np.mean(list(base.TMM_ABSORPTION.values()))
    )
    controls = (
        "paper_like_absorbed_power_control",
        "equal_absorbed_power_shape_control",
        "identical_Q_symmetry_control",
    )
    rows: list[dict[str, Any]] = []
    case_arrays: dict[tuple[float, str, str], dict[str, np.ndarray]] = {}

    for core_step_nm in args.core_steps_nm:
        geometry_args = SimpleNamespace(
            thermal_domain_um=args.thermal_domain_um,
            si_depth_um=20.0,
            core_step_nm=core_step_nm,
            flake_dz_nm=args.flake_dz_nm,
            thermal_model="paper-reduced",
        )
        expanded_geometry = base.build_straight_geometry(geometry_args)
        zero_q = np.zeros(expanded_geometry.material_id.shape, float)
        geometry, _ = base.select_thermal_model(
            expanded_geometry,
            zero_q,
            "paper-reduced",
        )
        system = base.assemble_system(geometry, "paper-reduced")
        for control in controls:
            for polarization in ("a", "b"):
                expanded_q, source = source_for_control(
                    expanded_geometry,
                    control=control,
                    polarization=polarization,
                    equal_power_W=equal_power_W,
                )
                _, q = base.select_thermal_model(
                    expanded_geometry,
                    expanded_q,
                    "paper-reduced",
                )
                solved = base.thermal.solve_assembled_thermal_system(
                    system,
                    source_W_m3=q,
                    relative_tolerance=1e-10,
                    max_iterations=12000,
                )
                metrics, fields = base.thermal.straight_edge_temperature_metrics(
                    solved.temperature_K,
                    geometry,
                )
                label = (
                    f"{control}_{polarization}_core"
                    f"{int(round(core_step_nm))}"
                )
                artifact_path = cases_dir / f"{label}.npz"
                arrays = {
                    "Q_areal_W_m2": base.areal_q(q, geometry.z_edges_m),
                    "temperature_flake_average_K": fields[
                        "temperature_flake_average_K"
                    ],
                    "grad_T_x_K_m": fields["grad_T_x_K_m"],
                    "grad_T_y_K_m": fields["grad_T_y_K_m"],
                    "grad_T_normal_K_m": fields["grad_T_normal_K_m"],
                    "grad_T_tangent_K_m": fields["grad_T_tangent_K_m"],
                    "grad_T_magnitude_K_m": fields["grad_T_magnitude_K_m"],
                }
                np.savez_compressed(
                    artifact_path,
                    x_edges_m=geometry.x_edges_m,
                    y_edges_m=geometry.y_edges_m,
                    **arrays,
                )
                case_arrays[(core_step_nm, control, polarization)] = arrays
                rows.append(
                    {
                        "control": control,
                        "polarization": polarization,
                        "core_step_nm": core_step_nm,
                        "source_power_W": solved.source_power_W,
                        "linear_residual_relative": (
                            solved.linear_residual_relative
                        ),
                        "energy_balance_relative_error": (
                            solved.energy_balance_relative_error
                        ),
                        "iterations": solved.iterations,
                        "Tmax_rise_K": metrics["Tmax_rise_K"],
                        "max_abs_grad_T_x_K_m": metrics[
                            "max_abs_grad_T_x_K_m"
                        ],
                        "max_abs_edge_normal_gradient_K_m": metrics[
                            "max_abs_edge_normal_gradient_K_m"
                        ],
                        "source_contract_json": json.dumps(
                            source,
                            sort_keys=True,
                        ),
                        "artifact_path": str(artifact_path.resolve()),
                        "artifact_sha256": sha256(artifact_path),
                    }
                )

    by_key = {
        (row["core_step_nm"], row["control"], row["polarization"]): row
        for row in rows
    }
    comparisons: dict[str, Any] = {}
    gates: list[bool] = []
    for core_step_nm in args.core_steps_nm:
        mesh: dict[str, Any] = {}
        for control in controls:
            a = by_key[(core_step_nm, control, "a")]
            b = by_key[(core_step_nm, control, "b")]
            mesh[control] = {
                "absorbed_power_b_over_a": (
                    b["source_power_W"] / a["source_power_W"]
                ),
                "Tmax_b_over_a": b["Tmax_rise_K"] / a["Tmax_rise_K"],
                "max_abs_grad_T_x_b_over_a": (
                    b["max_abs_grad_T_x_K_m"]
                    / a["max_abs_grad_T_x_K_m"]
                ),
                "max_abs_grad_T_n_b_over_a": (
                    b["max_abs_edge_normal_gradient_K_m"]
                    / a["max_abs_edge_normal_gradient_K_m"]
                ),
            }
            if control == "equal_absorbed_power_shape_control":
                power_error = abs(
                    a["source_power_W"] - b["source_power_W"]
                ) / equal_power_W
                mesh[control][
                    "equal_absorbed_power_relative_error"
                ] = power_error
                gates.append(power_error < 1e-12)
            if control == "identical_Q_symmetry_control":
                arrays_a = case_arrays[
                    (core_step_nm, control, "a")
                ]
                arrays_b = case_arrays[
                    (core_step_nm, control, "b")
                ]
                symmetry = {
                    name: {
                        "maximum_absolute_difference": float(
                            np.max(
                                np.abs(
                                    arrays_a[name][
                                        np.isfinite(arrays_a[name])
                                        & np.isfinite(arrays_b[name])
                                    ]
                                    - arrays_b[name][
                                        np.isfinite(arrays_a[name])
                                        & np.isfinite(arrays_b[name])
                                    ]
                                )
                            )
                        ),
                        "relative_L2_difference": relative_l2(
                            arrays_a[name],
                            arrays_b[name],
                        ),
                    }
                    for name in arrays_a
                }
                mesh[control]["field_symmetry"] = symmetry
                gates.extend(
                    item["relative_L2_difference"] < 1e-10
                    for item in symmetry.values()
                )
        comparisons[f"core_{int(round(core_step_nm))}_nm"] = mesh

    gates.extend(
        row["linear_residual_relative"] < 1e-8
        and row["energy_balance_relative_error"] < 0.01
        for row in rows
    )
    status = STATUS if all(gates) else "FAILED_ANALYTIC_THERMAL_SOURCE_CONTROLS"
    summary = {
        "status": status,
        "scope": (
            "offline paper-reduced TaIrTe4 thermal source controls; "
            "no Lumerical solve and no raw Maxwell-Q modification"
        ),
        "source_controls": {
            "paper_like_absorbed_power_control": (
                "polarization-dependent TMM absorbed power and "
                "Beer-Lambert depth profile"
            ),
            "equal_absorbed_power_shape_control": {
                "description": (
                    "same exact finite-half-plane absorbed power; only the "
                    "a/b Beer-Lambert depth profile differs"
                ),
                "target_power_W": equal_power_W,
                "normalization_scope": "analytic control only",
            },
            "identical_Q_symmetry_control": (
                "the exact a-profile Q array is supplied under both labels"
            ),
        },
        "comparisons": comparisons,
        "thermal_contract": {
            "identity": "paper Supplement Eq. S4 reduced flake-only Robin",
            "TaIrTe4_region": "y<=x",
            "z_min_G_W_m2K": base.thermal.G_TAIRTE4_SIO2_W_M2K,
            "z_max_G_W_m2K": base.thermal.G_TAIRTE4_AIR_W_M2K,
            "lateral_Dirichlet": False,
        },
        "generation_commit": git_commit(),
        "generation_command": shlex.join([sys.executable, *sys.argv]),
        "raw_artifacts_committed_to_git": False,
        "optimization_run": False,
    }
    summary_path = args.output_dir / "analytic_source_controls_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    csv_path = args.output_dir / "analytic_source_controls_cases.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    return 0 if status == STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
