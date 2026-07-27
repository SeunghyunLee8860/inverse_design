#!/usr/bin/env python3
"""Resolve the preserved 6 um raw-PTE thermal mesh subgate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np

from .explicit_thermal import build_explicit_geometry, solve_explicit_forward
from .run_fixed_q_thermal_convergence import (
    cell_volume,
    mapped_source,
    nrmse,
    probe_points,
    probe_temperature,
    relative_difference,
    sha256,
)
from .validate_large_background_local_q_mapping import (
    native_q_from_mapping,
)


STATUS_PASS = "VALIDATED_THERMAL_RAW_PTE_REFINED_SUBGATE"
STATUS_FAIL = "FAILED_THERMAL_RAW_PTE_REFINED_SUBGATE"
PTE_LIMIT = 5.0e-3
ENERGY_LIMIT = 1.0e-2
RESIDUAL_LIMIT = 1.0e-8
PREVIOUS_NATIVE_REFINED_RAW_PTE_CHANGE = 0.006295360570799928


@dataclass(frozen=True)
class Mesh:
    label: str
    core_xy_nm: float
    flake_dz_nm: float
    design_dz_nm: float


MESHES = (
    Mesh("preserved_refined_50nm", 50.0, 12.5, 50.0),
    Mesh("additional_40nm", 40.0, 10.0, 40.0),
    Mesh(
        "additional_33p333nm",
        100.0 / 3.0,
        25.0 / 3.0,
        100.0 / 3.0,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-mapping", required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_mapping = Path(args.input_mapping).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "thermal_raw_pte_refined_subgate_result.json"
    result: dict[str, object] = {
        "status": "BLOCKED_THERMAL_RAW_PTE_REFINED_SUBGATE_NOT_RUN",
        "passed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "fixed optical Q, 6 um named thermal footprint, preserved "
            "50 nm refined mesh followed by 40 and 33.333 nm meshes"
        ),
        "preserved_previous_checkpoint": {
            "native_to_refined_raw_PTE_relative_change": (
                PREVIOUS_NATIVE_REFINED_RAW_PTE_CHANGE
            ),
            "limit": PTE_LIMIT,
            "status": "RAW_PTE_LT_0P5PCT_UNRESOLVED",
            "modified": False,
        },
        "maxwell_run": False,
        "adjoint_run": False,
        "finite_difference_run": False,
        "optimization_run": False,
    }
    started = time.monotonic()
    try:
        if not input_mapping.is_file():
            raise FileNotFoundError(input_mapping)
        input_sha = sha256(input_mapping)
        if input_sha != args.expected_input_sha256:
            raise RuntimeError("input mapping SHA-256 mismatch")
        native = native_q_from_mapping(input_mapping)
        points, probe_shape = probe_points(6.0)
        cases = []
        raw_artifacts = []
        for mesh in MESHES:
            case_started = time.monotonic()
            count = int(round(2000.0 / mesh.core_xy_nm))
            if abs(count * mesh.core_xy_nm - 2000.0) > 1.0e-9:
                raise RuntimeError("mesh does not exactly tile design")
            rho = np.full((count, count), 0.5)
            kwargs = {
                "lateral_domain_m": 32.0e-6,
                "si_depth_m": 20.0e-6,
                "flake_span_m": 6.0e-6,
                "core_xy_cell_size_m": mesh.core_xy_nm * 1.0e-9,
                "flake_dz_m": mesh.flake_dz_nm * 1.0e-9,
                "design_dz_m": mesh.design_dz_nm * 1.0e-9,
            }
            geometry = build_explicit_geometry(rho, **kwargs)
            source, mapping = mapped_source(native, geometry)
            if (
                mapping["relative_power_error"] >= 5.0e-3
                or mapping["outside_TaIrTe4_nonzero_cell_count"] != 0
            ):
                raise RuntimeError("invalid source remap")
            solved = solve_explicit_forward(
                rho=rho, source_W_m3=source, **kwargs
            )
            theta = solved.solved.temperature_K
            probe = probe_temperature(
                geometry, theta, points, probe_shape
            )
            volume = cell_volume(geometry)
            flake_theta = theta[geometry.flake_mask]
            flake_volume = volume[geometry.flake_mask]
            raw_path = output / f"{mesh.label}.npz"
            np.savez_compressed(
                raw_path,
                source_W_m3=source,
                temperature_K=theta,
                common_probe_temperature_K=probe,
                rho=rho,
                flake_mask=geometry.flake_mask,
                x_edges_m=geometry.x_edges_m,
                y_edges_m=geometry.y_edges_m,
                z_edges_m=geometry.z_edges_m,
            )
            raw_record = {
                "path": str(raw_path),
                "byte_size": raw_path.stat().st_size,
                "sha256": sha256(raw_path),
            }
            raw_artifacts.append(raw_record)
            cases.append(
                {
                    "label": mesh.label,
                    "controls": {
                        "lateral_domain_um": 32.0,
                        "si_depth_um": 20.0,
                        "flake_span_um": 6.0,
                        "core_xy_cell_size_nm": mesh.core_xy_nm,
                        "flake_dz_nm": mesh.flake_dz_nm,
                        "design_dz_nm": mesh.design_dz_nm,
                    },
                    "grid_shape": list(theta.shape),
                    "total_cells": int(theta.size),
                    "mapped_Q_power_W": mapping["mapped_power_W"],
                    "Q_mapping_relative_power_error": mapping[
                        "relative_power_error"
                    ],
                    "Q_outside_flake_nonzero_count": mapping[
                        "outside_TaIrTe4_nonzero_cell_count"
                    ],
                    "Tmax_DeltaT_K": float(np.max(theta)),
                    "TaIrTe4_volume_average_DeltaT_K": float(
                        np.sum(flake_theta * flake_volume)
                        / np.sum(flake_volume)
                    ),
                    "PTE_objective_A": float(solved.objective_A),
                    "energy_balance_relative_error": float(
                        solved.solved.energy_balance_relative_error
                    ),
                    "linear_residual_relative": float(
                        solved.solved.linear_residual_relative
                    ),
                    "raw_artifact": raw_record,
                    "wall_s": time.monotonic() - case_started,
                    "_probe": probe,
                }
            )
            print(
                "THERMAL_RAW_PTE_REFINED "
                f"case={mesh.label} cells={theta.size} "
                f"PTE={solved.objective_A:.9e} "
                f"residual={solved.solved.linear_residual_relative:.3e}",
                flush=True,
            )
        comparisons = []
        for coarse, fine in zip(cases[:-1], cases[1:]):
            comparison = {
                "coarse": coarse["label"],
                "fine": fine["label"],
                "PTE_raw_relative_difference": relative_difference(
                    coarse["PTE_objective_A"], fine["PTE_objective_A"]
                ),
                "Tmax_relative_difference": relative_difference(
                    coarse["Tmax_DeltaT_K"], fine["Tmax_DeltaT_K"]
                ),
                "TaIrTe4_volume_average_relative_difference": (
                    relative_difference(
                        coarse["TaIrTe4_volume_average_DeltaT_K"],
                        fine["TaIrTe4_volume_average_DeltaT_K"],
                    )
                ),
                "TaIrTe4_common_probe_NRMSE": nrmse(
                    coarse["_probe"], fine["_probe"]
                ),
            }
            comparison["passed_raw_PTE_0p5pct"] = bool(
                comparison["PTE_raw_relative_difference"] < PTE_LIMIT
            )
            comparisons.append(comparison)
        for case in cases:
            del case["_probe"]
        worst_energy = max(
            case["energy_balance_relative_error"] for case in cases
        )
        worst_residual = max(
            case["linear_residual_relative"] for case in cases
        )
        passed = bool(
            all(
                item["passed_raw_PTE_0p5pct"]
                for item in comparisons
            )
            and worst_energy < ENERGY_LIMIT
            and worst_residual < RESIDUAL_LIMIT
        )
        result.update(
            {
                "status": STATUS_PASS if passed else STATUS_FAIL,
                "passed": passed,
                "input_native_Q_artifact": {
                    "path": str(input_mapping),
                    "byte_size": input_mapping.stat().st_size,
                    "sha256": input_sha,
                },
                "cases": cases,
                "comparisons": comparisons,
                "gates": {
                    "raw_PTE_relative_change_limit": PTE_LIMIT,
                    "worst_new_pair_raw_PTE_relative_change": max(
                        item["PTE_raw_relative_difference"]
                        for item in comparisons
                    ),
                    "energy_balance_limit": ENERGY_LIMIT,
                    "worst_energy_balance_relative_error": worst_energy,
                    "linear_residual_limit": RESIDUAL_LIMIT,
                    "worst_linear_residual_relative": worst_residual,
                },
                "raw_artifacts": raw_artifacts,
                "interpretation": (
                    "The historical native-to-50nm 0.6295% raw-PTE "
                    "difference remains a failed comparison. Passing "
                    "successive finer pairs resolves the production "
                    "discretization subgate without rewriting that result."
                ),
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": STATUS_FAIL,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        result["wall_s"] = time.monotonic() - started
        result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "result_path": str(result_path),
            }
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
