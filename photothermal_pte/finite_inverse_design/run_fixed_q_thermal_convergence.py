#!/usr/bin/env python3
"""Fixed-local-Q domain, depth, and mesh convergence for explicit thermal FVM.

The two TaIrTe4 footprints are named numerical scenarios.  This runner does
not select either footprint as fabrication truth and does not run an adjoint,
finite difference, transient solve, PTE optimization, or optical simulation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .explicit_thermal import build_explicit_geometry, solve_explicit_forward
from .finite_q_mapping import (
    build_conservative_embedding_remap,
    exact_nonzero_box,
    nodal_control_volume_edges,
    project_remap_to_material_support_along_axis,
)
from .run_large_background_local_q_thermal_adfd import (
    interface_diagnostics,
)
from .validate_large_background_local_q_mapping import (
    native_q_from_mapping,
)


STATUS_PASS = "VALIDATED_FIXED_Q_THERMAL_DOMAIN_DEPTH_MESH_CONVERGENCE"
STATUS_FAIL = "FAILED_FIXED_Q_THERMAL_DOMAIN_DEPTH_MESH_CONVERGENCE"
POWER_LIMIT = 5.0e-3
ENERGY_LIMIT = 1.0e-2
RESIDUAL_LIMIT = 1.0e-8
CONVERGENCE_LIMIT = 1.0e-2


@dataclass(frozen=True)
class Case:
    label: str
    lateral_domain_um: float
    si_depth_um: float
    core_xy_nm: float
    flake_dz_nm: float
    design_dz_nm: float


CASES = (
    Case("native", 32.0, 20.0, 100.0, 25.0, 100.0),
    Case("lateral_40um", 40.0, 20.0, 100.0, 25.0, 100.0),
    Case("si_depth_30um", 32.0, 30.0, 100.0, 25.0, 100.0),
    Case("refined", 32.0, 20.0, 50.0, 12.5, 50.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-mapping", required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def cell_volume(geometry) -> np.ndarray:
    return (
        np.diff(geometry.x_edges_m)[:, None, None]
        * np.diff(geometry.y_edges_m)[None, :, None]
        * np.diff(geometry.z_edges_m)[None, None, :]
    )


def mapped_source(native: dict[str, object], geometry) -> tuple[
    np.ndarray, dict[str, object]
]:
    target_edges = (
        geometry.x_edges_m,
        geometry.y_edges_m,
        geometry.z_edges_m,
    )
    mapped_total = np.zeros(geometry.material_id.shape, float)
    components: dict[str, object] = {}
    for component in "xyz":
        density = np.asarray(native["Q_components"][component], float)
        if not np.any(density):
            components[component] = {
                "native_power_W": float(
                    native["component_power_W"][component]
                ),
                "mapped_power_W": 0.0,
                "exact_zero": True,
            }
            continue
        box, outside_nonzero = exact_nonzero_box(density)
        if outside_nonzero:
            raise RuntimeError("native nonzero Q exists outside selected box")
        source_edges = tuple(
            nodal_control_volume_edges(
                native["native_coordinates"][component][axis]
            )[section.start : section.stop + 1]
            for axis, section in zip("xyz", box)
        )
        source = density[box]
        geometric = build_conservative_embedding_remap(
            source_edges_m=source_edges,
            target_edges_m=target_edges,
        )
        remap = project_remap_to_material_support_along_axis(
            geometric,
            target_edges_m=target_edges,
            target_support_mask=geometry.flake_mask,
            axis=2,
        )
        mapped = remap.apply(source)
        mapped_total += mapped
        components[component] = {
            "native_power_W": float(
                native["component_power_W"][component]
            ),
            "mapped_power_W": float(remap.power_target(mapped)),
            "exact_zero": False,
        }
    volume = cell_volume(geometry)
    mapped_power = float(np.sum(volume * mapped_total))
    native_power = float(native["P_Q_W"])
    error = abs(mapped_power - native_power) / max(
        abs(native_power), np.finfo(float).tiny
    )
    outside_power = float(
        np.sum(volume[~geometry.flake_mask] * mapped_total[
            ~geometry.flake_mask
        ])
    )
    outside_nonzero = int(
        np.count_nonzero(mapped_total[~geometry.flake_mask])
    )
    return mapped_total, {
        "native_power_W": native_power,
        "mapped_power_W": mapped_power,
        "relative_power_error": error,
        "outside_TaIrTe4_power_W": outside_power,
        "outside_TaIrTe4_nonzero_cell_count": outside_nonzero,
        "components": components,
    }


def probe_points(flake_span_um: float) -> tuple[np.ndarray, tuple[int, ...]]:
    span_m = flake_span_um * 1.0e-6
    x = np.arange(
        -0.5 * span_m + 50.0e-9,
        0.5 * span_m,
        100.0e-9,
    )
    y = x.copy()
    z = np.arange(-100.0e-9 + 12.5e-9, 0.0, 25.0e-9)
    mesh = np.meshgrid(x, y, z, indexing="ij")
    points = np.column_stack([axis.reshape(-1) for axis in mesh])
    return points, (x.size, y.size, z.size)


def probe_temperature(
    geometry, theta: np.ndarray, points: np.ndarray, shape: tuple[int, ...]
) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        (
            centers(geometry.x_edges_m),
            centers(geometry.y_edges_m),
            centers(geometry.z_edges_m),
        ),
        theta,
        method="linear",
        bounds_error=True,
    )
    return np.asarray(interpolator(points), float).reshape(shape)


def relative_difference(value: float, reference: float) -> float:
    return abs(value - reference) / max(
        abs(value), abs(reference), np.finfo(float).tiny
    )


def nrmse(value: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(value - reference)
        / max(
            np.linalg.norm(value),
            np.linalg.norm(reference),
            np.finfo(float).tiny,
        )
    )


def case_geometry(case: Case, flake_span_um: float):
    design_cells = int(round(2000.0 / case.core_xy_nm))
    rho = np.full((design_cells, design_cells), 0.5)
    geometry = build_explicit_geometry(
        rho,
        lateral_domain_m=case.lateral_domain_um * 1.0e-6,
        si_depth_m=case.si_depth_um * 1.0e-6,
        flake_span_m=flake_span_um * 1.0e-6,
        core_xy_cell_size_m=case.core_xy_nm * 1.0e-9,
        flake_dz_m=case.flake_dz_nm * 1.0e-9,
        design_dz_m=case.design_dz_nm * 1.0e-9,
    )
    return rho, geometry


def metrics(forward, source_record: dict[str, object]) -> dict[str, object]:
    geometry = forward.geometry
    theta = forward.solved.temperature_K
    volume = cell_volume(geometry)
    flake_volume = volume[geometry.flake_mask]
    flake_theta = theta[geometry.flake_mask]
    x = centers(geometry.x_edges_m)
    y = centers(geometry.y_edges_m)
    z = centers(geometry.z_edges_m)
    hotspot = np.unravel_index(int(np.argmax(theta)), theta.shape)
    flake_indices = np.argwhere(geometry.flake_mask)
    flake_hotspot = flake_indices[int(np.argmax(flake_theta))]
    pte_contributions = (
        forward.pte.temperature_source_A_K * forward.theta_active_K
    )
    pte_scale = float(np.sum(np.abs(pte_contributions)))
    return {
        "grid_shape": list(theta.shape),
        "total_cells": int(theta.size),
        "source": source_record,
        "Tmax_DeltaT_K": float(np.max(theta)),
        "hotspot_m": {
            "x": float(x[hotspot[0]]),
            "y": float(y[hotspot[1]]),
            "z": float(z[hotspot[2]]),
            "material_id": int(geometry.material_id[hotspot]),
        },
        "TaIrTe4_Tmax_DeltaT_K": float(np.max(flake_theta)),
        "TaIrTe4_volume_average_DeltaT_K": float(
            np.sum(flake_theta * flake_volume) / np.sum(flake_volume)
        ),
        "TaIrTe4_hotspot_m": {
            "x": float(x[flake_hotspot[0]]),
            "y": float(y[flake_hotspot[1]]),
            "z": float(z[flake_hotspot[2]]),
        },
        "PTE_objective_A": float(forward.objective_A),
        "PTE_absolute_contribution_scale_A": pte_scale,
        "PTE_cancellation_ratio": float(
            abs(forward.objective_A)
            / max(pte_scale, np.finfo(float).tiny)
        ),
        "source_power_W": float(forward.solved.source_power_W),
        "boundary_power_out_W": {
            key: float(value)
            for key, value in forward.solved.boundary_power_out_W.items()
        },
        "numerical_boundary_power_fraction": {
            key: float(value / forward.solved.source_power_W)
            for key, value in forward.solved.boundary_power_out_W.items()
        },
        "energy_balance_relative_error": float(
            forward.solved.energy_balance_relative_error
        ),
        "linear_residual_relative": float(
            forward.solved.linear_residual_relative
        ),
        "interface_diagnostics": interface_diagnostics(forward),
    }


def main() -> int:
    args = parse_args()
    input_mapping = Path(args.input_mapping).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "fixed_q_thermal_convergence_summary.json"
    result: dict[str, object] = {
        "status": "BLOCKED_FIXED_Q_THERMAL_CONVERGENCE_NOT_RUN",
        "passed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "fixed local optical Q; independent thermal domain, Si-depth, "
            "and mesh convergence for named 4 um and 6 um TaIrTe4 "
            "footprint scenarios"
        ),
        "pte_adjoint_run": False,
        "finite_difference_run": False,
        "optimization_run": False,
        "transient_run": False,
    }
    started = time.monotonic()
    try:
        if not input_mapping.is_file():
            raise FileNotFoundError(input_mapping)
        input_sha = sha256(input_mapping)
        if input_sha != args.expected_input_sha256:
            raise RuntimeError("input mapping SHA-256 does not match")
        native = native_q_from_mapping(input_mapping)
        scenarios = []
        raw_artifacts = []
        all_comparisons = []
        for flake_span_um in (4.0, 6.0):
            points, probe_shape = probe_points(flake_span_um)
            baseline_metrics = None
            baseline_probe = None
            case_records = []
            for case in CASES:
                case_started = time.monotonic()
                rho, geometry = case_geometry(case, flake_span_um)
                source, source_record = mapped_source(native, geometry)
                if (
                    source_record["relative_power_error"] >= POWER_LIMIT
                    or source_record[
                        "outside_TaIrTe4_nonzero_cell_count"
                    ]
                    != 0
                ):
                    raise RuntimeError(
                        f"{flake_span_um:g} um {case.label}: invalid source "
                        "mapping"
                    )
                forward = solve_explicit_forward(
                    rho=rho,
                    source_W_m3=source,
                    lateral_domain_m=case.lateral_domain_um * 1.0e-6,
                    si_depth_m=case.si_depth_um * 1.0e-6,
                    flake_span_m=flake_span_um * 1.0e-6,
                    core_xy_cell_size_m=case.core_xy_nm * 1.0e-9,
                    flake_dz_m=case.flake_dz_nm * 1.0e-9,
                    design_dz_m=case.design_dz_nm * 1.0e-9,
                )
                probe = probe_temperature(
                    geometry,
                    forward.solved.temperature_K,
                    points,
                    probe_shape,
                )
                current = {
                    "label": case.label,
                    "flake_span_um": flake_span_um,
                    "controls": {
                        "lateral_domain_um": case.lateral_domain_um,
                        "si_depth_um": case.si_depth_um,
                        "core_xy_cell_size_nm": case.core_xy_nm,
                        "flake_dz_nm": case.flake_dz_nm,
                        "design_dz_nm": case.design_dz_nm,
                    },
                    **metrics(forward, source_record),
                    "wall_s": time.monotonic() - case_started,
                }
                if baseline_metrics is None:
                    baseline_metrics = current
                    baseline_probe = probe
                    current["comparison_to_native"] = None
                else:
                    assert baseline_probe is not None
                    pte_scale = max(
                        current["PTE_absolute_contribution_scale_A"],
                        baseline_metrics[
                            "PTE_absolute_contribution_scale_A"
                        ],
                        np.finfo(float).tiny,
                    )
                    comparison = {
                        "TaIrTe4_field_probe_NRMSE": nrmse(
                            probe, baseline_probe
                        ),
                        "Tmax_relative_difference": relative_difference(
                            current["Tmax_DeltaT_K"],
                            baseline_metrics["Tmax_DeltaT_K"],
                        ),
                        "TaIrTe4_Tmax_relative_difference": (
                            relative_difference(
                                current["TaIrTe4_Tmax_DeltaT_K"],
                                baseline_metrics[
                                    "TaIrTe4_Tmax_DeltaT_K"
                                ],
                            )
                        ),
                        "TaIrTe4_volume_average_relative_difference": (
                            relative_difference(
                                current[
                                    "TaIrTe4_volume_average_DeltaT_K"
                                ],
                                baseline_metrics[
                                    "TaIrTe4_volume_average_DeltaT_K"
                                ],
                            )
                        ),
                        "PTE_raw_relative_difference": (
                            relative_difference(
                                current["PTE_objective_A"],
                                baseline_metrics["PTE_objective_A"],
                            )
                        ),
                        "PTE_contribution_normalized_difference": float(
                            abs(
                                current["PTE_objective_A"]
                                - baseline_metrics["PTE_objective_A"]
                            )
                            / pte_scale
                        ),
                    }
                    comparison["maximum_thermal_metric_difference"] = max(
                        comparison["TaIrTe4_field_probe_NRMSE"],
                        comparison["Tmax_relative_difference"],
                        comparison[
                            "TaIrTe4_volume_average_relative_difference"
                        ],
                        comparison[
                            "PTE_contribution_normalized_difference"
                        ],
                    )
                    comparison["passed"] = bool(
                        comparison["maximum_thermal_metric_difference"]
                        < CONVERGENCE_LIMIT
                    )
                    current["comparison_to_native"] = comparison
                    all_comparisons.append(comparison)
                raw_path = output / (
                    f"flake_{flake_span_um:g}um_{case.label}.npz"
                )
                np.savez_compressed(
                    raw_path,
                    theta_K=forward.solved.temperature_K,
                    source_W_m3=source,
                    rho=rho,
                    x_edges_m=geometry.x_edges_m,
                    y_edges_m=geometry.y_edges_m,
                    z_edges_m=geometry.z_edges_m,
                    material_id=geometry.material_id,
                    flake_mask=geometry.flake_mask,
                    common_probe_temperature_K=probe,
                )
                raw_record = {
                    "path": str(raw_path),
                    "byte_size": raw_path.stat().st_size,
                    "sha256": sha256(raw_path),
                }
                current["raw_artifact"] = raw_record
                raw_artifacts.append(raw_record)
                case_records.append(current)
                print(
                    "THERMAL_CONVERGENCE "
                    f"flake={flake_span_um:g}um case={case.label} "
                    f"cells={forward.solved.temperature_K.size} "
                    f"Tmax={current['Tmax_DeltaT_K']:.8e} "
                    f"energy={current['energy_balance_relative_error']:.3e} "
                    f"residual={current['linear_residual_relative']:.3e}",
                    flush=True,
                )
            scenarios.append(
                {
                    "name": f"TaIrTe4_{flake_span_um:g}um_footprint",
                    "flake_span_um": flake_span_um,
                    "fabrication_truth": False,
                    "cases": case_records,
                }
            )
        numerical_gates = {
            "mapping_power_limit": POWER_LIMIT,
            "energy_balance_limit": ENERGY_LIMIT,
            "linear_residual_limit": RESIDUAL_LIMIT,
            "convergence_metric_limit": CONVERGENCE_LIMIT,
            "worst_mapping_power_error": max(
                case["source"]["relative_power_error"]
                for scenario in scenarios
                for case in scenario["cases"]
            ),
            "worst_energy_balance_error": max(
                case["energy_balance_relative_error"]
                for scenario in scenarios
                for case in scenario["cases"]
            ),
            "worst_linear_residual": max(
                case["linear_residual_relative"]
                for scenario in scenarios
                for case in scenario["cases"]
            ),
            "worst_convergence_metric": max(
                comparison["maximum_thermal_metric_difference"]
                for comparison in all_comparisons
            ),
        }
        passed = bool(
            numerical_gates["worst_mapping_power_error"] < POWER_LIMIT
            and numerical_gates["worst_energy_balance_error"] < ENERGY_LIMIT
            and numerical_gates["worst_linear_residual"] < RESIDUAL_LIMIT
            and numerical_gates["worst_convergence_metric"]
            < CONVERGENCE_LIMIT
        )
        result.update(
            {
                "status": STATUS_PASS if passed else STATUS_FAIL,
                "passed": passed,
                "input_native_Q_artifact": {
                    "path": str(input_mapping),
                    "byte_size": input_mapping.stat().st_size,
                    "sha256": input_sha,
                    "note": (
                        "only immutable native Yee Q arrays are reused; "
                        "the prior mapped thermal Q is ignored"
                    ),
                },
                "native_P_Q_W": float(native["P_Q_W"]),
                "case_contract": [
                    {
                        "label": case.label,
                        "lateral_domain_um": case.lateral_domain_um,
                        "si_depth_um": case.si_depth_um,
                        "core_xy_cell_size_nm": case.core_xy_nm,
                        "flake_dz_nm": case.flake_dz_nm,
                        "design_dz_nm": case.design_dz_nm,
                    }
                    for case in CASES
                ],
                "common_probe": {
                    "method": (
                        "trilinear cell-center interpolation onto a fixed "
                        "100 nm x-y by 25 nm z grid covering the complete "
                        "TaIrTe4 footprint"
                    ),
                    "z_bounds_m": [-100.0e-9, 0.0],
                },
                "boundary_flux_interpretation": (
                    "lateral and bottom Dirichlet power entries are "
                    "numerical truncation-boundary fluxes, not intrinsic "
                    "physical heat-path fractions"
                ),
                "scenarios": scenarios,
                "gates": numerical_gates,
                "raw_artifacts": raw_artifacts,
            }
        )
    except Exception as exc:
        result["status"] = "BLOCKED_FIXED_Q_THERMAL_CONVERGENCE_EXECUTION"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        result["wall_s"] = time.monotonic() - started
        summary_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    return 0 if result.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
