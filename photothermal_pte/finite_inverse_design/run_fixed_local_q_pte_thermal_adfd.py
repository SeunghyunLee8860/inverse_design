#!/usr/bin/env python3
"""Fixed-local-Q PTE thermal-material/interface AD--FD certificate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from .explicit_thermal import (
    build_explicit_geometry,
    evaluate_explicit_thermal,
    solve_explicit_forward,
)
from .run_fixed_q_thermal_convergence import (
    cell_volume,
    mapped_source,
    sha256,
)
from .run_large_background_local_q_thermal_adfd import (
    interface_diagnostics,
)
from .validate_large_background_local_q_mapping import (
    native_q_from_mapping,
)


STATUS_PASS = "VALIDATED_FIXED_LOCAL_Q_PTE_THERMAL_ONLY_ADFD"
STATUS_FAIL = "FAILED_FIXED_LOCAL_Q_PTE_THERMAL_ONLY_ADFD"
ADFD_LIMIT = 5.0e-3
ENERGY_LIMIT = 1.0e-2
RESIDUAL_LIMIT = 1.0e-8
SIGNAL_RATIO_MINIMUM = 1.0e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-mapping", required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", default="0.01,0.005")
    return parser.parse_args()


def array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(json.dumps(values.shape).encode())
    digest.update(values.view(np.uint8))
    return digest.hexdigest()


def directions(
    shape: tuple[int, int], gradient: np.ndarray
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2026072707)
    x = np.linspace(-1.0, 1.0, shape[0])[:, None]
    y = np.linspace(-1.0, 1.0, shape[1])[None, :]
    x_cell = (
        -1.0
        + (np.arange(shape[0], dtype=float) + 0.5)
        * (2.0 / shape[0])
    )[:, None]
    y_cell = (
        -1.0
        + (np.arange(shape[1], dtype=float) + 0.5)
        * (2.0 / shape[1])
    )[None, :]
    raw = {
        "adjoint_aligned": gradient,
        "seeded_random": rng.normal(size=shape),
        "asymmetric_smooth": (
            np.sin(0.7 * np.pi * (x + 0.15))
            * np.cos(0.45 * np.pi * (y - 0.2))
        ),
        "central_localized": np.exp(
            -(x_cell**2 + y_cell**2) / (2.0 * 0.14**2)
        ),
        "design_edge_localized": np.exp(
            -(
                (x_cell - 0.88) ** 2
                + (y_cell + 0.23) ** 2
            )
            / (2.0 * 0.075**2)
        ),
    }
    result = {}
    for name, value in raw.items():
        maximum = float(np.max(np.abs(value)))
        if maximum == 0.0:
            raise RuntimeError(f"zero direction: {name}")
        result[name] = value / maximum
    return result


def main() -> int:
    args = parse_args()
    input_mapping = Path(args.input_mapping).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "fixed_local_q_pte_thermal_adfd_summary.json"
    result: dict[str, object] = {
        "status": "BLOCKED_FIXED_LOCAL_Q_PTE_THERMAL_ADFD_NOT_RUN",
        "passed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "fixed native optical Q; explicit thermal bulk-k/interface-G/"
            "surface derivative; uniform-45-degree PTE current surrogate"
        ),
        "maxwell_run": False,
        "optical_gradient_run": False,
        "nodal_81x81_mapping_run": False,
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
        steps = [float(value) for value in args.steps.split(",")]
        if not steps or any(step <= 0.0 for step in steps):
            raise ValueError("positive FD steps are required")
        native = native_q_from_mapping(input_mapping)
        scenarios = []
        rows = []
        raw_artifacts = []
        for flake_span_um in (4.0, 6.0):
            scenario_started = time.monotonic()
            rho = np.full((20, 20), 0.5)
            kwargs = {
                "lateral_domain_m": 32.0e-6,
                "si_depth_m": 20.0e-6,
                "flake_span_m": flake_span_um * 1.0e-6,
                "core_xy_cell_size_m": 100.0e-9,
                "flake_dz_m": 25.0e-9,
                "design_dz_m": 100.0e-9,
            }
            geometry = build_explicit_geometry(rho, **kwargs)
            source, source_record = mapped_source(native, geometry)
            if (
                source_record["relative_power_error"] >= 5.0e-3
                or source_record[
                    "outside_TaIrTe4_nonzero_cell_count"
                ]
                != 0
            ):
                raise RuntimeError("invalid fixed-Q mapping")
            source_sha = array_sha256(source)
            base = evaluate_explicit_thermal(
                rho=rho,
                source_W_m3=source,
                **kwargs,
            )
            gradient_l1 = float(np.sum(np.abs(base.gradient_rho_A)))
            direction_records = []
            for name, direction in directions(
                rho.shape, base.gradient_rho_A
            ).items():
                analytic = float(
                    np.sum(base.gradient_rho_A * direction)
                )
                components = {
                    "bulk_k_A": float(
                        np.sum(base.gradient_bulk_k_A * direction)
                    ),
                    "interface_G_A": float(
                        np.sum(base.gradient_interface_G_A * direction)
                    ),
                    "top_convection_k_A": float(
                        np.sum(
                            base.gradient_top_convection_k_A * direction
                        )
                    ),
                }
                component_sum = float(sum(components.values()))
                component_sum_error = abs(component_sum - analytic) / max(
                    abs(component_sum),
                    abs(analytic),
                    np.finfo(float).tiny,
                )
                signal_ratio = abs(analytic) / max(
                    gradient_l1, np.finfo(float).tiny
                )
                step_records = []
                for step in steps:
                    fd_started = time.monotonic()
                    plus_rho = rho + step * direction
                    minus_rho = rho - step * direction
                    plus = solve_explicit_forward(
                        rho=plus_rho,
                        source_W_m3=source,
                        **kwargs,
                    )
                    minus = solve_explicit_forward(
                        rho=minus_rho,
                        source_W_m3=source,
                        **kwargs,
                    )
                    finite_difference = (
                        plus.objective_A - minus.objective_A
                    ) / (2.0 * step)
                    error = abs(analytic - finite_difference) / max(
                        abs(analytic),
                        abs(finite_difference),
                        np.finfo(float).tiny,
                    )
                    row = {
                        "flake_span_um": flake_span_um,
                        "direction": name,
                        "step": step,
                        "adjoint_directional_A": analytic,
                        "finite_difference_directional_A": (
                            finite_difference
                        ),
                        "relative_error": error,
                        "signal_ratio": signal_ratio,
                        "included_in_gate": (
                            name == "adjoint_aligned"
                            or signal_ratio >= SIGNAL_RATIO_MINIMUM
                        ),
                        **components,
                        "component_sum_relative_error": component_sum_error,
                        "plus_source_array_sha256": array_sha256(source),
                        "minus_source_array_sha256": array_sha256(source),
                        "plus_energy_balance_relative_error": float(
                            plus.solved.energy_balance_relative_error
                        ),
                        "minus_energy_balance_relative_error": float(
                            minus.solved.energy_balance_relative_error
                        ),
                        "plus_linear_residual_relative": float(
                            plus.solved.linear_residual_relative
                        ),
                        "minus_linear_residual_relative": float(
                            minus.solved.linear_residual_relative
                        ),
                        "wall_s": time.monotonic() - fd_started,
                    }
                    rows.append(row)
                    step_records.append(row)
                    print(
                        "FIXED_Q_PTE_THERMAL_ADFD "
                        f"flake={flake_span_um:g}um direction={name} "
                        f"h={step:g} error={error:.6e} "
                        f"signal={signal_ratio:.3e}",
                        flush=True,
                    )
                direction_records.append(
                    {
                        "name": name,
                        "adjoint_directional_A": analytic,
                        "signal_ratio": signal_ratio,
                        "included_in_gate": step_records[0][
                            "included_in_gate"
                        ],
                        "components": components,
                        "component_sum_relative_error": (
                            component_sum_error
                        ),
                        "steps": step_records,
                    }
                )
            theta = base.solved.temperature_K
            volume = cell_volume(base.geometry)
            flake_volume = volume[base.geometry.flake_mask]
            flake_theta = theta[base.geometry.flake_mask]
            raw_path = output / f"flake_{flake_span_um:g}um_adfd.npz"
            np.savez_compressed(
                raw_path,
                rho=rho,
                source_W_m3=source,
                theta_K=theta,
                adjoint_active=base.adjoint_active,
                gradient_rho_A=base.gradient_rho_A,
                gradient_bulk_k_A=base.gradient_bulk_k_A,
                gradient_interface_G_A=base.gradient_interface_G_A,
                gradient_top_convection_k_A=(
                    base.gradient_top_convection_k_A
                ),
                x_edges_m=base.geometry.x_edges_m,
                y_edges_m=base.geometry.y_edges_m,
                z_edges_m=base.geometry.z_edges_m,
                material_id=base.geometry.material_id,
            )
            raw_record = {
                "path": str(raw_path),
                "byte_size": raw_path.stat().st_size,
                "sha256": sha256(raw_path),
            }
            raw_artifacts.append(raw_record)
            scenarios.append(
                {
                    "name": f"TaIrTe4_{flake_span_um:g}um_footprint",
                    "flake_span_um": flake_span_um,
                    "fabrication_truth": False,
                    "thermal_density_grid": {
                        "kind": (
                            "20x20 cell-centered native thermal control; "
                            "not the approved 81x81 nodal design mapping"
                        ),
                        "shape": list(rho.shape),
                        "baseline_rho": 0.5,
                    },
                    "source": {
                        **source_record,
                        "array_sha256": source_sha,
                        "held_bitwise_identical_in_all_plus_minus_runs": (
                            True
                        ),
                    },
                    "grid_shape": list(theta.shape),
                    "total_cells": int(theta.size),
                    "PTE_objective_A": float(base.objective_A),
                    "PTE_temperature_source_norm_A_K": float(
                        np.linalg.norm(
                            base.pte.temperature_source_A_K
                        )
                    ),
                    "Tmax_DeltaT_K": float(np.max(theta)),
                    "TaIrTe4_volume_average_DeltaT_K": float(
                        np.sum(flake_theta * flake_volume)
                        / np.sum(flake_volume)
                    ),
                    "gradient_norms_A": {
                        "total": float(
                            np.linalg.norm(base.gradient_rho_A)
                        ),
                        "bulk_k": float(
                            np.linalg.norm(base.gradient_bulk_k_A)
                        ),
                        "interface_G": float(
                            np.linalg.norm(
                                base.gradient_interface_G_A
                            )
                        ),
                        "top_convection_k": float(
                            np.linalg.norm(
                                base.gradient_top_convection_k_A
                            )
                        ),
                    },
                    "forward_energy_balance_relative_error": float(
                        base.solved.energy_balance_relative_error
                    ),
                    "forward_linear_residual_relative": float(
                        base.solved.linear_residual_relative
                    ),
                    "adjoint_linear_residual_relative": float(
                        base.adjoint_linear_residual_relative
                    ),
                    "forward_iterations": int(base.solved.iterations),
                    "adjoint_iterations": int(base.adjoint_iterations),
                    "boundary_power_out_W": {
                        key: float(value)
                        for key, value in (
                            base.solved.boundary_power_out_W.items()
                        )
                    },
                    "interface_diagnostics": interface_diagnostics(base),
                    "directions": direction_records,
                    "raw_artifact": raw_record,
                    "wall_s": time.monotonic() - scenario_started,
                }
            )
        gated_rows = [row for row in rows if row["included_in_gate"]]
        selected_rows = [
            row
            for row in gated_rows
            if row["step"] == min(steps)
        ]
        worst_selected_adfd = max(
            row["relative_error"] for row in selected_rows
        )
        worst_energy = max(
            [
                scenario["forward_energy_balance_relative_error"]
                for scenario in scenarios
            ]
            + [
                value
                for row in rows
                for value in (
                    row["plus_energy_balance_relative_error"],
                    row["minus_energy_balance_relative_error"],
                )
            ]
        )
        worst_residual = max(
            [
                value
                for scenario in scenarios
                for value in (
                    scenario["forward_linear_residual_relative"],
                    scenario["adjoint_linear_residual_relative"],
                )
            ]
            + [
                value
                for row in rows
                for value in (
                    row["plus_linear_residual_relative"],
                    row["minus_linear_residual_relative"],
                )
            ]
        )
        worst_component_sum = max(
            direction["component_sum_relative_error"]
            for scenario in scenarios
            for direction in scenario["directions"]
        )
        passed = bool(
            worst_selected_adfd < ADFD_LIMIT
            and worst_energy < ENERGY_LIMIT
            and worst_residual < RESIDUAL_LIMIT
            and worst_component_sum < 1.0e-12
            and all(
                scenario["source"][
                    "held_bitwise_identical_in_all_plus_minus_runs"
                ]
                for scenario in scenarios
            )
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
                "native_P_Q_W": float(native["P_Q_W"]),
                "thermal_operator": (
                    "K_T(rho) theta = M_V Q_fixed; "
                    "dI/drho = -lambda_T^T(dK_T/drho)theta"
                ),
                "thermal_gradient_components": [
                    "bulk design kappa",
                    "TaIrTe4/design internal-interface G",
                    "design exposed-surface half-cell kappa contribution",
                ],
                "PTE_functional": (
                    "uniform 45-degree weighting-field surrogate; "
                    "not solved finite-contact terminal current"
                ),
                "scenarios": scenarios,
                "rows": rows,
                "gates": {
                    "selected_FD_step": min(steps),
                    "AD_FD_relative_error_limit": ADFD_LIMIT,
                    "signal_ratio_minimum": SIGNAL_RATIO_MINIMUM,
                    "worst_selected_gated_AD_FD_relative_error": (
                        worst_selected_adfd
                    ),
                    "worst_energy_balance_relative_error": worst_energy,
                    "energy_balance_limit": ENERGY_LIMIT,
                    "worst_linear_residual_relative": worst_residual,
                    "linear_residual_limit": RESIDUAL_LIMIT,
                    "worst_gradient_component_sum_relative_error": (
                        worst_component_sum
                    ),
                },
                "raw_artifacts": raw_artifacts,
            }
        )
    except Exception as exc:
        result["status"] = "BLOCKED_FIXED_LOCAL_Q_PTE_THERMAL_ADFD_EXECUTION"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        result["wall_s"] = time.monotonic() - started
        summary_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    return 0 if result.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
