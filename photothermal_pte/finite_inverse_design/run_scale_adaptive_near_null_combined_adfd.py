#!/usr/bin/env python3
"""Extend only the unresolved near-null combined AD--FD directions.

The immutable five-direction sweep used h=(0.01, 0.005, 0.0025).  Its weak
central/random directional derivatives reached the FDTD finite-difference
resolution at the finest step.  This diagnostic adds only h=0.02 and tests
the coarser halving sequence (0.02, 0.01, 0.005), reusing every existing
h=0.01 and h=0.005 solve.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np

from .contract import design_nodal_coordinates_m
from .probe_v261_gpu_plane_wave_roi import load_lumapi
from .run_combined_physical_rho_pte_adfd import (
    array_sha256,
    physical_state,
    run_forward_density,
)
from .run_corrected_combined_physical_rho_pte_adfd import (
    FLUX_SIGNS,
    completed_forward,
    fd_step_convergence,
    objective_for_scenario,
    relative,
    thermal_state,
)
from .run_v261_large_background_tfsf_forward import sha256


STATUS_PASS = "VALIDATED_SCALE_ADAPTIVE_NEAR_NULL_COMBINED_ADFD"
STATUS_FAIL = "FAILED_SCALE_ADAPTIVE_NEAR_NULL_COMBINED_ADFD"
ADFD_RELATIVE_LIMIT = 1.0e-2
REQUIRED_EXISTING_STEPS = (0.01, 0.005)
ADDED_STEP = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-forward", required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--failed-result", required=True)
    parser.add_argument("--failed-result-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threads", type=int, default=16)
    return parser.parse_args()


def checked_file(path_text: str, expected_sha: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file() or sha256(path) != expected_sha:
        raise RuntimeError(f"missing or SHA-mismatched artifact: {path}")
    return path


def normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    scale = float(np.max(np.abs(values)))
    if not np.isfinite(scale) or scale == 0.0:
        raise RuntimeError("invalid direction")
    return values / scale


def near_null_directions() -> dict[str, np.ndarray]:
    x, y = design_nodal_coordinates_m()
    xn = x[:, None] / 1.0e-6
    yn = y[None, :] / 1.0e-6
    rng = np.random.default_rng(2026072711)
    return {
        "central_localized": normalized(
            np.exp(-((xn / 0.24) ** 2 + (yn / 0.24) ** 2))
        ),
        "fixed_seed_random": normalized(rng.normal(size=(81, 81))),
    }


def old_step(direction_data: dict, step: float) -> dict:
    matches = [
        row
        for row in direction_data["steps"]
        if np.isclose(float(row["step"]), step, rtol=0.0, atol=1.0e-15)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"missing unique immutable h={step:g} row")
    row = matches[0]
    return {
        "step": float(row["step"]),
        "finite_difference_directional_A": float(
            row["finite_difference_directional_A"]
        ),
        "relative_error": float(row["relative_error"]),
        "provenance": "immutable_failed_five_direction_sweep",
    }


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "scale_adaptive_near_null_combined_adfd.json"
    base_project = checked_file(args.base_forward, args.base_sha256)
    failed_path = checked_file(
        args.failed_result, args.failed_result_sha256
    )
    failed = json.loads(failed_path.read_text())
    result: dict = {
        "status": STATUS_FAIL,
        "passed": False,
        "full_combined_certificate_claimed": False,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "clipping": False,
        "original_failed_result": {
            "path": str(failed_path),
            "sha256": args.failed_result_sha256,
            "status": failed.get("status"),
        },
        "step_policy": {
            "immutable_sequence": [0.01, 0.005, 0.0025],
            "near_null_sequence": [0.02, 0.01, 0.005],
            "reason": (
                "avoid the measured finest-step FDTD cancellation floor "
                "without changing the density direction or gradient scale"
            ),
        },
    }
    fdtd = None
    started = time.monotonic()
    try:
        rho, _ = physical_state()
        if array_sha256(rho) != failed["physical_rho_sha256"]:
            raise RuntimeError("physical-rho contract changed")
        directions = near_null_directions()
        for name, direction in directions.items():
            if array_sha256(direction) != failed["directions"][name]["sha256"]:
                raise RuntimeError(f"{name} direction contract changed")
        if np.min(rho) <= ADDED_STEP or np.min(1.0 - rho) <= ADDED_STEP:
            raise RuntimeError("h=0.02 would require clipping")

        lumapi = load_lumapi()
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        base = run_forward_density(
            fdtd,
            rho=rho,
            project=base_project,
            threads=args.threads,
            flux_signs=FLUX_SIGNS,
            reuse_completed=True,
        )
        scenarios = {}
        for flake_um in (4.0, 6.0):
            kwargs, coupling, geometry, mapping, evaluation = thermal_state(
                rho, flake_um, base["native"]
            )
            scenarios[flake_um] = {
                "kwargs": kwargs,
                "coupling": coupling,
                "geometry": geometry,
                "mapping": mapping,
                "base_evaluation": evaluation,
            }

        new_pairs = {}
        for name, direction in directions.items():
            pair = {}
            for sign, sign_name in ((1.0, "plus"), (-1.0, "minus")):
                perturbed = rho + sign * ADDED_STEP * direction
                role = (
                    f"near_null_{name}_h{ADDED_STEP:g}_{sign_name}"
                    "_cpu_tfsf"
                )
                forward, forward_meta = completed_forward(
                    fdtd,
                    rho=perturbed,
                    base_project=base_project,
                    output=output,
                    role=role,
                    threads=args.threads,
                )
                pair[sign_name] = {
                    "forward": forward_meta,
                    "objectives": {},
                }
                for flake_um, data in scenarios.items():
                    objective, diagnostics = objective_for_scenario(
                        rho=perturbed,
                        forward=forward,
                        data=data,
                    )
                    pair[sign_name]["objectives"][f"{flake_um:g}um"] = {
                        "objective_A": objective,
                        **diagnostics,
                    }
            new_pairs[name] = pair

        cases = []
        all_pass = True
        all_closure = []
        all_mapping = []
        all_energy = []
        all_residual = []
        for flake_um in (4.0, 6.0):
            scenario_key = f"{flake_um:g}um"
            for name in directions:
                immutable = failed["scenarios"][scenario_key][
                    "directions"
                ][name]
                analytic = float(immutable["analytic_directional_A"])
                plus = new_pairs[name]["plus"]["objectives"][scenario_key]
                minus = new_pairs[name]["minus"]["objectives"][scenario_key]
                new_fd = (
                    float(plus["objective_A"])
                    - float(minus["objective_A"])
                ) / (2.0 * ADDED_STEP)
                rows = [
                    {
                        "step": ADDED_STEP,
                        "finite_difference_directional_A": new_fd,
                        "relative_error": relative(analytic, new_fd),
                        "provenance": "new_h0.02_pair",
                    },
                    *[
                        old_step(immutable, step)
                        for step in REQUIRED_EXISTING_STEPS
                    ],
                ]
                convergence = fd_step_convergence(rows)
                selected = rows[-1]
                passed = bool(
                    selected["relative_error"] < ADFD_RELATIVE_LIMIT
                    and convergence["step_convergence_passed"]
                )
                all_pass = all_pass and passed
                cases.append(
                    {
                        "scenario": scenario_key,
                        "direction": name,
                        "analytic_directional_A": analytic,
                        "rows": rows,
                        **convergence,
                        "selected_step": selected["step"],
                        "selected_relative_error": selected[
                            "relative_error"
                        ],
                        "selected_relative_error_limit": ADFD_RELATIVE_LIMIT,
                        "passed": passed,
                    }
                )
                for sign_name in ("plus", "minus"):
                    forward_meta = new_pairs[name][sign_name]["forward"]
                    diagnostics = new_pairs[name][sign_name][
                        "objectives"
                    ][scenario_key]
                    all_closure.append(
                        forward_meta["six_face_closure_relative_error"]
                    )
                    all_mapping.append(
                        diagnostics["Q_mapping_relative_error"]
                    )
                    all_energy.append(
                        diagnostics["energy_balance_relative_error"]
                    )
                    all_residual.append(
                        diagnostics["linear_residual_relative"]
                    )

        diagnostic_gates = {
            "worst_new_optical_closure_relative_error": max(all_closure),
            "optical_closure_limit": 5.0e-3,
            "worst_new_Q_mapping_relative_error": max(all_mapping),
            "Q_mapping_limit": 5.0e-3,
            "worst_new_thermal_energy_balance_relative_error": max(all_energy),
            "thermal_energy_balance_limit": 1.0e-2,
            "worst_new_linear_residual_relative": max(all_residual),
            "linear_residual_limit": 1.0e-8,
        }
        all_pass = bool(
            all_pass
            and diagnostic_gates[
                "worst_new_optical_closure_relative_error"
            ]
            < diagnostic_gates["optical_closure_limit"]
            and diagnostic_gates["worst_new_Q_mapping_relative_error"]
            < diagnostic_gates["Q_mapping_limit"]
            and diagnostic_gates[
                "worst_new_thermal_energy_balance_relative_error"
            ]
            < diagnostic_gates["thermal_energy_balance_limit"]
            and diagnostic_gates["worst_new_linear_residual_relative"]
            < diagnostic_gates["linear_residual_limit"]
        )
        result.update(
            {
                "status": STATUS_PASS if all_pass else STATUS_FAIL,
                "passed": all_pass,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "base_forward": {
                    "path": str(base_project),
                    "sha256": args.base_sha256,
                },
                "directions": {
                    name: {"sha256": array_sha256(direction)}
                    for name, direction in directions.items()
                },
                "cases": cases,
                "new_pairs": new_pairs,
                "diagnostic_gates": diagnostic_gates,
                "next_gate": (
                    "PUBLISH_COMPOSITE_COMBINED_PHYSICAL_RHO_CERTIFICATE"
                    if all_pass
                    else "RETAIN_COMBINED_PHYSICAL_RHO_FAIL_CLOSED"
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
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        result["wall_s"] = time.monotonic() - started
        result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "path": str(result_path)}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
