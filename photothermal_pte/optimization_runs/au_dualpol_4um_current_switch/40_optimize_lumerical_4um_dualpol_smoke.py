#!/usr/bin/env python3
"""Run exactly two fail-closed LD_MMA evaluations on the Lumerical carrier."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import nlopt
import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_design_mapping import (
    NOMINAL_MAPPING,
    exact_binary_cell_candidate,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_optimizer import (
    SMOKE_MAXEVAL,
    LumericalEvaluationDriver,
    OptimizerRuntime,
    SmokeEpigraphProblem,
    artifact,
    initial_latent_density,
    smoke_preflight,
)


def _git_commit(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    started = time.monotonic()
    result: dict[str, object] = {
        "status": "FAILED_LUMERICAL_4UM_DUALPOL_LD_MMA_SMOKE",
        "passed": False,
        "optimizer": "NLopt LD_MMA",
        "requested_maxeval": SMOKE_MAXEVAL,
        "Lumerical_HEAT_or_CHARGE_solves": 0,
        "FDTDX_Maxwell_solves": 0,
    }
    output: Path | None = None
    try:
        runtime = OptimizerRuntime.from_environment()
        output = runtime.output_root
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"refusing non-empty optimizer output root: {output}")
        output.mkdir(parents=True, exist_ok=True)
        preflight = smoke_preflight(runtime)
        _write_json(output / "smoke_preflight.json", preflight)
        if not preflight["passed"]:
            raise RuntimeError("optimizer smoke preflight failed")
        if os.environ.get("AU_LUMERICAL_OPT_PREFLIGHT_ONLY", "0") == "1":
            result = {
                "status": "PASSED_LUMERICAL_4UM_OPTIMIZER_SMOKE_PREFLIGHT_ONLY",
                "passed": True,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "solver-free optimizer and artifact preflight only",
                "smoke_preflight": artifact(output / "smoke_preflight.json"),
                "Lumerical_Maxwell_solves": 0,
                "custom_CUDA_solves": 0,
                "Lumerical_HEAT_or_CHARGE_solves": 0,
                "FDTDX_Maxwell_solves": 0,
                "optimizer_iterations": 0,
                "wall_s": time.monotonic() - started,
            }
            _write_json(output / "lumerical_dualpol_smoke_result.json", result)
            print(json.dumps(result, indent=2))
            return 0

        latent_initial = initial_latent_density()
        dfm_caps = np.asarray(preflight["DFM_caps_for_smoke"], np.float64)
        driver = LumericalEvaluationDriver(runtime)
        initial_physics = driver.evaluate(latent_initial)
        epigraph_initial_nA = 1.0e9 * min(
            float(initial_physics["currents_A"]["Ea"]),
            -float(initial_physics["currents_A"]["Eb"]),
        )
        problem = SmokeEpigraphProblem(
            driver.evaluate, beta=runtime.beta, dfm_caps=dfm_caps
        )
        variable_count = int(np.prod(CONTRACT.design_node_shape)) + 1
        optimizer = nlopt.opt(nlopt.LD_MMA, variable_count)
        optimizer.set_lower_bounds(np.r_[np.zeros(variable_count - 1), -100.0])
        optimizer.set_upper_bounds(np.r_[np.ones(variable_count - 1), 1000.0])
        optimizer.set_max_objective(problem.objective)
        optimizer.add_inequality_mconstraint(
            problem.constraints, np.full(4, 1.0e-6, np.float64)
        )
        optimizer.set_initial_step(
            np.r_[np.full(variable_count - 1, 0.01), 0.1]
        )
        optimizer.set_ftol_rel(0.0)
        optimizer.set_xtol_rel(0.0)
        optimizer.set_maxeval(SMOKE_MAXEVAL)
        vector_initial = np.r_[latent_initial.ravel(), epigraph_initial_nA]
        vector_final = optimizer.optimize(vector_initial)
        final_point = problem.point(vector_final)
        unique_evaluations = len(driver.history)
        topology_changed = not np.array_equal(
            vector_initial[:-1], vector_final[:-1]
        )
        gates = {
            "preflight_passed": bool(preflight["passed"]),
            "exactly_two_unique_physics_evaluations": unique_evaluations == 2,
            "NLopt_reported_two_function_evaluations": optimizer.get_numevals()
            == SMOKE_MAXEVAL,
            "optimizer_changed_latent_topology": topology_changed,
            "finite_final_vector": bool(np.all(np.isfinite(vector_final))),
            "final_vector_inside_bounds": bool(
                np.min(vector_final[:-1]) >= 0.0
                and np.max(vector_final[:-1]) <= 1.0
            ),
            "Ea_Eb_final_currents_finite": bool(
                np.isfinite(final_point["current_a_A"])
                and np.isfinite(final_point["current_b_A"])
            ),
            "no_Lumerical_HEAT_or_CHARGE": True,
            "no_FDTDX_Maxwell": True,
        }
        latent_final = vector_final[:-1].reshape(CONTRACT.design_node_shape)
        projected_final = NOMINAL_MAPPING.physical(latent_final, runtime.beta)
        binary_mask, binary_audit = exact_binary_cell_candidate(projected_final)
        final_npz = output / "smoke_final_state.npz"
        np.savez_compressed(
            final_npz,
            latent_initial=latent_initial,
            latent_final=latent_final,
            projected_final=projected_final,
            binary_candidate_cell_mask=binary_mask,
        )
        passed = all(gates.values())
        result = {
            "status": "PASSED_LUMERICAL_4UM_DUALPOL_LD_MMA_SMOKE"
            if passed
            else "FAILED_LUMERICAL_4UM_DUALPOL_LD_MMA_SMOKE",
            "passed": passed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": (
                "exactly two beta-4 LD_MMA evaluations on the canonical 81x81 "
                "Lumerical density carrier; development smoke only"
            ),
            "git_commit": _git_commit(Path(__file__).resolve().parents[3]),
            "optimizer": {
                "library": "NLopt",
                "version": nlopt.__version__,
                "algorithm": "LD_MMA",
                "requested_maxeval": SMOKE_MAXEVAL,
                "reported_numevals": optimizer.get_numevals(),
                "result_code": optimizer.last_optimize_result(),
                "initial_step_latent": 0.01,
                "initial_step_epigraph_nA": 0.1,
                "projection_beta": runtime.beta,
                "unique_physics_evaluations": unique_evaluations,
                "topology_changed": topology_changed,
            },
            "objective": "maximize t subject to t-I_Ea<=0 and t+I_Eb<=0",
            "current_sign_target": "I_Ea > 0 and I_Eb < 0",
            "initial": {
                "epigraph_nA": epigraph_initial_nA,
                "currents_nA": {
                    key: 1.0e9 * float(value)
                    for key, value in initial_physics["currents_A"].items()
                },
            },
            "final": {
                "epigraph_nA": float(vector_final[-1]),
                "currents_nA": {
                    "Ea": 1.0e9 * float(final_point["current_a_A"]),
                    "Eb": 1.0e9 * float(final_point["current_b_A"]),
                },
                "balanced_utility_nA": 1.0e9
                * float(final_point["balanced_utility_A"]),
                "opposite_current_switching_achieved": bool(
                    final_point["current_a_A"] > 0.0
                    and final_point["current_b_A"] < 0.0
                ),
                "DFM_values": np.asarray(final_point["DFM_values"]).tolist(),
                "binary_candidate_exact_500nm_audit": {
                    key: value
                    for key, value in binary_audit.items()
                    if key not in {"binary", "bad_solid", "bad_void"}
                },
                "binary_candidate_is_not_promoted": True,
                "ordinary_dispersive_Au_binary_reevaluation_required": True,
            },
            "smoke_preflight": artifact(output / "smoke_preflight.json"),
            "evaluation_history": driver.history,
            "callback_history": problem.callback_history,
            "gates": gates,
            "artifacts": {"final_state": artifact(final_npz)},
            "Lumerical_Maxwell_solves": {
                "forward": 2 * unique_evaluations,
                "adjoint": 2 * unique_evaluations,
            },
            "Lumerical_HEAT_or_CHARGE_solves": 0,
            "FDTDX_Maxwell_solves": 0,
            "production_optimization_enabled": False,
            "wall_s": time.monotonic() - started,
        }
    except Exception as error:
        result.update(
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            wall_s=time.monotonic() - started,
        )
    if output is None:
        fallback = Path(os.environ.get("EIDL_RUN_DIR", ".")).resolve()
        output = fallback
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "lumerical_dualpol_smoke_result.json", result)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
