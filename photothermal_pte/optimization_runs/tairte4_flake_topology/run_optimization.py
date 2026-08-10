#!/usr/bin/env python3
"""Restartable objective-led beta continuation for TaIrTe4 flake topology."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.optimization_support import (
    MAPPING,
    metrics,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT


HERE = Path(__file__).resolve().parent
EVALUATOR = HERE / "evaluate_objective_gradient.py"
BASE_FSP = Path("/data/seunghyun/tairte4/artifacts/tairte4_flake_topology/run010_uniform_rho0p5_Ea_forward_gpu5_retry2_20260810/tairte4_flake_forward_Ea.fsp")
BASE_SHA = "e4db45cd511965988bc5835e018042cb4f671de66fbd197d77f27b718868f12e"
JACOBIAN = Path("/data/seunghyun/tairte4/artifacts/tairte4_flake_topology/run010_component_yee_jacobian_20260810")
INITIAL_COMBINED = Path("/data/seunghyun/tairte4/artifacts/tairte4_flake_topology/run010_combined_adfd_Ea_gpu5_20260810/tairte4_flake_combined_adfd.npz")
INITIAL_COMBINED_RESULT = Path("/data/seunghyun/tairte4/artifacts/tairte4_flake_topology/run010_combined_adfd_Ea_gpu5_20260810/tairte4_flake_combined_adfd.json")

BETA_SCHEDULE = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
MIN_UPDATES = {2.0: 5, 4.0: 3, 8.0: 3, 16.0: 3, 32.0: 2, 64.0: 2}
SAFETY_MAX_UPDATES = {2.0: 8, 4.0: 6, 8.0: 5, 16.0: 4, 32.0: 4, 64.0: 3}
INITIAL_MOVE = {2.0: 0.020, 4.0: 0.018, 8.0: 0.015, 16.0: 0.012, 32.0: 0.010, 64.0: 0.0075}
MORPHOLOGY_WEIGHT = {2.0: 0.0, 4.0: 0.05, 8.0: 0.15, 16.0: 0.35, 32.0: 0.70, 64.0: 1.20}
SIGMA_A_S_M = 4.91e5
FULL_SOLID_TERMINAL_CONDUCTANCE_S = (
    SIGMA_A_S_M * 100.0e-9 * 24.0e-6 / 24.0e-6
)
REFERENCE_INCIDENT_POWER_W = 285.0e-6


def objective_power_fields(result: dict, objective_A: float) -> dict:
    source_power_W = float(result["forward"]["source_power_W"])
    if source_power_W <= 0.0:
        raise ValueError(f"non-positive simulated source power: {source_power_W}")
    return {
        "simulated_source_power_W": source_power_W,
        "reference_incident_power_W": REFERENCE_INCIDENT_POWER_W,
        "objective_at_reference_power_A": (
            float(objective_A) * REFERENCE_INCIDENT_POWER_W / source_power_W
        ),
        "responsivity_A_W": float(objective_A) / source_power_W,
    }


def emit(path: Path, event: str, **values) -> None:
    row = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "event": event, **values}
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def normalized(value: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(value)))
    return np.zeros_like(value) if scale == 0.0 else value / scale


def save_density(path: Path, rho: np.ndarray) -> None:
    np.savez_compressed(path, rho=np.asarray(rho, dtype=np.float64))


def evaluate(
    rho: np.ndarray,
    *,
    polarization: str,
    output: Path,
    gpu: int,
    events: Path,
    base_fsp: Path,
    base_sha: str,
    jacobian: Path,
) -> tuple[dict, np.ndarray, np.ndarray]:
    density = output.parent / f"{output.name}_rho.npz"
    save_density(density, rho)
    command = [
        sys.executable,
        "-m",
        "photothermal_pte.optimization_runs.tairte4_flake_topology.evaluate_objective_gradient",
        "--base-fsp", str(base_fsp),
        "--base-sha256", base_sha,
        "--jacobian-dir", str(jacobian),
        "--rho-npz", str(density),
        "--output-dir", str(output),
        "--polarization", polarization,
        "--gpu-device", f"GPU {gpu}",
        "--cuda-device", "0",
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    emit(events, "evaluation_start", output=str(output), command=command)
    completed = subprocess.run(command, cwd=HERE.parents[2], env=environment)
    result_path = output / "objective_gradient_result.json"
    if not result_path.is_file():
        raise RuntimeError(f"evaluation produced no result: {output}")
    result = json.loads(result_path.read_text())
    emit(events, "evaluation_end", output=str(output), returncode=completed.returncode, status=result.get("status"))
    if completed.returncode or not result.get("passed"):
        raise RuntimeError(f"solver evaluation failed closed: {result_path}")
    with np.load(Path(result["raw_artifact"]["path"])) as raw:
        objective_gradient = np.asarray(raw["gradient_total_A"], dtype=np.float64)
        conductance_gradient = np.asarray(
            raw["gradient_terminal_conductance_S"], dtype=np.float64
        )
    return result, objective_gradient, conductance_gradient


def publish_plot(
    published: Path,
    history: list[dict],
    rho: np.ndarray,
    gradient: np.ndarray,
    summary: dict,
    *,
    evaluation_id: int,
    accepted: bool,
    label: str,
    publish_latest: bool | None = None,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    extent = (
        1e6 * CONTRACT.design_bounds_m["x"][0],
        1e6 * CONTRACT.design_bounds_m["x"][1],
        1e6 * CONTRACT.design_bounds_m["y"][0],
        1e6 * CONTRACT.design_bounds_m["y"][1],
    )
    image = axes[0, 0].imshow(
        rho.T, origin="lower", extent=extent, vmin=0, vmax=1,
        cmap="gray_r", interpolation="nearest",
    )
    axes[0, 0].set_title("physical density: black=TaIrTe4 (1), white=void (0)")
    axes[0, 0].set_xlabel("Lumerical x=b (um)")
    axes[0, 0].set_ylabel("Lumerical y=a (um)")
    fig.colorbar(image, ax=axes[0, 0])
    axes[0, 1].hist(rho.ravel(), bins=40, range=(0, 1))
    axes[0, 1].set_title("physical-density histogram")
    grad_image = axes[0, 2].imshow(gradient.T, origin="lower", extent=extent, cmap="coolwarm")
    axes[0, 2].set_title("physical objective gradient (A)")
    fig.colorbar(grad_image, ax=axes[0, 2])
    accepted_rows = [row for row in history if row.get("accepted")]
    iteration = [row.get("accepted_update_index", row["global_iteration"]) for row in accepted_rows]
    has_reference_current = all(
        "objective_at_reference_power_A" in row for row in accepted_rows
    )
    if has_reference_current:
        plotted_objective = [
            1.0e9 * row["objective_at_reference_power_A"] for row in accepted_rows
        ]
        objective_ylabel = "equivalent current at 285 uW (nA)"
        objective_title = "accepted iteration vs 285 uW-equivalent PTE current"
    else:
        plotted_objective = [row["objective_A"] for row in accepted_rows]
        objective_ylabel = "raw current at simulated source power (A)"
        objective_title = "accepted iteration vs raw signed PTE current"
    axes[1, 0].plot(iteration, plotted_objective, marker="o")
    axes[1, 0].set_title(objective_title)
    axes[1, 0].set_xlabel("accepted update")
    axes[1, 0].set_ylabel(objective_ylabel)
    conductance_axis = axes[1, 0].twinx()
    conductance_axis.plot(
        iteration,
        [row["terminal_conductance_S"] / row["minimum_terminal_conductance_S"] for row in accepted_rows],
        color="tab:green", marker="s", alpha=0.65,
    )
    conductance_axis.axhline(1.0, color="tab:green", linestyle="--", linewidth=0.8)
    conductance_axis.set_ylabel("Gterminal / Gmin", color="tab:green")
    axes[1, 1].plot(iteration, [row["gray_fraction"] for row in accepted_rows], marker="o", label="gray fraction")
    axes[1, 1].plot(iteration, [row.get("binarization", np.nan) for row in accepted_rows], marker="s", label="4rho(1-rho)")
    beta_axis = axes[1, 1].twinx()
    beta_axis.step(iteration, [row["beta"] for row in accepted_rows], where="post", color="black", alpha=0.5, label="beta")
    axes[1, 1].set_title("binarization and beta")
    axes[1, 1].legend(loc="upper left")
    beta_axis.set_ylabel("beta")
    axes[1, 2].semilogy(iteration, [max(row.get("smooth_constraint", np.nan), 1e-12) for row in accepted_rows], marker="o", label="smooth solid+void")
    bad_axis = axes[1, 2].twinx()
    bad_axis.plot(iteration, [row["exact_bad_cells"] for row in accepted_rows], color="tab:red", marker="s", label="exact bad cells")
    axes[1, 2].set_title("500 nm constraints")
    axes[1, 2].set_ylabel("smooth residual")
    bad_axis.set_ylabel("bad cells")
    objective = float(history[-1]["objective_A"])
    reference_text = ""
    if "objective_at_reference_power_A" in history[-1]:
        reference_text = (
            f", I(285uW)={1.0e9 * history[-1]['objective_at_reference_power_A']:.3f} nA"
        )
    fig.suptitle(
        f"evaluation={evaluation_id}, {label}, accepted={accepted}; "
        f"beta={summary['beta']:g}, raw I={objective:.5e} A{reference_text}, "
        f"gray={summary['gray_fraction_0p01_0p99']:.3f}, "
        f"bad={summary['exact']['total_bad_cell_count']}"
    )
    destination = published / f"evaluation_{evaluation_id:04d}_{label}.png"
    fig.savefig(destination, dpi=140)
    write_latest = accepted if publish_latest is None else publish_latest
    if write_latest:
        fig.savefig(published / "latest_iteration.png", dpi=170)
    plt.close(fig)


def save_state(path: Path, **arrays) -> None:
    np.savez_compressed(path, **arrays)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--polarization", choices=("Ea", "Eb"), required=True)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--published-dir", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--base-fsp", type=Path, default=BASE_FSP)
    parser.add_argument("--base-sha256", default=BASE_SHA)
    parser.add_argument("--jacobian-dir", type=Path, default=JACOBIAN)
    parser.add_argument("--connectivity-fraction", type=float, default=0.10)
    parser.add_argument("--max-new-evaluations", type=int, default=0, help="0 means no pilot limit")
    args = parser.parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    published = args.published_dir.expanduser().resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    published.mkdir(parents=True, exist_ok=True)
    events = raw_root / "events.jsonl"
    state_path = raw_root / "optimization_state.npz"
    history_path = raw_root / "history.json"
    new_evaluations = 0
    base_fsp = args.base_fsp.expanduser().resolve()
    jacobian = args.jacobian_dir.expanduser().resolve()
    if not 0.0 < args.connectivity_fraction < 1.0:
        raise ValueError("connectivity fraction must lie in (0,1)")
    minimum_conductance = args.connectivity_fraction * FULL_SOLID_TERMINAL_CONDUCTANCE_S

    if state_path.is_file():
        state = np.load(state_path)
        latent = np.asarray(state["latent"], float)
        gradient_physical = np.asarray(state["gradient_physical_A"], float)
        gradient_conductance = np.asarray(state["gradient_terminal_conductance_S"], float)
        terminal_conductance = float(state["terminal_conductance_S"])
        objective = float(state["objective_A"])
        beta_index = int(state["beta_index"])
        stage_updates = int(state["stage_updates"])
        global_iteration = int(state["global_iteration"])
        first_moment = np.asarray(state["first_moment"], float)
        second_moment = np.asarray(state["second_moment"], float)
        adam_iteration = int(state["adam_iteration"])
        move = float(state["move"])
        evaluation_counter = int(state["evaluation_counter"])
        history = json.loads(history_path.read_text())
        emit(events, "resume", global_iteration=global_iteration, beta=BETA_SCHEDULE[beta_index])
    else:
        if CONTRACT.geometry_mode == "fixed_frame" and args.polarization == "Ea":
            initial = np.load(INITIAL_COMBINED)
            latent = np.full(MAPPING.shape, 0.5)
            gradient_physical = np.asarray(initial["gradient_total_A"], float)
            objective = float(json.loads(INITIAL_COMBINED_RESULT.read_text())["base_objective_A"])
            # Run010 predates the differentiable conductance safeguard.
            rho = MAPPING.physical(latent, BETA_SCHEDULE[0])
            result, gradient_physical, gradient_conductance = evaluate(
                rho, polarization=args.polarization,
                output=raw_root / "evaluation_initial_connectivity",
                gpu=args.gpu, events=events, base_fsp=base_fsp,
                base_sha=args.base_sha256, jacobian=jacobian,
            )
            objective = float(result["objective_A"])
            terminal_conductance = float(result["terminal_conductance_S"])
            new_evaluations += 1
        else:
            latent = np.full(MAPPING.shape, 0.5)
            rho = MAPPING.physical(latent, BETA_SCHEDULE[0])
            result, gradient_physical, gradient_conductance = evaluate(
                rho,
                polarization=args.polarization,
                output=raw_root / "evaluation_initial",
                gpu=args.gpu,
                events=events,
                base_fsp=base_fsp,
                base_sha=args.base_sha256,
                jacobian=jacobian,
            )
            objective = float(result["objective_A"])
            terminal_conductance = float(result["terminal_conductance_S"])
            new_evaluations += 1
        beta_index = 0
        stage_updates = 0
        global_iteration = 0
        first_moment = np.zeros(MAPPING.shape)
        second_moment = np.zeros(MAPPING.shape)
        adam_iteration = 0
        move = INITIAL_MOVE[BETA_SCHEDULE[0]]
        evaluation_counter = new_evaluations
        summary, _ = metrics(latent, BETA_SCHEDULE[0])
        history = [{
            "accepted": True,
            "global_iteration": 0,
            "stage_update": 0,
            "beta": BETA_SCHEDULE[0],
            "objective_A": objective,
            "gray_fraction": summary["gray_fraction_0p01_0p99"],
            "binarization": summary["binarization_mean_4rho1mrho"],
            "smooth_constraint": summary["smooth_solid_constraint"] + summary["smooth_void_constraint"],
            "exact_bad_cells": summary["exact"]["total_bad_cell_count"],
            "initial_uniform": True,
            "evaluation_id": evaluation_counter,
            "accepted_update_index": 0,
            "terminal_conductance_S": terminal_conductance,
            "minimum_terminal_conductance_S": minimum_conductance,
            **objective_power_fields(result, objective),
        }]
        publish_plot(
            published,
            history,
            rho,
            gradient_physical,
            summary,
            evaluation_id=evaluation_counter,
            accepted=True,
            label="initial",
        )

    while beta_index < len(BETA_SCHEDULE):
        beta = BETA_SCHEDULE[beta_index]
        summary, arrays = metrics(latent, beta)
        rho = arrays["rho"]
        gradient_latent = MAPPING.vjp(latent, gradient_physical, beta)
        constraint_gradient = np.sum(arrays["constraint_gradients"], axis=0)
        combined = normalized(gradient_latent) - MORPHOLOGY_WEIGHT[beta] * normalized(constraint_gradient)
        conductance_margin = (
            terminal_conductance - minimum_conductance
        ) / FULL_SOLID_TERMINAL_CONDUCTANCE_S
        if conductance_margin < 0.05:
            connectivity_weight = min(1.0, max(0.0, (0.05 - conductance_margin) / 0.05))
            combined += connectivity_weight * normalized(
                MAPPING.vjp(latent, gradient_conductance, beta)
            )
        adam_iteration += 1
        first_moment = 0.9 * first_moment + 0.1 * combined
        second_moment = 0.999 * second_moment + 0.001 * combined * combined
        mhat = first_moment / (1.0 - 0.9**adam_iteration)
        vhat = second_moment / (1.0 - 0.999**adam_iteration)
        direction = normalized(mhat / (np.sqrt(vhat) + 1.0e-8))
        candidate = np.clip(latent + move * direction, 0.0, 1.0)
        candidate_rho = MAPPING.physical(candidate, beta)
        evaluation_counter += 1
        evaluation_id = evaluation_counter
        result, candidate_gradient, candidate_conductance_gradient = evaluate(
            candidate_rho,
            polarization=args.polarization,
            output=raw_root / f"evaluation_{evaluation_id:04d}_beta{int(beta)}",
            gpu=args.gpu,
            events=events,
            base_fsp=base_fsp,
            base_sha=args.base_sha256,
            jacobian=jacobian,
        )
        new_evaluations += 1
        candidate_objective = float(result["objective_A"])
        candidate_conductance = float(result["terminal_conductance_S"])
        candidate_summary, _ = metrics(candidate, beta)
        current_bad = int(summary["exact"]["total_bad_cell_count"])
        candidate_bad = int(candidate_summary["exact"]["total_bad_cell_count"])
        objective_guard = candidate_objective >= objective - 0.002 * max(abs(objective), 1.0e-22)
        constraint_improved = (
            candidate_summary["smooth_solid_constraint"] + candidate_summary["smooth_void_constraint"]
            < summary["smooth_solid_constraint"] + summary["smooth_void_constraint"]
        )
        connectivity_feasible = candidate_conductance >= minimum_conductance
        accepted = bool(
            connectivity_feasible
            and (objective_guard if beta <= 8 else (objective_guard or constraint_improved))
        )
        history.append({
            "accepted": accepted,
            "global_iteration": global_iteration + (1 if accepted else 0),
            "evaluation_id": evaluation_id,
            "accepted_update_index": global_iteration + (1 if accepted else 0),
            "stage_update": stage_updates + (1 if accepted else 0),
            "beta": beta,
            "move": move,
            "objective_A": candidate_objective,
            "objective_change_A": candidate_objective - objective,
            "gray_fraction": candidate_summary["gray_fraction_0p01_0p99"],
            "binarization": candidate_summary["binarization_mean_4rho1mrho"],
            "smooth_constraint": candidate_summary["smooth_solid_constraint"] + candidate_summary["smooth_void_constraint"],
            "exact_bad_cells": candidate_bad,
            "previous_exact_bad_cells": current_bad,
            "smooth_constraint_improved": constraint_improved,
            "terminal_conductance_S": candidate_conductance,
            "minimum_terminal_conductance_S": minimum_conductance,
            "connectivity_feasible": connectivity_feasible,
            "solver_result": str(raw_root / f"evaluation_{evaluation_id:04d}_beta{int(beta)}" / "objective_gradient_result.json"),
            **objective_power_fields(result, candidate_objective),
        })
        history_path.write_text(json.dumps(history, indent=2) + "\n")
        publish_plot(
            published,
            history,
            candidate_rho,
            candidate_gradient,
            candidate_summary,
            evaluation_id=evaluation_id,
            accepted=accepted,
            label="candidate",
        )
        if accepted:
            latent = candidate
            objective = candidate_objective
            gradient_physical = candidate_gradient
            gradient_conductance = candidate_conductance_gradient
            terminal_conductance = candidate_conductance
            global_iteration += 1
            stage_updates += 1
            move = min(INITIAL_MOVE[beta], move * 1.05)
            (published / "latest_summary.json").write_text(json.dumps(candidate_summary, indent=2) + "\n")
        else:
            move *= 0.5
            emit(events, "candidate_rejected", beta=beta, move=move, objective_A=candidate_objective)
            if move < 0.0025:
                emit(events, "stage_move_floor", beta=beta)
                stage_updates = SAFETY_MAX_UPDATES[beta]

        save_state(
            state_path,
            latent=latent,
            gradient_physical_A=gradient_physical,
            gradient_terminal_conductance_S=gradient_conductance,
            terminal_conductance_S=np.asarray(terminal_conductance),
            objective_A=np.asarray(objective),
            beta_index=np.asarray(beta_index),
            stage_updates=np.asarray(stage_updates),
            global_iteration=np.asarray(global_iteration),
            first_moment=first_moment,
            second_moment=second_moment,
            adam_iteration=np.asarray(adam_iteration),
            move=np.asarray(move),
            evaluation_counter=np.asarray(evaluation_counter),
        )
        emit(events, "checkpoint", beta=beta, stage_updates=stage_updates, global_iteration=global_iteration, objective_A=objective)
        if args.max_new_evaluations and new_evaluations >= args.max_new_evaluations:
            emit(events, "pilot_limit_reached", new_evaluations=new_evaluations)
            return 0

        accepted_stage = [row for row in history if row.get("accepted") and row.get("beta") == beta and row.get("global_iteration", 0) > 0]
        plateau = False
        if stage_updates >= MIN_UPDATES[beta] and len(accepted_stage) >= 3:
            recent = accepted_stage[-3:]
            gains = [
                abs(recent[i]["objective_A"] - recent[i-1]["objective_A"])
                / max(abs(recent[i]["objective_A"]), abs(recent[i-1]["objective_A"]), 1.0e-30)
                for i in range(1, len(recent))
            ]
            plateau = max(gains) < 2.0e-3
        if plateau or stage_updates >= SAFETY_MAX_UPDATES[beta]:
            emit(events, "beta_advance", beta=beta, reason="objective_plateau" if plateau else "bounded_stage_budget")
            beta_index += 1
            stage_updates = 0
            first_moment.fill(0.0)
            second_moment.fill(0.0)
            adam_iteration = 0
            if beta_index < len(BETA_SCHEDULE):
                move = INITIAL_MOVE[BETA_SCHEDULE[beta_index]]
            save_state(
                state_path,
                latent=latent,
                gradient_physical_A=gradient_physical,
                gradient_terminal_conductance_S=gradient_conductance,
                terminal_conductance_S=np.asarray(terminal_conductance),
                objective_A=np.asarray(objective),
                beta_index=np.asarray(beta_index),
                stage_updates=np.asarray(stage_updates),
                global_iteration=np.asarray(global_iteration),
                first_moment=first_moment,
                second_moment=second_moment,
                adam_iteration=np.asarray(adam_iteration),
                move=np.asarray(move),
                evaluation_counter=np.asarray(evaluation_counter),
            )
            if beta_index < len(BETA_SCHEDULE):
                next_beta = BETA_SCHEDULE[beta_index]
                evaluation_counter += 1
                reprojected_rho = MAPPING.physical(latent, next_beta)
                reprojected_result, gradient_physical, gradient_conductance = evaluate(
                    reprojected_rho,
                    polarization=args.polarization,
                    output=raw_root / f"evaluation_{evaluation_counter:04d}_beta{int(next_beta)}_reprojection",
                    gpu=args.gpu,
                    events=events,
                    base_fsp=base_fsp,
                    base_sha=args.base_sha256,
                    jacobian=jacobian,
                )
                new_evaluations += 1
                objective = float(reprojected_result["objective_A"])
                terminal_conductance = float(reprojected_result["terminal_conductance_S"])
                reprojected_summary, _ = metrics(latent, next_beta)
                history.append({
                    "accepted": True,
                    "stage_reprojection": True,
                    "global_iteration": global_iteration,
                    "accepted_update_index": global_iteration,
                    "evaluation_id": evaluation_counter,
                    "stage_update": 0,
                    "beta": next_beta,
                    "objective_A": objective,
                    "gray_fraction": reprojected_summary["gray_fraction_0p01_0p99"],
                    "binarization": reprojected_summary["binarization_mean_4rho1mrho"],
                    "smooth_constraint": reprojected_summary["smooth_solid_constraint"] + reprojected_summary["smooth_void_constraint"],
                    "exact_bad_cells": reprojected_summary["exact"]["total_bad_cell_count"],
                    "terminal_conductance_S": terminal_conductance,
                    "minimum_terminal_conductance_S": minimum_conductance,
                    "solver_result": str(
                        raw_root / f"evaluation_{evaluation_counter:04d}_beta{int(next_beta)}_reprojection" / "objective_gradient_result.json"
                    ),
                    **objective_power_fields(reprojected_result, objective),
                })
                history_path.write_text(json.dumps(history, indent=2) + "\n")
                publish_plot(
                    published,
                    history,
                    reprojected_rho,
                    gradient_physical,
                    reprojected_summary,
                    evaluation_id=evaluation_counter,
                    accepted=True,
                    label="reprojection",
                )
                save_state(
                    state_path,
                    latent=latent,
                    gradient_physical_A=gradient_physical,
                    gradient_terminal_conductance_S=gradient_conductance,
                    terminal_conductance_S=np.asarray(terminal_conductance),
                    objective_A=np.asarray(objective),
                    beta_index=np.asarray(beta_index),
                    stage_updates=np.asarray(stage_updates),
                    global_iteration=np.asarray(global_iteration),
                    first_moment=first_moment,
                    second_moment=second_moment,
                    adam_iteration=np.asarray(adam_iteration),
                    move=np.asarray(move),
                    evaluation_counter=np.asarray(evaluation_counter),
                )
                emit(events, "stage_reprojection_evaluated", beta=next_beta, objective_A=objective)

    emit(events, "continuous_continuation_complete", objective_A=objective, global_iteration=global_iteration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
