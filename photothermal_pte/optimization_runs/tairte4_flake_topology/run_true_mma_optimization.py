#!/usr/bin/env python3
"""Fresh uniform-density contact-anchored PTE optimization with true MMA.

Historical Run014 used an Adam-like normalized direction and a fixed clipped
increment.  This driver deliberately contains no Adam state, no normalized
gradient sum and no post-update clipping.  Every accepted design is the
solution of a persistent MMA subproblem with explicit inequalities.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT
from photothermal_pte.optimization_runs.tairte4_flake_topology.optimization_support import (
    MAPPING,
    exact_binary_audit,
    metrics,
)


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
EVALUATOR = HERE / "evaluate_objective_gradient.py"
FINAL_EVALUATOR = HERE / "evaluate_binary_objective.py"
REFERENCE_INCIDENT_POWER_W = 285.0e-6
OBJECTIVE_SCALE_AT_REFERENCE_POWER_A = 10.0e-9
SIGMA_A_S_M = 4.91e5
FULL_SOLID_TERMINAL_CONDUCTANCE_S = SIGMA_A_S_M * CONTRACT.flake_thickness_m
BETA_SCHEDULE = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
MORPHOLOGY_START_BETA = 8.0
MORPHOLOGY_TARGET_CAP = {
    8.0: 2.0e-3,
    16.0: 1.0e-3,
    32.0: 5.0e-4,
    64.0: 2.0e-4,
    128.0: 1.0e-4,
}
MOVE_LIMIT = {
    # MMA trust-region bounds, not fixed update sizes. Run020 showed that
    # unbounded LD_MMA expanded a 1.4e-4 trial step to 0.5 at beta=1.
    1.0: 0.025,
    2.0: 0.020,
    4.0: 0.018,
    8.0: 0.015,
    16.0: 0.012,
    32.0: 0.010,
    64.0: 0.008,
    128.0: 0.006,
}
MINIMUM_STAGE_UPDATES = {
    1.0: 6,
    2.0: 5,
    4.0: 5,
    8.0: 5,
    16.0: 5,
    32.0: 5,
    64.0: 5,
    128.0: 8,
}
PLATEAU_WINDOW = 4
PLATEAU_RELATIVE_CHANGE = 2.0e-3
STATIONARY_MAX_STEP = 1.5e-3
MAXIMUM_STAGE_UPDATES = {
    1.0: 16,
    2.0: 14,
    4.0: 14,
    8.0: 16,
    16.0: 18,
    32.0: 20,
    64.0: 24,
    128.0: 40,
}
BINARIZATION_CONTINUATION_TARGET = {
    1.0: 0.86,
    2.0: 0.68,
    4.0: 0.46,
    8.0: 0.28,
    16.0: 0.14,
    32.0: 0.060,
    64.0: 0.020,
    128.0: 0.005,
}
PILOT_ACCEPTED_UPDATES = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def emit(path: Path, event: str, **values: object) -> None:
    row = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **values,
    }
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def verify_file(path: Path, expected: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or sha256(resolved) != expected:
        raise RuntimeError(f"missing or SHA-mismatched immutable input: {resolved}")
    return resolved


def evaluate(
    rho: np.ndarray,
    *,
    polarization: str,
    output: Path,
    gpu: int,
    events: Path,
    base_fsp: Path,
    base_sha256: str,
    jacobian_dir: Path,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    result_path = output / "objective_gradient_result.json"
    if result_path.is_file():
        previous = json.loads(result_path.read_text())
        if previous.get("passed"):
            raw_path = Path(previous["raw_artifact"]["path"])
            if sha256(raw_path) != previous["raw_artifact"]["sha256"]:
                raise RuntimeError("existing objective-gradient raw SHA mismatch")
            with np.load(raw_path) as raw:
                return (
                    previous,
                    np.asarray(raw["gradient_total_A"], dtype=np.float64),
                    np.asarray(raw["gradient_terminal_conductance_S"], dtype=np.float64),
                )
    if output.exists() and any(output.iterdir()):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archived = output.with_name(f"{output.name}_incomplete_{stamp}")
        output.rename(archived)
        emit(events, "incomplete_evaluation_archived", output=str(output), archive=str(archived))
    density = output.with_name(output.name + "_rho.npz")
    np.savez_compressed(density, rho=np.asarray(rho, dtype=np.float64))
    command = [
        sys.executable,
        "-m",
        "photothermal_pte.optimization_runs.tairte4_flake_topology.evaluate_objective_gradient",
        "--base-fsp", str(base_fsp),
        "--base-sha256", base_sha256,
        "--jacobian-dir", str(jacobian_dir),
        "--rho-npz", str(density),
        "--output-dir", str(output),
        "--polarization", polarization,
        "--gpu-device", f"GPU {gpu}",
        "--cuda-device", "0",
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    emit(events, "evaluation_start", output=str(output), command=command)
    completed = subprocess.run(command, cwd=REPOSITORY, env=environment)
    result_path = output / "objective_gradient_result.json"
    if not result_path.is_file():
        raise RuntimeError(f"evaluation produced no result: {output}")
    result = json.loads(result_path.read_text())
    emit(
        events,
        "evaluation_end",
        output=str(output),
        returncode=completed.returncode,
        status=result.get("status"),
    )
    if completed.returncode or not result.get("passed"):
        raise RuntimeError(f"solver evaluation failed closed: {result_path}")
    raw_path = Path(result["raw_artifact"]["path"])
    if sha256(raw_path) != result["raw_artifact"]["sha256"]:
        raise RuntimeError("objective-gradient raw artifact SHA mismatch")
    with np.load(raw_path) as raw:
        objective_gradient = np.asarray(raw["gradient_total_A"], dtype=np.float64)
        conductance_gradient = np.asarray(
            raw["gradient_terminal_conductance_S"], dtype=np.float64
        )
    return result, objective_gradient, conductance_gradient


def equivalent_current(objective_A: float, fixed_source_power_W: float) -> float:
    return float(objective_A) * REFERENCE_INCIDENT_POWER_W / fixed_source_power_W


def stage_morphology_caps(values: np.ndarray, beta: float) -> np.ndarray:
    if beta < MORPHOLOGY_START_BETA:
        return np.asarray([np.inf, np.inf])
    target = MORPHOLOGY_TARGET_CAP[beta]
    # Each new beta asks for a modest ten-percent improvement from its own
    # reprojected baseline, while never relaxing the absolute continuation
    # target.  The values are fixed for the entire stage and recorded.
    return np.maximum(target, 0.90 * np.asarray(values, dtype=np.float64))


def canonical_constraints(
    *,
    latent: np.ndarray,
    beta: float,
    terminal_conductance_S: float,
    gradient_terminal_conductance_physical_S: np.ndarray,
    minimum_terminal_conductance_S: float | None,
    morphology_caps: np.ndarray,
    device: str,
    include_terminal_conductance_constraint: bool = True,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, object], dict[str, object]]:
    """Return differentiable constraints for a continuation stage.

    ``terminal_conductance_S`` is always evaluated and recorded because it is
    useful physical diagnostics for a top/bottom-electrode device.  It is only
    made an optimization constraint when the caller explicitly opts in.  This
    keeps historical connectivity-constrained runs reproducible while letting
    the pure-terminal-current LD_MMA driver optimize the stated objective
    without an arbitrary conductance floor.
    """
    summary, arrays = metrics(latent, beta, device=device)
    names: list[str] = []
    values: list[float] = []
    gradients: list[np.ndarray] = []
    if include_terminal_conductance_constraint:
        if minimum_terminal_conductance_S is None or minimum_terminal_conductance_S <= 0.0:
            raise ValueError("terminal-conductance constraint requires a positive floor")
        names.append("minimum_terminal_conductance")
        values.append(1.0 - terminal_conductance_S / minimum_terminal_conductance_S)
        gradients.append(
            -MAPPING.vjp(latent, gradient_terminal_conductance_physical_S, beta)
            / minimum_terminal_conductance_S
        )
    if beta >= MORPHOLOGY_START_BETA:
        raw_values = np.asarray(
            [summary["smooth_solid_constraint"], summary["smooth_void_constraint"]],
            dtype=np.float64,
        )
        raw_gradients = np.asarray(arrays["constraint_gradients"], dtype=np.float64)
        names.extend(("500nm_solid_opening", "500nm_void_opening"))
        values.extend((raw_values / morphology_caps - 1.0).tolist())
        gradients.extend(
            [raw_gradients[index] / morphology_caps[index] for index in range(2)]
        )
    return (
        names,
        np.asarray(values, dtype=np.float64),
        (
            np.stack(gradients)
            if gradients
            else np.empty((0, *MAPPING.shape), dtype=np.float64)
        ),
        summary,
        arrays,
    )


def stage_convergence(history: list[dict[str, object]], beta: float) -> dict[str, object]:
    rows = [
        row for row in history
        if row.get("role") == "accepted_mma" and float(row["beta"]) == beta
    ]
    minimum = MINIMUM_STAGE_UPDATES[beta]
    if len(rows) < max(minimum, PLATEAU_WINDOW + 1):
        return {"converged": False, "reason": "minimum_stage_updates", "count": len(rows)}
    recent = rows[-(PLATEAU_WINDOW + 1):]
    objectives = np.asarray(
        [row["objective_at_reference_power_A"] for row in recent], dtype=float
    )
    pair_changes = np.abs(np.diff(objectives)) / np.maximum.reduce((
        np.abs(objectives[:-1]),
        np.abs(objectives[1:]),
        np.full(PLATEAU_WINDOW, 1.0e-12),
    ))
    plateau = bool(np.max(pair_changes) < PLATEAU_RELATIVE_CHANGE)
    stationary = bool(float(rows[-1]["mma_maximum_absolute_step"]) < STATIONARY_MAX_STEP)
    feasible = bool(float(rows[-1]["maximum_constraint_value"]) <= 1.0e-3)
    sharpness = float(rows[-1]["binarization"])
    continuation_sharpness = bool(
        beta < BETA_SCHEDULE[-1]
        and sharpness <= BINARIZATION_CONTINUATION_TARGET[beta]
    )
    final_geometry = bool(
        beta < BETA_SCHEDULE[-1]
        or (
            int(rows[-1]["exact_bad_cells"]) == 0
            and float(rows[-1]["gray_fraction_0p01_0p99"]) < 0.01
        )
    )
    converged = bool(
        feasible
        and final_geometry
        and (plateau or stationary or continuation_sharpness)
    )
    return {
        "converged": converged,
        "reason": (
            "objective_plateau" if converged and plateau
            else "mma_stationarity" if converged and stationary
            else "continuation_sharpness" if converged and continuation_sharpness
            else "constraints_or_final_geometry_unresolved"
        ),
        "count": len(rows),
        "recent_max_relative_objective_change": float(np.max(pair_changes)),
        "latest_maximum_absolute_step": float(rows[-1]["mma_maximum_absolute_step"]),
        "latest_binarization": sharpness,
        "continuation_binarization_target": BINARIZATION_CONTINUATION_TARGET[beta],
        "canonical_constraints_feasible": feasible,
        "final_geometry_gate": final_geometry,
    }


def save_driver_state(
    path: Path,
    *,
    latent: np.ndarray,
    gradient_physical_A: np.ndarray,
    gradient_conductance_S: np.ndarray,
    objective_A: float,
    terminal_conductance_S: float,
    fixed_source_power_W: float,
    beta_index: int,
    stage_updates: int,
    global_update: int,
    evaluation_counter: int,
    morphology_caps: np.ndarray,
    mma_state: object,
) -> None:
    np.savez_compressed(
        path,
        latent=latent,
        gradient_physical_A=gradient_physical_A,
        gradient_terminal_conductance_S=gradient_conductance_S,
        objective_A=np.asarray(objective_A),
        terminal_conductance_S=np.asarray(terminal_conductance_S),
        fixed_source_power_W=np.asarray(fixed_source_power_W),
        beta_index=np.asarray(beta_index),
        stage_updates=np.asarray(stage_updates),
        global_update=np.asarray(global_update),
        evaluation_counter=np.asarray(evaluation_counter),
        morphology_caps=np.asarray(morphology_caps),
        mma_iteration=np.asarray(mma_state.iteration),
        mma_xold1=mma_state.xold1,
        mma_xold2=mma_state.xold2,
        mma_low=mma_state.low,
        mma_upp=mma_state.upp,
    )


def load_driver_state(path: Path) -> dict[str, object]:
    from photothermal_pte.optimization_runs.tairte4_flake_topology.mma import MMAState

    with np.load(path) as data:
        return {
            "latent": np.asarray(data["latent"], dtype=float),
            "gradient_physical_A": np.asarray(data["gradient_physical_A"], dtype=float),
            "gradient_terminal_conductance_S": np.asarray(
                data["gradient_terminal_conductance_S"], dtype=float
            ),
            "objective_A": float(data["objective_A"]),
            "terminal_conductance_S": float(data["terminal_conductance_S"]),
            "fixed_source_power_W": float(data["fixed_source_power_W"]),
            "beta_index": int(data["beta_index"]),
            "stage_updates": int(data["stage_updates"]),
            "global_update": int(data["global_update"]),
            "evaluation_counter": int(data["evaluation_counter"]),
            "morphology_caps": np.asarray(data["morphology_caps"], dtype=float),
            "mma_state": MMAState(
                iteration=int(data["mma_iteration"]),
                xold1=np.asarray(data["mma_xold1"], dtype=float),
                xold2=np.asarray(data["mma_xold2"], dtype=float),
                low=np.asarray(data["mma_low"], dtype=float),
                upp=np.asarray(data["mma_upp"], dtype=float),
            ),
        }


def publish_plot(
    published: Path,
    history: list[dict[str, object]],
    rho: np.ndarray,
    gradient: np.ndarray,
    summary: dict[str, object],
    *,
    evaluation_id: int,
    label: str,
) -> Path:
    rows = [
        row for row in history
        if row.get("role") in {"uniform_initial", "accepted_mma", "nlopt_evaluation"}
    ]
    indices = [int(row["global_update"]) for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    extent = (
        1e6 * CONTRACT.design_bounds_m["x"][0],
        1e6 * CONTRACT.design_bounds_m["x"][1],
        1e6 * CONTRACT.design_bounds_m["y"][0],
        1e6 * CONTRACT.design_bounds_m["y"][1],
    )
    image = axes[0, 0].imshow(
        rho.T,
        origin="lower",
        extent=extent,
        vmin=0.0,
        vmax=1.0,
        cmap="gray_r",
        interpolation="nearest",
    )
    axes[0, 0].set_title("physical density: black=TaIrTe4 (1), white=void (0)")
    axes[0, 0].set_xlabel("Lumerical x=b (um)")
    axes[0, 0].set_ylabel("Lumerical y=a (um)")
    fig.colorbar(image, ax=axes[0, 0])
    axes[0, 1].hist(rho.ravel(), bins=40, range=(0.0, 1.0))
    axes[0, 1].set_title("physical-density histogram")
    limit = float(np.max(np.abs(gradient)))
    gradient_image = axes[0, 2].imshow(
        gradient.T,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    axes[0, 2].set_title("physical objective gradient (A per rho)")
    fig.colorbar(gradient_image, ax=axes[0, 2])
    currents = [1e9 * float(row["objective_at_reference_power_A"]) for row in rows]
    axes[1, 0].plot(indices, currents, "o-", linewidth=2, label="signed PTE current")
    axes[1, 0].set_title("FOM history")
    axes[1, 0].set_xlabel("optimizer evaluation/update")
    axes[1, 0].set_ylabel("I at fixed 285 uW (nA)")
    axes[1, 0].grid(alpha=0.25)
    if rows:
        axes[1, 0].annotate(
            f"latest={currents[-1]:.4g} nA\nbest={max(currents):.4g} nA",
            (indices[-1], currents[-1]),
            xytext=(-8, -30),
            textcoords="offset points",
            ha="right",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
    axes[1, 1].plot(indices, [row["gray_fraction_0p01_0p99"] for row in rows], "o-", label="gray fraction")
    axes[1, 1].plot(indices, [row["binarization"] for row in rows], "s-", label="4rho(1-rho)")
    beta_axis = axes[1, 1].twinx()
    beta_axis.step(indices, [row["beta"] for row in rows], where="post", color="black", label="beta")
    axes[1, 1].set_title("binarization and beta")
    axes[1, 1].legend(loc="upper left")
    beta_axis.set_ylabel("beta")
    axes[1, 2].plot(indices, [row["exact_bad_cells"] for row in rows], "s-", color="tab:red", label="exact bad nodes")
    constraint_axis = axes[1, 2].twinx()
    constraint_axis.plot(indices, [row["maximum_constraint_value"] for row in rows], "o-", color="tab:blue", label="max g(x)")
    constraint_axis.axhline(0.0, color="tab:blue", linestyle="--", linewidth=0.8)
    axes[1, 2].set_title("500 nm audit and MMA inequalities")
    axes[1, 2].set_ylabel("exact bad design nodes")
    constraint_axis.set_ylabel("max canonical g(x), feasible <=0")
    last = history[-1]
    algorithm = str(last.get("algorithm", "initial design"))
    fig.suptitle(
        f"{label}; evaluation={evaluation_id}; {algorithm}; beta={last['beta']:g}; "
        f"I(285uW)={1e9*last['objective_at_reference_power_A']:.4g} nA; "
        f"gray={summary['gray_fraction_0p01_0p99']:.4f}; "
        f"bad={summary['exact']['total_bad_cell_count']}"
    )
    destination = published / f"evaluation_{evaluation_id:04d}_{label}.png"
    destination_temporary = destination.with_name(destination.stem + ".tmp.png")
    latest = published / "latest_iteration.png"
    latest_temporary = latest.with_name(latest.stem + ".tmp.png")
    fig.savefig(destination_temporary, dpi=150, format="png")
    destination_temporary.replace(destination)
    fig.savefig(latest_temporary, dpi=170, format="png")
    latest_temporary.replace(latest)
    plt.close(fig)
    return destination


def record_manifest_entry(result: dict[str, object]) -> dict[str, object]:
    result_path = Path(result["raw_artifact"]["path"]).parent / "objective_gradient_result.json"
    return {
        "result": {
            "path": str(result_path),
            "size_bytes": result_path.stat().st_size,
            "sha256": sha256(result_path),
        },
        "raw_artifact": result["raw_artifact"],
        "forward_FSP": result["forward"]["project"],
    }


def main() -> int:
    from photothermal_pte.optimization_runs.tairte4_flake_topology.mma import (
        initialize_mma_state,
        mma_step,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--polarization", choices=("Ea", "Eb"), required=True)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--published-dir", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--base-fsp", required=True, type=Path)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--jacobian-dir", required=True, type=Path)
    parser.add_argument("--connectivity-fraction", type=float, default=0.10)
    parser.add_argument("--constraint-device", default="cuda:0")
    parser.add_argument("--max-new-evaluations", type=int, default=0)
    args = parser.parse_args()
    if CONTRACT.geometry_mode != "contact_anchored":
        raise RuntimeError("true-MMA production requires contact_anchored geometry")
    CONTRACT.validate()
    if not 0.0 < args.connectivity_fraction < 1.0:
        raise ValueError("connectivity fraction must lie in (0,1)")
    base_fsp = verify_file(args.base_fsp, args.base_sha256)
    jacobian_dir = args.jacobian_dir.expanduser().resolve()
    jacobian_result = jacobian_dir / "component_yee_jacobian_result.json"
    if not jacobian_result.is_file():
        raise RuntimeError(f"missing Jacobian certificate: {jacobian_result}")
    jacobian_payload = json.loads(jacobian_result.read_text())
    if not jacobian_payload.get("passed"):
        raise RuntimeError("component-Yee Jacobian certificate is not passed")

    raw_root = args.raw_root.expanduser().resolve()
    published = args.published_dir.expanduser().resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    published.mkdir(parents=True, exist_ok=True)
    events = raw_root / "events.jsonl"
    state_path = raw_root / "optimization_state.npz"
    history_path = raw_root / "history.json"
    manifest_path = published / "RAW_ARTIFACT_MANIFEST.json"
    minimum_conductance = args.connectivity_fraction * FULL_SOLID_TERMINAL_CONDUCTANCE_S
    new_evaluations = 0

    if state_path.is_file():
        state = load_driver_state(state_path)
        latent = state["latent"]
        gradient_physical = state["gradient_physical_A"]
        gradient_conductance = state["gradient_terminal_conductance_S"]
        objective = state["objective_A"]
        terminal_conductance = state["terminal_conductance_S"]
        fixed_source_power = state["fixed_source_power_W"]
        beta_index = state["beta_index"]
        stage_updates = state["stage_updates"]
        global_update = state["global_update"]
        evaluation_counter = state["evaluation_counter"]
        morphology_caps = state["morphology_caps"]
        mma_state = state["mma_state"]
        history = json.loads(history_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        emit(events, "resume", beta=BETA_SCHEDULE[beta_index], global_update=global_update)
    else:
        latent = np.full(MAPPING.shape, 0.5, dtype=np.float64)
        beta_index = 0
        beta = BETA_SCHEDULE[beta_index]
        rho = MAPPING.physical(latent, beta)
        if not np.all(rho == 0.5):
            raise RuntimeError("uniform latent=0.5 did not map to exact physical rho=0.5")
        evaluation_counter = 1
        result, gradient_physical, gradient_conductance = evaluate(
            rho,
            polarization=args.polarization,
            output=raw_root / "evaluation_0001_uniform_initial",
            gpu=args.gpu,
            events=events,
            base_fsp=base_fsp,
            base_sha256=args.base_sha256,
            jacobian_dir=jacobian_dir,
        )
        new_evaluations += 1
        objective = float(result["objective_A"])
        terminal_conductance = float(result["terminal_conductance_S"])
        fixed_source_power = float(result["forward"]["source_power_W"])
        stage_updates = 0
        global_update = 0
        mma_state = initialize_mma_state(latent)
        initial_summary, initial_arrays = metrics(latent, beta, device=args.constraint_device)
        initial_values = np.asarray([
            initial_summary["smooth_solid_constraint"],
            initial_summary["smooth_void_constraint"],
        ])
        morphology_caps = stage_morphology_caps(initial_values, beta)
        initial_constraint = 1.0 - terminal_conductance / minimum_conductance
        history = [{
            "role": "uniform_initial",
            "evaluation_id": evaluation_counter,
            "global_update": 0,
            "stage_update": 0,
            "beta": beta,
            "objective_A": objective,
            "objective_at_reference_power_A": equivalent_current(objective, fixed_source_power),
            "fixed_source_power_W": fixed_source_power,
            "source_power_relative_change": 0.0,
            "terminal_conductance_S": terminal_conductance,
            "minimum_terminal_conductance_S": minimum_conductance,
            "constraint_names": ["minimum_terminal_conductance"],
            "constraint_values": [initial_constraint],
            "maximum_constraint_value": initial_constraint,
            "gray_fraction_0p01_0p99": initial_summary["gray_fraction_0p01_0p99"],
            "binarization": initial_summary["binarization_mean_4rho1mrho"],
            "exact_bad_cells": initial_summary["exact"]["total_bad_cell_count"],
            "rho_mean": initial_summary["rho_mean"],
            "initial_density": "exact uniform rho=0.5",
            "symmetry_constraint": False,
            "volume_constraint": False,
        }]
        manifest = {
            "schema": "true-mma-contact-anchored-raw-artifact-manifest-v1",
            "raw_artifacts_committed_to_git": False,
            "base_FSP": {
                "path": str(base_fsp),
                "size_bytes": base_fsp.stat().st_size,
                "sha256": args.base_sha256,
            },
            "component_Yee_Jacobian": {
                "path": str(jacobian_dir),
                "certificate": str(jacobian_result),
                "certificate_sha256": sha256(jacobian_result),
            },
            "evaluations": {"0001": record_manifest_entry(result)},
        }
        write_json(history_path, history)
        write_json(manifest_path, manifest)
        write_json(published / "latest_summary.json", initial_summary)
        publish_plot(
            published,
            history,
            rho,
            gradient_physical,
            initial_summary,
            evaluation_id=evaluation_counter,
            label="uniform_initial",
        )
        save_driver_state(
            state_path,
            latent=latent,
            gradient_physical_A=gradient_physical,
            gradient_conductance_S=gradient_conductance,
            objective_A=objective,
            terminal_conductance_S=terminal_conductance,
            fixed_source_power_W=fixed_source_power,
            beta_index=beta_index,
            stage_updates=stage_updates,
            global_update=global_update,
            evaluation_counter=evaluation_counter,
            morphology_caps=morphology_caps,
            mma_state=mma_state,
        )

    while beta_index < len(BETA_SCHEDULE):
        beta = BETA_SCHEDULE[beta_index]
        names, fval, dfdx, summary, arrays = canonical_constraints(
            latent=latent,
            beta=beta,
            terminal_conductance_S=terminal_conductance,
            gradient_terminal_conductance_physical_S=gradient_conductance,
            minimum_terminal_conductance_S=minimum_conductance,
            morphology_caps=morphology_caps,
            device=args.constraint_device,
        )
        convergence = stage_convergence(history, beta)
        if convergence["converged"]:
            emit(events, "beta_stage_converged", beta=beta, diagnostics=convergence)
            if beta_index == len(BETA_SCHEDULE) - 1:
                rho = arrays["rho"]
                exact, _ = exact_binary_audit(rho)
                if exact["total_bad_cell_count"] != 0:
                    raise RuntimeError("final exact 500 nm audit is not zero")
                binary = (rho >= 0.5).astype(np.float64)
                binary_audit, _ = exact_binary_audit(binary)
                if binary_audit["total_bad_cell_count"] != 0:
                    raise RuntimeError("thresholded binary 500 nm audit failed")
                binary_path = raw_root / "final_exact_binary_density.npz"
                np.savez_compressed(binary_path, rho=binary)
                final_output = raw_root / "final_exact_binary_evaluation"
                command = [
                    sys.executable,
                    "-m",
                    "photothermal_pte.optimization_runs.tairte4_flake_topology.evaluate_binary_objective",
                    "--base-fsp", str(base_fsp),
                    "--base-sha256", args.base_sha256,
                    "--rho-npz", str(binary_path),
                    "--output-dir", str(final_output),
                    "--polarization", args.polarization,
                    "--gpu-device", f"GPU {args.gpu}",
                    "--cuda-device", "0",
                    "--reference-objective-A", str(objective),
                ]
                environment = dict(os.environ)
                environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
                final_result_path = final_output / "binary_objective_result.json"
                if final_result_path.is_file() and json.loads(final_result_path.read_text()).get("passed"):
                    completed_returncode = 0
                else:
                    if final_output.exists() and any(final_output.iterdir()):
                        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                        final_output.rename(final_output.with_name(
                            f"{final_output.name}_incomplete_{stamp}"
                        ))
                    completed_returncode = subprocess.run(
                        command, cwd=REPOSITORY, env=environment
                    ).returncode
                final_result = json.loads(final_result_path.read_text())
                if completed_returncode or not final_result.get("passed"):
                    raise RuntimeError("fresh final exact-binary evaluation failed")
                manifest["final_exact_binary_density"] = {
                    "path": str(binary_path),
                    "size_bytes": binary_path.stat().st_size,
                    "sha256": sha256(binary_path),
                }
                manifest["final_exact_binary_evaluation"] = final_result
                write_json(manifest_path, manifest)
                write_json(published / "FINAL_RESULT.json", {
                    "passed": True,
                    "status": "VALIDATED_TRUE_MMA_EXACT_BINARY_CONTACT_ANCHORED_PTE_OPTIMIZATION",
                    "polarization": args.polarization,
                    "algorithm": "persistent true MMA",
                    "initial_density": "uniform rho=0.5",
                    "final_beta": beta,
                    "accepted_updates": global_update,
                    "continuous_objective_A": objective,
                    "binary_result": final_result,
                    "exact_binary_audit": binary_audit,
                    "posthoc_morphology_repair": False,
                })
                emit(events, "optimization_complete", beta=beta, global_update=global_update)
                return 0

            beta_index += 1
            beta = BETA_SCHEDULE[beta_index]
            stage_updates = 0
            mma_state = initialize_mma_state(latent)
            evaluation_counter += 1
            rho = MAPPING.physical(latent, beta)
            result, gradient_physical, gradient_conductance = evaluate(
                rho,
                polarization=args.polarization,
                output=raw_root / f"evaluation_{evaluation_counter:04d}_beta{beta:g}_reprojection",
                gpu=args.gpu,
                events=events,
                base_fsp=base_fsp,
                base_sha256=args.base_sha256,
                jacobian_dir=jacobian_dir,
            )
            new_evaluations += 1
            objective = float(result["objective_A"])
            terminal_conductance = float(result["terminal_conductance_S"])
            current_source_power = float(result["forward"]["source_power_W"])
            source_change = abs(current_source_power - fixed_source_power) / fixed_source_power
            if source_change >= 0.005:
                raise RuntimeError("fixed-source-power audit changed by >=0.5%")
            stage_summary, _ = metrics(latent, beta, device=args.constraint_device)
            morphology_caps = stage_morphology_caps(np.asarray([
                stage_summary["smooth_solid_constraint"],
                stage_summary["smooth_void_constraint"],
            ]), beta)
            manifest["evaluations"][f"{evaluation_counter:04d}"] = record_manifest_entry(result)
            history.append({
                "role": "beta_reprojection",
                "evaluation_id": evaluation_counter,
                "global_update": global_update,
                "stage_update": 0,
                "beta": beta,
                "objective_A": objective,
                "objective_at_reference_power_A": equivalent_current(objective, fixed_source_power),
                "fixed_source_power_W": fixed_source_power,
                "source_power_relative_change": source_change,
                "terminal_conductance_S": terminal_conductance,
                "minimum_terminal_conductance_S": minimum_conductance,
                "morphology_caps": morphology_caps.tolist(),
                "gray_fraction_0p01_0p99": stage_summary["gray_fraction_0p01_0p99"],
                "binarization": stage_summary["binarization_mean_4rho1mrho"],
                "exact_bad_cells": stage_summary["exact"]["total_bad_cell_count"],
                "rho_mean": stage_summary["rho_mean"],
            })
            write_json(history_path, history)
            write_json(manifest_path, manifest)
            publish_plot(
                published,
                history,
                rho,
                gradient_physical,
                stage_summary,
                evaluation_id=evaluation_counter,
                label=f"beta{beta:g}_reprojection",
            )
            save_driver_state(
                state_path,
                latent=latent,
                gradient_physical_A=gradient_physical,
                gradient_conductance_S=gradient_conductance,
                objective_A=objective,
                terminal_conductance_S=terminal_conductance,
                fixed_source_power_W=fixed_source_power,
                beta_index=beta_index,
                stage_updates=stage_updates,
                global_update=global_update,
                evaluation_counter=evaluation_counter,
                morphology_caps=morphology_caps,
                mma_state=mma_state,
            )
            if args.max_new_evaluations and new_evaluations >= args.max_new_evaluations:
                emit(events, "pilot_limit_reached", new_evaluations=new_evaluations)
                return 0
            continue

        if stage_updates >= MAXIMUM_STAGE_UPDATES[beta]:
            raise RuntimeError(
                f"beta={beta:g} reached {MAXIMUM_STAGE_UPDATES[beta]} true-MMA updates "
                "without the measured convergence gate; refusing arbitrary promotion"
            )

        gradient_latent_A = MAPPING.vjp(latent, gradient_physical, beta)
        objective_gradient_minimize = (
            -gradient_latent_A
            * REFERENCE_INCIDENT_POWER_W
            / fixed_source_power
            / OBJECTIVE_SCALE_AT_REFERENCE_POWER_A
        )
        candidate_flat, candidate_mma_state, mma_diagnostics = mma_step(
            latent.ravel(),
            objective_gradient_minimize.ravel(),
            fval,
            dfdx.reshape(len(names), -1),
            mma_state,
            move_limit=MOVE_LIMIT[beta],
        )
        candidate = candidate_flat.reshape(MAPPING.shape)
        if np.array_equal(candidate, latent):
            raise RuntimeError("MMA returned an exactly stationary candidate before convergence")
        candidate_rho = MAPPING.physical(candidate, beta)
        evaluation_counter += 1
        result, candidate_gradient, candidate_conductance_gradient = evaluate(
            candidate_rho,
            polarization=args.polarization,
            output=raw_root / f"evaluation_{evaluation_counter:04d}_beta{beta:g}_mma",
            gpu=args.gpu,
            events=events,
            base_fsp=base_fsp,
            base_sha256=args.base_sha256,
            jacobian_dir=jacobian_dir,
        )
        new_evaluations += 1
        candidate_objective = float(result["objective_A"])
        candidate_conductance = float(result["terminal_conductance_S"])
        current_source_power = float(result["forward"]["source_power_W"])
        source_change = abs(current_source_power - fixed_source_power) / fixed_source_power
        if source_change >= 0.005:
            raise RuntimeError("fixed-source-power audit changed by >=0.5%")
        candidate_names, candidate_fval, _, candidate_summary, _ = canonical_constraints(
            latent=candidate,
            beta=beta,
            terminal_conductance_S=candidate_conductance,
            gradient_terminal_conductance_physical_S=candidate_conductance_gradient,
            minimum_terminal_conductance_S=minimum_conductance,
            morphology_caps=morphology_caps,
            device=args.constraint_device,
        )
        if candidate_names != names:
            raise RuntimeError("constraint ordering changed within one MMA stage")
        if candidate_fval[0] > 0.20:
            raise RuntimeError("terminal conductance violated its minimum by more than 20%")

        latent = candidate
        gradient_physical = candidate_gradient
        gradient_conductance = candidate_conductance_gradient
        objective = candidate_objective
        terminal_conductance = candidate_conductance
        mma_state = candidate_mma_state
        global_update += 1
        stage_updates += 1
        row = {
            "role": "accepted_mma",
            "algorithm": "persistent_separable_method_of_moving_asymptotes",
            "evaluation_id": evaluation_counter,
            "global_update": global_update,
            "stage_update": stage_updates,
            "beta": beta,
            "objective_A": objective,
            "objective_at_reference_power_A": equivalent_current(objective, fixed_source_power),
            "fixed_source_power_W": fixed_source_power,
            "source_power_relative_change": source_change,
            "terminal_conductance_S": terminal_conductance,
            "minimum_terminal_conductance_S": minimum_conductance,
            "constraint_names": candidate_names,
            "constraint_values": candidate_fval.tolist(),
            "maximum_constraint_value": float(np.max(candidate_fval)),
            "morphology_caps": morphology_caps.tolist(),
            "gray_fraction_0p01_0p99": candidate_summary["gray_fraction_0p01_0p99"],
            "binarization": candidate_summary["binarization_mean_4rho1mrho"],
            "exact_bad_cells": candidate_summary["exact"]["total_bad_cell_count"],
            "rho_mean": candidate_summary["rho_mean"],
            "mma_maximum_absolute_step": mma_diagnostics["maximum_absolute_step"],
            "mma_rms_step": mma_diagnostics["rms_step"],
            "mma_diagnostics": mma_diagnostics,
            "used_adam": False,
            "gradient_direction_normalization": False,
            "post_update_hard_clipping": False,
            "symmetry_constraint": False,
            "volume_constraint": False,
        }
        history.append(row)
        manifest["evaluations"][f"{evaluation_counter:04d}"] = record_manifest_entry(result)
        write_json(history_path, history)
        write_json(manifest_path, manifest)
        write_json(published / "latest_summary.json", candidate_summary)
        write_json(published / "optimization_history.json", history)
        iteration_record = published / f"iteration_{global_update:04d}.json"
        write_json(iteration_record, row)
        publish_plot(
            published,
            history,
            candidate_rho,
            candidate_gradient,
            candidate_summary,
            evaluation_id=evaluation_counter,
            label="accepted_true_mma",
        )
        save_driver_state(
            state_path,
            latent=latent,
            gradient_physical_A=gradient_physical,
            gradient_conductance_S=gradient_conductance,
            objective_A=objective,
            terminal_conductance_S=terminal_conductance,
            fixed_source_power_W=fixed_source_power,
            beta_index=beta_index,
            stage_updates=stage_updates,
            global_update=global_update,
            evaluation_counter=evaluation_counter,
            morphology_caps=morphology_caps,
            mma_state=mma_state,
        )
        emit(
            events,
            "accepted_true_mma_update",
            beta=beta,
            global_update=global_update,
            stage_update=stage_updates,
            objective_at_reference_power_A=row["objective_at_reference_power_A"],
            maximum_constraint_value=row["maximum_constraint_value"],
            mma_maximum_absolute_step=row["mma_maximum_absolute_step"],
        )
        if beta == BETA_SCHEDULE[0] and stage_updates == PILOT_ACCEPTED_UPDATES:
            pilot_passed = bool(
                row["mma_maximum_absolute_step"] <= MOVE_LIMIT[beta] + 1.0e-12
                and row["gray_fraction_0p01_0p99"] > 0.95
                and row["maximum_constraint_value"] <= 1.0e-3
                and np.isfinite(row["objective_at_reference_power_A"])
            )
            pilot = {
                "passed": pilot_passed,
                "status": (
                    "VALIDATED_LOW_BETA_MMA_PILOT"
                    if pilot_passed
                    else "FAILED_LOW_BETA_MMA_PILOT"
                ),
                "beta": beta,
                "accepted_updates": stage_updates,
                "maximum_absolute_step": row["mma_maximum_absolute_step"],
                "move_bound": MOVE_LIMIT[beta],
                "gray_fraction_0p01_0p99": row["gray_fraction_0p01_0p99"],
                "binarization": row["binarization"],
                "maximum_constraint_value": row["maximum_constraint_value"],
                "objective_at_reference_power_A": row["objective_at_reference_power_A"],
            }
            write_json(published / "LOW_BETA_PILOT_GATE.json", pilot)
            emit(events, "low_beta_pilot_gate", **pilot)
            if not pilot_passed:
                raise RuntimeError("low-beta MMA pilot failed closed")
        if args.max_new_evaluations and new_evaluations >= args.max_new_evaluations:
            emit(events, "pilot_limit_reached", new_evaluations=new_evaluations)
            return 0

    raise RuntimeError("unreachable: beta schedule exited without final validation")


if __name__ == "__main__":
    raise SystemExit(main())
