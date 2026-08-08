#!/usr/bin/env python3
"""Run 005: full fail-closed beta continuation to an exact binary design."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from optimization_support import (
    DISK_CONSTRAINT_START_BETA,
    MMA_TOPOLOGY_RESET_GLOBAL_ITERATION,
    adaptive_move_ceiling,
    calibrate_stage_caps,
    candidate_acceptance,
    catastrophic_exact_growth,
    constraint_contract,
    constraint_values_and_gradients,
    design_metrics,
    initialize_mma_state,
    low_beta_morphology_limited_transition,
    mma_effective_constraint_caps,
    mma_step,
    projected_binary_gate,
    save_mma_state,
    stage_caps,
    stage_convergence,
    smooth_feasibility_move_retries,
    transient_license_failure,
    two_consecutive_subthreshold_moves,
)
from record_state import artifact, record, sha256
from optimization_support import MMA_CAP_CONTRACT


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
RUN002 = HERE.parent / "run_002_gaussian10_w8p5_current_max"
PYTHON = Path("/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python")
RAW_ROOT = Path("/home/seunghyun/tairte4/raw_artifacts")
# Keep the original location as the reproducible default, but allow a resumed
# supervisor to place only future multi-GB solver artifacts on a roomier local
# filesystem.  Accepted checkpoints already recorded in the published summary
# retain their immutable absolute paths and are loaded from those paths.
RUN_RAW = Path(
    os.environ.get(
        "RUN005_RAW_ROOT",
        "/data/seunghyun/tairte4/raw_artifacts/run005_lowbeta_topology_pilot_20260808",
    )
).expanduser().resolve()
EVENTS = RUN_RAW / "driver_events.jsonl"
BASE_FSP = RAW_ROOT / "run002_selected_production_geometry_runsetup_v2_20260806/production_candidate_runsetup.fsp"
BASE_SHA = "a86644647b8bf03ec1b83c34d0cf18b6b1c5316342d845ccbdccf60df3d8f904"
JACOBIAN = RAW_ROOT / "run002_selected_production_complex_yee_jacobian_20260806"
INITIAL_RESULT = RAW_ROOT / "run002_selected_full_latent_adjoint_preparation_20260806/selected_full_latent_adjoint_preparation_result.json"
INITIAL_RESULT_SHA = "8fb48e9487c28fea9bebb771c49b5b10144e154b6b1c51d0e90d72155c0c54e4"
INITIAL_RAW = RAW_ROOT / "run002_selected_full_latent_adjoint_preparation_20260806/selected_full_latent_adjoint_preparation.npz"
INITIAL_RAW_SHA = "d3617baf54d54e735feba9d85c439ee77bcdf5ddaeec47e12c812bf036b2c87e"
INCIDENT_POWER = "1.3822950233084244e-13"
OBJECTIVE_SCALE = 1.0e8
BETA_SCHEDULE = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
MAXIMUM_ACCEPTED_UPDATES = {
    2: 16,
    4: 10,
    8: 10,
    16: 8,
    32: 6,
    64: 6,
    128: 4,
}
REPROJECTION_ONLY_BETA = 256
EXACT_DRC_GATE_START_BETA = 32.0
# Below beta=32 the smooth disk-opening constraints guide topology search, but
# a hard cap at roundoff-level feasibility can reject every useful MMA step
# when the current iterate sits on the cap. Keep the calibrated cap immutable
# and permit only a bounded ten-percent filter/trust band. This is not a cap
# reset: candidates outside the band are rejected before any Maxwell solve.
LOW_BETA_SMOOTH_TRUST_RELATIVE = 0.10
STRICT_SMOOTH_TRUST_RELATIVE = 5.0e-3
NO_PROGRESS_WINDOW = 6
MINIMUM_WINDOW_FOM_GAIN = 0.002
MINIMUM_WINDOW_EXACT_REDUCTION_FRACTION = 0.02
LICENSE_RETRY_LIMIT = 120
LICENSE_RETRY_DELAY_S = 30
STAGE_CAP_TARGET_OCCUPANCY = {
    4: 0.85,
    8: 0.88,
    16: 0.90,
    # Above beta=16 the task changes from topology exploration to bounded
    # morphology restoration.  Occupancy > 1 deliberately makes the incoming
    # checkpoint infeasible by a controlled amount; accepted steps must reduce
    # that violation while exact thresholded DRC is nonincreasing.
    32: 1.10,
    64: 1.15,
    128: 1.20,
}


def emit(event: str, **values) -> None:
    RUN_RAW.mkdir(parents=True, exist_ok=True)
    row = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "event": event, **values}
    with EVENTS.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def execute(command: list[str], label: str, env=None, allow_failure=False) -> int:
    emit("command_start", label=label, command=command)
    completed = subprocess.run(command, cwd=REPOSITORY, env=env)
    emit("command_end", label=label, returncode=completed.returncode)
    if completed.returncode and not allow_failure:
        raise RuntimeError(f"{label} failed with return code {completed.returncode}")
    return completed.returncode


def archive_transient_license_failure(output: Path, attempt: int) -> Path:
    """Preserve a failed solver directory before a clean deterministic retry."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = output.with_name(
        f"{output.name}_failed_license_attempt{attempt:03d}_{stamp}"
    )
    output.rename(archive)
    emit(
        "transient_license_failure_archived",
        evaluation=str(output),
        archive=str(archive),
        retry_attempt=attempt,
        retry_delay_s=LICENSE_RETRY_DELAY_S,
    )
    return archive


def archive_incomplete_evaluation(output: Path) -> Path:
    """Preserve an orphaned solver directory before restarting its evaluation.

    A detached solver can be terminated externally before it writes the compact
    diagnostic JSON.  Reusing that directory would mix two solver attempts and
    overwrite the partial FSP/log provenance.
    """

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = output.with_name(f"{output.name}_incomplete_{stamp}")
    output.rename(archive)
    emit(
        "incomplete_evaluation_archived",
        evaluation=str(output),
        archive=str(archive),
    )
    return archive


def solver_diagnostic_text(output: Path) -> str:
    """Return bounded engine-log text used only for license classification."""

    if not output.exists():
        return ""
    chunks: list[str] = []
    for path in sorted(output.glob("*_p0.log")):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        chunks.append(text[-65536:])
    return "\n".join(chunks)


def configure_single_physical_gpu(gpu: str, constraint_device: str) -> None:
    """Pin both Lumerical and PyTorch work to one physical GPU.

    PyTorch's ``cuda:0`` is a logical index.  Without an explicit visibility
    mask it means physical GPU 0 even when Lumerical is independently sent to
    physical GPU 2.  Set the mask before the first CUDA tensor is created so
    logical ``cuda:0`` and the requested Lumerical GPU identify one device.
    """

    if constraint_device != "cuda:0":
        raise RuntimeError(
            "GPU-only Run005 requires --constraint-device cuda:0 under a "
            "single-device CUDA_VISIBLE_DEVICES mask"
        )
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing not in (None, "", gpu):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES conflicts with --gpu: "
            f"{existing!r} != {gpu!r}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["RUN005_CONSTRAINT_TORCH_DEVICE"] = constraint_device


def verify_inputs() -> None:
    expected = (
        (BASE_FSP, BASE_SHA), (INITIAL_RESULT, INITIAL_RESULT_SHA),
        (INITIAL_RAW, INITIAL_RAW_SHA),
    )
    for path, digest in expected:
        if not path.exists() or sha256(path) != digest:
            raise RuntimeError(f"immutable input missing or SHA mismatch: {path}")
    if not JACOBIAN.is_dir():
        raise RuntimeError(f"missing certified material Jacobian: {JACOBIAN}")


def evaluate(latent_npz: Path, output: Path, beta: float, gpu: str) -> tuple[Path, Path]:
    retry_attempt = 0
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    while True:
        result = output / "selected_full_latent_adjoint_preparation_result.json"
        raw = output / "selected_full_latent_adjoint_preparation.npz"
        if result.exists():
            payload = json.loads(result.read_text())
            if payload.get("passed"):
                if not raw.exists() or sha256(raw) != payload["raw_artifact"]["sha256"]:
                    raise RuntimeError(f"existing solver result SHA mismatch: {result}")
                return result, raw
            if not transient_license_failure(payload, solver_diagnostic_text(output)):
                raise RuntimeError(f"existing solver result is invalid: {result}")
            if retry_attempt >= LICENSE_RETRY_LIMIT:
                raise RuntimeError(f"transient license retry limit reached: {result}")
            archive_transient_license_failure(output, retry_attempt)
            retry_attempt += 1
            time.sleep(LICENSE_RETRY_DELAY_S)

        if output.exists() and any(output.iterdir()) and not result.exists():
            archive_incomplete_evaluation(output)

        output.mkdir(parents=True, exist_ok=True)
        returncode = execute([
            str(PYTHON), str(RUN002 / "prepare_selected_full_latent_adjoint.py"),
            "--base-fsp", str(BASE_FSP), "--base-sha256", BASE_SHA,
            "--jacobian-dir", str(JACOBIAN), "--latent-npz", str(latent_npz),
            "--output-dir", str(output), "--gpu-device", f"GPU {gpu}",
            "--cuda-device", "0", "--beta", str(beta),
            "--incident-power-W", INCIDENT_POWER,
            "--allow-closed-unit-interval-latent",
        ], f"evaluate_{output.name}", env=environment, allow_failure=True)
        if returncode == 0:
            payload = json.loads(result.read_text())
            if not payload.get("passed") or not raw.exists() or sha256(raw) != payload["raw_artifact"]["sha256"]:
                raise RuntimeError(f"new solver result failed: {result}")
            return result, raw
        if not result.exists():
            raise RuntimeError(
                f"evaluate_{output.name} failed without a diagnostic result"
            )
        payload = json.loads(result.read_text())
        if not transient_license_failure(payload, solver_diagnostic_text(output)):
            raise RuntimeError(
                f"evaluate_{output.name} failed with non-license error"
            )


def summary() -> dict:
    path = HERE / "results/optimization_summary.json"
    return json.loads(path.read_text()) if path.exists() else {"history": []}


def initialize_summary_provenance() -> None:
    path = HERE / "results/optimization_summary.json"
    data = json.loads(path.read_text())
    data["base_FSP"] = artifact(BASE_FSP)
    data["material_Jacobian"] = {"path": str(JACOBIAN.resolve()), "sha256_contract": "directory recorded by Run 002 component-Jacobian manifest"}
    data["original_beta2_initial_result"] = artifact(INITIAL_RESULT)
    data["original_beta2_initial_NPZ"] = artifact(INITIAL_RAW)
    path.write_text(json.dumps(data, indent=2) + "\n")
    manifest_path = HERE / "manifests/RAW_ARTIFACT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["base_FSP"] = data["base_FSP"]
    manifest["material_Jacobian"] = data["material_Jacobian"]
    manifest["original_beta2_initial_result"] = data["original_beta2_initial_result"]
    manifest["original_beta2_initial_NPZ"] = data["original_beta2_initial_NPZ"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def already_recorded(beta: float, stage_iteration: int, role: str) -> bool:
    return any(
        float(row["beta"]) == float(beta)
        and int(row["stage_iteration"]) == stage_iteration
        and row["role"] == role
        for row in summary().get("history", [])
    )


def accepted_move_provenance(
    summary_data: dict[str, object], beta: float
) -> list[float]:
    moves: list[float] = []
    for row in summary_data.get("history", []):
        if float(row["beta"]) != float(beta) or row["role"] != "accepted_mma":
            continue
        if (
            float(beta) == DISK_CONSTRAINT_START_BETA
            and int(row.get("global_iteration", -1))
            <= MMA_TOPOLOGY_RESET_GLOBAL_ITERATION
        ):
            continue
        entry = summary_data["raw_artifacts"][row["tag"]]
        proposal = entry.get("proposal_result")
        if proposal is None:
            raise RuntimeError(f"accepted update lacks proposal provenance: {row['tag']}")
        payload = json.loads(Path(proposal["path"]).read_text())
        if sha256(Path(proposal["path"])) != proposal["sha256"]:
            raise RuntimeError(f"accepted proposal SHA mismatch: {row['tag']}")
        moves.append(float(payload["move"]))
    return moves


def checkpoint_git(tag: str, extra: list[Path] | None = None) -> None:
    tracked = [
        HERE / "STATUS.json", HERE / "run_config.json", HERE / "OPTIMIZATION_CONTRACT.md",
        HERE / "results/OPTIMIZATION_REPORT.md", HERE / "results/optimization_summary.json",
        HERE / "results/optimization_history.csv", HERE / "plots/iteration_fom_beta_constraints_drc.png",
        HERE / "manifests/RAW_ARTIFACT_MANIFEST.json",
        HERE / "stage_caps.json",
    ]
    if extra:
        tracked.extend(extra)
    existing = [str(path) for path in tracked if path.exists()]
    execute(["git", "add", *existing], f"git_add_{tag}")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPOSITORY).returncode:
        execute(["git", "commit", "-m", f"Record Run 005 {tag}"], f"git_commit_{tag}")
        execute(["git", "push", "origin", "HEAD"], f"git_push_{tag}", allow_failure=True)


def propose(
    current_result: Path,
    current_raw: Path,
    state_path: Path,
    beta: float,
    global_iteration: int,
    stage_iteration: int,
    retry: int,
    move: float,
) -> tuple[Path, Path, Path]:
    output = RUN_RAW / f"b{int(beta):04d}_s{stage_iteration:03d}_g{global_iteration:03d}_retry{retry}_proposal"
    result_path = output / "proposal_result.json"
    raw_path = output / "proposal.npz"
    candidate_state_path = output / "mma_state_candidate.npz"
    if result_path.exists():
        payload = json.loads(result_path.read_text())
        if payload.get("mma_cap_contract") == MMA_CAP_CONTRACT:
            if sha256(raw_path) != payload["raw_artifact"]["sha256"]:
                raise RuntimeError("existing proposal SHA mismatch")
            return result_path, raw_path, candidate_state_path
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive = output.with_name(
            f"{output.name}_stale_mma_cap_contract_{stamp}"
        )
        output.rename(archive)
        emit(
            "stale_proposal_contract_archived",
            proposal=str(output),
            archive=str(archive),
            previous_contract=payload.get("mma_cap_contract"),
            required_contract=MMA_CAP_CONTRACT,
        )
    # A supervisor interruption can occur after the directory is created but
    # before proposal_result.json is committed.  That directory is not an
    # accepted checkpoint.  Re-enter it and deterministically overwrite the
    # proposal payload; completed proposals are still validated and returned
    # by the SHA-checked branch above.
    output.mkdir(parents=True, exist_ok=True)
    current_payload = json.loads(current_result.read_text())
    current_data = np.load(current_raw)
    latent = np.asarray(current_data["latent"], float)
    gradient_A = np.asarray(current_data["gradient_latent_A"], float)
    values, gradients, _ = constraint_values_and_gradients(latent, beta)
    fixed_caps = stage_caps(beta)
    effective_caps = mma_effective_constraint_caps(values, beta)
    fval = values / effective_caps - 1.0
    dfdx = gradients.reshape(2, -1) / effective_caps[:, None]
    from optimization_support import load_mma_state
    state = load_mma_state(state_path)
    objective_gradient = -OBJECTIVE_SCALE * gradient_A.ravel() / float(current_payload["incident_power_W"])
    candidate_flat, candidate_state, diagnostics = mma_step(
        latent.ravel(), objective_gradient, fval, dfdx, state, move=move
    )
    candidate = candidate_flat.reshape(latent.shape)
    candidate_metrics, candidate_arrays = design_metrics(candidate, beta)
    np.savez_compressed(
        raw_path, latent=candidate, latent_previous=latent,
        rho=candidate_arrays["rho"], rho_previous=design_metrics(latent, beta)[1]["rho"],
        gradient_latent_A=gradient_A, constraint_values=values,
        constraint_caps=effective_caps,
    )
    save_mma_state(candidate_state_path, candidate_state)
    payload = {
        "status": "PROPOSED_RUN005_LOWBETA_TOPOLOGY_MMA_STEP", "passed": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "beta": beta, "global_iteration": global_iteration,
        "stage_iteration": stage_iteration, "retry": retry, "move": move,
        "objective_scale_W_per_A": OBJECTIVE_SCALE,
        "fixed_constraint_caps": fixed_caps.tolist(),
        "mma_effective_constraint_caps": effective_caps.tolist(),
        "mma_cap_contract": MMA_CAP_CONTRACT,
        "constraint_current": values.tolist(),
        "constraint_contract": constraint_contract(beta),
        "candidate_offline_metrics": candidate_metrics,
        "mma_diagnostics": diagnostics,
        "raw_artifact": artifact(raw_path), "candidate_state": artifact(candidate_state_path),
        "current_evaluation_result": artifact(current_result), "current_evaluation_NPZ": artifact(current_raw),
        "exact_DRC_used_as_gradient": False,
        "exact_DRC_used_as_hard_step_veto": bool(
            beta >= EXACT_DRC_GATE_START_BETA
        ),
        "posthoc_binary_repair": False,
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    return result_path, raw_path, candidate_state_path


def accept_candidate(
    current_result: Path, current_raw: Path,
    proposal_result: Path, proposal_raw: Path, candidate_state: Path,
    candidate_result: Path, candidate_raw: Path,
    accepted_state: Path,
) -> dict[str, object]:
    current_payload = json.loads(current_result.read_text())
    candidate_payload = json.loads(candidate_result.read_text())
    proposal_payload = json.loads(proposal_result.read_text())
    beta = float(candidate_payload["beta"])
    current_latent = np.asarray(np.load(current_raw)["latent"], float)
    candidate_latent = np.asarray(np.load(candidate_raw)["latent"], float)
    proposed_latent = np.asarray(np.load(proposal_raw)["latent"], float)
    if not np.array_equal(candidate_latent, proposed_latent):
        raise RuntimeError("evaluated latent does not equal proposed latent")
    current_metrics, _ = design_metrics(current_latent, beta)
    candidate_metrics, _ = design_metrics(candidate_latent, beta)
    current_values = np.asarray([
        current_metrics["solid_constraint"], current_metrics["void_constraint"]
    ], float)
    candidate_values = np.asarray([
        candidate_metrics["solid_constraint"], candidate_metrics["void_constraint"]
    ], float)
    current_exact = current_metrics["exact_binary_audit"]
    candidate_exact = candidate_metrics["exact_binary_audit"]
    exact_gate = beta >= EXACT_DRC_GATE_START_BETA
    smooth_trust_relative = (
        STRICT_SMOOTH_TRUST_RELATIVE if exact_gate
        else LOW_BETA_SMOOTH_TRUST_RELATIVE
    )
    decision = candidate_acceptance(
        float(current_payload["objective_A"]), float(candidate_payload["objective_A"]),
        current_values, candidate_values, stage_caps(beta),
        np.asarray([
            current_exact["solid_bad_cell_count"],
            current_exact["void_bad_cell_count"],
        ]) if exact_gate else None,
        np.asarray([
            candidate_exact["solid_bad_cell_count"],
            candidate_exact["void_bad_cell_count"],
        ]) if exact_gate else None,
        smooth_trust_relative=smooth_trust_relative,
    )
    decision.update({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "beta": beta, "physics_pass": bool(candidate_payload.get("passed")),
        "proposal": artifact(proposal_result), "candidate_evaluation": artifact(candidate_result),
        "fixed_constraint_caps": stage_caps(beta).tolist(),
        "constraint_contract": constraint_contract(beta),
        "current_constraints": current_values.tolist(), "candidate_constraints": candidate_values.tolist(),
        "exact_DRC_used_as_hard_step_veto": bool(exact_gate),
    })
    decision["accepted"] = bool(decision["accepted"] and candidate_payload.get("passed"))
    acceptance_path = candidate_result.parent / "acceptance.json"
    if decision["accepted"]:
        accepted_state.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_state, accepted_state)
        decision["accepted_mma_state"] = artifact(accepted_state)
    acceptance_path.write_text(json.dumps(decision, indent=2) + "\n")
    return decision


def finalize(current_raw: Path, beta: float, global_iteration: int, gpu: str) -> None:
    metrics, arrays = design_metrics(np.asarray(np.load(current_raw)["latent"], float), beta)
    if not projected_binary_gate(metrics):
        raise RuntimeError("projected binary gate is not satisfied")
    final_root = RUN_RAW / f"final_binary_g{global_iteration:03d}_b{int(beta):04d}"
    final_root.mkdir(parents=True, exist_ok=True)
    binary_npz = final_root / "thresholded_binary_density.npz"
    if not binary_npz.exists():
        np.savez_compressed(binary_npz, rho_binary=arrays["binary"].astype(np.uint8), source_beta=np.asarray(beta), source_global_iteration=np.asarray(global_iteration))
    output = final_root / "solver_evaluation"
    result = output / "thresholded_binary_evaluation_result.json"
    raw = output / "thresholded_binary_evaluation.npz"
    if not result.exists():
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        execute([
            str(PYTHON), str(RUN002 / "evaluate_thresholded_binary.py"),
            "--base-fsp", str(BASE_FSP), "--base-sha256", BASE_SHA,
            "--binary-npz", str(binary_npz), "--output-dir", str(output),
            "--gpu-device", f"GPU {gpu}", "--cuda-device", "0",
            "--incident-power-W", INCIDENT_POWER,
        ], "evaluate_final_thresholded_binary", env=environment)
    payload = json.loads(result.read_text())
    if not payload.get("passed") or payload.get("status") != "VALIDATED_THRESHOLDED_BINARY_GPU_MAXWELL_CUDA_THERMAL":
        raise RuntimeError("final binary GPU/CUDA evaluation failed")
    data = np.load(raw)
    rho = np.asarray(data["rho_binary"], float)
    if not np.all(np.isin(np.unique(rho), [0.0, 1.0])):
        raise RuntimeError("final density is not exact binary")
    q = np.asarray(data["Q_total_W_m3"], float)
    dz = np.diff(np.asarray(data["z_edges_m"], float))
    qxy = np.sum(q * dz[None, None, :], axis=2)
    temperature = np.asarray(data["thermal_temperature_grid_K"], float)
    with np.errstate(all="ignore"):
        txy = np.nanmax(temperature, axis=2)
    xedges = np.asarray(data["x_edges_m"]) * 1e6
    yedges = np.asarray(data["y_edges_m"]) * 1e6
    fig, axes_plot = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    panels = ((rho.T, [-9.3, 9.3, -9.3, 9.3], "exact binary design", "gray"), (qxy.T, [xedges[0], xedges[-1], yedges[0], yedges[-1]], "depth-integrated Q", "inferno"), (txy.T, [xedges[0], xedges[-1], yedges[0], yedges[-1]], "maximum T through z", "magma"))
    for ax, (field, extent, title, cmap) in zip(axes_plot, panels):
        im = ax.imshow(field, origin="lower", extent=extent, cmap=cmap)
        ax.set_title(title); ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
        fig.colorbar(im, ax=ax)
    final_plot = HERE / "plots/final_thresholded_binary_gpu_cuda_validation.png"
    fig.savefig(final_plot, dpi=180); plt.close(fig)
    summary_path = HERE / "results/optimization_summary.json"
    summary_data = json.loads(summary_path.read_text())
    status = "COMPLETED_FULLY_BINARIZED_EXACT_500NM_CONSTRAINED_PTE_OPTIMIZATION"
    summary_data.update({"status": status, "final_beta": beta, "final_global_iteration": global_iteration, "final_binary_result": payload, "final_binary_result_artifact": artifact(result), "final_binary_NPZ": artifact(raw), "final_binary_source": artifact(binary_npz), "final_plot": str(final_plot)})
    summary_path.write_text(json.dumps(summary_data, indent=2) + "\n")
    (HERE / "STATUS.json").write_text(json.dumps({"run_id": HERE.name, "status": status, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "final_beta": beta, "final_global_iteration": global_iteration, "final_FOM_A_per_W": payload["objective_A_per_incident_W"], "exact_binary_audit": payload["exact_binary_audit"]}, indent=2) + "\n")
    manifest_path = HERE / "manifests/RAW_ARTIFACT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({"status": status, "final_thresholded_binary": {"source": artifact(binary_npz), "evaluation_result": artifact(result), "evaluation_NPZ": artifact(raw)}})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (HERE / "results/OPTIMIZATION_REPORT.md").write_text(
        "# Run 005 fully binary PTE optimization\n\n"
        f"Status: `{status}`\n\nFinal beta: `{beta:g}`; global iteration: `{global_iteration}`.\n\n"
        f"Fresh thresholded-binary GPU Maxwell/CUDA thermal-PTE FOM: `{payload['objective_A_per_incident_W']:.12e} A/W`.\n\n"
        f"Exact solid/void bad cells: `{payload['exact_binary_audit']['solid_bad_cell_count']}` / `{payload['exact_binary_audit']['void_bad_cell_count']}`. No post-hoc binary repair was used.\n"
    )
    checkpoint_git("final_binary_validation", [final_plot])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--constraint-device", default="cuda:0")
    args = parser.parse_args()
    configure_single_physical_gpu(args.gpu, args.constraint_device)
    verify_inputs()
    emit(
        "driver_started",
        gpu=args.gpu,
        constraint_device=args.constraint_device,
        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        single_physical_gpu=True,
        method="plateau-gated beta continuation to exact binary 500 nm design",
    )
    current_result, current_raw = INITIAL_RESULT, INITIAL_RAW
    global_iteration = 0
    if not already_recorded(2.0, 0, "stage_baseline"):
        out = record(current_result, current_raw, 0, 0, "stage_baseline")
        initialize_summary_provenance()
        checkpoint_git(out["tag"], [Path(path) for path in out["plots"]])

    for beta_integer in BETA_SCHEDULE:
        beta = float(beta_integer)
        current_latent = np.asarray(np.load(current_raw)["latent"], float)
        target_occupancy = STAGE_CAP_TARGET_OCCUPANCY.get(beta_integer, 0.95)
        initial_caps, cap_audit = calibrate_stage_caps(
            current_latent, beta, target_occupancy
        )
        emit(
            "stage_cap_reprojection_audit",
            **cap_audit,
            immutable_within_stage=True,
        )
        if beta_integer >= REPROJECTION_ONLY_BETA:
            current_metrics, _ = design_metrics(current_latent, beta)
            exact = current_metrics["exact_binary_audit"]
            exact_total = int(
                exact["solid_bad_cell_count"] + exact["void_bad_cell_count"]
            )
            emit(
                "high_beta_solver_free_reprojection",
                beta=beta,
                gray_fraction=current_metrics["gray_fraction_0p01_0p99"],
                binarization_metric=current_metrics[
                    "binarization_metric_mean_4rho1mrho"
                ],
                exact_bad_total=exact_total,
            )
            if exact_total != 0:
                raise RuntimeError(
                    "exact 500 nm violations survived beta=128; do not spend "
                    "high-beta iterations on gradient-starved micro-repair"
                )
            if projected_binary_gate(current_metrics):
                emit(
                    "projected_binary_gate_passed",
                    beta=beta,
                    global_iteration=global_iteration,
                )
                finalize(current_raw, beta, global_iteration, args.gpu)
                emit("driver_finished", beta=beta, global_iteration=global_iteration)
                return 0
            continue
        if beta != 2.0:
            baseline_output = RUN_RAW / f"b{beta_integer:04d}_baseline_g{global_iteration:03d}"
            current_result, current_raw = evaluate(current_raw, baseline_output, beta, args.gpu)
            if not already_recorded(beta, 0, "stage_baseline"):
                out = record(current_result, current_raw, global_iteration, 0, "stage_baseline")
                checkpoint_git(out["tag"], [Path(path) for path in out["plots"]])
        current_latent = np.asarray(np.load(current_raw)["latent"], float)
        state_path = RUN_RAW / f"b{beta_integer:04d}_mma_state_stage_start.npz"
        if not state_path.exists():
            save_mma_state(state_path, initialize_mma_state(current_latent))
        accepted_rows = [row for row in summary()["history"] if float(row["beta"]) == beta and row["role"] == "accepted_mma"]
        stage_iteration = len(accepted_rows)
        if accepted_rows:
            last = accepted_rows[-1]
            entry = summary()["raw_artifacts"][last["tag"]]
            current_result = Path(entry["evaluation_result"]["path"])
            current_raw = Path(entry["evaluation_NPZ"]["path"])
            state_path = Path(entry["mma_state"]["path"])
            global_iteration = int(last["global_iteration"])
            current_latent = np.asarray(np.load(current_raw)["latent"], float)
        initial_values, _, _ = constraint_values_and_gradients(
            current_latent, beta
        )
        cap_occupancy = initial_values / initial_caps
        occupancy_interval = (
            [target_occupancy, target_occupancy]
            if target_occupancy > 1.0 else [0.8, 0.95]
        )
        emit(
            "stage_cap_reprojection_audit",
            beta=beta,
            initial_constraints=initial_values.tolist(),
            caps=initial_caps.tolist(),
            cap_occupancy=cap_occupancy.tolist(),
            recommended_occupancy_interval=occupancy_interval,
            occupancy_in_recommended_interval=[
                bool(
                    occupancy_interval[0] - 1.0e-12
                    <= value
                    <= occupancy_interval[1] + 1.0e-12
                )
                for value in cap_occupancy
            ],
            note="one checkpoint-calibrated cap epoch; immutable within this beta stage",
        )
        recovery_rows = [
            row for row in accepted_rows
            if not (
                beta == DISK_CONSTRAINT_START_BETA
                and int(row.get("global_iteration", -1))
                <= MMA_TOPOLOGY_RESET_GLOBAL_ITERATION
            )
        ]
        if beta == DISK_CONSTRAINT_START_BETA and not recovery_rows:
            state_path = RUN_RAW / (
                f"b{beta_integer:04d}_disk_constraint_recovery_"
                f"g{global_iteration:03d}_mma_state.npz"
            )
            if not state_path.exists():
                save_mma_state(state_path, initialize_mma_state(current_latent))
            emit(
                "constraint_contract_recovery",
                beta=beta,
                global_iteration=global_iteration,
                accepted_updates_before_disk_state_reset=len(accepted_rows),
                new_constraint_contract=constraint_contract(beta),
                mma_state_reset=str(state_path),
            )
        while True:
            history = summary()["history"]
            convergence = stage_convergence(history, beta)
            if convergence.converged:
                emit("stage_converged", beta=beta, global_iteration=global_iteration, diagnostics=convergence.__dict__)
                break
            accepted_moves_for_transition = accepted_move_provenance(
                summary(), beta
            )
            if two_consecutive_subthreshold_moves(
                accepted_moves_for_transition
            ):
                emit(
                    "stage_minimum_move_transition",
                    beta=beta,
                    global_iteration=global_iteration,
                    accepted_updates=len(accepted_moves_for_transition),
                    last_two_moves=accepted_moves_for_transition[-2:],
                    converged=False,
                    convergence_diagnostics=convergence.__dict__,
                    reason=(
                        "two consecutive minimum authorized moves indicate "
                        "a morphology-limited stage; advance projection "
                        "instead of repeating micro-repair or relaxing caps"
                    ),
                )
                break
            if low_beta_morphology_limited_transition(
                history, beta, accepted_moves_for_transition
            ):
                recent_beta_rows = [
                    row for row in history
                    if float(row["beta"]) == beta
                    and row["role"] == "accepted_mma"
                ]
                emit(
                    "stage_morphology_limited_transition",
                    beta=beta,
                    global_iteration=global_iteration,
                    accepted_updates=len(recent_beta_rows),
                    last_two_moves=accepted_moves_for_transition[-2:],
                    previous_exact_bad_total=(
                        int(recent_beta_rows[-2]["solid_bad_cells"])
                        + int(recent_beta_rows[-2]["void_bad_cells"])
                    ),
                    latest_exact_bad_total=(
                        int(recent_beta_rows[-1]["solid_bad_cells"])
                        + int(recent_beta_rows[-1]["void_bad_cells"])
                    ),
                    latest_relative_fom_change=float(
                        recent_beta_rows[-1]["relative_fom_change"]
                    ),
                    reason=(
                        "minimum authorized beta2 move worsened exact morphology; "
                        "advance projection instead of repeating micro-repair"
                    ),
                )
                break
            stage_maximum = MAXIMUM_ACCEPTED_UPDATES[beta_integer]
            if stage_iteration >= stage_maximum:
                # The bound is a per-beta work budget, not a terminal failure.
                # Once it is exhausted, advancing the projection is the only
                # authorized continuation: repeating the same beta would be
                # the blind micro-repair that this guard is meant to prevent.
                # Preserve the non-plateau diagnostics explicitly so a budget
                # transition is never misreported as numerical convergence.
                emit(
                    "stage_budget_transition",
                    beta=beta,
                    global_iteration=global_iteration,
                    accepted_updates=stage_iteration,
                    maximum_accepted_updates=stage_maximum,
                    converged=False,
                    convergence_diagnostics=convergence.__dict__,
                    reason=(
                        "bounded per-beta exploration budget exhausted; "
                        "advance projection without claiming plateau"
                    ),
                )
                break
            recent = [
                row for row in history
                if float(row["beta"]) == beta and row["role"] == "accepted_mma"
            ][-NO_PROGRESS_WINDOW:]
            if len(recent) == NO_PROGRESS_WINDOW:
                net_fom_gain = (
                    float(recent[-1]["objective_A_per_W"])
                    / float(recent[0]["objective_A_per_W"]) - 1.0
                )
                exact_first = int(recent[0]["solid_bad_cells"]) + int(recent[0]["void_bad_cells"])
                exact_last = int(recent[-1]["solid_bad_cells"]) + int(recent[-1]["void_bad_cells"])
                exact_reduction = (exact_first - exact_last) / max(exact_first, 1)
                if (
                    net_fom_gain < MINIMUM_WINDOW_FOM_GAIN
                    and exact_reduction < MINIMUM_WINDOW_EXACT_REDUCTION_FRACTION
                    and not convergence.converged
                ):
                    emit(
                        "automatic_no_progress_transition",
                        beta=beta,
                        global_iteration=global_iteration,
                        window=NO_PROGRESS_WINDOW,
                        net_fom_gain=net_fom_gain,
                        exact_bad_reduction_fraction=exact_reduction,
                        reason=(
                            "neither FOM nor exact morphology made meaningful "
                            "progress at this beta; advance projection"
                        ),
                    )
                    break
            summary_data = summary()
            accepted_moves = accepted_move_provenance(summary_data, beta)
            move_ceiling = adaptive_move_ceiling(history, beta, accepted_moves)
            move_retries = smooth_feasibility_move_retries(move_ceiling)
            emit(
                "adaptive_move_selected",
                beta=beta,
                stage_iteration=stage_iteration,
                accepted_move_min=min(accepted_moves) if accepted_moves else None,
                move_ceiling=move_ceiling,
                move_retries=list(move_retries),
                convergence_diagnostics=convergence.__dict__,
            )
            global_iteration += 1
            next_stage_iteration = stage_iteration + 1
            accepted = False
            advance_stage = False
            for retry, move in enumerate(move_retries):
                proposal_result, proposal_raw, candidate_state = propose(
                    current_result, current_raw, state_path, beta,
                    global_iteration, next_stage_iteration, retry, move,
                )
                proposal_payload = json.loads(proposal_result.read_text())
                current_values = np.asarray(
                    proposal_payload["constraint_current"], dtype=float
                )
                candidate_values = np.asarray([
                    proposal_payload["candidate_offline_metrics"]["solid_constraint"],
                    proposal_payload["candidate_offline_metrics"]["void_constraint"],
                ], dtype=float)
                fixed_caps = np.asarray(
                    proposal_payload["fixed_constraint_caps"], dtype=float
                )
                effective_caps = np.asarray(
                    proposal_payload["mma_effective_constraint_caps"], dtype=float
                )
                current_fixed_violation = float(np.linalg.norm(
                    np.maximum(current_values / fixed_caps - 1.0, 0.0)
                ))
                candidate_fixed_violation = float(np.linalg.norm(
                    np.maximum(candidate_values / fixed_caps - 1.0, 0.0)
                ))
                smooth_trust_relative = (
                    STRICT_SMOOTH_TRUST_RELATIVE
                    if beta >= EXACT_DRC_GATE_START_BETA
                    else LOW_BETA_SMOOTH_TRUST_RELATIVE
                )
                current_fixed_feasible = bool(np.all(
                    current_values <= fixed_caps * (1.0 + smooth_trust_relative)
                ))
                candidate_fixed_feasible = bool(np.all(
                    candidate_values <= fixed_caps * (1.0 + smooth_trust_relative)
                ))
                effective_cap_feasible = bool(np.all(
                    candidate_values <= effective_caps * (1.0 + 5.0e-3)
                ))
                if current_fixed_feasible:
                    smooth_prescreen_pass = bool(
                        candidate_fixed_feasible
                        and (
                            beta < EXACT_DRC_GATE_START_BETA
                            or effective_cap_feasible
                        )
                    )
                    smooth_reason = (
                        "candidate must remain inside the fixed-cap trust band; "
                        "from beta=32 it must also respect strict phase trust caps"
                    )
                else:
                    violation_reduction = (
                        (current_fixed_violation - candidate_fixed_violation)
                        / max(current_fixed_violation, 1.0e-300)
                    )
                    smooth_prescreen_pass = bool(
                        candidate_fixed_feasible or violation_reduction >= 0.01
                    )
                    smooth_reason = (
                        "infeasible candidate must reduce normalized smooth "
                        "violation by at least one percent"
                    )
                if not smooth_prescreen_pass:
                    emit(
                        "candidate_prescreen_rejected",
                        beta=beta,
                        global_iteration=global_iteration,
                        stage_iteration=next_stage_iteration,
                        retry=retry,
                        move=move,
                        reason=smooth_reason,
                        current_constraints=current_values.tolist(),
                        candidate_constraints=candidate_values.tolist(),
                        fixed_caps=fixed_caps.tolist(),
                        effective_caps=effective_caps.tolist(),
                        current_fixed_violation=current_fixed_violation,
                        candidate_fixed_violation=candidate_fixed_violation,
                        smooth_trust_relative=smooth_trust_relative,
                        effective_cap_feasible=effective_cap_feasible,
                        Maxwell_solves=0,
                        thermal_solves=0,
                    )
                    continue
                candidate_exact = proposal_payload["candidate_offline_metrics"]["exact_binary_audit"]
                current_metrics, _ = design_metrics(
                    np.asarray(np.load(current_raw)["latent"], float), beta
                )
                current_exact = current_metrics["exact_binary_audit"]
                current_exact_total = int(
                    current_exact["solid_bad_cell_count"]
                    + current_exact["void_bad_cell_count"]
                )
                candidate_exact_total = int(
                    candidate_exact["solid_bad_cell_count"]
                    + candidate_exact["void_bad_cell_count"]
                )
                if beta < EXACT_DRC_GATE_START_BETA:
                    catastrophic, catastrophic_limit = catastrophic_exact_growth(
                        current_exact_total, candidate_exact_total
                    )
                    emit(
                        "low_beta_exact_diagnostic",
                        beta=beta,
                        global_iteration=global_iteration,
                        stage_iteration=next_stage_iteration,
                        move=move,
                        current_exact_bad_total=current_exact_total,
                        candidate_exact_bad_total=candidate_exact_total,
                        catastrophic_limit=catastrophic_limit,
                        exact_used_as_step_veto=False,
                        catastrophic_guard_triggered=catastrophic,
                    )
                    if catastrophic:
                        emit(
                            "candidate_prescreen_rejected",
                            beta=beta,
                            global_iteration=global_iteration,
                            stage_iteration=next_stage_iteration,
                            retry=retry,
                            move=move,
                            reason=(
                                "low-beta exact bad-cell count exceeded the "
                                "catastrophic diagnostic guard"
                            ),
                            current_exact_bad_total=current_exact_total,
                            candidate_exact_bad_total=candidate_exact_total,
                            catastrophic_limit=catastrophic_limit,
                            Maxwell_solves=0,
                            thermal_solves=0,
                        )
                        # The retry schedule is bounded below by 0.0025.  A
                        # smaller authorized topology step may be prescreened,
                        # but no sub-threshold micro-repair is ever generated.
                        continue
                if (
                    beta >= EXACT_DRC_GATE_START_BETA
                    and candidate_exact_total > current_exact_total
                ):
                    emit(
                        "candidate_prescreen_rejected",
                        beta=beta,
                        global_iteration=global_iteration,
                        stage_iteration=next_stage_iteration,
                        retry=retry,
                        move=move,
                        reason="exact 500 nm total bad-cell count increased",
                        current_exact_bad_counts=[
                            current_exact["solid_bad_cell_count"],
                            current_exact["void_bad_cell_count"],
                        ],
                        candidate_exact_bad_counts=[
                            candidate_exact["solid_bad_cell_count"],
                            candidate_exact["void_bad_cell_count"],
                        ],
                        current_exact_bad_total=current_exact_total,
                        candidate_exact_bad_total=candidate_exact_total,
                        Maxwell_solves=0,
                        thermal_solves=0,
                    )
                    continue
                evaluation_output = RUN_RAW / f"b{beta_integer:04d}_s{next_stage_iteration:03d}_g{global_iteration:03d}_retry{retry}_evaluation"
                candidate_result, candidate_raw = evaluate(proposal_raw, evaluation_output, beta, args.gpu)
                accepted_state = RUN_RAW / f"b{beta_integer:04d}_s{next_stage_iteration:03d}_g{global_iteration:03d}_accepted_mma_state.npz"
                decision = accept_candidate(current_result, current_raw, proposal_result, proposal_raw, candidate_state, candidate_result, candidate_raw, accepted_state)
                duplicate_context = {
                    "beta", "global_iteration", "stage_iteration", "retry",
                    "proposal", "candidate_evaluation", "generated_at_utc",
                }
                emit(
                    "candidate_decision",
                    beta=beta,
                    global_iteration=global_iteration,
                    stage_iteration=next_stage_iteration,
                    retry=retry,
                    move=move,
                    **{
                        key: value for key, value in decision.items()
                        if key not in duplicate_context
                    },
                )
                if decision["accepted"]:
                    out = record(candidate_result, candidate_raw, global_iteration, next_stage_iteration, "accepted_mma", proposal_result, proposal_raw, accepted_state)
                    checkpoint_git(out["tag"], [Path(path) for path in out["plots"]])
                    current_result, current_raw, state_path = candidate_result, candidate_raw, accepted_state
                    stage_iteration = next_stage_iteration
                    accepted = True
                    updated_moves = accepted_move_provenance(summary(), beta)
                    if two_consecutive_subthreshold_moves(updated_moves):
                        emit(
                            "stage_minimum_move_transition",
                            beta=beta,
                            global_iteration=global_iteration,
                            accepted_moves=updated_moves[-2:],
                            threshold=0.0025,
                            converged=False,
                            reason=(
                                "two consecutive accepted minimum moves; "
                                "advance projection instead of cap relaxation "
                                "or repeated micro-repair"
                            ),
                        )
                        advance_stage = True
                    break
                raise RuntimeError(
                    "solver-backed candidate failed the FOM/physics acceptance "
                    "gate; continuation stopped without a smaller-move GPU retry"
                )
            if not accepted:
                raise RuntimeError(f"all candidate move retries rejected at beta={beta:g}, global iteration={global_iteration}")
            if advance_stage:
                break
        current_metrics = summary()["current_metrics"]
        if beta >= 64.0 and projected_binary_gate(current_metrics):
            emit("projected_binary_gate_passed", beta=beta, global_iteration=global_iteration)
            finalize(current_raw, beta, global_iteration, args.gpu)
            emit("driver_finished", beta=beta, global_iteration=global_iteration)
            return 0
    raise RuntimeError("full beta schedule ended before the projected-binary gate")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        emit("driver_failed", error=f"{type(error).__name__}: {error}")
        raise
