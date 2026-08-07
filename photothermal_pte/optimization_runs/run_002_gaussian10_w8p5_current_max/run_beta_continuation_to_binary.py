#!/usr/bin/env python3
"""Resume Run 002 and continue autonomously through the binary gate.

This supervisor is intentionally append-only and restartable.  It waits for
the already running global iteration 7, accepts it only through the explicit
acceptance script, then advances beta with one accepted constrained-MMA step
per stage.  Failed candidates are preserved and retried with a smaller move.
It never uses CPU FDTD and never commits raw NPZ/FSP files.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
PYTHON = Path("/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python")
RAW_ROOT = Path("/home/seunghyun/tairte4/raw_artifacts")
BASE_FSP = RAW_ROOT / "run002_selected_production_geometry_runsetup_v2_20260806/production_candidate_runsetup.fsp"
BASE_SHA = "a86644647b8bf03ec1b83c34d0cf18b6b1c5316342d845ccbdccf60df3d8f904"
JACOBIAN = RAW_ROOT / "run002_selected_production_complex_yee_jacobian_20260806"
INCIDENT_POWER = "1.3822950233084244e-13"
STATE_ROOT = RAW_ROOT / "run002_beta_continuation_state_20260807"
DRIVER_STATE = STATE_ROOT / "continuation_driver_state.json"
DRIVER_LOG = STATE_ROOT / "continuation_driver_events.jsonl"


def emit(event: str, **values) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    row = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **values,
    }
    with DRIVER_LOG.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def execute(command: list[str], label: str, env=None, allow_failure=False) -> int:
    emit("command_start", label=label, command=command)
    completed = subprocess.run(command, cwd=REPOSITORY, env=env)
    emit("command_end", label=label, returncode=completed.returncode)
    if completed.returncode and not allow_failure:
        raise RuntimeError(f"{label} failed with return code {completed.returncode}")
    return completed.returncode


def wait_for_result(path: Path, timeout_s: float = 7200.0) -> dict:
    started = time.monotonic()
    while not path.exists():
        if time.monotonic() - started > timeout_s:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(20.0)
    data = json.loads(path.read_text())
    if not data.get("passed"):
        raise RuntimeError(f"solver result failed: {path}")
    return data


def evaluate(latent_npz: Path, output: Path, beta: float, gpu: str) -> tuple[Path, Path]:
    result = output / "selected_full_latent_adjoint_preparation_result.json"
    raw = output / "selected_full_latent_adjoint_preparation.npz"
    if result.exists():
        wait_for_result(result)
        return result, raw
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    execute([
        str(PYTHON), str(HERE / "prepare_selected_full_latent_adjoint.py"),
        "--base-fsp", str(BASE_FSP),
        "--base-sha256", BASE_SHA,
        "--jacobian-dir", str(JACOBIAN),
        "--latent-npz", str(latent_npz),
        "--output-dir", str(output),
        "--gpu-device", f"GPU {gpu}",
        "--cuda-device", "0",
        "--beta", str(beta),
        "--incident-power-W", INCIDENT_POWER,
    ], f"evaluate_beta_{beta:g}_{output.name}", env=environment)
    wait_for_result(result)
    return result, raw


def record(
    result: Path,
    raw: Path,
    beta: float,
    global_iteration: int,
    stage_iteration: int,
    role: str,
    proposal_result: Path | None = None,
    proposal_raw: Path | None = None,
    mma_state: Path | None = None,
) -> None:
    tag = f"continuation_b{int(round(beta)):03d}_i{stage_iteration:03d}_g{global_iteration:03d}"
    summary_path = HERE / "results/beta_continuation_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if any(
            int(row["global_iteration"]) == global_iteration
            and int(row["stage_iteration"]) == stage_iteration
            and float(row["beta"]) == float(beta)
            for row in summary.get("history", [])
        ):
            emit("record_reused", tag=tag)
            return
    command = [
        str(PYTHON), str(HERE / "record_beta_continuation_evaluation.py"),
        "--evaluation-result", str(result),
        "--evaluation-raw", str(raw),
        "--run-directory", str(HERE),
        "--global-iteration", str(global_iteration),
        "--stage-iteration", str(stage_iteration),
        "--role", role,
    ]
    if proposal_result and proposal_raw and mma_state:
        command.extend([
            "--proposal-result", str(proposal_result),
            "--proposal-raw", str(proposal_raw),
            "--mma-state", str(mma_state),
        ])
    execute(command, f"record_{tag}")


def git_checkpoint(beta: float, global_iteration: int, stage_iteration: int) -> None:
    tag = f"continuation_b{int(round(beta)):03d}_i{stage_iteration:03d}_g{global_iteration:03d}"
    tracked = [
        HERE / "STATUS.json",
        HERE / "run_config.json",
        HERE / "manifests/RAW_ARTIFACT_MANIFEST.json",
        HERE / "results/BETA_CONTINUATION_REPORT.md",
        HERE / "results/beta_continuation_history.csv",
        HERE / "results/beta_continuation_summary.json",
        HERE / "plots/beta_continuation_fom_binarization_drc.png",
        HERE / f"plots/{tag}_binarization_and_drc.png",
        HERE / f"plots/{tag}_design_constraints_gradients.png",
    ]
    execute(["git", "add", *map(str, tracked)], f"git_add_{tag}")
    status = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=REPOSITORY
    ).returncode
    if status == 0:
        emit("git_checkpoint_reused", tag=tag)
        return
    execute(
        ["git", "commit", "-m", f"Record beta {beta:g} continuation iteration {global_iteration}"],
        f"git_commit_{tag}",
    )
    execute(["git", "push", "origin", "HEAD"], f"git_push_{tag}", allow_failure=True)


def propose_and_accept(
    current_result: Path,
    current_raw: Path,
    current_state: Path,
    beta: float,
    global_iteration: int,
    gpu: str,
) -> tuple[Path, Path, Path]:
    for retry, move in enumerate((0.02, 0.01, 0.005, 0.0025)):
        stem = f"run002_beta{int(round(beta)):03d}_iteration{global_iteration:03d}_retry{retry}_20260807"
        proposal_dir = RAW_ROOT / f"{stem}_proposal"
        evaluation_dir = RAW_ROOT / f"{stem}_evaluation"
        proposal_result = proposal_dir / "beta_continuation_proposal_result.json"
        proposal_raw = proposal_dir / "beta_continuation_proposal.npz"
        if not proposal_result.exists():
            execute([
                str(PYTHON), str(HERE / "propose_beta_continuation_step.py"),
                "--evaluation-result", str(current_result),
                "--evaluation-raw", str(current_raw),
                "--mma-state", str(current_state),
                "--output-dir", str(proposal_dir),
                "--global-iteration", str(global_iteration),
                "--move", str(move),
                "--constraint-reduction", "0.01",
            ], f"propose_beta_{beta:g}_g{global_iteration}_retry{retry}")
        candidate_result, candidate_raw = evaluate(proposal_raw, evaluation_dir, beta, gpu)
        accepted_state = STATE_ROOT / f"mma_state_accepted_g{global_iteration:03d}.npz"
        acceptance = evaluation_dir / "beta_continuation_acceptance.json"
        returncode = execute([
            str(PYTHON), str(HERE / "accept_beta_continuation_step.py"),
            "--proposal-result", str(proposal_result),
            "--proposal-raw", str(proposal_raw),
            "--evaluation-result", str(candidate_result),
            "--evaluation-raw", str(candidate_raw),
            "--accepted-mma-state", str(accepted_state),
            "--output-json", str(acceptance),
        ], f"accept_beta_{beta:g}_g{global_iteration}_retry{retry}", allow_failure=True)
        if returncode == 0:
            record(
                candidate_result,
                candidate_raw,
                beta,
                global_iteration,
                1,
                "accepted_mma",
                proposal_result,
                proposal_raw,
                accepted_state,
            )
            git_checkpoint(beta, global_iteration, 1)
            return candidate_result, candidate_raw, accepted_state
        emit("candidate_rejected", beta=beta, global_iteration=global_iteration, retry=retry, move=move)
    raise RuntimeError(f"all retries rejected at beta={beta}, global iteration={global_iteration}")


def current_metrics() -> dict:
    summary = json.loads((HERE / "results/beta_continuation_summary.json").read_text())
    return summary["current_metrics"]


def projected_binary_gate(metrics: dict) -> bool:
    exact = metrics["exact_binary_audit"]
    return bool(
        metrics["gray_fraction_0p01_0p99"] < 0.001
        and metrics["binarization_metric_mean_4rho1mrho"] < 0.001
        and exact["solid_bad_cell_count"] == 0
        and exact["void_bad_cell_count"] == 0
    )


def save_state(**values) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    DRIVER_STATE.write_text(json.dumps({
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **values,
    }, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    emit("driver_started", gpu=args.gpu)

    # Finish the already-running exact-DRC-backtracked global iteration 7.
    beta = 4.0
    global_iteration = 7
    proposal_dir = RAW_ROOT / "run002_beta004_iteration007_proposal_drcbacktrack_20260807"
    evaluation_dir = RAW_ROOT / "run002_beta004_iteration007_evaluation_20260807"
    proposal_result = proposal_dir / "beta_continuation_proposal_result.json"
    proposal_raw = proposal_dir / "beta_continuation_proposal.npz"
    evaluation_result = evaluation_dir / "selected_full_latent_adjoint_preparation_result.json"
    evaluation_raw = evaluation_dir / "selected_full_latent_adjoint_preparation.npz"
    wait_for_result(evaluation_result)
    state = STATE_ROOT / "mma_state_accepted_g007.npz"
    acceptance = evaluation_dir / "beta_continuation_acceptance.json"
    if not state.exists():
        execute([
            str(PYTHON), str(HERE / "accept_beta_continuation_step.py"),
            "--proposal-result", str(proposal_result),
            "--proposal-raw", str(proposal_raw),
            "--evaluation-result", str(evaluation_result),
            "--evaluation-raw", str(evaluation_raw),
            "--accepted-mma-state", str(state),
            "--output-json", str(acceptance),
        ], "accept_existing_global_007")
    record(
        evaluation_result,
        evaluation_raw,
        beta,
        global_iteration,
        2,
        "accepted_mma",
        proposal_result,
        proposal_raw,
        state,
    )
    git_checkpoint(beta, global_iteration, 2)
    current_result, current_raw, current_state = evaluation_result, evaluation_raw, state

    # One evaluated MMA update at every beta.  If the strict projected-binary
    # gate is still open after 128, continue powers of two up to 4096.
    schedule = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    for beta_value in schedule:
        beta = float(beta_value)
        baseline_dir = RAW_ROOT / f"run002_beta{beta_value:03d}_baseline_g{global_iteration:03d}_20260807"
        baseline_result, baseline_raw = evaluate(current_raw, baseline_dir, beta, args.gpu)
        record(baseline_result, baseline_raw, beta, global_iteration, 0, "stage_baseline")
        git_checkpoint(beta, global_iteration, 0)
        current_result, current_raw = baseline_result, baseline_raw
        global_iteration += 1
        current_result, current_raw, current_state = propose_and_accept(
            current_result,
            current_raw,
            current_state,
            beta,
            global_iteration,
            args.gpu,
        )
        metrics = current_metrics()
        save_state(
            status="RUNNING",
            beta=beta,
            global_iteration=global_iteration,
            current_result=str(current_result),
            current_raw=str(current_raw),
            current_mma_state=str(current_state),
            projected_binary_gate=projected_binary_gate(metrics),
        )
        if beta >= 128 and projected_binary_gate(metrics):
            emit("projected_binary_gate_passed", beta=beta, global_iteration=global_iteration)
            break
    else:
        raise RuntimeError("projected binary gate did not pass through beta=4096")

    # The thresholded-binary forward validation is a separate executable gate;
    # leave an explicit ready marker rather than falsely promoting here.
    save_state(
        status="READY_FOR_THRESHOLDED_BINARY_SOLVER_REEVALUATION",
        beta=beta,
        global_iteration=global_iteration,
        current_result=str(current_result),
        current_raw=str(current_raw),
        current_mma_state=str(current_state),
        projected_binary_gate=True,
    )
    emit("driver_finished_continuation", beta=beta, global_iteration=global_iteration)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        emit("driver_failed", error=f"{type(error).__name__}: {error}")
        raise
