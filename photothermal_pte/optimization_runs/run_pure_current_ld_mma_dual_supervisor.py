#!/usr/bin/env python3
"""Finish Run043 Eb, then run the same corrected LD_MMA contract for Ea.

No terminal-conductance/connectivity inequality is passed to either run.  The
top/bottom electrode weighting-potential boundary conditions remain part of
the electrical physics solved in every objective/gradient evaluation.

The supervisor may attach to an already-running Run043 process.  It never
starts Ea concurrently: Run043 must have a passed FINAL_RESULT and GPU 0 must
be idle first.  This is orchestration only; all optimizer physics and final
binary/500-nm gates remain in ``run_pure_current_ld_mma_optimization.py``.
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


REPOSITORY = Path(__file__).resolve().parents[2]
PARENT = Path("/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored")
BASE_FSP = PARENT / "run012_uniform_rho0p5_Ea_forward_retry_20260810/tairte4_flake_forward_Ea.fsp"
BASE_SHA256 = "454fa83bc918b4db0e25d28f7debf23de38977038bd716c8d7dc539d6b3e3d83"
JACOBIAN = PARENT / "run012_component_yee_jacobian_retry_20260810"
PREFLIGHT = REPOSITORY / "photothermal_pte/optimization_runs/true_mma_preflight/TRUE_MMA_PREFLIGHT.json"
GPU = int(os.environ.get("TAIRTE4_PURE_CURRENT_LD_MMA_GPU", "0"))
STATUS = REPOSITORY / "photothermal_pte/optimization_runs/PURE_CURRENT_LD_MMA_DUAL_RUN_STATUS.json"
RUN043_PUBLISHED = REPOSITORY / "photothermal_pte/optimization_runs/run_043_pure_current_ld_mma_shared_license_Eb_current_max"
RUN044_RAW = PARENT / "run044_pure_current_ld_mma_restoration_Ea_20260811"
RUN044_PUBLISHED = REPOSITORY / "photothermal_pte/optimization_runs/run_044_pure_current_ld_mma_restoration_Ea_current_max"
RUN044_INITIAL = PARENT / "run036_pure_current_ld_mma_morphology_from_beta1_Ea_20260810/evaluation_0017_beta1_pure_current_ld_mma_recovery2_latent.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_status(status: str, **values: object) -> None:
    payload = {
        "schema": "pure-terminal-current-nlopt-ld-mma-dual-supervisor-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "gpu": GPU,
        "algorithm": "NLopt LD_MMA",
        "objective": "signed full-flake terminal PTE current",
        "top_bottom_weighting_boundaries": {"top": 1.0, "bottom": 0.0},
        "terminal_conductance_constraint": False,
        "terminal_conductance_role": "diagnostic_only",
        "manual_move_limit": None,
        **values,
    }
    temporary = STATUS.with_suffix(STATUS.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(STATUS)
    print(json.dumps(payload), flush=True)


def passed(path: Path) -> bool:
    return path.is_file() and bool(json.loads(path.read_text()).get("passed"))


def process_matches(pid: int, token: str) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    return token in command


def gpu_compute_pids(gpu: int) -> list[int]:
    completed = subprocess.run(
        [
            "nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [int(value) for value in completed.stdout.split() if value.isdigit()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attach-run043-pid", type=int,
        help="Wait for this already-running Run043 Eb driver before starting Ea.",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0.0:
        raise ValueError("--poll-seconds must be positive")
    if not BASE_FSP.is_file() or sha256(BASE_FSP) != BASE_SHA256:
        raise RuntimeError("immutable base FSP is missing or SHA-mismatched")
    if not passed(PREFLIGHT):
        raise RuntimeError("existing optical/thermal/electrical/AD-FD preflight is not passed")
    certificate = JACOBIAN / "component_yee_jacobian_result.json"
    if not passed(certificate):
        raise RuntimeError("component-Yee Jacobian certificate is not passed")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(GPU)
    environment["TAIRTE4_TOPOLOGY_GEOMETRY"] = "contact_anchored"
    conda_library = "/home/eidl/miniconda3/envs/EIDL-Lumapi/lib"
    current_library_path = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = (
        conda_library if not current_library_path
        else f"{conda_library}:{current_library_path}"
    )
    run043_final = RUN043_PUBLISHED / "FINAL_RESULT.json"
    if not passed(run043_final):
        if args.attach_run043_pid is None:
            raise RuntimeError("Run043 is incomplete and no attach PID was supplied")
        write_status(
            "MONITORING_RUN043_EB",
            run="Run043",
            polarization="Eb",
            attached_pid=args.attach_run043_pid,
        )
        while process_matches(args.attach_run043_pid, "run043_pure_current_ld_mma"):
            time.sleep(args.poll_seconds)
        if not passed(run043_final):
            write_status(
                "BLOCKED_RUN043_EB_STOPPED_WITHOUT_FINAL_RESULT",
                run="Run043",
                polarization="Eb",
                attached_pid=args.attach_run043_pid,
            )
            raise RuntimeError("Run043 stopped without a passed FINAL_RESULT")

    if not RUN044_INITIAL.is_file():
        raise RuntimeError(f"Run044 audited Ea warm start is missing: {RUN044_INITIAL}")
    run044_final = RUN044_PUBLISHED / "FINAL_RESULT.json"
    if not passed(run044_final):
        if RUN044_RAW.exists() and any(RUN044_RAW.iterdir()):
            raise RuntimeError(
                f"Run044 raw root already contains data; refusing ambiguous overwrite: {RUN044_RAW}"
            )
        while True:
            active = gpu_compute_pids(GPU)
            if not active:
                break
            write_status("WAITING_FOR_IDLE_GPU_BEFORE_RUN044_EA", gpu_processes=active)
            time.sleep(args.poll_seconds)
        RUN044_RAW.mkdir(parents=True, exist_ok=True)
        RUN044_PUBLISHED.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "photothermal_pte.optimization_runs.tairte4_flake_topology.run_pure_current_ld_mma_optimization",
            "--polarization", "Ea",
            "--raw-root", str(RUN044_RAW),
            "--published-dir", str(RUN044_PUBLISHED),
            "--gpu", str(GPU),
            "--base-fsp", str(BASE_FSP),
            "--base-sha256", BASE_SHA256,
            "--jacobian-dir", str(JACOBIAN),
            "--constraint-device", "cuda:0",
            "--initial-latent-npz", str(RUN044_INITIAL),
            "--start-beta", "1.0",
        ]
        write_status(
            "RUNNING_RUN044_EA_WITH_CORRECTED_RESTORATION",
            run="Run044",
            polarization="Ea",
            native_ld_mma_warm_restart=True,
            command=command,
        )
        completed = subprocess.run(command, cwd=REPOSITORY, env=environment)
        if completed.returncode or not passed(run044_final):
            write_status(
                "BLOCKED_RUN044_EA_STOPPED_WITHOUT_FINAL_RESULT",
                run="Run044",
                polarization="Ea",
                returncode=completed.returncode,
            )
            raise RuntimeError("Run044 Ea stopped without a passed FINAL_RESULT")
    write_status("VALIDATED_PURE_CURRENT_NLOPT_LD_MMA_EA_EB_OPTIMIZATIONS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
