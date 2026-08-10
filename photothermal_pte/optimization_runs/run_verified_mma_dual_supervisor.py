#!/usr/bin/env python3
"""Run checkpointed, trust-region protected MMA for Ea and then Eb."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
PARENT = Path("/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored")
BASE_FSP = PARENT / "run012_uniform_rho0p5_Ea_forward_retry_20260810/tairte4_flake_forward_Ea.fsp"
BASE_SHA256 = "454fa83bc918b4db0e25d28f7debf23de38977038bd716c8d7dc539d6b3e3d83"
JACOBIAN = PARENT / "run012_component_yee_jacobian_retry_20260810"
PREFLIGHT_ROOT = PARENT / "run016_017_true_mma_preflight_20260810"
PREFLIGHT_REPORT = REPOSITORY / "photothermal_pte/optimization_runs/true_mma_preflight"
GPU = int(os.environ.get("TAIRTE4_VERIFIED_MMA_GPU", "1"))
STATUS = REPOSITORY / "photothermal_pte/optimization_runs/VERIFIED_MMA_DUAL_RUN_STATUS.json"
RUNS = (
    (
        "Run022", "Ea",
        PARENT / "run022_verified_mma_Ea_20260810",
        REPOSITORY / "photothermal_pte/optimization_runs/run_022_verified_mma_contact_anchored_Ea_current_max",
    ),
    (
        "Run023", "Eb",
        PARENT / "run023_verified_mma_Eb_20260810",
        REPOSITORY / "photothermal_pte/optimization_runs/run_023_verified_mma_contact_anchored_Eb_current_max",
    ),
)


def write_status(status: str, **values: object) -> None:
    payload = {
        "schema": "verified-persistent-mma-dual-supervisor-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "gpu": GPU,
        **values,
    }
    temporary = STATUS.with_suffix(STATUS.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(STATUS)
    print(json.dumps(payload), flush=True)


def passed(path: Path) -> bool:
    return path.is_file() and bool(json.loads(path.read_text()).get("passed"))


def run_checked(command: list[str], environment: dict[str, str]) -> None:
    completed = subprocess.run(command, cwd=REPOSITORY, env=environment)
    if completed.returncode:
        raise RuntimeError(f"command failed with return code {completed.returncode}: {command}")


def main() -> int:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(GPU)
    environment["TAIRTE4_TOPOLOGY_GEOMETRY"] = "contact_anchored"
    conda_library = "/home/eidl/miniconda3/envs/EIDL-Lumapi/lib"
    existing = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = conda_library if not existing else f"{conda_library}:{existing}"
    run_checked([
        sys.executable, "-m", "photothermal_pte.optimization_runs.audit_true_mma_preflight",
        "--output-dir", str(PREFLIGHT_REPORT),
        "--preflight-root", str(PREFLIGHT_ROOT),
    ], environment)
    preflight = PREFLIGHT_REPORT / "TRUE_MMA_PREFLIGHT.json"
    if not passed(preflight):
        raise RuntimeError("updated code/physics preflight is not passed")
    for label, polarization, raw, published in RUNS:
        final = published / "FINAL_RESULT.json"
        if passed(final):
            continue
        raw.mkdir(parents=True, exist_ok=True)
        published.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "-m",
            "photothermal_pte.optimization_runs.tairte4_flake_topology.run_true_mma_optimization",
            "--polarization", polarization,
            "--raw-root", str(raw),
            "--published-dir", str(published),
            "--gpu", str(GPU),
            "--base-fsp", str(BASE_FSP),
            "--base-sha256", BASE_SHA256,
            "--jacobian-dir", str(JACOBIAN),
            "--connectivity-fraction", "0.10",
            "--constraint-device", "cuda:0",
        ]
        write_status("RUNNING_VERIFIED_MMA", run=label, polarization=polarization, command=command)
        completed = subprocess.run(command, cwd=REPOSITORY, env=environment)
        if completed.returncode or not passed(final):
            write_status(
                "BLOCKED_VERIFIED_MMA_RUN",
                run=label,
                polarization=polarization,
                returncode=completed.returncode,
            )
            raise RuntimeError(f"{label} failed closed")
    write_status("VALIDATED_VERIFIED_MMA_EA_EB_OPTIMIZATIONS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
