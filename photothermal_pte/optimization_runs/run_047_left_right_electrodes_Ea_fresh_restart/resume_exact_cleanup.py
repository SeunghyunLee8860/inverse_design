#!/usr/bin/env python3
"""Resume Run047 at its immutable beta=32 checkpoint after cleanup bug fix."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
PYTHON = Path("/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python")
CONDA_LIBRARY_DIR = PYTHON.parents[1] / "lib"
ROOT = Path("/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored")
RAW = ROOT / "run047_Ea_fresh_current_max"
CHECKPOINT = RAW / "stage_0006_beta32.npz"
CHECKPOINT_SHA256 = "7e199b2bb798b3815e07f5ca42e043079893853f34e01913044e8a04fde7dd9c"
BASE_FSP = ROOT / "uniform_rho0p5_Ea_forward_queued/attempt_0002/tairte4_flake_forward_Ea.fsp"
BASE_SHA256 = "6274627f8e84cc61a8b5925472fc131041e7662b06d77141f3b52353d3578aa6"
JACOBIAN = ROOT / "component_yee_jacobian_v1"
RESULTS = HERE / "results"
STATE = HERE / "PIPELINE_STATE.json"
GPU = int(os.environ.get("RUN047_GPU", "5"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_state(status: str, **extra: object) -> None:
    payload = {
        "stage": "exact_cleanup_recovery",
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "gpu": GPU,
        "checkpoint": str(CHECKPOINT),
        "restart_beta": 32.0,
        **extra,
    }
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(STATE)


def main() -> int:
    if not CHECKPOINT.is_file() or sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("Run047 beta=32 recovery checkpoint is missing or changed")
    if not BASE_FSP.is_file() or sha256(BASE_FSP) != BASE_SHA256:
        raise RuntimeError("Run047 base FSP is missing or changed")
    final = RESULTS / "FINAL_RESULT.json"
    if final.exists():
        raise RuntimeError("Run047 already has a final result")
    cleanup = RAW / "forced_exact_500nm_cleanup"
    if not cleanup.is_dir() or any(cleanup.iterdir()):
        raise RuntimeError("expected the failed Run047 cleanup directory to be empty")

    command = [
        str(PYTHON), "-m",
        "photothermal_pte.optimization_runs.tairte4_flake_topology.run_ansys_dfm_ld_mma_optimization",
        "--polarization", "Ea",
        "--raw-root", str(RAW),
        "--published-dir", str(RESULTS),
        "--gpu", str(GPU),
        "--base-fsp", str(BASE_FSP),
        "--base-sha256", BASE_SHA256,
        "--jacobian-dir", str(JACOBIAN),
        "--constraint-device", "cuda:0",
        "--initial-latent-npz", str(CHECKPOINT),
        "--recovery-append",
        "--start-beta", "32",
        "--output-slug", "ansys_dfm_ld_mma_exact_cleanup_recovery",
    ]
    env = dict(os.environ)
    inherited = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(
        part for part in (str(CONDA_LIBRARY_DIR), inherited) if part
    )
    env.update(
        {
            "PYTHONPATH": str(REPOSITORY),
            "TAIRTE4_TOPOLOGY_GEOMETRY": "left_right_contact_anchored",
            "CUDA_VISIBLE_DEVICES": str(GPU),
            "LUMERICAL_LICENSE_RETRY_SECONDS": "30",
            "LUMERICAL_GPU_ENGINE_LOCK": "/tmp/seunghyun_lumerical_fdtd_gpu_engine.lock",
            "XDG_CONFIG_HOME": "/tmp/seunghyun_lumerical_run047_cleanup",
            "MPLCONFIGDIR": "/tmp/seunghyun_matplotlib_run047_cleanup",
        }
    )
    write_state("running", command=command)
    completed = subprocess.run(command, cwd=REPOSITORY, env=env)
    if completed.returncode or not final.is_file():
        write_state("failed", returncode=completed.returncode)
        return 1
    result = json.loads(final.read_text())
    write_state("complete", final_status=result.get("status"), passed=result.get("passed"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
