#!/usr/bin/env python3
"""Fresh Run047 optimization from uniform rho=0.5 with no warm restart.

The already validated optical forward, component-Yee Jacobian, and combined
AD-FD certificate are immutable inputs.  Only the optimization is restarted.
Run047 has new raw/published roots, so NLopt LD_MMA starts at evaluation 1,
beta=1, with a new internal MMA state and no checkpoint arguments.
"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
PYTHON = Path("/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python")
CONDA_LIBRARY_DIR = PYTHON.parents[1] / "lib"
ARTIFACT_ROOT = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored"
)
BASE_FSP = ARTIFACT_ROOT / (
    "uniform_rho0p5_Ea_forward_queued/attempt_0002/"
    "tairte4_flake_forward_Ea.fsp"
)
BASE_SHA256 = "6274627f8e84cc61a8b5925472fc131041e7662b06d77141f3b52353d3578aa6"
JACOBIAN_ROOT = ARTIFACT_ROOT / "component_yee_jacobian_v1"
ADFD_RESULT = ARTIFACT_ROOT / (
    "combined_adfd_Ea_reserved_v3/tairte4_flake_combined_adfd.json"
)
OPTIMIZATION_ROOT = ARTIFACT_ROOT / "run047_Ea_fresh_current_max"
RESULTS = HERE / "results"
STATE = HERE / "PIPELINE_STATE.json"
LOCK = Path("/tmp/seunghyun_run047_left_right_Ea.lock")
GPU = int(os.environ.get("RUN047_GPU", "5"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_state(stage: str, **extra: object) -> None:
    payload = {
        "stage": stage,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "gpu": GPU,
        "fresh_restart": True,
        "initial_density": 0.5,
        "warm_restart": False,
        "mma_internal_state_reused": False,
        **extra,
    }
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(STATE)


def environment() -> dict[str, str]:
    env = dict(os.environ)
    inherited_library_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(
        part for part in (str(CONDA_LIBRARY_DIR), inherited_library_path) if part
    )
    env.update(
        {
            "PYTHONPATH": str(REPOSITORY),
            "TAIRTE4_TOPOLOGY_GEOMETRY": "left_right_contact_anchored",
            "CUDA_VISIBLE_DEVICES": str(GPU),
            "LUMERICAL_LICENSE_RETRY_SECONDS": "30",
            "LUMERICAL_GPU_ENGINE_LOCK": "/tmp/seunghyun_lumerical_fdtd_gpu_engine.lock",
            "XDG_CONFIG_HOME": "/tmp/seunghyun_lumerical_run047",
            "MPLCONFIGDIR": "/tmp/seunghyun_matplotlib_run047",
        }
    )
    return env


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = LOCK.open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Run047 pipeline is already active") from error

    if not BASE_FSP.is_file() or sha256(BASE_FSP) != BASE_SHA256:
        raise RuntimeError("immutable uniform-rho0.5 Ea base FSP is missing or changed")
    jacobian_result = JACOBIAN_ROOT / "component_yee_jacobian_result.json"
    if not jacobian_result.is_file() or not json.loads(jacobian_result.read_text()).get("passed"):
        raise RuntimeError("immutable component-Yee Jacobian certificate is unavailable")
    if not ADFD_RESULT.is_file() or not json.loads(ADFD_RESULT.read_text()).get("passed"):
        raise RuntimeError("immutable combined Ea AD-FD certificate is unavailable")
    if OPTIMIZATION_ROOT.exists() or RESULTS.exists():
        raise RuntimeError("Run047 output already exists; refusing a non-fresh start")

    RESULTS.mkdir(parents=True)
    OPTIMIZATION_ROOT.mkdir(parents=True)
    write_state("optimization_Ea", status="starting_from_uniform_rho0p5")
    command = [
        str(PYTHON), "-m",
        "photothermal_pte.optimization_runs.tairte4_flake_topology.run_ansys_dfm_ld_mma_optimization",
        "--polarization", "Ea",
        "--raw-root", str(OPTIMIZATION_ROOT),
        "--published-dir", str(RESULTS),
        "--gpu", str(GPU),
        "--base-fsp", str(BASE_FSP),
        "--base-sha256", BASE_SHA256,
        "--jacobian-dir", str(JACOBIAN_ROOT),
        "--constraint-device", "cuda:0",
    ]
    # Deliberately no --initial-latent-npz, --recovery-append, or --start-beta.
    completed = subprocess.run(command, cwd=REPOSITORY, env=environment())
    final_result = RESULTS / "FINAL_RESULT.json"
    if completed.returncode or not final_result.is_file():
        write_state("optimization_Ea", status="failed", returncode=completed.returncode)
        raise RuntimeError("Run047 fresh optimization did not produce a final result")
    write_state("complete", status="complete", final_result=str(final_result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
