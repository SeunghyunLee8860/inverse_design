#!/usr/bin/env python3
"""Fresh Eb AD-FD precheck and optimization for the left/right electrodes."""

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
ADFD_ROOT = ARTIFACT_ROOT / "combined_adfd_Eb_run048_precheck"
ADFD_RESULT = ADFD_ROOT / "tairte4_flake_combined_adfd.json"
OPTIMIZATION_ROOT = ARTIFACT_ROOT / "run048_Eb_fresh_current_max"
RESULTS = HERE / "results"
STATE = HERE / "PIPELINE_STATE.json"
LOCK = Path("/tmp/seunghyun_run048_left_right_Eb.lock")
GPU = int(os.environ.get("RUN048_GPU", "5"))


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
        "polarization": "Eb",
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
            "XDG_CONFIG_HOME": "/tmp/seunghyun_lumerical_run048",
            "MPLCONFIGDIR": "/tmp/seunghyun_matplotlib_run048",
        }
    )
    return env


def run_checked(command: list[str], *, stage: str) -> None:
    write_state(stage, command=command)
    completed = subprocess.run(command, cwd=REPOSITORY, env=environment())
    if completed.returncode:
        write_state(stage, status="failed", returncode=completed.returncode)
        raise RuntimeError(f"{stage} failed with return code {completed.returncode}")


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = LOCK.open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Run048 pipeline is already active") from error

    if not BASE_FSP.is_file() or sha256(BASE_FSP) != BASE_SHA256:
        raise RuntimeError("immutable uniform-rho0.5 template FSP is missing or changed")
    jacobian_result = JACOBIAN_ROOT / "component_yee_jacobian_result.json"
    if not jacobian_result.is_file() or not json.loads(jacobian_result.read_text()).get("passed"):
        raise RuntimeError("immutable component-Yee Jacobian certificate is unavailable")
    if OPTIMIZATION_ROOT.exists() or RESULTS.exists():
        raise RuntimeError("Run048 output already exists; refusing a non-fresh start")

    if ADFD_ROOT.exists():
        if not ADFD_RESULT.is_file() or not json.loads(ADFD_RESULT.read_text()).get("passed"):
            raise RuntimeError("existing Run048 Eb AD-FD output is incomplete or failed")
    else:
        run_checked(
            [
                str(PYTHON), "-m",
                "photothermal_pte.optimization_runs.tairte4_flake_topology.validate_combined_adfd",
                "--base-fsp", str(BASE_FSP),
                "--base-sha256", BASE_SHA256,
                "--jacobian-dir", str(JACOBIAN_ROOT),
                "--output-dir", str(ADFD_ROOT),
                "--gpu-device", f"GPU {GPU}",
                "--cuda-device", "0",
                "--step", "0.005",
                "--polarization", "Eb",
            ],
            stage="combined_adfd_Eb",
        )
    if not ADFD_RESULT.is_file() or not json.loads(ADFD_RESULT.read_text()).get("passed"):
        raise RuntimeError("Run048 Eb combined AD-FD did not pass")

    RESULTS.mkdir(parents=True)
    OPTIMIZATION_ROOT.mkdir(parents=True)
    run_checked(
        [
            str(PYTHON), "-m",
            "photothermal_pte.optimization_runs.tairte4_flake_topology.run_ansys_dfm_ld_mma_optimization",
            "--polarization", "Eb",
            "--raw-root", str(OPTIMIZATION_ROOT),
            "--published-dir", str(RESULTS),
            "--gpu", str(GPU),
            "--base-fsp", str(BASE_FSP),
            "--base-sha256", BASE_SHA256,
            "--jacobian-dir", str(JACOBIAN_ROOT),
            "--constraint-device", "cuda:0",
        ],
        stage="optimization_Eb",
    )
    final_result = RESULTS / "FINAL_RESULT.json"
    if not final_result.is_file():
        raise RuntimeError("Run048 optimization returned without FINAL_RESULT.json")
    write_state("complete", status="complete", final_result=str(final_result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

