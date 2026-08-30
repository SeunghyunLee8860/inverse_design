#!/usr/bin/env python3
"""Resume Run050 at the next beta after the last audited stage checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
PYTHON = Path("/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python")
CONDA_LIBRARY_DIR = PYTHON.parents[1] / "lib"
ARTIFACT_ROOT = Path("/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored")
RAW_ROOT = ARTIFACT_ROOT / "run050_Eb_evaporated_fresh_current_max"
PUBLISHED = HERE / "results"
MANIFEST = PUBLISHED / "RAW_ARTIFACT_MANIFEST.json"
BASE_FSP = ARTIFACT_ROOT / (
    "production_input_uniform_rho0p5_Ea_forward_v1/"
    "tairte4_flake_forward_Ea.fsp"
)
BASE_SHA256 = "454fa83bc918b4db0e25d28f7debf23de38977038bd716c8d7dc539d6b3e3d83"
JACOBIAN_ROOT = ARTIFACT_ROOT / "production_input_component_yee_jacobian_v1"
GPU = int(os.environ.get("RUN050_GPU", "0"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    stages = manifest.get("stages", [])
    if not stages:
        raise RuntimeError("Run050 has no audited stage checkpoint to resume")
    last_stage = stages[-1]
    checkpoint_record = last_stage["checkpoint"]
    checkpoint = Path(checkpoint_record["path"])
    if not checkpoint.is_file() or sha256(checkpoint) != checkpoint_record["sha256"]:
        raise RuntimeError("latest Run050 stage checkpoint failed SHA-256 verification")
    last_beta = float(last_stage["beta"])
    start_beta = min(1024.0, max(256.0, 2.0 * last_beta))

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(
        part
        for part in (str(CONDA_LIBRARY_DIR), env.get("LD_LIBRARY_PATH", ""))
        if part
    )
    env.update(
        {
            "PYTHONPATH": str(REPOSITORY),
            "TAIRTE4_TOPOLOGY_GEOMETRY": "contact_anchored",
            "TAIRTE4_SIO2_INTERFACE_SCENARIO": "evaporated",
            "CUDA_VISIBLE_DEVICES": str(GPU),
            "LUMERICAL_LICENSE_RETRY_SECONDS": "30",
            "LUMERICAL_GPU_ENGINE_LOCK": "/tmp/seunghyun_lumerical_fdtd_gpu_engine.lock",
            "XDG_CONFIG_HOME": "/tmp/seunghyun_lumerical_run050",
            "MPLCONFIGDIR": "/tmp/seunghyun_matplotlib_run050",
        }
    )
    command = [
        str(PYTHON),
        "-m",
        "photothermal_pte.optimization_runs.tairte4_flake_topology."
        "run_ansys_dfm_ld_mma_optimization",
        "--polarization",
        "Eb",
        "--raw-root",
        str(RAW_ROOT),
        "--published-dir",
        str(PUBLISHED),
        "--gpu",
        str(GPU),
        "--base-fsp",
        str(BASE_FSP),
        "--base-sha256",
        BASE_SHA256,
        "--jacobian-dir",
        str(JACOBIAN_ROOT),
        "--constraint-device",
        "cuda:0",
        "--initial-latent-npz",
        str(checkpoint),
        "--recovery-append",
        "--start-beta",
        f"{start_beta:g}",
        "--output-slug",
        "ansys_dfm_ld_mma_high_beta_recovery",
    ]
    print(
        json.dumps(
            {
                "resume_checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_record["sha256"],
                "last_beta": last_beta,
                "start_beta": start_beta,
                "gpu": GPU,
            },
            indent=2,
        ),
        flush=True,
    )
    return subprocess.run(command, cwd=REPOSITORY, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
