#!/usr/bin/env python3
"""Restart Ea from the immutable pre-beta-4 Run051 design using fast rules."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
ARTIFACT_ROOT = Path("/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored")
SOURCE_SEED = ARTIFACT_ROOT / (
    "run051_constraint_aware_Ea_evaporated_v3/"
    "evaluation_0089_beta2_ansys_dfm_ld_mma_recovery_beta2_gpu1_retry2_latent.npz"
)
SOURCE_SEED_SHA256 = "50a755eb39cc0ed90cb38f3a7199146f52dd6e2fae28801d3b80694fba9e291e"
SEED_DIR = ARTIFACT_ROOT / "run053_Ea_prebeta4_seed_v1"
SEED = SEED_DIR / "initial_latent_from_run051_evaluation0089.npz"
RAW = ARTIFACT_ROOT / "run053_constraint_aware_Ea_evaporated_fast_from_beta4_v1"
PUBLISHED = HERE / "run_053_Ea_fast_from_beta4_results_v1"
BASE_FSP = ARTIFACT_ROOT / (
    "production_input_uniform_rho0p5_Ea_forward_v1/tairte4_flake_forward_Ea.fsp"
)
BASE_SHA256 = "454fa83bc918b4db0e25d28f7debf23de38977038bd716c8d7dc539d6b3e3d83"
JACOBIAN = ARTIFACT_ROOT / "production_input_component_yee_jacobian_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    gpu = int(os.environ.get("RUN053_EA_GPU", "4"))
    if sha256(SOURCE_SEED) != SOURCE_SEED_SHA256:
        raise RuntimeError("Run051 pre-beta-4 latent checkpoint changed")
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    if not SEED.exists():
        shutil.copy2(SOURCE_SEED, SEED)
    if sha256(SEED) != SOURCE_SEED_SHA256:
        raise RuntimeError("independent Run053 seed copy failed SHA verification")
    if RAW.exists() and any(RAW.iterdir()):
        raise RuntimeError(f"fresh Run053 raw path is nonempty: {RAW}")
    if PUBLISHED.exists() and any(PUBLISHED.iterdir()):
        raise RuntimeError(f"fresh Run053 published path is nonempty: {PUBLISHED}")

    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "TAIRTE4_TOPOLOGY_GEOMETRY": "contact_anchored",
            "TAIRTE4_SIO2_INTERFACE_SCENARIO": "evaporated",
            "XDG_CONFIG_HOME": f"/tmp/seunghyun_lumerical_run053_Ea_gpu{gpu}",
            "MPLCONFIGDIR": f"/tmp/seunghyun_matplotlib_run053_Ea_gpu{gpu}",
        }
    )
    command = [
        sys.executable,
        "-u",
        "-m",
        "photothermal_pte.optimization_runs.tairte4_flake_topology.run_ansys_dfm_ld_mma_optimization",
        "--polarization",
        "Ea",
        "--gpu",
        str(gpu),
        "--raw-root",
        str(RAW),
        "--published-dir",
        str(PUBLISHED),
        "--base-fsp",
        str(BASE_FSP),
        "--base-sha256",
        BASE_SHA256,
        "--jacobian-dir",
        str(JACOBIAN),
        "--constraint-device",
        "cuda:0",
        "--initial-latent-npz",
        str(SEED),
        "--start-beta",
        "4",
        "--constraint-aware-continuation",
        "--fast-continuation",
    ]
    return subprocess.run(command, cwd=REPOSITORY, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
