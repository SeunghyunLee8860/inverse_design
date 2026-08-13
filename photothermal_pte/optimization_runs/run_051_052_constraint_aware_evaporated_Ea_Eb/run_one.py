#!/usr/bin/env python3
"""Run one fresh polarization under its own site runres GPU/license claim."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
ARTIFACT_ROOT = Path("/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored")
BASE_FSP = ARTIFACT_ROOT / (
    "production_input_uniform_rho0p5_Ea_forward_v1/tairte4_flake_forward_Ea.fsp"
)
BASE_SHA256 = "454fa83bc918b4db0e25d28f7debf23de38977038bd716c8d7dc539d6b3e3d83"
JACOBIAN = ARTIFACT_ROOT / "production_input_component_yee_jacobian_v1"


def main() -> int:
    polarization = os.environ.get("CONSTRAINT_AWARE_POLARIZATION")
    if polarization not in {"Ea", "Eb"}:
        raise RuntimeError("CONSTRAINT_AWARE_POLARIZATION must be Ea or Eb")
    gpu = int(os.environ["CONSTRAINT_AWARE_GPU"])
    suffix = polarization.lower()
    raw = ARTIFACT_ROOT / f"run05{1 if polarization == 'Ea' else 2}_constraint_aware_{polarization}_evaporated_v3"
    published = HERE / f"run_05{1 if polarization == 'Ea' else 2}_{polarization}_results_v3"
    if raw.exists() and any(raw.iterdir()):
        raise RuntimeError(f"fresh raw path is nonempty: {raw}")
    if published.exists() and any(published.iterdir()):
        raise RuntimeError(f"fresh published path is nonempty: {published}")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["TAIRTE4_TOPOLOGY_GEOMETRY"] = "contact_anchored"
    environment["TAIRTE4_SIO2_INTERFACE_SCENARIO"] = "evaporated"
    environment["XDG_CONFIG_HOME"] = f"/tmp/seunghyun_lumerical_constraint_aware_v3_{suffix}"
    environment["MPLCONFIGDIR"] = f"/tmp/seunghyun_matplotlib_constraint_aware_v3_{suffix}"
    command = [
        sys.executable,
        "-u",
        "-m",
        "photothermal_pte.optimization_runs.tairte4_flake_topology.run_ansys_dfm_ld_mma_optimization",
        "--polarization", polarization,
        "--gpu", str(gpu),
        "--raw-root", str(raw),
        "--published-dir", str(published),
        "--base-fsp", str(BASE_FSP),
        "--base-sha256", BASE_SHA256,
        "--jacobian-dir", str(JACOBIAN),
        "--constraint-device", "cuda:0",
        "--constraint-aware-continuation",
    ]
    return subprocess.run(command, cwd=REPOSITORY, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
