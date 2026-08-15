#!/usr/bin/env python3
"""Launch fresh trust-region Run056 E||b on an explicitly selected GPU."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
ARTIFACT_ROOT = Path("/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored")
RAW = ARTIFACT_ROOT / "run056_bounded_official_dfm_exact_repair_Eb_evaporated_v1"
BASE_FSP = ARTIFACT_ROOT / (
    "production_input_uniform_rho0p5_Ea_forward_v1/tairte4_flake_forward_Ea.fsp"
)
BASE_SHA256 = "454fa83bc918b4db0e25d28f7debf23de38977038bd716c8d7dc539d6b3e3d83"
JACOBIAN = ARTIFACT_ROOT / "production_input_component_yee_jacobian_v1"


def main() -> int:
    gpu = int(os.environ.get("RUN056_GPU", "2"))
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "TAIRTE4_TOPOLOGY_GEOMETRY": "contact_anchored",
            "TAIRTE4_SIO2_INTERFACE_SCENARIO": "evaporated",
            "LUMERICAL_LICENSE_RETRY_SECONDS": "30",
            "XDG_CONFIG_HOME": f"/tmp/seunghyun_lumerical_run056_gpu{gpu}",
            "MPLCONFIGDIR": f"/tmp/seunghyun_matplotlib_run056_gpu{gpu}",
        }
    )
    command = [
        sys.executable,
        "-u",
        "-m",
        "photothermal_pte.optimization_runs.tairte4_flake_topology.run_official_dfm_exact_repair_optimization",
        "--polarization",
        "Eb",
        "--gpu",
        str(gpu),
        "--raw-root",
        str(RAW),
        "--published-dir",
        str(HERE / "results"),
        "--base-fsp",
        str(BASE_FSP),
        "--base-sha256",
        BASE_SHA256,
        "--jacobian-dir",
        str(JACOBIAN),
        "--constraint-device",
        "cuda:0",
    ]
    return subprocess.run(command, cwd=REPOSITORY, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
