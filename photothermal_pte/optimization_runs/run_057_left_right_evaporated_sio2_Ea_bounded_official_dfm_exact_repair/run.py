#!/usr/bin/env python3
"""Launch fresh left/right-contact evaporated-SiO2 E||a Run057."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
ARTIFACT_ROOT = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored"
)
RAW = ARTIFACT_ROOT / "run057_bounded_official_dfm_exact_repair_Ea_evaporated_v1"
BASE_FSP = ARTIFACT_ROOT / (
    "uniform_rho0p5_Ea_forward_queued/attempt_0002/"
    "tairte4_flake_forward_Ea.fsp"
)
BASE_SHA256 = "6274627f8e84cc61a8b5925472fc131041e7662b06d77141f3b52353d3578aa6"
JACOBIAN = ARTIFACT_ROOT / "component_yee_jacobian_v1"


def main() -> int:
    gpu = int(os.environ.get("RUN057_GPU", "0"))
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "TAIRTE4_TOPOLOGY_GEOMETRY": "left_right_contact_anchored",
            "TAIRTE4_SIO2_INTERFACE_SCENARIO": "evaporated",
            "LUMERICAL_LICENSE_RETRY_SECONDS": "30",
            "LUMERICAL_GPU_ENGINE_LOCK": f"/tmp/seunghyun_lumerical_run057_gpu{gpu}.lock",
            "XDG_CONFIG_HOME": f"/tmp/seunghyun_lumerical_run057_gpu{gpu}",
            "MPLCONFIGDIR": f"/tmp/seunghyun_matplotlib_run057_gpu{gpu}",
        }
    )
    command = [
        sys.executable,
        "-u",
        "-m",
        "photothermal_pte.optimization_runs.tairte4_flake_topology.run_official_dfm_exact_repair_optimization",
        "--polarization",
        "Ea",
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
    if os.environ.get("RUN057_RESUME", "0") == "1":
        command.append("--resume")
    return subprocess.run(command, cwd=REPOSITORY, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
