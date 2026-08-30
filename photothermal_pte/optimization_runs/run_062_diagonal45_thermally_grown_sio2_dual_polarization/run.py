#!/usr/bin/env python3
"""Launch one +45-degree structure jointly optimized for E||a and E||b."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
ARTIFACT_ROOT = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_rotated45_edge_contact_anchored"
)
RAW = ARTIFACT_ROOT / "run062_diagonal45_dual_softmin_thermally_grown_v1"
BASE_FSP = ARTIFACT_ROOT / (
    "base_uniform_forward_Ea_v23_run58_layout_ideal_terminal_no_Au_rho05/"
    "tairte4_flake_forward_Ea.fsp"
)
BASE_SHA256 = "90f41beb8df62c22177e1387646a734a369a170ce682bc8d6748ce598f0571d7"
JACOBIAN = ARTIFACT_ROOT / (
    "component_yee_jacobian_v2_run58_optical_ideal_terminal_no_Au"
)


def main() -> int:
    gpu = int(os.environ.get("RUN062_GPU", "3"))
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "TAIRTE4_TOPOLOGY_GEOMETRY": "diagonal_45_contact_anchored",
            "TAIRTE4_SIO2_INTERFACE_SCENARIO": "thermally_grown",
            "LUMERICAL_LICENSE_RETRY_SECONDS": "30",
            "LUMERICAL_GPU_ENGINE_LOCK": f"/tmp/seunghyun_lumerical_run062_gpu{gpu}.lock",
            "XDG_CONFIG_HOME": f"/tmp/seunghyun_lumerical_run062_gpu{gpu}",
            "MPLCONFIGDIR": f"/tmp/seunghyun_matplotlib_run062_gpu{gpu}",
        }
    )
    command = [
        sys.executable,
        "-u",
        "-m",
        "photothermal_pte.optimization_runs.tairte4_flake_topology.run_official_dfm_exact_repair_optimization",
        "--polarization",
        "Ea",
        "--objective-mode",
        "dual_softmin",
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
    if os.environ.get("RUN062_RESUME", "0") == "1":
        command.append("--resume")
    return subprocess.run(command, cwd=REPOSITORY, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
