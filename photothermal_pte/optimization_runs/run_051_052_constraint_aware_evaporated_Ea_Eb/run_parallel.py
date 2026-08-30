#!/usr/bin/env python3
"""Launch fresh constraint-aware Ea/Eb optimizations on distinct GPUs."""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser()
    # Defaults allow this file to be launched by the site `runres` wrapper,
    # whose outer `run` CLI accepts only its own -th/-GPU options.  Direct
    # invocations may still override the pair.
    parser.add_argument("--gpu-ea", type=int, default=0)
    parser.add_argument("--gpu-eb", type=int, default=4)
    parser.add_argument("--status", type=Path, default=HERE / "PARALLEL_STATUS.json")
    args = parser.parse_args()
    if args.gpu_ea == args.gpu_eb:
        raise RuntimeError("Ea and Eb require distinct physical GPUs")
    command = [
        sys.executable,
        "-u",
        "-m",
        "photothermal_pte.optimization_runs.run_ansys_dfm_parallel_optimizations",
        "--gpu-ea", str(args.gpu_ea),
        "--gpu-eb", str(args.gpu_eb),
        "--raw-ea", str(ARTIFACT_ROOT / "run051_constraint_aware_Ea_evaporated_v2"),
        "--raw-eb", str(ARTIFACT_ROOT / "run052_constraint_aware_Eb_evaporated_v2"),
        "--published-ea", str(HERE / "run_051_Ea_results_v2"),
        "--published-eb", str(HERE / "run_052_Eb_results_v2"),
        "--base-fsp", str(BASE_FSP),
        "--base-sha256", BASE_SHA256,
        "--jacobian-dir", str(JACOBIAN),
        "--status", str(args.status.expanduser().resolve()),
        "--poll-seconds", "20",
    ]
    return subprocess.run(command, cwd=REPOSITORY).returncode


if __name__ == "__main__":
    raise SystemExit(main())
