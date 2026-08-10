#!/usr/bin/env python3
"""Run fresh Run026 Ea and Run027 Eb with AUGLAG/L-BFGS on GPU 1."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
PARENT = Path("/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored")
BASE_FSP = PARENT / "run012_uniform_rho0p5_Ea_forward_retry_20260810/tairte4_flake_forward_Ea.fsp"
BASE_SHA256 = "454fa83bc918b4db0e25d28f7debf23de38977038bd716c8d7dc539d6b3e3d83"
JACOBIAN = PARENT / "run012_component_yee_jacobian_retry_20260810"
GPU = int(os.environ.get("TAIRTE4_AUGLAG_GPU", "1"))
STATUS = REPOSITORY / "photothermal_pte/optimization_runs/AUGLAG_LBFGS_DUAL_RUN_STATUS.json"
RUNS = (
    (
        "Run026", "Ea", PARENT / "run026_auglag_lbfgs_Ea_20260810",
        REPOSITORY / "photothermal_pte/optimization_runs/run_026_auglag_lbfgs_contact_anchored_Ea_current_max",
    ),
    (
        "Run027", "Eb", PARENT / "run027_auglag_lbfgs_Eb_20260810",
        REPOSITORY / "photothermal_pte/optimization_runs/run_027_auglag_lbfgs_contact_anchored_Eb_current_max",
    ),
)


def write_status(status: str, **values: object) -> None:
    payload = {
        "schema": "auglag-lbfgs-dual-supervisor-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status, "gpu": GPU, **values,
    }
    temporary = STATUS.with_suffix(STATUS.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(STATUS)


def passed(path: Path) -> bool:
    return path.is_file() and bool(json.loads(path.read_text()).get("passed"))


def main() -> int:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(GPU)
    environment["TAIRTE4_TOPOLOGY_GEOMETRY"] = "contact_anchored"
    library = "/home/eidl/miniconda3/envs/EIDL-Lumapi/lib"
    environment["LD_LIBRARY_PATH"] = library + (":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
    for label, polarization, raw, published in RUNS:
        final = published / "FINAL_RESULT.json"
        if passed(final):
            continue
        if raw.exists() and any(raw.iterdir()):
            raise RuntimeError(f"refusing nonempty fresh raw root: {raw}")
        raw.mkdir(parents=True, exist_ok=True)
        published.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "-m",
            "photothermal_pte.optimization_runs.tairte4_flake_topology.run_auglag_lbfgs_optimization",
            "--polarization", polarization, "--raw-root", str(raw),
            "--published-dir", str(published), "--gpu", str(GPU),
            "--base-fsp", str(BASE_FSP), "--base-sha256", BASE_SHA256,
            "--jacobian-dir", str(JACOBIAN), "--constraint-device", "cuda:0",
        ]
        write_status("RUNNING_AUGLAG_LBFGS", run=label, polarization=polarization, command=command)
        completed = subprocess.run(command, cwd=REPOSITORY, env=environment)
        if completed.returncode or not passed(final):
            write_status("BLOCKED_AUGLAG_LBFGS", run=label, polarization=polarization, returncode=completed.returncode)
            raise RuntimeError(f"{label} AUGLAG/L-BFGS failed closed")
    write_status("VALIDATED_AUGLAG_LBFGS_EA_EB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
