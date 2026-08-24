#!/usr/bin/env python3
"""Reserve Lumerical licenses and run both dual-polarization cases."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.optimization_runs.launch_run061_062_dual_polarization_sequential import (
    write_state,
)


LAUNCHER = HERE / "launch_run061_062_dual_polarization_sequential.py"
RUNRES = Path("/home/dhkim/bin/runres")
SITE_RUN = Path("/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/run")
LOCK = Path("/tmp/seunghyun_run061_062_dual_runres.lock")


def main() -> int:
    gpu = int(os.environ.get("RUN061_062_GPU", "3"))
    stream = LOCK.open("w")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Run061/062 dual queue is already active") from error

    environment = dict(os.environ)
    environment.update(
        {
            "RUN061_062_GPU": str(gpu),
            "RUN061_062_RESUME": "1",
            "MSOPT_RUN_CMD": str(SITE_RUN),
            "EIDL_RUN_BUSY_GPU_UTIL_THRESHOLD": "0",
            "PYTHONUNBUFFERED": "1",
            "PATH": f"{SITE_RUN.parent}:{os.environ.get('PATH', '')}",
            "LD_LIBRARY_PATH": "/home/eidl/miniconda3/envs/EIDL-Lumapi/lib",
        }
    )
    command = [
        str(RUNRES),
        "--reserve-wait",
        "86400",
        "--reserve-count",
        "9",
        "--reserve-tag",
        f"run061_062_dual_grown_gpu{gpu}",
        str(LAUNCHER),
        "-th",
        "8",
        "-GPU",
        str(gpu),
    ]
    write_state(
        "waiting_for_runres_reservation",
        command=command,
        runres=str(RUNRES),
    )
    completed = subprocess.run(command, cwd=REPOSITORY, env=environment)
    write_state("runres_finished", returncode=completed.returncode)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
