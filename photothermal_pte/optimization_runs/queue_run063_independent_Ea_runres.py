#!/usr/bin/env python3
"""Reserve nine Lumerical licenses and run independent E||a Run063."""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
RUN_DIRECTORY = HERE / "run_063_diagonal45_thermally_grown_sio2_Ea_independent"
LAUNCHER = RUN_DIRECTORY / "run.py"
RUNRES = Path("/home/dhkim/bin/runres")
SITE_RUN = Path("/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/run")
LOCK = Path("/tmp/seunghyun_run063_independent_Ea_runres.lock")
STATE = RUN_DIRECTORY / "RUN063_STATE.json"
LOG = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_rotated45_edge_contact_anchored/"
    "run063_diagonal45_single_Ea_thermally_grown_runres.log"
)


def write_state(stage: str, **extra: object) -> None:
    payload = {
        "stage": stage,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "run": 63,
        "objective_mode": "single",
        "objective": "maximize signed E||a terminal current",
        "geometry": "diagonal_45_contact_anchored",
        "thermal_interface": "thermally_grown",
        "G_TaIrTe4_SiO2_W_m2K": 7.37e6,
        "target_gpu": int(os.environ.get("RUN063_GPU", "3")),
        "reserved_license_count": 9,
        "log": str(LOG),
        **extra,
    }
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(STATE)


def main() -> int:
    gpu = int(os.environ.get("RUN063_GPU", "3"))
    stream = LOCK.open("w")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Run063 runres queue is already active") from error

    environment = dict(os.environ)
    environment.update(
        {
            "RUN063_GPU": str(gpu),
            "RUN063_RESUME": "1" if (RUN_DIRECTORY / "results/RAW_ARTIFACT_MANIFEST.json").is_file() else "0",
            "MSOPT_RUN_CMD": str(SITE_RUN),
            "LUM_RESERVE_MODULE_DIR": "/home/dhkim/dhkim_module",
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
        f"run063_diagonal45_single_Ea_grown_gpu{gpu}",
        str(LAUNCHER),
        "-th",
        "8",
        "-GPU",
        str(gpu),
    ]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    write_state("waiting_for_runres_reservation", command=command)
    with LOG.open("a") as log:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    write_state("runres_finished", returncode=completed.returncode)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
