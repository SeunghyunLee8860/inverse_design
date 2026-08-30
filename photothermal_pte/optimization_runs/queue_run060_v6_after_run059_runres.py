#!/usr/bin/env python3
"""Wait for corrected Run059, then hold nine licences for all of Run060."""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
RUN059 = HERE / "run_059_diagonal45_evaporated_sio2_Ea_bounded_official_dfm_exact_repair"
RUN060 = HERE / "run_060_diagonal45_evaporated_sio2_Eb_bounded_official_dfm_exact_repair"
RESULTS_NAME = "results_v6_fixed_TaIrTe4_contact_no_Au"
RUN059_FINAL = RUN059 / RESULTS_NAME / "FINAL_RESULT.json"
RUN060_FINAL = RUN060 / RESULTS_NAME / "FINAL_RESULT.json"
RUN060_MANIFEST = RUN060 / RESULTS_NAME / "RAW_ARTIFACT_MANIFEST.json"
RUN060_LAUNCHER = RUN060 / "run.py"
RUNRES = Path("/home/dhkim/bin/runres")
SITE_RUN = Path("/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/run")
STATE = RUN060 / "RUNRES_QUEUE_STATE.json"
LOCK = Path("/tmp/seunghyun_run060_v6_runres_queue.lock")
POLL_SECONDS = 30


def write_state(stage: str, **extra: object) -> None:
    payload = {
        "stage": stage,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "waiting_for": str(RUN059_FINAL),
        "target_gpu": 3,
        "reserved_license_count": 9,
        **extra,
    }
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(STATE)


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    stream = LOCK.open("w")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Run060 v6 runres queue is already active") from error

    while not RUN059_FINAL.is_file():
        write_state("waiting_for_run059")
        time.sleep(POLL_SECONDS)

    if RUN060_FINAL.is_file():
        write_state("already_complete")
        return 0

    environment = dict(os.environ)
    environment.update(
        {
            "RUN060_GPU": "3",
            "RUN060_RESUME": "1" if RUN060_MANIFEST.is_file() else "0",
            "MSOPT_RUN_CMD": str(SITE_RUN),
            "EIDL_RUN_BUSY_GPU_UTIL_THRESHOLD": "0",
            "PYTHONUNBUFFERED": "1",
            "PATH": f"{SITE_RUN.parent}:{os.environ.get('PATH', '')}",
            "LD_LIBRARY_PATH": "/home/eidl/miniconda3/envs/EIDL-Lumapi/lib",
        }
    )
    command = [
        str(RUNRES),
        "--reserve-wait", "86400",
        "--reserve-count", "9",
        "--reserve-tag", "run060_rotated45_v6_gpu3",
        str(RUN060_LAUNCHER),
        "-th", "8",
        "-GPU", "3",
    ]
    write_state(
        "waiting_for_runres_reservation",
        command=command,
        resume=environment["RUN060_RESUME"] == "1",
    )
    completed = subprocess.run(command, cwd=REPOSITORY, env=environment)
    write_state("runres_finished", returncode=completed.returncode)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
