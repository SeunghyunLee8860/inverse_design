#!/usr/bin/env python3
"""Wait for successful Run047 completion, then reserve GPU 5 for Run048."""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
RUN047 = REPOSITORY / (
    "photothermal_pte/optimization_runs/"
    "run_047_left_right_electrodes_Ea_fresh_restart"
)
RUN047_FINAL = RUN047 / "results/FINAL_RESULT.json"
RUN047_STATE = RUN047 / "PIPELINE_STATE.json"
STATE = HERE / "QUEUE_STATE.json"
LOCK = Path("/tmp/seunghyun_run048_queue.lock")
RUNRES = Path("/home/dhkim/bin/runres")
RUNNER = Path("/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/run")
POLL_SECONDS = 30


def write_state(stage: str, **extra: object) -> None:
    payload = {
        "stage": stage,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "waiting_for": str(RUN047_FINAL),
        "target_gpu": 5,
        "reserved_license_count": 9,
        **extra,
    }
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(STATE)


def run047_failed() -> bool:
    if not RUN047_STATE.is_file():
        return False
    state = json.loads(RUN047_STATE.read_text())
    return state.get("status") == "failed" and not run047_recovery_active()


def run047_recovery_active() -> bool:
    marker = b"resume_exact_cleanup.py"
    for command_line in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            if marker in command_line.read_bytes():
                return True
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return False


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    stream = LOCK.open("w")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Run048 queue is already active") from error

    while not RUN047_FINAL.is_file():
        if run047_failed():
            write_state("blocked", reason="Run047 failed; fail-closed before Run048")
            return 1
        write_state("waiting_for_run047")
        time.sleep(POLL_SECONDS)

    run047 = json.loads(RUN047_FINAL.read_text())
    geometry_gate = run047.get("final_geometry_gate", {})
    if not geometry_gate.get("passed", False):
        write_state(
            "blocked",
            reason="Run047 ended without an exact-binary 500 nm geometry",
            run047_status=run047.get("status"),
        )
        return 1

    command = [
        str(RUNRES),
        "--reserve-wait", "86400",
        "--reserve-count", "9",
        "--reserve-tag", "run048_Eb_gpu5_fresh",
        str(HERE / "run_pipeline.py"),
        "-th", "8",
        "-GPU", "5",
    ]
    env = dict(os.environ)
    env.update(
        {
            "RUN048_GPU": "5",
            "EIDL_RUN_BUSY_GPU_UTIL_THRESHOLD": "0",
            "MSOPT_RUN_CMD": str(RUNNER),
            "PYTHONUNBUFFERED": "1",
        }
    )
    write_state(
        "starting_runres",
        command=command,
        run047_status=run047.get("status"),
        run047_objective_preservation_passed=run047.get("passed"),
    )
    completed = subprocess.run(command, cwd=REPOSITORY, env=env)
    write_state("runres_finished", returncode=completed.returncode)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
