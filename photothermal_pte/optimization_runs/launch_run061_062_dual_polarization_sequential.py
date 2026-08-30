#!/usr/bin/env python3
"""Run the two shared-geometry dual-polarization designs on one GPU."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
CASES = (
    (61, HERE / "run_061_top_bottom_thermally_grown_sio2_dual_polarization"),
    (62, HERE / "run_062_diagonal45_thermally_grown_sio2_dual_polarization"),
)
STATE = HERE / "RUN061_062_DUAL_QUEUE_STATE.json"


def write_state(stage: str, **extra: object) -> None:
    payload = {
        "stage": stage,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "objective": "shared geometry smooth minimum of E||a and E||b current",
        "thermal_interface": "thermally_grown",
        "G_TaIrTe4_SiO2_W_m2K": 7.37e6,
        "target_gpu": int(os.environ.get("RUN061_062_GPU", "3")),
        "reserved_license_count": 9,
        **extra,
    }
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(STATE)


def main() -> int:
    gpu = int(os.environ.get("RUN061_062_GPU", "3"))
    resume = os.environ.get("RUN061_062_RESUME", "1") == "1"
    environment = dict(os.environ)
    for run, directory in CASES:
        final_path = directory / "results" / "FINAL_RESULT.json"
        if final_path.is_file():
            write_state("case_already_complete", run=run, final=str(final_path))
            continue
        environment[f"RUN{run:03d}_GPU"] = str(gpu)
        environment[f"RUN{run:03d}_RESUME"] = (
            "1"
            if resume and (directory / "results" / "RAW_ARTIFACT_MANIFEST.json").is_file()
            else "0"
        )
        write_state(
            "case_running",
            run=run,
            launcher=str(directory / "run.py"),
            resume=environment[f"RUN{run:03d}_RESUME"] == "1",
        )
        completed = subprocess.run(
            [sys.executable, "-u", str(directory / "run.py")],
            cwd=HERE.parents[1],
            env=environment,
        )
        if completed.returncode != 0:
            write_state("case_failed", run=run, returncode=completed.returncode)
            return completed.returncode
        write_state("case_complete", run=run, final=str(final_path))
    write_state("all_complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
