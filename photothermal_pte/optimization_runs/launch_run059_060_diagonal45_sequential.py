#!/usr/bin/env python3
"""Run the +45-degree E||a and E||b optimizations sequentially on one GPU."""

from __future__ import annotations

import os
from pathlib import Path
import json
import subprocess
import sys


HERE = Path(__file__).resolve().parent
RUNRES = Path("/home/dhkim/bin/runres")
SITE_RUN = Path("/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/run")
CASES = (
    (59, HERE / "run_059_diagonal45_evaporated_sio2_Ea_bounded_official_dfm_exact_repair" / "run.py"),
    (60, HERE / "run_060_diagonal45_evaporated_sio2_Eb_bounded_official_dfm_exact_repair" / "run.py"),
)


def main() -> int:
    gpu = int(os.environ.get("RUN059_060_GPU", "0"))
    resume = os.environ.get("RUN059_060_RESUME", "0") == "1"
    environment = dict(os.environ)
    for run, launcher in CASES:
        environment[f"RUN{run:03d}_GPU"] = str(gpu)
        published = launcher.parent / "results_v6_fixed_TaIrTe4_contact_no_Au"
        final_path = published / "FINAL_RESULT.json"
        if resume and final_path.is_file():
            final = json.loads(final_path.read_text())
            if final.get("passed"):
                continue
        has_checkpoint = (published / "RAW_ARTIFACT_MANIFEST.json").is_file()
        environment[f"RUN{run:03d}_RESUME"] = (
            "1" if resume and has_checkpoint else "0"
        )
        if run == 60:
            environment["MSOPT_RUN_CMD"] = str(SITE_RUN)
            command = [
                str(RUNRES),
                "--reserve-wait", "86400",
                "--reserve-count", "9",
                "--reserve-tag", f"run060_rotated45_v6_gpu{gpu}",
                str(launcher),
                "-th", "8",
                "-GPU", str(gpu),
            ]
        else:
            command = [sys.executable, "-u", str(launcher)]
        completed = subprocess.run(command, cwd=HERE.parents[1], env=environment)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
