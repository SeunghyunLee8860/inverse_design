#!/usr/bin/env python3
"""Run the +45-degree E||a and E||b optimizations sequentially on one GPU."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
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
        environment[f"RUN{run:03d}_RESUME"] = "1" if resume else "0"
        completed = subprocess.run(
            [sys.executable, "-u", str(launcher)],
            cwd=HERE.parents[1],
            env=environment,
        )
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
