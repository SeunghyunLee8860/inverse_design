#!/usr/bin/env python3
"""Keep one reserved license claim across Run051 cleanup and fresh fast Run052."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]


def run(command: list[str], environment: dict[str, str]) -> None:
    completed = subprocess.run(command, cwd=REPOSITORY, env=environment)
    if completed.returncode != 0:
        raise RuntimeError(f"pipeline command failed with {completed.returncode}: {command}")


def main() -> int:
    gpu = int(os.environ.get("RUN051_CLEANUP_GPU", "1"))
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["RUN051_CLEANUP_GPU"] = str(gpu)
    run([sys.executable, str(HERE / "run_user_forced_Ea_cleanup.py")], environment)
    environment.update(
        {
            "CONSTRAINT_AWARE_POLARIZATION": "Eb",
            "CONSTRAINT_AWARE_GPU": str(gpu),
            "CONSTRAINT_AWARE_GENERATION": "v4",
            "CONSTRAINT_AWARE_FAST_CONTINUATION": "1",
        }
    )
    run([sys.executable, str(HERE / "run_one.py")], environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
