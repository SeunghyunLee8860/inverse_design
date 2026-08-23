#!/usr/bin/env python3
"""Run the finite T11x15 +45/-45 linear-polarization Maxwell cases.

The validated Ea/Eb array artifacts are immutable inputs for comparison.  This
driver adds only the two coherent linear polarization angles required by the
paper-like inverse-T diagnostic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "84_run_v261_finite_t_z_array_gaussian_q.py"
RAW_ROOT = Path("/home/seunghyun/tairte4/raw_artifacts/finite_T_Z_array_Q")
PROGRESS = RAW_ROOT / "FINITE_T_DIAGONAL_Q_SUITE_PROGRESS.json"
CASES = ("linear_plus_45", "linear_minus_45")


def output_dir(polarization: str) -> Path:
    return RAW_ROOT / f"T11x15_{polarization}_Au_on"


def result_path(polarization: str) -> Path:
    return output_dir(polarization) / f"FINITE_T_{polarization}_Au_on_Q.json"


def validated(polarization: str) -> bool:
    path = result_path(polarization)
    if not path.exists():
        return False
    return str(json.loads(path.read_text()).get("status", "")).startswith("VALIDATED_FINITE_")


def write_progress(records: list[dict[str, object]], status: str) -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = PROGRESS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"status": status, "cases": records}, indent=2) + "\n")
    temporary.replace(PROGRESS)


def main() -> int:
    records: list[dict[str, object]] = []
    write_progress(records, "RUNNING_FINITE_T_DIAGONAL_Q_SUITE")
    for polarization in CASES:
        item: dict[str, object] = {
            "architecture": "T",
            "variant": "T11x15",
            "polarization": polarization,
            "output": str(output_dir(polarization)),
        }
        if validated(polarization):
            item["status"] = "SKIPPED_IMMUTABLE_VALIDATED_CASE"
            records.append(item)
            write_progress(records, "RUNNING_FINITE_T_DIAGONAL_Q_SUITE")
            continue
        directory = output_dir(polarization)
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(f"refusing incomplete/failed existing output: {directory}")
        environment = os.environ.copy()
        environment.update(
            {
                "ARRAY_ARCHITECTURE": "T",
                "ARRAY_POLARIZATION": polarization,
                "ARRAY_Q_OUTPUT": str(directory),
            }
        )
        started = time.monotonic()
        completed = subprocess.run([sys.executable, str(RUNNER)], env=environment, check=False)
        item["wall_time_s"] = time.monotonic() - started
        item["returncode"] = completed.returncode
        item["status"] = "VALIDATED" if completed.returncode == 0 and validated(polarization) else "FAILED"
        records.append(item)
        write_progress(records, "RUNNING_FINITE_T_DIAGONAL_Q_SUITE")
        if item["status"] != "VALIDATED":
            write_progress(records, "FAILED_FINITE_T_DIAGONAL_Q_SUITE")
            return 1
    write_progress(records, "VALIDATED_FINITE_T_DIAGONAL_Q_SUITE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
