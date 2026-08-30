#!/usr/bin/env python3
"""Run the four finite-array T11x15/Z1x3 Ea/Eb Maxwell cases sequentially.

This process is intended to be launched once under ``runres`` so the same
FlexNet reservation remains held between cases.  A validated immutable case is
skipped; an incomplete or failed existing case stops the suite fail-closed.
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
PROGRESS = RAW_ROOT / "FINITE_T_Z_ARRAY_Q_SUITE_PROGRESS.json"
CASES = (("T", "Ea"), ("T", "Eb"), ("Z", "Ea"), ("Z", "Eb"))


def label(architecture: str) -> str:
    return "T11x15" if architecture == "T" else "Z1x3"


def output_dir(architecture: str, polarization: str) -> Path:
    return RAW_ROOT / f"{label(architecture)}_{polarization}_Au_on"


def result_path(architecture: str, polarization: str) -> Path:
    return output_dir(architecture, polarization) / f"FINITE_{architecture}_{polarization}_Au_on_Q.json"


def validated(architecture: str, polarization: str) -> bool:
    path = result_path(architecture, polarization)
    if not path.exists():
        return False
    payload = json.loads(path.read_text())
    return str(payload.get("status", "")).startswith("VALIDATED_FINITE_")


def write_progress(records: list[dict[str, object]], status: str) -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = PROGRESS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"status": status, "cases": records}, indent=2) + "\n")
    temporary.replace(PROGRESS)


def main() -> int:
    records: list[dict[str, object]] = []
    write_progress(records, "RUNNING_FINITE_T_Z_ARRAY_Q_SUITE")
    for architecture, polarization in CASES:
        item: dict[str, object] = {
            "architecture": architecture,
            "variant": label(architecture),
            "polarization": polarization,
            "output": str(output_dir(architecture, polarization)),
        }
        if validated(architecture, polarization):
            item["status"] = "SKIPPED_IMMUTABLE_VALIDATED_CASE"
            records.append(item)
            write_progress(records, "RUNNING_FINITE_T_Z_ARRAY_Q_SUITE")
            continue
        directory = output_dir(architecture, polarization)
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(f"refusing incomplete/failed existing output: {directory}")
        environment = os.environ.copy()
        environment.update(
            {
                "ARRAY_ARCHITECTURE": architecture,
                "ARRAY_POLARIZATION": polarization,
                "ARRAY_Q_OUTPUT": str(directory),
            }
        )
        started = time.monotonic()
        completed = subprocess.run([sys.executable, str(RUNNER)], env=environment, check=False)
        item["wall_time_s"] = time.monotonic() - started
        item["returncode"] = completed.returncode
        item["status"] = "VALIDATED" if completed.returncode == 0 and validated(architecture, polarization) else "FAILED"
        records.append(item)
        write_progress(records, "RUNNING_FINITE_T_Z_ARRAY_Q_SUITE")
        if item["status"] != "VALIDATED":
            write_progress(records, "FAILED_FINITE_T_Z_ARRAY_Q_SUITE")
            return 1
    write_progress(records, "VALIDATED_FINITE_T_Z_ARRAY_Q_SUITE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
