#!/usr/bin/env python3
"""Run one T or Z six-polarization periodic optical suite under runres."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
RAW_ROOT = Path("/home/seunghyun/tairte4/raw_artifacts/periodic_T_Z_six_polarization_20260822")
POLS = (
    "x_b", "y_a", "linear_plus_45", "linear_minus_45", "CP_plus", "CP_minus"
)


def gpu_device() -> str:
    value = os.environ.get("LUMERICAL_SESSION_GPU_DEVICE", "").strip()
    if not value.startswith("GPU "):
        raise RuntimeError("driver must be launched through runres")
    return value


def completed(directory: Path, json_name: str, expected: str) -> bool:
    path = directory / json_name
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get("status") == expected
    except Exception:
        return False


def archive_incomplete(directory: Path) -> None:
    if not directory.exists() or not any(directory.iterdir()):
        return
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    directory.rename(directory.with_name(directory.name + f"_incomplete_{stamp}"))


def main() -> int:
    architecture = os.environ.get("PERIODIC_ARCHITECTURE", "").strip().upper()
    if architecture not in ("T", "Z"):
        raise RuntimeError("set PERIODIC_ARCHITECTURE=T or Z before launching through runres")
    root = RAW_ROOT / architecture
    status_path = root / "RUNRES_SUITE_STATUS.json"
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    if architecture == "T":
        runner = HERE / "17_run_v261_t2024_periodic_broadband_rta.py"
        json_name = "T2024_periodic_broadband_rta.json"
        expected = "COMPLETED_T2024_PERIODIC_BROADBAND_RTA"
    else:
        runner = HERE / "19_run_v261_z2022_m2_periodic_broadband_rta.py"
        json_name = "Z2022_M2_periodic_broadband_rta.json"
        expected = "COMPLETED_Z2022_M2_PERIODIC_BROADBAND_RTA"

    def publish(state: str, current: list[str] | None = None) -> None:
        status_path.write_text(
            json.dumps(
                {
                    "status": state,
                    "architecture": architecture,
                    "gpu_device": gpu_device(),
                    "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "scope": "periodic optical R/T/A only; no thermal/weighting/PTE",
                    "current_command": current,
                    "records": records,
                },
                indent=2,
            )
            + "\n"
        )

    publish("STARTING_PERIODIC_SIX_POLARIZATION_SUITE")
    for polarization in POLS:
        output = root / polarization
        if completed(output, json_name, expected):
            records.append({"polarization": polarization, "output": str(output), "status": "reused"})
            continue
        archive_incomplete(output)
        command = [
            sys.executable, str(runner), "--polarization", polarization,
            "--output-dir", str(output), "--gpu-device", gpu_device(),
        ]
        if architecture == "Z":
            command.extend(
                ["--handedness", "LH", "--geometry-variant", "centered_expanded_supercell_v4"]
            )
        publish("RUNNING_PERIODIC_SIX_POLARIZATION_SUITE", command)
        started = time.monotonic()
        result = subprocess.run(command, check=False)
        records.append(
            {
                "polarization": polarization,
                "output": str(output),
                "returncode": int(result.returncode),
                "wall_time_s": time.monotonic() - started,
            }
        )
        if result.returncode != 0:
            publish("FAILED_PERIODIC_SIX_POLARIZATION_SUITE", command)
            raise RuntimeError(f"fail-closed {architecture} {polarization}")
    publish("COMPLETED_PERIODIC_SIX_POLARIZATION_SUITE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
