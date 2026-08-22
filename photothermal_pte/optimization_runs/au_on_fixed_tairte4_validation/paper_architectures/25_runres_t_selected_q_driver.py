#!/usr/bin/env python3
"""Reserved-license driver for the selected inverse-T volumetric-Q case."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
RAW_ROOT = Path("/home/seunghyun/tairte4/raw_artifacts")
OUTPUT = RAW_ROOT / "paper_tairte4_T_selected_Q_11p825um_Eb"
STATUS_PATH = RAW_ROOT / "T_SELECTED_Q_RUNRES_DRIVER_STATUS.json"
EXPECTED_STATUS = "COMPLETED_T2024_TAIRTE4_OPTICAL_SMOKE"


def gpu_device() -> str:
    value = os.environ.get("LUMERICAL_SESSION_GPU_DEVICE", "").strip()
    if not value.startswith("GPU "):
        raise RuntimeError("run this driver through runres/run with a fixed GPU")
    return value


def completed() -> bool:
    path = OUTPUT / "T2024_TaIrTe4_optical_smoke.json"
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get("status") == EXPECTED_STATUS
    except Exception:
        return False


def archive_incomplete() -> None:
    if not OUTPUT.exists() or not any(OUTPUT.iterdir()):
        return
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    OUTPUT.rename(OUTPUT.with_name(OUTPUT.name + f"_incomplete_{stamp}"))


def publish(status: str, **extra: object) -> None:
    payload = {
        "status": status,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_device": gpu_device(),
        "LM_PROJECT_present": bool(os.environ.get("LM_PROJECT")),
        "output": str(OUTPUT),
        "selected_wavelength_um": 11.825,
        "polarization": "x_b",
        **extra,
    }
    STATUS_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    if completed():
        publish("COMPLETED_T_SELECTED_Q_RUNRES_DRIVER", reused=True)
        return 0
    archive_incomplete()
    command = [
        sys.executable,
        str(HERE / "07_run_v261_t2024_tairte4_optical_smoke.py"),
        "--wavelength-um",
        "11.825",
        "--polarization",
        "x_b",
        "--substrate-mode",
        "sio2_si_reduced_285nm",
        "--duration-ps",
        "1.0",
        "--output-dir",
        str(OUTPUT),
        "--gpu-device",
        gpu_device(),
    ]
    publish("RUNNING_T_SELECTED_Q_RUNRES_DRIVER", command=command)
    started = time.monotonic()
    process = subprocess.run(command, check=False)
    wall_time = time.monotonic() - started
    if process.returncode != 0 or not completed():
        publish(
            "FAILED_T_SELECTED_Q_RUNRES_DRIVER",
            command=command,
            returncode=int(process.returncode),
            wall_time_s=wall_time,
        )
        return 1
    publish(
        "COMPLETED_T_SELECTED_Q_RUNRES_DRIVER",
        command=command,
        returncode=0,
        wall_time_s=wall_time,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
