#!/usr/bin/env python3
"""Hold one runres reservation while running the first selected-Z Q gate."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
RAW_ROOT = Path("/home/seunghyun/tairte4/raw_artifacts")
POLARIZATION = os.environ.get("Z_SELECTED_POLARIZATION", "CP_plus")
if POLARIZATION not in ("CP_plus", "CP_minus"):
    raise RuntimeError(f"invalid Z_SELECTED_POLARIZATION={POLARIZATION!r}")
OUTPUT = RAW_ROOT / "paper_tairte4_Z_M2_selected_Q_5p25um" / f"LH_{POLARIZATION}"
STATUS = RAW_ROOT / "Z_M2_SELECTED_Q_RUNRES_DRIVER_STATUS.json"
EXPECTED = "COMPLETED_Z2022_M2_RECONSTRUCTED_SELECTED_Q"


def gpu_device() -> str:
    value = os.environ.get("LUMERICAL_SESSION_GPU_DEVICE", "").strip()
    if not value.startswith("GPU "):
        raise RuntimeError("run this driver through runres with a fixed GPU")
    return value


def completed() -> bool:
    path = OUTPUT / "Z2022_M2_selected_Q.json"
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get("status") == EXPECTED
    except Exception:
        return False


def publish(status: str, **extra: object) -> None:
    STATUS.write_text(
        json.dumps(
            {
                "status": status,
                "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "gpu_device": gpu_device(),
                "LM_PROJECT_present": bool(os.environ.get("LM_PROJECT")),
                "output": str(OUTPUT),
                "wavelength_um": 5.25,
                "handedness": "LH",
                "polarization": POLARIZATION,
                **extra,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> int:
    if completed():
        publish("COMPLETED_Z_SELECTED_Q_RUNRES_DRIVER", reused=True)
        return 0
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        archive = OUTPUT.with_name(
            OUTPUT.name + time.strftime("_incomplete_%Y%m%dT%H%M%SZ", time.gmtime())
        )
        OUTPUT.rename(archive)
    command = [
        sys.executable,
        str(HERE / "41_run_v261_z2022_m2_selected_q.py"),
        "--output-dir",
        str(OUTPUT),
        "--gpu-device",
        gpu_device(),
        "--handedness",
        "LH",
        "--polarization",
        POLARIZATION,
        "--wavelength-um",
        "5.25",
        "--duration-ps",
        "4.0",
    ]
    publish("RUNNING_Z_SELECTED_Q_RUNRES_DRIVER", command=command)
    started = time.monotonic()
    process = subprocess.run(command, check=False)
    wall = time.monotonic() - started
    if process.returncode != 0 or not completed():
        publish("FAILED_Z_SELECTED_Q_RUNRES_DRIVER", command=command, returncode=process.returncode, wall_time_s=wall)
        return 1
    publish("COMPLETED_Z_SELECTED_Q_RUNRES_DRIVER", command=command, returncode=0, wall_time_s=wall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
