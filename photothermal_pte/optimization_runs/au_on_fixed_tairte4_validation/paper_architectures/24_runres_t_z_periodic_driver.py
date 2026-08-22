#!/usr/bin/env python3
"""No-argument runres driver for the approved T then Z periodic screens."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
RAW_ROOT = Path("/home/seunghyun/tairte4/raw_artifacts")
T_ROOT = RAW_ROOT / "paper_tairte4_T_broadband_4_12"
Z_ROOT = RAW_ROOT / "paper_tairte4_Z_M2_broadband_4_12"
STATUS_PATH = RAW_ROOT / "T_Z_4_12_RUNRES_DRIVER_STATUS.json"


def gpu_device() -> str:
    value = os.environ.get("LUMERICAL_SESSION_GPU_DEVICE", "").strip()
    if not value.startswith("GPU "):
        raise RuntimeError("run this driver through runres/run so LUMERICAL_SESSION_GPU_DEVICE is fixed")
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
    target = directory.with_name(directory.name + f"_incomplete_{stamp}")
    directory.rename(target)


def publish_status(state: str, records: list[dict[str, object]], **extra: object) -> None:
    payload = {
        "status": state,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_device": gpu_device(),
        "LM_PROJECT_present": bool(os.environ.get("LM_PROJECT")),
        "records": records,
        **extra,
    }
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def run_case(command: list[str], output: Path, records: list[dict[str, object]]) -> None:
    full = command + ["--output-dir", str(output), "--gpu-device", gpu_device()]
    started = time.monotonic()
    publish_status("RUNNING_T_Z_4_12_RUNRES_DRIVER", records, current_command=full)
    completed_process = subprocess.run(full, check=False)
    record = {
        "command": full,
        "output": str(output),
        "returncode": int(completed_process.returncode),
        "wall_time_s": time.monotonic() - started,
    }
    records.append(record)
    if completed_process.returncode != 0:
        publish_status("FAILED_T_Z_4_12_RUNRES_DRIVER", records)
        raise RuntimeError(f"fail-closed case: {output}")


def main() -> int:
    device = gpu_device()
    records: list[dict[str, object]] = []
    publish_status("STARTING_T_Z_4_12_RUNRES_DRIVER", records)
    t_runner = str(HERE / "17_run_v261_t2024_periodic_broadband_rta.py")
    for key, polarization, omit in (
        ("T_Ea", "y_a", False),
        ("T_Eb", "x_b", False),
        ("bare_Ea", "y_a", True),
        ("bare_Eb", "x_b", True),
    ):
        directory = T_ROOT / key
        if completed(directory, "T2024_periodic_broadband_rta.json", "COMPLETED_T2024_PERIODIC_BROADBAND_RTA"):
            records.append({"output": str(directory), "status": "reused_completed_case"})
            continue
        archive_incomplete(directory)
        command = [sys.executable, t_runner, "--polarization", polarization]
        if omit:
            command.append("--omit-top-t-control")
        run_case(command, directory, records)

    z_runner = str(HERE / "19_run_v261_z2022_m2_periodic_broadband_rta.py")
    for handedness in ("LH", "RH"):
        for polarization in ("CP_plus", "CP_minus"):
            key = f"{handedness}_{polarization}"
            directory = Z_ROOT / key
            if completed(directory, "Z2022_M2_periodic_broadband_rta.json", "COMPLETED_Z2022_M2_PERIODIC_BROADBAND_RTA"):
                records.append({"output": str(directory), "status": "reused_completed_case"})
                continue
            archive_incomplete(directory)
            run_case(
                [sys.executable, z_runner, "--handedness", handedness, "--polarization", polarization],
                directory,
                records,
            )
    publish_status("COMPLETED_T_Z_4_12_RUNRES_DRIVER", records, reserved_gpu=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
