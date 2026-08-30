#!/usr/bin/env python3
"""Wait for an unclaimed licensed GPU, then run T and Z periodic screens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
PYTHON = Path("/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python")
LUMCHECK = Path("/home/dhkim/bin/lumcheck")
GPU_PREFERENCE = (5, 1, 4, 0, 3, 6, 2, 7)


def available_gpus() -> tuple[list[int], str]:
    completed = subprocess.run(
        [str(LUMCHECK)],
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    output = completed.stdout + completed.stderr
    match = re.search(r"Available GPUs:\s*([^\n]+)", output)
    if match is None or match.group(1).strip().lower().startswith("none"):
        return [], output
    found = [int(value) for value in re.findall(r"\d+", match.group(1))]
    ordered = [gpu for gpu in GPU_PREFERENCE if gpu in found]
    return ordered, output


def wait_for_gpu(poll_s: float) -> int:
    while True:
        gpus, output = available_gpus()
        if gpus:
            gpu = gpus[0]
            print(f"QUEUE_GPU_AVAILABLE GPU {gpu}", flush=True)
            return gpu
        current = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        active = [line for line in output.splitlines() if "solving" in line]
        print(f"QUEUE_WAIT {current}; active={len(active)}; no unclaimed licensed GPU", flush=True)
        time.sleep(poll_s)


def run_case(command: list[str], output: Path, poll_s: float) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty case directory: {output}")
    gpu = wait_for_gpu(poll_s)
    full = command + ["--output-dir", str(output), "--gpu-device", f"GPU {gpu}"]
    print("QUEUE_START " + " ".join(full), flush=True)
    started = time.monotonic()
    completed = subprocess.run(full, check=False)
    record = {
        "output": str(output),
        "gpu": gpu,
        "returncode": completed.returncode,
        "wall_time_s": time.monotonic() - started,
        "command": full,
    }
    print("QUEUE_RESULT " + json.dumps(record), flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"case failed closed: {output}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-raw-root", type=Path, required=True)
    parser.add_argument("--z-raw-root", type=Path, required=True)
    parser.add_argument("--poll-s", type=float, default=20.0)
    args = parser.parse_args()
    t_root = args.t_raw_root.expanduser().resolve()
    z_root = args.z_raw_root.expanduser().resolve()
    records: list[dict[str, object]] = []

    t_runner = str(HERE / "17_run_v261_t2024_periodic_broadband_rta.py")
    for key, polarization, omit in (
        ("T_Ea", "y_a", False),
        ("T_Eb", "x_b", False),
        ("bare_Ea", "y_a", True),
        ("bare_Eb", "x_b", True),
    ):
        command = [str(PYTHON), t_runner, "--polarization", polarization]
        if omit:
            command.append("--omit-top-t-control")
        records.append(run_case(command, t_root / key, args.poll_s))

    z_runner = str(HERE / "19_run_v261_z2022_m2_periodic_broadband_rta.py")
    for handedness in ("LH", "RH"):
        for polarization in ("CP_plus", "CP_minus"):
            key = f"{handedness}_{polarization}"
            command = [
                str(PYTHON),
                z_runner,
                "--handedness",
                handedness,
                "--polarization",
                polarization,
            ]
            records.append(run_case(command, z_root / key, args.poll_s))

    summary = {"status": "COMPLETED_T_AND_Z_PERIODIC_SCREENING_QUEUE", "records": records}
    queue_summary = t_root.parent / "T_Z_periodic_screening_queue_summary.json"
    queue_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
