#!/usr/bin/env python3
"""Reserved-license wrapper for the w0=8.5-um finite-T source gate."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def main() -> int:
    device = os.environ.get("LUMERICAL_SESSION_GPU_DEVICE", "").strip()
    if not device.startswith("GPU "):
        raise RuntimeError("run through runres with a reserved GPU")
    command = [
        sys.executable,
        str(HERE / "28_run_v261_finite_t_gaussian_source_only.py"),
        "--gpu-device",
        device,
        "--w0-um",
        "8.5",
        "--domain-x-um",
        "46.5",
        "--domain-y-um",
        "47.0",
        "--array-x-um",
        "34.5",
        "--array-y-um",
        "35.0",
        "--source-span-um",
        "34.0",
        "--output-dir",
        "/home/seunghyun/tairte4/raw_artifacts/paper_tairte4_finite_T_w0_8p5um_source_only",
    ]
    return int(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
