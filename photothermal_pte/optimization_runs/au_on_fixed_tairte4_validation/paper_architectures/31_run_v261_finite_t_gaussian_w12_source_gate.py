#!/usr/bin/env python3
"""Run the calibrated 12-um-waist finite multi-T source-only gate."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


HERE = Path(__file__).resolve().parent
DRIVER = HERE / "28_run_v261_finite_t_gaussian_source_only.py"
RAW = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "paper_tairte4_finite_T_target_w0_12um_source_only"
)

# SHA-pinned one-step source-object calibration previously validated for a
# physical 12-um target waist. It changes only the source-object parameter;
# it is not incident-power or Q rescaling.
SOURCE_OBJECT_W0_UM = 11.9168648897


def main() -> None:
    gpu = os.environ.get("LUMERICAL_SESSION_GPU_DEVICE", "GPU 1")
    sys.argv = [
        str(DRIVER),
        "--output-dir",
        str(RAW),
        "--gpu-device",
        gpu,
        "--w0-um",
        f"{SOURCE_OBJECT_W0_UM:.10f}",
        "--target-w0-um",
        "12.0",
        "--domain-x-um",
        "60.0",
        "--domain-y-um",
        "60.0",
        "--array-x-um",
        "16.5",
        "--array-y-um",
        "17.0",
        "--source-span-um",
        "50.0",
    ]
    runpy.run_path(str(DRIVER), run_name="__main__")


if __name__ == "__main__":
    main()
