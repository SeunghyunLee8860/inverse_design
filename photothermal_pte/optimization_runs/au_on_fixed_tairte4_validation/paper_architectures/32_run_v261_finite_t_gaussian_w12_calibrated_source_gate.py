#!/usr/bin/env python3
"""Run the second one-step calibration for the 12-um finite-T source gate."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


HERE = Path(__file__).resolve().parent
DRIVER = HERE / "28_run_v261_finite_t_gaussian_source_only.py"
RAW = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "paper_tairte4_finite_T_target_w0_12um_calibrated_source_only"
)

# Derived only from the prior saved source-only field:
# 11.9168648897 um * 12 um / 12.060005716259134 um.
SOURCE_OBJECT_W0_UM = 11.85757138436561


def main() -> None:
    gpu = os.environ.get("LUMERICAL_SESSION_GPU_DEVICE", "GPU 1")
    sys.argv = [
        str(DRIVER),
        "--output-dir",
        str(RAW),
        "--gpu-device",
        gpu,
        "--w0-um",
        f"{SOURCE_OBJECT_W0_UM:.14f}",
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
