#!/usr/bin/env python3
"""Run the corrected-cell planar-stack Ea closure isolation control."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
OUTPUT = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "paper_z2022_m2_corrected_cell_planar_Ea_5p3um_closure_control"
)


def main() -> int:
    path = HERE / "41_run_v261_z2022_m2_selected_q.py"
    spec = importlib.util.spec_from_file_location("z_planar_ea_control", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    old = sys.argv[:]
    try:
        sys.argv = [
            str(path),
            "--output-dir", str(OUTPUT),
            "--gpu-device", os.environ.get("LUMERICAL_SESSION_GPU_DEVICE", "GPU 5"),
            "--handedness", "LH",
            "--polarization", "y_a",
            "--geometry-variant", "figure_axis_corrected_v2",
            "--wavelength-um", "5.3",
            "--duration-ps", "6.0",
            "--omit-top-au-control",
        ]
        return int(module.main())
    finally:
        sys.argv = old


if __name__ == "__main__":
    raise SystemExit(main())
