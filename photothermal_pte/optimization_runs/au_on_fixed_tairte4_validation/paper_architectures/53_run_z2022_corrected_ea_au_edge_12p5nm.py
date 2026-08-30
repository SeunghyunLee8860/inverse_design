#!/usr/bin/env python3
"""Corrected Z Ea closure/convergence case with 12.5-nm Au-edge mesh."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
OUTPUT = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "paper_z2022_m2_corrected_Ea_au_edge_12p5nm"
)


def main() -> int:
    path = HERE / "41_run_v261_z2022_m2_selected_q.py"
    spec = importlib.util.spec_from_file_location("z_ea_edge_12p5", path)
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
            "--top-au-edge-mesh-nm", "12.5",
        ]
        return int(module.main())
    finally:
        sys.argv = old


if __name__ == "__main__":
    raise SystemExit(main())
