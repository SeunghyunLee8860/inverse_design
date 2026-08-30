#!/usr/bin/env python3
"""Run the corrected-v3 M2 Ea closure control with local 12.5-nm Au-edge mesh."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "41_run_v261_z2022_m2_selected_q.py"
OUTPUT = Path("/home/seunghyun/tairte4/raw_artifacts/paper_z2022_m2_figure_period_corrected_Ea_5p3um_v3_edge12p5nm")


def main() -> int:
    spec = importlib.util.spec_from_file_location("z2022_v3_ea_edge12p5_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    sys.argv = [
        str(RUNNER),
        "--output-dir", str(OUTPUT),
        "--handedness", "LH",
        "--polarization", "y_a",
        "--geometry-variant", "figure_period_corrected_v3",
        "--wavelength-um", "5.3",
        "--duration-ps", "6.0",
        "--top-au-edge-mesh-nm", "12.5",
    ]
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
