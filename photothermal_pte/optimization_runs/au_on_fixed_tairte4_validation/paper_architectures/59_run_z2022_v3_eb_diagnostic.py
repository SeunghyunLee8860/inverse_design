#!/usr/bin/env python3
"""Run corrected-v3 M2 Eb even when the paired optical closure remains diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "41_run_v261_z2022_m2_selected_q.py"
OUTPUT = Path("/home/seunghyun/tairte4/raw_artifacts/paper_z2022_m2_figure_period_corrected_Eb_5p3um_v3")


def main() -> int:
    spec = importlib.util.spec_from_file_location("z2022_v3_eb_diagnostic_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    sys.argv = [
        str(RUNNER),
        "--output-dir", str(OUTPUT),
        "--handedness", "LH",
        "--polarization", "x_b",
        "--geometry-variant", "figure_period_corrected_v3",
        "--wavelength-um", "5.3",
        "--duration-ps", "6.0",
    ]
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
