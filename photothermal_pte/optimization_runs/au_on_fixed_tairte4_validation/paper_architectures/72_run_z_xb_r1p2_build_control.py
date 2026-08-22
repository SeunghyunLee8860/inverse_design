#!/usr/bin/env python3
"""Run one centered-Z selected-Q control with the newer 2026 R1.2 build."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "41_run_v261_z2022_m2_selected_q.py"
OUTPUT = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "periodic_T_Z_six_polarization_20260822/selected_Q_diagnostics/"
    "Z_xb_5p3um_25nm_r1p2_build_control"
)
R1P2_ROOT = Path("/home/seunghyun/lumerical_r12/opt/lumerical/v261")


def main() -> int:
    os.environ["PERIODIC_LUMERICAL_ROOT"] = str(R1P2_ROOT)
    os.environ["PERIODIC_LUMERICAL_PYTHONPATH"] = str(R1P2_ROOT / "api/python")
    spec = importlib.util.spec_from_file_location("z_xb_r1p2_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    sys.argv = [
        str(RUNNER),
        "--output-dir", str(OUTPUT),
        "--gpu-device", os.environ.get("LUMERICAL_SESSION_GPU_DEVICE", "GPU 0"),
        "--polarization", "x_b",
        "--wavelength-um", "5.3",
        "--duration-ps", "4.0",
        "--handedness", "LH",
        "--geometry-variant", "centered_expanded_supercell_v4",
        "--mesh-refinement", "conformal variant 1",
    ]
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
