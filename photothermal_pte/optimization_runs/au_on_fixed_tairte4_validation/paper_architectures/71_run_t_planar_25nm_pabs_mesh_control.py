#!/usr/bin/env python3
"""Run the compact bare-T control at the Z suite's 25-nm lateral mesh.

This is a single-variable diagnostic for the unresolved Z volumetric-loss
closure.  It preserves the validated T optical stack and changes only dx=dy
from 10 nm to 25 nm.  It is not a replacement production certificate.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import os
import sys


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "07_run_v261_t2024_tairte4_optical_smoke.py"
OUTPUT = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "periodic_T_Z_six_polarization_20260822/selected_Q_diagnostics/"
    f"T_planar_{os.environ.get('T_PLANAR_DIAG_MESH_NM', '25').replace('.', 'p')}nm_"
    "mesh_control_corrected_runres"
)


def main() -> int:
    mesh_nm = os.environ.get("T_PLANAR_DIAG_MESH_NM", "25")
    spec = importlib.util.spec_from_file_location("t_planar_25nm_runner", RUNNER)
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
        "--wavelength-um", "4.75",
        "--duration-ps", "1.2",
        "--substrate-mode", "sio2_si_reduced_285nm",
        "--omit-top-t-control",
        "--lateral-mesh-nm", mesh_nm,
    ]
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
