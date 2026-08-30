#!/usr/bin/env python3
"""Run one non-overlapping group of centered-Z R1.2 reference-Q cases."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "68_runres_periodic_reference_q_suite.py"
R1P2_ROOT = Path("/home/seunghyun/lumerical_r12/opt/lumerical/v261")
RAW_ROOT = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "periodic_T_Z_six_polarization_20260822/selected_Q_r1p2"
)


def main() -> int:
    group_name = os.environ.get("Z_R1P2_GROUP_NAME", "group").strip()
    polarizations = os.environ.get("Z_R1P2_GROUP_POLS", "").strip()
    if not polarizations:
        raise RuntimeError("set Z_R1P2_GROUP_POLS")
    if not group_name.replace("_", "").isalnum():
        raise RuntimeError("invalid Z_R1P2_GROUP_NAME")
    os.environ.update(
        {
            "PERIODIC_ARCHITECTURE": "Z",
            "PERIODIC_Q_RAW_ROOT": str(RAW_ROOT),
            "PERIODIC_Q_POLARIZATIONS": polarizations,
            "PERIODIC_Q_STATUS_NAME": f"RUNRES_Q_SUITE_STATUS_{group_name}.json",
            "PERIODIC_Z_REFERENCE_WAVELENGTH_UM": "5.3",
            "PERIODIC_LUMERICAL_ROOT": str(R1P2_ROOT),
            "PERIODIC_LUMERICAL_PYTHONPATH": str(R1P2_ROOT / "api/python"),
        }
    )
    spec = importlib.util.spec_from_file_location("z_r1p2_group_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
