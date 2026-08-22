#!/usr/bin/env python3
"""Read-only result-card audit for the saved finite-187T completed FSP."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


FSP = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "paper_tairte4_finite_187T_w12_Q_11p825um_Eb_blocked_flux_dcard_20260822T1003Z/"
    "finite_187T_w12_Q.fsp"
)
OUTPUT = FSP.parent / "COMPLETED_MONITOR_DATA_AUDIT.json"
ROOT = Path("/home/seunghyun/lumerical_r12/opt/lumerical/v261")
API = ROOT / "api/python"
NAMES = [
    *(f"finite_187T_flux_{axis}_{side}" for axis in "xyz" for side in ("min", "max")),
    "finite_device_pabs::field",
    "finite_device_pabs::index",
]


def main() -> int:
    os.environ["VC_LUMERICAL_ROOT"] = str(ROOT)
    os.environ["LUMERICAL_ROOT"] = str(ROOT)
    os.environ["LUMERICAL_PYTHONPATH"] = str(API)
    os.environ["PATH"] = f"{ROOT / 'bin'}:{os.environ.get('PATH', '')}"
    sys.path.insert(0, str(API))
    import lumapi

    fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
    payload: dict[str, object] = {"fsp": str(FSP), "objects": {}}
    try:
        fdtd.load(str(FSP))
        for name in NAMES:
            row: dict[str, object] = {}
            try:
                row["object_count"] = int(fdtd.getnamednumber(name))
            except Exception as exc:
                row["object_count_error"] = f"{type(exc).__name__}: {exc}"
            for result_name in ("T", "power", "E", "index_detail"):
                try:
                    value = fdtd.getresult(name, result_name)
                    row[f"result_{result_name}"] = {
                        "available": True,
                        "keys": sorted(str(key) for key in value.keys()),
                    }
                except Exception as exc:
                    row[f"result_{result_name}"] = {
                        "available": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            payload["objects"][name] = row
    finally:
        fdtd.close()
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
