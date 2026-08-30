#!/usr/bin/env python3
"""Read-only audit of volume-monitor Poynting data in the corrected Z FSP."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np


FSP = Path("/home/seunghyun/tairte4/raw_artifacts/paper_z2022_m2_figure_period_corrected_Ea_5p3um_v3/Z2022_M2_selected_Q.fsp")
OUTPUT = FSP.parent / "saved_volume_poynting_data_audit.json"
MONITOR = "finite_device_pabs::field"


def main() -> int:
    root = Path(os.environ["LUMERICAL_ROOT"])
    api = Path(os.environ["LUMERICAL_PYTHONPATH"])
    os.environ.setdefault("VC_LUMERICAL_ROOT", str(root))
    os.environ["PATH"] = f"{root / 'bin'}:{os.environ.get('PATH', '')}"
    sys.path.insert(0, str(api))
    import lumapi

    fdtd = lumapi.FDTD(filename=str(FSP), hide=True, serverArgs={"platform": "offscreen"})
    result: dict[str, object] = {"FSP": str(FSP), "monitor": MONITOR, "datasets": {}}
    try:
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Px", "Py", "Pz"):
            try:
                value = np.asarray(fdtd.getdata(MONITOR, name, 1))
                result["datasets"][name] = {
                    "available": True,
                    "shape": list(value.shape),
                    "finite": bool(np.all(np.isfinite(value))),
                }
            except Exception as exc:
                result["datasets"][name] = {
                    "available": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
    finally:
        fdtd.close()
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
