#!/usr/bin/env python3
"""Read-only fitted-vs-finite-dt permittivity audit for the corrected Z run."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np


C0 = 299_792_458.0
FSP = Path("/home/seunghyun/tairte4/raw_artifacts/paper_z2022_m2_figure_period_corrected_Ea_5p3um_v3/Z2022_M2_selected_Q.fsp")
OUTPUT = FSP.parent / "finite_dt_permittivity_audit.json"
MATERIALS = ("TaIrTe4_100nm_2024T_substitution", "Au (Gold) - CRC")


def pair(value: complex) -> dict[str, float]:
    return {"real": float(np.real(value)), "imag": float(np.imag(value))}


def main() -> int:
    root = Path(os.environ["LUMERICAL_ROOT"])
    api = Path(os.environ["LUMERICAL_PYTHONPATH"])
    os.environ.setdefault("VC_LUMERICAL_ROOT", str(root))
    os.environ["PATH"] = f"{root / 'bin'}:{os.environ.get('PATH', '')}"
    sys.path.insert(0, str(api))
    import lumapi

    frequency = C0 / 5.3e-6
    fmin, fmax = frequency*(1-1e-9), frequency*(1+1e-9)
    fdtd = lumapi.FDTD(filename=str(FSP), hide=True, serverArgs={"platform": "offscreen"})
    try:
        try:
            raw_dt = fdtd.getdata("FDTD", "dt")
        except Exception:
            raw_dt = fdtd.getnamed("FDTD", "dt")
        dt = float(np.real(np.asarray(raw_dt).reshape(-1)[0]))
        materials: dict[str, object] = {}
        for material in MATERIALS:
            axes = []
            for component, axis in zip((1, 2, 3), "xyz"):
                fitted_n = complex(np.asarray(fdtd.getfdtdindex(material, np.asarray([frequency]), fmin, fmax, component)).reshape(-1)[0])
                fitted_eps = fitted_n**2
                numerical = complex(np.asarray(fdtd.getnumericalpermittivity(material, np.asarray([frequency]), fmin, fmax, dt, component)).reshape(-1)[0])
                axes.append({
                    "axis": axis,
                    "fitted_epsilon": pair(fitted_eps),
                    "finite_dt_numerical_epsilon": pair(numerical),
                    "imag_ratio_numerical_over_fitted": float(np.imag(numerical)/np.imag(fitted_eps)) if np.imag(fitted_eps) else None,
                    "relative_complex_difference": float(abs(numerical-fitted_eps)/max(abs(fitted_eps), 1e-300)),
                })
            materials[material] = axes
    finally:
        fdtd.close()
    payload = {"FSP": str(FSP), "frequency_Hz": frequency, "dt_s": dt, "materials": materials}
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
