#!/usr/bin/env python3
"""Read and freeze the single-frequency 4 um material contract.

This opens a material-database session only.  It does not run FDTD, thermal,
electrical, adjoint, or optimization calculations.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import CONTRACT


HERE = Path(__file__).resolve().parent
REPOSITORY = Path(__file__).resolve().parents[3]
PAPER_MODEL = REPOSITORY / "photothermal_pte/validation/paper_ir_sanity/run_lumerical_device_a_ir_q.py"
PERMITTIVITY = REPOSITORY / "photothermal_pte/bundle/perm_data.txt"
ORDAL = REPOSITORY / "photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/data/au_ordal_1987_nk.csv"
OUT = HERE / "results_materials_4um"
C0 = 299_792_458.0
PALIK_SI = "Si (Silicon) - Palik"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pair(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _interp_complex(path: Path, wavelength_nm: float, real_col: int, imag_col: int) -> complex:
    values = np.loadtxt(path, delimiter="," if path.suffix == ".csv" else None, comments="#")
    values = values[np.argsort(values[:, 0])]
    return complex(
        np.interp(wavelength_nm, values[:, 0], values[:, real_col]),
        np.interp(wavelength_nm, values[:, 0], values[:, imag_col]),
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    wavelength_m = CONTRACT.wavelength_m
    wavelength_nm = wavelength_m * 1e9
    paper = _load(PAPER_MODEL, "au_dualpol_material_paper_model")
    epsilon_sio2 = complex(paper.kitamura_2007_sio2_epsilon(wavelength_m))
    n_sio2 = complex(np.sqrt(epsilon_sio2))
    ta_table = np.loadtxt(PERMITTIVITY)
    ta_table = ta_table[np.argsort(ta_table[:, 0])]
    epsilon_ta = {
        axis: complex(
            np.interp(wavelength_nm, ta_table[:, 0], ta_table[:, column]),
            np.interp(wavelength_nm, ta_table[:, 0], ta_table[:, column + 1]),
        )
        for axis, column in (("a", 1), ("b", 3), ("c", 5))
    }
    ordal = np.genfromtxt(ORDAL, delimiter=",", names=True)
    n_au = complex(
        np.interp(wavelength_m * 1e6, ordal[ordal.dtype.names[0]], ordal[ordal.dtype.names[1]]),
        np.interp(wavelength_m * 1e6, ordal[ordal.dtype.names[0]], ordal[ordal.dtype.names[2]]),
    )
    epsilon_au = n_au**2

    sys.path.insert(0, str(paper.APPROVED_API))
    os.environ["VC_LUMERICAL_ROOT"] = str(paper.APPROVED_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(paper.APPROVED_ROOT)
    os.environ["LUMERICAL_PYTHONPATH"] = str(paper.APPROVED_API)
    import lumapi

    try:
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
    except Exception as exc:
        payload = {
            "status": "BLOCKED_LUMERICAL_4UM_MATERIAL_READBACK",
            "scope": "material database only",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "FDTD_solve_run": False,
        }
        (OUT / "4um_material_contract.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2), flush=True)
        return 3
    try:
        n_si = complex(np.asarray(fdtd.getindex(PALIK_SI, C0 / wavelength_m)).reshape(-1)[0])
    finally:
        fdtd.close()
    epsilon_si = n_si**2
    if epsilon_ta["c"] != epsilon_ta["b"]:
        raise RuntimeError("repository epsilon_c=epsilon_b closure changed")
    if any(value.imag <= 0.0 for value in (*epsilon_ta.values(), epsilon_au)):
        raise RuntimeError("passivity gate failed")
    if n_sio2.imag < 0.0 or n_si.imag < 0.0:
        raise RuntimeError("substrate passivity gate failed")
    payload = {
        "status": "VALIDATED_4UM_SINGLE_FREQUENCY_MATERIAL_READBACK",
        "scope": "material database/readback only; no field solve or optimization",
        "wavelength_m": wavelength_m,
        "axis_mapping": {"x": "b", "y": "a", "z": "c=b closure"},
        "materials": {
            "Au": {
                "n": _pair(n_au),
                "epsilon": _pair(epsilon_au),
                "source": "Ordal et al. 1987 tabulation",
                "source_DOI": "10.1364/AO.26.000744",
            },
            "TaIrTe4": {
                axis: {"epsilon": _pair(value)} for axis, value in epsilon_ta.items()
            },
            "SiO2": {
                "n": _pair(n_sio2),
                "epsilon": _pair(epsilon_sio2),
                "source": "Kitamura et al. 2007 model",
                "source_DOI": "10.1364/AO.46.008118",
            },
            "Si": {
                "n": _pair(n_si),
                "epsilon": _pair(epsilon_si),
                "source": "installed Lumerical v261 Si (Silicon) - Palik readback",
            },
        },
        "inputs": {
            "perm_data": {"path": str(PERMITTIVITY), "sha256": _sha256(PERMITTIVITY)},
            "Ordal_table": {"path": str(ORDAL), "sha256": _sha256(ORDAL)},
            "Kitamura_implementation": {"path": str(PAPER_MODEL), "sha256": _sha256(PAPER_MODEL)},
        },
        "gates": {
            "TaIrTe4_all_axes_passive": True,
            "epsilon_c_equals_epsilon_b": True,
            "Au_passive": True,
            "substrate_loss_nonnegative": True,
            "Lumerical_Si_readback_completed": True,
        },
        "FDTD_solve_run": False,
    }
    (OUT / "4um_material_contract.json").write_text(json.dumps(payload, indent=2) + "\n")
    report = f"""# 4 um material contract

Status: **{payload['status']}**

This checkpoint only reads materials. It does not run Maxwell, thermal,
electrical, adjoint, or optimization solves.

- Au (Ordal): n={n_au.real:.8g}+{n_au.imag:.8g}i, epsilon={epsilon_au.real:.8g}+{epsilon_au.imag:.8g}i
- TaIrTe4 epsilon_a={epsilon_ta['a'].real:.8g}+{epsilon_ta['a'].imag:.8g}i
- TaIrTe4 epsilon_b=epsilon_c={epsilon_ta['b'].real:.8g}+{epsilon_ta['b'].imag:.8g}i
- SiO2 (Kitamura): n={n_sio2.real:.8g}+{n_sio2.imag:.8g}i
- Si (installed Palik readback): n={n_si.real:.8g}+{n_si.imag:.8g}i
"""
    (OUT / "4UM_MATERIAL_CONTRACT.md").write_text(report, encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

