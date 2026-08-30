#!/usr/bin/env python3
"""Freeze the 10-um SiO2/Si optical substrate material contract.

This is a material-database/readback probe only.  It does not run FDTD,
thermal, electrical, adjoint, or optimization calculations.  Fused-silica
epsilon is evaluated from the repository's Kitamura-2007 implementation at
exactly 10 um.  Silicon n,k is read directly from the installed Lumerical
v261 Palik database at the same frequency and copied into a single-frequency
(n,k) material so the requested and realized values can be compared exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = Path(__file__).resolve().parents[3]
PAPER_MODEL = (
    REPOSITORY
    / "photothermal_pte"
    / "validation"
    / "paper_ir_sanity"
    / "run_lumerical_device_a_ir_q.py"
)
WAVELENGTH_M = 10.0e-6
C0_M_PER_S = 299_792_458.0
PALIK_SI = "Si (Silicon) - Palik"
SIO2_NAME = "au_design_SiO2_Kitamura_10um"
SI_NAME = "au_design_Si_Palik_10um"


def _load_paper_model():
    spec = importlib.util.spec_from_file_location("paper_ir_model_10um", PAPER_MODEL)
    if spec is None or spec.loader is None:
        raise ImportError(PAPER_MODEL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _complex(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar_index(fdtd, name: str) -> complex:
    raw = np.asarray(fdtd.getindex(name, C0_M_PER_S / WAVELENGTH_M)).reshape(-1)
    if raw.size != 1:
        raise RuntimeError(f"Unexpected index readback for {name}: {raw.shape}")
    return complex(raw[0])


def _install_nk(fdtd, name: str, index: complex) -> None:
    material = fdtd.addmaterial("(n,k) Material")
    fdtd.setmaterial(material, "name", name)
    fdtd.setmaterial(name, "Refractive Index", float(index.real))
    fdtd.setmaterial(name, "Imaginary Refractive Index", float(index.imag))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--offline-si-n",
        type=float,
        default=3.4215,
        help="Named lossless-Si fallback from Frey et al., Opt. Lett. 45, 4935 (2020), Table 1.",
    )
    parser.add_argument(
        "--offline-after-recorded-license-blocker",
        action="store_true",
        help="Write the fail-closed offline diagnostic after a separately recorded session-start failure.",
    )
    args = parser.parse_args()

    paper = _load_paper_model()
    requested_sio2_epsilon = complex(paper.kitamura_2007_sio2_epsilon(WAVELENGTH_M))
    requested_sio2_index = complex(np.sqrt(requested_sio2_epsilon))
    if requested_sio2_index.imag <= 0.0:
        raise RuntimeError("Kitamura SiO2 loss is not passive at 10 um")

    sys.path.insert(0, str(paper.APPROVED_API))
    os.environ["VC_LUMERICAL_ROOT"] = str(paper.APPROVED_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(paper.APPROVED_ROOT)
    import lumapi

    try:
        if args.offline_after_recorded_license_blocker:
            raise RuntimeError(
                "Recorded v261 session startup blocker: license expires-in-4-days warning "
                "was returned as appOpen failure on 2026-08-21"
            )
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
    except Exception as exc:
        # Fail closed for the requested Palik readback, while preserving the
        # independently traceable material values that can be used for an
        # explicitly named FDTDX diagnostic.  The lossless Si closure is not
        # silently promoted to a Palik result.
        fallback_si_index = complex(args.offline_si_n, 0.0)
        fallback = {
            "status": "BLOCKED_LUMERICAL_10UM_SI_PALIK_READBACK",
            "scope": "material/readback only; no FDTD, thermal, electrical, adjoint, or optimization",
            "wavelength_m": WAVELENGTH_M,
            "blocker": {
                "type": type(exc).__name__,
                "message": str(exc),
                "Palik_value_claimed": False,
            },
            "offline_diagnostic_contract": {
                "SiO2": {
                    "model": "Kitamura et al. 2007 Eq.21-24/Table 2 with sqrt(pi) correction",
                    "provenance": "https://doi.org/10.1364/AO.46.008118",
                    "n": _complex(requested_sio2_index),
                    "epsilon": _complex(requested_sio2_epsilon),
                },
                "Si": {
                    "model": "lossless measured-index diagnostic closure",
                    "n": _complex(fallback_si_index),
                    "epsilon": _complex(fallback_si_index**2),
                    "provenance": "Frey et al., Opt. Lett. 45, 4935 (2020), Table 1, 10-um n=3.4215",
                    "source": "https://doi.org/10.1364/OL.398778",
                    "limitation": (
                        "k=0 is an explicit diagnostic approximation; this is not the requested "
                        "installed-Lumerical Palik readback and is not a confidence interval"
                    ),
                },
            },
            "FDTD_solve_run": False,
            "inputs": {
                "paper_model_path": str(PAPER_MODEL),
                "paper_model_sha256": _sha256(PAPER_MODEL),
                "requested_Lumerical_material_database": PALIK_SI,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(fallback, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(fallback, indent=2), flush=True)
        return 3
    try:
        requested_si_index = _scalar_index(fdtd, PALIK_SI)
        _install_nk(fdtd, SIO2_NAME, requested_sio2_index)
        _install_nk(fdtd, SI_NAME, requested_si_index)
        realized_sio2_index = _scalar_index(fdtd, SIO2_NAME)
        realized_si_index = _scalar_index(fdtd, SI_NAME)
    finally:
        fdtd.close()

    requested_si_epsilon = requested_si_index**2
    realized_sio2_epsilon = realized_sio2_index**2
    realized_si_epsilon = realized_si_index**2
    sio2_error = abs(realized_sio2_epsilon - requested_sio2_epsilon) / abs(
        requested_sio2_epsilon
    )
    si_error = abs(realized_si_epsilon - requested_si_epsilon) / abs(requested_si_epsilon)
    gates = {
        "SiO2_readback_relative_error_lt_1e-12": bool(sio2_error < 1.0e-12),
        "Si_readback_relative_error_lt_1e-12": bool(si_error < 1.0e-12),
        "SiO2_loss_positive": bool(realized_sio2_index.imag > 0.0),
        "Si_loss_nonnegative": bool(realized_si_index.imag >= 0.0),
    }
    passed = all(gates.values())
    payload = {
        "status": (
            "VALIDATED_10UM_SIO2_SI_MATERIAL_READBACK"
            if passed
            else "FAILED_10UM_SIO2_SI_MATERIAL_READBACK"
        ),
        "scope": "material/readback only; no FDTD, thermal, electrical, adjoint, or optimization",
        "wavelength_m": WAVELENGTH_M,
        "materials": {
            "SiO2": {
                "model": "Kitamura et al. 2007 Eq.21-24/Table 2 with sqrt(pi) correction",
                "provenance": "doi:10.1364/AO.46.008118",
                "requested_n": _complex(requested_sio2_index),
                "requested_epsilon": _complex(requested_sio2_epsilon),
                "readback_n": _complex(realized_sio2_index),
                "readback_epsilon": _complex(realized_sio2_epsilon),
                "relative_epsilon_error": float(sio2_error),
            },
            "Si": {
                "model": "installed Lumerical v261 Si (Silicon) - Palik raw 10-um n,k",
                "provenance_limit": (
                    "explicit substrate closure; the TaIrTe4 paper does not certify "
                    "this exact Si optical dataset for the new 10-um Au-design problem"
                ),
                "requested_n": _complex(requested_si_index),
                "requested_epsilon": _complex(requested_si_epsilon),
                "readback_n": _complex(realized_si_index),
                "readback_epsilon": _complex(realized_si_epsilon),
                "relative_epsilon_error": float(si_error),
            },
        },
        "single_frequency_contract": True,
        "FDTD_solve_run": False,
        "gates": gates,
        "inputs": {
            "paper_model_path": str(PAPER_MODEL),
            "paper_model_sha256": _sha256(PAPER_MODEL),
            "Lumerical_material_database": PALIK_SI,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
