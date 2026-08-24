#!/usr/bin/env python3
"""Probe a causal Lumerical spatial-density carrier for 4-um Au.

This opens an FDTD design/material-database session but performs no Maxwell
solve.  The numerical design fraction is represented as electron density in
an Index perturbation material using its Drude model.  At unit fraction the
carrier is fitted to the frozen Ordal Au permittivity; at zero it is vacuum.

The installed 2026 R1.2 GPU engine explicitly rejects np-density grid
attributes.  This probe therefore validates only material-database readback
and also reports whether the installed engine can execute the carrier on GPU.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
MATERIAL_CONTRACT = HERE / "results_materials_4um/4um_material_contract.json"
LUMERICAL_ROOT = Path("/home/seunghyun/lumerical_r12/opt/lumerical/v261")
LUMAPI_PATH = LUMERICAL_ROOT / "api/python/lumapi.py"
VERSION_PATH = LUMERICAL_ROOT / "VERSION"
FDTD_ENGINE = LUMERICAL_ROOT / "bin/fdtd-engine"

C0_M_S = 299_792_458.0
EPS0_F_M = 8.854_187_8128e-12
ELEMENTARY_CHARGE_C = 1.602_176_634e-19
ELECTRON_MASS_KG = 9.109_383_7139e-31
GPU_NP_DENSITY_REJECTION = (
    b"Error: GPU simulation does not support grid attribute types of "
    b"'lc orientation', 'permittivity rotation', 'matrix transform', 'np density'."
)


@dataclass(frozen=True)
class DrudeCarrier:
    wavelength_m: float
    target_epsilon_real: float
    target_epsilon_imag: float
    epsilon_infinity: float
    omega_rad_s: float
    omega_p_rad_s: float
    gamma_rad_s: float
    electron_effective_mass_m0: float
    electron_density_m3: float
    electron_density_cm3: float
    electron_mobility_m2_Vs: float
    electron_mobility_cm2_Vs: float

    @property
    def target_epsilon(self) -> complex:
        return complex(self.target_epsilon_real, self.target_epsilon_imag)

    def epsilon(self, fraction: np.ndarray | float) -> np.ndarray:
        """Return the passive pole-strength interpolation at the target omega."""

        value = np.asarray(fraction, dtype=np.float64)
        if np.any((value < 0.0) | (value > 1.0)) or not np.all(np.isfinite(value)):
            raise ValueError("Au fraction must be finite in [0,1]")
        return self.epsilon_infinity - value * self.omega_p_rad_s**2 / (
            self.omega_rad_s**2 + 1j * self.gamma_rad_s * self.omega_rad_s
        )


def fit_drude_carrier(
    target_epsilon: complex,
    wavelength_m: float,
    *,
    epsilon_infinity: float = 1.0,
    electron_effective_mass_m0: float = 1.0,
) -> DrudeCarrier:
    """Fit a passive one-pole Drude carrier exactly at one frequency."""

    target = complex(target_epsilon)
    omega = 2.0 * np.pi * C0_M_S / float(wavelength_m)
    real_drop = float(epsilon_infinity - target.real)
    loss = float(target.imag)
    if not (
        wavelength_m > 0.0
        and real_drop > 0.0
        and loss > 0.0
        and electron_effective_mass_m0 > 0.0
    ):
        raise ValueError("target does not admit the passive Drude carrier")
    gamma = omega * loss / real_drop
    omega_p_squared = real_drop * (omega**2 + gamma**2)
    effective_mass = electron_effective_mass_m0 * ELECTRON_MASS_KG
    density_m3 = omega_p_squared * EPS0_F_M * effective_mass / (
        ELEMENTARY_CHARGE_C**2
    )
    mobility_m2_Vs = ELEMENTARY_CHARGE_C / (effective_mass * gamma)
    return DrudeCarrier(
        wavelength_m=float(wavelength_m),
        target_epsilon_real=float(target.real),
        target_epsilon_imag=float(target.imag),
        epsilon_infinity=float(epsilon_infinity),
        omega_rad_s=float(omega),
        omega_p_rad_s=float(np.sqrt(omega_p_squared)),
        gamma_rad_s=float(gamma),
        electron_effective_mass_m0=float(electron_effective_mass_m0),
        electron_density_m3=float(density_m3),
        electron_density_cm3=float(density_m3 / 1.0e6),
        electron_mobility_m2_Vs=float(mobility_m2_Vs),
        electron_mobility_cm2_Vs=float(mobility_m2_Vs * 1.0e4),
    )


def load_frozen_au() -> tuple[float, complex]:
    payload = json.loads(MATERIAL_CONTRACT.read_text(encoding="utf-8"))
    if payload.get("status") != "VALIDATED_4UM_SINGLE_FREQUENCY_MATERIAL_READBACK":
        raise RuntimeError("4-um material contract is not validated")
    au = payload["materials"]["Au"]["epsilon"]
    return float(payload["wavelength_m"]), complex(float(au["real"]), float(au["imag"]))


def installed_release() -> dict[str, Any]:
    values: dict[str, str] = {}
    for line in VERSION_PATH.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    required = ("MAJORRELEASE", "MINORRELEASE", "BUILDNUMBER")
    if any(key not in values for key in required):
        raise RuntimeError(f"unrecognized Lumerical VERSION file: {values}")
    engine = FDTD_ENGINE.read_bytes()
    rejection_present = GPU_NP_DENSITY_REJECTION in engine
    return {
        "major_release": values["MAJORRELEASE"],
        "minor_release": int(values["MINORRELEASE"]),
        "build_number": int(values["BUILDNUMBER"]),
        "version_file": str(VERSION_PATH),
        "fdtd_engine": str(FDTD_ENGINE),
        "installed_gpu_np_density_rejection_present": rejection_present,
        "installed_gpu_np_density_supported": not rejection_present,
    }


def load_lumapi():
    os.environ["VC_LUMERICAL_ROOT"] = str(LUMERICAL_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(LUMERICAL_ROOT)
    os.environ["LUMERICAL_PYTHONPATH"] = str(LUMAPI_PATH.parent)
    os.environ["PATH"] = f"{LUMERICAL_ROOT / 'bin'}:{os.environ.get('PATH', '')}"
    spec = importlib.util.spec_from_file_location("au_4um_np_drude_lumapi", LUMAPI_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(LUMAPI_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def complex_record(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def probe_material(carrier: DrudeCarrier) -> dict[str, Any]:
    """Configure the material and compare direct/time-domain-fit readbacks."""

    lumapi = load_lumapi()
    fdtd = None
    try:
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        base_id = fdtd.addmaterial("Dielectric")
        base_name = "au_4um_np_drude_vacuum_base"
        fdtd.setmaterial(base_id, "name", base_name)
        fdtd.setmaterial(base_name, "Refractive Index", 1.0)

        material_id = fdtd.addmaterial("Index perturbation")
        material_name = "au_4um_np_density_drude_carrier"
        fdtd.setmaterial(material_id, "name", material_name)
        settings = {
            "base material": base_name,
            "include np density": True,
            "include temperature effects": False,
            "use soref and bennett model": False,
            "electron effective mass": carrier.electron_effective_mass_m0,
            # The Lumerical material API uses cm^2/(V s) and cm^-3 here.
            "electron mobility": carrier.electron_mobility_cm2_Vs,
            "test value n": carrier.electron_density_cm3,
        }
        for property_name, value in settings.items():
            fdtd.setmaterial(material_name, property_name, value)
        readback = {
            property_name: fdtd.getmaterial(material_name, property_name)
            for property_name in settings
        }
        readback["np density model"] = fdtd.getmaterial(
            material_name, "np density model"
        )

        frequency = C0_M_S / carrier.wavelength_m
        direct_index = complex(
            np.asarray(fdtd.getindex(material_name, frequency)).reshape(-1)[0]
        )
        fit_fmin = C0_M_S / (1.05 * carrier.wavelength_m)
        fit_fmax = C0_M_S / (0.95 * carrier.wavelength_m)
        fitted_index = complex(
            np.asarray(
                fdtd.getfdtdindex(
                    material_name,
                    np.asarray([frequency]),
                    fit_fmin,
                    fit_fmax,
                    1,
                )
            ).reshape(-1)[0]
        )
        target = carrier.target_epsilon
        direct_error = abs(direct_index**2 - target) / abs(target)
        fitted_error = abs(fitted_index**2 - target) / abs(target)
        return {
            "lumerical_product_version": str(fdtd.version()),
            "material_name": material_name,
            "material_properties": str(fdtd.getmaterial(material_name)).splitlines(),
            "configured_settings": readback,
            "direct_index": complex_record(direct_index),
            "direct_epsilon": complex_record(direct_index**2),
            "direct_epsilon_relative_error": float(direct_error),
            "fdtd_fitted_index": complex_record(fitted_index),
            "fdtd_fitted_epsilon": complex_record(fitted_index**2),
            "fdtd_fitted_epsilon_relative_error": float(fitted_error),
            "fit_band_wavelength_m": [
                0.95 * carrier.wavelength_m,
                1.05 * carrier.wavelength_m,
            ],
            "passed": bool(
                readback["np density model"] == "Drude"
                and direct_error < 5.0e-4
                and fitted_error < 5.0e-4
                and (fitted_index**2).imag > 0.0
            ),
        }
    finally:
        if fdtd is not None:
            fdtd.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--require-installed-gpu-support", action="store_true")
    args = parser.parse_args()

    wavelength_m, epsilon_au = load_frozen_au()
    carrier = fit_drude_carrier(epsilon_au, wavelength_m)
    analytic_endpoint = complex(np.asarray(carrier.epsilon(1.0)).item())
    release = installed_release()
    result: dict[str, Any] = {
        "status": "BLOCKED_LUMERICAL_4UM_AU_NP_DRUDE_PROBE",
        "passed": False,
        "scope": "material-database readback only; no Maxwell solve",
        "generated_files_are_not_run_certificates": True,
        "maxwell_solve_run": False,
        "gpu_engine_acquired": False,
        "thermal_solve_run": False,
        "electrical_solve_run": False,
        "heat_or_charge_license_assumed": False,
        "frozen_target_epsilon": complex_record(epsilon_au),
        "carrier": asdict(carrier),
        "analytic_endpoint_epsilon": complex_record(analytic_endpoint),
        "analytic_endpoint_relative_error": float(
            abs(analytic_endpoint - epsilon_au) / abs(epsilon_au)
        ),
        "installed_release": release,
        "official_gpu_support_contract": {
            "installed_release": "2026 R1.2",
            "installed_engine_supports_np_density_on_gpu": release[
                "installed_gpu_np_density_supported"
            ],
            "required_for_candidate_gpu_route": "2026 R1.3 or newer",
            "r1p3_release_note": (
                "https://optics.ansys.com/hc/en-us/articles/"
                "53916763140499-2026-R1-3-Release-Notes"
            ),
            "np_density_material_documentation": (
                "https://optics.ansys.com/hc/en-us/articles/"
                "360034901753-np-Density-and-Temperature-Index-Perturbation-"
                "Simulation-object"
            ),
        },
    }
    try:
        material = probe_material(carrier)
        result["material_probe"] = material
        material_passed = bool(material["passed"])
        gpu_supported = bool(release["installed_gpu_np_density_supported"])
        result["material_readback_passed"] = material_passed
        result["production_gpu_ready"] = bool(material_passed and gpu_supported)
        result["passed"] = material_passed
        result["status"] = (
            "READY_FOR_B200_NP_DRUDE_ENDPOINT_AND_ADFD"
            if material_passed and gpu_supported
            else (
                "VALIDATED_4UM_AU_NP_DRUDE_MATERIAL_READBACK_"
                "BLOCKED_INSTALLED_R1P2_GPU_UNSUPPORTED"
                if material_passed
                else "FAILED_LUMERICAL_4UM_AU_NP_DRUDE_MATERIAL_READBACK"
            )
        )
        result["next_gate"] = (
            "Install Lumerical 2026 R1.3+ on the B200 host, repeat exact "
            "material/readback and GPU stability endpoints, then validate a "
            "custom fixed-grid dispersive optical adjoint with same-step FD."
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    rendered = json.dumps(result, indent=2, default=str) + "\n"
    if args.output_json is not None:
        output = args.output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        return 2
    if args.require_installed_gpu_support and result["status"] != (
        "READY_FOR_B200_NP_DRUDE_ENDPOINT_AND_ADFD"
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
