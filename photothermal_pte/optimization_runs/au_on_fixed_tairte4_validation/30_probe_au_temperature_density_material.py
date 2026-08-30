#!/usr/bin/env python3
"""Probe the v261 GPU-compatible temperature-attribute density route.

This is a numerical material parameterization probe.  The imported scalar is
named ``rho_temperature_carrier`` because it transports the design density
through Lumerical's temperature grid-attribute machinery; it is not a physical
temperature field and must never be coupled to the thermal solver.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
LEGACY = HERE.parent / "legacy_v261_optical_support" / "audit_source_only_gpu.py"
AU_N = 12.1
AU_K = 69.2
T_REF_K = 300.0


def load_source_audit():
    spec = importlib.util.spec_from_file_location("au_temperature_density_probe", LEGACY)
    if spec is None or spec.loader is None:
        raise ImportError(LEGACY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.configure_source_audit()
    return module.source_audit


def json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return [value.real, value.imag]
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    audit = load_source_audit()
    os.environ["VC_LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
    os.environ["LUMERICAL_PYTHONPATH"] = str(audit.APPROVED_API)
    os.environ["PATH"] = f"{audit.APPROVED_ROOT / 'bin'}:{os.environ.get('PATH','')}"
    if str(REPOSITORY) not in sys.path:
        sys.path.insert(0, str(REPOSITORY))
    for path in (audit.STAGE1, REPOSITORY / "photothermal_pte"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    helper = audit.load_module(audit.API_HELPER, "au_temperature_density_probe_api")
    installation = type(
        "Installation",
        (),
        {
            "version_key": "v261",
            "root": audit.APPROVED_ROOT,
            "lumapi_path": audit.APPROVED_API / "lumapi.py",
            "device_executable": audit.APPROVED_ROOT / "bin" / "device",
        },
    )()
    lumapi = helper.load_lumapi(installation)

    result: dict[str, Any] = {
        "status": "BLOCKED_AU_TEMPERATURE_DENSITY_MATERIAL_PROBE",
        "meaning": (
            "temperature is used only as a numerical carrier for rho; this is "
            "not physical temperature and is never passed to the thermal solver"
        ),
        "rho_to_attribute": "T_attribute_K = 300 K + rho K",
        "index_law": "n+ik = 1 + rho*((12.1+69.2i)-1)",
    }
    fdtd = None
    try:
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        material_id = fdtd.addmaterial("Index perturbation")
        material_name = "au_density_via_temperature_attribute"
        fdtd.setmaterial(material_id, "name", material_name)
        material_properties = fdtd.getmaterial(material_name)
        requested_settings = {
            "include np density": False,
            "include temperature effects": True,
            "linear sensitivity": True,
            "Tref": T_REF_K,
            "dn/dt": AU_N - 1.0,
            "dk/dt": AU_K,
        }
        default_settings = {}
        setting_errors = {}
        try:
            default_settings["base material"] = fdtd.getmaterial(
                material_name, "base material"
            )
        except Exception as exc:
            setting_errors["base material"] = f"{type(exc).__name__}: {exc}"
        for property_name, value in requested_settings.items():
            try:
                default_settings[property_name] = fdtd.getmaterial(
                    material_name, property_name
                )
                fdtd.setmaterial(material_name, property_name, value)
            except Exception as exc:
                setting_errors[property_name] = f"{type(exc).__name__}: {exc}"
        configured_settings = {}
        for property_name in requested_settings:
            try:
                configured_settings[property_name] = fdtd.getmaterial(
                    material_name, property_name
                )
            except Exception as exc:
                setting_errors[f"readback:{property_name}"] = (
                    f"{type(exc).__name__}: {exc}"
                )

        fdtd.addfdtd(
            {
                "name": "FDTD",
                "dimension": "3D",
                "x span": 2.0e-6,
                "y span": 2.0e-6,
                "z span": 2.0e-6,
            }
        )
        fdtd.addgridattribute("temperature")
        attribute_name = str(fdtd.get("name"))
        attribute_properties = fdtd.get()

        result.update(
            {
                "status": "COMPLETED_AU_TEMPERATURE_DENSITY_MATERIAL_PROBE",
                "solver_version": str(fdtd.version()),
                "material_name": material_name,
                "material_properties": material_properties,
                "material_default_settings": default_settings,
                "material_requested_settings": requested_settings,
                "material_configured_settings": configured_settings,
                "material_setting_errors": setting_errors,
                "temperature_attribute_default_name": attribute_name,
                "temperature_attribute_properties": attribute_properties,
                "requested_linear_sensitivity": {
                    "Tref_K": T_REF_K,
                    "dn_dT_per_K": AU_N - 1.0,
                    "dk_dT_per_K": AU_K,
                },
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        output.write_text(json.dumps(result, indent=2, default=json_default) + "\n")
    print(json.dumps(result, indent=2, default=json_default))
    return 0 if result["status"].startswith("COMPLETED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
