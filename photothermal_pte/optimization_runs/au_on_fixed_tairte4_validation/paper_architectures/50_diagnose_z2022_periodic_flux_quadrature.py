#!/usr/bin/env python3
"""Read-only diagnosis of periodic flux quadrature in a completed Z FSP."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


C0 = 299_792_458.0
DEFAULT_FSP = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "paper_z2022_m2_figure_digitized_Ea_5p3um_v2_matched_cv/"
    "Z2022_M2_selected_Q.fsp"
)
DEFAULT_OUTPUT = DEFAULT_FSP.parent / "periodic_flux_quadrature_diagnostic.json"


def scalar(value: object) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError(array.shape)
    return float(np.real(array[0]))


def frequency_slice(value: object) -> np.ndarray:
    array = np.asarray(value).squeeze()
    while array.ndim > 2 and array.shape[-1] == 1:
        array = array[..., 0]
    return array


def plane_audit(fdtd: object, name: str) -> dict[str, object]:
    x = np.asarray(fdtd.getdata(name, "x", 1), float).reshape(-1)
    y = np.asarray(fdtd.getdata(name, "y", 1), float).reshape(-1)
    fields = {
        key: frequency_slice(fdtd.getdata(name, key, 1))
        for key in ("Ex", "Ey", "Hx", "Hy")
    }
    shape = (x.size, y.size)
    for key, value in fields.items():
        if value.shape != shape:
            fields[key] = np.asarray(value).reshape(shape)
    sz = 0.5 * np.real(
        fields["Ex"] * np.conj(fields["Hy"])
        - fields["Ey"] * np.conj(fields["Hx"])
    )
    trapz = float(np.trapezoid(np.trapezoid(sz, y, axis=1), x, axis=0))
    # A periodic grid contains both representations of the seam.  The
    # half-weight endpoint trapezoid is equivalent to dropping one duplicate
    # endpoint for a uniform periodic mesh.
    periodic_drop_last = float(
        np.sum(sz[:-1, :-1])
        * float((x[-1] - x[0]) / (x.size - 1))
        * float((y[-1] - y[0]) / (y.size - 1))
    )
    return {
        "shape": list(shape),
        "x_bounds_m": [float(x[0]), float(x[-1])],
        "y_bounds_m": [float(y[0]), float(y[-1])],
        "x_endpoint_relative_mismatch": float(
            np.linalg.norm(sz[0] - sz[-1]) / max(np.linalg.norm(sz[0]), np.linalg.norm(sz[-1]), 1.0e-300)
        ),
        "y_endpoint_relative_mismatch": float(
            np.linalg.norm(sz[:, 0] - sz[:, -1]) / max(np.linalg.norm(sz[:, 0]), np.linalg.norm(sz[:, -1]), 1.0e-300)
        ),
        "manual_trapezoid_power_W": trapz,
        "manual_periodic_drop_last_power_W": periodic_drop_last,
        "lumerical_transmission_normalized": scalar(fdtd.transmission(name)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsp", type=Path, default=DEFAULT_FSP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--wavelength-um", type=float, default=5.3)
    args = parser.parse_args()
    root = Path(os.environ["LUMERICAL_ROOT"])
    api = Path(os.environ["LUMERICAL_PYTHONPATH"])
    os.environ.setdefault("VC_LUMERICAL_ROOT", str(root))
    os.environ["PATH"] = f"{root / 'bin'}:{os.environ.get('PATH', '')}"
    sys.path.insert(0, str(api))
    import lumapi

    fdtd = lumapi.FDTD(filename=str(args.fsp.resolve()), hide=True, serverArgs={"platform": "offscreen"})
    try:
        source_power = scalar(fdtd.sourcepower(C0 / (args.wavelength_um * 1.0e-6), 2, "Z2022_source_linear"))
        result = {
            "source_power_W": source_power,
            "top": plane_audit(fdtd, "Z2022_flux_top"),
            "bottom": plane_audit(fdtd, "Z2022_flux_bottom"),
        }
        for property_name in ("analysis script", "setup script"):
            try:
                result[f"pabs_{property_name.replace(' ', '_')}"] = str(
                    fdtd.getnamed("finite_device_pabs", property_name)
                )
            except Exception as exc:
                result[f"pabs_{property_name.replace(' ', '_')}_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
        for method in ("manual_trapezoid_power_W", "manual_periodic_drop_last_power_W"):
            result[f"absorbed_{method}"] = result["bottom"][method] - result["top"][method]
        result["absorbed_lumerical_W"] = (
            result["bottom"]["lumerical_transmission_normalized"]
            - result["top"]["lumerical_transmission_normalized"]
        ) * source_power
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        fdtd.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
