#!/usr/bin/env python3
"""Audit requested, fitted, numerical, and monitor epsilon in a saved Z Q run."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.finite_inverse_design.native_yee_q import (  # noqa: E402
    extract_native_yee_q,
    frequency_slice,
)
from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import (  # noqa: E402
    PABS_FIELD,
    PABS_GROUP,
    PABS_INDEX,
)


C0 = 299_792_458.0
WAVELENGTH_M = 5.30e-6
MATERIAL = "TaIrTe4_100nm_2024T_substitution"
DEFAULT_FSP = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "periodic_T_Z_six_polarization_20260822/selected_Q/Z/x_b/"
    "Z2022_M2_selected_Q.fsp"
)
DEFAULT_OUTPUT = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "periodic_T_Z_six_polarization_20260822/selected_Q_diagnostics/"
    "Z_x_b_material_fit_audit.json"
)


def complex_record(value: complex) -> dict[str, float]:
    return {"real": float(np.real(value)), "imag": float(np.imag(value))}


def trapezoid_weights(values: np.ndarray) -> np.ndarray:
    coordinate = np.asarray(values, float).reshape(-1)
    weights = np.empty_like(coordinate)
    weights[0] = 0.5 * (coordinate[1] - coordinate[0])
    weights[-1] = 0.5 * (coordinate[-1] - coordinate[-2])
    weights[1:-1] = 0.5 * (coordinate[2:] - coordinate[:-2])
    return weights


def raw_monitor_power(fdtd: object, monitor: str) -> float:
    x = np.asarray(fdtd.getdata(monitor, "x", 1), float).reshape(-1)
    y = np.asarray(fdtd.getdata(monitor, "y", 1), float).reshape(-1)
    ex = np.asarray(fdtd.getdata(monitor, "Ex", 1)).squeeze()
    ey = np.asarray(fdtd.getdata(monitor, "Ey", 1)).squeeze()
    hx = np.asarray(fdtd.getdata(monitor, "Hx", 1)).squeeze()
    hy = np.asarray(fdtd.getdata(monitor, "Hy", 1)).squeeze()
    pz = 0.5 * np.real(ex * np.conj(hy) - ey * np.conj(hx))
    if pz.shape != (x.size, y.size):
        raise RuntimeError(
            f"{monitor} Pz shape {pz.shape} != {(x.size, y.size)}"
        )
    return float(
        np.einsum(
            "i,j,ij->",
            trapezoid_weights(x),
            trapezoid_weights(y),
            pz,
            optimize=True,
        )
    )


def compare_field_monitor_plane(fdtd: object, power_monitor: str) -> dict[str, object]:
    volume_z = np.asarray(fdtd.getdata(PABS_FIELD, "z", 1), float).reshape(-1)
    plane_z = float(np.asarray(fdtd.getnamed(power_monitor, "z")).reshape(-1)[0])
    iz = int(np.argmin(np.abs(volume_z - plane_z)))
    record: dict[str, object] = {
        "requested_plane_z_m": plane_z,
        "volume_plane_z_m": float(volume_z[iz]),
        "absolute_z_mismatch_m": float(abs(volume_z[iz] - plane_z)),
        "components": {},
    }
    for component in "xy":
        volume = np.asarray(
            fdtd.getdata(PABS_FIELD, f"E{component}", 1)
        ).squeeze()
        plane = np.asarray(
            fdtd.getdata(power_monitor, f"E{component}", 1)
        ).squeeze()
        volume_plane = volume[:, :, iz]
        item: dict[str, object] = {
            "volume_shape": list(volume_plane.shape),
            "power_monitor_shape": list(plane.shape),
            "volume_l2": float(np.linalg.norm(volume_plane)),
            "power_monitor_l2": float(np.linalg.norm(plane)),
        }
        if volume_plane.shape == plane.shape:
            denominator = max(float(np.linalg.norm(plane)), np.finfo(float).tiny)
            item.update(
                {
                    "volume_over_power_l2": float(np.linalg.norm(volume_plane))
                    / denominator,
                    "relative_l2_difference": float(
                        np.linalg.norm(volume_plane - plane) / denominator
                    ),
                }
            )
        record["components"][component] = item
    return record


def scalar_complex(value: object) -> complex:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise RuntimeError(f"expected scalar, got {array.shape}")
    return complex(array[0])


def main() -> int:
    fsp = Path(os.environ.get("Z_Q_AUDIT_FSP", str(DEFAULT_FSP))).resolve()
    output = Path(os.environ.get("Z_Q_AUDIT_OUTPUT", str(DEFAULT_OUTPUT))).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lumerical_root = Path(os.environ.get("LUMERICAL_ROOT", "/opt/lumerical/v261"))
    lumerical_api = Path(
        os.environ.get("LUMERICAL_PYTHONPATH", str(lumerical_root / "api/python"))
    )
    sys.path.insert(0, str(lumerical_api))
    import lumapi

    fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
    result: dict[str, object] = {
        "status": "BLOCKED_Z_Q_MATERIAL_FIT_AUDIT",
        "fsp": str(fsp),
    }
    try:
        fdtd.load(str(fsp))
        frequency = C0 / WAVELENGTH_M
        source_start = max(4.0e-6, 0.95 * WAVELENGTH_M)
        source_stop = min(12.0e-6, 1.05 * WAVELENGTH_M)
        fmin = C0 / source_stop
        fmax = C0 / source_start
        broadband_fmin = C0 / 12.0e-6
        broadband_fmax = C0 / 4.0e-6
        dt = float(np.asarray(fdtd.getnamed("FDTD", "dt")).reshape(-1)[0])
        component_records: dict[str, object] = {}
        fdtd.cwnorm()
        q = extract_native_yee_q(
            fdtd,
            field_monitor=PABS_FIELD,
            index_monitor=PABS_INDEX,
            wavelength_m=WAVELENGTH_M,
        )
        source_power_default = float(
            np.real(np.asarray(fdtd.sourcepower(frequency, 2)).reshape(-1)[0])
        )
        q_power_cwnorm = float(q["P_Q_W"])
        fdtd.nonorm()
        q_power_nonorm = float(
            extract_native_yee_q(
                fdtd,
                field_monitor=PABS_FIELD,
                index_monitor=PABS_INDEX,
                wavelength_m=WAVELENGTH_M,
            )["P_Q_W"]
        )
        fdtd.cwnorm()
        source_power_named = float(
            np.real(
                np.asarray(
                    fdtd.sourcepower(frequency, 2, "Z2022_source_linear")
                ).reshape(-1)[0]
            )
        )
        transmission_top = float(
            np.real(np.asarray(fdtd.transmission("Z2022_flux_top")).reshape(-1)[0])
        )
        transmission_bottom = float(
            np.real(np.asarray(fdtd.transmission("Z2022_flux_bottom")).reshape(-1)[0])
        )
        raw_top_power = raw_monitor_power(fdtd, "Z2022_flux_top")
        raw_bottom_power = raw_monitor_power(fdtd, "Z2022_flux_bottom")
        top_field_pairing = compare_field_monitor_plane(fdtd, "Z2022_flux_top")
        fdtd.runanalysis(PABS_GROUP)
        pabs_total = float(
            np.real(
                np.asarray(fdtd.getresult(PABS_GROUP, "Pabs_total")["Pabs_total"])
                .reshape(-1)[0]
            )
        )
        base_shape = tuple(np.asarray(q["base_coordinates"][a]).size for a in "xyz")
        frequency_data = np.asarray(fdtd.getdata(PABS_FIELD, "f", 1), float).reshape(-1)
        fi = int(q["frequency_index_zero_based"])
        for number, component in enumerate("xyz", start=1):
            requested_n = scalar_complex(fdtd.getindex(MATERIAL, frequency, number))
            fitted_n = scalar_complex(
                fdtd.getfdtdindex(MATERIAL, frequency, fmin, fmax, number)
            )
            numerical_epsilon = scalar_complex(
                fdtd.getnumericalpermittivity(
                    MATERIAL, frequency, fmin, fmax, dt, number, 0
                )
            )
            broadband_fitted_n = scalar_complex(
                fdtd.getfdtdindex(
                    MATERIAL,
                    frequency,
                    broadband_fmin,
                    broadband_fmax,
                    number,
                )
            )
            broadband_numerical_epsilon = scalar_complex(
                fdtd.getnumericalpermittivity(
                    MATERIAL,
                    frequency,
                    broadband_fmin,
                    broadband_fmax,
                    dt,
                    number,
                    0,
                )
            )
            monitor_n = frequency_slice(
                np.asarray(fdtd.getdata(PABS_INDEX, f"index_{component}", 1)),
                base_shape,
                fi,
                frequency_data.size,
                f"index_{component}",
            )
            q_component = np.asarray(q["Q_components"][component], float)
            hotspot = np.unravel_index(int(np.argmax(q_component)), q_component.shape)
            hotspot_n = complex(monitor_n[hotspot])
            component_records[component] = {
                "requested_epsilon_getindex": complex_record(requested_n**2),
                "fitted_epsilon_getfdtdindex": complex_record(fitted_n**2),
                "numerical_epsilon_finite_dt": complex_record(numerical_epsilon),
                "broadband_4_12um_fitted_epsilon": complex_record(
                    broadband_fitted_n**2
                ),
                "broadband_4_12um_numerical_epsilon": complex_record(
                    broadband_numerical_epsilon
                ),
                "hotspot_index_monitor_epsilon": complex_record(hotspot_n**2),
                "hotspot_Q_W_m3": float(q_component[hotspot]),
                "hotspot_index": [int(value) for value in hotspot],
                "hotspot_coordinate_m": {
                    axis: float(q["native_coordinates"][component][axis][hotspot[i]])
                    for i, axis in enumerate("xyz")
                },
            }
        result.update(
            {
                "status": "COMPLETED_Z_Q_MATERIAL_FIT_AUDIT",
                "wavelength_m": WAVELENGTH_M,
                "source_fit_bounds_m": [source_start, source_stop],
                "solver_broadband_fit_bounds_m": [4.0e-6, 12.0e-6],
                "dt_s": dt,
                "material": MATERIAL,
                "normalization_audit": {
                    "source_power_default_W": source_power_default,
                    "source_power_named_W": source_power_named,
                    "default_over_named": source_power_default / source_power_named,
                    "transmission_top_signed": transmission_top,
                    "transmission_bottom_signed": transmission_bottom,
                    "raw_top_Pz_integral_W": raw_top_power,
                    "raw_bottom_Pz_integral_W": raw_bottom_power,
                    "raw_top_over_transmission_power": raw_top_power
                    / (transmission_top * source_power_named),
                    "raw_bottom_over_transmission_power": raw_bottom_power
                    / (transmission_bottom * source_power_named),
                    "flux_absorption_fraction": transmission_bottom - transmission_top,
                    "pabs_total_normalized": pabs_total,
                    "native_Q_over_named_source_power": float(q["P_Q_W"]) / source_power_named,
                    "native_Q_cwnorm_W": q_power_cwnorm,
                    "native_Q_nonorm_W": q_power_nonorm,
                },
                "top_field_monitor_pairing": top_field_pairing,
                "components": component_records,
            }
        )
    finally:
        fdtd.close()
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"].startswith("COMPLETED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
