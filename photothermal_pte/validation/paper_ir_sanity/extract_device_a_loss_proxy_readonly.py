#!/usr/bin/env python3
"""Read saved Device-A component fields/index for loss-attribution diagnostics.

This opens a completed FSP and calls only ``load``/``getdata``.  It does not
run or re-analyse Maxwell.  The reported Im(epsilon_eff)/Im(epsilon_TaIrTe4)
is explicitly a conformal-loss proxy, not a material occupancy or a certified
TaIrTe4 power fraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from matplotlib.path import Path as PolygonPath
import numpy as np

from photothermal_pte.validation.paper_ir_sanity import (
    run_lumerical_device_a_ir_q as runner,
)


EPS0 = 8.8541878128e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coordinate(fdtd: Any, monitor: str, axis: str) -> np.ndarray:
    return np.asarray(fdtd.getdata(monitor, axis, 1), float).reshape(-1)


def dual_widths(coordinate_m: np.ndarray) -> np.ndarray:
    values = np.asarray(coordinate_m, float).reshape(-1)
    edges = np.concatenate(
        (
            [values[0] - 0.5 * (values[1] - values[0])],
            0.5 * (values[:-1] + values[1:]),
            [values[-1] + 0.5 * (values[-1] - values[-2])],
        )
    )
    return np.diff(edges)


def real_stats(values: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": 0, "minimum": None, "median": None, "maximum": None}
    return {
        "count": int(finite.size),
        "minimum": float(np.min(finite)),
        "median": float(np.median(finite)),
        "maximum": float(np.max(finite)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case_dir = args.case_dir.resolve()
    fsp = case_dir / "finite_2um_optical_q.fsp"
    result_path = case_dir / "case_result.json"
    result = json.loads(result_path.read_text())
    if result.get("status") != "COMPLETED":
        raise RuntimeError("saved optical case is not completed")

    base = runner.load_base()
    base.TARGET_WAVELENGTH_M = runner.WAVELENGTH_M
    base.TARGET_FREQUENCY_HZ = runner.C0 / runner.WAVELENGTH_M
    os.environ["VC_LUMERICAL_ROOT"] = str(runner.APPROVED_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(runner.APPROVED_ROOT)
    os.environ["LUMERICAL_PYTHONPATH"] = str(runner.APPROVED_API)
    if str(runner.APPROVED_API) not in sys.path:
        sys.path.insert(0, str(runner.APPROVED_API))
    installation = SimpleNamespace(
        version_key="v261",
        root=runner.APPROVED_ROOT.resolve(),
        lumapi_path=(runner.APPROVED_API / "lumapi.py").resolve(),
        device_executable=(runner.APPROVED_ROOT / "bin" / "device").resolve(),
    )
    lumapi = base.load_lumapi(installation)
    geometry = result["pre_run_contract"]["geometry"]
    vertices_m = np.asarray(geometry["flake_vertices_um"], float) * 1e-6
    polygon = PolygonPath(vertices_m)
    thickness_m = float(geometry["flake_thickness_m"])
    requested = result["run_result"]["material_epsilon_readback"]["axes"]
    normalization_scale = float(
        result["run_result"]["normalization"]["scale_to_1_W_m2"]
    )
    omega = 2.0 * np.pi * runner.C0 / runner.WAVELENGTH_M

    fdtd = None
    try:
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        fdtd.load(str(fsp))
        field_common = {
            axis: coordinate(fdtd, base.PABS_FIELD, axis) for axis in "xyz"
        }
        index_common = {
            axis: coordinate(fdtd, base.PABS_INDEX, axis) for axis in "xyz"
        }
        field_delta = {
            axis: np.asarray(
                fdtd.getdata(base.PABS_FIELD, f"delta_{axis}", 1), float
            ).reshape(-1)
            for axis in "xyz"
        }
        index_delta = {
            axis: np.asarray(
                fdtd.getdata(base.PABS_INDEX, f"delta_{axis}", 1), float
            ).reshape(-1)
            for axis in "xyz"
        }
        components: dict[str, Any] = {}
        for component in "xyz":
            field_coordinates = {
                axis: np.array(field_common[axis], copy=True) for axis in "xyz"
            }
            index_coordinates = {
                axis: np.array(index_common[axis], copy=True) for axis in "xyz"
            }
            field_coordinates[component] += field_delta[component]
            index_coordinates[component] += index_delta[component]
            coordinate_mismatch = {
                axis: float(
                    np.max(
                        np.abs(
                            field_coordinates[axis] - index_coordinates[axis]
                        )
                    )
                )
                for axis in "xyz"
            }
            electric = np.asarray(
                fdtd.getdata(base.PABS_FIELD, f"E{component}", 1)
            ).squeeze()
            refractive_index = np.asarray(
                fdtd.getdata(base.PABS_INDEX, f"index_{component}", 1)
            ).squeeze()
            epsilon = refractive_index**2
            expected = tuple(index_coordinates[axis].size for axis in "xyz")
            if epsilon.shape != expected or electric.shape != expected:
                raise RuntimeError(
                    f"{component}: E {electric.shape}, epsilon {epsilon.shape}, "
                    f"coordinates {expected}"
                )
            bulk = complex(
                requested[component]["finite_dt_numerical_permittivity"]["real"],
                requested[component]["finite_dt_numerical_permittivity"]["imag"],
            )
            if abs(bulk.imag) <= 1e-12:
                raise RuntimeError(f"{component}: bulk-loss denominator is zero")
            proxy = epsilon.imag / bulk.imag
            x, y, z = (index_coordinates[axis] for axis in "xyz")
            xx, yy = np.meshgrid(x, y, indexing="ij")
            lateral = polygon.contains_points(
                np.column_stack((xx.ravel(), yy.ravel())), radius=1e-15
            ).reshape(xx.shape)
            support = lateral[:, :, None] & (
                (z[None, None, :] >= -thickness_m - 1e-15)
                & (z[None, None, :] <= 1e-15)
            )
            q_component = 0.5 * omega * EPS0 * epsilon.imag * np.abs(electric) ** 2
            wx, wy, wz = (
                dual_widths(index_coordinates[axis]) for axis in "xyz"
            )
            volume = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
            raw_power = float(np.sum(q_component * volume))
            center_support_power = float(np.sum(q_component[support] * volume[support]))
            proxy_weighted_power = float(
                np.sum(q_component * proxy * volume)
            )
            finite_proxy = proxy[np.isfinite(proxy)]
            near_fractional = np.isfinite(proxy) & (proxy > 1e-6) & (proxy < 1.0 - 1e-6)
            components[component] = {
                "shape": list(expected),
                "field_index_coordinate_mismatch_m": coordinate_mismatch,
                "maximum_coordinate_mismatch_m": max(coordinate_mismatch.values()),
                "requested_bulk_TaIrTe4_epsilon": {
                    "real": bulk.real,
                    "imag": bulk.imag,
                },
                "loss_proxy_definition": (
                    "Im(epsilon_eff_component)/Im(epsilon_bulk_TaIrTe4_component)"
                ),
                "loss_proxy_status": (
                    "DIAGNOSTIC_ONLY_NOT_OCCUPANCY_NOT_CERTIFIED_POWER_FRACTION"
                ),
                "loss_proxy_all_cells": real_stats(proxy),
                "loss_proxy_center_support_cells": real_stats(proxy[support]),
                "fractional_proxy_cell_count": int(np.count_nonzero(near_fractional)),
                "proxy_below_zero_cell_count": int(np.count_nonzero(finite_proxy < 0.0)),
                "proxy_above_one_cell_count": int(np.count_nonzero(finite_proxy > 1.0)),
                "proxy_above_one_plus_1e_minus_3_cell_count": int(
                    np.count_nonzero(finite_proxy > 1.001)
                ),
                "proxy_below_zero_range": real_stats(finite_proxy[finite_proxy < 0.0]),
                "proxy_above_one_range": real_stats(finite_proxy[finite_proxy > 1.0]),
                "native_raw_effective_loss_power_W": raw_power,
                "native_exact_center_support_power_W": center_support_power,
                "native_proxy_weighted_power_W_diagnostic_only": proxy_weighted_power,
                "at_1_W_m2_raw_effective_loss_power_W": (
                    raw_power * normalization_scale
                ),
                "at_1_W_m2_exact_center_support_power_W": (
                    center_support_power * normalization_scale
                ),
                "at_1_W_m2_proxy_weighted_power_W_diagnostic_only": (
                    proxy_weighted_power * normalization_scale
                ),
                "warning": (
                    "multiplying Q by this proxy can double-count conformal "
                    "mixing because Q already contains Im(epsilon_eff); this "
                    "number is not used as a thermal source"
                ),
            }
            del electric, refractive_index, epsilon, proxy, q_component, volume
        payload = {
            "status": "EXTRACTED_DEVICE_A_EFFECTIVE_LOSS_PROXY_READ_ONLY",
            "scope": (
                "saved FSP load/getdata only; no FDTD run, runanalysis, thermal, "
                "PTE, adjoint, or optimization"
            ),
            "case": {
                "directory": str(case_dir),
                "fsp_path": str(fsp),
                "fsp_size_bytes": fsp.stat().st_size,
                "fsp_sha256": sha256(fsp),
                "case_result_path": str(result_path),
                "case_result_sha256": sha256(result_path),
                "polarization": geometry["source"]["polarization_axis"],
                "normalization_scale_to_1_W_m2": normalization_scale,
            },
            "axis_mapping": "x=b, y=a, z=c",
            "components": components,
            "maximum_field_index_coordinate_mismatch_m": max(
                item["maximum_coordinate_mismatch_m"]
                for item in components.values()
            ),
            "thermal_source_promotion": False,
        }
    finally:
        if fdtd is not None:
            fdtd.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
