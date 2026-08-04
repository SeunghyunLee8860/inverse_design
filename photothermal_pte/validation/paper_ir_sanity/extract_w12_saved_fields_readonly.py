#!/usr/bin/env python3
"""Extract saved paper-IR field monitors without running FDTD.

The production runs retain a total-field E/H plane 50 nm above the flake and
an additional total-field plane at z=+0.5 um.  This tool reopens a completed
FSP, reads those monitors and the saved source-object field, and writes a new
audit directory.  It never calls ``run`` or ``runanalysis`` and never edits
the original artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    run_lumerical_device_a_ir_q as runner,
)
from photothermal_pte.validation.paper_ir_sanity.compare_paper_ir_smoke_q_convergence import (  # noqa: E402
    trapezoid_weights,
)


def monitor_fields(fdtd: Any, monitor: str) -> dict[str, Any]:
    coordinates = {
        axis: np.asarray(fdtd.getdata(monitor, axis, 1), float).reshape(-1)
        for axis in "xyz"
    }
    return {
        "coordinates": coordinates,
        "electric": {
            axis: np.asarray(fdtd.getdata(monitor, f"E{axis}", 1)).squeeze()
            for axis in "xyz"
        },
        "magnetic": {
            axis: np.asarray(fdtd.getdata(monitor, f"H{axis}", 1)).squeeze()
            for axis in "xyz"
        },
    }


def weights_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.outer(
        trapezoid_weights(x),
        trapezoid_weights(y),
    )


def component_metrics(
    values: dict[str, np.ndarray],
    weights: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        axis: {
            "maximum_absolute": float(np.max(np.abs(array))),
            "area_weighted_L2": float(
                np.sqrt(np.sum(weights * np.abs(array) ** 2))
            ),
        }
        for axis, array in values.items()
    }


def collocated_flake_midplane_fields(
    base: Any,
    fdtd: Any,
    z_target_m: float,
) -> dict[str, Any]:
    common = {
        axis: np.asarray(
            fdtd.getdata(base.PABS_FIELD, axis, 1), float
        ).reshape(-1)
        for axis in "xyz"
    }
    deltas = {
        axis: np.asarray(
            fdtd.getdata(base.PABS_FIELD, f"delta_{axis}", 1), float
        ).reshape(-1)
        for axis in "xyz"
    }
    component_coordinates: dict[str, dict[str, np.ndarray]] = {}
    for component in "xyz":
        component_coordinates[component] = {
            axis: np.array(common[axis], copy=True) for axis in "xyz"
        }
        component_coordinates[component][component] += deltas[component]
    shared = {}
    for axis in "xy":
        low = max(
            coordinates[axis][0]
            for coordinates in component_coordinates.values()
        )
        high = min(
            coordinates[axis][-1]
            for coordinates in component_coordinates.values()
        )
        shared[axis] = common[axis][
            (common[axis] >= low) & (common[axis] <= high)
        ]
    if any(
        not (
            coordinates["z"][0] <= z_target_m <= coordinates["z"][-1]
        )
        for coordinates in component_coordinates.values()
    ):
        raise RuntimeError("flake midplane is outside a component Yee support")
    plane_x, plane_y = np.meshgrid(
        shared["x"], shared["y"], indexing="ij"
    )
    points = np.column_stack(
        (
            plane_x.reshape(-1),
            plane_y.reshape(-1),
            np.full(plane_x.size, z_target_m),
        )
    )
    collocated: dict[str, np.ndarray] = {}
    native_contract: dict[str, Any] = {}
    for component in "xyz":
        values = np.asarray(
            fdtd.getdata(base.PABS_FIELD, f"E{component}", 1)
        ).squeeze()
        coordinates = component_coordinates[component]
        interpolator = RegularGridInterpolator(
            tuple(coordinates[axis] for axis in "xyz"),
            values,
            method="linear",
            bounds_error=True,
        )
        collocated[component] = interpolator(points).reshape(plane_x.shape)
        native_contract[component] = {
            "shape": list(values.shape),
            "bounds_m": {
                axis: [
                    float(coordinates[axis][0]),
                    float(coordinates[axis][-1]),
                ]
                for axis in "xyz"
            },
            "staggering_axis": component,
        }
    return {
        "x_m": shared["x"],
        "y_m": shared["y"],
        "z_m": z_target_m,
        "electric": collocated,
        "native_component_contract": native_contract,
        "collocation": (
            "Each native Yee E component was read with its own delta-c shifted "
            "coordinates and linearly interpolated to the common x/y plane at "
            "the physical flake midplane. The plane is restricted to the exact "
            "intersection of all component supports; same-index component "
            "pairing and extrapolation were not used."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.expanduser().resolve()
    case_path = artifact_dir / "case_result.json"
    fsp_path = artifact_dir / "finite_2um_optical_q.fsp"
    if not case_path.is_file() or not fsp_path.is_file():
        raise FileNotFoundError("completed case_result.json and FSP are required")
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else artifact_dir / "readonly_field_audit_v2"
    )
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    case = json.loads(case_path.read_text(encoding="utf-8"))
    if case.get("status") != "COMPLETED":
        raise RuntimeError("field extraction requires a completed case")
    if case.get("case") != "finite-flake":
        raise RuntimeError("field extraction is restricted to material cases")
    if not case["pre_run_contract"]["checks"]["all"]:
        raise RuntimeError("saved pre-run contract did not pass")

    os.environ["VC_LUMERICAL_ROOT"] = str(runner.APPROVED_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(runner.APPROVED_ROOT)
    os.environ["LUMERICAL_PYTHONPATH"] = str(runner.APPROVED_API)
    if str(runner.APPROVED_API) not in sys.path:
        sys.path.insert(0, str(runner.APPROVED_API))
    base = runner.load_base()
    installation = SimpleNamespace(
        version_key="v261",
        root=runner.APPROVED_ROOT.resolve(),
        lumapi_path=(runner.APPROVED_API / "lumapi.py").resolve(),
        device_executable=(runner.APPROVED_ROOT / "bin" / "device").resolve(),
    )
    lumapi = base.load_lumapi(installation)

    fdtd = None
    try:
        fdtd = lumapi.FDTD(
            str(fsp_path),
            hide=True,
            serverArgs={"platform": "offscreen"},
        )
        target = monitor_fields(fdtd, base.INCIDENT_REFERENCE_MONITOR)
        upper = monitor_fields(fdtd, "finite_E_xy_inside")
        flake = collocated_flake_midplane_fields(
            base,
            fdtd,
            -0.5 * runner.FLAKE_THICKNESS_M,
        )
        source_result = fdtd.getresult(base.SOURCE_NAME, "fields")
    finally:
        if fdtd is not None:
            fdtd.close()

    target_x = target["coordinates"]["x"]
    target_y = target["coordinates"]["y"]
    target_weights = weights_2d(target_x, target_y)
    eta0 = float(base.ETA0)
    ex_down = 0.5 * (
        target["electric"]["x"] - eta0 * target["magnetic"]["y"]
    )
    ey_down = 0.5 * (
        target["electric"]["y"] + eta0 * target["magnetic"]["x"]
    )
    downward_intensity_native = (
        np.abs(ex_down) ** 2 + np.abs(ey_down) ** 2
    ) / (2.0 * eta0)
    downward_intensity_native = np.asarray(
        downward_intensity_native, float
    ).reshape(target_x.size, target_y.size)
    target_e2 = np.asarray(
        sum(np.abs(target["electric"][axis]) ** 2 for axis in "xyz"),
        float,
    ).reshape(target_x.size, target_y.size)
    scale = float(case["run_result"]["normalization"]["scale_to_1_W_m2"])
    target_fit = runner.fit_elliptical_gaussian(
        target_x,
        target_y,
        downward_intensity_native * scale,
    )
    total_field_fit = runner.fit_elliptical_gaussian(
        target_x,
        target_y,
        target_e2 * scale,
    )
    flake_weights = weights_2d(flake["x_m"], flake["y_m"])
    flake_e2 = np.asarray(
        sum(np.abs(flake["electric"][axis]) ** 2 for axis in "xyz"),
        float,
    )
    flake_fit = runner.fit_elliptical_gaussian(
        flake["x_m"],
        flake["y_m"],
        flake_e2 * scale,
    )
    flake_fit["integrated_total_E2_area_V2_at_1_W_m2_center"] = (
        flake_fit.pop("integrated_incident_power_W")
    )

    upper_x = upper["coordinates"]["x"]
    upper_y = upper["coordinates"]["y"]
    upper_weights = weights_2d(upper_x, upper_y)
    source_E = np.asarray(source_result["E"])
    field_path = output / "saved_field_monitor_audit.npz"
    np.savez_compressed(
        field_path,
        target_x_m=target_x,
        target_y_m=target_y,
        target_z_m=target["coordinates"]["z"],
        target_Ex=target["electric"]["x"],
        target_Ey=target["electric"]["y"],
        target_Ez=target["electric"]["z"],
        target_Hx=target["magnetic"]["x"],
        target_Hy=target["magnetic"]["y"],
        target_Hz=target["magnetic"]["z"],
        target_downward_Ex=ex_down,
        target_downward_Ey=ey_down,
        target_downward_intensity_native_W_m2=downward_intensity_native,
        flake_x_m=flake["x_m"],
        flake_y_m=flake["y_m"],
        flake_z_m=np.asarray([flake["z_m"]]),
        flake_Ex_collocated=flake["electric"]["x"],
        flake_Ey_collocated=flake["electric"]["y"],
        flake_Ez_collocated=flake["electric"]["z"],
        upper_x_m=upper_x,
        upper_y_m=upper_y,
        upper_z_m=upper["coordinates"]["z"],
        upper_Ex=upper["electric"]["x"],
        upper_Ey=upper["electric"]["y"],
        upper_Ez=upper["electric"]["z"],
        upper_Hx=upper["magnetic"]["x"],
        upper_Hy=upper["magnetic"]["y"],
        upper_Hz=upper["magnetic"]["z"],
        source_profile_x_m=np.asarray(source_result["x"], float).reshape(-1),
        source_profile_y_m=np.asarray(source_result["y"], float).reshape(-1),
        source_profile_z_m=np.asarray(source_result["z"], float).reshape(-1),
        source_profile_E=source_E,
    )

    target_downward_power = float(
        np.sum(target_weights * downward_intensity_native) * scale
    )
    payload = {
        "status": "COMPLETED_READ_ONLY_FIELD_EXTRACTION",
        "case": {
            "geometry": case["pre_run_contract"]["geometry"]["geometry_name"],
            "polarization_deg": case["polarization_deg"],
            "generation_commit": case["generation_commit"],
        },
        "provenance": {
            "FDTD_solve_called": False,
            "runanalysis_called": False,
            "original_raw_artifacts_modified": False,
            "source_case_result": {
                "path": str(case_path),
                "size_bytes": case_path.stat().st_size,
                "sha256": base.sha256(case_path),
            },
            "source_FSP": {
                "path": str(fsp_path),
                "size_bytes": fsp_path.stat().st_size,
                "sha256": base.sha256(fsp_path),
            },
        },
        "near_stack_total_field_plane": {
            "z_m": float(target["coordinates"]["z"][0]),
            "requested_z_m": runner.INCIDENT_Z_M,
            "bounds_m": {
                "x": [float(target_x[0]), float(target_x[-1])],
                "y": [float(target_y[0]), float(target_y[-1])],
            },
            "total_field_component_metrics_native": {
                "electric": component_metrics(
                    target["electric"], target_weights
                ),
                "magnetic": component_metrics(
                    target["magnetic"], target_weights
                ),
            },
            "downward_decomposition_fit_at_1_W_m2_center": target_fit,
            "total_E2_fit_at_1_W_m2_center": total_field_fit,
            "integrated_downward_decomposition_power_W_at_1_W_m2_center": (
                target_downward_power
            ),
            "interpretation": (
                "This production incident-reference monitor contains total "
                "fields. Its realized Yee-plane coordinate is recorded rather "
                "than silently replaced by the requested z. The homogeneous-"
                "air downward E/H decomposition can include reflected, "
                "scattered, and evanescent contributions and is not called a "
                "pure incident-beam waist or power."
            ),
        },
        "flake_midplane_total_field": {
            "z_m": flake["z_m"],
            "bounds_m": {
                "x": [float(flake["x_m"][0]), float(flake["x_m"][-1])],
                "y": [float(flake["y_m"][0]), float(flake["y_m"][-1])],
            },
            "component_metrics_native_amplitude_collocated": component_metrics(
                flake["electric"], flake_weights
            ),
            "total_E2_spatial_fit_at_1_W_m2_center": flake_fit,
            "native_component_contract": flake[
                "native_component_contract"
            ],
            "collocation": flake["collocation"],
            "interpretation": (
                "This is the total Maxwell field inside TaIrTe4 at z=-65 nm. "
                "The fitted width is a spatial total-E2 diagnostic, not an "
                "incident Gaussian waist."
            ),
        },
        "upper_total_field_plane": {
            "z_m": float(upper["coordinates"]["z"][0]),
            "bounds_m": {
                "x": [float(upper_x[0]), float(upper_x[-1])],
                "y": [float(upper_y[0]), float(upper_y[-1])],
            },
            "component_metrics_native": {
                "electric": component_metrics(
                    upper["electric"], upper_weights
                ),
                "magnetic": component_metrics(
                    upper["magnetic"], upper_weights
                ),
            },
        },
        "certified_matching_empty_reference": {
            "incident_power_W_at_1_W_m2_center": case["run_result"][
                "normalization"
            ]["incident_power_W_at_1_W_m2"],
            "normalization_basis": case["run_result"]["normalization"][
                "normalization_basis"
            ],
            "empirical_flux_gain": False,
        },
        "source_profile": {
            "E_shape": list(source_E.shape),
            "all_finite": bool(np.all(np.isfinite(source_E))),
        },
        "field_artifact": {
            "path": str(field_path),
            "size_bytes": field_path.stat().st_size,
            "sha256": base.sha256(field_path),
        },
    }
    summary_path = output / "saved_field_monitor_audit.json"
    base.write_json(summary_path, payload)
    print(json.dumps(base.jsonable(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
