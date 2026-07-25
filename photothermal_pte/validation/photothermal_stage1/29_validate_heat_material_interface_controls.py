#!/usr/bin/env python3
"""Fail-closed v261 HEAT material/interface solver controls.

This workflow never runs the full finite device.  It first authenticates the
validated finite optical-Q artifact, then separately probes DEVICE licensing,
anisotropic thermal conductivity, and internal thermal-interface conductance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

import config_stage1 as config
from lumerical_api import (
    flatten_strings,
    jsonable,
    open_device,
    select_installation,
    utc_timestamp,
    write_json,
)


PR3_COMMIT = "053260da6fd0caec28ce155221bd18f683a0e5e7"
PR3_MANIFEST = (
    "photothermal_pte/reports/finite_2um_optical_q/"
    "RAW_ARTIFACT_MANIFEST.json"
)
EXPECTED_Q_SHA256 = (
    "7a63f82842751e7623e895701bac4ce92558679ed71bf13a3d404695a150e794"
)
EXPECTED_P_Q_W = 2.56071371086521e-12
POWER_LIMIT = 0.005
EXACT_BOUNDS_M = {
    "x": [-1.0e-6, 1.0e-6],
    "y": [-1.0e-6, 1.0e-6],
    "z": [-1.0e-7, 0.0],
}
REQUIRED_Q_FIELDS = (
    "x_m",
    "y_m",
    "z_m",
    "Q_on_W_m3",
    "Qx_W_m3",
    "Qy_W_m3",
    "Qz_W_m3",
    "exact_flake_mask",
    "incident_intensity_W_m2",
    "P_abs_volume_W",
    "P_abs_six_face_W",
    "absorption_cross_section_m2",
    "metadata_json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=(
            "q-precheck",
            "license-probe",
            "kappa-controls",
            "interface-controls",
            "all",
        ),
        default="q-precheck",
    )
    parser.add_argument(
        "--q-artifact",
        default=(
            str(config.OUTPUT_ROOT)
            + "/finite_convergence/domain/"
            "fixed_x_L16_pml24_dz5_w2_span6p8/finite_q_on_artifact.npz"
        ),
    )
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--lumerical-version", choices=("v261",), default="v261"
    )
    parser.add_argument(
        "--hide-gui", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(data: Any, key: str) -> float:
    values = np.asarray(data[key], float).reshape(-1)
    if values.size != 1 or not np.isfinite(values[0]):
        raise ValueError(f"{key} must contain one finite scalar")
    return float(values[0])


def integrate_3d(
    values: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> float:
    return float(
        np.trapezoid(
            np.trapezoid(np.trapezoid(values, z, axis=2), y, axis=1),
            x,
            axis=0,
        )
    )


def read_pr3_manifest() -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "show", f"{PR3_COMMIT}:{PR3_MANIFEST}"],
        cwd=config.REPOSITORY_ROOT.parent,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"cannot read PR #3 manifest at {PR3_COMMIT}: "
            f"{completed.stderr.strip()}"
        )
    manifest = json.loads(completed.stdout)
    matches = [
        item
        for item in manifest.get("artifacts", [])
        if item.get("sha256") == EXPECTED_Q_SHA256
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "PR #3 manifest must contain exactly one artifact with expected SHA"
        )
    return {
        "commit": PR3_COMMIT,
        "repository_path": PR3_MANIFEST,
        "entry": matches[0],
    }


def audit_q_artifact(path: Path) -> dict[str, Any]:
    manifest = read_pr3_manifest()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = sha256(path)
    if actual_sha != EXPECTED_Q_SHA256:
        raise RuntimeError(
            f"Q SHA mismatch: expected {EXPECTED_Q_SHA256}, got {actual_sha}"
        )

    with np.load(path, allow_pickle=False) as data:
        missing = sorted(set(REQUIRED_Q_FIELDS) - set(data.files))
        if missing:
            raise ValueError(f"Q artifact lacks required fields: {missing}")
        x, y, z = (
            np.asarray(data[key], float).reshape(-1)
            for key in ("x_m", "y_m", "z_m")
        )
        arrays = {
            key: np.asarray(data[key], float)
            for key in ("Qx_W_m3", "Qy_W_m3", "Qz_W_m3", "Q_on_W_m3")
        }
        shape = (x.size, y.size, z.size)
        if any(array.shape != shape for array in arrays.values()):
            raise ValueError("Q component shapes do not match x/y/z coordinates")
        if any(not np.all(np.diff(axis) > 0.0) for axis in (x, y, z)):
            raise ValueError("Q coordinates are not strictly increasing")
        if any(not np.all(np.isfinite(array)) for array in arrays.values()):
            raise ValueError("Q component contains NaN or Inf")
        if any(not np.all(np.isfinite(axis)) for axis in (x, y, z)):
            raise ValueError("Q coordinate contains NaN or Inf")

        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        if metadata.get("array_axis_order") != ["x", "y", "z"]:
            raise ValueError("artifact array_axis_order is not x,y,z")
        if metadata.get("exact_flake_bounds_m") != EXACT_BOUNDS_M:
            raise ValueError("artifact exact flake bounds do not match 2 um footprint")
        forbidden = {
            "clipping": bool(metadata.get("clipped", False)),
            "gain": bool(metadata.get("gain_applied", False)),
            "rescaling": bool(metadata.get("rescaled_to_flux", False)),
            "crop": bool(metadata.get("periodic_crop", False)),
            "tiling": bool(metadata.get("periodic_tiling", False)),
            "smoothing": bool(metadata.get("smoothing_applied", False)),
        }
        if any(forbidden.values()):
            raise ValueError(f"forbidden Q operation recorded: {forbidden}")

        expected_mask = (
            (x[:, None, None] >= EXACT_BOUNDS_M["x"][0])
            & (x[:, None, None] <= EXACT_BOUNDS_M["x"][1])
            & (y[None, :, None] >= EXACT_BOUNDS_M["y"][0])
            & (y[None, :, None] <= EXACT_BOUNDS_M["y"][1])
            & (z[None, None, :] >= EXACT_BOUNDS_M["z"][0])
            & (z[None, None, :] <= EXACT_BOUNDS_M["z"][1])
        )
        stored_mask = np.asarray(data["exact_flake_mask"], bool)
        if stored_mask.shape != shape or not np.array_equal(
            stored_mask, expected_mask
        ):
            raise ValueError("stored exact_flake_mask does not match coordinates")

        component_sum = (
            arrays["Qx_W_m3"]
            + arrays["Qy_W_m3"]
            + arrays["Qz_W_m3"]
        )
        q = arrays["Q_on_W_m3"]
        component_error = float(
            np.max(np.abs(component_sum - q))
            / max(float(np.max(np.abs(q))), np.finfo(float).tiny)
        )
        if component_error > 1.0e-12:
            raise ValueError("Qx+Qy+Qz does not reproduce Q_on")
        coordinate_tolerance_m = 1.0e-15
        physical_mask_with_roundoff = (
            (x[:, None, None] >= EXACT_BOUNDS_M["x"][0] - coordinate_tolerance_m)
            & (x[:, None, None] <= EXACT_BOUNDS_M["x"][1] + coordinate_tolerance_m)
            & (y[None, :, None] >= EXACT_BOUNDS_M["y"][0] - coordinate_tolerance_m)
            & (y[None, :, None] <= EXACT_BOUNDS_M["y"][1] + coordinate_tolerance_m)
            & (z[None, None, :] >= EXACT_BOUNDS_M["z"][0] - coordinate_tolerance_m)
            & (z[None, None, :] <= EXACT_BOUNDS_M["z"][1] + coordinate_tolerance_m)
        )
        outside_max = float(
            np.max(np.abs(q[~physical_mask_with_roundoff]))
        )
        if outside_max != 0.0:
            raise ValueError("Q is nonzero outside the exact TaIrTe4 mask")

        incident_intensity = scalar(data, "incident_intensity_W_m2")
        if incident_intensity != 1.0:
            raise ValueError("Q artifact is not normalized to 1 W/m2")
        p_metadata = scalar(data, "P_abs_volume_W")
        p_reintegrated = integrate_3d(q, x, y, z)
        reintegration_error = (
            abs(p_reintegrated - EXPECTED_P_Q_W) / EXPECTED_P_Q_W
        )
        if reintegration_error >= POWER_LIMIT:
            raise ValueError(
                f"Q reintegration error {reintegration_error} exceeds {POWER_LIMIT}"
            )

    return {
        "status": "Q_ARTIFACT_COMPATIBLE_2UM_FOOTPRINT",
        "blocker_release_candidate": (
            "BLOCKED_Q_ARTIFACT_INCOMPATIBLE_WITH_2UM_FOOTPRINT"
        ),
        "blocker_release_allowed": True,
        "pr3_manifest": manifest,
        "artifact": {
            "server_path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": actual_sha,
        },
        "required_fields": list(REQUIRED_Q_FIELDS),
        "shape_xyz": list(shape),
        "coordinate_order": ["x", "y", "z"],
        "coordinate_ranges_m": {
            "x": [float(x[0]), float(x[-1])],
            "y": [float(y[0]), float(y[-1])],
            "z": [float(z[0]), float(z[-1])],
        },
        "exact_TaIrTe4_bounds_m": EXACT_BOUNDS_M,
        "exact_mask_voxels": int(np.count_nonzero(stored_mask)),
        "coordinate_roundoff_tolerance_m": coordinate_tolerance_m,
        "roundoff_inclusive_physical_voxels": int(
            np.count_nonzero(physical_mask_with_roundoff)
        ),
        "Q_component_sum_max_relative_error": component_error,
        "maximum_Q_outside_exact_mask_W_m3": outside_max,
        "incident_intensity_W_m2": incident_intensity,
        "P_Q_metadata_W": p_metadata,
        "P_Q_expected_W": EXPECTED_P_Q_W,
        "P_Q_reintegrated_W": p_reintegrated,
        "reintegration_relative_error": reintegration_error,
        "reintegration_limit": POWER_LIMIT,
        "no_NaN_or_Inf": True,
        "forbidden_operations_applied": forbidden,
    }


def set_object(obj: Any, values: dict[str, Any]) -> None:
    for name, value in values.items():
        obj[name] = value


def add_solid_material(
    device: Any, name: str, conductivity: float | np.ndarray
) -> dict[str, Any]:
    device.addmodelmaterial()
    device.set("name", name)
    device.addhtmaterialproperty("Solid")
    property_name = f"{name} thermal"
    device.set("name", property_name)
    device.set("thermal conductivity.active model", "constant")
    device.set("thermal conductivity.constant", conductivity)
    returned = np.asarray(
        device.get("thermal conductivity.constant"), float
    ).reshape(-1)
    return {
        "model_material": name,
        "thermal_property": property_name,
        "conductivity_write_W_mK": np.asarray(
            conductivity, float
        ).reshape(-1).tolist(),
        "conductivity_readback_W_mK": returned.tolist(),
    }


def license_probe(output: Path, *, hide: bool) -> dict[str, Any]:
    installation = select_installation("v261")
    project = output / "device_license_probe.ldev"
    result: dict[str, Any] = {
        "status": "BLOCKED_LUMERICAL_LICENSE_UNAVAILABLE",
        "installation_root": str(installation.root),
        "lumapi_path": str(installation.lumapi_path),
        "device_executable": str(installation.device_executable),
        "session_startup": False,
        "save_succeeded": False,
        "load_succeeded": False,
        "solver_run_succeeded": False,
        "project_path": str(project),
    }
    try:
        with open_device(installation, hide=hide) as device:
            result["session_startup"] = True
            result["lumerical_version"] = str(device.version())
            region = device.addsimulationregion()
            set_object(
                region,
                {
                    "name": "license probe region",
                    "dimension": "3D",
                    "use relative coordinates": False,
                    "x": 0.0,
                    "x span": 1.0e-6,
                    "y": 0.0,
                    "y span": 1.0e-6,
                    "z min": 0.0,
                    "z max": 1.0e-6,
                },
            )
            result["material"] = add_solid_material(
                device, "license probe solid", 10.0
            )
            slab = device.addrect()
            set_object(
                slab,
                {
                    "name": "license probe slab",
                    "material": "license probe solid",
                    "x": 0.0,
                    "x span": 1.0e-6,
                    "y": 0.0,
                    "y span": 1.0e-6,
                    "z min": 0.0,
                    "z max": 1.0e-6,
                },
            )
            heat = device.addheatsolver()
            set_object(
                heat,
                {
                    "simulation region": "license probe region",
                    "solver mode": "steady state",
                    "solver physics": "thermal only",
                    "use defaults": False,
                    "min edge length": 25.0e-9,
                    "max edge length": 100.0e-9,
                    "max plc edge length": 100.0e-9,
                },
            )
            source = device.adduniformheat()
            set_object(
                source,
                {
                    "name": "license probe heat",
                    "source type": "3D",
                    "geometry type": "directly defined",
                    "x": 0.0,
                    "x span": 1.0e-6,
                    "y": 0.0,
                    "y span": 1.0e-6,
                    "z min": 0.0,
                    "z max": 1.0e-6,
                    "total power": 1.0e-6,
                },
            )
            device.addtemperaturebc("HEAT")
            for name, value in {
                "name": "license probe T bottom",
                "bc mode": "steady state",
                "sweep type": "single",
                "temperature": 300.0,
                "surface type": "simulation region",
                "z min": True,
            }.items():
                device.set(name, value)
            monitor = device.addtemperaturemonitor()
            set_object(
                monitor,
                {
                    "name": "license probe T",
                    "monitor type": "linear z",
                    "x": 0.0,
                    "y": 0.0,
                    "z min": 0.0,
                    "z max": 1.0e-6,
                },
            )
            device.save(str(project))
            result["save_succeeded"] = project.is_file()
            device.load(str(project))
            result["load_succeeded"] = True
            device.addjob(str(project), "HEAT")
            device.runjobs("HEAT", 0)
            device.load(str(project))
            available = flatten_strings(
                device.getresult("HEAT::license probe T")
            )
            result["temperature_monitor_results"] = available
            temperature_result = None
            for candidate in ("temperature", "thermal"):
                try:
                    temperature_result = device.getresult(
                        "HEAT::license probe T", candidate
                    )
                    result["temperature_result_name"] = candidate
                    break
                except Exception:
                    continue
            if temperature_result is None:
                raise RuntimeError(
                    f"no temperature result after solve; available={available}"
                )
            temperatures = np.asarray(
                temperature_result.get(
                    "T", temperature_result.get("temperature")
                ),
                float,
            ).reshape(-1)
            if temperatures.size == 0 or not np.all(np.isfinite(temperatures)):
                raise RuntimeError("temperature result is empty or non-finite")
            result["temperature_range_K"] = [
                float(np.min(temperatures)),
                float(np.max(temperatures)),
            ]
            result["solver_run_succeeded"] = True
            result["status"] = "DEVICE_LICENSE_API_SOLVER_AVAILABLE"
    except Exception as exc:
        result.update(
            {
                "status": (
                    "DEVICE_API_OR_SOLVER_PROBE_FAILED"
                    if result["session_startup"]
                    else "BLOCKED_LUMERICAL_LICENSE_UNAVAILABLE"
                ),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    result["passed"] = all(
        bool(result[key])
        for key in (
            "session_startup",
            "save_succeeded",
            "load_succeeded",
            "solver_run_succeeded",
        )
    )
    if not result["passed"] and not result["session_startup"]:
        result["status"] = "BLOCKED_LUMERICAL_LICENSE_UNAVAILABLE"
    return result


def output_directory(explicit: str | None) -> Path:
    if explicit:
        output = Path(explicit).expanduser().resolve()
    else:
        output = (
            config.OUTPUT_ROOT
            / f"{utc_timestamp()}_heat_material_interface_controls"
        )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def main() -> int:
    args = parse_args()
    output = output_directory(args.output_dir)
    command = shlex.join([sys.executable, *sys.argv])
    result: dict[str, Any] = {
        "phase": args.phase,
        "generation_command": command,
        "full_device_HEAT_executed": False,
        "transient_executed": False,
        "PTE_executed": False,
        "optimization_executed": False,
    }
    if args.phase not in ("q-precheck", "license-probe", "all"):
        raise NotImplementedError(
            f"{args.phase} will be enabled after the Q precheck checkpoint"
        )
    if args.phase in ("q-precheck", "all"):
        result["q_precheck"] = audit_q_artifact(
            Path(args.q_artifact).expanduser().resolve()
        )
    if args.phase in ("license-probe", "all"):
        result["license_probe"] = license_probe(
            output, hide=args.hide_gui
        )
    q_pass = (
        args.phase == "license-probe"
        or result["q_precheck"]["blocker_release_allowed"]
    )
    license_pass = (
        args.phase == "q-precheck"
        or result["license_probe"]["passed"]
    )
    if not license_pass:
        result["status"] = "BLOCKED_LUMERICAL_LICENSE_UNAVAILABLE"
    elif q_pass:
        result["status"] = (
            "DEVICE_LICENSE_API_PROBE_PASSED"
            if args.phase in ("license-probe", "all")
            else "Q_PRECHECK_PASSED"
        )
    write_json(output / f"{args.phase.replace('-', '_')}.json", result)
    print(json.dumps(jsonable(result), indent=2), flush=True)
    return 0 if q_pass and license_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
