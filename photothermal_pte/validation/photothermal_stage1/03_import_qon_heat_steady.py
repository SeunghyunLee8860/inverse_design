#!/usr/bin/env python3
"""Import the physical FDTD Q_on dataset into a steady-state HEAT model."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config_stage1 as config
from lumerical_api import Stage1Error, jsonable, open_device, select_installation, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lumerical-version", choices=("auto", "v261", "v251"), default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pipeline-placeholder", action="store_true")
    parser.add_argument("--hide-gui", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def trapz3(values: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    trap = np.trapezoid
    return float(trap(trap(trap(values, z, axis=2), y, axis=1), x, axis=0))


def set_object(obj: Any, values: dict[str, Any]) -> None:
    for name, value in values.items():
        obj[name] = value


def npz_scalar(data: Any, name: str) -> float:
    if name not in data:
        raise Stage1Error(f"Q artifact is missing required scalar {name!r}")
    value = np.asarray(data[name]).reshape(-1)
    if value.size != 1 or not np.isfinite(float(value[0])):
        raise Stage1Error(f"Q artifact scalar {name!r} is invalid")
    return float(value[0])


def thermal_inputs(placeholder: bool) -> tuple[dict[str, dict[str, Any]], str]:
    mode = "pipeline_placeholder" if placeholder else config.DESIGN_THERMAL_MODE
    design_k = config.DESIGN_K_W_MK
    label = ""
    if mode == "pipeline_placeholder":
        design_k = config.PLACEHOLDER_DESIGN_K_W_MK
        label = config.PLACEHOLDER_LABEL
    missing = []
    for name, value in (
        ("design.k", design_k),
        ("SiO2.k", config.SIO2_K_W_MK),
        ("Si.k", config.SI_K_W_MK),
    ):
        if value is None:
            missing.append(name)
    if missing:
        raise Stage1Error(
            "missing required physical thermal conductivity: "
            + ", ".join(missing)
            + "; provide STAGE1_*_K_W_MK variables. No database/default value was used."
        )
    return {
        "design": {"k": float(design_k), "rho": config.DESIGN_RHO_KG_M3, "cp": config.DESIGN_CP_J_KG_K},
        "TaIrTe4": {"k": list(config.TAIRTE4_K_DIAGONAL_W_MK), "rho": config.TAIRTE4_RHO_KG_M3, "cp": config.TAIRTE4_CP_J_KG_K},
        "SiO2": {"k": float(config.SIO2_K_W_MK), "rho": config.SIO2_RHO_KG_M3, "cp": config.SIO2_CP_J_KG_K},
        "Si": {"k": float(config.SI_K_W_MK), "rho": config.SI_RHO_KG_M3, "cp": config.SI_CP_J_KG_K},
    }, label


def add_solid_material(
    device: Any,
    name: str,
    conductivity: float | list[float],
    rho: float | None,
    cp: float | None,
) -> dict[str, Any]:
    device.addmodelmaterial()
    device.set("name", name)
    device.addhtmaterialproperty("Solid")
    device.set("name", f"{name} thermal")
    device.set("thermal conductivity.active model", "constant")
    requested = np.asarray(conductivity, dtype=float)
    if requested.size == 1:
        device.set("thermal conductivity.constant", float(requested.reshape(-1)[0]))
        returned = np.asarray(device.get("thermal conductivity.constant"), float).reshape(-1)
        if returned.size != 1 or not np.isclose(returned[0], requested.reshape(-1)[0]):
            raise Stage1Error(f"{name}: scalar thermal conductivity did not round-trip")
    else:
        # This is a capability gate, not a scalar approximation.  v261 currently
        # returns scalar zero for both a 3-vector and a 3x3 diagonal matrix.
        device.set("thermal conductivity.constant", requested)
        returned = np.asarray(device.get("thermal conductivity.constant"), float)
        if returned.shape != requested.shape or not np.allclose(returned, requested):
            raise Stage1Error(
                f"{name}: diagonal anisotropic thermal conductivity is unsupported by "
                f"this HEAT Solid property; requested={requested.tolist()}, "
                f"round_trip={returned.tolist()}. Refusing an isotropic approximation."
            )
    if rho is not None:
        device.set("mass density.constant", float(rho))
    if cp is not None:
        device.set("specific heat.active model", "constant")
        device.set("specific heat.constant", float(cp))
    return {
        "name": name,
        "thermal_conductivity_W_mK": requested.tolist() if requested.size > 1 else float(requested[0]),
        "rho_kg_m3": rho,
        "cp_J_kg_K": cp,
    }


def get_dataset(device: Any, object_name: str, result_name: str) -> dict[str, Any]:
    result = device.getresult(object_name, result_name)
    if not isinstance(result, dict):
        raise Stage1Error(f"{object_name}.{result_name} did not return a dataset")
    return result


def save_temperature_figure(path: Path, x: np.ndarray, y: np.ndarray, t: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    image = ax.pcolormesh(x * 1e6, y * 1e6, np.asarray(t).T, shading="auto")
    fig.colorbar(image, ax=ax, label="T [K]")
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    installation = select_installation(args.lumerical_version)
    source_path = output / "fdtd_qon" / "q_on_physical.npz"
    stage_dir = output / "heat_steady"
    stage_dir.mkdir(parents=True, exist_ok=True)
    summary_path = stage_dir / "heat_steady_summary.json"

    label = ""
    try:
        if not source_path.is_file():
            raise Stage1Error(f"missing FDTD source artifact: {source_path}")
        fdtd_summary_path = output / "fdtd_qon" / "fdtd_absorption_summary.json"
        if not fdtd_summary_path.is_file():
            raise Stage1Error(f"missing FDTD validation summary: {fdtd_summary_path}")
        fdtd_summary = json.loads(fdtd_summary_path.read_text())
        if not fdtd_summary.get("validated", False):
            raise Stage1Error("FDTD Q_on did not pass optical energy validation")
        data = np.load(source_path)
        x = np.asarray(data["x_m"], float).reshape(-1)
        y = np.asarray(data["y_m"], float).reshape(-1)
        z = np.asarray(data["z_m"], float).reshape(-1)
        q = np.asarray(data["Q_on_W_m3"], float)
        if q.shape != (x.size, y.size, z.size):
            raise Stage1Error(f"Q shape {q.shape} does not match x/y/z")
        if any(np.any(np.diff(axis) <= 0) for axis in (x, y, z)):
            raise Stage1Error("import coordinates are not strictly increasing")
        if not np.all(np.isfinite(q)):
            raise Stage1Error("Q contains NaN/Inf")
        p_fdtd = trapz3(q, x, y, z)
        incident_intensity = npz_scalar(data, "incident_intensity_W_m2")
        incident_power = npz_scalar(data, "incident_power_W")
        unit_cell_area = npz_scalar(data, "unit_cell_area_m2")
        expected_absorbed_power = npz_scalar(data, "expected_AQ_times_Pincident_W")
        exported_power = npz_scalar(data, "P_abs_volume_W")
        tiny = np.finfo(float).tiny
        source_identity_error = abs(p_fdtd - expected_absorbed_power) / max(
            abs(expected_absorbed_power), tiny
        )
        export_reintegration_error = abs(p_fdtd - exported_power) / max(
            abs(exported_power), tiny
        )
        flake_z_min = -100.0e-9
        flake_z_max = 0.0
        outside_flake = (z < flake_z_min) | (z > flake_z_max)
        outside_values = np.zeros_like(q)
        outside_values[:, :, outside_flake] = q[:, :, outside_flake]
        outside_flake_power = trapz3(outside_values, x, y, z)
        q_scale = max(float(np.max(np.abs(q))), tiny)
        seam_x_relative_max = float(np.max(np.abs(q[0, :, :] - q[-1, :, :])) / q_scale)
        seam_y_relative_max = float(np.max(np.abs(q[:, 0, :] - q[:, -1, :])) / q_scale)
        negative_values = np.minimum(q, 0.0)
        negative_power = trapz3(negative_values, x, y, z)
        preflight = {
            "status": "Q_ARTIFACT_PREFLIGHT_PASSED_THERMAL_PROPERTIES_NOT_YET_CHECKED",
            "source_artifact": str(source_path),
            "array_axis_order": ["x", "y", "z"],
            "coordinate_units": "m",
            "Q_units": "W/m^3",
            "shape_xyz": list(q.shape),
            "coordinates_strictly_increasing": True,
            "coordinate_ranges_m": {
                "x": [float(x[0]), float(x[-1])],
                "y": [float(y[0]), float(y[-1])],
                "z": [float(z[0]), float(z[-1])],
            },
            "periodic_seam": {
                "endpoint_coordinates_are_both_present": True,
                "x_endpoint_relative_max_difference": seam_x_relative_max,
                "y_endpoint_relative_max_difference": seam_y_relative_max,
                "integration_policy": "original coordinates and samples retained; no seam deletion or duplication",
            },
            "TaIrTe4_bounds_m": [flake_z_min, flake_z_max],
            "P_Q_original_grid_W": p_fdtd,
            "P_Q_exported_metadata_W": exported_power,
            "P_expected_AQ_times_Pincident_W": expected_absorbed_power,
            "export_reintegration_relative_error": export_reintegration_error,
            "source_power_identity_relative_error": source_identity_error,
            "incident_intensity_W_m2": incident_intensity,
            "unit_cell_area_m2": unit_cell_area,
            "incident_power_per_cell_W": incident_power,
            "normalization_mode": str(np.asarray(data["normalization_mode"]).reshape(-1)[0]),
            "unit_response_mode": bool(np.asarray(data["unit_response_mode"]).reshape(-1)[0]),
            "outside_TaIrTe4_power_W": outside_flake_power,
            "outside_TaIrTe4_power_fraction": outside_flake_power / max(abs(p_fdtd), tiny),
            "minimum_Q_W_m3": float(np.min(q)),
            "maximum_Q_W_m3": float(np.max(q)),
            "negative_voxel_count": int(np.count_nonzero(q < 0.0)),
            "integrated_negative_power_W": negative_power,
            "nan_or_inf_count": int(q.size - np.count_nonzero(np.isfinite(q))),
            "clipped": False,
            "smoothed": False,
            "rescaled_after_export": False,
        }
        write_json(stage_dir / "q_import_preflight.json", preflight)

        props, label = thermal_inputs(args.pipeline_placeholder)
        suffix = f"_{label}" if label else ""

        project_path = stage_dir / f"heat_steady{suffix}.ldev"
        with open_device(installation, hide=args.hide_gui) as device:
            material_records = {}
            for name in ("design", "TaIrTe4", "SiO2", "Si"):
                entry = props[name]
                material_records[name] = add_solid_material(
                    device, name, entry["k"], entry["rho"], entry["cp"]
                )

            period = 6.0e-6
            design_h = 0.6e-6
            flake_h = 0.1e-6
            sio2_h = 0.285e-6
            si_h = 2.0e-6
            z_bottom = -(flake_h + sio2_h + si_h)

            region = device.addsimulationregion()
            set_object(region, {
                "name": "stage1 physical region", "dimension": "3D",
                "use relative coordinates": False,
                "x": 0.0, "x span": period, "y": 0.0, "y span": period,
                "z min": z_bottom, "z max": design_h,
            })
            for name, material, z_min, z_max in (
                ("Si_substrate", "Si", z_bottom, -flake_h - sio2_h),
                ("SiO2_spacer", "SiO2", -flake_h - sio2_h, -flake_h),
                ("TaIrTe4_flake", "TaIrTe4", -flake_h, 0.0),
            ):
                rect = device.addrect()
                set_object(rect, {
                    "name": name, "material": material,
                    "x": 0.0, "x span": period, "y": 0.0, "y span": period,
                    "z min": z_min, "z max": z_max,
                })
            disk = device.addcircle()
            set_object(disk, {
                "name": "fixed_design_disk", "material": "design",
                "x": 0.0, "y": 0.0, "radius": config.FIXED_DESIGN_DISK_RADIUS_M,
                "z min": 0.0, "z max": design_h,
            })

            heat = device.addheatsolver()
            set_object(heat, {
                "simulation region": "stage1 physical region",
                "solver mode": "steady state", "solver physics": "thermal only",
            })
            source = device.addimportheat("HEAT")
            set_object(source, {
                "name": "Q_on_import", "scale factor": 1.0,
                "selected attribute": "Q",
            })
            device.putv("stage1_qx", x)
            device.putv("stage1_qy", y)
            device.putv("stage1_qz", z)
            device.putv("stage1_q", np.asfortranarray(q))
            device.eval(
                'stage1_Qds=rectilineardataset("Q_on",stage1_qx,stage1_qy,stage1_qz);'
                'stage1_Qds.addattribute("Q",stage1_q);'
                'select("HEAT::Q_on_import");importdataset(stage1_Qds);'
            )

            bc = device.addtemperaturebc("HEAT")
            set_object(bc, {
                "name": "T_bottom", "bc mode": "steady state",
                "sweep type": "single", "temperature": config.BATH_TEMPERATURE_K,
                "surface type": "simulation region", "z min": True,
            })
            mon_xy = device.addtemperaturemonitor("HEAT")
            set_object(mon_xy, {
                "name": "T_xy_tairte4", "monitor type": "2D z-normal",
                "x": 0.0, "x span": period, "y": 0.0, "y span": period,
                "z": -0.5 * flake_h,
            })
            mon_xz = device.addtemperaturemonitor("HEAT")
            set_object(mon_xz, {
                "name": "T_xz_center", "monitor type": "2D y-normal",
                "x": 0.0, "x span": period, "y": 0.0,
                "z min": z_bottom, "z max": design_h,
            })
            flux_xz = device.addheatfluxmonitor("HEAT")
            set_object(flux_xz, {
                "name": "heat_flux_xz", "monitor type": "2D y-normal",
                "x": 0.0, "x span": period, "y": 0.0,
                "z min": z_bottom, "z max": design_h,
            })
            device.save(str(project_path))
            device.addjob(str(project_path), "HEAT")
            device.runjobs("HEAT", 0)
            device.load(str(project_path))

            solver_results = str(device.getresult("HEAT"))
            thermal = get_dataset(device, "HEAT", "thermal")
            boundaries = get_dataset(device, "HEAT", "boundaries")
            integrated = get_dataset(device, "HEAT", "integrated")
            txy = get_dataset(device, "HEAT::T_xy_tairte4", "temperature")
            txz = get_dataset(device, "HEAT::T_xz_center", "temperature")
            flux_result_names = [name.strip() for name in str(device.getresult("HEAT::heat_flux_xz")).splitlines() if name.strip()]
            if not flux_result_names:
                raise Stage1Error("heat flux monitor returned no result")
            flux_result_name = "heat flux" if "heat flux" in flux_result_names else flux_result_names[0]
            heat_flux = get_dataset(device, "HEAT::heat_flux_xz", flux_result_name)

            p_import = float(np.asarray(integrated["Q"]).reshape(-1)[0])
            p_out = abs(float(np.asarray(boundaries["P_T_bottom"]).reshape(-1)[0]))
            import_error = abs(p_import - p_fdtd) / abs(p_fdtd)
            outflow_error = abs(p_out - p_import) / abs(p_import)
            temperature = np.asarray(thermal["T"], float).reshape(-1)
            if not np.all(np.isfinite(temperature)):
                raise Stage1Error("steady-state T contains NaN/Inf")
            t_min = float(np.min(temperature))
            t_max = float(np.max(temperature))
            if t_min < config.BATH_TEMPERATURE_K - 1e-6:
                raise Stage1Error(f"T_min={t_min} K is below the fixed bath")

            txy_x = np.asarray(txy["x"], float).reshape(-1)
            txy_y = np.asarray(txy["y"], float).reshape(-1)
            txy_T = np.asarray(txy["T"], float).squeeze()
            if txy_T.shape != (txy_x.size, txy_y.size):
                raise Stage1Error(f"unexpected TaIrTe4 monitor shape {txy_T.shape}")
            edge_order = 2 if min(txy_T.shape) >= 3 else 1
            grad_x, grad_y = np.gradient(txy_T, txy_x, txy_y, edge_order=edge_order)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            np.savez_compressed(
                stage_dir / f"temperature_steady{suffix}.npz",
                **{k: v for k, v in thermal.items() if k != "Lumerical_dataset"},
                T_xy_tairte4_K=txy_T,
                T_xy_x_m=txy_x,
                T_xy_y_m=txy_y,
                T_xz_K=np.asarray(txz["T"], float),
                T_xz_x_m=np.asarray(txz["x"], float),
                T_xz_z_m=np.asarray(txz["z"], float),
            )
            flux_arrays = {str(key): value for key, value in heat_flux.items() if key != "Lumerical_dataset"}
            np.savez_compressed(
                stage_dir / f"heat_flux_steady{suffix}.npz",
                **flux_arrays,
                result_name=flux_result_name,
                boundaries=json.dumps(jsonable(boundaries)),
                imported_power_W=p_import,
                bottom_outflow_power_W=p_out,
                gradT_x_K_m=grad_x,
                gradT_y_K_m=grad_y,
                gradT_magnitude_K_m=gradient_magnitude,
            )
            save_temperature_figure(
                stage_dir / f"T_xy_tairte4_midplane{suffix}.png",
                txy_x, txy_y, txy_T, label or "TaIrTe4 midplane",
            )
            device.save(str(project_path))

        passed = import_error < config.HEAT_IMPORT_POWER_ERROR_LIMIT and outflow_error < 0.05
        summary = {
            "status": "passed" if passed else "failed_acceptance_criteria",
            "validated": passed,
            "selected_version": installation.version_key,
            "placeholder_label": label or None,
            "thermal_properties": props,
            "anisotropic_conductivity_round_trip": "required and passed",
            "interfaces": "perfect thermal contact",
            "boundaries": {
                "bottom_Si": "fixed 300 K", "top": "adiabatic",
                "lateral": "adiabatic; validation cell, not a finite-device PTE model",
            },
            "import": {
                "x_range_m": [float(x[0]), float(x[-1])],
                "y_range_m": [float(y[0]), float(y[-1])],
                "z_range_m": [float(z[0]), float(z[-1])],
                "Q_min_W_m3": float(np.min(q)), "Q_max_W_m3": float(np.max(q)),
                "P_fdtd_abs_volume_W": p_fdtd, "P_heat_import_W": p_import,
                "import_power_error": import_error,
                "acceptance_limit": config.HEAT_IMPORT_POWER_ERROR_LIMIT,
                "interpolation": "Lumerical importdataset from the original rectilinear FDTD Pabs grid",
                "power_correction_factor": 1.0,
            },
            "steady": {
                "T_min_K": t_min, "T_max_K": t_max,
                "deltaT_max_K": t_max - config.BATH_TEMPERATURE_K,
                "T_avg_TaIrTe4_midplane_K": float(np.mean(txy_T)),
                "max_inplane_gradT_K_m": float(np.max(gradient_magnitude)),
                "bottom_outflow_power_W": p_out,
                "outflow_error": outflow_error,
                "solver_results": solver_results.splitlines(),
                "heat_flux_sign": "P_T_bottom is negative for heat leaving the domain; magnitude reported",
            },
        }
        write_json(summary_path, summary)
        print(json.dumps(jsonable(summary), indent=2), flush=True)
        return 0 if passed else 2
    except Exception as exc:
        failure = {
            "status": "failed",
            "validated": False,
            "selected_version": installation.version_key,
            "placeholder_label": label or None,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "scientific_result_available": False,
        }
        write_json(summary_path, failure)
        print(json.dumps(failure, indent=2), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
