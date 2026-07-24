#!/usr/bin/env python3
"""Export a physically normalized Q_on from the repository's fixed FDTD stack.

This is a forward-only validation.  It deliberately bypasses every mapping,
projection, optimizer, and adjoint path.  Spatial absorption is evaluated by
Lumerical's installed ``pabs_adv`` object-library analysis group, which treats
the three Yee-grid electric-field components separately before interpolation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

import config_stage1 as config
from lumerical_api import Stage1Error, jsonable, select_installation, write_json
from pabs_diagnostic_utils import (
    mesh_and_material_metadata, save_epsilon_and_material_slices,
)


PABS_NAME = "stage1_pabs_adv"
REFLECTION_MONITOR = "stage1_reflection"
TRANSMISSION_MONITOR = "stage1_transmission"
LOCAL_TOP_MONITOR = "stage1_flux_top"
LOCAL_BOTTOM_MONITOR = "stage1_flux_bottom"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lumerical-version", choices=("auto", "v261", "v251"), default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hide-gui", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--fixed-geometry",
        choices=("centered-disk", "none", "full-film", "centered-square", "centered-disk-gap"),
        default="centered-disk",
        help="Use the validation disk or make the entire design region air.",
    )
    parser.add_argument("--global-resolution", type=int)
    parser.add_argument("--design-resolution-xy", type=int)
    parser.add_argument("--flake-dz-nm", type=float)
    parser.add_argument("--pabs-z-padding-nm", type=float, default=50.0)
    parser.add_argument("--diagnostic-air-gap-nm", type=float, default=50.0)
    parser.add_argument(
        "--allow-nonpositive-diagnostic-flux",
        action="store_true",
        help="Record zero/negative diagnostic flux without clipping or gain.",
    )
    parser.add_argument("--postprocess-project", help="Completed FDTD .fsp to copy and postprocess without rerunning Maxwell")
    return parser.parse_args()


def require_increasing(name: str, values: Any) -> np.ndarray:
    axis = np.asarray(values, dtype=float).reshape(-1)
    if axis.size < 2 or not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
        raise Stage1Error(f"{name} must be finite and strictly increasing; shape={axis.shape}")
    return axis


def integrate_xyz(values: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Integrate in explicit x,y,z axis order without assuming uniform spacing."""
    q = np.asarray(values, dtype=float)
    expected = (x.size, y.size, z.size)
    if q.shape != expected:
        raise Stage1Error(f"Q shape {q.shape} != coordinate shape {expected}")
    trap = np.trapezoid
    return float(trap(trap(trap(q, z, axis=2), y, axis=1), x, axis=0))


def scalar(value: Any, label: str) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise Stage1Error(f"{label} is not scalar: shape={array.shape}")
    result = float(np.real(array[0]))
    if not np.isfinite(result):
        raise Stage1Error(f"{label} is not finite")
    return result


def configure_single_frequency(fdtd: Any, name: str) -> None:
    for prop, value in (
        ("override global monitor settings", True),
        ("use source limits", False),
        ("use wavelength spacing", True),
        ("wavelength center", config.WAVELENGTH_M),
        ("wavelength span", 0.0),
        ("frequency points", 1),
    ):
        try:
            fdtd.setnamed(name, prop, value)
        except Exception:
            pass


def add_power_monitor(fdtd: Any, name: str, z_m: float, x_span_m: float, y_span_m: float) -> None:
    monitor = fdtd.addpower()
    monitor["name"] = name
    monitor["monitor type"] = "2D Z-normal"
    monitor["x"] = 0.0
    monitor["x span"] = x_span_m
    monitor["y"] = 0.0
    monitor["y span"] = y_span_m
    monitor["z"] = z_m
    configure_single_frequency(fdtd, name)


def enable_periodic_correction(fdtd: Any, name: str) -> dict[str, Any]:
    """Enable the x/y periodic correction already shipped in pabs_adv."""
    script = str(fdtd.getnamed(name, "analysis script"))
    marker = "Periodic boundary condition correction"
    marker_position = script.find(marker)
    if marker_position < 0:
        raise Stage1Error("installed pabs_adv analysis has no periodic-correction section")
    tail = script[marker_position:]
    old = "if (0) {"
    position = tail.find(old)
    if position < 0:
        raise Stage1Error("cannot locate outer pabs_adv periodic-correction switch")
    absolute = marker_position + position
    modified = script[:absolute] + "if (1) {" + script[absolute + len(old) :]
    fdtd.setnamed(name, "analysis script", modified)
    return {
        "object_library_id": "pabs_adv",
        "installed_analysis_group": name,
        "yee_component_method": True,
        "periodic_correction_enabled": True,
        "periodic_axes": ["x", "y"],
        "formula_from_installed_group": "0.5*eps0*omega*sum_i(|E_i|^2*imag(eps_i))",
        "normalization_from_installed_group": "Pabs divided by sourcepower(f)",
    }


def fixed_binary_disk(model: Any) -> np.ndarray:
    x = np.asarray(model.design_x, float)
    y = np.asarray(model.design_y, float)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    disk = (xx**2 + yy**2) <= config.FIXED_DESIGN_DISK_RADIUS_M**2
    density = np.repeat(disk[:, :, None].astype(float), int(model.design_grids[2]), axis=2)
    # The disk is zero at all periodic edges; make the seam contract explicit.
    density[-1, :, :] = density[0, :, :]
    density[:, -1, :] = density[:, 0, :]
    values = np.unique(density)
    if not np.array_equal(values, np.array([0.0, 1.0])):
        raise Stage1Error(f"fixed geometry is not binary: values={values}")
    return density


def fixed_geometry(model: Any, kind: str) -> np.ndarray:
    if kind in {"centered-disk", "centered-disk-gap"}:
        return fixed_binary_disk(model)
    shape = (len(model.design_x), len(model.design_y), len(model.design_z))
    if kind == "none":
        return np.zeros(shape, dtype=float)
    if kind == "full-film":
        return np.ones(shape, dtype=float)
    if kind == "centered-square":
        x = np.asarray(model.design_x, float)
        y = np.asarray(model.design_y, float)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        square = (np.abs(xx) <= config.FIXED_DESIGN_DISK_RADIUS_M) & (np.abs(yy) <= config.FIXED_DESIGN_DISK_RADIUS_M)
        return np.repeat(square[:, :, None].astype(float), shape[2], axis=2)
    raise Stage1Error(f"unsupported fixed geometry {kind!r}")


def plot_slice(path: Path, a: np.ndarray, b: np.ndarray, image: np.ndarray, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    shown = ax.pcolormesh(a * 1e6, b * 1e6, np.asarray(image).T, shading="auto")
    fig.colorbar(shown, ax=ax, label=r"$Q_{\rm on}$ [W m$^{-3}$]")
    ax.set_xlabel(f"{xlabel} [um]")
    ax.set_ylabel(f"{ylabel} [um]")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    optical_dir = output / "fdtd_qon"
    optical_dir.mkdir(parents=True, exist_ok=True)
    installation = select_installation(args.lumerical_version)

    # Pin the existing builder to the requested installation before importing it.
    os.environ["VC_LUMERICAL_ROOT"] = str(installation.root)
    os.environ["LUMERICAL_ROOT"] = str(installation.root)
    selected_lumapi_dir = installation.lumapi_path.parent.resolve()
    os.environ["LUMERICAL_PYTHONPATH"] = str(selected_lumapi_dir)
    os.environ["TARGET_WL_UM"] = "4.0"
    if args.global_resolution is not None:
        os.environ["RES"] = str(args.global_resolution)
    if args.design_resolution_xy is not None:
        os.environ["DESIGN_RES_XY"] = str(args.design_resolution_xy)
    if args.flake_dz_nm is not None:
        os.environ["FILM_DZ_NM"] = str(args.flake_dz_nm)
    os.environ["ENABLE_LIVE_PLOTTING"] = "0"
    # A globally visible newer lumapi must not silently override the requested
    # installation. Remove only Lumerical API directories, then put the
    # selected one first. Fail closed below if lumapi was already imported
    # from a different installation.
    sys.path[:] = [
        entry
        for entry in sys.path
        if not (
            "/opt/lumerical/" in str(entry)
            and str(entry).rstrip("/").endswith("/api/python")
        )
    ]
    for path in (
        config.REPOSITORY_ROOT,
        config.REPOSITORY_ROOT / "bundle",
        selected_lumapi_dir,
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    preloaded_lumapi = sys.modules.get("lumapi")
    if preloaded_lumapi is not None:
        preloaded_path = Path(preloaded_lumapi.__file__).resolve()
        if preloaded_path != installation.lumapi_path:
            raise Stage1Error(
                "requested Lumerical API does not match preloaded lumapi: "
                f"requested={installation.lumapi_path}, loaded={preloaded_path}"
            )

    project_path = optical_dir / "fdtd_test.fsp"
    summary_path = optical_dir / "fdtd_absorption_summary.json"
    fdtd = None
    try:
        import eqc_lib as runtime
        import tairte4_volume_model as model
        import lumapi

        loaded_lumapi_path = Path(lumapi.__file__).resolve()
        if loaded_lumapi_path != installation.lumapi_path:
            raise Stage1Error(
                "requested Lumerical API was not loaded: "
                f"requested={installation.lumapi_path}, loaded={loaded_lumapi_path}"
            )

        design_gap_m = (
            args.diagnostic_air_gap_nm * 1e-9
            if args.fixed_geometry == "centered-disk-gap" else 0.0
        )

        if args.postprocess_project:
            completed_project = Path(args.postprocess_project).expanduser().resolve()
            if not completed_project.is_file():
                raise Stage1Error(f"completed postprocess project not found: {completed_project}")
            shutil.copy2(completed_project, project_path)
            fdtd = runtime.open_control(project_path)
            design_x = np.linspace(-0.5 * model.period_xy * 1e-6, 0.5 * model.period_xy * 1e-6, model.Nx)
            design_y = np.linspace(-0.5 * model.period_xy * 1e-6, 0.5 * model.period_xy * 1e-6, model.Ny)
            design_z = np.linspace(0.0, model.design_h * 1e-6, model.Nz)
            xx, yy = np.meshgrid(design_x, design_y, indexing="ij")
            disk = (xx**2 + yy**2) <= config.FIXED_DESIGN_DISK_RADIUS_M**2
            density = np.repeat(disk[:, :, None].astype(float), model.Nz, axis=2)
            pabs_metadata = {
                "object_library_id": "pabs_adv",
                "installed_analysis_group": PABS_NAME,
                "yee_component_method": True,
                "periodic_correction_enabled": True,
                "periodic_axes": ["x", "y"],
                "formula_from_installed_group": "0.5*eps0*omega*sum_i(|E_i|^2*imag(eps_i))",
                "normalization_from_installed_group": "Pabs divided by sourcepower(f)",
            }
            resource = f"postprocess-only copy of {completed_project}"
        else:
            sim = model.build_case("x")
            fdtd = sim.fdtd
            fdtd.switchtolayout()
            density = fixed_geometry(sim, args.fixed_geometry)
            design_x = np.asarray(sim.design_x, float)
            design_y = np.asarray(sim.design_y, float)
            design_z = np.asarray(sim.design_z, float)
            sim.update_design_density(density)
            if design_gap_m > 0:
                fdtd.setnamed("design", "z", float(model.design_c[2]) * 1e-6 + design_gap_m)
                design_z = design_z + design_gap_m
            runtime.pin_solver(fdtd)
            fdtd.setnamed("FDTD", "simulation time", config.FDTD_SIMULATION_TIME_S)

            pabs = fdtd.addobject("pabs_adv")
            pabs["name"] = PABS_NAME
            pabs["x"] = 0.0
            pabs["x span"] = float(model.Sx) * 1e-6
            pabs["y"] = 0.0
            pabs["y span"] = float(model.Sy) * 1e-6
            pabs["z"] = -0.5 * float(model.flake_h) * 1e-6
            pabs["z span"] = (float(model.flake_h) * 1e-6 + 2.0 * args.pabs_z_padding_nm * 1e-9)
            pabs_metadata = enable_periodic_correction(fdtd, PABS_NAME)

            reflection_z = (float(model.src_c[2]) + 0.15) * 1e-6
            transmission_z = -(float(model.flake_h) + float(model.sio2_h) + 0.40) * 1e-6
            local_top_z = 0.10e-6
            local_bottom_z = -(float(model.flake_h) * 1e-6 + 0.10e-6)
            add_power_monitor(fdtd, REFLECTION_MONITOR, reflection_z, float(model.Sx) * 1e-6, float(model.Sy) * 1e-6)
            add_power_monitor(fdtd, TRANSMISSION_MONITOR, transmission_z, float(model.Sx) * 1e-6, float(model.Sy) * 1e-6)
            add_power_monitor(fdtd, LOCAL_TOP_MONITOR, local_top_z, float(model.Sx) * 1e-6, float(model.Sy) * 1e-6)
            add_power_monitor(fdtd, LOCAL_BOTTOM_MONITOR, local_bottom_z, float(model.Sx) * 1e-6, float(model.Sy) * 1e-6)
            fdtd.setglobalsource("wavelength start", config.WAVELENGTH_M)
            fdtd.setglobalsource("wavelength stop", config.WAVELENGTH_M)
            fdtd.save(str(project_path))
            runtime.configure_session_resources(fdtd)
            resource = runtime.run_session(fdtd, "photothermal_stage1_qon")

        fdtd.runanalysis(PABS_NAME)

        pabs_dataset = fdtd.getresult(PABS_NAME, "Pabs")
        pabs_total_dataset = fdtd.getresult(PABS_NAME, "Pabs_total")
        x = require_increasing("x", pabs_dataset["x"])
        y = require_increasing("y", pabs_dataset["y"])
        z = require_increasing("z", pabs_dataset["z"])
        frequency = scalar(pabs_dataset["f"], "Pabs.f")
        wavelength = scalar(pabs_dataset["lambda"], "Pabs.lambda")
        q_normalized = np.asarray(pabs_dataset["Pabs"], float).squeeze()
        if q_normalized.shape != (x.size, y.size, z.size):
            raise Stage1Error(
                f"Pabs array ordering mismatch: {q_normalized.shape} != "
                f"(x,y,z)={(x.size, y.size, z.size)}"
            )

        source_power_fdtd = scalar(fdtd.sourcepower(frequency), "sourcepower")
        source_intensity_fdtd = scalar(fdtd.sourceintensity(frequency), "sourceintensity")
        if source_power_fdtd <= 0 or source_intensity_fdtd <= 0:
            raise Stage1Error(
                f"invalid source normalization: power={source_power_fdtd}, "
                f"intensity={source_intensity_fdtd}"
            )
        physical_scale = config.INCIDENT_INTENSITY_W_M2 / source_intensity_fdtd
        incident_power_physical = source_power_fdtd * physical_scale
        unit_cell_area = float(model.Sx * model.Sy) * 1e-12
        incident_power_area_check = config.INCIDENT_INTENSITY_W_M2 * unit_cell_area
        incident_power_crosscheck_error = abs(
            incident_power_physical - incident_power_area_check
        ) / incident_power_area_check

        q_raw = q_normalized * source_power_fdtd
        q_physical = q_raw * physical_scale
        p_abs_volume_raw = integrate_xyz(q_raw, x, y, z)
        p_abs_volume = integrate_xyz(q_physical, x, y, z)
        pabs_total_normalized = scalar(
            pabs_total_dataset["Pabs_total"], "pabs_adv.Pabs_total"
        )

        signed_reflection = scalar(fdtd.transmission(REFLECTION_MONITOR), "reflection transmission()")
        signed_transmission = scalar(
            fdtd.transmission(TRANSMISSION_MONITOR), "transmission transmission()"
        )
        signed_local_top = scalar(
            fdtd.transmission(LOCAL_TOP_MONITOR), "local-top transmission()"
        )
        signed_local_bottom = scalar(
            fdtd.transmission(LOCAL_BOTTOM_MONITOR), "local-bottom transmission()"
        )
        reflectance = signed_reflection
        transmittance = -signed_transmission
        absorptance_flux = 1.0 - reflectance - transmittance
        absorptance_local_flux = signed_local_bottom - signed_local_top
        p_abs_flux = absorptance_flux * incident_power_physical
        p_abs_local_flux = absorptance_local_flux * incident_power_physical
        if p_abs_flux <= 0 and not args.allow_nonpositive_diagnostic_flux:
            raise Stage1Error(
                f"non-positive flux absorption: R={reflectance}, T={transmittance}, "
                f"A={absorptance_flux}"
            )
        if p_abs_local_flux <= 0 and not args.allow_nonpositive_diagnostic_flux:
            raise Stage1Error(
                "non-positive local-box flux absorption: "
                f"top={signed_local_top}, bottom={signed_local_bottom}, "
                f"A_local={absorptance_local_flux}"
            )
        tiny = np.finfo(float).tiny
        energy_error = abs(p_abs_volume - p_abs_flux) / max(abs(p_abs_flux), tiny)
        local_energy_error = abs(p_abs_volume - p_abs_local_flux) / max(abs(p_abs_local_flux), tiny)
        global_vs_local_flux_error = abs(p_abs_flux - p_abs_local_flux) / max(abs(p_abs_local_flux), tiny)
        a_q_python = p_abs_volume_raw / source_power_fdtd
        diagnostic_metadata = mesh_and_material_metadata(
            fdtd, model, geometry=args.fixed_geometry, design_gap_m=design_gap_m
        )
        epsilon_slice_metadata = save_epsilon_and_material_slices(
            fdtd, optical_dir, x=x, y=y, z=z, geometry=args.fixed_geometry,
            disk_radius_m=config.FIXED_DESIGN_DISK_RADIUS_M,
            square_half_width_m=config.FIXED_DESIGN_DISK_RADIUS_M,
            design_gap_m=design_gap_m, design_height_m=float(model.design_h) * 1e-6,
        )

        negative = q_physical < 0
        negative_fraction = float(np.count_nonzero(negative) / q_physical.size)
        negative_power = integrate_xyz(np.minimum(q_physical, 0.0), x, y, z)
        minimum_index = np.unravel_index(int(np.argmin(q_physical)), q_physical.shape)
        minimum_location = [float(x[minimum_index[0]]), float(y[minimum_index[1]]), float(z[minimum_index[2]])]

        common = {
            "x_m": x,
            "y_m": y,
            "z_m": z,
            "wavelength_m": wavelength,
            "frequency_Hz": frequency,
            "polarization": config.POLARIZATION,
            "array_axis_order": np.array(["x", "y", "z"]),
            "array_memory_order": "C in NPZ/MAT; axes explicitly named",
        }
        np.savez_compressed(
            optical_dir / "q_on_raw.npz",
            **common,
            Q_on_W_m3=q_raw,
            source_power_fdtd_W=source_power_fdtd,
            source_intensity_fdtd_W_m2=source_intensity_fdtd,
        )
        np.savez_compressed(
            optical_dir / "q_on_physical.npz",
            **common,
            Q_on_W_m3=q_physical,
            incident_intensity_W_m2=config.INCIDENT_INTENSITY_W_M2,
            incident_power_W=incident_power_physical,
            P_abs_volume_W=p_abs_volume,
            P_abs_flux_W=p_abs_flux,
        )
        savemat(
            optical_dir / "q_on_for_lumerical.mat",
            {
                "x": x.reshape(-1, 1),
                "y": y.reshape(-1, 1),
                "z": z.reshape(-1, 1),
                "Q": np.asfortranarray(q_physical),
                "frequency": np.array([[frequency]]),
                "wavelength": np.array([[wavelength]]),
            },
            do_compression=True,
        )
        np.savez_compressed(
            optical_dir / "fixed_binary_design.npz",
            x_m=design_x,
            y_m=design_y,
            z_m=design_z,
            density=density,
        )
        fdtd.save(str(project_path))

        plot_slice(
            optical_dir / "q_xy_midplane.png", x, y,
            q_physical[:, :, int(np.argmin(np.abs(z + 0.5 * model.flake_h * 1e-6)))],
            "x", "y",
        )
        plot_slice(
            optical_dir / "q_xz_center.png", x, z,
            q_physical[:, int(np.argmin(np.abs(y))), :], "x", "z",
        )
        plot_slice(
            optical_dir / "q_yz_center.png", y, z,
            q_physical[int(np.argmin(np.abs(x))), :, :], "y", "z",
        )

        geometry_kind = {
            "centered-disk": config.FIXED_DESIGN_KIND,
            "none": "none_all_air",
            "full-film": "full_cell_uniform_n4_film",
            "centered-square": "grid_aligned_square_n4_block",
            "centered-disk-gap": "centered_binary_disk_with_air_gap",
        }[args.fixed_geometry]
        summary = {
            "status": "passed" if energy_error < config.OPTICAL_ENERGY_ERROR_LIMIT else "failed_acceptance_criteria",
            "validated": bool(energy_error < config.OPTICAL_ENERGY_ERROR_LIMIT),
            "selected_version": installation.version_key,
            "solver_version": str(fdtd.version()),
            "lumapi_path": str(loaded_lumapi_path),
            "resource": resource,
            "fixed_geometry": {
                "kind": geometry_kind,
                "disk_radius_m": (
                    config.FIXED_DESIGN_DISK_RADIUS_M
                    if args.fixed_geometry in {"centered-disk", "centered-disk-gap"}
                    else None
                ),
                "square_half_width_m": (
                    config.FIXED_DESIGN_DISK_RADIUS_M
                    if args.fixed_geometry == "centered-square" else None
                ),
                "diagnostic_air_gap_m": design_gap_m,
                "binary_values": np.unique(density).tolist(),
                "density_shape": list(density.shape),
                "mapping_called": False,
                "optimizer_called": False,
            },
            "actual_stack": {
                "period_x_m": float(model.Sx) * 1e-6,
                "period_y_m": float(model.Sy) * 1e-6,
                "Si_substrate_thickness_m": float(model.si_h) * 1e-6,
                "SiO2_thickness_m": float(model.sio2_h) * 1e-6,
                "TaIrTe4_thickness_m": float(model.flake_h) * 1e-6,
                "design_thickness_m": float(model.design_h) * 1e-6,
                "air_top_thickness_m": float(model.air_top) * 1e-6,
                "difference_from_requested_description": "repository Si is 2.0 um by default, not approximately 3.0 um",
            },
            "boundaries": {"x": "Periodic", "y": "Periodic", "z": "PML"},
            "simulation_time_s": config.FDTD_SIMULATION_TIME_S,
            "source": {
                "wavelength_m": wavelength,
                "frequency_Hz": frequency,
                "polarization": "x",
                "direction": "backward (-z)",
                "source_power_fdtd_W": source_power_fdtd,
                "source_intensity_fdtd_W_m2": source_intensity_fdtd,
                "target_incident_intensity_W_m2": config.INCIDENT_INTENSITY_W_M2,
                "physical_scale": physical_scale,
                "physical_incident_power_W": incident_power_physical,
                "intensity_times_unit_cell_area_W": incident_power_area_check,
                "incident_power_crosscheck_relative_error": incident_power_crosscheck_error,
            },
            "pabs_method": pabs_metadata,
            "diagnostics": {
                **diagnostic_metadata,
                "solver_epsilon_material_slices": epsilon_slice_metadata,
            },
            "grid": {
                "shape_xyz": list(q_physical.shape),
                "axis_order": ["x", "y", "z"],
                "x_range_m": [float(x[0]), float(x[-1])],
                "y_range_m": [float(y[0]), float(y[-1])],
                "z_range_m": [float(z[0]), float(z[-1])],
                "pabs_z_padding_requested_m": args.pabs_z_padding_nm * 1e-9,
                "strictly_increasing": True,
                "quadrature": "nested trapezoidal rule on actual coordinates",
                "global_resolution_cells_per_um": int(model.GLOBAL_RESOLUTION),
                "flake_dz_m": float(model.FLAKE_DZ_NM) * 1e-9,
            },
            "absorption": {
                "P_abs_volume_native_W": p_abs_volume_raw,
                "P_abs_volume_physical_W": p_abs_volume,
                "Pabs_total_normalized_from_lumerical": pabs_total_normalized,
                "A_Q_python_exported_integral": a_q_python,
                "internal_vs_python_relative_error": abs(a_q_python - pabs_total_normalized) / max(abs(pabs_total_normalized), np.finfo(float).tiny),
                "R": reflectance,
                "T": transmittance,
                "A_flux": absorptance_flux,
                "P_abs_flux_physical_W": p_abs_flux,
                "local_flux_top_signed": signed_local_top,
                "local_flux_bottom_signed": signed_local_bottom,
                "A_local_flux": absorptance_local_flux,
                "P_abs_local_flux_physical_W": p_abs_local_flux,
                "energy_error": energy_error,
                "local_energy_error": local_energy_error,
                "global_vs_local_flux_relative_error": global_vs_local_flux_error,
                "acceptance_limit": config.OPTICAL_ENERGY_ERROR_LIMIT,
            },
            "negative_Q": {
                "minimum_Q_W_m3": float(np.min(q_physical)),
                "negative_voxel_fraction": negative_fraction,
                "integrated_negative_power_W": negative_power,
                "minimum_location_xyz_m": minimum_location,
                "clipped": False,
            },
            "artifacts": {
                "project": str(project_path),
                "raw_npz": str(optical_dir / "q_on_raw.npz"),
                "physical_npz": str(optical_dir / "q_on_physical.npz"),
                "mat": str(optical_dir / "q_on_for_lumerical.mat"),
            },
        }
        write_json(summary_path, summary)
        print(json.dumps(jsonable(summary), indent=2), flush=True)
        return 0 if summary["validated"] else 2
    except Exception as exc:
        failure = {
            "status": "failed",
            "validated": False,
            "selected_version": installation.version_key,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(summary_path, failure)
        print(json.dumps(failure, indent=2), flush=True)
        return 1
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
