"""Layout and readback helpers for exact-Au 4-um Lumerical controls.

Nothing in this module opens or runs a Lumerical session.  The executable
runner imports these helpers only after the B200 preflight has passed.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_exact_au import (
    AU_MATERIAL,
    SIO2_MATERIAL,
    SOURCE_WAVELENGTH_BAND_M,
    TAIRTE4_MATERIAL,
    add_dispersive_materials,
    add_exact_stack_geometry,
    design_edges,
    exact_control_masks,
    sampled_material_data,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_mesh_contract import (
    GEOMETRY_CONTROLS,
    LumericalMeshSpec,
    POLARIZATIONS,
)
from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import (
    PABS_FIELD,
    PABS_GROUP,
    PABS_INDEX,
)


C0_M_S = 299_792_458.0
SOURCE_NAME = "au_dualpol_4um_scalar_Gaussian"
TARGET_MONITOR = "au_dualpol_4um_source_target_plane"
SOURCE_Z_M = 0.75e-6
WAIST_Z_M = 0.0
SOURCE_PROFILE_Z_M = WAIST_Z_M
Q_BOX_TOP_M = 0.50e-6
MONITOR_WAVELENGTH_M = CONTRACT.wavelength_m
MATERIAL_READBACK_COUNT = 81
MATERIAL_FIT_RELATIVE_GATE = 5.0e-3


def polarization_angle_deg(polarization: str) -> float:
    if polarization not in POLARIZATIONS:
        raise ValueError(f"unsupported polarization: {polarization}")
    # Lumerical angle 0 is Ex.  The repository axis contract is x=b, y=a.
    return {"Eb": 0.0, "Ea": 90.0}[polarization]


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_calibration_contract(
    spec: LumericalMeshSpec,
    polarization: str,
    *,
    source_object_w0_m: float,
) -> dict[str, Any]:
    """Return the exact settings that must match source and material runs."""

    spec.validate()
    angle = polarization_angle_deg(polarization)
    if not np.isfinite(source_object_w0_m) or source_object_w0_m <= 0.0:
        raise ValueError("source-object waist must be finite and positive")
    contract = {
        "schema": "au-dualpol-4um-source-calibration-v1",
        "mesh_spec": asdict(spec),
        "polarization": polarization,
        "polarization_angle_deg": angle,
        "axis_mapping": {"x": "b", "y": "a"},
        "source": {
            "type": "scalar Gaussian",
            "injection_axis": "z",
            "direction": "backward",
            "source_z_m": SOURCE_Z_M,
            "waist_plane_z_m": WAIST_Z_M,
            "source_object_w0_m": float(source_object_w0_m),
            "target_realized_w0_m": CONTRACT.gaussian_waist_m,
            "aperture_x_bounds_m": [
                -0.5 * CONTRACT.source_aperture_span_m,
                0.5 * CONTRACT.source_aperture_span_m,
            ],
            "aperture_y_bounds_m": [
                -0.5 * CONTRACT.source_aperture_span_m,
                0.5 * CONTRACT.source_aperture_span_m,
            ],
            "wavelength_band_m": list(SOURCE_WAVELENGTH_BAND_M),
            "analysis_wavelength_m": MONITOR_WAVELENGTH_M,
        },
    }
    return {
        **contract,
        "source_calibration_sha256": _canonical_json_sha256(contract),
    }


def validate_source_calibration_record(
    record: dict[str, Any], expected_contract: dict[str, Any]
) -> dict[str, Any]:
    """Validate a source-only result before a material run consumes it."""

    expected_sha = expected_contract["source_calibration_sha256"]
    measured = record.get("target_plane_metrics", {})
    incident_power = measured.get("downward_Poynting_power_W")
    gates = {
        "source_only_status_passed": str(record.get("status", "")).startswith(
            "PASSED_EXACT_AU_4UM_SOURCE_ONLY"
        ),
        "source_calibration_sha_matches": (
            record.get("source_calibration_sha256") == expected_sha
        ),
        "polarization_matches": (
            record.get("polarization") == expected_contract["polarization"]
        ),
        "source_only_all_gates_passed": bool(record.get("all_gates_passed")),
        "positive_finite_incident_power": bool(
            incident_power is not None
            and np.isfinite(float(incident_power))
            and float(incident_power) > 0.0
        ),
        "source_only_was_not_rescaled": record.get("Q_processing")
        == {
            "clipping": False,
            "smoothing": False,
            "gain": False,
            "field_or_Q_rescaling": False,
        },
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "expected_source_calibration_sha256": expected_sha,
        "recorded_source_calibration_sha256": record.get(
            "source_calibration_sha256"
        ),
        "incident_power_W": (
            float(incident_power) if incident_power is not None else None
        ),
    }


def _configure_single_frequency(item: Any) -> None:
    item["override global monitor settings"] = True
    item["use source limits"] = False
    item["use wavelength spacing"] = True
    item["wavelength center"] = MONITOR_WAVELENGTH_M
    item["wavelength span"] = 0.0
    item["frequency points"] = 1


def _add_solver(fdtd: Any, spec: LumericalMeshSpec) -> None:
    solver = fdtd.addfdtd()
    solver["name"] = "FDTD"
    solver["dimension"] = "3D"
    solver["x"] = 0.0
    solver["x span"] = spec.lateral_span_m
    solver["y"] = 0.0
    solver["y span"] = spec.lateral_span_m
    solver["z min"] = spec.z_min_m
    solver["z max"] = spec.z_max_m
    for axis in "xyz":
        solver[f"{axis} min bc"] = "PML"
        solver[f"{axis} max bc"] = "PML"
    solver["pml layers"] = spec.pml_layers
    solver["mesh type"] = "auto non-uniform"
    solver["mesh refinement"] = spec.conformal_mesh
    solver["mesh accuracy"] = spec.mesh_accuracy
    solver["simulation time"] = spec.simulation_time_s
    solver["auto shutoff min"] = spec.auto_shutoff_min
    solver["override simulation bandwidth for mesh generation"] = True
    solver["mesh wavelength min"] = SOURCE_WAVELENGTH_BAND_M[0]
    solver["mesh wavelength max"] = SOURCE_WAVELENGTH_BAND_M[1]


def _add_source(
    fdtd: Any, polarization: str, *, source_object_w0_m: float
) -> None:
    source = fdtd.addgaussian()
    source["name"] = SOURCE_NAME
    source["injection axis"] = "z"
    source["direction"] = "backward"
    source["polarization angle"] = polarization_angle_deg(polarization)
    source["source shape"] = "Gaussian"
    source["use scalar approximation"] = True
    source["beam parameters"] = "Waist size and position"
    source["waist radius w0"] = source_object_w0_m
    source["distance from waist"] = -(SOURCE_Z_M - WAIST_Z_M)
    half = 0.5 * CONTRACT.source_aperture_span_m
    source["x min"], source["x max"] = -half, half
    source["y min"], source["y max"] = -half, half
    source["z"] = SOURCE_Z_M
    source["override global source settings"] = True
    source["wavelength start"] = SOURCE_WAVELENGTH_BAND_M[0]
    source["wavelength stop"] = SOURCE_WAVELENGTH_BAND_M[1]


def _add_mesh_overrides(fdtd: Any, spec: LumericalMeshSpec) -> None:
    half_domain = 0.5 * spec.lateral_span_m
    outer = fdtd.addmesh()
    outer["name"] = "au_dualpol_4um_full_domain_xyz"
    outer["x min"], outer["x max"] = -half_domain, half_domain
    outer["y min"], outer["y max"] = -half_domain, half_domain
    outer["z min"], outer["z max"] = spec.z_min_m, spec.z_max_m
    outer["override x mesh"] = True
    outer["override y mesh"] = True
    outer["override z mesh"] = True
    outer["dx"] = spec.outer_dxy_m
    outer["dy"] = spec.outer_dxy_m
    outer["dz"] = spec.bulk_dz_m

    flake = fdtd.addmesh()
    flake["name"] = "au_dualpol_4um_flake_stack_xyz"
    flake["x min"] = -0.5 * CONTRACT.flake_span_x_m
    flake["x max"] = 0.5 * CONTRACT.flake_span_x_m
    flake["y min"] = -0.5 * CONTRACT.flake_span_y_m
    flake["y max"] = 0.5 * CONTRACT.flake_span_y_m
    # Include an explicit guard cell above Au and below SiO2 so the requested
    # dz is realized across every thin-material interface.
    flake["z min"] = -385.0e-9 - spec.stack_dz_m
    flake["z max"] = CONTRACT.design_thickness_m + spec.stack_dz_m
    flake["override x mesh"] = True
    flake["override y mesh"] = True
    flake["override z mesh"] = True
    flake["dx"] = spec.flake_dxy_m
    flake["dy"] = spec.flake_dxy_m
    flake["dz"] = spec.stack_dz_m


def _add_target_monitor(fdtd: Any) -> None:
    monitor = fdtd.addpower()
    monitor["name"] = TARGET_MONITOR
    monitor["monitor type"] = "2D Z-normal"
    half = 0.5 * CONTRACT.source_aperture_span_m
    monitor["x min"], monitor["x max"] = -half, half
    monitor["y min"], monitor["y max"] = -half, half
    monitor["z"] = SOURCE_PROFILE_Z_M
    _configure_single_frequency(monitor)
    try:
        monitor["spatial interpolation"] = "specified position"
    except Exception:
        pass


def control_volume_bounds(
    spec: LumericalMeshSpec,
) -> dict[str, tuple[float, float]]:
    """Return a closed Q/flux box below the source and inside all six PMLs."""

    spec.validate()
    lateral_clearance = max(0.50e-6, 2.0 * spec.outer_dxy_m)
    half = 0.5 * spec.lateral_span_m - lateral_clearance
    lower = spec.z_min_m + max(0.50e-6, 2.0 * spec.bulk_dz_m)
    if half <= 0.5 * CONTRACT.flake_span_x_m:
        raise ValueError("control volume does not contain the complete flake")
    if not lower < -385.0e-9 < CONTRACT.design_thickness_m < Q_BOX_TOP_M:
        raise ValueError("control volume does not contain the complete stack")
    if not Q_BOX_TOP_M < SOURCE_Z_M < spec.z_max_m:
        raise ValueError("source must be above the Q box and inside the z domain")
    return {"x": (-half, half), "y": (-half, half), "z": (lower, Q_BOX_TOP_M)}


def _add_flux_box(
    fdtd: Any, bounds: dict[str, tuple[float, float]]
) -> dict[str, dict[str, Any]]:
    faces: dict[str, dict[str, Any]] = {}
    for axis in "xyz":
        for side, position in zip(("min", "max"), bounds[axis], strict=True):
            name = f"au_dualpol_4um_flux_{axis}_{side}"
            monitor = fdtd.addpower()
            monitor["name"] = name
            monitor["monitor type"] = f"2D {axis.upper()}-normal"
            monitor[axis] = position
            for transverse in "xyz":
                if transverse != axis:
                    monitor[f"{transverse} min"] = bounds[transverse][0]
                    monitor[f"{transverse} max"] = bounds[transverse][1]
            _configure_single_frequency(monitor)
            faces[f"{axis}_{side}"] = {
                "name": name,
                "axis": axis,
                "side": side,
                "outward_sign": -1.0 if side == "min" else 1.0,
            }
    return faces


def _add_q_analysis(fdtd: Any, bounds: dict[str, tuple[float, float]]) -> None:
    pabs = fdtd.addobject("pabs_adv")
    pabs["name"] = PABS_GROUP
    for axis in "xyz":
        pabs[axis] = 0.5 * sum(bounds[axis])
        pabs[f"{axis} span"] = bounds[axis][1] - bounds[axis][0]
    # The analysis group owns these two monitors.  Set them to the one
    # requested frequency if the installed object exposes the properties.
    for name in (PABS_FIELD, PABS_INDEX):
        for property_name, value in (
            ("override global monitor settings", True),
            ("use source limits", False),
            ("use wavelength spacing", True),
            ("wavelength center", MONITOR_WAVELENGTH_M),
            ("wavelength span", 0.0),
            ("frequency points", 1),
        ):
            try:
                fdtd.setnamed(name, property_name, value)
            except Exception:
                pass


def build_layout(
    fdtd: Any,
    *,
    case: str,
    polarization: str,
    spec: LumericalMeshSpec,
    source_object_w0_m: float,
) -> dict[str, Any]:
    """Build one source-only or exact-stack forward-control layout."""

    spec.validate()
    if case not in ("source_only", *GEOMETRY_CONTROLS):
        raise ValueError(f"unsupported exact-Au control case: {case}")
    calibration = source_calibration_contract(
        spec, polarization, source_object_w0_m=source_object_w0_m
    )
    _add_solver(fdtd, spec)
    _add_source(fdtd, polarization, source_object_w0_m=source_object_w0_m)
    _add_mesh_overrides(fdtd, spec)
    if case == "source_only":
        _add_target_monitor(fdtd)
        return {
            "case": case,
            "classification": "all-air source calibration; no material or Q",
            "source_calibration_contract": calibration,
            "material_input_audit": None,
            "geometry": None,
            "control_volume_bounds_m": None,
            "flux_faces": {},
        }

    material_audit = add_dispersive_materials(fdtd)
    masks = exact_control_masks()
    half_domain = 0.5 * spec.lateral_span_m
    geometry = add_exact_stack_geometry(
        fdtd,
        masks[case],
        optical_x_bounds_m=(-half_domain, half_domain),
        optical_y_bounds_m=(-half_domain, half_domain),
        optical_z_min_m=spec.z_min_m,
    )
    bounds = control_volume_bounds(spec)
    _add_q_analysis(fdtd, bounds)
    faces = _add_flux_box(fdtd, bounds)
    return {
        "case": case,
        "classification": (
            "provisional exact-Au Maxwell/Q control; no thermal, electrical, "
            "PTE, adjoint, or optimization solve"
        ),
        "source_calibration_contract": calibration,
        "material_input_audit": material_audit,
        "geometry": geometry,
        "control_volume_bounds_m": {
            axis: list(values) for axis, values in bounds.items()
        },
        "flux_faces": faces,
    }


def _relative_error(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.abs(actual - target) / np.maximum(
        np.abs(target), np.finfo(float).tiny
    )


def material_fit_readback(fdtd: Any, *, dt_s: float) -> dict[str, Any]:
    """Compare fitted and finite-dt epsilon to sampled targets over the band."""

    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("finite positive FDTD dt is required")
    sampled = sampled_material_data()
    wavelengths = np.linspace(
        SOURCE_WAVELENGTH_BAND_M[0],
        SOURCE_WAVELENGTH_BAND_M[1],
        MATERIAL_READBACK_COUNT,
    )
    frequencies = np.sort(C0_M_S / wavelengths)
    fmin, fmax = float(frequencies[0]), float(frequencies[-1])
    definitions = {
        "Au": (
            AU_MATERIAL,
            ("epsilon_au", "epsilon_au", "epsilon_au"),
        ),
        "SiO2": (
            SIO2_MATERIAL,
            ("epsilon_sio2", "epsilon_sio2", "epsilon_sio2"),
        ),
        "TaIrTe4": (
            TAIRTE4_MATERIAL,
            ("epsilon_ta_x_b", "epsilon_ta_y_a", "epsilon_ta_z_c"),
        ),
    }
    materials: dict[str, Any] = {}
    all_fit_errors: list[float] = []
    all_numerical_errors: list[float] = []
    for label, (material_name, target_keys) in definitions.items():
        axes: dict[str, Any] = {}
        for component, (axis, target_key) in enumerate(
            zip("xyz", target_keys, strict=True), start=1
        ):
            target = np.interp(
                frequencies,
                sampled["frequency_hz"],
                sampled[target_key].real,
            ) + 1j * np.interp(
                frequencies,
                sampled["frequency_hz"],
                sampled[target_key].imag,
            )
            fitted_n = np.asarray(
                fdtd.getfdtdindex(
                    material_name, frequencies, fmin, fmax, component
                )
            ).reshape(-1)
            numerical = np.asarray(
                fdtd.getnumericalpermittivity(
                    material_name, frequencies, fmin, fmax, dt_s, component
                )
            ).reshape(-1)
            if fitted_n.size != frequencies.size or numerical.size != frequencies.size:
                raise RuntimeError(
                    f"{label}.{axis} material readback size mismatch"
                )
            fitted = fitted_n.astype(complex) ** 2
            numerical = numerical.astype(complex)
            fit_error = _relative_error(fitted, target)
            numerical_error = _relative_error(numerical, target)
            if not all(
                np.all(np.isfinite(value))
                for value in (fitted, numerical, fit_error, numerical_error)
            ):
                raise RuntimeError(f"{label}.{axis} material readback is non-finite")
            center = int(np.argmin(np.abs(frequencies - C0_M_S / MONITOR_WAVELENGTH_M)))
            axes[axis] = {
                "target_epsilon_at_4um": _complex_json(target[center]),
                "fitted_epsilon_at_4um": _complex_json(fitted[center]),
                "finite_dt_epsilon_at_4um": _complex_json(numerical[center]),
                "max_fitted_relative_error": float(np.max(fit_error)),
                "max_finite_dt_relative_error": float(np.max(numerical_error)),
            }
            all_fit_errors.append(float(np.max(fit_error)))
            all_numerical_errors.append(float(np.max(numerical_error)))
        materials[label] = {"Lumerical_name": material_name, "axes": axes}
    max_fit = max(all_fit_errors)
    max_numerical = max(all_numerical_errors)
    gates = {
        "all_sampled_fits_within_0p5pct": max_fit < MATERIAL_FIT_RELATIVE_GATE,
        "all_finite_dt_models_within_0p5pct": (
            max_numerical < MATERIAL_FIT_RELATIVE_GATE
        ),
    }
    return {
        "status": (
            "PASSED_LUMERICAL_4UM_MATERIAL_FIT_READBACK"
            if all(gates.values())
            else "FAILED_LUMERICAL_4UM_MATERIAL_FIT_READBACK"
        ),
        "frequency_count": int(frequencies.size),
        "frequency_range_Hz": [fmin, fmax],
        "wavelength_range_m": list(SOURCE_WAVELENGTH_BAND_M),
        "dt_s": float(dt_s),
        "max_fitted_relative_error": max_fit,
        "max_finite_dt_relative_error": max_numerical,
        "materials": materials,
        "gates": gates,
    }


def _complex_json(value: complex) -> dict[str, float]:
    item = complex(value)
    return {"real": float(item.real), "imag": float(item.imag)}


def interval_max_step(
    coordinate: np.ndarray, lower: float, upper: float
) -> float:
    values = np.asarray(coordinate, float).reshape(-1)
    if values.size < 2 or np.any(np.diff(values) <= 0.0):
        raise ValueError("mesh coordinate must be strictly increasing")
    centers = 0.5 * (values[:-1] + values[1:])
    selected = np.diff(values)[(centers >= lower) & (centers <= upper)]
    if selected.size == 0:
        raise RuntimeError(f"no mesh intervals in [{lower}, {upper}]")
    return float(np.max(selected))


def requested_mesh_readback_gates(
    coordinates: dict[str, np.ndarray], spec: LumericalMeshSpec
) -> dict[str, Any]:
    """Fail if Lumerical did not realize the requested local overrides."""

    flake_x = (-0.5 * CONTRACT.flake_span_x_m, 0.5 * CONTRACT.flake_span_x_m)
    flake_y = (-0.5 * CONTRACT.flake_span_y_m, 0.5 * CONTRACT.flake_span_y_m)
    stack_z = (-385.0e-9, CONTRACT.design_thickness_m)
    lower_bulk_z = (spec.z_min_m, -385.0e-9)
    upper_air_z = (CONTRACT.design_thickness_m, spec.z_max_m)
    actual = {
        "flake_max_dx_m": interval_max_step(coordinates["x"], *flake_x),
        "flake_max_dy_m": interval_max_step(coordinates["y"], *flake_y),
        "stack_max_dz_m": interval_max_step(coordinates["z"], *stack_z),
        "Si_bulk_max_dz_m": interval_max_step(
            coordinates["z"], *lower_bulk_z
        ),
        "upper_air_and_PML_max_dz_m": interval_max_step(
            coordinates["z"], *upper_air_z
        ),
    }
    tolerance = 1.0e-12
    gates = {
        "flake_dx_le_requested": actual["flake_max_dx_m"]
        <= spec.flake_dxy_m + tolerance,
        "flake_dy_le_requested": actual["flake_max_dy_m"]
        <= spec.flake_dxy_m + tolerance,
        "stack_dz_le_requested": actual["stack_max_dz_m"]
        <= spec.stack_dz_m + tolerance,
        "Si_bulk_dz_le_requested": actual["Si_bulk_max_dz_m"]
        <= spec.bulk_dz_m + tolerance,
        "upper_air_and_PML_dz_le_requested": actual[
            "upper_air_and_PML_max_dz_m"
        ]
        <= spec.bulk_dz_m + tolerance,
    }
    return {"actual": actual, "gates": gates, "all": all(gates.values())}


def coordinate_material_partition(
    native_coordinates: dict[str, np.ndarray], au_mask: np.ndarray
) -> dict[str, np.ndarray]:
    """Partition native Yee samples by the canonical provisional geometry.

    Interface samples are assigned deterministically by coordinate and layer
    priority.  This is a convergence diagnostic, not a claim that a
    conformal cut cell contains only the assigned material.  The raw epsilon
    arrays remain the authoritative numerical-interface readback.
    """

    x = np.asarray(native_coordinates["x"], float).reshape(-1)
    y = np.asarray(native_coordinates["y"], float).reshape(-1)
    z = np.asarray(native_coordinates["z"], float).reshape(-1)
    mask = np.asarray(au_mask)
    if mask.shape != CONTRACT.design_shape or not np.all((mask == 0) | (mask == 1)):
        raise ValueError("Au material partition requires the canonical binary mask")
    x_edges, y_edges = design_edges()
    ix = np.searchsorted(x_edges, x, side="right") - 1
    iy = np.searchsorted(y_edges, y, side="right") - 1
    valid_x = (ix >= 0) & (ix < mask.shape[0])
    valid_y = (iy >= 0) & (iy < mask.shape[1])
    clipped_ix = np.clip(ix, 0, mask.shape[0] - 1)
    clipped_iy = np.clip(iy, 0, mask.shape[1] - 1)
    occupied_xy = (
        mask[clipped_ix[:, None], clipped_iy[None, :]].astype(bool)
        & valid_x[:, None]
        & valid_y[None, :]
    )
    flake_xy = (
        (x[:, None] >= -0.5 * CONTRACT.flake_span_x_m)
        & (x[:, None] <= 0.5 * CONTRACT.flake_span_x_m)
        & (y[None, :] >= -0.5 * CONTRACT.flake_span_y_m)
        & (y[None, :] <= 0.5 * CONTRACT.flake_span_y_m)
    )
    au = occupied_xy[:, :, None] & (
        (z[None, None, :] >= 0.0)
        & (z[None, None, :] <= CONTRACT.design_thickness_m)
    )
    ta = flake_xy[:, :, None] & (
        (z[None, None, :] >= -CONTRACT.flake_thickness_m)
        & (z[None, None, :] < 0.0)
    )
    sio2_z = (
        (z >= -(CONTRACT.flake_thickness_m + CONTRACT.sio2_thickness_m))
        & (z <= -CONTRACT.flake_thickness_m)
    )
    sio2 = np.broadcast_to(sio2_z[None, None, :], au.shape) & ~ta
    si_z = z < -(CONTRACT.flake_thickness_m + CONTRACT.sio2_thickness_m)
    silicon = np.broadcast_to(si_z[None, None, :], au.shape)
    occupied = au | ta | sio2 | silicon
    air = ~occupied
    result = {
        "Au_coordinate_partition": au,
        "TaIrTe4_coordinate_partition": ta,
        "SiO2_coordinate_partition": sio2,
        "Si_coordinate_partition": silicon,
        "air_coordinate_partition": air,
    }
    coverage = sum(values.astype(np.uint8) for values in result.values())
    if not np.all(coverage == 1):
        raise RuntimeError("coordinate material partition is not disjoint/exhaustive")
    return result
