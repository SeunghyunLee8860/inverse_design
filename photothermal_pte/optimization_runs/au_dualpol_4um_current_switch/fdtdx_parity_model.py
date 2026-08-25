"""Fresh FDTDX object/array builder for the 4-um parity route.

No historical integer layout, rho-cubed law, or endpoint-scaled Au pole is
imported here.  Every placement is generated from the physical-coordinate
contract and audited after FDTDX resolves the rectilinear grid.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_ade import (
    BASES as AU_BASES,
    carrier_dt_s,
    carrier_omega_rad_s,
    coefficient_hash as au_coefficient_hash,
    coefficients_jax as au_coefficients_jax,
    lorentz_parameters as au_lorentz_parameters,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_contract import (
    PHYSICS,
    fdtdx_float32_grid_edges,
    fdtdx_runtime_audit,
    grid_edges,
    grid_hashes,
    placement_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_fixed_materials import (
    TA_A,
    TA_B,
    coefficient_hash as fixed_coefficient_hash,
    load_material_targets,
    lorentz_parameters as fixed_lorentz_parameters,
)


MODEL_SCHEMA = "fdtdx_4um_parity_model_v1"
MODEL_SEED = 20260825
TOTAL_PERIODS = 40
LATE_PERIODS = 4
PREVIOUS_PERIODS = 4
SOURCE_STARTUP_PERIODS = 4
CLOSED_SURFACE_PHASOR_APODIZATION = None


def polarization_vector(polarization: str) -> tuple[float, float, float]:
    if polarization == "Ea":
        return (0.0, 1.0, 0.0)
    if polarization == "Eb":
        return (1.0, 0.0, 0.0)
    raise ValueError(f"unknown polarization {polarization!r}")


def _slice_pair(value: slice) -> list[int]:
    return [int(value.start), int(value.stop)]


def _slice_payload(value: tuple[slice, slice, slice]) -> list[list[int]]:
    return [_slice_pair(part) for part in value]


def model_plan(polarization: str, *, air_only: bool = False) -> dict[str, object]:
    """Pure placement plan; safe to inspect without importing FDTDX/JAX."""

    pol = polarization_vector(polarization)
    placement = placement_contract()
    volumes = placement["volumes_cell_slices"]
    planes = placement["planes_edge_indices"]
    source_xy = placement["source_aperture_cell_slices_xy"]
    planned = {
        "fixed_silicon_substrate": volumes["Si"],
        "fixed_285nm_sio2": volumes["SiO2"],
        "fixed_tairte4": volumes["TaIrTe4"],
        "au_design": volumes["Au_design"],
        "gaussian_source": [source_xy[0], source_xy[1], [planes["source"]["index"], planes["source"]["index"] + 1]],
        "incident_plane": [source_xy[0], source_xy[1], [planes["incident_power"]["index"], planes["incident_power"]["index"] + 1]],
        "endpoint_field": [source_xy[0], source_xy[1], [planes["air_endpoint_field"]["index"], planes["air_endpoint_field"]["index"] + 1]],
        "flake_profile": [source_xy[0], source_xy[1], [planes["flake_profile"]["index"], planes["flake_profile"]["index"] + 1]],
        "material_flux": volumes["closed_flux_box"],
        "material_flux_td": volumes["closed_flux_box"],
    }
    return {
        "schema": MODEL_SCHEMA,
        "polarization": polarization,
        "electric_polarization_vector": list(pol),
        "air_only": bool(air_only),
        "grid_shape": [186, 186, 286],
        "planned_float64_grid_xyz_edges_sha256": grid_hashes()["xyz_edges_sha256"],
        "grid_xyz_edges_sha256": grid_hashes()["fdtdx_float32_xyz_edges_sha256"],
        "time": {
            "courant_factor": PHYSICS.courant_factor,
            "total_periods": TOTAL_PERIODS,
            "late_periods": LATE_PERIODS,
            "previous_periods": PREVIOUS_PERIODS,
            "source_startup_periods": SOURCE_STARTUP_PERIODS,
            "dt_s": carrier_dt_s(),
            "time_steps_total": 256_163,
        },
        "source": {
            "direction": "minus_z",
            "radius_m": 8.0e-6,
            "std_relative_to_radius": PHYSICS.gaussian_waist_m / (math.sqrt(2.0) * 8.0e-6),
            "target_intensity_1e2_radius_m": PHYSICS.gaussian_waist_m,
            "static_amplitude_is_not_physical_power": True,
            "requires_separate_Ea_Eb_all_air_calibration": True,
        },
        "planned_slices": planned,
        "material_hashes": {
            "Au_nk_square_ADE": au_coefficient_hash(),
            "TaIrTe4_fixed_ADE": fixed_coefficient_hash(),
        },
        "design": {
            "input": "one_shared_80x80_cell_occupancy_derived_from_81x81_nodes",
            "Au_pole_weights": ["rho", "rho_squared", "rho_squared"],
            "Au_z_cells": 20,
            "c4": "absent_all_positive_Lorentz",
        },
        "optimizer_enabled": False,
    }


def _lower_edge_constraint(fdtdx: Any, obj: Any, edges: tuple[np.ndarray, ...], lower: tuple[int, int, int]):
    return fdtdx.RealCoordinateConstraint(
        object=obj.name,
        axes=(0, 1, 2),
        sides=("-", "-", "-"),
        coordinates=tuple(float(edges[axis][index]) for axis, index in enumerate(lower)),
    )


def _shape_and_lower(payload: list[list[int]]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    lower = tuple(int(pair[0]) for pair in payload)
    shape = tuple(int(pair[1]) - int(pair[0]) for pair in payload)
    return shape, lower


def _positive_lorentz_model(fdtdx: Any, bases: tuple[Any, ...]):
    poles = []
    for basis in bases:
        params = au_lorentz_parameters(basis)
        poles.append(
            fdtdx.LorentzPole(
                resonance_frequency=params["omega0_rad_s"],
                damping=params["gamma_rad_s"],
                delta_epsilon=params["delta_epsilon"],
            )
        )
    return fdtdx.DispersionModel(poles=tuple(poles))


def build_model(
    polarization: str,
    *,
    backend: str = "gpu",
    air_only: bool = False,
) -> dict[str, Any]:
    """Allocate/place the exact 40-period parity setup without running fields."""

    plan = model_plan(polarization, air_only=air_only)
    runtime = fdtdx_runtime_audit()
    if runtime["status"] != "PASS":
        raise RuntimeError(f"FDTDX runtime provenance failed: {runtime}")

    import jax
    import jax.numpy as jnp
    import fdtdx

    cublas_runtime_version = None
    if backend == "gpu":
        try:
            from jax._src.lib import cuda_versions

            cublas_runtime_version = int(cuda_versions.cublas_get_version())
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError("cannot audit the loaded cuBLAS runtime version") from exc
        if cublas_runtime_version < 130200:
            raise RuntimeError(
                "cuBLAS runtime older than 13.2 is blocked because concurrent "
                "kernels can silently corrupt B200 results; observed "
                f"{cublas_runtime_version}"
            )

    x_edges, y_edges, z_edges = grid_edges()
    edges = (x_edges, y_edges, z_edges)
    grid = fdtdx.RectilinearGrid.custom(
        x_edges=x_edges,
        y_edges=y_edges,
        z_edges=z_edges,
    )
    period_s = PHYSICS.wavelength_m / 299_792_458.0
    config = fdtdx.SimulationConfig(
        grid=grid,
        time=TOTAL_PERIODS * period_s,
        dtype=jnp.float32,
        courant_factor=PHYSICS.courant_factor,
        backend=backend,
        gradient_config=None,
    )
    if float(config.time_step_duration) != carrier_dt_s():
        raise RuntimeError("FDTDX config dt differs from certified carrier dt")
    if config.time_steps_total != 256_163:
        raise RuntimeError(f"unexpected parity time-step count {config.time_steps_total}")

    objects: list[Any] = []
    constraints: list[Any] = []
    volume = fdtdx.SimulationVolume(
        name="air_volume",
        partial_grid_shape=grid.shape,
        material=fdtdx.Material(permittivity=1.0),
    )
    objects.append(volume)
    boundaries, boundary_constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig(
            thickness_grid_minx=PHYSICS.pml_cells_each_face,
            thickness_grid_maxx=PHYSICS.pml_cells_each_face,
            thickness_grid_miny=PHYSICS.pml_cells_each_face,
            thickness_grid_maxy=PHYSICS.pml_cells_each_face,
            thickness_grid_minz=PHYSICS.pml_cells_each_face,
            thickness_grid_maxz=PHYSICS.pml_cells_each_face,
        ),
        volume,
    )
    objects.extend(boundaries.values())
    constraints.extend(boundary_constraints)

    targets = load_material_targets()["materials"]
    planned = plan["planned_slices"]
    if air_only:
        silicon_material = sio2_material = flake_material = au_material = fdtdx.Material(permittivity=1.0)
    else:
        silicon_material = fdtdx.Material(permittivity=float(targets["Si"]["epsilon"]["real"]))
        sio2_material = fdtdx.Material(permittivity=float(targets["SiO2"]["epsilon"]["real"]))
        ta_params = fixed_lorentz_parameters(TA_A)
        ta_placeholder = fdtdx.DispersionModel(
            poles=(
                fdtdx.LorentzPole(
                    resonance_frequency=ta_params["omega0_rad_s"],
                    damping=ta_params["gamma_rad_s"],
                    delta_epsilon=ta_params["delta_epsilon"],
                ),
            )
        )
        flake_material = fdtdx.Material(permittivity=1.0, dispersion=ta_placeholder)
        au_material = fdtdx.Material(
            permittivity=1.0,
            dispersion=_positive_lorentz_model(fdtdx, AU_BASES),
        )

    material_specs = (
        ("fixed_silicon_substrate", silicon_material),
        ("fixed_285nm_sio2", sio2_material),
        ("fixed_tairte4", flake_material),
        ("au_design", au_material),
    )
    material_objects: dict[str, Any] = {}
    for name, material in material_specs:
        shape, lower = _shape_and_lower(planned[name])
        obj = fdtdx.UniformMaterialObject(
            name=name,
            partial_grid_shape=shape,
            material=material,
        )
        material_objects[name] = obj
        objects.append(obj)
        constraints.append(_lower_edge_constraint(fdtdx, obj, edges, lower))

    source_shape, source_lower = _shape_and_lower(planned["gaussian_source"])
    source = fdtdx.GaussianPlaneSource(
        name="gaussian_source",
        partial_grid_shape=source_shape,
        fixed_E_polarization_vector=polarization_vector(polarization),
        wave_character=fdtdx.WaveCharacter(wavelength=PHYSICS.wavelength_m),
        temporal_profile=fdtdx.SingleFrequencyProfile(
            phase_shift=0.0,
            num_startup_periods=SOURCE_STARTUP_PERIODS,
        ),
        radius=8.0e-6,
        std=PHYSICS.gaussian_waist_m / (math.sqrt(2.0) * 8.0e-6),
        direction="-",
    )
    objects.append(source)
    constraints.append(_lower_edge_constraint(fdtdx, source, edges, source_lower))

    wave = fdtdx.WaveCharacter(wavelength=PHYSICS.wavelength_m)
    previous_switch = fdtdx.OnOffSwitch(
        start_time=(TOTAL_PERIODS - PREVIOUS_PERIODS - LATE_PERIODS) * period_s,
        end_time=(TOTAL_PERIODS - LATE_PERIODS) * period_s,
    )
    late_switch = fdtdx.OnOffSwitch(
        start_time=(TOTAL_PERIODS - LATE_PERIODS) * period_s
    )
    simulation_end_s = (config.time_steps_total - 1) * float(config.time_step_duration)
    windows = {
        "previous": fdtdx.TukeyWindow(
            start_time=(TOTAL_PERIODS - PREVIOUS_PERIODS - LATE_PERIODS) * period_s,
            end_time=(TOTAL_PERIODS - LATE_PERIODS) * period_s,
            alpha=1.0,
        ),
        "late": fdtdx.TukeyWindow(
            start_time=(TOTAL_PERIODS - LATE_PERIODS) * period_s,
            end_time=simulation_end_s,
            alpha=1.0,
        ),
    }
    for prefix, material_name in (("au", "au_design"), ("tairte4", "fixed_tairte4")):
        target = material_objects[material_name]
        for window_name, switch in (
            ("previous", previous_switch),
            ("late", late_switch),
        ):
            detector = fdtdx.PhasorDetector(
                name=f"{prefix}_{window_name}",
                partial_grid_shape=target.partial_grid_shape,
                wave_characters=(wave,),
                components=("Ex", "Ey", "Ez"),
                dtype=jnp.complex64,
                switch=switch,
                apodization=windows[window_name],
                exact_interpolation=False,
                plot=False,
            )
            constraints.append(detector.same_position(target))
            objects.append(detector)

    incident_shape, incident_lower = _shape_and_lower(planned["incident_plane"])
    incident = fdtdx.PhasorPoyntingFluxDetector(
        name="incident_plane",
        partial_grid_shape=incident_shape,
        wave_characters=(wave,),
        direction="-",
        dtype=jnp.complex64,
        switch=late_switch,
        apodization=windows["late"],
        exact_interpolation=True,
    )
    objects.append(incident)
    constraints.append(_lower_edge_constraint(fdtdx, incident, edges, incident_lower))

    for detector_name in ("endpoint_field", "flake_profile"):
        detector_shape, detector_lower = _shape_and_lower(planned[detector_name])
        detector = fdtdx.PhasorDetector(
            name=detector_name,
            partial_grid_shape=detector_shape,
            wave_characters=(wave,),
            components=("Ex", "Ey", "Ez"),
            dtype=jnp.complex64,
            switch=late_switch,
            apodization=windows["late"],
            exact_interpolation=True,
            plot=False,
        )
        objects.append(detector)
        constraints.append(_lower_edge_constraint(fdtdx, detector, edges, detector_lower))

    flux_shape, flux_lower = _shape_and_lower(planned["material_flux"])
    closed = fdtdx.ClosedSurfacePhasorPoyntingFluxDetector(
        name="material_flux",
        partial_grid_shape=flux_shape,
        wave_characters=(wave,),
        orientation="inward",
        dtype=jnp.complex64,
        switch=late_switch,
        apodization=CLOSED_SURFACE_PHASOR_APODIZATION,
        exact_interpolation=True,
    )
    closed_td = fdtdx.ClosedSurfacePoyntingFluxDetector(
        name="material_flux_td",
        partial_grid_shape=flux_shape,
        orientation="inward",
        switch=late_switch,
    )
    for detector in (closed, closed_td):
        objects.append(detector)
        constraints.append(_lower_edge_constraint(fdtdx, detector, edges, flux_lower))

    key = jax.random.PRNGKey(MODEL_SEED)
    placed, base, params, config, info = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    base, placed, _ = fdtdx.apply_params(base, placed, params, key)
    realized_grid = config.resolved_grid
    if realized_grid is None:
        raise RuntimeError("FDTDX did not resolve the parity grid")

    actual_slices = {
        name: placed[name].grid_slice
        for name in planned
    }
    actual_payload = {name: _slice_payload(value) for name, value in actual_slices.items()}
    mismatches = {
        name: {"expected": planned[name], "actual": actual_payload[name]}
        for name in planned
        if actual_payload[name] != planned[name]
    }
    if mismatches:
        raise RuntimeError(f"physical-coordinate placement mismatch: {mismatches}")
    if tuple(realized_grid.shape) != (186, 186, 286):
        raise RuntimeError(f"realized grid shape changed: {realized_grid.shape}")
    expected_solver_edges = fdtdx_float32_grid_edges()
    edge_mismatch_axes = [
        axis
        for axis in range(3)
        if not np.array_equal(
            np.asarray(realized_grid.edges(axis)), expected_solver_edges[axis]
        )
    ]
    if edge_mismatch_axes:
        raise RuntimeError(
            "FDTDX realized grid edges differ from the certified float32 solver "
            f"grid on axes {edge_mismatch_axes}"
        )

    fixed_c1 = fixed_c2 = fixed_c3 = None
    if not air_only:
        if base.dispersive_c1 is None or base.fields.dispersive_P_curr is None:
            raise RuntimeError("dispersive state was not allocated")
        spatial_shape = tuple(int(value) for value in base.dispersive_c1.shape[-3:])
        fixed_c1 = jnp.zeros((3, 3, *spatial_shape), dtype=jnp.float32)
        fixed_c2 = jnp.zeros_like(fixed_c1)
        fixed_c3 = jnp.zeros_like(fixed_c1)
        ta_slice = actual_slices["fixed_tairte4"]
        for component, carrier in enumerate((TA_B, TA_A, TA_B)):
            fixed_c1 = fixed_c1.at[(0, component, *ta_slice)].set(np.float32(carrier.c1))
            fixed_c2 = fixed_c2.at[(0, component, *ta_slice)].set(np.float32(carrier.c2))
            fixed_c3 = fixed_c3.at[(0, component, *ta_slice)].set(np.float32(carrier.c3))
        au_slice = actual_slices["au_design"]
        for pole, basis in enumerate(AU_BASES):
            for component in range(3):
                fixed_c1 = fixed_c1.at[(pole, component, *au_slice)].set(np.float32(basis.c1))
                fixed_c2 = fixed_c2.at[(pole, component, *au_slice)].set(np.float32(basis.c2))
        base = (
            base
            .aset("dispersive_c1", fixed_c1)
            .aset("dispersive_c2", fixed_c2)
            .aset("dispersive_c3", fixed_c3)
        )
        if base.dispersive_c4 is not None:
            raise RuntimeError("positive-Lorentz parity model unexpectedly allocated c4")

    return {
        "schema": MODEL_SCHEMA,
        "plan": plan,
        "runtime": runtime,
        "cublas_runtime_version": cublas_runtime_version,
        "jax": jax,
        "jnp": jnp,
        "fdtdx": fdtdx,
        "key": key,
        "config": config,
        "grid": realized_grid,
        "placed": placed,
        "base": base,
        "slices": actual_slices,
        "placement": actual_payload,
        "fixed_c1": fixed_c1,
        "fixed_c2": fixed_c2,
        "fixed_c3": fixed_c3,
        "air_only": bool(air_only),
        "polarization": polarization,
        "omega_rad_s": carrier_omega_rad_s(),
        "placement_info": info,
    }


def arrays_for_density(model: dict[str, Any], rho_cell: Any):
    """Apply the shared 80x80 occupancy to only the three Au pole couplings."""

    if model["air_only"]:
        raise RuntimeError("air-only source model has no Au density carrier")
    jnp = model["jnp"]
    rho = jnp.asarray(rho_cell, dtype=jnp.float32)
    if rho.shape != (80, 80):
        raise ValueError(f"Au cell occupancy must be 80x80, got {rho.shape}")
    _, _, c3_cell, c4_cell = au_coefficients_jax(rho)
    if c4_cell.shape != (3, 80, 80):
        raise RuntimeError("unexpected Au coefficient shape")
    c3 = model["fixed_c3"]
    au_slice = model["slices"]["au_design"]
    expanded = jnp.broadcast_to(c3_cell[:, :, :, None], (3, 80, 80, 20))
    for component in range(3):
        c3 = c3.at[(slice(None), component, *au_slice)].set(expanded)
    return model["base"].reset().aset("dispersive_c3", c3)


def setup_audit(model: dict[str, Any]) -> dict[str, object]:
    """Small host-readable audit after allocation, before any FDTD time step."""

    base = model["base"]
    checks = {
        "schema": model["schema"] == MODEL_SCHEMA,
        "runtime": model["runtime"]["status"] == "PASS",
        "cublas_at_least_13_2": (
            model["cublas_runtime_version"] is None
            or model["cublas_runtime_version"] >= 130200
        ),
        "grid_shape": tuple(model["grid"].shape) == (186, 186, 286),
        "time_steps": model["config"].time_steps_total == 256_163,
        "dt": float(model["config"].time_step_duration) == carrier_dt_s(),
        "placement": model["placement"] == model["plan"]["planned_slices"],
        "pml_count": len(model["placed"].pml_objects) == 6,
    }
    arrays: dict[str, object] = {
        "E_shape": list(base.fields.E.shape),
        "H_shape": list(base.fields.H.shape),
        "inv_permittivity_shape": list(base.inv_permittivities.shape),
    }
    if model["air_only"]:
        checks["no_ADE"] = base.dispersive_c1 is None
    else:
        checks.update(
            three_ADE_poles=base.dispersive_c1 is not None and base.dispersive_c1.shape[:2] == (3, 3),
            three_polarization_states=(
                base.fields.dispersive_P_curr is not None
                and base.fields.dispersive_P_curr.shape[:2] == (3, 3)
            ),
            c4_absent=base.dispersive_c4 is None,
        )
        arrays.update(
            dispersive_c1_shape=list(base.dispersive_c1.shape),
            dispersive_P_curr_shape=list(base.fields.dispersive_P_curr.shape),
        )
    leaves = model["jax"].tree_util.tree_leaves(base)
    allocated_bytes = sum(
        int(getattr(leaf, "size", 0)) * int(getattr(leaf, "dtype", np.dtype("u1")).itemsize)
        for leaf in leaves
        if hasattr(leaf, "dtype")
    )
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "arrays": arrays,
        "array_container_leaf_bytes": allocated_bytes,
        "array_container_leaf_GiB": allocated_bytes / 2**30,
        "device_platforms": sorted({device.platform for device in model["jax"].devices()}),
        "optimizer_enabled": False,
        "field_steps_executed": 0,
    }
