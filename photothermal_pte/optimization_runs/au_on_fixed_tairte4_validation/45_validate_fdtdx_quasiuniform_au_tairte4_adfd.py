#!/usr/bin/env python3
"""Validate the FDTDX quasi-uniform-grid Au/TaIrTe4 optical gradient.

This is a physical-thickness bridge between the small 100 nm cubic-grid
algorithmic control and a future production Gaussian-beam model.  It uses the
official FDTDX ``main`` rectilinear-grid implementation, with 100 nm lateral
cells and 25 nm vertical cells.  Consequently the fixed 100 nm TaIrTe4 film is
four cells thick and the 50 nm Au design layer is two cells thick.

The optical source remains a compact diagnostic source.  It is deliberately
not called the production w0=8.5 um beam.  No thermal, PTE, electrode, or
optimization calculation is performed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
STAGE41 = HERE / "41_validate_au_on_fixed_tairte4_optical_adfd.py"
FDTDX_MAIN_SOURCE = Path("/home/seunghyun/.local/fdtdx_main_src")
FDTDX_MAIN_PYTHON = FDTDX_MAIN_SOURCE / "src"
FDTDX_DEPENDENCIES = Path("/home/seunghyun/.local/au_fdtdx")
POWER_SCALE_W = 1.0e-24


def _load_stage41():
    spec = importlib.util.spec_from_file_location("stage41_fdtdx_control", STAGE41)
    if spec is None or spec.loader is None:
        raise ImportError(STAGE41)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value):
    """Convert NumPy scalar diagnostics without weakening schema fidelity."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


@dataclass(frozen=True)
class BridgeConfig:
    dx_m: float = 100.0e-9
    dy_m: float = 100.0e-9
    dz_m: float = 25.0e-9
    domain_cells_xyz: tuple[int, int, int] = (40, 40, 160)
    pml_cells_xyz: tuple[int, int, int] = (8, 8, 32)
    design_cells_xyz: tuple[int, int, int] = (10, 10, 2)
    flake_cells_xyz: tuple[int, int, int] = (14, 14, 4)
    source_cells_xy: tuple[int, int] = (20, 20)
    total_periods: int = 14
    phasor_periods: int = 3
    checkpoints: int = 8
    courant_factor: float = 0.5
    seed: int = 20260821


def _slice_tuple(grid_slice: tuple[slice, slice, slice]) -> tuple[tuple[int, int], ...]:
    return tuple((int(item.start), int(item.stop)) for item in grid_slice)


def _direction_set(nx: int, ny: int, seed: int) -> dict[str, np.ndarray]:
    x = np.linspace(-1.0, 1.0, nx)[:, None]
    y = np.linspace(-1.0, 1.0, ny)[None, :]
    edge = np.zeros((nx, ny), dtype=np.float64)
    edge[:2, :] = 1.0
    edge[-2:, :] -= 0.8
    rng = np.random.default_rng(seed)
    values = {
        "uniform": np.ones((nx, ny), dtype=np.float64),
        "smooth_asymmetric": np.sin(0.7 * math.pi * x) * np.cos(0.55 * math.pi * y) + 0.23 * x,
        "central_localized": np.exp(-((x / 0.34) ** 2 + (y / 0.34) ** 2)),
        "design_edge_localized": edge,
        "fixed_seed_random": rng.standard_normal((nx, ny)),
    }
    return {name: value / np.linalg.norm(value) for name, value in values.items()}


def _baseline_density(nx: int, ny: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, nx)[:, None]
    y = np.linspace(-1.0, 1.0, ny)[None, :]
    rho = 0.53 + 0.08 * np.cos(math.pi * x) * np.cos(0.7 * math.pi * y) + 0.025 * x
    if rho.min() <= 0.15 or rho.max() >= 0.85:
        raise RuntimeError("Baseline density lacks unclipped central-FD margin")
    return rho.astype(np.float32)


def run_bridge(output_dir: Path, cfg: BridgeConfig, *, forward_only: bool = False) -> dict[str, object]:
    if not FDTDX_MAIN_PYTHON.is_dir() or not FDTDX_DEPENDENCIES.is_dir():
        raise RuntimeError("Pinned FDTDX main source or dependency environment is missing")
    import jax
    import jax.numpy as jnp
    import fdtdx
    from fdtdx.fdtd.fdtd import checkpointed_fdtd

    imported_fdtdx = Path(fdtdx.__file__).resolve()
    if FDTDX_MAIN_PYTHON.resolve() not in imported_fdtdx.parents:
        raise RuntimeError(
            "The pinned FDTDX main source was not imported: "
            f"expected a module below {FDTDX_MAIN_PYTHON}, got {imported_fdtdx}"
        )

    stage41 = _load_stage41()
    devices = jax.devices()
    if not devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"GPU-only contract violated: {devices}")

    wavelength_m = stage41.WAVELENGTH_M
    omega = 2.0 * math.pi * stage41.C0_M_PER_S / wavelength_m
    period_s = wavelength_m / stage41.C0_M_PER_S
    config = fdtdx.SimulationConfig(
        grid=fdtdx.QuasiUniformGrid(dx=cfg.dx_m, dy=cfg.dy_m, dz=cfg.dz_m),
        time=cfg.total_periods * period_s,
        dtype=jnp.float32,
        courant_factor=cfg.courant_factor,
        backend="gpu",
        gradient_config=fdtdx.GradientConfig(method="checkpointed", num_checkpoints=cfg.checkpoints),
    )
    dt = config.time_step_duration
    epsilon_au = complex(stage41.AU_N, stage41.AU_K) ** 2
    epsilon_tairte4 = stage41._load_tairte4_epsilon()
    fits = {
        "au": stage41._drude_fit(epsilon_au, omega, dt),
        "a": stage41._drude_fit(epsilon_tairte4["a"], omega, dt),
        "b": stage41._lorentz_fit(epsilon_tairte4["b"], omega, dt),
    }
    fits["c"] = dict(fits["b"])
    coefficients = {name: stage41._coefficient_triplet(fit, dt) for name, fit in fits.items()}

    # Use physical-time switches rather than precomputed time-step indices.  A
    # QuasiUniformGrid is resolved to an explicit RectilinearGrid by
    # ``place_objects``; floating-point edge reconstruction can change the
    # rounded step count by one.  Physical-time windows remain invariant across
    # that placement transition.
    previous_switch = fdtdx.OnOffSwitch(
        start_time=(cfg.total_periods - 2 * cfg.phasor_periods) * period_s,
        end_time=(cfg.total_periods - cfg.phasor_periods) * period_s,
    )
    late_switch = fdtdx.OnOffSwitch(
        start_time=(cfg.total_periods - cfg.phasor_periods) * period_s,
    )

    nx, ny, nz = cfg.domain_cells_xyz
    px, py, pz = cfg.pml_cells_xyz
    objects: list[object] = []
    constraints: list[object] = []
    volume = fdtdx.SimulationVolume(
        name="air_volume",
        partial_grid_shape=cfg.domain_cells_xyz,
        material=fdtdx.Material(permittivity=1.0),
    )
    objects.append(volume)
    boundary_config = fdtdx.BoundaryConfig(
        thickness_grid_minx=px,
        thickness_grid_maxx=px,
        thickness_grid_miny=py,
        thickness_grid_maxy=py,
        thickness_grid_minz=pz,
        thickness_grid_maxz=pz,
    )
    boundary_dict, boundary_constraints = fdtdx.boundary_objects_from_config(boundary_config, volume)
    constraints.extend(boundary_constraints)
    objects.extend(boundary_dict.values())

    au_dispersion = fdtdx.DispersionModel(
        poles=(
            fdtdx.DrudePole(
                plasma_frequency=fits["au"]["omega_p_rad_s"],
                damping=fits["au"]["gamma_rad_s"],
            ),
        )
    )
    tairte4_placeholder = fdtdx.DispersionModel(
        poles=(
            fdtdx.DrudePole(
                plasma_frequency=fits["a"]["omega_p_rad_s"],
                damping=fits["a"]["gamma_rad_s"],
            ),
        )
    )
    flake = fdtdx.UniformMaterialObject(
        name="fixed_tairte4",
        partial_grid_shape=cfg.flake_cells_xyz,
        material=fdtdx.Material(permittivity=stage41.EPS_INF, dispersion=tairte4_placeholder),
    )
    design = fdtdx.UniformMaterialObject(
        name="au_nanostructure_design",
        partial_grid_shape=cfg.design_cells_xyz,
        material=fdtdx.Material(permittivity=stage41.EPS_INF, dispersion=au_dispersion),
    )
    constraints.extend(
        [
            flake.place_at_center(volume),
            design.place_at_center(volume, axes=(0, 1)),
            design.place_above(flake),
        ]
    )
    objects.extend([flake, design])

    source_radius_m = 0.5 * cfg.source_cells_xy[0] * cfg.dx_m
    source_std = 0.42
    source = fdtdx.GaussianPlaneSource(
        name="compact_gaussian_source",
        partial_grid_shape=(*cfg.source_cells_xy, 1),
        fixed_E_polarization_vector=(1.0, 0.0, 0.0),
        wave_character=fdtdx.WaveCharacter(wavelength=wavelength_m),
        radius=source_radius_m,
        std=source_std,
        direction="-",
    )
    source_z_m = 1.0e-6
    constraints.extend(
        [
            source.place_at_center(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(2,), margins=(source_z_m,)),
        ]
    )
    objects.append(source)

    wave = fdtdx.WaveCharacter(wavelength=wavelength_m)
    for material_name, target in (("au", design), ("tairte4", flake)):
        for window_name, switch in (("previous", previous_switch), ("late", late_switch)):
            detector = fdtdx.PhasorDetector(
                name=f"{material_name}_{window_name}",
                partial_grid_shape=target.partial_grid_shape,
                wave_characters=(wave,),
                components=("Ex", "Ey", "Ez"),
                dtype=jnp.complex64,
                switch=switch,
                exact_interpolation=True,
                plot=False,
            )
            constraints.append(detector.same_position(target))
            objects.append(detector)

    flux_box = fdtdx.ClosedSurfacePhasorPoyntingFluxDetector(
        name="material_flux_late",
        partial_grid_shape=(18, 18, 22),
        wave_characters=(wave,),
        orientation="inward",
        dtype=jnp.complex64,
        switch=late_switch,
        exact_interpolation=True,
    )
    constraints.extend(
        [
            flux_box.place_at_center(volume, axes=(0, 1)),
            flux_box.place_at_center(volume, axes=(2,), margins=(25.0e-9,)),
        ]
    )
    objects.append(flux_box)

    net_down_detector = fdtdx.PhasorPoyntingFluxDetector(
        name="net_down_late",
        partial_grid_shape=(*cfg.source_cells_xy, 1),
        wave_characters=(wave,),
        direction="-",
        dtype=jnp.complex64,
        switch=late_switch,
        exact_interpolation=True,
    )
    constraints.extend(
        [
            net_down_detector.place_at_center(volume, axes=(0, 1)),
            net_down_detector.place_at_center(volume, axes=(2,), margins=(0.6e-6,)),
        ]
    )
    objects.append(net_down_detector)

    key = jax.random.PRNGKey(cfg.seed)
    placed_objects, base_arrays, _, config, _ = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    base_arrays, placed_objects, _ = fdtdx.apply_params(base_arrays, placed_objects, {}, key)
    if not config.has_nonuniform_grid:
        raise RuntimeError("Quasi-uniform bridge did not realize as a non-uniform solver grid")
    period_steps = int(round(period_s / config.time_step_duration))
    total_steps = config.time_steps_total
    if any(
        value is None
        for value in (base_arrays.dispersive_c1, base_arrays.dispersive_c2, base_arrays.dispersive_c3)
    ):
        raise RuntimeError("FDTDX did not allocate ADE arrays")

    au_slice = placed_objects["au_nanostructure_design"].grid_slice
    flake_slice = placed_objects["fixed_tairte4"].grid_slice
    au_shape = placed_objects["au_nanostructure_design"].grid_shape
    flake_shape = placed_objects["fixed_tairte4"].grid_shape
    if au_shape != cfg.design_cells_xyz or flake_shape != cfg.flake_cells_xyz:
        raise RuntimeError(f"Realized geometry mismatch: Au={au_shape}, TaIrTe4={flake_shape}")
    if au_slice[2].start != flake_slice[2].stop:
        raise RuntimeError(f"Au/TaIrTe4 are not face-adjacent: Au={au_slice}, TaIrTe4={flake_slice}")

    grid = config.resolved_grid
    if grid is None:
        raise RuntimeError("Missing realized RectilinearGrid")
    au_dvol = jnp.asarray(grid.cell_volume(_slice_tuple(au_slice)), dtype=jnp.float32)
    ta_dvol = jnp.asarray(grid.cell_volume(_slice_tuple(flake_slice)), dtype=jnp.float32)
    realized_au_thickness = float(grid.slice_extent(_slice_tuple(au_slice))[2])
    realized_ta_thickness = float(grid.slice_extent(_slice_tuple(flake_slice))[2])
    # Rectilinear edge coordinates are carried through the float32 solver
    # layout.  Audit the physical thickness at a tolerance tighter than
    # 0.001%, without demanding impossible femtometre-level equality.
    if not math.isclose(realized_au_thickness, 50.0e-9, rel_tol=1e-5, abs_tol=1e-15):
        raise RuntimeError(f"Au thickness mismatch: {realized_au_thickness}")
    if not math.isclose(realized_ta_thickness, 100.0e-9, rel_tol=1e-5, abs_tol=1e-15):
        raise RuntimeError(f"TaIrTe4 thickness mismatch: {realized_ta_thickness}")

    spatial_shape = base_arrays.dispersive_c1.shape[-3:]
    fixed_c1 = jnp.zeros((1, 3, *spatial_shape), dtype=jnp.float32)
    fixed_c2 = jnp.zeros_like(fixed_c1)
    fixed_c3 = jnp.zeros_like(fixed_c1)
    for component, axis in enumerate(("b", "a", "c")):
        c1, c2, c3 = coefficients[axis]
        index = (0, component, *flake_slice)
        fixed_c1 = fixed_c1.at[index].set(c1)
        fixed_c2 = fixed_c2.at[index].set(c2)
        fixed_c3 = fixed_c3.at[index].set(c3)
    au_c1, au_c2, au_c3 = coefficients["au"]
    for component in range(3):
        index = (0, component, *au_slice)
        fixed_c1 = fixed_c1.at[index].set(au_c1)
        fixed_c2 = fixed_c2.at[index].set(au_c2)
    au_strength_template = jnp.zeros(spatial_shape, dtype=jnp.float32)

    def arrays_for_density(rho: jax.Array):
        strength = jnp.broadcast_to((rho**3)[:, :, None], au_shape)
        full_strength = au_strength_template.at[au_slice].set(strength)
        del full_strength
        c3 = fixed_c3
        for component in range(3):
            index = (0, component, *au_slice)
            c3 = c3.at[index].set(au_c3 * strength)
        return (
            base_arrays.reset()
            .aset("dispersive_c1", fixed_c1)
            .aset("dispersive_c2", fixed_c2)
            .aset("dispersive_c3", c3)
        )

    au_prefactor = 0.5 * omega * stage41.EPS0_F_PER_M * epsilon_au.imag / POWER_SCALE_W
    ta_imag = jnp.asarray(
        [epsilon_tairte4["b"].imag, epsilon_tairte4["a"].imag, epsilon_tairte4["c"].imag],
        dtype=jnp.float32,
    )[:, None, None, None]
    ta_prefactor = 0.5 * omega * stage41.EPS0_F_PER_M / POWER_SCALE_W

    def powers_from_output(out_arrays, window_name: str, rho: jax.Array):
        au_e = out_arrays.detector_states[f"au_{window_name}"]["phasor"][0, 0]
        ta_e = out_arrays.detector_states[f"tairte4_{window_name}"]["phasor"][0, 0]
        strength = jnp.broadcast_to((rho**3)[:, :, None], au_shape)
        component_au = au_prefactor * jnp.sum(
            strength[None, ...] * jnp.abs(au_e) ** 2 * au_dvol[None, ...], axis=(1, 2, 3)
        )
        component_ta = ta_prefactor * jnp.sum(
            ta_imag * jnp.abs(ta_e) ** 2 * ta_dvol[None, ...], axis=(1, 2, 3)
        )
        p_au = jnp.sum(component_au)
        p_ta = jnp.sum(component_ta)
        return jnp.concatenate((jnp.stack((p_au, p_ta, p_au + p_ta)), component_au, component_ta))

    flux_object = placed_objects["material_flux_late"]
    net_down_object = placed_objects["net_down_late"]

    def solve_windows(rho: jax.Array):
        _, out = checkpointed_fdtd(arrays_for_density(rho), placed_objects, config, key, show_progress=False)
        late = powers_from_output(out, "late", rho)
        previous = powers_from_output(out, "previous", rho)
        p_six_scaled = flux_object.compute_net_flux(out.detector_states["material_flux_late"])[0] / POWER_SCALE_W
        p_net_down_scaled = net_down_object.compute_poynting_flux(out.detector_states["net_down_late"])[0] / POWER_SCALE_W
        return late, previous, p_six_scaled, p_net_down_scaled

    def objective_vector(rho: jax.Array):
        late, _, _, _ = solve_windows(rho)
        return late[:3]

    rho0_np = _baseline_density(cfg.design_cells_xyz[0], cfg.design_cells_xyz[1])
    rho0 = jnp.asarray(rho0_np)

    # Run the identical source, detector, grid and PML with every ADE field
    # coupling set to zero.  EPS_INF is one in both nominal material supports,
    # so this is an optically exact empty-air control on the same layout.  It
    # is reported as a signed diagnostic; raw and background-subtracted
    # closure must both remain visible and no result is rescaled.
    windows_jit = jax.jit(solve_windows).lower(rho0).compile()
    late_scaled, previous_scaled, p_six_scaled, p_net_down_scaled = windows_jit(rho0)
    late_all_w = np.asarray(late_scaled, dtype=np.float64) * POWER_SCALE_W
    previous_all_w = np.asarray(previous_scaled, dtype=np.float64) * POWER_SCALE_W
    window_changes = np.abs(late_all_w - previous_all_w) / np.maximum(np.abs(late_all_w), 1e-300)
    p_six_raw_w = float(p_six_scaled) * POWER_SCALE_W
    p_net_down_w = float(p_net_down_scaled) * POWER_SCALE_W

    def solve_empty_flux(empty_c3: jax.Array):
        empty = (
            base_arrays.reset()
            # Keep the passive pole recurrence layout but set its E coupling to
            # zero.  This is optically empty (P stays zero from zero initial
            # state) while preserving the source/kernel ADE contract.
            .aset("dispersive_c1", fixed_c1)
            .aset("dispersive_c2", fixed_c2)
            .aset("dispersive_c3", empty_c3)
        )
        _, out = fdtdx.run_fdtd(empty, placed_objects, config, key, show_progress=False)
        p_box = flux_object.compute_net_flux(out.detector_states["material_flux_late"])[0]
        p_down = net_down_object.compute_poynting_flux(out.detector_states["net_down_late"])[0]
        return p_box, p_down

    empty_c3 = jnp.zeros_like(fixed_c3)
    empty_flux_jit = jax.jit(solve_empty_flux).lower(empty_c3).compile()
    p_six_empty_scaled, p_net_down_empty_scaled = empty_flux_jit(empty_c3)
    # Unlike ``solve_windows``, ``solve_empty_flux`` returns native flux in W
    # and does not divide by POWER_SCALE_W.  Do not scale it a second time.
    p_six_empty_w = float(p_six_empty_scaled)
    p_net_down_empty_w = float(p_net_down_empty_scaled)
    p_six_corrected_w = p_six_raw_w - p_six_empty_w
    raw_closure_relative_error = abs(late_all_w[2] - p_six_raw_w) / max(abs(p_six_raw_w), 1e-300)
    closure_relative_error = abs(late_all_w[2] - p_six_corrected_w) / max(
        abs(p_six_corrected_w), 1e-300
    )

    placement_audit = {
        "source_slice": [list(value) for value in _slice_tuple(placed_objects["compact_gaussian_source"].grid_slice)],
        "closed_surface_slice": [list(value) for value in _slice_tuple(flux_object.grid_slice)],
        "net_down_detector_slice": [list(value) for value in _slice_tuple(net_down_object.grid_slice)],
    }
    if forward_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        forward_summary = {
            "status": "FDTDX_QUASIUNIFORM_FORWARD_CLOSURE_PROBE",
            "fdtdx_source_commit": _git_commit(FDTDX_MAIN_SOURCE),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "total_periods": cfg.total_periods,
            "phasor_periods_per_window": cfg.phasor_periods,
            "time_steps_total": total_steps,
            "grid_cell_size_m_xyz": [cfg.dx_m, cfg.dy_m, cfg.dz_m],
            "placement": placement_audit,
            "powers_W": dict(zip(("au", "tairte4", "total"), map(float, late_all_w[:3]))),
            "window_relative_change": dict(
                zip(("au", "tairte4", "total"), map(float, window_changes[:3]))
            ),
            "closed_surface_inward_raw_W": p_six_raw_w,
            "closed_surface_inward_empty_air_W": p_six_empty_w,
            "closed_surface_inward_material_minus_empty_W": p_six_corrected_w,
            "closed_surface_raw_closure_relative_error": raw_closure_relative_error,
            "closed_surface_background_subtracted_closure_relative_error": closure_relative_error,
            "net_down_material_W": p_net_down_w,
            "net_down_empty_air_W": p_net_down_empty_w,
            "note": (
                "Exact zero-coupling empty-air control on the same source/grid/PML; "
                "both raw and signed background-subtracted closure are reported. "
                "No Q clipping, gain, smoothing, or result rescaling."
            ),
        }
        path = output_dir / "fdtdx_quasiuniform_forward_closure_probe.json"
        path.write_text(json.dumps(forward_summary, indent=2, default=_json_default) + "\n")
        print(json.dumps(forward_summary, indent=2, default=_json_default))
        return forward_summary

    def vector_value_and_jacobian(rho: jax.Array):
        values, pullback = jax.vjp(objective_vector, rho)
        basis = jnp.eye(3, dtype=values.dtype)
        jacobian = jax.vmap(lambda cotangent: pullback(cotangent)[0])(basis)
        return values, jacobian

    compile_start = time.perf_counter()
    vector_value_and_jac = jax.jit(vector_value_and_jacobian).lower(rho0).compile()
    compile_seconds = time.perf_counter() - compile_start
    ad_start = time.perf_counter()
    scaled_powers, scaled_jacobian = vector_value_and_jac(rho0)
    jax.block_until_ready(scaled_jacobian)
    ad_seconds = time.perf_counter() - ad_start
    powers_w = np.asarray(scaled_powers, dtype=np.float64) * POWER_SCALE_W
    gradients_w = np.asarray(scaled_jacobian, dtype=np.float64) * POWER_SCALE_W
    observable_names = ("au", "tairte4", "total")
    gradient_l2 = {
        name: float(np.linalg.norm(gradients_w[index])) for index, name in enumerate(observable_names)
    }
    gradient_sum_error = float(
        np.linalg.norm(gradients_w[2] - gradients_w[0] - gradients_w[1])
        / max(np.linalg.norm(gradients_w[2]), 1e-300)
    )

    directions = _direction_set(cfg.design_cells_xyz[0], cfg.design_cells_xyz[1], cfg.seed)
    steps = (0.02, 0.01, 0.005)
    objective_jit = jax.jit(objective_vector).lower(rho0).compile()
    rows: list[dict[str, object]] = []
    fd_start = time.perf_counter()
    for direction_name, direction_np in directions.items():
        direction = jnp.asarray(direction_np, dtype=rho0.dtype)
        ad_vector = np.asarray([np.vdot(gradients_w[index], direction_np).real for index in range(3)])
        for h in steps:
            rho_plus = rho0 + h * direction
            rho_minus = rho0 - h * direction
            if float(jnp.min(rho_minus)) <= 0.0 or float(jnp.max(rho_plus)) >= 1.0:
                raise RuntimeError(f"FD {direction_name}, h={h} leaves [0,1]")
            plus_w = np.asarray(objective_jit(rho_plus), dtype=np.float64) * POWER_SCALE_W
            minus_w = np.asarray(objective_jit(rho_minus), dtype=np.float64) * POWER_SCALE_W
            fd_vector = (plus_w - minus_w) / (2.0 * h)
            for index, observable in enumerate(observable_names):
                ad_w = float(ad_vector[index])
                fd_w = float(fd_vector[index])
                strong = max(abs(ad_w), abs(fd_w)) >= 0.01 * gradient_l2[observable]
                rows.append(
                    {
                        "observable": observable,
                        "direction": direction_name,
                        "h": h,
                        "ad_W_per_unit_direction": ad_w,
                        "fd_W_per_unit_direction": fd_w,
                        "strong_relative_error": abs(ad_w - fd_w) / max(abs(fd_w), 1e-300),
                        "symmetric_normalized_error": abs(ad_w - fd_w)
                        / max(abs(ad_w) + abs(fd_w), 1e-300),
                        "gradient_l2_normalized_error": abs(ad_w - fd_w)
                        / max(gradient_l2[observable], 1e-300),
                        "strong_direction": strong,
                        "power_plus_W": float(plus_w[index]),
                        "power_minus_W": float(minus_w[index]),
                    }
                )
    fd_seconds = time.perf_counter() - fd_start

    finest_h = min(steps)
    finest_total = [
        row for row in rows if row["observable"] == "total" and float(row["h"]) == finest_h
    ]
    finest_total_strong = [row for row in finest_total if bool(row["strong_direction"])]
    max_finest_strong = max(float(row["strong_relative_error"]) for row in finest_total_strong)
    max_finest_gradient_normalized = max(
        float(row["gradient_l2_normalized_error"]) for row in finest_total
    )
    finite = bool(
        np.all(np.isfinite(late_all_w))
        and np.all(np.isfinite(previous_all_w))
        and np.all(np.isfinite(gradients_w))
        and np.isfinite(p_six_raw_w)
        and np.isfinite(p_six_empty_w)
        and np.isfinite(p_six_corrected_w)
        and np.isfinite(p_net_down_w)
        and all(np.isfinite(float(row["fd_W_per_unit_direction"])) for row in rows)
    )
    settled = bool(np.all(window_changes[:3] < 0.005))
    passed = bool(
        finite
        and settled
        and gradient_sum_error < 1e-6
        and max_finest_strong < 0.01
        and max_finest_gradient_normalized < 0.01
        and raw_closure_relative_error < 0.01
        and closure_relative_error < 0.01
    )
    status = (
        "VALIDATED_FDTDX_QUASIUNIFORM_AU_TAIRTE4_ADFD_BRIDGE"
        if passed
        else "FAILED_FDTDX_QUASIUNIFORM_AU_TAIRTE4_ADFD_BRIDGE"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fdtdx_quasiuniform_au_tairte4_adfd_directions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    images = [rho0_np, gradients_w[0], gradients_w[1], gradients_w[2]]
    titles = ["Au density", "dP_Au/drho", "dP_TaIrTe4/drho", "dP_total/drho"]
    for axis, image, title in zip(axes.flat[:4], images, titles):
        if title == "Au density":
            rendered = axis.imshow(image.T, origin="lower", cmap="gray", vmin=0, vmax=1)
        else:
            rendered = axis.imshow(image.T, origin="lower", cmap="coolwarm")
        axis.set_title(title)
        axis.set_xlabel("x=b design node")
        axis.set_ylabel("y=a design node")
        fig.colorbar(rendered, ax=axis, label="rho" if title == "Au density" else "W per rho")
    for direction_name in directions:
        current = [
            row
            for row in rows
            if row["observable"] == "total" and row["direction"] == direction_name
        ]
        axes[1, 1].loglog(
            [float(row["h"]) for row in current],
            [float(row["strong_relative_error"]) for row in current],
            marker="o",
            label=direction_name,
        )
    axes[1, 1].axhline(0.01, color="black", linestyle="--", label="1% gate")
    axes[1, 1].invert_xaxis()
    axes[1, 1].set_title("total-power directional AD-FD")
    axes[1, 1].set_xlabel("central-FD step h")
    axes[1, 1].set_ylabel("|AD-FD|/|FD|")
    axes[1, 1].legend(fontsize=7)
    axes[1, 2].bar(
        ("P_Au", "P_Ta", "P_total", "P_six raw"),
        (*late_all_w[:3], p_six_raw_w),
    )
    axes[1, 2].set_title("absorption and closed-surface flux")
    axes[1, 2].set_ylabel("W (compact-source normalization)")
    plot_path = output_dir / "fdtdx_quasiuniform_au_tairte4_adfd.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    summary = {
        "status": status,
        "scope": (
            "compact 3-D physical-thickness bridge on a quasi-uniform FDTDX grid; "
            "not the production w0=8.5 um beam, thermal/PTE model, electrode model, or optimization"
        ),
        "software": {
            "fdtdx_source_path": str(FDTDX_MAIN_SOURCE),
            "fdtdx_source_commit": _git_commit(FDTDX_MAIN_SOURCE),
            "fdtdx_import_path": fdtdx.__file__,
            "fdtdx_version": getattr(fdtdx, "__version__", "unknown"),
            "jax_version": jax.__version__,
            "jax_devices": [str(device) for device in devices],
            "python": sys.version,
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "grid": {
            "type": type(grid).__name__,
            "nonuniform": config.has_nonuniform_grid,
            "cell_size_m_xyz": [cfg.dx_m, cfg.dy_m, cfg.dz_m],
            "domain_cells_xyz": list(cfg.domain_cells_xyz),
            "domain_span_m_xyz": [nx * cfg.dx_m, ny * cfg.dy_m, nz * cfg.dz_m],
            "pml_cells_each_face_xyz": list(cfg.pml_cells_xyz),
            "pml_physical_thickness_m_xyz": [px * cfg.dx_m, py * cfg.dy_m, pz * cfg.dz_m],
            "realized_au_cells_xyz": list(au_shape),
            "realized_au_thickness_m": realized_au_thickness,
            "realized_tairte4_cells_xyz": list(flake_shape),
            "realized_tairte4_thickness_m": realized_ta_thickness,
            "au_slice": [list(value) for value in _slice_tuple(au_slice)],
            "tairte4_slice": [list(value) for value in _slice_tuple(flake_slice)],
            "placement_audit": placement_audit,
        },
        "source": {
            "kind": "FDTDX compact GaussianPlaneSource diagnostic",
            "wavelength_m": wavelength_m,
            "polarization": "x=b",
            "direction": "-z",
            "source_cells_xy": list(cfg.source_cells_xy),
            "source_radius_m": source_radius_m,
            "source_std_relative_to_radius": source_std,
            "derived_intensity_1_over_e2_radius_m": math.sqrt(2.0) * source_radius_m * source_std,
            "production_w0_8p5um": False,
            "normalization": "FDTDX internal source-energy normalization; no gain or result rescaling",
        },
        "materials": {
            "au_epsilon": [epsilon_au.real, epsilon_au.imag],
            "tairte4_epsilon": {
                axis: [value.real, value.imag] for axis, value in epsilon_tairte4.items()
            },
            "axis_mapping": {"x": "b", "y": "a", "z": "c=b closure"},
            "causal_fits": fits,
            "gray_law": (
                "Au passive Drude oscillator strength rho^3 on a fixed physical support; "
                "numerical topology relaxation, not a physical gray alloy"
            ),
        },
        "numerics": {
            "total_periods": cfg.total_periods,
            "phasor_periods_per_window": cfg.phasor_periods,
            "time_steps_total": total_steps,
            "period_steps": period_steps,
            "time_step_s": dt,
            "courant_factor": cfg.courant_factor,
            "checkpoint_count": cfg.checkpoints,
            "gradient_method": "checkpointed JAX reverse-mode AD through full dispersive FDTD",
            "dtype": "float32/complex64",
        },
        "results": {
            "powers_W": dict(zip(observable_names, map(float, late_all_w[:3]))),
            "component_powers_W": {
                "au_xyz": list(map(float, late_all_w[3:6])),
                "tairte4_xyz": list(map(float, late_all_w[6:9])),
            },
            "previous_window_powers_W": dict(zip(observable_names, map(float, previous_all_w[:3]))),
            "window_relative_change": dict(zip(observable_names, map(float, window_changes[:3]))),
            "closed_surface_inward_raw_power_W": p_six_raw_w,
            "closed_surface_inward_empty_air_power_W": p_six_empty_w,
            "closed_surface_inward_material_minus_empty_power_W": p_six_corrected_w,
            "closed_surface_raw_closure_relative_error": raw_closure_relative_error,
            "closed_surface_background_subtracted_closure_relative_error": closure_relative_error,
            "net_downward_power_above_structure_W": p_net_down_w,
            "net_downward_empty_air_power_W": p_net_down_empty_w,
            "gradient_l2_W_per_rho": gradient_l2,
            "gradient_sum_relative_error": gradient_sum_error,
            "max_total_strong_relative_error_finest_step": max_finest_strong,
            "max_total_gradient_l2_normalized_error_finest_step": max_finest_gradient_normalized,
            "compile_seconds": compile_seconds,
            "ad_execution_seconds": ad_seconds,
            "fd_sweep_seconds": fd_seconds,
        },
        "gates": {
            "gpu_only": True,
            "finite": finite,
            "physical_50nm_Au_and_100nm_TaIrTe4": True,
            "observable_window_changes_lt_0p5pct": settled,
            "closed_surface_closure_lt_1pct": raw_closure_relative_error < 0.01,
            "closed_surface_background_subtracted_closure_lt_1pct": closure_relative_error < 0.01,
            "gradient_component_sum_error_lt_1e-6": gradient_sum_error < 1e-6,
            "total_finest_strong_direction_error_lt_1pct": max_finest_strong < 0.01,
            "total_finest_multidirection_gradient_normalized_error_lt_1pct": (
                max_finest_gradient_normalized < 0.01
            ),
        },
        "files": {
            "directions_csv": csv_path.name,
            "plot": plot_path.name,
        },
    }
    summary_path = output_dir / "fdtdx_quasiuniform_au_tairte4_adfd_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    summary["files"]["summary_sha256"] = _sha256(summary_path)
    print(json.dumps(summary, indent=2, default=_json_default))
    if not passed:
        raise SystemExit(2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results_quasiuniform_25nm",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()
    cfg = BridgeConfig(
        total_periods=8 if args.quick else 14,
        phasor_periods=2 if args.quick else 3,
        checkpoints=4 if args.quick else 8,
    )
    run_bridge(args.output_dir, cfg, forward_only=args.forward_only)


if __name__ == "__main__":
    main()
