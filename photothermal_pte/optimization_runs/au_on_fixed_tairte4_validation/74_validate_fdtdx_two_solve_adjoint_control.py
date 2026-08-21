#!/usr/bin/env python3
"""Validate a checkpoint-free two-solve FDTDX adjoint on a small Au control.

The test uses the same finite-difference ADE material closures and ``rho**3``
Au-strength law as the frozen checkpointed control, but it never differentiates
through the time loop.  It performs one settled forward solve, one distributed-
current adjoint solve, and independent central-FD forward solves.  Failure is
reported fail-closed; no empirical gradient normalization is fitted.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
STAGE41_PATH = HERE / "41_validate_au_on_fixed_tairte4_optical_adfd.py"
TWO_SOLVE_PATH = HERE / "fdtdx_two_solve_adjoint.py"
OUTPUT_DEFAULT = HERE / "results_fdtdx_two_solve_adjoint_control"
POWER_SCALE_W = 1.0e-12


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _replace_named(objects, name: str, replacement):
    result = objects.copy()
    values = list(result.object_list)
    values[result.index(name)] = replacement
    return result.aset("object_list", values)


def _complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def run(
    output_dir: Path,
    total_periods: int,
    window_periods: int,
    objective_mode: str,
) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    import fdtdx

    stage41 = _load_module("stage41_two_solve_control", STAGE41_PATH)
    two_solve = _load_module("fdtdx_two_solve_adjoint", TWO_SOLVE_PATH)

    devices = jax.devices()
    if not devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"GPU-only contract violated: {devices}")
    if total_periods < 2 * window_periods + 4:
        raise ValueError("Need startup plus two disjoint phasor windows")

    wavelength_m = stage41.WAVELENGTH_M
    omega = 2.0 * math.pi * stage41.C0_M_PER_S / wavelength_m
    period_s = wavelength_m / stage41.C0_M_PER_S
    resolution_m = 100.0e-9
    domain_cells = 32
    pml_cells = 6
    design_cells = 8
    design_z_cells = 2
    flake_cells = 12
    flake_z_cells = 2
    courant_factor = 0.25

    config = fdtdx.SimulationConfig(
        grid=fdtdx.QuasiUniformGrid(
            dx=resolution_m,
            dy=resolution_m,
            dz=resolution_m,
        ),
        time=total_periods * period_s,
        dtype=jnp.float32,
        courant_factor=courant_factor,
        backend="gpu",
        gradient_config=None,
    )
    dt = config.time_step_duration
    period_steps = int(round(period_s / dt))
    total_steps = config.time_steps_total
    window_steps = window_periods * period_steps
    previous_steps = list(range(total_steps - 2 * window_steps, total_steps - window_steps))
    late_steps = list(range(total_steps - window_steps, total_steps))

    epsilon_au = complex(stage41.AU_N, stage41.AU_K) ** 2
    epsilon_ta = stage41._load_tairte4_epsilon()
    fits = {
        "au": stage41._drude_fit(epsilon_au, omega, dt),
        "a": stage41._drude_fit(epsilon_ta["a"], omega, dt),
        "b": stage41._lorentz_fit(epsilon_ta["b"], omega, dt),
    }
    fits["c"] = dict(fits["b"])
    coefficients = {
        name: stage41._coefficient_triplet(fit, dt) for name, fit in fits.items()
    }

    objects: list[object] = []
    constraints: list[object] = []
    domain_m = domain_cells * resolution_m
    volume = fdtdx.SimulationVolume(
        name="air_volume",
        partial_grid_shape=(domain_cells,) * 3,
        material=fdtdx.Material(permittivity=1.0),
    )
    objects.append(volume)
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=pml_cells, boundary_type="pml"
    )
    bounds, bound_constraints = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(bound_constraints)
    objects.extend(bounds.values())

    au_dispersion = fdtdx.DispersionModel(
        poles=(
            fdtdx.DrudePole(
                plasma_frequency=fits["au"]["omega_p_rad_s"],
                damping=fits["au"]["gamma_rad_s"],
            ),
        )
    )
    ta_placeholder = fdtdx.DispersionModel(
        poles=(
            fdtdx.DrudePole(
                plasma_frequency=fits["a"]["omega_p_rad_s"],
                damping=fits["a"]["gamma_rad_s"],
            ),
        )
    )
    flake = fdtdx.UniformMaterialObject(
        name="fixed_tairte4",
        partial_grid_shape=(flake_cells, flake_cells, flake_z_cells),
        material=fdtdx.Material(permittivity=1.0, dispersion=ta_placeholder),
    )
    design = fdtdx.UniformMaterialObject(
        name="au_nanostructure_design",
        partial_grid_shape=(design_cells, design_cells, design_z_cells),
        material=fdtdx.Material(permittivity=1.0, dispersion=au_dispersion),
    )
    interface_z = domain_cells // 2
    constraints.extend(
        [
            flake.place_at_center(volume, axes=(0, 1)),
            flake.set_grid_coordinates(
                axes=(2,), sides=("-",), coordinates=(interface_z - flake_z_cells,)
            ),
            design.place_at_center(volume, axes=(0, 1)),
            design.set_grid_coordinates(
                axes=(2,), sides=("-",), coordinates=(interface_z,)
            ),
        ]
    )
    objects.extend([flake, design])

    source_span = flake_cells + 8
    illumination = fdtdx.GaussianPlaneSource(
        name="gaussian_source",
        partial_grid_shape=(source_span, source_span, 1),
        fixed_E_polarization_vector=(1.0, 0.0, 0.0),
        wave_character=fdtdx.WaveCharacter(wavelength=wavelength_m),
        radius=0.5 * source_span * resolution_m,
        std=0.42,
        direction="-",
    )
    constraints.extend(
        [
            illumination.place_at_center(volume, axes=(0, 1)),
            illumination.set_grid_coordinates(
                axes=(2,), sides=("-",), coordinates=(domain_cells - pml_cells - 3,)
            ),
        ]
    )
    objects.append(illumination)

    adjoint_shape = (
        flake_cells,
        flake_cells,
        flake_z_cells + design_z_cells,
    )
    adjoint_source = two_solve.DistributedElectricCurrentSource(
        name="distributed_adjoint_source",
        partial_grid_shape=adjoint_shape,
        wave_character=fdtdx.WaveCharacter(wavelength=wavelength_m),
        temporal_profile=fdtdx.SingleFrequencyProfile(
            phase_shift=0.0,
            num_startup_periods=4,
        ),
        complex_profile=jnp.zeros((3, *adjoint_shape), dtype=jnp.complex64),
        static_amplitude_factor=0.0,
    )
    constraints.extend(
        [
            adjoint_source.place_at_center(volume, axes=(0, 1)),
            adjoint_source.set_grid_coordinates(
                axes=(2,),
                sides=("-",),
                coordinates=(interface_z - flake_z_cells,),
            ),
        ]
    )
    objects.append(adjoint_source)

    wave = fdtdx.WaveCharacter(wavelength=wavelength_m)
    for material_name, target in (("au", design), ("tairte4", flake)):
        for window_name, steps in (("previous", previous_steps), ("late", late_steps)):
            detector = fdtdx.PhasorDetector(
                name=f"{material_name}_{window_name}",
                partial_grid_shape=target.partial_grid_shape,
                wave_characters=(wave,),
                components=("Ex", "Ey", "Ez"),
                dtype=jnp.complex64,
                switch=fdtdx.OnOffSwitch(fixed_on_time_steps=steps),
                exact_interpolation=False,
                plot=False,
            )
            constraints.append(detector.same_position(target))
            objects.append(detector)

    key = jax.random.PRNGKey(20260821)
    placed, base, _, config, _ = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    base, placed, _ = fdtdx.apply_params(base, placed, {}, key)
    realized = config.resolved_grid
    if realized is None:
        raise RuntimeError("Missing realized grid")
    au_slice = placed["au_nanostructure_design"].grid_slice
    ta_slice = placed["fixed_tairte4"].grid_slice
    adj_slice = placed["distributed_adjoint_source"].grid_slice
    if ta_slice[2].stop != au_slice[2].start:
        raise RuntimeError("Au and TaIrTe4 are not face adjacent")
    if adj_slice[2].start != ta_slice[2].start or adj_slice[2].stop != au_slice[2].stop:
        raise RuntimeError("Adjoint source does not cover the complete lossy stack")

    spatial_shape = base.dispersive_c1.shape[-3:]
    fixed_c1 = jnp.zeros((1, 3, *spatial_shape), dtype=jnp.float32)
    fixed_c2 = jnp.zeros_like(fixed_c1)
    fixed_c3 = jnp.zeros_like(fixed_c1)
    for component, axis in enumerate(("b", "a", "c")):
        c1, c2, c3 = coefficients[axis]
        fixed_c1 = fixed_c1.at[(0, component, *ta_slice)].set(c1)
        fixed_c2 = fixed_c2.at[(0, component, *ta_slice)].set(c2)
        fixed_c3 = fixed_c3.at[(0, component, *ta_slice)].set(c3)
    au_c1, au_c2, au_c3 = coefficients["au"]
    for component in range(3):
        fixed_c1 = fixed_c1.at[(0, component, *au_slice)].set(au_c1)
        fixed_c2 = fixed_c2.at[(0, component, *au_slice)].set(au_c2)

    def arrays_for_density(rho):
        strength = jnp.broadcast_to((rho**3)[:, :, None], (design_cells, design_cells, design_z_cells))
        c3 = fixed_c3
        for component in range(3):
            c3 = c3.at[(0, component, *au_slice)].set(au_c3 * strength)
        return (
            base.reset()
            .aset("dispersive_c1", fixed_c1)
            .aset("dispersive_c2", fixed_c2)
            .aset("dispersive_c3", c3)
        )

    dvol_m3 = resolution_m**3
    prefactor = (
        0.5
        * omega
        * stage41.EPS0_F_PER_M
        * float(fdtdx.constants.eta0) ** 2
        * dvol_m3
        / POWER_SCALE_W
    )
    ta_imag = jnp.asarray(
        [epsilon_ta["b"].imag, epsilon_ta["a"].imag, epsilon_ta["c"].imag],
        dtype=jnp.float32,
    )[:, None, None, None]

    if objective_mode == "absorption":
        weight_au = jnp.ones((3, design_cells, design_cells, design_z_cells))
        weight_ta = jnp.ones((3, flake_cells, flake_cells, flake_z_cells))
    elif objective_mode == "signed_spatial":
        au_x = jnp.linspace(-1.0, 1.0, design_cells)[:, None, None]
        au_y = jnp.linspace(-1.0, 1.0, design_cells)[None, :, None]
        ta_x = jnp.linspace(-1.0, 1.0, flake_cells)[:, None, None]
        ta_y = jnp.linspace(-1.0, 1.0, flake_cells)[None, :, None]
        weight_au = jnp.stack(
            [
                jnp.broadcast_to(0.35 + 0.8 * au_x - 0.25 * au_y, (design_cells, design_cells, design_z_cells)),
                jnp.broadcast_to(-0.15 + 0.2 * au_x + 0.7 * au_y, (design_cells, design_cells, design_z_cells)),
                jnp.broadcast_to(0.1 - 0.4 * au_x + 0.3 * au_y, (design_cells, design_cells, design_z_cells)),
            ]
        )
        weight_ta = jnp.stack(
            [
                jnp.broadcast_to(-0.2 + 0.6 * ta_x + 0.15 * ta_y, (flake_cells, flake_cells, flake_z_cells)),
                jnp.broadcast_to(0.25 - 0.1 * ta_x + 0.65 * ta_y, (flake_cells, flake_cells, flake_z_cells)),
                jnp.broadcast_to(-0.05 + 0.35 * ta_x - 0.2 * ta_y, (flake_cells, flake_cells, flake_z_cells)),
            ]
        )
    else:
        raise ValueError(f"unknown objective_mode={objective_mode!r}")

    def objective_from_output(out, rho, window="late"):
        e_au = out.detector_states[f"au_{window}"]["phasor"][0, 0]
        e_ta = out.detector_states[f"tairte4_{window}"]["phasor"][0, 0]
        strength = jnp.broadcast_to((rho**3)[:, :, None], e_au.shape[1:])
        return prefactor * (
            epsilon_au.imag
            * jnp.sum(weight_au * strength[None, ...] * jnp.abs(e_au) ** 2)
            + jnp.sum(weight_ta * ta_imag * jnp.abs(e_ta) ** 2)
        )

    forward_objects = _replace_named(
        placed,
        "distributed_adjoint_source",
        placed["distributed_adjoint_source"].aset("static_amplitude_factor", 0.0),
    )

    rho_axis = jnp.linspace(-1.0, 1.0, design_cells)
    rho0 = (
        0.53
        + 0.07 * jnp.cos(math.pi * rho_axis[:, None]) * jnp.cos(0.7 * math.pi * rho_axis[None, :])
        + 0.02 * rho_axis[:, None]
    ).astype(jnp.float32)

    solve_forward = jax.jit(
        lambda rho: fdtdx.run_fdtd(
            arrays_for_density(rho), forward_objects, config, key, show_progress=False
        )[1]
    )
    compile_start = time.perf_counter()
    solve_forward_compiled = solve_forward.lower(rho0).compile()
    forward_compile_s = time.perf_counter() - compile_start
    forward_start = time.perf_counter()
    forward_out = solve_forward_compiled(rho0)
    objective0 = objective_from_output(forward_out, rho0)
    objective_previous = objective_from_output(forward_out, rho0, "previous")
    jax.block_until_ready(objective0)
    forward_s = time.perf_counter() - forward_start

    e_au = forward_out.detector_states["au_late"]["phasor"][0, 0]
    e_ta = forward_out.detector_states["tairte4_late"]["phasor"][0, 0]
    offset_xy = (flake_cells - design_cells) // 2
    e_stack = jnp.zeros((3, *adjoint_shape), dtype=jnp.complex64)
    coefficient = jnp.zeros((3, *adjoint_shape), dtype=jnp.float32)
    e_stack = e_stack.at[:, :, :, :flake_z_cells].set(e_ta)
    coefficient = coefficient.at[:, :, :, :flake_z_cells].set(
        prefactor * weight_ta * ta_imag
    )
    au_local = (
        slice(offset_xy, offset_xy + design_cells),
        slice(offset_xy, offset_xy + design_cells),
        slice(flake_z_cells, flake_z_cells + design_z_cells),
    )
    e_stack = e_stack.at[(slice(None), *au_local)].set(e_au)
    strength0 = jnp.broadcast_to((rho0**3)[:, :, None], e_au.shape[1:])
    coefficient = coefficient.at[(slice(None), *au_local)].set(
        prefactor * epsilon_au.imag * strength0[None, ...]
        * weight_au
    )
    wirtinger = two_solve.quadratic_wirtinger_derivative(e_stack, coefficient)
    adjoint_profile = two_solve.adjoint_current_from_wirtinger(
        wirtinger, config.courant_number
    )

    adjoint_arrays = arrays_for_density(rho0)
    adjoint_object = (
        placed["distributed_adjoint_source"]
        .aset("complex_profile", adjoint_profile)
        .aset("static_amplitude_factor", 1.0)
    )
    adjoint_object = adjoint_object.apply(
        key,
        adjoint_arrays.inv_permittivities,
        adjoint_arrays.inv_permeabilities,
        adjoint_arrays.dispersive_c1,
        adjoint_arrays.dispersive_c2,
        adjoint_arrays.dispersive_c3,
        adjoint_arrays.electric_conductivity,
        adjoint_arrays.dispersive_c4,
    )
    adjoint_objects = _replace_named(
        placed,
        "gaussian_source",
        placed["gaussian_source"].aset("static_amplitude_factor", 0.0),
    )
    adjoint_objects = _replace_named(
        adjoint_objects, "distributed_adjoint_source", adjoint_object
    )
    solve_adjoint = jax.jit(
        lambda: fdtdx.run_fdtd(
            adjoint_arrays, adjoint_objects, config, key, show_progress=False
        )[1]
    )
    compile_start = time.perf_counter()
    solve_adjoint_compiled = solve_adjoint.lower().compile()
    adjoint_compile_s = time.perf_counter() - compile_start
    adjoint_start = time.perf_counter()
    adjoint_out = solve_adjoint_compiled()
    e_adj_au = adjoint_out.detector_states["au_late"]["phasor"][0, 0]
    jax.block_until_ready(e_adj_au)
    adjoint_s = time.perf_counter() - adjoint_start

    d_eps = jnp.broadcast_to(
        (3.0 * rho0**2)[:, :, None] * (epsilon_au - 1.0),
        e_au.shape,
    )
    field_gradient_3d = two_solve.harmonic_material_gradient(
        e_au,
        e_adj_au,
        d_eps,
        omega,
        dt,
    )
    field_gradient = jnp.sum(field_gradient_3d, axis=(0, 3))
    direct_gradient = prefactor * epsilon_au.imag * jnp.sum(
        weight_au
        * (3.0 * rho0**2)[None, :, :, None]
        * jnp.abs(e_au) ** 2,
        axis=(0, 3),
    )
    gradient = field_gradient + direct_gradient

    rng = np.random.default_rng(20260821)
    directions = {
        "uniform": np.ones((design_cells, design_cells), dtype=np.float32),
        "fixed_seed_random": rng.standard_normal((design_cells, design_cells)).astype(np.float32),
    }
    for name in directions:
        directions[name] /= np.linalg.norm(directions[name])

    rows: list[dict[str, object]] = []
    fd_start = time.perf_counter()
    for direction_name, direction_np in directions.items():
        direction = jnp.asarray(direction_np)
        ad = float(jnp.sum(gradient * direction))
        ad_field = float(jnp.sum(field_gradient * direction))
        ad_direct = float(jnp.sum(direct_gradient * direction))
        for h in (0.01, 0.005):
            plus_out = solve_forward_compiled(rho0 + h * direction)
            minus_out = solve_forward_compiled(rho0 - h * direction)
            plus = float(objective_from_output(plus_out, rho0 + h * direction))
            minus = float(objective_from_output(minus_out, rho0 - h * direction))
            fd = (plus - minus) / (2.0 * h)
            rows.append(
                {
                    "direction": direction_name,
                    "h": h,
                    "adjoint_scaled": ad,
                    "adjoint_field_scaled": ad_field,
                    "direct_loss_scaled": ad_direct,
                    "central_fd_scaled": fd,
                    "relative_error": abs(ad - fd) / max(abs(fd), 1.0e-30),
                    "symmetric_normalized_error": abs(ad - fd)
                    / max(abs(ad) + abs(fd), 1.0e-30),
                    "plus_scaled": plus,
                    "minus_scaled": minus,
                }
            )
    jax.block_until_ready(plus_out.detector_states["au_late"]["phasor"])
    fd_s = time.perf_counter() - fd_start

    window_change = abs(float(objective0) - float(objective_previous)) / max(
        abs(float(objective0)), 1.0e-30
    )
    finest = [row for row in rows if float(row["h"]) == 0.005]
    max_error = max(float(row["relative_error"]) for row in finest)
    finite = bool(
        np.isfinite(float(objective0))
        and np.all(np.isfinite(np.asarray(gradient)))
        and all(np.isfinite(float(row["central_fd_scaled"])) for row in rows)
    )
    passed = finite and window_change < 0.005 and max_error < 0.01
    status = (
        "VALIDATED_FDTDX_CHECKPOINT_FREE_TWO_SOLVE_ADJOINT_CONTROL"
        if passed
        else "FAILED_FDTDX_CHECKPOINT_FREE_TWO_SOLVE_ADJOINT_CONTROL"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fdtdx_two_solve_adjoint_directions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for axis, image, title in (
        (axes[0, 0], np.asarray(rho0), "baseline Au density"),
        (axes[0, 1], np.asarray(field_gradient), "field-mediated gradient"),
        (axes[0, 2], np.asarray(direct_gradient), "direct-loss gradient"),
        (axes[1, 0], np.asarray(gradient), "total two-solve gradient"),
    ):
        cmap = "gray" if "density" in title else "coolwarm"
        im = axis.imshow(image.T, origin="lower", cmap=cmap)
        axis.set_title(title)
        fig.colorbar(im, ax=axis)
    for direction_name in directions:
        subset = [row for row in rows if row["direction"] == direction_name]
        axes[1, 1].plot(
            [float(row["central_fd_scaled"]) for row in subset],
            [float(row["adjoint_scaled"]) for row in subset],
            "o",
            label=direction_name,
        )
    limits = axes[1, 1].get_xlim()
    low = min(limits[0], axes[1, 1].get_ylim()[0])
    high = max(limits[1], axes[1, 1].get_ylim()[1])
    axes[1, 1].plot([low, high], [low, high], "k--", label="AD=FD")
    axes[1, 1].set_xlabel("central FD")
    axes[1, 1].set_ylabel("two-solve adjoint")
    axes[1, 1].legend()
    for direction_name in directions:
        subset = [row for row in rows if row["direction"] == direction_name]
        axes[1, 2].plot(
            [float(row["h"]) for row in subset],
            [float(row["relative_error"]) for row in subset],
            "o-",
            label=direction_name,
        )
    axes[1, 2].axhline(0.01, color="black", linestyle="--")
    axes[1, 2].set_yscale("log")
    axes[1, 2].invert_xaxis()
    axes[1, 2].set_xlabel("central-FD h")
    axes[1, 2].set_ylabel("relative error")
    axes[1, 2].legend()
    plot_path = output_dir / "fdtdx_two_solve_adjoint_control.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    summary = {
        "status": status,
        "scope": (
            "small uniform-grid dispersive Au-on-fixed-TaIrTe4 algorithmic control; "
            "not the 48 um production device"
        ),
        "method": {
            "objective_mode": objective_mode,
            "forward_solves_for_gradient": 1,
            "adjoint_solves_for_gradient": 1,
            "time_history_saved": False,
            "checkpoint_count": 0,
            "empirical_gradient_rescaling": False,
            "source": "complex distributed electric current on the native Yee component grid",
            "material_contraction": "finite-dt harmonic ADE epsilon derivative and unconjugated E_fwd*E_adj pairing",
        },
        "geometry": {
            "resolution_m": resolution_m,
            "domain_cells_xyz": [domain_cells] * 3,
            "pml_cells_each_face": pml_cells,
            "au_cells_xyz": [design_cells, design_cells, design_z_cells],
            "tairte4_cells_xyz": [flake_cells, flake_cells, flake_z_cells],
            "axis_mapping": {"x": "b", "y": "a", "z": "c=b"},
        },
        "materials": {
            "au_epsilon_10um": _complex_pair(epsilon_au),
            "tairte4_epsilon_10um": {
                axis: _complex_pair(value) for axis, value in epsilon_ta.items()
            },
            "au_strength_law": "rho^3",
        },
        "numerics": {
            "total_periods": total_periods,
            "phasor_window_periods": window_periods,
            "time_steps_total": total_steps,
            "period_steps": period_steps,
            "courant_factor": courant_factor,
            "gradient_config": None,
            "jax_devices": [str(device) for device in devices],
            "realized_grid_shape": list(realized.shape),
        },
        "results": {
            "objective_scaled": float(objective0),
            "objective_W": float(objective0) * POWER_SCALE_W,
            "previous_window_objective_scaled": float(objective_previous),
            "window_relative_change": window_change,
            "gradient_l2_scaled": float(jnp.linalg.norm(gradient)),
            "field_gradient_l2_scaled": float(jnp.linalg.norm(field_gradient)),
            "direct_gradient_l2_scaled": float(jnp.linalg.norm(direct_gradient)),
            "max_finest_direction_relative_error": max_error,
            "forward_compile_seconds": forward_compile_s,
            "forward_execution_seconds": forward_s,
            "adjoint_compile_seconds": adjoint_compile_s,
            "adjoint_execution_seconds": adjoint_s,
            "two_solve_execution_seconds": forward_s + adjoint_s,
            "fd_execution_seconds": fd_s,
        },
        "gates": {
            "gpu_only": True,
            "finite": finite,
            "window_change_lt_0p5pct": window_change < 0.005,
            "finest_direction_error_lt_1pct": max_error < 0.01,
        },
        "files": {
            "directions_csv": csv_path.name,
            "plot": plot_path.name,
        },
        "environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "python": sys.version,
            "jax": jax.__version__,
            "fdtdx": getattr(fdtdx, "__version__", "unknown"),
        },
    }
    summary_path = output_dir / "fdtdx_two_solve_adjoint_control_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--total-periods", type=int, default=12)
    parser.add_argument("--window-periods", type=int, default=3)
    parser.add_argument(
        "--objective-mode",
        choices=("absorption", "signed_spatial"),
        default="absorption",
    )
    args = parser.parse_args()
    run(args.output_dir, args.total_periods, args.window_periods, args.objective_mode)


if __name__ == "__main__":
    main()
