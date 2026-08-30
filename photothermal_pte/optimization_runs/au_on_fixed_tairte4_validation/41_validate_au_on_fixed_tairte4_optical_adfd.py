#!/usr/bin/env python3
"""Validate causal Au topology gradients above fixed anisotropic TaIrTe4.

This is the next optical gate after the isolated-Au 3-D control.  The Au is a
designable nanostructure material, not an electrode.  A 2-D density field is
extruded through a fixed Au thickness and scales a passive Drude-pole strength
as ``rho**3``.  A fixed TaIrTe4 slab immediately below the Au uses the
repository axis contract ``x=b, y=a, z=c=b``.  Each TaIrTe4 axis is represented
by a one-frequency causal ADE closure fitted exactly to ``perm_data.txt`` at
10 um; this closure is not claimed to be a measured broadband pole model.

The script differentiates three observables through the complete 3-D Maxwell
solve: Au absorption, TaIrTe4 absorption, and their sum.  Central finite
differences certify the total-power gradient.  No Lumerical CPU fallback is
allowed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


WAVELENGTH_M = 10.0e-6
WAVELENGTH_NM = WAVELENGTH_M * 1.0e9
AU_N = 12.1
AU_K = 69.2
EPS_INF = 1.0
EPS0_F_PER_M = 8.8541878128e-12
C0_M_PER_S = 299_792_458.0
POWER_SCALE_W = 1.0e-24
REPOSITORY = Path(__file__).resolve().parents[3]
PERMITTIVITY_PATH = REPOSITORY / "photothermal_pte" / "bundle" / "perm_data.txt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_tairte4_epsilon() -> dict[str, complex]:
    data = np.loadtxt(PERMITTIVITY_PATH)
    data = data[np.argsort(data[:, 0])]
    values: dict[str, complex] = {}
    for axis, column in (("a", 1), ("b", 3), ("c", 5)):
        values[axis] = complex(
            np.interp(WAVELENGTH_NM, data[:, 0], data[:, column]),
            np.interp(WAVELENGTH_NM, data[:, 0], data[:, column + 1]),
        )
    if values["c"] != values["b"]:
        raise RuntimeError("perm_data.txt no longer satisfies epsilon_c=epsilon_b")
    if any(value.imag <= 0.0 for value in values.values()):
        raise RuntimeError(f"TaIrTe4 passive-loss contract failed: {values}")
    return values


def _drude_fit(
    epsilon: complex,
    omega: float,
    dt: float,
    eps_inf: float = 1.0,
) -> dict[str, float]:
    """Fit the *discrete* ADE susceptibility exactly at the target omega."""

    chi = epsilon - eps_inf
    if not (chi.real < 0.0 and chi.imag > 0.0):
        raise ValueError(f"Passive one-pole Drude fit requires Re(chi)<0, Im(chi)>0: {chi}")
    theta = omega * dt
    omega_d_sq = (2.0 * math.sin(0.5 * theta) / dt) ** 2
    omega_s = math.sin(theta) / dt
    gamma = omega_d_sq * chi.imag / ((-chi.real) * omega_s)
    gamma_omega_s = gamma * omega_s
    omega_p_sq = (-chi.real) * (omega_d_sq**2 + gamma_omega_s**2) / omega_d_sq
    omega_p = math.sqrt(omega_p_sq)
    fitted = eps_inf + omega_p_sq / (-omega_d_sq - 1j * gamma_omega_s)
    return {
        "kind": "Drude",
        "fit_basis": "exact harmonic response of the finite-dt central-difference ADE recurrence",
        "omega_0_rad_s": 0.0,
        "gamma_rad_s": gamma,
        "coupling_sq_rad2_s2": omega_p_sq,
        "omega_p_rad_s": omega_p,
        "delta_epsilon": 0.0,
        "fit_relative_error": abs(fitted - epsilon) / abs(epsilon),
    }


def _lorentz_fit(
    epsilon: complex,
    omega: float,
    dt: float,
    eps_inf: float = 1.0,
    resonance_ratio: float = 2.0,
) -> dict[str, float]:
    """Fit one passive Lorentz pole exactly at omega.

    The resonance is an explicit numerical closure, chosen above the target
    frequency so both damping and oscillator strength remain positive.
    """

    chi = epsilon - eps_inf
    if not (chi.real > 0.0 and chi.imag > 0.0 and resonance_ratio > 1.0):
        raise ValueError(f"Passive Lorentz closure requires positive complex chi: {chi}")
    theta = omega * dt
    omega_d_sq = (2.0 * math.sin(0.5 * theta) / dt) ** 2
    omega_s = math.sin(theta) / dt
    omega_0 = resonance_ratio * omega
    detuning = omega_0 * omega_0 - omega_d_sq
    ratio = chi.imag / chi.real
    gamma = ratio * detuning / omega_s
    coupling_sq = chi.real * detuning * (1.0 + ratio * ratio)
    delta_epsilon = coupling_sq / (omega_0 * omega_0)
    fitted = eps_inf + coupling_sq / (detuning - 1j * gamma * omega_s)
    return {
        "kind": "Lorentz",
        "fit_basis": "exact harmonic response of the finite-dt central-difference ADE recurrence",
        "omega_0_rad_s": omega_0,
        "gamma_rad_s": gamma,
        "coupling_sq_rad2_s2": coupling_sq,
        "omega_p_rad_s": 0.0,
        "delta_epsilon": delta_epsilon,
        "fit_relative_error": abs(fitted - epsilon) / abs(epsilon),
    }


def _coefficient_triplet(fit: dict[str, float], dt: float) -> tuple[float, float, float]:
    gamma_dt = fit["gamma_rad_s"] * dt
    if gamma_dt >= 2.0:
        raise RuntimeError(f"ADE reverse-stability condition gamma*dt<2 failed: {gamma_dt}")
    denom = 1.0 + 0.5 * gamma_dt
    c1 = (2.0 - fit["omega_0_rad_s"] ** 2 * dt**2) / denom
    c2 = -(1.0 - 0.5 * gamma_dt) / denom
    c3 = fit["coupling_sq_rad2_s2"] * dt**2 / denom
    return c1, c2, c3


@dataclass(frozen=True)
class ControlConfig:
    resolution_m: float = 100.0e-9
    domain_cells: int = 40
    pml_cells: int = 8
    design_xy_cells: int = 10
    design_z_cells: int = 2
    flake_xy_cells: int = 14
    flake_z_cells: int = 2
    total_periods: int = 18
    phasor_periods: int = 4
    checkpoints: int = 8
    courant_factor: float = 0.25
    seed: int = 20260821


def _direction_set(nx: int, ny: int, seed: int) -> dict[str, np.ndarray]:
    x = np.linspace(-1.0, 1.0, nx)[:, None]
    y = np.linspace(-1.0, 1.0, ny)[None, :]
    edge = np.zeros((nx, ny), dtype=np.float64)
    edge[:2, :] = 1.0
    edge[-2:, :] -= 0.8
    rng = np.random.default_rng(seed)
    directions = {
        "uniform": np.ones((nx, ny), dtype=np.float64),
        "smooth_asymmetric": np.sin(0.7 * math.pi * x) * np.cos(0.55 * math.pi * y) + 0.23 * x,
        "central_localized": np.exp(-((x / 0.34) ** 2 + (y / 0.34) ** 2)),
        "design_edge_localized": edge,
        "fixed_seed_random": rng.standard_normal((nx, ny)),
    }
    for name, value in directions.items():
        directions[name] = value / np.linalg.norm(value)
    return directions


def _baseline_density(nx: int, ny: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, nx)[:, None]
    y = np.linspace(-1.0, 1.0, ny)[None, :]
    rho = 0.53 + 0.08 * np.cos(math.pi * x) * np.cos(0.7 * math.pi * y) + 0.025 * x
    if rho.min() <= 0.15 or rho.max() >= 0.85:
        raise RuntimeError("Baseline density lacks unclipped central-FD margin")
    return rho.astype(np.float32)


def _complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def run_control(output_dir: Path, cfg: ControlConfig) -> dict[str, object]:
    try:
        import jax
        import jax.numpy as jnp
        import fdtdx
        from fdtdx.fdtd.fdtd import checkpointed_fdtd
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("FDTDX/JAX import failed in the isolated GPU environment") from exc

    devices = jax.devices()
    if not devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"GPU-only contract violated: {devices}")

    omega = 2.0 * math.pi * C0_M_PER_S / WAVELENGTH_M
    period_s = WAVELENGTH_M / C0_M_PER_S
    config = fdtdx.SimulationConfig(
        resolution=cfg.resolution_m,
        time=cfg.total_periods * period_s,
        dtype=jnp.float32,
        courant_factor=cfg.courant_factor,
        backend="gpu",
        gradient_config=fdtdx.GradientConfig(method="checkpointed", num_checkpoints=cfg.checkpoints),
    )
    dt = config.time_step_duration
    epsilon_au = complex(AU_N, AU_K) ** 2
    epsilon_tairte4 = _load_tairte4_epsilon()
    fits = {
        "au": _drude_fit(epsilon_au, omega, dt),
        "a": _drude_fit(epsilon_tairte4["a"], omega, dt),
        "b": _lorentz_fit(epsilon_tairte4["b"], omega, dt),
    }
    fits["c"] = dict(fits["b"])
    coefficients = {name: _coefficient_triplet(fit, dt) for name, fit in fits.items()}
    period_steps = int(round(period_s / dt))
    phasor_steps = cfg.phasor_periods * period_steps
    total_steps = config.time_steps_total
    late_steps = list(range(total_steps - phasor_steps, total_steps))
    previous_steps = list(range(total_steps - 2 * phasor_steps, total_steps - phasor_steps))
    if previous_steps[0] < 0:
        raise RuntimeError("Simulation too short for two phasor windows")

    objects: list[object] = []
    constraints: list[object] = []
    domain_m = cfg.domain_cells * cfg.resolution_m
    volume = fdtdx.SimulationVolume(
        name="air_volume",
        partial_grid_shape=(cfg.domain_cells,) * 3,
        partial_real_shape=(domain_m,) * 3,
        material=fdtdx.Material(permittivity=1.0),
    )
    objects.append(volume)
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=cfg.pml_cells, boundary_type="pml")
    bound_dict, bound_constraints = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(bound_constraints)
    objects.extend(bound_dict.values())

    au_dispersion = fdtdx.DispersionModel(
        poles=(fdtdx.DrudePole(plasma_frequency=fits["au"]["omega_p_rad_s"], damping=fits["au"]["gamma_rad_s"]),)
    )
    tairte4_placeholder = fdtdx.DispersionModel(
        poles=(fdtdx.DrudePole(plasma_frequency=fits["a"]["omega_p_rad_s"], damping=fits["a"]["gamma_rad_s"]),)
    )
    flake = fdtdx.UniformMaterialObject(
        name="fixed_tairte4",
        partial_grid_shape=(cfg.flake_xy_cells, cfg.flake_xy_cells, cfg.flake_z_cells),
        material=fdtdx.Material(permittivity=EPS_INF, dispersion=tairte4_placeholder),
    )
    design = fdtdx.UniformMaterialObject(
        name="au_nanostructure_design",
        partial_grid_shape=(cfg.design_xy_cells, cfg.design_xy_cells, cfg.design_z_cells),
        material=fdtdx.Material(permittivity=EPS_INF, dispersion=au_dispersion),
    )
    interface_z = cfg.domain_cells // 2
    constraints.extend(
        [
            flake.place_at_center(volume, axes=(0, 1)),
            flake.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(interface_z - cfg.flake_z_cells,)),
            design.place_at_center(volume, axes=(0, 1)),
            design.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(interface_z,)),
        ]
    )
    objects.extend([flake, design])

    source_span = cfg.flake_xy_cells + 10
    source = fdtdx.GaussianPlaneSource(
        name="gaussian_source",
        partial_grid_shape=(source_span, source_span, 1),
        fixed_E_polarization_vector=(1.0, 0.0, 0.0),
        wave_character=fdtdx.WaveCharacter(wavelength=WAVELENGTH_M),
        radius=0.5 * source_span * cfg.resolution_m,
        std=0.42,
        direction="-",
    )
    source_z = cfg.domain_cells - cfg.pml_cells - 3
    constraints.extend(
        [
            source.place_at_center(volume, axes=(0, 1)),
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(source_z,)),
        ]
    )
    objects.append(source)

    wave = fdtdx.WaveCharacter(wavelength=WAVELENGTH_M)
    for material_name, target in (("au", design), ("tairte4", flake)):
        for window_name, steps in (("previous", previous_steps), ("late", late_steps)):
            detector = fdtdx.PhasorDetector(
                name=f"{material_name}_{window_name}",
                partial_grid_shape=target.partial_grid_shape,
                wave_characters=(wave,),
                components=("Ex", "Ey", "Ez"),
                dtype=jnp.complex64,
                switch=fdtdx.OnOffSwitch(fixed_on_time_steps=steps),
                exact_interpolation=True,
                plot=False,
            )
            constraints.append(detector.same_position(target))
            objects.append(detector)

    key = jax.random.PRNGKey(cfg.seed)
    placed_objects, base_arrays, _, config, _ = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    base_arrays, placed_objects, _ = fdtdx.apply_params(base_arrays, placed_objects, {}, key)
    if any(value is None for value in (base_arrays.dispersive_c1, base_arrays.dispersive_c2, base_arrays.dispersive_c3)):
        raise RuntimeError("FDTDX did not allocate ADE arrays")

    au_slice = placed_objects["au_nanostructure_design"].grid_slice
    flake_slice = placed_objects["fixed_tairte4"].grid_slice
    au_shape = placed_objects["au_nanostructure_design"].grid_shape
    flake_shape = placed_objects["fixed_tairte4"].grid_shape
    expected_au_shape = (cfg.design_xy_cells, cfg.design_xy_cells, cfg.design_z_cells)
    expected_flake_shape = (cfg.flake_xy_cells, cfg.flake_xy_cells, cfg.flake_z_cells)
    if au_shape != expected_au_shape or flake_shape != expected_flake_shape:
        raise RuntimeError(f"Realized geometry mismatch: Au={au_shape}, TaIrTe4={flake_shape}")
    if au_slice[2].start != flake_slice[2].stop:
        raise RuntimeError(f"Au/TaIrTe4 are not face-adjacent: Au={au_slice}, TaIrTe4={flake_slice}")

    spatial_shape = base_arrays.dispersive_c1.shape[-3:]
    fixed_c1 = jnp.zeros((1, 3, *spatial_shape), dtype=jnp.float32)
    fixed_c2 = jnp.zeros_like(fixed_c1)
    fixed_c3 = jnp.zeros_like(fixed_c1)
    axis_fit_by_component = ("b", "a", "c")
    for component, axis in enumerate(axis_fit_by_component):
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
        c3 = fixed_c3
        for component in range(3):
            index = (0, component, *au_slice)
            c3 = c3.at[index].set(au_c3 * strength)
        inv_c2 = jnp.where(fixed_c2 == 0.0, 0.0, 1.0 / fixed_c2)
        return (
            base_arrays.reset()
            .aset("dispersive_c1", fixed_c1)
            .aset("dispersive_c2", fixed_c2)
            .aset("dispersive_c3", c3)
            .aset("dispersive_inv_c2", inv_c2)
        )

    dvol_m3 = cfg.resolution_m**3
    au_prefactor = 0.5 * omega * EPS0_F_PER_M * epsilon_au.imag * dvol_m3 / POWER_SCALE_W
    ta_imag = jnp.asarray(
        [epsilon_tairte4["b"].imag, epsilon_tairte4["a"].imag, epsilon_tairte4["c"].imag],
        dtype=jnp.float32,
    )[:, None, None, None]
    ta_prefactor = 0.5 * omega * EPS0_F_PER_M * dvol_m3 / POWER_SCALE_W

    def powers_from_output(out_arrays, window_name: str, rho: jax.Array):
        au_e = out_arrays.detector_states[f"au_{window_name}"]["phasor"][0, 0]
        ta_e = out_arrays.detector_states[f"tairte4_{window_name}"]["phasor"][0, 0]
        strength = jnp.broadcast_to((rho**3)[:, :, None], au_shape)
        p_au = au_prefactor * jnp.sum(strength * jnp.sum(jnp.abs(au_e) ** 2, axis=0))
        p_ta = ta_prefactor * jnp.sum(ta_imag * jnp.abs(ta_e) ** 2)
        return jnp.stack((p_au, p_ta, p_au + p_ta))

    def solve_windows(rho: jax.Array):
        _, out = checkpointed_fdtd(arrays_for_density(rho), placed_objects, config, key, show_progress=False)
        return powers_from_output(out, "late", rho), powers_from_output(out, "previous", rho)

    def objective_vector(rho: jax.Array):
        late, _ = solve_windows(rho)
        return late

    def objective_total(rho: jax.Array):
        return objective_vector(rho)[2]

    rho0_np = _baseline_density(cfg.design_xy_cells, cfg.design_xy_cells)
    rho0 = jnp.asarray(rho0_np)
    compile_start = time.perf_counter()
    def vector_value_and_jacobian(rho: jax.Array):
        values, pullback = jax.vjp(objective_vector, rho)
        basis = jnp.eye(3, dtype=values.dtype)
        jacobian = jax.vmap(lambda cotangent: pullback(cotangent)[0])(basis)
        return values, jacobian

    vector_value_and_jac = jax.jit(vector_value_and_jacobian).lower(rho0).compile()
    compile_seconds = time.perf_counter() - compile_start
    run_start = time.perf_counter()
    scaled_powers, scaled_jacobian = vector_value_and_jac(rho0)
    jax.block_until_ready(scaled_jacobian)
    ad_seconds = time.perf_counter() - run_start
    powers_w = np.asarray(scaled_powers, dtype=np.float64) * POWER_SCALE_W
    gradients_w = np.asarray(scaled_jacobian, dtype=np.float64) * POWER_SCALE_W
    gradient_names = ("au", "tairte4", "total")
    gradient_l2 = {name: float(np.linalg.norm(gradients_w[idx])) for idx, name in enumerate(gradient_names)}
    gradient_sum_error = float(
        np.linalg.norm(gradients_w[2] - gradients_w[0] - gradients_w[1])
        / max(np.linalg.norm(gradients_w[2]), 1e-300)
    )

    windows_jit = jax.jit(solve_windows).lower(rho0).compile()
    late_scaled, previous_scaled = windows_jit(rho0)
    late_w = np.asarray(late_scaled, dtype=np.float64) * POWER_SCALE_W
    previous_w = np.asarray(previous_scaled, dtype=np.float64) * POWER_SCALE_W
    window_changes = np.abs(late_w - previous_w) / np.maximum(np.abs(late_w), 1e-300)

    directions = _direction_set(cfg.design_xy_cells, cfg.design_xy_cells, cfg.seed)
    steps = (0.02, 0.01, 0.005)
    objective_jit = jax.jit(objective_vector).lower(rho0).compile()
    rows: list[dict[str, object]] = []
    fd_start = time.perf_counter()
    for direction_name, direction_np in directions.items():
        direction = jnp.asarray(direction_np, dtype=rho0.dtype)
        ad_vector = np.asarray([np.vdot(gradients_w[idx], direction_np).real for idx in range(3)])
        for h in steps:
            rho_plus = rho0 + h * direction
            rho_minus = rho0 - h * direction
            if float(jnp.min(rho_minus)) <= 0.0 or float(jnp.max(rho_plus)) >= 1.0:
                raise RuntimeError(f"FD {direction_name}, h={h} leaves [0,1]")
            plus_w = np.asarray(objective_jit(rho_plus), dtype=np.float64) * POWER_SCALE_W
            minus_w = np.asarray(objective_jit(rho_minus), dtype=np.float64) * POWER_SCALE_W
            fd_vector = (plus_w - minus_w) / (2.0 * h)
            for idx, observable in enumerate(gradient_names):
                ad_w = float(ad_vector[idx])
                fd_w = float(fd_vector[idx])
                strong = max(abs(ad_w), abs(fd_w)) >= 0.01 * gradient_l2[observable]
                rows.append(
                    {
                        "observable": observable,
                        "direction": direction_name,
                        "h": h,
                        "ad_W_per_unit_direction": ad_w,
                        "fd_W_per_unit_direction": fd_w,
                        "strong_relative_error": abs(ad_w - fd_w) / max(abs(fd_w), 1e-300),
                        "symmetric_normalized_error": abs(ad_w - fd_w) / max(abs(ad_w) + abs(fd_w), 1e-300),
                        "gradient_l2_normalized_error": abs(ad_w - fd_w) / max(gradient_l2[observable], 1e-300),
                        "strong_direction": strong,
                        "power_plus_W": float(plus_w[idx]),
                        "power_minus_W": float(minus_w[idx]),
                    }
                )
    fd_seconds = time.perf_counter() - fd_start

    finest_h = min(steps)
    finest_total = [row for row in rows if row["observable"] == "total" and float(row["h"]) == finest_h]
    finest_total_strong = [row for row in finest_total if bool(row["strong_direction"])]
    max_finest_strong = max(float(row["strong_relative_error"]) for row in finest_total_strong)
    max_finest_gradient_normalized = max(float(row["gradient_l2_normalized_error"]) for row in finest_total)
    finite = bool(
        np.all(np.isfinite(powers_w))
        and np.all(np.isfinite(gradients_w))
        and all(np.isfinite(float(row["fd_W_per_unit_direction"])) for row in rows)
    )
    settled = bool(np.all(window_changes < 0.005))
    passed = (
        finite
        and settled
        and gradient_sum_error < 1e-6
        and max_finest_strong < 0.01
        and max_finest_gradient_normalized < 0.01
    )
    status = (
        "VALIDATED_AU_ON_FIXED_TAIRTE4_OPTICAL_ADFD_CONTROL"
        if passed
        else "FAILED_AU_ON_FIXED_TAIRTE4_OPTICAL_ADFD_CONTROL"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "au_on_fixed_tairte4_optical_adfd_directions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    images = [rho0_np, gradients_w[0], gradients_w[1], gradients_w[2]]
    titles = ["Au density", "dP_Au/drho", "dP_TaIrTe4/drho", "dP_total/drho"]
    cmaps = ["gray", "coolwarm", "coolwarm", "coolwarm"]
    for axis, image, title, cmap in zip(axes.flat[:4], images, titles, cmaps):
        im = axis.imshow(image.T, origin="lower", cmap=cmap, vmin=0, vmax=1) if title == "Au density" else axis.imshow(image.T, origin="lower", cmap=cmap)
        axis.set_title(title)
        axis.set_xlabel("x=b design node")
        axis.set_ylabel("y=a design node")
        fig.colorbar(im, ax=axis, label="rho" if title == "Au density" else "W per rho")
    for direction_name in directions:
        current = [row for row in rows if row["observable"] == "total" and row["direction"] == direction_name]
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
    axes[1, 2].bar(gradient_names, powers_w)
    axes[1, 2].set_title("absorbed powers")
    axes[1, 2].set_ylabel("W (control-source normalization)")
    plot_path = output_dir / "au_on_fixed_tairte4_optical_adfd.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    summary = {
        "status": status,
        "scope": "small 3-D optical algorithmic control; Au nanostructure design above fixed TaIrTe4; not an electrode model or production device",
        "material_contract": {
            "wavelength_m": WAVELENGTH_M,
            "au_n": AU_N,
            "au_k": AU_K,
            "au_epsilon": _complex_pair(epsilon_au),
            "tairte4_epsilon": {axis: _complex_pair(value) for axis, value in epsilon_tairte4.items()},
            "axis_mapping": {"x": "b", "y": "a", "z": "c=b closure"},
            "permittivity_table_path": str(PERMITTIVITY_PATH),
            "permittivity_table_sha256": _sha256(PERMITTIVITY_PATH),
            "causal_fits": fits,
            "fit_note": "one-frequency passive ADE closures exact at 10 um; not measured broadband pole fits",
            "au_density_law": "Drude pole strength rho^3 on a fixed Yee support; numerical causal relaxation, not a physical gray effective-medium claim",
        },
        "geometry": {
            "description": "fixed TaIrTe4 slab in air with a face-adjacent designable Au nanostructure layer above it",
            "resolution_m": cfg.resolution_m,
            "domain_cells_xyz": [cfg.domain_cells] * 3,
            "domain_span_m_xyz": [domain_m] * 3,
            "pml_cells_each_face": cfg.pml_cells,
            "au_design_cells_xyz": list(au_shape),
            "tairte4_cells_xyz": list(flake_shape),
            "au_slice": [[item.start, item.stop] for item in au_slice],
            "tairte4_slice": [[item.start, item.stop] for item in flake_slice],
            "source_z_index": source_z,
            "source_span_cells_xy": source_span,
            "direct_optical_contact": True,
        },
        "numerics": {
            "total_periods": cfg.total_periods,
            "phasor_periods_per_window": cfg.phasor_periods,
            "time_steps_total": total_steps,
            "period_steps": period_steps,
            "checkpoint_count": cfg.checkpoints,
            "courant_factor": cfg.courant_factor,
            "dtype": "float32/complex64",
            "gradient_method": "checkpointed JAX reverse-mode AD",
            "jax_devices": [str(device) for device in devices],
            "gamma_dt": {name: fit["gamma_rad_s"] * dt for name, fit in fits.items()},
        },
        "results": {
            "powers_W": dict(zip(gradient_names, map(float, powers_w))),
            "previous_window_powers_W": dict(zip(gradient_names, map(float, previous_w))),
            "late_window_powers_W": dict(zip(gradient_names, map(float, late_w))),
            "observable_window_relative_change": dict(zip(gradient_names, map(float, window_changes))),
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
            "all_observable_window_changes_lt_0p5pct": settled,
            "gradient_component_sum_error_lt_1e-6": gradient_sum_error < 1e-6,
            "total_finest_strong_direction_error_lt_1pct": max_finest_strong < 0.01,
            "total_finest_multidirection_gradient_normalized_error_lt_1pct": max_finest_gradient_normalized < 0.01,
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "fdtdx_version": getattr(fdtdx, "__version__", "unknown"),
            "jax_version": jax.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "files": {
            "directions_csv": csv_path.name,
            "plot": plot_path.name,
        },
    }
    summary_path = output_dir / "au_on_fixed_tairte4_optical_adfd_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["files"]["summary_sha256"] = _sha256(summary_path)
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--quick", action="store_true", help="short debugging run; never promoted")
    args = parser.parse_args()
    cfg = (
        ControlConfig(
            domain_cells=32,
            pml_cells=6,
            design_xy_cells=8,
            flake_xy_cells=12,
            total_periods=8,
            phasor_periods=2,
            checkpoints=4,
        )
        if args.quick
        else ControlConfig()
    )
    run_control(args.output_dir, cfg)


if __name__ == "__main__":
    main()
