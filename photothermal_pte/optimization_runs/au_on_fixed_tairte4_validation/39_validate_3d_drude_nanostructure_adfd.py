#!/usr/bin/env python3
"""Validate a fixed-grid 3-D causal-Drude Au density gradient on a GPU.

This is an algorithmic certification problem, not the production TaIrTe4
device.  It deliberately avoids a moving/conformal metal boundary.  A 2-D
density is extruded through a fixed Au thickness and scales only the passive
Drude pole strength, ``s(rho)=rho**3``.  The complete time-domain Maxwell
solve and Au absorption objective are differentiated with checkpointed JAX
reverse mode and compared with central finite differences.

The script requires fdtdx 0.6.2 in PYTHONPATH.  The repository does not vendor
that dependency or its raw runtime cache.
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
AU_N = 12.1
AU_K = 69.2
EPS_INF = 1.0
EPS0_F_PER_M = 8.8541878128e-12
C0_M_PER_S = 299_792_458.0
POWER_SCALE_W = 1.0e-24


def drude_fit_from_nk(wavelength_m: float, n: float, k: float, eps_inf: float = 1.0) -> dict[str, float]:
    """Fit one passive Drude pole exactly at one angular frequency."""

    omega = 2.0 * math.pi * C0_M_PER_S / wavelength_m
    epsilon = complex(n, k) ** 2
    chi = epsilon - eps_inf
    if not (chi.real < 0.0 and chi.imag > 0.0):
        raise ValueError(f"A passive one-pole Drude fit requires Re(chi)<0 and Im(chi)>0, got {chi}")
    gamma = omega * chi.imag / (-chi.real)
    omega_p_sq = (-chi.real) * (omega * omega + gamma * gamma)
    omega_p = math.sqrt(omega_p_sq)
    fitted = eps_inf - omega_p_sq / (omega * omega + 1j * gamma * omega)
    return {
        "omega_rad_s": omega,
        "omega_p_rad_s": omega_p,
        "gamma_rad_s": gamma,
        "epsilon_real": epsilon.real,
        "epsilon_imag": epsilon.imag,
        "fit_relative_error": abs(fitted - epsilon) / abs(epsilon),
    }


@dataclass(frozen=True)
class ControlConfig:
    resolution_m: float = 100.0e-9
    domain_cells: int = 40
    pml_cells: int = 8
    design_xy_cells: int = 10
    design_z_cells: int = 2
    total_periods: int = 18
    phasor_periods: int = 4
    checkpoints: int = 8
    courant_factor: float = 0.25
    seed: int = 20260821


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _direction_set(nx: int, ny: int, seed: int) -> dict[str, np.ndarray]:
    x = np.linspace(-1.0, 1.0, nx)[:, None]
    y = np.linspace(-1.0, 1.0, ny)[None, :]
    uniform = np.ones((nx, ny), dtype=np.float64)
    smooth = np.sin(0.7 * math.pi * x) * np.cos(0.55 * math.pi * y) + 0.23 * x
    central = np.exp(-((x / 0.34) ** 2 + (y / 0.34) ** 2))
    edge = np.zeros((nx, ny), dtype=np.float64)
    edge[:2, :] = 1.0
    edge[-2:, :] -= 0.8
    rng = np.random.default_rng(seed)
    random = rng.standard_normal((nx, ny))
    directions = {
        "uniform": uniform,
        "smooth_asymmetric": smooth,
        "central_localized": central,
        "design_edge_localized": edge,
        "fixed_seed_random": random,
    }
    for name, value in directions.items():
        norm = np.linalg.norm(value)
        if norm == 0.0:
            raise RuntimeError(f"Zero direction: {name}")
        directions[name] = value / norm
    return directions


def _baseline_density(nx: int, ny: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, nx)[:, None]
    y = np.linspace(-1.0, 1.0, ny)[None, :]
    rho = 0.53 + 0.08 * np.cos(math.pi * x) * np.cos(0.7 * math.pi * y) + 0.025 * x
    if rho.min() <= 0.15 or rho.max() >= 0.85:
        raise RuntimeError("Baseline density does not leave enough unclipped FD margin")
    return rho.astype(np.float32)


def run_control(output_dir: Path, cfg: ControlConfig) -> dict[str, object]:
    try:
        import jax
        import jax.numpy as jnp
        import fdtdx
        from fdtdx.fdtd.fdtd import checkpointed_fdtd
    except Exception as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError(
            "fdtdx/JAX import failed. Use PYTHONPATH=/home/seunghyun/.local/au_fdtdx "
            "with the EIDL-Lumapi Python environment."
        ) from exc

    devices = jax.devices()
    if not devices or any(d.platform != "gpu" for d in devices):
        raise RuntimeError(f"GPU-only contract violated; JAX devices are {devices}")

    fit = drude_fit_from_nk(WAVELENGTH_M, AU_N, AU_K, EPS_INF)
    period_s = WAVELENGTH_M / C0_M_PER_S
    sim_time_s = cfg.total_periods * period_s
    config = fdtdx.SimulationConfig(
        resolution=cfg.resolution_m,
        time=sim_time_s,
        dtype=jnp.float32,
        courant_factor=cfg.courant_factor,
        backend="gpu",
        gradient_config=fdtdx.GradientConfig(method="checkpointed", num_checkpoints=cfg.checkpoints),
    )
    period_steps = int(round(period_s / config.time_step_duration))
    phasor_steps = cfg.phasor_periods * period_steps
    total_steps = config.time_steps_total
    late_steps = list(range(max(0, total_steps - phasor_steps), total_steps))
    previous_steps = list(range(max(0, total_steps - 2 * phasor_steps), max(0, total_steps - phasor_steps)))
    if not previous_steps or not late_steps:
        raise RuntimeError("Simulation is too short for two independent phasor windows")

    objects: list[object] = []
    constraints: list[object] = []
    domain_m = cfg.domain_cells * cfg.resolution_m
    volume = fdtdx.SimulationVolume(
        name="air_volume",
        partial_grid_shape=(cfg.domain_cells, cfg.domain_cells, cfg.domain_cells),
        partial_real_shape=(domain_m, domain_m, domain_m),
        material=fdtdx.Material(permittivity=1.0),
    )
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=cfg.pml_cells, boundary_type="pml")
    bound_dict, bound_constraints = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(bound_constraints)
    objects.extend(bound_dict.values())

    drude = fdtdx.DispersionModel(
        poles=(
            fdtdx.DrudePole(
                plasma_frequency=fit["omega_p_rad_s"],
                damping=fit["gamma_rad_s"],
            ),
        )
    )
    design = fdtdx.UniformMaterialObject(
        name="au_design_box",
        partial_grid_shape=(cfg.design_xy_cells, cfg.design_xy_cells, cfg.design_z_cells),
        material=fdtdx.Material(permittivity=EPS_INF, dispersion=drude),
    )
    constraints.append(design.place_at_center(volume, axes=(0, 1, 2)))
    objects.append(design)

    source_span = cfg.design_xy_cells + 12
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
    for name, steps in (("phasor_previous", previous_steps), ("phasor_late", late_steps)):
        detector = fdtdx.PhasorDetector(
            name=name,
            partial_grid_shape=(cfg.design_xy_cells, cfg.design_xy_cells, cfg.design_z_cells),
            wave_characters=(wave,),
            components=("Ex", "Ey", "Ez"),
            dtype=jnp.complex64,
            switch=fdtdx.OnOffSwitch(fixed_on_time_steps=steps),
            exact_interpolation=True,
            plot=False,
        )
        constraints.append(detector.same_position(design))
        objects.append(detector)

    key = jax.random.PRNGKey(cfg.seed)
    placed_objects, base_arrays, _, config, _ = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    # Populate material and source arrays using the exact full-Au endpoint.
    empty_params: dict[str, object] = {}
    base_arrays, placed_objects, _ = fdtdx.apply_params(base_arrays, placed_objects, empty_params, key)
    if base_arrays.dispersive_c3 is None or base_arrays.dispersive_c1 is None or base_arrays.dispersive_c2 is None:
        raise RuntimeError("FDTDX did not allocate the expected Drude ADE arrays")

    design_slice = placed_objects["au_design_box"].grid_slice
    design_shape = placed_objects["au_design_box"].grid_shape
    if design_shape != (cfg.design_xy_cells, cfg.design_xy_cells, cfg.design_z_cells):
        raise RuntimeError(f"Unexpected realized design shape: {design_shape}")
    c3_endpoint = base_arrays.dispersive_c3
    full_strength_template = jnp.zeros(c3_endpoint.shape[-3:], dtype=jnp.float32)
    dvol_m3 = cfg.resolution_m**3
    omega = fit["omega_rad_s"]
    prefactor = 0.5 * omega * EPS0_F_PER_M * fit["epsilon_imag"] * dvol_m3 / POWER_SCALE_W

    def arrays_for_density(rho: jax.Array):
        strength_2d = rho**3
        strength_3d = jnp.broadcast_to(strength_2d[:, :, None], design_shape)
        full_strength = full_strength_template.at[design_slice].set(strength_3d)
        c3 = c3_endpoint * full_strength[None, None, ...]
        return base_arrays.reset().aset("dispersive_c3", c3)

    def scaled_power_from_detector(out_arrays, detector_name: str, rho: jax.Array):
        phasor = out_arrays.detector_states[detector_name]["phasor"][0, 0]
        e2 = jnp.sum(jnp.abs(phasor) ** 2, axis=0)
        strength = jnp.broadcast_to((rho**3)[:, :, None], design_shape)
        return prefactor * jnp.sum(strength * e2)

    def loss_and_windows(rho: jax.Array):
        arrays = arrays_for_density(rho)
        _, out = checkpointed_fdtd(arrays, placed_objects, config, key, show_progress=False)
        late = scaled_power_from_detector(out, "phasor_late", rho)
        previous = scaled_power_from_detector(out, "phasor_previous", rho)
        return late, previous

    def objective(rho: jax.Array):
        late, _ = loss_and_windows(rho)
        return late

    rho0_np = _baseline_density(cfg.design_xy_cells, cfg.design_xy_cells)
    rho0 = jnp.asarray(rho0_np)
    compile_start = time.perf_counter()
    vg = jax.jit(jax.value_and_grad(objective)).lower(rho0).compile()
    compile_seconds = time.perf_counter() - compile_start
    run_start = time.perf_counter()
    scaled_value, grad_scaled = vg(rho0)
    jax.block_until_ready(grad_scaled)
    ad_seconds = time.perf_counter() - run_start
    power_w = float(scaled_value) * POWER_SCALE_W
    gradient_w = np.asarray(grad_scaled, dtype=np.float64) * POWER_SCALE_W
    gradient_l2_w = float(np.linalg.norm(gradient_w))

    windows_jit = jax.jit(loss_and_windows).lower(rho0).compile()
    late_scaled, previous_scaled = windows_jit(rho0)
    late_scaled, previous_scaled = map(float, (late_scaled, previous_scaled))
    observable_change = abs(late_scaled - previous_scaled) / max(abs(late_scaled), 1e-30)

    directions = _direction_set(cfg.design_xy_cells, cfg.design_xy_cells, cfg.seed)
    steps = (0.02, 0.01, 0.005)
    objective_jit = jax.jit(objective).lower(rho0).compile()
    rows: list[dict[str, object]] = []
    fd_start = time.perf_counter()
    for direction_name, direction_np in directions.items():
        ad_w = float(np.vdot(gradient_w, direction_np).real)
        direction = jnp.asarray(direction_np, dtype=rho0.dtype)
        for h in steps:
            rho_plus = rho0 + h * direction
            rho_minus = rho0 - h * direction
            if float(jnp.min(rho_minus)) <= 0.0 or float(jnp.max(rho_plus)) >= 1.0:
                raise RuntimeError(f"FD direction {direction_name}, h={h} leaves [0,1]")
            plus = float(objective_jit(rho_plus)) * POWER_SCALE_W
            minus = float(objective_jit(rho_minus)) * POWER_SCALE_W
            fd_w = (plus - minus) / (2.0 * h)
            strong_rel = abs(ad_w - fd_w) / max(abs(fd_w), 1e-300)
            symmetric_norm = abs(ad_w - fd_w) / max(abs(ad_w) + abs(fd_w), 1e-300)
            gradient_normalized = abs(ad_w - fd_w) / max(gradient_l2_w, 1e-300)
            strong_direction = max(abs(ad_w), abs(fd_w)) >= 0.01 * gradient_l2_w
            rows.append(
                {
                    "direction": direction_name,
                    "h": h,
                    "ad_W_per_unit_direction": ad_w,
                    "fd_W_per_unit_direction": fd_w,
                    "strong_relative_error": strong_rel,
                    "symmetric_normalized_error": symmetric_norm,
                    "gradient_l2_normalized_error": gradient_normalized,
                    "strong_direction": strong_direction,
                    "power_plus_W": plus,
                    "power_minus_W": minus,
                }
            )
    fd_seconds = time.perf_counter() - fd_start

    strong_rows = [r for r in rows if bool(r["strong_direction"])]
    max_strong = max(float(r["strong_relative_error"]) for r in strong_rows)
    max_normalized = max(float(r["symmetric_normalized_error"]) for r in rows)
    finest_rows = [r for r in rows if float(r["h"]) == min(steps)]
    finest_strong_rows = [r for r in finest_rows if bool(r["strong_direction"])]
    max_finest_strong = max(float(r["strong_relative_error"]) for r in finest_strong_rows)
    max_finest_normalized = max(float(r["symmetric_normalized_error"]) for r in finest_rows)
    max_finest_gradient_normalized = max(float(r["gradient_l2_normalized_error"]) for r in finest_rows)
    finite = bool(
        np.isfinite(power_w)
        and np.all(np.isfinite(gradient_w))
        and all(np.isfinite(float(r["fd_W_per_unit_direction"])) for r in rows)
    )
    observable_settled = observable_change < 0.005
    passed = (
        finite
        and observable_settled
        and max_finest_strong < 0.01
        and max_finest_gradient_normalized < 0.01
    )
    status = "VALIDATED_3D_CAUSAL_DRUDE_AU_ADFD_CONTROL" if passed else "FAILED_3D_CAUSAL_DRUDE_AU_ADFD_CONTROL"

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "au_3d_drude_nanostructure_adfd_directions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    im0 = axes[0].imshow(rho0_np.T, origin="lower", cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("baseline physical density")
    axes[0].set_xlabel("x design node")
    axes[0].set_ylabel("y design node")
    fig.colorbar(im0, ax=axes[0], label="rho")
    im1 = axes[1].imshow(gradient_w.T, origin="lower", cmap="coolwarm")
    axes[1].set_title("3-D Maxwell AD gradient")
    axes[1].set_xlabel("x design node")
    axes[1].set_ylabel("y design node")
    fig.colorbar(im1, ax=axes[1], label="W per rho")
    for direction_name in directions:
        cur = [r for r in rows if r["direction"] == direction_name]
        axes[2].loglog(
            [float(r["h"]) for r in cur],
            [float(r["strong_relative_error"]) for r in cur],
            marker="o",
            label=direction_name,
        )
    axes[2].axhline(0.01, color="black", linestyle="--", label="1% gate")
    axes[2].invert_xaxis()
    axes[2].set_xlabel("central-FD step h")
    axes[2].set_ylabel("|AD-FD|/|FD|")
    axes[2].set_title("directional AD-FD convergence")
    axes[2].legend(fontsize=7)
    plot_path = output_dir / "au_3d_drude_nanostructure_adfd.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    summary = {
        "status": status,
        "scope": "small 3-D algorithmic Au nanostructure control; not production TaIrTe4 device",
        "physical_contract": {
            "wavelength_m": WAVELENGTH_M,
            "au_n": AU_N,
            "au_k": AU_K,
            "epsilon_target": [fit["epsilon_real"], fit["epsilon_imag"]],
            "epsilon_infinity": EPS_INF,
            "omega_p_rad_s": fit["omega_p_rad_s"],
            "gamma_rad_s": fit["gamma_rad_s"],
            "endpoint_fit_relative_error": fit["fit_relative_error"],
            "density_law": "Drude pole strength s(rho)=rho^3; numerical causal relaxation, not effective-medium claim",
            "geometry": "2-D rho extruded through a fixed two-cell Au thickness on a fixed Yee grid",
            "objective": "Au-only absorbed power from late-window E phasor and Im(epsilon_Au)*rho^3",
        },
        "numerics": {
            "resolution_m": cfg.resolution_m,
            "domain_cells_xyz": [cfg.domain_cells] * 3,
            "domain_span_m_xyz": [domain_m] * 3,
            "pml_cells_each_face": cfg.pml_cells,
            "design_cells_xyz": list(design_shape),
            "design_slice": [[s.start, s.stop] for s in design_slice],
            "source_z_index": source_z,
            "source_span_cells_xy": source_span,
            "total_periods": cfg.total_periods,
            "period_steps": period_steps,
            "time_steps_total": total_steps,
            "phasor_periods_per_window": cfg.phasor_periods,
            "checkpoint_count": cfg.checkpoints,
            "courant_factor": cfg.courant_factor,
            "omega_p_dt": fit["omega_p_rad_s"] * config.time_step_duration,
            "dtype": "float32/complex64",
            "gradient_method": "checkpointed reverse-mode AD",
            "jax_devices": [str(d) for d in devices],
        },
        "results": {
            "au_absorbed_power_W": power_w,
            "previous_window_power_W": previous_scaled * POWER_SCALE_W,
            "late_window_power_W": late_scaled * POWER_SCALE_W,
            "observable_window_relative_change": observable_change,
            "gradient_l2_W_per_rho": gradient_l2_w,
            "gradient_finite": finite,
            "max_strong_relative_error_all_steps": max_strong,
            "max_symmetric_normalized_error_all_steps": max_normalized,
            "max_strong_relative_error_finest_step": max_finest_strong,
            "max_symmetric_normalized_error_finest_step": max_finest_normalized,
            "max_gradient_l2_normalized_error_finest_step": max_finest_gradient_normalized,
            "near_null_directions_finest_step": [
                str(r["direction"]) for r in finest_rows if not bool(r["strong_direction"])
            ],
            "compile_seconds": compile_seconds,
            "ad_execution_seconds": ad_seconds,
            "fd_sweep_seconds": fd_seconds,
        },
        "gates": {
            "gpu_only": True,
            "finite": finite,
            "late_vs_previous_phasor_power_change_lt_0p5pct": observable_settled,
            "finest_strong_direction_error_lt_1pct": max_finest_strong < 0.01,
            "finest_multi_direction_gradient_normalized_error_lt_1pct": max_finest_gradient_normalized < 0.01,
            "note": "observable settling is reported but is not used to hide or rescale AD-FD error",
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
    summary_path = output_dir / "au_3d_drude_nanostructure_adfd_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["files"]["summary_sha256"] = _sha256(summary_path)
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--quick", action="store_true", help="shorter debugging run; never promoted")
    args = parser.parse_args()
    cfg = (
        ControlConfig(
            domain_cells=32,
            pml_cells=6,
            design_xy_cells=8,
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
