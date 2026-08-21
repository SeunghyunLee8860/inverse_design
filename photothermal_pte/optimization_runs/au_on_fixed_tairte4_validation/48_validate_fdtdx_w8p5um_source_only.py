#!/usr/bin/env python3
"""Audit a production-width scalar Gaussian source in FDTDX on GPU.

This is an empty-air source-only control at 10 um wavelength.  It separates
the production-like w0=8.5 um beam from the deliberately compact subwavelength
Gaussian used by stage 45.  No material, thermal, PTE, adjoint, or optimization
calculation is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx


FDTDX_SOURCE = Path("/home/seunghyun/.local/fdtdx_main_src")
WAVELENGTH_M = 10.0e-6
W0_M = 8.5e-6
DX_M = 250.0e-9
DY_M = 250.0e-9
DZ_M = 100.0e-9
DOMAIN_CELLS = (160, 160, 80)  # 40 x 40 x 8 um
PML_CELLS = (8, 8, 8)
SOURCE_CELLS = (120, 120, 1)  # 30 x 30 um aperture
SOURCE_RADIUS_M = 15.0e-6
SOURCE_STD = W0_M / (math.sqrt(2.0) * SOURCE_RADIUS_M)
TOTAL_PERIODS = 8
AVERAGE_PERIODS = 2


def _slice(grid_slice: tuple[slice, slice, slice]) -> list[list[int]]:
    return [[int(value.start), int(value.stop)] for value in grid_slice]


def _fit_beam(intensity: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    weights = np.maximum(np.asarray(intensity, dtype=np.float64), 0.0)
    total = float(weights.sum())
    if total <= 0 or not np.isfinite(total):
        raise RuntimeError("Non-positive target-plane intensity")
    x_grid = x[:, None]
    y_grid = y[None, :]
    x0 = float((weights * x_grid).sum() / total)
    y0 = float((weights * y_grid).sum() / total)
    sigma_x = math.sqrt(float((weights * (x_grid - x0) ** 2).sum() / total))
    sigma_y = math.sqrt(float((weights * (y_grid - y0) ** 2).sum() / total))
    w_x = 2.0 * sigma_x
    w_y = 2.0 * sigma_y
    model_unit = np.exp(-2.0 * (((x_grid - x0) / w_x) ** 2 + ((y_grid - y0) / w_y) ** 2))
    amplitude = float(np.sum(weights * model_unit) / np.sum(model_unit**2))
    residual = float(np.linalg.norm(weights - amplitude * model_unit) / np.linalg.norm(weights))
    boundary = np.concatenate((weights[0], weights[-1], weights[:, 0], weights[:, -1]))
    return {
        "center_x_m": x0,
        "center_y_m": y0,
        "w0_x_m": w_x,
        "w0_y_m": w_y,
        "mean_w0_m": 0.5 * (w_x + w_y),
        "w0_relative_error": abs(0.5 * (w_x + w_y) - W0_M) / W0_M,
        "ellipticity": abs(w_x - w_y) / max(0.5 * (w_x + w_y), 1e-300),
        "gaussian_fit_nrmse": residual,
        "square_boundary_intensity_over_peak": float(boundary.max() / weights.max()),
    }


def run(output_dir: Path) -> dict[str, object]:
    devices = jax.devices()
    if not devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"GPU-only contract violated: {devices}")
    period_s = WAVELENGTH_M / fdtdx.constants.c
    config = fdtdx.SimulationConfig(
        grid=fdtdx.QuasiUniformGrid(dx=DX_M, dy=DY_M, dz=DZ_M),
        time=TOTAL_PERIODS * period_s,
        dtype=jnp.float32,
        courant_factor=0.5,
        backend="gpu",
        gradient_config=None,
    )
    objects: list[object] = []
    constraints: list[object] = []
    volume = fdtdx.SimulationVolume(
        name="air_volume",
        partial_grid_shape=DOMAIN_CELLS,
        material=fdtdx.Material(permittivity=1.0),
    )
    objects.append(volume)
    boundaries, boundary_constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig(
            thickness_grid_minx=PML_CELLS[0],
            thickness_grid_maxx=PML_CELLS[0],
            thickness_grid_miny=PML_CELLS[1],
            thickness_grid_maxy=PML_CELLS[1],
            thickness_grid_minz=PML_CELLS[2],
            thickness_grid_maxz=PML_CELLS[2],
        ),
        volume,
    )
    objects.extend(boundaries.values())
    constraints.extend(boundary_constraints)

    wave = fdtdx.WaveCharacter(wavelength=WAVELENGTH_M)
    source = fdtdx.GaussianPlaneSource(
        name="gaussian_source",
        partial_grid_shape=SOURCE_CELLS,
        fixed_E_polarization_vector=(1.0, 0.0, 0.0),
        wave_character=wave,
        radius=SOURCE_RADIUS_M,
        std=SOURCE_STD,
        direction="-",
    )
    constraints.extend(
        [
            source.place_at_center(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(2,), margins=(2.0e-6,)),
        ]
    )
    objects.append(source)

    late = fdtdx.OnOffSwitch(start_time=(TOTAL_PERIODS - AVERAGE_PERIODS) * period_s)
    target_field = fdtdx.PhasorDetector(
        name="target_field",
        partial_grid_shape=(144, 144, 1),
        wave_characters=(wave,),
        components=("Ex", "Ey", "Ez"),
        dtype=jnp.complex64,
        switch=late,
        exact_interpolation=True,
        plot=False,
    )
    target_flux = fdtdx.PhasorPoyntingFluxDetector(
        name="target_flux",
        partial_grid_shape=(144, 144, 1),
        wave_characters=(wave,),
        direction="-",
        dtype=jnp.complex64,
        switch=late,
        exact_interpolation=True,
    )
    for detector in (target_field, target_flux):
        constraints.extend(
            [
                detector.place_at_center(volume, axes=(0, 1)),
                detector.place_at_center(volume, axes=(2,), margins=(0.0,)),
            ]
        )
        objects.append(detector)

    closed_fd = fdtdx.ClosedSurfacePhasorPoyntingFluxDetector(
        name="closed_fd",
        partial_grid_shape=(136, 136, 30),
        wave_characters=(wave,),
        orientation="inward",
        dtype=jnp.complex64,
        switch=late,
        exact_interpolation=True,
    )
    closed_td = fdtdx.ClosedSurfacePoyntingFluxDetector(
        name="closed_td",
        partial_grid_shape=(136, 136, 30),
        orientation="inward",
        switch=late,
    )
    for detector in (closed_fd, closed_td):
        constraints.extend(
            [
                detector.place_at_center(volume, axes=(0, 1)),
                detector.place_at_center(volume, axes=(2,), margins=(0.0,)),
            ]
        )
        objects.append(detector)

    key = jax.random.PRNGKey(20260821)
    placed, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, placed, _ = fdtdx.apply_params(arrays, placed, params, key)
    start = time.perf_counter()
    run_jit = jax.jit(
        lambda initial: fdtdx.run_fdtd(
            initial, placed, config, key, show_progress=False
        )[1]
    ).lower(arrays).compile()
    compile_seconds = time.perf_counter() - start
    start = time.perf_counter()
    output = run_jit(arrays)
    execution_seconds = time.perf_counter() - start

    phasor = np.asarray(output.detector_states["target_field"]["phasor"])[0, 0, :, :, :, 0]
    component_intensity = np.abs(phasor) ** 2
    intensity = np.sum(component_intensity, axis=0)
    target_slice = placed["target_field"].grid_slice
    grid = config.resolved_grid
    if grid is None:
        raise RuntimeError("Missing realized grid")
    x = np.asarray(grid.centers(0))[target_slice[0]]
    y = np.asarray(grid.centers(1))[target_slice[1]]
    beam_primary = _fit_beam(component_intensity[0], x, y)
    beam_vector = _fit_beam(intensity, x, y)
    component_fractions = np.sum(component_intensity, axis=(1, 2)) / np.sum(intensity)
    p_inc = float(placed["target_flux"].compute_poynting_flux(output.detector_states["target_flux"])[0])
    p_closed_fd = float(placed["closed_fd"].compute_net_flux(output.detector_states["closed_fd"])[0])
    td = np.asarray(output.detector_states["closed_td"]["poynting_flux"][:, 0])
    steps_per_period = int(round(period_s / config.time_step_duration))
    p_closed_td = float(np.mean(td[-AVERAGE_PERIODS * steps_per_period :]))
    closure = abs(p_closed_fd) / max(abs(p_inc), 1e-300)
    finite = bool(
        np.isfinite(intensity).all()
        and all(np.isfinite(value) for value in (p_inc, p_closed_fd, p_closed_td))
    )
    gates = {
        "gpu_only": True,
        "finite": finite,
        "primary_Ex_mean_w0_within_5pct": beam_primary["w0_relative_error"] < 0.05,
        "primary_Ex_center_shift_lt_0p25um": math.hypot(
            beam_primary["center_x_m"], beam_primary["center_y_m"]
        )
        < 0.25e-6,
        "primary_Ex_ellipticity_lt_5pct": beam_primary["ellipticity"] < 0.05,
        "primary_Ex_boundary_intensity_lt_1pct": beam_primary[
            "square_boundary_intensity_over_peak"
        ]
        < 0.01,
        "closed_surface_residual_lt_0p5pct_incident": closure < 0.005,
    }
    passed = all(gates.values())
    output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    rendered = axes[0].imshow(
        (intensity / intensity.max()).T,
        origin="lower",
        extent=(1e6 * x[0], 1e6 * x[-1], 1e6 * y[0], 1e6 * y[-1]),
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].set_title("target-plane total |E|² / max")
    axes[0].set_xlabel("x=b (µm)")
    axes[0].set_ylabel("y=a (µm)")
    fig.colorbar(rendered, ax=axes[0])
    ix = int(np.argmin(np.abs(x - beam_primary["center_x_m"])))
    iy = int(np.argmin(np.abs(y - beam_primary["center_y_m"])))
    ex_intensity = component_intensity[0]
    analytic_x = np.exp(-2.0 * ((x - beam_primary["center_x_m"]) / W0_M) ** 2)
    analytic_y = np.exp(-2.0 * ((y - beam_primary["center_y_m"]) / W0_M) ** 2)
    axes[1].plot(1e6 * x, ex_intensity[:, iy] / ex_intensity[:, iy].max(), label="realized x-cut")
    axes[1].plot(1e6 * x, analytic_x, "--", label="requested w0=8.5 µm")
    axes[1].plot(1e6 * y, ex_intensity[ix, :] / ex_intensity[ix, :].max(), label="realized y-cut")
    axes[1].plot(1e6 * y, analytic_y, ":", label="requested y-cut")
    axes[1].set_title("primary Ex intensity linecuts")
    axes[1].set_xlabel("coordinate (µm)")
    axes[1].set_ylabel("normalized intensity")
    axes[1].legend(fontsize=8)
    axes[2].bar(("Ex", "Ey", "Ez"), component_fractions)
    axes[2].set_title("field-component intensity fractions")
    axes[2].set_ylabel("fraction")
    axes[2].text(
        0.03,
        0.97,
        f"closure / incident = {100*closure:.4f}%\n"
        f"primary mean w0 = {1e6*beam_primary['mean_w0_m']:.4f} µm\n"
        f"primary ellipticity = {100*beam_primary['ellipticity']:.3f}%",
        transform=axes[2].transAxes,
        va="top",
    )
    plot_path = output_dir / "fdtdx_w8p5um_source_only.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    summary = {
        "status": (
            "VALIDATED_FDTDX_W8P5UM_SOURCE_ONLY"
            if passed
            else "FAILED_FDTDX_W8P5UM_SOURCE_ONLY"
        ),
        "scope": "empty-air production-width source audit only; no material, thermal, PTE, adjoint, or optimization",
        "software": {
            "fdtdx_import_path": fdtdx.__file__,
            "fdtdx_source_commit": subprocess.check_output(
                ["git", "-C", str(FDTDX_SOURCE), "rev-parse", "HEAD"], text=True
            ).strip(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "jax_devices": [str(device) for device in devices],
        },
        "source": {
            "wavelength_m": WAVELENGTH_M,
            "requested_w0_m": W0_M,
            "source_radius_m": SOURCE_RADIUS_M,
            "source_std_relative_to_radius": SOURCE_STD,
            "aperture_span_m_xy": [SOURCE_CELLS[0] * DX_M, SOURCE_CELLS[1] * DY_M],
            "nominal_square_edge_intensity_over_peak": math.exp(-2.0 * (SOURCE_RADIUS_M / W0_M) ** 2),
            "polarization": "x=b",
            "direction": "-z",
        },
        "grid": {
            "cell_size_m_xyz": [DX_M, DY_M, DZ_M],
            "domain_cells_xyz": list(DOMAIN_CELLS),
            "domain_span_m_xyz": [
                DOMAIN_CELLS[0] * DX_M,
                DOMAIN_CELLS[1] * DY_M,
                DOMAIN_CELLS[2] * DZ_M,
            ],
            "pml_cells_each_face_xyz": list(PML_CELLS),
            "source_slice": _slice(placed["gaussian_source"].grid_slice),
            "target_slice": _slice(target_slice),
            "closed_surface_slice": _slice(placed["closed_fd"].grid_slice),
        },
        "results": {
            "beam_fit_primary_Ex": beam_primary,
            "beam_fit_vector_total": beam_vector,
            "field_component_intensity_fractions_xyz": list(map(float, component_fractions)),
            "incident_target_plane_power_W": p_inc,
            "closed_surface_phasor_inward_W": p_closed_fd,
            "closed_surface_time_domain_inward_W": p_closed_td,
            "closed_surface_residual_over_incident_power": closure,
            "compile_seconds": compile_seconds,
            "execution_seconds": execution_seconds,
            "time_steps_total": config.time_steps_total,
            "steps_per_period": steps_per_period,
        },
        "gates": gates,
        "files": {"plot": plot_path.name},
    }
    path = output_dir / "fdtdx_w8p5um_source_only_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.output_dir)
    return 0 if summary["status"].startswith("VALIDATED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
