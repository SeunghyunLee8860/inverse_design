#!/usr/bin/env python3
"""GPU source-direction reciprocity control for FDTDX quasi-uniform grids.

This isolates source injection from Au/TaIrTe4 dispersion and topology AD.  It
checks both propagation signs for a periodic uniform plane wave and for the
finite six-PML Gaussian used by the compact bridge.  It is a numerical source
audit, not a production optical, thermal, PTE, or optimization result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx


WAVELENGTH_M = 10.0e-6
DX_M = 100.0e-9
DY_M = 100.0e-9
DZ_M = 25.0e-9
DOMAIN_CELLS = (40, 40, 160)
PML_CELLS = (8, 8, 32)
TOTAL_PERIODS = 8
AVERAGE_PERIODS = 2


def _run_case(kind: str, direction: str) -> dict[str, object]:
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
    lateral_type = "periodic" if kind == "uniform" else "pml"
    boundary = fdtdx.BoundaryConfig(
        boundary_type_minx=lateral_type,
        boundary_type_maxx=lateral_type,
        boundary_type_miny=lateral_type,
        boundary_type_maxy=lateral_type,
        thickness_grid_minx=PML_CELLS[0],
        thickness_grid_maxx=PML_CELLS[0],
        thickness_grid_miny=PML_CELLS[1],
        thickness_grid_maxy=PML_CELLS[1],
        thickness_grid_minz=PML_CELLS[2],
        thickness_grid_maxz=PML_CELLS[2],
    )
    boundary_objects, boundary_constraints = fdtdx.boundary_objects_from_config(boundary, volume)
    objects.extend(boundary_objects.values())
    constraints.extend(boundary_constraints)

    shared = {
        "name": "source",
        "partial_grid_shape": ((None, None, 1) if kind == "uniform" else (20, 20, 1)),
        "wave_character": fdtdx.WaveCharacter(wavelength=WAVELENGTH_M),
        "direction": direction,
        "fixed_E_polarization_vector": (1.0, 0.0, 0.0),
    }
    if kind == "uniform":
        source = fdtdx.UniformPlaneSource(**shared)
        constraints.append(source.same_size(volume, axes=(0, 1)))
    else:
        source = fdtdx.GaussianPlaneSource(radius=1.0e-6, std=0.42, **shared)
    constraints.append(source.place_at_center(volume, axes=(0, 1)))
    source_z_m = -1.0e-6 if direction == "+" else 1.0e-6
    detector_z_m = -0.5e-6 if direction == "+" else 0.5e-6
    constraints.append(
        source.place_at_center(volume, axes=(2,), margins=(source_z_m,))
    )
    objects.append(source)

    detector_shape = (None, None, 1) if kind == "uniform" else (20, 20, 1)
    detector = fdtdx.PoyntingFluxDetector(
        name="downstream_flux",
        partial_grid_shape=detector_shape,
        direction=direction,
        plot=False,
    )
    if kind == "uniform":
        constraints.append(detector.same_size(volume, axes=(0, 1)))
    constraints.extend(
        [
            detector.place_at_center(volume, axes=(0, 1)),
            detector.place_at_center(volume, axes=(2,), margins=(detector_z_m,)),
        ]
    )
    objects.append(detector)

    late_switch = fdtdx.OnOffSwitch(
        start_time=(TOTAL_PERIODS - AVERAGE_PERIODS) * period_s,
    )
    closed_td = fdtdx.ClosedSurfacePoyntingFluxDetector(
        name="closed_td",
        partial_grid_shape=(18, 18, 22),
        orientation="inward",
        switch=late_switch,
    )
    closed_fd = fdtdx.ClosedSurfacePhasorPoyntingFluxDetector(
        name="closed_fd",
        partial_grid_shape=(18, 18, 22),
        orientation="inward",
        wave_characters=(fdtdx.WaveCharacter(wavelength=WAVELENGTH_M),),
        switch=late_switch,
        dtype=jnp.complex64,
        exact_interpolation=True,
    )
    for closed in (closed_td, closed_fd):
        constraints.extend(
            [
                closed.place_at_center(volume, axes=(0, 1)),
                closed.place_at_center(volume, axes=(2,), margins=(25.0e-9,)),
            ]
        )
        objects.append(closed)

    # A larger source-free control volume reaches the lateral PML interface
    # and leaves ten z cells between its upstream face and the TFSF plane.
    # This tests whether the compact box misses discrete transverse flux.
    large_z_margin_m = 125.0e-9 if direction == "+" else -125.0e-9
    closed_td_large = fdtdx.ClosedSurfacePoyntingFluxDetector(
        name="closed_td_large",
        partial_grid_shape=(24, 24, 70),
        orientation="inward",
        switch=late_switch,
    )
    closed_fd_large = fdtdx.ClosedSurfacePhasorPoyntingFluxDetector(
        name="closed_fd_large",
        partial_grid_shape=(24, 24, 70),
        orientation="inward",
        wave_characters=(fdtdx.WaveCharacter(wavelength=WAVELENGTH_M),),
        switch=late_switch,
        dtype=jnp.complex64,
        exact_interpolation=True,
    )
    for closed in (closed_td_large, closed_fd_large):
        constraints.extend(
            [
                closed.place_at_center(volume, axes=(0, 1)),
                closed.place_at_center(volume, axes=(2,), margins=(large_z_margin_m,)),
            ]
        )
        objects.append(closed)

    key = jax.random.PRNGKey(20260821)
    placed, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    arrays, placed, _ = fdtdx.apply_params(arrays, placed, params, key)
    _, output = fdtdx.run_fdtd(
        arrays=arrays,
        objects=placed,
        config=config,
        key=key,
        show_progress=False,
    )
    flux = np.asarray(output.detector_states["downstream_flux"]["poynting_flux"][:, 0])
    steps_per_period = round(period_s / config.time_step_duration)
    average = flux[-AVERAGE_PERIODS * steps_per_period :]
    closed_td_values = np.asarray(output.detector_states["closed_td"]["poynting_flux"][:, 0])
    closed_td_mean = float(np.mean(closed_td_values))
    closed_fd_value = float(
        placed["closed_fd"].compute_net_flux(output.detector_states["closed_fd"])[0]
    )
    closed_td_large_values = np.asarray(
        output.detector_states["closed_td_large"]["poynting_flux"][:, 0]
    )
    closed_td_large_mean = float(np.mean(closed_td_large_values))
    closed_fd_large_value = float(
        placed["closed_fd_large"].compute_net_flux(output.detector_states["closed_fd_large"])[0]
    )
    return {
        "kind": kind,
        "direction": direction,
        "mean_downstream_power_W": float(np.mean(average)),
        "rms_downstream_power_W": float(np.sqrt(np.mean(average**2))),
        "last_window_relative_std": float(np.std(average) / max(abs(np.mean(average)), 1e-300)),
        "closed_surface_time_domain_mean_inward_W": closed_td_mean,
        "closed_surface_phasor_inward_W": closed_fd_value,
        "closed_surface_phasor_vs_time_relative_error": abs(closed_fd_value - closed_td_mean)
        / max(abs(closed_td_mean), abs(closed_fd_value), 1e-300),
        "large_closed_surface_time_domain_mean_inward_W": closed_td_large_mean,
        "large_closed_surface_phasor_inward_W": closed_fd_large_value,
        "large_closed_surface_residual_over_downstream_power": abs(closed_fd_large_value)
        / max(abs(float(np.mean(average))), 1e-300),
        "time_steps_total": config.time_steps_total,
        "steps_per_period": steps_per_period,
        "source_slice": [[int(s.start), int(s.stop)] for s in placed["source"].grid_slice],
        "detector_slice": [[int(s.start), int(s.stop)] for s in placed["downstream_flux"].grid_slice],
        "closed_surface_slice": [[int(s.start), int(s.stop)] for s in placed["closed_fd"].grid_slice],
        "large_closed_surface_slice": [
            [int(s.start), int(s.stop)] for s in placed["closed_fd_large"].grid_slice
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    devices = jax.devices()
    if not devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"GPU-only contract violated: {devices}")
    cases = [_run_case(kind, direction) for kind in ("uniform", "gaussian") for direction in ("+", "-")]
    powers = {(case["kind"], case["direction"]): case["mean_downstream_power_W"] for case in cases}
    ratios = {
        kind: abs(powers[(kind, "-")]) / max(abs(powers[(kind, "+")]), 1e-300)
        for kind in ("uniform", "gaussian")
    }
    passed = all(0.95 < value < 1.05 for value in ratios.values())
    summary = {
        "status": (
            "VALIDATED_FDTDX_QUASIUNIFORM_SOURCE_DIRECTION_RECIPROCITY"
            if passed
            else "FAILED_FDTDX_QUASIUNIFORM_SOURCE_DIRECTION_RECIPROCITY"
        ),
        "scope": "source-only GPU numerical control; not production optical/thermal/PTE/optimization",
        "fdtdx_import_path": fdtdx.__file__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "grid_cell_size_m_xyz": [DX_M, DY_M, DZ_M],
        "cases": cases,
        "minus_over_plus_power_magnitude_ratio": ratios,
        "gate": "each -z/+z downstream-power magnitude ratio within 5% of unity",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "fdtdx_quasiuniform_source_direction_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
