#!/usr/bin/env python3
"""Diagnose the native FDTDX Joule-loss / closed-flux identity.

This is a deliberately small, uniform-grid control.  It does not model the Au
inverse-design device.  Its only purpose is to test whether a native-Yee
electric-field phasor gives the same absorbed power as a matched closed
Poynting surface for a homogeneous conductive slab.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx


WAVELENGTH_M = 1.0e-6
GRID_M = 50.0e-9
PML_CELLS = 10
SIGMA_S_M = 300.0
EPS_REAL = 4.0
DOMAIN_XY_M = 3 * GRID_M
DOMAIN_Z_M = 4.0e-6
SOURCE_Z = PML_CELLS + 2
BOX_Z = SOURCE_Z + 8
SLAB_Z = BOX_Z + 2
SLAB_CELLS = 4
BOX_CELLS = 8


def relative(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1.0e-300)


def run(output: Path) -> dict[str, object]:
    devices = jax.devices()
    if not devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"GPU-only contract violated: {devices}")

    period = WAVELENGTH_M / fdtdx.constants.c
    switch = fdtdx.OnOffSwitch(
        start_after_periods=25,
        on_for_periods=10,
        period=period,
    )
    config = fdtdx.SimulationConfig(
        grid=fdtdx.UniformGrid(spacing=GRID_M),
        time=120.0e-15,
        dtype=jnp.float32,
        backend="gpu",
    )
    objects: list[object] = []
    constraints: list[object] = []
    volume = fdtdx.SimulationVolume(
        name="volume",
        partial_real_shape=(DOMAIN_XY_M, DOMAIN_XY_M, DOMAIN_Z_M),
    )
    objects.append(volume)
    boundaries, boundary_constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(
            thickness=PML_CELLS,
            override_types={
                "min_x": "periodic",
                "max_x": "periodic",
                "min_y": "periodic",
                "max_y": "periodic",
            },
        ),
        volume,
    )
    objects.extend(boundaries.values())
    constraints.extend(boundary_constraints)

    wave = fdtdx.WaveCharacter(wavelength=WAVELENGTH_M)
    source = fdtdx.UniformPlaneSource(
        name="source",
        partial_grid_shape=(None, None, 1),
        wave_character=wave,
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(0, 1)),
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(SOURCE_Z,)),
        ]
    )
    objects.append(source)

    slab = fdtdx.UniformMaterialObject(
        name="lossy_slab",
        partial_grid_shape=(None, None, SLAB_CELLS),
        material=fdtdx.Material(
            permittivity=EPS_REAL,
            electric_conductivity=SIGMA_S_M,
        ),
    )
    constraints.extend(
        [
            slab.same_size(volume, axes=(0, 1)),
            slab.place_at_center(volume, axes=(0, 1)),
            slab.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(SLAB_Z,)),
        ]
    )
    objects.append(slab)

    slab_field = fdtdx.PhasorDetector(
        name="slab_field",
        partial_grid_shape=(None, None, SLAB_CELLS),
        wave_characters=(wave,),
        components=("Ex", "Ey", "Ez"),
        switch=switch,
        exact_interpolation=False,
        plot=False,
    )
    constraints.append(slab_field.same_position(slab))
    objects.append(slab_field)

    phasor_box = fdtdx.ClosedSurfacePhasorPoyntingFluxDetector(
        name="phasor_box",
        partial_grid_shape=(None, None, BOX_CELLS),
        axes=(2,),
        wave_characters=(wave,),
        orientation="inward",
        switch=switch,
    )
    time_box = fdtdx.ClosedSurfacePoyntingFluxDetector(
        name="time_box",
        partial_grid_shape=(None, None, BOX_CELLS),
        axes=(2,),
        orientation="inward",
        switch=switch,
    )
    for detector in (phasor_box, time_box):
        constraints.extend(
            [
                detector.same_size(volume, axes=(0, 1)),
                detector.place_at_center(volume, axes=(0, 1)),
                detector.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(BOX_Z,)),
            ]
        )
        objects.append(detector)

    key = jax.random.PRNGKey(20260821)
    placed, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    arrays, placed, _ = fdtdx.apply_params(arrays, placed, params, key)
    slab_slice = placed["lossy_slab"].grid_slice
    volume_weights = config.resolved_grid.cell_volume(
        tuple((part.start, part.stop) for part in slab_slice)
    )

    start = time.perf_counter()
    out = jax.jit(
        lambda initial: fdtdx.run_fdtd(
            initial,
            placed,
            config,
            key,
            show_progress=False,
        )[1]
    ).lower(arrays).compile()(arrays)
    jax.block_until_ready(out.detector_states["slab_field"]["phasor"])
    runtime = time.perf_counter() - start

    eta0 = float(fdtdx.constants.eta0)
    field = out.detector_states["slab_field"]["phasor"][0, 0]
    p_joule = 0.5 * SIGMA_S_M * eta0**2 * jnp.sum(
        jnp.abs(field) ** 2 * volume_weights[None, ...]
    )
    p_phasor = eta0 * placed["phasor_box"].compute_net_flux(
        out.detector_states["phasor_box"]
    )[0]
    p_time = eta0 * jnp.mean(out.detector_states["time_box"]["poynting_flux"][:, 0])
    p_joule, p_phasor, p_time = map(float, (p_joule, p_phasor, p_time))

    result = {
        "status": (
            "VALIDATED_FDTDX_NATIVE_JOULE_FLUX_IDENTITY"
            if relative(p_joule, p_phasor) < 0.01
            else "FAILED_FDTDX_NATIVE_JOULE_FLUX_IDENTITY"
        ),
        "scope": "uniform-grid periodic lossy-slab numerical control only",
        "software": {
            "fdtdx_import_path": fdtdx.__file__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "contract": {
            "wavelength_m": WAVELENGTH_M,
            "grid_m": GRID_M,
            "epsilon_real": EPS_REAL,
            "sigma_S_m": SIGMA_S_M,
            "slab_cells": SLAB_CELLS,
            "slab_thickness_m": SLAB_CELLS * GRID_M,
            "recording_periods": 10,
        },
        "power_W": {
            "native_E_Joule": p_joule,
            "closed_phasor_inward": p_phasor,
            "closed_time_domain_inward": p_time,
        },
        "relative_errors": {
            "Joule_vs_phasor": relative(p_joule, p_phasor),
            "Joule_vs_time_domain": relative(p_joule, p_time),
            "phasor_vs_time_domain": relative(p_phasor, p_time),
        },
        "runtime_seconds_including_compile": runtime,
        "no_empirical_gain_clipping_smoothing_or_rescaling": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results_fdtdx_loss_power_identity"
        / "fdtdx_loss_power_identity.json",
    )
    args = parser.parse_args()
    result = run(args.output)
    raise SystemExit(0 if result["status"].startswith("VALIDATED") else 2)


if __name__ == "__main__":
    main()
