#!/usr/bin/env python3
"""Planar 1D control for the 10-um SiO2/Si FDTDX loss identity."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx


HERE = Path(__file__).resolve().parent
STAGE49 = HERE / "49_validate_fdtdx_lumerical_binary_endpoints.py"


def load_stage49():
    spec = importlib.util.spec_from_file_location("stage49_planar_control", STAGE49)
    if spec is None or spec.loader is None:
        raise ImportError(STAGE49)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def relative(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1.0e-300)


def run(
    output: Path,
    *,
    sigma_scale: float = 1.0,
    total_periods: int = 16,
    window_periods: int = 2,
) -> dict[str, object]:
    devices = jax.devices()
    if not devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"GPU-only contract violated: {devices}")
    stage49 = load_stage49()
    _, _, z_edges = stage49._grid_edges(include_substrate=True)
    lateral_edges = np.linspace(-150.0e-9, 150.0e-9, 4)
    grid = fdtdx.RectilinearGrid.custom(
        x_edges=lateral_edges,
        y_edges=lateral_edges,
        z_edges=z_edges,
    )
    wavelength = stage49.WAVELENGTH_M
    period = wavelength / fdtdx.constants.c
    config = fdtdx.SimulationConfig(
        grid=grid,
        time=total_periods * period,
        dtype=jnp.float32,
        backend="gpu",
        courant_factor=0.5,
    )
    epsilon_sio2, epsilon_si, provenance = stage49._load_substrate_contract(
        stage49.SUBSTRATE_MATERIAL_JSON
    )
    omega = 2 * np.pi * fdtdx.constants.c / wavelength
    sigma_requested = omega * fdtdx.constants.eps0 * epsilon_sio2.imag
    sigma = sigma_scale * sigma_requested

    objects: list[object] = []
    constraints: list[object] = []
    volume = fdtdx.SimulationVolume(
        name="volume",
        partial_grid_shape=grid.shape,
        material=fdtdx.Material(permittivity=1.0),
    )
    objects.append(volume)
    boundaries, boundary_constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(
            thickness=8,
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

    silicon = fdtdx.UniformMaterialObject(
        name="silicon",
        partial_grid_shape=(None, None, 39),
        material=fdtdx.Material(permittivity=float(epsilon_si.real)),
    )
    sio2 = fdtdx.UniformMaterialObject(
        name="sio2",
        partial_grid_shape=(None, None, 19),
        material=fdtdx.Material(
            permittivity=float(epsilon_sio2.real),
            electric_conductivity=float(sigma),
        ),
    )
    constraints.extend(
        [
            silicon.same_size(volume, axes=(0, 1)),
            silicon.place_relative_to(
                volume,
                axes=(2,),
                own_positions=(-1,),
                other_positions=(-1,),
            ),
            sio2.same_size(volume, axes=(0, 1)),
            sio2.place_above(silicon),
        ]
    )
    objects.extend((silicon, sio2))

    wave = fdtdx.WaveCharacter(wavelength=wavelength)
    source = fdtdx.UniformPlaneSource(
        name="source",
        partial_grid_shape=(None, None, 1),
        wave_character=wave,
        direction="-",
        fixed_E_polarization_vector=(1, 0, 0),
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(0, 1)),
            source.place_relative_to(
                volume,
                axes=(2,),
                own_positions=(-1,),
                other_positions=(-1,),
                margins=(float(z_edges[73] - z_edges[0]),),
            ),
        ]
    )
    objects.append(source)

    switch = fdtdx.OnOffSwitch(
        start_time=(total_periods - window_periods) * period
    )
    field = fdtdx.PhasorDetector(
        name="sio2_field",
        partial_grid_shape=(None, None, 19),
        wave_characters=(wave,),
        components=("Ex", "Ey", "Ez"),
        switch=switch,
        exact_interpolation=False,
        plot=False,
    )
    constraints.append(field.same_position(sio2))
    objects.append(field)
    phasor_box = fdtdx.ClosedSurfacePhasorPoyntingFluxDetector(
        name="phasor_box",
        partial_grid_shape=(None, None, 33),
        axes=(2,),
        wave_characters=(wave,),
        orientation="inward",
        switch=switch,
    )
    time_box = fdtdx.ClosedSurfacePoyntingFluxDetector(
        name="time_box",
        partial_grid_shape=(None, None, 33),
        axes=(2,),
        orientation="inward",
        switch=switch,
    )
    for detector in (phasor_box, time_box):
        constraints.extend(
            [
                detector.same_size(volume, axes=(0, 1)),
                detector.place_relative_to(
                    volume,
                    axes=(2,),
                    own_positions=(-1,),
                    other_positions=(-1,),
                    margins=(float(z_edges[37] - z_edges[0]),),
                ),
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
    sio2_slice = placed["sio2"].grid_slice
    volume_weights = stage49._electric_yee_dual_volumes(
        config.resolved_grid, sio2_slice
    )

    start = time.perf_counter()
    out = jax.jit(
        lambda initial: fdtdx.run_fdtd(
            initial, placed, config, key, show_progress=False
        )[1]
    ).lower(arrays).compile()(arrays)
    jax.block_until_ready(out.detector_states["sio2_field"]["phasor"])
    runtime = time.perf_counter() - start

    eta0 = float(fdtdx.constants.eta0)
    e = out.detector_states["sio2_field"]["phasor"][0, 0]
    p_q = float(0.5 * sigma * eta0**2 * jnp.sum(jnp.abs(e) ** 2 * volume_weights))
    p_phasor = float(
        eta0
        * placed["phasor_box"].compute_net_flux(out.detector_states["phasor_box"])[0]
    )
    p_time = float(
        eta0 * jnp.mean(out.detector_states["time_box"]["poynting_flux"][:, 0])
    )
    result = {
        "status": (
            "DIAGNOSED_FDTDX_PLANAR_SUBSTRATE_LOSSLESS_RESIDUAL"
            if sigma_scale == 0.0
            else (
                "VALIDATED_FDTDX_PLANAR_SUBSTRATE_LOSS_IDENTITY"
                if relative(p_q, p_phasor) < 0.01
                else "FAILED_FDTDX_PLANAR_SUBSTRATE_LOSS_IDENTITY"
            )
        ),
        "scope": "periodic x/y planar 285-nm-SiO2/lossless-Si numerical control only",
        "software": {
            "fdtdx_import_path": fdtdx.__file__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "material_contract": {
            "epsilon_sio2": [epsilon_sio2.real, epsilon_sio2.imag],
            "sigma_sio2_requested_S_m": sigma_requested,
            "sigma_scale": sigma_scale,
            "sigma_sio2_applied_S_m": sigma,
            "epsilon_si": [epsilon_si.real, epsilon_si.imag],
            "provenance": provenance,
        },
        "numerics": {
            "total_periods": total_periods,
            "window_periods": window_periods,
            "time_steps_total": config.time_steps_total,
        },
        "placement": {
            "grid_shape": list(config.resolved_grid.shape),
            "sio2_slice": [[part.start, part.stop] for part in sio2_slice],
            "source_slice": [
                [part.start, part.stop] for part in placed["source"].grid_slice
            ],
            "box_slice": [
                [part.start, part.stop] for part in placed["phasor_box"].grid_slice
            ],
        },
        "power_W": {
            "native_E_Joule": p_q,
            "closed_phasor_inward": p_phasor,
            "closed_time_domain_inward": p_time,
        },
        "relative_errors": {
            "Joule_vs_phasor": relative(p_q, p_phasor),
            "Joule_vs_time_domain": relative(p_q, p_time),
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
        default=HERE
        / "results_fdtdx_planar_substrate_loss"
        / "fdtdx_planar_substrate_loss.json",
    )
    parser.add_argument("--sigma-scale", type=float, default=1.0)
    parser.add_argument("--total-periods", type=int, default=16)
    parser.add_argument("--window-periods", type=int, default=2)
    args = parser.parse_args()
    if args.window_periods <= 0 or args.window_periods >= args.total_periods:
        raise ValueError("window-periods must be positive and less than total-periods")
    result = run(
        args.output,
        sigma_scale=args.sigma_scale,
        total_periods=args.total_periods,
        window_periods=args.window_periods,
    )
    raise SystemExit(
        0
        if result["status"].startswith(("VALIDATED", "DIAGNOSED"))
        else 2
    )


if __name__ == "__main__":
    main()
