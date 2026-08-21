#!/usr/bin/env python3
"""Cross-check material-bearing FDTDX endpoints against Lumerical.

The geometry matches the already validated exact-binary Lumerical endpoint
contract: 20 x 20 x 0.1 um TaIrTe4, optional 10 x 10 x 0.05 um Au, 10 um
E||b illumination, and a production-width scalar Gaussian.  A rectilinear
grid keeps 100 nm lateral cells over the material footprint and 25 nm vertical
cells through both films while coarsening distant air.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
STAGE41 = HERE / "41_validate_au_on_fixed_tairte4_optical_adfd.py"
LUMERICAL_SUMMARY = HERE / "results/lumerical_au_on_tairte4_binary_endpoints_summary.json"
FDTDX_SOURCE = Path("/home/seunghyun/.local/fdtdx_main_src")
SUBSTRATE_MATERIAL_JSON = (
    HERE
    / "results_10um_substrate_material"
    / "10um_substrate_material_readback.json"
)
WAVELENGTH_M = 10.0e-6
W0_M = 8.5e-6
POWER_SCALE_W = 1.0e-24


def _load_stage41():
    spec = importlib.util.spec_from_file_location("stage41_endpoint", STAGE41)
    if spec is None or spec.loader is None:
        raise ImportError(STAGE41)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _piecewise_edges(points: tuple[float, ...], widths: tuple[float, ...]) -> np.ndarray:
    if len(points) != len(widths) + 1:
        raise ValueError("points/widths mismatch")
    parts: list[np.ndarray] = []
    for index, width in enumerate(widths):
        start, stop = points[index], points[index + 1]
        cells = int(round((stop - start) / width))
        if cells <= 0 or not math.isclose(start + cells * width, stop, rel_tol=0, abs_tol=1e-15):
            raise ValueError(f"Non-integral segment {start}, {stop}, {width}")
        segment = np.linspace(start, stop, cells + 1, dtype=np.float64)
        parts.append(segment if index == 0 else segment[1:])
    edges = np.concatenate(parts)
    if np.any(np.diff(edges) <= 0):
        raise RuntimeError("Non-monotonic grid")
    return edges


def _grid_edges(
    include_substrate: bool = False,
    matched_substrate_interface: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lateral = _piecewise_edges(
        (-24e-6, -12e-6, 12e-6, 24e-6),
        (500e-9, 100e-9, 500e-9),
    )
    if include_substrate and matched_substrate_interface:
        # Keep the Yee dual cells at the lossy SiO2 interfaces locally matched.
        # Ten 15-nm Si cells sit immediately below the 19 x 15-nm oxide cells;
        # TaIrTe4 uses 5 x 20-nm cells above.  The remote Si/air remain coarse.
        vertical_parts = [
            np.linspace(-8e-6, -535e-9, 39, dtype=np.float64),
            np.linspace(-535e-9, -385e-9, 11, dtype=np.float64)[1:],
            np.linspace(-385e-9, -100e-9, 20, dtype=np.float64)[1:],
            np.linspace(-100e-9, 0.0, 6, dtype=np.float64)[1:],
            np.linspace(0.0, 50e-9, 3, dtype=np.float64)[1:],
            np.linspace(50e-9, 200e-9, 7, dtype=np.float64)[1:],
            np.linspace(200e-9, 8e-6, 40, dtype=np.float64)[1:],
        ]
        vertical = np.concatenate(vertical_parts)
    elif include_substrate:
        # Exact 285-nm oxide without forcing a uniform 5-nm global CFL step:
        # 39 coarse Si cells, 19 x 15-nm SiO2 cells, 4 x 25-nm TaIrTe4
        # cells, 2 x 25-nm Au cells, then locally fine and remote air.
        vertical_parts = [
            np.linspace(-8e-6, -385e-9, 40, dtype=np.float64),
            np.linspace(-385e-9, -100e-9, 20, dtype=np.float64)[1:],
            np.linspace(-100e-9, 0.0, 5, dtype=np.float64)[1:],
            np.linspace(0.0, 50e-9, 3, dtype=np.float64)[1:],
            np.linspace(50e-9, 200e-9, 7, dtype=np.float64)[1:],
            np.linspace(200e-9, 8e-6, 40, dtype=np.float64)[1:],
        ]
        vertical = np.concatenate(vertical_parts)
    else:
        vertical = _piecewise_edges(
            (-8e-6, -0.2e-6, 0.2e-6, 8e-6),
            (200e-9, 25e-9, 200e-9),
        )
    return lateral, lateral.copy(), vertical


def _load_substrate_contract(path: Path) -> tuple[complex, complex, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") == "VALIDATED_10UM_SIO2_SI_MATERIAL_READBACK":
        sio2 = payload["materials"]["SiO2"]["readback_epsilon"]
        silicon = payload["materials"]["Si"]["readback_epsilon"]
        source = "validated Lumerical v261 readback"
    elif payload.get("status") == "BLOCKED_LUMERICAL_10UM_SI_PALIK_READBACK":
        contract = payload["offline_diagnostic_contract"]
        sio2 = contract["SiO2"]["epsilon"]
        silicon = contract["Si"]["epsilon"]
        source = "explicit offline diagnostic; Palik readback remains blocked"
    else:
        raise RuntimeError(f"Unusable substrate material contract: {payload.get('status')}")
    epsilon_sio2 = complex(sio2["real"], sio2["imag"])
    epsilon_si = complex(silicon["real"], silicon["imag"])
    if epsilon_sio2.imag <= 0.0 or epsilon_si.imag < 0.0:
        raise RuntimeError("Substrate material passivity check failed")
    return epsilon_sio2, epsilon_si, {
        "path": str(path.resolve()),
        "status": payload["status"],
        "source_class": source,
        "Palik_Si_readback_validated": payload["status"]
        == "VALIDATED_10UM_SIO2_SI_MATERIAL_READBACK",
    }


def _slice_tuple(value: tuple[slice, slice, slice]) -> list[list[int]]:
    return [[int(part.start), int(part.stop)] for part in value]


def _electric_yee_dual_volumes(grid, grid_slice: tuple[slice, slice, slice]):
    """Return component-specific physical dual volumes for Ex/Ey/Ez.

    FDTDX follows the Taflove Yee convention: Ex=(i+1/2,j,k),
    Ey=(i,j+1/2,k), Ez=(i,j,k+1/2).  A component therefore uses the ordinary
    cell width along its own axis and the edge-centred dual width
    ``(d[i-1]+d[i])/2`` along the other two axes.  Cell-centre volumes are only
    equivalent on a uniform grid and can severely under-count interface loss
    when a material begins at a coarse-to-fine rectilinear transition.
    """

    import jax.numpy as jnp

    bounds = tuple((int(part.start), int(part.stop)) for part in grid_slice)
    widths = [np.asarray(grid.cell_widths(axis), dtype=np.float64) for axis in range(3)]
    edge_dual = [
        0.5 * (np.concatenate((axis_width[:1], axis_width[:-1])) + axis_width)
        for axis_width in widths
    ]
    volumes = []
    for component in range(3):
        selected = []
        for axis in range(3):
            lower, upper = bounds[axis]
            metric = widths[axis] if axis == component else edge_dual[axis]
            selected.append(metric[lower:upper])
        volumes.append(
            selected[0][:, None, None]
            * selected[1][None, :, None]
            * selected[2][None, None, :]
        )
    return jnp.asarray(np.stack(volumes), dtype=jnp.float32)


def _relative(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_diagnostic_outputs(summary: dict[str, object], output_dir: Path) -> None:
    """Write compact cross-solver tables/plots without changing raw results."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comparisons = summary["comparisons"]
    ratios = summary["endpoint_ratio"]
    cases = summary["cases"]
    rows = []
    for name in ("au0", "au1"):
        comp = comparisons[name]
        rows.append(
            {
                "case": name,
                "fdtdx_absorbed_fraction": comp["fdtdx_absorbed_fraction"],
                "lumerical_absorbed_fraction": comp["lumerical_absorbed_fraction"],
                "cross_solver_relative_difference": comp["absorbed_fraction_relative_difference"],
                "local_Q_to_empty_subtracted_flux_closure": comp[
                    "fdtdx_empty_subtracted_closure_relative"
                ],
                "late_window_relative_change": cases[name]["P_Q_window_relative_change"],
            }
        )
    csv_path = output_dir / "fdtdx_lumerical_binary_endpoints_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    labels = ["TaIrTe4 only", "Au / TaIrTe4"]
    x = np.arange(2)
    fdtdx_fraction = [comparisons[name]["fdtdx_absorbed_fraction"] for name in ("au0", "au1")]
    lum_fraction = [comparisons[name]["lumerical_absorbed_fraction"] for name in ("au0", "au1")]
    closure_pct = [
        100.0 * comparisons[name]["fdtdx_empty_subtracted_closure_relative"] for name in ("au0", "au1")
    ]
    window_pct = [100.0 * cases[name]["P_Q_window_relative_change"] for name in ("au0", "au1")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    width = 0.36
    axes[0].bar(x - width / 2, fdtdx_fraction, width, label="FDTDX")
    axes[0].bar(x + width / 2, lum_fraction, width, label="Lumerical")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("absorbed / incident power")
    axes[0].set_title("Exact-binary endpoints")
    axes[0].legend()

    axes[1].bar(
        ["FDTDX", "Lumerical"],
        [ratios["fdtdx_au1_over_au0"], ratios["lumerical_au1_over_au0"]],
        color=("#2878B5", "#F39B38"),
    )
    axes[1].set_ylabel("P(Au/TaIrTe4) / P(TaIrTe4)")
    axes[1].set_title(f"ratio error = {100.0 * ratios['relative_difference']:.4f}%")

    axes[2].bar(x - width / 2, closure_pct, width, label="Q / six-face closure")
    axes[2].bar(x + width / 2, window_pct, width, label="late-window change")
    axes[2].axhline(0.5, color="black", ls="--", lw=1, label="0.5% gate")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("relative difference (%)")
    axes[2].set_title("FDTDX internal gates")
    axes[2].legend(fontsize=8)
    fig.suptitle("FDTDX native-Yee material loss cross-check at 10 um, E || b")
    fig.savefig(output_dir / "fdtdx_lumerical_binary_endpoints.png", dpi=180)
    plt.close(fig)


def run(
    output_dir: Path,
    *,
    audit_only: bool,
    gradient_smoke: bool = False,
    gradient_checkpoints: int = 16,
    include_adjoint_aligned: bool = False,
    include_substrate: bool = False,
    substrate_empty_only: bool = False,
    substrate_loss_representation: str = "conductivity",
    substrate_material_json: Path = SUBSTRATE_MATERIAL_JSON,
    matched_substrate_interface_grid: bool = False,
    substrate_total_periods: int | None = None,
    substrate_window_periods: int | None = None,
    gradient_direction_names: tuple[str, ...] = (
        "smooth_asymmetric",
        "fixed_seed_random",
    ),
    gradient_steps: tuple[float, ...] = (0.01, 0.005),
    gradient_reference_json: Path | None = None,
    spatial_q_export: bool = False,
    spatial_q_raw_path: Path | None = None,
    spatial_q_weight_npz: Path | None = None,
    spatial_q_weight_summary_json: Path | None = None,
    spatial_q_weight_scenario: str = "thermally_grown",
    spatial_weighted_gradient_raw_path: Path | None = None,
) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    import fdtdx

    imported = Path(fdtdx.__file__).resolve()
    expected = FDTDX_SOURCE / "src"
    if expected.resolve() not in imported.parents:
        raise RuntimeError(f"Pinned FDTDX source not imported: {imported}")
    devices = jax.devices()
    if not devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"GPU-only contract violated: {devices}")
    stage41 = _load_stage41()
    x_edges, y_edges, z_edges = _grid_edges(
        include_substrate=include_substrate,
        matched_substrate_interface=matched_substrate_interface_grid,
    )
    grid = fdtdx.RectilinearGrid.custom(x_edges=x_edges, y_edges=y_edges, z_edges=z_edges)
    period_s = WAVELENGTH_M / stage41.C0_M_PER_S
    # The high-index/lossy substrate rings down more slowly than the previous
    # air-only endpoint.  Eight periods left a 0.535% substrate-only window
    # change and ~0.9% direct Q/box mismatch, so the substrate contract uses
    # 16 periods fail-closed.  The already validated air-only default remains
    # unchanged at eight periods.
    total_periods = (
        int(substrate_total_periods)
        if include_substrate and substrate_total_periods is not None
        else (16 if include_substrate else 8)
    )
    window_periods = (
        int(substrate_window_periods)
        if include_substrate and substrate_window_periods is not None
        else 2
    )
    if total_periods <= 2 * window_periods:
        raise ValueError(
            "total_periods must exceed two analysis windows: "
            f"total={total_periods}, window={window_periods}"
        )
    config = fdtdx.SimulationConfig(
        grid=grid,
        time=total_periods * period_s,
        dtype=jnp.float32,
        courant_factor=0.5,
        backend="gpu",
        gradient_config=(
            fdtdx.GradientConfig(method="checkpointed", num_checkpoints=gradient_checkpoints)
            if gradient_smoke
            else None
        ),
    )
    dt = config.time_step_duration
    omega = 2.0 * math.pi * stage41.C0_M_PER_S / WAVELENGTH_M
    epsilon_au = complex(stage41.AU_N, stage41.AU_K) ** 2
    epsilon_ta = stage41._load_tairte4_epsilon()
    epsilon_sio2 = None
    epsilon_si = None
    substrate_provenance = None
    if include_substrate:
        epsilon_sio2, epsilon_si, substrate_provenance = _load_substrate_contract(
            substrate_material_json
        )
        if substrate_loss_representation not in ("lorentz", "conductivity"):
            raise ValueError(substrate_loss_representation)
    fits = {
        "au": stage41._drude_fit(epsilon_au, omega, dt),
        "a": stage41._drude_fit(epsilon_ta["a"], omega, dt),
        "b": stage41._lorentz_fit(epsilon_ta["b"], omega, dt),
    }
    fits["c"] = dict(fits["b"])
    if include_substrate and substrate_loss_representation == "lorentz":
        fits["sio2"] = stage41._lorentz_fit(epsilon_sio2, omega, dt)
    coeff = {name: stage41._coefficient_triplet(fit, dt) for name, fit in fits.items()}

    objects: list[object] = []
    constraints: list[object] = []
    volume = fdtdx.SimulationVolume(
        name="air_volume",
        partial_grid_shape=grid.shape,
        material=fdtdx.Material(permittivity=1.0),
    )
    objects.append(volume)
    boundaries, boundary_constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig(
            thickness_grid_minx=8,
            thickness_grid_maxx=8,
            thickness_grid_miny=8,
            thickness_grid_maxy=8,
            thickness_grid_minz=8,
            thickness_grid_maxz=8,
        ),
        volume,
    )
    objects.extend(boundaries.values())
    constraints.extend(boundary_constraints)

    au_model = fdtdx.DispersionModel(
        poles=(
            fdtdx.DrudePole(
                plasma_frequency=fits["au"]["omega_p_rad_s"],
                damping=fits["au"]["gamma_rad_s"],
            ),
        )
    )
    ta_model = fdtdx.DispersionModel(
        poles=(
            fdtdx.DrudePole(
                plasma_frequency=fits["a"]["omega_p_rad_s"],
                damping=fits["a"]["gamma_rad_s"],
            ),
        )
    )
    sio2_model = None
    silicon = None
    sio2 = None
    if include_substrate and substrate_loss_representation == "lorentz":
        sio2_model = fdtdx.DispersionModel(
            poles=(
                fdtdx.LorentzPole(
                    resonance_frequency=fits["sio2"]["omega_0_rad_s"],
                    damping=fits["sio2"]["gamma_rad_s"],
                    delta_epsilon=fits["sio2"]["delta_epsilon"],
                ),
            )
        )
    if include_substrate:
        substrate_layout = (
            {
                "silicon_cells": 48,
                "sio2_start": 48,
                "sio2_stop": 67,
                "tairte4_cells": 5,
                "near_box_start": 46,
                "near_box_stop": 80,
                "deep_box_start": 30,
                "deep_box_stop": 82,
            }
            if matched_substrate_interface_grid
            else {
                "silicon_cells": 39,
                "sio2_start": 39,
                "sio2_stop": 58,
                "tairte4_cells": 4,
                "near_box_start": 37,
                "near_box_stop": 70,
                "deep_box_start": 20,
                "deep_box_stop": 72,
            }
        )
        silicon = fdtdx.UniformMaterialObject(
            name="fixed_silicon_substrate",
            partial_grid_shape=(None, None, substrate_layout["silicon_cells"]),
            material=fdtdx.Material(permittivity=float(epsilon_si.real)),
        )
        sio2_sigma = omega * stage41.EPS0_F_PER_M * epsilon_sio2.imag
        sio2 = fdtdx.UniformMaterialObject(
            name="fixed_285nm_sio2",
            partial_grid_shape=(None, None, 19),
            material=(
                fdtdx.Material(permittivity=stage41.EPS_INF, dispersion=sio2_model)
                if substrate_loss_representation == "lorentz"
                else fdtdx.Material(
                    permittivity=float(epsilon_sio2.real),
                    electric_conductivity=float(sio2_sigma),
                )
            ),
        )
    flake = fdtdx.UniformMaterialObject(
        name="fixed_tairte4",
        partial_grid_shape=(
            200,
            200,
            substrate_layout["tairte4_cells"] if include_substrate else 4,
        ),
        material=fdtdx.Material(permittivity=stage41.EPS_INF, dispersion=ta_model),
    )
    au = fdtdx.UniformMaterialObject(
        name="exact_binary_au",
        partial_grid_shape=(100, 100, 2),
        material=fdtdx.Material(permittivity=stage41.EPS_INF, dispersion=au_model),
    )
    if include_substrate:
        constraints.extend(
            [
                silicon.place_relative_to(
                    volume,
                    axes=(2,),
                    own_positions=(-1,),
                    other_positions=(-1,),
                ),
                sio2.place_above(silicon),
                flake.place_at_center(volume, axes=(0, 1)),
                flake.place_above(sio2),
                au.place_at_center(volume, axes=(0, 1)),
                au.place_above(flake),
            ]
        )
        objects.extend((silicon, sio2, flake, au))
    else:
        constraints.extend(
            [
                flake.place_at_center(volume, axes=(0, 1)),
                flake.place_at_center(volume, axes=(2,), margins=(-50e-9,)),
                au.place_at_center(volume, axes=(0, 1)),
                au.place_above(flake),
            ]
        )
        objects.extend((flake, au))

    # A 40-um square aperture has negligible requested-Gaussian intensity at
    # its boundary.  The circular mask radius is 20 um; std is chosen so its
    # untruncated intensity 1/e^2 radius is exactly 8.5 um.
    source_radius = 20.0e-6
    source_std = W0_M / (math.sqrt(2.0) * source_radius)
    source = fdtdx.GaussianPlaneSource(
        name="gaussian_source",
        partial_grid_shape=(272, 272, 1),
        fixed_E_polarization_vector=(1.0, 0.0, 0.0),
        wave_character=fdtdx.WaveCharacter(wavelength=WAVELENGTH_M),
        radius=source_radius,
        std=source_std,
        direction="-",
    )
    constraints.extend(
        [
            source.place_at_center(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(2,), margins=(1.0e-6,)),
        ]
    )
    objects.append(source)

    wave = fdtdx.WaveCharacter(wavelength=WAVELENGTH_M)
    previous = fdtdx.OnOffSwitch(
        start_time=(total_periods - 2 * window_periods) * period_s,
        end_time=(total_periods - window_periods) * period_s,
    )
    late = fdtdx.OnOffSwitch(start_time=(total_periods - window_periods) * period_s)
    detector_targets = [("au", au), ("tairte4", flake)]
    if include_substrate:
        # The oxide is laterally infinite in the numerical scene, but the
        # source aperture and closed control volume both exclude the 8-cell
        # PML on each side.  Measure exactly that same 272 x 272 footprint so
        # component Q and six-face power refer to one physical volume.
        detector_targets.append(("sio2", sio2))
    for material_name, target in detector_targets:
        for window_name, switch in (("previous", previous), ("late", late)):
            detector_shape = (
                (272, 272, 19)
                if material_name == "sio2"
                else target.partial_grid_shape
            )
            detector = fdtdx.PhasorDetector(
                name=f"{material_name}_{window_name}",
                partial_grid_shape=detector_shape,
                wave_characters=(wave,),
                components=("Ex", "Ey", "Ez"),
                dtype=jnp.complex64,
                switch=switch,
                # Keep each electric component on its native Yee location.
                # Co-locating all three components onto the Ez point is useful
                # for field visualization, but it is not the correct operation
                # for component-wise material dissipation at a metal interface.
                exact_interpolation=False,
                plot=False,
            )
            constraints.append(detector.same_position(target))
            objects.append(detector)
    if include_substrate:
        for window_name, switch in (("previous", previous), ("late", late)):
            detector = fdtdx.PhasorDetector(
                name=f"sio2_uniform_core_{window_name}",
                partial_grid_shape=(240, 240, 19),
                wave_characters=(wave,),
                components=("Ex", "Ey", "Ez"),
                dtype=jnp.complex64,
                switch=switch,
                exact_interpolation=False,
                plot=False,
            )
            constraints.append(detector.same_position(sio2))
            objects.append(detector)

    incident = fdtdx.PhasorPoyntingFluxDetector(
        name="incident_plane",
        partial_grid_shape=(272, 272, 1),
        wave_characters=(wave,),
        direction="-",
        dtype=jnp.complex64,
        switch=late,
        exact_interpolation=True,
    )
    constraints.extend(
        [
            incident.place_at_center(volume, axes=(0, 1)),
            incident.place_at_center(volume, axes=(2,), margins=(0.6e-6,)),
        ]
    )
    objects.append(incident)

    target_field = fdtdx.PhasorDetector(
        name="target_field",
        partial_grid_shape=(272, 272, 1),
        wave_characters=(wave,),
        components=("Ex", "Ey", "Ez"),
        dtype=jnp.complex64,
        switch=late,
        exact_interpolation=True,
        plot=False,
    )
    constraints.extend(
        [
            target_field.place_at_center(volume, axes=(0, 1)),
            target_field.place_at_center(volume, axes=(2,), margins=(0.2e-6,)),
        ]
    )
    objects.append(target_field)

    closed_shape = (
        (
            272,
            272,
            substrate_layout["near_box_stop"]
            - substrate_layout["near_box_start"],
        )
        if include_substrate
        else (220, 220, 16)
    )
    closed = fdtdx.ClosedSurfacePhasorPoyntingFluxDetector(
        name="material_flux",
        partial_grid_shape=closed_shape,
        wave_characters=(wave,),
        orientation="inward",
        dtype=jnp.complex64,
        switch=late,
        exact_interpolation=True,
    )
    constraints.append(closed.place_at_center(volume, axes=(0, 1)))
    if include_substrate:
        # The matched layout keeps fine Si cells below the oxide; the legacy
        # abrupt layout is retained only as a reproducible diagnostic option.
        constraints.append(
            closed.place_relative_to(
                volume,
                axes=(2,),
                own_positions=(-1,),
                other_positions=(-1,),
                margins=(
                    float(z_edges[substrate_layout["near_box_start"]] - z_edges[0]),
                ),
            )
        )
    else:
        constraints.append(closed.place_at_center(volume, axes=(2,), margins=(0.0,)))
    objects.append(closed)
    if include_substrate:
        deep_closed = fdtdx.ClosedSurfacePhasorPoyntingFluxDetector(
            name="material_flux_deep",
            partial_grid_shape=(
                272,
                272,
                substrate_layout["deep_box_stop"]
                - substrate_layout["deep_box_start"],
            ),
            wave_characters=(wave,),
            orientation="inward",
            dtype=jnp.complex64,
            switch=late,
            exact_interpolation=True,
        )
        td_deep_closed = fdtdx.ClosedSurfacePoyntingFluxDetector(
            name="material_flux_deep_td",
            partial_grid_shape=(
                272,
                272,
                substrate_layout["deep_box_stop"]
                - substrate_layout["deep_box_start"],
            ),
            orientation="inward",
            switch=late,
        )
        for detector in (deep_closed, td_deep_closed):
            constraints.extend(
                [
                    detector.place_at_center(volume, axes=(0, 1)),
                    detector.place_relative_to(
                        volume,
                        axes=(2,),
                        own_positions=(-1,),
                        other_positions=(-1,),
                        margins=(
                            float(
                                z_edges[substrate_layout["deep_box_start"]]
                                - z_edges[0]
                            ),
                        ),
                    ),
                ]
            )
            objects.append(detector)
        uniform_deep_closed = fdtdx.ClosedSurfacePhasorPoyntingFluxDetector(
            name="material_flux_uniform_core",
            partial_grid_shape=(
                240,
                240,
                substrate_layout["deep_box_stop"]
                - substrate_layout["deep_box_start"],
            ),
            wave_characters=(wave,),
            orientation="inward",
            dtype=jnp.complex64,
            switch=late,
            exact_interpolation=True,
        )
        uniform_td_deep_closed = fdtdx.ClosedSurfacePoyntingFluxDetector(
            name="material_flux_uniform_core_td",
            partial_grid_shape=(
                240,
                240,
                substrate_layout["deep_box_stop"]
                - substrate_layout["deep_box_start"],
            ),
            orientation="inward",
            switch=late,
        )
        for detector in (uniform_deep_closed, uniform_td_deep_closed):
            constraints.extend(
                [
                    detector.place_at_center(volume, axes=(0, 1)),
                    detector.place_relative_to(
                        volume,
                        axes=(2,),
                        own_positions=(-1,),
                        other_positions=(-1,),
                        margins=(
                            float(
                                z_edges[substrate_layout["deep_box_start"]]
                                - z_edges[0]
                            ),
                        ),
                    ),
                ]
            )
            objects.append(detector)

    key = jax.random.PRNGKey(20260821)
    placed, base, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    base, placed, _ = fdtdx.apply_params(base, placed, params, key)
    flake_slice = placed["fixed_tairte4"].grid_slice
    au_slice = placed["exact_binary_au"].grid_slice
    silicon_slice = placed["fixed_silicon_substrate"].grid_slice if include_substrate else None
    sio2_slice = placed["fixed_285nm_sio2"].grid_slice if include_substrate else None
    realized = config.resolved_grid
    if realized is None:
        raise RuntimeError("Missing realized grid")
    placement = {
        "flake_slice": _slice_tuple(flake_slice),
        "flake_extent_m": list(realized.slice_extent(tuple((x[0], x[1]) for x in _slice_tuple(flake_slice)))),
        "au_slice": _slice_tuple(au_slice),
        "au_extent_m": list(realized.slice_extent(tuple((x[0], x[1]) for x in _slice_tuple(au_slice)))),
        "source_slice": _slice_tuple(placed["gaussian_source"].grid_slice),
        "incident_slice": _slice_tuple(placed["incident_plane"].grid_slice),
        "target_slice": _slice_tuple(placed["target_field"].grid_slice),
        "closed_surface_slice": _slice_tuple(placed["material_flux"].grid_slice),
    }
    if include_substrate:
        placement.update(
            {
                "silicon_slice": _slice_tuple(silicon_slice),
                "silicon_extent_m": list(
                    realized.slice_extent(
                        tuple((x[0], x[1]) for x in _slice_tuple(silicon_slice))
                    )
                ),
                "sio2_slice": _slice_tuple(sio2_slice),
                "sio2_extent_m": list(
                    realized.slice_extent(
                        tuple((x[0], x[1]) for x in _slice_tuple(sio2_slice))
                    )
                ),
                "sio2_detector_slice": _slice_tuple(placed["sio2_late"].grid_slice),
                "deep_closed_surface_slice": _slice_tuple(
                    placed["material_flux_deep"].grid_slice
                ),
                "deep_td_closed_surface_slice": _slice_tuple(
                    placed["material_flux_deep_td"].grid_slice
                ),
                "uniform_core_sio2_detector_slice": _slice_tuple(
                    placed["sio2_uniform_core_late"].grid_slice
                ),
                "uniform_core_closed_surface_slice": _slice_tuple(
                    placed["material_flux_uniform_core"].grid_slice
                ),
                "uniform_core_td_closed_surface_slice": _slice_tuple(
                    placed["material_flux_uniform_core_td"].grid_slice
                ),
            }
        )
    audit = {
        "status": "FDTDX_LUMERICAL_BINARY_ENDPOINT_RUNSETUP_AUDIT",
        "software": {
            "fdtdx_import_path": fdtdx.__file__,
            "fdtdx_source_commit": subprocess.check_output(
                ["git", "-C", str(FDTDX_SOURCE), "rev-parse", "HEAD"], text=True
            ).strip(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "grid": {
            "shape_xyz": list(realized.shape),
            "cell_count": int(np.prod(realized.shape)),
            "bounds_m_xyz": [
                [float(realized.edges(axis)[0]), float(realized.edges(axis)[-1])]
                for axis in range(3)
            ],
            "min_spacing_m_xyz": list(realized.min_spacings),
            "max_spacing_m_xyz": [
                float(np.max(np.asarray(realized.cell_widths(axis)))) for axis in range(3)
            ],
            "pml_cells_each_face_xyz": [8, 8, 8],
        },
        "placement": placement,
        "source": {
            "wavelength_m": WAVELENGTH_M,
            "requested_w0_m": W0_M,
            "radius_m": source_radius,
            "std_relative_to_radius": source_std,
            "direction": "-z",
            "polarization": "x=b",
        },
        "numerics": {
            "time_steps_total": config.time_steps_total,
            "time_step_s": config.time_step_duration,
            "total_periods": total_periods,
            "window_periods": window_periods,
            "gradient_method": "checkpointed" if gradient_smoke else None,
            "gradient_checkpoints": gradient_checkpoints if gradient_smoke else None,
        },
        "substrate": {
            "included": include_substrate,
            "material_contract": substrate_provenance,
            "epsilon_sio2": (
                [epsilon_sio2.real, epsilon_sio2.imag] if include_substrate else None
            ),
            "epsilon_si": [epsilon_si.real, epsilon_si.imag] if include_substrate else None,
            "silicon_loss_model": (
                "static real epsilon; k=0 diagnostic approximation"
                if include_substrate
                else None
            ),
            "sio2_loss_representation": (
                substrate_loss_representation if include_substrate else None
            ),
            "sio2_equivalent_conductivity_S_m": (
                float(omega * stage41.EPS0_F_PER_M * epsilon_sio2.imag)
                if include_substrate
                else None
            ),
            "Palik_Si_readback_validated": (
                substrate_provenance["Palik_Si_readback_validated"]
                if include_substrate
                else None
            ),
            "matched_interface_grid": (
                matched_substrate_interface_grid if include_substrate else None
            ),
            "layout_indices": substrate_layout if include_substrate else None,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "fdtdx_lumerical_binary_endpoint_runsetup_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if audit_only:
        print(json.dumps(audit, indent=2))
        return audit

    if any(value is None for value in (base.dispersive_c1, base.dispersive_c2, base.dispersive_c3)):
        raise RuntimeError("Missing ADE arrays")
    spatial = base.dispersive_c1.shape[-3:]
    fixed_c1 = jnp.zeros((1, 3, *spatial), dtype=jnp.float32)
    fixed_c2 = jnp.zeros_like(fixed_c1)
    ta_c3 = jnp.zeros_like(fixed_c1)
    au_c3_template = jnp.zeros_like(fixed_c1)
    sio2_c3 = jnp.zeros_like(fixed_c1)
    if include_substrate and substrate_loss_representation == "lorentz":
        c1_sio2, c2_sio2, c3_sio2 = coeff["sio2"]
        for component in range(3):
            index = (0, component, *sio2_slice)
            fixed_c1 = fixed_c1.at[index].set(c1_sio2)
            fixed_c2 = fixed_c2.at[index].set(c2_sio2)
            sio2_c3 = sio2_c3.at[index].set(c3_sio2)
    for component, axis in enumerate(("b", "a", "c")):
        c1, c2, c3 = coeff[axis]
        index = (0, component, *flake_slice)
        fixed_c1 = fixed_c1.at[index].set(c1)
        fixed_c2 = fixed_c2.at[index].set(c2)
        ta_c3 = ta_c3.at[index].set(c3)
    c1_au, c2_au, c3_au = coeff["au"]
    for component in range(3):
        index = (0, component, *au_slice)
        fixed_c1 = fixed_c1.at[index].set(c1_au)
        fixed_c2 = fixed_c2.at[index].set(c2_au)
        au_c3_template = au_c3_template.at[index].set(c3_au)

    def arrays_for_case(strengths):
        ta_strength, au_strength = strengths[0], strengths[1]
        c3 = sio2_c3 + ta_strength * ta_c3 + au_strength * au_c3_template
        return (
            base.reset()
            .aset("dispersive_c1", fixed_c1)
            .aset("dispersive_c2", fixed_c2)
            .aset("dispersive_c3", c3)
        )

    au_volume = _electric_yee_dual_volumes(realized, au_slice)
    ta_volume = _electric_yee_dual_volumes(realized, flake_slice)
    sio2_volume = (
        _electric_yee_dual_volumes(realized, placed["sio2_late"].grid_slice)
        if include_substrate
        else None
    )
    sio2_uniform_core_volume = (
        _electric_yee_dual_volumes(
            realized, placed["sio2_uniform_core_late"].grid_slice
        )
        if include_substrate
        else None
    )
    # FDTDX evolves eta0-normalized fields: E_internal = E_SI / eta0 while H
    # is in SI units.  Its raw Poynting detector therefore returns S_SI/eta0.
    # Convert the electric phasor to SI before evaluating
    #   Q = 0.5*omega*eps0*Im(eps)*|E_SI|^2.
    # This is a documented unit conversion, not an empirical gain or endpoint
    # matching factor.
    eta0 = float(fdtdx.constants.eta0)
    prefactor = 0.5 * omega * stage41.EPS0_F_PER_M * eta0**2 / POWER_SCALE_W
    ta_imag = jnp.asarray(
        [epsilon_ta["b"].imag, epsilon_ta["a"].imag, epsilon_ta["c"].imag],
        dtype=jnp.float32,
    )[:, None, None, None]

    def powers(out, window: str, strengths):
        ta_strength, au_strength = strengths[0], strengths[1]
        e_au = out.detector_states[f"au_{window}"]["phasor"][0, 0]
        e_ta = out.detector_states[f"tairte4_{window}"]["phasor"][0, 0]
        p_au_comp = prefactor * epsilon_au.imag * au_strength * jnp.sum(
            jnp.abs(e_au) ** 2 * au_volume, axis=(1, 2, 3)
        )
        p_ta_comp = prefactor * ta_strength * jnp.sum(
            ta_imag * jnp.abs(e_ta) ** 2 * ta_volume, axis=(1, 2, 3)
        )
        material_components = [p_au_comp, p_ta_comp]
        total = p_au_comp.sum() + p_ta_comp.sum()
        if include_substrate:
            e_sio2 = out.detector_states[f"sio2_{window}"]["phasor"][0, 0]
            p_sio2_comp = prefactor * epsilon_sio2.imag * jnp.sum(
                jnp.abs(e_sio2) ** 2 * sio2_volume, axis=(1, 2, 3)
            )
            material_components.append(p_sio2_comp)
            total = total + p_sio2_comp.sum()
        return jnp.concatenate((*material_components, jnp.stack((total,))))

    def solve(strengths):
        _, out = fdtdx.run_fdtd(
            arrays_for_case(strengths), placed, config, key, show_progress=False
        )
        late_power = powers(out, "late", strengths)
        previous_power = powers(out, "previous", strengths)
        p_inc = eta0 * placed["incident_plane"].compute_poynting_flux(
            out.detector_states["incident_plane"]
        )[0]
        p_closed = eta0 * placed["material_flux"].compute_net_flux(
            out.detector_states["material_flux"]
        )[0]
        if include_substrate:
            p_closed_deep = eta0 * placed["material_flux_deep"].compute_net_flux(
                out.detector_states["material_flux_deep"]
            )[0]
            p_closed_deep_td = eta0 * jnp.mean(
                out.detector_states["material_flux_deep_td"]["poynting_flux"][:, 0]
            )
            e_sio2_uniform = out.detector_states["sio2_uniform_core_late"][
                "phasor"
            ][0, 0]
            p_sio2_uniform = prefactor * epsilon_sio2.imag * jnp.sum(
                jnp.abs(e_sio2_uniform) ** 2
                * sio2_uniform_core_volume,
                axis=(1, 2, 3),
            )
            p_closed_uniform = eta0 * placed[
                "material_flux_uniform_core"
            ].compute_net_flux(out.detector_states["material_flux_uniform_core"])[0]
            p_closed_uniform_td = eta0 * jnp.mean(
                out.detector_states["material_flux_uniform_core_td"][
                    "poynting_flux"
                ][:, 0]
            )
        else:
            p_closed_deep = p_closed
            p_closed_deep_td = p_closed
            p_sio2_uniform = jnp.zeros((3,), dtype=jnp.float32)
            p_closed_uniform = p_closed
            p_closed_uniform_td = p_closed
        target = out.detector_states["target_field"]["phasor"][0, 0]
        return (
            late_power,
            previous_power,
            p_inc,
            p_closed,
            p_closed_deep,
            p_closed_deep_td,
            p_sio2_uniform,
            p_closed_uniform,
            p_closed_uniform_td,
            target,
        )

    if gradient_smoke:
        from fdtdx.fdtd.fdtd import checkpointed_fdtd

        weighted_q_objective = spatial_q_weight_npz is not None
        if weighted_q_objective:
            if spatial_q_weight_summary_json is None:
                raise ValueError(
                    "--spatial-q-weight-summary-json is required with spatial weights"
                )
            if spatial_weighted_gradient_raw_path is None:
                raise ValueError(
                    "--spatial-weighted-gradient-raw-path is required with spatial weights"
                )
            if not include_substrate or spatial_q_export:
                raise ValueError(
                    "Spatially weighted Q requires substrate gradient-smoke mode"
                )
            if gradient_reference_json is not None:
                raise ValueError(
                    "Spatially weighted Q cannot reuse a total-power gradient reference"
                )
            if spatial_q_weight_scenario not in ("thermally_grown", "evaporated"):
                raise ValueError(spatial_q_weight_scenario)
        elif spatial_q_weight_summary_json is not None or spatial_weighted_gradient_raw_path is not None:
            raise ValueError("Spatial-weight metadata/raw path supplied without weight NPZ")
        if not gradient_direction_names and not include_adjoint_aligned:
            raise ValueError("At least one gradient direction is required")
        if not gradient_steps or any(step <= 0.0 for step in gradient_steps):
            raise ValueError(f"Gradient steps must be positive: {gradient_steps}")
        latent_shape = (20, 20)
        optical_repeat = (au_slice[0].stop - au_slice[0].start) // latent_shape[0]
        if optical_repeat != 5 or (au_slice[1].stop - au_slice[1].start) != 5 * latent_shape[1]:
            raise RuntimeError(f"Unexpected latent-to-Yee layout: Au slice={au_slice}")

        lx = jnp.linspace(-1.0, 1.0, latent_shape[0])[:, None]
        ly = jnp.linspace(-1.0, 1.0, latent_shape[1])[None, :]
        rho0 = (0.52 + 0.07 * jnp.cos(0.8 * math.pi * lx) * jnp.cos(0.65 * math.pi * ly) + 0.02 * lx).astype(
            jnp.float32
        )

        def optical_strength(rho):
            # Piecewise-constant 500-nm design pixels on the 100-nm Yee grid;
            # the two physical 25-nm Au layers share the same z-projected value.
            upsampled = jnp.repeat(jnp.repeat(rho, optical_repeat, axis=0), optical_repeat, axis=1)
            return jnp.broadcast_to((upsampled**3)[:, :, None], (100, 100, 2))

        def arrays_for_density(rho, ta_strength=1.0):
            strength = optical_strength(rho)
            c3 = sio2_c3 + ta_strength * ta_c3
            for component in range(3):
                c3 = c3.at[(0, component, *au_slice)].set(c3_au * strength)
            return (
                base.reset()
                .aset("dispersive_c1", fixed_c1)
                .aset("dispersive_c2", fixed_c2)
                .aset("dispersive_c3", c3)
            )

        spatial_weights = None
        spatial_weight_summary = None
        spatial_weight_raw_sha = None
        if weighted_q_objective:
            weight_path = spatial_q_weight_npz.expanduser().resolve()
            weight_summary_path = spatial_q_weight_summary_json.expanduser().resolve()
            spatial_weight_summary = json.loads(
                weight_summary_path.read_text(encoding="utf-8")
            )
            if (
                spatial_weight_summary.get("status")
                != "VALIDATED_NATIVE_YEE_THERMAL_SOURCE_ADJOINT_PULLBACK"
            ):
                raise RuntimeError("Fail-closed: native-Yee weight status mismatch")
            spatial_weight_raw_sha = _sha256(weight_path)
            if spatial_weight_raw_sha != spatial_weight_summary["raw_artifact"]["sha256"]:
                raise RuntimeError("Fail-closed: native-Yee weight SHA mismatch")
            with np.load(weight_path, allow_pickle=False) as weight_raw:
                spatial_weights = {
                    material: jnp.asarray(
                        np.stack(
                            [
                                np.asarray(
                                    weight_raw[
                                        f"weight_{spatial_q_weight_scenario}_{material}_{component}_A_W"
                                    ],
                                    dtype=np.float32,
                                )
                                for component in "xyz"
                            ]
                        ),
                        dtype=jnp.float32,
                    )
                    for material in ("au", "tairte4", "sio2")
                }
            expected_weight_shapes = {
                "au": tuple(au_volume.shape),
                "tairte4": tuple(ta_volume.shape),
                "sio2": tuple(sio2_volume.shape),
            }
            actual_weight_shapes = {
                material: tuple(value.shape)
                for material, value in spatial_weights.items()
            }
            if actual_weight_shapes != expected_weight_shapes:
                raise RuntimeError(
                    "Fail-closed native-Yee weight/grid mismatch: "
                    f"actual={actual_weight_shapes}, expected={expected_weight_shapes}"
                )

        def q_fields_for_density(out, window, rho, ta_strength=1.0):
            e_au = out.detector_states[f"au_{window}"]["phasor"][0, 0]
            e_ta = out.detector_states[f"tairte4_{window}"]["phasor"][0, 0]
            strength = optical_strength(rho)
            fields = {
                "au": prefactor
                * epsilon_au.imag
                * strength[None, ...]
                * jnp.abs(e_au) ** 2,
                "tairte4": prefactor
                * ta_strength
                * ta_imag
                * jnp.abs(e_ta) ** 2,
            }
            if include_substrate:
                e_sio2 = out.detector_states[f"sio2_{window}"]["phasor"][0, 0]
                fields["sio2"] = (
                    prefactor * epsilon_sio2.imag * jnp.abs(e_sio2) ** 2
                )
            return fields

        def powers_for_density(out, window, rho, ta_strength=1.0):
            q_fields = q_fields_for_density(out, window, rho, ta_strength)
            p_au_comp = jnp.sum(
                q_fields["au"] * au_volume, axis=(1, 2, 3)
            )
            p_ta_comp = jnp.sum(
                q_fields["tairte4"] * ta_volume, axis=(1, 2, 3)
            )
            material_components = [p_au_comp, p_ta_comp]
            total = p_au_comp.sum() + p_ta_comp.sum()
            if include_substrate:
                p_sio2_comp = jnp.sum(
                    q_fields["sio2"] * sio2_volume, axis=(1, 2, 3)
                )
                material_components.append(p_sio2_comp)
                total = total + p_sio2_comp.sum()
            return jnp.concatenate((*material_components, jnp.stack((total,))))

        def weighted_source_objective(out, window, rho):
            if spatial_weights is None:
                return powers_for_density(out, window, rho)[-1]
            q_fields = q_fields_for_density(out, window, rho)
            return (
                jnp.sum(q_fields["au"] * au_volume * spatial_weights["au"])
                + jnp.sum(
                    q_fields["tairte4"]
                    * ta_volume
                    * spatial_weights["tairte4"]
                )
                + jnp.sum(
                    q_fields["sio2"]
                    * sio2_volume
                    * spatial_weights["sio2"]
                )
            )

        def objective_with_aux(rho):
            _, out = checkpointed_fdtd(arrays_for_density(rho), placed, config, key, show_progress=False)
            late_power = powers_for_density(out, "late", rho)
            previous_power = powers_for_density(out, "previous", rho)
            late_objective = weighted_source_objective(out, "late", rho)
            previous_objective = weighted_source_objective(out, "previous", rho)
            p_inc = eta0 * placed["incident_plane"].compute_poynting_flux(
                out.detector_states["incident_plane"]
            )[0]
            p_closed = eta0 * placed["material_flux"].compute_net_flux(
                out.detector_states["material_flux"]
            )[0]
            p_closed_primary = (
                eta0
                * jnp.mean(
                    out.detector_states["material_flux_deep_td"]["poynting_flux"][
                        :, 0
                    ]
                )
                if include_substrate
                else p_closed
            )
            return late_objective, (
                late_power,
                previous_power,
                p_inc,
                p_closed,
                p_closed_primary,
                previous_objective,
            )

        def substrate_reference_aux():
            """Return Q and closed flux for the fixed optical substrate alone.

            The difference P_closed-P_Q is the finite-window numerical
            residual.  Subtracting that residual from the full box flux keeps
            the real SiO2 absorption instead of treating it as background.
            """
            zero = jnp.zeros_like(rho0)
            _, out = checkpointed_fdtd(
                arrays_for_density(zero, ta_strength=0.0),
                placed,
                config,
                key,
                show_progress=False,
            )
            late_power = powers_for_density(out, "late", zero, ta_strength=0.0)
            p_closed_primary = eta0 * jnp.mean(
                out.detector_states["material_flux_deep_td"]["poynting_flux"][:, 0]
            )
            return late_power[-1], p_closed_primary

        if spatial_q_export:
            if spatial_q_raw_path is None:
                raise ValueError("--spatial-q-raw-path is required for spatial Q export")
            if not include_substrate or sio2_volume is None:
                raise ValueError(
                    "Spatial Q export requires --include-substrate so Au, TaIrTe4, "
                    "and SiO2 are all represented"
                )
            print("[spatial-Q] executing one baseline forward", flush=True)
            export_start = time.perf_counter()
            _, export_out = fdtdx.run_fdtd(
                arrays_for_density(rho0), placed, config, key, show_progress=False
            )
            export_runtime = time.perf_counter() - export_start
            export_power = powers_for_density(export_out, "late", rho0)
            previous_export_power = powers_for_density(export_out, "previous", rho0)
            e_au = export_out.detector_states["au_late"]["phasor"][0, 0]
            e_ta = export_out.detector_states["tairte4_late"]["phasor"][0, 0]
            e_sio2 = export_out.detector_states["sio2_late"]["phasor"][0, 0]
            strength = optical_strength(rho0)
            physical_prefactor = prefactor * POWER_SCALE_W
            q_fields = {
                "au": physical_prefactor * epsilon_au.imag * strength[None, ...]
                * jnp.abs(e_au) ** 2,
                "tairte4": physical_prefactor * ta_imag * jnp.abs(e_ta) ** 2,
                "sio2": physical_prefactor * epsilon_sio2.imag * jnp.abs(e_sio2) ** 2,
            }
            volumes = {"au": au_volume, "tairte4": ta_volume, "sio2": sio2_volume}

            def component_coordinates(grid_slice, component: int):
                values = []
                metrics = []
                for axis, part in enumerate(grid_slice):
                    edges_axis = np.asarray(realized.edges(axis), dtype=np.float64)
                    centers_axis = 0.5 * (edges_axis[:-1] + edges_axis[1:])
                    widths_axis = np.diff(edges_axis)
                    edge_dual_axis = 0.5 * (
                        np.concatenate((widths_axis[:1], widths_axis[:-1]))
                        + widths_axis
                    )
                    samples = centers_axis if axis == component else edges_axis[:-1]
                    metric = widths_axis if axis == component else edge_dual_axis
                    values.append(samples[int(part.start) : int(part.stop)])
                    metrics.append(metric[int(part.start) : int(part.stop)])
                return values, metrics

            raw = spatial_q_raw_path.expanduser().resolve()
            raw.parent.mkdir(parents=True, exist_ok=True)
            payload: dict[str, np.ndarray] = {"rho": np.asarray(rho0, dtype=np.float32)}
            material_slices = {
                "au": au_slice,
                "tairte4": flake_slice,
                "sio2": placed["sio2_late"].grid_slice,
            }
            reintegrated = {}
            finite_nonnegative = True
            for material in ("au", "tairte4", "sio2"):
                q_value = np.asarray(q_fields[material], dtype=np.float32)
                volume_value = np.asarray(volumes[material], dtype=np.float32)
                payload[f"Q_{material}_W_m3"] = q_value
                payload[f"dual_volume_{material}_m3"] = volume_value
                component_power = np.sum(
                    q_value.astype(np.float64) * volume_value.astype(np.float64),
                    axis=(1, 2, 3),
                )
                reintegrated[material] = component_power
                finite_nonnegative = finite_nonnegative and bool(
                    np.all(np.isfinite(q_value)) and np.all(q_value >= 0.0)
                )
                for component, name in enumerate("xyz"):
                    coordinates, metrics = component_coordinates(
                        material_slices[material], component
                    )
                    for axis, axis_name in enumerate("xyz"):
                        payload[f"{material}_{name}_{axis_name}_m"] = coordinates[axis]
                        payload[
                            f"dual_width_{material}_{name}_{axis_name}_m"
                        ] = metrics[axis]
            np.savez_compressed(raw, **payload)
            expected = np.asarray(export_power, dtype=np.float64) * POWER_SCALE_W
            reintegrated_vector = np.concatenate(
                (reintegrated["au"], reintegrated["tairte4"], reintegrated["sio2"])
            )
            expected_vector = expected[:9]
            component_reintegration_error = np.abs(
                reintegrated_vector - expected_vector
            ) / np.maximum(np.abs(expected_vector), np.finfo(float).tiny)
            reintegration_error = float(
                np.max(component_reintegration_error)
            )
            closed = float(
                eta0
                * jnp.mean(
                    export_out.detector_states["material_flux_deep_td"]["poynting_flux"][:, 0]
                )
            )
            total_q = float(expected[-1])
            closure = _relative(total_q, closed)
            window = _relative(
                total_q,
                float(np.asarray(previous_export_power, dtype=np.float64)[-1] * POWER_SCALE_W),
            )
            gates = {
                "gpu_only": True,
                "finite_nonnegative_Q": finite_nonnegative,
                "native_Q_reintegration_lt_1e-6": reintegration_error < 1.0e-6,
                "Q_flux_closure_lt_0p5pct": closure < 0.005,
                "late_window_change_lt_0p5pct": window < 0.005,
                "no_clipping_smoothing_gain_or_rescaling": True,
            }
            passed = all(gates.values())
            result = {
                "status": (
                    "VALIDATED_FDTDX_SUBSTRATE_SPATIAL_NATIVE_YEE_Q_EXPORT"
                    if passed
                    else "FAILED_FDTDX_SUBSTRATE_SPATIAL_NATIVE_YEE_Q_EXPORT"
                ),
                "scope": (
                    "16-period/4-window baseline spatial native-Yee Qx/Qy/Qz export for "
                    "Au, TaIrTe4, and SiO2; no thermal/PTE/electrical/adjoint/optimization"
                ),
                "audit": audit,
                "P_Q_W": total_q,
                "component_power_W": {
                    "au_xyz": list(map(float, expected[:3])),
                    "tairte4_xyz": list(map(float, expected[3:6])),
                    "sio2_xyz": list(map(float, expected[6:9])),
                },
                "reintegrated_component_power_W": {
                    name: list(map(float, value)) for name, value in reintegrated.items()
                },
                "native_Q_reintegration_relative_error": reintegration_error,
                "native_Q_component_reintegration_relative_error": list(
                    map(float, component_reintegration_error)
                ),
                "closed_surface_inward_W": closed,
                "Q_flux_closure_relative": closure,
                "late_window_relative_change": window,
                "component_coordinate_contract": (
                    "Ex=(x center,y lower edge,z lower edge); "
                    "Ey=(x lower edge,y center,z lower edge); "
                    "Ez=(x lower edge,y lower edge,z center), with stored axis-wise "
                    "dual widths and dual volumes"
                ),
                "raw_artifact": {
                    "path": str(raw),
                    "bytes": raw.stat().st_size,
                    "sha256": _sha256(raw),
                },
                "runtime_seconds": export_runtime,
                "gates": gates,
                "no_clipping_smoothing_gain_or_result_rescaling": True,
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            result_path = output_dir / "fdtdx_substrate_spatial_native_yee_q_export.json"
            result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result, indent=2))
            return result

        reference = None
        reference_ad_directional: dict[str, float] = {}
        if gradient_reference_json is None:
            print(
                f"[gradient-smoke] compiling checkpointed AD with {gradient_checkpoints} checkpoints",
                flush=True,
            )
            compile_start = time.perf_counter()
            value_grad = (
                jax.jit(jax.value_and_grad(objective_with_aux, has_aux=True))
                .lower(rho0)
                .compile()
            )
            compile_seconds = time.perf_counter() - compile_start
            print(f"[gradient-smoke] AD compile complete: {compile_seconds:.3f} s", flush=True)
            ad_start = time.perf_counter()
            (value_scaled, aux), gradient_scaled = value_grad(rho0)
            jax.block_until_ready(gradient_scaled)
            ad_seconds = time.perf_counter() - ad_start
            print(f"[gradient-smoke] AD execution complete: {ad_seconds:.3f} s", flush=True)
            (
                late_scaled,
                previous_scaled,
                p_inc,
                p_closed,
                p_closed_primary,
                previous_objective_scaled,
            ) = aux
            value_w = float(value_scaled) * POWER_SCALE_W
            previous_objective_w = (
                float(previous_objective_scaled) * POWER_SCALE_W
            )
            gradient_w = np.asarray(gradient_scaled, dtype=np.float64) * POWER_SCALE_W
            late_w = np.asarray(late_scaled, dtype=np.float64) * POWER_SCALE_W
            previous_w = np.asarray(previous_scaled, dtype=np.float64) * POWER_SCALE_W
            grad_l2 = float(np.linalg.norm(gradient_w))
            substrate_reference_q_w = 0.0
            substrate_reference_closed_w = 0.0
            if include_substrate:
                print("[gradient-smoke] computing substrate-only closure reference", flush=True)
                substrate_reference_jit = jax.jit(substrate_reference_aux).lower().compile()
                substrate_reference_q_scaled, substrate_reference_closed = substrate_reference_jit()
                substrate_reference_q_w = (
                    float(substrate_reference_q_scaled) * POWER_SCALE_W
                )
                substrate_reference_closed_w = float(substrate_reference_closed)
        else:
            reference = json.loads(Path(gradient_reference_json).read_text(encoding="utf-8"))
            reference_numerics = reference["audit"]["numerics"]
            if (
                reference_numerics["total_periods"] != total_periods
                or reference_numerics["window_periods"] != window_periods
                or reference["audit"]["grid"]["shape_xyz"] != list(grid.shape)
            ):
                raise RuntimeError("Stored gradient reference does not match the current grid/time contract")
            if include_adjoint_aligned:
                raise RuntimeError("Forward-only refinement cannot reconstruct an adjoint-aligned direction")
            baseline = reference["baseline"]
            value_w = float(baseline["P_Q_W"])
            grad_l2 = float(baseline["gradient_l2_W"])
            component = baseline["component_power_W"]
            component_vectors = [component["au_xyz"], component["tairte4_xyz"]]
            if include_substrate:
                component_vectors.append(component["sio2_xyz"])
            late_w = np.asarray(
                [value for vector in component_vectors for value in vector] + [value_w],
                dtype=np.float64,
            )
            previous_w = late_w.copy()
            previous_w[-1] = value_w / (
                1.0 - float(baseline["late_window_relative_change"])
            )
            p_inc = float(baseline["incident_plane_W"])
            p_closed = float(baseline["near_phasor_closed_surface_inward_W"])
            p_closed_primary = float(baseline["closed_surface_inward_W"])
            closure_ref = baseline["closure_correction"]
            substrate_reference_q_w = float(closure_ref.get("substrate_only_Q_W", 0.0))
            substrate_reference_closed_w = float(
                closure_ref.get("substrate_only_closed_surface_W", 0.0)
            )
            gradient_w = np.asarray((grad_l2,), dtype=np.float64)
            previous_objective_w = value_w / (
                1.0 - float(baseline["late_window_relative_change"])
            )
            compile_seconds = 0.0
            ad_seconds = 0.0
            reference_ad_directional = {
                row["direction"]: float(row["ad_W_per_unit_direction"])
                for row in reference["directions"]
            }
            print(
                f"[gradient-smoke] reusing immutable AD reference {gradient_reference_json}; "
                "running forward FD only",
                flush=True,
            )

        objective_unit = "A" if weighted_q_objective else "W"
        directions_np = {}
        x_np = np.linspace(-1.0, 1.0, latent_shape[0])[:, None]
        y_np = np.linspace(-1.0, 1.0, latent_shape[1])[None, :]
        smooth = np.sin(0.7 * math.pi * x_np) * np.cos(0.55 * math.pi * y_np) + 0.21 * x_np
        random = np.random.default_rng(20260821).standard_normal(latent_shape)
        available_directions = {
            "smooth_asymmetric": smooth,
            "fixed_seed_random": random,
        }
        unknown_directions = set(gradient_direction_names) - set(available_directions)
        if unknown_directions:
            raise ValueError(f"Unknown gradient directions: {sorted(unknown_directions)}")
        for name in gradient_direction_names:
            direction = available_directions[name]
            directions_np[name] = direction / np.linalg.norm(direction)
        if include_adjoint_aligned:
            # This direction is formed only after the full reverse-mode solve.
            # It is the strongest possible local directional check and avoids
            # any near-null classification ambiguity.  The central FD solves
            # below remain independent forward simulations.
            directions_np["adjoint_aligned"] = gradient_w / grad_l2

        def objective_only(rho):
            return objective_with_aux(rho)[0]

        objective_jit = jax.jit(objective_only).lower(rho0).compile()
        rows = []
        fd_start = time.perf_counter()
        strong_direction_threshold_fraction = 0.05
        for direction_name, direction_np in directions_np.items():
            if reference is not None and direction_name not in reference_ad_directional:
                raise RuntimeError(
                    f"Stored AD reference has no direction {direction_name!r}"
                )
            ad_directional = (
                reference_ad_directional[direction_name]
                if reference is not None
                else float(np.vdot(gradient_w, direction_np).real)
            )
            for h in gradient_steps:
                print(f"[gradient-smoke] FD {direction_name}, h={h:g}", flush=True)
                direction = jnp.asarray(direction_np, dtype=rho0.dtype)
                plus = float(objective_jit(rho0 + h * direction)) * POWER_SCALE_W
                minus = float(objective_jit(rho0 - h * direction)) * POWER_SCALE_W
                fd_directional = (plus - minus) / (2.0 * h)
                # A local relative error is well-conditioned only when the
                # directional derivative is not near-null.  Near-null
                # directions are still gated by the gradient-L2-normalized
                # error below; no gradient or FD value is rescaled.
                strong = (
                    max(abs(ad_directional), abs(fd_directional))
                    >= strong_direction_threshold_fraction * grad_l2
                )
                rows.append(
                    {
                        "direction": direction_name,
                        "h": h,
                        "ad_W_per_unit_direction": ad_directional,
                        "fd_W_per_unit_direction": fd_directional,
                        "strong_direction": bool(strong),
                        "strong_relative_error": abs(ad_directional - fd_directional)
                        / max(abs(fd_directional), 1e-300),
                        "gradient_l2_normalized_error": abs(ad_directional - fd_directional)
                        / max(grad_l2, 1e-300),
                        "power_plus_W": plus,
                        "power_minus_W": minus,
                    }
                )
                print(
                    "[gradient-smoke] "
                    f"AD={ad_directional:.8e} {objective_unit}, "
                    f"FD={fd_directional:.8e} {objective_unit}, "
                    f"rel={rows[-1]['strong_relative_error']:.6%}",
                    flush=True,
                )
        fd_seconds = time.perf_counter() - fd_start

        finest_h = min(gradient_steps)
        finest = [row for row in rows if row["h"] == finest_h]
        strongest_error = max(
            (row["strong_relative_error"] for row in finest if row["strong_direction"]), default=0.0
        )
        normalized_error = max(row["gradient_l2_normalized_error"] for row in finest)
        if include_substrate:
            numerical_box_residual = substrate_reference_closed_w - substrate_reference_q_w
            # With a lossy fixed substrate the substrate-only closed flux is
            # physical, not an empty-space correction.  Poynting closure must
            # therefore compare the full local Q directly with the full box.
            corrected_closed = float(p_closed_primary)
            closure_correction = {
                "method": (
                    "none; direct late-window local-Q versus deep-box time-domain "
                    "matched-volume flux"
                ),
                "substrate_only_Q_W": substrate_reference_q_w,
                "substrate_only_closed_surface_W": substrate_reference_closed_w,
                "substrate_only_closed_minus_Q_residual_W": numerical_box_residual,
            }
        else:
            endpoint_reference = json.loads(
                (
                    output_dir.parent
                    / "results_fdtdx_lumerical_binary_endpoints"
                    / "fdtdx_lumerical_binary_endpoints_summary.json"
                ).read_text(encoding="utf-8")
            )
            empty_box = endpoint_reference["cases"]["empty"]["closed_surface_inward_W"]
            corrected_closed = float(p_closed) - empty_box
            closure_correction = {
                "method": "subtract air-only empty-box finite-window residual",
                "empty_box_W": empty_box,
            }
        total_q_w = float(late_w[-1])
        closure = _relative(total_q_w, corrected_closed)
        window_change = _relative(float(late_w[-1]), float(previous_w[-1]))
        objective_window_change = _relative(value_w, previous_objective_w)
        gates = {
            "gpu_only": True,
            "finite": bool(np.isfinite(value_w) and np.all(np.isfinite(gradient_w))),
            "no_density_clipping": True,
            "late_window_change_lt_0p5pct": window_change < 0.005,
            "weighted_objective_window_change_lt_0p5pct": (
                objective_window_change < 0.005
            ),
            "empty_subtracted_Q_flux_closure_lt_0p5pct": closure < 0.005,
            "finest_strong_direction_error_lt_1pct": strongest_error < 0.01,
            "finest_gradient_l2_normalized_error_lt_1pct": normalized_error < 0.01,
        }
        passed = all(gates.values())
        if weighted_q_objective:
            published_rows = [
                {
                    "direction": row["direction"],
                    "h": row["h"],
                    "AD_A_per_unit_direction": row[
                        "ad_W_per_unit_direction"
                    ],
                    "FD_A_per_unit_direction": row[
                        "fd_W_per_unit_direction"
                    ],
                    "strong_direction": row["strong_direction"],
                    "strong_relative_error": row["strong_relative_error"],
                    "gradient_l2_normalized_error": row[
                        "gradient_l2_normalized_error"
                    ],
                    "objective_plus_A": row["power_plus_W"],
                    "objective_minus_A": row["power_minus_W"],
                }
                for row in rows
            ]
            raw_gradient_path = spatial_weighted_gradient_raw_path.expanduser().resolve()
            raw_gradient_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                raw_gradient_path,
                rho=np.asarray(rho0, dtype=np.float32),
                gradient_A=np.asarray(gradient_w, dtype=np.float64),
                weighted_objective_A=np.asarray(value_w, dtype=np.float64),
                total_P_Q_W=np.asarray(total_q_w, dtype=np.float64),
            )
            result = {
                "status": (
                    "VALIDATED_FDTDX_NATIVE_YEE_SPATIALLY_WEIGHTED_PTE_SOURCE_GRADIENT"
                    if passed
                    else "FAILED_FDTDX_NATIVE_YEE_SPATIALLY_WEIGHTED_PTE_SOURCE_GRADIENT"
                ),
                "scope": (
                    "FDTDX reverse-mode derivative of the native-Yee spatial-Q "
                    "contraction with a frozen explicit-thermal source adjoint; "
                    "no thermal/electrical direct density term, full combined AD-FD, "
                    "or optimization"
                ),
                "audit": audit,
                "spatial_weight": {
                    "scenario": spatial_q_weight_scenario,
                    "summary_json": str(spatial_q_weight_summary_json.resolve()),
                    "raw_npz": str(spatial_q_weight_npz.resolve()),
                    "raw_sha256": spatial_weight_raw_sha,
                    "units": "A/W on component-native Yee power cells",
                    "normalization_or_rescaling": False,
                },
                "design": {
                    "latent_shape_xy": list(latent_shape),
                    "latent_pitch_m": 500e-9,
                    "yee_shape_xy": [100, 100],
                    "yee_pitch_m": 100e-9,
                    "au_z_cells": 2,
                    "au_thickness_m": 50e-9,
                    "relaxation": "passive Drude coupling strength s(rho)=rho^3",
                    "rho_min": float(jnp.min(rho0)),
                    "rho_max": float(jnp.max(rho0)),
                },
                "baseline": {
                    "weighted_source_objective_A": value_w,
                    "previous_weighted_source_objective_A": previous_objective_w,
                    "weighted_objective_window_relative_change": objective_window_change,
                    "P_Q_W": total_q_w,
                    "component_power_W": {
                        "au_xyz": list(map(float, late_w[:3])),
                        "tairte4_xyz": list(map(float, late_w[3:6])),
                        "sio2_xyz": list(map(float, late_w[6:9])),
                    },
                    "closed_surface_inward_W": float(p_closed_primary),
                    "Q_flux_closure_relative": closure,
                    "late_Q_window_relative_change": window_change,
                    "gradient_l2_A": grad_l2,
                },
                "directions": published_rows,
                "runtime": {
                    "compile_seconds": compile_seconds,
                    "ad_seconds": ad_seconds,
                    "central_fd_forward_count": 2
                    * len(directions_np)
                    * len(gradient_steps),
                    "central_fd_forward_seconds": fd_seconds,
                },
                "gates": gates,
                "raw_artifact": {
                    "path": str(raw_gradient_path),
                    "bytes": raw_gradient_path.stat().st_size,
                    "sha256": _sha256(raw_gradient_path),
                    "committed_to_git": False,
                },
                "no_clipping_smoothing_gain_or_gradient_rescaling": True,
                "next_gate": (
                    "add this optical source gradient to the validated explicit "
                    "thermal/electrical direct gradient and run full combined AD-FD"
                ),
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            result_path = output_dir / "fdtdx_spatially_weighted_pte_source_gradient.json"
            result_path.write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            with (
                output_dir / "fdtdx_spatially_weighted_pte_source_gradient.csv"
            ).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(published_rows[0]), lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(published_rows)
            print(json.dumps(result, indent=2))
            return result
        result = {
            "status": (
                (
                    "VALIDATED_FDTDX_DIAGNOSTIC_SUBSTRATE_NONUNIFORM_AU_GRADIENT_SMOKE"
                    if include_substrate
                    else "VALIDATED_FDTDX_PRODUCTION_WIDTH_NONUNIFORM_AU_GRADIENT_SMOKE"
                )
                if passed
                else (
                    "FAILED_FDTDX_DIAGNOSTIC_SUBSTRATE_NONUNIFORM_AU_GRADIENT_SMOKE"
                    if include_substrate
                    else "FAILED_FDTDX_PRODUCTION_WIDTH_NONUNIFORM_AU_GRADIENT_SMOKE"
                )
            ),
            "scope": (
                "diagnostic SiO2/Si-substrate nonuniform Au optical total-Q AD-FD smoke; "
                "no thermal/PTE/electrical/optimization; not production while Palik Si readback is blocked"
                if include_substrate
                else "production-width nonuniform Au optical total-Q AD-FD smoke; no thermal/PTE/electrical/optimization"
            ),
            "audit": audit,
            "design": {
                "latent_shape_xy": list(latent_shape),
                "latent_pitch_m": 500e-9,
                "yee_shape_xy": [100, 100],
                "yee_pitch_m": 100e-9,
                "au_z_cells": 2,
                "au_thickness_m": 50e-9,
                "relaxation": "passive Drude coupling strength s(rho)=rho^3",
                "rho_min": float(jnp.min(rho0)),
                "rho_max": float(jnp.max(rho0)),
            },
            "baseline": {
                "P_Q_W": value_w,
                "component_power_W": {
                    "au_xyz": list(map(float, late_w[:3])),
                    "tairte4_xyz": list(map(float, late_w[3:6])),
                    "sio2_xyz": (
                        list(map(float, late_w[6:9])) if include_substrate else None
                    ),
                },
                "incident_plane_W": float(p_inc),
                "near_phasor_closed_surface_inward_W": float(p_closed),
                "closed_surface_inward_W": float(p_closed_primary),
                "empty_subtracted_closed_surface_W": corrected_closed,
                "closure_correction": closure_correction,
                "Q_flux_closure_relative": closure,
                "late_window_relative_change": window_change,
                "gradient_l2_W": grad_l2,
            },
            "directions": rows,
            "direction_classification": {
                "strong_threshold_fraction_of_gradient_l2": strong_direction_threshold_fraction,
                "near_null_metric": "abs(AD-FD)/||gradient||_2",
                "no_empirical_gradient_rescaling": True,
            },
            "runtime": {
                "compile_seconds": compile_seconds,
                "ad_seconds": ad_seconds,
                "central_fd_forward_count": 2 * len(directions_np) * len(gradient_steps),
                "central_fd_forward_seconds": fd_seconds,
            },
            "gates": gates,
            "no_clipping_smoothing_gain_or_result_rescaling": True,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "fdtdx_production_width_nonuniform_au_gradient_smoke.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        with (output_dir / "fdtdx_production_width_nonuniform_au_gradient_smoke.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(result, indent=2))
        return result

    start = time.perf_counter()
    solve_jit = jax.jit(solve).lower(jnp.asarray((0.0, 0.0), dtype=jnp.float32)).compile()
    compile_seconds = time.perf_counter() - start
    cases: dict[str, dict[str, object]] = {}
    outputs = {}
    execution_seconds = 0.0
    endpoint_cases = (
        (("empty", (0.0, 0.0)),)
        if substrate_empty_only
        else (("empty", (0.0, 0.0)), ("au0", (1.0, 0.0)), ("au1", (1.0, 1.0)))
    )
    for name, strengths in endpoint_cases:
        start = time.perf_counter()
        solved = solve_jit(jnp.asarray(strengths, dtype=jnp.float32))
        execution_seconds += time.perf_counter() - start
        (
            late_scaled,
            previous_scaled,
            p_inc,
            p_closed,
            p_closed_deep,
            p_closed_deep_td,
            p_sio2_uniform,
            p_closed_uniform,
            p_closed_uniform_td,
            target,
        ) = solved
        late_w = np.asarray(late_scaled, dtype=np.float64) * POWER_SCALE_W
        previous_w = np.asarray(previous_scaled, dtype=np.float64) * POWER_SCALE_W
        cases[name] = {
            "strengths_tairte4_au": list(strengths),
            "component_power_W": {
                "au_xyz": list(map(float, late_w[:3])),
                "tairte4_xyz": list(map(float, late_w[3:6])),
                "sio2_xyz": (
                    list(map(float, late_w[6:9])) if include_substrate else None
                ),
            },
            "P_Q_W": float(late_w[-1]),
            "previous_P_Q_W": float(previous_w[-1]),
            "P_Q_window_relative_change": _relative(
                float(late_w[-1]), float(previous_w[-1])
            )
            if late_w[-1] != 0
            else 0.0,
            "incident_plane_signed_power_W": float(p_inc),
            "closed_surface_inward_W": float(p_closed),
            "deep_closed_surface_inward_W": float(p_closed_deep),
            "deep_time_domain_closed_surface_inward_W": float(p_closed_deep_td),
            "uniform_core_sio2_component_power_W": list(
                map(float, np.asarray(p_sio2_uniform) * POWER_SCALE_W)
            ),
            "uniform_core_sio2_P_Q_W": float(
                np.asarray(p_sio2_uniform).sum() * POWER_SCALE_W
            ),
            "uniform_core_closed_surface_inward_W": float(p_closed_uniform),
            "uniform_core_time_domain_closed_surface_inward_W": float(
                p_closed_uniform_td
            ),
        }
        outputs[name] = target

    if include_substrate and substrate_empty_only:
        case = cases["empty"]
        relative_errors = {
            "near_phasor_box": _relative(
                case["P_Q_W"], case["closed_surface_inward_W"]
            ),
            "deep_phasor_box": _relative(
                case["P_Q_W"], case["deep_closed_surface_inward_W"]
            ),
            "deep_time_domain_box": _relative(
                case["P_Q_W"], case["deep_time_domain_closed_surface_inward_W"]
            ),
            "deep_phasor_vs_time_domain": _relative(
                case["deep_closed_surface_inward_W"],
                case["deep_time_domain_closed_surface_inward_W"],
            ),
            "uniform_core_phasor_box": _relative(
                case["uniform_core_sio2_P_Q_W"],
                case["uniform_core_closed_surface_inward_W"],
            ),
            "uniform_core_time_domain_box": _relative(
                case["uniform_core_sio2_P_Q_W"],
                case["uniform_core_time_domain_closed_surface_inward_W"],
            ),
            "uniform_core_phasor_vs_time_domain": _relative(
                case["uniform_core_closed_surface_inward_W"],
                case["uniform_core_time_domain_closed_surface_inward_W"],
            ),
        }
        gates = {
            "gpu_only": True,
            "finite": bool(
                all(
                    np.isfinite(case[key])
                    for key in (
                        "P_Q_W",
                        "deep_time_domain_closed_surface_inward_W",
                        "incident_plane_signed_power_W",
                    )
                )
            ),
            "positive_total_absorption": case["P_Q_W"] > 0.0,
            "late_window_change_lt_0p5pct": (
                case["P_Q_window_relative_change"] < 0.005
            ),
            "deep_time_domain_Q_flux_closure_lt_0p5pct": (
                relative_errors["deep_time_domain_box"] < 0.005
            ),
            "no_clipping_smoothing_gain_or_result_rescaling": True,
        }
        diagnostic = {
            "status": (
                "VALIDATED_FDTDX_SUBSTRATE_ONLY_MATCHED_VOLUME_CLOSURE"
                if all(gates.values())
                else "FAILED_FDTDX_SUBSTRATE_ONLY_MATCHED_VOLUME_CLOSURE"
            ),
            "scope": "substrate-only flux-surface diagnostic; no TaIrTe4/Au/adjoint/thermal/PTE/electrical/optimization",
            "audit": audit,
            "case": case,
            "closure_contract": {
                "primary": "late-window material Q versus inward time-domain Poynting flux on the deep matched box",
                "phasor_boxes": "reported as detector-convergence diagnostics; not substituted for the primary time-domain balance",
            },
            "relative_errors": relative_errors,
            "gates": gates,
            "runtime": {
                "compile_seconds": compile_seconds,
                "one_case_execution_seconds": execution_seconds,
            },
            "no_clipping_smoothing_gain_or_result_rescaling": True,
        }
        (output_dir / "fdtdx_substrate_only_closure_diagnostic.json").write_text(
            json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(diagnostic, indent=2))
        return diagnostic

    if include_substrate:
        # The empty case retains the physical SiO2 layer.  Unlike the air-only
        # endpoint, its closed-box flux is physical and must not be subtracted
        # from the other cases.  Use direct matched-volume Poynting closure.
        substrate_only_q = cases["empty"]["P_Q_W"]
        substrate_only_closed = cases["empty"]["deep_time_domain_closed_surface_inward_W"]
        numerical_box_residual = substrate_only_closed - substrate_only_q
        closure: dict[str, dict[str, float]] = {}
        for name in ("empty", "au0", "au1"):
            corrected = cases[name]["deep_time_domain_closed_surface_inward_W"]
            closure[name] = {
                "deep_time_domain_closed_surface_W": corrected,
                "Q_flux_closure_relative": _relative(cases[name]["P_Q_W"], corrected),
                "near_phasor_Q_flux_closure_relative": _relative(
                    cases[name]["P_Q_W"], cases[name]["closed_surface_inward_W"]
                ),
                "deep_phasor_Q_flux_closure_relative": _relative(
                    cases[name]["P_Q_W"], cases[name]["deep_closed_surface_inward_W"]
                ),
            }
        finite = bool(
            all(
                np.isfinite(value)
                for case in cases.values()
                for value in (
                    case["P_Q_W"],
                    case["incident_plane_signed_power_W"],
                    case["deep_time_domain_closed_surface_inward_W"],
                )
            )
        )
        gates = {
            "gpu_only": True,
            "finite": finite,
            "positive_total_absorption": all(
                cases[name]["P_Q_W"] > 0.0 for name in ("empty", "au0", "au1")
            ),
            "each_late_window_change_lt_0p5pct": all(
                cases[name]["P_Q_window_relative_change"] < 0.005
                for name in ("empty", "au0", "au1")
            ),
            "each_Q_flux_closure_lt_0p5pct": all(
                closure[name]["Q_flux_closure_relative"] < 0.005
                for name in ("empty", "au0", "au1")
            ),
            "no_clipping_smoothing_gain_or_result_rescaling": True,
        }
        passed = all(gates.values())
        summary = {
            "status": (
                "VALIDATED_FDTDX_DIAGNOSTIC_SUBSTRATE_BINARY_ENDPOINT_CLOSURE"
                if passed
                else "FAILED_FDTDX_DIAGNOSTIC_SUBSTRATE_BINARY_ENDPOINT_CLOSURE"
            ),
            "scope": (
                "10-um Au/TaIrTe4/285-nm-SiO2/lossless-Si FDTDX diagnostic; "
                "not production while installed-Lumerical Palik Si readback remains blocked; "
                "no thermal/PTE/electrical/adjoint/optimization"
            ),
            "audit": audit,
            "materials": {
                "au_epsilon": [epsilon_au.real, epsilon_au.imag],
                "tairte4_epsilon": {
                    name: [value.real, value.imag] for name, value in epsilon_ta.items()
                },
                "sio2_epsilon": [epsilon_sio2.real, epsilon_sio2.imag],
                "si_epsilon": [epsilon_si.real, epsilon_si.imag],
                "axis_mapping": {"x": "b", "y": "a", "z": "c=b closure"},
                "material_Q_component_coordinates": (
                    "native Yee coordinates; exact_interpolation=False; "
                    "same 272x272 closed-volume footprint for SiO2"
                ),
            },
            "cases": cases,
            "closure": {
                "method": (
                    "direct matched-volume late-window P_Q versus deep-box inward "
                    "time-domain Poynting flux; no subtraction or rescaling"
                ),
                "substrate_only_Q_W": substrate_only_q,
                "substrate_only_closed_surface_W": substrate_only_closed,
                "substrate_only_closed_minus_Q_residual_W": numerical_box_residual,
                "cases": closure,
            },
            "runtime": {
                "compile_seconds": compile_seconds,
                "three_case_execution_seconds": execution_seconds,
            },
            "gates": gates,
        }
        summary_path = output_dir / "fdtdx_substrate_binary_endpoints_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return summary

    p_incident = abs(cases["empty"]["incident_plane_signed_power_W"])
    p_empty_box = cases["empty"]["closed_surface_inward_W"]
    lumerical = json.loads(LUMERICAL_SUMMARY.read_text(encoding="utf-8"))
    comparisons: dict[str, dict[str, float]] = {}
    for endpoint in ("0", "1"):
        name = f"au{endpoint}"
        p_q = cases[name]["P_Q_W"]
        corrected_box = cases[name]["closed_surface_inward_W"] - p_empty_box
        l_case = lumerical["cases"][endpoint]
        fdtdx_fraction = p_q / p_incident
        lumerical_fraction = l_case["P_Q_W"] / l_case["source_power_W"]
        comparisons[name] = {
            "fdtdx_absorbed_fraction": fdtdx_fraction,
            "lumerical_absorbed_fraction": lumerical_fraction,
            "absorbed_fraction_relative_difference": _relative(fdtdx_fraction, lumerical_fraction),
            "fdtdx_raw_closure_relative": _relative(
                p_q, cases[name]["closed_surface_inward_W"]
            ),
            "fdtdx_empty_subtracted_closure_relative": _relative(p_q, corrected_box),
            "fdtdx_empty_subtracted_closed_surface_W": corrected_box,
        }

    fdtdx_ratio = cases["au1"]["P_Q_W"] / cases["au0"]["P_Q_W"]
    lumerical_ratio = lumerical["cases"]["1"]["P_Q_W"] / lumerical["cases"]["0"]["P_Q_W"]
    ratio_error = _relative(fdtdx_ratio, lumerical_ratio)
    finite = bool(
        all(
            np.isfinite(value)
            for case in cases.values()
            for value in (
                case["P_Q_W"],
                case["incident_plane_signed_power_W"],
                case["closed_surface_inward_W"],
            )
        )
    )
    gates = {
        "gpu_only": True,
        "finite": finite,
        "positive_material_absorption": cases["au0"]["P_Q_W"] > 0 and cases["au1"]["P_Q_W"] > 0,
        "each_material_window_change_lt_0p5pct": all(
            cases[name]["P_Q_window_relative_change"] < 0.005 for name in ("au0", "au1")
        ),
        "each_absorbed_fraction_cross_solver_difference_lt_5pct": all(
            comparisons[name]["absorbed_fraction_relative_difference"] < 0.05
            for name in ("au0", "au1")
        ),
        "au1_over_au0_ratio_cross_solver_difference_lt_5pct": ratio_error < 0.05,
        "each_empty_subtracted_closure_lt_1pct": all(
            comparisons[name]["fdtdx_empty_subtracted_closure_relative"] < 0.01
            for name in ("au0", "au1")
        ),
    }
    passed = all(gates.values())
    summary = {
        "status": (
            "VALIDATED_FDTDX_LUMERICAL_BINARY_ENDPOINTS"
            if passed
            else "FAILED_FDTDX_LUMERICAL_BINARY_ENDPOINTS"
        ),
        "scope": "material-bearing production-width exact-binary optical endpoint cross-check; no thermal/PTE/adjoint/optimization",
        "audit": audit,
        "materials": {
            "au_epsilon": [epsilon_au.real, epsilon_au.imag],
            "tairte4_epsilon": {
                name: [value.real, value.imag] for name, value in epsilon_ta.items()
            },
            "axis_mapping": {"x": "b", "y": "a", "z": "c=b closure"},
            "fdtdx_field_unit_conversion": {
                "E_SI": "eta0 * E_internal",
                "H_SI": "H_internal",
                "S_SI": "eta0 * S_internal",
                "eta0_ohm": eta0,
            },
            "material_Q_component_coordinates": "native Yee coordinates; exact_interpolation=False",
            "no_clipping_smoothing_gain_or_rescaling": True,
        },
        "cases": cases,
        "comparisons": comparisons,
        "endpoint_ratio": {
            "fdtdx_au1_over_au0": fdtdx_ratio,
            "lumerical_au1_over_au0": lumerical_ratio,
            "relative_difference": ratio_error,
        },
        "runtime": {"compile_seconds": compile_seconds, "three_case_execution_seconds": execution_seconds},
        "gates": gates,
    }
    summary_path = output_dir / "fdtdx_lumerical_binary_endpoints_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_diagnostic_outputs(summary, output_dir)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--summarize-existing", action="store_true")
    parser.add_argument("--gradient-smoke", action="store_true")
    parser.add_argument("--spatial-q-export", action="store_true")
    parser.add_argument("--spatial-q-raw-path", type=Path)
    parser.add_argument("--spatial-q-weight-npz", type=Path)
    parser.add_argument("--spatial-q-weight-summary-json", type=Path)
    parser.add_argument(
        "--spatial-q-weight-scenario",
        choices=("thermally_grown", "evaporated"),
        default="thermally_grown",
    )
    parser.add_argument("--spatial-weighted-gradient-raw-path", type=Path)
    parser.add_argument("--gradient-checkpoints", type=int, default=16)
    parser.add_argument("--include-adjoint-aligned", action="store_true")
    parser.add_argument("--include-substrate", action="store_true")
    parser.add_argument("--matched-substrate-interface-grid", action="store_true")
    parser.add_argument("--substrate-empty-only", action="store_true")
    parser.add_argument("--substrate-total-periods", type=int)
    parser.add_argument("--substrate-window-periods", type=int)
    parser.add_argument(
        "--gradient-directions",
        default="smooth_asymmetric,fixed_seed_random",
        help="comma-separated subset of smooth_asymmetric,fixed_seed_random",
    )
    parser.add_argument(
        "--gradient-steps",
        default="0.01,0.005",
        help="comma-separated positive central-FD steps",
    )
    parser.add_argument(
        "--gradient-reference-json",
        type=Path,
        help="reuse stored AD directional values and execute forward FD only",
    )
    parser.add_argument(
        "--substrate-loss-representation",
        choices=("lorentz", "conductivity"),
        default="conductivity",
    )
    parser.add_argument(
        "--substrate-material-json",
        type=Path,
        default=SUBSTRATE_MATERIAL_JSON,
    )
    args = parser.parse_args()
    if args.summarize_existing:
        summary = json.loads(
            (args.output_dir / "fdtdx_lumerical_binary_endpoints_summary.json").read_text(encoding="utf-8")
        )
        _write_diagnostic_outputs(summary, args.output_dir)
        return 0
    result = run(
        args.output_dir,
        audit_only=args.audit_only,
        gradient_smoke=args.gradient_smoke or args.spatial_q_export,
        gradient_checkpoints=args.gradient_checkpoints,
        include_adjoint_aligned=args.include_adjoint_aligned,
        include_substrate=args.include_substrate,
        substrate_empty_only=args.substrate_empty_only,
        substrate_loss_representation=args.substrate_loss_representation,
        substrate_material_json=args.substrate_material_json,
        matched_substrate_interface_grid=args.matched_substrate_interface_grid,
        substrate_total_periods=args.substrate_total_periods,
        substrate_window_periods=args.substrate_window_periods,
        gradient_direction_names=tuple(
            name.strip() for name in args.gradient_directions.split(",") if name.strip()
        ),
        gradient_steps=tuple(
            float(value.strip())
            for value in args.gradient_steps.split(",")
            if value.strip()
        ),
        gradient_reference_json=args.gradient_reference_json,
        spatial_q_export=args.spatial_q_export,
        spatial_q_raw_path=args.spatial_q_raw_path,
        spatial_q_weight_npz=args.spatial_q_weight_npz,
        spatial_q_weight_summary_json=args.spatial_q_weight_summary_json,
        spatial_q_weight_scenario=args.spatial_q_weight_scenario,
        spatial_weighted_gradient_raw_path=args.spatial_weighted_gradient_raw_path,
    )
    if args.audit_only:
        return 0
    return 0 if result["status"].startswith("VALIDATED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
