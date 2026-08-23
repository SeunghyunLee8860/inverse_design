"""Shared 4 um FDTDX model for the dual-polarization Au design.

The builder is deliberately separate from the validated 10 um campaign.  It
uses the material values frozen by :mod:`01_probe_4um_materials`, keeps the
finite TaIrTe4 flake one micrometre away from the lateral PML, and changes
only the source polarization between the ``Ea`` and ``Eb`` cases.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)


HERE = Path(__file__).resolve().parent
FDTDX_SOURCE = Path(
    os.environ.get("FDTDX_SOURCE_DIR", "/home/seunghyun/.local/fdtdx_main_src")
)
MATERIAL_JSON = HERE / "results_materials_4um/4um_material_contract.json"
STAGE41 = (
    HERE.parent
    / "au_on_fixed_tairte4_validation"
    / "41_validate_au_on_fixed_tairte4_optical_adfd.py"
)
EPS0_F_PER_M = 8.8541878128e-12
C0_M_PER_S = 299_792_458.0


@dataclass(frozen=True)
class GridLayout:
    pml_cells: int = 8
    silicon_cells: int = 13
    sio2_cells: int = 3
    tairte4_cells: int = 5
    au_cells: int = 2
    flake_xy_cells: int = 160
    au_xy_cells: int = 80
    source_xy_cells: int = 160
    non_pml_xy_cells: int = 170
    source_z_start: int = 29
    target_z_start: int = 27
    incident_z_start: int = 28
    closed_z_start: int = 12
    closed_z_cells: int = 15


LAYOUT = GridLayout()


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _segments(parts: tuple[tuple[float, float, int], ...]) -> np.ndarray:
    values: list[np.ndarray] = []
    for index, (start, stop, cells) in enumerate(parts):
        if cells <= 0 or not stop > start:
            raise ValueError((start, stop, cells))
        segment = np.linspace(start, stop, cells + 1, dtype=np.float64)
        values.append(segment if index == 0 else segment[1:])
    result = np.concatenate(values)
    if np.any(np.diff(result) <= 0.0):
        raise RuntimeError("non-monotone rectilinear grid")
    return result


def grid_edges() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the frozen nonuniform 20 x 20 x 6 um optical grid."""

    lateral = _segments(
        (
            (-10.0e-6, -9.0e-6, 8),   # x/y PML
            (-9.0e-6, -8.0e-6, 5),    # air margin
            (-8.0e-6, 8.0e-6, 160),   # finite TaIrTe4 footprint
            (8.0e-6, 9.0e-6, 5),
            (9.0e-6, 10.0e-6, 8),
        )
    )
    vertical = _segments(
        (
            (-3.0e-6, -1.4e-6, 8),      # bottom z PML in Si
            (-1.4e-6, -0.385e-6, 5),    # resolved Si
            (-0.385e-6, -0.100e-6, 3),  # exact 285 nm SiO2
            (-0.100e-6, 0.0, 5),        # 100 nm TaIrTe4; dz=20 nm
            (0.0, 0.050e-6, 2),         # 50 nm Au; dz=25 nm
            (0.050e-6, 0.250e-6, 4),    # near-field air
            (0.250e-6, 0.750e-6, 2),
            (0.750e-6, 1.400e-6, 3),    # source begins at 0.75 um
            (1.400e-6, 3.000e-6, 8),    # top z PML
        )
    )
    return lateral, lateral.copy(), vertical


def load_material_contract() -> dict[str, Any]:
    payload = json.loads(MATERIAL_JSON.read_text(encoding="utf-8"))
    if payload.get("status") != "VALIDATED_4UM_SINGLE_FREQUENCY_MATERIAL_READBACK":
        raise RuntimeError(
            "4 um material contract is not promoted: " + str(payload.get("status"))
        )
    return payload


def _complex(item: dict[str, float]) -> complex:
    return complex(float(item["real"]), float(item["imag"]))


def polarization_vector(polarization: str) -> tuple[float, float, float]:
    if polarization == "Ea":
        return (0.0, 1.0, 0.0)
    if polarization == "Eb":
        return (1.0, 0.0, 0.0)
    raise ValueError(f"unknown polarization {polarization!r}")


def _slice(value: tuple[slice, slice, slice]) -> list[list[int]]:
    return [[int(part.start), int(part.stop)] for part in value]


def build_model(
    polarization: str,
    *,
    total_periods: int = 16,
    window_periods: int = 4,
    include_adjoint_source: bool = True,
    air_only_source_calibration: bool = False,
) -> dict[str, Any]:
    """Place the full optical model without running a time-domain solve."""

    if total_periods <= 2 * window_periods:
        raise ValueError("two disjoint phasor windows must fit in the solve")
    expected = FDTDX_SOURCE / "src"
    if str(expected) not in sys.path:
        sys.path.insert(0, str(expected))
    import jax
    import jax.numpy as jnp
    import fdtdx

    imported = Path(fdtdx.__file__).resolve()
    if expected.resolve() not in imported.parents:
        raise RuntimeError(f"unpinned FDTDX import: {imported}")

    stage41 = _load(STAGE41, "au_dualpol_stage41_fits")
    materials = load_material_contract()
    x_edges, y_edges, z_edges = grid_edges()
    grid = fdtdx.RectilinearGrid.custom(
        x_edges=x_edges, y_edges=y_edges, z_edges=z_edges
    )

    def real_coordinate_constraint(
        obj: object,
        axes: tuple[int, ...],
        sides: tuple[str, ...],
        indices: tuple[int, ...],
    ) -> object:
        """Align fixed-cell objects to exact custom-grid edges.

        ``set_grid_coordinates`` is intentionally unavailable on a nonuniform
        FDTDX grid.  The requested indices below are converted once to their
        exact physical edge coordinates; realized slices are audited after
        placement.
        """

        edge_arrays = (x_edges, y_edges, z_edges)
        return fdtdx.RealCoordinateConstraint(
            object=obj.name,
            axes=axes,
            sides=sides,
            coordinates=tuple(
                float(edge_arrays[axis][index])
                for axis, index in zip(axes, indices, strict=True)
            ),
        )
    period_s = CONTRACT.wavelength_m / C0_M_PER_S
    config = fdtdx.SimulationConfig(
        grid=grid,
        time=total_periods * period_s,
        dtype=jnp.float32,
        courant_factor=0.5,
        backend="gpu",
        gradient_config=None,
    )
    dt = float(config.time_step_duration)
    omega = 2.0 * math.pi * C0_M_PER_S / CONTRACT.wavelength_m
    epsilon_au = _complex(materials["materials"]["Au"]["epsilon"])
    epsilon_ta = {
        axis: _complex(materials["materials"]["TaIrTe4"][axis]["epsilon"])
        for axis in ("a", "b", "c")
    }
    epsilon_sio2 = _complex(materials["materials"]["SiO2"]["epsilon"])
    epsilon_si = _complex(materials["materials"]["Si"]["epsilon"])
    fits = {
        "au": stage41._drude_fit(epsilon_au, omega, dt),
        "a": stage41._drude_fit(epsilon_ta["a"], omega, dt),
        "b": stage41._lorentz_fit(epsilon_ta["b"], omega, dt),
    }
    fits["c"] = dict(fits["b"])
    coefficients = {
        name: stage41._coefficient_triplet(value, dt)
        for name, value in fits.items()
    }

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
            thickness_grid_minx=LAYOUT.pml_cells,
            thickness_grid_maxx=LAYOUT.pml_cells,
            thickness_grid_miny=LAYOUT.pml_cells,
            thickness_grid_maxy=LAYOUT.pml_cells,
            thickness_grid_minz=LAYOUT.pml_cells,
            thickness_grid_maxz=LAYOUT.pml_cells,
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
    ta_placeholder = fdtdx.DispersionModel(
        poles=(
            fdtdx.DrudePole(
                plasma_frequency=fits["a"]["omega_p_rad_s"],
                damping=fits["a"]["gamma_rad_s"],
            ),
        )
    )
    silicon = fdtdx.UniformMaterialObject(
        name="fixed_silicon_substrate",
        partial_grid_shape=(None, None, LAYOUT.silicon_cells),
        material=fdtdx.Material(
            permittivity=(1.0 if air_only_source_calibration else float(epsilon_si.real))
        ),
    )
    sio2 = fdtdx.UniformMaterialObject(
        name="fixed_285nm_sio2",
        partial_grid_shape=(None, None, LAYOUT.sio2_cells),
        material=fdtdx.Material(
            permittivity=(1.0 if air_only_source_calibration else float(epsilon_sio2.real))
        ),
    )
    flake = fdtdx.UniformMaterialObject(
        name="fixed_tairte4",
        partial_grid_shape=(
            LAYOUT.flake_xy_cells,
            LAYOUT.flake_xy_cells,
            LAYOUT.tairte4_cells,
        ),
        material=fdtdx.Material(permittivity=stage41.EPS_INF, dispersion=ta_placeholder),
    )
    au = fdtdx.UniformMaterialObject(
        name="au_design",
        partial_grid_shape=(LAYOUT.au_xy_cells, LAYOUT.au_xy_cells, LAYOUT.au_cells),
        material=fdtdx.Material(permittivity=stage41.EPS_INF, dispersion=au_model),
    )
    constraints.extend(
        [
            silicon.place_relative_to(
                volume, axes=(2,), own_positions=(-1,), other_positions=(-1,)
            ),
            sio2.place_above(silicon),
            flake.place_at_center(volume, axes=(0, 1)),
            flake.place_above(sio2),
            au.place_at_center(volume, axes=(0, 1)),
            au.place_above(flake),
        ]
    )
    objects.extend((silicon, sio2, flake, au))

    source_radius = 0.5 * CONTRACT.source_aperture_span_m
    source = fdtdx.GaussianPlaneSource(
        name="gaussian_source",
        partial_grid_shape=(LAYOUT.source_xy_cells, LAYOUT.source_xy_cells, 1),
        fixed_E_polarization_vector=polarization_vector(polarization),
        wave_character=fdtdx.WaveCharacter(wavelength=CONTRACT.wavelength_m),
        radius=source_radius,
        std=CONTRACT.gaussian_waist_m / (math.sqrt(2.0) * source_radius),
        direction="-",
    )
    constraints.extend(
        [
            source.place_at_center(volume, axes=(0, 1)),
            real_coordinate_constraint(
                source, (2,), ("-",), (LAYOUT.source_z_start,)
            ),
        ]
    )
    objects.append(source)

    wave = fdtdx.WaveCharacter(wavelength=CONTRACT.wavelength_m)
    previous = fdtdx.OnOffSwitch(
        start_time=(total_periods - 2 * window_periods) * period_s,
        end_time=(total_periods - window_periods) * period_s,
    )
    late = fdtdx.OnOffSwitch(
        start_time=(total_periods - window_periods) * period_s
    )
    simulation_end_s = (config.time_steps_total - 1) * dt
    phasor_windows = {
        "previous": fdtdx.TukeyWindow(
            start_time=(total_periods - 2 * window_periods) * period_s,
            end_time=(total_periods - window_periods) * period_s,
            alpha=1.0,
        ),
        "late": fdtdx.TukeyWindow(
            start_time=(total_periods - window_periods) * period_s,
            end_time=simulation_end_s,
            alpha=1.0,
        ),
    }
    for material_name, target in (("au", au), ("tairte4", flake)):
        for window_name, switch in (("previous", previous), ("late", late)):
            detector = fdtdx.PhasorDetector(
                name=f"{material_name}_{window_name}",
                partial_grid_shape=target.partial_grid_shape,
                wave_characters=(wave,),
                components=("Ex", "Ey", "Ez"),
                dtype=jnp.complex64,
                switch=switch,
                apodization=phasor_windows[window_name],
                exact_interpolation=False,
                plot=False,
            )
            constraints.append(detector.same_position(target))
            objects.append(detector)

    incident = fdtdx.PhasorPoyntingFluxDetector(
        name="incident_plane",
        partial_grid_shape=(LAYOUT.source_xy_cells, LAYOUT.source_xy_cells, 1),
        wave_characters=(wave,),
        direction="-",
        dtype=jnp.complex64,
        switch=late,
        apodization=phasor_windows["late"],
        exact_interpolation=True,
    )
    target = fdtdx.PhasorDetector(
        name="target_field",
        partial_grid_shape=(LAYOUT.source_xy_cells, LAYOUT.source_xy_cells, 1),
        wave_characters=(wave,),
        components=("Ex", "Ey", "Ez"),
        dtype=jnp.complex64,
        switch=late,
        apodization=phasor_windows["late"],
        exact_interpolation=True,
        plot=False,
    )
    for detector, z_start in (
        (incident, LAYOUT.incident_z_start),
        (target, LAYOUT.target_z_start),
    ):
        constraints.extend(
            [
                detector.place_at_center(volume, axes=(0, 1)),
                real_coordinate_constraint(detector, (2,), ("-",), (z_start,)),
            ]
        )
        objects.append(detector)

    closed = fdtdx.ClosedSurfacePhasorPoyntingFluxDetector(
        name="material_flux",
        partial_grid_shape=(
            LAYOUT.non_pml_xy_cells,
            LAYOUT.non_pml_xy_cells,
            LAYOUT.closed_z_cells,
        ),
        wave_characters=(wave,),
        orientation="inward",
        dtype=jnp.complex64,
        switch=late,
        apodization=phasor_windows["late"],
        exact_interpolation=True,
    )
    closed_td = fdtdx.ClosedSurfacePoyntingFluxDetector(
        name="material_flux_td",
        partial_grid_shape=(
            LAYOUT.non_pml_xy_cells,
            LAYOUT.non_pml_xy_cells,
            LAYOUT.closed_z_cells,
        ),
        orientation="inward",
        switch=late,
    )
    for detector in (closed, closed_td):
        constraints.append(
            real_coordinate_constraint(
                detector,
                (0, 1, 2),
                ("-", "-", "-"),
                (
                    LAYOUT.pml_cells,
                    LAYOUT.pml_cells,
                    LAYOUT.closed_z_start,
                ),
            )
        )
        objects.append(detector)

    if include_adjoint_source:
        from photothermal_pte.optimization_runs.au_on_fixed_tairte4_validation.fdtdx_two_solve_adjoint import (
            DistributedElectricCurrentSource,
        )

        adjoint_shape = (
            LAYOUT.flake_xy_cells,
            LAYOUT.flake_xy_cells,
            LAYOUT.sio2_cells + LAYOUT.tairte4_cells + LAYOUT.au_cells,
        )
        adjoint = DistributedElectricCurrentSource(
            name="distributed_adjoint_source",
            partial_grid_shape=adjoint_shape,
            wave_character=wave,
            temporal_profile=fdtdx.SingleFrequencyProfile(
                phase_shift=0.0, num_startup_periods=4
            ),
            complex_profile=jnp.zeros((3, *adjoint_shape), dtype=jnp.complex64),
            static_amplitude_factor=0.0,
        )
        constraints.extend(
            [
                adjoint.place_at_center(volume, axes=(0, 1)),
                real_coordinate_constraint(
                    adjoint,
                    (2,),
                    ("-",),
                    (LAYOUT.silicon_cells,),
                ),
            ]
        )
        objects.append(adjoint)

    key = jax.random.PRNGKey(20260823)
    placed, base, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    base, placed, _ = fdtdx.apply_params(base, placed, params, key)
    realized = config.resolved_grid
    if realized is None:
        raise RuntimeError("FDTDX did not resolve the grid")

    slices = {
        name: placed[name].grid_slice
        for name in (
            "fixed_silicon_substrate",
            "fixed_285nm_sio2",
            "fixed_tairte4",
            "au_design",
            "gaussian_source",
            "incident_plane",
            "target_field",
            "material_flux",
            "material_flux_td",
        )
    }
    if include_adjoint_source:
        slices["distributed_adjoint_source"] = placed[
            "distributed_adjoint_source"
        ].grid_slice
    if slices["fixed_tairte4"][2].stop != slices["au_design"][2].start:
        raise RuntimeError("Au and TaIrTe4 are not face adjacent")
    if slices["fixed_285nm_sio2"][2].stop != slices["fixed_tairte4"][2].start:
        raise RuntimeError("TaIrTe4 and SiO2 are not face adjacent")

    spatial_shape = base.dispersive_c1.shape[-3:]
    fixed_c1 = jnp.zeros((1, 3, *spatial_shape), dtype=jnp.float32)
    fixed_c2 = jnp.zeros_like(fixed_c1)
    fixed_c3 = jnp.zeros_like(fixed_c1)
    ta_slice = slices["fixed_tairte4"]
    au_slice = slices["au_design"]
    if not air_only_source_calibration:
        for component, axis in enumerate(("b", "a", "c")):
            c1, c2, c3 = coefficients[axis]
            fixed_c1 = fixed_c1.at[(0, component, *ta_slice)].set(c1)
            fixed_c2 = fixed_c2.at[(0, component, *ta_slice)].set(c2)
            fixed_c3 = fixed_c3.at[(0, component, *ta_slice)].set(c3)
        au_c1, au_c2, _ = coefficients["au"]
        for component in range(3):
            fixed_c1 = fixed_c1.at[(0, component, *au_slice)].set(au_c1)
            fixed_c2 = fixed_c2.at[(0, component, *au_slice)].set(au_c2)

    return {
        "jax": jax,
        "jnp": jnp,
        "fdtdx": fdtdx,
        "key": key,
        "config": config,
        "grid": realized,
        "placed": placed,
        "base": base,
        "slices": slices,
        "fixed_c1": fixed_c1,
        "fixed_c2": fixed_c2,
        "fixed_c3": fixed_c3,
        "coefficients": coefficients,
        "fits": fits,
        "epsilon": {
            "au": epsilon_au,
            "tairte4": epsilon_ta,
            "sio2": epsilon_sio2,
            "silicon": epsilon_si,
        },
        "omega_rad_s": omega,
        "polarization": polarization,
        "air_only_source_calibration": bool(air_only_source_calibration),
        "placement": {name: _slice(value) for name, value in slices.items()},
    }
