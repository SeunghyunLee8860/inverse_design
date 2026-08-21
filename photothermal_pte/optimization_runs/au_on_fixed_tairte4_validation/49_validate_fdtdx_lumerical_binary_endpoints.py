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


def _grid_edges() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lateral = _piecewise_edges(
        (-24e-6, -12e-6, 12e-6, 24e-6),
        (500e-9, 100e-9, 500e-9),
    )
    vertical = _piecewise_edges(
        (-8e-6, -0.2e-6, 0.2e-6, 8e-6),
        (200e-9, 25e-9, 200e-9),
    )
    return lateral, lateral.copy(), vertical


def _slice_tuple(value: tuple[slice, slice, slice]) -> list[list[int]]:
    return [[int(part.start), int(part.stop)] for part in value]


def _relative(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


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


def run(output_dir: Path, *, audit_only: bool) -> dict[str, object]:
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
    x_edges, y_edges, z_edges = _grid_edges()
    grid = fdtdx.RectilinearGrid.custom(x_edges=x_edges, y_edges=y_edges, z_edges=z_edges)
    period_s = WAVELENGTH_M / stage41.C0_M_PER_S
    total_periods = 8
    window_periods = 2
    config = fdtdx.SimulationConfig(
        grid=grid,
        time=total_periods * period_s,
        dtype=jnp.float32,
        courant_factor=0.5,
        backend="gpu",
        gradient_config=None,
    )
    dt = config.time_step_duration
    omega = 2.0 * math.pi * stage41.C0_M_PER_S / WAVELENGTH_M
    epsilon_au = complex(stage41.AU_N, stage41.AU_K) ** 2
    epsilon_ta = stage41._load_tairte4_epsilon()
    fits = {
        "au": stage41._drude_fit(epsilon_au, omega, dt),
        "a": stage41._drude_fit(epsilon_ta["a"], omega, dt),
        "b": stage41._lorentz_fit(epsilon_ta["b"], omega, dt),
    }
    fits["c"] = dict(fits["b"])
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
    flake = fdtdx.UniformMaterialObject(
        name="fixed_tairte4",
        partial_grid_shape=(200, 200, 4),
        material=fdtdx.Material(permittivity=stage41.EPS_INF, dispersion=ta_model),
    )
    au = fdtdx.UniformMaterialObject(
        name="exact_binary_au",
        partial_grid_shape=(100, 100, 2),
        material=fdtdx.Material(permittivity=stage41.EPS_INF, dispersion=au_model),
    )
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
    for material_name, target in (("au", au), ("tairte4", flake)):
        for window_name, switch in (("previous", previous), ("late", late)):
            detector = fdtdx.PhasorDetector(
                name=f"{material_name}_{window_name}",
                partial_grid_shape=target.partial_grid_shape,
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

    closed = fdtdx.ClosedSurfacePhasorPoyntingFluxDetector(
        name="material_flux",
        partial_grid_shape=(220, 220, 16),
        wave_characters=(wave,),
        orientation="inward",
        dtype=jnp.complex64,
        switch=late,
        exact_interpolation=True,
    )
    constraints.extend(
        [
            closed.place_at_center(volume, axes=(0, 1)),
            closed.place_at_center(volume, axes=(2,), margins=(0.0,)),
        ]
    )
    objects.append(closed)

    key = jax.random.PRNGKey(20260821)
    placed, base, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    base, placed, _ = fdtdx.apply_params(base, placed, params, key)
    flake_slice = placed["fixed_tairte4"].grid_slice
    au_slice = placed["exact_binary_au"].grid_slice
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
        c3 = ta_strength * ta_c3 + au_strength * au_c3_template
        return (
            base.reset()
            .aset("dispersive_c1", fixed_c1)
            .aset("dispersive_c2", fixed_c2)
            .aset("dispersive_c3", c3)
        )

    au_volume = jnp.asarray(realized.cell_volume(tuple((x[0], x[1]) for x in _slice_tuple(au_slice))))
    ta_volume = jnp.asarray(realized.cell_volume(tuple((x[0], x[1]) for x in _slice_tuple(flake_slice))))
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
            jnp.abs(e_au) ** 2 * au_volume[None, ...], axis=(1, 2, 3)
        )
        p_ta_comp = prefactor * ta_strength * jnp.sum(
            ta_imag * jnp.abs(e_ta) ** 2 * ta_volume[None, ...], axis=(1, 2, 3)
        )
        return jnp.concatenate((p_au_comp, p_ta_comp, jnp.stack((p_au_comp.sum() + p_ta_comp.sum(),))))

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
        target = out.detector_states["target_field"]["phasor"][0, 0]
        return late_power, previous_power, p_inc, p_closed, target

    start = time.perf_counter()
    solve_jit = jax.jit(solve).lower(jnp.asarray((0.0, 0.0), dtype=jnp.float32)).compile()
    compile_seconds = time.perf_counter() - start
    cases: dict[str, dict[str, object]] = {}
    outputs = {}
    execution_seconds = 0.0
    for name, strengths in (("empty", (0.0, 0.0)), ("au0", (1.0, 0.0)), ("au1", (1.0, 1.0))):
        start = time.perf_counter()
        solved = solve_jit(jnp.asarray(strengths, dtype=jnp.float32))
        execution_seconds += time.perf_counter() - start
        late_scaled, previous_scaled, p_inc, p_closed, target = solved
        late_w = np.asarray(late_scaled, dtype=np.float64) * POWER_SCALE_W
        previous_w = np.asarray(previous_scaled, dtype=np.float64) * POWER_SCALE_W
        cases[name] = {
            "strengths_tairte4_au": list(strengths),
            "component_power_W": {
                "au_xyz": list(map(float, late_w[:3])),
                "tairte4_xyz": list(map(float, late_w[3:6])),
            },
            "P_Q_W": float(late_w[6]),
            "previous_P_Q_W": float(previous_w[6]),
            "P_Q_window_relative_change": _relative(float(late_w[6]), float(previous_w[6]))
            if late_w[6] != 0
            else 0.0,
            "incident_plane_signed_power_W": float(p_inc),
            "closed_surface_inward_W": float(p_closed),
        }
        outputs[name] = target

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
    args = parser.parse_args()
    if args.summarize_existing:
        summary = json.loads(
            (args.output_dir / "fdtdx_lumerical_binary_endpoints_summary.json").read_text(encoding="utf-8")
        )
        _write_diagnostic_outputs(summary, args.output_dir)
        return 0
    result = run(args.output_dir, audit_only=args.audit_only)
    if args.audit_only:
        return 0
    return 0 if result["status"].startswith("VALIDATED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
