#!/usr/bin/env python3
"""Production-size checkpoint-free two-solve Maxwell-gradient equivalence.

This runner keeps the frozen 48 um substrate-bearing optical contract and the
validated native-Yee thermal-source weights.  It performs exactly one forward
and one reciprocal adjoint solve.  It never differentiates through the time
loop and stores no field history or checkpoints.  The resulting 20x20 Maxwell
source gradient is compared with the immutable checkpointed reference vector.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
STAGE49 = HERE / "49_validate_fdtdx_lumerical_binary_endpoints.py"
TWO_SOLVE = HERE / "fdtdx_two_solve_adjoint.py"
DYNAMIC_PTE = HERE / "fdtdx_dynamic_pte.py"
STAGE67 = HERE / "67_validate_explicit_thermal_weighting_fixed_spatial_q_adfd.py"
DEFAULT_WEIGHTS = Path(
    "/home/seunghyun/tairte4/raw_artifacts/native_yee_thermal_source_adjoint_weights/"
    "native_yee_thermal_source_adjoint_weights.npz"
)
DEFAULT_WEIGHT_SUMMARY = (
    HERE
    / "results_native_yee_thermal_source_adjoint_weights"
    / "native_yee_thermal_source_adjoint_weights_summary.json"
)
DEFAULT_REFERENCE = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "fdtdx_spatially_weighted_pte_source_gradient_16period/"
    "fdtdx_spatially_weighted_pte_source_gradient_raw.npz"
)
DEFAULT_REFERENCE_SUMMARY = (
    HERE
    / "results_fdtdx_spatially_weighted_pte_source_gradient_16period_gpu3"
    / "fdtdx_spatially_weighted_pte_source_gradient_summary.json"
)
DEFAULT_OUTPUT = HERE / "results_fdtdx_production_two_solve_equivalence"
DEFAULT_RAW_OUTPUT = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "fdtdx_production_two_solve_equivalence/"
    "fdtdx_production_two_solve_gradient.npz"
)
POWER_SCALE_W = 1.0e-24


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace_named(objects, name: str, replacement):
    result = objects.copy()
    values = list(result.object_list)
    values[result.index(name)] = replacement
    return result.aset("object_list", values)


def _block_reduce(values, latent_shape=(20, 20), repeat=5):
    return values.reshape(latent_shape[0], repeat, latent_shape[1], repeat).sum(
        axis=(1, 3)
    )


def run(
    output_dir: Path,
    weight_path: Path,
    weight_summary_path: Path,
    reference_path: Path,
    reference_summary_path: Path,
    raw_gradient_path: Path,
    scenario: str,
    audit_only: bool,
    dynamic_pte_weights: bool,
    cuda_device: int,
) -> dict[str, object]:
    pipeline_start = time.perf_counter()
    import jax
    import jax.numpy as jnp
    import fdtdx

    stage49 = _load("stage49_two_solve", STAGE49)
    stage41 = stage49._load_stage41()
    two_solve = _load("fdtdx_two_solve_production", TWO_SOLVE)
    devices = jax.devices()
    if not devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"GPU-only contract violated: {devices}")
    if scenario not in ("thermally_grown", "evaporated"):
        raise ValueError(scenario)

    weight_summary = json.loads(weight_summary_path.read_text(encoding="utf-8"))
    reference_summary = json.loads(reference_summary_path.read_text(encoding="utf-8"))
    weight_sha = _sha256(weight_path)
    reference_sha = _sha256(reference_path)
    if weight_summary.get("status") != "VALIDATED_NATIVE_YEE_THERMAL_SOURCE_ADJOINT_PULLBACK":
        raise RuntimeError("Fail-closed weight status mismatch")
    if weight_sha != weight_summary["raw_artifact"]["sha256"]:
        raise RuntimeError("Fail-closed weight SHA mismatch")
    if reference_summary.get("status") != "VALIDATED_FDTDX_NATIVE_YEE_SPATIALLY_WEIGHTED_PTE_SOURCE_GRADIENT":
        raise RuntimeError("Fail-closed checkpointed reference status mismatch")
    if reference_sha != reference_summary["raw_artifact"]["sha256"]:
        raise RuntimeError("Fail-closed checkpointed gradient SHA mismatch")
    if reference_summary["spatial_weight"]["raw_sha256"] != weight_sha:
        raise RuntimeError("Fail-closed reference/weight dependency mismatch")

    with np.load(weight_path, allow_pickle=False) as raw:
        rho_np = np.asarray(raw["rho"], dtype=np.float32)
        weights_np = {
            material: np.stack(
                [
                    np.asarray(
                        raw[f"weight_{scenario}_{material}_{component}_A_W"],
                        dtype=np.float32,
                    )
                    for component in "xyz"
                ]
            )
            for material in ("au", "tairte4", "sio2")
        }
    with np.load(reference_path, allow_pickle=False) as raw:
        reference_rho = np.asarray(raw["rho"], dtype=np.float32)
        reference_gradient = np.asarray(raw["gradient_A"], dtype=np.float64)
        reference_objective = float(raw["weighted_objective_A"])
    if not np.array_equal(rho_np, reference_rho):
        raise RuntimeError("Fail-closed density mismatch between weight and reference")
    if rho_np.shape != (20, 20) or reference_gradient.shape != (20, 20):
        raise RuntimeError("Unexpected production density/gradient shape")

    x_edges, y_edges, z_edges = stage49._grid_edges(
        include_substrate=True, matched_substrate_interface=True
    )
    grid = fdtdx.RectilinearGrid.custom(
        x_edges=x_edges, y_edges=y_edges, z_edges=z_edges
    )
    period_s = stage49.WAVELENGTH_M / stage41.C0_M_PER_S
    total_periods = 16
    window_periods = 4
    config = fdtdx.SimulationConfig(
        grid=grid,
        time=total_periods * period_s,
        dtype=jnp.float32,
        courant_factor=0.5,
        backend="gpu",
        gradient_config=None,
    )
    dt = config.time_step_duration
    omega = 2.0 * math.pi * stage41.C0_M_PER_S / stage49.WAVELENGTH_M
    epsilon_au = complex(stage41.AU_N, stage41.AU_K) ** 2
    epsilon_ta = stage41._load_tairte4_epsilon()
    epsilon_sio2, epsilon_si, substrate_provenance = stage49._load_substrate_contract(
        stage49.SUBSTRATE_MATERIAL_JSON
    )
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
    silicon = fdtdx.UniformMaterialObject(
        name="fixed_silicon_substrate",
        partial_grid_shape=(None, None, 48),
        material=fdtdx.Material(permittivity=float(epsilon_si.real)),
    )
    sio2_sigma = omega * stage41.EPS0_F_PER_M * epsilon_sio2.imag
    sio2 = fdtdx.UniformMaterialObject(
        name="fixed_285nm_sio2",
        partial_grid_shape=(None, None, 19),
        material=fdtdx.Material(
            permittivity=float(epsilon_sio2.real),
            electric_conductivity=float(sio2_sigma),
        ),
    )
    flake = fdtdx.UniformMaterialObject(
        name="fixed_tairte4",
        partial_grid_shape=(200, 200, 5),
        material=fdtdx.Material(permittivity=stage41.EPS_INF, dispersion=ta_model),
    )
    au = fdtdx.UniformMaterialObject(
        name="exact_binary_au",
        partial_grid_shape=(100, 100, 2),
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

    source_radius = 20.0e-6
    source = fdtdx.GaussianPlaneSource(
        name="gaussian_source",
        partial_grid_shape=(272, 272, 1),
        fixed_E_polarization_vector=(1.0, 0.0, 0.0),
        wave_character=fdtdx.WaveCharacter(wavelength=stage49.WAVELENGTH_M),
        radius=source_radius,
        std=stage49.W0_M / (math.sqrt(2.0) * source_radius),
        direction="-",
    )
    constraints.extend(
        [
            source.place_at_center(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(2,), margins=(1.0e-6,)),
        ]
    )
    objects.append(source)

    adjoint_shape = (272, 272, 26)
    adjoint_source = two_solve.DistributedElectricCurrentSource(
        name="distributed_adjoint_source",
        partial_grid_shape=adjoint_shape,
        wave_character=fdtdx.WaveCharacter(wavelength=stage49.WAVELENGTH_M),
        temporal_profile=fdtdx.SingleFrequencyProfile(
            phase_shift=0.0, num_startup_periods=4
        ),
        complex_profile=jnp.zeros((3, *adjoint_shape), dtype=jnp.complex64),
        static_amplitude_factor=0.0,
    )
    constraints.extend(
        [
            adjoint_source.place_at_center(volume, axes=(0, 1)),
            adjoint_source.place_relative_to(
                volume,
                axes=(2,),
                own_positions=(-1,),
                other_positions=(-1,),
                margins=(float(z_edges[48] - z_edges[0]),),
            ),
        ]
    )
    objects.append(adjoint_source)

    wave = fdtdx.WaveCharacter(wavelength=stage49.WAVELENGTH_M)
    previous = fdtdx.OnOffSwitch(
        start_time=(total_periods - 2 * window_periods) * period_s,
        end_time=(total_periods - window_periods) * period_s,
    )
    late = fdtdx.OnOffSwitch(
        start_time=(total_periods - window_periods) * period_s
    )
    for material_name, target, detector_shape in (
        ("au", au, au.partial_grid_shape),
        ("tairte4", flake, flake.partial_grid_shape),
        ("sio2", sio2, (272, 272, 19)),
    ):
        for window_name, switch in (("previous", previous), ("late", late)):
            detector = fdtdx.PhasorDetector(
                name=f"{material_name}_{window_name}",
                partial_grid_shape=detector_shape,
                wave_characters=(wave,),
                components=("Ex", "Ey", "Ez"),
                dtype=jnp.complex64,
                switch=switch,
                exact_interpolation=False,
                plot=False,
            )
            constraints.append(detector.same_position(target))
            objects.append(detector)

    key = jax.random.PRNGKey(20260821)
    placed, base, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    base, placed, _ = fdtdx.apply_params(base, placed, params, key)
    realized = config.resolved_grid
    if realized is None:
        raise RuntimeError("Missing realized grid")
    au_slice = placed["exact_binary_au"].grid_slice
    ta_slice = placed["fixed_tairte4"].grid_slice
    sio2_detector_slice = placed["sio2_late"].grid_slice
    adjoint_slice = placed["distributed_adjoint_source"].grid_slice
    expected_slices = {
        "au": ((94, 194), (94, 194), (72, 74)),
        "tairte4": ((44, 244), (44, 244), (67, 72)),
        "sio2": ((8, 280), (8, 280), (48, 67)),
        "adjoint": ((8, 280), (8, 280), (48, 74)),
    }
    actual_slices = {
        name: tuple((int(p.start), int(p.stop)) for p in value)
        for name, value in {
            "au": au_slice,
            "tairte4": ta_slice,
            "sio2": sio2_detector_slice,
            "adjoint": adjoint_slice,
        }.items()
    }
    if actual_slices != expected_slices:
        raise RuntimeError(
            f"Fail-closed production placement mismatch: {actual_slices}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "status": "AUDITED_FDTDX_PRODUCTION_CHECKPOINT_FREE_TWO_SOLVE_RUNSETUP",
        "gpu_only": True,
        "gradient_config": None,
        "checkpoint_count": 0,
        "time_history_saved": False,
        "wavelength_m": stage49.WAVELENGTH_M,
        "waist_m": stage49.W0_M,
        "grid_shape_xyz": list(realized.shape),
        "grid_cells": int(np.prod(realized.shape)),
        "grid_bounds_m_xyz": [
            [float(realized.edges(axis)[0]), float(realized.edges(axis)[-1])]
            for axis in range(3)
        ],
        "min_spacing_m_xyz": list(realized.min_spacings),
        "max_spacing_m_xyz": [
            float(np.max(np.asarray(realized.cell_widths(axis))))
            for axis in range(3)
        ],
        "pml_cells_each_face": 8,
        "total_periods": total_periods,
        "window_periods": window_periods,
        "time_steps_total": config.time_steps_total,
        "placement": actual_slices,
        "adjoint_complex_profile_shape": [3, *adjoint_shape],
        "adjoint_complex_profile_bytes": int(
            3 * np.prod(adjoint_shape) * np.dtype(np.complex64).itemsize
        ),
        "weight_sha256": weight_sha,
        "reference_sha256": reference_sha,
        "scenario": scenario,
    }
    audit_path = output_dir / "fdtdx_production_two_solve_runsetup_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if audit_only:
        print(json.dumps(audit, indent=2), flush=True)
        return audit

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

    rho = jnp.asarray(rho_np)
    weights = {
        material: jnp.asarray(value) for material, value in weights_np.items()
    }

    def optical_strength(density):
        upsampled = jnp.repeat(jnp.repeat(density, 5, axis=0), 5, axis=1)
        return jnp.broadcast_to((upsampled**3)[:, :, None], (100, 100, 2))

    def arrays_for_density(density):
        c3 = fixed_c3
        strength = optical_strength(density)
        for component in range(3):
            c3 = c3.at[(0, component, *au_slice)].set(au_c3 * strength)
        return (
            base.reset()
            .aset("dispersive_c1", fixed_c1)
            .aset("dispersive_c2", fixed_c2)
            .aset("dispersive_c3", c3)
        )

    volumes = {
        "au": stage49._electric_yee_dual_volumes(realized, au_slice),
        "tairte4": stage49._electric_yee_dual_volumes(realized, ta_slice),
        "sio2": stage49._electric_yee_dual_volumes(realized, sio2_detector_slice),
    }
    expected_shapes = {
        material: tuple(volumes[material].shape)
        for material in ("au", "tairte4", "sio2")
    }
    if any(tuple(weights[name].shape) != expected_shapes[name] for name in expected_shapes):
        raise RuntimeError("Fail-closed native-Yee weight shape mismatch")
    eta0 = float(fdtdx.constants.eta0)
    prefactor = (
        0.5 * omega * stage41.EPS0_F_PER_M * eta0**2 / POWER_SCALE_W
    )
    ta_imag = jnp.asarray(
        [epsilon_ta["b"].imag, epsilon_ta["a"].imag, epsilon_ta["c"].imag],
        dtype=jnp.float32,
    )[:, None, None, None]

    def fields(out, window: str, density):
        e_au = out.detector_states[f"au_{window}"]["phasor"][0, 0]
        e_ta = out.detector_states[f"tairte4_{window}"]["phasor"][0, 0]
        e_sio2 = out.detector_states[f"sio2_{window}"]["phasor"][0, 0]
        return e_au, e_ta, e_sio2, optical_strength(density)

    def objective(out, window: str, density):
        e_au, e_ta, e_sio2, strength = fields(out, window, density)
        return prefactor * (
            epsilon_au.imag
            * jnp.sum(weights["au"] * volumes["au"] * strength[None] * jnp.abs(e_au) ** 2)
            + jnp.sum(weights["tairte4"] * volumes["tairte4"] * ta_imag * jnp.abs(e_ta) ** 2)
            + epsilon_sio2.imag
            * jnp.sum(weights["sio2"] * volumes["sio2"] * jnp.abs(e_sio2) ** 2)
        )

    forward_objects = _replace_named(
        placed,
        "distributed_adjoint_source",
        placed["distributed_adjoint_source"].aset("static_amplitude_factor", 0.0),
    )
    solve_forward = jax.jit(
        lambda density: fdtdx.run_fdtd(
            arrays_for_density(density),
            forward_objects,
            config,
            key,
            show_progress=False,
        )[1]
    )
    start = time.perf_counter()
    solve_forward_compiled = solve_forward.lower(rho).compile()
    forward_compile_s = time.perf_counter() - start
    start = time.perf_counter()
    forward_out = solve_forward_compiled(rho)
    objective_scaled = objective(forward_out, "late", rho)
    previous_scaled = objective(forward_out, "previous", rho)
    jax.block_until_ready(objective_scaled)
    forward_s = time.perf_counter() - start

    e_au, e_ta, e_sio2, strength = fields(forward_out, "late", rho)
    dynamic_result = None
    dynamic_pte_seconds = 0.0
    frozen_weights_np = weights_np
    if dynamic_pte_weights:
        dynamic_start = time.perf_counter()
        dynamic_pte = _load("fdtdx_dynamic_pte_production", DYNAMIC_PTE)
        stage67 = _load("stage67_dynamic_pte_production", STAGE67)
        forward = stage67._load(stage67.STAGE65, "stage65_dynamic_pte_production")
        electrical = stage67._load(stage67.STAGE54, "stage54_dynamic_pte_production")
        coupled = stage67._load(stage67.STAGE62, "stage62_dynamic_pte_production")
        topology = stage67._load(
            forward.TOPOLOGY_THERMAL, "topology_dynamic_pte_production"
        )
        fvm = stage67._load(
            Path(stage67.__file__).parents[2]
            / "validation"
            / "photothermal_stage1"
            / "anisotropic_heat_fvm.py",
            "fvm_dynamic_pte_production",
        )
        overlap = stage67._load(
            forward.STAGE64, "overlap_dynamic_pte_production"
        )
        physical_prefactor = prefactor * POWER_SCALE_W
        q_fields = {
            "au": np.asarray(
                physical_prefactor
                * epsilon_au.imag
                * strength[None]
                * jnp.abs(e_au) ** 2,
                dtype=np.float64,
            ),
            "tairte4": np.asarray(
                physical_prefactor * ta_imag * jnp.abs(e_ta) ** 2,
                dtype=np.float64,
            ),
            "sio2": np.asarray(
                physical_prefactor * epsilon_sio2.imag * jnp.abs(e_sio2) ** 2,
                dtype=np.float64,
            ),
        }
        dynamic_result = dynamic_pte.evaluate_and_pullback(
            rho=rho_np.astype(np.float64),
            q_fields_W_m3=q_fields,
            dual_volumes_m3={
                name: np.asarray(value, dtype=np.float64)
                for name, value in volumes.items()
            },
            material_slices={
                "au": au_slice,
                "tairte4": ta_slice,
                "sio2": sio2_detector_slice,
            },
            realized_grid=realized,
            scenario=scenario,
            cuda_device=cuda_device,
            overlap=overlap,
            forward=forward,
            stage67=stage67,
            electrical=electrical,
            coupled=coupled,
            topology=topology,
            fvm=fvm,
        )
        weights_np = {
            name: np.asarray(value, dtype=np.float32)
            for name, value in dynamic_result["native_weights_A_W"].items()
        }
        weights = {name: jnp.asarray(value) for name, value in weights_np.items()}
        dynamic_pte_seconds = time.perf_counter() - dynamic_start
    e_stack = jnp.zeros((3, *adjoint_shape), dtype=jnp.complex64)
    coefficient = jnp.zeros((3, *adjoint_shape), dtype=jnp.float32)
    e_stack = e_stack.at[:, :, :, 0:19].set(e_sio2)
    coefficient = coefficient.at[:, :, :, 0:19].set(
        prefactor * epsilon_sio2.imag * weights["sio2"]
    )
    e_stack = e_stack.at[:, 36:236, 36:236, 19:24].set(e_ta)
    coefficient = coefficient.at[:, 36:236, 36:236, 19:24].set(
        prefactor * ta_imag * weights["tairte4"]
    )
    e_stack = e_stack.at[:, 86:186, 86:186, 24:26].set(e_au)
    coefficient = coefficient.at[:, 86:186, 86:186, 24:26].set(
        prefactor
        * epsilon_au.imag
        * weights["au"]
        * strength[None]
    )
    wirtinger = two_solve.quadratic_wirtinger_derivative(e_stack, coefficient)
    adjoint_profile = two_solve.adjoint_current_from_wirtinger(
        wirtinger, config.courant_number
    )
    adjoint_arrays = arrays_for_density(rho)
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
    start = time.perf_counter()
    solve_adjoint_compiled = solve_adjoint.lower().compile()
    adjoint_compile_s = time.perf_counter() - start
    start = time.perf_counter()
    adjoint_out = solve_adjoint_compiled()
    e_adj_au = adjoint_out.detector_states["au_late"]["phasor"][0, 0]
    jax.block_until_ready(e_adj_au)
    adjoint_s = time.perf_counter() - start

    d_strength = jnp.broadcast_to(
        jnp.repeat(jnp.repeat(3.0 * rho**2, 5, axis=0), 5, axis=1)[:, :, None],
        (100, 100, 2),
    )
    d_epsilon = jnp.broadcast_to(
        d_strength[None] * (epsilon_au - 1.0), e_au.shape
    )
    field_voxel = jnp.sum(
        two_solve.harmonic_material_gradient(
            e_au, e_adj_au, d_epsilon, omega, dt
        )
        * volumes["au"],
        axis=(0, 3),
    )
    direct_voxel = prefactor * epsilon_au.imag * jnp.sum(
        weights["au"]
        * volumes["au"]
        * d_strength[None]
        * jnp.abs(e_au) ** 2,
        axis=(0, 3),
    )
    gradient_scaled = _block_reduce(field_voxel + direct_voxel)
    field_gradient_scaled = _block_reduce(field_voxel)
    direct_gradient_scaled = _block_reduce(direct_voxel)
    gradient = np.asarray(gradient_scaled, dtype=np.float64) * POWER_SCALE_W
    field_gradient = (
        np.asarray(field_gradient_scaled, dtype=np.float64) * POWER_SCALE_W
    )
    direct_gradient = (
        np.asarray(direct_gradient_scaled, dtype=np.float64) * POWER_SCALE_W
    )
    objective_value = float(objective_scaled) * POWER_SCALE_W
    previous_value = float(previous_scaled) * POWER_SCALE_W

    reference_norm = float(np.linalg.norm(reference_gradient))
    gradient_norm = float(np.linalg.norm(gradient))
    difference_norm = float(np.linalg.norm(gradient - reference_gradient))
    normalized_vector_error = difference_norm / max(reference_norm, 1.0e-300)
    norm_error = abs(gradient_norm - reference_norm) / max(reference_norm, 1.0e-300)
    cosine = float(
        np.sum(gradient * reference_gradient)
        / max(gradient_norm * reference_norm, 1.0e-300)
    )
    cosine = float(np.clip(cosine, -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cosine)))
    objective_error = abs(objective_value - reference_objective) / max(
        abs(reference_objective), 1.0e-300
    )
    window_change = abs(objective_value - previous_value) / max(
        abs(objective_value), 1.0e-300
    )
    finite = bool(
        np.isfinite(objective_value)
        and np.all(np.isfinite(gradient))
        and np.all(np.isfinite(field_gradient))
        and np.all(np.isfinite(direct_gradient))
    )
    dynamic_metrics = None
    combined_gradient = None
    if dynamic_result is not None:
        weight_errors = {
            material: float(
                np.linalg.norm(weights_np[material] - frozen_weights_np[material])
                / max(np.linalg.norm(frozen_weights_np[material]), 1.0e-300)
            )
            for material in ("au", "tairte4", "sio2")
        }
        combined_gradient = (
            gradient
            + np.asarray(dynamic_result["gradient_thermal_A"], dtype=np.float64)
            + np.asarray(dynamic_result["gradient_electrical_A"], dtype=np.float64)
        )
        dynamic_metrics = {
            "objective_A": float(dynamic_result["objective_A"]),
            "weighted_source_contraction_A": float(
                dynamic_result["native_weighted_contraction_A"]
            ),
            "objective_vs_weighted_contraction_relative_error": abs(
                float(dynamic_result["objective_A"])
                - float(dynamic_result["native_weighted_contraction_A"])
            )
            / max(abs(float(dynamic_result["objective_A"])), 1.0e-300),
            "native_vs_explicit_weighted_contraction_relative_error": float(
                dynamic_result["weighted_contraction_relative_error"]
            ),
            "weight_relative_l2_error_vs_frozen_baseline": weight_errors,
            "thermal_direct_gradient_l2_A": float(
                np.linalg.norm(dynamic_result["gradient_thermal_A"])
            ),
            "electrical_direct_gradient_l2_A": float(
                np.linalg.norm(dynamic_result["gradient_electrical_A"])
            ),
            "combined_gradient_l2_A": float(np.linalg.norm(combined_gradient)),
            "thermal_residual": float(dynamic_result["thermal_residual"]),
            "thermal_adjoint_residual": float(
                dynamic_result["thermal_adjoint_residual"]
            ),
            "thermal_energy_balance": float(
                dynamic_result["thermal_energy_balance"]
            ),
            "electrical_residual": float(dynamic_result["electrical_residual"]),
            "electrical_adjoint_residual": float(
                dynamic_result["electrical_adjoint_residual"]
            ),
            "electrical_terminal_balance": float(
                dynamic_result["electrical_terminal_balance"]
            ),
        }
    gates = {
        "gpu_only": True,
        "checkpoint_count_zero": True,
        "time_history_saved_false": True,
        "finite": finite,
        "objective_matches_checkpointed_reference_lt_0p5pct": objective_error < 0.005,
        "gradient_vector_error_lt_1pct": normalized_vector_error < 0.01,
        "gradient_norm_error_lt_1pct": norm_error < 0.01,
        "gradient_angle_lt_1deg": angle_deg < 1.0,
        "late_window_change_lt_0p5pct": window_change < 0.005,
        "no_empirical_gradient_rescaling": True,
    }
    if dynamic_metrics is not None:
        gates.update(
            {
                "dynamic_weights_match_frozen_baseline_lt_0p5pct": max(
                    dynamic_metrics["weight_relative_l2_error_vs_frozen_baseline"].values()
                ) < 0.005,
                "dynamic_objective_matches_weighted_contraction_lt_0p5pct": dynamic_metrics[
                    "objective_vs_weighted_contraction_relative_error"
                ] < 0.005,
                "dynamic_native_explicit_pullback_lt_1e-6": dynamic_metrics[
                    "native_vs_explicit_weighted_contraction_relative_error"
                ] < 1.0e-6,
                "dynamic_thermal_residual_lt_1e-8": max(
                    dynamic_metrics["thermal_residual"],
                    dynamic_metrics["thermal_adjoint_residual"],
                ) < 1.0e-8,
                "dynamic_electrical_residual_lt_1e-8": max(
                    dynamic_metrics["electrical_residual"],
                    dynamic_metrics["electrical_adjoint_residual"],
                ) < 1.0e-8,
                "dynamic_energy_balance_lt_1pct": dynamic_metrics[
                    "thermal_energy_balance"
                ] < 0.01,
                "dynamic_terminal_balance_lt_1pct": dynamic_metrics[
                    "electrical_terminal_balance"
                ] < 0.01,
            }
        )
    passed = all(gates.values())
    if dynamic_pte_weights:
        status = (
            "VALIDATED_FDTDX_PRODUCTION_DYNAMIC_CHECKPOINT_FREE_PTE_ITERATION"
            if passed
            else "FAILED_FDTDX_PRODUCTION_DYNAMIC_CHECKPOINT_FREE_PTE_ITERATION"
        )
    else:
        status = (
            "VALIDATED_FDTDX_PRODUCTION_CHECKPOINT_FREE_TWO_SOLVE_EQUIVALENCE"
            if passed
            else "FAILED_FDTDX_PRODUCTION_CHECKPOINT_FREE_TWO_SOLVE_EQUIVALENCE"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_gradient_path.parent.mkdir(parents=True, exist_ok=True)
    raw_payload = {
        "rho": rho_np,
        "gradient_A": gradient,
        "field_gradient_A": field_gradient,
        "direct_gradient_A": direct_gradient,
        "reference_gradient_A": reference_gradient,
    }
    if dynamic_result is not None and combined_gradient is not None:
        raw_payload.update(
            gradient_thermal_A=np.asarray(dynamic_result["gradient_thermal_A"]),
            gradient_electrical_A=np.asarray(dynamic_result["gradient_electrical_A"]),
            gradient_combined_A=combined_gradient,
        )
    np.savez_compressed(raw_gradient_path, **raw_payload)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for axis, image, title in (
        (axes[0, 0], rho_np, "production density"),
        (axes[0, 1], gradient, "checkpoint-free two-solve gradient"),
        (axes[0, 2], reference_gradient, "frozen checkpointed gradient"),
        (axes[1, 0], gradient - reference_gradient, "two-solve minus reference"),
        (axes[1, 1], field_gradient, "field-mediated branch"),
        (axes[1, 2], direct_gradient, "direct Au-loss branch"),
    ):
        cmap = "gray" if "density" in title else "coolwarm"
        image_artist = axis.imshow(image.T, origin="lower", cmap=cmap)
        axis.set_title(title)
        fig.colorbar(image_artist, ax=axis)
    figure_path = output_dir / "fdtdx_production_two_solve_equivalence.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    summary = {
        "status": status,
        "scope": (
            (
                "48 um substrate-bearing, 8.5 um-waist, 20x20-density E||b "
                "dynamic PTE iteration: current Maxwell Q, explicit thermal/"
                "electrical/adjoint update, native-Yee source pullback, and "
                "checkpoint-free Maxwell adjoint; no optimization"
            )
            if dynamic_pte_weights
            else (
                "48 um substrate-bearing, 8.5 um-waist, 20x20-density E||b "
                "Maxwell-source gradient equivalence; no thermal/electrical direct "
                "gradient, optimization, checkpointing, or time-history reverse pass"
            )
        ),
        "method": {
            "forward_solves": 1,
            "adjoint_solves": 1,
            "checkpoint_count": 0,
            "time_history_saved": False,
            "gradient_config": None,
            "scenario": scenario,
            "empirical_gradient_rescaling": False,
            "dynamic_PTE_weights_recomputed_from_current_forward": dynamic_pte_weights,
        },
        "contract": {
            "wavelength_m": stage49.WAVELENGTH_M,
            "waist_m": stage49.W0_M,
            "domain_bounds_m_xyz": [
                [float(realized.edges(axis)[0]), float(realized.edges(axis)[-1])]
                for axis in range(3)
            ],
            "grid_shape_xyz": list(realized.shape),
            "grid_cells": int(np.prod(realized.shape)),
            "pml_cells_each_face": 8,
            "total_periods": total_periods,
            "window_periods": window_periods,
            "time_steps_total": config.time_steps_total,
            "axis_mapping": {"x": "b", "y": "a", "z": "c=b"},
            "substrate": substrate_provenance,
            "placement": actual_slices,
        },
        "results": {
            "objective_A": objective_value,
            "checkpointed_reference_objective_A": reference_objective,
            "objective_relative_error": objective_error,
            "window_relative_change": window_change,
            "gradient_l2_A": gradient_norm,
            "checkpointed_reference_gradient_l2_A": reference_norm,
            "gradient_vector_normalized_error": normalized_vector_error,
            "gradient_norm_relative_error": norm_error,
            "gradient_angle_deg": angle_deg,
            "field_gradient_l2_A": float(np.linalg.norm(field_gradient)),
            "direct_gradient_l2_A": float(np.linalg.norm(direct_gradient)),
        },
        "dynamic_PTE_iteration": dynamic_metrics,
        "runtime": {
            "forward_compile_seconds": forward_compile_s,
            "forward_execution_seconds": forward_s,
            "adjoint_compile_seconds": adjoint_compile_s,
            "adjoint_execution_seconds": adjoint_s,
            "two_solve_execution_seconds": forward_s + adjoint_s,
            "two_solve_compile_plus_execution_seconds": (
                forward_compile_s + forward_s + adjoint_compile_s + adjoint_s
            ),
            "frozen_checkpointed_execution_seconds": reference_summary["runtime"]["ad_seconds"],
            "execution_speedup_vs_checkpointed": (
                reference_summary["runtime"]["ad_seconds"] / (forward_s + adjoint_s)
            ),
            "dynamic_thermal_electrical_pullback_seconds": dynamic_pte_seconds,
            "full_pipeline_seconds_before_report_write": (
                time.perf_counter() - pipeline_start
            ),
        },
        "provenance": {
            "weight_path": str(weight_path),
            "weight_sha256": weight_sha,
            "reference_path": str(reference_path),
            "reference_sha256": reference_sha,
            "reference_summary": str(reference_summary_path),
        },
        "raw_artifact": {
            "path": str(raw_gradient_path),
            "bytes": raw_gradient_path.stat().st_size,
            "sha256": _sha256(raw_gradient_path),
            "committed_to_git": False,
        },
        "gates": gates,
    }
    summary_path = output_dir / "fdtdx_production_two_solve_equivalence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--weight-npz", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--weight-summary-json", type=Path, default=DEFAULT_WEIGHT_SUMMARY
    )
    parser.add_argument("--reference-npz", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--reference-summary-json", type=Path, default=DEFAULT_REFERENCE_SUMMARY
    )
    parser.add_argument(
        "--raw-gradient-npz", type=Path, default=DEFAULT_RAW_OUTPUT
    )
    parser.add_argument(
        "--scenario", choices=("thermally_grown", "evaporated"), default="thermally_grown"
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--dynamic-pte-weights", action="store_true")
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()
    run(
        args.output_dir,
        args.weight_npz,
        args.weight_summary_json,
        args.reference_npz,
        args.reference_summary_json,
        args.raw_gradient_npz,
        args.scenario,
        args.audit_only,
        args.dynamic_pte_weights,
        args.cuda_device,
    )


if __name__ == "__main__":
    main()
