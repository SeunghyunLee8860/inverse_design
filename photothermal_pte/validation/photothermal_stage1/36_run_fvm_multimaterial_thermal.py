#!/usr/bin/env python3
"""Run finite-Q anisotropic, finite-G multi-material Cartesian FVM cases.

The first case is provisional until the separate domain/depth/mesh/interface
sensitivity workflow passes.  Temperature is solved as a rise above the
300 K bath and reported per the unit incident intensity of 1 W/m2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np

import config_stage1 as config
from anisotropic_heat_fvm import (
    internal_face_heat_flux_density,
    solve_steady_diagonal_kappa,
)
from lumerical_api import utc_timestamp, write_json


TAIRTE4_BOUNDS_Z_M = (-100.0e-9, 0.0)
OXIDE_BOUNDS_Z_M = (-385.0e-9, -100.0e-9)
DESIGN_BOUNDS_Z_M = (0.0, 600.0e-9)
TAIRTE4_HALF_SPAN_M = 1.0e-6
DESIGN_RADIUS_M = 1.5e-6
TAIRTE4_K_W_MK = np.asarray([14.4, 3.8, 1.0])
SIO2_K_W_MK = 1.38
SI_K_W_MK = 145.0
G_BOTTOM_BASELINE_W_M2K = 7.37e6
G_TOP_BASELINE_W_M2K = 7.37e6
G_OXIDE_SI_BASELINE_W_M2K = 1.1e9
INCIDENT_INTENSITY_W_M2 = 1.0
EXPECTED_Q_POWER_W = 2.56071371086521e-12
POWER_LIMIT = 0.005
ENERGY_LIMIT = 0.01


def parse_g(value: str) -> float | None:
    return None if value.strip().lower() == "perfect" else float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-artifact",
        default=str(
            config.OUTPUT_ROOT
            / "fvm_finite_q_import"
            / "import_v4"
            / "finite_q_exact_flake_source.npz"
        ),
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--case-id", default="baseline_L8um_Si5um")
    parser.add_argument("--lateral-domain-um", type=float, default=8.0)
    parser.add_argument("--si-depth-um", type=float, default=5.0)
    parser.add_argument(
        "--G-bottom", type=parse_g, default=G_BOTTOM_BASELINE_W_M2K
    )
    parser.add_argument(
        "--G-top", type=parse_g, default=G_TOP_BASELINE_W_M2K
    )
    parser.add_argument(
        "--G-oxide-si",
        type=parse_g,
        default=G_OXIDE_SI_BASELINE_W_M2K,
    )
    parser.add_argument(
        "--near-lateral-step-nm", type=float, default=50.0
    )
    parser.add_argument(
        "--oxide-cells", type=int, default=19
    )
    parser.add_argument(
        "--design-step-nm", type=float, default=50.0
    )
    parser.add_argument(
        "--max-outer-step-um", type=float, default=1.0
    )
    parser.add_argument(
        "--max-si-step-um", type=float, default=0.5
    )
    parser.add_argument(
        "--source-coarsening-factor", type=int, default=1
    )
    parser.add_argument(
        "--source-refinement-factor", type=int, default=1
    )
    parser.add_argument(
        "--source-refinement-factor-z", type=int, default=1
    )
    parser.add_argument(
        "--exposed-h-W-m2K", type=float, default=0.0
    )
    parser.add_argument(
        "--tairte4-kz-W-mK", type=float, default=1.0
    )
    parser.add_argument(
        "--far-xy-boundary",
        choices=("fixed", "adiabatic"),
        default="fixed",
    )
    parser.add_argument(
        "--top-disk-support",
        choices=("suspended-overhang", "oxide-supported-overhang"),
        default="suspended-overhang",
    )
    parser.add_argument(
        "--physical-scenario-label",
        default="numerical_convergence_checkpoint_parameters",
    )
    return parser.parse_args()


def clean_output_directory(explicit: str | None, case_id: str) -> Path:
    output = (
        Path(explicit).expanduser().resolve()
        if explicit
        else config.OUTPUT_ROOT
        / "fvm_multimaterial_thermal"
        / f"{utc_timestamp()}_{case_id}"
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remesh_source(
    edges: tuple[np.ndarray, np.ndarray, np.ndarray],
    power_per_cell_W: np.ndarray,
    *,
    coarsening_factor: int,
    refinement_factor: int,
    refinement_factor_z: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, str]:
    """Conservatively restrict or refine source-energy control volumes."""
    if (
        coarsening_factor < 1
        or refinement_factor < 1
        or refinement_factor_z < 1
    ):
        raise ValueError("source mesh factors must be positive integers")
    if coarsening_factor > 1 and (
        refinement_factor > 1 or refinement_factor_z > 1
    ):
        raise ValueError("cannot coarsen and refine source simultaneously")
    if refinement_factor > 1 and refinement_factor_z > 1:
        raise ValueError(
            "use either global or z-only source refinement, not both"
        )
    remeshed_edges = tuple(np.asarray(item, float) for item in edges)
    remeshed_power = np.asarray(power_per_cell_W, float)
    mode = "native"
    if coarsening_factor > 1:
        mode = f"conservative_coarsening_x{coarsening_factor}"
        for axis in range(3):
            old_edges = remeshed_edges[axis]
            starts = np.arange(0, old_edges.size - 1, coarsening_factor)
            new_edge_indices = np.append(starts, old_edges.size - 1)
            new_edges = old_edges[new_edge_indices]
            remeshed_power = np.add.reduceat(
                remeshed_power, starts, axis=axis
            )
            edge_list = list(remeshed_edges)
            edge_list[axis] = new_edges
            remeshed_edges = tuple(edge_list)
    elif refinement_factor > 1 or refinement_factor_z > 1:
        factors = (
            (refinement_factor,) * 3
            if refinement_factor > 1
            else (1, 1, refinement_factor_z)
        )
        mode = "piecewise_constant_refinement_" + "x".join(
            str(item) for item in factors
        )
        for axis in range(3):
            axis_factor = factors[axis]
            if axis_factor == 1:
                continue
            old_edges = remeshed_edges[axis]
            segments = [
                np.linspace(
                    old_edges[index],
                    old_edges[index + 1],
                    axis_factor + 1,
                )[:-1]
                for index in range(old_edges.size - 1)
            ]
            new_edges = np.concatenate((*segments, old_edges[-1:]))
            remeshed_power = (
                np.repeat(remeshed_power, axis_factor, axis=axis)
                / axis_factor
            )
            edge_list = list(remeshed_edges)
            edge_list[axis] = new_edges
            remeshed_edges = tuple(edge_list)
    widths = tuple(np.diff(item) for item in remeshed_edges)
    volume = (
        widths[0][:, None, None]
        * widths[1][None, :, None]
        * widths[2][None, None, :]
    )
    remeshed_q = remeshed_power / volume
    return remeshed_edges, remeshed_q, mode


def growing_positions(
    length_m: float,
    *,
    initial_step_m: float,
    growth: float,
    maximum_step_m: float,
) -> np.ndarray:
    if length_m <= 0.0:
        return np.asarray([0.0])
    positions = [0.0]
    step = initial_step_m
    while positions[-1] + step < length_m:
        positions.append(positions[-1] + step)
        step = min(step * growth, maximum_step_m)
    remainder = length_m - positions[-1]
    if len(positions) > 1 and remainder < 0.4 * (
        positions[-1] - positions[-2]
    ):
        positions[-1] = length_m
    else:
        positions.append(length_m)
    return np.asarray(positions)


def extend_lateral_edges(
    exact_flake_edges_m: np.ndarray,
    *,
    lateral_span_m: float,
    near_step_m: float,
    maximum_outer_step_m: float,
) -> tuple[np.ndarray, slice]:
    half_span = 0.5 * lateral_span_m
    if half_span <= DESIGN_RADIUS_M:
        raise ValueError("lateral domain must extend beyond the design disk")
    core = np.asarray(exact_flake_edges_m, float)
    if not np.isclose(core[0], -TAIRTE4_HALF_SPAN_M) or not np.isclose(
        core[-1], TAIRTE4_HALF_SPAN_M
    ):
        raise ValueError("source lateral edges do not match exact flake")
    near_limit = min(1.7e-6, half_span)
    near = np.arange(
        TAIRTE4_HALF_SPAN_M + near_step_m,
        near_limit + 0.5 * near_step_m,
        near_step_m,
    )
    if near.size and near[-1] > near_limit:
        near[-1] = near_limit
    start = near[-1] if near.size else TAIRTE4_HALF_SPAN_M
    outer_relative = growing_positions(
        half_span - start,
        initial_step_m=near_step_m,
        growth=1.4,
        maximum_step_m=maximum_outer_step_m,
    )[1:]
    positive = np.concatenate((near, start + outer_relative))
    positive = positive[positive > TAIRTE4_HALF_SPAN_M]
    negative = -positive[::-1]
    edges = np.concatenate((negative, core, positive))
    if not np.all(np.diff(edges) > 0.0):
        raise RuntimeError("extended lateral edges are not increasing")
    core_start = negative.size
    return edges, slice(core_start, core_start + core.size - 1)


def build_z_edges(
    source_z_edges_m: np.ndarray,
    *,
    si_depth_m: float,
    oxide_cells: int,
    design_step_m: float,
    maximum_si_step_m: float,
) -> tuple[np.ndarray, dict[str, slice]]:
    source_edges = np.asarray(source_z_edges_m, float)
    if not np.isclose(source_edges[0], TAIRTE4_BOUNDS_Z_M[0]) or not np.isclose(
        source_edges[-1], TAIRTE4_BOUNDS_Z_M[1]
    ):
        raise ValueError("source z edges do not match the exact flake")
    si_relative = growing_positions(
        si_depth_m,
        initial_step_m=25.0e-9,
        growth=1.35,
        maximum_step_m=maximum_si_step_m,
    )
    si_edges = OXIDE_BOUNDS_Z_M[0] - si_relative[::-1]
    oxide_edges = np.linspace(
        OXIDE_BOUNDS_Z_M[0], OXIDE_BOUNDS_Z_M[1], oxide_cells + 1
    )
    design_cells = int(
        round((DESIGN_BOUNDS_Z_M[1] - DESIGN_BOUNDS_Z_M[0]) / design_step_m)
    )
    if design_cells < 1:
        raise ValueError("design mesh has no cells")
    design_edges = np.linspace(
        DESIGN_BOUNDS_Z_M[0], DESIGN_BOUNDS_Z_M[1], design_cells + 1
    )
    edges = np.concatenate(
        (
            si_edges,
            oxide_edges[1:],
            source_edges[1:],
            design_edges[1:],
        )
    )
    counts = {
        "si": si_edges.size - 1,
        "oxide": oxide_edges.size - 1,
        "flake": source_edges.size - 1,
        "design": design_edges.size - 1,
    }
    offset = 0
    slices: dict[str, slice] = {}
    for name in ("si", "oxide", "flake", "design"):
        slices[name] = slice(offset, offset + counts[name])
        offset += counts[name]
    if not np.all(np.diff(edges) > 0.0):
        raise RuntimeError("z edges are not increasing")
    return edges, slices


def interface_face_index(edges: np.ndarray, coordinate_m: float) -> int:
    matches = np.flatnonzero(
        np.isclose(edges, coordinate_m, rtol=0.0, atol=1.0e-18)
    )
    if matches.size != 1 or matches[0] == 0 or matches[0] == edges.size - 1:
        raise ValueError(f"cannot resolve internal interface at {coordinate_m}")
    return int(matches[0] - 1)


def interface_statistics(
    flux_z: np.ndarray,
    *,
    face_index: int,
    conductance_W_m2K: float | None,
    x_widths_m: np.ndarray,
    y_widths_m: np.ndarray,
    connected: np.ndarray,
) -> dict[str, Any]:
    face_flux = flux_z[:, :, face_index]
    area = x_widths_m[:, None] * y_widths_m[None, :]
    selected_flux = face_flux[connected]
    selected_area = area[connected]
    power_W = float(np.sum(selected_flux * selected_area))
    absolute_power_W = float(np.sum(np.abs(selected_flux) * selected_area))
    if conductance_W_m2K is None:
        jump = np.zeros_like(selected_flux)
    else:
        jump = np.abs(selected_flux) / conductance_W_m2K
    return {
        "G_W_m2K": conductance_W_m2K,
        "perfect_contact": conductance_W_m2K is None,
        "connected_face_count": int(np.count_nonzero(connected)),
        "signed_positive_z_power_W": power_W,
        "absolute_transmitted_power_W": absolute_power_W,
        "area_weighted_mean_temperature_jump_K": float(
            np.sum(jump * selected_area) / np.sum(selected_area)
        ),
        "maximum_temperature_jump_K": float(np.max(jump)),
    }


def main() -> int:
    args = parse_args()
    if not np.isfinite(args.tairte4_kz_W_mK) or args.tairte4_kz_W_mK <= 0.0:
        raise ValueError("TaIrTe4 kz must be finite and positive")
    output = clean_output_directory(args.output_dir, args.case_id)
    command = shlex.join([sys.executable, *sys.argv])
    source_path = Path(args.source_artifact).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    with np.load(source_path, allow_pickle=False) as source_data:
        source_x_edges = np.asarray(source_data["x_edges_m"], float)
        source_y_edges = np.asarray(source_data["y_edges_m"], float)
        source_z_edges = np.asarray(source_data["z_edges_m"], float)
        source_q = np.asarray(source_data["Q_fvm_W_m3"], float)
        source_power_per_cell = np.asarray(
            source_data["source_power_per_cell_W"], float
        )
        source_incident_intensity = float(
            np.asarray(
                source_data["incident_intensity_W_m2"], float
            ).reshape(-1)[0]
        )
        source_artifact_sha = str(
            np.asarray(source_data["source_artifact_sha256"]).item()
        )
    (
        (source_x_edges, source_y_edges, source_z_edges),
        source_q,
        source_mesh_mode,
    ) = remesh_source(
        (source_x_edges, source_y_edges, source_z_edges),
        source_power_per_cell,
        coarsening_factor=args.source_coarsening_factor,
        refinement_factor=args.source_refinement_factor,
        refinement_factor_z=args.source_refinement_factor_z,
    )
    if source_incident_intensity != INCIDENT_INTENSITY_W_M2:
        raise ValueError("source is not normalized to 1 W/m2")
    source_power_W = float(np.sum(source_power_per_cell))
    if abs(source_power_W - EXPECTED_Q_POWER_W) / EXPECTED_Q_POWER_W >= POWER_LIMIT:
        raise ValueError("source artifact does not preserve expected optical Q")

    lateral_span_m = args.lateral_domain_um * 1.0e-6
    si_depth_m = args.si_depth_um * 1.0e-6
    x_edges, x_source_slice = extend_lateral_edges(
        source_x_edges,
        lateral_span_m=lateral_span_m,
        near_step_m=args.near_lateral_step_nm * 1.0e-9,
        maximum_outer_step_m=args.max_outer_step_um * 1.0e-6,
    )
    y_edges, y_source_slice = extend_lateral_edges(
        source_y_edges,
        lateral_span_m=lateral_span_m,
        near_step_m=args.near_lateral_step_nm * 1.0e-9,
        maximum_outer_step_m=args.max_outer_step_um * 1.0e-6,
    )
    z_edges, z_slices = build_z_edges(
        source_z_edges,
        si_depth_m=si_depth_m,
        oxide_cells=args.oxide_cells,
        design_step_m=args.design_step_nm * 1.0e-9,
        maximum_si_step_m=args.max_si_step_um * 1.0e-6,
    )
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    z = 0.5 * (z_edges[:-1] + z_edges[1:])
    shape = (x.size, y.size, z.size)
    xy_flake = (
        (np.abs(x[:, None]) < TAIRTE4_HALF_SPAN_M)
        & (np.abs(y[None, :]) < TAIRTE4_HALF_SPAN_M)
    )
    xy_design = (
        x[:, None] ** 2 + y[None, :] ** 2 <= DESIGN_RADIUS_M**2
    )
    active = np.zeros(shape, bool)
    material_id = np.zeros(shape, np.uint8)
    active[:, :, z_slices["si"]] = True
    material_id[:, :, z_slices["si"]] = 1
    active[:, :, z_slices["oxide"]] = True
    material_id[:, :, z_slices["oxide"]] = 2
    active[:, :, z_slices["flake"]] = xy_flake[:, :, None]
    material_id[:, :, z_slices["flake"]] = (
        3 * xy_flake[:, :, None]
    )
    if args.top_disk_support == "oxide-supported-overhang":
        xy_support = xy_design & ~xy_flake
        active[:, :, z_slices["flake"]] |= xy_support[:, :, None]
        material_id[:, :, z_slices["flake"]] = np.where(
            xy_support[:, :, None],
            np.uint8(5),
            material_id[:, :, z_slices["flake"]],
        )
    active[:, :, z_slices["design"]] = xy_design[:, :, None]
    material_id[:, :, z_slices["design"]] = (
        4 * xy_design[:, :, None]
    )

    kappa = np.ones((*shape, 3), float)
    kappa[material_id == 1] = SI_K_W_MK
    kappa[material_id == 2] = SIO2_K_W_MK
    tairte4_kappa = np.asarray(
        [TAIRTE4_K_W_MK[0], TAIRTE4_K_W_MK[1], args.tairte4_kz_W_mK]
    )
    kappa[material_id == 3] = tairte4_kappa
    kappa[material_id == 4] = SIO2_K_W_MK
    kappa[material_id == 5] = SIO2_K_W_MK
    source = np.zeros(shape, float)
    source[
        x_source_slice,
        y_source_slice,
        z_slices["flake"],
    ] = source_q
    if np.any(source[~active] != 0.0):
        raise RuntimeError("mapped optical source enters inactive air")
    volume = (
        np.diff(x_edges)[:, None, None]
        * np.diff(y_edges)[None, :, None]
        * np.diff(z_edges)[None, None, :]
    )
    imported_power_W = float(np.sum(source * volume))
    import_error = abs(
        imported_power_W - EXPECTED_Q_POWER_W
    ) / EXPECTED_Q_POWER_W
    if import_error >= POWER_LIMIT:
        raise RuntimeError("production-grid Q import failed 0.5% gate")

    resistance_z = np.zeros((shape[0], shape[1], shape[2] - 1))
    interface_configuration = (
        (
            "oxide_Si",
            OXIDE_BOUNDS_Z_M[0],
            args.G_oxide_si,
            1,
            2,
        ),
        (
            "TaIrTe4_bottom",
            TAIRTE4_BOUNDS_Z_M[0],
            args.G_bottom,
            2,
            3,
        ),
        (
            "TaIrTe4_top",
            TAIRTE4_BOUNDS_Z_M[1],
            args.G_top,
            3,
            4,
        ),
    )
    interface_faces: dict[str, int] = {}
    interface_masks: dict[str, np.ndarray] = {}
    for (
        name,
        coordinate,
        conductance,
        lower_material,
        upper_material,
    ) in interface_configuration:
        face = interface_face_index(z_edges, coordinate)
        interface_faces[name] = face
        connected = (
            (material_id[:, :, face] == lower_material)
            & (material_id[:, :, face + 1] == upper_material)
        )
        interface_masks[name] = connected
        if conductance is not None:
            resistance_z[:, :, face][connected] = 1.0 / conductance

    dirichlet_temperature_K = {"z_min": 0.0}
    if args.far_xy_boundary == "fixed":
        dirichlet_temperature_K.update(
            {
                "x_min": 0.0,
                "x_max": 0.0,
                "y_min": 0.0,
                "y_max": 0.0,
            }
        )
    solved = solve_steady_diagonal_kappa(
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        z_edges_m=z_edges,
        kappa_W_mK=kappa,
        source_W_m3=source,
        dirichlet_temperature_K=dirichlet_temperature_K,
        interface_resistance_m2K_W={"z": resistance_z},
        active_mask=active,
        exposed_heat_transfer_W_m2K=args.exposed_h_W_m2K,
        ambient_temperature_K=0.0,
        relative_tolerance=1.0e-10,
        max_iterations=10000,
    )
    delta_T = solved.temperature_K
    if not np.all(np.isfinite(delta_T[active])):
        raise RuntimeError("production temperature contains NaN or Inf")
    fluxes = internal_face_heat_flux_density(
        temperature_K=delta_T,
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        z_edges_m=z_edges,
        kappa_W_mK=kappa,
        interface_resistance_m2K_W={"z": resistance_z},
        active_mask=active,
    )
    flake_mask = material_id == 3
    flake_volume = volume[flake_mask]
    flake_delta_T = delta_T[flake_mask]
    maximum_flat_index = int(np.nanargmax(delta_T))
    maximum_index = np.unravel_index(maximum_flat_index, shape)
    boundary_powers = solved.boundary_power_out_W
    bottom_power_W = boundary_powers["z_min"]
    lateral_power_W = sum(
        boundary_powers.get(name, 0.0)
        for name in ("x_min", "x_max", "y_min", "y_max")
    )
    convection_power_W = boundary_powers.get(
        "exposed_convection", 0.0
    )
    escaped_power_W = sum(boundary_powers.values())
    interface_results = {}
    for name, _, conductance, _, _ in interface_configuration:
        face = interface_faces[name]
        interface_results[name] = interface_statistics(
            fluxes["z"],
            face_index=face,
            conductance_W_m2K=conductance,
            x_widths_m=np.diff(x_edges),
            y_widths_m=np.diff(y_edges),
            connected=interface_masks[name],
        )

    raw_path = output / "temperature_flux_3d.npz"
    np.savez_compressed(
        raw_path,
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        z_edges_m=z_edges,
        delta_T_K_per_W_m2=delta_T,
        temperature_K_at_unit_intensity=delta_T + 300.0,
        Q_W_m3_per_W_m2=source,
        material_id=material_id,
        active_solid_mask=active,
        kappa_diagonal_W_mK=kappa,
        interface_resistance_z_m2K_W=resistance_z,
        heat_flux_x_W_m2=fluxes["x"],
        heat_flux_y_W_m2=fluxes["y"],
        heat_flux_z_W_m2=fluxes["z"],
    )
    passed = bool(
        import_error < POWER_LIMIT
        and solved.energy_balance_relative_error < ENERGY_LIMIT
        and solved.linear_residual_relative < 1.0e-8
    )
    result = {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        "case_id": args.case_id,
        "physical_scenario_label": args.physical_scenario_label,
        "status": (
            "PASSED_PROVISIONAL_MULTIMATERIAL_FVM_CASE"
            if passed
            else "FAILED_MULTIMATERIAL_FVM_CASE"
        ),
        "passed": passed,
        "provisional_until_sensitivity_passes": True,
        "solver_attribution": (
            "independent conservative Cartesian Python/SciPy FVM; "
            "not a Lumerical HEAT result"
        ),
        "unit_response_mode": True,
        "incident_intensity_W_m2": INCIDENT_INTENSITY_W_M2,
        "reported_temperature_quantity": (
            "Delta T / incident intensity [K/(W/m2)]"
        ),
        "geometry": {
            "lateral_domain_m": lateral_span_m,
            "Si_depth_m": si_depth_m,
            "TaIrTe4_bounds_m": {
                "x": [-1.0e-6, 1.0e-6],
                "y": [-1.0e-6, 1.0e-6],
                "z": list(TAIRTE4_BOUNDS_Z_M),
            },
            "bottom_SiO2_bounds_z_m": list(OXIDE_BOUNDS_Z_M),
            "design": {
                "shape": "single centered disk",
                "radius_m": DESIGN_RADIUS_M,
                "z_bounds_m": list(DESIGN_BOUNDS_Z_M),
                "material": "SiO2",
                "thermal_support_scenario": args.top_disk_support,
                "fabrication_geometry_confirmed_from_repository": False,
                "fabrication_geometry_blocker": (
                    "BLOCKED_FABRICATION_GEOMETRY_UNCONFIRMED"
                ),
            },
        },
        "materials_W_mK": {
            "TaIrTe4_diagonal": tairte4_kappa.tolist(),
            "TaIrTe4_kz_note": (
                "numerical scenario; repository does not establish a "
                "confidence interval"
            ),
            "SiO2": SIO2_K_W_MK,
            "Si": SI_K_W_MK,
        },
        "interfaces": interface_results,
        "boundary_conditions": {
            "bottom_Si": "DeltaT=0 K (T=300 K)",
            "far_x_y_Si_and_SiO2": (
                "DeltaT=0 K (T=300 K)"
                if args.far_xy_boundary == "fixed"
                else "adiabatic"
            ),
            "far_x_y_boundary_mode": args.far_xy_boundary,
            "top_and_internal_solid_air_surfaces": (
                "adiabatic"
                if args.exposed_h_W_m2K == 0.0
                else "Robin convection to DeltaT=0 K ambient"
            ),
            "exposed_heat_transfer_W_m2K": args.exposed_h_W_m2K,
            "periodic": False,
            "thermal_PML": False,
        },
        "grid": {
            "shape": list(shape),
            "total_cell_count": int(np.prod(shape)),
            "active_solid_cell_count": int(np.count_nonzero(active)),
            "material_cell_counts": {
                "Si": int(np.count_nonzero(material_id == 1)),
                "SiO2_bottom": int(np.count_nonzero(material_id == 2)),
                "TaIrTe4": int(np.count_nonzero(material_id == 3)),
                "SiO2_design": int(np.count_nonzero(material_id == 4)),
                "SiO2_support": int(np.count_nonzero(material_id == 5)),
            },
            "minimum_steps_m": {
                "x": float(np.min(np.diff(x_edges))),
                "y": float(np.min(np.diff(y_edges))),
                "z": float(np.min(np.diff(z_edges))),
            },
            "maximum_steps_m": {
                "x": float(np.max(np.diff(x_edges))),
                "y": float(np.max(np.diff(y_edges))),
                "z": float(np.max(np.diff(z_edges))),
            },
        },
        "source": {
            "artifact_path": str(source_path),
            "artifact_file_sha256": sha256_file(source_path),
            "optical_source_artifact_sha256": source_artifact_sha,
            "expected_power_W": EXPECTED_Q_POWER_W,
            "mapped_source_power_W": imported_power_W,
            "mapping_relative_error": import_error,
            "thermal_source_mesh_mode": source_mesh_mode,
            "clipping": False,
            "smoothing": False,
            "gain": False,
            "rescaling": False,
            "tiling": False,
            "outside_flake_deletion": False,
        },
        "temperature_response": {
            "DeltaT_max_K_per_W_m2": float(np.nanmax(delta_T)),
            "TaIrTe4_volume_average_DeltaT_K_per_W_m2": float(
                np.sum(flake_delta_T * flake_volume)
                / np.sum(flake_volume)
            ),
            "TaIrTe4_max_DeltaT_K_per_W_m2": float(
                np.max(flake_delta_T)
            ),
            "hotspot_location_m": {
                "x": float(x[maximum_index[0]]),
                "y": float(y[maximum_index[1]]),
                "z": float(z[maximum_index[2]]),
            },
        },
        "power_balance": {
            "generated_W": solved.source_power_W,
            "bottom_outflow_W": bottom_power_W,
            "lateral_outflow_W": lateral_power_W,
            "top_convection_outflow_W": convection_power_W,
            "exposed_convection_outflow_W": convection_power_W,
            "total_escaped_W": escaped_power_W,
            "relative_error": solved.energy_balance_relative_error,
            "numerical_truncation_boundary_flux": {
                "interpretation": (
                    "numerical boundary flux; not a physical heat-path "
                    "fraction"
                ),
                "bottom_fraction_of_generated": (
                    bottom_power_W / solved.source_power_W
                ),
                "lateral_fraction_of_generated": (
                    lateral_power_W / solved.source_power_W
                ),
            },
        },
        "linear_solver": {
            "name": solved.solver,
            "iterations": solved.iterations,
            "relative_residual": solved.linear_residual_relative,
        },
        "raw_field_path": str(raw_path),
        "generation_command": command,
        "next_required_gate": (
            "DOMAIN_DEPTH_MESH_INTERFACE_BOUNDARY_SENSITIVITY"
        ),
        "criteria": {
            "Q_mapping_relative_error_lt": POWER_LIMIT,
            "energy_balance_relative_error_lt": ENERGY_LIMIT,
            "linear_residual_relative_lt": 1.0e-8,
        },
    }
    write_json(output / "case_result.json", result)
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
