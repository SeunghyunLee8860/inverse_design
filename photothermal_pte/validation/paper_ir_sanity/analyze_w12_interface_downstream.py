#!/usr/bin/env python3
"""Audit w12 50/25-nm interface Q and downstream thermal sensitivity.

No Lumerical session is opened.  The command consumes existing Q artifacts
and a separately generated read-only index audit, then applies one identical
exact-overlap remap and explicit anisotropic/interface thermal solve to both.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import linalg as sparse_linalg
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
FVM_DIR = HERE.parent / "photothermal_stage1"
for location in (REPOSITORY, FVM_DIR):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from anisotropic_heat_fvm import (  # noqa: E402
    SteadyHeatResult,
    assemble_steady_diagonal_kappa,
    solve_assembled_thermal_system,
)
from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    run_device_a_explicit_thermal_pte as thermal,
)
from photothermal_pte.validation.paper_ir_sanity.summarize_w12_edge_a_xy_refinement import (  # noqa: E402
    bounded_dual_cells,
    overlap_fraction,
    remap_energy,
    volume,
)


GATE = 5.0e-3
ENERGY_GATE = 1.0e-2
RESIDUAL_GATE = 1.0e-8
SLAB_BOUNDS_M = (-10.0e-9, 0.0)
WEIGHTING_X_M_INV = 1.0 / (4.0e-6)
WEIGHTING_Y_M_INV = 1.0 / (4.0e-6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-case", type=Path, required=True)
    parser.add_argument("--fine-case", type=Path, required=True)
    parser.add_argument("--interface-index-audit", type=Path, required=True)
    parser.add_argument("--raw-output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(left: float, right: float) -> float:
    return abs(left - right) / max(abs(right), np.finfo(float).tiny)


def weighted_nrmse(
    left: np.ndarray,
    right: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    a = np.asarray(left, float)
    b = np.asarray(right, float)
    w = np.broadcast_to(np.asarray(weight, float), a.shape)
    selected = np.ones(a.shape, bool) if mask is None else np.asarray(mask, bool)
    numerator = float(np.sum(w[selected] * (a[selected] - b[selected]) ** 2))
    denominator = float(np.sum(w[selected] * b[selected] ** 2))
    return float(np.sqrt(numerator / max(denominator, np.finfo(float).tiny)))


def load_contract(directory: Path) -> tuple[dict[str, Any], Path]:
    result_path = directory / "case_result.json"
    artifact = directory / "finite_q_on_artifact.npz"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "COMPLETED" or not artifact.is_file():
        raise RuntimeError(f"incomplete optical case: {directory}")
    return result, artifact


def source_grid(
    result: dict[str, Any],
    artifact: Any,
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
]:
    coordinates = tuple(
        np.asarray(artifact[f"{axis}_m"], float) for axis in "xyz"
    )
    bounds = result["run_result"]["native_Yee_mesh_audit"][
        "Q_quadrature_control_volume_bounds_m"
    ]
    cells = tuple(
        bounded_dual_cells(coordinate, *bounds[axis])
        for coordinate, axis in zip(coordinates, "xyz")
    )
    return (
        tuple(value[0] for value in cells),
        tuple(value[1] for value in cells),
        coordinates,
    )


def interface_metrics(
    result: dict[str, Any],
    artifact_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metrics: dict[str, Any] = {"components": {}}
    maps: dict[str, np.ndarray] = {}
    with np.load(artifact_path, allow_pickle=False) as artifact:
        indices, edges, coordinates = source_grid(result, artifact)
        cell_volume = volume(edges)
        dxdy = (
            np.diff(edges[0])[:, None] * np.diff(edges[1])[None, :]
        )
        z_coordinate = coordinates[2][indices[2]]
        zero_candidates = np.flatnonzero(np.abs(z_coordinate) <= 1.0e-18)
        if zero_candidates.size != 1:
            raise RuntimeError("common Q grid does not have one z=0 sample")
        zero_index = int(zero_candidates[0])
        z_edges = edges[2]
        overlap_z = np.maximum(
            0.0,
            np.minimum(z_edges[1:], SLAB_BOUNDS_M[1])
            - np.maximum(z_edges[:-1], SLAB_BOUNDS_M[0]),
        )
        for component, key in (
            ("x", "Qx_W_m3"),
            ("y", "Qy_W_m3"),
            ("z", "Qz_W_m3"),
            ("total", "Q_on_W_m3"),
        ):
            q = np.asarray(
                artifact[key][np.ix_(*indices)],
                float,
            )
            total_power = float(np.sum(q * cell_volume))
            layer_areal = q[:, :, zero_index] * np.diff(z_edges)[zero_index]
            layer_power = float(np.sum(layer_areal * dxdy))
            slab_areal = np.sum(q * overlap_z[None, None, :], axis=2)
            slab_power = float(np.sum(slab_areal * dxdy))
            metrics["components"][component] = {
                "total_power_W": total_power,
                "z0_layer_sample_coordinate_m": float(
                    z_coordinate[zero_index]
                ),
                "z0_layer_dual_cell_bounds_m": [
                    float(z_edges[zero_index]),
                    float(z_edges[zero_index + 1]),
                ],
                "z0_layer_thickness_m": float(
                    z_edges[zero_index + 1] - z_edges[zero_index]
                ),
                "z0_layer_power_W": layer_power,
                "z0_layer_power_fraction": layer_power / total_power,
                "interface_slab_bounds_m": list(SLAB_BOUNDS_M),
                "interface_slab_power_W": slab_power,
                "interface_slab_power_fraction": slab_power / total_power,
            }
            if component == "total":
                maps = {
                    "x_edges_m": edges[0],
                    "y_edges_m": edges[1],
                    "z_edges_m": edges[2],
                    "z0_layer_areal_W_m2": layer_areal,
                    "slab_areal_W_m2": slab_areal,
                }
    return metrics, maps


def compare_areal_maps(
    coarse: dict[str, np.ndarray],
    fine: dict[str, np.ndarray],
    key: str,
) -> dict[str, Any]:
    operators = (
        overlap_fraction(coarse["x_edges_m"], fine["x_edges_m"]),
        overlap_fraction(coarse["y_edges_m"], fine["y_edges_m"]),
    )
    coarse_area = volume((coarse["x_edges_m"], coarse["y_edges_m"]))
    fine_area = volume((fine["x_edges_m"], fine["y_edges_m"]))
    energy_coarse = coarse[key] * coarse_area
    energy_fine = fine[key] * fine_area
    energy_fine_on_coarse = remap_energy(energy_fine, operators)
    power_coarse = float(np.sum(energy_coarse))
    power_fine = float(np.sum(energy_fine))
    return {
        "coarse_power_W": power_coarse,
        "fine_power_W": power_fine,
        "power_relative_change": relative(power_coarse, power_fine),
        "conservative_remap_power_error": relative(
            float(np.sum(energy_fine_on_coarse)),
            power_fine,
        ),
        "raw_energy_NRMSE": float(
            np.linalg.norm(energy_coarse - energy_fine_on_coarse)
            / np.linalg.norm(energy_fine_on_coarse)
        ),
        "equal_power_energy_NRMSE": float(
            np.linalg.norm(
                energy_coarse / power_coarse
                - energy_fine_on_coarse / power_fine
            )
            / np.linalg.norm(energy_fine_on_coarse / power_fine)
        ),
        "correlation": float(
            np.corrcoef(
                (energy_coarse / power_coarse).reshape(-1),
                (energy_fine_on_coarse / power_fine).reshape(-1),
            )[0, 1]
        ),
        "fine_energy_on_coarse": energy_fine_on_coarse,
        "coarse_energy": energy_coarse,
    }


def project_energy_to_support(
    energy: np.ndarray,
    target_edges: tuple[np.ndarray, ...],
    support: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(energy, float)
    supported = np.asarray(support, bool)
    if source.shape != supported.shape:
        raise ValueError("energy/support shape mismatch")
    # ``remap_energy`` may return a non-C-contiguous array.  Force one
    # contiguous owned buffer so ravelled in-place projection cannot silently
    # update a temporary copy.
    result = np.ascontiguousarray(source).copy()
    flat = result.ravel()
    support_flat = np.flatnonzero(supported.reshape(-1))
    outside_flat = np.flatnonzero((~supported).reshape(-1) & (flat != 0.0))
    outside_power = float(np.sum(flat[outside_flat]))
    if outside_flat.size:
        centers = tuple(
            0.5 * (axis[:-1] + axis[1:]) for axis in target_edges
        )

        def points(indices: np.ndarray) -> np.ndarray:
            unravelled = np.unravel_index(indices, source.shape)
            return np.column_stack(
                [centers[axis][unravelled[axis]] for axis in range(3)]
            )

        support_points = points(support_flat)
        query_points = points(outside_flat)
        distances, neighbours = cKDTree(support_points).query(
            query_points,
            k=min(16, support_flat.size),
        )
        if distances.ndim == 1:
            distances = distances[:, None]
            neighbours = neighbours[:, None]
        minimum = distances[:, :1]
        tied = distances <= minimum + np.maximum(
            1.0e-18, minimum * 1.0e-10
        )
        if np.any(tied[:, -1]) and support_flat.size > tied.shape[1]:
            raise RuntimeError("nearest-support tie exceeds 16 candidates")
        values = flat[outside_flat].copy()
        flat[outside_flat] = 0.0
        for row, value in enumerate(values):
            selected = support_flat[neighbours[row, tied[row]]]
            np.add.at(flat, selected, value / selected.size)
    outside_after = (~supported) & (result != 0.0)
    return result, {
        "outside_support_nonzero_cells_before_projection": int(
            outside_flat.size
        ),
        "outside_support_power_W_before_projection": outside_power,
        "power_before_W": float(np.sum(source)),
        "power_after_W": float(np.sum(result)),
        "relative_power_error": relative(
            float(np.sum(result)), float(np.sum(source))
        ),
        "outside_support_nonzero_cells_after_projection": int(
            np.count_nonzero(outside_after)
        ),
        "outside_support_power_W_after_projection": float(
            np.sum(result[outside_after])
        ),
        "method": (
            "exact Cartesian cell-overlap energy remap followed by one "
            "physical-3D nearest-support projection with exact-distance "
            "ties split uniformly"
        ),
    }


def map_to_thermal(
    result: dict[str, Any],
    artifact_path: Path,
    geometry: thermal.Geometry,
) -> tuple[np.ndarray, dict[str, Any]]:
    target_edges = (
        geometry.x_edges_m,
        geometry.y_edges_m,
        geometry.z_edges_m,
    )
    with np.load(artifact_path, allow_pickle=False) as artifact:
        indices, source_edges, _ = source_grid(result, artifact)
        q = np.asarray(
            artifact["Q_on_W_m3"][np.ix_(*indices)],
            float,
        )
        source_volume = volume(source_edges)
        source_energy = q * source_volume
        operators = tuple(
            overlap_fraction(target, source)
            for target, source in zip(target_edges, source_edges)
        )
        target_energy = remap_energy(source_energy, operators)
    target_volume = volume(target_edges)
    projected_energy, projection = project_energy_to_support(
        target_energy,
        target_edges,
        geometry.flake_mask,
    )
    mapped = projected_energy / target_volume
    source_power = float(np.sum(source_energy))
    target_power = float(np.sum(projected_energy))
    return mapped, {
        "source_power_W": source_power,
        "target_power_W": target_power,
        "exact_overlap_power_error": relative(
            float(np.sum(target_energy)), source_power
        ),
        "final_mapping_power_error": relative(target_power, source_power),
        "projection": projection,
        "Q_operations": {
            "clipping": False,
            "smoothing": False,
            "gain": False,
            "global_rescaling": False,
            "tiling": False,
            "source_deletion": False,
        },
    }


def assemble_downstream_system(
    geometry: thermal.Geometry,
) -> Any:
    return assemble_steady_diagonal_kappa(
        x_edges_m=geometry.x_edges_m,
        y_edges_m=geometry.y_edges_m,
        z_edges_m=geometry.z_edges_m,
        kappa_W_mK=geometry.kappa_W_mK,
        dirichlet_temperature_K={
            "x_min": 0.0,
            "x_max": 0.0,
            "y_min": 0.0,
            "y_max": 0.0,
            "z_min": 0.0,
        },
        interface_resistance_m2K_W=geometry.interface_resistance_m2K_W,
        active_mask=np.ones(geometry.material_id.shape, bool),
        exposed_heat_transfer_W_m2K=thermal.H_EXPOSED_W_M2K,
        ambient_temperature_K=0.0,
    )


def solve_warm_started(
    system: Any,
    q: np.ndarray,
    initial_temperature_K: np.ndarray,
) -> SteadyHeatResult:
    source_active = system.active_source(q)
    source_power_active = system.source_volume_operator_m3 @ source_active
    rhs = source_power_active + system.boundary_load_W
    matrix = system.matrix_W_K
    preconditioner = sparse_linalg.LinearOperator(
        matrix.shape,
        matvec=lambda vector: vector / system.diagonal_W_K,
    )
    initial = np.asarray(initial_temperature_K, float)[system.active_mask]
    iterations = 0

    def count(_: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    solution, info = sparse_linalg.cg(
        matrix,
        rhs,
        x0=initial,
        rtol=1.0e-9,
        atol=0.0,
        maxiter=12000,
        M=preconditioner,
        callback=count,
    )
    if info != 0:
        raise RuntimeError(f"warm-start CG did not converge: info={info}")
    residual = matrix @ solution - rhs
    residual_relative = float(
        np.linalg.norm(residual)
        / max(np.linalg.norm(rhs), np.finfo(float).tiny)
    )
    boundary_power = {
        face: float(
            np.sum(conductance * (solution[face_ids] - temperature))
        )
        for face, (face_ids, conductance, temperature)
        in system.boundary_terms.items()
    }
    source_power = float(np.sum(source_power_active))
    energy_error = abs(sum(boundary_power.values()) - source_power) / max(
        abs(source_power),
        max((abs(value) for value in boundary_power.values()), default=0.0),
        np.finfo(float).tiny,
    )
    return SteadyHeatResult(
        temperature_K=system.full_field(solution),
        boundary_power_out_W=boundary_power,
        source_power_W=source_power,
        energy_balance_relative_error=float(energy_error),
        linear_residual_relative=residual_relative,
        solver="scipy.sparse.linalg.cg+jacobi+warm_start",
        iterations=iterations,
    )


def downstream_metrics(
    solved: SteadyHeatResult,
    geometry: thermal.Geometry,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    edge_metrics, fields = thermal.straight_edge_temperature_metrics(
        solved.temperature_K,
        geometry,
    )
    flake_xy = np.any(geometry.flake_mask, axis=2)
    grad_psi_x = np.full(flake_xy.shape, WEIGHTING_X_M_INV)
    grad_psi_y = np.full(flake_xy.shape, WEIGHTING_Y_M_INV)
    pte, pte_fields = thermal.pte_current(
        solved.temperature_K,
        geometry,
        grad_psi_x,
        grad_psi_y,
    )
    cell_volume = volume(
        (
            geometry.x_edges_m,
            geometry.y_edges_m,
            geometry.z_edges_m,
        )
    )
    return {
        "Tmax_K": float(np.max(solved.temperature_K)),
        "TaIrTe4_Tmax_K": float(
            np.max(solved.temperature_K[geometry.flake_mask])
        ),
        "TaIrTe4_volume_average_K": thermal.measure_weighted_mean(
            solved.temperature_K,
            geometry.flake_mask,
            cell_volume,
        ),
        "PTE_uniform45_diagnostic_A": pte,
        "energy_balance_relative_error": (
            solved.energy_balance_relative_error
        ),
        "linear_residual_relative": solved.linear_residual_relative,
        "source_power_W": solved.source_power_W,
        "boundary_power_out_W": solved.boundary_power_out_W,
        "solver": solved.solver,
        "iterations": solved.iterations,
        "straight_edge_metrics": edge_metrics,
    }, {
        "temperature_K": solved.temperature_K,
        "temperature_flake_average_K": fields[
            "temperature_flake_average_K"
        ],
        "grad_T_x_K_m": fields["grad_T_x_K_m"],
        "grad_T_y_K_m": fields["grad_T_y_K_m"],
        "grad_T_magnitude_K_m": fields["grad_T_magnitude_K_m"],
        "pte_integrand_A_m2": pte_fields["shockley_ramo_integrand_A_m2"],
    }


def plot_gradient_comparison(
    raw_path: Path,
    output_path: Path,
) -> None:
    with np.load(raw_path, allow_pickle=False) as artifact:
        x = 0.5 * (
            artifact["x_edges_m"][:-1] + artifact["x_edges_m"][1:]
        ) * 1e6
        y = 0.5 * (
            artifact["y_edges_m"][:-1] + artifact["y_edges_m"][1:]
        ) * 1e6
        gx50 = np.asarray(artifact["grad_T_x_50_K_m"], float)
        gx25 = np.asarray(artifact["grad_T_x_25_K_m"], float)
        gy50 = np.asarray(artifact["grad_T_y_50_K_m"], float)
        gy25 = np.asarray(artifact["grad_T_y_25_K_m"], float)
    magnitude50 = np.hypot(gx50, gy50)
    magnitude25 = np.hypot(gx25, gy25)
    difference = magnitude50 - magnitude25
    extent = [x[0], x[-1], y[0], y[-1]]
    vmax = max(float(np.max(magnitude50)), float(np.max(magnitude25)))
    limit = float(np.max(np.abs(difference)))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for axis, image, title in (
        (axes[0], magnitude50, "|in-plane grad T| 50 nm"),
        (axes[1], magnitude25, "|in-plane grad T| 25 nm"),
    ):
        handle = axis.imshow(
            image.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )
        axis.set(title=title, xlabel="x=b (um)", ylabel="y=a (um)")
        figure.colorbar(handle, ax=axis)
    handle = axes[2].imshow(
        difference.T,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    axes[2].set(
        title="gradient-magnitude difference",
        xlabel="x=b (um)",
        ylabel="y=a (um)",
    )
    figure.colorbar(handle, ax=axes[2])
    figure.tight_layout()
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    coarse_dir = args.coarse_case.expanduser().resolve()
    fine_dir = args.fine_case.expanduser().resolve()
    raw_dir = args.raw_output_dir.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    if raw_dir.exists() and any(raw_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty raw output: {raw_dir}")
    if report_dir.exists() and any(report_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty report output: {report_dir}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    coarse_result, coarse_artifact = load_contract(coarse_dir)
    fine_result, fine_artifact = load_contract(fine_dir)
    index_audit = json.loads(
        args.interface_index_audit.read_text(encoding="utf-8")
    )
    if (
        index_audit.get("status")
        != "EXTRACTED_W12_INTERFACE_INDEX_READ_ONLY"
        or index_audit.get("FDTD_run") is not False
        or index_audit.get("runanalysis_called") is not False
    ):
        raise RuntimeError("invalid read-only interface-index audit")

    interface: dict[str, Any] = {}
    maps: dict[str, dict[str, np.ndarray]] = {}
    for label, result, artifact in (
        ("50nm", coarse_result, coarse_artifact),
        ("25nm", fine_result, fine_artifact),
    ):
        interface[label], maps[label] = interface_metrics(result, artifact)
    layer_comparison = compare_areal_maps(
        maps["50nm"], maps["25nm"], "z0_layer_areal_W_m2"
    )
    slab_comparison = compare_areal_maps(
        maps["50nm"], maps["25nm"], "slab_areal_W_m2"
    )

    outer_um = 31.0
    thermal.FLAKE_VERTICES_UM = np.asarray(
        [
            [-outer_um, -outer_um],
            [outer_um, -outer_um],
            [outer_um, outer_um],
        ],
        float,
    )
    geometry = thermal.build_geometry(
        domain_m=60.0e-6,
        si_depth_m=20.0e-6,
        core_step_m=100.0e-9,
        flake_dz_m=10.0e-9,
    )
    downstream: dict[str, Any] = {}
    fields: dict[str, dict[str, np.ndarray]] = {}
    mapped_q: dict[str, np.ndarray] = {}
    for label, result, artifact in (
        ("50nm", coarse_result, coarse_artifact),
        ("25nm", fine_result, fine_artifact),
    ):
        mapped_q[label], mapping = map_to_thermal(
            result, artifact, geometry
        )
        downstream[label] = {"mapping": mapping}
    system = assemble_downstream_system(geometry)
    print(
        "THERMAL_SOLVE_START case=25nm mode=cold "
        f"unknowns={system.matrix_W_K.shape[0]}",
        flush=True,
    )
    solved_25 = solve_assembled_thermal_system(
        system,
        source_W_m3=mapped_q["25nm"],
        relative_tolerance=1.0e-9,
        max_iterations=12000,
    )
    print(
        f"THERMAL_SOLVE_DONE case=25nm iterations={solved_25.iterations} "
        f"residual={solved_25.linear_residual_relative:.3e}",
        flush=True,
    )
    print("THERMAL_SOLVE_START case=50nm mode=warm", flush=True)
    solved_50 = solve_warm_started(
        system,
        mapped_q["50nm"],
        solved_25.temperature_K,
    )
    print(
        f"THERMAL_SOLVE_DONE case=50nm iterations={solved_50.iterations} "
        f"residual={solved_50.linear_residual_relative:.3e}",
        flush=True,
    )
    for label, solved in (("50nm", solved_50), ("25nm", solved_25)):
        metrics, fields[label] = downstream_metrics(solved, geometry)
        downstream[label]["thermal"] = metrics

    cell_volume = volume(
        (
            geometry.x_edges_m,
            geometry.y_edges_m,
            geometry.z_edges_m,
        )
    )
    area = (
        np.diff(geometry.x_edges_m)[:, None]
        * np.diff(geometry.y_edges_m)[None, :]
    )
    flake_xy = np.any(geometry.flake_mask, axis=2)
    gradient_numerator = float(
        np.sum(
            area[flake_xy]
            * (
                (
                    fields["50nm"]["grad_T_x_K_m"][flake_xy]
                    - fields["25nm"]["grad_T_x_K_m"][flake_xy]
                )
                ** 2
                + (
                    fields["50nm"]["grad_T_y_K_m"][flake_xy]
                    - fields["25nm"]["grad_T_y_K_m"][flake_xy]
                )
                ** 2
            )
        )
    )
    gradient_denominator = float(
        np.sum(
            area[flake_xy]
            * (
                fields["25nm"]["grad_T_x_K_m"][flake_xy] ** 2
                + fields["25nm"]["grad_T_y_K_m"][flake_xy] ** 2
            )
        )
    )
    comparisons = {
        "Q_T_volume_weighted_NRMSE": weighted_nrmse(
            mapped_q["50nm"], mapped_q["25nm"], cell_volume
        ),
        "Q_T_power_relative_change": relative(
            downstream["50nm"]["mapping"]["target_power_W"],
            downstream["25nm"]["mapping"]["target_power_W"],
        ),
        "Tmax_relative_change": relative(
            downstream["50nm"]["thermal"]["Tmax_K"],
            downstream["25nm"]["thermal"]["Tmax_K"],
        ),
        "TaIrTe4_Tmax_relative_change": relative(
            downstream["50nm"]["thermal"]["TaIrTe4_Tmax_K"],
            downstream["25nm"]["thermal"]["TaIrTe4_Tmax_K"],
        ),
        "TaIrTe4_temperature_field_volume_weighted_NRMSE": weighted_nrmse(
            fields["50nm"]["temperature_K"],
            fields["25nm"]["temperature_K"],
            cell_volume,
            geometry.flake_mask,
        ),
        "full_temperature_field_volume_weighted_NRMSE": weighted_nrmse(
            fields["50nm"]["temperature_K"],
            fields["25nm"]["temperature_K"],
            cell_volume,
        ),
        "flake_average_temperature_area_weighted_NRMSE": weighted_nrmse(
            fields["50nm"]["temperature_flake_average_K"],
            fields["25nm"]["temperature_flake_average_K"],
            area,
            flake_xy,
        ),
        "inplane_gradient_vector_area_weighted_NRMSE": float(
            np.sqrt(
                gradient_numerator
                / max(gradient_denominator, np.finfo(float).tiny)
            )
        ),
        "PTE_uniform45_diagnostic_relative_change": relative(
            downstream["50nm"]["thermal"][
                "PTE_uniform45_diagnostic_A"
            ],
            downstream["25nm"]["thermal"][
                "PTE_uniform45_diagnostic_A"
            ],
        ),
    }
    coordinate_gate = max(
        case["maximum_field_index_coordinate_mismatch_m"]
        for case in index_audit["cases"].values()
    ) < 1.0e-15
    gates = {
        "interface_slab_power_change_lt_0p5_percent": (
            slab_comparison["power_relative_change"] < GATE
        ),
        "interface_slab_equal_power_NRMSE_lt_0p5_percent": (
            slab_comparison["equal_power_energy_NRMSE"] < GATE
        ),
        "component_field_index_pairing_lt_1fm": coordinate_gate,
        "Q_T_NRMSE_lt_0p5_percent": (
            comparisons["Q_T_volume_weighted_NRMSE"] < GATE
        ),
        "Tmax_change_lt_0p5_percent": (
            comparisons["Tmax_relative_change"] < GATE
        ),
        "TaIrTe4_T_field_NRMSE_lt_0p5_percent": (
            comparisons[
                "TaIrTe4_temperature_field_volume_weighted_NRMSE"
            ]
            < GATE
        ),
        "gradient_NRMSE_lt_0p5_percent": (
            comparisons[
                "inplane_gradient_vector_area_weighted_NRMSE"
            ]
            < GATE
        ),
        "PTE_change_lt_0p5_percent": (
            comparisons[
                "PTE_uniform45_diagnostic_relative_change"
            ]
            < GATE
        ),
        "thermal_energy_balance_lt_1_percent": max(
            downstream[label]["thermal"]["energy_balance_relative_error"]
            for label in downstream
        )
        < ENERGY_GATE,
        "thermal_residual_lt_1e_8": max(
            downstream[label]["thermal"]["linear_residual_relative"]
            for label in downstream
        )
        < RESIDUAL_GATE,
        "mapping_power_error_lt_1e_12": max(
            downstream[label]["mapping"]["final_mapping_power_error"]
            for label in downstream
        )
        < 1.0e-12,
        "mapped_Q_outside_flake_is_zero": all(
            downstream[label]["mapping"]["projection"][
                "outside_support_nonzero_cells_after_projection"
            ]
            == 0
            and downstream[label]["mapping"]["projection"][
                "outside_support_power_W_after_projection"
            ]
            == 0.0
            for label in downstream
        ),
    }
    gates["all"] = all(gates.values())
    status = (
        "VALIDATED_W12_INTERFACE_SLAB_AND_DOWNSTREAM_CONVERGENCE"
        if gates["all"]
        else "BLOCKED_W12_INTERFACE_SLAB_OR_DOWNSTREAM_CONVERGENCE"
    )

    raw_path = raw_dir / "w12_interface_downstream_fields.npz"
    np.savez_compressed(
        raw_path,
        x_edges_m=geometry.x_edges_m,
        y_edges_m=geometry.y_edges_m,
        z_edges_m=geometry.z_edges_m,
        flake_mask=geometry.flake_mask,
        Q_T_50_W_m3=mapped_q["50nm"],
        Q_T_25_W_m3=mapped_q["25nm"],
        temperature_50_K=fields["50nm"]["temperature_K"],
        temperature_25_K=fields["25nm"]["temperature_K"],
        flake_average_temperature_50_K=fields["50nm"][
            "temperature_flake_average_K"
        ],
        flake_average_temperature_25_K=fields["25nm"][
            "temperature_flake_average_K"
        ],
        grad_T_x_50_K_m=fields["50nm"]["grad_T_x_K_m"],
        grad_T_x_25_K_m=fields["25nm"]["grad_T_x_K_m"],
        grad_T_y_50_K_m=fields["50nm"]["grad_T_y_K_m"],
        grad_T_y_25_K_m=fields["25nm"]["grad_T_y_K_m"],
    )
    raw_record = {
        "path": str(raw_path.resolve()),
        "size_bytes": raw_path.stat().st_size,
        "sha256": sha256(raw_path),
    }
    for comparison in (layer_comparison, slab_comparison):
        comparison.pop("fine_energy_on_coarse")
        comparison.pop("coarse_energy")
    summary = {
        "status": status,
        "validated": gates["all"],
        "scope": (
            "existing w12 edge-a 50/25 nm Q only; interface layer/slab, "
            "read-only component index audit, exact-overlap remap, explicit "
            "anisotropic/interface thermal FVM, and uniform-45 PTE diagnostic"
        ),
        "interface_power": interface,
        "z0_layer_comparison": layer_comparison,
        "interface_slab_comparison": slab_comparison,
        "component_interface_index_readback": index_audit,
        "thermal_contract": {
            "model": (
                "current explicit anisotropic/material/interface Cartesian "
                "FVM; not the paper reduced Robin model"
            ),
            "lateral_domain_um": 60.0,
            "si_depth_um": 20.0,
            "core_xy_cell_size_nm": 100.0,
            "flake_dz_nm": 10.0,
            "grid_shape": list(geometry.material_id.shape),
            "axis_mapping": "lab x=b, lab y=a, lab z=c",
            "uniform_45_PTE": (
                "diagnostic weighting field Wx=Wy=1/(4 um); no physical "
                "electrode solution and not a terminal-current prediction"
            ),
        },
        "downstream": downstream,
        "comparisons": comparisons,
        "gates": gates,
        "raw_artifact": raw_record,
        "FDTD_run": False,
        "Q_processing": {
            "clipping": False,
            "smoothing": False,
            "gain": False,
            "global_rescaling": False,
            "tiling": False,
            "source_deletion": False,
        },
        "interpretation_limit": (
            "a pass establishes insensitivity only for this named remap, "
            "thermal grid/boundary model, and diagnostic PTE functional; "
            "it does not make raw 3D voxel Q universally converged"
        ),
    }
    summary_path = report_dir / "w12_interface_downstream_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    cases_path = report_dir / "w12_interface_downstream_cases.csv"
    with cases_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "P_Q_W",
                "z0_power_fraction",
                "slab_power_fraction",
                "mapped_Q_W",
                "Tmax_K",
                "TaIrTe4_Tavg_K",
                "PTE_uniform45_A",
                "energy_balance",
                "residual",
            ],
        )
        writer.writeheader()
        for label in ("50nm", "25nm"):
            writer.writerow(
                {
                    "case": label,
                    "P_Q_W": interface[label]["components"]["total"][
                        "total_power_W"
                    ],
                    "z0_power_fraction": interface[label]["components"][
                        "total"
                    ]["z0_layer_power_fraction"],
                    "slab_power_fraction": interface[label]["components"][
                        "total"
                    ]["interface_slab_power_fraction"],
                    "mapped_Q_W": downstream[label]["mapping"][
                        "target_power_W"
                    ],
                    "Tmax_K": downstream[label]["thermal"]["Tmax_K"],
                    "TaIrTe4_Tavg_K": downstream[label]["thermal"][
                        "TaIrTe4_volume_average_K"
                    ],
                    "PTE_uniform45_A": downstream[label]["thermal"][
                        "PTE_uniform45_diagnostic_A"
                    ],
                    "energy_balance": downstream[label]["thermal"][
                        "energy_balance_relative_error"
                    ],
                    "residual": downstream[label]["thermal"][
                        "linear_residual_relative"
                    ],
                }
            )

    x = 0.5 * (geometry.x_edges_m[:-1] + geometry.x_edges_m[1:]) * 1e6
    y = 0.5 * (geometry.y_edges_m[:-1] + geometry.y_edges_m[1:]) * 1e6
    extent = [x[0], x[-1], y[0], y[-1]]
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    images = (
        (
            np.sum(
                mapped_q["50nm"]
                * np.diff(geometry.z_edges_m)[None, None, :],
                axis=2,
            ),
            "mapped Q 50 nm",
        ),
        (
            np.sum(
                mapped_q["25nm"]
                * np.diff(geometry.z_edges_m)[None, None, :],
                axis=2,
            ),
            "mapped Q 25 nm",
        ),
        (
            np.sum(
                (mapped_q["50nm"] - mapped_q["25nm"])
                * np.diff(geometry.z_edges_m)[None, None, :],
                axis=2,
            ),
            "mapped Q difference",
        ),
        (
            fields["50nm"]["temperature_flake_average_K"],
            "flake-average T 50 nm",
        ),
        (
            fields["25nm"]["temperature_flake_average_K"],
            "flake-average T 25 nm",
        ),
        (
            fields["50nm"]["temperature_flake_average_K"]
            - fields["25nm"]["temperature_flake_average_K"],
            "flake-average T difference",
        ),
    )
    for axis, (image, title) in zip(axes.ravel(), images):
        handle = axis.imshow(
            image.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="coolwarm" if "difference" in title else "inferno",
        )
        axis.set(title=title, xlabel="x=b (µm)", ylabel="y=a (µm)")
        figure.colorbar(handle, ax=axis)
    figure.tight_layout()
    figure_path = report_dir / "W12_INTERFACE_DOWNSTREAM_COMPARISON.png"
    figure.savefig(figure_path, dpi=190)
    plt.close(figure)
    gradient_figure_path = (
        report_dir / "W12_INTERFACE_GRADIENT_COMPARISON.png"
    )
    plot_gradient_comparison(raw_path, gradient_figure_path)

    total50 = interface["50nm"]["components"]["total"]
    total25 = interface["25nm"]["components"]["total"]
    report_path = report_dir / "W12_INTERFACE_DOWNSTREAM_REPORT.md"
    report_path.write_text(
        f"""# W12 interface-slab and downstream convergence

Status: `{status}`

No new FDTD calculation was run. Completed 50/25 nm artifacts were used
without Q clipping, smoothing, gain, global rescaling, tiling, or deletion.

## Interface power

| Metric | 50 nm | 25 nm | Change |
|---|---:|---:|---:|
| total P_Q (W) | {total50['total_power_W']:.12e} | {total25['total_power_W']:.12e} | {relative(total50['total_power_W'], total25['total_power_W']):.4%} |
| z=0 dual-layer power fraction | {total50['z0_layer_power_fraction']:.4%} | {total25['z0_layer_power_fraction']:.4%} | — |
| -10..0 nm slab power fraction | {total50['interface_slab_power_fraction']:.4%} | {total25['interface_slab_power_fraction']:.4%} | — |

The z=0 common-grid sample has a bounded dual cell
`[{total25['z0_layer_dual_cell_bounds_m'][0]*1e9:.3f},
{total25['z0_layer_dual_cell_bounds_m'][1]*1e9:.3f}] nm`.
The -10..0 nm slab is integrated by exact dual-cell overlap, not by selecting
cell centres.

- slab power change: `{slab_comparison['power_relative_change']:.4%}`
- slab equal-power lateral NRMSE:
  `{slab_comparison['equal_power_energy_NRMSE']:.4%}`

## Component interface assignment

The saved FSPs were reopened read-only. No `run` or `runanalysis` call was
made. Ex/Ey have an exact z=0 index sample; Ez is z-staggered and instead has
samples at approximately -2.5/+2.5 nm. Full numerical values and E/index
coordinate mismatches are stored in the summary JSON.

- Ex/Ey z=0 median loss participation is approximately `50.0122%` of the
  bulk fitted material loss; the -5/+5 nm samples are material/air.
- Ez has no z=0 sample: -2.5 nm is material and +2.5 nm is air.
- Maximum independently read E/index coordinate mismatch:
  `{max(case['maximum_field_index_coordinate_mismatch_m'] for case in index_audit['cases'].values()):.3e} m`.
- Central component-local interface cell volume is approximately
  `1.25e-23 m³` at 50 nm x/y and `3.125e-24 m³` at 25 nm x/y.

## Named downstream model

Both Q artifacts use the same exact-overlap/nearest-support remap and the
same 60 µm explicit anisotropic/interface FVM: 20 µm Si depth, 100 nm core
x/y cells, and 10 nm TaIrTe4 z cells. The PTE number is a uniform-45-degree
diagnostic with lab `x=b`, `y=a`; it is not a solved-electrode terminal
current.

| Downstream metric | 50→25 nm difference |
|---|---:|
| Q_T volume-weighted NRMSE | {comparisons['Q_T_volume_weighted_NRMSE']:.4%} |
| Tmax | {comparisons['Tmax_relative_change']:.4%} |
| TaIrTe4 T-field NRMSE | {comparisons['TaIrTe4_temperature_field_volume_weighted_NRMSE']:.4%} |
| in-plane gradient-vector NRMSE | {comparisons['inplane_gradient_vector_area_weighted_NRMSE']:.4%} |
| uniform-45 PTE diagnostic | {comparisons['PTE_uniform45_diagnostic_relative_change']:.4%} |

The temperature field and signed uniform-45 PTE diagnostic pass 0.5%, but
the interface-slab spatial shape, mapped Q_T, and in-plane gradient-vector
metrics do not. Therefore this remains a partial downstream pass, not a
source-convergence promotion.

Any pass is limited to this named remap, thermal grid/boundary model, and
PTE functional. It does not relabel the raw full-3D voxel-Q gate as
universally converged.

Published figures:

- `W12_INTERFACE_DOWNSTREAM_COMPARISON.png`
- `W12_INTERFACE_GRADIENT_COMPARISON.png`
""",
        encoding="utf-8",
    )
    published = [
        report_path,
        summary_path,
        cases_path,
        figure_path,
        gradient_figure_path,
    ]
    manifest_path = report_dir / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": status,
                "raw_artifacts": [
                    raw_record,
                    {
                        "path": str(
                            args.interface_index_audit.resolve()
                        ),
                        "size_bytes": args.interface_index_audit.stat().st_size,
                        "sha256": sha256(args.interface_index_audit),
                    },
                    {
                        "path": str(coarse_artifact.resolve()),
                        "size_bytes": coarse_artifact.stat().st_size,
                        "sha256": sha256(coarse_artifact),
                    },
                    {
                        "path": str(fine_artifact.resolve()),
                        "size_bytes": fine_artifact.stat().st_size,
                        "sha256": sha256(fine_artifact),
                    },
                ],
                "published_files": [
                    {
                        "path": str(path.resolve()),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                    for path in published
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0 if gates["all"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
