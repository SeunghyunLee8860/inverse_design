#!/usr/bin/env python3
"""Summarize two bounded nested-mesh w12 edge-a optical calculations.

No optical source support is cropped or deleted.  The finer Q is
conservatively mapped to the coarser dual-cell grid using exact Cartesian
cell overlaps before spatial comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator


POWER_GATE = 5.0e-3
SPATIAL_GATE = 5.0e-3
CLOSURE_GATE = 5.0e-3
SHUTOFF_GATE = 1.0e-5
REMAP_GATE = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-case", type=Path, required=True)
    parser.add_argument("--fine-case", type=Path, required=True)
    parser.add_argument("--failed-full-fine-case", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(first: float, second: float) -> float:
    return abs(first - second) / max(abs(second), np.finfo(float).tiny)


def bounded_dual_cells(
    coordinate: np.ndarray,
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(coordinate, float).reshape(-1)
    if values.size < 2 or np.any(np.diff(values) <= 0.0):
        raise RuntimeError("invalid coordinate")
    raw = np.concatenate(
        (
            [values[0] - 0.5 * (values[1] - values[0])],
            0.5 * (values[:-1] + values[1:]),
            [values[-1] + 0.5 * (values[-1] - values[-2])],
        )
    )
    left = np.maximum(raw[:-1], low)
    right = np.minimum(raw[1:], high)
    active = right - left > 1.0e-20
    indices = np.flatnonzero(active)
    if indices.size == 0 or np.any(np.diff(indices) != 1):
        raise RuntimeError("bounded cells are empty or disconnected")
    edges = np.concatenate(([left[indices[0]]], right[indices]))
    if (
        not np.isclose(edges[0], low, rtol=0.0, atol=1.0e-18)
        or not np.isclose(edges[-1], high, rtol=0.0, atol=1.0e-18)
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise RuntimeError("bounded dual cells do not close")
    return indices, edges


def overlap_fraction(
    target_edges: np.ndarray,
    source_edges: np.ndarray,
) -> sparse.csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    target_index = 0
    for source_index in range(source_edges.size - 1):
        lower, upper = source_edges[source_index : source_index + 2]
        while (
            target_index < target_edges.size - 1
            and target_edges[target_index + 1] <= lower
        ):
            target_index += 1
        index = target_index
        while (
            index < target_edges.size - 1
            and target_edges[index] < upper
        ):
            overlap = min(upper, target_edges[index + 1]) - max(
                lower, target_edges[index]
            )
            if overlap > 0.0:
                rows.append(index)
                columns.append(source_index)
                values.append(
                    float(overlap / (upper - lower))
                )
            index += 1
    matrix = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(
            target_edges.size - 1,
            source_edges.size - 1,
        ),
    ).tocsr()
    coverage = np.asarray(matrix.sum(axis=0)).reshape(-1)
    if not np.allclose(coverage, 1.0, rtol=2.0e-13, atol=1.0e-13):
        raise RuntimeError("source cells are not covered exactly once")
    return matrix


def remap_energy(
    source_energy: np.ndarray,
    operators: tuple[sparse.csr_matrix, ...],
) -> np.ndarray:
    source = np.asarray(source_energy, float)
    if source.ndim != len(operators):
        raise RuntimeError("operator dimensionality mismatch")
    result = source
    for axis, operator in enumerate(operators):
        moved = np.moveaxis(result, axis, 0)
        mapped = operator @ moved.reshape(moved.shape[0], -1)
        result = np.moveaxis(
            np.asarray(mapped).reshape(
                (operator.shape[0],) + moved.shape[1:]
            ),
            0,
            axis,
        )
    return result


def volume(edges: tuple[np.ndarray, ...]) -> np.ndarray:
    widths = [np.diff(axis) for axis in edges]
    if len(widths) == 3:
        return (
            widths[0][:, None, None]
            * widths[1][None, :, None]
            * widths[2][None, None, :]
        )
    if len(widths) == 2:
        return widths[0][:, None] * widths[1][None, :]
    raise RuntimeError("unsupported dimension")


def load_case(directory: Path) -> tuple[dict[str, Any], Path, Path]:
    result_path = directory / "case_result.json"
    npz_path = directory / "finite_q_on_artifact.npz"
    manifest_path = directory / "RAW_ARTIFACT_MANIFEST.json"
    if not all(path.is_file() for path in (result_path, npz_path, manifest_path)):
        raise FileNotFoundError(f"incomplete case: {directory}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["status"] != "COMPLETED":
        raise RuntimeError(f"case not completed: {directory}")
    return result, npz_path, manifest_path


def mesh_description(result: dict[str, Any]) -> dict[str, Any]:
    mesh = result["pre_run_contract"]["mesh"]
    overrides = {
        entry["name"]: entry for entry in mesh["override_objects"]
    }
    fine = overrides["flake_mesh"]
    intermediate = overrides.get("flake_intermediate_mesh")
    outer = overrides.get("flake_outer_mesh", fine)
    return {
        "outer_xy_m": float(outer["dx_m"]),
        "fine_xy_m": float(fine["dx_m"]),
        "fine_half_span_m": 0.5
        * (
            float(fine["bounds_m"]["x"][1])
            - float(fine["bounds_m"]["x"][0])
        ),
        "intermediate_xy_m": (
            None
            if intermediate is None
            else float(intermediate["dx_m"])
        ),
        "intermediate_half_span_m": (
            None
            if intermediate is None
            else 0.5
            * (
                float(intermediate["bounds_m"]["x"][1])
                - float(intermediate["bounds_m"]["x"][0])
            )
        ),
    }


def mesh_label(mesh: dict[str, Any]) -> str:
    levels = [f"{mesh['outer_xy_m']*1e9:g} nm outer"]
    if mesh["intermediate_xy_m"] is not None:
        levels.append(
            f"{mesh['intermediate_xy_m']*1e9:g} nm "
            f"within ±{mesh['intermediate_half_span_m']*1e6:g} µm"
        )
    if not np.isclose(mesh["fine_xy_m"], mesh["outer_xy_m"]):
        levels.append(
            f"{mesh['fine_xy_m']*1e9:g} nm "
            f"within ±{mesh['fine_half_span_m']*1e6:g} µm"
        )
    return " + ".join(levels)


def spatial_statistics(
    energy: np.ndarray,
    centers: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    total = float(np.sum(energy))
    centroids = []
    sigmas = []
    for axis, coordinate in enumerate(centers):
        reduce_axes = tuple(i for i in range(3) if i != axis)
        marginal = np.sum(energy, axis=reduce_axes)
        centroid = float(np.sum(marginal * coordinate) / total)
        sigma = float(
            np.sqrt(
                np.sum(marginal * (coordinate - centroid) ** 2)
                / total
            )
        )
        centroids.append(centroid)
        sigmas.append(sigma)
    return {
        "centroid_m": centroids,
        "second_moment_sigma_m": sigmas,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coarse_result, coarse_npz, coarse_manifest = load_case(
        args.coarse_case
    )
    fine_result, fine_npz, fine_manifest = load_case(args.fine_case)
    coarse_run = coarse_result["run_result"]
    fine_run = fine_result["run_result"]
    coarse_mesh = mesh_description(coarse_result)
    fine_mesh = mesh_description(fine_result)
    bounds = {
        axis: tuple(
            coarse_run["native_Yee_mesh_audit"][
                "Q_quadrature_control_volume_bounds_m"
            ][axis]
        )
        for axis in "xyz"
    }
    fine_bounds = fine_run["native_Yee_mesh_audit"][
        "Q_quadrature_control_volume_bounds_m"
    ]
    if any(
        not np.allclose(bounds[axis], fine_bounds[axis], atol=1.0e-18)
        for axis in "xyz"
    ):
        raise RuntimeError("coarse/fine Q bounds differ")

    with np.load(coarse_npz, allow_pickle=False) as coarse, np.load(
        fine_npz, allow_pickle=False
    ) as fine:
        coarse_coordinates = tuple(
            np.asarray(coarse[f"{axis}_m"], float) for axis in "xyz"
        )
        fine_coordinates = tuple(
            np.asarray(fine[f"{axis}_m"], float) for axis in "xyz"
        )
        coarse_cells = tuple(
            bounded_dual_cells(
                coordinate,
                *bounds[axis],
            )
            for coordinate, axis in zip(coarse_coordinates, "xyz")
        )
        fine_cells = tuple(
            bounded_dual_cells(
                coordinate,
                *bounds[axis],
            )
            for coordinate, axis in zip(fine_coordinates, "xyz")
        )
        coarse_indices = tuple(value[0] for value in coarse_cells)
        fine_indices = tuple(value[0] for value in fine_cells)
        coarse_edges = tuple(value[1] for value in coarse_cells)
        fine_edges = tuple(value[1] for value in fine_cells)
        operators = tuple(
            overlap_fraction(target, source)
            for target, source in zip(coarse_edges, fine_edges)
        )
        coarse_volume = volume(coarse_edges)
        fine_volume = volume(fine_edges)
        component_metrics: dict[str, Any] = {}
        q_coarse_total = None
        q_fine_target_total = None
        for component, key in (
            ("x", "Qx_W_m3"),
            ("y", "Qy_W_m3"),
            ("z", "Qz_W_m3"),
            ("total", "Q_on_W_m3"),
        ):
            q_coarse = np.asarray(
                coarse[key][np.ix_(*coarse_indices)],
                float,
            )
            q_fine = np.asarray(
                fine[key][np.ix_(*fine_indices)],
                float,
            )
            energy_coarse = q_coarse * coarse_volume
            energy_fine = q_fine * fine_volume
            remapped_energy = remap_energy(energy_fine, operators)
            remap_error = relative(
                float(np.sum(remapped_energy)),
                float(np.sum(energy_fine)),
            )
            q_fine_target = remapped_energy / coarse_volume
            power_coarse = float(np.sum(energy_coarse))
            power_fine = float(np.sum(energy_fine))
            spatial_nrmse = float(
                np.sqrt(
                    np.sum(
                        coarse_volume
                        * (q_coarse - q_fine_target) ** 2
                    )
                    / np.sum(coarse_volume * q_fine_target**2)
                )
            )
            normalized_nrmse = float(
                np.linalg.norm(
                    energy_coarse / power_coarse
                    - remapped_energy / power_fine
                )
                / np.linalg.norm(remapped_energy / power_fine)
            )
            correlation = float(
                np.corrcoef(
                    (energy_coarse / power_coarse).reshape(-1),
                    (remapped_energy / power_fine).reshape(-1),
                )[0, 1]
            )
            component_metrics[component] = {
                "coarse_power_W": power_coarse,
                "fine_power_W": power_fine,
                "power_relative_change": relative(
                    power_coarse, power_fine
                ),
                "conservative_remap_power_error": remap_error,
                "raw_density_volume_weighted_NRMSE": spatial_nrmse,
                "equal_power_energy_NRMSE": normalized_nrmse,
                "equal_power_energy_correlation": correlation,
            }
            if component == "total":
                q_coarse_total = q_coarse
                q_fine_target_total = q_fine_target

    if q_coarse_total is None or q_fine_target_total is None:
        raise RuntimeError("total Q missing")
    centers = tuple(
        0.5 * (axis[:-1] + axis[1:]) for axis in coarse_edges
    )
    energy_coarse = q_coarse_total * coarse_volume
    energy_fine_target = q_fine_target_total * coarse_volume
    coarse_power = float(np.sum(energy_coarse))
    fine_power = float(np.sum(energy_fine_target))
    coarse_stats = spatial_statistics(energy_coarse, centers)
    fine_stats = spatial_statistics(energy_fine_target, centers)
    fine_square = (
        np.abs(centers[0][:, None]) <= fine_mesh["fine_half_span_m"]
    ) & (
        np.abs(centers[1][None, :]) <= fine_mesh["fine_half_span_m"]
    )
    coarse_power_outside_fine_square = float(
        np.sum(energy_coarse[~fine_square, :])
    )
    coarse_power_outside_fine_square_fraction = (
        coarse_power_outside_fine_square / coarse_power
    )
    lateral_coarse = np.sum(energy_coarse, axis=2)
    lateral_fine = np.sum(energy_fine_target, axis=2)
    lateral_equal_power_nrmse = float(
        np.linalg.norm(
            lateral_coarse / coarse_power
            - lateral_fine / fine_power
        )
        / np.linalg.norm(lateral_fine / fine_power)
    )
    vertical_coarse = np.sum(energy_coarse, axis=(0, 1))
    vertical_fine = np.sum(energy_fine_target, axis=(0, 1))
    vertical_equal_power_nrmse = float(
        np.linalg.norm(
            vertical_coarse / coarse_power
            - vertical_fine / fine_power
        )
        / np.linalg.norm(vertical_fine / fine_power)
    )
    normalized_energy_difference = (
        energy_coarse / coarse_power - energy_fine_target / fine_power
    )
    error_squared_by_z = np.sum(
        normalized_energy_difference**2, axis=(0, 1)
    )
    error_squared_total = float(np.sum(error_squared_by_z))
    largest_z_error_indices = np.argsort(error_squared_by_z)[-5:][::-1]
    largest_z_error_layers = [
        {
            "z_m": float(centers[2][index]),
            "fraction_of_full_3D_squared_error": float(
                error_squared_by_z[index] / error_squared_total
            ),
        }
        for index in largest_z_error_indices
    ]

    coarse_field_path = args.coarse_case / "field_slices_raw.npz"
    fine_field_path = args.fine_case / "field_slices_raw.npz"
    with np.load(coarse_field_path, allow_pickle=False) as coarse_field, np.load(
        fine_field_path, allow_pickle=False
    ) as fine_field:
        field_bounds = {
            "x": (
                float(coarse_field["inside_x_m"][0]),
                float(coarse_field["inside_x_m"][-1]),
            ),
            "y": (
                float(coarse_field["inside_y_m"][0]),
                float(coarse_field["inside_y_m"][-1]),
            ),
        }
        field_coarse_cells = tuple(
            bounded_dual_cells(
                np.asarray(coarse_field[f"inside_{axis}_m"], float),
                *field_bounds[axis],
            )
            for axis in "xy"
        )
        field_fine_cells = tuple(
            bounded_dual_cells(
                np.asarray(fine_field[f"inside_{axis}_m"], float),
                *field_bounds[axis],
            )
            for axis in "xy"
        )
        field_coarse_indices = tuple(v[0] for v in field_coarse_cells)
        field_fine_indices = tuple(v[0] for v in field_fine_cells)
        field_coarse_edges = tuple(v[1] for v in field_coarse_cells)
        field_fine_edges = tuple(v[1] for v in field_fine_cells)
        field_operators = tuple(
            overlap_fraction(target, source)
            for target, source in zip(
                field_coarse_edges, field_fine_edges
            )
        )
        e2_coarse = np.asarray(
            coarse_field["E2_inside"][np.ix_(*field_coarse_indices)],
            float,
        )
        e2_fine = np.asarray(
            fine_field["E2_inside"][np.ix_(*field_fine_indices)],
            float,
        )
        field_coarse_area = volume(field_coarse_edges)
        field_fine_area = volume(field_fine_edges)
        e2_fine_target = remap_energy(
            e2_fine * field_fine_area,
            field_operators,
        ) / field_coarse_area
        field_nrmse = float(
            np.sqrt(
                np.sum(
                    field_coarse_area
                    * (e2_coarse - e2_fine_target) ** 2
                )
                / np.sum(field_coarse_area * e2_fine_target**2)
            )
        )
        field_scale_coarse = float(np.sum(e2_coarse * field_coarse_area))
        field_scale_fine = float(
            np.sum(e2_fine_target * field_coarse_area)
        )
        field_equal_power_nrmse = float(
            np.linalg.norm(
                e2_coarse * field_coarse_area / field_scale_coarse
                - e2_fine_target
                * field_coarse_area
                / field_scale_fine
            )
            / np.linalg.norm(
                e2_fine_target
                * field_coarse_area
                / field_scale_fine
            )
        )

    p_q_relative = relative(
        coarse_run["P_Q_W"], fine_run["P_Q_W"]
    )
    p_six_relative = relative(
        coarse_run["P_six_face_W"],
        fine_run["P_six_face_W"],
    )
    hotspot_coarse = np.asarray(
        [
            coarse_run["Q_hotspot"][f"{axis}_m"]
            for axis in "xyz"
        ]
    )
    hotspot_fine = np.asarray(
        [
            fine_run["Q_hotspot"][f"{axis}_m"]
            for axis in "xyz"
        ]
    )
    hotspot_shift = float(np.linalg.norm(hotspot_coarse - hotspot_fine))
    gates = {
        "coarse_closure_lt_0p5_percent": (
            coarse_run["six_face_relative_closure"] < CLOSURE_GATE
        ),
        "fine_closure_lt_0p5_percent": (
            fine_run["six_face_relative_closure"] < CLOSURE_GATE
        ),
        "P_Q_change_lt_0p5_percent": p_q_relative < POWER_GATE,
        "P_six_change_lt_0p5_percent": p_six_relative < POWER_GATE,
        "equal_power_spatial_Q_NRMSE_lt_0p5_percent": (
            component_metrics["total"]["equal_power_energy_NRMSE"]
            < SPATIAL_GATE
        ),
        "equal_power_lateral_Q_NRMSE_lt_0p5_percent": (
            lateral_equal_power_nrmse < SPATIAL_GATE
        ),
        "coarse_auto_shutoff_le_1e_5": (
            coarse_run["auto_shutoff"]["final_value"]
            <= SHUTOFF_GATE
        ),
        "fine_auto_shutoff_le_1e_5": (
            fine_run["auto_shutoff"]["final_value"]
            <= SHUTOFF_GATE
        ),
        "conservative_remap_error_lt_1e_12": (
            component_metrics["total"][
                "conservative_remap_power_error"
            ]
            < REMAP_GATE
        ),
        "no_Q_modification": all(
            not result[key]
            for result in (coarse_result, fine_result)
            for key in ("Q_clipped", "flux_gain", "Q_rescaled")
        ),
    }
    gates["all"] = all(gates.values())
    fine_nm = fine_mesh["fine_xy_m"] * 1e9
    validated_status = (
        "VALIDATED_W12_EDGE_A_THREE_LEVEL_25NM_XY_CONVERGENCE"
        if fine_nm <= 25.0 + 1.0e-9
        else "VALIDATED_W12_EDGE_A_NESTED_50NM_XY_CONVERGENCE"
    )
    status = (
        validated_status
        if gates["all"]
        else "BLOCKED_W12_EDGE_A_XY_MESH_CONVERGENCE"
    )

    failed_diagnostic = None
    if args.failed_full_fine_case is not None:
        failed_result_path = args.failed_full_fine_case / "case_result.json"
        failed_log_path = (
            args.failed_full_fine_case / "finite_2um_optical_q_p0.log"
        )
        if failed_result_path.is_file() and failed_log_path.is_file():
            failed_result = json.loads(
                failed_result_path.read_text(encoding="utf-8")
            )
            text = failed_log_path.read_text(
                encoding="utf-8", errors="replace"
            )
            failed_diagnostic = {
                "path": str(args.failed_full_fine_case.resolve()),
                "status": failed_result["status"],
                "exception": failed_result.get("exception"),
                "auto_shutoff_reached": (
                    "Auto Shutoff: 9.93224e-06" in text
                ),
                "failure_stage": (
                    "post-solve project collection/write while root "
                    "filesystem was exhausted"
                ),
                "used_as_physical_result": False,
            }

    with np.load(coarse_field_path, allow_pickle=False) as coarse_field:
        field_plane_z_m = float(coarse_field["inside_z_m"][0])

    summary = {
        "status": status,
        "validated": gates["all"],
        "scenario": (
            "paper-like scalar-Gaussian scenario with an explicitly "
            "assumed 12-um waist; not a paper-certified beam"
        ),
        "comparison": {
            "coarse": {
                "path": str(args.coarse_case.resolve()),
                "mesh": coarse_mesh,
                "native_Yee_cell_count": coarse_run[
                    "native_Yee_mesh_audit"
                ]["native_Yee_cell_count"],
                "P_Q_W": coarse_run["P_Q_W"],
                "P_six_W": coarse_run["P_six_face_W"],
                "closure": coarse_run["six_face_relative_closure"],
                "auto_shutoff": coarse_run["auto_shutoff"]["final_value"],
            },
            "fine": {
                "path": str(args.fine_case.resolve()),
                "mesh": fine_mesh,
                "native_Yee_cell_count": fine_run[
                    "native_Yee_mesh_audit"
                ]["native_Yee_cell_count"],
                "P_Q_W": fine_run["P_Q_W"],
                "P_six_W": fine_run["P_six_face_W"],
                "closure": fine_run["six_face_relative_closure"],
                "auto_shutoff": fine_run["auto_shutoff"]["final_value"],
            },
            "P_Q_relative_change": p_q_relative,
            "P_six_relative_change": p_six_relative,
            "hotspot_shift_m": hotspot_shift,
            "lateral_equal_power_Q_NRMSE": lateral_equal_power_nrmse,
            "vertical_equal_power_Q_NRMSE": vertical_equal_power_nrmse,
            "largest_full_3D_error_layers": largest_z_error_layers,
            "field_plane_z_m": field_plane_z_m,
            "field_E2_raw_area_weighted_NRMSE": field_nrmse,
            "field_E2_equal_integral_NRMSE": field_equal_power_nrmse,
            "coarse_power_outside_fine_square_W": (
                coarse_power_outside_fine_square
            ),
            "coarse_power_outside_fine_square_fraction": (
                coarse_power_outside_fine_square_fraction
            ),
        },
        "component_metrics": component_metrics,
        "spatial_statistics": {
            "coarse": coarse_stats,
            "fine_remapped_to_coarse": fine_stats,
        },
        "gates": gates,
        "failed_full_fine_mesh_diagnostic": failed_diagnostic,
        "Q_processing": {
            "clipping": False,
            "smoothing": False,
            "gain": False,
            "global_rescaling": False,
            "tiling": False,
            "source_deletion": False,
            "equal_power_normalization_used_for_shape_metrics_only": True,
        },
        "not_claimed": [
            "not an experimentally reproduced beam",
            "not a paper-certified beam",
            "not a thermal, PTE, adjoint, gradient, or optimization result",
            "not a four-case mesh convergence certificate",
        ],
    }

    # Equal-power lateral maps and edge-normal profiles on the coarse grid.
    lateral_density_coarse = np.sum(
        q_coarse_total * np.diff(coarse_edges[2])[None, None, :],
        axis=2,
    )
    lateral_density_fine = np.sum(
        q_fine_target_total * np.diff(coarse_edges[2])[None, None, :],
        axis=2,
    )
    normalized_coarse = lateral_density_coarse / coarse_power
    normalized_fine = lateral_density_fine / fine_power
    extent = [
        centers[0][0] * 1e6,
        centers[0][-1] * 1e6,
        centers[1][0] * 1e6,
        centers[1][-1] * 1e6,
    ]
    vmax = max(float(np.max(normalized_coarse)), float(np.max(normalized_fine)))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    images = (
        (normalized_coarse, mesh_label(coarse_mesh)),
        (
            normalized_fine,
            f"{mesh_label(fine_mesh)} → coarse grid",
        ),
        (
            normalized_fine - normalized_coarse,
            "equal-power difference",
        ),
    )
    for index, (axis, (image, title)) in enumerate(zip(axes, images)):
        if index < 2:
            handle = axis.imshow(
                image.T,
                origin="lower",
                extent=extent,
                aspect="equal",
                vmin=0.0,
                vmax=vmax,
            )
        else:
            limit = float(np.max(np.abs(image)))
            handle = axis.imshow(
                image.T,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
        axis.set(title=title, xlabel="x (µm)", ylabel="y (µm)")
        figure.colorbar(handle, ax=axis)
    figure.tight_layout()
    map_path = args.output_dir / "W12_EDGE_A_XY_REFINEMENT_Q_MAPS.png"
    figure.savefig(map_path, dpi=200)
    plt.close(figure)

    line_n = np.linspace(-20.0e-6, 20.0e-6, 801)
    line_x = -line_n / np.sqrt(2.0)
    line_y = line_n / np.sqrt(2.0)
    query = np.column_stack((line_x, line_y))
    profile_coarse = RegularGridInterpolator(
        centers[:2], normalized_coarse, bounds_error=True
    )(query)
    profile_fine = RegularGridInterpolator(
        centers[:2], normalized_fine, bounds_error=True
    )(query)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        line_n * 1e6,
        profile_coarse,
        label=mesh_label(coarse_mesh),
    )
    axis.plot(
        line_n * 1e6,
        profile_fine,
        "--",
        label=f"{mesh_label(fine_mesh)} → coarse grid",
    )
    axis.axvline(0.0, color="k", lw=1, alpha=0.4)
    axis.set(
        xlabel="edge-normal n=(y−x)/√2 (µm)",
        ylabel="equal-power areal Q",
        title="Straight 45° edge-normal Q profile",
    )
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    profile_path = (
        args.output_dir / "W12_EDGE_A_XY_REFINEMENT_EDGE_PROFILE.png"
    )
    figure.savefig(profile_path, dpi=200)
    plt.close(figure)

    summary_path = (
        args.output_dir / "w12_edge_a_xy_refinement_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    cases_path = args.output_dir / "w12_edge_a_xy_refinement_cases.csv"
    with cases_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "outer_xy_nm",
                "fine_xy_nm",
                "fine_half_span_um",
                "P_Q_W",
                "P_six_W",
                "closure_percent",
                "auto_shutoff",
                "native_Yee_cells",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case": "coarse",
                "outer_xy_nm": coarse_mesh["outer_xy_m"] * 1e9,
                "fine_xy_nm": coarse_mesh["fine_xy_m"] * 1e9,
                "fine_half_span_um": (
                    coarse_mesh["fine_half_span_m"] * 1e6
                ),
                "P_Q_W": coarse_run["P_Q_W"],
                "P_six_W": coarse_run["P_six_face_W"],
                "closure_percent": 100
                * coarse_run["six_face_relative_closure"],
                "auto_shutoff": coarse_run["auto_shutoff"]["final_value"],
                "native_Yee_cells": coarse_run[
                    "native_Yee_mesh_audit"
                ]["native_Yee_cell_count"],
            }
        )
        writer.writerow(
            {
                "case": "nested_refined",
                "outer_xy_nm": fine_mesh["outer_xy_m"] * 1e9,
                "fine_xy_nm": fine_mesh["fine_xy_m"] * 1e9,
                "fine_half_span_um": (
                    fine_mesh["fine_half_span_m"] * 1e6
                ),
                "P_Q_W": fine_run["P_Q_W"],
                "P_six_W": fine_run["P_six_face_W"],
                "closure_percent": 100
                * fine_run["six_face_relative_closure"],
                "auto_shutoff": fine_run["auto_shutoff"]["final_value"],
                "native_Yee_cells": fine_run[
                    "native_Yee_mesh_audit"
                ]["native_Yee_cell_count"],
            }
        )

    report_path = (
        args.output_dir / "W12_EDGE_A_XY_REFINEMENT_REPORT.md"
    )
    total_metric = component_metrics["total"]
    report_path.write_text(
        f"""# W12 edge-a nested x/y refinement

Status: `{status}`

This is a **paper-like scalar-Gaussian scenario with an explicitly assumed
12 µm waist**. It is not an experimentally reproduced or paper-certified
beam.

## Mesh contract

- Coarse: {mesh_label(coarse_mesh)}.
- Refined: {mesh_label(fine_mesh)}.
- The coarse artifact places
  `{coarse_power_outside_fine_square_fraction:.4%}` of absorbed power outside
  the finest square. That outer source support remains solved on the coarser
  nested levels; it is not cropped, deleted, smoothed, gained, tiled, or
  rescaled.
- Both use TaIrTe4 `dz=5 nm`, six PML boundaries, the same scalar source,
  material, incident reference, and control-volume definitions.

## Results

| Metric | {mesh_label(coarse_mesh)} | {mesh_label(fine_mesh)} | Relative change |
|---|---:|---:|---:|
| P_Q (W) | {coarse_run['P_Q_W']:.12e} | {fine_run['P_Q_W']:.12e} | {p_q_relative:.4%} |
| P_six (W) | {coarse_run['P_six_face_W']:.12e} | {fine_run['P_six_face_W']:.12e} | {p_six_relative:.4%} |
| Six-face closure | {coarse_run['six_face_relative_closure']:.4%} | {fine_run['six_face_relative_closure']:.4%} | — |
| Auto-shutoff | {coarse_run['auto_shutoff']['final_value']:.6e} | {fine_run['auto_shutoff']['final_value']:.6e} | — |
| Native Yee cells | {coarse_run['native_Yee_mesh_audit']['native_Yee_cell_count']:,} | {fine_run['native_Yee_mesh_audit']['native_Yee_cell_count']:,} | — |

Exact cell-overlap remapping preserves fine-grid power to
`{total_metric['conservative_remap_power_error']:.3e}` relative error.

- Raw volume-weighted spatial-Q NRMSE:
  `{total_metric['raw_density_volume_weighted_NRMSE']:.4%}`
- Equal-power full 3D Q NRMSE:
  `{total_metric['equal_power_energy_NRMSE']:.4%}`
- Equal-power lateral Q NRMSE:
  `{lateral_equal_power_nrmse:.4%}`
- Equal-power vertical Q marginal NRMSE:
  `{vertical_equal_power_nrmse:.4%}`
- Equal-power Q correlation:
  `{total_metric['equal_power_energy_correlation']:.9f}`
- Hotspot displacement: `{hotspot_shift*1e9:.3f} nm`
- `z={field_plane_z_m*1e6:.6g} µm` total-field E² raw area-weighted NRMSE:
  `{field_nrmse:.4%}`
- `z={field_plane_z_m*1e6:.6g} µm` total-field E² equal-integral NRMSE:
  `{field_equal_power_nrmse:.4%}`

| Q component | Power change | Equal-power 3D NRMSE | Correlation |
|---|---:|---:|---:|
| x | {component_metrics['x']['power_relative_change']:.4%} | {component_metrics['x']['equal_power_energy_NRMSE']:.4%} | {component_metrics['x']['equal_power_energy_correlation']:.9f} |
| y | {component_metrics['y']['power_relative_change']:.4%} | {component_metrics['y']['equal_power_energy_NRMSE']:.4%} | {component_metrics['y']['equal_power_energy_correlation']:.9f} |
| z | {component_metrics['z']['power_relative_change']:.4%} | {component_metrics['z']['equal_power_energy_NRMSE']:.4%} | {component_metrics['z']['equal_power_energy_correlation']:.9f} |

The E² plane is a total-field diagnostic and is not called a pure incident
beam waist measurement.

The full-3D discrepancy is localized: the layer at
`z={largest_z_error_layers[0]['z_m']*1e9:.6g} nm` contributes
`{largest_z_error_layers[0]['fraction_of_full_3D_squared_error']:.4%}` of
the squared equal-power 3D error. This localization is diagnostic evidence;
it does not permit replacing the failed 3D gate by either marginal metric.

## Gate

The strict spatial-Q promotion gate is 0.5%. The per-gate booleans are stored
in the summary JSON. Passing total power and lateral-Q metrics does not
override the failed full-3D spatial-Q gate.

No thermal, PTE, adjoint, gradient, or optimization calculation was run.
""",
        encoding="utf-8",
    )

    raw_files: list[dict[str, Any]] = []
    for label, directory, manifest_path in (
        ("coarse", args.coarse_case, coarse_manifest),
        ("fine", args.fine_case, fine_manifest),
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, entry in manifest.get("raw_artifacts", {}).items():
            path = Path(entry["server_path"])
            if path.is_file():
                if path.stat().st_size != int(entry["size_bytes"]):
                    raise RuntimeError(
                        f"{label} raw artifact size mismatch: {path}"
                    )
                raw_files.append(
                    {
                        "case": label,
                        "name": name,
                        "path": str(path.resolve()),
                        "byte_size": path.stat().st_size,
                        "sha256": entry["sha256"],
                        "generation_command": entry.get(
                            "generation_command",
                            manifest.get("generation_command"),
                        ),
                        "generation_commit": entry.get(
                            "generation_commit",
                            manifest.get("generation_commit"),
                        ),
                    }
                )
    generated_files = [
        report_path,
        summary_path,
        cases_path,
        map_path,
        profile_path,
    ]
    manifest_out = {
        "status": status,
        "raw_artifacts": raw_files,
        "published_files": [
            {
                "path": str(path.resolve()),
                "byte_size": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in generated_files
        ],
    }
    manifest_output_path = args.output_dir / "RAW_ARTIFACT_MANIFEST.json"
    manifest_output_path.write_text(
        json.dumps(manifest_out, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0 if gates["all"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
