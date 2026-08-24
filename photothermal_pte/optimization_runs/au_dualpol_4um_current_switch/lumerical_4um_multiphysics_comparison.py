"""Conservative Lumerical native-Yee Q remap and downstream comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import sparse

from photothermal_pte.finite_inverse_design.native_yee_q import trapezoid_weights
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_mesh_contract import (
    RELATIVE_GATE,
)


COMPONENTS = ("x", "y", "z")
SYMMETRIC_CURRENT_CANCELLATION_GATE = 1.0e-6
LUMERICAL_MATERIAL_FILTER_RELATIVE_TOLERANCE = 1.0e-15


def _overlap_operator(
    coordinates: np.ndarray,
    widths: np.ndarray,
    target_edges: np.ndarray,
    clip_bounds: tuple[float, float] | None = None,
) -> sparse.csr_matrix:
    """Map source dual-cell power to target cells by overlap fraction."""

    coordinate = np.asarray(coordinates, dtype=np.float64)
    width = np.asarray(widths, dtype=np.float64)
    edges = np.asarray(target_edges, dtype=np.float64)
    if coordinate.ndim != 1 or width.shape != coordinate.shape:
        raise ValueError("source coordinate/width arrays must be matching 1-D arrays")
    if np.any(width <= 0.0) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("source widths and target-edge intervals must be positive")
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for source_index, (center, full_width) in enumerate(
        zip(coordinate, width, strict=True)
    ):
        full_low = center - 0.5 * full_width
        full_high = center + 0.5 * full_width
        low = full_low if clip_bounds is None else max(full_low, clip_bounds[0])
        high = full_high if clip_bounds is None else min(full_high, clip_bounds[1])
        denominator = high - low
        if denominator <= 0.0:
            raise RuntimeError("source cell does not overlap the clipped material")
        first = max(int(np.searchsorted(edges, low, side="right")) - 1, 0)
        last = min(
            int(np.searchsorted(edges, high, side="left")) + 1,
            edges.size - 1,
        )
        for target_index in range(first, last):
            overlap = max(
                0.0,
                min(high, edges[target_index + 1])
                - max(low, edges[target_index]),
            )
            if overlap > 0.0:
                rows.append(target_index)
                columns.append(source_index)
                values.append(overlap / denominator)
    operator = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(edges.size - 1, coordinate.size),
    ).tocsr()
    column_sum = np.asarray(operator.sum(axis=0)).reshape(-1)
    error = float(np.max(np.abs(column_sum - 1.0)))
    # Physical layer bounds are decimal nm/um values while the thermal edges
    # are accumulated floating-point sequences.  Permit their sub-attometre
    # round-trip mismatch here; the independently summed total-power remap
    # still has to pass the much stricter 1e-12 conservation gate downstream.
    if error > 1.0e-10:
        raise RuntimeError(f"target grid does not conservatively cover source: {error}")
    return operator


def _apply_axis(
    array: np.ndarray, operator: sparse.spmatrix, axis: int
) -> np.ndarray:
    moved = np.moveaxis(array, axis, 0)
    mapped = operator @ moved.reshape(moved.shape[0], -1)
    restored = np.asarray(mapped).reshape(
        (operator.shape[0],) + moved.shape[1:]
    )
    return np.moveaxis(restored, 0, axis)


def _forward(
    power: np.ndarray,
    operators: tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix],
) -> np.ndarray:
    mapped = power
    for axis, operator in enumerate(operators):
        mapped = _apply_axis(mapped, operator, axis)
    return mapped


def _covered_target_edges(
    source_coordinates: np.ndarray,
    source_widths: np.ndarray,
    full_target_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    low = float(np.min(source_coordinates - 0.5 * source_widths))
    high = float(np.max(source_coordinates + 0.5 * source_widths))
    edges = np.asarray(full_target_edges, dtype=np.float64)
    cells = np.flatnonzero((edges[1:] > low) & (edges[:-1] < high))
    if cells.size == 0 or not np.array_equal(
        cells, np.arange(cells[0], cells[-1] + 1)
    ):
        raise RuntimeError("source support does not select contiguous target cells")
    selected = np.arange(cells[0], cells[-1] + 1)
    selected_edges = edges[selected[0] : selected[-1] + 2]
    if selected_edges[0] > low or selected_edges[-1] < high:
        raise RuntimeError("selected target subgrid does not cover source dual cells")
    return selected, selected_edges


def map_lumerical_q_to_thermal(
    raw: Mapping[str, np.ndarray],
    target_edges_m: tuple[np.ndarray, np.ndarray, np.ndarray],
    source_scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map three native staggered Q components to common thermal-cell power."""

    if not np.isfinite(source_scale) or source_scale <= 0.0:
        raise ValueError("source_scale must be positive and finite")
    target_shape = tuple(len(edges) - 1 for edges in target_edges_m)
    total = np.zeros(target_shape, dtype=np.float64)
    records: dict[str, Any] = {}
    for component in COMPONENTS:
        q = np.asarray(raw[f"Q{component}_W_m3"], dtype=np.float64)
        coordinates = tuple(
            np.asarray(raw[f"Q{component}_{axis}_m"], dtype=np.float64)
            for axis in COMPONENTS
        )
        if q.shape != tuple(axis.size for axis in coordinates):
            raise RuntimeError(
                f"Q{component} shape {q.shape} does not match native coordinates"
            )
        if not np.all(np.isfinite(q)) or np.any(q < 0.0):
            raise RuntimeError(f"Q{component} must be finite and nonnegative")
        widths = tuple(trapezoid_weights(axis) for axis in coordinates)
        target_indices_and_edges = tuple(
            _covered_target_edges(coordinates[axis], widths[axis], target_edges_m[axis])
            for axis in range(3)
        )
        indices = tuple(item[0] for item in target_indices_and_edges)
        local_edges = tuple(item[1] for item in target_indices_and_edges)
        operators = tuple(
            _overlap_operator(coordinates[axis], widths[axis], local_edges[axis])
            for axis in range(3)
        )
        native_power = (
            q
            * widths[0][:, None, None]
            * widths[1][None, :, None]
            * widths[2][None, None, :]
            * source_scale
        )
        mapped = _forward(native_power, operators)
        total[np.ix_(*indices)] += mapped
        native_total = float(np.sum(native_power))
        mapped_total = float(np.sum(mapped))
        error = abs(native_total - mapped_total) / max(
            abs(native_total), np.finfo(float).tiny
        )
        records[component] = {
            "native_power_W": native_total,
            "mapped_power_W": mapped_total,
            "relative_conservation_error": error,
            "native_shape": list(q.shape),
            "target_index_bounds": [
                [int(index[0]), int(index[-1])] for index in indices
            ],
        }
    native_total = float(sum(row["native_power_W"] for row in records.values()))
    mapped_total = float(np.sum(total))
    total_error = abs(native_total - mapped_total) / max(
        abs(native_total), np.finfo(float).tiny
    )
    audit = {
        "component": records,
        "native_total_power_W": native_total,
        "mapped_total_power_W": mapped_total,
        "relative_conservation_error": total_error,
        "finite_nonnegative": bool(
            np.all(np.isfinite(total)) and np.all(total >= 0.0)
        ),
        "operations_absent": ["clipping", "smoothing", "gain", "tiling"],
    }
    return total, audit


def _overlap_widths(
    coordinates: np.ndarray,
    widths: np.ndarray,
    bounds: tuple[float, float],
) -> np.ndarray:
    low = np.maximum(coordinates - 0.5 * widths, bounds[0])
    high = np.minimum(coordinates + 0.5 * widths, bounds[1])
    return np.maximum(high - low, 0.0)


def _target_material_edges(
    full_edges: np.ndarray, bounds: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.asarray(full_edges, dtype=np.float64)
    cells = np.flatnonzero(
        (edges[:-1] >= bounds[0] - 2.0e-18)
        & (edges[1:] <= bounds[1] + 2.0e-18)
    )
    if cells.size == 0 or not np.array_equal(
        cells, np.arange(cells[0], cells[-1] + 1)
    ):
        raise RuntimeError(f"material bounds {bounds} do not select target cells")
    selected = np.arange(cells[0], cells[-1] + 1)
    return selected, edges[selected[0] : selected[-1] + 2]


def _material_domains(
    coordinates: tuple[np.ndarray, np.ndarray, np.ndarray],
    widths: tuple[np.ndarray, np.ndarray, np.ndarray],
    case: str,
    target_edges_m: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, tuple[tuple[float, float], ...]]:
    support = tuple(
        (
            float(np.min(coordinates[axis] - 0.5 * widths[axis])),
            float(np.max(coordinates[axis] + 0.5 * widths[axis])),
        )
        for axis in range(3)
    )
    covering_target_bounds = []
    for axis in range(3):
        edges = np.asarray(target_edges_m[axis], dtype=np.float64)
        cells = np.flatnonzero(
            (edges[1:] > support[axis][0]) & (edges[:-1] < support[axis][1])
        )
        if cells.size == 0:
            raise RuntimeError("thermal grid does not cover native Q support")
        covering_target_bounds.append(
            (float(edges[cells[0]]), float(edges[cells[-1] + 1]))
        )
    domains: dict[str, tuple[tuple[float, float], ...]] = {
        "TaIrTe4": ((-8.0e-6, 8.0e-6), (-8.0e-6, 8.0e-6), (-0.1e-6, 0.0)),
        "SiO2": (
            covering_target_bounds[0],
            covering_target_bounds[1],
            (-0.385e-6, -0.1e-6),
        ),
        "Si": (
            covering_target_bounds[0],
            covering_target_bounds[1],
            (covering_target_bounds[2][0], -0.385e-6),
        ),
    }
    if case == "full":
        domains["Au"] = (
            (-4.0e-6, 4.0e-6),
            (-4.0e-6, 4.0e-6),
            (0.0, 0.05e-6),
        )
    elif case != "empty":
        raise ValueError("material-aware exact-control remap requires empty or full")
    return domains


def map_lumerical_material_q_to_thermal(
    raw: Mapping[str, np.ndarray],
    target_edges_m: tuple[np.ndarray, np.ndarray, np.ndarray],
    source_scale: float,
    *,
    case: str,
    material_imaginary_epsilon: Mapping[str, Mapping[str, float]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reconstruct physical-material loss and remap only within that material.

    ``Q / Im(epsilon_effective)`` recovers the common positive field-loss
    factor at each native component sample.  Multiplying it by the fitted
    finite-dt material loss and the exact dual-cell/material overlap gives a
    material-resolved power without assigning cut-cell loss to thermal air.
    No global or local closure rescaling is applied.
    """

    if not np.isfinite(source_scale) or source_scale <= 0.0:
        raise ValueError("source_scale must be positive and finite")
    target_shape = tuple(len(edges) - 1 for edges in target_edges_m)
    total = np.zeros(target_shape, dtype=np.float64)
    material_records: dict[str, dict[str, float]] = {}
    native_total = 0.0
    reconstructed_total = 0.0
    for component in COMPONENTS:
        q = np.asarray(raw[f"Q{component}_W_m3"], dtype=np.float64)
        epsilon = np.asarray(raw[f"epsilon_{component}"])
        coordinates = tuple(
            np.asarray(raw[f"Q{component}_{axis}_m"], dtype=np.float64)
            for axis in COMPONENTS
        )
        if q.shape != epsilon.shape or q.shape != tuple(
            axis.size for axis in coordinates
        ):
            raise RuntimeError(f"Q/epsilon/coordinate shape mismatch for {component}")
        if not np.all(np.isfinite(q)) or np.any(q < 0.0):
            raise RuntimeError(f"Q{component} must be finite and nonnegative")
        widths = tuple(trapezoid_weights(axis) for axis in coordinates)
        native_component = float(
            np.einsum(
                "i,j,k,ijk->",
                widths[0],
                widths[1],
                widths[2],
                q,
                optimize=True,
            )
            * source_scale
        )
        native_total += native_component
        effective_loss = np.imag(epsilon)
        invalid = (effective_loss <= 0.0) & (q > np.max(q) * 1.0e-14)
        if np.any(invalid):
            raise RuntimeError(
                f"Q{component} is positive where effective Im(epsilon) is nonpositive"
            )
        field_loss_factor = np.zeros_like(q)
        positive = effective_loss > 0.0
        field_loss_factor[positive] = q[positive] / effective_loss[positive]
        domains = _material_domains(coordinates, widths, case, target_edges_m)
        component_reconstructed = 0.0
        for material, bounds in domains.items():
            material_loss = float(
                material_imaginary_epsilon.get(material, {}).get(component, 0.0)
            )
            if material_loss <= 0.0:
                continue
            overlap = tuple(
                _overlap_widths(coordinates[axis], widths[axis], bounds[axis])
                for axis in range(3)
            )
            # A Yee dual cell that only touches a decimal-nm material boundary
            # can acquire a ~1e-12 fractional overlap from binary roundoff.
            # Exclude only those numerical zero-contacts; retain every finite
            # overlap larger than one part per billion of its native width.
            source_indices = tuple(
                np.flatnonzero(overlap[axis] > widths[axis] * 1.0e-9)
                for axis in range(3)
            )
            if any(index.size == 0 for index in source_indices):
                continue
            source_power = (
                field_loss_factor[np.ix_(*source_indices)]
                * material_loss
                * overlap[0][source_indices[0]][:, None, None]
                * overlap[1][source_indices[1]][None, :, None]
                * overlap[2][source_indices[2]][None, None, :]
                * source_scale
            )
            target = tuple(
                _target_material_edges(target_edges_m[axis], bounds[axis])
                for axis in range(3)
            )
            target_indices = tuple(item[0] for item in target)
            target_edges = tuple(item[1] for item in target)
            operator_list = []
            for axis in range(3):
                try:
                    operator_list.append(
                        _overlap_operator(
                            coordinates[axis][source_indices[axis]],
                            widths[axis][source_indices[axis]],
                            target_edges[axis],
                            clip_bounds=bounds[axis],
                        )
                    )
                except RuntimeError as error:
                    raise RuntimeError(
                        f"{material} Q{component} axis {COMPONENTS[axis]}: {error}"
                    ) from error
            operators = tuple(operator_list)
            mapped = _forward(source_power, operators)
            total[np.ix_(*target_indices)] += mapped
            source_total = float(np.sum(source_power))
            mapped_total = float(np.sum(mapped))
            component_reconstructed += source_total
            record = material_records.setdefault(
                material,
                {"reconstructed_power_W": 0.0, "mapped_power_W": 0.0},
            )
            record["reconstructed_power_W"] += source_total
            record["mapped_power_W"] += mapped_total
        reconstructed_total += component_reconstructed
    mapped_total = float(np.sum(total))
    reconstruction_error = abs(reconstructed_total - native_total) / max(
        abs(native_total), np.finfo(float).tiny
    )
    remap_error = abs(mapped_total - reconstructed_total) / max(
        abs(reconstructed_total), np.finfo(float).tiny
    )
    return total, {
        "method": "physical_material_overlap_from_saved_effective_epsilon_v1",
        "native_total_power_W": native_total,
        "reconstructed_material_power_W": reconstructed_total,
        "mapped_total_power_W": mapped_total,
        "native_reconstruction_relative_error": reconstruction_error,
        "relative_conservation_error": remap_error,
        "material": material_records,
        "finite_nonnegative": bool(
            np.all(np.isfinite(total)) and np.all(total >= 0.0)
        ),
        "global_or_local_rescaling": False,
        "operations_absent": ["clipping", "smoothing", "gain", "tiling"],
    }


def _lumerical_almostequal(
    values: np.ndarray,
    reference: complex,
    relative_tolerance: float = LUMERICAL_MATERIAL_FILTER_RELATIVE_TOLERANCE,
) -> np.ndarray:
    """Match the real and imaginary index parts as in Ansys' example script."""

    value = np.asarray(values)
    reference_value = complex(reference)

    def part_matches(part: np.ndarray, target: float) -> np.ndarray:
        scale = np.maximum(np.abs(part), abs(target))
        # Lumerical's official script uses almostequal(..., rel_diff=1e-15).
        # Preserve an exact zero comparison when both values are zero instead
        # of adding an undocumented absolute tolerance.
        return np.abs(part - target) <= relative_tolerance * scale

    return part_matches(np.real(value), reference_value.real) & part_matches(
        np.imag(value), reference_value.imag
    )


def map_lumerical_official_pabs_to_thermal(
    raw: Mapping[str, np.ndarray],
    target_edges_m: tuple[np.ndarray, np.ndarray, np.ndarray],
    source_scale: float,
    *,
    case: str,
    material_index_x: Mapping[str, complex],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the official pabs_adv multi-material filter, then remap.

    Ansys' ``usr_absorption_advanced_material.lsf`` multiplies the common-grid
    advanced ``Pabs`` result by a material mask obtained by comparing the
    index monitor's ``index_x`` against ``getfdtdindex(material)``.  This
    function implements that rule directly on the saved Lumerical arrays.
    Conformal mixed-index samples are deliberately left unassigned and their
    absorbed power is reported; it is never redistributed or rescaled.
    """

    if not np.isfinite(source_scale) or source_scale <= 0.0:
        raise ValueError("source_scale must be positive and finite")
    q = np.asarray(raw["Pabs_W_m3"], dtype=np.float64)
    index_x = np.asarray(raw["Pabs_index_x"])
    coordinates = tuple(
        np.asarray(raw[f"Pabs_{axis}_m"], dtype=np.float64)
        for axis in COMPONENTS
    )
    expected_shape = tuple(axis.size for axis in coordinates)
    if q.shape != expected_shape or index_x.shape != expected_shape:
        raise RuntimeError(
            "Pabs/index_x/coordinate shape mismatch: "
            f"{q.shape}, {index_x.shape}, {expected_shape}"
        )
    if not np.all(np.isfinite(q)):
        raise RuntimeError("official Lumerical Pabs must be finite")
    if not np.all(np.isfinite(index_x)):
        raise RuntimeError("official Lumerical index_x must be finite")

    widths = tuple(trapezoid_weights(axis) for axis in coordinates)
    native_power = (
        q
        * widths[0][:, None, None]
        * widths[1][None, :, None]
        * widths[2][None, None, :]
        * source_scale
    )
    native_total = float(np.sum(native_power))
    negative_total = float(-np.sum(native_power[native_power < 0.0]))
    negative_relative = negative_total / max(
        abs(native_total), np.finfo(float).tiny
    )
    target_shape = tuple(len(edges) - 1 for edges in target_edges_m)
    total = np.zeros(target_shape, dtype=np.float64)
    domains = _material_domains(
        coordinates,
        widths,
        case,
        target_edges_m,
    )
    assigned = np.zeros(q.shape, dtype=bool)
    material_records: dict[str, dict[str, float | int]] = {}
    filtered_total = 0.0
    mapped_total = 0.0

    for material, bounds in domains.items():
        if material not in material_index_x:
            continue
        material_mask = _lumerical_almostequal(
            index_x,
            material_index_x[material],
        )
        if np.any(assigned & material_mask):
            raise RuntimeError("official material-index masks overlap")
        assigned |= material_mask
        source_indices = tuple(
            np.flatnonzero(np.any(material_mask, axis=tuple(
                other for other in range(3) if other != axis
            )))
            for axis in range(3)
        )
        if any(index.size == 0 for index in source_indices):
            material_records[material] = {
                "matched_sample_count": 0,
                "filtered_power_W": 0.0,
                "mapped_power_W": 0.0,
            }
            continue
        local_mask = material_mask[np.ix_(*source_indices)]
        source_power = native_power[np.ix_(*source_indices)] * local_mask
        target = tuple(
            _target_material_edges(target_edges_m[axis], bounds[axis])
            for axis in range(3)
        )
        target_indices = tuple(item[0] for item in target)
        target_edges = tuple(item[1] for item in target)
        operators = tuple(
            _overlap_operator(
                coordinates[axis][source_indices[axis]],
                widths[axis][source_indices[axis]],
                target_edges[axis],
                clip_bounds=bounds[axis],
            )
            for axis in range(3)
        )
        mapped = _forward(source_power, operators)
        total[np.ix_(*target_indices)] += mapped
        material_filtered = float(np.sum(source_power))
        material_mapped = float(np.sum(mapped))
        filtered_total += material_filtered
        mapped_total += material_mapped
        material_records[material] = {
            "matched_sample_count": int(np.count_nonzero(material_mask)),
            "filtered_power_W": material_filtered,
            "mapped_power_W": material_mapped,
        }

    omission = native_total - filtered_total
    omission_relative = abs(omission) / max(
        abs(native_total), np.finfo(float).tiny
    )
    remap_error = abs(mapped_total - filtered_total) / max(
        abs(filtered_total), np.finfo(float).tiny
    )
    return total, {
        "method": "ansys_usr_absorption_advanced_material_index_x_filter_v1",
        "material_filter_relative_tolerance": (
            LUMERICAL_MATERIAL_FILTER_RELATIVE_TOLERANCE
        ),
        "native_total_power_W": native_total,
        "negative_absorption_magnitude_W": negative_total,
        "negative_absorption_relative": negative_relative,
        "official_material_filtered_power_W": filtered_total,
        "unassigned_absorption_power_W": omission,
        "unassigned_absorption_relative": omission_relative,
        "mapped_total_power_W": mapped_total,
        "relative_conservation_error": remap_error,
        "material": material_records,
        "finite": bool(np.all(np.isfinite(total))),
        "minimum_mapped_cell_power_W": float(np.min(total)),
        "global_or_local_rescaling": False,
        "operations_absent": ["clipping", "smoothing", "gain", "tiling"],
    }


def volume_l2_nrmse(
    coarse_power_W: np.ndarray,
    fine_power_W: np.ndarray,
    cell_volume_m3: np.ndarray,
) -> float:
    """Volume-weighted NRMSE of power density, with the finer field as norm."""

    coarse = np.asarray(coarse_power_W, dtype=np.float64)
    fine = np.asarray(fine_power_W, dtype=np.float64)
    volume = np.asarray(cell_volume_m3, dtype=np.float64)
    if coarse.shape != fine.shape or coarse.shape != volume.shape:
        raise ValueError("coarse, fine, and volume arrays must have the same shape")
    coarse_density = coarse / volume
    fine_density = fine / volume
    numerator = float(np.sum((coarse_density - fine_density) ** 2 * volume))
    denominator = float(np.sum(fine_density**2 * volume))
    return float(np.sqrt(numerator / max(denominator, np.finfo(float).tiny)))


def downstream_metrics(
    *,
    coarse_power_W: np.ndarray,
    fine_power_W: np.ndarray,
    cell_volume_m3: np.ndarray,
    coarse_ta_temperature_K: np.ndarray,
    fine_ta_temperature_K: np.ndarray,
    coarse_tmax_K: float,
    fine_tmax_K: float,
    coarse_current_A: float,
    fine_current_A: float,
    coarse_current_absolute_scale_A: float,
    fine_current_absolute_scale_A: float,
    expect_zero_current: bool,
) -> tuple[dict[str, float], dict[str, bool]]:
    fine_ta = np.asarray(fine_ta_temperature_K, dtype=np.float64)
    coarse_ta = np.asarray(coarse_ta_temperature_K, dtype=np.float64)
    if coarse_ta.shape != fine_ta.shape:
        raise ValueError("TaIrTe4 temperature fields must have matching shapes")
    metrics: dict[str, float] = {
        "remapped_Q_volume_L2_NRMSE": volume_l2_nrmse(
            coarse_power_W, fine_power_W, cell_volume_m3
        ),
        "TaIrTe4_temperature_NRMSE": float(
            np.linalg.norm(coarse_ta - fine_ta)
            / max(np.linalg.norm(fine_ta), np.finfo(float).tiny)
        ),
        "Tmax_change_relative": abs(coarse_tmax_K - fine_tmax_K)
        / max(abs(fine_tmax_K), np.finfo(float).tiny),
    }
    gates = {key: value < RELATIVE_GATE for key, value in metrics.items()}
    if expect_zero_current:
        cancellation = max(
            abs(coarse_current_A)
            / max(abs(coarse_current_absolute_scale_A), np.finfo(float).tiny),
            abs(fine_current_A)
            / max(abs(fine_current_absolute_scale_A), np.finfo(float).tiny),
        )
        metrics["symmetric_current_cancellation_max_relative"] = cancellation
        metrics["symmetric_current_difference_abs_A"] = abs(
            coarse_current_A - fine_current_A
        )
        gates["symmetric_current_cancellation_lt_1ppm"] = (
            cancellation < SYMMETRIC_CURRENT_CANCELLATION_GATE
        )
    else:
        change = abs(coarse_current_A - fine_current_A) / max(
            abs(fine_current_A), np.finfo(float).tiny
        )
        metrics["signed_current_change_relative"] = change
        gates["signed_current_change_relative"] = change < RELATIVE_GATE
        gates["signed_current_sign_preserved"] = bool(
            coarse_current_A != 0.0
            and fine_current_A != 0.0
            and np.signbit(coarse_current_A) == np.signbit(fine_current_A)
        )
    return metrics, gates


def thermal_cell_volumes(
    edges: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    widths = tuple(np.diff(axis) for axis in edges)
    return (
        widths[0][:, None, None]
        * widths[1][None, :, None]
        * widths[2][None, None, :]
    )
