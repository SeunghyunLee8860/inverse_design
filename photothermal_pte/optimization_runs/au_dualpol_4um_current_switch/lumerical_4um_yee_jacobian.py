"""Au density to Lumerical component-Yee permittivity Jacobian.

The optical design variable is the canonical nonperiodic 81x81 projected
nodal occupancy.  Lumerical's conformal material map generally makes
``epsilon_x``, ``epsilon_y``, and ``epsilon_z`` different functions of that
state, even though the imported complex index is isotropic.  Consequently an
analytic derivative of the input n-k law alone is not the discrete Maxwell
material derivative.

This module measures the complete local
``importnk2 -> index_detail -> epsilon_component`` map in layout mode.  A
period-5 coloring separates the local responses, so no Maxwell solve per
pixel is required.  Centered differences are used in the interior and
feasible one-sided differences are used at exact 0/1 endpoints.  The result
is an explicit complex sparse operator with an exact real-design transpose.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
from typing import Any

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from photothermal_pte.finite_inverse_design.yee_material_jacobian import (
    SparseYeeMaterialJacobian,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_density import (
    DENSITY_IMPORT_OBJECT,
    canonical_density_nodes,
    density_nodes,
    density_state_sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_forward import (
    DENSITY_CONTROL,
)


COMPONENTS = "xyz"
COLOR_PERIOD = 5
# The R1.2 ``index_detail`` map can change its Ey interface assignment over
# perturbations in the 3e-6--1e-5 range for a nonuniform projected density.
# B200 layout-only sweeps at the beta-2 production state show that the same
# map is smooth below 1e-6 for all three Yee components. Keep construction
# and the independent check on that converged small-step branch; the 10/3
# ratio still makes the validation step independent of the build step.
BUILD_STEP = 1.0e-6
CHECK_STEP = 3.0e-7
MAX_LOCAL_ASSIGNMENT_DISTANCE_M = 225.0e-9
NONZERO_ABSOLUTE_THRESHOLD = 1.0e-10
NONZERO_RELATIVE_THRESHOLD = 1.0e-14
NONLOCAL_TAIL_RELATIVE_THRESHOLD = 1.0e-11
MAPPING_FD_RELATIVE_LIMIT = 5.0e-4
TRANSPOSE_DOT_RELATIVE_LIMIT = 1.0e-12

IndexDetail = dict[str, np.ndarray]
DensityEvaluator = Callable[[np.ndarray], IndexDetail]


def validate_completed_density_record(
    record: Mapping[str, Any],
    projected_density: np.ndarray,
    *,
    forward_fsp_sha256: str,
) -> dict[str, Any]:
    """Bind a completed import-density result to one FSP and nodal state."""

    rho = canonical_density_nodes(projected_density)
    expected_density_sha = density_state_sha256(rho)
    recorded_density_sha = (
        record.get("layout", {})
        .get("geometry", {})
        .get("density_state", {})
        .get("density_state_sha256")
    )
    artifact_matches = [
        artifact
        for artifact in record.get("raw_artifacts", [])
        if str(artifact.get("path", "")).endswith(".fsp")
        and artifact.get("sha256") == forward_fsp_sha256
    ]
    processing = record.get("Q_processing")
    processing_unmodified = bool(
        isinstance(processing, Mapping)
        and processing.get("clipping") is False
        and processing.get("smoothing") is False
        and processing.get("gain") is False
        and processing.get("field_or_Q_rescaling") is False
        and processing.get("global_rescaling", False) is False
        and processing.get("tiling", False) is False
    )
    gates = {
        "forward_status_passed": str(record.get("status", "")).startswith(
            "PASSED_PROVISIONAL_LUMERICAL_4UM_import_"
        ),
        "forward_all_gates_passed": record.get("all_gates_passed") is True,
        "forward_case_is_import_density": record.get("case") == DENSITY_CONTROL,
        "density_state_sha_matches": recorded_density_sha == expected_density_sha,
        "forward_FSP_sha_matches_raw_artifact": len(artifact_matches) == 1,
        "forward_fields_and_Q_were_not_rescaled": processing_unmodified,
        "accelerator_policy_recorded": record.get("accelerator_policy")
        in ("development", "b200"),
        "solver_version_recorded": bool(record.get("solver_version")),
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "expected_density_state_sha256": expected_density_sha,
        "recorded_density_state_sha256": recorded_density_sha,
        "forward_fsp_sha256": forward_fsp_sha256,
        "matching_forward_artifacts": artifact_matches,
    }


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def component_coordinates(
    detail: Mapping[str, np.ndarray], component: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the physical coordinate axes for one native E/epsilon grid."""

    if component not in COMPONENTS:
        raise ValueError(f"invalid Yee component {component!r}")
    return tuple(
        np.asarray(
            detail[f"{axis}_offset"] if axis == component else detail[axis],
            dtype=np.float64,
        ).reshape(-1)
        for axis in COMPONENTS
    )


def validate_index_detail(detail: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Validate one component-specific epsilon readback and hash its grid."""

    coordinate_keys = tuple(
        key for axis in COMPONENTS for key in (axis, f"{axis}_offset")
    )
    coordinates: dict[str, np.ndarray] = {}
    for key in coordinate_keys:
        if key not in detail:
            raise KeyError(f"index_detail is missing {key!r}")
        value = np.asarray(detail[key], dtype=np.float64).reshape(-1)
        if value.size < 2 or not np.all(np.isfinite(value)):
            raise ValueError(f"index_detail {key} is not a finite coordinate")
        if not np.all(np.diff(value) > 0.0):
            raise ValueError(f"index_detail {key} is not strictly increasing")
        coordinates[key] = value
    base_shape = tuple(coordinates[axis].size for axis in COMPONENTS)
    component_shapes: dict[str, tuple[int, int, int]] = {}
    for component in COMPONENTS:
        key = f"epsilon_{component}"
        if key not in detail:
            raise KeyError(f"index_detail is missing {key!r}")
        epsilon = np.asarray(detail[key], dtype=np.complex128)
        if epsilon.shape != base_shape:
            raise ValueError(f"{key} shape {epsilon.shape} != {base_shape}")
        if not np.all(np.isfinite(epsilon)):
            raise ValueError(f"{key} contains NaN or Inf")
        component_shapes[component] = epsilon.shape
    frequency = np.asarray(detail.get("frequency_hz", []), float).reshape(-1)
    if frequency.size != 1 or not np.isfinite(frequency[0]) or frequency[0] <= 0.0:
        raise ValueError("index_detail requires one finite positive frequency")
    hash_payload = np.concatenate(
        [coordinates[key] for key in coordinate_keys] + [frequency]
    )
    return {
        "base_shape": list(base_shape),
        "component_shapes": {
            key: list(value) for key, value in component_shapes.items()
        },
        "frequency_hz": float(frequency[0]),
        "coordinate_sha256": _array_sha256(hash_payload),
        "epsilon_sha256": {
            component: _array_sha256(
                np.asarray(detail[f"epsilon_{component}"], np.complex128)
            )
            for component in COMPONENTS
        },
    }


def _require_same_grid(
    audit: Mapping[str, Any], baseline: Mapping[str, Any], *, label: str
) -> None:
    for key in ("base_shape", "component_shapes", "coordinate_sha256"):
        if audit[key] != baseline[key]:
            raise RuntimeError(f"{label} changed the frozen Yee grid: {key}")
    frequency_scale = max(abs(float(baseline["frequency_hz"])), 1.0)
    if (
        abs(float(audit["frequency_hz"]) - float(baseline["frequency_hz"]))
        > 1.0e-12 * frequency_scale
    ):
        raise RuntimeError(f"{label} changed the objective frequency")


def set_lumerical_projected_density(fdtd: Any, projected_density: np.ndarray) -> None:
    """Replace the one authorized imported Au-density object in layout mode."""

    rho = canonical_density_nodes(projected_density)
    if int(fdtd.getnamednumber(DENSITY_IMPORT_OBJECT)) != 1:
        raise RuntimeError(
            f"expected exactly one Lumerical object {DENSITY_IMPORT_OBJECT!r}"
        )
    from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.au_density_relaxation import (
        lumerical_import_index,
    )

    x, y, z = density_nodes()
    fdtd.select(DENSITY_IMPORT_OBJECT)
    result = fdtd.importnk2(lumerical_import_index(rho, z_samples=z.size), x, y, z)
    if result is not None and int(result) != 1:
        raise RuntimeError("Lumerical importnk2 returned failure")


def read_lumerical_index_detail(
    fdtd: Any,
    *,
    monitor_name: str,
    wavelength_m: float = CONTRACT.wavelength_m,
) -> IndexDetail:
    """Read one frequency of Lumerical's realized component-Yee epsilon."""

    dataset = fdtd.getresult(monitor_name, "index_detail")
    frequency = np.asarray(dataset["f"], dtype=np.float64).reshape(-1)
    if frequency.size == 0 or not np.all(np.isfinite(frequency)):
        raise RuntimeError("index_detail has no finite frequency")
    target = 299_792_458.0 / float(wavelength_m)
    frequency_index = int(np.argmin(np.abs(frequency - target)))
    if abs(frequency[frequency_index] - target) / target > 1.0e-9:
        raise RuntimeError("index_detail does not contain the objective frequency")
    detail: IndexDetail = {
        key: np.asarray(dataset[key], dtype=np.float64).reshape(-1)
        for axis in COMPONENTS
        for key in (axis, f"{axis}_offset")
    }
    shape = tuple(detail[axis].size for axis in COMPONENTS)
    for component in COMPONENTS:
        raw = np.asarray(dataset[f"index_{component}"])
        if raw.shape == shape and frequency.size == 1:
            index = raw
        elif raw.shape == (*shape, frequency.size):
            index = raw[..., frequency_index]
        else:
            raise RuntimeError(
                f"index_{component} shape {raw.shape} is incompatible with "
                f"{shape} and {frequency.size} frequencies"
            )
        detail[f"epsilon_{component}"] = np.asarray(index, np.complex128) ** 2
    detail["frequency_hz"] = np.asarray([frequency[frequency_index]], float)
    validate_index_detail(detail)
    return detail


def _append_response(
    *,
    component: str,
    derivative: np.ndarray,
    active_nodes: np.ndarray,
    nodes_xy: tuple[np.ndarray, np.ndarray],
    baseline: Mapping[str, np.ndarray],
    shape: tuple[int, int, int],
    row_parts: list[np.ndarray],
    column_parts: list[np.ndarray],
    value_parts: list[np.ndarray],
    maximum_local_distance_m: float,
    maximum_assignment_distance: dict[str, float],
    suppressed_tail: dict[str, dict[str, float | int]],
) -> None:
    flat = np.asarray(derivative, np.complex128).reshape(-1)
    response_scale = float(np.max(np.abs(flat)))
    if response_scale == 0.0:
        return
    threshold = max(
        NONZERO_ABSOLUTE_THRESHOLD,
        NONZERO_RELATIVE_THRESHOLD * response_scale,
    )
    rows = np.flatnonzero(np.abs(flat) > threshold)
    if rows.size == 0:
        return
    node_indices = np.argwhere(active_nodes)
    if node_indices.size == 0:
        raise RuntimeError("nonzero material response has no active density node")
    tree = cKDTree(
        np.column_stack(
            (nodes_xy[0][node_indices[:, 0]], nodes_xy[1][node_indices[:, 1]])
        )
    )
    ix, iy, _ = np.unravel_index(rows, shape)
    coordinates = component_coordinates(baseline, component)
    distance, nearest = tree.query(
        np.column_stack((coordinates[0][ix], coordinates[1][iy])), k=1
    )
    distance = np.asarray(distance, float)
    nearest = np.asarray(nearest, int)
    nonlocal_rows = distance > maximum_local_distance_m
    if np.any(nonlocal_rows):
        relative = np.abs(flat[rows]) / response_scale
        numerical_tail = nonlocal_rows & (
            relative <= NONLOCAL_TAIL_RELATIVE_THRESHOLD
        )
        significant = nonlocal_rows & ~numerical_tail
        if np.any(significant):
            worst = int(np.argmax(np.where(significant, relative, -1.0)))
            raise RuntimeError(
                f"{component} colored material response is nonlocal: "
                f"distance={distance[worst]:.9e} m, "
                f"relative_response={relative[worst]:.9e}, "
                f"limit={maximum_local_distance_m:.9e} m"
            )
        record = suppressed_tail[component]
        record["count"] = int(record["count"]) + int(np.count_nonzero(numerical_tail))
        record["maximum_abs"] = max(
            float(record["maximum_abs"]),
            float(np.max(np.abs(flat[rows[numerical_tail]]))),
        )
        record["maximum_relative"] = max(
            float(record["maximum_relative"]),
            float(np.max(relative[numerical_tail])),
        )
        keep = ~numerical_tail
        rows = rows[keep]
        distance = distance[keep]
        nearest = nearest[keep]
        if rows.size == 0:
            return
    maximum_assignment_distance[component] = max(
        maximum_assignment_distance[component], float(np.max(distance))
    )
    selected = node_indices[nearest]
    row_parts.append(rows)
    column_parts.append(selected[:, 0] * nodes_xy[1].size + selected[:, 1])
    value_parts.append(flat[rows])


def build_colored_material_jacobian(
    evaluate: DensityEvaluator,
    projected_density: np.ndarray,
    *,
    step: float = BUILD_STEP,
    color_period: int = COLOR_PERIOD,
    maximum_local_distance_m: float = MAX_LOCAL_ASSIGNMENT_DISTANCE_M,
) -> tuple[SparseYeeMaterialJacobian, dict[str, Any], IndexDetail]:
    """Measure a sparse local component-Yee epsilon Jacobian in layout mode."""

    rho = canonical_density_nodes(projected_density)
    if not np.isfinite(step) or not 0.0 < step < 0.5:
        raise ValueError("material-Jacobian step must lie in (0,0.5)")
    if int(color_period) != color_period or color_period < 3:
        raise ValueError("color period must be an integer >= 3")
    if maximum_local_distance_m <= 0.0:
        raise ValueError("maximum local assignment distance must be positive")
    x_nodes, y_nodes, _ = density_nodes()
    baseline = evaluate(rho)
    baseline_audit = validate_index_detail(baseline)
    shapes = {
        component: tuple(np.asarray(baseline[f"epsilon_{component}"]).shape)
        for component in COMPONENTS
    }
    rows: dict[str, list[np.ndarray]] = {component: [] for component in COMPONENTS}
    columns: dict[str, list[np.ndarray]] = {
        component: [] for component in COMPONENTS
    }
    values: dict[str, list[np.ndarray]] = {component: [] for component in COMPONENTS}
    maximum_assignment = {component: 0.0 for component in COMPONENTS}
    suppressed_tail: dict[str, dict[str, float | int]] = {
        component: {"count": 0, "maximum_abs": 0.0, "maximum_relative": 0.0}
        for component in COMPONENTS
    }
    evaluation_count = 1
    centered_node_count = 0
    lower_node_count = 0
    upper_node_count = 0
    try:
        for color_x in range(color_period):
            for color_y in range(color_period):
                color = np.zeros(rho.shape, dtype=bool)
                color[color_x::color_period, color_y::color_period] = True
                centered = color & (rho >= step) & (rho <= 1.0 - step)
                lower = color & (rho < step)
                upper = color & (rho > 1.0 - step)
                centered_node_count += int(np.count_nonzero(centered))
                lower_node_count += int(np.count_nonzero(lower))
                upper_node_count += int(np.count_nonzero(upper))
                responses: list[tuple[np.ndarray, IndexDetail, IndexDetail | None, float]] = []
                if np.any(centered):
                    plus = evaluate(rho + step * centered)
                    minus = evaluate(rho - step * centered)
                    _require_same_grid(
                        validate_index_detail(plus), baseline_audit, label="plus"
                    )
                    _require_same_grid(
                        validate_index_detail(minus), baseline_audit, label="minus"
                    )
                    evaluation_count += 2
                    responses.append((centered, plus, minus, 2.0 * step))
                if np.any(lower):
                    plus = evaluate(rho + step * lower)
                    _require_same_grid(
                        validate_index_detail(plus),
                        baseline_audit,
                        label="lower-endpoint plus",
                    )
                    evaluation_count += 1
                    responses.append((lower, plus, baseline, step))
                if np.any(upper):
                    minus = evaluate(rho - step * upper)
                    _require_same_grid(
                        validate_index_detail(minus),
                        baseline_audit,
                        label="upper-endpoint minus",
                    )
                    evaluation_count += 1
                    responses.append((upper, baseline, minus, step))
                for active_nodes, positive, negative, denominator in responses:
                    assert negative is not None
                    for component in COMPONENTS:
                        derivative = (
                            np.asarray(positive[f"epsilon_{component}"])
                            - np.asarray(negative[f"epsilon_{component}"])
                        ) / denominator
                        _append_response(
                            component=component,
                            derivative=derivative,
                            active_nodes=active_nodes,
                            nodes_xy=(x_nodes, y_nodes),
                            baseline=baseline,
                            shape=shapes[component],
                            row_parts=rows[component],
                            column_parts=columns[component],
                            value_parts=values[component],
                            maximum_local_distance_m=maximum_local_distance_m,
                            maximum_assignment_distance=maximum_assignment,
                            suppressed_tail=suppressed_tail,
                        )
    finally:
        roundtrip = evaluate(rho)
        evaluation_count += 1
    roundtrip_audit = validate_index_detail(roundtrip)
    _require_same_grid(roundtrip_audit, baseline_audit, label="baseline roundtrip")
    roundtrip_error = max(
        float(
            np.max(
                np.abs(
                    np.asarray(roundtrip[f"epsilon_{component}"])
                    - np.asarray(baseline[f"epsilon_{component}"])
                )
            )
        )
        for component in COMPONENTS
    )
    matrices: dict[str, sparse.csr_matrix] = {}
    matrix_audit: dict[str, Any] = {}
    for component in COMPONENTS:
        if not values[component]:
            raise RuntimeError(f"empty component-Yee Jacobian for {component}")
        matrix = sparse.csr_matrix(
            (
                np.concatenate(values[component]),
                (
                    np.concatenate(rows[component]),
                    np.concatenate(columns[component]),
                ),
            ),
            shape=(int(np.prod(shapes[component])), rho.size),
        )
        matrix.sum_duplicates()
        matrices[component] = matrix
        row_nnz = np.diff(matrix.indptr)
        matrix_audit[component] = {
            "shape": list(matrix.shape),
            "nnz": int(matrix.nnz),
            "active_row_count": int(np.count_nonzero(row_nnz)),
            "maximum_nonzeros_per_Yee_sample": int(np.max(row_nnz)),
        }
    operator = SparseYeeMaterialJacobian(
        density_shape=rho.shape,
        component_shapes=shapes,
        matrices=matrices,
    )
    metadata = {
        "method": (
            "nonperiodic period-5 colored layout-only finite difference of "
            "Lumerical importnk2 to component index_detail epsilon"
        ),
        "density_shape": list(rho.shape),
        "step": float(step),
        "color_period": int(color_period),
        "color_count": int(color_period**2),
        "centered_node_count": centered_node_count,
        "lower_endpoint_node_count": lower_node_count,
        "upper_endpoint_node_count": upper_node_count,
        "endpoint_policy": "feasible one-sided derivative at exact 0/1",
        "layout_index_detail_evaluations": evaluation_count,
        "Maxwell_solves": 0,
        "per_pixel_Maxwell_solves": False,
        "maximum_local_assignment_distance_limit_m": float(
            maximum_local_distance_m
        ),
        "maximum_assignment_distance_m": maximum_assignment,
        "suppressed_nonlocal_numerical_tail": suppressed_tail,
        "nonlocal_tail_relative_threshold": NONLOCAL_TAIL_RELATIVE_THRESHOLD,
        "baseline": baseline_audit,
        "roundtrip": roundtrip_audit,
        "baseline_roundtrip_epsilon_max_abs_error": roundtrip_error,
        "matrices": matrix_audit,
        "empirical_gradient_scaling": False,
    }
    return operator, metadata, baseline


def transpose_dot_error(
    operator: SparseYeeMaterialJacobian,
    direction: np.ndarray,
    cotangent: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """Audit JVP/VJP under the real-design bilinear pairing."""

    tangent = operator.jvp(direction)
    left = float(
        np.real(
            sum(
                np.sum(np.asarray(cotangent[component]) * tangent[component])
                for component in COMPONENTS
            )
        )
    )
    right = float(np.vdot(direction, operator.vjp(cotangent)))
    relative = abs(left - right) / max(
        abs(left), abs(right), np.finfo(float).tiny
    )
    return {"left": left, "right": right, "relative_error": float(relative)}


def _normalized_directions(rho: np.ndarray) -> dict[str, tuple[np.ndarray, str]]:
    x, y, _ = density_nodes()
    xn = x[:, None] / (0.5 * CONTRACT.design_span_x_m)
    yn = y[None, :] / (0.5 * CONTRACT.design_span_y_m)
    rng = np.random.default_rng(4_002_608_24)
    raw = {
        "interior_uniform": np.ones(rho.shape),
        "interior_smooth_asymmetric": (
            np.cos(0.43 * np.pi * (xn - 0.17))
            * np.sin(0.68 * np.pi * (yn + 0.09))
            + 0.13 * xn
            - 0.07 * yn
        ),
        "interior_central_localized": np.exp(
            -(xn**2 + yn**2) / (2.0 * 0.14**2)
        ),
        "interior_fixed_seed_random": rng.normal(size=rho.shape),
    }
    margin = (rho >= CHECK_STEP) & (rho <= 1.0 - CHECK_STEP)
    directions: dict[str, tuple[np.ndarray, str]] = {}
    for name, value in raw.items():
        direction = np.asarray(value, float) * margin
        norm = float(np.max(np.abs(direction)))
        if norm > 0.0:
            directions[name] = (direction / norm, "centered")
    lower = rho < BUILD_STEP
    if np.any(lower):
        directions["lower_endpoint_feasible"] = (lower.astype(float), "forward")
    upper = rho > 1.0 - BUILD_STEP
    if np.any(upper):
        directions["upper_endpoint_feasible"] = (-upper.astype(float), "forward")
    if not directions:
        raise RuntimeError("density state has no admissible Jacobian check direction")
    return directions


def validate_material_jacobian(
    evaluate: DensityEvaluator,
    projected_density: np.ndarray,
    operator: SparseYeeMaterialJacobian,
    *,
    step: float = CHECK_STEP,
) -> dict[str, Any]:
    """Run independent mapping FD and exact transpose checks.

    Interior directions use centered differences. Exact or near-saturated
    endpoint subsets use the only feasible one-sided direction. No Maxwell
    solve occurs; ``evaluate`` is expected to update and read layout only.
    """

    rho = canonical_density_nodes(projected_density)
    if not np.isfinite(step) or not 0.0 < step < BUILD_STEP:
        raise ValueError("check step must be finite and smaller than build step")
    directions = _normalized_directions(rho)
    baseline = evaluate(rho)
    baseline_audit = validate_index_detail(baseline)
    rng = np.random.default_rng(17_042_608)
    cotangent = {
        component: (
            rng.normal(size=operator.component_shapes[component])
            + 1j * rng.normal(size=operator.component_shapes[component])
        )
        for component in COMPONENTS
    }
    records: dict[str, Any] = {}
    evaluation_count = 1
    try:
        for name, (direction, scheme) in directions.items():
            tangent = operator.jvp(direction)
            if scheme == "centered":
                positive = evaluate(rho + step * direction)
                negative = evaluate(rho - step * direction)
                denominator = 2.0 * step
                evaluation_count += 2
            else:
                positive = evaluate(rho + step * direction)
                negative = baseline
                denominator = step
                evaluation_count += 1
            _require_same_grid(
                validate_index_detail(positive),
                baseline_audit,
                label=f"{name} positive",
            )
            _require_same_grid(
                validate_index_detail(negative),
                baseline_audit,
                label=f"{name} negative",
            )
            finite_difference = {
                component: (
                    np.asarray(positive[f"epsilon_{component}"])
                    - np.asarray(negative[f"epsilon_{component}"])
                )
                / denominator
                for component in COMPONENTS
            }
            difference_norm = float(
                np.sqrt(
                    sum(
                        np.linalg.norm(
                            tangent[component] - finite_difference[component]
                        )
                        ** 2
                        for component in COMPONENTS
                    )
                )
            )
            reference_norm = float(
                max(
                    np.sqrt(
                        sum(
                            np.linalg.norm(tangent[component]) ** 2
                            for component in COMPONENTS
                        )
                    ),
                    np.sqrt(
                        sum(
                            np.linalg.norm(finite_difference[component]) ** 2
                            for component in COMPONENTS
                        )
                    ),
                    np.finfo(float).tiny,
                )
            )
            component_errors: dict[str, Any] = {}
            for component in COMPONENTS:
                difference = tangent[component] - finite_difference[component]
                absolute_l2 = float(np.linalg.norm(difference))
                component_reference = max(
                    float(np.linalg.norm(tangent[component])),
                    float(np.linalg.norm(finite_difference[component])),
                    np.finfo(float).tiny,
                )
                component_errors[component] = {
                    "absolute_l2_error": absolute_l2,
                    "maximum_abs_error": float(np.max(np.abs(difference))),
                    "relative_l2_error": absolute_l2 / component_reference,
                }
            transpose = transpose_dot_error(operator, direction, cotangent)
            records[name] = {
                "scheme": scheme,
                "direction_sha256": _array_sha256(direction),
                "mapping_FD_step": float(step),
                "mapping_FD_relative_error": difference_norm / reference_norm,
                "component_mapping_FD_errors": component_errors,
                "transpose_dot": transpose,
            }
    finally:
        roundtrip = evaluate(rho)
        evaluation_count += 1
    roundtrip_error = max(
        float(
            np.max(
                np.abs(
                    np.asarray(roundtrip[f"epsilon_{component}"])
                    - np.asarray(baseline[f"epsilon_{component}"])
                )
            )
        )
        for component in COMPONENTS
    )
    _require_same_grid(
        validate_index_detail(roundtrip), baseline_audit, label="validation roundtrip"
    )
    worst_fd = max(
        float(record["mapping_FD_relative_error"]) for record in records.values()
    )
    worst_dot = max(
        float(record["transpose_dot"]["relative_error"])
        for record in records.values()
    )
    gates = {
        "mapping_FD_relative_error_lt_5e_4": worst_fd
        < MAPPING_FD_RELATIVE_LIMIT,
        "transpose_dot_relative_error_lt_1e_12": worst_dot
        < TRANSPOSE_DOT_RELATIVE_LIMIT,
        "baseline_layout_roundtrip_exact": roundtrip_error == 0.0,
    }
    return {
        "mode": "full_independent_mapping_FD_and_transpose",
        "independent_mapping_FD_performed": True,
        "passed": all(gates.values()),
        "directions": records,
        "worst_mapping_FD_relative_error": worst_fd,
        "mapping_FD_relative_limit": MAPPING_FD_RELATIVE_LIMIT,
        "worst_transpose_dot_relative_error": worst_dot,
        "transpose_dot_relative_limit": TRANSPOSE_DOT_RELATIVE_LIMIT,
        "baseline_roundtrip_epsilon_max_abs_error": roundtrip_error,
        "layout_index_detail_evaluations": evaluation_count,
        "Maxwell_solves": 0,
        "gates": gates,
    }


def validate_material_jacobian_transpose_only(
    projected_density: np.ndarray,
    operator: SparseYeeMaterialJacobian,
) -> dict[str, Any]:
    """Audit the exact discrete JVP/VJP pairing without another layout FD.

    The colored finite differences used to *construct* ``operator`` remain
    mandatory at every density state. This function omits only the extra,
    independently directed mapping finite differences used as a periodic
    self-audit. It is suitable only when the production orchestrator supplies
    a hash-verified full-FD certificate for the current beta stage.
    """

    rho = canonical_density_nodes(projected_density)
    directions = _normalized_directions(rho)
    rng = np.random.default_rng(17_042_608)
    cotangent = {
        component: (
            rng.normal(size=operator.component_shapes[component])
            + 1j * rng.normal(size=operator.component_shapes[component])
        )
        for component in COMPONENTS
    }
    records = {
        name: {
            "scheme": scheme,
            "direction_sha256": _array_sha256(direction),
            "transpose_dot": transpose_dot_error(
                operator, direction, cotangent
            ),
        }
        for name, (direction, scheme) in directions.items()
    }
    worst_dot = max(
        float(record["transpose_dot"]["relative_error"])
        for record in records.values()
    )
    gates = {
        "transpose_dot_relative_error_lt_1e_12": worst_dot
        < TRANSPOSE_DOT_RELATIVE_LIMIT,
    }
    return {
        "mode": "transpose_only_with_stage_FD_certificate",
        "independent_mapping_FD_performed": False,
        "passed": all(gates.values()),
        "directions": records,
        "worst_transpose_dot_relative_error": worst_dot,
        "transpose_dot_relative_limit": TRANSPOSE_DOT_RELATIVE_LIMIT,
        "layout_index_detail_evaluations": 0,
        "Maxwell_solves": 0,
        "gates": gates,
    }
