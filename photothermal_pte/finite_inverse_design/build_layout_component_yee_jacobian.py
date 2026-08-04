#!/usr/bin/env python3
"""Build exact component Yee Jacobians from v261 layout index_detail."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from .contract import DESIGN_BOUNDS_M, design_nodal_coordinates_m
from .probe_v261_cpu_tfsf_device import FREQUENCY_HZ, PABS_FIELD, PABS_INDEX
from .probe_v261_gpu_plane_wave_roi import load_lumapi
from .run_combined_physical_rho_pte_adfd import (
    physical_state,
    set_imported_density,
)
from .run_v261_large_background_mixed_optical_adfd import (
    monitor_electric,
)
from .run_v261_large_background_tfsf_forward import sha256
from .yee_material_jacobian import SparseYeeMaterialJacobian


STATUS_PASS = "VALIDATED_COMPONENT_WISE_YEE_MATERIAL_JACOBIAN"
STATUS_FAIL = "FAILED_COMPONENT_WISE_YEE_MATERIAL_JACOBIAN"
COLOR_PERIOD = 5
BUILD_STEP = 1.0e-4
CHECK_STEP = 1.0e-5
NONZERO_THRESHOLD = 1.0e-8
MAX_LOCAL_DISTANCE_M = 80.0e-9
FD_LIMIT = 1.0e-8
DOT_LIMIT = 1.0e-12
COORDINATE_LIMIT_M = 2.0e-18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-project", required=True)
    parser.add_argument("--forward-sha256", required=True)
    parser.add_argument("--adjoint-project", required=True)
    parser.add_argument("--adjoint-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def maximum_difference(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, float).reshape(-1)
    b = np.asarray(right, float).reshape(-1)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b)))


def index_detail(fdtd: object) -> dict[str, np.ndarray]:
    dataset = fdtd.getresult(PABS_INDEX, "index_detail")
    frequency = np.asarray(dataset["f"], float).reshape(-1)
    frequency_index = int(np.argmin(np.abs(frequency - FREQUENCY_HZ)))
    result = {
        key: np.asarray(dataset[key], float).reshape(-1)
        for key in ("x", "x_offset", "y", "y_offset", "z", "z_offset")
    }
    shape = (
        result["x"].size,
        result["y"].size,
        result["z"].size,
    )
    for component in "xyz":
        raw = np.asarray(dataset[f"index_{component}"])
        expected = (*shape, frequency.size)
        if raw.shape != expected:
            raise RuntimeError(
                f"index_{component} shape {raw.shape} != {expected}"
            )
        result[f"epsilon_{component}"] = raw[
            ..., frequency_index
        ] ** 2
    result["frequency_hz"] = np.asarray(
        [frequency[frequency_index]], float
    )
    return result


def component_coordinates(
    detail: dict[str, np.ndarray], component: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        detail["x_offset"] if component == "x" else detail["x"],
        detail["y_offset"] if component == "y" else detail["y"],
        detail["z_offset"] if component == "z" else detail["z"],
    )


def directions() -> dict[str, np.ndarray]:
    x, y = design_nodal_coordinates_m()
    xn = x[:, None] / 1.0e-6
    yn = y[None, :] / 1.0e-6
    rng = np.random.default_rng(811325)
    raw = {
        "uniform": np.ones((81, 81)),
        "smooth_asymmetric": (
            np.cos(0.43 * np.pi * (xn - 0.17))
            * np.sin(0.68 * np.pi * (yn + 0.09))
            + 0.13 * xn
            - 0.07 * yn
        ),
        "central_localized": np.exp(
            -(xn**2 + yn**2) / (2.0 * 0.14**2)
        ),
        "design_edge_localized": np.exp(
            -((xn - 0.88) ** 2 + (yn + 0.23) ** 2)
            / (2.0 * 0.075**2)
        ),
        "fixed_seed_random": rng.normal(size=(81, 81)),
    }
    return {
        name: value / np.max(np.abs(value))
        for name, value in raw.items()
    }


def nearest_colored_node(
    coordinate: np.ndarray,
    nodes: np.ndarray,
    color: int,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.arange(color, nodes.size, COLOR_PERIOD)
    distance = np.abs(
        np.asarray(coordinate, float)[:, None] - nodes[candidates][None, :]
    )
    local = np.argmin(distance, axis=1)
    return candidates[local], distance[np.arange(coordinate.size), local]


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    forward_path = Path(args.forward_project).expanduser().resolve()
    adjoint_path = Path(args.adjoint_project).expanduser().resolve()
    for role, path, expected in (
        ("forward", forward_path, args.forward_sha256),
        ("adjoint", adjoint_path, args.adjoint_sha256),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != expected:
            raise RuntimeError(f"{role} FSP SHA-256 mismatch")

    fdtd = None
    try:
        lumapi = load_lumapi()
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        fdtd.load(str(forward_path))
        _, forward_field_grid = monitor_electric(fdtd, PABS_FIELD)
        completed_detail = index_detail(fdtd)
        fdtd.switchtolayout()
        layout_detail = index_detail(fdtd)
        completed_layout_epsilon_error = max(
            float(
                np.max(
                    np.abs(
                        completed_detail[f"epsilon_{component}"]
                        - layout_detail[f"epsilon_{component}"]
                    )
                )
            )
            for component in "xyz"
        )
        rho, _ = physical_state()
        x_nodes, y_nodes = design_nodal_coordinates_m()
        shapes = {
            component: layout_detail[f"epsilon_{component}"].shape
            for component in "xyz"
        }
        row_parts = {component: [] for component in "xyz"}
        column_parts = {component: [] for component in "xyz"}
        value_parts = {component: [] for component in "xyz"}
        maximum_assignment_distance = {component: 0.0 for component in "xyz"}

        for color_x in range(COLOR_PERIOD):
            for color_y in range(COLOR_PERIOD):
                mask = np.zeros((81, 81), float)
                mask[color_x::COLOR_PERIOD, color_y::COLOR_PERIOD] = 1.0
                pair = {}
                for sign, label in ((1.0, "plus"), (-1.0, "minus")):
                    set_imported_density(fdtd, rho + sign * BUILD_STEP * mask)
                    pair[label] = index_detail(fdtd)
                for component in "xyz":
                    derivative = (
                        pair["plus"][f"epsilon_{component}"]
                        - pair["minus"][f"epsilon_{component}"]
                    ) / (2.0 * BUILD_STEP)
                    flat = derivative.reshape(-1)
                    rows = np.flatnonzero(np.abs(flat) > NONZERO_THRESHOLD)
                    if rows.size == 0:
                        continue
                    ix, iy, _ = np.unravel_index(rows, shapes[component])
                    coordinates = component_coordinates(
                        layout_detail, component
                    )
                    node_x, distance_x = nearest_colored_node(
                        coordinates[0][ix], x_nodes, color_x
                    )
                    node_y, distance_y = nearest_colored_node(
                        coordinates[1][iy], y_nodes, color_y
                    )
                    distance = np.sqrt(distance_x**2 + distance_y**2)
                    maximum_assignment_distance[component] = max(
                        maximum_assignment_distance[component],
                        float(np.max(distance)),
                    )
                    if np.any(distance > MAX_LOCAL_DISTANCE_M):
                        raise RuntimeError(
                            f"{component} layout response is not local "
                            f"under period-{COLOR_PERIOD} coloring"
                        )
                    row_parts[component].append(rows)
                    column_parts[component].append(
                        node_x * y_nodes.size + node_y
                    )
                    value_parts[component].append(flat[rows])
        set_imported_density(fdtd, rho)
        baseline_detail = index_detail(fdtd)
        baseline_roundtrip_error = max(
            float(
                np.max(
                    np.abs(
                        baseline_detail[f"epsilon_{component}"]
                        - layout_detail[f"epsilon_{component}"]
                    )
                )
            )
            for component in "xyz"
        )
        matrices = {}
        for component in "xyz":
            rows = np.concatenate(row_parts[component])
            columns = np.concatenate(column_parts[component])
            values = np.concatenate(value_parts[component])
            matrices[component] = sparse.csr_matrix(
                (values, (rows, columns)),
                shape=(int(np.prod(shapes[component])), 81 * 81),
            )
            matrices[component].sum_duplicates()
        operator = SparseYeeMaterialJacobian(
            density_shape=(81, 81),
            component_shapes=shapes,
            matrices=matrices,
        )

        direction_records = {}
        rho_directions = directions()
        rng = np.random.default_rng(71381)
        cotangent = {
            component: (
                rng.normal(size=shapes[component])
                + 1j * rng.normal(size=shapes[component])
            )
            for component in "xyz"
        }
        for name, direction in rho_directions.items():
            pair = {}
            for sign, label in ((1.0, "plus"), (-1.0, "minus")):
                set_imported_density(
                    fdtd, rho + sign * CHECK_STEP * direction
                )
                pair[label] = index_detail(fdtd)
            finite_difference = {
                component: (
                    pair["plus"][f"epsilon_{component}"]
                    - pair["minus"][f"epsilon_{component}"]
                )
                / (2.0 * CHECK_STEP)
                for component in "xyz"
            }
            tangent = operator.jvp(direction)
            delta_norm = np.sqrt(
                sum(
                    np.linalg.norm(
                        tangent[component] - finite_difference[component]
                    )
                    ** 2
                    for component in "xyz"
                )
            )
            reference_norm = max(
                np.sqrt(
                    sum(
                        np.linalg.norm(tangent[component]) ** 2
                        for component in "xyz"
                    )
                ),
                np.sqrt(
                    sum(
                        np.linalg.norm(finite_difference[component]) ** 2
                        for component in "xyz"
                    )
                ),
                np.finfo(float).tiny,
            )
            left = float(
                np.real(
                    sum(
                        np.sum(
                            cotangent[component] * tangent[component]
                        )
                        for component in "xyz"
                    )
                )
            )
            right = float(np.vdot(direction, operator.vjp(cotangent)))
            direction_records[name] = {
                "direction_sha256": array_sha256(direction),
                "mapping_only_centered_FD_step": CHECK_STEP,
                "mapping_only_FD_relative_error": float(
                    delta_norm / reference_norm
                ),
                "JVP_VJP_dot_relative_error": float(
                    abs(left - right)
                    / max(abs(left), abs(right), np.finfo(float).tiny)
                ),
                "clipping": False,
                "Maxwell_solves": 0,
            }
        set_imported_density(fdtd, rho)

        fdtd.load(str(adjoint_path))
        _, adjoint_field_grid = monitor_electric(fdtd, PABS_FIELD)
        coordinate_audit = {"components": {}}
        maximum_coordinate_mismatch = 0.0
        for component in "xyz":
            detail_coordinates = component_coordinates(
                completed_detail, component
            )
            component_index = "xyz".index(component)
            field_coordinates = [
                np.asarray(forward_field_grid[axis], float).copy()
                for axis in "xyz"
            ]
            field_coordinates[component_index] += forward_field_grid[
                f"delta_{component}"
            ]
            forward_adjoint = max(
                maximum_difference(
                    forward_field_grid[axis],
                    adjoint_field_grid[axis],
                )
                for axis in "xyz"
            )
            forward_adjoint = max(
                forward_adjoint,
                maximum_difference(
                    forward_field_grid[f"delta_{component}"],
                    adjoint_field_grid[f"delta_{component}"],
                ),
            )
            field_index = max(
                maximum_difference(left, right)
                for left, right in zip(
                    field_coordinates, detail_coordinates
                )
            )
            maximum_coordinate_mismatch = max(
                maximum_coordinate_mismatch,
                forward_adjoint,
                field_index,
            )
            matrix = matrices[component]
            row_nnz = np.diff(matrix.indptr)
            coordinate_audit["components"][component] = {
                "shape": list(shapes[component]),
                "coordinate_bounds_m": {
                    axis: [float(value[0]), float(value[-1])]
                    for axis, value in zip("xyz", detail_coordinates)
                },
                "maximum_forward_adjoint_coordinate_mismatch_m": (
                    forward_adjoint
                ),
                "maximum_field_index_detail_coordinate_mismatch_m": (
                    field_index
                ),
                "maximum_local_assignment_distance_m": (
                    maximum_assignment_distance[component]
                ),
                "J_shape": list(matrix.shape),
                "J_nnz": int(matrix.nnz),
                "maximum_J_nonzeros_per_Yee_sample": int(
                    np.max(row_nnz)
                ),
                "active_J_row_count": int(np.count_nonzero(row_nnz)),
            }

        matrix_artifacts = {}
        for component, matrix in matrices.items():
            path = output / f"J_{component}.npz"
            sparse.save_npz(path, matrix)
            matrix_artifacts[component] = {
                "path": str(path),
                "byte_size": path.stat().st_size,
                "sha256": sha256(path),
            }
        layout_path = output / "component_yee_coordinates.npz"
        arrays = {"rho": rho}
        for component in "xyz":
            for axis, value in zip(
                "xyz", component_coordinates(completed_detail, component)
            ):
                arrays[f"{component}_{axis}_m"] = value
        np.savez_compressed(layout_path, **arrays)
        layout_artifact = {
            "path": str(layout_path),
            "byte_size": layout_path.stat().st_size,
            "sha256": sha256(layout_path),
        }

        worst_fd = max(
            item["mapping_only_FD_relative_error"]
            for item in direction_records.values()
        )
        worst_dot = max(
            item["JVP_VJP_dot_relative_error"]
            for item in direction_records.values()
        )
        passed = bool(
            completed_layout_epsilon_error == 0.0
            and baseline_roundtrip_error == 0.0
            and maximum_coordinate_mismatch < COORDINATE_LIMIT_M
            and worst_fd < FD_LIMIT
            and worst_dot < DOT_LIMIT
        )
        result = {
            "status": STATUS_PASS if passed else STATUS_FAIL,
            "passed": passed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": (
                "explicit component J_c=d epsilon_Yee,c/d rho81x81 "
                "from v261 layout index_detail"
            ),
            "density_shape": [81, 81],
            "density_bounds_m": {
                "x": [float(x_nodes[0]), float(x_nodes[-1])],
                "y": [float(y_nodes[0]), float(y_nodes[-1])],
            },
            "density_to_solver_chain": (
                "rho81x81 -> epsilon=1+rho*(1.38^2-1) -> "
                "n=sqrt(epsilon) -> importnk2 81x81x13 -> v261 "
                "component-specific conformal index_detail -> "
                "epsilon_Yee,c=index_c^2"
            ),
            "construction": {
                "method": (
                    f"period-{COLOR_PERIOD}x{COLOR_PERIOD} local coloring, "
                    "centered layout-only material perturbations"
                ),
                "centered_step": BUILD_STEP,
                "color_count": COLOR_PERIOD**2,
                "layout_index_detail_evaluations": (
                    2 * COLOR_PERIOD**2
                    + 2 * len(direction_records)
                    + 2
                ),
                "Maxwell_solves": 0,
                "per_pixel_Maxwell_solves": False,
                "nonzero_threshold": NONZERO_THRESHOLD,
                "maximum_allowed_local_assignment_distance_m": (
                    MAX_LOCAL_DISTANCE_M
                ),
                "empirical_normalization": False,
                "gradient_rescaling": False,
            },
            "coordinate_audit": coordinate_audit,
            "maximum_coordinate_mismatch_m": maximum_coordinate_mismatch,
            "completed_vs_layout_index_detail_epsilon_max_abs_error": (
                completed_layout_epsilon_error
            ),
            "baseline_layout_roundtrip_epsilon_max_abs_error": (
                baseline_roundtrip_error
            ),
            "directions": direction_records,
            "gates": {
                "worst_mapping_only_FD_relative_error": worst_fd,
                "mapping_only_FD_limit": FD_LIMIT,
                "worst_JVP_VJP_dot_relative_error": worst_dot,
                "dot_limit": DOT_LIMIT,
                "coordinate_mismatch_limit_m": COORDINATE_LIMIT_M,
            },
            "artifacts": {
                "forward_FSP": {
                    "path": str(forward_path),
                    "byte_size": forward_path.stat().st_size,
                    "sha256": args.forward_sha256,
                },
                "adjoint_FSP": {
                    "path": str(adjoint_path),
                    "byte_size": adjoint_path.stat().st_size,
                    "sha256": args.adjoint_sha256,
                },
                "component_J": matrix_artifacts,
                "coordinates": layout_artifact,
            },
            "optimization_run": False,
        }
        result_path = output / "component_yee_jacobian_result.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0 if passed else 2
    finally:
        if fdtd is not None:
            fdtd.close()


if __name__ == "__main__":
    raise SystemExit(main())
