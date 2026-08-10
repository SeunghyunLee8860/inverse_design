#!/usr/bin/env python3
"""Build a nonuniform complex density-to-Yee Jacobian without Maxwell solves.

The input is a completed, SHA-pinned uniform imported-material FSP.  Lumerical
is switched to layout and queried for its component-specific ``index_detail``
response to colored nodal perturbations.  No CPU or GPU Maxwell solve occurs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.finite_inverse_design.probe_v261_cpu_tfsf_device import (  # noqa: E402
    PABS_FIELD,
    PABS_INDEX,
)
from photothermal_pte.finite_inverse_design.run_v261_large_background_mixed_optical_adfd import (  # noqa: E402
    monitor_electric,
)
from photothermal_pte.finite_inverse_design.yee_material_jacobian import (  # noqa: E402
    SparseYeeMaterialJacobian,
)
from photothermal_pte.optimization_runs.gaussian10_contract import (  # noqa: E402
    silica_10um,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology import (  # noqa: E402
    optical as tairte4_flake_optical,
)

import run_complex_material_control as material_control  # noqa: E402
import audit_production_candidate_geometry as production_geometry  # noqa: E402


ISOLATED_IMPORTED_OBJECT = "rho0.5_imported_complex_block"
COLOR_PERIOD = 5
BUILD_STEP = 1.0e-4
CHECK_STEP = 1.0e-5
NONZERO_THRESHOLD = 1.0e-9
NONLOCAL_NUMERICAL_RELATIVE_THRESHOLD = 1.0e-12
ISOLATED_MAX_LOCAL_DISTANCE_M = 125.0e-9
# Period-5 coloring on the 100 nm TaIrTe4 design grid has a 250 nm
# nearest-color Voronoi radius.  Keep a 10% guard inside that aliasing limit.
# Near binary endpoints, Lumerical's conformal import stencil reaches the
# exact sqrt(150^2 + 100^2) nm = 180.278 nm x-component sample, so the older
# 175 nm diagnostic bound was too small even though the assignment is unique.
PRODUCTION_MAX_LOCAL_DISTANCE_M = 225.0e-9
FD_LIMIT = 1.0e-7
DOT_LIMIT = 1.0e-12
COORDINATE_LIMIT_M = 2.0e-18


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def epsilon_sio2() -> complex:
    material = silica_10um()
    return complex(material["epsilon_real"], material["epsilon_imag"])


def baseline_density(
    nodes: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    half_span_m: float = 5.0e-6,
) -> np.ndarray:
    if nodes is None:
        nodes = material_control.imported_nodes()
    x, y, _ = nodes
    xn = x[:, None] / half_span_m
    yn = y[None, :] / half_span_m
    rho = (
        0.5
        + 0.12 * np.sin(0.70 * np.pi * xn) * np.cos(0.55 * np.pi * yn)
        + 0.04 * xn
        - 0.03 * yn
    )
    if np.min(rho) <= 0.2 or np.max(rho) >= 0.8:
        raise RuntimeError("nonuniform baseline lacks centered-FD margin")
    return rho


def directions(
    nodes: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    half_span_m: float = 5.0e-6,
) -> dict[str, np.ndarray]:
    if nodes is None:
        nodes = material_control.imported_nodes()
    x, y, _ = nodes
    xn = x[:, None] / half_span_m
    yn = y[None, :] / half_span_m
    rng = np.random.default_rng(1008501)
    raw = {
        "uniform": np.ones((x.size, y.size)),
        "smooth_asymmetric": (
            np.cos(0.43 * np.pi * (xn - 0.17))
            * np.sin(0.68 * np.pi * (yn + 0.09))
            + 0.13 * xn
            - 0.07 * yn
        ),
        "central_localized": np.exp(-(xn**2 + yn**2) / (2.0 * 0.14**2)),
        "design_edge_localized": np.exp(
            -((xn - 0.88) ** 2 + (yn + 0.23) ** 2) / (2.0 * 0.075**2)
        ),
        "fixed_seed_random": rng.normal(size=(x.size, y.size)),
    }
    return {
        name: value / np.max(np.abs(value))
        for name, value in raw.items()
    }


def imported_index(
    rho: np.ndarray,
    nodes: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    if nodes is None:
        nodes = material_control.imported_nodes()
    density = np.asarray(rho, float)
    x, y, z = nodes
    if density.shape != (x.size, y.size):
        raise ValueError(f"rho shape {density.shape} differs from {(x.size, y.size)}")
    if np.any(density < 0.0) or np.any(density > 1.0):
        raise ValueError("rho leaves [0,1]")
    epsilon = 1.0 + density * (epsilon_sio2() - 1.0)
    index = np.sqrt(epsilon)
    if np.any(index.imag < 0.0):
        raise RuntimeError("passive complex square-root branch was not selected")
    return np.repeat(index[:, :, None], z.size, axis=2)


def set_density(
    fdtd: object,
    rho: np.ndarray,
    *,
    imported_object: str,
    nodes: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    if int(fdtd.getnamednumber(imported_object)) != 1:
        raise RuntimeError(f"expected exactly one {imported_object!r}")
    x, y, z = nodes
    fdtd.select(imported_object)
    if int(fdtd.importnk2(imported_index(rho, nodes), x, y, z)) != 1:
        raise RuntimeError("importnk2 returned failure")


def set_tairte4_flake_density(
    fdtd: object,
    rho: np.ndarray,
    *,
    imported_object: str,
    nodes: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    if int(fdtd.getnamednumber(imported_object)) != 1:
        raise RuntimeError(f"expected exactly one {imported_object!r}")
    expected_nodes = tairte4_flake_optical.design_nodes()
    if any(
        np.max(np.abs(np.asarray(left) - np.asarray(right))) > 1e-18
        for left, right in zip(nodes, expected_nodes)
    ):
        raise RuntimeError("TaIrTe4 design-node contract mismatch")
    index, _ = tairte4_flake_optical.anisotropic_index(rho)
    fdtd.select(imported_object)
    if int(fdtd.importnk2(index, *nodes)) != 1:
        raise RuntimeError("anisotropic TaIrTe4 importnk2 returned failure")


def index_detail(fdtd: object) -> dict[str, np.ndarray]:
    dataset = fdtd.getresult(PABS_INDEX, "index_detail")
    frequency = np.asarray(dataset["f"], float).reshape(-1)
    target_hz = 299792458.0 / 10.0e-6
    frequency_index = int(np.argmin(np.abs(frequency - target_hz)))
    result = {
        key: np.asarray(dataset[key], float).reshape(-1)
        for key in ("x", "x_offset", "y", "y_offset", "z", "z_offset")
    }
    shape = (result["x"].size, result["y"].size, result["z"].size)
    for component in "xyz":
        raw = np.asarray(dataset[f"index_{component}"])
        if raw.shape != (*shape, frequency.size):
            raise RuntimeError(
                f"index_{component} shape {raw.shape} != {(*shape, frequency.size)}"
            )
        result[f"index_{component}"] = raw[..., frequency_index]
        result[f"epsilon_{component}"] = result[f"index_{component}"] ** 2
    result["frequency_hz"] = np.asarray([frequency[frequency_index]], float)
    return result


def component_coordinates(
    detail: dict[str, np.ndarray], component: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        detail["x_offset"] if component == "x" else detail["x"],
        detail["y_offset"] if component == "y" else detail["y"],
        detail["z_offset"] if component == "z" else detail["z"],
    )


def nearest_colored_node(
    coordinate: np.ndarray, nodes: np.ndarray, color: int
) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.arange(color, nodes.size, COLOR_PERIOD)
    distance = np.abs(
        np.asarray(coordinate, float)[:, None] - nodes[candidates][None, :]
    )
    local = np.argmin(distance, axis=1)
    return candidates[local], distance[np.arange(coordinate.size), local]


def open_fdtd() -> tuple[object, object]:
    wrapper = material_control.load_source_wrapper()
    audit = wrapper.source_audit
    os.environ["VC_LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
    os.environ["LUMERICAL_ROOT"] = str(audit.APPROVED_ROOT)
    os.environ["LUMERICAL_PYTHONPATH"] = str(audit.APPROVED_API)
    os.environ["PATH"] = f"{audit.APPROVED_ROOT / 'bin'}:{os.environ.get('PATH','')}"
    for path in (audit.STAGE1, REPOSITORY / "photothermal_pte"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    helper = audit.load_module(audit.API_HELPER, "run002_jacobian_lumerical_api")
    installation = type(
        "Installation",
        (),
        {
            "version_key": "v261",
            "root": audit.APPROVED_ROOT,
            "lumapi_path": audit.APPROVED_API / "lumapi.py",
            "device_executable": audit.APPROVED_ROOT / "bin" / "device",
        },
    )()
    lumapi = helper.load_lumapi(installation)
    return lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"}), audit


def build_tairte4_local_epsilon_operator(
    fdtd: object,
    rho: np.ndarray,
    *,
    step: float = BUILD_STEP,
    color_period: int = COLOR_PERIOD,
    include_saturated_endpoints: bool = False,
) -> tuple[SparseYeeMaterialJacobian, dict[str, object]]:
    """Build the exact local component-epsilon Jacobian at ``rho``.

    This is a layout-only colored finite-difference of Lumerical's own
    ``importnk2 -> index_detail`` map.  It performs no Maxwell solve.  The
    Centered differences are used away from endpoints. Exact one-sided
    differences are used at 0/1 without clipping or moving the baseline.
    """

    value = np.asarray(rho, dtype=np.float64)
    nodes = tairte4_flake_optical.design_nodes()
    if value.shape != (nodes[0].size, nodes[1].size):
        raise ValueError("TaIrTe4 local-J density shape mismatch")
    if not np.all(np.isfinite(value)) or np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError("local-J density is outside the unit interval")
    if color_period < 3:
        raise ValueError("local-J color period is too small")
    set_tairte4_flake_density(
        fdtd,
        value,
        imported_object=tairte4_flake_optical.DESIGN_OBJECT,
        nodes=nodes,
    )
    baseline = index_detail(fdtd)
    shapes = {c: baseline[f"epsilon_{c}"].shape for c in "xyz"}
    row_parts = {c: [] for c in "xyz"}
    column_parts = {c: [] for c in "xyz"}
    value_parts = {c: [] for c in "xyz"}
    maximum_assignment_distance = {c: 0.0 for c in "xyz"}
    suppressed_nonlocal_numerical_tail = {
        c: {"count": 0, "maximum_abs": 0.0, "maximum_relative": 0.0}
        for c in "xyz"
    }
    evaluation_count = 1

    def append_response(
        component: str,
        derivative: np.ndarray,
        active_nodes: np.ndarray,
        *,
        perturbation_class: str,
        color: tuple[int, int],
    ) -> None:
        flat = derivative.reshape(-1)
        rows = np.flatnonzero(np.abs(flat) > NONZERO_THRESHOLD)
        if rows.size == 0:
            return
        node_indices = np.argwhere(active_nodes)
        if node_indices.size == 0:
            raise RuntimeError("nonzero local-J response has no active perturbation node")
        tree = cKDTree(
            np.column_stack((nodes[0][node_indices[:, 0]], nodes[1][node_indices[:, 1]]))
        )
        ix, iy, _ = np.unravel_index(rows, shapes[component])
        coordinates = component_coordinates(baseline, component)
        distance, nearest = tree.query(
            np.column_stack((coordinates[0][ix], coordinates[1][iy])), k=1
        )
        nonlocal_rows = distance > PRODUCTION_MAX_LOCAL_DISTANCE_M
        if np.any(nonlocal_rows):
            response = np.abs(flat[rows])
            response_scale = float(np.max(response))
            relative_response = response / max(response_scale, np.finfo(float).tiny)
            numerical_tail = nonlocal_rows & (
                relative_response <= NONLOCAL_NUMERICAL_RELATIVE_THRESHOLD
            )
            physically_significant_nonlocal = nonlocal_rows & ~numerical_tail
            if np.any(physically_significant_nonlocal):
                first = int(np.flatnonzero(physically_significant_nonlocal)[0])
                yee_coordinate = (
                    float(coordinates[0][ix[first]]),
                    float(coordinates[1][iy[first]]),
                )
                nearest_node = node_indices[int(np.asarray(nearest)[first])]
                node_coordinate = (
                    float(nodes[0][nearest_node[0]]),
                    float(nodes[1][nearest_node[1]]),
                )
                raise RuntimeError(
                    f"{component} local-J response is nonlocal: "
                    f"max_distance_m={float(np.max(distance)):.9e}, "
                    f"max_nonlocal_abs={float(np.max(response[physically_significant_nonlocal])):.9e}, "
                    f"max_response_abs={response_scale:.9e}, "
                    f"max_nonlocal_relative={float(np.max(relative_response[physically_significant_nonlocal])):.9e}, "
                    f"nonlocal_count={int(np.count_nonzero(physically_significant_nonlocal))}, "
                    f"perturbation_class={perturbation_class}, color={color}, "
                    f"first_Yee_xy_m={yee_coordinate}, first_nearest_node_xy_m={node_coordinate}"
                )
            tail = suppressed_nonlocal_numerical_tail[component]
            tail["count"] += int(np.count_nonzero(numerical_tail))
            tail["maximum_abs"] = max(
                float(tail["maximum_abs"]), float(np.max(response[numerical_tail]))
            )
            tail["maximum_relative"] = max(
                float(tail["maximum_relative"]),
                float(np.max(relative_response[numerical_tail])),
            )
            keep = ~numerical_tail
            rows = rows[keep]
            distance = distance[keep]
            nearest = np.asarray(nearest)[keep]
            if rows.size == 0:
                return
        maximum_assignment_distance[component] = max(
            maximum_assignment_distance[component], float(np.max(distance))
        )
        selected = node_indices[np.asarray(nearest, dtype=int)]
        row_parts[component].append(rows)
        column_parts[component].append(selected[:, 0] * nodes[1].size + selected[:, 1])
        value_parts[component].append(flat[rows])

    for color_x in range(color_period):
        for color_y in range(color_period):
            color = np.zeros_like(value, dtype=bool)
            color[color_x::color_period, color_y::color_period] = True
            centered = color & (value >= step) & (value <= 1.0 - step)
            lower = color & (value < step)
            upper = color & (value > 1.0 - step)
            responses: list[tuple[str, np.ndarray, np.ndarray]] = []
            if np.any(centered):
                pair = {}
                for sign, label in ((1.0, "plus"), (-1.0, "minus")):
                    set_tairte4_flake_density(
                        fdtd,
                        value + sign * step * centered,
                        imported_object=tairte4_flake_optical.DESIGN_OBJECT,
                        nodes=nodes,
                    )
                    pair[label] = index_detail(fdtd)
                    evaluation_count += 1
                for component in "xyz":
                    responses.append((
                        component,
                        (pair["plus"][f"epsilon_{component}"] - pair["minus"][f"epsilon_{component}"]) / (2.0 * step),
                        centered,
                    ))
            endpoint_cases = (
                ((lower, 1.0), (upper, -1.0))
                if include_saturated_endpoints
                else ()
            )
            for endpoint_mask, sign in endpoint_cases:
                if not np.any(endpoint_mask):
                    continue
                set_tairte4_flake_density(
                    fdtd,
                    value + sign * step * endpoint_mask,
                    imported_object=tairte4_flake_optical.DESIGN_OBJECT,
                    nodes=nodes,
                )
                endpoint = index_detail(fdtd)
                evaluation_count += 1
                for component in "xyz":
                    derivative = (
                        endpoint[f"epsilon_{component}"] - baseline[f"epsilon_{component}"]
                    ) / (sign * step)
                    responses.append((component, derivative, endpoint_mask))
            for component, derivative, active_nodes in responses:
                if active_nodes is centered:
                    perturbation_class = "centered"
                elif active_nodes is lower:
                    perturbation_class = "lower_endpoint"
                else:
                    perturbation_class = "upper_endpoint"
                append_response(
                    component,
                    derivative,
                    active_nodes,
                    perturbation_class=perturbation_class,
                    color=(color_x, color_y),
                )
    set_tairte4_flake_density(
        fdtd,
        value,
        imported_object=tairte4_flake_optical.DESIGN_OBJECT,
        nodes=nodes,
    )
    matrices = {}
    for component in "xyz":
        matrix = sparse.csr_matrix(
            (
                np.concatenate(value_parts[component]),
                (np.concatenate(row_parts[component]), np.concatenate(column_parts[component])),
            ),
            shape=(int(np.prod(shapes[component])), value.size),
        )
        matrix.sum_duplicates()
        matrices[component] = matrix
    operator = SparseYeeMaterialJacobian(
        density_shape=value.shape,
        component_shapes=shapes,
        matrices=matrices,
    )
    return operator, {
        "method": "current-density layout-only colored centered FD of v261 importnk2-to-index_detail",
        "step": float(step),
        "color_period": int(color_period),
        "layout_index_detail_evaluations": int(evaluation_count + 1),
        "maximum_assignment_distance_m": maximum_assignment_distance,
        "suppressed_nonlocal_numerical_tail": suppressed_nonlocal_numerical_tail,
        "nonlocal_numerical_relative_threshold": NONLOCAL_NUMERICAL_RELATIVE_THRESHOLD,
        "saturated_endpoint_policy": (
            "one-sided layout FD"
            if include_saturated_endpoints
            else "active-set frozen; only centered physical-density nodes are differentiated"
        ),
        "frozen_lower_endpoint_node_count": (
            0 if include_saturated_endpoints else int(np.count_nonzero(value < step))
        ),
        "frozen_upper_endpoint_node_count": (
            0 if include_saturated_endpoints else int(np.count_nonzero(value > 1.0 - step))
        ),
        "Maxwell_solves": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-project", required=True, type=Path)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--geometry",
        choices=("isolated", "production", "selected_production", "tairte4_flake"),
        default="isolated",
        help="Select the imported-object/nodal-coordinate contract.",
    )
    args = parser.parse_args()
    if args.geometry == "tairte4_flake":
        imported_object = tairte4_flake_optical.DESIGN_OBJECT
        nodes = tairte4_flake_optical.design_nodes()
        half_span_m = 0.5 * min(
            tairte4_flake_optical.CONTRACT.design_span_x_m,
            tairte4_flake_optical.CONTRACT.design_span_y_m,
        )
        maximum_local_distance_m = PRODUCTION_MAX_LOCAL_DISTANCE_M
        validated_status = "VALIDATED_TAIRTE4_FLAKE_COMPLEX_COMPONENT_YEE_JACOBIAN"
        failed_status = "FAILED_TAIRTE4_FLAKE_COMPLEX_COMPONENT_YEE_JACOBIAN"
        contract = tairte4_flake_optical.CONTRACT
        scope = (
            f"10 um {contract.design_span_x_m * 1e6:g}x"
            f"{contract.design_span_y_m * 1e6:g}x"
            f"{contract.flake_thickness_m * 1e6:g} um anisotropic "
            f"TaIrTe4-to-air {contract.geometry_mode} design; layout-only "
            f"J_c=d epsilon_Yee,c/d rho on {nodes[0].size}x{nodes[1].size} nodes"
        )
        density_setter = set_tairte4_flake_density
    elif args.geometry == "selected_production":
        imported_object = production_geometry.SELECTED_DESIGN_OBJECT
        nodes = production_geometry.design_nodes(
            production_geometry.SELECTED_DESIGN_BOUNDS,
            production_geometry.SELECTED_DESIGN_SHAPE,
        )
        half_span_m = 9.3e-6
        maximum_local_distance_m = PRODUCTION_MAX_LOCAL_DISTANCE_M
        validated_status = "VALIDATED_SELECTED_PRODUCTION_COMPLEX_COMPONENT_YEE_JACOBIAN"
        failed_status = "FAILED_SELECTED_PRODUCTION_COMPLEX_COMPONENT_YEE_JACOBIAN"
        scope = (
            "10 um selected 18.6x18.6x1 um imported complex-SiO2 candidate; "
            "layout-only J_c=d epsilon_Yee,c/d rho on 373x373 nodes"
        )
        density_setter = set_density
    elif args.geometry == "production":
        imported_object = production_geometry.DESIGN_OBJECT
        nodes = production_geometry.design_nodes()
        half_span_m = 10.0e-6
        maximum_local_distance_m = PRODUCTION_MAX_LOCAL_DISTANCE_M
        validated_status = "VALIDATED_PRODUCTION_COMPLEX_COMPONENT_YEE_JACOBIAN"
        failed_status = "FAILED_PRODUCTION_COMPLEX_COMPONENT_YEE_JACOBIAN"
        scope = (
            "10 um production 20x20x1 um imported complex-SiO2 candidate; "
            "layout-only J_c=d epsilon_Yee,c/d rho on 201x201 nodes"
        )
        density_setter = set_density
    else:
        imported_object = ISOLATED_IMPORTED_OBJECT
        nodes = material_control.imported_nodes()
        half_span_m = 5.0e-6
        maximum_local_distance_m = ISOLATED_MAX_LOCAL_DISTANCE_M
        validated_status = "VALIDATED_NONUNIFORM_COMPLEX_COMPONENT_YEE_JACOBIAN_SMOKE"
        failed_status = "FAILED_NONUNIFORM_COMPLEX_COMPONENT_YEE_JACOBIAN_SMOKE"
        scope = (
            "10 um isolated 10x10x1 um imported complex-SiO2 control; "
            "layout-only J_c=d epsilon_Yee,c/d rho on 101x101 nodes; "
            "not the final production geometry"
        )
        density_setter = set_density
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    base_project = args.base_project.expanduser().resolve()
    result_path = output / "component_yee_jacobian_result.json"
    result: dict[str, object] = {
        "status": "BLOCKED_NONUNIFORM_COMPLEX_COMPONENT_YEE_JACOBIAN",
        "Maxwell_solves": 0,
        "optimization_run": False,
    }
    fdtd = None
    try:
        if not base_project.is_file():
            raise FileNotFoundError(base_project)
        actual_sha = sha256(base_project)
        if actual_sha != args.base_sha256:
            raise RuntimeError(f"base FSP SHA mismatch: {actual_sha}")
        fdtd, audit = open_fdtd()
        fdtd.load(str(base_project))
        _, field_grid = monitor_electric(fdtd, PABS_FIELD)
        completed_detail = index_detail(fdtd)
        fdtd.switchtolayout()
        baseline = (
            np.full((nodes[0].size, nodes[1].size), 0.5, dtype=np.float64)
            if args.geometry == "tairte4_flake"
            else baseline_density(nodes, half_span_m)
        )
        density_setter(fdtd, baseline, imported_object=imported_object, nodes=nodes)
        layout_detail = index_detail(fdtd)
        x_nodes, y_nodes, z_nodes = nodes
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
                mask = np.zeros_like(baseline)
                mask[color_x::COLOR_PERIOD, color_y::COLOR_PERIOD] = 1.0
                pair = {}
                for sign, label in ((1.0, "plus"), (-1.0, "minus")):
                    density_setter(
                        fdtd,
                        baseline + sign * BUILD_STEP * mask,
                        imported_object=imported_object,
                        nodes=nodes,
                    )
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
                    coordinates = component_coordinates(layout_detail, component)
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
                    if np.any(distance > maximum_local_distance_m):
                        raise RuntimeError(
                            f"{component} response is nonlocal under period-{COLOR_PERIOD} coloring; "
                            f"max distance={float(np.max(distance)):.6e} m"
                        )
                    row_parts[component].append(rows)
                    column_parts[component].append(node_x * y_nodes.size + node_y)
                    value_parts[component].append(flat[rows])

        density_setter(fdtd, baseline, imported_object=imported_object, nodes=nodes)
        baseline_detail = index_detail(fdtd)
        matrices = {}
        for component in "xyz":
            if not row_parts[component]:
                raise RuntimeError(f"empty {component} Jacobian")
            rows = np.concatenate(row_parts[component])
            columns = np.concatenate(column_parts[component])
            values = np.concatenate(value_parts[component])
            matrix = sparse.csr_matrix(
                (values, (rows, columns)),
                shape=(int(np.prod(shapes[component])), baseline.size),
            )
            matrix.sum_duplicates()
            matrices[component] = matrix
        operator = SparseYeeMaterialJacobian(
            density_shape=baseline.shape,
            component_shapes=shapes,
            matrices=matrices,
        )

        rng = np.random.default_rng(1008502)
        cotangent = {
            component: (
                rng.normal(size=shapes[component])
                + 1j * rng.normal(size=shapes[component])
            )
            for component in "xyz"
        }
        direction_records = {}
        for name, direction in directions(nodes, half_span_m).items():
            pair = {}
            for sign, label in ((1.0, "plus"), (-1.0, "minus")):
                density_setter(
                    fdtd,
                    baseline + sign * CHECK_STEP * direction,
                    imported_object=imported_object,
                    nodes=nodes,
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
                    np.linalg.norm(tangent[c] - finite_difference[c]) ** 2
                    for c in "xyz"
                )
            )
            reference_norm = max(
                np.sqrt(sum(np.linalg.norm(tangent[c]) ** 2 for c in "xyz")),
                np.sqrt(
                    sum(np.linalg.norm(finite_difference[c]) ** 2 for c in "xyz")
                ),
                np.finfo(float).tiny,
            )
            left = float(
                np.real(sum(np.sum(cotangent[c] * tangent[c]) for c in "xyz"))
            )
            right = float(np.vdot(direction, operator.vjp(cotangent)))
            direction_records[name] = {
                "direction_sha256": array_sha256(direction),
                "mapping_only_centered_FD_step": CHECK_STEP,
                "mapping_only_FD_relative_error": float(delta_norm / reference_norm),
                "JVP_VJP_dot_relative_error": float(
                    abs(left - right)
                    / max(abs(left), abs(right), np.finfo(float).tiny)
                ),
            }
        density_setter(fdtd, baseline, imported_object=imported_object, nodes=nodes)

        coordinate_audit = {"components": {}}
        maximum_coordinate_mismatch = 0.0
        for component in "xyz":
            detail_coordinates = component_coordinates(completed_detail, component)
            component_index = "xyz".index(component)
            field_coordinates = [
                np.asarray(field_grid[axis], float).copy() for axis in "xyz"
            ]
            field_coordinates[component_index] += field_grid[f"delta_{component}"]
            mismatch = max(
                maximum_difference(left, right)
                for left, right in zip(field_coordinates, detail_coordinates)
            )
            maximum_coordinate_mismatch = max(maximum_coordinate_mismatch, mismatch)
            row_nnz = np.diff(matrices[component].indptr)
            support_axes = []
            for axis, coordinate, node_coordinate in zip(
                "xyz", detail_coordinates, (x_nodes, y_nodes, z_nodes)
            ):
                support_axes.append(
                    (coordinate >= node_coordinate[0] - 2.0e-18)
                    & (coordinate <= node_coordinate[-1] + 2.0e-18)
                )
            support_intersection = (
                support_axes[0][:, None, None]
                & support_axes[1][None, :, None]
                & support_axes[2][None, None, :]
            )
            coordinate_audit["components"][component] = {
                "shape": list(shapes[component]),
                "coordinate_bounds_m": {
                    axis: [float(values[0]), float(values[-1])]
                    for axis, values in zip("xyz", detail_coordinates)
                },
                "staggering_offset_axis": component,
                "maximum_field_index_coordinate_mismatch_m": mismatch,
                "maximum_local_assignment_distance_m": maximum_assignment_distance[
                    component
                ],
                "J_shape": list(matrices[component].shape),
                "J_nnz": int(matrices[component].nnz),
                "maximum_J_nonzeros_per_Yee_sample": int(np.max(row_nnz)),
                "active_J_row_count": int(np.count_nonzero(row_nnz)),
                "exact_design_support_intersection": {
                    "Yee_sample_count": int(np.count_nonzero(support_intersection)),
                    "active_J_rows_inside": int(
                        np.count_nonzero(
                            (row_nnz > 0) & support_intersection.reshape(-1)
                        )
                    ),
                    "active_J_rows_outside": int(
                        np.count_nonzero(
                            (row_nnz > 0) & ~support_intersection.reshape(-1)
                        )
                    ),
                },
            }

        matrix_artifacts = {}
        for component, matrix in matrices.items():
            path = output / f"J_{component}.npz"
            sparse.save_npz(path, matrix)
            matrix_artifacts[component] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        layout_path = output / "component_yee_coordinates_and_density.npz"
        arrays = {"rho": baseline, "x_nodes_m": x_nodes, "y_nodes_m": y_nodes}
        for component in "xyz":
            for axis, values in zip(
                "xyz", component_coordinates(completed_detail, component)
            ):
                arrays[f"{component}_{axis}_m"] = values
        np.savez_compressed(layout_path, **arrays)

        worst_fd = max(
            row["mapping_only_FD_relative_error"]
            for row in direction_records.values()
        )
        worst_dot = max(
            row["JVP_VJP_dot_relative_error"]
            for row in direction_records.values()
        )
        roundtrip_error = max(
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
        passed = bool(
            roundtrip_error == 0.0
            and maximum_coordinate_mismatch < COORDINATE_LIMIT_M
            and worst_fd < FD_LIMIT
            and worst_dot < DOT_LIMIT
        )
        result = {
            "status": validated_status if passed else failed_status,
            "passed": passed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": scope,
            "geometry": args.geometry,
            "imported_object": imported_object,
            "base_FSP": {
                "path": str(base_project),
                "size_bytes": base_project.stat().st_size,
                "sha256": actual_sha,
            },
            "density_shape": list(baseline.shape),
            "density_range": [float(np.min(baseline)), float(np.max(baseline))],
            "density_sha256": array_sha256(baseline),
            "density_to_solver_chain": (
                "rho -> component epsilon_c=1+rho*(epsilon_TaIrTe4,c-1) -> passive sqrt -> importnk2 -> v261 component index_detail -> epsilon_Yee,c=index_c^2"
                if args.geometry == "tairte4_flake"
                else "rho -> epsilon=1+rho*(epsilon_SiO2(10um)-1) -> passive sqrt -> importnk2 -> v261 component index_detail -> epsilon_Yee,c=index_c^2"
            ),
            "epsilon_endpoints": (
                {
                    component: [value.real, value.imag]
                    for component, value in tairte4_flake_optical.material_epsilon().items()
                }
                if args.geometry == "tairte4_flake"
                else {"SiO2": [epsilon_sio2().real, epsilon_sio2().imag]}
            ),
            "construction": {
                "color_period": COLOR_PERIOD,
                "color_count": COLOR_PERIOD**2,
                "build_step": BUILD_STEP,
                "check_step": CHECK_STEP,
                "layout_index_detail_evaluations": 2 * COLOR_PERIOD**2
                + 2 * len(direction_records)
                + 3,
                "Maxwell_solves": 0,
                "per_pixel_Maxwell_solves": False,
                "empirical_normalization": False,
                "gradient_rescaling": False,
            },
            "coordinate_audit": coordinate_audit,
            "maximum_coordinate_mismatch_m": maximum_coordinate_mismatch,
            "baseline_layout_roundtrip_epsilon_max_abs_error": roundtrip_error,
            "directions": direction_records,
            "gates": {
                "worst_mapping_only_FD_relative_error": worst_fd,
                "mapping_only_FD_limit": FD_LIMIT,
                "worst_JVP_VJP_dot_relative_error": worst_dot,
                "dot_limit": DOT_LIMIT,
                "coordinate_mismatch_limit_m": COORDINATE_LIMIT_M,
                "maximum_local_assignment_distance_limit_m": maximum_local_distance_m,
            },
            "artifacts": {
                "component_J": matrix_artifacts,
                "coordinates_and_density": {
                    "path": str(layout_path),
                    "size_bytes": layout_path.stat().st_size,
                    "sha256": sha256(layout_path),
                },
            },
            "optimization_run": False,
        }
    except Exception as exc:
        result.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
        result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
