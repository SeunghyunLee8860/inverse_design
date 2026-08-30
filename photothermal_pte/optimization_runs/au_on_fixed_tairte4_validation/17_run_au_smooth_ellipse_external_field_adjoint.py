#!/usr/bin/env python3
"""AD--FD certificate for the smooth scalar-Au ellipse boundary kernel.

This wrapper reuses the fixed-external-field objective and GPU FieldRegion
adjoint implementation from stage 12, but replaces the rejected rectangular
endpoint rule with an endpoint-free Gaussian quadrature over every edge of a
512-vertex smooth closed ellipse.  The shape parameter is the x semi-axis.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE_SCRIPT = HERE / "12_run_au_sharp_interface_external_field_adjoint.py"
ELLIPSE_HALF_Y_M = 10.0e-6
ELLIPSE_VERTICES = 512
SMOOTH_FD_CASES = {
    0.10: (
        "smooth_ellipse_a7p9_b10_edge25_forward",
        "smooth_ellipse_a8p1_b10_edge25_forward",
    ),
    0.05: (
        "smooth_ellipse_a7p95_b10_edge25_forward",
        "smooth_ellipse_a8p05_b10_edge25_forward",
    ),
}
BASELINE_CASE = "smooth_ellipse_a8p0_b10_edge25_forward"


def load_source_module():
    spec = importlib.util.spec_from_file_location("au_rect_external_field", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source = load_source_module()


def ellipse_boundary_quadrature(
    *, half_width_m: float, gauss_order_per_edge: int
) -> dict[str, np.ndarray]:
    """Return endpoint-free boundary samples and the x-radius shape velocity."""

    if gauss_order_per_edge < 1:
        raise ValueError("gauss_order_per_edge must be positive")
    if half_width_m <= 0.0:
        raise ValueError("half_width_m must be positive")
    theta = np.arange(ELLIPSE_VERTICES, dtype=float) * (
        2.0 * np.pi / ELLIPSE_VERTICES
    )
    vertices = np.column_stack(
        (half_width_m * np.cos(theta), ELLIPSE_HALF_Y_M * np.sin(theta))
    )
    vertex_derivative = np.column_stack((np.cos(theta), np.zeros_like(theta)))
    next_vertices = np.roll(vertices, -1, axis=0)
    next_derivative = np.roll(vertex_derivative, -1, axis=0)
    edge_vector = next_vertices - vertices
    edge_length = np.linalg.norm(edge_vector, axis=1)
    if np.any(edge_length <= 0.0):
        raise RuntimeError("degenerate ellipse polygon edge")
    # CCW polygon: the outward normal is the right-hand normal.
    normals_xy = np.column_stack((edge_vector[:, 1], -edge_vector[:, 0]))
    normals_xy /= edge_length[:, None]

    legendre_coordinate, legendre_weight = np.polynomial.legendre.leggauss(
        gauss_order_per_edge
    )
    fraction = 0.5 * (legendre_coordinate + 1.0)
    weights = 0.5 * edge_length[:, None] * legendre_weight[None, :]
    points = (
        vertices[:, None, :] * (1.0 - fraction[None, :, None])
        + next_vertices[:, None, :] * fraction[None, :, None]
    )
    shape_velocity = (
        vertex_derivative[:, None, :] * (1.0 - fraction[None, :, None])
        + next_derivative[:, None, :] * fraction[None, :, None]
    )
    normals = np.zeros((*points.shape[:2], 3), float)
    normals[..., :2] = normals_xy[:, None, :]
    normal_velocity = np.sum(shape_velocity * normals[..., :2], axis=-1)
    return {
        "points_xy": points,
        "normals_xyz": normals,
        "normal_velocity_m_per_m": normal_velocity,
        "arc_weights_m": weights,
    }


def ellipse_polygon_boundary_integral(
    forward_fields,
    adjoint_fields,
    *,
    half_width_m: float,
    epsilon_au: complex,
    gauss_order_per_edge: int,
    gauss_order_z: int = 1,
) -> dict[str, object]:
    """Integrate the moving lateral surface without edge or z endpoints."""

    if gauss_order_z < 1:
        raise ValueError("gauss_order_z must be positive")

    quadrature = ellipse_boundary_quadrature(
        half_width_m=half_width_m,
        gauss_order_per_edge=gauss_order_per_edge,
    )
    points = quadrature["points_xy"]
    normals = quadrature["normals_xyz"]
    normal_velocity = quadrature["normal_velocity_m_per_m"]
    weights = quadrature["arc_weights_m"]
    lateral_x = points[..., 0].reshape(-1)
    lateral_y = points[..., 1].reshape(-1)
    lateral_normal = normals.reshape(-1, 3)
    lateral_velocity = normal_velocity.reshape(-1)
    lateral_weight = weights.reshape(-1)
    z_legendre, z_legendre_weight = np.polynomial.legendre.leggauss(
        gauss_order_z
    )
    depth = source.AU_Z_MAX_M - source.AU_Z_MIN_M
    z_nodes = (
        0.5 * (source.AU_Z_MIN_M + source.AU_Z_MAX_M)
        + 0.5 * depth * z_legendre
    )
    z_weights = 0.5 * depth * z_legendre_weight
    x = np.repeat(lateral_x, gauss_order_z)
    y = np.repeat(lateral_y, gauss_order_z)
    z = np.tile(z_nodes, lateral_x.size)
    wavelength = np.full_like(x, source.WAVELENGTH_M)
    normal = np.repeat(lateral_normal, gauss_order_z, axis=0)

    surface_weights = (
        lateral_weight[:, None] * z_weights[None, :]
    ).reshape(-1)
    surface_velocity = np.repeat(lateral_velocity, gauss_order_z)
    # The no-interpolation Yee-field evaluator can allocate several large
    # temporary arrays.  Batch the exact same quadrature to keep the full-z
    # certificate independent of host-memory limits.
    batch_size = 8192
    positive = 0.0
    negative = 0.0
    value = 0.0
    all_finite = True
    for start in range(0, x.size, batch_size):
        stop = min(start + batch_size, x.size)
        selection = slice(start, stop)
        ef = np.asarray(
            forward_fields.getfield(
                x[selection], y[selection], z[selection], wavelength[selection]
            ),
            complex,
        )
        df = np.asarray(
            forward_fields.getDfield(
                x[selection], y[selection], z[selection], wavelength[selection]
            ),
            complex,
        )
        ea = np.asarray(
            adjoint_fields.getfield(
                x[selection], y[selection], z[selection], wavelength[selection]
            ),
            complex,
        )
        da = np.asarray(
            adjoint_fields.getDfield(
                x[selection], y[selection], z[selection], wavelength[selection]
            ),
            complex,
        )
        expected = (stop - start, 3)
        if any(array.shape != expected for array in (ef, df, ea, da)):
            raise RuntimeError("unexpected smooth-boundary vector field shape")
        finite = all(np.all(np.isfinite(array)) for array in (ef, df, ea, da))
        all_finite = all_finite and finite
        if not finite:
            raise RuntimeError("non-finite smooth-boundary field")
        local_normal = normal[selection]
        ef_parallel = ef - np.sum(ef * local_normal, axis=-1)[:, None] * local_normal
        ea_parallel = ea - np.sum(ea * local_normal, axis=-1)[:, None] * local_normal
        df_perp = np.sum(df * local_normal, axis=-1)[:, None] * local_normal
        da_perp = np.sum(da * local_normal, axis=-1)[:, None] * local_normal
        kernel = (
            2.0
            * source.EPS0
            * (epsilon_au - source.AIR_EPSILON)
            * np.sum(ef_parallel * ea_parallel, axis=-1)
            + (1.0 / source.AIR_EPSILON - 1.0 / epsilon_au)
            / source.EPS0
            * np.sum(df_perp * da_perp, axis=-1)
        )
        weighted = (
            np.real(kernel)
            * surface_velocity[selection]
            * surface_weights[selection]
        )
        positive += float(np.sum(weighted[weighted > 0.0]))
        negative += float(np.sum(weighted[weighted < 0.0]))
        value += float(np.sum(weighted))
    return {
        "rule": (
            "tensor-product endpoint-free Gauss-Legendre integration over "
            "every edge of a 512-vertex CCW ellipse and the full Au depth"
        ),
        "ellipse_vertex_count": ELLIPSE_VERTICES,
        "gauss_order_per_edge": gauss_order_per_edge,
        "gauss_order_z": gauss_order_z,
        "sample_count": int(x.size),
        "ellipse_x_semi_axis_m": half_width_m,
        "ellipse_y_semi_axis_m": ELLIPSE_HALF_Y_M,
        "z_quadrature_node_bounds_m": [
            float(np.min(z_nodes)),
            float(np.max(z_nodes)),
        ],
        "fixed_depth_m": depth,
        "z_endpoints_sampled": False,
        "interpolation_batch_size": batch_size,
        "polygon_vertices_sampled": False,
        "normal_velocity_range_m_per_m": [
            float(np.min(normal_velocity)),
            float(np.max(normal_velocity)),
        ],
        "positive_contribution_J_proxy_per_m": positive,
        "negative_contribution_J_proxy_per_m": negative,
        "total_J_proxy_per_m": value,
        "all_finite": all_finite,
    }


def official_ellipse_integral(
    forward_fields,
    adjoint_fields,
    *,
    half_width_m: float,
    epsilon_au: complex,
    n_points: int,
    half_y_m: float,
) -> dict[str, object]:
    del half_y_m
    order = {201: 1, 401: 2, 801: 4, 1601: 8}[int(n_points)]
    result = ellipse_polygon_boundary_integral(
        forward_fields,
        adjoint_fields,
        half_width_m=half_width_m,
        epsilon_au=epsilon_au,
        gauss_order_per_edge=order,
        gauss_order_z=1,
    )
    result["legacy_n_points_argument"] = int(n_points)
    return result


def midpoint_ellipse_integral(
    forward_fields,
    adjoint_fields,
    *,
    half_width_m: float,
    half_y_m: float,
    epsilon_au: complex,
    dy_m: float,
    dz_m: float,
) -> dict[str, object]:
    del half_y_m
    edge_order = int(round(100.0e-9 / float(dy_m)))
    z_order = int(
        round((source.AU_Z_MAX_M - source.AU_Z_MIN_M) / float(dz_m))
    )
    return ellipse_polygon_boundary_integral(
        forward_fields,
        adjoint_fields,
        half_width_m=half_width_m,
        epsilon_au=epsilon_au,
        gauss_order_per_edge=max(1, edge_order),
        gauss_order_z=max(1, z_order),
    )


def option_present(arguments: list[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in arguments)


def option_value(arguments: list[str], option: str) -> str | None:
    for index, value in enumerate(arguments):
        if value.startswith(f"{option}="):
            return value.split("=", 1)[1]
        if value == option and index + 1 < len(arguments):
            return arguments[index + 1]
    return None


def main() -> int:
    arguments = list(sys.argv[1:])
    output_value = option_value(arguments, "--output-dir")
    if output_value is None:
        raise ValueError("--output-dir is required")
    if not option_present(arguments, "--corner-free-control"):
        arguments.append("--corner-free-control")
    if not option_present(arguments, "--baseline-case"):
        arguments.extend(("--baseline-case", BASELINE_CASE))

    source.CORNER_FREE_FD_CASES = SMOOTH_FD_CASES
    source.CORNER_FREE_HALF_Y_M = ELLIPSE_HALF_Y_M
    source.official_center_depth_integral = official_ellipse_integral
    source.midpoint_surface_integral = midpoint_ellipse_integral
    sys.argv = [sys.argv[0], *arguments]
    source.main()

    result_path = Path(output_value).expanduser().resolve() / (
        "au_sharp_interface_external_field_result.json"
    )
    result = json.loads(result_path.read_text())
    result["geometry_control"] = {
        "representation": "smooth_closed_binary_scalar_Au_ellipse",
        "ellipse_x_semi_axis_m": 8.0e-6,
        "ellipse_y_semi_axis_m": ELLIPSE_HALF_Y_M,
        "ellipse_vertex_count": ELLIPSE_VERTICES,
        "shape_parameter": "x semi-axis",
        "fixed_depth_m": 50.0e-9,
        "lateral_90_degree_corners": False,
        "polygon_vertices_sampled_by_quadrature": False,
    }
    result["boundary_quadrature_method_selected"] = (
        "endpoint-free Gauss-Legendre closed-ellipse lateral boundary integral"
    )
    result["production_Au_optimization_permitted"] = False
    if option_present(arguments, "--fd-precheck-only"):
        result["status"] = (
            "READY_AU_SMOOTH_ELLIPSE_EXTERNAL_FIELD_ADJOINT_PRECHECK"
            if result.get("precheck_passed")
            else "FAILED_AU_SMOOTH_ELLIPSE_EXTERNAL_FIELD_FD_PRECHECK"
        )
        result["remaining_blocker"] = (
            "run the smooth-boundary GPU adjoint; no optical gradient has "
            "been certified by this precheck"
        )
        result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "precheck_passed": result.get("precheck_passed"),
                    "FD_step_plateau_relative_change": result.get(
                        "FD_step_plateau_relative_change"
                    ),
                },
                indent=2,
            )
        )
        return 0 if result.get("precheck_passed") else 2
    result["remaining_blocker"] = (
        "this certifies only the field-mediated smooth-shape kernel for a fixed "
        "external objective; the direct moving-Au P_Q term remains unvalidated"
    )
    if result.get("passed"):
        result["status"] = "VALIDATED_AU_SMOOTH_ELLIPSE_BOUNDARY_KERNEL_ADFD"
    else:
        result["status"] = "FAILED_AU_SMOOTH_ELLIPSE_BOUNDARY_KERNEL_ADFD"
    result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result.get("passed"),
                "FD_step_plateau_relative_change": result.get(
                    "FD_step_plateau_relative_change"
                ),
                "boundary_quadrature_final_relative_change": result.get(
                    "boundary_quadrature_final_relative_change"
                ),
                "AD_FD_comparison": result.get("AD_FD_comparison"),
                "gates": result.get("gates"),
            },
            indent=2,
        )
    )
    return 0 if result.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
