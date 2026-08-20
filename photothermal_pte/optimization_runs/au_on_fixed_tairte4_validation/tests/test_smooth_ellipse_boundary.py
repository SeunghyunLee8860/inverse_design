from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ELLIPSE = load("smooth_ellipse_control", "16_run_au_smooth_ellipse_width_control.py")
ADJOINT = load(
    "smooth_ellipse_adjoint", "17_run_au_smooth_ellipse_external_field_adjoint.py"
)
PVA_ADJOINT = load(
    "pva_smooth_ellipse_adjoint",
    "19_run_au_pva_smooth_ellipse_external_field_adjoint.py",
)


def test_ellipse_vertices_are_counter_clockwise_and_have_exact_bounds() -> None:
    half_x = 8.0e-6
    half_y = 10.0e-6
    vertices = ELLIPSE.ellipse_vertices(half_x, half_y, 512)
    assert vertices.shape == (512, 2)
    np.testing.assert_allclose(np.max(np.abs(vertices[:, 0])), half_x)
    np.testing.assert_allclose(np.max(np.abs(vertices[:, 1])), half_y)
    signed_twice_area = np.sum(
        vertices[:, 0] * np.roll(vertices[:, 1], -1)
        - vertices[:, 1] * np.roll(vertices[:, 0], -1)
    )
    assert signed_twice_area > 0.0


def test_shape_velocity_integral_recovers_polygon_area_derivative() -> None:
    quadrature = ADJOINT.ellipse_boundary_quadrature(
        half_width_m=8.0e-6, gauss_order_per_edge=4
    )
    numerical = np.sum(
        quadrature["normal_velocity_m_per_m"]
        * quadrature["arc_weights_m"]
    )
    # A regular 512-vertex polygon has A=a*b*N*sin(2*pi/N)/2.
    exact_polygon_derivative = (
        ADJOINT.ELLIPSE_HALF_Y_M
        * ADJOINT.ELLIPSE_VERTICES
        * np.sin(2.0 * np.pi / ADJOINT.ELLIPSE_VERTICES)
        / 2.0
    )
    np.testing.assert_allclose(numerical, exact_polygon_derivative, rtol=1.0e-13)
    np.testing.assert_allclose(numerical, np.pi * ADJOINT.ELLIPSE_HALF_Y_M, rtol=3.0e-5)


def test_quadrature_excludes_polygon_vertices() -> None:
    quadrature = ADJOINT.ellipse_boundary_quadrature(
        half_width_m=8.0e-6, gauss_order_per_edge=8
    )
    points = quadrature["points_xy"].reshape(-1, 2)
    vertices = ELLIPSE.ellipse_vertices(8.0e-6, ADJOINT.ELLIPSE_HALF_Y_M, 512)
    # Gauss-Legendre nodes lie strictly inside every edge.
    minimum_distance = np.min(
        np.linalg.norm(points[:, None, :] - vertices[None, :, :], axis=-1)
    )
    assert minimum_distance > 0.0


def test_full_depth_quadrature_order_tracks_requested_dz() -> None:
    class ConstantFields:
        @staticmethod
        def getfield(x, y, z, wavelength):
            del y, z, wavelength
            return np.ones((np.asarray(x).size, 3), complex)

        @staticmethod
        def getDfield(x, y, z, wavelength):
            del y, z, wavelength
            return np.ones((np.asarray(x).size, 3), complex)

    result = ADJOINT.midpoint_ellipse_integral(
        ConstantFields(),
        ConstantFields(),
        half_width_m=8.0e-6,
        half_y_m=10.0e-6,
        epsilon_au=complex(-100.0, 10.0),
        dy_m=25.0e-9,
        dz_m=5.0e-9,
    )
    assert result["gauss_order_per_edge"] == 4
    assert result["gauss_order_z"] == 10
    assert result["sample_count"] == 512 * 4 * 10
    assert result["z_endpoints_sampled"] is False


def test_pva_case_family_is_symmetric_about_eight_microns() -> None:
    assert PVA_ADJOINT.ELLIPSE_HALF_Y_M == 18.0e-6
    assert PVA_ADJOINT.BASELINE_CASE.endswith("a8p0_b18_edge50_forward_retry")
    assert PVA_ADJOINT.PVA_FD_CASES[0.10] == (
        "pva5_smooth_ellipse_a7p9_b18_edge50_forward",
        "pva5_smooth_ellipse_a8p1_b18_edge50_forward",
    )
    assert PVA_ADJOINT.PVA_FD_CASES[0.05] == (
        "pva5_smooth_ellipse_a7p95_b18_edge50_forward",
        "pva5_smooth_ellipse_a8p05_b18_edge50_forward",
    )
