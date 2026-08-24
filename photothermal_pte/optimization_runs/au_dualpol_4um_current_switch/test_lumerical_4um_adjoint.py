from __future__ import annotations

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_adjoint import (
    reconstruct_fieldregion_only_cw,
    validate_raw_against_jacobian,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_yee_jacobian import (
    component_coordinates,
    validate_index_detail,
)


def test_fieldregion_only_cw_reconstruction_removes_anchor_spectrum() -> None:
    rng = np.random.default_rng(34)
    desired = rng.normal(size=(4, 3, 2, 1, 3)) + 1j * rng.normal(
        size=(4, 3, 2, 1, 3)
    )
    ratio = 1.23 + 0.08j
    first = desired * ratio
    average = first / ((ratio + 1.0) / 2.0)
    reconstructed, audit = reconstruct_fieldregion_only_cw(first, average)
    assert np.allclose(reconstructed, desired, rtol=2.0e-15, atol=2.0e-15)
    assert audit["two_normalization_state_spatial_residual"] < 1.0e-14
    assert audit["empirical_gradient_rescaling"] is False


def _synthetic_index_detail() -> dict[str, np.ndarray]:
    coordinates = {
        "x": np.array([-1.0e-6, 1.0e-6]),
        "x_offset": np.array([-0.9e-6, 1.1e-6]),
        "y": np.array([-1.2e-6, 0.8e-6]),
        "y_offset": np.array([-1.1e-6, 0.9e-6]),
        "z": np.array([-0.4e-6, 0.1e-6]),
        "z_offset": np.array([-0.35e-6, 0.15e-6]),
        "frequency_hz": np.array([74.9481145e12]),
    }
    epsilon = np.full((2, 2, 2), -10.0 + 1.2j)
    return coordinates | {
        f"epsilon_{component}": epsilon.copy() for component in "xyz"
    }


def _raw_from_detail(detail: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "Qx_x_m": detail["x_offset"].copy(),
        "Qx_y_m": detail["y"].copy(),
        "Qx_z_m": detail["z"].copy(),
        "Qy_x_m": detail["x"].copy(),
        "Qy_y_m": detail["y_offset"].copy(),
        "Qy_z_m": detail["z"].copy(),
        "Qz_x_m": detail["x"].copy(),
        "Qz_y_m": detail["y"].copy(),
        "Qz_z_m": detail["z_offset"].copy(),
    } | {
        f"epsilon_{component}": detail[f"epsilon_{component}"].copy()
        for component in "xyz"
    }


def test_raw_jacobian_grid_accepts_only_sub_attometre_api_roundoff() -> None:
    detail = _synthetic_index_detail()
    metadata = {
        "baseline_index_detail_audit": validate_index_detail(detail),
        "component_coordinates_m": {
            component: component_coordinates(detail, component)
            for component in "xyz"
        },
    }
    raw = _raw_from_detail(detail)
    raw["Qx_x_m"][0] += 1.0e-21
    accepted = validate_raw_against_jacobian(raw, metadata)
    assert accepted["passed"] is True
    assert accepted["coordinate_sha256_identical"] is False

    raw["Qx_x_m"][0] += 3.0e-18
    rejected = validate_raw_against_jacobian(raw, metadata)
    assert rejected["passed"] is False
    assert rejected["gates"]["coordinate_arrays_match_lt_2e_18"] is False
