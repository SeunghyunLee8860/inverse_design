from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_density import (
    density_nodes,
    density_state_sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_yee_jacobian import (
    build_colored_material_jacobian,
    transpose_dot_error,
    validate_completed_density_record,
    validate_index_detail,
    validate_material_jacobian,
)


def _detail_from_density(rho: np.ndarray, *, nonlocal_response: bool = False):
    x, y, _ = density_nodes()
    z = np.asarray([-5.0e-9, 55.0e-9])
    value = np.asarray(rho, float)
    local = 1.0 + 2.0 * value + 0.3 * value**2
    coupled_x = local.copy()
    coupled_x[1:, :] += 0.2 * value[:-1, :]
    coupled_y = local.copy()
    coupled_y[:, 1:] -= 0.15 * value[:, :-1]
    if nonlocal_response:
        coupled_x[-1, -1] += 0.01 * value[0, 0]
    result = {
        "x": x,
        "x_offset": x + 20.0e-9,
        "y": y,
        "y_offset": y + 25.0e-9,
        "z": z,
        "z_offset": z + 2.0e-9,
        "frequency_hz": np.asarray([299_792_458.0 / CONTRACT.wavelength_m]),
    }
    for component, plane in zip(
        "xyz", (coupled_x, coupled_y, 0.8 * local), strict=True
    ):
        result[f"epsilon_{component}"] = np.repeat(
            ((1.0 + 0.4j) * plane)[:, :, None], z.size, axis=2
        )
    return result


def _nonuniform_density() -> np.ndarray:
    x, y, _ = density_nodes()
    xn = x[:, None] / (0.5 * CONTRACT.design_span_x_m)
    yn = y[None, :] / (0.5 * CONTRACT.design_span_y_m)
    rho = 0.5 + 0.18 * np.sin(0.7 * np.pi * xn) * np.cos(0.6 * np.pi * yn)
    rho[0, 0] = 0.0
    rho[-1, -1] = 1.0
    return rho


def test_colored_complex_yee_operator_matches_independent_mapping_fd() -> None:
    rho = _nonuniform_density()
    operator, metadata, _ = build_colored_material_jacobian(
        _detail_from_density, rho
    )
    assert metadata["Maxwell_solves"] == 0
    assert metadata["color_count"] == 25
    assert metadata["lower_endpoint_node_count"] == 1
    assert metadata["upper_endpoint_node_count"] == 1
    assert metadata["baseline_roundtrip_epsilon_max_abs_error"] == 0.0
    assert all(matrix.nnz > 0 for matrix in operator.matrices.values())

    rng = np.random.default_rng(260824)
    direction = rng.normal(size=rho.shape)
    direction /= np.max(np.abs(direction))
    direction[(rho == 0.0) | (rho == 1.0)] = 0.0
    step = 1.0e-6
    plus = _detail_from_density(rho + step * direction)
    minus = _detail_from_density(rho - step * direction)
    finite_difference = {
        component: (
            plus[f"epsilon_{component}"] - minus[f"epsilon_{component}"]
        )
        / (2.0 * step)
        for component in "xyz"
    }
    tangent = operator.jvp(direction)
    error = np.sqrt(
        sum(
            np.linalg.norm(tangent[component] - finite_difference[component])
            ** 2
            for component in "xyz"
        )
    )
    scale = np.sqrt(
        sum(np.linalg.norm(finite_difference[component]) ** 2 for component in "xyz")
    )
    assert error / scale < 5.0e-10


def test_colored_complex_yee_operator_has_exact_real_design_transpose() -> None:
    rho = _nonuniform_density()
    operator, _, _ = build_colored_material_jacobian(_detail_from_density, rho)
    rng = np.random.default_rng(42)
    direction = rng.normal(size=rho.shape)
    cotangent = {
        component: (
            rng.normal(size=operator.component_shapes[component])
            + 1j * rng.normal(size=operator.component_shapes[component])
        )
        for component in "xyz"
    }
    audit = transpose_dot_error(operator, direction, cotangent)
    assert audit["relative_error"] < 2.0e-14


def test_independent_mapping_audit_covers_interior_and_endpoint_directions() -> None:
    rho = _nonuniform_density()
    operator, _, _ = build_colored_material_jacobian(_detail_from_density, rho)
    audit = validate_material_jacobian(_detail_from_density, rho, operator)
    assert audit["passed"] is True
    assert audit["gates"]["baseline_layout_roundtrip_exact"] is True
    assert audit["directions"]["lower_endpoint_feasible"]["scheme"] == "forward"
    assert audit["directions"]["upper_endpoint_feasible"]["scheme"] == "forward"
    assert audit["worst_transpose_dot_relative_error"] < 1.0e-12


def test_coloring_fails_closed_when_material_response_is_nonlocal() -> None:
    rho = _nonuniform_density()

    def nonlocal_evaluator(value):
        return _detail_from_density(value, nonlocal_response=True)

    with pytest.raises(RuntimeError, match="nonlocal"):
        build_colored_material_jacobian(nonlocal_evaluator, rho)


def test_coloring_fails_closed_if_density_perturbation_changes_yee_grid() -> None:
    rho = _nonuniform_density()
    baseline_sum = float(np.sum(rho))

    def moving_grid(value):
        detail = _detail_from_density(value)
        if float(np.sum(value)) != baseline_sum:
            detail["x_offset"] = detail["x_offset"] + 1.0e-12
        return detail

    with pytest.raises(RuntimeError, match="changed the frozen Yee grid"):
        build_colored_material_jacobian(moving_grid, rho)


def test_index_detail_rejects_wrong_component_shape_or_frequency() -> None:
    detail = _detail_from_density(_nonuniform_density())
    broken_shape = dict(detail)
    broken_shape["epsilon_y"] = broken_shape["epsilon_y"][:-1]
    with pytest.raises(ValueError, match="epsilon_y shape"):
        validate_index_detail(broken_shape)
    broken_frequency = dict(detail)
    broken_frequency["frequency_hz"] = np.asarray([1.0, 2.0])
    with pytest.raises(ValueError, match="one finite positive frequency"):
        validate_index_detail(broken_frequency)


def test_completed_forward_record_is_bound_to_density_and_fsp_hash() -> None:
    rho = _nonuniform_density()
    record = {
        "status": "PASSED_PROVISIONAL_LUMERICAL_4UM_import_density_deadbeef_Ea_"
        "CONTROL_DEVELOPMENT_GPU_NOT_B200_CERTIFIED",
        "case": "import_density",
        "all_gates_passed": True,
        "accelerator_policy": "development",
        "solver_version": "8.35.4413",
        "Q_processing": {
            "clipping": False,
            "smoothing": False,
            "gain": False,
            "field_or_Q_rescaling": False,
            "global_rescaling": False,
            "tiling": False,
        },
        "layout": {
            "geometry": {
                "density_state": {
                    "density_state_sha256": density_state_sha256(rho)
                }
            }
        },
        "raw_artifacts": [
            {"path": "/external/forward.fsp", "sha256": "fsp-sha"}
        ],
    }
    assert validate_completed_density_record(
        record, rho, forward_fsp_sha256="fsp-sha"
    )["passed"]
    changed = rho.copy()
    changed[40, 40] += 1.0e-6
    invalid = validate_completed_density_record(
        record, changed, forward_fsp_sha256="fsp-sha"
    )
    assert invalid["passed"] is False
    assert invalid["gates"]["density_state_sha_matches"] is False
