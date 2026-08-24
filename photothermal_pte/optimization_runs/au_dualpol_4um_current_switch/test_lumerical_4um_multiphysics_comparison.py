from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.finite_inverse_design.native_yee_q import trapezoid_weights
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_multiphysics_comparison import (
    downstream_metrics,
    map_lumerical_material_q_to_thermal,
    map_lumerical_q_to_thermal,
    thermal_cell_volumes,
    volume_l2_nrmse,
)


def _raw(scale: float = 1.0) -> dict[str, np.ndarray]:
    coordinate = np.asarray((-0.5, 0.5), dtype=np.float64)
    result: dict[str, np.ndarray] = {}
    for component in "xyz":
        result[f"Q{component}_W_m3"] = np.full((2, 2, 2), scale)
        for axis in "xyz":
            result[f"Q{component}_{axis}_m"] = coordinate.copy()
    return result


def test_native_q_remap_is_conservative_and_linear() -> None:
    edges = tuple(np.asarray((-1.0, 0.0, 1.0)) for _ in range(3))
    mapped, audit = map_lumerical_q_to_thermal(_raw(), edges, 2.5)
    width = trapezoid_weights(np.asarray((-0.5, 0.5)))
    expected = 3.0 * np.sum(width) ** 3 * 2.5
    assert audit["native_total_power_W"] == pytest.approx(expected)
    assert np.sum(mapped) == pytest.approx(expected)
    assert audit["relative_conservation_error"] < 1e-14
    doubled, _ = map_lumerical_q_to_thermal(_raw(2.0), edges, 2.5)
    assert np.array_equal(doubled, 2.0 * mapped)


def test_material_remap_keeps_loss_out_of_thermal_air() -> None:
    coordinate = np.asarray((-0.075, -0.025, 0.025)) * 1.0e-6
    lateral = np.asarray((-0.05, 0.05)) * 1.0e-6
    raw: dict[str, np.ndarray] = {}
    for component in "xyz":
        raw[f"Q{component}_W_m3"] = np.ones((2, 2, 3))
        raw[f"epsilon_{component}"] = 1j * np.ones((2, 2, 3))
        raw[f"Q{component}_x_m"] = lateral.copy()
        raw[f"Q{component}_y_m"] = lateral.copy()
        raw[f"Q{component}_z_m"] = coordinate.copy()
    edges = (
        np.asarray((-0.1, 0.0, 0.1)) * 1.0e-6,
        np.asarray((-0.1, 0.0, 0.1)) * 1.0e-6,
        np.asarray((-0.1, -0.05, 0.0, 0.05, 0.1)) * 1.0e-6,
    )
    mapped, audit = map_lumerical_material_q_to_thermal(
        raw,
        edges,
        1.0,
        case="empty",
        material_imaginary_epsilon={
            "TaIrTe4": {axis: 1.0 for axis in "xyz"},
            "SiO2": {axis: 0.0 for axis in "xyz"},
        },
    )
    assert np.all(mapped[:, :, 2:] == 0.0)
    assert audit["relative_conservation_error"] < 1.0e-14


def test_volume_l2_uses_power_density_and_finer_norm() -> None:
    volume = np.asarray((1.0, 2.0))
    fine = np.asarray((2.0, 4.0))
    coarse = 1.1 * fine
    assert volume_l2_nrmse(coarse, fine, volume) == pytest.approx(0.1)


def test_downstream_gate_requires_current_sign_for_asymmetric_case() -> None:
    power = np.ones((2, 2, 2))
    volume = np.ones_like(power)
    temperature = np.ones((2, 2))
    metrics, gates = downstream_metrics(
        coarse_power_W=power,
        fine_power_W=power,
        cell_volume_m3=volume,
        coarse_ta_temperature_K=temperature,
        fine_ta_temperature_K=temperature,
        coarse_tmax_K=1.0,
        fine_tmax_K=1.0,
        coarse_current_A=-1.0,
        fine_current_A=1.0,
        coarse_current_absolute_scale_A=1.0,
        fine_current_absolute_scale_A=1.0,
        expect_zero_current=False,
    )
    assert all(
        value == pytest.approx(0.0)
        for key, value in metrics.items()
        if key != "signed_current_change_relative"
    )
    assert gates["signed_current_sign_preserved"] is False
    assert gates["signed_current_change_relative"] is False


def test_symmetric_zero_current_uses_cancellation_not_relative_change() -> None:
    power = np.ones((2, 2, 2))
    volume = np.ones_like(power)
    temperature = np.ones((2, 2))
    metrics, gates = downstream_metrics(
        coarse_power_W=power,
        fine_power_W=power,
        cell_volume_m3=volume,
        coarse_ta_temperature_K=temperature,
        fine_ta_temperature_K=temperature,
        coarse_tmax_K=1.0,
        fine_tmax_K=1.0,
        coarse_current_A=3.0e-17,
        fine_current_A=1.0e-17,
        coarse_current_absolute_scale_A=2.0e-7,
        fine_current_absolute_scale_A=2.0e-7,
        expect_zero_current=True,
    )
    assert metrics["symmetric_current_cancellation_max_relative"] == pytest.approx(
        1.5e-10
    )
    assert gates["symmetric_current_cancellation_lt_1ppm"] is True
    assert "signed_current_change_relative" not in metrics
