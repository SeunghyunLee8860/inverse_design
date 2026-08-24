from __future__ import annotations

import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_4um_model,
    fdtdx_increment_state_exact_binary_control as control,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_increment_state_material import (
    physical_increment_material_data,
)


class _Drude:
    def __init__(self, plasma_frequency: float, damping: float):
        self.kind = "Drude"
        self.omega_0 = 0.0
        self.gamma = damping
        self.coupling_sq = plasma_frequency**2


class _Lorentz:
    def __init__(
        self,
        resonance_frequency: float,
        damping: float,
        delta_epsilon: float,
    ):
        self.kind = "Lorentz"
        self.omega_0 = resonance_frequency
        self.gamma = damping
        self.coupling_sq = delta_epsilon * resonance_frequency**2


def _coefficient_function(poles, dt):
    columns = [np.zeros((len(poles), 3), dtype=np.float64) for _ in range(3)]
    for index, pole in enumerate(poles):
        denominator = 1.0 + 0.5 * pole.gamma * dt
        values = (
            (1.0 - 0.5 * pole.gamma * dt) / denominator,
            pole.omega_0**2 * dt**2 / denominator,
            pole.coupling_sq * dt**2 / denominator,
        )
        for destination, value in zip(columns, values, strict=True):
            destination[index] = value
    return tuple(columns)


def _susceptibility_function(coeff_a, coeff_c, coeff_b, omega, dt):
    z_minus = np.exp(-1j * omega * dt)
    denominator = (
        (z_minus - 1.0) * (z_minus - np.asarray(coeff_a))
        + np.asarray(coeff_c) * z_minus
    )
    return np.sum(np.asarray(coeff_b) * z_minus / denominator)


def test_physical_increment_material_data_uses_one_passive_pole(monkeypatch):
    fake_fdtdx = SimpleNamespace(DrudePole=_Drude, LorentzPole=_Lorentz)
    increment_module = ModuleType("fdtdx.increment_state")
    increment_module.compute_increment_state_coefficients_per_axis = (
        _coefficient_function
    )
    increment_module.susceptibility_from_increment_coefficients = (
        _susceptibility_function
    )
    parent_module = ModuleType("fdtdx")
    parent_module.increment_state = increment_module
    monkeypatch.setitem(sys.modules, "fdtdx", parent_module)
    monkeypatch.setitem(sys.modules, "fdtdx.increment_state", increment_module)

    wavelength = 4.0e-6
    omega = 2.0 * math.pi * 299_792_458.0 / wavelength
    result = physical_increment_material_data(
        fake_fdtdx,
        dt_s=1.6678204759907602e-17,
        omega_rad_s=omega,
        wavelength_m=wavelength,
        epsilon_au=complex(-830.37, 127.16),
        epsilon_ta={
            "a": complex(-30.713256371885343, 50.848086107787424),
            "b": complex(15.900726644538812, 9.289194887622557),
            "c": complex(15.900726644538812, 9.289194887622557),
        },
    )

    assert result["optimizer_start_allowed"] is False
    assert result["gray_material_law_defined"] is False
    assert all(len(value) == 1 for value in result["poles"].values())
    assert result["coefficient_endpoints"]["au"][0][1] == 0.0
    assert result["coefficient_endpoints"]["a"][0][1] == 0.0
    assert result["coefficient_endpoints"]["b"][0][1] > 0.0
    assert all(
        fit["fit_relative_error"] < 1.0e-4
        for fit in result["fits"].values()
    )


def test_builder_rejects_unknown_or_two_pole_increment_state_before_import():
    with pytest.raises(ValueError, match="unknown dispersive"):
        fdtdx_4um_model.build_model(
            "Ea", dispersive_state_representation="not-a-state"
        )
    with pytest.raises(ValueError, match="two-pole"):
        fdtdx_4um_model.build_model(
            "Ea",
            include_adjoint_source=False,
            material_law_contract={"candidate": True},
            dispersive_state_representation="increment",
        )


def test_control_removes_unavailable_source_normalization(monkeypatch):
    synthetic = {
        "flux": {
            "source_reference_all_air_unscaled_W": 1.0,
            "absorbed_fraction_of_all_air_source": 99.0,
            "Q_vs_closed_phasor_symmetric_relative": 0.0,
        },
        "gates": {
            "absorbed_fraction_physical": False,
            "Q_closed_phasor_closure": True,
        },
        "failed_gates": ["absorbed_fraction_physical"],
        "ready": False,
        "common_285uW_reporting": {"late_total_Q_W": 1.0},
    }
    monkeypatch.setattr(
        control,
        "_power_evaluation",
        lambda *args, **kwargs: (synthetic, {"large_raw": np.ones(1)}),
    )

    result = control._unnormalized_closure_evaluation(
        {}, object(), np.zeros((80, 80), dtype=np.uint8)
    )

    assert result["ready"] is True
    assert result["source_normalization_available"] is False
    assert "absorbed_fraction_physical" not in result["gates"]
    assert "absorbed_fraction_of_all_air_source" not in result["flux"]
    assert "common_285uW_reporting" not in result


def test_gpu_wrapper_rejects_busy_device_before_export():
    wrapper = Path(control.__file__).with_name(
        "run_fdtdx_increment_state_control_gpu.sh"
    ).read_text(encoding="utf-8")
    busy_check = wrapper.index("refusing busy GPU")
    export = wrapper.index("export CUDA_VISIBLE_DEVICES")
    assert busy_check < export
    assert "--query-compute-apps" in wrapper
    assert "Lumerical" not in wrapper
