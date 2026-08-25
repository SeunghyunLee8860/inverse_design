from __future__ import annotations

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.pde_mesh_convergence_4um import (
    fine_to_coarse_cell_average,
    pde_mesh_convergence_audit,
)


def _fields() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.2, 2.0, 6)
    coarse = x[:, None] + 0.3 * x[None, :]
    fine = np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)
    return coarse, fine


def test_aligned_pde_convergence_passes_below_half_percent() -> None:
    coarse, fine = _fields()
    audit = pde_mesh_convergence_audit(
        coarse_current_A=1.004e-9,
        fine_current_A=1.0e-9,
        coarse_ta_temperature_K=coarse,
        fine_ta_temperature_K=fine,
        coarse_peak_temperature_K=4.016,
        fine_peak_temperature_K=4.0,
    )
    assert audit["passed"] is True
    assert audit["metrics"]["current_relative_change"] == pytest.approx(0.004)
    assert audit["metrics"]["ta_temperature_field_nrmse"] == 0.0


def test_pde_convergence_fails_current_sign_or_relative_gate() -> None:
    coarse, fine = _fields()
    relative = pde_mesh_convergence_audit(
        coarse_current_A=1.01e-9,
        fine_current_A=1.0e-9,
        coarse_ta_temperature_K=coarse,
        fine_ta_temperature_K=fine,
        coarse_peak_temperature_K=4.0,
        fine_peak_temperature_K=4.0,
    )
    assert relative["passed"] is False
    assert relative["gates"]["current_relative_change_lt_0p5pct"] is False

    sign = pde_mesh_convergence_audit(
        coarse_current_A=-1.0e-9,
        fine_current_A=1.0e-9,
        coarse_ta_temperature_K=coarse,
        fine_ta_temperature_K=fine,
        coarse_peak_temperature_K=4.0,
        fine_peak_temperature_K=4.0,
    )
    assert sign["passed"] is False
    assert sign["gates"]["current_sign_preserved"] is False


def test_pde_convergence_fails_nonconverged_temperature_field() -> None:
    coarse, fine = _fields()
    changed = coarse.copy()
    changed[2, 3] += 0.2
    audit = pde_mesh_convergence_audit(
        coarse_current_A=1.0e-9,
        fine_current_A=1.0e-9,
        coarse_ta_temperature_K=changed,
        fine_ta_temperature_K=fine,
        coarse_peak_temperature_K=4.0,
        fine_peak_temperature_K=4.0,
    )
    assert audit["passed"] is False
    assert audit["gates"]["ta_temperature_field_nrmse_lt_0p5pct"] is False


def test_pde_comparison_rejects_non_aligned_or_nonfinite_fields() -> None:
    coarse, fine = _fields()
    with pytest.raises(ValueError, match="exact 2x2"):
        fine_to_coarse_cell_average(fine[::2], coarse_shape=coarse.shape)
    fine[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fine_to_coarse_cell_average(fine, coarse_shape=coarse.shape)
