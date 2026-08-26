from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_adfd import (
    SMOOTH_DIRECTION_COEFFICIENTS,
    array_sha256,
    centered_adfd_metrics,
    centered_density_pair,
    centered_pair_reconstruction_metrics,
    independent_smooth_direction,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_design_mapping import (
    OPTIMIZER_250NM_MAPPING,
)


_COMPARATOR_PATH = Path(__file__).with_name(
    "36_compare_lumerical_4um_ea_combined_adfd.py"
)
_COMPARATOR_SPEC = importlib.util.spec_from_file_location(
    "au_dualpol_4um_combined_adfd_comparator_test", _COMPARATOR_PATH
)
assert _COMPARATOR_SPEC is not None and _COMPARATOR_SPEC.loader is not None
_COMPARATOR = importlib.util.module_from_spec(_COMPARATOR_SPEC)
_COMPARATOR_SPEC.loader.exec_module(_COMPARATOR)


def test_pair_contract_binds_ea_and_eb_status_to_polarization() -> None:
    legacy_ea = _COMPARATOR._pair_contract(
        {"status": "PREPARED_LUMERICAL_4UM_EA_LATENT_COMBINED_ADFD_PAIR"}
    )
    assert legacy_ea["polarization"] == "Ea"
    assert legacy_ea["is_latent"] is True
    assert legacy_ea["validated_status"] == (
        "VALIDATED_LUMERICAL_4UM_EA_LATENT_COMBINED_ADFD"
    )

    eb = _COMPARATOR._pair_contract(
        {
            "status": "PREPARED_LUMERICAL_4UM_EB_LATENT_COMBINED_ADFD_PAIR",
            "polarization": "Eb",
        }
    )
    assert eb["polarization"] == "Eb"
    assert eb["validated_status"] == (
        "VALIDATED_LUMERICAL_4UM_EB_LATENT_COMBINED_ADFD"
    )

    with pytest.raises(RuntimeError, match="polarization/status mismatch"):
        _COMPARATOR._pair_contract(
            {
                "status": "PREPARED_LUMERICAL_4UM_EB_LATENT_COMBINED_ADFD_PAIR",
                "polarization": "Ea",
            }
        )


def test_independent_direction_is_deterministic_smooth_and_normalized() -> None:
    first = independent_smooth_direction((81, 81))
    second = independent_smooth_direction((81, 81))
    assert np.array_equal(first, second)
    assert np.max(np.abs(first)) == 1.0
    assert np.max(np.abs(np.diff(first, axis=0))) < 0.1
    assert np.max(np.abs(np.diff(first, axis=1))) < 0.1
    assert array_sha256(first, label="adfd-latent-direction-v1") == (
        "44f111f4a7669b2e0d42f8bd1978f9489d0a48f8973774d7b361445293b6c280"
    )


def test_smooth_direction_family_is_low_frequency_and_independent() -> None:
    directions = [
        independent_smooth_direction((81, 81), index)
        for index in range(len(SMOOTH_DIRECTION_COEFFICIENTS))
    ]
    for direction in directions:
        assert np.max(np.abs(direction)) == 1.0
        assert np.max(np.abs(np.diff(direction, axis=0))) < 0.15
        assert np.max(np.abs(np.diff(direction, axis=1))) < 0.15
    normalized = [direction.ravel() / np.linalg.norm(direction) for direction in directions]
    gram = np.asarray([[np.vdot(left, right) for right in normalized] for left in normalized])
    np.testing.assert_allclose(np.diag(gram), 1.0, rtol=0.0, atol=5.0e-16)
    off_diagonal = gram - np.diag(np.diag(gram))
    assert np.max(np.abs(off_diagonal)) < 0.05
    with pytest.raises(ValueError, match="direction index"):
        independent_smooth_direction((81, 81), len(directions))


def test_centered_density_pair_is_exact_and_feasible() -> None:
    baseline = np.full((11, 9), 0.5)
    direction, plus, minus = centered_density_pair(baseline, step=0.0025)
    assert np.allclose(0.5 * (plus + minus), baseline, rtol=0.0, atol=1.0e-16)
    assert np.allclose((plus - minus) / 0.005, direction, rtol=0.0, atol=2.0e-14)
    with pytest.raises(ValueError, match="leaves"):
        centered_density_pair(np.zeros((11, 9)), step=0.0025)


def test_pair_reconstruction_uses_step_scaled_float64_roundoff() -> None:
    x = np.linspace(0.29, 0.71, 81, dtype=np.float64)
    baseline = np.broadcast_to(x[:, None], (81, 81)).copy()
    direction, plus, minus = centered_density_pair(baseline, step=0.0025)
    metrics = centered_pair_reconstruction_metrics(
        baseline=baseline,
        direction=direction,
        plus=plus,
        minus=minus,
        step=0.0025,
    )
    assert metrics["within_float64_roundoff"] is True
    assert metrics["midpoint_max_abs_error"] <= metrics[
        "midpoint_float64_roundoff_tolerance"
    ]
    assert metrics["direction_max_abs_error"] <= metrics[
        "direction_float64_roundoff_tolerance"
    ]


def test_centered_metrics_recovers_quadratic_directional_derivative() -> None:
    direction = independent_smooth_direction((9, 7))
    gradient = np.arange(direction.size, dtype=float).reshape(direction.shape) * 1.0e-12
    derivative = float(np.sum(gradient * direction))
    baseline = -5.0e-9
    step = 0.0025
    curvature = 3.0e-9
    plus = baseline + step * derivative + curvature * step**2
    minus = baseline - step * derivative + curvature * step**2
    metrics = centered_adfd_metrics(
        gradient=gradient,
        direction=direction,
        step=step,
        baseline_current_A=baseline,
        plus_current_A=plus,
        minus_current_A=minus,
    )
    assert metrics["relative_error"] < 2.0e-11
    assert metrics["same_nonzero_sign"] is True
    assert np.isclose(metrics["centered_midpoint_minus_baseline_A"], curvature * step**2)


def test_centered_metrics_accepts_complete_latent_mapping_chain() -> None:
    rng = np.random.default_rng(20260824)
    x = np.linspace(-1.0, 1.0, 81)[:, None]
    y = np.linspace(-1.0, 1.0, 81)[None, :]
    latent = 0.5 + 0.16 * np.sin(0.8 * np.pi * x) * np.cos(0.6 * np.pi * y)
    direction = independent_smooth_direction(latent.shape)
    projected_gradient = rng.standard_normal(latent.shape) * 1.0e-12
    beta = 4.0
    latent_gradient = OPTIMIZER_250NM_MAPPING.vjp(
        latent, projected_gradient, beta
    )
    step = 1.0e-4

    def objective(value: np.ndarray) -> float:
        return float(
            np.vdot(
                projected_gradient,
                OPTIMIZER_250NM_MAPPING.physical(value, beta),
            )
        )

    metrics = centered_adfd_metrics(
        gradient=latent_gradient,
        direction=direction,
        step=step,
        baseline_current_A=objective(latent),
        plus_current_A=objective(latent + step * direction),
        minus_current_A=objective(latent - step * direction),
    )
    projected_contraction = float(
        np.vdot(
            projected_gradient,
            OPTIMIZER_250NM_MAPPING.jvp(latent, direction, beta),
        )
    )
    latent_contraction = float(np.vdot(latent_gradient, direction))
    assert np.isclose(projected_contraction, latent_contraction, rtol=1.0e-13)
    assert metrics["relative_error"] < 1.0e-7
    assert metrics["same_nonzero_sign"] is True
