"""Solver-free controls for the two-active-source CW reconstruction."""

from __future__ import annotations

import numpy as np

from run_production_combined_adfd_smoke import reconstruct_fieldregion_only_cw


def test_reconstruct_fieldregion_only_cw_from_first_and_average() -> None:
    rng = np.random.default_rng(20260806)
    fieldregion = rng.normal(size=(4, 3, 2, 3)) + 1j * rng.normal(
        size=(4, 3, 2, 3)
    )
    gaussian_spectrum = 0.8 - 0.15j
    fieldregion_spectrum = 1.3 + 0.2j
    first = fieldregion * fieldregion_spectrum / gaussian_spectrum
    average = 2.0 * fieldregion * fieldregion_spectrum / (
        gaussian_spectrum + fieldregion_spectrum
    )

    reconstructed, metadata = reconstruct_fieldregion_only_cw(first, average)

    assert np.allclose(reconstructed, fieldregion)
    assert metadata["two_normalization_state_spatial_residual"] < 1.0e-14
    assert metadata["uses_finite_difference_fit"] is False
    assert metadata["empirical_gradient_rescaling"] is False
