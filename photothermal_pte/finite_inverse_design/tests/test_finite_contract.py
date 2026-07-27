from __future__ import annotations

import numpy as np

from photothermal_pte.finite_inverse_design import contract


def test_optical_roi_is_exactly_the_physical_two_micron_square():
    assert contract.OPTICAL_ROI_BOUNDS_M == {
        "x": (-1.0e-6, 1.0e-6),
        "y": (-1.0e-6, 1.0e-6),
    }
    assert contract.DESIGN_BOUNDS_M["x"] == contract.OPTICAL_ROI_BOUNDS_M["x"]
    assert contract.DESIGN_BOUNDS_M["y"] == contract.OPTICAL_ROI_BOUNDS_M["y"]
    for span_m in contract.LARGE_FLAKE_SPAN_CANDIDATES_M:
        flake = contract.flake_bounds_m(span_m)
        assert flake["x"][0] < contract.OPTICAL_ROI_BOUNDS_M["x"][0]
        assert flake["x"][1] > contract.OPTICAL_ROI_BOUNDS_M["x"][1]
        assert flake["y"][0] < contract.OPTICAL_ROI_BOUNDS_M["y"][0]
        assert flake["y"][1] > contract.OPTICAL_ROI_BOUNDS_M["y"][1]
        assert flake["z"] == contract.FLAKE_BOUNDS_Z_M


def test_domain_and_pml_candidates_change_only_padding():
    roi_span = (
        contract.OPTICAL_ROI_BOUNDS_M["x"][1]
        - contract.OPTICAL_ROI_BOUNDS_M["x"][0]
    )
    assert np.isclose(roi_span, 2.0e-6, rtol=0.0, atol=1.0e-18)
    assert all(
        domain_span > roi_span
        for domain_span in contract.OPTICAL_DOMAIN_SPAN_CANDIDATES_M
    )
    assert contract.PML_LAYER_CANDIDATES == (16, 24, 32)
    assert contract.PML_LAYERS in contract.PML_LAYER_CANDIDATES


def test_roi_stays_inside_every_optical_domain():
    roi_half_span = max(
        abs(value)
        for bounds in contract.OPTICAL_ROI_BOUNDS_M.values()
        for value in bounds
    )
    domain_half_span = 0.5 * min(
        contract.OPTICAL_DOMAIN_SPAN_CANDIDATES_M
    )
    assert roi_half_span < domain_half_span


def test_empty_air_tfsf_pass_does_not_promote_device_source():
    source = contract.OPTICAL_SOURCE.lower()
    assert contract.OPTICAL_SOURCE_STATUS == (
        "VALIDATED_EMPTY_AIR_CPU_TFSF_SOURCE_GATE_DEVICE_PENDING"
    )
    assert "device geometry still pending" in source
    assert "cpu tfsf" in source
    assert "normal-incidence" in source
    assert "gaussian" not in source
    assert contract.INCIDENT_INTENSITY_W_M2 == 1.0
    assert contract.ANALYSIS_WAVELENGTH_M == 4.0e-6
