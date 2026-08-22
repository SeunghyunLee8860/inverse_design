import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.beam_response_contract import (
    AU_THICKNESS_M,
    FLAKE_BOUNDS_M,
    OPTICAL_DOMAIN_SPAN_M,
    RESPONSE_CONTROL_VOLUME_HALF_SPAN_M,
    SOURCE_TO_PML_MINIMUM_CLEARANCE_M,
    domain_center_m,
    electrode_bounds_m,
    source_bounds_m,
    sweep_inputs,
)


def test_full_sweep_has_29_unique_forward_inputs():
    inputs = sweep_inputs()
    assert len(inputs) == 29
    assert len({item["id"] for item in inputs}) == 29
    assert sum(item["kind"] == "waist" for item in inputs) == 5
    assert sum(item["kind"] == "position" for item in inputs) == 24


def test_au_terminal_footprints_never_extend_flake():
    low, high = FLAKE_BOUNDS_M
    for contact_axis in ("x", "y"):
        bounds = electrode_bounds_m(contact_axis)
        assert len(bounds) == 2
        for electrode in bounds:
            assert electrode["z"] == (0.0, AU_THICKNESS_M)
            for axis in "xy":
                assert electrode[axis][0] >= low
                assert electrode[axis][1] <= high


def test_extreme_beam_positions_keep_source_and_fixed_flake_clear_of_pml():
    half_domain = 0.5 * OPTICAL_DOMAIN_SPAN_M
    low, high = FLAKE_BOUNDS_M
    for x_um in (-10.0, 10.0):
        for y_um in (-10.0, 10.0):
            center = domain_center_m(x_um, y_um)
            source = source_bounds_m(x_um, y_um)
            for axis in "xy":
                domain_low = center[axis] - half_domain
                domain_high = center[axis] + half_domain
                assert source[axis][0] - domain_low >= SOURCE_TO_PML_MINIMUM_CLEARANCE_M
                assert domain_high - source[axis][1] >= SOURCE_TO_PML_MINIMUM_CLEARANCE_M
                assert low - domain_low >= SOURCE_TO_PML_MINIMUM_CLEARANCE_M
                assert domain_high - high >= SOURCE_TO_PML_MINIMUM_CLEARANCE_M
                assert (
                    -RESPONSE_CONTROL_VOLUME_HALF_SPAN_M - domain_low
                    >= SOURCE_TO_PML_MINIMUM_CLEARANCE_M
                )
                assert (
                    domain_high - RESPONSE_CONTROL_VOLUME_HALF_SPAN_M
                    >= SOURCE_TO_PML_MINIMUM_CLEARANCE_M
                )
    assert np.allclose(FLAKE_BOUNDS_M, (-12.0e-6, 12.0e-6), rtol=0.0, atol=0.0)
