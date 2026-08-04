import numpy as np
import pytest

from photothermal_pte.finite_inverse_design.large_background_contract import (
    Bounds3D,
    baseline_contract,
    infer_realized_pml_bounds,
)


def test_baseline_geometry_gaps_and_fixed_omega() -> None:
    contract = baseline_contract()
    audit = contract.geometry_audit_input()
    assert audit["periodic_or_bloch_boundary"] is False
    assert audit["finite_optical_flake"] is False
    assert np.isclose(
        audit["design_to_tfsf_gaps_m"]["x"]["min_m"], 0.3e-6
    )
    assert np.isclose(
        audit["tfsf_to_fdtd_outer_gaps_m"]["x"]["min_m"], 1.9e-6
    )
    assert np.isclose(
        audit["omega_q_to_tfsf_gaps_m"]["x"]["min_m"], 0.15e-6
    )
    assert audit["omega_q_and_six_face_bounds_m"]["x"] == pytest.approx(
        [-1.15e-6, 1.15e-6]
    )
    assert audit["omega_q_and_six_face_bounds_m"]["y"] == pytest.approx(
        [-1.15e-6, 1.15e-6]
    )
    assert audit["omega_q_and_six_face_bounds_m"]["z"] == pytest.approx(
        [-150e-9, 750e-9]
    )


def test_tfsf_cannot_touch_pml_inner() -> None:
    with pytest.raises(ValueError, match="TFSF"):
        baseline_contract(lateral_domain_um=2.6, tfsf_span_um=2.6)


def test_realized_mesh_layer_audit() -> None:
    outer = Bounds3D((-4.4, 4.4), (-4.4, 4.4), (-5.4, 5.4))
    coordinates = {
        "x": np.linspace(-4.4, 4.4, 89),
        "y": np.linspace(-4.4, 4.4, 89),
        "z": np.linspace(-5.4, 5.4, 109),
    }
    audit = infer_realized_pml_bounds(coordinates, outer, pml_layers=24)
    assert audit["all_axes_fdtd_outer_mesh_readback_matches"]
    assert np.allclose(
        audit["x"]["pml_inner_from_layer_count_m"], [-2.0, 2.0]
    )
    assert audit["x"]["mesh_point_outer_bounds_m"] == pytest.approx(
        [-4.4, 4.4]
    )
