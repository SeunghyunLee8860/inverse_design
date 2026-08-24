from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_maxwell_contract import (
    CONTRACT,
    binary_mask_sha256,
    canonical_binary_mask,
    exact_au_geometry_audit,
    exact_au_geometry_sha256,
)


def test_contract_preserves_lumerical_plus_custom_gpu_pde_architecture() -> None:
    payload = asdict(CONTRACT)
    assert payload["maxwell_solver"].startswith("Ansys Lumerical FDTD")
    assert payload["maxwell_accelerator_required"] == "NVIDIA B200"
    assert "custom CUDA" in payload["thermal_solver"]
    assert "custom CUDA" in payload["electrical_solver"]
    assert payload["continuous_geometry_parameters_allowed"] is True
    assert payload["gray_au_material_in_maxwell_allowed"] is False
    assert payload["gray_au_material_in_thermal_allowed"] is False
    assert payload["gray_au_material_in_electrical_allowed"] is False
    assert payload["exact_binary_required_for_every_physics_evaluation"] is True
    assert payload["numerical_interface_cut_cells_allowed"] is True
    assert payload["different_optical_thermal_electrical_design_fields_allowed"] is False
    assert payload["exact_binary_required_for_final_promotion"] is True
    assert payload["exact_dispersive_au_required_in_every_maxwell_evaluation"] is True
    assert payload["np_density_as_au_topology_variable_allowed"] is False
    assert payload["bundled_lumopt_topology_gradient_allowed_without_au_adfd"] is False
    assert payload["fdtdx_allowed"] is False
    assert payload["jax_maxwell_allowed"] is False


def test_binary_mask_hash_is_shape_and_layout_sensitive() -> None:
    mask = np.asarray([[0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    assert binary_mask_sha256(mask) == binary_mask_sha256(mask.astype(float))
    assert binary_mask_sha256(mask) != binary_mask_sha256(mask.T)
    changed = mask.copy()
    changed[0, 0] = 1
    assert binary_mask_sha256(mask) != binary_mask_sha256(changed)


def test_physical_geometry_hash_binds_scale_origin_thickness_and_axes() -> None:
    mask = np.asarray([[0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    x = np.asarray([-1.0, 0.0, 1.0]) * 1.0e-6
    y = np.asarray([-1.5, -0.5, 0.5, 1.5]) * 1.0e-6
    z = np.asarray([0.0, 50.0e-9])
    baseline = exact_au_geometry_sha256(
        mask, x_edges_m=x, y_edges_m=y, z_bounds_m=z
    )
    assert baseline != exact_au_geometry_sha256(
        mask, x_edges_m=x + 0.1e-6, y_edges_m=y, z_bounds_m=z
    )
    assert baseline != exact_au_geometry_sha256(
        mask, x_edges_m=2.0 * x, y_edges_m=y, z_bounds_m=z
    )
    assert baseline != exact_au_geometry_sha256(
        mask, x_edges_m=x, y_edges_m=y, z_bounds_m=[0.0, 60.0e-9]
    )
    with pytest.raises(ValueError, match="x=b"):
        exact_au_geometry_sha256(
            mask,
            x_edges_m=x,
            y_edges_m=y,
            z_bounds_m=z,
            axis_x="a",
            axis_y="b",
        )
    audit = exact_au_geometry_audit(
        mask, x_edges_m=x, y_edges_m=y, z_bounds_m=z
    )
    assert audit["geometry_sha256"] == baseline
    assert audit["mask_payload_sha256"] == binary_mask_sha256(mask)
    assert audit["occupied_cell_count"] == 4


@pytest.mark.parametrize(
    ("x_edges", "y_edges", "z_bounds"),
    [
        ([0.0, 1.0], [0.0, 1.0, 2.0, 3.0], [0.0, 1.0]),
        ([0.0, 1.0, 2.0], [0.0, 1.0, 1.0, 3.0], [0.0, 1.0]),
        ([0.0, 1.0, 2.0], [0.0, 1.0, 2.0, 3.0], [1.0, 0.0]),
    ],
)
def test_physical_geometry_rejects_bad_coordinate_contract(
    x_edges: list[float], y_edges: list[float], z_bounds: list[float]
) -> None:
    with pytest.raises(ValueError):
        exact_au_geometry_sha256(
            np.zeros((2, 3), dtype=np.uint8),
            x_edges_m=x_edges,
            y_edges_m=y_edges,
            z_bounds_m=z_bounds,
        )


@pytest.mark.parametrize(
    "bad",
    [
        np.asarray([0, 1]),
        np.asarray([[0.0, 0.5]]),
        np.asarray([[0.0, np.nan]]),
        np.asarray([["0", "1"]]),
        np.empty((0, 2)),
    ],
)
def test_binary_mask_rejects_nonphysical_inputs(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        canonical_binary_mask(bad)
