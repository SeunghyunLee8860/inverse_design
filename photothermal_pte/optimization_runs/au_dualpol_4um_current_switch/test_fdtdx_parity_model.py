from __future__ import annotations

import inspect

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import fdtdx_parity_model
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_model import (
    model_plan,
    polarization_vector,
)


def test_model_plan_places_every_object_at_contract_indices() -> None:
    plan = model_plan("Ea")
    assert plan["grid_shape"] == [186, 186, 286]
    assert plan["planned_slices"] == {
        "fixed_silicon_substrate": [[0, 186], [0, 186], [0, 53]],
        "fixed_285nm_sio2": [[0, 186], [0, 186], [53, 167]],
        "fixed_tairte4": [[13, 173], [13, 173], [167, 207]],
        "au_design": [[53, 133], [53, 133], [207, 227]],
        "gaussian_source": [[13, 173], [13, 173], [241, 242]],
        "incident_plane": [[13, 173], [13, 173], [236, 237]],
        "endpoint_field": [[13, 173], [13, 173], [228, 229]],
        "flake_profile": [[13, 173], [13, 173], [207, 208]],
        "material_flux": [[12, 174], [12, 174], [53, 236]],
        "material_flux_td": [[12, 174], [12, 174], [53, 236]],
    }


def test_model_plan_freezes_time_source_and_material_hashes() -> None:
    plan = model_plan("Eb")
    assert plan["time"] == {
        "courant_factor": 0.25,
        "total_periods": 40,
        "late_periods": 4,
        "previous_periods": 4,
        "source_startup_periods": 4,
        "dt_s": 2.083451820604655e-18,
        "time_steps_total": 256_163,
    }
    assert np.isclose(plan["source"]["std_relative_to_radius"], 1 / (2 * np.sqrt(2)))
    assert plan["source"]["target_intensity_1e2_radius_m"] == 4e-6
    assert plan["material_hashes"] == {
        "Au_nk_square_ADE": "71f6738a4c587387c334c3a31edcf8df1ff9415b8fdf2d66537b7a65b6b07b0f",
        "TaIrTe4_fixed_ADE": "fa9a435d79a7d01db22ec695940ebe993e6234b62fe5567fbd55a1664d08ede5",
    }
    assert plan["optimizer_enabled"] is False


def test_polarization_axis_mapping_is_not_ambiguous() -> None:
    assert polarization_vector("Ea") == (0.0, 1.0, 0.0)
    assert polarization_vector("Eb") == (1.0, 0.0, 0.0)
    assert model_plan("Ea")["electric_polarization_vector"] == [0.0, 1.0, 0.0]
    assert model_plan("Eb")["electric_polarization_vector"] == [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="unknown polarization"):
        polarization_vector("Ex")


def test_new_builder_has_no_legacy_density_or_layout_dependency() -> None:
    source = inspect.getsource(fdtdx_parity_model)
    assert "material_fraction" not in source
    assert "fdtdx_4um_model" not in source
    assert "combined_4um" not in source
    assert "LAYOUT" not in source
    assert "rho**3" not in source and "rho ** 3" not in source
    assert "au_coefficients_jax" in source
    assert "fixed_lorentz_parameters" in source
    assert "cublas_get_version" in source
    assert "cublas_runtime_version < 130200" in source


def test_air_only_plan_preserves_grid_and_marks_no_physical_material_claim() -> None:
    physical = model_plan("Ea", air_only=False)
    air = model_plan("Ea", air_only=True)
    assert air["air_only"] is True
    assert physical["air_only"] is False
    assert air["planned_slices"] == physical["planned_slices"]
    assert air["grid_xyz_edges_sha256"] == physical["grid_xyz_edges_sha256"]
    assert air["planned_float64_grid_xyz_edges_sha256"] == physical[
        "planned_float64_grid_xyz_edges_sha256"
    ]
