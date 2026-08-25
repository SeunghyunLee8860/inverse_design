from __future__ import annotations

from pathlib import Path

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_contract import (
    FDTDX_SOURCE_COMMIT,
    PHYSICS,
    fdtdx_runtime_audit,
    grid_audit,
    grid_edges,
    grid_hashes,
    parity_contract,
    placement_contract,
)


def _bounds(edges: np.ndarray, pair: list[int]) -> tuple[float, float]:
    return float(edges[pair[0]]), float(edges[pair[1]])


def test_parity_grid_has_requested_cell_counts_and_pitches() -> None:
    x, y, z = grid_edges()
    assert (x.size - 1, y.size - 1, z.size - 1) == (186, 186, 286)
    assert (x.size - 1) * (y.size - 1) * (z.size - 1) == 9_894_456
    assert np.allclose(np.diff(x)[13:173], 100e-9, rtol=0.0, atol=2e-18)
    assert np.allclose(np.diff(y)[13:173], 100e-9, rtol=0.0, atol=2e-18)
    assert np.allclose(np.diff(z)[53:227], 2.5e-9, rtol=0.0, atol=2e-18)
    assert np.max(np.diff(z)[8:53]) <= 50e-9 + 2e-18
    assert np.max(np.diff(z)[227:278]) <= 50e-9 + 2e-18
    assert grid_audit()["status"] == "PASS"


def test_all_geometry_is_derived_from_physical_edges() -> None:
    x, y, z = grid_edges()
    placement = placement_contract()
    volumes = placement["volumes_cell_slices"]
    expected = {
        "Si": ((-10e-6, 10e-6), (-10e-6, 10e-6), (-3e-6, -0.385e-6)),
        "SiO2": ((-10e-6, 10e-6), (-10e-6, 10e-6), (-0.385e-6, -0.1e-6)),
        "TaIrTe4": ((-8e-6, 8e-6), (-8e-6, 8e-6), (-0.1e-6, 0.0)),
        "Au_design": ((-4e-6, 4e-6), (-4e-6, 4e-6), (0.0, 0.05e-6)),
        "closed_flux_box": ((-8.2e-6, 8.2e-6), (-8.2e-6, 8.2e-6), (-0.385e-6, 0.5e-6)),
    }
    for name, expected_bounds in expected.items():
        indices = volumes[name]
        actual = (_bounds(x, indices[0]), _bounds(y, indices[1]), _bounds(z, indices[2]))
        assert np.allclose(actual, expected_bounds, rtol=0.0, atol=2e-18)

    assert placement["source_aperture_cell_slices_xy"] == [[13, 173], [13, 173]]
    assert placement["pml_cell_slices"] == {
        "x_minus": [0, 8],
        "x_plus": [178, 186],
        "y_minus": [0, 8],
        "y_plus": [178, 186],
        "z_minus": [0, 8],
        "z_plus": [278, 286],
    }


def test_source_and_monitor_planes_are_exact_grid_edges() -> None:
    _, _, z = grid_edges()
    planes = placement_contract()["planes_edge_indices"]
    expected = {
        "source": PHYSICS.source_z_m,
        "incident_power": PHYSICS.incident_monitor_z_m,
        "air_endpoint_field": PHYSICS.endpoint_monitor_z_m,
        "flake_profile": PHYSICS.flake_plane_z_m,
    }
    for name, coordinate in expected.items():
        assert np.isclose(z[planes[name]["index"]], coordinate, rtol=0.0, atol=2e-18)


def test_grid_hashes_bind_the_complete_rectilinear_grid() -> None:
    hashes = grid_hashes()
    assert hashes["x_edges_sha256"] == hashes["y_edges_sha256"]
    assert set(hashes) == {
        "x_edges_sha256",
        "y_edges_sha256",
        "z_edges_sha256",
        "xyz_edges_sha256",
    }
    assert all(len(value) == 64 for value in hashes.values())
    assert hashes == {
        "x_edges_sha256": "4b1a47ea8a97be981807bfc03f9d1412632c74f5361fee0ea25ee78e08fc524e",
        "y_edges_sha256": "4b1a47ea8a97be981807bfc03f9d1412632c74f5361fee0ea25ee78e08fc524e",
        "z_edges_sha256": "8b09a1c773a751ad39843b71d4fb92f50819e53b1dac6a59bcf84d1613c8ff4f",
        "xyz_edges_sha256": "15e2ce87ec5485de2712718b0f12a289e64233a69b98f4cae23b3cb5349e7805",
    }


def test_cfl_and_memory_audit_exposes_cost_without_claiming_peak_memory() -> None:
    resources = grid_audit()["resources"]
    assert resources["status"] == "ANALYTIC_LOWER_BOUND_ONLY"
    assert resources["field_run_feasibility"].startswith("UNDETERMINED")
    assert np.isclose(resources["time"]["dt_s"], 2.0834738305187266e-18)
    assert resources["time"]["total_steps"] == 256_160
    assert resources["time"]["late_window_steps"] == 25_616
    assert resources["work"]["cell_steps_per_forward"] == 2_534_563_848_960
    assert resources["work"]["estimated_wall_time_s"] is None
    one_pole = resources["memory"]["pole_cases_no_c4"]["1"]
    assert one_pole["persistent_array_lower_bound_bytes"] == 985_961_856
    assert one_pole["one_dynamic_checkpoint_lower_bound_bytes"] == 511_026_816
    assert "XLA_temporaries_and_cotangents" in resources["memory"]["excluded_from_lower_bound"]


def test_contract_forbids_legacy_route_and_optimizer_prematurity() -> None:
    contract = parity_contract()
    assert contract["authority"]["fdtdx_allowed_as_final_authority"] is False
    assert contract["authority"]["lumerical_heat_charge_calls_allowed_in_this_route"] is False
    assert contract["topology"]["latent_shape"] == [81, 81]
    assert contract["topology"]["physical_cell_shape"] == [80, 80]
    assert contract["topology"]["one_shared_occupancy_for_all_physics"] is True
    assert contract["optical_density_law"]["rho_cubed_allowed"] is False
    assert contract["optical_density_law"]["c3_only_scaling_allowed"] is False
    assert contract["gates"]["optimizer_enabled"] is False
    assert "legacy_scripts_10_12_13" in contract["hard_prohibitions"]
    assert contract["coordinates"]["Ea_electric_vector"] == [0.0, 1.0, 0.0]
    assert contract["coordinates"]["Eb_electric_vector"] == [1.0, 0.0, 0.0]
    assert contract["objective"]["constraints"] == [
        "t - I_Ea <= 0",
        "t + I_Eb <= 0",
    ]


def test_runtime_audit_requires_exact_clean_commit_and_import_tree(tmp_path: Path) -> None:
    source = tmp_path / "fdtdx-src"
    module = source / "src" / "fdtdx" / "__init__.py"
    module.parent.mkdir(parents=True)
    module.touch()

    def clean_runner(arguments: list[str], cwd: Path) -> str:
        assert cwd == source.resolve()
        if arguments == ["rev-parse", "HEAD"]:
            return FDTDX_SOURCE_COMMIT
        if arguments == ["status", "--porcelain"]:
            return ""
        raise AssertionError(arguments)

    passed = fdtdx_runtime_audit(source, git_runner=clean_runner, resolved_module=module)
    assert passed["status"] == "PASS"

    def wrong_commit_runner(arguments: list[str], cwd: Path) -> str:
        if arguments == ["rev-parse", "HEAD"]:
            return "0" * 40
        return " M src/fdtdx/__init__.py"

    blocked = fdtdx_runtime_audit(
        source,
        git_runner=wrong_commit_runner,
        resolved_module=tmp_path / "other" / "fdtdx" / "__init__.py",
    )
    assert blocked["status"] == "BLOCKED"
    assert blocked["checks"] == {
        "source_exists": True,
        "commit_is_pinned": False,
        "source_tree_is_clean": False,
        "import_resolves_under_pinned_source": False,
    }
