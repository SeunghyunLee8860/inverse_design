from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    lumerical_maxwell_contract as maxwell_contract,
)

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_maxwell_contract import (
    CONTRACT,
    binary_mask_sha256,
    canonical_binary_mask,
    canonical_projected_density,
    exact_au_geometry_audit,
    exact_au_geometry_sha256,
    projected_density_sha256,
)


def test_contract_preserves_lumerical_plus_custom_gpu_pde_architecture() -> None:
    payload = asdict(CONTRACT)
    assert payload["maxwell_solver"].startswith("Ansys Lumerical FDTD")
    assert payload["maxwell_accelerator_required"] == "NVIDIA B200"
    assert "custom CUDA" in payload["thermal_solver"]
    assert "custom CUDA" in payload["electrical_solver"]
    assert payload["density_topology_required"] is True
    assert payload["shape_or_level_set_required"] is False
    assert payload["continuous_relaxation_allowed_during_optimization"] is True
    assert payload["projected_density_grid"].startswith("81x81 nodes")
    assert "exact discrete transpose" in payload["custom_pde_density_map"]
    assert payload["exact_binary_required_for_every_physics_evaluation"] is False
    assert payload["numerical_interface_cut_cells_allowed"] is True
    assert payload["different_optical_thermal_electrical_design_fields_allowed"] is False
    assert payload["exact_binary_required_for_final_promotion"] is True
    assert payload["exact_dispersive_au_required_at_material_endpoint"] is True
    assert payload["exact_dispersive_au_required_for_final_reevaluation"] is True
    assert payload["optical_relaxation_law"] == "christiansen_nk_then_square_v1"
    assert payload["optical_rho_power"] is None
    assert payload["density_filter_radius_m"] == 250e-9
    assert payload["minimum_solid_feature_m"] == 250e-9
    assert payload["minimum_void_feature_m"] == 250e-9
    assert payload["np_density_as_au_topology_variable_allowed"] is False
    assert payload["bundled_lumopt_topology_gradient_allowed_without_au_adfd"] is False
    assert payload["fdtdx_allowed"] is False
    assert payload["jax_maxwell_allowed"] is False


def test_source_audit_allows_absent_lumopt2_but_not_partial_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = tmp_path / "lumopt" / "geometries" / "topology.py"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        "params.eps_levels=[{0},{1}]\ndF_dEps = real(dF_dEps)\n",
        encoding="utf-8",
    )
    topology2 = tmp_path / "lumopt2" / "parametrization" / "topology.py"
    deps2 = tmp_path / "lumopt2" / "parametrization" / "d_eps_calculator.py"
    monkeypatch.setattr(maxwell_contract, "LEGACY_LUMOPT_TOPOLOGY", legacy)
    monkeypatch.setattr(maxwell_contract, "LUMOPT2_TOPOLOGY", topology2)
    monkeypatch.setattr(maxwell_contract, "LUMOPT2_DEPS", deps2)

    absent = maxwell_contract._source_audit()
    assert absent["passed"] is True
    assert absent["lumopt2_status"] == "NOT_INSTALLED_CUSTOM_AU_ROUTE_REQUIRED"

    topology2.parent.mkdir(parents=True)
    topology2.write_text("material_index: float\n", encoding="utf-8")
    partial = maxwell_contract._source_audit()
    assert partial["passed"] is False
    assert partial["error"] == "partial LumOpt2 installation cannot be audited"


def test_installation_audit_rejects_the_wrong_minor_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "v261"
    api = root / "api" / "python" / "lumapi.py"
    engine = root / "bin" / "fdtd-engine"
    api.parent.mkdir(parents=True)
    engine.parent.mkdir(parents=True)
    api.write_text("# test\n", encoding="utf-8")
    engine.write_text("test\n", encoding="utf-8")
    monkeypatch.setattr(maxwell_contract, "LUMERICAL_ROOT", root)
    monkeypatch.setattr(maxwell_contract, "LUMAPI_PATH", api)
    (root / "VERSION").write_text(
        "MAJORRELEASE=2026R1\nMINORRELEASE=0\nBUILDNUMBER=4413\n",
        encoding="utf-8",
    )
    assert maxwell_contract._installation_audit()["passed"] is False
    (root / "VERSION").write_text(
        "MAJORRELEASE=2026R1\nMINORRELEASE=2\nBUILDNUMBER=4522\n",
        encoding="utf-8",
    )
    assert maxwell_contract._installation_audit()["passed"] is True


def test_development_accelerator_policy_never_issues_b200_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        maxwell_contract,
        "_run_nvidia_smi",
        lambda: [
            {
                "index": 6,
                "name": "NVIDIA RTX 6000 Ada Generation",
                "uuid": "GPU-test",
                "memory_total_MiB": 49140,
                "compute_capability": "8.9",
                "driver_version": "test",
            }
        ],
    )
    monkeypatch.setattr(
        maxwell_contract,
        "_source_audit",
        lambda: {"passed": True},
    )
    monkeypatch.setattr(
        maxwell_contract,
        "_installation_audit",
        lambda: {"passed": True},
    )
    monkeypatch.setattr(
        maxwell_contract,
        "_fdtd_solve_license_audit",
        lambda: {"passed": True, "tasks_available": 10},
    )
    monkeypatch.setattr(maxwell_contract, "LUMAPI_PATH", Path(__file__))
    development = maxwell_contract.audit_environment(
        requested_gpu_index=6,
        accelerator_policy="development",
    )
    assert development["status"] == (
        "READY_FOR_LUMERICAL_DEVELOPMENT_GPU_NOT_B200_CERTIFIED"
    )
    assert development["b200_promotion_certified"] is False
    strict = maxwell_contract.audit_environment(
        requested_gpu_index=6,
        accelerator_policy="b200",
    )
    assert strict["status"] == "BLOCKED_LUMERICAL_GPU_PREFLIGHT"


def test_license_task_exhaustion_blocks_gpu_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        maxwell_contract,
        "_run_nvidia_smi",
        lambda: [{"index": 2, "name": "NVIDIA test", "uuid": "GPU-test"}],
    )
    monkeypatch.setattr(maxwell_contract, "_source_audit", lambda: {"passed": True})
    monkeypatch.setattr(maxwell_contract, "LUMAPI_PATH", Path(__file__))
    monkeypatch.setattr(
        maxwell_contract,
        "_fdtd_solve_license_audit",
        lambda: {
            "passed": False,
            "tasks_available": 9,
            "tasks_required": 10,
        },
    )
    result = maxwell_contract.audit_environment(
        requested_gpu_index=2,
        accelerator_policy="development",
    )
    assert result["status"] == "BLOCKED_LUMERICAL_GPU_PREFLIGHT"
    assert result["gates"]["fdtd_solve_license_tasks_available"] is False


def test_explicit_direct_checkout_defers_capacity_to_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AU_LUMERICAL_LICENSE_MODE", "direct_checkout")
    monkeypatch.setattr(
        maxwell_contract.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("lmstat must not run in direct mode"),
    )

    result = maxwell_contract._fdtd_solve_license_audit()

    assert result["passed"] is True
    assert result["passed_via"] == "direct_solver_checkout"
    assert result["prelaunch_capacity_verified"] is False
    assert result["reservation_verified"] is False


def test_unknown_license_mode_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AU_LUMERICAL_LICENSE_MODE", "guess")
    with pytest.raises(ValueError, match="AU_LUMERICAL_LICENSE_MODE"):
        maxwell_contract._fdtd_solve_license_audit()


def test_license_audit_accepts_exact_server_verified_project_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = "PROJECT_seunghyun_au4um_smoke_123"
    lmstat = f"""
Users of lum_fdtd_solve:  (Total of 60 licenses issued;  Total of 57 licenses in use)

    10 RESERVATIONs for PROJECT {project}
    user host display (v1.0) (server/1055 123), start Tue 8/25 08:00

Users of lum_fdtd_gui:  (Total of 20 licenses issued;  Total of 4 licenses in use)
    4 RESERVATIONs for PROJECT {project}
"""
    monkeypatch.setenv("LM_PROJECT", project)
    monkeypatch.setattr(
        maxwell_contract.subprocess,
        "run",
        lambda *args, **kwargs: maxwell_contract.subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout=lmstat, stderr=""
        ),
    )

    result = maxwell_contract._fdtd_solve_license_audit()

    assert result["tasks_available"] == 3
    assert result["reservation_tasks_for_project"] == 10
    assert result["reservation_verified"] is True
    assert result["passed_via"] == "verified_project_reservation"
    assert result["passed"] is True


def test_license_audit_rejects_unverified_lm_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lmstat = """
Users of lum_fdtd_solve:  (Total of 60 licenses issued;  Total of 57 licenses in use)
    10 RESERVATIONs for PROJECT PROJECT_someone_else_456
Users of lum_fdtd_gui:  (Total of 20 licenses issued;  Total of 4 licenses in use)
"""
    monkeypatch.setenv("LM_PROJECT", "PROJECT_spoofed_123")
    monkeypatch.setattr(
        maxwell_contract.subprocess,
        "run",
        lambda *args, **kwargs: maxwell_contract.subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout=lmstat, stderr=""
        ),
    )

    result = maxwell_contract._fdtd_solve_license_audit()

    assert result["tasks_available"] == 3
    assert result["reservation_tasks_for_project"] == 0
    assert result["reservation_verified"] is False
    assert result["passed_via"] is None
    assert result["passed"] is False


def test_license_audit_retries_consumed_project_reservation_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = "PROJECT_seunghyun_au4um_smoke_789"
    consumed = """
Users of lum_fdtd_solve:  (Total of 60 licenses issued;  Total of 57 licenses in use)
Users of lum_fdtd_gui:  (Total of 20 licenses issued;  Total of 4 licenses in use)
"""
    restored = f"""
Users of lum_fdtd_solve:  (Total of 60 licenses issued;  Total of 57 licenses in use)
    10 RESERVATIONs for PROJECT {project}
Users of lum_fdtd_gui:  (Total of 20 licenses issued;  Total of 4 licenses in use)
"""
    outputs = iter((consumed, restored))
    monkeypatch.setenv("LM_PROJECT", project)
    monkeypatch.setenv("AU_LUMERICAL_LICENSE_AUDIT_WAIT_S", "10")
    monkeypatch.setattr(maxwell_contract.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        maxwell_contract.subprocess,
        "run",
        lambda *args, **kwargs: maxwell_contract.subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout=next(outputs), stderr=""
        ),
    )

    result = maxwell_contract._fdtd_solve_license_audit()

    assert result["passed"] is True
    assert result["passed_via"] == "verified_project_reservation"
    assert result["reservation_tasks_for_project"] == 10
    assert result["audit_attempts"] == 2


def test_projected_density_hash_accepts_gray_and_is_layout_sensitive() -> None:
    density = np.asarray([[0.0, 0.25, 0.5], [0.75, 1.0, 0.125]])
    assert np.array_equal(canonical_projected_density(density), density)
    assert projected_density_sha256(density) != projected_density_sha256(density.T)
    changed = density.copy()
    changed[0, 0] = 0.01
    assert projected_density_sha256(density) != projected_density_sha256(changed)


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


@pytest.mark.parametrize(
    "bad",
    [
        np.asarray([0.0, 0.5]),
        np.asarray([[0.0, 1.01]]),
        np.asarray([[0.0, np.nan]]),
        np.empty((0, 2)),
    ],
)
def test_projected_density_rejects_bad_inputs(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        canonical_projected_density(bad)
