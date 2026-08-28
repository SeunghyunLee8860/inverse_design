from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_continuation import (
    BETA_SCHEDULE,
    FINAL_GRAYNESS_CAP,
    INITIAL_MAXIMIN_WARM_MAXIMUM_CHANGE,
    MINIMUM_CONTINUATION_EVALUATIONS,
    STAGE_FTOL_REL,
    STAGE_MAXEVAL,
    STAGE_XTOL_REL,
    ContinuationEpigraphProblem,
    active_design_constraint_names,
    continuation_contract,
    grayness_value_gradient,
    linearized_maximin_box_warm_start,
    stage_objective_progress,
    stage_design_caps,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_design_mapping import (
    calibrated_lumerical_250nm_dfm_caps,
    smooth_lumerical_250nm_constraints,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_optimizer import (
    uniform_initial_latent_density,
)


def _latent() -> np.ndarray:
    x = np.linspace(-1.0, 1.0, CONTRACT.design_node_shape[0])[:, None]
    y = np.linspace(-1.0, 1.0, CONTRACT.design_node_shape[1])[None, :]
    return 0.5 + 0.12 * np.sin(0.7 * np.pi * x) * np.cos(0.9 * np.pi * y)


def _fake_evaluation(latent: np.ndarray) -> dict[str, object]:
    value = np.asarray(latent, dtype=np.float64)
    return {
        "passed": True,
        "currents_A": {
            "Ea": float(-8.0e-9 + 1.0e-12 * np.sum(value - 0.5)),
            "Eb": float(-16.0e-9 - 2.0e-12 * np.sum(value - 0.5)),
        },
        "gradient_Ea_projected_A": np.full(CONTRACT.design_node_shape, 1.0e-12),
        "gradient_Eb_projected_A": np.full(CONTRACT.design_node_shape, -2.0e-12),
    }


def test_production_start_is_exact_uniform_rho_half() -> None:
    latent = uniform_initial_latent_density()
    assert latent.shape == CONTRACT.design_node_shape
    assert np.array_equal(latent, np.full(CONTRACT.design_node_shape, 0.5))


def test_production_stages_do_not_stop_on_tiny_balancing_step() -> None:
    assert STAGE_FTOL_REL == 0.0
    assert STAGE_XTOL_REL == 0.0


def test_continuation_safety_ceiling_and_lifecycle_are_explicit() -> None:
    assert MINIMUM_CONTINUATION_EVALUATIONS == 512
    assert STAGE_MAXEVAL[1.0] == 64
    assert STAGE_MAXEVAL[128.0] == 96
    assert INITIAL_MAXIMIN_WARM_MAXIMUM_CHANGE == 0.05
    lifecycle = continuation_contract()["MMA_lifecycle"]
    assert lifecycle["normal"] == "exactly one MMA object for each fixed beta"
    assert "crash recovery only" in lifecycle["same_beta_new_MMA"]


def test_continuation_contract_requires_optical_lateral_and_pde_convergence() -> None:
    from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_continuation import (
        continuation_contract,
    )

    required = continuation_contract()["final_promotion_requires"]
    assert any("100-to-50-nm optical lateral" in item for item in required)
    assert any("adaptive custom-CUDA PDE convergence" in item for item in required)
    assert any("current and temperature changes" in item for item in required)


def test_linearized_maximin_warm_start_improves_worst_utility_plane() -> None:
    latent = np.full(CONTRACT.design_node_shape, 0.5)
    gradient_a = np.zeros_like(latent)
    gradient_b = np.zeros_like(latent)
    gradient_a[0, 0] = 2.0
    gradient_a[0, 1] = -1.0
    gradient_b[0, 0] = 1.0
    gradient_b[0, 1] = -2.0
    result = linearized_maximin_box_warm_start(
        latent=latent,
        current_a_A=-2.0,
        current_b_A=3.0,
        gradient_a_latent_A=gradient_a,
        gradient_b_latent_A=gradient_b,
        maximum_change=0.1,
    )

    assert result["method"] == "exact_linearized_two_utility_box_dual_v1"
    assert result["maximum_abs_change"] == 0.1
    assert result["predicted_balanced_utility_A"] > -3.0
    assert result["predicted_improvement_A"] > 0.0
    assert np.all(np.asarray(result["latent"]) >= 0.0)
    assert np.all(np.asarray(result["latent"]) <= 1.0)


def test_design_constraints_activate_zero_one_two_three() -> None:
    counts = [len(active_design_constraint_names(beta)) for beta in BETA_SCHEDULE]
    assert counts == [0, 0, 1, 2, 3, 3, 3, 3]
    assert "grayness" in active_design_constraint_names(16.0)[-1]
    assert all(
        "conductance" not in name
        for beta in BETA_SCHEDULE
        for name in active_design_constraint_names(beta)
    )


def test_grayness_gradient_matches_directional_finite_difference() -> None:
    rng = np.random.default_rng(20260825)
    latent = _latent()
    direction = rng.standard_normal(CONTRACT.design_node_shape)
    direction /= np.max(np.abs(direction))
    beta = 16.0
    value, gradient = grayness_value_gradient(latent, beta)
    step = 1.0e-6
    plus = grayness_value_gradient(latent + step * direction, beta)[0]
    minus = grayness_value_gradient(latent - step * direction, beta)[0]
    finite_difference = (plus - minus) / (2.0 * step)
    adjoint = float(np.vdot(gradient, direction))
    assert 0.0 < value <= 1.0
    assert (
        abs(adjoint - finite_difference)
        / max(abs(adjoint), abs(finite_difference), 1.0e-14)
        < 1.0e-6
    )


def test_stage_caps_are_monotone_and_final_caps_are_calibrated() -> None:
    latent = _latent()
    previous = np.full(2, np.inf)
    previous_gray = np.inf
    for beta in BETA_SCHEDULE:
        baseline = smooth_lumerical_250nm_constraints(latent, beta)[0]
        baseline_grayness = grayness_value_gradient(latent, beta)[0]
        record = stage_design_caps(
            beta=beta,
            baseline_dfm_values=baseline,
            baseline_grayness=baseline_grayness,
            previous_dfm_caps=previous,
            previous_grayness_cap=previous_gray,
        )
        current = np.asarray(record["DFM_caps"])
        assert np.all(current <= previous)
        previous = current
        current_gray = float(record["grayness_cap"])
        assert current_gray <= previous_gray
        previous_gray = current_gray
    calibrated, _ = calibrated_lumerical_250nm_dfm_caps()
    assert np.allclose(previous, calibrated, rtol=0.0, atol=0.0)


def test_high_beta_problem_has_two_epigraph_plus_three_design_constraints() -> None:
    latent = _latent()
    baseline = smooth_lumerical_250nm_constraints(latent, 16.0)[0]
    caps = stage_design_caps(
        beta=16.0,
        baseline_dfm_values=baseline,
        baseline_grayness=grayness_value_gradient(latent, 16.0)[0],
        previous_dfm_caps=None,
        previous_grayness_cap=None,
    )
    problem = ContinuationEpigraphProblem(
        _fake_evaluation,
        beta=16.0,
        dfm_caps=np.asarray(caps["DFM_caps"]),
        grayness_cap=float(caps["grayness_cap"]),
    )
    vector = np.r_[latent.ravel(), -8.0]
    values = np.empty(problem.total_constraint_count)
    gradients = np.empty((problem.total_constraint_count, vector.size))
    problem.constraints(values, vector, gradients)
    assert problem.design_constraint_count == 3
    assert problem.total_constraint_count == 5
    assert values.shape == (5,)
    assert gradients.shape == (5, vector.size)
    assert np.all(gradients[2:, -1] == 0.0)


def test_grayness_cap_is_staged_then_fixed_at_final_beta_entry() -> None:
    latent = uniform_initial_latent_density()
    stage_16 = stage_design_caps(
        beta=16.0,
        baseline_dfm_values=smooth_lumerical_250nm_constraints(latent, 16.0)[0],
        baseline_grayness=grayness_value_gradient(latent, 16.0)[0],
        previous_dfm_caps=None,
        previous_grayness_cap=None,
    )
    assert stage_16["grayness_cap"] == pytest.approx(0.9)
    stage_128 = stage_design_caps(
        beta=128.0,
        baseline_dfm_values=smooth_lumerical_250nm_constraints(latent, 128.0)[0],
        baseline_grayness=grayness_value_gradient(latent, 128.0)[0],
        previous_dfm_caps=np.asarray(stage_16["DFM_caps"]),
        previous_grayness_cap=float(stage_16["grayness_cap"]),
    )
    assert stage_128["grayness_cap"] == FINAL_GRAYNESS_CAP


def test_objective_plateau_gate_rejects_a_recently_improving_stage() -> None:
    improving = [
        {
            "balanced_utility_nA": float(index),
            "maximum_design_constraint": -0.1,
        }
        for index in range(8)
    ]
    plateau = [
        {
            "balanced_utility_nA": 1.0 + 1.0e-5 * index,
            "maximum_design_constraint": -0.1,
        }
        for index in range(8)
    ]
    assert stage_objective_progress(improving)["converged"] is False
    assert stage_objective_progress(plateau)["converged"] is True


def test_problem_requests_one_force_stop_after_physics_plateau() -> None:
    stopped: list[bool] = []

    def evaluation(latent: np.ndarray) -> dict[str, object]:
        value = float(np.mean(latent))
        return {
            "passed": True,
            "currents_A": {
                "Ea": (2.0 + 1.0e-5 * value) * 1.0e-9,
                "Eb": -(2.0 + 1.0e-5 * value) * 1.0e-9,
            },
            "gradient_Ea_projected_A": np.zeros(CONTRACT.design_node_shape),
            "gradient_Eb_projected_A": np.zeros(CONTRACT.design_node_shape),
        }

    problem = ContinuationEpigraphProblem(
        evaluation, beta=1.0, dfm_caps=np.full(2, np.inf), grayness_cap=np.inf
    )
    problem.bind_force_stop(lambda: stopped.append(True))
    for index in range(8):
        latent = np.full(CONTRACT.design_node_shape, 0.45 + 0.01 * index)
        problem.point(np.r_[latent.ravel(), 1.0])
    assert stopped == [True]
    assert problem.plateau_stop_requested is True
    assert problem.plateau_result is not None
    assert problem.plateau_result["converged"] is True


def test_problem_selects_best_physics_point_not_nlopt_terminal_point() -> None:
    def evaluation(latent: np.ndarray) -> dict[str, object]:
        level = float(np.mean(latent))
        return {
            "passed": True,
            "currents_A": {"Ea": level * 1.0e-9, "Eb": -level * 1.0e-9},
            "gradient_Ea_projected_A": np.full(
                CONTRACT.design_node_shape, 1.0e-9 / np.prod(CONTRACT.design_node_shape)
            ),
            "gradient_Eb_projected_A": np.full(
                CONTRACT.design_node_shape,
                -1.0e-9 / np.prod(CONTRACT.design_node_shape),
            ),
        }

    problem = ContinuationEpigraphProblem(
        evaluation,
        beta=1.0,
        dfm_caps=np.full(2, np.inf),
        grayness_cap=np.inf,
    )
    better = np.full(CONTRACT.design_node_shape, 0.6)
    worse_terminal = np.full(CONTRACT.design_node_shape, 0.4)
    problem.point(np.r_[better.ravel(), 0.6])
    problem.point(np.r_[worse_terminal.ravel(), 0.4])
    selected = problem.selected_candidate()
    assert np.array_equal(selected["latent"], better)
    assert selected["callback_index"] == 0


def test_stage_caps_are_checkpointed_before_first_maxwell_evaluation() -> None:
    source = (
        Path(__file__)
        .with_name("41_optimize_lumerical_4um_dualpol_continuation.py")
        .read_text(encoding="utf-8")
    )
    cap_setup = source.index('if int(state["attempt"]) == 0:')
    checkpoint = source.index("_save_checkpoint(checkpoint_path, **state)", cap_setup)
    first_maxwell = source.index("initial_physics = driver.evaluate(latent_initial)")
    assert cap_setup < checkpoint < first_maxwell


def test_mma_uses_global_density_bounds_not_a_stage_start_box() -> None:
    source = (
        Path(__file__)
        .with_name("41_optimize_lumerical_4um_dualpol_continuation.py")
        .read_text(encoding="utf-8")
    )
    assert "np.zeros(variable_count - 1" in source
    assert "np.ones(variable_count - 1" in source
    assert "latent_initial.ravel() -" not in source


def test_launcher_uses_one_mma_per_beta_and_callback_plateau_stop() -> None:
    source = (
        Path(__file__)
        .with_name("41_optimize_lumerical_4um_dualpol_continuation.py")
        .read_text(encoding="utf-8")
    )
    assert source.count("nlopt.opt(nlopt.LD_MMA") == 1
    assert "problem.bind_force_stop(optimizer.force_stop)" in source
    assert "except nlopt.ForcedStop:" in source
    assert "objective_extension_required" not in source
    assert "MAXIMUM_STAGE_ATTEMPTS" not in source
    assert "tightened_final_grayness_cap" not in source


def test_uniform_b200_launcher_is_direct_and_clears_restart_state() -> None:
    source = (
        Path(__file__)
        .with_name("launch_lumerical_b200_uniform_rho0p5.sh")
        .read_text(encoding="utf-8")
    )
    assert 'license_mode="direct_checkout"' in source
    assert "unset AU_LUMERICAL_RESTART_CHECKPOINT" in source
    assert "unset AU_LUMERICAL_RESTART_MANIFEST" in source
    assert "runres" not in source
    assert "--preflight-only" in source
    assert "NEW_UNIFORM_OUTPUT_ROOT" in source


def _load_continuation_driver():
    path = Path(__file__).with_name("41_optimize_lumerical_4um_dualpol_continuation.py")
    name = "_test_lumerical_4um_continuation_driver"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_beta_fd_certificate_is_recorded_once_and_reverified(tmp_path: Path) -> None:
    driver = _load_continuation_driver()
    certificate = tmp_path / "component_yee_jacobian_result.json"
    certificate.write_text(
        json.dumps(
            {
                "status": "PASSED_LUMERICAL_4UM_COMPONENT_YEE_JACOBIAN",
                "passed": True,
                "git_commit": driver._git_commit(),
                "optimization_beta": 1.0,
                "validation_scope": "stage-entry",
                "validation": {
                    "mode": "full_independent_mapping_FD_and_transpose",
                    "independent_mapping_FD_performed": True,
                },
                "gates": {"mapping_FD_and_transpose_passed": True},
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "component_yee_independent_FD_cadence": {
            "schema": "component-yee-independent-fd-cadence-v1",
            "stage_certificates": {},
        }
    }
    initial = {
        "density_state": {"density_state_sha256": "representative"},
        "Jacobian_result": driver.artifact(certificate),
        "Jacobian_validation": {
            "independent_mapping_FD_performed": True,
        },
    }
    recorded = driver._record_stage_fd_certificate(
        manifest=manifest,
        beta=1.0,
        attempt=4,
        initial_physics=initial,
    )
    assert recorded == certificate
    assert driver._stage_fd_certificate(manifest, 1.0) == certificate
    with pytest.raises(RuntimeError, match="refusing to replace"):
        driver._record_stage_fd_certificate(
            manifest=manifest,
            beta=1.0,
            attempt=5,
            initial_physics=initial,
        )


def test_final_precursor_fd_runs_only_after_exact_binary_physics_passes() -> None:
    source = (
        Path(__file__)
        .with_name("41_optimize_lumerical_4um_dualpol_continuation.py")
        .read_text(encoding="utf-8")
    )
    exact_switch = source.index("if exact_switching:")
    final_full_chain = source.index("driver.audit_full_chain_latent_adfd", exact_switch)
    final_fd = source.index(
        "driver.audit_final_binary_precursor_independent_fd", final_full_chain
    )
    final_promotion = source.index("_save_final_binary_mask", final_fd)
    assert exact_switch < final_full_chain < final_fd < final_promotion


def test_passed_preflight_manifest_can_transition_to_full_run() -> None:
    driver = _load_continuation_driver()
    manifest = {
        "status": driver.PREFLIGHT_STATUS,
        "passed": True,
        "preflight_only": True,
        "stages": [],
        "final": None,
    }
    assert driver._completed_manifest_latent(manifest) is None


def test_malformed_passed_preflight_manifest_fails_closed() -> None:
    driver = _load_continuation_driver()
    manifest = {
        "status": driver.PREFLIGHT_STATUS,
        "passed": True,
        "preflight_only": True,
        "stages": [{"unexpected": "physics result"}],
        "final": None,
    }
    with pytest.raises(RuntimeError, match="preflight-only.*malformed"):
        driver._completed_manifest_latent(manifest)


def test_explicit_stopped_checkpoint_restart_is_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = _load_continuation_driver()
    latent = _latent()
    checkpoint = tmp_path / "continuation_checkpoint.npz"
    driver._save_checkpoint(
        checkpoint,
        latent=latent,
        beta_index=0,
        attempt=2,
        dfm_caps=np.full(2, np.inf),
        grayness_cap=np.inf,
    )
    terminal_state = tmp_path / "stage_final_state.npz"
    np.savez_compressed(terminal_state, latent_final=latent)
    manifest = tmp_path / "production_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "STOPPED_UNRESOLVED_STAGE_OBJECTIVE_OR_DESIGN_GATES",
                "passed": False,
                "git_commit": "old-commit",
                "blocking_attempts": 2,
                "latest": {"beta": 1.0, "attempt": 1},
                "stages": [
                    {
                        "state_artifact": driver.artifact(terminal_state),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AU_LUMERICAL_RESTART_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("AU_LUMERICAL_RESTART_MANIFEST", str(manifest))
    state, provenance = driver._restart_seed_from_environment()
    assert np.array_equal(state["latent"], latent)
    assert state["attempt"] == 2
    assert provenance["resumed_beta_index"] == 0
    assert provenance["source_status"].startswith("STOPPED_")


def test_portable_restart_manifest_resolves_relative_terminal_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = _load_continuation_driver()
    latent = _latent()
    checkpoint = tmp_path / "continuation_checkpoint.npz"
    driver._save_checkpoint(
        checkpoint,
        latent=latent,
        beta_index=0,
        attempt=4,
        dfm_caps=np.full(2, np.inf),
        grayness_cap=np.inf,
    )
    terminal_state = tmp_path / "terminal_stage_state.npz"
    np.savez_compressed(terminal_state, latent_final=latent)
    relative_artifact = driver.artifact(terminal_state)
    relative_artifact["path"] = terminal_state.name
    manifest = tmp_path / "restart_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "STOPPED_FOR_PORTABLE_B200_MIGRATION",
                "passed": False,
                "git_commit": "old-commit",
                "blocking_attempts": 4,
                "latest": {"beta": 1.0, "attempt": 3},
                "stages": [{"state_artifact": relative_artifact}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AU_LUMERICAL_RESTART_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("AU_LUMERICAL_RESTART_MANIFEST", str(manifest))
    state, provenance = driver._restart_seed_from_environment()
    assert np.array_equal(state["latent"], latent)
    assert state["attempt"] == 4
    assert provenance["source_terminal_state"]["path"] == str(terminal_state)


def test_committed_b200_migration_bundle_is_hash_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _load_continuation_driver()
    bundle_root = Path(__file__).resolve().parent / "b200_migration"
    bundle = json.loads(
        (bundle_root / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    assert bundle["schema"] == "au-lumerical-b200-migration-bundle-v1"
    for record in bundle["files"].values():
        path = bundle_root / record["path"]
        assert path.stat().st_size == record["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    monkeypatch.setenv(
        "AU_LUMERICAL_RESTART_CHECKPOINT",
        str(bundle_root / "continuation_checkpoint.npz"),
    )
    monkeypatch.setenv(
        "AU_LUMERICAL_RESTART_MANIFEST",
        str(bundle_root / "restart_manifest.json"),
    )
    state, provenance = driver._restart_seed_from_environment()
    assert state["beta_index"] == 0
    assert state["attempt"] == 4
    assert np.min(state["latent"]) == 0.0
    assert np.max(state["latent"]) == 1.0
    assert provenance["source_status"].startswith("STOPPED_FOR_B200_MIGRATION")
