from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_design_mapping import (
    OPTIMIZER_250NM_MAPPING,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_optimizer import (
    CURRENT_SCALE_A,
    LumericalEvaluationDriver,
    OptimizerRuntime,
    SmokeEpigraphProblem,
    artifact,
    full_chain_adfd_step,
    initial_latent_density,
)


def _fake_evaluation(latent: np.ndarray) -> dict[str, object]:
    value = np.asarray(latent, float)
    gradient_a = np.full(CONTRACT.design_node_shape, 2.0e-12)
    gradient_b = np.full(CONTRACT.design_node_shape, -3.0e-12)
    return {
        "passed": True,
        "currents_A": {
            "Ea": float(-9.0e-9 + 2.0e-12 * np.sum(value - 0.5)),
            "Eb": float(-17.0e-9 - 3.0e-12 * np.sum(value - 0.5)),
        },
        "gradient_Ea_projected_A": gradient_a,
        "gradient_Eb_projected_A": gradient_b,
    }


def test_initial_latent_matches_validated_field_independent_contract() -> None:
    latent = initial_latent_density()
    assert latent.shape == (81, 81)
    assert np.min(latent) >= 0.34 - 2.0 * np.finfo(float).eps
    assert np.max(latent) <= 0.66 + 2.0 * np.finfo(float).eps


def test_epigraph_callbacks_have_exact_sign_and_scale() -> None:
    latent = initial_latent_density()
    problem = SmokeEpigraphProblem(
        _fake_evaluation, beta=4.0, dfm_caps=np.asarray((1.0, 1.0))
    )
    vector = np.r_[latent.ravel(), -9.0]
    result = np.empty(4)
    gradient = np.empty((4, vector.size))
    problem.constraints(result, vector, gradient)
    point = problem.point(vector)

    assert (
        result[0] == (-9.0 * CURRENT_SCALE_A - point["current_a_A"]) / CURRENT_SCALE_A
    )
    assert (
        result[1] == (-9.0 * CURRENT_SCALE_A + point["current_b_A"]) / CURRENT_SCALE_A
    )
    assert gradient[0, -1] == 1.0
    assert gradient[1, -1] == 1.0
    assert np.allclose(
        gradient[0, :-1],
        -np.asarray(point["gradient_a_latent_A"]).ravel() / CURRENT_SCALE_A,
    )
    assert np.allclose(
        gradient[1, :-1],
        np.asarray(point["gradient_b_latent_A"]).ravel() / CURRENT_SCALE_A,
    )


def test_same_latent_different_epigraph_does_not_repeat_physics() -> None:
    calls = 0

    def evaluate(latent: np.ndarray) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _fake_evaluation(latent)

    latent = initial_latent_density()
    problem = SmokeEpigraphProblem(evaluate, beta=4.0, dfm_caps=np.asarray((1.0, 1.0)))
    problem.point(np.r_[latent.ravel(), -9.0])
    second = problem.point(np.r_[latent.ravel(), -8.5])
    assert calls == 1
    assert second["epigraph_nA"] == -8.5


def test_production_retention_prunes_only_declared_heavy_transients(tmp_path) -> None:
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    removable = (
        evaluation / "forward.fsp",
        evaluation / "result.h5",
        evaluation / "forward_raw.npz",
        evaluation / "gray_q_cuda_pde_pullback.npz",
    )
    retained = (
        evaluation / "evaluation_result.json",
        evaluation / "signed_projected_gradients.npz",
        evaluation / "adjoint_gradient.npz",
        evaluation / "solver.log",
    )
    for path in (*removable, *retained):
        path.write_bytes(b"test")
    driver = object.__new__(LumericalEvaluationDriver)
    record = driver._prune_completed_evaluation(evaluation)
    assert record["pruned_file_count"] == len(removable)
    assert all(not path.exists() for path in removable)
    assert all(path.exists() for path in retained)
    assert (evaluation / "ARTIFACT_RETENTION.json").is_file()


def test_driver_runs_full_fd_once_then_uses_hash_bound_certificate(
    tmp_path, monkeypatch
) -> None:
    runtime = OptimizerRuntime(
        output_root=tmp_path / "attempt",
        source_calibration={},
        gpu_index=7,
        threads=8,
        accelerator_policy="b200",
        beta=4.0,
    )
    driver = LumericalEvaluationDriver(runtime)
    forward_fsp = tmp_path / "forward.fsp"
    forward_result = tmp_path / "forward.json"
    density = tmp_path / "density.npz"
    forward_fsp.write_bytes(b"fsp")
    forward_result.write_text("{}", encoding="utf-8")
    density.write_bytes(b"density")
    calls: list[list[str]] = []

    def fake_command(script: str, *arguments: str, log_path: Path) -> None:
        assert script == "26_build_lumerical_4um_yee_jacobian.py"
        values = list(arguments)
        calls.append(values)
        output = Path(values[values.index("--output-dir") + 1])
        mode = values[values.index("--independent-fd-validation") + 1]
        output.mkdir(parents=True)
        result = {
            "passed": True,
            "optimization_beta": runtime.beta,
            "validation": {
                "mode": mode,
                "independent_mapping_FD_performed": mode == "full",
            },
        }
        if mode != "full":
            result["independent_fd_certificate"] = {"passed": True}
        (output / "component_yee_jacobian_result.json").write_text(
            json.dumps(result), encoding="utf-8"
        )

    monkeypatch.setattr(driver, "_command", fake_command)
    forward = {"fsp": forward_fsp, "result_path": forward_result}
    first = driver._jacobian(forward, density, tmp_path / "jacobian_1")
    second = driver._jacobian(forward, density, tmp_path / "jacobian_2")

    assert first["full_independent_FD_performed"] is True
    assert second["full_independent_FD_performed"] is False
    assert calls[0][calls[0].index("--independent-fd-validation") + 1] == "full"
    assert (
        calls[1][calls[1].index("--independent-fd-validation") + 1]
        == "stage-certified-transpose-only"
    )
    assert "--independent-fd-certificate" in calls[1]
    recovered = LumericalEvaluationDriver(runtime)
    recovered.bind_independent_fd_certificate(first["result_path"])
    assert recovered._independent_fd_validation_pending is False
    assert recovered._independent_fd_certificate == first["result_path"]


def test_completed_cache_is_bound_to_state_and_runtime_contract(tmp_path: Path) -> None:
    runtime = OptimizerRuntime(
        output_root=tmp_path / "attempt",
        source_calibration={},
        gpu_index=7,
        threads=8,
        accelerator_policy="b200",
        beta=4.0,
    )
    driver = LumericalEvaluationDriver(runtime)
    evaluation = tmp_path / "cached"
    evaluation.mkdir()
    gradients = evaluation / "signed_projected_gradients.npz"
    np.savez_compressed(
        gradients,
        gradient_Ea_projected_A=np.ones(CONTRACT.design_node_shape),
        gradient_Eb_projected_A=-np.ones(CONTRACT.design_node_shape),
    )
    state_hash = "state-hash"
    result = {
        "passed": True,
        "density_state": {"density_state_sha256": state_hash},
        "evaluation_contract": driver._evaluation_contract,
        "evaluation_contract_sha256": driver._evaluation_contract_sha256,
        "artifacts": {"gradients": artifact(gradients)},
    }
    result_path = evaluation / "evaluation_result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    loaded = driver._load_completed(evaluation, expected_state_hash=state_hash)
    assert loaded["cache_hit"] is True
    result["evaluation_contract"] = {**driver._evaluation_contract, "threads": 99}
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(RuntimeError, match="cached evaluation contract failed"):
        driver._load_completed(evaluation, expected_state_hash=state_hash)


def test_full_chain_adfd_uses_current_only_perturbations(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = OptimizerRuntime(
        output_root=tmp_path / "attempt",
        source_calibration={},
        gpu_index=7,
        threads=8,
        accelerator_policy="b200",
        beta=4.0,
    )
    driver = LumericalEvaluationDriver(runtime)
    latent = np.full(CONTRACT.design_node_shape, 0.5)
    x = np.linspace(-1.0, 1.0, CONTRACT.design_node_shape[0])[:, None]
    y = np.linspace(-1.0, 1.0, CONTRACT.design_node_shape[1])[None, :]
    gradient_a = 1.0e-12 * (1.2 + 0.2 * x - 0.1 * y)
    gradient_b = -1.0e-12 * (1.1 - 0.1 * x + 0.2 * y)

    def currents(value: np.ndarray) -> dict[str, float]:
        rho = OPTIMIZER_250NM_MAPPING.physical(value, runtime.beta)
        return {
            "Ea": float(2.0e-9 + np.vdot(gradient_a, rho)),
            "Eb": float(-2.0e-9 + np.vdot(gradient_b, rho)),
        }

    def fake_current_only(value: np.ndarray, output: Path) -> dict[str, object]:
        output.mkdir(parents=True)
        record = {"passed": True, "currents_A": currents(value)}
        (output / "current_only_result.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
        return record

    monkeypatch.setattr(driver, "_evaluate_currents_only", fake_current_only)
    baseline = {
        "passed": True,
        "evaluation_contract": driver._evaluation_contract,
        "currents_A": currents(latent),
        "gradient_Ea_projected_A": gradient_a,
        "gradient_Eb_projected_A": gradient_b,
    }
    result = driver.audit_full_chain_latent_adfd(
        latent, baseline, tmp_path / "adfd", validation_scope="stage-entry"
    )
    assert result["passed"] is True
    assert result["solver_counts"]["additional_Lumerical_forward"] == 4
    assert result["solver_counts"]["additional_Lumerical_adjoint"] == 0
    assert result["solver_counts"]["additional_layout_only_Jacobian_sessions"] == 0


def test_full_chain_adfd_step_scales_after_beta_16() -> None:
    assert full_chain_adfd_step(1.0) == pytest.approx(0.0025)
    assert full_chain_adfd_step(16.0) == pytest.approx(0.0025)
    assert full_chain_adfd_step(32.0) == pytest.approx(0.00125)
    assert full_chain_adfd_step(64.0) == pytest.approx(0.000625)
    assert full_chain_adfd_step(128.0) == pytest.approx(0.0003125)
