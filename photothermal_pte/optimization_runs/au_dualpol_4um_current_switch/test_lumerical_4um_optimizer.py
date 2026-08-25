from __future__ import annotations

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_optimizer import (
    CURRENT_SCALE_A,
    LumericalEvaluationDriver,
    SmokeEpigraphProblem,
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

    assert result[0] == (-9.0 * CURRENT_SCALE_A - point["current_a_A"]) / CURRENT_SCALE_A
    assert result[1] == (-9.0 * CURRENT_SCALE_A + point["current_b_A"]) / CURRENT_SCALE_A
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
    problem = SmokeEpigraphProblem(
        evaluate, beta=4.0, dfm_caps=np.asarray((1.0, 1.0))
    )
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
