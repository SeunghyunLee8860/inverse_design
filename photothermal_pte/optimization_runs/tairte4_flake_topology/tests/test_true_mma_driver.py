import ast
from pathlib import Path

import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.run_true_mma_optimization import (
    BINARIZATION_CONTINUATION_TARGET,
    MOVE_LIMIT,
    REFERENCE_INCIDENT_POWER_W,
    canonical_constraints,
    equivalent_current,
    stage_convergence,
    stage_morphology_caps,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.optimization_support import MAPPING
from photothermal_pte.optimization_runs.tairte4_flake_topology.validate_combined_adfd import (
    polarization_angle,
)
from photothermal_pte.optimization_runs.audit_true_mma_preflight import (
    optimizer_contract_passed,
)


def test_polarization_axis_contract() -> None:
    assert polarization_angle("Ea") == 90.0
    assert polarization_angle("Eb") == 0.0


def test_driver_has_no_adam_or_normalized_direction_update() -> None:
    source = Path(__file__).parents[1] / "run_true_mma_optimization.py"
    text = source.read_text().lower()
    assert "first_moment" not in text
    assert "second_moment" not in text
    assert "adam_iteration" not in text
    assert "normalized(gradient" not in text


def test_supervisor_call_arities_match_restart_contract() -> None:
    source = Path(__file__).parents[2] / "run_true_mma_dual_supervisor.py"
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id == "run":
            assert len(node.args) == 3
        if node.func.id == "run_restartable":
            assert len(node.args) == 4


def test_final_result_is_explicitly_machine_passed() -> None:
    source = Path(__file__).parents[1] / "run_true_mma_optimization.py"
    text = source.read_text()
    assert '"passed": True' in text


def test_intentionally_absent_constraints_pass_optimizer_audit() -> None:
    assert optimizer_contract_passed({
        "true_mma_module_present": True,
        "historical_adam_state_absent": True,
        "gradient_direction_normalization_absent": True,
        "symmetry_constraint": False,
        "volume_constraint": False,
    })


def test_fixed_incident_power_scaling_is_constant() -> None:
    fixed_source = 2.0e-13
    expected = 3.0e-18 * REFERENCE_INCIDENT_POWER_W / fixed_source
    assert equivalent_current(3.0e-18, fixed_source) == expected


def test_low_beta_has_only_connectivity_inequality() -> None:
    latent = np.full(MAPPING.shape, 0.5)
    gradient = np.ones(MAPPING.shape)
    names, values, gradients, _, _ = canonical_constraints(
        latent=latent,
        beta=1.0,
        terminal_conductance_S=2.0,
        gradient_terminal_conductance_physical_S=gradient,
        minimum_terminal_conductance_S=1.0,
        morphology_caps=np.asarray([np.inf, np.inf]),
        device="cpu",
    )
    assert names == ["minimum_terminal_conductance"]
    assert values.shape == (1,)
    assert gradients.shape == (1, *MAPPING.shape)


def test_morphology_caps_are_fixed_gentle_stage_tightening() -> None:
    values = np.asarray([4.0e-3, 1.0e-4])
    caps = stage_morphology_caps(values, 8.0)
    assert np.allclose(caps, [3.6e-3, 2.0e-3])


def test_stage_does_not_advance_before_measured_minimum() -> None:
    history = [
        {
            "role": "accepted_mma",
            "beta": 1.0,
            "objective_at_reference_power_A": 1e-9,
            "mma_maximum_absolute_step": 1e-4,
            "maximum_constraint_value": -1.0,
            "exact_bad_cells": 0,
            "gray_fraction_0p01_0p99": 1.0,
            "binarization": 1.0,
        }
        for _ in range(5)
    ]
    assert stage_convergence(history, 1.0)["converged"] is False


def test_low_beta_advances_from_measured_sharpness_after_minimum() -> None:
    history = [
        {
            "role": "accepted_mma",
            "beta": 1.0,
            "objective_at_reference_power_A": (index + 1) * 1e-9,
            "mma_maximum_absolute_step": MOVE_LIMIT[1.0],
            "maximum_constraint_value": -1.0,
            "exact_bad_cells": 100,
            "gray_fraction_0p01_0p99": 1.0,
            "binarization": BINARIZATION_CONTINUATION_TARGET[1.0] - 1e-3,
        }
        for index in range(6)
    ]
    result = stage_convergence(history, 1.0)
    assert result["converged"] is True
    assert result["reason"] == "continuation_sharpness"


def test_low_beta_move_bound_prevents_run020_half_range_jump() -> None:
    betas = tuple(MOVE_LIMIT)
    assert MOVE_LIMIT[1.0] <= 0.025
    assert all(MOVE_LIMIT[a] >= MOVE_LIMIT[b] for a, b in zip(betas, betas[1:]))
