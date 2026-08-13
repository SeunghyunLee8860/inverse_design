from pathlib import Path

import nlopt
import numpy as np
import pytest

from photothermal_pte.optimization_runs.tairte4_flake_topology.run_true_mma_optimization import (
    BINARIZATION_CONTINUATION_TARGET,
    MOVE_LIMIT,
    REFERENCE_INCIDENT_POWER_W,
    TRANSIENT_LICENSE_MARKERS,
    canonical_constraints,
    discard_regenerable_evaluation_solver_files,
    equivalent_current,
    stage_convergence,
    stage_morphology_caps,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.run_ansys_dfm_ld_mma_optimization import (
    BETA_FACTOR,
    DFM_ACTIVATION_BETA,
    adaptive_plateau_diagnostic,
    beta_sequence,
    dfm_penalty_weight,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.optimization_support import (
    FILTER_RADIUS_M,
    MINIMUM_FEATURE_M,
    MAPPING,
    morphology_values_gradients,
)
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


def test_optimizer_discards_only_regenerable_per_evaluation_fsp() -> None:
    evaluator = Path(__file__).parents[1] / "evaluate_objective_gradient.py"
    evaluator_text = evaluator.read_text()
    driver = Path(__file__).parents[1] / "run_true_mma_optimization.py"
    driver_text = driver.read_text()
    assert '"--discard-fsp-after-success"' in driver_text
    assert "discard_regenerable_projects" in evaluator_text
    assert '"objective_gradient_npz_retained": True' in evaluator_text
    assert '"density_and_optimizer_checkpoints_retained": True' in evaluator_text


def test_hpc_checkout_failure_is_retryable() -> None:
    assert "unable to checkout the requested hpc license" in TRANSIENT_LICENSE_MARKERS
    assert "ansysli exited or could not read server port" in TRANSIENT_LICENSE_MARKERS
    assert (
        "could not match resource name provided or the resource may not be active"
        in TRANSIENT_LICENSE_MARKERS
    )


def test_solver_cleanup_preserves_checkpoints_and_logs(tmp_path: Path) -> None:
    project = tmp_path / "forward.fsp"
    engine_output = tmp_path / "forward" / "forward_output.h5"
    engine_output.parent.mkdir()
    checkpoint = tmp_path / "objective_gradient.npz"
    log = tmp_path / "forward_p0.log"
    project.write_bytes(b"project")
    engine_output.write_bytes(b"engine")
    checkpoint.write_bytes(b"checkpoint")
    log.write_text("log")

    discarded = discard_regenerable_evaluation_solver_files(tmp_path)

    assert {Path(row["path"]).name for row in discarded} == {
        "forward.fsp",
        "forward_output.h5",
    }
    assert not project.exists()
    assert not engine_output.exists()
    assert checkpoint.read_bytes() == b"checkpoint"
    assert log.read_text() == "log"


def test_ansys_driver_supports_audited_warm_restart() -> None:
    source = (
        Path(__file__).parents[1] / "run_ansys_dfm_ld_mma_optimization.py"
    ).read_text()
    assert '"--initial-latent-npz"' in source
    assert '"--recovery-append"' in source
    assert "prior asymptotes are not serializable" in source


def test_new_parallel_supervisor_pins_contact_geometry_and_distinct_gpus() -> None:
    source = Path(__file__).parents[2] / "run_ansys_dfm_parallel_optimizations.py"
    text = source.read_text()
    assert 'env["TAIRTE4_TOPOLOGY_GEOMETRY"] = "contact_anchored"' in text
    assert "args.gpu_ea == args.gpu_eb" in text


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


def test_filter_radius_is_inside_ansys_minimum_feature_interval() -> None:
    assert FILTER_RADIUS_M < MINIMUM_FEATURE_M
    assert MINIMUM_FEATURE_M < 2.0 * FILTER_RADIUS_M


def test_binary_like_bad_feature_keeps_nonzero_dfm_gradient() -> None:
    latent = np.zeros(MAPPING.shape, dtype=float)
    center = tuple(size // 2 for size in MAPPING.shape)
    latent[center[0] - 6:center[0] + 7, center[1]] = 1.0
    values, gradients, _ = morphology_values_gradients(
        latent, beta=16.0, device="cpu"
    )
    assert values[0] > 0.0
    assert np.linalg.norm(gradients[0]) > 1.0e-8
    assert np.all(np.isfinite(gradients))


def test_ks_max_aggregation_exposes_local_bad_feature() -> None:
    latent = np.zeros(MAPPING.shape, dtype=float)
    center = tuple(size // 2 for size in MAPPING.shape)
    latent[center[0] - 6:center[0] + 7, center[1]] = 1.0
    mean_values, _, _ = morphology_values_gradients(
        latent, beta=16.0, device="cpu", aggregation="mean"
    )
    ks_values, ks_gradients, _ = morphology_values_gradients(
        latent, beta=16.0, device="cpu", aggregation="ks_max"
    )
    assert ks_values[0] > mean_values[0]
    assert np.linalg.norm(ks_gradients[0]) > 1.0e-8
    assert np.all(np.isfinite(ks_gradients))


def test_ansys_style_beta_and_dfm_penalty_contract() -> None:
    schedule = beta_sequence()
    assert schedule[0] == 1.0
    assert np.allclose(np.asarray(schedule[1:-1]) / np.asarray(schedule[:-2]), BETA_FACTOR)
    assert schedule[-1] == 1024.0
    assert schedule[-4:] == (128.0, 256.0, 512.0, 1024.0)
    assert beta_sequence(8.5)[:4] == (8.5, 17.0, 34.0, 68.0)
    assert dfm_penalty_weight(2.0) == 0.0
    assert dfm_penalty_weight(DFM_ACTIVATION_BETA) == 10.0
    assert dfm_penalty_weight(8.0) == 40.0
    assert dfm_penalty_weight(16.0) == 160.0
    assert dfm_penalty_weight(32.0) == 640.0
    assert dfm_penalty_weight(64.0) == 2560.0
    assert dfm_penalty_weight(128.0) == 1.0e4


def _plateau_row(
    evaluation_id: int,
    *,
    fom: float = 1.0,
    rms_step: float = 1.0e-6,
    gray: float = 0.2,
    smooth: float = 1.0e-4,
    exact_bad: int = 10,
) -> dict[str, object]:
    return {
        "evaluation_id": evaluation_id,
        "objective_at_reference_power_A": fom,
        "rms_step_from_previous_evaluation": rms_step,
        "gray_fraction_0p01_0p99": gray,
        "smooth_solid_constraint": 0.5 * smooth,
        "smooth_void_constraint": 0.5 * smooth,
        "exact_bad_cells": exact_bad,
    }


def test_adaptive_plateau_requires_five_stage_evaluations() -> None:
    rows = [_plateau_row(i) for i in range(1, 5)]
    result = adaptive_plateau_diagnostic(rows)
    assert result["passed"] is False
    assert result["reason"] == "minimum_stage_evaluations_not_reached"


def test_adaptive_plateau_reports_constraint_stagnation() -> None:
    rows = [_plateau_row(i, exact_bad=20) for i in range(1, 6)]
    result = adaptive_plateau_diagnostic(rows)
    assert result["passed"] is True
    assert result["constraint_plateau"] is True


def test_adaptive_plateau_does_not_call_changing_constraints_stagnant() -> None:
    rows = [
        _plateau_row(i, smooth=1.0e-4 * (1.0 - 0.1 * i), exact_bad=30 - 4 * i)
        for i in range(1, 6)
    ]
    result = adaptive_plateau_diagnostic(rows)
    assert result["passed"] is True
    assert result["constraint_plateau"] is False


def test_adaptive_plateau_detects_joint_fom_design_and_gray_stagnation() -> None:
    rows = [_plateau_row(i, fom=1.0 + i * 1.0e-7) for i in range(1, 6)]
    result = adaptive_plateau_diagnostic(rows)
    assert result["passed"] is True
    assert result["window_evaluation_ids"] == [2, 3, 4, 5]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fom", 1.01),
        ("rms_step", 2.0e-4),
        ("gray", 0.21),
    ],
)
def test_adaptive_plateau_rejects_a_meaningfully_moving_metric(
    field: str, value: float
) -> None:
    rows = [_plateau_row(i) for i in range(1, 6)]
    kwargs = {field: value}
    rows[-1] = _plateau_row(5, **kwargs)
    result = adaptive_plateau_diagnostic(rows)
    assert result["passed"] is False
    assert result["reason"] == "stage_still_moving"


def test_ansys_style_driver_hard_constraints_are_explicit_opt_in() -> None:
    source = (
        Path(__file__).parents[1] / "run_ansys_dfm_ld_mma_optimization.py"
    ).read_text()
    assert "morphology_penalty_weight=penalty_weight" in source
    assert '"--hard-morphology-constraints"' in source
    assert 'action="store_true"' in source
    assert "hard_constraint_count = 2" in source
    assert "hard_constraint_count = 0" in source
    assert "0.0 if args.hard_morphology_constraints" in source
    assert '"--hard-rho-init"' in source
    assert "optimizer_rho_init = float(args.hard_rho_init)" in source
    assert '"fixed_caps": hard_fixed_caps.tolist()' in source
    assert "morphology_caps = hard_fixed_caps.copy()" in source
    assert 'morphology_aggregation="ks_max"' in source
    assert 'row.get("morphology_aggregation") != "ks_max"' in source
    assert "rho_init=optimizer_rho_init" in source
    assert '"hard_constraint_ccsa_parameters"' in source
    assert "restoration_block" not in source
    assert "manual_move_limit" in source


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


def test_low_beta_can_explicitly_disable_the_connectivity_inequality() -> None:
    latent = np.full(MAPPING.shape, 0.5)
    gradient = np.ones(MAPPING.shape)
    names, values, gradients, _, _ = canonical_constraints(
        latent=latent,
        beta=1.0,
        terminal_conductance_S=2.0,
        gradient_terminal_conductance_physical_S=gradient,
        minimum_terminal_conductance_S=None,
        morphology_caps=np.asarray([np.inf, np.inf]),
        device="cpu",
        include_terminal_conductance_constraint=False,
    )
    assert names == []
    assert values.shape == (0,)
    assert gradients.shape == (0, *MAPPING.shape)


def test_pure_driver_can_activate_500nm_solid_void_constraints_at_beta1() -> None:
    latent = np.full(MAPPING.shape, 0.5)
    gradient = np.ones(MAPPING.shape)
    names, values, gradients, _, _ = canonical_constraints(
        latent=latent,
        beta=1.0,
        terminal_conductance_S=2.0,
        gradient_terminal_conductance_physical_S=gradient,
        minimum_terminal_conductance_S=None,
        morphology_caps=np.asarray([1.0e-2, 1.0e-2]),
        device="cpu",
        include_terminal_conductance_constraint=False,
        morphology_start_beta=1.0,
    )
    assert names == ["500nm_solid_opening", "500nm_void_opening"]
    assert values.shape == (2,)
    assert gradients.shape == (2, *MAPPING.shape)


def test_pure_current_mma_uses_projection_scaled_native_initialization() -> None:
    from photothermal_pte.optimization_runs.tairte4_flake_topology.run_pure_current_ld_mma_optimization import (
        fixed_stage_morphology_caps,
        stage_mma_controls,
    )

    beta1 = stage_mma_controls(1.0)
    beta8 = stage_mma_controls(8.0)
    assert np.isclose(beta1["initial_step"], 0.025)
    assert np.isclose(beta1["rho_init"], 10.0)
    assert np.isclose(beta1["xtol_rel"], 1.0e-9)
    assert beta8["initial_step"] < beta1["initial_step"]
    assert beta8["rho_init"] > beta1["rho_init"]
    # The cap is the fixed physical target, not the current residual. Starting
    # infeasible is intentional: LD_MMA must actually reduce the violation.
    assert np.allclose(
        fixed_stage_morphology_caps(np.asarray([4.0e-3, 1.0e-4]), 8.0),
        [2.0e-3, 2.0e-3],
    )
    assert np.allclose(
        fixed_stage_morphology_caps(np.asarray([1.0e-3, 1.0e-4]), 1.0),
        [8.0e-3, 8.0e-3],
    )


def test_density_loader_canonicalizes_only_machine_roundoff(tmp_path) -> None:
    from photothermal_pte.optimization_runs.tairte4_flake_topology.evaluate_objective_gradient import (
        load_rho,
    )

    rho = np.full(MAPPING.shape, 0.5)
    rho[0, 0] = -5.551115123125783e-17
    path = tmp_path / "roundoff.npz"
    np.savez_compressed(path, rho=rho)
    loaded = load_rho(path)
    assert loaded[0, 0] == 0.0

    rho[0, 0] = -1.0e-8
    path = tmp_path / "physical_violation.npz"
    np.savez_compressed(path, rho=rho)
    with pytest.raises(RuntimeError, match=r"rho must be finite in \[0,1\]"):
        load_rho(path)


def test_pure_current_lumopt_style_fixed_budget_continuation_contract() -> None:
    from photothermal_pte.optimization_runs.tairte4_flake_topology.run_pure_current_ld_mma_optimization import (
        PURE_CURRENT_BETA_FACTOR,
        PURE_CURRENT_BETA_SCHEDULE,
        PURE_CURRENT_CONTINUATION_MAX_EVALUATIONS,
        PURE_CURRENT_MAX_BETA,
        PURE_CURRENT_NLOPT_FTOL_REL,
    )

    assert PURE_CURRENT_NLOPT_FTOL_REL == 1.0e-6
    assert PURE_CURRENT_BETA_FACTOR == 2.0
    assert PURE_CURRENT_BETA_SCHEDULE[:3] == (1.0, 2.0, 4.0)
    assert PURE_CURRENT_MAX_BETA == 128.0
    assert PURE_CURRENT_BETA_SCHEDULE[-1] == 128.0
    assert PURE_CURRENT_CONTINUATION_MAX_EVALUATIONS == 20


def test_pure_current_fixed_budget_disables_early_stops_for_every_stage() -> None:
    from photothermal_pte.optimization_runs.tairte4_flake_topology.run_pure_current_ld_mma_optimization import (
        stage_numerical_tolerances,
    )

    infeasible = stage_numerical_tolerances(np.asarray([-0.2, 0.1]))
    assert infeasible == {"ftol_rel": 0.0, "xtol_rel": 0.0}
    feasible = stage_numerical_tolerances(np.asarray([-0.2, -0.1]))
    assert feasible == {"ftol_rel": 0.0, "xtol_rel": 0.0}


def test_pure_current_final_gate_requires_binary_and_exact_500nm_geometry() -> None:
    from photothermal_pte.optimization_runs.tairte4_flake_topology.run_pure_current_ld_mma_optimization import (
        continuous_final_gate,
    )

    full_solid = np.ones(MAPPING.shape)
    assert continuous_final_gate(full_solid)["passed"] is True
    gray = np.full(MAPPING.shape, 0.5)
    result = continuous_final_gate(gray)
    assert result["passed"] is False
    assert result["gray_fraction_0p01_0p99"] == 1.0


def test_transient_license_failure_is_narrow_and_fail_closed(tmp_path) -> None:
    from photothermal_pte.optimization_runs.tairte4_flake_topology.run_true_mma_optimization import (
        TRANSIENT_FIELDREGION_PUTV_MARKER,
        transient_license_failure,
    )

    output = tmp_path / "evaluation"
    output.mkdir()
    (output / "fdtd.log").write_text(
        "Insufficient FlexNet Publisher license count of lum_fdtd_solve"
    )
    retryable, markers = transient_license_failure(output)
    assert retryable
    assert "insufficient flexnet publisher license count" in markers

    (output / "fdtd.log").write_text(
        "could not match resource name provided or the resource may not be active"
    )
    retryable, markers = transient_license_failure(output)
    assert retryable
    assert (
        "could not match resource name provided or the resource may not be active"
        in markers
    )

    (output / "fdtd.log").write_text("Maxwell energy-closure gate failed")
    retryable, markers = transient_license_failure(output)
    assert not retryable
    assert markers == []

    (output / "result.json").write_text(
        "LumApiError: Failed to put variable\n"
        "in import_named_fieldregion_profile\n"
        'fdtd.putv("ad_Ey", value)\n'
    )
    retryable, markers = transient_license_failure(output)
    assert retryable
    assert TRANSIENT_FIELDREGION_PUTV_MARKER in markers

    (output / "result.json").write_text("LumApiError: Failed to put variable")
    retryable, markers = transient_license_failure(output)
    assert not retryable
    assert markers == []


def test_pure_current_beta_promotion_uses_normal_stop_and_feasibility_not_raw_plateau() -> None:
    from photothermal_pte.optimization_runs.tairte4_flake_topology.run_pure_current_ld_mma_optimization import (
        continuation_stage_completion,
    )

    trial_history = [
        {
            "beta": 2.0,
            "maximum_constraint_value": -0.1,
            "objective_at_reference_power_A": value,
        }
        for value in (1.0e-9, 1.5e-9, 1.1e-9)
    ]
    completed = continuation_stage_completion(
        trial_history, 2.0, 1.0e-6, nlopt.MAXEVAL_REACHED, 20
    )
    assert completed["ready"] is True
    assert completed["raw_trial_objective_plateau_used_as_gate"] is False
    infeasible = [{**trial_history[-1], "maximum_constraint_value": 0.1}]
    assert continuation_stage_completion(
        infeasible, 2.0, 1.0e-6, nlopt.FTOL_REACHED, 20
    )["ready"] is False


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
