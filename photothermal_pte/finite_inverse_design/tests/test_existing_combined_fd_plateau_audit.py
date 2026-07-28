from photothermal_pte.finite_inverse_design.audit_existing_combined_fd_plateau import (
    analyze_direction,
)


def test_analyze_direction_detects_fine_step_resolution_floor() -> None:
    analytic = 1.0e-3
    base = 1.0
    fd_values = [1.001e-3, 1.002e-3, 1.010e-3]
    steps = []
    for h, fd in zip([0.01, 0.005, 0.0025], fd_values, strict=True):
        numerator = 2.0 * h * fd
        forward = {"P_Q_W": 1.0 + numerator, "P_six_W": 1.0 + numerator}
        backward = {"P_Q_W": 1.0, "P_six_W": 1.0}
        steps.append(
            {
                "step": h,
                "finite_difference_directional_A": fd,
                "relative_error": abs(fd - analytic) / abs(analytic),
                "plus": {
                    "forward": forward,
                    "objectives": {"4.0": {"objective_A": base + numerator}},
                },
                "minus": {
                    "forward": backward,
                    "objectives": {"4.0": {"objective_A": base}},
                },
            }
        )

    result = analyze_direction(
        scenario="4um",
        direction="weak",
        data={
            "analytic_directional_A": analytic,
            "steps": steps,
            "step_convergence_passed": False,
        },
        gradient_l2_A=1.0,
        aligned_directional_A=1.0,
        base_objective_A=base,
    )

    assert result["direction_strength_fraction_of_gradient_l2"] == 1.0e-3
    assert result["directional_response_fraction_of_adjoint_aligned"] == 1.0e-3
    assert result["finer_step_difference_grew"]
    assert result["fine_to_coarse_difference_ratio"] > 1.0
