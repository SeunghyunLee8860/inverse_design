from __future__ import annotations

import json

import numpy as np

from photothermal_pte.optimization_runs.optimization_telemetry import (
    OptimizationTelemetry,
)


def test_iteration_telemetry_writes_history_design_and_constraints(tmp_path):
    logger = OptimizationTelemetry(tmp_path / "run", extent_um=(-1, 1, -1, 1))
    x = np.linspace(-1.0, 1.0, 9)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    latent = np.clip(0.5 + 0.2 * np.cos(np.pi * xx) * np.sin(np.pi * yy), 0, 1)
    filtered = 0.8 * latent + 0.1
    projected = 1.0 / (1.0 + np.exp(-4.0 * (filtered - 0.5)))
    directory = logger.record(
        iteration=0,
        latent=latent,
        filtered=filtered,
        projected=projected,
        metrics={
            "objective_scaled": 1.0,
            "best_feasible_scaled": 1.0,
            "current_A": 1.0e-9,
            "current_per_power_A_W": 2.0e-6,
            "g_solid": -1.0e-3,
            "g_void": -2.0e-3,
            "physical_gradient_l2": 3.0,
            "latent_gradient_l2": 2.0,
            "mma_step_l2": 0.1,
            "optical_closure_relative": 1.0e-3,
            "Q_mapping_relative_error": 2.0e-4,
            "thermal_energy_balance_relative": 3.0e-4,
            "linear_residual_relative": 1.0e-10,
        },
        constraint_fields={"solid penalty": latent**2, "void penalty": (1-latent)**2},
        physical_fields={"Q": latent, "temperature": filtered, "current": projected-0.5},
    )
    assert (directory / "record.json").is_file()
    assert (directory / "design.npz").is_file()
    assert (directory / "design.png").is_file()
    assert (directory / "constraint_fields.png").is_file()
    assert (directory / "physical_fields.png").is_file()
    assert (logger.live / "iteration_history.png").is_file()
    record = json.loads((directory / "record.json").read_text())
    assert 0.0 <= record["grayness"] <= 1.0
    assert len(logger.history_path.read_text().splitlines()) == 1


def test_telemetry_rejects_out_of_range_projected_density(tmp_path):
    logger = OptimizationTelemetry(tmp_path / "run", extent_um=(-1, 1, -1, 1))
    values = np.ones((3, 3))
    try:
        logger.record(
            iteration=0,
            latent=values,
            filtered=values,
            projected=1.1 * values,
            metrics={},
        )
    except ValueError as exc:
        assert "within [0,1]" in str(exc)
    else:
        raise AssertionError("out-of-range projected density was accepted")
