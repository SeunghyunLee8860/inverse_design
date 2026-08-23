#!/usr/bin/env python3
"""Audit discrete adjoint-current time staggering for the Eb field term."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.combined_4um import (
    CompiledOpticalRunner,
    combined_gradient,
    discrete_au_susceptibility,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.material_fraction import (
    d_au_material_fraction_drho,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_on_fixed_tairte4_validation.fdtdx_two_solve_adjoint import (
    harmonic_material_gradient,
)


HERE = Path(__file__).resolve().parent
OUT = HERE / "results_4um_adjoint_source_phase_diagnostic"
CALIBRATION = (
    HERE
    / "results_fdtdx_4um_source_calibration"
    / "fdtdx_4um_source_calibration.json"
)
REFERENCE = (
    HERE
    / "results_4um_eb_optical_gradient_diagnostic"
    / "eb_optical_gradient_decomposition.json"
)
EXPONENTS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def main() -> None:
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    scale = float(calibration["reporting_incident_power_W"]) / float(
        calibration["common_reference_incident_power_W"]
    )
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    field_fd = float(reference["rows"][-1]["FD"]["field_A"])
    rho = np.full(CONTRACT.design_shape, 0.5, dtype=np.float64)
    runner = CompiledOpticalRunner.create("Eb", rho)
    base = combined_gradient(runner, rho, scale, 0)
    direction = np.asarray(base["gradient_total_A"], dtype=np.float64)
    direction /= np.max(np.abs(direction))
    fields = np.asarray(base["fields"]["au"])
    production_phase = complex(base["adjoint_source_phase_factor"])
    profile = np.asarray(base["adjoint_profile"]) / production_phase
    jnp = runner.model["jnp"]
    d_strength = jnp.broadcast_to(
        jnp.asarray(d_au_material_fraction_drho(rho))[:, :, None], fields.shape[1:]
    )
    d_epsilon = jnp.broadcast_to(
        d_strength[None] * discrete_au_susceptibility(runner), fields.shape
    )
    theta = float(runner.model["omega_rad_s"]) * float(
        runner.model["config"].time_step_duration
    )
    rows = []
    for exponent in EXPONENTS:
        phase_factor = np.exp(1j * exponent * theta)
        if exponent == 1.0:
            adjoint_output = base["adjoint_output"]
            runtime_s = float(base["adjoint_s"])
        else:
            adjoint_output, runtime_s = runner.run_adjoint(
                rho, profile * phase_factor
            )
        adjoint_field = adjoint_output.detector_states["au_late"]["phasor"][0, 0]
        field_voxel = harmonic_material_gradient(
            jnp.asarray(fields),
            adjoint_field,
            d_epsilon,
            float(runner.model["omega_rad_s"]),
            float(runner.model["config"].time_step_duration),
        ) * jnp.asarray(runner.volumes["au"])
        field_gradient = (
            np.asarray(jnp.sum(field_voxel, axis=(0, 3)), dtype=np.float64) * scale
        )
        directional = float(np.vdot(field_gradient, direction))
        rows.append(
            {
                "phase_exponent_k_in_exp_i_k_omega_dt": exponent,
                "phase_factor": [phase_factor.real, phase_factor.imag],
                "field_AD_A": directional,
                "field_FD_A": field_fd,
                "relative_error": abs(directional - field_fd) / abs(field_fd),
                "adjoint_execution_seconds": runtime_s,
            }
        )
    summary = {
        "status": "DIAGNOSTIC_ONLY_DISCRETE_TIME_STAGGERING",
        "polarization": "Eb",
        "omega_dt_rad": theta,
        "rows": rows,
        "best_candidate": min(rows, key=lambda row: row["relative_error"]),
        "no_empirical_amplitude_rescaling": True,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "adjoint_source_phase_diagnostic.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
