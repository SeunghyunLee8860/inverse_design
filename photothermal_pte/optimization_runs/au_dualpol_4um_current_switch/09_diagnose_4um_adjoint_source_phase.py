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
    audit as material_fraction_audit,
    d_au_material_fraction_drho,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_4um_model import (
    MATERIAL_JSON,
    grid_edges_sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_on_fixed_tairte4_validation.fdtdx_two_solve_adjoint import (
    harmonic_material_gradient,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.validation_provenance import (
    OPTICAL_DECOMPOSITION_STATUS,
    array_sha256,
    load_current_source_calibration,
    require_material_fraction,
    require_single_visible_gpu,
    require_status,
    sha256,
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
    require_single_visible_gpu()
    calibration = load_current_source_calibration(CALIBRATION)
    scale = float(calibration["reporting_incident_power_W"]) / float(
        calibration["common_reference_incident_power_W"]
    )
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    require_status(reference, OPTICAL_DECOMPOSITION_STATUS, "Eb FD reference")
    require_material_fraction(reference, "Eb FD reference")
    if reference.get("source_calibration_sha256") != sha256(CALIBRATION):
        raise RuntimeError("Eb FD reference is not linked to the current source calibration")
    if reference.get("material_contract_sha256") != sha256(MATERIAL_JSON):
        raise RuntimeError("Eb FD reference is not linked to the current material contract")
    if reference.get("optical_grid_edges_sha256") != grid_edges_sha256():
        raise RuntimeError("Eb FD reference was generated on a different optical grid")
    if reference.get("polarization") != "Eb" or float(reference.get("density", -1.0)) != 0.5:
        raise RuntimeError("Eb FD reference polarization/density contract changed")
    field_fd = float(reference["rows"][-1]["FD"]["field_A"])
    rho = np.full(CONTRACT.design_shape, 0.5, dtype=np.float64)
    runner = CompiledOpticalRunner.create("Eb", rho)
    base = combined_gradient(runner, rho, scale, 0)
    direction = np.asarray(base["gradient_total_A"], dtype=np.float64)
    direction /= np.max(np.abs(direction))
    if reference.get("direction_sha256") != array_sha256(direction):
        raise RuntimeError("Eb FD direction is stale or differs from the current combined gradient")
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
        "au_material_fraction": material_fraction_audit(),
        "source_calibration_sha256": sha256(CALIBRATION),
        "material_contract_sha256": sha256(MATERIAL_JSON),
        "reference_sha256": sha256(REFERENCE),
        "optical_grid_edges_sha256": grid_edges_sha256(),
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
