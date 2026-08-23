#!/usr/bin/env python3
"""Separate the Eb Maxwell field and explicit-loss directional derivatives."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.combined_4um import (
    CompiledOpticalRunner,
    combined_gradient,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_4um_model import (
    MATERIAL_JSON,
    grid_edges_sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.material_fraction import (
    audit as material_fraction_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.validation_provenance import (
    array_sha256,
    load_current_source_calibration,
    require_single_visible_gpu,
    sha256,
)


HERE = Path(__file__).resolve().parent
OUT = HERE / "results_4um_eb_optical_gradient_diagnostic"
CALIBRATION = (
    HERE
    / "results_fdtdx_4um_source_calibration"
    / "fdtdx_4um_source_calibration.json"
)
STEPS = (0.005, 0.0025)


def weighted_optical_objective(
    runner: CompiledOpticalRunner,
    output,
    rho_for_explicit_loss: np.ndarray,
    weights: dict[str, np.ndarray],
    scale: float,
) -> float:
    _, q = runner.fields_and_q(output, rho_for_explicit_loss, scale)
    value = 0.0
    for material in ("au", "tairte4"):
        value += float(
            np.vdot(weights[material], q[material] * runner.volumes[material])
        )
    return value


def relative_error(ad: float, fd: float) -> float:
    return abs(ad - fd) / max(abs(fd), np.finfo(float).tiny)


def main() -> None:
    require_single_visible_gpu()
    calibration = load_current_source_calibration(CALIBRATION)
    scale = float(calibration["reporting_incident_power_W"]) / float(
        calibration["common_reference_incident_power_W"]
    )
    rho = np.full(CONTRACT.design_shape, 0.5, dtype=np.float64)
    runner = CompiledOpticalRunner.create("Eb", rho)
    base = combined_gradient(runner, rho, scale, 0)
    direction = np.asarray(base["gradient_total_A"], dtype=np.float64)
    direction /= np.max(np.abs(direction))
    weights = base["native_weights_A_W"]
    base_output = base["optical_output"]

    ad = {
        "field_A": float(np.vdot(base["gradient_optical_field_A"], direction)),
        "explicit_loss_A": float(
            np.vdot(base["gradient_optical_direct_loss_A"], direction)
        ),
        "optical_total_A": float(np.vdot(base["gradient_optical_A"], direction)),
    }
    rows = []
    for step in STEPS:
        plus_rho = rho + step * direction
        minus_rho = rho - step * direction
        plus_output, plus_s = runner.run_forward(plus_rho)
        minus_output, minus_s = runner.run_forward(minus_rho)

        optical_plus = weighted_optical_objective(
            runner, plus_output, plus_rho, weights, scale
        )
        optical_minus = weighted_optical_objective(
            runner, minus_output, minus_rho, weights, scale
        )
        field_plus = weighted_optical_objective(
            runner, plus_output, rho, weights, scale
        )
        field_minus = weighted_optical_objective(
            runner, minus_output, rho, weights, scale
        )
        loss_plus = weighted_optical_objective(
            runner, base_output, plus_rho, weights, scale
        )
        loss_minus = weighted_optical_objective(
            runner, base_output, minus_rho, weights, scale
        )
        fd = {
            "field_A": (field_plus - field_minus) / (2.0 * step),
            "explicit_loss_A": (loss_plus - loss_minus) / (2.0 * step),
            "optical_total_A": (optical_plus - optical_minus) / (2.0 * step),
        }
        rows.append(
            {
                "step": step,
                "AD": ad,
                "FD": fd,
                "relative_error": {
                    key: relative_error(ad[key], fd[key]) for key in ad
                },
                "forward_seconds": {"plus": plus_s, "minus": minus_s},
                "decomposition_closure_relative": abs(
                    fd["optical_total_A"]
                    - (fd["field_A"] + fd["explicit_loss_A"])
                )
                / max(abs(fd["optical_total_A"]), np.finfo(float).tiny),
            }
        )

    previous = np.asarray(
        base["adjoint_output"].detector_states["au_previous"]["phasor"][0, 0]
    )
    late = np.asarray(
        base["adjoint_output"].detector_states["au_late"]["phasor"][0, 0]
    )
    susceptibility_actual = complex(base["discrete_au_susceptibility"])
    susceptibility_target = complex(base["target_au_susceptibility"])
    summary = {
        "status": "DIAGNOSTIC_ONLY_NOT_AN_OPTIMIZATION_GATE",
        "polarization": "Eb",
        "density": 0.5,
        "direction": "normalized combined-gradient aligned",
        "direction_sha256": array_sha256(direction),
        "au_material_fraction": material_fraction_audit(),
        "source_calibration_sha256": sha256(CALIBRATION),
        "material_contract_sha256": sha256(MATERIAL_JSON),
        "optical_grid_edges_sha256": grid_edges_sha256(),
        "rows": rows,
        "adjoint_previous_to_late_field_relative": float(
            np.linalg.norm(late - previous) / max(np.linalg.norm(late), 1e-30)
        ),
        "discrete_au_susceptibility": [
            susceptibility_actual.real,
            susceptibility_actual.imag,
        ],
        "target_au_susceptibility": [
            susceptibility_target.real,
            susceptibility_target.imag,
        ],
        "discrete_susceptibility_relative_difference": abs(
            susceptibility_actual - susceptibility_target
        )
        / abs(susceptibility_target),
        "no_empirical_rescaling": True,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "eb_optical_gradient_decomposition.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Eb optical-gradient decomposition diagnostic",
        "",
        "Status: `DIAGNOSTIC_ONLY_NOT_AN_OPTIMIZATION_GATE`",
        "",
        "The fixed baseline thermal-source adjoint weights are held constant while",
        "the Maxwell field-mediated and explicit Au-loss derivatives are separated.",
        "No empirical normalization or gradient rescaling is applied.",
        "",
        "| h | field AD/FD error | explicit-loss AD/FD error | optical-total AD/FD error | FD decomposition closure |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['step']:.4g} | {100*row['relative_error']['field_A']:.6f}% "
            f"| {100*row['relative_error']['explicit_loss_A']:.6f}% "
            f"| {100*row['relative_error']['optical_total_A']:.6f}% "
            f"| {100*row['decomposition_closure_relative']:.6f}% |"
        )
    report.extend(
        [
            "",
            f"Actual float32-ADE susceptibility differs from the requested value by "
            f"{100*summary['discrete_susceptibility_relative_difference']:.6f}%.",
        ]
    )
    (OUT / "EB_OPTICAL_GRADIENT_DECOMPOSITION.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
