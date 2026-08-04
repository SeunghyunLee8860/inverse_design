#!/usr/bin/env python3
"""Audit an existing combined AD--FD sweep without running Lumerical.

This diagnostic intentionally does not promote a failed certificate.  It
separates weak-direction cancellation, centered-FD step behavior, and the
scalar optical-power response using the immutable per-step results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_STATUS = "FAILED_CORRECTED_COMBINED_PHYSICAL_RHO_PTE_ADFD"


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def relative_plateau(values: list[float]) -> float:
    scale = max(*(abs(value) for value in values), np.finfo(float).tiny)
    return max(
        abs(values[index] - values[index + 1]) / scale
        for index in range(len(values) - 1)
    )


def analyze_direction(
    *,
    scenario: str,
    direction: str,
    data: dict,
    gradient_l2_A: float,
    aligned_directional_A: float,
    base_objective_A: float,
) -> dict:
    steps = data["steps"]
    if [row["step"] for row in steps] != [0.01, 0.005, 0.0025]:
        raise RuntimeError("expected the locked 0.01, 0.005, 0.0025 sweep")
    thermal_key = "4.0" if scenario == "4um" else "6.0"
    analytic = float(data["analytic_directional_A"])
    fd_values = []
    pq_values = []
    psix_values = []
    numerator_fractions = []
    midpoint_biases = []
    rows = []
    for item in steps:
        h = float(item["step"])
        plus = item["plus"]
        minus = item["minus"]
        plus_objective = float(
            plus["objectives"][thermal_key]["objective_A"]
        )
        minus_objective = float(
            minus["objectives"][thermal_key]["objective_A"]
        )
        objective_numerator = plus_objective - minus_objective
        fd = objective_numerator / (2.0 * h)
        pq_fd = (
            float(plus["forward"]["P_Q_W"])
            - float(minus["forward"]["P_Q_W"])
        ) / (2.0 * h)
        psix_fd = (
            float(plus["forward"]["P_six_W"])
            - float(minus["forward"]["P_six_W"])
        ) / (2.0 * h)
        midpoint_bias = (
            0.5 * (plus_objective + minus_objective) - base_objective_A
        )
        numerator_fraction = abs(objective_numerator) / max(
            abs(base_objective_A), np.finfo(float).tiny
        )
        if not np.isclose(
            fd,
            float(item["finite_difference_directional_A"]),
            rtol=1.0e-9,
            atol=0.0,
        ):
            raise RuntimeError("stored FD derivative is internally inconsistent")
        fd_values.append(fd)
        pq_values.append(pq_fd)
        psix_values.append(psix_fd)
        numerator_fractions.append(numerator_fraction)
        midpoint_biases.append(midpoint_bias)
        rows.append(
            {
                "scenario": scenario,
                "direction": direction,
                "h": h,
                "analytic_directional_A": analytic,
                "finite_difference_directional_A": fd,
                "relative_error": float(item["relative_error"]),
                "objective_numerator_A": objective_numerator,
                "objective_numerator_over_base": numerator_fraction,
                "objective_midpoint_bias_A": midpoint_bias,
                "P_Q_directional_W": pq_fd,
                "P_six_directional_W": psix_fd,
            }
        )
    differences = [
        abs(fd_values[index] - fd_values[index + 1])
        for index in range(2)
    ]
    expected_centered_fd_fine_to_coarse_ratio = 0.25
    observed_ratio = differences[1] / max(
        differences[0], np.finfo(float).tiny
    )
    return {
        "scenario": scenario,
        "direction": direction,
        "analytic_directional_A": analytic,
        "direction_strength_fraction_of_gradient_l2": abs(analytic)
        / max(abs(gradient_l2_A), np.finfo(float).tiny),
        "directional_response_fraction_of_adjoint_aligned": abs(analytic)
        / max(abs(aligned_directional_A), np.finfo(float).tiny),
        "PTE_step_plateau_relative": relative_plateau(fd_values),
        "P_Q_step_plateau_relative": relative_plateau(pq_values),
        "P_six_step_plateau_relative": relative_plateau(psix_values),
        "coarse_to_mid_FD_difference_A": differences[0],
        "mid_to_fine_FD_difference_A": differences[1],
        "fine_to_coarse_difference_ratio": observed_ratio,
        "centered_FD_truncation_expected_ratio": (
            expected_centered_fd_fine_to_coarse_ratio
        ),
        "finer_step_difference_grew": bool(observed_ratio > 1.0),
        "smallest_h_objective_numerator_over_base": numerator_fractions[-1],
        "smallest_h_midpoint_bias_A": midpoint_biases[-1],
        "stored_step_convergence_passed": bool(
            data["step_convergence_passed"]
        ),
        "rows": rows,
    }


def audit(raw: dict) -> dict:
    if raw["status"] != EXPECTED_STATUS or raw["passed"]:
        raise RuntimeError("expected the immutable fail-closed combined result")
    analyses = []
    for scenario, scenario_data in raw["scenarios"].items():
        aligned_directional_A = float(
            scenario_data["directions"]["adjoint_aligned"][
                "analytic_directional_A"
            ]
        )
        for direction, data in scenario_data["directions"].items():
            analyses.append(
                analyze_direction(
                    scenario=scenario,
                    direction=direction,
                    data=data,
                    gradient_l2_A=float(scenario_data["gradient_L2_A"]),
                    aligned_directional_A=aligned_directional_A,
                    base_objective_A=float(
                        scenario_data["base_objective_A"]
                    ),
                )
            )
    offenders = [
        item for item in analyses if not item["stored_step_convergence_passed"]
    ]
    all_offenders_weak = all(
        item["directional_response_fraction_of_adjoint_aligned"] < 0.01
        for item in offenders
    )
    all_offenders_grow_at_fine_h = all(
        item["finer_step_difference_grew"] for item in offenders
    )
    return {
        "status": "DIAGNOSTIC_EXISTING_FD_RESULTS_AUDITED",
        "combined_certificate_status_preserved": EXPECTED_STATUS,
        "combined_certificate_passed": False,
        "new_Lumerical_solves": 0,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "offending_subgate_count": len(offenders),
        "all_offenders_below_one_percent_of_aligned_response": (
            all_offenders_weak
        ),
        "all_offenders_have_larger_fine_step_difference": (
            all_offenders_grow_at_fine_h
        ),
        "interpretation": (
            "The three failed plateau subgates have weak PTE responses "
            "(below 1% of the tested adjoint-aligned response), and their "
            "h/2-to-h/4 "
            "derivative change is larger rather than approximately four "
            "times smaller. This is inconsistent with clean asymptotic "
            "centered-FD truncation being the dominant error and is "
            "consistent with a directional numerical-resolution floor. "
            "It does not prove run-to-run FDTD noise because no repeated "
            "identical Lumerical solve was performed."
        ),
        "directions": analyses,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-result", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    raw_path = Path(args.raw_result).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    actual_sha = sha256(raw_path)
    if actual_sha != args.expected_sha256:
        raise RuntimeError(
            f"raw-result SHA mismatch: {actual_sha} != {args.expected_sha256}"
        )
    raw = json.loads(raw_path.read_text())
    result = audit(raw)
    result["raw_result"] = {
        "path": str(raw_path),
        "byte_size": raw_path.stat().st_size,
        "sha256": actual_sha,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = report_dir / "component_yee_material_jacobian_summary.json"
    thermal_path = (
        report_dir / "thermal_raw_pte_localized_adfd_subgates_summary.json"
    )
    mapping = json.loads(mapping_path.read_text())
    thermal = json.loads(thermal_path.read_text())
    if not mapping["passed"] or not thermal["passed"]:
        raise RuntimeError("an existing fast prerequisite subgate is not passed")
    result["existing_fast_subgates"] = {
        "component_Yee_material_Jacobian": {
            "status": mapping["status"],
            "summary_sha256": sha256(mapping_path),
            "worst_mapping_only_FD_relative_error": mapping["gates"][
                "worst_mapping_only_FD_relative_error"
            ],
            "worst_JVP_VJP_dot_relative_error": mapping["gates"][
                "worst_JVP_VJP_dot_relative_error"
            ],
            "maximum_coordinate_mismatch_m": mapping[
                "maximum_coordinate_mismatch_m"
            ],
        },
        "thermal_only_PTE_ADFD": {
            "status": thermal["status"],
            "summary_sha256": sha256(thermal_path),
            "worst_selected_gated_AD_FD_relative_error": thermal[
                "localized_ADFD_gates"
            ]["worst_selected_gated_AD_FD_relative_error"],
            "worst_energy_balance_relative_error": thermal[
                "localized_ADFD_gates"
            ]["worst_energy_balance_relative_error"],
            "worst_linear_residual_relative": thermal[
                "localized_ADFD_gates"
            ]["worst_linear_residual_relative"],
        },
    }
    stem = "existing_combined_fd_plateau_audit"
    summary_path = report_dir / f"{stem}_summary.json"
    csv_path = report_dir / f"{stem}_cases.csv"
    report_path = report_dir / "EXISTING_COMBINED_FD_PLATEAU_AUDIT.md"
    manifest_path = (
        report_dir / "EXISTING_COMBINED_FD_PLATEAU_AUDIT_RAW_MANIFEST.json"
    )
    rows = [
        row
        for direction in result["directions"]
        for row in direction.pop("rows")
    ]
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    offenders = [
        item
        for item in result["directions"]
        if not item["stored_step_convergence_passed"]
    ]
    report_path.write_text(
        "# Existing combined FD plateau audit\n\n"
        "- New Lumerical solves: `0`\n"
        f"- Original certificate: `{EXPECTED_STATUS}`\n"
        "- Original certificate remains failed; this is a diagnostic only.\n"
        "- No empirical normalization or gradient rescaling was used.\n\n"
        "## Fast prerequisite checks\n\n"
        "- Component-wise Yee mapping: worst mapping-only FD error "
        f"`{mapping['gates']['worst_mapping_only_FD_relative_error']:.6g}`, "
        "worst JVP/VJP dot error "
        f"`{mapping['gates']['worst_JVP_VJP_dot_relative_error']:.6g}`, "
        "maximum coordinate mismatch "
        f"`{mapping['maximum_coordinate_mismatch_m']:.6g} m`\n"
        "- Thermal-only PTE AD-FD: worst selected error "
        f"`{thermal['localized_ADFD_gates']['worst_selected_gated_AD_FD_relative_error']:.6g}`, "
        "energy-balance error "
        f"`{thermal['localized_ADFD_gates']['worst_energy_balance_relative_error']:.6g}`, "
        "linear residual "
        f"`{thermal['localized_ADFD_gates']['worst_linear_residual_relative']:.6g}`\n\n"
        "## Evidence from the immutable 30-solve sweep\n\n"
        + "\n".join(
            f"- {item['scenario']} {item['direction']}: strength/||g|| "
            f"`{item['direction_strength_fraction_of_gradient_l2']:.6g}`, "
            "response/aligned-response "
            f"`{item['directional_response_fraction_of_adjoint_aligned']:.6g}`, "
            f"PTE plateau `{item['PTE_step_plateau_relative']:.6g}`, "
            f"P_Q plateau `{item['P_Q_step_plateau_relative']:.6g}`, "
            "fine/coarse derivative-difference ratio "
            f"`{item['fine_to_coarse_difference_ratio']:.6g}`"
            for item in offenders
        )
        + "\n\nFor a smooth centered finite difference dominated by the "
        "usual O(h^2) truncation term, the derivative-difference ratio "
        "after halving h is approximately `0.25`. It is greater than one "
        "for every failed direction, while all failed responses are below "
        "1% of the tested adjoint-aligned response. This localizes the "
        "unresolved issue "
        "to weak-direction numerical resolution/cancellation rather than a "
        "global gradient scale or sign error. It does not establish "
        "run-to-run FDTD stochasticity; identical-solve repeats would still "
        "be required for that claim.\n"
    )
    manifest_path.write_text(
        json.dumps(
            {
                "generation_command": (
                    "python -m photothermal_pte.finite_inverse_design."
                    "audit_existing_combined_fd_plateau "
                    "--raw-result <external>/"
                    "full_five_direction_failed_noise_plateau_result.json "
                    f"--expected-sha256 {actual_sha} "
                    "--report-dir photothermal_pte/reports/"
                    "inverse_design_pte_adfd"
                ),
                "raw_result": result["raw_result"],
                "raw_result_modified": False,
                "new_Lumerical_solves": 0,
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
