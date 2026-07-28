#!/usr/bin/env python3
"""Publish the fail-closed corrected five-direction combined diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> dict:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return {
        "path": str(path.resolve()),
        "byte_size": path.stat().st_size,
        "sha256": hasher.hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-result", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    raw_path = Path(args.raw_result).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    raw = json.loads(raw_path.read_text())
    if raw["status"] != "FAILED_CORRECTED_COMBINED_PHYSICAL_RHO_PTE_ADFD":
        raise RuntimeError("expected the immutable failed five-direction result")
    if raw["passed"] or raw["empirical_normalization"] or raw["gradient_rescaling"]:
        raise RuntimeError("invalid fail-closed provenance")
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scenario, scenario_data in raw["scenarios"].items():
        for direction, direction_data in scenario_data["directions"].items():
            for step in direction_data["steps"]:
                rows.append(
                    {
                        "scenario": scenario,
                        "direction": direction,
                        "step": step["step"],
                        "analytic_directional_A": direction_data[
                            "analytic_directional_A"
                        ],
                        "finite_difference_directional_A": step[
                            "finite_difference_directional_A"
                        ],
                        "relative_error": step["relative_error"],
                        "step_plateau_relative": direction_data[
                            "step_plateau_relative"
                        ],
                        "step_plateau_limit": direction_data[
                            "step_plateau_relative_limit"
                        ],
                        "step_convergence_passed": direction_data[
                            "step_convergence_passed"
                        ],
                    }
                )

    failed = [
        {
            "scenario": scenario,
            "direction": direction,
            "step_plateau_relative": data["step_plateau_relative"],
            "step_plateau_limit": data["step_plateau_relative_limit"],
        }
        for scenario, scenario_data in raw["scenarios"].items()
        for direction, data in scenario_data["directions"].items()
        if not data["step_convergence_passed"]
    ]
    summary = {
        "status": raw["status"],
        "passed": False,
        "raw_result": digest(raw_path),
        "gates": raw["gates"],
        "failed_step_plateau_subgates": failed,
        "interpretation": (
            "All requested AD-FD, multidirection, angle, closure, mapping, "
            "energy, residual, and transpose limits pass. The full status "
            "remains failed because central-4um and fixed-seed-random "
            "centered-FD derivatives do not meet the 0.1% step plateau."
        ),
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "gray_law_sensitivity_run": False,
        "full_latent_adfd_run": False,
        "optimization_run": False,
        "next_gate": "REDUCE_OFFENDING_DIRECTION_FDTD_FD_NOISE_FLOOR",
    }
    stem = "corrected_combined_five_direction_diagnostic"
    (report_dir / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (report_dir / f"{stem}_cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (report_dir / "CORRECTED_COMBINED_FIVE_DIRECTION_DIAGNOSTIC_REPORT.md").write_text(
        "# Corrected combined five-direction diagnostic\n\n"
        f"- Status: `{raw['status']}`\n"
        "- Full certificate: **not validated**\n"
        f"- Worst strong error: `{raw['gates']['worst_strong_direction_relative_error']:.8g}`\n"
        f"- Worst multidirection normalized error: `{raw['gates']['worst_multidirection_normalized_error']:.8g}`\n"
        f"- Worst subspace angle: `{raw['gates']['worst_directional_subspace_gradient_angle_deg']:.8g} deg`\n"
        f"- Optical closure: `{raw['gates']['worst_optical_closure_relative_error']:.8g}`\n"
        f"- Q mapping: `{raw['gates']['worst_Q_mapping_relative_error']:.8g}`\n"
        f"- Thermal energy balance: `{raw['gates']['worst_thermal_energy_balance_relative_error']:.8g}`\n"
        f"- Linear residual: `{raw['gates']['worst_linear_residual_relative']:.8g}`\n"
        f"- Mapping transpose: `{raw['gates']['mapping_transpose_relative_error']:.8g}`\n\n"
        "## Unresolved subgate\n\n"
        + "\n".join(
            f"- {item['scenario']} {item['direction']}: plateau "
            f"`{item['step_plateau_relative']:.8g}` > "
            f"`{item['step_plateau_limit']:.8g}`"
            for item in failed
        )
        + "\n\nThe small directional derivatives reach the current FDTD "
        "finite-difference noise floor. No normalization or gradient "
        "rescaling was applied. Gray-law sensitivity, full latent AD-FD, "
        "and optimization remain blocked.\n"
    )
    raw_files = [
        digest(path)
        for path in sorted(raw_path.parent.iterdir())
        if path.is_file() and path.suffix in {".fsp", ".json"}
    ]
    manifest = {
        "generation_command": (
            "python -m photothermal_pte.finite_inverse_design."
            "publish_corrected_combined_five_direction "
            "--raw-result <external>/full_five_direction_failed_noise_plateau_result.json "
            "--report-dir photothermal_pte/reports/inverse_design_pte_adfd"
        ),
        "raw_result": digest(raw_path),
        "raw_artifacts": raw_files,
        "raw_FSP_committed_to_git": False,
    }
    (
        report_dir
        / "CORRECTED_COMBINED_FIVE_DIRECTION_DIAGNOSTIC_RAW_MANIFEST.json"
    ).write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
