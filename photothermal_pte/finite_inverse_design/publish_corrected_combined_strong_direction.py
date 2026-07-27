#!/usr/bin/env python3
"""Publish the corrected combined strong-direction diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


STATUS = "DIAGNOSTIC_PASSED_CORRECTED_COMBINED_STRONG_DIRECTION_ADFD"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "byte_size": path.stat().st_size,
        "sha256": sha256(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-result", required=True)
    parser.add_argument("--immutable-monotonic-result", required=True)
    parser.add_argument("--report-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_path = Path(args.raw_result).expanduser().resolve()
    old_path = Path(args.immutable_monotonic_result).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    raw = json.loads(raw_path.read_text())
    if raw["status"] != STATUS or not raw["strong_direction_gate_passed"]:
        raise RuntimeError("strong-direction diagnostic did not pass")
    if raw["passed"] or not raw["strong_only_diagnostic"]:
        raise RuntimeError("strong-only result must not claim full validation")
    if raw["empirical_normalization"] or raw["gradient_rescaling"]:
        raise RuntimeError("forbidden gradient adjustment is present")

    rows = []
    for scenario, scenario_data in raw["scenarios"].items():
        direction = scenario_data["directions"]["adjoint_aligned"]
        for step in direction["steps"]:
            for sign in ("plus", "minus"):
                forward = step[sign]["forward"]
                objective = step[sign]["objectives"][
                    str(float(scenario[:-2]))
                ]
                rows.append(
                    {
                        "scenario": scenario,
                        "step": step["step"],
                        "sign": sign,
                        "analytic_directional_A": direction[
                            "analytic_directional_A"
                        ],
                        "finite_difference_directional_A": step[
                            "finite_difference_directional_A"
                        ],
                        "relative_error": step["relative_error"],
                        "P_Q_W": forward["P_Q_W"],
                        "P_six_W": forward["P_six_W"],
                        "six_face_closure_relative_error": forward[
                            "six_face_closure_relative_error"
                        ],
                        "Q_mapping_relative_error": objective[
                            "Q_mapping_relative_error"
                        ],
                        "energy_balance_relative_error": objective[
                            "energy_balance_relative_error"
                        ],
                        "linear_residual_relative": objective[
                            "linear_residual_relative"
                        ],
                        "FSP_sha256": forward["project"]["sha256"],
                        "reused_completed": forward["reused_completed"],
                    }
                )

    summary = {
        "status": STATUS,
        "passed_full_combined_certificate": False,
        "strong_direction_gate_passed": True,
        "scope": (
            "adjoint-aligned direction only; not the five-direction "
            "combined physical-rho certificate"
        ),
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "raw_result": artifact(raw_path),
        "immutable_pre_plateau_heuristic_result": artifact(old_path),
        "gates": raw["gates"],
        "scenarios": {
            scenario: {
                "analytic_directional_A": data["directions"][
                    "adjoint_aligned"
                ]["analytic_directional_A"],
                "steps": [
                    {
                        "step": row["step"],
                        "finite_difference_directional_A": row[
                            "finite_difference_directional_A"
                        ],
                        "relative_error": row["relative_error"],
                    }
                    for row in data["directions"]["adjoint_aligned"][
                        "steps"
                    ]
                ],
                "strict_monotone_difference_reduction": data[
                    "directions"
                ]["adjoint_aligned"][
                    "strict_monotone_difference_reduction"
                ],
                "step_plateau_relative": data["directions"][
                    "adjoint_aligned"
                ]["step_plateau_relative"],
                "step_plateau_relative_limit": data["directions"][
                    "adjoint_aligned"
                ]["step_plateau_relative_limit"],
                "coordinate_mismatch_m": data["coordinate_mismatch_m"],
            }
            for scenario, data in raw["scenarios"].items()
        },
        "next_gate": (
            "CENTRAL_EDGE_SMOOTH_RANDOM_COMBINED_PHYSICAL_RHO_PTE_ADFD"
        ),
    }

    summary_path = (
        report_dir
        / "corrected_combined_strong_direction_diagnostic_summary.json"
    )
    cases_path = (
        report_dir
        / "corrected_combined_strong_direction_diagnostic_cases.csv"
    )
    report_path = (
        report_dir
        / "CORRECTED_COMBINED_STRONG_DIRECTION_DIAGNOSTIC_REPORT.md"
    )
    manifest_path = (
        report_dir
        / "CORRECTED_COMBINED_STRONG_DIRECTION_DIAGNOSTIC_RAW_MANIFEST.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    with cases_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    scenario_lines = []
    for scenario, data in summary["scenarios"].items():
        errors = ", ".join(
            f"`h={row['step']:g}: {row['relative_error']:.8g}`"
            for row in data["steps"]
        )
        scenario_lines.append(
            f"- {scenario}: {errors}; plateau "
            f"`{data['step_plateau_relative']:.8g}`; strict monotone "
            f"`{str(data['strict_monotone_difference_reduction']).lower()}`"
        )
    report_path.write_text(
        "# Corrected combined strong-direction diagnostic\n\n"
        f"- Status: `{STATUS}`\n"
        "- Scope: one adjoint-aligned direction only; this is not the final "
        "five-direction certificate.\n"
        "- Empirical normalization: absent\n"
        "- Gradient rescaling: absent\n"
        "- Old Stage 10 and pre-plateau diagnostic raw results: preserved\n\n"
        "## Directional results\n\n"
        + "\n".join(scenario_lines)
        + "\n\n"
        "The centered-FD derivatives form a solver-noise-limited plateau "
        "well below `0.1%`. Strict monotone reduction is reported separately "
        "and is false; it is not hidden or rewritten. The independent strong "
        "AD-FD gate remains `1%`.\n\n"
        "## Worst auxiliary gates\n\n"
        f"- strong relative error: "
        f"`{raw['gates']['worst_strong_direction_relative_error']:.8g}`\n"
        f"- optical closure: "
        f"`{raw['gates']['worst_optical_closure_relative_error']:.8g}`\n"
        f"- Q mapping: "
        f"`{raw['gates']['worst_Q_mapping_relative_error']:.8g}`\n"
        f"- thermal energy balance: "
        f"`{raw['gates']['worst_thermal_energy_balance_relative_error']:.8g}`\n"
        f"- linear residual: "
        f"`{raw['gates']['worst_linear_residual_relative']:.8g}`\n"
        f"- mapping transpose: "
        f"`{raw['gates']['mapping_transpose_relative_error']:.8g}`\n\n"
        "The next gate is central-localized, design-edge-localized, "
        "smooth/asymmetric, and fixed-seed-random combined AD-FD. Gray-law "
        "sensitivity, full latent AD-FD, and optimization remain blocked.\n"
    )

    raw_dir = raw_path.parent
    raw_artifacts = [
        artifact(path)
        for path in sorted(raw_dir.iterdir())
        if path.is_file()
        and path.suffix in {".fsp", ".json"}
        and path not in {raw_path, old_path}
    ]
    manifest = {
        "generation_command": (
            "python -m photothermal_pte.finite_inverse_design."
            "publish_corrected_combined_strong_direction "
            "--raw-result <external>/corrected_combined_physical_rho_pte_adfd.json "
            "--immutable-monotonic-result "
            "<external>/strong_only_monotonic_heuristic_result.json "
            "--report-dir photothermal_pte/reports/inverse_design_pte_adfd"
        ),
        "raw_result": artifact(raw_path),
        "immutable_pre_plateau_heuristic_result": artifact(old_path),
        "raw_artifacts": raw_artifacts,
        "raw_FSP_committed_to_git": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
