#!/usr/bin/env python3
"""Publish the immutable scale-adaptive near-null AD--FD diagnostic."""

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
    parser.add_argument("--adaptive-result", required=True)
    parser.add_argument("--rejected-result")
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    adaptive_path = Path(args.adaptive_result).expanduser().resolve()
    adaptive = json.loads(adaptive_path.read_text())
    if adaptive["status"] not in {
        "VALIDATED_SCALE_ADAPTIVE_NEAR_NULL_COMBINED_ADFD",
        "FAILED_SCALE_ADAPTIVE_NEAR_NULL_COMBINED_ADFD",
    }:
        raise RuntimeError("unexpected adaptive result status")
    if any(
        adaptive.get(key)
        for key in ("empirical_normalization", "gradient_rescaling", "clipping")
    ):
        raise RuntimeError("forbidden gradient manipulation detected")
    rejected_path = (
        Path(args.rejected_result).expanduser().resolve()
        if args.rejected_result
        else None
    )
    rejected = (
        json.loads(rejected_path.read_text())
        if rejected_path is not None
        else None
    )
    rows = []
    for case in adaptive["cases"]:
        for row in case["rows"]:
            rows.append(
                {
                    "scenario": case["scenario"],
                    "direction": case["direction"],
                    "step": row["step"],
                    "provenance": row["provenance"],
                    "analytic_directional_A": case[
                        "analytic_directional_A"
                    ],
                    "finite_difference_directional_A": row[
                        "finite_difference_directional_A"
                    ],
                    "relative_error": row["relative_error"],
                    "step_plateau_relative": case[
                        "step_plateau_relative"
                    ],
                    "step_plateau_limit": case[
                        "step_plateau_relative_limit"
                    ],
                    "case_passed": case["passed"],
                }
            )
    failed = [
        {
            "scenario": case["scenario"],
            "direction": case["direction"],
            "selected_relative_error": case["selected_relative_error"],
            "step_plateau_relative": case["step_plateau_relative"],
            "step_plateau_limit": case["step_plateau_relative_limit"],
        }
        for case in adaptive["cases"]
        if not case["passed"]
    ]
    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = "scale_adaptive_near_null_combined_adfd"
    summary_path = report_dir / f"{stem}_summary.json"
    cases_path = report_dir / f"{stem}_cases.csv"
    report_path = report_dir / (
        "SCALE_ADAPTIVE_NEAR_NULL_COMBINED_ADFD_DIAGNOSTIC_REPORT.md"
    )
    manifest_path = report_dir / (
        "SCALE_ADAPTIVE_NEAR_NULL_COMBINED_ADFD_RAW_MANIFEST.json"
    )
    summary = {
        "status": adaptive["status"],
        "passed": adaptive["passed"],
        "raw_result": digest(adaptive_path),
        "immutable_original_failure": adaptive["original_failed_result"],
        "step_policy": adaptive["step_policy"],
        "diagnostic_gates": adaptive["diagnostic_gates"],
        "failed_cases": failed,
        "rejected_recovery_result": (
            digest(rejected_path) if rejected_path is not None else None
        ),
        "rejected_recovery_reason": (
            "central h=0.02 plus came from an interrupted orphan process; its "
            "FSP byte size and six-face closure differed from the three clean "
            "pairs and its objective trend contradicted h=0.01"
            if rejected is not None
            else None
        ),
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "clipping": False,
        "gray_law_sensitivity_run": False,
        "full_latent_adfd_run": False,
        "optimization_run": False,
        "next_gate": adaptive["next_gate"],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    with cases_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    case_lines = "\n".join(
        "- "
        f"{case['scenario']} {case['direction']}: selected error "
        f"`{100*case['selected_relative_error']:.6g}%`, plateau "
        f"`{100*case['step_plateau_relative']:.6g}%` "
        f"({'pass' if case['passed'] else 'fail'})"
        for case in adaptive["cases"]
    )
    report_path.write_text(
        "# Scale-adaptive near-null combined AD–FD diagnostic\n\n"
        f"- Status: `{adaptive['status']}`\n"
        f"- Passed: `{str(adaptive['passed']).lower()}`\n"
        "- Empirical normalization / gradient rescaling / clipping: `false`\n"
        "- Original sequence: `0.01 -> 0.005 -> 0.0025`\n"
        "- Near-null sequence: `0.02 -> 0.01 -> 0.005`\n\n"
        "## Cases\n\n"
        f"{case_lines}\n\n"
        "The original five-direction raw result remains unchanged. A rejected "
        "orphan-recovery run, when listed in the manifest, is retained only as "
        "a provenance diagnostic and is not used to promote a certificate.\n"
    )
    raw_artifacts = []
    for path in sorted(adaptive_path.parent.iterdir()):
        if path.is_file() and path.suffix in {".fsp", ".json", ".log"}:
            raw_artifacts.append(digest(path.resolve()))
    manifest = {
        "generation_command": (
            "python -m photothermal_pte.finite_inverse_design."
            "publish_scale_adaptive_near_null_diagnostic "
            "--adaptive-result <external>/"
            "scale_adaptive_near_null_combined_adfd.json "
            "--rejected-result <external>/"
            "scale_adaptive_near_null_combined_adfd.json "
            "--report-dir photothermal_pte/reports/inverse_design_pte_adfd"
        ),
        "adaptive_result": digest(adaptive_path),
        "rejected_result": (
            digest(rejected_path) if rejected_path is not None else None
        ),
        "raw_artifacts": raw_artifacts,
        "published": {
            "summary": digest(summary_path),
            "cases": digest(cases_path),
            "report": digest(report_path),
        },
        "raw_FSP_committed_to_git": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"status": adaptive["status"], "summary": str(summary_path)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
