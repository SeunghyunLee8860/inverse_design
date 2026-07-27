#!/usr/bin/env python3
"""Publish a fail-closed diagnostic for combined physical-rho AD--FD."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


PUBLISHED_STATUS = (
    "DIAGNOSTIC_FAILED_COMBINED_PHYSICAL_RHO_PTE_ADFD"
)
EXPECTED_RAW_STATUS = "FAILED_COMBINED_PHYSICAL_RHO_PTE_ADFD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--generation-command", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    target = path.expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    return {
        "path": str(target),
        "byte_size": target.stat().st_size,
        "sha256": sha256(target),
    }


def collect_projects(value: object, result: dict[str, dict]) -> None:
    if isinstance(value, dict):
        if {"path", "byte_size", "sha256"} <= set(value):
            path = Path(str(value["path"])).expanduser().resolve()
            if path.suffix == ".fsp":
                actual = artifact(path)
                if actual["byte_size"] != value["byte_size"]:
                    raise RuntimeError(f"FSP byte-size mismatch: {path}")
                if actual["sha256"] != value["sha256"]:
                    raise RuntimeError(f"FSP SHA-256 mismatch: {path}")
                result[str(path)] = actual
        for child in value.values():
            collect_projects(child, result)
    elif isinstance(value, list):
        for child in value:
            collect_projects(child, result)


def main() -> int:
    args = parse_args()
    result_path = Path(args.result).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    raw = json.loads(result_path.read_text())
    if raw.get("status") != EXPECTED_RAW_STATUS or raw.get("passed"):
        raise RuntimeError(
            "refusing to publish a non-failed result as a diagnostic"
        )
    scenarios = raw["scenarios"]
    rows = []
    for scenario_name, scenario in scenarios.items():
        for row in scenario["fd_rows"]:
            rows.append(
                {
                    "scenario": scenario_name,
                    "step": row["step"],
                    "objective_plus_A": row["objective_plus_A"],
                    "objective_minus_A": row["objective_minus_A"],
                    "finite_difference_directional_A": row[
                        "finite_difference_directional_A"
                    ],
                    "thermal_material_directional_A": scenario[
                        "thermal_material_directional_A"
                    ],
                    "optical_Q_directional_A": scenario[
                        "optical_Q_directional_A"
                    ],
                    "combined_adjoint_directional_A": scenario[
                        "combined_adjoint_directional_A"
                    ],
                    "relative_error": row["relative_error"],
                    "passed_1pct_diagnostic": (
                        row["relative_error"] < 1.0e-2
                    ),
                    "passed_original_0p5pct_gate": (
                        row["relative_error"] < 5.0e-3
                    ),
                }
            )
    projects: dict[str, dict] = {}
    collect_projects(raw, projects)
    for role, value in raw.get("resume_projects", {}).items():
        checked = artifact(Path(value["path"]))
        if checked["sha256"] != value["sha256"]:
            raise RuntimeError(
                f"resume project SHA-256 mismatch for {role}"
            )
        projects[str(Path(value["path"]).resolve())] = checked
    raw_npz = artifact(Path(raw["raw_artifact"]["path"]))
    if raw_npz["sha256"] != raw["raw_artifact"]["sha256"]:
        raise RuntimeError("raw NPZ SHA-256 mismatch")
    summary = {
        "status": PUBLISHED_STATUS,
        "raw_status": raw["status"],
        "passed": False,
        "diagnostic_only": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "Combined physical-density AD--FD exceeded the original "
            "0.5% gate and is preserved without empirical normalization "
            "or gradient rescaling."
        ),
        "physical_density": raw["physical_density"],
        "adjoint_source_definition": raw["adjoint_source_definition"],
        "scalar_P_Q_source_reused": raw["scalar_P_Q_source_reused"],
        "layout_directional_epsilon": raw[
            "layout_directional_epsilon"
        ],
        "scenarios": {
            name: {
                "base_objective_A": value["base_objective_A"],
                "thermal_material_directional_A": value[
                    "thermal_material_directional_A"
                ],
                "optical_Q_directional_A": value[
                    "optical_Q_directional_A"
                ],
                "combined_adjoint_directional_A": value[
                    "combined_adjoint_directional_A"
                ],
                "cpu_gpu_adjoint_field_NRMSE": value[
                    "cpu_gpu_adjoint_field_NRMSE"
                ],
                "fd_rows": [
                    {
                        "step": row["step"],
                        "finite_difference_directional_A": row[
                            "finite_difference_directional_A"
                        ],
                        "relative_error": row["relative_error"],
                    }
                    for row in value["fd_rows"]
                ],
            }
            for name, value in scenarios.items()
        },
        "gates": raw["gates"],
        "normalization_or_gradient_rescaling_used": False,
        "next_gate": (
            "RHO05_REPRESENTATION_EQUIVALENCE_THEN_"
            "COMPONENT_YEE_MATERIAL_JACOBIAN"
        ),
        "optimization_run": False,
    }
    summary_path = (
        report_dir / "combined_physical_rho_pte_adfd_diagnostic.json"
    )
    csv_path = (
        report_dir
        / "combined_physical_rho_pte_adfd_diagnostic_cases.csv"
    )
    report_path = (
        report_dir / "COMBINED_PHYSICAL_RHO_PTE_ADFD_DIAGNOSTIC.md"
    )
    manifest_path = (
        report_dir
        / "COMBINED_PHYSICAL_RHO_PTE_ADFD_DIAGNOSTIC_MANIFEST.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    report_lines = [
        "# Combined physical-density PTE AD–FD diagnostic",
        "",
        f"Status: `{PUBLISHED_STATUS}`",
        "",
        "This is a preserved failed checkpoint, not a validation.",
        "No empirical normalization or gradient rescaling was used.",
        "",
        "The Maxwell source was the spatial native-Yee vector source",
        "`R_Q^T(dI_PTE/dQ_thermal)`; the scalar `P_Q` source was not",
        "reused. The unresolved path is the component-wise",
        "density-to-Yee material Jacobian/collocation.",
        "",
        "## Directional results",
        "",
        "| scenario | h | adjoint (A) | FD (A) | relative error |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['scenario']} | {row['step']:.4g} | "
            f"{row['combined_adjoint_directional_A']:.9e} | "
            f"{row['finite_difference_directional_A']:.9e} | "
            f"{row['relative_error']:.9e} |"
        )
    gates = raw["gates"]
    report_lines.extend(
        [
            "",
            "## Preserved gates",
            "",
            f"- Worst selected AD–FD error: "
            f"`{gates['worst_selected_combined_AD_FD_relative_error']:.9e}`",
            f"- Q mapping error: "
            f"`{gates['worst_Q_mapping_relative_error']:.9e}`",
            f"- Six-face closure: "
            f"`{gates['worst_six_face_closure_relative_error']:.9e}`",
            f"- Thermal energy balance: "
            f"`{gates['worst_thermal_energy_balance_relative_error']:.9e}`",
            f"- Worst linear residual: "
            f"`{gates['worst_forward_or_adjoint_linear_residual_relative']:.9e}`",
            f"- CPU/GPU adjoint field NRMSE: "
            f"`{gates['four_um_CPU_GPU_adjoint_field_NRMSE']:.9e}`",
            "",
            "The workflow stops here until uniform rho=0.5 representation",
            "equivalence and the nonuniform component-specific Yee",
            "material Jacobian/JVP/VJP are validated.",
            "",
        ]
    )
    report_path.write_text("\n".join(report_lines))
    manifest = {
        "status": PUBLISHED_STATUS,
        "generation_command": args.generation_command,
        "raw_result": artifact(result_path),
        "raw_npz": raw_npz,
        "fsp_projects": list(projects.values()),
        "published": {
            "summary": artifact(summary_path),
            "cases_csv": artifact(csv_path),
            "report": artifact(report_path),
        },
        "raw_artifacts_committed_to_git": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
