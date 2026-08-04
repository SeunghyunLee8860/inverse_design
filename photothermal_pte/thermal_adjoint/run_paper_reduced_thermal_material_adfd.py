#!/usr/bin/env python3
"""Certify the paper-reduced rho-dependent Robin thermal derivative.

The optical heat source is held fixed.  This isolates

    -lambda_T^T (dK_T/d rho) theta + lambda_T^T db_T/d rho

before it is combined with the Maxwell absorption derivative.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PHOTOTHERMAL = HERE.parent
REPOSITORY = PHOTOTHERMAL.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from paper_reduced_thermal import (  # noqa: E402
    DesignSurfaceMap,
    G_AIR_W_M2K,
    G_EVAPORATED_SIO2_W_M2K,
    G_THERMAL_SIO2_W_M2K,
    KAPPA_TAIRTE4_W_MK,
    T_BATH_K,
    boundary_diagnostics,
    evaluate_reduced_paper_thermal,
)


STEPS = np.asarray([2e-2, 1e-2, 5e-3, 2.5e-3, 1e-3, 5e-4, 1e-4])
SCENARIOS = {
    "thermally_grown_SiO2_baseline": G_THERMAL_SIO2_W_M2K,
    "evaporated_SiO2_sensitivity": G_EVAPORATED_SIO2_W_M2K,
}


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _report(summary: dict) -> str:
    lines = [
        "# Paper-reduced rho-dependent thermal-material AD–FD",
        "",
        f"**Status: `{summary['status']}`**",
        "",
        "This is the isolated fixed-Q certificate for the reduced TaIrTe4 "
        "surface-boundary model. It is not the older Si/SiO2/TaIrTe4 bulk "
        "FVM model and it is not a final terminal-current prediction.",
        "",
        "## Contract",
        "",
        "- Optical material label: **n=4 optical proxy + paper SiO2 thermal "
        "boundary**.",
        "- TaIrTe4 kappa: diag(14.4, 3.8, 1.0) W/(m K).",
        "- Bath temperature: 300 K. The numerically solved unknown is "
        "theta=T-300 K.",
        "- Bottom substrate Robin G: 7.37e6 W/(m2 K).",
        "- Top design law: `G=1+rho_bar*(G_SiO2-1)` W/(m2 K).",
        "- No bulk k_air, k_SiO2, k_Si, or G_SiO2/Si is introduced.",
        "- The finite local PTE mask remains a numerical A m surrogate; a "
        "weighting-potential/finite-contact solve is still blocked.",
        "",
        "## Discrete derivative",
        "",
        "`g_i=A_i*G(rho_i)`, `(K_T)_ii += g_i`, and the absolute-temperature "
        "load is `b_i += g_i*T_bath`. In the exactly shifted theta system, "
        "`b=0` and the same thermal derivative is",
        "",
        "`dF/drho_i = -lambda_i*A_i*(G_SiO2-G_air)*theta_i`.",
        "",
        "## Scenario results",
        "",
        "| Scenario | G_SiO2 (W/m2K) | AD | best FD | rel. error | "
        "energy | residual | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for scenario in summary["scenarios"]:
        lines.append(
            f"| {scenario['name']} | {scenario['G_SiO2_W_m2K']:.6e} | "
            f"{scenario['adjoint_directional']:.6e} | "
            f"{scenario['best_step']['finite_difference']:.6e} | "
            f"{scenario['best_step']['relative_error']:.6e} | "
            f"{scenario['energy_balance_relative_error']:.6e} | "
            f"{scenario['linear_residual_relative']:.6e} | "
            f"{scenario['passed']} |"
        )
    lines.extend(
        [
            "",
            "The thermally-grown value is the paper baseline. The evaporated "
            "value is a named fabrication sensitivity, not a confidence "
            "interval and not a replacement baseline.",
            "",
            "## Retained blocker",
            "",
            "- `BLOCKED_PHYSICAL_WEIGHTING_POTENTIAL_OR_FINITE_FLAKE_MASK`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-npz", required=True)
    parser.add_argument("--density-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--report-dir",
        default=str(
            PHOTOTHERMAL / "reports" / "inverse_design_pte_adfd"
        ),
    )
    parser.add_argument(
        "--relative-error-limit", type=float, default=1.0e-4
    )
    args = parser.parse_args()
    q_path = Path(args.q_npz).expanduser().resolve()
    density_path = Path(args.density_npz).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    q_data = np.load(q_path)
    q_full = np.asarray(q_data["q_thermal_W_m3_per_W_m2"], float)
    if q_full.shape[:2] != (24, 24) or q_full.shape[2] < 2:
        raise RuntimeError(f"unexpected thermal Q shape {q_full.shape}")
    source = q_full[:, :, -2:]
    density_data = np.load(density_path)
    density = np.asarray(density_data["density"], float)
    direction = np.asarray(density_data["direction"], float)
    surface_map = DesignSurfaceMap(
        physical_shape=density.shape,
        face_shape=(24, 24),
    )
    rho_face = surface_map.apply(density)
    direction_face = surface_map.apply(direction)
    if np.min(rho_face - STEPS[0] * direction_face) < 0.0:
        raise RuntimeError("negative thermal density FD endpoint")
    if np.max(rho_face + STEPS[0] * direction_face) > 1.0:
        raise RuntimeError("thermal density FD endpoint exceeds one")

    rows: list[dict] = []
    scenario_summaries: list[dict] = []
    raw_arrays: dict[str, np.ndarray] = {
        "rho_face": rho_face,
        "direction_face": direction_face,
        "fixed_source_W_m3_per_W_m2": source,
    }
    for name, G_sio2 in SCENARIOS.items():
        baseline = evaluate_reduced_paper_thermal(
            rho_face=rho_face,
            source_W_m3=source,
            G_sio2_W_m2K=G_sio2,
        )
        analytic = float(
            np.sum(baseline.gradient_rho_face_A_m * direction_face)
        )
        scenario_rows = []
        for step in STEPS:
            plus = evaluate_reduced_paper_thermal(
                rho_face=rho_face + step * direction_face,
                source_W_m3=source,
                G_sio2_W_m2K=G_sio2,
            ).objective_A_m
            minus = evaluate_reduced_paper_thermal(
                rho_face=rho_face - step * direction_face,
                source_W_m3=source,
                G_sio2_W_m2K=G_sio2,
            ).objective_A_m
            finite_difference = (plus - minus) / (2.0 * step)
            relative_error = abs(finite_difference - analytic) / max(
                abs(finite_difference), abs(analytic), 1e-300
            )
            row = {
                "scenario": name,
                "G_SiO2_W_m2K": G_sio2,
                "step": float(step),
                "adjoint_directional": analytic,
                "finite_difference": finite_difference,
                "absolute_error": abs(finite_difference - analytic),
                "relative_error": relative_error,
                "plus_objective_A_m_per_W_m2": plus,
                "minus_objective_A_m_per_W_m2": minus,
            }
            rows.append(row)
            scenario_rows.append(row)
        best = min(scenario_rows, key=lambda item: item["relative_error"])
        diagnostics = boundary_diagnostics(baseline)
        passed = (
            best["relative_error"] < args.relative_error_limit
            and baseline.solved.energy_balance_relative_error < 0.01
            and baseline.solved.linear_residual_relative < 1.0e-8
        )
        scenario_summaries.append(
            {
                "name": name,
                "G_SiO2_W_m2K": G_sio2,
                "passed": passed,
                "objective_A_m_per_W_m2": baseline.objective_A_m,
                "adjoint_directional": analytic,
                "best_step": best,
                "gradient_thermal_material_l2_A_m": float(
                    np.linalg.norm(baseline.gradient_rho_face_A_m)
                ),
                "temperature_rise_max_K_per_W_m2": float(
                    np.max(baseline.solved.temperature_K)
                ),
                "source_power_W_per_W_m2": (
                    baseline.solved.source_power_W
                ),
                "energy_balance_relative_error": (
                    baseline.solved.energy_balance_relative_error
                ),
                "linear_residual_relative": (
                    baseline.solved.linear_residual_relative
                ),
                "boundary": diagnostics,
            }
        )
        raw_arrays[f"{name}_temperature_rise_K_per_W_m2"] = (
            baseline.solved.temperature_K
        )
        raw_arrays[f"{name}_thermal_gradient_face_A_m"] = (
            baseline.gradient_rho_face_A_m
        )
        raw_arrays[f"{name}_G_design_W_m2K"] = (
            baseline.G_design_W_m2K
        )

    passed = all(item["passed"] for item in scenario_summaries)
    raw = output / "paper_reduced_thermal_material_adfd.npz"
    np.savez_compressed(raw, **raw_arrays)
    summary = {
        "schema_version": 1,
        "generated_at_utc": _utc(),
        "status": (
            "VALIDATED_PAPER_REDUCED_THERMAL_MATERIAL_ADFD"
            if passed
            else "FAILED_PAPER_REDUCED_THERMAL_MATERIAL_ADFD"
        ),
        "passed": passed,
        "scope": (
            "fixed optical Q; rho-dependent paper-reduced TaIrTe4 surface "
            "Robin thermal derivative"
        ),
        "material_label": (
            "n=4 optical proxy + paper SiO2 thermal boundary"
        ),
        "paper_contract": {
            "TaIrTe4_kappa_W_mK": list(KAPPA_TAIRTE4_W_MK),
            "TaIrTe4_air_G_W_m2K": G_AIR_W_M2K,
            "TaIrTe4_thermally_grown_SiO2_G_W_m2K": (
                G_THERMAL_SIO2_W_M2K
            ),
            "TaIrTe4_evaporated_SiO2_G_W_m2K": (
                G_EVAPORATED_SIO2_W_M2K
            ),
            "bath_temperature_K": T_BATH_K,
            "unknown_solved": "theta=T-T_bath",
            "design_G_law": (
                "G_air + rho_bar*(G_SiO2-G_air)"
            ),
            "bulk_air_SiO2_Si_omitted": True,
            "SiO2_Si_G_omitted": True,
        },
        "mapping": {
            "physical_shape": list(surface_map.physical_shape),
            "thermal_face_shape": list(surface_map.face_shape),
            "block_shape": list(surface_map.block_shape),
            "rule": (
                "drop duplicated x/y fencepost, average physical z, then "
                "10x10 area average onto each thermal face"
            ),
        },
        "inputs": {
            "fixed_Q": {
                "path": str(q_path),
                "bytes": q_path.stat().st_size,
                "sha256": _sha256(q_path),
            },
            "physical_density_and_direction": {
                "path": str(density_path),
                "bytes": density_path.stat().st_size,
                "sha256": _sha256(density_path),
            },
        },
        "scenarios": scenario_summaries,
        "gates": {
            "relative_error_limit": args.relative_error_limit,
            "energy_balance_limit": 0.01,
            "linear_residual_limit": 1.0e-8,
        },
        "git": {
            "branch": _git("branch", "--show-current"),
            "head_before_generated_reports": _git("rev-parse", "HEAD"),
        },
        "raw_artifact": {
            "path": str(raw),
            "bytes": raw.stat().st_size,
            "sha256": _sha256(raw),
            "committed_to_git": False,
        },
        "blockers": [
            "BLOCKED_COMBINED_RHO_DEPENDENT_MAXWELL_THERMAL_ADFD",
            "BLOCKED_PHYSICAL_WEIGHTING_POTENTIAL_OR_FINITE_FLAKE_MASK",
        ],
    }
    summary_path = (
        report_dir / "paper_reduced_thermal_material_adfd_summary.json"
    )
    csv_path = (
        report_dir / "paper_reduced_thermal_material_adfd_cases.csv"
    )
    report_path = (
        report_dir / "PAPER_REDUCED_THERMAL_MATERIAL_ADFD_REPORT.md"
    )
    manifest_path = (
        report_dir
        / "PAPER_REDUCED_THERMAL_MATERIAL_ADFD_RAW_ARTIFACT_MANIFEST.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(csv_path, rows)
    report_path.write_text(_report(summary), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "generation_command": (
            f"{sys.executable} {Path(__file__).resolve()} "
            f"--q-npz {q_path} --density-npz {density_path} "
            f"--output-dir {output} --report-dir {report_dir}"
        ),
        "raw_artifacts": [summary["raw_artifact"]],
        "repository_artifacts": [
            str(path)
            for path in (summary_path, csv_path, report_path, manifest_path)
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
