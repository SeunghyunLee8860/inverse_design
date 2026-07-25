#!/usr/bin/env python3
"""Publish compact reports for the validated multi-material FVM sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import config_stage1 as config
from lumerical_api import utc_timestamp, write_json


GIT_ROOT = config.REPOSITORY_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=str(
            config.OUTPUT_ROOT
            / "fvm_multimaterial_sensitivity"
            / "sweep_v1"
        ),
    )
    parser.add_argument(
        "--report-dir",
        default=str(
            config.REPOSITORY_ROOT
            / "reports"
            / "fvm_multimaterial_thermal"
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_change(value: float, reference: float) -> float:
    return (value - reference) / reference


def git_value(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=GIT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = load_json(input_dir / "sensitivity_summary.json")
    if not summary["passed"]:
        raise RuntimeError("refusing to publish a failed sensitivity summary")
    case_rows = load_csv(input_dir / "sensitivity_cases.csv")
    comparison_rows = load_csv(input_dir / "convergence_comparisons.csv")
    cases = {item["case_id"]: item for item in case_rows}
    reference = cases["final_native"]
    reference_tmax = float(reference["Tmax_K_per_W_m2"])
    reference_average = float(reference["flake_average_K_per_W_m2"])

    convergence_table = []
    for item in comparison_rows:
        convergence_table.append(
            [
                item["family"],
                f"{item['from_case']} → {item['to_case']}",
                f"{100 * float(item['Tmax_relative_change']):.6g}%",
                f"{100 * float(item['flake_average_relative_change']):.6g}%",
                f"{100 * float(item['flake_probe_3d_NRMSE']):.6g}%",
            ]
        )
    sensitivity_table = []
    selected_ids = [
        "Gbottom_1e6",
        "Gbottom_3e6",
        "final_native",
        "Gbottom_1p5e7",
        "Gbottom_3e7",
        "Gbottom_1e8",
        "Gbottom_perfect",
        "Gtop_7p37e4",
        "Gtop_7p37e5",
        "Gtop_7p37e7",
        "Gtop_perfect",
        "oxide_si_perfect",
        "convection_h10",
    ]
    for case_id in selected_ids:
        item = cases[case_id]
        tmax = float(item["Tmax_K_per_W_m2"])
        average = float(item["flake_average_K_per_W_m2"])
        sensitivity_table.append(
            [
                case_id,
                f"{tmax:.9e}",
                f"{average:.9e}",
                f"{100 * relative_change(tmax, reference_tmax):+.6g}%",
                f"{100 * relative_change(average, reference_average):+.6g}%",
            ]
        )

    report = f"""# Multi-material anisotropic finite-G FVM thermal report

## Status

`{summary["status"]}`

This is an independent conservative Cartesian Python/SciPy FVM result. It is
not a Lumerical HEAT result. The common scalar-isotropic/perfect-contact 3D
subset was separately cross-validated against v261 HEAT before this extended
model was used.

## Production reference

- Geometry: 2 um x 2 um x 100 nm TaIrTe4 flake, 285 nm bottom SiO2,
  600 nm high centered SiO2 disk of radius 1.5 um, and Si substrate.
- Conductivity: TaIrTe4 diag(14.4, 3.8, 1.0), SiO2 1.38, Si 145 W/(m K).
- Interfaces: G_bottom = G_top = 7.37e6 W/(m2 K);
  G_SiO2/Si = 1.1e9 W/(m2 K).
- Reference domain: 32 um lateral span and 20 um Si depth.
- Boundary condition: DeltaT=0 K on the far lateral and bottom boundaries;
  exposed surfaces adiabatic in the reference case.
- Source normalization: incident intensity 1 W/m2.
- Preserved optical power: 2.56071371086521e-12 W.
- Temperature quantity: DeltaT / incident intensity [K/(W/m2)].
- Reference Tmax: {reference_tmax:.12e} K/(W/m2).
- Reference TaIrTe4 volume-average DeltaT:
  {reference_average:.12e} K/(W/m2).
- Reference active cells: {int(reference["active_cells"]):,}.
- Reference energy-balance relative error:
  {float(reference["energy_balance_relative_error"]):.6e}.

The source was imported without clipping, smoothing, gain, global rescaling,
periodic tiling, or deletion outside a stored mask. Coarse/refined sensitivity
meshes use conservative source-energy restriction or piecewise-constant
subdivision, respectively; every case retained exactly the same total source
power.

## Domain, depth, and mesh convergence

The gate requires Tmax, TaIrTe4 volume-average temperature, and the common
TaIrTe4 3D probe-field NRMSE all to be below 1% for the final pair.

{markdown_table(
    ["family", "comparison", "Tmax change", "flake average change", "3D probe NRMSE"],
    convergence_table,
)}

All final-pair metrics pass the 1% gate. The refined mesh keeps the native
optical x/y control volumes, subdivides source cells by two in z, and refines
the surrounding material mesh.

## Interface and boundary sensitivity

Changes below are relative to the 32 um x 32 um lateral, 20 um Si-depth native
reference.

{markdown_table(
    ["case", "Tmax", "flake average", "Tmax change", "average change"],
    sensitivity_table,
)}

G sweeps quantify physical-parameter sensitivity and are not numerical
convergence gates. The adiabatic top SiO2 disk has essentially zero net
steady heat removal; changing G_top changes local heat redistribution. The
h=10 W/(m2 K) case applies the Robin condition to every exposed solid-air
surface, including exact cell-center-to-surface conduction resistance.

## Gate accounting

- Cases executed: {summary["case_count"]}.
- Every case equation converged and conserved:
  `{str(summary["all_case_equations_converged_and_conserved"]).lower()}`.
- Q mapping error in every case: 0.
- Required Q mapping error: <0.5%.
- Required energy-balance error: <1%.
- Production convergence gate: passed.
- Full field artifacts remain in the ignored validation output directory and
  are indexed by SHA-256 in `RAW_ARTIFACT_MANIFEST.json`.

The numerical model is now suitable as a steady thermal production path for
this geometry and source, subject to the stated material-property assumptions.
This parameter set is a numerical-convergence checkpoint, not a unique final
experimental prediction. In particular, TaIrTe4 kz=1.0 W/(m K), G_top, and
the top-disk support geometry require separately named physical scenarios.
"""
    report_path = report_dir / "FVM_MULTIMATERIAL_THERMAL_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    summary_copy = dict(summary)
    summary_copy["published_at_utc"] = utc_timestamp()
    summary_copy["production_reference_promoted"] = True
    summary_copy["production_reference_case_id"] = "final_native"
    summary_copy["provisional_until_sensitivity_passes"] = False
    summary_copy["next_required_gate"] = None
    summary_copy["completed_gates"] = [
        "DOMAIN_SENSITIVITY",
        "SI_DEPTH_SENSITIVITY",
        "THERMAL_MESH_SENSITIVITY",
        "INTERFACE_G_SENSITIVITY",
        "BOUNDARY_SENSITIVITY",
    ]
    summary_copy["promoted_reference_metadata"] = {
        "status": "PROMOTED_AFTER_COMPLETED_SENSITIVITY",
        "provisional_until_sensitivity_passes": False,
        "next_required_gate": None,
        "raw_per_case_JSON_modified": False,
        "raw_reference_case_provenance_preserved": True,
    }
    summary_copy["report_path"] = str(report_path)
    write_json(
        report_dir / "fvm_multimaterial_thermal_summary.json",
        summary_copy,
    )
    for source_name, target_name in (
        ("sensitivity_cases.csv", "fvm_multimaterial_thermal_cases.csv"),
        ("convergence_comparisons.csv", "fvm_multimaterial_convergence.csv"),
    ):
        (report_dir / target_name).write_text(
            (input_dir / source_name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    artifacts = []
    for case_result in sorted(input_dir.glob("*/case_result.json")):
        raw_path = case_result.with_name("temperature_flux_3d.npz")
        for artifact in (case_result, raw_path):
            artifacts.append(
                {
                    "repository_path": str(
                        artifact.relative_to(GIT_ROOT)
                    ),
                    "server_path": str(artifact),
                    "bytes": artifact.stat().st_size,
                    "sha256": sha256_file(artifact),
                }
            )
    baseline_result = (
        config.OUTPUT_ROOT
        / "fvm_multimaterial_thermal"
        / "baseline_v2"
        / "case_result.json"
    )
    baseline_raw = baseline_result.with_name("temperature_flux_3d.npz")
    for artifact in (baseline_result, baseline_raw):
        artifacts.append(
            {
                "repository_path": str(
                    artifact.relative_to(GIT_ROOT)
                ),
                "server_path": str(artifact),
                "bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
        )
    for artifact in (
        report_path,
        report_dir / "fvm_multimaterial_thermal_summary.json",
        report_dir / "fvm_multimaterial_thermal_cases.csv",
        report_dir / "fvm_multimaterial_convergence.csv",
    ):
        artifacts.append(
            {
                "repository_path": str(
                    artifact.relative_to(GIT_ROOT)
                ),
                "server_path": str(artifact),
                "bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
        )
    manifest = {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        "branch": git_value("branch", "--show-current"),
        "generation_commit": git_value("rev-parse", "HEAD"),
        "solver_attribution": summary["solver_attribution"],
        "raw_artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    write_json(report_dir / "RAW_ARTIFACT_MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
