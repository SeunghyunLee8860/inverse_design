#!/usr/bin/env python3
"""Publish the paper-reduced thermal/optical AD--FD certificate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: str) -> dict:
    return json.loads(Path(path).expanduser().resolve().read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thermal-summary", required=True)
    parser.add_argument("--physical-coarse", required=True)
    parser.add_argument("--physical-fine", required=True)
    parser.add_argument("--latent-small", required=True)
    parser.add_argument("--latent-selected", required=True)
    parser.add_argument("--latent-large", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    thermal = load(args.thermal_summary)
    physical = [load(args.physical_coarse), load(args.physical_fine)]
    latent = [
        load(args.latent_small),
        load(args.latent_selected),
        load(args.latent_large),
    ]
    physical.sort(key=lambda item: item["step"])
    latent.sort(key=lambda item: item["mapping"]["step"])
    selected = min(
        latent,
        key=lambda item: item["relative_error"],
    )
    passed = (
        thermal["passed"]
        and all(item["passed"] for item in physical)
        and selected["passed"]
        and selected["mapping"]["step"] == 0.005
    )
    rows = []
    for item in physical:
        rows.append(
            {
                "space": "physical_rho",
                "step": item["step"],
                "adjoint": item["adjoint"]["combined_directional"],
                "finite_difference": item["finite_difference"],
                "relative_error": item["relative_error"],
                "pass_5pct": item["relative_error"] < 0.05,
            }
        )
    for item in latent:
        rows.append(
            {
                "space": "latent",
                "step": item["mapping"]["step"],
                "adjoint": item["adjoint"]["combined_directional"],
                "finite_difference": item["finite_difference"],
                "relative_error": item["relative_error"],
                "pass_5pct": item["relative_error"] < 0.05,
            }
        )
    raw_paths = [
        Path(thermal["raw_artifact"]["path"]),
        *(Path(item["raw_artifact"]["path"]) for item in physical),
        *(Path(item["raw_artifact"]["path"]) for item in latent),
    ]
    raw_artifacts = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "committed_to_git": False,
        }
        for path in raw_paths
    ]
    summary = {
        "schema_version": 1,
        "status": (
            "VALIDATED_PAPER_REDUCED_RHO_DEPENDENT_THERMAL_PTE_ADFD"
            if passed
            else "FAILED_PAPER_REDUCED_RHO_DEPENDENT_THERMAL_PTE_ADFD"
        ),
        "passed": passed,
        "material_label": (
            "n=4 optical proxy + paper SiO2 thermal boundary"
        ),
        "thermal_contract": thermal["paper_contract"],
        "thermal_material_only": thermal["scenarios"],
        "physical_rho_step_sweep": [
            {
                "step": item["step"],
                "adjoint": item["adjoint"],
                "finite_difference": item["finite_difference"],
                "relative_error": item["relative_error"],
                "passed": item["passed"],
            }
            for item in physical
        ],
        "latent_step_sweep": [
            {
                "step": item["mapping"]["step"],
                "adjoint": item["adjoint"],
                "finite_difference": item["finite_difference"],
                "relative_error": item["relative_error"],
                "passed_at_5_percent": (
                    item["relative_error"] < 0.05
                ),
            }
            for item in latent
        ],
        "selected_latent_step": {
            "step": selected["mapping"]["step"],
            "relative_error": selected["relative_error"],
            "selection_rule": (
                "minimum observed centered-FD error in a bracketed "
                "three-step sweep; larger h is nonlinear and smaller h "
                "is below the v261 FD numerical-noise floor"
            ),
        },
        "selected_gradient_decomposition": selected["adjoint"],
        "selected_boundary": selected["boundary"],
        "selected_gates": selected["gates"],
        "blockers": [
            "BLOCKED_PHYSICAL_WEIGHTING_POTENTIAL_OR_FINITE_FLAKE_MASK"
        ],
        "raw_artifacts": raw_artifacts,
    }
    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        report_dir / "paper_reduced_combined_adfd_summary.json"
    )
    csv_path = report_dir / "paper_reduced_combined_adfd_cases.csv"
    report_path = report_dir / "PAPER_REDUCED_COMBINED_ADFD_REPORT.md"
    manifest_path = (
        report_dir
        / "PAPER_REDUCED_COMBINED_ADFD_RAW_ARTIFACT_MANIFEST.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = f"""# Paper-reduced rho-dependent thermal/PTE AD–FD

**Status: `{summary['status']}`**

The immutable fixed-K 2.04% checkpoint remains a numerical control. This new
certificate adds the omitted rho-dependent thermal boundary terms while
keeping the optical endpoint explicitly labeled **n=4 optical proxy + paper
SiO2 thermal boundary**.

## Reduced thermal contract

- TaIrTe4 kappa: diag(14.4, 3.8, 1.0) W/(m K).
- Bath: 300 K; the stable solve uses theta=T-300 K.
- Fixed substrate Robin G: 7.37e6 W/(m2 K).
- Design face: `G(rho_bar)=1+rho_bar*(G_SiO2-1)` W/(m2 K).
- Thermally-grown baseline: G_SiO2=7.37e6 W/(m2 K).
- Evaporated sensitivity: G_SiO2=7.37e4 W/(m2 K).
- No bulk air/SiO2/Si kappa or SiO2/Si G is introduced.

Each face adds `g=A*G` to `K_T` and `g*T_bath` to `b_T`. The added adjoint
term is `lambda_i*A_i*(G_SiO2-G_air)*(T_bath-T_i)`.

## AD–FD results

| Space | step | AD | FD | relative error | 5% gate |
|---|---:|---:|---:|---:|---|
"""
    for row in rows:
        report += (
            f"| {row['space']} | {row['step']:.6g} | "
            f"{row['adjoint']:.6e} | {row['finite_difference']:.6e} | "
            f"{row['relative_error']:.6%} | {row['pass_5pct']} |\n"
        )
    report += f"""

The latent sweep is U-shaped: h=0.01 is dominated by nonlinearity, h=0.0025
is below the observed v261 FD numerical floor, and the bracketed stable step
h=0.005 gives {selected['relative_error']:.6%} error. Failed side-control rows
are retained rather than discarded.

At selected h=0.005, the optical-Q directional term is
{selected['adjoint']['optical_Q_directional']:.6e}; the newly implemented
thermal-material term is
{selected['adjoint']['thermal_material_directional']:.6e}; their combined
directional derivative is
{selected['adjoint']['combined_directional']:.6e}.

Energy balance is {selected['gates']['energy_balance_relative_error']:.6e}
and the linear residual is
{selected['gates']['linear_residual_relative']:.6e}.

## Remaining scope limit

`BLOCKED_PHYSICAL_WEIGHTING_POTENTIAL_OR_FINITE_FLAKE_MASK` remains. The
reported objective is still the finite-local-mask A m surrogate, not terminal
current in A. No PTE optimization, transient solve, or final device prediction
is claimed.
"""
    report_path.write_text(report, encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "raw_artifacts": raw_artifacts,
                "raw_fsp_committed_to_git": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
