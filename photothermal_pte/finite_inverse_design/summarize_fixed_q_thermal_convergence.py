#!/usr/bin/env python3
"""Publish the fixed-Q thermal convergence certificate."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .contract import (
    G_SIO2_SI_W_M2K,
    G_TAIRTE4_AIR_W_M2K,
    G_TAIRTE4_BOTTOM_SIO2_W_M2K,
    G_TAIRTE4_DEPOSITED_SIO2_W_M2K,
    H_SIO2_AIR_W_M2K,
    KAPPA_AIR_W_MK,
    KAPPA_SI_W_MK,
    KAPPA_SIO2_W_MK,
    KAPPA_TAIRTE4_W_MK,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-summary", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--generation-command", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percent(value: float) -> str:
    return f"{100.0 * value:.6f}%"


def main() -> int:
    args = parse_args()
    raw_summary_path = Path(args.raw_summary).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    raw = json.loads(raw_summary_path.read_text())
    if not raw.get("passed"):
        raise RuntimeError(
            f"refusing to publish non-passing raw result: {raw['status']}"
        )

    report_path = report_dir / "FIXED_Q_THERMAL_CONVERGENCE_REPORT.md"
    summary_path = (
        report_dir / "fixed_q_thermal_convergence_summary.json"
    )
    csv_path = report_dir / "fixed_q_thermal_convergence_cases.csv"
    manifest_path = (
        report_dir / "FIXED_Q_THERMAL_CONVERGENCE_RAW_ARTIFACT_MANIFEST.json"
    )

    rows = []
    for scenario in raw["scenarios"]:
        for case in scenario["cases"]:
            comparison = case["comparison_to_native"] or {}
            rows.append(
                {
                    "scenario": scenario["name"],
                    "flake_span_um": scenario["flake_span_um"],
                    "case": case["label"],
                    **case["controls"],
                    "cells": case["total_cells"],
                    "source_power_W": case["source_power_W"],
                    "source_mapping_error": case["source"][
                        "relative_power_error"
                    ],
                    "Tmax_DeltaT_K": case["Tmax_DeltaT_K"],
                    "TaIrTe4_volume_average_DeltaT_K": case[
                        "TaIrTe4_volume_average_DeltaT_K"
                    ],
                    "PTE_objective_A": case["PTE_objective_A"],
                    "PTE_cancellation_ratio": case[
                        "PTE_cancellation_ratio"
                    ],
                    "field_probe_NRMSE": comparison.get(
                        "TaIrTe4_field_probe_NRMSE"
                    ),
                    "Tmax_relative_difference": comparison.get(
                        "Tmax_relative_difference"
                    ),
                    "flake_average_relative_difference": comparison.get(
                        "TaIrTe4_volume_average_relative_difference"
                    ),
                    "PTE_raw_relative_difference": comparison.get(
                        "PTE_raw_relative_difference"
                    ),
                    "PTE_contribution_normalized_difference": comparison.get(
                        "PTE_contribution_normalized_difference"
                    ),
                    "energy_balance_relative_error": case[
                        "energy_balance_relative_error"
                    ],
                    "linear_residual_relative": case[
                        "linear_residual_relative"
                    ],
                    "raw_sha256": case["raw_artifact"]["sha256"],
                }
            )
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    published = {
        "status": raw["status"],
        "passed": raw["passed"],
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": raw["scope"],
        "native_P_Q_W": raw["native_P_Q_W"],
        "input_native_Q_artifact": raw["input_native_Q_artifact"],
        "case_contract": raw["case_contract"],
        "common_probe": raw["common_probe"],
        "boundary_flux_interpretation": raw[
            "boundary_flux_interpretation"
        ],
        "gates": raw["gates"],
        "thermal_model": {
            "TaIrTe4_kappa_W_mK": KAPPA_TAIRTE4_W_MK.tolist(),
            "SiO2_kappa_W_mK": KAPPA_SIO2_W_MK,
            "Si_kappa_W_mK": KAPPA_SI_W_MK,
            "air_kappa_W_mK": KAPPA_AIR_W_MK,
            "TaIrTe4_bottom_SiO2_G_W_m2K": (
                G_TAIRTE4_BOTTOM_SIO2_W_M2K
            ),
            "TaIrTe4_air_G_W_m2K": G_TAIRTE4_AIR_W_M2K,
            "TaIrTe4_deposited_design_SiO2_G_W_m2K": (
                G_TAIRTE4_DEPOSITED_SIO2_W_M2K
            ),
            "SiO2_Si_G_W_m2K": G_SIO2_SI_W_M2K,
            "exposed_SiO2_air_h_W_m2K": H_SIO2_AIR_W_M2K,
            "gray_bulk_k_law": (
                "k_air + rho*(k_SiO2-k_air), componentwise"
            ),
            "gray_top_G_law": (
                "G_air + rho*(G_deposited_SiO2-G_air)"
            ),
            "rho_in_this_checkpoint": 0.5,
        },
        "scenarios": raw["scenarios"],
        "forbidden_and_absent": [
            "Q clipping",
            "Q smoothing",
            "empirical gain",
            "global Q rescaling",
            "periodic tiling",
            "source deletion",
            "optimization",
            "transient solve",
        ],
        "next_gate": "FIXED_LOCAL_Q_PTE_THERMAL_ONLY_AD_FD",
    }
    summary_path.write_text(json.dumps(published, indent=2) + "\n")

    table_lines = []
    for scenario in raw["scenarios"]:
        for case in scenario["cases"]:
            comparison = case["comparison_to_native"]
            table_lines.append(
                "| "
                + " | ".join(
                    [
                        scenario["name"],
                        case["label"],
                        str(case["total_cells"]),
                        f"{case['Tmax_DeltaT_K']:.9e}",
                        f"{case['TaIrTe4_volume_average_DeltaT_K']:.9e}",
                        f"{case['PTE_objective_A']:.9e}",
                        (
                            "baseline"
                            if comparison is None
                            else percent(
                                comparison["TaIrTe4_field_probe_NRMSE"]
                            )
                        ),
                        (
                            "baseline"
                            if comparison is None
                            else percent(
                                comparison[
                                    "maximum_thermal_metric_difference"
                                ]
                            )
                        ),
                    ]
                )
                + " |"
            )
    gates = raw["gates"]
    report = rf"""# Fixed-local-Q thermal domain/depth/mesh convergence

Status: `{raw['status']}`

## Scope and interpretation

This certificate reuses only the immutable native Yee-grid absorption arrays
from the matched \(dz=2.5\) nm optical run.  Every thermal target grid is
remapped independently.  It does not reuse a previously mapped source and
does not run Maxwell, an adjoint, finite differences, a transient solve, or
optimization.

The 4 µm and 6 µm TaIrTe4 footprints are separate named numerical scenarios.
Neither is promoted as fabrication truth or as a final experimental
prediction.  Lateral and bottom Dirichlet power entries are numerical
truncation-boundary fluxes, not intrinsic physical heat-path fractions.

Native optical power is `{raw['native_P_Q_W']:.16e} W`.  There is no
clipping, smoothing, gain, global rescaling, tiling, or source deletion.

## Thermal physical model held fixed

- TaIrTe4:
  \(\boldsymbol{{\kappa}}=\operatorname{{diag}}(14.4,3.8,1.0)\)
  W/(m K).
- Bulk SiO2 / Si / air:
  `1.38 / 145 / 0.026 W/(m K)`.
- TaIrTe4/bottom-SiO2:
  `G=7.37e6 W/(m2 K)`; SiO2/Si:
  `G=1.1e9 W/(m2 K)`.
- TaIrTe4/air: `G=1 W/(m2 K)`.
- Deposited design-SiO2 endpoint:
  `G=7.37e4 W/(m2 K)`.
- Exposed SiO2/air: `h=10 W/(m2 K)`.
- This checkpoint holds \(\rho=0.5\), with
  \(k(\rho)=k_{{air}}+\rho(k_{{SiO2}}-k_{{air}})\) and
  \(G(\rho)=G_{{air}}+\rho(G_{{deposited\ SiO2}}-G_{{air}})\).
  Those gray laws are numerical relaxations, not measured gray-composite
  properties.  Their sensitivity is a later, separate gate.

## Independent controls

- Native: 32 µm lateral, 20 µm Si depth, 100/25/100 nm
  core-xy/flake-z/design-z grid.
- Lateral: only the lateral domain is enlarged to 40 µm.
- Depth: only Si depth is enlarged to 30 µm.
- Refined: the domain/depth remain 32/20 µm and the grid becomes
  50/12.5/50 nm.
- The complete TaIrTe4 field is compared on a fixed 100 nm by 100 nm by
  25 nm common probe grid using trilinear cell-center interpolation.

## Results

| scenario | case | cells | Tmax ΔT (K) | TaIrTe4 average ΔT (K) | PTE objective (A) | field NRMSE | worst gated difference |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table_lines)}

The raw relative PTE change is retained in the JSON/CSV.  Because a uniform
45-degree weighting field can strongly cancel a nearly symmetric
temperature field, the convergence gate uses
\(|I-I_0|/\max(\sum |w_i T_i|)\), not a potentially ill-conditioned division
by a near-zero signed current.

## Gates

- Worst Q mapping power error: `{gates['worst_mapping_power_error']:.6e}`
  (limit `{gates['mapping_power_limit']:.6e}`).
- Worst energy-balance error:
  `{gates['worst_energy_balance_error']:.6e}` (limit
  `{gates['energy_balance_limit']:.6e}`).
- Worst linear residual: `{gates['worst_linear_residual']:.6e}` (limit
  `{gates['linear_residual_limit']:.6e}`).
- Worst temperature/PTE convergence metric:
  `{gates['worst_convergence_metric']:.6e}` (limit
  `{gates['convergence_metric_limit']:.6e}`).

The next gate is fixed-local-Q PTE thermal-only AD–FD.  This report does not
claim that gate has run.
"""
    report_path.write_text(report)

    manifest = {
        "status": raw["status"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_command": args.generation_command,
        "raw_summary": {
            "path": str(raw_summary_path),
            "byte_size": raw_summary_path.stat().st_size,
            "sha256": sha256(raw_summary_path),
        },
        "input_native_Q_artifact": raw["input_native_Q_artifact"],
        "raw_case_artifacts": raw["raw_artifacts"],
        "git_policy": (
            "raw per-case NPZ artifacts remain outside Git; only this "
            "path/size/SHA-256 manifest is committed"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "status": raw["status"],
        "report": str(report_path),
        "summary": str(summary_path),
        "csv": str(csv_path),
        "manifest": str(manifest_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
