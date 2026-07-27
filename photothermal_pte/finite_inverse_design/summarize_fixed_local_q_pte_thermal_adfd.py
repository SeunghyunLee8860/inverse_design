#!/usr/bin/env python3
"""Publish the fixed-local-Q PTE thermal-only AD--FD certificate."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


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


def main() -> int:
    args = parse_args()
    raw_path = Path(args.raw_summary).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    raw = json.loads(raw_path.read_text())
    if not raw.get("passed"):
        raise RuntimeError(
            f"refusing non-passing certificate: {raw['status']}"
        )
    report_path = report_dir / "FIXED_LOCAL_Q_PTE_THERMAL_ADFD_REPORT.md"
    summary_path = (
        report_dir / "fixed_local_q_pte_thermal_adfd_summary.json"
    )
    csv_path = report_dir / "fixed_local_q_pte_thermal_adfd_cases.csv"
    manifest_path = (
        report_dir
        / "FIXED_LOCAL_Q_PTE_THERMAL_ADFD_RAW_ARTIFACT_MANIFEST.json"
    )

    with csv_path.open("w", newline="") as stream:
        fields = list(raw["rows"][0])
        writer = csv.DictWriter(
            stream, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(raw["rows"])

    published = {
        "status": raw["status"],
        "passed": raw["passed"],
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": raw["scope"],
        "input_native_Q_artifact": raw["input_native_Q_artifact"],
        "native_P_Q_W": raw["native_P_Q_W"],
        "thermal_operator": raw["thermal_operator"],
        "thermal_gradient_components": raw[
            "thermal_gradient_components"
        ],
        "PTE_functional": raw["PTE_functional"],
        "scenarios": raw["scenarios"],
        "gates": raw["gates"],
        "forbidden_and_absent": [
            "Maxwell rerun",
            "optical-Q gradient",
            "81x81 nodal mapping",
            "Q clipping/smoothing/gain/rescaling/tiling/deletion",
            "transient",
            "optimization",
        ],
        "next_gate": "81X81_NODAL_TO_OPTICAL_THERMAL_MAPPING_JVP_VJP",
    }
    summary_path.write_text(json.dumps(published, indent=2) + "\n")

    scenario_lines = []
    direction_lines = []
    selected_step = raw["gates"]["selected_FD_step"]
    for scenario in raw["scenarios"]:
        scenario_lines.append(
            "| "
            + " | ".join(
                [
                    scenario["name"],
                    str(scenario["total_cells"]),
                    f"{scenario['PTE_objective_A']:.9e}",
                    f"{scenario['Tmax_DeltaT_K']:.9e}",
                    f"{scenario['gradient_norms_A']['total']:.9e}",
                    f"{scenario['forward_energy_balance_relative_error']:.3e}",
                    f"{scenario['forward_linear_residual_relative']:.3e}",
                    f"{scenario['adjoint_linear_residual_relative']:.3e}",
                ]
            )
            + " |"
        )
        for direction in scenario["directions"]:
            selected = next(
                row
                for row in direction["steps"]
                if row["step"] == selected_step
            )
            direction_lines.append(
                "| "
                + " | ".join(
                    [
                        scenario["name"],
                        direction["name"],
                        f"{direction['signal_ratio']:.3e}",
                        str(direction["included_in_gate"]),
                        f"{selected['adjoint_directional_A']:.9e}",
                        (
                            f"{selected['finite_difference_directional_A']:.9e}"
                        ),
                        f"{selected['relative_error']:.6e}",
                        f"{selected['bulk_k_A']:.9e}",
                        f"{selected['interface_G_A']:.9e}",
                        f"{selected['top_convection_k_A']:.9e}",
                    ]
                )
                + " |"
            )
    gates = raw["gates"]
    report = rf"""# Fixed-local-Q PTE thermal-only AD–FD

Status: `{raw['status']}`

## What this validates

The matched \(dz=2.5\) nm native Yee \(Q\) is remapped once per named
thermal footprint and then held bitwise identical in the baseline, plus, and
minus thermal solves.  Maxwell and the optical-\(Q\) derivative are absent.
The differentiated system is

\[
K_T(\rho)\theta=M_VQ_{{fixed}},\qquad
\frac{{dI}}{{d\rho}}=
-\lambda_T^T\frac{{dK_T}}{{d\rho}}\theta .
\]

The discrete gradient contains all three implemented thermal paths:

1. bulk design \(k(\rho)\);
2. internal TaIrTe4/design \(G(\rho)\);
3. the design exposed-surface half-cell conductivity contribution.

The objective is the established uniform-45-degree weighting-field PTE
surrogate.  It is not yet a finite-contact solved weighting potential or a
terminal experimental current.

This stage intentionally uses the native 20×20 cell-centered thermal density
control at \(\rho=0.5\).  It does not claim the approved 81×81 nodal mapping;
that mapping and its JVP/VJP are the next separate gate.

## Baselines

| scenario | cells | PTE objective (A) | Tmax ΔT (K) | gradient norm (A) | energy error | forward residual | adjoint residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(scenario_lines)}

## Centered AD–FD at selected step \(h={selected_step:g}\)

| scenario | direction | signal ratio | gated | adjoint (A) | FD (A) | relative error | bulk-k | interface-G | surface-k |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(direction_lines)}

The adjoint-aligned direction is always gated.  A fixed independent direction
is gated when its directional signal is at least
`{gates['signal_ratio_minimum']:.3e}` of the gradient L1 norm; lower-signal
directions remain published diagnostics rather than being divided by a
near-null slope.

## Gates

- Worst selected, conditioned AD–FD error:
  `{gates['worst_selected_gated_AD_FD_relative_error']:.6e}`
  (limit `{gates['AD_FD_relative_error_limit']:.6e}`).
- Worst energy-balance error:
  `{gates['worst_energy_balance_relative_error']:.6e}`
  (limit `{gates['energy_balance_limit']:.6e}`).
- Worst forward/adjoint linear residual:
  `{gates['worst_linear_residual_relative']:.6e}`
  (limit `{gates['linear_residual_limit']:.6e}`).
- Worst sum-of-gradient-components error:
  `{gates['worst_gradient_component_sum_relative_error']:.6e}`.

No Maxwell solve, 81×81 mapping, transient calculation, or optimization is
claimed here.
"""
    report_path.write_text(report)

    manifest = {
        "status": raw["status"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_command": args.generation_command,
        "raw_summary": {
            "path": str(raw_path),
            "byte_size": raw_path.stat().st_size,
            "sha256": sha256(raw_path),
        },
        "input_native_Q_artifact": raw["input_native_Q_artifact"],
        "raw_case_artifacts": raw["raw_artifacts"],
        "git_policy": (
            "raw per-scenario NPZ remains outside Git; only path, size, "
            "SHA-256, reports, JSON, and CSV are committed"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": raw["status"],
                "report": str(report_path),
                "summary": str(summary_path),
                "csv": str(csv_path),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
