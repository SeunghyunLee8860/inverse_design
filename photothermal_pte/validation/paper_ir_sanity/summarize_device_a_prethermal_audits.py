#!/usr/bin/env python3
"""Publish Device-A weighting and material-Q audits before thermal propagation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def relative_change(coarse: float, fine: float) -> float:
    return abs(fine - coarse) / max(abs(fine), 1e-300)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material-audit", type=Path, required=True)
    parser.add_argument("--weighting-100nm", type=Path, required=True)
    parser.add_argument("--weighting-50nm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    material = load(args.material_audit)
    w100 = load(args.weighting_100nm)
    w50 = load(args.weighting_50nm)
    d100 = w100["diagnostics"]
    d50 = w50["diagnostics"]
    comparison = {
        "p99_weighting_gradient_relative_change_100_to_50nm": relative_change(
            d100["weighting_gradient_p99_m_inv"],
            d50["weighting_gradient_p99_m_inv"],
        ),
        "raw_max_weighting_gradient_relative_change_100_to_50nm": relative_change(
            d100["weighting_gradient_max_m_inv"],
            d50["weighting_gradient_max_m_inv"],
        ),
        "top_contact_length_proxy_relative_change": relative_change(
            d100["top_contact_cells"] * 100.0,
            d50["top_contact_cells"] * 50.0,
        ),
        "bottom_contact_length_proxy_relative_change": relative_change(
            d100["bottom_contact_cells"] * 100.0,
            d50["bottom_contact_cells"] * 50.0,
        ),
    }
    payload = {
        "status": "COMPLETED_DEVICE_A_PRETHERMAL_AUDITS_WITH_INTERFACE_BLOCKER",
        "weighting_potential": {
            "100nm": w100,
            "50nm": w50,
            "mesh_comparison": comparison,
            "production_interpretation": (
                "the p99 weighting-field metric is stable while the one-cell "
                "raw maximum is not; raw maximum is diagnostic only"
            ),
        },
        "material_Q_support_E_parallel_a": material,
        "full_metal_inclusive_thermal_blocker": (
            "the exact common-grid power partition closes, but "
            f"{100.0 * material['power_at_unit_central_intensity_W']['ambiguous_fraction_of_common_grid_power']:.4f}% "
            "of Q is carried by lateral conformal/interface samples and "
            "finite exact Au/Ti absorption is also present; do not silently "
            "project all optical Q into TaIrTe4"
        ),
        "next_gate": (
            "complete E-parallel-b optical Q, repeat the support audit, then "
            "declare a conservative component/material interface remap before "
            "any full thermal/PTE current claim"
        ),
        "thermal_run": False,
        "PTE_run": False,
        "adjoint_run": False,
        "optimization_run": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "device_a_prethermal_audit_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    with (args.output_dir / "device_a_weighting_mesh_cases.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "core_step_nm",
                "top_contact_cells",
                "bottom_contact_cells",
                "psi_min",
                "psi_max",
                "linear_residual_relative",
                "grad_psi_p99_m_inv",
                "grad_psi_raw_max_m_inv",
            ]
        )
        for step, data in ((100, d100), (50, d50)):
            writer.writerow(
                [
                    step,
                    data["top_contact_cells"],
                    data["bottom_contact_cells"],
                    data["psi_min"],
                    data["psi_max"],
                    data["linear_residual_relative"],
                    data["weighting_gradient_p99_m_inv"],
                    data["weighting_gradient_max_m_inv"],
                ]
            )
    power = material["power_at_unit_central_intensity_W"]
    report = f"""# Device A pre-thermal audit

Status: `{payload['status']}`

This checkpoint does not contain a thermal, PTE, adjoint, or optimization run.

## Weighting potential

The frozen Figure-2 digitized contact segments were used with code axes
`x=b`, `y=a`. Both 100 nm and 50 nm grids pass the contact, finite-field,
potential-range, and residual gates.

| Metric | 100 nm | 50 nm | relative change |
|---|---:|---:|---:|
| p99 $|\\nabla\\psi|$ [1/m] | {d100['weighting_gradient_p99_m_inv']:.9e} | {d50['weighting_gradient_p99_m_inv']:.9e} | {comparison['p99_weighting_gradient_relative_change_100_to_50nm']:.4%} |
| raw max $|\\nabla\\psi|$ [1/m] | {d100['weighting_gradient_max_m_inv']:.9e} | {d50['weighting_gradient_max_m_inv']:.9e} | {comparison['raw_max_weighting_gradient_relative_change_100_to_50nm']:.4%} |
| residual | {d100['linear_residual_relative']:.3e} | {d50['linear_residual_relative']:.3e} | -- |

The robust p99 metric is stable; the one-cell raw maximum is not a production
gate.

## E-parallel-a material-Q support

| Partition | Power at unit central intensity [W] |
|---|---:|
| TaIrTe4 exact support | {power['TaIrTe4_W']:.12e} |
| Ti exact support | {power['Ti_W']:.12e} |
| Au exact support | {power['Au_W']:.12e} |
| conformal/interface ambiguous | {power['conformal_interface_ambiguous_W']:.12e} |
| common-grid total | {power['common_grid_total_W']:.12e} |

Partition closure is {power['relative_partition_residual']:.3e}; the common-grid
versus native-component total difference is
{power['relative_common_native_difference']:.4%}. The ambiguous fraction is
{power['ambiguous_fraction_of_common_grid_power']:.4%}. It is retained rather
than clipped, rescaled, deleted, or silently assigned to a bulk material.

## Fail-closed consequence

The current thermal mapper projects every optical-Q sample into TaIrTe4. That
path must not be used for this electrode-bearing Device-A artifact. A declared
conservative component/material interface remap and an explicit metal thermal
scenario are required before a full terminal-current result can be promoted.
"""
    (args.output_dir / "DEVICE_A_PRETHERMAL_AUDIT_REPORT.md").write_text(report)
    print(json.dumps({"status": payload["status"], **comparison}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
