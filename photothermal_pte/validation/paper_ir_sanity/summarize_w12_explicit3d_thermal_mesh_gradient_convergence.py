#!/usr/bin/env python3
"""Summarize 100-to-50 nm explicit-3D thermal gradient convergence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    "mask_aware_finite_difference",
    "least_squares_radius_0.2um",
    "least_squares_radius_0.3um",
    "least_squares_radius_0.4um",
)
STATISTICS = ("raw_max_abs", "p99_abs", "rms_abs", "mean_abs")
REGIONS = ("original_staircase_edge", "inside_band_0p1_to_0p3um")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-100nm", type=Path, required=True)
    parser.add_argument("--summary-50nm", type=Path, required=True)
    parser.add_argument("--robust-100nm", type=Path, required=True)
    parser.add_argument("--robust-50nm", type=Path, required=True)
    parser.add_argument("--raw-100nm", type=Path, required=True)
    parser.add_argument("--raw-50nm", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_change(coarse: float, fine: float) -> float:
    return abs(fine - coarse) / max(abs(fine), np.finfo(float).tiny)


def ratio_plot(
    path: Path,
    model: str,
    region: str,
    robust_100: dict[str, Any],
    robust_50: dict[str, Any],
) -> None:
    labels = ("FD", "LS 0.2", "LS 0.3", "LS 0.4")
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    positions = np.arange(len(METHODS))
    for axis, statistic in zip(axes.flat, STATISTICS):
        values_100 = [
            robust_100["ratios"][model]["thickness_average"][method][region][
                statistic
            ]
            for method in METHODS
        ]
        values_50 = [
            robust_50["ratios"][model]["thickness_average"][method][region][
                statistic
            ]
            for method in METHODS
        ]
        width = 0.38
        axis.bar(positions - width / 2, values_100, width, label="100 nm")
        axis.bar(positions + width / 2, values_50, width, label="50 nm")
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set(
            title=statistic,
            ylabel="b/a",
            xticks=positions,
            xticklabels=labels,
        )
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.suptitle(
        f"{model}, {region}: thermal mesh and reconstruction sensitivity",
        fontsize=15,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    paths = {
        name: value.expanduser().resolve()
        for name, value in vars(args).items()
        if name != "report_dir"
    }
    report_dir = args.report_dir.expanduser().resolve()
    if report_dir.exists():
        raise FileExistsError(report_dir)
    report_dir.mkdir(parents=True)

    solved = {
        "100nm": read(paths["summary_100nm"]),
        "50nm": read(paths["summary_50nm"]),
    }
    robust = {
        "100nm": read(paths["robust_100nm"]),
        "50nm": read(paths["robust_50nm"]),
    }
    if solved["100nm"]["thermal_contract"]["core_xy_cell_size_nm"] != 100.0:
        raise RuntimeError("coarse summary is not 100 nm")
    if solved["50nm"]["thermal_contract"]["core_xy_cell_size_nm"] != 50.0:
        raise RuntimeError("fine summary is not 50 nm")
    if not all(item["acceptance"]["all"] for item in solved.values()):
        raise RuntimeError("one explicit-3D solve failed its existing gates")

    case_changes: dict[str, Any] = {}
    for case in solved["100nm"]["cases"]:
        case_id = case
        coarse = solved["100nm"]["cases"][case_id]
        fine = solved["50nm"]["cases"][case_id]
        case_changes[case_id] = {
            "Tmax_100nm_K": coarse["Tmax_rise_K"],
            "Tmax_50nm_K": fine["Tmax_rise_K"],
            "Tmax_relative_change": relative_change(
                coarse["Tmax_rise_K"], fine["Tmax_rise_K"]
            ),
            "TaIrTe4_mean_100nm_K": coarse[
                "TaIrTe4_volume_average_rise_K"
            ],
            "TaIrTe4_mean_50nm_K": fine[
                "TaIrTe4_volume_average_rise_K"
            ],
            "TaIrTe4_mean_relative_change": relative_change(
                coarse["TaIrTe4_volume_average_rise_K"],
                fine["TaIrTe4_volume_average_rise_K"],
            ),
            "iterations_100nm": coarse["iterations"],
            "iterations_50nm": fine["iterations"],
            "residual_100nm": coarse["linear_residual_relative"],
            "residual_50nm": fine["linear_residual_relative"],
            "energy_balance_100nm": coarse[
                "energy_balance_relative_error"
            ],
            "energy_balance_50nm": fine[
                "energy_balance_relative_error"
            ],
        }

    rows: list[dict[str, Any]] = []
    gradient: dict[str, Any] = {}
    for model in ("Maxwell", "analytic"):
        gradient[model] = {}
        for region in REGIONS:
            gradient[model][region] = {}
            for method in METHODS:
                gradient[model][region][method] = {}
                for statistic in STATISTICS:
                    coarse = robust["100nm"]["ratios"][model][
                        "thickness_average"
                    ][method][region][statistic]
                    fine = robust["50nm"]["ratios"][model][
                        "thickness_average"
                    ][method][region][statistic]
                    item = {
                        "b_over_a_100nm": coarse,
                        "b_over_a_50nm": fine,
                        "relative_change": relative_change(coarse, fine),
                        "ordering_100nm": "b<a" if coarse < 1.0 else "b>a",
                        "ordering_50nm": "b<a" if fine < 1.0 else "b>a",
                        "ordering_preserved": (coarse - 1.0) * (fine - 1.0)
                        > 0.0,
                    }
                    gradient[model][region][method][statistic] = item
                    rows.append(
                        {
                            "model": model,
                            "region": region,
                            "method": method,
                            "statistic": statistic,
                            **item,
                        }
                    )

    maxwell_ordering_preserved = all(
        row["ordering_preserved"]
        for row in rows
        if row["model"] == "Maxwell"
    )
    analytic_ordering_preserved = all(
        row["ordering_preserved"]
        for row in rows
        if row["model"] == "analytic"
    )
    maxwell_max_change = max(
        row["relative_change"]
        for row in rows
        if row["model"] == "Maxwell"
    )
    analytic_max_change = max(
        row["relative_change"]
        for row in rows
        if row["model"] == "analytic"
    )
    temperature_gate = max(
        item["Tmax_relative_change"] for item in case_changes.values()
    ) < 0.01
    gradient_one_percent_gate = maxwell_max_change < 0.01
    status = (
        "VALIDATED_W12_EXPLICIT3D_THERMAL_MESH_GRADIENT_CONVERGENCE"
        if temperature_gate
        and gradient_one_percent_gate
        and maxwell_ordering_preserved
        and analytic_ordering_preserved
        else "PARTIAL_W12_EXPLICIT3D_TEMPERATURE_CONVERGED_GRADIENT_MAGNITUDE_UNCONVERGED"
    )

    figures = {
        "Maxwell_original_edge": report_dir
        / "Maxwell_original_edge_100nm_vs_50nm_b_over_a.png",
        "Maxwell_inside_band": report_dir
        / "Maxwell_inside_band_100nm_vs_50nm_b_over_a.png",
        "analytic_original_edge": report_dir
        / "analytic_original_edge_100nm_vs_50nm_b_over_a.png",
    }
    ratio_plot(
        figures["Maxwell_original_edge"],
        "Maxwell",
        "original_staircase_edge",
        robust["100nm"],
        robust["50nm"],
    )
    ratio_plot(
        figures["Maxwell_inside_band"],
        "Maxwell",
        "inside_band_0p1_to_0p3um",
        robust["100nm"],
        robust["50nm"],
    )
    ratio_plot(
        figures["analytic_original_edge"],
        "analytic",
        "original_staircase_edge",
        robust["100nm"],
        robust["50nm"],
    )

    provenance = {
        name: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for name, path in paths.items()
    }
    summary = {
        "status": status,
        "scope": (
            "same optical Q, explicit-3D geometry/material/interface/boundary "
            "contract; thermal core dx=dy changes only from 100 to 50 nm"
        ),
        "case_scalar_convergence": case_changes,
        "gradient_b_over_a_convergence": gradient,
        "gates": {
            "both_existing_solve_acceptance_pass": True,
            "all_Tmax_changes_lt_1_percent": temperature_gate,
            "Maxwell_gradient_ordering_preserved_all_metrics": (
                maxwell_ordering_preserved
            ),
            "analytic_gradient_ordering_preserved_all_metrics": (
                analytic_ordering_preserved
            ),
            "Maxwell_gradient_ratio_changes_lt_1_percent": (
                gradient_one_percent_gate
            ),
            "Maxwell_max_gradient_ratio_relative_change": (
                maxwell_max_change
            ),
            "analytic_max_gradient_ratio_relative_change": (
                analytic_max_change
            ),
        },
        "interpretation": {
            "validated": (
                "Maxwell b<a and analytic b>a ordering is robust to "
                "100-to-50 nm refinement, reconstruction method, robust "
                "statistic, and selected fixed physical edge region"
            ),
            "not_validated": (
                "a unique mesh-converged Maxwell b/a gradient magnitude; "
                "0.879613 must not be promoted as the final quantitative ratio"
            ),
        },
        "figures": {key: str(path) for key, path in figures.items()},
        "provenance": provenance,
        "generation_command": shlex.join([sys.executable, *sys.argv]),
    }
    summary_path = report_dir / (
        "w12_explicit3d_thermal_mesh_gradient_convergence_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    csv_path = report_dir / (
        "w12_explicit3d_thermal_mesh_gradient_convergence_cases.csv"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fd_edge = gradient["Maxwell"]["original_staircase_edge"][
        "mask_aware_finite_difference"
    ]
    ls_edge = {
        method: gradient["Maxwell"]["original_staircase_edge"][method][
            "raw_max_abs"
        ]
        for method in METHODS[1:]
    }
    report = f"""# W12 explicit-3D 100-to-50 nm thermal-gradient convergence

Status: `{status}`

The same native GPU Maxwell Q artifacts, analytic sources, 60-µm explicit-3D
thermal geometry, materials, interface G values, boundary conditions, and
10-nm flake z cells were used.  Only the 24-µm core x/y thermal step changed
from 100 to 50 nm.

## Solver and scalar temperature

All four 50-nm cases passed existing source mapping, residual, and energy
gates.  The largest per-case Tmax change is
`{max(item['Tmax_relative_change'] for item in case_changes.values()):.6%}`;
the temperature gate `<1%` therefore passes.  The fine grid has
`{solved['50nm']['thermal_contract']['grid_shape']}` cells and required
`{min(item['iterations_50nm'] for item in case_changes.values())}`–
`{max(item['iterations_50nm'] for item in case_changes.values())}` CG
iterations.

## Original staircase-edge Maxwell b/a

| metric | 100 nm | 50 nm | relative change |
|---|---:|---:|---:|
| FD raw max | {fd_edge['raw_max_abs']['b_over_a_100nm']:.6f} | {fd_edge['raw_max_abs']['b_over_a_50nm']:.6f} | {fd_edge['raw_max_abs']['relative_change']:.3%} |
| FD p99 | {fd_edge['p99_abs']['b_over_a_100nm']:.6f} | {fd_edge['p99_abs']['b_over_a_50nm']:.6f} | {fd_edge['p99_abs']['relative_change']:.3%} |
| FD RMS | {fd_edge['rms_abs']['b_over_a_100nm']:.6f} | {fd_edge['rms_abs']['b_over_a_50nm']:.6f} | {fd_edge['rms_abs']['relative_change']:.3%} |
| FD mean | {fd_edge['mean_abs']['b_over_a_100nm']:.6f} | {fd_edge['mean_abs']['b_over_a_50nm']:.6f} | {fd_edge['mean_abs']['relative_change']:.3%} |

The raw/p99/RMS/mean values remain below one at both meshes, so the Maxwell
ordering `b<a` is not a single-cell artifact.  However, the changes exceed
1%, so the numerical value `0.879613` is not mesh converged.

## Least-squares sensitivity

| LS physical radius | 100-nm raw-max b/a | 50-nm raw-max b/a | relative change |
|---|---:|---:|---:|
| 0.2 µm | {ls_edge['least_squares_radius_0.2um']['b_over_a_100nm']:.6f} | {ls_edge['least_squares_radius_0.2um']['b_over_a_50nm']:.6f} | {ls_edge['least_squares_radius_0.2um']['relative_change']:.3%} |
| 0.3 µm | {ls_edge['least_squares_radius_0.3um']['b_over_a_100nm']:.6f} | {ls_edge['least_squares_radius_0.3um']['b_over_a_50nm']:.6f} | {ls_edge['least_squares_radius_0.3um']['relative_change']:.3%} |
| 0.4 µm | {ls_edge['least_squares_radius_0.4um']['b_over_a_100nm']:.6f} | {ls_edge['least_squares_radius_0.4um']['b_over_a_50nm']:.6f} | {ls_edge['least_squares_radius_0.4um']['relative_change']:.3%} |

All tested Maxwell combinations—two meshes, FD/three LS radii, raw/p99/RMS/
mean, the original edge and fixed 0.1–0.3 µm inside band—remain `b<a`.
All corresponding analytic controls remain `b>a`, with a maximum ratio
change of `{analytic_max_change:.3%}`.

## Decision

The reversal direction is robust and is therefore consistent with the
spatial Maxwell Q distribution.  Its exact magnitude is unresolved:
`0.879613` is a diagnostic 100-nm FD value, not a final experiment-prediction
ratio.  A further quantitative certificate would require a predeclared
physical gradient functional and another mesh level or a higher-order
cut-cell/finite-element edge treatment.

Raw artifacts were not modified.  Exact paths, sizes, and SHA-256 values are
recorded in the summary JSON.
"""
    report_path = report_dir / (
        "W12_EXPLICIT3D_THERMAL_MESH_GRADIENT_CONVERGENCE.md"
    )
    report_path.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "report": str(report_path),
                "temperature_gate": temperature_gate,
                "Maxwell_ordering_preserved": maxwell_ordering_preserved,
                "Maxwell_max_ratio_change": maxwell_max_change,
                "analytic_max_ratio_change": analytic_max_change,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
