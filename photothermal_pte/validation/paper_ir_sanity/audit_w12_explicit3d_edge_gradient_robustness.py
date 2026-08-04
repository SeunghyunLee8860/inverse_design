#!/usr/bin/env python3
"""Offline robustness audit for the W12 explicit-3D edge gradient.

This command does not run Maxwell or thermal solvers.  It compares the
checkpointed mask-aware finite-difference gradient against local 2-D
least-squares plane fits, and replaces a single raw edge maximum with several
fixed-region statistics.
"""

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


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    run_device_a_explicit_thermal_pte as thermal,
)
from photothermal_pte.validation.paper_ir_sanity.coordinate_plot import (  # noqa: E402
    cell_field,
    strict_centered_xy_mask,
)


CASES = ("Maxwell_a", "Maxwell_b", "analytic_a", "analytic_b")
PROJECTIONS = ("surface", "midplane", "thickness_average")
RADII_M = (0.2e-6, 0.3e-6, 0.4e-6)
SQRT2 = np.sqrt(2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def original_edge_mask(mask: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    air = ~mask
    air_minus_x = np.zeros_like(mask)
    air_minus_x[1:, :] = air[:-1, :]
    air_plus_y = np.zeros_like(mask)
    air_plus_y[:, :-1] = air[:, 1:]
    return (
        mask
        & (air_minus_x | air_plus_y)
        & (np.abs(tangent) <= 10.0e-6)
    )


def local_plane_gradient(
    temperature: np.ndarray,
    mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    evaluation_mask: np.ndarray,
    radius_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit T=T0+gx*dx+gy*dy to nearby same-material cell centres."""
    gx = np.full_like(temperature, np.nan)
    gy = np.full_like(temperature, np.nan)
    counts: list[int] = []
    ranks: list[int] = []
    condition_numbers: list[float] = []
    for i, j in zip(*np.nonzero(evaluation_mask)):
        i0 = int(np.searchsorted(x, x[i] - radius_m, side="left"))
        i1 = int(np.searchsorted(x, x[i] + radius_m, side="right"))
        j0 = int(np.searchsorted(y, y[j] - radius_m, side="left"))
        j1 = int(np.searchsorted(y, y[j] + radius_m, side="right"))
        xx = x[i0:i1, None] - x[i]
        yy = y[None, j0:j1] - y[j]
        local_mask = mask[i0:i1, j0:j1] & (
            xx**2 + yy**2 <= radius_m**2 * (1.0 + 1.0e-12)
        )
        local_i, local_j = np.nonzero(local_mask)
        if local_i.size < 6:
            continue
        dx = x[i0 + local_i] - x[i]
        dy = y[j0 + local_j] - y[j]
        design = np.column_stack(
            (
                np.ones(local_i.size),
                dx / radius_m,
                dy / radius_m,
            )
        )
        values = temperature[i0 + local_i, j0 + local_j]
        coefficient, _, rank, singular = np.linalg.lstsq(
            design, values, rcond=None
        )
        if rank < 3:
            continue
        gx[i, j] = coefficient[1] / radius_m
        gy[i, j] = coefficient[2] / radius_m
        counts.append(int(local_i.size))
        ranks.append(int(rank))
        condition_numbers.append(float(singular[0] / singular[-1]))
    valid = evaluation_mask & np.isfinite(gx) & np.isfinite(gy)
    return gx, gy, {
        "radius_m": radius_m,
        "requested_cell_count": int(np.count_nonzero(evaluation_mask)),
        "valid_cell_count": int(np.count_nonzero(valid)),
        "minimum_neighbour_count": min(counts) if counts else 0,
        "maximum_neighbour_count": max(counts) if counts else 0,
        "maximum_condition_number": (
            max(condition_numbers) if condition_numbers else float("inf")
        ),
        "all_rank_three": bool(ranks and min(ranks) == 3),
    }


def statistics(
    gradient_n: np.ndarray,
    selected: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    valid = selected & np.isfinite(gradient_n)
    values = np.abs(gradient_n[valid])
    if values.size == 0:
        raise RuntimeError("empty gradient statistic region")
    maximum_index = int(
        np.argmax(np.where(valid, np.abs(gradient_n), -np.inf))
    )
    i, j = np.unravel_index(maximum_index, gradient_n.shape)
    return {
        "cell_count": int(values.size),
        "raw_max_abs_K_m": float(np.max(values)),
        "p99_abs_K_m": float(np.percentile(values, 99.0)),
        "rms_abs_K_m": float(np.sqrt(np.mean(values**2))),
        "mean_abs_K_m": float(np.mean(values)),
        "median_abs_K_m": float(np.median(values)),
        "raw_max_over_p99": float(
            np.max(values) / max(np.percentile(values, 99.0), np.finfo(float).tiny)
        ),
        "raw_max_location_m": {"x": float(x[i]), "y": float(y[j])},
    }


def difference_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    selected: np.ndarray,
) -> dict[str, float]:
    valid = selected & np.isfinite(reference) & np.isfinite(candidate)
    ref = reference[valid]
    test = candidate[valid]
    denominator = max(float(np.linalg.norm(ref)), np.finfo(float).tiny)
    return {
        "cell_count": int(ref.size),
        "NRMSE": float(np.linalg.norm(test - ref) / denominator),
        "correlation": float(np.corrcoef(ref, test)[0, 1]),
        "mean_signed_difference_K_m": float(np.mean(test - ref)),
    }


def shared_scale_figure(
    path: Path,
    model: str,
    projection: str,
    finite: dict[str, np.ndarray],
    least_squares: dict[str, np.ndarray],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    display_mask: np.ndarray,
) -> None:
    cases = (f"{model}_a", f"{model}_b")
    arrays = [
        finite[case] for case in cases
    ] + [
        least_squares[case] for case in cases
    ]
    limit = max(
        float(np.nanmax(np.abs(np.where(display_mask, item, np.nan))))
        for item in arrays
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 10), constrained_layout=True)
    for row, case in enumerate(cases):
        for column, (name, collection) in enumerate(
            (("finite difference", finite), ("LS radius 0.3 µm", least_squares))
        ):
            image = cell_field(
                axes[row, column],
                x_edges,
                y_edges,
                np.where(display_mask, collection[case], np.nan),
                coordinate_scale=1e6,
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
            axes[row, column].set(
                title=f"{case.replace('_', ' ')}, {name}",
                xlabel="x=b (µm)",
                ylabel="y=a (µm)",
                xlim=(-12, 12),
                ylim=(-12, 12),
            )
            figure.colorbar(image, ax=axes[row, column], label="∂nT (K/m)")
    figure.suptitle(
        f"{model} {projection}: one common color scale", fontsize=15
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def ratio(
    metrics: dict[str, dict[str, Any]],
    case_a: str,
    case_b: str,
) -> dict[str, float]:
    return {
        key.replace("_K_m", ""): (
            metrics[case_b][key] / metrics[case_a][key]
        )
        for key in (
            "raw_max_abs_K_m",
            "p99_abs_K_m",
            "rms_abs_K_m",
            "mean_abs_K_m",
            "median_abs_K_m",
        )
    }


def main() -> int:
    args = parse_args()
    input_path = args.input_npz.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    if report_dir.exists():
        raise FileExistsError(report_dir)
    report_dir.mkdir(parents=True)

    with np.load(input_path, allow_pickle=False) as raw:
        x_edges = np.asarray(raw["x_edges_m"], float)
        y_edges = np.asarray(raw["y_edges_m"], float)
        x = 0.5 * (x_edges[:-1] + x_edges[1:])
        y = 0.5 * (y_edges[:-1] + y_edges[1:])
        mask = np.any(np.asarray(raw["flake_mask"], bool), axis=2)
        n = (-x[:, None] + y[None, :]) / SQRT2
        tangent = (x[:, None] + y[None, :]) / SQRT2
        edge = original_edge_mask(mask, tangent)
        evaluation = (
            mask
            & (n >= -0.65e-6)
            & (n <= 0.0)
            & (np.abs(tangent) <= 11.0e-6)
        )
        display = (
            mask
            & (n >= -2.0e-6)
            & (n <= 0.0)
            & (np.abs(tangent) <= 12.0e-6)
        )
        strict_display = display & strict_centered_xy_mask(mask)
        regions = {
            "original_staircase_edge": edge,
            "inside_n_0p1um": (
                mask
                & (np.abs(n + 0.1e-6) <= 0.06e-6)
                & (np.abs(tangent) <= 10.0e-6)
            ),
            "inside_n_0p2um": (
                mask
                & (np.abs(n + 0.2e-6) <= 0.06e-6)
                & (np.abs(tangent) <= 10.0e-6)
            ),
            "inside_n_0p3um": (
                mask
                & (np.abs(n + 0.3e-6) <= 0.06e-6)
                & (np.abs(tangent) <= 10.0e-6)
            ),
            "inside_band_0p1_to_0p3um": (
                mask
                & (n >= -0.3e-6)
                & (n <= -0.1e-6)
                & (np.abs(tangent) <= 10.0e-6)
            ),
        }

        summary: dict[str, Any] = {
            "status": "COMPLETED_OFFLINE_W12_EXPLICIT3D_EDGE_GRADIENT_ROBUSTNESS_AUDIT",
            "scope": "stored T only; no new FDTD or thermal solve",
            "input": {
                "path": str(input_path),
                "size_bytes": input_path.stat().st_size,
                "sha256": sha256(input_path),
            },
            "coordinate_contract": {
                "x": "crystal b",
                "y": "crystal a",
                "n": "(-x+y)/sqrt(2)",
                "t": "(x+y)/sqrt(2)",
            },
            "regions": {
                key: {
                    "cell_count": int(np.count_nonzero(value)),
                    "definition": key,
                }
                for key, value in regions.items()
            },
            "cases": {},
            "ratios": {},
        }
        csv_rows: list[dict[str, Any]] = []
        finite_for_figure: dict[str, np.ndarray] = {}
        ls_for_figure: dict[str, np.ndarray] = {}

        for case in CASES:
            summary["cases"][case] = {}
            for projection in PROJECTIONS:
                temperature = np.asarray(
                    raw[f"{case}__temperature_{projection}_K"], float
                )
                gx_fd, gy_fd = thermal.cell_gradient(
                    temperature, mask, x, y
                )
                gn_fd = (-gx_fd + gy_fd) / SQRT2
                methods: dict[str, np.ndarray] = {
                    "mask_aware_finite_difference": gn_fd
                }
                method_audits: dict[str, Any] = {
                    "mask_aware_finite_difference": {
                        "definition": (
                            "centered interior; TaIrTe4-side one-sided x/y "
                            "differences at the staircase edge"
                        )
                    }
                }
                for radius_m in RADII_M:
                    gx_ls, gy_ls, ls_audit = local_plane_gradient(
                        temperature,
                        mask,
                        x,
                        y,
                        evaluation,
                        radius_m,
                    )
                    name = f"least_squares_radius_{radius_m*1e6:.1f}um"
                    methods[name] = (-gx_ls + gy_ls) / SQRT2
                    method_audits[name] = ls_audit
                projection_summary: dict[str, Any] = {
                    "method_audit": method_audits,
                    "methods": {},
                    "least_squares_vs_finite_difference": {},
                }
                for method, field in methods.items():
                    projection_summary["methods"][method] = {}
                    for region, selected in regions.items():
                        stat = statistics(field, selected, x, y)
                        projection_summary["methods"][method][region] = stat
                        csv_rows.append(
                            {
                                "case": case,
                                "projection": projection,
                                "method": method,
                                "region": region,
                                **{
                                    key: value
                                    for key, value in stat.items()
                                    if not isinstance(value, dict)
                                },
                            }
                        )
                    if method != "mask_aware_finite_difference":
                        projection_summary[
                            "least_squares_vs_finite_difference"
                        ][method] = difference_metrics(
                            gn_fd, field, evaluation
                        )
                summary["cases"][case][projection] = projection_summary
                if projection == "thickness_average":
                    finite_for_figure[case] = gn_fd
                    ls_for_figure[case] = methods[
                        "least_squares_radius_0.3um"
                    ]

        for model in ("Maxwell", "analytic"):
            summary["ratios"][model] = {}
            for projection in PROJECTIONS:
                summary["ratios"][model][projection] = {}
                for method in (
                    "mask_aware_finite_difference",
                    "least_squares_radius_0.2um",
                    "least_squares_radius_0.3um",
                    "least_squares_radius_0.4um",
                ):
                    summary["ratios"][model][projection][method] = {}
                    for region in regions:
                        by_case = {
                            case: summary["cases"][case][projection][
                                "methods"
                            ][method][region]
                            for case in (f"{model}_a", f"{model}_b")
                        }
                        summary["ratios"][model][projection][method][region] = (
                            ratio(
                                by_case,
                                f"{model}_a",
                                f"{model}_b",
                            )
                        )

    figure_paths = {
        model: report_dir
        / f"{model}_thickness_average_fd_vs_ls_shared_scale.png"
        for model in ("Maxwell", "analytic")
    }
    for model, path in figure_paths.items():
        shared_scale_figure(
            path,
            model,
            "thickness average",
            finite_for_figure,
            ls_for_figure,
            x_edges,
            y_edges,
            strict_display,
        )
    summary["figures"] = {
        key: str(value.resolve()) for key, value in figure_paths.items()
    }
    summary["generation_command"] = shlex.join([sys.executable, *sys.argv])

    summary_path = report_dir / "w12_edge_gradient_robustness_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    csv_path = report_dir / "w12_edge_gradient_robustness_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    def table(model: str, projection: str, region: str) -> str:
        rows = []
        for method in (
            "mask_aware_finite_difference",
            "least_squares_radius_0.2um",
            "least_squares_radius_0.3um",
            "least_squares_radius_0.4um",
        ):
            values = summary["ratios"][model][projection][method][region]
            rows.append(
                f"| {method} | {values['raw_max_abs']:.6f} | "
                f"{values['p99_abs']:.6f} | {values['rms_abs']:.6f} | "
                f"{values['mean_abs']:.6f} |"
            )
        return "\n".join(rows)

    report = f"""# W12 explicit-3D edge-gradient robustness audit

Status: `{summary['status']}`

This audit used only the stored explicit-3D temperature artifact.  It did not
run FDTD or a thermal solve.

## Provenance correction

The published `0.879613` is reproduced from the same explicit-3D artifact by
selecting the original 142-cell staircase edge and taking the ratio of the
separate a/b raw maxima.  The separate `0.798934` audit value used a broader
0.5-µm inside-edge band.  They are different spatial comparators, not
different optical checkpoints.

## Maxwell thickness-average, original staircase edge

| reconstruction | raw-max b/a | p99 b/a | RMS b/a | mean b/a |
|---|---:|---:|---:|---:|
{table('Maxwell', 'thickness_average', 'original_staircase_edge')}

## Maxwell thickness-average, fixed 0.1–0.3 µm inside band

| reconstruction | raw-max b/a | p99 b/a | RMS b/a | mean b/a |
|---|---:|---:|---:|---:|
{table('Maxwell', 'thickness_average', 'inside_band_0p1_to_0p3um')}

## Analytic thickness-average, original staircase edge

| reconstruction | raw-max b/a | p99 b/a | RMS b/a | mean b/a |
|---|---:|---:|---:|---:|
{table('analytic', 'thickness_average', 'original_staircase_edge')}

The JSON additionally retains surface and midplane results, individual fixed
normal bands at 0.1, 0.2, and 0.3 µm, least-squares radius sensitivity,
neighbour counts, fit condition numbers, field NRMSE, and correlations.

All new a/b and finite-difference/least-squares plots use one common color
scale.  No raw single-cell maximum is promoted without p99, RMS, and mean.

## Provenance

- input: `{input_path}`
- SHA-256: `{summary['input']['sha256']}`
- JSON: `{summary_path.name}`
- CSV: `{csv_path.name}`
- command: `{summary['generation_command']}`
"""
    report_path = report_dir / "W12_EXPLICIT3D_EDGE_GRADIENT_ROBUSTNESS.md"
    report_path.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "report": str(report_path),
                "Maxwell_thickness_average_original_edge": summary["ratios"][
                    "Maxwell"
                ]["thickness_average"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
