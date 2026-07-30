#!/usr/bin/env python3
"""Mesh-common physical-line gradient audit for the straight 45-degree edge.

The Cartesian FVM still represents the half-plane with a stair-step active
mask.  This audit does not claim to remove that geometry error.  It removes
the additional ambiguity of comparing cell-wise maxima at different cell
centres by fitting every temperature field on the same physical ``(n,t)``
sample set and extrapolating the fitted slope to the exact edge ``n=0``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


STATUS_PASS = "VALIDATED_ROBUST_LOCAL_EDGE_GRADIENT_METRIC"
STATUS_FAIL = "UNRESOLVED_LOCAL_EDGE_GRADIENT_METRIC"
SQRT2 = np.sqrt(2.0)
T_SAMPLES_UM = np.arange(-8.0, 8.0 + 0.1, 0.2)
T_FIT_OFFSETS_UM = np.arange(-0.6, 0.6 + 0.01, 0.1)
N_BANDS_UM = {
    "primary": (-1.5, -0.4, 0.1),
    "narrow": (-1.0, -0.3, 0.1),
    "wide": (-2.0, -0.5, 0.1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_directory(root: Path, polarization: str, mesh_nm: int) -> Path:
    suffix = "_v2" if mesh_nm == 200 else ""
    return root / (
        f"audit_paper_reduced_{polarization}_core{mesh_nm}_"
        f"20260730{suffix}"
    )


def quadratic_edge_fit(
    x_m: np.ndarray,
    y_m: np.ndarray,
    temperature_K: np.ndarray,
    n_band_um: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    interpolator = RegularGridInterpolator(
        (x_m, y_m),
        np.asarray(temperature_K, float),
        method="linear",
        bounds_error=True,
    )
    n_um = np.arange(
        n_band_um[0],
        n_band_um[1] + 0.5 * n_band_um[2],
        n_band_um[2],
    )
    derivative_n = np.empty(T_SAMPLES_UM.size)
    derivative_t = np.empty(T_SAMPLES_UM.size)
    fit_residual = np.empty(T_SAMPLES_UM.size)
    for index, tangent_um in enumerate(T_SAMPLES_UM):
        nn, tt_offset = np.meshgrid(
            n_um,
            T_FIT_OFFSETS_UM,
            indexing="ij",
        )
        tt = tangent_um + tt_offset
        x_um = (tt - nn) / SQRT2
        y_um = (tt + nn) / SQRT2
        values = interpolator(
            np.column_stack((x_um.ravel(), y_um.ravel())) * 1e-6
        )
        design = np.column_stack(
            (
                np.ones(nn.size),
                nn.ravel(),
                tt_offset.ravel(),
                nn.ravel() ** 2,
                (nn * tt_offset).ravel(),
                tt_offset.ravel() ** 2,
            )
        )
        coefficients, _, _, _ = np.linalg.lstsq(
            design,
            values,
            rcond=None,
        )
        prediction = design @ coefficients
        derivative_n[index] = coefficients[1] * 1e6
        derivative_t[index] = coefficients[2] * 1e6
        fit_residual[index] = np.linalg.norm(values - prediction) / max(
            np.linalg.norm(values),
            np.finfo(float).tiny,
        )
    derivative_x = (-derivative_n + derivative_t) / SQRT2
    derivative_y = (derivative_n + derivative_t) / SQRT2
    return {
        "t_m": T_SAMPLES_UM * 1e-6,
        "dT_dn_K_m": derivative_n,
        "dT_dt_K_m": derivative_t,
        "dT_dx_K_m": derivative_x,
        "dT_dy_K_m": derivative_y,
        "fit_relative_residual": fit_residual,
    }


def aggregate(
    coordinate_m: np.ndarray,
    values: np.ndarray,
) -> dict[str, float]:
    absolute = np.abs(np.asarray(values, float))
    return {
        "maximum_abs_K_m": float(np.max(absolute)),
        "p99_abs_K_m": float(np.percentile(absolute, 99.0)),
        "edge_strip_mean_abs_K_m": float(np.mean(absolute)),
        "edge_integrated_abs_K": float(
            np.trapezoid(absolute, coordinate_m)
        ),
    }


def relative_change(value: float, reference: float) -> float:
    return abs(value - reference) / max(
        abs(reference),
        np.finfo(float).tiny,
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    profile_dir = args.output_dir / "profiles"
    profile_dir.mkdir()
    rows: list[dict[str, Any]] = []
    profiles: dict[tuple[str, str, int, str], dict[str, np.ndarray]] = {}
    raw_diagnostic: dict[str, Any] = {}
    input_paths: list[Path] = []
    for polarization in ("a", "b"):
        for mesh_nm in (200, 100, 50):
            directory = run_directory(
                args.artifact_root,
                polarization,
                mesh_nm,
            )
            summary_path = directory / "summary.json"
            profile_path = directory / "straight_edge_profiles.npz"
            input_paths.extend((summary_path, profile_path))
            summary = json.loads(summary_path.read_text())
            with np.load(profile_path, allow_pickle=False) as raw:
                x_m = 0.5 * (
                    np.asarray(raw["x_edges_m"][:-1], float)
                    + np.asarray(raw["x_edges_m"][1:], float)
                )
                y_m = 0.5 * (
                    np.asarray(raw["y_edges_m"][:-1], float)
                    + np.asarray(raw["y_edges_m"][1:], float)
                )
                temperatures = {
                    "analytic_paper_source": np.asarray(
                        raw["analytic_temperature_flake_average_K"],
                        float,
                    ),
                    "legacy_Lumerical_edge_Q": np.asarray(
                        raw[
                            "remapped_Lumerical_temperature_flake_average_K"
                        ],
                        float,
                    ),
                }
            raw_diagnostic[f"{polarization}_{mesh_nm}nm"] = {
                "analytic_paper_source": summary["analytic_offset_cases"][0][
                    "straight_edge_metrics"
                ],
                "legacy_Lumerical_edge_Q": summary[
                    "remapped_Lumerical_thermal_solve"
                ]["straight_edge_metrics"],
            }
            for source, temperature in temperatures.items():
                for band_name, band in N_BANDS_UM.items():
                    fitted = quadratic_edge_fit(
                        x_m,
                        y_m,
                        temperature,
                        band,
                    )
                    profiles[
                        (source, polarization, mesh_nm, band_name)
                    ] = fitted
                    profile_path_out = profile_dir / (
                        f"{source}_{polarization}_{mesh_nm}nm_"
                        f"{band_name}.npz"
                    )
                    np.savez_compressed(profile_path_out, **fitted)
                    for component in ("x", "n"):
                        metrics = aggregate(
                            fitted["t_m"],
                            fitted[f"dT_d{component}_K_m"],
                        )
                        rows.append(
                            {
                                "source": source,
                                "polarization": polarization,
                                "mesh_nm": mesh_nm,
                                "fit_band": band_name,
                                "component": component,
                                **metrics,
                                "fit_residual_p99": float(
                                    np.percentile(
                                        fitted["fit_relative_residual"],
                                        99.0,
                                    )
                                ),
                                "profile_path": str(
                                    profile_path_out.resolve()
                                ),
                                "profile_sha256": sha256(profile_path_out),
                            }
                        )

    by_key = {
        (
            row["source"],
            row["polarization"],
            row["mesh_nm"],
            row["fit_band"],
            row["component"],
        ): row
        for row in rows
    }
    metric_names = (
        "maximum_abs_K_m",
        "p99_abs_K_m",
        "edge_strip_mean_abs_K_m",
        "edge_integrated_abs_K",
    )
    convergence: dict[str, Any] = {}
    for source in ("analytic_paper_source", "legacy_Lumerical_edge_Q"):
        for polarization in ("a", "b"):
            for component in ("x", "n"):
                key = f"{source}_{polarization}_{component}"
                convergence[key] = {}
                for coarse, refined in ((200, 100), (100, 50), (200, 50)):
                    row_coarse = by_key[
                        (
                            source,
                            polarization,
                            coarse,
                            "primary",
                            component,
                        )
                    ]
                    row_refined = by_key[
                        (
                            source,
                            polarization,
                            refined,
                            "primary",
                            component,
                        )
                    ]
                    convergence[key][f"{coarse}_to_{refined}"] = {
                        metric: relative_change(
                            row_coarse[metric],
                            row_refined[metric],
                        )
                        for metric in metric_names
                    }

    band_sensitivity: dict[str, Any] = {}
    for source in ("analytic_paper_source", "legacy_Lumerical_edge_Q"):
        for polarization in ("a", "b"):
            for mesh_nm in (200, 100, 50):
                for component in ("x", "n"):
                    primary = by_key[
                        (
                            source,
                            polarization,
                            mesh_nm,
                            "primary",
                            component,
                        )
                    ]
                    label = (
                        f"{source}_{polarization}_{mesh_nm}nm_{component}"
                    )
                    band_sensitivity[label] = {
                        band: {
                            metric: relative_change(
                                by_key[
                                    (
                                        source,
                                        polarization,
                                        mesh_nm,
                                        band,
                                        component,
                                    )
                                ][metric],
                                primary[metric],
                            )
                            for metric in metric_names
                        }
                        for band in ("narrow", "wide")
                    }

    ratios: dict[str, Any] = {}
    for source in ("analytic_paper_source", "legacy_Lumerical_edge_Q"):
        ratios[source] = {}
        for mesh_nm in (200, 100, 50):
            ratios[source][f"{mesh_nm}nm"] = {}
            for component in ("x", "n"):
                a = by_key[
                    (source, "a", mesh_nm, "primary", component)
                ]
                b = by_key[
                    (source, "b", mesh_nm, "primary", component)
                ]
                ratios[source][f"{mesh_nm}nm"][component] = {
                    metric: b[metric] / a[metric]
                    for metric in metric_names
                }

    production_metrics = (
        "p99_abs_K_m",
        "edge_strip_mean_abs_K_m",
        "edge_integrated_abs_K",
    )
    production_stable_100nm = all(
        convergence[key]["100_to_50"][metric] < 0.01
        for key in convergence
        for metric in production_metrics
    )
    fit_band_sensitivity_lt_10pct = all(
        change[metric] < 0.10
        for comparison in band_sensitivity.values()
        for change in comparison.values()
        for metric in production_metrics
    )
    fit_residual_gate = all(
        row["fit_residual_p99"] < 0.01 for row in rows
    )
    status = (
        STATUS_PASS
        if (
            production_stable_100nm
            and fit_band_sensitivity_lt_10pct
            and fit_residual_gate
        )
        else STATUS_FAIL
    )
    proposed_mesh_nm = 100 if status == STATUS_PASS else None
    summary = {
        "status": status,
        "scope": (
            "same paper-reduced geometry, Robin boundary, and physical-nearest "
            "Q remap at 200/100/50 nm"
        ),
        "exact_coordinate_contract": {
            "n": "(-x+y)/sqrt(2), positive into air",
            "t": "(x+y)/sqrt(2)",
            "exact_edge": "n=0",
            "common_t_samples_um": [
                float(T_SAMPLES_UM[0]),
                float(T_SAMPLES_UM[-1]),
                0.2,
            ],
            "quadratic_terms": ["1", "n", "t", "n^2", "n*t", "t^2"],
            "primary_inside_fit_band_um": list(N_BANDS_UM["primary"]),
            "derivative_location": "quadratic fit evaluated at n=0, t=t_sample",
            "warning": (
                "physical-line fitting stabilizes the observable but does not "
                "remove the underlying Cartesian stair-step geometry error"
            ),
        },
        "raw_cell_maximum_role": "diagnostic only",
        "production_candidate_metrics": list(production_metrics),
        "convergence": convergence,
        "fit_band_sensitivity": band_sensitivity,
        "ratios_b_over_a": ratios,
        "production_100nm_lt_1pct": production_stable_100nm,
        "fit_band_sensitivity_lt_10pct": (
            fit_band_sensitivity_lt_10pct
        ),
        "fit_residual_p99_lt_1pct": fit_residual_gate,
        "proposed_cheapest_mesh_nm": proposed_mesh_nm,
        "50nm_interpretation": (
            "one-time diagnostic only; it has no finer-mesh confirmation and "
            "is not promoted as a production edge-gradient mesh"
        ),
        "refinement_decision": (
            "retain current conservative grid and robust physical-line "
            "comparator"
            if status == STATUS_PASS
            else (
                "do not promote; next compare an exact-half-plane cut-cell "
                "operator before attempting conservative AMR"
            )
        ),
        "boundary_refinement_options": {
            "current_grid_plus_robust_comparator": {
                "cost": "lowest",
                "implementation": "completed here",
                "conservation": "unchanged conservative Cartesian FVM",
                "geometry_error": "stair-step remains explicit",
            },
            "conservative_local_AMR": {
                "cost": "high",
                "implementation": (
                    "new nonuniform topology, prolongation/restriction, and "
                    "coarse/fine flux-matching operator required"
                ),
                "conservation": (
                    "only valid with exact coarse/fine face-flux matching for "
                    "the anisotropic tensor"
                ),
            },
            "cut_cell_embedded_boundary": {
                "cost": "moderate-to-high",
                "implementation": (
                    "exact half-plane cell area/face fractions and small-cell "
                    "conditioning treatment required"
                ),
                "conservation": (
                    "natural finite-volume conservation if every truncated "
                    "face conductance is assembled consistently"
                ),
                "preferred_next_if_needed": True,
            },
        },
        "input_artifacts": [
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in input_paths
        ],
        "no_new_FDTD": True,
        "optimization_run": False,
    }
    summary_path = args.output_dir / "robust_edge_gradient_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    csv_path = args.output_dir / "robust_edge_gradient_cases.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
        constrained_layout=True,
    )
    for column, source in enumerate(
        ("analytic_paper_source", "legacy_Lumerical_edge_Q")
    ):
        for row_index, component in enumerate(("x", "n")):
            ax = axes[row_index, column]
            for polarization, color in (("a", "tab:blue"), ("b", "tab:orange")):
                for mesh_nm, style in ((200, ":"), (100, "--"), (50, "-")):
                    fitted = profiles[
                        (source, polarization, mesh_nm, "primary")
                    ]
                    ax.plot(
                        fitted["t_m"] * 1e6,
                        np.abs(fitted[f"dT_d{component}_K_m"]),
                        style,
                        color=color,
                        label=f"{polarization}, {mesh_nm} nm",
                    )
            ax.set(
                xlabel="exact edge tangent t (µm)",
                ylabel=fr"$|\partial_{{{component}}}T|$ (K/m)",
                title=f"{source}: fitted dT/d{component} at n=0",
            )
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7, ncol=2)
    figure.savefig(
        args.output_dir / "ROBUST_EDGE_GRADIENT_PROFILES.png",
        dpi=180,
    )
    plt.close(figure)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        constrained_layout=True,
    )
    for ax, source in zip(
        axes,
        ("analytic_paper_source", "legacy_Lumerical_edge_Q"),
    ):
        labels = []
        values = []
        for polarization in ("a", "b"):
            for component in ("x", "n"):
                labels.append(f"{polarization}: dT/d{component}")
                values.append(
                    100.0
                    * convergence[
                        f"{source}_{polarization}_{component}"
                    ]["100_to_50"]["edge_strip_mean_abs_K_m"]
                )
        ax.bar(labels, values)
        ax.axhline(1.0, color="black", linestyle="--", label="1%")
        ax.set(
            ylabel="100→50 nm relative change (%)",
            title=f"{source}: robust strip-mean convergence",
        )
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    figure.savefig(
        args.output_dir / "ROBUST_EDGE_GRADIENT_CONVERGENCE.png",
        dpi=180,
    )
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
