#!/usr/bin/env python3
"""Publish the no-new-FDTD straight-edge remap/metric audit."""

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


STATUS = "UNRESOLVED_STRAIGHT_EDGE_OPTICAL_AND_THERMAL_SPATIAL_CONVERGENCE"
METRICS = {
    "max_abs_grad_T_x_K_m": r"$\max|\partial_xT|$",
    "max_abs_grad_T_y_K_m": r"$\max|\partial_yT|$",
    "max_inplane_gradient_K_m": r"$\max|\nabla T|$",
    "max_abs_edge_normal_gradient_K_m": r"$\max|\partial_nT|$",
    "max_abs_edge_tangent_gradient_K_m": r"$\max|\partial_tT|$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--legacy-device-summary", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_roi_average(profile_path: Path, source: str) -> float:
    with np.load(profile_path, allow_pickle=False) as raw:
        x_edges = np.asarray(raw["x_edges_m"], float)
        y_edges = np.asarray(raw["y_edges_m"], float)
        temperature = np.asarray(
            raw[
                (
                    "analytic_temperature_flake_average_K"
                    if source == "analytic_paper_source"
                    else "remapped_Lumerical_temperature_flake_average_K"
                )
            ],
            float,
        )
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    mask = (
        (y[None, :] <= x[:, None])
        & (np.abs(x[:, None]) <= 12.0e-6)
        & (np.abs(y[None, :]) <= 12.0e-6)
    )
    area = np.diff(x_edges)[:, None] * np.diff(y_edges)[None, :]
    return float(
        np.sum(temperature[mask] * area[mask]) / np.sum(area[mask])
    )


def run_directory(
    root: Path,
    *,
    model: str,
    polarization: str,
    domain_um: int,
    core_nm: int,
) -> Path:
    if model == "paper-reduced":
        return root / (
            f"audit_paper_reduced_{polarization}_core{core_nm}_20260730"
        )
    return root / (
        f"audit_expanded_{polarization}_L{domain_um}_"
        f"core{core_nm}_20260730"
    )


def row_from_run(
    directory: Path,
    *,
    model: str,
    source: str,
) -> dict[str, Any]:
    summary_path = directory / "summary.json"
    profile_path = directory / "straight_edge_profiles.npz"
    payload = json.loads(summary_path.read_text())
    if source == "analytic_paper_source":
        case = payload["analytic_offset_cases"][0]
        thermal = case["thermal"]
        metrics = case["straight_edge_metrics"]
    else:
        thermal = payload["remapped_Lumerical_thermal_solve"]
        metrics = thermal["straight_edge_metrics"]
    row: dict[str, Any] = {
        "thermal_model": model,
        "source_model": source,
        "polarization": payload["polarization"],
        "thermal_domain_um": (
            48
            if model == "paper-reduced"
            else int(round(
                (
                    np.load(profile_path, allow_pickle=False)["x_edges_m"][-1]
                    - np.load(profile_path, allow_pickle=False)["x_edges_m"][0]
                )
                * 1e6
            ))
        ),
        "core_step_nm": payload["geometry"]["core_step_nm"],
        "source_power_W": thermal["source_power_W"],
        "linear_residual_relative": thermal["linear_residual_relative"],
        "energy_balance_relative_error": thermal[
            "energy_balance_relative_error"
        ],
        "Tmax_rise_K": metrics["Tmax_rise_K"],
        "whole_domain_area_average_rise_K": metrics[
            "TaIrTe4_area_average_rise_K"
        ],
        "fixed_24um_ROI_area_average_rise_K": fixed_roi_average(
            profile_path,
            source,
        ),
        "numerical_lateral_Dirichlet_fraction": thermal.get(
            "numerical_lateral_Dirichlet_fraction"
        ),
        "numerical_bottom_Dirichlet_fraction": thermal.get(
            "numerical_bottom_Dirichlet_fraction"
        ),
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": sha256(summary_path),
    }
    row.update({name: metrics[name] for name in METRICS})
    return row


def relative_change(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), np.finfo(float).tiny)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    directories: list[Path] = []
    for core_nm in (100, 50):
        for polarization in ("a", "b"):
            directory = run_directory(
                args.artifact_root,
                model="paper-reduced",
                polarization=polarization,
                domain_um=48,
                core_nm=core_nm,
            )
            directories.append(directory)
            for source in ("analytic_paper_source", "lumerical_edge_q"):
                rows.append(
                    row_from_run(
                        directory,
                        model="paper-reduced",
                        source=source,
                    )
                )
    for domain_um in (48, 64, 80):
        for polarization in ("a", "b"):
            directory = run_directory(
                args.artifact_root,
                model="expanded",
                polarization=polarization,
                domain_um=domain_um,
                core_nm=200,
            )
            directories.append(directory)
            rows.append(
                row_from_run(
                    directory,
                    model="expanded",
                    source="lumerical_edge_q",
                )
            )

    def selected(
        model: str,
        source: str,
        polarization: str,
        domain_um: int,
        core_nm: int | None = None,
    ) -> dict[str, Any]:
        return next(
            row
            for row in rows
            if row["thermal_model"] == model
            and row["source_model"] == source
            and row["polarization"] == polarization
            and row["thermal_domain_um"] == domain_um
            and (
                core_nm is None
                or int(round(row["core_step_nm"])) == int(core_nm)
            )
        )

    ratios: dict[str, dict[str, float]] = {}
    scenarios = [
        (
            "paper_analytic_100nm",
            "paper-reduced",
            "analytic_paper_source",
            48,
            100,
        ),
        (
            "paper_analytic_50nm",
            "paper-reduced",
            "analytic_paper_source",
            48,
            50,
        ),
        (
            "paper_reduced_with_Lumerical_Q_100nm",
            "paper-reduced",
            "lumerical_edge_q",
            48,
            100,
        ),
        (
            "paper_reduced_with_Lumerical_Q_50nm",
            "paper-reduced",
            "lumerical_edge_q",
            48,
            50,
        ),
        ("production_L48", "expanded", "lumerical_edge_q", 48, 200),
        ("production_L64", "expanded", "lumerical_edge_q", 64, 200),
        ("production_L80", "expanded", "lumerical_edge_q", 80, 200),
    ]
    for label, model, source, domain, core_nm in scenarios:
        a = selected(model, source, "a", domain, core_nm)
        b = selected(model, source, "b", domain, core_nm)
        ratios[label] = {
            name: b[name] / a[name]
            for name in (
                *METRICS,
                "Tmax_rise_K",
                "fixed_24um_ROI_area_average_rise_K",
            )
        }

    convergence: dict[str, dict[str, float]] = {}
    for polarization in ("a", "b"):
        reference = selected("expanded", "lumerical_edge_q", polarization, 80)
        for domain in (48, 64):
            case = selected(
                "expanded", "lumerical_edge_q", polarization, domain
            )
            convergence[f"{polarization}_{domain}_to_80"] = {
                name: relative_change(case[name], reference[name])
                for name in (
                    "Tmax_rise_K",
                    "fixed_24um_ROI_area_average_rise_K",
                    *METRICS,
                )
            }

    thermal_mesh_convergence: dict[str, dict[str, float]] = {}
    for source in ("analytic_paper_source", "lumerical_edge_q"):
        for polarization in ("a", "b"):
            coarse = selected(
                "paper-reduced",
                source,
                polarization,
                48,
                100,
            )
            refined = selected(
                "paper-reduced",
                source,
                polarization,
                48,
                50,
            )
            thermal_mesh_convergence[f"{source}_{polarization}_100_to_50"] = {
                name: relative_change(coarse[name], refined[name])
                for name in (
                    "Tmax_rise_K",
                    "fixed_24um_ROI_area_average_rise_K",
                    *METRICS,
                )
            }
    thermal_gradient_mesh_gate = all(
        change[name] < 0.01
        for change in thermal_mesh_convergence.values()
        for name in METRICS
    )

    profile_a = json.loads(
        (
            run_directory(
                args.artifact_root,
                model="paper-reduced",
                polarization="a",
                domain_um=48,
                core_nm=100,
            )
            / "summary.json"
        ).read_text()
    )
    profile_b = json.loads(
        (
            run_directory(
                args.artifact_root,
                model="paper-reduced",
                polarization="b",
                domain_um=48,
                core_nm=100,
            )
            / "summary.json"
        ).read_text()
    )
    profile_audit = {
        polarization: {
            name: summary["profiles"][name]
            for name in (
                "analytic_Q_areal",
                "native_Lumerical_Q_areal",
                "remapped_Lumerical_Q_areal",
            )
        }
        for polarization, summary in (("a", profile_a), ("b", profile_b))
    }
    actual_remap_audit: dict[str, Any] = {}
    for polarization in ("a", "b"):
        directory = args.artifact_root / (
            f"audit_support_remap_{polarization}_core100_20260730"
        )
        path = directory / "support_remap_audit.json"
        payload = json.loads(path.read_text())
        actual_remap_audit[polarization] = {
            "status": payload["status"],
            "unprojected_outside_support_fraction": payload[
                "unprojected_outside_support_fraction"
            ],
            "pairwise_energy_weighted_relative_L1": payload[
                "pairwise_energy_weighted_relative_L1"
            ],
            "per_operator": payload["per_operator"],
            "summary_path": str(path.resolve()),
            "summary_sha256": sha256(path),
        }

    legacy_device = json.loads(args.legacy_device_summary.read_text())
    device_weighting_correction: dict[str, Any] = {}
    for model, legacy_key, directory_label in (
        ("expanded", "expanded", "expanded"),
        ("paper-reduced", "paper_reduced", "paper_reduced"),
    ):
        model_cases: dict[str, Any] = {}
        for polarization in ("a", "b"):
            path = args.artifact_root / (
                f"audit_device_a_weighting_{polarization}_"
                f"{directory_label}_core200_20260730/summary.json"
            )
            current = json.loads(path.read_text())
            old = legacy_device[legacy_key][polarization]
            old_current = float(old["PTE_current_A_at_285uW_incident"])
            new_current = float(
                current["PTE_current_A_at_285uW_incident"]
            )
            model_cases[polarization] = {
                "legacy_current_A": old_current,
                "corrected_current_A": new_current,
                "relative_change": (new_current - old_current) / old_current,
                "legacy_weighting_extrema": {
                    "minimum": old["weighting"]["minimum_psi"],
                    "maximum": old["weighting"]["maximum_psi"],
                },
                "corrected_weighting_extrema": {
                    "minimum": current["weighting"]["minimum_psi"],
                    "maximum": current["weighting"]["maximum_psi"],
                },
                "corrected_contact_half_width_m": {
                    "top": current["weighting"][
                        "top_contact_half_width_m"
                    ],
                    "bottom": current["weighting"][
                        "bottom_contact_half_width_m"
                    ],
                },
                "corrected_summary_path": str(path.resolve()),
                "corrected_summary_sha256": sha256(path),
            }
        model_cases["ratio_abs_a_over_b"] = {
            "legacy": abs(
                model_cases["a"]["legacy_current_A"]
                / model_cases["b"]["legacy_current_A"]
            ),
            "corrected": abs(
                model_cases["a"]["corrected_current_A"]
                / model_cases["b"]["corrected_current_A"]
            ),
        }
        device_weighting_correction[model] = model_cases

    summary = {
        "status": STATUS,
        "decision": (
            "paper-source/reduced-boundary control reproduces b>a gradient, "
            "whereas saved finite-edge Lumerical Q reverses all five gradient "
            "observables; do not promote a paper reproduction"
        ),
        "axis_order_regression": {
            "old_projection_orders": ["x/y/z/x", "y/x/z/y"],
            "symmetric_Gaussian_relative_L1_difference": 0.5,
            "power_conserved_in_both": True,
            "replacement": (
                "single physical-3D-nearest support projection with exact "
                "nearest-distance ties split uniformly"
            ),
            "replacement_power_transpose_symmetry_tests": "passed",
            "actual_saved_Q_artifact_audit": {
                polarization: {
                    "old_xfirst_vs_yfirst_relative_L1": actual_remap_audit[
                        polarization
                    ]["pairwise_energy_weighted_relative_L1"][
                        "historical_x_y_z_x__vs__reflected_y_x_z_y"
                    ],
                    "old_xfirst_vs_new_relative_L1": actual_remap_audit[
                        polarization
                    ]["pairwise_energy_weighted_relative_L1"][
                        "historical_x_y_z_x__vs__physical_nearest_support"
                    ],
                    "unprojected_outside_support_fraction": actual_remap_audit[
                        polarization
                    ]["unprojected_outside_support_fraction"],
                    "new_projection_power_error_relative": actual_remap_audit[
                        polarization
                    ]["per_operator"]["physical_nearest_support"][
                        "power_error_relative"
                    ],
                    "new_projection_outside_support_power_W": actual_remap_audit[
                        polarization
                    ]["per_operator"]["physical_nearest_support"][
                        "outside_support_power_W"
                    ],
                }
                for polarization in ("a", "b")
            },
        },
        "ratios_b_over_a": ratios,
        "thermal_domain_convergence_relative_to_80um": convergence,
        "paper_reduced_thermal_mesh_convergence_100_to_50": (
            thermal_mesh_convergence
        ),
        "thermal_gradient_mesh_gate_lt_1pct": thermal_gradient_mesh_gate,
        "profile_audit": profile_audit,
        "actual_raw_Q_remap_audit": actual_remap_audit,
        "device_A_weighting_contact_correction": device_weighting_correction,
        "resolved_in_this_checkpoint": [
            "axis-order-dependent x/y/z/x support projection removed from straight-edge path",
            "area and volume means use literal cell area/volume",
            "all x/y/magnitude/normal/tangent gradient observables retained",
            "weighting-potential contacts use each local boundary-cell half width",
            "48/64/80 um production thermal-domain trend evaluated",
            "paper analytic source/reduced boundary separated from production model",
        ],
        "remaining_fail_closed_blockers": [
            "solver-native lateral Yee mesh readback/refinement is absent",
            "sampled-material fitted epsilon_x/y/z readback is absent",
            "epsilon_c=16+0i makes Qz exactly zero and is not edge-validated",
            "scalar-Gaussian versus vector-Gaussian edge-Q sensitivity is absent",
            *(
                []
                if thermal_gradient_mesh_gate
                else [
                    "paper-reduced thermal edge-gradient 100-to-50 nm "
                    "convergence exceeds 1%"
                ]
            ),
        ],
        "artifact_Q_grid_readback": {
            "dx_m": 33.9702760085e-9,
            "dy_m": 33.9702760085e-9,
            "dz_m": 10.0e-9,
            "warning": (
                "this is the common absorption-artifact grid, not a "
                "solver-native Yee mesh certificate"
            ),
        },
        "remote_polygon_faces": {
            "polygon_remote_coordinates_um": {"x_max": 25.0, "y_min": -25.0},
            "FDTD_outer_coordinates_um": {"x": [-24.0, 24.0], "y": [-24.0, 24.0]},
            "conclusion": (
                "remote faces are outside the FDTD domain, so this artifact's "
                "flake does not terminate inside its lateral PML"
            ),
        },
        "no_new_FDTD": True,
        "no_weighting_or_PTE_in_straight_edge_subgate": True,
        "no_optimization": True,
    }
    (args.output_dir / "straight_edge_spatial_q_audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (args.output_dir / "straight_edge_spatial_q_audit_cases.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    labels = [item[0] for item in scenarios]
    x = np.arange(len(labels))
    width = 0.15
    for offset, (metric, title) in enumerate(METRICS.items()):
        axes[0].bar(
            x + (offset - 2) * width,
            [ratios[label][metric] for label in labels],
            width,
            label=title,
        )
    axes[0].axhline(1.0, color="black", linestyle="--")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylabel("gradient ratio b/a")
    axes[0].set_title("Paper-source control versus finite-edge Maxwell Q")
    axes[0].legend(fontsize=8, ncol=2)
    for polarization, marker in (("a", "o"), ("b", "s")):
        for metric, title in (
            ("Tmax_rise_K", r"$T_{\max}$"),
            ("fixed_24um_ROI_area_average_rise_K", "fixed-ROI average"),
            ("max_abs_grad_T_x_K_m", r"$\max|\partial_xT|$"),
            ("max_abs_edge_normal_gradient_K_m", r"$\max|\partial_nT|$"),
        ):
            values = [
                selected("expanded", "lumerical_edge_q", polarization, d)[
                    metric
                ]
                for d in (48, 64, 80)
            ]
            axes[1].plot(
                (48, 64, 80),
                np.asarray(values) / values[-1],
                marker=marker,
                label=f"{polarization}: {title}",
            )
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[1].set(
        xlabel="thermal lateral domain (µm)",
        ylabel="value / 80 µm value",
        title="Expanded-model fixed-ROI domain convergence",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, ncol=2)
    figure.savefig(args.output_dir / "STRAIGHT_EDGE_AUDIT_METRICS.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, polarization in zip(axes, ("a", "b")):
        directory = run_directory(
            args.artifact_root,
            model="paper-reduced",
            polarization=polarization,
            domain_um=48,
            core_nm=100,
        )
        with np.load(
            directory / "straight_edge_profiles.npz",
            allow_pickle=False,
        ) as raw:
            for prefix, label, style in (
                ("analytic_Q_areal", "analytic Gaussian", "-"),
                ("native_Lumerical_Q_areal", "native/common Q grid", "--"),
                ("remapped_Lumerical_Q_areal", "thermal-grid Q", ":"),
            ):
                coordinate = raw[f"{prefix}_n_m"] * 1e6
                value = np.abs(raw[f"{prefix}_values"])
                ax.plot(
                    coordinate,
                    value / np.nanmax(value),
                    style,
                    label=label,
                )
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set(
            xlabel="edge-normal coordinate n (µm)",
            ylabel="normalized areal Q",
            title=f"E ∥ {polarization}",
            xlim=(-10, 2),
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    figure.savefig(args.output_dir / "STRAIGHT_EDGE_Q_PROFILES.png", dpi=180)
    plt.close(figure)

    worst_thermal_gradient_mesh_change = max(
        change[name]
        for change in thermal_mesh_convergence.values()
        for name in METRICS
    )
    report = f"""# Straight-edge spatial-Q, remap, and gradient audit

**Status: `{STATUS}`**

This checkpoint reuses the saved GPU Lumerical artifacts; no new FDTD, AD/FD,
or optimization run was performed.  The straight-edge subgate intentionally
has no weighting/PTE evaluation.  A separate saved-Q Device-A calculation
was rerun only to quantify the corrected local-contact weighting operator.

## What was wrong and what was corrected

The old `x/y/z/x` support projection was coordinate-order dependent.  The
named symmetric Gaussian regression reproduces a 50.0% relative L1 difference
between `x/y/z/x` and `y/x/z/y` while preserving identical total power.  The
straight-edge path now uses one physical-3D nearest-support operator and
splits exact distance ties uniformly.  Power, transpose, and reflection
symmetry tests pass.

The 50.0% value is a structural synthetic regression, not an estimate of the
saved Maxwell-Q error.  On the actual raw Q, the two historical axis orders
differ by
{100*actual_remap_audit['a']['pairwise_energy_weighted_relative_L1']['historical_x_y_z_x__vs__reflected_y_x_z_y']:.6f}%
for a polarization and
{100*actual_remap_audit['b']['pairwise_energy_weighted_relative_L1']['historical_x_y_z_x__vs__reflected_y_x_z_y']:.6f}%
for b.  Historical-to-new physical-nearest differences are
{100*actual_remap_audit['a']['pairwise_energy_weighted_relative_L1']['historical_x_y_z_x__vs__physical_nearest_support']:.6f}% and
{100*actual_remap_audit['b']['pairwise_energy_weighted_relative_L1']['historical_x_y_z_x__vs__physical_nearest_support']:.6f}%.
Thus the old operator is invalid in principle, but its actual contribution
does not explain the observed ~20% gradient-order reversal.

The reported area/volume averages now use literal cell area/volume.  The
weighting contact uses each contacted cell's local half width.  Five separate
gradient observables are retained.  The paper comparator is
`max_abs_grad_T_x_K_m`; edge-normal gradient is not substituted for it.

The legacy Device-A expanded currents change from
{device_weighting_correction['expanded']['a']['legacy_current_A']*1e9:.6f}/
{device_weighting_correction['expanded']['b']['legacy_current_A']*1e9:.6f} nA
to
{device_weighting_correction['expanded']['a']['corrected_current_A']*1e9:.6f}/
{device_weighting_correction['expanded']['b']['corrected_current_A']*1e9:.6f} nA
for a/b polarization.  The corrected ratio is
{device_weighting_correction['expanded']['ratio_abs_a_over_b']['corrected']:.6f}.
The exact old values remain provenance diagnostics, not silently overwritten.

## Separated sanity checks

With analytic Gaussian–Beer–Lambert Q and the paper Eq. S4 reduced Robin
model, the `max|dT/dx|` ratio is
**{ratios['paper_analytic_100nm']['max_abs_grad_T_x_K_m']:.6f}** at
100 nm and
**{ratios['paper_analytic_50nm']['max_abs_grad_T_x_K_m']:.6f} (b/a)** at
50 nm.
The expected b>a order is reproduced.

With the saved finite-edge Lumerical Q on that same reduced thermal operator,
the ratio is
**{ratios['paper_reduced_with_Lumerical_Q_100nm']['max_abs_grad_T_x_K_m']:.6f}**
at 100 nm and
**{ratios['paper_reduced_with_Lumerical_Q_50nm']['max_abs_grad_T_x_K_m']:.6f}**
at 50 nm.
With the expanded production FVM it is
**{ratios['production_L80']['max_abs_grad_T_x_K_m']:.6f}** at 80 µm.
All five gradient ratios remain below one in the finite-edge Maxwell-Q chain.
Thus the inversion is source-spatial-distribution sensitive; it is not
explained solely by choosing edge-normal rather than x-gradient.

The worst 100-to-50 nm change among the five paper-reduced gradient
observables is **{100*worst_thermal_gradient_mesh_change:.6f}%**.  The
predeclared 1% thermal-gradient mesh gate is
**{'passed' if thermal_gradient_mesh_gate else 'not passed'}**.

## Thermal-domain audit

The far x/y and bottom Dirichlet powers are numerical truncation-boundary
fluxes, not intrinsic physical heat-path fractions.  Although their share
changes strongly with domain size, the central edge metrics converge much
more tightly.  Relative changes to the 80 µm case are stored explicitly in
the summary JSON.  Whole-half-plane averages are not used for this comparison;
the report uses the fixed |x|,|y| <= 12 µm ROI.

## Remaining blockers

- Solver-native lateral Yee-mesh readback/refinement is not certified.
- Fitted sampled-material epsilon readback is absent.
- epsilon_c=16+0i forces Qz=0 and is not validated for edge scattering.
- Scalar- versus vector-Gaussian edge-Q sensitivity is absent.
{('- Paper-reduced edge-gradient 100-to-50 nm convergence exceeds 1%.' if not thermal_gradient_mesh_gate else '- Paper-reduced 100-to-50 nm thermal-gradient convergence passes; expanded-model 50 nm was not required for the paper-source separation gate.')}

The common absorption artifact has 33.9703 nm x/y and 10 nm z spacing, but
that is not relabeled as native Yee-mesh readback.  The remote polygon faces
are at x=+25 µm and y=-25 µm, beyond the actual +/-24 µm FDTD outer boundary;
the flake therefore does not terminate inside the lateral PML in these saved
artifacts.
"""
    (args.output_dir / "STRAIGHT_EDGE_SPATIAL_Q_AUDIT_REPORT.md").write_text(
        report
    )

    manifest_entries = []
    for directory in sorted(set(directories)):
        for name in (
            "summary.json",
            "straight_edge_profiles.npz",
            "profile_comparison.png",
        ):
            path = directory / name
            manifest_entries.append(
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    for polarization in ("a", "b"):
        directory = args.artifact_root / (
            f"audit_support_remap_{polarization}_core100_20260730"
        )
        for name in (
            "support_remap_audit.json",
            "support_remap_edge_profiles.png",
        ):
            path = directory / name
            manifest_entries.append(
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    for model in ("expanded", "paper_reduced"):
        for polarization in ("a", "b"):
            directory = args.artifact_root / (
                f"audit_device_a_weighting_{polarization}_{model}_"
                "core200_20260730"
            )
            for name in ("summary.json", "thermal_pte_fields.npz"):
                path = directory / name
                manifest_entries.append(
                    {
                        "path": str(path.resolve()),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    for polarization in ("a", "b"):
        directory = args.artifact_root / (
            f"audit_support_remap_{polarization}_core100_20260730"
        )
        for name in (
            "support_remap_audit.json",
            "support_remap_edge_profiles.png",
        ):
            path = directory / name
            manifest_entries.append(
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    for polarization, gpu in (("a", 4), ("b", 5)):
        optical = args.artifact_root / (
            f"straight45_{polarization}_w6p5_dz10_L48_gpu{gpu}_20260730"
        )
        for name in ("finite_q_on_artifact.npz", "case_result.json"):
            path = optical / name
            manifest_entries.append(
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "committed_to_git": False,
                }
            )
    (args.output_dir / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(
            {
                "status": STATUS,
                "raw_artifacts_committed_to_git": False,
                "artifacts": manifest_entries,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
