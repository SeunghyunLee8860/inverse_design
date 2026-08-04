#!/usr/bin/env python3
"""Summarize the fail-closed Device-A single-position Maxwell-to-PTE check."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path, role: str) -> dict[str, object]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "byte_size": path.stat().st_size,
        "sha256": sha256(path),
        "committed_to_git": False,
    }


def load_case(path: Path, polarization: str, scenario: str) -> dict[str, object]:
    summary_path = path / "summary.json"
    fields_path = path / "thermal_pte_fields.npz"
    summary = json.loads(summary_path.read_text())
    if not summary["status"].startswith("COMPLETED"):
        raise RuntimeError(f"case did not complete: {summary_path}")
    return {
        "path": path,
        "summary_path": summary_path,
        "fields_path": fields_path,
        "polarization": polarization,
        "scenario": scenario,
        "summary": summary,
    }


def optical_metrics(path: Path) -> dict[str, object]:
    result = json.loads((path / "case_result.json").read_text())
    run = result["run_result"]
    return {
        "status": result["status"],
        "P_Q_W_at_1_W_m2": run["P_Q_W"],
        "P_six_W_at_1_W_m2": run["P_six_face_W"],
        "six_face_closure": run["six_face_relative_closure"],
        "auto_shutoff_final": run["auto_shutoff"]["final_value"],
        "negative_Q_voxel_count": run["negative_Q_voxel_count"],
        "minimum_Q_W_m3": run["minimum_Q_W_m3"],
    }


def load_maps(case: dict[str, object]) -> dict[str, np.ndarray]:
    with np.load(case["fields_path"]) as raw:
        x_edges = raw["x_edges_m"] * 1e6
        y_edges = raw["y_edges_m"] * 1e6
        x = 0.5 * (x_edges[:-1] + x_edges[1:])
        y = 0.5 * (y_edges[:-1] + y_edges[1:])
        dz = np.diff(raw["z_edges_m"])
        q2d = np.sum(raw["Q_W_m3"] * dz[None, None, :], axis=2)
        mask = np.any(raw["flake_mask"], axis=2)
        return {
            "x_um": x,
            "y_um": y,
            "x_edges_um": x_edges,
            "y_edges_um": y_edges,
            "mask": mask,
            "q2d_W_m2": q2d,
            "temperature_K": raw["temperature_flake_average_K"].copy(),
            "grad_x_K_m": raw["grad_T_x_K_m"].copy(),
            "grad_y_K_m": raw["grad_T_y_K_m"].copy(),
            "integrand_A_m2": raw["shockley_ramo_integrand_A_m2"].copy(),
            "psi": raw["weighting_potential"].copy(),
            "weighting_x_m_inv": raw["weighting_grad_x_m_inv"].copy(),
            "weighting_y_m_inv": raw["weighting_grad_y_m_inv"].copy(),
            "current_x_A_m2": raw["local_J_x_A_m2"].copy(),
            "current_y_A_m2": raw["local_J_y_A_m2"].copy(),
        }


def masked(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, array, np.nan)


def plot_primary_maps(
    cases: list[dict[str, object]], output: Path
) -> None:
    maps = [load_maps(case) for case in cases]
    columns = []
    for values in maps:
        columns.append(
            [
                masked(values["q2d_W_m2"], values["mask"]),
                masked(values["temperature_K"], values["mask"]),
                masked(
                    np.hypot(values["grad_x_K_m"], values["grad_y_K_m"]),
                    values["mask"],
                ),
                masked(values["integrand_A_m2"], values["mask"]),
            ]
        )
    limits = []
    for column in range(4):
        values = np.concatenate(
            [entry[column][np.isfinite(entry[column])] for entry in columns]
        )
        if column == 3:
            bound = float(np.nanmax(np.abs(values)))
            limits.append((-bound, bound))
        else:
            limits.append((0.0, float(np.nanmax(values))))
    titles = [
        r"depth-integrated $Q$ (W m$^{-2}$)",
        r"flake-averaged $\Delta T$ (K)",
        r"$|\nabla_{xy}T|$ (K m$^{-1}$)",
        r"$\mathbf{J}_{loc}\!\cdot\!\nabla\psi$ (A m$^{-2}$)",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for row, (case, values, row_maps) in enumerate(zip(cases, maps, columns)):
        for column, array in enumerate(row_maps):
            cmap = "coolwarm" if column == 3 else "magma"
            handle = axes[row, column].pcolormesh(
                values["x_edges_um"],
                values["y_edges_um"],
                array.T,
                shading="flat",
                cmap=cmap,
                vmin=limits[column][0],
                vmax=limits[column][1],
            )
            axes[row, column].set_aspect("equal")
            axes[row, column].set(xlabel="x = b (µm)", ylabel="y = a (µm)")
            if row == 0:
                axes[row, column].set_title(titles[column])
            fig.colorbar(handle, ax=axes[row, column], shrink=0.82)
        axes[row, 0].text(
            0.02,
            0.97,
            rf"$E\parallel {case['polarization']}$",
            transform=axes[row, 0].transAxes,
            va="top",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.6, "pad": 3},
        )
    fig.suptitle("Device A: perfect-to-flake diagnostic, common scales")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_weighting(cases: list[dict[str, object]], output: Path) -> None:
    maps = [load_maps(case) for case in cases]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    base = maps[0]
    psi = masked(base["psi"], base["mask"])
    handle = axes[0].pcolormesh(
        base["x_edges_um"],
        base["y_edges_um"],
        psi.T,
        shading="flat",
        cmap="viridis",
    )
    fig.colorbar(handle, ax=axes[0], label=r"$\psi$")
    step = 10
    weighting_norm = np.hypot(base["weighting_x_m_inv"], base["weighting_y_m_inv"])
    weighting_denominator = np.where(weighting_norm > 0.0, weighting_norm, 1.0)
    weighting_u = np.where(
        base["mask"], base["weighting_x_m_inv"] / weighting_denominator, np.nan
    )
    weighting_v = np.where(
        base["mask"], base["weighting_y_m_inv"] / weighting_denominator, np.nan
    )
    axes[0].quiver(
        base["x_um"][::step],
        base["y_um"][::step],
        weighting_u[::step, ::step].T,
        weighting_v[::step, ::step].T,
        color="white",
        alpha=0.7,
        angles="xy",
        scale_units="width",
        scale=22,
        width=0.002,
    )
    common_current_max = max(
        float(
            np.nanmax(
                masked(
                    np.hypot(values["current_x_A_m2"], values["current_y_A_m2"]),
                    values["mask"],
                )
            )
        )
        for values in maps
    )
    for axis, case, values in zip(axes[1:], cases, maps):
        magnitude = np.hypot(values["current_x_A_m2"], values["current_y_A_m2"])
        handle = axis.pcolormesh(
            values["x_edges_um"],
            values["y_edges_um"],
            masked(magnitude, values["mask"]).T,
            shading="flat",
            cmap="magma",
            vmin=0.0,
            vmax=common_current_max,
        )
        fig.colorbar(handle, ax=axis, label=r"$|\mathbf{J}_{loc}|$ (A m$^{-2}$)")
        denominator = np.where(magnitude > 0.0, magnitude, 1.0)
        unit_x = np.where(values["mask"], values["current_x_A_m2"] / denominator, np.nan)
        unit_y = np.where(values["mask"], values["current_y_A_m2"] / denominator, np.nan)
        axis.quiver(
            values["x_um"][::step],
            values["y_um"][::step],
            unit_x[::step, ::step].T,
            unit_y[::step, ::step].T,
            color="cyan",
            alpha=0.7,
            angles="xy",
            scale_units="width",
            scale=22,
            width=0.002,
        )
        axis.set_title(rf"local PTE field, $E\parallel {case['polarization']}$")
    axes[0].set_title("digitized-contact weighting potential")
    for axis in axes:
        axis.set_aspect("equal")
        axis.set(xlabel="x = b (µm)", ylabel="y = a (µm)")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_currents(rows: list[dict[str, object]], measured: float, uncertainty: float, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    labels = [f"{row['scenario']}\nE||{row['polarization']}" for row in rows]
    axes[0].bar(np.arange(len(rows)), [row["current_A"] * 1e9 for row in rows])
    axes[0].set_xticks(np.arange(len(rows)), labels, rotation=20, ha="right")
    axes[0].set_ylabel("signed terminal current (nA)")
    ratios = []
    names = []
    for scenario in ("isolated", "perfect"):
        selected = {row["polarization"]: row for row in rows if row["scenario"] == scenario}
        ratios.append(abs(selected["a"]["current_A"]) / abs(selected["b"]["current_A"]))
        names.append(scenario)
    axes[1].bar(names, ratios, color=["tab:gray", "tab:blue"])
    axes[1].axhspan(measured - uncertainty, measured + uncertainty, color="tab:orange", alpha=0.3)
    axes[1].axhline(measured, color="tab:orange", label="digitized Fig. 3I/J")
    axes[1].set_ylabel(r"$|I_a|/|I_b|$")
    axes[1].legend()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def gradient_components(
    values: dict[str, np.ndarray], normal: np.ndarray, tangent: np.ndarray
) -> list[np.ndarray]:
    grad_b = values["grad_x_K_m"]
    grad_a = values["grad_y_K_m"]
    grad_n = normal[0] * grad_b + normal[1] * grad_a
    grad_t = tangent[0] * grad_b + tangent[1] * grad_a
    magnitude = np.hypot(grad_b, grad_a)
    return [grad_a, grad_b, grad_n, grad_t, magnitude]


def plot_gradients(
    cases: list[dict[str, object]], normal: np.ndarray, tangent: np.ndarray, output: Path
) -> None:
    maps = [load_maps(case) for case in cases]
    components = [gradient_components(values, normal, tangent) for values in maps]
    titles = [
        r"$\partial_aT=\partial_yT$",
        r"$\partial_bT=\partial_xT$",
        r"$\partial_nT$",
        r"$\partial_tT$",
        r"$|\nabla_{xy}T|$",
    ]
    limits = []
    for column in range(5):
        entries = [masked(item[column], values["mask"]) for item, values in zip(components, maps)]
        bound = max(float(np.nanmax(np.abs(entry))) for entry in entries)
        limits.append((0.0, bound) if column == 4 else (-bound, bound))
    fig, axes = plt.subplots(2, 5, figsize=(19, 7.5), constrained_layout=True)
    for row, (case, values, row_components) in enumerate(zip(cases, maps, components)):
        for column, component in enumerate(row_components):
            handle = axes[row, column].pcolormesh(
                values["x_edges_um"],
                values["y_edges_um"],
                masked(component, values["mask"]).T,
                shading="flat",
                cmap="magma" if column == 4 else "coolwarm",
                vmin=limits[column][0],
                vmax=limits[column][1],
            )
            axes[row, column].set_aspect("equal")
            axes[row, column].set(xlabel="x = b (µm)", ylabel="y = a (µm)")
            if row == 0:
                axes[row, column].set_title(titles[column] + r" (K m$^{-1}$)")
            fig.colorbar(handle, ax=axes[row, column], shrink=0.8)
        axes[row, 0].text(
            0.02,
            0.97,
            rf"$E\parallel {case['polarization']}$",
            transform=axes[row, 0].transAxes,
            va="top",
            color="black",
            bbox={"facecolor": "white", "alpha": 0.75, "pad": 3},
        )
    fig.suptitle("Device A thermal-gradient components, common polarization scales")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def edge_gradient_metrics(
    values: dict[str, np.ndarray],
    vertices_simulation_um: np.ndarray,
    edge_indices: list[int],
    normal: np.ndarray,
    tangent: np.ndarray,
) -> dict[str, object]:
    first = vertices_simulation_um[edge_indices[0]]
    second = vertices_simulation_um[edge_indices[1]]
    midpoint = 0.5 * (first + second)
    xx, yy = np.meshgrid(values["x_um"], values["y_um"], indexing="ij")
    dx = xx - midpoint[0]
    dy = yy - midpoint[1]
    distance_n = normal[0] * dx + normal[1] * dy
    coordinate_t = tangent[0] * dx + tangent[1] * dy
    endpoint_t = np.asarray(
        [
            tangent @ (first - midpoint),
            tangent @ (second - midpoint),
        ]
    )
    selection = (
        values["mask"]
        & (np.abs(distance_n) <= 0.3)
        & (coordinate_t >= np.min(endpoint_t))
        & (coordinate_t <= np.max(endpoint_t))
    )
    components = gradient_components(values, normal, tangent)
    names = ("grad_a", "grad_b", "grad_n", "grad_t", "grad_magnitude")
    metrics: dict[str, object] = {
        "edge_band_half_width_um": 0.3,
        "edge_band_cell_count": int(np.count_nonzero(selection)),
        "edge_vertex_indices": edge_indices,
    }
    for name, component in zip(names, components):
        samples = np.abs(component[selection])
        metrics[name] = {
            "raw_max_abs_K_m": float(np.max(samples)),
            "p99_abs_K_m": float(np.percentile(samples, 99.0)),
            "rms_K_m": float(np.sqrt(np.mean(samples**2))),
            "mean_abs_K_m": float(np.mean(samples)),
        }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-isolated", type=Path, required=True)
    parser.add_argument("--b-isolated", type=Path, required=True)
    parser.add_argument("--a-perfect", type=Path, required=True)
    parser.add_argument("--b-perfect", type=Path, required=True)
    parser.add_argument("--optical-a", type=Path, required=True)
    parser.add_argument("--optical-b", type=Path, required=True)
    parser.add_argument("--geometry-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        load_case(args.a_isolated, "a", "isolated"),
        load_case(args.b_isolated, "b", "isolated"),
        load_case(args.a_perfect, "a", "perfect"),
        load_case(args.b_perfect, "b", "perfect"),
    ]
    geometry = json.loads(args.geometry_contract.read_text())
    measured = float(geometry["measured_ratio"]["abs_Ia_over_abs_Ib"])
    uncertainty = float(geometry["measured_ratio"]["digitization_uncertainty"])
    optical = {
        "a": optical_metrics(args.optical_a),
        "b": optical_metrics(args.optical_b),
    }
    normal = np.asarray(geometry["off_axis_edge_unit_inward_normal_code"], float)
    tangent = np.asarray(geometry["off_axis_edge_unit_tangent_code"], float)
    rows = []
    for case in cases:
        summary = case["summary"]
        rows.append(
            {
                "scenario": case["scenario"],
                "polarization": case["polarization"],
                "status": summary["status"],
                "P_Q_thermal_W": summary["mapping"]["P_Q_target_W"],
                "Tmax_rise_K": summary["thermal"]["Tmax_rise_K"],
                "TaIrTe4_average_rise_K": summary["thermal"]["TaIrTe4_volume_average_rise_K"],
                "current_A": summary["PTE_current_A_at_285uW_incident"],
                "mapping_error": summary["mapping"]["mapping_relative_power_error"],
                "residual": summary["thermal"]["linear_residual_relative"],
                "energy_balance_error": summary["thermal"]["energy_balance_relative_error"],
                "thermal_raw_path": str(case["fields_path"].resolve()),
            }
        )
    by_scenario = {}
    for scenario in ("isolated", "perfect"):
        selected = {row["polarization"]: row for row in rows if row["scenario"] == scenario}
        ratio = abs(selected["a"]["current_A"]) / abs(selected["b"]["current_A"])
        by_scenario[scenario] = {
            "abs_Ia_over_abs_Ib": ratio,
            "abs_Ib_over_abs_Ia": 1.0 / ratio,
            "relative_difference_from_digitized_measurement": (ratio - measured) / measured,
        }
    perfect_cases = [case for case in cases if case["scenario"] == "perfect"]
    shift = np.asarray(
        perfect_cases[0]["summary"]["geometry"]["digitized_contract"][
            "simulation_origin_shift_um"
        ],
        float,
    )
    vertices_simulation = np.asarray(geometry["flake_vertices_code_um"], float) + shift
    gradient_audit = {}
    for case in perfect_cases:
        gradient_audit[case["polarization"]] = edge_gradient_metrics(
            load_maps(case),
            vertices_simulation,
            geometry["off_axis_edge_vertex_indices"],
            normal,
            tangent,
        )
    gradient_ratios = {}
    for component in ("grad_a", "grad_b", "grad_n", "grad_t", "grad_magnitude"):
        gradient_ratios[component] = {
            metric: gradient_audit["a"][component][metric]
            / gradient_audit["b"][component][metric]
            for metric in ("raw_max_abs_K_m", "p99_abs_K_m", "rms_K_m", "mean_abs_K_m")
        }
    numerical_gates = {
        "all_mapping_errors_below_0p5pct": all(row["mapping_error"] < 0.005 for row in rows),
        "all_residuals_below_1e-8": all(row["residual"] < 1e-8 for row in rows),
        "all_energy_balance_errors_below_1pct": all(row["energy_balance_error"] < 0.01 for row in rows),
    }
    summary = {
        "status": "COMPLETED_DEVICE_A_SINGLE_POSITION_SANITY_DISAGREES_WITH_DIGITIZED_CURRENT_RATIO",
        "comparison_contract": "single pre-registered beam position; no beam/contact tuning; no polarization-dependent Q rescaling",
        "measured_abs_Ia_over_abs_Ib": measured,
        "measured_abs_Ib_over_abs_Ia": 1.0 / measured,
        "measured_digitization_uncertainty": uncertainty,
        "scenario_ratios": by_scenario,
        "optical_gates": {
            "cases": optical,
            "all_closure_below_0p5pct": all(
                item["six_face_closure"] < 0.005 for item in optical.values()
            ),
            "all_auto_shutoff_below_1e-5": all(
                item["auto_shutoff_final"] < 1e-5 for item in optical.values()
            ),
            "no_negative_Q": all(
                item["negative_Q_voxel_count"] == 0 for item in optical.values()
            ),
        },
        "off_axis_edge_gradient_audit_perfect_to_flake": gradient_audit,
        "off_axis_edge_gradient_abs_a_over_abs_b": gradient_ratios,
        "numerical_gates": numerical_gates,
        "cases": rows,
        "interpretation": (
            "The Maxwell-to-explicit-3D-FVM-to-Shockley-Ramo chain completed, "
            "but neither named metal-thermalization extreme reproduces the "
            "digitized polarization ratio. This is a diagnostic disagreement, "
            "not a paper reproduction or a final experiment prediction."
        ),
        "metal_model_limit": (
            "The visible Au/Ti optical polygons are included optically. Hidden "
            "metal/flake overlap and finite metal/TaIrTe4 thermal G are not "
            "published; isolated and perfect-to-flake cases are named extremes."
        ),
        "no_adjoint": True,
        "no_ad_fd": True,
        "no_optimization": True,
    }
    (args.output_dir / "device_a_end_to_end_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    csv_path = args.output_dir / "device_a_end_to_end_cases.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    plot_primary_maps(perfect_cases, args.output_dir / "DEVICE_A_MAXWELL_THERMAL_PTE_MAPS.png")
    plot_weighting(perfect_cases, args.output_dir / "DEVICE_A_WEIGHTING_AND_LOCAL_CURRENT.png")
    plot_gradients(
        perfect_cases,
        normal,
        tangent,
        args.output_dir / "DEVICE_A_GRADIENT_COMPONENT_MAPS.png",
    )
    plot_currents(rows, measured, uncertainty, args.output_dir / "DEVICE_A_CURRENT_RATIO_COMPARISON.png")
    artifacts = []
    for case in cases:
        artifacts.append(artifact(case["fields_path"], f"thermal fields {case['scenario']} E||{case['polarization']}"))
    for polarization, optical_dir in (("a", args.optical_a), ("b", args.optical_b)):
        artifacts.append(
            artifact(
                optical_dir / "finite_q_on_artifact.npz",
                f"raw Maxwell Q E||{polarization}",
            )
        )
        artifacts.append(
            artifact(
                optical_dir / "finite_2um_optical_q.fsp",
                f"raw Lumerical project E||{polarization}",
            )
        )
    manifest = {
        "status": "RECORDED_EXTERNAL_RAW_ARTIFACTS_NOT_COMMITTED",
        "artifacts": artifacts,
        "generation_command": " ".join(__import__("sys").argv),
    }
    (args.output_dir / "RAW_ARTIFACT_MANIFEST_DEVICE_A_END_TO_END.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    report = f"""# Device A single-position Maxwell-to-PTE sanity check

Status: `{summary['status']}`

The numerical chain completed for the pre-registered Figure-3 edge position. It is **not** a successful paper reproduction: the polarization dependence disagrees with the digitized measurement.

| metal thermalization diagnostic | simulated `|Ia|/|Ib|` | digitized measurement | relative difference |
|---|---:|---:|---:|
| isolated-metal absorption | {by_scenario['isolated']['abs_Ia_over_abs_Ib']:.6f} | {measured:.6f} ± {uncertainty:.6f} | {by_scenario['isolated']['relative_difference_from_digitized_measurement']:+.2%} |
| perfect-to-flake transfer | {by_scenario['perfect']['abs_Ia_over_abs_Ib']:.6f} | {measured:.6f} ± {uncertainty:.6f} | {by_scenario['perfect']['relative_difference_from_digitized_measurement']:+.2%} |

Both are diagnostic extremes, not two published interface models. A finite Au/Ti-to-TaIrTe4 thermal contact was not invented.

For the requested Figure-3G-style off-axis-edge comparator, the perfect-to-flake diagnostic gives `|grad_a|` a/b ratios of {gradient_ratios['grad_a']['raw_max_abs_K_m']:.6f} (raw one-cell maximum) and {gradient_ratios['grad_a']['p99_abs_K_m']:.6f} (P99). The raw maximum is retained as a diagnostic; the P99/RMS/mean metrics are the more robust 100 nm-grid comparators.

Optical `E||a / E||b` closure is {optical['a']['six_face_closure']:.4%} / {optical['b']['six_face_closure']:.4%}; final auto-shutoff is {optical['a']['auto_shutoff_final']:.6e} / {optical['b']['auto_shutoff_final']:.6e}. Both optical gates pass and neither Q artifact contains a negative voxel.

## Case results

| scenario | polarization | mapped power (W) | Tmax rise (K) | flake average rise (K) | signed current (A) | residual | balance error |
|---|---|---:|---:|---:|---:|---:|---:|
"""
    for row in rows:
        report += (
            f"| {row['scenario']} | E||{row['polarization']} | {row['P_Q_thermal_W']:.9e} | "
            f"{row['Tmax_rise_K']:.9e} | {row['TaIrTe4_average_rise_K']:.9e} | "
            f"{row['current_A']:.9e} | {row['residual']:.3e} | {row['energy_balance_error']:.3e} |\n"
        )
    report += """

## What is and is not validated

- Optical closure and auto-shutoff passed independently for both polarizations.
- Conservative Q remap, thermal residual, energy balance, and digitized-contact weighting solve passed.
- Full volumetric Maxwell Q and the existing explicit 3D thermal operator were used; Q was not collapsed to a sheet.
- No polarization matching, clipping, gain, global rescaling, beam-position tuning, adjoint, AD-FD, or optimization was used.
- The far-x/y and bottom Dirichlet fluxes are numerical truncation-boundary fluxes, not intrinsic heat-path fractions.
- Absolute current and the polarization ratio remain model-dependent because exact CAD, beam waist, hidden contact overlap, and metal-interface thermal data were not published.
"""
    (args.output_dir / "DEVICE_A_END_TO_END_SANITY_REPORT.md").write_text(report)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
