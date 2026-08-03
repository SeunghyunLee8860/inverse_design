#!/usr/bin/env python3
"""Publish fixed-Lumerical-coordinate plots for the Device-A 9x2x2 matrix."""

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
from matplotlib.path import Path as PolygonPath
import numpy as np

from photothermal_pte.validation.paper_ir_sanity import (
    run_device_a_explicit_thermal_pte as thermal,
)


SCENARIOS = ("thermally_grown", "evaporated")
POLARIZATIONS = ("a", "b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--position-contract", type=Path, required=True)
    parser.add_argument("--optical-root", type=Path, required=True)
    parser.add_argument("--thermal-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def finite_values(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    selected = np.asarray(values, float)
    use = np.isfinite(selected)
    if mask is not None:
        use &= mask
    return selected[use]


def pcolor(
    ax: plt.Axes,
    x_um: np.ndarray,
    y_um: np.ndarray,
    values: np.ndarray,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    flake_vertices_um: np.ndarray,
    beam_center_um: np.ndarray,
):
    mesh = ax.pcolormesh(
        x_um,
        y_um,
        np.ma.masked_invalid(values).T,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    closed = np.vstack((flake_vertices_um, flake_vertices_um[0]))
    ax.plot(closed[:, 0], closed[:, 1], color="cyan", linewidth=0.7)
    ax.scatter(
        beam_center_um[0], beam_center_um[1], s=32, marker="+", color="lime",
        linewidth=1.2,
    )
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("Lumerical x = crystal b (µm)")
    ax.set_ylabel("Lumerical y = crystal a (µm)")
    return mesh


def load_summary(root: Path, scenario: str, label: str, pol: str) -> dict[str, Any]:
    path = root / scenario / label / pol / "summary.json"
    payload = json.loads(path.read_text())
    if payload["status"] != "COMPLETED_DEVICE_A_NINE_POSITION_THERMAL_CASE":
        raise RuntimeError(f"thermal case not complete: {path}")
    return payload


def load_fields(root: Path, scenario: str, label: str, pol: str):
    path = root / scenario / label / pol / "thermal_lumerical_coordinate_fields.npz"
    if not path.is_file():
        raise RuntimeError(f"missing thermal fields: {path}")
    return np.load(path, allow_pickle=False)


def raw_optical_qxy(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as raw:
        x = np.asarray(raw["x_m"], float)
        y = np.asarray(raw["y_m"], float)
        z = np.asarray(raw["z_m"], float)
        q = np.asarray(raw["Q_on_W_m3"], float)
        flake = np.asarray(raw["exact_flake_mask"], bool)
    z_edges = thermal.nodal_control_volume_edges(z)
    dz = np.diff(z_edges)
    # This is the stored full Maxwell loss restricted only for visualization
    # by the artifact's exact flake support.  Production thermal attribution
    # is independently performed by exact optical/thermal volume overlap.
    qxy = np.sum(np.where(flake, q, 0.0) * dz[None, None, :], axis=2)
    return x * 1.0e6, y * 1.0e6, qxy


def make_raw_q_mosaics(
    contract: dict[str, Any], optical_root: Path, output_dir: Path,
    flake_vertices_um: np.ndarray,
) -> list[Path]:
    outputs: list[Path] = []
    for pol in POLARIZATIONS:
        fields = []
        maximum = 0.0
        for case in contract["cases"]:
            artifact = optical_root / case["label"] / pol / "finite" / "finite_q_on_artifact.npz"
            x, y, qxy = raw_optical_qxy(artifact)
            fields.append((case, x, y, qxy))
            maximum = max(maximum, float(np.nanmax(qxy)))
        fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
        for ax, (case, x, y, qxy) in zip(axes.ravel(), fields, strict=True):
            m = pcolor(
                ax, x, y, qxy,
                f"{case['label']}  E||{pol}", "inferno", 0.0, maximum,
                flake_vertices_um,
                np.asarray(case["beam_center_lumerical_um"], float),
            )
        fig.colorbar(m, ax=axes, label="Raw Maxwell TaIrTe₄-support Q areal (W/m²)")
        fig.suptitle(
            f"Device-A raw optical Q, E||{pol} — fixed Lumerical x=b, y=a",
            fontsize=15,
        )
        path = output_dir / f"RAW_OPTICAL_Q_LUMERICAL_COORDINATES_E{pol}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(path)
    return outputs


def make_case_panels(
    contract: dict[str, Any], thermal_root: Path, output_dir: Path,
    flake_vertices_um: np.ndarray,
) -> list[Path]:
    outputs: list[Path] = []
    panel_dir = output_dir / "case_panels"
    panel_dir.mkdir(exist_ok=True)
    for scenario in SCENARIOS:
        for case in contract["cases"]:
            loaded = {
                pol: load_fields(thermal_root, scenario, case["label"], pol)
                for pol in POLARIZATIONS
            }
            arrays: dict[str, dict[str, np.ndarray]] = {}
            for pol, raw in loaded.items():
                valid = np.asarray(raw["strict_valid_xy_mask"], bool)
                arrays[pol] = {
                    "Q": np.asarray(raw["Q_areal_W_m2"], float),
                    "T": np.asarray(raw["temperature_flake_average_K"], float),
                    "G": np.where(valid, np.asarray(raw["grad_T_magnitude_K_m"], float), np.nan),
                    "I": np.where(valid, np.asarray(raw["strict_current_contribution_A_m2"], float), np.nan),
                }
            scales = {}
            for key in ("Q", "T", "G"):
                combined = np.concatenate([finite_values(arrays[p][key]) for p in POLARIZATIONS])
                scales[key] = (0.0, float(np.max(combined)))
            current = np.concatenate([finite_values(arrays[p]["I"]) for p in POLARIZATIONS])
            imax = float(np.max(np.abs(current)))
            scales["I"] = (-imax, imax)
            fig, axes = plt.subplots(2, 4, figsize=(21, 10), constrained_layout=True)
            last_mesh = {}
            for row, pol in enumerate(POLARIZATIONS):
                raw = loaded[pol]
                x = centers(np.asarray(raw["x_edges_m"], float)) * 1.0e6
                y = centers(np.asarray(raw["y_edges_m"], float)) * 1.0e6
                beam = np.asarray(case["beam_center_lumerical_um"], float)
                for col, (key, title, cmap, unit) in enumerate((
                    ("Q", "mapped TaIrTe₄ Q", "inferno", "W/m²"),
                    ("T", "thickness-avg ΔT", "magma", "K"),
                    ("G", "strict-centered |∇T|", "viridis", "K/m"),
                    ("I", "strict current contribution", "coolwarm", "A/m²"),
                )):
                    last_mesh[key] = pcolor(
                        axes[row, col], x, y, arrays[pol][key],
                        f"E||{pol}: {title}", cmap, *scales[key],
                        flake_vertices_um, beam,
                    )
            for col, (key, unit) in enumerate((("Q", "W/m²"), ("T", "K"), ("G", "K/m"), ("I", "A/m²"))):
                fig.colorbar(last_mesh[key], ax=axes[:, col], label=unit)
            summaries = {
                pol: load_summary(thermal_root, scenario, case["label"], pol)
                for pol in POLARIZATIONS
            }
            annotation = "   ".join(
                f"E||{pol}: I={summaries[pol]['thermal']['production_current_A']*1e9:.4g} nA, "
                f"Tmax={summaries[pol]['thermal']['Tmax_rise_K']:.4g} K"
                for pol in POLARIZATIONS
            )
            fig.suptitle(
                f"{case['label']} — {scenario.replace('_', ' ')} SiO₂ interface\n"
                f"Lumerical x=b, y=a; green +=source center; cyan=flake; {annotation}",
                fontsize=14,
            )
            path = panel_dir / f"{scenario}_{case['label']}_Q_T_GRADIENT_CURRENT.png"
            fig.savefig(path, dpi=170)
            plt.close(fig)
            for raw in loaded.values():
                raw.close()
            outputs.append(path)
    return outputs


def collect_table(
    contract: dict[str, Any], optical_root: Path, thermal_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for case in contract["cases"]:
            for pol in POLARIZATIONS:
                summary = load_summary(thermal_root, scenario, case["label"], pol)
                result_path = optical_root / case["label"] / pol / "finite" / "case_result.json"
                optical = json.loads(result_path.read_text())["run_result"]
                with load_fields(thermal_root, scenario, case["label"], pol) as field:
                    valid = np.asarray(field["strict_valid_xy_mask"], bool)
                    gradient = np.asarray(field["grad_T_magnitude_K_m"], float)[valid]
                rows.append({
                    "scenario": scenario,
                    "G_TaIrTe4_SiO2_W_m2K": summary["G_TaIrTe4_SiO2_W_m2K"],
                    "position": case["label"],
                    "category": case["category"],
                    "vertical_level": case["vertical_level"],
                    "polarization": pol,
                    "beam_x_b_um": case["beam_center_lumerical_um"][0],
                    "beam_y_a_um": case["beam_center_lumerical_um"][1],
                    "P_Q_W_at_1_W_m2": optical["P_Q_W"],
                    "P_six_W_at_1_W_m2": optical["P_six_face_W"],
                    "six_face_closure": optical["six_face_relative_closure"],
                    "auto_shutoff": optical["auto_shutoff"]["final_value"],
                    "mapped_source_power_W_at_285uW_incident": summary["thermal"]["source_power_W"],
                    "Tmax_rise_K": summary["thermal"]["Tmax_rise_K"],
                    "TaIrTe4_volume_average_rise_K": summary["thermal"]["TaIrTe4_volume_average_rise_K"],
                    "strict_gradient_max_K_m": float(np.max(gradient)),
                    "strict_gradient_p99_K_m": float(np.percentile(gradient, 99.0)),
                    "production_current_A": summary["thermal"]["production_current_A"],
                    "strict_current_A": summary["thermal"]["strict_current_A"],
                    "mapping_relative_power_error": summary["mapping"]["mapping_relative_power_error"],
                    "thermal_energy_balance_relative_error": summary["thermal"]["energy_balance_relative_error"],
                    "linear_residual_relative": summary["thermal"]["linear_residual_relative"],
                    "lateral_numerical_boundary_flux_fraction": summary["thermal"]["lateral_numerical_boundary_flux_fraction"],
                })
    return rows


def make_summary_plots(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    labels = []
    ordered = []
    for scenario in SCENARIOS:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        labels = list(dict.fromkeys(row["position"] for row in scenario_rows))
        fig, axes = plt.subplots(2, 2, figsize=(17, 10), constrained_layout=True)
        x = np.arange(len(labels))
        width = 0.36
        for offset, pol, color in ((-width/2, "a", "tab:blue"), (width/2, "b", "tab:orange")):
            selected = [next(row for row in scenario_rows if row["position"] == label and row["polarization"] == pol) for label in labels]
            axes[0, 0].bar(x + offset, [r["mapped_source_power_W_at_285uW_incident"]*1e6 for r in selected], width, label=f"E||{pol}", color=color)
            axes[0, 1].bar(x + offset, [r["Tmax_rise_K"] for r in selected], width, label=f"E||{pol}", color=color)
            axes[1, 0].bar(x + offset, [r["strict_gradient_p99_K_m"] for r in selected], width, label=f"E||{pol}", color=color)
            axes[1, 1].bar(x + offset, [r["production_current_A"]*1e9 for r in selected], width, label=f"E||{pol}", color=color)
        for ax, ylabel in zip(axes.ravel(), ("mapped absorbed power (µW)", "Tmax rise (K)", "strict |∇T| P99 (K/m)", "production current (nA)"), strict=True):
            ax.set_xticks(x, labels, rotation=35, ha="right")
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25)
            ax.legend()
        fig.suptitle(f"Device-A nine-position summary — {scenario.replace('_', ' ')} SiO₂; fixed Lumerical x=b, y=a")
        path = output_dir / f"NINE_POSITION_SUMMARY_{scenario.upper()}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.36
    for offset, scenario, color in ((-width/2, "thermally_grown", "tab:green"), (width/2, "evaporated", "tab:red")):
        for metric, ax, title in (("Tmax_rise_K", axes[0], "Tmax b/a"), ("production_current_A", axes[1], "|current b/a|")):
            values = []
            for label in labels:
                a = next(r for r in rows if r["scenario"] == scenario and r["position"] == label and r["polarization"] == "a")[metric]
                b = next(r for r in rows if r["scenario"] == scenario and r["position"] == label and r["polarization"] == "b")[metric]
                values.append(abs(b/a) if a != 0.0 else np.nan)
            ax.bar(x + offset, values, width, label=scenario.replace("_", " "), color=color)
            ax.set_title(title)
    for ax in axes:
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_ylabel("E||b / E||a ratio")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    path = output_dir / "POLARIZATION_AND_INTERFACE_G_RATIOS.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)
    return outputs


def main() -> int:
    args = parse_args()
    contract = json.loads(args.position_contract.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    first = json.loads((args.optical_root / contract["cases"][0]["label"] / "a" / "finite" / "case_result.json").read_text())
    flake_vertices = np.asarray(first["pre_run_contract"]["geometry"]["flake_vertices_um"], float)
    raw_plots = make_raw_q_mosaics(contract, args.optical_root, args.output_dir, flake_vertices)
    case_plots = make_case_panels(contract, args.thermal_root, args.output_dir, flake_vertices)
    rows = collect_table(contract, args.optical_root, args.thermal_root)
    summary_plots = make_summary_plots(rows, args.output_dir)
    csv_path = args.output_dir / "device_a_nine_position_two_interface_results.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.output_dir / "device_a_nine_position_two_interface_summary.json"
    json_path.write_text(json.dumps({
        "status": "COMPLETED_DEVICE_A_NINE_POSITION_TWO_INTERFACE_SUMMARY",
        "coordinate_frame": contract["coordinate_frame"],
        "rows": rows,
        "interpretation": {
            "thermally_grown_and_evaporated_are_named_physical_scenarios": True,
            "absolute_current_is_not_claimed_as_experimentally_reproduced": True,
            "current_definition": "volume-integrated anisotropic Shockley-Ramo PTE current; strict-centered map is an additional diagnostic",
            "thermal_Q_source": "TaIrTe4-only exact intersection-density overlap mapping at 285 uW incident power",
            "SiO2_optical_loss_is_not_used_as_a_thermal_source": True,
            "far_boundary_flux": "numerical truncation flux, not a physical heat-path fraction",
        },
    }, indent=2) + "\n")
    manifest_path = args.output_dir / "RAW_ARTIFACT_MANIFEST.json"
    artifacts = []
    for case in contract["cases"]:
        for pol in POLARIZATIONS:
            path = args.optical_root / case["label"] / pol / "finite" / "finite_q_on_artifact.npz"
            artifacts.append({"position": case["label"], "polarization": pol, "path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest_path.write_text(json.dumps({
        "raw_artifacts_committed_to_git": False,
        "raw_optical_artifacts": artifacts,
        "generated_report_artifacts": [str(path.resolve()) for path in raw_plots + case_plots + summary_plots],
    }, indent=2) + "\n")
    report_path = args.output_dir / "DEVICE_A_NINE_POSITION_TWO_INTERFACE_REPORT.md"
    report_path.write_text(
        "# Device-A nine-position, two-interface-G result\n\n"
        "All spatial plots use the fixed Lumerical frame **x = crystal b, y = crystal a**. "
        "The device, PML, monitor, and mesh geometry are invariant; only the scalar-Gaussian source is translated.\n\n"
        "The two TaIrTe4/SiO2 conductances are reported as separate named physical scenarios: "
        "thermally grown (7.37e6 W/m2K) and evaporated (7.37e4 W/m2K). "
        "Neither is promoted as a fabrication-independent truth.\n\n"
        "Thermal Q uses TaIrTe4-only, exact optical-cell/thermal-material intersection-density mapping at 285 uW incident power. "
        "SiO2 absorption remains in the optical audit but is not added as a thermal source. No Q clipping, smoothing, gain, rescaling, or tiling is used.\n\n"
        "Current is a full-volume anisotropic Shockley-Ramo PTE integral. A strict-centered current-density map is also shown, with cells masked unless all +/-x and +/-y TaIrTe4 neighbours exist. "
        "Because the digitized-model resistance differs from the measured device, absolute current is not called an experimental reproduction.\n\n"
        f"- [CSV]({csv_path.name})\n- [JSON]({json_path.name})\n- [manifest]({manifest_path.name})\n"
    )
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
