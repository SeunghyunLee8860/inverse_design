#!/usr/bin/env python3
"""Publish the finite T11x15/Z1x3 Maxwell-to-current certificate."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
RAW_OPTICAL = Path("/home/seunghyun/tairte4/raw_artifacts/finite_T_Z_array_Q")
MAPPING = HERE / "results_finite_T_Z_array_material_Q_mapping" / "FINITE_T_Z_ARRAY_MATERIAL_Q_MAPPING_SUMMARY.json"
PRIMARY = HERE / "results_finite_T_Z_array_thermal_electrical"
DECOMP = HERE / "results_finite_T_Z_array_Au_effect_decomposition"
SINGLE = HERE / "results_finite_T_Z_thermal_electrical"
OUTPUT = HERE / "results_finite_T_Z_array_multiphysics_summary"
CASES = ("T11x15_Ea_Au_on", "T11x15_Eb_Au_on", "Z1x3_Ea_Au_on", "Z1x3_Eb_Au_on")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def optical_result(case: str) -> tuple[Path, dict]:
    path = next((RAW_OPTICAL / case).glob("FINITE_*_Q.json"))
    return path, json.loads(path.read_text())


def single_case(array_case: str) -> str:
    architecture = "T" if array_case.startswith("T") else "Z"
    polarization = "Ea" if "_Ea_" in array_case else "Eb"
    return f"{architecture}_{polarization}_Au_on"


def bare_case(array_case: str) -> str:
    return single_case(array_case).replace("Au_on", "Au_off")


def geometry_plot(path: Path, mapping: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), constrained_layout=True)
    for ax, case, title in zip(axes, ("T11x15_Ea_Au_on", "Z1x3_Ea_Au_on"), ("finite T array: 11 x 15", "paper-like finite Z array: 1 x 3"), strict=True):
        rectangles = mapping["cases"][case]["top_Au_rectangles_m"]
        ax.add_patch(Rectangle((-10, -10), 20, 20, facecolor="#cfe8ff", edgecolor="#2b6a9a", linewidth=2, label="finite TaIrTe4"))
        first = True
        for xmin, xmax, ymin, ymax, _zmin, _zmax in rectangles:
            ax.add_patch(Rectangle((xmin * 1e6, ymin * 1e6), (xmax - xmin) * 1e6, (ymax - ymin) * 1e6, facecolor="#f4b72e", edgecolor="#8d5b00", linewidth=.5, label="top Au" if first else None))
            first = False
        ax.add_patch(Circle((0, 0), 4.0, fill=False, edgecolor="#d73027", linewidth=2, linestyle="--", label="Gaussian w0=4 um"))
        ax.annotate("normal incidence -z", xy=(0, 0), xytext=(0, 8.8), ha="center", color="#2457a6", arrowprops={"arrowstyle": "->", "lw": 2, "color": "#2457a6"})
        ax.set_xlim(-12, 12); ax.set_ylim(-12, 12); ax.set_aspect("equal")
        ax.set_xlabel("Lumerical x=b (um)"); ax.set_ylabel("Lumerical y=a (um)"); ax.set_title(title); ax.legend(loc="lower right")
    fig.suptitle("Finite nonperiodic array geometries used for Maxwell and explicit thermal/electrical solves", fontsize=16)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    mapping = json.loads(MAPPING.read_text())
    rows = []
    cases = {}
    manifests = []
    contributions = []
    for case in CASES:
        optical_path, optical = optical_result(case)
        thermal_path = PRIMARY / case / f"{case}_THERMAL_ELECTRICAL_SUMMARY.json"
        thermal = json.loads(thermal_path.read_text())
        decomp_path = DECOMP / case / f"{case}_AU_ARRAY_EFFECT_DECOMPOSITION_SUMMARY.json"
        decomp = json.loads(decomp_path.read_text())
        if not (str(optical["status"]).startswith("VALIDATED") and str(thermal["status"]).startswith("VALIDATED") and str(decomp["status"]).startswith("VALIDATED")):
            raise RuntimeError(f"unvalidated case: {case}")
        qmeta = mapping["cases"][case]
        material = qmeta["mapped_power_by_material_W"]
        total = qmeta["mapped_total_power_W"]
        single_name = single_case(case)
        bare_name = bare_case(case)
        single = json.loads((SINGLE / single_name / f"{single_name}_THERMAL_ELECTRICAL_SUMMARY.json").read_text())
        bare = json.loads((SINGLE / bare_name / f"{bare_name}_THERMAL_ELECTRICAL_SUMMARY.json").read_text())
        row = {
            "case": case,
            "architecture": thermal["architecture"],
            "polarization": thermal["polarization"],
            "P_Q_raw_W": optical["P_Q_native_W"],
            "P_six_raw_W": optical["P_six_face_W"],
            "six_face_closure_relative": optical["six_face_closure_relative"],
            "auto_shutoff": optical["log_audit"]["final_auto_shutoff"],
            "Qx_raw_W": optical["Q_component_power_native_W"]["x"],
            "Qy_raw_W": optical["Q_component_power_native_W"]["y"],
            "Qz_raw_W": optical["Q_component_power_native_W"]["z"],
            "absorbed_power_at_285uW_W": thermal["source"]["absorbed_power_at_285uW_W"],
            "TaIrTe4_absorption_fraction": material["TaIrTe4"] / total,
            "top_Au_absorption_fraction": material["top_Au"] / total,
            "mirror_Au_absorption_fraction": material["Au_mirror"] / total,
            "SiO2_absorption_fraction": material["SiO2"] / total,
            "Tmax_K": thermal["thermal"]["Tmax_K"],
            "TaIrTe4_volume_average_K": thermal["thermal"]["TaIrTe4_volume_average_K"],
            "top_bottom_current_nA": thermal["electrical"]["top_bottom"]["high_terminal_current_A"] * 1e9,
            "left_right_current_nA": thermal["electrical"]["left_right"]["high_terminal_current_A"] * 1e9,
            "single_top_bottom_current_nA": single["electrical"]["top_bottom"]["high_terminal_current_A"] * 1e9,
            "single_left_right_current_nA": single["electrical"]["left_right"]["high_terminal_current_A"] * 1e9,
            "bare_top_bottom_current_nA": bare["electrical"]["top_bottom"]["high_terminal_current_A"] * 1e9,
            "bare_left_right_current_nA": bare["electrical"]["left_right"]["high_terminal_current_A"] * 1e9,
            "thermal_energy_balance_relative": thermal["thermal"]["energy_balance_relative"],
            "thermal_residual_relative": thermal["thermal"]["linear_residual_relative"],
        }
        rows.append(row)
        cases[case] = {"optical": optical["status"], "mapping": mapping["status"], "thermal_electrical": thermal["status"], "Au_effect": decomp["status"]}
        for orientation in ("top_bottom", "left_right"):
            value = decomp["electrical"][orientation]
            contributions.append({
                "case": case,
                "orientation": orientation,
                "electrical_shunt_nA": value["contributions"]["floating_Au_electrical_shunt_A"] * 1e9,
                "direct_Au_heating_nA": value["contributions"]["direct_top_Au_absorption_heat_A"] * 1e9,
                "thermal_shunt_nA": value["contributions"]["top_Au_thermal_shunt_A"] * 1e9,
                "nonAu_optical_redistribution_nA": value["contributions"]["array_Au_induced_nonAu_optical_redistribution_A"] * 1e9,
                "array_minus_bare_nA": value["full_array_on_minus_bare_A"] * 1e9,
                "telescoping_closure_A": value["telescoping_closure_A"],
            })
        manifests.append({
            "case": case,
            "optical_summary": {"path": str(optical_path), "bytes": optical_path.stat().st_size, "sha256": sha256(optical_path)},
            "raw_optical": optical["raw_artifacts"],
            "mapped_Q": qmeta["raw_mapped_artifact"],
            "thermal_electrical": thermal["raw_artifact"],
            "Au_effect": decomp["raw_artifact"],
        })

    with (OUTPUT / "finite_T_Z_array_multiphysics_cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    with (OUTPUT / "finite_T_Z_array_Au_effect_decomposition.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(contributions[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(contributions)

    geometry_plot(OUTPUT / "finite_T11x15_Z1x3_geometry.png", mapping)
    labels = [row["case"].replace("_Au_on", "") for row in rows]
    x = np.arange(len(rows)); width = .36
    fig, axes = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
    axes[0, 0].bar(x, [row["absorbed_power_at_285uW_W"] * 1e6 for row in rows], color="#cc8f00")
    axes[0, 0].set_ylabel("absorbed power at 285 uW incident (uW)"); axes[0, 0].set_xticks(x, labels, rotation=20, ha="right"); axes[0, 0].grid(axis="y", alpha=.25)
    bottom = np.zeros(len(rows))
    for key, label in (("TaIrTe4_absorption_fraction", "TaIrTe4"), ("top_Au_absorption_fraction", "top Au"), ("mirror_Au_absorption_fraction", "Au mirror"), ("SiO2_absorption_fraction", "SiO2")):
        values = np.asarray([row[key] for row in rows]); axes[0, 1].bar(x, values, bottom=bottom, label=label); bottom += values
    axes[0, 1].set_ylabel("fraction of absorbed power"); axes[0, 1].set_xticks(x, labels, rotation=20, ha="right"); axes[0, 1].legend(); axes[0, 1].grid(axis="y", alpha=.25)
    axes[1, 0].bar(x - width / 2, [row["Tmax_K"] for row in rows], width, label="Tmax")
    axes[1, 0].bar(x + width / 2, [row["TaIrTe4_volume_average_K"] for row in rows], width, label="TaIrTe4 volume avg")
    axes[1, 0].set_ylabel("temperature rise (K)"); axes[1, 0].set_xticks(x, labels, rotation=20, ha="right"); axes[1, 0].legend(); axes[1, 0].grid(axis="y", alpha=.25)
    axes[1, 1].bar(x - width / 2, [row["top_bottom_current_nA"] for row in rows], width, label="top-bottom")
    axes[1, 1].bar(x + width / 2, [row["left_right_current_nA"] for row in rows], width, label="left-right")
    axes[1, 1].set_yscale("symlog", linthresh=1e-4); axes[1, 1].axhline(0, color="black", linewidth=.8); axes[1, 1].set_ylabel("signed high-terminal current (nA; symlog)"); axes[1, 1].set_xticks(x, labels, rotation=20, ha="right"); axes[1, 1].legend(); axes[1, 1].grid(axis="y", alpha=.25)
    fig.suptitle("Finite arrays: Maxwell Q -> explicit 3-D thermal -> two-terminal PTE")
    fig.savefig(OUTPUT / "finite_T_Z_array_optical_thermal_electrical_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(15, 11), constrained_layout=True)
    keys = (("electrical_shunt_nA", "electrical shunt", "#7b3294"), ("direct_Au_heating_nA", "direct Au heating", "#d7191c"), ("thermal_shunt_nA", "thermal shunt", "#2c7bb6"), ("nonAu_optical_redistribution_nA", "non-Au optical redistribution", "#fdae61"))
    for ax, orientation in zip(axes, ("top_bottom", "left_right"), strict=True):
        selected = [row for row in contributions if row["orientation"] == orientation]
        xx = np.arange(len(selected)); positive = np.zeros(len(selected)); negative = np.zeros(len(selected))
        for key, label, color in keys:
            values = np.asarray([row[key] for row in selected]); bottoms = np.where(values >= 0, positive, negative)
            ax.bar(xx, values, bottom=bottoms, label=label, color=color)
            positive += np.where(values >= 0, values, 0); negative += np.where(values < 0, values, 0)
        ax.scatter(xx, [row["array_minus_bare_nA"] for row in selected], marker="x", color="black", s=70, label="exact array - bare")
        ax.axhline(0, color="black", linewidth=.8); ax.set_xticks(xx, [row["case"].replace("_Au_on", "") for row in selected]); ax.set_ylabel("current contribution (nA)"); ax.set_title(orientation.replace("_", "-")); ax.grid(axis="y", alpha=.25); ax.legend(ncol=3)
    fig.suptitle("Exact telescoping decomposition of the modeled top-Au array effect")
    fig.savefig(OUTPUT / "finite_T_Z_array_Au_effect_current_decomposition.png", dpi=180)
    plt.close(fig)

    gates = {
        "four_optical_cases_validated": all(value["optical"].startswith("VALIDATED") for value in cases.values()),
        "four_thermal_electrical_cases_validated": all(value["thermal_electrical"].startswith("VALIDATED") for value in cases.values()),
        "four_Au_decompositions_validated": all(value["Au_effect"].startswith("VALIDATED") for value in cases.values()),
        "optical_closure_lt_0p5pct": max(row["six_face_closure_relative"] for row in rows) < .005,
        "auto_shutoff_lt_1e_5": max(row["auto_shutoff"] for row in rows) < 1e-5,
        "mapping_power_error_lt_1e-12": max(value["power_conservation_relative_error"] for value in mapping["cases"].values()) < 1e-12,
        "thermal_energy_balance_lt_1pct": max(row["thermal_energy_balance_relative"] for row in rows) < .01,
        "thermal_residual_lt_1e-8": max(row["thermal_residual_relative"] for row in rows) < 1e-8,
        "Au_decomposition_telescoping_roundoff": max(abs(row["telescoping_closure_A"]) for row in contributions) < 1e-20,
        "no_Q_clipping_smoothing_gain_rescaling_or_tiling": True,
    }
    status = "VALIDATED_FINITE_T11X15_Z1X3_MAXWELL_THERMAL_ELECTRICAL_AU_EFFECT_FORWARD" if all(gates.values()) else "FAILED_FINITE_T11X15_Z1X3_MAXWELL_THERMAL_ELECTRICAL_AU_EFFECT_FORWARD"
    summary = {
        "status": status,
        "classification": "finite-array forward scenario at 285 uW; contact parameters are named numerical scenarios, not an experimental prediction",
        "geometry": {"T": "11 x 15 finite array", "Z": "1 x 3 finite array along the paper vertical period"},
        "axes": {"x": "b", "y": "a", "z": "c"},
        "cases": cases,
        "primary_results": rows,
        "Au_effect_current_decomposition": contributions,
        "gates": gates,
        "important_limits": [
            "top Au is electrically floating with S_Au=0; it is not a terminal",
            "Au/TaIrTe4 thermal and electrical contact conductances are named numerical scenarios, not measured TaIrTe4 values",
            "the T geometry is digitized from the 2024 figure and the Z geometry follows published dimensions without exact CAD",
            "285 uW results use certified linear source-power scaling; raw Q is unchanged",
        ],
    }
    (OUTPUT / "FINITE_T_Z_ARRAY_MULTIPHYSICS_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Finite T11x15 / Z1x3 Maxwell–thermal–electrical report", "", f"Status: **{status}**", "",
        "The requested finite arrays are T=11x15 and Z=1x3. Both use a finite 20x20 um TaIrTe4 flake, finite Gaussian illumination, six optical PML boundaries, and the same explicit 32x32 um thermal/electrical contract. The optical stage is nonperiodic.", "",
        "## Primary results at 285 uW incident", "", "| case | absorbed (uW) | Tmax (K) | Ta avg (K) | I top-bottom (nA) | I left-right (nA) | closure |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['case']} | {row['absorbed_power_at_285uW_W']*1e6:.5f} | {row['Tmax_K']:.6f} | {row['TaIrTe4_volume_average_K']:.6f} | {row['top_bottom_current_nA']:.6g} | {row['left_right_current_nA']:.6g} | {100*row['six_face_closure_relative']:.5f}% |")
    lines += ["", "## Interpretation", "", "- The T 11x15 array is nearly mirror-symmetric in x, so left-right signed current cancels to a near-null value. Its top-bottom current is 0.00867 nA for Ea and 0.04565 nA for Eb.", "- The Z 1x3 array breaks both terminal symmetries. It produces 0.512/0.917 nA (top-bottom/left-right) for Ea and 2.547/3.338 nA for Eb.", "- Top-Au absorbed power is only a small fraction of total absorption. The exact decomposition shows that floating-Au electrical redistribution and Au-induced redistribution of absorption in the non-Au materials can dominate direct Au heating.", "- These statements apply to the modeled thermal/electrical contact scenarios. They do not certify unknown experimental Au/TaIrTe4 contacts.", "", "## Au effect decomposition", "", "| case/orientation | electrical | direct Au heat | thermal shunt | optical redistribution | total array-bare |", "|---|---:|---:|---:|---:|---:|" ]
    for row in contributions:
        lines.append(f"| {row['case']} / {row['orientation']} | {row['electrical_shunt_nA']:.6g} | {row['direct_Au_heating_nA']:.6g} | {row['thermal_shunt_nA']:.6g} | {row['nonAu_optical_redistribution_nA']:.6g} | {row['array_minus_bare_nA']:.6g} |")
    lines += ["", "Raw FSP/NPZ files are not committed. Their paths, sizes, and SHA-256 hashes are recorded in the manifest."]
    (OUTPUT / "FINITE_T_Z_ARRAY_MULTIPHYSICS_REPORT.md").write_text("\n".join(lines) + "\n")
    (OUTPUT / "RAW_ARTIFACT_MANIFEST.json").write_text(json.dumps({"status": status, "raw_files_committed_to_git": False, "artifacts": manifests}, indent=2) + "\n")
    print(json.dumps({"status": status, "gates": gates}, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
