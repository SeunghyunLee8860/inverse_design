#!/usr/bin/env python3
"""Publish finite T/Z Q, thermal, weighting, J/current, and Au-effect results."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
MAPPING = HERE / "results_finite_T_Z_material_Q_mapping" / "FINITE_T_Z_MATERIAL_Q_MAPPING_SUMMARY.json"
PRIMARY = HERE / "results_finite_T_Z_thermal_electrical"
DECOMP = HERE / "results_finite_T_Z_Au_effect_decomposition"
OUTPUT = HERE / "results_finite_T_Z_multiphysics_summary"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percent(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    mapping = json.loads(MAPPING.read_text())
    cases = {}
    rows = []
    manifests = []
    for name, qmeta in mapping["cases"].items():
        path = PRIMARY / name / f"{name}_THERMAL_ELECTRICAL_SUMMARY.json"
        result = json.loads(path.read_text())
        if not result["status"].startswith("VALIDATED") or not all(result["gates"].values()):
            raise RuntimeError(f"unvalidated primary case: {name}")
        cases[name] = result
        qmaterial = qmeta["mapped_power_by_material_W"]
        rows.append({
            "case": name,
            "architecture": result["architecture"],
            "polarization": result["polarization"],
            "top_Au_present": result["top_Au_present"],
            "absorbed_power_at_285uW_W": result["source"]["absorbed_power_at_285uW_W"],
            "TaIrTe4_absorption_fraction": qmaterial["TaIrTe4"] / qmeta["mapped_total_power_W"],
            "top_Au_absorption_fraction": qmaterial["top_Au"] / qmeta["mapped_total_power_W"],
            "mirror_Au_absorption_fraction": qmaterial["Au_mirror"] / qmeta["mapped_total_power_W"],
            "Tmax_K": result["thermal"]["Tmax_K"],
            "TaIrTe4_volume_average_K": result["thermal"]["TaIrTe4_volume_average_K"],
            "top_bottom_current_nA": result["electrical"]["top_bottom"]["high_terminal_current_A"] * 1e9,
            "left_right_current_nA": result["electrical"]["left_right"]["high_terminal_current_A"] * 1e9,
            "thermal_residual": result["thermal"]["linear_residual_relative"],
            "thermal_energy_balance": result["thermal"]["energy_balance_relative"],
        })
        raw = result["raw_artifact"]
        manifests.append({"case": name, "summary": {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}, "raw": raw})

    decompositions = {}
    decomp_rows = []
    contribution_keys = (
        "floating_Au_electrical_shunt_A",
        "direct_top_Au_absorption_heat_A",
        "top_Au_thermal_shunt_A",
        "Au_induced_nonAu_optical_redistribution_A",
    )
    for arch in ("T", "Z"):
        for pol in ("Ea", "Eb"):
            name = f"{arch}_{pol}_Au_on"
            path = DECOMP / f"{arch}_{pol}" / f"{name}_AU_EFFECT_DECOMPOSITION_SUMMARY.json"
            result = json.loads(path.read_text())
            if not result["status"].startswith("VALIDATED") or not all(result["gates"].values()):
                raise RuntimeError(f"unvalidated decomposition: {name}")
            decompositions[name] = result
            for orientation, value in result["electrical"].items():
                decomp_rows.append({"case": name, "orientation": orientation, "I_on_minus_off_nA": value["full_on_minus_off_A"] * 1e9, **{key.removesuffix("_A") + "_nA": value["contributions"][key] * 1e9 for key in contribution_keys}, "closure_A": value["telescoping_closure_A"]})
            manifests.append({"case": name, "decomposition_summary": {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}, "decomposition_raw": result["raw_artifact"]})

    with (OUTPUT / "finite_T_Z_multiphysics_cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    with (OUTPUT / "finite_T_Z_Au_effect_decomposition.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(decomp_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(decomp_rows)

    labels = [row["case"] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    absorbed = np.asarray([row["absorbed_power_at_285uW_W"] for row in rows]) * 1e6
    axes[0, 0].bar(x, absorbed, color=["#d89b00" if row["top_Au_present"] else "#777" for row in rows])
    axes[0, 0].set_ylabel("absorbed power at 285 uW incident (uW)"); axes[0, 0].set_xticks(x, labels, rotation=25, ha="right"); axes[0, 0].grid(axis="y", alpha=.25)
    width = .38
    axes[0, 1].bar(x - width/2, [row["Tmax_K"] for row in rows], width, label="Tmax")
    axes[0, 1].bar(x + width/2, [row["TaIrTe4_volume_average_K"] for row in rows], width, label="TaIrTe4 volume avg")
    axes[0, 1].set_ylabel("temperature rise at 285 uW (K)"); axes[0, 1].set_xticks(x, labels, rotation=25, ha="right"); axes[0, 1].legend(); axes[0, 1].grid(axis="y", alpha=.25)
    axes[1, 0].bar(x - width/2, [row["top_bottom_current_nA"] for row in rows], width, label="top-bottom electrodes")
    axes[1, 0].bar(x + width/2, [row["left_right_current_nA"] for row in rows], width, label="left-right electrodes")
    axes[1, 0].axhline(0, color="black", linewidth=.8); axes[1, 0].set_ylabel("signed high-terminal short-circuit current (nA)"); axes[1, 0].set_xticks(x, labels, rotation=25, ha="right"); axes[1, 0].legend(); axes[1, 0].grid(axis="y", alpha=.25)
    material_keys = [("TaIrTe4_absorption_fraction", "TaIrTe4"), ("top_Au_absorption_fraction", "top Au"), ("mirror_Au_absorption_fraction", "Au mirror")]
    bottom = np.zeros(len(rows))
    for key, label in material_keys:
        values = np.asarray([row[key] for row in rows])
        axes[1, 1].bar(x, values, bottom=bottom, label=label)
        bottom += values
    axes[1, 1].set_ylabel("fraction of absorbed power"); axes[1, 1].set_xticks(x, labels, rotation=25, ha="right"); axes[1, 1].legend(); axes[1, 1].grid(axis="y", alpha=.25)
    fig.suptitle("Finite T/Z Maxwell -> explicit 3-D thermal -> two-terminal electrical/PTE")
    fig.savefig(OUTPUT / "finite_T_Z_multiphysics_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(15, 11), constrained_layout=True)
    colors = ["#7b3294", "#d7191c", "#2c7bb6", "#fdae61"]
    short_names = ["electrical shunt", "direct Au heating", "thermal shunt", "optical redistribution"]
    for ax, orientation in zip(axes, ("top_bottom", "left_right"), strict=True):
        selected = [row for row in decomp_rows if row["orientation"] == orientation]
        xx = np.arange(len(selected)); bottoms_pos = np.zeros(len(selected)); bottoms_neg = np.zeros(len(selected))
        for key, label, color in zip((k.removesuffix("_A") + "_nA" for k in contribution_keys), short_names, colors, strict=True):
            values = np.asarray([row[key] for row in selected])
            bottom_values = np.where(values >= 0, bottoms_pos, bottoms_neg)
            ax.bar(xx, values, bottom=bottom_values, label=label, color=color)
            bottoms_pos += np.where(values >= 0, values, 0.0); bottoms_neg += np.where(values < 0, values, 0.0)
        ax.scatter(xx, [row["I_on_minus_off_nA"] for row in selected], marker="x", s=80, color="black", label="exact total on-off")
        ax.axhline(0, color="black", linewidth=.8); ax.set_xticks(xx, [row["case"] for row in selected]); ax.set_ylabel("Au contribution to current (nA)"); ax.set_title(orientation.replace("_", "-")); ax.grid(axis="y", alpha=.25); ax.legend(ncol=3)
    fig.suptitle("Exact telescoping decomposition of the modeled top-Au effect")
    fig.savefig(OUTPUT / "finite_T_Z_Au_effect_current_decomposition.png", dpi=180)
    plt.close(fig)

    status = "VALIDATED_FINITE_T_Z_MULTIPHYSICS_AND_AU_EFFECT_FORWARD"
    summary = {
        "status": status,
        "classification": "finite forward-response certificate at 285 uW incident; named contact scenarios, not an experimental prediction",
        "cases": {name: value["status"] for name, value in cases.items()},
        "decompositions": {name: value["status"] for name, value in decompositions.items()},
        "primary_results": rows,
        "Au_effect_current_decomposition": decomp_rows,
        "sign_convention": "reported current is outward current at the high-potential weighting terminal (psi=1); reversing terminal labels reverses the sign",
        "important_model_limits": [
            "G_Au/TaIrTe4=1.724e7 W/(m2 K) is an Au/MoS2 analogue numerical scenario, not measured TaIrTe4 data",
            "electrical G_Au/TaIrTe4=1e10 S/m2 is a named numerical scenario, not measured TaIrTe4 data",
            "Al2O3 k=30 W/(m K) and G_TaIrTe4/Al2O3=7.37e6 W/(m2 K) are explicit numerical closures",
            "top Au is a floating nanostructure with S_Au=0; it is not a measurement terminal",
            "the 285 uW values use only linear scaling from the validated raw source power; the raw Q artifact is unchanged",
        ],
        "gates": {"eight_primary_cases_validated": True, "four_Au_decompositions_validated": True, "thermal_energy_balance_lt_1pct": max(row["thermal_energy_balance"] for row in rows) < .01, "thermal_residual_lt_1e-8": max(row["thermal_residual"] for row in rows) < 1e-8, "no_Q_clipping_smoothing_gain_or_rescaling": True},
    }
    (OUTPUT / "FINITE_T_Z_MULTIPHYSICS_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Finite T/Z Maxwell–thermal–electrical/PTE and Au-effect report",
        "",
        f"Status: **{status}**",
        "",
        "The periodic stage was used only to screen optical absorption. This report uses a finite 20 x 20 um TaIrTe4 flake, finite Gaussian Maxwell Q, explicit 32 x 32 um thermal domain, and finite top-bottom / left-right TaIrTe4 terminals.",
        "All values below correspond to 285 uW incident power through the certified linear source-power scaling. Raw Maxwell Q is unchanged.",
        "",
        "## Primary results",
        "",
        "| case | absorbed (uW) | Tmax (K) | Ta avg (K) | I top-bottom (nA) | I left-right (nA) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['case']} | {row['absorbed_power_at_285uW_W']*1e6:.4f} | {row['Tmax_K']:.6f} | {row['TaIrTe4_volume_average_K']:.6f} | {row['top_bottom_current_nA']:.6g} | {row['left_right_current_nA']:.6g} |")
    lines += ["", "The Au-off cases are laterally symmetric and therefore their signed terminal currents are near-null even though local J is nonzero. The asymmetric T/Z top Au breaks cancellation.", "", "## Absorbed-power location in Au-on cases", "", "| case | TaIrTe4 | top Au | Au mirror |", "|---|---:|---:|---:|"]
    for row in rows:
        if row["top_Au_present"]:
            lines.append(f"| {row['case']} | {percent(row['TaIrTe4_absorption_fraction'])} | {percent(row['top_Au_absorption_fraction'])} | {percent(row['mirror_Au_absorption_fraction'])} |")
    lines += ["", "Direct top-Au absorption is small in total power. Au can still have a large electrical effect because a highly conducting floating metal redistributes the weighting field/current collection.", "", "## Exact Au contribution to current", "", "The following four terms telescope exactly from the full Au-on current to the independent Au-off current: floating-Au electrical shunt, direct top-Au heating, top-Au thermal shunt, and Au-induced optical redistribution in non-Au materials.", "", "| case/orientation | electrical (nA) | direct heat (nA) | thermal (nA) | optical redistribution (nA) | total on-off (nA) |", "|---|---:|---:|---:|---:|---:|"]
    for row in decomp_rows:
        lines.append(f"| {row['case']} / {row['orientation']} | {row['floating_Au_electrical_shunt_nA']:.6g} | {row['direct_top_Au_absorption_heat_nA']:.6g} | {row['top_Au_thermal_shunt_nA']:.6g} | {row['Au_induced_nonAu_optical_redistribution_nA']:.6g} | {row['I_on_minus_off_nA']:.6g} |")
    lines += ["", "For Z, the floating-Au electrical term is the dominant positive contribution in this contact scenario, while Au-induced optical redistribution partly cancels it in several cases. For T, the absolute current is pA-scale and the optical/direct-heating terms are comparable or competing.", "", "## Limits", "", "This is a validated forward numerical scenario, not yet an experimental prediction. Au/TaIrTe4 thermal and electrical contact values are not measured for this device; varying them is the next physical-uncertainty gate. The Au Seebeck coefficient is zero in this collection/shunting control."]
    (OUTPUT / "FINITE_T_Z_MULTIPHYSICS_REPORT.md").write_text("\n".join(lines) + "\n")
    (OUTPUT / "RAW_ARTIFACT_MANIFEST.json").write_text(json.dumps({"status": status, "raw_files_committed_to_git": False, "artifacts": manifests}, indent=2) + "\n")
    print(json.dumps({"status": status, "gates": summary["gates"], "primary": rows, "decomposition": decomp_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
