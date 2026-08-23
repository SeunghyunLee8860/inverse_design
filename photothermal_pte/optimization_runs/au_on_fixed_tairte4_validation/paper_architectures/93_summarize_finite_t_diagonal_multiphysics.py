#!/usr/bin/env python3
"""Publish finite T11x15 Ea/Eb/+45/-45 Maxwell-to-current comparison."""

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
RAW_OPTICAL = Path("/home/seunghyun/tairte4/raw_artifacts/finite_T_Z_array_Q")
AXIAL_MAPPING = HERE / "results_finite_T_Z_array_material_Q_mapping" / "FINITE_T_Z_ARRAY_MATERIAL_Q_MAPPING_SUMMARY.json"
DIAGONAL_MAPPING = HERE / "results_finite_T_diagonal_material_Q_mapping" / "FINITE_T_DIAGONAL_MATERIAL_Q_MAPPING_SUMMARY.json"
AXIAL_THERMAL = HERE / "results_finite_T_Z_array_thermal_electrical"
DIAGONAL_THERMAL = HERE / "results_finite_T_diagonal_thermal_electrical"
OUTPUT = HERE / "results_finite_T_diagonal_multiphysics_summary"
CASES = (
    "T11x15_Ea_Au_on",
    "T11x15_Eb_Au_on",
    "T11x15_linear_plus_45_Au_on",
    "T11x15_linear_minus_45_Au_on",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def optical_result(case: str) -> tuple[Path, dict]:
    path = next((RAW_OPTICAL / case).glob("FINITE_*_Q.json"))
    return path, json.loads(path.read_text())


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    axial_mapping = json.loads(AXIAL_MAPPING.read_text())["cases"]
    diagonal_mapping = json.loads(DIAGONAL_MAPPING.read_text())["cases"]
    rows = []
    manifest = []
    for case in CASES:
        diagonal = "linear_" in case
        optical_path, optical = optical_result(case)
        mapping = diagonal_mapping[case] if diagonal else axial_mapping[case]
        thermal_root = DIAGONAL_THERMAL if diagonal else AXIAL_THERMAL
        thermal_path = thermal_root / case / f"{case}_THERMAL_ELECTRICAL_SUMMARY.json"
        thermal = json.loads(thermal_path.read_text())
        if not str(optical["status"]).startswith("VALIDATED"):
            raise RuntimeError(f"unvalidated optical case: {case}")
        if not str(thermal["status"]).startswith("VALIDATED"):
            raise RuntimeError(f"unvalidated thermal/electrical case: {case}")
        material = mapping["mapped_power_by_material_W"]
        total = mapping["mapped_total_power_W"]
        row = {
            "case": case,
            "polarization": optical["polarization"],
            "polarization_angle_deg": optical.get("polarization_angle_deg"),
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
            "Tmax_K": thermal["thermal"]["Tmax_K"],
            "TaIrTe4_volume_average_K": thermal["thermal"]["TaIrTe4_volume_average_K"],
            "top_bottom_current_nA": thermal["electrical"]["top_bottom"]["high_terminal_current_A"] * 1e9,
            "left_right_current_nA": thermal["electrical"]["left_right"]["high_terminal_current_A"] * 1e9,
            "thermal_energy_balance_relative": thermal["thermal"]["energy_balance_relative"],
            "thermal_residual_relative": thermal["thermal"]["linear_residual_relative"],
        }
        rows.append(row)
        manifest.append(
            {
                "case": case,
                "optical_summary": {
                    "path": str(optical_path),
                    "bytes": optical_path.stat().st_size,
                    "sha256": sha256(optical_path),
                },
                "raw_optical": optical["raw_artifacts"],
                "mapped_Q": mapping["raw_mapped_artifact"],
                "thermal_electrical": thermal["raw_artifact"],
            }
        )

    csv_path = OUTPUT / "finite_T_diagonal_multiphysics_cases.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    labels = ["E||a", "E||b", "+45 deg", "-45 deg"]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    axes[0, 0].bar(x, [row["absorbed_power_at_285uW_W"] * 1e6 for row in rows], color="#d39c00")
    axes[0, 0].set_ylabel("absorbed power at 285 uW incident (uW)")
    axes[0, 1].bar(x, [row["Tmax_K"] for row in rows], color="#d95f02", label="Tmax")
    axes[0, 1].bar(x, [row["TaIrTe4_volume_average_K"] for row in rows], color="#7570b3", alpha=.8, label="TaIrTe4 avg")
    axes[0, 1].set_ylabel("temperature rise (K)"); axes[0, 1].legend()
    axes[1, 0].bar(x, [row["top_bottom_current_nA"] for row in rows], color="#1b9e77")
    axes[1, 0].axhline(0, color="black", linewidth=.8); axes[1, 0].set_ylabel("top-bottom signed current (nA)")
    axes[1, 1].bar(x, [row["left_right_current_nA"] for row in rows], color="#377eb8")
    axes[1, 1].axhline(0, color="black", linewidth=.8); axes[1, 1].set_ylabel("left-right signed current (nA)")
    for ax in axes.flat:
        ax.set_xticks(x, labels); ax.grid(axis="y", alpha=.25)
    fig.suptitle("Finite T11x15: axial and diagonal polarization under unchanged physics")
    fig.savefig(OUTPUT / "finite_T11x15_four_linear_polarization_comparison.png", dpi=180)
    plt.close(fig)

    plus = rows[2]
    minus = rows[3]
    gates = {
        "all_optical_closure_lt_0p5pct": max(row["six_face_closure_relative"] for row in rows) < 0.005,
        "all_auto_shutoff_lt_1e-5": max(row["auto_shutoff"] for row in rows) < 1e-5,
        "all_thermal_energy_balance_lt_1pct": max(row["thermal_energy_balance_relative"] for row in rows) < 0.01,
        "all_thermal_residual_lt_1e-8": max(row["thermal_residual_relative"] for row in rows) < 1e-8,
    }
    status = (
        "VALIDATED_FINITE_T11X15_FOUR_LINEAR_POLARIZATION_MAXWELL_THERMAL_ELECTRICAL"
        if all(gates.values())
        else "FAILED_FINITE_T11X15_FOUR_LINEAR_POLARIZATION_GATE"
    )
    summary = {
        "status": status,
        "classification": (
            "finite nonperiodic T11x15 Gaussian Maxwell-to-explicit-3D-thermal/electrical "
            "forward comparison; no adjoint or optimization"
        ),
        "fixed_contract": {
            "wavelength_um": 4.75,
            "Gaussian_w0_um": 4.0,
            "finite_flake_um": [20.0, 20.0],
            "FDTD_lateral_um": [24.0, 24.0],
            "boundaries": "six PML for optical; unchanged physical thermal/electrical boundaries",
            "axis_mapping": "Lumerical x=b, y=a, z=c",
        },
        "gates": gates,
        "diagonal_contrast": {
            "plus45_minus_minus45_absorbed_power_W": plus["absorbed_power_at_285uW_W"] - minus["absorbed_power_at_285uW_W"],
            "plus45_minus_minus45_top_bottom_current_nA": plus["top_bottom_current_nA"] - minus["top_bottom_current_nA"],
            "plus45_minus_minus45_left_right_current_nA": plus["left_right_current_nA"] - minus["left_right_current_nA"],
        },
        "cases": rows,
    }
    (OUTPUT / "FINITE_T_DIAGONAL_MULTIPHYSICS_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Finite T11x15 axial/diagonal polarization Maxwell-to-current report",
        "",
        f"Status: `{status}`",
        "",
        "The electrodes and weighting solves are unchanged between polarizations. Only the coherent linear source angle changes.",
        "",
        "| polarization | absorbed power at 285 uW (uW) | Tmax (K) | top-bottom I (nA) | left-right I (nA) | closure |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, row in zip(labels, rows, strict=True):
        lines.append(
            f"| {label} | {row['absorbed_power_at_285uW_W']*1e6:.9g} | {row['Tmax_K']:.9g} | "
            f"{row['top_bottom_current_nA']:.9g} | {row['left_right_current_nA']:.9g} | "
            f"{row['six_face_closure_relative']:.6%} |"
        )
    lines.extend(
        [
            "",
            "The +/-45 cases are independent coherent Maxwell solves, not arithmetic averages of Ea/Eb Q.",
            "Raw NPZ/FSP remain outside Git; no clipping, smoothing, gain, global rescaling, or tiling was used.",
        ]
    )
    (OUTPUT / "FINITE_T_DIAGONAL_MULTIPHYSICS_REPORT.md").write_text("\n".join(lines) + "\n")
    (OUTPUT / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps({"status": status, "raw_files_committed_to_git": False, "artifacts": manifest}, indent=2) + "\n"
    )
    print(json.dumps({"status": status, "gates": gates, "diagonal_contrast": summary["diagonal_contrast"]}, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
