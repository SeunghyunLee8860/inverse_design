#!/usr/bin/env python3
"""Compare the Kitamura-substrate Device-A scenario with paper targets."""

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

from photothermal_pte.validation.paper_ir_sanity import (
    run_lumerical_device_a_ir_q as optical_runner,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def artifact_record(path: Path, role: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "committed_to_git": False,
    }


def optical_metrics(case_dir: Path, incident_dir: Path) -> dict[str, Any]:
    case = load_json(case_dir / "case_result.json")
    run = case["run_result"]
    acceptance = run.get("acceptance", {})
    substrate = run["substrate_epsilon_readback"]
    material = run["material_resolved_absorption"]
    incident_result = incident_dir / "case_result.json"
    incident_npz = incident_dir / "incident_reference.npz"
    with np.load(incident_npz, allow_pickle=False) as raw:
        beam_fit = optical_runner.fit_elliptical_gaussian(
            np.asarray(raw["x_m"], float),
            np.asarray(raw["y_m"], float),
            np.asarray(raw["downward_intensity_W_m2"], float),
        )
    return {
        "path": str(case_dir.resolve()),
        "status": case["status"],
        "P_Q_full_control_volume_W_at_1_W_m2": run["P_Q_W"],
        "P_Q_TaIrTe4_exact_support_W_at_1_W_m2": material[
            "P_Q_TaIrTe4_exact_support_W"
        ],
        "P_Q_SiO2_exact_support_W_at_1_W_m2": material[
            "P_Q_SiO2_exact_support_W"
        ],
        "P_six_face_W_at_1_W_m2": run["P_six_face_W"],
        "six_face_relative_closure": run["six_face_relative_closure"],
        "auto_shutoff": run["auto_shutoff"],
        "component_power_W": run["component_power_W"],
        "incident_reference": {
            "case_result_path": str(incident_result.resolve()),
            "field_NPZ_path": str(incident_npz.resolve()),
            "beam_fit_at_target_plane": beam_fit,
            "requested_physical_waist_m": case["waist_um"] * 1.0e-6,
            "evidence_scope": (
                "downward E/H decomposition at z=+50 nm in the matching "
                "empty layered stack; it includes the substrate reflection "
                "contract and is not called a pure free-space incident waist"
            ),
        },
        "substrate_epsilon_readback": substrate,
        "all_acceptance_gates": bool(acceptance) and all(
            bool(value) for value in acceptance.values()
        ),
        "acceptance": acceptance,
    }


def thermal_metrics(case_dir: Path) -> dict[str, Any]:
    summary = load_json(case_dir / "summary.json")
    current = summary["PTE_current_A_at_requested_incident_power"]
    return {
        "path": str(case_dir.resolve()),
        "status": summary["status"],
        "thermal": summary["thermal"],
        "weighting": summary["weighting"],
        "weighting_gate_passed": summary["weighting_gate_passed"],
        "two_terminal_resistance_audit": summary[
            "two_terminal_resistance_audit"
        ],
        "absolute_current_certification": summary[
            "absolute_current_certification"
        ],
        "requested_incident_power_W": summary["requested_incident_power_W"],
        "PTE_current_A": current,
        "PTE_current_pA": None if current is None else 1.0e12 * current,
        "mapping": {
            "P_Q_source_W": summary["mapping"]["P_Q_source_W"],
            "P_Q_target_W": summary["mapping"]["P_Q_target_W"],
            "mapping_relative_power_error": summary["mapping"][
                "mapping_relative_power_error"
            ],
        },
    }


def plot_current(summary: dict[str, Any], path: Path) -> None:
    simulated = np.asarray(
        [
            abs(summary["thermal"][axis]["PTE_current_pA"])
            for axis in ("a", "b")
        ]
    )
    target = np.asarray([122.0, 143.0])
    absorbed_uW = np.asarray(
        [
            summary["thermal"][axis]["mapping"]["P_Q_source_W"] * 1.0e6
            for axis in ("a", "b")
        ]
    )
    tmax = np.asarray(
        [
            summary["thermal"][axis]["thermal"]["Tmax_rise_K"]
            for axis in ("a", "b")
        ]
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.2))
    x = np.arange(2)
    axes[0].bar(x, absorbed_uW, color=["tab:blue", "tab:orange"])
    axes[0].set_xticks(x, [r"$E\parallel a$", r"$E\parallel b$"])
    axes[0].set_ylabel(r"mapped TaIrTe$_4$ power ($\mu$W)")
    axes[0].set_title("Maxwell Q at 284.40 µW")
    axes[1].bar(x, tmax, color=["tab:blue", "tab:orange"])
    axes[1].set_xticks(x, [r"$E\parallel a$", r"$E\parallel b$"])
    axes[1].set_ylabel(r"maximum $\Delta T$ (K)")
    axes[1].set_title("Paper-reduced thermal solve")
    inset = axes[1].inset_axes([0.54, 0.50, 0.42, 0.42])
    ratios = [
        summary["comparison"]["simulated_abs_Ia_over_abs_Ib"],
        summary["experimental_target"]["abs_Ia_over_abs_Ib"],
    ]
    inset.bar([0, 1], ratios, color=["tab:blue", "tab:red"])
    inset.axhline(1.0, color="black", linestyle="--", linewidth=1)
    inset.set_xticks([0, 1], ["sim", "paper"], fontsize=8)
    inset.set_ylabel(r"$|I_a|/|I_b|$", fontsize=8)
    axes[2].bar(x - 0.18, simulated, width=0.36, label="simulation")
    axes[2].bar(x + 0.18, target, width=0.36, label="paper visual estimate")
    axes[2].set_xticks(x, [r"$E\parallel a$", r"$E\parallel b$"])
    axes[2].set_ylabel("absolute current (pA)")
    axes[2].set_yscale("log")
    axes[2].legend(fontsize=8)
    axes[2].set_title("Current magnitude (no fitting)")
    fig.suptitle("Device-A 11-µm current comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def report(summary: dict[str, Any]) -> str:
    sim = summary["comparison"]
    target = summary["experimental_target"]
    optical_a = summary["optical"]["a"]
    optical_b = summary["optical"]["b"]
    fit_a = optical_a["incident_reference"]["beam_fit_at_target_plane"]
    fit_b = optical_b["incident_reference"]["beam_fit_at_target_plane"]
    return f"""# Device-A 11-µm measured-current comparison

Status: `{summary['status']}`

This result uses Kitamura-2007 SiO2, Palik Si as an explicit unpublished
closure, the digitized Device-A geometry, exact 284.40-µW input power, and the
paper-reduced TaIrTe4 Robin thermal model.  It is not fitted to the measured
current or the 213-ohm resistance.

## Result

| Quantity | Simulation | Paper target |
|---|---:|---:|
| `|Ia|` | {sim['simulated_abs_Ia_pA']:.6g} pA | about 122 pA from Fig. 3I, with SI fit 110.6 pA at 3675 Hz |
| `|Ib|` | {sim['simulated_abs_Ib_pA']:.6g} pA | about 143 pA from Fig. 3I |
| `|Ia|/|Ib|` | {sim['simulated_abs_Ia_over_abs_Ib']:.6g} | {target['abs_Ia_over_abs_Ib']:.6g} ± {target['ratio_uncertainty']:.6g} |

## Optical and beam gates

| Quantity | E || a | E || b |
|---|---:|---:|
| full matched-volume `P_Q` at central 1 W/m2 | {optical_a['P_Q_full_control_volume_W_at_1_W_m2']:.9e} W | {optical_b['P_Q_full_control_volume_W_at_1_W_m2']:.9e} W |
| TaIrTe4-support `P_Q` at central 1 W/m2 | {optical_a['P_Q_TaIrTe4_exact_support_W_at_1_W_m2']:.9e} W | {optical_b['P_Q_TaIrTe4_exact_support_W_at_1_W_m2']:.9e} W |
| six-face closure | {100*optical_a['six_face_relative_closure']:.5f}% | {100*optical_b['six_face_relative_closure']:.5f}% |
| auto-shutoff | {optical_a['auto_shutoff']['final_value']:.4e} | {optical_b['auto_shutoff']['final_value']:.4e} |
| realized effective waist at target plane | {fit_a['waist_effective_geometric_mean_m']*1e6:.5f} µm | {fit_b['waist_effective_geometric_mean_m']*1e6:.5f} µm |
| Gaussian fit RMS / peak | {100*fit_a['fit_relative_RMS_over_peak']:.5f}% | {100*fit_b['fit_relative_RMS_over_peak']:.5f}% |

The requested physical waist is 8.75 µm.  The target-plane profile is read
from the matching empty SiO2/Si stack and is therefore evidence for the
realized downward field in that layered reference, not a claim of an
independently measured experimental waist.

The Figure-3I axis prints nA, but the Figure-3H/SI maps and the independent
SI frequency fit are in pA.  Both interpretations remain recorded; pA is the
physically consistent comparison.

## Certification limits

- All optical closure, auto-shutoff, source-mapping, thermal energy-balance,
  residual, and weighting-potential gates are reported independently.
- The polarization-ratio gate fails: the simulation gives
  `{sim['simulated_abs_Ia_over_abs_Ib']:.6g}`, versus the digitized paper
  value `{target['abs_Ia_over_abs_Ib']:.6g}`.  The trend is reversed and is
  not called a reproduction.
- The digitized geometry predicts a two-terminal resistance far from the
  measured 213 ohm.  Therefore absolute-current agreement is not certified
  and no conductivity/current rescaling is applied.
- The exact beam definition, objective transmission, CAD/contact resistance,
  and scan coordinates are unpublished.  The result is a named paper-like
  scenario, not a unique reconstruction of the authors' hidden model.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-json", type=Path, required=True)
    parser.add_argument("--optical-a", type=Path, required=True)
    parser.add_argument("--optical-b", type=Path, required=True)
    parser.add_argument("--empty-a", type=Path, required=True)
    parser.add_argument("--empty-b", type=Path, required=True)
    parser.add_argument("--thermal-a", type=Path, required=True)
    parser.add_argument("--thermal-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = load_json(args.contract_json)
    optical_cases = {
        "a": optical_metrics(args.optical_a, args.empty_a),
        "b": optical_metrics(args.optical_b, args.empty_b),
    }
    thermal_cases = {
        "a": thermal_metrics(args.thermal_a),
        "b": thermal_metrics(args.thermal_b),
    }
    ia_pA = abs(float(thermal_cases["a"]["PTE_current_pA"]))
    ib_pA = abs(float(thermal_cases["b"]["PTE_current_pA"]))
    ratio = ia_pA / ib_pA
    target = contract["experimental_targets"]["figure_3J_digitized_ratio"]
    numerical_pass = all(
        item["all_acceptance_gates"] for item in optical_cases.values()
    ) and all(
        item["mapping"]["mapping_relative_power_error"] < 0.005
        and item["thermal"]["energy_balance_relative_error"] < 0.01
        and item["thermal"]["linear_residual_relative"] < 1.0e-8
        and item["weighting_gate_passed"]
        for item in thermal_cases.values()
    )
    ratio_relative_difference = abs(ratio - target["value"]) / target["value"]
    absolute_certified = all(
        item["absolute_current_certification"]
        == "GEOMETRY_RESISTANCE_GATE_PASSED"
        for item in thermal_cases.values()
    )
    if not numerical_pass:
        status = "FAILED_DEVICE_A_MEASURED_REPRODUCTION_NUMERICAL_GATE"
    elif ratio_relative_difference >= 0.10:
        status = "FAILED_DEVICE_A_PAPER_LIKE_CURRENT_POLARIZATION_RATIO"
    elif absolute_certified:
        status = "VALIDATED_DEVICE_A_MEASURED_CURRENT_SCENARIO"
    else:
        status = "COMPLETED_DEVICE_A_PAPER_LIKE_CURRENT_ABSOLUTE_CERTIFICATION_BLOCKED"
    summary = {
        "status": status,
        "contract": contract,
        "optical": optical_cases,
        "thermal": thermal_cases,
        "experimental_target": {
            "abs_Ia_over_abs_Ib": target["value"],
            "ratio_uncertainty": target["uncertainty"],
            "SI_Ea_fitted_current_at_3675Hz_pA": contract[
                "experimental_targets"
            ]["SI_Figure_S5_off_axis_Ea_frequency_fit"][
                "fitted_current_at_measurement_frequency_pA"
            ],
            "Figure3I_visual_Ia_Ib_pA_assuming_unit_typo": [122.0, 143.0],
        },
        "comparison": {
            "simulated_abs_Ia_pA": ia_pA,
            "simulated_abs_Ib_pA": ib_pA,
            "simulated_abs_Ia_over_abs_Ib": ratio,
            "ratio_relative_difference_vs_paper": ratio_relative_difference,
            "absolute_current_not_rescaled": True,
        },
        "gates": {
            "numerical_pass": numerical_pass,
            "polarization_ratio_within_10pct": ratio_relative_difference < 0.10,
            "absolute_current_geometry_resistance_gate": absolute_certified,
        },
        "no_Q_or_current_rescaling": True,
        "no_adjoint": True,
        "no_optimization": True,
    }
    json_path = args.output_dir / "device_a_measured_reproduction_summary.json"
    report_path = args.output_dir / "DEVICE_A_MEASURED_REPRODUCTION_REPORT.md"
    plot_path = args.output_dir / "device_a_current_comparison.png"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    report_path.write_text(report(summary))
    plot_current(summary, plot_path)
    csv_path = args.output_dir / "device_a_measured_reproduction_cases.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "polarization",
                "P_Q_full_W_at_1_W_m2",
                "P_Q_TaIrTe4_W_at_1_W_m2",
                "six_face_closure",
                "auto_shutoff",
                "realized_effective_waist_um",
                "mapped_Q_W_at_284p40uW",
                "Tmax_rise_K",
                "PTE_current_A",
                "PTE_current_pA",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for axis in ("a", "b"):
            optical = optical_cases[axis]
            thermal = thermal_cases[axis]
            writer.writerow(
                {
                    "polarization": axis,
                    "P_Q_full_W_at_1_W_m2": optical[
                        "P_Q_full_control_volume_W_at_1_W_m2"
                    ],
                    "P_Q_TaIrTe4_W_at_1_W_m2": optical[
                        "P_Q_TaIrTe4_exact_support_W_at_1_W_m2"
                    ],
                    "six_face_closure": optical[
                        "six_face_relative_closure"
                    ],
                    "auto_shutoff": optical["auto_shutoff"]["final_value"],
                    "realized_effective_waist_um": optical[
                        "incident_reference"
                    ]["beam_fit_at_target_plane"][
                        "waist_effective_geometric_mean_m"
                    ]
                    * 1.0e6,
                    "mapped_Q_W_at_284p40uW": thermal["mapping"][
                        "P_Q_source_W"
                    ],
                    "Tmax_rise_K": thermal["thermal"]["Tmax_rise_K"],
                    "PTE_current_A": thermal["PTE_current_A"],
                    "PTE_current_pA": thermal["PTE_current_pA"],
                }
            )
    raw_records = []
    for axis, directory in (("a", args.optical_a), ("b", args.optical_b)):
        for filename, role in (
            ("finite_q_on_artifact.npz", f"raw_Maxwell_Q_{axis}"),
            ("finite_2um_optical_q.fsp", f"Lumerical_project_{axis}"),
            ("case_result.json", f"optical_case_result_{axis}"),
        ):
            raw_records.append(artifact_record(directory / filename, role))
    manifest_path = args.output_dir / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": status,
                "raw_artifacts": raw_records,
                "raw_NPZ_and_FSP_are_not_committed": True,
                "no_Q_clipping_smoothing_gain_or_rescaling": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"status": status, "summary": str(json_path)}, indent=2))
    return 0 if numerical_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
