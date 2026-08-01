#!/usr/bin/env python3
"""Summarize Device-A three-position Maxwell/analytic terminal currents."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PAPER_RATIO = 0.836590
PAPER_RATIO_UNCERTAINTY = 0.008526


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_artifact(path: Path, role: str) -> dict[str, object]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "committed_to_git": False,
    }


def strict_mask(mask: np.ndarray) -> np.ndarray:
    output = np.zeros_like(mask, dtype=bool)
    output[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-contract", type=Path, required=True)
    parser.add_argument("--geometry-contract", type=Path, required=True)
    parser.add_argument("--optical-summary", type=Path, required=True)
    parser.add_argument("--analytic-summary", type=Path, required=True)
    parser.add_argument("--thermal-batch-index", type=Path, required=True)
    parser.add_argument("--s0-a-isolated", type=Path, required=True)
    parser.add_argument("--s0-b-isolated", type=Path, required=True)
    parser.add_argument("--s0-a-perfect", type=Path, required=True)
    parser.add_argument("--s0-b-perfect", type=Path, required=True)
    parser.add_argument("--contact-diagnostic-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_thermal(path: Path, position: str, polarization: str, scenario: str, s_um: float) -> dict[str, Any]:
    summary_path = path / "summary.json"
    fields_path = path / "thermal_pte_fields.npz"
    summary = json.loads(summary_path.read_text())
    return {
        "position_label": position,
        "signed_s_from_edge_um": s_um,
        "polarization": polarization,
        "scenario": scenario,
        "status": summary["status"],
        "P_Q_thermal_W": float(summary["mapping"]["P_Q_target_W"]),
        "Tmax_rise_K": float(summary["thermal"]["Tmax_rise_K"]),
        "flake_average_rise_K": float(summary["thermal"]["TaIrTe4_volume_average_rise_K"]),
        "PTE_current_A": float(summary["PTE_current_A_at_285uW_incident"]),
        "PTE_current_nA": float(summary["PTE_current_A_at_285uW_incident"]) * 1e9,
        "mapping_relative_power_error": float(summary["mapping"]["mapping_relative_power_error"]),
        "linear_residual_relative": float(summary["thermal"]["linear_residual_relative"]),
        "energy_balance_relative_error": float(summary["thermal"]["energy_balance_relative_error"]),
        "weighting_gate_passed": bool(summary["weighting_gate_passed"]),
        "summary_path": str(summary_path.resolve()),
        "fields_path": str(fields_path.resolve()),
        "thermal_domain_um": float(summary["geometry"]["thermal_domain_um"]),
        "Si_depth_um": float(summary["geometry"]["si_depth_um"]),
        "core_step_nm": float(summary["geometry"]["core_step_nm"]),
        "flake_dz_nm": float(summary["geometry"]["flake_dz_nm"]),
    }


def load_maps(fields_path: str, normal: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(fields_path) as raw:
        x_edges = np.asarray(raw["x_edges_m"], float) * 1e6
        y_edges = np.asarray(raw["y_edges_m"], float) * 1e6
        dz = np.diff(np.asarray(raw["z_edges_m"], float))
        flake = np.any(raw["flake_mask"], axis=2)
        strict = strict_mask(flake)
        grad_b = np.asarray(raw["grad_T_x_K_m"], float)
        grad_a = np.asarray(raw["grad_T_y_K_m"], float)
        return {
            "x_edges_um": x_edges,
            "y_edges_um": y_edges,
            "flake": flake,
            "strict": strict,
            "q_areal": np.sum(np.asarray(raw["Q_W_m3"], float) * dz[None, None, :], axis=2),
            "temperature": np.asarray(raw["temperature_flake_average_K"], float),
            "grad_a": grad_a,
            "grad_b": grad_b,
            "grad_n": normal[0] * grad_b + normal[1] * grad_a,
            "grad_magnitude": np.hypot(grad_a, grad_b),
            "integrand": np.asarray(raw["shockley_ramo_integrand_A_m2"], float),
        }


def plot_fields(rows: list[dict[str, Any]], normal: np.ndarray, scenario: str, output: Path) -> None:
    selected = sorted(
        (row for row in rows if row["scenario"] == scenario),
        key=lambda row: (row["signed_s_from_edge_um"], row["polarization"]),
    )
    maps = [load_maps(row["fields_path"], normal) for row in selected]
    columns = (
        ("q_areal", "depth-integrated Q", "magma", False, False),
        ("temperature", "flake-averaged ΔT", "magma", False, False),
        ("grad_a", r"$\partial_aT$", "coolwarm", True, True),
        ("grad_b", r"$\partial_bT$", "coolwarm", True, True),
        ("grad_n", r"$\partial_nT$", "coolwarm", True, True),
        ("grad_magnitude", r"$|\nabla_{ab}T|$", "magma", False, True),
        ("integrand", r"$\mathbf{J}_{PTE}\cdot\nabla\psi$ dz", "coolwarm", True, True),
    )
    limits = []
    for name, _, _, signed, strict in columns:
        samples = np.concatenate(
            [
                values[name][values["strict"] if strict else values["flake"]]
                for values in maps
            ]
        )
        if signed:
            bound = float(np.percentile(np.abs(samples), 99.5))
            limits.append((-bound, bound))
        else:
            limits.append((0.0, float(np.percentile(samples, 99.5))))
    figure, axes = plt.subplots(6, 7, figsize=(27, 23), constrained_layout=True)
    for row_index, (row, values) in enumerate(zip(selected, maps)):
        for column, ((name, title, cmap, _, use_strict), limit) in enumerate(zip(columns, limits)):
            mask = values["strict"] if use_strict else values["flake"]
            shown = np.where(mask, values[name], np.nan)
            handle = axes[row_index, column].pcolormesh(
                values["x_edges_um"],
                values["y_edges_um"],
                shown.T,
                shading="flat",
                cmap=cmap,
                vmin=limit[0],
                vmax=limit[1],
            )
            axes[row_index, column].set_aspect("equal")
            axes[row_index, column].set(xlabel="x=b (µm)", ylabel="y=a (µm)")
            if row_index == 0:
                axes[row_index, column].set_title(title)
            figure.colorbar(handle, ax=axes[row_index, column], shrink=0.72)
        axes[row_index, 0].text(
            0.02,
            0.96,
            f"s={row['signed_s_from_edge_um']:.1f} µm, E||{row['polarization']}",
            transform=axes[row_index, 0].transAxes,
            va="top",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 3},
        )
    figure.suptitle(
        f"Device-A Maxwell Q → explicit-3D thermal/PTE maps: {scenario}; strict 4-neighbour gradient mask"
    )
    figure.savefig(output, dpi=145)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scan = json.loads(args.scan_contract.read_text())
    s_by_internal = {
        "sminus1": float(scan["cases"][0]["signed_s_from_edge_um"]),
        "s0": float(scan["cases"][1]["signed_s_from_edge_um"]),
        "splus1": float(scan["cases"][2]["signed_s_from_edge_um"]),
    }
    batch = json.loads(args.thermal_batch_index.read_text())
    if batch["status"] != "COMPLETED_DEVICE_A_POSITION_MAXWELL_THERMAL_BATCH":
        raise RuntimeError("Maxwell thermal batch is not complete")
    rows = []
    for item in batch["completed_cases"]:
        scenario = (
            "isolated" if item["metal_thermalization"] == "isolated-lower-bound" else "perfect"
        )
        rows.append(
            load_thermal(
                Path(item["output_dir"]),
                item["position_label"],
                item["polarization"],
                scenario,
                s_by_internal[item["position_label"]],
            )
        )
    s0_inputs = {
        ("a", "isolated"): args.s0_a_isolated,
        ("b", "isolated"): args.s0_b_isolated,
        ("a", "perfect"): args.s0_a_perfect,
        ("b", "perfect"): args.s0_b_perfect,
    }
    for (polarization, scenario), path in s0_inputs.items():
        rows.append(load_thermal(path, "s0", polarization, scenario, s_by_internal["s0"]))
    rows.sort(key=lambda row: (row["scenario"], row["signed_s_from_edge_um"], row["polarization"]))
    numerical_gates = {
        "all_thermal_domains_match_immutable_s0_60um": all(
            row["thermal_domain_um"] == 60.0 for row in rows
        ),
        "all_mapping_error_lt_0p5percent": all(
            row["mapping_relative_power_error"] < 0.005 for row in rows
        ),
        "all_residual_lt_1e_8": all(row["linear_residual_relative"] < 1e-8 for row in rows),
        "all_energy_balance_lt_1percent": all(row["energy_balance_relative_error"] < 0.01 for row in rows),
        "all_weighting_gates_passed": all(row["weighting_gate_passed"] for row in rows),
        "all_cases_completed": all(str(row["status"]).startswith("COMPLETED") for row in rows),
    }
    ratios: dict[str, dict[str, Any]] = {}
    for scenario in ("isolated", "perfect"):
        ratios[scenario] = {}
        for position in ("sminus1", "s0", "splus1"):
            selected = {
                row["polarization"]: row
                for row in rows
                if row["scenario"] == scenario and row["position_label"] == position
            }
            ratio = abs(selected["a"]["PTE_current_A"]) / abs(selected["b"]["PTE_current_A"])
            ratios[scenario][position] = {
                "signed_s_from_edge_um": s_by_internal[position],
                "I_a_A": selected["a"]["PTE_current_A"],
                "I_b_A": selected["b"]["PTE_current_A"],
                "abs_Ia_over_abs_Ib": ratio,
            }
        ordered = [ratios[scenario][position] for position in ("sminus1", "s0", "splus1")]
        ratios[scenario]["current_change_per_1um"] = {
            polarization: [
                ordered[index + 1][f"I_{polarization}_A"] - ordered[index][f"I_{polarization}_A"]
                for index in range(2)
            ]
            for polarization in ("a", "b")
        }
    analytic = json.loads(args.analytic_summary.read_text())
    if float(analytic["thermal_contract"]["lateral_domain_m"]) != 60.0e-6:
        raise RuntimeError("analytic comparison is not the promoted 60-um contract")
    analytic_alias = {"sminus1": "s-1um", "s0": "s0", "splus1": "s+1um"}
    analytic_ratios = {
        position: float(
            analytic["cases"][f"{analytic_alias[position]}_a"]["abs_Ia_over_abs_Ib"]
        )
        for position in ("sminus1", "s0", "splus1")
    }
    all_maxwell_above_one = all(
        ratios[scenario][position]["abs_Ia_over_abs_Ib"] > 1.0
        for scenario in ("isolated", "perfect")
        for position in ("sminus1", "s0", "splus1")
    )
    any_crossing = any(
        min(ratios[scenario][position]["abs_Ia_over_abs_Ib"] for position in ("sminus1", "s0", "splus1")) < 1.0
        < max(ratios[scenario][position]["abs_Ia_over_abs_Ib"] for position in ("sminus1", "s0", "splus1"))
        for scenario in ("isolated", "perfect")
    )
    analytic_maxwell_opposite = all(value < 1.0 for value in analytic_ratios.values()) and all_maxwell_above_one
    if not all(numerical_gates.values()):
        primary_status = "NUMERICAL_GATE_FAILED"
    elif any_crossing:
        primary_status = "POSITION_SENSITIVE_REVERSAL"
    elif all_maxwell_above_one:
        primary_status = "ROBUST_MAXWELL_REVERSAL"
    elif analytic_maxwell_opposite:
        primary_status = "ANALYTIC_MAXWELL_SOURCE_MISMATCH"
    else:
        primary_status = "POSITION_SENSITIVITY_INCONCLUSIVE"
    optical = json.loads(args.optical_summary.read_text())
    contact = None
    if args.contact_diagnostic_summary is not None:
        contact = json.loads(args.contact_diagnostic_summary.read_text())
        if contact.get("contact_optical_scattering_dominant", False):
            primary_status = "CONTACT_OPTICAL_SCATTERING_DOMINANT"
    geometry = json.loads(args.geometry_contract.read_text())
    normal = np.asarray(geometry["off_axis_edge_unit_inward_normal_code"], float)
    normal /= np.linalg.norm(normal)
    plot_fields(rows, normal, "isolated", args.output_dir / "DEVICE_A_POSITION_FIELDS_ISOLATED.png")
    plot_fields(rows, normal, "perfect", args.output_dir / "DEVICE_A_POSITION_FIELDS_PERFECT.png")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for scenario, style in (("isolated", "-"), ("perfect", "--")):
        points = [ratios[scenario][position] for position in ("sminus1", "s0", "splus1")]
        s = [point["signed_s_from_edge_um"] for point in points]
        axes[0].plot(s, [point["I_a_A"] * 1e9 for point in points], "o" + style, label=f"E||a, {scenario}")
        axes[0].plot(s, [point["I_b_A"] * 1e9 for point in points], "s" + style, label=f"E||b, {scenario}")
        axes[1].plot(s, [point["abs_Ia_over_abs_Ib"] for point in points], "o" + style, label=scenario)
    axes[1].axhspan(PAPER_RATIO - PAPER_RATIO_UNCERTAINTY, PAPER_RATIO + PAPER_RATIO_UNCERTAINTY, color="tab:orange", alpha=0.25, label="digitized paper band")
    axes[1].axhline(1.0, color="black", linestyle=":")
    for polarization, color in (("a", "tab:blue"), ("b", "tab:orange")):
        maxwell = [ratios["isolated"][position][f"I_{polarization}_A"] * 1e9 for position in ("sminus1", "s0", "splus1")]
        analytic_values = [analytic["cases"][f"{analytic_alias[position]}_{polarization}"]["PTE_terminal_current_nA"] for position in ("sminus1", "s0", "splus1")]
        s = [s_by_internal[position] for position in ("sminus1", "s0", "splus1")]
        axes[2].plot(s, maxwell, "o-", color=color, label=f"Maxwell E||{polarization}")
        axes[2].plot(s, analytic_values, "s--", color=color, alpha=0.75, label=f"analytic E||{polarization}")
    axes[0].set_ylabel("signed terminal current (nA)")
    axes[1].set_ylabel(r"$|I_a|/|I_b|$")
    axes[2].set_ylabel("signed terminal current (nA)")
    for axis in axes:
        axis.set_xlabel("signed s from digitized edge (µm)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(args.output_dir / "DEVICE_A_POSITION_CURRENT_AND_RATIO.png", dpi=180)
    plt.close(figure)
    artifacts = []
    for row in rows:
        artifacts.append(raw_artifact(Path(row["fields_path"]), f"thermal fields {row['position_label']} E||{row['polarization']} {row['scenario']}"))
        artifacts.append(raw_artifact(Path(row["summary_path"]), f"thermal summary {row['position_label']} E||{row['polarization']} {row['scenario']}"))
    manifest = {
        "status": "RECORDED_EXTERNAL_RAW_ARTIFACTS_NOT_COMMITTED",
        "artifacts": artifacts,
        "optical_manifest": str((args.output_dir / "RAW_ARTIFACT_MANIFEST_POSITION_OPTICAL.json").resolve()),
        "analytic_manifest": str((args.output_dir / "RAW_ARTIFACT_MANIFEST_ANALYTIC_60um.json").resolve()),
        "generation_command": " ".join(sys.argv),
    }
    (args.output_dir / "RAW_ARTIFACT_MANIFEST_POSITION_THERMAL.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {
        "status": primary_status,
        "primary_classification_precedence": [
            "NUMERICAL_GATE_FAILED",
            "CONTACT_OPTICAL_SCATTERING_DOMINANT",
            "POSITION_SENSITIVE_REVERSAL",
            "ROBUST_MAXWELL_REVERSAL",
            "ANALYTIC_MAXWELL_SOURCE_MISMATCH",
        ],
        "secondary_diagnostics": {
            "analytic_maxwell_source_mismatch": analytic_maxwell_opposite,
            "all_three_positions_Maxwell_ratio_gt_1_both_metal_bounds": all_maxwell_above_one,
            "ratio_crosses_one_within_plus_minus_1um": any_crossing,
        },
        "numerical_gates": numerical_gates,
        "optical_status": optical["status"],
        "thermal_contract": batch["thermal_contract"],
        "Maxwell_ratios": ratios,
        "analytic_ratios": analytic_ratios,
        "paper_digitized_ratio": PAPER_RATIO,
        "paper_digitized_uncertainty": PAPER_RATIO_UNCERTAINTY,
        "contact_diagnostic": contact,
        "cases": rows,
        "terminal_current_contract": "full flake-cell volume integral; cell volume multiplied exactly once",
        "no_empirical_current_normalization_or_polarization_rescaling": True,
        "no_adjoint_adfd_or_optimization": True,
    }
    (args.output_dir / "device_a_position_sensitivity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    csv_rows = [{key: value for key, value in row.items() if key not in ("summary_path", "fields_path")} for row in rows]
    with (args.output_dir / "device_a_position_sensitivity_cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
    table = []
    for position in ("sminus1", "s0", "splus1"):
        iso = ratios["isolated"][position]
        perfect = ratios["perfect"][position]
        table.append(
            f"| {iso['signed_s_from_edge_um']:.1f} | {iso['I_a_A']*1e9:.6g} | {iso['I_b_A']*1e9:.6g} | {iso['abs_Ia_over_abs_Ib']:.6f} | {perfect['abs_Ia_over_abs_Ib']:.6f} | {analytic_ratios[position]:.6f} |"
        )
    report = f"""# Device-A beam-position terminal-current sensitivity

Status: `{primary_status}`

All promoted comparisons use the immutable s0 thermal contract: 60-um
lateral domain, 20-um Si depth, 100-nm core cells, and 10-nm TaIrTe4 cells.
The earlier 48-um analytic run is preserved but excluded.

| signed s (um) | Maxwell Ia isolated (nA) | Maxwell Ib isolated (nA) | Maxwell ratio isolated | Maxwell ratio perfect | analytic ratio |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

The paper digitization gives `{PAPER_RATIO:.6f} ± {PAPER_RATIO_UNCERTAINTY:.6f}`.
The analytic source already inputs the larger b-polarized TMM absorption and
is a control, not a paper reproduction. No empirical current normalization,
polarization matching, Q clipping, smoothing, gain, or rescaling was used.

All displayed gradient and local-integrand maps use the strict four-neighbour
mask requested by the user: a cell is hidden if any of ±x or ±y lies outside
the TaIrTe4 mask. Temperature and Q maps retain the full physical flake.
"""
    (args.output_dir / "DEVICE_A_POSITION_SENSITIVITY_REPORT.md").write_text(report)
    print(json.dumps(summary, indent=2))
    return 0 if all(numerical_gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
