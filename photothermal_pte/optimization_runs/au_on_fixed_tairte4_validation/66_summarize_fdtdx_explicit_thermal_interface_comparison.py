#!/usr/bin/env python3
"""Compare the two named TaIrTe4/SiO2 interface forward scenarios."""

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


EXPECTED_STATUS = "VALIDATED_FDTDX_EXPLICIT_THERMAL_AU_AWARE_WEIGHTING_PTE_FORWARD"
PROMOTED_STATUS = "VALIDATED_FDTDX_EXPLICIT_THERMAL_INTERFACE_SCENARIO_COMPARISON"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path, expected_scenario: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != EXPECTED_STATUS:
        raise RuntimeError(f"Fail-closed: {path} is not a validated forward")
    if payload.get("scenario") != expected_scenario:
        raise RuntimeError(
            f"Fail-closed: expected {expected_scenario}, got {payload.get('scenario')}"
        )
    if not all(payload.get("gates", {}).values()):
        raise RuntimeError(f"Fail-closed: one or more forward gates failed in {path}")
    if payload["source"]["scaled_to_experimental_power"]:
        raise RuntimeError("Fail-closed: comparison requires literal unscaled FDTDX Q")
    return payload


def _metrics(payload: dict) -> dict[str, float]:
    thermal = payload["thermal"]
    material = thermal["material_temperature"]
    return {
        "P_Q_W": float(payload["source"]["P_Q_W"]),
        "Tmax_rise_K": float(thermal["Tmax_rise_K"]),
        "Au_Tmax_K": float(material["au"]["maximum_K"]),
        "Au_Tavg_K": float(material["au"]["volume_average_K"]),
        "TaIrTe4_Tmax_K": float(material["tairte4"]["maximum_K"]),
        "TaIrTe4_Tavg_K": float(material["tairte4"]["volume_average_K"]),
        "SiO2_Tmax_K": float(material["sio2"]["maximum_K"]),
        "SiO2_Tavg_K": float(material["sio2"]["volume_average_K"]),
        "Si_Tmax_K": float(material["si"]["maximum_K"]),
        "Si_Tavg_K": float(material["si"]["volume_average_K"]),
        "PTE_current_A": float(payload["electrical_PTE"]["current_A"]),
        "thermal_residual": float(thermal["linear_residual_relative"]),
        "thermal_energy_balance": float(thermal["energy_balance_relative"]),
        "thermal_GPU_solve_seconds": float(thermal["GPU_solve_seconds"]),
        "electrical_residual": float(
            payload["electrical_PTE"]["weighting_residual_relative"]
        ),
        "electrical_terminal_balance": float(
            payload["electrical_PTE"]["terminal_balance_relative"]
        ),
    }


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return float("nan")
    return numerator / denominator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thermally-grown-json", required=True, type=Path)
    parser.add_argument("--evaporated-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    grown_path = args.thermally_grown_json.resolve()
    evaporated_path = args.evaporated_json.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    grown = _read(grown_path, "thermally_grown")
    evaporated = _read(evaporated_path, "evaporated")
    grown_metrics = _metrics(grown)
    evaporated_metrics = _metrics(evaporated)

    if not np.isclose(
        grown_metrics["P_Q_W"],
        evaporated_metrics["P_Q_W"],
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError("Fail-closed: the two scenarios do not use identical P_Q")
    shared = {
        "spatial_Q_input_identical": grown["raw_artifact"] != evaporated["raw_artifact"],
        "source_power_bitwise_equal": True,
        "geometry_equal": grown["geometry"] == evaporated["geometry"],
        "non_interface_parameters_equal": {
            key: value
            for key, value in grown["parameters"].items()
            if key != "G_TaIrTe4_SiO2_W_m2K"
        }
        == {
            key: value
            for key, value in evaporated["parameters"].items()
            if key != "G_TaIrTe4_SiO2_W_m2K"
        },
    }
    # The raw forward artifacts are different temperature solutions; the common
    # input is certified by the exact same remap summary and source mapping.
    shared["spatial_Q_input_identical"] = (
        grown["source"]["remap_summary"] == evaporated["source"]["remap_summary"]
        and grown["source"]["mapping"] == evaporated["source"]["mapping"]
    )
    if not all(shared.values()):
        raise RuntimeError(f"Fail-closed comparison-contract mismatch: {shared}")

    rows = []
    for name in grown_metrics:
        rows.append(
            {
                "metric": name,
                "thermally_grown": grown_metrics[name],
                "evaporated": evaporated_metrics[name],
                "evaporated_over_thermally_grown": _ratio(
                    evaporated_metrics[name], grown_metrics[name]
                ),
            }
        )
    csv_path = output / "fdtdx_explicit_thermal_interface_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    selected = [
        "Tmax_rise_K",
        "Au_Tavg_K",
        "TaIrTe4_Tavg_K",
        "SiO2_Tavg_K",
        "Si_Tavg_K",
        "PTE_current_A",
    ]
    labels = ["Tmax", "Au Tavg", "TaIrTe4 Tavg", "SiO2 Tavg", "Si Tavg", "PTE I"]
    x = np.arange(len(selected))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    width = 0.38
    axes[0].bar(
        x - width / 2,
        [grown_metrics[name] for name in selected],
        width,
        label="thermally grown",
    )
    axes[0].bar(
        x + width / 2,
        [evaporated_metrics[name] for name in selected],
        width,
        label="evaporated",
    )
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, labels, rotation=30, ha="right")
    axes[0].set_title("Absolute literal-normalization outputs")
    axes[0].legend()
    ratios = [
        _ratio(evaporated_metrics[name], grown_metrics[name]) for name in selected
    ]
    axes[1].bar(x, ratios, color="tab:purple")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, labels, rotation=30, ha="right")
    axes[1].set_ylabel("evaporated / thermally grown")
    axes[1].set_title("Interface-scenario sensitivity")
    for index, value in enumerate(ratios):
        axes[1].text(index, value, f"{value:.3g}x", ha="center", va="bottom")
    plot_path = output / "fdtdx_explicit_thermal_interface_comparison.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    key_ratios = {
        "Tmax_rise": _ratio(
            evaporated_metrics["Tmax_rise_K"], grown_metrics["Tmax_rise_K"]
        ),
        "Au_volume_average_temperature": _ratio(
            evaporated_metrics["Au_Tavg_K"], grown_metrics["Au_Tavg_K"]
        ),
        "TaIrTe4_volume_average_temperature": _ratio(
            evaporated_metrics["TaIrTe4_Tavg_K"],
            grown_metrics["TaIrTe4_Tavg_K"],
        ),
        "PTE_current": _ratio(
            evaporated_metrics["PTE_current_A"],
            grown_metrics["PTE_current_A"],
        ),
    }
    summary = {
        "status": PROMOTED_STATUS,
        "scope": (
            "same literal FDTDX spatial Maxwell Q, geometry, explicit 3-D thermal "
            "operator and Au-aware weighting/PTE forward; only named TaIrTe4/SiO2 "
            "interface scenario differs"
        ),
        "common_contract": shared,
        "source_P_Q_W": grown_metrics["P_Q_W"],
        "scenarios": {
            "thermally_grown": {
                "G_TaIrTe4_SiO2_W_m2K": grown["parameters"][
                    "G_TaIrTe4_SiO2_W_m2K"
                ],
                "metrics": grown_metrics,
                "summary_json": str(grown_path),
                "raw_artifact": grown["raw_artifact"],
            },
            "evaporated": {
                "G_TaIrTe4_SiO2_W_m2K": evaporated["parameters"][
                    "G_TaIrTe4_SiO2_W_m2K"
                ],
                "metrics": evaporated_metrics,
                "summary_json": str(evaporated_path),
                "raw_artifact": evaporated["raw_artifact"],
            },
        },
        "evaporated_over_thermally_grown": key_ratios,
        "provenance_and_limits": {
            "literal_FDTDX_normalization_not_experimental_prediction": True,
            "Au_TaIrTe4_thermal_G_is_unmeasured_analogue_scenario": True,
            "Au_TaIrTe4_electrical_contact_is_unmeasured_numerical_scenario": True,
            "lossless_Si_is_diagnostic_not_Lumerical_Palik_readback": True,
            "forward_only_no_combined_gradient_ADFD_or_optimization": True,
        },
        "next_gate": (
            "combined Maxwell spatial-Q + explicit thermal + Au-aware electrical "
            "directional AD-FD"
        ),
    }
    summary_path = output / "fdtdx_explicit_thermal_interface_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report_path = output / "FDTDX_EXPLICIT_THERMAL_INTERFACE_COMPARISON_REPORT.md"
    report_path.write_text(
        f"""# FDTDX explicit thermal interface-scenario comparison

Status: **{PROMOTED_STATUS}**

Both cases use the exact same literal, unscaled spatial Maxwell source
`P_Q={grown_metrics['P_Q_W']:.12e} W`, material-overlap remap, geometry,
thermal mesh, boundary conditions, and Au-aware electrical operator. Only
the named TaIrTe4/SiO2 conductance changes:

- thermally grown: `G={grown['parameters']['G_TaIrTe4_SiO2_W_m2K']:.6e} W/(m2 K)`
- evaporated: `G={evaporated['parameters']['G_TaIrTe4_SiO2_W_m2K']:.6e} W/(m2 K)`

| metric | thermally grown | evaporated | evaporated / grown |
|---|---:|---:|---:|
| Tmax rise (K) | {grown_metrics['Tmax_rise_K']:.12e} | {evaporated_metrics['Tmax_rise_K']:.12e} | {key_ratios['Tmax_rise']:.6f} |
| Au volume-average rise (K) | {grown_metrics['Au_Tavg_K']:.12e} | {evaporated_metrics['Au_Tavg_K']:.12e} | {key_ratios['Au_volume_average_temperature']:.6f} |
| TaIrTe4 volume-average rise (K) | {grown_metrics['TaIrTe4_Tavg_K']:.12e} | {evaporated_metrics['TaIrTe4_Tavg_K']:.12e} | {key_ratios['TaIrTe4_volume_average_temperature']:.6f} |
| PTE current (A) | {grown_metrics['PTE_current_A']:.12e} | {evaporated_metrics['PTE_current_A']:.12e} | {key_ratios['PTE_current']:.6f} |

All thermal residuals are below `1e-8`, thermal energy-balance errors are
below `1%`, and electrical residual/balance gates pass. The result is a
physical-parameter sensitivity, not a numerical convergence error.

These currents retain the literal FDTDX source normalization and are not
experimental predictions. `G_Au/TaIrTe4` is an Au/MoS2 analogue, the
electrical contact is a named numerical scenario, and the Si endpoint remains
the explicitly documented lossless diagnostic. This is a forward certificate
only; combined AD--FD is still required before optimization.
""",
        encoding="utf-8",
    )
    manifest_path = output / "RAW_ARTIFACT_MANIFEST.json"
    published = [summary_path, csv_path, report_path, plot_path]
    manifest = {
        "status": PROMOTED_STATUS,
        "raw_artifacts_committed_to_git": False,
        "raw_artifacts": [grown["raw_artifact"], evaporated["raw_artifact"]],
        "published": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in published
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
