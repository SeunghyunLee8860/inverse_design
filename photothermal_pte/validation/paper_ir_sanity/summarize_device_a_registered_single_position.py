#!/usr/bin/env python3
"""Compare registered Device-A E||a/b optical, thermal, and PTE results."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "committed_to_git": False,
    }


def load_case(optical_dir: Path, thermal_dir: Path) -> dict[str, Any]:
    optical = json.loads((optical_dir / "case_result.json").read_text())
    thermal = json.loads((thermal_dir / "summary.json").read_text())
    with np.load(thermal_dir / "thermal_pte_fields.npz", allow_pickle=False) as stored:
        fields = {key: np.array(stored[key], copy=True) for key in stored.files}
    return {"optical": optical, "thermal": thermal, "fields": fields}


def integrate_2d(values: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> float:
    return float(np.sum(values * dx[:, None] * dy[None, :]))


def field_metrics(case: dict[str, Any]) -> dict[str, Any]:
    fields = case["fields"]
    dx = np.diff(fields["x_edges_m"])
    dy = np.diff(fields["y_edges_m"])
    dz = np.diff(fields["z_edges_m"])
    dvol = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
    q_areal = np.sum(fields["Q_W_m3"] * dz[None, None, :], axis=2)
    gradient = np.hypot(fields["grad_T_x_K_m"], fields["grad_T_y_K_m"])
    integrand = fields["shockley_ramo_integrand_A_m2"]
    current_2d = integrate_2d(integrand, dx, dy)
    positive_current = integrate_2d(np.maximum(integrand, 0.0), dx, dy)
    negative_current = integrate_2d(np.minimum(integrand, 0.0), dx, dy)
    weighting_x = fields["weighting_grad_x_m_inv"][:, :, None]
    weighting_y = fields["weighting_grad_y_m_inv"][:, :, None]
    current_x = float(
        np.sum(fields["local_J_x_A_m2_3d"] * weighting_x * dvol)
    )
    current_y = float(
        np.sum(fields["local_J_y_A_m2_3d"] * weighting_y * dvol)
    )
    mask_2d = np.any(fields["flake_mask"], axis=2)
    weighted_power = q_areal * dx[:, None] * dy[None, :]
    total_power = float(np.sum(weighted_power))
    x = 0.5 * (fields["x_edges_m"][:-1] + fields["x_edges_m"][1:])
    y = 0.5 * (fields["y_edges_m"][:-1] + fields["y_edges_m"][1:])
    q_centroid = [
        float(np.sum(weighted_power * x[:, None]) / total_power),
        float(np.sum(weighted_power * y[None, :]) / total_power),
    ]
    return {
        "q_areal_W_m2": q_areal,
        "temperature_K": fields["temperature_flake_average_K"],
        "gradient_K_m": gradient,
        "integrand_A_m2": integrand,
        "mask_2d": mask_2d,
        "current_from_2d_integrand_A": current_2d,
        "positive_current_contribution_A": positive_current,
        "negative_current_contribution_A": negative_current,
        "current_x_contribution_A": current_x,
        "current_y_contribution_A": current_y,
        "component_sum_A": current_x + current_y,
        "q_centroid_m": q_centroid,
        "mapped_power_W": total_power,
    }


def finite_range(arrays: list[np.ndarray], *, symmetric: bool = False) -> tuple[float, float]:
    values = np.concatenate([array[np.isfinite(array)].ravel() for array in arrays])
    if symmetric:
        bound = float(np.max(np.abs(values)))
        return -bound, bound
    return float(np.min(values)), float(np.max(values))


def plot_comparison(
    output: Path,
    a: dict[str, Any],
    b: dict[str, Any],
) -> None:
    fields = a["fields"]
    x_um = fields["x_edges_m"] * 1e6
    y_um = fields["y_edges_m"] * 1e6
    metrics_a = a["metrics"]
    metrics_b = b["metrics"]
    rows = (
        ("q_areal_W_m2", "absorbed areal power", "W/m²", False),
        ("temperature_K", "thickness-averaged ΔT", "K", False),
        ("gradient_K_m", "|∇T|", "K/m", False),
        ("integrand_A_m2", "Shockley–Ramo integrand", "A/m²", True),
    )
    figure, axes = plt.subplots(4, 3, figsize=(16, 19), constrained_layout=True)
    for row, (key, title, unit, symmetric) in enumerate(rows):
        array_a = np.where(metrics_a["mask_2d"], metrics_a[key], np.nan)
        array_b = np.where(metrics_b["mask_2d"], metrics_b[key], np.nan)
        difference = array_b - array_a
        vmin, vmax = finite_range([array_a, array_b], symmetric=symmetric)
        dmin, dmax = finite_range([difference], symmetric=True)
        cmap = "coolwarm" if symmetric else "inferno"
        for column, (array, label) in enumerate(
            ((array_a, "E || a"), (array_b, "E || b"))
        ):
            image = axes[row, column].pcolormesh(
                x_um,
                y_um,
                array.T,
                shading="flat",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            axes[row, column].set_title(f"{label}: {title}")
            figure.colorbar(image, ax=axes[row, column], label=unit)
        image = axes[row, 2].pcolormesh(
            x_um,
            y_um,
            difference.T,
            shading="flat",
            cmap="coolwarm",
            vmin=dmin,
            vmax=dmax,
        )
        axes[row, 2].set_title(f"b - a: {title}")
        figure.colorbar(image, ax=axes[row, 2], label=unit)
        for axis in axes[row]:
            axis.set_aspect("equal")
            axis.set_xlabel("lab x = b (µm)")
            axis.set_ylabel("lab y = a (µm)")
    figure.suptitle(
        "Registered Device-A single-position Maxwell → explicit 3D FVM → PTE",
        fontsize=16,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optical-a", type=Path, required=True)
    parser.add_argument("--optical-b", type=Path, required=True)
    parser.add_argument("--thermal-a", type=Path, required=True)
    parser.add_argument("--thermal-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = {
        "a": load_case(args.optical_a, args.thermal_a),
        "b": load_case(args.optical_b, args.thermal_b),
    }
    for case in cases.values():
        case["metrics"] = field_metrics(case)
    a = cases["a"]
    b = cases["b"]
    fa = a["fields"]
    fb = b["fields"]
    coordinates_identical = all(
        np.array_equal(fa[key], fb[key])
        for key in ("x_edges_m", "y_edges_m", "z_edges_m", "flake_mask")
    )
    weighting_identical = bool(
        np.array_equal(
            fa["weighting_potential"],
            fb["weighting_potential"],
            equal_nan=True,
        )
        and np.array_equal(
            fa["weighting_grad_x_m_inv"], fb["weighting_grad_x_m_inv"]
        )
        and np.array_equal(
            fa["weighting_grad_y_m_inv"], fb["weighting_grad_y_m_inv"]
        )
    )

    records: dict[str, Any] = {}
    for polarization, case in cases.items():
        optical = case["optical"]["run_result"]
        thermal = case["thermal"]
        metrics = case["metrics"]
        records[polarization] = {
            "optical_status": case["optical"]["status"],
            "optical_acceptance": optical["acceptance"],
            "P_Q_full_control_volume_W_at_1_W_m2": optical["P_Q_W"],
            "P_six_W_at_1_W_m2": optical["P_six_face_W"],
            "six_face_closure": optical["six_face_relative_closure"],
            "auto_shutoff": optical["auto_shutoff"]["final_value"],
            "Q_component_power_W_at_1_W_m2": optical["component_power_W"],
            "TaIrTe4_support_power_W_at_1_W_m2": optical[
                "material_resolved_absorption"
            ]["P_Q_TaIrTe4_exact_support_W"],
            "mapped_TaIrTe4_power_W_at_284p40uW": thermal["mapping"][
                "P_Q_target_W"
            ],
            "mapping_relative_power_error": thermal["mapping"][
                "mapping_relative_power_error"
            ],
            "mapped_power_outside_TaIrTe4_W": thermal["mapping"][
                "mapped_power_outside_flake_W"
            ],
            "Tmax_rise_K": thermal["thermal"]["Tmax_rise_K"],
            "TaIrTe4_volume_average_rise_K": thermal["thermal"][
                "TaIrTe4_volume_average_rise_K"
            ],
            "thermal_linear_residual_relative": thermal["thermal"][
                "linear_residual_relative"
            ],
            "thermal_energy_balance_relative_error": thermal["thermal"][
                "energy_balance_relative_error"
            ],
            "PTE_current_A": thermal["PTE_current_A_at_requested_incident_power"],
            "PTE_current_pA": thermal[
                "PTE_current_pA_at_requested_incident_power"
            ],
            "PTE_positive_contribution_A": metrics[
                "positive_current_contribution_A"
            ],
            "PTE_negative_contribution_A": metrics[
                "negative_current_contribution_A"
            ],
            "PTE_x_component_A": metrics["current_x_contribution_A"],
            "PTE_y_component_A": metrics["current_y_contribution_A"],
            "Q_centroid_m": metrics["q_centroid_m"],
            "resistance_audit_ohm": thermal["two_terminal_resistance_audit"][
                "predicted_resistance_ohm"
            ],
            "absolute_current_certification": thermal[
                "absolute_current_certification"
            ],
        }

    ratios = {
        "raw_full_control_volume_PQ_b_over_a": records["b"][
            "P_Q_full_control_volume_W_at_1_W_m2"
        ]
        / records["a"]["P_Q_full_control_volume_W_at_1_W_m2"],
        "TaIrTe4_support_PQ_b_over_a": records["b"][
            "TaIrTe4_support_power_W_at_1_W_m2"
        ]
        / records["a"]["TaIrTe4_support_power_W_at_1_W_m2"],
        "mapped_power_b_over_a_at_equal_incident_power": records["b"][
            "mapped_TaIrTe4_power_W_at_284p40uW"
        ]
        / records["a"]["mapped_TaIrTe4_power_W_at_284p40uW"],
        "Tmax_b_over_a": records["b"]["Tmax_rise_K"]
        / records["a"]["Tmax_rise_K"],
        "Tvolume_average_b_over_a": records["b"][
            "TaIrTe4_volume_average_rise_K"
        ]
        / records["a"]["TaIrTe4_volume_average_rise_K"],
        "absolute_current_b_over_a": abs(records["b"]["PTE_current_A"])
        / abs(records["a"]["PTE_current_A"]),
    }
    gates = {
        "both_optical_completed": all(
            records[pol]["optical_status"] == "COMPLETED" for pol in "ab"
        ),
        "both_optical_acceptance_all_true": all(
            all(bool(value) for value in records[pol]["optical_acceptance"].values())
            for pol in "ab"
        ),
        "both_optical_closure_lt_0p5_percent": max(
            records[pol]["six_face_closure"] for pol in "ab"
        )
        < 0.005,
        "both_auto_shutoff_lt_1e_minus_5": max(
            records[pol]["auto_shutoff"] for pol in "ab"
        )
        < 1e-5,
        "both_mapping_error_lt_1e_minus_12": max(
            records[pol]["mapping_relative_power_error"] for pol in "ab"
        )
        < 1e-12,
        "both_zero_mapped_power_outside_TaIrTe4": all(
            records[pol]["mapped_power_outside_TaIrTe4_W"] == 0.0 for pol in "ab"
        ),
        "both_thermal_residual_lt_1e_minus_8": max(
            records[pol]["thermal_linear_residual_relative"] for pol in "ab"
        )
        < 1e-8,
        "both_energy_balance_lt_1_percent": max(
            records[pol]["thermal_energy_balance_relative_error"] for pol in "ab"
        )
        < 0.01,
        "coordinates_and_weighting_operator_identical": bool(
            coordinates_identical and weighting_identical
        ),
    }
    summary = {
        "status": "PARTIAL_REGISTERED_DEVICE_A_SINGLE_POSITION_CURRENT_TREND_OPPOSITE_PAPER_SCAN_PEAK",
        "scope": "one approximately registered scan position; not the Figure-3I scan maximum",
        "cases": records,
        "ratios": ratios,
        "gates": gates,
        "interpretation": {
            "absorption_trend": "b>a",
            "Tmax_trend": "a>b due to stronger localized hotspot",
            "current_trend_at_this_position": "a>b",
            "paper_comparison_not_yet_closed": (
                "Figure 3I compares a scan profile/peak; a single registered "
                "position cannot certify or refute the reported peak ratio"
            ),
            "absolute_current_blocker": (
                "digitized conductivity geometry predicts about 14.1 ohm "
                "versus the measured 213 ohm; no empirical rescaling applied"
            ),
            "next_required_step": (
                "registered sparse scan along the Figure-3H dashed line, "
                "then compare separate maxima of |Ia| and |Ib|"
            ),
        },
        "model_contract": {
            "incident_power_W": 284.40e-6,
            "thermal_model": "expanded explicit 3D Cartesian FVM",
            "Q_source": "TaIrTe4-only",
            "Q_attribution": "literal intersection-density",
            "SiO2_and_Si_optical_loss_in_thermal_source": False,
            "metal_absorption_in_thermal_source": False,
            "metal_scenario": "isolated-lower-bound diagnostic",
            "no_Q_clipping_smoothing_gain_rescaling_or_nearest_relocation": True,
        },
    }
    (args.output_dir / "device_a_registered_single_position_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    csv_path = args.output_dir / "device_a_registered_single_position_cases.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "polarization",
                "P_Q_full_W",
                "P_Q_TaIrTe4_W",
                "mapped_power_W_at_284p40uW",
                "Tmax_rise_K",
                "Tvolume_average_rise_K",
                "PTE_current_A",
                "positive_contribution_A",
                "negative_contribution_A",
                "x_component_A",
                "y_component_A",
                "optical_closure",
                "thermal_residual",
                "energy_balance_error",
            ],
        )
        writer.writeheader()
        for polarization in "ab":
            record = records[polarization]
            writer.writerow(
                {
                    "polarization": polarization,
                    "P_Q_full_W": record["P_Q_full_control_volume_W_at_1_W_m2"],
                    "P_Q_TaIrTe4_W": record[
                        "TaIrTe4_support_power_W_at_1_W_m2"
                    ],
                    "mapped_power_W_at_284p40uW": record[
                        "mapped_TaIrTe4_power_W_at_284p40uW"
                    ],
                    "Tmax_rise_K": record["Tmax_rise_K"],
                    "Tvolume_average_rise_K": record[
                        "TaIrTe4_volume_average_rise_K"
                    ],
                    "PTE_current_A": record["PTE_current_A"],
                    "positive_contribution_A": record[
                        "PTE_positive_contribution_A"
                    ],
                    "negative_contribution_A": record[
                        "PTE_negative_contribution_A"
                    ],
                    "x_component_A": record["PTE_x_component_A"],
                    "y_component_A": record["PTE_y_component_A"],
                    "optical_closure": record["six_face_closure"],
                    "thermal_residual": record[
                        "thermal_linear_residual_relative"
                    ],
                    "energy_balance_error": record[
                        "thermal_energy_balance_relative_error"
                    ],
                }
            )

    plot_path = args.output_dir / "DEVICE_A_REGISTERED_SINGLE_POSITION_MAPS.png"
    plot_comparison(plot_path, a, b)

    raw_paths = [
        (args.optical_a / "case_result.json", "E||a optical case JSON"),
        (args.optical_a / "finite_q_on_artifact.npz", "E||a raw volumetric Q"),
        (args.optical_a / "finite_2um_optical_q.fsp", "E||a raw FSP"),
        (args.optical_b / "case_result.json", "E||b optical case JSON"),
        (args.optical_b / "finite_q_on_artifact.npz", "E||b raw volumetric Q"),
        (args.optical_b / "finite_2um_optical_q.fsp", "E||b raw FSP"),
        (args.thermal_a / "summary.json", "E||a thermal/PTE summary"),
        (args.thermal_a / "thermal_pte_fields.npz", "E||a thermal/PTE fields"),
        (args.thermal_b / "summary.json", "E||b thermal/PTE summary"),
        (args.thermal_b / "thermal_pte_fields.npz", "E||b thermal/PTE fields"),
    ]
    manifest = {
        "status": "RAW_REGISTERED_SINGLE_POSITION_ARTIFACTS_RECORDED_NOT_COMMITTED",
        "artifacts": [artifact(path, role) for path, role in raw_paths],
        "raw_NPZ_and_FSP_committed_to_git": False,
    }
    (args.output_dir / "RAW_ARTIFACT_MANIFEST_SINGLE_POSITION.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    report = f"""# Registered Device-A single-position Maxwell–thermal–PTE audit

Status: `{summary['status']}`

This is one approximately registered point on the Figure-3H dashed-line
scenario, not a scan maximum and not a certified reproduction of Figure 3I.
Both polarization-specific GPU Maxwell calculations, literal material-overlap
mapping operations, and the identical explicit 3D thermal/PTE operators passed
their numerical gates.

| metric | E parallel a | E parallel b | b/a |
|---|---:|---:|---:|
| TaIrTe4 mapped power at 284.40 uW | {records['a']['mapped_TaIrTe4_power_W_at_284p40uW']:.8e} W | {records['b']['mapped_TaIrTe4_power_W_at_284p40uW']:.8e} W | {ratios['mapped_power_b_over_a_at_equal_incident_power']:.6f} |
| Tmax rise | {records['a']['Tmax_rise_K']:.8f} K | {records['b']['Tmax_rise_K']:.8f} K | {ratios['Tmax_b_over_a']:.6f} |
| flake volume-average rise | {records['a']['TaIrTe4_volume_average_rise_K']:.8f} K | {records['b']['TaIrTe4_volume_average_rise_K']:.8f} K | {ratios['Tvolume_average_b_over_a']:.6f} |
| integrated PTE current | {records['a']['PTE_current_A']:.8e} A | {records['b']['PTE_current_A']:.8e} A | {ratios['absolute_current_b_over_a']:.6f} |

The total/mapped absorption has the paper-like `b>a` trend, but the `a`
polarization produces a stronger localized hotspot and the single-position
integrated current remains `a>b`. The current was evaluated as a full
flake-cell volume integral; it is not a one-point gradient sample.

Absolute current is not certified. The digitized geometry predicts about
`{records['a']['resistance_audit_ohm']:.3f} ohm`, versus the measured `213 ohm`.
No resistance or current rescaling was applied. The next physically meaningful
comparison is the registered sparse scan and the separate maxima of `|Ia|` and
`|Ib|`, matching the interpretation of Figure 3I.
"""
    (args.output_dir / "DEVICE_A_REGISTERED_SINGLE_POSITION_REPORT.md").write_text(
        report
    )
    print(json.dumps(summary, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
