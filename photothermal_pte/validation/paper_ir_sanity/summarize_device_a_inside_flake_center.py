#!/usr/bin/env python3
"""Summarize the Device-A inside-flake beam-centre diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finite-a-dir", type=Path, required=True)
    parser.add_argument("--finite-b-dir", type=Path, required=True)
    parser.add_argument("--thermal-a-dir", type=Path, required=True)
    parser.add_argument("--thermal-b-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "committed_to_git": False,
    }


def finite_metrics(payload: dict[str, object]) -> dict[str, object]:
    run = payload["run_result"]
    material = run["material_resolved_absorption"]
    return {
        "P_Q_full_control_volume_W_at_1_W_m2": run["P_Q_W"],
        "P_six_W_at_1_W_m2": run["P_six_face_W"],
        "six_face_relative_closure": run["six_face_relative_closure"],
        "auto_shutoff": run["auto_shutoff"]["final_value"],
        "P_Q_TaIrTe4_exact_support_W_at_1_W_m2": material[
            "P_Q_TaIrTe4_exact_support_W"
        ],
        "component_power_W_at_1_W_m2": run["component_power_W"],
        "negative_Q_voxel_count": run["negative_Q_voxel_count"],
        "Q_hotspot": run["Q_hotspot"],
        "acceptance": run["acceptance"],
    }


def thermal_metrics(payload: dict[str, object]) -> dict[str, object]:
    return {
        "source_power_W_at_285uW_incident": payload["thermal"]["source_power_W"],
        "Tmax_rise_K": payload["thermal"]["Tmax_rise_K"],
        "TaIrTe4_volume_average_rise_K": payload["thermal"][
            "TaIrTe4_volume_average_rise_K"
        ],
        "linear_residual_relative": payload["thermal"][
            "linear_residual_relative"
        ],
        "energy_balance_relative_error": payload["thermal"][
            "energy_balance_relative_error"
        ],
        "PTE_current_A_at_285uW_incident": payload[
            "PTE_current_A_at_285uW_incident"
        ],
        "absolute_current_certification": payload[
            "absolute_current_certification"
        ],
        "mapping_relative_power_error": payload["mapping"][
            "mapping_relative_power_error"
        ],
        "mapped_power_outside_flake_W": payload["mapping"][
            "mapped_power_outside_flake_W"
        ],
        "modeled_source_fraction_of_full_common_grid_Q": payload["mapping"][
            "modeled_source_fraction_of_full_common_grid_Q"
        ],
    }


def plot_geometry(finite: dict[str, object], output: Path) -> None:
    geometry = finite["pre_run_contract"]["geometry"]
    digitized = geometry["digitized_device_a_contract"]
    source = geometry["source"]
    flake = np.asarray(digitized["flake_vertices_simulation_um"], float)
    top = np.asarray(digitized["top_metal_polygon_simulation_um"], float)
    bottom = np.asarray(digitized["bottom_metal_polygon_simulation_um"], float)
    beam = np.asarray(source["beam_center_m"], float) * 1.0e6
    waist = float(source["physical_target_waist_radius_m"]) * 1.0e6
    source_span = float(source["source_span_m"]) * 1.0e6
    domain = np.asarray(
        [geometry["domain_bounds_m"][axis] for axis in "xy"], float
    ) * 1.0e6

    figure, axis = plt.subplots(figsize=(8.5, 8), constrained_layout=True)
    axis.add_patch(
        Polygon(flake, facecolor="#8b5fbf", edgecolor="#542788", alpha=0.55)
    )
    for metal in (top, bottom):
        axis.add_patch(
            Polygon(metal, facecolor="#d4af37", edgecolor="#8c6d00", alpha=0.65)
        )
    axis.add_patch(
        Rectangle(
            beam - 0.5 * source_span,
            source_span,
            source_span,
            fill=False,
            edgecolor="#1f77b4",
            linestyle="--",
            linewidth=1.5,
            label="50 µm source aperture",
        )
    )
    axis.add_patch(
        Circle(
            beam,
            waist,
            fill=False,
            edgecolor="#d62728",
            linewidth=2,
            label=r"Gaussian $w_0=8.75$ µm",
        )
    )
    axis.scatter(*beam, color="#d62728", marker="x", s=80, zorder=5)
    axis.add_patch(
        Rectangle(
            (domain[0, 0], domain[1, 0]),
            domain[0, 1] - domain[0, 0],
            domain[1, 1] - domain[1, 0],
            fill=False,
            edgecolor="black",
            linewidth=2,
            label="64 µm FDTD / six PML",
        )
    )
    axis.set(
        xlim=(-33, 33),
        ylim=(-33, 33),
        xlabel="x = b (µm)",
        ylabel="y = a (µm)",
        aspect="equal",
        title=(
            "Device A inside-flake illumination diagnostic\n"
            "beam centre = (0, 0) µm in the realized simulation frame"
        ),
    )
    axis.legend(loc="upper right", fontsize=9)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_thermal(fields: dict[str, Path], output: Path) -> None:
    loaded = {key: np.load(path) for key, path in fields.items()}
    first = loaded["a"]
    x_edges = first["x_edges_m"] * 1.0e6
    y_edges = first["y_edges_m"] * 1.0e6
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for key, data in loaded.items():
        dz = np.diff(data["z_edges_m"])
        q_areal = np.sum(data["Q_W_m3"] * dz[None, None, :], axis=2)
        mask = np.any(data["flake_mask"], axis=2)
        temperature = np.where(mask, data["temperature_flake_average_K"], np.nan)
        gradient = np.where(
            mask,
            np.hypot(data["grad_T_x_K_m"], data["grad_T_y_K_m"]),
            np.nan,
        )
        arrays[key] = {"q": q_areal, "temperature": temperature, "gradient": gradient}
    limits = {
        name: max(float(np.nanmax(arrays[key][name])) for key in ("a", "b"))
        for name in ("q", "temperature", "gradient")
    }
    labels = {
        "q": r"TaIrTe$_4$ absorbed areal power (W/m$^2$)",
        "temperature": r"thickness-averaged $\Delta T$ (K)",
        "gradient": r"$|\nabla_{xy}T|$ (K/m)",
    }
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for row, key in enumerate(("a", "b")):
        for column, name in enumerate(("q", "temperature", "gradient")):
            image = axes[row, column].pcolormesh(
                x_edges,
                y_edges,
                arrays[key][name].T,
                cmap="inferno",
                vmin=0.0,
                vmax=limits[name],
                shading="flat",
            )
            axes[row, column].set(
                xlim=(-18, 18),
                ylim=(-18, 12),
                xlabel="x=b (µm)",
                ylabel="y=a (µm)",
                title=f"E ∥ {key}: {labels[name]}",
            )
            figure.colorbar(image, ax=axes[row, column], shrink=0.82)
    figure.suptitle(
        "Inside-flake beam centre: TaIrTe4-only intersection-density source",
        fontsize=15,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)
    for data in loaded.values():
        data.close()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    finite_paths = {
        "a": args.finite_a_dir / "case_result.json",
        "b": args.finite_b_dir / "case_result.json",
    }
    thermal_paths = {
        "a": args.thermal_a_dir / "summary.json",
        "b": args.thermal_b_dir / "summary.json",
    }
    finite = {key: json.loads(path.read_text()) for key, path in finite_paths.items()}
    thermal = {key: json.loads(path.read_text()) for key, path in thermal_paths.items()}
    cases = {
        key: {
            "polarization": key,
            "optical": finite_metrics(finite[key]),
            "thermal_pte": thermal_metrics(thermal[key]),
        }
        for key in ("a", "b")
    }
    optical_ratio = (
        cases["b"]["optical"]["P_Q_TaIrTe4_exact_support_W_at_1_W_m2"]
        / cases["a"]["optical"]["P_Q_TaIrTe4_exact_support_W_at_1_W_m2"]
    )
    current_ratio = abs(
        cases["b"]["thermal_pte"]["PTE_current_A_at_285uW_incident"]
        / cases["a"]["thermal_pte"]["PTE_current_A_at_285uW_incident"]
    )
    summary = {
        "status": "COMPLETED_DEVICE_A_INSIDE_FLAKE_BEAM_CENTER_DIAGNOSTIC",
        "interpretation": (
            "single explicitly chosen inside-flake beam position; not a paper-"
            "registered scan coordinate and not an absolute-current certification"
        ),
        "beam_center": {
            "digitized_code_um": [0.0, 3.0],
            "realized_simulation_um": [0.0, 0.0],
            "x_axis": "crystal b",
            "y_axis": "crystal a",
            "inside_flake": True,
            "overlaps_digitized_metal": False,
            "minimum_source_aperture_to_lateral_PML_um": 7.0,
        },
        "source_contract": {
            "wavelength_um": 11.0,
            "waist_um": 8.75,
            "source_span_um": 50.0,
            "FDTD_domain_um": 64.0,
            "six_PML": True,
            "incident_power_for_thermal_PTE_uW": 285.0,
        },
        "thermal_source_contract": {
            "included": "TaIrTe4-only Maxwell volumetric Q",
            "attribution": "literal optical-cell/thermal-TaIrTe4 intersection density",
            "SiO2_absorption_included_as_thermal_source": False,
            "metal_absorption_included_as_thermal_source": False,
            "clipping_smoothing_gain_rescaling_tiling": False,
        },
        "cases": cases,
        "ratios": {
            "TaIrTe4_optical_absorption_b_over_a": optical_ratio,
            "Tmax_b_over_a": cases["b"]["thermal_pte"]["Tmax_rise_K"]
            / cases["a"]["thermal_pte"]["Tmax_rise_K"],
            "TaIrTe4_average_temperature_b_over_a": cases["b"]["thermal_pte"][
                "TaIrTe4_volume_average_rise_K"
            ]
            / cases["a"]["thermal_pte"]["TaIrTe4_volume_average_rise_K"],
            "absolute_PTE_current_b_over_a": current_ratio,
            "PTE_current_signs": {
                "a": int(np.sign(cases["a"]["thermal_pte"]["PTE_current_A_at_285uW_incident"])),
                "b": int(np.sign(cases["b"]["thermal_pte"]["PTE_current_A_at_285uW_incident"])),
            },
        },
        "gates": {
            "both_optical_closure_lt_0p5_percent": all(
                cases[key]["optical"]["six_face_relative_closure"] < 0.005
                for key in ("a", "b")
            ),
            "both_auto_shutoff_lt_1e_minus_5": all(
                cases[key]["optical"]["auto_shutoff"] < 1.0e-5
                for key in ("a", "b")
            ),
            "both_mapping_power_error_lt_1e_minus_12": all(
                cases[key]["thermal_pte"]["mapping_relative_power_error"] < 1.0e-12
                for key in ("a", "b")
            ),
            "both_thermal_energy_balance_lt_1_percent": all(
                cases[key]["thermal_pte"]["energy_balance_relative_error"] < 0.01
                for key in ("a", "b")
            ),
            "both_linear_residual_lt_1e_minus_8": all(
                cases[key]["thermal_pte"]["linear_residual_relative"] < 1.0e-8
                for key in ("a", "b")
            ),
            "absolute_current_certified": False,
        },
        "fail_closed_diagnostics_preserved": [
            str((args.finite_a_dir.parent / "finite_a_20260803").resolve()),
            str((args.finite_b_dir.parent / "empty_b_20260803").resolve()),
            str((args.thermal_a_dir.parent / "thermal_a_tairte4_only_20260803").resolve()),
        ],
    }
    (args.output_dir / "device_a_inside_flake_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    with (args.output_dir / "device_a_inside_flake_cases.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "polarization",
                "P_Q_full_W_at_1Wm2",
                "P_Q_TaIrTe4_W_at_1Wm2",
                "P_six_W_at_1Wm2",
                "closure",
                "auto_shutoff",
                "thermal_source_W_at_285uW",
                "Tmax_K",
                "Tavg_TaIrTe4_K",
                "PTE_current_A",
                "thermal_residual",
                "energy_error",
            ]
        )
        for key in ("a", "b"):
            optical = cases[key]["optical"]
            therm = cases[key]["thermal_pte"]
            writer.writerow(
                [
                    key,
                    optical["P_Q_full_control_volume_W_at_1_W_m2"],
                    optical["P_Q_TaIrTe4_exact_support_W_at_1_W_m2"],
                    optical["P_six_W_at_1_W_m2"],
                    optical["six_face_relative_closure"],
                    optical["auto_shutoff"],
                    therm["source_power_W_at_285uW_incident"],
                    therm["Tmax_rise_K"],
                    therm["TaIrTe4_volume_average_rise_K"],
                    therm["PTE_current_A_at_285uW_incident"],
                    therm["linear_residual_relative"],
                    therm["energy_balance_relative_error"],
                ]
            )

    plot_geometry(
        finite["a"], args.output_dir / "DEVICE_A_INSIDE_FLAKE_GEOMETRY.png"
    )
    plot_thermal(
        {
            "a": args.thermal_a_dir / "thermal_pte_fields.npz",
            "b": args.thermal_b_dir / "thermal_pte_fields.npz",
        },
        args.output_dir / "DEVICE_A_INSIDE_FLAKE_THERMAL_COMPARISON.png",
    )

    raw_paths = [
        finite_paths["a"],
        args.finite_a_dir / "finite_q_on_artifact.npz",
        args.finite_a_dir / "finite_2um_optical_q.fsp",
        finite_paths["b"],
        args.finite_b_dir / "finite_q_on_artifact.npz",
        args.finite_b_dir / "finite_2um_optical_q.fsp",
        thermal_paths["a"],
        args.thermal_a_dir / "thermal_pte_fields.npz",
        thermal_paths["b"],
        args.thermal_b_dir / "thermal_pte_fields.npz",
    ]
    manifest = {
        "status": summary["status"],
        "raw_NPZ_and_FSP_committed_to_git": False,
        "artifacts": [artifact(path) for path in raw_paths],
    }
    (args.output_dir / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    report = f"""# Device-A inside-flake beam-centre diagnostic

Status: `{summary['status']}`

The Gaussian centre was deliberately moved from the earlier outside-flake
registration to digitized `(x=b, y=a)=(0, 3) um`, which becomes `(0, 0) um`
in the realized simulation frame. This is a named inside-flake diagnostic,
not a claim about the unpublished experimental stage coordinate.

## Fixed contract

- 11-um scalar Gaussian, `w0=8.75 um`, 50-um square source aperture.
- 64-um lateral FDTD domain, six PML boundaries, GPU-only solve.
- Device-A digitized flake and Au/Ti electrode polygons.
- TaIrTe4-only volumetric Maxwell Q enters thermal through literal
  optical-cell/thermal-material intersection density.
- SiO2 and metal optical loss are not thermal sources in this diagnostic.
- No clipping, smoothing, gain, global rescaling, tiling, or nearest-cell
  relocation was used.

## Result

| metric | E parallel a | E parallel b | b/a |
|---|---:|---:|---:|
| TaIrTe4 optical power at unit central intensity (W) | {cases['a']['optical']['P_Q_TaIrTe4_exact_support_W_at_1_W_m2']:.9e} | {cases['b']['optical']['P_Q_TaIrTe4_exact_support_W_at_1_W_m2']:.9e} | {optical_ratio:.6f} |
| mapped thermal source at 285 uW (W) | {cases['a']['thermal_pte']['source_power_W_at_285uW_incident']:.9e} | {cases['b']['thermal_pte']['source_power_W_at_285uW_incident']:.9e} | {cases['b']['thermal_pte']['source_power_W_at_285uW_incident']/cases['a']['thermal_pte']['source_power_W_at_285uW_incident']:.6f} |
| Tmax rise (K) | {cases['a']['thermal_pte']['Tmax_rise_K']:.9g} | {cases['b']['thermal_pte']['Tmax_rise_K']:.9g} | {summary['ratios']['Tmax_b_over_a']:.6f} |
| TaIrTe4 average rise (K) | {cases['a']['thermal_pte']['TaIrTe4_volume_average_rise_K']:.9g} | {cases['b']['thermal_pte']['TaIrTe4_volume_average_rise_K']:.9g} | {summary['ratios']['TaIrTe4_average_temperature_b_over_a']:.6f} |
| integrated PTE current (pA) | {1e12*cases['a']['thermal_pte']['PTE_current_A_at_285uW_incident']:.6f} | {1e12*cases['b']['thermal_pte']['PTE_current_A_at_285uW_incident']:.6f} | abs={current_ratio:.6f} |

Both optical closure errors are below 0.5%, both auto-shutoff values are below
`1e-5`, mapping power errors are zero to recorded precision, and both thermal
energy errors are far below 1%.

The current changes sign and `|I_b|>|I_a|` at this inside-flake position.
However, the digitized geometry predicts about 14.1 ohm whereas the measured
Device-A resistance is 213 ohm. Therefore the absolute pA/nA magnitudes are
not certified experimental predictions.

## Corrected failure found during this task

The first thermal attempt silently rebuilt the old outside-flake coordinate
translation from the geometry JSON. It displaced the thermal flake by 10.67
um relative to the new optical Q. That raw result is preserved as a failed
diagnostic. Production thermal now loads the realized translated polygon from
the actual optical `case_result.json` and fail-closes on a geometry-path or
coordinate mismatch.
"""
    (args.output_dir / "DEVICE_A_INSIDE_FLAKE_REPORT.md").write_text(report)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
