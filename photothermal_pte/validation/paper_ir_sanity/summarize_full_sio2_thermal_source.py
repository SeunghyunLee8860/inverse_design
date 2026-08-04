#!/usr/bin/env python3
"""Compare TaIrTe4-only and material-resolved TaIrTe4+SiO2 heating."""

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


SIO2_K_W_MK = 1.38
SI_K_W_MK = 145.0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_thermal(directory: Path) -> dict[str, Any]:
    summary_path = directory / "summary.json"
    fields_path = directory / "thermal_pte_fields.npz"
    summary = read_json(summary_path)
    if summary.get("status") != "COMPLETED_STRAIGHT_45_EDGE_THERMAL_CONTROL":
        raise RuntimeError(f"thermal case did not complete: {summary_path}")
    with np.load(fields_path, allow_pickle=False) as stored:
        fields = {key: np.array(stored[key], copy=True) for key in stored.files}
    return {
        "directory": directory,
        "summary_path": summary_path,
        "fields_path": fields_path,
        "summary": summary,
        "fields": fields,
    }


def load_optical(directory: Path) -> dict[str, Any]:
    result_path = directory / "case_result.json"
    artifact_path = directory / "finite_q_on_artifact.npz"
    result = read_json(result_path)
    if result.get("status") != "COMPLETED":
        raise RuntimeError(f"optical case did not pass: {result_path}")
    return {
        "directory": directory,
        "result_path": result_path,
        "artifact_path": artifact_path,
        "result": result,
    }


def safe_relative(new: float, old: float) -> float:
    return (new - old) / max(abs(old), np.finfo(float).tiny)


def flake_mask_2d(fields: dict[str, np.ndarray]) -> np.ndarray:
    return np.any(fields["flake_mask"], axis=2)


def q_areal(fields: dict[str, np.ndarray], z_selector: np.ndarray) -> np.ndarray:
    dz = np.diff(fields["z_edges_m"])
    return np.sum(
        fields["Q_W_m3"][:, :, z_selector] * dz[z_selector][None, None, :],
        axis=2,
    )


def plot_sources_and_temperature(
    combined: dict[str, dict[str, Any]], output: Path
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 17), constrained_layout=True)
    arrays: dict[tuple[str, str], np.ndarray] = {}
    for pol in ("a", "b"):
        fields = combined[pol]["fields"]
        z = 0.5 * (fields["z_edges_m"][:-1] + fields["z_edges_m"][1:])
        ta = (z > -130e-9) & (z < 0.0)
        oxide = (z > -415e-9) & (z < -130e-9)
        arrays[(pol, "ta_q")] = q_areal(fields, ta)
        arrays[(pol, "oxide_q")] = q_areal(fields, oxide)
        arrays[(pol, "temp")] = np.where(
            flake_mask_2d(fields), fields["temperature_flake_average_K"], np.nan
        )
    for row, (key, title, units) in enumerate(
        (
            ("ta_q", "TaIrTe4 absorbed areal source", "W/m²"),
            ("oxide_q", "SiO2 absorbed areal source", "W/m²"),
            ("temp", "TaIrTe4 thickness-averaged ΔT", "K"),
        )
    ):
        vmax = max(float(np.nanmax(arrays[(pol, key)])) for pol in ("a", "b"))
        for col, pol in enumerate(("a", "b")):
            fields = combined[pol]["fields"]
            image = axes[row, col].pcolormesh(
                fields["x_edges_m"] * 1e6,
                fields["y_edges_m"] * 1e6,
                np.ma.masked_invalid(arrays[(pol, key)].T),
                shading="flat",
                cmap="inferno",
                vmin=0.0,
                vmax=vmax,
            )
            axes[row, col].set_aspect("equal")
            axes[row, col].set(
                title=f"E ∥ {pol}: {title}",
                xlabel="lab x=b (µm)",
                ylabel="lab y=a (µm)",
            )
            fig.colorbar(image, ax=axes[row, col], label=units)
    fig.suptitle("Material-resolved Maxwell heating: TaIrTe4 + full 285-nm SiO2")
    fig.savefig(output / "FULL_SIO2_Q_AND_TEMPERATURE.png", dpi=190)
    plt.close(fig)


def plot_thermal_change(
    baseline: dict[str, dict[str, Any]],
    combined: dict[str, dict[str, Any]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for row, pol in enumerate(("a", "b")):
        old = baseline[pol]["fields"]
        new = combined[pol]["fields"]
        mask = flake_mask_2d(new)
        old_t = np.where(mask, old["temperature_flake_average_K"], np.nan)
        new_t = np.where(mask, new["temperature_flake_average_K"], np.nan)
        delta = new_t - old_t
        vmax = max(float(np.nanmax(old_t)), float(np.nanmax(new_t)))
        for col, (values, title, lo, hi, cmap) in enumerate(
            (
                (old_t, "TaIrTe4-only Q", 0.0, vmax, "inferno"),
                (new_t, "TaIrTe4 + SiO2 Q", 0.0, vmax, "inferno"),
                (
                    delta,
                    "increment caused by SiO2 Q",
                    float(np.nanmin(delta)),
                    float(np.nanmax(delta)),
                    "viridis",
                ),
            )
        ):
            image = axes[row, col].pcolormesh(
                new["x_edges_m"] * 1e6,
                new["y_edges_m"] * 1e6,
                np.ma.masked_invalid(values.T),
                shading="flat",
                cmap=cmap,
                vmin=lo,
                vmax=hi,
            )
            axes[row, col].set_aspect("equal")
            axes[row, col].set(
                title=f"E ∥ {pol}: {title}",
                xlabel="lab x=b (µm)",
                ylabel="lab y=a (µm)",
            )
            fig.colorbar(image, ax=axes[row, col], label="K")
    fig.suptitle("Thermal impact of retaining SiO2 optical absorption")
    fig.savefig(output / "SIO2_HEAT_SOURCE_THERMAL_IMPACT.png", dpi=190)
    plt.close(fig)


def case_row(
    pol: str,
    optical: dict[str, Any],
    baseline: dict[str, Any],
    combined: dict[str, Any],
) -> dict[str, Any]:
    run = optical["result"]["run_result"]
    resolved = run["material_resolved_absorption"]
    old = baseline["summary"]
    new = combined["summary"]
    old_edge = old["straight_edge_metrics"]
    new_edge = new["straight_edge_metrics"]
    material_power = new["mapping"]["material_resolved_source_target_power_W"]
    return {
        "polarization": pol,
        "P_Q_full_W_at_1_W_m2": run["P_Q_W"],
        "P_Q_TaIrTe4_W_at_1_W_m2": resolved["P_Q_TaIrTe4_exact_support_W"],
        "P_Q_SiO2_W_at_1_W_m2": resolved["P_Q_SiO2_exact_support_W"],
        "SiO2_fraction_of_Ta_plus_SiO2": resolved["P_Q_SiO2_exact_support_W"]
        / resolved["P_Q_TaIrTe4_plus_SiO2_W"],
        "six_face_closure": run["six_face_relative_closure"],
        "auto_shutoff": run["auto_shutoff"]["final_value"],
        "mapped_TaIrTe4_source_W_at_285uW": material_power["TaIrTe4_source_W"],
        "mapped_SiO2_source_W_at_285uW": material_power["SiO2_source_W"],
        "mapping_relative_power_error": new["mapping"]["mapping_relative_power_error"],
        "Tmax_Ta_only_K": old["thermal"]["Tmax_rise_K"],
        "Tmax_Ta_plus_SiO2_K": new["thermal"]["Tmax_rise_K"],
        "Tmax_relative_change": safe_relative(
            new["thermal"]["Tmax_rise_K"], old["thermal"]["Tmax_rise_K"]
        ),
        "Tavg_Ta_only_K": old["thermal"]["TaIrTe4_volume_average_rise_K"],
        "Tavg_Ta_plus_SiO2_K": new["thermal"]["TaIrTe4_volume_average_rise_K"],
        "Tavg_relative_change": safe_relative(
            new["thermal"]["TaIrTe4_volume_average_rise_K"],
            old["thermal"]["TaIrTe4_volume_average_rise_K"],
        ),
        "edge_P99_Ta_only_K_m": old_edge["p99_abs_edge_normal_gradient_K_m"],
        "edge_P99_Ta_plus_SiO2_K_m": new_edge[
            "p99_abs_edge_normal_gradient_K_m"
        ],
        "edge_P99_relative_change": safe_relative(
            new_edge["p99_abs_edge_normal_gradient_K_m"],
            old_edge["p99_abs_edge_normal_gradient_K_m"],
        ),
        "thermal_energy_balance": new["thermal"]["energy_balance_relative_error"],
        "thermal_linear_residual": new["thermal"]["linear_residual_relative"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for pol in ("a", "b"):
        parser.add_argument(f"--optical-{pol}-dir", type=Path, required=True)
        parser.add_argument(f"--baseline-thermal-{pol}-dir", type=Path, required=True)
        parser.add_argument(f"--combined-thermal-{pol}-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    optical = {
        pol: load_optical(getattr(args, f"optical_{pol}_dir").resolve())
        for pol in ("a", "b")
    }
    baseline = {
        pol: load_thermal(getattr(args, f"baseline_thermal_{pol}_dir").resolve())
        for pol in ("a", "b")
    }
    combined = {
        pol: load_thermal(getattr(args, f"combined_thermal_{pol}_dir").resolve())
        for pol in ("a", "b")
    }
    plot_sources_and_temperature(combined, args.output_dir)
    plot_thermal_change(baseline, combined, args.output_dir)

    cases = [
        case_row(pol, optical[pol], baseline[pol], combined[pol])
        for pol in ("a", "b")
    ]
    by_pol = {row["polarization"]: row for row in cases}
    ratios = {
        key: by_pol["b"][key] / by_pol["a"][key]
        for key in (
            "P_Q_TaIrTe4_W_at_1_W_m2",
            "P_Q_SiO2_W_at_1_W_m2",
            "Tmax_Ta_plus_SiO2_K",
            "Tavg_Ta_plus_SiO2_K",
            "edge_P99_Ta_plus_SiO2_K_m",
        )
    }
    gates = {
        "both_optical_cases_passed": all(
            optical[pol]["result"]["status"] == "COMPLETED" for pol in ("a", "b")
        ),
        "six_face_closure_lt_0p5_percent": all(
            by_pol[pol]["six_face_closure"] < 0.005 for pol in ("a", "b")
        ),
        "auto_shutoff_lt_1e_minus_5": all(
            by_pol[pol]["auto_shutoff"] <= 1e-5 for pol in ("a", "b")
        ),
        "mapping_error_lt_0p5_percent": all(
            by_pol[pol]["mapping_relative_power_error"] < 0.005
            for pol in ("a", "b")
        ),
        "energy_balance_lt_1_percent": all(
            by_pol[pol]["thermal_energy_balance"] < 0.01 for pol in ("a", "b")
        ),
        "linear_residual_lt_1e_minus_8": all(
            by_pol[pol]["thermal_linear_residual"] < 1e-8 for pol in ("a", "b")
        ),
    }
    passed = all(gates.values())
    summary = {
        "status": (
            "VALIDATED_MATERIAL_RESOLVED_TAIRTE4_PLUS_SIO2_THERMAL_SOURCE"
            if passed
            else "FAILED_MATERIAL_RESOLVED_TAIRTE4_PLUS_SIO2_THERMAL_SOURCE"
        ),
        "classification": (
            "straight-45-edge w0=8.75 um diagnostic; explicit expanded 3D FVM, "
            "not a paper reproduction"
        ),
        "source_contract": {
            "TaIrTe4": "full Maxwell volumetric Q on exact TaIrTe4 support",
            "SiO2": "full Maxwell volumetric Q on exact 285-nm SiO2 support",
            "mapping": "separate conservative remaps to the matching thermal materials",
            "forbidden_operations": [
                "clipping",
                "smoothing",
                "gain",
                "global rescaling",
                "tiling",
                "cross-material source projection",
            ],
        },
        "thermal_conductivity_contract": {
            "SiO2_W_mK": SIO2_K_W_MK,
            "Si_W_mK": SI_K_W_MK,
            "temperature_context": "named 300-K bulk-reference scenario",
            "paper_supplied_values": False,
            "warning": (
                "thin-film/process-dependent SiO2 and doped/device Si values "
                "require separate sensitivity; these are not promoted as measured values"
            ),
        },
        "cases": cases,
        "b_over_a_ratios": ratios,
        "gates": gates,
        "PTE_weighting_adjoint_optimization_run": False,
    }
    (args.output_dir / "full_sio2_thermal_source_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (args.output_dir / "full_sio2_thermal_source_cases.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)

    artifacts = []
    for family in (optical, baseline, combined):
        for pol in ("a", "b"):
            paths = (
                (family[pol]["result_path"], family[pol]["artifact_path"])
                if family is optical
                else (family[pol]["summary_path"], family[pol]["fields_path"])
            )
            for path in paths:
                artifacts.append(
                    {
                        "polarization": pol,
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    (args.output_dir / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(
            {
                "raw_artifacts_committed_to_git": False,
                "artifacts": artifacts,
                "plots_and_summaries_are_git_safe": True,
            },
            indent=2,
        )
        + "\n"
    )

    report = f"""# Material-resolved TaIrTe4 + SiO2 thermal-source control

Status: `{summary['status']}`

The Maxwell source spans the full 130-nm TaIrTe4 and 285-nm SiO2 layers.
The two unmodified volumetric sources are mapped independently into matching
thermal-material cells.  SiO2 absorption is never projected into TaIrTe4.

## Thermal conductivity assumptions

- SiO2: `{SIO2_K_W_MK} W/(m K)`.
- Si: `{SI_K_W_MK} W/(m K)`.

These are named 300-K bulk-reference scenario assumptions used by the expanded
FVM.  They are not values supplied by the TaIrTe4 paper and are not substitutes
for thin-film/process/doping sensitivity.

## Results

| metric | E||a | E||b |
|---|---:|---:|
| SiO2 fraction of modeled optical heat | {100*by_pol['a']['SiO2_fraction_of_Ta_plus_SiO2']:.6f}% | {100*by_pol['b']['SiO2_fraction_of_Ta_plus_SiO2']:.6f}% |
| Tmax, Ta-only | {by_pol['a']['Tmax_Ta_only_K']:.9e} K | {by_pol['b']['Tmax_Ta_only_K']:.9e} K |
| Tmax, Ta+SiO2 | {by_pol['a']['Tmax_Ta_plus_SiO2_K']:.9e} K | {by_pol['b']['Tmax_Ta_plus_SiO2_K']:.9e} K |
| Tmax change | {100*by_pol['a']['Tmax_relative_change']:.6f}% | {100*by_pol['b']['Tmax_relative_change']:.6f}% |
| strict-centred edge-gradient P99 change | {100*by_pol['a']['edge_P99_relative_change']:.6f}% | {100*by_pol['b']['edge_P99_relative_change']:.6f}% |

No PTE, weighting-potential, adjoint, AD-FD, or optimization calculation is part
of this control.
"""
    (args.output_dir / "FULL_SIO2_THERMAL_SOURCE_REPORT.md").write_text(report)
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
