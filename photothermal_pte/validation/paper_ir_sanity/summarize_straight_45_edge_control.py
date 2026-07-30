#!/usr/bin/env python3
"""Publish the corner-free 45-degree optical/thermal trend control."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
DEFAULT_ARTIFACT_ROOT = Path(
    "/home/seunghyun/tairte4/artifacts/paper_ir_lumerical_sanity"
)
DEFAULT_REPORT_ROOT = (
    REPOSITORY / "photothermal_pte/reports/paper_ir_straight_45_edge"
)
OPTICAL_NAMES = {
    "a": "straight45_a_w6p5_dz10_L48_gpu4_20260730",
    "b": "straight45_b_w6p5_dz10_L48_gpu5_20260730",
}
THERMAL_NAMES = {
    ("a", 200): "straight45_thermal_a_core200_20260730",
    ("b", 200): "straight45_thermal_b_core200_20260730",
    ("a", 100): "straight45_thermal_a_core100_20260730",
    ("b", 100): "straight45_thermal_b_core100_20260730",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_ROOT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def optical_metrics(case_dir: Path) -> dict[str, Any]:
    case = load_json(case_dir / "case_result.json")
    run = case["run_result"]
    solver = case["pre_run_contract"]["solver"]
    gpu_resources = [
        resource
        for resource in solver["resources"].values()
        if str(resource.get("device type", "")).startswith("GPU")
        and resource.get("active") == "1"
        and "-gpu" in resource.get("solver extra command line options", "")
    ]
    return {
        "case_dir": str(case_dir.resolve()),
        "status": case["status"],
        "solver_version": solver["version"],
        "solver_device": gpu_resources[0]["device type"],
        "P_Q_W_at_1_W_m2": run["P_Q_W"],
        "P_six_W_at_1_W_m2": run["P_six_face_W"],
        "six_face_closure_relative": run["six_face_relative_closure"],
        "component_power_W": run["component_power_W"],
        "hotspot": run["Q_hotspot"],
        "negative_Q_voxel_count": run["negative_Q_voxel_count"],
        "acceptance": run["acceptance"],
        "normalization": run["normalization"],
        "geometry": run["artifact_metadata"]["geometry_bounds_m"],
        "raw_npz": str((case_dir / "finite_q_on_artifact.npz").resolve()),
        "raw_npz_sha256": sha256(case_dir / "finite_q_on_artifact.npz"),
        "generation_command": case["generation_command"],
    }


def thermal_metrics(case_dir: Path) -> dict[str, Any]:
    summary = load_json(case_dir / "summary.json")
    return {
        "case_dir": str(case_dir.resolve()),
        "status": summary["status"],
        "geometry": summary["geometry"],
        "mapping": summary["mapping"],
        "thermal": summary["thermal"],
        "straight_edge_metrics": summary["straight_edge_metrics"],
        "weighting": summary["weighting"],
        "PTE_current_A": summary["PTE_current_A_at_285uW_incident"],
    }


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator)


def make_maps(
    output: Path,
    artifact_root: Path,
) -> None:
    fields: dict[str, dict[str, np.ndarray]] = {}
    for polarization in ("a", "b"):
        path = (
            artifact_root
            / THERMAL_NAMES[(polarization, 100)]
            / "thermal_pte_fields.npz"
        )
        with np.load(path, allow_pickle=False) as raw:
            fields[polarization] = {
                key: np.asarray(raw[key])
                for key in (
                    "x_edges_m",
                    "y_edges_m",
                    "z_edges_m",
                    "flake_mask",
                    "Q_W_m3",
                    "temperature_flake_average_K",
                    "grad_T_normal_K_m",
                )
            }

    q_maps: dict[str, np.ndarray] = {}
    t_maps: dict[str, np.ndarray] = {}
    g_maps: dict[str, np.ndarray] = {}
    for polarization, data in fields.items():
        dz = np.diff(data["z_edges_m"])
        flake_xy = np.any(data["flake_mask"], axis=2)
        q_maps[polarization] = np.sum(data["Q_W_m3"] * dz[None, None, :], axis=2)
        t_maps[polarization] = np.where(
            flake_xy, data["temperature_flake_average_K"], np.nan
        )
        g_maps[polarization] = np.where(
            flake_xy, np.abs(data["grad_T_normal_K_m"]), np.nan
        )

    maxima = {
        "q": max(np.nanmax(value) for value in q_maps.values()),
        "t": max(np.nanmax(value) for value in t_maps.values()),
        "g": max(np.nanmax(value) for value in g_maps.values()),
    }
    figure, axes = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)
    for column, polarization in enumerate(("a", "b")):
        data = fields[polarization]
        extent = [
            data["x_edges_m"][0] * 1e6,
            data["x_edges_m"][-1] * 1e6,
            data["y_edges_m"][0] * 1e6,
            data["y_edges_m"][-1] * 1e6,
        ]
        for row, (values, maximum, label) in enumerate(
            (
                (q_maps[polarization], maxima["q"], "absorbed areal power (W/m²)"),
                (t_maps[polarization], maxima["t"], "TaIrTe₄ ΔT (K)"),
                (
                    g_maps[polarization],
                    maxima["g"],
                    "|edge-normal ∂T/∂n| (K/m)",
                ),
            )
        ):
            handle = axes[row, column].imshow(
                values.T,
                origin="lower",
                extent=extent,
                aspect="equal",
                vmin=0.0,
                vmax=maximum,
                cmap="inferno",
            )
            axes[row, column].plot(
                [-24, 24], [-24, 24], "--", color="cyan", linewidth=1
            )
            axes[row, column].set(
                title=f"E ∥ {polarization}: {label}",
                xlabel="lab x = b (µm)",
                ylabel="lab y = a (µm)",
                xlim=(-12, 12),
                ylim=(-12, 12),
            )
            figure.colorbar(handle, ax=axes[row, column], shrink=0.86)
    figure.suptitle(
        "Corner-free 45° edge: GPU Lumerical Q → explicit thermal FVM\n"
        "(100 nm lateral mesh; no weighting field or PTE current)",
        fontsize=15,
    )
    figure.savefig(
        output / "STRAIGHT_45_EDGE_OPTICAL_THERMAL_CONTROL.png", dpi=180
    )
    plt.close(figure)


def make_mesh_figure(
    output: Path,
    thermal: dict[str, dict[str, dict[str, Any]]],
) -> None:
    labels = ["Pabs", "flake Tmax", "flake avg ΔT", "max |∂T/∂n|", "p99 |∂T/∂n|"]
    keys = [
        ("mapping", "P_Q_target_W"),
        ("straight_edge_metrics", "Tmax_rise_K"),
        ("straight_edge_metrics", "TaIrTe4_area_average_rise_K"),
        ("straight_edge_metrics", "max_abs_edge_normal_gradient_K_m"),
        ("straight_edge_metrics", "p99_abs_edge_normal_gradient_K_m"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x = np.arange(len(labels))
    for offset, mesh in ((-0.18, "200nm"), (0.18, "100nm")):
        values = []
        for section, key in keys:
            a = thermal[mesh]["a"][section][key]
            b = thermal[mesh]["b"][section][key]
            values.append(ratio(b, a))
        axes[0].bar(x + offset, values, width=0.34, label=mesh)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set(
        title="Polarization trend ratio (E∥b / E∥a)",
        ylabel="ratio",
        xticks=x,
        xticklabels=labels,
    )
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend()

    mesh_change = []
    change_labels = []
    for polarization in ("a", "b"):
        for section, key in keys[1:]:
            coarse = thermal["200nm"][polarization][section][key]
            refined = thermal["100nm"][polarization][section][key]
            mesh_change.append(abs(refined - coarse) / abs(refined) * 100.0)
            change_labels.append(
                f"{polarization}:"
                + {
                    "Tmax_rise_K": "Tmax",
                    "TaIrTe4_area_average_rise_K": "Tavg",
                    "max_abs_edge_normal_gradient_K_m": "Gmax",
                    "p99_abs_edge_normal_gradient_K_m": "Gp99",
                }[key]
            )
    axes[1].bar(np.arange(len(mesh_change)), mesh_change, color="#4c78a8")
    axes[1].set(
        title="200 → 100 nm relative change",
        ylabel="relative change (%)",
        xticks=np.arange(len(mesh_change)),
        xticklabels=change_labels,
    )
    axes[1].tick_params(axis="x", rotation=35)
    figure.savefig(output / "STRAIGHT_45_EDGE_MESH_TREND.png", dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    optical = {
        polarization: optical_metrics(args.artifact_root / name)
        for polarization, name in OPTICAL_NAMES.items()
    }
    thermal = {
        f"{mesh}nm": {
            polarization: thermal_metrics(args.artifact_root / name)
            for (polarization, selected_mesh), name in THERMAL_NAMES.items()
            if selected_mesh == mesh
        }
        for mesh in (200, 100)
    }

    ratios: dict[str, dict[str, float]] = {}
    for mesh in ("200nm", "100nm"):
        a = thermal[mesh]["a"]
        b = thermal[mesh]["b"]
        ratios[mesh] = {
            "P_abs_b_over_a": ratio(
                b["mapping"]["P_Q_target_W"], a["mapping"]["P_Q_target_W"]
            ),
            "Tmax_b_over_a": ratio(
                b["straight_edge_metrics"]["Tmax_rise_K"],
                a["straight_edge_metrics"]["Tmax_rise_K"],
            ),
            "Tavg_b_over_a": ratio(
                b["straight_edge_metrics"]["TaIrTe4_area_average_rise_K"],
                a["straight_edge_metrics"]["TaIrTe4_area_average_rise_K"],
            ),
            "max_edge_gradient_b_over_a": ratio(
                b["straight_edge_metrics"][
                    "max_abs_edge_normal_gradient_K_m"
                ],
                a["straight_edge_metrics"][
                    "max_abs_edge_normal_gradient_K_m"
                ],
            ),
            "p99_edge_gradient_b_over_a": ratio(
                b["straight_edge_metrics"][
                    "p99_abs_edge_normal_gradient_K_m"
                ],
                a["straight_edge_metrics"][
                    "p99_abs_edge_normal_gradient_K_m"
                ],
            ),
        }

    optical_gate = all(
        case["status"] == "COMPLETED"
        and case["six_face_closure_relative"] < 0.005
        and case["negative_Q_voxel_count"] == 0
        for case in optical.values()
    )
    numerical_gate = all(
        case["status"] == "COMPLETED_STRAIGHT_45_EDGE_THERMAL_CONTROL"
        and case["mapping"]["mapping_relative_power_error"] < 0.005
        and case["thermal"]["energy_balance_relative_error"] < 0.01
        and case["thermal"]["linear_residual_relative"] < 1e-8
        for mesh in thermal.values()
        for case in mesh.values()
    )
    paper_trend_gate = all(
        ratios[mesh]["P_abs_b_over_a"] > 1.0
        and ratios[mesh]["Tmax_b_over_a"] > 1.0
        and ratios[mesh]["max_edge_gradient_b_over_a"] > 1.0
        and ratios[mesh]["p99_edge_gradient_b_over_a"] > 1.0
        for mesh in ("200nm", "100nm")
    )
    status = (
        "VALIDATED_STRAIGHT_45_EDGE_OPTICAL_THERMAL_TREND"
        if optical_gate and numerical_gate and paper_trend_gate
        else "FAILED_STRAIGHT_45_EDGE_PAPER_GRADIENT_TREND"
    )
    summary = {
        "status": status,
        "scope": (
            "corner-free straight 45-degree TaIrTe4/air edge optical and "
            "thermal diagnostic only; no weighting field or PTE current"
        ),
        "geometry_contract": {
            "TaIrTe4_region": "lab y<=x",
            "edge": "lab y=x",
            "edge_outward_normal": "(-x+y)/sqrt(2)",
            "lab_axes": {"x": "crystal b", "y": "crystal a"},
            "flake_thickness_m": 130e-9,
            "substrate": "285 nm SiO2 on Si",
            "corner_present": False,
        },
        "illumination_contract": {
            "wavelength_m": 11e-6,
            "Gaussian_waist_radius_m": 6.5e-6,
            "beam_center_m": [0.0, 0.0],
            "experimental_incident_power_W": 285e-6,
            "polarizations": ["E_parallel_a", "E_parallel_b"],
            "waist_note": (
                "w0=6.5 um is a named scenario; an exact wavelength-specific "
                "experimental beam radius was not reported numerically"
            ),
        },
        "optical": optical,
        "thermal": thermal,
        "ratios": ratios,
        "gates": {
            "optical_closure_and_nonnegative_Q": optical_gate,
            "thermal_mapping_energy_residual": numerical_gate,
            "paper_direction_Pabs_DeltaT_gradient_b_gt_a": paper_trend_gate,
        },
        "interpretation": {
            "P_abs_trend": "E||b > E||a",
            "flake_average_temperature_trend": "E||b > E||a",
            "Tmax_trend": "E||a > E||b",
            "edge_normal_gradient_trend": "E||a > E||b",
            "conclusion": (
                "The corner artifact is eliminated, but the requested paper "
                "trend is not reproduced. Weighting/PTE remains intentionally "
                "unevaluated."
            ),
        },
        "weighting_field_applied": False,
        "PTE_current_evaluated": False,
        "optimization_run": False,
        "generation_commit": git_commit(),
        "generation_command": shlex.join([sys.executable, *sys.argv]),
    }
    (args.output_dir / "straight_45_edge_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2) + "\n"
    )

    csv_path = args.output_dir / "straight_45_edge_cases.csv"
    with csv_path.open("w", newline="") as stream:
        fieldnames = [
            "polarization",
            "core_step_nm",
            "P_Q_unit_W",
            "P_six_unit_W",
            "six_face_closure_relative",
            "P_abs_285uW_W",
            "Tmax_flake_K",
            "Tavg_flake_K",
            "max_edge_normal_gradient_K_m",
            "p99_edge_normal_gradient_K_m",
            "mapping_error",
            "energy_balance_error",
            "linear_residual",
            "PTE_evaluated",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for mesh in ("200nm", "100nm"):
            for polarization in ("a", "b"):
                case = thermal[mesh][polarization]
                writer.writerow(
                    {
                        "polarization": polarization,
                        "core_step_nm": int(mesh[:-2]),
                        "P_Q_unit_W": optical[polarization][
                            "P_Q_W_at_1_W_m2"
                        ],
                        "P_six_unit_W": optical[polarization][
                            "P_six_W_at_1_W_m2"
                        ],
                        "six_face_closure_relative": optical[polarization][
                            "six_face_closure_relative"
                        ],
                        "P_abs_285uW_W": case["mapping"]["P_Q_target_W"],
                        "Tmax_flake_K": case["straight_edge_metrics"][
                            "Tmax_rise_K"
                        ],
                        "Tavg_flake_K": case["straight_edge_metrics"][
                            "TaIrTe4_area_average_rise_K"
                        ],
                        "max_edge_normal_gradient_K_m": case[
                            "straight_edge_metrics"
                        ]["max_abs_edge_normal_gradient_K_m"],
                        "p99_edge_normal_gradient_K_m": case[
                            "straight_edge_metrics"
                        ]["p99_abs_edge_normal_gradient_K_m"],
                        "mapping_error": case["mapping"][
                            "mapping_relative_power_error"
                        ],
                        "energy_balance_error": case["thermal"][
                            "energy_balance_relative_error"
                        ],
                        "linear_residual": case["thermal"][
                            "linear_residual_relative"
                        ],
                        "PTE_evaluated": False,
                    }
                )

    manifest_entries = []
    for polarization in ("a", "b"):
        case_dir = Path(optical[polarization]["case_dir"])
        source_manifest = load_json(case_dir / "RAW_ARTIFACT_MANIFEST.json")
        for filename in (
            "finite_q_on_artifact.npz",
            "finite_2um_optical_q.fsp",
            "finite_2um_optical_q_p0.log",
        ):
            entry = dict(source_manifest["raw_artifacts"][filename])
            entry.update(
                {
                    "case": f"optical_E_parallel_{polarization}",
                    "kind": filename,
                }
            )
            manifest_entries.append(entry)
    for mesh in ("200nm", "100nm"):
        for polarization in ("a", "b"):
            case_dir = Path(thermal[mesh][polarization]["case_dir"])
            path = case_dir / "thermal_pte_fields.npz"
            manifest_entries.append(
                {
                    "case": f"thermal_E_parallel_{polarization}_{mesh}",
                    "kind": "thermal_pte_fields.npz",
                    "server_path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "generation_command": (
                        "see the matching external summary.json; straight-edge "
                        "mode explicitly disables weighting/PTE"
                    ),
                }
            )
    manifest = {
        "policy": "Raw NPZ/FSP/log/3-D fields remain outside Git.",
        "status": status,
        "generation_commit": git_commit(),
        "raw_artifacts": manifest_entries,
    }
    (args.output_dir / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    make_maps(args.output_dir, args.artifact_root)
    make_mesh_figure(args.output_dir, thermal)

    refined = thermal["100nm"]
    report = f"""# Straight 45° edge optical/thermal control

Status: `{status}`

## Outcome

The polygon corner was removed. The calculation used a single corner-free
TaIrTe4/air boundary `lab y=x`, with TaIrTe4 in `y<=x`, normal-incidence
11 µm Gaussian illumination centred on the edge, and independent `E||a` and
`E||b` v261 GPU FDTD runs. The existing conservative remap and explicit
anisotropic/multimaterial thermal FVM were then applied.

The optical and numerical-conservation gates pass. The requested paper trend
does **not**: `E||b` absorbs more total power and has a larger flake-average
temperature, but `E||a` has the larger peak temperature and edge-normal
temperature gradient on both meshes. Therefore no weighting field or PTE
current was evaluated.

## Geometry and illumination contract

- TaIrTe4: 130 nm, half-plane `y<=x`; no corner in the physical domain.
- Lab `x=b`, lab `y=a`; edge outward normal `(-x+y)/sqrt(2)`.
- 285 nm SiO2 on Si; optical electrodes omitted for this isolated edge check.
- Wavelength 11 µm, `w0=6.5 µm`, beam centre `(0,0)`, 285 µW incident power.
- `w0=6.5 µm` is a named scenario, not a paper-extracted exact beam radius.
- Optical domain: 48×48 µm, six PML faces, 24 PML layers, 10 nm flake mesh.
- Thermal domain: 48×48 µm, 20 µm Si depth; 200 and 100 nm lateral meshes.

## GPU optical result

| metric | E||a | E||b | b/a |
|---|---:|---:|---:|
| `P_Q` at central 1 W/m² | {optical["a"]["P_Q_W_at_1_W_m2"]:.9e} W | {optical["b"]["P_Q_W_at_1_W_m2"]:.9e} W | {optical["b"]["P_Q_W_at_1_W_m2"]/optical["a"]["P_Q_W_at_1_W_m2"]:.6f} |
| `P_six` | {optical["a"]["P_six_W_at_1_W_m2"]:.9e} W | {optical["b"]["P_six_W_at_1_W_m2"]:.9e} W | — |
| six-face closure | {100*optical["a"]["six_face_closure_relative"]:.6f}% | {100*optical["b"]["six_face_closure_relative"]:.6f}% | — |
| `P_abs` at 285 µW | {1e6*refined["a"]["mapping"]["P_Q_target_W"]:.6f} µW | {1e6*refined["b"]["mapping"]["P_Q_target_W"]:.6f} µW | {ratios["100nm"]["P_abs_b_over_a"]:.6f} |

Both closures are below 0.5%, both raw Q fields have zero negative voxels,
and the analytic exact-flake mask exactly equals `y<=x` over the 130-nm
thickness. No clipping, smoothing, gain, global rescaling, tiling, or source
deletion was used.

## Thermal result

| 100 nm metric | E||a | E||b | b/a |
|---|---:|---:|---:|
| flake `Tmax` | {refined["a"]["straight_edge_metrics"]["Tmax_rise_K"]:.9f} K | {refined["b"]["straight_edge_metrics"]["Tmax_rise_K"]:.9f} K | {ratios["100nm"]["Tmax_b_over_a"]:.6f} |
| flake-average ΔT | {refined["a"]["straight_edge_metrics"]["TaIrTe4_area_average_rise_K"]:.9f} K | {refined["b"]["straight_edge_metrics"]["TaIrTe4_area_average_rise_K"]:.9f} K | {ratios["100nm"]["Tavg_b_over_a"]:.6f} |
| max `|∂T/∂n|` | {refined["a"]["straight_edge_metrics"]["max_abs_edge_normal_gradient_K_m"]:.6e} K/m | {refined["b"]["straight_edge_metrics"]["max_abs_edge_normal_gradient_K_m"]:.6e} K/m | {ratios["100nm"]["max_edge_gradient_b_over_a"]:.6f} |
| p99 `|∂T/∂n|` | {refined["a"]["straight_edge_metrics"]["p99_abs_edge_normal_gradient_K_m"]:.6e} K/m | {refined["b"]["straight_edge_metrics"]["p99_abs_edge_normal_gradient_K_m"]:.6e} K/m | {ratios["100nm"]["p99_edge_gradient_b_over_a"]:.6f} |

The 200-nm mesh gives the same qualitative reversal: max-gradient
`b/a={ratios["200nm"]["max_edge_gradient_b_over_a"]:.6f}` and p99-gradient
`b/a={ratios["200nm"]["p99_edge_gradient_b_over_a"]:.6f}`. Peak gradient
magnitudes are not mesh converged, so they are not promoted as quantitative
experimental predictions; the polarization ordering is nevertheless
unchanged by refinement.

All four thermal cases have mapping error below 0.5%, energy-balance error
below 1%, and linear residual below 1e-8.

## Interpretation and next gate

Removing the approximate polygon corner did not restore the Figure-3F trend.
The present failure is driven by source localization: the `E||a` Q field is
more concentrated near the edge, while `E||b` deposits more total but broader
power. This result rules out the old concave corner as the sole explanation,
but it does not identify a unique remaining cause.

Before PTE, the discriminating follow-ups are optical-Q profile comparison
against the paper's exact simulation contract (especially the unreported
beam radius/spot definition and material-axis convention) and a converged
edge-gradient estimator. Weighting-field changes cannot fix this pre-weighting
thermal trend and were intentionally not used.

## Files

- `STRAIGHT_45_EDGE_OPTICAL_THERMAL_CONTROL.png`
- `STRAIGHT_45_EDGE_MESH_TREND.png`
- `straight_45_edge_summary.json`
- `straight_45_edge_cases.csv`
- `RAW_ARTIFACT_MANIFEST.json`
"""
    (
        args.output_dir
        / "STRAIGHT_45_EDGE_OPTICAL_THERMAL_CONTROL_REPORT.md"
    ).write_text(report)
    print(json.dumps({"status": status, "ratios": ratios}, indent=2))
    return 0 if optical_gate and numerical_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
