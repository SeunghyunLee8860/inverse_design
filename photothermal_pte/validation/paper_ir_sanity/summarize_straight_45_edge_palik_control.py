#!/usr/bin/env python3
"""Summarize the corrected-substrate corner-free 45-degree edge control."""

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
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_case(directory: Path) -> dict[str, Any]:
    result_path = directory / "case_result.json"
    artifact_path = directory / "finite_q_on_artifact.npz"
    result = read_json(result_path)
    if result.get("status") != "COMPLETED":
        raise RuntimeError(f"optical case did not complete: {result_path}")
    return {
        "directory": directory,
        "result_path": result_path,
        "artifact_path": artifact_path,
        "result": result,
        "run": result["run_result"],
    }


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


def cell_centres(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(edges[:-1]) + np.asarray(edges[1:]))


def flake_areal_q(fields: dict[str, np.ndarray]) -> np.ndarray:
    return np.sum(
        fields["Q_W_m3"]
        * np.diff(fields["z_edges_m"])[None, None, :],
        axis=2,
    )


def plot_main(
    thermal: dict[str, dict[str, Any]],
    output: Path,
) -> None:
    rows: list[tuple[str, str, str]] = [
        ("Q", "absorbed areal power (W/m²)", "inferno"),
        ("T", "TaIrTe₄ thickness-averaged ΔT (K)", "inferno"),
        ("grad", "strict-centred |∂T/∂n| (K/m)", "magma"),
    ]
    arrays: dict[tuple[str, str], np.ndarray] = {}
    for pol in ("a", "b"):
        fields = thermal[pol]["fields"]
        mask = np.any(fields["flake_mask"], axis=2)
        arrays[(pol, "Q")] = np.where(mask, flake_areal_q(fields), np.nan)
        arrays[(pol, "T")] = np.where(
            mask, fields["temperature_flake_average_K"], np.nan
        )
        arrays[(pol, "grad")] = np.where(
            mask, np.abs(fields["grad_T_normal_K_m"]), np.nan
        )

    fig, axes = plt.subplots(3, 2, figsize=(12, 17), constrained_layout=True)
    for row, (key, label, cmap) in enumerate(rows):
        vmax = max(float(np.nanmax(arrays[(pol, key)])) for pol in ("a", "b"))
        for col, pol in enumerate(("a", "b")):
            fields = thermal[pol]["fields"]
            x_edges = fields["x_edges_m"] * 1e6
            y_edges = fields["y_edges_m"] * 1e6
            image = axes[row, col].pcolormesh(
                x_edges,
                y_edges,
                np.ma.masked_invalid(arrays[(pol, key)].T),
                cmap=cmap,
                vmin=0.0,
                vmax=vmax,
                shading="flat",
            )
            axes[row, col].set_aspect("equal")
            axes[row, col].plot(
                [x_edges[0], x_edges[-1]],
                [x_edges[0], x_edges[-1]],
                "--",
                color="cyan",
                lw=0.8,
            )
            axes[row, col].set(
                title=f"E ∥ {pol}: {label}",
                xlabel="lab x = b (µm)",
                ylabel="lab y = a (µm)",
            )
            fig.colorbar(image, ax=axes[row, col])
    fig.suptitle(
        "Corner-free 45° edge: Palik SiO₂/Si GPU FDTD Q → explicit 3D FVM\n"
        "w₀=8.75 µm; 100-nm optical/thermal lateral baseline; no weighting or PTE"
    )
    fig.savefig(output / "STRAIGHT_45_EDGE_PALIK_OPTICAL_THERMAL.png", dpi=190)
    plt.close(fig)


def diagonal_profile(
    values: np.ndarray,
    fields: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    x = cell_centres(fields["x_edges_m"])
    y = cell_centres(fields["y_edges_m"])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    normal = (-xx + yy) / np.sqrt(2.0)
    tangent = (xx + yy) / np.sqrt(2.0)
    step = max(float(np.max(np.diff(x))), float(np.max(np.diff(y))))
    selected = (
        np.any(fields["flake_mask"], axis=2)
        & np.isfinite(values)
        & (np.abs(tangent) <= 0.55 * step)
    )
    order = np.argsort(normal[selected])
    return normal[selected][order] * 1e6, values[selected][order]


def plot_profiles(
    thermal: dict[str, dict[str, Any]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for pol, color in (("a", "tab:blue"), ("b", "tab:orange")):
        fields = thermal[pol]["fields"]
        quantities = (
            (flake_areal_q(fields), "absorbed areal Q (W/m²)"),
            (fields["temperature_flake_average_K"], "ΔT (K)"),
            (np.abs(fields["grad_T_normal_K_m"]), "|∂T/∂n| (K/m)"),
        )
        for axis, (values, ylabel) in zip(axes, quantities):
            normal_um, profile = diagonal_profile(values, fields)
            axis.plot(normal_um, profile, label=f"E ∥ {pol}", color=color)
            axis.set(xlabel="edge-normal n (µm); TaIrTe₄ is n≤0", ylabel=ylabel)
            axis.grid(alpha=0.25)
    for axis in axes:
        axis.axvline(0.0, color="black", ls="--", lw=0.8)
        axis.legend()
    fig.suptitle("Centred edge-normal profiles (strict-centred gradient only)")
    fig.savefig(output / "STRAIGHT_45_EDGE_PALIK_PROFILES.png", dpi=190)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optical-a-dir", type=Path, required=True)
    parser.add_argument("--optical-b-dir", type=Path, required=True)
    parser.add_argument("--thermal-a-dir", type=Path, required=True)
    parser.add_argument("--thermal-b-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    optical = {
        "a": load_case(args.optical_a_dir.resolve()),
        "b": load_case(args.optical_b_dir.resolve()),
    }
    thermal = {
        "a": load_thermal(args.thermal_a_dir.resolve()),
        "b": load_thermal(args.thermal_b_dir.resolve()),
    }
    plot_main(thermal, args.output_dir)
    plot_profiles(thermal, args.output_dir)

    cases: list[dict[str, Any]] = []
    for pol in ("a", "b"):
        run = optical[pol]["run"]
        resolved = run["material_resolved_absorption"]
        thermal_summary = thermal[pol]["summary"]
        metrics = thermal_summary["straight_edge_metrics"]
        fields = thermal[pol]["fields"]
        gx = fields["grad_T_x_K_m"]
        gy = fields["grad_T_y_K_m"]
        gn = fields["grad_T_normal_K_m"]
        gt = fields["grad_T_tangent_K_m"]
        gradient_valid = (
            np.isfinite(gx) & np.isfinite(gy) & np.isfinite(gn) & np.isfinite(gt)
        )
        cartesian_norm_sq = gx[gradient_valid] ** 2 + gy[gradient_valid] ** 2
        rotated_norm_sq = gn[gradient_valid] ** 2 + gt[gradient_valid] ** 2
        rotation_identity_error = float(
            np.max(
                np.abs(cartesian_norm_sq - rotated_norm_sq)
                / np.maximum(cartesian_norm_sq, np.finfo(float).tiny)
            )
        )
        flake_xy = np.any(fields["flake_mask"], axis=2)
        cases.append(
            {
                "polarization": pol,
                "P_Q_full_control_volume_W_at_1_W_m2": run["P_Q_W"],
                "P_Q_TaIrTe4_W_at_1_W_m2": resolved[
                    "P_Q_TaIrTe4_exact_support_W"
                ],
                "P_Q_outside_TaIrTe4_W_at_1_W_m2": resolved[
                    "P_Q_outside_TaIrTe4_support_W"
                ],
                "P_six_W_at_1_W_m2": run["P_six_face_W"],
                "matched_volume_closure": run["six_face_relative_closure"],
                "auto_shutoff": run["auto_shutoff"]["final_value"],
                "P_abs_TaIrTe4_W_at_285uW_incident": thermal_summary["mapping"][
                    "P_Q_source_W"
                ],
                "Tmax_K": thermal_summary["thermal"]["Tmax_rise_K"],
                "Tavg_TaIrTe4_K": thermal_summary["thermal"][
                    "TaIrTe4_volume_average_rise_K"
                ],
                "strict_centered_max_abs_dTdn_K_m": metrics[
                    "max_abs_edge_normal_gradient_K_m"
                ],
                "strict_centered_p99_abs_dTdn_K_m": metrics[
                    "p99_abs_edge_normal_gradient_K_m"
                ],
                "thermal_energy_balance": thermal_summary["thermal"][
                    "energy_balance_relative_error"
                ],
                "thermal_linear_residual": thermal_summary["thermal"][
                    "linear_residual_relative"
                ],
                "mapping_power_error": thermal_summary["mapping"][
                    "mapping_relative_power_error"
                ],
                "gradient_rotation_identity_max_relative_error": (
                    rotation_identity_error
                ),
                "finite_gradient_samples_outside_flake": int(
                    np.count_nonzero(np.isfinite(gn) & ~flake_xy)
                ),
            }
        )
    by_pol = {case["polarization"]: case for case in cases}

    def ratio(key: str) -> float:
        return float(by_pol["b"][key] / by_pol["a"][key])

    ratios = {
        "P_Q_TaIrTe4_b_over_a": ratio("P_Q_TaIrTe4_W_at_1_W_m2"),
        "Tmax_b_over_a": ratio("Tmax_K"),
        "Tavg_b_over_a": ratio("Tavg_TaIrTe4_K"),
        "strict_centered_max_abs_dTdn_b_over_a": ratio(
            "strict_centered_max_abs_dTdn_K_m"
        ),
        "strict_centered_p99_abs_dTdn_b_over_a": ratio(
            "strict_centered_p99_abs_dTdn_K_m"
        ),
    }
    gates = {
        "both_optical_cases_completed": all(
            optical[pol]["result"]["status"] == "COMPLETED" for pol in ("a", "b")
        ),
        "matched_volume_closure_lt_0p5_percent": all(
            by_pol[pol]["matched_volume_closure"] < 0.005 for pol in ("a", "b")
        ),
        "auto_shutoff_lt_1e_minus_5": all(
            by_pol[pol]["auto_shutoff"] <= 1e-5 for pol in ("a", "b")
        ),
        "thermal_mapping_error_lt_0p5_percent": all(
            by_pol[pol]["mapping_power_error"] < 0.005 for pol in ("a", "b")
        ),
        "thermal_energy_balance_lt_1_percent": all(
            by_pol[pol]["thermal_energy_balance"] < 0.01 for pol in ("a", "b")
        ),
        "thermal_residual_lt_1e_minus_8": all(
            by_pol[pol]["thermal_linear_residual"] < 1e-8 for pol in ("a", "b")
        ),
        "gradient_rotation_identity_lt_1e_minus_12": all(
            by_pol[pol]["gradient_rotation_identity_max_relative_error"] < 1e-12
            for pol in ("a", "b")
        ),
        "no_finite_gradient_sample_outside_flake": all(
            by_pol[pol]["finite_gradient_samples_outside_flake"] == 0
            for pol in ("a", "b")
        ),
    }
    numerical_pass = all(gates.values())
    status = (
        "COMPLETED_CORRECTED_SUBSTRATE_STRAIGHT_45_EDGE_CONTROL"
        if numerical_pass
        else "FAILED_CORRECTED_SUBSTRATE_STRAIGHT_45_EDGE_CONTROL"
    )
    summary = {
        "status": status,
        "classification": (
            "corner-free straight-45-degree diagnostic with an explicitly "
            "assumed w0=8.75 um scalar Gaussian; not a paper reproduction"
        ),
        "geometry": {
            "TaIrTe4_support": "y<=x half-plane; remote triangle faces outside PML",
            "electrodes": False,
            "weighting_field": False,
            "PTE_current": False,
            "periodic": False,
            "boundaries": "six PML optical; explicit expanded thermal FVM",
        },
        "heat_source_scope": {
            "raw_optical_Q": (
                "all electromagnetic loss inside the matched local control volume; "
                "this includes the upper 50 nm of lossy Palik SiO2 plus interface "
                "samples, the 130 nm TaIrTe4 layer, and 50 nm top padding"
            ),
            "thermal_Q": "derived TaIrTe4-supported volumetric Q only",
            "SiO2_absorption_used_as_thermal_source": False,
            "reason": (
                "the matched closure volume samples only the upper 50 nm of the "
                "285 nm oxide and therefore is not a complete substrate-heating source"
            ),
        },
        "substrate_readback": optical["a"]["run"]["substrate_epsilon_readback"],
        "cases": cases,
        "ratios": ratios,
        "gates": gates,
        "paper_gradient_trend_b_over_a_gt_1": bool(
            ratios["strict_centered_p99_abs_dTdn_b_over_a"] > 1.0
        ),
        "optical_mesh_status": (
            "100-nm lateral baseline only; no 50-nm promotion is claimed by this report"
        ),
    }
    summary_path = args.output_dir / "straight_45_edge_palik_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    csv_path = args.output_dir / "straight_45_edge_palik_cases.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)

    artifacts: list[dict[str, Any]] = []
    for family in (optical, thermal):
        for pol in ("a", "b"):
            for key in (
                ("artifact_path", "result_path")
                if family is optical
                else ("fields_path", "summary_path")
            ):
                path = family[pol][key]
                artifacts.append(
                    {
                        "polarization": pol,
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    manifest_path = args.output_dir / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(
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

    report = f"""# Corrected-substrate corner-free 45° edge control

Status: `{status}`

This is a simple `y=x` half-plane control, not the Device-A polygon and not a
paper reproduction.  It uses the explicitly assumed scalar-Gaussian
`w0=8.75 um` scenario.  There are no electrodes, corners, weighting field,
PTE current, adjoint, or optimization.

## Optical/material contract

- Six PML faces; no periodic/Bloch boundaries.
- TaIrTe4: 130 nm; `epsilon_x=epsilon_b`, `epsilon_y=epsilon_a`,
  `epsilon_z=epsilon_b`.
- Substrate: 285 nm `SiO2 (Glass) - Palik` on `Si (Silicon) - Palik`.
- Full raw control-volume Q is preserved.  Thermal uses only the separately
  saved `Q_TaIrTe4_only_W_m3`; lossy substrate Q is not projected into the
  flake.
- No clipping, smoothing, gain, global rescaling, tiling, or polarization
  matching was used.

## FDTD geometry and source

- Domain: `x,y=[-30,+30] um`, `z=[-3.415,+10] um`.
- All six boundaries are PML with 24 layers; there is no periodic/Bloch face.
- Scalar-Gaussian source plane: `z=+5 um`, square aperture `50 x 50 um2`,
  propagation along `-z`, and focus/waist at the TaIrTe4 midplane `z=-65 nm`.
- Target-plane physical waist radius: `8.75 um`; the calibrated Lumerical
  source-object radius is `8.610602974768 um`.
- The complete top/front/side/layer view is
  [STRAIGHT_45_EDGE_FDTD_GEOMETRY_ALL_VIEWS.png](STRAIGHT_45_EDGE_FDTD_GEOMETRY_ALL_VIEWS.png).

## SiO2 optical loss versus thermal source

The answer is deliberately split.  The unmodified raw optical Q and its
six-face closure include every loss sample in `z=[-180,+50] nm`, including
the upper 50 nm of lossy Palik SiO2 and interface samples.  The present
thermal solve did **not** use that SiO2 loss: it used the separately saved
TaIrTe4-supported volumetric Q only.  The matched optical control volume
does not cover the lower 235 nm of the oxide, so it cannot be presented as
a complete explicit-substrate heating source.  A full-substrate thermal-Q
case requires a new material-resolved Q extraction spanning the entire
285 nm SiO2 (and any intended Si absorption), followed by conservative
mapping into those thermal materials.

## Results

| metric | E||a | E||b | b/a |
|---|---:|---:|---:|
| TaIrTe4 P_Q at 1 W/m2 | {by_pol['a']['P_Q_TaIrTe4_W_at_1_W_m2']:.9e} | {by_pol['b']['P_Q_TaIrTe4_W_at_1_W_m2']:.9e} | {ratios['P_Q_TaIrTe4_b_over_a']:.6f} |
| matched-volume closure | {100*by_pol['a']['matched_volume_closure']:.6f}% | {100*by_pol['b']['matched_volume_closure']:.6f}% | — |
| Tmax | {by_pol['a']['Tmax_K']:.9e} K | {by_pol['b']['Tmax_K']:.9e} K | {ratios['Tmax_b_over_a']:.6f} |
| TaIrTe4 average T | {by_pol['a']['Tavg_TaIrTe4_K']:.9e} K | {by_pol['b']['Tavg_TaIrTe4_K']:.9e} K | {ratios['Tavg_b_over_a']:.6f} |
| strict-centred P99 abs(dT/dn) | {by_pol['a']['strict_centered_p99_abs_dTdn_K_m']:.9e} K/m | {by_pol['b']['strict_centered_p99_abs_dTdn_K_m']:.9e} K/m | {ratios['strict_centered_p99_abs_dTdn_b_over_a']:.6f} |

The gradient is not evaluated on a staircase-edge cell.  A value is retained
only when all four `+x,-x,+y,-y` TaIrTe4 neighbours exist; otherwise it is
masked.  The comparator uses the nearest fully centred interior band.
The pixelwise coordinate-rotation identity has maximum relative errors
`{by_pol['a']['gradient_rotation_identity_max_relative_error']:.3e}` and
`{by_pol['b']['gradient_rotation_identity_max_relative_error']:.3e}` for
`E||a/b`, and there are no finite gradient samples outside the flake.

## Interpretation

Numerical pass: `{numerical_pass}`.  Paper-like b/a gradient trend: `{summary['paper_gradient_trend_b_over_a_gt_1']}`.
This 100-nm lateral run is a baseline.  It is not promoted as a mesh-converged
production optical Q without a targeted refinement comparison.
"""
    (args.output_dir / "STRAIGHT_45_EDGE_PALIK_CONTROL_REPORT.md").write_text(report)
    print(json.dumps(summary, indent=2))
    return 0 if numerical_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
