#!/usr/bin/env python3
"""Publish the four Run-002 CUDA thermal/PTE scenario controls."""

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


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PLOTS = HERE / "plots"
MANIFEST = HERE / "manifests" / "RAW_ARTIFACT_MANIFEST.json"
SCENARIOS = (
    "grown_grown",
    "grown_evaporated",
    "evaporated_grown",
    "evaporated_evaporated",
)
LABELS = {
    "grown_grown": "grown / grown",
    "grown_evaporated": "grown / evap.",
    "evaporated_grown": "evap. / grown",
    "evaporated_evaporated": "evap. / evap.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def parse_case(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("case must be SCENARIO=/absolute/raw/directory")
    name, directory = value.split("=", 1)
    if name not in SCENARIOS:
        raise argparse.ArgumentTypeError(f"unknown scenario {name!r}")
    return name, Path(directory).expanduser().resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--mapped-q", type=Path, required=True)
    parser.add_argument("--mapped-q-sha256", required=True)
    parser.add_argument("--incident-power-w", type=float, required=True)
    parser.add_argument("--preserved-failed-diagnostic", type=Path)
    args = parser.parse_args()

    case_directories = dict(args.case)
    if set(case_directories) != set(SCENARIOS) or len(args.case) != len(SCENARIOS):
        raise RuntimeError("exactly one raw directory is required for each of the four scenarios")
    if not np.isfinite(args.incident_power_w) or args.incident_power_w <= 0:
        raise RuntimeError("incident power must be finite and positive")

    mapped_q = args.mapped_q.expanduser().resolve()
    if sha256(mapped_q) != args.mapped_q_sha256:
        raise RuntimeError("mapped-Q SHA mismatch")
    q_data = np.load(mapped_q)
    flake_mask = np.asarray(q_data["mask_physical_TaIrTe4"], bool)
    x_edges, y_edges, z_edges = (
        np.asarray(q_data[f"{axis}_edges_m"], float) for axis in "xyz"
    )
    dz = np.diff(z_edges)
    x_um = 0.5 * (x_edges[:-1] + x_edges[1:]) * 1e6
    y_um = 0.5 * (y_edges[:-1] + y_edges[1:]) * 1e6

    rows: list[dict[str, object]] = []
    raw_records: list[dict[str, object]] = []
    maps: dict[str, np.ndarray] = {}
    for scenario in SCENARIOS:
        directory = case_directories[scenario]
        result_path = directory / "production_cuda_thermal_pte_result.json"
        result = json.loads(result_path.read_text())
        if result.get("scenario") != scenario or not result.get("passed", False):
            raise RuntimeError(f"scenario did not pass: {scenario}")
        artifact = result["artifact"]
        artifact_path = Path(artifact["path"])
        if (
            artifact_path.stat().st_size != artifact["size_bytes"]
            or sha256(artifact_path) != artifact["sha256"]
        ):
            raise RuntimeError(f"temperature artifact provenance mismatch: {scenario}")
        temperature = np.load(artifact_path)
        theta = np.asarray(temperature["theta_K"], float)
        for axis, reference in zip("xyz", (x_edges, y_edges, z_edges)):
            if not np.array_equal(np.asarray(temperature[f"{axis}_edges_m"]), reference):
                raise RuntimeError(f"temperature/Q grid mismatch for {scenario}, axis {axis}")

        flake_values = theta[flake_mask]
        weighted_numerator = np.sum(np.where(flake_mask, theta, 0.0) * dz[None, None, :], axis=2)
        weighted_denominator = np.sum(flake_mask * dz[None, None, :], axis=2)
        maps[scenario] = np.divide(
            weighted_numerator,
            weighted_denominator,
            out=np.full(weighted_denominator.shape, np.nan),
            where=weighted_denominator > 0,
        )
        boundaries = result["boundary_power_W"]
        source_power = float(result["source_power_W"])
        reciprocity_difference = abs(
            float(result["PTE_current_A"]) - float(result["PTE_reciprocal_A"])
        )
        reciprocity_scale = reciprocity_difference / max(
            float(result["reciprocity_Cauchy_normalized_error"]),
            np.finfo(float).tiny,
        )
        rows.append(
            {
                "scenario": scenario,
                "bottom_interface": scenario.split("_", 1)[0],
                "design_interface": scenario.split("_", 1)[1],
                "source_power_W": source_power,
                "incident_power_W": args.incident_power_w,
                "Tmax_all_materials_rise_K": float(result["Tmax_rise_K"]),
                "Tmax_TaIrTe4_rise_K": float(np.nanmax(flake_values)),
                "Tavg_TaIrTe4_rise_K": float(np.nanmean(flake_values)),
                "PTE_current_A": float(result["PTE_current_A"]),
                "PTE_current_A_per_W_incident": float(result["PTE_current_A"])
                / args.incident_power_w,
                "forward_iterations": int(result["forward"]["iterations"]),
                "forward_residual": float(result["forward"]["residual"]),
                "forward_seconds": float(result["forward"]["seconds"]),
                "adjoint_iterations": int(result["adjoint"]["iterations"]),
                "adjoint_residual": float(result["adjoint"]["residual"]),
                "adjoint_seconds": float(result["adjoint"]["seconds"]),
                "energy_balance_error": float(result["energy_balance_error"]),
                "reciprocity_raw_near_null_relative_error": float(
                    result["reciprocity_raw_near_null_relative_error"]
                ),
                "reciprocity_Cauchy_normalized_error": float(
                    result["reciprocity_Cauchy_normalized_error"]
                ),
                "PTE_signal_Cauchy_scale_fraction": abs(float(result["PTE_current_A"]))
                / max(reciprocity_scale, np.finfo(float).tiny),
                "lateral_boundary_power_fraction": float(
                    boundaries["x_min"]
                    + boundaries["x_max"]
                    + boundaries["y_min"]
                    + boundaries["y_max"]
                )
                / source_power,
                "bottom_boundary_power_fraction": float(boundaries["z_min"])
                / source_power,
                "exposed_surface_power_fraction": float(
                    boundaries["material_specific_exposed"]
                )
                / source_power,
                "temperature_artifact_path": str(artifact_path),
                "temperature_artifact_size_bytes": artifact["size_bytes"],
                "temperature_artifact_sha256": artifact["sha256"],
            }
        )
        raw_records.append(
            {
                "scenario": scenario,
                "raw_directory": str(directory),
                "artifacts": [artifact_record(result_path), artifact_record(artifact_path)],
            }
        )

    worst = {
        "forward_residual": max(float(row["forward_residual"]) for row in rows),
        "adjoint_residual": max(float(row["adjoint_residual"]) for row in rows),
        "energy_balance_error": max(float(row["energy_balance_error"]) for row in rows),
        "reciprocity_Cauchy_normalized_error": max(
            float(row["reciprocity_Cauchy_normalized_error"]) for row in rows
        ),
    }
    passed = (
        max(worst["forward_residual"], worst["adjoint_residual"]) < 1e-8
        and worst["energy_balance_error"] < 0.01
        and worst["reciprocity_Cauchy_normalized_error"] < 1e-8
    )
    status = (
        "VALIDATED_PRODUCTION_CUDA_THERMAL_PTE_SCENARIOS"
        if passed
        else "FAILED_PRODUCTION_CUDA_THERMAL_PTE_SCENARIOS"
    )

    diagnostic: dict[str, object] | None = None
    if args.preserved_failed_diagnostic:
        directory = args.preserved_failed_diagnostic.expanduser().resolve()
        result_path = directory / "production_cuda_thermal_pte_result.json"
        diagnostic_result = json.loads(result_path.read_text())
        if diagnostic_result.get("passed") is not False:
            raise RuntimeError("preserved diagnostic is not the expected failed result")
        diagnostic = {
            "role": "preserved_near_null_relative_metric_failure",
            "reason": (
                "The raw relative difference divided by two nearly-zero PTE currents; "
                "no empirical normalization or gradient rescaling was introduced."
            ),
            "raw_directory": str(directory),
            "artifacts": [artifact_record(path) for path in sorted(directory.iterdir()) if path.is_file()],
        }

    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / "production_cuda_thermal_pte_cases.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": status,
        "passed": passed,
        "scope": (
            "rho=0.5 production 3D CUDA thermal forward/implicit-adjoint controls; "
            "no Maxwell adjoint, AD-FD, design-window selection, or optimization"
        ),
        "mapped_Q": artifact_record(mapped_q),
        "incident_power_W": args.incident_power_w,
        "thermal_operator": {
            "grid_shape_xyz": list(flake_mask.shape),
            "active_unknowns": int(json.loads((case_directories[SCENARIOS[0]] / "production_cuda_thermal_pte_result.json").read_text())["active_unknowns"]),
            "linear_solve_device": "CUDA-only float64",
            "CPU_linear_solve_fallback": False,
            "external_boundaries": (
                "far x/y and Si bottom Dirichlet at 300 K; every exposed solid/air "
                "face uses material-specific Robin conductance"
            ),
            "weighting_field_m_inv": [15625.0, 15625.0],
            "weighting_model": "uniform 45-degree surrogate, not full electrodes",
        },
        "gates": {
            "linear_residual_max": 1e-8,
            "thermal_energy_balance_max": 0.01,
            "Cauchy_normalized_reciprocity_max": 1e-8,
            "observed_worst": worst,
        },
        "near_null_interpretation": {
            "rho05_baseline_is_center_symmetric": True,
            "PTE_current_is_a_null_control_not_an_optimization_performance_claim": True,
            "raw_near_null_relative_error_is_diagnostic_only": True,
            "empirical_normalization_or_gradient_rescaling": False,
        },
        "cases": rows,
        "preserved_failed_diagnostic": diagnostic,
        "optimization_iterations": 0,
    }
    summary_path = RESULTS / "production_cuda_thermal_pte_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    labels = [LABELS[name] for name in SCENARIOS]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].bar(labels, [float(row["Tmax_all_materials_rise_K"]) * 1e9 for row in rows], label="all-material Tmax")
    axes[0, 0].bar(labels, [float(row["Tmax_TaIrTe4_rise_K"]) * 1e9 for row in rows], label="TaIrTe4 Tmax", alpha=0.75)
    axes[0, 0].set_ylabel("temperature rise (nK)")
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 1].bar(labels, [float(row["Tavg_TaIrTe4_rise_K"]) * 1e9 for row in rows])
    axes[0, 1].set_ylabel("TaIrTe4 volume-average rise (nK)")
    axes[0, 1].tick_params(axis="x", rotation=20)
    axes[1, 0].bar(labels, [float(row["PTE_current_A_per_W_incident"]) * 1e12 for row in rows])
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_ylabel("PTE response (pA/W incident)")
    axes[1, 0].set_title("Near-null symmetric rho=0.5 diagnostic")
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 1].semilogy(labels, [float(row["forward_residual"]) for row in rows], "o-", label="forward residual")
    axes[1, 1].semilogy(labels, [float(row["adjoint_residual"]) for row in rows], "s-", label="adjoint residual")
    axes[1, 1].semilogy(labels, [float(row["energy_balance_error"]) for row in rows], "^-", label="energy balance")
    axes[1, 1].semilogy(labels, [float(row["reciprocity_Cauchy_normalized_error"]) for row in rows], "d-", label="Cauchy reciprocity")
    axes[1, 1].axhline(1e-8, color="black", linestyle=":", label="residual/reciprocity gate")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].tick_params(axis="x", rotation=20)
    fig.suptitle("Run 002 production CUDA thermal/PTE scenario controls")
    metrics_plot = PLOTS / "production_cuda_thermal_pte_summary.png"
    fig.savefig(metrics_plot, dpi=180)
    plt.close(fig)

    common_vmax = max(float(np.nanmax(value)) for value in maps.values()) * 1e9
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True, sharex=True, sharey=True)
    image = None
    for axis, scenario in zip(axes.flat, SCENARIOS):
        image = axis.pcolormesh(x_um, y_um, maps[scenario].T * 1e9, shading="auto", cmap="inferno", vmin=0, vmax=common_vmax)
        axis.set_title(LABELS[scenario])
        axis.set_aspect("equal")
        axis.set_xlabel("x=b (um)")
        axis.set_ylabel("y=a (um)")
    assert image is not None
    fig.colorbar(image, ax=axes, label="TaIrTe4 thickness-averaged temperature rise (nK)")
    fig.suptitle("Common-scale rho=0.5 temperature maps")
    maps_plot = PLOTS / "production_cuda_thermal_pte_temperature_maps.png"
    fig.savefig(maps_plot, dpi=180)
    plt.close(fig)

    report_path = RESULTS / "PRODUCTION_CUDA_THERMAL_PTE_REPORT.md"
    largest_signal_fraction = max(
        float(row["PTE_signal_Cauchy_scale_fraction"]) for row in rows
    )
    case_lines = "\n".join(
        "| {scenario} | {Tmax_all_materials_rise_K:.6e} | {Tmax_TaIrTe4_rise_K:.6e} | "
        "{Tavg_TaIrTe4_rise_K:.6e} | {PTE_current_A_per_W_incident:.6e} | "
        "{forward_residual:.3e} | {adjoint_residual:.3e} | {energy_balance_error:.3e} |".format(**row)
        for row in rows
    )
    report_path.write_text(
        f"""# Production CUDA thermal/PTE scenario controls

Status: `{status}`

The exact material-attributed volumetric Q was applied to the same explicit
3D anisotropic thermal operator for all four named bottom/design interface-G
scenarios. Matrix assembly was performed on the host; every production linear
forward and implicit-adjoint solve used CUDA float64. There was no CPU linear
solve fallback, Q modification, empirical normalization, or gradient
rescaling.

| bottom/design scenario | all-material Tmax rise (K) | TaIrTe4 Tmax rise (K) | TaIrTe4 average rise (K) | PTE response (A/W incident) | forward residual | adjoint residual | energy balance |
|---|---:|---:|---:|---:|---:|---:|---:|
{case_lines}

The current is a near-null diagnostic: uniform rho=0.5 and the centered source
are symmetric under the present uniform 45-degree weighting surrogate. It is
not an optimized current or an experimental prediction. Consequently, the raw
relative difference between two approximately `1e-26 A` reciprocal forms is
ill-conditioned and is retained only as a diagnostic. The scale-aware Cauchy
normalized reciprocity error is used for the numerical gate; its worst value
is `{worst['reciprocity_Cauchy_normalized_error']:.6e}`. The largest PTE
bilinear signal is only `{largest_signal_fraction:.6e}` of its Cauchy scale,
which quantitatively identifies this baseline as a cancellation-dominated
null control.

The weighting field is `(15625, 15625) 1/m`, corresponding to a unit potential
difference across opposite diagonal equipotential lines of the finite 32 um
flake. It is a production surrogate, not the full experimental electrode
operator. Maxwell adjoint, combined AD-FD, coarse-gradient design-window
selection, and optimization have not run.
"""
    )

    manifest = json.loads(MANIFEST.read_text())
    manifest["production_cuda_thermal_pte"] = {
        "status": status,
        "raw_artifacts_committed_to_git": False,
        "mapped_Q": artifact_record(mapped_q),
        "promoted_cases": raw_records,
        "preserved_failed_diagnostic": diagnostic,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": status,
                "summary": str(summary_path),
                "csv": str(csv_path),
                "report": str(report_path),
                "plots": [str(metrics_plot), str(maps_plot)],
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
