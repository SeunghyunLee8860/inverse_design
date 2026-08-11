#!/usr/bin/env python3
"""Publish the Run-002 production combined physical-rho AD-FD smoke."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def checked_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def artifact_from_result(value: dict) -> dict[str, object]:
    path = Path(value["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = record(path)
    expected_size = value.get("size_bytes", value.get("byte_size"))
    if actual["size_bytes"] != expected_size or actual["sha256"] != value["sha256"]:
        raise RuntimeError(f"raw artifact provenance mismatch: {path}")
    return actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--fixed-mesh-audit", type=Path, required=True)
    parser.add_argument("--pre-fix-mesh-audit", type=Path, required=True)
    parser.add_argument("--failed-directory", type=Path, action="append", default=[])
    args = parser.parse_args()
    raw = args.raw_directory.expanduser().resolve()
    result_path = raw / "production_combined_adfd_smoke_result.json"
    result = checked_json(result_path)
    if result.get("status") != "VALIDATED_PRODUCTION_COMBINED_PHYSICAL_RHO_ADFD_SMOKE" or not result.get("passed"):
        raise RuntimeError("combined physical-rho smoke is not validated")
    fixed_path = args.fixed_mesh_audit.expanduser().resolve()
    old_path = args.pre_fix_mesh_audit.expanduser().resolve()
    fixed = checked_json(fixed_path)
    old = checked_json(old_path)
    if not fixed.get("passed") or not old.get("passed"):
        raise RuntimeError("mesh audits must be completed")

    promoted_artifacts = [
        record(result_path),
        artifact_from_result(result["raw_artifact"]),
        artifact_from_result(result["adjoint_source"]["template"]),
        artifact_from_result(result["adjoint"]["project"]),
        artifact_from_result(result["base_forward"]["project"]),
        artifact_from_result(result["base_forward"]["native_Q"]),
    ]
    for sign in ("plus", "minus"):
        forward = result["FD_pair"][sign]["forward"]
        promoted_artifacts.extend(
            [artifact_from_result(forward["project"]), artifact_from_result(forward["native_Q"])]
        )
    fixed_artifact = artifact_from_result(fixed["artifact"])
    old_artifact = artifact_from_result(old["artifact"])

    failures = []
    for directory in args.failed_directory:
        directory = directory.expanduser().resolve()
        failure_path = directory / "production_combined_adfd_smoke_result.json"
        failure = checked_json(failure_path)
        if failure.get("passed", False):
            raise RuntimeError(f"expected preserved failure: {failure_path}")
        failures.append(
            {
                "raw_directory": str(directory),
                "status": failure.get("status"),
                "error": failure.get("error"),
                "result": record(failure_path),
            }
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS / "production_combined_adfd_smoke_summary.json"
    published = dict(result)
    published["mesh_parity_audit"] = {
        "pre_fix": record(old_path),
        "fixed": record(fixed_path),
        "fixed_maximum_source_to_mesh_mismatch_m": max(
            fixed["coordinate_comparisons"][axis]["to_mesh_nodes"]["maximum_mismatch_m"]
            for axis in "xyz"
        ),
        "fixed_nonmatching_coordinate_count_at_2e-18m": sum(
            fixed["coordinate_comparisons"][axis]["to_mesh_nodes"]["nonmatching_count_2e-18m"]
            for axis in "xyz"
        ),
    }
    published["preserved_failed_attempts"] = failures
    published["scope_warning"] = (
        "One adjoint-aligned h=0.005 physical-rho smoke only; this is not a "
        "multi-direction, gray-law, full-latent, or optimization certificate."
    )
    summary_path.write_text(json.dumps(published, indent=2) + "\n")

    csv_path = RESULTS / "production_combined_adfd_smoke_cases.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            lineterminator="\n",
            fieldnames=[
                "direction", "step", "adjoint_A", "finite_difference_A",
                "relative_error", "plus_objective_A", "minus_objective_A",
                "worst_optical_closure", "worst_Q_mapping_error",
                "worst_thermal_residual", "worst_thermal_energy_balance",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "direction": "adjoint_aligned",
                "step": result["step"],
                "adjoint_A": result["adjoint_directional_A"],
                "finite_difference_A": result["finite_difference_directional_A"],
                "relative_error": result["combined_AD_FD_relative_error"],
                "plus_objective_A": result["FD_pair"]["plus"]["objective_A"],
                "minus_objective_A": result["FD_pair"]["minus"]["objective_A"],
                "worst_optical_closure": result["gates"]["worst_optical_closure"],
                "worst_Q_mapping_error": result["gates"]["worst_Q_mapping_error"],
                "worst_thermal_residual": result["gates"]["worst_thermal_residual"],
                "worst_thermal_energy_balance": result["gates"]["worst_thermal_energy_balance"],
            }
        )

    old_nm = [
        old["coordinate_comparisons"][axis]["to_mesh_nodes"]["maximum_mismatch_m"] * 1e9
        for axis in "xyz"
    ]
    fixed_nm = [
        fixed["coordinate_comparisons"][axis]["to_mesh_nodes"]["maximum_mismatch_m"] * 1e9
        for axis in "xyz"
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    ad = result["adjoint_directional_A"]
    fd = result["finite_difference_directional_A"]
    lo, hi = sorted([ad, fd])
    margin = max((hi - lo) * 2.5, abs(hi) * 0.03)
    axes[0].plot([lo - margin, hi + margin], [lo - margin, hi + margin], "k--", label="ideal AD=FD")
    axes[0].scatter([fd], [ad], s=80, color="tab:blue", zorder=3)
    axes[0].set(
        xlabel="finite-difference directional derivative (A)",
        ylabel="adjoint directional derivative (A)",
        title=f"physical-rho smoke\nrelative error={result['combined_AD_FD_relative_error']*100:.3f}%",
    )
    axes[0].legend()
    labels = ["AD-FD", "closure", "Q map", "thermal residual", "energy balance"]
    ratios = [
        result["gates"]["combined_AD_FD_relative_error"] / result["gates"]["combined_AD_FD_limit"],
        result["gates"]["worst_optical_closure"] / 0.005,
        max(result["gates"]["worst_Q_mapping_error"], 1e-18) / 0.005,
        result["gates"]["worst_thermal_residual"] / 1e-8,
        result["gates"]["worst_thermal_energy_balance"] / 0.01,
    ]
    axes[1].barh(labels, ratios, color=["tab:blue" if value < 1 else "tab:red" for value in ratios])
    axes[1].axvline(1.0, color="black", linestyle="--", label="gate")
    axes[1].set_xscale("log")
    axes[1].set(xlabel="metric / gate (log scale)", title="Gate margins")
    axes[1].legend()
    positions = np.arange(3)
    axes[2].bar(positions - 0.18, old_nm, width=0.36, label="deleted/disabled forward source")
    axes[2].bar(positions + 0.18, fixed_nm, width=0.36, label="enabled, zero-amplitude anchor")
    axes[2].set_xticks(positions, ["x", "y", "z"])
    axes[2].set_yscale("symlog", linthresh=1e-10)
    axes[2].set(ylabel="maximum source/mesh mismatch (nm)", title="Forward/adjoint mesh parity")
    axes[2].legend(fontsize=8)
    fig.suptitle("Run 002 production combined physical-rho AD-FD smoke")
    plot_path = PLOTS / "production_combined_adfd_smoke.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    report_path = RESULTS / "PRODUCTION_COMBINED_ADFD_SMOKE_REPORT.md"
    report_path.write_text(
        f"""# Production combined physical-rho PTE AD-FD smoke

Status: `{result['status']}`

This checkpoint validates one nonuniform 201×201 physical-density baseline,
one adjoint-aligned direction, and centered finite difference at
`h={result['step']}`. It is a smoke certificate, not yet a multi-direction,
gray-law, latent/filter/projection, or optimization certificate.

## Result

| quantity | value | gate |
|---|---:|---:|
| adjoint directional derivative | {result['adjoint_directional_A']:.12e} A | — |
| centered-FD directional derivative | {result['finite_difference_directional_A']:.12e} A | — |
| combined AD–FD relative error | {result['combined_AD_FD_relative_error']*100:.6f}% | <1% |
| component-J transpose error | {result['gates']['worst_component_J_transpose_error']:.6e} | <1e-12 |
| Q-remap transpose error | {result['gates']['worst_Q_pullback_transpose_error']:.6e} | <1e-12 |
| worst optical closure | {result['gates']['worst_optical_closure']:.6e} | <0.5% |
| worst Q mapping error | {result['gates']['worst_Q_mapping_error']:.6e} | <0.5% |
| worst thermal residual | {result['gates']['worst_thermal_residual']:.6e} | <1e-8 |
| worst thermal energy balance | {result['gates']['worst_thermal_energy_balance']:.6e} | <1% |

The base objective is `{result['base_objective_A']:.12e} A` for incident power
`{result['incident_power_W']:.12e} W`. Optical and thermal-material gradient
norms are `{result['gradient_norms_A']['optical']:.12e} A` and
`{result['gradient_norms_A']['thermal_material']:.12e} A`, respectively. No
empirical normalization, gradient rescaling, Q clipping, smoothing, gain, or
rescaling was used.

## Mesh-parity fix

The first attempts were fail-closed before any FD pair because changing the
FieldRegion from monitor to source mode regenerated the auto-nonuniform mesh.
Deleting or disabling the Gaussian source produced maximum coordinate
mismatches of `{max(old_nm):.6f} nm`. The successful contract keeps the
Gaussian source enabled with amplitude exactly zero during the adjoint. This
retains its mesh anchors but injects no forward illumination. The resulting
maximum mismatch is `{max(fixed_nm):.6e} nm`, with zero coordinates exceeding
`2e-18 m`. Forward and adjoint field arrays then have zero reported coordinate
mismatch.

The preserved v1–v5 failures are diagnostics and were not relabeled. They
contain no completed plus/minus FD sweep and no optimization iteration.

## Solver scope

- Maxwell: three forward solves total (the nonuniform base was reused) and
  one GPU adjoint solve; no CPU FDTD fallback.
- Thermal: three CUDA float64 forward solves and one CUDA float64 adjoint;
  no CPU linear-solve fallback.
- Optimization iterations: 0.

Before optimization, Run 002 still requires broader directional/step evidence,
gray-law sensitivity, the full latent/filter/projection pullback, and a
production design-window decision.
"""
    )

    manifest = json.loads(MANIFEST.read_text())
    manifest["production_combined_physical_rho_adfd_smoke"] = {
        "status": result["status"],
        "raw_artifacts_committed_to_git": False,
        "raw_directory": str(raw),
        "artifacts": promoted_artifacts,
        "mesh_audits": {
            "pre_fix": [record(old_path), old_artifact],
            "fixed": [record(fixed_path), fixed_artifact],
        },
        "preserved_failed_attempts": failures,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "report": str(report_path), "summary": str(summary_path), "csv": str(csv_path), "plot": str(plot_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
