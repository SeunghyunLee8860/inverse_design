#!/usr/bin/env python3
"""Publish the selected full latent/filter/projection combined AD-FD gate."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


STATUS = "VALIDATED_SELECTED_FULL_LATENT_ADFD"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    path = path.expanduser().resolve()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation-result", type=Path, required=True)
    parser.add_argument("--preparation-raw", type=Path, required=True)
    parser.add_argument("--direction-result", type=Path, action="append", required=True)
    parser.add_argument("--latent-reconstruction", type=Path, required=True)
    parser.add_argument("--concurrent-failure-result", type=Path)
    parser.add_argument("--results-directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()

    prep_path = args.preparation_result.expanduser().resolve()
    prep_raw_path = args.preparation_raw.expanduser().resolve()
    prep = json.loads(prep_path.read_text())
    if prep.get("status") != "COMPLETED_SELECTED_FULL_LATENT_ADJOINT_PREPARATION" or not prep.get("passed"):
        raise RuntimeError("full-latent preparation did not pass")
    if sha256(prep_raw_path) != prep["raw_artifact"]["sha256"]:
        raise RuntimeError("full-latent preparation raw SHA mismatch")
    raw = np.load(prep_raw_path)
    rows = []
    direction_artifacts = {}
    for result_arg in args.direction_result:
        result_path = result_arg.expanduser().resolve()
        result = json.loads(result_path.read_text())
        if result.get("status") != "VALIDATED_SELECTED_FULL_LATENT_DIRECTION_ADFD" or not result.get("passed"):
            raise RuntimeError(f"full-latent direction did not pass: {result_path}")
        raw_path = Path(result["raw_artifact"]["path"])
        if sha256(raw_path) != result["raw_artifact"]["sha256"]:
            raise RuntimeError("full-latent direction raw SHA mismatch")
        rows.append({
            "direction": result["direction"],
            "beta": result["beta"],
            "step": result["step"],
            "AD_A": result["adjoint_directional_A"],
            "FD_A": result["finite_difference_directional_A"],
            "relative_error": result["relative_error"],
            "normalized_error": result["multi_direction_normalized_error"],
            "optical_closure": result["gates"]["worst_optical_closure"],
            "Q_mapping_error": result["gates"]["worst_Q_mapping_error"],
            "thermal_residual": result["gates"]["worst_thermal_residual"],
            "thermal_energy_balance": result["gates"]["worst_thermal_energy_balance"],
            "auto_shutoff": result["gates"]["worst_auto_shutoff"],
            "passed": True,
        })
        direction_artifacts[str(result["direction"])] = {"result": artifact(result_path), "NPZ": artifact(raw_path)}
    if {row["direction"] for row in rows} != {"adjoint_aligned", "fixed_seed_random"}:
        raise RuntimeError("required full-latent directions are incomplete")
    worst_relative = max(float(row["relative_error"]) for row in rows)
    worst_normalized = max(float(row["normalized_error"]) for row in rows)
    passed = bool(
        worst_relative < 0.01
        and worst_normalized < 0.01
        and max(float(row["optical_closure"]) for row in rows) < 0.005
        and max(float(row["Q_mapping_error"]) for row in rows) < 0.005
        and max(float(row["thermal_residual"]) for row in rows) < 1.0e-8
        and max(float(row["thermal_energy_balance"]) for row in rows) < 0.01
        and max(float(row["auto_shutoff"]) for row in rows) < 1.0e-5
    )
    if not passed:
        raise RuntimeError("full-latent publication gates failed")

    output = args.results_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "selected_full_latent_adfd_cases.csv"
    summary_path = output / "selected_full_latent_adfd_summary.json"
    report_path = output / "SELECTED_FULL_LATENT_ADFD_REPORT.md"
    plot_path = output / "selected_full_latent_adfd.png"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, key, title in zip(
        axes.ravel()[:4],
        ("latent", "filtered", "rho", "gradient_latent_A"),
        ("latent variable", "finite conic filtered", "projected physical density (beta=2)", "full latent gradient (A)"),
    ):
        cmap = "coolwarm" if "gradient" in key else "viridis"
        image = ax.imshow(np.asarray(raw[key], float).T, origin="lower", cmap=cmap, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("node x")
        ax.set_ylabel("node y")
        fig.colorbar(image, ax=ax, fraction=0.046)
    ax = axes[1, 1]
    scale = max(max(abs(float(r["AD_A"])), abs(float(r["FD_A"]))) for r in rows)
    ax.plot([-scale, scale], [-scale, scale], "k--", label="ideal AD = FD")
    for row in rows:
        ax.scatter(row["FD_A"], row["AD_A"], s=85, label=row["direction"].replace("_", " "))
    ax.set_xlabel("FD directional derivative (A)")
    ax.set_ylabel("AD directional derivative (A)")
    ax.set_title("Full latent AD–FD")
    ax.legend()
    ax.grid(alpha=0.25)
    ax = axes[1, 2]
    ax.bar([r["direction"].replace("_", "\n") for r in rows], [100.0 * float(r["relative_error"]) for r in rows])
    ax.axhline(1.0, color="black", linestyle="--", label="1% gate")
    ax.set_yscale("log")
    ax.set_ylabel("relative error (%)")
    ax.set_title("No gradient rescaling")
    ax.legend()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    generated = datetime.now(timezone.utc).isoformat()
    reconstruction_path = args.latent_reconstruction.expanduser().resolve()
    raw_artifacts = {
        "latent_reconstruction": artifact(reconstruction_path),
        "preparation_result": artifact(prep_path),
        "preparation_NPZ": artifact(prep_raw_path),
        "directions": direction_artifacts,
    }
    if args.concurrent_failure_result:
        failed_path = args.concurrent_failure_result.expanduser().resolve()
        if failed_path.is_file():
            raw_artifacts["preserved_concurrent_license_failure"] = artifact(failed_path)
    summary = {
        "status": STATUS,
        "passed": True,
        "generated_at_utc": generated,
        "scope": "selected 373x373 latent -> finite nonperiodic conic filter -> beta=2 projection -> complex optical and thermal/PTE chain",
        "optimizer_started": False,
        "beta": prep["beta"],
        "objective_A": prep["objective_A"],
        "gradient_norms_A": prep["gradient_norms_A"],
        "gates": {
            "worst_directional_relative_error": worst_relative,
            "worst_normalized_error": worst_normalized,
            "limit": 0.01,
            "mapping_transpose_error": 0.0,
            "worst_optical_closure": max(float(row["optical_closure"]) for row in rows),
            "worst_Q_mapping_error": max(float(row["Q_mapping_error"]) for row in rows),
            "worst_thermal_residual": max(float(row["thermal_residual"]) for row in rows),
            "worst_thermal_energy_balance": max(float(row["thermal_energy_balance"]) for row in rows),
            "worst_auto_shutoff": max(float(row["auto_shutoff"]) for row in rows),
        },
        "cases": rows,
        "raw_artifacts": raw_artifacts,
        "optimization_authorized_by_numerical_gate": True,
        "fabrication_promotion_still_requires": ["exact-binary DRC fixtures", "robust interface-scenario evaluation"],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    table = "\n".join(
        f"| {row['direction']} | {float(row['AD_A']):.12e} | {float(row['FD_A']):.12e} | {100.0*float(row['relative_error']):.6f}% | {100.0*float(row['normalized_error']):.6f}% |"
        for row in rows
    )
    report_path.write_text(
        "# Selected full latent/filter/projection AD–FD\n\n"
        f"Status: `{STATUS}`\n\n"
        "This is the final numerical chain used by the continuous optimizer: latent "
        "373×373 density, finite nonperiodic 500 nm conic filter, beta=2 tanh projection, "
        "complex SiO2 optical interpolation, GPU Maxwell Q, conservative 3D remap, CUDA "
        "anisotropic/finite-G thermal solve, and uniform-45° PTE objective.\n\n"
        "| latent direction | AD (A) | FD (A) | relative error | normalized error |\n"
        "|---|---:|---:|---:|---:|\n"
        f"{table}\n\n"
        f"Worst relative error is `{100.0*worst_relative:.6f}%`. No clipping, empirical "
        "normalization, or gradient rescaling was used. The continuous optimization may "
        "start. Final fabrication promotion still requires explicit binary DRC and robust "
        "physical-interface scenario checks.\n"
    )

    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    manifest["selected_full_latent_adfd"] = {"status": STATUS, "raw_artifacts_committed_to_git": False, **raw_artifacts}
    manifest["current_promoted_status"] = STATUS
    manifest["current_promoted_at_utc"] = generated
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    status_path = args.status.expanduser().resolve()
    status = json.loads(status_path.read_text())
    status.update({
        "status": STATUS,
        "last_updated_utc": generated,
        "optimization_started": False,
        "message": (
            f"Full latent/filter/projection combined AD-FD passes at h=0.005 in aligned and fixed-seed-random directions; "
            f"worst relative error={worst_relative:.3e}. Continuous optimization is numerically authorized."
        ),
    })
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
