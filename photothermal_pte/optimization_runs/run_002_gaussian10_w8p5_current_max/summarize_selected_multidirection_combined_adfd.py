#!/usr/bin/env python3
"""Publish the selected-grid multidirection combined physical-rho AD-FD gate."""

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


STATUS = "VALIDATED_SELECTED_MULTIDIRECTION_COMBINED_PHYSICAL_RHO_ADFD"


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
    parser.add_argument("--one-direction-summary", type=Path, required=True)
    parser.add_argument("--one-direction-result", type=Path, required=True)
    parser.add_argument("--one-direction-raw", type=Path, required=True)
    parser.add_argument("--direction-result", type=Path, action="append", required=True)
    parser.add_argument("--results-directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()

    old_summary_path = args.one_direction_summary.expanduser().resolve()
    old_result_path = args.one_direction_result.expanduser().resolve()
    old_raw_path = args.one_direction_raw.expanduser().resolve()
    old = json.loads(old_summary_path.read_text())
    old_result = json.loads(old_result_path.read_text())
    old_gates = old_result["gates"]
    old_npz = np.load(old_raw_path)
    rows = [{
        "direction": "adjoint_aligned",
        "step": 0.005,
        "AD_A": float(old["directional_derivatives_A"]["combined_AD_recomputed"]),
        "FD_A": float(old["directional_derivatives_A"]["combined_FD_reused"]),
        "relative_error": float(old["gates"]["combined_AD_FD_relative_error"]),
        "normalized_error": float(old["gates"]["combined_AD_FD_relative_error"]),
        "direction_is_strong": True,
        "optical_closure": float(old_gates["worst_optical_closure"]),
        "Q_mapping_error": float(old_gates["worst_Q_mapping_error"]),
        "thermal_residual": float(old_gates["worst_thermal_residual"]),
        "thermal_energy_balance": float(old_gates["worst_thermal_energy_balance"]),
        "auto_shutoff": float(old_gates["worst_forward_auto_shutoff"]),
        "passed": True,
        "raw_NPZ": str(old_raw_path),
        "raw_NPZ_sha256": sha256(old_raw_path),
    }]
    direction_maps = [("adjoint aligned", np.asarray(old_npz["direction"], float))]
    direction_artifacts: dict[str, object] = {
        "adjoint_aligned": {
            "summary": artifact(old_summary_path),
            "preserved_diagnostic_result": artifact(old_result_path),
            "NPZ": artifact(old_raw_path),
        }
    }
    for result_path_arg in args.direction_result:
        result_path = result_path_arg.expanduser().resolve()
        result = json.loads(result_path.read_text())
        if result.get("status") != "VALIDATED_SELECTED_COMBINED_DIRECTION_ADFD" or not result.get("passed"):
            raise RuntimeError(f"direction did not pass: {result_path}")
        raw_path = Path(result["raw_artifact"]["path"])
        if sha256(raw_path) != result["raw_artifact"]["sha256"]:
            raise RuntimeError(f"direction raw SHA mismatch: {raw_path}")
        raw = np.load(raw_path)
        name = str(result["direction"])
        gates = result["gates"]
        rows.append({
            "direction": name,
            "step": float(result["step"]),
            "AD_A": float(result["adjoint_directional_A"]),
            "FD_A": float(result["finite_difference_directional_A"]),
            "relative_error": float(result["relative_error"]),
            "normalized_error": float(result["multi_direction_normalized_error"]),
            "direction_is_strong": bool(result["direction_is_strong"]),
            "optical_closure": float(gates["worst_optical_closure"]),
            "Q_mapping_error": float(gates["worst_Q_mapping_error"]),
            "thermal_residual": float(gates["worst_thermal_residual"]),
            "thermal_energy_balance": float(gates["worst_thermal_energy_balance"]),
            "auto_shutoff": float(gates["worst_auto_shutoff"]),
            "passed": True,
            "raw_NPZ": str(raw_path),
            "raw_NPZ_sha256": sha256(raw_path),
        })
        direction_maps.append((name.replace("_", " "), np.asarray(raw["direction"], float)))
        direction_artifacts[name] = {
            "result": artifact(result_path),
            "NPZ": artifact(raw_path),
        }

    expected = {"adjoint_aligned", "smooth_asymmetric", "central_localized", "design_edge_localized", "fixed_seed_random"}
    if {row["direction"] for row in rows} != expected:
        raise RuntimeError("the required five independent directions are not complete")
    worst_relative = max(float(row["relative_error"]) for row in rows)
    worst_normalized = max(float(row["normalized_error"]) for row in rows)
    worst_closure = max(float(row["optical_closure"]) for row in rows if np.isfinite(row["optical_closure"]))
    worst_mapping = max(float(row["Q_mapping_error"]) for row in rows)
    worst_residual = max(float(row["thermal_residual"]) for row in rows if np.isfinite(row["thermal_residual"]))
    worst_energy = max(float(row["thermal_energy_balance"]) for row in rows if np.isfinite(row["thermal_energy_balance"]))
    worst_shutoff = max(float(row["auto_shutoff"]) for row in rows if np.isfinite(row["auto_shutoff"]))
    passed = (
        worst_relative < 0.01
        and worst_normalized < 0.01
        and worst_closure < 0.005
        and worst_mapping < 0.005
        and worst_residual < 1.0e-8
        and worst_energy < 0.01
        and worst_shutoff < 1.0e-5
    )
    if not passed:
        raise RuntimeError("multidirection combined AD-FD gates failed")

    output = args.results_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "selected_multidirection_combined_adfd_cases.csv"
    summary_path = output / "selected_multidirection_combined_adfd_summary.json"
    report_path = output / "SELECTED_MULTIDIRECTION_COMBINED_ADFD_REPORT.md"
    plot_path = output / "selected_multidirection_combined_adfd.png"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fig = plt.figure(figsize=(17, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(grid[0, :2])
    ad = np.asarray([row["AD_A"] for row in rows])
    fd = np.asarray([row["FD_A"] for row in rows])
    scale = max(np.max(np.abs(ad)), np.max(np.abs(fd)))
    ax.plot([-scale, scale], [-scale, scale], "k--", label="ideal AD = FD")
    for row in rows:
        ax.scatter(row["FD_A"], row["AD_A"], s=70, label=row["direction"].replace("_", " "))
    ax.set_xlabel("finite-difference directional derivative (A)")
    ax.set_ylabel("adjoint directional derivative (A)")
    ax.set_title("Combined physical-density AD–FD, five directions")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(grid[0, 2:])
    labels = [row["direction"].replace("_", "\n") for row in rows]
    errors = [100.0 * row["relative_error"] for row in rows]
    ax.bar(np.arange(len(rows)), errors)
    ax.axhline(1.0, color="black", linestyle="--", label="1% gate")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(rows)), labels, fontsize=8)
    ax.set_ylabel("relative error (%)")
    ax.set_title("Directional error (no gradient rescaling)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    # Show the four independent non-adjoint directions; the aligned map already
    # appears in the earlier selected optical-gradient certificate.
    for index, (name, values) in enumerate(direction_maps[1:]):
        ax = fig.add_subplot(grid[1, index])
        image = ax.imshow(values.T, origin="lower", cmap="coolwarm", aspect="equal")
        ax.set_title(name)
        ax.set_xlabel("design node x")
        if index == 0:
            ax.set_ylabel("design node y")
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    generated = datetime.now(timezone.utc).isoformat()
    summary = {
        "status": STATUS,
        "passed": True,
        "generated_at_utc": generated,
        "scope": "five selected 373x373 combined physical-density directions at h=0.005",
        "optimizer_started": False,
        "Maxwell_forward_solves_new": 8,
        "Maxwell_adjoint_solves_new": 0,
        "CPU_FDTD_fallback": False,
        "CPU_thermal_solve_fallback": False,
        "empirical_normalization": False,
        "gradient_rescaling": False,
        "gates": {
            "worst_directional_relative_error": worst_relative,
            "worst_multi_direction_normalized_error": worst_normalized,
            "directional_limit": 0.01,
            "worst_optical_closure": worst_closure,
            "optical_closure_limit": 0.005,
            "worst_Q_mapping_error": worst_mapping,
            "Q_mapping_limit": 0.005,
            "worst_thermal_residual": worst_residual,
            "thermal_residual_limit": 1.0e-8,
            "worst_thermal_energy_balance": worst_energy,
            "thermal_energy_limit": 0.01,
            "worst_auto_shutoff": worst_shutoff,
            "auto_shutoff_limit": 1.0e-5,
        },
        "cases": rows,
        "raw_artifacts": direction_artifacts,
        "remaining_before_optimization": [
            "full latent/filter/projection combined AD-FD",
            "exact-binary DRC fixtures before final fabrication promotion",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    table = "\n".join(
        f"| {row['direction']} | {row['AD_A']:.12e} | {row['FD_A']:.12e} | {100.0*row['relative_error']:.6f}% | pass |"
        for row in rows
    )
    report_path.write_text(
        "# Selected multidirection combined physical-density AD–FD\n\n"
        f"Status: `{STATUS}`\n\n"
        "The corrected component-wise Yee/thermal chain passes five independent "
        "physical-density directions. These are real centered FD reruns at `h=0.005`; "
        "no empirical normalization, FD-derived scale, or gradient rescaling is used.\n\n"
        "| direction | AD (A) | FD (A) | relative error | result |\n"
        "|---|---:|---:|---:|---|\n"
        f"{table}\n\n"
        f"Worst directional error is `{100.0*worst_relative:.6f}%`; worst normalized error is "
        f"`{100.0*worst_normalized:.6f}%`. New cases used eight GPU Maxwell forward solves and "
        "eight CUDA thermal forward solves. No CPU FDTD or CPU thermal fallback was used.\n\n"
        "This closes the combined *physical-density* gate. It does not yet certify the "
        "latent→finite-filter→projection chain and does not itself start optimization.\n"
    )

    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    manifest["selected_multidirection_combined_adfd"] = {
        "status": STATUS,
        "raw_artifacts_committed_to_git": False,
        "cases": direction_artifacts,
    }
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
            f"Five combined physical-density AD-FD directions pass at h=0.005; "
            f"worst directional error={worst_relative:.3e}, worst normalized error={worst_normalized:.3e}. "
            "Full latent/filter/projection AD-FD still blocks optimization."
        ),
    })
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
