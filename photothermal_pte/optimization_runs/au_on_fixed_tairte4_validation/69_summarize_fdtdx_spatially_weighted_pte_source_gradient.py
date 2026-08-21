#!/usr/bin/env python3
"""Audit and publish the native-Yee spatially weighted FDTDX gradient.

This is intentionally narrower than a combined PTE gradient certificate.  It
checks only the Maxwell/source branch obtained by contracting native Yee-cell
absorbed powers with the frozen explicit-thermal source adjoint (A/W).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_STATUS = "VALIDATED_FDTDX_NATIVE_YEE_SPATIALLY_WEIGHTED_PTE_SOURCE_GRADIENT"
WEIGHT_STATUS = "VALIDATED_NATIVE_YEE_THERMAL_SOURCE_ADJOINT_PULLBACK"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), np.finfo(float).tiny)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--raw-gradient-npz", required=True, type=Path)
    parser.add_argument("--weight-summary-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    result_path = args.result_json.expanduser().resolve()
    raw_path = args.raw_gradient_npz.expanduser().resolve()
    weight_summary_path = args.weight_summary_json.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    result = json.loads(result_path.read_text(encoding="utf-8"))
    weight_summary = json.loads(weight_summary_path.read_text(encoding="utf-8"))
    raw_sha = _sha256(raw_path)
    recorded_raw = result["raw_artifact"]
    if raw_sha != recorded_raw["sha256"]:
        raise RuntimeError(
            f"Gradient raw SHA mismatch: {raw_sha} != {recorded_raw['sha256']}"
        )
    if weight_summary.get("status") != WEIGHT_STATUS:
        raise RuntimeError("Fail-closed: thermal-source weight status mismatch")
    if (
        result["spatial_weight"]["raw_sha256"]
        != weight_summary["raw_artifact"]["sha256"]
    ):
        raise RuntimeError("Fail-closed: weight SHA mismatch between checkpoints")

    with np.load(raw_path, allow_pickle=False) as raw:
        rho = np.asarray(raw["rho"], dtype=np.float64)
        gradient = np.asarray(raw["gradient_A"], dtype=np.float64)
        objective = float(raw["weighted_objective_A"])
        total_q = float(raw["total_P_Q_W"])

    baseline = result["baseline"]
    weight_scenario = result["spatial_weight"]["scenario"]
    expected_weighted_objective = float(
        weight_summary["scenarios"][weight_scenario][
            "native_weighted_source_value_A"
        ]
    )
    raw_finite = bool(
        np.all(np.isfinite(rho))
        and np.all(np.isfinite(gradient))
        and np.isfinite(objective)
        and np.isfinite(total_q)
    )
    shape_contract = bool(rho.shape == (20, 20) and gradient.shape == rho.shape)
    objective_error = _relative(objective, baseline["weighted_source_objective_A"])
    power_error = _relative(total_q, baseline["P_Q_W"])
    gradient_norm = float(np.linalg.norm(gradient))
    gradient_norm_error = _relative(gradient_norm, baseline["gradient_l2_A"])
    checkpoint_value_error = _relative(objective, expected_weighted_objective)

    rows = result["directions"]
    if not rows:
        raise RuntimeError("No AD-FD directional rows were generated")
    strongest_error = max(
        (float(row["strong_relative_error"]) for row in rows if row["strong_direction"]),
        default=float("inf"),
    )
    normalized_error = max(
        float(row["gradient_l2_normalized_error"]) for row in rows
    )
    gates = {
        "generation_status_validated": result.get("status") == EXPECTED_STATUS,
        "generation_gates_all_true": all(result["gates"].values()),
        "weight_status_and_SHA_chain_validated": True,
        "gradient_raw_SHA_matches_generation_record": True,
        "finite_raw_arrays": raw_finite,
        "rho_and_gradient_shape_20x20": shape_contract,
        "raw_objective_matches_result_lt_1e-12": objective_error < 1.0e-12,
        "raw_total_Q_matches_result_lt_1e-12": power_error < 1.0e-12,
        "raw_gradient_norm_matches_result_lt_1e-12": gradient_norm_error < 1.0e-12,
        # The FDTDX run and the exported-Q checkpoint both use float32 field/Q
        # arrays, but form the contraction in separate executions.  Keep this
        # rerun-consistency gate two orders tighter than the 0.5% optical gate
        # without pretending the two float32 reductions are bitwise identical.
        "base_weighted_value_matches_pullback_checkpoint_lt_1e-4": (
            checkpoint_value_error < 1.0e-4
        ),
        "strong_direction_AD_FD_error_lt_1pct": strongest_error < 0.01,
        "gradient_L2_normalized_error_lt_1pct": normalized_error < 0.01,
        "optical_Q_flux_closure_lt_0p5pct": baseline["Q_flux_closure_relative"] < 0.005,
        "late_Q_and_weighted_objective_change_lt_0p5pct": (
            baseline["late_Q_window_relative_change"] < 0.005
            and baseline["weighted_objective_window_relative_change"] < 0.005
        ),
        "no_clipping_smoothing_gain_or_gradient_rescaling": result[
            "no_clipping_smoothing_gain_or_gradient_rescaling"
        ],
    }
    passed = all(gates.values())
    status = EXPECTED_STATUS if passed else "FAILED_FDTDX_NATIVE_YEE_SPATIALLY_WEIGHTED_PTE_SOURCE_GRADIENT_AUDIT"

    ad = np.asarray([row["AD_A_per_unit_direction"] for row in rows], dtype=float)
    fd = np.asarray([row["FD_A_per_unit_direction"] for row in rows], dtype=float)
    errors = 100.0 * np.asarray(
        [row["gradient_l2_normalized_error"] for row in rows], dtype=float
    )
    labels = [f"{row['direction']}\nh={row['h']:g}" for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), constrained_layout=True)
    density_image = axes[0, 0].imshow(
        rho.T,
        origin="lower",
        extent=(-5, 5, -5, 5),
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0, 0].set_title("nonuniform Au density")
    axes[0, 0].set_xlabel("x=b (um)")
    axes[0, 0].set_ylabel("y=a (um)")
    fig.colorbar(density_image, ax=axes[0, 0], label="rho")

    gradient_scale = max(float(np.max(np.abs(gradient))), np.finfo(float).tiny)
    gradient_image = axes[0, 1].imshow(
        gradient.T,
        origin="lower",
        extent=(-5, 5, -5, 5),
        cmap="coolwarm",
        vmin=-gradient_scale,
        vmax=gradient_scale,
    )
    axes[0, 1].set_title("Maxwell source gradient")
    axes[0, 1].set_xlabel("x=b (um)")
    axes[0, 1].set_ylabel("y=a (um)")
    fig.colorbar(gradient_image, ax=axes[0, 1], label="A per rho")

    limit = 1.1 * max(float(np.max(np.abs(ad))), float(np.max(np.abs(fd))), 1e-30)
    axes[1, 0].plot([-limit, limit], [-limit, limit], "k--", label="ideal AD=FD")
    for index, row in enumerate(rows):
        axes[1, 0].scatter(fd[index], ad[index], s=70, label=labels[index])
    axes[1, 0].set_xlim(-limit, limit)
    axes[1, 0].set_ylim(-limit, limit)
    axes[1, 0].set_aspect("equal", adjustable="box")
    axes[1, 0].set_xlabel("central FD directional derivative (A)")
    axes[1, 0].set_ylabel("AD directional derivative (A)")
    axes[1, 0].set_title("Spatially weighted source AD-FD")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].bar(np.arange(len(rows)), errors)
    axes[1, 1].axhline(1.0, color="k", linestyle="--", label="1% gate")
    axes[1, 1].set_xticks(np.arange(len(rows)), labels, rotation=20, ha="right")
    axes[1, 1].set_ylabel("|AD-FD| / ||gradient||2 (%)")
    axes[1, 1].set_title("Directional gradient error")
    axes[1, 1].legend()
    figure_path = output / "fdtdx_spatially_weighted_pte_source_gradient.png"
    fig.suptitle(status.replace("_", " "), fontsize=12)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    summary = {
        "status": status,
        "scope": (
            "native-Yee Maxwell/source derivative contracted with the frozen "
            "explicit-thermal source adjoint; excludes direct thermal/electrical "
            "density derivatives, full combined AD-FD, and optimization"
        ),
        "source_result_json": str(result_path),
        "spatial_weight": result["spatial_weight"],
        "baseline": baseline,
        "worst_strong_direction_relative_error": strongest_error,
        "worst_gradient_l2_normalized_error": normalized_error,
        "raw_reintegration": {
            "objective_relative_error": objective_error,
            "total_Q_relative_error": power_error,
            "gradient_norm_relative_error": gradient_norm_error,
            "pullback_checkpoint_weighted_value_relative_error": checkpoint_value_error,
        },
        "runtime": result["runtime"],
        "gates": gates,
        "raw_artifact": {
            "path": str(raw_path),
            "bytes": raw_path.stat().st_size,
            "sha256": raw_sha,
            "committed_to_git": False,
        },
        "next_gate": (
            "sum the Maxwell source gradient and the already validated direct "
            "thermal/electrical gradients, then run end-to-end combined directional AD-FD"
        ),
    }
    summary_path = output / "fdtdx_spatially_weighted_pte_source_gradient_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = f"""# FDTDX native-Yee spatially weighted PTE-source gradient

Status: **{status}**

This checkpoint differentiates the Maxwell heat-source branch only. Native
Yee-cell powers `Q_c dV_c` for Au, TaIrTe4, and SiO2 are contracted with the
frozen explicit-thermal source adjoint in A/W. Each electric component remains
on its own staggered physical coordinates. The thermal/electrical direct
density terms are not included here and no optimization is run.

| metric | value |
|---|---:|
| weighted source objective | {baseline['weighted_source_objective_A']:.12e} A |
| Stage-68 source contraction | {expected_weighted_objective:.12e} A |
| contraction checkpoint difference | {100.0 * checkpoint_value_error:.9f}% |
| total optical P_Q | {baseline['P_Q_W']:.12e} W |
| gradient L2 norm | {baseline['gradient_l2_A']:.12e} A |
| strongest-direction AD-FD error | {100.0 * strongest_error:.9f}% |
| gradient-L2-normalized error | {100.0 * normalized_error:.9f}% |
| matched-volume Q/flux closure | {100.0 * baseline['Q_flux_closure_relative']:.6f}% |
| late-Q change | {100.0 * baseline['late_Q_window_relative_change']:.6f}% |
| weighted-objective late change | {100.0 * baseline['weighted_objective_window_relative_change']:.6f}% |
| reverse AD runtime | {result['runtime']['ad_seconds']:.3f} s |
| central-FD forward runtime | {result['runtime']['central_fd_forward_seconds']:.3f} s |

The raw gradient NPZ is outside Git and pinned by SHA-256 `{raw_sha}`. No
clipping, smoothing, gain, objective matching, or gradient rescaling is used.
Passing this report does **not** certify the full PTE gradient. The next gate
must recompute Maxwell Q, explicit thermal transport, and the Au-aware
electrical weighting/current for each central-FD perturbation and compare that
end-to-end derivative with the sum of the three analytic/adjoint branches.
"""
    report_path = output / "FDTDX_SPATIALLY_WEIGHTED_PTE_SOURCE_GRADIENT_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    published = [
        result_path,
        output / "fdtdx_spatially_weighted_pte_source_gradient.csv",
        summary_path,
        report_path,
        figure_path,
    ]
    manifest = {
        "status": status,
        "raw_artifact": summary["raw_artifact"],
        "weight_artifact": {
            "path": result["spatial_weight"]["raw_npz"],
            "sha256": result["spatial_weight"]["raw_sha256"],
            "committed_to_git": False,
        },
        "published": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in published
        ],
    }
    manifest_path = output / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "gates": gates}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
