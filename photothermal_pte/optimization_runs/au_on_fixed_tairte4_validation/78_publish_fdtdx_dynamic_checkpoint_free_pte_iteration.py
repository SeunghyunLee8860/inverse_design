#!/usr/bin/env python3
"""Publish the dynamic, checkpoint-free production PTE iteration certificate."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_fdtdx_dynamic_checkpoint_free_pte_iteration"
DYNAMIC_SUMMARY = RESULTS / "fdtdx_production_two_solve_equivalence_summary.json"
DYNAMIC_RAW = Path(
    "/home/seunghyun/tairte4/raw_artifacts/fdtdx_dynamic_checkpoint_free_pte_iteration/"
    "fdtdx_dynamic_checkpoint_free_pte_iteration_raw.npz"
)
FROZEN_SUMMARY = (
    HERE
    / "results_full_combined_pte_multidirection_adfd"
    / "full_combined_pte_multidirection_adfd_summary.json"
)
FROZEN_RAW = Path(
    "/home/seunghyun/tairte4/raw_artifacts/full_combined_pte_multidirection_adfd/"
    "full_combined_pte_multidirection_adfd_raw.npz"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    candidate_norm = float(np.linalg.norm(candidate))
    reference_norm = float(np.linalg.norm(reference))
    scale = max(reference_norm, 1.0e-300)
    cosine = float(
        np.sum(candidate * reference)
        / max(candidate_norm * reference_norm, 1.0e-300)
    )
    return {
        "candidate_norm_A": candidate_norm,
        "reference_norm_A": reference_norm,
        "normalized_vector_error": float(np.linalg.norm(candidate - reference) / scale),
        "norm_relative_error": abs(candidate_norm - reference_norm) / scale,
        "angle_deg": float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))),
    }


def main() -> int:
    dynamic_summary = json.loads(DYNAMIC_SUMMARY.read_text(encoding="utf-8"))
    frozen_summary = json.loads(FROZEN_SUMMARY.read_text(encoding="utf-8"))
    if dynamic_summary.get("status") != (
        "VALIDATED_FDTDX_PRODUCTION_DYNAMIC_CHECKPOINT_FREE_PTE_ITERATION"
    ):
        raise RuntimeError("Fail-closed dynamic iteration status")
    if frozen_summary.get("status") != (
        "VALIDATED_FULL_COMBINED_FDTDX_THERMAL_WEIGHTING_PTE_MULTIDIRECTION_ADFD"
    ):
        raise RuntimeError("Fail-closed frozen combined status")
    if sha256(DYNAMIC_RAW) != dynamic_summary["raw_artifact"]["sha256"]:
        raise RuntimeError("Fail-closed dynamic raw SHA")
    if sha256(FROZEN_RAW) != frozen_summary["raw_artifact"]["sha256"]:
        raise RuntimeError("Fail-closed frozen raw SHA")

    with np.load(DYNAMIC_RAW, allow_pickle=False) as raw:
        rho = np.asarray(raw["rho"], dtype=np.float64)
        dynamic = {
            "optical": np.asarray(raw["gradient_A"], dtype=np.float64),
            "thermal": np.asarray(raw["gradient_thermal_A"], dtype=np.float64),
            "electrical": np.asarray(raw["gradient_electrical_A"], dtype=np.float64),
            "combined": np.asarray(raw["gradient_combined_A"], dtype=np.float64),
        }
    with np.load(FROZEN_RAW, allow_pickle=False) as raw:
        frozen_rho = np.asarray(raw["rho"], dtype=np.float64)
        frozen = {
            "optical": np.asarray(raw["gradient_optical_A"], dtype=np.float64),
            "thermal": np.asarray(raw["gradient_thermal_A"], dtype=np.float64),
            "electrical": np.asarray(raw["gradient_electrical_A"], dtype=np.float64),
            "combined": np.asarray(raw["gradient_total_A"], dtype=np.float64),
        }
    if not np.array_equal(rho, frozen_rho):
        raise RuntimeError("Fail-closed baseline density mismatch")

    component_metrics = {
        name: metrics(dynamic[name], frozen[name]) for name in dynamic
    }
    dynamic_values = dynamic_summary["dynamic_PTE_iteration"]
    gates = dict(dynamic_summary["gates"])
    gates.update(
        {
            "dynamic_combined_vector_error_lt_1pct": component_metrics["combined"][
                "normalized_vector_error"
            ] < 0.01,
            "dynamic_combined_norm_error_lt_1pct": component_metrics["combined"][
                "norm_relative_error"
            ] < 0.01,
            "dynamic_combined_angle_lt_1deg": component_metrics["combined"][
                "angle_deg"
            ] < 1.0,
            "dynamic_direct_components_match_frozen_lt_0p5pct": max(
                component_metrics["thermal"]["normalized_vector_error"],
                component_metrics["electrical"]["normalized_vector_error"],
            ) < 0.005,
        }
    )
    status = (
        "VALIDATED_FDTDX_PRODUCTION_DYNAMIC_CHECKPOINT_FREE_PTE_ITERATION"
        if all(gates.values())
        else "FAILED_FDTDX_PRODUCTION_DYNAMIC_CHECKPOINT_FREE_PTE_ITERATION"
    )

    rows = [
        {"gradient_branch": name, **values}
        for name, values in component_metrics.items()
    ]
    csv_path = RESULTS / "fdtdx_dynamic_checkpoint_free_pte_iteration_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    scale = max(float(np.max(np.abs(dynamic["combined"]))), 1.0e-300)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    panels = (
        (axes[0, 0], rho, "current physical density", "gray", None, None),
        (axes[0, 1], dynamic["optical"], "dynamic Maxwell-source gradient", "coolwarm", -scale, scale),
        (axes[0, 2], dynamic["thermal"], "dynamic thermal/contact direct", "coolwarm", -scale, scale),
        (axes[1, 0], dynamic["electrical"], "dynamic electrical/weighting direct", "coolwarm", -scale, scale),
        (axes[1, 1], dynamic["combined"], "dynamic combined gradient", "coolwarm", -scale, scale),
        (axes[1, 2], dynamic["combined"] - frozen["combined"], "dynamic minus frozen", "coolwarm", None, None),
    )
    for axis, image, title, cmap, vmin, vmax in panels:
        artist = axis.imshow(image.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        fig.colorbar(artist, ax=axis)
    plot_path = RESULTS / "fdtdx_dynamic_checkpoint_free_pte_iteration.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    summary = {
        "status": status,
        "scope": (
            "one production-size dynamic physical-density PTE value-and-gradient "
            "evaluation: current Maxwell Q, current explicit thermal/electrical "
            "solutions and adjoints, current native-Yee dI/dQ, then one reciprocal "
            "Maxwell solve; no checkpoint stack, time-history reverse pass, or optimization"
        ),
        "method": dynamic_summary["method"],
        "contract": dynamic_summary["contract"],
        "dynamic_PTE_iteration": dynamic_values,
        "gradient_equivalence": component_metrics,
        "runtime": dynamic_summary["runtime"],
        "gates": gates,
        "raw_artifact": dynamic_summary["raw_artifact"],
        "outputs": {"metrics_csv": str(csv_path), "figure": str(plot_path)},
        "next_gate": (
            "call the dynamic evaluator from a short optimizer smoke while reusing "
            "compiled forward/adjoint executables; substrate provenance remains blocked"
        ),
    }
    summary_path = RESULTS / "fdtdx_dynamic_checkpoint_free_pte_iteration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = f"""# Dynamic checkpoint-free production PTE iteration

Status: `{status}`

Unlike the earlier fixed-weight equivalence, this run recomputed all
iteration-dependent quantities from the current density: native-Yee Maxwell
`Q`, explicit 3-D temperature, electrical weighting potential, thermal and
electrical adjoints, and the native-Yee `dI/dp` source weights. It then ran one
reciprocal Maxwell adjoint. No FDTD checkpoint or field time history was kept.

| metric | result |
|---|---:|
| dynamic combined vector error vs frozen AD | {100 * component_metrics['combined']['normalized_vector_error']:.6f}% |
| dynamic combined norm error | {100 * component_metrics['combined']['norm_relative_error']:.6f}% |
| dynamic combined angle | {component_metrics['combined']['angle_deg']:.6f} deg |
| native vs explicit source-adjoint contraction | {dynamic_values['native_vs_explicit_weighted_contraction_relative_error']:.3e} |
| PTE objective vs weighted-Q contraction | {dynamic_values['objective_vs_weighted_contraction_relative_error']:.3e} |
| forward + Maxwell adjoint execution | {dynamic_summary['runtime']['two_solve_execution_seconds']:.3f} s |
| first compile + forward + Maxwell adjoint | {dynamic_summary['runtime']['two_solve_compile_plus_execution_seconds']:.3f} s |
| full measured pipeline after runsetup audit | {dynamic_summary['runtime']['full_pipeline_seconds_from_audit_to_summary']:.3f} s |
| speedup of the two Maxwell solves vs frozen checkpoint AD | {dynamic_summary['runtime']['execution_speedup_vs_checkpointed']:.3f}x |

The full measured pipeline includes the current thermal/electrical solves,
their adjoints, remap pullback, plotting, and JSON/NPZ output. A persistent
optimizer process can reuse JIT compilation; this run does not yet measure
multi-iteration steady-state timing.

The inherited substrate status remains
`{dynamic_summary['contract']['substrate']['status']}`. This is a numerical
contract equivalence, not a paper-certified substrate material claim.
"""
    report_path = RESULTS / "FDTDX_DYNAMIC_CHECKPOINT_FREE_PTE_ITERATION_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    manifest = {
        "status": status,
        "raw_artifact": dynamic_summary["raw_artifact"],
        "inputs": {
            "frozen_combined": {
                "path": str(FROZEN_RAW),
                "bytes": FROZEN_RAW.stat().st_size,
                "sha256": sha256(FROZEN_RAW),
            }
        },
        "published": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in (summary_path, csv_path, plot_path, report_path)
        ],
        "generation_command": (
            "/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python "
            "78_publish_fdtdx_dynamic_checkpoint_free_pte_iteration.py"
        ),
    }
    (RESULTS / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
