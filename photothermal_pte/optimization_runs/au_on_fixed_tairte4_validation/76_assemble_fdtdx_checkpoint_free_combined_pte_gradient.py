#!/usr/bin/env python3
"""Assemble and certify the checkpoint-free combined physical-rho gradient.

This is an offline operation.  It replaces only the frozen checkpointed
Maxwell-source branch by the validated forward-plus-adjoint two-solve branch,
then adds the independently validated fixed-Q thermal/contact and
electrical/weighting direct branches.  No FDTD, thermal, electrical, FD, or
optimization solve is executed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_TWO_SOLVE_SUMMARY = (
    HERE
    / "results_fdtdx_production_two_solve_equivalence"
    / "fdtdx_production_two_solve_equivalence_summary.json"
)
DEFAULT_DIRECT_SUMMARY = (
    HERE
    / "results_explicit_thermal_weighting_fixed_spatial_q_adfd"
    / "explicit_thermal_weighting_fixed_spatial_q_adfd_summary.json"
)
DEFAULT_FROZEN_SUMMARY = (
    HERE
    / "results_full_combined_pte_multidirection_adfd"
    / "full_combined_pte_multidirection_adfd_summary.json"
)
DEFAULT_TWO_SOLVE_RAW = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "fdtdx_production_two_solve_equivalence/"
    "fdtdx_production_two_solve_gradient.npz"
)
DEFAULT_DIRECT_RAW = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "explicit_thermal_weighting_fixed_spatial_q_adfd/"
    "explicit_thermal_weighting_fixed_spatial_q_adfd_raw.npz"
)
DEFAULT_FROZEN_RAW = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "full_combined_pte_multidirection_adfd/"
    "full_combined_pte_multidirection_adfd_raw.npz"
)
DEFAULT_OUTPUT = HERE / "results_fdtdx_checkpoint_free_combined_pte_gradient"
DEFAULT_RAW_OUTPUT = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "fdtdx_checkpoint_free_combined_pte_gradient/"
    "fdtdx_checkpoint_free_combined_pte_gradient_raw.npz"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vector_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
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


def load_summary(path: Path, expected_status: str) -> dict[str, object]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != expected_status:
        raise RuntimeError(
            f"Fail-closed status mismatch for {path}: {summary.get('status')}"
        )
    return summary


def run(
    output_dir: Path,
    raw_output: Path,
    two_solve_summary_path: Path,
    direct_summary_path: Path,
    frozen_summary_path: Path,
    two_solve_raw_path: Path,
    direct_raw_path: Path,
    frozen_raw_path: Path,
) -> dict[str, object]:
    two_summary = load_summary(
        two_solve_summary_path,
        "VALIDATED_FDTDX_PRODUCTION_CHECKPOINT_FREE_TWO_SOLVE_EQUIVALENCE",
    )
    direct_summary = load_summary(
        direct_summary_path,
        "VALIDATED_EXPLICIT_THERMAL_WEIGHTING_FIXED_SPATIAL_Q_ADFD",
    )
    frozen_summary = load_summary(
        frozen_summary_path,
        "VALIDATED_FULL_COMBINED_FDTDX_THERMAL_WEIGHTING_PTE_MULTIDIRECTION_ADFD",
    )

    inputs = {
        "two_solve": {
            "path": two_solve_raw_path,
            "sha256": sha256(two_solve_raw_path),
            "expected_sha256": two_summary["raw_artifact"]["sha256"],
        },
        "direct": {
            "path": direct_raw_path,
            "sha256": sha256(direct_raw_path),
            "expected_sha256": direct_summary["raw_artifact"]["sha256"],
        },
        "frozen_combined": {
            "path": frozen_raw_path,
            "sha256": sha256(frozen_raw_path),
            "expected_sha256": frozen_summary["raw_artifact"]["sha256"],
        },
    }
    for name, item in inputs.items():
        if item["sha256"] != item["expected_sha256"]:
            raise RuntimeError(f"Fail-closed SHA mismatch for {name}")

    with np.load(two_solve_raw_path, allow_pickle=False) as raw:
        rho_two = np.asarray(raw["rho"], dtype=np.float64)
        optical = np.asarray(raw["gradient_A"], dtype=np.float64)
    with np.load(direct_raw_path, allow_pickle=False) as raw:
        rho_direct = np.asarray(raw["rho"], dtype=np.float64)
        thermal = np.asarray(
            raw["gradient_thermal_thermally_grown_A"], dtype=np.float64
        )
        electrical = np.asarray(
            raw["gradient_electrical_thermally_grown_A"], dtype=np.float64
        )
    with np.load(frozen_raw_path, allow_pickle=False) as raw:
        rho_frozen = np.asarray(raw["rho"], dtype=np.float64)
        frozen_optical = np.asarray(raw["gradient_optical_A"], dtype=np.float64)
        frozen_thermal = np.asarray(raw["gradient_thermal_A"], dtype=np.float64)
        frozen_electrical = np.asarray(raw["gradient_electrical_A"], dtype=np.float64)
        frozen_total = np.asarray(raw["gradient_total_A"], dtype=np.float64)
        directions = {
            key.removeprefix("direction_"): np.asarray(raw[key], dtype=np.float64)
            for key in raw.files
            if key.startswith("direction_")
        }

    if not (
        np.array_equal(rho_two, rho_direct)
        and np.array_equal(rho_two, rho_frozen)
    ):
        raise RuntimeError("Fail-closed baseline density mismatch")
    if not np.array_equal(thermal, frozen_thermal):
        raise RuntimeError("Fail-closed thermal direct-gradient mismatch")
    if not np.array_equal(electrical, frozen_electrical):
        raise RuntimeError("Fail-closed electrical direct-gradient mismatch")

    combined = optical + thermal + electrical
    optical_metrics = vector_metrics(optical, frozen_optical)
    combined_metrics = vector_metrics(combined, frozen_total)
    frozen_baseline_objective = float(
        direct_summary["scenarios"]["thermally_grown"]["objective_A"]
    )
    baseline_objective_error = abs(
        float(two_summary["results"]["objective_A"]) - frozen_baseline_objective
    ) / max(abs(frozen_baseline_objective), 1e-300)

    direction_rows: list[dict[str, object]] = []
    combined_norm = max(float(np.linalg.norm(combined)), 1.0e-300)
    for frozen_row in frozen_summary["directions"]:
        name = str(frozen_row["direction"])
        if name == "combined_adjoint_aligned":
            direction = frozen_total / max(float(np.linalg.norm(frozen_total)), 1e-300)
        else:
            direction = directions[name]
        ad_optical = float(np.sum(optical * direction))
        ad_thermal = float(np.sum(thermal * direction))
        ad_electrical = float(np.sum(electrical * direction))
        ad_total = ad_optical + ad_thermal + ad_electrical
        fd_total = float(frozen_row["FD_total_A"])
        absolute_error = abs(ad_total - fd_total)
        strength = abs(fd_total) / combined_norm
        direction_rows.append(
            {
                "direction": name,
                "h": float(frozen_row["h"]),
                "AD_optical_A": ad_optical,
                "AD_thermal_A": ad_thermal,
                "AD_electrical_A": ad_electrical,
                "AD_total_A": ad_total,
                "frozen_FD_total_A": fd_total,
                "direction_strength_vs_gradient_norm": strength,
                "strong": strength >= 0.01,
                "strong_relative_error": absolute_error / max(abs(fd_total), 1.0e-300),
                "gradient_l2_normalized_error": absolute_error / combined_norm,
            }
        )

    strong_errors = [
        float(row["strong_relative_error"])
        for row in direction_rows
        if bool(row["strong"])
    ]
    normalized_errors = [
        float(row["gradient_l2_normalized_error"]) for row in direction_rows
    ]
    gates = {
        "all_input_statuses_and_SHA256_verified": True,
        "baseline_density_exactly_identical": True,
        "direct_thermal_gradient_exactly_preserved": True,
        "direct_electrical_gradient_exactly_preserved": True,
        "checkpoint_free_optical_vector_error_lt_1pct": optical_metrics[
            "normalized_vector_error"
        ] < 0.01,
        "combined_vector_error_lt_1pct": combined_metrics[
            "normalized_vector_error"
        ] < 0.01,
        "combined_norm_error_lt_1pct": combined_metrics["norm_relative_error"] < 0.01,
        "combined_angle_lt_1deg": combined_metrics["angle_deg"] < 1.0,
        "strong_direction_error_lt_1pct": max(strong_errors) < 0.01,
        "all_direction_normalized_error_lt_1pct": max(normalized_errors) < 0.01,
        "checkpoint_count_zero": int(two_summary["method"]["checkpoint_count"]) == 0,
        "time_history_saved_false": not bool(two_summary["method"]["time_history_saved"]),
        "no_empirical_gradient_rescaling": not bool(
            two_summary["method"]["empirical_gradient_rescaling"]
        ),
    }
    status = (
        "VALIDATED_FDTDX_PRODUCTION_CHECKPOINT_FREE_COMBINED_PTE_GRADIENT_EQUIVALENCE"
        if all(gates.values())
        else "FAILED_FDTDX_PRODUCTION_CHECKPOINT_FREE_COMBINED_PTE_GRADIENT_EQUIVALENCE"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        raw_output,
        rho=rho_two,
        gradient_optical_checkpoint_free_A=optical,
        gradient_thermal_direct_A=thermal,
        gradient_electrical_direct_A=electrical,
        gradient_total_checkpoint_free_A=combined,
        gradient_total_frozen_reference_A=frozen_total,
    )
    raw_artifact = {
        "path": str(raw_output),
        "bytes": raw_output.stat().st_size,
        "sha256": sha256(raw_output),
        "committed_to_git": False,
    }

    csv_path = output_dir / "fdtdx_checkpoint_free_combined_pte_gradient_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(direction_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(direction_rows)

    vmax = max(
        float(np.max(np.abs(optical))),
        float(np.max(np.abs(thermal))),
        float(np.max(np.abs(electrical))),
        float(np.max(np.abs(combined))),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    panels = (
        (axes[0, 0], rho_two, "baseline density", "gray", None, None),
        (axes[0, 1], optical, "checkpoint-free optical", "coolwarm", -vmax, vmax),
        (axes[0, 2], thermal, "thermal/contact direct", "coolwarm", -vmax, vmax),
        (axes[1, 0], electrical, "electrical/weighting direct", "coolwarm", -vmax, vmax),
        (axes[1, 1], combined, "checkpoint-free combined", "coolwarm", -vmax, vmax),
        (
            axes[1, 2],
            combined - frozen_total,
            "combined minus frozen reference",
            "coolwarm",
            None,
            None,
        ),
    )
    for axis, image, title, cmap, vmin, vmax_panel in panels:
        artist = axis.imshow(
            image.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax_panel
        )
        axis.set_title(title)
        fig.colorbar(artist, ax=axis)
    figure_path = output_dir / "fdtdx_checkpoint_free_combined_pte_gradient.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    manifest = {
        "status": status,
        "raw_artifact": raw_artifact,
        "inputs": {
            key: {
                "path": str(value["path"]),
                "bytes": Path(value["path"]).stat().st_size,
                "sha256": value["sha256"],
            }
            for key, value in inputs.items()
        },
        "generation_command": (
            "/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python "
            "76_assemble_fdtdx_checkpoint_free_combined_pte_gradient.py"
        ),
    }
    manifest_path = output_dir / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    summary = {
        "status": status,
        "scope": (
            "offline assembly of the validated production checkpoint-free Maxwell-source "
            "gradient with frozen independently validated thermal/contact and "
            "electrical/weighting direct gradients; no new solve or optimization"
        ),
        "scenario": "thermally_grown",
        "method": {
            "gradient_equation": "g_total = g_Maxwell,no-checkpoint + g_thermal + g_electrical",
            "forward_solves_per_optical_gradient": 1,
            "adjoint_solves_per_optical_gradient": 1,
            "checkpoint_count": 0,
            "time_history_saved": False,
            "empirical_gradient_rescaling": False,
        },
        "gradient_decomposition": {
            "optical_norm_A": float(np.linalg.norm(optical)),
            "thermal_direct_norm_A": float(np.linalg.norm(thermal)),
            "electrical_direct_norm_A": float(np.linalg.norm(electrical)),
            "combined_norm_A": float(np.linalg.norm(combined)),
        },
        "optical_equivalence": optical_metrics,
        "combined_equivalence": combined_metrics,
        "baseline_objective_relative_error": baseline_objective_error,
        "directions": direction_rows,
        "worst_strong_direction_relative_error": max(strong_errors),
        "worst_all_direction_gradient_l2_normalized_error": max(normalized_errors),
        "runtime": two_summary["runtime"],
        "known_material_scope": two_summary["contract"]["substrate"],
        "gates": gates,
        "raw_artifact": raw_artifact,
        "outputs": {
            "cases_csv": str(csv_path),
            "figure": str(figure_path),
            "manifest": str(manifest_path),
        },
        "next_gate": (
            "replace the checkpointed Maxwell VJP inside the already validated latent/"
            "filter/projection chain, then run a short production optimizer smoke; the "
            "substrate material provenance remains explicitly blocked"
        ),
    }
    summary_path = output_dir / "fdtdx_checkpoint_free_combined_pte_gradient_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = rf"""# Checkpoint-free production combined PTE gradient

Status: `{status}`

This certificate changes only the Maxwell-source derivative implementation.
The old checkpointed reverse-through-time result remains immutable as the
reference. The new route stores no time history and performs one forward plus
one reciprocal adjoint FDTD solve. No empirical normalization or gradient
rescaling is used.

## Combined chain

\[
g_{{\rho}} = g_{{\rm Maxwell,no\ checkpoint}}
              + g_{{\rm thermal/contact}}
              + g_{{\rm electrical/weighting}}.
\]

The thermal/contact and electrical/weighting arrays are bitwise identical to
the independently validated fixed-spatial-Q certificate. Only the optical
array was replaced.

| metric | result |
|---|---:|
| optical vector error vs frozen checkpoint | {100.0 * optical_metrics['normalized_vector_error']:.6f}% |
| combined vector error vs frozen checkpoint | {100.0 * combined_metrics['normalized_vector_error']:.6f}% |
| combined norm error | {100.0 * combined_metrics['norm_relative_error']:.6f}% |
| combined angle | {combined_metrics['angle_deg']:.6f} deg |
| worst strong-direction AD--FD error | {100.0 * max(strong_errors):.6f}% |
| worst all-direction normalized error | {100.0 * max(normalized_errors):.6f}% |
| forward + adjoint execution | {two_summary['runtime']['two_solve_execution_seconds']:.3f} s |
| speedup vs frozen checkpointed AD | {two_summary['runtime']['execution_speedup_vs_checkpointed']:.3f}x |

The `design_edge_localized` direction is near-null; its raw relative error is
reported in the CSV but is not treated as a strong-direction metric. Its error
normalized by the full gradient norm remains below 1%.

## Important scope boundary

This validates the derivative implementation for the frozen explicit optical
material contract. It does not turn the inherited substrate model into a
paper-certified material: the recorded substrate status remains
`{two_summary['contract']['substrate']['status']}`. It also does not execute an
optimization.

Raw NPZ arrays are outside Git and are pinned in `RAW_ARTIFACT_MANIFEST.json`.
"""
    report_path = output_dir / "FDTDX_CHECKPOINT_FREE_COMBINED_PTE_GRADIENT_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if not all(gates.values()):
        raise SystemExit(2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--two-solve-summary", type=Path, default=DEFAULT_TWO_SOLVE_SUMMARY)
    parser.add_argument("--direct-summary", type=Path, default=DEFAULT_DIRECT_SUMMARY)
    parser.add_argument("--frozen-summary", type=Path, default=DEFAULT_FROZEN_SUMMARY)
    parser.add_argument("--two-solve-raw", type=Path, default=DEFAULT_TWO_SOLVE_RAW)
    parser.add_argument("--direct-raw", type=Path, default=DEFAULT_DIRECT_RAW)
    parser.add_argument("--frozen-raw", type=Path, default=DEFAULT_FROZEN_RAW)
    args = parser.parse_args()
    run(
        args.output_dir,
        args.raw_output,
        args.two_solve_summary,
        args.direct_summary,
        args.frozen_summary,
        args.two_solve_raw,
        args.direct_raw,
        args.frozen_raw,
    )


if __name__ == "__main__":
    main()
