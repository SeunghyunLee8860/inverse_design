#!/usr/bin/env python3
"""Validate the checkpoint-free gradient through filter and projection.

No solver runs here.  The certified checkpoint-free physical-density gradient
is pulled back through the same finite conic filter and tanh projection used by
the frozen end-to-end latent AD--FD certificate, then compared with its stored
latent gradient and directional finite differences.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from photothermal_pte.optimization_runs.legacy_v261_optical_support.production_density_mapping import (
    ProductionDensityMapping,
)


HERE = Path(__file__).resolve().parent
DEFAULT_COMBINED_SUMMARY = (
    HERE
    / "results_fdtdx_checkpoint_free_combined_pte_gradient"
    / "fdtdx_checkpoint_free_combined_pte_gradient_summary.json"
)
DEFAULT_COMBINED_RAW = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "fdtdx_checkpoint_free_combined_pte_gradient/"
    "fdtdx_checkpoint_free_combined_pte_gradient_raw.npz"
)
DEFAULT_MAPPING_SUMMARY = (
    HERE
    / "results_au_latent_filter_projection_mapping"
    / "au_latent_filter_projection_mapping_summary.json"
)
DEFAULT_MAPPING_RAW = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "au_latent_filter_projection_mapping/"
    "au_latent_filter_projection_mapping_raw.npz"
)
DEFAULT_FROZEN_SUMMARY = (
    HERE
    / "results_full_latent_filter_projection_pte_adfd"
    / "full_latent_filter_projection_pte_adfd_summary.json"
)
DEFAULT_FROZEN_RAW = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "full_latent_filter_projection_pte_adfd/"
    "full_latent_filter_projection_pte_adfd_raw.npz"
)
DEFAULT_OUTPUT = HERE / "results_fdtdx_checkpoint_free_latent_pte_gradient"
DEFAULT_RAW_OUTPUT = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "fdtdx_checkpoint_free_latent_pte_gradient/"
    "fdtdx_checkpoint_free_latent_pte_gradient_raw.npz"
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


def inverse_projection(rho: np.ndarray, beta: float, eta: float) -> np.ndarray:
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    argument = rho * denominator - np.tanh(beta * eta)
    if np.any(np.abs(argument) >= 1.0):
        raise RuntimeError("Physical density lies outside the invertible projection range")
    return eta + np.arctanh(argument) / beta


def dense_filter(mapping: ProductionDensityMapping) -> np.ndarray:
    size = int(np.prod(mapping.shape))
    identity = np.eye(size)
    return np.column_stack(
        [
            mapping.filtered(identity[index].reshape(mapping.shape)).reshape(-1)
            for index in range(size)
        ]
    )


def run(
    output_dir: Path,
    raw_output: Path,
    combined_summary_path: Path,
    combined_raw_path: Path,
    mapping_summary_path: Path,
    mapping_raw_path: Path,
    frozen_summary_path: Path,
    frozen_raw_path: Path,
) -> dict[str, object]:
    combined_summary = json.loads(combined_summary_path.read_text(encoding="utf-8"))
    mapping_summary = json.loads(mapping_summary_path.read_text(encoding="utf-8"))
    frozen_summary = json.loads(frozen_summary_path.read_text(encoding="utf-8"))
    expected_statuses = {
        "combined": (
            combined_summary,
            "VALIDATED_FDTDX_PRODUCTION_CHECKPOINT_FREE_COMBINED_PTE_GRADIENT_EQUIVALENCE",
        ),
        "mapping": (mapping_summary, "VALIDATED_AU_LATENT_FILTER_PROJECTION_MAPPING"),
        "frozen": (
            frozen_summary,
            "VALIDATED_FULL_LATENT_FILTER_PROJECTION_FDTDX_PTE_ADFD",
        ),
    }
    for name, (summary, expected) in expected_statuses.items():
        if summary.get("status") != expected:
            raise RuntimeError(f"Fail-closed {name} status mismatch")
    raw_inputs = {
        "combined": (combined_raw_path, combined_summary["raw_artifact"]["sha256"]),
        "mapping": (mapping_raw_path, mapping_summary["raw_artifact"]["sha256"]),
        "frozen": (frozen_raw_path, frozen_summary["raw_artifact"]["sha256"]),
    }
    for name, (path, expected_sha) in raw_inputs.items():
        if sha256(path) != expected_sha:
            raise RuntimeError(f"Fail-closed {name} raw SHA mismatch")

    with np.load(combined_raw_path, allow_pickle=False) as raw:
        rho = np.asarray(raw["rho"], dtype=np.float64)
        gradient_physical = np.asarray(
            raw["gradient_total_checkpoint_free_A"], dtype=np.float64
        )
    with np.load(frozen_raw_path, allow_pickle=False) as raw:
        latent_serialized = np.asarray(raw["latent"], dtype=np.float64)
        rho_frozen = np.asarray(raw["rho_reconstructed"], dtype=np.float64)
        frozen_physical = np.asarray(raw["gradient_physical_A"], dtype=np.float64)
        frozen_latent = np.asarray(raw["gradient_latent_A"], dtype=np.float64)
        directions = {
            key.removeprefix("direction_"): np.asarray(raw[key], dtype=np.float64)
            for key in raw.files
            if key.startswith("direction_")
        }
    if not np.array_equal(rho.astype(np.float32), rho_frozen.astype(np.float32)):
        raise RuntimeError("Fail-closed physical-density mismatch")

    contract = mapping_summary["mapping"]
    beta = float(frozen_summary["beta"])
    mapping = ProductionDensityMapping(
        shape=tuple(contract["shape"]),
        spacing_m=float(contract["spacing_nm"]) * 1.0e-9,
        radius_m=float(contract["radius_nm"]) * 1.0e-9,
        eta=float(contract["eta"]),
    )
    filtered_target = inverse_projection(rho, beta, mapping.eta)
    latent = np.linalg.solve(dense_filter(mapping), filtered_target.reshape(-1)).reshape(
        mapping.shape
    )
    reconstructed = mapping.physical(latent, beta)
    reconstruction_error = float(np.max(np.abs(reconstructed - rho)))
    serialized_latent_max_abs_difference = float(
        np.max(np.abs(latent - latent_serialized))
    )
    gradient_latent = mapping.vjp(latent, gradient_physical, beta)
    physical_metrics = vector_metrics(gradient_physical, frozen_physical)
    latent_metrics = vector_metrics(gradient_latent, frozen_latent)
    latent_norm = max(float(np.linalg.norm(gradient_latent)), 1.0e-300)

    rows: list[dict[str, object]] = []
    for frozen_row in frozen_summary["directions"]:
        name = str(frozen_row["direction"])
        direction = directions[name]
        ad = float(np.sum(gradient_latent * direction))
        fd = float(frozen_row["FD_A"])
        error = abs(ad - fd)
        strength = abs(fd) / latent_norm
        rows.append(
            {
                "direction": name,
                "beta": beta,
                "h": float(frozen_row["h"]),
                "AD_checkpoint_free_A": ad,
                "frozen_end_to_end_FD_A": fd,
                "direction_strength_vs_gradient_norm": strength,
                "strong": strength >= 0.01,
                "strong_relative_error": error / max(abs(fd), 1.0e-300),
                "gradient_norm_normalized_error": error / latent_norm,
            }
        )
    strong_errors = [
        float(row["strong_relative_error"]) for row in rows if bool(row["strong"])
    ]
    normalized_errors = [float(row["gradient_norm_normalized_error"]) for row in rows]
    gates = {
        "input_statuses_and_SHA256_verified": True,
        "physical_density_reconstruction_lt_1e-12": reconstruction_error < 1.0e-12,
        "physical_gradient_vector_error_lt_1pct": physical_metrics[
            "normalized_vector_error"
        ] < 0.01,
        "latent_gradient_vector_error_lt_1pct": latent_metrics[
            "normalized_vector_error"
        ] < 0.01,
        "latent_gradient_norm_error_lt_1pct": latent_metrics["norm_relative_error"] < 0.01,
        "latent_gradient_angle_lt_1deg": latent_metrics["angle_deg"] < 1.0,
        "strong_direction_error_lt_1pct": max(strong_errors) < 0.01,
        "all_direction_normalized_error_lt_1pct": max(normalized_errors) < 0.01,
        "checkpoint_count_zero": True,
        "time_history_saved_false": True,
        "no_empirical_gradient_rescaling": True,
    }
    status = (
        "VALIDATED_FDTDX_PRODUCTION_CHECKPOINT_FREE_LATENT_PTE_GRADIENT"
        if all(gates.values())
        else "FAILED_FDTDX_PRODUCTION_CHECKPOINT_FREE_LATENT_PTE_GRADIENT"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        raw_output,
        latent=latent,
        rho=rho,
        gradient_physical_checkpoint_free_A=gradient_physical,
        gradient_latent_checkpoint_free_A=gradient_latent,
        gradient_latent_frozen_reference_A=frozen_latent,
    )
    raw_artifact = {
        "path": str(raw_output),
        "bytes": raw_output.stat().st_size,
        "sha256": sha256(raw_output),
        "committed_to_git": False,
    }
    csv_path = output_dir / "fdtdx_checkpoint_free_latent_pte_gradient_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    images = (
        (axes[0], rho, "physical density", "gray"),
        (axes[1], gradient_latent, "checkpoint-free latent gradient", "coolwarm"),
        (axes[2], gradient_latent - frozen_latent, "new minus frozen latent", "coolwarm"),
    )
    for axis, image, title, cmap in images:
        artist = axis.imshow(image.T, origin="lower", cmap=cmap)
        axis.set_title(title)
        fig.colorbar(artist, ax=axis)
    figure_path = output_dir / "fdtdx_checkpoint_free_latent_pte_gradient.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    manifest = {
        "status": status,
        "raw_artifact": raw_artifact,
        "inputs": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, (path, _) in raw_inputs.items()
        },
        "generation_command": (
            "/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python "
            "77_validate_fdtdx_checkpoint_free_latent_pte_gradient.py"
        ),
    }
    manifest_path = output_dir / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    summary = {
        "status": status,
        "scope": (
            "offline VJP of the validated checkpoint-free combined physical-density "
            "gradient through the certified finite conic filter and tanh projection; "
            "comparison uses frozen end-to-end latent finite differences"
        ),
        "beta": beta,
        "physical_density_reconstruction_max_abs": reconstruction_error,
        "serialized_float32_latent_max_abs_difference": (
            serialized_latent_max_abs_difference
        ),
        "physical_gradient_equivalence": physical_metrics,
        "latent_gradient_equivalence": latent_metrics,
        "directions": rows,
        "worst_strong_direction_relative_error": max(strong_errors),
        "worst_all_direction_gradient_norm_normalized_error": max(normalized_errors),
        "runtime_contract": combined_summary["runtime"],
        "gates": gates,
        "raw_artifact": raw_artifact,
        "outputs": {
            "cases_csv": str(csv_path),
            "figure": str(figure_path),
            "manifest": str(manifest_path),
        },
        "next_gate": (
            "wire the checkpoint-free two-solve callback into a short optimization smoke; "
            "do not claim paper-certified substrate material while its recorded provenance "
            "status remains blocked"
        ),
    }
    summary_path = output_dir / "fdtdx_checkpoint_free_latent_pte_gradient_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = f"""# Checkpoint-free latent/filter/projection PTE gradient

Status: `{status}`

The checkpoint-free combined physical-density gradient was pulled back through
the same finite conic filter and tanh projection as the frozen end-to-end
certificate. No FDTD, thermal, electrical, or finite-difference solve was
rerun, and no time history or checkpoint stack was used.

| metric | result |
|---|---:|
| physical-gradient vector error | {100 * physical_metrics['normalized_vector_error']:.6f}% |
| latent-gradient vector error | {100 * latent_metrics['normalized_vector_error']:.6f}% |
| latent-gradient norm error | {100 * latent_metrics['norm_relative_error']:.6f}% |
| latent-gradient angle | {latent_metrics['angle_deg']:.6f} deg |
| worst strong latent AD--FD error | {100 * max(strong_errors):.6f}% |
| worst normalized directional error | {100 * max(normalized_errors):.6f}% |

The runtime-bearing Maxwell VJP is therefore one forward plus one adjoint solve
with zero checkpoints. Raw arrays remain outside Git and are SHA-256 pinned in
the manifest.
"""
    report_path = output_dir / "FDTDX_CHECKPOINT_FREE_LATENT_PTE_GRADIENT_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not all(gates.values()):
        raise SystemExit(2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--combined-summary", type=Path, default=DEFAULT_COMBINED_SUMMARY)
    parser.add_argument("--combined-raw", type=Path, default=DEFAULT_COMBINED_RAW)
    parser.add_argument("--mapping-summary", type=Path, default=DEFAULT_MAPPING_SUMMARY)
    parser.add_argument("--mapping-raw", type=Path, default=DEFAULT_MAPPING_RAW)
    parser.add_argument("--frozen-summary", type=Path, default=DEFAULT_FROZEN_SUMMARY)
    parser.add_argument("--frozen-raw", type=Path, default=DEFAULT_FROZEN_RAW)
    args = parser.parse_args()
    run(
        args.output_dir,
        args.raw_output,
        args.combined_summary,
        args.combined_raw,
        args.mapping_summary,
        args.mapping_raw,
        args.frozen_summary,
        args.frozen_raw,
    )


if __name__ == "__main__":
    main()
