#!/usr/bin/env python3
"""Publish the Run-002 production finite filter/projection certificate."""

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
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--failed-diagnostic-directory", type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw_directory.expanduser().resolve()
    failed_raw = args.failed_diagnostic_directory.expanduser().resolve()
    result_path = raw / "production_finite_filter_projection_result.json"
    failed_result_path = failed_raw / "production_finite_filter_projection_result.json"
    result = json.loads(result_path.read_text())
    failed_result = json.loads(failed_result_path.read_text())
    if result.get("status") != "VALIDATED_PRODUCTION_FINITE_FILTER_PROJECTION" or not result.get("passed"):
        raise RuntimeError("finite filter/projection is not validated")
    if failed_result.get("status") != "FAILED_PRODUCTION_FINITE_FILTER_PROJECTION" or failed_result.get("passed"):
        raise RuntimeError("the first fail-closed diagnostic is not preserved")
    artifact = Path(result["raw_artifact"]["path"])
    if record(artifact) != result["raw_artifact"]:
        raise RuntimeError("finite filter/projection artifact provenance mismatch")
    data = np.load(artifact)

    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS / "production_finite_filter_projection_summary.json"
    summary = dict(result)
    summary["preserved_failed_diagnostic"] = {
        "path": str(failed_result_path),
        "status": failed_result["status"],
        "reason": "initial gate incorrectly treated the coarsest centered-FD truncation error as the converged error",
        "worst_all_step_relative_error": failed_result["gates"]["mapping_only_fd_relative_error"]["value"],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    cases_path = RESULTS / "production_finite_filter_projection_cases.csv"
    with cases_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_type", "beta", "direction", "step", "adjoint", "finite_difference", "relative_error"],
            lineterminator="\n",
        )
        writer.writeheader()
        for case in result["dot_cases"]:
            writer.writerow(
                {
                    "case_type": "jvp_vjp_dot",
                    "beta": case["beta"],
                    "direction": case["direction"],
                    "step": "",
                    "adjoint": case["direction_dot_vjp"],
                    "finite_difference": case["jvp_dot_cotangent"],
                    "relative_error": case["cauchy_normalized_error"],
                }
            )
        for case in result["fd_cases"]:
            writer.writerow(
                {
                    "case_type": "mapping_only_centered_fd",
                    "beta": case["beta"],
                    "direction": case["direction"],
                    "step": case["step"],
                    "adjoint": case["adjoint_directional_derivative"],
                    "finite_difference": case["finite_difference_directional_derivative"],
                    "relative_error": case["relative_error"],
                }
            )

    report_path = RESULTS / "PRODUCTION_FINITE_FILTER_PROJECTION_REPORT.md"
    gates = result["gates"]
    report_path.write_text(
        f"""# Production finite filter/projection validation

Status: `{result['status']}`

The frozen 373×373, 50 nm nodal window uses a 500 nm conic radius and tanh
projection with eta=0.5. The filter is finite and nonperiodic. Its forward
operator is `D^-1 C`, where `C` is zero-padded convolution and `D` is the
truncated edge-kernel sum. The exact transpose is `C D^-1`; using the forward
normalization order as the transpose would be wrong at the boundary.

## Gates

| gate | result | limit |
|---|---:|---:|
| constant preservation | {gates['constant_preservation_max_abs']['value']:.3e} | <1e-14 |
| opposite-edge wrap | {gates['opposite_edge_wrap_max_abs']['value']:.3e} | exactly 0 |
| worst JVP/VJP Cauchy error | {gates['jvp_vjp_cauchy_normalized_error']['value']:.3e} | <1e-12 |
| worst mapping FD at h=2.5e-4 | {gates['mapping_only_fd_finest_step_relative_error']['value']:.3e} | <1e-5 |
| non-monotone h→h/2 trajectories | {gates['mapping_only_fd_h_to_h2_monotonic']['failure_count']} | 0 |

Five directions (uniform, smooth asymmetric, central localized,
design-edge localized, and fixed-seed random) were checked at beta
2, 4, 8, 16, and 32. Centered FD used h=0.001, 0.0005, and 0.00025 without
latent clipping. All 25 trajectories converge monotonically under h→h/2.

The first execution is retained as a fail-closed diagnostic. It used the
maximum error over *all* FD steps as the final gate, so the expected beta=32
coarse-step truncation error `{result['fd_diagnostics']['worst_relative_error_over_all_steps']:.3e}`
failed. No result was rescaled. The corrected certificate separately requires
monotonic step convergence and the declared finest-step tolerance.

This validates only the latent→finite-filter→projection mapping and its exact
transpose. It is not an exact-binary DRC certificate, gray-law certificate,
full Maxwell/thermal latent AD-FD certificate, or authorization to optimize.
No Maxwell solve, thermal solve, or optimizer iteration ran here.
"""
    )

    latent = np.asarray(data["latent"], float)
    filtered = np.asarray(data["filtered"], float)
    projected = np.asarray(data["projected_beta8"], float)
    edge = np.asarray(data["edge_impulse_filtered"], float)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.3), constrained_layout=True)
    for axis, field, title in zip(
        axes[0],
        (latent, filtered, projected),
        ("latent", "finite conic filtered", "projected (beta=8)"),
    ):
        image = axis.imshow(field.T, origin="lower", cmap="viridis", extent=(-9.3, 9.3, -9.3, 9.3))
        fig.colorbar(image, ax=axis)
        axis.set(title=title, xlabel="x=b (µm)", ylabel="y=a (µm)")
    axes[1, 0].imshow(edge.T, origin="lower", cmap="magma", aspect="auto")
    axes[1, 0].set(title="left-edge impulse: no opposite wrap", xlabel="x index", ylabel="y index")
    for name in result["directions"]:
        selected = [case for case in result["fd_cases"] if case["direction"] == name and case["beta"] == 32.0]
        axes[1, 1].loglog([case["step"] for case in selected], [case["relative_error"] for case in selected], "o-", label=name)
    axes[1, 1].axhline(1e-5, color="black", ls="--", label="finest-step gate")
    axes[1, 1].set(title="beta=32 mapping-only FD", xlabel="centered-FD step", ylabel="relative error")
    axes[1, 1].legend(fontsize=7)
    betas = sorted(set(case["beta"] for case in result["dot_cases"]))
    dot_max = [max(case["cauchy_normalized_error"] for case in result["dot_cases"] if case["beta"] == beta) for beta in betas]
    axes[1, 2].semilogy(betas, dot_max, "o-")
    axes[1, 2].axhline(1e-12, color="black", ls="--")
    axes[1, 2].set(title="worst JVP/VJP dot error", xlabel="projection beta", ylabel="Cauchy-normalized error")
    fig.suptitle("Run 002 finite nonperiodic production density mapping")
    plot_path = PLOTS / "production_finite_filter_projection.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    manifest = json.loads(MANIFEST.read_text())
    manifest["production_finite_filter_projection"] = {
        "status": result["status"],
        "raw_artifacts_committed_to_git": False,
        "raw_directory": str(raw),
        "artifacts": [record(result_path), record(artifact)],
        "preserved_failed_diagnostic": {
            "raw_directory": str(failed_raw),
            "artifacts": [
                record(failed_result_path),
                record(Path(failed_result["raw_artifact"]["path"])),
            ],
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "report": str(report_path), "summary": str(summary_path), "cases": str(cases_path), "plot": str(plot_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
