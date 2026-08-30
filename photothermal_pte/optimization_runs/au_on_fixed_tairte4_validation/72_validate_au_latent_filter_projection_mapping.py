#!/usr/bin/env python3
"""Certify the 20x20 Au latent -> filter -> projection mapping.

This is a solver-free layout/chain-rule gate.  It does not select a final
fabrication rule or run an optimizer.  The numerical mapping uses a finite,
nonperiodic, row-normalized conic filter and the exact transpose followed by
the standard eta=0.5 tanh projection.
"""

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

from photothermal_pte.optimization_runs.legacy_v261_optical_support.production_density_mapping import (
    ProductionDensityMapping,
)


STATUS_PASS = "VALIDATED_AU_LATENT_FILTER_PROJECTION_MAPPING"
STATUS_FAIL = "FAILED_AU_LATENT_FILTER_PROJECTION_MAPPING"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directions(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    x = np.linspace(-1.0, 1.0, shape[0])[:, None]
    y = np.linspace(-1.0, 1.0, shape[1])[None, :]
    rng = np.random.default_rng(20260821)
    raw = {
        "uniform": np.ones(shape),
        "smooth_asymmetric": np.sin(0.72 * np.pi * x) * np.cos(0.53 * np.pi * y) + 0.2 * y,
        "central_localized": np.exp(-((x - 0.08) ** 2 + (y + 0.1) ** 2) / 0.07),
        "design_edge_localized": np.exp(-((x + 0.86) ** 2 + (y - 0.3) ** 2) / 0.025),
        "fixed_seed_random": rng.normal(size=shape),
    }
    return {name: value / np.linalg.norm(value) for name, value in raw.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gradient-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--raw-output-npz", required=True, type=Path)
    parser.add_argument("--spacing-nm", type=float, default=500.0)
    parser.add_argument("--filter-radius-nm", type=float, default=750.0)
    args = parser.parse_args()
    if args.spacing_nm <= 0.0 or args.filter_radius_nm <= args.spacing_nm:
        raise ValueError("filter radius must exceed one design-pixel spacing")

    input_path = args.physical_gradient_npz.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    raw_output = args.raw_output_npz.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(input_path, allow_pickle=False) as raw:
        latent = np.asarray(raw["rho"], dtype=np.float64)
        if "gradient_total_A" in raw:
            cotangent = np.asarray(raw["gradient_total_A"], dtype=np.float64)
        else:
            cotangent = np.asarray(raw["gradient_A"], dtype=np.float64)
    if latent.shape != (20, 20) or cotangent.shape != latent.shape:
        raise RuntimeError("Expected matching 20x20 latent/cotangent arrays")
    if not np.all(np.isfinite(latent)) or not np.all(np.isfinite(cotangent)):
        raise RuntimeError("Non-finite mapping input")
    if np.min(latent) <= 0.0 or np.max(latent) >= 1.0:
        raise RuntimeError("Mapping FD baseline must be strictly inside (0,1)")

    mapping = ProductionDensityMapping(
        shape=latent.shape,
        spacing_m=args.spacing_nm * 1.0e-9,
        radius_m=args.filter_radius_nm * 1.0e-9,
        eta=0.5,
    )
    directions = _directions(latent.shape)
    betas = (1.0, 2.0, 4.0, 8.0)
    steps = (0.01, 0.005, 0.0025)
    rows: list[dict[str, object]] = []
    worst_dot = 0.0
    worst_fd = 0.0
    monotone_failures = 0
    rng = np.random.default_rng(20260822)
    transpose_cotangent = rng.normal(size=latent.shape)
    for beta in betas:
        for direction_name, direction in directions.items():
            jvp = mapping.jvp(latent, direction, beta)
            vjp = mapping.vjp(latent, transpose_cotangent, beta)
            left = float(np.vdot(jvp, transpose_cotangent))
            right = float(np.vdot(direction, vjp))
            dot_scale = max(
                float(np.linalg.norm(jvp) * np.linalg.norm(transpose_cotangent)),
                np.finfo(float).tiny,
            )
            dot_error = abs(left - right) / dot_scale
            worst_dot = max(worst_dot, dot_error)
            gradient_latent = mapping.vjp(latent, cotangent, beta)
            ad = float(np.vdot(gradient_latent, direction))
            step_errors = []
            for step in steps:
                plus = latent + step * direction
                minus = latent - step * direction
                if np.min(minus) <= 0.0 or np.max(plus) >= 1.0:
                    raise RuntimeError(f"{direction_name} beta={beta} FD requires clipping")
                objective_plus = float(np.vdot(cotangent, mapping.physical(plus, beta)))
                objective_minus = float(np.vdot(cotangent, mapping.physical(minus, beta)))
                fd = (objective_plus - objective_minus) / (2.0 * step)
                normalized_error = abs(ad - fd) / max(
                    float(np.linalg.norm(gradient_latent)), np.finfo(float).tiny
                )
                step_errors.append(normalized_error)
                worst_fd = max(worst_fd, normalized_error)
                rows.append(
                    {
                        "beta": beta,
                        "direction": direction_name,
                        "h": step,
                        "AD": ad,
                        "FD": fd,
                        "gradient_norm_normalized_error": normalized_error,
                        "JVP_VJP_dot_error": dot_error,
                    }
                )
            if step_errors[-1] > step_errors[0] * 1.05 + 1.0e-13:
                monotone_failures += 1

    impulse = np.zeros(latent.shape)
    impulse[0, latent.shape[1] // 2] = 1.0
    filtered_impulse = mapping.filtered(impulse)
    opposite_edge_wrap = float(np.max(np.abs(filtered_impulse[-2:, :])))
    mapping_audit = mapping.audit()
    gates = {
        "finite_nonperiodic_filter_constant_error_lt_1e-12": mapping_audit[
            "constant_preservation_max_abs"
        ] < 1.0e-12,
        "no_opposite_edge_wrap": opposite_edge_wrap == 0.0,
        "JVP_VJP_dot_error_lt_1e-12": worst_dot < 1.0e-12,
        "mapping_only_FD_normalized_error_lt_1e-6": worst_fd < 1.0e-6,
        "FD_error_no_material_h_to_h_over_2_regression": monotone_failures == 0,
        "all_FDs_unclipped": True,
    }
    passed = all(gates.values())
    status = STATUS_PASS if passed else STATUS_FAIL

    np.savez_compressed(
        raw_output,
        latent=latent.astype(np.float32),
        filtered=mapping.filtered(latent).astype(np.float32),
        physical_beta1=mapping.physical(latent, 1.0).astype(np.float32),
        physical_beta2=mapping.physical(latent, 2.0).astype(np.float32),
        physical_beta4=mapping.physical(latent, 4.0).astype(np.float32),
        physical_beta8=mapping.physical(latent, 8.0).astype(np.float32),
        cotangent=cotangent,
    )
    csv_path = output / "au_latent_filter_projection_mapping_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    images = (
        (latent, "latent baseline"),
        (mapping.filtered(latent), "finite conic filtered"),
        (mapping.physical(latent, 1.0), "projected beta=1"),
        (mapping.physical(latent, 2.0), "projected beta=2"),
        (mapping.physical(latent, 4.0), "projected beta=4"),
        (mapping.physical(latent, 8.0), "projected beta=8"),
    )
    for axis, (image, title) in zip(axes.ravel(), images):
        plotted = axis.imshow(image.T, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.set_xlabel("x=b design index")
        axis.set_ylabel("y=a design index")
        fig.colorbar(plotted, ax=axis)
    fig.suptitle(status.replace("_", " "), fontsize=11)
    plot_path = output / "au_latent_filter_projection_mapping.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    summary = {
        "status": status,
        "scope": (
            "solver-free 20x20 Au latent-to-finite-filter-to-tanh-projection "
            "JVP/VJP and mapping-only central FD; no final fabrication rule and no optimization"
        ),
        "mapping": mapping_audit,
        "filter_radius_role": (
            "750 nm is an explicit numerical mapping scenario for the current 500 nm "
            "design pixels, not a fabricated minimum-feature confidence interval"
        ),
        "beta_values": list(betas),
        "FD_steps": list(steps),
        "direction_count": len(directions),
        "worst_JVP_VJP_dot_error": worst_dot,
        "worst_mapping_FD_gradient_norm_normalized_error": worst_fd,
        "opposite_edge_wrap_max_abs": opposite_edge_wrap,
        "monotone_failure_count": monotone_failures,
        "gates": gates,
        "input_artifact": {
            "path": str(input_path),
            "bytes": input_path.stat().st_size,
            "sha256": _sha256(input_path),
        },
        "raw_artifact": {
            "path": str(raw_output),
            "bytes": raw_output.stat().st_size,
            "sha256": _sha256(raw_output),
            "committed_to_git": False,
        },
        "next_gate": (
            "choose and record the Au optimization fabrication/filter contract, then "
            "recompute the full physical gradient at its mapped baseline and perform "
            "end-to-end latent AD-FD before optimization"
        ),
    }
    summary_path = output / "au_latent_filter_projection_mapping_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = f"""# Au latent/filter/projection mapping validation

Status: **{status}**

The current 20x20 Au design layout uses 500 nm pixels. This solver-free gate
tests a finite nonperiodic 750 nm conic-filter scenario and eta=0.5 tanh
projection at beta 1, 2, 4, and 8. The filter is row-normalized at truncated
boundaries and its exact transpose is used.

| metric | value | gate |
|---|---:|---:|
| constant preservation max error | {mapping_audit['constant_preservation_max_abs']:.3e} | <1e-12 |
| opposite-edge wrap | {opposite_edge_wrap:.3e} | =0 |
| worst JVP/VJP dot error | {worst_dot:.3e} | <1e-12 |
| worst mapping-only FD / gradient-norm error | {worst_fd:.3e} | <1e-6 |
| h-to-h/2 regression count | {monotone_failures} | 0 |

The 750 nm radius is an explicit numerical scenario, not a final fabrication
minimum-feature claim. This checkpoint validates only the mapping calculus;
the full latent Maxwell/thermal/electrical AD--FD remains required before
optimization.
"""
    report_path = output / "AU_LATENT_FILTER_PROJECTION_MAPPING_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    published = [summary_path, csv_path, plot_path, report_path]
    manifest = {
        "status": status,
        "input_artifact": summary["input_artifact"],
        "raw_artifact": summary["raw_artifact"],
        "published": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in published
        ],
    }
    manifest_path = output / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
