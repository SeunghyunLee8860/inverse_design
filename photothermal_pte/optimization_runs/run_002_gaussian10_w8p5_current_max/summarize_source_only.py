#!/usr/bin/env python3
"""Promote one raw Run-002 source-only result into tracked evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_entry(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--diagnostic-dir", action="append", default=[])
    args = parser.parse_args()
    raw = Path(args.raw_dir).expanduser().resolve()
    result_path = raw / "source_only_case_result.json"
    fields_path = raw / "paper_ir_source_only_fields.npz"
    payload = json.loads(result_path.read_text())
    if payload["status"] != "VALIDATED_EXPLICIT_WAIST_SCALAR_GAUSSIAN_SOURCE_ONLY":
        raise RuntimeError(f"source-only gate did not pass: {payload['status']}")
    if not payload["source_only_gate_passed"] or not all(
        payload["acceptance"].values()
    ):
        raise RuntimeError("source-only acceptance is not complete")

    focus = payload["planes"]["flake_target_plane"]
    profile = payload["source_object_profile"]
    summary = {
        "status": "VALIDATED_GAUSSIAN10_W8P5_SOURCE_ONLY",
        "scope": "homogeneous-air GPU source-only; no material, Q, thermal, PTE, adjoint, or optimization",
        "wavelength_um": 10.0,
        "target_realized_waist_um": 8.5,
        "source_object_waist_um": 8.36043075475035,
        "target_plane": {
            "fitted_waist_x_um": 1.0e6 * focus["fitted_waist_x_m"],
            "fitted_waist_y_um": 1.0e6 * focus["fitted_waist_y_m"],
            "fitted_effective_waist_um": 1.0e6
            * focus["fitted_waist_effective_m"],
            "Gaussian_fit_NRMSE": focus["Gaussian_fit_NRMSE"],
            "ellipticity": focus["fitted_xy_ellipticity"],
            "beam_center_error_um": 1.0e6 * focus["beam_center_error_m"],
            "downward_power_over_sourcepower": focus[
                "downward_Poynting_power_over_sourcepower"
            ],
            "incident_power_closure_relative": abs(
                focus["downward_Poynting_power_over_sourcepower"] - 1.0
            ),
            "boundary_max_intensity_over_peak": focus[
                "boundary_max_intensity_over_peak"
            ],
            "longitudinal_Ez_E2_fraction": focus[
                "longitudinal_Ez_E2_fraction"
            ],
        },
        "source_object_profile": {
            "fitted_effective_waist_um": 1.0e6
            * profile["fitted_waist_effective_m"],
            "square_captured_fraction": profile[
                "fitted_infinite_Gaussian_square_captured_fraction"
            ],
            "boundary_max_intensity_over_peak": profile[
                "boundary_max_intensity_over_peak"
            ],
        },
        "solver": {
            "version": payload["pre_run"]["version"],
            "GPU_resource_used": payload["GPU_resource_used"],
            "wall_time_s": payload["solver_wall_time_s"],
            "final_auto_shutoff": payload["log_audit"]["final_auto_shutoff"],
            "precise_GPU_memory_GiB": payload["log_audit"][
                "precise_GPU_memory_GiB"
            ],
            "logged_grid": payload["log_audit"]["logged_grid"],
            "CPU_FDTD_fallback": False,
        },
        "acceptance": payload["acceptance"],
        "raw_field_artifact": raw_entry(fields_path),
    }
    result_dir = HERE / "results"
    plot_dir = HERE / "plots"
    result_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    (result_dir / "source_only_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    with np.load(fields_path) as arrays:
        x = arrays["flake_target_plane_x_m"] * 1.0e6
        y = arrays["flake_target_plane_y_m"] * 1.0e6
        intensity = arrays["flake_target_plane_downward_intensity_W_m2"]
    normalized = intensity / np.max(intensity)
    ix = int(np.argmin(np.abs(x)))
    iy = int(np.argmin(np.abs(y)))
    ideal_x = np.exp(-2.0 * (x / 8.5) ** 2)
    ideal_y = np.exp(-2.0 * (y / 8.5) ** 2)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    image = axes[0].pcolormesh(x, y, normalized.T, shading="auto", cmap="magma")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x (µm)")
    axes[0].set_ylabel("y (µm)")
    axes[0].set_title("Target-plane normalized downward intensity")
    fig.colorbar(image, ax=axes[0], label=r"$I/I_{\max}$")
    axes[1].plot(x, normalized[:, iy], label="realized x cut")
    axes[1].plot(y, normalized[ix, :], label="realized y cut")
    axes[1].plot(x, ideal_x, "k--", label=r"ideal $w_0=8.5\,\mu$m")
    axes[1].plot(y, ideal_y, "k--", alpha=0.0)
    axes[1].set_xlim(-20.0, 20.0)
    axes[1].set_ylim(-0.02, 1.04)
    axes[1].set_xlabel("lateral coordinate (µm)")
    axes[1].set_ylabel(r"$I/I_{\max}$")
    axes[1].set_title("Center linecuts")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle(
        "Run 002 source-only GPU validation — λ=10 µm, target w₀=8.5 µm"
    )
    figure_path = plot_dir / "source_only_target_plane.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    report = f"""# Run 002 Gaussian-source GPU report

Status: `VALIDATED_GAUSSIAN10_W8P5_SOURCE_ONLY`

This is a homogeneous-air **source-only** certificate. It contains no
TaIrTe4, SiO2, optical absorption Q, thermal solve, PTE current, adjoint, or
optimization result.

## Fixed contract and readback

- analysis wavelength: 10 µm;
- scalar Gaussian, waist-size-and-position definition;
- target-plane requested waist: 8.5 µm;
- one-step calibrated source-object waist: 8.36043075475035 µm;
- source span: 40×40 µm²;
- FDTD lateral span: 48×48 µm²;
- z bounds: -8 to +8 µm; source/focus z: +5/0 µm;
- all six boundaries PML, 24 layers; periodic/Bloch disabled;
- conformal variant 1, mesh accuracy 3;
- GPU FDTD only; no CPU fallback.

## Measured source gate

| metric | result | gate |
|---|---:|---:|
| fitted waist x | {summary['target_plane']['fitted_waist_x_um']:.6f} µm | within 0.5% of 8.5 µm |
| fitted waist y | {summary['target_plane']['fitted_waist_y_um']:.6f} µm | within 0.5% of 8.5 µm |
| effective fitted waist | {summary['target_plane']['fitted_effective_waist_um']:.6f} µm | diagnostic |
| Gaussian-fit NRMSE | {100*summary['target_plane']['Gaussian_fit_NRMSE']:.6f}% | <0.5% |
| ellipticity | {100*summary['target_plane']['ellipticity']:.6f}% | <0.5% |
| incident-power closure | {100*summary['target_plane']['incident_power_closure_relative']:.6f}% | <0.5% |
| target boundary max/peak | {summary['target_plane']['boundary_max_intensity_over_peak']:.6e} | <1e-3 |
| source-square captured fraction | {summary['source_object_profile']['square_captured_fraction']:.9f} | >=0.999 |
| auto shutoff | {summary['solver']['final_auto_shutoff']:.6e} | <=1e-5 |
| solver wall time | {summary['solver']['wall_time_s']:.3f} s | diagnostic |
| GPU memory | {summary['solver']['precise_GPU_memory_GiB']:.3f} GiB | diagnostic |

All acceptance booleans are true. The uncalibrated 8.5 µm source-object run,
which realized 8.64190 µm and failed only the waist gate, remains preserved as
a diagnostic raw artifact; it was not relabeled as passing.

## Consequence

The source contract is now authorized for the next **single forward material
smoke test**. It does not authorize optimization. The remaining gates are the
complex 10 µm SiO2/TaIrTe4 material readback and closure, material-resolved Q
remap, coarse physical-gradient design-window selection, Gaussian combined
AD–FD smoke test, and production-scale CUDA thermal parity.
"""
    (result_dir / "SOURCE_ONLY_GPU_REPORT.md").write_text(report)

    manifest = {
        "status": summary["status"],
        "raw_artifacts_committed_to_git": False,
        "generation_command": payload["generation_command"],
        "generation_commit": payload["generation_commit"],
        "source_only_raw_artifacts": [
            raw_entry(path)
            for path in sorted(raw.iterdir())
            if path.is_file()
        ],
        "diagnostic_raw_artifacts": [
            {
                "directory": str(directory),
                "artifacts": [
                    raw_entry(path)
                    for path in sorted(directory.iterdir())
                    if path.is_file()
                ],
            }
            for directory in (
                Path(value).expanduser().resolve()
                for value in args.diagnostic_dir
            )
        ],
    }
    (HERE / "manifests" / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
