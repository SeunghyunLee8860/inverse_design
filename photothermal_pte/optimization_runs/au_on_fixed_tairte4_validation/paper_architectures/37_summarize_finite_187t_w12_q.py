#!/usr/bin/env python3
"""Publish plots and provenance for the validated finite-187T Q artifact."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


HERE = Path(__file__).resolve().parent
RAW = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "paper_tairte4_finite_187T_w12_Q_11p825um_Eb"
)
OUTPUT = HERE / "results_finite_187T_w12_Q_11p825um_Eb"


def positive_norm(values: np.ndarray) -> LogNorm:
    positive = np.asarray(values)[np.asarray(values) > 0.0]
    return LogNorm(vmin=max(float(np.percentile(positive, 1.0)), float(np.max(positive)) * 1e-6), vmax=float(np.max(positive)))


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = json.loads((RAW / "FINITE_187T_W12_Q_FINAL.json").read_text())
    if result["status"] != "VALIDATED_FINITE_187T_W12_VOLUMETRIC_Q":
        raise RuntimeError("finite Q is not validated")
    with np.load(RAW / "finite_187T_w12_Q.npz") as data:
        x = np.asarray(data["common_x_m"], float)
        y = np.asarray(data["common_y_m"], float)
        z = np.asarray(data["common_z_m"], float)
        q = np.asarray(data["Q_common_W_m3"], float)
    qxy = np.trapezoid(q, z, axis=2)
    iy = int(np.argmin(np.abs(y)))
    ix = int(np.argmin(np.abs(x)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    image = axes[0, 0].pcolormesh(x * 1e6, y * 1e6, qxy.T, shading="auto", cmap="inferno", norm=positive_norm(qxy))
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_title("Depth-integrated all-material Q")
    axes[0, 0].set_xlabel("x=b (um)")
    axes[0, 0].set_ylabel("y=a (um)")
    fig.colorbar(image, ax=axes[0, 0], label="W/m2")

    xz = q[:, iy, :]
    image = axes[0, 1].pcolormesh(x * 1e6, z * 1e6, xz.T, shading="auto", cmap="inferno", norm=positive_norm(xz))
    axes[0, 1].set_title(f"Q x-z section; y={y[iy]*1e6:.3f} um")
    axes[0, 1].set_xlabel("x=b (um)")
    axes[0, 1].set_ylabel("z (um)")
    fig.colorbar(image, ax=axes[0, 1], label="W/m3")

    yz = q[ix, :, :]
    image = axes[1, 0].pcolormesh(y * 1e6, z * 1e6, yz.T, shading="auto", cmap="inferno", norm=positive_norm(yz))
    axes[1, 0].set_title(f"Q y-z section; x={x[ix]*1e6:.3f} um")
    axes[1, 0].set_xlabel("y=a (um)")
    axes[1, 0].set_ylabel("z (um)")
    fig.colorbar(image, ax=axes[1, 0], label="W/m3")

    powers = result["Q_component_power_native_W"]
    axes[1, 1].bar(["Qx", "Qy", "Qz"], [powers[k] * 1e15 for k in "xyz"], color=["#377eb8", "#ff7f00", "#4daf4a"])
    axes[1, 1].set_ylabel("absorbed power (fW)")
    axes[1, 1].set_title("Authoritative native-Yee component powers")
    axes[1, 1].grid(axis="y", alpha=0.25)
    fig.suptitle("Finite 11x17 inverse-T array; w0=12 um Gaussian; lambda=11.825 um; E||b", fontsize=15)
    fig.savefig(OUTPUT / "finite_187T_w12_Q_overview.png", dpi=220)
    plt.close(fig)

    face_rows = []
    for face, values in result["six_face"]["faces"].items():
        face_rows.append({"face": face, **values})
    with (OUTPUT / "finite_187T_w12_Q_cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(face_rows[0]))
        writer.writeheader()
        writer.writerows(face_rows)

    summary = {
        "status": result["status"],
        "source": result["source"],
        "array": result["array"],
        "domain": result["domain"],
        "P_Q_native_W": result["P_Q_native_W"],
        "P_Q_pabs_W": result["P_Q_pabs_W"],
        "P_six_face_W": result["P_six_face_W"],
        "six_face_closure_relative": result["six_face_closure_relative"],
        "Q_component_power_native_W": result["Q_component_power_native_W"],
        "hotspot": result["hotspot"],
        "gates": result["gates"],
        "scope_exclusions": result["scope_exclusions"],
        "raw_artifacts_committed_to_git": False,
    }
    (OUTPUT / "finite_187T_w12_Q_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "raw_artifacts_committed_to_git": False,
        "raw_artifacts": result["raw_artifacts"],
        "generation": {
            "forward_code": "33_run_v261_finite_multi_t_gaussian_q.py",
            "read_only_extraction_code": "36_extract_completed_finite_187t_q.py",
            "reservation_launcher": "/home/dhkim/bin/runres",
            "CPU_FDTD_fallback": False,
        },
        "diagnostic_failures_preserved": [
            "/home/seunghyun/tairte4/raw_artifacts/paper_tairte4_finite_187T_w12_Q_11p825um_Eb_blocked_common_import_20260822T1000Z",
            "/home/seunghyun/tairte4/raw_artifacts/paper_tairte4_finite_187T_w12_Q_11p825um_Eb_blocked_flux_dcard_20260822T1003Z",
        ],
    }
    (OUTPUT / "RAW_ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUTPUT / "FINITE_187T_W12_Q_REPORT.md").write_text(
        "# Finite 187-inverse-T Gaussian volumetric-Q certificate\n\n"
        "Status: `VALIDATED_FINITE_187T_W12_VOLUMETRIC_Q`\n\n"
        "- TaIrTe4 is the active anisotropic material (`x=b`, `y=a`, `z=c=b` closure).\n"
        "- 11 x 17 finite inverse-T elements; no periodic/Bloch boundary.\n"
        "- scalar Gaussian, physical target `w0=12 um`, `lambda=11.825 um`, `E||b`.\n"
        "- 60 x 60 um lateral FDTD domain; six PML, 24 layers.\n"
        f"- P_Q(native Yee) = `{result['P_Q_native_W']:.12e} W`.\n"
        f"- P_six = `{result['P_six_face_W']:.12e} W`.\n"
        f"- six-face closure = `{100*result['six_face_closure_relative']:.6f}%`.\n"
        f"- Qx/Qy/Qz = `{powers['x']:.12e}`, `{powers['y']:.12e}`, `{powers['z']:.12e} W`.\n"
        f"- hotspot = `({result['hotspot']['x_m']*1e6:.3f}, {result['hotspot']['y_m']*1e6:.3f}, {result['hotspot']['z_m']*1e6:.3f}) um`.\n\n"
        "No Q clipping, smoothing, gain, rescaling, or post-solve tiling was used. "
        "The common-grid total-Q image is a collocated visualization; component-wise native Yee integration is the power authority. "
        "Thermal, weighting potential, PTE, adjoint, and optimization were not run in this certificate.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
