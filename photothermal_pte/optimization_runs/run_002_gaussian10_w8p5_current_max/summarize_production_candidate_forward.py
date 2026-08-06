#!/usr/bin/env python3
"""Publish the Run-002 rho=0.5 production-candidate GPU forward gate."""

from __future__ import annotations

import argparse
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


def artifacts(directory: Path) -> list[dict[str, object]]:
    return [
        {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(directory.iterdir())
        if path.is_file()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-directory", required=True, type=Path)
    args = parser.parse_args()
    raw = args.raw_directory.expanduser().resolve()
    result = json.loads((raw / "production_candidate_forward_result.json").read_text())
    if not result.get("passed", False):
        raise RuntimeError("production-candidate forward did not pass")
    for artifact_key in ("output_project", "native_Q_artifact"):
        artifact = result[artifact_key]
        artifact_path = Path(artifact["path"]).resolve()
        if artifact_path.parent != raw:
            raise RuntimeError(f"{artifact_key} is outside the requested raw directory")
        if artifact_path.stat().st_size != artifact["size_bytes"]:
            raise RuntimeError(f"{artifact_key} byte-size mismatch")
        if sha256(artifact_path) != artifact["sha256"]:
            raise RuntimeError(f"{artifact_key} SHA-256 mismatch")
    q_path = Path(result["native_Q_artifact"]["path"])
    data = np.load(q_path)
    qxy = {}
    z_power_density = {}
    component_metrics = {}
    for component in "xyz":
        q = np.asarray(data[f"Q{component}_W_m3"], float)
        x = np.asarray(data[f"Q{component}_x_m"], float)
        y = np.asarray(data[f"Q{component}_y_m"], float)
        z = np.asarray(data[f"Q{component}_z_m"], float)
        qxy[component] = np.trapezoid(q, z, axis=2)
        z_power_density[component] = np.trapezoid(
            np.trapezoid(q, y, axis=1), x, axis=0
        )
        hotspot = np.unravel_index(int(np.argmax(q)), q.shape)
        component_metrics[component] = {
            "native_shape": list(q.shape),
            "coordinate_bounds_m": {
                "x": [float(x[0]), float(x[-1])],
                "y": [float(y[0]), float(y[-1])],
                "z": [float(z[0]), float(z[-1])],
            },
            "hotspot_m": [
                float(x[hotspot[0]]),
                float(y[hotspot[1]]),
                float(z[hotspot[2]]),
            ],
            "Q_maximum_W_m3": float(q[hotspot]),
            "all_finite": bool(np.all(np.isfinite(q))),
            "negative_count": int(np.count_nonzero(q < 0.0)),
        }
    result["component_spatial_metrics"] = component_metrics
    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS / "production_candidate_forward_summary.json"
    report_path = RESULTS / "PRODUCTION_CANDIDATE_FORWARD_REPORT.md"
    plot_path = PLOTS / "production_candidate_forward_native_q.png"
    summary_path.write_text(json.dumps(result, indent=2) + "\n")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for column, component in enumerate("xyz"):
        x = np.asarray(data[f"Q{component}_x_m"], float) * 1e6
        y = np.asarray(data[f"Q{component}_y_m"], float) * 1e6
        image = axes[0, column].pcolormesh(
            x,
            y,
            qxy[component].T,
            shading="auto",
            cmap="inferno",
        )
        fig.colorbar(image, ax=axes[0, column], label=r"$\int Q_c dz$ (W/m²)")
        axes[0, column].set(
            xlabel="x=b (µm)",
            ylabel="y=a (µm)",
            title=rf"Native $Q_{component}$ depth integral",
            aspect="equal",
        )
        z = np.asarray(data[f"Q{component}_z_m"], float) * 1e6
        axes[1, 0].plot(
            z,
            z_power_density[component] * 1e6,
            label=rf"$Q_{component}$",
        )
    axes[1, 0].axvline(-0.385, color="0.5", linestyle=":", linewidth=1)
    axes[1, 0].axvline(-0.100, color="0.5", linestyle=":", linewidth=1)
    axes[1, 0].axvline(0.0, color="0.5", linestyle=":", linewidth=1)
    axes[1, 0].axvline(1.0, color="0.5", linestyle=":", linewidth=1)
    axes[1, 0].set(
        xlabel="z (µm)",
        ylabel=r"$\iint Q_c dxdy$ (W/m)",
        title="Native depth power-density profiles",
    )
    axes[1, 0].legend()
    powers = result["Q_component_power_W"]
    total = result["P_Q_W"]
    axes[1, 1].bar(
        list("xyz"),
        [powers[component] / total * 100 for component in "xyz"],
        color=["C0", "C1", "C2"],
    )
    axes[1, 1].set(
        xlabel="component",
        ylabel="fraction of P_Q (%)",
        title="Absorbed-power decomposition",
    )
    gates = {
        "closure": result["six_face_closure_relative"] / 0.005,
        "shutoff": result["log_audit"]["final_auto_shutoff"] / 1e-5,
        "negative Q": 0.0 if result["Q_minimum_W_m3"] >= 0.0 else 1.0,
    }
    axes[1, 2].bar(
        list(gates.keys()),
        list(gates.values()),
        color=["C0", "C1", "C3"],
    )
    axes[1, 2].axhline(1.0, color="black", linestyle="--", linewidth=1, label="gate")
    axes[1, 2].set(ylabel="measured / allowed", title="Forward gate margins")
    axes[1, 2].legend()
    fig.suptitle("Run 002 rho=0.5 production-candidate GPU forward — native component Q")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    component_rows = []
    for component in "xyz":
        metric = component_metrics[component]
        component_rows.append(
            f"| {component} | {result['Q_component_power_W'][component]:.12e} | "
            f"{result['Q_component_power_W'][component]/result['P_Q_W']*100:.6f} | "
            f"{metric['hotspot_m']} | {metric['negative_count']} |"
        )
    report_path.write_text(
        f"""# Run 002 production-candidate GPU forward

Status: `{result['status']}`

This is the first actual GPU Maxwell forward for the matched-volume rho=0.5
coarse production-candidate stack.  It is not a thermal, PTE, adjoint, or
optimization result.

| metric | value |
|:--|--:|
| source power | {result['source_power_W']:.12e} W |
| P_Q | {result['P_Q_W']:.12e} W |
| P_six | {result['P_six_W']:.12e} W |
| six-face closure | {result['six_face_closure_relative']*100:.6f}% |
| final auto-shutoff | {result['log_audit']['final_auto_shutoff']:.6e} |
| solver wall time | {result['solver_wall_time_s']:.3f} s |
| GPU memory | {result['log_audit']['precise_GPU_memory_GiB']:.3f} GiB |
| logged grid points | {result['log_audit']['logged_grid']['grid_points']:,} |

| component | power (W) | fraction of P_Q (%) | native hotspot xyz (m) | negative cells |
|:--:|--:|--:|:--|--:|
{chr(10).join(component_rows)}

No Q clipping, smoothing, gain, or rescaling was used.  The three component
maps remain on their native staggered Yee coordinates; they were not summed by
array index for the plots.

## Runtime implication

One forward required about {result['solver_wall_time_s']:.1f} seconds.  A
forward+adjoint iteration on the full 20×20 µm coarse canvas will therefore be
on the order of minutes.  The reviewed gradient-L1 window-selection step is
still required before the 50 nm production optimizer is enabled.
"""
    )
    manifest = json.loads(MANIFEST.read_text())
    manifest_entry = {
        "status": result["status"],
        "raw_directory": str(raw),
        "artifacts": artifacts(raw),
    }
    superseded = raw.parent / "run002_production_candidate_rho05_forward_20260806"
    if superseded.is_dir() and superseded != raw:
        old_result_path = superseded / "production_candidate_forward_result.json"
        old_project = superseded / "production_candidate_forward.fsp"
        if old_result_path.is_file() and old_project.is_file():
            old_result = json.loads(old_result_path.read_text())
            manifest_entry["superseded_diagnostic"] = {
                "reason": (
                    "result JSON hashed the FSP before a final save; the embedded "
                    "SHA therefore does not match the resulting file"
                ),
                "raw_directory": str(superseded),
                "embedded_output_project": old_result.get("output_project"),
                "actual_output_project": {
                    "path": str(old_project),
                    "size_bytes": old_project.stat().st_size,
                    "sha256": sha256(old_project),
                },
                "promoted": False,
            }
    manifest["production_candidate_rho05_forward"] = manifest_entry
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
