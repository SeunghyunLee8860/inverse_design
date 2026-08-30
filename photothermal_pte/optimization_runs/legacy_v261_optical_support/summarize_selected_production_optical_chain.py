#!/usr/bin/env python3
"""Publish selected-window runsetup, GPU forward, and component-J checkpoints."""

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


def record(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-directory", required=True, type=Path)
    parser.add_argument("--forward-directory", required=True, type=Path)
    parser.add_argument("--jacobian-directory", required=True, type=Path)
    parser.add_argument("--failed-geometry-directory", required=True, type=Path)
    args = parser.parse_args()
    geometry_path = args.geometry_directory.resolve() / "production_candidate_geometry_audit.json"
    forward_path = args.forward_directory.resolve() / "production_candidate_forward_result.json"
    jacobian_path = args.jacobian_directory.resolve() / "component_yee_jacobian_result.json"
    failed_path = args.failed_geometry_directory.resolve() / "production_candidate_geometry_audit.json"
    geometry = json.loads(geometry_path.read_text())
    forward = json.loads(forward_path.read_text())
    jacobian = json.loads(jacobian_path.read_text())
    failed = json.loads(failed_path.read_text())
    if not geometry.get("passed") or geometry.get("design_contract") != "selected_18p6um_373":
        raise RuntimeError("selected geometry audit is not validated")
    if not forward.get("passed"):
        raise RuntimeError("selected GPU forward is not validated")
    if jacobian.get("status") != "VALIDATED_SELECTED_PRODUCTION_COMPLEX_COMPONENT_YEE_JACOBIAN" or not jacobian.get("passed"):
        raise RuntimeError("selected component-Yee Jacobian is not validated")
    if "KeyError" not in failed.get("error", ""):
        raise RuntimeError("expected first fail-closed geometry diagnostic is absent")
    if forward["input_project"]["sha256"] != geometry["project"]["sha256"]:
        raise RuntimeError("GPU forward did not use the validated geometry FSP")
    if jacobian["base_FSP"]["sha256"] != forward["output_project"]["sha256"]:
        raise RuntimeError("component Jacobian did not use the completed forward FSP")

    summary = {
        "status": "VALIDATED_SELECTED_PRODUCTION_OPTICAL_CHAIN",
        "passed": True,
        "scope": "selected 18.6 um / 373x373 runsetup, rho=0.5 GPU forward, and layout-only component-Yee Jacobian",
        "geometry": geometry,
        "forward": forward,
        "jacobian": jacobian,
        "preserved_failed_geometry_diagnostic": record(failed_path),
        "thermal_solve": False,
        "adjoint_Maxwell_solve": False,
        "optimization_iterations": 0,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS / "selected_production_optical_chain_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    report_path = RESULTS / "SELECTED_PRODUCTION_OPTICAL_CHAIN_REPORT.md"
    report_path.write_text(
        f"""# Selected production optical chain

Status: `VALIDATED_SELECTED_PRODUCTION_OPTICAL_CHAIN`

This checkpoint replaces the coarse 20×20 µm / 201×201 optical-layout
assumption for future production work. It uses the frozen centered
18.6×18.6 µm window with 373×373 nodes at 50 nm and the unchanged 10 µm,
w0=8.5 µm scalar-Gaussian, six-PML contract.

## Runsetup and forward

- Realized global mesh: `{geometry['mesh_readback']['shape_xyz']}`
  ({geometry['mesh_readback']['grid_points']:,} grid points).
- Minimum dx/dy/dz: `{geometry['mesh_readback']['minimum_step_m']['x']:.6e}` /
  `{geometry['mesh_readback']['minimum_step_m']['y']:.6e}` /
  `{geometry['mesh_readback']['minimum_step_m']['z']:.6e} m`.
- GPU solver wall time: `{forward['solver_wall_time_s']:.3f} s`.
- P_Q: `{forward['P_Q_W']:.12e} W`; P_six: `{forward['P_six_W']:.12e} W`.
- Six-face closure: `{forward['six_face_closure_relative']:.6e}` (<0.5%).
- Final auto-shutoff: `{forward['log_audit']['final_auto_shutoff']:.6e}` (<1e-5).
- No Q clipping, smoothing, gain, or rescaling.

## Component-specific material Jacobian

- Density shape: `{jacobian['density_shape']}`.
- Worst mapping-only FD error: `{jacobian['gates']['worst_mapping_only_FD_relative_error']:.6e}` (<1e-7).
- Worst JVP/VJP dot error: `{jacobian['gates']['worst_JVP_VJP_dot_relative_error']:.6e}` (<1e-12).
- Maximum forward/index coordinate mismatch:
  `{jacobian['maximum_coordinate_mismatch_m']:.6e} m` (<2e-18 m).
- Active J rows outside the exact selected support: zero for x, y, and z.
- Maxwell solves used to build J: zero; per-pixel Maxwell solves: false.

The first geometry attempt remains a failed diagnostic: the final audit used
the old coarse object name and raised a KeyError. No field solve occurred in
that failed attempt. The corrected run used a new directory.

This checkpoint does not certify thermal gray laws, the Maxwell adjoint, full
latent AD-FD, exact-binary DRC, or optimization.
"""
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    mesh = geometry["mesh_readback"]
    axes[0].bar(["dx", "dy", "dz"], [mesh["minimum_step_m"][key] * 1e9 for key in "xyz"])
    axes[0].set(title="Selected runsetup mesh", ylabel="minimum step (nm)")
    components = list("xyz")
    axes[1].bar(components, [forward["Q_component_power_W"][key] * 1e15 for key in components])
    axes[1].set(title="Native absorbed-power components", ylabel="power (fW)")
    names = list(jacobian["directions"])
    fd = [jacobian["directions"][name]["mapping_only_FD_relative_error"] for name in names]
    dot = [jacobian["directions"][name]["JVP_VJP_dot_relative_error"] for name in names]
    x = np.arange(len(names))
    axes[2].semilogy(x, fd, "o-", label="mapping FD")
    axes[2].semilogy(x, dot, "s-", label="JVP/VJP")
    axes[2].axhline(1e-7, color="tab:blue", ls="--")
    axes[2].axhline(1e-12, color="tab:orange", ls="--")
    axes[2].set_xticks(x, [name.replace("_", "\n") for name in names], fontsize=7)
    axes[2].set(title="Component-J validation", ylabel="relative error")
    axes[2].legend()
    fig.suptitle("Run 002 selected 18.6 µm production optical chain")
    plot_path = PLOTS / "selected_production_optical_chain.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    manifest = json.loads(MANIFEST.read_text())
    # Preserve the historical top-level source-only provenance fields while
    # publishing an explicit promoted status for the cumulative run.
    manifest["current_promoted_status"] = summary["status"]
    manifest["current_promoted_at_utc"] = jacobian["generated_at_utc"]
    manifest["selected_production_optical_chain"] = {
        "status": summary["status"],
        "raw_artifacts_committed_to_git": False,
        "geometry": {"result": record(geometry_path), "FSP": record(Path(geometry["project"]["path"]))},
        "forward": {
            "result": record(forward_path),
            "FSP": record(Path(forward["output_project"]["path"])),
            "native_Q": record(Path(forward["native_Q_artifact"]["path"])),
        },
        "component_J": {
            "result": record(jacobian_path),
            "coordinates": record(Path(jacobian["artifacts"]["coordinates_and_density"]["path"])),
            **{key: record(Path(value["path"])) for key, value in jacobian["artifacts"]["component_J"].items()},
        },
        "preserved_failed_geometry_diagnostic": record(failed_path),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "report": str(report_path), "summary": str(summary_path), "plot": str(plot_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
