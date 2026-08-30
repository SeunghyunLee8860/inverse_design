#!/usr/bin/env python3
"""Publish the selected 11.825-um inverse-T volumetric-Q certificate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_RAW = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "paper_tairte4_T_selected_Q_11p825um_Eb"
)


def load_smoke_summary_module():
    path = HERE / "08_summarize_t2024_tairte4_optical_smoke.py"
    spec = importlib.util.spec_from_file_location("t2024_smoke_summary_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw_dir = args.raw_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = raw_dir / "T2024_TaIrTe4_optical_smoke.json"
    npz_path = raw_dir / "T2024_TaIrTe4_native_q.npz"
    fsp_path = raw_dir / "T2024_TaIrTe4_optical_smoke.fsp"
    result = json.loads(result_path.read_text())
    if result.get("status") != "COMPLETED_T2024_TAIRTE4_OPTICAL_SMOKE":
        raise RuntimeError(f"selected Q raw result is not complete: {result.get('status')}")
    if not all(result.get("gates", {}).values()):
        raise RuntimeError("selected Q raw result did not pass every gate")
    wavelength_um = float(result["contract"]["source"]["wavelength_m"]) * 1.0e6
    polarization = str(result["contract"]["source"]["polarization"])
    if abs(wavelength_um - 11.825) > 1.0e-9 or polarization != "x_b":
        raise RuntimeError("raw artifact is not the approved 11.825-um E||b case")

    helpers = load_smoke_summary_module()
    with np.load(npz_path) as raw:
        partition, areal = helpers.material_partition(raw, result)
        figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
        vertices = np.asarray(
            result["contract"]["geometry"]["polygons"][0]["vertices_nm"], float
        )
        axes[0, 0].fill(vertices[:, 0], vertices[:, 1], color="#f6c64e", edgecolor="#7a5300")
        axes[0, 0].set_aspect("equal")
        axes[0, 0].set_xlim(-750, 750)
        axes[0, 0].set_ylim(-500, 500)
        axes[0, 0].set_title("periodic inverse-T unit cell")
        axes[0, 0].set_xlabel("x=b (nm)")
        axes[0, 0].set_ylabel("y=a (nm)")
        maxima = max(float(np.max(areal[c]["Q_W_m2"])) for c in "xyz")
        for axis, component in zip((axes[0, 1], axes[0, 2], axes[1, 0]), "xyz"):
            item = areal[component]
            image = axis.pcolormesh(
                item["x"] * 1.0e9,
                item["y"] * 1.0e9,
                item["Q_W_m2"].T,
                shading="auto",
                cmap="inferno",
                vmin=0.0,
                vmax=maxima,
            )
            axis.set_aspect("equal")
            axis.set_title(f"depth-integrated $Q_{component}$")
            axis.set_xlabel("x=b (nm)")
            axis.set_ylabel("y=a (nm)")
            figure.colorbar(image, ax=axis, label="W/m$^2$")
        powers = result["Q_component_power_native_W"]
        axes[1, 1].bar(list("xyz"), [float(powers[c]) * 1.0e18 for c in "xyz"], color=["#2c7fb8", "#f28e2b", "#59a14f"])
        axes[1, 1].set_ylabel("native Yee component power (aW/cell)")
        axes[1, 1].set_title("$Q_x,Q_y,Q_z$ power")
        axes[1, 2].axis("off")
        axes[1, 2].text(
            0.0,
            1.0,
            "\n".join(
                [
                    f"TaIrTe4, E||b, wavelength = {wavelength_um:.3f} um",
                    f"P_Q = {result['P_Q_pabs_periodic_W']:.9e} W/cell",
                    f"P_flux = {result['P_flux_absorbed_W']:.9e} W/cell",
                    f"closure = {100*result['closure_relative']:.5f}%",
                    f"auto-shutoff = {result['log_audit']['final_auto_shutoff']:.3e}",
                    f"GPU wall time = {result['solver_wall_time_s']:.2f} s",
                    "No clipping/smoothing/gain/rescaling",
                    "No thermal/PTE/adjoint/optimization",
                ]
            ),
            va="top",
            family="monospace",
        )
        plot_path = output / "T_SELECTED_Q_11p825um_Eb.png"
        figure.suptitle("Selected periodic inverse-T volumetric-Q certificate")
        figure.savefig(plot_path, dpi=220)
        plt.close(figure)

    summary = {
        "status": "VALIDATED_T_SELECTED_PERIODIC_VOLUMETRIC_Q",
        "scope": "periodic unit-cell optical Q certificate; not finite Gaussian or PTE",
        "wavelength_um": wavelength_um,
        "polarization": polarization,
        "axis_mapping": "Lumerical x=b, y=a, z=c=b closure",
        "metrics": {
            key: result[key]
            for key in (
                "solver_version",
                "solver_wall_time_s",
                "source_power_W",
                "P_flux_absorbed_W",
                "P_Q_pabs_periodic_W",
                "P_Q_native_uncorrected_W",
                "Q_component_power_native_W",
                "closure_relative",
                "reflection",
                "log_audit",
                "gates",
            )
        },
        "geometric_material_partition_native_W": partition,
        "rules": {
            "no_Q_clipping_smoothing_gain_rescaling": True,
            "raw_FSP_NPZ_committed": False,
            "thermal_PTE_adjoint_optimization_run": False,
        },
        "next_gate": "finite multi-T Gaussian source-only/runsetup audit before finite Q",
    }
    summary_path = output / "T_SELECTED_Q_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    report_path = output / "T_SELECTED_Q_REPORT.md"
    report_path.write_text(
        f"""# Selected inverse-T periodic volumetric Q

Status: `VALIDATED_T_SELECTED_PERIODIC_VOLUMETRIC_Q`

This is a TaIrTe4 periodic unit-cell optical certificate at `{wavelength_um:.3f} um`
for `E||b` (`Lumerical x=b`). It was selected from the 4-12 um T-minus-bare
screen. It is not a finite Gaussian-beam, thermal, PTE, adjoint, or optimization
result.

| metric | value |
|---|---:|
| source power per cell | {result['source_power_W']:.12e} W |
| component-resolved P_Q | {result['P_Q_pabs_periodic_W']:.12e} W |
| absorbed flux | {result['P_flux_absorbed_W']:.12e} W |
| closure | {100*result['closure_relative']:.6f}% |
| auto-shutoff | {result['log_audit']['final_auto_shutoff']:.6e} |
| GPU wall time | {result['solver_wall_time_s']:.3f} s |

All gates passed. Qx/Qy/Qz were evaluated on their own physical Yee component
coordinates. There are no negative, NaN, or Inf Q cells. No Q clipping,
smoothing, gain, global rescaling, or polarization matching was used.
"""
    )
    manifest = {
        "raw_artifacts_committed_to_git": False,
        "generation_command": (
            "runres --reserve-count 9 --reserve-tag tairte4_T_selected_Q_11p825_Eb "
            "25_runres_t_selected_q_driver.py -th 8 -GPU 5"
        ),
        "raw_artifacts": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (fsp_path, npz_path, result_path)
        ],
        "published_artifacts": [],
    }
    manifest_path = output / "RAW_ARTIFACT_MANIFEST.json"
    published = (plot_path, summary_path, report_path)
    manifest["published_artifacts"] = [
        {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in published
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
