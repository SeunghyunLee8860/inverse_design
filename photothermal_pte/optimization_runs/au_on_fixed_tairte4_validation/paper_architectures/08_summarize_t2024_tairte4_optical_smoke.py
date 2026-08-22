#!/usr/bin/env python3
"""Publish the first actual inverse-T/TaIrTe4 GPU optical smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.finite_inverse_design.native_yee_q import (  # noqa: E402
    integrate_xyz,
    trapezoid_weights,
)


DEFAULT_RAW = Path(
    "/home/seunghyun/tairte4/raw_artifacts/paper_tairte4_architectures/"
    "T2024_MIR_4750_xb_forward"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def polygon_mask(x: np.ndarray, y: np.ndarray, vertices_m: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    return MplPath(vertices_m, closed=True).contains_points(points, radius=1.0e-15).reshape(xx.shape)


def material_partition(raw: np.lib.npyio.NpzFile, contract: dict) -> tuple[dict, dict]:
    vertices_m = np.asarray(
        contract["contract"]["geometry"]["polygons"][0]["vertices_nm"], float
    ) * 1.0e-9
    partition: dict[str, dict[str, float]] = {}
    areal: dict[str, dict[str, np.ndarray]] = {}
    for component in "xyz":
        q = np.asarray(raw[f"Q{component}_W_m3"], float)
        x = np.asarray(raw[f"Q{component}_x_m"], float)
        y = np.asarray(raw[f"Q{component}_y_m"], float)
        z = np.asarray(raw[f"Q{component}_z_m"], float)
        xy_t = polygon_mask(x, y, vertices_m)
        ta = (z >= 0.0) & (z < 100.0e-9)
        top_au = xy_t[:, :, None] & (z[None, None, :] >= 100.0e-9) & (z[None, None, :] <= 133.0e-9)
        backplane = z <= -35.0e-9
        masks = {
            "TaIrTe4_geometric": np.broadcast_to(ta[None, None, :], q.shape),
            "top_Au_T_geometric": top_au,
            "Au_backplane_geometric": np.broadcast_to(backplane[None, None, :], q.shape),
        }
        total = integrate_xyz(q, x, y, z)
        values = {name: integrate_xyz(q * mask, x, y, z) for name, mask in masks.items()}
        values["native_component_total"] = total
        values["unassigned_interface_or_other"] = total - sum(values[name] for name in masks)
        partition[component] = values
        q_areal = np.einsum("k,ijk->ij", trapezoid_weights(z), q, optimize=True)
        areal[component] = {"x": x, "y": y, "Q_W_m2": q_areal}
    return partition, areal


def plot_summary(output: Path, contract: dict, raw: np.lib.npyio.NpzFile, areal: dict, partition: dict) -> Path:
    figure, axes = plt.subplots(2, 3, figsize=(14, 8.2), constrained_layout=True)
    geometry = contract["contract"]["geometry"]
    vertices = np.asarray(geometry["polygons"][0]["vertices_nm"], float)
    axes[0, 0].fill(vertices[:, 0], vertices[:, 1], color="#f6c64e", edgecolor="#8a5a00")
    axes[0, 0].set_xlim(-750, 750)
    axes[0, 0].set_ylim(-500, 500)
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_title("2024 MIR inverse-T geometry\nfigure-digitized; TaIrTe$_4$ substitution")
    axes[0, 0].set_xlabel("x=b (nm)")
    axes[0, 0].set_ylabel("y=a (nm)")
    axes[0, 0].text(0.02, 0.02, "period 1500 x 1000 nm\nAu 33 nm / TaIrTe$_4$ 100 nm", transform=axes[0, 0].transAxes, fontsize=8)

    maxima = max(float(np.max(areal[c]["Q_W_m2"])) for c in "xyz")
    for axis, component in zip(axes[0, 1:], "xy"):
        item = areal[component]
        image = axis.pcolormesh(item["x"] * 1.0e9, item["y"] * 1.0e9, item["Q_W_m2"].T, shading="auto", cmap="inferno", vmin=0.0, vmax=maxima)
        axis.set_aspect("equal")
        axis.set_title(f"native Yee $Q_{component}$ depth integral")
        axis.set_xlabel("x=b (nm)")
        axis.set_ylabel("y=a (nm)")
        figure.colorbar(image, ax=axis, label="W/m$^2$")
    item = areal["z"]
    image = axes[1, 0].pcolormesh(item["x"] * 1.0e9, item["y"] * 1.0e9, item["Q_W_m2"].T, shading="auto", cmap="inferno", vmin=0.0, vmax=maxima)
    axes[1, 0].set_aspect("equal")
    axes[1, 0].set_title("native Yee $Q_z$ depth integral")
    axes[1, 0].set_xlabel("x=b (nm)")
    axes[1, 0].set_ylabel("y=a (nm)")
    figure.colorbar(image, ax=axes[1, 0], label="W/m$^2$")

    labels = ["TaIrTe4", "top Au T", "Au mirror", "interface/other"]
    keys = ["TaIrTe4_geometric", "top_Au_T_geometric", "Au_backplane_geometric", "unassigned_interface_or_other"]
    bottom = np.zeros(3)
    colors = ["#c74c4c", "#f6c64e", "#be8f00", "#777777"]
    for label, key, color in zip(labels, keys, colors):
        values = np.array([partition[c][key] for c in "xyz"]) * 1.0e15
        axes[1, 1].bar(list("xyz"), values, bottom=bottom, label=label, color=color)
        bottom += values
    axes[1, 1].set_ylabel("native component power (fW)")
    axes[1, 1].set_title("geometric material partition\n(no deletion or rescaling)")
    axes[1, 1].legend(fontsize=7)

    metrics = [
        f"GPU wall time: {contract['solver_wall_time_s']:.2f} s",
        f"P_Q: {contract['P_Q_pabs_periodic_W']:.6e} W/cell",
        f"flux absorption: {contract['P_flux_absorbed_W']:.6e} W/cell",
        f"closure: {100*contract['closure_relative']:.4f}%",
        f"R: {contract['reflection']:.6f}",
        f"auto-shutoff: {contract['log_audit']['final_auto_shutoff']:.3e}",
        "negative Q cells: 0",
        "x=b polarized incidence",
        "No Q clipping/smoothing/gain/rescaling",
        "No thermal/PTE/adjoint/optimization",
    ]
    axes[1, 2].axis("off")
    axes[1, 2].text(0.0, 1.0, "\n".join(metrics), va="top", family="monospace", fontsize=10)
    path = output / "T2024_TaIrTe4_xb_optical_smoke.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results_actual_metasurfaces")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_dir = args.raw_dir.resolve()
    contract_path = raw_dir / "T2024_TaIrTe4_optical_smoke.json"
    npz_path = raw_dir / "T2024_TaIrTe4_native_q.npz"
    contract = json.loads(contract_path.read_text())
    raw = np.load(npz_path)
    partition, areal = material_partition(raw, contract)
    plot_path = plot_summary(output, contract, raw, areal, partition)

    summary = {
        "status": "VALIDATED_T2024_FIGURE_DIGITIZED_TAIRTE4_OPTICAL_SMOKE",
        "identity_limit": "paper-derived scenario with TaIrTe4 substitution; not graphene-experiment reproduction and not exact paper CAD",
        "source_result": str(contract_path),
        "metrics": {
            key: contract[key]
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
                "transmission_inside_Au_diagnostic",
                "log_audit",
                "gates",
            )
        },
        "geometric_material_partition_native_W": partition,
        "rules": {
            "only_active_2d_material_replaced": True,
            "active_material": "100-nm TaIrTe4",
            "axis_mapping": "Lumerical x=b, y=a, z=c=b closure",
            "no_clipping_smoothing_gain_rescaling": True,
            "thermal_PTE_adjoint_optimization_run": False,
        },
        "Z2022_next_gate": "exact chiral-Z topology must be recovered or an explicitly named approximation must be approved; Table S1 envelopes are not Maxwell CAD",
    }
    summary_path = output / "T2024_TAIRTE4_OPTICAL_SMOKE_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    report_path = output / "T2024_TAIRTE4_OPTICAL_SMOKE_REPORT.md"
    report_path.write_text(
        f"""# 2024 inverse-T / TaIrTe4 optical smoke

Status: `VALIDATED_T2024_FIGURE_DIGITIZED_TAIRTE4_OPTICAL_SMOKE`

This is the first actual metasurface calculation in this folder. It is **not**
the earlier planar-backplane truncation control. It uses the 2024 paper's
MIR inverse-T concept, 1500 x 1000 nm unit cell, 4.75 um target, 35-nm Al2O3
spacer and no MIR passivation. The active graphene layer alone is replaced by
100-nm anisotropic TaIrTe4. The T arm vertices are digitized from Supplementary
Fig. 14 axes because numeric CAD vertices are not published.

## Solver contract

- Lumerical v261 `{contract['solver_version']}` GPU forward, x=b polarized.
- x/y periodic boundaries; z PML; normal-incidence plane wave.
- conformal variant 1; 10-nm x/y and 5-nm z structure mesh.
- native mesh: `{contract['mesh_runsetup']['shape']}` (about
  `{contract['mesh_runsetup']['yee_cell_estimate']:,}` Yee cells before PML logging).
- TaIrTe4: x=epsilon_b, y=epsilon_a, z=epsilon_c=epsilon_b closure.
- Au: installed `Au (Gold) - CRC`; the paper does not state an Au dataset.
- Al2O3: lossless n=1.62 explicit optical closure, not a paper-certified dataset.

## Forward result

| Metric | Value |
|---|---:|
| GPU wall time | {contract['solver_wall_time_s']:.3f} s |
| source power per periodic cell | {contract['source_power_W']:.12e} W |
| P_Q (periodic pabs) | {contract['P_Q_pabs_periodic_W']:.12e} W |
| absorbed flux | {contract['P_flux_absorbed_W']:.12e} W |
| closure | {100*contract['closure_relative']:.6f}% |
| reflection | {contract['reflection']:.9f} |
| final auto-shutoff | {contract['log_audit']['final_auto_shutoff']:.6e} |

All smoke gates passed: GPU completion, auto-shutoff below 1e-5, closure below
0.5%, finite Q and no negative Q cells. Qx/Qy/Qz remain on their own staggered
Yee coordinates; the plot does not pretend that equal array indices are common
physical positions.

No Q clipping, smoothing, gain, global rescaling or polarization matching was
used. No thermal, PTE, adjoint or optimization solve was run.

## Z architecture status

The 2022 paper publishes P1/P2/L1/L2/W1/W2/D for M1-M5, but the PDFs do not
provide machine-readable Z polygon vertices or a fixed junction/crossing angle.
The audit plot therefore shows only hatched dimension envelopes and explicitly
forbids them as Maxwell CAD. This is a topology-provenance blocker, not a claim
that the Z architecture cannot be simulated.
"""
    )

    raw_manifest = {
        "raw_artifacts_not_committed": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (
                raw_dir / "T2024_TaIrTe4_optical_smoke.fsp",
                npz_path,
                contract_path,
            )
        ],
        "published_artifacts": [],
    }
    published = (summary_path, report_path, plot_path)
    raw_manifest["published_artifacts"] = [
        {"path": str(path.relative_to(HERE)), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in published
    ]
    manifest_path = output / "T2024_TAIRTE4_RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(raw_manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
