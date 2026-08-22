#!/usr/bin/env python3
"""Publish six-polarization periodic reference-Q maps for T or Z."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse


HERE = Path(__file__).resolve().parent
RAW = Path("/home/seunghyun/tairte4/raw_artifacts/periodic_T_Z_six_polarization_20260822/selected_Q")
POLS = (
    "x_b", "y_a", "linear_plus_45", "linear_minus_45", "CP_plus", "CP_minus"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dual_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, float)
    edges = np.empty(centers.size + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def overlap_matrix(target: np.ndarray, source: np.ndarray) -> sparse.csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row, (left, right) in enumerate(zip(target[:-1], target[1:])):
        hits = np.flatnonzero((source[:-1] < right) & (source[1:] > left))
        for column in hits:
            overlap = min(right, source[column + 1]) - max(left, source[column])
            if overlap > 0.0:
                rows.append(row); columns.append(int(column)); values.append(float(overlap))
    return sparse.coo_matrix(
        (values, (rows, columns)), shape=(target.size - 1, source.size - 1)
    ).tocsr()


def common_q(data: np.lib.npyio.NpzFile, px: float, py: float, step: float):
    xe = np.linspace(-px / 2, px / 2, int(round(px / step)) + 1)
    ye = np.linspace(-py / 2, py / 2, int(round(py / step)) + 1)
    cell_area = np.diff(xe)[:, None] * np.diff(ye)[None, :]
    total = np.zeros(cell_area.shape)
    components: dict[str, np.ndarray] = {}
    powers: dict[str, float] = {}
    for component in "xyz":
        q = np.asarray(data[f"Q{component}_W_m3"], float)
        sx = dual_edges(data[f"Q{component}_x_m"])
        sy = dual_edges(data[f"Q{component}_y_m"])
        sz = dual_edges(data[f"Q{component}_z_m"])
        areal = np.sum(q * np.diff(sz)[None, None, :], axis=2)
        mapped_power = overlap_matrix(xe, sx) @ areal @ overlap_matrix(ye, sy).T
        mapped = np.asarray(mapped_power) / cell_area
        components[component] = mapped
        powers[component] = float(np.sum(mapped_power))
        total += mapped
    return xe, ye, total, components, powers


def main() -> int:
    architecture = os.environ.get("PERIODIC_ARCHITECTURE", "").strip().upper()
    if architecture not in ("T", "Z"):
        raise RuntimeError("set PERIODIC_ARCHITECTURE=T or Z")
    output = HERE / f"results_periodic_{architecture}_six_polarization_reference_Q"
    output.mkdir(parents=True, exist_ok=True)
    if architecture == "T":
        json_name = "T2024_TaIrTe4_optical_smoke.json"
        npz_name = "T2024_TaIrTe4_native_q.npz"
        fsp_name = "T2024_TaIrTe4_optical_smoke.fsp"
        expected = "COMPLETED_T2024_TAIRTE4_OPTICAL_SMOKE"
        step = 10e-9
    else:
        json_name = "Z2022_M2_selected_Q.json"
        npz_name = "Z2022_M2_selected_Q.npz"
        fsp_name = "Z2022_M2_selected_Q.fsp"
        expected = "COMPLETED_Z2022_M2_CENTERED_EXPANDED_SELECTED_Q"
        step = 25e-9
    loaded: dict[str, dict[str, object]] = {}
    artifacts: list[dict[str, object]] = []
    for polarization in POLS:
        root = RAW / architecture / polarization
        metadata_path, npz_path, fsp_path = root / json_name, root / npz_name, root / fsp_name
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("status") != expected or not all(metadata.get("gates", {}).values()):
            raise RuntimeError(f"{architecture}/{polarization} Q gate: {metadata.get('status')}")
        geometry = metadata["contract"]["geometry"] if architecture == "T" else metadata["geometry"]
        px = float(geometry["period_x_nm"]) * 1e-9
        py = float(geometry["period_y_nm"]) * 1e-9
        with np.load(npz_path, allow_pickle=False) as data:
            xe, ye, total, components, powers = common_q(data, px, py, step)
        loaded[polarization] = {
            "metadata": metadata, "xe": xe, "ye": ye, "total": total,
            "components": components, "powers": powers,
        }
        for path in (metadata_path, npz_path, fsp_path):
            artifacts.append(
                {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            )

    vmax = max(float(np.percentile(item["total"], 99.8)) for item in loaded.values())
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, polarization in zip(axes.flat, POLS):
        item = loaded[polarization]
        image = ax.pcolormesh(
            item["xe"] * 1e6, item["ye"] * 1e6, item["total"].T,
            shading="flat", cmap="inferno", vmin=0.0, vmax=vmax,
        )
        ax.set_aspect("equal")
        ax.set_title(f"{polarization}: depth-integrated Q")
        ax.set_xlabel("x=b (um)"); ax.set_ylabel("y=a (um)")
        fig.colorbar(image, ax=ax, label="W/m2 per 1 W/m2 incident")
    fig.suptitle(f"Periodic {architecture}: six-polarization reference Q (no thermal/PTE)")
    overview = output / f"{architecture}_six_polarization_reference_Q.png"
    fig.savefig(overview, dpi=220)
    plt.close(fig)

    rows: list[dict[str, object]] = []
    for polarization in POLS:
        item = loaded[polarization]
        metadata = item["metadata"]
        row = {
            "polarization": polarization,
            "P_Q_W": metadata["P_Q_pabs_periodic_W"],
            "P_flux_W": metadata["P_flux_absorbed_W"],
            "closure_relative": metadata["closure_relative"],
            "Qx_common_W": item["powers"]["x"],
            "Qy_common_W": item["powers"]["y"],
            "Qz_common_W": item["powers"]["z"],
            "auto_shutoff": metadata["log_audit"]["final_auto_shutoff"],
        }
        rows.append(row)
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
        case_vmax = max(float(np.percentile(item["components"][c], 99.8)) for c in "xyz")
        for ax, component in zip(axes[:3], "xyz"):
            image = ax.pcolormesh(
                item["xe"] * 1e6, item["ye"] * 1e6,
                item["components"][component].T, shading="flat", cmap="inferno",
                vmin=0.0, vmax=case_vmax,
            )
            ax.set_aspect("equal"); ax.set_title(f"Q{component}")
            ax.set_xlabel("x=b (um)"); ax.set_ylabel("y=a (um)")
            fig.colorbar(image, ax=ax, label="W/m2")
        axes[3].bar(list("xyz"), [item["powers"][c] for c in "xyz"])
        axes[3].set_title("common-grid component powers")
        axes[3].set_ylabel("W/cell per 1 W/m2 incident")
        fig.suptitle(
            f"{architecture} {polarization}: P_Q={row['P_Q_W']:.6e} W, closure={row['closure_relative']:.3%}"
        )
        fig.savefig(output / f"{architecture}_{polarization}_Q_components.png", dpi=200)
        plt.close(fig)

    csv_path = output / f"{architecture}_six_polarization_reference_Q.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "status": f"VALIDATED_PERIODIC_{architecture}_SIX_POLARIZATION_REFERENCE_Q",
        "scope": "periodic optical volumetric Q only; no thermal/weighting/PTE",
        "cases": rows,
        "no_clipping_smoothing_gain_rescaling": True,
    }
    (output / f"PERIODIC_{architecture}_SIX_POLARIZATION_Q_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (output / "RAW_ARTIFACT_MANIFEST.json").write_text(json.dumps(artifacts, indent=2) + "\n")
    report_lines = [
        f"# Periodic {architecture} six-polarization reference Q", "",
        f"Status: `{summary['status']}`", "",
        "This is optical Q only. Periodic temperature, weighting field and PTE were not computed.", "",
        "| polarization | P_Q (W/cell) | P_flux (W/cell) | closure | Qx | Qy | Qz |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['polarization']} | {row['P_Q_W']:.7e} | {row['P_flux_W']:.7e} | "
            f"{row['closure_relative']:.4%} | {row['Qx_common_W']:.7e} | "
            f"{row['Qy_common_W']:.7e} | {row['Qz_common_W']:.7e} |"
        )
    (output / f"PERIODIC_{architecture}_SIX_POLARIZATION_Q_REPORT.md").write_text(
        "\n".join(report_lines) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
