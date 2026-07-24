#!/usr/bin/env python3
"""Collect the optical-only Pabs diagnostic matrix and independent mesh sweeps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-root", required=True)
    parser.add_argument("--flat-dir", required=True)
    parser.add_argument("--disk-dir", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def case_row(label: str, geometry: str, directory: Path) -> dict[str, Any]:
    summary_path = directory / "fdtd_absorption_summary.json"
    summary = read_json(summary_path)
    absorption = summary["absorption"]
    a_internal = float(absorption["Pabs_total_normalized_from_lumerical"])
    a_python = float(
        absorption.get(
            "A_Q_python_exported_integral",
            absorption["P_abs_volume_native_W"]
            / summary["source"]["source_power_fdtd_W"],
        )
    )
    a_flux = float(absorption["A_local_flux"])
    return {
        "label": label,
        "geometry": geometry,
        "case_dir": str(directory),
        "A_Q_lumerical_internal": a_internal,
        "A_Q_python_exported": a_python,
        "A_local_flux": a_flux,
        "A_Q_minus_A_local_flux": a_python - a_flux,
        "signed_mismatch": (a_python - a_flux) / a_flux,
        "absolute_mismatch": abs(a_python - a_flux) / abs(a_flux),
        "internal_minus_python": a_internal - a_python,
        "global_flux_absorptance": float(absorption["A_flux"]),
        "summary_path": str(summary_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_geometry(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [row["label"] for row in rows]
    x = np.arange(len(rows))
    width = 0.38
    fig, axes = plt.subplots(
        2, 1, figsize=(8.0, 7.5), gridspec_kw={"height_ratios": [2.0, 1.0]}
    )
    axes[0].bar(
        x - width / 2,
        [row["A_Q_python_exported"] for row in rows],
        width,
        label=r"$A_Q$",
    )
    axes[0].bar(
        x + width / 2,
        [row["A_local_flux"] for row in rows],
        width,
        label=r"$A_{\rm local\ flux}$",
    )
    axes[0].set_ylabel("absorptance")
    axes[0].set_xticks(x, labels)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        x, [100 * row["signed_mismatch"] for row in rows], color="#cc6677"
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("signed mismatch [%]")
    axes[1].set_xticks(x, labels)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_sweep(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    xlabel: str,
) -> None:
    rows = sorted(rows, key=lambda row: float(row[x_key]))
    x = [float(row[x_key]) for row in rows]
    mismatch = [100 * float(row["signed_mismatch"]) for row in rows]
    a_q = [float(row["A_Q_python_exported"]) for row in rows]
    a_flux = [float(row["A_local_flux"]) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.6), sharex=True)
    axes[0].plot(x, a_q, "o-", label=r"$A_Q$")
    axes[0].plot(x, a_flux, "s-", label=r"$A_{\rm local\ flux}$")
    axes[0].set_ylabel("absorptance")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(x, mismatch, "o-", color="#cc6677")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("signed mismatch [%]")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_disk_slices(path: Path, diagnostics_path: Path) -> None:
    diagnostics = read_json(diagnostics_path)
    slice_path = Path(
        diagnostics["diagnostics"]["solver_epsilon_material_slices"]["path"]
    )
    data = np.load(slice_path)
    x = np.asarray(data["x_m"], float) * 1e6
    y = np.asarray(data["y_m"], float) * 1e6
    z = np.asarray(data["z_m"], float) * 1e6
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.6))
    images = (
        (x, y, np.imag(data["epsilon_x_xy"]), r"Im($\epsilon_x$), xy in TaIrTe4"),
        (x, y, data["material_id_xy"], "material ID, xy in TaIrTe4"),
        (x, z, np.imag(data["epsilon_x_xz"]), r"Im($\epsilon_x$), xz at y=0"),
        (x, z, data["material_id_xz"], "material ID, xz at y=0"),
    )
    for ax, (a, b, values, title) in zip(axes.flat, images):
        shown = ax.pcolormesh(a, b, np.asarray(values).T, shading="auto")
        fig.colorbar(shown, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x [um]")
        ax.set_ylabel("y [um]" if b is y else "z [um]")
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = Path(args.diagnostic_root).expanduser().resolve()
    flat = Path(args.flat_dir).expanduser().resolve()
    disk = Path(args.disk_dir).expanduser().resolve()
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)

    geometry_rows = [
        case_row("A", "no design", flat),
        case_row("B", "full-cell uniform n=4 film, touching", root / "geometry_matrix/B_full_film/fdtd_qon"),
        case_row("C", "grid-aligned square n=4 block, touching", root / "geometry_matrix/C_square/fdtd_qon"),
        case_row("D", "circular n=4 disk, touching", disk),
        case_row("E", "circular n=4 disk, 50 nm air gap", root / "geometry_matrix/E_disk_gap50nm/fdtd_qon"),
    ]
    z_cases = [
        (10.0, root / "mesh_sweep_z/disk_dz10nm/fdtd_qon"),
        (5.0, disk),
        (2.5, root / "mesh_sweep_z/disk_dz2p5nm/fdtd_qon"),
    ]
    xy_cases = [
        (50.0, root / "mesh_sweep_xy/disk_designres20/fdtd_qon"),
        (25.0, disk),
        (12.5, root / "mesh_sweep_xy/disk_designres80/fdtd_qon"),
    ]
    z_rows = []
    for dz_nm, directory in z_cases:
        row = case_row(str(dz_nm), "circular disk", directory)
        row["flake_dz_nm"] = dz_nm
        z_rows.append(row)
    xy_rows = []
    for step_nm, directory in xy_cases:
        row = case_row(str(step_nm), "circular disk", directory)
        row["design_edge_step_nm"] = step_nm
        xy_rows.append(row)

    delta = read_json(root / "deltaq/deltaq_spatial_summary.json")
    flat_diag = read_json(flat / "pabs_case_diagnostics.json")
    disk_diag = read_json(disk / "pabs_case_diagnostics.json")
    result = {
        "scope": {
            "maxwell_only": True,
            "heat_run": False,
            "strict_geometric_clipping": False,
            "flux_gain_applied": False,
        },
        "deltaq": delta,
        "geometry_matrix": geometry_rows,
        "z_interface_mesh_sweep": z_rows,
        "xy_edge_mesh_sweep": xy_rows,
        "flat_diagnostics": flat_diag,
        "disk_diagnostics": disk_diag,
    }
    (output / "pabs_diagnostic_results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    write_csv(output / "geometry_matrix.csv", geometry_rows)
    write_csv(output / "mesh_sweep_z.csv", z_rows)
    write_csv(output / "mesh_sweep_xy.csv", xy_rows)
    plot_geometry(output / "geometry_matrix.png", geometry_rows)
    plot_sweep(
        output / "mesh_sweep_z.png",
        z_rows,
        x_key="flake_dz_nm",
        xlabel="TaIrTe4 override dz [nm]",
    )
    plot_sweep(
        output / "mesh_sweep_xy.png",
        xy_rows,
        x_key="design_edge_step_nm",
        xlabel="design edge grid step [nm]",
    )
    plot_disk_slices(
        output / "disk_epsilon_material_slices.png",
        disk / "pabs_case_diagnostics.json",
    )

    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
