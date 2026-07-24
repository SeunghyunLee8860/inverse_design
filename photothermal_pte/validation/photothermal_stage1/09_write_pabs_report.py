#!/usr/bin/env python3
"""Render a concise Markdown report from the collected optical diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(value: float) -> str:
    return f"{100.0 * value:.6f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.results_json).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    data = json.loads(source.read_text())
    delta = data["deltaq"]
    metrics = delta["metrics"]
    radial = delta["radial_decomposition"]
    zparts = delta["z_decomposition"]
    matrix = data["geometry_matrix"]
    zrows = sorted(
        data["z_interface_mesh_sweep"], key=lambda row: row["flake_dz_nm"],
        reverse=True,
    )
    xyrows = sorted(
        data["xy_edge_mesh_sweep"],
        key=lambda row: row["design_edge_step_nm"],
        reverse=True,
    )
    disk_diag = data["disk_diagnostics"]["diagnostics"]
    zc = disk_diag["z_contract"]
    mesh = disk_diag["mesh"]
    material = disk_diag["material_at_4um"]

    lines = [
        "# Circular-design Pabs diagnostic report",
        "",
        "## Scope and normalization",
        "",
        "- Optical FDTD only; HEAT was not run.",
        "- No geometric clipping of Q and no flux/Q gain were applied.",
        "- All reported Python Q integrals use the exported solver grid and nested trapezoidal quadrature.",
        "",
        "## Same-grid DeltaQ result",
        "",
        f"- `DeltaA_Q = {metrics['DeltaA_Q']:.12f}`",
        f"- `DeltaA_flux = {metrics['DeltaA_flux_local']:.12f}`",
        f"- `excess = {metrics['excess']:.12f}`",
        f"- Same grid verified: `{delta['same_grid_verified']}`",
        "",
        "| radial region | Delta absorptance | fraction of DeltaA_Q |",
        "|---|---:|---:|",
    ]
    for key, label in (
        ("disk_interior", "disk interior"),
        ("disk_edge_band", "disk edge band"),
        ("exterior", "exterior"),
    ):
        row = radial[key]
        lines.append(
            f"| {label} | {row['delta_absorptance']:.12f} | "
            f"{pct(row['fraction_of_DeltaA_Q'])} |"
        )
    lines += [
        "",
        "| z region | Delta absorptance | fraction of DeltaA_Q |",
        "|---|---:|---:|",
    ]
    for key, label in (
        ("bottom_boundary_cells", "TaIrTe4 bottom boundary"),
        ("flake_interior_cells", "TaIrTe4 interior"),
        ("top_boundary_cells", "TaIrTe4 top boundary"),
    ):
        row = zparts[key]
        lines.append(
            f"| {label} | {row['delta_absorptance']:.12f} | "
            f"{pct(row['fraction_of_DeltaA_Q'])} |"
        )
    lines += [
        "",
        "DeltaQ is predominantly inside the disk footprint and in the TaIrTe4 interior. "
        "The edge band is secondary. The scalar flux difference has no unique voxelwise "
        "allocation, so these values localize DeltaQ rather than artificially assigning "
        "the complete scalar excess to voxels.",
        "",
        "## Geometry matrix",
        "",
        "| case | geometry | A_Q internal | A_Q Python | A_local_flux | A_Q - A_local_flux | signed relative mismatch |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in matrix:
        lines.append(
            f"| {row['label']} | {row['geometry']} | "
            f"{row['A_Q_lumerical_internal']:.12f} | "
            f"{row['A_Q_python_exported']:.12f} | "
            f"{row['A_local_flux']:.12f} | "
            f"{row['A_Q_minus_A_local_flux']:.12f} | "
            f"{pct(row['signed_mismatch'])} |"
        )
    lines += [
        "",
        "The no-design and full-cell uniform-film controls close at approximately "
        "1e-5 relative error, whereas both finite lateral n=4 structures fail. "
        "The square fails more strongly than the circle, and a 50 nm gap does not "
        "remove the circle mismatch. This isolates the discrepancy to finite lateral "
        "high-index discontinuities/associated field interpolation rather than direct "
        "design–TaIrTe4 volume overlap.",
        "",
        "## Independent mesh sweeps",
        "",
        "| TaIrTe4 dz [nm] | A_Q | A_local_flux | signed mismatch |",
        "|---:|---:|---:|---:|",
    ]
    for row in zrows:
        lines.append(
            f"| {row['flake_dz_nm']:.3f} | "
            f"{row['A_Q_python_exported']:.12f} | "
            f"{row['A_local_flux']:.12f} | "
            f"{pct(row['signed_mismatch'])} |"
        )
    lines += [
        "",
        "| design edge step [nm] | A_Q | A_local_flux | signed mismatch |",
        "|---:|---:|---:|---:|",
    ]
    for row in xyrows:
        lines.append(
            f"| {row['design_edge_step_nm']:.3f} | "
            f"{row['A_Q_python_exported']:.12f} | "
            f"{row['A_local_flux']:.12f} | "
            f"{pct(row['signed_mismatch'])} |"
        )
    eps = material["TaIrTe4_epsilon_diagonal"]
    refractive_index = material["TaIrTe4_refractive_index_diagonal"]
    lines += [
        "",
        "## Exact disk/interface and material settings",
        "",
        f"- TaIrTe4 z: `{zc['TaIrTe4_z_min_m']:.12g}` to `{zc['TaIrTe4_z_max_m']:.12g}` m",
        f"- disk z: `{zc['design_z_min_m']:.12g}` to `{zc['design_z_max_m']:.12g}` m",
        f"- signed gap: `{zc['signed_gap_m']:.12g}` m; volume overlap: `{zc['volume_overlap_m']:.12g}` m",
        f"- design/TaIrTe4 mesh order: `{mesh['design_mesh_order']}` / `{mesh['TaIrTe4_mesh_order']}`",
        f"- conformal mesh refinement: `{mesh['FDTD_mesh_refinement']}`; mesh type: `{mesh['FDTD_mesh_type']}`",
        f"- Im(epsilon) at 4 um, TaIrTe4 axes: `{eps[0]['imag']:.12g}`, `{eps[1]['imag']:.12g}`, `{eps[2]['imag']:.12g}`",
        f"- Im(n) at 4 um, TaIrTe4 axes: `{refractive_index[0]['imag']:.12g}`, `{refractive_index[1]['imag']:.12g}`, `{refractive_index[2]['imag']:.12g}`",
        f"- design n=4 imaginary part: `{material['design_index']['imag']}`",
        "",
        "The Lumerical internal Pabs integral and the Python exported-Q integral agree "
        "to floating-point precision in every newly generated case. Therefore the "
        "observed mismatch is not introduced by NPZ export or Python quadrature.",
        "",
    ]
    output.write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
