#!/usr/bin/env python3
"""Publish the fail-closed W12 edge-local 12.5-nm runsetup audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_AUDIT_DIR = Path(
    "/data/seunghyun/tairte4/artifacts/paper_ir_lumerical_sanity/"
    "w12_edge45_a_L60_edgeband12p5_h0p5_xy25_h15_xy50_h22_contract_"
    "retry2_20260731"
)
DEFAULT_REPORT_DIR = (
    REPOSITORY
    / "photothermal_pte"
    / "reports"
    / "paper_ir_w12_edge_local_12p5nm_runsetup"
)
REFERENCE_25NM_CASE = Path(
    "/data/seunghyun/tairte4/artifacts/paper_ir_lumerical_sanity/"
    "w12_edge45_a_L60_threelevel_xy25_h15_xy50_h22_dz5_pml24_t4_gpu5_"
    "20260731/case_result.json"
)
REFERENCE_NATIVE_CELLS = 396_307_080
REFERENCE_GPU_MEMORY_GIB = 31.764


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def plot_summary(
    output: Path,
    audit: dict[str, Any],
    coordinates: dict[str, np.ndarray],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 11))
    figure.subplots_adjust(
        left=0.07,
        right=0.97,
        bottom=0.08,
        top=0.90,
        wspace=0.24,
        hspace=0.30,
    )

    axis = axes[0, 0]
    contract = audit["edge_box_contract"]
    axis.set_aspect("equal")
    axis.fill_between(
        [-23, 23],
        [-23, 23],
        [-23, -23],
        color="#e78574",
        alpha=0.55,
    )
    axis.plot([-23, 23], [-23, 23], color="#8a2d23", linewidth=2)
    shift = contract["edge_band_half_width_um"] * np.sqrt(2.0)
    axis.plot([-23, 23], [-23 + shift, 23 + shift], "--", color="#6f3fa0")
    axis.plot([-23, 23], [-23 - shift, 23 - shift], "--", color="#6f3fa0")
    for box in contract["boxes"]:
        bounds = box["bounds_m"]
        xmin, xmax = np.asarray(bounds["x"]) * 1e6
        ymin, ymax = np.asarray(bounds["y"]) * 1e6
        axis.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor="#087e8b",
                linewidth=0.42,
            )
        )
    axis.add_patch(
        plt.Rectangle(
            (-15, -15),
            30,
            30,
            fill=False,
            edgecolor="#1e5aa8",
            linewidth=2,
        )
    )
    axis.add_patch(
        plt.Rectangle(
            (-22, -22),
            44,
            44,
            fill=False,
            edgecolor="#d27b00",
            linewidth=2,
        )
    )
    axis.set(
        xlim=(-23, 23),
        ylim=(-23, 23),
        xlabel="x=b (µm)",
        ylabel="y=a (µm)",
        title=(
            f"A. Requested diagonal band ({contract['box_count']} boxes)\n"
            "purple: |n|≤0.5 µm; blue/orange: inherited 25/50 nm"
        ),
    )

    axis = axes[0, 1]
    for direction, color in (("x", "#1769aa"), ("y", "#d1495b")):
        coordinate = coordinates[direction] * 1e6
        midpoint = 0.5 * (coordinate[:-1] + coordinate[1:])
        step = np.diff(coordinate) * 1e9
        axis.plot(midpoint, step, color=color, label=f"d{direction}")
    axis.axvspan(-15, 15, color="#1e5aa8", alpha=0.08, label="25-nm square span")
    axis.axvspan(-22, -15, color="#d27b00", alpha=0.07)
    axis.axvspan(15, 22, color="#d27b00", alpha=0.07, label="50-nm span")
    axis.axhline(12.5, color="#087e8b", linestyle="--", linewidth=1.5)
    axis.set(
        xlim=(-27.5, 27.5),
        ylim=(0, 110),
        xlabel="global native coordinate (µm)",
        ylabel="realized step (nm)",
        title=(
            "B. Realized global x/y mesh\n"
            "diagonal boxes collapse to 12.5-nm coordinate intervals"
        ),
    )
    axis.legend(fontsize=8, ncol=2)

    axis = axes[1, 0]
    new_cells = audit["realized_native_mesh"]["cell_count"]
    estimate = audit["resource_estimate"]
    categories = ["25-nm\nreference", "requested\nedge boxes"]
    cells = [REFERENCE_NATIVE_CELLS / 1e6, new_cells / 1e6]
    memory = [
        REFERENCE_GPU_MEMORY_GIB,
        estimate["estimated_GPU_memory_GiB"],
    ]
    positions = np.arange(2)
    width = 0.35
    bars1 = axis.bar(
        positions - width / 2,
        cells,
        width,
        label="Yee cells (million)",
        color="#2f75b5",
    )
    twin = axis.twinx()
    bars2 = twin.bar(
        positions + width / 2,
        memory,
        width,
        label="GPU memory (GiB)",
        color="#d95f43",
    )
    twin.axhline(
        estimate["RTX6000_Ada_capacity_GiB"],
        color="#7c203a",
        linestyle="--",
        label="GPU capacity",
    )
    axis.bar_label(bars1, fmt="%.1f", fontsize=8)
    twin.bar_label(bars2, fmt="%.1f", fontsize=8)
    axis.set_xticks(positions, categories)
    axis.set_ylabel("Yee cells (million)")
    twin.set_ylabel("estimated GPU memory (GiB)")
    axis.set_title("C. Resource preflight: 94.4 GiB > 48.0 GiB")
    handles1, labels1 = axis.get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    axis.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc="upper left")

    axis = axes[1, 1]
    axis.axis("off")
    witnesses = audit["off_edge_mesh_witnesses"]
    lines = [
        "D. Fail-closed witness audit",
        "",
        "Requested edge: max dx/dy = "
        f"{audit['requested_edge_band_readback']['maximum_dx_m']*1e9:.6f} / "
        f"{audit['requested_edge_band_readback']['maximum_dy_m']*1e9:.6f} nm  PASS",
        "TaIrTe₄: max dz = "
        f"{audit['requested_edge_band_readback']['maximum_TaIrTe4_dz_m']*1e9:.6f} nm  PASS",
        "",
    ]
    for witness in witnesses:
        x_um, y_um = witness["position_um"]
        expected = witness["inherited_expected_step_nm"]
        realized = witness["realized_step_nm"]
        lines.extend(
            [
                f"off-edge ({x_um:.3f}, {y_um:.3f}) µm, "
                f"|n|={abs(witness['edge_normal_n_um']):.3f} µm",
                "  inherited dx/dy "
                f"{expected['dx']:.1f}/{expected['dy']:.1f} nm → "
                f"realized {realized['dx']:.1f}/{realized['dy']:.1f} nm  FAIL",
            ]
        )
    lines.extend(
        [
            "",
            "No FDTD time stepping started.",
            "No E||a Q, E||b Q, thermal, PTE, adjoint, or optimization ran.",
        ]
    )
    axis.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        family="monospace",
        fontsize=9.5,
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": "#fff5e6",
            "edgecolor": "#b45f06",
        },
    )

    figure.suptitle(
        "W12 edge-local 12.5-nm Lumerical runsetup audit — blocked before FDTD",
        fontsize=16,
        weight="bold",
    )
    figure.savefig(output, dpi=210, facecolor="white")
    plt.close(figure)


def write_cases_csv(
    path: Path,
    audit: dict[str, Any],
) -> None:
    estimate = audit["resource_estimate"]
    rows = [
        {
            "case": "completed_Ea_25nm_reference",
            "FDTD_time_stepping": True,
            "native_Yee_cells": REFERENCE_NATIVE_CELLS,
            "estimated_or_precise_GPU_memory_GiB": REFERENCE_GPU_MEMORY_GIB,
            "edge_dx_nm": 25.0,
            "edge_dy_nm": 25.0,
            "flake_dz_nm": 5.0,
            "off_edge_inherited_mesh_retained": True,
            "status": "EXISTING_REFERENCE",
        },
        {
            "case": "edge_local_12p5nm_runsetup",
            "FDTD_time_stepping": False,
            "native_Yee_cells": audit["realized_native_mesh"]["cell_count"],
            "estimated_or_precise_GPU_memory_GiB": estimate[
                "estimated_GPU_memory_GiB"
            ],
            "edge_dx_nm": audit["requested_edge_band_readback"][
                "maximum_dx_m"
            ]
            * 1e9,
            "edge_dy_nm": audit["requested_edge_band_readback"][
                "maximum_dy_m"
            ]
            * 1e9,
            "flake_dz_nm": audit["requested_edge_band_readback"][
                "maximum_TaIrTe4_dz_m"
            ]
            * 1e9,
            "off_edge_inherited_mesh_retained": audit["gates"][
                "inherited_off_edge_25_50_100nm_mesh_retained"
            ],
            "status": audit["status"],
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    audit = summary["runsetup_audit"]
    mesh = audit["realized_native_mesh"]
    estimate = audit["resource_estimate"]
    report = f"""# W12 edge-local 12.5-nm runsetup audit

Status: `{summary['status']}`

![Runsetup audit](W12_EDGE_LOCAL_12P5NM_RUNSETUP_AUDIT.png)

## Outcome

The requested 108 overlapping axis-aligned boxes have no analytic gap and
do realize `dx=dy=12.5 nm` across the requested `|n|<=0.5 µm` edge band.
TaIrTe4 retains `dz=5 nm`.  However, Lumerical FDTD exposes one rectilinear
native `x/y/z` coordinate set.  Because the diagonal box union covers every
central x and y interval, runsetup refined the complete central coordinate
ranges rather than only the diagonal physical band.

- native coordinate counts: `{mesh['coordinate_counts']['x']} x
  {mesh['coordinate_counts']['y']} x {mesh['coordinate_counts']['z']}`
- native Yee cells: `{mesh['cell_count']:,}`
- existing 25-nm reference: `{REFERENCE_NATIVE_CELLS:,}` cells
- cell-count ratio: `{estimate['realized_cell_ratio']:.6f}`
- estimated GPU memory: `{estimate['estimated_GPU_memory_GiB']:.3f} GiB`
- RTX 6000 Ada capacity: `{estimate['RTX6000_Ada_capacity_GiB']:.3f} GiB`
- estimated runtime if memory existed:
  `{estimate['estimated_wall_time_hours']:.3f} h` (rough scaling only)

At the off-edge witness `(0.123,5.123) µm`, `|n|=3.536 µm`, the inherited
25/25-nm step became 12.5/12.5 nm.  The `(18.123,0.123)` and
`(25.123,0.123) µm` witnesses likewise retain their x step but have their
inherited 25-nm y step replaced by 12.5 nm.  The required off-edge 25/50/100
nm hierarchy is therefore not preserved.

The preflight gate failed and **no FDTD time stepping started**.  There is no
new `E||a` Q artifact and `E||b` was not authorized.  No thermal, PTE,
adjoint, AD-FD, or optimization calculation ran.

## Requested physical questions

1. Does the polarization-gradient reversal survive 25 to edge-local 12.5 nm?
   **Unresolved:** no valid edge-local 12.5-nm solve exists.
2. Does the reversal depend only on the one `z=0` voxel?
   **Unresolved in this checkpoint.**
3. Does it survive interface-slab and lateral-Q integration?
   **Unresolved in this checkpoint.**
4. Does it survive downstream temperature-gradient calculation?
   **Unresolved:** thermal remap was correctly not run without optical input.
5. Can the current 50-nm production Q remain?
   **Yes, only as the current operational reference for the already passed
   total-power/lateral and downstream metrics.**  It is not promoted as a
   strict edge-gradient or full-3D-interface convergence certificate; the
   edge-local refinement blocker remains explicit.

## Why the requested construction is blocked

Ansys documents the FDTD mesh as a graded Cartesian/rectangular mesh and its
datasets as rectilinear `x/y/z` coordinates.  Axis-aligned override boxes can
restrict a volume, but a diagonal union spanning all x and y coordinates
cannot produce an independently rotated strip in this solver contract.
The actual runsetup readback, rather than that documentation alone, is the
decisive evidence here.

- https://optics.ansys.com/hc/en-us/articles/360034382634
- https://optics.ansys.com/hc/en-us/articles/360034901833
- https://optics.ansys.com/hc/en-us/articles/360034409554

Possible alternatives require a new user-approved contract: shorten the
12.5-nm segment, rotate the complete optical coordinate/material problem and
validate the anisotropic tensor representation, or use an unstructured
solver.  None was silently substituted here.

## Provenance

- runsetup directory: `{summary['raw_artifact_directory']}`
- generation command: `{summary['generation_command']}`
- code commit at generation: `{summary['generation_commit']}`
- raw FSP/NPZ committed to Git: `false`
"""
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.report_dir.exists()
        and any(args.report_dir.iterdir())
        and not args.overwrite
    ):
        raise RuntimeError(f"refusing to overwrite non-empty {args.report_dir}")
    args.report_dir.mkdir(parents=True, exist_ok=True)

    audit_path = args.audit_dir / "EDGE_LOCAL_12P5NM_RUNSETUP_AUDIT.json"
    case_path = args.audit_dir / "case_result.json"
    mesh_path = args.audit_dir / "edge_local_12p5nm_runsetup_mesh_coordinates.npz"
    audit = load_json(audit_path)
    case = load_json(case_path)
    with np.load(mesh_path, allow_pickle=False) as archive:
        coordinates = {
            axis: np.asarray(archive[f"solver_{axis}_m"], float)
            for axis in "xyz"
        }

    expected_status = "BLOCKED_EDGE_LOCAL_12P5NM_RUNSETUP_AUDIT"
    if audit["status"] != expected_status:
        raise RuntimeError(f"unexpected audit status: {audit['status']}")
    if audit["FDTD_started"] or case["heat_run"]:
        raise RuntimeError("scope gate violated")

    summary = {
        "status": (
            "BLOCKED_EDGE_LOCAL_12P5NM_RECTILINEAR_MESH_NOT_LOCAL"
        ),
        "validated": False,
        "generation_command": shlex.join([sys.executable, *sys.argv]),
        "generation_commit": case["generation_commit"],
        "raw_artifact_directory": str(args.audit_dir.resolve()),
        "runsetup_audit": audit,
        "final_decisions": {
            "polarization_reversal_survives_25_to_12p5nm": None,
            "reversal_depends_only_on_z0_voxel": None,
            "reversal_survives_interface_slab_and_lateral_Q": None,
            "reversal_survives_downstream_temperature_gradient": None,
            "retain_50nm_operational_reference": True,
            "promote_50nm_strict_edge_gradient_convergence": False,
        },
        "scope": {
            "runsetup_only": True,
            "FDTD_time_stepping": False,
            "E_parallel_b_authorized": False,
            "thermal": False,
            "PTE": False,
            "adjoint": False,
            "AD_FD": False,
            "optimization": False,
        },
    }
    summary_path = (
        args.report_dir / "w12_edge_local_12p5nm_runsetup_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = args.report_dir / "w12_edge_local_12p5nm_runsetup_cases.csv"
    write_cases_csv(csv_path, audit)
    figure_path = (
        args.report_dir / "W12_EDGE_LOCAL_12P5NM_RUNSETUP_AUDIT.png"
    )
    plot_summary(figure_path, audit, coordinates)
    report_path = (
        args.report_dir / "W12_EDGE_LOCAL_12P5NM_RUNSETUP_REPORT.md"
    )
    write_report(report_path, summary)

    raw = [
        record(audit_path, "raw_runsetup_audit_JSON"),
        record(case_path, "raw_fail_closed_case_result_JSON"),
        record(mesh_path, "raw_native_mesh_coordinates_NPZ"),
        record(
            args.audit_dir / "EDGE_LOCAL_12P5NM_MESH_BOX_GEOMETRY.png",
            "raw_mesh_box_geometry_PNG",
        ),
        record(REFERENCE_25NM_CASE, "existing_25nm_Ea_reference_case_JSON"),
    ]
    published = [
        record(summary_path, "published_summary_JSON"),
        record(csv_path, "published_cases_CSV"),
        record(figure_path, "published_runsetup_audit_PNG"),
        record(report_path, "published_report"),
    ]
    manifest = {
        "status": summary["status"],
        "raw_FSP_or_NPZ_committed_to_Git": False,
        "generation_command": summary["generation_command"],
        "generation_commit": summary["generation_commit"],
        "artifacts": raw + published,
    }
    manifest_path = args.report_dir / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "native_Yee_cells": audit["realized_native_mesh"][
                    "cell_count"
                ],
                "estimated_GPU_memory_GiB": audit["resource_estimate"][
                    "estimated_GPU_memory_GiB"
                ],
                "FDTD_started": audit["FDTD_started"],
                "report_dir": str(args.report_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
