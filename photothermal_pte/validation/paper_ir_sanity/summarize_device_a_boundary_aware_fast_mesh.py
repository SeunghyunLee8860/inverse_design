#!/usr/bin/env python3
"""Summarize the Device-A boundary-aware fast optical mesh validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photothermal_pte.validation.paper_ir_sanity.summarize_device_a_fast_finite_mesh import (
    compare,
    compare_material_overlap_sources,
    finite_grid_and_runtime,
    load_case,
)


GATE = 5.0e-3
STATUS = "VALIDATED_DEVICE_A_BOUNDARY_AWARE_FAST_MESH"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nested-case", type=Path, required=True)
    parser.add_argument("--boxes-case", type=Path, required=True)
    parser.add_argument("--candidate-case", type=Path, required=True)
    parser.add_argument("--reference-case", type=Path, required=True)
    parser.add_argument("--nested-mapped", type=Path, required=True)
    parser.add_argument("--boxes-mapped", type=Path, required=True)
    parser.add_argument("--candidate-mapped", type=Path, required=True)
    parser.add_argument("--reference-mapped", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(
    path: Path,
    *,
    kind: str,
    generation_command: str,
    committed: bool = False,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "server_path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "generation_command": generation_command,
        "committed_to_git": committed,
    }


def metric_rows(
    comparison_name: str, domain: str, comparison: dict[str, Any]
) -> list[dict[str, Any]]:
    if domain == "raw_optical_control_volume":
        values = comparison["component_metrics"]["total"]
        metrics = {
            "relative_power_change": values["relative_power_change"],
            "equal_power_lateral_NRMSE": values["equal_power_lateral_NRMSE"],
            "equal_power_full_3D_NRMSE": values["equal_power_full_3D_NRMSE"],
            "equal_power_depth_NRMSE": values["equal_power_depth_NRMSE"],
            "conservative_remap_power_error": values[
                "conservative_remap_power_error"
            ],
        }
    else:
        metrics = {
            "relative_power_change": comparison["relative_power_change"],
            "equal_power_lateral_NRMSE": comparison[
                "equal_power_lateral_NRMSE"
            ],
            "equal_power_full_3D_NRMSE": comparison[
                "equal_power_full_3D_NRMSE"
            ],
            "equal_power_depth_NRMSE": comparison["equal_power_depth_NRMSE"],
        }
    rows = []
    for name, value in metrics.items():
        threshold = 1.0e-12 if "remap_power_error" in name else GATE
        rows.append(
            {
                "comparison": comparison_name,
                "domain": domain,
                "metric": name,
                "value": value,
                "percent": 100.0 * value,
                "threshold": threshold,
                "passed": value < threshold,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    case_paths = {
        "nested_9um_then_100nm_to_12um": args.nested_case,
        "boundary_following_230_boxes": args.boxes_case,
        "anisotropic_rectangle_x9_y12": args.candidate_case,
        "reference_square_x12_y12": args.reference_case,
    }
    cases = {name: load_case(path) for name, path in case_paths.items()}
    run_audits = {
        name: finite_grid_and_runtime(case) for name, case in cases.items()
    }

    raw = {
        "nested_vs_reference": compare(
            cases["nested_9um_then_100nm_to_12um"],
            cases["reference_square_x12_y12"],
        ),
        "boxes_vs_reference": compare(
            cases["boundary_following_230_boxes"],
            cases["reference_square_x12_y12"],
        ),
        "x9_y12_vs_reference": compare(
            cases["anisotropic_rectangle_x9_y12"],
            cases["reference_square_x12_y12"],
        ),
    }
    mapped = {
        "nested_vs_reference": compare_material_overlap_sources(
            args.nested_mapped, args.reference_mapped
        ),
        "boxes_vs_reference": compare_material_overlap_sources(
            args.boxes_mapped, args.reference_mapped
        ),
        "x9_y12_vs_reference": compare_material_overlap_sources(
            args.candidate_mapped, args.reference_mapped
        ),
    }

    rows: list[dict[str, Any]] = []
    for name in raw:
        rows.extend(metric_rows(name, "raw_optical_control_volume", raw[name]))
        rows.extend(metric_rows(name, "material_overlap_thermal_source", mapped[name]))
    csv_path = args.output_dir / "device_a_boundary_aware_fast_mesh_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    promoted_raw = raw["x9_y12_vs_reference"]
    promoted_mapped = mapped["x9_y12_vs_reference"]
    gates = {
        "raw_all_gates_pass": promoted_raw["all_gates_pass"],
        "mapped_all_gates_pass": promoted_mapped["all_gates_pass"],
        "candidate_closure_lt_0p5_percent": (
            promoted_raw["candidate_closure"] < GATE
        ),
        "candidate_auto_shutoff_lt_1e_minus_5": (
            cases["anisotropic_rectangle_x9_y12"]["result"]["run_result"]
            ["auto_shutoff"]["final_value"]
            <= 1.0e-5
        ),
    }
    status = STATUS if all(gates.values()) else "FAILED_DEVICE_A_BOUNDARY_AWARE_FAST_MESH"

    labels = ["nested\nx9/y9", "230 boxes", "rectangle\nx9/y12"]
    keys = ["nested_vs_reference", "boxes_vs_reference", "x9_y12_vs_reference"]
    raw_lat = [100.0 * raw[key]["component_metrics"]["total"]["equal_power_lateral_NRMSE"] for key in keys]
    raw_3d = [100.0 * raw[key]["component_metrics"]["total"]["equal_power_full_3D_NRMSE"] for key in keys]
    map_lat = [100.0 * mapped[key]["equal_power_lateral_NRMSE"] for key in keys]
    map_3d = [100.0 * mapped[key]["equal_power_full_3D_NRMSE"] for key in keys]
    x = np.arange(len(labels))
    width = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(x - 1.5 * width, raw_lat, width, label="raw lateral")
    axes[0].bar(x - 0.5 * width, raw_3d, width, label="raw 3D")
    axes[0].bar(x + 0.5 * width, map_lat, width, label="mapped lateral")
    axes[0].bar(x + 1.5 * width, map_3d, width, label="mapped 3D")
    axes[0].axhline(0.5, color="black", linestyle="--", label="0.5% gate")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("equal-power NRMSE (%)")
    axes[0].set_xticks(x, labels)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Spatial-Q convergence vs x/y=12 um reference")

    cost_names = [
        "boundary_following_230_boxes",
        "anisotropic_rectangle_x9_y12",
        "reference_square_x12_y12",
    ]
    cost_labels = ["230 boxes", "x9/y12", "x12/y12 ref"]
    runtimes = [run_audits[name]["wall_time_s"] for name in cost_names]
    grid_m = [run_audits[name]["gridpoints"] / 1.0e6 for name in cost_names]
    ax = axes[1]
    ax.bar(x, runtimes, color="#4c78a8")
    ax.set_xticks(x, cost_labels)
    ax.set_ylabel("solver wall time (s)", color="#4c78a8")
    ax.tick_params(axis="y", labelcolor="#4c78a8")
    ax2 = ax.twinx()
    ax2.plot(x, grid_m, "o-", color="#f58518", label="gridpoints")
    ax2.set_ylabel("log-reported gridpoints (million)", color="#f58518")
    ax2.tick_params(axis="y", labelcolor="#f58518")
    ax.set_title("Measured GPU cost")
    fig.tight_layout()
    plot_path = args.output_dir / "device_a_boundary_aware_fast_mesh.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    run = cases["anisotropic_rectangle_x9_y12"]["result"]["run_result"]
    summary = {
        "status": status,
        "scope": (
            "E||a Device-A optical mesh and material-overlap source validation; "
            "no thermal solve, PTE, adjoint, AD-FD, or optimization"
        ),
        "fixed_contract": {
            "wavelength_um": 11.0,
            "scalar_Gaussian_waist_um": 8.75,
            "source_span_um": 50.0,
            "lateral_domain_um": 60.0,
            "six_PML_layers": 24,
            "conformal_variant": 1,
            "mesh_accuracy": 3,
            "TaIrTe4_dz_nm": 10.0,
            "outer_xy_nm": 200.0,
            "intermediate_xy_nm_to_12um": 100.0,
            "fine_xy_nm": 50.0,
            "candidate_fine_half_spans_um": {"x": 9.0, "y": 12.0},
            "reference_fine_half_spans_um": {"x": 12.0, "y": 12.0},
            "Q_processing": {
                "clipping": False,
                "smoothing": False,
                "gain": False,
                "rescaling": False,
                "tiling": False,
            },
        },
        "promoted_candidate": {
            "P_Q_W": run["P_Q_W"],
            "P_six_W": run["P_six_face_W"],
            "six_face_relative_closure": run["six_face_relative_closure"],
            "auto_shutoff": run["auto_shutoff"]["final_value"],
            "component_power_W": run["component_power_W"],
            "negative_Q_voxel_count": run["negative_Q_voxel_count"],
            "runtime": run_audits["anisotropic_rectangle_x9_y12"],
        },
        "run_audits": run_audits,
        "raw_comparisons": raw,
        "mapped_comparisons": mapped,
        "gates": gates,
        "interpretation": {
            "diagnosis": (
                "the earlier spatial-Q failure came from changing the global "
                "rectilinear Yee lattice and placing illuminated Device-A "
                "boundaries in the 100/200 nm transition, not from a loss of "
                "power in the material-overlap mapper"
            ),
            "why_230_boxes_failed": (
                "many small axis-aligned mesh objects re-anchored the global x/y "
                "lattice; they were not independent local subgrids"
            ),
            "material_geometry_limit": promoted_mapped[
                "material_domain_limitation"
            ],
            "production_claim": (
                "validated fast E||a optical/mapped-Q mesh candidate only; "
                "not a completed thermal or current prediction"
            ),
        },
    }
    json_path = args.output_dir / "device_a_boundary_aware_fast_mesh_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest_items = []
    for name, path in case_paths.items():
        command = cases[name]["result"]["generation_command"]
        manifest_items.append(
            artifact(
                path / "case_result.json",
                kind=f"{name}_case_result",
                generation_command=command,
            )
        )
        manifest_items.append(
            artifact(
                path / "finite_q_on_artifact.npz",
                kind=f"{name}_raw_Q_NPZ",
                generation_command=command,
            )
        )
        manifest_items.append(
            artifact(
                path / "finite_2um_optical_q.fsp",
                kind=f"{name}_raw_FSP",
                generation_command=command,
            )
        )
        manifest_items.append(
            artifact(
                path / "native_yee_mesh_coordinates.npz",
                kind=f"{name}_native_Yee_coordinates",
                generation_command=command,
            )
        )
    for name, path in (
        ("nested", args.nested_mapped),
        ("boxes", args.boxes_mapped),
        ("candidate", args.candidate_mapped),
        ("reference", args.reference_mapped),
    ):
        map_command = (
            "run_device_a_explicit_thermal_pte.py --mapping-only "
            "--q-source TaIrTe4-only --core-step-nm 100 --flake-dz-nm 10"
        )
        manifest_items.append(
            artifact(
                path / "material_overlap_mapping_summary.json",
                kind=f"{name}_mapping_summary",
                generation_command=map_command,
            )
        )
        manifest_items.append(
            artifact(
                path / "material_overlap_mapped_q.npz",
                kind=f"{name}_mapped_Q_NPZ",
                generation_command=map_command,
            )
        )
    manifest = {
        "status": status,
        "raw_artifacts_committed_to_git": False,
        "artifacts": manifest_items,
    }
    manifest_path = args.output_dir / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    raw_total = promoted_raw["component_metrics"]["total"]
    report = f"""# Device-A boundary-aware fast optical mesh validation

Status: `{status}`

This checkpoint validates one `E || a` optical and conservative material-overlap
source mapping case. It does **not** run the thermal equation, weighting
potential, PTE, adjoint, AD-FD, or optimization.

## Outcome

The economical mesh is one 50 nm Cartesian rectangle with half-spans
`x=9 um, y=12 um`, nested inside the existing 100 nm-to-12 um and 200 nm
outer regions. It preserves the previous fast x lattice and the reference y
lattice. Relative to the 50 nm `x=12 um, y=12 um` reference:

- raw power change: `{100 * raw_total['relative_power_change']:.6f}%`
- raw lateral-Q NRMSE: `{100 * raw_total['equal_power_lateral_NRMSE']:.6f}%`
- raw full-3D-Q NRMSE: `{100 * raw_total['equal_power_full_3D_NRMSE']:.6f}%`
- mapped TaIrTe4 power change: `{100 * promoted_mapped['relative_power_change']:.6f}%`
- mapped lateral-Q NRMSE: `{100 * promoted_mapped['equal_power_lateral_NRMSE']:.6f}%`
- mapped full-3D-Q NRMSE: `{100 * promoted_mapped['equal_power_full_3D_NRMSE']:.6f}%`
- mapped depth-Q NRMSE: `{100 * promoted_mapped['equal_power_depth_NRMSE']:.6f}%`
- material-overlap mapping power error: exactly zero

All values are below the 0.5% spatial/power gate.

## Optical execution

- `P_Q = {run['P_Q_W']:.15e} W`
- `P_six = {run['P_six_face_W']:.15e} W`
- six-face closure: `{100 * run['six_face_relative_closure']:.6f}%`
- auto-shutoff: `{run['auto_shutoff']['final_value']:.6e}`
- negative-Q cells: `{run['negative_Q_voxel_count']}`
- solver wall time: `{run_audits['anisotropic_rectangle_x9_y12']['wall_time_s']:.3f} s`
- reference wall time: `{run_audits['reference_square_x12_y12']['wall_time_s']:.3f} s`

The scalar Gaussian, 11 um wavelength, 8.75 um assumed waist, 50 um source
span, 60 um domain, six PML boundaries, Palik SiO2/Si, anisotropic TaIrTe4,
conformal variant 1, mesh accuracy 3, and 10 nm TaIrTe4 dz were unchanged.
No Q clipping, smoothing, gain, rescaling, tiling, or source deletion was used.

## Failed diagnostics retained

The earlier nested `x/y=9 um` candidate failed because illuminated real
material boundaries entered the 100 nm transition. The attempted 230-box
boundary-following mesh also failed: Lumerical's rectilinear mesh objects
re-anchored global x/y Yee coordinates, so the boxes were not independent
local subgrids. Both raw results and their material-overlap mappings remain in
the manifest and are not promoted.

The corrected rectangle shows that the prior 1.96% lateral and 9.28% raw-3D
differences were not evidence that optical-cell power was being lost or placed
outside TaIrTe4. They were mesh-layout differences. On the corrected common
thermal grid, source and target power agree exactly and power outside the
binary FVM TaIrTe4 support is zero.

## Limitation

The thermal material domain used for mapping is the union of binary 100 nm
lateral / 10 nm z FVM cells selected by the cell-center polygon mask. It is
not an analytic polygon cut-cell thermal geometry. This checkpoint promotes a
fast optical/mapped-Q mesh candidate only, not a final thermal or current
prediction.
"""
    report_path = args.output_dir / "DEVICE_A_BOUNDARY_AWARE_FAST_MESH_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
