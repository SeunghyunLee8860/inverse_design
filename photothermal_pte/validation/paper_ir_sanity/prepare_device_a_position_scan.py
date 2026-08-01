#!/usr/bin/env python3
"""Freeze the three-point Figure-3H/I Device-A scan contract.

This is geometry-only bookkeeping.  It does not run Maxwell, thermal, PTE,
adjoint, AD-FD, or optimization.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.validation.paper_ir_sanity.run_lumerical_device_a_ir_q import (  # noqa: E402
    load_digitized_device_a_contract,
)


SCAN_OFFSETS_UM = (-1.0, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain-um", type=float, default=60.0)
    parser.add_argument("--source-span-um", type=float, default=50.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = json.loads(args.geometry_contract.read_text())
    frozen = load_digitized_device_a_contract(
        args.geometry_contract,
        domain_um=args.domain_um,
        source_span_um=args.source_span_um,
    )
    midpoint = np.asarray(raw["off_axis_edge_midpoint_code_um"], float)
    tangent = np.asarray(raw["off_axis_edge_unit_tangent_code"], float)
    inward = np.asarray(raw["off_axis_edge_unit_inward_normal_code"], float)
    baseline = np.asarray(raw["pre_registered_beam_center_code_um"], float)
    tangent /= np.linalg.norm(tangent)
    inward /= np.linalg.norm(inward)
    s0 = float(np.dot(baseline - midpoint, inward))
    tangent_offset = float(np.dot(baseline - midpoint, tangent))
    if abs(tangent_offset) > 1.0e-9:
        raise RuntimeError(
            "pre-registered beam center is not on the frozen edge-normal scan "
            f"line: tangent offset={tangent_offset:.9g} um"
        )
    origin_shift = np.asarray(frozen["simulation_origin_shift_um"], float)
    baseline_sim = baseline + origin_shift
    half_source = 0.5 * args.source_span_um
    half_domain = 0.5 * args.domain_um
    cases = []
    for delta_s in SCAN_OFFSETS_UM:
        s_um = s0 + delta_s
        center_code = midpoint + s_um * inward
        center_sim = center_code + origin_shift
        source_offset = delta_s * inward
        clearance = {
            "x_min": float(center_sim[0] - half_source + half_domain),
            "x_max": float(half_domain - center_sim[0] - half_source),
            "y_min": float(center_sim[1] - half_source + half_domain),
            "y_max": float(half_domain - center_sim[1] - half_source),
        }
        cases.append(
            {
                "label": f"s{delta_s:+.0f}um" if delta_s else "s0",
                "delta_s_from_s0_um": delta_s,
                "signed_s_from_edge_um": s_um,
                "beam_center_code_um": center_code.tolist(),
                "beam_center_simulation_um": center_sim.tolist(),
                "source_only_offset_simulation_um": source_offset.tolist(),
                "source_aperture_PML_clearance_um": clearance,
                "minimum_source_aperture_PML_clearance_um": min(clearance.values()),
                "reuse_existing_artifact": delta_s == 0.0,
            }
        )
    contract = {
        "status": "DEVICE_A_THREE_POSITION_SCAN_CONTRACT_FROZEN_NOT_SOLVED",
        "scope": (
            "geometry/source-coordinate audit only; no Maxwell, thermal, PTE, "
            "adjoint, AD-FD, optimization, or parameter fitting"
        ),
        "coordinate_axes": {"x": "crystal b", "y": "crystal a"},
        "scan_line": {
            "origin_code_um": midpoint.tolist(),
            "direction_increasing_s": "inward normal from TaIrTe4/air edge",
            "unit_direction_code": inward.tolist(),
            "unit_tangent_code": tangent.tolist(),
            "signed_coordinate": "r(s)=edge_midpoint+s*inward_normal",
            "s_zero_meaning": "digitized TaIrTe4/air edge midpoint",
            "existing_single_position_s0_um": s0,
            "existing_single_position_tangent_offset_um": tangent_offset,
            "paper_interpretation": (
                "Figure-3H black-dotted-line counterpart on the digitized Device-A "
                "geometry; not exact experimental stage coordinates"
            ),
        },
        "uncertainty_um": {
            "visible_flake_edge": raw["uncertainty_um"]["visible_flake_edge"],
            "beam_center_registration": raw["uncertainty_um"][
                "beam_center_registration"
            ],
            "exact_stage_coordinate_available": False,
        },
        "frozen_simulation_frame": {
            "origin_shift_um": origin_shift.tolist(),
            "baseline_beam_center_simulation_um": baseline_sim.tolist(),
            "device_PML_monitors_fixed_across_scan": True,
            "local_mesh_center_fixed_at_s0": True,
            "only_source_center_moves": True,
            "domain_um": args.domain_um,
            "source_span_um": args.source_span_um,
        },
        "cases": cases,
        "limits": {
            "new_position_optical_solves": 4,
            "additional_Au_Ti_off_optical_diagnostic": 1,
            "no_25nm_or_12p5nm_refinement": True,
            "no_dense_scan": True,
        },
    }
    (args.output_dir / "device_a_position_scan_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )
    with (args.output_dir / "device_a_position_scan_cases.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "label",
                "delta_s_from_s0_um",
                "signed_s_from_edge_um",
                "beam_x_code_um",
                "beam_y_code_um",
                "beam_x_simulation_um",
                "beam_y_simulation_um",
                "source_offset_x_um",
                "source_offset_y_um",
                "minimum_PML_clearance_um",
                "reuse_existing_artifact",
            ]
        )
        for case in cases:
            writer.writerow(
                [
                    case["label"],
                    case["delta_s_from_s0_um"],
                    case["signed_s_from_edge_um"],
                    *case["beam_center_code_um"],
                    *case["beam_center_simulation_um"],
                    *case["source_only_offset_simulation_um"],
                    case["minimum_source_aperture_PML_clearance_um"],
                    case["reuse_existing_artifact"],
                ]
            )

    flake = np.asarray(raw["flake_vertices_code_um"], float)
    top = np.asarray(raw["top_metal_polygon_code_um"], float)
    bottom = np.asarray(raw["bottom_metal_polygon_code_um"], float)
    figure, axis = plt.subplots(figsize=(8.5, 8), constrained_layout=True)
    axis.add_patch(Polygon(flake, facecolor="#8b5fbf", alpha=0.45, edgecolor="#542788"))
    axis.add_patch(Polygon(top, facecolor="#d4af37", alpha=0.65, edgecolor="#8c6d00"))
    axis.add_patch(Polygon(bottom, facecolor="#d4af37", alpha=0.65, edgecolor="#8c6d00"))
    line_s = np.linspace(-2.0, 8.0, 200)
    line = midpoint[None, :] + line_s[:, None] * inward[None, :]
    axis.plot(line[:, 0], line[:, 1], "k--", lw=1.4, label="digitized scan line")
    axis.scatter(*midpoint, marker="x", s=80, color="black", label=r"$s=0$ edge")
    colors = ("#1f77b4", "#d62728", "#2ca02c")
    for color, case in zip(colors, cases):
        center = np.asarray(case["beam_center_code_um"])
        axis.scatter(*center, s=80, color=color, zorder=5)
        axis.annotate(
            f"{case['label']}\n$s={case['signed_s_from_edge_um']:.0f}$ µm",
            center + np.asarray([0.25, 0.25]),
            color=color,
        )
    axis.set(
        xlabel="x = b (µm)",
        ylabel="y = a (µm)",
        title="Digitized Device A: frozen three-position source scan",
        aspect="equal",
    )
    axis.legend(loc="upper right")
    figure.savefig(args.output_dir / "DEVICE_A_POSITION_SCAN_GEOMETRY.png", dpi=200)
    plt.close(figure)

    report = f"""# Device-A three-position scan contract

Status: `DEVICE_A_THREE_POSITION_SCAN_CONTRACT_FROZEN_NOT_SOLVED`

This checkpoint freezes the low-cost position-sensitivity test. It does not
run Maxwell, thermal, PTE, adjoint, AD-FD, optimization, or empirical fitting.

The scan origin is the digitized off-axis TaIrTe4/air edge midpoint
`{midpoint.tolist()}` um. Increasing `s` follows the inward unit normal
`{inward.tolist()}`. Thus `s=0` is the digitized edge, and the pre-registered
single-position beam lies at `s0={s0:.9f}` um. The exact experimental stage
coordinate is unavailable; this is the Figure-3H black-dotted-line counterpart
on the figure-digitized geometry.

Across all three cases the Device-A polygons, PML, monitors, simulation origin,
and local 50-nm mesh remain fixed. Only the scalar-Gaussian source center moves.
The three signed coordinates are 2, 3, and 4 um from the edge. The minimum
source-aperture/PML clearance remains {min(c['minimum_source_aperture_PML_clearance_um'] for c in cases):.6f} um.

The `s0` optical and thermal artifacts are immutable and reused. At `s0±1 um`,
only the a/b GPU optical cases are new (four solves maximum). A separate Au/Ti-
off `E||a` diagnostic is conditional on the position cases passing their gates.
"""
    (args.output_dir / "DEVICE_A_POSITION_SCAN_CONTRACT.md").write_text(report)
    print(json.dumps(contract, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
