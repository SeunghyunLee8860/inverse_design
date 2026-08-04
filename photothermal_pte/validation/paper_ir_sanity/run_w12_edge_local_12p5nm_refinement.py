#!/usr/bin/env python3
"""Fail-closed 12.5-nm edge-local mesh probe for the fixed W12 contract.

This wrapper adds a union of small axis-aligned mesh override boxes along the
straight TaIrTe4 edge y=x.  Before any FDTD time stepping, it reads the
realized native Cartesian mesh after ``runsetup`` and verifies:

* <=12.5 nm dx/dy throughout the requested edge band;
* <=5 nm dz in TaIrTe4;
* the inherited 25/50/100-nm mesh remains unchanged away from the edge; and
* the estimated GPU memory fits the selected RTX 6000 Ada device.

If any preflight gate fails, a JSON/NPZ/PNG diagnostic is retained and the
wrapper raises before the GPU solver can start.  No CPU FDTD fallback exists.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.validation.paper_ir_sanity import (
    run_lumerical_device_a_ir_q as production,
)


REFERENCE_25NM_CASE = Path(
    "/data/seunghyun/tairte4/artifacts/paper_ir_lumerical_sanity/"
    "w12_edge45_a_L60_threelevel_xy25_h15_xy50_h22_dz5_pml24_t4_gpu5_"
    "20260731/case_result.json"
)
REFERENCE_25NM_LOG = REFERENCE_25NM_CASE.with_name(
    "finite_2um_optical_q_p0.log"
)
REFERENCE_NATIVE_CELLS = 396_307_080
REFERENCE_PRECISE_GPU_GIB = 31.764
REFERENCE_WALL_TIME_S = 2435.954237
RTX_6000_ADA_CAPACITY_GIB = 49140.0 / 1024.0
MESH_TOLERANCE_M = 1.0e-12


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def edge_box_contract(args: argparse.Namespace) -> dict[str, Any]:
    half_width = args.edge_band_half_width_um
    side = args.edge_box_side_um
    half_side = 0.5 * side
    u_limit = math.sqrt(2.0) * args.edge_segment_half_xy_um
    requested_spacing = args.edge_box_max_tangent_spacing_um
    interval_count = int(math.ceil((2.0 * u_limit) / requested_spacing))
    centers_u = np.linspace(-u_limit, u_limit, interval_count + 1)
    realized_spacing = float(np.max(np.diff(centers_u)))
    coverage_margin = (
        math.sqrt(2.0) * half_side
        - half_width
        - 0.5 * realized_spacing
    )
    if coverage_margin < -1.0e-12:
        raise ValueError(
            "edge boxes leave a mathematical gap in the requested band: "
            f"coverage margin={coverage_margin:g} um"
        )
    centers_xy = centers_u / math.sqrt(2.0)
    boxes = []
    z_min = -production.FLAKE_THICKNESS_M - 10.0 * args.flake_dz_nm * 1e-9
    z_max = 10.0 * args.flake_dz_nm * 1e-9
    for index, coordinate_um in enumerate(centers_xy):
        boxes.append(
            {
                "name": f"edge_local_12p5_mesh_{index:03d}",
                "center_u_um": float(centers_u[index]),
                "center_xy_um": [float(coordinate_um), float(coordinate_um)],
                "bounds_m": {
                    "x": [
                        (coordinate_um - half_side) * 1e-6,
                        (coordinate_um + half_side) * 1e-6,
                    ],
                    "y": [
                        (coordinate_um - half_side) * 1e-6,
                        (coordinate_um + half_side) * 1e-6,
                    ],
                    "z": [z_min, z_max],
                },
            }
        )
    return {
        "edge_equation": "y=x; TaIrTe4 support y<=x",
        "normal_coordinate": "n=(-x+y)/sqrt(2)",
        "tangent_coordinate": "u=(x+y)/sqrt(2)",
        "edge_band_half_width_um": half_width,
        "edge_segment_half_xy_um": args.edge_segment_half_xy_um,
        "edge_segment_u_bounds_um": [-u_limit, u_limit],
        "box_side_um": side,
        "requested_max_tangent_spacing_um": requested_spacing,
        "realized_max_tangent_spacing_um": realized_spacing,
        "analytic_no_gap_coverage_margin_um": coverage_margin,
        "box_count": len(boxes),
        "boxes": boxes,
    }


def add_edge_boxes(
    fdtd: Any,
    args: argparse.Namespace,
    contract: dict[str, Any],
) -> None:
    mesh_m = args.edge_local_xy_mesh_nm * 1e-9
    dz_m = args.flake_dz_nm * 1e-9
    for box in contract["boxes"]:
        bounds = box["bounds_m"]
        mesh = fdtd.addmesh()
        mesh["name"] = box["name"]
        for axis in "xyz":
            mesh[f"{axis} min"], mesh[f"{axis} max"] = bounds[axis]
        mesh["override x mesh"] = 1
        mesh["override y mesh"] = 1
        mesh["override z mesh"] = 1
        mesh["dx"] = mesh_m
        mesh["dy"] = mesh_m
        mesh["dz"] = dz_m


def plot_edge_boxes(
    output: Path,
    args: argparse.Namespace,
    contract: dict[str, Any],
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 8.0))
    axis.set_aspect("equal")
    axis.set_xlim(-23, 23)
    axis.set_ylim(-23, 23)
    axis.fill_between(
        [-23, 23],
        [-23, 23],
        [-23, -23],
        color="#e78574",
        alpha=0.55,
        label=r"TaIrTe$_4$: $y\leq x$",
    )
    axis.plot([-23, 23], [-23, 23], color="#8a2d23", linewidth=2)
    n_shift = args.edge_band_half_width_um * math.sqrt(2.0)
    axis.plot(
        [-23, 23],
        [-23 + n_shift, 23 + n_shift],
        "--",
        color="#56368a",
        linewidth=1.3,
    )
    axis.plot(
        [-23, 23],
        [-23 - n_shift, 23 - n_shift],
        "--",
        color="#56368a",
        linewidth=1.3,
        label=r"requested $|n|\leq0.5\,\mu$m",
    )
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
                linewidth=0.45,
                alpha=0.75,
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
            label="inherited 25-nm square",
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
            label="inherited 50-nm square",
        )
    )
    axis.set(
        xlabel="lab x = crystal b (µm)",
        ylabel="lab y = crystal a (µm)",
        title=(
            "Requested edge-local 12.5-nm override boxes\n"
            f"{contract['box_count']} axis-aligned boxes; geometry only"
        ),
    )
    axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "EDGE_LOCAL_12P5NM_MESH_BOX_GEOMETRY.png", dpi=210)
    plt.close(figure)


def native_coordinate(fdtd: Any, axis: str) -> np.ndarray:
    raw = fdtd.getresult("FDTD", axis)
    if isinstance(raw, dict):
        raw = raw[axis]
    values = np.asarray(raw, float).reshape(-1)
    if values.size < 3 or np.any(np.diff(values) <= 0.0):
        raise RuntimeError(f"invalid realized FDTD {axis} coordinate")
    return values


def interval_step_at(coordinate: np.ndarray, position_m: float) -> float:
    index = int(np.searchsorted(coordinate, position_m, side="right") - 1)
    index = min(max(index, 0), coordinate.size - 2)
    return float(coordinate[index + 1] - coordinate[index])


def maximum_step_over_interval(
    coordinate: np.ndarray,
    lower_m: float,
    upper_m: float,
) -> float:
    low = coordinate[:-1]
    high = coordinate[1:]
    intersects = (high > lower_m) & (low < upper_m)
    if not np.any(intersects):
        raise RuntimeError(
            f"no mesh cells intersect interval [{lower_m:g},{upper_m:g}]"
        )
    return float(np.max(np.diff(coordinate)[intersects]))


def inherited_step_nm(value_um: float, args: argparse.Namespace) -> float:
    absolute = abs(value_um)
    if absolute < args.refinement_half_span_um:
        return args.local_xy_mesh_nm
    if absolute < args.intermediate_half_span_um:
        return args.intermediate_xy_mesh_nm
    return args.outer_local_xy_mesh_nm


def read_mesh_override(fdtd: Any, name: str) -> dict[str, Any]:
    def scalar(property_name: str) -> float:
        return float(np.asarray(fdtd.getnamed(name, property_name)).squeeze())

    return {
        "name": name,
        "bounds_m": {
            axis: [
                scalar(f"{axis} min"),
                scalar(f"{axis} max"),
            ]
            for axis in "xyz"
        },
        "dx_m": scalar("dx"),
        "dy_m": scalar("dy"),
        "dz_m": scalar("dz"),
    }


def build_runsetup_audit(
    fdtd: Any,
    args: argparse.Namespace,
    contract: dict[str, Any],
    pre_run: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    coordinates = {axis: native_coordinate(fdtd, axis) for axis in "xyz"}
    steps = {axis: np.diff(values) for axis, values in coordinates.items()}
    counts = {axis: int(values.size) for axis, values in coordinates.items()}
    cell_count = int(np.prod([counts[axis] - 1 for axis in "xyz"], dtype=np.int64))

    edge_extent_m = (
        args.edge_segment_half_xy_um + 0.5 * args.edge_box_side_um
    ) * 1e-6
    edge_dx = maximum_step_over_interval(
        coordinates["x"], -edge_extent_m, edge_extent_m
    )
    edge_dy = maximum_step_over_interval(
        coordinates["y"], -edge_extent_m, edge_extent_m
    )
    flake_dz = maximum_step_over_interval(
        coordinates["z"], -production.FLAKE_THICKNESS_M, 0.0
    )

    witnesses_um = [
        [0.123, 5.123],
        [18.123, 0.123],
        [25.123, 0.123],
    ]
    witnesses: list[dict[str, Any]] = []
    for x_um, y_um in witnesses_um:
        actual_nm = {
            "dx": interval_step_at(coordinates["x"], x_um * 1e-6) * 1e9,
            "dy": interval_step_at(coordinates["y"], y_um * 1e-6) * 1e9,
        }
        expected_nm = {
            "dx": inherited_step_nm(x_um, args),
            "dy": inherited_step_nm(y_um, args),
        }
        normal_um = (-x_um + y_um) / math.sqrt(2.0)
        witnesses.append(
            {
                "position_um": [x_um, y_um],
                "edge_normal_n_um": normal_um,
                "outside_requested_edge_band": (
                    abs(normal_um) > args.edge_band_half_width_um
                ),
                "inherited_expected_step_nm": expected_nm,
                "realized_step_nm": actual_nm,
                "inherited_mesh_retained": all(
                    actual_nm[key] >= expected_nm[key] - 1.0e-3
                    for key in ("dx", "dy")
                ),
            }
        )

    cell_ratio = cell_count / REFERENCE_NATIVE_CELLS
    dt_s = float(pre_run["material"]["epsilon_readback"]["dt_s"])
    # The pre-run dt readback is not used as a runtime predictor: it differs
    # unexpectedly from the solved 25-nm artifact despite the unchanged 5-nm
    # z override.  Use a transparent geometric Courant scaling instead.
    geometric_time_step_ratio = math.sqrt(
        (
            2.0 / args.edge_local_xy_mesh_nm**2
            + 1.0 / args.flake_dz_nm**2
        )
        / (
            2.0 / args.local_xy_mesh_nm**2
            + 1.0 / args.flake_dz_nm**2
        )
    )
    memory_gib = REFERENCE_PRECISE_GPU_GIB * cell_ratio
    runtime_s = (
        REFERENCE_WALL_TIME_S * cell_ratio * geometric_time_step_ratio
    )
    all_names = [
        "flake_outer_mesh",
        "flake_intermediate_mesh",
        "flake_mesh",
        *[box["name"] for box in contract["boxes"]],
    ]
    override_readback = [read_mesh_override(fdtd, name) for name in all_names]

    gates = {
        "edge_band_max_dx_le_12p5nm_plus_tolerance": (
            edge_dx <= args.edge_local_xy_mesh_nm * 1e-9 + MESH_TOLERANCE_M
        ),
        "edge_band_max_dy_le_12p5nm_plus_tolerance": (
            edge_dy <= args.edge_local_xy_mesh_nm * 1e-9 + MESH_TOLERANCE_M
        ),
        "TaIrTe4_max_dz_le_5nm_plus_tolerance": (
            flake_dz <= args.flake_dz_nm * 1e-9 + MESH_TOLERANCE_M
        ),
        "all_edge_boxes_read_back": len(override_readback) == len(all_names),
        "edge_band_has_no_analytic_box_gap": (
            contract["analytic_no_gap_coverage_margin_um"] >= -1.0e-12
        ),
        "inherited_off_edge_25_50_100nm_mesh_retained": all(
            witness["inherited_mesh_retained"] for witness in witnesses
        ),
        "estimated_GPU_memory_fits_RTX6000_Ada": (
            memory_gib < 0.90 * RTX_6000_ADA_CAPACITY_GIB
        ),
        "no_CPU_FDTD_fallback": True,
        "conformal_variant_1": (
            str(pre_run["mesh"]["refinement"]).strip().lower()
            == "conformal variant 1"
        ),
        "six_PML_24_layers": (
            all(
                str(value).strip().upper() == "PML"
                for value in pre_run["boundaries"].values()
            )
            and args.pml_layers == 24
        ),
    }
    gates["all_before_FDTD"] = all(gates.values())
    status = (
        "PASSED_EDGE_LOCAL_12P5NM_RUNSETUP_AUDIT"
        if gates["all_before_FDTD"]
        else "BLOCKED_EDGE_LOCAL_12P5NM_RUNSETUP_AUDIT"
    )
    return {
        "status": status,
        "FDTD_started": False,
        "contract": {
            "wavelength_um": 11.0,
            "scalar_Gaussian": True,
            "physical_waist_um": args.waist_um,
            "source_span_um": args.source_span_um,
            "lateral_domain_um": args.domain_um,
            "PML_layers": args.pml_layers,
            "mesh_refinement": pre_run["mesh"]["refinement"],
            "TaIrTe4_thickness_nm": production.FLAKE_THICKNESS_M * 1e9,
            "TaIrTe4_dz_nm": args.flake_dz_nm,
            "epsilon_mapping": "x=b, y=a, z=b",
            "straight_edge": "y=x; TaIrTe4 y<=x",
            "periodic_or_Bloch": False,
            "CPU_FDTD_fallback": False,
            "Q_processing": {
                "clipping": False,
                "smoothing": False,
                "gain": False,
                "rescaling": False,
                "polarization_matching": False,
            },
        },
        "edge_box_contract": contract,
        "mesh_override_readback": override_readback,
        "realized_native_mesh": {
            "coordinate_counts": counts,
            "cell_count": cell_count,
            "bounds_m": {
                axis: [float(values[0]), float(values[-1])]
                for axis, values in coordinates.items()
            },
            "minimum_step_m": {
                axis: float(np.min(values)) for axis, values in steps.items()
            },
            "maximum_step_m": {
                axis: float(np.max(values)) for axis, values in steps.items()
            },
        },
        "requested_edge_band_readback": {
            "maximum_dx_m": edge_dx,
            "maximum_dy_m": edge_dy,
            "maximum_TaIrTe4_dz_m": flake_dz,
        },
        "off_edge_mesh_witnesses": witnesses,
        "resource_estimate": {
            "method": (
                "linear cell-count scaling from the completed 25-nm a-case; "
                "runtime additionally scales by a geometric Courant ratio; "
                "runsetup dt readback is diagnostic only"
            ),
            "reference_25nm_case": str(REFERENCE_25NM_CASE),
            "reference_25nm_log": str(REFERENCE_25NM_LOG),
            "reference_native_cells": REFERENCE_NATIVE_CELLS,
            "reference_precise_GPU_memory_GiB": REFERENCE_PRECISE_GPU_GIB,
            "reference_wall_time_s": REFERENCE_WALL_TIME_S,
            "realized_cell_ratio": cell_ratio,
            "runsetup_dt_readback_s_diagnostic_only": dt_s,
            "runtime_time_step_scaling_method": (
                "geometric Courant ratio from inherited 25-nm xy / 5-nm z "
                "to requested 12.5-nm xy / unchanged 5-nm z"
            ),
            "geometric_time_step_ratio": geometric_time_step_ratio,
            "estimated_GPU_memory_GiB": memory_gib,
            "RTX6000_Ada_capacity_GiB": RTX_6000_ADA_CAPACITY_GIB,
            "estimated_wall_time_s": runtime_s,
            "estimated_wall_time_hours": runtime_s / 3600.0,
        },
        "source_PML_monitor_bounds": {
            **pre_run["object_bounds_readback_m"],
            "PML_inner_bounds": {
                axis: [
                    float(coordinates[axis][args.pml_layers]),
                    float(coordinates[axis][-args.pml_layers - 1]),
                ]
                for axis in "xyz"
            },
            "PML_inner_bounds_method": (
                "native runsetup coordinate at 24 cells from each nominal "
                "outer boundary"
            ),
        },
        "gates": gates,
        "diagnosis": (
            "A Lumerical FDTD rectilinear mesh is represented by global 1-D "
            "x/y/z coordinates.  The diagonal-box union spans the full "
            "central x and y intervals, so runsetup determines whether the "
            "supposed strip remains local or collapses to a central square."
        ),
    }, coordinates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--edge-local-xy-mesh-nm", type=float, default=12.5)
    parser.add_argument("--edge-band-half-width-um", type=float, default=0.5)
    parser.add_argument("--edge-segment-half-xy-um", type=float, default=15.0)
    parser.add_argument("--edge-box-side-um", type=float, default=1.0)
    parser.add_argument(
        "--edge-box-max-tangent-spacing-um",
        type=float,
        default=0.4,
    )
    custom, remaining = parser.parse_known_args()
    original_argv = sys.argv
    sys.argv = [original_argv[0], *remaining]
    try:
        args = ORIGINAL_PARSE_ARGS()
    finally:
        sys.argv = original_argv
    for key, value in vars(custom).items():
        setattr(args, key, value)
    if args.geometry != "straight-45-edge":
        parser.error("edge-local refinement requires straight-45-edge geometry")
    if args.local_xy_mesh_nm != 25.0 or args.refinement_half_span_um != 15.0:
        parser.error("inherited finest square must remain 25 nm over ±15 um")
    if (
        args.intermediate_xy_mesh_nm != 50.0
        or args.intermediate_half_span_um != 22.0
    ):
        parser.error("inherited intermediate square must remain 50 nm over ±22 um")
    if args.edge_local_xy_mesh_nm != 12.5:
        parser.error("edge-local mesh is fixed at 12.5 nm")
    if args.edge_segment_half_xy_um < args.refinement_half_span_um:
        parser.error("edge segment must cover the complete 25-nm square diagonal")
    if not args.contract_only:
        parser.error(
            "this checkpoint is runsetup-only; remove no gate and do not "
            "start FDTD until its audit is explicitly passed"
        )
    return args


ORIGINAL_PARSE_ARGS = production.parse_args
ORIGINAL_ADD_GEOMETRY = production.add_geometry_and_monitors
ORIGINAL_ASSERT_CONTRACT = production.assert_contract


def add_geometry_and_monitors(
    base: Any,
    fdtd: Any,
    model: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    setup = ORIGINAL_ADD_GEOMETRY(base, fdtd, model, args)
    contract = edge_box_contract(args)
    add_edge_boxes(fdtd, args, contract)
    setup["edge_local_12p5nm_contract"] = contract
    setup["geometry"]["large_domain_mesh_policy"][
        "edge_local_12p5nm_override"
    ] = {
        key: value for key, value in contract.items() if key != "boxes"
    }
    plot_edge_boxes(Path(args.output_dir), args, contract)
    return setup


def assert_contract(
    base: Any,
    fdtd: Any,
    runtime: Any,
    args: argparse.Namespace,
    setup: dict[str, Any],
) -> dict[str, Any]:
    pre_run = ORIGINAL_ASSERT_CONTRACT(base, fdtd, runtime, args, setup)
    audit, coordinates = build_runsetup_audit(
        fdtd,
        args,
        setup["edge_local_12p5nm_contract"],
        pre_run,
    )
    output = Path(args.output_dir)
    write_json(output / "EDGE_LOCAL_12P5NM_RUNSETUP_AUDIT.json", audit)
    np.savez_compressed(
        output / "edge_local_12p5nm_runsetup_mesh_coordinates.npz",
        **{f"solver_{axis}_m": values for axis, values in coordinates.items()},
        metadata_json=np.asarray(json.dumps(audit)),
    )
    pre_run["edge_local_12p5nm_runsetup_audit"] = audit
    if not audit["gates"]["all_before_FDTD"]:
        failed = [
            key for key, value in audit["gates"].items() if not value
        ]
        raise RuntimeError(
            "edge-local 12.5-nm preflight failed before FDTD: "
            f"{failed}; see EDGE_LOCAL_12P5NM_RUNSETUP_AUDIT.json"
        )
    return pre_run


def main() -> int:
    production.parse_args = parse_args
    production.add_geometry_and_monitors = add_geometry_and_monitors
    production.assert_contract = assert_contract
    return int(production.main())


if __name__ == "__main__":
    raise SystemExit(main())
