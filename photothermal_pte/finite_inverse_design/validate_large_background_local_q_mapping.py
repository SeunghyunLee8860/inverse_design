#!/usr/bin/env python3
"""Validate local native-Yee Q embedding and its exact transpose."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from .explicit_thermal import build_explicit_geometry
from .finite_q_mapping import (
    build_conservative_embedding_remap,
    exact_nonzero_box,
    nodal_control_volume_edges,
)
from .native_yee_q import extract_native_yee_q
from .probe_v261_cpu_tfsf_device import PABS_FIELD, PABS_INDEX
from .probe_v261_gpu_plane_wave_roi import load_lumapi
from .run_v261_large_background_mixed_optical_adfd import WAVELENGTH_M


STATUS_PASS = "VALIDATED_LOCAL_Q_OPTICAL_THERMAL_MAPPING"
STATUS_FAIL = "FAILED_LOCAL_Q_OPTICAL_THERMAL_MAPPING"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thermal-cell-size-nm", type=float, default=100.0)
    parser.add_argument("--thermal-domain-um", type=float, default=32.0)
    parser.add_argument("--thermal-si-depth-um", type=float, default=20.0)
    parser.add_argument("--thermal-flake-span-um", type=float, default=4.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    project = Path(args.project).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "local_q_mapping_summary.json"
    result: dict[str, object] = {
        "status": "BLOCKED_LOCAL_Q_MAPPING_NOT_RUN",
        "passed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "local Omega_Q numerical certificate only; not a complete "
            "physical plane-wave illumination or final thermal source"
        ),
        "forbidden_and_absent": [
            "source clipping",
            "nonzero source deletion",
            "smoothing",
            "gain",
            "global rescaling",
            "periodic tiling",
        ],
        "thermal_run": False,
        "pte_run": False,
        "optimization_run": False,
    }
    fdtd = None
    started = time.monotonic()
    try:
        cell_size_m = args.thermal_cell_size_nm * 1.0e-9
        design_cells = int(round(2.0e-6 / cell_size_m))
        if not np.isclose(design_cells * cell_size_m, 2.0e-6):
            raise ValueError("thermal cell size must divide the design span")
        rho = np.full((design_cells, design_cells), 0.5)
        geometry = build_explicit_geometry(
            rho,
            lateral_domain_m=args.thermal_domain_um * 1.0e-6,
            si_depth_m=args.thermal_si_depth_um * 1.0e-6,
            flake_span_m=args.thermal_flake_span_um * 1.0e-6,
            cell_size_m=cell_size_m,
        )
        target_edges = (
            geometry.x_edges_m,
            geometry.y_edges_m,
            geometry.z_edges_m,
        )
        target_shape = tuple(axis.size - 1 for axis in target_edges)

        lumapi = load_lumapi()
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        fdtd.load(str(project))
        native = extract_native_yee_q(
            fdtd,
            field_monitor=PABS_FIELD,
            index_monitor=PABS_INDEX,
            wavelength_m=WAVELENGTH_M,
        )
        mapped_total = np.zeros(target_shape, float)
        component_records = {}
        active_data = {}
        remaps = {}
        for component in "xyz":
            density = np.asarray(
                native["Q_components"][component], float
            )
            nonzero_count = int(np.count_nonzero(density))
            if nonzero_count == 0:
                component_records[component] = {
                    "nonzero_count": 0,
                    "native_power_W": native["component_power_W"][
                        component
                    ],
                    "mapped_power_W": 0.0,
                    "skipped_exact_zero_component": True,
                }
                continue
            box, outside_nonzero = exact_nonzero_box(density)
            if outside_nonzero != 0:
                raise RuntimeError("nonzero Q exists outside selected box")
            component_edges = tuple(
                nodal_control_volume_edges(
                    native["native_coordinates"][component][axis]
                )[section.start : section.stop + 1]
                for axis, section in zip("xyz", box)
            )
            source = density[box]
            remap = build_conservative_embedding_remap(
                source_edges_m=component_edges,
                target_edges_m=target_edges,
            )
            mapped = remap.apply(source)
            source_power = remap.power_source(source)
            mapped_power = remap.power_target(mapped)
            native_power = float(
                native["component_power_W"][component]
            )
            component_records[component] = {
                "nonzero_count": nonzero_count,
                "selected_box": [
                    [section.start, section.stop] for section in box
                ],
                "outside_selected_box_nonzero_count": outside_nonzero,
                "source_edges_bounds_m": {
                    axis: [float(edges[0]), float(edges[-1])]
                    for axis, edges in zip("xyz", component_edges)
                },
                "native_power_W": native_power,
                "selected_source_power_W": source_power,
                "mapped_power_W": mapped_power,
                "source_vs_native_relative_error": abs(
                    source_power - native_power
                )
                / max(abs(native_power), np.finfo(float).tiny),
                "mapped_vs_native_relative_error": abs(
                    mapped_power - native_power
                )
                / max(abs(native_power), np.finfo(float).tiny),
                "skipped_exact_zero_component": False,
            }
            mapped_total += mapped
            active_data[component] = source
            remaps[component] = remap

        target_volume = (
            np.diff(target_edges[0])[:, None, None]
            * np.diff(target_edges[1])[None, :, None]
            * np.diff(target_edges[2])[None, None, :]
        )
        mapped_total_power = float(np.sum(target_volume * mapped_total))
        native_total_power = float(native["P_Q_W"])
        mapping_error = abs(
            mapped_total_power - native_total_power
        ) / max(abs(native_total_power), np.finfo(float).tiny)

        rng = np.random.default_rng(2026072703)
        target_sensitivity = rng.normal(size=target_shape)
        left = float(np.sum(target_sensitivity * mapped_total))
        right = 0.0
        for component, source in active_data.items():
            right += float(
                np.sum(
                    remaps[component].transpose(target_sensitivity)
                    * source
                )
            )
        transpose_error = abs(left - right) / max(
            abs(left), abs(right), np.finfo(float).tiny
        )

        raw_path = output / "local_q_thermal_mapping.npz"
        np.savez_compressed(
            raw_path,
            Q_thermal_W_m3=mapped_total,
            thermal_x_edges_m=target_edges[0],
            thermal_y_edges_m=target_edges[1],
            thermal_z_edges_m=target_edges[2],
            thermal_material_id=geometry.material_id,
            native_Qx_W_m3=native["Q_components"]["x"],
            native_Qy_W_m3=native["Q_components"]["y"],
            native_Qz_W_m3=native["Q_components"]["z"],
            native_x_m=native["base_coordinates"]["x"],
            native_y_m=native["base_coordinates"]["y"],
            native_z_m=native["base_coordinates"]["z"],
            native_delta_x_m=native["delta_coordinates"]["x"],
            native_delta_y_m=native["delta_coordinates"]["y"],
            native_delta_z_m=native["delta_coordinates"]["z"],
        )
        result.update(
            {
                "status": (
                    STATUS_PASS
                    if mapping_error < 5.0e-3
                    and transpose_error < 1.0e-12
                    else STATUS_FAIL
                ),
                "passed": bool(
                    mapping_error < 5.0e-3
                    and transpose_error < 1.0e-12
                ),
                "optical_project": {
                    "path": str(project),
                    "byte_size": project.stat().st_size,
                    "sha256": sha256(project),
                },
                "omega_Q_bounds_m": {
                    "x": [-1.15e-6, 1.15e-6],
                    "y": [-1.15e-6, 1.15e-6],
                    "z": [-0.15e-6, 0.75e-6],
                },
                "native_P_Q_W": native_total_power,
                "mapped_thermal_P_Q_W": mapped_total_power,
                "mapping_relative_power_error": mapping_error,
                "mapping_power_error_limit": 5.0e-3,
                "transpose_dot_test": {
                    "left": left,
                    "right": right,
                    "relative_error": transpose_error,
                    "limit": 1.0e-12,
                },
                "components": component_records,
                "thermal_target": {
                    "shape": list(target_shape),
                    "lateral_domain_m": geometry.lateral_domain_m,
                    "si_depth_m": geometry.si_depth_m,
                    "flake_span_m": geometry.flake_span_m,
                    "core_cell_size_m": geometry.cell_size_m,
                    "embedding_note": (
                        "target-only cells receive exact zero; every nonzero "
                        "native source control volume is covered once"
                    ),
                },
                "raw_artifact": {
                    "path": str(raw_path),
                    "byte_size": raw_path.stat().st_size,
                    "sha256": sha256(raw_path),
                },
            }
        )
    except Exception as exc:
        result["status"] = "BLOCKED_LOCAL_Q_MAPPING_EXECUTION"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        result["wall_s"] = time.monotonic() - started
        if fdtd is not None:
            fdtd.close()
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    return 0 if result.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
