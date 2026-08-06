#!/usr/bin/env python3
"""Deposit material-attributed native Q on the Run-002 3D thermal grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.finite_inverse_design.finite_q_mapping import (  # noqa: E402
    apply_material_intersection_density_separable,
    nodal_control_volume_edges,
)


MATERIALS = ("Si", "bottom_SiO2", "physical_TaIrTe4", "design_effective_SiO2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def growing_positions(length_m: float, initial_m: float, maximum_m: float, growth: float) -> np.ndarray:
    values = [0.0]
    step = initial_m
    while values[-1] + step < length_m:
        values.append(values[-1] + step)
        step = min(maximum_m, step * growth)
    values.append(length_m)
    return np.unique(np.asarray(values))


def thermal_edges() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    core = np.linspace(-16e-6, 16e-6, 321)
    outer = growing_positions(16e-6, 100e-9, 1e-6, 1.45)[1:]
    lateral = np.concatenate((-16e-6 - outer[::-1], core, 16e-6 + outer))
    si_top = -0.385e-6
    si = si_top - growing_positions(20e-6, 25e-9, 0.5e-6, 1.35)[::-1]
    oxide = np.linspace(-0.385e-6, -0.100e-6, 20)
    flake = np.linspace(-0.100e-6, 0.0, 5)
    design = np.linspace(0.0, 1.0e-6, 21)
    z = np.concatenate((si, oxide[1:], flake[1:], design[1:]))
    if any(np.any(np.diff(axis) <= 0.0) for axis in (lateral, lateral, z)):
        raise RuntimeError("thermal grid is not strictly increasing")
    return lateral, lateral.copy(), z


def material_masks(edges: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict[str, np.ndarray]:
    centers = tuple(0.5 * (axis[:-1] + axis[1:]) for axis in edges)
    x, y, z = centers
    shape = (x.size, y.size, z.size)
    xy_all = np.ones((x.size, y.size), bool)
    flake_xy = (np.abs(x[:, None]) < 16e-6) & (np.abs(y[None, :]) < 16e-6)
    design_xy = (np.abs(x[:, None]) < 10e-6) & (np.abs(y[None, :]) < 10e-6)
    masks = {
        "Si": xy_all[:, :, None] & (z[None, None, :] < -0.385e-6),
        "bottom_SiO2": xy_all[:, :, None] & ((z[None, None, :] > -0.385e-6) & (z[None, None, :] < -0.100e-6)),
        "physical_TaIrTe4": flake_xy[:, :, None] & ((z[None, None, :] > -0.100e-6) & (z[None, None, :] < 0.0)),
        "design_effective_SiO2": design_xy[:, :, None] & ((z[None, None, :] > 0.0) & (z[None, None, :] < 1.0e-6)),
    }
    for name, mask in masks.items():
        if mask.shape != shape or not np.any(mask):
            raise RuntimeError(f"invalid material mask {name}")
    occupied = sum(mask.astype(np.uint8) for mask in masks.values())
    if np.max(occupied) != 1:
        raise RuntimeError("thermal material masks overlap")
    return masks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-q", required=True, type=Path)
    parser.add_argument("--native-q-sha256", required=True)
    parser.add_argument("--attribution-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    q_path = args.native_q.expanduser().resolve()
    if sha256(q_path) != args.native_q_sha256:
        raise RuntimeError("native-Q SHA mismatch")
    attribution_path = args.attribution_json.expanduser().resolve()
    attribution = json.loads(attribution_path.read_text())
    if not attribution.get("passed", False):
        raise RuntimeError("input attribution gate did not pass")
    native = np.load(q_path)
    target_edges = thermal_edges()
    masks = material_masks(target_edges)
    shape = tuple(axis.size - 1 for axis in target_edges)
    material_sources = {name: np.zeros(shape, float) for name in MATERIALS}
    mapping_records: dict[str, object] = {}
    for component in "xyz":
        density = np.asarray(native[f"Q{component}_W_m3"], float)
        source_edges = tuple(
            nodal_control_volume_edges(np.asarray(native[f"Q{component}_{axis}_m"], float))
            for axis in "xyz"
        )
        mapping_records[component] = {}
        for name in MATERIALS:
            mapped, _, metrics = apply_material_intersection_density_separable(
                source_density=density,
                source_edges_m=source_edges,
                target_edges_m=target_edges,
                target_material_support_mask=masks[name],
            )
            material_sources[name] += mapped
            expected = attribution["component_records"][component]["material_intersection_power_W"][name]
            relative = abs(metrics["target_integrated_power_W"] - expected) / max(abs(expected), 1e-300)
            mapping_records[component][name] = {**metrics, "expected_attribution_power_W": expected, "relative_to_attribution": relative}
    total_source = sum(material_sources.values())
    widths = tuple(np.diff(axis) for axis in target_edges)
    volume = widths[0][:, None, None] * widths[1][None, :, None] * widths[2][None, None, :]
    material_power = {name: float(np.sum(material_sources[name] * volume)) for name in MATERIALS}
    total_power = float(np.sum(total_source * volume))
    expected_total = attribution["power_W"]["physical_thermal_source"]
    worst_mapping_error = max(record[name]["relative_power_error"] for record in mapping_records.values() for name in MATERIALS)
    worst_attribution_error = max(record[name]["relative_to_attribution"] for record in mapping_records.values() for name in MATERIALS)
    outside = sum(int(np.count_nonzero(material_sources[name][~masks[name]])) for name in MATERIALS)
    npz_path = output / "production_thermal_grid_material_q.npz"
    np.savez_compressed(
        npz_path,
        x_edges_m=target_edges[0],
        y_edges_m=target_edges[1],
        z_edges_m=target_edges[2],
        Q_total_W_m3=total_source,
        **{f"Q_{name}_W_m3": value for name, value in material_sources.items()},
        **{f"mask_{name}": value for name, value in masks.items()},
    )
    passed = bool(
        worst_mapping_error < 1e-12
        and worst_attribution_error < 1e-12
        and abs(total_power - expected_total) / max(abs(expected_total), 1e-300) < 1e-12
        and outside == 0
        and np.all(np.isfinite(total_source))
        and np.min(total_source) >= 0.0
    )
    result = {
        "status": "VALIDATED_PRODUCTION_3D_THERMAL_Q_DEPOSITION" if passed else "FAILED_PRODUCTION_3D_THERMAL_Q_DEPOSITION",
        "passed": passed,
        "thermal_grid": {
            "shape_xyz": list(shape),
            "domain_bounds_m": {axis: [float(edge[0]), float(edge[-1])] for axis, edge in zip("xyz", target_edges)},
            "minimum_step_m": {axis: float(np.min(np.diff(edge))) for axis, edge in zip("xyz", target_edges)},
            "maximum_step_m": {axis: float(np.max(np.diff(edge))) for axis, edge in zip("xyz", target_edges)},
            "finite_flake_span_m": 32e-6,
            "design_span_m": 20e-6,
            "si_depth_m": 20e-6,
        },
        "power_W": {"total_mapped": total_power, "expected_material_attributed": expected_total, **material_power},
        "gates": {
            "worst_internal_mapping_power_error": worst_mapping_error,
            "worst_relative_to_attribution": worst_attribution_error,
            "total_relative_to_attribution": abs(total_power - expected_total) / max(abs(expected_total), 1e-300),
            "nonzero_cells_outside_own_material": outside,
        },
        "component_material_records": mapping_records,
        "artifact": {"path": str(npz_path), "size_bytes": npz_path.stat().st_size, "sha256": sha256(npz_path)},
        "method": {
            "exact_cartesian_material_intersection": True,
            "full_cell_power_forced_into_material": False,
            "nearest_material_relocation": False,
            "clipping_smoothing_gain_or_rescaling": False,
            "thermal_solve": False,
            "adjoint_solve": False,
            "optimization_iterations": 0,
        },
    }
    result_path = output / "production_thermal_q_deposition_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
