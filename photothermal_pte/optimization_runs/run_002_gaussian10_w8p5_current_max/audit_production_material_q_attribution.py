#!/usr/bin/env python3
"""Partition native Run-002 Q by literal material-volume intersection.

This is an offline attribution audit.  It does not relocate a full cut-cell
power into the nearest material and does not run Maxwell, thermal, adjoint, or
optimization solvers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


PHYSICAL_FLAKE_HALF_SPAN_M = 16.0e-6
DESIGN_HALF_SPAN_M = 10.0e-6
Z_SI_OXIDE_M = -0.385e-6
Z_OXIDE_FLAKE_M = -0.100e-6
Z_FLAKE_DESIGN_M = 0.0
Z_DESIGN_AIR_M = 1.0e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nodal_edges(coordinate_m: np.ndarray) -> np.ndarray:
    coordinate = np.asarray(coordinate_m, float).reshape(-1)
    if coordinate.size < 2 or np.any(np.diff(coordinate) <= 0.0):
        raise ValueError("native coordinate must be strictly increasing")
    return np.concatenate(
        (coordinate[:1], 0.5 * (coordinate[:-1] + coordinate[1:]), coordinate[-1:])
    )


def overlap_lengths(edges_m: np.ndarray, lower_m: float, upper_m: float) -> np.ndarray:
    lower = np.maximum(edges_m[:-1], float(lower_m))
    upper = np.minimum(edges_m[1:], float(upper_m))
    return np.maximum(upper - lower, 0.0)


def box_power(
    density_W_m3: np.ndarray,
    edges_m: tuple[np.ndarray, np.ndarray, np.ndarray],
    bounds_m: dict[str, tuple[float, float]],
) -> float:
    weights = [
        overlap_lengths(edges_m[index], *bounds_m[axis])
        for index, axis in enumerate("xyz")
    ]
    return float(
        np.einsum(
            "ijk,i,j,k->",
            np.asarray(density_W_m3, float),
            weights[0],
            weights[1],
            weights[2],
            optimize=True,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-result", required=True, type=Path)
    parser.add_argument("--native-q", required=True, type=Path)
    parser.add_argument("--native-q-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    forward_path = args.forward_result.expanduser().resolve()
    q_path = args.native_q.expanduser().resolve()
    if sha256(q_path) != args.native_q_sha256:
        raise RuntimeError("native-Q SHA-256 mismatch")
    forward = json.loads(forward_path.read_text())
    if forward["native_Q_artifact"]["sha256"] != args.native_q_sha256:
        raise RuntimeError("forward-result/native-Q provenance mismatch")
    native = np.load(q_path)

    component_records: dict[str, object] = {}
    summed = {
        "Si": 0.0,
        "bottom_SiO2": 0.0,
        "physical_TaIrTe4": 0.0,
        "design_effective_SiO2": 0.0,
        "artificial_extended_TaIrTe4_outside_physical_flake": 0.0,
        "air_or_nonmaterial_intersection": 0.0,
    }
    for component in "xyz":
        density = np.asarray(native[f"Q{component}_W_m3"], float)
        coordinates = tuple(
            np.asarray(native[f"Q{component}_{axis}_m"], float) for axis in "xyz"
        )
        edges = tuple(nodal_edges(value) for value in coordinates)
        domain = {axis: (edge[0], edge[-1]) for axis, edge in zip("xyz", edges)}
        total = box_power(density, edges, domain)
        full_layer = box_power(
            density,
            edges,
            {
                "x": domain["x"],
                "y": domain["y"],
                "z": (Z_OXIDE_FLAKE_M, Z_FLAKE_DESIGN_M),
            },
        )
        material = {
            "Si": box_power(
                density,
                edges,
                {"x": domain["x"], "y": domain["y"], "z": (domain["z"][0], Z_SI_OXIDE_M)},
            ),
            "bottom_SiO2": box_power(
                density,
                edges,
                {"x": domain["x"], "y": domain["y"], "z": (Z_SI_OXIDE_M, Z_OXIDE_FLAKE_M)},
            ),
            "physical_TaIrTe4": box_power(
                density,
                edges,
                {
                    "x": (-PHYSICAL_FLAKE_HALF_SPAN_M, PHYSICAL_FLAKE_HALF_SPAN_M),
                    "y": (-PHYSICAL_FLAKE_HALF_SPAN_M, PHYSICAL_FLAKE_HALF_SPAN_M),
                    "z": (Z_OXIDE_FLAKE_M, Z_FLAKE_DESIGN_M),
                },
            ),
            "design_effective_SiO2": box_power(
                density,
                edges,
                {
                    "x": (-DESIGN_HALF_SPAN_M, DESIGN_HALF_SPAN_M),
                    "y": (-DESIGN_HALF_SPAN_M, DESIGN_HALF_SPAN_M),
                    "z": (Z_FLAKE_DESIGN_M, Z_DESIGN_AIR_M),
                },
            ),
        }
        material["artificial_extended_TaIrTe4_outside_physical_flake"] = (
            full_layer - material["physical_TaIrTe4"]
        )
        assigned_and_artificial = sum(material.values())
        material["air_or_nonmaterial_intersection"] = total - assigned_and_artificial
        if material["air_or_nonmaterial_intersection"] < -1e-18 * max(total, 1e-300):
            raise RuntimeError("material boxes overlap or exceed component power")
        for name, value in material.items():
            summed[name] += value
        component_records[component] = {
            "native_shape": list(density.shape),
            "dual_cell_bounds_m": {axis: list(domain[axis]) for axis in "xyz"},
            "full_native_power_W": total,
            "material_intersection_power_W": material,
            "partition_identity_error_W": total - sum(material.values()),
            "negative_Q_count": int(np.count_nonzero(density < 0.0)),
            "all_finite": bool(np.all(np.isfinite(density))),
        }

    reintegrated = sum(
        component_records[component]["full_native_power_W"] for component in "xyz"
    )
    physical_thermal_source = sum(
        summed[name]
        for name in ("Si", "bottom_SiO2", "physical_TaIrTe4", "design_effective_SiO2")
    )
    partitioned = sum(summed.values())
    reference = float(forward["P_Q_W"])
    result = {
        "status": "VALIDATED_PRODUCTION_MATERIAL_INTERSECTION_Q_ATTRIBUTION",
        "scope": "offline literal native-dual-cell/material-volume intersection; no relocation or solve",
        "input": {
            "forward_result": str(forward_path),
            "native_Q": str(q_path),
            "native_Q_size_bytes": q_path.stat().st_size,
            "native_Q_sha256": args.native_q_sha256,
        },
        "geometry_m": {
            "physical_flake_xy": [-PHYSICAL_FLAKE_HALF_SPAN_M, PHYSICAL_FLAKE_HALF_SPAN_M],
            "design_xy": [-DESIGN_HALF_SPAN_M, DESIGN_HALF_SPAN_M],
            "z_interfaces": [Z_SI_OXIDE_M, Z_OXIDE_FLAKE_M, Z_FLAKE_DESIGN_M, Z_DESIGN_AIR_M],
        },
        "component_records": component_records,
        "power_W": {
            "forward_P_Q": reference,
            "native_reintegrated": reintegrated,
            "physical_thermal_source": physical_thermal_source,
            **summed,
        },
        "relative": {
            "native_reintegration_error": abs(reintegrated - reference) / max(abs(reference), 1e-300),
            "partition_identity_error": abs(partitioned - reintegrated) / max(abs(reintegrated), 1e-300),
            "physical_thermal_source_fraction_of_full_P_Q": physical_thermal_source / reference,
            "artificial_background_fraction_of_full_P_Q": summed["artificial_extended_TaIrTe4_outside_physical_flake"] / reference,
            "air_or_nonmaterial_fraction_of_full_P_Q": summed["air_or_nonmaterial_intersection"] / reference,
        },
        "method": {
            "cut_cell_rule": "Q_cell times literal material-intersection volume",
            "full_cell_power_forced_into_material": False,
            "nearest_material_relocation": False,
            "clipping_smoothing_gain_or_rescaling": False,
            "Maxwell_solves": 0,
            "thermal_solves": 0,
            "adjoint_solves": 0,
            "optimization_iterations": 0,
        },
    }
    gates = (
        result["relative"]["native_reintegration_error"] < 5e-12
        and result["relative"]["partition_identity_error"] < 5e-12
        and all(record["all_finite"] for record in component_records.values())
        and all(record["negative_Q_count"] == 0 for record in component_records.values())
    )
    result["passed"] = bool(gates)
    if not gates:
        result["status"] = "FAILED_PRODUCTION_MATERIAL_INTERSECTION_Q_ATTRIBUTION"
    result_path = output / "production_material_q_attribution.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if gates else 1


if __name__ == "__main__":
    raise SystemExit(main())
