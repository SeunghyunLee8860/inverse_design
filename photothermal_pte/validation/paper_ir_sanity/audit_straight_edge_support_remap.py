#!/usr/bin/env python3
"""Audit straight-edge Q support projection without another Maxwell solve.

The audit compares the historical sequential x/y/z/x projection, its
x/y-reflected y/x/z/y counterpart, and the coordinate-order-free nearest
physical-support projection.  It operates on the immutable raw optical NPZ
and writes diagnostics only; it does not solve the thermal problem.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
THERMAL_SCRIPT = HERE / "run_device_a_explicit_thermal_pte.py"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


thermal = load_module("straight_edge_remap_thermal", THERMAL_SCRIPT)

from photothermal_pte.finite_inverse_design.finite_q_mapping import (  # noqa: E402
    build_conservative_embedding_remap,
    nodal_control_volume_edges,
    project_remap_to_material_support_along_axis,
    project_remap_to_nearest_material_support,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optical-case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thermal-domain-um", type=float, default=48.0)
    parser.add_argument("--si-depth-um", type=float, default=20.0)
    parser.add_argument("--core-step-nm", type=float, default=100.0)
    parser.add_argument("--flake-dz-nm", type=float, default=26.0)
    return parser.parse_args()


def project_axis_sequence(
    base: Any,
    *,
    edges: tuple[np.ndarray, np.ndarray, np.ndarray],
    support: np.ndarray,
    order: tuple[int, ...],
) -> Any:
    remap = base
    for axis in order:
        remap = project_remap_to_material_support_along_axis(
            remap,
            target_edges_m=edges,
            target_support_mask=support,
            axis=axis,
        )
    return remap


def integrated_metrics(
    q: np.ndarray,
    volume: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> dict[str, float]:
    energy_xy = np.sum(q * volume, axis=2)
    total = float(np.sum(energy_xy))
    if total <= 0.0:
        raise RuntimeError("nonpositive integrated source in remap audit")
    xx, yy = np.meshgrid(x_m, y_m, indexing="ij")
    return {
        "power_W": total,
        "centroid_x_m": float(np.sum(xx * energy_xy) / total),
        "centroid_y_m": float(np.sum(yy * energy_xy) / total),
        "centroid_edge_normal_m": float(
            np.sum((yy - xx) / np.sqrt(2.0) * energy_xy) / total
        ),
        "centroid_edge_tangent_m": float(
            np.sum((xx + yy) / np.sqrt(2.0) * energy_xy) / total
        ),
    }


def relative_l1(
    first: np.ndarray,
    second: np.ndarray,
    volume: np.ndarray,
) -> float:
    numerator = float(np.sum(np.abs(first - second) * volume))
    denominator = float(np.sum(np.abs(first) * volume))
    return numerator / max(denominator, np.finfo(float).tiny)


def edge_profile(
    q: np.ndarray,
    volume: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    bin_width_m: float,
    tangent_window_m: float = 5.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    energy_xy = np.sum(q * volume, axis=2)
    area = np.sum(volume, axis=2)
    xx, yy = np.meshgrid(x_m, y_m, indexing="ij")
    normal = (yy - xx) / np.sqrt(2.0)
    tangent = (xx + yy) / np.sqrt(2.0)
    selected = np.abs(tangent) <= tangent_window_m
    lower = np.floor(np.min(normal[selected]) / bin_width_m) * bin_width_m
    upper = np.ceil(np.max(normal[selected]) / bin_width_m) * bin_width_m
    edges = np.arange(lower, upper + 1.01 * bin_width_m, bin_width_m)
    index = np.digitize(normal[selected], edges) - 1
    valid = (index >= 0) & (index < edges.size - 1)
    numerator = np.bincount(
        index[valid],
        weights=energy_xy[selected][valid],
        minlength=edges.size - 1,
    )
    denominator = np.bincount(
        index[valid],
        weights=area[selected][valid],
        minlength=edges.size - 1,
    )
    profile = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0.0,
    )
    return 0.5 * (edges[:-1] + edges[1:]), profile


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outer_um = 0.5 * args.thermal_domain_um + 1.0
    thermal.FLAKE_VERTICES_UM = np.asarray(
        [
            [-outer_um, -outer_um],
            [outer_um, -outer_um],
            [outer_um, outer_um],
        ],
        float,
    )
    geometry = thermal.build_geometry(
        domain_m=args.thermal_domain_um * 1.0e-6,
        si_depth_m=args.si_depth_um * 1.0e-6,
        core_step_m=args.core_step_nm * 1.0e-9,
        flake_dz_m=args.flake_dz_nm * 1.0e-9,
    )
    artifact = args.optical_case_dir / "finite_q_on_artifact.npz"
    with np.load(artifact, allow_pickle=False) as raw:
        source_axes = tuple(
            np.asarray(raw[name], float) for name in ("x_m", "y_m", "z_m")
        )
        q = np.asarray(raw["Q_on_W_m3"], float)
    source_edges = tuple(nodal_control_volume_edges(axis) for axis in source_axes)
    target_edges = (
        geometry.x_edges_m,
        geometry.y_edges_m,
        geometry.z_edges_m,
    )
    base = build_conservative_embedding_remap(
        source_edges_m=source_edges,
        target_edges_m=target_edges,
    )
    remaps = {
        "historical_x_y_z_x": project_axis_sequence(
            base,
            edges=target_edges,
            support=geometry.flake_mask,
            order=(0, 1, 2, 0),
        ),
        "reflected_y_x_z_y": project_axis_sequence(
            base,
            edges=target_edges,
            support=geometry.flake_mask,
            order=(1, 0, 2, 1),
        ),
        "physical_nearest_support": project_remap_to_nearest_material_support(
            base,
            target_edges_m=target_edges,
            target_support_mask=geometry.flake_mask,
        ),
    }
    mapped = {name: remap.apply(q) for name, remap in remaps.items()}
    volume = base.target_volume_m3
    x = 0.5 * (geometry.x_edges_m[:-1] + geometry.x_edges_m[1:])
    y = 0.5 * (geometry.y_edges_m[:-1] + geometry.y_edges_m[1:])
    source_power = base.power_source(q)
    base_q = base.apply(q)
    outside_before = float(
        np.sum(base_q[~geometry.flake_mask] * volume[~geometry.flake_mask])
    )
    metrics = {
        name: {
            **integrated_metrics(values, volume, x, y),
            "outside_support_power_W": float(
                np.sum(
                    values[~geometry.flake_mask]
                    * volume[~geometry.flake_mask]
                )
            ),
            "power_error_relative": abs(
                remaps[name].power_target(values) - source_power
            )
            / abs(source_power),
        }
        for name, values in mapped.items()
    }
    pairwise = {}
    names = list(mapped)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            pairwise[f"{left}__vs__{right}"] = relative_l1(
                mapped[left],
                mapped[right],
                volume,
            )

    profiles = {}
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for name, values in mapped.items():
        position, profile = edge_profile(
            values,
            volume,
            x,
            y,
            bin_width_m=args.core_step_nm * 1.0e-9,
        )
        finite = np.isfinite(profile)
        position = position[finite]
        profile = profile[finite]
        profiles[name] = {
            "normal_coordinate_m": position.tolist(),
            "areal_Q_W_m2": profile.tolist(),
        }
        axis.plot(position * 1.0e6, profile, label=name)
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("edge-normal coordinate n=(y-x)/sqrt(2) (µm)")
    axis.set_ylabel("tangent-window mean areal Q (W/m²)")
    axis.set_title("Same raw Maxwell Q under three support projections")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.savefig(args.output_dir / "support_remap_edge_profiles.png", dpi=220)
    plt.close(fig)

    summary = {
        "status": (
            "VALIDATED_COORDINATE_ORDER_FREE_SUPPORT_REMAP"
            if (
                metrics["physical_nearest_support"]["power_error_relative"]
                < 5.0e-13
                and abs(
                    metrics["physical_nearest_support"][
                        "outside_support_power_W"
                    ]
                )
                < 1.0e-24
                and pairwise[
                    "historical_x_y_z_x__vs__reflected_y_x_z_y"
                ]
                > 0.0
            )
            else "FAILED_COORDINATE_ORDER_FREE_SUPPORT_REMAP"
        ),
        "raw_optical_artifact": {
            "path": str(artifact.resolve()),
            "size_bytes": artifact.stat().st_size,
            "sha256": sha256(artifact),
        },
        "geometry": {
            "thermal_domain_um": args.thermal_domain_um,
            "si_depth_um": args.si_depth_um,
            "core_step_nm": args.core_step_nm,
            "flake_dz_nm": args.flake_dz_nm,
            "shape": list(geometry.flake_mask.shape),
            "support": "straight 45-degree half-plane TaIrTe4, y<=x",
        },
        "source_power_W": source_power,
        "unprojected_outside_support_power_W": outside_before,
        "unprojected_outside_support_fraction": outside_before / source_power,
        "per_operator": metrics,
        "pairwise_energy_weighted_relative_L1": pairwise,
        "interpretation": (
            "The historical order comparison is a structural remap audit, "
            "not an estimate that the former thermal result had the same "
            "relative error. The physical-nearest operator has no coordinate "
            "axis sequence and splits exact physical-distance ties."
        ),
        "profiles": profiles,
        "no_new_Maxwell_solve": True,
        "no_thermal_solve": True,
    }
    (args.output_dir / "support_remap_audit.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "profiles"}, indent=2))
    return 0 if summary["status"].startswith("VALIDATED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
