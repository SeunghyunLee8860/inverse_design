#!/usr/bin/env python3
"""Regenerate explicit-3D maps from saved fields without running a solver."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    compare_w12_50nm_maxwell_analytic_explicit3d as comparison,
)


CASES = ("Maxwell_a", "Maxwell_b", "analytic_a", "analytic_b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.input_npz, allow_pickle=False) as raw:
        geometry = SimpleNamespace(
            x_edges_m=np.asarray(raw["x_edges_m"], float),
            y_edges_m=np.asarray(raw["y_edges_m"], float),
            z_edges_m=np.asarray(raw["z_edges_m"], float),
            flake_mask=np.asarray(raw["flake_mask"], bool),
        )
        field_names = (
            "temperature_surface_K",
            "temperature_midplane_K",
            "temperature_thickness_average_K",
            "grad_b_K_m",
            "grad_a_K_m",
            "grad_n_K_m",
            "grad_t_K_m",
            "grad_magnitude_K_m",
        )
        fields = {
            case: {
                name: np.asarray(raw[f"{case}__{name}"], float)
                for name in field_names
            }
            for case in CASES
        }

    flake_z = np.flatnonzero(np.any(geometry.flake_mask, axis=(0, 1)))
    z_centers = 0.5 * (
        geometry.z_edges_m[flake_z] + geometry.z_edges_m[flake_z + 1]
    )
    mid_index = int(flake_z[np.argmin(np.abs(z_centers + 65.0e-9))])
    q_maps = {}
    with np.load(args.input_npz, allow_pickle=False) as raw:
        for case in CASES:
            q_maps[case] = {
                "Q_areal_W_m2": np.asarray(raw[f"{case}__Q_areal_W_m2"], float),
                "Q_midplane_W_m3": np.asarray(
                    raw[f"{case}__Q_total_W_m3"][:, :, mid_index], float
                ),
            }
    maps = (
        ("explicit3d_depth_integrated_Q.png", q_maps, "Q_areal_W_m2", "q'' (W/m²)"),
        ("explicit3d_volumetric_Q_midplane.png", q_maps, "Q_midplane_W_m3", "Q (W/m³)"),
        ("explicit3d_temperature_surface.png", fields, "temperature_surface_K", "ΔT (K)"),
        ("explicit3d_temperature_midplane.png", fields, "temperature_midplane_K", "ΔT (K)"),
        ("explicit3d_gradient_magnitude.png", fields, "grad_magnitude_K_m", "|∇T| (K/m)"),
    )
    for filename, arrays, key, label in maps:
        comparison.map_figure(args.report_dir / filename, geometry, arrays, key, label)
    comparison.gradient_figure(
        args.report_dir / "explicit3d_Maxwell_gradients.png",
        geometry,
        fields,
        "Maxwell",
    )
    comparison.gradient_figure(
        args.report_dir / "explicit3d_analytic_gradients.png",
        geometry,
        fields,
        "analytic",
    )
    print(f"regenerated 7 coordinate-faithful figures in {args.report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
