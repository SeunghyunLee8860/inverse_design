#!/usr/bin/env python3
"""Audit non-periodic Yee integration weights on the finite design support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .probe_v261_gpu_plane_wave_roi import load_lumapi
from .nonperiodic_yee_metric import (
    clipped_component_yee_volumes,
    clipped_voronoi_weights,
)
from .run_v261_large_background_mixed_optical_adfd import (
    DESIGN_FIELD,
    FIELD_REGION,
    SIO2_EPSILON,
    PABS_FIELD,
    PABS_INDEX,
    absorption_objective_and_source,
    component_volumes,
    fieldregion_profile,
    gradient_from_adjoint,
    monitor_electric,
    monitor_epsilon,
)


DESIGN_BOUNDS_M = {
    "x": (-1.0e-6, 1.0e-6),
    "y": (-1.0e-6, 1.0e-6),
    "z": (0.0, 0.6e-6),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--adjoint-project")
    parser.add_argument("--template-project")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = Path(args.project).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    fdtd = None
    try:
        lumapi = load_lumapi()
        fdtd = lumapi.FDTD(hide=True, serverArgs={"platform": "offscreen"})
        fdtd.load(str(project))
        forward_electric, grid = monitor_electric(fdtd, DESIGN_FIELD)
        current = component_volumes(grid)
        clipped = clipped_component_yee_volumes(grid, DESIGN_BOUNDS_M)
        physical_volume = np.prod(
            [upper - lower for lower, upper in DESIGN_BOUNDS_M.values()]
        )
        result = {
            "project": str(project),
            "design_bounds_m": DESIGN_BOUNDS_M,
            "physical_volume_m3": physical_volume,
            "base_coordinates": {
                axis: {
                    "count": int(grid[axis].size),
                    "minimum_m": float(np.min(grid[axis])),
                    "maximum_m": float(np.max(grid[axis])),
                    "first_three_m": grid[axis][:3].tolist(),
                    "last_three_m": grid[axis][-3:].tolist(),
                }
                for axis in "xyz"
            },
            "delta": {
                axis: {
                    "minimum_m": float(np.min(grid[f"delta_{axis}"])),
                    "maximum_m": float(np.max(grid[f"delta_{axis}"])),
                    "first_three_m": grid[f"delta_{axis}"][:3].tolist(),
                    "last_three_m": grid[f"delta_{axis}"][-3:].tolist(),
                }
                for axis in "xyz"
            },
            "components": {},
        }
        for component in range(3):
            component_coordinates = {}
            for axis_index, axis in enumerate("xyz"):
                coordinate = np.array(grid[axis], copy=True)
                if component == axis_index:
                    coordinate += grid[f"delta_{axis}"]
                axis_weights = clipped_voronoi_weights(
                    coordinate, *DESIGN_BOUNDS_M[axis]
                )
                component_coordinates[axis] = {
                    "minimum_m": float(np.min(coordinate)),
                    "maximum_m": float(np.max(coordinate)),
                    "positive_clipped_weight_count": int(
                        np.count_nonzero(axis_weights)
                    ),
                    "clipped_weight_sum_m": float(
                        np.sum(axis_weights)
                    ),
                }
            current_sum = float(np.sum(current[component]))
            clipped_sum = float(np.sum(clipped[component]))
            result["components"]["xyz"[component]] = {
                "coordinates": component_coordinates,
                "current_trapezoid_volume_m3": current_sum,
                "current_over_physical": current_sum / physical_volume,
                "clipped_voronoi_volume_m3": clipped_sum,
                "clipped_over_physical": clipped_sum / physical_volume,
            }
        if args.adjoint_project and args.template_project:
            pabs_electric, pabs_grid = monitor_electric(fdtd, PABS_FIELD)
            pabs_epsilon, pabs_index_grid = monitor_epsilon(fdtd, PABS_INDEX)
            for axis in "xyzf":
                if not np.array_equal(
                    pabs_grid[axis], pabs_index_grid[axis]
                ):
                    raise RuntimeError(f"pabs E/index {axis} grids differ")
            _, q_source, _ = absorption_objective_and_source(
                pabs_electric, pabs_epsilon, pabs_grid
            )
            _, profile_scale = fieldregion_profile(q_source)

            template = Path(args.template_project).expanduser().resolve()
            fdtd.switchtolayout()
            fdtd.load(str(template))
            base_amplitude = float(
                fdtd.getnamed(FIELD_REGION, "base amplitude")
            )

            adjoint_project = Path(
                args.adjoint_project
            ).expanduser().resolve()
            fdtd.switchtolayout()
            fdtd.load(str(adjoint_project))
            adjoint_electric, adjoint_grid = monitor_electric(
                fdtd, DESIGN_FIELD
            )
            for axis in "xyzf":
                if not np.allclose(
                    grid[axis],
                    adjoint_grid[axis],
                    rtol=0.0,
                    atol=1.0e-15,
                ):
                    raise RuntimeError(
                        f"forward/adjoint design {axis} grids differ"
                    )
            derivative = np.full(
                forward_electric.shape,
                SIO2_EPSILON - 1.0,
                np.complex128,
            )
            clipped_gradient, clipped_components = gradient_from_adjoint(
                forward_electric=forward_electric,
                adjoint_electric=adjoint_electric,
                design_grid=grid,
                d_epsilon_d_rho=derivative,
                profile_scale=profile_scale,
                base_amplitude=base_amplitude,
                design_bounds_m=DESIGN_BOUNDS_M,
            )
            old_components = []
            for component in range(3):
                value = np.sum(
                    (2.0 * 8.8541878128e-12 / base_amplitude)
                    * current[component][..., None]
                    * forward_electric[..., component]
                    * (adjoint_electric[..., component] * profile_scale)
                    * derivative[..., component]
                )
                old_components.append(float(np.real(value)))
            result["saved_field_gradient_audit"] = {
                "adjoint_project": str(adjoint_project),
                "template_project": str(template),
                "profile_scale": profile_scale,
                "base_amplitude": base_amplitude,
                "old_unclipped_gradient_W_per_rho": float(
                    sum(old_components)
                ),
                "old_unclipped_components_xyz_W_per_rho": old_components,
                "clipped_voronoi_gradient_W_per_rho": clipped_gradient,
                "clipped_voronoi_components_xyz_W_per_rho": (
                    clipped_components
                ),
            }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0
    finally:
        if fdtd is not None:
            fdtd.close()


if __name__ == "__main__":
    raise SystemExit(main())
