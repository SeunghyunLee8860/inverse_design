#!/usr/bin/env python3
"""Audit independently controlled explicit-thermal grid dimensions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .explicit_thermal import build_explicit_geometry


STATUS = "AUDITED_EXPLICIT_THERMAL_INDEPENDENT_GRID_PARAMETERS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def grid_record(geometry) -> dict[str, object]:
    z = 0.5 * (geometry.z_edges_m[:-1] + geometry.z_edges_m[1:])
    return {
        "shape": list(geometry.material_id.shape),
        "total_cells": int(geometry.material_id.size),
        "core_xy_cell_size_m": geometry.core_xy_cell_size_m,
        "flake_dz_m": geometry.flake_dz_m,
        "design_dz_m": geometry.design_dz_m,
        "flake_z_cell_count": int(
            np.count_nonzero((z > -100.0e-9) & (z < 0.0))
        ),
        "design_z_cell_count": int(
            np.count_nonzero((z > 0.0) & (z < 600.0e-9))
        ),
        "x_bounds_m": [
            float(geometry.x_edges_m[0]),
            float(geometry.x_edges_m[-1]),
        ],
        "y_bounds_m": [
            float(geometry.y_edges_m[0]),
            float(geometry.y_edges_m[-1]),
        ],
        "z_key_faces_m": [-385.0e-9, -100.0e-9, 0.0, 600.0e-9],
    }


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rho_100 = np.full((20, 20), 0.5)
    legacy = build_explicit_geometry(
        rho_100,
        lateral_domain_m=32.0e-6,
        si_depth_m=20.0e-6,
        flake_span_m=4.0e-6,
        cell_size_m=100.0e-9,
    )
    explicit = build_explicit_geometry(
        rho_100,
        lateral_domain_m=32.0e-6,
        si_depth_m=20.0e-6,
        flake_span_m=4.0e-6,
        core_xy_cell_size_m=100.0e-9,
        flake_dz_m=25.0e-9,
        design_dz_m=100.0e-9,
    )
    baseline_equal = bool(
        np.array_equal(legacy.x_edges_m, explicit.x_edges_m)
        and np.array_equal(legacy.y_edges_m, explicit.y_edges_m)
        and np.array_equal(legacy.z_edges_m, explicit.z_edges_m)
        and np.array_equal(legacy.material_id, explicit.material_id)
        and np.array_equal(
            legacy.interface_resistance_m2K_W["z"],
            explicit.interface_resistance_m2K_W["z"],
        )
        and np.array_equal(legacy.kappa_W_mK, explicit.kappa_W_mK)
    )
    xy_refined = build_explicit_geometry(
        np.full((40, 40), 0.5),
        lateral_domain_m=32.0e-6,
        si_depth_m=20.0e-6,
        flake_span_m=4.0e-6,
        core_xy_cell_size_m=50.0e-9,
        flake_dz_m=25.0e-9,
        design_dz_m=100.0e-9,
    )
    flake_z_refined = build_explicit_geometry(
        rho_100,
        lateral_domain_m=32.0e-6,
        si_depth_m=20.0e-6,
        flake_span_m=4.0e-6,
        core_xy_cell_size_m=100.0e-9,
        flake_dz_m=12.5e-9,
        design_dz_m=100.0e-9,
    )
    design_z_refined = build_explicit_geometry(
        rho_100,
        lateral_domain_m=32.0e-6,
        si_depth_m=20.0e-6,
        flake_span_m=4.0e-6,
        core_xy_cell_size_m=100.0e-9,
        flake_dz_m=25.0e-9,
        design_dz_m=50.0e-9,
    )
    records = {
        "baseline_legacy": grid_record(legacy),
        "baseline_explicit": grid_record(explicit),
        "xy_refined_only": grid_record(xy_refined),
        "flake_z_refined_only": grid_record(flake_z_refined),
        "design_z_refined_only": grid_record(design_z_refined),
    }
    passed = bool(
        baseline_equal
        and records["xy_refined_only"]["flake_z_cell_count"] == 4
        and records["xy_refined_only"]["design_z_cell_count"] == 6
        and records["flake_z_refined_only"]["flake_z_cell_count"] == 8
        and records["flake_z_refined_only"]["design_z_cell_count"] == 6
        and records["design_z_refined_only"]["flake_z_cell_count"] == 4
        and records["design_z_refined_only"]["design_z_cell_count"] == 12
    )
    result = {
        "status": STATUS if passed else "FAILED_EXPLICIT_THERMAL_GRID_AUDIT",
        "passed": passed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "legacy_baseline_bitwise_equal": baseline_equal,
        "parameters": [
            "core_xy_cell_size_m",
            "flake_dz_m",
            "design_dz_m",
        ],
        "one_at_a_time_variants": records,
        "solver_run": False,
        "optimization_run": False,
    }
    (output / "explicit_thermal_grid_contract_audit_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    report = f"""# Explicit thermal independent-grid contract audit

Status: `{result["status"]}`

- legacy 100 nm baseline versus explicit parameter baseline, bitwise grid,
  material, kappa, and z-interface equality: `{baseline_equal}`;
- `core_xy_cell_size_m`: independently changes lateral core cells;
- `flake_dz_m`: independently changes TaIrTe4 z cells;
- `design_dz_m`: independently changes design extrusion z cells.

Realized TaIrTe4/design z-cell counts:

- baseline: `4 / 6`;
- xy-only refinement: `4 / 6`;
- TaIrTe4-z-only refinement: `8 / 6`;
- design-z-only refinement: `4 / 12`.

This is a geometry/operator-parameter audit. No thermal solve or optimization
was run, and no convergence status is claimed here.
"""
    (output / "EXPLICIT_THERMAL_GRID_CONTRACT_AUDIT.md").write_text(report)
    print(json.dumps(result, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
