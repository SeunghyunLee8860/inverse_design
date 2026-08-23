#!/usr/bin/env python3
"""Publish full-domain z-mesh variants without claiming field convergence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_4um_model import (
    grid_edges,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.material_fraction import (
    audit as material_fraction_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.mesh_variants import (
    FULL_DOMAIN_Z,
    PARTIAL_MATERIAL_Z,
    edges_sha256,
    variant_audit,
    variant_edges,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.production_readiness import (
    DEVICE_CERTIFICATE,
    SOURCE_CALIBRATION,
    sha256,
)


HERE = Path(__file__).resolve().parent
OUT = HERE / "results_4um_full_z_mesh_runsetup"
FACTORS = (1, 2, 4)


def main() -> int:
    baseline = tuple(np.asarray(value, dtype=np.float64) for value in grid_edges())
    factor_one = variant_edges(1, FULL_DOMAIN_Z)
    if not all(
        np.array_equal(left, right)
        for left, right in zip(baseline, factor_one, strict=True)
    ):
        raise RuntimeError("factor-1 full-z variant changed the baseline grid")
    if edges_sha256(baseline) != edges_sha256(factor_one):
        raise RuntimeError("factor-1 full-z hash does not match the baseline")
    full = [variant_audit(factor, FULL_DOMAIN_Z) for factor in FACTORS]
    partial = [variant_audit(factor, PARTIAL_MATERIAL_Z) for factor in FACTORS]
    payload = {
        "status": "AUDITED_SHARED_LINEAR_FULL_Z_MESH_VARIANTS_NOT_SOLVED",
        "scope": (
            "deterministic grid/layout audit only; no Maxwell, thermal, "
            "electrical, adjoint, or optimization solve"
        ),
        "au_material_fraction": material_fraction_audit(),
        "device_contract_sha256": sha256(DEVICE_CERTIFICATE),
        "baseline_source_calibration_sha256": sha256(SOURCE_CALIBRATION),
        "factor_one_matches_baseline_exactly": True,
        "full_domain_variants": full,
        "historical_partial_material_variants": partial,
        "required_per_variant_work": [
            "new all-air source calibration",
            "previous-vs-late Q stationarity",
            "target and discrete-ADE Q/closed-flux closure",
            "conservative remap, temperature, and signed current comparison",
        ],
        "promotion": {
            "is_mesh_certificate": False,
            "reason": "target device is unconfirmed and no variant field solve was run",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "FULL_Z_MESH_RUNSETUP.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Full-domain optical z-mesh runsetup",
        "",
        "Status: `AUDITED_SHARED_LINEAR_FULL_Z_MESH_VARIANTS_NOT_SOLVED`",
        "",
        "The historical sweep refined only SiO2, TaIrTe4, and Au. These new",
        "variants refine every z segment, including resolved Si, air, and both",
        "z-PML regions, while the x/y grid and lateral PML remain fixed.",
        "Factor 1 is bitwise identical to the current baseline edge arrays.",
        "",
        "| factor | grid shape | Yee cells | z-PML cells/face | Si dz (nm) | SiO2 dz (nm) | TaIrTe4 dz (nm) | Au dz (nm) | near-air dz (nm) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for audit in full:
        segments = audit["segments"]
        lines.append(
            f"| {audit['factor']} | {'x'.join(map(str, audit['grid_shape_xyz']))} | "
            f"{audit['yee_cell_count']} | {audit['pml_cells_each_face_xyz'][2]} | "
            f"{1e9*segments['resolved_si']['uniform_dz_m']:.3f} | "
            f"{1e9*segments['sio2']['uniform_dz_m']:.3f} | "
            f"{1e9*segments['tairte4']['uniform_dz_m']:.3f} | "
            f"{1e9*segments['au']['uniform_dz_m']:.3f} | "
            f"{1e9*segments['near_air']['uniform_dz_m']:.3f} |"
        )
    lines.extend(
        [
            "",
            "This file is not a convergence certificate. Every variant needs its",
            "own source calibration, time/closure gates, and downstream comparison.",
            "The physical-device contract must also be confirmed first.",
        ]
    )
    (OUT / "FULL_Z_MESH_RUNSETUP.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
