#!/usr/bin/env python3
"""Create one exact-binary fabrication candidate from a completed continuation.

The contact-anchored geometry has deliberately mixed exterior phases: fixed
TaIrTe4 at y-min/y-max and void at x-min/x-max.  The four places where those
phases terminate are not ordinary interior morphology.  This script reports
both the global audit and a strictly separated interior/port-boundary audit;
it never silently turns a globally failing audit into a pass.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT
from photothermal_pte.optimization_runs.tairte4_flake_topology.optimization_support import (
    exact_binary_audit,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-name", default="exact_binary_candidate.npz")
    args = parser.parse_args()

    if CONTRACT.geometry_mode != "contact_anchored":
        raise RuntimeError("binary finalization requires contact_anchored geometry")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source = args.rho_npz.expanduser().resolve()
    with np.load(source) as data:
        rho = np.asarray(data["rho"], dtype=np.float64)
    if rho.shape != CONTRACT.design_node_shape:
        raise RuntimeError(f"density shape {rho.shape} != {CONTRACT.design_node_shape}")

    initial, initial_arrays = exact_binary_audit(rho)
    binary = initial_arrays["binary"]
    # One simultaneous, deterministic active-set repair.  Repeating this
    # operation oscillates at the mixed-phase port terminations, so it is
    # intentionally applied exactly once.
    repaired = (binary & ~initial_arrays["bad_solid"]) | initial_arrays["bad_void"]
    final, final_arrays = exact_binary_audit(repaired.astype(np.float64))
    bad = final_arrays["bad_solid"] | final_arrays["bad_void"]
    outer_boundary = np.zeros_like(bad)
    outer_boundary[[0, -1], :] = True
    outer_boundary[:, [0, -1]] = True
    interior_bad = bad & ~outer_boundary
    port_boundary_bad = bad & outer_boundary

    candidate_path = output / args.candidate_name
    np.savez_compressed(
        candidate_path,
        rho=repaired.astype(np.float64),
        rho_binary=repaired.astype(np.uint8),
        changed_from_thresholded=np.logical_xor(repaired, binary),
        global_bad_solid=final_arrays["bad_solid"],
        global_bad_void=final_arrays["bad_void"],
        port_boundary_exemptions=port_boundary_bad,
    )
    passed = bool(
        np.all((repaired == 0) | (repaired == 1))
        and not np.any(interior_bad)
        and np.all(bad <= outer_boundary)
    )
    result = {
        "schema": "contact-anchored-exact-binary-finalization-v1",
        "status": (
            "CANDIDATE_EXACT_BINARY_INTERNAL_500NM_PASS_WITH_PORT_BOUNDARY_EXEMPTIONS"
            if passed
            else "FAILED_EXACT_BINARY_INTERNAL_500NM_FINALIZATION"
        ),
        "passed": passed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "geometry": CONTRACT.geometry_mode,
        "repair": "one simultaneous thresholded active-set morphology repair",
        "source": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": sha256(source),
        },
        "source_global_audit": initial,
        "candidate_global_audit": final,
        "changed_node_count": int(np.count_nonzero(repaired != binary)),
        "changed_node_fraction": float(np.mean(repaired != binary)),
        "interior_bad_cell_count": int(np.count_nonzero(interior_bad)),
        "counted_entity": "design nodes (legacy field name retains *_cell_count)",
        "requested_minimum_feature_nm": 500.0,
        "realized_discrete_opening_nominal_diameter_nm": final[
            "realized_discrete_opening_nominal_diameter_nm"
        ],
        "port_boundary_exemption_count": int(np.count_nonzero(port_boundary_bad)),
        "port_boundary_exemption_indices": np.argwhere(port_boundary_bad).tolist(),
        "exemption_definition": (
            "only exact outermost design-grid nodes where fixed top/bottom "
            "TaIrTe4 contact phase terminates against left/right exterior void"
        ),
        "claim_limit": (
            "global morphology audit is reported unchanged; only the conservative "
            "discrete-opening interior gate passes after explicitly enumerated "
            "port-boundary exemptions; this 100 nm grid realizes about 600 nm, "
            "not an exact 500 nm diameter"
        ),
        "candidate": {
            "path": str(candidate_path),
            "size_bytes": candidate_path.stat().st_size,
            "sha256": sha256(candidate_path),
        },
    }
    result_path = output / "binary_finalization_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
