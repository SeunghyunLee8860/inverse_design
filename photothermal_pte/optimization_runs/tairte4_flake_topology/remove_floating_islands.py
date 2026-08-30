#!/usr/bin/env python3
"""Remove exact-binary TaIrTe4 components disconnected from both terminals.

This is a deterministic diagnostic transform, not an optimizer update.  The
input artifact is preserved; a new NPZ and a connectivity/DFM audit JSON are
written for a fresh Maxwell -> thermal -> electrical reevaluation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT
from photothermal_pte.optimization_runs.tairte4_flake_topology.optimization_support import (
    exact_binary_audit,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def component_audit(binary: np.ndarray, connectivity: int) -> tuple[np.ndarray, list[dict[str, object]]]:
    if connectivity == 4:
        structure = ndimage.generate_binary_structure(2, 1)
    elif connectivity == 8:
        structure = ndimage.generate_binary_structure(2, 2)
    else:
        raise ValueError("connectivity must be 4 or 8")
    labels, count = ndimage.label(binary, structure=structure)
    records: list[dict[str, object]] = []
    terminal_labels: set[int] = set()
    if CONTRACT.contact_axis == "y":
        low_edge, high_edge = labels[:, 0], labels[:, -1]
    else:
        low_edge, high_edge = labels[0, :], labels[-1, :]
    terminal_labels.update(int(value) for value in np.unique(low_edge) if value != 0)
    terminal_labels.update(int(value) for value in np.unique(high_edge) if value != 0)
    for label in range(1, count + 1):
        mask = labels == label
        records.append(
            {
                "label": label,
                "node_count": int(np.count_nonzero(mask)),
                "touches_low_terminal": bool(np.any(low_edge == label)),
                "touches_high_terminal": bool(np.any(high_edge == label)),
                "terminal_connected": label in terminal_labels,
            }
        )
    keep = np.isin(labels, list(terminal_labels))
    return keep, records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--audit-json", required=True, type=Path)
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    with np.load(source) as data:
        rho = np.asarray(data["rho"], dtype=np.float64)
    if rho.shape != CONTRACT.design_node_shape:
        raise RuntimeError(f"rho shape {rho.shape} != contract {CONTRACT.design_node_shape}")
    if not np.all((rho == 0.0) | (rho == 1.0)):
        raise RuntimeError("input must be exact binary")
    binary = rho.astype(bool)

    keep4, components4 = component_audit(binary, 4)
    keep8, components8 = component_audit(binary, 8)
    if not np.array_equal(keep4, keep8):
        raise RuntimeError("4- and 8-neighbour terminal-connected supports differ")
    cleaned = binary & keep4
    removed = binary & ~cleaned
    if not np.any(removed):
        raise RuntimeError("no floating solid component was found")

    before_dfm, _ = exact_binary_audit(
        binary, geometry_mode=CONTRACT.geometry_mode, contact_axis=CONTRACT.contact_axis
    )
    after_dfm, after_arrays = exact_binary_audit(
        cleaned, geometry_mode=CONTRACT.geometry_mode, contact_axis=CONTRACT.contact_axis
    )

    output = args.output_npz.expanduser().resolve()
    audit_path = args.audit_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or audit_path.exists():
        raise RuntimeError("refusing to overwrite an existing output")
    np.savez_compressed(
        output,
        rho=cleaned.astype(np.float64),
        rho_binary=cleaned.astype(np.uint8),
        removed_floating_solid=removed,
        bad_solid=after_arrays["bad_solid"],
        bad_void=after_arrays["bad_void"],
    )
    record = {
        "schema": "exact-binary-terminal-disconnected-island-removal-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PREPARED_FLOATING_ISLAND_REMOVED_DIAGNOSTIC",
        "axis_contract": "Lumerical x=b, y=a, z=c",
        "geometry_mode": CONTRACT.geometry_mode,
        "contact_axis": CONTRACT.contact_axis,
        "source": {"path": str(source), "size_bytes": source.stat().st_size, "sha256": sha256(source)},
        "output": {"path": str(output), "size_bytes": output.stat().st_size, "sha256": sha256(output)},
        "shape": list(binary.shape),
        "solid_nodes_before": int(np.count_nonzero(binary)),
        "solid_nodes_after": int(np.count_nonzero(cleaned)),
        "removed_solid_nodes": int(np.count_nonzero(removed)),
        "removed_solid_fraction_of_design": float(np.mean(removed)),
        "removed_solid_fraction_of_original_solid": float(np.count_nonzero(removed) / np.count_nonzero(binary)),
        "components_4_neighbour": components4,
        "components_8_neighbour": components8,
        "terminal_connected_support_identical_for_4_and_8_neighbour": True,
        "exact_500nm_audit_before": before_dfm,
        "exact_500nm_audit_after": after_dfm,
        "transform": "all exact-solid components touching neither electrical terminal were changed from rho=1 to rho=0",
    }
    audit_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
