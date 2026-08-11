#!/usr/bin/env python3
"""Build deterministic exact-500-nm binary cleanup candidates.

The two phases cannot be repaired simultaneously without an ordering choice:
removing a thin solid can create a small void, while filling a small void can
create a thin solid.  This tool therefore produces and audits both orderings.
It does not remove electrically isolated components; connectivity is outside
the approved minimum-feature contract.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def active_set_repair(
    initial: np.ndarray,
    order: str,
    maximum_iterations: int,
) -> tuple[np.ndarray, list[dict[str, object]], str]:
    value = np.asarray(initial, dtype=bool).copy()
    history: list[dict[str, object]] = []
    seen: set[bytes] = set()
    phases = ("solid", "void") if order == "solid_first" else ("void", "solid")
    stop_reason = "maximum_iterations"
    for iteration in range(maximum_iterations + 1):
        audit, arrays = exact_binary_audit(value.astype(np.float64))
        history.append(
            {
                "iteration": iteration,
                "solid_bad": int(audit["solid_bad_cell_count"]),
                "void_bad": int(audit["void_bad_cell_count"]),
                "total_bad": int(audit["total_bad_cell_count"]),
                "solid_fraction": float(audit["solid_fraction"]),
            }
        )
        if bool(audit["passed"]):
            stop_reason = "exact_global_gate_passed"
            break
        key = np.packbits(value, bitorder="little").tobytes()
        if key in seen:
            stop_reason = "cycle_detected"
            break
        seen.add(key)
        if iteration == maximum_iterations:
            break
        for phase in phases:
            _, arrays = exact_binary_audit(value.astype(np.float64))
            if phase == "solid":
                value[arrays["bad_solid"]] = False
            else:
                value[arrays["bad_void"]] = True
    return value, history, stop_reason


def artifact(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--maximum-iterations", type=int, default=100)
    args = parser.parse_args()
    if CONTRACT.geometry_mode != "contact_anchored":
        raise RuntimeError("set TAIRTE4_TOPOLOGY_GEOMETRY=contact_anchored")
    if args.maximum_iterations <= 0:
        raise ValueError("maximum iterations must be positive")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source = args.rho_npz.expanduser().resolve()
    with np.load(source) as data:
        rho = np.asarray(data["rho"], dtype=np.float64)
    if rho.shape != CONTRACT.design_node_shape:
        raise RuntimeError(f"density shape {rho.shape} != {CONTRACT.design_node_shape}")
    source_audit, source_arrays = exact_binary_audit(rho)
    thresholded = source_arrays["binary"]

    result: dict[str, object] = {
        "schema": "contact-anchored-two-order-exact-binary-repair-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": artifact(source),
        "source_audit": source_audit,
        "connectivity_cleanup_applied": False,
        "candidates": {},
    }
    plot_rows: list[tuple[str, np.ndarray, np.ndarray]] = []
    for order in ("solid_first", "void_first"):
        candidate, history, stop_reason = active_set_repair(
            thresholded, order, args.maximum_iterations
        )
        audit, arrays = exact_binary_audit(candidate.astype(np.float64))
        changed = candidate != thresholded
        path = output / f"{order}_exact_binary_candidate.npz"
        np.savez_compressed(
            path,
            rho=candidate.astype(np.float64),
            rho_binary=candidate.astype(np.uint8),
            changed_from_thresholded=changed,
            bad_solid=arrays["bad_solid"],
            bad_void=arrays["bad_void"],
        )
        result["candidates"][order] = {
            "passed": bool(audit["passed"]),
            "stop_reason": stop_reason,
            "audit": audit,
            "changed_node_count": int(np.count_nonzero(changed)),
            "changed_node_fraction": float(np.mean(changed)),
            "iteration_history": history,
            "artifact": artifact(path),
        }
        plot_rows.append((order, candidate, changed))

    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    extent = (-12.0, 12.0, -10.0, 10.0)
    axes[0, 0].imshow(thresholded.T, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=1)
    axes[0, 0].set_title("thresholded source")
    source_bad = source_arrays["bad_solid"] | source_arrays["bad_void"]
    axes[1, 0].imshow(source_bad.T, origin="lower", extent=extent, cmap="Reds", vmin=0, vmax=1)
    axes[1, 0].set_title(f"source exact-bad={np.count_nonzero(source_bad)}")
    for column, (order, candidate, changed) in enumerate(plot_rows, start=1):
        audit = result["candidates"][order]["audit"]
        axes[0, column].imshow(candidate.T, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=1)
        axes[0, column].set_title(f"{order}; bad={audit['total_bad_cell_count']}")
        axes[1, column].imshow(changed.T, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=1)
        axes[1, column].set_title(f"changed nodes={np.count_nonzero(changed)}")
    for axis in axes.ravel():
        axis.set_xlabel("Lumerical x=b (um)")
        axis.set_ylabel("Lumerical y=a (um)")
    plot_path = output / "exact_binary_repair_candidates.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    result["plot"] = artifact(plot_path)
    passed_orders = [name for name, row in result["candidates"].items() if row["passed"]]
    result["passed_orders"] = passed_orders
    result["passed"] = bool(passed_orders)
    result["status"] = (
        "VALIDATED_EXACT_500NM_BINARY_REPAIR_CANDIDATES"
        if passed_orders
        else "FAILED_EXACT_500NM_BINARY_REPAIR_CANDIDATES"
    )
    result_path = output / "exact_binary_repair_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
