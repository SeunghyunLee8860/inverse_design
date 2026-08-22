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
    disk,
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
    *,
    geometry_mode: str | None = None,
    contact_axis: str | None = None,
) -> tuple[np.ndarray, list[dict[str, object]], str]:
    value = np.asarray(initial, dtype=bool).copy()
    locked_solid = (
        CONTRACT.fixed_design_solid_mask
        if geometry_mode == "diagonal_45_contact_anchored"
        else np.zeros_like(value)
    )
    if locked_solid.shape != value.shape:
        raise ValueError("locked contact mask does not match exact-repair design")
    value[locked_solid] = True
    history: list[dict[str, object]] = []
    seen: set[bytes] = set()
    phases = ("solid", "void") if order == "solid_first" else ("void", "solid")
    stop_reason = "maximum_iterations"
    for iteration in range(maximum_iterations + 1):
        audit, arrays = exact_binary_audit(
            value.astype(np.float64),
            geometry_mode=geometry_mode,
            contact_axis=contact_axis,
        )
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
            _, arrays = exact_binary_audit(
                value.astype(np.float64),
                geometry_mode=geometry_mode,
                contact_axis=contact_axis,
            )
            if phase == "solid":
                value[arrays["bad_solid"]] = False
            else:
                value[arrays["bad_void"]] = True
            value[locked_solid] = True
    return value, history, stop_reason


def _disk_support_actions(
    value: np.ndarray,
    bad_solid: np.ndarray,
    bad_void: np.ndarray,
) -> list[np.ndarray]:
    """Return local remove/fill or grow-to-disk actions for exact defects."""

    structure = disk()
    radius = (structure.shape[0] - 1) // 2
    offsets = np.argwhere(structure) - radius
    shape = value.shape
    actions: list[np.ndarray] = []
    for phase, bad in ((True, bad_solid), (False, bad_void)):
        positions = np.argwhere(bad)
        if not len(positions):
            continue
        # Removing a thin solid or filling a narrow void is one legitimate
        # repair.  It is intentionally kept alongside the alternative of
        # growing that phase to a complete 500-nm support disk.
        removed = value.copy()
        removed[bad] = not phase
        actions.append(removed)
        for point_i, point_j in positions:
            for centre_i in range(
                max(0, int(point_i) - radius),
                min(shape[0], int(point_i) + radius + 1),
            ):
                for centre_j in range(
                    max(0, int(point_j) - radius),
                    min(shape[1], int(point_j) + radius + 1),
                ):
                    grown = value.copy()
                    for offset_i, offset_j in offsets:
                        index_i = centre_i + int(offset_i)
                        index_j = centre_j + int(offset_j)
                        if 0 <= index_i < shape[0] and 0 <= index_j < shape[1]:
                            grown[index_i, index_j] = phase
                    actions.append(grown)
    return actions


def _repair_score(
    candidate: np.ndarray,
    source: np.ndarray,
    audit: dict[str, object],
    objective_gradient: np.ndarray | None,
) -> tuple[float, float, int]:
    delta = candidate.astype(np.float64) - source.astype(np.float64)
    predicted_change = (
        0.0
        if objective_gradient is None
        else float(np.sum(np.asarray(objective_gradient, dtype=np.float64) * delta))
    )
    return (
        float(audit["total_bad_cell_count"]),
        -predicted_change,
        int(np.count_nonzero(delta)),
    )


def gradient_aware_exact_repair(
    initial: np.ndarray,
    *,
    objective_gradient: np.ndarray | None = None,
    geometry_mode: str,
    contact_axis: str,
    beam_width: int = 64,
    maximum_iterations: int = 32,
    maximum_candidates: int = 4,
) -> dict[str, object]:
    """Find exact-feasible 500-nm candidates without another physics solve.

    The old alternating opening repair can enter a two-state solid/void cycle.
    This bounded beam search starts from both alternating orderings and adds
    the missing geometric choice: grow a surviving defect into a complete
    support disk.  Candidate ordering uses the latest physical objective
    gradient only as a first-order ranking; every returned geometry must pass
    the independent exact binary audit.
    """

    source = np.asarray(initial, dtype=bool)
    if source.ndim != 2:
        raise ValueError("exact repair requires a two-dimensional binary design")
    if objective_gradient is not None:
        objective_gradient = np.asarray(objective_gradient, dtype=np.float64)
        if objective_gradient.shape != source.shape or not np.all(
            np.isfinite(objective_gradient)
        ):
            raise ValueError("objective gradient must be finite and match the design")
    if beam_width <= 0 or maximum_iterations <= 0 or maximum_candidates <= 0:
        raise ValueError("repair search budgets must be positive")
    locked_solid = (
        CONTRACT.fixed_design_solid_mask
        if geometry_mode == "diagonal_45_contact_anchored"
        else np.zeros_like(source)
    )
    if locked_solid.shape != source.shape:
        raise ValueError("locked contact mask does not match exact-repair design")
    source = source.copy()
    source[locked_solid] = True
    if objective_gradient is not None:
        objective_gradient = np.asarray(objective_gradient).copy()
        objective_gradient[locked_solid] = 0.0

    source_audit, _ = exact_binary_audit(
        source.astype(np.float64),
        geometry_mode=geometry_mode,
        contact_axis=contact_axis,
    )
    seeds: list[tuple[str, np.ndarray, list[dict[str, object]], str]] = [
        ("unrepaired", source.copy(), [], "source_design")
    ]
    for order in ("solid_first", "void_first"):
        candidate, history, reason = active_set_repair(
            source,
            order,
            maximum_iterations,
            geometry_mode=geometry_mode,
            contact_axis=contact_axis,
        )
        seeds.append((order, candidate, history, reason))

    visited: set[bytes] = set()
    beam: list[tuple[np.ndarray, list[dict[str, object]]]] = []
    for order, candidate, history, reason in seeds:
        key = np.packbits(candidate, bitorder="little").tobytes()
        if key in visited:
            continue
        visited.add(key)
        beam.append((candidate, [{"seed_order": order, "seed_stop_reason": reason, "seed_history": history}]))

    feasible: list[tuple[tuple[float, float, int], np.ndarray, list[dict[str, object]], dict[str, object]]] = []
    explored = 0
    for search_iteration in range(maximum_iterations + 1):
        next_states: list[
            tuple[
                tuple[float, float, int],
                np.ndarray,
                list[dict[str, object]],
            ]
        ] = []
        for candidate, history in beam:
            audit, arrays = exact_binary_audit(
                candidate.astype(np.float64),
                geometry_mode=geometry_mode,
                contact_axis=contact_axis,
            )
            score = _repair_score(candidate, source, audit, objective_gradient)
            explored += 1
            if bool(audit["passed"]):
                feasible.append((score, candidate.copy(), history, audit))
                continue
            for action in _disk_support_actions(
                candidate,
                arrays["bad_solid"],
                arrays["bad_void"],
            ):
                action[locked_solid] = True
                key = np.packbits(action, bitorder="little").tobytes()
                if key in visited:
                    continue
                visited.add(key)
                action_audit, _ = exact_binary_audit(
                    action.astype(np.float64),
                    geometry_mode=geometry_mode,
                    contact_axis=contact_axis,
                )
                action_score = _repair_score(
                    action, source, action_audit, objective_gradient
                )
                next_states.append(
                    (
                        action_score,
                        action,
                        history
                        + [
                            {
                                "search_iteration": search_iteration + 1,
                                "exact_bad_cells": int(
                                    action_audit["total_bad_cell_count"]
                                ),
                                "changed_node_count": int(
                                    np.count_nonzero(action != source)
                                ),
                                "predicted_objective_change_A": -action_score[1],
                            }
                        ],
                    )
                )
        if len(feasible) >= maximum_candidates or not next_states:
            break
        next_states.sort(key=lambda item: item[0])
        beam = [(item[1], item[2]) for item in next_states[:beam_width]]

    feasible.sort(key=lambda item: item[0])
    rows: list[dict[str, object]] = []
    arrays: list[np.ndarray] = []
    for rank, (score, candidate, history, audit) in enumerate(
        feasible[:maximum_candidates]
    ):
        rows.append(
            {
                "rank": rank,
                "exact_audit": audit,
                "changed_node_count": score[2],
                "changed_node_fraction": float(score[2] / source.size),
                "predicted_objective_change_A": -score[1],
                "search_history": history,
            }
        )
        arrays.append(candidate.astype(np.float64))
    return {
        "schema": "gradient-aware-exact-500nm-repair-v1",
        "geometry_mode": geometry_mode,
        "contact_axis": contact_axis,
        "source_audit": source_audit,
        "explored_state_count": explored,
        "visited_state_count": len(visited),
        "passed": bool(rows),
        "candidates": rows,
        "candidate_arrays": arrays,
    }


def artifact(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--maximum-iterations", type=int, default=100)
    args = parser.parse_args()
    if CONTRACT.geometry_mode not in {
        "contact_anchored",
        "left_right_contact_anchored",
        "diagonal_45_contact_anchored",
    }:
        raise RuntimeError("select a contact-anchored geometry")
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
    extent = tuple(
        value * 1.0e6
        for axis in ("x", "y")
        for value in CONTRACT.design_bounds_m[axis]
    )
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
