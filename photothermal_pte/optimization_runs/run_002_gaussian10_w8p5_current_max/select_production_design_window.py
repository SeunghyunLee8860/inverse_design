#!/usr/bin/env python3
"""Select the Run-002 production window from the validated physical gradient."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


STATUS = "VALIDATED_PRODUCTION_DESIGN_WINDOW_SELECTION"
TARGET = 0.90
SPACING_UM = 0.1
CANVAS = {"x": [-10.0, 10.0], "y": [-10.0, 10.0]}
REVIEWED = {
    "a_positive_strip_12x6": {"x": [-6.0, 6.0], "y": [1.0, 7.0]},
    "a_negative_strip_12x6": {"x": [-6.0, 6.0], "y": [-7.0, -1.0]},
    "b_positive_strip_6x12": {"x": [1.0, 7.0], "y": [-6.0, 6.0]},
    "b_negative_strip_6x12": {"x": [-7.0, -1.0], "y": [-6.0, 6.0]},
    "centered_control_10x10": {"x": [-5.0, 5.0], "y": [-5.0, 5.0]},
}
PROMOTED = {"x": [-9.3, 9.3], "y": [-9.3, 9.3]}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def mask(x: np.ndarray, y: np.ndarray, bounds: dict) -> np.ndarray:
    return (
        (x[:, None] >= bounds["x"][0] - 1e-12)
        & (x[:, None] <= bounds["x"][1] + 1e-12)
        & (y[None, :] >= bounds["y"][0] - 1e-12)
        & (y[None, :] <= bounds["y"][1] + 1e-12)
    )


def metrics(absolute_gradient: np.ndarray, selected: np.ndarray, bounds: dict) -> dict:
    total = float(np.sum(absolute_gradient))
    retained = float(np.sum(absolute_gradient[selected]))
    return {
        "bounds_um": bounds,
        "node_count": int(np.count_nonzero(selected)),
        "area_um2": float(
            (bounds["x"][1] - bounds["x"][0])
            * (bounds["y"][1] - bounds["y"][0])
        ),
        "absolute_gradient_L1_A": retained,
        "absolute_gradient_L1_fraction": retained / total,
        "passes_90_percent_gate": bool(retained / total >= TARGET),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-result", type=Path, required=True)
    parser.add_argument("--combined-npz", type=Path, required=True)
    parser.add_argument("--combined-npz-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result_path = args.combined_result.expanduser().resolve()
    combined = json.loads(result_path.read_text())
    if combined.get("status") != "VALIDATED_PRODUCTION_COMBINED_PHYSICAL_RHO_ADFD_SMOKE":
        raise RuntimeError("combined physical-rho gradient is not validated")
    source = args.combined_npz.expanduser().resolve()
    if sha256(source) != args.combined_npz_sha256:
        raise RuntimeError("combined gradient NPZ SHA mismatch")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("refusing non-empty output directory")
    output.mkdir(parents=True, exist_ok=True)

    data = np.load(source)
    gradient = np.asarray(data["gradient_total_A"], float)
    if gradient.shape != (201, 201) or not np.all(np.isfinite(gradient)):
        raise RuntimeError("unexpected coarse physical-gradient array")
    x = np.linspace(CANVAS["x"][0], CANVAS["x"][1], gradient.shape[0])
    y = np.linspace(CANVAS["y"][0], CANVAS["y"][1], gradient.shape[1])
    if max(np.max(np.abs(np.diff(x) - SPACING_UM)), np.max(np.abs(np.diff(y) - SPACING_UM))) > 1e-12:
        raise RuntimeError("coarse node spacing changed")
    absolute = np.abs(gradient)
    candidates = {name: metrics(absolute, mask(x, y, bounds), bounds) for name, bounds in REVIEWED.items()}
    promoted_mask = mask(x, y, PROMOTED)
    promoted = metrics(absolute, promoted_mask, PROMOTED)
    if any(value["passes_90_percent_gate"] for value in candidates.values()):
        raise RuntimeError("an original reviewed small window unexpectedly passes; selection contract must be re-reviewed")
    if not promoted["passes_90_percent_gate"]:
        raise RuntimeError("approved expanded centered window misses the 90% gate")
    previous_half = 9.2
    previous_bounds = {"x": [-previous_half, previous_half], "y": [-previous_half, previous_half]}
    previous = metrics(absolute, mask(x, y, previous_bounds), previous_bounds)
    if previous["passes_90_percent_gate"]:
        raise RuntimeError("18.6 um was not the first 0.2-um-step centered square to pass")

    npz_path = output / "production_design_window_selection.npz"
    np.savez_compressed(
        npz_path,
        x_um=x,
        y_um=y,
        gradient_total_A=gradient,
        gradient_absolute_A=absolute,
        promoted_mask=promoted_mask,
    )
    result = {
        "status": STATUS,
        "passed": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"combined_result": artifact(result_path), "combined_gradient": artifact(source)},
        "canvas_um": CANVAS,
        "coarse_node_spacing_nm": SPACING_UM * 1000.0,
        "selection_gate": {"metric": "absolute combined physical-density gradient L1 fraction", "minimum": TARGET},
        "original_reviewed_candidates": candidates,
        "original_candidates_all_failed": True,
        "promoted_window": {
            "name": "centered_18p6um",
            **promoted,
            "selection_reason": "smallest centered square on the predeclared 0.2 um span sequence that retains at least 90%",
            "production_node_spacing_nm": 50.0,
            "production_shape": [373, 373],
        },
        "immediately_smaller_centered_control": {"name": "centered_18p4um", **previous},
        "full_canvas_area_um2": 400.0,
        "promoted_area_fraction": promoted["area_um2"] / 400.0,
        "optimizer_started": False,
        "Maxwell_solves": 0,
        "thermal_solves": 0,
        "raw_artifact": artifact(npz_path),
    }
    result_path_out = output / "production_design_window_selection_result.json"
    result_path_out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
