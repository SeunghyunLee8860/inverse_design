#!/usr/bin/env python3
"""Regenerate every optimization-evaluation plot from immutable raw artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

from photothermal_pte.optimization_runs.tairte4_flake_topology.run_optimization import (
    publish_plot,
)


EVALUATION_PATTERN = re.compile(r"evaluation_(\d+)")


def read_json_with_retry(path: Path, attempts: int = 10) -> list[dict]:
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.2)
    raise AssertionError("unreachable")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_artifacts(raw_root: Path, row: dict) -> tuple[int, str, Path, Path]:
    if row.get("initial_uniform"):
        return (
            1,
            "initial",
            raw_root / "evaluation_initial_rho.npz",
            raw_root / "evaluation_initial" / "objective_gradient.npz",
        )

    result_path = Path(row["solver_result"])
    evaluation_dir = result_path.parent
    match = EVALUATION_PATTERN.search(evaluation_dir.name)
    if match is None:
        raise ValueError(f"cannot extract evaluation id from {evaluation_dir}")
    evaluation_id = int(row.get("evaluation_id", match.group(1)))
    label = "reprojection" if row.get("stage_reprojection") else "candidate"
    return (
        evaluation_id,
        label,
        raw_root / f"{evaluation_dir.name}_rho.npz",
        evaluation_dir / "objective_gradient.npz",
    )


def regenerate(raw_root: Path, published: Path) -> dict:
    history = read_json_with_retry(raw_root / "history.json")
    published.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []
    last_accepted: tuple[int, str, np.ndarray, np.ndarray, dict, int] | None = None

    for history_index, row in enumerate(history):
        evaluation_id, label, rho_path, gradient_path = locate_artifacts(raw_root, row)
        if not rho_path.is_file() or not gradient_path.is_file():
            raise FileNotFoundError(f"incomplete evaluation {evaluation_id}: {rho_path}, {gradient_path}")
        with np.load(rho_path) as data:
            rho = np.asarray(data["rho"], dtype=np.float64)
        with np.load(gradient_path) as data:
            gradient = np.asarray(data["gradient_total_A"], dtype=np.float64)
        summary = {
            "beta": float(row["beta"]),
            "gray_fraction_0p01_0p99": float(row["gray_fraction"]),
            "exact": {"total_bad_cell_count": int(row["exact_bad_cells"])},
        }
        publish_plot(
            published,
            history[: history_index + 1],
            rho,
            gradient,
            summary,
            evaluation_id=evaluation_id,
            accepted=bool(row["accepted"]),
            label=label,
            publish_latest=False,
        )
        plot_path = published / f"evaluation_{evaluation_id:04d}_{label}.png"
        record = {
            "evaluation_id": evaluation_id,
            "label": label,
            "accepted": bool(row["accepted"]),
            "accepted_update_index": int(row.get("accepted_update_index", row["global_iteration"])),
            "beta": float(row["beta"]),
            "objective_A": float(row["objective_A"]),
            "plot": plot_path.name,
            "plot_sha256": sha256(plot_path),
        }
        index.append(record)
        if row["accepted"]:
            last_accepted = (
                evaluation_id, label, rho, gradient, summary, history_index
            )

    if last_accepted is not None:
        evaluation_id, label, rho, gradient, summary, history_index = last_accepted
        publish_plot(
            published,
            history[: history_index + 1],
            rho,
            gradient,
            summary,
            evaluation_id=evaluation_id,
            accepted=True,
            label=label,
            publish_latest=True,
        )

    payload = {
        "schema": "tairte4-iteration-plot-index-v1",
        "red_density_contour_present": False,
        "density_rendering": "gray_r: black=rho_1_TaIrTe4, white=rho_0_void",
        "evaluation_count": len(index),
        "latest_evaluation_id": index[-1]["evaluation_id"] if index else None,
        "plots": index,
    }
    (published / "iteration_plot_index.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--published-dir", required=True, type=Path)
    args = parser.parse_args()
    payload = regenerate(args.raw_root.resolve(), args.published_dir.resolve())
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
