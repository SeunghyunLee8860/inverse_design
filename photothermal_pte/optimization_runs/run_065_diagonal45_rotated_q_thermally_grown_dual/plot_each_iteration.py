#!/usr/bin/env python3
"""Render every completed Run065 full-physics evaluation without rerunning physics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REFERENCE_POWER_W = 285.0e-6
PATTERN = re.compile(r"evaluation_(\d+)_beta([^_]+)_official_ansys_dfm$")
DEFAULT_RAW = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_rotated45_edge_contact_anchored/"
    "run065_diagonal45_rotated_q_dual_thermally_grown_v3_from_uniform"
)


def diamond_coordinates(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    u = np.linspace(-12.0, 12.0, shape[0])
    v = np.linspace(-12.0, 12.0, shape[1])
    uu, vv = np.meshgrid(u, v, indexing="ij")
    return (uu - vv) / np.sqrt(2.0), (uu + vv) / np.sqrt(2.0)


def signed_limit(values: np.ndarray) -> float:
    limit = float(np.nanpercentile(np.abs(values), 99.5))
    return max(limit, np.finfo(float).tiny)


def draw_field(axis, x, y, values, *, cmap, vmin=None, vmax=None, title: str):
    image = axis.pcolormesh(
        x.T,
        y.T,
        values.T,
        shading="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    axis.set_aspect("equal")
    axis.set_xlabel("x = b (um)")
    axis.set_ylabel("y = a (um)")
    axis.set_title(title, fontsize=10)
    return image


def render(directory: Path, output: Path) -> None:
    match = PATTERN.match(directory.name)
    if match is None:
        raise ValueError(f"unrecognized evaluation directory: {directory}")
    evaluation = int(match.group(1))
    beta = match.group(2)
    with np.load(directory / "objective_gradient.npz") as combined:
        rho = np.asarray(combined["rho"], dtype=np.float64)
        objective = float(combined["objective_A"])
        objective_a = float(combined["objective_Ea_A"])
        objective_b = float(combined["objective_Eb_A"])
        gradient_a = np.asarray(combined["gradient_Ea_A"], dtype=np.float64)
        gradient_b = np.asarray(combined["gradient_Eb_A"], dtype=np.float64)
        gradient_dual = np.asarray(combined["gradient_total_A"], dtype=np.float64)
    with np.load(directory / "Ea" / "objective_gradient.npz") as data:
        temperature_a = np.asarray(data["temperature_K"], dtype=np.float64)
    with np.load(directory / "Eb" / "objective_gradient.npz") as data:
        temperature_b = np.asarray(data["temperature_K"], dtype=np.float64)
    result = json.loads((directory / "objective_gradient_result.json").read_text())
    source_power = 0.5 * sum(float(value) for value in result["source_powers_W"].values())
    scale = REFERENCE_POWER_W / source_power
    current_scale_nA = scale * 1.0e9
    x, y = diamond_coordinates(rho.shape)
    figure, axes = plt.subplots(2, 3, figsize=(15.2, 9.6), constrained_layout=True)
    structure = draw_field(
        axes[0, 0], x, y, rho, cmap="gray_r", vmin=0.0, vmax=1.0,
        title="45 deg TaIrTe4 density (black=1)",
    )
    figure.colorbar(structure, ax=axes[0, 0], shrink=0.80)
    for axis, temperature, label in (
        (axes[0, 1], temperature_a, "E||a"),
        (axes[0, 2], temperature_b, "E||b"),
    ):
        scaled = temperature * scale
        image = draw_field(
            axis, x, y, scaled, cmap="inferno", vmin=0.0,
            vmax=float(np.nanmax(scaled)), title=f"Temperature rise {label} at 285 uW (K)",
        )
        figure.colorbar(image, ax=axis, shrink=0.80)
    for axis, gradient, label in (
        (axes[1, 0], gradient_a, "dI_a/drho"),
        (axes[1, 1], gradient_b, "dI_b/drho"),
        (axes[1, 2], gradient_dual, "dual soft-min dI/drho"),
    ):
        scaled = gradient * current_scale_nA
        limit = signed_limit(scaled)
        image = draw_field(
            axis, x, y, scaled, cmap="coolwarm", vmin=-limit, vmax=limit,
            title=f"{label} at 285 uW (nA/node)",
        )
        figure.colorbar(image, ax=axis, shrink=0.80)
    figure.suptitle(
        f"Run065 evaluation {evaluation:04d}, beta={beta} | "
        f"Ia={objective_a * current_scale_nA:+.3f} nA | "
        f"Ib={objective_b * current_scale_nA:+.3f} nA | "
        f"dual={objective * current_scale_nA:+.3f} nA",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def process(raw: Path) -> int:
    output_dir = raw / "iteration_figures"
    count = 0
    for directory in sorted(raw.glob("evaluation_*_beta*_official_ansys_dfm")):
        match = PATTERN.match(directory.name)
        if match is None or not (directory / "objective_gradient.npz").is_file():
            continue
        output = output_dir / f"evaluation_{int(match.group(1)):04d}_beta{match.group(2)}.png"
        if output.is_file():
            continue
        render(directory, output)
        shutil.copyfile(output, output_dir / "LATEST.png")
        print(output, flush=True)
        count += 1
    return count


def pid_alive(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()
    while True:
        process(args.raw_root)
        if not args.watch or not pid_alive(args.pid):
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
