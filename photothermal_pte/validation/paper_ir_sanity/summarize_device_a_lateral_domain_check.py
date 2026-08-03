#!/usr/bin/env python3
"""Compare one 60-um Device-A thermal case against a larger lateral domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-case-dir", type=Path, required=True)
    parser.add_argument("--refined-case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def relative_change(base: float, refined: float) -> float:
    return abs(refined - base) / max(abs(base), abs(refined), 1.0e-300)


def coordinate_indices(
    base: np.ndarray, refined: np.ndarray, tolerance_m: float = 1.0e-15
) -> tuple[np.ndarray, np.ndarray]:
    base_indices = []
    refined_indices = []
    for i, coordinate in enumerate(base):
        j = int(np.argmin(np.abs(refined - coordinate)))
        if abs(refined[j] - coordinate) <= tolerance_m:
            base_indices.append(i)
            refined_indices.append(j)
    if not base_indices:
        raise RuntimeError("the two thermal grids have no coincident coordinates")
    return np.asarray(base_indices, int), np.asarray(refined_indices, int)


def nrmse(base: np.ndarray, refined: np.ndarray, mask: np.ndarray) -> float:
    difference = (refined - base)[mask]
    reference = base[mask]
    return float(
        np.linalg.norm(difference)
        / max(np.linalg.norm(reference), 1.0e-300)
    )


def discretization(summary: dict, fallback_domain_um: float) -> dict:
    return summary.get(
        "thermal_discretization",
        {
            "lateral_domain_um": fallback_domain_um,
            "Si_depth_um": 20.0,
            "core_xy_cell_size_nm": 100.0,
            "flake_dz_nm": 10.0,
            "source": "legacy batch contract fallback",
        },
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_summary = json.loads((args.base_case_dir / "summary.json").read_text())
    refined_summary = json.loads((args.refined_case_dir / "summary.json").read_text())
    base_path = args.base_case_dir / "thermal_lumerical_coordinate_fields.npz"
    refined_path = (
        args.refined_case_dir / "thermal_lumerical_coordinate_fields.npz"
    )
    with np.load(base_path, allow_pickle=False) as base, np.load(
        refined_path, allow_pickle=False
    ) as refined:
        xb = 0.5 * (base["x_edges_m"][:-1] + base["x_edges_m"][1:])
        yb = 0.5 * (base["y_edges_m"][:-1] + base["y_edges_m"][1:])
        xr = 0.5 * (refined["x_edges_m"][:-1] + refined["x_edges_m"][1:])
        yr = 0.5 * (refined["y_edges_m"][:-1] + refined["y_edges_m"][1:])
        ib, ir = coordinate_indices(xb, xr)
        jb, jr = coordinate_indices(yb, yr)
        base_temperature = np.asarray(base["temperature_flake_average_K"], float)[
            np.ix_(ib, jb)
        ]
        refined_temperature = np.asarray(
            refined["temperature_flake_average_K"], float
        )[np.ix_(ir, jr)]
        base_gradient = np.asarray(base["grad_T_magnitude_K_m"], float)[
            np.ix_(ib, jb)
        ]
        refined_gradient = np.asarray(refined["grad_T_magnitude_K_m"], float)[
            np.ix_(ir, jr)
        ]
        mask = (
            np.asarray(base["strict_valid_xy_mask"], bool)[np.ix_(ib, jb)]
            & np.asarray(refined["strict_valid_xy_mask"], bool)[np.ix_(ir, jr)]
        )
        if not np.any(mask):
            raise RuntimeError("no common strict-valid TaIrTe4 cells")
        temperature_nrmse = nrmse(base_temperature, refined_temperature, mask)
        gradient_nrmse = nrmse(base_gradient, refined_gradient, mask)

        temperature_difference = np.where(
            mask, refined_temperature - base_temperature, np.nan
        )
        gradient_difference = np.where(
            mask, refined_gradient - base_gradient, np.nan
        )
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
        for ax, values, title, unit in (
            (axes[0], temperature_difference, "larger-domain minus 60-um dT", "K"),
            (axes[1], gradient_difference, "larger-domain minus 60-um |grad T|", "K/m"),
        ):
            limit = float(np.nanmax(np.abs(values)))
            image = ax.pcolormesh(
                xb[ib] * 1.0e6,
                yb[jb] * 1.0e6,
                np.ma.masked_invalid(values).T,
                shading="auto",
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
            ax.set_aspect("equal")
            ax.set_xlabel("Lumerical x = crystal b (um)")
            ax.set_ylabel("Lumerical y = crystal a (um)")
            ax.set_title(title)
            fig.colorbar(image, ax=ax, label=unit)
        plot_path = args.output_dir / "LATERAL_DOMAIN_DIFFERENCE_LUMERICAL_COORDINATES.png"
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)

    base_thermal = base_summary["thermal"]
    refined_thermal = refined_summary["thermal"]
    metrics = {
        "Tmax_relative_change": relative_change(
            base_thermal["Tmax_rise_K"], refined_thermal["Tmax_rise_K"]
        ),
        "TaIrTe4_volume_average_relative_change": relative_change(
            base_thermal["TaIrTe4_volume_average_rise_K"],
            refined_thermal["TaIrTe4_volume_average_rise_K"],
        ),
        "production_current_relative_change": relative_change(
            base_thermal["production_current_A"],
            refined_thermal["production_current_A"],
        ),
        "temperature_field_NRMSE": temperature_nrmse,
        "gradient_field_NRMSE": gradient_nrmse,
    }
    gates = {name: value < 0.01 for name, value in metrics.items()}
    output = {
        "status": (
            "PASSED_DEVICE_A_LATERAL_DOMAIN_ROBUSTNESS"
            if all(gates.values())
            else "FAILED_DEVICE_A_LATERAL_DOMAIN_ROBUSTNESS"
        ),
        "coordinate_frame": "fixed Lumerical x=crystal b, y=crystal a",
        "base_case": str(args.base_case_dir.resolve()),
        "refined_case": str(args.refined_case_dir.resolve()),
        "base_thermal_discretization": discretization(base_summary, 60.0),
        "refined_thermal_discretization": discretization(refined_summary, 72.0),
        "metrics": metrics,
        "gates_lt_1_percent": gates,
        "interpretation": (
            "Lateral/far-boundary power is numerical truncation flux and is not "
            "interpreted as a physical heat-path fraction."
        ),
        "plot": str(plot_path.resolve()),
    }
    output_path = args.output_dir / "device_a_lateral_domain_check.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
