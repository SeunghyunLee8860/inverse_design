#!/usr/bin/env python3
"""Spatially decompose the circular-design increment DeltaQ = Qdisk - Qflat.

This script is post-processing only.  It never clips Q, applies a geometric
gain, rescales to flux absorption, or opens HEAT.
"""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from lumerical_api import Stage1Error, jsonable, write_json


FLAKE_Z_MIN_M = -100.0e-9
FLAKE_Z_MAX_M = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disk-dir", required=True)
    parser.add_argument("--flat-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--disk-radius-m", type=float, default=1.5e-6)
    parser.add_argument("--edge-half-width-cells", type=int, default=2)
    parser.add_argument("--z-boundary-cells", type=int, default=2)
    return parser.parse_args()


def integrate_xyz(q: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    trap = np.trapezoid
    return float(trap(trap(trap(q, z, axis=2), y, axis=1), x, axis=0))


def load_case(path: Path) -> dict[str, Any]:
    data_path = path / "q_on_physical.npz"
    summary_path = path / "fdtd_absorption_summary.json"
    if not data_path.is_file() or not summary_path.is_file():
        raise Stage1Error(f"missing case artifacts under {path}")
    data = np.load(data_path)
    result = {
        "x": np.asarray(data["x_m"], float).reshape(-1),
        "y": np.asarray(data["y_m"], float).reshape(-1),
        "z": np.asarray(data["z_m"], float).reshape(-1),
        "q": np.asarray(data["Q_on_W_m3"], float),
        "incident_power": float(np.asarray(data["incident_power_W"]).reshape(-1)[0]),
        "summary": json.loads(summary_path.read_text()),
        "path": str(path),
    }
    if result["q"].shape != (result["x"].size, result["y"].size, result["z"].size):
        raise Stage1Error(f"invalid Q shape in {data_path}: {result['q'].shape}")
    return result


def assert_same_grid(disk: dict[str, Any], flat: dict[str, Any]) -> None:
    for axis in ("x", "y", "z"):
        a = disk[axis]
        b = flat[axis]
        if a.shape != b.shape or not np.allclose(a, b, rtol=0.0, atol=1e-15):
            raise Stage1Error(f"{axis} grids differ: {a.shape} versus {b.shape}")
    if not np.isclose(
        disk["incident_power"], flat["incident_power"], rtol=1e-12, atol=0.0
    ):
        raise Stage1Error("incident powers differ")


def masked_integral(
    delta_q: np.ndarray,
    mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> float:
    return integrate_xyz(delta_q * np.asarray(mask, float), x, y, z)


def signed_norm(values: np.ndarray) -> TwoSlopeNorm:
    limit = float(max(abs(np.nanmin(values)), abs(np.nanmax(values))))
    if limit == 0:
        limit = 1.0
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def plot_map(
    path: Path,
    a: np.ndarray,
    b: np.ndarray,
    values: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.7, 5.1))
    image = ax.pcolormesh(
        a, b, np.asarray(values).T, shading="auto", cmap="coolwarm",
        norm=signed_norm(values),
    )
    fig.colorbar(image, ax=ax, label=r"$\Delta Q$ [W m$^{-3}$]")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "deltaq_spatial_summary.json"
    try:
        disk = load_case(Path(args.disk_dir).expanduser().resolve())
        flat = load_case(Path(args.flat_dir).expanduser().resolve())
        assert_same_grid(disk, flat)
        x, y, z = disk["x"], disk["y"], disk["z"]
        delta_q = disk["q"] - flat["q"]
        incident_power = disk["incident_power"]

        p_delta = integrate_xyz(delta_q, x, y, z)
        delta_a_q = p_delta / incident_power
        a_flux_disk = float(disk["summary"]["absorption"]["A_local_flux"])
        a_flux_flat = float(flat["summary"]["absorption"]["A_local_flux"])
        delta_a_flux = a_flux_disk - a_flux_flat
        excess = delta_a_q - delta_a_flux

        dx = float(np.median(np.diff(x)))
        dy = float(np.median(np.diff(y)))
        if not np.isclose(dx, dy, rtol=1e-9, atol=1e-15):
            raise Stage1Error(f"edge-band definition requires dx=dy; got {dx}, {dy}")
        edge_width = args.edge_half_width_cells * dx
        xx, yy = np.meshgrid(x, y, indexing="ij")
        radius = np.sqrt(xx**2 + yy**2)
        radial_masks = {
            "disk_interior": radius < args.disk_radius_m - edge_width,
            "disk_edge_band": np.abs(radius - args.disk_radius_m) <= edge_width,
            "exterior": radius > args.disk_radius_m + edge_width,
        }
        radial_sum = sum(mask.astype(int) for mask in radial_masks.values())
        if not np.all(radial_sum == 1):
            raise Stage1Error("radial masks do not partition every x/y sample")

        in_flake = (z >= FLAKE_Z_MIN_M - 1e-18) & (z <= FLAKE_Z_MAX_M + 1e-18)
        flake_z = z[in_flake]
        dz = float(np.median(np.diff(flake_z)))
        boundary_width = args.z_boundary_cells * dz
        z_masks = {
            "below_flake_padding": z < FLAKE_Z_MIN_M - 1e-18,
            "bottom_boundary_cells": (
                in_flake & (z <= FLAKE_Z_MIN_M + boundary_width + 1e-18)
            ),
            "flake_interior_cells": (
                in_flake
                & (z > FLAKE_Z_MIN_M + boundary_width + 1e-18)
                & (z < FLAKE_Z_MAX_M - boundary_width - 1e-18)
            ),
            "top_boundary_cells": (
                in_flake & (z >= FLAKE_Z_MAX_M - boundary_width - 1e-18)
            ),
            "above_flake_padding": z > FLAKE_Z_MAX_M + 1e-18,
        }
        z_sum = sum(mask.astype(int) for mask in z_masks.values())
        if not np.all(z_sum == 1):
            raise Stage1Error("z masks do not partition every z sample")

        radial_results = {}
        for name, mask_xy in radial_masks.items():
            power = masked_integral(delta_q, mask_xy[:, :, None], x, y, z)
            radial_results[name] = {
                "delta_power_W": power,
                "delta_absorptance": power / incident_power,
                "fraction_of_DeltaA_Q": (power / incident_power) / delta_a_q,
                "fraction_of_excess": (power / incident_power) / excess,
            }

        z_results = {}
        for name, mask_z in z_masks.items():
            power = masked_integral(delta_q, mask_z[None, None, :], x, y, z)
            z_results[name] = {
                "delta_power_W": power,
                "delta_absorptance": power / incident_power,
                "fraction_of_DeltaA_Q": (power / incident_power) / delta_a_q,
                "fraction_of_excess": (power / incident_power) / excess,
                "z_samples_m": z[mask_z].tolist(),
            }

        cross_results: dict[str, dict[str, float]] = {}
        for radial_name, mask_xy in radial_masks.items():
            cross_results[radial_name] = {}
            for z_name, mask_z in z_masks.items():
                power = masked_integral(
                    delta_q,
                    mask_xy[:, :, None] & mask_z[None, None, :],
                    x, y, z,
                )
                cross_results[radial_name][z_name] = power / incident_power

        np.savez_compressed(
            output / "delta_q_disk_minus_flat.npz",
            x_m=x, y_m=y, z_m=z, DeltaQ_W_m3=delta_q,
            incident_power_W=incident_power,
            disk_radius_m=args.disk_radius_m,
            edge_band_half_width_m=edge_width,
            disk_interior_mask=radial_masks["disk_interior"],
            disk_edge_band_mask=radial_masks["disk_edge_band"],
            exterior_mask=radial_masks["exterior"],
        )
        savable_masks = {
            name: mask.astype(np.uint8) for name, mask in z_masks.items()
        }
        np.savez_compressed(output / "delta_q_z_masks.npz", z_m=z, **savable_masks)

        iz = int(np.argmin(np.abs(z + 50e-9)))
        iy = int(np.argmin(np.abs(y)))
        plot_map(
            output / "delta_q_xy_midplane.png",
            x * 1e6, y * 1e6, delta_q[:, :, iz],
            xlabel="x [um]", ylabel="y [um]",
            title=f"DeltaQ at z={z[iz] * 1e9:.1f} nm",
        )
        plot_map(
            output / "delta_q_xz_center.png",
            x * 1e6, z * 1e9, delta_q[:, iy, :],
            xlabel="x [um]", ylabel="z [nm]",
            title="DeltaQ at y=0",
        )

        signed_distance = radius - args.disk_radius_m
        bin_edges = np.linspace(float(np.min(signed_distance)), float(np.max(signed_distance)), 121)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        edge_map = np.full((bin_centers.size, z.size), np.nan)
        edge_counts = np.zeros(bin_centers.size, dtype=int)
        for index in range(bin_centers.size):
            mask = (
                (signed_distance >= bin_edges[index])
                & (signed_distance < bin_edges[index + 1])
            )
            edge_counts[index] = int(np.count_nonzero(mask))
            if edge_counts[index]:
                edge_map[index, :] = np.mean(delta_q[mask, :], axis=0)
        fig, ax = plt.subplots(figsize=(7.2, 5.1))
        image = ax.pcolormesh(
            bin_centers * 1e6, z * 1e9, edge_map.T,
            shading="auto", cmap="coolwarm", norm=signed_norm(edge_map),
        )
        fig.colorbar(image, ax=ax, label=r"radial-bin mean $\Delta Q$ [W m$^{-3}$]")
        ax.axvline(0.0, color="k", lw=1.2, ls="--", label="disk edge")
        ax.axhline(FLAKE_Z_MIN_M * 1e9, color="k", lw=0.8, ls=":")
        ax.axhline(0.0, color="k", lw=0.8, ls=":")
        ax.set_xlabel("signed distance from disk edge r-R [um]")
        ax.set_ylabel("z [nm]")
        ax.set_title("DeltaQ versus disk-edge distance and depth")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "delta_q_edge_distance_vs_z.png", dpi=190)
        plt.close(fig)
        np.savez_compressed(
            output / "delta_q_edge_distance_vs_z.npz",
            edge_distance_m=bin_centers, z_m=z,
            DeltaQ_radial_bin_mean_W_m3=edge_map,
            radial_bin_sample_count=edge_counts,
        )

        with (output / "delta_q_region_matrix.csv").open("w", newline="") as handle:
            z_names = list(z_masks)
            writer = csv.writer(handle)
            writer.writerow(["radial_region", *z_names, "row_sum"])
            for radial_name in radial_masks:
                values = [cross_results[radial_name][name] for name in z_names]
                writer.writerow([radial_name, *values, sum(values)])

        largest_radial = max(
            radial_results, key=lambda name: abs(radial_results[name]["delta_absorptance"])
        )
        largest_z = max(
            z_results, key=lambda name: abs(z_results[name]["delta_absorptance"])
        )
        summary = {
            "status": "completed",
            "disk_case": disk["path"],
            "flat_case": flat["path"],
            "same_grid_verified": True,
            "no_clipping": True,
            "no_flux_gain": True,
            "metrics": {
                "DeltaA_Q": delta_a_q,
                "DeltaA_flux_local": delta_a_flux,
                "excess": excess,
                "A_local_flux_disk": a_flux_disk,
                "A_local_flux_flat": a_flux_flat,
                "incident_power_W": incident_power,
            },
            "mask_definitions": {
                "dx_m": dx, "dy_m": dy,
                "disk_radius_m": args.disk_radius_m,
                "radial_edge_half_width_cells": args.edge_half_width_cells,
                "radial_edge_half_width_m": edge_width,
                "flake_z_min_m": FLAKE_Z_MIN_M,
                "flake_z_max_m": FLAKE_Z_MAX_M,
                "flake_dz_m": dz,
                "z_boundary_cells_each_side": args.z_boundary_cells,
                "z_boundary_width_m": boundary_width,
            },
            "radial_decomposition": radial_results,
            "z_decomposition": z_results,
            "radial_by_z_delta_absorptance": cross_results,
            "largest_absolute_radial_DeltaQ_region": largest_radial,
            "largest_absolute_z_DeltaQ_region": largest_z,
            "interpretation_guard": (
                "DeltaA_flux is a scalar and has no unique spatial allocation. "
                "These maps localize DeltaQ; assigning the entire excess to a "
                "specific voxel requires the geometry/mesh controls."
            ),
        }
        write_json(summary_path, summary)
        print(json.dumps(jsonable(summary), indent=2), flush=True)
        return 0
    except Exception as exc:
        failure = {
            "status": "failed",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(summary_path, failure)
        print(json.dumps(failure, indent=2), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
