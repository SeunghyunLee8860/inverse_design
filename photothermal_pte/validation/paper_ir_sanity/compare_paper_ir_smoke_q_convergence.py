#!/usr/bin/env python3
"""Compare saved 1.2 ps and 4 ps paper-IR smoke Q without FDTD."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[3]
WAIST_M = 2.0e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        text=True,
    ).strip()


def trapezoid_weights(coordinate: np.ndarray) -> np.ndarray:
    values = np.asarray(coordinate, float)
    weights = np.empty_like(values)
    weights[0] = 0.5 * (values[1] - values[0])
    weights[-1] = 0.5 * (values[-1] - values[-2])
    weights[1:-1] = 0.5 * (values[2:] - values[:-2])
    return weights


def load_q(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as raw:
        coordinates = {
            axis: np.asarray(raw[f"{axis}_m"], float)
            for axis in "xyz"
        }
        suffix = (
            "common_grid_W_m3"
            if "Q_common_grid_W_m3" in raw.files
            else "native_W_m3"
        )
        components = {
            axis: np.asarray(raw[f"Q{axis}_{suffix}"], float)
            for axis in "xyz"
        }
        total = np.asarray(raw[f"Q_{suffix}"], float)
    expected_shape = tuple(coordinates[axis].size for axis in "xyz")
    if total.shape != expected_shape:
        raise RuntimeError(f"Q shape {total.shape} != {expected_shape}")
    if not all(
        np.array_equal(total.shape, components[axis].shape)
        for axis in "xyz"
    ):
        raise RuntimeError("component Q shape mismatch")
    if np.any(~np.isfinite(total)) or any(
        np.any(~np.isfinite(components[axis])) for axis in "xyz"
    ):
        raise RuntimeError("Q contains NaN or Inf")
    return {
        "coordinates": coordinates,
        "components": components,
        "total": total,
        "suffix": suffix,
    }


def volume_weights(coordinates: dict[str, np.ndarray]) -> np.ndarray:
    return (
        trapezoid_weights(coordinates["x"])[:, None, None]
        * trapezoid_weights(coordinates["y"])[None, :, None]
        * trapezoid_weights(coordinates["z"])[None, None, :]
    )


def integrate(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(np.asarray(values, float) * weights))


def relative_change(first: float, second: float) -> float:
    return abs(second - first) / max(abs(first), np.finfo(float).tiny)


def weighted_correlation(
    first: np.ndarray,
    second: np.ndarray,
    weights: np.ndarray,
) -> float:
    weight = weights / np.sum(weights)
    mean_first = float(np.sum(weight * first))
    mean_second = float(np.sum(weight * second))
    centered_first = first - mean_first
    centered_second = second - mean_second
    numerator = float(np.sum(weight * centered_first * centered_second))
    denominator = np.sqrt(
        float(np.sum(weight * centered_first**2))
        * float(np.sum(weight * centered_second**2))
    )
    return numerator / denominator


def cosine_similarity(
    first: np.ndarray,
    second: np.ndarray,
    weights: np.ndarray,
) -> float:
    numerator = float(np.sum(weights * first * second))
    denominator = np.sqrt(
        float(np.sum(weights * first**2))
        * float(np.sum(weights * second**2))
    )
    return numerator / denominator


def distribution_metrics(
    q: np.ndarray,
    coordinates: dict[str, np.ndarray],
    weights: np.ndarray,
) -> dict[str, Any]:
    power = integrate(q, weights)
    normalized = q / power
    meshes = np.meshgrid(
        coordinates["x"],
        coordinates["y"],
        coordinates["z"],
        indexing="ij",
        sparse=True,
    )
    centroid = {
        axis: integrate(normalized * meshes[index], weights)
        for index, axis in enumerate("xyz")
    }
    normal = (meshes[1] - meshes[0]) / np.sqrt(2.0)
    tangent = (meshes[0] + meshes[1]) / np.sqrt(2.0)
    centroid_normal = integrate(normalized * normal, weights)
    centroid_tangent = integrate(normalized * tangent, weights)
    sigma = {
        axis: np.sqrt(
            integrate(
                normalized * (meshes[index] - centroid[axis]) ** 2,
                weights,
            )
        )
        for index, axis in enumerate("xyz")
    }
    sigma_normal = np.sqrt(
        integrate(
            normalized * (normal - centroid_normal) ** 2,
            weights,
        )
    )
    sigma_tangent = np.sqrt(
        integrate(
            normalized * (tangent - centroid_tangent) ** 2,
            weights,
        )
    )
    hotspot = np.unravel_index(int(np.argmax(q)), q.shape)
    return {
        "power_W": power,
        "normalized": normalized,
        "centroid_m": {
            **centroid,
            "edge_normal": centroid_normal,
            "edge_tangent": centroid_tangent,
        },
        "sigma_m": {
            **sigma,
            "edge_normal": sigma_normal,
            "edge_tangent": sigma_tangent,
        },
        "hotspot": {
            "index_xyz": [int(value) for value in hotspot],
            "x_m": float(coordinates["x"][hotspot[0]]),
            "y_m": float(coordinates["y"][hotspot[1]]),
            "z_m": float(coordinates["z"][hotspot[2]]),
            "Q_W_m3": float(q[hotspot]),
        },
    }


def edge_profile(
    q: np.ndarray,
    coordinates: dict[str, np.ndarray],
    *,
    tangent_window_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    wx = trapezoid_weights(coordinates["x"])
    wy = trapezoid_weights(coordinates["y"])
    wz = trapezoid_weights(coordinates["z"])
    areal = np.sum(q * wz[None, None, :], axis=2)
    x, y = np.meshgrid(
        coordinates["x"],
        coordinates["y"],
        indexing="ij",
    )
    normal = (y - x) / np.sqrt(2.0)
    tangent = (x + y) / np.sqrt(2.0)
    step = min(float(np.median(np.diff(coordinates[axis]))) for axis in "xy")
    lower = float(np.min(normal))
    upper = float(np.max(normal))
    edges = np.arange(lower, upper + 1.01 * step, step)
    centers = 0.5 * (edges[:-1] + edges[1:])
    area = wx[:, None] * wy[None, :]
    selected = np.abs(tangent) <= tangent_window_m
    indices = np.digitize(normal[selected], edges) - 1
    profile = np.full(centers.size, np.nan)
    for index in range(centers.size):
        mask = indices == index
        if np.any(mask):
            local_weight = area[selected][mask]
            profile[index] = float(
                np.sum(areal[selected][mask] * local_weight)
                / np.sum(local_weight)
            )
    finite = np.isfinite(profile)
    integral = float(np.trapezoid(profile[finite], centers[finite]))
    profile[finite] /= integral
    return centers, profile


def profile_comparison(
    coordinate: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> dict[str, Any]:
    finite = np.isfinite(first) & np.isfinite(second)
    a = first[finite]
    b = second[finite]
    n = coordinate[finite]
    nrmse = float(np.linalg.norm(b - a) / np.linalg.norm(a))
    correlation = float(np.corrcoef(a, b)[0, 1])

    def moments(values: np.ndarray) -> dict[str, float]:
        weights = np.maximum(values, 0.0)
        weights /= np.trapezoid(weights, n)
        center = float(np.trapezoid(n * weights, n))
        sigma = float(
            np.sqrt(np.trapezoid((n - center) ** 2 * weights, n))
        )
        peak = int(np.argmax(values))
        return {
            "center_m": center,
            "sigma_m": sigma,
            "peak_location_m": float(n[peak]),
            "peak_value_normalized_per_m": float(values[peak]),
        }

    return {
        "NRMSE": nrmse,
        "correlation": correlation,
        "first": moments(a),
        "second": moments(b),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-1p2ps", type=Path, required=True)
    parser.add_argument("--artifact-4ps", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tangent-window-um", type=float, default=2.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    first = load_q(args.artifact_1p2ps)
    second = load_q(args.artifact_4ps)
    for axis in "xyz":
        if not np.array_equal(
            first["coordinates"][axis],
            second["coordinates"][axis],
        ):
            raise RuntimeError(f"{axis} coordinates differ; no interpolation")
    coordinates = first["coordinates"]
    weights = volume_weights(coordinates)
    first_metrics = distribution_metrics(
        first["total"],
        coordinates,
        weights,
    )
    second_metrics = distribution_metrics(
        second["total"],
        coordinates,
        weights,
    )
    component_power = {}
    for axis in "xyz":
        p_first = integrate(first["components"][axis], weights)
        p_second = integrate(second["components"][axis], weights)
        component_power[axis] = {
            "P_1p2ps_W": p_first,
            "P_4ps_W": p_second,
            "relative_change": relative_change(p_first, p_second),
        }

    normalized_first = first_metrics.pop("normalized")
    normalized_second = second_metrics.pop("normalized")
    spatial_nrmse = float(
        np.sqrt(
            np.sum(weights * (normalized_second - normalized_first) ** 2)
            / np.sum(weights * normalized_first**2)
        )
    )
    correlation = weighted_correlation(
        normalized_first,
        normalized_second,
        weights,
    )
    cosine = cosine_similarity(
        normalized_first,
        normalized_second,
        weights,
    )
    profile_coordinate, profile_first = edge_profile(
        first["total"],
        coordinates,
        tangent_window_m=args.tangent_window_um * 1e-6,
    )
    profile_coordinate_second, profile_second = edge_profile(
        second["total"],
        coordinates,
        tangent_window_m=args.tangent_window_um * 1e-6,
    )
    if not np.array_equal(profile_coordinate, profile_coordinate_second):
        raise RuntimeError("edge-normal profile coordinates differ")
    profile = profile_comparison(
        profile_coordinate,
        profile_first,
        profile_second,
    )
    centroid_shift = np.sqrt(
        sum(
            (
                second_metrics["centroid_m"][axis]
                - first_metrics["centroid_m"][axis]
            )
            ** 2
            for axis in "xyz"
        )
    )
    hotspot_shift = np.sqrt(
        sum(
            (
                second_metrics["hotspot"][f"{axis}_m"]
                - first_metrics["hotspot"][f"{axis}_m"]
            )
            ** 2
            for axis in "xyz"
        )
    )
    sigma_change = {
        axis: relative_change(
            first_metrics["sigma_m"][axis],
            second_metrics["sigma_m"][axis],
        )
        for axis in (
            "x",
            "y",
            "z",
            "edge_normal",
            "edge_tangent",
        )
    }
    total_power_change = relative_change(
        first_metrics["power_W"],
        second_metrics["power_W"],
    )
    primary_gate = total_power_change < 0.005 and spatial_nrmse < 0.005
    payload = {
        "status": (
            "VALIDATED_DIAGNOSTIC_Q_OBSERVABLE_CONVERGENCE"
            if primary_gate
            else "FAILED_DIAGNOSTIC_Q_OBSERVABLE_CONVERGENCE"
        ),
        "validated_for_diagnostic_heat_source": primary_gate,
        "promoted_to_production_Q": False,
        "FDTD_run": False,
        "comparison": "saved 1.2 ps versus saved 4 ps common-grid Q",
        "grid_contract": {
            "shape_xyz": list(first["total"].shape),
            "coordinates_bitwise_equal": True,
            "bounds_m": {
                axis: [
                    float(coordinates[axis][0]),
                    float(coordinates[axis][-1]),
                ]
                for axis in "xyz"
            },
            "interpolation_used": False,
            "volume_quadrature": "tensor-product trapezoid/dual-cell weights",
        },
        "power": {
            "P_Q_1p2ps_W": first_metrics["power_W"],
            "P_Q_4ps_W": second_metrics["power_W"],
            "relative_change": total_power_change,
            "components": component_power,
        },
        "normalized_spatial_Q": {
            "definition": "Q divided by integral(Q dV)",
            "volume_weighted_NRMSE": spatial_nrmse,
            "volume_weighted_Pearson_correlation": correlation,
            "volume_weighted_cosine_similarity": cosine,
        },
        "distribution_1p2ps": first_metrics,
        "distribution_4ps": second_metrics,
        "centroid_shift_m": centroid_shift,
        "centroid_shift_over_2um_waist": centroid_shift / WAIST_M,
        "sigma_relative_change": sigma_change,
        "hotspot_shift_m": hotspot_shift,
        "edge_normal_profile": {
            "tangent_window_m": args.tangent_window_um * 1e-6,
            **profile,
        },
        "acceptance": {
            "P_Q_relative_change_lt_0p5_percent": total_power_change < 0.005,
            "normalized_spatial_Q_NRMSE_lt_0p5_percent": (
                spatial_nrmse < 0.005
            ),
            "primary_all": primary_gate,
            "auto_shutoff_gate": {
                "passed": False,
                "1p2ps_final": 1.81076e-5,
                "4ps_final": 1.80982e-5,
                "threshold": 1.0e-5,
                "kept_separate_from_observable_Q_convergence": True,
            },
        },
        "artifacts": {
            "1p2ps": {
                "path": str(args.artifact_1p2ps.resolve()),
                "size_bytes": args.artifact_1p2ps.stat().st_size,
                "sha256": sha256(args.artifact_1p2ps),
            },
            "4ps": {
                "path": str(args.artifact_4ps.resolve()),
                "size_bytes": args.artifact_4ps.stat().st_size,
                "sha256": sha256(args.artifact_4ps),
            },
        },
        "generation_commit": git_commit(),
    }
    (args.output_dir / "q_observable_convergence.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (
        args.output_dir / "q_component_power_convergence.csv"
    ).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "component",
                "P_1p2ps_W",
                "P_4ps_W",
                "relative_change",
            ],
        )
        writer.writeheader()
        for axis in "xyz":
            writer.writerow({"component": axis, **component_power[axis]})
        writer.writerow(
            {
                "component": "total",
                "P_1p2ps_W": first_metrics["power_W"],
                "P_4ps_W": second_metrics["power_W"],
                "relative_change": total_power_change,
            }
        )
    np.savez(
        args.output_dir / "q_observable_convergence_profiles.npz",
        edge_normal_coordinate_m=profile_coordinate,
        normalized_profile_1p2ps=profile_first,
        normalized_profile_4ps=profile_second,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.2),
        constrained_layout=True,
    )
    axes[0].plot(
        profile_coordinate * 1e6,
        profile_first * 1e-6,
        label="1.2 ps",
    )
    axes[0].plot(
        profile_coordinate * 1e6,
        profile_second * 1e-6,
        "--",
        label="4 ps",
    )
    axes[0].axvline(0.0, color="black", linewidth=0.8, linestyle=":")
    axes[0].set(
        xlabel="edge-normal n (µm)",
        ylabel="normalized profile (1/µm)",
        title=f"Edge-normal profile; NRMSE={profile['NRMSE']:.3%}",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    labels = ["P_Q", "Qx", "Qy", "Qz", "spatial Q"]
    values = [
        total_power_change,
        component_power["x"]["relative_change"],
        component_power["y"]["relative_change"],
        component_power["z"]["relative_change"],
        spatial_nrmse,
    ]
    axes[1].bar(labels, np.asarray(values) * 100.0)
    axes[1].axhline(0.5, color="red", linestyle="--", label="0.5%")
    axes[1].set(
        ylabel="relative change / NRMSE (%)",
        title="Saved-Q observable convergence",
    )
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    figure.savefig(
        args.output_dir / "q_observable_convergence.png",
        dpi=180,
    )
    plt.close(figure)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if primary_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
