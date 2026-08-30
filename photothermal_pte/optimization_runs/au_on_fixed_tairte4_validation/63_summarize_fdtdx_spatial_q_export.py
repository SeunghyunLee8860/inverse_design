#!/usr/bin/env python3
"""Independently audit and publish the native-Yee spatial-Q export."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


MATERIALS = ("au", "tairte4", "sio2")
COMPONENTS = ("x", "y", "z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), np.finfo(float).tiny)


def _nearest_mismatch(first: np.ndarray, second: np.ndarray) -> float:
    """Largest distance from either 1-D coordinate set to its nearest peer."""
    def directed(a: np.ndarray, b: np.ndarray) -> float:
        indices = np.searchsorted(b, a)
        low = np.clip(indices - 1, 0, len(b) - 1)
        high = np.clip(indices, 0, len(b) - 1)
        return float(np.max(np.minimum(np.abs(a - b[low]), np.abs(a - b[high]))))

    return max(directed(first, second), directed(second, first))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--raw-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    result_path = args.result_json.resolve()
    raw_path = args.raw_npz.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    raw_sha = _sha256(raw_path)
    if raw_sha != result["raw_artifact"]["sha256"]:
        raise RuntimeError(
            f"Raw artifact SHA mismatch: {raw_sha} != {result['raw_artifact']['sha256']}"
        )

    rows: list[dict[str, object]] = []
    material_power: dict[str, float] = {}
    component_maps: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    max_reintegration_error = 0.0
    max_volume_factorization_error = 0.0
    max_coordinate_stagger_m = 0.0
    finite_nonnegative = True
    coordinate_contract = True

    with np.load(raw_path, allow_pickle=False) as raw:
        for material in MATERIALS:
            q = np.asarray(raw[f"Q_{material}_W_m3"], dtype=np.float64)
            volume = np.asarray(raw[f"dual_volume_{material}_m3"], dtype=np.float64)
            expected = np.asarray(result["component_power_W"][f"{material}_xyz"])
            if q.shape != volume.shape or q.shape[0] != 3:
                raise RuntimeError(f"Invalid {material} Q/volume shape: {q.shape}, {volume.shape}")
            finite_nonnegative &= bool(
                np.all(np.isfinite(q)) and np.all(np.isfinite(volume))
                and np.all(q >= 0.0) and np.all(volume > 0.0)
            )
            integrated = np.sum(q * volume, axis=(1, 2, 3))
            errors = np.abs(integrated - expected) / np.maximum(
                np.abs(expected), np.finfo(float).tiny
            )
            max_reintegration_error = max(max_reintegration_error, float(np.max(errors)))
            material_power[material] = float(np.sum(integrated))

            coordinates_by_component: dict[str, dict[str, np.ndarray]] = {}
            for component_index, component in enumerate(COMPONENTS):
                coordinates_by_component[component] = {}
                metrics = []
                for axis_index, axis in enumerate(COMPONENTS):
                    coordinate = np.asarray(raw[f"{material}_{component}_{axis}_m"])
                    width = np.asarray(raw[f"dual_width_{material}_{component}_{axis}_m"])
                    coordinates_by_component[component][axis] = coordinate
                    metrics.append(width)
                    coordinate_contract &= bool(
                        len(coordinate) == q.shape[axis_index + 1]
                        and len(width) == len(coordinate)
                        and np.all(np.diff(coordinate) > 0.0)
                        and np.all(width > 0.0)
                    )
                reconstructed_volume = (
                    metrics[0][:, None, None]
                    * metrics[1][None, :, None]
                    * metrics[2][None, None, :]
                )
                volume_error = float(
                    np.max(np.abs(reconstructed_volume - volume[component_index]))
                    / max(float(np.max(volume[component_index])), np.finfo(float).tiny)
                )
                max_volume_factorization_error = max(
                    max_volume_factorization_error, volume_error
                )
                column_power = np.sum(
                    q[component_index] * volume[component_index], axis=2
                )
                lateral_area = metrics[0][:, None] * metrics[1][None, :]
                q_areal = column_power / lateral_area
                component_maps[(material, component)] = (
                    coordinates_by_component[component]["x"],
                    coordinates_by_component[component]["y"],
                    q_areal,
                )
                rows.append(
                    {
                        "material": material,
                        "component": component,
                        "shape": "x".join(map(str, q[component_index].shape)),
                        "power_W": integrated[component_index],
                        "fraction_of_total_Q": integrated[component_index] / result["P_Q_W"],
                        "reintegration_relative_error": errors[component_index],
                        "dual_volume_factorization_relative_error": volume_error,
                        "x_min_m": coordinates_by_component[component]["x"][0],
                        "x_max_m": coordinates_by_component[component]["x"][-1],
                        "y_min_m": coordinates_by_component[component]["y"][0],
                        "y_max_m": coordinates_by_component[component]["y"][-1],
                        "z_min_m": coordinates_by_component[component]["z"][0],
                        "z_max_m": coordinates_by_component[component]["z"][-1],
                    }
                )

            for axis in COMPONENTS:
                for first, second in (("x", "y"), ("x", "z"), ("y", "z")):
                    max_coordinate_stagger_m = max(
                        max_coordinate_stagger_m,
                        _nearest_mismatch(
                            coordinates_by_component[first][axis],
                            coordinates_by_component[second][axis],
                        ),
                    )

    total_reintegrated = sum(material_power.values())
    total_error = _relative(total_reintegrated, result["P_Q_W"])
    gates = {
        "raw_SHA_matches_generation_record": True,
        "finite_nonnegative_Q_and_positive_dual_volume": finite_nonnegative,
        "coordinate_shape_monotonicity_contract": coordinate_contract,
        "dual_volume_axis_factorization_lt_1e-6": max_volume_factorization_error < 1.0e-6,
        "component_Q_reintegration_lt_1e-6": max_reintegration_error < 1.0e-6,
        "total_Q_reintegration_lt_1e-6": total_error < 1.0e-6,
        "generation_Q_flux_closure_lt_0p5pct": result["Q_flux_closure_relative"] < 0.005,
        "generation_late_window_change_lt_0p5pct": result["late_window_relative_change"] < 0.005,
        "no_clipping_smoothing_gain_or_rescaling": True,
    }
    passed = all(gates.values())
    status = (
        "VALIDATED_FDTDX_SUBSTRATE_SPATIAL_NATIVE_YEE_Q_ARTIFACT"
        if passed
        else "FAILED_FDTDX_SUBSTRATE_SPATIAL_NATIVE_YEE_Q_ARTIFACT"
    )

    csv_path = output / "fdtdx_spatial_native_yee_q_components.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
    for row_index, material in enumerate(MATERIALS):
        for col_index, component in enumerate(COMPONENTS):
            axis = axes[row_index, col_index]
            x, y, image = component_maps[(material, component)]
            positive = image[image > 0.0]
            maximum = float(np.max(positive))
            minimum = max(float(np.min(positive)), maximum * 1.0e-7)
            mesh = axis.pcolormesh(
                x * 1.0e6,
                y * 1.0e6,
                image.T,
                shading="nearest",
                norm=LogNorm(vmin=minimum, vmax=maximum),
            )
            axis.set_aspect("equal")
            axis.set_title(f"{material}: Q{component} depth-integrated")
            axis.set_xlabel("x=b (um)")
            axis.set_ylabel("y=a (um)")
            fig.colorbar(mesh, ax=axis, label="W/m2 (log display)")
    component_plot_path = output / "fdtdx_spatial_native_yee_q_components.png"
    fig.savefig(component_plot_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    material_values = [material_power[name] for name in MATERIALS]
    axes[0].bar(MATERIALS, np.asarray(material_values) * 1.0e15)
    axes[0].set_ylabel("absorbed power (fW)")
    axes[0].set_title("Material-resolved absorption")
    component_values = [row["power_W"] for row in rows]
    component_labels = [f"{row['material']} Q{row['component']}" for row in rows]
    axes[1].barh(component_labels, np.asarray(component_values) * 1.0e15)
    axes[1].set_xlabel("absorbed power (fW)")
    axes[1].set_title("Native Yee component breakdown")
    power_plot_path = output / "fdtdx_spatial_q_power_breakdown.png"
    fig.savefig(power_plot_path, dpi=180)
    plt.close(fig)

    summary = {
        "status": status,
        "scope": (
            "independent SHA, coordinate, dual-width/volume, and Q*dV audit of the "
            "16-period/4-window FDTDX Au/TaIrTe4/SiO2 spatial native-Yee heat source; "
            "no optical-to-thermal remap, thermal solve, PTE, adjoint, or optimization"
        ),
        "source_result_json": str(result_path),
        "raw_artifact": {
            "path": str(raw_path),
            "bytes": raw_path.stat().st_size,
            "sha256": raw_sha,
            "committed_to_git": False,
        },
        "P_Q_generation_W": result["P_Q_W"],
        "P_Q_reintegrated_W": total_reintegrated,
        "total_Q_reintegration_relative_error": total_error,
        "maximum_component_Q_reintegration_relative_error": max_reintegration_error,
        "maximum_dual_volume_factorization_relative_error": max_volume_factorization_error,
        "maximum_component_coordinate_stagger_m": max_coordinate_stagger_m,
        "material_power_W": material_power,
        "material_fraction_of_total_Q": {
            name: value / total_reintegrated for name, value in material_power.items()
        },
        "generation_Q_flux_closure_relative": result["Q_flux_closure_relative"],
        "generation_late_window_relative_change": result["late_window_relative_change"],
        "coordinate_contract": result["component_coordinate_contract"],
        "gates": gates,
        "next_gate": (
            "conservative overlap remap of every component-native dual cell into one "
            "explicit Au/TaIrTe4/SiO2 thermal grid, followed by power conservation audit"
        ),
    }
    summary_path = output / "fdtdx_spatial_native_yee_q_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = f"""# FDTDX substrate spatial native-Yee Q artifact

Status: **{status}**

This checkpoint exports the actual spatial `Qx`, `Qy`, and `Qz` arrays for
Au, TaIrTe4, and lossy SiO2 from the validated 16-period/4-window GPU FDTDX
forward. Each component retains its own staggered physical coordinates,
axis-wise dual widths, and dual volumes. No array-index pairing between
different Yee components is used.

| metric | value |
|---|---:|
| total P_Q | {result['P_Q_W']:.12e} W |
| independently reintegrated P_Q | {total_reintegrated:.12e} W |
| total reintegration error | {100*total_error:.9f}% |
| worst component reintegration error | {100*max_reintegration_error:.9f}% |
| dual-volume factorization error | {100*max_volume_factorization_error:.9f}% |
| matched-volume Q/flux closure | {100*result['Q_flux_closure_relative']:.6f}% |
| late-window change | {100*result['late_window_relative_change']:.6f}% |
| runtime | {result['runtime_seconds']:.3f} s |

Material powers are Au `{material_power['au']:.12e} W`, TaIrTe4
`{material_power['tairte4']:.12e} W`, and SiO2
`{material_power['sio2']:.12e} W`. The raw NPZ is not committed to Git. Its
path, byte size, and SHA-256 are recorded in the manifest.

The maps use a logarithmic color display only; the stored and integrated Q
arrays are unmodified. No clipping, smoothing, gain, global rescaling, or
polarization matching is performed.

This is not yet a coupled thermal/PTE validation. The next fail-closed gate is
an overlap-based conservative remap of every component-native dual cell into
one explicit Au/TaIrTe4/SiO2 thermal grid. Only after that mapping preserves
power may the Maxwell source replace the fixed-Q source in the coupled
thermal/weighting operator.
"""
    report_path = output / "FDTDX_SPATIAL_NATIVE_YEE_Q_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    published = (result_path, summary_path, csv_path, component_plot_path, power_plot_path, report_path)
    manifest = {
        "status": status,
        "raw_artifact": summary["raw_artifact"],
        "published": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in published
        ],
    }
    manifest_path = output / "RAW_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
