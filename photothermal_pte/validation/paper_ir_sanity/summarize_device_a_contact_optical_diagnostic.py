#!/usr/bin/env python3
"""Compare Device-A s0 E||a optical Q with digitized Au/Ti on versus off."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse


DOMINANCE_THRESHOLD = 0.10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dual_widths(values: np.ndarray, bounds: list[float]) -> np.ndarray:
    coordinates = np.asarray(values, float)
    edges = np.empty(coordinates.size + 1)
    edges[1:-1] = 0.5 * (coordinates[:-1] + coordinates[1:])
    edges[0], edges[-1] = map(float, bounds)
    return np.diff(np.clip(edges, float(bounds[0]), float(bounds[1])))


def dual_edges(values: np.ndarray, bounds: list[float]) -> np.ndarray:
    coordinates = np.asarray(values, float)
    edges = np.empty(coordinates.size + 1)
    edges[1:-1] = 0.5 * (coordinates[:-1] + coordinates[1:])
    edges[0], edges[-1] = map(float, bounds)
    return np.clip(edges, float(bounds[0]), float(bounds[1]))


def overlap_matrix(target: np.ndarray, source: np.ndarray) -> sparse.csr_matrix:
    rows, columns, values = [], [], []
    source_index = 0
    for target_index in range(target.size - 1):
        while source_index + 1 < source.size and source[source_index + 1] <= target[target_index]:
            source_index += 1
        index = source_index
        while index + 1 < source.size and source[index] < target[target_index + 1]:
            overlap = min(target[target_index + 1], source[index + 1]) - max(
                target[target_index], source[index]
            )
            if overlap > 0.0:
                rows.append(target_index)
                columns.append(index)
                values.append(float(overlap))
            index += 1
    return sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(target.size - 1, source.size - 1),
    )


def remap_q2d(source: dict[str, object], target: dict[str, object]) -> tuple[np.ndarray, float]:
    source_x_edges = dual_edges(source["x_m"], source["x_bounds_m"])
    source_y_edges = dual_edges(source["y_m"], source["y_bounds_m"])
    target_x_edges = dual_edges(target["x_m"], target["x_bounds_m"])
    target_y_edges = dual_edges(target["y_m"], target["y_bounds_m"])
    overlap_x = overlap_matrix(target_x_edges, source_x_edges)
    overlap_y = overlap_matrix(target_y_edges, source_y_edges)
    source_q = np.asarray(source["q2d_W_m2"], float)
    target_energy = overlap_x @ source_q @ overlap_y.T
    target_area = np.diff(target_x_edges)[:, None] * np.diff(target_y_edges)[None, :]
    mapped = np.asarray(target_energy) / target_area
    source_power = float(
        np.sum(source_q * np.diff(source_x_edges)[:, None] * np.diff(source_y_edges)[None, :])
    )
    target_power = float(np.sum(mapped * target_area))
    error = abs(target_power - source_power) / max(abs(source_power), np.finfo(float).tiny)
    return mapped, error


def load_q(path: Path) -> dict[str, object]:
    case = json.loads((path / "case_result.json").read_text())
    with np.load(path / "finite_q_on_artifact.npz") as raw:
        metadata = json.loads(str(raw["metadata_json"][0]))
        bounds = metadata["realized_pre_run_contract"]["geometry"][
            "pabs_nominal_control_volume_bounds_m"
        ]
        x = np.asarray(raw["x_m"], float)
        y = np.asarray(raw["y_m"], float)
        z = np.asarray(raw["z_m"], float)
        wx, wy, wz = (
            dual_widths(x, bounds["x"]),
            dual_widths(y, bounds["y"]),
            dual_widths(z, bounds["z"]),
        )
        volume = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
        mask_key = (
            "TaIrTe4_support_mask"
            if "TaIrTe4_support_mask" in raw.files
            else "exact_flake_mask"
        )
        mask = np.asarray(raw[mask_key], bool)
        q = np.asarray(raw["Q_on_W_m3"], float) * mask
        components = {
            component: float(np.sum(np.asarray(raw[f"Q{component}_W_m3"], float) * mask * volume))
            for component in ("x", "y", "z")
        }
    q2d = np.sum(q * wz[None, None, :], axis=2)
    return {
        "path": path,
        "case": case,
        "x_m": x,
        "y_m": y,
        "x_bounds_m": bounds["x"],
        "y_bounds_m": bounds["y"],
        "wx_m": wx,
        "wy_m": wy,
        "q2d_W_m2": q2d,
        "TaIrTe4_power_W": float(np.sum(q * volume)),
        "component_power_W": components,
        "raw_total_P_Q_W": float(case["run_result"]["P_Q_W"]),
        "closure": float(case["run_result"]["six_face_relative_closure"]),
        "auto_shutoff": float(case["run_result"]["auto_shutoff"]["final_value"]),
    }


def thermal_current(path: Path) -> float:
    summary = json.loads((path / "summary.json").read_text())
    return float(summary["PTE_current_A_at_285uW_incident"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--on-optical", type=Path, required=True)
    parser.add_argument("--off-optical", type=Path, required=True)
    parser.add_argument("--on-a-isolated", type=Path, required=True)
    parser.add_argument("--off-a-isolated", type=Path, required=True)
    parser.add_argument("--on-a-perfect", type=Path, required=True)
    parser.add_argument("--off-a-perfect", type=Path, required=True)
    parser.add_argument("--b-isolated", type=Path, required=True)
    parser.add_argument("--b-perfect", type=Path, required=True)
    parser.add_argument("--geometry-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    on, off = load_q(args.on_optical), load_q(args.off_optical)
    coordinate_grids_identical = all(
        np.array_equal(on[coordinate], off[coordinate])
        for coordinate in ("x_m", "y_m")
    )
    remap_error = 0.0
    if not coordinate_grids_identical:
        mapped_off, remap_error = remap_q2d(off, on)
        off = dict(off)
        off["q2d_W_m2"] = mapped_off
        off["x_m"], off["y_m"] = on["x_m"], on["y_m"]
        off["wx_m"], off["wy_m"] = on["wx_m"], on["wy_m"]
    geometry = json.loads(args.geometry_contract.read_text())
    frozen = on["case"]["pre_run_contract"]["geometry"]["digitized_device_a_contract"]
    shift = np.asarray(frozen["simulation_origin_shift_um"], float)
    midpoint = np.asarray(geometry["off_axis_edge_midpoint_code_um"], float) + shift
    normal = np.asarray(geometry["off_axis_edge_unit_inward_normal_code"], float)
    normal /= np.linalg.norm(normal)
    tangent = np.asarray(geometry["off_axis_edge_unit_tangent_code"], float)
    tangent /= np.linalg.norm(tangent)
    indices = geometry["off_axis_edge_vertex_indices"]
    vertices = np.asarray(geometry["flake_vertices_code_um"], float) + shift
    endpoint_t = np.sort((vertices[indices] - midpoint) @ tangent)
    xx, yy = np.meshgrid(np.asarray(on["x_m"]) * 1e6, np.asarray(on["y_m"]) * 1e6, indexing="ij")
    delta = np.stack((xx - midpoint[0], yy - midpoint[1]), axis=-1)
    distance_n = delta @ normal
    coordinate_t = delta @ tangent
    edge_band = (
        (np.abs(distance_n) <= 1.0)
        & (coordinate_t >= endpoint_t[0])
        & (coordinate_t <= endpoint_t[1])
    )
    area = np.asarray(on["wx_m"])[:, None] * np.asarray(on["wy_m"])[None, :]
    metrics: dict[str, object] = {}
    normalized = {}
    for label, data in (("AuTi_on", on), ("AuTi_off", off)):
        q2d = np.asarray(data["q2d_W_m2"], float)
        power = float(np.sum(q2d * area))
        normalized[label] = q2d / power
        metrics[label] = {
            "raw_total_P_Q_W": data["raw_total_P_Q_W"],
            "TaIrTe4_power_W": power,
            "component_power_W": data["component_power_W"],
            "edge_band_half_width_um": 1.0,
            "edge_localized_TaIrTe4_power_fraction": float(np.sum(q2d[edge_band] * area[edge_band]) / power),
            "six_face_closure": data["closure"],
            "auto_shutoff": data["auto_shutoff"],
        }
    difference = normalized["AuTi_off"] - normalized["AuTi_on"]
    weighted_nrmse = float(
        np.sqrt(np.sum(difference**2 * area) / np.sum(normalized["AuTi_on"] ** 2 * area))
    )
    edge_fraction_change = abs(
        metrics["AuTi_off"]["edge_localized_TaIrTe4_power_fraction"]
        - metrics["AuTi_on"]["edge_localized_TaIrTe4_power_fraction"]
    ) / max(abs(metrics["AuTi_on"]["edge_localized_TaIrTe4_power_fraction"]), 1e-30)
    bins = np.linspace(-4.0, 4.0, 65)
    centers = 0.5 * (bins[:-1] + bins[1:])
    profiles = {}
    tangent_selection = (coordinate_t >= endpoint_t[0]) & (coordinate_t <= endpoint_t[1])
    for label in ("AuTi_on", "AuTi_off"):
        profile = np.full(centers.shape, np.nan)
        for index in range(centers.size):
            selected = tangent_selection & (distance_n >= bins[index]) & (distance_n < bins[index + 1])
            if np.any(selected):
                profile[index] = float(
                    np.sum(normalized[label][selected] * area[selected]) / np.sum(area[selected])
                )
        profiles[label] = profile
    currents = {}
    for scenario, on_a, off_a, b in (
        ("isolated", args.on_a_isolated, args.off_a_isolated, args.b_isolated),
        ("perfect", args.on_a_perfect, args.off_a_perfect, args.b_perfect),
    ):
        ia_on, ia_off, ib = thermal_current(on_a), thermal_current(off_a), thermal_current(b)
        ratio_on, ratio_off = abs(ia_on) / abs(ib), abs(ia_off) / abs(ib)
        currents[scenario] = {
            "I_a_on_A": ia_on,
            "I_a_off_A": ia_off,
            "I_b_baseline_A": ib,
            "ratio_on": ratio_on,
            "ratio_off": ratio_off,
            "relative_Ia_change": abs(ia_off - ia_on) / abs(ia_on),
            "relative_ratio_change": abs(ratio_off - ratio_on) / ratio_on,
        }
    maximum_current_change = max(value["relative_Ia_change"] for value in currents.values())
    dominant = (
        weighted_nrmse >= DOMINANCE_THRESHOLD
        and edge_fraction_change >= DOMINANCE_THRESHOLD
        and maximum_current_change >= DOMINANCE_THRESHOLD
    )
    summary = {
        "status": "COMPLETED_DEVICE_A_AU_TI_OPTICAL_SCATTERING_DIAGNOSTIC",
        "not_actual_device_or_paper_reproduction": True,
        "dominance_threshold_predeclared": DOMINANCE_THRESHOLD,
        "dominance_rule": "equal-power TaIrTe4 q2d NRMSE, edge-fraction relative change, and maximum |Ia| relative change must all be at least 10%",
        "contact_optical_scattering_dominant": dominant,
        "spatial_comparison_grid": {
            "coordinate_grids_identical": coordinate_grids_identical,
            "common_grid": "Au/Ti-on bounded dual-cell x/y grid",
            "AuTi_off_exact_overlap_conservative_remap_power_error": remap_error,
            "simple_array_index_pairing_used": False,
        },
        "metrics": metrics,
        "equal_power_TaIrTe4_q2d_NRMSE": weighted_nrmse,
        "edge_localized_fraction_relative_change": edge_fraction_change,
        "terminal_current": currents,
        "no_raw_Q_rescaling": True,
        "equal_power_copy_used_only_for_spatial_shape_diagnostic": True,
    }
    (args.output_dir / "device_a_contact_optical_diagnostic_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figure, axes = plt.subplots(1, 4, figsize=(18, 4.4), constrained_layout=True)
    vmax = max(float(np.percentile(normalized[label], 99.5)) for label in normalized)
    for axis, label in zip(axes[:2], ("AuTi_on", "AuTi_off")):
        handle = axis.pcolormesh(np.asarray(on["x_m"]) * 1e6, np.asarray(on["y_m"]) * 1e6, normalized[label].T, shading="nearest", cmap="magma", vmin=0.0, vmax=vmax)
        axis.set_title(label + " equal-power TaIrTe4 Q")
        figure.colorbar(handle, ax=axis)
    bound = float(np.percentile(np.abs(difference), 99.5))
    handle = axes[2].pcolormesh(np.asarray(on["x_m"]) * 1e6, np.asarray(on["y_m"]) * 1e6, difference.T, shading="nearest", cmap="coolwarm", vmin=-bound, vmax=bound)
    axes[2].set_title("off - on normalized Q")
    figure.colorbar(handle, ax=axes[2])
    axes[3].plot(centers, profiles["AuTi_on"], label="Au/Ti on")
    axes[3].plot(centers, profiles["AuTi_off"], label="Au/Ti off")
    axes[3].set(xlabel="edge-normal n (µm)", ylabel="equal-power profile", title="digitized-edge normal profile")
    axes[3].legend()
    for axis in axes[:3]:
        axis.set_aspect("equal")
        axis.set(xlabel="x=b (µm)", ylabel="y=a (µm)")
    figure.savefig(args.output_dir / "DEVICE_A_AU_TI_ON_OFF_Q_COMPARISON.png", dpi=180)
    plt.close(figure)
    artifacts = []
    for label, path in (("on", args.on_optical), ("off", args.off_optical)):
        for filename in ("finite_q_on_artifact.npz", "finite_2um_optical_q.fsp", "case_result.json"):
            target = path / filename
            artifacts.append({"role": f"AuTi {label} {filename}", "path": str(target.resolve()), "size_bytes": target.stat().st_size, "sha256": sha256(target), "committed_to_git": False})
    thermal_paths = {
        "AuTi on E||a isolated": args.on_a_isolated,
        "AuTi off E||a no-metal control": args.off_a_isolated,
        "AuTi on E||a perfect": args.on_a_perfect,
        "baseline E||b isolated": args.b_isolated,
        "baseline E||b perfect": args.b_perfect,
    }
    for label, path in thermal_paths.items():
        for filename in ("thermal_pte_fields.npz", "summary.json"):
            target = path / filename
            artifacts.append({"role": f"{label} {filename}", "path": str(target.resolve()), "size_bytes": target.stat().st_size, "sha256": sha256(target), "committed_to_git": False})
    (args.output_dir / "RAW_ARTIFACT_MANIFEST_CONTACT_OPTICAL.json").write_text(json.dumps({"status": "RECORDED_EXTERNAL_RAW_ARTIFACTS_NOT_COMMITTED", "artifacts": artifacts, "generation_command": " ".join(sys.argv)}, indent=2) + "\n")
    report = f"""# Device-A Au/Ti optical-scattering diagnostic

Status: `{summary['status']}`

This removes the digitized Au/Ti polygons only from the optical s0 E||a
geometry. The baseline thermal/contact geometry and weighting potential are
retained. It is not an actual electrode-free device prediction or paper
reproduction.

- Equal-power TaIrTe4 Q spatial NRMSE: `{weighted_nrmse:.4%}`
- Edge-localized fraction relative change: `{edge_fraction_change:.4%}`
- Maximum isolated/perfect |Ia| relative change: `{maximum_current_change:.4%}`
- Predeclared 10% all-three dominance rule: `{dominant}`

Raw terminal currents use unmodified Q. Equal-power normalization is used only
for the spatial-shape map/profile diagnostic.
"""
    (args.output_dir / "DEVICE_A_CONTACT_OPTICAL_DIAGNOSTIC_REPORT.md").write_text(report)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
