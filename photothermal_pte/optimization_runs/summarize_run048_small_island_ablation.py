#!/usr/bin/env python3
"""Remove and evaluate only the small floating Run 048 TaIrTe4 island.

The promoted Run 048 artifact is immutable.  This script creates a diagnostic
copy in which the single 18-node floating component near
``(x=b, y=a)=(-6.65, -3.45) um`` is changed to air.  It then audits or
summarizes a fresh Maxwell -> thermal -> electrical evaluation without
changing any other design node.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy import ndimage


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.optimization_runs.summarize_final_exact_binary_eight_case import (
    TARGET_POWER_W,
    current_decomposition,
    full_density,
    strict_centered_gradient,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.optimization_support import (
    exact_binary_audit,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.thermal import (
    _piecewise_edges,
)


ORIGINAL_DENSITY = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored/"
    "run048_Eb_fresh_current_max/forced_exact_500nm_cleanup/solid_first_density.npz"
)
ORIGINAL_DIR = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored/"
    "run048_Eb_fresh_current_max/forced_exact_500nm_cleanup/solid_first_objective"
)
ABLATION_ROOT = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored/"
    "run048_remove_small_floating_island_20260818"
)
ABLATION_DENSITY = ABLATION_ROOT / "run048_small_island_removed_exact_binary.npz"
GEOMETRY_AUDIT = ABLATION_ROOT / "small_island_transform_audit.json"
ABLATION_DIR = ABLATION_ROOT / "small_island_removed_Eb_gpu_objective"
REPORT_DIR = REPOSITORY / "photothermal_pte" / "reports" / "run048_small_island_ablation"

EXPECTED_NODE_COUNT = 18
EXPECTED_CENTROID_UM = np.asarray((-6.65, -3.45), dtype=float)
SPACING_UM = 0.1
X_MIN_UM = -10.0
Y_MIN_UM = -12.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path, role: str) -> dict[str, object]:
    return {
        "role": role,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def relative_change(new: float, old: float) -> float:
    return (float(new) - float(old)) / max(abs(float(old)), np.finfo(float).tiny)


def relative_l2(new: np.ndarray, old: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        new = new[mask]
        old = old[mask]
    return float(np.linalg.norm(new - old) / max(np.linalg.norm(old), np.finfo(float).tiny))


def prepare() -> None:
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    with np.load(ORIGINAL_DENSITY) as raw:
        rho = np.asarray(raw["rho"], dtype=np.float64)
    if not np.all((rho == 0.0) | (rho == 1.0)):
        raise RuntimeError("Run 048 source density is not exact binary")

    binary = rho > 0.5
    labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=bool))
    candidates: list[dict[str, object]] = []
    for label in range(1, count + 1):
        ii, jj = np.where(labels == label)
        x = X_MIN_UM + SPACING_UM * ii
        y = Y_MIN_UM + SPACING_UM * jj
        centroid = np.asarray((np.mean(x), np.mean(y)))
        candidates.append(
            {
                "label": label,
                "node_count": int(ii.size),
                "centroid_um": centroid.tolist(),
                "distance_to_expected_centroid_um": float(
                    np.linalg.norm(centroid - EXPECTED_CENTROID_UM)
                ),
                "center_bbox_um": [float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))],
                "physical_support_bbox_um": [
                    float(np.min(x) - 0.05),
                    float(np.max(x) + 0.05),
                    float(np.min(y) - 0.05),
                    float(np.max(y) + 0.05),
                ],
            }
        )
    selected = min(candidates, key=lambda row: row["distance_to_expected_centroid_um"])
    if selected["node_count"] != EXPECTED_NODE_COUNT:
        raise RuntimeError(f"expected an 18-node component, found {selected}")
    if selected["distance_to_expected_centroid_um"] > 0.1:
        raise RuntimeError(f"small component is not at the audited Run 048 coordinate: {selected}")

    removed = labels == int(selected["label"])
    ablated = rho.copy()
    ablated[removed] = 0.0
    if int(np.count_nonzero(rho != ablated)) != EXPECTED_NODE_COUNT:
        raise RuntimeError("the transform changed nodes outside the selected small island")
    original_audit, _ = exact_binary_audit(
        rho,
        geometry_mode="left_right_contact_anchored",
        contact_axis="x",
    )
    ablated_audit, _ = exact_binary_audit(
        ablated,
        geometry_mode="left_right_contact_anchored",
        contact_axis="x",
    )
    if not bool(ablated_audit["passed"]):
        raise RuntimeError(f"island removal broke the exact 500 nm gate: {ablated_audit}")

    np.savez_compressed(ABLATION_DENSITY, rho=ablated)
    audit = {
        "schema": "run048-small-floating-island-transform-v1",
        "status": "PREPARED_RUN048_SMALL_FLOATING_ISLAND_ABLATION",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "axis_contract": "Lumerical x=b, y=a, z=c",
        "source_density": artifact(ORIGINAL_DENSITY, "immutable selected Run 048 exact density"),
        "output_density": artifact(ABLATION_DENSITY, "Run 048 with only the selected small island changed to air"),
        "selected_component": selected,
        "removed_node_count": int(np.count_nonzero(removed)),
        "removed_area_um2": float(np.count_nonzero(removed) * SPACING_UM**2),
        "changed_nodes_outside_selected_component": 0,
        "original_exact_500nm_audit": original_audit,
        "ablated_exact_500nm_audit": ablated_audit,
        "all_other_design_nodes_bitwise_identical": True,
    }
    GEOMETRY_AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


def load_case(directory: Path, density_path: Path) -> dict[str, object]:
    result_path = directory / "binary_objective_result.json"
    fields_path = directory / "binary_objective_fields.npz"
    result = json.loads(result_path.read_text())
    with np.load(fields_path) as raw:
        rho = np.asarray(raw["rho_binary"], dtype=np.float64)
        q = np.asarray(raw["mapped_Q_W_m3"], dtype=np.float64)
        temperature = np.asarray(raw["nodal_temperature_K"], dtype=np.float64)
        psi = np.asarray(raw["weighting_potential"], dtype=np.float64)
        grad_psi = np.asarray(raw["weighting_gradient_element_m_inv"], dtype=np.float64)
    with np.load(density_path) as raw:
        selected = np.asarray(raw["rho"], dtype=np.float64)
    if not np.array_equal(rho, selected):
        raise RuntimeError(f"stored field density differs from {density_path}")
    scale = TARGET_POWER_W / float(result["forward"]["source_power_W"])
    rho_full = full_density(rho)
    temperature_scaled = temperature * scale
    gx, gy, gmag = strict_centered_gradient(temperature_scaled, rho_full > 0.5)
    current = current_decomposition(rho_full, temperature, grad_psi, scale)
    x_edges, y_edges, z_edges = _piecewise_edges()
    dx, dy, dz = np.diff(x_edges), np.diff(y_edges), np.diff(z_edges)
    volume = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
    q_scaled = q * scale
    return {
        "result": result,
        "result_path": result_path,
        "fields_path": fields_path,
        "rho": rho,
        "rho_full": rho_full,
        "q": q_scaled,
        "qxy": np.sum(q_scaled * dz[None, None, :], axis=2),
        "mapped_power_W": float(np.sum(q_scaled * volume)),
        "temperature": temperature_scaled,
        "psi": psi,
        "gx": gx,
        "gy": gy,
        "gmag": gmag,
        "current": current,
        "xc_um": 0.5 * (x_edges[:-1] + x_edges[1:]) * 1e6,
        "yc_um": 0.5 * (y_edges[:-1] + y_edges[1:]) * 1e6,
    }


def plot_map(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    value: np.ndarray,
    title: str,
    cmap: str = "magma",
    centered: bool = False,
) -> None:
    finite = value[np.isfinite(value)]
    norm = None
    if centered and finite.size:
        bound = max(abs(float(np.min(finite))), abs(float(np.max(finite))), np.finfo(float).tiny)
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    image = axis.pcolormesh(x, y, value.T, shading="auto", cmap=cmap, norm=norm)
    axis.set_aspect("equal")
    axis.set_xlabel("Lumerical x=b (µm)")
    axis.set_ylabel("Lumerical y=a (µm)")
    axis.set_title(title)
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.03)


def summarize() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    audit = json.loads(GEOMETRY_AUDIT.read_text())
    original = load_case(ORIGINAL_DIR, ORIGINAL_DENSITY)
    ablated = load_case(ABLATION_DIR, ABLATION_DENSITY)
    retained = ablated["rho_full"] > 0.5
    common_grad = np.isfinite(original["gmag"]) & np.isfinite(ablated["gmag"])
    old_result = original["result"]
    new_result = ablated["result"]

    old_i = float(old_result["equivalent_objective_at_285uW_A"])
    new_i = float(new_result["equivalent_objective_at_285uW_A"])
    metrics = {
        "schema": "run048-small-floating-island-ablation-comparison-v1",
        "status": "COMPLETED_RUN048_SMALL_FLOATING_ISLAND_ABLATION",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "axis_contract": "Lumerical x=b, y=a, z=c",
        "unchanged_contract": "Run 048 Eb, left-right electrodes, thermally-grown SiO2, identical source/mesh/PML/thermal/electrical operators",
        "removed_component": audit["selected_component"],
        "removed_node_count": audit["removed_node_count"],
        "removed_area_um2": audit["removed_area_um2"],
        "exact_500nm_bad_nodes_after": audit["ablated_exact_500nm_audit"]["total_bad_cell_count"],
        "original": {
            "P_Q_raw_W": float(old_result["forward"]["P_Q_W"]),
            "mapped_P_Q_at_285uW_W": original["mapped_power_W"],
            "Tmax_at_285uW_K": float(np.max(original["temperature"][original["rho_full"] > 0.5])),
            "max_strict_gradT_K_m": float(np.nanmax(original["gmag"])),
            "terminal_current_at_285uW_A": old_i,
            "terminal_conductance_S": float(old_result["terminal_conductance_S"]),
            "closure": float(old_result["forward"]["closure"]),
        },
        "small_island_removed": {
            "P_Q_raw_W": float(new_result["forward"]["P_Q_W"]),
            "mapped_P_Q_at_285uW_W": ablated["mapped_power_W"],
            "Tmax_at_285uW_K": float(np.max(ablated["temperature"][retained])),
            "max_strict_gradT_K_m": float(np.nanmax(ablated["gmag"])),
            "terminal_current_at_285uW_A": new_i,
            "terminal_conductance_S": float(new_result["terminal_conductance_S"]),
            "closure": float(new_result["forward"]["closure"]),
            "all_physical_gates_passed": bool(new_result["physical_gates_passed"]),
        },
    }
    metrics["relative_change_removed_vs_original"] = {
        key: relative_change(metrics["small_island_removed"][key], metrics["original"][key])
        for key in metrics["original"]
        if key in metrics["small_island_removed"]
    }
    metrics["relative_L2_difference"] = {
        "full_3D_mapped_Q": relative_l2(ablated["q"], original["q"]),
        "depth_integrated_Qxy": relative_l2(ablated["qxy"], original["qxy"]),
        "temperature_on_retained_support": relative_l2(
            ablated["temperature"], original["temperature"], retained
        ),
        "strict_gradient_on_common_valid_support": relative_l2(
            ablated["gmag"], original["gmag"], common_grad
        ),
        "terminal_current_integrand": relative_l2(
            ablated["current"]["current_density_total_A_m2"],
            original["current"]["current_density_total_A_m2"],
        ),
    }
    metrics["interpretation"] = (
        "This fresh end-to-end ablation measures the net indirect optothermal role of only the selected "
        "18-node floating island; it has no direct terminal conduction path."
    )
    metrics["no_clipping_smoothing_gain_or_rescaling"] = True

    summary_path = REPORT_DIR / "run048_small_island_ablation_summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2) + "\n")
    csv_path = REPORT_DIR / "run048_small_island_ablation_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("metric", "original", "small_island_removed", "relative_change"))
        for key, old in metrics["original"].items():
            if key in metrics["small_island_removed"] and isinstance(old, (int, float)):
                writer.writerow(
                    (
                        key,
                        old,
                        metrics["small_island_removed"][key],
                        metrics["relative_change_removed_vs_original"][key],
                    )
                )

    design_nodes_x = np.linspace(-10.0, 10.0, original["rho"].shape[0])
    design_nodes_y = np.linspace(-12.0, 12.0, original["rho"].shape[1])
    full_nodes = np.arange(-12.0, 12.0 + 0.05, 0.1)
    cells = np.arange(-12.0 + 0.05, 12.0, 0.1)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    plot_map(axes[0, 0], design_nodes_x, design_nodes_y, original["rho"], "original Run 048", "gray_r")
    plot_map(axes[0, 1], design_nodes_x, design_nodes_y, ablated["rho"], "small island -> air", "gray_r")
    plot_map(axes[0, 2], design_nodes_x, design_nodes_y, original["rho"] - ablated["rho"], "removed 18-node island", "Reds")
    plot_map(axes[1, 0], original["xc_um"], original["yc_um"], original["qxy"], "original depth-integrated Q")
    plot_map(axes[1, 1], ablated["xc_um"], ablated["yc_um"], ablated["qxy"], "island removed depth-integrated Q")
    plot_map(axes[1, 2], ablated["xc_um"], ablated["yc_um"], ablated["qxy"] - original["qxy"], "Q difference", "coolwarm", True)
    fig.suptitle("Run 048 single-small-island ablation: geometry and Maxwell heat source")
    geometry_plot = REPORT_DIR / "run048_small_island_ablation_geometry_Q.png"
    fig.savefig(geometry_plot, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    for column, (title, old, new, cmap, centered) in enumerate(
        (
            ("TaIrTe4 ΔT (K)", original["temperature"], ablated["temperature"], "magma", False),
            ("strict |grad T| (K/m)", original["gmag"], ablated["gmag"], "viridis", False),
            (
                "current integrand (A/m²)",
                original["current"]["current_density_total_A_m2"],
                ablated["current"]["current_density_total_A_m2"],
                "coolwarm",
                True,
            ),
        )
    ):
        coordinates = full_nodes if column < 2 else cells
        plot_map(axes[0, column], coordinates, coordinates, old, f"original {title}", cmap, centered)
        plot_map(axes[1, column], coordinates, coordinates, new, f"small island removed {title}", cmap, centered)
    fig.suptitle("Run 048 single-small-island ablation: downstream thermal/PTE fields")
    fields_plot = REPORT_DIR / "run048_small_island_ablation_thermal_current.png"
    fig.savefig(fields_plot, dpi=180)
    plt.close(fig)

    # Standalone final certificate for the user-selected island-free geometry.
    fig, axes = plt.subplots(3, 4, figsize=(21, 15), constrained_layout=True)
    rho_full_removed = full_density(ablated["rho"])
    plot_map(axes[0, 0], full_nodes, full_nodes, rho_full_removed, "exact binary: black=TaIrTe4", "gray_r")
    plot_map(axes[0, 1], ablated["xc_um"], ablated["yc_um"], ablated["qxy"], "all-material depth-integrated Q (W/m²)")
    plot_map(axes[0, 2], full_nodes, full_nodes, np.where(rho_full_removed > 0.5, ablated["temperature"], np.nan), "TaIrTe4 ΔT at 285 µW (K)")
    plot_map(axes[0, 3], full_nodes, full_nodes, np.where(rho_full_removed > 0.5, ablated["psi"], np.nan), "weighting potential ψ (high terminal=1)", "viridis")
    plot_map(axes[1, 0], full_nodes, full_nodes, ablated["gx"], "strict-centered ∂T/∂b (K/m)", "coolwarm", True)
    plot_map(axes[1, 1], full_nodes, full_nodes, ablated["gy"], "strict-centered ∂T/∂a (K/m)", "coolwarm", True)
    plot_map(axes[1, 2], full_nodes, full_nodes, ablated["gmag"], "strict-centered |∇T| (K/m)", "viridis")
    plot_map(axes[1, 3], cells, cells, ablated["current"]["current_density_total_A_m2"], "total PTE current integrand (A/m²)", "coolwarm", True)
    plot_map(axes[2, 0], cells, cells, ablated["current"]["current_density_b_A_m2"], "b-component current integrand (A/m²)", "coolwarm", True)
    plot_map(axes[2, 1], cells, cells, ablated["current"]["current_density_a_A_m2"], "a-component current integrand (A/m²)", "coolwarm", True)
    axes[2, 2].axis("off")
    axes[2, 3].axis("off")
    axes[2, 2].text(
        0.02,
        0.98,
        (
            "Run 048 diagnostic: 18-node island -> air\n"
            "thermally grown SiO2; left-right electrodes; E||b\n"
            f"exact 500 nm bad nodes = {metrics['exact_500nm_bad_nodes_after']}\n"
            f"P_Q(285 µW) = {metrics['small_island_removed']['mapped_P_Q_at_285uW_W']*1e6:.4f} µW\n"
            f"closure = {metrics['small_island_removed']['closure']*100:.4f}%\n"
            f"Tmax = {metrics['small_island_removed']['Tmax_at_285uW_K']:.6g} K\n"
            f"I = {new_i*1e9:.6f} nA\n"
            f"I_b = {ablated['current']['integrated_b_component_A']*1e9:.6f} nA\n"
            f"I_a = {ablated['current']['integrated_a_component_A']*1e9:.6f} nA\n"
            "No clipping/smoothing/gain/rescaling."
        ),
        va="top",
        family="monospace",
        fontsize=12,
    )
    fig.suptitle("Run 048 small-island-removed exact-binary physical fields", fontsize=16)
    final_fields_plot = REPORT_DIR / "run048_small_island_removed_final_fields.png"
    fig.savefig(final_fields_plot, dpi=180)
    plt.close(fig)

    old_nA, new_nA = old_i * 1e9, new_i * 1e9
    change = metrics["relative_change_removed_vs_original"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    axes[0].bar(("original", "island removed"), (old_nA, new_nA))
    axes[0].set_ylabel("terminal current at 285 µW (nA)")
    axes[0].set_title(f"ΔI = {change['terminal_current_at_285uW_A']*100:.3f}%")
    axes[1].bar(("original", "island removed"), (metrics["original"]["P_Q_raw_W"]*1e15, metrics["small_island_removed"]["P_Q_raw_W"]*1e15))
    axes[1].set_ylabel("raw Maxwell P_Q (fW)")
    axes[1].set_title(f"ΔP_Q = {change['P_Q_raw_W']*100:.3f}%")
    axes[2].bar(("original", "island removed"), (metrics["original"]["Tmax_at_285uW_K"], metrics["small_island_removed"]["Tmax_at_285uW_K"]))
    axes[2].set_ylabel("TaIrTe4 Tmax rise at 285 µW (K)")
    axes[2].set_title(f"ΔTmax = {change['Tmax_at_285uW_K']*100:.3f}%")
    scalar_plot = REPORT_DIR / "run048_small_island_ablation_scalar_comparison.png"
    fig.savefig(scalar_plot, dpi=180)
    plt.close(fig)

    manifest_path = REPORT_DIR / "RAW_ARTIFACT_MANIFEST.json"
    manifest = {
        "schema": "run048-small-island-ablation-artifact-manifest-v1",
        "generated_at_utc": metrics["generated_at_utc"],
        "raw_files_committed": False,
        "generation_command": (
            "TAIRTE4_TOPOLOGY_GEOMETRY=left_right_contact_anchored "
            "TAIRTE4_SIO2_INTERFACE_SCENARIO=thermally_grown CUDA_VISIBLE_DEVICES=5 "
            "python -m photothermal_pte.optimization_runs.tairte4_flake_topology.evaluate_binary_objective ..."
        ),
        "artifacts": [
            artifact(ORIGINAL_DENSITY, "original Run 048 exact density"),
            artifact(ORIGINAL_DIR / "binary_objective_result.json", "original Run 048 result"),
            artifact(ORIGINAL_DIR / "binary_objective_fields.npz", "original Run 048 fields"),
            artifact(ABLATION_DENSITY, "small-island-removed exact density"),
            artifact(GEOMETRY_AUDIT, "small-island transform audit"),
            artifact(ABLATION_DIR / "binary_objective_result.json", "small-island-removed result"),
            artifact(ABLATION_DIR / "binary_objective_fields.npz", "small-island-removed fields"),
            artifact(ABLATION_DIR / "exact_binary_forward_Eb.fsp", "small-island-removed raw FSP"),
            artifact(geometry_plot, "geometry and Q comparison plot"),
            artifact(fields_plot, "thermal and current comparison plot"),
            artifact(final_fields_plot, "small-island-removed final physical fields plot"),
            artifact(scalar_plot, "scalar ablation comparison plot"),
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    report_path = REPORT_DIR / "RUN048_SMALL_ISLAND_ABLATION_REPORT.md"
    report_path.write_text(
        f"""# Run 048 single-small-island ablation

Status: `COMPLETED_RUN048_SMALL_FLOATING_ISLAND_ABLATION`

The immutable selected Run 048 exact-binary density was preserved. Only the 18-node floating TaIrTe4 component near `(x=b,y=a)=(-6.65,-3.45) um` was changed to air, followed by one fresh GPU Maxwell solve and the unchanged CUDA thermal/electrical calculation.

## Geometry audit

- Removed solid nodes: `{metrics['removed_node_count']}` (`{metrics['removed_area_um2']:.3f} um^2`)
- All other nodes: bitwise identical
- Exact 500 nm solid/void bad nodes after removal: `{metrics['exact_500nm_bad_nodes_after']}`

## Fresh end-to-end comparison at 285 µW

| Metric | Original | Small island removed | Relative change |
|---|---:|---:|---:|
| Raw Maxwell P_Q | {metrics['original']['P_Q_raw_W']:.9e} W | {metrics['small_island_removed']['P_Q_raw_W']:.9e} W | {change['P_Q_raw_W']*100:.4f}% |
| Mapped P_Q | {metrics['original']['mapped_P_Q_at_285uW_W']:.9e} W | {metrics['small_island_removed']['mapped_P_Q_at_285uW_W']:.9e} W | {change['mapped_P_Q_at_285uW_W']*100:.4f}% |
| TaIrTe4 Tmax rise | {metrics['original']['Tmax_at_285uW_K']:.9g} K | {metrics['small_island_removed']['Tmax_at_285uW_K']:.9g} K | {change['Tmax_at_285uW_K']*100:.4f}% |
| Max strict grad T | {metrics['original']['max_strict_gradT_K_m']:.9g} K/m | {metrics['small_island_removed']['max_strict_gradT_K_m']:.9g} K/m | {change['max_strict_gradT_K_m']*100:.4f}% |
| Terminal current | {old_nA:.6f} nA | {new_nA:.6f} nA | {change['terminal_current_at_285uW_A']*100:.4f}% |
| Terminal conductance | {metrics['original']['terminal_conductance_S']:.9e} S | {metrics['small_island_removed']['terminal_conductance_S']:.9e} S | {change['terminal_conductance_S']*100:.6f}% |

All optical closure, conservative mapping, thermal residual/energy, electrical residual, finite-value, and GPU-only physical gates passed: `{metrics['small_island_removed']['all_physical_gates_passed']}`. No clipping, smoothing, gain, polarization matching, or rescaling was used.
"""
    )
    print(json.dumps(metrics, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "summarize"))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        summarize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
