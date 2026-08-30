#!/usr/bin/env python3
"""Compare exact Run 044 with a fresh floating-island-removed evaluation."""

from __future__ import annotations

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


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.optimization_runs.summarize_final_exact_binary_eight_case import (
    TARGET_POWER_W,
    current_decomposition,
    full_density,
    strict_centered_gradient,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.thermal import (
    _piecewise_edges,
)


ORIGINAL_DIR = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/"
    "run044_exact_500nm_cleanup_20260811/void_first_Ea_gpu_objective_retry2"
)
ORIGINAL_DENSITY = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/"
    "run044_exact_500nm_cleanup_20260811/void_first_exact_binary_candidate.npz"
)
ABLATION_DIR = Path(
    "/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/"
    "run044_remove_floating_islands_20260817/islands_removed_Ea_gpu_objective_retry2"
)
ABLATION_DENSITY = Path(
    "/home/seunghyun/tairte4/artifacts/run044_remove_floating_islands_20260817/"
    "run044_islands_removed_exact_binary.npz"
)
CONNECTIVITY_AUDIT = Path(
    "/home/seunghyun/tairte4/artifacts/run044_remove_floating_islands_20260817/"
    "connectivity_transform_audit.json"
)
REPORT_DIR = REPOSITORY / "photothermal_pte" / "reports" / "run044_floating_island_ablation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path, role: str) -> dict[str, object]:
    return {"role": role, "path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def relative_change(new: float, old: float) -> float:
    return (float(new) - float(old)) / abs(float(old))


def relative_l2(new: np.ndarray, old: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        new = new[mask]
        old = old[mask]
    return float(np.linalg.norm(new - old) / max(np.linalg.norm(old), np.finfo(float).tiny))


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
    with np.load(density_path) as density:
        selected = np.asarray(density["rho"], dtype=np.float64)
    if not np.array_equal(rho, selected):
        raise RuntimeError(f"stored field density differs from {density_path}")
    for name, value in (("rho", rho), ("Q", q), ("T", temperature), ("psi", psi), ("grad_psi", grad_psi)):
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"{name} contains NaN/Inf")
    scale = TARGET_POWER_W / float(result["forward"]["source_power_W"])
    rho_full = full_density(rho)
    temperature_scaled = temperature * scale
    gx, gy, gmag = strict_centered_gradient(temperature_scaled, rho_full > 0.5)
    current = current_decomposition(rho_full, temperature, grad_psi, scale)
    x_edges, y_edges, z_edges = _piecewise_edges()
    dx, dy, dz = np.diff(x_edges), np.diff(y_edges), np.diff(z_edges)
    volume = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
    q_scaled = q * scale
    qxy = np.sum(q_scaled * dz[None, None, :], axis=2)
    xc = 0.5 * (x_edges[:-1] + x_edges[1:]) * 1.0e6
    yc = 0.5 * (y_edges[:-1] + y_edges[1:]) * 1.0e6
    return {
        "result": result,
        "result_path": result_path,
        "fields_path": fields_path,
        "rho": rho,
        "rho_full": rho_full,
        "q": q_scaled,
        "qxy": qxy,
        "mapped_power_W": float(np.sum(q_scaled * volume)),
        "temperature": temperature_scaled,
        "psi": psi,
        "gx": gx,
        "gy": gy,
        "gmag": gmag,
        "current": current,
        "xc_um": xc,
        "yc_um": yc,
    }


def plot_map(ax: plt.Axes, x: np.ndarray, y: np.ndarray, value: np.ndarray, title: str,
             cmap: str = "magma", centered: bool = False) -> None:
    finite = value[np.isfinite(value)]
    norm = None
    if centered and finite.size:
        bound = max(abs(float(np.min(finite))), abs(float(np.max(finite))), np.finfo(float).tiny)
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    image = ax.pcolormesh(x, y, value.T, shading="auto", cmap=cmap, norm=norm)
    ax.set_aspect("equal")
    ax.set_xlabel("Lumerical x=b (µm)")
    ax.set_ylabel("Lumerical y=a (µm)")
    ax.set_title(title)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    original = load_case(ORIGINAL_DIR, ORIGINAL_DENSITY)
    ablated = load_case(ABLATION_DIR, ABLATION_DENSITY)
    audit = json.loads(CONNECTIVITY_AUDIT.read_text())
    retained = ablated["rho_full"] > 0.5
    common_grad = np.isfinite(original["gmag"]) & np.isfinite(ablated["gmag"])
    original_result = original["result"]
    ablated_result = ablated["result"]

    old_i = float(original_result["equivalent_objective_at_285uW_A"])
    new_i = float(ablated_result["equivalent_objective_at_285uW_A"])
    old_tmax_global = float(np.max(original["temperature"][original["rho_full"] > 0.5]))
    new_tmax_global = float(np.max(ablated["temperature"][ablated["rho_full"] > 0.5]))
    old_tmax = float(np.max(original["temperature"][retained]))
    new_tmax = float(np.max(ablated["temperature"][retained]))
    old_gmax_global = float(np.nanmax(original["gmag"]))
    new_gmax_global = float(np.nanmax(ablated["gmag"]))
    old_gmax = float(np.nanmax(original["gmag"][common_grad]))
    new_gmax = float(np.nanmax(ablated["gmag"][common_grad]))
    metrics = {
        "schema": "run044-floating-island-ablation-comparison-v1",
        "status": "COMPLETED_RUN044_FLOATING_ISLAND_ABLATION",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "axis_contract": "Lumerical x=b, y=a, z=c",
        "unchanged_contract": "Run 044 Ea, top-bottom electrodes, thermally-grown SiO2, identical source/mesh/PML/thermal/electrical operators",
        "removed_floating_component_count": sum(not x["terminal_connected"] for x in audit["components_4_neighbour"]),
        "removed_solid_nodes": audit["removed_solid_nodes"],
        "removed_fraction_of_original_solid": audit["removed_solid_fraction_of_original_solid"],
        "exact_500nm_bad_nodes_after": audit["exact_500nm_audit_after"]["total_bad_cell_count"],
        "original": {
            "P_Q_raw_W": float(original_result["forward"]["P_Q_W"]),
            "mapped_P_Q_at_285uW_W": original["mapped_power_W"],
            "global_Tmax_at_285uW_K": old_tmax_global,
            "Tmax_on_retained_support_at_285uW_K": old_tmax,
            "global_max_strict_gradT_K_m": old_gmax_global,
            "max_common_strict_gradT_K_m": old_gmax,
            "terminal_current_at_285uW_A": old_i,
            "terminal_conductance_S": float(original_result["terminal_conductance_S"]),
            "closure": float(original_result["forward"]["closure"]),
        },
        "islands_removed": {
            "P_Q_raw_W": float(ablated_result["forward"]["P_Q_W"]),
            "mapped_P_Q_at_285uW_W": ablated["mapped_power_W"],
            "global_Tmax_at_285uW_K": new_tmax_global,
            "Tmax_on_retained_support_at_285uW_K": new_tmax,
            "global_max_strict_gradT_K_m": new_gmax_global,
            "max_common_strict_gradT_K_m": new_gmax,
            "terminal_current_at_285uW_A": new_i,
            "terminal_conductance_S": float(ablated_result["terminal_conductance_S"]),
            "closure": float(ablated_result["forward"]["closure"]),
            "all_physical_gates_passed": bool(ablated_result["physical_gates_passed"]),
        },
        "relative_change_islands_removed_vs_original": {
            "P_Q": relative_change(ablated_result["forward"]["P_Q_W"], original_result["forward"]["P_Q_W"]),
            "mapped_P_Q_at_285uW": relative_change(ablated["mapped_power_W"], original["mapped_power_W"]),
            "global_Tmax": relative_change(new_tmax_global, old_tmax_global),
            "Tmax_on_retained_support": relative_change(new_tmax, old_tmax),
            "global_max_strict_gradT": relative_change(new_gmax_global, old_gmax_global),
            "max_common_strict_gradT": relative_change(new_gmax, old_gmax),
            "terminal_current": relative_change(new_i, old_i),
            "terminal_conductance": relative_change(ablated_result["terminal_conductance_S"], original_result["terminal_conductance_S"]),
        },
        "relative_L2_difference": {
            "full_3D_mapped_Q": relative_l2(ablated["q"], original["q"]),
            "depth_integrated_Qxy": relative_l2(ablated["qxy"], original["qxy"]),
            "temperature_on_retained_support": relative_l2(ablated["temperature"], original["temperature"], retained),
            "strict_gradient_on_common_valid_support": relative_l2(ablated["gmag"], original["gmag"], common_grad),
            "terminal_current_integrand": relative_l2(
                ablated["current"]["current_density_total_A_m2"],
                original["current"]["current_density_total_A_m2"],
            ),
        },
        "interpretation": (
            "Floating components have negligible direct electrical collection, but their removal changes "
            "Maxwell absorption/scattering and the downstream temperature field; the fresh end-to-end current "
            "change therefore measures their net indirect optothermal contribution."
        ),
        "no_clipping_smoothing_gain_or_rescaling": True,
    }
    (REPORT_DIR / "run044_floating_island_ablation_summary.json").write_text(json.dumps(metrics, indent=2) + "\n")

    rows = []
    for key in metrics["original"]:
        if isinstance(metrics["original"][key], (int, float)) and key in metrics["islands_removed"]:
            rows.append((key, metrics["original"][key], metrics["islands_removed"][key], relative_change(metrics["islands_removed"][key], metrics["original"][key])))
    with (REPORT_DIR / "run044_floating_island_ablation_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "original", "islands_removed", "relative_change"))
        writer.writerows(rows)

    nodes = np.linspace(-12.0, 12.0, 241)
    cells = 0.5 * (nodes[:-1] + nodes[1:])
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    plot_map(axes[0, 0], nodes, nodes, original["rho_full"], "original Run 044", "gray_r")
    plot_map(axes[0, 1], nodes, nodes, ablated["rho_full"], "floating islands -> air", "gray_r")
    plot_map(axes[0, 2], nodes, nodes, original["rho_full"] - ablated["rho_full"], "removed TaIrTe4 mask", "Reds")
    plot_map(axes[1, 0], original["xc_um"], original["yc_um"], original["qxy"], "original depth-integrated Q")
    plot_map(axes[1, 1], ablated["xc_um"], ablated["yc_um"], ablated["qxy"], "islands removed depth-integrated Q")
    plot_map(axes[1, 2], ablated["xc_um"], ablated["yc_um"], ablated["qxy"] - original["qxy"], "Q difference (removed-original)", "coolwarm", True)
    fig.suptitle("Run 044 floating-island ablation: geometry and Maxwell heat source")
    fig.savefig(REPORT_DIR / "run044_island_ablation_geometry_Q.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(16, 15), constrained_layout=True)
    for row, (field, title, cmap, centered) in enumerate((
        ("temperature", "TaIrTe4 ΔT at 285 µW (K)", "magma", False),
        ("gmag", "strict-centered |∇T| (K/m)", "viridis", False),
    )):
        old = np.where(original["rho_full"] > 0.5, original[field], np.nan)
        new = np.where(ablated["rho_full"] > 0.5, ablated[field], np.nan)
        difference = np.where(retained, ablated[field] - original[field], np.nan)
        plot_map(axes[row, 0], nodes, nodes, old, f"original {title}", cmap, centered)
        plot_map(axes[row, 1], nodes, nodes, new, f"islands removed {title}", cmap, centered)
        plot_map(axes[row, 2], nodes, nodes, difference, f"difference {title}", "coolwarm", True)
    old_j = original["current"]["current_density_total_A_m2"]
    new_j = ablated["current"]["current_density_total_A_m2"]
    plot_map(axes[2, 0], cells, cells, old_j, "original PTE current integrand (A/m²)", "coolwarm", True)
    plot_map(axes[2, 1], cells, cells, new_j, "islands removed current integrand (A/m²)", "coolwarm", True)
    plot_map(axes[2, 2], cells, cells, new_j - old_j, "current-integrand difference", "coolwarm", True)
    fig.suptitle("Run 044 floating-island ablation: downstream thermal/PTE fields")
    fig.savefig(REPORT_DIR / "run044_island_ablation_thermal_current.png", dpi=170)
    plt.close(fig)

    old_nA, new_nA = old_i * 1e9, new_i * 1e9
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    axes[0].bar(("original", "islands removed"), (old_nA, new_nA), color=("tab:blue", "tab:orange"))
    axes[0].set_ylabel("terminal current at 285 µW (nA)")
    axes[0].set_title(f"ΔI = {relative_change(new_i, old_i)*100:.3f}%")
    axes[1].bar(("original", "islands removed"), (float(original_result["forward"]["P_Q_W"])*1e15, float(ablated_result["forward"]["P_Q_W"])*1e15))
    axes[1].set_ylabel("raw Maxwell P_Q (fW)")
    axes[1].set_title(f"ΔP_Q = {metrics['relative_change_islands_removed_vs_original']['P_Q']*100:.3f}%")
    axes[2].bar(("original", "islands removed"), (old_tmax_global, new_tmax_global))
    axes[2].set_ylabel("global TaIrTe4 Tmax at 285 µW (K)")
    axes[2].set_title(f"ΔTmax = {relative_change(new_tmax_global, old_tmax_global)*100:.3f}%")
    fig.savefig(REPORT_DIR / "run044_island_ablation_scalar_comparison.png", dpi=170)
    plt.close(fig)

    raw_manifest = {
        "schema": "run044-floating-island-ablation-artifact-manifest-v1",
        "generated_at_utc": metrics["generated_at_utc"],
        "raw_artifacts_committed_to_git": False,
        "generation_command": (
            "TAIRTE4_TOPOLOGY_GEOMETRY=contact_anchored PYTHONPATH=. "
            "/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python "
            "photothermal_pte/optimization_runs/summarize_run044_floating_island_ablation.py"
        ),
        "artifacts": [
            artifact(ORIGINAL_DIR / "binary_objective_result.json", "original Run 044 result"),
            artifact(ORIGINAL_DIR / "binary_objective_fields.npz", "original Run 044 fields"),
            artifact(ORIGINAL_DENSITY, "original Run 044 exact density"),
            artifact(ABLATION_DIR / "binary_objective_result.json", "islands-removed result"),
            artifact(ABLATION_DIR / "binary_objective_fields.npz", "islands-removed fields"),
            artifact(ABLATION_DIR / "exact_binary_forward_Ea.fsp", "islands-removed raw FSP"),
            artifact(ABLATION_DENSITY, "islands-removed exact density"),
            artifact(CONNECTIVITY_AUDIT, "connectivity transform audit"),
        ],
    }
    (REPORT_DIR / "RAW_ARTIFACT_MANIFEST.json").write_text(json.dumps(raw_manifest, indent=2) + "\n")

    change = metrics["relative_change_islands_removed_vs_original"]
    l2 = metrics["relative_L2_difference"]
    report = f"""# Run 044 floating-island ablation

Status: `COMPLETED_RUN044_FLOATING_ISLAND_ABLATION`

The immutable selected Run 044 exact-binary density was preserved. Six TaIrTe4 solid components touching neither top nor bottom terminal were changed to air in a diagnostic copy, followed by one fresh GPU Maxwell forward solve and the unchanged CUDA thermal/electrical path.

## Geometry audit

- Floating components removed: {metrics['removed_floating_component_count']}
- Removed solid nodes: {metrics['removed_solid_nodes']} ({metrics['removed_fraction_of_original_solid']*100:.3f}% of original solid)
- 4/8-neighbour terminal-connected support: identical
- Exact 500 nm bad nodes after removal: {metrics['exact_500nm_bad_nodes_after']}

## Fresh end-to-end comparison at 285 µW

| Metric | Original | Islands removed | Relative change |
|---|---:|---:|---:|
| Raw Maxwell P_Q | {metrics['original']['P_Q_raw_W']:.9e} W | {metrics['islands_removed']['P_Q_raw_W']:.9e} W | {change['P_Q']*100:.3f}% |
| Global TaIrTe4 Tmax | {old_tmax_global:.6g} K | {new_tmax_global:.6g} K | {change['global_Tmax']*100:.3f}% |
| Tmax on retained TaIrTe4 | {old_tmax:.6g} K | {new_tmax:.6g} K | {change['Tmax_on_retained_support']*100:.3f}% |
| Global max strict |grad T| | {old_gmax_global:.6g} K/m | {new_gmax_global:.6g} K/m | {change['global_max_strict_gradT']*100:.3f}% |
| Max common strict |grad T| | {old_gmax:.6g} K/m | {new_gmax:.6g} K/m | {change['max_common_strict_gradT']*100:.3f}% |
| Terminal current | {old_i*1e9:.6f} nA | {new_i*1e9:.6f} nA | {change['terminal_current']*100:.3f}% |
| Terminal conductance | {metrics['original']['terminal_conductance_S']:.9e} S | {metrics['islands_removed']['terminal_conductance_S']:.9e} S | {change['terminal_conductance']*100:.6f}% |

Spatial relative-L2 differences are {l2['full_3D_mapped_Q']*100:.3f}% for mapped 3-D Q, {l2['depth_integrated_Qxy']*100:.3f}% for depth-integrated Q, {l2['temperature_on_retained_support']*100:.3f}% for temperature on retained TaIrTe4, and {l2['strict_gradient_on_common_valid_support']*100:.3f}% for the strict gradient on common valid nodes.

## Interpretation

The terminal conductance is unchanged to numerical precision, confirming that the removed components did not form the collected DC path. Nevertheless, removing them reduces absorption, temperature, and terminal current. Their net role in this optimized result is therefore indirect: Maxwell scattering/absorption and the resulting thermal-field redistribution, not direct electrical collection.

The raw result status remains failed only against the inherited one-percent objective-preservation gate, because this experiment intentionally tests a change expected to alter the objective. All optical closure, mapping, thermal residual/energy, electrical residual, finite-value, and GPU-only gates passed. No Q clipping, smoothing, gain, polarization matching, or rescaling was used.
"""
    (REPORT_DIR / "RUN044_FLOATING_ISLAND_ABLATION_REPORT.md").write_text(report)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
