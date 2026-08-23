#!/usr/bin/env python3
"""Publish rotated Run059/060 fields from completed no-Au artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("TAIRTE4_TOPOLOGY_GEOMETRY", "diagonal_45_contact_anchored")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import (  # noqa: E402
    CONTRACT,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.electrical import (  # noqa: E402
    build_rotated_device_mesh,
    solve_short_circuit_current_density,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.optimization_support import (  # noqa: E402
    exact_binary_audit,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.rotated_device import (  # noqa: E402
    device_to_crystal_coordinates,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.thermal import (  # noqa: E402
    _piecewise_edges,
)


REPORT_DIR = (
    REPOSITORY
    / "photothermal_pte"
    / "reports"
    / "run059_060_rotated45_ideal_terminal_no_Au"
)
TARGET_POWER_W = 285.0e-6
STEP_M = 100.0e-9
THICKNESS_M = 100.0e-9
SIGMA_XY_S_M = (1.10e5, 4.91e5)
SEEBECK_XY_V_K = (27.0e-6, -6.0e-6)

RUNS = {
    59: REPOSITORY / "photothermal_pte/optimization_runs/run_059_diagonal45_evaporated_sio2_Ea_bounded_official_dfm_exact_repair/results_v5_no_Au",
    60: REPOSITORY / "photothermal_pte/optimization_runs/run_060_diagonal45_evaporated_sio2_Eb_bounded_official_dfm_exact_repair/results_v5_no_Au",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rotated_coordinates(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count_x, count_y = values.shape[:2]
    u = np.linspace(-12.0e-6, 12.0e-6, count_x)
    v = np.linspace(-12.0e-6, 12.0e-6, count_y)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    x, y = device_to_crystal_coordinates(uu, vv)
    return x * 1.0e6, y * 1.0e6


def element_to_cell(values: np.ndarray, *, sum_triangles: bool = False) -> np.ndarray:
    array = np.asarray(values)
    cells = 240 * 240
    if array.shape[0] != 2 * cells:
        raise ValueError(f"unexpected element array shape {array.shape}")
    combined = array[:cells] + array[cells:]
    if not sum_triangles:
        combined = 0.5 * combined
    return combined.reshape(240, 240, *array.shape[1:])


def symmetric_norm(values: np.ndarray) -> TwoSlopeNorm:
    finite = np.asarray(values)[np.isfinite(values)]
    bound = max(
        abs(float(np.min(finite))),
        abs(float(np.max(finite))),
        np.finfo(float).tiny,
    )
    return TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)


def rotated_map(
    axis: plt.Axes,
    x_um: np.ndarray,
    y_um: np.ndarray,
    values: np.ndarray,
    title: str,
    *,
    cmap: str = "viridis",
    centered: bool = False,
    terminal_masks: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    norm = symmetric_norm(values) if centered else None
    image = axis.pcolormesh(
        x_um.T,
        y_um.T,
        np.asarray(values).T,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    if terminal_masks is not None:
        low, high = terminal_masks
        axis.contour(
            x_um.T,
            y_um.T,
            low.T.astype(float),
            levels=(0.5,),
            colors=("#f28e2b",),
            linewidths=1.0,
        )
        axis.contour(
            x_um.T,
            y_um.T,
            high.T.astype(float),
            levels=(0.5,),
            colors=("#e15759",),
            linewidths=1.0,
        )
    axis.set_aspect("equal")
    axis.set_xlim(-17.5, 17.5)
    axis.set_ylim(-17.5, 17.5)
    axis.set_xlabel("x = b (um)")
    axis.set_ylabel("y = a (um)")
    axis.set_title(title, fontsize=9)
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.03)


def rectangular_map(
    axis: plt.Axes,
    x_um: np.ndarray,
    y_um: np.ndarray,
    values: np.ndarray,
    title: str,
) -> None:
    image = axis.pcolormesh(
        x_um,
        y_um,
        np.asarray(values).T,
        shading="auto",
        cmap="inferno",
    )
    edge = 12.0 * np.sqrt(2.0)
    axis.plot((0, edge, 0, -edge, 0), (edge, 0, -edge, 0, edge), color="white", lw=0.9)
    axis.set_aspect("equal")
    axis.set_xlim(-18.0, 18.0)
    axis.set_ylim(-18.0, 18.0)
    axis.set_xlabel("x = b (um)")
    axis.set_ylabel("y = a (um)")
    axis.set_title(title, fontsize=9)
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.03)


def load_run(run: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = RUNS[run]
    final = json.loads((root / "FINAL_RESULT.json").read_text())
    chosen = final["chosen_candidate"]["result"]
    with np.load(root / "chosen_exact_candidate_fields.npz") as raw:
        rho = np.asarray(raw["rho_binary"], dtype=np.float64)
        q = np.asarray(raw["mapped_Q_W_m3"], dtype=np.float64)
        temperature = np.asarray(raw["nodal_temperature_K"], dtype=np.float64)
        psi = np.asarray(raw["weighting_potential"], dtype=np.float64)
        grad_psi_element = np.asarray(
            raw["weighting_gradient_element_m_inv"], dtype=np.float64
        )
    audit, _ = exact_binary_audit(
        rho,
        geometry_mode="diagonal_45_contact_anchored",
        contact_axis="diagonal_45",
    )
    if not audit["passed"] or not np.array_equal(np.unique(rho), (0.0, 1.0)):
        raise RuntimeError(f"Run {run:03d} selected geometry is not exact-feasible")

    source_power = float(chosen["forward"]["source_power_W"])
    scale = TARGET_POWER_W / source_power
    mesh = build_rotated_device_mesh(24.0e-6, STEP_M)
    tri = mesh.triangles
    rho_element = np.mean(rho.ravel()[tri], axis=1)
    grad_t_element = np.einsum(
        "eai,ei->ea", mesh.gradients_m_inv, temperature.ravel()[tri]
    )
    alpha_xy = np.column_stack(
        (
            SIGMA_XY_S_M[0] * SEEBECK_XY_V_K[0] * rho_element**2,
            SIGMA_XY_S_M[1] * SEEBECK_XY_V_K[1] * rho_element**2,
        )
    )
    contribution_element_xy_A = (
        -THICKNESS_M
        * mesh.triangle_area_m2[:, None]
        * grad_psi_element
        * alpha_xy
        * grad_t_element
        * scale
    )
    contribution_cell_xy_A_m2 = (
        element_to_cell(contribution_element_xy_A, sum_triangles=True)
        / (STEP_M * STEP_M)
    )
    contribution_cell_total_A_m2 = np.sum(contribution_cell_xy_A_m2, axis=2)
    reintegrated_A = float(np.sum(contribution_element_xy_A))
    certified_A = float(chosen["equivalent_objective_at_285uW_A"])
    reintegration_error = abs(reintegrated_A - certified_A) / max(
        abs(certified_A), np.finfo(float).tiny
    )
    if reintegration_error > 2.0e-12:
        raise RuntimeError(
            f"Run {run:03d} current reintegration error {reintegration_error}"
        )

    short = solve_short_circuit_current_density(
        mesh,
        rho,
        temperature,
        thickness_m=THICKNESS_M,
        sigma_xy_S_m=SIGMA_XY_S_M,
        seebeck_xy_V_K=SEEBECK_XY_V_K,
        sigma_void_fraction=CONTRACT.sigma_void_fraction,
        sigma_penalty=CONTRACT.sigma_penalty,
        alpha_penalty=CONTRACT.alpha_penalty,
        terminal_axis="diagonal_45",
    )
    short_current_A = float(short.terminal_current_A * scale)
    short_error = abs(short_current_A - certified_A) / max(
        abs(certified_A), np.finfo(float).tiny
    )
    if short_error > 2.0e-7 or short.continuity_residual > 1.0e-8:
        raise RuntimeError(
            f"Run {run:03d} short-circuit current check failed: "
            f"current={short_error}, continuity={short.continuity_residual}"
        )

    grad_t_cell = element_to_cell(grad_t_element) * scale
    grad_psi_cell = element_to_cell(grad_psi_element)
    electric_cell = element_to_cell(short.electric_field_element_V_m) * scale
    j_thermo_cell = element_to_cell(
        short.thermoelectric_current_density_element_A_m2
    ) * scale
    j_conductive_cell = element_to_cell(
        short.conductive_current_density_element_A_m2
    ) * scale
    j_total_cell = element_to_cell(short.total_current_density_element_A_m2) * scale
    rho_cell = 0.25 * (
        rho[:-1, :-1] + rho[1:, :-1] + rho[:-1, 1:] + rho[1:, 1:]
    )

    x_nodes_um, y_nodes_um = rotated_coordinates(rho)
    x_cells_um, y_cells_um = rotated_coordinates(rho_cell)
    node_masks_flat = CONTRACT.terminal_node_masks(mesh.nodes_m)
    terminal_masks = tuple(mask.reshape(rho.shape) for mask in node_masks_flat)

    x_edges, y_edges, z_edges = _piecewise_edges()
    dx, dy, dz = np.diff(x_edges), np.diff(y_edges), np.diff(z_edges)
    if q.shape != (dx.size, dy.size, dz.size):
        raise RuntimeError(f"Run {run:03d} Q/thermal grid mismatch")
    q_scaled = q * scale
    qxy_W_m2 = np.sum(q_scaled * dz[None, None, :], axis=2)
    q_power_W = float(
        np.sum(q_scaled * dx[:, None, None] * dy[None, :, None] * dz[None, None, :])
    )
    qx_um = 0.5e6 * (x_edges[:-1] + x_edges[1:])
    qy_um = 0.5e6 * (y_edges[:-1] + y_edges[1:])

    arrays = {
        "rho_binary": rho.astype(np.uint8),
        "x_nodes_global_b_um": x_nodes_um,
        "y_nodes_global_a_um": y_nodes_um,
        "x_cells_global_b_um": x_cells_um,
        "y_cells_global_a_um": y_cells_um,
        "temperature_rise_nodal_K": temperature * scale,
        "temperature_gradient_b_cell_K_m": grad_t_cell[:, :, 0],
        "temperature_gradient_a_cell_K_m": grad_t_cell[:, :, 1],
        "temperature_gradient_magnitude_cell_K_m": np.linalg.norm(grad_t_cell, axis=2),
        "weighting_potential": psi,
        "weighting_gradient_b_cell_m_inv": grad_psi_cell[:, :, 0],
        "weighting_gradient_a_cell_m_inv": grad_psi_cell[:, :, 1],
        "weighting_gradient_magnitude_cell_m_inv": np.linalg.norm(grad_psi_cell, axis=2),
        "short_circuit_potential_nodal_V": short.potential_V * scale,
        "electric_field_b_cell_V_m": electric_cell[:, :, 0],
        "electric_field_a_cell_V_m": electric_cell[:, :, 1],
        "electric_field_magnitude_cell_V_m": np.linalg.norm(electric_cell, axis=2),
        "thermoelectric_J_b_cell_A_m2": j_thermo_cell[:, :, 0],
        "thermoelectric_J_a_cell_A_m2": j_thermo_cell[:, :, 1],
        "conductive_J_b_cell_A_m2": j_conductive_cell[:, :, 0],
        "conductive_J_a_cell_A_m2": j_conductive_cell[:, :, 1],
        "total_J_b_cell_A_m2": j_total_cell[:, :, 0],
        "total_J_a_cell_A_m2": j_total_cell[:, :, 1],
        "total_J_magnitude_cell_A_m2": np.linalg.norm(j_total_cell, axis=2),
        "terminal_current_contribution_b_cell_A_m2": contribution_cell_xy_A_m2[:, :, 0],
        "terminal_current_contribution_a_cell_A_m2": contribution_cell_xy_A_m2[:, :, 1],
        "terminal_current_contribution_total_cell_A_m2": contribution_cell_total_A_m2,
        "mapped_Q_depth_integrated_W_m2": qxy_W_m2,
        "mapped_Q_x_global_b_um": qx_um,
        "mapped_Q_y_global_a_um": qy_um,
    }
    metrics = {
        "run": run,
        "polarization": final["polarization"],
        "status": final["status"],
        "passed": bool(final["passed"]),
        "continuous_current_nA": float(final["continuous_reference_current_A"]) * 1.0e9,
        "exact_current_nA": certified_A * 1.0e9,
        "exact_change_percent": float(final["exact_cleanup_relative_current_change"]) * 100.0,
        "exact_500nm_bad_nodes": int(audit["total_bad_cell_count"]),
        "source_power_scale_to_285uW": scale,
        "mapped_absorbed_power_at_285uW_uW": q_power_W * 1.0e6,
        "maximum_temperature_rise_K": float(np.max(arrays["temperature_rise_nodal_K"])),
        "maximum_temperature_gradient_K_m": float(np.max(arrays["temperature_gradient_magnitude_cell_K_m"])),
        "maximum_total_J_A_m2": float(np.max(arrays["total_J_magnitude_cell_A_m2"])),
        "positive_current_contribution_nA": float(
            np.sum(np.maximum(contribution_element_xy_A.sum(axis=1), 0.0)) * 1.0e9
        ),
        "negative_current_contribution_nA": float(
            np.sum(np.minimum(contribution_element_xy_A.sum(axis=1), 0.0)) * 1.0e9
        ),
        "b_current_contribution_nA": float(
            np.sum(contribution_element_xy_A[:, 0]) * 1.0e9
        ),
        "a_current_contribution_nA": float(
            np.sum(contribution_element_xy_A[:, 1]) * 1.0e9
        ),
        "current_reintegration_relative_error": reintegration_error,
        "short_circuit_current_nA": short_current_A * 1.0e9,
        "short_circuit_current_relative_error": short_error,
        "short_circuit_continuity_residual": float(short.continuity_residual),
    }
    arrays["terminal_low_mask"] = terminal_masks[0]
    arrays["terminal_high_mask"] = terminal_masks[1]
    return metrics, arrays


def field_figure(metrics: dict[str, Any], data: dict[str, np.ndarray]) -> Path:
    run = int(metrics["run"])
    x_n, y_n = data["x_nodes_global_b_um"], data["y_nodes_global_a_um"]
    x_c, y_c = data["x_cells_global_b_um"], data["y_cells_global_a_um"]
    terminal_masks = (data["terminal_low_mask"], data["terminal_high_mask"])
    fig, axes = plt.subplots(4, 4, figsize=(22, 21), constrained_layout=True)
    rotated_map(
        axes[0, 0], x_n, y_n, data["rho_binary"],
        "Exact binary (black=TaIrTe4); contours=ideal terminals",
        cmap="gray_r", terminal_masks=terminal_masks,
    )
    rectangular_map(
        axes[0, 1], data["mapped_Q_x_global_b_um"], data["mapped_Q_y_global_a_um"],
        data["mapped_Q_depth_integrated_W_m2"], "Depth-integrated absorbed Q (W/m2)",
    )
    rotated_map(
        axes[0, 2], x_n, y_n, data["temperature_rise_nodal_K"],
        "Temperature rise at 285 uW (K)", cmap="inferno",
    )
    rotated_map(
        axes[0, 3], x_c, y_c, data["temperature_gradient_magnitude_cell_K_m"],
        "|grad T| (K/m)", cmap="magma",
    )
    rotated_map(
        axes[1, 0], x_c, y_c, data["temperature_gradient_b_cell_K_m"],
        "dT/db (K/m)", cmap="coolwarm", centered=True,
    )
    rotated_map(
        axes[1, 1], x_c, y_c, data["temperature_gradient_a_cell_K_m"],
        "dT/da (K/m)", cmap="coolwarm", centered=True,
    )
    rotated_map(
        axes[1, 2], x_n, y_n, data["weighting_potential"],
        "Weighting potential psi", cmap="viridis", terminal_masks=terminal_masks,
    )
    rotated_map(
        axes[1, 3], x_c, y_c, data["weighting_gradient_magnitude_cell_m_inv"],
        "|grad psi| (1/m)", cmap="cividis",
    )
    rotated_map(
        axes[2, 0], x_n, y_n, data["short_circuit_potential_nodal_V"] * 1.0e6,
        "Short-circuit potential (uV)", cmap="coolwarm", centered=True,
    )
    rotated_map(
        axes[2, 1], x_c, y_c, data["electric_field_magnitude_cell_V_m"],
        "Short-circuit |E| (V/m)", cmap="cividis",
    )
    rotated_map(
        axes[2, 2], x_c, y_c, data["total_J_b_cell_A_m2"],
        "Total local J_b (A/m2)", cmap="coolwarm", centered=True,
    )
    rotated_map(
        axes[2, 3], x_c, y_c, data["total_J_a_cell_A_m2"],
        "Total local J_a (A/m2)", cmap="coolwarm", centered=True,
    )
    rotated_map(
        axes[3, 0], x_c, y_c, data["total_J_magnitude_cell_A_m2"],
        "Total local |J| (A/m2)", cmap="plasma",
    )
    rotated_map(
        axes[3, 1], x_c, y_c,
        data["terminal_current_contribution_total_cell_A_m2"] / 1.0e3,
        "Signed terminal contribution (nA/um2)", cmap="coolwarm", centered=True,
    )
    rotated_map(
        axes[3, 2], x_c, y_c,
        data["terminal_current_contribution_b_cell_A_m2"] / 1.0e3,
        "b contribution (nA/um2)", cmap="coolwarm", centered=True,
    )
    rotated_map(
        axes[3, 3], x_c, y_c,
        data["terminal_current_contribution_a_cell_A_m2"] / 1.0e3,
        "a contribution (nA/um2)", cmap="coolwarm", centered=True,
    )
    status = "PASS" if metrics["passed"] else "1% GATE FAIL"
    fig.suptitle(
        f"Run {run:03d} E||{metrics['polarization'][-1]}: rotated 45 deg, no Au, "
        f"evaporated SiO2 | I={metrics['exact_current_nA']:+.3f} nA | {status}",
        fontsize=15,
    )
    output = REPORT_DIR / f"run{run:03d}_rotated45_no_Au_exact_fields.png"
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def convergence_figure(metrics: list[dict[str, Any]]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    for row in metrics:
        run = int(row["run"])
        history = json.loads((RUNS[run] / "optimization_history.json").read_text())
        evaluation = np.asarray([item["global_full_physics_evaluation"] for item in history])
        current = np.asarray([item["objective_at_reference_power_A"] for item in history]) * 1.0e9
        gray = np.asarray([item["gray_fraction_0p01_0p99"] for item in history]) * 100.0
        bad = np.asarray([item["exact_bad_cells"] for item in history])
        beta = np.asarray([item["beta"] for item in history])
        label = f"Run {run:03d} E||{row['polarization'][-1]}"
        axes[0, 0].plot(evaluation, current, label=label)
        axes[0, 1].plot(evaluation, gray, label=label)
        axes[1, 0].plot(evaluation, bad, label=label)
        axes[1, 1].step(evaluation, beta, where="post", label=label)
    axes[0, 0].set(ylabel="Continuous terminal current at 285 uW (nA)")
    axes[0, 1].set(ylabel="Gray nodes 0.01 < rho < 0.99 (%)")
    axes[1, 0].set(ylabel="Thresholded exact-500-nm bad nodes", xlabel="Evaluation")
    axes[1, 1].set(ylabel="Projection beta", xlabel="Evaluation", yscale="log", yticks=(1, 2, 4, 8, 16, 32, 64, 128))
    axes[1, 1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Run059/060 optimization convergence")
    output = REPORT_DIR / "run059_060_optimization_convergence.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def comparison_figure(metrics: list[dict[str, Any]]) -> Path:
    labels = [f"Run {row['run']:03d}\nE||{row['polarization'][-1]}" for row in metrics]
    x = np.arange(len(metrics))
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), constrained_layout=True)
    continuous = np.asarray([row["continuous_current_nA"] for row in metrics])
    exact = np.asarray([row["exact_current_nA"] for row in metrics])
    axes[0].bar(x - 0.18, continuous, 0.36, color="0.6", label="continuous")
    axes[0].bar(x + 0.18, exact, 0.36, color=("#4e79a7", "#f28e2b"), label="exact")
    axes[0].set(ylabel="Terminal current at 285 uW (nA)", xticks=x, xticklabels=labels)
    axes[0].legend()
    change = np.asarray([row["exact_change_percent"] for row in metrics])
    axes[1].bar(x, change, color=("#4e79a7", "#f28e2b"))
    axes[1].axhline(-1.0, color="#e15759", ls="--", label="1% gate")
    axes[1].set(ylabel="Exact cleanup current change (%)", xticks=x, xticklabels=labels)
    axes[1].legend()
    positive = np.asarray([row["positive_current_contribution_nA"] for row in metrics])
    negative = np.asarray([row["negative_current_contribution_nA"] for row in metrics])
    axes[2].bar(x, positive, color="#59a14f", label="positive")
    axes[2].bar(x, negative, color="#e15759", label="negative")
    axes[2].scatter(x, exact, color="black", marker="x", s=70, label="net")
    axes[2].set(ylabel="Integrated signed contribution (nA)", xticks=x, xticklabels=labels)
    axes[2].legend()
    b_term = np.asarray([row["b_current_contribution_nA"] for row in metrics])
    a_term = np.asarray([row["a_current_contribution_nA"] for row in metrics])
    axes[3].bar(x - 0.18, b_term, 0.36, color="#4e79a7", label="b term")
    axes[3].bar(x + 0.18, a_term, 0.36, color="#f28e2b", label="a term")
    axes[3].set(ylabel="Axis current contribution (nA)", xticks=x, xticklabels=labels)
    axes[3].legend()
    for axis in axes:
        axis.axhline(0.0, color="black", lw=0.8)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Run059/060 final exact-binary current comparison")
    output = REPORT_DIR / "run059_060_final_current_comparison.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> int:
    CONTRACT.validate()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, Any]] = []
    outputs: list[Path] = []
    for run in RUNS:
        row, arrays = load_run(run)
        metrics.append(row)
        outputs.append(field_figure(row, arrays))
        data_path = REPORT_DIR / f"run{run:03d}_rotated45_no_Au_derived_fields.npz"
        np.savez_compressed(data_path, **arrays)
        outputs.append(data_path)
    outputs.append(convergence_figure(metrics))
    outputs.append(comparison_figure(metrics))
    summary = {
        "schema": "run059-060-rotated45-no-Au-field-publication-v1",
        "model": {
            "geometry": "24 x 24 um device rotated +45 degrees",
            "axis_contract": "global x=b, y=a",
            "optical": "Run58 axis-aligned optical proxy without Au",
            "thermal_interface": "evaporated TaIrTe4/SiO2, G=73700 W/m2/K",
            "electrodes": "ideal equipotential masks only in electrical solves",
            "target_incident_power_W": TARGET_POWER_W,
        },
        "runs": metrics,
        "published_outputs": [
            {"path": str(path.relative_to(REPOSITORY)), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in outputs
        ],
        "optimization_rerun": False,
    }
    summary_path = REPORT_DIR / "run059_060_field_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
