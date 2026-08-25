#!/usr/bin/env python3
"""Publish complete Run061/062 exact-binary field maps without rerunning FDTD."""

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
    build_rectangular_mesh,
    build_rotated_device_mesh,
    solve_short_circuit_current_density,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.rotated_device import (  # noqa: E402
    device_to_crystal_coordinates,
)


REPORT_DIR = (
    REPOSITORY
    / "photothermal_pte"
    / "reports"
    / "run061_062_dual_polarization_thermally_grown"
)
TARGET_POWER_W = 285.0e-6
STEP_M = 100.0e-9
THICKNESS_M = 100.0e-9
SIGMA_XY_S_M = (1.10e5, 4.91e5)
SEEBECK_XY_V_K = (27.0e-6, -6.0e-6)

RUNS = {
    61: {
        "geometry": "top_bottom",
        "root": REPOSITORY
        / "photothermal_pte/optimization_runs/run_061_top_bottom_thermally_grown_sio2_dual_polarization/results",
    },
    62: {
        "geometry": "diagonal_45",
        "root": REPOSITORY
        / "photothermal_pte/optimization_runs/run_062_diagonal45_thermally_grown_sio2_dual_polarization/results",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def full_flake_density(run: int, design: np.ndarray) -> np.ndarray:
    rho = np.asarray(design, dtype=np.float64)
    if run == 62:
        if rho.shape != (241, 241):
            raise ValueError(f"Run062 density shape is {rho.shape}, expected (241, 241)")
        return rho
    if rho.shape != (241, 201):
        raise ValueError(f"Run061 density shape is {rho.shape}, expected (241, 201)")
    full = np.ones((241, 241), dtype=np.float64)
    full[:, 20:221] = rho
    return full


def local_coordinates(count: int, *, cells: bool) -> tuple[np.ndarray, np.ndarray]:
    if cells:
        values = np.linspace(-12.0e-6 + 0.5 * STEP_M, 12.0e-6 - 0.5 * STEP_M, count)
    else:
        values = np.linspace(-12.0e-6, 12.0e-6, count)
    return np.meshgrid(values, values, indexing="ij")


def display_coordinates(run: int, count: int, *, cells: bool) -> tuple[np.ndarray, np.ndarray]:
    u, v = local_coordinates(count, cells=cells)
    if run == 62:
        x, y = device_to_crystal_coordinates(u, v)
    else:
        x, y = u, v
    return x * 1.0e6, y * 1.0e6


def element_to_cell(values: np.ndarray, *, sum_triangles: bool = False) -> np.ndarray:
    array = np.asarray(values)
    cells = 240 * 240
    if array.shape[0] != 2 * cells:
        raise ValueError(f"unexpected triangular-element array shape {array.shape}")
    combined = array[:cells] + array[cells:]
    if not sum_triangles:
        combined *= 0.5
    return combined.reshape(240, 240, *array.shape[1:])


def thermal_edges(diagonal: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if diagonal:
        negative_outer = np.asarray((-32, -28, -24, -20, -19), float) * 1.0e-6
        negative_shoulder = np.arange(-19.0, -17.0, 0.25) * 1.0e-6
        core = np.arange(-17.0, 17.0 + 0.05, 0.1) * 1.0e-6
        positive_shoulder = np.arange(17.25, 19.0 + 0.125, 0.25) * 1.0e-6
        positive_outer = np.asarray((20, 24, 28, 32), float) * 1.0e-6
    else:
        negative_outer = np.asarray((-32, -28, -24, -20, -16, -14), float) * 1.0e-6
        negative_shoulder = np.arange(-14.0, -12.0, 0.25) * 1.0e-6
        core = np.arange(-12.0, 12.0 + 0.05, 0.1) * 1.0e-6
        positive_shoulder = np.arange(12.25, 14.0 + 0.125, 0.25) * 1.0e-6
        positive_outer = np.asarray((16, 20, 24, 28, 32), float) * 1.0e-6
    lateral = np.unique(
        np.concatenate(
            (negative_outer, negative_shoulder, core, positive_shoulder, positive_outer)
        )
    )
    z = np.asarray(
        (
            -20.0,
            -12.0,
            -8.0,
            -5.0,
            -3.0,
            -2.0,
            -1.25,
            -0.8,
            -0.55,
            -0.385,
            -0.30,
            -0.20,
            -0.10,
            -0.09,
            -0.08,
            -0.07,
            -0.06,
            -0.05,
            -0.04,
            -0.03,
            -0.02,
            -0.01,
            0.0,
            0.01,
            0.02,
            0.05,
            0.10,
            0.20,
            0.40,
            0.70,
            1.0,
            1.25,
            1.50,
            2.0,
        ),
        dtype=float,
    ) * 1.0e-6
    return lateral, lateral.copy(), z


def contact_masks(run: int, x_m: np.ndarray, y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if run == 62:
        u = (x_m + y_m) / np.sqrt(2.0)
        return u <= -10.0e-6 + 1.0e-18, u >= 10.0e-6 - 1.0e-18
    return y_m <= -10.0e-6 + 1.0e-18, y_m >= 10.0e-6 - 1.0e-18


def symmetric_norm(values: np.ndarray) -> TwoSlopeNorm:
    finite = np.asarray(values)[np.isfinite(values)]
    bound = max(
        abs(float(np.min(finite))),
        abs(float(np.max(finite))),
        np.finfo(float).tiny,
    )
    return TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)


def flake_outline(run: int) -> tuple[np.ndarray, np.ndarray]:
    if run == 62:
        edge = 12.0 * np.sqrt(2.0)
        return np.asarray((0.0, edge, 0.0, -edge, 0.0)), np.asarray(
            (edge, 0.0, -edge, 0.0, edge)
        )
    return np.asarray((-12.0, 12.0, 12.0, -12.0, -12.0)), np.asarray(
        (-12.0, -12.0, 12.0, 12.0, -12.0)
    )


def field_map(
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
    image = axis.pcolormesh(
        x_um.T,
        y_um.T,
        np.asarray(values).T,
        shading="auto",
        cmap=cmap,
        norm=symmetric_norm(values) if centered else None,
        rasterized=True,
    )
    if terminal_masks is not None:
        for mask, color in zip(terminal_masks, ("#f28e2b", "#e15759")):
            axis.contour(
                x_um.T,
                y_um.T,
                mask.T.astype(float),
                levels=(0.5,),
                colors=(color,),
                linewidths=1.2,
            )
    axis.set_aspect("equal")
    axis.set_xlim(-17.8, 17.8)
    axis.set_ylim(-17.8, 17.8)
    axis.set_xlabel("x = b (um)")
    axis.set_ylabel("y = a (um)")
    axis.set_title(title, fontsize=9)
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.03)


def q_map(
    axis: plt.Axes,
    run: int,
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
        rasterized=True,
    )
    outline_x, outline_y = flake_outline(run)
    axis.plot(outline_x, outline_y, color="white", lw=1.0)
    axis.set_aspect("equal")
    axis.set_xlim(-18.0, 18.0)
    axis.set_ylim(-18.0, 18.0)
    axis.set_xlabel("x = b (um)")
    axis.set_ylabel("y = a (um)")
    axis.set_title(title, fontsize=9)
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.03)


def load_case(run: int, polarization: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = Path(RUNS[run]["root"])
    final = json.loads((root / "FINAL_RESULT.json").read_text())
    result = final["chosen_candidate"]["result"]["polarization_results"][polarization]
    artifact = result["raw_artifact"]
    raw_path = Path(artifact["path"])
    if not raw_path.is_file() or sha256(raw_path) != artifact["sha256"]:
        raise RuntimeError(f"missing or changed selected field artifact: {raw_path}")
    with np.load(raw_path) as raw:
        rho = full_flake_density(run, raw["rho_binary"])
        q = np.asarray(raw["mapped_Q_W_m3"], dtype=np.float64)
        temperature = np.asarray(raw["nodal_temperature_K"], dtype=np.float64)
        psi = np.asarray(raw["weighting_potential"], dtype=np.float64)
        grad_psi_element = np.asarray(
            raw["weighting_gradient_element_m_inv"], dtype=np.float64
        )

    exact = final["exact_binary_audit"]
    if not exact["passed"] or exact["total_bad_cell_count"] != 0:
        raise RuntimeError(f"Run{run:03d} final geometry is not exact-feasible")
    if not np.array_equal(np.unique(rho), (0.0, 1.0)):
        raise RuntimeError(f"Run{run:03d} full density is not exact binary")

    source_power = float(result["forward"]["source_power_W"])
    scale = TARGET_POWER_W / source_power
    diagonal = run == 62
    mesh = (
        build_rotated_device_mesh(24.0e-6, STEP_M)
        if diagonal
        else build_rectangular_mesh(24.0e-6, 24.0e-6, STEP_M)
    )
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
    contribution_cell_total_A_m2 = contribution_cell_xy_A_m2.sum(axis=2)
    certified_A = float(result["equivalent_objective_at_285uW_A"])
    reintegrated_A = float(np.sum(contribution_element_xy_A))
    reintegration_error = abs(reintegrated_A - certified_A) / max(
        abs(certified_A), np.finfo(float).tiny
    )
    if reintegration_error > 2.0e-12:
        raise RuntimeError(f"Run{run:03d} {polarization} current reintegration failed")

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
        terminal_axis="diagonal_45" if diagonal else "y",
    )
    short_current_A = float(short.terminal_current_A * scale)
    short_error = abs(short_current_A - certified_A) / max(
        abs(certified_A), np.finfo(float).tiny
    )
    if short_error > 2.0e-7 or short.continuity_residual > 1.0e-8:
        raise RuntimeError(f"Run{run:03d} {polarization} short-circuit check failed")

    grad_t_cell = element_to_cell(grad_t_element) * scale
    grad_psi_cell = element_to_cell(grad_psi_element)
    electric_cell = element_to_cell(short.electric_field_element_V_m) * scale
    j_thermo_cell = element_to_cell(short.thermoelectric_current_density_element_A_m2) * scale
    j_conductive_cell = element_to_cell(short.conductive_current_density_element_A_m2) * scale
    j_total_cell = element_to_cell(short.total_current_density_element_A_m2) * scale
    x_nodes_um, y_nodes_um = display_coordinates(run, 241, cells=False)
    x_cells_um, y_cells_um = display_coordinates(run, 240, cells=True)
    x_nodes_m = x_nodes_um * 1.0e-6
    y_nodes_m = y_nodes_um * 1.0e-6
    terminals = contact_masks(run, x_nodes_m, y_nodes_m)

    x_edges, y_edges, z_edges = thermal_edges(diagonal)
    dx, dy, dz = np.diff(x_edges), np.diff(y_edges), np.diff(z_edges)
    if q.shape != (dx.size, dy.size, dz.size):
        raise RuntimeError(f"Run{run:03d} Q shape {q.shape} does not match thermal grid")
    q_scaled = q * scale
    qxy_W_m2 = np.sum(q_scaled * dz[None, None, :], axis=2)
    q_power_W = float(
        np.sum(q_scaled * dx[:, None, None] * dy[None, :, None] * dz[None, None, :])
    )

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
        "mapped_Q_x_global_b_um": 0.5e6 * (x_edges[:-1] + x_edges[1:]),
        "mapped_Q_y_global_a_um": 0.5e6 * (y_edges[:-1] + y_edges[1:]),
        "terminal_low_mask": terminals[0],
        "terminal_high_mask": terminals[1],
    }
    exact_A = float(final["exact_reference_currents_A"][polarization])
    continuous_A = float(final["continuous_reference_currents_A"][polarization])
    metrics = {
        "run": run,
        "geometry": RUNS[run]["geometry"],
        "polarization": polarization,
        "global_status": final["status"],
        "global_passed": bool(final["passed"]),
        "polarization_gate_passed": bool(
            final["per_polarization_objective_preservation_gate_passed"][polarization]
        ),
        "continuous_current_nA": continuous_A * 1.0e9,
        "exact_current_nA": exact_A * 1.0e9,
        "exact_change_percent": 100.0 * (exact_A / continuous_A - 1.0),
        "exact_500nm_bad_nodes": int(exact["total_bad_cell_count"]),
        "source_power_scale_to_285uW": scale,
        "mapped_absorbed_power_at_285uW_uW": q_power_W * 1.0e6,
        "maximum_temperature_rise_K": float(np.max(arrays["temperature_rise_nodal_K"])),
        "maximum_temperature_gradient_K_m": float(
            np.max(arrays["temperature_gradient_magnitude_cell_K_m"])
        ),
        "maximum_total_J_A_m2": float(np.max(arrays["total_J_magnitude_cell_A_m2"])),
        "positive_current_contribution_nA": float(
            np.sum(np.maximum(contribution_element_xy_A.sum(axis=1), 0.0)) * 1.0e9
        ),
        "negative_current_contribution_nA": float(
            np.sum(np.minimum(contribution_element_xy_A.sum(axis=1), 0.0)) * 1.0e9
        ),
        "b_current_contribution_nA": float(np.sum(contribution_element_xy_A[:, 0]) * 1.0e9),
        "a_current_contribution_nA": float(np.sum(contribution_element_xy_A[:, 1]) * 1.0e9),
        "current_reintegration_relative_error": reintegration_error,
        "short_circuit_current_nA": short_current_A * 1.0e9,
        "short_circuit_current_relative_error": short_error,
        "short_circuit_continuity_residual": float(short.continuity_residual),
        "source_field_artifact": artifact,
    }
    return metrics, arrays


def field_figure(metrics: dict[str, Any], data: dict[str, np.ndarray]) -> Path:
    run = int(metrics["run"])
    polarization = str(metrics["polarization"])
    x_n, y_n = data["x_nodes_global_b_um"], data["y_nodes_global_a_um"]
    x_c, y_c = data["x_cells_global_b_um"], data["y_cells_global_a_um"]
    terminals = (data["terminal_low_mask"], data["terminal_high_mask"])
    fig, axes = plt.subplots(4, 4, figsize=(22, 21), constrained_layout=True)
    field_map(
        axes[0, 0],
        x_n,
        y_n,
        data["rho_binary"],
        "Exact binary (black=TaIrTe4); contours=ideal terminals",
        cmap="gray_r",
        terminal_masks=terminals,
    )
    q_map(
        axes[0, 1],
        run,
        data["mapped_Q_x_global_b_um"],
        data["mapped_Q_y_global_a_um"],
        data["mapped_Q_depth_integrated_W_m2"],
        "Depth-integrated absorbed Q (W/m2)",
    )
    field_map(
        axes[0, 2], x_n, y_n, data["temperature_rise_nodal_K"],
        "Temperature rise at 285 uW (K)", cmap="inferno",
    )
    field_map(
        axes[0, 3], x_c, y_c, data["temperature_gradient_magnitude_cell_K_m"],
        "|grad T| (K/m)", cmap="magma",
    )
    field_map(
        axes[1, 0], x_c, y_c, data["temperature_gradient_b_cell_K_m"],
        "dT/db (K/m)", cmap="coolwarm", centered=True,
    )
    field_map(
        axes[1, 1], x_c, y_c, data["temperature_gradient_a_cell_K_m"],
        "dT/da (K/m)", cmap="coolwarm", centered=True,
    )
    field_map(
        axes[1, 2], x_n, y_n, data["weighting_potential"],
        "Weighting potential psi", cmap="viridis", terminal_masks=terminals,
    )
    field_map(
        axes[1, 3], x_c, y_c, data["weighting_gradient_magnitude_cell_m_inv"],
        "|grad psi| (1/m)", cmap="cividis",
    )
    field_map(
        axes[2, 0], x_n, y_n, data["short_circuit_potential_nodal_V"] * 1.0e6,
        "Short-circuit potential (uV)", cmap="coolwarm", centered=True,
    )
    field_map(
        axes[2, 1], x_c, y_c, data["electric_field_magnitude_cell_V_m"],
        "Short-circuit |E| (V/m)", cmap="cividis",
    )
    field_map(
        axes[2, 2], x_c, y_c, data["total_J_b_cell_A_m2"],
        "Total local J_b (A/m2)", cmap="coolwarm", centered=True,
    )
    field_map(
        axes[2, 3], x_c, y_c, data["total_J_a_cell_A_m2"],
        "Total local J_a (A/m2)", cmap="coolwarm", centered=True,
    )
    field_map(
        axes[3, 0], x_c, y_c, data["total_J_magnitude_cell_A_m2"],
        "Total local |J| (A/m2)", cmap="plasma",
    )
    field_map(
        axes[3, 1], x_c, y_c,
        data["terminal_current_contribution_total_cell_A_m2"] / 1.0e3,
        "Signed terminal contribution (nA/um2)", cmap="coolwarm", centered=True,
    )
    field_map(
        axes[3, 2], x_c, y_c,
        data["terminal_current_contribution_b_cell_A_m2"] / 1.0e3,
        "b contribution (nA/um2)", cmap="coolwarm", centered=True,
    )
    field_map(
        axes[3, 3], x_c, y_c,
        data["terminal_current_contribution_a_cell_A_m2"] / 1.0e3,
        "a contribution (nA/um2)", cmap="coolwarm", centered=True,
    )
    geometry = "top-bottom" if run == 61 else "+45-degree diagonal"
    gate = "PASS" if metrics["polarization_gate_passed"] else "1% GATE FAIL"
    fig.suptitle(
        f"Run {run:03d} {polarization}: {geometry}, thermally grown SiO2, no Au | "
        f"I={metrics['exact_current_nA']:+.3f} nA | {gate}",
        fontsize=15,
    )
    output = REPORT_DIR / f"run{run:03d}_{polarization}_exact_fields.png"
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def convergence_figure() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    for run, config in RUNS.items():
        history = json.loads((Path(config["root"]) / "optimization_history.json").read_text())
        evaluation = np.asarray([row["global_full_physics_evaluation"] for row in history])
        for polarization, style in (("Ea", "-"), ("Eb", "--")):
            current = np.asarray(
                [row["polarization_objectives_at_reference_power_A"][polarization] for row in history]
            ) * 1.0e9
            axes[0, 0].plot(evaluation, current, style, label=f"Run {run:03d} {polarization}")
        softmin = np.asarray([row["objective_at_reference_power_A"] for row in history]) * 1.0e9
        gray = np.asarray([row["gray_fraction_0p01_0p99"] for row in history]) * 100.0
        bad = np.asarray([row["exact_bad_cells"] for row in history])
        beta = np.asarray([row["beta"] for row in history])
        axes[0, 0].plot(evaluation, softmin, ":", lw=2.0, label=f"Run {run:03d} soft-min")
        axes[0, 1].plot(evaluation, gray, label=f"Run {run:03d}")
        axes[1, 0].plot(evaluation, bad, label=f"Run {run:03d}")
        axes[1, 1].step(evaluation, beta, where="post", label=f"Run {run:03d}")
    axes[0, 0].set_ylabel("Signed current at 285 uW (nA)")
    axes[0, 1].set_ylabel("Gray nodes 0.01 < rho < 0.99 (%)")
    axes[1, 0].set(ylabel="Thresholded 500-nm bad nodes", xlabel="Evaluation")
    axes[1, 1].set(
        ylabel="Projection beta",
        xlabel="Evaluation",
        yscale="log",
        yticks=(1, 2, 4, 8, 16, 32, 64, 128),
    )
    axes[1, 1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Run061/062 shared-geometry dual-polarization convergence")
    output = REPORT_DIR / "run061_062_optimization_convergence.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def comparison_figure(metrics: list[dict[str, Any]]) -> Path:
    labels = [f"Run {row['run']:03d}\n{row['polarization']}" for row in metrics]
    x = np.arange(len(metrics))
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.8), constrained_layout=True)
    continuous = np.asarray([row["continuous_current_nA"] for row in metrics])
    exact = np.asarray([row["exact_current_nA"] for row in metrics])
    axes[0].bar(x - 0.18, continuous, 0.36, color="0.6", label="continuous")
    axes[0].bar(x + 0.18, exact, 0.36, color="#4e79a7", label="exact")
    axes[0].set(ylabel="Signed current at 285 uW (nA)", xticks=x, xticklabels=labels)
    axes[0].legend()
    changes = np.asarray([row["exact_change_percent"] for row in metrics])
    axes[1].bar(x, changes, color="#f28e2b")
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
    axes[3].set(ylabel="Crystal-axis contribution (nA)", xticks=x, xticklabels=labels)
    axes[3].legend()
    for axis in axes:
        axis.axhline(0.0, color="black", lw=0.8)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Run061/062 final exact-binary current decomposition")
    output = REPORT_DIR / "run061_062_final_current_comparison.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, Any]] = []
    outputs: list[Path] = []
    for run in RUNS:
        for polarization in ("Ea", "Eb"):
            row, arrays = load_case(run, polarization)
            metrics.append(row)
            outputs.append(field_figure(row, arrays))
            data_path = REPORT_DIR / f"run{run:03d}_{polarization}_derived_fields.npz"
            np.savez_compressed(data_path, **arrays)
            outputs.append(data_path)
    outputs.append(convergence_figure())
    outputs.append(comparison_figure(metrics))
    summary = {
        "schema": "run061-062-dual-polarization-complete-field-publication-v1",
        "model": {
            "axis_contract": "global x=b, y=a",
            "run061_geometry": "24 x 24 um top-bottom device",
            "run062_geometry": "24 x 24 um device physically plotted at +45 degrees",
            "run062_optical": "Run58 axis-aligned optical proxy without Au",
            "thermal_interface": "thermally grown TaIrTe4/SiO2, G=7.37e6 W/m2/K",
            "electrodes": "ideal equipotential regions only in electrical solves",
            "target_incident_power_W": TARGET_POWER_W,
        },
        "cases": metrics,
        "published_outputs": [
            {
                "path": str(path.relative_to(REPOSITORY)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in outputs
        ],
        "optimization_rerun": False,
        "postprocessing_only": True,
    }
    summary_path = REPORT_DIR / "run061_062_field_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
