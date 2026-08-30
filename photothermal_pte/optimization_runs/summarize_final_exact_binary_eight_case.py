#!/usr/bin/env python3
"""Publish the exact-binary physics matrix for Runs 044/045/046/048/055--058.

This is an offline publisher.  It does not launch Lumerical or solve a new
thermal/electrical system.  Every selected input is the fresh GPU-Maxwell and
CUDA thermal/electrical artifact evaluated on a 0/1, exact-500-nm-cleaned
candidate.  The script independently reintegrates Q and terminal current,
constructs strict-centred temperature-gradient diagnostics, and records raw
artifact hashes.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.optimization_runs.tairte4_flake_topology.electrical import (
    build_rectangular_mesh,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.optimization_support import (
    exact_binary_audit,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.thermal import (
    _piecewise_edges,
)


REPORT_DIR = REPOSITORY / "photothermal_pte" / "reports" / "final_exact_binary_eight_case_matrix"
TARGET_POWER_W = 285.0e-6
STEP_M = 100.0e-9
THICKNESS_M = 100.0e-9
SIGMA_XY_S_M = (1.10e5, 4.91e5)  # Lumerical x=b, y=a.
SEEBECK_XY_V_K = (27.0e-6, -6.0e-6)


@dataclass(frozen=True)
class Case:
    run: int
    contact_axis: str
    contact_label: str
    interface: str
    conductance: float
    polarization: str
    candidate: str
    field_dir: Path
    density_path: Path
    interface_provenance: str


CASES = (
    Case(44, "y", "top-bottom", "thermally_grown", 7.37e6, "Ea", "void_first",
         Path("/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/run044_exact_500nm_cleanup_20260811/void_first_Ea_gpu_objective_retry2"),
         Path("/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/run044_exact_500nm_cleanup_20260811/void_first_exact_binary_candidate.npz"),
         "legacy thermal.py default plus launch environment; not embedded in the legacy raw JSON"),
    Case(45, "y", "top-bottom", "thermally_grown", 7.37e6, "Eb", "void_first",
         Path("/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/run045_exact_500nm_cleanup_20260811/void_first_Eb_gpu_objective_retry1"),
         Path("/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/run045_exact_500nm_cleanup_20260811/void_first_exact_binary_candidate.npz"),
         "legacy thermal.py default plus launch environment; not embedded in the legacy raw JSON"),
    Case(46, "x", "left-right", "thermally_grown", 7.37e6, "Ea", "solid_first",
         Path("/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored/run046_Ea_current_max/forced_exact_500nm_cleanup/solid_first_objective"),
         Path("/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored/run046_Ea_current_max/forced_exact_500nm_cleanup/solid_first_density.npz"),
         "legacy thermal.py default plus launch environment; not embedded in the legacy raw JSON"),
    Case(48, "x", "left-right", "thermally_grown", 7.37e6, "Eb", "solid_first",
         Path("/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored/run048_Eb_fresh_current_max/forced_exact_500nm_cleanup/solid_first_objective"),
         Path("/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored/run048_Eb_fresh_current_max/forced_exact_500nm_cleanup/solid_first_density.npz"),
         "legacy thermal.py default plus launch environment; not embedded in the legacy raw JSON"),
    Case(55, "y", "top-bottom", "evaporated", 7.37e4, "Ea", "exact_candidate_01",
         Path("/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/run055_bounded_official_dfm_exact_repair_Ea_evaporated_v1/exact_attempt_beta128/exact_candidate_01_physics"),
         Path("/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/run055_bounded_official_dfm_exact_repair_Ea_evaporated_v1/exact_attempt_beta128/exact_candidate_01.npz"),
         "embedded thermal_interface_contract in raw JSON"),
    Case(56, "y", "top-bottom", "evaporated", 7.37e4, "Eb", "exact_candidate_02",
         Path("/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/run056_bounded_official_dfm_exact_repair_Eb_evaporated_v1/exact_attempt_beta128/exact_candidate_02_physics"),
         Path("/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/run056_bounded_official_dfm_exact_repair_Eb_evaporated_v1/exact_attempt_beta128/exact_candidate_02.npz"),
         "embedded thermal_interface_contract in raw JSON"),
    Case(57, "x", "left-right", "evaporated", 7.37e4, "Ea", "exact_candidate_03",
         Path("/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored/run057_bounded_official_dfm_exact_repair_Ea_evaporated_v1/exact_attempt_beta128/exact_candidate_03_physics"),
         Path("/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored/run057_bounded_official_dfm_exact_repair_Ea_evaporated_v1/exact_attempt_beta128/exact_candidate_03.npz"),
         "embedded thermal_interface_contract in raw JSON"),
    Case(58, "x", "left-right", "evaporated", 7.37e4, "Eb", "exact_candidate_00",
         Path("/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored/run058_bounded_official_dfm_exact_repair_Eb_evaporated_v1/exact_attempt_beta128/exact_candidate_00_physics"),
         Path("/data/seunghyun/tairte4/artifacts/tairte4_left_right_contact_anchored/run058_bounded_official_dfm_exact_repair_Eb_evaporated_v1/exact_attempt_beta128/exact_candidate_00.npz"),
         "embedded thermal_interface_contract in raw JSON"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path, role: str) -> dict[str, Any]:
    return {"role": role, "path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def full_density(rho: np.ndarray) -> np.ndarray:
    if rho.ndim != 2 or max(rho.shape) != 241:
        raise RuntimeError(f"unexpected exact density shape {rho.shape}")
    output = np.ones((241, 241), dtype=np.float64)
    i0 = (241 - rho.shape[0]) // 2
    j0 = (241 - rho.shape[1]) // 2
    output[i0:i0 + rho.shape[0], j0:j0 + rho.shape[1]] = rho
    return output


def strict_centered_gradient(temperature: np.ndarray, solid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.zeros_like(solid, dtype=bool)
    valid[1:-1, 1:-1] = (
        solid[1:-1, 1:-1] & solid[:-2, 1:-1] & solid[2:, 1:-1]
        & solid[1:-1, :-2] & solid[1:-1, 2:]
    )
    gx = np.full_like(temperature, np.nan, dtype=np.float64)
    gy = np.full_like(temperature, np.nan, dtype=np.float64)
    central_x = (temperature[2:, 1:-1] - temperature[:-2, 1:-1]) / (2.0 * STEP_M)
    central_y = (temperature[1:-1, 2:] - temperature[1:-1, :-2]) / (2.0 * STEP_M)
    gx[1:-1, 1:-1][valid[1:-1, 1:-1]] = central_x[valid[1:-1, 1:-1]]
    gy[1:-1, 1:-1][valid[1:-1, 1:-1]] = central_y[valid[1:-1, 1:-1]]
    return gx, gy, np.hypot(gx, gy)


def plot_map(ax: plt.Axes, x_um: np.ndarray, y_um: np.ndarray, values: np.ndarray,
             title: str, cmap: str = "magma", centered: bool = False) -> None:
    finite = values[np.isfinite(values)]
    norm = None
    if centered and finite.size:
        bound = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite))), np.finfo(float).tiny)
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    image = ax.pcolormesh(x_um, y_um, values.T, shading="auto", cmap=cmap, norm=norm)
    ax.set_aspect("equal")
    ax.set_xlabel("Lumerical x=b (µm)")
    ax.set_ylabel("Lumerical y=a (µm)")
    ax.set_title(title, fontsize=9)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)


def current_decomposition(rho_full: np.ndarray, temperature: np.ndarray,
                          grad_psi: np.ndarray, scale: float) -> dict[str, Any]:
    mesh = build_rectangular_mesh(24.0e-6, 24.0e-6, STEP_M)
    tri = mesh.triangles
    rho_element = np.mean(rho_full.ravel()[tri], axis=1)
    grad_t = np.einsum("eai,ei->ea", mesh.gradients_m_inv, temperature.ravel()[tri])
    alpha = np.zeros((tri.shape[0], 2), dtype=np.float64)
    alpha[:, 0] = SIGMA_XY_S_M[0] * SEEBECK_XY_V_K[0] * rho_element**2
    alpha[:, 1] = SIGMA_XY_S_M[1] * SEEBECK_XY_V_K[1] * rho_element**2
    element_xy = -THICKNESS_M * mesh.triangle_area_m2[:, None] * grad_psi * alpha * grad_t
    element_xy *= scale
    ncell = 240 * 240
    cell_xy = element_xy[:ncell].reshape(240, 240, 2) + element_xy[ncell:].reshape(240, 240, 2)
    cell_density_xy = cell_xy / (STEP_M * STEP_M)
    total_map = np.sum(cell_density_xy, axis=2)
    return {
        "integrated_total_A": float(np.sum(element_xy)),
        "integrated_b_component_A": float(np.sum(element_xy[:, 0])),
        "integrated_a_component_A": float(np.sum(element_xy[:, 1])),
        "positive_total_A": float(np.sum(np.maximum(element_xy.sum(axis=1), 0.0))),
        "negative_total_A": float(np.sum(np.minimum(element_xy.sum(axis=1), 0.0))),
        "current_density_total_A_m2": total_map,
        "current_density_b_A_m2": cell_density_xy[:, :, 0],
        "current_density_a_A_m2": cell_density_xy[:, :, 1],
        "grad_temperature_element_K_m": grad_t * scale,
    }


def process(case: Case) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    result_path = case.field_dir / "binary_objective_result.json"
    fields_path = case.field_dir / "binary_objective_fields.npz"
    if not (result_path.is_file() and fields_path.is_file() and case.density_path.is_file()):
        raise FileNotFoundError(f"missing selected artifact for Run {case.run:03d}")
    result = json.loads(result_path.read_text())
    with np.load(fields_path) as fields:
        rho = np.asarray(fields["rho_binary"], dtype=np.float64)
        q = np.asarray(fields["mapped_Q_W_m3"], dtype=np.float64)
        temperature = np.asarray(fields["nodal_temperature_K"], dtype=np.float64)
        psi = np.asarray(fields["weighting_potential"], dtype=np.float64)
        grad_psi = np.asarray(fields["weighting_gradient_element_m_inv"], dtype=np.float64)
    with np.load(case.density_path) as selected_density:
        density_keys = [k for k in selected_density.files if np.asarray(selected_density[k]).ndim == 2]
        if not density_keys:
            raise RuntimeError(f"Run {case.run:03d} selected density has no 2-D array")
        rho_from_selected = np.asarray(selected_density[density_keys[0]], dtype=np.float64)
    if not np.array_equal(rho, rho_from_selected):
        raise RuntimeError(f"Run {case.run:03d} field density differs from selected density artifact")
    if sha256(fields_path) != result["raw_artifact"]["sha256"]:
        raise RuntimeError(f"Run {case.run:03d} field SHA differs from raw result manifest")
    if sha256(case.density_path) != result["inputs"]["binary_density"]["sha256"]:
        raise RuntimeError(f"Run {case.run:03d} density SHA differs from raw result manifest")
    if not np.array_equal(np.unique(rho), np.asarray((0.0, 1.0))):
        raise RuntimeError(f"Run {case.run:03d} is not exact binary")
    for name, values in (("Q", q), ("temperature", temperature), ("psi", psi), ("grad_psi", grad_psi)):
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"Run {case.run:03d} {name} contains NaN/Inf")
    audit, _ = exact_binary_audit(
        rho,
        geometry_mode="contact_anchored" if case.contact_axis == "y" else "left_right_contact_anchored",
        contact_axis=case.contact_axis,
    )
    if not audit["passed"]:
        raise RuntimeError(f"Run {case.run:03d} exact audit unexpectedly failed: {audit}")
    rho_full = full_density(rho)
    scale = TARGET_POWER_W / float(result["forward"]["source_power_W"])
    x_edges, y_edges, z_edges = _piecewise_edges()
    widths = (np.diff(x_edges), np.diff(y_edges), np.diff(z_edges))
    if q.shape != tuple(w.size for w in widths):
        raise RuntimeError(f"Run {case.run:03d} Q shape {q.shape} does not match thermal grid")
    volume = widths[0][:, None, None] * widths[1][None, :, None] * widths[2][None, None, :]
    mapped_power_raw = float(np.sum(q * volume))
    # The conservative mapper deliberately attributes only the optical-cell
    # power that intersects an absorbing material volume.  Therefore its
    # target power can be lower than native P_Q; that difference is a physical
    # material-participation diagnostic, not a conservation error.  The
    # mapper's own source-to-target conservation error is retained in the raw
    # result gate.
    material_attributed_fraction = mapped_power_raw / abs(float(result["forward"]["P_Q_W"]))
    q_scaled = q * scale
    qxy = np.sum(q_scaled * widths[2][None, None, :], axis=2)
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    flake_z = (zc >= -0.1e-6 - 1e-18) & (zc < 0.0 + 1e-18)
    qxy_flake = np.sum(q_scaled[:, :, flake_z] * widths[2][None, None, flake_z], axis=2)
    qz_power = np.sum(q_scaled * widths[0][:, None, None] * widths[1][None, :, None], axis=(0, 1)) * widths[2]
    temperature_scaled = temperature * scale
    solid = rho_full >= 0.5
    gx, gy, gmag = strict_centered_gradient(temperature_scaled, solid)
    current = current_decomposition(rho_full, temperature, grad_psi, scale)
    certified_current = float(result["equivalent_objective_at_285uW_A"])
    recompute_error = abs(current["integrated_total_A"] - certified_current) / max(abs(certified_current), np.finfo(float).tiny)
    if recompute_error > 1.0e-12:
        raise RuntimeError(f"Run {case.run:03d} current reintegration failed: I={recompute_error}")
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    q_hot = np.unravel_index(int(np.nanargmax(qxy)), qxy.shape)
    t_masked = np.where(solid, temperature_scaled, np.nan)
    t_hot = np.unravel_index(int(np.nanargmax(t_masked)), t_masked.shape)
    gates = result["gates"]
    physical_pass = bool(
        float(gates["optical_closure"]) < 0.005
        and float(gates["Q_mapping_error"]) < 0.005
        and float(gates["thermal_forward_residual"]) < 1e-8
        and float(gates["thermal_energy_balance"]) < 0.01
        and float(gates["electrical_weighting_residual"]) < 1e-8
    )
    if case.interface == "evaporated":
        embedded = result.get("thermal_interface_contract", {})
        if embedded.get("TaIrTe4_SiO2_scenario") != "evaporated" or not np.isclose(
            float(embedded.get("G_TaIrTe4_SiO2_W_m2K", np.nan)), case.conductance
        ):
            raise RuntimeError(f"Run {case.run:03d} embedded interface contract mismatch")
    metrics = {
        "run": case.run,
        "contact_axis": case.contact_axis,
        "electrodes": case.contact_label,
        "interface_scenario": case.interface,
        "G_TaIrTe4_SiO2_W_m2K": case.conductance,
        "interface_provenance": case.interface_provenance,
        "polarization": case.polarization,
        "selected_candidate": case.candidate,
        "axis_contract": "Lumerical x=b, y=a, z=c",
        "exact_binary": True,
        "exact_500nm_bad_nodes": int(audit["total_bad_cell_count"]),
        "exact_500nm_passed": bool(audit["passed"]),
        "solid_fraction_design": float(audit["solid_fraction"]),
        "raw_source_power_W": float(result["forward"]["source_power_W"]),
        "linear_scale_to_285uW": scale,
        "P_Q_raw_W": float(result["forward"]["P_Q_W"]),
        "P_six_raw_W": float(result["forward"]["P_six_W"]),
        "P_Q_at_285uW_W": float(result["forward"]["P_Q_W"]) * scale,
        "mapped_P_Q_at_285uW_W": mapped_power_raw * scale,
        "Q_mapping_conservation_relative_error": float(gates["Q_mapping_error"]),
        "material_attributed_fraction_of_native_P_Q": material_attributed_fraction,
        "negative_Q_cell_count": int(np.count_nonzero(q < 0.0)),
        "Q_min_W_m3": float(np.min(q)),
        "six_face_closure": float(result["forward"]["closure"]),
        "current_continuous_at_285uW_A": float(result["reference_continuous_objective_A"]) * scale,
        "current_exact_at_285uW_A": certified_current,
        "current_change_from_continuous_fraction": float(result["relative_objective_change_from_continuous"]),
        "one_percent_objective_preservation_passed": bool(result["binary_objective_preserved_within_one_percent"]),
        "current_reintegration_relative_error": recompute_error,
        "current_b_component_at_285uW_A": current["integrated_b_component_A"],
        "current_a_component_at_285uW_A": current["integrated_a_component_A"],
        "current_positive_at_285uW_A": current["positive_total_A"],
        "current_negative_at_285uW_A": current["negative_total_A"],
        "terminal_conductance_S": float(result["terminal_conductance_S"]),
        "Tmax_at_285uW_K": float(np.nanmax(t_masked)),
        "T_hotspot_x_b_um": float((-12.0 + 0.1 * t_hot[0])),
        "T_hotspot_y_a_um": float((-12.0 + 0.1 * t_hot[1])),
        "Qxy_hotspot_x_b_um": float(xc[q_hot[0]] * 1e6),
        "Qxy_hotspot_y_a_um": float(yc[q_hot[1]] * 1e6),
        "max_strict_abs_dT_db_K_m": float(np.nanmax(np.abs(gx))),
        "max_strict_abs_dT_da_K_m": float(np.nanmax(np.abs(gy))),
        "max_strict_gradT_K_m": float(np.nanmax(gmag)),
        "strict_gradient_valid_node_fraction": float(np.count_nonzero(np.isfinite(gmag)) / gmag.size),
        "thermal_forward_residual": float(gates["thermal_forward_residual"]),
        "thermal_energy_balance": float(gates["thermal_energy_balance"]),
        "electrical_weighting_residual": float(gates["electrical_weighting_residual"]),
        "physical_gates_passed": physical_pass,
        "no_Q_postprocessing": bool(not result["Q_clipping_smoothing_gain_or_rescaling"]),
        "raw_result_status_preserved": result["status"],
    }
    arrays = {
        "rho": rho,
        "rho_full": rho_full,
        "qxy": qxy,
        "qxy_flake": qxy_flake,
        "qz_power": qz_power,
        "temperature": temperature_scaled,
        "psi": psi,
        "gx": gx,
        "gy": gy,
        "gmag": gmag,
        "current_total": current["current_density_total_A_m2"],
        "current_b": current["current_density_b_A_m2"],
        "current_a": current["current_density_a_A_m2"],
        "xc_um": xc * 1e6,
        "yc_um": yc * 1e6,
        "zc_um": zc * 1e6,
    }
    manifest = [
        artifact(result_path, f"Run {case.run:03d} exact-binary scalar result"),
        artifact(fields_path, f"Run {case.run:03d} exact-binary Q/T/psi fields"),
        artifact(case.density_path, f"Run {case.run:03d} selected exact-binary density"),
        artifact(Path(result["forward"]["project"]["path"]), f"Run {case.run:03d} raw FSP (not committed)"),
    ]
    for raw_path in sorted(case.field_dir.glob("*_output.h5")) + sorted(case.field_dir.glob("*_p0.log")):
        manifest.append(artifact(raw_path, f"Run {case.run:03d} raw Maxwell engine artifact (not committed)"))
    return metrics, arrays, manifest


def per_case_plots(case: Case, metrics: dict[str, Any], a: dict[str, np.ndarray]) -> list[Path]:
    stem = f"run{case.run:03d}_{case.interface}_{case.contact_label}_{case.polarization}"
    outputs: list[Path] = []
    nodes = np.linspace(-12.0, 12.0, 241)
    cells = 0.5 * (nodes[:-1] + nodes[1:])
    # The main 12-panel physical-field certificate.
    fig, axes = plt.subplots(3, 4, figsize=(21, 15), constrained_layout=True)
    plot_map(axes[0, 0], nodes, nodes, a["rho_full"], "exact binary: black=TaIrTe4", "gray_r")
    plot_map(axes[0, 1], a["xc_um"], a["yc_um"], a["qxy"], "all-material depth-integrated Q (W/m²)")
    plot_map(axes[0, 2], a["xc_um"], a["yc_um"], a["qxy_flake"], "TaIrTe4-depth Q (W/m²)")
    plot_map(axes[0, 3], nodes, nodes, np.where(a["rho_full"] > 0.5, a["temperature"], np.nan), "TaIrTe4 ΔT at 285 µW (K)")
    plot_map(axes[1, 0], nodes, nodes, a["gx"], "strict-centered ∂T/∂b (K/m)", "coolwarm", True)
    plot_map(axes[1, 1], nodes, nodes, a["gy"], "strict-centered ∂T/∂a (K/m)", "coolwarm", True)
    plot_map(axes[1, 2], nodes, nodes, a["gmag"], "strict-centered |∇T| (K/m)", "viridis")
    plot_map(axes[1, 3], nodes, nodes, np.where(a["rho_full"] > 0.5, a["psi"], np.nan), "weighting potential ψ (high terminal=1)", "viridis")
    plot_map(axes[2, 0], cells, cells, a["current_total"], "total PTE current integrand (A/m²)", "coolwarm", True)
    plot_map(axes[2, 1], cells, cells, a["current_b"], "b-component current integrand (A/m²)", "coolwarm", True)
    plot_map(axes[2, 2], cells, cells, a["current_a"], "a-component current integrand (A/m²)", "coolwarm", True)
    axes[2, 3].axis("off")
    text = (
        f"Run {case.run:03d}: {case.interface}, {case.contact_label}, E||{case.polarization[-1]}\n"
        f"G = {case.conductance:.3e} W/(m² K)\n"
        f"exact 500 nm bad nodes = {metrics['exact_500nm_bad_nodes']}\n"
        f"P_Q(285 µW) = {metrics['P_Q_at_285uW_W']*1e6:.4f} µW\n"
        f"closure = {metrics['six_face_closure']*100:.4f}%\n"
        f"Tmax = {metrics['Tmax_at_285uW_K']:.5g} K\n"
        f"I_exact = {metrics['current_exact_at_285uW_A']*1e9:.5g} nA\n"
        f"I_cont = {metrics['current_continuous_at_285uW_A']*1e9:.5g} nA\n"
        f"ΔI = {metrics['current_change_from_continuous_fraction']*100:.3f}%\n"
        "No clipping/smoothing/gain/global rescaling.\n"
        "285 µW values use only the certified linear source-power scaling."
    )
    axes[2, 3].text(0.02, 0.98, text, va="top", family="monospace", fontsize=12)
    fig.suptitle(f"Final exact-binary physical fields — Run {case.run:03d}", fontsize=16)
    path = REPORT_DIR / f"{stem}_fields.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    outputs.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    mid_qx = int(np.argmin(np.abs(a["xc_um"])))
    mid_qy = int(np.argmin(np.abs(a["yc_um"])))
    mid = 120
    axes[0, 0].plot(a["xc_um"], a["qxy"][:, mid_qy], label="all material")
    axes[0, 0].plot(a["xc_um"], a["qxy_flake"][:, mid_qy], label="TaIrTe4 z-range")
    axes[0, 0].set(xlabel="x=b (µm), y=0", ylabel="depth-integrated Q (W/m²)")
    axes[0, 0].legend()
    axes[0, 1].plot(a["yc_um"], a["qxy"][mid_qx, :], label="all material")
    axes[0, 1].plot(a["yc_um"], a["qxy_flake"][mid_qx, :], label="TaIrTe4 z-range")
    axes[0, 1].set(xlabel="y=a (µm), x=0", ylabel="depth-integrated Q (W/m²)")
    axes[0, 1].legend()
    axes[1, 0].plot(nodes, a["temperature"][:, mid], label="ΔT")
    axes[1, 0].plot(nodes, a["gx"][:, mid] / 1e5, label="∂T/∂b / 1e5")
    axes[1, 0].set(xlabel="x=b (µm), y=0", ylabel="K or scaled K/m")
    axes[1, 0].legend()
    axes[1, 1].plot(cells, a["current_total"][:, 119], label="total")
    axes[1, 1].plot(cells, a["current_b"][:, 119], label="b component")
    axes[1, 1].plot(cells, a["current_a"][:, 119], label="a component")
    axes[1, 1].set(xlabel="x=b (µm), y≈0", ylabel="current integrand (A/m²)")
    axes[1, 1].legend()
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
    fig.suptitle(f"Center profiles — Run {case.run:03d}")
    path = REPORT_DIR / f"{stem}_profiles.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    outputs.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(a["zc_um"], a["qz_power"] * 1e6, marker="o", ms=3)
    axes[0].axvspan(-0.1, 0.0, color="tab:red", alpha=0.12, label="TaIrTe4")
    axes[0].set(xlabel="z=c (µm)", ylabel="absorbed layer power at 285 µW (µW)", xlim=(-1.0, 1.5))
    axes[0].legend(); axes[0].grid(alpha=0.25)
    labels = ["b term", "a term", "total", "positive", "negative"]
    vals = np.asarray([
        metrics["current_b_component_at_285uW_A"], metrics["current_a_component_at_285uW_A"],
        metrics["current_exact_at_285uW_A"], metrics["current_positive_at_285uW_A"],
        metrics["current_negative_at_285uW_A"],
    ]) * 1e9
    axes[1].bar(labels, vals, color=["tab:blue", "tab:orange", "black", "tab:red", "tab:cyan"])
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(ylabel="integrated terminal-current contribution (nA)")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(f"Depth absorption and current decomposition — Run {case.run:03d}")
    path = REPORT_DIR / f"{stem}_decomposition.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    outputs.append(path)
    return outputs


def summary_plots(metrics: list[dict[str, Any]], arrays: dict[int, dict[str, np.ndarray]]) -> list[Path]:
    labels = [f"R{m['run']:03d}\n{m['polarization']}" for m in metrics]
    x = np.arange(len(metrics))
    colors = ["tab:blue" if m["interface_scenario"] == "thermally_grown" else "tab:orange" for m in metrics]
    outputs: list[Path] = []
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), constrained_layout=True)
    for ax, m in zip(axes.ravel(), metrics):
        rho = arrays[int(m["run"])]["rho_full"]
        ax.imshow(rho.T, origin="lower", extent=(-12, 12, -12, 12), cmap="gray_r", vmin=0, vmax=1)
        ax.set_aspect("equal")
        ax.set_title(f"Run {m['run']:03d}: {m['electrodes']}, {m['interface_scenario']}, {m['polarization']}")
        ax.set_xlabel("x=b (µm)"); ax.set_ylabel("y=a (µm)")
    fig.suptitle("Final exact-binary structures (black=TaIrTe4; every exact-500-nm bad count = 0)", fontsize=16)
    path = REPORT_DIR / "final_exact_binary_structures.png"
    fig.savefig(path, dpi=180); plt.close(fig); outputs.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    exact = np.asarray([m["current_exact_at_285uW_A"] for m in metrics]) * 1e9
    cont = np.asarray([m["current_continuous_at_285uW_A"] for m in metrics]) * 1e9
    axes[0, 0].bar(x - 0.2, cont, width=0.4, label="continuous checkpoint", color="0.65")
    axes[0, 0].bar(x + 0.2, exact, width=0.4, label="forced exact binary", color=colors)
    axes[0, 0].set(ylabel="current at 285 µW (nA)", xticks=x, xticklabels=labels); axes[0, 0].legend()
    axes[0, 1].bar(x, 100 * np.asarray([m["current_change_from_continuous_fraction"] for m in metrics]), color=colors)
    axes[0, 1].axhline(-1, color="tab:red", ls="--", label="legacy 1% preservation gate")
    axes[0, 1].set(ylabel="exact-binary current change (%)", xticks=x, xticklabels=labels); axes[0, 1].legend()
    axes[1, 0].bar(x, np.asarray([m["P_Q_at_285uW_W"] for m in metrics]) * 1e6, color=colors)
    axes[1, 0].set(ylabel="absorbed P_Q at 285 µW (µW)", xticks=x, xticklabels=labels)
    axes[1, 1].bar(x, [m["Tmax_at_285uW_K"] for m in metrics], color=colors)
    axes[1, 1].set(ylabel="TaIrTe4 Tmax rise at 285 µW (K)", xticks=x, xticklabels=labels)
    for ax in axes.ravel(): ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Exact-binary eight-case physical metrics")
    path = REPORT_DIR / "final_exact_binary_physics_metrics.png"
    fig.savefig(path, dpi=180); plt.close(fig); outputs.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    components = np.asarray([[m["current_b_component_at_285uW_A"], m["current_a_component_at_285uW_A"]] for m in metrics]) * 1e9
    axes[0].bar(x - 0.2, components[:, 0], width=0.4, label="b/x thermoelectric term")
    axes[0].bar(x + 0.2, components[:, 1], width=0.4, label="a/y thermoelectric term")
    axes[0].axhline(0, color="black", lw=0.8); axes[0].set(ylabel="integrated contribution (nA)", xticks=x, xticklabels=labels); axes[0].legend()
    posneg = np.asarray([[m["current_positive_at_285uW_A"], m["current_negative_at_285uW_A"]] for m in metrics]) * 1e9
    axes[1].bar(x, posneg[:, 0], label="positive spatial contribution", color="tab:red")
    axes[1].bar(x, posneg[:, 1], label="negative spatial contribution", color="tab:cyan")
    axes[1].axhline(0, color="black", lw=0.8); axes[1].set(ylabel="integrated contribution (nA)", xticks=x, xticklabels=labels); axes[1].legend()
    for ax in axes: ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Terminal-current decomposition of exact-binary structures")
    path = REPORT_DIR / "final_exact_binary_current_components.png"
    fig.savefig(path, dpi=180); plt.close(fig); outputs.append(path)

    pairs = [(44, 45), (46, 48), (55, 56), (57, 58)]
    by_run = {int(m["run"]): m for m in metrics}
    pair_labels = ["grown\ntop-bottom", "grown\nleft-right", "evaporated\ntop-bottom", "evaporated\nleft-right"]
    ea = np.asarray([by_run[a]["current_exact_at_285uW_A"] for a, _ in pairs]) * 1e9
    eb = np.asarray([by_run[b]["current_exact_at_285uW_A"] for _, b in pairs]) * 1e9
    px = np.arange(len(pairs))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    axes[0].bar(px - 0.2, ea, width=0.4, label="E||a optimized run")
    axes[0].bar(px + 0.2, eb, width=0.4, label="E||b optimized run")
    axes[0].set(xticks=px, xticklabels=pair_labels, ylabel="exact-binary current at 285 µW (nA)")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.25)
    ratio = eb / ea
    axes[1].bar(px, ratio, color=["tab:blue", "tab:blue", "tab:orange", "tab:orange"])
    axes[1].axhline(1.0, color="black", ls="--")
    axes[1].set(xticks=px, xticklabels=pair_labels, ylabel="I(E||b optimized) / I(E||a optimized)")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Paired exact-binary outcomes (independently optimized structures; not cross-polarization selectivity)")
    path = REPORT_DIR / "final_exact_binary_pair_comparisons.png"
    fig.savefig(path, dpi=180); plt.close(fig); outputs.append(path)
    return outputs


def report_text(metrics: list[dict[str, Any]]) -> str:
    rows = []
    for m in metrics:
        rows.append(
            f"| {m['run']:03d} | {m['electrodes']} | {m['interface_scenario']} | {m['polarization']} | "
            f"{m['exact_500nm_bad_nodes']} | {m['P_Q_at_285uW_W']*1e6:.4f} | {m['Tmax_at_285uW_K']:.5g} | "
            f"{m['current_continuous_at_285uW_A']*1e9:.4f} | {m['current_exact_at_285uW_A']*1e9:.4f} | "
            f"{m['current_change_from_continuous_fraction']*100:.3f}% | {m['physical_gates_passed']} |"
        )
    failed_objective = [f"Run {m['run']:03d}" for m in metrics if not m["one_percent_objective_preservation_passed"]]
    return f"""# Final exact-binary eight-case PTE matrix

Status: **COMPLETED_EXACT_BINARY_EIGHT_CASE_PHYSICS_MATRIX_WITH_OBJECTIVE_PRESERVATION_DIAGNOSTICS**

## Outcome

Runs 044, 045, 046, 048, 055, 056, 057, and 058 form the complete 2×2×2 matrix: top/bottom versus left/right electrodes, thermally-grown versus evaporated TaIrTe4/SiO2 interface, and `Ea` versus `Eb` illumination. Every promoted structure is exactly 0/1 and has **zero** bad nodes in the requested discrete 500 nm solid-and-void opening audit.

The table reports the fresh physical evaluation of the forced exact structure, even when its current is lower than the continuous checkpoint. The old 1% objective-preservation gate is not rewritten: it fails for {', '.join(failed_objective)}. This does not invalidate their Maxwell/thermal/electrical solution; it records the performance cost of enforcing the final geometry.

| Run | electrodes | interface | pol. | 500 nm bad | P_Q @285 µW (µW) | Tmax rise (K) | continuous I (nA) | exact I (nA) | change | physical gates |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

## What was calculated

- Optical: each exact candidate has its own fresh v261 GPU Maxwell forward solution. `P_Q`, six-face power, closure, and the conservative volumetric `Q(x,y,z)` mapped to the explicit 3-D thermal grid are retained.
- Thermal: the stored CUDA solution uses explicit air, 285 nm SiO2, and 20 µm Si, anisotropic TaIrTe4 `k=(3.8,14.4,1.0) W/(m K)` in Lumerical `(x=b,y=a,z=c)`, finite TaIrTe4/SiO2 G, and SiO2/Si `G=1.1e9 W/(m² K)`.
- Electrical: terminal current is the full triangular-FEM integral over the 100 nm TaIrTe4 sheet, not a single point and not a strict-gradient proxy. The plotted strict-centered gradients are diagnostic maps only; a node is NaN unless all `±x` and `±y` solid neighbours exist.
- Current decomposition: `b/x`, `a/y`, positive spatial, and negative spatial contributions are independently reintegrated from the stored temperature and weighting fields. Their total agrees with the certified terminal current to roundoff.

## Interface provenance

Runs 055–058 embed the evaporated interface contract (`G=7.37e4 W/(m² K)`) directly in their raw JSON. Runs 044–048 are legacy artifacts created before this metadata field was added: their thermally-grown scenario (`G=7.37e6 W/(m² K)`) follows the then-default `thermal.py` execution contract and launch environment, but is **not** represented as newly embedded raw metadata. The legacy raw JSON files are unchanged.

## Scaling and integrity

Raw FDTD excitation powers differ slightly by exact geometry. Values labelled “at 285 µW” apply the linear factor `285e-6/source_power_W`, exactly as in the original objective certificates. This is physical linear-response normalization, not an empirical fit. No Q clipping, smoothing, gain, global rescaling, tiling, or source deletion is used. Raw NPZ/FSP/H5 files remain outside Git; their paths, sizes, and SHA-256 values are in `RAW_ARTIFACT_MANIFEST.json`.

## Figure guide

- `final_exact_binary_structures.png`: all eight exact structures on common Lumerical axes.
- `final_exact_binary_physics_metrics.png`: exact versus continuous current, performance cost, absorbed power, and peak temperature.
- `final_exact_binary_current_components.png`: axis and signed-spatial current decomposition.
- `final_exact_binary_pair_comparisons.png`: paired Ea/Eb optimized-run outcomes. Because each bar uses a different optimized structure, this is not a same-device polarization-selectivity measurement.
- `run*_fields.png`: per-case exact density, Q, temperature, strict gradients, weighting potential, and total/component current maps.
- `run*_profiles.png`: central Q, temperature/gradient, and current profiles.
- `run*_decomposition.png`: absorption versus depth and integrated current terms.

This report does not claim that the forced repair is the optimum of a new binary combinatorial optimization. It is the requested physically re-evaluated, manufacturability-clean final structure for each completed run.
"""


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for path in REPORT_DIR.glob("*.png"):
        path.unlink()
    metrics: list[dict[str, Any]] = []
    arrays: dict[int, dict[str, np.ndarray]] = {}
    manifest_inputs: list[dict[str, Any]] = []
    generated_plots: list[Path] = []
    for case in CASES:
        m, a, artifacts = process(case)
        metrics.append(m); arrays[case.run] = a; manifest_inputs.extend(artifacts)
        generated_plots.extend(per_case_plots(case, m, a))
    generated_plots.extend(summary_plots(metrics, arrays))

    status = "COMPLETED_EXACT_BINARY_EIGHT_CASE_PHYSICS_MATRIX_WITH_OBJECTIVE_PRESERVATION_DIAGNOSTICS"
    by_run = {int(m["run"]): m for m in metrics}
    pair_specs = (
        ("thermally_grown_top_bottom", 44, 45),
        ("thermally_grown_left_right", 46, 48),
        ("evaporated_top_bottom", 55, 56),
        ("evaporated_left_right", 57, 58),
    )
    pair_comparisons = [
        {
            "pair": label,
            "Ea_optimized_run": ea_run,
            "Eb_optimized_run": eb_run,
            "Ea_optimized_exact_current_at_285uW_A": by_run[ea_run]["current_exact_at_285uW_A"],
            "Eb_optimized_exact_current_at_285uW_A": by_run[eb_run]["current_exact_at_285uW_A"],
            "Eb_over_Ea_optimized_run_current_ratio": by_run[eb_run]["current_exact_at_285uW_A"] / by_run[ea_run]["current_exact_at_285uW_A"],
            "interpretation": "different independently optimized structures; not same-device polarization selectivity",
        }
        for label, ea_run, eb_run in pair_specs
    ]
    summary = {
        "schema": "final-exact-binary-eight-case-matrix-v1",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "axis_contract": "Lumerical x=b, y=a, z=c",
        "target_incident_power_W": TARGET_POWER_W,
        "matrix_complete": True,
        "run_ids": [c.run for c in CASES],
        "all_exact_binary": all(m["exact_binary"] for m in metrics),
        "all_exact_500nm_bad_nodes_zero": all(m["exact_500nm_bad_nodes"] == 0 for m in metrics),
        "all_physical_gates_passed": all(m["physical_gates_passed"] for m in metrics),
        "all_one_percent_objective_preservation_passed": all(m["one_percent_objective_preservation_passed"] for m in metrics),
        "one_percent_objective_preservation_failures": [m["run"] for m in metrics if not m["one_percent_objective_preservation_passed"]],
        "pair_comparisons": pair_comparisons,
        "cases": metrics,
    }
    summary_path = REPORT_DIR / "final_exact_binary_eight_case_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    csv_path = REPORT_DIR / "final_exact_binary_eight_case_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)

    report_path = REPORT_DIR / "FINAL_EXACT_BINARY_EIGHT_CASE_REPORT.md"
    report_path.write_text(report_text(metrics))
    manifest_path = REPORT_DIR / "RAW_ARTIFACT_MANIFEST.json"
    output_paths = [report_path, summary_path, csv_path, *generated_plots]
    manifest = {
        "schema": "final-exact-binary-eight-case-artifact-manifest-v1",
        "status": status,
        "generated_at_utc": summary["generated_at_utc"],
        "generation_command": "/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python photothermal_pte/optimization_runs/summarize_final_exact_binary_eight_case.py",
        "raw_artifacts_not_committed": manifest_inputs,
        "published_outputs": [artifact(p, "published report artifact") for p in output_paths],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "status": status,
        "report_dir": str(REPORT_DIR),
        "runs": [m["run"] for m in metrics],
        "currents_nA": {str(m["run"]): m["current_exact_at_285uW_A"] * 1e9 for m in metrics},
        "plots": len(generated_plots),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
