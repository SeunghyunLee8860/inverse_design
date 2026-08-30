#!/usr/bin/env python3
"""GPU fixed-Q AD--FD control for an Au/air layer on fixed TaIrTe4.

This is deliberately a thermal-material/operator checkpoint.  It does not
reuse the optical total-Q gradient and it is not a production PTE result.  A
20 x 20 field of 500-nm physical densities describes a 50-nm Au/air layer on
top of a fixed 100-nm TaIrTe4 sheet.  The TaIrTe4 bottom uses the paper-reduced
thermally-grown-SiO2 Robin boundary, while the top design layer is exposed to
ambient air.

The Au/TaIrTe4 conductance has not been measured for this device.  Therefore
the script runs named numerical scenarios and never promotes one as the
experimental truth.  The literature analogue is the *calculated* Au/MoS2
room-temperature resistance 5.8e-8 m2 K/W reported by Mao et al.; it is not a
TaIrTe4 measurement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import ndimage, sparse

from photothermal_pte.optimization_runs.au_on_fixed_tairte4_validation.material_model import (
    AU_BULK_THERMAL_CONDUCTIVITY_W_MK,
)
from photothermal_pte.optimization_runs.cuda_thermal_adjoint import PersistentCudaCSR


NX = 20
NY = 20
DX_M = 500.0e-9
DY_M = 500.0e-9
TA_DZ_M = 100.0e-9
AU_DZ_M = 50.0e-9
K_AIR_W_MK = 0.026
# Lumerical x=b, y=a, z=c.
K_TA_XYZ_W_MK = np.asarray((3.8, 14.4, 1.0), dtype=np.float64)
K_AU_W_MK = float(AU_BULK_THERMAL_CONDUCTIVITY_W_MK)
G_TA_AIR_W_M2K = 1.0
G_TA_SIO2_W_M2K = 7.37e6
H_TOP_AIR_W_M2K = 10.0
GRAY_EXPONENT = 1.0
FIXED_Q_POWER_W = 1.0e-6
G_SCENARIOS_W_M2K = {
    "low_numerical_scenario": 1.0e6,
    "Au_MoS2_theory_analogue_not_TaIrTe4_data": 1.0 / 5.8e-8,
    "high_numerical_scenario": 1.0e8,
    "perfect_contact_limit": math.inf,
}


def relative(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), np.finfo(float).tiny)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _id(layer: int, i: int, j: int) -> int:
    return layer * NX * NY + i * NY + j


@dataclass(frozen=True)
class EdgeDerivative:
    left: int
    right: int | None
    rho_index: int
    dg_drho_W_K: float
    label: str


@dataclass(frozen=True)
class ThermalSystem:
    matrix_W_K: sparse.csr_matrix
    rhs_W: np.ndarray
    objective_K_inv: np.ndarray
    derivative_terms: tuple[EdgeDerivative, ...]
    rho: np.ndarray
    g_au_ta_W_m2K: float
    source_power_W: float
    bottom_conductance_W_K: np.ndarray
    top_conductance_W_K: np.ndarray


def _add_edge(
    row: list[int], col: list[int], data: list[float], left: int, right: int, g: float
) -> None:
    row.extend((left, right, left, right))
    col.extend((left, right, right, left))
    data.extend((g, g, -g, -g))


def _add_boundary(
    row: list[int], col: list[int], data: list[float], cell: int, g: float
) -> None:
    row.append(cell)
    col.append(cell)
    data.append(g)


def base_density() -> np.ndarray:
    x = np.linspace(-1.0, 1.0, NX)[:, None]
    y = np.linspace(-1.0, 1.0, NY)[None, :]
    return 0.52 + 0.07 * np.cos(0.8 * np.pi * x) * np.cos(0.65 * np.pi * y) + 0.02 * x


def build_system(rho: np.ndarray, g_au_ta_W_m2K: float) -> ThermalSystem:
    density = np.asarray(rho, dtype=np.float64)
    if density.shape != (NX, NY):
        raise ValueError(f"rho shape {density.shape} != {(NX, NY)}")
    if np.any((density <= 0.0) | (density >= 1.0)):
        raise ValueError("thermal AD-FD control requires unclipped 0<rho<1")
    if not (g_au_ta_W_m2K > 0.0 or math.isinf(g_au_ta_W_m2K)):
        raise ValueError("Au/TaIrTe4 conductance must be positive")

    phi = density**GRAY_EXPONENT
    dphi = GRAY_EXPONENT * density ** (GRAY_EXPONENT - 1.0)
    k_top = K_AIR_W_MK + phi * (K_AU_W_MK - K_AIR_W_MK)
    dk_drho = dphi * (K_AU_W_MK - K_AIR_W_MK)
    count = 2 * NX * NY
    row: list[int] = []
    col: list[int] = []
    data: list[float] = []
    derivative_terms: list[EdgeDerivative] = []

    # Fixed anisotropic TaIrTe4 lateral conduction.
    for i in range(NX):
        for j in range(NY):
            left = _id(0, i, j)
            if i + 1 < NX:
                _add_edge(
                    row, col, data, left, _id(0, i + 1, j),
                    K_TA_XYZ_W_MK[0] * DY_M * TA_DZ_M / DX_M,
                )
            if j + 1 < NY:
                _add_edge(
                    row, col, data, left, _id(0, i, j + 1),
                    K_TA_XYZ_W_MK[1] * DX_M * TA_DZ_M / DY_M,
                )

    # Density-dependent Au/air-layer lateral conduction.  Each half-cell
    # resistance is differentiated separately, so both neighboring pixels
    # receive their exact contribution.
    for i in range(NX):
        for j in range(NY):
            left = _id(1, i, j)
            if i + 1 < NX:
                right = _id(1, i + 1, j)
                resistance = 0.5 * DX_M / k_top[i, j] + 0.5 * DX_M / k_top[i + 1, j]
                area = DY_M * AU_DZ_M
                g = area / resistance
                _add_edge(row, col, data, left, right, g)
                for ii, jj in ((i, j), (i + 1, j)):
                    dg = (
                        area / resistance**2 * 0.5 * DX_M / k_top[ii, jj] ** 2
                        * dk_drho[ii, jj]
                    )
                    derivative_terms.append(
                        EdgeDerivative(left, right, ii * NY + jj, dg, "Au_layer_x")
                    )
            if j + 1 < NY:
                right = _id(1, i, j + 1)
                resistance = 0.5 * DY_M / k_top[i, j] + 0.5 * DY_M / k_top[i, j + 1]
                area = DX_M * AU_DZ_M
                g = area / resistance
                _add_edge(row, col, data, left, right, g)
                for ii, jj in ((i, j), (i, j + 1)):
                    dg = (
                        area / resistance**2 * 0.5 * DY_M / k_top[ii, jj] ** 2
                        * dk_drho[ii, jj]
                    )
                    derivative_terms.append(
                        EdgeDerivative(left, right, ii * NY + jj, dg, "Au_layer_y")
                    )

    area_xy = DX_M * DY_M
    half_ta = 0.5 * TA_DZ_M / K_TA_XYZ_W_MK[2]
    r_air = half_ta + 1.0 / G_TA_AIR_W_M2K + 0.5 * AU_DZ_M / K_AIR_W_MK
    r_au = half_ta + (0.0 if math.isinf(g_au_ta_W_m2K) else 1.0 / g_au_ta_W_m2K) + 0.5 * AU_DZ_M / K_AU_W_MK
    bottom_g = np.full((NX, NY), area_xy / (half_ta + 1.0 / G_TA_SIO2_W_M2K))
    top_g = np.zeros((NX, NY), dtype=np.float64)

    for i in range(NX):
        for j in range(NY):
            rho_index = i * NY + j
            ta = _id(0, i, j)
            top = _id(1, i, j)
            # Physical-density relaxation: rho is the parallel contact-area
            # fraction, not an effective-series resistance.
            g_vertical = area_xy * ((1.0 - phi[i, j]) / r_air + phi[i, j] / r_au)
            dg_vertical = area_xy * dphi[i, j] * (1.0 / r_au - 1.0 / r_air)
            _add_edge(row, col, data, ta, top, g_vertical)
            derivative_terms.append(
                EdgeDerivative(ta, top, rho_index, dg_vertical, "Ta_to_Au_or_air_parallel_area")
            )

            # The upper surface exchanges heat with ambient.  The explicit
            # half-cell resistance retains the derivative of k_top.
            top_resistance = 0.5 * AU_DZ_M / k_top[i, j] + 1.0 / H_TOP_AIR_W_M2K
            top_g[i, j] = area_xy / top_resistance
            dg_top = (
                area_xy / top_resistance**2 * 0.5 * AU_DZ_M / k_top[i, j] ** 2
                * dk_drho[i, j]
            )
            _add_boundary(row, col, data, top, top_g[i, j])
            derivative_terms.append(
                EdgeDerivative(top, None, rho_index, dg_top, "top_ambient")
            )
            _add_boundary(row, col, data, ta, bottom_g[i, j])

    matrix = sparse.coo_matrix((data, (row, col)), shape=(count, count)).tocsr()
    matrix.sum_duplicates()
    if np.max(np.abs((matrix - matrix.T).data), initial=0.0) > 1.0e-15:
        raise RuntimeError("thermal matrix is not symmetric")

    x = (np.arange(NX) + 0.5 - NX / 2.0) * DX_M
    y = (np.arange(NY) + 0.5 - NY / 2.0) * DY_M
    xx, yy = np.meshgrid(x, y, indexing="ij")
    source_shape = np.exp(-2.0 * ((xx + 0.65e-6) ** 2 + (yy - 0.45e-6) ** 2) / (2.6e-6) ** 2)
    source_power = FIXED_Q_POWER_W * source_shape / np.sum(source_shape)
    rhs = np.zeros(count, dtype=np.float64)
    rhs[: NX * NY] = source_power.reshape(-1)

    # A smooth signed TaIrTe4 temperature functional supplies a non-null,
    # PTE-like thermal objective without introducing the electrical operator.
    weight = (xx / (4.0e-6)) * np.exp(-((xx / 4.0e-6) ** 2 + (yy / 4.0e-6) ** 2))
    weight -= np.mean(weight)
    weight /= np.sum(np.abs(weight))
    objective = np.zeros(count, dtype=np.float64)
    objective[: NX * NY] = weight.reshape(-1)
    return ThermalSystem(
        matrix_W_K=matrix,
        rhs_W=rhs,
        objective_K_inv=objective,
        derivative_terms=tuple(derivative_terms),
        rho=density.copy(),
        g_au_ta_W_m2K=float(g_au_ta_W_m2K),
        source_power_W=float(np.sum(rhs)),
        bottom_conductance_W_K=bottom_g,
        top_conductance_W_K=top_g,
    )


def solve_gpu(
    system: ThermalSystem, cuda_device: int, *, need_adjoint: bool = True
) -> tuple[np.ndarray, np.ndarray | None, dict[str, float | int | bool]]:
    operator = PersistentCudaCSR(system.matrix_W_K, cuda_device=cuda_device)
    start = perf_counter()
    forward = operator.solve(
        system.rhs_W,
        relative_tolerance=1.0e-11,
        max_iterations=20000,
        residual_check_interval=10,
    )
    adjoint = None
    if need_adjoint:
        adjoint = operator.solve(
            system.objective_K_inv,
            relative_tolerance=1.0e-11,
            max_iterations=20000,
            residual_check_interval=10,
        )
    seconds = perf_counter() - start
    return forward.solution, (adjoint.solution if adjoint is not None else None), {
        "forward_relative_residual": float(forward.explicit_relative_residual),
        "adjoint_relative_residual": (
            float(adjoint.explicit_relative_residual) if adjoint is not None else 0.0
        ),
        "forward_iterations": int(forward.iterations),
        "adjoint_iterations": int(adjoint.iterations) if adjoint is not None else 0,
        "two_solve_wall_s": float(seconds),
        "CPU_linear_solve_fallback": False,
    }


def objective(system: ThermalSystem, temperature: np.ndarray) -> float:
    return float(system.objective_K_inv @ temperature)


def thermal_gradient(
    system: ThermalSystem, temperature: np.ndarray, adjoint: np.ndarray
) -> np.ndarray:
    gradient = np.zeros(NX * NY, dtype=np.float64)
    for term in system.derivative_terms:
        if term.right is None:
            contribution = -term.dg_drho_W_K * adjoint[term.left] * temperature[term.left]
        else:
            contribution = -term.dg_drho_W_K * (
                adjoint[term.left] - adjoint[term.right]
            ) * (
                temperature[term.left] - temperature[term.right]
            )
        gradient[term.rho_index] += contribution
    return gradient.reshape(NX, NY)


def energy_balance(system: ThermalSystem, temperature: np.ndarray) -> dict[str, float]:
    ta = temperature[: NX * NY].reshape(NX, NY)
    top = temperature[NX * NY :].reshape(NX, NY)
    bottom_power = float(np.sum(system.bottom_conductance_W_K * ta))
    top_power = float(np.sum(system.top_conductance_W_K * top))
    error = abs(bottom_power + top_power - system.source_power_W) / system.source_power_W
    return {
        "source_power_W": system.source_power_W,
        "bottom_power_W": bottom_power,
        "top_power_W": top_power,
        "relative_error": float(error),
    }


def directions(gradient: np.ndarray) -> dict[str, np.ndarray]:
    x = np.linspace(-1.0, 1.0, NX)[:, None]
    y = np.linspace(-1.0, 1.0, NY)[None, :]
    rng = np.random.default_rng(20260821)
    raw = {
        "uniform": np.ones((NX, NY)),
        "smooth_asymmetric": np.sin(0.7 * np.pi * x) * np.cos(0.55 * np.pi * y) + 0.21 * x,
        "central_localized": np.exp(-((x - 0.1) ** 2 + (y + 0.08) ** 2) / 0.07),
        "design_edge_localized": np.exp(-((x - 0.88) ** 2 + (y - 0.25) ** 2) / 0.025),
        "fixed_seed_random": ndimage.gaussian_filter(rng.normal(size=(NX, NY)), sigma=1.5),
        "adjoint_aligned": gradient.copy(),
    }
    result: dict[str, np.ndarray] = {}
    for name, value in raw.items():
        norm = np.linalg.norm(value)
        if norm == 0.0:
            raise RuntimeError(f"zero direction: {name}")
        result[name] = value / norm
    return result


def interface_series_control(g_value: float) -> dict[str, float | str]:
    q_W_m2 = 2.0e5
    interface_jump = 0.0 if math.isinf(g_value) else q_W_m2 / g_value
    total_delta = q_W_m2 * (
        TA_DZ_M / K_TA_XYZ_W_MK[2]
        + (0.0 if math.isinf(g_value) else 1.0 / g_value)
        + AU_DZ_M / K_AU_W_MK
    )
    reconstructed_jump = total_delta - q_W_m2 * (
        TA_DZ_M / K_TA_XYZ_W_MK[2] + AU_DZ_M / K_AU_W_MK
    )
    absolute_error = abs(reconstructed_jump - interface_jump)
    relative_error = absolute_error / max(abs(interface_jump), np.finfo(float).tiny)
    return {
        "q_W_m2": q_W_m2,
        "analytic_interface_jump_K": interface_jump,
        "reconstructed_interface_jump_K": reconstructed_jump,
        "absolute_jump_error_K": absolute_error,
        "relative_jump_error": relative_error if not math.isinf(g_value) else 0.0,
        "perfect_contact": bool(math.isinf(g_value)),
        "note": "independent 1D series-resistance algebra control",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--cuda-device", required=True, type=int)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to exactly one physical GPU")
    output = args.output_dir.expanduser().resolve()
    raw_dir = args.raw_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    rho0 = base_density()
    steps = (0.01, 0.005, 0.0025)
    all_rows: list[dict[str, object]] = []
    scenario_results: dict[str, object] = {}
    worst_residual = 0.0
    worst_energy = 0.0
    worst_fine_error = 0.0
    raw_arrays: dict[str, np.ndarray] = {"rho": rho0}

    for scenario_name, g_value in G_SCENARIOS_W_M2K.items():
        system = build_system(rho0, g_value)
        temperature, adjoint, solver = solve_gpu(system, args.cuda_device)
        if adjoint is None:
            raise RuntimeError("base thermal adjoint was not solved")
        gradient = thermal_gradient(system, temperature, adjoint)
        base_objective = objective(system, temperature)
        energy = energy_balance(system, temperature)
        scenario_rows: list[dict[str, object]] = []
        for direction_name, direction in directions(gradient).items():
            ad = float(np.sum(gradient * direction))
            for step in steps:
                values: list[float] = []
                residuals: list[float] = []
                energies: list[float] = []
                for sign in (1.0, -1.0):
                    local_rho = rho0 + sign * step * direction
                    if np.any((local_rho <= 0.0) | (local_rho >= 1.0)):
                        raise RuntimeError("FD perturbation clipped")
                    local_system = build_system(local_rho, g_value)
                    local_temperature, _, local_solver = solve_gpu(
                        local_system, args.cuda_device, need_adjoint=False
                    )
                    values.append(objective(local_system, local_temperature))
                    residuals.extend(
                        (local_solver["forward_relative_residual"], local_solver["adjoint_relative_residual"])
                    )
                    energies.append(energy_balance(local_system, local_temperature)["relative_error"])
                fd = (values[0] - values[1]) / (2.0 * step)
                row = {
                    "scenario": scenario_name,
                    "G_Au_Ta_W_m2K": "inf" if math.isinf(g_value) else g_value,
                    "direction": direction_name,
                    "step": step,
                    "AD_K_per_rho": ad,
                    "FD_K_per_rho": fd,
                    "relative_error": relative(ad, fd),
                    "plus_objective_K": values[0],
                    "minus_objective_K": values[1],
                    "worst_residual": max(residuals),
                    "worst_energy_balance": max(energies),
                    "clipping": False,
                }
                scenario_rows.append(row)
                all_rows.append(row)
        fine = [row for row in scenario_rows if row["step"] == min(steps)]
        scenario_worst_fine = max(float(row["relative_error"]) for row in fine)
        worst_fine_error = max(worst_fine_error, scenario_worst_fine)
        worst_residual = max(
            worst_residual,
            float(solver["forward_relative_residual"]),
            float(solver["adjoint_relative_residual"]),
            max(float(row["worst_residual"]) for row in scenario_rows),
        )
        worst_energy = max(
            worst_energy,
            float(energy["relative_error"]),
            max(float(row["worst_energy_balance"]) for row in scenario_rows),
        )
        ta = temperature[: NX * NY].reshape(NX, NY)
        top = temperature[NX * NY :].reshape(NX, NY)
        scenario_results[scenario_name] = {
            "G_Au_Ta_W_m2K": "inf" if math.isinf(g_value) else g_value,
            "provenance": (
                "perfect contact numerical limit" if math.isinf(g_value)
                else "Au/MoS2 theoretical analogue; not TaIrTe4 data"
                if "analogue" in scenario_name else "numerical sensitivity scenario; not confidence interval"
            ),
            "base_objective_K": base_objective,
            "Tmax_Ta_K": float(np.max(ta)),
            "Tmax_top_design_layer_K": float(np.max(top)),
            "temperature_jump_cell_center_mean_K": float(np.mean(ta - top)),
            "gradient_l2_K_per_rho": float(np.linalg.norm(gradient)),
            "solver": solver,
            "energy_balance": energy,
            "worst_fine_step_ADFD_relative_error": scenario_worst_fine,
            "interface_series_control": interface_series_control(g_value),
        }
        raw_arrays[f"T_Ta_{scenario_name}"] = ta
        raw_arrays[f"T_top_{scenario_name}"] = top
        raw_arrays[f"gradient_{scenario_name}"] = gradient

    passed = bool(worst_fine_error < 0.01 and worst_residual < 1.0e-8 and worst_energy < 0.01)
    status = "VALIDATED_AU_THERMAL_MATERIAL_INTERFACE_CONTROL" if passed else "FAILED_AU_THERMAL_MATERIAL_INTERFACE_CONTROL"
    summary = {
        "status": status,
        "scope": "fixed-Q thermal material/interface operator only; no Maxwell, PTE, electrical, or optimization",
        "geometry": {
            "design_cells_xy": [NX, NY],
            "design_pixel_m": [DX_M, DY_M],
            "design_span_m": [NX * DX_M, NY * DY_M],
            "TaIrTe4_thickness_m": TA_DZ_M,
            "Au_or_air_layer_thickness_m": AU_DZ_M,
            "lateral_boundary": "adiabatic control",
            "bottom_boundary": "paper-reduced thermally-grown-SiO2 Robin to bath",
            "top_boundary": "ambient Robin",
        },
        "materials": {
            "k_TaIrTe4_xyz_W_mK": K_TA_XYZ_W_MK.tolist(),
            "k_Au_bulk_reference_W_mK": K_AU_W_MK,
            "k_air_W_mK": K_AIR_W_MK,
            "G_TaIrTe4_air_W_m2K": G_TA_AIR_W_M2K,
            "G_TaIrTe4_thermally_grown_SiO2_W_m2K": G_TA_SIO2_W_M2K,
            "h_top_air_W_m2K": H_TOP_AIR_W_M2K,
            "Au_bulk_note": "reference scenario, not a certified 50-nm-film value",
        },
        "gray_law": {
            "interpretation": "rho is parallel Au-contact area fraction",
            "phi": "rho^1",
            "vertical_face_conductance": "A*((1-phi)/R_Ta-air + phi/R_Ta-Au)",
            "lateral_top_layer": "harmonic half-cell resistance using k_air+phi*(k_Au-k_air)",
            "clipping_or_rescaling": False,
        },
        "fixed_Q": {
            "power_W": FIXED_Q_POWER_W,
            "location": "TaIrTe4 only",
            "optical_gradient_included": False,
        },
        "scenarios": scenario_results,
        "gates": {
            "worst_h0p0025_ADFD_relative_error": worst_fine_error,
            "required_ADFD_relative_error": 0.01,
            "worst_linear_residual": worst_residual,
            "required_linear_residual": 1.0e-8,
            "worst_energy_balance": worst_energy,
            "required_energy_balance": 0.01,
            "CPU_linear_solve_fallback": False,
        },
        "limitations": [
            "No direct Au/TaIrTe4 thermal-boundary-conductance measurement was identified.",
            "The Au/MoS2 value is a theoretical analogue, not a TaIrTe4 parameter.",
            "The current validated optical checkpoint omits SiO2/Si; coupled production remains blocked.",
            "Electrical shunting, weighting-field changes, PTE, and Au thermopower are not evaluated here.",
        ],
    }

    raw_path = raw_dir / "au_thermal_material_interface_adfd_raw.npz"
    np.savez_compressed(raw_path, **raw_arrays)
    csv_path = output / "au_thermal_material_interface_adfd_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    summary_path = output / "au_thermal_material_interface_adfd_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "raw_artifact_committed_to_git": False,
        "raw_artifact": {
            "path": str(raw_path),
            "bytes": raw_path.stat().st_size,
            "sha256": sha256(raw_path),
            "generation_command": "CUDA_VISIBLE_DEVICES=<one GPU> python 52_validate_au_thermal_material_interface_adfd.py ...",
        },
        "published": [
            {"path": str(summary_path), "bytes": summary_path.stat().st_size, "sha256": sha256(summary_path)},
            {"path": str(csv_path), "bytes": csv_path.stat().st_size, "sha256": sha256(csv_path)},
        ],
    }
    (output / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
