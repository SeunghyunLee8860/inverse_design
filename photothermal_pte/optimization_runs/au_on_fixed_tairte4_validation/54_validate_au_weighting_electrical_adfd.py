#!/usr/bin/env python3
"""GPU electrical/weighting AD--FD control for floating Au on fixed TaIrTe4.

The fixed TaIrTe4 sheet carries the high/low measurement terminals.  The Au
design is a floating nanostructure, not an electrode.  It can nevertheless
alter the TaIrTe4 weighting field through lateral Au conduction and finite
vertical Au/TaIrTe4 electrical contact.  Au Seebeck is set to zero in this
control so that only shunting/current-crowding and weighting-field changes are
tested.
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

import numpy as np
from scipy import ndimage, sparse

from photothermal_pte.optimization_runs.au_on_fixed_tairte4_validation.material_model import (
    AU_BULK_ELECTRICAL_CONDUCTIVITY_S_M,
)
from photothermal_pte.optimization_runs.cuda_thermal_adjoint import PersistentCudaCSR


N_TA = 40
N_DESIGN = 20
DESIGN_OFFSET = (N_TA - N_DESIGN) // 2
STEP_M = 500.0e-9
TA_THICKNESS_M = 100.0e-9
AU_THICKNESS_M = 50.0e-9
# Lumerical x=b, y=a.
SIGMA_TA_XY_S_M = np.asarray((1.10e5, 4.91e5), dtype=np.float64)
SEEBECK_TA_XY_V_K = np.asarray((27.0e-6, -6.0e-6), dtype=np.float64)
SIGMA_AU_S_M = float(AU_BULK_ELECTRICAL_CONDUCTIVITY_S_M)
SIGMA_FLOOR_FRACTION = 1.0e-8
CONTACT_FLOOR_FRACTION = 1.0e-10
CONTACT_SCENARIOS_S_M2 = {
    "rho_c_1e-8_numerical_scenario": 1.0e8,
    "rho_c_1e-10_numerical_scenario": 1.0e10,
    "rho_c_1e-12_numerical_scenario": 1.0e12,
    "near_ideal_rho_c_1e-14_numerical_limit": 1.0e14,
}


def relative(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), np.finfo(float).tiny)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ta_id(i: int, j: int) -> int:
    return i * N_TA + j


def au_id(i: int, j: int) -> int:
    return N_TA * N_TA + i * N_DESIGN + j


@dataclass(frozen=True)
class EdgeDerivative:
    left: int
    right: int
    rho_index: int
    dg_drho_S: float
    label: str


@dataclass(frozen=True)
class ElectricalSystem:
    full_matrix_S: sparse.csr_matrix
    reduced_matrix_S: sparse.csr_matrix
    reduced_rhs_A: np.ndarray
    free: np.ndarray
    fixed: np.ndarray
    fixed_values_V: np.ndarray
    objective_gradient_psi_A: np.ndarray
    derivative_terms: tuple[EdgeDerivative, ...]
    temperature_K: np.ndarray
    rho: np.ndarray
    contact_conductance_S_m2: float


def base_density() -> np.ndarray:
    x = np.linspace(-1.0, 1.0, N_DESIGN)[:, None]
    y = np.linspace(-1.0, 1.0, N_DESIGN)[None, :]
    return 0.52 + 0.07 * np.cos(0.75 * np.pi * x) * np.cos(0.6 * np.pi * y) + 0.018 * x


def _add_edge(
    row: list[int], col: list[int], data: list[float], left: int, right: int, g: float
) -> None:
    row.extend((left, right, left, right))
    col.extend((left, right, right, left))
    data.extend((g, g, -g, -g))


def build_system(rho: np.ndarray, contact_S_m2: float) -> ElectricalSystem:
    density = np.asarray(rho, dtype=np.float64)
    if density.shape != (N_DESIGN, N_DESIGN):
        raise ValueError("invalid Au density shape")
    if np.any((density <= 0.0) | (density >= 1.0)):
        raise ValueError("electrical AD-FD control requires unclipped density")
    if contact_S_m2 <= 0.0:
        raise ValueError("contact conductance must be positive")

    node_count = N_TA * N_TA + N_DESIGN * N_DESIGN
    row: list[int] = []
    col: list[int] = []
    data: list[float] = []
    derivative_terms: list[EdgeDerivative] = []

    # Fixed anisotropic TaIrTe4 sheet.
    for i in range(N_TA):
        for j in range(N_TA):
            left = ta_id(i, j)
            if i + 1 < N_TA:
                _add_edge(
                    row, col, data, left, ta_id(i + 1, j),
                    SIGMA_TA_XY_S_M[0] * TA_THICKNESS_M,
                )
            if j + 1 < N_TA:
                _add_edge(
                    row, col, data, left, ta_id(i, j + 1),
                    SIGMA_TA_XY_S_M[1] * TA_THICKNESS_M,
                )

    sigma_floor = SIGMA_AU_S_M * SIGMA_FLOOR_FRACTION
    sigma = sigma_floor + density * (SIGMA_AU_S_M - sigma_floor)
    dsigma = np.full_like(density, SIGMA_AU_S_M - sigma_floor)
    # Floating Au nanostructure sheet.  Harmonic half-cell resistance is used
    # at every lateral face and differentiated for both adjacent pixels.
    for i in range(N_DESIGN):
        for j in range(N_DESIGN):
            left = au_id(i, j)
            if i + 1 < N_DESIGN:
                right = au_id(i + 1, j)
                resistance = 0.5 * STEP_M / sigma[i, j] + 0.5 * STEP_M / sigma[i + 1, j]
                g = AU_THICKNESS_M * STEP_M / resistance
                _add_edge(row, col, data, left, right, g)
                for ii, jj in ((i, j), (i + 1, j)):
                    dg = (
                        AU_THICKNESS_M * STEP_M / resistance**2
                        * 0.5 * STEP_M / sigma[ii, jj] ** 2 * dsigma[ii, jj]
                    )
                    derivative_terms.append(
                        EdgeDerivative(left, right, ii * N_DESIGN + jj, dg, "Au_sheet_x")
                    )
            if j + 1 < N_DESIGN:
                right = au_id(i, j + 1)
                resistance = 0.5 * STEP_M / sigma[i, j] + 0.5 * STEP_M / sigma[i, j + 1]
                g = AU_THICKNESS_M * STEP_M / resistance
                _add_edge(row, col, data, left, right, g)
                for ii, jj in ((i, j), (i, j + 1)):
                    dg = (
                        AU_THICKNESS_M * STEP_M / resistance**2
                        * 0.5 * STEP_M / sigma[ii, jj] ** 2 * dsigma[ii, jj]
                    )
                    derivative_terms.append(
                        EdgeDerivative(left, right, ii * N_DESIGN + jj, dg, "Au_sheet_y")
                    )

            ti = DESIGN_OFFSET + i
            tj = DESIGN_OFFSET + j
            contact_floor = contact_S_m2 * CONTACT_FLOOR_FRACTION
            g_contact = STEP_M**2 * (contact_floor + density[i, j] * contact_S_m2)
            dg_contact = STEP_M**2 * contact_S_m2
            _add_edge(row, col, data, ta_id(ti, tj), left, g_contact)
            derivative_terms.append(
                EdgeDerivative(
                    ta_id(ti, tj), left, i * N_DESIGN + j, dg_contact,
                    "vertical_Au_Ta_electrical_contact",
                )
            )

    matrix = sparse.coo_matrix((data, (row, col)), shape=(node_count, node_count)).tocsr()
    matrix.sum_duplicates()
    # Low terminal y=-10 um, high terminal y=+10 um on TaIrTe4 only.
    low = np.asarray([ta_id(i, 0) for i in range(N_TA)], dtype=np.int64)
    high = np.asarray([ta_id(i, N_TA - 1) for i in range(N_TA)], dtype=np.int64)
    fixed = np.concatenate((low, high))
    fixed_values = np.concatenate((np.zeros(low.size), np.ones(high.size)))
    free_mask = np.ones(node_count, dtype=bool)
    free_mask[fixed] = False
    free = np.flatnonzero(free_mask)
    reduced = matrix[free][:, free].tocsr()
    rhs = -np.asarray(matrix[free][:, fixed] @ fixed_values).reshape(-1)

    # Fixed asymmetric temperature field in TaIrTe4.  Au Seebeck is zero, so
    # the objective contains TaIrTe4 thermoelectric generation only.
    coord = (np.arange(N_TA) + 0.5 - N_TA / 2.0) * STEP_M
    xx, yy = np.meshgrid(coord, coord, indexing="ij")
    temperature = np.exp(-2.0 * ((xx + 1.1e-6) ** 2 + (yy - 2.2e-6) ** 2) / (3.2e-6) ** 2)
    temperature *= 1.0
    objective_gradient = np.zeros(node_count, dtype=np.float64)
    for i in range(N_TA):
        for j in range(N_TA):
            left = ta_id(i, j)
            if i + 1 < N_TA:
                right = ta_id(i + 1, j)
                coefficient = (
                    SIGMA_TA_XY_S_M[0] * TA_THICKNESS_M
                    * SEEBECK_TA_XY_V_K[0]
                    * (temperature[i + 1, j] - temperature[i, j])
                )
                objective_gradient[left] -= coefficient
                objective_gradient[right] += coefficient
            if j + 1 < N_TA:
                right = ta_id(i, j + 1)
                coefficient = (
                    SIGMA_TA_XY_S_M[1] * TA_THICKNESS_M
                    * SEEBECK_TA_XY_V_K[1]
                    * (temperature[i, j + 1] - temperature[i, j])
                )
                objective_gradient[left] -= coefficient
                objective_gradient[right] += coefficient
    return ElectricalSystem(
        full_matrix_S=matrix,
        reduced_matrix_S=reduced,
        reduced_rhs_A=rhs,
        free=free,
        fixed=fixed,
        fixed_values_V=fixed_values,
        objective_gradient_psi_A=objective_gradient,
        derivative_terms=tuple(derivative_terms),
        temperature_K=temperature,
        rho=density.copy(),
        contact_conductance_S_m2=float(contact_S_m2),
    )


def solve_gpu(
    system: ElectricalSystem, cuda_device: int, *, need_adjoint: bool = True
) -> tuple[np.ndarray, np.ndarray | None, dict[str, float | int | bool]]:
    operator = PersistentCudaCSR(system.reduced_matrix_S, cuda_device=cuda_device)
    forward = operator.solve(
        system.reduced_rhs_A,
        relative_tolerance=1.0e-11,
        max_iterations=30000,
        residual_check_interval=10,
    )
    psi = np.zeros(system.full_matrix_S.shape[0], dtype=np.float64)
    psi[system.fixed] = system.fixed_values_V
    psi[system.free] = forward.solution
    adjoint_full = None
    adjoint_residual = 0.0
    adjoint_iterations = 0
    if need_adjoint:
        rhs_adjoint = system.objective_gradient_psi_A[system.free]
        adjoint = operator.solve(
            rhs_adjoint,
            relative_tolerance=1.0e-11,
            max_iterations=30000,
            residual_check_interval=10,
        )
        adjoint_full = np.zeros_like(psi)
        adjoint_full[system.free] = adjoint.solution
        adjoint_residual = float(adjoint.explicit_relative_residual)
        adjoint_iterations = int(adjoint.iterations)
    return psi, adjoint_full, {
        "weighting_relative_residual": float(forward.explicit_relative_residual),
        "adjoint_relative_residual": adjoint_residual,
        "weighting_iterations": int(forward.iterations),
        "adjoint_iterations": adjoint_iterations,
        "CPU_linear_solve_fallback": False,
    }


def objective(system: ElectricalSystem, psi: np.ndarray) -> float:
    return float(system.objective_gradient_psi_A @ psi)


def gradient(system: ElectricalSystem, psi: np.ndarray, adjoint: np.ndarray) -> np.ndarray:
    result = np.zeros(N_DESIGN * N_DESIGN, dtype=np.float64)
    for term in system.derivative_terms:
        result[term.rho_index] += -term.dg_drho_S * (
            adjoint[term.left] - adjoint[term.right]
        ) * (
            psi[term.left] - psi[term.right]
        )
    return result.reshape(N_DESIGN, N_DESIGN)


def audit(system: ElectricalSystem, psi: np.ndarray) -> dict[str, float]:
    residual_full = np.asarray(system.full_matrix_S @ psi).reshape(-1)
    free_residual = np.linalg.norm(residual_full[system.free]) / max(
        np.linalg.norm(system.reduced_rhs_A), np.finfo(float).tiny
    )
    low_current = float(np.sum(residual_full[system.fixed[:N_TA]]))
    high_current = float(np.sum(residual_full[system.fixed[N_TA:]]))
    terminal_balance = abs(low_current + high_current) / max(
        abs(low_current), abs(high_current), np.finfo(float).tiny
    )
    return {
        "free_equation_relative_residual": float(free_residual),
        "low_terminal_current_A_per_V": low_current,
        "high_terminal_current_A_per_V": high_current,
        "terminal_current_balance": terminal_balance,
        "psi_min": float(np.min(psi)),
        "psi_max": float(np.max(psi)),
    }


def directions(g: np.ndarray) -> dict[str, np.ndarray]:
    x = np.linspace(-1.0, 1.0, N_DESIGN)[:, None]
    y = np.linspace(-1.0, 1.0, N_DESIGN)[None, :]
    rng = np.random.default_rng(20260821)
    raw = {
        "uniform": np.ones((N_DESIGN, N_DESIGN)),
        "smooth_asymmetric": np.sin(0.72 * np.pi * x) * np.cos(0.53 * np.pi * y) + 0.2 * y,
        "central_localized": np.exp(-((x - 0.08) ** 2 + (y + 0.1) ** 2) / 0.07),
        "design_edge_localized": np.exp(-((x + 0.86) ** 2 + (y - 0.3) ** 2) / 0.025),
        "fixed_seed_random": ndimage.gaussian_filter(rng.normal(size=(N_DESIGN, N_DESIGN)), 1.5),
        "adjoint_aligned": g.copy(),
    }
    return {name: value / np.linalg.norm(value) for name, value in raw.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--cuda-device", required=True, type=int)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to one physical GPU")
    output = args.output_dir.resolve()
    raw_dir = args.raw_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    rho0 = base_density()
    steps = (0.01, 0.005, 0.0025)
    rows: list[dict[str, object]] = []
    scenarios: dict[str, object] = {}
    raw_arrays: dict[str, np.ndarray] = {"rho": rho0}
    worst_error = 0.0
    worst_residual = 0.0
    worst_balance = 0.0
    for scenario_name, contact in CONTACT_SCENARIOS_S_M2.items():
        system = build_system(rho0, contact)
        psi, adjoint, solver = solve_gpu(system, args.cuda_device)
        if adjoint is None:
            raise RuntimeError("missing electrical adjoint")
        g = gradient(system, psi, adjoint)
        base_current = objective(system, psi)
        base_audit = audit(system, psi)
        scenario_rows: list[dict[str, object]] = []
        for direction_name, direction in directions(g).items():
            ad = float(np.sum(g * direction))
            for step in steps:
                values: list[float] = []
                residuals: list[float] = []
                balances: list[float] = []
                for sign in (1.0, -1.0):
                    local_rho = rho0 + sign * step * direction
                    if np.any((local_rho <= 0.0) | (local_rho >= 1.0)):
                        raise RuntimeError("FD perturbation clipped")
                    local = build_system(local_rho, contact)
                    local_psi, _, local_solver = solve_gpu(local, args.cuda_device, need_adjoint=False)
                    local_audit = audit(local, local_psi)
                    values.append(objective(local, local_psi))
                    residuals.append(float(local_solver["weighting_relative_residual"]))
                    balances.append(local_audit["terminal_current_balance"])
                fd = (values[0] - values[1]) / (2.0 * step)
                row = {
                    "scenario": scenario_name,
                    "contact_conductance_S_m2": contact,
                    "contact_resistivity_ohm_m2": 1.0 / contact,
                    "direction": direction_name,
                    "step": step,
                    "AD_A_per_rho": ad,
                    "FD_A_per_rho": fd,
                    "relative_error": relative(ad, fd),
                    "plus_current_A": values[0],
                    "minus_current_A": values[1],
                    "worst_residual": max(residuals),
                    "worst_terminal_balance": max(balances),
                    "clipping": False,
                }
                rows.append(row)
                scenario_rows.append(row)
        fine = [row for row in scenario_rows if row["step"] == min(steps)]
        fine_error = max(float(row["relative_error"]) for row in fine)
        worst_error = max(worst_error, fine_error)
        worst_residual = max(
            worst_residual,
            float(solver["weighting_relative_residual"]),
            float(solver["adjoint_relative_residual"]),
            float(base_audit["free_equation_relative_residual"]),
            max(float(row["worst_residual"]) for row in scenario_rows),
        )
        worst_balance = max(
            worst_balance,
            float(base_audit["terminal_current_balance"]),
            max(float(row["worst_terminal_balance"]) for row in scenario_rows),
        )
        ta_psi = psi[: N_TA * N_TA].reshape(N_TA, N_TA)
        au_psi = psi[N_TA * N_TA :].reshape(N_DESIGN, N_DESIGN)
        scenarios[scenario_name] = {
            "contact_conductance_S_m2": contact,
            "contact_resistivity_ohm_m2": 1.0 / contact,
            "provenance": "numerical scenario, not measured Au/TaIrTe4 contact data",
            "base_current_A_for_fixed_unit_temperature_field": base_current,
            "gradient_l2_A_per_rho": float(np.linalg.norm(g)),
            "worst_fine_step_ADFD_relative_error": fine_error,
            "solver": solver,
            "audit": base_audit,
            "mean_vertical_potential_difference": float(
                np.mean(
                    ta_psi[
                        DESIGN_OFFSET : DESIGN_OFFSET + N_DESIGN,
                        DESIGN_OFFSET : DESIGN_OFFSET + N_DESIGN,
                    ] - au_psi
                )
            ),
        }
        raw_arrays[f"psi_Ta_{scenario_name}"] = ta_psi
        raw_arrays[f"psi_Au_{scenario_name}"] = au_psi
        raw_arrays[f"gradient_{scenario_name}"] = g

    passed = bool(worst_error < 0.01 and worst_residual < 1.0e-8 and worst_balance < 1.0e-8)
    status = "VALIDATED_FLOATING_AU_WEIGHTING_ELECTRICAL_CONTROL" if passed else "FAILED_FLOATING_AU_WEIGHTING_ELECTRICAL_CONTROL"
    summary = {
        "status": status,
        "scope": "electrical weighting/shunting operator only; fixed temperature, no Maxwell/thermal/optimization",
        "geometry": {
            "fixed_TaIrTe4_cells_xy": [N_TA, N_TA],
            "fixed_TaIrTe4_span_m": [N_TA * STEP_M, N_TA * STEP_M],
            "Au_design_cells_xy": [N_DESIGN, N_DESIGN],
            "Au_design_span_m": [N_DESIGN * STEP_M, N_DESIGN * STEP_M],
            "pixel_m": STEP_M,
            "terminals": "TaIrTe4 y-min=0, y-max=1; Au is floating",
        },
        "materials": {
            "sigma_TaIrTe4_xy_S_m": SIGMA_TA_XY_S_M.tolist(),
            "Seebeck_TaIrTe4_xy_V_K": SEEBECK_TA_XY_V_K.tolist(),
            "sigma_Au_bulk_reference_S_m": SIGMA_AU_S_M,
            "S_Au_V_K": 0.0,
            "Au_note": "bulk reference; not certified 50-nm-film transport",
        },
        "gray_law": {
            "Au_sheet_sigma": "sigma_floor + rho*(sigma_Au-sigma_floor)",
            "vertical_contact": "A*(G_floor + rho*G_contact)",
            "sigma_floor_fraction": SIGMA_FLOOR_FRACTION,
            "contact_floor_fraction": CONTACT_FLOOR_FRACTION,
            "floor_purpose": "fixed-shape SPD operator for gray AD-FD; reported, not physical air conduction",
        },
        "scenarios": scenarios,
        "gates": {
            "worst_h0p0025_ADFD_relative_error": worst_error,
            "required_ADFD_relative_error": 0.01,
            "worst_linear_residual": worst_residual,
            "required_linear_residual": 1.0e-8,
            "worst_terminal_current_balance": worst_balance,
            "required_terminal_current_balance": 1.0e-8,
            "CPU_linear_solve_fallback": False,
        },
        "limitations": [
            "No device-specific Au/TaIrTe4 electrical contact resistivity was supplied or identified.",
            "All contact resistivities are named numerical scenarios, not a confidence interval.",
            "S_Au=0 isolates weighting-field and shunting effects; metal thermopower remains a later sensitivity.",
            "This control uses a fixed temperature field and is not a coupled PTE prediction.",
        ],
    }
    raw_path = raw_dir / "au_weighting_electrical_adfd_raw.npz"
    np.savez_compressed(raw_path, **raw_arrays)
    csv_path = output / "au_weighting_electrical_adfd_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output / "au_weighting_electrical_adfd_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "raw_artifact_committed_to_git": False,
        "raw_artifact": {
            "path": str(raw_path), "bytes": raw_path.stat().st_size, "sha256": sha256(raw_path),
            "generation_command": "CUDA_VISIBLE_DEVICES=<one GPU> python 54_validate_au_weighting_electrical_adfd.py ...",
        },
        "published": [
            {"path": str(summary_path), "bytes": summary_path.stat().st_size, "sha256": sha256(summary_path)},
            {"path": str(csv_path), "bytes": csv_path.stat().st_size, "sha256": sha256(csv_path)},
        ],
    }
    (output / "RAW_ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
