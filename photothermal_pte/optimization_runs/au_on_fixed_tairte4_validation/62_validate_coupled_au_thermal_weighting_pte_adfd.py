#!/usr/bin/env python3
"""GPU fixed-Q coupled Au thermal/contact + weighting/electrical PTE AD--FD."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, sparse

from photothermal_pte.optimization_runs.au_on_fixed_tairte4_validation.material_model import (
    AU_BULK_THERMAL_CONDUCTIVITY_W_MK,
)
from photothermal_pte.optimization_runs.cuda_thermal_adjoint import PersistentCudaCSR


HERE = Path(__file__).resolve().parent
STAGE54 = HERE / "54_validate_au_weighting_electrical_adfd.py"
N_TA = 40
N_DESIGN = 20
OFFSET = 10
STEP_M = 500.0e-9
TA_DZ_M = 100.0e-9
AU_DZ_M = 50.0e-9
K_TA_XY_W_MK = (3.8, 14.4)  # Lumerical x=b, y=a.
K_TA_Z_W_MK = 1.0
K_AIR_W_MK = 0.026
K_AU_W_MK = float(AU_BULK_THERMAL_CONDUCTIVITY_W_MK)
G_TA_AIR_W_M2K = 1.0
G_TA_SIO2_W_M2K = 7.37e6
H_AU_AIR_W_M2K = 10.0
G_AU_TA_W_M2K = 1.0 / 5.8e-8
ELECTRICAL_CONTACT_S_M2 = 1.0e10
SOURCE_POWER_W = 1.0e-6


def _load_stage54():
    spec = importlib.util.spec_from_file_location("stage54_au_electrical", STAGE54)
    if spec is None or spec.loader is None:
        raise ImportError(STAGE54)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), np.finfo(float).tiny)


def ta_id(i: int, j: int) -> int:
    return i * N_TA + j


def au_id(i: int, j: int) -> int:
    return N_TA * N_TA + i * N_DESIGN + j


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
    derivative_terms: tuple[EdgeDerivative, ...]
    bottom_conductance_W_K: np.ndarray
    uncovered_top_conductance_W_K: np.ndarray
    design_top_conductance_W_K: np.ndarray
    source_power_W: float


def _add_edge(row, col, data, left: int, right: int, conductance: float) -> None:
    row.extend((left, right, left, right))
    col.extend((left, right, right, left))
    data.extend((conductance, conductance, -conductance, -conductance))


def _add_boundary(row, col, data, node: int, conductance: float) -> None:
    row.append(node)
    col.append(node)
    data.append(conductance)


def base_density() -> np.ndarray:
    x = np.linspace(-1.0, 1.0, N_DESIGN)[:, None]
    y = np.linspace(-1.0, 1.0, N_DESIGN)[None, :]
    return 0.52 + 0.07 * np.cos(0.75 * np.pi * x) * np.cos(0.6 * np.pi * y) + 0.018 * x


def build_thermal(rho: np.ndarray) -> ThermalSystem:
    density = np.asarray(rho, dtype=np.float64)
    if density.shape != (N_DESIGN, N_DESIGN) or np.any((density <= 0) | (density >= 1)):
        raise ValueError("thermal control requires unclipped 20x20 density")
    node_count = N_TA * N_TA + N_DESIGN * N_DESIGN
    row: list[int] = []
    col: list[int] = []
    data: list[float] = []
    terms: list[EdgeDerivative] = []
    area = STEP_M**2
    half_ta = 0.5 * TA_DZ_M / K_TA_Z_W_MK

    for i in range(N_TA):
        for j in range(N_TA):
            left = ta_id(i, j)
            if i + 1 < N_TA:
                _add_edge(row, col, data, left, ta_id(i + 1, j), K_TA_XY_W_MK[0] * TA_DZ_M)
            if j + 1 < N_TA:
                _add_edge(row, col, data, left, ta_id(i, j + 1), K_TA_XY_W_MK[1] * TA_DZ_M)

    phi = density
    k_design = K_AIR_W_MK + phi * (K_AU_W_MK - K_AIR_W_MK)
    dk = np.full_like(phi, K_AU_W_MK - K_AIR_W_MK)
    for i in range(N_DESIGN):
        for j in range(N_DESIGN):
            left = au_id(i, j)
            if i + 1 < N_DESIGN:
                right = au_id(i + 1, j)
                resistance = 0.5 * STEP_M / k_design[i, j] + 0.5 * STEP_M / k_design[i + 1, j]
                conductance = STEP_M * AU_DZ_M / resistance
                _add_edge(row, col, data, left, right, conductance)
                for ii, jj in ((i, j), (i + 1, j)):
                    derivative = (
                        STEP_M * AU_DZ_M / resistance**2 * 0.5 * STEP_M
                        / k_design[ii, jj] ** 2 * dk[ii, jj]
                    )
                    terms.append(EdgeDerivative(left, right, ii * N_DESIGN + jj, derivative, "Au_layer_x"))
            if j + 1 < N_DESIGN:
                right = au_id(i, j + 1)
                resistance = 0.5 * STEP_M / k_design[i, j] + 0.5 * STEP_M / k_design[i, j + 1]
                conductance = STEP_M * AU_DZ_M / resistance
                _add_edge(row, col, data, left, right, conductance)
                for ii, jj in ((i, j), (i, j + 1)):
                    derivative = (
                        STEP_M * AU_DZ_M / resistance**2 * 0.5 * STEP_M
                        / k_design[ii, jj] ** 2 * dk[ii, jj]
                    )
                    terms.append(EdgeDerivative(left, right, ii * N_DESIGN + jj, derivative, "Au_layer_y"))

    bottom = np.full((N_TA, N_TA), area / (half_ta + 1.0 / G_TA_SIO2_W_M2K))
    uncovered = np.zeros((N_TA, N_TA), dtype=np.float64)
    design_top = np.zeros((N_DESIGN, N_DESIGN), dtype=np.float64)
    r_air = half_ta + 1.0 / G_TA_AIR_W_M2K + 0.5 * AU_DZ_M / K_AIR_W_MK
    r_au = half_ta + 1.0 / G_AU_TA_W_M2K + 0.5 * AU_DZ_M / K_AU_W_MK
    for i in range(N_TA):
        for j in range(N_TA):
            node = ta_id(i, j)
            _add_boundary(row, col, data, node, bottom[i, j])
            if not (OFFSET <= i < OFFSET + N_DESIGN and OFFSET <= j < OFFSET + N_DESIGN):
                uncovered[i, j] = area / (half_ta + 1.0 / G_TA_AIR_W_M2K)
                _add_boundary(row, col, data, node, uncovered[i, j])

    for i in range(N_DESIGN):
        for j in range(N_DESIGN):
            index = i * N_DESIGN + j
            ta = ta_id(OFFSET + i, OFFSET + j)
            top = au_id(i, j)
            vertical = area * ((1.0 - phi[i, j]) / r_air + phi[i, j] / r_au)
            dvertical = area * (1.0 / r_au - 1.0 / r_air)
            _add_edge(row, col, data, ta, top, vertical)
            terms.append(EdgeDerivative(ta, top, index, dvertical, "Ta_to_Au_or_air_parallel_area"))
            resistance = 0.5 * AU_DZ_M / k_design[i, j] + 1.0 / H_AU_AIR_W_M2K
            design_top[i, j] = area / resistance
            dtop = area / resistance**2 * 0.5 * AU_DZ_M / k_design[i, j] ** 2 * dk[i, j]
            _add_boundary(row, col, data, top, design_top[i, j])
            terms.append(EdgeDerivative(top, None, index, dtop, "design_top_ambient"))

    matrix = sparse.coo_matrix((data, (row, col)), shape=(node_count, node_count)).tocsr()
    matrix.sum_duplicates()
    coords = (np.arange(N_TA) + 0.5 - N_TA / 2.0) * STEP_M
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    shape = np.exp(-2.0 * ((xx + 1.1e-6) ** 2 + (yy - 2.2e-6) ** 2) / (3.2e-6) ** 2)
    source = SOURCE_POWER_W * shape / np.sum(shape)
    rhs = np.zeros(node_count, dtype=np.float64)
    rhs[: N_TA * N_TA] = source.reshape(-1)
    return ThermalSystem(matrix, rhs, tuple(terms), bottom, uncovered, design_top, float(np.sum(rhs)))


def solve_linear(matrix: sparse.csr_matrix, rhs: np.ndarray, cuda_device: int):
    operator = PersistentCudaCSR(matrix, cuda_device=cuda_device)
    return operator.solve(rhs, relative_tolerance=1.0e-11, max_iterations=30000, residual_check_interval=10)


def electrical_load(temperature: np.ndarray, electrical) -> np.ndarray:
    load = np.zeros(N_TA * N_TA + N_DESIGN * N_DESIGN, dtype=np.float64)
    for i in range(N_TA):
        for j in range(N_TA):
            left = ta_id(i, j)
            if i + 1 < N_TA:
                right = ta_id(i + 1, j)
                value = electrical.SIGMA_TA_XY_S_M[0] * TA_DZ_M * electrical.SEEBECK_TA_XY_V_K[0] * (temperature[i + 1, j] - temperature[i, j])
                load[left] -= value
                load[right] += value
            if j + 1 < N_TA:
                right = ta_id(i, j + 1)
                value = electrical.SIGMA_TA_XY_S_M[1] * TA_DZ_M * electrical.SEEBECK_TA_XY_V_K[1] * (temperature[i, j + 1] - temperature[i, j])
                load[left] -= value
                load[right] += value
    return load


def temperature_pullback(psi: np.ndarray, electrical) -> np.ndarray:
    gradient = np.zeros((N_TA, N_TA), dtype=np.float64)
    for i in range(N_TA):
        for j in range(N_TA):
            left = ta_id(i, j)
            if i + 1 < N_TA:
                right = ta_id(i + 1, j)
                scale = electrical.SIGMA_TA_XY_S_M[0] * TA_DZ_M * electrical.SEEBECK_TA_XY_V_K[0]
                contribution = scale * (psi[right] - psi[left])
                gradient[i, j] -= contribution
                gradient[i + 1, j] += contribution
            if j + 1 < N_TA:
                right = ta_id(i, j + 1)
                scale = electrical.SIGMA_TA_XY_S_M[1] * TA_DZ_M * electrical.SEEBECK_TA_XY_V_K[1]
                contribution = scale * (psi[right] - psi[left])
                gradient[i, j] -= contribution
                gradient[i, j + 1] += contribution
    return gradient


def thermal_gradient(system: ThermalSystem, temperature: np.ndarray, adjoint: np.ndarray) -> np.ndarray:
    result = np.zeros(N_DESIGN * N_DESIGN, dtype=np.float64)
    for term in system.derivative_terms:
        if term.right is None:
            value = -term.dg_drho_W_K * adjoint[term.left] * temperature[term.left]
        else:
            value = -term.dg_drho_W_K * (adjoint[term.left] - adjoint[term.right]) * (temperature[term.left] - temperature[term.right])
        result[term.rho_index] += value
    return result.reshape(N_DESIGN, N_DESIGN)


def energy_balance(system: ThermalSystem, temperature: np.ndarray) -> float:
    ta = temperature[: N_TA * N_TA].reshape(N_TA, N_TA)
    au = temperature[N_TA * N_TA :].reshape(N_DESIGN, N_DESIGN)
    output = np.sum(system.bottom_conductance_W_K * ta)
    output += np.sum(system.uncovered_top_conductance_W_K * ta)
    output += np.sum(system.design_top_conductance_W_K * au)
    return float(abs(output - system.source_power_W) / system.source_power_W)


def evaluate(rho: np.ndarray, cuda_device: int, *, need_gradient: bool):
    electrical = _load_stage54()
    thermal = build_thermal(rho)
    forward = solve_linear(thermal.matrix_W_K, thermal.rhs_W, cuda_device)
    ta_temperature = forward.solution[: N_TA * N_TA].reshape(N_TA, N_TA)
    electrical_base = electrical.build_system(rho, ELECTRICAL_CONTACT_S_M2)
    electrical_system = replace(electrical_base, objective_gradient_psi_A=electrical_load(ta_temperature, electrical))
    psi, electrical_adjoint, electrical_solver = electrical.solve_gpu(electrical_system, cuda_device, need_adjoint=need_gradient)
    current = electrical.objective(electrical_system, psi)
    electrical_audit = electrical.audit(electrical_system, psi)
    result = {
        "objective_A": current,
        "temperature": forward.solution,
        "weighting": psi,
        "thermal_residual": float(forward.explicit_relative_residual),
        "electrical_residual": float(electrical_solver["weighting_relative_residual"]),
        "electrical_balance": float(electrical_audit["terminal_current_balance"]),
        "energy_balance": energy_balance(thermal, forward.solution),
    }
    if need_gradient:
        rhs = np.zeros_like(forward.solution)
        rhs[: N_TA * N_TA] = temperature_pullback(psi, electrical).reshape(-1)
        thermal_adjoint = solve_linear(thermal.matrix_W_K, rhs, cuda_device)
        g_thermal = thermal_gradient(thermal, forward.solution, thermal_adjoint.solution)
        g_electrical = electrical.gradient(electrical_system, psi, electrical_adjoint)
        result.update(
            gradient_thermal=g_thermal,
            gradient_electrical=g_electrical,
            gradient_total=g_thermal + g_electrical,
            thermal_adjoint_residual=float(thermal_adjoint.explicit_relative_residual),
            electrical_adjoint_residual=float(electrical_solver["adjoint_relative_residual"]),
        )
    return result


def directions(gradient: np.ndarray) -> dict[str, np.ndarray]:
    x = np.linspace(-1.0, 1.0, N_DESIGN)[:, None]
    y = np.linspace(-1.0, 1.0, N_DESIGN)[None, :]
    rng = np.random.default_rng(20260821)
    raw = {
        "adjoint_aligned": gradient,
        "smooth_asymmetric": np.sin(0.72 * np.pi * x) * np.cos(0.53 * np.pi * y) + 0.2 * y,
        "central_localized": np.exp(-((x - 0.08) ** 2 + (y + 0.1) ** 2) / 0.07),
        "design_edge_localized": np.exp(-((x + 0.86) ** 2 + (y - 0.3) ** 2) / 0.025),
        "fixed_seed_random": ndimage.gaussian_filter(rng.normal(size=(N_DESIGN, N_DESIGN)), 1.5),
    }
    return {name: value / np.linalg.norm(value) for name, value in raw.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    raw_dir = args.raw_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rho = base_density()
    base = evaluate(rho, args.cuda_device, need_gradient=True)
    rows = []
    for name, direction in directions(base["gradient_total"]).items():
        ad = float(np.sum(base["gradient_total"] * direction))
        for step in (0.01, 0.005, 0.0025):
            plus = evaluate(rho + step * direction, args.cuda_device, need_gradient=False)
            minus = evaluate(rho - step * direction, args.cuda_device, need_gradient=False)
            fd = (plus["objective_A"] - minus["objective_A"]) / (2.0 * step)
            strong = bool(abs(ad) >= 0.05 * np.linalg.norm(base["gradient_total"]))
            rows.append({
                "direction": name,
                "h": step,
                "AD_A": ad,
                "FD_A": fd,
                "strong": strong,
                "strong_relative_error": _relative(ad, fd),
                "gradient_l2_normalized_error": abs(ad - fd) / np.linalg.norm(base["gradient_total"]),
                "plus_A": plus["objective_A"],
                "minus_A": minus["objective_A"],
            })
    finest = [row for row in rows if row["h"] == 0.0025]
    worst_strong = max((row["strong_relative_error"] for row in finest if row["strong"]), default=0.0)
    worst_normalized = max(row["gradient_l2_normalized_error"] for row in finest)
    residual = max(base["thermal_residual"], base["thermal_adjoint_residual"], base["electrical_residual"], base["electrical_adjoint_residual"])
    gates = {
        "strong_direction_error_lt_1pct": bool(worst_strong < 0.01),
        "multi_direction_gradient_l2_normalized_error_lt_1pct": bool(worst_normalized < 0.01),
        "linear_residual_lt_1e-8": bool(residual < 1.0e-8),
        "thermal_energy_balance_lt_1pct": bool(base["energy_balance"] < 0.01),
        "electrical_terminal_balance_lt_1pct": bool(base["electrical_balance"] < 0.01),
        "unclipped_density": bool(np.all((rho > 0.0) & (rho < 1.0))),
        "CPU_linear_solve_fallback": False,
    }
    passed = all(value for key, value in gates.items() if key != "CPU_linear_solve_fallback") and not gates["CPU_linear_solve_fallback"]
    raw_path = raw_dir / "au_coupled_thermal_weighting_pte_fixed_q_raw.npz"
    np.savez_compressed(
        raw_path,
        rho=rho,
        temperature_K=base["temperature"],
        weighting_potential=base["weighting"],
        gradient_total_A=base["gradient_total"],
        gradient_thermal_A=base["gradient_thermal"],
        gradient_electrical_A=base["gradient_electrical"],
    )
    summary = {
        "status": "VALIDATED_COUPLED_AU_THERMAL_WEIGHTING_PTE_FIXED_Q_CONTROL" if passed else "FAILED_COUPLED_AU_THERMAL_WEIGHTING_PTE_FIXED_Q_CONTROL",
        "scope": "fixed-Q coupled Au thermal/contact and floating-Au electrical/weighting PTE operator; no Maxwell optical gradient or optimization",
        "geometry": {"TaIrTe4_shape": [40, 40], "Au_design_shape": [20, 20], "pitch_m": STEP_M, "TaIrTe4_thickness_m": TA_DZ_M, "Au_thickness_m": AU_DZ_M},
        "parameters": {
            "k_Ta_x_b_y_a_z_W_mK": [*K_TA_XY_W_MK, K_TA_Z_W_MK],
            "k_Au_W_mK": K_AU_W_MK,
            "G_Ta_SiO2_W_m2K": G_TA_SIO2_W_M2K,
            "G_Ta_air_W_m2K": G_TA_AIR_W_M2K,
            "G_Au_Ta_W_m2K": G_AU_TA_W_M2K,
            "G_Au_Ta_provenance": "Au/MoS2 calculated analogue; numerical scenario, not TaIrTe4 measurement",
            "electrical_contact_S_m2": ELECTRICAL_CONTACT_S_M2,
            "electrical_contact_provenance": "numerical scenario, not measured Au/TaIrTe4 contact",
            "S_Au_V_K": 0.0,
        },
        "base": {
            "objective_A": base["objective_A"],
            "source_power_W": SOURCE_POWER_W,
            "Tmax_K": float(np.max(base["temperature"])),
            "gradient_norms_A": {
                "total": float(np.linalg.norm(base["gradient_total"])),
                "thermal": float(np.linalg.norm(base["gradient_thermal"])),
                "electrical": float(np.linalg.norm(base["gradient_electrical"])),
            },
            "residual_max": residual,
            "thermal_energy_balance": base["energy_balance"],
            "electrical_terminal_balance": base["electrical_balance"],
        },
        "directions": rows,
        "worst_finest_strong_error": worst_strong,
        "worst_finest_gradient_l2_normalized_error": worst_normalized,
        "gates": gates,
        "raw_artifact": {"path": str(raw_path), "bytes": raw_path.stat().st_size, "sha256": _sha256(raw_path)},
        "no_clipping_smoothing_gain_or_gradient_rescaling": True,
        "optimization_iterations": 0,
    }
    summary_path = output / "au_coupled_thermal_weighting_pte_fixed_q_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    csv_path = output / "au_coupled_thermal_weighting_pte_fixed_q_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes[0, 0].imshow(rho.T, origin="lower", cmap="gray_r", vmin=0, vmax=1)
    axes[0, 0].set_title("Au physical density")
    axes[0, 1].imshow(base["temperature"][: N_TA*N_TA].reshape(N_TA, N_TA).T, origin="lower")
    axes[0, 1].set_title("TaIrTe4 temperature rise")
    axes[1, 0].imshow(base["weighting"][: N_TA*N_TA].reshape(N_TA, N_TA).T, origin="lower", vmin=0, vmax=1)
    axes[1, 0].set_title("TaIrTe4 weighting potential")
    axes[1, 1].imshow(base["gradient_total"].T, origin="lower", cmap="coolwarm")
    axes[1, 1].set_title("coupled physical-density gradient")
    plot_path = output / "au_coupled_thermal_weighting_pte_fixed_q_fields.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    report_path = output / "AU_COUPLED_THERMAL_WEIGHTING_PTE_FIXED_Q_REPORT.md"
    report_path.write_text(
        f"""# Coupled Au thermal/weighting PTE fixed-Q control

Status: **{summary['status']}**

This control couples the same 20x20 Au density to the Au/air thermal layer,
Au/TaIrTe4 thermal contact, floating-Au sheet conductivity, and finite
Au/TaIrTe4 electrical contact. The temperature generated by a fixed Gaussian
Q is passed to the anisotropic TaIrTe4 thermoelectric operator; both the
thermal-material/contact and electrical weighting/contact derivatives are
included.

At `h=0.0025`, the worst strong-direction AD--FD error is
`{100*worst_strong:.6f}%`; the worst multi-direction gradient-L2-normalized
error is `{100*worst_normalized:.6f}%`. Maximum linear residual is
`{residual:.3e}`, thermal energy balance is `{100*base['energy_balance']:.6f}%`,
and electrical terminal balance is `{100*base['electrical_balance']:.6f}%`.

`G_Au/Ta={G_AU_TA_W_M2K:.6e} W/(m2 K)` is an Au/MoS2 calculated analogue,
not a TaIrTe4 measurement. Electrical contact is also a named numerical
scenario. Au Seebeck is zero here to isolate shunting/weighting effects.

This is not a Maxwell-coupled PTE result. The next gate replaces fixed Q by
the spatial Au+TaIrTe4+SiO2 FDTDX source, contracts its native-Yee sensitivity
with the thermal adjoint, and repeats full directional AD--FD before any
optimization.
""",
        encoding="utf-8",
    )
    manifest = {
        "status": summary["status"],
        "raw_artifact": summary["raw_artifact"],
        "published": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (summary_path, csv_path, plot_path, report_path)
        ],
    }
    (output / "RAW_ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "worst_strong": worst_strong, "worst_normalized": worst_normalized, "residual": residual}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
