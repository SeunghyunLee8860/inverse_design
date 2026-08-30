#!/usr/bin/env python3
"""GPU fixed-Q certificate for the explicit TaIrTe4-to-void thermal branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import ndimage, sparse

from photothermal_pte.optimization_runs.cuda_thermal_adjoint import PersistentCudaCSR
from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT
from photothermal_pte.optimization_runs.tairte4_flake_topology.electrical import (
    build_rectangular_mesh,
    solve_weighting_and_adjoint,
)
from photothermal_pte.optimization_runs.tairte4_flake_topology.thermal import (
    G_SIO2_SI_W_M2K,
    G_TAIRTE4_SIO2_W_M2K,
    K_TAIRTE4_XYZ_W_MK,
    TOP_AIR_CONVECTION_W_M2K,
    boundary_energy_error,
    build_state,
    cell_to_node,
    flake_cell_temperature,
    flake_temperature_transpose,
    map_native_q,
    thermal_density_gradient,
)


SIGMA_XY_S_M = (1.10e5, 4.91e5)
SEEBECK_XY_V_K = (27.0e-6, -6.0e-6)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), np.finfo(float).tiny)


class CachedCudaSolve:
    def __init__(self, device: int):
        self.device = device
        self.reference: sparse.csr_matrix | None = None
        self.operator: PersistentCudaCSR | None = None

    def __call__(self, matrix: sparse.csr_matrix, rhs: np.ndarray) -> np.ndarray:
        candidate = sparse.csr_matrix(matrix, dtype=np.float64)
        if self.operator is None:
            self.reference = candidate.copy()
            self.operator = PersistentCudaCSR(candidate, cuda_device=self.device)
        else:
            assert self.reference is not None
            difference = candidate - self.reference
            mismatch = 0.0 if difference.nnz == 0 else float(np.max(np.abs(difference.data)))
            if mismatch > 1e-13 * max(float(np.max(np.abs(self.reference.data))), 1.0):
                raise RuntimeError("electrical forward/adjoint operators differ")
        return self.operator.solve(
            rhs, relative_tolerance=1e-10, max_iterations=30000
        ).solution


def full_flake_density(design_nodal: np.ndarray) -> np.ndarray:
    result = np.ones(CONTRACT.flake_node_shape, dtype=np.float64)
    result[CONTRACT.design_node_slices] = design_nodal
    return result


def directions(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    x = np.linspace(-1.0, 1.0, shape[0])[:, None]
    y = np.linspace(-1.0, 1.0, shape[1])[None, :]
    rng = np.random.default_rng(20260810)
    raw = {
        "smooth_asymmetric": 0.65 * np.sin(np.pi * x) * np.cos(0.5 * np.pi * y) + 0.35 * x,
        "central_localized": np.exp(-((x - 0.12) ** 2 + (y + 0.08) ** 2) / 0.08),
        "design_edge_localized": np.exp(-((x - 0.88) ** 2 + (y + 0.35) ** 2) / 0.025),
        "fixed_seed_random": ndimage.gaussian_filter(rng.normal(size=shape), sigma=4.0),
    }
    return {name: value / np.max(np.abs(value)) for name, value in raw.items()}


def solve_thermal(state, q_density, device: int):
    source_active = state.system.active_source(q_density)
    source_power = np.asarray(state.system.source_volume_operator_m3 @ source_active)
    operator = PersistentCudaCSR(state.system.matrix_W_K, cuda_device=device)
    solved = operator.solve(
        source_power,
        relative_tolerance=1e-10,
        max_iterations=30000,
        residual_check_interval=25,
    )
    return operator, solved, source_power


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-q", required=True, type=Path)
    parser.add_argument("--native-q-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--directions", default="smooth_asymmetric")
    parser.add_argument("--steps", default="0.01,0.005,0.0025")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to one physical GPU")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    native_path = args.native_q.expanduser().resolve()
    actual_sha = sha256(native_path)
    if actual_sha != args.native_q_sha256:
        raise RuntimeError("native Q SHA mismatch")

    base_density = np.full(CONTRACT.design_node_shape, 0.5, dtype=np.float64)
    started = perf_counter()
    state = build_state(base_density)
    assembly_s = perf_counter() - started
    native = np.load(native_path)
    q_density, mapping = map_native_q(native, state)
    if mapping["relative_mapping_error"] >= 0.005:
        raise RuntimeError("Q material-intersection mapping gate failed")
    operator, forward, source_power = solve_thermal(state, q_density, args.cuda_device)
    flake_temperature = cell_to_node(flake_cell_temperature(state, forward.solution))
    electrical_mesh = build_rectangular_mesh(
        CONTRACT.flake_span_m, CONTRACT.flake_span_m, CONTRACT.design_step_m
    )
    electrical_solver = CachedCudaSolve(args.cuda_device)
    electrical = solve_weighting_and_adjoint(
        electrical_mesh,
        full_flake_density(base_density),
        flake_temperature,
        thickness_m=CONTRACT.flake_thickness_m,
        sigma_xy_S_m=SIGMA_XY_S_M,
        seebeck_xy_V_K=SEEBECK_XY_V_K,
        sigma_void_fraction=CONTRACT.sigma_void_fraction,
        sigma_penalty=CONTRACT.sigma_penalty,
        alpha_penalty=CONTRACT.alpha_penalty,
        linear_solve=electrical_solver,
        terminal_axis=CONTRACT.contact_axis,
    )
    thermal_objective = float(np.sum(electrical.gradient_temperature_K_inv * flake_temperature))
    current_identity_error = relative(thermal_objective, electrical.current_A)
    thermal_rhs = flake_temperature_transpose(
        state, electrical.gradient_temperature_K_inv
    )
    adjoint = operator.solve(
        thermal_rhs,
        relative_tolerance=1e-10,
        max_iterations=30000,
        residual_check_interval=25,
    )
    gradient = thermal_density_gradient(state, forward.solution, adjoint.solution)
    energy, boundary_power = boundary_energy_error(state, forward.solution, source_power)

    requested = [item.strip() for item in args.directions.split(",") if item.strip()]
    all_directions = directions(CONTRACT.design_node_shape)
    if any(name not in all_directions for name in requested):
        raise ValueError(f"unknown directions: {requested}")
    steps = [float(item) for item in args.steps.split(",")]
    rows = []
    for name in requested:
        direction = all_directions[name]
        ad = float(np.sum(gradient * direction))
        for step in steps:
            objectives = []
            residuals = []
            energies = []
            seconds = []
            for sign in (1.0, -1.0):
                density = base_density + sign * step * direction
                if np.any((density <= 0.0) | (density >= 1.0)):
                    raise RuntimeError("FD direction would clip")
                local_state = build_state(density)
                local_operator, solved, local_power = solve_thermal(
                    local_state, q_density, args.cuda_device
                )
                del local_operator
                local_temperature = cell_to_node(
                    flake_cell_temperature(local_state, solved.solution)
                )
                objectives.append(
                    float(np.sum(electrical.gradient_temperature_K_inv * local_temperature))
                )
                residuals.append(solved.explicit_relative_residual)
                local_energy, _ = boundary_energy_error(
                    local_state, solved.solution, local_power
                )
                energies.append(local_energy)
                seconds.append(solved.solve_seconds)
            fd = (objectives[0] - objectives[1]) / (2.0 * step)
            rows.append(
                {
                    "direction": name,
                    "step": step,
                    "AD_A": ad,
                    "FD_A": fd,
                    "relative_error": relative(ad, fd),
                    "plus_objective_A": objectives[0],
                    "minus_objective_A": objectives[1],
                    "worst_residual": max(residuals),
                    "worst_energy_balance": max(energies),
                    "forward_seconds": seconds,
                    "clipping": False,
                }
            )

    fine_rows = [row for row in rows if np.isclose(row["step"], min(steps))]
    worst_fine_error = max(row["relative_error"] for row in fine_rows)
    worst_residual = max(
        forward.explicit_relative_residual,
        adjoint.explicit_relative_residual,
        max(row["worst_residual"] for row in rows),
    )
    worst_energy = max(energy, max(row["worst_energy_balance"] for row in rows))
    passed = bool(
        worst_fine_error < 0.01
        and worst_residual < 1e-8
        and worst_energy < 0.01
        and mapping["relative_mapping_error"] < 0.005
        and current_identity_error < 1e-8
    )
    raw = output / "tairte4_flake_fixed_Q_thermal_adfd.npz"
    np.savez_compressed(
        raw,
        base_design_density=base_density,
        mapped_Q_W_m3=q_density,
        temperature_active_K=forward.solution,
        thermal_adjoint_active=adjoint.solution,
        gradient_thermal_A=gradient,
        flake_nodal_temperature_K=flake_temperature,
        weighting_potential=electrical.weighting_potential,
        **{f"direction_{name}": all_directions[name] for name in requested},
    )
    result = {
        "status": "VALIDATED_TAIRTE4_FLAKE_FIXED_Q_THERMAL_ADFD" if passed else "FAILED_TAIRTE4_FLAKE_FIXED_Q_THERMAL_ADFD",
        "passed": passed,
        "scope": "fixed native-Q explicit thermal-material/interface physical-density AD-FD; optical and direct electrical-rho gradients are excluded",
        "axis_contract": "Lumerical x=b, y=a, z=c",
        "physical_contract": {
            "kappa_TaIrTe4_xyz_W_mK": K_TAIRTE4_XYZ_W_MK.tolist(),
            "G_TaIrTe4_SiO2_W_m2K": G_TAIRTE4_SIO2_W_M2K,
            "G_SiO2_Si_W_m2K": G_SIO2_SI_W_M2K,
            "top_air_convection_W_m2K": TOP_AIR_CONVECTION_W_M2K,
            "TaIrTe4_air_G_1_reduced_boundary_double_counted": False,
            "bottom_gray_law": "parallel area fraction of TaIrTe4/SiO2 and air/SiO2 paths",
        },
        "shape_xyz": list(state.system.shape),
        "unknowns": int(state.system.matrix_W_K.shape[0]),
        "matrix_nnz": int(state.system.matrix_W_K.nnz),
        "assembly_seconds": assembly_s,
        "native_Q": {"path": str(native_path), "sha256": actual_sha},
        "Q_mapping": mapping,
        "source_power_W": float(np.sum(source_power)),
        "Tmax_rise_K": float(np.max(forward.solution)),
        "base_PTE_current_A": electrical.current_A,
        "fixed_electrical_linear_functional_A": thermal_objective,
        "current_linear_identity_error": current_identity_error,
        "forward": {"iterations": forward.iterations, "residual": forward.explicit_relative_residual, "seconds": forward.solve_seconds},
        "adjoint": {"iterations": adjoint.iterations, "residual": adjoint.explicit_relative_residual, "seconds": adjoint.solve_seconds},
        "energy_balance_error": energy,
        "boundary_power_W": boundary_power,
        "directional_AD_FD": rows,
        "worst_fine_step_relative_error": worst_fine_error,
        "worst_residual": worst_residual,
        "worst_energy_balance": worst_energy,
        "CPU_linear_solve_fallback": False,
        "Q_clipping_smoothing_gain_or_rescaling": False,
        "raw_artifact": {"path": str(raw), "size_bytes": raw.stat().st_size, "sha256": sha256(raw)},
        "Maxwell_solves": 0,
        "optimization_iterations": 0,
    }
    (output / "tairte4_flake_fixed_Q_thermal_adfd.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
