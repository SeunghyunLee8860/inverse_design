#!/usr/bin/env python3
"""Validate Run-002 nodal mapping and fixed-Q thermal-material AD--FD."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
FVM = REPOSITORY / "photothermal_pte" / "validation" / "photothermal_stage1"
if str(FVM) not in sys.path:
    sys.path.insert(0, str(FVM))

from anisotropic_heat_fvm import assemble_steady_diagonal_kappa  # noqa: E402
from photothermal_pte.optimization_runs.cuda_thermal_adjoint import (  # noqa: E402
    PersistentCudaCSR,
    solve_forward_adjoint_cuda,
)
from photothermal_pte.thermal_adjoint.local_pte_functional import (  # noqa: E402
    build_local_pte_functional,
)
from run_production_cuda_thermal_pte import (  # noqa: E402
    SCENARIOS,
    T_BATH,
    face_before,
    material_specific_exposed,
)
from selected_thermal_density_mapping import (  # noqa: E402
    selected_nodal_to_thermal_cell,
    selected_nodal_to_thermal_cell_transpose,
)


K_AIR = 0.026
K_SIO2 = 1.38


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nodal_to_cell(density: np.ndarray) -> np.ndarray:
    values = np.asarray(density, float)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("nodal density must be a two-dimensional grid")
    return 0.25 * (
        values[:-1, :-1]
        + values[1:, :-1]
        + values[:-1, 1:]
        + values[1:, 1:]
    )


def nodal_to_cell_transpose(cell_values: np.ndarray) -> np.ndarray:
    values = np.asarray(cell_values, float)
    result = np.zeros((values.shape[0] + 1, values.shape[1] + 1), float)
    result[:-1, :-1] += 0.25 * values
    result[1:, :-1] += 0.25 * values
    result[:-1, 1:] += 0.25 * values
    result[1:, 1:] += 0.25 * values
    return result


def relative(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), np.finfo(float).tiny)


def base_density_and_directions(shape: tuple[int, int]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    x = np.linspace(-1.0, 1.0, shape[0])[:, None]
    y = np.linspace(-1.0, 1.0, shape[1])[None, :]
    base = 0.5 + 0.035 * x + 0.025 * np.sin(np.pi * x) * np.cos(0.5 * np.pi * y)
    rng = np.random.default_rng(2026080602)
    raw = {
        "uniform": np.ones(shape),
        "smooth_asymmetric": 0.65 * np.sin(np.pi * x) * np.cos(0.5 * np.pi * y) + 0.35 * x,
        "central_localized": np.exp(-((x + 0.15) ** 2 + (y - 0.1) ** 2) / 0.12),
        "design_edge_localized": np.exp(-((x - 0.82) ** 2 + (y + 0.55) ** 2) / 0.04),
        "fixed_seed_random": rng.normal(size=shape),
    }
    directions = {name: value / np.max(np.abs(value)) for name, value in raw.items()}
    if np.min(base) <= 0.02 or np.max(base) >= 0.98:
        raise RuntimeError("base density has inadequate no-clipping FD margin")
    return base, directions


@dataclass(frozen=True)
class State:
    system: object
    source_power_W: np.ndarray
    c_A_K: np.ndarray
    material: np.ndarray
    kappa: np.ndarray
    rho_id: np.ndarray
    rho_cell: np.ndarray
    interface_resistance: dict[str, np.ndarray]
    active: np.ndarray
    widths: tuple[np.ndarray, np.ndarray, np.ndarray]
    top_face: int
    top_delta_G: float
    gray_exponent: float
    dphi_drho_cell: np.ndarray


def build_state(
    data: np.lib.npyio.NpzFile,
    scenario: str,
    rho_cell: np.ndarray,
    gray_exponent: float = 1.0,
) -> State:
    edges = tuple(np.asarray(data[f"{axis}_edges_m"], float) for axis in "xyz")
    widths = tuple(np.diff(edge) for edge in edges)
    shape = tuple(width.size for width in widths)
    masks = {
        name: np.asarray(data[f"mask_{name}"], bool)
        for name in ("Si", "bottom_SiO2", "physical_TaIrTe4", "design_effective_SiO2")
    }
    material = np.zeros(shape, np.uint8)
    material[masks["Si"]] = 1
    material[masks["bottom_SiO2"]] = 2
    material[masks["physical_TaIrTe4"]] = 3
    material[masks["design_effective_SiO2"]] = 4
    active = material > 0
    selected_x = np.flatnonzero(np.any(masks["design_effective_SiO2"], axis=(1, 2)))
    selected_y = np.flatnonzero(np.any(masks["design_effective_SiO2"], axis=(0, 2)))
    selected_z = np.flatnonzero(np.any(masks["design_effective_SiO2"], axis=(0, 1)))
    if rho_cell.shape != (selected_x.size, selected_y.size):
        raise RuntimeError(
            f"thermal density shape {rho_cell.shape} != {(selected_x.size, selected_y.size)}"
        )
    exact_design = np.zeros(shape, bool)
    exact_design[np.ix_(selected_x, selected_y, selected_z)] = True
    if not np.array_equal(exact_design, masks["design_effective_SiO2"]):
        raise RuntimeError("design mask is not an exact Cartesian extrusion")

    kappa = np.full((*shape, 3), K_AIR, float)
    kappa[masks["Si"]] = 145.0
    kappa[masks["bottom_SiO2"]] = K_SIO2
    kappa[masks["physical_TaIrTe4"]] = [14.4, 3.8, 1.0]
    phi = rho_cell**gray_exponent
    dphi_drho = gray_exponent * rho_cell ** (gray_exponent - 1.0)
    design_k = K_AIR + phi * (K_SIO2 - K_AIR)
    for z_index in selected_z:
        for component in range(3):
            kappa[selected_x[:, None], selected_y[None, :], z_index, component] = design_k

    rho_id = np.full(shape, -1, np.int64)
    ids_2d = np.arange(rho_cell.size, dtype=np.int64).reshape(rho_cell.shape)
    for z_index in selected_z:
        rho_id[np.ix_(selected_x, selected_y, [z_index])] = ids_2d[:, :, None]

    rx = np.zeros((shape[0] - 1, shape[1], shape[2]))
    ry = np.zeros((shape[0], shape[1] - 1, shape[2]))
    rz = np.zeros((shape[0], shape[1], shape[2] - 1))
    si_face = face_before(edges[2], -0.385e-6)
    rz[:, :, si_face] = 1.0 / 1.1e9
    bottom_face = face_before(edges[2], -0.100e-6)
    connected = masks["bottom_SiO2"][:, :, bottom_face] & masks["physical_TaIrTe4"][:, :, bottom_face + 1]
    rz[:, :, bottom_face][connected] = 1.0 / SCENARIOS[scenario][0]
    top_face = face_before(edges[2], 0.0)
    connected = masks["physical_TaIrTe4"][:, :, top_face] & masks["design_effective_SiO2"][:, :, top_face + 1]
    top_g = np.ones((shape[0], shape[1]), float)
    delta_g = SCENARIOS[scenario][1] - 1.0
    top_g[np.ix_(selected_x, selected_y)] = 1.0 + phi * delta_g
    rz[:, :, top_face][connected] = 1.0 / top_g[connected]

    system = assemble_steady_diagonal_kappa(
        x_edges_m=edges[0],
        y_edges_m=edges[1],
        z_edges_m=edges[2],
        kappa_W_mK=kappa,
        active_mask=active,
        interface_resistance_m2K_W={"x": rx, "y": ry, "z": rz},
        dirichlet_temperature_K={
            "x_min": T_BATH,
            "x_max": T_BATH,
            "y_min": T_BATH,
            "y_max": T_BATH,
            "z_min": T_BATH,
        },
    )
    system = material_specific_exposed(system, material, widths)
    pte = build_local_pte_functional(
        x_edges_m=edges[0],
        y_edges_m=edges[1],
        z_edges_m=edges[2],
        active_mask=active,
        active_ids=system.active_ids,
        fom_mask=masks["physical_TaIrTe4"],
    )
    wx = wy = 1.0 / 64.0e-6
    c = -(
        wx
        * pte.sigma_a_S_m
        * pte.seebeck_a_V_K
        * (pte.derivative_x_m_inv.T @ pte.integration_volume_m3)
        + wy
        * pte.sigma_b_S_m
        * pte.seebeck_b_V_K
        * (pte.derivative_y_m_inv.T @ pte.integration_volume_m3)
    )
    source = np.asarray(data["Q_total_W_m3"], float)
    source_active = system.active_source(source)
    source_power = np.asarray(system.source_volume_operator_m3 @ source_active)
    return State(
        system=system,
        source_power_W=source_power,
        c_A_K=np.asarray(c).reshape(-1),
        material=material,
        kappa=kappa,
        rho_id=rho_id,
        rho_cell=np.array(rho_cell, copy=True),
        interface_resistance={"x": rx, "y": ry, "z": rz},
        active=active,
        widths=widths,
        top_face=top_face,
        top_delta_G=delta_g,
        gray_exponent=gray_exponent,
        dphi_drho_cell=dphi_drho,
    )


def thermal_cell_gradient(state: State, theta_active: np.ndarray, adjoint_active: np.ndarray) -> dict[str, np.ndarray]:
    theta = np.zeros(state.active.shape, float)
    adjoint = np.zeros(state.active.shape, float)
    theta[state.active] = theta_active
    adjoint[state.active] = adjoint_active
    bulk = np.zeros(int(np.max(state.rho_id)) + 1, float)
    interface = np.zeros_like(bulk)
    dk = K_SIO2 - K_AIR
    dphi_flat = state.dphi_drho_cell.reshape(-1)
    widths = state.widths

    for axis in range(3):
        if axis == 0:
            left_t, right_t = theta[:-1], theta[1:]
            left_l, right_l = adjoint[:-1], adjoint[1:]
            left_k, right_k = state.kappa[:-1, :, :, 0], state.kappa[1:, :, :, 0]
            left_d, right_d = widths[0][:-1, None, None], widths[0][1:, None, None]
            area = widths[1][None, :, None] * widths[2][None, None, :]
            left_id, right_id = state.rho_id[:-1], state.rho_id[1:]
            valid = state.active[:-1] & state.active[1:]
            resistance = state.interface_resistance["x"]
        elif axis == 1:
            left_t, right_t = theta[:, :-1], theta[:, 1:]
            left_l, right_l = adjoint[:, :-1], adjoint[:, 1:]
            left_k, right_k = state.kappa[:, :-1, :, 1], state.kappa[:, 1:, :, 1]
            left_d, right_d = widths[1][None, :-1, None], widths[1][None, 1:, None]
            area = widths[0][:, None, None] * widths[2][None, None, :]
            left_id, right_id = state.rho_id[:, :-1], state.rho_id[:, 1:]
            valid = state.active[:, :-1] & state.active[:, 1:]
            resistance = state.interface_resistance["y"]
        else:
            left_t, right_t = theta[:, :, :-1], theta[:, :, 1:]
            left_l, right_l = adjoint[:, :, :-1], adjoint[:, :, 1:]
            left_k, right_k = state.kappa[:, :, :-1, 2], state.kappa[:, :, 1:, 2]
            left_d, right_d = widths[2][None, None, :-1], widths[2][None, None, 1:]
            area = widths[0][:, None, None] * widths[1][None, :, None]
            left_id, right_id = state.rho_id[:, :, :-1], state.rho_id[:, :, 1:]
            valid = state.active[:, :, :-1] & state.active[:, :, 1:]
            resistance = state.interface_resistance["z"]

        area = np.broadcast_to(area, left_t.shape)
        total_r = 0.5 * left_d / left_k + resistance + 0.5 * right_d / right_k
        common = -(left_l - right_l) * (left_t - right_t)
        left_selected = valid & (left_id >= 0)
        right_selected = valid & (right_id >= 0)
        left_dk = np.zeros_like(left_k)
        right_dk = np.zeros_like(right_k)
        left_dk[left_selected] = dk * dphi_flat[left_id[left_selected]]
        right_dk[right_selected] = dk * dphi_flat[right_id[right_selected]]
        left_dg = area / total_r**2 * 0.5 * left_d / left_k**2 * left_dk
        right_dg = area / total_r**2 * 0.5 * right_d / right_k**2 * right_dk
        np.add.at(bulk, left_id[left_selected], (common * left_dg)[left_selected])
        np.add.at(bulk, right_id[right_selected], (common * right_dg)[right_selected])

    top = state.top_face
    upper_id = state.rho_id[:, :, top + 1]
    selected = upper_id >= 0
    area = widths[0][:, None] * widths[1][None, :]
    lower_k = state.kappa[:, :, top, 2]
    upper_k = state.kappa[:, :, top + 1, 2]
    rho = np.zeros(upper_id.shape, float)
    rho[selected] = state.rho_cell.reshape(-1)[upper_id[selected]]
    phi = rho**state.gray_exponent
    dphi = state.gray_exponent * rho ** (state.gray_exponent - 1.0)
    g = 1.0 + phi * state.top_delta_G
    resistance = np.zeros_like(g)
    resistance[selected] = 1.0 / g[selected]
    total_r = 0.5 * widths[2][top] / lower_k + resistance + 0.5 * widths[2][top + 1] / upper_k
    dresistance = np.zeros_like(g)
    dresistance[selected] = -state.top_delta_G * dphi[selected] / g[selected] ** 2
    dg = -area / total_r**2 * dresistance
    common = -(adjoint[:, :, top] - adjoint[:, :, top + 1]) * (theta[:, :, top] - theta[:, :, top + 1])
    np.add.at(interface, upper_id[selected], (common * dg)[selected])
    shape = state.rho_cell.shape
    return {"bulk_k": bulk.reshape(shape), "interface_G": interface.reshape(shape), "total": (bulk + interface).reshape(shape)}


def boundary_energy(state: State, theta: np.ndarray) -> float:
    boundary = sum(float(np.sum(g * theta[cell_ids])) for cell_ids, g, _ in state.system.boundary_terms.values())
    source = float(np.sum(state.source_power_W))
    return abs(boundary - source) / max(abs(source), np.finfo(float).tiny)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapped-q", type=Path, required=True)
    parser.add_argument("--mapped-q-sha256", required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, default="grown_grown")
    parser.add_argument("--steps", default="0.01,0.005,0.0025")
    parser.add_argument("--direction", default="smooth_asymmetric")
    parser.add_argument(
        "--density-contract",
        choices=("coarse_201_to_200", "selected_373_to_186"),
        default="coarse_201_to_200",
    )
    parser.add_argument("--gray-exponent", type=float, choices=(1.0, 2.0, 3.0), default=1.0)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("refusing non-empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    q_path = args.mapped_q.expanduser().resolve()
    if sha256(q_path) != args.mapped_q_sha256:
        raise RuntimeError("mapped-Q SHA mismatch")
    data = np.load(q_path)
    design_mask = np.asarray(data["mask_design_effective_SiO2"], bool)
    nx = int(np.count_nonzero(np.any(design_mask, axis=(1, 2))))
    ny = int(np.count_nonzero(np.any(design_mask, axis=(0, 2))))
    if args.density_contract == "selected_373_to_186":
        if (nx, ny) != (186, 186):
            raise RuntimeError(f"selected thermal design shape {(nx, ny)} != (186, 186)")
        nodal_shape = (373, 373)
        density_forward = selected_nodal_to_thermal_cell
        density_transpose = selected_nodal_to_thermal_cell_transpose
    else:
        nodal_shape = (nx + 1, ny + 1)
        density_forward = nodal_to_cell
        density_transpose = nodal_to_cell_transpose
    base, direction_set = base_density_and_directions(nodal_shape)
    if args.direction not in direction_set:
        raise RuntimeError(f"unknown direction {args.direction!r}")
    direction = direction_set[args.direction]

    rng = np.random.default_rng(2026080603)
    mapping_rows = []
    for name, probe in direction_set.items():
        dual = rng.normal(size=(nx, ny))
        left = float(np.sum(density_forward(probe) * dual))
        right = float(np.sum(probe * density_transpose(dual)))
        # The map is exactly linear.  Use the same O(1e-3) physical-density
        # scale as the solver AD-FD checks so subtraction roundoff is not
        # mistaken for a transpose failure.
        step = 2.5e-3
        fd = float(np.sum(dual * (density_forward(base + step * probe) - density_forward(base - step * probe))) / (2 * step))
        mapping_rows.append({"direction": name, "dot_error": relative(left, right), "FD_error": relative(left, fd)})

    rho_cell = density_forward(base)
    started = perf_counter()
    state = build_state(data, args.scenario, rho_cell, args.gray_exponent)
    assembly_seconds = perf_counter() - started
    pair = solve_forward_adjoint_cuda(
        state.system.matrix_W_K,
        state.source_power_W,
        state.c_A_K,
        cuda_device=args.cuda_device,
        relative_tolerance=1e-10,
        max_iterations=30000,
    )
    objective = float(np.dot(state.c_A_K, pair.forward.solution))
    cell_gradient_parts = thermal_cell_gradient(state, pair.forward.solution, pair.adjoint.solution)
    nodal_gradient_parts = {name: density_transpose(value) for name, value in cell_gradient_parts.items()}
    adjoint_directional = float(np.sum(nodal_gradient_parts["total"] * direction))
    steps = [float(value) for value in args.steps.split(",")]
    fd_rows = []
    for step in steps:
        objectives = []
        residuals = []
        energies = []
        solve_seconds = []
        for sign in (1.0, -1.0):
            density = base + sign * step * direction
            if np.min(density) <= 0.0 or np.max(density) >= 1.0:
                raise RuntimeError("FD density would clip")
            local_state = build_state(
                data,
                args.scenario,
                density_forward(density),
                args.gray_exponent,
            )
            operator = PersistentCudaCSR(local_state.system.matrix_W_K, cuda_device=args.cuda_device)
            solved = operator.solve(local_state.source_power_W, relative_tolerance=1e-10, max_iterations=30000)
            objectives.append(float(np.dot(local_state.c_A_K, solved.solution)))
            residuals.append(solved.explicit_relative_residual)
            energies.append(boundary_energy(local_state, solved.solution))
            solve_seconds.append(solved.solve_seconds)
        finite_difference = (objectives[0] - objectives[1]) / (2.0 * step)
        fd_rows.append(
            {
                "step": step,
                "adjoint_directional_A": adjoint_directional,
                "finite_difference_directional_A": finite_difference,
                "relative_error": relative(adjoint_directional, finite_difference),
                "plus_objective_A": objectives[0],
                "minus_objective_A": objectives[1],
                "worst_forward_residual": max(residuals),
                "worst_energy_balance_error": max(energies),
                "forward_solve_seconds": solve_seconds,
                "clipping": False,
            }
        )
    best = min(fd_rows, key=lambda row: row["relative_error"])
    mapping_worst_dot = max(row["dot_error"] for row in mapping_rows)
    mapping_worst_fd = max(row["FD_error"] for row in mapping_rows)
    passed = bool(
        mapping_worst_dot < 1e-12
        and mapping_worst_fd < 1e-8
        and max(pair.forward.explicit_relative_residual, pair.adjoint.explicit_relative_residual) < 1e-8
        and boundary_energy(state, pair.forward.solution) < 0.01
        and best["relative_error"] < 0.01
    )
    raw = output / f"thermal_material_adfd_{args.scenario}.npz"
    np.savez_compressed(
        raw,
        base_nodal_density=base,
        direction=direction,
        gradient_total=nodal_gradient_parts["total"],
        gradient_bulk_k=nodal_gradient_parts["bulk_k"],
        gradient_interface_G=nodal_gradient_parts["interface_G"],
        temperature_active_K=pair.forward.solution,
        adjoint_active=pair.adjoint.solution,
    )
    result = {
        "status": (
            "VALIDATED_SELECTED_FIXED_Q_THERMAL_GRAY_ADFD"
            if passed and args.density_contract == "selected_373_to_186"
            else "VALIDATED_PRODUCTION_FIXED_Q_THERMAL_MATERIAL_ADFD"
            if passed
            else "FAILED_SELECTED_FIXED_Q_THERMAL_GRAY_ADFD"
            if args.density_contract == "selected_373_to_186"
            else "FAILED_PRODUCTION_FIXED_Q_THERMAL_MATERIAL_ADFD"
        ),
        "passed": passed,
        "scenario": args.scenario,
        "scope": "fixed Maxwell Q; phi_p(rho)=rho^p applied consistently to bulk kappa and TaIrTe4/design interface G",
        "density_contract": args.density_contract,
        "gray_exponent": args.gray_exponent,
        "gray_interpretation": "named numerical relaxation, not a measured mixture law or confidence interval",
        "nodal_shape": list(nodal_shape),
        "thermal_cell_shape": [nx, ny],
        "mapping_checks": mapping_rows,
        "mapping_worst_dot_error": mapping_worst_dot,
        "mapping_worst_FD_error": mapping_worst_fd,
        "mapping_gates": {
            "transpose_error_max": 1e-12,
            "linear_mapping_FD_error_max": 1e-8,
        },
        "base_density_range": [float(np.min(base)), float(np.max(base))],
        "direction": args.direction,
        "objective_A": objective,
        "adjoint_directional_A": adjoint_directional,
        "gradient_norms": {name: float(np.linalg.norm(value)) for name, value in nodal_gradient_parts.items()},
        "assembly_seconds": assembly_seconds,
        "forward": {"iterations": pair.forward.iterations, "residual": pair.forward.explicit_relative_residual, "seconds": pair.forward.solve_seconds},
        "adjoint": {"iterations": pair.adjoint.iterations, "residual": pair.adjoint.explicit_relative_residual, "seconds": pair.adjoint.solve_seconds},
        "energy_balance_error": boundary_energy(state, pair.forward.solution),
        "FD_step_sweep": fd_rows,
        "best_FD_relative_error": best["relative_error"],
        "raw_artifact": {"path": str(raw), "size_bytes": raw.stat().st_size, "sha256": sha256(raw)},
        "CPU_linear_solve_fallback": False,
        "Q_clipping_smoothing_gain_or_rescaling": False,
        "optimization_iterations": 0,
    }
    (output / "production_thermal_material_adfd_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
