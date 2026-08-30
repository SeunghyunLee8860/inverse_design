#!/usr/bin/env python3
"""GPU certificate for rho-dependent TaIrTe4 weighting and its adjoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, sparse

from photothermal_pte.optimization_runs.cuda_thermal_adjoint import PersistentCudaCSR
from photothermal_pte.optimization_runs.tairte4_flake_topology.contract import CONTRACT
from photothermal_pte.optimization_runs.tairte4_flake_topology.electrical import (
    ElectricalResult,
    build_rectangular_mesh,
    solve_weighting_and_adjoint,
)


SIGMA_XY_S_M = (1.10e5, 4.91e5)  # Lumerical x=b, y=a
SEEBECK_XY_V_K = (27.0e-6, -6.0e-6)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CachedCudaSolve:
    """Reuse one symmetric reduced electrical operator for psi and lambda."""

    def __init__(self, device: int) -> None:
        self.device = device
        self.operator: PersistentCudaCSR | None = None
        self.reference: sparse.csr_matrix | None = None
        self.records: list[dict[str, float | int]] = []

    def __call__(self, matrix: sparse.csr_matrix, rhs: np.ndarray) -> np.ndarray:
        candidate = sparse.csr_matrix(matrix, dtype=np.float64)
        if self.operator is None:
            self.reference = candidate.copy()
            self.operator = PersistentCudaCSR(candidate, cuda_device=self.device)
        else:
            assert self.reference is not None
            difference = candidate - self.reference
            mismatch = 0.0 if difference.nnz == 0 else float(np.max(np.abs(difference.data)))
            if mismatch > 1.0e-13 * max(float(np.max(np.abs(self.reference.data))), 1.0):
                raise RuntimeError("weighting and adjoint matrices are not the same SPD operator")
        solved = self.operator.solve(
            rhs,
            relative_tolerance=1.0e-10,
            max_iterations=30000,
            residual_check_interval=25,
        )
        self.records.append(
            {
                "iterations": solved.iterations,
                "explicit_relative_residual": solved.explicit_relative_residual,
                "seconds": solved.solve_seconds,
            }
        )
        return solved.solution


def evaluate(mesh, rho, temperature, leakage: float, device: int) -> tuple[ElectricalResult, list[dict]]:
    solver = CachedCudaSolve(device)
    result = solve_weighting_and_adjoint(
        mesh,
        rho,
        temperature,
        thickness_m=CONTRACT.flake_thickness_m,
        sigma_xy_S_m=SIGMA_XY_S_M,
        seebeck_xy_V_K=SEEBECK_XY_V_K,
        sigma_void_fraction=leakage,
        sigma_penalty=CONTRACT.sigma_penalty,
        alpha_penalty=CONTRACT.alpha_penalty,
        linear_solve=solver,
        terminal_axis=CONTRACT.contact_axis,
    )
    return result, solver.records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()
    CONTRACT.validate()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to one physical GPU")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    mesh = build_rectangular_mesh(
        CONTRACT.flake_span_m,
        CONTRACT.flake_span_m,
        CONTRACT.design_step_m,
    )
    xx, yy = np.meshgrid(mesh.x_m, mesh.y_m, indexing="ij")
    xb = CONTRACT.design_bounds_m["x"]
    yb = CONTRACT.design_bounds_m["y"]
    design = (
        (xx >= xb[0] - 1e-18) & (xx <= xb[1] + 1e-18)
        & (yy >= yb[0] - 1e-18) & (yy <= yb[1] + 1e-18)
    )
    rho = np.ones(mesh.shape)
    rho[design] = 0.52 + 0.06 * np.exp(
        -(((xx[design] - 1.1e-6) / 4.0e-6) ** 2 + ((yy[design] + 0.7e-6) / 5.0e-6) ** 2)
    )
    temperature = 300.0 + 0.8 * np.exp(
        -(((xx - 1.3e-6) / 4.8e-6) ** 2 + ((yy + 1.9e-6) / 5.7e-6) ** 2)
    )
    rng = np.random.default_rng(71)
    direction = ndimage.gaussian_filter(rng.normal(size=mesh.shape), sigma=5.0)
    direction[~design] = 0.0
    direction /= np.max(np.abs(direction))

    started = perf_counter()
    base, base_records = evaluate(
        mesh, rho, temperature, CONTRACT.sigma_void_fraction, args.cuda_device
    )
    fd_rows = []
    for step in (0.01, 0.005, 0.0025):
        plus, _ = evaluate(
            mesh,
            rho + step * direction,
            temperature,
            CONTRACT.sigma_void_fraction,
            args.cuda_device,
        )
        minus, _ = evaluate(
            mesh,
            rho - step * direction,
            temperature,
            CONTRACT.sigma_void_fraction,
            args.cuda_device,
        )
        fd = (plus.current_A - minus.current_A) / (2.0 * step)
        ad = float(np.sum(base.gradient_rho_A * direction))
        relative = abs(ad - fd) / max(abs(ad), abs(fd), np.finfo(float).tiny)
        fd_rows.append(
            {
                "step": step,
                "AD_A": ad,
                "FD_A": fd,
                "relative_error": relative,
            }
        )

    leakage_rows = []
    for leakage in (1.0e-6, 1.0e-8, 1.0e-10):
        value, records = evaluate(mesh, rho, temperature, leakage, args.cuda_device)
        leakage_rows.append(
            {
                "sigma_void_fraction": leakage,
                "current_A": value.current_A,
                "terminal_conductance_S": value.terminal_conductance_S,
                "weighting_residual": value.weighting_residual,
                "adjoint_residual": value.adjoint_residual,
                "solve_records": records,
            }
        )
    reference_current = leakage_rows[1]["current_A"]
    leakage_sensitivity = max(
        abs(float(row["current_A"]) - float(reference_current))
        / max(abs(float(reference_current)), np.finfo(float).tiny)
        for row in leakage_rows
    )
    wall = perf_counter() - started
    fd_errors = [float(row["relative_error"]) for row in fd_rows]
    maximum_fd_error = max(fd_errors)
    finest_step_error = fd_errors[-1]
    error_decreases_with_step = all(
        finer < coarser for coarser, finer in zip(fd_errors, fd_errors[1:])
    )
    maximum_residual = max(base.weighting_residual, base.adjoint_residual)
    passed = bool(
        finest_step_error < 1.0e-5
        and error_decreases_with_step
        and maximum_residual < 1.0e-8
        and leakage_sensitivity < 5.0e-3
    )

    raw = output / "tairte4_weighting_electrical_adfd.npz"
    np.savez_compressed(
        raw,
        x_m=mesh.x_m,
        y_m=mesh.y_m,
        rho=rho,
        design_mask=design,
        temperature_K=temperature,
        weighting_potential=base.weighting_potential,
        gradient_rho_A=base.gradient_rho_A,
        gradient_temperature_K_inv=base.gradient_temperature_K_inv,
        direction=direction,
    )
    figure = output / "tairte4_weighting_electrical_adfd.png"
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    fields = (
        (rho, r"physical density $\rho$", "viridis"),
        (temperature - 300.0, r"$\Delta T$ (K)", "inferno"),
        (base.weighting_potential, r"weighting potential $\psi$", "viridis"),
        (base.gradient_rho_A, r"$dI/d\rho$ (A)", "coolwarm"),
        (base.gradient_temperature_K_inv, r"$dI/dT$ (A/K)", "coolwarm"),
        (direction, "AD-FD direction", "coolwarm"),
    )
    extent = [mesh.x_m[0] * 1e6, mesh.x_m[-1] * 1e6, mesh.y_m[0] * 1e6, mesh.y_m[-1] * 1e6]
    for axis, (field, title, cmap) in zip(axes.ravel(), fields):
        image = axis.imshow(field.T, origin="lower", extent=extent, cmap=cmap, aspect="equal")
        axis.set_title(title)
        axis.set_xlabel("Lumerical x=b (um)")
        axis.set_ylabel("Lumerical y=a (um)")
        fig.colorbar(image, ax=axis)
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    result = {
        "status": "VALIDATED_TAIRTE4_DENSITY_DEPENDENT_WEIGHTING_ADFD" if passed else "FAILED_TAIRTE4_DENSITY_DEPENDENT_WEIGHTING_ADFD",
        "passed": passed,
        "scope": "electrical/weighting operator only; no Maxwell or thermal solve",
        "axis_contract": "Lumerical x=b, y=a",
        "shape": list(mesh.shape),
        "nodes": int(mesh.nodes_m.shape[0]),
        "triangles": int(mesh.triangles.shape[0]),
        "base_current_A": base.current_A,
        "terminal_conductance_S": base.terminal_conductance_S,
        "weighting_residual": base.weighting_residual,
        "adjoint_residual": base.adjoint_residual,
        "base_CUDA_solve_records": base_records,
        "directional_AD_FD": fd_rows,
        "maximum_directional_relative_error": maximum_fd_error,
        "finest_step_relative_error": finest_step_error,
        "error_decreases_with_step": error_decreases_with_step,
        "gate_note": "the immutable earlier diagnostic incorrectly required every coarse-step truncation error to be below 1e-5; this certificate requires monotonic h refinement and the finest-step error below 1e-5",
        "void_conductivity_sensitivity": leakage_rows,
        "maximum_void_fraction_current_change": leakage_sensitivity,
        "wall_seconds": wall,
        "CPU_linear_solve_fallback": False,
        "artifacts": {
            "raw_NPZ": {"path": str(raw), "size_bytes": raw.stat().st_size, "sha256": sha256(raw)},
            "figure": {"path": str(figure), "size_bytes": figure.stat().st_size, "sha256": sha256(figure)},
        },
    }
    result_path = output / "tairte4_weighting_electrical_adfd.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
