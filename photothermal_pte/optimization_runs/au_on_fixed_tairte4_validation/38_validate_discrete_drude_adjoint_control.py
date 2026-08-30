#!/usr/bin/env python3
"""Validate a discrete dispersive-material adjoint on a fixed 1-D Yee grid.

This is an algorithmic control, not a replacement for the 3-D production
solver.  It demonstrates the derivative structure required by a lossy-metal
topology route: a passive causal Drude pole is interpolated on a *fixed*
discrete grid, the exact discrete Maxwell operator is differentiated, and the
material-loss contribution to the objective is included explicitly.

The control intentionally does not use a moving conformal boundary.  It is the
minimal reproducible distinction between the failed v261 moving-Au
``d-epsilon`` approximation and a solver-discrete dispersive adjoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


C0 = 299_792_458.0
EPS0 = 8.854_187_8128e-12
N_AU = 12.1
K_AU = 69.2
WAVELENGTH_M = 10.0e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class DrudePole:
    epsilon_inf: float
    omega_p: float
    gamma: float

    def epsilon(self, omega: float, strength: np.ndarray | float) -> np.ndarray:
        return self.epsilon_inf - np.asarray(strength) * self.omega_p**2 / (
            omega**2 + 1j * self.gamma * omega
        )


def fit_single_frequency_passive_drude(
    target_epsilon: complex, omega: float, epsilon_inf: float = 1.0
) -> DrudePole:
    """Return a passive one-pole Drude model exact at one frequency.

    With the exp(-i omega t) convention,

        epsilon = epsilon_inf - omega_p^2 / (omega^2 + i gamma omega).

    A passive solution exists here because Re(epsilon) < epsilon_inf and
    Im(epsilon) > 0.
    """

    real_drop = epsilon_inf - float(np.real(target_epsilon))
    loss = float(np.imag(target_epsilon))
    if not (real_drop > 0.0 and loss > 0.0 and omega > 0.0):
        raise ValueError("target does not admit this passive one-pole Drude fit")
    gamma = omega * loss / real_drop
    omega_p_sq = real_drop * (omega**2 + gamma**2)
    return DrudePole(epsilon_inf, float(np.sqrt(omega_p_sq)), float(gamma))


def interpolation(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Causal pole-strength interpolation and its exact derivative.

    The pole strength is non-negative for rho in [0,1], so every gray control
    remains passive.  The cubic law is a numerical scenario, not a claimed Au
    effective-medium law.  Binary endpoints are exact air and exact Au.
    """

    rho = np.asarray(rho, dtype=float)
    return rho**3, 3.0 * rho**2


@dataclass
class DiscreteControl:
    matrix: np.ndarray
    rhs: np.ndarray
    field: np.ndarray
    objective: float
    gradient: np.ndarray
    epsilon_material: np.ndarray
    residual: float
    design_slice: slice
    x_m: np.ndarray


def solve_control(rho: np.ndarray, *, cells: int = 241) -> DiscreteControl:
    if cells < 101 or cells % 2 == 0:
        raise ValueError("cells must be odd and at least 101")
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 1 or np.any((rho < 0.0) | (rho > 1.0)):
        raise ValueError("rho must be a 1-D array within [0,1]")

    omega = 2.0 * np.pi * C0 / WAVELENGTH_M
    target_epsilon = complex(N_AU, K_AU) ** 2
    pole = fit_single_frequency_passive_drude(target_epsilon, omega)

    # Dimensionless x/lambda coordinates keep the Helmholtz matrix well
    # scaled.  Physical dx is retained in the dissipated-power objective.
    length_lambda = 6.0
    x_lambda = np.linspace(-0.5 * length_lambda, 0.5 * length_lambda, cells)
    dx_lambda = float(x_lambda[1] - x_lambda[0])
    dx_m = dx_lambda * WAVELENGTH_M
    x_m = x_lambda * WAVELENGTH_M

    design_count = rho.size
    if design_count >= cells - 20:
        raise ValueError("design vector leaves no absorbing boundary margin")
    start = (cells - design_count) // 2
    design_slice = slice(start, start + design_count)

    strength, dstrength = interpolation(rho)
    epsilon_material = np.ones(cells, dtype=complex)
    epsilon_material[design_slice] = pole.epsilon(omega, strength)
    deps_drho = -dstrength * pole.omega_p**2 / (
        omega**2 + 1j * pole.gamma * omega
    )

    # A fixed lossy sponge makes this a compact open-boundary numerical
    # control.  The sponge is outside the design and independent of rho.
    edge_cells = 36
    sponge = np.zeros(cells)
    ramp = np.linspace(1.0, 0.0, edge_cells, endpoint=False) ** 3
    sponge[:edge_cells] = 10.0 * ramp
    sponge[-edge_cells:] = 10.0 * ramp[::-1]
    epsilon_total = epsilon_material + 1j * sponge

    inv_dx2 = 1.0 / dx_lambda**2
    matrix = np.diag(-2.0 * inv_dx2 + (2.0 * np.pi) ** 2 * epsilon_total)
    matrix += np.diag(np.full(cells - 1, inv_dx2), 1)
    matrix += np.diag(np.full(cells - 1, inv_dx2), -1)

    rhs = np.zeros(cells, dtype=complex)
    source_center = edge_cells + 12
    source_width = 4.0
    grid_index = np.arange(cells)
    rhs[:] = np.exp(-0.5 * ((grid_index - source_center) / source_width) ** 2)
    rhs *= np.exp(1j * 0.37)

    field = np.linalg.solve(matrix, rhs)
    residual = float(
        np.linalg.norm(matrix @ field - rhs) / max(np.linalg.norm(rhs), 1e-300)
    )

    material_loss = np.imag(epsilon_material[design_slice])
    loss_weight = 0.5 * omega * EPS0 * material_loss * dx_m
    field_design = field[design_slice]
    objective = float(np.sum(loss_weight * np.abs(field_design) ** 2))

    objective_field_source = np.zeros(cells, dtype=complex)
    objective_field_source[design_slice] = loss_weight * field_design
    adjoint = np.linalg.solve(matrix.conj().T, objective_field_source)

    operator_term = -2.0 * np.real(
        np.conj(adjoint[design_slice])
        * (2.0 * np.pi) ** 2
        * deps_drho
        * field_design
    )
    direct_loss_term = (
        0.5
        * omega
        * EPS0
        * np.imag(deps_drho)
        * dx_m
        * np.abs(field_design) ** 2
    )
    gradient = operator_term + direct_loss_term

    return DiscreteControl(
        matrix=matrix,
        rhs=rhs,
        field=field,
        objective=objective,
        gradient=gradient,
        epsilon_material=epsilon_material,
        residual=residual,
        design_slice=design_slice,
        x_m=x_m,
    )


def normalized_direction(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=float)
    scale = float(np.max(np.abs(direction)))
    if scale == 0.0:
        raise ValueError("zero direction")
    return direction / scale


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--design-cells", type=int, default=81)
    parser.add_argument("--fd-steps", default="0.005,0.0025,0.00125,0.000625")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    design_cells = int(args.design_cells)
    u = np.linspace(-1.0, 1.0, design_cells)
    rho = 0.50 + 0.10 * np.cos(np.pi * u) + 0.035 * np.sin(3.0 * np.pi * u)
    if np.min(rho) < 0.2 or np.max(rho) > 0.8:
        raise RuntimeError("baseline density lacks clipping-free FD margin")

    directions = {
        "uniform": normalized_direction(np.ones(design_cells)),
        "smooth_asymmetric": normalized_direction(
            np.sin(1.3 * np.pi * u) + 0.35 * np.cos(2.2 * np.pi * u)
        ),
        "central_localized": normalized_direction(np.exp(-(u / 0.18) ** 2)),
        "design_edge_localized": normalized_direction(
            np.exp(-((u + 0.78) / 0.11) ** 2)
        ),
        "fixed_seed_random": normalized_direction(
            np.random.default_rng(20260821).normal(size=design_cells)
        ),
    }
    fd_steps = [float(value) for value in args.fd_steps.split(",")]
    if not fd_steps or any(step <= 0.0 for step in fd_steps):
        raise ValueError("FD steps must be positive")

    baseline = solve_control(rho)
    gradient_norm = float(np.linalg.norm(baseline.gradient))
    rows: list[dict[str, float | str | bool]] = []
    for name, direction in directions.items():
        ad = float(np.dot(baseline.gradient, direction))
        gradient_scale = gradient_norm * float(np.linalg.norm(direction))
        near_null = bool(abs(ad) <= 1.0e-10 * gradient_scale)
        for step in fd_steps:
            plus_rho = rho + step * direction
            minus_rho = rho - step * direction
            clipping_free = bool(
                np.min(minus_rho) > 0.0 and np.max(plus_rho) < 1.0
            )
            if not clipping_free:
                raise RuntimeError(f"{name}, h={step} is not clipping-free")
            plus = solve_control(plus_rho)
            minus = solve_control(minus_rho)
            fd = (plus.objective - minus.objective) / (2.0 * step)
            relative_error = abs(ad - fd) / max(abs(ad), abs(fd), 1e-300)
            gradient_normalized_error = abs(ad - fd) / max(
                gradient_scale, 1e-300
            )
            rows.append(
                {
                    "direction": name,
                    "h": step,
                    "AD": ad,
                    "FD": fd,
                    "relative_error": relative_error,
                    "gradient_scale": gradient_scale,
                    "gradient_normalized_error": gradient_normalized_error,
                    "near_null_direction": near_null,
                    "plus_residual": plus.residual,
                    "minus_residual": minus.residual,
                    "clipping_free": clipping_free,
                }
            )

    omega = 2.0 * np.pi * C0 / WAVELENGTH_M
    endpoint = complex(N_AU, K_AU) ** 2
    pole = fit_single_frequency_passive_drude(endpoint, omega)
    endpoint_readback = complex(pole.epsilon(omega, 1.0))
    endpoint_error = abs(endpoint_readback - endpoint) / abs(endpoint)
    finest = min(fd_steps)
    finest_rows = [row for row in rows if row["h"] == finest]
    strong_finest_rows = [
        row for row in finest_rows if not bool(row["near_null_direction"])
    ]
    if not strong_finest_rows:
        raise RuntimeError("direction set contains no strong validation direction")
    max_strong_finest_relative_error = max(
        float(row["relative_error"]) for row in strong_finest_rows
    )
    max_finest_gradient_normalized_error = max(
        float(row["gradient_normalized_error"]) for row in finest_rows
    )
    max_residual = max(
        baseline.residual,
        *(float(row["plus_residual"]) for row in rows),
        *(float(row["minus_residual"]) for row in rows),
    )
    passed = bool(
        endpoint_error < 1.0e-12
        and max_residual < 1.0e-10
        and max_strong_finest_relative_error < 1.0e-5
        and max_finest_gradient_normalized_error < 1.0e-6
    )
    status = (
        "VALIDATED_DISCRETE_PASSIVE_DRUDE_ADJOINT_CONTROL"
        if passed
        else "FAILED_DISCRETE_PASSIVE_DRUDE_ADJOINT_CONTROL"
    )

    csv_path = output / "au_discrete_drude_adjoint_directions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    figure_path = output / "au_discrete_drude_adjoint_fd.png"
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for name in directions:
        selected = [row for row in rows if row["direction"] == name]
        axes[0].loglog(
            [row["h"] for row in selected],
            [row["gradient_normalized_error"] for row in selected],
            marker="o",
            label=name,
        )
    axes[0].axhline(1.0e-6, color="k", linestyle="--", label="gate")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("central-FD half-step h")
    axes[0].set_ylabel("|AD-FD| / (||gradient|| ||direction||)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)

    x_design_um = baseline.x_m[baseline.design_slice] * 1.0e6
    axes[1].plot(x_design_um, rho, label="rho")
    axes[1].plot(
        x_design_um,
        np.abs(baseline.field[baseline.design_slice])
        / np.max(np.abs(baseline.field[baseline.design_slice])),
        label="|E| / max_design|E|",
    )
    axes[1].set_xlabel("x (um)")
    axes[1].set_ylabel("normalized value")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.suptitle("Fixed-grid passive-Drude discrete adjoint control")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    summary = {
        "status": status,
        "scope": (
            "one-dimensional fixed-grid algorithmic control; not a 3-D "
            "Lumerical or production PTE result"
        ),
        "target": {
            "wavelength_m": WAVELENGTH_M,
            "n": N_AU,
            "k": K_AU,
            "epsilon": [endpoint.real, endpoint.imag],
        },
        "passive_Drude_fit": {
            "epsilon_inf": pole.epsilon_inf,
            "omega_p_rad_s": pole.omega_p,
            "gamma_rad_s": pole.gamma,
            "endpoint_readback": [endpoint_readback.real, endpoint_readback.imag],
            "endpoint_relative_error": endpoint_error,
            "passivity": bool(pole.omega_p > 0.0 and pole.gamma > 0.0),
        },
        "density_interpolation": {
            "law": "Drude pole strength s(rho)=rho^3",
            "binary_endpoints_exact": True,
            "gray_law_status": "numerical causal scenario, not an Au effective medium",
        },
        "discrete_control": {
            "design_cells": design_cells,
            "baseline_objective": baseline.objective,
            "baseline_residual": baseline.residual,
            "maximum_residual": max_residual,
            "FD_steps": fd_steps,
            "directions": list(directions),
            "finest_step": finest,
            "maximum_strong_direction_finest_step_relative_error": (
                max_strong_finest_relative_error
            ),
            "maximum_all_direction_gradient_normalized_error": (
                max_finest_gradient_normalized_error
            ),
            "strong_direction_relative_AD_FD_gate": 1.0e-5,
            "all_direction_gradient_normalized_gate": 1.0e-6,
            "near_null_rule": "|AD| <= 1e-10 ||gradient||_2 ||direction||_2",
        },
        "interpretation": (
            "A passive dispersive metal on a fixed discrete grid has an exact "
            "adjoint when both the Maxwell-operator and direct material-loss "
            "derivatives are included. Extending this control to production "
            "requires the 3-D Yee curl/PML operator and Drude/CCPR auxiliary "
            "states; it does not validate v261's moving-conformal-Au d-epsilon."
        ),
    }
    summary_path = output / "au_discrete_drude_adjoint_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report_path = output / "AU_DISCRETE_DRUDE_ADJOINT_CONTROL_REPORT.md"
    report_path.write_text(
        f"""# Discrete passive-Drude adjoint control

Status: `{status}`

This is a one-dimensional fixed-grid algorithmic control, not a 3-D
production result.  The exact 10-um Au endpoint
`epsilon={endpoint.real:.6f}+{endpoint.imag:.6f}i` is represented by a passive
one-pole Drude model with positive `omega_p` and `gamma`.  Density changes the
pole strength through `s(rho)=rho^3`; it does not move a conformal CAD
boundary.

The discrete adjoint includes both terms:

```text
-2 Re[lambda^H (dA/drho) E] + E^H (dW_loss/drho) E
```

The baseline linear residual is `{baseline.residual:.3e}` and the largest
residual over all central-FD solves is `{max_residual:.3e}`.  At the finest
step `h={finest:g}`, the largest relative AD--FD error over the strong
directions is `{100.0*max_strong_finest_relative_error:.6g}%`.  The
central-localized direction is near-null; using the common gradient scale, the
largest normalized error over all five directions is
`{max_finest_gradient_normalized_error:.3e}`.  A cancellation-dominated
near-null FD is not mislabeled as a 100% physical-gradient error.

This pass proves the required mathematical repair: optimize causal dispersive
state parameters on a fixed discrete Maxwell operator and differentiate the
same operator.  It does **not** make the failed v261 moving-Au boundary
gradient valid.  A production implementation still requires 3-D Yee/PML and
Drude/CCPR auxiliary-state AD--FD certification, followed by exact-binary
Lumerical endpoint cross-validation.
""",
        encoding="utf-8",
    )

    manifest_path = output / "AU_DISCRETE_DRUDE_ADJOINT_MANIFEST.json"
    artifacts = []
    for role, path in (
        ("summary_json", summary_path),
        ("directional_csv", csv_path),
        ("plot", figure_path),
        ("report", report_path),
    ):
        artifacts.append(
            {
                "role": role,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "status": status,
        "generation_command": (
            "python 38_validate_discrete_drude_adjoint_control.py"
        ),
        "artifacts": artifacts,
        "no_Lumerical_Maxwell_solve": True,
        "no_thermal_PTE_or_optimization": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
