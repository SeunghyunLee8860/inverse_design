#!/usr/bin/env python3
"""Certify the fixed-K thermal/PTE part of the inverse-design chain.

This is a discrete derivative certificate, not a production temperature or
current prediction.  It deliberately uses a finite local readout mask because
the repository does not yet define electrodes or a weighting-potential solve.
The thermal design material above TaIrTe4 is also omitted because the optical
repository defines it only by refractive index, not by thermal properties.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.sparse import linalg as sparse_linalg


HERE = Path(__file__).resolve().parent
PHOTOTHERMAL = HERE.parent
REPOSITORY = PHOTOTHERMAL.parent
FVM = PHOTOTHERMAL / "validation" / "photothermal_stage1"
for path in (HERE, FVM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from anisotropic_heat_fvm import (  # noqa: E402
    assemble_steady_diagonal_kappa,
    solve_assembled_thermal_system,
)
from local_pte_functional import build_local_pte_functional  # noqa: E402


STATUS_PASS = "VALIDATED_FIXED_K_THERMAL_PTE_ADFD"
STATUS_FAIL = "FAILED_FIXED_K_THERMAL_PTE_ADFD"
NUMERICAL_SOURCE_POWER_W = 1.0e-12
G_TOP_W_M2K = 7.37e6
G_BOTTOM_W_M2K = 1.1e9
SEEDS = {
    "transpose": 2026072601,
    "temperature": 2026072602,
    "source_unrestricted": 2026072603,
    "source_positive": 2026072604,
}
T_STEPS = np.asarray([1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
Q_STEPS = np.asarray([1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
N_DIRECTIONS = 8


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_error(numerical: float, analytic: float, floor: float) -> float:
    return abs(numerical - analytic) / max(abs(analytic), floor)


def _grid_and_materials():
    # Same lateral period and vertical layer boundaries as the optical bundle.
    x_edges = np.linspace(-3.0e-6, 3.0e-6, 25)
    y_edges = np.linspace(-3.0e-6, 3.0e-6, 25)
    z_edges = np.asarray(
        [
            -2.385e-6,
            -1.885e-6,
            -1.385e-6,
            -0.885e-6,
            -0.385e-6,
            -0.2425e-6,
            -0.1e-6,
            -0.05e-6,
            0.0,
        ]
    )
    shape = (x_edges.size - 1, y_edges.size - 1, z_edges.size - 1)
    z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])
    material = np.empty(shape, dtype=np.int8)
    material[:, :, z_centres < -0.385e-6] = 1
    material[
        :, :, (z_centres >= -0.385e-6) & (z_centres < -0.1e-6)
    ] = 2
    material[:, :, z_centres >= -0.1e-6] = 3
    kappa = np.empty((*shape, 3), float)
    kappa[material == 1] = [145.0, 145.0, 145.0]
    kappa[material == 2] = [1.38, 1.38, 1.38]
    kappa[material == 3] = [14.4, 3.8, 1.0]

    resistance_z = np.zeros((shape[0], shape[1], shape[2] - 1))
    oxide_si_face = int(np.flatnonzero(np.isclose(z_edges, -0.385e-6))[0] - 1)
    flake_oxide_face = int(np.flatnonzero(np.isclose(z_edges, -0.1e-6))[0] - 1)
    resistance_z[:, :, oxide_si_face] = 1.0 / G_BOTTOM_W_M2K
    resistance_z[:, :, flake_oxide_face] = 1.0 / G_TOP_W_M2K
    return (
        x_edges,
        y_edges,
        z_edges,
        material,
        kappa,
        {"z": resistance_z},
        oxide_si_face,
        flake_oxide_face,
    )


def _source_and_mask(x_edges, y_edges, z_edges, material, cell_volume):
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    lateral = np.exp(
        -2.0
        * (
            ((xx - 0.55e-6) / 1.15e-6) ** 2
            + ((yy + 0.35e-6) / 0.85e-6) ** 2
        )
    )
    lateral *= 1.0 + 0.14 * np.cos(2.0 * np.pi * xx / 6.0e-6)
    source = np.zeros(material.shape, float)
    source[material == 3] = np.broadcast_to(
        lateral[:, :, None], material.shape
    )[material == 3]
    unscaled_power = float(np.sum(source * cell_volume))
    source *= NUMERICAL_SOURCE_POWER_W / unscaled_power

    # Numerical local readout window only. It is not an electrode model.
    z = 0.5 * (z_edges[:-1] + z_edges[1:])
    mask = (
        (np.abs(x[:, None, None]) <= 2.0e-6)
        & (np.abs(y[None, :, None]) <= 1.75e-6)
        & (z[None, None, :] >= -0.1e-6)
        & (z[None, None, :] < 0.0)
    )
    return source, mask


def _best_step(rows: list[dict]) -> dict:
    return min(rows, key=lambda item: item["relative_error"])


def _directional_checks(
    *,
    functional,
    matrix,
    source_operator,
    boundary_load,
    source_active,
    temperature_active,
):
    factor = sparse_linalg.splu(matrix.tocsc())
    c_t = functional.temperature_source_A_m_K
    lambda_t = sparse_linalg.spsolve(matrix.T.tocsc(), c_t)
    gradient_q = np.asarray(source_operator.T @ lambda_t).reshape(-1)
    rows: list[dict] = []

    rng = np.random.default_rng(SEEDS["temperature"])
    temperature_scale = max(float(np.linalg.norm(temperature_active)), 1.0)
    for direction_id in range(N_DIRECTIONS):
        direction = rng.normal(size=temperature_active.size)
        direction /= np.linalg.norm(direction)
        delta = temperature_scale * direction
        analytic = float(np.dot(c_t, delta))
        floor = max(abs(analytic) * 1e-12, 1e-30)
        direction_rows = []
        for step in T_STEPS:
            plus = functional.evaluate_active(temperature_active + step * delta)
            minus = functional.evaluate_active(temperature_active - step * delta)
            fd = (plus - minus) / (2.0 * step)
            item = {
                "space": "temperature",
                "family": "unrestricted",
                "direction": direction_id,
                "step": float(step),
                "analytic": analytic,
                "finite_difference": float(fd),
                "absolute_error": abs(float(fd) - analytic),
                "relative_error": _relative_error(float(fd), analytic, floor),
            }
            rows.append(item)
            direction_rows.append(item)

    def evaluate_source(active_q):
        rhs = source_operator @ active_q + boundary_load
        return functional.evaluate_active(factor.solve(rhs))

    for family, seed in (
        ("unrestricted", SEEDS["source_unrestricted"]),
        ("positivity_preserving", SEEDS["source_positive"]),
    ):
        rng = np.random.default_rng(seed)
        for direction_id in range(N_DIRECTIONS):
            if family == "unrestricted":
                direction = rng.normal(size=source_active.size)
                direction /= np.linalg.norm(direction)
                delta = np.linalg.norm(source_active) * direction
            else:
                multiplier = rng.uniform(-0.8, 0.8, size=source_active.size)
                delta = source_active * multiplier
            analytic = float(np.dot(gradient_q, delta))
            floor = max(abs(analytic) * 1e-12, 1e-30)
            direction_rows = []
            for step in Q_STEPS:
                plus_q = source_active + step * delta
                minus_q = source_active - step * delta
                plus = evaluate_source(plus_q)
                minus = evaluate_source(minus_q)
                fd = (plus - minus) / (2.0 * step)
                item = {
                    "space": "source_Q",
                    "family": family,
                    "direction": direction_id,
                    "step": float(step),
                    "analytic": analytic,
                    "finite_difference": float(fd),
                    "absolute_error": abs(float(fd) - analytic),
                    "relative_error": _relative_error(float(fd), analytic, floor),
                    "minimum_Q_plus_W_m3": float(np.min(plus_q)),
                    "minimum_Q_minus_W_m3": float(np.min(minus_q)),
                }
                rows.append(item)
                direction_rows.append(item)

    transpose_rng = np.random.default_rng(SEEDS["transpose"])
    transpose_errors = []
    for _ in range(N_DIRECTIONS):
        q_test = transpose_rng.normal(size=source_active.size)
        t_test = transpose_rng.normal(size=source_active.size)
        left = float(np.dot(t_test, source_operator @ q_test))
        right = float(np.dot(source_operator.T @ t_test, q_test))
        transpose_errors.append(
            abs(left - right) / max(abs(left), abs(right), 1e-300)
        )
    return lambda_t, gradient_q, rows, transpose_errors


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _report(summary: dict) -> str:
    gates = summary["gates"]
    return f"""# Fixed-K thermal/PTE AD–FD certificate

**Status: `{summary['status']}`**

This certificate validates the discrete
`Q -> K_T^-1 -> temperature -> local PTE functional` chain. It is **not** a
final device prediction. The top optical design material has no thermal
material law in the repository, and the finite readout mask is a numerical
surrogate rather than an electrode/weighting-potential model.

## Frozen numerical contract

- Lateral cell: 6 um x 6 um, periodic x/y.
- Layers: 2 um Si, 285 nm SiO2, 100 nm TaIrTe4.
- Bottom: fixed DeltaT=0; top: adiabatic.
- TaIrTe4 kappa: diag(14.4, 3.8, 1.0) W/(m K).
- SiO2 kappa: 1.38 W/(m K); Si kappa: 145 W/(m K).
- Named interface scenario: G_top=7.37e6 and G_bottom=1.1e9 W/(m2 K).
- Synthetic positive asymmetric source power: {summary['forward']['source_power_W']:.16e} W.
- Grid: {summary['grid']['shape']} cells; dx=dy=250 nm; nonuniform z.
- PTE functional unit: A m. Cell volume is applied exactly once.

## Discrete adjoint

`K_T theta = M_V Q + b`,
`K_T^T lambda = c_T`, and
`dF/dQ = M_V^T lambda`.

The adjoint source is the literal transpose of the same sparse local-PTE
functional used forward:
`c_T = -(sigma_a S_a D_x^T + sigma_b S_b D_y^T) V/sqrt(2)`.

## Gates

| Gate | Value | Limit | Pass |
|---|---:|---:|---|
| Linear residual | {gates['linear_residual_relative']:.6e} | 1e-8 | {gates['linear_residual_pass']} |
| Energy balance | {gates['energy_balance_relative_error']:.6e} | 1% | {gates['energy_balance_pass']} |
| Matrix asymmetry | {gates['matrix_asymmetry_relative']:.6e} | 1e-13 | {gates['matrix_symmetry_pass']} |
| Minimum eigenvalue | {gates['minimum_eigenvalue_W_K']:.6e} W/K | >0 | {gates['positive_definite_pass']} |
| Temperature AD-FD, worst best-step | {gates['temperature_worst_best_relative_error']:.6e} | 1e-6 | {gates['temperature_adfd_pass']} |
| Q AD-FD, worst best-step | {gates['source_worst_best_relative_error']:.6e} | 1e-6 | {gates['source_adfd_pass']} |
| Volume transpose identity | {gates['volume_transpose_worst_relative_error']:.6e} | 1e-13 | {gates['volume_transpose_pass']} |

## Physical blockers retained

- `BLOCKED_FULL_RHO_DEPENDENT_THERMAL_MATERIAL_MODEL`: no kappa(rho),
  interface-G(rho), or thermal topology interpolation is specified.
- `BLOCKED_PHYSICAL_WEIGHTING_POTENTIAL_OR_FINITE_FLAKE_MASK`: the optical
  model is periodic, while a nonzero device PTE current needs finite contacts,
  a finite flake readout region, or a solved weighting potential.
- Therefore the omitted term
  `-lambda^T (dK_T/drho) theta` is zero only in this fixed-K certificate, not
  in a future full multiphysics topology derivative.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=f"/tmp/tairte4-pte-adfd-{_utc()}",
        help="External directory for raw NPZ; it is not committed.",
    )
    parser.add_argument(
        "--report-dir",
        default=str(
            PHOTOTHERMAL / "reports" / "inverse_design_pte_adfd"
        ),
    )
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    (
        x_edges,
        y_edges,
        z_edges,
        material,
        kappa,
        interface,
        oxide_si_face,
        flake_oxide_face,
    ) = _grid_and_materials()
    system = assemble_steady_diagonal_kappa(
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        z_edges_m=z_edges,
        kappa_W_mK=kappa,
        dirichlet_temperature_K={"z_min": 0.0},
        interface_resistance_m2K_W=interface,
        periodic_axes=("x", "y"),
    )
    source, fom_mask = _source_and_mask(
        x_edges, y_edges, z_edges, material, system.cell_volume_m3
    )
    solved = solve_assembled_thermal_system(system, source_W_m3=source)
    theta = solved.temperature_K[system.active_mask]
    functional = build_local_pte_functional(
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        z_edges_m=z_edges,
        active_mask=system.active_mask,
        active_ids=system.active_ids,
        fom_mask=fom_mask,
        periodic_axes=(),
    )
    lambda_t, gradient_q, rows, transpose_errors = _directional_checks(
        functional=functional,
        matrix=system.matrix_W_K,
        source_operator=system.source_volume_operator_m3,
        boundary_load=system.boundary_load_W,
        source_active=source[system.active_mask],
        temperature_active=theta,
    )

    matrix = system.matrix_W_K
    asymmetry = matrix - matrix.T
    matrix_asymmetry = float(
        sparse_linalg.norm(asymmetry)
        / max(sparse_linalg.norm(matrix), np.finfo(float).tiny)
    )
    minimum_eigenvalue = float(
        sparse_linalg.eigsh(
            matrix, k=1, which="SA", return_eigenvectors=False, tol=1e-10
        )[0]
    )
    temperature_best = [
        _best_step(
            [
                row
                for row in rows
                if row["space"] == "temperature"
                and row["direction"] == direction
            ]
        )
        for direction in range(N_DIRECTIONS)
    ]
    source_best = [
        _best_step(
            [
                row
                for row in rows
                if row["space"] == "source_Q"
                and row["family"] == family
                and row["direction"] == direction
            ]
        )
        for family in ("unrestricted", "positivity_preserving")
        for direction in range(N_DIRECTIONS)
    ]
    temperature_worst = max(row["relative_error"] for row in temperature_best)
    source_worst = max(row["relative_error"] for row in source_best)
    gates = {
        "linear_residual_relative": solved.linear_residual_relative,
        "linear_residual_pass": solved.linear_residual_relative < 1e-8,
        "energy_balance_relative_error": solved.energy_balance_relative_error,
        "energy_balance_pass": solved.energy_balance_relative_error < 0.01,
        "matrix_asymmetry_relative": matrix_asymmetry,
        "matrix_symmetry_pass": matrix_asymmetry < 1e-13,
        "minimum_eigenvalue_W_K": minimum_eigenvalue,
        "positive_definite_pass": minimum_eigenvalue > 0.0,
        "temperature_worst_best_relative_error": temperature_worst,
        "temperature_adfd_pass": temperature_worst < 1e-6,
        "source_worst_best_relative_error": source_worst,
        "source_adfd_pass": source_worst < 1e-6,
        "volume_transpose_worst_relative_error": max(transpose_errors),
        "volume_transpose_pass": max(transpose_errors) < 1e-13,
    }
    passed = all(
        value for key, value in gates.items() if key.endswith("_pass")
    )

    raw_npz = output / "fixed_k_thermal_pte_adfd_raw.npz"
    np.savez_compressed(
        raw_npz,
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        z_edges_m=z_edges,
        material_id=material,
        kappa_diagonal_W_mK=kappa,
        interface_resistance_z_m2K_W=interface["z"],
        source_W_m3=source,
        temperature_K=solved.temperature_K,
        fom_mask=fom_mask,
        temperature_adjoint=lambda_t,
        gradient_Q_A_m4_W=gradient_q,
        temperature_source_A_m_K=functional.temperature_source_A_m_K,
    )
    summary = {
        "schema_version": 1,
        "generated_at_utc": _utc(),
        "status": STATUS_PASS if passed else STATUS_FAIL,
        "passed": passed,
        "scope": "fixed-K discrete thermal/PTE derivative certificate",
        "not_a_production_prediction": True,
        "git": {
            "branch": _git("branch", "--show-current"),
            "head_before_generated_reports": _git("rev-parse", "HEAD"),
        },
        "grid": {
            "shape": list(system.shape),
            "x_bounds_m": [float(x_edges[0]), float(x_edges[-1])],
            "y_bounds_m": [float(y_edges[0]), float(y_edges[-1])],
            "z_bounds_m": [float(z_edges[0]), float(z_edges[-1])],
            "x_edges_m": x_edges.tolist(),
            "y_edges_m": y_edges.tolist(),
            "z_edges_m": z_edges.tolist(),
            "periodic_axes": list(system.periodic_axes),
            "bottom_boundary": "Dirichlet DeltaT=0",
            "top_boundary": "adiabatic",
        },
        "materials": {
            "Si_kappa_W_mK": [145.0, 145.0, 145.0],
            "SiO2_kappa_W_mK": [1.38, 1.38, 1.38],
            "TaIrTe4_kappa_W_mK": [14.4, 3.8, 1.0],
            "TaIrTe4_kz_note": "estimated numerical scenario",
            "optical_design_thermal_material": "UNSPECIFIED_AND_OMITTED",
        },
        "interfaces": {
            "SiO2_Si_G_W_m2K": G_BOTTOM_W_M2K,
            "TaIrTe4_SiO2_G_W_m2K": G_TOP_W_M2K,
            "SiO2_Si_z_face_index": oxide_si_face,
            "TaIrTe4_SiO2_z_face_index": flake_oxide_face,
            "face_rule": "dx1/(2*k1)+1/G+dx2/(2*k2)",
        },
        "forward": {
            "source_kind": "synthetic positive asymmetric numerical source",
            "source_power_W": solved.source_power_W,
            "numerical_source_power_W": NUMERICAL_SOURCE_POWER_W,
            "temperature_max_K_per_unit_run": float(
                np.nanmax(solved.temperature_K)
            ),
            "fom_A_m": functional.evaluate_active(theta),
            "linear_solver": solved.solver,
            "iterations": solved.iterations,
            "boundary_power_out_W": solved.boundary_power_out_W,
        },
        "pte": {
            "readout_mask": (
                "finite local numerical window; not electrodes or a solved "
                "weighting potential"
            ),
            "sigma_a_S_m": functional.sigma_a_S_m,
            "sigma_b_S_m": functional.sigma_b_S_m,
            "Seebeck_a_V_K": functional.seebeck_a_V_K,
            "Seebeck_b_V_K": functional.seebeck_b_V_K,
            "functional_unit": "A m",
            "temperature_adjoint_equation": "K_T^T lambda = c_T",
            "Q_gradient_equation": "dF/dQ = M_V^T lambda",
        },
        "directional_derivatives": {
            "seeds": SEEDS,
            "number_of_directions_per_family": N_DIRECTIONS,
            "temperature_steps": T_STEPS.tolist(),
            "source_steps": Q_STEPS.tolist(),
            "temperature_best_rows": temperature_best,
            "source_best_rows": source_best,
        },
        "gates": gates,
        "blockers": [
            "BLOCKED_FULL_RHO_DEPENDENT_THERMAL_MATERIAL_MODEL",
            "BLOCKED_PHYSICAL_WEIGHTING_POTENTIAL_OR_FINITE_FLAKE_MASK",
        ],
        "raw_artifact": {
            "path": str(raw_npz),
            "bytes": raw_npz.stat().st_size,
            "sha256": _sha256(raw_npz),
            "committed_to_git": False,
        },
    }
    summary_path = report_dir / "thermal_pte_adfd_summary.json"
    csv_path = report_dir / "thermal_pte_adfd_cases.csv"
    report_path = report_dir / "THERMAL_PTE_ADFD_REPORT.md"
    manifest_path = report_dir / "RAW_ARTIFACT_MANIFEST.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(csv_path, rows)
    report_path.write_text(_report(summary), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "generation_command": (
            f"{sys.executable} {Path(__file__).resolve()} "
            f"--output-dir {output} --report-dir {report_dir}"
        ),
        "raw_artifacts": [summary["raw_artifact"]],
        "repository_artifacts": [
            {
                "path": str(path.relative_to(REPOSITORY)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (report_path, summary_path, csv_path)
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
