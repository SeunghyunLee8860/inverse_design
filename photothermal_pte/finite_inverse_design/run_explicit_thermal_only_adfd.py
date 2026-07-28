#!/usr/bin/env python3
"""Certify the fixed-Q explicit thermal-material/interface rho gradient."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

import numpy as np

from .contract import (
    G_SIO2_SI_W_M2K,
    G_TAIRTE4_AIR_W_M2K,
    G_TAIRTE4_BOTTOM_SIO2_W_M2K,
    G_TAIRTE4_DEPOSITED_SIO2_W_M2K,
    H_SIO2_AIR_W_M2K,
    KAPPA_AIR_W_MK,
    KAPPA_SI_W_MK,
    KAPPA_SIO2_W_MK,
    KAPPA_TAIRTE4_W_MK,
)
from .explicit_thermal import (
    evaluate_explicit_thermal,
    solve_explicit_forward,
)


STATUS_PASS = "VALIDATED_FINITE_EXPLICIT_THERMAL_ONLY_ADFD"
STATUS_FAIL = "FAILED_FINITE_EXPLICIT_THERMAL_ONLY_ADFD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--cell-size-nm", type=float, default=100.0)
    parser.add_argument("--lateral-domain-um", type=float, default=32.0)
    parser.add_argument("--si-depth-um", type=float, default=20.0)
    parser.add_argument("--flake-span-um", type=float, default=4.0)
    parser.add_argument(
        "--steps",
        default="0.01,0.005,0.0025,0.00125",
    )
    parser.add_argument(
        "--directions",
        default="uniform,sinusoidal,center_edge,seeded_random",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directions(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    x = np.linspace(-1.0, 1.0, shape[0])[:, None]
    y = np.linspace(-1.0, 1.0, shape[1])[None, :]
    radius = np.sqrt(x**2 + y**2)
    rng = np.random.default_rng(2026072701)
    raw = {
        "uniform": np.ones(shape),
        "sinusoidal": np.sin(np.pi * x) * np.cos(0.5 * np.pi * y),
        "center_edge": np.where(radius <= 0.45, 1.0, -0.35),
        "seeded_random": rng.normal(size=shape),
    }
    result = {}
    for name, value in raw.items():
        value = np.asarray(value, float)
        value /= np.max(np.abs(value))
        result[name] = value
    return result


def base_density(shape: tuple[int, int]) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, shape[0])[:, None]
    y = np.linspace(-1.0, 1.0, shape[1])[None, :]
    return 0.5 + 0.06 * np.cos(np.pi * x) * np.cos(np.pi * y)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    steps = tuple(float(item) for item in args.steps.split(","))
    requested_directions = tuple(
        item.strip() for item in args.directions.split(",") if item.strip()
    )
    cell_size_m = args.cell_size_nm * 1.0e-9
    design_cells = int(round(2.0e-6 / cell_size_m))
    rho = base_density((design_cells, design_cells))
    all_directions = directions(rho.shape)
    unknown = set(requested_directions) - set(all_directions)
    if unknown:
        raise ValueError(f"unknown directions: {sorted(unknown)}")
    kwargs = {
        "lateral_domain_m": args.lateral_domain_um * 1.0e-6,
        "si_depth_m": args.si_depth_um * 1.0e-6,
        "flake_span_m": args.flake_span_um * 1.0e-6,
        "cell_size_m": cell_size_m,
    }
    print("BASE_ADJOINT_START", flush=True)
    start = time.perf_counter()
    base = evaluate_explicit_thermal(rho=rho, **kwargs)
    base_elapsed = time.perf_counter() - start
    print(f"BASE_ADJOINT_DONE {base_elapsed:.3f}s", flush=True)
    source = base.source_W_m3.copy()
    cases: list[dict] = []
    for name in requested_directions:
        direction = all_directions[name]
        analytic = float(np.sum(base.gradient_rho_A * direction))
        component_directionals = {
            "bulk_k": float(np.sum(base.gradient_bulk_k_A * direction)),
            "interface_G": float(
                np.sum(base.gradient_interface_G_A * direction)
            ),
            "top_convection_k": float(
                np.sum(base.gradient_top_convection_k_A * direction)
            ),
        }
        for step in steps:
            print(f"FD_START {name} h={step}", flush=True)
            case_start = time.perf_counter()
            plus = solve_explicit_forward(
                rho=rho + step * direction,
                source_W_m3=source,
                **kwargs,
            )
            minus = solve_explicit_forward(
                rho=rho - step * direction,
                source_W_m3=source,
                **kwargs,
            )
            finite_difference = (
                plus.objective_A - minus.objective_A
            ) / (2.0 * step)
            relative_error = abs(finite_difference - analytic) / max(
                abs(finite_difference),
                abs(analytic),
                np.finfo(float).tiny,
            )
            case = {
                "direction": name,
                "step": step,
                "adjoint_directional_A": analytic,
                "finite_difference_directional_A": finite_difference,
                "relative_error": relative_error,
                "bulk_k_directional_A": component_directionals["bulk_k"],
                "interface_G_directional_A": component_directionals[
                    "interface_G"
                ],
                "top_convection_k_directional_A": component_directionals[
                    "top_convection_k"
                ],
                "plus_objective_A": plus.objective_A,
                "minus_objective_A": minus.objective_A,
                "plus_forward_residual": (
                    plus.solved.linear_residual_relative
                ),
                "minus_forward_residual": (
                    minus.solved.linear_residual_relative
                ),
                "plus_energy_error": (
                    plus.solved.energy_balance_relative_error
                ),
                "minus_energy_error": (
                    minus.solved.energy_balance_relative_error
                ),
                "elapsed_s": time.perf_counter() - case_start,
            }
            cases.append(case)
            print(
                "FD_DONE "
                f"{name} h={step} error={relative_error:.6e} "
                f"elapsed={case['elapsed_s']:.3f}s",
                flush=True,
            )

    primary_step = 0.005
    primary = [
        case for case in cases if np.isclose(case["step"], primary_step)
    ]
    best_by_direction = {
        name: min(
            case["relative_error"]
            for case in cases
            if case["direction"] == name
        )
        for name in requested_directions
    }
    all_linear_residuals = [
        base.solved.linear_residual_relative,
        base.adjoint_linear_residual_relative,
        *[
            value
            for case in cases
            for value in (
                case["plus_forward_residual"],
                case["minus_forward_residual"],
            )
        ],
    ]
    all_energy_errors = [
        base.solved.energy_balance_relative_error,
        *[
            value
            for case in cases
            for value in (
                case["plus_energy_error"],
                case["minus_energy_error"],
            )
        ],
    ]
    passed = bool(
        primary
        and max(case["relative_error"] for case in primary) < 0.01
        and max(best_by_direction.values()) < 0.01
        and max(all_linear_residuals) < 1.0e-8
        and max(all_energy_errors) < 0.01
    )

    raw_path = output / "finite_explicit_thermal_only_adfd.npz"
    np.savez_compressed(
        raw_path,
        rho=rho,
        source_W_m3=base.source_W_m3,
        theta_K=base.solved.temperature_K,
        thermal_adjoint=base.system.full_field(base.adjoint_active),
        gradient_rho_A=base.gradient_rho_A,
        gradient_bulk_k_A=base.gradient_bulk_k_A,
        gradient_interface_G_A=base.gradient_interface_G_A,
        gradient_top_convection_k_A=(
            base.gradient_top_convection_k_A
        ),
        x_edges_m=base.geometry.x_edges_m,
        y_edges_m=base.geometry.y_edges_m,
        z_edges_m=base.geometry.z_edges_m,
        material_id=base.geometry.material_id,
        interface_resistance_x_m2K_W=(
            base.geometry.interface_resistance_m2K_W["x"]
        ),
        interface_resistance_y_m2K_W=(
            base.geometry.interface_resistance_m2K_W["y"]
        ),
        interface_resistance_z_m2K_W=(
            base.geometry.interface_resistance_m2K_W["z"]
        ),
    )
    volume = (
        np.diff(base.geometry.x_edges_m)[:, None, None]
        * np.diff(base.geometry.y_edges_m)[None, :, None]
        * np.diff(base.geometry.z_edges_m)[None, None, :]
    )
    source_power = float(np.sum(base.source_W_m3 * volume))
    command = shlex.join([sys.executable, *sys.argv])
    summary = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "status": STATUS_PASS if passed else STATUS_FAIL,
        "passed": passed,
        "scope": (
            "fixed-Q thermal-material/interface physical-density AD-FD "
            "unit certificate; not optical or full latent validation"
        ),
        "solver": (
            "independent conservative Cartesian Python/SciPy FVM; "
            "not Lumerical HEAT"
        ),
        "geometry": {
            "thermal_lateral_domain_m": kwargs["lateral_domain_m"],
            "Si_depth_m": kwargs["si_depth_m"],
            "finite_TaIrTe4_flake_span_m": kwargs["flake_span_m"],
            "TaIrTe4_thickness_m": 100.0e-9,
            "design_span_m": 2.0e-6,
            "design_height_m": 600.0e-9,
            "bottom_SiO2_thickness_m": 285.0e-9,
            "grid_shape": list(base.system.shape),
            "total_cells": int(np.prod(base.system.shape)),
            "AD_certificate_core_cell_size_m": cell_size_m,
            "mesh_note": (
                "mathematical AD-FD grid; physical mesh convergence is a "
                "separate gate"
            ),
            "periodic": False,
        },
        "materials_W_mK": {
            "TaIrTe4_diagonal": KAPPA_TAIRTE4_W_MK.tolist(),
            "SiO2": KAPPA_SIO2_W_MK,
            "Si": KAPPA_SI_W_MK,
            "air": KAPPA_AIR_W_MK,
            "gray_design_law": (
                "k_air + rho*(k_SiO2-k_air), applied to x/y/z"
            ),
        },
        "interfaces_W_m2K": {
            "TaIrTe4_bottom_thermally_grown_SiO2": (
                G_TAIRTE4_BOTTOM_SIO2_W_M2K
            ),
            "TaIrTe4_exposed_air": G_TAIRTE4_AIR_W_M2K,
            "TaIrTe4_design_deposited_SiO2_endpoint": (
                G_TAIRTE4_DEPOSITED_SIO2_W_M2K
            ),
            "gray_top_law": (
                "G_air + rho*(G_deposited_SiO2-G_air)"
            ),
            "bottom_SiO2_Si_candidate": G_SIO2_SI_W_M2K,
        },
        "boundary_conditions": {
            "far_x_min_x_max_y_min_y_max": (
                "fixed DeltaT=0 K numerical truncation reservoirs"
            ),
            "bottom_Si": "fixed DeltaT=0 K",
            "top_exposed_surface": (
                f"Robin h={H_SIO2_AIR_W_M2K} W/(m2 K), "
                "ambient DeltaT=0 K"
            ),
            "flake_sidewalls": (
                f"explicit TaIrTe4/air interface G="
                f"{G_TAIRTE4_AIR_W_M2K} W/(m2 K)"
            ),
            "unlisted_adiabatic_physical_faces": False,
        },
        "source": {
            "kind": (
                "deterministic positive asymmetric fixed-Q thermal control"
            ),
            "is_optical_artifact": False,
            "total_power_W": source_power,
            "changed_during_FD": False,
            "normalization_note": (
                "control definition only; no optical-Q promotion or "
                "post-solve gain"
            ),
        },
        "base": {
            "objective_A": base.objective_A,
            "DeltaT_max_K": float(np.nanmax(base.solved.temperature_K)),
            "forward_linear_residual_relative": (
                base.solved.linear_residual_relative
            ),
            "adjoint_linear_residual_relative": (
                base.adjoint_linear_residual_relative
            ),
            "energy_balance_relative_error": (
                base.solved.energy_balance_relative_error
            ),
            "forward_iterations": base.solved.iterations,
            "adjoint_iterations": base.adjoint_iterations,
            "base_elapsed_s": base_elapsed,
        },
        "gradient_norms_A": {
            "combined": float(np.linalg.norm(base.gradient_rho_A)),
            "bulk_k": float(np.linalg.norm(base.gradient_bulk_k_A)),
            "interface_G": float(
                np.linalg.norm(base.gradient_interface_G_A)
            ),
            "top_convection_k": float(
                np.linalg.norm(base.gradient_top_convection_k_A)
            ),
        },
        "directions": list(requested_directions),
        "steps": list(steps),
        "primary_step": primary_step,
        "worst_primary_relative_error": max(
            case["relative_error"] for case in primary
        ),
        "best_relative_error_by_direction": best_by_direction,
        "worst_linear_residual_relative": max(all_linear_residuals),
        "worst_energy_balance_relative_error": max(all_energy_errors),
        "criteria": {
            "primary_directional_relative_error_lt": 0.01,
            "best_step_each_direction_relative_error_lt": 0.01,
            "linear_residual_relative_lt": 1.0e-8,
            "energy_balance_relative_error_lt": 0.01,
        },
        "raw_artifact": {
            "path": str(raw_path),
            "bytes": raw_path.stat().st_size,
            "sha256": sha256_file(raw_path),
        },
        "generation_command": command,
        "not_executed": [
            "finite-device FDTD",
            "optical adjoint",
            "combined optical-thermal AD-FD",
            "latent filter/projection AD-FD",
            "optimization",
        ],
    }
    summary_path = (
        report_dir / "finite_explicit_thermal_only_adfd_summary.json"
    )
    cases_path = (
        report_dir / "finite_explicit_thermal_only_adfd_cases.csv"
    )
    manifest_path = (
        report_dir
        / "FINITE_EXPLICIT_THERMAL_ONLY_ADFD_RAW_ARTIFACT_MANIFEST.json"
    )
    report_path = (
        report_dir / "FINITE_EXPLICIT_THERMAL_ONLY_ADFD_REPORT.md"
    )
    write_json(summary_path, summary)
    with cases_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "generation_command": command,
            "raw_artifacts_committed_to_git": False,
            "artifacts": [summary["raw_artifact"]],
        },
    )
    rows = "\n".join(
        "| {direction} | {step:.5g} | {adjoint_directional_A:.6e} | "
        "{finite_difference_directional_A:.6e} | {relative_error:.6e} |".format(
            **case
        )
        for case in cases
    )
    report = f"""# Finite explicit thermal-only AD–FD

Status: `{summary['status']}`

This is an exact discrete fixed-Q certificate for the thermal branch only.
It is not a finite-device optical, combined, latent, or optimization result.

## Model

- finite TaIrTe4 flake: {args.flake_span_um:g} µm square × 100 nm;
- design: 2 µm square × 600 nm;
- thermal domain: {args.lateral_domain_um:g} µm square, Si depth
  {args.si_depth_um:g} µm;
- AD-certificate core grid: {args.cell_size_nm:g} nm,
  shape `{base.system.shape}`;
- no periodic boundaries;
- far x/y and bottom: fixed DeltaT=0 K numerical reservoirs;
- top: Robin h={H_SIO2_AIR_W_M2K:g} W/(m2 K);
- flake sidewalls: explicit TaIrTe4/air G={G_TAIRTE4_AIR_W_M2K:g}
  W/(m2 K).

The fixed thermal control source is identical in every FD solve.  The
projected physical density changes both the full 3D design conductivity
`k_air + rho*(k_SiO2-k_air)` and the TaIrTe4/design contact conductance
`G_air + rho*(G_deposited_SiO2-G_air)`.

## Exact adjoint

`K(rho) theta = M_Q Q`, `K(rho)^T lambda = dI_PTE/dtheta`, and

`dI/drho = -lambda^T (dK/drho) theta`.

Every bulk, interface, and top-convection face derivative is formed from the
same two-half-cell series resistance used by the forward matrix.

## Directional AD–FD

| direction | h | adjoint [A] | centered FD [A] | relative error |
|---|---:|---:|---:|---:|
{rows}

Worst primary-step (`h=0.005`) error:
`{summary['worst_primary_relative_error']:.6e}`.

Gradient L2 norms [A]: combined
`{summary['gradient_norms_A']['combined']:.6e}`, bulk-k
`{summary['gradient_norms_A']['bulk_k']:.6e}`, interface-G
`{summary['gradient_norms_A']['interface_G']:.6e}`, and top-convection-k
`{summary['gradient_norms_A']['top_convection_k']:.6e}`.

Worst linear residual: `{summary['worst_linear_residual_relative']:.6e}`.
Worst energy-balance error:
`{summary['worst_energy_balance_relative_error']:.6e}`.

## Scope boundary

The 100 nm grid certifies differentiation, not physical mesh convergence.
The source is a named synthetic fixed-Q thermal control, not a promoted
optical artifact.  Optical-only, combined physical-rho, and full latent
AD–FD remain required before the complete gradient can be called validated.
"""
    report_path.write_text(report)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
