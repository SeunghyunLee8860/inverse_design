#!/usr/bin/env python3
"""Fixed-local-Q explicit thermal-material AD--FD numerical scenarios."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.sparse import linalg as sparse_linalg

from .contract import (
    DESIGN_BOUNDS_M,
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
    _face_before,
    _rho_conductance_gradient,
    build_explicit_geometry,
    solve_explicit_forward,
)


STATUS_PASS = "VALIDATED_NAMED_LOCAL_Q_EXPLICIT_THERMAL_ADFD_SCENARIOS"
STATUS_FAIL = "FAILED_NAMED_LOCAL_Q_EXPLICIT_THERMAL_ADFD_SCENARIOS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--scenario",
        action="append",
        required=True,
        help="Named scenario as FLakeSpanUm:local_q_thermal_mapping.npz",
    )
    parser.add_argument("--cell-size-nm", type=float, default=100.0)
    parser.add_argument("--lateral-domain-um", type=float, default=32.0)
    parser.add_argument("--si-depth-um", type=float, default=20.0)
    parser.add_argument("--steps", default="0.01,0.005")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_scenarios(values: list[str]) -> list[tuple[float, Path]]:
    scenarios = []
    for value in values:
        span, separator, raw_path = value.partition(":")
        if not separator:
            raise ValueError(f"invalid scenario {value!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        scenarios.append((float(span), path))
    return scenarios


def base_density(shape: tuple[int, int]) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, shape[0])[:, None]
    y = np.linspace(-1.0, 1.0, shape[1])[None, :]
    return 0.5 + 0.04 * np.cos(np.pi * x) * np.cos(np.pi * y)


def directions(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    x = np.linspace(-1.0, 1.0, shape[0])[:, None]
    y = np.linspace(-1.0, 1.0, shape[1])[None, :]
    rng = np.random.default_rng(2026072704)
    raw = {
        "uniform": np.ones(shape),
        "sinusoidal": np.sin(np.pi * x) * np.cos(0.5 * np.pi * y),
        "seeded_random": rng.normal(size=shape),
    }
    return {
        name: value / np.max(np.abs(value))
        for name, value in raw.items()
    }


def central_flake_average_weights(geometry) -> tuple[np.ndarray, np.ndarray]:
    x = 0.5 * (geometry.x_edges_m[:-1] + geometry.x_edges_m[1:])
    y = 0.5 * (geometry.y_edges_m[:-1] + geometry.y_edges_m[1:])
    central_xy = (
        (x[:, None] > DESIGN_BOUNDS_M["x"][0])
        & (x[:, None] < DESIGN_BOUNDS_M["x"][1])
        & (y[None, :] > DESIGN_BOUNDS_M["y"][0])
        & (y[None, :] < DESIGN_BOUNDS_M["y"][1])
    )
    mask = geometry.flake_mask & central_xy[:, :, None]
    volume = (
        np.diff(geometry.x_edges_m)[:, None, None]
        * np.diff(geometry.y_edges_m)[None, :, None]
        * np.diff(geometry.z_edges_m)[None, None, :]
    )
    selected_volume = float(np.sum(volume[mask]))
    if selected_volume <= 0.0:
        raise RuntimeError("central flake objective mask is empty")
    weights = np.zeros(mask.shape, float)
    weights[mask] = volume[mask] / selected_volume
    return weights, mask


def objective_from_forward(forward) -> tuple[float, np.ndarray, np.ndarray]:
    weights, mask = central_flake_average_weights(forward.geometry)
    value = float(np.sum(weights * forward.solved.temperature_K))
    return value, weights, mask


def evaluate_with_adjoint(
    *,
    rho: np.ndarray,
    source_W_m3: np.ndarray,
    kwargs: dict[str, float],
) -> dict[str, object]:
    forward = solve_explicit_forward(
        rho=rho,
        source_W_m3=source_W_m3,
        **kwargs,
    )
    objective, weights, central_mask = objective_from_forward(forward)
    system = forward.system
    rhs = weights[system.active_mask].reshape(-1)
    matrix = system.matrix_W_K
    preconditioner = sparse_linalg.LinearOperator(
        matrix.shape,
        matvec=lambda vector: vector / system.diagonal_W_K,
    )
    iterations = 0

    def count(_: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    adjoint, info = sparse_linalg.cg(
        matrix.T,
        rhs,
        rtol=1.0e-11,
        atol=0.0,
        maxiter=12000,
        M=preconditioner,
        callback=count,
    )
    if info != 0:
        raise RuntimeError(f"thermal adjoint CG failed with info={info}")
    residual = matrix.T @ adjoint - rhs
    residual_relative = float(
        np.linalg.norm(residual)
        / max(np.linalg.norm(rhs), np.finfo(float).tiny)
    )
    theta = forward.solved.temperature_K
    adjoint_full = system.full_field(adjoint)
    gradients = _rho_conductance_gradient(
        geometry=forward.geometry,
        theta=theta,
        adjoint=adjoint_full,
    )
    flake_temperature = theta[forward.geometry.flake_mask]
    x = 0.5 * (
        forward.geometry.x_edges_m[:-1]
        + forward.geometry.x_edges_m[1:]
    )
    y = 0.5 * (
        forward.geometry.y_edges_m[:-1]
        + forward.geometry.y_edges_m[1:]
    )
    z = 0.5 * (
        forward.geometry.z_edges_m[:-1]
        + forward.geometry.z_edges_m[1:]
    )
    hotspot = np.unravel_index(int(np.nanargmax(theta)), theta.shape)
    flake_hotspot = np.argwhere(forward.geometry.flake_mask)[
        int(np.argmax(flake_temperature))
    ]
    return {
        "forward": forward,
        "objective_K": objective,
        "objective_weights": weights,
        "objective_mask": central_mask,
        "adjoint": adjoint_full,
        "adjoint_residual_relative": residual_relative,
        "adjoint_iterations": iterations,
        "gradient_total_K": gradients["total"],
        "gradient_bulk_k_K": gradients["bulk_k"],
        "gradient_interface_G_K": gradients["interface_G"],
        "gradient_top_convection_k_K": gradients[
            "top_convection_k"
        ],
        "Tmax_K": float(np.nanmax(theta)),
        "hotspot_m": {
            "x": float(x[hotspot[0]]),
            "y": float(y[hotspot[1]]),
            "z": float(z[hotspot[2]]),
            "material_id": int(
                forward.geometry.material_id[hotspot]
            ),
        },
        "TaIrTe4_Tmax_K": float(np.max(flake_temperature)),
        "TaIrTe4_hotspot_m": {
            "x": float(x[flake_hotspot[0]]),
            "y": float(y[flake_hotspot[1]]),
            "z": float(z[flake_hotspot[2]]),
        },
        "TaIrTe4_volume_average_K": float(np.mean(flake_temperature)),
    }


def interface_diagnostics(forward) -> dict[str, object]:
    geometry = forward.geometry
    theta = forward.solved.temperature_K
    widths = (
        np.diff(geometry.x_edges_m),
        np.diff(geometry.y_edges_m),
        np.diff(geometry.z_edges_m),
    )
    records = {}
    for name, z_m in (
        ("SiO2_Si", -385.0e-9),
        ("TaIrTe4_bottom_SiO2", -100.0e-9),
        ("TaIrTe4_top_design_or_air", 0.0),
    ):
        face = _face_before(geometry.z_edges_m, z_m)
        lower = theta[:, :, face]
        upper = theta[:, :, face + 1]
        resistance = geometry.interface_resistance_m2K_W["z"][
            :, :, face
        ]
        lower_k = geometry.kappa_W_mK[:, :, face, 2]
        upper_k = geometry.kappa_W_mK[:, :, face + 1, 2]
        total_resistance = (
            0.5 * widths[2][face] / lower_k
            + resistance
            + 0.5 * widths[2][face + 1] / upper_k
        )
        flux = (lower - upper) / total_resistance
        area = widths[0][:, None] * widths[1][None, :]
        selected = resistance > 0.0
        record = {
            "face_index": face,
            "nonzero_interface_resistance_face_count": int(
                np.count_nonzero(selected)
            ),
            "signed_plus_z_power_W": float(
                np.sum(area[selected] * flux[selected])
                if np.any(selected)
                else 0.0
            ),
            "absolute_power_W": float(
                np.sum(area[selected] * np.abs(flux[selected]))
                if np.any(selected)
                else 0.0
            ),
            "mean_adjacent_cell_temperature_jump_K": float(
                np.mean(np.abs(lower[selected] - upper[selected]))
                if np.any(selected)
                else np.mean(np.abs(lower - upper))
            ),
            "mean_contact_temperature_jump_K": float(
                np.mean(np.abs(flux[selected]) * resistance[selected])
                if np.any(selected)
                else 0.0
            ),
            "interface_area_m2": float(np.sum(area[selected])),
            "minimum_G_W_m2K": float(
                np.min(1.0 / resistance[selected])
                if np.any(selected)
                else np.inf
            ),
            "maximum_G_W_m2K": float(
                np.max(1.0 / resistance[selected])
                if np.any(selected)
                else np.inf
            ),
        }
        if name == "TaIrTe4_top_design_or_air":
            conductance = np.zeros_like(resistance)
            conductance[selected] = 1.0 / resistance[selected]
            subinterfaces = {}
            for label, subset in (
                (
                    "gray_design_contact",
                    selected
                    & (conductance > G_TAIRTE4_AIR_W_M2K),
                ),
                (
                    "air_contact",
                    selected
                    & np.isclose(
                        conductance,
                        G_TAIRTE4_AIR_W_M2K,
                        rtol=0.0,
                        atol=1.0e-12,
                    ),
                ),
            ):
                subinterfaces[label] = {
                    "face_count": int(np.count_nonzero(subset)),
                    "area_m2": float(np.sum(area[subset])),
                    "signed_plus_z_power_W": float(
                        np.sum(area[subset] * flux[subset])
                    ),
                    "absolute_power_W": float(
                        np.sum(area[subset] * np.abs(flux[subset]))
                    ),
                    "mean_contact_temperature_jump_K": float(
                        np.mean(
                            np.abs(flux[subset])
                            * resistance[subset]
                        )
                        if np.any(subset)
                        else 0.0
                    ),
                    "minimum_G_W_m2K": float(
                        np.min(conductance[subset])
                        if np.any(subset)
                        else np.inf
                    ),
                    "maximum_G_W_m2K": float(
                        np.max(conductance[subset])
                        if np.any(subset)
                        else np.inf
                    ),
                }
            record["subinterfaces"] = subinterfaces
        records[name] = record
    return records


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "local_q_explicit_thermal_adfd_summary.json"
    csv_path = output / "local_q_explicit_thermal_adfd_cases.csv"
    result: dict[str, object] = {
        "status": "BLOCKED_NAMED_LOCAL_Q_EXPLICIT_THERMAL_ADFD_NOT_RUN",
        "passed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "4 um and 6 um named numerical scenarios using only local "
            "Omega_Q; not final physical plane-wave heating or PTE"
        ),
        "pte_run": False,
        "optimization_run": False,
        "transient_run": False,
    }
    started = time.monotonic()
    try:
        steps = [float(value) for value in args.steps.split(",")]
        cell_size_m = args.cell_size_nm * 1.0e-9
        design_cells = int(round(2.0e-6 / cell_size_m))
        rho = base_density((design_cells, design_cells))
        all_directions = directions(rho.shape)
        cases = []
        scenario_results = []
        raw_arrays = {}
        for span_um, mapping_path in parse_scenarios(args.scenario):
            kwargs = {
                "lateral_domain_m": args.lateral_domain_um * 1.0e-6,
                "si_depth_m": args.si_depth_um * 1.0e-6,
                "flake_span_m": span_um * 1.0e-6,
                "cell_size_m": cell_size_m,
            }
            mapping = np.load(mapping_path, allow_pickle=False)
            source = np.asarray(mapping["Q_thermal_W_m3"], float)
            geometry = build_explicit_geometry(rho, **kwargs)
            for name, expected, actual in (
                ("x", geometry.x_edges_m, mapping["thermal_x_edges_m"]),
                ("y", geometry.y_edges_m, mapping["thermal_y_edges_m"]),
                ("z", geometry.z_edges_m, mapping["thermal_z_edges_m"]),
            ):
                if not np.array_equal(expected, actual):
                    raise RuntimeError(
                        f"{span_um:g} um mapping/thermal {name} grids differ"
                    )
            volume = (
                np.diff(geometry.x_edges_m)[:, None, None]
                * np.diff(geometry.y_edges_m)[None, :, None]
                * np.diff(geometry.z_edges_m)[None, None, :]
            )
            outside_flake_power = float(
                np.sum(volume[~geometry.flake_mask] * source[
                    ~geometry.flake_mask
                ])
            )
            outside_flake_nonzero = int(
                np.count_nonzero(source[~geometry.flake_mask])
            )
            if outside_flake_power != 0.0 or outside_flake_nonzero != 0:
                raise RuntimeError(
                    f"{span_um:g} um source is not confined to exact "
                    "TaIrTe4 support: "
                    f"{outside_flake_power} W in "
                    f"{outside_flake_nonzero} cells"
                )
            base = evaluate_with_adjoint(
                rho=rho, source_W_m3=source, kwargs=kwargs
            )
            forward = base["forward"]
            mapped_power = float(np.sum(volume * source))
            direction_rows = []
            for direction_name, direction in all_directions.items():
                analytic = float(
                    np.sum(base["gradient_total_K"] * direction)
                )
                components = {
                    "bulk_k": float(
                        np.sum(base["gradient_bulk_k_K"] * direction)
                    ),
                    "interface_G": float(
                        np.sum(
                            base["gradient_interface_G_K"] * direction
                        )
                    ),
                    "top_convection_k": float(
                        np.sum(
                            base["gradient_top_convection_k_K"]
                            * direction
                        )
                    ),
                }
                for step in steps:
                    case_started = time.monotonic()
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
                    plus_value = objective_from_forward(plus)[0]
                    minus_value = objective_from_forward(minus)[0]
                    finite_difference = (
                        plus_value - minus_value
                    ) / (2.0 * step)
                    relative_error = abs(
                        analytic - finite_difference
                    ) / max(
                        abs(analytic),
                        abs(finite_difference),
                        np.finfo(float).tiny,
                    )
                    row = {
                        "flake_span_um": span_um,
                        "direction": direction_name,
                        "step": step,
                        "adjoint_directional_K": analytic,
                        "finite_difference_directional_K": finite_difference,
                        "relative_error": relative_error,
                        "bulk_k_directional_K": components["bulk_k"],
                        "interface_G_directional_K": components[
                            "interface_G"
                        ],
                        "top_convection_k_directional_K": components[
                            "top_convection_k"
                        ],
                        "plus_energy_error": (
                            plus.solved.energy_balance_relative_error
                        ),
                        "minus_energy_error": (
                            minus.solved.energy_balance_relative_error
                        ),
                        "plus_linear_residual": (
                            plus.solved.linear_residual_relative
                        ),
                        "minus_linear_residual": (
                            minus.solved.linear_residual_relative
                        ),
                        "elapsed_s": time.monotonic() - case_started,
                    }
                    cases.append(row)
                    direction_rows.append(row)
                    print(
                        f"THERMAL_FD span={span_um:g} direction="
                        f"{direction_name} h={step:g} "
                        f"error={relative_error:.6e}",
                        flush=True,
                    )
            scenario_results.append(
                {
                    "flake_span_um": span_um,
                    "mapping_artifact": {
                        "path": str(mapping_path),
                        "byte_size": mapping_path.stat().st_size,
                        "sha256": sha256(mapping_path),
                    },
                    "grid_shape": list(geometry.material_id.shape),
                    "total_cells": int(geometry.material_id.size),
                    "material_cell_counts": {
                        str(int(material)): int(count)
                        for material, count in zip(
                            *np.unique(
                                geometry.material_id,
                                return_counts=True,
                            )
                        )
                    },
                    "source_power_W": mapped_power,
                    "source_support": {
                        "exact_TaIrTe4_only": True,
                        "outside_TaIrTe4_power_W": (
                            outside_flake_power
                        ),
                        "outside_TaIrTe4_nonzero_cell_count": (
                            outside_flake_nonzero
                        ),
                    },
                    "objective_central_flake_average_DeltaT_K": (
                        base["objective_K"]
                    ),
                    "Tmax_DeltaT_K": base["Tmax_K"],
                    "hotspot_m": base["hotspot_m"],
                    "TaIrTe4_Tmax_DeltaT_K": base["TaIrTe4_Tmax_K"],
                    "TaIrTe4_hotspot_m": base["TaIrTe4_hotspot_m"],
                    "TaIrTe4_volume_average_DeltaT_K": (
                        base["TaIrTe4_volume_average_K"]
                    ),
                    "boundary_power_out_W": (
                        forward.solved.boundary_power_out_W
                    ),
                    "boundary_power_fraction_of_source": {
                        boundary: power / mapped_power
                        for boundary, power in (
                            forward.solved.boundary_power_out_W.items()
                        )
                    },
                    "energy_balance_relative_error": (
                        forward.solved.energy_balance_relative_error
                    ),
                    "forward_linear_residual_relative": (
                        forward.solved.linear_residual_relative
                    ),
                    "adjoint_linear_residual_relative": (
                        base["adjoint_residual_relative"]
                    ),
                    "gradient_norms_K": {
                        "total": float(
                            np.linalg.norm(base["gradient_total_K"])
                        ),
                        "bulk_k": float(
                            np.linalg.norm(base["gradient_bulk_k_K"])
                        ),
                        "interface_G": float(
                            np.linalg.norm(
                                base["gradient_interface_G_K"]
                            )
                        ),
                        "top_convection_k": float(
                            np.linalg.norm(
                                base["gradient_top_convection_k_K"]
                            )
                        ),
                    },
                    "interface_diagnostics": interface_diagnostics(
                        forward
                    ),
                    "maximum_AD_FD_relative_error": max(
                        row["relative_error"] for row in direction_rows
                    ),
                }
            )
            key = f"flake_{span_um:g}um".replace(".", "p")
            raw_arrays[f"{key}_rho"] = rho
            raw_arrays[f"{key}_source_W_m3"] = source
            raw_arrays[f"{key}_theta_K"] = (
                forward.solved.temperature_K
            )
            raw_arrays[f"{key}_gradient_total_K"] = (
                base["gradient_total_K"]
            )
            raw_arrays[f"{key}_x_edges_m"] = geometry.x_edges_m
            raw_arrays[f"{key}_y_edges_m"] = geometry.y_edges_m
            raw_arrays[f"{key}_z_edges_m"] = geometry.z_edges_m
            raw_arrays[f"{key}_material_id"] = geometry.material_id

        worst_adfd = max(row["relative_error"] for row in cases)
        worst_energy = max(
            [
                scenario["energy_balance_relative_error"]
                for scenario in scenario_results
            ]
            + [
                value
                for row in cases
                for value in (
                    row["plus_energy_error"],
                    row["minus_energy_error"],
                )
            ]
        )
        worst_residual = max(
            [
                value
                for scenario in scenario_results
                for value in (
                    scenario["forward_linear_residual_relative"],
                    scenario["adjoint_linear_residual_relative"],
                )
            ]
            + [
                value
                for row in cases
                for value in (
                    row["plus_linear_residual"],
                    row["minus_linear_residual"],
                )
            ]
        )
        passed = bool(
            worst_adfd < 0.01
            and worst_energy < 0.01
            and worst_residual < 1.0e-8
        )
        raw_path = output / "local_q_explicit_thermal_adfd_raw.npz"
        np.savez_compressed(raw_path, **raw_arrays)
        result.update(
            {
                "status": STATUS_PASS if passed else STATUS_FAIL,
                "passed": passed,
                "objective": (
                    "volume-average DeltaT in central 2x2 um TaIrTe4; "
                    "not PTE current"
                ),
                "fixed_source_during_FD": True,
                "source_is_complete_physical_plane_wave": False,
                "thermal_density_control": {
                    "formula": (
                        "rho=0.5+0.04*cos(pi*xhat)*cos(pi*yhat)"
                    ),
                    "optical_source_density": (
                        "uniform rho=0.5 from the immutable optical FSP"
                    ),
                    "self_consistent_combined_optical_thermal": False,
                    "purpose": (
                        "thermal-material/interface-only directional "
                        "gradient certificate"
                    ),
                },
                "materials_W_mK": {
                    "TaIrTe4_diagonal": KAPPA_TAIRTE4_W_MK.tolist(),
                    "SiO2": KAPPA_SIO2_W_MK,
                    "Si": KAPPA_SI_W_MK,
                    "air": KAPPA_AIR_W_MK,
                    "gray_design_law": (
                        "k_air+rho*(k_SiO2-k_air)"
                    ),
                },
                "interfaces_W_m2K": {
                    "TaIrTe4_air": G_TAIRTE4_AIR_W_M2K,
                    "TaIrTe4_bottom_SiO2": (
                        G_TAIRTE4_BOTTOM_SIO2_W_M2K
                    ),
                    "TaIrTe4_deposited_design_SiO2": (
                        G_TAIRTE4_DEPOSITED_SIO2_W_M2K
                    ),
                    "SiO2_Si_named_candidate": G_SIO2_SI_W_M2K,
                },
                "boundaries": {
                    "far_x_y": (
                        "fixed DeltaT=0 numerical truncation reservoirs"
                    ),
                    "bottom_Si": "fixed DeltaT=0",
                    "top_exposed": (
                        f"Robin h={H_SIO2_AIR_W_M2K} W/(m2 K)"
                    ),
                    "flake_sidewalls": (
                        "explicit TaIrTe4/air internal interface G=1; "
                        "not adiabatic"
                    ),
                },
                "scenarios": scenario_results,
                "cases": cases,
                "gates": {
                    "worst_AD_FD_relative_error": worst_adfd,
                    "limit_AD_FD": 0.01,
                    "worst_energy_balance_relative_error": worst_energy,
                    "limit_energy_balance": 0.01,
                    "worst_linear_residual_relative": worst_residual,
                    "limit_linear_residual": 1.0e-8,
                },
                "raw_artifact": {
                    "path": str(raw_path),
                    "byte_size": raw_path.stat().st_size,
                    "sha256": sha256(raw_path),
                },
            }
        )
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
            writer.writeheader()
            writer.writerows(cases)
    except Exception as exc:
        result["status"] = "BLOCKED_NAMED_LOCAL_Q_THERMAL_ADFD_EXECUTION"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        result["wall_s"] = time.monotonic() - started
        summary_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    return 0 if result.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
