#!/usr/bin/env python3
"""Compare paper-form Gaussian/Beer-Lambert Q with saved Maxwell edge Q.

The paper does not publish a numerical objective transmission or one exact
11-um spot radius.  This diagnostic therefore uses the named w0=6.5 um
scenario and fixes full-plane absorbed power with the independently validated
11-um TMM absorption.  Beer-Lambert controls only the depth shape.  No
weighting potential or PTE current is evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
THERMAL_SCRIPT = HERE / "run_device_a_explicit_thermal_pte.py"
PERMITTIVITY_PATH = REPOSITORY / "photothermal_pte/bundle/perm_data.txt"
INCIDENT_POWER_W = 285.0e-6
WAVELENGTH_M = 11.0e-6
THICKNESS_M = 130.0e-9
WAIST_M = 6.5e-6
TMM_ABSORPTION = {"a": 0.17673296, "b": 0.26328721}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


thermal = load_module("straight_edge_thermal_module", THERMAL_SCRIPT)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()


def nodal_edges(nodes: np.ndarray) -> np.ndarray:
    values = np.asarray(nodes, float)
    edges = np.empty(values.size + 1)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--polarization", choices=("a", "b"), required=True)
    parser.add_argument("--optical-case-dir", type=Path, required=True)
    parser.add_argument("--lumerical-thermal-dir", type=Path)
    parser.add_argument(
        "--solve-remapped-lumerical",
        action="store_true",
        help="solve the saved/remapped Maxwell Q on the newly built grid",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thermal-domain-um", type=float, default=48.0)
    parser.add_argument("--si-depth-um", type=float, default=20.0)
    parser.add_argument("--core-step-nm", type=float, default=100.0)
    parser.add_argument("--flake-dz-nm", type=float, default=26.0)
    parser.add_argument(
        "--thermal-model",
        choices=("expanded", "paper-reduced"),
        default="expanded",
        help=(
            "expanded production FVM or Supplement Eq. S4 reduced flake-only "
            "Robin control"
        ),
    )
    parser.add_argument(
        "--offsets-um",
        type=float,
        nargs="+",
        default=[0.0],
        help="beam displacement along outward edge normal",
    )
    parser.add_argument("--profile-tangent-window-um", type=float, default=5.0)
    return parser.parse_args()


def build_straight_geometry(args: argparse.Namespace) -> Any:
    outer_um = 0.5 * args.thermal_domain_um + 1.0
    thermal.FLAKE_VERTICES_UM = np.asarray(
        [
            [-outer_um, -outer_um],
            [outer_um, -outer_um],
            [outer_um, outer_um],
        ],
        float,
    )
    return thermal.build_geometry(
        domain_m=args.thermal_domain_um * 1e-6,
        si_depth_m=args.si_depth_um * 1e-6,
        core_step_m=args.core_step_nm * 1e-9,
        flake_dz_m=args.flake_dz_nm * 1e-9,
    )


def assemble_expanded_system(geometry: Any) -> Any:
    return thermal.assemble_steady_diagonal_kappa(
        x_edges_m=geometry.x_edges_m,
        y_edges_m=geometry.y_edges_m,
        z_edges_m=geometry.z_edges_m,
        kappa_W_mK=geometry.kappa_W_mK,
        dirichlet_temperature_K={
            "x_min": 0.0,
            "x_max": 0.0,
            "y_min": 0.0,
            "y_max": 0.0,
            "z_min": 0.0,
        },
        interface_resistance_m2K_W=geometry.interface_resistance_m2K_W,
        active_mask=np.ones(geometry.material_id.shape, bool),
        exposed_heat_transfer_W_m2K=thermal.H_EXPOSED_W_M2K,
        ambient_temperature_K=0.0,
    )


def optical_constants(polarization: str) -> dict[str, Any]:
    data = np.loadtxt(PERMITTIVITY_PATH)
    order = np.argsort(data[:, 0])
    wavelength_nm = data[order, 0]
    if polarization == "a":
        epsilon_samples = data[order, 1] + 1j * data[order, 2]
    else:
        epsilon_samples = data[order, 3] + 1j * data[order, 4]
    target_nm = WAVELENGTH_M * 1e9
    epsilon = complex(
        np.interp(target_nm, wavelength_nm, epsilon_samples.real)
        + 1j * np.interp(target_nm, wavelength_nm, epsilon_samples.imag)
    )
    refractive_index = complex(np.sqrt(epsilon))
    beta = 4.0 * np.pi * refractive_index.imag / WAVELENGTH_M
    absorbed_depth_fraction = 1.0 - np.exp(-beta * THICKNESS_M)
    return {
        "epsilon": {"real": epsilon.real, "imag": epsilon.imag},
        "refractive_index": {
            "real": refractive_index.real,
            "imag": refractive_index.imag,
        },
        "beta_m_inv": beta,
        "penetration_depth_m": 1.0 / beta,
        "absorbed_depth_fraction_over_130nm": absorbed_depth_fraction,
        "effective_entrance_factor_for_TMM_total_absorption": (
            TMM_ABSORPTION[polarization] / absorbed_depth_fraction
        ),
    }


def analytic_q(
    geometry: Any,
    polarization: str,
    offset_m: float,
) -> tuple[np.ndarray, dict[str, float]]:
    x = 0.5 * (geometry.x_edges_m[:-1] + geometry.x_edges_m[1:])
    y = 0.5 * (geometry.y_edges_m[:-1] + geometry.y_edges_m[1:])
    z = 0.5 * (geometry.z_edges_m[:-1] + geometry.z_edges_m[1:])
    constants = optical_constants(polarization)
    beta = constants["beta_m_inv"]

    # n=(y-x)/sqrt(2) points from TaIrTe4 (n<=0) into air.
    center_x = -offset_m / np.sqrt(2.0)
    center_y = offset_m / np.sqrt(2.0)
    lateral = 2.0 / (np.pi * WAIST_M**2) * np.exp(
        -2.0
        * (
            (x[:, None] - center_x) ** 2
            + (y[None, :] - center_y) ** 2
        )
        / WAIST_M**2
    )
    depth = np.zeros_like(z)
    in_depth = (z >= -THICKNESS_M) & (z <= 0.0)
    depth[in_depth] = (
        beta
        * np.exp(-beta * (-z[in_depth]))
        / constants["absorbed_depth_fraction_over_130nm"]
    )
    q = (
        INCIDENT_POWER_W
        * TMM_ABSORPTION[polarization]
        * lateral[:, :, None]
        * depth[None, None, :]
    )
    q = np.where(geometry.flake_mask, q, 0.0)
    volume = (
        np.diff(geometry.x_edges_m)[:, None, None]
        * np.diff(geometry.y_edges_m)[None, :, None]
        * np.diff(geometry.z_edges_m)[None, None, :]
    )
    return q, {
        "beam_offset_normal_m": offset_m,
        "full_plane_absorbed_power_W": (
            INCIDENT_POWER_W * TMM_ABSORPTION[polarization]
        ),
        "finite_half_plane_absorbed_power_W": float(np.sum(q * volume)),
    }


def binned_profile(
    values: np.ndarray,
    x_edges_m: np.ndarray,
    y_edges_m: np.ndarray,
    *,
    bin_width_m: float,
    tangent_window_m: float,
    n_min_m: float = -12e-6,
    n_max_m: float = 3e-6,
) -> tuple[np.ndarray, np.ndarray]:
    x = 0.5 * (x_edges_m[:-1] + x_edges_m[1:])
    y = 0.5 * (y_edges_m[:-1] + y_edges_m[1:])
    normal = (y[None, :] - x[:, None]) / np.sqrt(2.0)
    tangent = (x[:, None] + y[None, :]) / np.sqrt(2.0)
    bins = np.arange(n_min_m, n_max_m + bin_width_m, bin_width_m)
    centers = 0.5 * (bins[:-1] + bins[1:])
    profile = np.full(centers.shape, np.nan)
    window = np.abs(tangent) <= tangent_window_m
    indices = np.digitize(normal[window], bins) - 1
    selected_values = np.asarray(values, float)[window]
    area = (
        np.diff(x_edges_m)[:, None] * np.diff(y_edges_m)[None, :]
    )
    selected_area = area[window]
    for index in range(centers.size):
        selected = indices == index
        if np.any(selected):
            profile[index] = float(
                np.sum(selected_values[selected] * selected_area[selected])
                / np.sum(selected_area[selected])
            )
    return centers, profile


def profile_metrics(coordinate_m: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(values)
    coordinate = coordinate_m[finite]
    data = np.asarray(values[finite], float)
    if data.size == 0 or np.max(np.abs(data)) == 0.0:
        return {
            "peak_value": 0.0,
            "peak_location_m": None,
            "absolute_centroid_m": None,
            "FWHM_m": None,
        }
    absolute = np.abs(data)
    peak_index = int(np.argmax(absolute))
    weights = absolute / np.sum(absolute)
    above = absolute >= 0.5 * absolute[peak_index]
    return {
        "peak_value": float(data[peak_index]),
        "peak_location_m": float(coordinate[peak_index]),
        "absolute_centroid_m": float(np.sum(coordinate * weights)),
        "FWHM_m": (
            float(np.max(coordinate[above]) - np.min(coordinate[above]))
            if np.count_nonzero(above) > 1
            else 0.0
        ),
    }


def areal_q(q: np.ndarray, z_edges_m: np.ndarray) -> np.ndarray:
    return np.sum(q * np.diff(z_edges_m)[None, None, :], axis=2)


def native_lumerical_areal_q(
    artifact_path: Path,
    physical_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(artifact_path, allow_pickle=False) as raw:
        x = np.asarray(raw["x_m"], float)
        y = np.asarray(raw["y_m"], float)
        z = np.asarray(raw["z_m"], float)
        q = np.asarray(raw["Q_on_W_m3"], float) * physical_scale
    return x, y, np.sum(q * np.diff(nodal_edges(z))[None, None, :], axis=2)


def existing_lumerical_fields(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as raw:
        return {
            key: np.asarray(raw[key])
            for key in (
                "x_edges_m",
                "y_edges_m",
                "z_edges_m",
                "Q_W_m3",
                "temperature_flake_average_K",
                "grad_T_normal_K_m",
            )
        }


def select_thermal_model(
    expanded_geometry: Any,
    expanded_q: np.ndarray,
    thermal_model: str,
) -> tuple[Any, np.ndarray]:
    if thermal_model == "expanded":
        return expanded_geometry, expanded_q
    return thermal.reduced_flake_geometry(expanded_geometry, expanded_q)


def assemble_system(geometry: Any, thermal_model: str) -> Any:
    if thermal_model == "expanded":
        return assemble_expanded_system(geometry)
    return thermal.assemble_steady_diagonal_kappa(
        x_edges_m=geometry.x_edges_m,
        y_edges_m=geometry.y_edges_m,
        z_edges_m=geometry.z_edges_m,
        kappa_W_mK=geometry.kappa_W_mK,
        dirichlet_temperature_K={},
        interface_resistance_m2K_W=geometry.interface_resistance_m2K_W,
        active_mask=geometry.flake_mask,
        surface_robin_heat_transfer_W_m2K={
            "z_min": thermal.G_TAIRTE4_SIO2_W_M2K,
            "z_max": thermal.G_TAIRTE4_AIR_W_M2K,
        },
        surface_robin_temperature_K={"z_min": 0.0, "z_max": 0.0},
    )


def boundary_flux_audit(solved: Any, thermal_model: str) -> dict[str, Any]:
    source = float(solved.source_power_W)
    powers = {
        key: float(value)
        for key, value in solved.boundary_power_out_W.items()
    }
    audit: dict[str, Any] = {
        "boundary_power_out_W": powers,
        "boundary_power_fraction_of_source": {
            key: value / source for key, value in powers.items()
        },
    }
    if thermal_model == "expanded":
        lateral = sum(
            powers.get(face, 0.0)
            for face in ("x_min", "x_max", "y_min", "y_max")
        )
        audit.update(
            {
                "numerical_lateral_Dirichlet_power_W": lateral,
                "numerical_lateral_Dirichlet_fraction": lateral / source,
                "numerical_bottom_Dirichlet_power_W": powers.get(
                    "z_min", 0.0
                ),
                "numerical_bottom_Dirichlet_fraction": powers.get(
                    "z_min", 0.0
                )
                / source,
                "interpretation": (
                    "lateral/bottom Dirichlet flux is numerical truncation-"
                    "boundary flux, not an intrinsic physical heat-path "
                    "fraction"
                ),
            }
        )
    else:
        audit["interpretation"] = (
            "surface Robin z_min/z_max are the reduced paper model's "
            "substrate/air bath paths"
        )
    return audit


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    expanded_geometry = build_straight_geometry(args)
    zero_q = np.zeros(expanded_geometry.material_id.shape, float)
    geometry, _ = select_thermal_model(
        expanded_geometry,
        zero_q,
        args.thermal_model,
    )
    system = assemble_system(geometry, args.thermal_model)
    cases = []
    central_payload: dict[str, np.ndarray] = {}
    for offset_um in args.offsets_um:
        expanded_q, source = analytic_q(
            expanded_geometry, args.polarization, offset_um * 1e-6
        )
        _, q = select_thermal_model(
            expanded_geometry,
            expanded_q,
            args.thermal_model,
        )
        solved = thermal.solve_assembled_thermal_system(
            system,
            source_W_m3=q,
            relative_tolerance=1e-10,
            max_iterations=12000,
        )
        metrics, fields = thermal.straight_edge_temperature_metrics(
            solved.temperature_K, geometry
        )
        case = {
            **source,
            "thermal": {
                "source_power_W": solved.source_power_W,
                "linear_residual_relative": solved.linear_residual_relative,
                "energy_balance_relative_error": (
                    solved.energy_balance_relative_error
                ),
                "iterations": solved.iterations,
                **boundary_flux_audit(solved, args.thermal_model),
            },
            "straight_edge_metrics": metrics,
        }
        cases.append(case)
        if np.isclose(offset_um, 0.0):
            central_payload = {
                "analytic_Q_areal_W_m2": areal_q(
                    q, geometry.z_edges_m
                ),
                "analytic_temperature_flake_average_K": fields[
                    "temperature_flake_average_K"
                ],
                **{
                    f"analytic_{name}": fields[name]
                    for name in (
                        "grad_T_x_K_m",
                        "grad_T_y_K_m",
                        "grad_T_normal_K_m",
                        "grad_T_tangent_K_m",
                        "grad_T_magnitude_K_m",
                    )
                },
            }

    if not central_payload:
        expanded_q, _ = analytic_q(
            expanded_geometry, args.polarization, 0.0
        )
        _, q = select_thermal_model(
            expanded_geometry,
            expanded_q,
            args.thermal_model,
        )
        solved = thermal.solve_assembled_thermal_system(
            system,
            source_W_m3=q,
            relative_tolerance=1e-10,
            max_iterations=12000,
        )
        _, fields = thermal.straight_edge_temperature_metrics(
            solved.temperature_K, geometry
        )
        central_payload = {
            "analytic_Q_areal_W_m2": areal_q(q, geometry.z_edges_m),
            "analytic_temperature_flake_average_K": fields[
                "temperature_flake_average_K"
            ],
            **{
                f"analytic_{name}": fields[name]
                for name in (
                    "grad_T_x_K_m",
                    "grad_T_y_K_m",
                    "grad_T_normal_K_m",
                    "grad_T_tangent_K_m",
                    "grad_T_magnitude_K_m",
                )
            },
        }

    expanded_mapped_q, mapping = thermal.load_and_map_q(
        args.optical_case_dir / "finite_q_on_artifact.npz",
        args.optical_case_dir / "case_result.json",
        expanded_geometry,
    )
    _, mapped_q = select_thermal_model(
        expanded_geometry,
        expanded_mapped_q,
        args.thermal_model,
    )
    remapped_thermal_metrics = None
    if args.solve_remapped_lumerical:
        remapped_solved = thermal.solve_assembled_thermal_system(
            system,
            source_W_m3=mapped_q,
            relative_tolerance=1e-10,
            max_iterations=12000,
        )
        remapped_metrics, remapped_fields = (
            thermal.straight_edge_temperature_metrics(
                remapped_solved.temperature_K, geometry
            )
        )
        existing = {
            "x_edges_m": geometry.x_edges_m,
            "y_edges_m": geometry.y_edges_m,
            "z_edges_m": geometry.z_edges_m,
            "Q_W_m3": mapped_q,
            "temperature_flake_average_K": remapped_fields[
                "temperature_flake_average_K"
            ],
            **{
                name: remapped_fields[name]
                for name in (
                    "grad_T_x_K_m",
                    "grad_T_y_K_m",
                    "grad_T_normal_K_m",
                    "grad_T_tangent_K_m",
                    "grad_T_magnitude_K_m",
                )
            },
        }
        remapped_thermal_metrics = {
            "straight_edge_metrics": remapped_metrics,
            "source_power_W": remapped_solved.source_power_W,
            "linear_residual_relative": (
                remapped_solved.linear_residual_relative
            ),
            "energy_balance_relative_error": (
                remapped_solved.energy_balance_relative_error
            ),
            "iterations": remapped_solved.iterations,
            "source": "saved Lumerical Q conservatively remapped to this grid",
            **boundary_flux_audit(
                remapped_solved,
                args.thermal_model,
            ),
        }
    elif args.lumerical_thermal_dir is not None:
        existing = existing_lumerical_fields(
            args.lumerical_thermal_dir / "thermal_pte_fields.npz"
        )
    else:
        raise ValueError(
            "provide --lumerical-thermal-dir or "
            "--solve-remapped-lumerical"
        )
    if (
        existing["Q_W_m3"].shape != mapped_q.shape
        or not np.array_equal(existing["x_edges_m"], geometry.x_edges_m)
        or not np.array_equal(existing["y_edges_m"], geometry.y_edges_m)
        or not np.array_equal(existing["z_edges_m"], geometry.z_edges_m)
    ):
        raise RuntimeError("saved Lumerical thermal grid does not match")
    mapped_difference = np.linalg.norm(
        existing["Q_W_m3"] - mapped_q
    ) / np.linalg.norm(mapped_q)

    bin_width_m = args.core_step_nm * 1e-9
    tangent_window_m = args.profile_tangent_window_um * 1e-6
    profiles: dict[str, dict[str, Any]] = {}

    def add_profile(
        name: str,
        values: np.ndarray,
        x_edges: np.ndarray,
        y_edges: np.ndarray,
    ) -> None:
        coordinate, profile = binned_profile(
            values,
            x_edges,
            y_edges,
            bin_width_m=bin_width_m,
            tangent_window_m=tangent_window_m,
        )
        profiles[name] = {
            "coordinate_m": coordinate,
            "values": profile,
            "metrics": profile_metrics(coordinate, profile),
        }

    add_profile(
        "analytic_Q_areal",
        central_payload["analytic_Q_areal_W_m2"],
        geometry.x_edges_m,
        geometry.y_edges_m,
    )
    add_profile(
        "analytic_T",
        central_payload["analytic_temperature_flake_average_K"],
        geometry.x_edges_m,
        geometry.y_edges_m,
    )
    add_profile(
        "analytic_grad_normal",
        central_payload["analytic_grad_T_normal_K_m"],
        geometry.x_edges_m,
        geometry.y_edges_m,
    )
    add_profile(
        "remapped_Lumerical_Q_areal",
        areal_q(mapped_q, geometry.z_edges_m),
        geometry.x_edges_m,
        geometry.y_edges_m,
    )
    add_profile(
        "remapped_Lumerical_T",
        existing["temperature_flake_average_K"],
        existing["x_edges_m"],
        existing["y_edges_m"],
    )
    add_profile(
        "remapped_Lumerical_grad_normal",
        existing["grad_T_normal_K_m"],
        existing["x_edges_m"],
        existing["y_edges_m"],
    )
    native_x, native_y, native_q = native_lumerical_areal_q(
        args.optical_case_dir / "finite_q_on_artifact.npz",
        mapping["physical_incident_power_scale"],
    )
    add_profile(
        "native_Lumerical_Q_areal",
        native_q,
        nodal_edges(native_x),
        nodal_edges(native_y),
    )

    profile_npz = {}
    for name, profile in profiles.items():
        profile_npz[f"{name}_n_m"] = profile.pop("coordinate_m")
        profile_npz[f"{name}_values"] = profile.pop("values")
    np.savez(
        args.output_dir / "straight_edge_profiles.npz",
        x_edges_m=geometry.x_edges_m,
        y_edges_m=geometry.y_edges_m,
        **central_payload,
        remapped_Lumerical_Q_areal_W_m2=areal_q(
            mapped_q, geometry.z_edges_m
        ),
        remapped_Lumerical_temperature_flake_average_K=existing[
            "temperature_flake_average_K"
        ],
        **{
            f"remapped_Lumerical_{name}": existing[name]
            for name in (
                "grad_T_x_K_m",
                "grad_T_y_K_m",
                "grad_T_normal_K_m",
                "grad_T_tangent_K_m",
                "grad_T_magnitude_K_m",
            )
            if name in existing
        },
        **profile_npz,
    )

    summary = {
        "status": (
            "COMPLETED_STRAIGHT_EDGE_ANALYTIC_Q_CONTROL"
            if all(
                case["thermal"]["energy_balance_relative_error"] < 0.01
                and case["thermal"]["linear_residual_relative"] < 1e-8
                for case in cases
            )
            and mapping["mapping_relative_power_error"] < 0.005
            and (
                remapped_thermal_metrics is None
                or (
                    remapped_thermal_metrics[
                        "energy_balance_relative_error"
                    ]
                    < 0.01
                    and remapped_thermal_metrics[
                        "linear_residual_relative"
                    ]
                    < 1e-8
                )
            )
            else "FAILED_STRAIGHT_EDGE_ANALYTIC_Q_CONTROL"
        ),
        "polarization": args.polarization,
        "source_contract": {
            "paper_equations": ["Supplement Eq. S1", "Supplement Eq. S2"],
            "incident_power_W": INCIDENT_POWER_W,
            "wavelength_m": WAVELENGTH_M,
            "waist_radius_m": WAIST_M,
            "TMM_full_plane_absorption": TMM_ABSORPTION[
                args.polarization
            ],
            "normalization": (
                "full-plane total absorption fixed by independent TMM; "
                "Beer-Lambert beta controls only normalized depth shape"
            ),
            "not_claimed": (
                "exact paper COMSOL source because numerical objective "
                "transmission and exact 11-um beam radius are unpublished"
            ),
            "optical_constants": optical_constants(args.polarization),
        },
        "geometry": {
            "TaIrTe4_region": "y<=x",
            "edge": "y=x",
            "edge_normal": "n=(y-x)/sqrt(2), positive into air",
            "grid_shape": list(system.shape),
            "core_step_nm": args.core_step_nm,
            "flake_dz_nm": args.flake_dz_nm,
            "thermal_model": args.thermal_model,
        },
        "thermal_model_contract": (
            {
                "identity": (
                    "paper Supplement Eq. S4 reduced flake-only Robin control"
                ),
                "z_min_G_W_m2K": thermal.G_TAIRTE4_SIO2_W_M2K,
                "z_max_G_W_m2K": thermal.G_TAIRTE4_AIR_W_M2K,
                "bulk_Si_SiO2_air": False,
                "lateral_Dirichlet": False,
            }
            if args.thermal_model == "paper-reduced"
            else {
                "identity": "expanded explicit-material production FVM",
                "bulk_Si_SiO2_air": True,
                "lateral_Dirichlet": True,
                "bottom_Dirichlet": True,
            }
        ),
        "analytic_offset_cases": cases,
        "saved_Lumerical_mapping": {
            **mapping,
            "existing_to_recomputed_mapped_Q_NRMSE": mapped_difference,
        },
        "remapped_Lumerical_thermal_solve": remapped_thermal_metrics,
        "profiles": {
            name: value["metrics"] for name, value in profiles.items()
        },
        "weighting_field_applied": False,
        "PTE_current_evaluated": False,
        "generation_commit": git_commit(),
        "generation_command": shlex.join([sys.executable, *sys.argv]),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, suffix, ylabel in (
        (axes[0], "Q_areal", "Q areal (W/m²)"),
        (axes[1], "T", "ΔT (K)"),
        (axes[2], "grad_normal", "|∂T/∂n| (K/m)"),
    ):
        analytic = profiles[f"analytic_{suffix}"]
        remapped = profiles[f"remapped_Lumerical_{suffix}"]
        ax.plot(
            profile_npz[f"analytic_{suffix}_n_m"] * 1e6,
            np.abs(profile_npz[f"analytic_{suffix}_values"]),
            label="analytic Gaussian–Beer–Lambert",
        )
        ax.plot(
            profile_npz[f"remapped_Lumerical_{suffix}_n_m"] * 1e6,
            np.abs(profile_npz[f"remapped_Lumerical_{suffix}_values"]),
            label="saved Lumerical Q chain",
        )
        if suffix == "Q_areal":
            ax.plot(
                profile_npz["native_Lumerical_Q_areal_n_m"] * 1e6,
                np.abs(profile_npz["native_Lumerical_Q_areal_values"]),
                "--",
                label="native Yee/common-grid Q",
            )
        ax.axvline(0.0, color="black", linestyle=":", linewidth=1)
        ax.set(
            xlabel="edge-normal n (µm; air is positive)",
            ylabel=ylabel,
            title=f"E ∥ {args.polarization}: {suffix.replace('_', ' ')}",
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    figure.savefig(args.output_dir / "profile_comparison.png", dpi=180)
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"].startswith("COMPLETED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
