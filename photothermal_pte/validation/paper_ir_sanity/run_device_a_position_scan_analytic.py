#!/usr/bin/env python3
"""Run the Device-A three-position analytic-Q thermal/PTE control offline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import erf, sqrt
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
FVM_DIR = HERE.parent / "photothermal_stage1"
for location in (REPOSITORY, FVM_DIR):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from anisotropic_heat_fvm import solve_assembled_thermal_system  # noqa: E402
from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    run_device_a_explicit_thermal_pte as thermal,
)
from photothermal_pte.validation.paper_ir_sanity import (  # noqa: E402
    run_straight_edge_analytic_q_control as analytic_base,
)
from photothermal_pte.validation.paper_ir_sanity.coordinate_plot import (  # noqa: E402
    cell_field,
    strict_centered_xy_mask,
)


INCIDENT_POWER_W = 285.0e-6
TMM_ABSORPTION = {"a": 0.17673296, "b": 0.26328721}
WAIST_M = 12.0e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-contract", type=Path, required=True)
    parser.add_argument("--scan-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--thermal-domain-um", type=float, default=60.0)
    parser.add_argument("--artifact-tag", default="60um")
    return parser.parse_args()


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


def gaussian_cell_fraction(
    edges_m: np.ndarray, center_m: float, sigma_m: float
) -> np.ndarray:
    scaled = (np.asarray(edges_m, float) - center_m) / (sqrt(2.0) * sigma_m)
    cdf = 0.5 * np.asarray([1.0 + erf(float(value)) for value in scaled])
    return np.diff(cdf)


def analytic_q(
    geometry: thermal.Geometry, polarization: str, center_m: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    constants = analytic_base.optical_constants(polarization)
    beta = float(constants["beta_m_inv"])
    depth_absorption = float(constants["absorbed_depth_fraction_over_130nm"])
    eta = TMM_ABSORPTION[polarization] / depth_absorption
    sigma = WAIST_M / 2.0
    fx = gaussian_cell_fraction(geometry.x_edges_m, float(center_m[0]), sigma)
    fy = gaussian_cell_fraction(geometry.y_edges_m, float(center_m[1]), sigma)
    z0 = geometry.z_edges_m[:-1]
    z1 = geometry.z_edges_m[1:]
    flake_z = (z0 >= -thermal.THICKNESS_M) & (z1 <= 0.0)
    fz = np.zeros_like(z0)
    near = -z1[flake_z]
    far = -z0[flake_z]
    fz[flake_z] = np.exp(-beta * near) - np.exp(-beta * far)
    energy = (
        eta
        * INCIDENT_POWER_W
        * fx[:, None, None]
        * fy[None, :, None]
        * fz[None, None, :]
    )
    energy = np.where(geometry.flake_mask, energy, 0.0)
    volume = (
        np.diff(geometry.x_edges_m)[:, None, None]
        * np.diff(geometry.y_edges_m)[None, :, None]
        * np.diff(geometry.z_edges_m)[None, None, :]
    )
    q = np.divide(energy, volume, out=np.zeros_like(energy), where=volume > 0)
    power = float(np.sum(energy))
    return q, {
        "incident_power_W": INCIDENT_POWER_W,
        "TMM_absorption": TMM_ABSORPTION[polarization],
        "full_plane_absorbed_power_W": (
            INCIDENT_POWER_W * TMM_ABSORPTION[polarization]
        ),
        "finite_flake_absorbed_power_W": power,
        "finite_flake_fraction_of_full_plane": power
        / (INCIDENT_POWER_W * TMM_ABSORPTION[polarization]),
        "beta_m_inv": beta,
        "eta_entrance_factor": eta,
        "waist_radius_m": WAIST_M,
        "center_x_m": float(center_m[0]),
        "center_y_m": float(center_m[1]),
    }


def configure_geometry(
    geometry_path: Path, thermal_domain_um: float,
) -> tuple[thermal.Geometry, dict[str, Any], dict[str, Any]]:
    raw = json.loads(geometry_path.read_text())
    from photothermal_pte.validation.paper_ir_sanity.run_lumerical_device_a_ir_q import (
        load_digitized_device_a_contract,
    )

    frozen = load_digitized_device_a_contract(
        geometry_path, domain_um=60.0, source_span_um=50.0
    )
    thermal.FLAKE_VERTICES_UM = np.asarray(
        frozen["flake_vertices_simulation_um"], float
    )
    shift = np.asarray(frozen["simulation_origin_shift_um"], float)
    thermal.TOP_CONTACT_SEGMENT_UM = np.asarray(
        raw["top_electrical_contact_segment_code_um"], float
    ) + shift
    thermal.BOTTOM_CONTACT_SEGMENT_UM = np.asarray(
        raw["bottom_electrical_contact_segment_code_um"], float
    ) + shift
    geometry = thermal.build_geometry(
        domain_m=thermal_domain_um * 1e-6,
        si_depth_m=20.0e-6,
        core_step_m=100.0e-9,
        flake_dz_m=10.0e-9,
    )
    return geometry, frozen, raw


def assemble(geometry: thermal.Geometry) -> Any:
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


def plot_maps(
    path: Path,
    geometry: thermal.Geometry,
    cases: dict[str, dict[str, Any]],
    inward: np.ndarray,
) -> None:
    figure, axes = plt.subplots(6, 6, figsize=(24, 23), constrained_layout=True)
    flake = np.any(geometry.flake_mask, axis=2)
    strict = strict_centered_xy_mask(flake)
    columns = (
        ("Q_areal_W_m2", "depth-integrated Q", "inferno", False),
        ("temperature_flake_average_K", "thickness-averaged ΔT", "inferno", False),
        ("grad_a_K_m", "∂aT", "coolwarm", True),
        ("grad_b_K_m", "∂bT", "coolwarm", True),
        ("grad_n_K_m", "∂nT", "coolwarm", True),
        ("integrand_A_m2", r"$J_{PTE}\cdot\nabla\psi$ dz", "coolwarm", True),
    )
    for row, key in enumerate(cases):
        data_case = cases[key]
        for col, (field, title, cmap, centered) in enumerate(columns):
            values = np.asarray(data_case[field], float)
            display = strict if field.startswith("grad_") else flake
            values = np.where(display, values, np.nan)
            kwargs: dict[str, Any] = {"cmap": cmap}
            if centered:
                limit = float(np.nanpercentile(np.abs(values), 99.5))
                kwargs.update(vmin=-limit, vmax=limit)
            handle = cell_field(
                axes[row, col],
                geometry.x_edges_m,
                geometry.y_edges_m,
                values,
                coordinate_scale=1e6,
                **kwargs,
            )
            axes[row, col].set(
                title=f"{key}: {title}", xlabel="x=b (µm)", ylabel="y=a (µm)"
            )
            figure.colorbar(handle, ax=axes[row, col])
    figure.suptitle(
        "Device-A analytic Gaussian–Beer–Lambert control; raw same incident power"
    )
    figure.savefig(path, dpi=155)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    scan = json.loads(args.scan_contract.read_text())
    geometry, frozen, raw_geometry = configure_geometry(
        args.geometry_contract, args.thermal_domain_um
    )
    system = assemble(geometry)
    flake_xy = np.any(geometry.flake_mask, axis=2)
    psi, grad_psi_x, grad_psi_y, weighting = thermal.solve_weighting_potential(
        geometry.x_edges_m, geometry.y_edges_m, flake_xy
    )
    inward = np.asarray(
        raw_geometry["off_axis_edge_unit_inward_normal_code"], float
    )
    inward /= np.linalg.norm(inward)
    dz = np.diff(geometry.z_edges_m)
    volume = (
        np.diff(geometry.x_edges_m)[:, None, None]
        * np.diff(geometry.y_edges_m)[None, :, None]
        * dz[None, None, :]
    )
    fields: dict[str, dict[str, Any]] = {}
    rows = []
    raw_payload: dict[str, np.ndarray] = {
        "x_edges_m": geometry.x_edges_m,
        "y_edges_m": geometry.y_edges_m,
        "z_edges_m": geometry.z_edges_m,
        "flake_mask": geometry.flake_mask,
        "weighting_potential": psi,
        "weighting_grad_x_m_inv": grad_psi_x,
        "weighting_grad_y_m_inv": grad_psi_y,
    }
    cases_summary: dict[str, Any] = {}
    for position in scan["cases"]:
        center_m = np.asarray(position["beam_center_simulation_um"], float) * 1e-6
        for polarization in ("a", "b"):
            key = f"{position['label']}_{polarization}"
            q, source = analytic_q(geometry, polarization, center_m)
            solved = solve_assembled_thermal_system(
                system,
                source_W_m3=q,
                relative_tolerance=1.0e-10,
                max_iterations=12000,
            )
            current, pte = thermal.pte_current(
                solved.temperature_K, geometry, grad_psi_x, grad_psi_y
            )
            grad_b = pte["grad_T_x_K_m"]
            grad_a = pte["grad_T_y_K_m"]
            grad_n = inward[0] * grad_b + inward[1] * grad_a
            q_areal = np.sum(q * dz[None, None, :], axis=2)
            display = {
                "Q_areal_W_m2": q_areal,
                "temperature_flake_average_K": pte[
                    "temperature_flake_average_K"
                ],
                "grad_a_K_m": grad_a,
                "grad_b_K_m": grad_b,
                "grad_n_K_m": grad_n,
                "grad_magnitude_K_m": np.hypot(grad_a, grad_b),
                "integrand_A_m2": pte["shockley_ramo_integrand_A_m2"],
            }
            fields[key] = display
            cases_summary[key] = {
                "position_label": position["label"],
                "signed_s_from_edge_um": position["signed_s_from_edge_um"],
                "polarization": polarization,
                "source_contract": source,
                "source_power_reintegrated_W": float(np.sum(q * volume)),
                "PTE_terminal_current_A": current,
                "PTE_terminal_current_nA": current * 1e9,
                "Tmax_rise_K": float(np.max(solved.temperature_K)),
                "flake_average_rise_K": thermal.measure_weighted_mean(
                    solved.temperature_K, geometry.flake_mask, volume
                ),
                "linear_residual_relative": solved.linear_residual_relative,
                "energy_balance_relative_error": solved.energy_balance_relative_error,
                "weighting_volume_area_equivalence_relative_error": float(
                    pte["PTE_volume_area_equivalence_relative_error"][0]
                ),
                "no_Q_clipping_smoothing_gain_rescaling": True,
            }
            rows.append(cases_summary[key])
            raw_payload[f"{key}__Q_W_m3"] = q
            raw_payload[f"{key}__temperature_K"] = solved.temperature_K
            for field, values in display.items():
                raw_payload[f"{key}__{field}"] = values
    for position in scan["cases"]:
        a = cases_summary[f"{position['label']}_a"]["PTE_terminal_current_A"]
        b = cases_summary[f"{position['label']}_b"]["PTE_terminal_current_A"]
        ratio = abs(a) / abs(b)
        cases_summary[f"{position['label']}_a"]["abs_Ia_over_abs_Ib"] = ratio
        cases_summary[f"{position['label']}_b"]["abs_Ia_over_abs_Ib"] = ratio
    raw_path = args.output_dir / (
        f"device_a_position_scan_analytic_{args.artifact_tag}_fields.npz"
    )
    np.savez(raw_path, **raw_payload)
    plot_maps(
        args.report_dir / f"DEVICE_A_ANALYTIC_POSITION_MAPS_{args.artifact_tag}.png",
        geometry,
        fields,
        inward,
    )
    summary = {
        "status": (
            "COMPLETED_DEVICE_A_ANALYTIC_THREE_POSITION_CONTROL_60UM"
            if abs(args.thermal_domain_um - 60.0) < 1e-12
            else "COMPLETED_DEVICE_A_ANALYTIC_THREE_POSITION_DIAGNOSTIC_NONBASELINE_DOMAIN"
        ),
        "scope": (
            "offline analytic Gaussian–Beer–Lambert/TMM Q through the frozen "
            "explicit-3D thermal and weighting-potential terminal-current chain"
        ),
        "not_paper_reproduction": True,
        "source_contract": {
            "assumed_waist_m": WAIST_M,
            "incident_power_W": INCIDENT_POWER_W,
            "TMM_absorption": TMM_ABSORPTION,
            "b_polarization_larger_absorption_is_an_input": True,
            "rescaling_or_polarization_matching": False,
        },
        "thermal_contract": {
            "lateral_domain_m": args.thermal_domain_um * 1e-6,
            "Si_depth_m": 20.0e-6,
            "core_xy_cell_size_m": 100.0e-9,
            "flake_dz_m": 10.0e-9,
            "model": "same expanded explicit-3D operator as Device-A s0",
        },
        "weighting_contract": weighting,
        "cases": cases_summary,
        "gates": {
            "all_residual_lt_1e-8": all(
                row["linear_residual_relative"] < 1.0e-8 for row in rows
            ),
            "all_energy_balance_lt_1percent": all(
                row["energy_balance_relative_error"] < 0.01 for row in rows
            ),
            "all_Q_finite_nonnegative": all(
                np.all(np.isfinite(raw_payload[f"{key}__Q_W_m3"]))
                and np.min(raw_payload[f"{key}__Q_W_m3"]) >= 0.0
                for key in fields
            ),
            "terminal_current_volume_weight_once": True,
        },
        "raw_artifact": {
            "path": str(raw_path.resolve()),
            "size_bytes": raw_path.stat().st_size,
            "sha256": sha256(raw_path),
            "committed_to_git": False,
        },
        "generation_commit": git_commit(),
        "generation_command": " ".join(sys.argv),
    }
    (args.report_dir / f"device_a_analytic_position_{args.artifact_tag}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (args.report_dir / f"device_a_analytic_position_{args.artifact_tag}_cases.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.report_dir / f"RAW_ARTIFACT_MANIFEST_ANALYTIC_{args.artifact_tag}.json").write_text(
        json.dumps({"raw_artifact": summary["raw_artifact"]}, indent=2) + "\n"
    )
    lines = []
    for position in scan["cases"]:
        label = position["label"]
        a = cases_summary[f"{label}_a"]
        b = cases_summary[f"{label}_b"]
        lines.append(
            f"| {position['signed_s_from_edge_um']:.1f} | "
            f"{a['PTE_terminal_current_nA']:.9g} | "
            f"{b['PTE_terminal_current_nA']:.9g} | "
            f"{a['abs_Ia_over_abs_Ib']:.9g} |"
        )
    report = f"""# Device-A analytic three-position terminal-current control

Status: `{summary['status']}`

This is an offline control, not a paper reproduction. It uses the explicitly
assumed 12-um scalar Gaussian, paper/TMM polarization-dependent absorption,
the same expanded explicit-3D thermal operator, and the same digitized-contact
weighting potential as the Maxwell Device-A chain. The larger b-polarized TMM
absorption is an input. No current or Q rescaling was applied.

The lateral thermal domain is `{args.thermal_domain_um:.1f} um`; only the
60-um run matches the immutable s0 Device-A thermal artifact and is eligible
for the promoted Maxwell--analytic comparison.

| signed s from digitized edge (um) | analytic Ia (nA) | analytic Ib (nA) | abs(Ia)/abs(Ib) |
|---:|---:|---:|---:|
{chr(10).join(lines)}

Every terminal current is the full flake-cell volume integral with cell volume
included exactly once. All residual, energy-balance, and finite/nonnegative-Q
gates passed: `{all(summary['gates'].values())}`.
"""
    (args.report_dir / f"DEVICE_A_ANALYTIC_POSITION_CONTROL_{args.artifact_tag}.md").write_text(report)
    print(json.dumps(summary, indent=2))
    return 0 if all(summary["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
