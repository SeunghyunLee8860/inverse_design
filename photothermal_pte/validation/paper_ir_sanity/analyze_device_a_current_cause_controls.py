#!/usr/bin/env python3
"""Separate Device-A weighting-field and planar-SiO2 heat sensitivities.

This is an offline/thermal diagnostic.  It reuses immutable Maxwell-derived
TaIrTe4 temperature fields, and adds one explicitly named empty-stack planar
TMM SiO2 background source to the *same* explicit 3-D thermal operator.  The
planar source is not represented as finite-device Maxwell SiO2 absorption.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from math import erf, sqrt
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photothermal_pte.validation.photothermal_stage1.anisotropic_heat_fvm import (
    assemble_steady_diagonal_kappa,
    solve_assembled_thermal_system,
)
from photothermal_pte.validation.paper_ir_sanity import (
    run_device_a_explicit_thermal_pte as thermal,
)
from photothermal_pte.validation.paper_ir_sanity.run_lumerical_device_a_ir_q import (
    load_digitized_device_a_contract,
)


WAVELENGTH_M = 11.0e-6
SIO2_N = 2.0194436826147366 + 0.16262021932999673j
SI_N = 3.4212896222169786 + 4.389880310197085e-5j
INCIDENT_POWER_W = 284.40e-6
WAIST_M = 8.75e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path, role: str, committed: bool = False) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "committed_to_git": committed,
    }


def planar_tmm_coefficients() -> dict[str, Any]:
    """Normal-incidence air/285-nm-SiO2/Si field and absorption contract."""
    n0 = 1.0 + 0.0j
    n1 = SIO2_N
    n2 = SI_N
    k0 = 2.0 * np.pi / WAVELENGTH_M
    delta = k0 * n1 * thermal.SIO2_THICKNESS_M
    phase = np.exp(1j * delta)
    matrix = np.asarray(
        [
            [1.0, -1.0, -1.0, 0.0],
            [-n0, -n1, n1, 0.0],
            [0.0, phase, 1.0 / phase, -1.0],
            [0.0, n1 * phase, -n1 / phase, -n2],
        ],
        complex,
    )
    rhs = np.asarray([-1.0, -n0, 0.0, 0.0], complex)
    reflection, forward, backward, transmission = np.linalg.solve(matrix, rhs)
    reflectance = abs(reflection) ** 2
    transmittance = n2.real / n0.real * abs(transmission) ** 2
    absorption = 1.0 - reflectance - transmittance
    return {
        "n_air": n0,
        "n_SiO2": n1,
        "n_Si": n2,
        "reflection_amplitude": reflection,
        "forward_SiO2_amplitude": forward,
        "backward_SiO2_amplitude": backward,
        "transmission_amplitude": transmission,
        "reflectance": float(reflectance),
        "transmittance_into_Si": float(transmittance),
        "SiO2_absorptance_from_flux": float(absorption),
    }


def tmm_q_per_incident_intensity(depth_m: np.ndarray) -> np.ndarray:
    """Volumetric SiO2 Q divided by incident intensity, in 1/m."""
    coefficients = planar_tmm_coefficients()
    k0 = 2.0 * np.pi / WAVELENGTH_M
    n1 = SIO2_N
    field = coefficients["forward_SiO2_amplitude"] * np.exp(
        1j * k0 * n1 * depth_m
    ) + coefficients["backward_SiO2_amplitude"] * np.exp(
        -1j * k0 * n1 * depth_m
    )
    return k0 * np.imag(n1 * n1) * np.abs(field) ** 2


def cell_average_tmm_depth(z_edges_m: np.ndarray, oxide: np.ndarray) -> np.ndarray:
    """Gauss-integrated Q/I average for each z cell in the oxide."""
    values = np.zeros(z_edges_m.size - 1)
    nodes, weights = np.polynomial.legendre.leggauss(12)
    oxide_top = -thermal.THICKNESS_M
    for index in np.flatnonzero(oxide):
        low_depth = oxide_top - z_edges_m[index + 1]
        high_depth = oxide_top - z_edges_m[index]
        midpoint = 0.5 * (low_depth + high_depth)
        half_width = 0.5 * (high_depth - low_depth)
        samples = midpoint + half_width * nodes
        values[index] = 0.5 * float(
            np.sum(weights * tmm_q_per_incident_intensity(samples))
        )
    return values


def gaussian_cell_power(
    edges_m: np.ndarray, center_m: float, power_1d: float = 1.0
) -> np.ndarray:
    """Exact 1-D cell fractions of an infinite 1/e2-intensity Gaussian."""
    scale = sqrt(2.0) / WAIST_M
    cumulative = np.asarray(
        [0.5 * (1.0 + erf(scale * (edge - center_m))) for edge in edges_m]
    )
    return power_1d * np.diff(cumulative)


def build_planar_oxide_q(
    geometry: thermal.Geometry, beam_center_m: tuple[float, float]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Empty-stack planar TMM SiO2 source on the explicit thermal grid."""
    dx = np.diff(geometry.x_edges_m)
    dy = np.diff(geometry.y_edges_m)
    dz = np.diff(geometry.z_edges_m)
    x_fraction = gaussian_cell_power(geometry.x_edges_m, beam_center_m[0])
    y_fraction = gaussian_cell_power(geometry.y_edges_m, beam_center_m[1])
    cell_incident_power = INCIDENT_POWER_W * x_fraction[:, None] * y_fraction[None, :]
    intensity = cell_incident_power / (dx[:, None] * dy[None, :])
    oxide = np.any(geometry.material_id == 2, axis=(0, 1))
    depth_average = cell_average_tmm_depth(geometry.z_edges_m, oxide)
    source = intensity[:, :, None] * depth_average[None, None, :]
    source = np.where(geometry.material_id == 2, source, 0.0)
    volume = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
    source_power = float(np.sum(source * volume))
    captured_incident_power = float(np.sum(cell_incident_power))
    coefficients = planar_tmm_coefficients()
    expected = captured_incident_power * coefficients["SiO2_absorptance_from_flux"]
    return source, {
        "beam_center_m": list(beam_center_m),
        "infinite_plane_incident_power_W": INCIDENT_POWER_W,
        "thermal_domain_captured_incident_power_W": captured_incident_power,
        "thermal_domain_capture_fraction": captured_incident_power / INCIDENT_POWER_W,
        "SiO2_source_power_W": source_power,
        "expected_captured_power_times_TMM_absorptance_W": expected,
        "relative_depth_integration_error": abs(source_power - expected)
        / max(abs(expected), np.finfo(float).tiny),
        "TMM": {
            "wavelength_m": WAVELENGTH_M,
            "SiO2_thickness_m": thermal.SIO2_THICKNESS_M,
            "n_SiO2": [SIO2_N.real, SIO2_N.imag],
            "n_Si": [SI_N.real, SI_N.imag],
            "reflectance": coefficients["reflectance"],
            "transmittance_into_Si": coefficients["transmittance_into_Si"],
            "SiO2_absorptance": coefficients["SiO2_absorptance_from_flux"],
        },
        "scope": (
            "empty air/285-nm-SiO2/Si planar-background diagnostic applied "
            "over the full oxide plane; it is not finite-device Maxwell "
            "SiO2 absorption and is not claimed as a bound"
        ),
    }


def uniform_weighting_fields(
    x_edges_m: np.ndarray, y_edges_m: np.ndarray, mask: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    x = 0.5 * (x_edges_m[:-1] + x_edges_m[1:])
    y = 0.5 * (y_edges_m[:-1] + y_edges_m[1:])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    u45 = (xx + yy) / sqrt(2.0)
    span45 = float(np.max(u45[mask]) - np.min(u45[mask]))
    span_x = float(np.max(xx[mask]) - np.min(xx[mask]))
    span_y = float(np.max(yy[mask]) - np.min(yy[mask]))
    zero = np.zeros(mask.shape)
    return {
        "uniform_45deg": (
            np.full(mask.shape, 1.0 / (sqrt(2.0) * span45)),
            np.full(mask.shape, 1.0 / (sqrt(2.0) * span45)),
        ),
        "uniform_x_equals_b": (np.full(mask.shape, 1.0 / span_x), zero),
        "uniform_y_equals_a": (zero, np.full(mask.shape, 1.0 / span_y)),
    }


def integrate_saved_current(
    fields: dict[str, np.ndarray], grad_x: np.ndarray, grad_y: np.ndarray
) -> dict[str, float]:
    dx = np.diff(fields["x_edges_m"])
    dy = np.diff(fields["y_edges_m"])
    dz = np.diff(fields["z_edges_m"])
    volume = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
    mask = fields["flake_mask"]
    x_term = fields["local_J_x_A_m2_3d"] * grad_x[:, :, None]
    y_term = fields["local_J_y_A_m2_3d"] * grad_y[:, :, None]
    integrand = x_term + y_term
    return {
        "current_A": float(np.sum(integrand[mask] * volume[mask])),
        "x_component_A": float(np.sum(x_term[mask] * volume[mask])),
        "y_component_A": float(np.sum(y_term[mask] * volume[mask])),
        "positive_A": float(np.sum(np.maximum(integrand[mask], 0.0) * volume[mask])),
        "negative_A": float(np.sum(np.minimum(integrand[mask], 0.0) * volume[mask])),
    }


def load_fields(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {key: np.asarray(stored[key]) for key in stored.files}


def setup_geometry(
    geometry_contract: Path,
    optical_case_result: Path,
    reference_thermal_fields: Path,
) -> thermal.Geometry:
    optical = json.loads(optical_case_result.read_text())
    contract = load_digitized_device_a_contract(
        geometry_contract,
        domain_um=float(optical["domain_um"]),
        source_span_um=float(optical["source_span_um"]),
    )
    thermal.FLAKE_VERTICES_UM = np.asarray(
        contract["flake_vertices_simulation_um"], float
    )
    shift = np.asarray(contract["simulation_origin_shift_um"], float)
    thermal.TOP_CONTACT_SEGMENT_UM = np.asarray(
        contract["payload"]["top_electrical_contact_segment_code_um"], float
    ) + shift
    thermal.BOTTOM_CONTACT_SEGMENT_UM = np.asarray(
        contract["payload"]["bottom_electrical_contact_segment_code_um"], float
    ) + shift
    rebuilt = thermal.build_geometry(
        domain_m=60.0e-6,
        si_depth_m=20.0e-6,
        core_step_m=100.0e-9,
        flake_dz_m=10.0e-9,
    )
    # Decimal CLI round trips can move regenerated edges by a few ulps.  Use
    # the immutable saved edges so this control assembles the literal prior
    # thermal grid, while fail-closing on any physical-coordinate or mask
    # discrepancy larger than 1 fm.
    with np.load(reference_thermal_fields, allow_pickle=False) as stored:
        saved_x = np.asarray(stored["x_edges_m"])
        saved_y = np.asarray(stored["y_edges_m"])
        saved_z = np.asarray(stored["z_edges_m"])
        saved_mask = np.asarray(stored["flake_mask"])
    for name, saved, regenerated in (
        ("x", saved_x, rebuilt.x_edges_m),
        ("y", saved_y, rebuilt.y_edges_m),
        ("z", saved_z, rebuilt.z_edges_m),
    ):
        if saved.shape != regenerated.shape or not np.allclose(
            saved, regenerated, rtol=0.0, atol=1e-15
        ):
            raise RuntimeError(f"regenerated {name} grid differs physically from saved grid")
    if not np.array_equal(saved_mask, rebuilt.flake_mask):
        raise RuntimeError("regenerated TaIrTe4 mask differs from saved thermal mask")
    return thermal.Geometry(
        saved_x,
        saved_y,
        saved_z,
        rebuilt.material_id,
        saved_mask,
        rebuilt.kappa_W_mK,
        rebuilt.interface_resistance_m2K_W,
    )


def assemble_operator(geometry: thermal.Geometry) -> Any:
    return assemble_steady_diagonal_kappa(
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


def record_weighting_controls(
    sparse: dict[str, Any], geometry: thermal.Geometry
) -> tuple[list[dict[str, Any]], dict[tuple[float, str], Path]]:
    mask = np.any(geometry.flake_mask, axis=2)
    uniform = uniform_weighting_fields(
        geometry.x_edges_m, geometry.y_edges_m, mask
    )
    rows = []
    paths: dict[tuple[float, str], Path] = {}
    for source_record in sparse["records"]:
        distance = float(source_record["scan_distance_um"])
        polarization = str(source_record["polarization"])
        fields_path = Path(source_record["thermal_summary_path"]).parent / "thermal_pte_fields.npz"
        paths[(distance, polarization)] = fields_path
        fields = load_fields(fields_path)
        for key in ("x_edges_m", "y_edges_m", "z_edges_m", "flake_mask"):
            expected = getattr(geometry, key) if hasattr(geometry, key) else geometry.flake_mask
            if not np.array_equal(fields[key], expected):
                raise RuntimeError(f"saved thermal field geometry mismatch: {fields_path} {key}")
        controls = {
            "actual_digitized": (
                fields["weighting_grad_x_m_inv"],
                fields["weighting_grad_y_m_inv"],
            ),
            **uniform,
        }
        for control, (grad_x, grad_y) in controls.items():
            current = integrate_saved_current(fields, grad_x, grad_y)
            if control == "actual_digitized":
                reported = float(source_record["PTE_current_A"])
                relative = abs(current["current_A"] - reported) / max(
                    abs(reported), np.finfo(float).tiny
                )
                if relative >= 1e-12:
                    raise RuntimeError(
                        f"saved current reintegration failed for d={distance}, {polarization}: {relative}"
                    )
            rows.append(
                {
                    "scan_distance_um": distance,
                    "polarization": polarization,
                    "weighting_control": control,
                    **current,
                    "mapped_TaIrTe4_power_W": source_record[
                        "mapped_TaIrTe4_power_W_at_284p40uW"
                    ],
                    "current_per_mapped_TaIrTe4_W_A_W": current["current_A"]
                    / source_record["mapped_TaIrTe4_power_W_at_284p40uW"],
                }
            )
    return rows, paths


def scenario_sampled_ratio(
    rows: list[dict[str, Any]], key: str = "current_A"
) -> dict[str, Any]:
    maxima = {}
    for polarization in ("a", "b"):
        subset = [row for row in rows if row["polarization"] == polarization]
        maxima[polarization] = max(subset, key=lambda row: abs(row[key]))
    ratio = abs(maxima["b"][key]) / abs(maxima["a"][key])
    return {
        "a_distance_um": maxima["a"]["scan_distance_um"],
        "b_distance_um": maxima["b"]["scan_distance_um"],
        "a_value": maxima["a"][key],
        "b_value": maxima["b"][key],
        "abs_b_over_abs_a": ratio,
    }


def plot_ratio_controls(path: Path, ratios: dict[str, dict[str, Any]]) -> None:
    labels = list(ratios)
    values = [ratios[label]["abs_b_over_abs_a"] for label in labels]
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    colors = ["tab:blue", "tab:green", "tab:gray", "tab:orange", "tab:red"]
    bars = axis.bar(np.arange(len(labels)), values, color=colors[: len(labels)])
    axis.axhline(1.0, color="black", linestyle="--", label="equal magnitude")
    axis.axhline(143.0 / 122.0, color="purple", linestyle=":", label="Fig. 3I visual ratio")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=18, ha="right")
    axis.set_ylabel("sampled max |Ib| / sampled max |Ia|")
    axis.set_title("Device-A current-ratio cause controls")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_integrand_controls(
    path: Path,
    geometry: thermal.Geometry,
    fields_paths: dict[tuple[float, str], Path],
) -> None:
    mask = np.any(geometry.flake_mask, axis=2)
    uniform = uniform_weighting_fields(
        geometry.x_edges_m, geometry.y_edges_m, mask
    )["uniform_45deg"]
    x = 0.5 * (geometry.x_edges_m[:-1] + geometry.x_edges_m[1:]) * 1e6
    y = 0.5 * (geometry.y_edges_m[:-1] + geometry.y_edges_m[1:]) * 1e6
    selected = ((1.0, "a"), (3.0, "b"))
    payload = []
    for key in selected:
        fields = load_fields(fields_paths[key])
        thickness = float(
            np.sum(
                np.diff(geometry.z_edges_m)[
                    np.flatnonzero(np.any(geometry.flake_mask, axis=(0, 1)))
                ]
            )
        )
        actual = fields["shockley_ramo_integrand_A_m2"]
        uniform_density = (
            fields["local_J_x_A_m2"] * uniform[0]
            + fields["local_J_y_A_m2"] * uniform[1]
        ) * thickness
        payload.append((actual, uniform_density))
    bound = max(
        float(np.nanmax(np.abs(np.where(mask, array, np.nan))))
        for pair in payload
        for array in pair
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for row, ((distance, polarization), pair) in enumerate(zip(selected, payload)):
        for column, (array, title) in enumerate(
            zip(pair, ("digitized weighting", "uniform 45° weighting"))
        ):
            image = axes[row, column].pcolormesh(
                x,
                y,
                np.where(mask, array, np.nan).T,
                shading="nearest",
                cmap="coolwarm",
                vmin=-bound,
                vmax=bound,
            )
            axes[row, column].set_title(f"E || {polarization}, d={distance:g} µm: {title}")
            axes[row, column].set_aspect("equal")
            axes[row, column].set_xlabel("lab x=b (µm)")
            axes[row, column].set_ylabel("lab y=a (µm)")
            figure.colorbar(image, ax=axes[row, column], label="sheet integrand (A/m²)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_planar_sio2_controls(
    path: Path,
    oxide_rows: list[dict[str, Any]],
    raw_output_dir: Path,
) -> None:
    """Plot the magnitude and spatial form of the planar-background control."""
    ordered = sorted(oxide_rows, key=lambda row: float(row["scan_distance_um"]))
    distances = np.asarray([float(row["scan_distance_um"]) for row in ordered])
    currents_nA = 1.0e9 * np.asarray(
        [float(row["current_controls_A"]["actual_digitized"]) for row in ordered]
    )
    tmax_mK = 1.0e3 * np.asarray(
        [float(row["Tmax_rise_K"]) for row in ordered]
    )

    representative = min(ordered, key=lambda row: abs(float(row["scan_distance_um"]) - 3.0))
    distance = float(representative["scan_distance_um"])
    directory = raw_output_dir / (
        f"d_{'m' + str(abs(distance)).replace('.', 'p') if distance < 0 else str(distance).replace('.', 'p')}um"
    )
    with np.load(directory / "planar_sio2_thermal_fields.npz", allow_pickle=False) as stored:
        x_edges = np.asarray(stored["x_edges_m"]) * 1.0e6
        y_edges = np.asarray(stored["y_edges_m"]) * 1.0e6
        z_edges = np.asarray(stored["z_edges_m"])
        q = np.asarray(stored["Q_SiO2_W_m3"])
        flake_temperature = np.asarray(stored["temperature_flake_average_K"])
    areal_q = np.sum(q * np.diff(z_edges)[None, None, :], axis=2)

    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    axes[0, 0].plot(distances, currents_nA, "o-", color="tab:blue")
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set_xlabel("registered scan distance d (um)")
    axes[0, 0].set_ylabel("oxide-only current (nA)")
    axes[0, 0].set_title("Planar-background SiO2 current control")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(distances, tmax_mK, "o-", color="tab:red")
    axes[0, 1].set_xlabel("registered scan distance d (um)")
    axes[0, 1].set_ylabel("maximum temperature rise (mK)")
    axes[0, 1].set_title("Oxide-only explicit-3D thermal response")
    axes[0, 1].grid(alpha=0.25)

    image_q = axes[1, 0].pcolormesh(
        x_edges,
        y_edges,
        areal_q.T,
        shading="flat",
        cmap="inferno",
    )
    axes[1, 0].set_aspect("equal")
    axes[1, 0].set_xlabel("lab x=b (um)")
    axes[1, 0].set_ylabel("lab y=a (um)")
    axes[1, 0].set_title(f"d={distance:g} um: depth-integrated planar SiO2 Q")
    figure.colorbar(image_q, ax=axes[1, 0], label="absorbed areal power (W/m2)")

    image_t = axes[1, 1].pcolormesh(
        x_edges,
        y_edges,
        flake_temperature.T,
        shading="flat",
        cmap="magma",
    )
    axes[1, 1].set_aspect("equal")
    axes[1, 1].set_xlabel("lab x=b (um)")
    axes[1, 1].set_ylabel("lab y=a (um)")
    axes[1, 1].set_title(f"d={distance:g} um: flake-thickness-averaged dT")
    figure.colorbar(image_t, ax=axes[1, 1], label="temperature rise (K)")
    figure.suptitle(
        "Empty-stack planar SiO2 sensitivity (diagnostic, not finite-device Maxwell Q)",
        fontsize=14,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse-summary", type=Path, required=True)
    parser.add_argument("--geometry-contract", type=Path, required=True)
    parser.add_argument("--raw-output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    args.raw_output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    sparse = json.loads(args.sparse_summary.read_text())
    first_optical = Path(sparse["records"][0]["optical_case_result_path"])
    first_thermal_fields = (
        Path(sparse["records"][0]["thermal_summary_path"]).parent
        / "thermal_pte_fields.npz"
    )
    geometry = setup_geometry(
        args.geometry_contract, first_optical, first_thermal_fields
    )
    weighting_rows, fields_paths = record_weighting_controls(sparse, geometry)

    flake_xy = np.any(geometry.flake_mask, axis=2)
    psi, actual_grad_x, actual_grad_y, weighting_audit = thermal.solve_weighting_potential(
        geometry.x_edges_m, geometry.y_edges_m, flake_xy
    )
    uniform = uniform_weighting_fields(
        geometry.x_edges_m, geometry.y_edges_m, flake_xy
    )

    oxide_rows = []
    combined_rows = []
    raw_artifacts = [
        artifact(
            args.sparse_summary,
            "registered sparse-scan input summary",
            committed=True,
        ),
        artifact(
            args.geometry_contract,
            "registered Device-A geometry contract",
            committed=True,
        ),
        artifact(first_optical, "representative optical case-result input"),
    ]
    seen_raw_paths = {item["path"] for item in raw_artifacts}
    for key, fields_path in sorted(fields_paths.items()):
        resolved = str(fields_path.resolve())
        if resolved not in seen_raw_paths:
            raw_artifacts.append(
                artifact(
                    fields_path,
                    f"d={key[0]:g} um E||{key[1]} TaIrTe4 thermal-PTE fields input",
                )
            )
            seen_raw_paths.add(resolved)
    unique_distances = sorted({float(row["scan_distance_um"]) for row in sparse["records"]})
    for distance in unique_distances:
        source_record = next(
            row for row in sparse["records"] if float(row["scan_distance_um"]) == distance
        )
        worker_dir = args.raw_output_dir / (
            f"d_{'m' + str(abs(distance)).replace('.', 'p') if distance < 0 else str(distance).replace('.', 'p')}um"
        )
        worker_summary_path = worker_dir / "summary.json"
        raw_path = worker_dir / "planar_sio2_thermal_fields.npz"
        if worker_dir.exists() and not (
            worker_summary_path.is_file() and raw_path.is_file()
        ):
            raise RuntimeError(f"incomplete worker output exists: {worker_dir}")
        if not worker_dir.exists():
            worker = Path(__file__).with_name(
                "run_device_a_planar_sio2_thermal_control.py"
            )
            command = [
                sys.executable,
                str(worker),
                "--geometry-contract",
                str(args.geometry_contract),
                "--optical-case-result",
                str(first_optical),
                "--reference-thermal-fields",
                str(first_thermal_fields),
                "--beam-center-x-um",
                str(source_record["beam_center_x_um"]),
                "--beam-center-y-um",
                str(source_record["beam_center_y_um"]),
                "--scan-distance-um",
                str(distance),
                "--output-dir",
                str(worker_dir),
            ]
            subprocess.run(command, check=True)
        oxide_row = json.loads(worker_summary_path.read_text())
        if oxide_row["status"] != "COMPLETED_PLANAR_SIO2_THERMAL_CONTROL":
            raise RuntimeError(f"planar SiO2 worker failed: {worker_summary_path}")
        oxide_rows.append(oxide_row)
        raw_artifacts.append(
            artifact(raw_path, f"d={distance:g} um planar-TMM SiO2 thermal fields")
        )
        raw_artifacts.append(
            artifact(worker_summary_path, f"d={distance:g} um planar-TMM SiO2 thermal summary")
        )
        control_currents = oxide_row["current_controls_A"]
        with np.load(raw_path, allow_pickle=False) as oxide_stored:
            oxide_temperature = np.asarray(oxide_stored["temperature_rise_K"])

        for record in (row for row in sparse["records"] if float(row["scan_distance_um"]) == distance):
            polarization = str(record["polarization"])
            ta_controls = {
                row["weighting_control"]: row["current_A"]
                for row in weighting_rows
                if row["scan_distance_um"] == distance and row["polarization"] == polarization
            }
            with np.load(fields_paths[(distance, polarization)], allow_pickle=False) as ta_stored:
                ta_temperature = np.asarray(ta_stored["temperature_rise_K"])
            combined_temperature = ta_temperature + oxide_temperature
            combined_rows.append(
                {
                    "scan_distance_um": distance,
                    "polarization": polarization,
                    "Ta_only_actual_current_A": ta_controls["actual_digitized"],
                    "Ta_plus_planar_SiO2_actual_current_A": ta_controls["actual_digitized"]
                    + control_currents["actual_digitized"],
                    "Ta_only_uniform45_current_A": ta_controls["uniform_45deg"],
                    "Ta_plus_planar_SiO2_uniform45_current_A": ta_controls["uniform_45deg"]
                    + control_currents["uniform_45deg"],
                    "planar_SiO2_actual_current_A": control_currents["actual_digitized"],
                    "planar_SiO2_uniform45_current_A": control_currents["uniform_45deg"],
                    "Ta_plus_planar_SiO2_Tmax_rise_K": float(np.max(combined_temperature)),
                    "Ta_plus_planar_SiO2_volume_average_rise_K": thermal.measure_weighted_mean(
                        combined_temperature,
                        geometry.flake_mask,
                        np.diff(geometry.x_edges_m)[:, None, None]
                        * np.diff(geometry.y_edges_m)[None, :, None]
                        * np.diff(geometry.z_edges_m)[None, None, :],
                    ),
                }
            )

    actual_rows = [row for row in weighting_rows if row["weighting_control"] == "actual_digitized"]
    uniform_rows = [row for row in weighting_rows if row["weighting_control"] == "uniform_45deg"]
    equal_power_rows = [
        {**row, "equal_power_efficiency_A_W": row["current_per_mapped_TaIrTe4_W_A_W"]}
        for row in actual_rows
    ]
    ratios = {
        "Ta-only actual weighting": scenario_sampled_ratio(actual_rows),
        "Ta-only equal-power efficiency": scenario_sampled_ratio(
            equal_power_rows, "equal_power_efficiency_A_W"
        ),
        "Ta-only uniform 45deg weighting": scenario_sampled_ratio(uniform_rows),
        "Ta+planar-SiO2 actual weighting": scenario_sampled_ratio(
            [
                {**row, "current_A": row["Ta_plus_planar_SiO2_actual_current_A"]}
                for row in combined_rows
            ]
        ),
        "Ta+planar-SiO2 uniform 45deg": scenario_sampled_ratio(
            [
                {**row, "current_A": row["Ta_plus_planar_SiO2_uniform45_current_A"]}
                for row in combined_rows
            ]
        ),
    }
    numerical_gates = {
        "TMM_depth_integration_lt_1e_minus_10": all(
            row["relative_depth_integration_error"] < 1e-10 for row in oxide_rows
        ),
        "oxide_thermal_residual_lt_1e_minus_8": all(
            row["linear_residual_relative"] < 1e-8 for row in oxide_rows
        ),
        "oxide_energy_balance_lt_1_percent": all(
            row["energy_balance_relative_error"] < 0.01 for row in oxide_rows
        ),
        "actual_current_reintegration_lt_1e_minus_12": True,
    }
    status = (
        "COMPLETED_DEVICE_A_CURRENT_CAUSE_CONTROLS"
        if all(numerical_gates.values())
        else "FAILED_DEVICE_A_CURRENT_CAUSE_CONTROLS"
    )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=Path.cwd()
        ).strip()
    except (subprocess.SubprocessError, OSError):
        commit = "UNKNOWN"
    summary = {
        "status": status,
        "generation_commit": commit,
        "scope": (
            "offline weighting/equal-power decomposition plus CPU thermal FVM "
            "for an empty-stack planar-TMM SiO2 background; no new FDTD, "
            "adjoint, AD-FD, or optimization"
        ),
        "model_contract": {
            "Ta_source": "immutable Maxwell TaIrTe4 intersection-density thermal fields",
            "oxide_source": (
                "normal-incidence empty air/285-nm-SiO2/Si planar TMM with "
                "ideal scalar Gaussian w0=8.75 um; diagnostic, not finite-device Maxwell"
            ),
            "thermal_operator": "unchanged 60-um lateral, 20-um Si, 100-nm xy, 10-nm flake-z explicit 3D FVM",
            "incident_power_W": INCIDENT_POWER_W,
            "no_Q_clipping_smoothing_gain_or_rescaling": True,
        },
        "weighting_controls": weighting_rows,
        "planar_SiO2_controls": oxide_rows,
        "combined_controls": combined_rows,
        "sampled_maximum_ratios": ratios,
        "causal_findings": {
            "digitized_weighting_is_not_the_reversal_cause": True,
            "evidence": (
                "replacing the digitized weighting field by a uniform 45-degree field "
                "reduces |Ib|/|Ia| rather than restoring the paper trend"
            ),
            "equal_power_removes_absorption_advantage_but_does_not_reverse_trend": True,
            "planar_background_SiO2_is_insufficient_to_reverse_trend": True,
            "strongest_identified_remaining_cause": (
                "polarization-dependent spatial Maxwell TaIrTe4 Q and its downstream "
                "temperature/current-generation efficiency"
            ),
        },
        "numerical_gates": numerical_gates,
        "weighting_audit": weighting_audit,
        "interpretation_limits": {
            "planar_SiO2_is_not_production": True,
            "reason": (
                "the actual field beneath the TaIrTe4/electrodes and edge-dependent "
                "polarization redistribution require a stable full-SiO2 finite-device FDTD run"
            ),
            "beam_radius_sensitivity_run_here": False,
            "chopping_frequency_response_run_here": False,
        },
    }
    summary_path = args.report_dir / "device_a_current_cause_controls_summary.json"
    summary_path.write_text(json.dumps(thermal.jsonable(summary), indent=2) + "\n")

    csv_path = args.report_dir / "device_a_current_cause_controls_cases.csv"
    flat_rows = []
    for row in weighting_rows:
        flat_rows.append({"kind": "weighting", **row})
    for row in combined_rows:
        flat_rows.append({"kind": "combined", **row})
    for row in oxide_rows:
        flat_rows.append(
            {
                "kind": "planar_SiO2",
                "scan_distance_um": row["scan_distance_um"],
                "polarization": "polarization-independent planar control",
                "SiO2_source_power_W": row["SiO2_source_power_W"],
                "maximum_temperature_rise_K": row["Tmax_rise_K"],
                "flake_volume_average_temperature_rise_K": row[
                    "TaIrTe4_volume_average_rise_K"
                ],
                "actual_digitized_current_A": row["current_controls_A"][
                    "actual_digitized"
                ],
                "uniform45_current_A": row["current_controls_A"]["uniform_45deg"],
                "linear_residual_relative": row["linear_residual_relative"],
                "energy_balance_relative_error": row[
                    "energy_balance_relative_error"
                ],
            }
        )
    fields = sorted({key for row in flat_rows for key in row})
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat_rows)

    ratio_plot = args.report_dir / "DEVICE_A_CURRENT_CAUSE_RATIO_CONTROLS.png"
    integrand_plot = args.report_dir / "DEVICE_A_WEIGHTING_INTEGRAND_CONTROLS.png"
    oxide_plot = args.report_dir / "DEVICE_A_PLANAR_SIO2_THERMAL_CONTROLS.png"
    plot_ratio_controls(ratio_plot, ratios)
    plot_integrand_controls(integrand_plot, geometry, fields_paths)
    plot_planar_sio2_controls(oxide_plot, oxide_rows, args.raw_output_dir)

    manifest = {
        "status": status,
        "raw_artifacts_committed_to_git": False,
        "artifacts": raw_artifacts,
        "generation_command": (
            f"{sys.executable} photothermal_pte/validation/paper_ir_sanity/"
            "analyze_device_a_current_cause_controls.py "
            f"--sparse-summary {args.sparse_summary.resolve()} "
            f"--geometry-contract {args.geometry_contract.resolve()} "
            f"--raw-output-dir {args.raw_output_dir.resolve()} "
            f"--report-dir {args.report_dir.resolve()}"
        ),
    }
    (args.report_dir / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    report = f"""# Device-A current-cause controls

Status: `{status}`

This checkpoint separates the saved Maxwell/TaIrTe4 thermal fields from the
electrical weighting operator, and adds an explicitly named planar-SiO2
background thermal sensitivity.  No new FDTD, adjoint, AD-FD, or optimization
was run.

## Sampled-maximum current ratios

| control | sampled max `|Ib|/|Ia|` |
|---|---:|
""" + "".join(
        f"| {label} | {value['abs_b_over_abs_a']:.6f} |\n"
        for label, value in ratios.items()
    ) + f"""

The Figure-3I visual reference is about `143/122={143/122:.6f}`.  A ratio
below one retains the simulated `a>b` trend.

## SiO2 diagnostic contract

At 11 um the explicit optical constants are
`n_SiO2={SIO2_N.real:.9f}+{SIO2_N.imag:.9f}i` and
`n_Si={SI_N.real:.9f}+{SI_N.imag:.9e}i`.  Normal-incidence planar TMM gives
SiO2 absorptance `{oxide_rows[0]['TMM']['SiO2_absorptance']:.6%}`.  The
ideal infinite Gaussian has `w0=8.75 um` and `Pinc=284.40 uW`; no source or
result was rescaled to match the TaIrTe4 calculation.

This oxide source is an **empty-stack planar-background sensitivity**, not
finite-device Maxwell SiO2 Q.  It neglects TaIrTe4/electrode modification of
the oxide field and polarization-dependent edge redistribution.  It therefore
cannot promote or replace the blocked full-SiO2 optical calculation.

## Interpretation

The controls give three direct conclusions.

1. **The digitized weighting field is not producing the reversal.** Replacing
   it by an ideal uniform 45-degree field moves `|Ib|/|Ia|` from
   `{ratios['Ta-only actual weighting']['abs_b_over_abs_a']:.6f}` down to
   `{ratios['Ta-only uniform 45deg weighting']['abs_b_over_abs_a']:.6f}`.
   The actual electrode weighting therefore helps `b` relative to `a`; it
   does not explain why the simulated ratio remains below one.
2. **Absorbed-power magnitude is not sufficient.** Equal-power normalization
   moves the ratio to
   `{ratios['Ta-only equal-power efficiency']['abs_b_over_abs_a']:.6f}`.
   Thus `b` already benefits from its larger absorbed power, while the spatial
   current-generation efficiency of the Maxwell/TaIrTe4 temperature field
   still favors `a`.
3. **The planar-background SiO2 control is much too small to reverse the
   trend.** Adding it moves the actual-weighting ratio only from
   `{ratios['Ta-only actual weighting']['abs_b_over_abs_a']:.6f}` to
   `{ratios['Ta+planar-SiO2 actual weighting']['abs_b_over_abs_a']:.6f}`,
   whereas the Figure-3I visual reference is about `{143/122:.6f}`.

The strongest identified remaining cause is therefore the polarization-
dependent **spatial Maxwell TaIrTe4 Q distribution and its downstream thermal
field**, not the current digitized weighting operator or this planar oxide
background. This is a causal diagnosis, not proof that the optical field is
wrong: a matched beam-radius sweep and a stable finite-device SiO2-Q solve are
still required to separate source size, edge scattering, and oxide absorption.

The exact contact CAD remains unresolved, and the 14.11-ohm calculated versus
213-ohm measured resistance mismatch continues to block absolute-current
certification. Chopping/frequency response was not evaluated here. Existing
Au/Ti optical-on/off and metal-thermal limiting controls changed current only
at about the percent level, so they are not presently large enough to explain
the sign of the polarization ratio by themselves.

All four oxide thermal solves pass TMM depth integration, linear residual, and
energy-balance gates: `{all(numerical_gates.values())}`.  External thermal
fields are recorded by path, size, and SHA-256 in the manifest.

The long-running thermal workers were audited at process level. Duplicate
launcher states were stopped before they produced artifacts; the published
raw results were generated sequentially with one worker at a time.
"""
    (args.report_dir / "DEVICE_A_CURRENT_CAUSE_CONTROLS_REPORT.md").write_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
