#!/usr/bin/env python3
"""Device-A evaporated-SiO2 internal-interface-G thermal sensitivity.

This keeps the immutable Maxwell Q, explicit bulk geometry, conductivities,
outer boundaries, and electrical weighting field fixed.  Only the internal
TaIrTe4/SiO2 z-face resistance changes from the paper thermally-grown value
to the paper evaporated-SiO2 estimate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photothermal_pte.validation.paper_ir_sanity import (
    run_device_a_explicit_thermal_pte as thermal,
)
from photothermal_pte.validation.paper_ir_sanity.analyze_device_a_current_cause_controls import (
    assemble_operator,
    load_fields,
    setup_geometry,
)
from photothermal_pte.validation.photothermal_stage1.anisotropic_heat_fvm import (
    AssembledThermalSystem,
    solve_assembled_thermal_system,
)


G_GROWN_W_M2K = 7.37e6
G_EVAPORATED_W_M2K = 7.37e4


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


def evaporated_geometry(
    geometry: thermal.Geometry,
) -> tuple[thermal.Geometry, np.ndarray, dict[str, Any]]:
    material = geometry.material_id
    lower = material[:, :, :-1]
    upper = material[:, :, 1:]
    target = ((lower == 2) & (upper == 3)) | ((lower == 3) & (upper == 2))
    if not np.any(target):
        raise RuntimeError("no internal TaIrTe4/SiO2 z faces")
    interfaces = {
        axis: np.array(values, copy=True)
        for axis, values in geometry.interface_resistance_m2K_W.items()
    }
    old = interfaces["z"][target]
    if not np.allclose(old, 1.0 / G_GROWN_W_M2K, rtol=1e-13, atol=0.0):
        raise RuntimeError("baseline TaIrTe4/SiO2 faces do not carry grown-SiO2 G")
    interfaces["z"][target] = 1.0 / G_EVAPORATED_W_M2K
    unchanged_x = np.array_equal(
        interfaces["x"], geometry.interface_resistance_m2K_W["x"]
    )
    unchanged_y = np.array_equal(
        interfaces["y"], geometry.interface_resistance_m2K_W["y"]
    )
    changed_z = interfaces["z"] != geometry.interface_resistance_m2K_W["z"]
    if not unchanged_x or not unchanged_y or not np.array_equal(changed_z, target):
        raise RuntimeError("interface mutation escaped the target TaIrTe4/SiO2 faces")
    area_xy = (
        np.diff(geometry.x_edges_m)[:, None]
        * np.diff(geometry.y_edges_m)[None, :]
    )
    area = np.broadcast_to(area_xy[:, :, None], target.shape)
    face_z = np.broadcast_to(
        geometry.z_edges_m[1:-1][None, None, :], target.shape
    )
    audit = {
        "target_definition": "internal z faces with material IDs {TaIrTe4=3, SiO2=2}",
        "changed_face_count": int(np.count_nonzero(target)),
        "changed_area_m2": float(np.sum(area[target])),
        "changed_face_z_bounds_m": [
            float(np.min(face_z[target])),
            float(np.max(face_z[target])),
        ],
        "baseline_G_W_m2K": G_GROWN_W_M2K,
        "evaporated_G_W_m2K": G_EVAPORATED_W_M2K,
        "baseline_Rpp_m2K_W": 1.0 / G_GROWN_W_M2K,
        "evaporated_Rpp_m2K_W": 1.0 / G_EVAPORATED_W_M2K,
        "resistance_ratio": G_GROWN_W_M2K / G_EVAPORATED_W_M2K,
        "x_interface_array_unchanged": unchanged_x,
        "y_interface_array_unchanged": unchanged_y,
        "only_target_z_faces_changed": bool(np.array_equal(changed_z, target)),
    }
    changed = thermal.Geometry(
        geometry.x_edges_m,
        geometry.y_edges_m,
        geometry.z_edges_m,
        geometry.material_id,
        geometry.flake_mask,
        geometry.kappa_W_mK,
        interfaces,
    )
    return changed, target, audit


def linear_residual(
    system: AssembledThermalSystem,
    temperature_K: np.ndarray,
    source_W_m3: np.ndarray,
) -> float:
    active_temperature = np.asarray(temperature_K)[system.active_mask].reshape(-1)
    rhs = (
        system.source_volume_operator_m3 @ system.active_source(source_W_m3)
        + system.boundary_load_W
    )
    residual = system.matrix_W_K @ active_temperature - rhs
    return float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), np.finfo(float).tiny))


def interface_flux_metrics(
    geometry: thermal.Geometry,
    temperature_K: np.ndarray,
    target: np.ndarray,
    conductance_W_m2K: float,
) -> dict[str, Any]:
    lower_material = geometry.material_id[:, :, :-1]
    upper_material = geometry.material_id[:, :, 1:]
    dz = np.diff(geometry.z_edges_m)
    area_xy = (
        np.diff(geometry.x_edges_m)[:, None]
        * np.diff(geometry.y_edges_m)[None, :]
    )
    area = np.broadcast_to(area_xy[:, :, None], target.shape)
    k_lower = geometry.kappa_W_mK[:, :, :-1, 2]
    k_upper = geometry.kappa_W_mK[:, :, 1:, 2]
    lower_half = np.broadcast_to(dz[:-1][None, None, :] / (2.0 * k_lower), target.shape)
    upper_half = np.broadcast_to(dz[1:][None, None, :] / (2.0 * k_upper), target.shape)
    rpp = lower_half + 1.0 / conductance_W_m2K + upper_half
    t_lower = temperature_K[:, :, :-1]
    t_upper = temperature_K[:, :, 1:]
    ta_is_upper = (upper_material == 3) & (lower_material == 2) & target
    ta_is_lower = (lower_material == 3) & (upper_material == 2) & target
    delta_center = np.zeros(target.shape)
    delta_center[ta_is_upper] = (t_upper - t_lower)[ta_is_upper]
    delta_center[ta_is_lower] = (t_lower - t_upper)[ta_is_lower]
    flux = np.zeros(target.shape)
    flux[target] = delta_center[target] / rpp[target]
    power = flux * area
    contact_jump = flux / conductance_W_m2K
    target_area = float(np.sum(area[target]))
    return {
        "TaIrTe4_to_SiO2_power_W_signed": float(np.sum(power[target])),
        "area_weighted_heat_flux_W_m2": float(np.sum(power[target]) / target_area),
        "area_weighted_contact_jump_K": float(
            np.sum(contact_jump[target] * area[target]) / target_area
        ),
        "maximum_abs_contact_jump_K": float(np.max(np.abs(contact_jump[target]))),
        "minimum_signed_heat_flux_W_m2": float(np.min(flux[target])),
        "maximum_signed_heat_flux_W_m2": float(np.max(flux[target])),
    }


def current_metrics(
    temperature_K: np.ndarray,
    geometry: thermal.Geometry,
    fields: dict[str, np.ndarray],
) -> tuple[dict[str, Any], np.ndarray]:
    grad_x = np.asarray(fields["weighting_grad_x_m_inv"], float)
    grad_y = np.asarray(fields["weighting_grad_y_m_inv"], float)
    potential = np.asarray(fields["weighting_potential"], float)
    current, produced = thermal.pte_current(temperature_K, geometry, grad_x, grad_y)
    strict_current, strict = thermal.pte_current_strict_centered(
        temperature_K, geometry, potential
    )
    face_current, face = thermal.pte_current_internal_face_bilinear(
        temperature_K, geometry, potential
    )
    dx = np.diff(geometry.x_edges_m)[:, None, None]
    dy = np.diff(geometry.y_edges_m)[None, :, None]
    dz = np.diff(geometry.z_edges_m)[None, None, :]
    volume = dx * dy * dz
    mask = geometry.flake_mask
    x_integrand = produced["local_J_x_A_m2_3d"] * grad_x[:, :, None]
    y_integrand = produced["local_J_y_A_m2_3d"] * grad_y[:, :, None]
    return {
        "production_volume_current_A": current,
        "production_x_term_A": float(np.sum(x_integrand[mask] * volume[mask])),
        "production_y_term_A": float(np.sum(y_integrand[mask] * volume[mask])),
        "production_volume_area_equivalence_relative_error": float(
            produced["PTE_volume_area_equivalence_relative_error"][0]
        ),
        "strict_four_neighbor_current_A": strict_current,
        "strict_valid_xy_cell_count": int(strict["valid_xy_cell_count"]),
        "symmetric_internal_face_current_A": face_current,
        "symmetric_internal_face_count": int(
            face["connected_x_face_count"] + face["connected_y_face_count"]
        ),
    }, np.asarray(produced["temperature_flake_average_K"])


def nrmse(candidate: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    difference = np.asarray(candidate)[mask] - np.asarray(reference)[mask]
    denominator = np.linalg.norm(np.asarray(reference)[mask])
    return float(np.linalg.norm(difference) / max(denominator, np.finfo(float).tiny))


def run_case(
    record: dict[str, Any],
    baseline_geometry: thermal.Geometry,
    evaporated: thermal.Geometry,
    evaporated_system: AssembledThermalSystem,
    target_faces: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    fields_path = Path(record["thermal_summary_path"]).parent / "thermal_pte_fields.npz"
    fields = load_fields(fields_path)
    source = np.asarray(fields["Q_W_m3"], float)
    baseline_temperature = np.asarray(fields["temperature_rise_K"], float)
    print(
        f"[evaporated-G] d={record['scan_distance_um']:g} E||{record['polarization']}: solve",
        flush=True,
    )
    solved = solve_assembled_thermal_system(
        evaporated_system,
        source_W_m3=source,
        relative_tolerance=1.0e-10,
        max_iterations=12000,
    )
    baseline_current, baseline_average = current_metrics(
        baseline_temperature, baseline_geometry, fields
    )
    evaporated_current, evaporated_average = current_metrics(
        solved.temperature_K, evaporated, fields
    )
    volume = evaporated_system.cell_volume_m3
    flake = evaporated.flake_mask
    baseline_tavg = thermal.measure_weighted_mean(baseline_temperature, flake, volume)
    evaporated_tavg = thermal.measure_weighted_mean(solved.temperature_K, flake, volume)
    mask_xy = np.any(flake, axis=2)
    x = 0.5 * (evaporated.x_edges_m[:-1] + evaporated.x_edges_m[1:])
    y = 0.5 * (evaporated.y_edges_m[:-1] + evaporated.y_edges_m[1:])
    bgx, bgy, valid = thermal.strict_centered_cell_gradient(
        baseline_average, mask_xy, x, y
    )
    egx, egy, valid_e = thermal.strict_centered_cell_gradient(
        evaporated_average, mask_xy, x, y
    )
    if not np.array_equal(valid, valid_e):
        raise RuntimeError("strict gradient masks differ")
    gradient_reference = np.concatenate((bgx[valid], bgy[valid]))
    gradient_candidate = np.concatenate((egx[valid], egy[valid]))
    saved_current = float(record["PTE_current_A"])
    row = {
        "scan_distance_um": float(record["scan_distance_um"]),
        "polarization": str(record["polarization"]),
        "source_power_W": float(np.sum(source * volume)),
        "baseline_Tmax_rise_K": float(np.max(baseline_temperature)),
        "evaporated_Tmax_rise_K": float(np.max(solved.temperature_K)),
        "Tmax_evaporated_over_baseline": float(
            np.max(solved.temperature_K) / np.max(baseline_temperature)
        ),
        "baseline_TaIrTe4_volume_average_rise_K": baseline_tavg,
        "evaporated_TaIrTe4_volume_average_rise_K": evaporated_tavg,
        "Tavg_evaporated_over_baseline": evaporated_tavg / baseline_tavg,
        "TaIrTe4_temperature_field_NRMSE": nrmse(
            solved.temperature_K, baseline_temperature, flake
        ),
        "TaIrTe4_thickness_average_field_NRMSE": nrmse(
            evaporated_average, baseline_average, mask_xy
        ),
        "strict_centered_inplane_gradient_NRMSE": float(
            np.linalg.norm(gradient_candidate - gradient_reference)
            / max(np.linalg.norm(gradient_reference), np.finfo(float).tiny)
        ),
        "baseline": baseline_current,
        "evaporated": evaporated_current,
        "production_current_evaporated_over_baseline": (
            evaporated_current["production_volume_current_A"]
            / baseline_current["production_volume_current_A"]
        ),
        "strict_current_evaporated_over_baseline": (
            evaporated_current["strict_four_neighbor_current_A"]
            / baseline_current["strict_four_neighbor_current_A"]
        ),
        "face_current_evaporated_over_baseline": (
            evaporated_current["symmetric_internal_face_current_A"]
            / baseline_current["symmetric_internal_face_current_A"]
        ),
        "saved_baseline_current_reintegration_relative_error": abs(
            baseline_current["production_volume_current_A"] - saved_current
        ) / max(abs(saved_current), np.finfo(float).tiny),
        "evaporated_linear_residual_relative": solved.linear_residual_relative,
        "evaporated_energy_balance_relative_error": solved.energy_balance_relative_error,
        "evaporated_iterations": solved.iterations,
        "baseline_interface": interface_flux_metrics(
            baseline_geometry, baseline_temperature, target_faces, G_GROWN_W_M2K
        ),
        "evaporated_interface": interface_flux_metrics(
            evaporated, solved.temperature_K, target_faces, G_EVAPORATED_W_M2K
        ),
    }
    print(
        f"[evaporated-G] complete d={row['scan_distance_um']:g} E||{row['polarization']}: "
        f"I={evaporated_current['production_volume_current_A']*1e9:.6f} nA, "
        f"Tavg ratio={row['Tavg_evaporated_over_baseline']:.6f}, "
        f"iter={solved.iterations}",
        flush=True,
    )
    return row, baseline_average, evaporated_average


def paired_ratios(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["scan_distance_um"], row["polarization"]): row for row in rows}
    output = []
    for distance in (1.0, 3.0, 5.0):
        a = lookup[(distance, "a")]
        b = lookup[(distance, "b")]
        result: dict[str, Any] = {"scan_distance_um": distance}
        for scenario in ("baseline", "evaporated"):
            for discretization, key in (
                ("production", "production_volume_current_A"),
                ("strict", "strict_four_neighbor_current_A"),
                ("face", "symmetric_internal_face_current_A"),
            ):
                result[f"{scenario}_{discretization}_Ib_over_Ia"] = (
                    b[scenario][key] / a[scenario][key]
                )
        result["production_ratio_change_relative"] = (
            result["evaporated_production_Ib_over_Ia"]
            / result["baseline_production_Ib_over_Ia"]
            - 1.0
        )
        output.append(result)
    return output


def plot_summary(path: Path, rows: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> None:
    labels = [f"d={row['scan_distance_um']:g}, {row['polarization']}" for row in rows]
    x = np.arange(len(rows))
    baseline = np.asarray([row["baseline"]["production_volume_current_A"] for row in rows]) * 1e9
    evaporated = np.asarray([row["evaporated"]["production_volume_current_A"] for row in rows]) * 1e9
    figure, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    axes[0].bar(x - 0.18, baseline, 0.36, label="thermally-grown G")
    axes[0].bar(x + 0.18, evaporated, 0.36, label="evaporated G")
    axes[0].set_xticks(x, labels, rotation=24, ha="right")
    axes[0].set_ylabel("production PTE current (nA)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    distance = np.asarray([row["scan_distance_um"] for row in pairs])
    axes[1].plot(distance, [row["baseline_production_Ib_over_Ia"] for row in pairs], "o-", label="baseline production")
    axes[1].plot(distance, [row["evaporated_production_Ib_over_Ia"] for row in pairs], "o-", label="evaporated production")
    axes[1].plot(distance, [row["evaporated_strict_Ib_over_Ia"] for row in pairs], "s--", label="evaporated strict")
    axes[1].plot(distance, [row["evaporated_face_Ib_over_Ia"] for row in pairs], "^--", label="evaporated symmetric-face")
    axes[1].axhline(1.0, color="black", linewidth=0.9)
    axes[1].set_xlabel("registered edge-normal distance d (um)")
    axes[1].set_ylabel("Ib/Ia")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle("Evaporated-SiO2 interface-G sensitivity; immutable Maxwell Q")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_temperature_maps(
    path: Path,
    geometry: thermal.Geometry,
    maps: dict[tuple[float, str, str], np.ndarray],
) -> None:
    x = 0.5 * (geometry.x_edges_m[:-1] + geometry.x_edges_m[1:]) * 1e6
    y = 0.5 * (geometry.y_edges_m[:-1] + geometry.y_edges_m[1:]) * 1e6
    mask = np.any(geometry.flake_mask, axis=2)
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    for axis, distance, polarization in zip(
        axes.ravel(), (1.0, 1.0, 3.0, 3.0, 5.0, 5.0), ("a", "b", "a", "b", "a", "b")
    ):
        difference = maps[(distance, polarization, "evaporated")] - maps[(distance, polarization, "baseline")]
        image = axis.pcolormesh(x, y, np.where(mask, difference, np.nan).T, shading="nearest", cmap="magma")
        axis.set_aspect("equal")
        axis.set_xlim(-10.0, 5.0)
        axis.set_ylim(-12.0, 5.0)
        axis.set_title(f"d={distance:g} um, E||{polarization}")
        axis.set_xlabel("lab x=b (um)")
        axis.set_ylabel("lab y=a (um)")
        figure.colorbar(image, ax=axis, label="evaporated minus grown DeltaT (K)")
    figure.suptitle("Temperature change caused only by TaIrTe4/SiO2 interface G")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse-summary", type=Path, required=True)
    parser.add_argument("--geometry-contract", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    sparse = json.loads(args.sparse_summary.read_text())
    records = [
        row for row in sparse["records"]
        if float(row["scan_distance_um"]) in (1.0, 3.0, 5.0)
    ]
    first_fields = Path(records[0]["thermal_summary_path"]).parent / "thermal_pte_fields.npz"
    baseline_geometry = setup_geometry(
        args.geometry_contract, Path(records[0]["optical_case_result_path"]), first_fields
    )
    evaporated, target_faces, interface_audit = evaporated_geometry(baseline_geometry)
    if args.audit_only:
        audit = {
            "status": "PASSED_DEVICE_A_EVAPORATED_SIO2_INTERFACE_RUNSETUP_AUDIT",
            "interface_audit": interface_audit,
            "grid_shape": list(baseline_geometry.material_id.shape),
            "bulk_kappa_bitwise_unchanged": bool(
                np.array_equal(evaporated.kappa_W_mK, baseline_geometry.kappa_W_mK)
            ),
            "material_id_bitwise_unchanged": bool(
                np.array_equal(evaporated.material_id, baseline_geometry.material_id)
            ),
            "flake_mask_bitwise_unchanged": bool(
                np.array_equal(evaporated.flake_mask, baseline_geometry.flake_mask)
            ),
        }
        (args.report_dir / "device_a_evaporated_sio2_interface_runsetup_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n"
        )
        print(json.dumps(audit, indent=2), flush=True)
        return 0
    print("[evaporated-G] assembling one modified thermal operator", flush=True)
    system = assemble_operator(evaporated)
    rows: list[dict[str, Any]] = []
    maps: dict[tuple[float, str, str], np.ndarray] = {}
    raw = [
        artifact(args.sparse_summary, "registered sparse-scan summary", True),
        artifact(args.geometry_contract, "registered geometry contract", True),
    ]
    seen = {entry["path"] for entry in raw}
    for record in records:
        row, baseline_map, evaporated_map = run_case(
            record, baseline_geometry, evaporated, system, target_faces
        )
        rows.append(row)
        key = (row["scan_distance_um"], row["polarization"])
        maps[(*key, "baseline")] = baseline_map
        maps[(*key, "evaporated")] = evaporated_map
        fields_path = Path(record["thermal_summary_path"]).parent / "thermal_pte_fields.npz"
        if str(fields_path.resolve()) not in seen:
            raw.append(artifact(fields_path, f"d={key[0]:g} um E||{key[1]} immutable fields"))
            seen.add(str(fields_path.resolve()))
    rows.sort(key=lambda row: (row["scan_distance_um"], row["polarization"]))
    pairs = paired_ratios(rows)
    aggregate = {
        "Tavg_amplification_range": [
            min(row["Tavg_evaporated_over_baseline"] for row in rows),
            max(row["Tavg_evaporated_over_baseline"] for row in rows),
        ],
        "production_current_amplification_range": [
            min(row["production_current_evaporated_over_baseline"] for row in rows),
            max(row["production_current_evaporated_over_baseline"] for row in rows),
        ],
        "evaporated_production_Ib_over_Ia_range": [
            min(row["evaporated_production_Ib_over_Ia"] for row in pairs),
            max(row["evaporated_production_Ib_over_Ia"] for row in pairs),
        ],
        "production_Ib_over_Ia_relative_increase_range": [
            min(row["production_ratio_change_relative"] for row in pairs),
            max(row["production_ratio_change_relative"] for row in pairs),
        ],
        "all_three_current_discretizations_remain_Ib_over_Ia_below_one": all(
            row[f"evaporated_{kind}_Ib_over_Ia"] < 1.0
            for row in pairs
            for kind in ("production", "strict", "face")
        ),
        "contact_jump_amplification_range": [
            min(
                row["evaporated_interface"]["area_weighted_contact_jump_K"]
                / row["baseline_interface"]["area_weighted_contact_jump_K"]
                for row in rows
            ),
            max(
                row["evaporated_interface"]["area_weighted_contact_jump_K"]
                / row["baseline_interface"]["area_weighted_contact_jump_K"]
                for row in rows
            ),
        ],
    }
    gates = {
        "only_target_interface_faces_changed": interface_audit["only_target_z_faces_changed"],
        "saved_baseline_current_reintegration_lt_1e_minus_12": all(row["saved_baseline_current_reintegration_relative_error"] < 1e-12 for row in rows),
        "evaporated_residual_lt_1e_minus_8": all(row["evaporated_linear_residual_relative"] < 1e-8 for row in rows),
        "evaporated_energy_balance_lt_1_percent": all(row["evaporated_energy_balance_relative_error"] < 0.01 for row in rows),
        "source_power_positive_and_unchanged_input": all(row["source_power_W"] > 0.0 for row in rows),
        "current_discretizations_finite": all(
            np.isfinite(value)
            for row in rows
            for scenario in ("baseline", "evaporated")
            for value in row[scenario].values()
            if isinstance(value, (int, float))
        ),
    }
    status = (
        "VALIDATED_DEVICE_A_EVAPORATED_SIO2_INTERFACE_G_SENSITIVITY"
        if all(gates.values())
        else "FAILED_DEVICE_A_EVAPORATED_SIO2_INTERFACE_G_SENSITIVITY"
    )
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "UNKNOWN"
    summary = {
        "status": status,
        "generation_commit": commit,
        "generation_code_sha256": sha256(Path(__file__)),
        "scenario_identity": (
            "explicit-3D sensitivity: only internal TaIrTe4/SiO2 G changed "
            "from thermally-grown baseline to evaporated-SiO2 estimate"
        ),
        "not_claimed": "not a paper Device-A baseline reproduction or fabrication prediction",
        "interface_audit": interface_audit,
        "fixed_contract": {
            "Maxwell_Q": "immutable material-overlap TaIrTe4-only fields",
            "bulk_kappa": "unchanged",
            "TaIrTe4_air_G": "unchanged at 1 W/(m2 K)",
            "SiO2_Si_G": "unchanged at 1.1e9 W/(m2 K)",
            "outer_boundaries": "unchanged explicit-3D numerical boundaries",
            "weighting_field": "immutable digitized Device-A field",
            "Q_rescaling_or_matching": False,
        },
        "cases": rows,
        "paired_Ib_over_Ia": pairs,
        "aggregate_interpretation": aggregate,
        "numerical_gates": gates,
    }
    summary_path = args.report_dir / "device_a_evaporated_sio2_interface_summary.json"
    summary_path.write_text(json.dumps(thermal.jsonable(summary), indent=2) + "\n")
    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        flat = {key: value for key, value in row.items() if not isinstance(value, dict)}
        for scenario in ("baseline", "evaporated"):
            flat.update({f"{scenario}_{key}": value for key, value in row[scenario].items()})
        csv_rows.append({"record_type": "case", **flat})
    csv_rows.extend({"record_type": "paired_ratio", **row} for row in pairs)
    fields = sorted({key for row in csv_rows for key in row})
    with (args.report_dir / "device_a_evaporated_sio2_interface_cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
    plot_summary(args.report_dir / "DEVICE_A_EVAPORATED_SIO2_CURRENT_SENSITIVITY.png", rows, pairs)
    plot_temperature_maps(args.report_dir / "DEVICE_A_EVAPORATED_SIO2_TEMPERATURE_CHANGE.png", evaporated, maps)
    manifest = {
        "status": status,
        "generation_code_sha256": sha256(Path(__file__)),
        "raw_artifacts_committed_to_git": False,
        "artifacts": raw,
        "derived_3D_fields_committed_to_git": False,
        "generation_command": (
            f"{sys.executable} {Path(__file__).resolve()} "
            f"--sparse-summary {args.sparse_summary.resolve()} "
            f"--geometry-contract {args.geometry_contract.resolve()} "
            f"--report-dir {args.report_dir.resolve()}"
        ),
    }
    (args.report_dir / "RAW_ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pair_lines = "".join(
        f"| {row['scan_distance_um']:.0f} | {row['baseline_production_Ib_over_Ia']:.6f} | "
        f"{row['evaporated_production_Ib_over_Ia']:.6f} | {row['evaporated_strict_Ib_over_Ia']:.6f} | "
        f"{row['evaporated_face_Ib_over_Ia']:.6f} | {100*row['production_ratio_change_relative']:.3f}% |\n"
        for row in pairs
    )
    case_lines = "".join(
        f"| {row['scan_distance_um']:.0f} | {row['polarization']} | "
        f"{row['baseline_TaIrTe4_volume_average_rise_K']:.6g} | "
        f"{row['evaporated_TaIrTe4_volume_average_rise_K']:.6g} | "
        f"{row['Tavg_evaporated_over_baseline']:.6f} | "
        f"{row['baseline']['production_volume_current_A']*1e9:.6f} | "
        f"{row['evaporated']['production_volume_current_A']*1e9:.6f} |\n"
        for row in rows
    )
    report = f"""# Device-A evaporated-SiO2 interface-G sensitivity

Status: `{status}`

Only the internal TaIrTe4/SiO2 face conductance changed from
`7.37e6` to `7.37e4 W/(m2 K)`. This is a named evaporated-interface
sensitivity, not a replacement paper baseline and not a fabrication
prediction. Maxwell Q, bulk materials, all other interfaces/boundaries, and
the electrical weighting field are unchanged. No Q rescaling was used.

| d (um) | pol | grown Tavg (K) | evaporated Tavg (K) | Tavg ratio | grown I (nA) | evaporated I (nA) |
|---:|:---:|---:|---:|---:|---:|---:|
{case_lines}

| d (um) | grown production Ib/Ia | evaporated production Ib/Ia | evaporated strict Ib/Ia | evaporated face Ib/Ia | production ratio change |
|---:|---:|---:|---:|---:|---:|
{pair_lines}

All numerical gates are recorded in the summary JSON. Production current is
the unchanged full-volume implementation. Strict four-neighbour and symmetric
internal-face values are diagnostics that test sensitivity to the earlier
boundary-gradient concern.

The lower G raises TaIrTe4 average temperature by
`{aggregate['Tavg_amplification_range'][0]:.2f}--{aggregate['Tavg_amplification_range'][1]:.2f}x`
and production current by
`{aggregate['production_current_amplification_range'][0]:.2f}--{aggregate['production_current_amplification_range'][1]:.2f}x`.
It moves `Ib/Ia` from `0.813--0.845` to
`{aggregate['evaporated_production_Ib_over_Ia_range'][0]:.3f}--{aggregate['evaporated_production_Ib_over_Ia_range'][1]:.3f}`, but all three
current discretizations remain below one. Therefore this interface scenario
strongly reduces, but does not reverse, the present simulated polarization
trend and does not reproduce the paper's approximate `Ib/Ia~1.17`.

Absolute current is not certified because the digitized Device-A geometry's
two-terminal resistance remains far from the measured resistance. The
evaporated value is a literature-based numerical scenario; actual fabrication
must determine whether the full bottom contact is thermally grown, evaporated,
or spatially mixed. The temperature-change image uses independent per-panel
color scales and is diagnostic only; scalar JSON metrics must be used for
cross-panel comparison.

No FDTD, new Q, adjoint, AD-FD, or optimization was run. Raw NPZ inputs remain
external and SHA-pinned; derived 3D temperatures are reproducible and are not
committed.
"""
    (args.report_dir / "DEVICE_A_EVAPORATED_SIO2_INTERFACE_REPORT.md").write_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
