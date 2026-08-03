#!/usr/bin/env python3
"""Solve the frozen 9x2 Device-A optical matrix for two interface-G cases."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from photothermal_pte.validation.paper_ir_sanity import (
    run_device_a_explicit_thermal_pte as thermal,
)


SCENARIOS = {
    "thermally_grown": thermal.G_TAIRTE4_THERMALLY_GROWN_SIO2_W_M2K,
    "evaporated": thermal.G_TAIRTE4_EVAPORATED_SIO2_W_M2K,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--position-contract", type=Path, required=True)
    parser.add_argument("--geometry-contract", type=Path, required=True)
    parser.add_argument("--optical-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--only-scenario", choices=tuple(SCENARIOS), default=None)
    parser.add_argument("--only-label", action="append", default=[])
    parser.add_argument("--only-polarization", choices=("a", "b"), default=None)
    parser.add_argument("--thermal-domain-um", type=float, default=60.0)
    parser.add_argument("--si-depth-um", type=float, default=20.0)
    parser.add_argument("--core-step-nm", type=float, default=100.0)
    parser.add_argument("--flake-dz-nm", type=float, default=10.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assemble(geometry: thermal.Geometry):
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


def configure_fixed_geometry(
    geometry_payload: dict[str, Any], optical_result: dict[str, Any]
) -> None:
    realized = optical_result["pre_run_contract"]["geometry"][
        "digitized_device_a_contract"
    ]
    shift = np.asarray(realized["simulation_origin_shift_um"], float)
    if not np.allclose(shift, [0.0, -3.0], rtol=0.0, atol=1.0e-12):
        raise RuntimeError(f"unexpected optical coordinate shift: {shift}")
    thermal.FLAKE_VERTICES_UM = np.asarray(
        realized["flake_vertices_simulation_um"], float
    )
    thermal.TOP_CONTACT_SEGMENT_UM = (
        np.asarray(geometry_payload["top_electrical_contact_segment_code_um"], float)
        + shift
    )
    thermal.BOTTOM_CONTACT_SEGMENT_UM = (
        np.asarray(geometry_payload["bottom_electrical_contact_segment_code_um"], float)
        + shift
    )


def validate_optical_matrix(
    contract: dict[str, Any], optical_root: Path,
    only_labels: set[str] | None = None,
    only_polarization: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    invariant = None
    for case in contract["cases"]:
        if only_labels and case["label"] not in only_labels:
            continue
        polarizations = (
            (only_polarization,) if only_polarization else ("a", "b")
        )
        for polarization in polarizations:
            directory = optical_root / case["label"] / polarization / "finite"
            result_path = directory / "case_result.json"
            artifact_path = directory / "finite_q_on_artifact.npz"
            if not result_path.is_file() or not artifact_path.is_file():
                raise RuntimeError(f"incomplete optical case: {directory}")
            result = json.loads(result_path.read_text())
            acceptance = result["run_result"]["acceptance"]
            if result["status"] != "COMPLETED" or not all(acceptance.values()):
                raise RuntimeError(f"optical gates failed: {directory}")
            geometry = result["pre_run_contract"]["geometry"]
            source = geometry["source"]
            center = np.asarray(source["beam_center_m"], float) * 1.0e6
            expected = np.asarray(case["beam_center_lumerical_um"], float)
            if not np.allclose(center, expected, rtol=0.0, atol=1.0e-9):
                raise RuntimeError(f"source coordinate mismatch: {directory}")
            current_invariant = {
                "domain_bounds_m": geometry["domain_bounds_m"],
                "flake_vertices_um": geometry["flake_vertices_um"],
                "fixed_local_mesh_center_m": source["fixed_local_mesh_center_m"],
                "axis_contract": geometry["coordinate_contract"],
            }
            if invariant is None:
                invariant = current_invariant
            elif current_invariant != invariant:
                raise RuntimeError(f"case-varying geometry/domain/mesh: {directory}")
            records.append(
                {
                    **case,
                    "polarization": polarization,
                    "optical_dir": directory,
                    "optical_result": result,
                    "optical_result_path": result_path,
                    "optical_artifact_path": artifact_path,
                }
            )
    return records


def strict_maps(
    temperature: np.ndarray,
    geometry: thermal.Geometry,
    weighting_potential: np.ndarray,
) -> dict[str, Any]:
    x = 0.5 * (geometry.x_edges_m[:-1] + geometry.x_edges_m[1:])
    y = 0.5 * (geometry.y_edges_m[:-1] + geometry.y_edges_m[1:])
    dz = np.diff(geometry.z_edges_m)
    flake_z = np.flatnonzero(np.any(geometry.flake_mask, axis=(0, 1)))
    thickness = float(np.sum(dz[flake_z]))
    average = np.sum(
        temperature[:, :, flake_z] * dz[flake_z][None, None, :], axis=2
    ) / thickness
    mask = np.any(geometry.flake_mask, axis=2)
    grad_x, grad_y, valid = thermal.strict_centered_cell_gradient(
        average, mask, x, y
    )
    strict_current, strict = thermal.pte_current_strict_centered(
        temperature, geometry, weighting_potential
    )
    area = (
        np.diff(geometry.x_edges_m)[:, None]
        * np.diff(geometry.y_edges_m)[None, :]
    )
    contribution = np.sum(strict["cell_contribution_A"], axis=2)
    current_density = np.zeros_like(contribution)
    current_density[valid] = contribution[valid] / area[valid]
    return {
        "temperature_flake_average_K": average,
        "grad_T_x_K_m": grad_x,
        "grad_T_y_K_m": grad_y,
        "grad_T_magnitude_K_m": np.sqrt(grad_x**2 + grad_y**2),
        "strict_valid_xy_mask": valid,
        "strict_current_A": strict_current,
        "strict_current_contribution_A_m2": current_density,
        "strict_current_contract": strict["contract"],
    }


def write_progress(
    path: Path,
    contract: dict[str, Any],
    completed: list[dict[str, Any]],
    status: str,
    args: argparse.Namespace,
) -> None:
    payload = {
        "status": status,
        "coordinate_frame": contract["coordinate_frame"],
        "thermal_contract": {
            "lateral_domain_um": args.thermal_domain_um,
            "Si_depth_um": args.si_depth_um,
            "core_xy_cell_size_nm": args.core_step_nm,
            "flake_dz_nm": args.flake_dz_nm,
            "bulk_kappa_W_mK": {
                "TaIrTe4_x_b_y_a_z_c": [3.8, 14.4, 1.0],
                "SiO2": 1.38,
                "Si": 145.0,
                "air": 0.026,
            },
            "G_TaIrTe4_air_W_m2K": 1.0,
            "G_SiO2_Si_W_m2K": 1.1e9,
            "interface_scenarios_W_m2K": SCENARIOS,
            "far_xy_and_bottom": "fixed DeltaT=0 numerical truncation",
            "exposed_h_W_m2K": 10.0,
        },
        "completed_cases": completed,
    }
    path.write_text(json.dumps(thermal.jsonable(payload), indent=2) + "\n")


def main() -> int:
    args = parse_args()
    contract = json.loads(args.position_contract.read_text())
    geometry_payload = json.loads(args.geometry_contract.read_text())
    records = validate_optical_matrix(
        contract,
        args.optical_root,
        set(args.only_label) if args.only_label else None,
        args.only_polarization,
    )
    if not records:
        raise RuntimeError("no selected optical records")
    configure_fixed_geometry(geometry_payload, records[0]["optical_result"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    index_path = args.output_root / "device_a_nine_position_thermal_batch_index.json"
    completed: list[dict[str, Any]] = []
    artifact_hashes: dict[Path, str] = {}

    def cached_sha256(path: Path) -> str:
        resolved = path.resolve()
        if resolved not in artifact_hashes:
            artifact_hashes[resolved] = sha256(resolved)
        return artifact_hashes[resolved]

    selected_scenarios = (
        (args.only_scenario,) if args.only_scenario else tuple(SCENARIOS)
    )
    for scenario in selected_scenarios:
        conductance = SCENARIOS[scenario]
        print(f"[nine-thermal] build {scenario} geometry/operator", flush=True)
        geometry = thermal.build_geometry(
            domain_m=args.thermal_domain_um * 1.0e-6,
            si_depth_m=args.si_depth_um * 1.0e-6,
            core_step_m=args.core_step_nm * 1.0e-9,
            flake_dz_m=args.flake_dz_nm * 1.0e-9,
            tairte4_sio2_G_W_m2K=conductance,
        )
        system = assemble(geometry)
        flake_xy = np.any(geometry.flake_mask, axis=2)
        weighting_potential, weighting_grad_x, weighting_grad_y, weighting = (
            thermal.solve_weighting_potential(
                geometry.x_edges_m, geometry.y_edges_m, flake_xy
            )
        )
        previous_temperature = None
        for record in records:
            output = (
                args.output_root
                / scenario
                / record["label"]
                / record["polarization"]
            )
            summary_path = output / "summary.json"
            fields_path = output / "thermal_lumerical_coordinate_fields.npz"
            if summary_path.is_file() and fields_path.is_file():
                summary = json.loads(summary_path.read_text())
                if summary.get("status") == "COMPLETED_DEVICE_A_NINE_POSITION_THERMAL_CASE":
                    completed.append(summary["index_record"])
                    continue
                raise RuntimeError(f"refusing ambiguous resume output: {output}")
            if output.exists():
                # A disconnect may leave a directory before either atomic
                # result file has been published.  Preserve it verbatim for
                # provenance, then restart only that incomplete case.
                if summary_path.exists():
                    raise RuntimeError(
                        f"summary exists without complete fields: {output}"
                    )
                suffix = 1
                while True:
                    quarantine = output.with_name(
                        f"{output.name}_interrupted_resume_{suffix:03d}"
                    )
                    if not quarantine.exists():
                        output.rename(quarantine)
                        break
                    suffix += 1
            output.mkdir(parents=True, exist_ok=False)
            print(
                f"[nine-thermal] {scenario} {record['label']} "
                f"E||{record['polarization']}",
                flush=True,
            )
            q, mapping = thermal.load_and_map_q(
                record["optical_artifact_path"],
                record["optical_result_path"],
                geometry,
                "isolated-lower-bound",
                "TaIrTe4-only",
                "intersection-density",
                285.0e-6,
            )
            solved = thermal.solve_assembled_thermal_system(
                system,
                source_W_m3=q,
                initial_temperature_K=previous_temperature,
                relative_tolerance=1.0e-10,
                max_iterations=12000,
            )
            previous_temperature = solved.temperature_K
            production_current, production = thermal.pte_current(
                solved.temperature_K,
                geometry,
                weighting_grad_x,
                weighting_grad_y,
            )
            maps = strict_maps(
                solved.temperature_K, geometry, weighting_potential
            )
            dz = np.diff(geometry.z_edges_m)
            q_areal = np.sum(q * dz[None, None, :], axis=2)
            flake_z = np.flatnonzero(np.any(geometry.flake_mask, axis=(0, 1)))
            midplane = solved.temperature_K[:, :, flake_z[len(flake_z) // 2]]
            surface = solved.temperature_K[:, :, flake_z[-1]]
            np.savez_compressed(
                fields_path,
                x_edges_m=geometry.x_edges_m,
                y_edges_m=geometry.y_edges_m,
                z_edges_m=geometry.z_edges_m,
                flake_mask=geometry.flake_mask,
                Q_areal_W_m2=q_areal,
                temperature_rise_K=solved.temperature_K,
                temperature_flake_average_K=maps["temperature_flake_average_K"],
                temperature_flake_midplane_K=midplane,
                temperature_flake_surface_K=surface,
                grad_T_x_K_m=maps["grad_T_x_K_m"],
                grad_T_y_K_m=maps["grad_T_y_K_m"],
                grad_T_magnitude_K_m=maps["grad_T_magnitude_K_m"],
                strict_valid_xy_mask=maps["strict_valid_xy_mask"],
                strict_current_contribution_A_m2=(
                    maps["strict_current_contribution_A_m2"]
                ),
                production_current_integrand_A_m2=(
                    production["shockley_ramo_integrand_A_m2"]
                ),
                weighting_potential=weighting_potential,
            )
            volume = system.cell_volume_m3
            flake = geometry.flake_mask
            boundary = solved.boundary_power_out_W
            lateral_flux = sum(
                boundary[key] for key in ("x_min", "x_max", "y_min", "y_max")
            ) / solved.source_power_W
            gates = {
                "mapping_error_lt_0p5_percent": (
                    mapping["mapping_relative_power_error"] < 0.005
                ),
                "linear_residual_lt_1e_minus_8": (
                    solved.linear_residual_relative < 1.0e-8
                ),
                "energy_balance_lt_1_percent": (
                    solved.energy_balance_relative_error < 0.01
                ),
                "Q_finite_nonnegative": bool(
                    np.all(np.isfinite(q)) and np.count_nonzero(q < 0.0) == 0
                ),
                "strict_current_finite": bool(np.isfinite(maps["strict_current_A"])),
                "production_current_finite": bool(np.isfinite(production_current)),
            }
            index_record = {
                "scenario": scenario,
                "G_TaIrTe4_SiO2_W_m2K": conductance,
                "position_label": record["label"],
                "category": record["category"],
                "vertical_level": record["vertical_level"],
                "polarization": record["polarization"],
                "beam_center_lumerical_um": record["beam_center_lumerical_um"],
                "source_power_W": solved.source_power_W,
                "Tmax_rise_K": float(np.max(solved.temperature_K)),
                "TaIrTe4_volume_average_rise_K": thermal.measure_weighted_mean(
                    solved.temperature_K, flake, volume
                ),
                "production_current_A": production_current,
                "strict_current_A": maps["strict_current_A"],
                "linear_residual_relative": solved.linear_residual_relative,
                "energy_balance_relative_error": solved.energy_balance_relative_error,
                "lateral_numerical_boundary_flux_fraction": lateral_flux,
                "summary_path": str(summary_path.resolve()),
                "fields_path": str(fields_path.resolve()),
            }
            summary = {
                "status": (
                    "COMPLETED_DEVICE_A_NINE_POSITION_THERMAL_CASE"
                    if all(gates.values())
                    else "FAILED_DEVICE_A_NINE_POSITION_THERMAL_CASE"
                ),
                "coordinate_frame": contract["coordinate_frame"],
                "position": record["label"],
                "polarization": record["polarization"],
                "beam_center_lumerical_um": record["beam_center_lumerical_um"],
                "interface_scenario": scenario,
                "G_TaIrTe4_SiO2_W_m2K": conductance,
                "G_TaIrTe4_air_W_m2K": thermal.G_TAIRTE4_AIR_W_M2K,
                "thermal_discretization": {
                    "lateral_domain_um": args.thermal_domain_um,
                    "Si_depth_um": args.si_depth_um,
                    "core_xy_cell_size_nm": args.core_step_nm,
                    "flake_dz_nm": args.flake_dz_nm,
                },
                "mapping": mapping,
                "thermal": {
                    **index_record,
                    "boundary_power_out_W": boundary,
                    "solver": solved.solver,
                    "iterations": solved.iterations,
                },
                "weighting": weighting,
                "strict_gradient_and_current_contract": (
                    maps["strict_current_contract"]
                ),
                "gates": gates,
                "optical_artifacts": {
                    "case_result": str(record["optical_result_path"].resolve()),
                    "case_result_sha256": cached_sha256(record["optical_result_path"]),
                    "raw_Q_NPZ": str(record["optical_artifact_path"].resolve()),
                    "raw_Q_NPZ_sha256": cached_sha256(record["optical_artifact_path"]),
                },
                "index_record": index_record,
                "no_Q_clipping_smoothing_gain_rescaling_tiling": True,
                "no_adjoint_ADFD_or_optimization": True,
            }
            summary_path.write_text(json.dumps(thermal.jsonable(summary), indent=2) + "\n")
            if not all(gates.values()):
                raise RuntimeError(f"thermal gates failed: {output}: {gates}")
            completed.append(index_record)
            write_progress(index_path, contract, completed, "IN_PROGRESS", args)
            del q, solved, production, maps
            gc.collect()
        del system, geometry
        gc.collect()
    write_progress(
        index_path,
        contract,
        completed,
        "COMPLETED_DEVICE_A_NINE_POSITION_TWO_INTERFACE_G_THERMAL_BATCH",
        args,
    )
    print(index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
