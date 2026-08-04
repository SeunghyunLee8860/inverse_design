#!/usr/bin/env python3
"""Run one empty-stack planar-TMM SiO2 thermal sensitivity case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from photothermal_pte.validation.paper_ir_sanity import (
    run_device_a_explicit_thermal_pte as thermal,
)
from photothermal_pte.validation.paper_ir_sanity.analyze_device_a_current_cause_controls import (
    assemble_operator,
    build_planar_oxide_q,
    setup_geometry,
    uniform_weighting_fields,
)
from photothermal_pte.validation.photothermal_stage1.anisotropic_heat_fvm import (
    solve_assembled_thermal_system,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-contract", type=Path, required=True)
    parser.add_argument("--optical-case-result", type=Path, required=True)
    parser.add_argument("--reference-thermal-fields", type=Path, required=True)
    parser.add_argument("--beam-center-x-um", type=float, required=True)
    parser.add_argument("--beam-center-y-um", type=float, required=True)
    parser.add_argument("--scan-distance-um", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    print(f"[planar-SiO2] d={args.scan_distance_um:g} um: geometry", flush=True)
    geometry = setup_geometry(
        args.geometry_contract,
        args.optical_case_result,
        args.reference_thermal_fields,
    )
    beam_center = (args.beam_center_x_um * 1e-6, args.beam_center_y_um * 1e-6)
    q_oxide, source_audit = build_planar_oxide_q(geometry, beam_center)
    print(
        f"[planar-SiO2] source={source_audit['SiO2_source_power_W']:.9e} W; assembling",
        flush=True,
    )
    system = assemble_operator(geometry)
    print("[planar-SiO2] solving", flush=True)
    solved = solve_assembled_thermal_system(
        system,
        source_W_m3=q_oxide,
        relative_tolerance=1e-10,
        max_iterations=12000,
    )
    flake_xy = np.any(geometry.flake_mask, axis=2)
    psi, actual_grad_x, actual_grad_y, weighting_audit = thermal.solve_weighting_potential(
        geometry.x_edges_m, geometry.y_edges_m, flake_xy
    )
    actual_current, actual_fields = thermal.pte_current(
        solved.temperature_K, geometry, actual_grad_x, actual_grad_y
    )
    controls = {"actual_digitized": actual_current}
    for name, gradients in uniform_weighting_fields(
        geometry.x_edges_m, geometry.y_edges_m, flake_xy
    ).items():
        controls[name] = thermal.pte_current(
            solved.temperature_K, geometry, gradients[0], gradients[1]
        )[0]
    volume = (
        np.diff(geometry.x_edges_m)[:, None, None]
        * np.diff(geometry.y_edges_m)[None, :, None]
        * np.diff(geometry.z_edges_m)[None, None, :]
    )
    summary = {
        "status": (
            "COMPLETED_PLANAR_SIO2_THERMAL_CONTROL"
            if source_audit["relative_depth_integration_error"] < 1e-10
            and solved.linear_residual_relative < 1e-8
            and solved.energy_balance_relative_error < 0.01
            else "FAILED_PLANAR_SIO2_THERMAL_CONTROL"
        ),
        "scan_distance_um": args.scan_distance_um,
        **source_audit,
        "Tmax_rise_K": float(np.max(solved.temperature_K)),
        "TaIrTe4_volume_average_rise_K": thermal.measure_weighted_mean(
            solved.temperature_K, geometry.flake_mask, volume
        ),
        "linear_residual_relative": solved.linear_residual_relative,
        "energy_balance_relative_error": solved.energy_balance_relative_error,
        "iterations": solved.iterations,
        "current_controls_A": controls,
        "weighting_audit": weighting_audit,
        "no_FDTD": True,
        "no_adjoint_ADFD_or_optimization": True,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(thermal.jsonable(summary), indent=2) + "\n"
    )
    np.savez_compressed(
        args.output_dir / "planar_sio2_thermal_fields.npz",
        x_edges_m=geometry.x_edges_m,
        y_edges_m=geometry.y_edges_m,
        z_edges_m=geometry.z_edges_m,
        Q_SiO2_W_m3=q_oxide,
        temperature_rise_K=solved.temperature_K,
        temperature_flake_average_K=actual_fields["temperature_flake_average_K"],
        weighting_potential=psi,
        weighting_grad_x_m_inv=actual_grad_x,
        weighting_grad_y_m_inv=actual_grad_y,
    )
    print(
        f"[planar-SiO2] complete d={args.scan_distance_um:g} um; "
        f"I={actual_current:.9e} A",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
