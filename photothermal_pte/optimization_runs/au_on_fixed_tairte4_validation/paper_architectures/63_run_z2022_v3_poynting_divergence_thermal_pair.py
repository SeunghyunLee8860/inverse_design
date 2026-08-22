#!/usr/bin/env python3
"""Run the same periodic thermal operator for paired Ea/Eb signed Qdiv.

This is a diagnostic bridge while the native volumetric-loss monitor remains
blocked by optical closure.  It intentionally retains signed Poynting-
divergence cells and must not be promoted as a physical heat-source result.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
RAW_ROOT = Path("/home/seunghyun/tairte4/raw_artifacts")
INPUT = RAW_ROOT / "paper_z2022_m2_v3_ea_eb_poynting_divergence"
OUTPUT = RAW_ROOT / "paper_z2022_m2_v3_ea_eb_poynting_divergence_thermal"
TARGET_INCIDENT_INTENSITY_W_M2 = 1.0


def load_module(filename: str, name: str):
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-device", type=int, default=5)
    parser.add_argument("--input-dir", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if str(REPOSITORY) not in sys.path:
        sys.path.insert(0, str(REPOSITORY))
    from photothermal_pte.optimization_runs.cuda_thermal_adjoint import PersistentCudaCSR
    from photothermal_pte.validation.photothermal_stage1.anisotropic_heat_fvm import (
        assemble_steady_diagonal_kappa,
    )

    contract = load_module("48_run_z2022_m2_periodic_ea_eb_thermal_pair.py", "z2022_thermal_contract")
    remap = load_module("39_run_finite_187t_large_sheet_thermal_pte.py", "z2022_thermal_remap")
    source = args.input_dir.expanduser().resolve()
    summary = json.loads((source / "Z2022_M2_EA_EB_POYNTING_DIVERGENCE_Q.json").read_text())
    if summary.get("status") != "DIAGNOSTIC_Z2022_M2_PAIRED_POYNTING_DIVERGENCE_Q":
        raise RuntimeError("paired conservative Qdiv diagnostic is missing")
    geometry = summary["geometry"]
    period_x = float(geometry["period_x_nm"]) * 1.0e-9
    period_y = float(geometry["period_y_nm"]) * 1.0e-9
    edges = contract.thermal_edges(period_x, period_y)
    centers = tuple(0.5 * (edge[:-1] + edge[1:]) for edge in edges)
    dx, dy, dz = (np.diff(edge) for edge in edges)
    volume = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
    kappa, material_counts = contract.build_kappa(*centers, geometry)
    assembly = assemble_steady_diagonal_kappa(
        x_edges_m=edges[0],
        y_edges_m=edges[1],
        z_edges_m=edges[2],
        kappa_W_mK=kappa,
        active_mask=np.ones(kappa.shape[:3], bool),
        dirichlet_temperature_K={"z_min": 0.0},
        periodic_axes=("x", "y"),
    )
    operator = PersistentCudaCSR(assembly.matrix_W_K, cuda_device=args.cuda_device)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    saved: dict[str, np.ndarray] = {}
    results: dict[str, object] = {}
    with np.load(source / "Z2022_M2_EA_EB_POYNTING_DIVERGENCE_Q.npz", allow_pickle=False) as data:
        source_edges = tuple(np.asarray(data[f"{axis}_edges_m"], float) for axis in "xyz")
        for pol in ("Ea", "Eb"):
            q_native = np.asarray(data[f"{pol}_Qdiv_W_m3"], float)
            source_volume = (
                np.diff(source_edges[0])[:, None, None]
                * np.diff(source_edges[1])[None, :, None]
                * np.diff(source_edges[2])[None, None, :]
            )
            source_power = float(np.sum(q_native * source_volume))
            q_remapped = remap.conservative_remap_density(q_native, source_edges, edges)
            remapped_power = float(np.sum(q_remapped * volume))
            mapping_error = abs(remapped_power - source_power) / max(abs(source_power), np.finfo(float).tiny)
            if mapping_error >= 0.005:
                raise RuntimeError(f"{pol}: conservative remap error {mapping_error:.6e}")
            optical_json = Path(summary["cases"][pol]["source_npz"]).with_name("Z2022_M2_selected_Q.json")
            optical = json.loads(optical_json.read_text())
            target_incident_power = TARGET_INCIDENT_INTENSITY_W_M2 * period_x * period_y
            scale = target_incident_power / float(optical["source_power_W"])
            q = q_remapped * scale
            active_q = assembly.active_source(q)
            rhs = np.asarray(assembly.source_volume_operator_m3 @ active_q).reshape(-1)
            started = perf_counter()
            solved = operator.solve(rhs, relative_tolerance=1.0e-10, max_iterations=30000)
            wall = perf_counter() - started
            temperature = assembly.full_field(solved.solution)
            residual = float(
                np.linalg.norm(assembly.matrix_W_K @ solved.solution - rhs)
                / max(np.linalg.norm(rhs), np.finfo(float).tiny)
            )
            power = float(np.sum(assembly.source_volume_operator_m3 @ active_q))
            boundary_power = {
                face: float(np.sum(g * (solved.solution[ids] - bath)))
                for face, (ids, g, bath) in assembly.boundary_terms.items()
            }
            energy_error = abs(sum(boundary_power.values()) - power) / max(abs(power), 1.0e-300)
            flake = np.flatnonzero((centers[2] >= 0.0) & (centers[2] < 0.100e-6))
            weights = dz[flake] / np.sum(dz[flake])
            tflake = np.tensordot(temperature[:, :, flake], weights, axes=(2, 0))
            grad_b = contract.periodic_gradient(tflake, float(dx[0]), 0)
            grad_a = contract.periodic_gradient(tflake, float(dy[0]), 1)
            qxy = np.sum(q * dz[None, None, :], axis=2)
            saved[f"{pol}_Q_W_m3"] = q
            saved[f"{pol}_Qxy_W_m2"] = qxy
            saved[f"{pol}_temperature_K"] = temperature
            saved[f"{pol}_TaIrTe4_temperature_K"] = tflake
            saved[f"{pol}_dT_db_K_m"] = grad_b
            saved[f"{pol}_dT_da_K_m"] = grad_a
            saved[f"{pol}_gradT_K_m"] = np.hypot(grad_b, grad_a)
            negative_power = float(np.sum(np.minimum(q, 0.0) * volume))
            positive_power = float(np.sum(np.maximum(q, 0.0) * volume))
            results[pol] = {
                "P_Q_W_at_1_W_m2_incident": power,
                "Q_mapping_error_relative": mapping_error,
                "remapped_negative_to_positive_power": abs(negative_power) / max(positive_power, np.finfo(float).tiny),
                "Tmin_K_per_W_m2": float(np.min(temperature)),
                "Tmax_K_per_W_m2": float(np.max(temperature)),
                "TaIrTe4_Tmax_K_per_W_m2": float(np.max(tflake)),
                "max_abs_dT_db_K_m_per_W_m2": float(np.max(np.abs(grad_b))),
                "max_abs_dT_da_K_m_per_W_m2": float(np.max(np.abs(grad_a))),
                "residual_relative": residual,
                "energy_balance_relative": energy_error,
                "solve_wall_time_s": wall,
            }
    saved.update({"x_m": centers[0], "y_m": centers[1], "z_m": centers[2]})
    out_npz = output / "Z2022_M2_POYNTING_DIVERGENCE_EA_EB_THERMAL.npz"
    np.savez_compressed(out_npz, **saved)
    payload = {
        "status": "DIAGNOSTIC_Z2022_M2_PAIRED_THERMAL_BLOCKED_LOCAL_Q_OSCILLATION",
        "classification": "paired identical-operator thermal response to signed conservative Qdiv; not promoted physical thermal result",
        "axis_mapping": "x=b, y=a, z=c",
        "incident_intensity_W_m2": TARGET_INCIDENT_INTENSITY_W_M2,
        "thermal_contract": {
            "lateral_boundary": "periodic x/y",
            "bottom_boundary": "fixed DeltaT=0 at z=-1 um",
            "top_boundary": "adiabatic",
            "internal_contact": "perfect-contact comparative screen",
            "material_cell_counts": material_counts,
            "same_operator_for_Ea_and_Eb": True,
        },
        "cases": results,
        "gates": {
            "both_mapping_lt_0p5pct": all(results[p]["Q_mapping_error_relative"] < 0.005 for p in results),
            "both_residual_lt_1e_8": all(results[p]["residual_relative"] < 1e-8 for p in results),
            "both_energy_balance_lt_1pct": all(results[p]["energy_balance_relative"] < 0.01 for p in results),
            "physical_nonnegative_Q": False,
        },
        "scope_exclusions": ["weighting potential", "PTE current", "adjoint", "optimization"],
    }
    (output / "Z2022_M2_POYNTING_DIVERGENCE_EA_EB_THERMAL.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
