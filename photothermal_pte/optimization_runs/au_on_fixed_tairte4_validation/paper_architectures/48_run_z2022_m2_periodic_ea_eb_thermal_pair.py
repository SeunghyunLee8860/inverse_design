#!/usr/bin/env python3
"""Paired periodic thermal screen for the corrected 2022 M2 Z cell.

This is deliberately a unit-cell thermal comparison, not a PTE-current or
paper-device reproduction.  Both optical polarizations use the same grid,
materials, interfaces and boundary operator.  The lateral faces are periodic,
the bottom is a zero-temperature-rise bath, and the exposed top is adiabatic.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
RAW_ROOT = Path("/home/seunghyun/tairte4/raw_artifacts")
EA_DIR = RAW_ROOT / "paper_z2022_m2_figure_digitized_Ea_5p3um_v2_matched_cv"
EB_DIR = RAW_ROOT / "paper_z2022_m2_figure_digitized_Eb_5p3um_v2_matched_cv"
OUTPUT = RAW_ROOT / "paper_z2022_m2_figure_digitized_ea_eb_periodic_thermal"
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def segmented_edges(parts_um: list[tuple[float, float, float]]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start, stop, step in parts_um:
        count = int(round((stop - start) / step))
        chunk = np.linspace(start, stop, count + 1)
        chunks.append(chunk if not chunks else chunk[1:])
    return np.concatenate(chunks) * 1.0e-6


def thermal_edges(period_x_m: float, period_y_m: float) -> tuple[np.ndarray, ...]:
    nx = int(round(period_x_m / 25.0e-9))
    ny = int(round(period_y_m / 25.0e-9))
    x = np.linspace(-0.5 * period_x_m, 0.5 * period_x_m, nx + 1)
    y = np.linspace(-0.5 * period_y_m, 0.5 * period_y_m, ny + 1)
    z = segmented_edges(
        [
            (-1.000, -0.685, 0.035),
            (-0.685, -0.400, 0.015),
            (-0.400, -0.200, 0.010),
            (-0.200, 0.000, 0.010),
            (0.000, 0.100, 0.005),
            (0.100, 0.130, 0.005),
            (0.130, 0.250, 0.020),
            (0.250, 0.650, 0.050),
        ]
    )
    return x, y, z


def build_kappa(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    geometry: dict[str, object],
) -> tuple[np.ndarray, dict[str, int]]:
    shape = (x.size, y.size, z.size)
    material = np.full(shape, "air", dtype="U12")
    material[:, :, z < -0.685e-6] = "Si"
    material[:, :, (z >= -0.685e-6) & (z < -0.400e-6)] = "SiO2"
    material[:, :, (z >= -0.400e-6) & (z < -0.200e-6)] = "Au_mirror"
    material[:, :, (z >= -0.200e-6) & (z < 0.0)] = "Al2O3"
    material[:, :, (z >= 0.0) & (z < 0.100e-6)] = "TaIrTe4"
    xx, yy = np.meshgrid(x * 1.0e9, y * 1.0e9, indexing="ij")
    for polygon in geometry["polygons"]:
        vertices = np.asarray(polygon["vertices_nm"], float)
        # The disclosed M2 closure is represented by axis-aligned rectangles.
        inside = (
            (xx >= np.min(vertices[:, 0]))
            & (xx <= np.max(vertices[:, 0]))
            & (yy >= np.min(vertices[:, 1]))
            & (yy <= np.max(vertices[:, 1]))
        )
        for iz in np.flatnonzero(
            (z >= float(polygon["z_min_nm"]) * 1.0e-9)
            & (z < float(polygon["z_max_nm"]) * 1.0e-9)
        ):
            layer = material[:, :, iz]
            layer[inside] = "Au_Z"
    values = {
        "air": (0.026, 0.026, 0.026),
        "Si": (148.0, 148.0, 148.0),
        "SiO2": (1.4, 1.4, 1.4),
        "Au_mirror": (317.0, 317.0, 317.0),
        "Au_Z": (317.0, 317.0, 317.0),
        "Al2O3": (1.5, 1.5, 1.5),
        # Lumerical x=b, y=a, z=c.
        "TaIrTe4": (3.8, 14.4, 1.0),
    }
    kappa = np.empty((*shape, 3), float)
    for name, tensor in values.items():
        kappa[material == name] = tensor
    return kappa, {name: int(np.count_nonzero(material == name)) for name in values}


def periodic_gradient(field: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    return (np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis)) / (2.0 * spacing)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-device", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if str(REPOSITORY) not in sys.path:
        sys.path.insert(0, str(REPOSITORY))
    from photothermal_pte.optimization_runs.cuda_thermal_adjoint import PersistentCudaCSR
    from photothermal_pte.validation.photothermal_stage1.anisotropic_heat_fvm import (
        assemble_steady_diagonal_kappa,
    )

    helpers = load_module("39_run_finite_187t_large_sheet_thermal_pte.py", "z_thermal_remap")
    inputs = {"Ea": EA_DIR, "Eb": EB_DIR}
    optical: dict[str, dict[str, object]] = {}
    for pol, folder in inputs.items():
        payload = json.loads((folder / "Z2022_M2_selected_Q.json").read_text())
        if payload.get("status") != "COMPLETED_Z2022_M2_FIGURE_CORRECTED_SELECTED_Q":
            raise RuntimeError(f"{pol} optical gate did not pass")
        npz = folder / "Z2022_M2_selected_Q.npz"
        entries = [x for x in payload["raw_artifacts"] if Path(x["path"]).name == npz.name]
        if len(entries) != 1 or sha256(npz) != entries[0]["sha256"]:
            raise RuntimeError(f"{pol} Q SHA mismatch")
        optical[pol] = payload
    if optical["Ea"]["geometry"] != optical["Eb"]["geometry"]:
        raise RuntimeError("Ea/Eb geometry mismatch")

    geometry = optical["Ea"]["geometry"]
    period_x = float(geometry["period_x_nm"]) * 1.0e-9
    period_y = float(geometry["period_y_nm"]) * 1.0e-9
    edges = thermal_edges(period_x, period_y)
    centers = tuple(0.5 * (edge[:-1] + edge[1:]) for edge in edges)
    dx, dy, dz = (np.diff(edge) for edge in edges)
    volume = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
    kappa, counts = build_kappa(*centers, geometry)
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
    results: dict[str, dict[str, object]] = {}
    saved: dict[str, np.ndarray] = {}
    for pol, folder in inputs.items():
        component_target: dict[str, np.ndarray] = {}
        source_power_components: dict[str, float] = {}
        with np.load(folder / "Z2022_M2_selected_Q.npz", allow_pickle=False) as data:
            for component in "xyz":
                q = np.asarray(data[f"Q{component}_W_m3"], float)
                source_edges = tuple(
                    helpers.dual_edges(np.asarray(data[f"Q{component}_{axis}_m"], float))
                    for axis in "xyz"
                )
                source_volume = (
                    np.diff(source_edges[0])[:, None, None]
                    * np.diff(source_edges[1])[None, :, None]
                    * np.diff(source_edges[2])[None, None, :]
                )
                source_power_components[component] = float(np.sum(q * source_volume))
                component_target[component] = helpers.conservative_remap_density(
                    q, source_edges, edges
                )
        q_native = sum(component_target.values())
        mapped_power = float(np.sum(q_native * volume))
        source_q_power = float(sum(source_power_components.values()))
        mapping_error = abs(mapped_power - source_q_power) / max(abs(source_q_power), 1.0e-300)
        if mapping_error >= 0.005:
            raise RuntimeError(f"{pol} conservative remap error {mapping_error:.6e}")
        target_incident_power = TARGET_INCIDENT_INTENSITY_W_M2 * period_x * period_y
        scale = target_incident_power / float(optical[pol]["source_power_W"])
        q = q_native * scale
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
        source_power = float(np.sum(assembly.source_volume_operator_m3 @ active_q))
        boundary_power = {
            face: float(np.sum(g * (solved.solution[ids] - bath)))
            for face, (ids, g, bath) in assembly.boundary_terms.items()
        }
        energy_error = abs(sum(boundary_power.values()) - source_power) / max(
            abs(source_power), 1.0e-300
        )
        flake = np.flatnonzero((centers[2] >= 0.0) & (centers[2] < 0.100e-6))
        weights = dz[flake] / np.sum(dz[flake])
        tflake = np.tensordot(temperature[:, :, flake], weights, axes=(2, 0))
        grad_b = periodic_gradient(tflake, float(dx[0]), 0)
        grad_a = periodic_gradient(tflake, float(dy[0]), 1)
        qxy = np.sum(q * dz[None, None, :], axis=2)
        saved[f"{pol}_Q_W_m3"] = q
        saved[f"{pol}_Qxy_W_m2"] = qxy
        saved[f"{pol}_temperature_K"] = temperature
        saved[f"{pol}_TaIrTe4_temperature_K"] = tflake
        saved[f"{pol}_dT_db_K_m"] = grad_b
        saved[f"{pol}_dT_da_K_m"] = grad_a
        saved[f"{pol}_gradT_K_m"] = np.hypot(grad_b, grad_a)
        results[pol] = {
            "P_Q_W_at_1_W_m2_incident": source_power,
            "Q_mapping_error_relative": mapping_error,
            "Tmax_K_per_W_m2": float(np.max(temperature)),
            "TaIrTe4_Tmax_K_per_W_m2": float(np.max(tflake)),
            "max_abs_dT_db_K_m_per_W_m2": float(np.max(np.abs(grad_b))),
            "max_abs_dT_da_K_m_per_W_m2": float(np.max(np.abs(grad_a))),
            "residual_relative": residual,
            "energy_balance_relative": energy_error,
            "solve_wall_time_s": wall,
        }
    saved.update({"x_m": centers[0], "y_m": centers[1], "z_m": centers[2]})
    npz_path = output / "Z2022_M2_PERIODIC_EA_EB_THERMAL.npz"
    np.savez_compressed(npz_path, **saved)
    gates = {
        "both_optical_passed": True,
        "both_mapping_lt_0p5pct": all(x["Q_mapping_error_relative"] < 0.005 for x in results.values()),
        "both_residual_lt_1e_8": all(x["residual_relative"] < 1.0e-8 for x in results.values()),
        "both_energy_balance_lt_1pct": all(x["energy_balance_relative"] < 0.01 for x in results.values()),
    }
    payload = {
        "status": "VALIDATED_Z2022_M2_PERIODIC_EA_EB_THERMAL_SCREEN" if all(gates.values()) else "FAILED_Z2022_M2_PERIODIC_EA_EB_THERMAL_GATE",
        "classification": "paired periodic unit-cell thermal screen; not finite-device PTE or paper thermal reproduction",
        "axis_mapping": "x=b, y=a, z=c",
        "incident_normalization": {"intensity_W_m2": TARGET_INCIDENT_INTENSITY_W_M2, "no_polarization_power_matching": True},
        "thermal_contract": {
            "lateral_boundaries": "periodic x/y",
            "bottom_boundary": "fixed DeltaT=0 at z=-1.0 um",
            "top_boundary": "adiabatic",
            "internal_interfaces": "perfect-contact comparative screen",
            "kappa_W_mK": {"TaIrTe4": [3.8, 14.4, 1.0], "Si": 148.0, "SiO2": 1.4, "Au": 317.0, "Al2O3": 1.5, "air": 0.026},
            "material_cell_counts": counts,
        },
        "cases": results,
        "gates": gates,
        "raw_artifact": {"path": str(npz_path), "size_bytes": npz_path.stat().st_size, "sha256": sha256(npz_path)},
        "scope_exclusions": ["weighting potential", "PTE current", "adjoint", "optimization"],
    }
    (output / "Z2022_M2_PERIODIC_EA_EB_THERMAL.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
