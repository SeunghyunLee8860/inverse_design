#!/usr/bin/env python3
"""CUDA thermal/PTE diagnostic for the validated finite-187T optical Q.

The optical artifact has a finite Au inverse-T array but laterally extended
TaIrTe4/lower layers.  This script therefore uses a named *large-sheet*
diagnostic: the entire certified Q support is retained, lateral thermal faces
are adiabatic, the Si bottom is fixed at DeltaT=0, and ideal full-width
top/bottom electrical contacts define an analytic weighting field.  It is not
an experimental finite-contact prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy import sparse


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
DEFAULT_RAW_Q = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "paper_tairte4_finite_187T_w12_Q_11p825um_Eb/finite_187T_w12_Q.npz"
)
DEFAULT_OUTPUT = Path(
    "/home/seunghyun/tairte4/raw_artifacts/"
    "paper_tairte4_finite_187T_w12_large_sheet_thermal_pte"
)
REPORT_INCIDENT_POWER_W = 285.0e-6

T_VERTICES_UM = np.asarray(
    [
        (-0.60, -0.35),
        (0.60, -0.35),
        (0.60, -0.25),
        (0.10, -0.25),
        (0.10, 0.35),
        (-0.10, 0.35),
        (-0.10, -0.25),
        (-0.60, -0.25),
    ],
    float,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dual_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, float)
    edges = np.empty(centers.size + 1, float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def overlap_matrix(target_edges: np.ndarray, source_edges: np.ndarray) -> sparse.csr_matrix:
    """Return target-by-source overlap lengths without extrapolation."""
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    source_index = 0
    for target_index in range(target_edges.size - 1):
        left, right = target_edges[target_index : target_index + 2]
        while source_index + 1 < source_edges.size and source_edges[source_index + 1] <= left:
            source_index += 1
        cursor = source_index
        while cursor + 1 < source_edges.size and source_edges[cursor] < right:
            overlap = min(right, source_edges[cursor + 1]) - max(left, source_edges[cursor])
            if overlap > 0.0:
                rows.append(target_index)
                cols.append(cursor)
                values.append(float(overlap))
            cursor += 1
    return sparse.coo_matrix(
        (values, (rows, cols)),
        shape=(target_edges.size - 1, source_edges.size - 1),
    ).tocsr()


def conservative_remap_density(
    q_source: np.ndarray,
    source_edges: tuple[np.ndarray, np.ndarray, np.ndarray],
    target_edges: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    lx = overlap_matrix(target_edges[0], source_edges[0])
    ly = overlap_matrix(target_edges[1], source_edges[1])
    lz = overlap_matrix(target_edges[2], source_edges[2])
    stage_x = np.asarray(lx @ q_source.reshape(q_source.shape[0], -1)).reshape(
        target_edges[0].size - 1, q_source.shape[1], q_source.shape[2]
    )
    stage_y = np.empty(
        (stage_x.shape[0], target_edges[1].size - 1, stage_x.shape[2]),
        float,
    )
    for iz in range(stage_x.shape[2]):
        stage_y[:, :, iz] = np.asarray(ly @ stage_x[:, :, iz].T).T
    stage_z = np.asarray(lz @ stage_y.reshape(-1, stage_y.shape[2]).T).T.reshape(
        stage_y.shape[0], stage_y.shape[1], target_edges[2].size - 1
    )
    dx, dy, dz = (np.diff(edges) for edges in target_edges)
    volume = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
    return stage_z / volume


def target_lateral_edges() -> np.ndarray:
    # Covers the union of all three component-specific Yee dual cells.  The
    # array footprint is aligned to a 50 nm grid so that the 100 nm T stem
    # and cap thickness each have two cells; the remote sheet is
    # coarsened because it contains neither patterned metal nor appreciable Q.
    left = np.linspace(-27.125, -9.0, 38, dtype=float)
    core = np.linspace(-9.0, 9.0, 361, dtype=float)
    right = np.linspace(9.0, 27.25, 38, dtype=float)
    return np.concatenate((left[:-1], core, right[1:])) * 1.0e-6


def target_z_edges() -> np.ndarray:
    # Optical-closure stack is retained exactly for this diagnostic.
    return np.asarray(
        [
            -20.520,
            -15.0,
            -10.0,
            -5.0,
            -2.0,
            -1.0,
            -0.870,
            -0.520,
            -0.420,
            -0.320,
            -0.235,
            -0.135,
            -0.035,
            0.0,
            0.050,
            0.100,
            0.133,
            0.250,
            0.500,
            0.800,
            1.200,
        ],
        float,
    ) * 1.0e-6


def inverse_t_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(x * 1.0e6, y * 1.0e6, indexing="ij")
    mask = np.zeros(xx.shape, bool)
    for ix in range(11):
        for iy in range(17):
            local_x = xx - (ix - 5) * 1.5
            local_y = yy - (iy - 8) * 1.0
            cap = (np.abs(local_x) < 0.60) & (-0.35 < local_y) & (local_y < -0.25)
            stem = (np.abs(local_x) < 0.10) & (-0.25 < local_y) & (local_y < 0.35)
            mask |= cap | stem
    return mask


def build_kappa(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    shape = (x.size, y.size, z.size)
    kappa = np.empty((*shape, 3), float)
    material = np.full(shape, "air", dtype="U12")
    material[:, :, z < -0.520e-6] = "Si"
    material[:, :, (z >= -0.520e-6) & (z < -0.235e-6)] = "SiO2"
    material[:, :, (z >= -0.235e-6) & (z < -0.035e-6)] = "Au_mirror"
    material[:, :, (z >= -0.035e-6) & (z < 0.0)] = "Al2O3"
    material[:, :, (z >= 0.0) & (z < 0.100e-6)] = "TaIrTe4"
    top_indices = np.flatnonzero((z >= 0.100e-6) & (z < 0.133e-6))
    tmask = inverse_t_mask(x, y)
    for iz in top_indices:
        layer = material[:, :, iz]
        layer[tmask] = "Au_T"

    values = {
        "air": (0.026, 0.026, 0.026),
        "Si": (148.0, 148.0, 148.0),
        "SiO2": (1.4, 1.4, 1.4),
        "Au_mirror": (317.0, 317.0, 317.0),
        "Au_T": (317.0, 317.0, 317.0),
        "Al2O3": (1.5, 1.5, 1.5),
        # Lumerical x=b, y=a, z=c.
        "TaIrTe4": (3.8, 14.4, 1.0),
    }
    for name, tensor in values.items():
        kappa[material == name] = tensor
    counts = {name: int(np.count_nonzero(material == name)) for name in values}
    return kappa, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-npz", type=Path, default=DEFAULT_RAW_Q)
    parser.add_argument("--q-json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--polarization", choices=("Ea", "Eb"), default="Eb")
    parser.add_argument("--cuda-device", type=int, default=5)
    args = parser.parse_args()
    raw_q = args.q_npz.expanduser().resolve()
    q_json = (
        args.q_json.expanduser().resolve()
        if args.q_json is not None
        else raw_q.parent / "FINITE_187T_W12_Q.json"
    )
    output = args.output_dir.expanduser().resolve()
    output_npz = output / f"finite_187T_large_sheet_thermal_pte_{args.polarization}.npz"
    output_json = output / f"FINITE_187T_LARGE_SHEET_THERMAL_PTE_{args.polarization}.json"
    if str(REPOSITORY) not in sys.path:
        sys.path.insert(0, str(REPOSITORY))
    from photothermal_pte.optimization_runs.cuda_thermal_adjoint import PersistentCudaCSR
    from photothermal_pte.validation.photothermal_stage1.anisotropic_heat_fvm import (
        assemble_steady_diagonal_kappa,
    )

    output.mkdir(parents=True, exist_ok=True)
    optical = json.loads(q_json.read_text())
    if optical.get("status") != "VALIDATED_FINITE_187T_W12_VOLUMETRIC_Q":
        raise RuntimeError(f"optical Q is not validated: {optical.get('status')}")
    expected_polarization = "E||a" if args.polarization == "Ea" else "E||b"
    if expected_polarization not in str(optical.get("source", {}).get("polarization", "")):
        raise RuntimeError("requested thermal polarization does not match optical artifact")
    actual_sha = sha256(raw_q)
    expected_entries = [
        item for item in optical.get("raw_artifacts", [])
        if Path(str(item.get("path", ""))).name == raw_q.name
    ]
    if len(expected_entries) != 1 or actual_sha != expected_entries[0].get("sha256"):
        raise RuntimeError(f"Q artifact SHA mismatch: {actual_sha}")
    source_power_w = float(optical["source_power_W"])
    source_scale = REPORT_INCIDENT_POWER_W / source_power_w

    x_edges = target_lateral_edges()
    y_edges = target_lateral_edges()
    z_edges = target_z_edges()
    target_edges = (x_edges, y_edges, z_edges)
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    z = 0.5 * (z_edges[:-1] + z_edges[1:])

    component_target: dict[str, np.ndarray] = {}
    source_component_power: dict[str, float] = {}
    target_component_power: dict[str, float] = {}
    dx, dy, dz = np.diff(x_edges), np.diff(y_edges), np.diff(z_edges)
    target_volume = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
    with np.load(raw_q, allow_pickle=False) as data:
        for component in "xyz":
            q = np.asarray(data[f"Q{component}_W_m3"], float)
            source_edges = tuple(
                dual_edges(np.asarray(data[f"Q{component}_{axis}_m"], float))
                for axis in "xyz"
            )
            source_volume = (
                np.diff(source_edges[0])[:, None, None]
                * np.diff(source_edges[1])[None, :, None]
                * np.diff(source_edges[2])[None, None, :]
            )
            source_component_power[component] = float(np.sum(q * source_volume))
            mapped = conservative_remap_density(q, source_edges, target_edges)
            component_target[component] = mapped
            target_component_power[component] = float(np.sum(mapped * target_volume))
    q_unit = sum(component_target.values())
    q_report = q_unit * source_scale
    p_source_unit = float(sum(source_component_power.values()))
    p_target_unit = float(np.sum(q_unit * target_volume))
    mapping_error = abs(p_target_unit - p_source_unit) / p_source_unit
    if mapping_error >= 0.005:
        raise RuntimeError(f"Q remap power error {mapping_error:.6e}")

    kappa, material_counts = build_kappa(x, y, z)
    rz = np.zeros((x.size, y.size, z.size - 1), float)
    sio2_si_edge = int(np.flatnonzero(np.isclose(z_edges, -0.520e-6, atol=1e-15))[0])
    rz[:, :, sio2_si_edge - 1] = 1.0 / 1.1e9
    assembly_started = perf_counter()
    system = assemble_steady_diagonal_kappa(
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        z_edges_m=z_edges,
        kappa_W_mK=kappa,
        active_mask=np.ones((x.size, y.size, z.size), bool),
        interface_resistance_m2K_W={"z": rz},
        dirichlet_temperature_K={"z_min": 0.0},
    )
    assembly_seconds = perf_counter() - assembly_started
    source_active = system.active_source(q_report)
    rhs = np.asarray(system.source_volume_operator_m3 @ source_active).reshape(-1) + system.boundary_load_W
    operator = PersistentCudaCSR(system.matrix_W_K, cuda_device=args.cuda_device)
    solved = operator.solve(rhs, relative_tolerance=1.0e-10, max_iterations=30000)
    temperature = system.full_field(solved.solution)
    residual = float(
        np.linalg.norm(system.matrix_W_K @ solved.solution - rhs)
        / max(np.linalg.norm(rhs), np.finfo(float).tiny)
    )
    boundary_power = {
        face: float(np.sum(conductance * (solved.solution[ids] - bath)))
        for face, (ids, conductance, bath) in system.boundary_terms.items()
    }
    source_power = float(np.sum(system.source_volume_operator_m3 @ source_active))
    energy_error = abs(sum(boundary_power.values()) - source_power) / max(
        abs(source_power), max(abs(value) for value in boundary_power.values())
    )

    flake_iz = np.flatnonzero((z >= 0.0) & (z < 0.100e-6))
    flake_weights = dz[flake_iz] / np.sum(dz[flake_iz])
    temperature_flake = np.tensordot(temperature[:, :, flake_iz], flake_weights, axes=(2, 0))
    grad_b, grad_a = np.gradient(temperature_flake, x, y, edge_order=2)
    gradient_magnitude = np.hypot(grad_b, grad_a)

    span_b = float(x_edges[-1] - x_edges[0])
    span_a = float(y_edges[-1] - y_edges[0])
    psi = np.broadcast_to((y[None, :] - y_edges[0]) / span_a, temperature_flake.shape).copy()
    grad_psi_b = np.zeros_like(psi)
    grad_psi_a = np.full_like(psi, 1.0 / span_a)
    weighting_b = -grad_psi_b
    weighting_a = -grad_psi_a
    sigma_b, sigma_a = 1.10e5, 4.91e5
    seebeck_b, seebeck_a = 27.0e-6, -6.0e-6
    thickness = 100.0e-9
    j_pte_b = -sigma_b * seebeck_b * grad_b
    j_pte_a = -sigma_a * seebeck_a * grad_a
    current_integrand = thickness * (j_pte_b * grad_psi_b + j_pte_a * grad_psi_a)
    current = float(np.sum(current_integrand * dx[:, None] * dy[None, :]))
    conductance = sigma_a * thickness * span_b / span_a
    open_circuit_voltage = -current / conductance

    strict = np.ones_like(temperature_flake, bool)
    strict[[0, -1], :] = False
    strict[:, [0, -1]] = False
    grad_b_strict = np.where(strict, grad_b, np.nan)
    grad_a_strict = np.where(strict, grad_a, np.nan)
    gradient_magnitude_strict = np.where(strict, gradient_magnitude, np.nan)
    integrand_strict = np.where(strict, current_integrand, np.nan)

    np.savez_compressed(
        output_npz,
        x_m=x,
        y_m=y,
        z_m=z,
        x_edges_m=x_edges,
        y_edges_m=y_edges,
        z_edges_m=z_edges,
        Q_unit_W_m3=q_unit,
        Q_285uW_W_m3=q_report,
        temperature_3d_K=temperature,
        temperature_flake_K=temperature_flake,
        grad_b_K_m=grad_b_strict,
        grad_a_K_m=grad_a_strict,
        gradient_magnitude_K_m=gradient_magnitude_strict,
        weighting_potential=psi,
        weighting_field_b_m_inv=weighting_b,
        weighting_field_a_m_inv=weighting_a,
        J_PTE_b_A_m2=j_pte_b,
        J_PTE_a_A_m2=j_pte_a,
        terminal_current_integrand_A_m2=integrand_strict,
    )

    gates = {
        "Q_mapping_error_lt_0p5pct": mapping_error < 0.005,
        "thermal_residual_lt_1e_8": residual < 1.0e-8,
        "thermal_energy_balance_lt_1pct": energy_error < 0.01,
        "finite_fields": bool(
            np.all(np.isfinite(temperature))
            and np.all(np.isfinite(psi))
            and np.all(np.isfinite(current_integrand))
        ),
    }
    payload = {
        "status": "VALIDATED_LARGE_SHEET_DIAGNOSTIC_THERMAL_WEIGHTING_PTE" if all(gates.values()) else "FAILED_LARGE_SHEET_DIAGNOSTIC_GATE",
        "classification": "large finite computational sheet with ideal full-width y-edge contacts; not experimental finite-contact prediction",
        "input_Q": {
            "path": str(raw_q),
            "sha256": actual_sha,
            "certificate": str(q_json),
            "certificate_sha256": sha256(q_json),
            "polarization": args.polarization,
        },
        "axis_mapping": "x=b, y=a, z=c",
        "geometry": {
            "lateral_bounds_um": {"x_b": [x_edges[0] * 1e6, x_edges[-1] * 1e6], "y_a": [y_edges[0] * 1e6, y_edges[-1] * 1e6]},
            "stack": "air / finite Au T array / laterally extended TaIrTe4 100nm / Al2O3 35nm / Au mirror 200nm / SiO2 285nm optical closure / Si",
            "thermal_cells": list(system.shape),
            "material_cell_counts": material_counts,
        },
        "thermal_materials_W_mK": {
            "air": 0.026,
            "Au": 317.0,
            "Al2O3_scenario": 1.5,
            "TaIrTe4_x_b_y_a_z_c": [3.8, 14.4, 1.0],
            "SiO2": 1.4,
            "Si": 148.0,
        },
        "interface_model": {"SiO2_Si_G_W_m2K": 1.1e9, "all_other_interfaces": "perfect-contact diagnostic assumption"},
        "boundaries": {"x_y_sides": "adiabatic", "top": "adiabatic", "Si_bottom": "fixed DeltaT=0"},
        "illumination": {
            "reported_incident_power_W": REPORT_INCIDENT_POWER_W,
            "certified_source_power_W": source_power_w,
            "linear_scale_factor": source_scale,
            "note": "certified linear incident-power scaling; no clipping/smoothing/gain/shape rescaling",
        },
        "Q": {
            "source_component_power_W": source_component_power,
            "target_component_power_W": target_component_power,
            "unit_source_total_W": p_source_unit,
            "mapped_unit_total_W": p_target_unit,
            "mapping_relative_error": mapping_error,
            "absorbed_power_at_285uW_W": source_power,
        },
        "thermal": {
            "Tmax_K": float(np.max(temperature_flake)),
            "Tavg_flake_K": float(
                np.sum(temperature_flake * dx[:, None] * dy[None, :])
                / np.sum(dx[:, None] * dy[None, :])
            ),
            "max_gradient_K_m": float(np.nanmax(gradient_magnitude_strict)),
            "linear_residual_relative": residual,
            "energy_balance_relative_error": energy_error,
            "boundary_power_out_W": boundary_power,
            "assembly_seconds": assembly_seconds,
            "CUDA_device": args.cuda_device,
            "CUDA_PCG_iterations": solved.iterations,
            "CUDA_solve_seconds": solved.solve_seconds,
        },
        "electrical": {
            "contact_0": "full-width y_min, psi=0",
            "contact_1": "full-width y_max, psi=1",
            "weighting_potential": "analytic exact solution psi=(y-ymin)/(ymax-ymin) for homogeneous rectangular sheet",
            "sigma_x_b_y_a_S_m": [sigma_b, sigma_a],
            "Seebeck_x_b_y_a_V_K": [seebeck_b, seebeck_a],
            "terminal_conductance_S": conductance,
            "short_circuit_current_A": current,
            "open_circuit_voltage_V": open_circuit_voltage,
        },
        "gates": gates,
        "raw_artifacts": [],
    }
    payload["raw_artifacts"] = [
        {"path": str(output_npz), "size_bytes": output_npz.stat().st_size, "sha256": sha256(output_npz)}
    ]
    output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
