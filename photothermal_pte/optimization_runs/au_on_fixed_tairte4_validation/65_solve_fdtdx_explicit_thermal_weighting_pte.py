#!/usr/bin/env python3
"""Solve explicit 3-D thermal + Au-aware weighting PTE from remapped FDTDX Q."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photothermal_pte.optimization_runs.cuda_thermal_adjoint import PersistentCudaCSR
from photothermal_pte.validation.photothermal_stage1.anisotropic_heat_fvm import (
    assemble_steady_diagonal_kappa,
)


HERE = Path(__file__).resolve().parent
STAGE54 = HERE / "54_validate_au_weighting_electrical_adfd.py"
STAGE62 = HERE / "62_validate_coupled_au_thermal_weighting_pte_adfd.py"
STAGE64 = HERE / "64_validate_fdtdx_material_overlap_thermal_remap.py"
TOPOLOGY_THERMAL = HERE.parent / "tairte4_flake_topology" / "thermal.py"
K_AIR_W_MK = 0.026
K_SIO2_W_MK = 1.38
K_SI_W_MK = 145.0
K_TA_XYZ_W_MK = np.asarray((3.8, 14.4, 1.0), dtype=np.float64)
K_AU_W_MK = 317.0
G_SIO2_SI_W_M2K = 1.1e9
G_TA_AIR_W_M2K = 1.0
G_TA_SIO2_SCENARIOS = {
    "thermally_grown": 7.37e6,
    "evaporated": 7.37e4,
}
G_AU_TA_W_M2K = 1.0 / 5.8e-8
TOP_AIR_CONVECTION_W_M2K = 10.0
ELECTRICAL_CONTACT_S_M2 = 1.0e10


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), np.finfo(float).tiny)


def _face_before(edges: np.ndarray, value: float) -> int:
    match = np.flatnonzero(np.isclose(edges, value, rtol=0.0, atol=2.0e-18))
    if match.size != 1 or match[0] == 0:
        raise RuntimeError(f"Required thermal face {value:.9e} m is absent")
    return int(match[0] - 1)


def _centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def _thermal_state(rho: np.ndarray, g_ta_sio2: float, topology, fvm):
    edges = topology._piecewise_edges()
    widths = tuple(np.diff(axis) for axis in edges)
    centers = tuple(_centers(axis) for axis in edges)
    shape = tuple(len(axis) for axis in centers)
    x, y, z = centers
    x_ta = (x >= -10.0e-6) & (x < 10.0e-6)
    y_ta = (y >= -10.0e-6) & (y < 10.0e-6)
    z_ta = (z >= -0.1e-6) & (z < 0.0)
    x_au = (x >= -5.0e-6) & (x < 5.0e-6)
    y_au = (y >= -5.0e-6) & (y < 5.0e-6)
    z_au = (z >= 0.0) & (z < 0.05e-6)
    z_sio2 = (z >= -0.385e-6) & (z < -0.1e-6)
    z_si = z < -0.385e-6
    ta_mask = x_ta[:, None, None] & y_ta[None, :, None] & z_ta[None, None, :]
    au_mask = x_au[:, None, None] & y_au[None, :, None] & z_au[None, None, :]
    sio2_mask = np.broadcast_to(z_sio2[None, None, :], shape)
    si_mask = np.broadcast_to(z_si[None, None, :], shape)

    ix_au = np.flatnonzero(x_au)
    iy_au = np.flatnonzero(y_au)
    iz_au = np.flatnonzero(z_au)
    if (len(ix_au), len(iy_au)) != (100, 100):
        raise RuntimeError("Au thermal footprint is not 100x100 at 100 nm")
    if rho.shape != (20, 20):
        raise ValueError("Expected the 20x20 optical Au density")
    rho_100nm = np.repeat(np.repeat(rho, 5, axis=0), 5, axis=1)
    k_au_effective = K_AIR_W_MK + rho_100nm * (K_AU_W_MK - K_AIR_W_MK)

    kappa = np.full((*shape, 3), K_AIR_W_MK, dtype=np.float64)
    kappa[si_mask] = K_SI_W_MK
    kappa[sio2_mask] = K_SIO2_W_MK
    kappa[ta_mask] = K_TA_XYZ_W_MK
    for iz in iz_au:
        for component in range(3):
            kappa[np.ix_(ix_au, iy_au, [iz], [component])] = (
                k_au_effective[:, :, None, None]
            )

    rx = np.zeros((shape[0] - 1, shape[1], shape[2]), dtype=np.float64)
    ry = np.zeros((shape[0], shape[1] - 1, shape[2]), dtype=np.float64)
    rz = np.zeros((shape[0], shape[1], shape[2] - 1), dtype=np.float64)
    sio2_si_face = _face_before(edges[2], -0.385e-6)
    ta_sio2_face = _face_before(edges[2], -0.1e-6)
    ta_top_face = _face_before(edges[2], 0.0)
    rz[:, :, sio2_si_face] = 1.0 / G_SIO2_SI_W_M2K
    rz[np.ix_(np.flatnonzero(x_ta), np.flatnonzero(y_ta), [ta_sio2_face])] = (
        1.0 / g_ta_sio2
    )
    # Outside the Au design footprint, TaIrTe4 sees the paper-derived air TBC.
    ta_top = np.ix_(np.flatnonzero(x_ta), np.flatnonzero(y_ta), [ta_top_face])
    rz[ta_top] = 1.0 / G_TA_AIR_W_M2K

    lower_dz = widths[2][ta_top_face]
    upper_dz = widths[2][ta_top_face + 1]
    r_air = (
        0.5 * lower_dz / K_TA_XYZ_W_MK[2]
        + 1.0 / G_TA_AIR_W_M2K
        + 0.5 * upper_dz / K_AIR_W_MK
    )
    r_au = (
        0.5 * lower_dz / K_TA_XYZ_W_MK[2]
        + 1.0 / G_AU_TA_W_M2K
        + 0.5 * upper_dz / K_AU_W_MK
    )
    conductance_per_area = (1.0 - rho_100nm) / r_air + rho_100nm / r_au
    r_equivalent_interface = (
        1.0 / conductance_per_area
        - 0.5 * lower_dz / K_TA_XYZ_W_MK[2]
        - 0.5 * upper_dz / k_au_effective
    )
    if np.any(r_equivalent_interface < -1.0e-15):
        raise RuntimeError("Au/Ta parallel-area equivalent interface became negative")
    rz[np.ix_(ix_au, iy_au, [ta_top_face])] = np.maximum(
        r_equivalent_interface, 0.0
    )[:, :, None]

    system = fvm.assemble_steady_diagonal_kappa(
        x_edges_m=edges[0],
        y_edges_m=edges[1],
        z_edges_m=edges[2],
        kappa_W_mK=kappa,
        active_mask=np.ones(shape, dtype=bool),
        interface_resistance_m2K_W={"x": rx, "y": ry, "z": rz},
        dirichlet_temperature_K={
            "x_min": 0.0,
            "x_max": 0.0,
            "y_min": 0.0,
            "y_max": 0.0,
            "z_min": 0.0,
        },
        surface_robin_heat_transfer_W_m2K={"z_max": TOP_AIR_CONVECTION_W_M2K},
        surface_robin_temperature_K={"z_max": 0.0},
    )
    return {
        "edges": edges,
        "widths": widths,
        "centers": centers,
        "system": system,
        "kappa": kappa,
        "masks": {"au": au_mask, "tairte4": ta_mask, "sio2": sio2_mask, "si": si_mask},
        "rho_100nm": rho_100nm,
        "interface_resistance": {"x": rx, "y": ry, "z": rz},
        "faces": {
            "SiO2_Si": sio2_si_face,
            "TaIrTe4_SiO2": ta_sio2_face,
            "TaIrTe4_Au_or_air": ta_top_face,
        },
    }


def _map_thermal_q(remap: np.lib.npyio.NpzFile, state: dict, overlap):
    full_power = np.zeros(state["system"].shape, dtype=np.float64)
    records = {}
    for material in ("au", "tairte4", "sio2"):
        source_q = np.asarray(remap[f"Q_{material}_thermal_W_m3"], dtype=np.float64)
        source_edges = tuple(
            np.asarray(remap[f"{material}_{axis}_edges_m"], dtype=np.float64)
            for axis in "xyz"
        )
        source_volume = (
            np.diff(source_edges[0])[:, None, None]
            * np.diff(source_edges[1])[None, :, None]
            * np.diff(source_edges[2])[None, None, :]
        )
        material_mask = state["masks"][material]
        material_indices = (
            np.flatnonzero(np.any(material_mask, axis=(1, 2))),
            np.flatnonzero(np.any(material_mask, axis=(0, 2))),
            np.flatnonzero(np.any(material_mask, axis=(0, 1))),
        )
        if any(
            not np.array_equal(index, np.arange(index[0], index[-1] + 1))
            for index in material_indices
        ):
            raise RuntimeError(f"{material} mask is not one rectangular cell block")
        material_edges = tuple(
            state["edges"][axis][index[0] : index[-1] + 2]
            for axis, index in enumerate(material_indices)
        )
        operators = tuple(
            overlap._overlap_operator(
                0.5 * (source_edges[axis][:-1] + source_edges[axis][1:]),
                np.diff(source_edges[axis]),
                material_edges[axis],
            )[0]
            for axis in range(3)
        )
        source_power = source_q * source_volume
        material_power = overlap._forward(source_power, operators)
        mapped_power = np.zeros_like(full_power)
        mapped_power[np.ix_(*material_indices)] = material_power
        outside = float(np.sum(mapped_power[~material_mask]))
        total = float(np.sum(material_power))
        if outside != 0.0:
            raise RuntimeError(f"{material} remap deposited power outside material: {outside}")
        full_power += mapped_power
        records[material] = {
            "source_power_W": float(np.sum(source_power)),
            "mapped_power_W": total,
            "relative_error": _relative(float(np.sum(source_power)), total),
            "power_outside_material_W": outside,
        }
    q_full = full_power / state["system"].cell_volume_m3
    return q_full, full_power, records


def _boundary_energy(state: dict, temperature: np.ndarray, source_power: np.ndarray):
    powers = {
        name: float(np.sum(conductance * temperature[cell_ids]))
        for name, (cell_ids, conductance, _) in state["system"].boundary_terms.items()
    }
    source = float(np.sum(source_power))
    error = abs(sum(powers.values()) - source) / max(abs(source), np.finfo(float).tiny)
    return error, powers


def _ta_temperature_500nm(state: dict, full_temperature: np.ndarray) -> np.ndarray:
    x, y, z = state["centers"]
    ix = np.flatnonzero((x >= -10e-6) & (x < 10e-6))
    iy = np.flatnonzero((y >= -10e-6) & (y < 10e-6))
    iz = np.flatnonzero((z >= -0.1e-6) & (z < 0.0))
    weights = state["widths"][2][iz]
    average_z = np.tensordot(
        full_temperature[np.ix_(ix, iy, iz)], weights / np.sum(weights), axes=(2, 0)
    )
    if average_z.shape != (200, 200):
        raise RuntimeError(f"Unexpected Ta temperature shape: {average_z.shape}")
    return average_z.reshape(40, 5, 40, 5).mean(axis=(1, 3))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remap-summary-json", required=True, type=Path)
    parser.add_argument("--raw-remap-npz", required=True, type=Path)
    parser.add_argument("--raw-spatial-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--raw-output-npz", required=True, type=Path)
    parser.add_argument(
        "--ta-sio2-scenario",
        choices=tuple(G_TA_SIO2_SCENARIOS),
        required=True,
    )
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("GPU-only solve requires CUDA_VISIBLE_DEVICES")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    remap_summary_path = args.remap_summary_json.resolve()
    remap_path = args.raw_remap_npz.resolve()
    spatial_path = args.raw_spatial_npz.resolve()
    raw_output_path = args.raw_output_npz.resolve()
    remap_summary = json.loads(remap_summary_path.read_text(encoding="utf-8"))
    if _sha256(remap_path) != remap_summary["output_thermal_Q"]["sha256"]:
        raise RuntimeError("Fail-closed: thermal-Q remap SHA mismatch")
    if _sha256(spatial_path) != remap_summary["input_spatial_Q"]["sha256"]:
        raise RuntimeError("Fail-closed: source spatial-Q SHA mismatch")

    topology = _load(TOPOLOGY_THERMAL, "au_stage65_topology_thermal")
    fvm = _load(
        Path(__file__).parents[2]
        / "validation"
        / "photothermal_stage1"
        / "anisotropic_heat_fvm.py",
        "au_stage65_fvm",
    )
    overlap = _load(STAGE64, "au_stage65_overlap")
    electrical = _load(STAGE54, "au_stage65_electrical")
    coupled = _load(STAGE62, "au_stage65_coupled")
    with np.load(spatial_path, allow_pickle=False) as spatial:
        rho = np.asarray(spatial["rho"], dtype=np.float64)
    assembly_start = perf_counter()
    state = _thermal_state(
        rho, G_TA_SIO2_SCENARIOS[args.ta_sio2_scenario], topology, fvm
    )
    assembly_seconds = perf_counter() - assembly_start
    with np.load(remap_path, allow_pickle=False) as remap:
        q_full, source_power, mapping = _map_thermal_q(remap, state, overlap)

    solve_start = perf_counter()
    operator = PersistentCudaCSR(
        state["system"].matrix_W_K, cuda_device=args.cuda_device
    )
    solution = operator.solve(
        source_power.reshape(-1),
        relative_tolerance=1.0e-9,
        max_iterations=30000,
        residual_check_interval=25,
    )
    solve_seconds = perf_counter() - solve_start
    temperature = solution.solution
    full_temperature = temperature.reshape(state["system"].shape)
    energy_error, boundary_power = _boundary_energy(state, temperature, source_power.reshape(-1))

    ta_temperature_500 = _ta_temperature_500nm(state, full_temperature)
    electrical_base = electrical.build_system(rho, ELECTRICAL_CONTACT_S_M2)
    electrical_system = replace(
        electrical_base,
        objective_gradient_psi_A=coupled.electrical_load(
            ta_temperature_500, electrical
        ),
    )
    psi, _, electrical_solver = electrical.solve_gpu(
        electrical_system, args.cuda_device, need_adjoint=False
    )
    current = electrical.objective(electrical_system, psi)
    electrical_audit = electrical.audit(electrical_system, psi)

    masks = state["masks"]
    temperatures = {
        material: {
            "maximum_K": float(np.max(full_temperature[mask])),
            "volume_average_K": float(
                np.sum(full_temperature[mask] * state["system"].cell_volume_m3[mask])
                / np.sum(state["system"].cell_volume_m3[mask])
            ),
        }
        for material, mask in masks.items()
    }
    gates = {
        "input_SHAs_match": True,
        "material_Q_mapping_lt_1e-12": max(
            record["relative_error"] for record in mapping.values()
        ) < 1.0e-12,
        "linear_residual_lt_1e-8": solution.explicit_relative_residual < 1.0e-8,
        "thermal_energy_balance_lt_1pct": energy_error < 0.01,
        "electrical_residual_lt_1e-8": electrical_solver[
            "weighting_relative_residual"
        ] < 1.0e-8,
        "electrical_terminal_balance_lt_1pct": electrical_audit[
            "terminal_current_balance"
        ] < 0.01,
        "finite_temperature_and_weighting": bool(
            np.all(np.isfinite(temperature)) and np.all(np.isfinite(psi))
        ),
        "GPU_linear_solve_no_CPU_fallback": True,
        "no_Q_clipping_smoothing_gain_or_global_rescaling": True,
    }
    passed = all(gates.values())
    status = (
        "VALIDATED_FDTDX_EXPLICIT_THERMAL_AU_AWARE_WEIGHTING_PTE_FORWARD"
        if passed
        else "FAILED_FDTDX_EXPLICIT_THERMAL_AU_AWARE_WEIGHTING_PTE_FORWARD"
    )

    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        raw_output_path,
        temperature_K=full_temperature.astype(np.float32),
        q_full_W_m3=q_full.astype(np.float32),
        rho=rho.astype(np.float32),
        ta_temperature_500nm_K=ta_temperature_500.astype(np.float32),
        weighting_potential=psi.astype(np.float32),
        x_edges_m=state["edges"][0],
        y_edges_m=state["edges"][1],
        z_edges_m=state["edges"][2],
    )
    raw_sha = _sha256(raw_output_path)

    x, y, z = state["centers"]
    ix_ta = np.flatnonzero((x >= -10e-6) & (x < 10e-6))
    iy_ta = np.flatnonzero((y >= -10e-6) & (y < 10e-6))
    iz_ta = np.flatnonzero((z >= -0.1e-6) & (z < 0.0))
    ta_weights = state["widths"][2][iz_ta]
    ta_temperature_100 = np.tensordot(
        full_temperature[np.ix_(ix_ta, iy_ta, iz_ta)],
        ta_weights / np.sum(ta_weights),
        axes=(2, 0),
    )
    y_mid = int(np.argmin(np.abs(y)))
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    image = axes[0, 0].imshow(
        ta_temperature_100.T,
        origin="lower",
        extent=(-10, 10, -10, 10),
    )
    axes[0, 0].set_title("TaIrTe4 thickness-averaged temperature rise")
    axes[0, 0].set_xlabel("x=b (um)")
    axes[0, 0].set_ylabel("y=a (um)")
    fig.colorbar(image, ax=axes[0, 0], label="K")
    image = axes[0, 1].pcolormesh(
        state["edges"][0] * 1e6,
        state["edges"][2] * 1e6,
        full_temperature[:, y_mid, :].T,
        shading="flat",
    )
    axes[0, 1].set_ylim(-1.0, 0.5)
    axes[0, 1].set_title("x-z temperature at y=0")
    axes[0, 1].set_xlabel("x=b (um)")
    axes[0, 1].set_ylabel("z (um)")
    fig.colorbar(image, ax=axes[0, 1], label="K")
    image = axes[1, 0].imshow(
        psi[: electrical.N_TA * electrical.N_TA].reshape(electrical.N_TA, electrical.N_TA).T,
        origin="lower",
        extent=(-10, 10, -10, 10),
        vmin=0.0,
        vmax=1.0,
    )
    axes[1, 0].set_title("Au-aware TaIrTe4 weighting potential")
    axes[1, 0].set_xlabel("x=b (um)")
    axes[1, 0].set_ylabel("y=a (um)")
    fig.colorbar(image, ax=axes[1, 0], label="psi")
    axes[1, 1].bar(boundary_power.keys(), np.asarray(list(boundary_power.values())) * 1e15)
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].set_ylabel("outward boundary power (fW)")
    axes[1, 1].set_title("Thermal energy paths (numerical boundaries)")
    field_plot = output / "fdtdx_explicit_thermal_weighting_fields.png"
    fig.savefig(field_plot, dpi=180)
    plt.close(fig)

    summary = {
        "status": status,
        "scope": (
            "one FDTDX Maxwell-Q explicit 3-D Au/TaIrTe4/SiO2/Si thermal forward "
            "and Au-aware floating-layer weighting/PTE forward; no combined gradient, "
            "AD-FD, or optimization"
        ),
        "scenario": args.ta_sio2_scenario,
        "parameters": {
            "k_air_W_mK": K_AIR_W_MK,
            "k_SiO2_W_mK": K_SIO2_W_MK,
            "k_Si_W_mK": K_SI_W_MK,
            "k_TaIrTe4_xyz_xb_ya_z_W_mK": K_TA_XYZ_W_MK.tolist(),
            "k_Au_W_mK": K_AU_W_MK,
            "G_SiO2_Si_W_m2K": G_SIO2_SI_W_M2K,
            "G_TaIrTe4_SiO2_W_m2K": G_TA_SIO2_SCENARIOS[args.ta_sio2_scenario],
            "G_TaIrTe4_air_W_m2K": G_TA_AIR_W_M2K,
            "G_Au_TaIrTe4_W_m2K": G_AU_TA_W_M2K,
            "G_Au_TaIrTe4_provenance": (
                "Au/MoS2 calculated analogue; named numerical scenario, not TaIrTe4 data"
            ),
            "electrical_contact_S_m2": ELECTRICAL_CONTACT_S_M2,
            "electrical_contact_provenance": (
                "named numerical scenario, not measured Au/TaIrTe4 contact"
            ),
            "S_Au_V_K": 0.0,
        },
        "geometry": {
            "thermal_shape": list(state["system"].shape),
            "thermal_lateral_bounds_m": [
                state["edges"][0][0], state["edges"][0][-1],
                state["edges"][1][0], state["edges"][1][-1],
            ],
            "thermal_z_bounds_m": [state["edges"][2][0], state["edges"][2][-1]],
            "TaIrTe4_footprint_m": [20e-6, 20e-6],
            "TaIrTe4_thickness_m": 100e-9,
            "Au_design_footprint_m": [10e-6, 10e-6],
            "Au_thickness_m": 50e-9,
        },
        "source": {
            "remap_summary": str(remap_summary_path),
            "P_Q_W": float(np.sum(source_power)),
            "literal_FDTDX_normalization": True,
            "scaled_to_experimental_power": False,
            "mapping": mapping,
        },
        "thermal": {
            "Tmax_rise_K": float(np.max(temperature)),
            "material_temperature": temperatures,
            "linear_residual_relative": float(solution.explicit_relative_residual),
            "iterations": int(solution.iterations),
            "energy_balance_relative": energy_error,
            "boundary_power_out_W": boundary_power,
            "assembly_seconds": assembly_seconds,
            "GPU_solve_seconds": solve_seconds,
        },
        "electrical_PTE": {
            "current_A": current,
            "terminal_axis": "y=a; low y=-10um, high y=+10um",
            "weighting_residual_relative": electrical_solver[
                "weighting_relative_residual"
            ],
            "terminal_balance_relative": electrical_audit[
                "terminal_current_balance"
            ],
            "psi_min": electrical_audit["psi_min"],
            "psi_max": electrical_audit["psi_max"],
        },
        "gates": gates,
        "raw_artifact": {
            "path": str(raw_output_path),
            "bytes": raw_output_path.stat().st_size,
            "sha256": raw_sha,
            "committed_to_git": False,
        },
        "next_gate": (
            "differentiate the same Maxwell-Q remap + explicit thermal + Au-aware "
            "electrical chain and run combined directional AD-FD"
        ),
    }
    summary_path = output / "fdtdx_explicit_thermal_weighting_pte_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path = output / "FDTDX_EXPLICIT_THERMAL_WEIGHTING_PTE_REPORT.md"
    report_path.write_text(
        f"""# FDTDX explicit thermal and Au-aware weighting/PTE forward

Status: **{status}**

Scenario: **{args.ta_sio2_scenario}** TaIrTe4/SiO2 contact with
`G={G_TA_SIO2_SCENARIOS[args.ta_sio2_scenario]:.6e} W/(m2 K)`.

The validated spatial Au+TaIrTe4+SiO2 Maxwell source is conservatively placed
in the explicit 3-D Au/TaIrTe4/SiO2/Si FVM. The literal source power is
`{np.sum(source_power):.12e} W`; no experimental-power scaling is applied.
The GPU solve gives `Tmax={np.max(temperature):.12e} K`, residual
`{solution.explicit_relative_residual:.3e}`, and energy-balance error
`{100*energy_error:.6f}%`.

The thickness-averaged TaIrTe4 temperature is then passed to the already
validated two-layer electrical operator. The Au topology changes lateral Au
conductance, finite Au/TaIrTe4 contact, and therefore the weighting potential.
Au thermopower is zero in this control. The resulting literal-normalization
PTE current is `{current:.12e} A`; electrical residual is
`{electrical_solver['weighting_relative_residual']:.3e}`.

`G_Au/TaIrTe4={G_AU_TA_W_M2K:.6e} W/(m2 K)` is an Au/MoS2 analogue and the
electrical contact is a numerical scenario. Neither is promoted as measured
TaIrTe4 data. The gray Au/air layer uses an area-fraction thermal/electrical
relaxation and is not claimed to be a fabricated effective medium.

This validates a forward chain only. It is not yet a combined Maxwell+
thermal+electrical gradient certificate and does not authorize optimization.
""",
        encoding="utf-8",
    )
    manifest = {
        "status": status,
        "raw_artifact": summary["raw_artifact"],
        "input_remap_raw": remap_summary["output_thermal_Q"],
        "published": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (summary_path, report_path, field_plot)
        ],
    }
    (output / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
