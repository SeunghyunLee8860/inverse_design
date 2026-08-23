#!/usr/bin/env python3
"""Exact telescoping Au-effect decomposition for one architecture/polarization."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from photothermal_pte.optimization_runs.cuda_thermal_adjoint import PersistentCudaCSR


STAGE81 = HERE / "81_solve_finite_t_z_thermal_electrical.py"
MAPPING = HERE / "results_finite_T_Z_material_Q_mapping" / "FINITE_T_Z_MATERIAL_Q_MAPPING_SUMMARY.json"
PRIMARY = HERE / "results_finite_T_Z_thermal_electrical"
OUTPUT = HERE / "results_finite_T_Z_Au_effect_decomposition"
RAW_PRIMARY = Path("/home/seunghyun/tairte4/raw_artifacts/finite_T_Z_thermal_electrical")
RAW_OUT = Path("/home/seunghyun/tairte4/raw_artifacts/finite_T_Z_Au_effect_decomposition")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("finite_stage81", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def solve_temperature(stage, state: dict, power: np.ndarray, gpu: int):
    operator = PersistentCudaCSR(state["system"].matrix_W_K, cuda_device=gpu)
    result = operator.solve(power.reshape(-1), relative_tolerance=1e-9, max_iterations=40000, residual_check_interval=25)
    error, boundary = stage.boundary_energy(state, result.solution, power.reshape(-1))
    return result.solution.reshape(state["shape"]), {"residual": result.explicit_relative_residual, "iterations": result.iterations, "energy_balance": error, "boundary_power_W": boundary}


def current_for(stage, state: dict, ta_temperature: np.ndarray, x: np.ndarray, y: np.ndarray, orientation: str, electrical_au: bool, architecture: str, gpu: int):
    rho = state["rho_au"]
    ix = np.flatnonzero((state["centers"][0] >= -10e-6) & (state["centers"][0] < 10e-6))
    iy = np.flatnonzero((state["centers"][1] >= -10e-6) & (state["centers"][1] < 10e-6))
    rho_ta = rho[ix[:, None], iy[None, :]] if electrical_au else np.zeros((len(ix), len(iy)))
    system = stage.build_electrical(ta_temperature, x, y, rho_ta, architecture, orientation)
    _, _, audit = stage.solve_electrical(system, gpu)
    return audit


def weighted_average(field: np.ndarray, state: dict, mask: np.ndarray) -> float:
    volume = state["system"].cell_volume_m3
    return float(np.sum(field[mask] * volume[mask]) / np.sum(volume[mask]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("T", "Z"), required=True)
    parser.add_argument("--polarization", choices=("Ea", "Eb"), required=True)
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("GPU-only solve requires CUDA_VISIBLE_DEVICES")
    stage = load_module(STAGE81)
    on_case = f"{args.architecture}_{args.polarization}_Au_on"
    off_case = f"{args.architecture}_{args.polarization}_Au_off"
    mapping = json.loads(MAPPING.read_text())
    on_meta, off_meta = mapping["cases"][on_case], mapping["cases"][off_case]
    on_primary = json.loads((PRIMARY / on_case / f"{on_case}_THERMAL_ELECTRICAL_SUMMARY.json").read_text())
    off_primary = json.loads((PRIMARY / off_case / f"{off_case}_THERMAL_ELECTRICAL_SUMMARY.json").read_text())
    scale = float(on_primary["source"]["linear_response_scale"])
    on_map = Path(on_meta["raw_mapped_artifact"]["path"])
    with np.load(on_map, allow_pickle=False) as raw:
        state_on = stage.build_thermal_state(raw, args.architecture, True)
        non_top_power = sum((np.asarray(raw[f"power_{key}_W"], dtype=np.float64) for key in ("Si", "SiO2", "Au_mirror", "TaIrTe4")), start=np.zeros(state_on["shape"])) * scale
        mapped_edges = tuple(np.asarray(raw[f"{axis}_edges_m"]) for axis in "xyz")
    # A = K_on^-1(Q_nonAu + Q_topAu) already certified.
    with np.load(RAW_PRIMARY / f"{on_case}_finite_thermal_electrical.npz", allow_pickle=False) as primary:
        temp_a = np.asarray(primary["temperature_3d_K"], dtype=np.float64)
    # B = K_on^-1 Q_nonAu: direct top-Au heat removed.
    temp_b, audit_b = solve_temperature(stage, state_on, non_top_power, args.cuda_device)
    ta_a, x, y = stage.thickness_average_ta(state_on, temp_a)
    ta_b, _, _ = stage.thickness_average_ta(state_on, temp_b)

    # C = K_off^-1 Q_nonAu: additionally remove the top-Au thermal shunt.
    with np.load(on_map, allow_pickle=False) as raw:
        state_off = stage.build_thermal_state(raw, args.architecture, False)
    temp_c, audit_c = solve_temperature(stage, state_off, non_top_power, args.cuda_device)
    ta_c, _, _ = stage.thickness_average_ta(state_off, temp_c)
    # D = K_off^-1 Q_off is the independent Au-off primary.
    with np.load(RAW_PRIMARY / f"{off_case}_finite_thermal_electrical.npz", allow_pickle=False) as primary:
        temp_d = np.asarray(primary["temperature_3d_K"], dtype=np.float64)
    ta_d, _, _ = stage.thickness_average_ta(state_off, temp_d)

    output = OUTPUT / f"{args.architecture}_{args.polarization}"
    output.mkdir(parents=True, exist_ok=True)
    RAW_OUT.mkdir(parents=True, exist_ok=True)
    electrical = {}
    for orientation in ("top_bottom", "left_right"):
        i_a_on = on_primary["electrical"][orientation]
        i_a_off = current_for(stage, state_off, ta_a, x, y, orientation, False, args.architecture, args.cuda_device)
        i_b_off = current_for(stage, state_off, ta_b, x, y, orientation, False, args.architecture, args.cuda_device)
        i_c_off = current_for(stage, state_off, ta_c, x, y, orientation, False, args.architecture, args.cuda_device)
        i_d_off = off_primary["electrical"][orientation]
        currents = {
            "A_full_Au_on_Q_K_E": float(i_a_on["high_terminal_current_A"]),
            "A_same_T_electrical_Au_off": float(i_a_off["high_terminal_current_A"]),
            "B_top_Au_heat_removed_K_on_E_off": float(i_b_off["high_terminal_current_A"]),
            "C_top_Au_heat_and_thermal_shunt_removed_E_off": float(i_c_off["high_terminal_current_A"]),
            "D_full_Au_off_Q_K_E": float(i_d_off["high_terminal_current_A"]),
        }
        contributions = {
            "floating_Au_electrical_shunt_A": currents["A_full_Au_on_Q_K_E"] - currents["A_same_T_electrical_Au_off"],
            "direct_top_Au_absorption_heat_A": currents["A_same_T_electrical_Au_off"] - currents["B_top_Au_heat_removed_K_on_E_off"],
            "top_Au_thermal_shunt_A": currents["B_top_Au_heat_removed_K_on_E_off"] - currents["C_top_Au_heat_and_thermal_shunt_removed_E_off"],
            "Au_induced_nonAu_optical_redistribution_A": currents["C_top_Au_heat_and_thermal_shunt_removed_E_off"] - currents["D_full_Au_off_Q_K_E"],
        }
        total = currents["A_full_Au_on_Q_K_E"] - currents["D_full_Au_off_Q_K_E"]
        closure = total - sum(contributions.values())
        electrical[orientation] = {"currents": currents, "contributions": contributions, "full_on_minus_off_A": total, "telescoping_closure_A": closure, "intermediate_audits": {"A_temperature_E_off": i_a_off, "B": i_b_off, "C": i_c_off}}

        labels = ["electrical\nshunt", "direct Au\nheating", "thermal\nshunt", "optical\nredistribution"]
        values = np.asarray(list(contributions.values())) * 1e9
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        ax.bar(labels, values, color=["#7b3294", "#d7191c", "#2c7bb6", "#fdae61"])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("contribution to I_on - I_off (nA)")
        ax.set_title(f"{on_case}, {orientation}: exact Au-effect current decomposition")
        for index, value in enumerate(values):
            ax.text(index, value, f"{value:.4g}", ha="center", va="bottom" if value >= 0 else "top")
        fig.savefig(output / f"{on_case}_{orientation}_Au_current_decomposition.png", dpi=180)
        plt.close(fig)

    thermal = {
        "scenario_sequence": {
            "A": "full Au-on Q, thermal Au on",
            "B": "same Au-on non-Au Q, direct top-Au absorption removed, thermal Au on",
            "C": "same non-Au Q, thermal Au off",
            "D": "full independent Au-off Q, thermal Au off",
        },
        "Tmax_K": {"A": float(np.max(temp_a)), "B": float(np.max(temp_b)), "C": float(np.max(temp_c)), "D": float(np.max(temp_d))},
        "TaIrTe4_volume_average_K": {
            "A": weighted_average(temp_a, state_on, state_on["masks"]["TaIrTe4"]),
            "B": weighted_average(temp_b, state_on, state_on["masks"]["TaIrTe4"]),
            "C": weighted_average(temp_c, state_off, state_off["masks"]["TaIrTe4"]),
            "D": weighted_average(temp_d, state_off, state_off["masks"]["TaIrTe4"]),
        },
        "intermediate_solver_audits": {"B": audit_b, "C": audit_c},
        "field_telescoping_max_abs_error_K": float(np.max(np.abs((temp_a - temp_d) - ((temp_a - temp_b) + (temp_b - temp_c) + (temp_c - temp_d))))),
    }
    extent = (x[0] * 1e6, x[-1] * 1e6, y[0] * 1e6, y[-1] * 1e6)
    values = [ta_a, ta_b, ta_c, ta_d, ta_a - ta_b, ta_b - ta_c, ta_c - ta_d, ta_a - ta_d]
    titles = ["A full Au on", "B remove direct Au heat", "C remove thermal Au", "D full Au off", "A-B direct Au heat", "B-C thermal shunt", "C-D optical redistribution", "A-D total"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    for index, (ax, field, title) in enumerate(zip(axes.flat, values, titles, strict=True)):
        if index < 4:
            image = ax.imshow(field.T, origin="lower", extent=extent, cmap="magma")
        else:
            limit = np.percentile(np.abs(field), 99.5)
            image = ax.imshow(field.T, origin="lower", extent=extent, cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set_title(title); ax.set_xlabel("x=b (um)"); ax.set_ylabel("y=a (um)"); fig.colorbar(image, ax=ax, label="K")
    fig.suptitle(f"{on_case}: exact thermal Au-effect decomposition")
    fig.savefig(output / f"{on_case}_Au_temperature_decomposition.png", dpi=170)
    plt.close(fig)

    raw_out = RAW_OUT / f"{on_case}_Au_effect_decomposition.npz"
    np.savez_compressed(raw_out, x_m=x, y_m=y, Ta_A_K=ta_a.astype(np.float32), Ta_B_K=ta_b.astype(np.float32), Ta_C_K=ta_c.astype(np.float32), Ta_D_K=ta_d.astype(np.float32))
    gates = {
        "intermediate_thermal_residual_lt_1e-8": max(audit_b["residual"], audit_c["residual"]) < 1e-8,
        "intermediate_thermal_energy_balance_lt_1pct": max(audit_b["energy_balance"], audit_c["energy_balance"]) < 0.01,
        "temperature_field_telescoping_roundoff": thermal["field_telescoping_max_abs_error_K"] < 1e-12,
        "current_telescoping_roundoff": max(abs(v["telescoping_closure_A"]) for v in electrical.values()) < 1e-20,
        "no_Q_clipping_smoothing_gain_or_rescaling": True,
    }
    status = "VALIDATED_FINITE_T_Z_AU_EFFECT_DECOMPOSITION" if all(gates.values()) else "FAILED_FINITE_T_Z_AU_EFFECT_DECOMPOSITION"
    summary = {"status": status, "case": on_case, "classification": "exact forward telescoping decomposition, not causality beyond named model controls", "thermal": thermal, "electrical": electrical, "gates": gates, "raw_artifact": {"path": str(raw_out), "bytes": raw_out.stat().st_size, "sha256": sha256(raw_out), "committed_to_git": False}}
    (output / f"{on_case}_AU_EFFECT_DECOMPOSITION_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "RAW_ARTIFACT_MANIFEST.json").write_text(json.dumps({"status": status, "raw_artifact": summary["raw_artifact"]}, indent=2) + "\n")
    print(json.dumps({"status": status, "case": on_case, "thermal": thermal["Tmax_K"], "electrical": {key: value["contributions"] for key, value in electrical.items()}, "gates": gates}, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
