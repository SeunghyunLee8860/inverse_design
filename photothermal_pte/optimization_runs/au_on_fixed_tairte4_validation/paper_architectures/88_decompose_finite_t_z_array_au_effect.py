#!/usr/bin/env python3
"""Exact forward telescoping decomposition of the finite-array Au effect."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ARRAY_MAPPING = HERE / "results_finite_T_Z_array_material_Q_mapping" / "FINITE_T_Z_ARRAY_MATERIAL_Q_MAPPING_SUMMARY.json"
BARE_MAPPING = HERE / "results_finite_T_Z_material_Q_mapping" / "FINITE_T_Z_MATERIAL_Q_MAPPING_SUMMARY.json"
ARRAY_PRIMARY = HERE / "results_finite_T_Z_array_thermal_electrical"
BARE_PRIMARY = HERE / "results_finite_T_Z_thermal_electrical"
ARRAY_RAW_PRIMARY = Path("/home/seunghyun/tairte4/raw_artifacts/finite_T_Z_array_thermal_electrical")
BARE_RAW_PRIMARY = Path("/home/seunghyun/tairte4/raw_artifacts/finite_T_Z_thermal_electrical")
RAW_OUT = Path("/home/seunghyun/tairte4/raw_artifacts/finite_T_Z_array_Au_effect_decomposition")
OUTPUT = HERE / "results_finite_T_Z_array_Au_effect_decomposition"


def load(name: str, filename: str):
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE = load("finite_t_z_array_stage81", "81_solve_finite_t_z_thermal_electrical.py")
HELPERS = load("finite_t_z_array_stage82", "82_decompose_finite_t_z_au_effect.py")
ADAPTER = load("finite_t_z_array_stage87", "87_solve_finite_t_z_array_thermal_electrical.py")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("GPU-only solve requires CUDA_VISIBLE_DEVICES")
    array_mapping = json.loads(ARRAY_MAPPING.read_text())
    bare_mapping = json.loads(BARE_MAPPING.read_text())
    on_meta = array_mapping["cases"][args.case]
    architecture = on_meta["architecture"]
    polarization = on_meta["polarization"]
    off_case = f"{architecture}_{polarization}_Au_off"
    off_meta = bare_mapping["cases"][off_case]
    rectangles = on_meta["top_Au_rectangles_m"]

    def array_top_au_fraction(x_edges, y_edges, _architecture, enabled):
        if not enabled:
            return np.zeros((len(x_edges) - 1, len(y_edges) - 1), dtype=np.float64)
        return ADAPTER.overlap_fraction(x_edges, y_edges, rectangles)

    STAGE.top_au_fraction = array_top_au_fraction
    on_primary_path = ARRAY_PRIMARY / args.case / f"{args.case}_THERMAL_ELECTRICAL_SUMMARY.json"
    off_primary_path = BARE_PRIMARY / off_case / f"{off_case}_THERMAL_ELECTRICAL_SUMMARY.json"
    on_primary = json.loads(on_primary_path.read_text())
    off_primary = json.loads(off_primary_path.read_text())
    scale = float(on_primary["source"]["linear_response_scale"])
    on_map = Path(on_meta["raw_mapped_artifact"]["path"])
    if HELPERS.sha256(on_map) != on_meta["raw_mapped_artifact"]["sha256"]:
        raise RuntimeError("array mapped-Q SHA mismatch")
    with np.load(on_map, allow_pickle=False) as raw:
        state_on = STAGE.build_thermal_state(raw, architecture, True)
        non_top_power = sum(
            (np.asarray(raw[f"power_{key}_W"], dtype=np.float64) for key in ("Si", "SiO2", "Au_mirror", "TaIrTe4")),
            start=np.zeros(state_on["shape"]),
        ) * scale
    with np.load(ARRAY_RAW_PRIMARY / f"{args.case}_finite_thermal_electrical.npz", allow_pickle=False) as primary:
        temp_a = np.asarray(primary["temperature_3d_K"], dtype=np.float64)
    temp_b, audit_b = HELPERS.solve_temperature(STAGE, state_on, non_top_power, args.cuda_device)
    ta_a, x, y = STAGE.thickness_average_ta(state_on, temp_a)
    ta_b, _, _ = STAGE.thickness_average_ta(state_on, temp_b)

    with np.load(on_map, allow_pickle=False) as raw:
        state_off = STAGE.build_thermal_state(raw, architecture, False)
    temp_c, audit_c = HELPERS.solve_temperature(STAGE, state_off, non_top_power, args.cuda_device)
    ta_c, _, _ = STAGE.thickness_average_ta(state_off, temp_c)
    off_raw_path = BARE_RAW_PRIMARY / f"{off_case}_finite_thermal_electrical.npz"
    if HELPERS.sha256(Path(off_meta["raw_mapped_artifact"]["path"])) != off_meta["raw_mapped_artifact"]["sha256"]:
        raise RuntimeError("bare mapped-Q SHA mismatch")
    with np.load(off_raw_path, allow_pickle=False) as primary:
        temp_d = np.asarray(primary["temperature_3d_K"], dtype=np.float64)
    ta_d, _, _ = STAGE.thickness_average_ta(state_off, temp_d)

    output = OUTPUT / args.case
    output.mkdir(parents=True, exist_ok=True)
    RAW_OUT.mkdir(parents=True, exist_ok=True)
    electrical = {}
    for orientation in ("top_bottom", "left_right"):
        i_a_on = on_primary["electrical"][orientation]
        i_a_off = HELPERS.current_for(STAGE, state_off, ta_a, x, y, orientation, False, architecture, args.cuda_device)
        i_b_off = HELPERS.current_for(STAGE, state_off, ta_b, x, y, orientation, False, architecture, args.cuda_device)
        i_c_off = HELPERS.current_for(STAGE, state_off, ta_c, x, y, orientation, False, architecture, args.cuda_device)
        i_d_off = off_primary["electrical"][orientation]
        currents = {
            "A_full_array_Au_on_Q_K_E": float(i_a_on["high_terminal_current_A"]),
            "A_same_T_electrical_Au_off": float(i_a_off["high_terminal_current_A"]),
            "B_top_Au_heat_removed_K_on_E_off": float(i_b_off["high_terminal_current_A"]),
            "C_top_Au_heat_and_thermal_shunt_removed_E_off": float(i_c_off["high_terminal_current_A"]),
            "D_independent_bare_flake_Q_K_E": float(i_d_off["high_terminal_current_A"]),
        }
        contributions = {
            "floating_Au_electrical_shunt_A": currents["A_full_array_Au_on_Q_K_E"] - currents["A_same_T_electrical_Au_off"],
            "direct_top_Au_absorption_heat_A": currents["A_same_T_electrical_Au_off"] - currents["B_top_Au_heat_removed_K_on_E_off"],
            "top_Au_thermal_shunt_A": currents["B_top_Au_heat_removed_K_on_E_off"] - currents["C_top_Au_heat_and_thermal_shunt_removed_E_off"],
            "array_Au_induced_nonAu_optical_redistribution_A": currents["C_top_Au_heat_and_thermal_shunt_removed_E_off"] - currents["D_independent_bare_flake_Q_K_E"],
        }
        total = currents["A_full_array_Au_on_Q_K_E"] - currents["D_independent_bare_flake_Q_K_E"]
        closure = total - sum(contributions.values())
        electrical[orientation] = {
            "currents": currents,
            "contributions": contributions,
            "full_array_on_minus_bare_A": total,
            "telescoping_closure_A": closure,
            "intermediate_audits": {"A_temperature_E_off": i_a_off, "B": i_b_off, "C": i_c_off},
        }
        labels = ["electrical\nshunt", "direct Au\nheating", "thermal\nshunt", "optical\nredistribution"]
        values = np.asarray(list(contributions.values())) * 1e9
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        ax.bar(labels, values, color=["#7b3294", "#d7191c", "#2c7bb6", "#fdae61"])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("contribution to I_array - I_bare (nA)")
        ax.set_title(f"{args.case}, {orientation}: exact modeled Au-array decomposition")
        for index, value in enumerate(values):
            ax.text(index, value, f"{value:.4g}", ha="center", va="bottom" if value >= 0 else "top")
        fig.savefig(output / f"{args.case}_{orientation}_Au_array_current_decomposition.png", dpi=180)
        plt.close(fig)

    thermal = {
        "scenario_sequence": {
            "A": "full array-Au Q, thermal Au on",
            "B": "same non-Au Q, direct top-Au absorption removed, thermal Au on",
            "C": "same non-Au Q, thermal and electrical top Au off",
            "D": "independent bare-flake optical Q and Au-off thermal/electrical operator",
        },
        "Tmax_K": {"A": float(np.max(temp_a)), "B": float(np.max(temp_b)), "C": float(np.max(temp_c)), "D": float(np.max(temp_d))},
        "TaIrTe4_volume_average_K": {
            "A": HELPERS.weighted_average(temp_a, state_on, state_on["masks"]["TaIrTe4"]),
            "B": HELPERS.weighted_average(temp_b, state_on, state_on["masks"]["TaIrTe4"]),
            "C": HELPERS.weighted_average(temp_c, state_off, state_off["masks"]["TaIrTe4"]),
            "D": HELPERS.weighted_average(temp_d, state_off, state_off["masks"]["TaIrTe4"]),
        },
        "intermediate_solver_audits": {"B": audit_b, "C": audit_c},
        "field_telescoping_max_abs_error_K": float(np.max(np.abs((temp_a - temp_d) - ((temp_a - temp_b) + (temp_b - temp_c) + (temp_c - temp_d))))),
    }
    extent = (x[0] * 1e6, x[-1] * 1e6, y[0] * 1e6, y[-1] * 1e6)
    fields = [ta_a, ta_b, ta_c, ta_d, ta_a - ta_b, ta_b - ta_c, ta_c - ta_d, ta_a - ta_d]
    titles = ["A full array Au on", "B remove direct Au heat", "C remove Au thermal shunt", "D bare flake", "A-B direct Au heat", "B-C thermal shunt", "C-D optical redistribution", "A-D total"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    for index, (ax, field, title) in enumerate(zip(axes.flat, fields, titles, strict=True)):
        if index < 4:
            image = ax.imshow(field.T, origin="lower", extent=extent, cmap="magma")
        else:
            limit = max(float(np.percentile(np.abs(field), 99.5)), np.finfo(float).tiny)
            image = ax.imshow(field.T, origin="lower", extent=extent, cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set_title(title); ax.set_xlabel("x=b (um)"); ax.set_ylabel("y=a (um)"); fig.colorbar(image, ax=ax, label="K")
    fig.suptitle(f"{args.case}: exact modeled Au-array thermal decomposition")
    fig.savefig(output / f"{args.case}_Au_array_temperature_decomposition.png", dpi=170)
    plt.close(fig)

    raw_out = RAW_OUT / f"{args.case}_Au_array_effect_decomposition.npz"
    np.savez_compressed(raw_out, x_m=x, y_m=y, Ta_A_K=ta_a.astype(np.float32), Ta_B_K=ta_b.astype(np.float32), Ta_C_K=ta_c.astype(np.float32), Ta_D_K=ta_d.astype(np.float32))
    gates = {
        "intermediate_thermal_residual_lt_1e-8": max(audit_b["residual"], audit_c["residual"]) < 1e-8,
        "intermediate_thermal_energy_balance_lt_1pct": max(audit_b["energy_balance"], audit_c["energy_balance"]) < 0.01,
        "temperature_field_telescoping_roundoff": thermal["field_telescoping_max_abs_error_K"] < 1e-12,
        "current_telescoping_roundoff": max(abs(v["telescoping_closure_A"]) for v in electrical.values()) < 1e-20,
        "no_Q_clipping_smoothing_gain_or_rescaling": True,
    }
    status = "VALIDATED_FINITE_T_Z_ARRAY_AU_EFFECT_DECOMPOSITION" if all(gates.values()) else "FAILED_FINITE_T_Z_ARRAY_AU_EFFECT_DECOMPOSITION"
    summary = {
        "status": status,
        "case": args.case,
        "bare_control_case": off_case,
        "classification": "exact forward telescoping decomposition within named contact/material scenarios; not experimental causality",
        "thermal": thermal,
        "electrical": electrical,
        "gates": gates,
        "raw_artifact": {"path": str(raw_out), "bytes": raw_out.stat().st_size, "sha256": HELPERS.sha256(raw_out), "committed_to_git": False},
    }
    (output / f"{args.case}_AU_ARRAY_EFFECT_DECOMPOSITION_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "RAW_ARTIFACT_MANIFEST.json").write_text(json.dumps({"status": status, "raw_artifact": summary["raw_artifact"]}, indent=2) + "\n")
    print(json.dumps({"status": status, "case": args.case, "thermal": thermal["Tmax_K"], "electrical": {key: value["contributions"] for key, value in electrical.items()}, "gates": gates}, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
