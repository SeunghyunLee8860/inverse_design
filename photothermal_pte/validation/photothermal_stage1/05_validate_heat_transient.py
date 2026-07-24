#!/usr/bin/env python3
"""Run a guarded transient Q_on pulse test after all steady gates pass."""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config_stage1 as config
from lumerical_api import Stage1Error, jsonable, open_device, select_installation, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lumerical-version", choices=("auto", "v261", "v251"), default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hide-gui", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Stage1Error(f"missing prerequisite: {path}")
    return json.loads(path.read_text())


def missing_transient_properties() -> list[str]:
    values = {
        "design.rho": config.DESIGN_RHO_KG_M3,
        "design.cp": config.DESIGN_CP_J_KG_K,
        "TaIrTe4.rho": config.TAIRTE4_RHO_KG_M3,
        "TaIrTe4.cp": config.TAIRTE4_CP_J_KG_K,
        "SiO2.rho": config.SIO2_RHO_KG_M3,
        "SiO2.cp": config.SIO2_CP_J_KG_K,
        "Si.rho": config.SI_RHO_KG_M3,
        "Si.cp": config.SI_CP_J_KG_K,
    }
    return [name for name, value in values.items() if value is None]


def characteristic_time(properties: dict[str, Any]) -> float:
    """Conservative diffusion time based on the slowest supplied solid."""
    lengths = {"design": 0.6e-6, "TaIrTe4": 0.1e-6, "SiO2": 0.285e-6, "Si": 12e-6}
    candidates = []
    for name, length in lengths.items():
        entry = properties[name]
        k = np.asarray(entry["k"], float)
        k_min = float(np.min(k))
        alpha = k_min / (float(entry["rho"]) * float(entry["cp"]))
        candidates.append(length**2 / alpha)
    return max(candidates)


def time_series(device: Any) -> dict[str, np.ndarray]:
    thermal = device.getresult("HEAT", "thermal")
    monitor = device.getresult("HEAT::T_xy_tairte4", "temperature")
    time_key = "t" if "t" in thermal else "time"
    if time_key not in thermal:
        raise Stage1Error(f"transient thermal dataset has no time parameter: {list(thermal)}")
    time = np.asarray(thermal[time_key], float).reshape(-1)
    temperature = np.asarray(thermal["T"], float).squeeze()
    if temperature.shape[-1] != time.size:
        raise Stage1Error(f"thermal T shape {temperature.shape} does not end in time={time.size}")
    hotspot = np.max(temperature.reshape(-1, time.size), axis=0)

    x = np.asarray(monitor["x"], float).reshape(-1)
    y = np.asarray(monitor["y"], float).reshape(-1)
    monitor_t = np.asarray(monitor["T"], float).squeeze()
    if monitor_t.shape != (x.size, y.size, time.size):
        raise Stage1Error(
            f"TaIrTe4 transient monitor shape {monitor_t.shape} != "
            f"{(x.size, y.size, time.size)}"
        )
    average = np.mean(monitor_t, axis=(0, 1))
    grad_max = np.empty(time.size)
    edge_order = 2 if min(x.size, y.size) >= 3 else 1
    for index in range(time.size):
        gx, gy = np.gradient(monitor_t[:, :, index], x, y, edge_order=edge_order)
        grad_max[index] = np.max(np.sqrt(gx**2 + gy**2))
    return {
        "time_s": time, "T_hotspot_K": hotspot,
        "T_avg_TaIrTe4_K": average,
        "max_abs_gradT_TaIrTe4_K_m": grad_max,
    }


def run_transient(
    installation: Any,
    baseline: Path,
    project: Path,
    *,
    tau: float,
    refinement: int,
    hide: bool,
) -> dict[str, np.ndarray]:
    t_off = 10.0 * tau
    t_end = 20.0 * tau
    max_dt = tau / (100.0 * refinement)
    min_dt = max_dt / 20.0
    with open_device(installation, hide=hide) as device:
        device.load(str(baseline))
        device.switchtolayout()
        for material, rho, cp in (
            ("design thermal", config.DESIGN_RHO_KG_M3, config.DESIGN_CP_J_KG_K),
            ("TaIrTe4 thermal", config.TAIRTE4_RHO_KG_M3, config.TAIRTE4_CP_J_KG_K),
            ("SiO2 thermal", config.SIO2_RHO_KG_M3, config.SIO2_CP_J_KG_K),
            ("Si thermal", config.SI_RHO_KG_M3, config.SI_CP_J_KG_K),
        ):
            device.select(f"materials::{material}")
            device.set("mass density.constant", float(rho))
            device.set("specific heat.active model", "constant")
            device.set("specific heat.constant", float(cp))
        device.setnamed("HEAT", "solver mode", "transient")
        device.setnamed("HEAT", "transient min time step", min_dt)
        device.setnamed("HEAT", "transient max time step", max_dt)
        device.setnamed("HEAT", "shutter mode", "pulse on")
        device.setnamed("HEAT", "shutter ton", 0.0)
        device.setnamed("HEAT", "shutter toff", t_off)
        device.setnamed("HEAT", "shutter tslew", min_dt)
        device.setnamed("HEAT::boundary conditions::T_bottom", "bc mode", "transient")
        device.setnamed("HEAT::boundary conditions::T_bottom", "sweep type", "single")
        device.setnamed(
            "HEAT::boundary conditions::T_bottom", "transient time steps",
            np.array([0.0, t_end]),
        )
        device.setnamed(
            "HEAT::boundary conditions::T_bottom", "transient value table",
            np.array([config.BATH_TEMPERATURE_K, config.BATH_TEMPERATURE_K]),
        )
        if device.getnamednumber("HEAT::T_volume_transient") == 0:
            monitor = device.addtemperaturemonitor("HEAT")
            monitor["name"] = "T_volume_transient"
            monitor["monitor type"] = "3D"
            monitor["x"] = 0.0
            monitor["x span"] = float(device.getnamed("stage1 physical region", "x span"))
            monitor["y"] = 0.0
            monitor["y span"] = float(device.getnamed("stage1 physical region", "y span"))
            monitor["z min"] = float(device.getnamed("stage1 physical region", "z min"))
            monitor["z max"] = float(device.getnamed("stage1 physical region", "z max"))
        device.save(str(project))
        device.addjob(str(project), "HEAT")
        device.runjobs("HEAT", 0)
        device.load(str(project))
        result = time_series(device)
        result["min_time_step_s"] = np.array(min_dt)
        result["max_time_step_s"] = np.array(max_dt)
        result["t_off_s"] = np.array(t_off)
        device.save(str(project))
        return result


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    stage_dir = output / "heat_transient"
    stage_dir.mkdir(parents=True, exist_ok=True)
    summary_path = stage_dir / "transient_summary.json"
    installation = select_installation(args.lumerical_version)
    missing = missing_transient_properties()
    try:
        steady = read_json(output / "heat_steady" / "heat_steady_summary.json")
        scaling = read_json(output / "heat_scaling" / "heat_scaling_summary.json")
        if not steady.get("validated") or not scaling.get("validated"):
            raise Stage1Error("steady and scaling/domain gates must both pass before transient")
        if steady.get("placeholder_label"):
            raise Stage1Error("placeholder transient is prohibited")
        if missing:
            skipped = {
                "status": "skipped_missing_material_properties",
                "validated": False, "executed": False,
                "selected_version": installation.version_key,
                "missing_rho_cp": missing,
                "reason": "Transient HEAT requires explicit rho and cp for every solid; no placeholders allowed.",
            }
            write_json(summary_path, skipped)
            print(json.dumps(skipped, indent=2), flush=True)
            return 3

        props = steady["thermal_properties"]
        tau = characteristic_time(props)
        domain_rows = scaling["domain_convergence"]["rows"]
        baseline = Path(domain_rows[-1]["project"])
        steady_delta = float(domain_rows[-1]["deltaT_max_K"])
        coarse = run_transient(
            installation, baseline, stage_dir / "heat_transient_coarse.ldev",
            tau=tau, refinement=1, hide=args.hide_gui,
        )
        fine = run_transient(
            installation, baseline, stage_dir / "heat_transient.ldev",
            tau=tau, refinement=2, hide=args.hide_gui,
        )

        t = fine["time_s"]
        t_off = float(fine["t_off_s"])
        on_index = int(np.argmin(np.abs(t - t_off)))
        final_on_delta = float(fine["T_hotspot_K"][on_index] - config.BATH_TEMPERATURE_K)
        steady_error = abs(final_on_delta - steady_delta) / max(abs(steady_delta), np.finfo(float).tiny)
        after = fine["T_hotspot_K"][on_index:]
        monotonic_tolerance = max(1e-6, 1e-5 * max(abs(final_on_delta), 1.0))
        cooling_monotonic = bool(np.all(np.diff(after) <= monotonic_tolerance))
        coarse_on_fine = np.interp(t, coarse["time_s"], coarse["T_hotspot_K"])
        timestep_error = float(
            np.max(np.abs(coarse_on_fine - fine["T_hotspot_K"]))
            / max(abs(final_on_delta), np.finfo(float).tiny)
        )
        no_overshoot = bool(
            np.max(fine["T_hotspot_K"][: on_index + 1])
            <= max(fine["T_hotspot_K"][on_index], config.BATH_TEMPERATURE_K + steady_delta)
            + monotonic_tolerance
        )
        passed = (
            steady_error < config.TRANSIENT_STEADY_ERROR_LIMIT
            and cooling_monotonic and no_overshoot
            and timestep_error < config.TRANSIENT_TIMESTEP_ERROR_LIMIT
        )

        np.savez_compressed(stage_dir / "transient_temperature.npz", **fine)
        with (stage_dir / "T_hotspot_vs_time.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("time_s", "T_hotspot_K", "T_avg_TaIrTe4_K", "max_abs_gradT_TaIrTe4_K_m"))
            writer.writerows(zip(
                fine["time_s"], fine["T_hotspot_K"],
                fine["T_avg_TaIrTe4_K"], fine["max_abs_gradT_TaIrTe4_K_m"],
            ))
        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        ax.plot(fine["time_s"], fine["T_hotspot_K"], label="fine")
        ax.plot(coarse["time_s"], coarse["T_hotspot_K"], "--", label="coarse")
        ax.axvline(t_off, color="k", lw=1, label="source off")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("hotspot T [K]")
        ax.legend()
        fig.tight_layout()
        fig.savefig(stage_dir / "T_hotspot_vs_time.png", dpi=180)
        plt.close(fig)

        summary = {
            "status": "passed" if passed else "failed_acceptance_criteria",
            "validated": passed, "executed": True,
            "selected_version": installation.version_key,
            "characteristic_time_s": tau, "t_off_s": t_off,
            "long_on_deltaT_K": final_on_delta,
            "steady_deltaT_K": steady_delta,
            "long_on_vs_steady_error": steady_error,
            "long_on_vs_steady_limit": config.TRANSIENT_STEADY_ERROR_LIMIT,
            "cooling_monotonic": cooling_monotonic,
            "no_nonphysical_overshoot": no_overshoot,
            "time_step_refinement_error": timestep_error,
            "time_step_refinement_limit": config.TRANSIENT_TIMESTEP_ERROR_LIMIT,
            "coarse_max_time_step_s": float(coarse["max_time_step_s"]),
            "fine_max_time_step_s": float(fine["max_time_step_s"]),
        }
        write_json(summary_path, summary)
        print(json.dumps(jsonable(summary), indent=2), flush=True)
        return 0 if passed else 2
    except Exception as exc:
        failure = {
            "status": "failed", "validated": False, "executed": False,
            "selected_version": installation.version_key,
            "exception_type": type(exc).__name__, "exception": str(exc),
            "traceback": traceback.format_exc(),
            "missing_rho_cp": missing,
        }
        write_json(summary_path, failure)
        print(json.dumps(failure, indent=2), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
