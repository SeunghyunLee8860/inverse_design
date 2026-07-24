#!/usr/bin/env python3
"""Validate linear Q scaling and Si-substrate domain convergence in HEAT."""

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


SCALES = (0.5, 1.0, 2.0)
SI_THICKNESSES_M = (3.0e-6, 6.0e-6, 12.0e-6)
DOMAIN_CONVERGENCE_LIMIT = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lumerical-version", choices=("auto", "v261", "v251"), default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hide-gui", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Stage1Error(f"missing prerequisite summary: {path}")
    return json.loads(path.read_text())


def run_loaded(device: Any, project: Path) -> None:
    device.save(str(project))
    device.addjob(str(project), "HEAT")
    device.runjobs("HEAT", 0)
    device.load(str(project))


def metrics(device: Any) -> dict[str, float]:
    thermal = device.getresult("HEAT", "thermal")
    monitor = device.getresult("HEAT::T_xy_tairte4", "temperature")
    all_t = np.asarray(thermal["T"], float).reshape(-1)
    x = np.asarray(monitor["x"], float).reshape(-1)
    y = np.asarray(monitor["y"], float).reshape(-1)
    t = np.asarray(monitor["T"], float).squeeze()
    if t.shape != (x.size, y.size) or not np.all(np.isfinite(t)):
        raise Stage1Error(f"invalid TaIrTe4 monitor result shape={t.shape}")
    gx, gy = np.gradient(t, x, y, edge_order=2 if min(t.shape) >= 3 else 1)
    return {
        "deltaT_max_K": float(np.max(all_t) - config.BATH_TEMPERATURE_K),
        "deltaT_avg_TaIrTe4_K": float(np.mean(t) - config.BATH_TEMPERATURE_K),
        "max_abs_gradT_TaIrTe4_K_m": float(np.max(np.sqrt(gx**2 + gy**2))),
    }


def fit_line(scales: np.ndarray, values: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(scales, values, 1)
    prediction = slope * scales + intercept
    residual = float(np.sum((values - prediction) ** 2))
    total = float(np.sum((values - np.mean(values)) ** 2))
    r2 = 1.0 if total == 0 and residual == 0 else 1.0 - residual / total
    reference = float(values[int(np.where(np.isclose(scales, 1.0))[0][0])])
    slope_error = abs(float(slope) - reference) / max(abs(reference), np.finfo(float).tiny)
    return {
        "slope": float(slope), "intercept": float(intercept),
        "R2": float(r2), "scale1_reference": reference,
        "slope_relative_error": float(slope_error),
    }


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    stage_dir = output / "heat_scaling"
    stage_dir.mkdir(parents=True, exist_ok=True)
    summary_path = stage_dir / "heat_scaling_summary.json"
    installation = select_installation(args.lumerical_version)
    try:
        steady_summary = load_json(output / "heat_steady" / "heat_steady_summary.json")
        if not steady_summary.get("validated", False):
            raise Stage1Error("steady HEAT prerequisite did not pass; scaling is blocked")
        placeholder = steady_summary.get("placeholder_label")
        suffix = f"_{placeholder}" if placeholder else ""
        baseline = output / "heat_steady" / f"heat_steady{suffix}.ldev"
        if not baseline.is_file():
            raise Stage1Error(f"missing steady project: {baseline}")

        scale_rows: list[dict[str, Any]] = []
        with open_device(installation, hide=args.hide_gui) as device:
            for scale in SCALES:
                device.load(str(baseline))
                device.switchtolayout()
                device.setnamed("HEAT::Q_on_import", "scale factor", float(scale))
                project = stage_dir / f"heat_scale_{scale:g}{suffix}.ldev"
                run_loaded(device, project)
                scale_rows.append({"scale": scale, **metrics(device), "project": str(project)})

            # Start each domain test from the unmodified baseline and move the
            # physical Si bottom together with the simulation-region boundary.
            domain_rows: list[dict[str, Any]] = []
            top_si = -(0.100e-6 + 0.285e-6)
            for thickness in SI_THICKNESSES_M:
                device.load(str(baseline))
                device.switchtolayout()
                z_bottom = top_si - thickness
                device.setnamed("stage1 physical region", "z min", z_bottom)
                device.setnamed("Si_substrate", "z min", z_bottom)
                device.setnamed("HEAT::Q_on_import", "scale factor", 1.0)
                project = stage_dir / f"heat_si_{thickness * 1e6:g}um{suffix}.ldev"
                run_loaded(device, project)
                domain_rows.append({
                    "Si_thickness_m": thickness, "z_bottom_m": z_bottom,
                    **metrics(device), "project": str(project),
                })

        scales = np.asarray([row["scale"] for row in scale_rows], float)
        fits = {}
        for key in ("deltaT_max_K", "deltaT_avg_TaIrTe4_K", "max_abs_gradT_TaIrTe4_K_m"):
            fits[key] = fit_line(scales, np.asarray([row[key] for row in scale_rows], float))
        linear_pass = all(
            entry["R2"] > config.HEAT_SCALING_R2_LIMIT
            and entry["slope_relative_error"] < config.HEAT_SCALING_SLOPE_ERROR_LIMIT
            for entry in fits.values()
        )

        d6, d12 = domain_rows[-2], domain_rows[-1]
        domain_changes = {}
        for key in ("deltaT_max_K", "deltaT_avg_TaIrTe4_K", "max_abs_gradT_TaIrTe4_K_m"):
            domain_changes[key] = abs(d12[key] - d6[key]) / max(abs(d12[key]), np.finfo(float).tiny)
        domain_converged = all(value < DOMAIN_CONVERGENCE_LIMIT for value in domain_changes.values())

        with (stage_dir / f"scaling_results{suffix}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scale_rows[0]))
            writer.writeheader()
            writer.writerows(scale_rows)
        with (stage_dir / f"domain_convergence{suffix}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(domain_rows[0]))
            writer.writeheader()
            writer.writerows(domain_rows)

        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        ax.plot(scales, [row["deltaT_max_K"] for row in scale_rows], "o-", label="max")
        ax.plot(scales, [row["deltaT_avg_TaIrTe4_K"] for row in scale_rows], "s-", label="TaIrTe4 avg")
        ax.set_xlabel("Q scale")
        ax.set_ylabel("Delta T [K]")
        ax.legend()
        fig.tight_layout()
        fig.savefig(stage_dir / f"heat_scaling{suffix}.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        ax.plot(
            [row["Si_thickness_m"] * 1e6 for row in domain_rows],
            [row["deltaT_max_K"] for row in domain_rows], "o-",
        )
        ax.set_xlabel("Si substrate thickness [um]")
        ax.set_ylabel("maximum Delta T [K]")
        fig.tight_layout()
        fig.savefig(stage_dir / f"substrate_convergence{suffix}.png", dpi=180)
        plt.close(fig)

        passed = linear_pass and domain_converged
        summary = {
            "status": "passed" if passed else "failed_acceptance_criteria",
            "validated": passed,
            "selected_version": installation.version_key,
            "placeholder_label": placeholder,
            "scaling": {
                "rows": scale_rows, "fits": fits,
                "R2_limit": config.HEAT_SCALING_R2_LIMIT,
                "slope_relative_error_limit": config.HEAT_SCALING_SLOPE_ERROR_LIMIT,
                "passed": linear_pass,
            },
            "domain_convergence": {
                "rows": domain_rows,
                "relative_change_12um_vs_6um": domain_changes,
                "provisional_limit": DOMAIN_CONVERGENCE_LIMIT,
                "converged": domain_converged,
                "three_um_is_final_physical_result": False,
                "note": (
                    "3 um is never promoted automatically. The deepest two domains "
                    "must converge before any finite-substrate result is interpreted."
                ),
            },
        }
        write_json(summary_path, summary)
        print(json.dumps(jsonable(summary), indent=2), flush=True)
        return 0 if passed else 2
    except Exception as exc:
        failure = {
            "status": "failed", "validated": False,
            "selected_version": installation.version_key,
            "exception_type": type(exc).__name__, "exception": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(summary_path, failure)
        print(json.dumps(failure, indent=2), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
