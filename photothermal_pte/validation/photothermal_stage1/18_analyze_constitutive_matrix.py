#!/usr/bin/env python3
"""Analyze the optical-only constitutive diagnostic matrix."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TAIRTE4_Z_MIN_M = -100e-9
TAIRTE4_Z_MAX_M = 0.0
WAVELENGTH_M = 4.0e-6


def load_component_module() -> Any:
    path = Path(__file__).with_name("13_analyze_native_yee_components.py")
    spec = importlib.util.spec_from_file_location("native_components", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tmm(epsilon_flake: complex) -> dict[str, float]:
    """Normal-incidence air/TaIrTe4/SiO2/Si characteristic matrix."""
    matrix = np.eye(2, dtype=complex)
    for index, thickness in (
        (np.sqrt(epsilon_flake), 100e-9),
        (1.38 + 0j, 285e-9),
    ):
        phase = 2.0 * np.pi * index * thickness / WAVELENGTH_M
        c = np.cos(phase)
        s = np.sin(phase)
        matrix = matrix @ np.asarray(
            [[c, -1j * s / index], [-1j * index * s, c]]
        )
    n0 = 1.0
    ns = 3.425
    denominator = (
        n0 * matrix[0, 0]
        + n0 * ns * matrix[0, 1]
        + matrix[1, 0]
        + ns * matrix[1, 1]
    )
    reflection = (
        n0 * matrix[0, 0]
        + n0 * ns * matrix[0, 1]
        - matrix[1, 0]
        - ns * matrix[1, 1]
    ) / denominator
    transmission = 2.0 * n0 / denominator
    r = float(abs(reflection) ** 2)
    t = float((ns / n0) * abs(transmission) ** 2)
    return {"R": r, "T": t, "A": 1.0 - r - t}


def monitor_epsilon(native: dict[str, np.ndarray]) -> dict[str, complex]:
    x = np.asarray(native["x_m"], float)
    y = np.asarray(native["y_m"], float)
    z = np.asarray(native["z_m"], float)
    ix = int(np.argmin(np.abs(x)))
    iy = int(np.argmin(np.abs(y)))
    iz = int(np.argmin(np.abs(z + 50e-9)))
    return {
        axis: complex(np.asarray(native[f"index_{axis}"])[ix, iy, iz]) ** 2
        for axis in "xyz"
    }


def field_ratios(native: dict[str, np.ndarray], component: Any) -> dict[str, float]:
    x = np.asarray(native["x_m"], float)
    y = np.asarray(native["y_m"], float)
    z = np.asarray(native["z_m"], float)
    integrals: dict[str, float] = {}
    for axis_index, axis in enumerate("xyz"):
        coordinates = [x, y, z]
        coordinates[axis_index] = coordinates[axis_index] + np.asarray(
            native[f"delta_{axis}_m"], float
        )
        native_z = coordinates[2]
        mask = (
            (native_z >= TAIRTE4_Z_MIN_M - 1e-15)
            & (native_z <= TAIRTE4_Z_MAX_M + 1e-15)
        )
        values = (
            np.abs(np.asarray(native[f"E{axis}"])) ** 2
            * mask[None, None, :]
        )
        integrals[axis] = component.integrate_xyz(values, *coordinates)
    total = sum(integrals.values())
    return {
        f"field_energy_proxy_{axis}_fraction": value / total
        for axis, value in integrals.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    component = load_component_module()

    rows: list[dict[str, Any]] = []
    for entry in manifest["cases"]:
        fdtd_dir = Path(entry["fdtd_dir"]).expanduser().resolve()
        native_dir = Path(entry["native_dir"]).expanduser().resolve()
        case = component.load_case(native_dir, fdtd_dir)
        result = component.component_absorption(case)
        native = {
            key: np.asarray(value)
            for key, value in case.items()
            if isinstance(value, np.ndarray)
        }
        epsilon = monitor_epsilon(native)
        summary = case["summary"]
        absorption = summary["absorption"]
        six_face = absorption["six_face_flux"]["A_six_face_net"]
        angle = float(
            summary.get("source", {}).get("polarization_angle_deg", 0.0)
        )
        tmm_x = tmm(epsilon["x"])
        tmm_y = tmm(epsilon["y"])
        wx = float(np.cos(np.deg2rad(angle)) ** 2)
        wy = float(np.sin(np.deg2rad(angle)) ** 2)
        row = {
            **entry,
            "solver_version_reported": summary["solver_version"],
            "polarization_angle_deg": angle,
            "A_flux_six_face": float(six_face),
            "A_Qx": result["native_integrals"]["x"],
            "A_Qy": result["native_integrals"]["y"],
            "A_Qz": result["native_integrals"]["z"],
            "A_Q_total": result["total_native"],
            "A_Q_pabs_internal": result["pabs_adv_internal"],
            "A_Q_python_export": float(
                absorption["A_Q_python_exported_integral"]
            ),
            "Q_minus_flux": result["total_native"] - float(six_face),
            "relative_mismatch": (
                result["total_native"] - float(six_face)
            ) / max(abs(float(six_face)), np.finfo(float).tiny),
            "TMM_A_x": tmm_x["A"],
            "TMM_A_y": tmm_y["A"],
            "TMM_A_for_polarization": wx * tmm_x["A"] + wy * tmm_y["A"],
            "epsilon_x_index_monitor": str(epsilon["x"]),
            "epsilon_y_index_monitor": str(epsilon["y"]),
            "epsilon_z_index_monitor": str(epsilon["z"]),
            **field_ratios(native, component),
        }
        rows.append(row)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output / "constitutive_matrix.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output / "constitutive_matrix.json").write_text(
        json.dumps({"manifest": str(manifest_path), "cases": rows}, indent=2)
    )

    by_name = {row["name"]: row for row in rows}
    flat_names = [
        name for name in ("flat_x_sampled_v261", "flat_y_sampled_v261", "flat_45_sampled_v261")
        if name in by_name
    ]
    if flat_names:
        x = np.arange(len(flat_names))
        fig, ax = plt.subplots(figsize=(8.0, 5.1))
        ax.bar(x - 0.22, [by_name[n]["A_Q_total"] for n in flat_names], 0.22, label="native Q")
        ax.bar(x, [by_name[n]["A_flux_six_face"] for n in flat_names], 0.22, label="six-face flux")
        ax.bar(x + 0.22, [by_name[n]["TMM_A_for_polarization"] for n in flat_names], 0.22, label="TMM")
        ax.set_xticks(x, [n.replace("_sampled_v261", "") for n in flat_names])
        ax.set_ylabel("absorptance")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "flat_polarization_Q_flux_TMM.png", dpi=190)
        plt.close(fig)

    def plot_sweep(category: str, x_key: str, filename: str, xlabel: str) -> None:
        selected = sorted(
            (r for r in rows if r.get("category") == category),
            key=lambda r: float(r[x_key]),
        )
        if not selected:
            return
        xvalues = [float(r[x_key]) for r in selected]
        fig, ax = plt.subplots(figsize=(7.8, 5.0))
        ax.plot(xvalues, [r["A_Q_total"] for r in selected], "o-", label="native Q total")
        ax.plot(xvalues, [r["A_Qy"] for r in selected], "s-", label="Qy")
        ax.plot(xvalues, [r["A_flux_six_face"] for r in selected], "^-", label="six-face flux")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("absorptance")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=190)
        plt.close(fig)

    plot_sweep("eps_y_sweep", "eps_y_imag_scale", "eps_y_imag_sweep.png", "Im(epsilon_y) scale")
    plot_sweep("time_sweep", "simulation_time_ps", "simulation_time_sweep.png", "simulation time [ps]")

    tolerance = 0.01
    flat_y = by_name.get("flat_y_sampled_v261")
    flat_45 = by_name.get("flat_45_sampled_v261")
    flat_x = by_name.get("flat_x_sampled_v261")
    if flat_y and abs(flat_y["relative_mismatch"]) > tolerance:
        decision = "flat y-pol 실패: epsilon_y fit/material/constitutive path 문제"
    elif flat_45 and abs(flat_45["relative_mismatch"]) > tolerance:
        decision = "flat y-pol 통과, 45-degree 실패: simultaneous x/y anisotropic accounting 문제"
    elif (
        flat_x and flat_y and flat_45
        and all(abs(r["relative_mismatch"]) <= tolerance for r in (flat_x, flat_y, flat_45))
    ):
        decision = "flat x/y/45 통과: patterned-case 및 material/version controls로 분기"
    else:
        decision = "flat decision incomplete"

    lines = [
        "# Constitutive-update versus Pabs/index-sampling diagnostic",
        "",
        "- Optical FDTD only; HEAT was not run.",
        "- No Qy deletion, clipping, or flux/Pabs gain was applied.",
        "",
        "## Primary decision",
        "",
        f"**{decision}**",
        "",
        "| case | A_Qx | A_Qy | A_Qz | A_Q total | six-face flux | TMM | mismatch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['A_Qx']:.9f} | {row['A_Qy']:.9f} | "
            f"{row['A_Qz']:.3e} | {row['A_Q_total']:.9f} | "
            f"{row['A_flux_six_face']:.9f} | {row['TMM_A_for_polarization']:.9f} | "
            f"{100*row['relative_mismatch']:.3f}% |"
        )
    lines += [
        "",
        "The TMM uses the index-monitor epsilon (index squared) from the completed "
        "FDTD case, with the exact 100 nm TaIrTe4 / 285 nm SiO2 / semi-infinite "
        "Si stack at normal incidence.",
    ]
    (output / "CONSTITUTIVE_DIAGNOSTIC_REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"decision": decision, "case_count": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
