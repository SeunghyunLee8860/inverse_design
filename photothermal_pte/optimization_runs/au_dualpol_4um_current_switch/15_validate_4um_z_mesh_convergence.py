#!/usr/bin/env python3
"""Fail-closed z-mesh convergence for the 4 um Au/TaIrTe4 stack.

The optimized density, x/y grid, source, material endpoints, and historical
O3/TE1 gray law are frozen.  Only the optical z discretization of SiO2,
TaIrTe4, and Au is refined.  Every optical mesh receives its own all-air
incident-power calibration before Q is conservatively mapped to the identical
explicit thermal/electrical operator.

This script diagnoses the current blocked gray checkpoint.  It does not
promote the gray material law or restart optimization.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_4um_model as optical_model,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.combined_4um import (
    EPS0_F_PER_M,
    electric_yee_dual_volumes,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.multiphysics_4um import (
    build_thermal_state,
    evaluate_fixed_source,
    map_native_q_to_thermal,
    tairte4_temperature,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.paths import raw_path
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.historical_checkpoint import (
    CHECKPOINT,
    EXPECTED_CHECKPOINT_SHA256,
    load_densities,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.mesh_variants import (
    PARTIAL_MATERIAL_Z,
    mesh_context as variant_mesh_context,
    variant_edges,
    variant_layout,
)


HERE = Path(__file__).resolve().parent
OUT = HERE / "results_4um_dualpol_au_z_mesh_convergence"
RAW = raw_path("z_mesh_convergence")
TOTAL_PERIODS = 24
WINDOW_PERIODS = 4
LEVELS = (1, 2, 4, 8)
POLARIZATIONS = ("Ea", "Eb")
GATE_POWER = 5.0e-3
GATE_FIELD = 5.0e-3
GATE_CURRENT = 5.0e-3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def segments(parts: tuple[tuple[float, float, int], ...]) -> np.ndarray:
    values: list[np.ndarray] = []
    for index, (start, stop, cells) in enumerate(parts):
        segment = np.linspace(start, stop, cells + 1, dtype=np.float64)
        values.append(segment if index == 0 else segment[1:])
    return np.concatenate(values)


def refined_edges(factor: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return variant_edges(factor, PARTIAL_MATERIAL_Z)


def refined_layout(factor: int):
    return variant_layout(factor, PARTIAL_MATERIAL_Z)


def mesh_context(factor: int):
    return variant_mesh_context(factor, PARTIAL_MATERIAL_Z)


def build_at_mesh(
    factor: int,
    polarization: str,
    *,
    air_only: bool,
):
    with mesh_context(factor):
        return optical_model.build_model(
            polarization,
            total_periods=TOTAL_PERIODS,
            window_periods=WINDOW_PERIODS,
            include_adjoint_source=False,
            air_only_source_calibration=air_only,
        )


def mesh_audit(factor: int, model: dict[str, object]) -> dict[str, object]:
    edges = refined_edges(factor)
    widths = tuple(np.diff(value) for value in edges)
    slices = model["slices"]
    expected = {
        "fixed_285nm_sio2": 3 * factor,
        "fixed_tairte4": 5 * factor,
        "au_design": 2 * factor,
    }
    for name, count in expected.items():
        realized = int(slices[name][2].stop - slices[name][2].start)
        if realized != count:
            raise RuntimeError(f"{name}: expected {count} z cells, got {realized}")
    z_bounds = {
        name: [
            float(edges[2][slices[name][2].start]),
            float(edges[2][slices[name][2].stop]),
        ]
        for name in expected
    }
    exact = {
        "fixed_285nm_sio2": (-0.385e-6, -0.100e-6),
        "fixed_tairte4": (-0.100e-6, 0.0),
        "au_design": (0.0, 0.050e-6),
    }
    for name, bounds in exact.items():
        if not np.allclose(z_bounds[name], bounds, rtol=0.0, atol=2.0e-18):
            raise RuntimeError(f"{name} bounds changed: {z_bounds[name]}")
    return {
        "factor": factor,
        "grid_shape": [int(value) for value in model["grid"].shape],
        "yee_cell_count": int(np.prod(model["grid"].shape)),
        "x_bounds_m": [float(edges[0][0]), float(edges[0][-1])],
        "y_bounds_m": [float(edges[1][0]), float(edges[1][-1])],
        "z_bounds_m": [float(edges[2][0]), float(edges[2][-1])],
        "central_dx_m": 100.0e-9,
        "central_dy_m": 100.0e-9,
        "sio2_dz_m": float(285.0e-9 / (3 * factor)),
        "tairte4_dz_m": float(100.0e-9 / (5 * factor)),
        "au_dz_m": float(50.0e-9 / (2 * factor)),
        "layer_bounds_m": z_bounds,
        "source_bounds": model["placement"]["gaussian_source"],
        "incident_monitor_bounds": model["placement"]["incident_plane"],
        "closed_monitor_bounds": model["placement"]["material_flux_td"],
        "min_grid_step_m": float(min(np.min(value) for value in widths)),
        "max_grid_step_m": float(max(np.max(value) for value in widths)),
    }


def source_only_power(factor: int, polarization: str) -> tuple[float, float]:
    model = build_at_mesh(factor, polarization, air_only=True)
    arrays = (
        model["base"]
        .reset()
        .aset("dispersive_c1", model["fixed_c1"])
        .aset("dispersive_c2", model["fixed_c2"])
        .aset("dispersive_c3", model["fixed_c3"])
    )
    start = time.perf_counter()
    _, output = model["fdtdx"].run_fdtd(
        arrays, model["placed"], model["config"], model["key"], show_progress=False
    )
    target = output.detector_states["target_field"]["phasor"]
    model["jax"].block_until_ready(target)
    runtime = time.perf_counter() - start
    eta0 = float(model["fdtdx"].constants.eta0)
    power = float(
        eta0
        * np.asarray(
            model["placed"]["incident_plane"].compute_poynting_flux(
                output.detector_states["incident_plane"]
            )
        )[0]
    )
    if not np.isfinite(power) or power <= 0.0:
        raise RuntimeError(f"invalid source-only power {power}")
    return power, runtime


@dataclass
class ForwardRunner:
    factor: int
    polarization: str
    model: dict[str, object]
    solve: object
    volumes: dict[str, np.ndarray]
    physical_prefactor: float
    ta_imag: np.ndarray
    compile_s: float

    @classmethod
    def create(cls, factor: int, polarization: str) -> "ForwardRunner":
        model = build_at_mesh(factor, polarization, air_only=False)
        jax = model["jax"]
        jnp = model["jnp"]
        fdtdx = model["fdtdx"]
        au_slice = model["slices"]["au_design"]
        au_c3 = float(model["coefficients"]["au"][2])

        def arrays_for_density(density):
            strength = density**3
            c3 = model["fixed_c3"]
            for component in range(3):
                c3 = c3.at[(0, component, *au_slice)].set(
                    au_c3 * strength[:, :, None]
                )
            return (
                model["base"]
                .reset()
                .aset("dispersive_c1", model["fixed_c1"])
                .aset("dispersive_c2", model["fixed_c2"])
                .aset("dispersive_c3", c3)
            )

        def forward(density):
            return fdtdx.run_fdtd(
                arrays_for_density(density),
                model["placed"],
                model["config"],
                model["key"],
                show_progress=False,
            )[1]

        example = jnp.full(CONTRACT.design_shape, 0.5, dtype=jnp.float32)
        start = time.perf_counter()
        solve = jax.jit(forward).lower(example).compile()
        compile_s = time.perf_counter() - start
        eta0 = float(fdtdx.constants.eta0)
        prefactor = 0.5 * float(model["omega_rad_s"]) * EPS0_F_PER_M * eta0**2
        ta_imag = np.asarray(
            [
                model["epsilon"]["tairte4"]["b"].imag,
                model["epsilon"]["tairte4"]["a"].imag,
                model["epsilon"]["tairte4"]["c"].imag,
            ],
            dtype=np.float64,
        )[:, None, None, None]
        volumes = {
            "au": electric_yee_dual_volumes(
                model["grid"], model["slices"]["au_design"]
            ),
            "tairte4": electric_yee_dual_volumes(
                model["grid"], model["slices"]["fixed_tairte4"]
            ),
        }
        return cls(
            factor=factor,
            polarization=polarization,
            model=model,
            solve=solve,
            volumes=volumes,
            physical_prefactor=prefactor,
            ta_imag=ta_imag,
            compile_s=compile_s,
        )

    def run(self, rho: np.ndarray, source_scale: float):
        start = time.perf_counter()
        output = self.solve(self.model["jnp"].asarray(rho, dtype=self.model["jnp"].float32))
        marker = output.detector_states["au_late"]["phasor"]
        self.model["jax"].block_until_ready(marker)
        runtime = time.perf_counter() - start
        e_au = np.asarray(output.detector_states["au_late"]["phasor"][0, 0])
        e_ta = np.asarray(output.detector_states["tairte4_late"]["phasor"][0, 0])
        strength = np.asarray(rho, dtype=np.float64) ** 3
        q = {
            "au": (
                source_scale
                * self.physical_prefactor
                * float(self.model["epsilon"]["au"].imag)
                * strength[None, :, :, None]
                * np.abs(e_au) ** 2
            ),
            "tairte4": (
                source_scale
                * self.physical_prefactor
                * self.ta_imag
                * np.abs(e_ta) ** 2
            ),
        }
        p_components = {
            f"{material}_{axis}_W": float(np.sum(q[material][component] * self.volumes[material][component]))
            for material in ("au", "tairte4")
            for component, axis in enumerate(("x", "y", "z"))
        }
        p_total = float(sum(p_components.values()))
        eta0 = float(self.model["fdtdx"].constants.eta0)
        p_six = source_scale * eta0 * float(
            np.mean(
                np.asarray(
                    output.detector_states["material_flux_td"]["poynting_flux"]
                )[:, 0]
            )
        )
        closure = abs(p_total - p_six) / max(abs(p_six), np.finfo(float).tiny)
        return output, q, p_components, p_total, p_six, closure, runtime


def thermal_volume(state) -> np.ndarray:
    return (
        state.widths[0][:, None, None]
        * state.widths[1][None, :, None]
        * state.widths[2][None, None, :]
    )


def volume_l2(power: np.ndarray, reference: np.ndarray, volume: np.ndarray) -> float:
    difference = (power - reference) / volume
    density_reference = reference / volume
    numerator = math.sqrt(float(np.sum(difference**2 * volume)))
    denominator = math.sqrt(float(np.sum(density_reference**2 * volume)))
    return numerator / max(denominator, np.finfo(float).tiny)


def relative(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), np.finfo(float).tiny)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("GPU required: set CUDA_VISIBLE_DEVICES")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    densities = load_densities()
    cuda_device = int(os.environ.get("THERMAL_CUDA_DEVICE", "0"))

    audits = []
    for factor in LEVELS:
        audit_model = build_at_mesh(factor, "Ea", air_only=False)
        audits.append(mesh_audit(factor, audit_model))
    runsetup = {
        "status": "VALIDATED_4UM_AU_Z_MESH_RUNSETUP",
        "scope": "z-only optical refinement; x/y, geometry, source, and material endpoints fixed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(CHECKPOINT.resolve()),
            "bytes": CHECKPOINT.stat().st_size,
            "sha256": sha256(CHECKPOINT),
            "expected_sha256": EXPECTED_CHECKPOINT_SHA256,
            "sha256_matches_expected": True,
        },
        "levels": audits,
        "time_contract": {
            "total_periods": TOTAL_PERIODS,
            "phasor_window_periods": WINDOW_PERIODS,
        },
        "density_cases": list(densities),
        "gray_law": "historical diagnostic only: optical rho^3, thermal/electrical rho^1",
    }
    write_json(OUT / "z_mesh_runsetup_audit.json", runsetup)
    if args.audit_only:
        print(json.dumps(runsetup, indent=2), flush=True)
        return 0

    source_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    stored: dict[tuple[int, str, str], dict[str, object]] = {}
    incident_reference: dict[int, float] = {}
    cached_factors: set[int] = set()

    # A failed coarse-to-fine result is still immutable numerical evidence.
    # Reuse only complete factors whose raw SHA-256 values still match, so a
    # finer extension never spends GPU time repeating already certified solves.
    prior_path = OUT / "Z_MESH_CONVERGENCE_SUMMARY.json"
    if prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        prior_checkpoint = prior.get("runsetup", {}).get("checkpoint", {})
        if prior_checkpoint.get("sha256") == sha256(CHECKPOINT):
            prior_cases = prior.get("case_results", [])
            prior_sources = prior.get("source_calibration_cases", [])
            for factor in LEVELS:
                selected_cases = [
                    row for row in prior_cases if int(row["factor"]) == factor
                ]
                selected_sources = [
                    row for row in prior_sources if int(row["factor"]) == factor
                ]
                expected_cases = len(densities) * len(POLARIZATIONS)
                valid = (
                    len(selected_cases) == expected_cases
                    and len(selected_sources) == len(POLARIZATIONS)
                )
                if valid:
                    for row in selected_cases:
                        raw_path = Path(str(row["raw_path"]))
                        if (
                            not raw_path.exists()
                            or raw_path.stat().st_size != int(row["raw_bytes"])
                            or sha256(raw_path) != row["raw_sha256"]
                        ):
                            valid = False
                            break
                if not valid:
                    continue
                source_rows.extend(selected_sources)
                case_rows.extend(selected_cases)
                incident_reference[factor] = float(
                    np.mean([float(row["incident_power_W"]) for row in selected_sources])
                )
                for row in selected_cases:
                    raw_path = Path(str(row["raw_path"]))
                    with np.load(raw_path, allow_pickle=False) as raw:
                        source_power = np.asarray(raw["source_power_W"], dtype=np.float64)
                        ta_temperature = np.asarray(
                            raw["ta_temperature_K"], dtype=np.float64
                        )
                    thermal_state = build_thermal_state(densities[str(row["density_case"])])
                    stored[(factor, str(row["density_case"]), str(row["polarization"]))] = {
                        "row": row,
                        "source_power_W": source_power,
                        "ta_temperature_K": ta_temperature,
                        "thermal_volume_m3": thermal_volume(thermal_state),
                    }
                cached_factors.add(factor)
                print(f"[resume] reusing complete factor {factor}", flush=True)

    for factor in LEVELS:
        if factor in cached_factors:
            continue
        source_powers = []
        for polarization in POLARIZATIONS:
            power, runtime = source_only_power(factor, polarization)
            source_powers.append(power)
            source_rows.append(
                {
                    "factor": factor,
                    "polarization": polarization,
                    "incident_power_W": power,
                    "runtime_s": runtime,
                }
            )
            print(
                f"[source] f={factor} {polarization}: {power:.9e} W, {runtime:.2f} s",
                flush=True,
            )
        mismatch = abs(source_powers[0] - source_powers[1]) / max(source_powers)
        if mismatch >= GATE_POWER:
            raise RuntimeError(f"source polarization mismatch at factor {factor}: {mismatch}")
        incident_reference[factor] = float(np.mean(source_powers))

        for polarization in POLARIZATIONS:
            runner = ForwardRunner.create(factor, polarization)
            audit = next(item for item in audits if item["factor"] == factor)
            source_scale = CONTRACT.reporting_incident_power_W / incident_reference[factor]
            for density_name, rho in densities.items():
                _, q, components, p_q, p_six, closure, runtime = runner.run(
                    rho, source_scale
                )
                thermal_state = build_thermal_state(rho)
                source_power, mapping, _ = map_native_q_to_thermal(
                    thermal_state,
                    q_fields_W_m3=q,
                    dual_volumes_m3=runner.volumes,
                    material_slices={
                        "au": runner.model["slices"]["au_design"],
                        "tairte4": runner.model["slices"]["fixed_tairte4"],
                    },
                    realized_grid=runner.model["grid"],
                )
                evaluated = evaluate_fixed_source(
                    rho, source_power, cuda_device, need_gradient=False
                )
                ta_temperature = tairte4_temperature(
                    evaluated["state"], evaluated["temperature"]
                )
                raw_path = RAW / f"f{factor}_{density_name}_{polarization}.npz"
                np.savez(
                    raw_path,
                    source_power_W=np.asarray(source_power, dtype=np.float64),
                    ta_temperature_K=np.asarray(ta_temperature, dtype=np.float64),
                    depth_power_W=np.sum(source_power, axis=(0, 1)),
                )
                row = {
                    "factor": factor,
                    "density_case": density_name,
                    "polarization": polarization,
                    "sio2_dz_nm": 95.0 / factor,
                    "tairte4_dz_nm": 20.0 / factor,
                    "au_dz_nm": 25.0 / factor,
                    "yee_cell_count": audit["yee_cell_count"],
                    "compile_s": runner.compile_s,
                    "forward_s": runtime,
                    "P_Q_W": p_q,
                    "P_six_W": p_six,
                    "closure_relative": closure,
                    **components,
                    "mapping_error_relative": float(
                        max(record["relative_error"] for record in mapping.values())
                    ),
                    "Tmax_K": float(np.max(evaluated["temperature"])),
                    "current_A": float(evaluated["objective_A"]),
                    "current_nA": 1.0e9 * float(evaluated["objective_A"]),
                    "thermal_energy_balance_relative": float(
                        evaluated["thermal_audit"]["energy_balance_relative"]
                    ),
                    "thermal_residual_relative": float(
                        evaluated["thermal_audit"]["relative_residual"]
                    ),
                    "electrical_residual_relative": float(
                        evaluated["electrical_audit"]["relative_residual"]
                    ),
                    "raw_path": str(raw_path.resolve()),
                    "raw_bytes": raw_path.stat().st_size,
                    "raw_sha256": sha256(raw_path),
                }
                row["physics_gates_pass"] = bool(
                    closure < GATE_POWER
                    and row["mapping_error_relative"] < 5.0e-3
                    and row["thermal_energy_balance_relative"] < 1.0e-2
                    and row["thermal_residual_relative"] < 1.0e-8
                    and row["electrical_residual_relative"] < 1.0e-8
                )
                case_rows.append(row)
                stored[(factor, density_name, polarization)] = {
                    "row": row,
                    "source_power_W": source_power,
                    "ta_temperature_K": ta_temperature,
                    "thermal_volume_m3": thermal_volume(evaluated["state"]),
                    "depth_power_W": np.sum(source_power, axis=(0, 1)),
                }
                print(
                    f"[case] f={factor} {density_name} {polarization}: "
                    f"Pq={p_q:.9e} W closure={100*closure:.4f}% "
                    f"I={row['current_nA']:+.6f} nA",
                    flush=True,
                )

    comparison_rows: list[dict[str, object]] = []
    for coarse, fine in zip(LEVELS[:-1], LEVELS[1:]):
        for density_name in densities:
            for polarization in POLARIZATIONS:
                left = stored[(coarse, density_name, polarization)]
                right = stored[(fine, density_name, polarization)]
                lrow, rrow = left["row"], right["row"]
                component_changes = {}
                for material in ("au", "tairte4"):
                    for axis in ("x", "y", "z"):
                        key = f"{material}_{axis}_W"
                        component_changes[f"{key}_change_over_total"] = (
                            abs(float(lrow[key]) - float(rrow[key]))
                            / max(abs(float(rrow["P_Q_W"])), np.finfo(float).tiny)
                        )
                row = {
                    "coarse_factor": coarse,
                    "fine_factor": fine,
                    "density_case": density_name,
                    "polarization": polarization,
                    "P_Q_relative_change": relative(lrow["P_Q_W"], rrow["P_Q_W"]),
                    "P_six_relative_change": relative(
                        lrow["P_six_W"], rrow["P_six_W"]
                    ),
                    "remapped_Q_volume_L2_NRMSE": volume_l2(
                        left["source_power_W"],
                        right["source_power_W"],
                        right["thermal_volume_m3"],
                    ),
                    "Ta_temperature_NRMSE": float(
                        np.linalg.norm(
                            left["ta_temperature_K"] - right["ta_temperature_K"]
                        )
                        / max(
                            np.linalg.norm(right["ta_temperature_K"]),
                            np.finfo(float).tiny,
                        )
                    ),
                    "Tmax_relative_change": relative(lrow["Tmax_K"], rrow["Tmax_K"]),
                    "current_relative_change": relative(
                        lrow["current_A"], rrow["current_A"]
                    ),
                    "current_sign_preserved": bool(
                        np.sign(lrow["current_A"]) == np.sign(rrow["current_A"])
                    ),
                    **component_changes,
                }
                row["comparison_pass"] = bool(
                    row["P_Q_relative_change"] < GATE_POWER
                    and row["P_six_relative_change"] < GATE_POWER
                    and row["remapped_Q_volume_L2_NRMSE"] < GATE_FIELD
                    and row["Ta_temperature_NRMSE"] < GATE_FIELD
                    and row["Tmax_relative_change"] < GATE_FIELD
                    and row["current_relative_change"] < GATE_CURRENT
                    and row["current_sign_preserved"]
                    and all(value < GATE_POWER for value in component_changes.values())
                )
                comparison_rows.append(row)

    final_pair = [
        row
        for row in comparison_rows
        if row["coarse_factor"] == LEVELS[-2]
        and row["fine_factor"] == LEVELS[-1]
    ]
    physics_pass = all(bool(row["physics_gates_pass"]) for row in case_rows)
    convergence_pass = all(bool(row["comparison_pass"]) for row in final_pair)
    status = (
        "VALIDATED_4UM_AU_Z_MESH_CONVERGENCE"
        if physics_pass and convergence_pass
        else "BLOCKED_4UM_AU_Z_MESH_CONVERGENCE"
    )

    with (OUT / "z_mesh_cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(case_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(case_rows)
    with (OUT / "z_mesh_comparisons.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(comparison_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    colors = {"Ea": "tab:blue", "Eb": "tab:orange"}
    for column, density_name in enumerate(densities):
        for polarization in POLARIZATIONS:
            selected = [
                row
                for row in case_rows
                if row["density_case"] == density_name
                and row["polarization"] == polarization
            ]
            selected.sort(key=lambda item: item["factor"])
            x = [item["au_dz_nm"] for item in selected]
            axes[0, column].plot(
                x,
                [item["current_nA"] for item in selected],
                "o-",
                color=colors[polarization],
                label=polarization,
            )
            axes[1, column].plot(
                x,
                [1.0e6 * item["P_Q_W"] for item in selected],
                "o-",
                color=colors[polarization],
                label=polarization,
            )
        axes[0, column].axhline(0.0, color="black", lw=0.8)
        axes[0, column].set_title(density_name)
        axes[0, column].set_ylabel("PTE current (nA)")
        axes[1, column].set_ylabel("absorbed power (uW)")
        axes[1, column].set_xlabel("Au dz (nm; decreasing is finer)")
        axes[0, column].invert_xaxis()
        axes[1, column].invert_xaxis()
        axes[0, column].legend()
        axes[1, column].legend()
    fig.savefig(OUT / "z_mesh_current_power_convergence.png", dpi=180)
    plt.close(fig)

    metric_names = (
        "P_Q_relative_change",
        "remapped_Q_volume_L2_NRMSE",
        "Ta_temperature_NRMSE",
        "current_relative_change",
    )
    labels = [f"{row['density_case']}\n{row['polarization']}" for row in final_pair]
    x = np.arange(len(final_pair))
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    width = 0.18
    for index, metric in enumerate(metric_names):
        ax.bar(
            x + (index - 1.5) * width,
            [100.0 * float(row[metric]) for row in final_pair],
            width=width,
            label=metric,
        )
    ax.axhline(100.0 * GATE_FIELD, color="black", ls="--", label="0.5% gate")
    ax.set_xticks(x, labels)
    ax.set_ylabel("relative change / NRMSE (%)")
    ax.set_title("Final z-mesh refinement pair")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "z_mesh_final_pair_metrics.png", dpi=180)
    plt.close(fig)

    summary = {
        "status": status,
        "scope": (
            "fixed beta=256 robust checkpoint; historical O3/TE1 law; "
            "z-only FDTDX refinement with per-mesh all-air source calibration; "
            "identical conservative thermal/electrical downstream operator"
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runsetup": runsetup,
        "source_calibration_cases": source_rows,
        "case_results": case_rows,
        "comparison_results": comparison_rows,
        "gates": {
            "power_relative": GATE_POWER,
            "field_NRMSE": GATE_FIELD,
            "current_relative": GATE_CURRENT,
            "physics_gates_pass": physics_pass,
            "final_pair_convergence_pass": convergence_pass,
        },
        "next_gate": (
            "OPTICAL_XY_AND_COMBINED_GRADIENT_CONVERGENCE"
            if status.startswith("VALIDATED_")
            else "DIAGNOSE_TIME_AND_ABSORPTION_CLOSURE_THEN_DEFINE_FULL_Z_SWEEP"
        ),
    }
    write_json(OUT / "Z_MESH_CONVERGENCE_SUMMARY.json", summary)

    lines = [
        "# 4 um Au/TaIrTe4 z-mesh convergence",
        "",
        f"Status: `{status}`",
        "",
        "The exact current Au/FDTDX checkpoint had no prior z-mesh convergence certificate.",
        "AD-FD on the baseline grid certifies differentiation of that discrete grid only.",
        "",
        "The density, x/y mesh, source, material endpoints, and historical O3/TE1 gray law are frozen.",
        "The gray law remains diagnostic and is not promoted as physical Au.",
        "",
        "| factor | Au dz (nm) | TaIrTe4 dz (nm) | SiO2 dz (nm) | Yee cells |",
        "|---:|---:|---:|---:|---:|",
    ]
    for audit in audits:
        lines.append(
            f"| {audit['factor']} | {1e9*audit['au_dz_m']:.3f} | "
            f"{1e9*audit['tairte4_dz_m']:.3f} | {1e9*audit['sio2_dz_m']:.3f} | "
            f"{audit['yee_cell_count']} |"
        )
    failed_physics = [row for row in case_rows if not row["physics_gates_pass"]]
    lines.extend(
        [
            "",
            "## Independent physics-gate failures",
            "",
            f"Overall physics gates pass: `{physics_pass}`.  "
            "The table lists every case that failed before pairwise convergence was considered.",
            "",
            "| factor | density | pol | Q/flux closure | mapping | thermal balance | thermal residual | electrical residual |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in failed_physics:
        lines.append(
            f"| {row['factor']} | {row['density_case']} | {row['polarization']} | "
            f"{100*row['closure_relative']:.4f}% | "
            f"{row['mapping_error_relative']:.3e} | "
            f"{row['thermal_energy_balance_relative']:.3e} | "
            f"{row['thermal_residual_relative']:.3e} | "
            f"{row['electrical_residual_relative']:.3e} |"
        )
    lines.extend(
        [
            "",
            "## Final refinement-pair comparison",
            "",
            "| density | pol | dP_Q | Q NRMSE | T NRMSE | dTmax | dI | sign | pass |",
            "|---|---|---:|---:|---:|---:|---:|:---:|:---:|",
        ]
    )
    for row in final_pair:
        lines.append(
            f"| {row['density_case']} | {row['polarization']} | "
            f"{100*row['P_Q_relative_change']:.4f}% | "
            f"{100*row['remapped_Q_volume_L2_NRMSE']:.4f}% | "
            f"{100*row['Ta_temperature_NRMSE']:.4f}% | "
            f"{100*row['Tmax_relative_change']:.4f}% | "
            f"{100*row['current_relative_change']:.4f}% | "
            f"{row['current_sign_preserved']} | {row['comparison_pass']} |"
        )
    lines.extend(
        [
            "",
            "No Q clipping, smoothing, gain, polarization matching, or closure rescaling is used.",
            "Each optical mesh is normalized only by its own all-air incident-power calibration.",
            "This was a partial z sweep: it refined Au, TaIrTe4, and SiO2 only. The Si substrate,",
            "air regions, and z-PML discretization were fixed, so the failed result is not a",
            "certificate for the full optical z domain. Time-window stationarity was also not",
            "measured by this run and must be separated from spatial error before extending it.",
            "",
            f"Next gate: `{summary['next_gate']}`.",
        ]
    )
    (OUT / "Z_MESH_CONVERGENCE_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "final_pair": final_pair}, indent=2), flush=True)
    return 0 if status.startswith("VALIDATED_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
