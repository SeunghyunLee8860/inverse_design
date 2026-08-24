#!/usr/bin/env python3
"""Fail-closed full-domain optical z convergence for the 4 um stack.

The optimized density, x/y grid, source, material endpoints, and shared-linear
Au law are frozen. Every z segment, including Si, air, and both z-PMLs, is
refined. Every optical mesh and polarization receives its own all-air
incident-power calibration before Q is conservatively mapped to the identical
explicit thermal/electrical operator.

This script validates z convergence only. It cannot certify optical x/y,
thermal, or electrical meshes and does not restart optimization.
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
    FULL_DOMAIN_Z,
    mesh_context as variant_mesh_context,
    variant_audit,
    variant_edges,
    variant_layout,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.material_fraction import (
    audit as material_fraction_audit,
    au_material_fraction,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.production_readiness import (
    DEVICE_CERTIFICATE,
)


HERE = Path(__file__).resolve().parent
OUT = HERE / "results_4um_shared_linear_full_z_convergence"
RAW = raw_path("shared_linear_full_z_convergence")
STABLE_TIME_CERTIFICATE = (
    HERE
    / "results_4um_stable_time_contract"
    / "STABLE_TIME_CONTRACT_SUMMARY.json"
)
STABLE_TIME_STATUS = "VALIDATED_FACTOR8_CF0P25_TIME_MATERIAL_CONTRACT"
COMBINED_IMPLEMENTATION = HERE / "combined_4um.py"
MULTIPHYSICS_IMPLEMENTATION = HERE / "multiphysics_4um.py"
TOTAL_PERIODS = 40
WINDOW_PERIODS = 4
COURANT_FACTOR = 0.25
LEVELS = (1, 2, 4)
POLARIZATIONS = ("Ea", "Eb")
GATE_POWER = 5.0e-3
GATE_FIELD = 5.0e-3
GATE_CURRENT = 5.0e-3
GATE_Q_FLUX = 2.0e-2
GATE_TD_PHASOR = 5.0e-3
IMPLEMENTATION_VERSION = "full-domain-z-shared-linear-stable-time-v1"


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
    return variant_edges(factor, FULL_DOMAIN_Z)


def refined_layout(factor: int):
    return variant_layout(factor, FULL_DOMAIN_Z)


def mesh_context(factor: int):
    return variant_mesh_context(factor, FULL_DOMAIN_Z)


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
            courant_factor=COURANT_FACTOR,
            include_adjoint_source=False,
            air_only_source_calibration=air_only,
        )


def mesh_audit(factor: int, model: dict[str, object]) -> dict[str, object]:
    edges = refined_edges(factor)
    widths = tuple(np.diff(value) for value in edges)
    expected_audit = variant_audit(factor, FULL_DOMAIN_Z)
    if list(model["grid"].shape) != expected_audit["grid_shape_xyz"]:
        raise RuntimeError(
            f"factor {factor} realized grid does not match full-domain variant"
        )
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
        "full_domain_variant": expected_audit,
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
    au_imag: float
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
            strength = au_material_fraction(density)
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
        au_imag = float(model["discrete_susceptibility"]["au"].imag)
        ta_imag = np.asarray(
            [
                model["discrete_susceptibility"][axis].imag
                for axis in ("b", "a", "c")
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
            au_imag=au_imag,
            ta_imag=ta_imag,
            compile_s=compile_s,
        )

    def run(self, rho: np.ndarray, source_scale: float):
        start = time.perf_counter()
        output = self.solve(self.model["jnp"].asarray(rho, dtype=self.model["jnp"].float32))
        marker = output.detector_states["au_late"]["phasor"]
        self.model["jax"].block_until_ready(marker)
        runtime = time.perf_counter() - start
        late_fields = {
            "au": np.asarray(output.detector_states["au_late"]["phasor"][0, 0]),
            "tairte4": np.asarray(
                output.detector_states["tairte4_late"]["phasor"][0, 0]
            ),
        }
        previous_fields = {
            "au": np.asarray(
                output.detector_states["au_previous"]["phasor"][0, 0]
            ),
            "tairte4": np.asarray(
                output.detector_states["tairte4_previous"]["phasor"][0, 0]
            ),
        }
        e_au = late_fields["au"]
        e_ta = late_fields["tairte4"]
        strength = np.asarray(
            au_material_fraction(np.asarray(rho, dtype=np.float64)),
            dtype=np.float64,
        )
        q = {
            "au": (
                source_scale
                * self.physical_prefactor
                * self.au_imag
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
        p_closed_td = source_scale * eta0 * float(
            np.mean(
                np.asarray(
                    output.detector_states["material_flux_td"]["poynting_flux"]
                )[:, 0]
            )
        )
        p_closed_phasor = source_scale * eta0 * float(
            np.asarray(
                self.model["placed"]["material_flux"].compute_net_flux(
                    output.detector_states["material_flux"]
                )
            )[0]
        )
        late_e2 = 0.0
        previous_e2 = 0.0
        difference_e2 = 0.0
        for material in ("au", "tairte4"):
            volume = self.volumes[material]
            late_e2 += float(np.sum(np.abs(late_fields[material]) ** 2 * volume))
            previous_e2 += float(
                np.sum(np.abs(previous_fields[material]) ** 2 * volume)
            )
            difference_e2 += float(
                np.sum(
                    np.abs(late_fields[material] - previous_fields[material]) ** 2
                    * volume
                )
            )
        stationarity = {
            "complex_E_spatial_NRMSE": math.sqrt(difference_e2)
            / max(math.sqrt(late_e2), np.finfo(float).tiny),
            "volume_integrated_E2_change_relative": relative(
                late_e2, previous_e2
            ),
        }
        return (
            output,
            q,
            p_components,
            p_total,
            p_closed_td,
            p_closed_phasor,
            relative(p_total, p_closed_td),
            relative(p_total, p_closed_phasor),
            relative(p_closed_td, p_closed_phasor),
            stationarity,
            runtime,
        )


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
    if not STABLE_TIME_CERTIFICATE.is_file():
        raise RuntimeError(
            f"full-z sweep requires stable-time certificate: {STABLE_TIME_CERTIFICATE}"
        )
    stable_time = json.loads(STABLE_TIME_CERTIFICATE.read_text(encoding="utf-8"))
    if stable_time.get("status") != STABLE_TIME_STATUS:
        raise RuntimeError("stable-time certificate has not passed")
    stable_runsetup = stable_time.get("runsetup", {})
    stable_checkpoint = stable_runsetup.get("checkpoint", {})
    if (
        stable_runsetup.get("optical_model_sha256")
        != sha256(Path(optical_model.__file__).resolve())
        or stable_runsetup.get("device_contract_sha256")
        != sha256(DEVICE_CERTIFICATE)
        or stable_checkpoint.get("sha256") != sha256(CHECKPOINT)
    ):
        raise RuntimeError("stable-time certificate is stale for current inputs")
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    densities = load_densities()
    cuda_device = int(os.environ.get("THERMAL_CUDA_DEVICE", "0"))

    audits = []
    for factor in LEVELS:
        audit_model = build_at_mesh(factor, "Ea", air_only=False)
        audits.append(mesh_audit(factor, audit_model))
    case_contract = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "script_sha256": sha256(Path(__file__).resolve()),
        "optical_model_sha256": sha256(Path(optical_model.__file__).resolve()),
        "combined_implementation_sha256": sha256(COMBINED_IMPLEMENTATION),
        "multiphysics_implementation_sha256": sha256(MULTIPHYSICS_IMPLEMENTATION),
        "device_contract_sha256": sha256(DEVICE_CERTIFICATE),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "stable_time_certificate_sha256": sha256(STABLE_TIME_CERTIFICATE),
        "mesh_mode": FULL_DOMAIN_Z,
        "time": {
            "total_periods": TOTAL_PERIODS,
            "phasor_window_periods": WINDOW_PERIODS,
            "courant_factor": COURANT_FACTOR,
        },
        "au_material_fraction": material_fraction_audit(),
        "absorption_density_law": (
            "Q proportional to realized float32 discrete-ADE "
            "susceptibility.imag*abs(E)**2"
        ),
        "current_law": "I=sum(-sigma*S*DeltaT*Deltapsi)",
    }
    case_contract_sha256 = hashlib.sha256(
        json.dumps(case_contract, sort_keys=True).encode("utf-8")
    ).hexdigest()
    runsetup = {
        "status": "AUDITED_FULL_DOMAIN_4UM_AU_Z_MESH_RUNSETUP_NOT_SOLVED",
        "scope": (
            "full optical z refinement including materials, Si, air, and "
            "z-PML; x/y and downstream meshes fixed"
        ),
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
            "courant_factor": COURANT_FACTOR,
        },
        "stable_time_certificate": {
            "path": str(STABLE_TIME_CERTIFICATE.resolve()),
            "sha256": sha256(STABLE_TIME_CERTIFICATE),
            "status": stable_time["status"],
        },
        "density_cases": list(densities),
        "case_contract": case_contract,
        "case_contract_sha256": case_contract_sha256,
        "promotion": {"is_full_z_mesh_certificate": False},
    }
    write_json(OUT / "FULL_Z_RUNSETUP.json", runsetup)
    if args.audit_only:
        print(json.dumps(runsetup, indent=2), flush=True)
        return 0

    source_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    stored: dict[tuple[int, str, str], dict[str, object]] = {}
    incident_reference: dict[tuple[int, str], float] = {}
    cached_sources: set[tuple[int, str]] = set()
    cached_cases: set[tuple[int, str, str]] = set()
    progress_path = OUT / "FULL_Z_PROGRESS.json"

    def save_progress() -> None:
        payload = {
            "status": "IN_PROGRESS_FULL_DOMAIN_Z_CONVERGENCE",
            "case_contract_sha256": case_contract_sha256,
            "source_calibration_cases": source_rows,
            "case_results": case_rows,
        }
        temporary = progress_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(progress_path)

    if progress_path.is_file():
        prior = json.loads(progress_path.read_text(encoding="utf-8"))
        if prior.get("case_contract_sha256") == case_contract_sha256:
            for row in prior.get("source_calibration_cases", []):
                key = (int(row["factor"]), str(row["polarization"]))
                if key in cached_sources or key[0] not in LEVELS or key[1] not in POLARIZATIONS:
                    raise RuntimeError(f"invalid cached source key {key}")
                power = float(row["incident_power_W"])
                if not np.isfinite(power) or power <= 0.0:
                    raise RuntimeError(f"invalid cached source power for {key}")
                source_rows.append(row)
                incident_reference[key] = power
                cached_sources.add(key)
            for row in prior.get("case_results", []):
                key = (
                    int(row["factor"]),
                    str(row["density_case"]),
                    str(row["polarization"]),
                )
                if (
                    key in cached_cases
                    or key[0] not in LEVELS
                    or key[1] not in densities
                    or key[2] not in POLARIZATIONS
                    or row.get("case_contract_sha256") != case_contract_sha256
                ):
                    raise RuntimeError(f"invalid cached material-case key {key}")
                saved_raw_path = Path(str(row["raw_path"]))
                if (
                    not saved_raw_path.is_file()
                    or saved_raw_path.stat().st_size != int(row["raw_bytes"])
                    or sha256(saved_raw_path) != row["raw_sha256"]
                ):
                    raise RuntimeError(f"cached raw artifact failed provenance: {key}")
                with np.load(saved_raw_path, allow_pickle=False) as raw:
                    source_power = np.asarray(raw["source_power_W"], dtype=np.float64)
                    ta_temperature = np.asarray(raw["ta_temperature_K"], dtype=np.float64)
                thermal_state = build_thermal_state(densities[key[1]])
                case_rows.append(row)
                stored[key] = {
                    "row": row,
                    "source_power_W": source_power,
                    "ta_temperature_K": ta_temperature,
                    "thermal_volume_m3": thermal_volume(thermal_state),
                }
                cached_cases.add(key)
            print(
                f"[resume] verified {len(cached_sources)} source and "
                f"{len(cached_cases)} material cases",
                flush=True,
            )

    for factor in LEVELS:
        source_powers = []
        for polarization in POLARIZATIONS:
            source_key = (factor, polarization)
            if source_key in cached_sources:
                source_powers.append(incident_reference[source_key])
                print(f"[resume] source f={factor} {polarization}", flush=True)
                continue
            power, runtime = source_only_power(factor, polarization)
            source_powers.append(power)
            row = {
                "factor": factor,
                "polarization": polarization,
                "incident_power_W": power,
                "runtime_s": runtime,
                "grid_edges_sha256": next(
                    item for item in audits if item["factor"] == factor
                )["full_domain_variant"]["grid_edges_sha256"],
                "courant_factor": COURANT_FACTOR,
                "total_periods": TOTAL_PERIODS,
                "window_periods": WINDOW_PERIODS,
            }
            source_rows.append(row)
            incident_reference[source_key] = float(power)
            cached_sources.add(source_key)
            save_progress()
            print(
                f"[source] f={factor} {polarization}: {power:.9e} W, {runtime:.2f} s",
                flush=True,
            )
        mismatch = abs(source_powers[0] - source_powers[1]) / max(source_powers)
        if mismatch >= GATE_POWER:
            raise RuntimeError(f"source polarization mismatch at factor {factor}: {mismatch}")
        for polarization, power in zip(POLARIZATIONS, source_powers, strict=True):
            incident_reference[(factor, polarization)] = float(power)

        for polarization in POLARIZATIONS:
            pending_density_names = [
                name
                for name in densities
                if (factor, name, polarization) not in cached_cases
            ]
            if not pending_density_names:
                print(f"[resume] all f={factor} {polarization} cases", flush=True)
                continue
            runner = ForwardRunner.create(factor, polarization)
            audit = next(item for item in audits if item["factor"] == factor)
            source_scale = (
                CONTRACT.reporting_incident_power_W
                / incident_reference[(factor, polarization)]
            )
            for density_name, rho in densities.items():
                case_key = (factor, density_name, polarization)
                if case_key in cached_cases:
                    continue
                (
                    _,
                    q,
                    components,
                    p_q,
                    p_closed_td,
                    p_closed_phasor,
                    q_td_closure,
                    q_phasor_closure,
                    td_phasor_difference,
                    stationarity,
                    runtime,
                ) = runner.run(rho, source_scale)
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
                    "case_contract_sha256": case_contract_sha256,
                    "factor": factor,
                    "density_case": density_name,
                    "polarization": polarization,
                    "grid_edges_sha256": audit["full_domain_variant"][
                        "grid_edges_sha256"
                    ],
                    "source_incident_power_W": incident_reference[
                        (factor, polarization)
                    ],
                    "source_power_scale": source_scale,
                    "courant_factor": COURANT_FACTOR,
                    "total_periods": TOTAL_PERIODS,
                    "window_periods": WINDOW_PERIODS,
                    "sio2_dz_nm": 95.0 / factor,
                    "tairte4_dz_nm": 20.0 / factor,
                    "au_dz_nm": 25.0 / factor,
                    "yee_cell_count": audit["yee_cell_count"],
                    "compile_s": runner.compile_s,
                    "forward_s": runtime,
                    "P_Q_W": p_q,
                    "P_closed_td_W": p_closed_td,
                    "P_closed_phasor_W": p_closed_phasor,
                    "Q_vs_closed_td_relative": q_td_closure,
                    "Q_vs_closed_phasor_relative": q_phasor_closure,
                    "closed_td_vs_phasor_relative": td_phasor_difference,
                    **stationarity,
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
                    q_td_closure < GATE_Q_FLUX
                    and q_phasor_closure < GATE_Q_FLUX
                    and td_phasor_difference < GATE_TD_PHASOR
                    and stationarity["complex_E_spatial_NRMSE"] < GATE_FIELD
                    and stationarity["volume_integrated_E2_change_relative"]
                    < GATE_FIELD
                    and row["mapping_error_relative"] < 5.0e-3
                    and row["thermal_energy_balance_relative"] < 1.0e-2
                    and row["thermal_residual_relative"] < 1.0e-8
                    and row["electrical_residual_relative"] < 1.0e-8
                )
                case_rows.append(row)
                stored[case_key] = {
                    "row": row,
                    "source_power_W": source_power,
                    "ta_temperature_K": ta_temperature,
                    "thermal_volume_m3": thermal_volume(evaluated["state"]),
                    "depth_power_W": np.sum(source_power, axis=(0, 1)),
                }
                cached_cases.add(case_key)
                save_progress()
                print(
                    f"[case] f={factor} {density_name} {polarization}: "
                    f"Pq={p_q:.9e} W Q/flux={100*q_phasor_closure:.4f}% "
                    f"stationarity={100*stationarity['complex_E_spatial_NRMSE']:.4f}% "
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
                    "P_closed_phasor_relative_change": relative(
                        lrow["P_closed_phasor_W"], rrow["P_closed_phasor_W"]
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
                    and row["P_closed_phasor_relative_change"] < GATE_POWER
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
        "VALIDATED_SHARED_LINEAR_FULL_DOMAIN_Z_CONVERGENCE"
        if physics_pass and convergence_pass
        else "BLOCKED_SHARED_LINEAR_FULL_DOMAIN_Z_CONVERGENCE"
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

    selected_source_rows = [
        row for row in source_rows if int(row["factor"]) == LEVELS[-1]
    ]
    selected_source_calibration = (
        {
            "grid_edges_sha256": audits[-1]["full_domain_variant"][
                "grid_edges_sha256"
            ],
            "courant_factor": COURANT_FACTOR,
            "total_periods": TOTAL_PERIODS,
            "window_periods": WINDOW_PERIODS,
            "cases": [
                {
                    "polarization": row["polarization"],
                    "incident_power_W": row["incident_power_W"],
                }
                for row in selected_source_rows
            ],
            "common_reference_incident_power_W": float(
                np.mean(
                    [float(row["incident_power_W"]) for row in selected_source_rows]
                )
            ),
        }
        if physics_pass and convergence_pass
        else None
    )
    summary = {
        "status": status,
        "scope": (
            "fixed beta=256 robust checkpoint; shared-linear Au law; "
            "full-domain-z FDTDX refinement with per-polarization, per-mesh "
            "all-air source calibration; "
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
        "selected_optical_z_contract": (
            {
                "mesh_mode": FULL_DOMAIN_Z,
                "mesh_factor": LEVELS[-1],
                "grid_edges_sha256": audits[-1]["full_domain_variant"][
                    "grid_edges_sha256"
                ],
                "total_periods": TOTAL_PERIODS,
                "window_periods": WINDOW_PERIODS,
                "courant_factor": COURANT_FACTOR,
            }
            if physics_pass and convergence_pass
            else None
        ),
        "selected_source_calibration": selected_source_calibration,
        "promotion": {
            "is_full_z_mesh_certificate": bool(physics_pass and convergence_pass)
        },
        "next_gate": "RUN_OPTICAL_XY_CONVERGENCE",
    }
    summary_path = OUT / "FULL_Z_CONVERGENCE_SUMMARY.json"
    write_json(summary_path, summary)
    completed_progress = {
        "status": "COMPLETE_FULL_DOMAIN_Z_CONVERGENCE",
        "case_contract_sha256": case_contract_sha256,
        "summary_sha256": sha256(summary_path),
        "source_calibration_cases": source_rows,
        "case_results": case_rows,
    }
    write_json(progress_path, completed_progress)

    lines = [
        "# 4 um Au/TaIrTe4 z-mesh convergence",
        "",
        f"Status: `{status}`",
        "",
        "The exact current Au/FDTDX checkpoint had no prior z-mesh convergence certificate.",
        "AD-FD on the baseline grid certifies differentiation of that discrete grid only.",
        "",
        "The density, x/y mesh, source, material endpoints, and shared-linear Au law are frozen.",
        "Every physical z region and both z-PMLs are refined together.",
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
            "| factor | density | pol | Q/phasor | E stationarity | mapping | thermal balance | thermal residual | electrical residual |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in failed_physics:
        lines.append(
            f"| {row['factor']} | {row['density_case']} | {row['polarization']} | "
            f"{100*row['Q_vs_closed_phasor_relative']:.4f}% | "
            f"{100*row['complex_E_spatial_NRMSE']:.4f}% | "
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
            "Each polarization on each optical mesh is normalized only by its own all-air incident-power calibration.",
            "Every material, Si, air, and z-PML segment is refined. Each material run must also pass",
            "previous-versus-late field stationarity and Q/TD/phasor closed-flux consistency before",
            "the final spatial pair is considered.",
            "",
            f"Next gate: `{summary['next_gate']}`.",
        ]
    )
    (OUT / "FULL_Z_CONVERGENCE_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "final_pair": final_pair}, indent=2), flush=True)
    return 0 if status.startswith("VALIDATED_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
