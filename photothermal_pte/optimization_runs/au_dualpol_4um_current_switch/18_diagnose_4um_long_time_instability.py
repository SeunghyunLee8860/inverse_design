#!/usr/bin/env python3
"""Isolate the material driver of the factor-8 long-time FDTD instability."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_4um_model as optical_model,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.combined_4um import (
    EPS0_F_PER_M,
    electric_yee_dual_volumes,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.historical_checkpoint import (
    CHECKPOINT,
    EXPECTED_CHECKPOINT_SHA256,
    load_densities,
    sha256 as checkpoint_sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.material_fraction import (
    audit as material_fraction_audit,
    au_material_fraction,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.mesh_variants import (
    PARTIAL_MATERIAL_Z,
    mesh_context,
    variant_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.production_readiness import (
    DEVICE_CERTIFICATE,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.validation_provenance import (
    require_single_visible_gpu,
    sha256,
)


HERE = Path(__file__).resolve().parent
OUT = HERE / "results_4um_long_time_instability"
FDTDX_POYNTING = (
    optical_model.FDTDX_SOURCE
    / "src/fdtdx/objects/detectors/poynting_flux.py"
)
FACTOR = 8
TOTAL_PERIODS = 32
WINDOW_PERIODS = 4
POLARIZATION = "Eb"
DENSITY_CASE = "eta_0.35"
CASES = {
    "substrates_only": (0.0, 0.0),
    "au_only": (1.0, 0.0),
    "tairte4_only": (0.0, 1.0),
    "full_dispersion": (1.0, 1.0),
}
FIELD_NRMSE_BLOCK = 5.0e-2
FIELD_E2_CHANGE_BLOCK = 1.0e-1


def relative(left: float, right: float) -> float:
    return abs(left - right) / max(abs(right), np.finfo(float).tiny)


def weighted_field_metrics(
    late: dict[str, np.ndarray],
    previous: dict[str, np.ndarray],
    volumes: dict[str, np.ndarray],
) -> dict[str, float]:
    late_e2 = 0.0
    previous_e2 = 0.0
    difference_e2 = 0.0
    for material in ("au", "tairte4"):
        volume = volumes[material]
        late_e2 += float(np.sum(np.abs(late[material]) ** 2 * volume))
        previous_e2 += float(np.sum(np.abs(previous[material]) ** 2 * volume))
        difference_e2 += float(
            np.sum(np.abs(late[material] - previous[material]) ** 2 * volume)
        )
    return {
        "previous_volume_integrated_E2_V2_m": previous_e2,
        "late_volume_integrated_E2_V2_m": late_e2,
        "volume_integrated_E2_change_relative": relative(late_e2, previous_e2),
        "complex_E_spatial_NRMSE": math.sqrt(difference_e2)
        / max(math.sqrt(late_e2), np.finfo(float).tiny),
    }


def load_cached_case(name: str, runsetup_sha: str) -> dict[str, object] | None:
    path = OUT / f"{name}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "COMPLETE_LONG_TIME_ISOLATION_CASE"
        or payload.get("runsetup_sha256") != runsetup_sha
        or payload.get("case") != name
    ):
        return None
    row = payload.get("row")
    return row if isinstance(row, dict) else None


def save_case(name: str, runsetup_sha: str, row: dict[str, object]) -> None:
    payload = {
        "status": "COMPLETE_LONG_TIME_ISOLATION_CASE",
        "runsetup_sha256": runsetup_sha,
        "case": name,
        "row": row,
    }
    path = OUT / f"{name}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def compile_runner(rho: np.ndarray):
    with mesh_context(FACTOR, PARTIAL_MATERIAL_Z):
        model = optical_model.build_model(
            POLARIZATION,
            total_periods=TOTAL_PERIODS,
            window_periods=WINDOW_PERIODS,
            include_adjoint_source=False,
        )
    jax, jnp, fdtdx = model["jax"], model["jnp"], model["fdtdx"]
    spatial_shape = model["fixed_c1"].shape
    ta = {
        key: jnp.zeros(spatial_shape, dtype=jnp.float32)
        for key in ("c1", "c2", "c3")
    }
    au = {
        key: jnp.zeros(spatial_shape, dtype=jnp.float32)
        for key in ("c1", "c2", "c3")
    }
    ta_slice = model["slices"]["fixed_tairte4"]
    au_slice = model["slices"]["au_design"]
    for component, axis in enumerate(("b", "a", "c")):
        for index, key in enumerate(("c1", "c2", "c3")):
            ta[key] = ta[key].at[(0, component, *ta_slice)].set(
                model["coefficients"][axis][index]
            )
    fraction = jnp.asarray(au_material_fraction(rho), dtype=jnp.float32)
    for component in range(3):
        for index, key in enumerate(("c1", "c2")):
            au[key] = au[key].at[(0, component, *au_slice)].set(
                model["coefficients"]["au"][index]
            )
        au["c3"] = au["c3"].at[(0, component, *au_slice)].set(
            model["coefficients"]["au"][2] * fraction[:, :, None]
        )

    def forward(activation):
        au_on, ta_on = activation[0], activation[1]
        arrays = (
            model["base"]
            .reset()
            .aset("dispersive_c1", au_on * au["c1"] + ta_on * ta["c1"])
            .aset("dispersive_c2", au_on * au["c2"] + ta_on * ta["c2"])
            .aset("dispersive_c3", au_on * au["c3"] + ta_on * ta["c3"])
        )
        return fdtdx.run_fdtd(
            arrays,
            model["placed"],
            model["config"],
            model["key"],
            show_progress=False,
        )[1]

    example = jnp.asarray((1.0, 1.0), dtype=jnp.float32)
    start = time.perf_counter()
    solve = jax.jit(forward).lower(example).compile()
    compile_s = time.perf_counter() - start
    volumes = {
        "au": electric_yee_dual_volumes(model["grid"], au_slice),
        "tairte4": electric_yee_dual_volumes(model["grid"], ta_slice),
    }
    return model, solve, volumes, compile_s


def run_case(
    name: str,
    activation: tuple[float, float],
    model,
    solve,
    volumes: dict[str, np.ndarray],
) -> dict[str, object]:
    jnp = model["jnp"]
    start = time.perf_counter()
    output = solve(jnp.asarray(activation, dtype=jnp.float32))
    marker = output.detector_states["au_late"]["phasor"]
    model["jax"].block_until_ready(marker)
    runtime_s = time.perf_counter() - start
    fields = {}
    maxima = {}
    for window in ("previous", "late"):
        fields[window] = {
            material: np.asarray(
                output.detector_states[f"{material}_{window}"]["phasor"][0, 0]
            )
            for material in ("au", "tairte4")
        }
        maxima[window] = {
            material: float(np.max(np.abs(value)))
            for material, value in fields[window].items()
        }
    metrics = weighted_field_metrics(fields["late"], fields["previous"], volumes)
    eta0 = float(model["fdtdx"].constants.eta0)
    closed_td = eta0 * float(
        np.mean(
            np.asarray(
                output.detector_states["material_flux_td"]["poynting_flux"]
            )[:, 0]
        )
    )
    closed_phasor = eta0 * float(
        np.asarray(
            model["placed"]["material_flux"].compute_net_flux(
                output.detector_states["material_flux"]
            )
        )[0]
    )
    prefactor = (
        0.5
        * float(model["omega_rad_s"])
        * EPS0_F_PER_M
        * eta0**2
    )
    fraction = np.asarray(
        au_material_fraction(load_densities()[DENSITY_CASE]), dtype=np.float64
    )
    q_power = 0.0
    if activation[0]:
        q_au = (
            prefactor
            * float(model["discrete_susceptibility"]["au"].imag)
            * fraction[None, :, :, None]
            * np.abs(fields["late"]["au"]) ** 2
        )
        q_power += float(np.sum(q_au * volumes["au"]))
    if activation[1]:
        ta_imag = np.asarray(
            [
                model["discrete_susceptibility"][axis].imag
                for axis in ("b", "a", "c")
            ],
            dtype=np.float64,
        )[:, None, None, None]
        q_ta = prefactor * ta_imag * np.abs(fields["late"]["tairte4"]) ** 2
        q_power += float(np.sum(q_ta * volumes["tairte4"]))
    unstable = bool(
        metrics["complex_E_spatial_NRMSE"] >= FIELD_NRMSE_BLOCK
        or metrics["volume_integrated_E2_change_relative"]
        >= FIELD_E2_CHANGE_BLOCK
    )
    return {
        "case": name,
        "activation_Au_TaIrTe4": list(activation),
        "runtime_s": runtime_s,
        "field_max_abs": maxima,
        **metrics,
        "P_discrete_ADE_Q_late_W": q_power,
        "P_closed_td_W": closed_td,
        "P_closed_phasor_W": closed_phasor,
        "closed_td_vs_phasor_relative": relative(closed_td, closed_phasor),
        "instability_gate": {
            "field_NRMSE_limit": FIELD_NRMSE_BLOCK,
            "E2_change_limit": FIELD_E2_CHANGE_BLOCK,
            "unstable": unstable,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    rho = load_densities()[DENSITY_CASE]
    runsetup = {
        "status": "AUDITED_LONG_TIME_INSTABILITY_RUNSETUP_NOT_SOLVED",
        "diagnostic_script_sha256": sha256(Path(__file__).resolve()),
        "optical_model_sha256": sha256(Path(optical_model.__file__).resolve()),
        "fdtdx_poynting_detector_sha256": sha256(FDTDX_POYNTING),
        "device_contract_sha256": sha256(DEVICE_CERTIFICATE),
        "checkpoint": {
            "path": str(CHECKPOINT.resolve()),
            "sha256": checkpoint_sha256(),
            "expected_sha256": EXPECTED_CHECKPOINT_SHA256,
        },
        "mesh": variant_audit(FACTOR, PARTIAL_MATERIAL_Z),
        "time": {
            "total_periods": TOTAL_PERIODS,
            "window_periods": WINDOW_PERIODS,
        },
        "polarization": POLARIZATION,
        "density_case": DENSITY_CASE,
        "density_min_max_mean": [
            float(np.min(rho)),
            float(np.max(rho)),
            float(np.mean(rho)),
        ],
        "cases": {name: list(value) for name, value in CASES.items()},
        "au_material_fraction": material_fraction_audit(),
        "absorption_loss_basis": optical_model.ABSORPTION_LOSS_BASIS,
        "gates": {
            "complex_E_spatial_NRMSE": FIELD_NRMSE_BLOCK,
            "volume_integrated_E2_change_relative": FIELD_E2_CHANGE_BLOCK,
        },
        "promotion": {"is_mesh_certificate": False},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    runsetup_path = OUT / "LONG_TIME_INSTABILITY_RUNSETUP.json"
    runsetup_path.write_text(json.dumps(runsetup, indent=2) + "\n", encoding="utf-8")
    if args.audit_only:
        print(json.dumps(runsetup, indent=2), flush=True)
        return 0
    require_single_visible_gpu()
    runsetup_sha = sha256(runsetup_path)
    model, solve, volumes, compile_s = compile_runner(rho)
    rows = []
    for name, activation in CASES.items():
        print(f"[isolation] {name}", flush=True)
        row = load_cached_case(name, runsetup_sha)
        if row is None:
            row = run_case(name, activation, model, solve, volumes)
            save_case(name, runsetup_sha, row)
        else:
            print("[isolation] resuming verified aggregate case", flush=True)
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
    unstable = {
        str(row["case"]): bool(row["instability_gate"]["unstable"])
        for row in rows
    }
    if unstable["substrates_only"]:
        driver = "SOURCE_SUBSTRATE_PML_OR_BASE_FDTD"
    elif unstable["au_only"] and unstable["tairte4_only"]:
        driver = "BOTH_AU_AND_TAIRTE4_INDEPENDENTLY"
    elif unstable["au_only"]:
        driver = "AU_DISPERSION"
    elif unstable["tairte4_only"]:
        driver = "TAIRTE4_DISPERSION"
    elif unstable["full_dispersion"]:
        driver = "COUPLED_AU_TAIRTE4_DISPERSION"
    else:
        driver = "NOT_REPRODUCED_AT_32_PERIODS"
    status = f"DIAGNOSED_LONG_TIME_INSTABILITY_DRIVER_{driver}"
    summary = {
        "status": status,
        "runsetup": runsetup,
        "runsetup_sha256": runsetup_sha,
        "compile_s": compile_s,
        "case_results": rows,
        "unstable_cases": unstable,
        "inferred_driver": driver,
        "promotion": {"is_mesh_certificate": False},
    }
    (OUT / "LONG_TIME_INSTABILITY_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Long-time FDTD instability isolation",
        "",
        f"Status: `{status}`",
        "",
        "| case | E spatial NRMSE | E2 change | closed TD (W) | closed phasor (W) | unstable |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {100*row['complex_E_spatial_NRMSE']:.4f}% | "
            f"{100*row['volume_integrated_E2_change_relative']:.4f}% | "
            f"{row['P_closed_td_W']:.6e} | {row['P_closed_phasor_W']:.6e} | "
            f"{row['instability_gate']['unstable']} |"
        )
    lines.extend(
        [
            "",
            f"Inferred driver: `{driver}`.",
            "",
            "This is an isolation diagnostic on the partial factor-8 grid, not a mesh certificate.",
        ]
    )
    (OUT / "LONG_TIME_INSTABILITY_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "unstable_cases": unstable}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
