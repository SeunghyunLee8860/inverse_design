#!/usr/bin/env python3
"""Separate Au z-resolution effects from time-step/Courant instability.

This is a diagnostic, not a mesh-convergence certificate.  Only the Au ADE
coupling is active; TaIrTe4 is deliberately disabled.  The same physical
time, two phasor windows, density, lateral grid, source, substrate, and PML
are used for every case.  Factors 1/2/4/8 at Courant 0.5 expose dependence on
the partially refined material z grid.  A factor-8, Courant-0.25 case changes
only the time step on the finest geometry and tests whether reducing dt
remediates the failure.
"""

from __future__ import annotations

import argparse
import hashlib
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
OUT = HERE / "results_4um_au_courant_z_coupling"
FDTDX_UPDATE = optical_model.FDTDX_SOURCE / "src/fdtdx/fdtd/update.py"
FDTDX_DISPERSION = optical_model.FDTDX_SOURCE / "src/fdtdx/dispersion.py"
TOTAL_PERIODS = 32
WINDOW_PERIODS = 4
POLARIZATION = "Eb"
DENSITY_CASE = "eta_0.35"
CASES = (
    ("partial_f1_cf0p5", 1, 0.5),
    ("partial_f2_cf0p5", 2, 0.5),
    ("partial_f4_cf0p5", 4, 0.5),
    ("partial_f8_cf0p5", 8, 0.5),
    ("partial_f8_cf0p25", 8, 0.25),
)
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
        payload.get("status") != "COMPLETE_AU_COURANT_Z_CASE"
        or payload.get("runsetup_sha256") != runsetup_sha
        or payload.get("case") != name
    ):
        return None
    row = payload.get("row")
    return row if isinstance(row, dict) else None


def save_case(name: str, runsetup_sha: str, row: dict[str, object]) -> None:
    payload = {
        "status": "COMPLETE_AU_COURANT_Z_CASE",
        "runsetup_sha256": runsetup_sha,
        "case": name,
        "row": row,
    }
    path = OUT / f"{name}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def compile_runner(rho: np.ndarray, factor: int, courant_factor: float):
    with mesh_context(factor, PARTIAL_MATERIAL_Z):
        model = optical_model.build_model(
            POLARIZATION,
            total_periods=TOTAL_PERIODS,
            window_periods=WINDOW_PERIODS,
            courant_factor=courant_factor,
            include_adjoint_source=False,
        )
    jax, jnp, fdtdx = model["jax"], model["jnp"], model["fdtdx"]
    au = {
        key: jnp.zeros(model["fixed_c1"].shape, dtype=jnp.float32)
        for key in ("c1", "c2", "c3")
    }
    au_slice = model["slices"]["au_design"]
    fraction = jnp.asarray(au_material_fraction(rho), dtype=jnp.float32)
    for component in range(3):
        au["c1"] = au["c1"].at[(0, component, *au_slice)].set(
            model["coefficients"]["au"][0]
        )
        au["c2"] = au["c2"].at[(0, component, *au_slice)].set(
            model["coefficients"]["au"][1]
        )
        au["c3"] = au["c3"].at[(0, component, *au_slice)].set(
            model["coefficients"]["au"][2] * fraction[:, :, None]
        )

    def forward(c3_scale):
        arrays = (
            model["base"]
            .reset()
            .aset("dispersive_c1", au["c1"])
            .aset("dispersive_c2", au["c2"])
            .aset("dispersive_c3", c3_scale * au["c3"])
        )
        return fdtdx.run_fdtd(
            arrays,
            model["placed"],
            model["config"],
            model["key"],
            show_progress=False,
        )[1]

    start = time.perf_counter()
    solve = jax.jit(forward).lower(jnp.float32(1.0)).compile()
    compile_s = time.perf_counter() - start
    volumes = {
        "au": electric_yee_dual_volumes(model["grid"], au_slice),
        "tairte4": electric_yee_dual_volumes(
            model["grid"], model["slices"]["fixed_tairte4"]
        ),
    }
    return model, solve, volumes, compile_s


def run_case(
    name: str,
    factor: int,
    courant_factor: float,
    rho: np.ndarray,
) -> dict[str, object]:
    model, solve, volumes, compile_s = compile_runner(rho, factor, courant_factor)
    start = time.perf_counter()
    output = solve(model["jnp"].float32(1.0))
    marker = output.detector_states["au_late"]["phasor"]
    model["jax"].block_until_ready(marker)
    runtime_s = time.perf_counter() - start
    fields: dict[str, dict[str, np.ndarray]] = {}
    maxima: dict[str, dict[str, float]] = {}
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
        0.5 * float(model["omega_rad_s"]) * EPS0_F_PER_M * eta0**2
    )
    fraction = np.asarray(au_material_fraction(rho), dtype=np.float64)
    q_au = (
        prefactor
        * float(model["discrete_susceptibility"]["au"].imag)
        * fraction[None, :, :, None]
        * np.abs(fields["late"]["au"]) ** 2
    )
    q_power = float(np.sum(q_au * volumes["au"]))
    failed = bool(
        metrics["complex_E_spatial_NRMSE"] >= FIELD_NRMSE_BLOCK
        or metrics["volume_integrated_E2_change_relative"]
        >= FIELD_E2_CHANGE_BLOCK
    )
    c1, c2, c3 = (
        float(np.float32(value)) for value in model["coefficients"]["au"]
    )
    return {
        "case": name,
        "mesh_factor": factor,
        "mesh_mode": PARTIAL_MATERIAL_Z,
        "courant_factor": courant_factor,
        "time_step_s": float(model["config"].time_step_duration),
        "time_steps_total": int(model["config"].time_steps_total),
        "compile_s": compile_s,
        "runtime_s": runtime_s,
        "au_float32_ADE": {
            "c1": c1,
            "c2": c2,
            "c3": c3,
            "c1_plus_c2": float(np.float32(c1) + np.float32(c2)),
            "discrete_susceptibility_real": float(
                model["discrete_susceptibility"]["au"].real
            ),
            "discrete_susceptibility_imag": float(
                model["discrete_susceptibility"]["au"].imag
            ),
        },
        "field_max_abs": maxima,
        **metrics,
        "P_discrete_ADE_Q_late_W": q_power,
        "P_closed_td_W": closed_td,
        "P_closed_phasor_W": closed_phasor,
        "closed_td_vs_phasor_relative": relative(closed_td, closed_phasor),
        "stationarity_gate": {
            "field_NRMSE_limit": FIELD_NRMSE_BLOCK,
            "E2_change_limit": FIELD_E2_CHANGE_BLOCK,
            "failed": failed,
        },
    }


def classify(rows: list[dict[str, object]]) -> tuple[str, str]:
    by_name = {str(row["case"]): row for row in rows}
    fine = bool(by_name["partial_f8_cf0p5"]["stationarity_gate"]["failed"])
    reduced = bool(
        by_name["partial_f8_cf0p25"]["stationarity_gate"]["failed"]
    )
    if fine and not reduced:
        return (
            "DIAGNOSED_AU_ADE_TIME_STEP_DEPENDENT_INSTABILITY",
            "COURANT_HALVING_REMEDIATES_IDENTICAL_FINE_Z_GEOMETRY",
        )
    if fine and reduced:
        return (
            "BLOCKED_AU_ADE_FAILURE_NOT_REMEDIATED_BY_COURANT_HALVING",
            "COURANT_HALVING_INSUFFICIENT",
        )
    if not fine and reduced:
        return (
            "BLOCKED_NONMONOTONIC_AU_ADE_COURANT_RESPONSE",
            "LOWER_COURANT_CREATED_FAILURE",
        )
    return (
        "PASSED_AU_ONLY_32_PERIOD_STATIONARITY_IN_TESTED_CASES",
        "FACTOR8_FAILURE_NOT_REPRODUCED",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    rho = load_densities()[DENSITY_CASE]
    runsetup = {
        "status": "AUDITED_AU_COURANT_Z_COUPLING_NOT_SOLVED",
        "diagnostic_script_sha256": sha256(Path(__file__).resolve()),
        "optical_model_sha256": sha256(Path(optical_model.__file__).resolve()),
        "fdtdx_update_sha256": sha256(FDTDX_UPDATE),
        "fdtdx_dispersion_sha256": sha256(FDTDX_DISPERSION),
        "device_contract_sha256": sha256(DEVICE_CERTIFICATE),
        "checkpoint": {
            "path": str(CHECKPOINT.resolve()),
            "sha256": checkpoint_sha256(),
            "expected_sha256": EXPECTED_CHECKPOINT_SHA256,
        },
        "time": {
            "total_periods": TOTAL_PERIODS,
            "window_periods": WINDOW_PERIODS,
            "source_profile": "SingleFrequencyProfile with four-period linear ramp",
        },
        "polarization": POLARIZATION,
        "density_case": DENSITY_CASE,
        "density_sha256": hashlib.sha256(
            np.ascontiguousarray(rho, dtype=np.float64).tobytes()
        ).hexdigest(),
        "cases": [
            {
                "name": name,
                "mesh": variant_audit(factor, PARTIAL_MATERIAL_Z),
                "courant_factor": courant,
            }
            for name, factor, courant in CASES
        ],
        "controlled_variables": {
            "active_dispersion": "Au only; TaIrTe4 ADE coupling exactly zero",
            "same_physical_time": True,
            "same_lateral_grid_source_substrate_and_pml": True,
            "factor8_courant_pair_has_identical_spatial_grid": True,
            "incident_power_calibration": (
                "not required for stationarity and same-run Q/flux closure; "
                "no downstream current is evaluated"
            ),
        },
        "au_material_fraction": material_fraction_audit(),
        "absorption_loss_basis": optical_model.ABSORPTION_LOSS_BASIS,
        "gates": {
            "complex_E_spatial_NRMSE": FIELD_NRMSE_BLOCK,
            "volume_integrated_E2_change_relative": FIELD_E2_CHANGE_BLOCK,
        },
        "promotion": {"is_mesh_certificate": False},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    runsetup_path = OUT / "AU_COURANT_Z_RUNSETUP.json"
    runsetup_path.write_text(json.dumps(runsetup, indent=2) + "\n", encoding="utf-8")
    if args.audit_only:
        print(json.dumps(runsetup, indent=2), flush=True)
        return 0
    require_single_visible_gpu()
    runsetup_sha = sha256(runsetup_path)
    rows: list[dict[str, object]] = []
    for name, factor, courant in CASES:
        print(f"[au-courant-z] {name}", flush=True)
        row = load_cached_case(name, runsetup_sha)
        if row is None:
            row = run_case(name, factor, courant, rho)
            save_case(name, runsetup_sha, row)
        else:
            print("[au-courant-z] resuming verified aggregate case", flush=True)
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
    status, inference = classify(rows)
    summary = {
        "status": status,
        "inference": inference,
        "runsetup": runsetup,
        "runsetup_sha256": runsetup_sha,
        "case_results": rows,
        "promotion": {"is_mesh_certificate": False},
    }
    (OUT / "AU_COURANT_Z_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Au-only z/Courant coupling diagnostic",
        "",
        f"Status: `{status}`",
        "",
        f"Inference: `{inference}`",
        "",
        "| case | dt (s) | E spatial NRMSE | E2 change | Q (W) | closed TD (W) | closed phasor (W) | failed |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['time_step_s']:.6e} | "
            f"{100*row['complex_E_spatial_NRMSE']:.4f}% | "
            f"{100*row['volume_integrated_E2_change_relative']:.4f}% | "
            f"{row['P_discrete_ADE_Q_late_W']:.6e} | "
            f"{row['P_closed_td_W']:.6e} | {row['P_closed_phasor_W']:.6e} | "
            f"{row['stationarity_gate']['failed']} |"
        )
    lines.extend(
        [
            "",
            "This Au-only partial-material-z sweep diagnoses time/spatial coupling; it is not a mesh certificate.",
            "A production z-mesh sweep must use the stable time contract and refine the full z domain.",
        ]
    )
    (OUT / "AU_COURANT_Z_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "inference": inference}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
