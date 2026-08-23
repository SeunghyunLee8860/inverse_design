#!/usr/bin/env python3
"""Validate the refitted factor-8 model under the reduced-Courant contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch import (
    fdtdx_4um_model as optical_model,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.historical_checkpoint import (
    CHECKPOINT,
    EXPECTED_CHECKPOINT_SHA256,
    load_densities,
    sha256 as checkpoint_sha256,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.material_fraction import (
    audit as material_fraction_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.mesh_variants import (
    PARTIAL_MATERIAL_Z,
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
ISOLATION_SCRIPT = HERE / "18_diagnose_4um_long_time_instability.py"
OUT = HERE / "results_4um_stable_time_contract"
FACTOR = 8
COURANT_FACTOR = 0.25
WINDOW_PERIODS = 4
POLARIZATION = "Eb"
DENSITY_CASE = "eta_0.35"
CASES = (
    ("au_only_32p", 32, (1.0, 0.0)),
    ("full_dispersion_32p", 32, (1.0, 1.0)),
    ("full_dispersion_40p", 40, (1.0, 1.0)),
)
FIELD_NRMSE_LIMIT = 5.0e-3
FIELD_E2_CHANGE_LIMIT = 5.0e-3
TD_PHASOR_FLUX_LIMIT = 5.0e-3
Q_FLUX_LIMIT = 2.0e-2
FULL_32_40_Q_CHANGE_LIMIT = 5.0e-3


def relative(left: float, right: float) -> float:
    return abs(left - right) / max(abs(right), np.finfo(float).tiny)


def load_cached(name: str, runsetup_sha: str) -> dict[str, object] | None:
    path = OUT / f"{name}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "COMPLETE_STABLE_TIME_CASE"
        or payload.get("runsetup_sha256") != runsetup_sha
        or payload.get("case") != name
    ):
        return None
    row = payload.get("row")
    return row if isinstance(row, dict) else None


def save_cached(name: str, runsetup_sha: str, row: dict[str, object]) -> None:
    payload = {
        "status": "COMPLETE_STABLE_TIME_CASE",
        "runsetup_sha256": runsetup_sha,
        "case": name,
        "row": row,
    }
    path = OUT / f"{name}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def compile_for_periods(isolation, rho: np.ndarray, total_periods: int):
    isolation.FACTOR = FACTOR
    isolation.TOTAL_PERIODS = total_periods
    isolation.WINDOW_PERIODS = WINDOW_PERIODS
    isolation.POLARIZATION = POLARIZATION
    isolation.DENSITY_CASE = DENSITY_CASE
    original = optical_model.build_model

    def reduced_courant_build(*args, **kwargs):
        kwargs["courant_factor"] = COURANT_FACTOR
        return original(*args, **kwargs)

    optical_model.build_model = reduced_courant_build
    try:
        return isolation.compile_runner(rho)
    finally:
        optical_model.build_model = original


def run_group(
    isolation,
    rho: np.ndarray,
    total_periods: int,
    pending: list[tuple[str, tuple[float, float]]],
) -> dict[str, dict[str, object]]:
    model, solve, volumes, compile_s = compile_for_periods(
        isolation, rho, total_periods
    )
    output: dict[str, dict[str, object]] = {}
    for name, activation in pending:
        row = isolation.run_case(name, activation, model, solve, volumes)
        row["total_periods"] = total_periods
        row["window_periods"] = WINDOW_PERIODS
        row["courant_factor"] = COURANT_FACTOR
        row["time_step_s"] = float(model["config"].time_step_duration)
        row["time_steps_total"] = int(model["config"].time_steps_total)
        row["compile_s_shared_for_period_group"] = compile_s
        row["material_fit_relative_error"] = {
            key: float(value["fit_relative_error"])
            for key, value in model["fits"].items()
        }
        row["material_gamma_adjustment_relative"] = {
            key: float(value["gamma_adjustment_relative"])
            for key, value in model["fits"].items()
        }
        row["Q_vs_closed_phasor_relative"] = relative(
            float(row["P_discrete_ADE_Q_late_W"]),
            float(row["P_closed_phasor_W"]),
        )
        row["strict_time_gate"] = {
            "field_NRMSE_limit": FIELD_NRMSE_LIMIT,
            "E2_change_limit": FIELD_E2_CHANGE_LIMIT,
            "TD_phasor_flux_limit": TD_PHASOR_FLUX_LIMIT,
            "Q_phasor_flux_limit": Q_FLUX_LIMIT,
            "passed": bool(
                float(row["complex_E_spatial_NRMSE"]) < FIELD_NRMSE_LIMIT
                and float(row["volume_integrated_E2_change_relative"])
                < FIELD_E2_CHANGE_LIMIT
                and float(row["closed_td_vs_phasor_relative"])
                < TD_PHASOR_FLUX_LIMIT
                and float(row["Q_vs_closed_phasor_relative"])
                < Q_FLUX_LIMIT
            ),
        }
        output[name] = row
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    rho = load_densities()[DENSITY_CASE]
    runsetup = {
        "status": "AUDITED_STABLE_TIME_CONTRACT_NOT_SOLVED",
        "validation_script_sha256": sha256(Path(__file__).resolve()),
        "isolation_implementation_sha256": sha256(ISOLATION_SCRIPT),
        "optical_model_sha256": sha256(Path(optical_model.__file__).resolve()),
        "fdtdx_update_sha256": sha256(
            optical_model.FDTDX_SOURCE / "src/fdtdx/fdtd/update.py"
        ),
        "device_contract_sha256": sha256(DEVICE_CERTIFICATE),
        "checkpoint": {
            "path": str(CHECKPOINT.resolve()),
            "sha256": checkpoint_sha256(),
            "expected_sha256": EXPECTED_CHECKPOINT_SHA256,
        },
        "mesh": variant_audit(FACTOR, PARTIAL_MATERIAL_Z),
        "courant_factor": COURANT_FACTOR,
        "polarization": POLARIZATION,
        "density_case": DENSITY_CASE,
        "cases": [
            {
                "name": name,
                "total_periods": periods,
                "activation_Au_TaIrTe4": list(activation),
            }
            for name, periods, activation in CASES
        ],
        "gates": {
            "complex_E_spatial_NRMSE": FIELD_NRMSE_LIMIT,
            "volume_integrated_E2_change_relative": FIELD_E2_CHANGE_LIMIT,
            "closed_td_vs_phasor_relative": TD_PHASOR_FLUX_LIMIT,
            "Q_vs_closed_phasor_relative": Q_FLUX_LIMIT,
            "full_32_vs_40_Q_change_relative": FULL_32_40_Q_CHANGE_LIMIT,
            "material_fit_relative_error": (
                optical_model.FLOAT32_ADE_REFIT_RELATIVE_TOLERANCE
            ),
        },
        "au_material_fraction": material_fraction_audit(),
        "absorption_loss_basis": optical_model.ABSORPTION_LOSS_BASIS,
        "scope": (
            "time/material/absorption validation on a partial-material-z "
            "diagnostic grid; not a mesh certificate"
        ),
        "promotion": {"is_mesh_certificate": False},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    runsetup_path = OUT / "STABLE_TIME_CONTRACT_RUNSETUP.json"
    runsetup_path.write_text(json.dumps(runsetup, indent=2) + "\n", encoding="utf-8")
    if args.audit_only:
        print(json.dumps(runsetup, indent=2), flush=True)
        return 0
    require_single_visible_gpu()
    runsetup_sha = sha256(runsetup_path)
    isolation = optical_model._load(ISOLATION_SCRIPT, "stable_time_isolation_impl")
    rows_by_name: dict[str, dict[str, object]] = {}
    for total_periods in sorted({case[1] for case in CASES}):
        group = [
            (name, activation)
            for name, periods, activation in CASES
            if periods == total_periods
        ]
        pending = []
        for name, activation in group:
            cached = load_cached(name, runsetup_sha)
            if cached is None:
                pending.append((name, activation))
            else:
                print(f"[stable-time] {name}: verified cache", flush=True)
                rows_by_name[name] = cached
        if pending:
            print(
                f"[stable-time] compiling {total_periods} periods for "
                + ", ".join(name for name, _ in pending),
                flush=True,
            )
            completed = run_group(isolation, rho, total_periods, pending)
            for name, row in completed.items():
                save_cached(name, runsetup_sha, row)
                rows_by_name[name] = row
                print(json.dumps(row, indent=2), flush=True)
    rows = [rows_by_name[name] for name, _, _ in CASES]
    full_32 = rows_by_name["full_dispersion_32p"]
    full_40 = rows_by_name["full_dispersion_40p"]
    q_change = relative(
        float(full_32["P_discrete_ADE_Q_late_W"]),
        float(full_40["P_discrete_ADE_Q_late_W"]),
    )
    all_cases_pass = all(bool(row["strict_time_gate"]["passed"]) for row in rows)
    material_pass = all(
        max(float(value) for value in row["material_fit_relative_error"].values())
        < optical_model.FLOAT32_ADE_REFIT_RELATIVE_TOLERANCE
        for row in rows
    )
    passed = bool(
        all_cases_pass
        and material_pass
        and q_change < FULL_32_40_Q_CHANGE_LIMIT
    )
    status = (
        "VALIDATED_FACTOR8_CF0P25_TIME_MATERIAL_CONTRACT"
        if passed
        else "BLOCKED_FACTOR8_CF0P25_TIME_MATERIAL_CONTRACT"
    )
    summary = {
        "status": status,
        "runsetup": runsetup,
        "runsetup_sha256": runsetup_sha,
        "case_results": rows,
        "full_32_vs_40_Q_change_relative": q_change,
        "all_case_gates_passed": all_cases_pass,
        "material_fit_gate_passed": material_pass,
        "promotion": {"is_mesh_certificate": False},
    }
    (OUT / "STABLE_TIME_CONTRACT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Factor-8 reduced-Courant time/material validation",
        "",
        f"Status: `{status}`",
        "",
        "| case | E NRMSE | E2 change | Q (W) | Q/phasor | TD/phasor | pass |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {100*row['complex_E_spatial_NRMSE']:.4f}% | "
            f"{100*row['volume_integrated_E2_change_relative']:.4f}% | "
            f"{row['P_discrete_ADE_Q_late_W']:.6e} | "
            f"{100*row['Q_vs_closed_phasor_relative']:.4f}% | "
            f"{100*row['closed_td_vs_phasor_relative']:.4f}% | "
            f"{row['strict_time_gate']['passed']} |"
        )
    lines.extend(
        [
            "",
            f"Full-material 32/40-period Q change: {100*q_change:.4f}%.",
            "",
            "This validates a time/material contract only on the partial factor-8 grid.",
            "It is explicitly not an optical, thermal, or electrical mesh certificate.",
        ]
    )
    (OUT / "STABLE_TIME_CONTRACT_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "full_32_vs_40_Q_change_relative": q_change}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
