#!/usr/bin/env python3
"""Validate official Lumerical Pabs remap and custom CUDA downstream output."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_control_comparison import (
    compare_control_pair,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_multiphysics_comparison import (
    downstream_metrics,
    map_lumerical_official_pabs_to_thermal,
    thermal_cell_volumes,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.multiphysics_4um import (
    N_TA,
    build_electrical_system,
    build_thermal_state,
    current_integrand,
    solve_electrical,
    solve_thermal,
    tairte4_temperature,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.validation_provenance import (
    require_single_visible_gpu,
    sha256,
)


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} does not contain a JSON object")
    return payload


def _validate_pabs_companion(
    pabs_path: Path,
    source_result_json: Path,
) -> None:
    audit_path = pabs_path.with_suffix(".json")
    if not audit_path.is_file():
        raise FileNotFoundError(f"official Pabs companion audit is missing: {audit_path}")
    audit = _load_json(audit_path)
    if audit.get("status") != "EXTRACTED_LUMERICAL_OFFICIAL_PABS_INDEX_X":
        raise RuntimeError("official Pabs companion audit did not pass")
    output = audit.get("output_npz")
    source = audit.get("source_result_json")
    if not isinstance(output, dict) or not isinstance(source, dict):
        raise RuntimeError("official Pabs companion provenance is incomplete")
    if Path(output.get("path", "")).resolve() != pabs_path:
        raise RuntimeError("official Pabs companion path does not match its audit")
    if output.get("sha256") != sha256(pabs_path):
        raise RuntimeError("official Pabs companion SHA256 does not match")
    if Path(source.get("path", "")).resolve() != source_result_json:
        raise RuntimeError("official Pabs companion source JSON path does not match")
    if source.get("sha256") != sha256(source_result_json):
        raise RuntimeError("official Pabs companion source JSON SHA256 does not match")


def _run_downstream(
    result: dict[str, object],
    pabs_path: Path,
    rho: np.ndarray,
    cuda_device: int,
    case: str,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    state = build_thermal_state(rho)
    normalization = result["reporting_normalization"]
    incident = float(normalization["source_only_incident_power_W_raw"])
    reporting = float(normalization["target_reporting_incident_power_W"])
    scale = reporting / incident
    fit = result["material_fit_readback"]["materials"]
    material_index_x = {
        material: complex(
            np.sqrt(
                complex(
                    float(fit[material]["axes"]["x"]["fitted_epsilon_at_4um"]["real"]),
                    float(fit[material]["axes"]["x"]["fitted_epsilon_at_4um"]["imag"]),
                )
            )
        )
        for material in ("Au", "TaIrTe4", "SiO2")
    }
    with np.load(pabs_path, allow_pickle=False) as raw:
        source_power, mapping = map_lumerical_official_pabs_to_thermal(
            raw,
            state.edges,
            scale,
            case=case,
            material_index_x=material_index_x,
        )
    expected_power = float(result["P_Q_pabs_W_raw"]) * scale
    native_vs_json = abs(mapping["native_total_power_W"] - expected_power) / max(
        abs(expected_power), np.finfo(float).tiny
    )
    start = time.perf_counter()
    temperature, thermal_audit = solve_thermal(state, source_power, cuda_device)
    ta_temperature = tairte4_temperature(state, temperature)
    electrical = build_electrical_system(rho, ta_temperature)
    psi, current, electrical_audit = solve_electrical(electrical, cuda_device)
    runtime = time.perf_counter() - start
    integrand = current_integrand(ta_temperature, psi)
    integrand_current = float(np.sum(integrand) * CONTRACT.design_pitch_m**2)
    current_absolute_scale = float(
        np.sum(np.abs(integrand)) * CONTRACT.design_pitch_m**2
    )
    current_consistency = abs(integrand_current - current) / max(
        current_absolute_scale, np.finfo(float).tiny
    )
    current_cancellation = abs(current) / max(
        current_absolute_scale, np.finfo(float).tiny
    )
    gates = {
        "mapping_conservation_lt_1e-12": (
            mapping["relative_conservation_error"] < 1.0e-12
        ),
        "official_spatial_Pabs_matches_json_lt_1e-12": native_vs_json < 1.0e-12,
        "official_material_filter_unassigned_absorption_lt_0p5pct": (
            mapping["unassigned_absorption_relative"] < 5.0e-3
        ),
        "official_Pabs_negative_interpolation_artifact_lt_1e-12": (
            mapping["negative_absorption_relative"] < 1.0e-12
        ),
        "thermal_residual_lt_1e-8": thermal_audit["relative_residual"] < 1.0e-8,
        "thermal_energy_balance_lt_1pct": (
            thermal_audit["energy_balance_relative"] < 1.0e-2
        ),
        "electrical_residual_lt_1e-8": (
            electrical_audit["relative_residual"] < 1.0e-8
        ),
        "electrical_terminal_balance_lt_1pct": (
            electrical_audit["terminal_balance_relative"] < 1.0e-2
        ),
        "current_integrand_consistency_lt_1e-12": current_consistency < 1.0e-12,
        "finite": bool(
            np.all(np.isfinite(source_power))
            and np.all(np.isfinite(temperature))
            and np.all(np.isfinite(ta_temperature))
            and np.all(np.isfinite(psi))
            and np.all(np.isfinite(integrand))
        ),
    }
    summary = {
        "runtime_s": runtime,
        "source_scale_to_reporting_power": scale,
        "source_power_W": float(np.sum(source_power)),
        "expected_official_Pabs_at_reporting_power_W": expected_power,
        "official_spatial_Pabs_vs_json_relative": native_vs_json,
        "official_pabs_npz": {
            "path": str(pabs_path),
            "sha256": sha256(pabs_path),
        },
        "mapping": mapping,
        "Tmax_K": float(np.max(temperature)),
        "TaIrTe4_Tmax_K": float(np.max(ta_temperature)),
        "current_A": current,
        "current_nA": current * 1.0e9,
        "current_from_integrand_A": integrand_current,
        "current_absolute_integrand_scale_A": current_absolute_scale,
        "current_integrand_consistency_relative": current_consistency,
        "current_cancellation_relative": current_cancellation,
        "thermal": thermal_audit,
        "electrical": electrical_audit,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
    arrays = {
        "source_power_W": source_power,
        "temperature_K": temperature,
        "TaIrTe4_temperature_K": ta_temperature,
        "weighting_potential_TaIrTe4": psi[: N_TA * N_TA].reshape(N_TA, N_TA),
        "current_integrand_A_m2": integrand,
    }
    return summary, arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-json", required=True, type=Path)
    parser.add_argument("--fine-json", required=True, type=Path)
    parser.add_argument("--coarse-pabs-npz", type=Path)
    parser.add_argument("--fine-pabs-npz", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"downstream output directory already exists: {output}")
    require_single_visible_gpu()
    comparison = compare_control_pair(args.coarse_json, args.fine_json)
    if not comparison["all_gates_passed"]:
        raise RuntimeError("Maxwell scalar/endpoint sub-gate must pass first")
    case = str(comparison["contract"]["case"])
    if case not in {"empty", "full"}:
        raise RuntimeError("downstream z-pair validator currently accepts empty/full")
    rho = np.full(CONTRACT.design_shape, 1.0 if case == "full" else 0.0)
    coarse_result = _load_json(args.coarse_json.resolve())
    fine_result = _load_json(args.fine_json.resolve())
    raw_paths = {
        "coarse": Path(comparison["artifacts"]["coarse_raw_npz"]),
        "fine": Path(comparison["artifacts"]["fine_raw_npz"]),
    }
    requested_pabs = {
        "coarse": args.coarse_pabs_npz,
        "fine": args.fine_pabs_npz,
    }
    pabs_paths: dict[str, Path] = {}
    result_json_paths = {
        "coarse": args.coarse_json.resolve(),
        "fine": args.fine_json.resolve(),
    }
    for label in ("coarse", "fine"):
        candidate = (
            requested_pabs[label].resolve()
            if requested_pabs[label] is not None
            else raw_paths[label].resolve()
        )
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        with np.load(candidate, allow_pickle=False) as probe:
            required = {
                "Pabs_W_m3",
                "Pabs_index_x",
                "Pabs_x_m",
                "Pabs_y_m",
                "Pabs_z_m",
            }
            if not required.issubset(probe.files):
                raise RuntimeError(
                    f"{label} Pabs NPZ lacks official arrays; run "
                    "29_extract_lumerical_4um_official_pabs.py or provide "
                    f"--{label}-pabs-npz"
                )
        if candidate != raw_paths[label].resolve():
            _validate_pabs_companion(candidate, result_json_paths[label])
        pabs_paths[label] = candidate
    reporting_values = {
        float(result["reporting_normalization"]["target_reporting_incident_power_W"])
        for result in (coarse_result, fine_result)
    }
    if reporting_values != {CONTRACT.reporting_incident_power_W}:
        raise RuntimeError("result reporting power does not match current contract")
    print(
        f"[{case}] coarse official Lumerical Pabs -> custom CUDA thermal/electrical",
        flush=True,
    )
    coarse, coarse_arrays = _run_downstream(
        coarse_result,
        pabs_paths["coarse"],
        rho,
        0,
        case,
    )
    print(
        f"[{case}] fine official Lumerical Pabs -> custom CUDA thermal/electrical",
        flush=True,
    )
    fine, fine_arrays = _run_downstream(
        fine_result,
        pabs_paths["fine"],
        rho,
        0,
        case,
    )
    state = build_thermal_state(rho)
    metrics, metric_gates = downstream_metrics(
        coarse_power_W=coarse_arrays["source_power_W"],
        fine_power_W=fine_arrays["source_power_W"],
        cell_volume_m3=thermal_cell_volumes(state.edges),
        coarse_ta_temperature_K=coarse_arrays["TaIrTe4_temperature_K"],
        fine_ta_temperature_K=fine_arrays["TaIrTe4_temperature_K"],
        coarse_tmax_K=float(coarse["Tmax_K"]),
        fine_tmax_K=float(fine["Tmax_K"]),
        coarse_current_A=float(coarse["current_A"]),
        fine_current_A=float(fine["current_A"]),
        coarse_current_absolute_scale_A=float(
            coarse["current_absolute_integrand_scale_A"]
        ),
        fine_current_absolute_scale_A=float(
            fine["current_absolute_integrand_scale_A"]
        ),
        expect_zero_current=case in {"empty", "full"},
    )
    all_passed = bool(
        coarse["all_gates_passed"]
        and fine["all_gates_passed"]
        and all(metric_gates.values())
    )
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / f"{case}_z_multiphysics_pair_raw.npz"
    np.savez_compressed(
        raw_path,
        **{f"coarse_{key}": value for key, value in coarse_arrays.items()},
        **{f"fine_{key}": value for key, value in fine_arrays.items()},
    )
    payload = {
        "schema": "lumerical-4um-z-multiphysics-pair-v2",
        "status": (
            "PASSED_LUMERICAL_4UM_Z_MULTIPHYSICS_PAIR_DEVELOPMENT"
            if all_passed
            else "BLOCKED_LUMERICAL_4UM_Z_MULTIPHYSICS_PAIR_DEVELOPMENT"
        ),
        "case": case,
        "polarization": comparison["contract"]["polarization"],
        "maxwell_comparison": comparison,
        "coarse": coarse,
        "fine": fine,
        "metrics": metrics,
        "gates": metric_gates,
        "all_gates_passed": all_passed,
        "raw_artifact": {
            "path": str(raw_path),
            "size_bytes": raw_path.stat().st_size,
            "sha256": sha256(raw_path),
        },
        "solver_contract": {
            "thermal": "repository custom CUDA finite-volume steady heat solver",
            "electrical": "repository custom CUDA weighting-potential solver",
            "Lumerical_HEAT_or_CHARGE_used": False,
            "reporting_incident_power_W": CONTRACT.reporting_incident_power_W,
            "current_sign": (
                "positive conventional current along solver +x (x_min to x_max)"
            ),
        },
        "scope": (
            "single exact empty/full case and polarization on RTX development "
            "results; not an Eb, simple-L, final-topology, or B200 certificate"
        ),
    }
    json_path = output / f"{case}_z_multiphysics_pair.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
