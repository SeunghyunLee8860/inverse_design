#!/usr/bin/env python3
"""Validate official Lumerical Pabs remap and custom CUDA downstream output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_control_comparison import (
    compare_control_pair,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_multiphysics_comparison import (
    downstream_metrics,
    thermal_cell_volumes,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_official_downstream import (
    run_official_pabs_downstream,
    validate_official_pabs_npz,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.multiphysics_4um import (
    build_thermal_state,
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
        validate_official_pabs_npz(
            candidate,
            source_result_json=result_json_paths[label],
            source_raw_npz=raw_paths[label],
        )
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
    coarse, coarse_arrays = run_official_pabs_downstream(
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
    fine, fine_arrays = run_official_pabs_downstream(
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
