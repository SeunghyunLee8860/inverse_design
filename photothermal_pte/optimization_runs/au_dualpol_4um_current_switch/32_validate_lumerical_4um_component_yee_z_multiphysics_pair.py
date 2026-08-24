#!/usr/bin/env python3
"""Validate component-Yee material Q and custom CUDA downstream convergence."""

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
    run_component_yee_downstream,
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


def _require_result_raw_hash(result: dict[str, object], raw_path: Path) -> None:
    expected = None
    for artifact in result.get("raw_artifacts", []):
        if Path(artifact.get("path", "")).resolve() == raw_path.resolve():
            expected = artifact.get("sha256")
            break
    if expected is None:
        raise RuntimeError(f"result JSON does not attest raw NPZ: {raw_path}")
    if sha256(raw_path) != expected:
        raise RuntimeError(f"raw NPZ SHA256 differs from result JSON: {raw_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-json", required=True, type=Path)
    parser.add_argument("--fine-json", required=True, type=Path)
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
        raise RuntimeError("component-Yee pair validator accepts only empty/full")
    rho = np.full(CONTRACT.design_shape, 1.0 if case == "full" else 0.0)
    result_paths = {
        "coarse": args.coarse_json.resolve(),
        "fine": args.fine_json.resolve(),
    }
    results = {label: _load_json(path) for label, path in result_paths.items()}
    raw_paths = {
        "coarse": Path(comparison["artifacts"]["coarse_raw_npz"]).resolve(),
        "fine": Path(comparison["artifacts"]["fine_raw_npz"]).resolve(),
    }
    for label in ("coarse", "fine"):
        if not raw_paths[label].is_file():
            raise FileNotFoundError(raw_paths[label])
        _require_result_raw_hash(results[label], raw_paths[label])
    reporting_values = {
        float(result["reporting_normalization"]["target_reporting_incident_power_W"])
        for result in results.values()
    }
    if reporting_values != {CONTRACT.reporting_incident_power_W}:
        raise RuntimeError("result reporting power does not match current contract")

    rows: dict[str, dict[str, object]] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for label in ("coarse", "fine"):
        print(
            f"[{case}] {label} component-Yee Q -> custom CUDA thermal/electrical",
            flush=True,
        )
        rows[label], arrays[label] = run_component_yee_downstream(
            results[label], raw_paths[label], rho, 0, case
        )
    state = build_thermal_state(rho)
    metrics, metric_gates = downstream_metrics(
        coarse_power_W=arrays["coarse"]["source_power_W"],
        fine_power_W=arrays["fine"]["source_power_W"],
        cell_volume_m3=thermal_cell_volumes(state.edges),
        coarse_ta_temperature_K=arrays["coarse"]["TaIrTe4_temperature_K"],
        fine_ta_temperature_K=arrays["fine"]["TaIrTe4_temperature_K"],
        coarse_tmax_K=float(rows["coarse"]["Tmax_K"]),
        fine_tmax_K=float(rows["fine"]["Tmax_K"]),
        coarse_current_A=float(rows["coarse"]["current_A"]),
        fine_current_A=float(rows["fine"]["current_A"]),
        coarse_current_absolute_scale_A=float(
            rows["coarse"]["current_absolute_integrand_scale_A"]
        ),
        fine_current_absolute_scale_A=float(
            rows["fine"]["current_absolute_integrand_scale_A"]
        ),
        expect_zero_current=True,
    )
    all_passed = bool(
        rows["coarse"]["all_gates_passed"]
        and rows["fine"]["all_gates_passed"]
        and all(metric_gates.values())
    )
    output.mkdir(parents=True, exist_ok=True)
    raw_output = output / f"{case}_component_yee_z_pair_raw.npz"
    np.savez_compressed(
        raw_output,
        **{
            f"{label}_{key}": value
            for label in ("coarse", "fine")
            for key, value in arrays[label].items()
        },
    )
    payload = {
        "schema": "lumerical-4um-component-yee-z-multiphysics-pair-v1",
        "status": (
            "PASSED_LUMERICAL_4UM_COMPONENT_YEE_Z_MULTIPHYSICS_PAIR_DEVELOPMENT"
            if all_passed
            else "BLOCKED_LUMERICAL_4UM_COMPONENT_YEE_Z_MULTIPHYSICS_PAIR_DEVELOPMENT"
        ),
        "case": case,
        "polarization": comparison["contract"]["polarization"],
        "maxwell_comparison": comparison,
        "coarse": rows["coarse"],
        "fine": rows["fine"],
        "metrics": metrics,
        "gates": metric_gates,
        "all_gates_passed": all_passed,
        "raw_artifact": {
            "path": str(raw_output),
            "size_bytes": raw_output.stat().st_size,
            "sha256": sha256(raw_output),
        },
        "solver_contract": {
            "Maxwell": "Lumerical FDTD native component-Yee Q only",
            "thermal": "repository custom CUDA finite-volume steady heat solver",
            "electrical": "repository custom CUDA weighting-potential solver",
            "Lumerical_HEAT_or_CHARGE_used": False,
            "component_material_pairs": [
                "Qx with epsilon_x",
                "Qy with epsilon_y",
                "Qz with epsilon_z",
            ],
            "reporting_incident_power_W": CONTRACT.reporting_incident_power_W,
        },
        "scope": (
            "single symmetric exact empty/full Ea RTX development pair; "
            "not an Eb, optimized topology, or B200 certificate"
        ),
    }
    json_path = output / f"{case}_component_yee_z_pair.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
