#!/usr/bin/env python3
"""Triage MCM6 CV0/CV1/staircase at one fixed Lumerical mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.lumerical_4um_interface_comparison import (
    INTERFACE_METHODS,
    compare_normalized_maxwell,
    normalized_maxwell_bundle,
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


CASES = ("empty", "full")
FIXED_MESH_KEYS = (
    "flake_dxy_m",
    "stack_dz_m",
    "bulk_dz_m",
    "outer_dxy_m",
    "mesh_accuracy",
    "pml_layers",
    "lateral_span_m",
    "z_min_m",
    "z_max_m",
    "simulation_time_s",
    "auto_shutoff_min",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} does not contain a JSON object")
    return payload


def _raw_npz(payload: dict[str, Any], result_path: Path) -> tuple[Path, str]:
    records = [
        item
        for item in payload.get("raw_artifacts", [])
        if isinstance(item, dict) and str(item.get("path", "")).endswith("_raw.npz")
    ]
    if len(records) != 1:
        raise RuntimeError(f"{result_path} must name exactly one raw NPZ")
    path = Path(records[0]["path"]).resolve()
    actual = sha256(path)
    if actual != records[0].get("sha256"):
        raise RuntimeError(f"raw NPZ SHA mismatch: {path}")
    return path, actual


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise RuntimeError(f"missing required field: {'.'.join(keys)}")
        value = value[key]
    return value


def _validate_contract(entries: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    first = entries[("cv0", "empty")]["result"]
    common_paths = (
        ("polarization",),
        ("accelerator_policy",),
        ("B200_promotion_certified",),
        ("solver_version",),
        ("GPU_log_evidence", "requested_gpu_uuid"),
    )
    for (method, case), entry in entries.items():
        payload = entry["result"]
        if not payload.get("all_gates_passed") or not str(
            payload.get("status", "")
        ).startswith("PASSED_"):
            raise RuntimeError(f"{method}/{case} result did not pass solver gates")
        source = payload.get("source_calibration_validation")
        if not isinstance(source, dict) or not source.get("passed"):
            raise RuntimeError(f"{method}/{case} lacks passed source calibration")
        if payload.get("case") != case:
            raise RuntimeError(f"entry case mismatch for {method}/{case}")
        mesh = payload.get("mesh_spec")
        if not isinstance(mesh, dict) or mesh.get("conformal_mesh") != INTERFACE_METHODS[
            method
        ]:
            raise RuntimeError(f"entry mesh method mismatch for {method}/{case}")
        source_mesh = _nested(payload, "source_calibration_contract", "mesh_spec")
        if source_mesh != mesh:
            raise RuntimeError(f"source/material mesh contract mismatch for {method}/{case}")
        # The empty control instantiates no Au geometry, so an Au fit setting in
        # an older empty-control JSON is inert.  MCM6 is a physical requirement
        # only for the full-Au member of each interface-method pair.
        if case == "full" and int(
            _nested(payload, "layout", "material_input_audit", "Au_fit", "max_coefficients")
        ) != 6:
            raise RuntimeError(f"{method}/{case} does not use Au MCM6")
        for keys in common_paths:
            if _nested(payload, *keys) != _nested(first, *keys):
                raise RuntimeError(f"cross-method mismatch at {'.'.join(keys)}")

    reference_mesh = entries[("cv0", "empty")]["result"]["mesh_spec"]
    for entry in entries.values():
        mesh = entry["result"]["mesh_spec"]
        for key in FIXED_MESH_KEYS:
            if mesh.get(key) != reference_mesh.get(key):
                raise RuntimeError(f"interface triage changes fixed mesh axis {key}")
    geometry = {}
    for case in CASES:
        hashes = {
            _nested(
                entries[(method, case)]["result"],
                "layout",
                "geometry",
                "exact_au_geometry",
                "geometry_sha256",
            )
            for method in INTERFACE_METHODS
        }
        if len(hashes) != 1:
            raise RuntimeError(f"geometry hash differs across methods for {case}")
        geometry[case] = hashes.pop()
    return {
        "polarization": first["polarization"],
        "accelerator_policy": first["accelerator_policy"],
        "B200_promotion_certified": first["B200_promotion_certified"],
        "solver_version": first["solver_version"],
        "requested_gpu_uuid": first["GPU_log_evidence"]["requested_gpu_uuid"],
        "fixed_mesh": {key: reference_mesh[key] for key in FIXED_MESH_KEYS},
        "geometry_sha256": geometry,
        "methods": INTERFACE_METHODS,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entry",
        action="append",
        nargs=4,
        metavar=("METHOD", "CASE", "RESULT_JSON", "PABS_NPZ"),
        required=True,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    require_single_visible_gpu()

    entries: dict[tuple[str, str], dict[str, Any]] = {}
    for method, case, result_text, pabs_text in args.entry:
        if method not in INTERFACE_METHODS or case not in CASES:
            raise ValueError(f"invalid interface entry: {method}/{case}")
        key = (method, case)
        if key in entries:
            raise ValueError(f"duplicate interface entry: {method}/{case}")
        result_path = Path(result_text).resolve()
        result = _load_json(result_path)
        raw_path, raw_sha = _raw_npz(result, result_path)
        pabs_provenance = validate_official_pabs_npz(
            Path(pabs_text),
            source_result_json=result_path,
            source_raw_npz=raw_path,
        )
        entries[key] = {
            "result_path": result_path,
            "result_sha256": sha256(result_path),
            "result": result,
            "raw_path": raw_path,
            "raw_sha256": raw_sha,
            "pabs_path": Path(pabs_text).resolve(),
            "pabs_provenance": pabs_provenance,
        }
    required = {(method, case) for method in INTERFACE_METHODS for case in CASES}
    if set(entries) != required:
        raise RuntimeError(f"six method/case entries are required; missing {required-set(entries)}")
    contract = _validate_contract(entries)

    summaries: dict[str, dict[str, Any]] = {method: {} for method in INTERFACE_METHODS}
    arrays: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    maxwell: dict[tuple[str, str], dict[str, Any]] = {}
    artifacts: dict[str, dict[str, Any]] = {method: {} for method in INTERFACE_METHODS}
    for method in INTERFACE_METHODS:
        for case in CASES:
            entry = entries[(method, case)]
            rho = np.full(CONTRACT.design_shape, 1.0 if case == "full" else 0.0)
            print(f"[{method}/{case}] official Pabs -> custom CUDA", flush=True)
            summary, case_arrays = run_official_pabs_downstream(
                entry["result"], entry["pabs_path"], rho, 0, case
            )
            summaries[method][case] = summary
            arrays[(method, case)] = case_arrays
            with np.load(entry["raw_path"], allow_pickle=False) as raw:
                maxwell[(method, case)] = normalized_maxwell_bundle(
                    entry["result"], raw
                )
            artifacts[method][case] = {
                "result_json": str(entry["result_path"]),
                "result_json_sha256": entry["result_sha256"],
                "raw_npz": str(entry["raw_path"]),
                "raw_npz_sha256": entry["raw_sha256"],
                "official_pabs": entry["pabs_provenance"],
            }

    pair_order = (("cv0", "staircase"), ("cv1", "staircase"), ("cv0", "cv1"))
    comparisons: dict[str, dict[str, Any]] = {}
    for candidate, reference in pair_order:
        pair_key = f"{candidate}_vs_{reference}"
        comparisons[pair_key] = {}
        for case in CASES:
            maxwell_metrics, maxwell_gates = compare_normalized_maxwell(
                maxwell[(candidate, case)], maxwell[(reference, case)]
            )
            state = build_thermal_state(
                np.full(CONTRACT.design_shape, 1.0 if case == "full" else 0.0)
            )
            candidate_summary = summaries[candidate][case]
            reference_summary = summaries[reference][case]
            candidate_arrays = arrays[(candidate, case)]
            reference_arrays = arrays[(reference, case)]
            multi_metrics, multi_gates = downstream_metrics(
                coarse_power_W=candidate_arrays["source_power_W"],
                fine_power_W=reference_arrays["source_power_W"],
                cell_volume_m3=thermal_cell_volumes(state.edges),
                coarse_ta_temperature_K=candidate_arrays["TaIrTe4_temperature_K"],
                fine_ta_temperature_K=reference_arrays["TaIrTe4_temperature_K"],
                coarse_tmax_K=float(candidate_summary["Tmax_K"]),
                fine_tmax_K=float(reference_summary["Tmax_K"]),
                coarse_current_A=float(candidate_summary["current_A"]),
                fine_current_A=float(reference_summary["current_A"]),
                coarse_current_absolute_scale_A=float(
                    candidate_summary["current_absolute_integrand_scale_A"]
                ),
                fine_current_absolute_scale_A=float(
                    reference_summary["current_absolute_integrand_scale_A"]
                ),
                expect_zero_current=True,
            )
            comparisons[pair_key][case] = {
                "candidate": candidate,
                "reference": reference,
                "Maxwell_metrics": maxwell_metrics,
                "Maxwell_gates": maxwell_gates,
                "downstream_metrics": multi_metrics,
                "downstream_gates": multi_gates,
                "all_gates_passed": all(maxwell_gates.values())
                and all(multi_gates.values()),
            }

    staircase_material_gate = all(
        summaries["staircase"][case]["gates"][
            "official_material_filter_unassigned_absorption_lt_0p5pct"
        ]
        for case in CASES
    )
    cv0_staircase_maxwell_gate = all(
        all(comparisons["cv0_vs_staircase"][case]["Maxwell_gates"].values())
        for case in CASES
    )
    select_staircase = staircase_material_gate and cv0_staircase_maxwell_gate
    selection = {
        "staircase_material_assignment_lt_0p5pct_both_controls": (
            staircase_material_gate
        ),
        "cv0_vs_staircase_Maxwell_agreement_lt_0p5pct_both_controls": (
            cv0_staircase_maxwell_gate
        ),
        "next_z_refinement_candidate": "staircase" if select_staircase else None,
        "is_final_mesh_selection": False,
        "requires_linked_staircase_z_convergence": True,
    }

    output.mkdir(parents=True)
    raw_output = output / "interface_method_triage_raw.npz"
    np.savez_compressed(
        raw_output,
        **{
            f"{method}_{case}_{name}": value
            for (method, case), values in arrays.items()
            for name, value in values.items()
        },
    )
    payload = {
        "schema": "lumerical-4um-interface-method-triage-v1",
        "status": (
            "SELECTED_STAIRCASE_FOR_LINKED_Z_REFINEMENT_DEVELOPMENT"
            if select_staircase
            else "BLOCKED_INTERFACE_METHOD_TRIAGE_DEVELOPMENT"
        ),
        "contract": contract,
        "individual": summaries,
        "pairwise": comparisons,
        "selection": selection,
        "artifacts": artifacts,
        "raw_output": {
            "path": str(raw_output),
            "size_bytes": raw_output.stat().st_size,
            "sha256": sha256(raw_output),
        },
        "scope": (
            "single 5/50-nm Ea empty/full RTX triage; not linked z convergence, "
            "Eb/simple-L/final-topology, B200, or production certification"
        ),
    }
    json_output = output / "interface_method_triage.json"
    json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if select_staircase else 2


if __name__ == "__main__":
    raise SystemExit(main())
