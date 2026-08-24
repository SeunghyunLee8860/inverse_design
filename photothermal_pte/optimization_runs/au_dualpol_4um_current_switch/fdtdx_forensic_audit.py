"""Read-only forensic audit of the committed FDTDX campaign state.

This module deliberately does not import JAX, FDTDX, the Maxwell model, or the
multiphysics solvers.  It audits only committed source and result records, so a
new session can establish the campaign's blockers before attempting a solve.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
FULL_Z_SUMMARY = (
    HERE
    / "results_4um_shared_linear_full_z_convergence"
    / "FULL_Z_CONVERGENCE_SUMMARY.json"
)
ROBUST_FINAL = (
    HERE
    / "results_4um_dualpol_au_robust_projection_ld_mma"
    / "FINAL_RESULT.json"
)
DEVICE_CONTRACT = HERE / "physical_device_contract.json"
MATERIAL_CONTRACT = HERE / "results_materials_4um" / "4um_material_contract.json"
OPTICAL_MODEL = HERE / "fdtdx_4um_model.py"
PRODUCTION_MESH_CERTIFICATE = (
    HERE
    / "results_4um_shared_linear_mesh_convergence"
    / "MESH_CONVERGENCE_SUMMARY.json"
)
PRODUCTION_GRADIENT_CERTIFICATE = (
    HERE
    / "results_4um_shared_linear_combined_adfd"
    / "COMBINED_ADFD_SUMMARY.json"
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _boundary_config_keywords(source: str) -> list[str]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute) and function.attr == "BoundaryConfig":
            return sorted(
                keyword.arg for keyword in node.keywords if keyword.arg is not None
            )
    raise RuntimeError("fdtdx.BoundaryConfig call was not found")


def _nominal_currents(case_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, float]] = {}
    for case in case_results:
        if case.get("density_case") != "eta_0.50_nominal":
            continue
        factor = int(case["factor"])
        polarization = str(case["polarization"])
        grouped.setdefault(factor, {})[polarization] = float(case["current_A"])

    rows: list[dict[str, Any]] = []
    for factor, currents in sorted(grouped.items()):
        if set(currents) != {"Ea", "Eb"}:
            raise RuntimeError(f"incomplete nominal current pair for factor {factor}")
        ea = currents["Ea"]
        eb = currents["Eb"]
        rows.append(
            {
                "factor": factor,
                "Ea_current_A": ea,
                "Eb_current_A": eb,
                "opposite_sign": ea * eb < 0.0,
                "requested_orientation_Ea_positive_Eb_negative": ea > 0.0
                and eb < 0.0,
            }
        )
    return rows


def _binary_candidate_audit(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = payload.get("binary_candidates")
    if not isinstance(candidates, list):
        raise RuntimeError("robust final result has no binary candidate list")
    rows = []
    for candidate in candidates:
        ea = float(candidate["I_a_A"])
        eb = float(candidate["I_b_A"])
        rows.append(
            {
                "name": str(candidate["name"]),
                "Ea_current_A": ea,
                "Eb_current_A": eb,
                "exact_500nm_bad_cells": int(candidate["exact_bad_cells"]),
                "opposite_sign": ea * eb < 0.0,
                "requested_orientation_Ea_positive_Eb_negative": ea > 0.0
                and eb < 0.0,
            }
        )
    return {
        "campaign_status": payload.get("status"),
        "campaign_opposite_sign_gate": bool(payload.get("opposite_sign_gate")),
        "candidate_count": len(rows),
        "exact_500nm_pass_count": sum(
            row["exact_500nm_bad_cells"] == 0 for row in rows
        ),
        "opposite_sign_count": sum(row["opposite_sign"] for row in rows),
        "requested_orientation_count": sum(
            row["requested_orientation_Ea_positive_Eb_negative"] for row in rows
        ),
        "candidates": rows,
    }


def _mesh_audit(payload: dict[str, Any]) -> dict[str, Any]:
    runsetup = payload.get("runsetup")
    if not isinstance(runsetup, dict):
        raise RuntimeError("full-z result has no embedded runsetup")
    levels = runsetup.get("levels")
    comparisons = payload.get("comparison_results")
    if not isinstance(levels, list) or not isinstance(comparisons, list):
        raise RuntimeError("full-z result has incomplete mesh records")

    dx = [float(level["central_dx_m"]) for level in levels]
    dy = [float(level["central_dy_m"]) for level in levels]
    final_pair = [
        row
        for row in comparisons
        if int(row["coarse_factor"]) == 2 and int(row["fine_factor"]) == 4
    ]
    metrics = (
        "P_Q_relative_change",
        "remapped_Q_volume_L2_NRMSE",
        "Ta_temperature_NRMSE",
        "Tmax_relative_change",
        "current_relative_change",
    )
    maxima = {
        metric: max(float(row[metric]) for row in final_pair) for metric in metrics
    }
    return {
        "campaign_status": payload.get("status"),
        "factors": [int(level["factor"]) for level in levels],
        "central_dx_m": dx,
        "central_dy_m": dy,
        "xy_refined": len(set(dx)) > 1 or len(set(dy)) > 1,
        "comparison_count": len(comparisons),
        "passing_comparison_count": sum(
            bool(row.get("comparison_pass")) for row in comparisons
        ),
        "final_pair_comparison_count": len(final_pair),
        "passing_final_pair_count": sum(
            bool(row.get("comparison_pass")) for row in final_pair
        ),
        "factor_2_to_4_worst_relative_changes": maxima,
        "selected_optical_z_contract": payload.get("selected_optical_z_contract"),
        "production_mesh_certificate_exists": PRODUCTION_MESH_CERTIFICATE.is_file(),
        "production_combined_adfd_certificate_exists": (
            PRODUCTION_GRADIENT_CERTIFICATE.is_file()
        ),
    }


def audit() -> dict[str, Any]:
    """Return a machine-readable, fail-closed snapshot of FDTDX blockers."""

    full_z = _load_object(FULL_Z_SUMMARY)
    robust = _load_object(ROBUST_FINAL)
    device = _load_object(DEVICE_CONTRACT)
    materials = _load_object(MATERIAL_CONTRACT)
    source = OPTICAL_MODEL.read_text(encoding="utf-8")

    nominal = _nominal_currents(full_z["case_results"])
    binary = _binary_candidate_audit(robust)
    mesh = _mesh_audit(full_z)
    confirmations = device.get("confirmations")
    if not isinstance(confirmations, dict):
        raise RuntimeError("device contract has no confirmations object")
    unresolved = sorted(name for name, value in confirmations.items() if value is not True)

    ta = materials["materials"]["TaIrTe4"]
    raw_paths = [Path(str(case["raw_path"])) for case in full_z["case_results"]]
    boundary_keywords = _boundary_config_keywords(source)
    explicit_pml_prefixes = (
        "alpha_",
        "kappa_",
        "sigma_",
    )
    pml_profile_keywords = [
        name
        for name in boundary_keywords
        if name.startswith(explicit_pml_prefixes)
    ]

    blockers = [
        "DEVICE_CONTRACT_UNCONFIRMED",
        "NO_OPPOSITE_SIGN_NOMINAL_ENDPOINT",
        "EXACT_BINARY_CANDIDATES_FAIL_SIGN",
        "OPTICAL_Z_CONVERGENCE_FAILED",
        "OPTICAL_XY_NOT_REFINED",
        "THERMAL_MESH_NOT_CERTIFIED",
        "ELECTRICAL_MESH_NOT_CERTIFIED",
        "COMBINED_ADFD_NOT_CERTIFIED_ON_SELECTED_MESH",
        "GRAY_AU_RELAXATION_NOT_A_PHYSICAL_CONSTITUTIVE_LAW",
        "BINARY_METAL_INTERFACE_NOT_SUBPIXEL_SMOOTHED",
        "PML_PROFILE_NOT_EXPLICIT_OR_CONVERGED",
        "FDTDX_SOURCE_TREE_NOT_FULLY_PINNED_IN_REPOSITORY",
        "RAW_ARTIFACTS_ARE_EXTERNAL_ABSOLUTE_PATHS",
        "TAIRTE4_C_AXIS_COPIED_FROM_B_AXIS",
    ]

    return {
        "status": "BLOCKED_FDTDX_FORENSIC_AUDIT",
        "scope": "committed historical FDTDX records and source; no solver execution",
        "blockers": blockers,
        "endpoint": {
            "nominal_by_z_factor": nominal,
            "all_nominal_pairs_opposite_sign": all(
                row["opposite_sign"] for row in nominal
            ),
            "all_nominal_pairs_requested_orientation": all(
                row["requested_orientation_Ea_positive_Eb_negative"]
                for row in nominal
            ),
            "exact_binary": binary,
        },
        "mesh": mesh,
        "device": {
            "status": device.get("status"),
            "confirmation_count": len(confirmations),
            "confirmed_count": len(confirmations) - len(unresolved),
            "unresolved_confirmations": unresolved,
        },
        "materials": {
            "TaIrTe4_c_equals_b": ta["c"]["epsilon"] == ta["b"]["epsilon"],
            "TaIrTe4_axis_mapping": materials.get("axis_mapping"),
            "Au_epsilon_at_4um": materials["materials"]["Au"]["epsilon"],
            "single_frequency_readback_only": materials.get("scope")
            == "material database/readback only; no field solve or optimization",
        },
        "implementation": {
            "boundary_config_keywords": boundary_keywords,
            "explicit_pml_profile_keywords": pml_profile_keywords,
            "pml_profile_explicit": bool(pml_profile_keywords),
            "uses_uniform_material_object_for_tairte4": (
                'name="fixed_tairte4"' in source
                and "fdtdx.UniformMaterialObject" in source
            ),
            "uses_uniform_material_object_for_au": (
                'name="au_design"' in source
                and "fdtdx.UniformMaterialObject" in source
            ),
            "requests_subpixel_smoothing": "subpixel_smoothing" in source,
            "raw_artifact_count": len(raw_paths),
            "raw_artifacts_absolute": all(path.is_absolute() for path in raw_paths),
            "raw_artifacts_present_in_current_environment": sum(
                path.is_file() for path in raw_paths
            ),
        },
        "decision": {
            "resume_historical_optimizer": False,
            "promote_historical_geometry": False,
            "next_action": (
                "freeze the measured device contract, then validate exact-binary "
                "Maxwell reference structures and a multidimensional convergence plan"
            ),
        },
    }


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
