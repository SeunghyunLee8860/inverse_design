#!/usr/bin/env python3
"""Canonical opt-in two-pole material-law contract for the fresh z ladder.

The contract is solver-free and candidate-only.  It binds one canonical
numerical case, the source material table, the exact pinned FDTDX recurrence,
and realized float32 pole coefficients for Au and every TaIrTe4 principal axis.
It does not modify the historical single-pole builder or authorize a solve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    MeshSpec,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_ade_precision_diagnostic import (
    FIT_RELATIVE_TOLERANCE,
    MATERIAL_CONTRACT,
    WAVELENGTH_M,
    analyze_material_axis,
    file_sha256,
    load_material_epsilon,
    realized_float32_cfl,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    FreshCaseSpec,
    load_case_contract,
)


VERSION = "fdtdx-fresh-stable-two-pole-material-v1"
ALGORITHM_IMPLEMENTATION = (
    Path(__file__).resolve().parent / "fdtdx_fresh_ade_precision_diagnostic.py"
)
SUPPORTED_Z_FACTORS = (8, 16, 32)
SUPPORTED_TOTAL_PERIODS = (24, 32)
AXIS_TO_SOLVER_COMPONENT = {"b": "x", "a": "y", "c": "z"}


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pinned_files(fdtdx_source: Path) -> dict[str, str]:
    root = fdtdx_source.expanduser().resolve()
    update = root / "src/fdtdx/fdtd/update.py"
    dispersion = root / "src/fdtdx/dispersion.py"
    if not update.is_file() or not dispersion.is_file():
        raise RuntimeError("FDTDX source is missing update.py or dispersion.py")
    return {
        "update_sha256": file_sha256(update),
        "dispersion_sha256": file_sha256(dispersion),
    }


def _require_supported_case(spec: FreshCaseSpec) -> None:
    expected_mesh = MeshSpec(z_factor=spec.mesh.z_factor)
    checks = {
        "z_factor_supported": spec.mesh.z_factor in SUPPORTED_Z_FACTORS,
        "only_full_domain_z_differs_from_default": spec.mesh == expected_mesh,
        "total_periods_supported": (
            spec.time.total_periods in SUPPORTED_TOTAL_PERIODS
        ),
        "window_periods_is_4": spec.time.window_periods == 4,
        "courant_factor_is_0p25": spec.time.courant_factor == 0.25,
    }
    if not all(checks.values()):
        raise RuntimeError(f"unsupported two-pole candidate case: {checks}")


def material_law_contract(
    spec: FreshCaseSpec,
    case_payload: Mapping[str, Any],
    case_file_sha256: str,
    fdtdx_source: Path,
) -> dict[str, Any]:
    """Build a deterministic candidate law for one exact numerical case."""

    _require_supported_case(spec)
    epsilon = load_material_epsilon()
    analyses = {
        name: analyze_material_axis(spec.mesh.z_factor, name, value)
        for name, value in epsilon.items()
    }
    checks = {
        "all_two_pole_fits_found": all(
            item["stable_two_pole_candidate"].get("found", False)
            for item in analyses.values()
        ),
        "all_two_pole_fit_errors_below_gate": all(
            item["stable_two_pole_candidate"].get("fit_gate_passed", False)
            for item in analyses.values()
        ),
        "all_strengths_positive": all(
            pole["positive_strength"]
            for item in analyses.values()
            for pole in item["stable_two_pole_candidate"]["poles"]
        ),
        "all_recurrence_roots_not_above_one": all(
            pole["recurrence_roots_not_above_one"]
            for item in analyses.values()
            for pole in item["stable_two_pole_candidate"]["poles"]
        ),
        "all_c3_reconstruct_exactly_in_float32": all(
            pole["c3"] == pole["reconstructed_float32_c3"]
            for item in analyses.values()
            for pole in item["stable_two_pole_candidate"]["poles"]
        ),
        "tairte4_b_c_candidates_identical": (
            analyses["b"]["stable_two_pole_candidate"]
            == analyses["c"]["stable_two_pole_candidate"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"two-pole material candidate failed: {checks}")

    payload: dict[str, Any] = {
        "version": VERSION,
        "algorithm": {
            "name": "stable positive two-pole realized-float32 carrier fit",
            "gamma_ratio_range": [0.01, 10.0],
            "gamma_samples": 400_001,
            "target_wavelength_m": WAVELENGTH_M,
            "target_relative_error_limit": FIT_RELATIVE_TOLERANCE,
            "selection": (
                "two distinct stable float32 recurrence phases bracketing the "
                "target; solve two positive float32 c3 strengths"
            ),
            "no_gray_material_interpolation": True,
        },
        "case_binding": {
            "case_file_sha256": case_file_sha256,
            "case_contract_sha256": case_payload["case_contract_sha256"],
            "z_factor": spec.mesh.z_factor,
            "time_spec": dict(case_payload["time_spec"]),
            "realized_float32_cfl": realized_float32_cfl(spec.mesh.z_factor),
        },
        "material_binding": {
            "source_contract_sha256": file_sha256(MATERIAL_CONTRACT),
            "target_epsilon": {
                name: [value.real, value.imag] for name, value in epsilon.items()
            },
            "tairte4_crystal_to_solver_axis": AXIS_TO_SOLVER_COMPONENT,
        },
        "implementation_binding": {
            "algorithm_implementation_sha256": file_sha256(
                ALGORITHM_IMPLEMENTATION
            ),
            "pinned_fdtdx": _pinned_files(fdtdx_source),
        },
        "material_axes": {
            name: {
                "pole_kind": item["pole_kind"],
                "omega_0_rad_s": item["omega_0_rad_s"],
                "current_single_pole_refit": item["current_single_pole_refit"],
                "candidate": item["stable_two_pole_candidate"],
            }
            for name, item in analyses.items()
        },
        "checks": checks,
        "promotion": {
            "candidate_only": True,
            "is_material_certificate": False,
            "is_mesh_certificate": False,
            "optimizer_start_allowed": False,
            "requires_exact_solver_readback": True,
            "requires_same_law_z8_z16_z32_runs": True,
        },
    }
    payload["material_law_contract_sha256"] = canonical_sha256(payload)
    return payload


def material_law_from_contract(
    payload: Mapping[str, Any],
    spec: FreshCaseSpec,
    case_payload: Mapping[str, Any],
    case_file_sha256: str,
    fdtdx_source: Path,
) -> dict[str, Any]:
    """Reject any extra, stale, or edited field by exact reconstruction."""

    if not isinstance(payload, Mapping):
        raise TypeError("material-law contract must be one JSON object")
    expected = material_law_contract(
        spec, case_payload, case_file_sha256, fdtdx_source
    )
    if dict(payload) != expected:
        raise ValueError("material-law contract is not the exact canonical contract")
    return expected


def load_material_law_contract(
    path: Path,
    expected_sha256: str,
    spec: FreshCaseSpec,
    case_payload: Mapping[str, Any],
    case_file_sha256: str,
    fdtdx_source: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    supplied = path.expanduser()
    resolved = supplied.resolve()
    normalized_sha = expected_sha256.strip().lower()
    sha_ok = len(normalized_sha) == 64 and all(
        character in "0123456789abcdef" for character in normalized_sha
    )
    exists = resolved.is_file()
    actual_sha = file_sha256(resolved) if exists else None
    checks = {
        "path_is_absolute": supplied.is_absolute(),
        "file_exists": exists,
        "expected_sha256_is_lowercase_hex": sha_ok,
        "file_sha256_matches": exists and sha_ok and actual_sha == normalized_sha,
    }
    if not all(checks.values()):
        raise RuntimeError(f"material-law file audit failed: {checks}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    canonical = material_law_from_contract(
        payload, spec, case_payload, case_file_sha256, fdtdx_source
    )
    return canonical, {
        "path": str(resolved),
        "expected_sha256": normalized_sha,
        "actual_sha256": actual_sha,
        "material_law_contract_sha256": canonical[
            "material_law_contract_sha256"
        ],
        "checks": checks,
        "ready": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-contract", type=Path, required=True)
    parser.add_argument("--case-contract-sha256", required=True)
    parser.add_argument("--fdtdx-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser()
    if not output.is_absolute():
        parser.error("--output must be absolute")
    output = output.resolve()
    if not output.parent.is_dir() or output.exists():
        parser.error("output parent must exist and output must not exist")
    spec, case_payload, case_audit = load_case_contract(
        args.case_contract, args.case_contract_sha256
    )
    payload = material_law_contract(
        spec,
        case_payload,
        case_audit["actual_sha256"],
        args.fdtdx_source,
    )
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "WROTE_CANDIDATE_TWO_POLE_MATERIAL_LAW",
                "output": str(output),
                "file_sha256": file_sha256(output),
                "material_law_contract_sha256": payload[
                    "material_law_contract_sha256"
                ],
                "candidate_only": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
