#!/usr/bin/env python3
"""Solver-free/CPU audit of the assumed 4-um physical-device model.

The rectangular electrical calculation below validates only the implemented
Shockley--Ramo sign and discretization.  It cannot validate the target flake,
electrodes, crystal angle, contacts, illumination, or patterned-Au role.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.contract import (
    CONTRACT,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.electrical_prototype_4um import (
    build_exact_binary_system,
    current_integrand_A_m2,
    evaluate_current_A,
    solve_weighting_cpu,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.multiphysics_4um import (
    SEEBECK_TA_XY_V_K,
    SIGMA_TA_XY_S_M,
    TA_THICKNESS_M,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.production_readiness import (
    REQUIRED_DEVICE_CONFIRMATIONS,
    readiness_audit,
)


VERSION = "fdtdx-physical-device-blocker-audit-v1"
STATUS = "VALIDATED_BLOCKED_FDTDX_PHYSICAL_DEVICE_AUDIT"
INVALID_STATUS = "INVALID_FDTDX_PHYSICAL_DEVICE_AUDIT"
REPORT_NAME = "FDTDX_PHYSICAL_DEVICE_BLOCKER_AUDIT.json"
DEVICE_STATUS = "BLOCKED_DEVICE_GEOMETRY_CONFIRMATION_REQUIRED"
LOCAL_MAIN_PAPER = (
    "Adv Funct Materials - 2026 - Blevins - Large Transverse "
    "Thermoelectric Effect in Weyl Semimetal TaIrTe4 Engineered for.pdf"
)
LOCAL_SUPPLEMENT = "adfm75986-sup-0001-suppmat-2.pdf"
HISTORICAL_PAPER_CONTRACT = (
    "photothermal_pte/reports/paper_ir_device_a_measured_reproduction/"
    "device_a_measured_reproduction_contract.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def device_contract_audit(path: Path) -> dict[str, Any]:
    payload = _read_object(path)
    confirmations = payload.get("confirmations")
    if not isinstance(confirmations, dict):
        confirmations = {}
    assumptions = payload.get("current_code_assumptions")
    if not isinstance(assumptions, dict):
        assumptions = {}
    required_inputs = payload.get("required_user_inputs")
    if not isinstance(required_inputs, list):
        required_inputs = []
    contract = CONTRACT.audit()
    checks = {
        "status_is_explicitly_blocked": payload.get("status") == DEVICE_STATUS,
        "confirmation_keys_exact": set(confirmations)
        == set(REQUIRED_DEVICE_CONFIRMATIONS),
        "all_confirmations_are_false": bool(confirmations)
        and all(value is False for value in confirmations.values()),
        "required_user_inputs_nonempty": bool(required_inputs)
        and all(isinstance(value, str) and value.strip() for value in required_inputs),
        "contract_flake_is_16um_square": CONTRACT.flake_span_x_m == 16.0e-6
        and CONTRACT.flake_span_y_m == 16.0e-6,
        "contract_flake_thickness_is_100nm": CONTRACT.flake_thickness_m
        == 100.0e-9,
        "contract_design_is_centered_8um_by_50nm": CONTRACT.design_span_x_m
        == 8.0e-6
        and CONTRACT.design_span_y_m == 8.0e-6
        and CONTRACT.design_thickness_m == 50.0e-9,
        "contract_axes_are_fixed_xb_ya": CONTRACT.axis_x == "b"
        and CONTRACT.axis_y == "a",
        "contract_terminals_are_full_x_edges": CONTRACT.low_terminal == "x_min"
        and CONTRACT.high_terminal == "x_max",
        "optical_electrodes_are_absent": contract["optical_electrodes_included"]
        is False,
        "assumption_ledger_nonempty": bool(assumptions),
    }
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "status": payload.get("status"),
        "confirmations": confirmations,
        "confirmed_count": sum(value is True for value in confirmations.values()),
        "unconfirmed": [
            name for name in REQUIRED_DEVICE_CONFIRMATIONS if confirmations.get(name) is not True
        ],
        "required_user_inputs": required_inputs,
        "current_code_assumptions": assumptions,
        "checks": checks,
        "ready": all(checks.values()),
    }


def rectangular_sign_audit(temperature_gradient_K_m: float = 1.0e5) -> dict[str, Any]:
    """Validate the sign algebra on the implemented ideal rectangle."""

    start = time.perf_counter()
    empty = np.zeros(CONTRACT.design_shape, dtype=np.uint8)
    system = build_exact_binary_system(
        empty,
        patterned_au_electrically_active=False,
    )
    psi, solver = solve_weighting_cpu(system)
    n = system.ta_node_ids.shape[0]
    psi_ta = psi[system.ta_node_ids]
    expected_psi = np.broadcast_to(
        np.arange(n, dtype=np.float64)[:, None] / (n - 1),
        (n, n),
    )
    coordinate = (np.arange(n, dtype=np.float64) + 0.5) * system.step_m
    temperature_x = (
        300.0
        + temperature_gradient_K_m * coordinate[:, None]
        + np.zeros((n, n), dtype=np.float64)
    )
    temperature_y = (
        300.0
        + temperature_gradient_K_m * coordinate[None, :]
        + np.zeros((n, n), dtype=np.float64)
    )
    current_x = evaluate_current_A(system, psi, temperature_x)
    current_y = evaluate_current_A(system, psi, temperature_y)
    reversed_current_x = evaluate_current_A(system, 1.0 - psi, temperature_x)
    expected_current_x = -(
        SIGMA_TA_XY_S_M[0]
        * TA_THICKNESS_M
        * SEEBECK_TA_XY_V_K[0]
        * temperature_gradient_K_m
        * n
        * system.step_m
    )
    mapped_current_x = float(
        np.sum(current_integrand_A_m2(system, psi, temperature_x))
        * system.step_m**2
    )
    current_scale = max(abs(current_x), np.finfo(float).tiny)
    checks = {
        "ta_grid_is_160x160": system.ta_node_ids.shape == (160, 160),
        "only_full_x_edges_are_fixed": np.array_equal(
            system.fixed[:n], system.ta_node_ids[0, :]
        )
        and np.array_equal(system.fixed[n:], system.ta_node_ids[-1, :]),
        "weighting_potential_matches_rectangular_ramp": float(
            np.max(np.abs(psi_ta - expected_psi))
        )
        < 2.0e-10,
        "x_gradient_matches_discrete_signed_formula": abs(
            current_x - expected_current_x
        )
        / abs(expected_current_x)
        < 1.0e-11,
        "pure_y_gradient_collects_zero_for_full_x_edges": abs(current_y)
        / current_scale
        < 1.0e-12,
        "terminal_swap_flips_current": abs(reversed_current_x + current_x)
        / current_scale
        < 1.0e-12,
        "current_map_integrates_to_objective": abs(mapped_current_x - current_x)
        / current_scale
        < 1.0e-12,
        "solver_residual_lt_2e_minus_10": solver["explicit_free_residual"]
        < 2.0e-10,
        "terminal_balance_lt_2e_minus_10": solver["terminal_balance_relative"]
        < 2.0e-10,
    }
    return {
        "scope": "implemented rectangular Ta-only thin-sheet diagnostic",
        "wall_seconds": time.perf_counter() - start,
        "temperature_gradient_K_m": temperature_gradient_K_m,
        "weighting_potential_max_abs_error": float(
            np.max(np.abs(psi_ta - expected_psi))
        ),
        "current_x_A": current_x,
        "expected_current_x_A": expected_current_x,
        "current_y_A": current_y,
        "terminal_swapped_current_x_A": reversed_current_x,
        "mapped_current_x_A": mapped_current_x,
        "solver": solver,
        "checks": checks,
        "ready": all(checks.values()),
    }


def paper_evidence_audit(papers_root: Path, repository: Path) -> dict[str, Any]:
    root = papers_root.expanduser().resolve()
    main = root / LOCAL_MAIN_PAPER
    supplement = root / LOCAL_SUPPLEMENT
    historical_path = repository / HISTORICAL_PAPER_CONTRACT
    historical = _read_object(historical_path) if historical_path.is_file() else {}
    historical_sources = historical.get("sources", {})
    if not isinstance(historical_sources, dict):
        historical_sources = {}
    embedded_main = historical_sources.get("main_paper", {})
    embedded_supplement = historical_sources.get("supporting_information", {})
    if not isinstance(embedded_main, dict):
        embedded_main = {}
    if not isinstance(embedded_supplement, dict):
        embedded_supplement = {}
    local_main_sha = sha256(main) if main.is_file() else None
    return {
        "papers_root": str(root),
        "local_main": {
            "path": str(main),
            "exists": main.is_file(),
            "sha256": local_main_sha,
        },
        "local_supplement": {
            "path": str(supplement),
            "exists": supplement.is_file(),
            "sha256": sha256(supplement) if supplement.is_file() else None,
        },
        "historical_device_A_contract": {
            "path": str(historical_path),
            "exists": historical_path.is_file(),
            "sha256": sha256(historical_path)
            if historical_path.is_file()
            else None,
            "embedded_main": embedded_main,
            "embedded_supplement": embedded_supplement,
            "embedded_paths_are_currently_available": bool(
                Path(str(embedded_main.get("path", ""))).is_file()
                and Path(str(embedded_supplement.get("path", ""))).is_file()
            ),
            "local_main_matches_embedded_bytes": bool(
                local_main_sha is not None
                and local_main_sha == embedded_main.get("sha256")
            ),
        },
        "paper_equation_basis_complete_in_current_papers_root": bool(
            main.is_file() and supplement.is_file()
        ),
        "historical_device_A_is_not_target_device_authority": True,
    }


def build_audit(device_path: Path, papers_root: Path) -> dict[str, Any]:
    here = Path(__file__).resolve().parent
    repository = here.parents[2]
    dirty = _git(repository, "status", "--porcelain", "--untracked-files=all")
    device = device_contract_audit(device_path)
    electrical = rectangular_sign_audit()
    papers = paper_evidence_audit(papers_root, repository)
    production = readiness_audit(device_path=device_path)
    integrity_checks = {
        "repository_clean_while_auditing": dirty == "",
        "device_block_contract_is_well_formed": device["ready"],
        "rectangular_sign_algebra_is_internally_valid": electrical["ready"],
        "production_readiness_remains_false": production["ready"] is False,
        "historical_device_A_contract_is_explicitly_non_authoritative": papers[
            "historical_device_A_is_not_target_device_authority"
        ]
        is True,
    }
    blocking_conditions = {
        "all_target_device_confirmations_missing": device["confirmed_count"] == 0,
        "paper_equation_basis_incomplete_in_current_papers_root": papers[
            "paper_equation_basis_complete_in_current_papers_root"
        ]
        is False,
        "electrode_polygons_unsupported": True,
        "arbitrary_crystal_rotation_and_offdiagonal_tensors_unsupported": True,
        "three_dimensional_weighting_field_unsupported": True,
        "patterned_au_thermoelectric_source_omitted_Sau_assumed_zero": True,
        "optical_electrodes_omitted": True,
        "actual_geometry_electrical_mesh_unconverged": True,
        "electrical_contact_and_au_role_unconfirmed": True,
    }
    audit_valid = all(integrity_checks.values()) and all(blocking_conditions.values())
    generator = Path(__file__).resolve()
    return {
        "version": VERSION,
        "status": STATUS if audit_valid else INVALID_STATUS,
        "audit_valid": audit_valid,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "target-device blocker and rectangular electrical-sign audit; "
            "no Maxwell, Lumerical, GPU, thermal solve, or optimization"
        ),
        "integrity_checks": integrity_checks,
        "failed_integrity_checks": [
            name for name, passed in integrity_checks.items() if not passed
        ],
        "blocking_conditions": blocking_conditions,
        "device_contract": device,
        "rectangular_electrical_sign": electrical,
        "paper_evidence": papers,
        "production_readiness": {
            "ready": production["ready"],
            "failed_checks": production["failed_checks"],
            "errors": production["errors"],
        },
        "decision": {
            "rectangular_prototype_math_internally_valid": electrical["ready"],
            "rectangular_prototype_is_target_device": False,
            "current_sign_is_target_device_prediction": False,
            "physical_device_contract_may_be_promoted": False,
            "thermal_or_electrical_production_mesh_may_be_selected": False,
            "FDTDX_optimizer_start_allowed": False,
        },
        "next_required_user_inputs": device["required_user_inputs"],
        "provenance": {
            "repository_commit": _git(repository, "rev-parse", "HEAD"),
            "repository_dirty_porcelain": dirty,
            "generator_path": str(generator),
            "generator_sha256": sha256(generator),
            "physical_device_contract_sha256": device["sha256"],
            "gpu_used": False,
            "lumerical_used": False,
            "maxwell_solve_run": False,
            "thermal_solve_run": False,
        },
        "optimizer_start_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument(
        "--device-contract",
        type=Path,
        default=here / "physical_device_contract.json",
    )
    parser.add_argument(
        "--papers-root",
        type=Path,
        default=Path("/home/seunghyun200/papers"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser()
    if not output.is_absolute() or not output.parent.is_dir() or output.exists():
        parser.error("--output must be a new absolute file under an existing directory")
    payload = build_audit(args.device_contract, args.papers_root)
    _atomic_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "status": payload["status"],
                "audit_valid": payload["audit_valid"],
                "failed_integrity_checks": payload["failed_integrity_checks"],
                "blocking_conditions": payload["blocking_conditions"],
                "optimizer_start_allowed": False,
            }
        )
    )
    return 0 if payload["audit_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
