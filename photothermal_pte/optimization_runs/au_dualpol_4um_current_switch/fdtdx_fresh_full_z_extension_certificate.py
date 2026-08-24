"""Fail-closed z8-to-z16 extension certificate for the exact-binary L500 case."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import traceback
from typing import Any, Mapping

import numpy as np

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    C0_M_PER_S,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    FreshCaseSpec,
    load_case_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_exact_binary_pilot import (
    validate_source_pair,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_full_z_certificate import (
    CFL_DT_RTOL,
    COURANT_FACTOR,
    TOTAL_PERIODS,
    compare_full_z_pair,
    expected_full_z_case,
    source_raw_grid_audit,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_full_z_extension_case import (
    expected_extension_case,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_time_settling_certificate import (
    POLARIZATIONS,
    _all_true,
    _material_case_audit,
    sha256,
)


CERTIFICATE_NAME = "FDTDX_FRESH_FULL_Z16_EXTENSION_CERTIFICATE.json"
STATUS_READY = "VALIDATED_FDTDX_FRESH_L500_Z16_EXTENSION_PENDING_Z32"
STATUS_BLOCKED = "BLOCKED_FDTDX_FRESH_L500_Z16_EXTENSION"
LEVELS = ("z8", "z16")
Z_FACTOR = {"z8": 8, "z16": 16}
SUCCESSIVE_PAIRS = (("z8", "z16"),)
PRIOR_FAILED_PAIR = ("z4", "z8")
EXPECTED_PRIOR_STATUS = "BLOCKED_FDTDX_FRESH_L500_FULL_DOMAIN_Z_CONVERGENCE"
EXPECTED_PRIOR_FAILED_GATE = "both_successive_full_z_comparisons_and_selection_pass"

ALLOWED_CROSS_COMMIT_PATHS = frozenset(
    (
        "photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/CODE_HANDOFF.md",
        "photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/FDTDX_FRESH_CONVERGENCE_DESIGN.md",
        "photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/FDTDX_FRESH_HASHED_RUNNER.md",
        "photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/fdtdx_fresh_full_z_certificate.py",
        "photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/fdtdx_fresh_full_z_extension_case.py",
        "photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/fdtdx_fresh_full_z_extension_certificate.py",
        "photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/run_fdtdx_fresh_full_z_extension_gpu.sh",
        "photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/test_fdtdx_fresh_full_z_certificate.py",
        "photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/test_fdtdx_fresh_full_z_extension.py",
    )
)


def expected_case(level: str) -> FreshCaseSpec:
    if level == "z8":
        return expected_full_z_case("z8")
    if level == "z16":
        return expected_extension_case("z16")
    raise ValueError(f"extension level must be one of {LEVELS}")


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def cross_commit_audit(repository: Path, commits: set[Any]) -> dict[str, Any]:
    invalid = sorted(repr(value) for value in commits if not isinstance(value, str))
    ordered = sorted(value for value in commits if isinstance(value, str))
    comparisons = []
    changed_union: set[str] = set()
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            changed = tuple(
                name
                for name in _git(
                    repository, "diff", "--name-only", left, right
                ).splitlines()
                if name
            )
            changed_union.update(changed)
            comparisons.append(
                {
                    "left": left,
                    "right": right,
                    "changed_paths": list(changed),
                }
            )
    ready = (
        len(ordered) >= 2
        and not invalid
        and changed_union.issubset(ALLOWED_CROSS_COMMIT_PATHS)
    )
    return {
        "recorded_commits": ordered,
        "invalid_commit_values": invalid,
        "comparisons": comparisons,
        "changed_paths_union": sorted(changed_union),
        "allowed_paths": sorted(ALLOWED_CROSS_COMMIT_PATHS),
        "runner_or_physics_paths_unchanged": ready,
        "ready": ready,
    }


def prior_certificate_audit(
    path: Path, expected_sha256: str, campaign_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    supplied = path.expanduser()
    resolved = supplied.resolve()
    normalized = expected_sha256.strip().lower()
    exists = resolved.is_file()
    actual = sha256(resolved) if exists else None
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    nonselection = {
        name: value
        for name, value in payload["gates"].items()
        if name != EXPECTED_PRIOR_FAILED_GATE
    }
    checks = {
        "path_is_absolute": supplied.is_absolute(),
        "path_is_under_campaign_root": resolved.is_relative_to(campaign_root),
        "sha256_is_lowercase_hex": len(normalized) == 64
        and all(character in "0123456789abcdef" for character in normalized),
        "file_exists_and_sha256_matches": exists and actual == normalized,
        "status_is_exact_blocked_full_z": payload.get("status")
        == EXPECTED_PRIOR_STATUS
        and payload.get("ready") is False,
        "only_top_level_selection_gate_failed": payload.get("failed_gates")
        == [EXPECTED_PRIOR_FAILED_GATE]
        and bool(nonselection)
        and _all_true(nonselection),
        "both_original_pairs_were_retained_and_failed": set(
            payload.get("successive_comparisons", {})
        )
        == {"z2_to_z4", "z4_to_z8"}
        and all(
            item.get("pass") is False
            for item in payload["successive_comparisons"].values()
        ),
        "prior_z4_to_z8_failure_retained": payload["successive_comparisons"][
            "z4_to_z8"
        ].get("pass")
        is False,
        "prior_optimizer_and_mesh_claims_forbidden": payload.get(
            "optimizer_start_allowed"
        )
        is False
        and payload.get("is_mesh_certificate") is False
        and payload.get("is_full_domain_z_resolution_certificate") is False,
        "prior_certificate_repository_was_clean": payload.get(
            "provenance", {}
        ).get("certificate_repository_dirty_porcelain")
        == "",
    }
    return payload, {
        "path": str(resolved),
        "expected_sha256": normalized,
        "actual_sha256": actual,
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "ready": all(checks.values()),
    }


def _load_level(
    campaign_root: Path,
    level: str,
    contract_sha256: str,
    source_pair_sha256: str,
) -> dict[str, Any]:
    spec = expected_case(level)
    contract_path = (
        campaign_root / "contracts" / f"l500_full_z_{level}.json"
    ).resolve()
    loaded_spec, contract_payload, contract_audit = load_case_contract(
        contract_path, contract_sha256
    )
    if loaded_spec != spec:
        raise RuntimeError(f"{level} is not the exact expected extension case")

    pair_path = (
        campaign_root
        / f"source_pair_full_z_{level}"
        / "FDTDX_FRESH_SOURCE_ONLY_PAIR.json"
    ).resolve()
    pair_payload, pair_audit = validate_source_pair(
        pair_path, source_pair_sha256, expected_case=spec
    )
    source_grid_audit = source_raw_grid_audit(
        pair_payload, spec, campaign_root
    )

    payloads: dict[str, Any] = {}
    cases: dict[str, Any] = {}
    snapshots: dict[str, Any] = {}
    for polarization in POLARIZATIONS:
        report_path = (
            campaign_root
            / f"l500_full_z_{level}_{polarization}"
            / "FDTDX_FRESH_EXACT_BINARY_PILOT.json"
        ).resolve()
        payload, audit, snapshot = _material_case_audit(
            report_path,
            campaign_root,
            TOTAL_PERIODS,
            polarization,
            spec,
            contract_path,
            contract_sha256,
            pair_path,
            source_pair_sha256,
        )
        payloads[polarization] = payload
        cases[polarization] = audit
        snapshots[polarization] = snapshot
    return {
        "spec": spec,
        "contract_path": contract_path,
        "contract_payload": contract_payload,
        "contract_audit": contract_audit,
        "pair_path": pair_path,
        "pair_payload": pair_payload,
        "pair_audit": pair_audit,
        "source_grid_audit": source_grid_audit,
        "payloads": payloads,
        "cases": cases,
        "snapshots": snapshots,
    }


def _same_xy_and_masks(levels: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
    reference = levels["z8"]["snapshots"]["Ea"]
    snapshots = [
        levels[level]["snapshots"][polarization]
        for level in LEVELS
        for polarization in POLARIZATIONS
    ]
    return {
        "x_y_grid_edges_identical": reference is not None
        and all(
            snapshot is not None
            and all(
                np.array_equal(
                    snapshot["grid_edges"][axis],
                    reference["grid_edges"][axis],
                )
                for axis in (0, 1)
            )
            for snapshot in snapshots
        ),
        "binary_masks_identical": reference is not None
        and all(
            snapshot is not None
            and np.array_equal(snapshot["design_mask"], reference["design_mask"])
            and np.array_equal(snapshot["solver_mask"], reference["solver_mask"])
            for snapshot in snapshots
        ),
        "tangential_probe_weights_identical": reference is not None
        and all(
            snapshot is not None
            and np.array_equal(
                snapshot["probe_weights"][:2],
                reference["probe_weights"][:2],
            )
            for snapshot in snapshots
        ),
    }


def _cfl_diagnostics(
    levels: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, bool]]:
    expected = []
    realized = []
    end_times = []
    for level in LEVELS:
        for polarization in POLARIZATIONS:
            snapshot = levels[level]["snapshots"][polarization]
            payload = levels[level]["payloads"][polarization]
            expected.append(
                COURANT_FACTOR
                / (
                    C0_M_PER_S
                    * math.sqrt(
                        sum(
                            1.0
                            / float(
                                np.min(
                                    np.diff(
                                        snapshot["grid_edges"][axis].astype(
                                            np.float64
                                        )
                                    )
                                )
                            )
                            ** 2
                            for axis in range(3)
                        )
                    )
                )
            )
            dt = float(payload["time_contract"]["time_step_s"])
            realized.append(dt)
            end_times.append(
                dt * int(payload["time_contract"]["time_steps_total"])
            )
    checks = {
        "realized_dt_matches_raw_grid_CFL_formula": all(
            math.isclose(actual, target, rel_tol=CFL_DT_RTOL, abs_tol=0.0)
            for actual, target in zip(realized, expected, strict=True)
        ),
        "realized_end_times_agree_within_one_coarse_step": max(end_times)
        - min(end_times)
        <= max(realized),
    }
    return {
        "realized_dt_s": realized,
        "expected_raw_grid_CFL_dt_s": expected,
        "end_times_s": end_times,
    }, checks


def build_extension_certificate(
    root: Path,
    prior_certificate: Path,
    prior_certificate_sha256: str,
    contract_sha256s: Mapping[str, str],
    source_pair_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    campaign_root = root.expanduser().resolve()
    if not root.expanduser().is_absolute() or not campaign_root.is_dir():
        raise RuntimeError("campaign root must be an existing absolute directory")
    if set(contract_sha256s) != set(LEVELS):
        raise ValueError(f"contract hashes must contain exactly {LEVELS}")
    if set(source_pair_sha256s) != set(LEVELS):
        raise ValueError(f"source-pair hashes must contain exactly {LEVELS}")

    prior_payload, prior_audit = prior_certificate_audit(
        prior_certificate, prior_certificate_sha256, campaign_root
    )
    levels = {
        level: _load_level(
            campaign_root,
            level,
            contract_sha256s[level],
            source_pair_sha256s[level],
        )
        for level in LEVELS
    }
    snapshots = {
        level: levels[level]["snapshots"] for level in LEVELS
    }
    payloads = {
        level: levels[level]["payloads"] for level in LEVELS
    }
    source_pairs = {
        level: levels[level]["pair_payload"] for level in LEVELS
    }
    comparison = compare_full_z_pair(
        "z8",
        "z16",
        snapshots,
        payloads,
        source_pairs,
        z_factors=Z_FACTOR,
        successive_pairs=SUCCESSIVE_PAIRS,
    )

    repository = Path(__file__).resolve().parents[3]
    material_commits = {
        level: {
            payloads[level][polarization]["provenance"]["repository_commit"]
            for polarization in POLARIZATIONS
        }
        for level in LEVELS
    }
    source_commits = {
        level: levels[level]["pair_payload"]["provenance"][
            "certificate_repository_commit"
        ]
        for level in LEVELS
    }
    all_commits = {
        commit
        for values in material_commits.values()
        for commit in values
    } | set(source_commits.values()) | {_git(repository, "rev-parse", "HEAD")}
    commit_audit = cross_commit_audit(repository, all_commits)
    xy_masks = _same_xy_and_masks(levels)
    time_diagnostics, time_checks = _cfl_diagnostics(levels)

    flat_payloads = [
        payloads[level][polarization]
        for level in LEVELS
        for polarization in POLARIZATIONS
    ]
    flat_cases = [
        levels[level]["cases"][polarization]
        for level in LEVELS
        for polarization in POLARIZATIONS
    ]
    base_mesh = levels["z8"]["contract_payload"]["mesh_spec"]
    gates = {
        "prior_blocked_certificate_revalidated": prior_audit["ready"] is True,
        "both_canonical_extension_contracts_revalidated": all(
            levels[level]["contract_audit"]["ready"] is True
            for level in LEVELS
        ),
        "both_source_pairs_revalidated": all(
            levels[level]["pair_audit"]["ready"] is True
            for level in LEVELS
        ),
        "both_source_raw_grids_and_runners_revalidated": all(
            levels[level]["source_grid_audit"]["ready"] is True
            for level in LEVELS
        ),
        "all_material_artifacts_and_recomputed_physics_ready": all(
            case["artifact_ready"] is True for case in flat_cases
        ),
        "all_four_material_cases_internally_ready": all(
            payload["evaluation"]["ready"] is True for payload in flat_payloads
        ),
        "only_full_domain_z_factor_changes": all(
            {
                name: value
                for name, value in levels[level]["contract_payload"][
                    "mesh_spec"
                ].items()
                if name != "z_factor"
            }
            == {
                name: value
                for name, value in base_mesh.items()
                if name != "z_factor"
            }
            and levels[level]["contract_payload"]["mesh_spec"]["z_factor"]
            == Z_FACTOR[level]
            and levels[level]["spec"].time == levels["z8"]["spec"].time
            for level in LEVELS
        ),
        **xy_masks,
        **time_checks,
        "source_and_material_commit_match_within_each_level": all(
            len(material_commits[level]) == 1
            and source_commits[level] in material_commits[level]
            for level in LEVELS
        ),
        "cross_commit_changes_exclude_runners_and_physics": commit_audit[
            "ready"
        ]
        is True,
        "source_runner_hash_identical": len(
            {
                record["runner_sha256"]
                for level in LEVELS
                for record in levels[level]["source_grid_audit"][
                    "records"
                ].values()
            }
        )
        == 1,
        "source_pair_generator_hash_identical": len(
            {
                levels[level]["pair_payload"]["provenance"][
                    "certificate_generator_sha256"
                ]
                for level in LEVELS
            }
        )
        == 1,
        "material_runner_hash_identical": len(
            {payload["provenance"]["runner_sha256"] for payload in flat_payloads}
        )
        == 1,
        "material_contract_hash_identical": len(
            {
                payload["provenance"]["material_contract_sha256"]
                for payload in flat_payloads
            }
        )
        == 1,
        "fdtdx_source_provenance_identical": len(
            {
                json.dumps(
                    payload["provenance"]["fdtdx_source"],
                    sort_keys=True,
                )
                for payload in flat_payloads
            }
        )
        == 1,
        "runtime_lock_identical": len(
            {
                json.dumps(
                    payload["provenance"]["runtime_lock"],
                    sort_keys=True,
                )
                for payload in flat_payloads
            }
        )
        == 1,
        "z8_to_z16_physical_comparison_passes": comparison["pass"] is True,
        "optimizer_remains_forbidden": all(
            payload.get("optimizer_start_allowed") is False
            for payload in flat_payloads
        ),
    }
    ready = all(gates.values())
    return {
        "status": STATUS_READY if ready else STATUS_BLOCKED,
        "ready": ready,
        "scope": (
            "z8-to-z16 full-domain-z extension for exact-binary L500 at "
            "24 periods and Courant 0.25; prior z4-to-z8 failure retained"
        ),
        "campaign_root": str(campaign_root),
        "prior_certificate": {
            "audit": prior_audit,
            "z4_to_z8": prior_payload["successive_comparisons"]["z4_to_z8"],
        },
        "levels": {
            level: {
                "z_factor": Z_FACTOR[level],
                "grid_shape_xyz": levels[level]["contract_payload"][
                    "resolved_mesh"
                ]["grid_shape_xyz"],
                "vertical_segments": levels[level]["contract_payload"][
                    "resolved_mesh"
                ]["vertical_segments"],
                "contract_audit": levels[level]["contract_audit"],
                "source_pair_audit": levels[level]["pair_audit"],
                "source_raw_grid_audit": levels[level]["source_grid_audit"],
                "material_cases": levels[level]["cases"],
            }
            for level in LEVELS
        },
        "z8_to_z16_comparison": comparison,
        "selection": {
            "selected_level": None,
            "confirmation_level": None,
            "prior_z4_to_z8_pass": False,
            "z8_to_z16_pass": comparison["pass"],
            "two_successive_fine_pairs_pass": False,
            "policy": (
                "z16 extension can diagnose the next interval but cannot "
                "select a mesh because z4-to-z8 failed; z16-to-z32 is required"
            ),
        },
        "time_diagnostics": time_diagnostics,
        "cross_commit_audit": commit_audit,
        "gates": gates,
        "failed_gates": [name for name, value in gates.items() if not value],
        "is_full_domain_z_resolution_certificate": False,
        "is_mesh_certificate": False,
        "optimizer_start_allowed": False,
        "next_allowed_step": (
            "run z32 under the identical exact-L500, 24-period, Courant-0.25 "
            "contract; require z8-to-z16 and z16-to-z32 to pass before selection"
        ),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_extension_certificate(
    root: Path,
    prior_certificate: Path,
    prior_certificate_sha256: str,
    contract_sha256s: Mapping[str, str],
    source_pair_sha256s: Mapping[str, str],
    output_directory: Path,
) -> dict[str, Any]:
    output = output_directory.expanduser().resolve()
    if not output_directory.expanduser().is_absolute() or not output.is_dir():
        raise RuntimeError("output directory must be an existing absolute directory")
    if any(output.iterdir()):
        raise RuntimeError("output directory must be empty before certification")
    result = build_extension_certificate(
        root,
        prior_certificate,
        prior_certificate_sha256,
        contract_sha256s,
        source_pair_sha256s,
    )
    repository = Path(__file__).resolve().parents[3]
    dirty = _git(repository, "status", "--porcelain", "--untracked-files=all")
    result["provenance"] = {
        "certificate_repository_commit": _git(
            repository, "rev-parse", "HEAD"
        ),
        "certificate_repository_dirty_porcelain": dirty,
        "certificate_generator_path": str(Path(__file__).resolve()),
        "certificate_generator_sha256": sha256(Path(__file__).resolve()),
    }
    result["gates"]["certificate_repository_clean"] = dirty == ""
    result["ready"] = all(result["gates"].values())
    result["status"] = STATUS_READY if result["ready"] else STATUS_BLOCKED
    result["failed_gates"] = [
        name for name, value in result["gates"].items() if not value
    ]
    _atomic_json(output / CERTIFICATE_NAME, result)
    return result


def _lower_sha(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior-certificate", type=Path, required=True)
    parser.add_argument("--prior-certificate-sha256", required=True)
    parser.add_argument("--z8-contract-sha256", required=True)
    parser.add_argument("--z16-contract-sha256", required=True)
    parser.add_argument("--z8-source-pair-sha256", required=True)
    parser.add_argument("--z16-source-pair-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = write_extension_certificate(
            args.root,
            args.prior_certificate,
            _lower_sha(
                args.prior_certificate_sha256, "prior certificate SHA256"
            ),
            {
                "z8": _lower_sha(
                    args.z8_contract_sha256, "z8 contract SHA256"
                ),
                "z16": _lower_sha(
                    args.z16_contract_sha256, "z16 contract SHA256"
                ),
            },
            {
                "z8": _lower_sha(
                    args.z8_source_pair_sha256, "z8 source-pair SHA256"
                ),
                "z16": _lower_sha(
                    args.z16_source_pair_sha256,
                    "z16 source-pair SHA256",
                ),
            },
            args.output_dir,
        )
    except Exception as error:
        failure = {
            "status": "BLOCKED_FDTDX_FRESH_L500_Z16_EXTENSION_EXCEPTION",
            "ready": False,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "is_full_domain_z_resolution_certificate": False,
            "is_mesh_certificate": False,
            "optimizer_start_allowed": False,
        }
        output = args.output_dir.expanduser().resolve()
        if output.is_dir() and not any(output.iterdir()):
            _atomic_json(output / CERTIFICATE_NAME, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2
    summary = {
        "status": result["status"],
        "ready": result["ready"],
        "failed_gates": result["failed_gates"],
        "selection": result["selection"],
        "z8_to_z16": {
            "pass": result["z8_to_z16_comparison"]["pass"],
            "metrics": result["z8_to_z16_comparison"].get("metrics"),
        },
        "report": str(
            args.output_dir.expanduser().resolve() / CERTIFICATE_NAME
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
