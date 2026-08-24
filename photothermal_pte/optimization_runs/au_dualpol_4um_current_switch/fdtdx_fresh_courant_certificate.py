"""Fail-closed certificate for the 24-period FDTDX Courant ladder."""

from __future__ import annotations

import argparse
import json
import math
import numpy as np
import os
from pathlib import Path
import subprocess
import traceback
from typing import Any, Mapping

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence import (
    OPTICAL_PAIR_GATES,
    MeshSpec,
    evaluate_pair,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract import (
    FreshCaseSpec,
    TimeSpec,
    case_contract,
    load_case_contract,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_exact_binary_pilot import (
    combined_weighted_nrmse,
    relative_difference,
    validate_source_pair,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_metrics import (
    weighted_complex_nrmse,
)
from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_time_settling_certificate import (
    POLARIZATIONS,
    REFERENCE_NAME,
    _all_true,
    _material_case_audit,
    sha256,
)


CERTIFICATE_NAME = "FDTDX_FRESH_COURANT_CERTIFICATE.json"
STATUS_READY = "VALIDATED_FDTDX_FRESH_L500_COURANT_CONVERGENCE"
STATUS_BLOCKED = "BLOCKED_FDTDX_FRESH_L500_COURANT_CONVERGENCE"
LEVELS = ("c0p5", "c0p375", "c0p25", "c0p1875")
COURANT = {"c0p5": 0.5, "c0p375": 0.375, "c0p25": 0.25, "c0p1875": 0.1875}
SUCCESSIVE_PAIRS = (("c0p5", "c0p375"), ("c0p375", "c0p25"), ("c0p25", "c0p1875"))
SELECTED_LEVEL = "c0p25"
CONFIRMATION_LEVEL = "c0p1875"
TOTAL_PERIODS = 24
WINDOW_PERIODS = 4
DT_SCALING_RTOL = 5.0e-12


def _arrays_identical(
    cases: Mapping[str, Mapping[str, Mapping[str, Any] | None]], key: str
) -> bool:
    values: list[Any] = []
    for level in LEVELS:
        for polarization in POLARIZATIONS:
            snapshot = cases[level][polarization]
            if snapshot is None:
                return False
            values.append(snapshot[key])
    if not values:
        return False
    if isinstance(values[0], tuple):
        return all(
            isinstance(value, tuple)
            and len(value) == len(values[0])
            and all(
                np.array_equal(values[0][index], value[index])
                for index in range(len(values[0]))
            )
            for value in values[1:]
        )
    return all(
        np.array_equal(values[0], value) for value in values[1:]
    )


def _nested_arrays_identical(
    cases: Mapping[str, Mapping[str, Mapping[str, Any] | None]], key: str
) -> bool:
    for material in ("au", "tairte4"):
        arrays = []
        for level in LEVELS:
            for polarization in POLARIZATIONS:
                snapshot = cases[level][polarization]
                if snapshot is None:
                    return False
                arrays.append(snapshot[key][material])
        if not all(
            np.array_equal(arrays[0], value)
            for value in arrays[1:]
        ):
            return False
    return True


def expected_courant_case(level: str) -> FreshCaseSpec:
    if level not in LEVELS:
        raise ValueError(f"Courant level must be one of {LEVELS}")
    return FreshCaseSpec(
        mesh=MeshSpec(),
        time=TimeSpec(
            total_periods=TOTAL_PERIODS,
            window_periods=WINDOW_PERIODS,
            courant_factor=COURANT[level],
        ),
    )


def compare_courant_pair(
    coarse_level: str,
    fine_level: str,
    snapshots: Mapping[str, Mapping[str, Mapping[str, Any] | None]],
    payloads: Mapping[str, Mapping[str, Mapping[str, Any]]],
    source_pairs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare both polarizations on two same-grid, different-dt cases."""

    if (coarse_level, fine_level) not in SUCCESSIVE_PAIRS:
        raise ValueError("only declared successive Courant pairs may be compared")
    if any(
        snapshots[level][polarization] is None
        for level in (coarse_level, fine_level)
        for polarization in POLARIZATIONS
    ):
        return {
            "coarse_level": coarse_level,
            "fine_level": fine_level,
            "pass": False,
            "error": "one or more raw snapshots failed artifact validation",
            "checks": {},
        }

    source_change = {
        polarization: relative_difference(
            source_pairs[coarse_level]["comparison"]["unscaled_incident_power_W"][
                polarization
            ],
            source_pairs[fine_level]["comparison"]["unscaled_incident_power_W"][
                polarization
            ],
        )
        for polarization in POLARIZATIONS
    }
    source_change["mean"] = relative_difference(
        source_pairs[coarse_level]["comparison"]["mean_unscaled_incident_power_W"],
        source_pairs[fine_level]["comparison"]["mean_unscaled_incident_power_W"],
    )

    per_polarization: dict[str, Any] = {}
    for polarization in POLARIZATIONS:
        coarse = snapshots[coarse_level][polarization]
        fine = snapshots[fine_level][polarization]
        material_field = {
            material: weighted_complex_nrmse(
                fine["fields_late"][material],
                coarse["fields_late"][material],
                fine["volumes"][material],
            )
            for material in ("au", "tairte4")
        }
        material_component_q: dict[str, Any] = {}
        flat_component_changes: list[float] = []
        for material in ("au", "tairte4"):
            material_component_q[material] = {
                axis: relative_difference(
                    coarse["power_late"]["by_material"][material]["component_W"][axis],
                    fine["power_late"]["by_material"][material]["component_W"][axis],
                )
                for axis in ("x", "y", "z")
            }
            material_component_q[material]["total"] = relative_difference(
                coarse["power_late"]["by_material"][material]["total_W"],
                fine["power_late"]["by_material"][material]["total_W"],
            )
            flat_component_changes.extend(material_component_q[material].values())
        per_polarization[polarization] = {
            "total_Q_relative_change": relative_difference(
                coarse["power_late"]["total_W"], fine["power_late"]["total_W"]
            ),
            "material_component_Q_relative_change": material_component_q,
            "material_component_Q_max_relative_change": max(flat_component_changes),
            "complex_E_fixed_probe_NRMSE": weighted_complex_nrmse(
                fine["probe"], coarse["probe"], fine["probe_weights"]
            ),
            "material_region_complex_E_NRMSE": material_field,
            "material_region_complex_E_max_NRMSE": max(material_field.values()),
            "conservative_Q_volume_L2_NRMSE": combined_weighted_nrmse(
                fine["q_late"], coarse["q_late"], fine["volumes"]
            ),
        }

    closure_values = [
        float(payloads[level][polarization]["evaluation"]["flux"][name])
        for level in (coarse_level, fine_level)
        for polarization in POLARIZATIONS
        for name in (
            "Q_vs_closed_phasor_symmetric_relative",
            "Q_vs_closed_td_symmetric_relative",
        )
    ]
    metrics = {
        "source_power_relative_change": max(source_change.values()),
        "q_closed_flux_relative": max(closure_values),
        "stationarity_complex_E_NRMSE": max(
            float(
                payloads[fine_level][polarization]["evaluation"][
                    "field_stationarity"
                ]["maximum_complex_E_NRMSE"]
            )
            for polarization in POLARIZATIONS
        ),
        "total_Q_relative_change": max(
            item["total_Q_relative_change"] for item in per_polarization.values()
        ),
        "material_component_Q_max_relative_change": max(
            item["material_component_Q_max_relative_change"]
            for item in per_polarization.values()
        ),
        "complex_E_fixed_probe_NRMSE": max(
            item["complex_E_fixed_probe_NRMSE"]
            for item in per_polarization.values()
        ),
        "conservative_Q_volume_L2_NRMSE": max(
            item["conservative_Q_volume_L2_NRMSE"]
            for item in per_polarization.values()
        ),
    }
    evaluated = evaluate_pair(metrics)
    return {
        "coarse_level": coarse_level,
        "coarse_courant_factor": COURANT[coarse_level],
        "fine_level": fine_level,
        "fine_courant_factor": COURANT[fine_level],
        "same_spatial_grid_comparison": True,
        "fixed_probe_method": (
            "exact common physical [-4,+4] um x/y cells at z=0.250 um; "
            "component-specific Yee area weights; no interpolation needed"
        ),
        "Q_method": (
            "same physical control volumes with component-specific stored Yee "
            "dual volumes; no remap needed"
        ),
        "source_power_relative_change": source_change,
        "per_polarization": per_polarization,
        "metrics": metrics,
        "limits": OPTICAL_PAIR_GATES,
        "checks": evaluated["checks"],
        "pass": evaluated["pass"],
    }


def courant_selection_gates(
    case_ready: Mapping[str, Mapping[str, bool]],
    pair_pass: Mapping[tuple[str, str], bool],
) -> dict[str, bool]:
    return {
        "all_four_Courant_levels_internally_ready": all(
            case_ready[level][polarization] is True
            for level in LEVELS
            for polarization in POLARIZATIONS
        ),
        "c0p5_to_c0p375_coarse_comparison_was_evaluated": isinstance(
            pair_pass[("c0p5", "c0p375")], bool
        ),
        "c0p375_to_c0p25_cross_comparison_passes": pair_pass[
            ("c0p375", "c0p25")
        ]
        is True,
        "c0p25_to_c0p1875_cross_comparison_passes": pair_pass[
            ("c0p25", "c0p1875")
        ]
        is True,
    }


def _all_close(values: list[float], rtol: float) -> bool:
    return bool(values) and all(
        math.isclose(value, values[0], rel_tol=rtol, abs_tol=0.0)
        for value in values[1:]
    )


def courant_raw_schema_checks(
    cases: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, bool]:
    schemas = [
        cases[level][polarization]["raw"]["declared_arrays"]
        for level in LEVELS
        for polarization in POLARIZATIONS
    ]
    non_time = [
        {name: shape for name, shape in schema.items() if name != "closed_td"}
        for schema in schemas
    ]
    closed_samples = {
        level: [
            cases[level][polarization]["raw"]["declared_arrays"].get("closed_td")
            for polarization in POLARIZATIONS
        ]
        for level in LEVELS
    }
    shapes_valid = all(
        isinstance(shape, list)
        and len(shape) == 2
        and isinstance(shape[0], int)
        and shape[0] > 0
        and shape[1] == 1
        for values in closed_samples.values()
        for shape in values
    )
    polarization_counts_match = all(
        values[0] == values[1] for values in closed_samples.values()
    )
    scaled_counts = [
        closed_samples[level][0][0] * COURANT[level] for level in LEVELS
    ] if shapes_valid and polarization_counts_match else []
    return {
        "non_time_raw_array_schema_identical": bool(non_time)
        and all(value == non_time[0] for value in non_time[1:]),
        "closed_td_shapes_valid": shapes_valid,
        "closed_td_polarization_sample_counts_match": polarization_counts_match,
        "closed_td_sample_count_scales_inverse_with_Courant": bool(scaled_counts)
        and max(scaled_counts) - min(scaled_counts) <= 1.0,
    }


ALLOWED_CROSS_COMMIT_PATHS = frozenset((
    "photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/fdtdx_fresh_courant_certificate.py",
    "photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/fdtdx_fresh_time_settling_certificate.py",
    "photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/test_fdtdx_fresh_courant_certificate.py",
    "photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/test_fdtdx_fresh_time_settling_certificate.py",
))


def cross_commit_audit(repository: Path, commits: set[Any]) -> dict[str, Any]:
    invalid = sorted(repr(value) for value in commits if not isinstance(value, str))
    ordered = sorted(value for value in commits if isinstance(value, str))
    comparisons = []
    changed_union: set[str] = set()
    for index, left in enumerate(ordered):
        for right in ordered[index + 1:]:
            changed = tuple(
                name for name in _git(repository, "diff", "--name-only", left, right).splitlines()
                if name
            )
            changed_union.update(changed)
            comparisons.append({"left": left, "right": right, "changed_paths": list(changed)})
    ready = (
        bool(ordered)
        and not invalid
        and changed_union.issubset(ALLOWED_CROSS_COMMIT_PATHS)
    )
    return {
        "recorded_commits": ordered,
        "invalid_commit_values": invalid,
        "comparisons": comparisons,
        "changed_paths_union": sorted(changed_union),
        "allowed_paths": sorted(ALLOWED_CROSS_COMMIT_PATHS),
        "only_certificate_and_test_files_changed": ready,
        "ready": ready,
    }


def build_courant_certificate(
    root: Path,
    contract_sha256s: Mapping[str, str],
    source_pair_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    campaign_root = root.expanduser().resolve()
    if not root.expanduser().is_absolute() or not campaign_root.is_dir():
        raise RuntimeError("campaign root must be an existing absolute directory")
    if set(contract_sha256s) != set(LEVELS):
        raise ValueError(f"contract SHA mapping must contain exactly {LEVELS}")
    if set(source_pair_sha256s) != set(LEVELS):
        raise ValueError(f"source-pair SHA mapping must contain exactly {LEVELS}")

    contracts: dict[str, Any] = {}
    source_pairs: dict[str, Any] = {}
    payloads: dict[str, dict[str, Any]] = {}
    cases: dict[str, dict[str, Any]] = {}
    snapshots: dict[str, dict[str, Any]] = {}
    for level in LEVELS:
        spec = expected_courant_case(level)
        contract_path = (
            campaign_root / "contracts" / f"l500_anchor_t24_{level}.json"
        ).resolve()
        loaded_spec, contract_payload, contract_audit = load_case_contract(
            contract_path, contract_sha256s[level]
        )
        if loaded_spec != spec:
            raise RuntimeError(f"{level} is not the exact expected Courant case")
        contracts[level] = {
            "spec": spec,
            "payload": contract_payload,
            "audit": contract_audit,
        }

        pair_path = (
            campaign_root
            / f"source_pair_t24_{level}"
            / "FDTDX_FRESH_SOURCE_ONLY_PAIR.json"
        ).resolve()
        try:
            pair_payload, pair_audit = validate_source_pair(
                pair_path, source_pair_sha256s[level], expected_case=spec
            )
        except Exception as error:
            pair_payload = {}
            pair_audit = {
                "ready": False,
                "error": repr(error),
                "checks": {},
                "failed_checks": ["source_pair_revalidation_exception"],
            }
        source_pairs[level] = {
            "path": str(pair_path),
            "payload": pair_payload,
            "audit": pair_audit,
        }

        payloads[level] = {}
        cases[level] = {}
        snapshots[level] = {}
        for polarization in POLARIZATIONS:
            report = (
                campaign_root
                / f"l500_t24_{level}_{polarization}"
                / "FDTDX_FRESH_EXACT_BINARY_PILOT.json"
            ).resolve()
            payload, audit, snapshot = _material_case_audit(
                report,
                campaign_root,
                TOTAL_PERIODS,
                polarization,
                spec,
                contract_path,
                contract_sha256s[level],
                pair_path,
                source_pair_sha256s[level],
            )
            payloads[level][polarization] = payload
            cases[level][polarization] = audit
            snapshots[level][polarization] = snapshot

    pair_results = {
        f"{coarse}_to_{fine}": compare_courant_pair(
            coarse,
            fine,
            snapshots,
            payloads,
            {level: source_pairs[level]["payload"] for level in LEVELS},
        )
        for coarse, fine in SUCCESSIVE_PAIRS
    }
    case_ready = {
        level: {
            polarization: payloads[level][polarization]["evaluation"]["ready"]
            for polarization in POLARIZATIONS
        }
        for level in LEVELS
    }
    selection = courant_selection_gates(
        case_ready,
        {
            pair: pair_results[f"{pair[0]}_to_{pair[1]}"]["pass"]
            for pair in SUCCESSIVE_PAIRS
        },
    )
    flat_payloads = [
        payloads[level][polarization]
        for level in LEVELS
        for polarization in POLARIZATIONS
    ]
    flat_cases = [
        cases[level][polarization]
        for level in LEVELS
        for polarization in POLARIZATIONS
    ]
    repository_commits = {
        payload["provenance"]["repository_commit"] for payload in flat_payloads
    }
    source_pair_commits = {
        item["payload"].get("provenance", {}).get("certificate_repository_commit")
        for item in source_pairs.values()
    }
    repository = Path(__file__).resolve().parents[3]
    commit_audit = cross_commit_audit(
        repository, repository_commits | source_pair_commits
    )
    dt_per_courant = [
        float(payloads[level][polarization]["time_contract"]["time_step_s"])
        / COURANT[level]
        for level in LEVELS
        for polarization in POLARIZATIONS
    ]
    end_times = [
        float(payloads[level][polarization]["time_contract"]["time_step_s"])
        * int(payloads[level][polarization]["time_contract"]["time_steps_total"])
        for level in LEVELS
        for polarization in POLARIZATIONS
    ]
    maximum_dt = max(
        float(payloads[level][polarization]["time_contract"]["time_step_s"])
        for level in LEVELS
        for polarization in POLARIZATIONS
    )
    schema_checks = courant_raw_schema_checks(cases)
    gates = {
        "all_canonical_case_contracts_revalidated": all(
            contracts[level]["audit"]["ready"] is True for level in LEVELS
        ),
        "all_source_pairs_revalidated": all(
            source_pairs[level]["audit"].get("ready") is True for level in LEVELS
        ),
        "all_material_artifacts_and_recomputed_physics_ready": all(
            case["artifact_ready"] is True for case in flat_cases
        ),
        "all_material_cases_internally_ready": all(
            case_ready[level][polarization] is True
            for level in LEVELS
            for polarization in POLARIZATIONS
        ),
        "only_Courant_factor_changes_across_contracts": all(
            contracts[level]["spec"].mesh == contracts[LEVELS[0]]["spec"].mesh
            and contracts[level]["spec"].pml_alpha_scale
            == contracts[LEVELS[0]]["spec"].pml_alpha_scale
            and contracts[level]["spec"].pml_target_reflection
            == contracts[LEVELS[0]]["spec"].pml_target_reflection
            and contracts[level]["spec"].time.total_periods == TOTAL_PERIODS
            and contracts[level]["spec"].time.window_periods == WINDOW_PERIODS
            and contracts[level]["spec"].time.courant_factor == COURANT[level]
            for level in LEVELS
        ),
        "realized_dt_scales_with_Courant": _all_close(
            dt_per_courant, DT_SCALING_RTOL
        ),
        "realized_end_times_agree_within_one_coarse_step": max(end_times)
        - min(end_times)
        <= maximum_dt,
        "grid_edges_identical_for_exact_same_cell_comparison": _arrays_identical(
            snapshots, "grid_edges"
        ),
        "Yee_dual_volumes_identical": _nested_arrays_identical(snapshots, "volumes"),
        "exact_L500_masks_identical": _arrays_identical(snapshots, "design_mask")
        and _arrays_identical(snapshots, "solver_mask"),
        **schema_checks,
        "placement_identical": len(
            {json.dumps(payload["placement"], sort_keys=True) for payload in flat_payloads}
        )
        == 1,
        "same_polarization_source_contract_identical": all(
            len(
                {
                    json.dumps(payloads[level][polarization]["source_contract"], sort_keys=True)
                    for level in LEVELS
                }
            )
            == 1
            for polarization in POLARIZATIONS
        ),
        "cross_commit_changes_are_certificate_only": commit_audit["ready"] is True,
        "source_and_material_repository_commit_match_per_level": all(
            source_pairs[level]["payload"]
            .get("provenance", {})
            .get("certificate_repository_commit")
            == payloads[level][polarization]["provenance"]["repository_commit"]
            for level in LEVELS
            for polarization in POLARIZATIONS
        ),
        "source_pair_generator_hash_identical": len(
            {
                item["payload"].get("provenance", {}).get(
                    "certificate_generator_sha256"
                )
                for item in source_pairs.values()
            }
        )
        == 1,
        "fdtdx_source_provenance_identical": len(
            {
                json.dumps(payload["provenance"]["fdtdx_source"], sort_keys=True)
                for payload in flat_payloads
            }
        )
        == 1,
        "runtime_lock_identical": len(
            {
                json.dumps(payload["provenance"]["runtime_lock"], sort_keys=True)
                for payload in flat_payloads
            }
        )
        == 1,
        "runner_hash_identical": len(
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
        "required_successive_comparisons_and_selection_pass": _all_true(selection),
        "optimizer_remains_forbidden": all(
            payload.get("optimizer_start_allowed") is False for payload in flat_payloads
        ),
    }
    ready = all(gates.values())
    return {
        "status": STATUS_READY if ready else STATUS_BLOCKED,
        "ready": ready,
        "scope": (
            "Courant convergence of one exact-binary L500 optical reference at "
            "24 periods on one fixed FDTDX spatial grid"
        ),
        "campaign_root": str(campaign_root),
        "contracts": {
            level: {
                "courant_factor": COURANT[level],
                "case_contract_sha256": contracts[level]["payload"][
                    "case_contract_sha256"
                ],
                "file_audit": contracts[level]["audit"],
            }
            for level in LEVELS
        },
        "source_pairs": {
            level: {
                "path": source_pairs[level]["path"],
                "audit": source_pairs[level]["audit"],
                "unscaled_incident_power_W": source_pairs[level]["payload"].get(
                    "comparison", {}
                ).get("unscaled_incident_power_W"),
            }
            for level in LEVELS
        },
        "cross_commit_audit": commit_audit,
        "cases": cases,
        "successive_comparisons": pair_results,
        "selection": {
            "selected_level": SELECTED_LEVEL if ready else None,
            "selected_courant_factor": COURANT[SELECTED_LEVEL] if ready else None,
            "confirmation_level": CONFIRMATION_LEVEL if ready else None,
            "confirmation_courant_factor": (
                COURANT[CONFIRMATION_LEVEL] if ready else None
            ),
            "policy": (
                "retain the evaluated failed c0p5-to-c0p375 coarse comparison; "
                "select c0p25 only when all four levels are internally ready and "
                "both c0p375-to-c0p25 and c0p25-to-c0p1875 pass"
            ),
            "gates": selection,
            "failed_gates": [name for name, passed in selection.items() if not passed],
        },
        "realized_time_diagnostics": {
            "dt_per_courant_s": dt_per_courant,
            "end_times_s": end_times,
            "maximum_time_step_s": maximum_dt,
        },
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "is_time_step_certificate": ready,
        "is_mesh_certificate": False,
        "optimizer_start_allowed": False,
        "next_allowed_step": (
            "run the exact L500 reference on the full-domain-z resolution ladder "
            "at 24 periods and selected Courant; do not optimize"
        ),
    }


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_courant_certificate(
    root: Path,
    contract_sha256s: Mapping[str, str],
    source_pair_sha256s: Mapping[str, str],
    output_directory: Path,
) -> dict[str, Any]:
    output = output_directory.expanduser().resolve()
    if not output_directory.expanduser().is_absolute() or not output.is_dir():
        raise RuntimeError("output directory must be an existing absolute directory")
    if any(output.iterdir()):
        raise RuntimeError("output directory must be empty before certification")
    result = build_courant_certificate(root, contract_sha256s, source_pair_sha256s)
    repository = Path(__file__).resolve().parents[3]
    dirty = _git(repository, "status", "--porcelain", "--untracked-files=all")
    result["provenance"] = {
        "certificate_repository_commit": _git(repository, "rev-parse", "HEAD"),
        "certificate_repository_dirty_porcelain": dirty,
        "certificate_generator_path": str(Path(__file__).resolve()),
        "certificate_generator_sha256": sha256(Path(__file__).resolve()),
    }
    result["gates"]["certificate_repository_clean"] = dirty == ""
    result["ready"] = all(result["gates"].values())
    result["status"] = STATUS_READY if result["ready"] else STATUS_BLOCKED
    result["is_time_step_certificate"] = result["ready"]
    result["failed_gates"] = [
        name for name, passed in result["gates"].items() if not passed
    ]
    if not result["ready"]:
        for name in (
            "selected_level",
            "selected_courant_factor",
            "confirmation_level",
            "confirmation_courant_factor",
        ):
            result["selection"][name] = None
    _atomic_json(output / CERTIFICATE_NAME, result)
    return result


def _level_sha(values: list[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        try:
            level, digest = item.split("=", 1)
        except ValueError as error:
            raise ValueError(f"{label} entries must use LEVEL=SHA256") from error
        if level in result:
            raise ValueError(f"duplicate {label} level {level}")
        normalized = digest.strip().lower()
        if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
            raise ValueError(f"{label} {level} is not a lowercase SHA256")
        result[level] = normalized
    if set(result) != set(LEVELS):
        raise ValueError(f"{label} entries must contain exactly {LEVELS}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--contract-sha256", action="append", default=[], metavar="LEVEL=SHA256")
    parser.add_argument("--source-pair-sha256", action="append", default=[], metavar="LEVEL=SHA256")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = write_courant_certificate(
            args.root,
            _level_sha(args.contract_sha256, "contract"),
            _level_sha(args.source_pair_sha256, "source-pair"),
            args.output_dir,
        )
    except Exception as error:
        failure = {
            "status": "BLOCKED_FDTDX_FRESH_L500_COURANT_EXCEPTION",
            "ready": False,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "is_time_step_certificate": False,
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
        "selection": result["selection"],
        "failed_gates": result["failed_gates"],
        "successive_comparisons": {
            name: {"pass": item["pass"], "metrics": item.get("metrics")}
            for name, item in result["successive_comparisons"].items()
        },
        "report": str(args.output_dir.expanduser().resolve() / CERTIFICATE_NAME),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
