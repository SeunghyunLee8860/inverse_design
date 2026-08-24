from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import traceback
from pathlib import Path
from typing import Any


CERTIFICATE_NAME = "FDTDX_FRESH_SOURCE_ONLY_PAIR.json"
CASE_STATUS = "VALIDATED_FDTDX_FRESH_SOURCE_ONLY_CASE"
PAIR_STATUS = "VALIDATED_FDTDX_FRESH_SOURCE_ONLY_PAIR"
BLOCKED_STATUS = "BLOCKED_FDTDX_FRESH_SOURCE_ONLY_PAIR"
EXPECTED_SCOPE = "all-air source-only on validated fresh anchor"
POWER_MISMATCH_RELATIVE_LIMIT = 5.0e-3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"source-only report must contain one JSON object: {resolved}")
    return payload


def _raw_audit(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload["raw"]
    raw_path = Path(raw["path"]).expanduser()
    is_absolute = raw_path.is_absolute()
    resolved = raw_path.resolve()
    exists = resolved.is_file()
    actual_sha256 = sha256(resolved) if exists else None
    return {
        "path": str(resolved),
        "path_is_absolute": is_absolute,
        "exists": exists,
        "recorded_sha256": raw["sha256"],
        "actual_sha256": actual_sha256,
        "sha256_matches": exists and actual_sha256 == raw["sha256"],
        "arrays": raw["arrays"],
    }


def _case_audit(
    report_path: Path,
    expected_polarization: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = report_path.expanduser().resolve()
    payload = _load_json(resolved)
    evaluation = payload["evaluation"]
    evaluation_gates = evaluation["gates"]
    raw = _raw_audit(payload)
    source = payload["provenance"]["fdtdx_source"]
    audit = {
        "report_path": str(resolved),
        "report_sha256": sha256(resolved),
        "expected_polarization": expected_polarization,
        "recorded_polarization": payload.get("polarization"),
        "status": payload.get("status"),
        "ready": payload.get("ready"),
        "scope": payload.get("scope"),
        "evaluation_ready": evaluation.get("ready"),
        "evaluation_gates": evaluation_gates,
        "all_evaluation_gates_true": bool(evaluation_gates)
        and all(value is True for value in evaluation_gates.values()),
        "all_air_material_readback_ready": payload["all_air_material_readback"].get(
            "ready"
        ),
        "per_case_scale_not_authorized_until_pair_comparison": payload.get(
            "per_case_scale_not_authorized_until_pair_comparison"
        ),
        "incident_power_W": evaluation["flux"]["incident_plane_signed_W"],
        "reporting_incident_power_W": payload["reporting_incident_power_W"],
        "repository_commit": payload["provenance"]["repository_commit"],
        "repository_dirty_porcelain": payload["provenance"][
            "repository_dirty_porcelain"
        ],
        "fdtdx_commit": source["commit"],
        "fdtdx_dirty_porcelain": source["dirty_porcelain"],
        "raw": raw,
    }
    return payload, audit


def build_pair_certificate(
    ea_report: Path,
    eb_report: Path,
) -> dict[str, Any]:
    ea, ea_audit = _case_audit(ea_report, "Ea")
    eb, eb_audit = _case_audit(eb_report, "Eb")
    cases = {"Ea": ea_audit, "Eb": eb_audit}

    powers = {
        polarization: float(case["incident_power_W"])
        for polarization, case in cases.items()
    }
    powers_finite_positive = all(
        math.isfinite(power) and power > 0.0 for power in powers.values()
    )
    mean_power = sum(powers.values()) / 2.0 if powers_finite_positive else math.nan
    relative_mismatch = (
        abs(powers["Ea"] - powers["Eb"]) / mean_power
        if powers_finite_positive
        else math.inf
    )
    target_ea = float(ea_audit["reporting_incident_power_W"])
    target_eb = float(eb_audit["reporting_incident_power_W"])
    targets_finite_positive = all(
        math.isfinite(target) and target > 0.0 for target in (target_ea, target_eb)
    )
    common_target = target_ea if targets_finite_positive else math.nan
    common_power_scale = (
        common_target / mean_power
        if powers_finite_positive and targets_finite_positive
        else math.nan
    )
    common_field_scale = (
        math.sqrt(common_power_scale)
        if math.isfinite(common_power_scale) and common_power_scale > 0.0
        else math.nan
    )

    gates = {
        "input_reports_are_distinct": ea_audit["report_path"]
        != eb_audit["report_path"],
        "expected_polarizations": ea_audit["recorded_polarization"] == "Ea"
        and eb_audit["recorded_polarization"] == "Eb",
        "case_status_and_ready": all(
            case["status"] == CASE_STATUS
            and case["ready"] is True
            and case["evaluation_ready"] is True
            for case in cases.values()
        ),
        "case_scope_exact": all(
            case["scope"] == EXPECTED_SCOPE for case in cases.values()
        ),
        "case_evaluation_gates_all_true": all(
            case["all_evaluation_gates_true"] for case in cases.values()
        ),
        "case_all_air_material_readback_ready": all(
            case["all_air_material_readback_ready"] is True
            for case in cases.values()
        ),
        "per_case_scaling_remained_unauthorized": all(
            case["per_case_scale_not_authorized_until_pair_comparison"] is True
            for case in cases.values()
        ),
        "raw_paths_absolute_and_distinct": all(
            case["raw"]["path_is_absolute"] for case in cases.values()
        )
        and ea_audit["raw"]["path"] != eb_audit["raw"]["path"],
        "raw_files_exist": all(case["raw"]["exists"] for case in cases.values()),
        "raw_sha256_matches": all(
            case["raw"]["sha256_matches"] for case in cases.values()
        ),
        "mesh_contract_identical": ea["mesh"] == eb["mesh"],
        "time_contract_identical": ea["time_contract"] == eb["time_contract"],
        "all_air_material_readback_identical": ea["all_air_material_readback"]
        == eb["all_air_material_readback"],
        "raw_array_schema_identical": ea_audit["raw"]["arrays"]
        == eb_audit["raw"]["arrays"],
        "fdtdx_source_provenance_identical": ea["provenance"]["fdtdx_source"]
        == eb["provenance"]["fdtdx_source"],
        "runtime_lock_identical": ea["provenance"]["runtime_lock"]
        == eb["provenance"]["runtime_lock"],
        "source_repository_commit_identical": ea_audit["repository_commit"]
        == eb_audit["repository_commit"],
        "source_repositories_clean": all(
            case["repository_dirty_porcelain"] == ""
            and case["fdtdx_dirty_porcelain"] == ""
            for case in cases.values()
        ),
        "reporting_power_target_identical_finite_positive": targets_finite_positive
        and target_ea == target_eb,
        "source_powers_finite_positive": powers_finite_positive,
        "source_power_relative_mismatch": relative_mismatch
        <= POWER_MISMATCH_RELATIVE_LIMIT,
    }
    ready = all(gates.values())
    scaled_powers = {
        polarization: power * common_power_scale
        for polarization, power in powers.items()
    }
    return {
        "status": PAIR_STATUS if ready else BLOCKED_STATUS,
        "ready": ready,
        "scope": "dual-polarization all-air source normalization pair",
        "normalization_policy": {
            "per_polarization_power_matching_forbidden": True,
            "common_reference": "arithmetic mean of the unscaled Ea and Eb incident powers",
            "power_scaling_rule": "multiply power-like observables by common_power_scale",
            "field_scaling_rule": "multiply complex fields by common_field_amplitude_scale",
        },
        "cases": cases,
        "comparison": {
            "unscaled_incident_power_W": powers,
            "mean_unscaled_incident_power_W": mean_power,
            "relative_power_mismatch": relative_mismatch,
            "relative_power_mismatch_limit": POWER_MISMATCH_RELATIVE_LIMIT,
        },
        "common_normalization": {
            "reporting_target_incident_power_W": common_target,
            "common_power_scale": common_power_scale,
            "common_field_amplitude_scale": common_field_scale,
            "scaled_incident_power_W": scaled_powers,
            "scaled_power_relative_error_from_target": {
                polarization: abs(scaled - common_target) / common_target
                for polarization, scaled in scaled_powers.items()
            },
        },
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "source_case_contracts": {
            "mesh": ea["mesh"],
            "time_contract": ea["time_contract"],
            "all_air_material_readback": ea["all_air_material_readback"],
            "fdtdx_source": ea["provenance"]["fdtdx_source"],
            "runtime_lock": ea["provenance"]["runtime_lock"],
        },
    }


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_pair_certificate(
    ea_report: Path,
    eb_report: Path,
    output_directory: Path,
) -> dict[str, Any]:
    output = output_directory.expanduser().resolve()
    if not output.is_absolute() or not output.is_dir():
        raise RuntimeError("output directory must be an existing absolute directory")
    if any(output.iterdir()):
        raise RuntimeError("output directory must be empty before pair certification")

    result = build_pair_certificate(ea_report, eb_report)
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
    result["status"] = PAIR_STATUS if result["ready"] else BLOCKED_STATUS
    result["failed_gates"] = [
        name for name, passed in result["gates"].items() if not passed
    ]
    _atomic_json(output / CERTIFICATE_NAME, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ea", type=Path, required=True)
    parser.add_argument("--eb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = write_pair_certificate(args.ea, args.eb, args.output_dir)
    except Exception as error:
        failure = {
            "status": "BLOCKED_FDTDX_FRESH_SOURCE_ONLY_PAIR_EXCEPTION",
            "ready": False,
            "error": repr(error),
            "traceback": traceback.format_exc(),
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
        "comparison": result["comparison"],
        "common_normalization": result["common_normalization"],
        "report": str(args.output_dir.expanduser().resolve() / CERTIFICATE_NAME),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
