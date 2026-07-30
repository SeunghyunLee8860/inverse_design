#!/usr/bin/env python3
"""Publish the fail-closed paper-IR source-only certification status.

This summarizer is intentionally solver-free.  It combines the paper-beam
audit with source-only startup probes and historical resource references.
It must never convert a license/startup failure into a source certificate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[3]
BLOCKED_STATUS = "BLOCKED_LUMERICAL_LICENSE_UNAVAILABLE_BEFORE_SOURCE_ONLY"
ACCEPTED_PROBE_STATUSES = {
    "BLOCKED_SOURCE_ONLY_EXECUTION",
    "BLOCKED_LUMERICAL_LICENSE_UNAVAILABLE",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def artifact_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-summary", required=True)
    parser.add_argument("--probe-result", action="append", required=True)
    parser.add_argument("--probe-manifest", action="append", required=True)
    parser.add_argument("--historical-resource-audit", required=True)
    parser.add_argument("--historical-runtime-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    audit_path = Path(args.audit_summary).expanduser().resolve()
    probe_results = [
        Path(value).expanduser().resolve() for value in args.probe_result
    ]
    probe_manifests = [
        Path(value).expanduser().resolve() for value in args.probe_manifest
    ]
    resource_path = Path(args.historical_resource_audit).expanduser().resolve()
    runtime_path = Path(args.historical_runtime_summary).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    if len(probe_results) != len(probe_manifests):
        raise ValueError("each probe result must have one probe manifest")

    audit = read_json(audit_path)
    probes = [read_json(path) for path in probe_results]
    resource = read_json(resource_path)
    runtime = read_json(runtime_path)
    if audit.get("status") != "PROPOSED_PAPER_IR_SOURCE_CONTRACT_READY_FOR_GPU_PROBE":
        raise ValueError(f"unexpected audit status: {audit.get('status')}")
    for probe in probes:
        if probe.get("status") not in ACCEPTED_PROBE_STATUSES:
            raise ValueError(f"unexpected probe status: {probe.get('status')}")
        scope = probe.get("scope", {})
        if any(
            bool(scope.get(key))
            for key in (
                "FDTD_source_only",
                "TaIrTe4",
                "substrate",
                "thermal",
                "PTE",
                "weighting_potential",
                "adjoint",
                "gradient",
                "optimization",
                "CPU_FDTD_fallback",
            )
        ):
            raise ValueError("a blocked probe incorrectly reports executed scope")

    selected = audit["selected_source_only_candidate"]
    historical_failure = resource["failure"]
    historical_memory = float(
        historical_failure["memory"]["precise_total_GiB"]
    )
    historical_points = int(historical_failure["total_gridpoints"])
    scale_48_to_60 = (60.0 / 48.0) ** 2
    scaled_reference = {
        "basis": (
            "lateral-area scaling of a historical 48-um high-index material "
            "case; not a source-only runsetup estimate"
        ),
        "grid_points": round(historical_points * scale_48_to_60),
        "GPU_memory_GiB": historical_memory * scale_48_to_60,
        "certification_use": False,
    }
    historical_runtime = runtime["proposed_GPU_runtime_projection"]
    old_mean = float(historical_runtime["observed_mean_s"])
    old_maximum = float(historical_runtime["observed_maximum_s"])
    old_five_case_lower_bound = 5.0 * old_mean

    cases = []
    for index, (path, probe) in enumerate(zip(probe_results, probes), 1):
        cases.append(
            {
                "case": f"contract_only_probe_{index}",
                "path": str(path),
                "status": probe["status"],
                "contract_only": bool(probe.get("contract_only")),
                "FDTD_session_opened": False,
                "runsetup_completed": False,
                "GPU_solve_started": False,
                "CPU_fallback": False,
                "exception_type": probe.get("exception_type", ""),
                "license_signature_detected": (
                    "ANSYSLI exited or could not read server port"
                    in str(probe.get("exception", ""))
                ),
            }
        )

    gates = {
        "requested_vs_realized_width_relative_error_below_0p5pct": None,
        "beam_center_error_below_one_cell": None,
        "square_capture_at_least_99p9pct": None,
        "realized_source_boundary_max_below_1e_minus_3": None,
        "incident_power_closure_below_0p5pct": None,
        "field_time_convergence_below_0p5pct": None,
        "no_NaN_or_Inf": None,
        "GPU_only_no_CPU_fallback": None,
        "auto_shutoff_at_most_1e_minus_5": None,
    }
    payload = {
        "status": BLOCKED_STATUS,
        "generation_commit": git_commit(),
        "preserved_prior_status": (
            "VALIDATED_OFFLINE_MASKED_PLANAR_AND_SOURCE_AUDIT"
        ),
        "paper_beam_audit_status": audit["status"],
        "selected_candidate": selected,
        "source_model_decision": {
            "scalar_candidate_production_approved": False,
            "matched_vectorial_thin_lens_required": True,
            "reason": selected["production_blocker"],
        },
        "legacy_w0_2um": {
            "status": "DIAGNOSTIC_ONLY_INVALID_FOR_PAPER_LIKE_BEAM",
            "allowed_for_thermal_or_PTE": False,
        },
        "probes": cases,
        "actual_solver_readback": {
            "available": False,
            "reason": (
                "the Lumerical FDTD session failed before runsetup because "
                "ANSYSLI did not provide a server port"
            ),
            "grid": None,
            "GPU_memory": None,
            "runtime": None,
            "realized_beam": None,
        },
        "source_only_gates": gates,
        "historical_noncertifying_resource_reference": {
            "source": str(resource_path),
            "case_description": "48-um high-index material case",
            "grid_points": historical_points,
            "GPU_memory_GiB": historical_memory,
            "maximum_progress_percent": historical_failure[
                "phase"
            ]["maximum_logged_progress_percent"],
            "scaled_60um_lateral_reference": scaled_reference,
        },
        "historical_nonpredictive_runtime_reference": {
            "source": str(runtime_path),
            "old_w0_2um_four_ps_mean_s": old_mean,
            "old_w0_2um_four_ps_max_s": old_maximum,
            "five_case_old_grid_arithmetic_lower_bound_s": (
                old_five_case_lower_bound
            ),
            "five_case_old_grid_arithmetic_lower_bound_min": (
                old_five_case_lower_bound / 60.0
            ),
            "usable_as_new_60um_runtime_estimate": False,
        },
        "total_expected_GPU_time": {
            "value": None,
            "status": "UNRESOLVED_UNTIL_LICENSE_AND_RUNSETUP",
            "reason": (
                "the new homogeneous-air grid and memory were not generated; "
                "old 12/48-um material cases are not a defensible runtime "
                "model for the 60-um source-only contract"
            ),
        },
        "next_four_optical_cases": {
            "executed": False,
            "worth_executing_now": False,
            "blockers": [
                "Lumerical license/session startup unavailable",
                "source-only realized-beam gates not evaluated",
                "matched scalar/vectorial source comparison incomplete",
            ],
        },
        "prohibited_scope_confirmed_not_run": [
            "TaIrTe4 planar/edge optical cases",
            "thermal",
            "weighting potential",
            "PTE",
            "adjoint",
            "gradient",
            "optimization",
            "CPU FDTD fallback",
        ],
    }
    summary_path = output / "paper_ir_source_only_certification_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    cases_path = output / "paper_ir_source_only_cases.csv"
    with cases_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)

    external_records = [
        artifact_record(audit_path, "paper_beam_audit"),
        artifact_record(resource_path, "historical_resource_reference"),
        artifact_record(runtime_path, "historical_runtime_reference"),
    ]
    for result_path, manifest_path in zip(probe_results, probe_manifests):
        external_records.append(
            artifact_record(result_path, "blocked_source_only_probe")
        )
        external_records.append(
            artifact_record(manifest_path, "external_probe_manifest")
        )
    manifest = {
        "raw_artifacts_committed_to_git": False,
        "new_raw_FSP_or_NPZ_generated": False,
        "generation_commit": git_commit(),
        "generation_command": " ".join(
            [
                "python3",
                "photothermal_pte/validation/paper_ir_sanity/"
                "summarize_paper_ir_source_only_certification.py",
                "[arguments recorded by this report]",
            ]
        ),
        "artifacts": external_records,
    }
    (output / "RAW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    gate_rows = "\n".join(
        f"| {name} | NOT EVALUATED |" for name in gates
    )
    report = f"""# Paper-IR source-only beam certification

Status: `{BLOCKED_STATUS}`

The prior `VALIDATED_OFFLINE_MASKED_PLANAR_AND_SOURCE_AUDIT` result remains
unchanged.  The paper/SI audit completed, but the new source-only case did not
open a Lumerical FDTD session.  It is therefore neither a failed beam nor a
certified beam: its optical observables were never produced.

## Paper contract

The paper reports a 7–13 µm Block LaserTune QCL, a 40x reflective objective
with NA=0.4, an approximately 9–16 µm diffraction-limited spot, and 11 µm /
285 µW for Figure 3.  It does not publish whether the spot is a radius,
diameter, FWHM, or 1/e^2 width, nor the exact 11-µm waist plane or pupil fill.
The detailed `PAPER_REPORTED`, `PAPER_INFERRED`, and `EXPLICIT_ASSUMPTION`
records are in `paper_ir_beam_contract_summary.json`.

The first source-only candidate is an explicit assumption:

- wavelength: 11 µm
- Gaussian 1/e^2 intensity radius: 12.0 µm
- eta=lambda/(pi*w0): {selected['eta_paraxial']:.6f}
- Rayleigh range: {selected['Rayleigh_range_m']*1e6:.6f} µm
- backward-source distance from waist: {selected['distance_from_waist_property_m']*1e6:.6f} µm
- source span/domain: 50/60 µm
- analytic square capture: {selected['source_aperture_metrics']['square_captured_fraction']:.8%}
- analytic source-boundary maximum/mean: {selected['source_aperture_metrics']['boundary_max_intensity_over_peak']:.8e} / {selected['source_aperture_metrics']['boundary_mean_intensity_over_peak']:.8e}

The scalar model is not production-approved.  A matched NA=0.4 vector
thin-lens comparison is still required.  The old nominal `w0=2 µm` artifacts
remain `DIAGNOSTIC_ONLY_INVALID_FOR_PAPER_LIKE_BEAM` and are forbidden for
thermal, PTE, or Figure-3 reproduction.

## Startup probes

Two contract-only attempts failed before session creation.  Both report:
`ANSYSLI exited or could not read server port`.  Neither attempt completed
`runsetup`, started a GPU solve, or invoked CPU fallback.  No TaIrTe4,
substrate, thermal, PTE, weighting-potential, adjoint, gradient, or
optimization calculation ran.

## Source-only gates

| Gate | Result |
|---|---|
{gate_rows}

These are `NOT EVALUATED`, not failures and not passes.

## Grid, memory, and runtime

Actual contract-only grid/memory readback is unavailable because the session
did not open.  For context only, a historical 48-µm high-index material case
had {historical_points:,} grid points and a {historical_memory:.3f}-GiB
precise GPU-memory estimate.  Blind lateral-area scaling to 60 µm would give
about {scaled_reference['grid_points']:,} points and
{scaled_reference['GPU_memory_GiB']:.3f} GiB, but this is **not** a
homogeneous-air source-only estimate and cannot certify feasibility.

The old 12-µm nominal-w0=2-µm 4-ps cases averaged {old_mean:.3f} s.  Five
times that old-grid mean is {old_five_case_lower_bound/60.0:.2f} min, but it
is only historical arithmetic, not a prediction for the 60-µm contract.
Total expected GPU time remains
`UNRESOLVED_UNTIL_LICENSE_AND_RUNSETUP`.

## Decision

The four planar/finite-edge optical cases are not worth executing now.  First
restore license/session startup, obtain the actual source-only grid/memory
readback, execute one GPU-only homogeneous-air case, and pass the realized
beam gates.  Then perform the matched scalar/vectorial comparison before any
material case.  No production-Q promotion is made by this checkpoint.
"""
    (output / "PAPER_IR_SOURCE_ONLY_CERTIFICATION_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps({"status": BLOCKED_STATUS, "cases": cases}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
